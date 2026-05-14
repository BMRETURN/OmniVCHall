import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 路径设置 (根据你的环境调整)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vcd_new.utils import load_qa_data, read_video, PatchProcessor
from vcd_new.tools.positive_tools import MotionSaliencyExtractor, DINOv3SaliencyExtractor

# ==============================================================================
# 核心转换函数 (从 Pixel 到 Patch)
# ==============================================================================
def transform_pixel_to_patch(saliency_map, indices, h_resized, w_resized, proc):
    """
    将像素级显著性图变换为 Patch 级分数的原始值（不包含 Sigmoid）。
    包含：Interpolate -> Pad -> Patch Pooling -> Token Merging -> Flatten
    """
    # 确保输入是 Tensor
    if not isinstance(saliency_map, torch.Tensor):
        raise ValueError(f"Expected Tensor, got {type(saliency_map)}")

    # 维度处理: 确保是 [1, 1, T, H, W]
    # extract 返回通常是 [T, H, W] 或 [1, T, H, W]
    if saliency_map.dim() == 3: # [T, H, W]
        saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)
    elif saliency_map.dim() == 4: # [C, T, H, W] or [B, T, H, W]
        saliency_map = saliency_map.unsqueeze(0)
    
    # 1. 插值/Resize (线性)
    # 将图像调整到适应 Patch 划分的大小
    # 注意：输入必须是 5D [B, C, T, H, W]
    saliency_map = F.interpolate(
        saliency_map, 
        size=(len(indices), h_resized, w_resized), 
        mode='trilinear',
        align_corners=False
    )
    
    # 2. 时间维度 Padding
    t_orig = saliency_map.shape[2]
    pad_t = -t_orig % proc.temporal_patch_size
    if pad_t != 0:
        saliency_map = torch.cat([
            saliency_map, 
            saliency_map[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
        ], dim=2)
    
    # 3. Patch Pooling (Spatial-Temporal)
    patch_sal = F.avg_pool3d(
        saliency_map, 
        kernel_size=(proc.temporal_patch_size, proc.patch_size, proc.patch_size),
        stride=(proc.temporal_patch_size, proc.patch_size, proc.patch_size)
    )

    # 4. Token Merging (Qwen-VL 特有的 2x2 merge)
    token_sal = F.avg_pool3d(
        patch_sal,
        kernel_size=(1, proc.merge_size, proc.merge_size),
        stride=(1, proc.merge_size, proc.merge_size)
    )

    # 返回 Flatten 后的一维向量
    return token_sal.view(-1)

# ==============================================================================
# 数据集处理函数
# ==============================================================================
def process_split(qa_dir, video_dir, output_cache_dir, device):
    """处理单个数据集切片 (Train/Val/Test) 并保存 Patch 级权重"""
    os.makedirs(output_cache_dir, exist_ok=True)
    
    print(f"Loading data from {qa_dir}...")
    data = load_qa_data(qa_dir)
    # 过滤无效数据 (必须有 answer)
    data = [d for d in data if "answer" in d or "yn_answer" in d or "mc_answer" in d]
    
    print(f"Initializing Extractors on {device}...")
    # Motion Extractor (CPU/OpenCV)
    motion_extractor = MotionSaliencyExtractor()
    # Visual Extractor (GPU)
    visual_extractor = DINOv3SaliencyExtractor(device=device)
    # Patch Processor (用于计算 Grid)
    patch_processor = PatchProcessor()

    print(f"Start processing {len(data)} items -> {output_cache_dir}")
    
    skipped = 0
    errors = 0
    processed = 0

    split_name = os.path.basename(os.path.normpath(qa_dir))

    for item in tqdm(data):
        # 获取唯一 ID 作为文件名
        q_type = None
        qa_id = None
        if "s_ynqa_id" in item:
            q_type, qa_id = "s_ynqa", item["s_ynqa_id"]
        elif "m_ynqa_id" in item:
            q_type, qa_id = "m_ynqa", item["m_ynqa_id"]
        elif "s_mcqa_id" in item:
            q_type, qa_id = "s_mcqa", item["s_mcqa_id"]
        elif "m_mcqa_id" in item:
            q_type, qa_id = "m_mcqa", item["m_mcqa_id"]
        if q_type is None or qa_id is None:
            errors += 1
            continue
        video_id = item["video_id"]
        cache_key = f"{split_name}_{q_type}_{qa_id}_{video_id}"
        save_path = os.path.join(output_cache_dir, f"{cache_key}.pt")
        
        # 断点续传
        if os.path.exists(save_path):
            skipped += 1
            continue

        # 查找视频
        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            p = os.path.join(video_dir, video_id + ext)
            if os.path.exists(p): video_path = p; break
        
        if not video_path:
            errors += 1
            continue

        try:
            # 1. 读取视频
            frames = read_video(video_path)
            if frames is None or len(frames) == 0:
                errors += 1
                continue
            
            # 2. 获取 Metadata 和 Indices
            meta = patch_processor.get_video_metadata(video_path)
            indices = patch_processor.get_sampling_indices(meta['total_frames'], meta['fps'])
            
            # 计算 Resize 目标尺寸 (h_bar, w_bar)
            grid_t, grid_h, grid_w, h_bar, w_bar = patch_processor.get_smart_resize_grid(
                len(indices), meta['height'], meta['width']
            )

            # 3. 提取并转换 (No Grad)
            with torch.no_grad():
                # --- Motion Saliency ---
                # extract 返回 numpy array [T, H, W]
                m_pixel = motion_extractor.extract_motion_saliency(frames, indices=indices.tolist())
                
                # 异常检测
                if isinstance(m_pixel, int) or m_pixel is None:
                    errors += 1; continue
                
                # 转 Tensor -> GPU -> Transform
                m_tensor = torch.from_numpy(np.array(m_pixel)).to(device, dtype=torch.float32)
                w_m = transform_pixel_to_patch(m_tensor, indices, h_bar, w_bar, patch_processor)
                
                # --- Visual Saliency ---
                # extract 返回 tensor [T, H, W]
                v_pixel = visual_extractor.extract_dino_video_pixel_last(frames, indices=indices.tolist())
                
                # 异常检测
                if isinstance(v_pixel, int) or v_pixel is None:
                    errors += 1; continue
                
                # 确保 Tensor 格式
                if not isinstance(v_pixel, torch.Tensor):
                     v_tensor = torch.from_numpy(np.array(v_pixel)).to(device, dtype=torch.float32)
                else:
                     v_tensor = v_pixel.to(device, dtype=torch.float32)
                
                w_v = transform_pixel_to_patch(v_tensor, indices, h_bar, w_bar, patch_processor)

            # 4. 保存结果 (转回 CPU, FP16 以进一步节省空间)
            # 这里保存的 w_m 和 w_v 已经是 Patch 级别的了 (形状为 [N_tokens])
            torch.save({
                "w_m": w_m.cpu().half(),
                "w_v": w_v.cpu().half(),
                "video_id": video_id
            }, save_path)
            
            processed += 1
            
        except Exception as e:
            print(f"Error {qa_id}: {e}")
            errors += 1
            continue
            
    print(f"Done. Processed: {processed}, Skipped: {skipped}, Missing/Error: {errors}")

# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == "__main__":
    # 配置
    base_qa_dir = "../../dataset/MyBench"
    video_dir = "../../dataset/MyBench/all_video"
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    
    # 缓存根目录
    cache_root = "./saliency_cache"
    
    # 1. Process Train
    process_split(
        os.path.join(base_qa_dir, "train"),
        video_dir,
        os.path.join(cache_root, "train"),
        device
    )
    
    # 2. Process Val
    process_split(
        os.path.join(base_qa_dir, "val"),
        video_dir,
        os.path.join(cache_root, "val"),
        device
    )
    
    # 3. Process Test
    process_split(
        os.path.join(base_qa_dir, "test"),
        video_dir,
        os.path.join(cache_root, "test"),
        device
    )
    
    print("All preprocessing done!")
