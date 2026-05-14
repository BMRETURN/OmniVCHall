import os
import re
import sys
from pathlib import Path
# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from vcd_new.utils import (
    read_video,
    load_qa_data,
    load_embeddings,
    save_video_to_temp,
    PatchProcessor,
    answer_question_original,
    answer_question_negative,
    answer_question_positive,
    transform_pixel_to_patch
)
from vcd_new.train.optimizer import VCDPolicy, VCDTrainer
from vcd_new.models.selector import QFormerToolRouter
from vcd_new.models.gate import QueryVisualFusionGater
from vcd_new.tools.negative_tools import (
    ReverseVideo, SampleVideo, ShuffleVideo, BlurVideo, NoiseVideo,
    HorizontalMirrorVideo, VerticalMirrorVideo, GrayscaleVideo
)
from vcd_new.tools.positive_tools import MotionSaliencyExtractor, DINOv3SaliencyExtractor

# ==============================================================================
# 新增：从缓存加载 Patch 权重的辅助函数
# ==============================================================================
def get_cached_saliency(qa_id, cache_dir, device):
    """
    尝试从缓存加载 Patch 级的 m_raw 和 v_raw
    """
    if cache_dir is None: return None
    path = os.path.join(cache_dir, f"{qa_id}.pt")
    if not os.path.exists(path): return None
    
    try:
        data = torch.load(path)
        # 加载并转到 GPU/FP32
        return {
            "w_m": data["w_m"].to(device, dtype=torch.float32),
            "w_v": data["w_v"].to(device, dtype=torch.float32)
        }
    except Exception:
        return None

# ==============================================================================
# 修改后的评估函数
# ==============================================================================
def evaluate_dataset(dataset, model, processor, policy, tools_embeddings, tools_dict, 
                     patch_processor, motion_sal_extractor, visual_sal_extractor, 
                     video_dir, primary_device, output_dir, cache_dir=None, stage_name="Validation", epoch=None):
    
    print(f"\n>>> Running {stage_name}..." + (f" (Epoch {epoch})" if epoch is not None else ""))
    os.makedirs(output_dir, exist_ok=True)
    
    policy.eval() 
    
    metrics = {k: {"correct": 0, "total": 0} for k in ["s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"]}
    detailed_logs = []
    
    pbar = tqdm(dataset, desc=stage_name)
    for item in pbar:
        video_id = item["video_id"]
        # 解析 QA
        if "s_ynqa_id" in item: q_type="s_ynqa"; q=item["yn_question"]; gt=item["yn_answer"]
        elif "m_ynqa_id" in item: q_type="m_ynqa"; q=item["yn_question"]; gt=item["yn_answer"]
        elif "s_mcqa_id" in item: q_type="s_mcqa"; q=item["mc_question"]; gt=item["mc_answer"]
        elif "m_mcqa_id" in item: q_type="m_mcqa"; q=item["mc_question"]; gt=item["mc_answer"]
        else: continue
        
        opts = item.get("mc_option", None)
        qa_id = item.get("s_ynqa_id") or item.get("m_ynqa_id") or item.get("s_mcqa_id") or item.get("m_mcqa_id")

        # 找视频
        v_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            p = os.path.join(video_dir, video_id + ext)
            if os.path.exists(p): v_path = p; break
        if not v_path: continue

        try:
            # 1. 原始推理
            with torch.no_grad():
                orig_logits, last_hidden_states, _ = answer_question_original(model, processor, v_path, q, opts)
                        
            state = last_hidden_states.detach().to(primary_device, dtype=torch.float32)

            # 2. 策略决策
            sel_tools, beta_tensor, _, _ = policy.get_action_and_log_prob(state, tools_embeddings, std_dev=0.0)
            beta_val = beta_tensor.item()

            # 3. 负样本流
            frames = read_video(v_path)
            neg_frames = frames
            for t_name in sel_tools: neg_frames = tools_dict[t_name].process(neg_frames)
            neg_path = save_video_to_temp(neg_frames, v_path)
            with torch.no_grad():
                neg_logits = answer_question_negative(model, processor, neg_path, q, opts)
            os.remove(neg_path)

            # 4. 显着性 (修改部分：优先读缓存)
            cache_data = get_cached_saliency(qa_id, cache_dir, primary_device)
            
            if cache_data:
                # 缓存命中：直接计算
                w_m, w_v = cache_data["w_m"], cache_data["w_v"]
                # 线性融合 -> Sigmoid
                combined_raw = beta_val * w_m + (1.0 - beta_val) * w_v
                weights = torch.sigmoid(combined_raw)
            else:
                # 缓存未命中：在线计算 (Fallback)
                meta = patch_processor.get_video_metadata(v_path)
                indices = patch_processor.get_sampling_indices(meta['total_frames'], meta['fps'])
                with torch.no_grad():
                    m_sals = torch.tensor(motion_sal_extractor.extract_motion_saliency(frames, indices=indices.tolist()))
                    v_sals = torch.tensor(visual_sal_extractor.extract_dino_video_pixel_last(frames, indices=indices.tolist()))
                    m_input = m_sals.unsqueeze(0).unsqueeze(0).to(primary_device)
                    v_input = v_sals.unsqueeze(0).unsqueeze(0).to(primary_device)
                
                gt_h, gt_h, gw, h_bar, w_bar = patch_processor.get_smart_resize_grid(len(indices), meta['height'], meta['width'])
                m_patch_raw = transform_pixel_to_patch(indices, h_bar, w_bar, m_input, patch_processor)
                v_patch_raw = transform_pixel_to_patch(indices, h_bar, w_bar, v_input, patch_processor)
                
                combined_patch_raw = beta_val * m_patch_raw + (1.0 - beta_val) * v_patch_raw
                weights = torch.sigmoid(combined_patch_raw)

            # 5. 正样本 VCD
            with torch.no_grad():
                pred, _ = answer_question_positive(model, processor, v_path, weights, q, orig_logits, neg_logits, opts)

            # 6. 统计
            pred_clean = re.search(r'\b(a|b|c|d|yes|no)\b', str(pred).lower())
            pred_token = pred_clean.group(1) if pred_clean else str(pred).lower().strip()
            gt_clean = re.search(r'\b(a|b|c|d|yes|no)\b', str(gt).lower())
            gt_token = gt_clean.group(1) if gt_clean else str(gt).lower().strip()

            is_correct = (pred_token == gt_token)
            if is_correct > 0: metrics[q_type]["correct"] += 1
            metrics[q_type]["total"] += 1
            
            detailed_logs.append({
                "qa_id": qa_id, "gt": str(gt), "pred": str(pred),
                "is_correct": (is_correct>0), "beta": round(beta_val, 4)
            })
            
        except Exception as e:
            print(f"Eval Error {video_id}: {e}")
            continue

    # 输出结果
    total_acc_sum = 0
    valid_types = 0
    print(f"--- {stage_name} Summary ---")
    summary_dict = {}
    for q_t, res in metrics.items():
        if res["total"] > 0:
            acc = res["correct"] / res["total"]
            print(f"{q_t}: {acc:.2%} ({res['correct']}/{res['total']})")
            total_acc_sum += acc
            valid_types += 1
            summary_dict[q_t] = acc
        else:
            print(f"{q_t}: N/A")
            summary_dict[q_t] = 0.0
    
    avg_acc = total_acc_sum / valid_types if valid_types > 0 else 0.0
    print(f"Average Accuracy: {avg_acc:.2%}")
    summary_dict["average_acc"] = avg_acc

    # 保存文件
    prefix = f"{stage_name.lower()}"
    if epoch: prefix += f"_epoch_{epoch}"
    with open(os.path.join(output_dir, f"{prefix}_metrics.json"), "w") as f: json.dump(summary_dict, f, indent=4)
    with open(os.path.join(output_dir, f"{prefix}_logs.json"), "w") as f: json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    return avg_acc

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(2025)
    
    # 配置
    model_name = "Qwen3-VL-8B-Instruct"
    model_dir = "../../checkpoints/Qwen3-VL-8B-Instruct"
    # 数据集路径
    train_qa_dir = "../../dataset/MyBench/train"
    val_qa_dir = "../../dataset/MyBench/val"
    test_qa_dir = "../../dataset/MyBench/test"
    video_dir = "../../dataset/MyBench/all_video"
    tools_dir = "../tools/tools_embeddings_qwen3vl.pkl"
    
    # 缓存路径 (预处理脚本生成的)
    cache_root = "./saliency_cache"
    test_cache = os.path.join(cache_root, "test")
    
    # 输出
    test_out = "test_output"
    
    batch_size = 32
    epochs = 5
    os.makedirs(test_out, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Loading Tools & Embeddings...")
    tools_embeddings = load_embeddings(tools_dir, device)

    print("Loading Data...")
    train_data = load_qa_data(train_qa_dir, shuffle=True)
    val_data = load_qa_data(val_qa_dir, shuffle=False)
    test_data = load_qa_data(test_qa_dir, shuffle=False)

    # val_output_path = os.path.join("val_output", "val_data.json")
    # with open(val_output_path, 'w', encoding='utf-8') as f:
    #     json.dump(val_data, f, ensure_ascii=False, indent=2)
    # print(f"Validation data saved to: {val_output_path}")
    
    # # 保存测试集
    # test_output_path = os.path.join("test_output", "test_data.json")
    # with open(test_output_path, 'w', encoding='utf-8') as f:
    #     json.dump(test_data, f, ensure_ascii=False, indent=2)
    # print(f"Test data saved to: {test_output_path}")
        
    print(f"Data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    print(f"Loading {model_name}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype="float16", device_map=device)
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    processor = AutoProcessor.from_pretrained(model_dir)

    # 工具初始化
    tools_dict = {
        'ReverseVideo': ReverseVideo(), 'SampleVideo': SampleVideo(), 'ShuffleVideo': ShuffleVideo(),
        'BlurVideo': BlurVideo(), 'NoiseVideo': NoiseVideo(), 'HorizontalMirrorVideo': HorizontalMirrorVideo(),
        'VerticalMirrorVideo': VerticalMirrorVideo(), 'GrayscaleVideo': GrayscaleVideo()
    }
    tool_names = list(tools_dict.keys())

    motion_sal_extractor = MotionSaliencyExtractor()
    visual_sal_extractor = DINOv3SaliencyExtractor(device=device)
    patch_processor = PatchProcessor()

    # Policy 初始化
    selector = QFormerToolRouter(num_tools=8, d_in=4096, d_model=1024, device=device)
    gater = QueryVisualFusionGater(embed_dim=4096).to(device)
    policy = VCDPolicy(selector, gater, tool_names).to(device)

    print("\n=== Final Testing ===")
    if os.path.exists("vcd_policy_best.pth"):
        policy.load_state_dict(torch.load("vcd_policy_best.pth"))
        print("Loaded Best Model.")
    
    evaluate_dataset(
        test_data, model, processor, policy, tools_embeddings, tools_dict,
        patch_processor, motion_sal_extractor, visual_sal_extractor,
        video_dir, device, test_out, 
        cache_dir=test_cache, # 传入测试集缓存
        stage_name="Testing"
    )
