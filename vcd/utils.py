import os
import json
import random
import glob
import pickle
import cv2
import numpy as np
from decord import VideoReader, cpu
from typing import List, Tuple, Optional
import tempfile
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList


def read_video(video_path: str, frame_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """
    读取视频，返回帧列表（RGB 格式，NumPy 数组）
    
    Args:
        video_path: 视频文件路径
        frame_range: 起始帧和截止帧的元组 (start_frame, end_frame)，默认为None表示读取全部帧
        
    Returns:
        帧列表，每个元素是(H, W, 3)形状的NumPy数组
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    
    if frame_range is not None:
        start_frame, end_frame = frame_range
        # 确保帧范围在有效范围内
        start_frame = max(0, start_frame)
        end_frame = min(len(vr) - 1, end_frame)
        # 读取指定范围的帧
        frames = [vr[i].asnumpy() for i in range(start_frame, end_frame + 1)]  # RGB, shape (H, W, 3)
    else:
        # 读取全部帧
        frames = [vr[i].asnumpy() for i in range(len(vr))]  # RGB, shape (H, W, 3)
    return frames


def load_qa_data(qa_source_dir, shuffle=True):
    # 获取所有包含'qa'的JSON文件
    pattern = os.path.join(qa_source_dir, "*qa*.json")
    json_files = glob.glob(pattern)
    random.seed(2025)
    
    # 读取所有JSON文件
    all_data = []
    file_info = {}
    
    for file_path in json_files:
        print(f"正在读取文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 记录原始文件信息
            original_count = len(data)
            file_info[os.path.basename(file_path)] = original_count        

            all_data.extend(data)
            print(f"从 {os.path.basename(file_path)} 读取了 {original_count} 个项目")
    
    if shuffle:
        random.shuffle(all_data)
    
    return all_data


def load_embeddings(embeddings_path, device):
    """
    从本地加载embeddings
    """
    with open(embeddings_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    for key in embeddings_dict:
        embeddings_dict[key] = embeddings_dict[key].to(device)
    return embeddings_dict


def save_video_to_temp(frames, original_video_path):
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()  # 先关闭文件，然后用cv2写入
    
    if len(frames) > 0:
        # 获取帧的尺寸
        height, width, _ = frames[0].shape
        
        # 获取原视频的帧率
        vr = cv2.VideoCapture(original_video_path)
        fps = vr.get(cv2.CAP_PROP_FPS)
        vr.release()
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # 写入处理后的帧
        for frame in frames:
            # 转换RGB到BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
    
    return temp_path


class PatchProcessor:
    def __init__(self):
        self.patch_size = 16
        self.temporal_patch_size = 2
        self.merge_size = 2  
        self.min_pixels = 4096
        self.max_pixels = 25165824
        self.min_frames = 4
        self.max_frames = 768
        self.fps = 2.0  

    def get_video_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        metadata = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return metadata

    def get_sampling_indices(self, total_frames, video_fps):
        num_frames = int(total_frames / video_fps * self.fps)
        num_frames = max(self.min_frames, min(num_frames, self.max_frames, total_frames))
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
        return indices

    def get_smart_resize_grid(self, num_frames, height, width):
        # 复现 Qwen3-VL 的 smart_resize 逻辑
        factor = self.patch_size * self.merge_size  # 32
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        t_bar = math.ceil(num_frames / self.temporal_patch_size) * self.temporal_patch_size
        
        # 像素数检查与缩放
        total_pixels = t_bar * h_bar * w_bar
        if total_pixels > self.max_pixels:
            scale = math.sqrt(self.max_pixels / total_pixels)
            h_bar = round((h_bar * scale) / factor) * factor
            w_bar = round((w_bar * scale) / factor) * factor
        elif total_pixels < self.min_pixels:
            scale = math.sqrt(self.min_pixels / total_pixels)
            h_bar = round((h_bar * scale) / factor) * factor
            w_bar = round((w_bar * scale) / factor) * factor
        
        # 计算 Grid: (T, H, W) 这里的 H,W 是 patch 级别的
        grid_t = t_bar // self.temporal_patch_size
        grid_h = h_bar // self.patch_size
        grid_w = w_bar // self.patch_size
        return grid_t, grid_h, grid_w, h_bar, w_bar
    

def compute_patch_saliency_weights(indices, h_resized, w_resized, combined_sal, proc):
    combined_sal = F.interpolate(combined_sal, size=(len(indices), h_resized, w_resized), mode='trilinear')
    
    t_orig = combined_sal.shape[2]
    pad_t = -t_orig % proc.temporal_patch_size
    if pad_t != 0:
        combined_sal = torch.cat([combined_sal, combined_sal[:, :, -1:].repeat(1, 1, pad_t, 1, 1)], dim=2)
    
    patch_sal = F.avg_pool3d(
        combined_sal, 
        kernel_size=(proc.temporal_patch_size, proc.patch_size, proc.patch_size),
        stride=(proc.temporal_patch_size, proc.patch_size, proc.patch_size)
    )

    token_sal = F.avg_pool3d(
        patch_sal,
        kernel_size=(1, proc.merge_size, proc.merge_size),
        stride=(1, proc.merge_size, proc.merge_size)
    )

    final_weights = torch.sigmoid(token_sal.view(-1))
        
    return final_weights


class VCDLogitsProcessor(LogitsProcessor):
    def __init__(self, original_logits_list, negative_logits_list, alpha=1.0):
        """
        三流 VCD Logits 处理器
        公式: Final_Logits = Positive_Logits + alpha * (Original_Logits - Negative_Logits)
        """
        self.original_logits_list = original_logits_list
        self.negative_logits_list = negative_logits_list
        self.alpha = alpha
        self.ptr = 0  # 追踪生成步数

    def __call__(self, input_ids, scores):
        # scores 在此处即为 Positive Logits (因为模型已经挂载了显著性权重的 Hook)
        positive_logits = scores

        # 确保不越界
        if self.ptr < len(self.original_logits_list) and self.ptr < len(self.negative_logits_list):
            
            # 1. 获取预计算的 Logits
            orig_t = self.original_logits_list[self.ptr].to(positive_logits.device)
            neg_t = self.negative_logits_list[self.ptr].to(positive_logits.device)

            # 2. 维度对齐 (Batch, Vocab)
            # 如果预计算的 logits 是 [Batch, 1, Vocab]，则 squeeze
            if orig_t.dim() == 3: orig_t = orig_t.squeeze(1)
            if neg_t.dim() == 3: neg_t = neg_t.squeeze(1)
            if orig_t.dim() == 1: orig_t = orig_t.unsqueeze(0)
            if neg_t.dim() == 1: neg_t = neg_t.unsqueeze(0)
            
            # 确保 batch 维度匹配 (假设 batch=1)
            if orig_t.shape[0] != positive_logits.shape[0]:
                 # 简单处理 batch 不匹配的情况，取第一条
                 orig_t = orig_t[0].unsqueeze(0)
                 neg_t = neg_t[0].unsqueeze(0)

            # 3. 计算 VCD 修正项 (Original - Negative)
            # 这是“抗幻觉”的方向向量
            contrastive_penalty = orig_t - neg_t

            # 4. 融合
            final_logits = positive_logits + self.alpha * contrastive_penalty
            
            self.ptr += 1
            return final_logits
        
        else:
            # 超过预计算长度，仅使用 Positive Logits
            self.ptr += 1
            return positive_logits


def _build_vqa_prompt(question, options=None):
    if options is not None:
        options_str = "\n".join(options)
        return f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下多项选择题。

要求：你的回答只能从给定的选项中选择（A, B, C），不要包含其他内容

问题：{question}
选项：
{options_str}
"""
    return f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下问题。

要求：你的回答只能是"yes"或"no"，不要包含其他内容

问题：{question}
"""


def _extract_generation_step_logits(generated_ids, input_ids):
    """
    返回按生成步组织的 logits 列表: List[Tensor(B=1, Vocab)]
    说明:
      - generated_ids.scores 的组织是 [t][batch, vocab]，t 是生成步。
      - 旧逻辑按 batch 维访问 scores，容易错位。
    """
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids.sequences)]
    if len(trimmed) == 0:
        return []
    # 当前训练/评估脚本均为 batch=1；为避免 silent bug，显式约束
    if len(trimmed) != 1:
        raise ValueError(f"Expected batch size 1, got {len(trimmed)}")
    gen_len = len(trimmed[0])
    step_logits = []
    for t in range(min(gen_len, len(generated_ids.scores))):
        score_t = generated_ids.scores[t]  # [B, V]
        step_logits.append(score_t[0:1])
    return step_logits
        

def answer_question_original(model, processor, video_path, question, options=None, max_new_tokens=8):
    prompt = _build_vqa_prompt(question, options)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 获取模型的隐藏状态和logits
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False
        )
    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
    generated_logits = _extract_generation_step_logits(generated_ids, inputs.input_ids)
    # generated_logits [batch_size, 1, vocab_size]
    # last_hidden_states [batch_size, input_token_len, embed_dim]
    # inputs.input_ids [batch_size, input_token_len]
    return generated_logits, last_hidden_states[0], inputs.input_ids


def answer_question_negative(model, processor, negative_video_path, question, options=None, max_new_tokens=8):
    prompt = _build_vqa_prompt(question, options)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": negative_video_path,  
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False
        )

    generated_logits = _extract_generation_step_logits(generated_ids, inputs.input_ids)
    return generated_logits


def answer_question_positive(
    model,
    processor,
    video_path,
    patch_weights,
    question,
    original_logits,
    negative_logits,
    options=None,
    max_new_tokens=8
):
    """
    输入:
        model: Qwen3-VL 模型
        patch_weights: 针对视觉 Patch 的显着性权重 Tensor [N_patches]
        original_logits: 原始视频生成的 logits 列表
        negative_logits: 负样本视频生成的 logits 列表
    流程:
        1. 注册 Hook 到 model.model.visual，将 patch_weights 注入视觉特征。
        2. 调用 model.generate，并通过 VCDLogitsProcessor 融合三方 Logits。
    """
    
    # --- 1. 构建 Prompt ---
    prompt = _build_vqa_prompt(question, options)
    
    # --- 2. 准备输入 ---
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # --- 3. 定义并注册 Hook (核心修正) ---
    def visual_forward_hook(module, input, output):
        # 1. 解包 Output
        # Transformers 模型的输出通常是 tuple (hidden_states, ...) 或 ModelOutput 对象
        # 报错提示它是 tuple，所以我们需要取出第一个元素
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            visual_feats = output[0]
        elif hasattr(output, 'last_hidden_state'): # 兼容 ModelOutput 对象
            visual_feats = output.last_hidden_state
            is_tuple = False # 标记处理逻辑不同
        else:
            visual_feats = output
        
        # 2. 统一 Tensor 形状处理
        is_2d_input = False
        if visual_feats.dim() == 2:
            visual_feats = visual_feats.unsqueeze(0) # [1, N, D]
            is_2d_input = True
            
        B, N, D = visual_feats.shape
        
        # 3. 准备权重
        w = patch_weights.to(visual_feats.device, dtype=visual_feats.dtype)
        
        # 4. 权重对齐逻辑 (插值)
        if w.shape[0] != N:
            # print(f"Info: Resizing weights from {w.shape[0]} to {N}")
            w = torch.nn.functional.interpolate(
                w.reshape(1, 1, -1),   
                size=N,                
                mode='linear',
                align_corners=False
            ).reshape(-1)              
            
        # 5. 应用权重 (广播乘法)
        # [B, N, D] * [1, N, 1]
        weighted_feats = visual_feats * w.reshape(1, N, 1)
        
        # 6. 还原形状
        if is_2d_input:
            weighted_feats = weighted_feats.squeeze(0)
            
        # 7. 打包返回 (保持原结构)
        if is_tuple:
            # 如果原输出是 tuple，我们需要返回一个新的 tuple，替换第一个元素
            return (weighted_feats,) + output[1:]
        elif hasattr(output, 'last_hidden_state'):
            # 如果是 ModelOutput 对象，通常可以直接修改属性（慎用）或返回 Tensor
            # 大多数情况下，Visual Encoder 返回 tuple 比较多，这里作为兜底
            output.last_hidden_state = weighted_feats
            return output
        else:
            return weighted_feats

    # 注册 Hook 到 model.model.visual
    try:
        visual_module = model.model.visual
    except AttributeError:
        visual_module = model.visual
        
    hook_handle = visual_module.register_forward_hook(visual_forward_hook)

    # --- 4. 生成与对比解码 ---
    try:
        # 初始化 VCD Logits Processor
        vcd_processor = VCDLogitsProcessor(
            original_logits_list=original_logits,
            negative_logits_list=negative_logits,
            alpha=1.0  
        )
        
        # 执行生成
        # 此时 model 内部的 visual features 已经被 hook 修改 (Positive Stream)
        # model.generate 输出的 scores 经过 vcd_processor 修正
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 对于 Yes/No 或选项，建议使用 Greedy Search
                logits_processor=LogitsProcessorList([vcd_processor]),
                output_scores=True,
                return_dict_in_generate=True
            )
        
    finally:
        # 务必移除 Hook，确保不影响后续代码
        hook_handle.remove()

    # --- 5. 解码结果 ---
    # 提取新生成的 tokens
    new_tokens = generated_ids.sequences[0][len(inputs.input_ids[0]):]
    generated_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    
    # 收集最终的 logits (混合后的)
    final_mixed_logits = []
    if generated_ids.scores:
        for score in generated_ids.scores:
            final_mixed_logits.append(score)

    return generated_text, generated_ids


def transform_pixel_to_patch(indices, h_resized, w_resized, saliency_map, proc):
    # 1. 插值/Resize (线性)
    # mode='trilinear' 是线性的
    saliency_map = F.interpolate(
        saliency_map, 
        size=(len(indices), h_resized, w_resized), 
        mode='trilinear',
        align_corners=False
    )
    
    # 2. 时间维度 Padding (线性)
    t_orig = saliency_map.shape[2]
    pad_t = -t_orig % proc.temporal_patch_size
    if pad_t != 0:
        saliency_map = torch.cat([
            saliency_map, 
            saliency_map[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
        ], dim=2)
    
    # 3. Patch Pooling (线性)
    patch_sal = F.avg_pool3d(
        saliency_map, 
        kernel_size=(proc.temporal_patch_size, proc.patch_size, proc.patch_size),
        stride=(proc.temporal_patch_size, proc.patch_size, proc.patch_size)
    )

    # 4. Token Merging (线性)
    token_sal = F.avg_pool3d(
        patch_sal,
        kernel_size=(1, proc.merge_size, proc.merge_size),
        stride=(1, proc.merge_size, proc.merge_size)
    )

    # 返回 Flatten 后的一维向量，不加 Sigmoid
    return token_sal.view(-1)
