import math
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,6"
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list



# 主函数
def main():
    model = "InternVL3_5-30B-A3B"
    path = "../../checkpoints/InternVL3_5-30B-A3B"
    qa_source_dir = "../../video_generated/m_mcqa.json"
    video_dir = "../../video_generated/all_video"
    os.makedirs(model, exist_ok=True)
    qa_target_dir = f"{model}/m_mcqa_{model}.json"

    # 加载模型
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    
    # 读取问题文件
    with open(qa_source_dir, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # 创建结果数据结构
    results = []
    start_index = 0
    
    if os.path.exists(qa_target_dir):
        print(f"检测到已有结果文件: {qa_target_dir}")
        with open(qa_target_dir, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        
        if existing_results:
            # 创建已完成的m_mcqa_id集合
            completed_ids = {item["m_mcqa_id"] for item in existing_results}
            results = existing_results  # 保留已完成的结果
            
            # 找到下一个需要处理的索引
            for i, item in enumerate(questions_data):
                if item["m_mcqa_id"] not in completed_ids:
                    start_index = i
                    break
            
            print(f"已找到 {len(completed_ids)} 个已完成的问题，从索引 {start_index} 开始继续处理")
        else:
            print("结果文件为空，从头开始处理")
    else:
        print("未找到现有结果文件，从头开始处理")
    
    # 处理每个问题
    for i, item in enumerate(questions_data[start_index:], start=start_index):
        m_mcqa_id = item["m_mcqa_id"]
        video_id = item["video_id"]
        question = item["mc_question"]
        options = item["mc_option"]
        
        print(f"处理问题 {m_mcqa_id}/{len(questions_data)}: {question}")
        
        # 查找对应的视频文件
        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            potential_path = os.path.join(video_dir, video_id + ext)
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if video_path is None:
            print(f"警告: 未找到视频文件 {video_id}")
            # 使用空结果填充
            result_item = {
                "m_mcqa_id": m_mcqa_id,
                "video_id": video_id,
                "answer": "",
                "reasoning": ""
            }
            results.append(result_item)
            continue
        
        # 生成答案
        try:
            options_str = "\n".join(options)
            prompt = f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下多项选择题。

        要求：
        1. 你的回答只能从给定的选项中选择（A, B, C），不要包含其他内容
        2. 请你提供推理过程

        问题：{question}
        选项：{options_str}

        请严格按照以下JSON格式返回结果：
        {{
            "answer": "A"或"B"或"C",
            "reasoning": "你的推理过程"
        }}
        """
            pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + prompt
            # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
            response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                        num_patches_list=num_patches_list, history=None, return_history=True)
            print(f"模型响应: {response}")
            
            # 解析响应
            # 尝试提取JSON部分
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                response_data = json.loads(json_str)
                answer = response_data.get("answer", "")
                reasoning = response_data.get("reasoning", "")
            else:
                # 如果没有找到JSON格式，尝试其他方式解析
                lines = response.split("\n")
                answer = ""
                reasoning = ""
                for line in lines:
                    if "answer" in line.lower():
                        if "A" in line.upper():
                            answer = "A"
                        elif "B" in line.upper():
                            answer = "B"
                        elif "C" in line.upper():
                            answer = "C"
                    if "reason" in line.lower() or "推理" in line:
                        reasoning = line.split(":", 1)[-1].strip()
        except Exception as e:
            print(f"处理问题 {m_mcqa_id} 时出错: {e}")
            answer = ""
            reasoning = ""
        
        # 添加到结果中
        result_item = {
            "m_mcqa_id": m_mcqa_id,
            "video_id": video_id,
            "answer": answer,
            "reasoning": reasoning
        }
        results.append(result_item)
        
        # 每处理20个问题保存一次结果
        if (i + 1) % 20 == 0:
            with open(qa_target_dir, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已保存 {i+1} 个问题的结果")
    
    # 保存最终结果
    with open(qa_target_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("所有问题处理完成!")

if __name__ == "__main__":
    main()