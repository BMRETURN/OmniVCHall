import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import json
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import av
import torch
import numpy as np


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# 加载模型和处理器
def load_model():
    print("正在加载 LLaVA-NeXT-Video-34B 模型...")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        "../../checkpoints/LLaVA-NeXT-Video-34B", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    processor = LlavaNextVideoProcessor.from_pretrained("../../checkpoints/LLaVA-NeXT-Video-34B")
    print("模型加载完成!")
    return model, processor


# 根据问题和视频生成答案
def answer_question(model, processor, video_path, question, options):
    # 构建prompt
    options_str = "\n".join(options)
    prompt = f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下多项选择题。

要求：
1. 你的回答只能从给定的选项中选择（A, B, C），不要包含其他内容
2. 请你提供推理过程

问题：{question}
选项：
{options_str}

请严格按照以下JSON格式返回结果：
{{
    "answer": "A"或"B"或"C",
    "reasoning": "你的推理过程"
}}
"""
    
    # 打开视频文件
    container = av.open(video_path)
    
    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)
    
    # 定义对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        },
    ]

    prompt_formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt_formatted, videos=clip, padding=True, return_tensors="pt").to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = processor.decode(output[0][2:], skip_special_tokens=True)
    
    # 提取ASSISTANT后的部分
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response

# 主函数
def main():
    model = "LLaVA-NeXT-Video-34B"
    qa_source_dir = "../../video_real/s_mcqa.json"
    video_dir = "../../video_real/all_video"
    os.makedirs(model, exist_ok=True)
    qa_target_dir = f"{model}/s_mcqa_{model}.json"

    # 加载模型
    model, processor = load_model()
    
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
            # 创建已完成的s_mcqa_id集合
            completed_ids = {item["s_mcqa_id"] for item in existing_results}
            results = existing_results  # 保留已完成的结果
            
            # 找到下一个需要处理的索引
            for i, item in enumerate(questions_data):
                if item["s_mcqa_id"] not in completed_ids:
                    start_index = i
                    break
            
            print(f"已找到 {len(completed_ids)} 个已完成的问题，从索引 {start_index} 开始继续处理")
        else:
            print("结果文件为空，从头开始处理")
    else:
        print("未找到现有结果文件，从头开始处理")
    
    # 处理每个问题
    for i, item in enumerate(questions_data[start_index:], start=start_index):
        s_mcqa_id = item["s_mcqa_id"]
        video_id = item["video_id"]
        question = item["mc_question"]
        options = item["mc_option"]
        
        print(f"处理问题 {s_mcqa_id}/{len(questions_data)}: {question}")
        
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
                "s_mcqa_id": s_mcqa_id,
                "video_id": video_id,
                "answer": "",
                "reasoning": ""
            }
            results.append(result_item)
            continue
        
        # 生成答案
        try:
            response = answer_question(model, processor, video_path, question, options)
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
            print(f"处理问题 {s_mcqa_id} 时出错: {e}")
            answer = ""
            reasoning = ""
        
        # 添加到结果中
        result_item = {
            "s_mcqa_id": s_mcqa_id,
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