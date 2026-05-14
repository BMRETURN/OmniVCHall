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
def load_model(device):
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


# 根据视频生成描述
def generate_caption(model, processor, video_path):
    # 构建prompt
    prompt = "你是一个专业的AI视频问答助手，请使用完整且连续的一段话描述这个视频的内容。"
    
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


# 预先构建视频路径映射以避免重复查找
def build_video_mapping(video_dir):
    video_mapping = {}
    if os.path.exists(video_dir):
        for filename in os.listdir(video_dir):
            name, ext = os.path.splitext(filename)
            if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                video_mapping[name] = os.path.join(video_dir, filename)
    return video_mapping


# 主函数
def main():
    model = "LLaVA-NeXT-Video-34B"
    caption_source_dir = "../../video_generated/caption.json"
    video_dir = "../../video_generated/all_video"
    os.makedirs(model, exist_ok=True)
    caption_target_dir = f"{model}/caption_{model}.json"    

    # 加载模型
    model, processor = load_model()
    
    # 读取caption文件
    with open(caption_source_dir, "r", encoding="utf-8") as f:
        captions_data = json.load(f)
    
    # 创建视频路径映射
    video_mapping = build_video_mapping(video_dir)
    
    # 创建结果数据结构
    results = []
    
    # 处理每个视频
    for i, item in enumerate(captions_data):
        caption_id = item.get("caption_id", "")
        video_id = item.get("video_id", "")
        
        print(f"处理视频 {video_id} ({i+1}/{len(captions_data)})")
        
        # 查找对应的视频文件
        video_path = video_mapping.get(video_id)
        
        if video_path is None:
            print(f"警告: 未找到视频文件 {video_id}")
            # 使用空结果填充
            result_item = {
                "caption_id": caption_id,
                "video_id": video_id,
                "caption_a": ""
            }
            results.append(result_item)
            continue
        
        # 生成描述
        try:
            caption_a = generate_caption(model, processor, video_path)
            print(f"模型生成描述: {caption_a}")
        except Exception as e:
            print(f"处理视频 {video_id} 时出错: {e}")
            caption_a = ""
        
        # 添加到结果中
        result_item = {
            "caption_id": caption_id,
            "video_id": video_id,
            "caption_a": caption_a
        }
        results.append(result_item)
        
        # 每处理20个视频保存一次结果
        if (i + 1) % 20 == 0:
            with open(caption_target_dir, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已保存 {i+1} 个视频的结果")
    
    # 保存最终结果
    with open(caption_target_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("所有视频处理完成!")

if __name__ == "__main__":
    main()