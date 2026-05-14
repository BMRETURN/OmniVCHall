import os
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch


# 加载模型和处理器
def load_model():
    print("正在加载 Qwen3-VL-32B-Instruct 模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "/home/storage1/wenbinxing/Qwen3-VL-32B-Instruct", 
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("/home/storage1/wenbinxing/Qwen3-VL-32B-Instruct")
    print("模型加载完成!")
    return model, processor


# 根据视频生成描述
def generate_caption(model, processor, video_path):
    # 构建prompt
    prompt = "你是一个专业的AI视频问答助手，请使用完整且连续的一段话描述这个视频的内容。"
    
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
    
    # 减少max_new_tokens以加快生成速度
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


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
    model = "Qwen3-VL-32B-Instruct"
    caption_source_dir = "../../video_real/caption.json"
    video_dir = "../../video_real/all_video"
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