import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


# 加载模型和处理器
def load_model():
    print("正在加载 Molmo2-8B 模型...")
    model_name = "../../checkpoints//Molmo2-8B"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
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
    "answer": "你的答案，只能是A或B或C",
    "reasoning": "你的推理过程"
}}
"""
    
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=prompt),
                dict(type="video", video=video_path),
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # generate output
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    # only get generated tokens; decode them to text
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


# 主函数
def main():
    model = "Molmo2-8B"
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
    
    # 处理每个问题
    for i, item in enumerate(questions_data):
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
                    if "answer" or "答案" in line.lower():
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