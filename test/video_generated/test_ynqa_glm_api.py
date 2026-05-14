import os
import json
import torch
from openai import OpenAI


def load_model():
    print("正在初始化OpenAI客户端...")
    client = OpenAI(base_url="", api_key=os.getenv("", ""))
    print("OpenAI客户端初始化完成!")
    return client


# 根据问题和视频生成答案
def answer_question(client, video_path, question):
    # 构建prompt
    prompt = f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下问题。

要求：
1. 你的回答只能是"yes"或"no"，不要包含其他内容
2. 请你提供推理过程

问题：{question}

请严格按照以下JSON格式返回结果：
{{
    "answer": "yes/no",
    "reasoning": "你的推理过程"
}}
"""
    completion = client.chat.completions.create(
        model="glm-4.6v-flash",
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
        ],
        max_tokens = 10000,
    )
    return completion.choices[0].message.content


# 主函数
def main():
    model = "glm-4.6v-flash"
    qa_source_dir = "../../video_generated/m_ynqa.json"
    video_dir = "../../video_generated/all_video"
    os.makedirs(model, exist_ok=True)
    qa_target_dir = f"{model}/m_ynqa_{model}.json"

    client = load_model()
    
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
            # 创建已完成的m_ynqa_id集合
            completed_ids = {item["m_ynqa_id"] for item in existing_results}
            results = existing_results  # 保留已完成的结果
            
            # 找到下一个需要处理的索引
            for i, item in enumerate(questions_data):
                if item["m_ynqa_id"] not in completed_ids:
                    start_index = i
                    break
            
            print(f"已找到 {len(completed_ids)} 个已完成的问题，从索引 {start_index} 开始继续处理")
        else:
            print("结果文件为空，从头开始处理")
    else:
        print("未找到现有结果文件，从头开始处理")
    
    # 处理每个问题
    for i, item in enumerate(questions_data[start_index:], start=start_index):
        m_ynqa_id = item["m_ynqa_id"]
        video_id = item["video_id"]
        question = item["yn_question"]
        
        print(f"处理问题 {m_ynqa_id}/{len(questions_data)}: {question}")
        
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
                "m_ynqa_id": m_ynqa_id,
                "video_id": video_id,
                "answer": "",
                "reasoning": ""
            }
            results.append(result_item)
            continue
        
        # 生成答案
        try:
            response = answer_question(client, video_path, question)
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
                        if "yes" in line.lower():
                            answer = "yes"
                        elif "no" in line.lower():
                            answer = "no"
                    if "reason" in line.lower() or "推理" in line:
                        reasoning = line.split(":", 1)[-1].strip()
        except Exception as e:
            print(f"处理问题 {m_ynqa_id} 时出错: {e}")
            answer = ""
            reasoning = ""
        
        # 添加到结果中
        result_item = {
            "m_ynqa_id": m_ynqa_id,
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
