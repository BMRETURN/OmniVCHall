# import os
# import json
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# import torch


# # 加载模型和处理器
# def load_model():
#     print("正在加载 Qwen3-VL-32B-Instruct 模型...")
#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#         "/home/storage1/wenbinxing/Qwen3-VL-32B-Instruct", 
#         dtype="auto", 
#         device_map="auto"
#     )
#     processor = AutoProcessor.from_pretrained("/home/storage1/wenbinxing/Qwen3-VL-32B-Instruct")
#     print("模型加载完成!")
#     return model, processor


# # 根据问题和视频生成答案
# def answer_question(model, processor, video_path, question, options):
#     # 构建prompt
#     options_str = "\n".join(options)
#     prompt = f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下多项选择题。

# 要求：
# 1. 你的回答只能从给定的选项中选择（A, B, C），不要包含其他内容
# 2. 请你提供推理过程

# 问题：{question}
# 选项：
# {options_str}

# 请严格按照以下JSON格式返回结果：
# {{
#     "answer": "A"或"B"或"C",
#     "reasoning": "你的推理过程"
# }}
# """
    
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "video",
#                     "video": video_path,
#                 },
#                 {"type": "text", "text": prompt},
#             ],
#         }
#     ]
    
#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         return_tensors="pt"
#     )
#     inputs = inputs.to(model.device)
    
#     generated_ids = model.generate(**inputs, max_new_tokens=512)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     return output_text[0]


# # 主函数
# def main():
#     model = "Qwen3-VL-32B-Instruct"
#     qa_source_dir = "../../video_generated/m_mcqa.json"
#     video_dir = "../../video_generated/all_video"
#     os.makedirs(model, exist_ok=True)
#     qa_target_dir = f"{model}/m_mcqa_{model}.json"

#     # 加载模型
#     model, processor = load_model()
    
#     # 读取问题文件
#     with open(qa_source_dir, "r", encoding="utf-8") as f:
#         questions_data = json.load(f)
    
#     # 创建结果数据结构
#     results = []
    
#     # 处理每个问题
#     for i, item in enumerate(questions_data):
#         m_mcqa_id = item["m_mcqa_id"]
#         video_id = item["video_id"]
#         question = item["mc_question"]
#         options = item["mc_option"]
        
#         print(f"处理问题 {m_mcqa_id}/{len(questions_data)}: {question}")
        
#         # 查找对应的视频文件
#         video_path = None
#         for ext in [".mp4", ".avi", ".mov", ".mkv"]:
#             potential_path = os.path.join(video_dir, video_id + ext)
#             if os.path.exists(potential_path):
#                 video_path = potential_path
#                 break
        
#         if video_path is None:
#             print(f"警告: 未找到视频文件 {video_id}")
#             # 使用空结果填充
#             result_item = {
#                 "m_mcqa_id": m_mcqa_id,
#                 "video_id": video_id,
#                 "answer": "",
#                 "reasoning": ""
#             }
#             results.append(result_item)
#             continue
        
#         # 生成答案
#         try:
#             response = answer_question(model, processor, video_path, question, options)
#             print(f"模型响应: {response}")
            
#             # 解析响应
#             # 尝试提取JSON部分
#             if "{" in response and "}" in response:
#                 json_start = response.find("{")
#                 json_end = response.rfind("}") + 1
#                 json_str = response[json_start:json_end]
#                 response_data = json.loads(json_str)
#                 answer = response_data.get("answer", "")
#                 reasoning = response_data.get("reasoning", "")
#             else:
#                 # 如果没有找到JSON格式，尝试其他方式解析
#                 lines = response.split("\n")
#                 answer = ""
#                 reasoning = ""
#                 for line in lines:
#                     if "answer" in line.lower():
#                         if "A" in line.upper():
#                             answer = "A"
#                         elif "B" in line.upper():
#                             answer = "B"
#                         elif "C" in line.upper():
#                             answer = "C"
#                     if "reason" in line.lower() or "推理" in line:
#                         reasoning = line.split(":", 1)[-1].strip()
#         except Exception as e:
#             print(f"处理问题 {m_mcqa_id} 时出错: {e}")
#             answer = ""
#             reasoning = ""
        
#         # 添加到结果中
#         result_item = {
#             "m_mcqa_id": m_mcqa_id,
#             "video_id": video_id,
#             "answer": answer,
#             "reasoning": reasoning
#         }
#         results.append(result_item)
        
#         # 每处理20个问题保存一次结果
#         if (i + 1) % 20 == 0:
#             with open(qa_target_dir, "w", encoding="utf-8") as f:
#                 json.dump(results, f, ensure_ascii=False, indent=2)
#             print(f"已保存 {i+1} 个问题的结果")
    
#     # 保存最终结果
#     with open(qa_target_dir, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)
    
#     print("所有问题处理完成!")

# if __name__ == "__main__":
#     main()



import os
# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import time


# 获取GPU显存使用情况
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        memory_info = torch.cuda.memory_stats()
        allocated = memory_info['allocated_bytes.all.current'] / 1024**3  # 转换为GB
        reserved = memory_info['reserved_bytes.all.current'] / 1024**3  # 转换为GB
        return allocated, reserved
    return 0, 0


# 加载模型和处理器
def load_model():
    print("正在加载Qwen3-VL-8B-Instruct 模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "../../checkpoints/Qwen3-VL-8B-Instruct", 
        dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("../../checkpoints/Qwen3-VL-8B-Instruct")
    print("模型加载完成!")
    
    # 记录初始显存使用
    initial_allocated, initial_reserved = get_gpu_memory_usage()
    print(f"初始显存分配: {initial_allocated:.2f} GB, 预留: {initial_reserved:.2f} GB")
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
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# 主函数
def main():
    model = "Qwen3-VL-2B-Instruct"
    qa_source_dir = "../../video_generated/s_mcqa.json"
    video_dir = "../../video_generated/all_video"

    # 加载模型
    model, processor = load_model()
    
    # 读取问题文件
    with open(qa_source_dir, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # 记录总体统计信息
    total_duration = 0
    total_questions = len(questions_data)
    memory_usages = []
    
    # 处理每个问题
    for i, item in enumerate(questions_data):
        s_mcqa_id = item["s_mcqa_id"]
        video_id = item["video_id"]
        question = item["mc_question"]
        options = item["mc_option"]
        
        # 记录开始时间
        start_time = time.time()
        
        # 记录开始时的显存使用
        start_allocated, start_reserved = get_gpu_memory_usage()
        
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
        
        # 记录结束时间并计算耗时
        end_time = time.time()
        duration = end_time - start_time
        total_duration += duration
        
        # 记录结束时的显存使用
        end_allocated, end_reserved = get_gpu_memory_usage()
        memory_diff = end_allocated - start_allocated
        memory_usages.append(end_allocated)
        
        print(f"问题 {s_mcqa_id} 处理完成")
        print(f"  - 耗时: {duration:.2f} 秒")
        print(f"  - 显存变化: {start_allocated:.2f} GB -> {end_allocated:.2f} GB ({memory_diff:+.2f} GB)")
        print("-" * 60)
        
        # 每处理20个问题输出一次统计
        if (i + 1) % 20 == 0:
            avg_duration_so_far = total_duration / (i + 1)
            avg_memory_so_far = sum(memory_usages) / len(memory_usages)
            print(f"已处理 {i+1} 个问题，当前平均值:")
            print(f"  - 平均处理时间: {avg_duration_so_far:.2f} 秒")
            print(f"  - 平均显存使用: {avg_memory_so_far:.2f} GB")
            print("=" * 60)
    
    # 计算并打印总体统计信息
    avg_duration = total_duration / total_questions if total_questions > 0 else 0
    avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
    
    print(f"\n总体统计:")
    print(f"总问题数: {total_questions}")
    print(f"总处理时间: {total_duration:.2f} 秒")
    print(f"平均处理时间: {avg_duration:.2f} 秒/问题")
    print(f"平均显存使用: {avg_memory_usage:.2f} GB")
    
    print("所有问题处理完成!")

if __name__ == "__main__":
    main()