import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu   
from scipy.spatial import cKDTree
import numpy as np
import math

MAX_NUM_FRAMES=180 # Indicates the maximum number of frames received after the videos are packed. The actual maximum number of valid frames is MAX_NUM_FRAMES * MAX_NUM_PACKING.
MAX_NUM_PACKING=3  # indicates the maximum packing number of video frames. valid range: 1-6
TIME_SCALE = 0.1 


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video(video_path, choose_fps=3, force_packing=None):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
        
    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
        
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]      
    frame_idx =  np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)
    
    print(video_path, ' duration:', video_duration)
    print(f'get video frames={len(frame_idx)}, packing_nums={packing_nums}')
    
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    
    return frames, frame_ts_id_group



# 主函数
def main():
    device = "cuda:0" 
    model = "MiniCPM-V-4_5"
    qa_source_dir = "../../video_generated/m_ynqa.json"
    video_dir = "../../video_generated/all_video"
    os.makedirs(model, exist_ok=True)
    qa_target_dir = f"{model}/m_ynqa_{model}.json"

    # 加载模型
    model = AutoModel.from_pretrained('../../checkpoints/MiniCPM-V-4_5', trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained('../../checkpoints/MiniCPM-V-4_5', trust_remote_code=True)  # or openbmb/MiniCPM-o-2_6
    fps = 2 # fps for video
    force_packing = None # You can set force_packing to ensure that 3D packing is forcibly enabled; otherwise, encode_video will dynamically set the packing quantity based on the duration.
    
    # 读取问题文件
    with open(qa_source_dir, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # 创建结果数据结构
    results = []
    
    # 处理每个问题
    for i, item in enumerate(questions_data):
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
            prompt = f"""你是一个专业的AI视频问答助手。请根据视频内容，回答以下问题。

问题：{question}

要求：你在回答中只能是"yes"或"no"，不要包含其他内容

"""         
            frames, frame_ts_id_group = encode_video(video_path, fps, force_packing=force_packing)
            msgs = [
                {'role': 'user', 'content': frames + [prompt]}, 
            ]
            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                use_image_id=False,
                max_slice_nums=1,
                temporal_ids=frame_ts_id_group
            )
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
                    if "yes" in line.lower():
                        answer = "yes"
                    elif "no" in line.lower():
                        answer = "no"
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