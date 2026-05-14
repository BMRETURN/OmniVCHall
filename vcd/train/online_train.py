import os
import re
import sys
from pathlib import Path

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
    compute_patch_saliency_weights,
    answer_question_original,
    answer_question_negative,
    answer_question_positive
)
from vcd_new.train.optimizer import VCDPolicy, VCDTrainer
from vcd_new.models.selector import QFormerToolRouter
from vcd_new.models.gate import QueryVisualFusionGater
from vcd_new.tools.negative_tools import (
    ReverseVideo,
    SampleVideo,
    ShuffleVideo,
    BlurVideo,
    NoiseVideo,
    HorizontalMirrorVideo,
    VerticalMirrorVideo,
    GrayscaleVideo
)
from vcd_new.tools.positive_tools import (
    MotionSaliencyExtractor,
    DINOv3SaliencyExtractor
)


def evaluate_dataset(dataset, model, processor, policy, tools_embeddings, tools_dict, 
                     patch_processor, motion_sal_extractor, visual_sal_extractor, 
                     video_dir, primary_device, output_dir, stage_name="Validation", epoch=None):
    
    print(f"\n>>> Running {stage_name}..." + (f" (Epoch {epoch})" if epoch is not None else ""))
    os.makedirs(output_dir, exist_ok=True)
    
    policy.eval() # 切换到评估模式 (关闭 Dropout, BatchNorm)
    
    # 初始化统计字典
    metrics = {k: {"correct": 0, "total": 0} for k in ["s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"]}
    # 用于保存详细的推理日志
    detailed_logs = []
    
    pbar = tqdm(dataset, desc=stage_name)
    for item in pbar:
        # --- 1. 数据解析 (与训练逻辑保持严格一致) ---
        video_id = item["video_id"]
        
        # 确定题型、问题和 GT
        if "s_ynqa_id" in item: q_type="s_ynqa"; q=item["yn_question"]; gt=item["yn_answer"]
        elif "m_ynqa_id" in item: q_type="m_ynqa"; q=item["yn_question"]; gt=item["yn_answer"]
        elif "s_mcqa_id" in item: q_type="s_mcqa"; q=item["mc_question"]; gt=item["mc_answer"]
        elif "m_mcqa_id" in item: q_type="m_mcqa"; q=item["mc_question"]; gt=item["mc_answer"]
        else: continue # 跳过未知格式
        
        opts = item.get("mc_option", None)
        qa_id = item.get("s_ynqa_id") or item.get("m_ynqa_id") or item.get("s_mcqa_id") or item.get("m_mcqa_id")

        # 找视频
        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            p = os.path.join(video_dir, video_id + ext)
            if os.path.exists(p): video_path = p; break
        if not video_path: continue

        try:
            # --- 2. VCD 推理流程 (No Grad) ---
            # Step A: 原始推理
            with torch.no_grad():
                orig_logits, last_hidden_states, _ = answer_question_original(model, processor, video_path, q, opts)
            
            # 状态跨设备传输 & 转 FP32
            state = last_hidden_states.detach().to(primary_device, dtype=torch.float32)

            # Step B: 策略决策 (Deterministic: std_dev=0.0)
            # 验证集不应该采样，应使用网络输出的确定性结果
            sel_tools, beta_tensor, _, _ = policy.get_action_and_log_prob(state, tools_embeddings, std_dev=0.0)
            beta_val = beta_tensor.item()

            # Step C: 负样本流
            frames = read_video(video_path)
            neg_frames = frames
            for t_name in sel_tools:
                neg_frames = tools_dict[t_name].process(neg_frames)
            neg_path = save_video_to_temp(neg_frames, video_path)
            
            with torch.no_grad():
                neg_logits = answer_question_negative(model, processor, neg_path, q, opts)
            os.remove(neg_path)

            # Step D: 显着性计算
            meta = patch_processor.get_video_metadata(video_path)
            indices = patch_processor.get_sampling_indices(meta['total_frames'], meta['fps'])
            with torch.no_grad():
                m_sals = torch.tensor(motion_sal_extractor.extract_motion_saliency(frames, indices=indices.tolist()))
                v_sals = torch.tensor(visual_sal_extractor.extract_dino_video_pixel_last(frames, indices=indices.tolist()))
            
            grid_t, grid_h, grid_w, h_bar, w_bar = patch_processor.get_smart_resize_grid(len(indices), meta['height'], meta['width'])
            comb_sal = (beta_val * m_sals + (1 - beta_val) * v_sals).unsqueeze(0).unsqueeze(0)
            weights = compute_patch_saliency_weights(indices, h_bar, w_bar, comb_sal, patch_processor)

            # Step E: 正样本 VCD 推理 (得到最终预测)
            with torch.no_grad():
                pred, _ = answer_question_positive(model, processor, video_path, weights, q, orig_logits, neg_logits, opts)

            # --- 3. 统计结果 ---
            # 简单清洗文本进行比对 (A/B/C or Yes/No)
            pred_clean = re.search(r'\b(a|b|c|d|yes|no)\b', str(pred).lower())
            pred_token = pred_clean.group(1) if pred_clean else str(pred).lower().strip()
            gt_clean = re.search(r'\b(a|b|c|d|yes|no)\b', str(gt).lower())
            gt_token = gt_clean.group(1) if gt_clean else str(gt).lower().strip()
            
            is_correct = (pred_token == gt_token)
            
            if is_correct: metrics[q_type]["correct"] += 1
            metrics[q_type]["total"] += 1
            
            # 记录详细日志
            detailed_logs.append({
                "qa_id": qa_id,
                "video_id": video_id,
                "type": q_type,
                "question": q,
                "gt": gt_token,
                "pred": pred_token,
                "is_correct": is_correct,
                "beta": round(beta_val, 4),
            })
            
        except Exception as e:
            print(f"Eval Error {video_id}: {e}")
            continue

    # --- 4. 汇总与打印 ---
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
    
    # 这里使用简单的宏平均 (Macro Average) 作为主要指标，你也可以改用微平均 (Micro Average)
    avg_acc = total_acc_sum / valid_types if valid_types > 0 else 0.0
    print(f"Average Accuracy: {avg_acc:.2%}")
    summary_dict["average_acc"] = avg_acc

    # --- 5. 保存结果文件 ---
    filename_prefix = f"{stage_name.lower()}"
    if epoch is not None:
        filename_prefix += f"_epoch_{epoch+1}"
    
    # 保存 Metrics
    with open(os.path.join(output_dir, f"{filename_prefix}_metrics.json"), "w") as f:
        json.dump(summary_dict, f, indent=4)
        
    # 保存详细日志 (用于分析 Case)
    with open(os.path.join(output_dir, f"{filename_prefix}_details.json"), "w") as f:
        json.dump(detailed_logs, f, ensure_ascii=False, indent=2)
        
    return avg_acc


if __name__ == "__main__":
    torch.manual_seed(2025)
    model = "Qwen3-VL-8B-Instruct"
    model_dir = "../../checkpoints/Qwen3-VL-8B-Instruct"
    train_qa_source_dir = "../../dataset/MyBench/train"
    test_qa_source_dir = "../../dataset/MyBench/test"
    val_qa_source_dir = "../../dataset/MyBench/val"
    video_dir = "../../dataset/MyBench/all_video"
    tools_dir = "../tools/tools_embeddings_qwen3vl.pkl"
    test_output_dir = "test_output"
    val_output_dir = "val_output"
    batch_size = 32
    epochs = 3

    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    device = "cuda:6" if torch.cuda.is_available() else "cpu"

    print("正在加载工具描述embeddings...")
    tools_embeddings = load_embeddings(tools_dir, device)

    print("正在加载数据...")
    train_data = load_qa_data(train_qa_source_dir, shuffle=True)
    val_data = load_qa_data(val_qa_source_dir, shuffle=False)
    test_data = load_qa_data(test_qa_source_dir, shuffle=False)
    print(f"Data Loaded -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    print(f"正在加载{model}模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, 
        dtype="float16", 
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tools_dict = {
                'ReverseVideo': ReverseVideo(),
                'SampleVideo': SampleVideo(),
                'ShuffleVideo': ShuffleVideo(),
                'BlurVideo': BlurVideo(),
                'NoiseVideo': NoiseVideo(),
                'HorizontalMirrorVideo': HorizontalMirrorVideo(),
                'VerticalMirrorVideo': VerticalMirrorVideo(),
                'GrayscaleVideo': GrayscaleVideo()
            }
    tool_names = list(tools_dict.keys())

    motion_sal_extractor = MotionSaliencyExtractor()
    visual_sal_extractor = DINOv3SaliencyExtractor(device=device)
    patch_processor = PatchProcessor()

    selector = QFormerToolRouter(
        num_tools=8, d_in=4096, d_model=1024,
        n_heads=8, n_query_tokens=16,
        n_cond_blocks=2, n_tool_blocks=2,
        use_l2norm=True,
        device=device
    )
    gater = QueryVisualFusionGater(embed_dim=4096).to(device)

    policy = VCDPolicy(selector, gater, tool_names).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)
    trainer = VCDTrainer(policy, optimizer, accumulation_steps=batch_size)

    print("\n=== Start Training ===")
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n>>> Epoch {epoch + 1}/{epochs}")
        
        # --- Train Phase ---
        policy.train()

        train_rewards = []
        train_losses = []
        
        pbar = tqdm(train_data, desc="Training")
        for item in pbar:
            # --- 数据解析 (训练部分) ---
            vid = item["video_id"]
            if "s_ynqa_id" in item: q=item["yn_question"]; gt=item["yn_answer"]
            elif "m_ynqa_id" in item: q=item["yn_question"]; gt=item["yn_answer"]
            elif "s_mcqa_id" in item: q=item["mc_question"]; gt=item["mc_answer"]
            elif "m_mcqa_id" in item: q=item["mc_question"]; gt=item["mc_answer"]
            else: continue
            opts = item.get("mc_option", None)

            v_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                p = os.path.join(video_dir, vid + ext)
                if os.path.exists(p): v_path = p; break
            if not v_path: continue

            try:
                # 1. 原始推理
                with torch.no_grad():
                    orig_logits, last_hidden, _ = answer_question_original(model, processor, v_path, q, opts)
                
                # 跨设备 FP32
                state = last_hidden.detach().to(dtype=torch.float32)

                # 2. 策略采样 (Train: std_dev=0.1)
                sel_tools, beta_tensor, log_p_router, log_p_gater = policy.get_action_and_log_prob(state, tools_embeddings)
                beta_val = beta_tensor.item()

                # 3. 负样本流
                frames = read_video(v_path)
                neg_frames = frames
                for t in sel_tools: neg_frames = tools_dict[t].process(neg_frames)
                neg_path = save_video_to_temp(neg_frames, v_path)
                with torch.no_grad():
                    neg_logits = answer_question_negative(model, processor, neg_path, q, opts)
                os.remove(neg_path)

                # 4. 显着性
                meta = patch_processor.get_video_metadata(v_path)
                indices = patch_processor.get_sampling_indices(meta['total_frames'], meta['fps'])
                with torch.no_grad():
                    m_sals = torch.tensor(motion_sal_extractor.extract_motion_saliency(frames, indices=indices.tolist()))
                    v_sals = torch.tensor(visual_sal_extractor.extract_dino_video_pixel_last(frames, indices=indices.tolist()))
                
                gt_h, gt_h, gw, h_bar, w_bar = patch_processor.get_smart_resize_grid(len(indices), meta['height'], meta['width'])
                comb_sal = (beta_val * m_sals + (1 - beta_val) * v_sals).unsqueeze(0).unsqueeze(0)
                weights = compute_patch_saliency_weights(indices, h_bar, w_bar, comb_sal, patch_processor)

                # 5. 正样本推理
                with torch.no_grad():
                    pred, _ = answer_question_positive(model, processor, v_path, weights, q, orig_logits, neg_logits, opts)

                # 6. Step (更新)
                reward = trainer.compute_reward(pred, gt)
                loss = trainer.step(reward, log_p_router, log_p_gater)
                
                train_rewards.append(reward)
                # 只有当执行了 optimizer step 时 loss 才有意义，平滑一下显示
                if loss != 0: train_losses.append(loss)
                
                pbar.set_postfix({"Loss": f"{loss:.4f}", "AvgR": f"{np.mean(train_rewards):.3f}"})

            except Exception as e:
                print(f"Train Error: {e}")
                continue

        # --- Validation Phase ---
        # 传入 output_dir 用于保存详细结果
        val_acc = evaluate_dataset(
            val_data, model, processor, policy, tools_embeddings, tools_dict,
            patch_processor, motion_sal_extractor, visual_sal_extractor,
            video_dir, device, val_output_dir, stage_name="Validation", epoch=epoch+1
        )
        
        # --- Checkpointing ---
        # 保存每个 Epoch 的 Policy
        torch.save(policy.state_dict(), f"vcd_policy_epoch_{epoch+1}.pth")
        
        # 保存最佳 Policy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(policy.state_dict(), "vcd_policy_best.pth")
            print(f"!!! New Best Model Saved (Acc: {best_val_acc:.2%}) !!!")

    # ==============================================================================
    # 7. Testing Phase (使用最佳模型)
    # ==============================================================================
    print("\n=== Training Finished. Running Test on Best Model... ===")
    
    if os.path.exists("vcd_policy_best.pth"):
        policy.load_state_dict(torch.load("vcd_policy_best.pth"))
        print("Loaded vcd_policy_best.pth")
    else:
        print("Warning: Best model not found, utilizing weights from last epoch.")
    
    # 运行测试并保存到 test_output_dir
    test_acc = evaluate_dataset(
        test_data, model, processor, policy, tools_embeddings, tools_dict,
        patch_processor, motion_sal_extractor, visual_sal_extractor,
        video_dir, device, test_output_dir, stage_name="Testing", epoch=None
    )
    
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")
    print(f"Detailed logs saved to {test_output_dir}")



    # results = []
    # total_rewards = 0
    # print("开始训练流程...")

    # # 处理每个问题
    # for i, item in enumerate(questions_data):
    #     video_id = item["video_id"]
    #     if "s_ynqa_id" in item:
    #         qa_id = item["s_ynqa_id"]
    #         question = item["yn_question"]
    #         ground_truth = item["yn_answer"]
    #         question_type = "s_ynqa"
    #     elif "m_ynqa_id" in item:
    #         qa_id = item["m_ynqa_id"]
    #         question = item["yn_question"]
    #         ground_truth = item["yn_answer"]
    #         question_type = "m_ynqa"
    #     elif "s_mcqa_id" in item:
    #         qa_id = item["s_mcqa_id"]
    #         question = item["mc_question"]
    #         ground_truth = item["mc_answer"]
    #         question_type = "s_mcqa"
    #     elif "m_mcqa_id" in item:
    #         qa_id = item["m_mcqa_id"]
    #         question = item["mc_question"]
    #         ground_truth = item["mc_answer"]
    #         question_type = "m_mcqa"

    #     if "mc_option" in item:
    #         options = item["mc_option"]
    #     else:
    #         options = None
        
    #     print(f"Step {i+1}/{len(questions_data)} | Video: {video_id} | Q: {question}")
        
    #     # 查找对应的视频文件
    #     video_path = None
    #     for ext in [".mp4", ".avi", ".mov", ".mkv"]:
    #         potential_path = os.path.join(video_dir, video_id + ext)
    #         if os.path.exists(potential_path):
    #             video_path = potential_path
    #             break
        
    #     if video_path is None:
    #         print(f"警告: 未找到视频文件 {video_id}")

    #     try:
    #         with torch.no_grad():
    #             original_logits, last_hidden_states, input_ids = answer_question_original(
    #                 model, processor, video_path, question, options
    #             )

    #         video_emb_state = last_hidden_states.detach().to(dtype=torch.float32)

    #         selected_tools, beta_tensor, router_log_prob, gater_log_prob = policy.get_action_and_log_prob(
    #             video_emb_state, tools_embeddings, std_dev=0.1 
    #         )
    #         beta_val = beta_tensor.item()

    #         print(f"-> Policy Action: Beta={beta_val:.3f}, Tools={selected_tools}")

    #         video_frames = read_video(video_path)
    #         negative_frames = video_frames
    #         # 应用选中的工具
    #         for tool_name in selected_tools:
    #             tool = tools_dict[tool_name]
    #             negative_frames = tool.process(negative_frames)
    #         negative_video_path = save_video_to_temp(negative_frames, video_path)

    #         with torch.no_grad():
    #             negative_logits = answer_question_negative(
    #                 model, processor, negative_video_path, question, options
    #             )
    #         os.remove(negative_video_path)

    #         video_metadata = patch_processor.get_video_metadata(video_path)
    #         indices = patch_processor.get_sampling_indices(video_metadata['total_frames'], video_metadata['fps'])
    #         motion_sals = torch.tensor(motion_sal_extractor.extract_motion_saliency(video_frames, indices=indices.tolist()))
    #         visual_sals = torch.tensor(visual_sal_extractor.extract_dino_video_pixel_last(video_frames, indices=indices.tolist()))
            
    #         grid_t, grid_h, grid_w, h_bar, w_bar = patch_processor.get_smart_resize_grid(
    #             len(indices), video_metadata['height'], video_metadata['width']
    #         )
    #         combined_sal = (beta_val * motion_sals + (1 - beta_val) * visual_sals).unsqueeze(0).unsqueeze(0)
    #         final_weights = compute_patch_saliency_weights(indices, h_bar, w_bar, combined_sal, patch_processor)

    #         with torch.no_grad():
    #             pred_answer, _ = answer_question_positive(
    #                 model, processor, video_path, final_weights, question, 
    #                 original_logits, negative_logits, options
    #             )
            
    #         reward = trainer.compute_reward(pred_answer, ground_truth)
    #         loss_val = trainer.step(reward, router_log_prob, gater_log_prob)

    #         total_rewards += reward
    #         avg_reward = total_rewards / (i + 1)
            
    #         print(f"-> Result: GT={ground_truth} | Pred={pred_answer}")
    #         print(f"-> Metric: Reward={reward} | Loss={loss_val:.4f} | Avg Reward={avg_reward:.3f}")
    #         print("-" * 50)
            
    #         # 记录结果用于后续分析
    #         results.append({
    #             "qa_id": qa_id,
    #             "video_id": video_id,
    #             "gt": ground_truth,
    #             "pred": pred_answer,
    #             "reward": reward,
    #             "beta": beta_val,
    #             "tools": selected_tools
    #         })

    #         # 定期保存 Checkpoint
    #         if (i + 1) % 50 == 0:
    #             ckpt_path = f"vcd_policy_ckpt_{i+1}.pth"
    #             torch.save({
    #                 'policy_state_dict': policy.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'baseline': trainer.baseline
    #             }, ckpt_path)
    #             print(f"模型已保存至 {ckpt_path}")

    #     except Exception as e:
    #         print(f"处理视频 {video_id} 时发生错误: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         continue

    # # 训练结束保存最终模型
    # torch.save(policy.state_dict(), "vcd_policy_final.pth")
    
    # # 保存训练日志
    # with open("training_logs.json", "w") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
