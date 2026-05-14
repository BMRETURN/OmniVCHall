import os
import json
from typing import Dict, List


def evaluate_ynqa_answer(predictions: List[Dict], ground_truth: List[Dict], qa_type) -> Dict[str, float]:
    """
    评估 Yes/No QA 类型问题的回答
    
    Args:
        predictions: 模型预测结果列表，每个元素包含 '{qa_type}_id' 和 'answer' 键
        ground_truth: 真实答案列表，每个元素包含 '{qa_type}_id' 和 'yn_answer' 键
    
    Returns:
        包含各种评估指标的字典
    """
    # 创建真实答案的映射字典
    truth_dict = {item[f'{qa_type}_id']: item['yn_answer'].lower() for item in ground_truth}
    
    # 初始化计数器
    total_questions = 0
    correct_answers = 0
    yes_questions = 0
    no_questions = 0
    correct_yes = 0
    correct_no = 0
    false_positives = 0  # 错误地回答"yes"（实际应为"no"）
    false_negatives = 0  # 错误地回答"no"（实际应为"yes"）
    
    # 遍历预测结果
    for pred in predictions:
        qa_id = pred[f'{qa_type}_id']
        pred_answer = pred['answer'].lower().strip()
        
        # 确保预测的问题ID在真实答案中存在
        if qa_id not in truth_dict:
            continue
            
        truth_answer = truth_dict[qa_id]
        
        # 只考虑预测答案为 yes 或 no 的情况
        if pred_answer not in ['yes', 'no']:
            continue
            
        total_questions += 1
        
        # 统计 yes 和 no 问题的数量
        if truth_answer == 'yes':
            yes_questions += 1
        elif truth_answer == 'no':
            no_questions += 1
            
        # 计算正确答案数
        if pred_answer == truth_answer:
            correct_answers += 1
            if truth_answer == 'yes':
                correct_yes += 1
            elif truth_answer == 'no':
                correct_no += 1
        else:
            # 计算错误分类情况
            if pred_answer == 'yes' and truth_answer == 'no':
                false_positives += 1
            elif pred_answer == 'no' and truth_answer == 'yes':
                false_negatives += 1
    
    # 计算评估指标
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    yes_accuracy = correct_yes / yes_questions if yes_questions > 0 else 0.0
    no_accuracy = correct_no / no_questions if no_questions > 0 else 0.0
    false_positive_rate = false_positives / no_questions if no_questions > 0 else 0.0
    false_negative_rate = false_negatives / yes_questions if yes_questions > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Yes Accuracy": yes_accuracy,
        "No Accuracy": no_accuracy,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate
    }


# def evaluate_mcqa_answer(predictions: List[Dict], ground_truth: List[Dict], qa_type) -> Dict[str, float]:
#     """
#     评估 Multiple-Choice QA 类型问题的回答
    
#     Args:
#         predictions: 模型预测结果列表，每个元素包含 '{qa_type}_id' 和 'answer' 键
#         ground_truth: 真实答案列表，每个元素包含 '{qa_type}_id' 和 'mc_answer' 键
    
#     Returns:
#         包含各种评估指标的字典
#     """
#     # 创建真实答案的映射字典，以 {qa_type}_id 为键
#     truth_dict = {item[f'{qa_type}_id']: item['mc_answer'].upper() for item in ground_truth}
    
#     # 初始化计数器
#     total_questions = 0
#     correct_answers = 0
    
#     # 用于计算每个选项的准确率、精确率和召回率
#     # tp: true positives, fp: false positives, fn: false negatives
#     option_tp = {'A': 0, 'B': 0, 'C': 0}  # 真正例
#     option_fp = {'A': 0, 'B': 0, 'C': 0}  # 假正例
#     option_fn = {'A': 0, 'B': 0, 'C': 0}  # 假负例
#     option_support = {'A': 0, 'B': 0, 'C': 0}  # 每个类别在真实标签中的数量
    
#     # 遍历预测结果
#     for pred in predictions:
#         qa_id = pred[f'{qa_type}_id']
#         pred_answer = pred['answer'].upper().strip()
        
#         # 确保预测的问题ID在真实答案中存在
#         if qa_id not in truth_dict:
#             continue
            
#         truth_answer = truth_dict[qa_id]
        
#         # 只考虑预测答案为 A, B 或 C 的情况
#         if pred_answer not in ['A', 'B', 'C']:
#             continue
            
#         total_questions += 1
        
#         # 统计各选项在真实标签中的数量
#         if truth_answer in option_support:
#             option_support[truth_answer] += 1
            
#         # 计算正确答案数
#         if pred_answer == truth_answer:
#             correct_answers += 1
#             # 增加真正例计数
#             if truth_answer in option_tp:
#                 option_tp[truth_answer] += 1
#         else:
#             # 增加假正例和假负例计数
#             if pred_answer in option_fp:
#                 option_fp[pred_answer] += 1
#             if truth_answer in option_fn:
#                 option_fn[truth_answer] += 1
    
#     # 计算总体准确率
#     accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
#     # 计算每个选项的准确率、精确率和召回率
#     option_metrics = {}
#     macro_precision = 0.0
#     macro_recall = 0.0
#     valid_classes = 0  # 用于计算宏平均的有效类别数
    
#     for option in ['A', 'B', 'C']:
#         # 准确率已在上面计算，这里计算精确率和召回率
#         precision = option_tp[option] / (option_tp[option] + option_fp[option]) if (option_tp[option] + option_fp[option]) > 0 else 0.0
#         recall = option_tp[option] / (option_tp[option] + option_fn[option]) if (option_tp[option] + option_fn[option]) > 0 else 0.0
        
#         option_metrics[f"{option} Precision"] = precision
#         option_metrics[f"{option} Recall"] = recall
        
#         # 累加宏平均值（只计算有支持度的类别）
#         if option_support[option] > 0:
#             macro_precision += precision
#             macro_recall += recall
#             valid_classes += 1
    
#     # 计算宏平均精确率和召回率
#     macro_precision = macro_precision / valid_classes if valid_classes > 0 else 0.0
#     macro_recall = macro_recall / valid_classes if valid_classes > 0 else 0.0
    
#     # 构建结果字典
#     result = {
#         "Accuracy": accuracy,
#         "Macro Precision": macro_precision,
#         "Macro Recall": macro_recall
#     }
#     result.update(option_metrics)
    
#     return result
def evaluate_mcqa_answer(predictions: List[Dict], ground_truth: List[Dict], qa_type) -> Dict[str, float]:
    """
    评估 Multiple-Choice QA 类型问题的回答
    """
    truth_dict = {item[f'{qa_type}_id']: item['mc_answer'].upper() for item in ground_truth}

    total_questions = 0
    correct_answers = 0

    option_tp = {'A': 0, 'B': 0, 'C': 0}
    option_fp = {'A': 0, 'B': 0, 'C': 0}
    option_fn = {'A': 0, 'B': 0, 'C': 0}
    option_support = {'A': 0, 'B': 0, 'C': 0}

    for pred in predictions:
        qa_id = pred.get(f'{qa_type}_id')
        if qa_id not in truth_dict:
            continue

        pred_answer = str(pred.get('answer', '')).upper().strip()
        truth_answer = truth_dict[qa_id]

        if pred_answer not in ['A', 'B', 'C']:
            continue

        total_questions += 1

        if truth_answer in option_support:
            option_support[truth_answer] += 1

        if pred_answer == truth_answer:
            correct_answers += 1
            option_tp[truth_answer] += 1
        else:
            option_fp[pred_answer] += 1
            option_fn[truth_answer] += 1

    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

    option_metrics = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    valid_classes = 0

    # （可选）Micro 统计：对多分类单标签任务，micro-F1 == accuracy（在仅统计有效样本时成立）
    micro_tp = micro_fp = micro_fn = 0

    for option in ['A', 'B', 'C']:
        tp = option_tp[option]
        fp = option_fp[option]
        fn = option_fn[option]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        option_metrics[f"{option} Precision"] = precision
        option_metrics[f"{option} Recall"] = recall
        option_metrics[f"{option} F1"] = f1

        if option_support[option] > 0:
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            valid_classes += 1

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_precision = macro_precision / valid_classes if valid_classes > 0 else 0.0
    macro_recall = macro_recall / valid_classes if valid_classes > 0 else 0.0
    macro_f1 = macro_f1 / valid_classes if valid_classes > 0 else 0.0

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    result = {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1
    }
    result.update(option_metrics)
    return result

def compute_ynqa_answer_metrics(model_name: str, predictions_path: str, ground_truth_path: str, qa_type) -> dict:
    """
    计算Yes/No QA类型问题的评估指标
    
    Args:
        model_name: 模型名称
        predictions_path: 模型预测结果文件路径
        ground_truth_path: 真实答案文件路径
    
    Returns:
        包含评估指标的字典
    """
    # 加载预测结果和真实答案
    predictions = load_json_data(predictions_path)
    ground_truth = load_json_data(ground_truth_path)
    
    # 计算评估指标
    metrics = evaluate_ynqa_answer(predictions, ground_truth, qa_type)
    
    # 添加模型信息
    metrics["model_name"] = model_name
    
    return metrics


def compute_mcqa_answer_metrics(model_name: str, predictions_path: str, ground_truth_path: str, qa_type) -> dict:
    """
    计算Multiple-Choice QA类型问题的评估指标
    
    Args:
        model_name: 模型名称
        predictions_path: 模型预测结果文件路径
        ground_truth_path: 真实答案文件路径
        qa_type: QA类型 (如 "m_mcqa")
    
    Returns:
        包含评估指标的字典
    """
    # 加载预测结果和真实答案
    predictions = load_json_data(predictions_path)
    ground_truth = load_json_data(ground_truth_path)
    
    # 计算评估指标
    metrics = evaluate_mcqa_answer(predictions, ground_truth, qa_type)
    
    # 添加模型信息
    metrics["model_name"] = model_name
    
    return metrics


def load_json_data(filepath: str) -> List[Dict]:
    """
    加载 JSON 数据文件
    
    Args:
        filepath: JSON 文件路径
    
    Returns:
        解析后的数据列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metrics(metrics: dict, output_path: str):
    """
    保存评估指标到文件
    
    Args:
        metrics: 评估指标字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Qwen3-VL-2B-Instruct
    # LLaVA-NeXT-Video-7B
    model_name = "InternVL3_5-30B-A3B"
    qa_type = "m_mcqa"
    predictions_dir = "./InternVL3_5-30B-A3B"
    ground_truth_dir = "../../video_real"
    
    # # Yes/No QA 文件路径
    # ynqa_prediction_file = os.path.join(predictions_dir, f"{qa_type}_{model_name}.json")
    # ynqa_ground_truth_file = os.path.join(ground_truth_dir, f"{qa_type}.json")
    # ynqa_metrics_output = os.path.join(predictions_dir, f"{qa_type}_metrics.json")
    
    # # 检查预测文件是否存在
    # if not os.path.exists(ynqa_prediction_file):
    #     print(f"预测文件不存在: {ynqa_prediction_file}")
    # else:
    #     # 计算Yes/No QA指标
    #     print("正在计算Yes/No QA评估指标...")
    #     ynqa_metrics = compute_ynqa_answer_metrics(model_name, ynqa_prediction_file, ynqa_ground_truth_file, qa_type)
        
    #     # 保存指标
    #     save_metrics(ynqa_metrics, ynqa_metrics_output)
    #     print(f"评估指标已保存到: {ynqa_metrics_output}")

    # Multiple-Choice QA 文件路径
    mcqa_prediction_file = os.path.join(predictions_dir, f"{qa_type}_{model_name}.json")
    mcqa_ground_truth_file = os.path.join(ground_truth_dir, f"{qa_type}.json")
    mcqa_metrics_output = os.path.join(predictions_dir, f"{qa_type}_metrics.json")
    
    # 检查预测文件是否存在
    if not os.path.exists(mcqa_prediction_file):
        print(f"预测文件不存在: {mcqa_prediction_file}")
    else:
        # 计算Multiple-Choice QA指标
        print(f"正在计算 {qa_type} 评估指标...")
        mcqa_metrics = compute_mcqa_answer_metrics(model_name, mcqa_prediction_file, mcqa_ground_truth_file, qa_type)
        
        # 保存指标
        save_metrics(mcqa_metrics, mcqa_metrics_output)
        print(f"{qa_type} 评估指标已保存到: {mcqa_metrics_output}")