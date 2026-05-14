import os
import json
from typing import Dict, List
from collections import defaultdict


def evaluate_ynqa_answer_by_type(predictions: List[Dict], ground_truth: List[Dict], qa_type) -> Dict[str, float]:
    """
    按 type_id 分类评估 Yes/No QA 类型问题的回答
    """
    # 创建真实答案的映射字典，包含 type_id 信息
    truth_dict = {}
    for item in ground_truth:
        truth_dict[item[f'{qa_type}_id']] = {
            'answer': item['yn_answer'].lower(),
            'type_id': item['type_id']
        }
    
    # 初始化各类别的计数器
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    
    # 遍历预测结果
    for pred in predictions:
        qa_id = pred[f'{qa_type}_id']
        pred_answer = pred['answer'].lower().strip()
        
        # 确保预测的问题ID在真实答案中存在
        if qa_id not in truth_dict:
            continue
            
        truth_info = truth_dict[qa_id]
        truth_answer = truth_info['answer']
        type_id = truth_info['type_id']
        
        # 只考虑预测答案为 yes 或 no 的情况
        if pred_answer not in ['yes', 'no']:
            continue
            
        # 将 type_id 转换为列表以便统一处理
        type_list = [type_id] if isinstance(type_id, int) else type_id
        
        # 更新各类别的统计
        for t in type_list:
            type_total[t] += 1
            if pred_answer == truth_answer:
                type_correct[t] += 1
    
    # 计算各类别的准确率
    results = {}
    for type_num in range(1, 9):  # 1-8 类型
        total = type_total[type_num]
        correct = type_correct[type_num]
        accuracy = correct / total if total > 0 else 0.0
        results[f'type_{type_num}_correct'] = correct
        results[f'type_{type_num}_total'] = total
        results[f'type_{type_num}_accuracy'] = accuracy
    
    # 计算总体指标
    overall_total = sum(type_total.values())
    overall_correct = sum(type_correct.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    
    results['overall_correct'] = overall_correct
    results['overall_total'] = overall_total
    results['overall_accuracy'] = overall_accuracy
    
    return results


def evaluate_mcqa_answer_by_type(predictions: List[Dict], ground_truth: List[Dict], qa_type) -> Dict[str, float]:
    """
    按 type_id 分类评估 Multiple-Choice QA 类型问题的回答
    """
    # 创建真实答案的映射字典，包含 type_id 信息
    truth_dict = {}
    for item in ground_truth:
        truth_dict[item[f'{qa_type}_id']] = {
            'answer': item['mc_answer'].upper(),
            'type_id': item['type_id']
        }

    # 初始化各类别的计数器
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    for pred in predictions:
        qa_id = pred.get(f'{qa_type}_id')
        if qa_id not in truth_dict:
            continue

        pred_answer = str(pred.get('answer', '')).upper().strip()
        truth_info = truth_dict[qa_id]
        truth_answer = truth_info['answer']
        type_id = truth_info['type_id']

        if pred_answer not in ['A', 'B', 'C']:
            continue

        # 将 type_id 转换为列表以便统一处理
        type_list = [type_id] if isinstance(type_id, int) else type_id
        
        # 更新各类别的统计
        for t in type_list:
            type_total[t] += 1
            if pred_answer == truth_answer:
                type_correct[t] += 1

    # 计算各类别的指标
    results = {}
    for type_num in range(1, 9):  # 1-8 类型
        total = type_total[type_num]
        correct = type_correct[type_num]
        accuracy = correct / total if total > 0 else 0.0
        results[f'type_{type_num}_correct'] = correct
        results[f'type_{type_num}_total'] = total
        results[f'type_{type_num}_accuracy'] = accuracy
    
    # 计算总体指标
    overall_total = sum(type_total.values())
    overall_correct = sum(type_correct.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    
    results['overall_correct'] = overall_correct
    results['overall_total'] = overall_total
    results['overall_accuracy'] = overall_accuracy

    return results


def compute_ynqa_answer_metrics_by_type(model_name: str, predictions_path: str, ground_truth_path: str, qa_type) -> dict:
    """
    按 type_id 分类计算Yes/No QA类型问题的评估指标
    """
    # 加载预测结果和真实答案
    predictions = load_json_data(predictions_path)
    ground_truth = load_json_data(ground_truth_path)
    
    # 计算评估指标
    metrics = evaluate_ynqa_answer_by_type(predictions, ground_truth, qa_type)
    
    # 添加模型信息
    metrics["model_name"] = model_name
    metrics["qa_type"] = qa_type
    
    return metrics


def compute_mcqa_answer_metrics_by_type(model_name: str, predictions_path: str, ground_truth_path: str, qa_type) -> dict:
    """
    按 type_id 分类计算Multiple-Choice QA类型问题的评估指标
    """
    # 加载预测结果和真实答案
    predictions = load_json_data(predictions_path)
    ground_truth = load_json_data(ground_truth_path)
    
    # 计算评估指标
    metrics = evaluate_mcqa_answer_by_type(predictions, ground_truth, qa_type)
    
    # 添加模型信息
    metrics["model_name"] = model_name
    metrics["qa_type"] = qa_type
    
    return metrics


def load_json_data(filepath: str) -> List[Dict]:
    """
    加载 JSON 数据文件
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metrics(metrics: dict, output_path: str):
    """
    保存评估指标到文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def print_detailed_results(metrics: dict):
    """
    打印详细的按类型统计结果
    """
    print(f"\n模型: {metrics['model_name']}, 问题类型: {metrics['qa_type']}")
    print(f"总体准确率: {metrics['overall_accuracy']:.4f} ({metrics['overall_correct']}/{metrics['overall_total']})")
    print("\n按类型统计:")
    for i in range(1, 9):
        correct = metrics[f'type_{i}_correct']
        total = metrics[f'type_{i}_total']
        accuracy = metrics[f'type_{i}_accuracy']
        print(f"  类型 {i}: 准确率={accuracy:.4f}, 正确={correct}, 总数={total}")


if __name__ == "__main__":
    model_name = "InternVL3_5-30B-A3B"
    predictions_dir = "./InternVL3_5-30B-A3B"
    ground_truth_dir = "../../video_generated"
    
    # 定义所有需要评估的 QA 类型
    qa_types = ["s_ynqa", "s_mcqa", "m_ynqa", "m_mcqa"]
    
    all_metrics = {}
    
    for qa_type in qa_types:
        prediction_file = os.path.join(predictions_dir, f"{qa_type}_{model_name}.json")
        ground_truth_file = os.path.join(ground_truth_dir, f"{qa_type}.json")
        metrics_output = os.path.join(predictions_dir, f"{qa_type}_metrics_by_type.json")
        
        # 检查预测文件是否存在
        if not os.path.exists(prediction_file):
            print(f"预测文件不存在: {prediction_file}")
            continue
        
        print(f"正在计算 {qa_type} 评估指标...")
        
        # 根据 QA 类型选择相应的评估函数
        if qa_type.endswith('_ynqa'):
            metrics = compute_ynqa_answer_metrics_by_type(
                model_name, prediction_file, ground_truth_file, qa_type
            )
        else:  # _mcqa
            metrics = compute_mcqa_answer_metrics_by_type(
                model_name, prediction_file, ground_truth_file, qa_type
            )
        
        # 保存指标
        save_metrics(metrics, metrics_output)
        print(f"{qa_type} 评估指标已保存到: {metrics_output}")
        
        # 打印详细结果
        print_detailed_results(metrics)
        
        all_metrics[qa_type] = metrics
    
    # 保存所有指标到一个文件
    all_metrics_output = os.path.join(predictions_dir, "all_metrics_by_type.json")
    save_metrics(all_metrics, all_metrics_output)
    print(f"\n所有评估指标已汇总保存到: {all_metrics_output}")