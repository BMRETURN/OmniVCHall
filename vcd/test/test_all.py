import os
import json
from typing import Dict, List
import numpy as np

def calculate_overall_accuracy():
    """
    计算整体的准确率，包括video_real和video_generated两个场景下的所有问题类型
    """
    models = ["Qwen3-VL-8B-Instruct"]
    scenarios = ["video_real", "video_generated"]
    question_types = ["s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"]
    
    overall_results = {}
    
    for model in models:
        overall_results[model] = {}
        
        for scenario in scenarios:
            overall_results[model][scenario] = {}
            
            total_correct = 0
            total_questions = 0
            
            for qtype in question_types:
                # 构建预测结果和真实答案的路径
                predictions_path = f"{scenario}/{model}/{qtype}_{model}.json"
                ground_truth_path = f"{scenario}/GT/{qtype}.json"
                
                if not os.path.exists(predictions_path):
                    print(f"警告: 预测文件不存在: {predictions_path}")
                    continue
                
                if not os.path.exists(ground_truth_path):
                    print(f"警告: 真实答案文件不存在: {ground_truth_path}")
                    continue
                
                # 加载数据
                predictions = load_json_data(predictions_path)
                ground_truth = load_json_data(ground_truth_path)
                
                # 计算准确率
                if qtype in ["s_ynqa", "m_ynqa"]:
                    accuracy_info = calculate_ynqa_accuracy(predictions, ground_truth, qtype)
                else:  # s_mcqa, m_mcqa
                    accuracy_info = calculate_mcqa_accuracy(predictions, ground_truth, qtype)
                
                # 保存当前问题类型的准确率
                overall_results[model][scenario][qtype] = accuracy_info
                
                # 累计总数
                total_correct += accuracy_info['correct']
                total_questions += accuracy_info['total']
                
                print(f"{model} - {scenario} - {qtype}: {accuracy_info['accuracy']*100:.2f}% "
                      f"({accuracy_info['correct']}/{accuracy_info['total']})")
            
            # 计算场景整体准确率
            overall_acc = total_correct / total_questions if total_questions > 0 else 0.0
            overall_results[model][scenario]['overall'] = {
                'accuracy': overall_acc,
                'correct': total_correct,
                'total': total_questions
            }
            
            print(f"\n{model} - {scenario} - 总体准确率: {overall_acc*100:.2f}% "
                  f"({total_correct}/{total_questions})")
            print("-" * 60)
    
    # 计算所有场景的整体准确率
    grand_total_correct = 0
    grand_total_questions = 0
    
    for model in models:
        for scenario in scenarios:
            if 'overall' in overall_results[model][scenario]:
                grand_total_correct += overall_results[model][scenario]['overall']['correct']
                grand_total_questions += overall_results[model][scenario]['overall']['total']
    
    overall_accuracy = grand_total_correct / grand_total_questions if grand_total_questions > 0 else 0.0
    
    print(f"\n{model} - 所有场景总体准确率: {overall_accuracy*100:.2f}% "
          f"({grand_total_correct}/{grand_total_questions})")
    
    return overall_results, overall_accuracy


def calculate_ynqa_accuracy(predictions: List[Dict], ground_truth: List[Dict], qa_type: str) -> Dict:
    """
    计算Yes/No问题的准确率
    """
    # 创建真实答案的映射字典
    truth_dict = {item[f'{qa_type}_id']: item['yn_answer'].lower() for item in ground_truth}
    
    total_questions = 0
    correct_answers = 0
    
    for pred in predictions:
        qa_id = pred[f'{qa_type}_id']
        pred_answer = str(pred['answer']).lower().strip()
        
        if qa_id not in truth_dict:
            continue
            
        truth_answer = truth_dict[qa_id]
        
        # 只考虑预测答案为 yes 或 no 的情况
        if pred_answer not in ['yes', 'no']:
            continue
            
        total_questions += 1
        
        if pred_answer == truth_answer:
            correct_answers += 1
    
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct_answers,
        'total': total_questions
    }


def calculate_mcqa_accuracy(predictions: List[Dict], ground_truth: List[Dict], qa_type: str) -> Dict:
    """
    计算Multiple Choice问题的准确率
    """
    # 创建真实答案的映射字典
    truth_dict = {item[f'{qa_type}_id']: item['mc_answer'].upper() for item in ground_truth}
    
    total_questions = 0
    correct_answers = 0
    
    for pred in predictions:
        qa_id = pred[f'{qa_type}_id']
        pred_answer = str(pred['answer']).upper().strip()
        
        if qa_id not in truth_dict:
            continue
            
        truth_answer = truth_dict[qa_id]
        
        # 只考虑预测答案为 A, B, C 的情况
        if pred_answer not in ['A', 'B', 'C']:
            continue
            
        total_questions += 1
        
        if pred_answer == truth_answer:
            correct_answers += 1
    
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct_answers,
        'total': total_questions
    }


def load_json_data(filepath: str) -> List[Dict]:
    """
    加载 JSON 数据文件
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_overall_metrics(results: Dict, overall_accuracy: float, output_path: str):
    """
    保存整体评估指标到文件
    """
    final_result = {
        "overall_accuracy": overall_accuracy,
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"整体评估指标已保存到: {output_path}")


def print_detailed_report(results: Dict, overall_accuracy: float):
    """
    打印详细报告
    """
    print("\n" + "="*80)
    print("详细测试报告")
    print("="*80)
    
    models = list(results.keys())
    scenarios = list(list(results.values())[0].keys())
    question_types = ["s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"]
    
    for model in models:
        print(f"\n模型: {model}")
        print("-" * 60)
        
        for scenario in scenarios:
            print(f"\n  场景: {scenario}")
            print(f"  {'问题类型':<12} {'准确率':<10} {'正确数/总数':<15}")
            print("  " + "-" * 40)
            
            total_correct = 0
            total_questions = 0
            
            for qtype in question_types:
                if qtype in results[model][scenario]:
                    acc_info = results[model][scenario][qtype]
                    accuracy_pct = acc_info['accuracy'] * 100
                    correct_total = f"{acc_info['correct']}/{acc_info['total']}"
                    print(f"  {qtype:<12} {accuracy_pct:>7.2f}%   {correct_total:<15}")
                    
                    total_correct += acc_info['correct']
                    total_questions += acc_info['total']
            
            # 显示场景总体统计
            if total_questions > 0:
                scenario_acc = total_correct / total_questions * 100
                print(f"  总体准确率: {scenario_acc:>7.2f}%   ({total_correct}/{total_questions})")
    
    print(f"\n整体准确率: {overall_accuracy*100:.2f}%")
    print("="*80)


if __name__ == "__main__":
    # 执行整体准确率计算
    results, overall_acc = calculate_overall_accuracy()
    
    # 打印详细报告
    print_detailed_report(results, overall_acc)
    
    # 保存结果
    save_overall_metrics(results, overall_acc, "overall_evaluation_metrics.json")
    
    # 另外，也可以分别计算各个维度的准确率
    print("\n" + "="*80)
    print("按问题类型汇总")
    print("="*80)
    
    # 按问题类型统计
    for qtype in ["s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"]:
        total_correct = 0
        total_questions = 0
        
        for model in ["Qwen3-VL-8B-Instruct"]:
            for scenario in ["video_real", "video_generated"]:
                if qtype in results[model][scenario]:
                    acc_info = results[model][scenario][qtype]
                    total_correct += acc_info['correct']
                    total_questions += acc_info['total']
        
        if total_questions > 0:
            type_accuracy = total_correct / total_questions
            print(f"{qtype}: {type_accuracy*100:.2f}% ({total_correct}/{total_questions})")
    
    print("\n" + "="*80)
    print("按场景汇总")
    print("="*80)
    
    # 按场景统计
    for scenario in ["video_real", "video_generated"]:
        total_correct = 0
        total_questions = 0
        
        for model in ["Qwen3-VL-8B-Instruct"]:
            for qtype in ["s_ynqa", "m_ynqa", "s_mcqa", "m_mcqa"]:
                if qtype in results[model][scenario]:
                    acc_info = results[model][scenario][qtype]
                    total_correct += acc_info['correct']
                    total_questions += acc_info['total']
        
        if total_questions > 0:
            scenario_accuracy = total_correct / total_questions
            print(f"{scenario}: {scenario_accuracy*100:.2f}% ({total_correct}/{total_questions})")