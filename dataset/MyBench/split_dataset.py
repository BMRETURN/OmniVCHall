import json
import os

def load_json_file(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, filepath):
    """保存JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_data_by_video_ids(data, video_ids, id_field):
    """根据视频ID列表提取数据"""
    extracted_data = []
    video_id_set = set(video_ids)  # 转换为集合以提高查找效率
    
    for item in data:
        if item['video_id'] in video_id_set:
            extracted_data.append(item)
    
    # 更新ID字段，确保连续性
    for i, item in enumerate(extracted_data):
        item[id_field] = i + 1
    
    return extracted_data

def main():
    # 加载数据集划分文件
    split_file = 'dataset_split_detailed.json'
    with open(split_file, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    # 获取训练集、验证集和测试集的视频ID列表
    train_videos = split_data['total']['train']
    val_videos = split_data['total']['val']
    test_videos = split_data['total']['test']
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    # 定义源文件和目标文件路径
    source_dirs = [
        '/home/lthpc/student/xwb/Mybenchmark/video_generated',
        '/home/lthpc/student/xwb/Mybenchmark/video_real'
    ]
    
    target_dir = '/home/lthpc/student/xwb/Mybenchmark/dataset'
    os.makedirs(target_dir, exist_ok=True)
    
    # 定义要处理的文件
    files_to_process = [
        ('caption.json', 'caption_id'),
        ('s_mcqa.json', 's_mcqa_id'),
        ('s_ynqa.json', 's_ynqa_id'),
        ('m_mcqa.json', 'm_mcqa_id'),
        ('m_ynqa.json', 'm_ynqa_id')
    ]
    
    # 对每个数据集划分（训练集、验证集、测试集）进行处理
    splits = {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }
    
    for split_name, video_ids in splits.items():
        print(f"\nProcessing {split_name} set...")
        
        # 合并来自两个源目录的数据
        for filename, id_field in files_to_process:
            merged_data = []
            
            # 从两个源目录加载数据
            for source_dir in source_dirs:
                file_path = os.path.join(source_dir, filename)
                if os.path.exists(file_path):
                    data = load_json_file(file_path)
                    merged_data.extend(data)
            
            # 根据视频ID提取对应的数据
            extracted_data = extract_data_by_video_ids(merged_data, video_ids, id_field)
            
            # 保存提取的数据
            output_file = os.path.join(target_dir, f'{split_name}_{filename}')
            save_json_file(extracted_data, output_file)
            
            print(f"  {filename}: {len(extracted_data)} items -> {output_file}")

if __name__ == "__main__":
    main()