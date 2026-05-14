import os
import json
import random

def get_video_files(folder_path, subfolder_name=None):
    """
    获取指定文件夹内特定子文件夹的所有视频文件名（不含扩展名）
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    
    # 确定搜索路径
    if subfolder_name:
        search_path = os.path.join(folder_path, subfolder_name)
    else:
        search_path = folder_path
    
    if os.path.exists(search_path):
        for file in os.listdir(search_path):
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_files.append(os.path.splitext(file)[0])
    
    return sorted(video_files)

def split_dataset_random(video_list, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    随机按照指定比例划分数据集
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)
    
    total = len(video_list)
    if total == 0:
        return [], [], []
    
    # 创建副本并随机打乱
    shuffled_list = video_list.copy()
    random.shuffle(shuffled_list)
    
    # 计算各集合大小
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # 划分数据集
    train_set = sorted(shuffled_list[:train_size])  # 保持排序以便查看
    val_set = sorted(shuffled_list[train_size:train_size + val_size])
    test_set = sorted(shuffled_list[train_size + val_size:])
    
    return train_set, val_set, test_set

def save_split_info(split_info, output_file):
    """
    保存划分信息到JSON文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

def main():
    # 定义路径
    video_generated_path = '/home/lthpc/student/xwb/Mybenchmark/video_generated'
    video_real_path = '/home/lthpc/student/xwb/Mybenchmark/video_real'
    output_path = '/home/lthpc/student/xwb/Mybenchmark/dataset'
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 获取视频文件列表
    # 对于video_generated，查找视频文件夹
    generated_videos = []
    possible_generated_dirs = ['all_video', 'HailuoVideo', 'KlingVideo', 'HunyuanVideo']
    
    for subdir in possible_generated_dirs:
        subdir_path = os.path.join(video_generated_path, subdir)
        if os.path.exists(subdir_path):
            generated_videos.extend(get_video_files(video_generated_path, subdir))
    
    # 如果在子目录中没有找到，则在根目录查找
    if not generated_videos:
        generated_videos = get_video_files(video_generated_path)
    
    # 对于video_real，查找视频文件夹
    real_videos = []
    possible_real_dirs = ['all_video', 'allVideo', 'VidHalluc', 'Event', 'MHBench', 'HailuoVideo', 'KlingVideo', 'HunyuanVideo']
    
    for subdir in possible_real_dirs:
        subdir_path = os.path.join(video_real_path, subdir)
        if os.path.exists(subdir_path):
            real_videos.extend(get_video_files(video_real_path, subdir))
    
    # 如果在子目录中没有找到，则在根目录查找
    if not real_videos:
        real_videos = get_video_files(video_real_path)
    
    print(f"Generated videos count: {len(generated_videos)}")
    print(f"Real videos count: {len(real_videos)}")
    
    # 分别对两类视频进行随机划分
    gen_train, gen_val, gen_test = split_dataset_random(generated_videos, 0.7, 0.1, 0.2, seed=42)
    real_train, real_val, real_test = split_dataset_random(real_videos, 0.7, 0.1, 0.2, seed=42)
    
    # 组合总的训练集、验证集和测试集
    total_train = sorted(gen_train + real_train)
    total_val = sorted(gen_val + real_val)
    total_test = sorted(gen_test + real_test)
    
    # 保存划分结果
    split_info = {
        "generated": {
            "train": gen_train,
            "val": gen_val,
            "test": gen_test,
            "train_count": len(gen_train),
            "val_count": len(gen_val),
            "test_count": len(gen_test)
        },
        "real": {
            "train": real_train,
            "val": real_val,
            "test": real_test,
            "train_count": len(real_train),
            "val_count": len(real_val),
            "test_count": len(real_test)
        },
        "total": {
            "train": total_train,
            "val": total_val,
            "test": total_test,
            "train_count": len(total_train),
            "val_count": len(total_val),
            "test_count": len(total_test)
        }
    }
    
    # 保存详细信息
    save_split_info(split_info, os.path.join(output_path, 'dataset_split_detailed.json'))
    
    # 保存简单的文件名列表
    with open(os.path.join(output_path, 'train_videos.txt'), 'w') as f:
        for video in total_train:
            f.write(f"{video}\n")
    
    with open(os.path.join(output_path, 'val_videos.txt'), 'w') as f:
        for video in total_val:
            f.write(f"{video}\n")
    
    with open(os.path.join(output_path, 'test_videos.txt'), 'w') as f:
        for video in total_test:
            f.write(f"{video}\n")
    
    # 输出统计信息
    print("\n数据集划分完成！")
    print(f"Generated videos - Train: {len(gen_train)}, Val: {len(gen_val)}, Test: {len(gen_test)}")
    print(f"Real videos - Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")
    print(f"Total - Train: {len(total_train)}, Val: {len(total_val)}, Test: {len(total_test)}")
    print(f"\n划分文件已保存到: {output_path}")

if __name__ == "__main__":
    main()