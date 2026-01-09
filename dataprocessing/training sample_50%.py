
import os
import random
import shutil

# 输入和输出路径
base_folder = r'C:\Users\PC\Desktop\第一篇数据\轴承T_3\model3\覆盖率50%'
output_base = r'C:\Users\PC\Desktop\model3'

# 创建输出目录
os.makedirs(os.path.join(output_base, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'support'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'query'), exist_ok=True)

# 每类抽取样本数量
samples_per_class = 45
train_count = 15
support_count = 15
query_count = 15

if __name__ == "__main__":
    total_samples = {
        'train': 0,
        'support': 0,
        'query': 0
    }

    for class_id in range(10):
        class_folder = os.path.join(base_folder, f'class_{class_id}')
        if not os.path.exists(class_folder):
            print(f"❌ 类别 {class_id} 的文件夹不存在: {class_folder}")
            continue

        # 获取所有样本文件
        samples = [f for f in os.listdir(class_folder) if f.startswith(f'class_{class_id}_sample_') and f.endswith('.xlsx')]
        if len(samples) < samples_per_class:
            print(f"⚠️ 类别 {class_id} 的样本不足 {samples_per_class} 个，跳过处理。")
            continue

        # 随机选取30个样本
        selected_samples = random.sample(samples, samples_per_class)

        # 分成三部分
        train_samples = selected_samples[:train_count]
        support_samples = selected_samples[train_count:train_count + support_count]
        query_samples = selected_samples[train_count + support_count:]

        # 复制样本到对应目录
        def copy_samples(file_list, target_dir):
            for src_file in file_list:
                src_path = os.path.join(class_folder, src_file)
                dst_path = os.path.join(target_dir, src_file)
                shutil.copy(src_path, dst_path)
                total_samples[os.path.basename(target_dir)] += 1

        train_dir = os.path.join(output_base, 'train')
        support_dir = os.path.join(output_base, 'support')
        query_dir = os.path.join(output_base, 'query')

        copy_samples(train_samples, train_dir)
        copy_samples(support_samples, support_dir)
        copy_samples(query_samples, query_dir)

        print(f"✅ 类别 {class_id} 的样本已按比例复制完成")

    print("📊 总样本统计：")
    print(f"  训练集 (train): {total_samples['train']} 个")
    print(f"  支持集 (support): {total_samples['support']} 个")
    print(f"  查询集 (query): {total_samples['query']} 个")
    print(f"🎉 所有样本复制完成，保存在：{output_base}")