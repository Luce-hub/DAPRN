
import os
import pandas as pd
import numpy as np

# 设置路径
base_input_path = r'C:\Users\PC\Desktop\第一篇数据\轴承T_3\model0\train'
output_base_folder = r'C:\Users\PC\Desktop\覆盖率50%'

def extract_samples(df, sample_length=1024, step=512):
    """
    提取固定长度的样本，按指定步长滑动。
    """
    samples = []
    for start in range(0, len(df), step):
        end = start + sample_length
        if end > len(df):
            break
        samples.append(df.iloc[start:end])
    return samples

# 主程序逻辑
if __name__ == "__main__":
    total_samples = 0  # 总样本数统计

    for class_index in range(10):  # 遍历 class0.xlsx ~ class9.xlsx
        input_file = os.path.join(base_input_path, f'class{class_index}.xlsx')

        print(f"正在处理: {input_file}")
        try:
            df = pd.read_excel(input_file)
        except Exception as e:
            print(f"❌ 无法读取文件 {input_file}，错误：{e}")
            continue

        # 创建该类别专属文件夹
        class_output_folder = os.path.join(output_base_folder, f'class_{class_index}')
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)

        # 提取样本
        samples = extract_samples(df, sample_length=1024, step=512)
        print(f"  提取到 {len(samples)} 个样本")

        # 保存样本
        for i, sample in enumerate(samples):
            output_file = os.path.join(class_output_folder, f'class_{class_index}_sample_{i}.xlsx')
            sample.to_excel(output_file, index=False)
            total_samples += 1

    print(f"✅ 所有样本提取完成，共生成 {total_samples} 个样本。")
    print(f"已保存至：{output_base_folder}")

