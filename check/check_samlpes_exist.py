import pandas as pd
import os

def filter_missing_hashes(csv_file, txt_file, output_file, column_name='sha256'):
    """
    比较 CSV 中的 sha256 值是否在 TXT 文件中出现。
    如果没有出现，则将该行保存到新的 CSV 中。
    """
    
    # 1. 读取 TXT 文件，存入集合(set)以实现 O(1) 的极速查找
    # 将所有值转为小写并去除首尾空格
    print(f"正在读取文本文件: {txt_file}...")
    if not os.path.exists(txt_file):
        print("错误：找不到 TXT 文件")
        return

    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_hashes = {line.strip().lower() for line in f if line.strip()}

    # 2. 读取 CSV 文件
    print(f"正在读取 CSV 文件: {csv_file}...")
    if not os.path.exists(csv_file):
        print("错误：找不到 CSV 文件")
        return

    df = pd.read_csv(csv_file)

    # 检查列是否存在
    if column_name not in df.columns:
        print(f"错误：CSV 中未找到列名 '{column_name}'")
        print(f"现有列名为: {list(df.columns)}")
        return

    # 3. 执行过滤逻辑
    # 逻辑：判断 sha256 列的值(lower) 是否 不在(not in) txt_hashes 集合中
    print("正在比对数据...")
    missing_mask = df[column_name].astype(str).str.lower().apply(lambda x: x not in txt_hashes)
    
    df_missing = df[missing_mask]

    # 4. 保存结果
    if not df_missing.empty:
        df_missing.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"处理完成！")
        print(f"共有 {len(df)} 条记录，其中 {len(df_missing)} 条在 TXT 中未找到。")
        print(f"结果已保存至: {output_file}")
    else:
        print("比对完毕：CSV 中的所有哈希值均已在 TXT 中找到，无需生成新文件。")

if __name__ == "__main__":
    # 配置你的文件名
    INPUT_CSV = '/Users/tsing/Downloads/code/detection/resource/dataset_split_relaxed/val.csv'      # 原始 CSV
    INPUT_TXT = '/Users/tsing/Downloads/code/detection/resource/all.txt'    # 参考 TXT
    OUTPUT_CSV = 'missing_val.csv'  # 结果输出
    SHA_COLUMN = 'sha256'       # CSV 中哈希值所在的列名

    filter_missing_hashes(INPUT_CSV, INPUT_TXT, OUTPUT_CSV, SHA_COLUMN)