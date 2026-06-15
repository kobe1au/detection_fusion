import os
import shutil
import pandas as pd

# ================= 配置区域 =================
# 1. 源文件所在的文件夹（所有文件当前所在的目录）
source_dir = "/root/autodl-tmp/pts_1/all"  # <--- 需要修改

# 2. 存放 train.csv, test.csv, val.csv 的文件夹路径
csv_dir = "/root/autodl-tmp/results/labels"             # <--- 需要修改

# 3. 目标主文件夹路径 (按你的要求已经写好)
target_base_dir = "/root/autodl-tmp/pts"

# 4. CSV 文件中，记录“文件名”的那一列的表头名称
filename_col = "sha256"                   # <--- 需要修改 (如 'image_id', 'file_name' 等)

# 5. 如果你的 CSV 里的文件名没有后缀 (例如只有 '0001' 而不是 '0001.pt')，请在此处填写后缀
# 如果 CSV 里已经包含后缀，请保持为空字符串 ""
file_extension = ".pt"                         # 例如 ".pt", ".jpg" 等
# ============================================

splits = ["train", "test", "val"]

def move_files():
    for split in splits:
        # 创建对应的目标文件夹 (如 /root/autodl-tmp/pts/train)
        target_dir = os.path.join(target_base_dir, split)
        os.makedirs(target_dir, exist_ok=True)
        
        # 拼接 CSV 文件路径
        csv_path = os.path.join(csv_dir, f"{split}.csv")
        
        # 检查 CSV 文件是否存在
        if not os.path.exists(csv_path):
            print(f"⚠️ 警告: 找不到 CSV 文件 -> {csv_path}，跳过该部分。")
            continue
            
        print(f"正在处理 {split} 集...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ 读取 {csv_path} 失败: {e}")
            continue
            
        if filename_col not in df.columns:
            print(f"❌ CSV 中找不到列名为 '{filename_col}' 的列，请检查你的 CSV 表头！")
            continue

        moved_count = 0
        missing_count = 0
        
        # 遍历 CSV 中的每一行文件名
        for raw_name in df[filename_col]:
            # 处理文件名和后缀
            file_name = str(raw_name) + file_extension
            
            src_path = os.path.join(source_dir, file_name)
            dst_path = os.path.join(target_dir, file_name)
            
            # 如果源文件存在，则执行 mv 操作
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                moved_count += 1
            else:
                # 目标文件夹可能已经存在该文件，或者源文件本来就缺失
                if not os.path.exists(dst_path):
                    missing_count += 1
                
        print(f"✅ [{split}] 处理完成！成功移动: {moved_count} 个，未找到: {missing_count} 个。")
        print("-" * 50)

if __name__ == "__main__":
    move_files()
    print("🎉 所有文件移动操作执行完毕！")