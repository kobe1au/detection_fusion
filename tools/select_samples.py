#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import argparse
from pathlib import Path


def load_csv_sha256(csv_path: Path, column_name: str = "sha256") -> set[str]:
    """读取 CSV 中指定列的 sha256，统一转小写后放入集合"""
    sha256_set = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV 文件没有表头")

        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        if column_name.lower() not in field_map:
            raise ValueError(
                f"CSV 中未找到列: {column_name}，实际列名: {reader.fieldnames}"
            )

        real_col = field_map[column_name.lower()]

        for row in reader:
            value = (row.get(real_col) or "").strip().lower()
            if value:
                sha256_set.add(value)

    return sha256_set


def iter_pt_files(folder: Path, recursive: bool = False):
    """遍历文件夹中的 .pt 文件"""
    if recursive:
        for p in folder.rglob("*.pt"):
            if p.is_file():
                yield p
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() == ".pt":
                yield p


def get_filename_sha256(file_path: Path) -> str:
    """
    从文件名提取 sha256
    例如:
    abcdef.pt -> abcdef
    """
    return file_path.stem.strip().lower()


def main():
    parser = argparse.ArgumentParser(
        description="比较 CSV 的 sha256 列与文件夹中 sha256.pt 的文件名；存在则保留，不存在则删除"
    )
    parser.add_argument("--csv_path", help="CSV 文件路径")
    parser.add_argument("--folder_path", help="要检查的文件夹路径")
    parser.add_argument(
        "--column",
        default="sha256",
        help="CSV 中的哈希列名，默认: sha256"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将删除哪些文件，不实际删除"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    folder_path = Path(args.folder_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"文件夹不存在: {folder_path}")

    csv_hashes = load_csv_sha256(csv_path, args.column)
    print(f"[INFO] CSV 中读取到 {len(csv_hashes)} 个唯一 sha256")

    total_files = 0
    kept_files = 0
    deleted_files = 0
    error_files = 0
    deleting_files = 0

    for file_path in iter_pt_files(folder_path, recursive=args.recursive):
        total_files += 1
        try:
            file_hash = get_filename_sha256(file_path)

            if file_hash in csv_hashes:
                kept_files += 1
            else:
                if args.dry_run:
                    deleting_files += 1
                else:
                    file_path.unlink()
                    deleted_files += 1
                

        except Exception as e:
            error_files += 1
            print(f"[ERROR]  {file_path} -> {e}")

    print("\n===== 处理完成 =====")
    print(f"总文件数: {total_files}")
    print(f"保留文件: {kept_files}")
    print(f"要删除文件: {deleting_files}")
    print(f"删除文件: {deleted_files}")
    print(f"错误文件: {error_files}")


if __name__ == "__main__":
    main()