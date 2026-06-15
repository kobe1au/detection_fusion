#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd


def collect_existing_stems(data_dir: str) -> set[str]:
    """
    收集目录下所有文件的“主文件名”（不带扩展名），
    例如:
      abc123.pt   -> abc123
      deadbeef    -> deadbeef
    """
    existing = set()
    for root, _, files in os.walk(data_dir):
        for fname in files:
            stem = Path(fname).stem
            existing.add(stem)
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="根据文件夹中的实际文件，过滤 CSV 中不存在对应 sha256 文件的行。"
    )
    parser.add_argument("--csv", default="/root/autodl-tmp/result/labels/train.csv", help="输入 CSV 路径")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/all_samples/train", help="存放样本文件的目录")
    parser.add_argument("--out_csv", default="/root/autodl-tmp/result/new/train.csv", help="输出清洗后的 CSV 路径")
    parser.add_argument(
        "--sha_col",
        default="sha256",
        help="CSV 中 sha256 列名，默认是 sha256",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    sha_col = args.sha_col

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    df = pd.read_csv(csv_path)

    if sha_col not in df.columns:
        raise ValueError(f"CSV 中不存在列: {sha_col}")

    df[sha_col] = df[sha_col].astype(str).str.strip().str.lower()

    existing_sha = collect_existing_stems(str(data_dir))

    before = len(df)
    df_clean = df[df[sha_col].isin(existing_sha)].copy()
    after = len(df_clean)
    removed = before - after

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"输入 CSV 行数: {before}")
    print(f"保留行数: {after}")
    print(f"删除行数: {removed}")
    print(f"输出文件: {out_csv}")


if __name__ == "__main__":
    main()