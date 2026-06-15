#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_pt_distribution.py

统计 processed .pt 数据集中每个 APK 样本的：
- dex 数
- total_api_events
- mean_api_events_per_dex
- max_api_events_single_dex
- total_nodes
- total_edges
- max_nodes_single_dex
- complexity = total_api_events * total_nodes

并输出：
1. 终端摘要统计（mean / median / p90 / p95 / p99 / max）
2. 详细样本级 CSV
3. split 级别统计 CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def summarize_array(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def load_one_pt(pt_path: Path) -> Dict[str, Any]:
    obj = torch.load(pt_path, map_location="cpu")
    dex_list = obj if isinstance(obj, list) else [obj]

    num_dex = 0
    total_api_events = 0
    total_nodes = 0
    total_edges = 0
    max_api_events_single_dex = 0
    max_nodes_single_dex = 0
    api_events_per_dex: List[int] = []
    nodes_per_dex: List[int] = []
    edges_per_dex: List[int] = []

    for dex in dex_list:
        if not isinstance(dex, dict):
            continue
        num_dex += 1

        call_x = dex.get("call_x", dex.get("cfg_x", None))
        edge_index = dex.get("call_edge_index", dex.get("cfg_edge_index", None))
        api_ids = dex.get("api_ids", None)

        num_api_events = safe_int(getattr(api_ids, "shape", [0])[0] if api_ids is not None else 0)
        num_nodes = safe_int(getattr(call_x, "shape", [0])[0] if call_x is not None else 0)
        num_edges = safe_int(getattr(edge_index, "shape", [0, 0])[1] if edge_index is not None else 0)

        api_events_per_dex.append(num_api_events)
        nodes_per_dex.append(num_nodes)
        edges_per_dex.append(num_edges)

        total_api_events += num_api_events
        total_nodes += num_nodes
        total_edges += num_edges
        max_api_events_single_dex = max(max_api_events_single_dex, num_api_events)
        max_nodes_single_dex = max(max_nodes_single_dex, num_nodes)

    mean_api_events_per_dex = float(np.mean(api_events_per_dex)) if api_events_per_dex else 0.0
    mean_nodes_per_dex = float(np.mean(nodes_per_dex)) if nodes_per_dex else 0.0
    mean_edges_per_dex = float(np.mean(edges_per_dex)) if edges_per_dex else 0.0

    complexity = int(total_api_events * total_nodes)
    max_dex_complexity = 0
    if api_events_per_dex and nodes_per_dex:
        max_dex_complexity = int(max(a * n for a, n in zip(api_events_per_dex, nodes_per_dex)))

    return {
        "sid": pt_path.stem,
        "num_dex": num_dex,
        "total_api_events": total_api_events,
        "mean_api_events_per_dex": mean_api_events_per_dex,
        "max_api_events_single_dex": max_api_events_single_dex,
        "total_nodes": total_nodes,
        "mean_nodes_per_dex": mean_nodes_per_dex,
        "max_nodes_single_dex": max_nodes_single_dex,
        "total_edges": total_edges,
        "mean_edges_per_dex": mean_edges_per_dex,
        "complexity": complexity,
        "max_dex_complexity": max_dex_complexity,
        "pt_path": str(pt_path),
    }


def collect_rows(pt_root: Path, splits: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in splits:
        split_dir = pt_root / split
        if not split_dir.exists():
            print(f"[WARN] split dir not found: {split_dir}")
            continue

        files = sorted(split_dir.rglob("*.pt"))
        print(f"[INFO] split={split} files={len(files)}")

        for pt_path in files:
            try:
                row = load_one_pt(pt_path)
                row["split"] = split
                rows.append(row)
            except Exception as e:
                rows.append({
                    "sid": pt_path.stem,
                    "split": split,
                    "num_dex": -1,
                    "total_api_events": -1,
                    "mean_api_events_per_dex": -1,
                    "max_api_events_single_dex": -1,
                    "total_nodes": -1,
                    "mean_nodes_per_dex": -1,
                    "max_nodes_single_dex": -1,
                    "total_edges": -1,
                    "mean_edges_per_dex": -1,
                    "complexity": -1,
                    "max_dex_complexity": -1,
                    "pt_path": str(pt_path),
                    "error": str(e),
                })
    return pd.DataFrame(rows)


def print_metric_summary(df: pd.DataFrame, metric: str) -> None:
    valid = df[df[metric] >= 0][metric].to_numpy()
    stats = summarize_array(valid)
    print(
        f"{metric:>22s} | "
        f"count={stats['count']:>6d} | "
        f"mean={stats['mean']:>12.2f} | "
        f"median={stats['median']:>12.2f} | "
        f"p90={stats['p90']:>12.2f} | "
        f"p95={stats['p95']:>12.2f} | "
        f"p99={stats['p99']:>12.2f} | "
        f"max={stats['max']:>12.2f}"
    )


def build_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "num_dex",
        "total_api_events",
        "mean_api_events_per_dex",
        "max_api_events_single_dex",
        "total_nodes",
        "mean_nodes_per_dex",
        "max_nodes_single_dex",
        "total_edges",
        "mean_edges_per_dex",
        "complexity",
        "max_dex_complexity",
    ]

    rows = []
    groups = [("ALL", df)] + list(df.groupby("split"))
    for split_name, subdf in groups:
        for metric in metrics:
            valid = subdf[subdf[metric] >= 0][metric].to_numpy()
            stats = summarize_array(valid)
            rows.append({
                "split": split_name,
                "metric": metric,
                **stats,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_root", type=str, required=True, help="根目录，内部含 train/val/test 等子目录")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out_csv", type=str, default="pt_distribution_detail.csv")
    parser.add_argument("--out_summary_csv", type=str, default="pt_distribution_summary.csv")
    parser.add_argument(
        "--sort_by",
        type=str,
        default="complexity",
        choices=[
            "complexity", "total_api_events", "total_nodes",
            "max_api_events_single_dex", "max_nodes_single_dex", "num_dex"
        ],
    )
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    pt_root = Path(args.pt_root)
    df = collect_rows(pt_root, args.splits)
    if df.empty:
        print("No .pt files found.")
        return

    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\n[OK] detail csv -> {args.out_csv}")

    summary_df = build_summary_df(df)
    summary_df.to_csv(args.out_summary_csv, index=False, encoding="utf-8")
    print(f"[OK] summary csv -> {args.out_summary_csv}")

    print("\n========== GLOBAL SUMMARY ==========")
    for metric in [
        "num_dex",
        "total_api_events",
        "mean_api_events_per_dex",
        "max_api_events_single_dex",
        "total_nodes",
        "mean_nodes_per_dex",
        "max_nodes_single_dex",
        "total_edges",
        "complexity",
        "max_dex_complexity",
    ]:
        print_metric_summary(df, metric)

    print(f"\n========== TOP-{args.topk} BY {args.sort_by} ==========")
    show_cols = [
        "split", "sid", "num_dex",
        "total_api_events", "max_api_events_single_dex",
        "total_nodes", "max_nodes_single_dex",
        "total_edges", "complexity", "max_dex_complexity",
    ]
    ranked = df[df[args.sort_by] >= 0].sort_values(args.sort_by, ascending=False).head(args.topk)
    print(ranked[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
