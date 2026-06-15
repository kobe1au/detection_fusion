#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate complementarity between API-only and graph-only best checkpoints.

This focused wrapper answers whether the two single-modal baselines have enough
complementary signal to justify API+graph fusion.

Example:
    python test_m/single_modal_complementarity.py \
      --base config/base.yaml \
      --split test \
      --api-ckpt experiments/api_baseline/42/best_api_baseline.pt \
      --graph-ckpt experiments/gatv2_baseline/42/best_gatv2_baseline.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv

from complementarity import (  # noqa: E402
    ModelSpec,
    aligned_arrays,
    collect_predictions,
    load_checkpoint_config,
    load_yaml,
    metric_summary,
    pairwise_complementarity,
    resolve_existing_path,
    select_device,
    validate_full_config,
    write_csv,
)


DEFAULT_API_CKPT = "experiments/api_baseline/42/best_api_baseline.pt"
DEFAULT_GRAPH_CKPT = "experiments/gatv2_baseline/42/best_gatv2_baseline.pt"


def _case_rows(api_pack, graph_pack) -> list[dict[str, Any]]:
    sids, labels, pa, pg, proba_a, proba_g = aligned_arrays(api_pack, graph_pack)
    rows: list[dict[str, Any]] = []
    for idx, sid in enumerate(sids):
        ca = bool(pa[idx] == labels[idx])
        cg = bool(pg[idx] == labels[idx])
        if ca and cg:
            case = "both_correct"
        elif ca and not cg:
            case = "api_only_correct"
        elif (not ca) and cg:
            case = "graph_only_correct"
        else:
            case = "both_wrong"
        rows.append({
            "sid": sid,
            "label": int(labels[idx]),
            "api_pred": int(pa[idx]),
            "graph_pred": int(pg[idx]),
            "case": case,
            "api_conf": float(proba_a[idx].max()),
            "graph_conf": float(proba_g[idx].max()),
            "api_prob_1": float(proba_a[idx, 1]) if proba_a.shape[1] > 1 else 0.0,
            "graph_prob_1": float(proba_g[idx, 1]) if proba_g.shape[1] > 1 else 0.0,
        })
    return rows


def _interpret(pair: dict[str, Any]) -> str:
    gain = float(pair.get("oracle_gain_over_best_f1", 0.0))
    err_j = float(pair.get("error_jaccard", 0.0))
    ens_gain = float(pair.get("best_prob_ensemble_f1", 0.0)) - max(
        float(pair.get("a_f1", 0.0)),
        float(pair.get("b_f1", 0.0)),
    )

    if gain < 0.005:
        signal = "互补性很弱：API 与 graph 大多错在同一批样本上，融合 clean F1 很难明显超过单模态。"
    elif gain < 0.015:
        signal = "互补性有限：融合可能有小幅收益，但需要 gate/加权融合足够稳定。"
    else:
        signal = "互补性较明显：如果融合模型没涨，重点排查融合机制是否学会选择可靠模态。"

    if err_j > 0.65:
        overlap = "错误重合度偏高。"
    elif err_j < 0.45:
        overlap = "错误重合度不高，有一定互补空间。"
    else:
        overlap = "错误重合度中等。"

    if ens_gain > 0.003:
        ensemble = "简单概率加权 ensemble 已经有收益，可以考虑 graph-heavy/API-heavy late fusion 或可学习 late fusion。"
    else:
        ensemble = "简单概率加权 ensemble 收益不明显，说明互补信号不容易被线性融合利用。"

    return f"{signal} {overlap} {ensemble}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate complementarity between API-only and graph-only best.pt"
    )
    parser.add_argument("--base", default="config/base.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--api-ckpt", default=DEFAULT_API_CKPT)
    parser.add_argument("--graph-ckpt", default=DEFAULT_GRAPH_CKPT)
    parser.add_argument("--out-dir", default="test_m/results/single_modal_complementarity")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    data_root = os.getenv("DATA_ROOT", ".")
    base_cfg = validate_full_config(load_yaml(args.base))
    batch_size = int(args.batch_size or base_cfg["train"].get("eval_batch_size", 8))

    api_ckpt = resolve_existing_path(data_root, args.api_ckpt)
    graph_ckpt = resolve_existing_path(data_root, args.graph_ckpt)
    api_cfg = load_checkpoint_config(base_cfg, api_ckpt)
    graph_cfg = load_checkpoint_config(base_cfg, graph_ckpt)

    specs = [
        ModelSpec(name="api", ckpt_path=api_ckpt, cfg=api_cfg),
        ModelSpec(name="graph", ckpt_path=graph_ckpt, cfg=graph_cfg),
    ]

    device = select_device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.split}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] DATA_ROOT={data_root}")
    print(f"[info] split={args.split} device={device} amp={use_amp}")
    print(f"[info] batch_size={batch_size} num_workers={args.num_workers}")
    print(f"[info] api_ckpt={api_ckpt}")
    print(f"[info] graph_ckpt={graph_ckpt}")
    print(f"[info] output={out_dir}")

    packs = [
        collect_predictions(
            spec,
            data_root=data_root,
            split=args.split,
            device=device,
            batch_size=batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
        )
        for spec in specs
    ]

    api_pack, graph_pack = packs
    model_rows = [metric_summary(api_pack), metric_summary(graph_pack)]
    pair = pairwise_complementarity(api_pack, graph_pack)
    case_rows = _case_rows(api_pack, graph_pack)

    write_csv(out_dir / "model_metrics.csv", model_rows)
    write_csv(out_dir / "single_modal_complementarity.csv", [pair])
    write_csv(out_dir / "sample_cases.csv", case_rows)
    write_csv(out_dir / "predictions_api.csv", api_pack.rows)
    write_csv(out_dir / "predictions_graph.csv", graph_pack.rows)

    summary = {
        "split": args.split,
        "api_ckpt": api_ckpt,
        "graph_ckpt": graph_ckpt,
        "model_metrics": model_rows,
        "complementarity": pair,
        "interpretation": _interpret(pair),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cases = {k: 0 for k in ("both_correct", "api_only_correct", "graph_only_correct", "both_wrong")}
    for row in case_rows:
        cases[row["case"]] += 1
    n = max(len(case_rows), 1)

    print("\n=== Single-modal metrics ===")
    for row in model_rows:
        print(
            f"{row['name']:8s} mode={row['fusion_mode']:8s} n={row['n']:5d} "
            f"F1={row['macro_f1']:.4f} Acc={row['accuracy']:.4f}"
        )

    print("\n=== Complementarity ===")
    print(f"both_correct         = {cases['both_correct']:5d} ({cases['both_correct']/n:.3f})")
    print(f"api_only_correct     = {cases['api_only_correct']:5d} ({cases['api_only_correct']/n:.3f})")
    print(f"graph_only_correct   = {cases['graph_only_correct']:5d} ({cases['graph_only_correct']/n:.3f})")
    print(f"both_wrong           = {cases['both_wrong']:5d} ({cases['both_wrong']/n:.3f})")
    print(f"disagreement_rate    = {pair['disagreement_rate']:.4f}")
    print(f"error_jaccard        = {pair['error_jaccard']:.4f}")
    print(f"oracle_f1            = {pair['oracle_f1']:.4f}")
    print(f"oracle_gain_over_best= {pair['oracle_gain_over_best_f1']:+.4f}")
    print(f"best_ensemble_f1     = {pair['best_prob_ensemble_f1']:.4f} @ api_weight={pair['best_weight_for_a']:.1f}")
    print("\n=== Interpretation ===")
    print(_interpret(pair))
    print(f"\n[done] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
