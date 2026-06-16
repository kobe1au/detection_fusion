from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


BRANCHES = ("api", "graph", "manifest", "joint")

SCALAR_TYPES = (str, int, float, bool, type(None))

ATTENTION_KEYS = (
    "mean_semantic_attention_entropy",
    "mean_cross_modal_attention",
    "semantic_residual_gate",
    "mean_semantic_reliability_prior_api",
    "mean_semantic_reliability_prior_graph",
    "mean_semantic_reliability_prior_manifest",
    "mean_semantic_attention_to_api",
    "mean_semantic_attention_to_graph",
    "mean_semantic_attention_to_manifest",
    "macro_f1",
    "acc",
    "auc",
    "ap",
)

SELECTIVE_KEYS = (
    "coverage",
    "selective_metrics_defined",
    "selective_risk",
    "selective_acc",
    "selective_macro_f1",
    "aurc",
    "acceptance_score_mean",
    "acceptance_score_p10",
    "acceptance_score_p50",
    "acceptance_score_p90",
    "macro_f1",
    "acc",
    "auc",
    "ap",
    "ece_10",
    "brier",
)


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _safe_load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _run_identity(summary_path: Path, results_root: Path) -> dict[str, str]:
    try:
        rel = summary_path.parent.relative_to(results_root)
        parts = rel.parts
    except ValueError:
        parts = summary_path.parent.parts
    experiment = parts[-2] if len(parts) >= 2 else summary_path.parent.name
    seed = parts[-1] if parts else ""
    return {
        "experiment": experiment,
        "seed": seed,
        "run_dir": str(summary_path.parent),
        "summary_path": str(summary_path),
    }


def _scalar_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {
        str(key): value
        for key, value in metrics.items()
        if isinstance(value, SCALAR_TYPES)
    }


def _add_metric_row(
    rows: list[dict[str, Any]],
    identity: dict[str, str],
    section: str,
    scenario: str,
    metrics: Any,
) -> None:
    scalars = _scalar_metrics(metrics)
    if not scalars:
        return
    rows.append(
        {
            **identity,
            "section": section,
            "scenario": scenario,
            **scalars,
        }
    )


def _summary_metric_rows(summary_path: Path, results_root: Path) -> list[dict[str, Any]]:
    summary = _safe_load_yaml(summary_path)
    identity = _run_identity(summary_path, results_root)
    rows: list[dict[str, Any]] = []
    for section in ("val_selection", "val_calibration", "test"):
        _add_metric_row(rows, identity, section, section, summary.get(section))
    for section in ("val_robust", "robust"):
        nested = summary.get(section) or {}
        if not isinstance(nested, dict):
            continue
        for scenario, metrics in nested.items():
            _add_metric_row(rows, identity, section, str(scenario), metrics)
    extra_eval = summary.get("extra_eval") or {}
    if isinstance(extra_eval, dict):
        for scenario, metrics in extra_eval.items():
            _add_metric_row(rows, identity, "extra_eval", str(scenario), metrics)
    extra_json = summary_path.parent / "metrics_extra_eval.json"
    if extra_json.exists():
        for scenario, metrics in _safe_load_json(extra_json).items():
            _add_metric_row(rows, identity, "extra_eval_json", str(scenario), metrics)
    return rows


def collect_metric_rows(results_root: Path) -> pd.DataFrame:
    summary_paths = sorted(results_root.rglob("summary.yaml"))
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        rows.extend(_summary_metric_rows(path, results_root))
    return pd.DataFrame.from_records(rows)


def _branch_reliability_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame.from_records(records)
    base_columns = ["experiment", "seed", "run_dir", "section", "scenario"]
    for row in metrics.to_dict("records"):
        for branch in BRANCHES:
            prefix = f"{branch}_reliability_"
            branch_record = {
                key: row.get(key)
                for key in base_columns
                if key in row
            }
            branch_record["branch"] = branch
            copied = False
            for key, value in row.items():
                if key.startswith(prefix):
                    branch_record[key[len(prefix):]] = value
                    copied = True
            accuracy_key = f"{branch}_branch_accuracy"
            if accuracy_key in row:
                branch_record["branch_accuracy"] = row.get(accuracy_key)
                copied = True
            if copied:
                records.append(branch_record)
    return pd.DataFrame.from_records(records)


def _selected_columns(metrics: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    base = ["experiment", "seed", "run_dir", "section", "scenario"]
    selected = [key for key in [*base, *keys] if key in metrics.columns]
    frame = metrics[selected].copy()
    metric_keys = [key for key in keys if key in frame.columns]
    if metric_keys:
        frame = frame[frame[metric_keys].notna().any(axis=1)]
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect robust tri-modal experiment summaries into paper tables."
    )
    parser.add_argument("--results-root", default="results/tri_modal_robust")
    parser.add_argument("--out-dir", default="tables")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    metrics = collect_metric_rows(results_root)
    _write_csv(metrics, out_dir / "main_results.csv")
    _write_csv(_branch_reliability_rows(metrics), out_dir / "i1_reliability_calibration_summary.csv")
    _write_csv(_selected_columns(metrics, ATTENTION_KEYS), out_dir / "i2_attention_ablation.csv")
    _write_csv(_selected_columns(metrics, SELECTIVE_KEYS), out_dir / "i3_selective_results.csv")
    run_index_columns = [
        key
        for key in ("experiment", "seed", "run_dir", "summary_path")
        if key in metrics.columns
    ]
    run_index = metrics[run_index_columns].drop_duplicates() if run_index_columns else pd.DataFrame()
    _write_csv(run_index, out_dir / "summary_index.csv")


if __name__ == "__main__":
    main()
