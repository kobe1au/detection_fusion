from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    "classification_threshold",
    "fixed_0_5_acc",
    "fixed_0_5_macro_f1",
    "fixed_0_5_f1_pos",
    "fixed_0_5_recall_pos",
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
    "conformal_alpha",
    "conformal_class_conditional",
    "conformal_acceptance_rate",
    "conformal_rejection_rate",
    "conformal_empty_set_rate",
    "conformal_ambiguous_set_rate",
    "conformal_benign_acceptance_rate",
    "conformal_malware_acceptance_rate",
    "conformal_malware_rejection_rate",
    "conformal_malware_fn_after_rejection",
    "conformal_accepted_malware_fn_rate",
    "conformal_malware_fn_count",
    "conformal_accepted_malware_count",
    "conformal_selective_risk",
    "conformal_selective_acc",
    "conformal_empirical_coverage_benign",
    "conformal_empirical_coverage_malware",
    "risk_control_threshold",
    "risk_control_risk_level",
    "risk_control_acceptance_rate",
    "risk_control_rejection_rate",
    "risk_control_selective_risk",
    "risk_control_selective_acc",
    "risk_control_malware_fn_rate_after_rejection",
    "risk_control_accepted_malware_fn_rate",
    "risk_control_malware_fn_count",
    "risk_control_accepted_malware_count",
    "risk_control_num_accepted",
    "risk_control_num_rejected",
    "risk_control_target_met_empirically",
)

AGGREGATE_GROUP_COLUMNS = (
    "method",
    "method_protocol_sha256",
    "method_implementation_sha256",
    "section",
    "scenario",
)

DEFAULT_AGGREGATE_METRICS = {
    "classification_threshold",
    "fixed_0_5_acc",
    "fixed_0_5_macro_f1",
    "fixed_0_5_f1_pos",
    "fixed_0_5_recall_pos",
    "acc",
    "f1",
    "macro_f1",
    "recall",
    "auc",
    "ap",
    "brier",
    "ece_10",
    "confidence_accuracy_gap",
    "coverage",
    "selective_risk",
    "selective_acc",
    "selective_macro_f1",
    "aurc",
    "acceptance_score_mean",
    "acceptance_score_p10",
    "acceptance_score_p50",
    "acceptance_score_p90",
    "conformal_alpha",
    "conformal_acceptance_rate",
    "conformal_rejection_rate",
    "conformal_empty_set_rate",
    "conformal_ambiguous_set_rate",
    "conformal_benign_acceptance_rate",
    "conformal_malware_acceptance_rate",
    "conformal_malware_rejection_rate",
    "conformal_malware_fn_after_rejection",
    "conformal_accepted_malware_fn_rate",
    "conformal_malware_fn_count",
    "conformal_accepted_malware_count",
    "conformal_selective_risk",
    "conformal_selective_acc",
    "conformal_empirical_coverage_benign",
    "conformal_empirical_coverage_malware",
    "risk_control_threshold",
    "risk_control_risk_level",
    "risk_control_acceptance_rate",
    "risk_control_rejection_rate",
    "risk_control_selective_risk",
    "risk_control_selective_acc",
    "risk_control_malware_fn_rate_after_rejection",
    "risk_control_accepted_malware_fn_rate",
    "risk_control_malware_fn_count",
    "risk_control_accepted_malware_count",
    "risk_control_num_accepted",
    "risk_control_num_rejected",
    "risk_control_target_met_empirically",
    "mean_semantic_attention_entropy",
    "mean_cross_modal_attention",
    "semantic_residual_gate",
    "mean_semantic_reliability_prior_api",
    "mean_semantic_reliability_prior_graph",
    "mean_semantic_reliability_prior_manifest",
    "mean_semantic_attention_to_api",
    "mean_semantic_attention_to_graph",
    "mean_semantic_attention_to_manifest",
}

BRANCH_AGGREGATE_SUFFIXES = (
    "_reliability_brier",
    "_reliability_ece_10",
    "_reliability_auc",
    "_reliability_ap",
    "_reliability_mean",
    "_branch_accuracy",
    "_reliability_accuracy_gap",
)

NON_METRIC_COLUMNS = {
    "experiment",
    "method",
    "seed",
    "run_dir",
    "summary_path",
    "resolved_config_sha256",
    "method_protocol_sha256",
    "method_implementation_sha256",
    "section",
    "scenario",
    "pt_dir",
    "csv",
    "perturb_type",
}


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


def _run_identity(
    summary_path: Path,
    results_root: Path,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    embedded = (summary or {}).get("run_identity") or {}
    if isinstance(embedded, dict) and embedded.get("experiment_name"):
        experiment = str(embedded["experiment_name"])
        identity: dict[str, Any] = {
            "experiment": experiment,
            "method": _method_name(
                str(embedded.get("method_name") or _method_name(experiment))
            ),
            "seed": str(embedded.get("seed", "")),
            "run_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
        }
        for key, value in embedded.items():
            if key not in {"experiment_name", "method_name", "seed"} and isinstance(
                value, SCALAR_TYPES
            ):
                identity[str(key)] = value
        return identity
    if summary_path.name.startswith("summary_"):
        experiment = summary_path.stem[len("summary_") :]
        seed_match = re.search(r"(?:^|_)seed_(\d+)(?:_|$)", experiment)
        return {
            "experiment": experiment,
            "method": _method_name(experiment),
            "seed": seed_match.group(1) if seed_match else "",
            "run_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
        }
    try:
        rel = summary_path.parent.relative_to(results_root)
        parts = rel.parts
    except ValueError:
        parts = summary_path.parent.parts
    experiment = parts[-2] if len(parts) >= 2 else summary_path.parent.name
    seed = parts[-1] if parts else ""
    return {
        "experiment": experiment,
        "method": _method_name(experiment),
        "seed": seed,
        "run_dir": str(summary_path.parent),
        "summary_path": str(summary_path),
    }


def _method_name(experiment: str) -> str:
    match = re.fullmatch(r"(.+)_seed_\d+", str(experiment))
    if match:
        return match.group(1)
    return str(experiment)


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


def _summary_metric_rows(
    summary_path: Path,
    results_root: Path,
    summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = summary if summary is not None else _safe_load_yaml(summary_path)
    identity = _run_identity(summary_path, results_root, summary)
    rows: list[dict[str, Any]] = []
    _add_metric_row(
        rows, identity, "val_selection", "val_selection", summary.get("val_selection")
    )
    posthoc_metrics = summary.get("val_posthoc_calibration")
    if posthoc_metrics is None:
        posthoc_metrics = summary.get("val_calibration")
    _add_metric_row(
        rows,
        identity,
        "val_posthoc_calibration",
        "val_posthoc_calibration",
        posthoc_metrics,
    )
    _add_metric_row(
        rows,
        identity,
        "val_conformal_calibration",
        "val_conformal_calibration",
        summary.get("val_conformal_calibration"),
    )
    _add_metric_row(rows, identity, "test", "test", summary.get("test"))
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
    summary_paths = sorted(
        set(results_root.rglob("summary.yaml"))
        | set(results_root.rglob("summary_*.yaml"))
    )
    rows: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()
    for path in summary_paths:
        summary = _safe_load_yaml(path)
        payload = yaml.safe_dump(summary, sort_keys=True, allow_unicode=True).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        # Users often copy a run's nested summary.yaml to results/summary_*.yaml.
        # Count the run once even when both files are retained for convenience.
        if fingerprint in seen_payloads:
            continue
        seen_payloads.add(fingerprint)
        rows.extend(_summary_metric_rows(path, results_root, summary))
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


def _aggregate_metric_allowed(column: str, allowlist: set[str]) -> bool:
    if column in allowlist:
        return True
    return any(column.endswith(suffix) for suffix in BRANCH_AGGREGATE_SUFFIXES)


def aggregate_metrics(
    metrics: pd.DataFrame,
    metric_allowlist: set[str] | None = None,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    allowlist = set(metric_allowlist or DEFAULT_AGGREGATE_METRICS)
    frame = metrics.copy()
    if "method" not in frame.columns and "experiment" in frame.columns:
        frame["method"] = frame["experiment"].map(_method_name)
    group_columns = [column for column in AGGREGATE_GROUP_COLUMNS if column in frame.columns]
    if not group_columns:
        return pd.DataFrame()
    numeric_columns: list[str] = []
    for column in frame.columns:
        if column in NON_METRIC_COLUMNS:
            continue
        if not _aggregate_metric_allowed(column, allowlist):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            frame[column] = numeric
            numeric_columns.append(column)
    records: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby(group_columns, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(group_columns, group_values))
        for metric in numeric_columns:
            values = group_frame[metric].dropna()
            if values.empty:
                continue
            records.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "count": int(values.size),
                }
            )
    return pd.DataFrame.from_records(records)


def _filter_methods(aggregate: pd.DataFrame, prefixes: tuple[str, ...], extras: tuple[str, ...] = ()) -> pd.DataFrame:
    if aggregate.empty or "method" not in aggregate.columns:
        return pd.DataFrame()
    methods = aggregate["method"].astype(str)
    mask = methods.isin({"final", *extras})
    for prefix in prefixes:
        mask |= methods.str.startswith(prefix)
    return aggregate[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect robust tri-modal experiment summaries into paper tables."
    )
    parser.add_argument("--results-root", default="results/tri_modal_robust")
    parser.add_argument("--out-dir", default="tables")
    parser.add_argument(
        "--aggregate-metrics",
        nargs="*",
        default=None,
        help=(
            "Optional explicit metric allowlist for aggregate_*.csv. "
            "Raw main_results.csv is unaffected."
        ),
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    metrics = collect_metric_rows(results_root)
    aggregate = aggregate_metrics(
        metrics,
        metric_allowlist=set(args.aggregate_metrics) if args.aggregate_metrics else None,
    )
    _write_csv(metrics, out_dir / "main_results.csv")
    _write_csv(aggregate, out_dir / "aggregate_main_results.csv")
    _write_csv(_filter_methods(aggregate, ("i1_",)), out_dir / "aggregate_i1_ablation.csv")
    _write_csv(_filter_methods(aggregate, ("i2_",)), out_dir / "aggregate_i2_ablation.csv")
    _write_csv(
        _filter_methods(
            aggregate,
            ("i3_",),
            extras=("module_no_i3_selective_rejection",),
        ),
        out_dir / "aggregate_i3_ablation.csv",
    )
    _write_csv(_branch_reliability_rows(metrics), out_dir / "i1_reliability_calibration_summary.csv")
    _write_csv(_selected_columns(metrics, ATTENTION_KEYS), out_dir / "i2_attention_ablation.csv")
    _write_csv(_selected_columns(metrics, SELECTIVE_KEYS), out_dir / "i3_selective_results.csv")
    run_index_columns = [
        key
        for key in (
            "experiment",
            "method",
            "seed",
            "method_protocol_sha256",
            "method_implementation_sha256",
            "resolved_config_sha256",
            "model_fusion_mode",
            "combination_rule",
            "evidential_certainty_enabled",
            "classification_threshold_enabled",
            "classification_threshold_objective",
            "classification_min_malware_recall",
            "selective_prediction_mode",
            "risk_control_level",
            "target_coverage",
            "conformal_uses_raw_conflict",
            "run_dir",
            "summary_path",
        )
        if key in metrics.columns
    ]
    run_index = metrics[run_index_columns].drop_duplicates() if run_index_columns else pd.DataFrame()
    _write_csv(run_index, out_dir / "summary_index.csv")


if __name__ == "__main__":
    main()
