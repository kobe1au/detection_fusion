from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


BASELINE_METRIC_SCHEMA_VERSION = 16
METRIC_SUMMARY_SCHEMA_VERSION = BASELINE_METRIC_SCHEMA_VERSION
CARE_SUMMARY_SCHEMA_VERSION = 2
CARE_PROTOCOL_ID = "care_droid_v1"
SCALAR_TYPES = (str, int, float, bool, type(None))
PRIMARY_CLASSIFICATION_SECTIONS = frozenset(
    {"test", "robust", "extra_eval"}
)
CLASSIFICATION_PROTOCOL_ID = "binary_argmax_fixed_0_5_v1"

AGGREGATE_GROUP_COLUMNS = (
    "method",
    "method_protocol_sha256",
    "method_implementation_sha256",
    "section",
    "scenario",
)
NON_METRIC_COLUMNS = {
    "metric_schema_version",
    "summary_schema_version",
    "experiment",
    "method",
    "seed",
    "run_dir",
    "summary_path",
    "method_protocol_id",
    "method_protocol_sha256",
    "method_implementation_sha256",
    "section",
    "scenario",
    "guarantee_scope",
    "crc_status",
}
DEFAULT_AGGREGATE_METRICS = {
    "acc",
    "macro_f1",
    "f1_pos",
    "recall_pos",
    "auc",
    "ap",
    "brier",
    "ece_10",
    "log_loss",
    "coverage",
    "accepted_accuracy",
    "accepted_macro_f1",
    "accepted_fn_count",
    "empirical_malware_accepted_fn_risk",
    "corrected_malware_accepted_fn_risk",
    "structural_reject_count",
    "rejected_count",
    "lambda",
    "corrected_risk",
    "empirical_risk",
    "overall_coverage",
    "N_malware",
    "error_auroc",
    "error_auprc",
    "checkpoint_score",
}


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _method_name(experiment: str) -> str:
    match = re.fullmatch(r"(.+)_seed_\d+", str(experiment))
    return match.group(1) if match else str(experiment)


def _scalar_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, SCALAR_TYPES)
    }


def _add_row(
    rows: list[dict[str, Any]],
    identity: Mapping[str, Any],
    *,
    section: str,
    scenario: str,
    metrics: Mapping[str, Any] | None,
) -> None:
    scalars = _scalar_metrics(metrics)
    if scalars:
        rows.append(
            {
                **identity,
                "section": str(section),
                "scenario": str(scenario),
                **scalars,
            }
        )


def _validate_baseline_summary(path: Path, summary: Mapping[str, Any]) -> None:
    if summary.get("metric_schema_version") != BASELINE_METRIC_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses metric_schema_version="
            f"{summary.get('metric_schema_version')!r}; expected "
            f"{BASELINE_METRIC_SCHEMA_VERSION}"
        )
    identity = summary.get("run_identity")
    required = {
        "experiment_name",
        "method_name",
        "seed",
        "method_protocol_id",
        "method_protocol_sha256",
        "method_implementation_sha256",
    }
    if not isinstance(identity, Mapping):
        raise ValueError(f"{path} is missing run_identity")
    missing = sorted(required - set(identity))
    if missing:
        raise ValueError(f"{path} run_identity is missing {missing}")
    required_sections = {
        "expert_val_checkpoint_selection",
        "val_model_selection",
        "val_decision_calibration",
    }
    missing_sections = sorted(
        key
        for key in required_sections
        if not isinstance(summary.get(key), Mapping)
    )
    if missing_sections:
        raise ValueError(
            f"{path} is missing schema-v16 fixed-rule role sections: "
            f"{missing_sections}"
        )
    training_split = summary.get("training_split")
    if not isinstance(training_split, Mapping):
        raise ValueError(f"{path} is missing baseline training_split")
    if (
        training_split.get("protocol_id")
        != "baseline_expert_train_expert_val_package_group_disjoint_v1"
    ):
        raise ValueError(f"{path} uses an unsupported baseline role split")
    rule = summary.get("classification_rule")
    if not isinstance(rule, Mapping):
        raise ValueError(f"{path} is missing classification_rule")
    if (
        rule.get("protocol_id") != CLASSIFICATION_PROTOCOL_ID
        or float(rule.get("threshold", float("nan"))) != 0.5
        or bool(rule.get("fitted", True))
    ):
        raise ValueError(
            f"{path} does not use the unfitted argmax/0.5 classifier"
        )
def _baseline_rows(
    path: Path,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_baseline_summary(path, summary)
    embedded = summary["run_identity"]
    experiment = str(embedded["experiment_name"])
    identity: dict[str, Any] = {
        "metric_schema_version": BASELINE_METRIC_SCHEMA_VERSION,
        "experiment": experiment,
        "method": _method_name(str(embedded["method_name"])),
        "seed": str(embedded["seed"]),
        "run_dir": str(path.parent),
        "summary_path": str(path),
    }
    for key, value in embedded.items():
        if (
            key not in {"experiment_name", "method_name", "seed"}
            and isinstance(value, SCALAR_TYPES)
        ):
            identity[str(key)] = value
    rows: list[dict[str, Any]] = []
    for section in (
        "expert_val_checkpoint_selection",
        "val_model_selection",
        "val_decision_calibration",
        "test",
    ):
        _add_row(
            rows,
            identity,
            section=section,
            scenario=section,
            metrics=summary.get(section),
        )
    for section in ("robust", "extra_eval"):
        scenarios = summary.get(section) or {}
        if isinstance(scenarios, Mapping):
            for scenario, metrics in scenarios.items():
                _add_row(
                    rows,
                    identity,
                    section=section,
                    scenario=str(scenario),
                    metrics=metrics,
                )
    return rows


def _validate_care_summary(path: Path, summary: Mapping[str, Any]) -> None:
    if summary.get("summary_schema_version") != CARE_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has unsupported CARE summary_schema_version="
            f"{summary.get('summary_schema_version')!r}"
        )
    if summary.get("method_protocol_id") != CARE_PROTOCOL_ID:
        raise ValueError(f"{path} is not a {CARE_PROTOCOL_ID} summary")
    if not isinstance(summary.get("test"), Mapping):
        raise ValueError(f"{path} omits CARE test scenarios")
    if not isinstance(summary.get("decision_calibration"), Mapping):
        raise ValueError(f"{path} omits CARE decision calibration")
    if not str(summary.get("role_identity_sha256", "")):
        raise ValueError(f"{path} omits immutable CARE role identity")


def _care_test_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("raw_selected_classification") or {}
    selective = payload.get("selective") or {}
    if not isinstance(raw, Mapping) or not isinstance(selective, Mapping):
        raise ValueError("CARE test cell omits raw/selective metrics")
    metrics = {
        "acc": raw.get("accuracy"),
        "macro_f1": raw.get("macro_f1"),
        "f1_pos": raw.get("malware_f1"),
        "recall_pos": raw.get("malware_recall"),
        "auc": raw.get("auc"),
        "ap": raw.get("ap"),
        "brier": raw.get("brier"),
        "log_loss": raw.get("log_loss"),
        "classification_threshold": 0.5,
        "classification_protocol": CLASSIFICATION_PROTOCOL_ID,
        **_scalar_metrics(selective),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _care_rows(
    path: Path,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_care_summary(path, summary)
    experiment = str(summary.get("experiment_name") or "care_droid")
    identity = {
        "summary_schema_version": CARE_SUMMARY_SCHEMA_VERSION,
        "experiment": experiment,
        "method": _method_name(experiment),
        "seed": str(summary.get("seed", "")),
        "run_dir": str(path.parent),
        "summary_path": str(path),
        "method_protocol_id": CARE_PROTOCOL_ID,
        "method_protocol_sha256": str(
            summary.get("method_protocol_sha256")
            or summary["role_identity_sha256"]
        ),
        "method_implementation_sha256": str(
            summary.get("method_implementation_sha256")
            or (summary.get("artifacts") or {}).get(
                "pipeline_sha256", ""
            )
        ),
    }
    rows: list[dict[str, Any]] = []
    stage = summary.get("stage_a") or {}
    if isinstance(stage, Mapping):
        _add_row(
            rows,
            identity,
            section="stage_a",
            scenario="clean_agm_checkpoint_selection",
            metrics={
                "checkpoint_score": stage.get("best_score"),
                "best_epoch": stage.get("best_epoch"),
                "epochs_ran": stage.get("epochs_ran"),
            },
        )
    _add_row(
        rows,
        identity,
        section="decision_calibration",
        scenario="natural_crc",
        metrics=summary.get("decision_calibration"),
    )
    diagnostics = summary.get("oof_diagnostics") or {}
    if isinstance(diagnostics, Mapping):
        selected = diagnostics.get("selected_path_correctness")
        _add_row(
            rows,
            identity,
            section="oof_risk",
            scenario="selected_path",
            metrics=selected if isinstance(selected, Mapping) else None,
        )
        path_rows = diagnostics.get("path_correctness") or {}
        if isinstance(path_rows, Mapping):
            for path_name, metrics in path_rows.items():
                _add_row(
                    rows,
                    identity,
                    section="oof_risk",
                    scenario=str(path_name),
                    metrics=metrics,
                )
        _add_row(
            rows,
            identity,
            section="oof_routing",
            scenario="switch_repair_destruction",
            metrics=diagnostics.get("routing_switch"),
        )
        _add_row(
            rows,
            identity,
            section="oof_path_diversity",
            scenario="agm_vs_fallback_oracle",
            metrics=diagnostics.get("oracle_path_diversity"),
        )
    test = summary["test"]
    for scenario, payload in test.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} CARE test cell {scenario!r} is invalid")
        _add_row(
            rows,
            identity,
            section="test" if str(scenario) == "clean" else "robust",
            scenario=str(scenario),
            metrics=_care_test_metrics(payload),
        )
    return rows


def _summary_rows(
    path: Path,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if summary.get("method_protocol_id") == CARE_PROTOCOL_ID:
        return _care_rows(path, summary)
    return _baseline_rows(path, summary)


def collect_metric_rows(results_root: Path) -> pd.DataFrame:
    paths = sorted(
        set(results_root.rglob("summary.yaml"))
        | set(results_root.rglob("summary_*.yaml"))
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        summary = _safe_load_yaml(path)
        fingerprint = hashlib.sha256(
            yaml.safe_dump(
                summary,
                sort_keys=True,
                allow_unicode=True,
            ).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rows.extend(_summary_rows(path, summary))
    return pd.DataFrame.from_records(rows)


def aggregate_metrics(
    metrics: pd.DataFrame,
    metric_allowlist: set[str] | None = None,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    allowlist = set(metric_allowlist or DEFAULT_AGGREGATE_METRICS)
    frame = metrics.copy()
    if "method" not in frame and "experiment" in frame:
        frame["method"] = frame["experiment"].map(_method_name)
    groups = [
        key for key in AGGREGATE_GROUP_COLUMNS if key in frame.columns
    ]
    if not groups:
        return pd.DataFrame()
    numeric: list[str] = []
    for column in frame.columns:
        if column in NON_METRIC_COLUMNS or column not in allowlist:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            frame[column] = converted
            numeric.append(column)
    records: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby(groups, dropna=False):
        values = group if isinstance(group, tuple) else (group,)
        identity = dict(zip(groups, values))
        for metric in numeric:
            observed = group_frame[metric].dropna()
            if observed.empty:
                continue
            records.append(
                {
                    **identity,
                    "metric": metric,
                    "mean": float(observed.mean()),
                    "std": (
                        float(observed.std(ddof=1))
                        if len(observed) > 1
                        else None
                    ),
                    "count": int(observed.size),
                }
            )
    result = pd.DataFrame.from_records(records)
    if "std" in result:
        result["std"] = pd.Series(
            [record["std"] for record in records],
            index=result.index,
            dtype=object,
        )
    return result


def build_paper_classification_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build the primary table under the common unfitted argmax/0.5 rule."""

    if metrics.empty:
        return pd.DataFrame()
    if "section" not in metrics:
        raise ValueError("Collected metrics are missing section")
    frame = metrics[
        metrics["section"].astype(str).isin(
            PRIMARY_CLASSIFICATION_SECTIONS
        )
    ].copy()
    if frame.empty:
        return frame
    required = (
        "acc",
        "macro_f1",
        "recall_pos",
        "classification_threshold",
        "classification_protocol",
    )
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(
            f"Primary classification comparison is missing metrics {missing}"
        )
    invalid = frame[list(required)].isna().any(axis=1)
    if bool(invalid.any()):
        raise ValueError(
            "Primary classification comparison contains incomplete rows"
        )
    numeric_threshold = pd.to_numeric(
        frame["classification_threshold"], errors="coerce"
    )
    if bool((numeric_threshold != 0.5).any()):
        raise ValueError(
            "Primary classification comparison contains a fitted or "
            "non-0.5 threshold"
        )
    if bool(
        (
            frame["classification_protocol"].astype(str)
            != CLASSIFICATION_PROTOCOL_ID
        ).any()
    ):
        raise ValueError(
            "Primary classification comparison mixes prediction rules"
        )
    identity = [
        key
        for key in (
            "experiment",
            "method",
            "seed",
            "section",
            "scenario",
            "run_dir",
            "summary_path",
            "method_protocol_id",
            "method_protocol_sha256",
            "method_implementation_sha256",
        )
        if key in frame
    ]
    optional = [
        key
        for key in (
            "f1_pos",
            "auc",
            "ap",
            "brier",
            "ece_10",
        )
        if key in frame
    ]
    out = frame[[*identity, *required, *optional]].copy()
    return out


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect baseline and frozen CARE-Droid summaries."
    )
    parser.add_argument("--results-root", default="results/tri_modal_robust")
    parser.add_argument("--out-dir", default="tables")
    args = parser.parse_args()

    metrics = collect_metric_rows(Path(args.results_root))
    aggregate = aggregate_metrics(metrics)
    primary = build_paper_classification_table(metrics)
    out_dir = Path(args.out_dir)
    _write_csv(metrics, out_dir / "main_results.csv")
    _write_csv(aggregate, out_dir / "aggregate_main_results.csv")
    _write_csv(primary, out_dir / "paper_classification_results.csv")
    care_ablation = aggregate[
        aggregate.get("method", pd.Series(dtype=object))
        .astype(str)
        .str.startswith("care_ablation_")
    ]
    _write_csv(care_ablation, out_dir / "aggregate_care_ablation.csv")
    if not metrics.empty and "section" in metrics:
        selective = metrics[
            metrics["section"].astype(str).isin(
                {"decision_calibration", "test", "robust"}
            )
        ]
    else:
        selective = pd.DataFrame()
    _write_csv(selective, out_dir / "care_selective_results.csv")
    index_columns = [
        key
        for key in (
            "experiment",
            "method",
            "seed",
            "method_protocol_id",
            "method_protocol_sha256",
            "method_implementation_sha256",
            "run_dir",
            "summary_path",
        )
        if key in metrics
    ]
    run_index = (
        metrics[index_columns].drop_duplicates()
        if index_columns
        else pd.DataFrame()
    )
    _write_csv(run_index, out_dir / "summary_index.csv")


if __name__ == "__main__":
    main()
