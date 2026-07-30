from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


BRANCHES = ("api", "graph", "manifest", "joint")
METRIC_SUMMARY_SCHEMA_VERSION = 14
FORCED_CLASSIFICATION_SECTIONS = frozenset({"test", "robust", "extra_eval"})

SCALAR_TYPES = (str, int, float, bool, type(None))

SELECTIVE_KEYS = (
    "metric_schema_version",
    "selective_prediction_mode",
    "selective_score_type",
    "classification_threshold",
    "fixed_0_5_acc",
    "fixed_0_5_macro_f1",
    "fixed_0_5_f1_pos",
    "fixed_0_5_recall_pos",
    "coverage",
    "selective_eligible_rate",
    "num_ineligible_forced_reject",
    "selective_metrics_defined",
    "selective_risk",
    "selective_acc",
    "selective_macro_f1",
    "aurc",
    "aurc_defined",
    "aurc_tie_policy",
    "malware_fn_risk_aurc",
    "malware_fn_risk_aurc_defined",
    "malware_fn_risk_aurc_tie_policy",
    "malware_fn_risk_aurc_target",
    "malware_fn_risk_aurc_denominator",
    "malware_fn_risk_aurc_coverage_normalization",
    "selective_ranking_num_eligible",
    "selective_max_achievable_coverage",
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
    "conformal_accepted_fn_risk_among_malware",
    "conformal_fn_rate_given_accepted_malware",
    "conformal_malware_fn_count",
    "conformal_accepted_malware_count",
    "conformal_selective_risk",
    "conformal_selective_acc",
    "conformal_num_accepted",
    "conformal_num_rejected",
    "conformal_num_empty_sets",
    "conformal_num_ambiguous_sets",
    "conformal_num_ineligible_forced_reject",
    "conformal_ineligible_set_policy",
    "conformal_empirical_coverage_benign",
    "conformal_empirical_coverage_malware",
    "risk_control_threshold",
    "risk_control_acceptance_comparison",
    "risk_control_risk_level",
    "risk_control_risk_target",
    "risk_control_guarantee_type",
    "risk_control_guarantee_scope",
    "risk_control_risk_numerator",
    "risk_control_risk_denominator",
    "risk_control_eligibility_rule",
    "risk_control_calibration_feasible",
    "risk_control_calibration_corrected_risk",
    "risk_control_acceptance_rate",
    "risk_control_rejection_rate",
    "risk_control_selective_risk",
    "risk_control_selective_acc",
    "risk_control_accepted_fn_risk_among_malware",
    "risk_control_fn_rate_given_accepted_malware",
    "risk_control_malware_fn_count",
    "risk_control_accepted_malware_count",
    "risk_control_num_accepted",
    "risk_control_num_rejected",
    "risk_control_num_ineligible_forced_reject",
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
    "selective_eligible_rate",
    "num_ineligible_forced_reject",
    "selective_risk",
    "selective_acc",
    "selective_macro_f1",
    "aurc",
    "aurc_defined",
    "malware_fn_risk_aurc",
    "malware_fn_risk_aurc_defined",
    "selective_ranking_num_eligible",
    "selective_max_achievable_coverage",
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
    "conformal_accepted_fn_risk_among_malware",
    "conformal_fn_rate_given_accepted_malware",
    "conformal_malware_fn_count",
    "conformal_accepted_malware_count",
    "conformal_selective_risk",
    "conformal_selective_acc",
    "conformal_empirical_coverage_benign",
    "conformal_empirical_coverage_malware",
    "risk_control_threshold",
    "risk_control_risk_level",
    "risk_control_calibration_feasible",
    "risk_control_calibration_corrected_risk",
    "risk_control_acceptance_rate",
    "risk_control_rejection_rate",
    "risk_control_selective_risk",
    "risk_control_selective_acc",
    "risk_control_accepted_fn_risk_among_malware",
    "risk_control_fn_rate_given_accepted_malware",
    "risk_control_malware_fn_count",
    "risk_control_accepted_malware_count",
    "risk_control_num_accepted",
    "risk_control_num_rejected",
    "risk_control_num_ineligible_forced_reject",
    "risk_control_target_met_empirically",
    "tcp_mse",
    "tcp_mae",
    "tcp_pearson",
    "tcp_spearman",
    "error_auroc",
    "robust_mean_macro_f1",
    "robust_joint_mean_macro_f1",
    "robust_mean_gain",
    "robust_worst_macro_f1",
    "clean_macro_f1",
}

BRANCH_AGGREGATE_SUFFIXES = (
    "_competence_tcp_mse",
    "_competence_tcp_mae",
    "_competence_tcp_pearson",
    "_competence_tcp_spearman",
    "_competence_error_auroc",
    "_branch_accuracy",
)

NON_METRIC_COLUMNS = {
    "metric_schema_version",
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


def _validate_result_summary_schema(
    summary_path: Path,
    summary: dict[str, Any],
) -> None:
    """Reject legacy summaries whose metric semantics are no longer compatible."""

    version = summary.get("metric_schema_version")
    if version != METRIC_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"{summary_path} uses metric_schema_version={version!r}; expected="
            f"{METRIC_SUMMARY_SCHEMA_VERSION}. Rerun with the current code and "
            "collect from a new-run-only results root."
        )
    identity = summary.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{summary_path} is missing the required run_identity mapping")
    required_identity = {
        "experiment_name",
        "method_name",
        "seed",
        "method_protocol_id",
        "method_protocol_sha256",
        "method_implementation_sha256",
    }
    missing = sorted(required_identity - set(identity))
    if missing:
        raise ValueError(
            f"{summary_path} run_identity is missing required fields: {missing}"
        )
    validation_sections = (
        "val_model_selection",
        "val_model_selection_threshold_fit",
        "val_decision_calibration",
    )
    invalid_validation = [
        key
        for key in validation_sections
        if not isinstance(summary.get(key), dict)
    ]
    if invalid_validation:
        raise ValueError(
            f"{summary_path} is missing schema-v14 two-role validation "
            f"sections: {invalid_validation}"
        )
    retired_validation = sorted(
        {
            "val_selection",
            "val_posthoc_calibration",
            "val_conformal_calibration",
        }
        & set(summary)
    )
    if retired_validation:
        raise ValueError(
            f"{summary_path} contains retired validation sections "
            f"{retired_validation}; rerun with the two-role producer"
        )
    stage_b = summary.get("stage_b_training")
    if isinstance(stage_b, dict) and bool(stage_b.get("enabled", False)):
        stage_b_threshold = stage_b.get("classification_threshold")
        deployment_threshold = summary.get("classification_threshold")
        if not isinstance(stage_b_threshold, dict) or not bool(
            stage_b_threshold.get("locked_by_stage_b", False)
        ):
            raise ValueError(
                f"{summary_path} has no locked Stage-B classification threshold"
            )
        if not isinstance(deployment_threshold, dict):
            raise ValueError(
                f"{summary_path} has no deployment classification threshold"
            )
        try:
            stage_b_value = float(stage_b_threshold["threshold"])
            deployment_value = float(deployment_threshold["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{summary_path} has an invalid Stage-B threshold payload"
            ) from exc
        if (
            not math.isfinite(stage_b_value)
            or not math.isfinite(deployment_value)
            or not math.isclose(
                stage_b_value,
                deployment_value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                f"{summary_path} Stage-B and deployment thresholds disagree"
            )


def _run_identity(
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    # Schema validation requires this embedded identity before this function is
    # called. Inferring method/seed from filenames or directory layouts would
    # therefore be unreachable and, worse, could silently mix incompatible runs.
    embedded = summary["run_identity"]
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
    summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = summary if summary is not None else _safe_load_yaml(summary_path)
    _validate_result_summary_schema(summary_path, summary)
    identity = _run_identity(summary_path, summary)
    identity["metric_schema_version"] = int(summary["metric_schema_version"])
    rows: list[dict[str, Any]] = []
    _add_metric_row(
        rows,
        identity,
        "val_model_selection",
        "val_model_selection",
        summary.get("val_model_selection"),
    )
    _add_metric_row(
        rows,
        identity,
        "val_model_selection_threshold_fit",
        "val_model_selection_threshold_fit",
        summary.get("val_model_selection_threshold_fit"),
    )
    _add_metric_row(
        rows,
        identity,
        "val_decision_calibration",
        "val_decision_calibration",
        summary.get("val_decision_calibration"),
    )
    stage_b = summary.get("stage_b_training")
    if isinstance(stage_b, dict):
        _add_metric_row(
            rows,
            identity,
            "stage_b_lifecycle",
            "stage_b_lifecycle",
            stage_b,
        )
        cache = stage_b.get("cache")
        if isinstance(cache, dict):
            for source, metrics in cache.items():
                _add_metric_row(
                    rows,
                    identity,
                    "stage_b_cache",
                    str(source),
                    metrics,
                )
        competence = stage_b.get("competence")
        if isinstance(competence, dict):
            _add_metric_row(
                rows,
                identity,
                "stage_b_competence_fit",
                "fit",
                competence,
            )
            diagnostics = competence.get("diagnostics")
            if isinstance(diagnostics, dict):
                for source, source_metrics in diagnostics.items():
                    if not isinstance(source_metrics, dict):
                        continue
                    for expert, metrics in source_metrics.items():
                        _add_metric_row(
                            rows,
                            identity,
                            "stage_b_competence_diagnostics",
                            f"{source}/{expert}",
                            metrics,
                        )
        router = stage_b.get("router")
        if isinstance(router, dict):
            _add_metric_row(
                rows,
                identity,
                "stage_b_router",
                "selected",
                router.get("selected"),
            )
            candidates = router.get("candidates")
            if isinstance(candidates, list):
                for index, metrics in enumerate(candidates):
                    scenario = (
                        f"degradation_weight_"
                        f"{metrics.get('degradation_loss_weight')}"
                        if isinstance(metrics, dict)
                        else f"candidate_{index}"
                    )
                    _add_metric_row(
                        rows,
                        identity,
                        "stage_b_router_candidate",
                        scenario,
                        metrics,
                    )
    _add_metric_row(rows, identity, "test", "test", summary.get("test"))
    robust = summary.get("robust") or {}
    if isinstance(robust, dict):
        for scenario, metrics in robust.items():
            _add_metric_row(rows, identity, "robust", str(scenario), metrics)
    extra_eval = summary.get("extra_eval") or {}
    if isinstance(extra_eval, dict):
        for scenario, metrics in extra_eval.items():
            _add_metric_row(rows, identity, "extra_eval", str(scenario), metrics)
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
        rows.extend(_summary_metric_rows(path, summary))
    return pd.DataFrame.from_records(rows)


def _branch_competence_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame.from_records(records)
    stage_rows = metrics[
        metrics.get("section", pd.Series(dtype=object)).astype(str)
        == "stage_b_competence_diagnostics"
    ]
    base_columns = [
        "experiment",
        "method",
        "seed",
        "run_dir",
        "section",
        "scenario",
    ]
    metric_columns = (
        "defined",
        "num_rows",
        "tcp_mse",
        "tcp_mae",
        "tcp_pearson",
        "tcp_spearman",
        "error_auroc",
    )
    for row in stage_rows.to_dict("records"):
        scenario = str(row.get("scenario") or "")
        if "/" not in scenario:
            continue
        source, expert = scenario.rsplit("/", 1)
        if expert not in BRANCHES:
            continue
        records.append(
            {
                **{
                    key: row.get(key)
                    for key in base_columns
                    if key in row
                },
                "source": source,
                "expert": expert,
                **{
                    key: row.get(key)
                    for key in metric_columns
                    if key in row
                },
            }
        )
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
                    # A sample standard deviation is undefined for one run.
                    # ``None`` becomes an empty CSV cell instead of falsely
                    # claiming zero between-seed variation.
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "count": int(values.size),
                }
            )
    aggregate = pd.DataFrame.from_records(records)
    if "std" in aggregate.columns:
        # Preserve Python ``None`` in record/YAML consumers while pandas emits
        # the same value as an empty, parseable field in CSV output.
        aggregate["std"] = pd.Series(
            [record["std"] for record in records],
            index=aggregate.index,
            dtype=object,
        )
    return aggregate


def build_paper_forced_classification_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Expose only fixed-0.5 decisions for cross-method classification tables.

    The primary main table gives every method the same model-selection
    macro-F1 threshold budget. This secondary table intentionally discards
    those fitted operating points and renames the stored fixed-0.5 metrics to
    canonical paper columns for a no-threshold-tuning sensitivity comparison.
    """

    if metrics.empty:
        return pd.DataFrame()
    if "section" not in metrics:
        raise ValueError("Collected metrics are missing the section column")
    frame = metrics[
        metrics["section"].astype(str).isin(FORCED_CLASSIFICATION_SECTIONS)
    ].copy()
    if frame.empty:
        return frame

    required = (
        "fixed_0_5_acc",
        "fixed_0_5_macro_f1",
        "fixed_0_5_f1_pos",
        "fixed_0_5_recall_pos",
    )
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise ValueError(
            "Formal forced-classification rows require explicit fixed-0.5 "
            f"metrics; missing columns={missing}"
        )
    incomplete = frame[list(required)].isna().any(axis=1)
    if bool(incomplete.any()):
        identity_keys = [
            key
            for key in ("experiment", "seed", "section", "scenario")
            if key in frame
        ]
        examples = frame.loc[incomplete, identity_keys].head(5)
        raise ValueError(
            "Formal forced-classification rows contain missing fixed-0.5 "
            f"metrics; examples={examples.to_dict('records')}"
        )

    identity_columns = [
        key
        for key in (
            "metric_schema_version",
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
            "perturb_strength",
        )
        if key in frame
    ]
    threshold_independent = [
        key for key in ("auc", "ap", "brier", "ece_10") if key in frame
    ]
    out = frame[
        [
            *identity_columns,
            *required,
            *threshold_independent,
        ]
    ].copy()
    out = out.rename(
        columns={
            "fixed_0_5_acc": "acc",
            "fixed_0_5_macro_f1": "macro_f1",
            "fixed_0_5_f1_pos": "f1_pos",
            "fixed_0_5_recall_pos": "recall_pos",
        }
    )
    out["classification_threshold"] = 0.5
    out["classification_protocol"] = "forced_fixed_0_5_v1"
    return out


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
    paper_forced = build_paper_forced_classification_table(metrics)
    paper_forced_aggregate = aggregate_metrics(
        paper_forced,
        metric_allowlist={
            "classification_threshold",
            "acc",
            "macro_f1",
            "f1_pos",
            "recall_pos",
            "auc",
            "ap",
            "brier",
            "ece_10",
        },
    )
    _write_csv(metrics, out_dir / "main_results.csv")
    _write_csv(aggregate, out_dir / "aggregate_main_results.csv")
    _write_csv(
        paper_forced,
        out_dir / "paper_forced_classification_results.csv",
    )
    _write_csv(
        paper_forced_aggregate,
        out_dir / "aggregate_paper_forced_classification_results.csv",
    )
    _write_csv(
        _filter_methods(aggregate, ("ablation_i1_",)),
        out_dir / "aggregate_i1_ablation.csv",
    )
    _write_csv(
        _filter_methods(aggregate, ("ablation_i2_",)),
        out_dir / "aggregate_i2_ablation.csv",
    )
    _write_csv(
        _filter_methods(
            aggregate,
            ("i3_", "ablation_i3_"),
            extras=("module_no_i3_decision_layer",),
        ),
        out_dir / "aggregate_i3_ablation.csv",
    )
    _write_csv(
        _branch_competence_rows(metrics),
        out_dir / "i1_tcp_competence_summary.csv",
    )
    _write_csv(_selected_columns(metrics, SELECTIVE_KEYS), out_dir / "i3_selective_results.csv")
    run_index_columns = [
        key
        for key in (
            "experiment",
            "method",
            "seed",
            "metric_schema_version",
            "method_protocol_id",
            "method_protocol_sha256",
            "method_implementation_sha256",
            "resolved_config_sha256",
            "model_fusion_mode",
            "training_objective",
            "stage_a_primary_expert",
            "stage_a_atomic_aux_weight",
            "stage_b_enabled",
            "stage_b_fit_population",
            "i1_target",
            "i1_inputs",
            "i1_regression",
            "i1_degraded_loss_weight",
            "i1_ranking_weight",
            "i1_clean_noninferiority_relative_tolerance",
            "i1_clean_noninferiority_absolute_tolerance",
            "i2_formula",
            "i2_clean_anchor_kl_weight",
            "i2_clean_noninferiority_tolerance",
            "i2_degraded_source_noninferiority_tolerance",
            "i2_minimum_robust_gain",
            "classification_threshold_enabled",
            "classification_threshold_objective",
            "classification_threshold_selection_rule",
            "classification_threshold_constraint",
            "selective_prediction_mode",
            "selective_score_type",
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
