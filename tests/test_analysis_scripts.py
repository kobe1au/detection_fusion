from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fusion.baseline_train import (
    BASELINE_METRIC_SCHEMA_VERSION as PRODUCER_METRIC_SCHEMA_VERSION,
)
from scripts.collect_experiment_results import (
    METRIC_SUMMARY_SCHEMA_VERSION as COLLECTOR_METRIC_SCHEMA_VERSION,
    aggregate_metrics,
    build_paper_classification_table,
    collect_metric_rows,
)


def _baseline_summary(*, seed: int = 42, macro_f1: float = 0.9) -> dict:
    return {
        "metric_schema_version": PRODUCER_METRIC_SCHEMA_VERSION,
        "run_identity": {
            "experiment_name": f"baseline_concat_seed_{seed}",
            "method_name": "baseline_concat",
            "seed": seed,
            "method_protocol_id": "baseline_v1",
            "method_protocol_sha256": "a" * 64,
            "method_implementation_sha256": "b" * 64,
        },
        "training_split": {
            "protocol_id": (
                "baseline_expert_train_expert_val_"
                "package_group_disjoint_v1"
            ),
        },
        "classification_rule": {
            "protocol_id": "binary_argmax_fixed_0_5_v1",
            "threshold": 0.5,
            "fitted": False,
        },
        "expert_val_checkpoint_selection": {
            "macro_f1": macro_f1 - 0.01
        },
        "val_model_selection": {"macro_f1": macro_f1 - 0.01},
        "val_decision_calibration": {"macro_f1": macro_f1 - 0.02},
        "test": {
            "macro_f1": macro_f1,
            "acc": macro_f1 + 0.01,
            "f1_pos": macro_f1 - 0.03,
            "recall_pos": macro_f1 - 0.04,
            "auc": 0.95,
            "classification_threshold": 0.5,
            "classification_protocol": "binary_argmax_fixed_0_5_v1",
        },
    }


def _care_summary(*, seed: int = 42, macro_f1: float = 0.91) -> dict:
    return {
        "summary_schema_version": 2,
        "experiment_name": f"care_droid_seed_{seed}",
        "method_name": "care_droid",
        "method_protocol_id": "care_droid_v1",
        "method_protocol_sha256": "c" * 64,
        "method_implementation_sha256": "d" * 64,
        "role_identity_sha256": "e" * 64,
        "seed": seed,
        "stage_a": {
            "best_score": macro_f1 - 0.01,
            "best_epoch": 7,
            "epochs_ran": 15,
        },
        "decision_calibration": {
            "lambda": 0.72,
            "crc_status": "feasible",
            "corrected_risk": 0.04,
            "empirical_risk": 0.03,
            "overall_coverage": 0.8,
            "N_malware": 240,
        },
        "oof_diagnostics": {
            "selected_path_correctness": {
                "brier": 0.11,
                "error_auroc": 0.74,
            },
            "path_correctness": {
                "agm": {"brier": 0.12, "error_auroc": 0.72},
            },
            "routing_switch": {
                "repair_count": 8,
                "destruction_count": 2,
            },
            "oracle_path_diversity": {
                "agm_wrong_fallback_correct_count": 20,
            },
        },
        "test": {
            "clean": {
                "raw_selected_classification": {
                    "accuracy": macro_f1 + 0.005,
                    "macro_f1": macro_f1,
                    "malware_f1": macro_f1 - 0.005,
                    "malware_recall": macro_f1 - 0.01,
                    "auc": 0.96,
                    "ap": 0.95,
                    "brier": 0.08,
                },
                "selective": {
                    "coverage": 0.8,
                    "accepted_accuracy": 0.97,
                    "accepted_macro_f1": 0.96,
                    "accepted_fn_count": 3,
                    "empirical_malware_accepted_fn_risk": 0.01,
                    "corrected_malware_accepted_fn_risk": 0.014,
                    "guarantee_scope": "natural_test_expected_crc",
                },
            },
            "api_event_dropout@0.5": {
                "raw_selected_classification": {
                    "accuracy": macro_f1 - 0.02,
                    "macro_f1": macro_f1 - 0.025,
                    "malware_f1": macro_f1 - 0.027,
                    "malware_recall": macro_f1 - 0.03,
                    "auc": 0.94,
                    "ap": 0.93,
                    "brier": 0.10,
                },
                "selective": {
                    "coverage": 0.75,
                    "accepted_accuracy": 0.95,
                    "accepted_macro_f1": 0.94,
                    "accepted_fn_count": 5,
                    "empirical_malware_accepted_fn_risk": 0.02,
                    "corrected_malware_accepted_fn_risk": 0.024,
                    "guarantee_scope": "empirical_only_distribution_shift",
                },
            },
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def test_collector_parses_current_baseline_and_care_schemas(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "baseline" / "summary.yaml", _baseline_summary())
    _write(tmp_path / "care" / "summary.yaml", _care_summary())

    rows = collect_metric_rows(tmp_path)

    assert set(rows["method"]) == {"baseline_concat", "care_droid"}
    care_clean = rows[
        (rows["method"] == "care_droid")
        & (rows["section"] == "test")
        & (rows["scenario"] == "clean")
    ].iloc[0]
    assert care_clean["macro_f1"] == pytest.approx(0.91)
    assert care_clean["coverage"] == pytest.approx(0.8)
    assert care_clean["classification_threshold"] == pytest.approx(0.5)
    assert (
        rows[
            (rows["method"] == "care_droid")
            & (rows["section"] == "robust")
        ]["scenario"].tolist()
        == ["api_event_dropout@0.5"]
    )


def test_collector_deduplicates_copied_summary_payloads(
    tmp_path: Path,
) -> None:
    payload = _care_summary()
    _write(tmp_path / "run" / "summary.yaml", payload)
    _write(tmp_path / "summary_copy.yaml", payload)
    rows = collect_metric_rows(tmp_path)
    assert len(rows[rows["section"] == "test"]) == 1


def test_seed_aggregation_reports_undefined_single_seed_std(
    tmp_path: Path,
) -> None:
    for seed, score in ((42, 0.90), (2024, 0.92)):
        _write(
            tmp_path / str(seed) / "summary.yaml",
            _care_summary(seed=seed, macro_f1=score),
        )
    rows = collect_metric_rows(tmp_path)
    aggregate = aggregate_metrics(rows)
    result = aggregate[
        (aggregate["method"] == "care_droid")
        & (aggregate["section"] == "test")
        & (aggregate["scenario"] == "clean")
        & (aggregate["metric"] == "macro_f1")
    ].iloc[0]
    assert result["mean"] == pytest.approx(0.91)
    assert result["count"] == 2

    one = aggregate_metrics(rows[rows["seed"].astype(str) == "42"])
    single = one[
        (one["section"] == "test")
        & (one["scenario"] == "clean")
        & (one["metric"] == "macro_f1")
    ].iloc[0]
    assert single["std"] is None


def test_primary_classification_table_uses_common_half_threshold(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "care" / "summary.yaml", _care_summary())
    _write(tmp_path / "baseline" / "summary.yaml", _baseline_summary())
    table = build_paper_classification_table(
        collect_metric_rows(tmp_path)
    )
    clean = table[
        (table["method"] == "care_droid")
        & (table["scenario"] == "clean")
    ].iloc[0]
    assert clean["macro_f1"] == pytest.approx(0.91)
    assert clean["classification_threshold"] == pytest.approx(0.5)
    assert clean["classification_protocol"] == (
        "binary_argmax_fixed_0_5_v1"
    )


def test_baseline_metric_schema_versions_match() -> None:
    assert COLLECTOR_METRIC_SCHEMA_VERSION == PRODUCER_METRIC_SCHEMA_VERSION
