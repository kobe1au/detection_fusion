from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.analyze_reliability_evidence import (
    evidence_bin_effects_table,
    natural_degradation_subset_table,
    reliability_signal_diagnostics_table,
    reliability_table,
)
from fusion.train import METRIC_SUMMARY_SCHEMA_VERSION as PRODUCER_METRIC_SCHEMA_VERSION
from scripts.collect_experiment_results import (
    METRIC_SUMMARY_SCHEMA_VERSION as COLLECTOR_METRIC_SCHEMA_VERSION,
    aggregate_metrics,
    collect_metric_rows,
)
from scripts.build_natural_subset_csvs import build_subsets


def test_evidence_bin_effects_report_branch_accuracy_by_tercile():
    frame = pd.DataFrame(
        {
            "experiment": ["final_seed_42"] * 3,
            "seed": ["42"] * 3,
            "diagnostic_file": ["gate_diagnostics.csv"] * 3,
            "split": ["test_clean"] * 3,
            "api_integrity": [0.1, 0.5, 0.9],
            "api_correct": [0, 1, 1],
            "predicted_reliability_api": [0.2, 0.6, 0.8],
        }
    )

    table = evidence_bin_effects_table(frame, bin_scope="group")
    api_rows = table[
        (table["evidence"] == "api_integrity")
        & (table["branch"] == "api")
    ].set_index("evidence_bin")

    assert set(api_rows.index) == {"low", "mid", "high"}
    assert api_rows.loc["low", "branch_accuracy"] == pytest.approx(0.0)
    assert api_rows.loc["mid", "branch_accuracy"] == pytest.approx(1.0)
    assert api_rows.loc["high", "predicted_reliability_mean"] == pytest.approx(0.8)


def test_evidence_bin_effects_include_effective_integrity_and_discount_bins():
    frame = pd.DataFrame(
        {
            "experiment": ["final_seed_42"] * 3,
            "seed": ["42"] * 3,
            "diagnostic_file": ["gate_diagnostics.csv"] * 3,
            "split": ["test_clean"] * 3,
            "effective_api_integrity": [0.2, 0.6, 0.95],
            "discount_api": [0.1, 0.4, 0.9],
            "api_correct": [0, 1, 1],
            "predicted_reliability_api": [0.3, 0.5, 0.8],
        }
    )

    table = evidence_bin_effects_table(frame, bin_scope="group")

    assert "effective_api_integrity" in set(table["evidence"])
    discount_rows = table[
        (table["evidence"] == "discount_api")
        & (table["branch"] == "api")
    ].set_index("evidence_bin")
    assert set(discount_rows.index) == {"low", "mid", "high"}
    assert discount_rows.loc["low", "branch_accuracy"] == pytest.approx(0.0)


def test_reliability_table_marks_auc_and_ap_undefined_for_single_class_correctness():
    frame = pd.DataFrame(
        {
            "experiment": ["final_seed_42", "final_seed_42"],
            "seed": ["42", "42"],
            "diagnostic_file": ["gate_diagnostics.csv", "gate_diagnostics.csv"],
            "split": ["test_clean", "test_clean"],
            "predicted_reliability_api": [0.8, 0.9],
            "api_correct": [1, 1],
        }
    )

    table = reliability_table(frame)
    row = table.iloc[0]

    assert row["auc_defined"] == 0
    assert np.isnan(row["auc"])
    assert row["ap_defined"] == 0
    assert np.isnan(row["ap"])


def test_reliability_signal_diagnostics_reports_certainty_and_permutation_gap():
    frame = pd.DataFrame(
        {
            "experiment": ["final"] * 6,
            "seed": ["42"] * 6,
            "diagnostic_file": ["gate_diagnostics.csv"] * 6,
            "split": ["test_clean"] * 6,
            "predicted_reliability_api": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "api_integrity": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
            "uncertainty_proxy_api": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            "api_correct": [0, 0, 0, 1, 1, 1],
        }
    )
    row = reliability_signal_diagnostics_table(
        frame, permutations=20, random_seed=7
    ).iloc[0]
    assert row["reliability_auc"] == pytest.approx(1.0)
    assert row["intrinsic_integrity_auc"] == pytest.approx(1.0)
    assert row["evidential_certainty_auc"] == pytest.approx(1.0)
    assert row["reliability_permutation_gap"] > 0.0


def test_natural_degradation_subset_table_reports_low_integrity_final_metrics():
    frame = pd.DataFrame(
        {
            "experiment": ["final_seed_42"] * 6,
            "seed": ["42"] * 6,
            "diagnostic_file": ["gate_diagnostics.csv"] * 6,
            "split": ["test_clean"] * 6,
            "api_integrity": [0.05, 0.10, 0.20, 0.70, 0.80, 0.90],
            "label": [0, 1, 1, 0, 1, 0],
            "prob_malware": [0.20, 0.80, 0.40, 0.10, 0.90, 0.30],
            "pred": [0, 1, 0, 0, 1, 0],
        }
    )

    table = natural_degradation_subset_table(frame, quantile=1.0 / 3.0, min_count=1)
    row = table[table["subset"] == "api_low_integrity"].iloc[0]

    assert row["count"] == 2
    assert 0.10 < row["threshold"] < 0.20
    assert row["acc"] == pytest.approx(1.0)
    assert row["macro_f1"] == pytest.approx(1.0)


def test_natural_degradation_analysis_rejects_fractional_labels():
    frame = pd.DataFrame(
        {
            "experiment": ["final_seed_42", "final_seed_42"],
            "seed": ["42", "42"],
            "diagnostic_file": ["gate_diagnostics.csv"] * 2,
            "split": ["test_clean", "test_clean"],
            "api_integrity": [0.1, 0.9],
            "label": [0.5, 1.0],
            "prob_malware": [0.2, 0.8],
            "pred": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="analysis label"):
        natural_degradation_subset_table(frame, quantile=0.25, min_count=1)


def test_natural_degradation_subset_table_reports_conflict_and_acceptance_risk():
    frame = pd.DataFrame(
        {
            "experiment": ["final_seed_42"] * 6,
            "seed": ["42"] * 6,
            "diagnostic_file": ["gate_diagnostics.csv"] * 6,
            "split": ["test_clean"] * 6,
            "raw_conflict": [0.01, 0.02, 0.30, 0.40, 0.80, 0.90],
            "acceptance_score": [0.95, 0.85, 0.70, 0.40, 0.20, 0.10],
            "label": [0, 1, 1, 0, 1, 0],
            "prob_malware": [0.20, 0.80, 0.40, 0.60, 0.30, 0.70],
            "pred": [0, 1, 0, 1, 0, 1],
            "rejected": [0, 0, 0, 1, 1, 1],
        }
    )

    table = natural_degradation_subset_table(frame, quantile=1.0 / 3.0, min_count=1)
    conflict_row = table[table["subset"] == "raw_high_conflict"].iloc[0]
    acceptance_row = table[table["subset"] == "low_acceptance"].iloc[0]

    assert conflict_row["count"] == 2
    assert conflict_row["error_rate"] == pytest.approx(1.0)
    assert acceptance_row["count"] == 2
    assert acceptance_row["rejection_rate"] == pytest.approx(1.0)
    assert np.isnan(acceptance_row["selective_risk"])


def _natural_subset_inputs(tmp_path):
    ids = [f"sample-{index}" for index in range(6)]
    diagnostics_path = tmp_path / "gate_diagnostics.csv"
    test_csv_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "sid": ids,
            "split": ["test_clean"] * len(ids),
            "effective_api_integrity": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "api_graph_anchor_support": [0.2, 0.1, 0.3, 0.8, 0.9, 0.7],
            "predictive_conflict": [0.1, 0.2, 0.8, 0.3, 0.9, 0.4],
            "acceptance_score": [0.9, 0.8, 0.2, 0.7, 0.1, 0.6],
        }
    ).to_csv(diagnostics_path, index=False)
    pd.DataFrame(
        {
            "sha256": ids,
            "label": [0, 1, 0, 1, 0, 1],
            "year": [2023] * len(ids),
        }
    ).to_csv(test_csv_path, index=False)
    return diagnostics_path, test_csv_path


def test_natural_subset_builder_writes_predictive_conflict_with_provenance(tmp_path):
    diagnostics_path, test_csv_path = _natural_subset_inputs(tmp_path)
    output_dir = tmp_path / "subsets"
    output_dir.mkdir()

    summary = build_subsets(
        diagnostics_path=diagnostics_path,
        test_csv_path=test_csv_path,
        output_dir=output_dir,
        split="test_clean",
        quantile=1.0 / 3.0,
        min_count=1,
    )

    assert {row["subset"] for row in summary} == {
        "api_low_effective_integrity",
        "api_graph_low_support",
        "predictive_high_conflict",
        "low_acceptance",
    }
    assert (output_dir / "test_predictive_high_conflict.csv").is_file()
    manifest = json.loads(
        (output_dir / "subset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3
    assert manifest["sample_count"] == 6
    assert len(manifest["diagnostics_sha256"]) == 64
    assert len(manifest["test_csv_sha256"]) == 64
    assert len(manifest["subsets"]) == 4
    assert all(len(record["csv_sha256"]) == 64 for record in manifest["subsets"])


def test_natural_subset_builder_rejects_incomplete_diagnostics(tmp_path):
    diagnostics_path, test_csv_path = _natural_subset_inputs(tmp_path)
    diagnostics = pd.read_csv(diagnostics_path).iloc[:-1]
    diagnostics.to_csv(diagnostics_path, index=False)

    with pytest.raises(ValueError, match="one clean diagnostic row for every test sample"):
        build_subsets(
            diagnostics_path=diagnostics_path,
            test_csv_path=test_csv_path,
            output_dir=tmp_path / "subsets",
            split="test_clean",
            quantile=1.0 / 3.0,
            min_count=1,
        )


def test_aggregate_metrics_groups_seed_runs_by_method(tmp_path):
    for seed, macro_f1 in ((42, 0.7), (2024, 0.9)):
        run_dir = tmp_path / f"final_seed_{seed}" / str(seed)
        run_dir.mkdir(parents=True)
        (run_dir / "summary.yaml").write_text(
            f"""
metric_schema_version: 5
run_identity:
  experiment_name: final_seed_{seed}
  method_name: final_seed_{seed}
  seed: {seed}
  method_protocol_id: trusted-fusion-v1
  method_protocol_sha256: shared-protocol
  method_implementation_sha256: shared-implementation
test:
  macro_f1: {macro_f1}
  acc: 0.8
  rejection_threshold: 0.{seed % 10}
  sample_count: 123
""",
            encoding="utf-8",
        )

    metrics = collect_metric_rows(tmp_path)
    aggregate = aggregate_metrics(metrics)
    row = aggregate[
        (aggregate["method"] == "final")
        & (aggregate["section"] == "test")
        & (aggregate["scenario"] == "test")
        & (aggregate["metric"] == "macro_f1")
    ].iloc[0]

    assert row["mean"] == pytest.approx(0.8)
    assert row["count"] == 2
    assert "rejection_threshold" not in set(aggregate["metric"])
    assert "sample_count" not in set(aggregate["metric"])


def test_aggregate_metrics_marks_single_run_std_undefined_and_serializable(tmp_path):
    metrics = pd.DataFrame.from_records(
        [
            {
                "experiment": "final_seed_42",
                "method": "final",
                "seed": "42",
                "method_protocol_id": "trusted-fusion-v1",
                "method_protocol_sha256": "shared-protocol",
                "method_implementation_sha256": "shared-implementation",
                "section": "test",
                "scenario": "test",
                "macro_f1": 0.91,
            }
        ]
    )

    aggregate = aggregate_metrics(metrics)
    row = aggregate[aggregate["metric"] == "macro_f1"].iloc[0]

    assert row["mean"] == pytest.approx(0.91)
    assert row["count"] == 1
    assert row["std"] is None

    yaml_rows = yaml.safe_load(yaml.safe_dump(aggregate.to_dict("records")))
    assert yaml_rows[0]["std"] is None

    csv_path = tmp_path / "aggregate.csv"
    aggregate.to_csv(csv_path, index=False)
    csv_rows = pd.read_csv(csv_path)
    assert pd.isna(csv_rows.loc[0, "std"])


def test_result_collector_reads_flat_summaries_and_deduplicates_copies(tmp_path):
    payload = """
metric_schema_version: 5
run_identity:
  experiment_name: evidential_seed_42
  method_name: evidential_trusted_fusion
  seed: 42
  method_protocol_id: trusted-fusion-v1
  method_protocol_sha256: protocol123
  method_implementation_sha256: implementation456
  selective_prediction_mode: risk_control
  selective_score_type: msp
test:
  macro_f1: 0.91
  malware_fn_risk_aurc: 0.012
  conformal_empty_set_rate: 0.03
  risk_control_guarantee_type: expected_crc
  risk_control_guarantee_scope: exchangeable_expected_risk
  risk_control_calibration_feasible: true
  risk_control_calibration_corrected_risk: 0.04
"""
    nested = tmp_path / "evidential_seed_42" / "42"
    nested.mkdir(parents=True)
    (nested / "summary.yaml").write_text(payload, encoding="utf-8")
    (tmp_path / "summary_evidential_seed_42.yaml").write_text(payload, encoding="utf-8")

    metrics = collect_metric_rows(tmp_path)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["experiment"] == "evidential_seed_42"
    assert row["method"] == "evidential_trusted_fusion"
    assert str(row["seed"]) == "42"
    assert row["method_protocol_id"] == "trusted-fusion-v1"
    assert row["method_protocol_sha256"] == "protocol123"
    assert row["method_implementation_sha256"] == "implementation456"
    assert row["conformal_empty_set_rate"] == pytest.approx(0.03)
    assert row["malware_fn_risk_aurc"] == pytest.approx(0.012)
    assert row["selective_score_type"] == "msp"
    assert row["risk_control_guarantee_type"] == "expected_crc"
    assert row["risk_control_guarantee_scope"] == "exchangeable_expected_risk"
    assert bool(row["risk_control_calibration_feasible"]) is True
    assert row["risk_control_calibration_corrected_risk"] == pytest.approx(0.04)


def test_result_collector_normalizes_embedded_seed_method_names(tmp_path):
    for seed, macro_f1 in ((42, 0.90), (2024, 0.92), (3407, 0.91)):
        payload = {
            "metric_schema_version": 5,
            "run_identity": {
                "experiment_name": f"evidential_seed_{seed}",
                "method_name": f"evidential_seed_{seed}",
                "seed": seed,
                "method_protocol_id": "trusted-fusion-v1",
                "method_protocol_sha256": "shared-protocol",
                "method_implementation_sha256": "shared-implementation",
            },
            "test": {
                "macro_f1": macro_f1,
                "classification_threshold": 0.4,
                "risk_control_acceptance_rate": 0.8,
            },
        }
        path = tmp_path / f"summary_evidential_seed_{seed}.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    metrics = collect_metric_rows(tmp_path)
    aggregate = aggregate_metrics(metrics)

    assert set(metrics["method"]) == {"evidential"}
    row = aggregate[
        (aggregate["method"] == "evidential")
        & (aggregate["section"] == "test")
        & (aggregate["scenario"] == "test")
        & (aggregate["metric"] == "macro_f1")
    ].iloc[0]
    assert row["mean"] == pytest.approx(0.91)
    assert row["count"] == 3
    assert "classification_threshold" in set(aggregate["metric"])
    assert "risk_control_acceptance_rate" in set(aggregate["metric"])


def test_result_collector_rejects_legacy_metric_semantics(tmp_path):
    path = tmp_path / "summary_legacy.yaml"
    path.write_text(
        "test:\n  macro_f1: 0.99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metric_schema_version"):
        collect_metric_rows(tmp_path)


def test_result_collector_requires_embedded_identity_in_current_schema(tmp_path):
    path = tmp_path / "summary_final_seed_42.yaml"
    path.write_text(
        "metric_schema_version: 5\ntest:\n  macro_f1: 0.99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required run_identity"):
        collect_metric_rows(tmp_path)


def test_metric_summary_schema_versions_match():
    assert PRODUCER_METRIC_SCHEMA_VERSION == 5
    assert COLLECTOR_METRIC_SCHEMA_VERSION == PRODUCER_METRIC_SCHEMA_VERSION


@pytest.mark.parametrize("version", [1, 2, 3, 4, "5"])
def test_result_collector_rejects_wrong_metric_schema_version(tmp_path, version):
    payload = {
        "metric_schema_version": version,
        "run_identity": {
            "experiment_name": "wrong_schema",
            "method_name": "wrong_schema",
            "seed": 42,
            "method_protocol_id": "trusted-fusion-v1",
            "method_protocol_sha256": "protocol",
            "method_implementation_sha256": "implementation",
        },
        "test": {"macro_f1": 0.9},
    }
    path = tmp_path / "summary.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected=5"):
        collect_metric_rows(tmp_path)


def test_result_collector_uses_only_canonical_sections_and_embedded_extra_eval(tmp_path):
    payload = {
        "metric_schema_version": 5,
        "run_identity": {
            "experiment_name": "canonical",
            "method_name": "canonical",
            "seed": 42,
            "method_protocol_id": None,
            "method_protocol_sha256": "protocol",
            "method_implementation_sha256": "implementation",
        },
        "val_posthoc_calibration": {"macro_f1": 0.8},
        "val_calibration": {"macro_f1": 0.1},
        "extra_eval": {"natural_subset": {"macro_f1": 0.7}},
    }
    run_dir = tmp_path / "canonical" / "42"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    (run_dir / "metrics_extra_eval.json").write_text(
        json.dumps({"natural_subset": {"macro_f1": 0.2}}), encoding="utf-8"
    )

    metrics = collect_metric_rows(tmp_path)

    posthoc = metrics[metrics["section"] == "val_posthoc_calibration"]
    assert len(posthoc) == 1
    assert posthoc.iloc[0]["macro_f1"] == pytest.approx(0.8)
    extra = metrics[metrics["section"] == "extra_eval"]
    assert len(extra) == 1
    assert extra.iloc[0]["macro_f1"] == pytest.approx(0.7)
    assert "extra_eval_json" not in set(metrics["section"])
