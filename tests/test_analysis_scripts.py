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
from scripts.collect_experiment_results import aggregate_metrics, collect_metric_rows
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


def test_reliability_signal_diagnostics_excludes_auxiliary_joint_head():
    frame = pd.DataFrame(
        {
            "experiment": ["final"] * 4,
            "seed": ["42"] * 4,
            "diagnostic_file": ["gate_diagnostics.csv"] * 4,
            "split": ["test_clean"] * 4,
            "predicted_reliability_api": [0.2, 0.3, 0.8, 0.9],
            "api_integrity": [0.2, 0.4, 0.7, 0.9],
            "api_correct": [0, 0, 1, 1],
            "predicted_reliability_joint": [0.1, 0.2, 0.8, 0.9],
            "joint_correct": [0, 0, 1, 1],
        }
    )

    table = reliability_signal_diagnostics_table(frame, permutations=5)

    assert set(table["branch"]) == {"api"}


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
    legacy_path = output_dir / "test_raw_high_conflict.csv"
    legacy_path.write_text("sha256,label\nlegacy,0\n", encoding="utf-8")

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
    assert not legacy_path.exists()
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


def test_result_collector_reads_flat_summaries_and_deduplicates_copies(tmp_path):
    payload = """
run_identity:
  experiment_name: evidential_seed_42
  method_name: evidential_trusted_fusion
  seed: 42
  method_protocol_sha256: protocol123
  method_implementation_sha256: implementation456
test:
  macro_f1: 0.91
  conformal_empty_set_rate: 0.03
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
    assert row["method_protocol_sha256"] == "protocol123"
    assert row["method_implementation_sha256"] == "implementation456"
    assert row["conformal_empty_set_rate"] == pytest.approx(0.03)


def test_result_collector_normalizes_embedded_seed_method_names(tmp_path):
    for seed, macro_f1 in ((42, 0.90), (2024, 0.92), (3407, 0.91)):
        payload = {
            "run_identity": {
                "experiment_name": f"evidential_seed_{seed}",
                "method_name": f"evidential_seed_{seed}",
                "seed": seed,
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
