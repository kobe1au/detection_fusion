from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_reliability_evidence import (
    evidence_bin_effects_table,
    reliability_table,
)
from scripts.collect_experiment_results import aggregate_metrics, collect_metric_rows


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
