from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fusion.train import (
    METRIC_SUMMARY_SCHEMA_VERSION as PRODUCER_METRIC_SCHEMA_VERSION,
)
from scripts.analyze_reliability_evidence import (
    METRIC_SUMMARY_SCHEMA_VERSION as ANALYZER_METRIC_SCHEMA_VERSION,
    _read_diagnostics,
    competence_bin_table,
    competence_table,
    pairwise_ordering_table,
)
from scripts.build_natural_subset_csvs import build_subsets
from scripts.collect_experiment_results import (
    METRIC_SUMMARY_SCHEMA_VERSION as COLLECTOR_METRIC_SCHEMA_VERSION,
    aggregate_metrics,
    build_paper_forced_classification_table,
    collect_metric_rows,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bound_summary(
    diagnostics_path: Path,
    *,
    metric_schema_version: int = PRODUCER_METRIC_SCHEMA_VERSION,
    protocol_id: str | None = "tcp_joint_anchor_crc_v1",
) -> None:
    frame = pd.read_csv(diagnostics_path)
    protocol_sha = "a" * 64
    implementation_sha = "b" * 64
    artifact_key = diagnostics_path.stem
    payload = {
        "metric_schema_version": metric_schema_version,
        "run_identity": {
            "experiment_name": "competence_anchored_seed_42",
            "method_name": "competence_anchored_seed_42",
            "seed": 42,
            "method_protocol_id": protocol_id,
            "method_protocol_sha256": protocol_sha,
            "method_implementation_sha256": implementation_sha,
        },
        "validation_split": {
            "role_assignment_semantic_sha256": "c" * 64,
        },
        "diagnostic_artifacts": {
            artifact_key: {
                "path": str(diagnostics_path.resolve()),
                "sha256": _sha256(diagnostics_path),
                "splits": sorted(
                    frame["split"].astype(str).unique().tolist()
                ),
                "method_protocol_id": protocol_id,
                "method_protocol_sha256": protocol_sha,
                "method_implementation_sha256": implementation_sha,
                "pipeline_model_state_sha256": "d" * 64,
                "pipeline_decision_metadata_sha256": "e" * 64,
                "validation_role_assignment_semantic_sha256": "c" * 64,
            }
        },
    }
    (diagnostics_path.parent / "summary.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _competence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": ["main"] * 4,
            "seed": ["42"] * 4,
            "diagnostic_file": ["gate_diagnostics.csv"] * 4,
            "split": ["test_clean"] * 4,
            "label": [0, 1, 0, 1],
            "api_alive": [1, 1, 1, 1],
            "api_prob": [0.1, 0.8, 0.4, 0.7],
            "api_correct": [1, 1, 1, 1],
            "predicted_competence_api": [0.9, 0.8, 0.6, 0.7],
            "graph_alive": [1, 1, 1, 1],
            "graph_prob": [0.2, 0.7, 0.6, 0.6],
            "graph_correct": [1, 1, 0, 1],
            "predicted_competence_graph": [0.8, 0.7, 0.4, 0.6],
            "manifest_alive": [1, 1, 1, 1],
            "manifest_prob": [0.3, 0.6, 0.2, 0.9],
            "manifest_correct": [1, 1, 1, 1],
            "predicted_competence_manifest": [0.7, 0.6, 0.8, 0.9],
            "joint_alive": [1, 1, 1, 1],
            "joint_prob": [0.05, 0.95, 0.25, 0.85],
            "joint_correct": [1, 1, 1, 1],
            "predicted_competence_joint": [0.95, 0.95, 0.75, 0.85],
        }
    )


def test_analyzer_reads_only_hash_bound_schema_v13_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate_diagnostics.csv"
    pd.DataFrame(
        {
            "sid": ["a"],
            "split": ["test_clean"],
            "predicted_competence_api": [0.8],
            "api_correct": [1],
        }
    ).to_csv(path, index=False)
    _write_bound_summary(path)

    frame = _read_diagnostics([path])

    assert frame["experiment"].tolist() == ["competence_anchored_seed_42"]
    assert frame["method"].tolist() == ["competence_anchored_seed_42"]
    assert frame["seed"].tolist() == ["42"]
    assert frame["method_protocol_sha256"].tolist() == ["a" * 64]
    assert frame["method_implementation_sha256"].tolist() == ["b" * 64]


def test_analyzer_accepts_bound_extra_eval_and_protocol_less_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate_diagnostics_extra_eval.csv"
    pd.DataFrame(
        {
            "sid": ["a"],
            "split": ["natural_branch_disagreement"],
        }
    ).to_csv(path, index=False)
    _write_bound_summary(path, protocol_id=None)

    frame = _read_diagnostics([path])

    assert frame["diagnostic_file"].tolist() == [
        "gate_diagnostics_extra_eval.csv"
    ]
    assert frame["method_protocol_id"].tolist() == [""]


def test_analyzer_rejects_legacy_schema_tampering_and_split_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate_diagnostics.csv"
    original = pd.DataFrame({"sid": ["a"], "split": ["test_clean"]})
    original.to_csv(path, index=False)
    _write_bound_summary(path, metric_schema_version=12)
    with pytest.raises(ValueError, match="expected 14"):
        _read_diagnostics([path])

    _write_bound_summary(path)
    original.assign(split=["test_changed"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="changed after"):
        _read_diagnostics([path])


def test_analyzer_requires_one_identity_per_validation_role_and_disjoint_roles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate_diagnostics.csv"
    pd.DataFrame(
        {
            "sid": ["a", "b", "c", "d"],
            "split": [
                "val_model_selection",
                "val_model_selection",
                "val_decision_calibration",
                "val_decision_calibration",
            ],
        }
    ).to_csv(path, index=False)
    _write_bound_summary(path)
    frame = _read_diagnostics([path])
    selection = frame[frame["split"] == "val_model_selection"]
    decision = frame[frame["split"] == "val_decision_calibration"]
    assert not selection["sid"].duplicated().any()
    assert not decision["sid"].duplicated().any()
    assert set(selection["sid"]).isdisjoint(decision["sid"])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            {
                "sid": ["a", "a", "c"],
                "split": [
                    "val_model_selection",
                    "val_model_selection",
                    "val_decision_calibration",
                ],
            },
            "duplicate sample ids",
        ),
        (
            {
                "sid": ["a", "a"],
                "split": [
                    "val_model_selection",
                    "val_decision_calibration",
                ],
            },
            "validation roles overlap",
        ),
    ],
)
def test_analyzer_rejects_invalid_validation_role_identities(
    tmp_path: Path,
    rows: dict[str, list[str]],
    message: str,
) -> None:
    path = tmp_path / "gate_diagnostics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _write_bound_summary(path)

    with pytest.raises(ValueError, match=message):
        _read_diagnostics([path])


def test_competence_table_audits_continuous_tcp_not_binary_correctness() -> None:
    table = competence_table(_competence_frame())
    api = table[table["branch"] == "api"].iloc[0]
    assert api["count"] == 4
    assert api["tcp_mse"] == pytest.approx(0.0)
    assert api["tcp_mae"] == pytest.approx(0.0)
    assert api["tcp_pearson"] == pytest.approx(1.0)
    assert api["branch_accuracy"] == pytest.approx(1.0)


def test_competence_analyzer_handles_pairwise_ordering_and_bins() -> None:
    frame = _competence_frame()
    ordering = pairwise_ordering_table(frame)
    assert not ordering.empty
    assert set(ordering.columns) >= {
        "left_branch",
        "right_branch",
        "ordering_accuracy",
    }
    bins = competence_bin_table(frame, bins=5)
    assert not bins.empty
    assert set(bins["branch"]) == {
        "api",
        "graph",
        "manifest",
        "joint",
    }
    assert (
        (ordering["left_branch"] == "manifest")
        & (ordering["right_branch"] == "joint")
    ).any()
    assert bool(((bins["mean_competence"] >= 0) & (bins["mean_competence"] <= 1)).all())


def _write_registered_natural_source_summary(path: Path) -> None:
    _write_bound_summary(path)


def _natural_subset_inputs(tmp_path: Path) -> tuple[Path, Path]:
    calibration_ids = [f"calibration-{index}" for index in range(6)]
    test_ids = [f"sample-{index}" for index in range(6)]
    diagnostics_path = tmp_path / "gate_diagnostics.csv"
    test_csv_path = tmp_path / "test.csv"
    calibration = pd.DataFrame(
        {
            "sid": calibration_ids,
            "split": ["val_model_selection"] * 6,
            "label": [0, 1, 0, 1, 0, 1],
            "api_alive": [1] * 6,
            "graph_alive": [1] * 6,
            "manifest_alive": [1] * 6,
            "api_pred": [0, 1, 0, 1, 0, 1],
            "graph_pred": [0, 1, 0, 1, 0, 1],
            "manifest_pred": [0, 1, 0, 1, 0, 1],
            "api_prob": [0.10, 0.90, 0.10, 0.10, 0.05, 0.05],
            "graph_prob": [0.10, 0.90, 0.20, 0.40, 0.50, 0.50],
            "manifest_prob": [0.10, 0.90, 0.30, 0.70, 0.90, 0.95],
            "predicted_competence_api": [0.70, 0.65, 0.60, 0.55, 0.50, 0.45],
            "predicted_competence_graph": [0.75] * 6,
            "predicted_competence_manifest": [0.80, 0.85, 0.90, 0.95, 1.00, 1.00],
        }
    )
    target = pd.DataFrame(
        {
            "sid": test_ids,
            "split": ["test_clean"] * 6,
            "label": [0, 1, 0, 1, 0, 1],
            "api_alive": [1] * 6,
            "graph_alive": [1] * 6,
            "manifest_alive": [1] * 6,
            "api_pred": [1, 1, 0, 1, 0, 1],
            "graph_pred": [0, 0, 0, 1, 1, 1],
            "manifest_pred": [0, 1, 1, 1, 0, 1],
            "api_prob": [0.95, 0.90, 0.10, 0.90, 0.10, 0.90],
            "graph_prob": [0.05, 0.10, 0.10, 0.90, 0.90, 0.90],
            "manifest_prob": [0.05, 0.90, 0.90, 0.90, 0.10, 0.90],
            "predicted_competence_api": [0.05, 0.90, 0.90, 0.10, 0.80, 0.90],
            "predicted_competence_graph": [0.90, 0.10, 0.90, 0.90, 0.20, 0.90],
            "predicted_competence_manifest": [0.85, 0.85, 0.10, 0.80, 0.75, 0.90],
        }
    )
    pd.concat([calibration, target], ignore_index=True).to_csv(
        diagnostics_path, index=False
    )
    _write_registered_natural_source_summary(diagnostics_path)
    pd.DataFrame(
        {
            "sha256": test_ids,
            "label": [0, 1, 0, 1, 0, 1],
            "year": [2023] * 6,
        }
    ).to_csv(test_csv_path, index=False)
    return diagnostics_path, test_csv_path


def test_natural_subset_builder_uses_validation_frozen_competence_protocol(
    tmp_path: Path,
) -> None:
    diagnostics, test_csv = _natural_subset_inputs(tmp_path)
    output = tmp_path / "subsets"
    summary = build_subsets(
        diagnostics_path=diagnostics,
        test_csv_path=test_csv,
        output_dir=output,
        min_count=1,
        min_calibration_count=1,
    )

    assert {row["subset"] for row in summary} == {
        "branch_disagreement",
        "api_only_wrong",
        "graph_only_wrong",
        "manifest_only_wrong",
        "competence_imbalance",
        "high_cross_modal_conflict",
    }
    manifest = json.loads(
        (output / "subset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 9
    assert (
        manifest["protocol_id"]
        == "competence_validation_natural_difficulty_v1"
    )
    assert manifest["calibration_split"] == "val_model_selection"
    assert manifest["target_split"] == "test_clean"
    assert manifest["protocol_guarantees"][
        "model_selection_disjoint_from_decision_calibration"
    ] is True
    assert set(manifest["thresholds"]) == {
        "competence_imbalance",
        "high_cross_modal_conflict",
    }
    assert all(
        item["source_split"] == "val_model_selection"
        for item in manifest["thresholds"].values()
    )
    assert (
        manifest["source_run"]["method_protocol_id"]
        == "tcp_joint_anchor_crc_v1"
    )


def test_natural_subset_thresholds_ignore_test_score_changes(
    tmp_path: Path,
) -> None:
    diagnostics, test_csv = _natural_subset_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_subsets(
        diagnostics_path=diagnostics,
        test_csv_path=test_csv,
        output_dir=first,
        min_count=1,
        min_calibration_count=1,
    )
    first_manifest = json.loads(
        (first / "subset_manifest.json").read_text(encoding="utf-8")
    )

    frame = pd.read_csv(diagnostics)
    target = frame["split"] == "test_clean"
    frame.loc[target, "predicted_competence_api"] = [0, 1, 0, 1, 0, 1]
    frame.loc[target, "api_prob"] = [0.99, 0.99, 0.01, 0.99, 0.01, 0.99]
    frame.to_csv(diagnostics, index=False)
    _write_registered_natural_source_summary(diagnostics)
    build_subsets(
        diagnostics_path=diagnostics,
        test_csv_path=test_csv,
        output_dir=second,
        min_count=1,
        min_calibration_count=1,
    )
    second_manifest = json.loads(
        (second / "subset_manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["thresholds"] == second_manifest["thresholds"]


def test_natural_subset_builder_rejects_duplicate_calibration_identity(
    tmp_path: Path,
) -> None:
    diagnostics, test_csv = _natural_subset_inputs(tmp_path)
    frame = pd.read_csv(diagnostics)
    duplicate = frame[frame["split"] == "val_model_selection"].iloc[[0]]
    pd.concat([frame, duplicate], ignore_index=True).to_csv(
        diagnostics, index=False
    )
    _write_registered_natural_source_summary(diagnostics)
    with pytest.raises(ValueError, match="duplicate sample ids"):
        build_subsets(
            diagnostics_path=diagnostics,
            test_csv_path=test_csv,
            output_dir=tmp_path / "subsets",
            min_count=1,
            min_calibration_count=1,
        )


def _summary_payload(
    *,
    seed: int = 42,
    macro_f1: float = 0.9,
) -> dict:
    return {
        "metric_schema_version": 14,
        "run_identity": {
            "experiment_name": f"competence_anchored_seed_{seed}",
            "method_name": f"competence_anchored_seed_{seed}",
            "seed": seed,
            "method_protocol_id": "tcp_joint_anchor_crc_v1",
            "method_protocol_sha256": "shared-protocol",
            "method_implementation_sha256": "shared-implementation",
            "selective_prediction_mode": "risk_control",
            "selective_score_type": "malware_fn_probability_anchor",
        },
        "val_model_selection": {"macro_f1": macro_f1 - 0.01},
        "val_model_selection_threshold_fit": {
            "macro_f1": macro_f1 - 0.01
        },
        "val_decision_calibration": {"macro_f1": macro_f1 - 0.02},
        "stage_b_training": {
            "enabled": True,
            "deployment": "anchored_joint_late",
            "classification_threshold": {
                "threshold": 0.42,
                "locked_by_stage_b": True,
            },
            "competence": {
                "diagnostics": {
                    "val_model_selection_clean": {
                        "api": {
                            "defined": True,
                            "num_rows": 10,
                            "tcp_mse": 0.02,
                            "tcp_mae": 0.10,
                        }
                    }
                }
            },
            "router": {
                "selected": {
                    "degradation_loss_weight": 0.1,
                    "robust_mean_macro_f1": macro_f1 - 0.03,
                }
            },
        },
        "classification_threshold": {
            "threshold": 0.42,
            "locked_by_stage_b": True,
        },
        "test": {
            "macro_f1": macro_f1,
            "fixed_0_5_macro_f1": macro_f1 - 0.02,
            "fixed_0_5_acc": 0.88,
            "fixed_0_5_f1_pos": 0.87,
            "fixed_0_5_recall_pos": 0.86,
            "auc": 0.95,
        },
    }


def test_collector_reads_schema14_two_role_and_stage_b_sections(
    tmp_path: Path,
) -> None:
    payload = _summary_payload()
    run_dir = tmp_path / "main" / "42"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    # A convenience copy must not double-count the run.
    (tmp_path / "summary_copy.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )

    metrics = collect_metric_rows(tmp_path)

    assert set(metrics["section"]) >= {
        "val_model_selection",
        "val_decision_calibration",
        "stage_b_lifecycle",
        "stage_b_router",
        "stage_b_competence_diagnostics",
        "test",
    }
    assert len(metrics[metrics["section"] == "test"]) == 1
    test = metrics[metrics["section"] == "test"].iloc[0]
    assert test["selective_score_type"] == "malware_fn_probability_anchor"
    competence = metrics[
        metrics["section"] == "stage_b_competence_diagnostics"
    ].iloc[0]
    assert competence["scenario"] == "val_model_selection_clean/api"
    assert competence["tcp_mse"] == pytest.approx(0.02)


def test_collector_rejects_retired_three_role_schema14_sections(
    tmp_path: Path,
) -> None:
    payload = _summary_payload()
    payload["val_selection"] = payload.pop("val_model_selection")
    payload["val_posthoc_calibration"] = payload.pop(
        "val_model_selection_threshold_fit"
    )
    payload["val_conformal_calibration"] = payload.pop(
        "val_decision_calibration"
    )
    run_dir = tmp_path / "legacy" / "42"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="two-role validation"):
        collect_metric_rows(tmp_path)


def test_collector_aggregates_seeds_and_marks_single_std_undefined(
    tmp_path: Path,
) -> None:
    for seed, score in ((42, 0.7), (2024, 0.9)):
        path = tmp_path / str(seed)
        path.mkdir()
        (path / "summary.yaml").write_text(
            yaml.safe_dump(_summary_payload(seed=seed, macro_f1=score)),
            encoding="utf-8",
        )
    metrics = collect_metric_rows(tmp_path)
    aggregate = aggregate_metrics(metrics)
    row = aggregate[
        (aggregate["method"] == "competence_anchored")
        & (aggregate["section"] == "test")
        & (aggregate["metric"] == "macro_f1")
    ].iloc[0]
    assert row["mean"] == pytest.approx(0.8)
    assert row["count"] == 2

    one = aggregate_metrics(metrics[metrics["seed"].astype(str) == "42"])
    single = one[
        (one["section"] == "test") & (one["metric"] == "macro_f1")
    ].iloc[0]
    assert single["std"] is None


def test_paper_forced_classification_uses_fixed_point_five_metrics() -> None:
    metrics = pd.DataFrame.from_records(
        [
            {
                "experiment": "competence_anchored_seed_42",
                "method": "competence_anchored",
                "seed": "42",
                "section": "test",
                "scenario": "test",
                "classification_threshold": 0.37,
                "macro_f1": 0.93,
                "acc": 0.94,
                "fixed_0_5_macro_f1": 0.88,
                "fixed_0_5_acc": 0.89,
                "fixed_0_5_f1_pos": 0.87,
                "fixed_0_5_recall_pos": 0.86,
                "auc": 0.96,
            }
        ]
    )
    table = build_paper_forced_classification_table(metrics)
    assert table.iloc[0]["macro_f1"] == pytest.approx(0.88)
    assert table.iloc[0]["classification_threshold"] == pytest.approx(0.5)


def test_metric_schema_versions_match_and_old_summaries_fail(
    tmp_path: Path,
) -> None:
    assert PRODUCER_METRIC_SCHEMA_VERSION == 14
    assert COLLECTOR_METRIC_SCHEMA_VERSION == 14
    assert ANALYZER_METRIC_SCHEMA_VERSION == 14
    path = tmp_path / "summary.yaml"
    payload = _summary_payload()
    payload["metric_schema_version"] = 13
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected=14"):
        collect_metric_rows(tmp_path)
