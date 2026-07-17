from __future__ import annotations

import csv
from pathlib import Path

import pytest
import pandas as pd
import torch
from torch_geometric.data import Data

from fusion.constants import EvidenceIndex
from fusion.dataset import FatalDatasetConfigError, RobustTriModalDataset
from fusion.evidence import build_evidence
from fusion.manifest_features import DEFAULT_CATEGORIES, vectorize_manifest_record
from fusion.perturbations import apply_manifest_permission_mask
from fusion.quality import (
    OBSERVABLE_REQUIRED_FIELDS,
    OBSERVABLE_SCHEMA_VERSION,
    OBSERVABLE_SIGNAL_FIELDS,
    compute_api_graph_anchor_support,
    compute_api_integrity_v2,
    compute_graph_integrity_v2,
    compute_manifest_code_support_and_conflict,
    compute_manifest_integrity_v2,
    compute_raw_alive,
    refresh_observable_metadata,
    refresh_observable_signals,
)
from extract.extract_graph_api import ApiEvent, select_api_events
from scripts.build_tri_modal_pts_direct import build_observable_payload
from scripts.diagnose_observable_signals import _distribution_table, _output_checks, _trend_table


def _old_list_pt(tmp_path: Path, sid: str = "sample") -> tuple[Path, Path]:
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    torch.save(
        [
            {
                "call_x": torch.ones(2, 8),
                "call_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
                "api_ids": torch.tensor([1, 2], dtype=torch.long),
                "api_type_ids": torch.tensor([1, 0], dtype=torch.long),
                "api_sensitive_mask": torch.zeros(2),
                "api_method_index": torch.tensor([0, 1], dtype=torch.long),
                "api_in_graph_mask": torch.ones(2),
                "method_api_edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                "manifest_x": torch.ones(16),
                "manifest_permission_ids": torch.tensor([1], dtype=torch.long),
                "manifest_intent_ids": torch.tensor([1], dtype=torch.long),
                "manifest_category_counts": torch.ones(12),
                "manifest_stats": torch.ones(11),
                "q_manifest": torch.tensor([1.0]),
                "pert_manifest": torch.tensor([0.0]),
            }
        ],
        pt_dir / f"{sid}.pt",
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1})
    return pt_dir, csv_path


def test_observable_schema_strict_missing_fields(tmp_path: Path):
    pt_dir, csv_path = _old_list_pt(tmp_path, "strict_missing")
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )
    with pytest.raises(FatalDatasetConfigError, match="top-level mapping"):
        dataset[0]


def test_observable_schema_strict_accepts_complete_merged_payload(tmp_path: Path):
    pt_dir = tmp_path / "strict_pts"
    pt_dir.mkdir()
    sid = "strict_complete"
    dex = {
        "call_x": torch.ones(2, 8),
        "call_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
        "api_ids": torch.tensor([1, 2], dtype=torch.long),
        "api_type_ids": torch.tensor([1, 0], dtype=torch.long),
        "api_sensitive_mask": torch.zeros(2),
        "api_method_index": torch.tensor([0, 1], dtype=torch.long),
        "api_in_graph_mask": torch.ones(2),
        "method_api_edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    }
    code_payload = {
        "dex_list": [dex],
        "observable_metadata": {
            "api_parse_ok": True,
            "api_parse_error": "",
            "api_event_count_raw": 2,
            "api_event_count_kept": 2,
            "api_event_count_before_method_budget": 2,
            "api_event_count_after_method_budget": 2,
            "api_event_count_before_extractor_budget": 2,
            "api_event_count_after_extractor_budget": 2,
            "api_extractor_coverage": 1.0,
            "api_truncated_by_extractor_budget": 0.0,
            "api_truncated": False,
            "api_truncation_ratio": 0.0,
            "api_known_type_count": 1,
            "api_unknown_type_count": 1,
            "dex_parse_ok": True,
            "dex_file_count": 1,
            "dex_parse_success_ratio": 1.0,
            "class_count": 1,
            "method_count": 2,
            "graph_parse_ok": True,
            "graph_parse_error": "",
            "graph_timeout": False,
            "graph_node_count_raw": 2,
            "graph_edge_count_raw": 1,
            "graph_isolated_node_count": 0,
            "graph_largest_component_ratio": 1.0,
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        },
        "direct_build_meta": {"dex_success_ratio": 1.0},
    }
    manifest_payload = {
        "manifest_x": torch.ones(16),
        "manifest_permission_ids": torch.tensor([1], dtype=torch.long),
        "manifest_intent_ids": torch.tensor([1], dtype=torch.long),
        "manifest_category_counts": torch.ones(12),
        "manifest_component_category_counts": torch.zeros(12),
        "manifest_permission_category_map": torch.zeros((1, 12)),
        "manifest_intent_category_map": torch.zeros((1, 12)),
        "manifest_stats": torch.ones(11),
        "manifest_meta": {},
        "manifest_permission_dim": 1,
        "manifest_intent_dim": 1,
        "manifest_feature_dim": 0,
        "q_manifest": torch.tensor([1.0]),
        "pert_manifest": torch.tensor([0.0]),
        "observable_metadata": {
            "manifest_parse_ok": True,
            "manifest_parse_error": "",
            "manifest_has_content": True,
            "manifest_vocab_coverage": 1.0,
            "manifest_permission_count": 1,
            "manifest_component_count": 0,
            "manifest_intent_count": 1,
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        },
    }
    torch.save(
        build_observable_payload(
            [dex],
            code_payload,
            manifest_payload,
            build_fingerprint="test",
        ),
        pt_dir / f"{sid}.pt",
    )
    csv_path = tmp_path / "strict_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1})
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )
    assert dataset[0].schema_version == OBSERVABLE_SCHEMA_VERSION


def test_current_schema_rejects_list_payload_even_with_nested_metadata(tmp_path: Path):
    pt_dir, csv_path = _old_list_pt(tmp_path, "nested_old")
    path = pt_dir / "nested_old.pt"
    raw = torch.load(path, map_location="cpu", weights_only=False)
    nested = {key: 0 for key in OBSERVABLE_REQUIRED_FIELDS}
    nested.update(
        {
            "api_parse_error": "",
            "graph_parse_error": "",
            "manifest_parse_error": "",
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        }
    )
    raw[0]["observable_metadata"] = nested
    torch.save(raw, path)
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )
    with pytest.raises(FatalDatasetConfigError, match="top-level mapping"):
        dataset[0]


def test_api_integrity_parse_failed_low():
    assert compute_api_integrity_v2({"api_parse_ok": False, "dex_parse_ok": True}) == 0.0


def test_graph_integrity_timeout_low():
    assert compute_graph_integrity_v2(
        {"dex_parse_ok": True, "graph_parse_ok": True, "graph_timeout": True}
    ) == 0.0


def test_manifest_integrity_parse_failed_low():
    assert compute_manifest_integrity_v2({"manifest_parse_ok": False}) == 0.0


def test_empty_but_parse_ok_integrity_stays_high_and_alive_is_zero():
    source = {
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "dex_parse_success_ratio": 1.0,
        "api_event_count_raw": 0,
        "api_event_count_kept": 0,
        "api_known_type_count": 0,
        "api_unknown_type_count": 0,
        "api_truncation_ratio": 0.0,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "graph_node_count_raw": 0,
        "graph_edge_count_raw": 0,
        "graph_isolated_node_count": 0,
        "graph_largest_component_ratio": 0.0,
        "method_count": 0,
        "manifest_parse_ok": True,
        "manifest_has_content": False,
        "manifest_vocab_coverage": 1.0,
        "manifest_permission_count": 0,
        "manifest_component_count": 0,
        "manifest_intent_count": 0,
    }
    assert compute_api_integrity_v2(source) == pytest.approx(1.0)
    assert compute_graph_integrity_v2(source) == pytest.approx(1.0)
    assert compute_manifest_integrity_v2(source) == pytest.approx(1.0)
    assert compute_raw_alive(source) == (0.0, 0.0, 0.0)


def test_api_integrity_decreases_with_visible_dropout_and_partial_dex_parse():
    clean = {
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "dex_parse_success_ratio": 1.0,
        "api_event_count_before_encoder_budget": 10,
        "api_event_count_after_encoder_budget": 10,
        "api_event_count_kept": 10,
        "api_known_type_count": 10,
        "api_unknown_type_count": 0,
    }
    # Half the post-budget events removed by synthetic dropout.
    dropped = {**clean, "api_event_count_kept": 5, "api_known_type_count": 5}
    partial = {**clean, "dex_parse_success_ratio": 0.5}
    assert compute_api_integrity_v2(clean) > compute_api_integrity_v2(dropped)
    assert compute_api_integrity_v2(clean) > compute_api_integrity_v2(partial)
    fully_dropped = {**clean, "api_event_count_kept": 0, "api_known_type_count": 0}
    assert compute_api_integrity_v2(fully_dropped) < 0.5


def test_api_encoder_budget_does_not_reduce_clean_api_integrity():
    clean_budgeted = {
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "dex_parse_success_ratio": 1.0,
        "api_event_count_raw": 5800,
        "api_event_count_before_encoder_budget": 5800,
        "api_event_count_after_encoder_budget": 2048,
        "api_event_count_kept": 2048,
        "api_known_type_count": 2048,
        "api_unknown_type_count": 0,
        "api_truncation_ratio": 0.0,
    }
    degraded_visible_events = {
        **clean_budgeted,
        "api_event_count_kept": 205,
        "api_known_type_count": 205,
    }

    assert compute_api_integrity_v2(clean_budgeted) == pytest.approx(1.0)
    assert compute_api_integrity_v2(degraded_visible_events) < compute_api_integrity_v2(clean_budgeted)


def test_api_budget_accounting_separates_extractor_and_runtime_coverage():
    data = {
        "api_ids": torch.arange(100),
        "api_type_ids": torch.ones(100, dtype=torch.long),
        "api_event_count_kept": 500,
        "api_event_count_before_method_budget": 1000,
        "api_event_count_after_method_budget": 800,
        "api_event_count_before_extractor_budget": 800,
        "api_event_count_after_extractor_budget": 500,
        "api_event_count_before_encoder_budget": 500,
        "api_event_count_after_encoder_budget": 100,
    }

    refresh_observable_metadata(data)

    assert data["api_extractor_coverage"] == pytest.approx(0.5)
    assert data["api_runtime_encoder_coverage"] == pytest.approx(0.2)
    assert data["api_encoder_coverage"] == pytest.approx(0.1)
    assert data["api_truncated_by_extractor_budget"] == 1.0
    assert data["api_truncated_by_encoder_budget"] == 1.0


def test_api_event_selection_keeps_sensitive_calls_and_contiguous_prefix_fill():
    events = [
        ApiEvent(
            old_method_idx=0,
            token=f"api-{index}",
            category_id=0,
            sensitive=index in {4, 7},
        )
        for index in range(10)
    ]

    selected = select_api_events(events, 5)

    assert [event.token for event in selected] == [
        "api-0",
        "api-1",
        "api-2",
        "api-4",
        "api-7",
    ]


def test_limit_api_events_records_encoder_budget_without_changing_extraction_counts():
    dataset = RobustTriModalDataset.__new__(RobustTriModalDataset)
    dataset.max_api_events_per_sample = 3
    parts = {
        "api_ids": torch.arange(5),
        "api_type_ids": torch.ones(5, dtype=torch.long),
        "api_sensitive_mask": torch.zeros(5),
        "api_method_index": torch.arange(5),
        "api_in_graph_mask": torch.ones(5),
        "method_api_edge_index": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long),
    }

    out = dataset._limit_api_events(parts)

    # No sensitive events here -> the budget keeps a contiguous prefix of the
    # non-sensitive events (preserving sequence locality), i.e. the first N.
    assert out["api_ids"].tolist() == [0, 1, 2]
    assert out["api_event_count_before_encoder_budget"] == 5
    assert out["api_event_count_after_encoder_budget"] == 3
    assert out["api_encoder_coverage"] == pytest.approx(0.6)
    assert out["api_truncated_by_encoder_budget"] == 1.0


def test_graph_integrity_decreases_for_partial_dex_and_extreme_fragmentation():
    clean = {
        "dex_parse_ok": True,
        "dex_parse_success_ratio": 1.0,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "graph_node_count_raw": 10,
        "graph_edge_count_raw": 9,
        "graph_isolated_node_count": 0,
        "graph_largest_component_ratio": 1.0,
        "method_count": 10,
    }
    fragmented = {
        **clean,
        "graph_edge_count_raw": 1,
        "graph_isolated_node_count": 9,
        "graph_largest_component_ratio": 0.01,
    }
    partial = {**clean, "dex_parse_success_ratio": 0.5}
    assert compute_graph_integrity_v2(clean) > compute_graph_integrity_v2(fragmented)
    assert compute_graph_integrity_v2(clean) > compute_graph_integrity_v2(partial)


def test_graph_integrity_decreases_when_node_features_are_masked():
    clean = {
        "dex_parse_ok": True,
        "dex_parse_success_ratio": 1.0,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "graph_node_count_raw": 10,
        "graph_edge_count_raw": 9,
        "graph_isolated_node_count": 0,
        "graph_largest_component_ratio": 1.0,
        "graph_feature_valid_ratio": 1.0,
        "method_count": 10,
    }
    masked = {**clean, "graph_feature_valid_ratio": 0.0}

    assert compute_graph_integrity_v2(masked) < compute_graph_integrity_v2(clean)


def test_api_graph_anchor_support_ignores_invalid_edges():
    edges = torch.tensor([[0, 1, 99], [0, 1, 2]], dtype=torch.long)
    assert compute_api_graph_anchor_support(edges, 3, 2) == pytest.approx(0.8)


def test_api_graph_anchor_support_accepts_ghost_offset_storage_nodes():
    edges = torch.tensor([[1], [0]], dtype=torch.long)
    assert compute_api_graph_anchor_support(edges, 1, 1) == 0.0
    assert compute_api_graph_anchor_support(edges, 1, 1, 2) == pytest.approx(1.0)


def test_graph_integrity_tracks_runtime_edge_retention():
    source = {
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "graph_node_count_raw": 10,
        "graph_edge_count_raw": 5,
        "graph_isolated_node_count": 0,
        "graph_largest_component_ratio": 1.0,
        "method_count": 10,
        "dex_parse_success_ratio": 1.0,
        "graph_edge_retention_ratio": 0.5,
    }
    assert compute_graph_integrity_v2(source) < 1.0


def test_manifest_support_and_directional_conflicts_are_separate():
    api = torch.tensor([1.0, 0.0, 1.0])
    graph = torch.tensor([0.0, 1.0, 0.0])
    manifest = torch.tensor([1.0, 0.0, 0.0])
    support, manifest_to_code, code_to_manifest = compute_manifest_code_support_and_conflict(
        api, graph, manifest
    )
    assert support == pytest.approx(1.0 / 3.0)
    assert manifest_to_code == pytest.approx(0.0)
    assert code_to_manifest == pytest.approx(2.0 / 3.0)
    assert support + manifest_to_code != pytest.approx(1.0)


def test_alive_uses_raw_availability():
    source = {
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "api_event_count_kept": 0,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "graph_node_count_raw": 2,
        "manifest_parse_ok": True,
        "manifest_has_content": False,
    }
    assert compute_raw_alive(source) == (0.0, 1.0, 0.0)


def test_clean_empty_manifest_is_not_synthetic_perturbation():
    payload = vectorize_manifest_record(
        {"parse_error": "", "permissions": [], "intent_actions": [], "intent_categories": []},
        {
            "categories": list(DEFAULT_CATEGORIES),
            "permission_vocab": [],
            "intent_vocab": [],
            "feature_vocab": [],
        },
        manifest_dim=len(DEFAULT_CATEGORIES) + 11,
    )
    assert payload["observable_metadata"]["manifest_has_content"] is False
    assert payload["pert_manifest"].item() == 0.0


def _evidence_data() -> Data:
    data = Data()
    for key, value in {
        "api_integrity": 0.8,
        "graph_integrity": 0.7,
        "manifest_integrity": 0.6,
        "code_integrity": 0.75,
        "api_graph_anchor_support": 0.5,
        "manifest_code_support": 0.4,
        "manifest_to_code_conflict": 0.2,
        "code_to_manifest_conflict": 0.3,
        "api_alive": 1.0,
        "graph_alive": 1.0,
        "manifest_alive": 1.0,
        "pert_api": 0.9,
        "pert_graph": 0.8,
        "pert_manifest": 0.7,
    }.items():
        setattr(data, key, torch.tensor([[value]], dtype=torch.float32))
    data.api_semantic_category_counts = torch.ones(1, 12)
    data.graph_semantic_category_counts = torch.ones(1, 12)
    data.manifest_category_counts = torch.ones(1, 12)
    return data


def test_main_evidence_excludes_perturbation_fields():
    data = _evidence_data()
    logits = torch.zeros(1, 2)
    emb = torch.zeros(1, 4)
    evidence, diagnostics = build_evidence(
        data,
        logits,
        logits,
        logits,
        logits,
        emb,
        emb,
        emb,
        use_consistency_evidence=True,
        use_conflict_evidence=True,
    )
    assert evidence.shape == (1, EvidenceIndex.BASE_DIM)
    assert all(key not in diagnostics for key in ("pert_api", "pert_graph", "pert_manifest"))
    before = evidence.clone()
    data.pert_api.fill_(0.0)
    after, _ = build_evidence(
        data,
        logits,
        logits,
        logits,
        logits,
        emb,
        emb,
        emb,
        use_consistency_evidence=True,
        use_conflict_evidence=True,
    )
    assert torch.equal(before, after)


def test_api_visibility_uses_runtime_encoder_coverage_only():
    data = _evidence_data()
    data.api_encoder_coverage = torch.tensor([[0.08]], dtype=torch.float32)
    data.api_extractor_coverage = torch.tensor([[0.10]], dtype=torch.float32)
    data.api_runtime_encoder_coverage = torch.tensor([[0.75]], dtype=torch.float32)
    logits = torch.zeros(1, 2)
    emb = torch.zeros(1, 4)

    evidence, diagnostics = build_evidence(
        data,
        logits,
        logits,
        logits,
        logits,
        emb,
        emb,
        emb,
        use_consistency_evidence=True,
        use_conflict_evidence=True,
    )

    assert evidence[0, EvidenceIndex.API_ENCODER_COVERAGE].item() == pytest.approx(0.75)
    assert diagnostics["api_encoder_coverage"].item() == pytest.approx(0.75)
    assert diagnostics["api_runtime_encoder_coverage"].item() == pytest.approx(0.75)
    assert diagnostics["api_total_pipeline_coverage"].item() == pytest.approx(0.08)
    assert diagnostics["effective_api_integrity"].item() == pytest.approx(0.8 * 0.75)


def test_manifest_perturbation_refreshes_observables():
    data = {
        "api_ids": torch.tensor([1]),
        "api_type_ids": torch.tensor([1]),
        "api_in_graph_mask": torch.tensor([1.0]),
        "method_api_edge_index": torch.tensor([[0], [0]], dtype=torch.long),
        "api_semantic_category_counts": torch.tensor([1.0] + [0.0] * 11),
        "graph_semantic_category_counts": torch.tensor([1.0] + [0.0] * 11),
        "x": torch.ones(1, 4),
        "real_node_mask": torch.ones(1, dtype=torch.bool),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "manifest_x": torch.tensor([1.0, 0.0]),
        "manifest_permission_dim": 2,
        "manifest_intent_dim": 0,
        "manifest_feature_dim": 0,
        "manifest_permission_ids": torch.tensor([1]),
        "manifest_intent_ids": torch.empty(0, dtype=torch.long),
        "manifest_permission_category_map": torch.tensor([[0.0, 1.0] + [0.0] * 10, [0.0] * 12]),
        "manifest_category_counts": torch.tensor([1.0, 1.0] + [0.0] * 10),
        "manifest_stats": torch.zeros(11),
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "manifest_parse_ok": True,
        "manifest_vocab_coverage": 1.0,
        "manifest_component_count": 0,
        "pert_manifest": 0.0,
    }
    refresh_observable_signals(data)
    before_conflict = data["manifest_to_code_conflict"]
    out = apply_manifest_permission_mask(data, 1.0)
    assert out["manifest_to_code_conflict"] < before_conflict
    assert "manifest_integrity" in out
    assert "manifest_code_support" in out
    assert "manifest_alive" in out


def test_api_graph_anchor_support_not_named_consistency():
    data = _evidence_data()
    logits = torch.zeros(1, 2)
    emb = torch.zeros(1, 4)
    _, diagnostics = build_evidence(
        data,
        logits,
        logits,
        logits,
        logits,
        emb,
        emb,
        emb,
        use_consistency_evidence=True,
        use_conflict_evidence=True,
    )
    assert "api_graph_anchor_support" in diagnostics
    assert "api_graph_consistency" not in diagnostics


def test_diagnostic_distribution_and_trend_checks_report_variation():
    rows = []
    for strength, integrity in ((0.1, 0.9), (0.5, 0.6), (0.9, 0.2)):
        for sid, label in (("a", 0), ("b", 1)):
            row = {
                "split": "fixture",
                "scenario": "api_event_dropout",
                "strength": strength,
                "sid": sid,
                "label": label,
            }
            row.update({signal: 0.5 for signal in OBSERVABLE_SIGNAL_FIELDS})
            row["api_integrity"] = integrity
            rows.append(row)
    distribution = _distribution_table(pd.DataFrame.from_records(rows))
    trends = _trend_table(distribution)
    checks = _output_checks(distribution, trends)
    api_trend = trends[
        (trends["scenario"] == "api_event_dropout")
        & (trends["signal"] == "api_integrity")
    ].iloc[0]
    assert api_trend["varies_with_strength"]
    assert api_trend["expected_direction"] == "nonincreasing"
    assert api_trend["monotonic_expected"]
    assert checks["passed"].all()


def test_diagnostic_output_checks_reject_out_of_range_signal():
    bad_distribution = pd.DataFrame.from_records(
        [
            {
                "split": "fixture",
                "scenario": "clean",
                "strength": 0.0,
                "signal": "api_integrity",
                "count": 1,
                "mean": 1.2,
                "variance": 0.0,
                "p05": 1.2,
                "p25": 1.2,
                "p50": 1.2,
                "p75": 1.2,
                "p95": 1.2,
            }
        ]
    )
    trends = _trend_table(bad_distribution)
    checks = _output_checks(bad_distribution, trends).set_index("check")
    assert not bool(checks.loc["distribution_signal_range", "passed"])
