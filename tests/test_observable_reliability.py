from __future__ import annotations

import csv
from pathlib import Path

import pytest
import pandas as pd
import torch
from torch_geometric.data import Data

from fusion.dataset import FatalDatasetConfigError, RobustTriModalDataset, robust_collate_fn
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
    refresh_observable_signals,
)
from scripts.build_tri_modal_pts_direct import build_observable_payload
from scripts.diagnose_observable_signals import _distribution_table, _output_checks, _trend_table


def _legacy_pt(tmp_path: Path, sid: str = "sample") -> tuple[Path, Path]:
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
    pt_dir, csv_path = _legacy_pt(tmp_path, "strict_missing")
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
        strict_observable_schema=True,
    )
    with pytest.raises(FatalDatasetConfigError, match="observable schema is incomplete"):
        dataset[0]


def test_observable_schema_non_strict_fallback(tmp_path: Path):
    pt_dir, csv_path = _legacy_pt(tmp_path, "compat")
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
        strict_observable_schema=False,
    )
    sample = dataset[0]
    batch = robust_collate_fn([sample])
    assert sample.schema_version == "legacy-fallback"
    assert batch["graph_batch"].api_integrity.shape == (1, 1)
    assert batch["observable_metadata"]["schema_version"] == ["legacy-fallback"]


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
            "api_truncated": False,
            "api_truncation_ratio": 0.0,
            "api_known_type_count": 1,
            "api_unknown_type_count": 1,
            "dex_parse_ok": True,
            "dex_file_count": 1,
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
        "manifest_stats": torch.ones(11),
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
        strict_observable_schema=True,
    )
    assert dataset[0].schema_version == OBSERVABLE_SCHEMA_VERSION


def test_observable_schema_strict_rejects_legacy_list_with_nested_metadata(tmp_path: Path):
    pt_dir, csv_path = _legacy_pt(tmp_path, "nested_legacy")
    path = pt_dir / "nested_legacy.pt"
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
        strict_observable_schema=True,
    )
    with pytest.raises(FatalDatasetConfigError, match="observable schema is incomplete"):
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


def test_api_integrity_decreases_with_truncation_and_partial_dex_parse():
    clean = {
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "dex_parse_success_ratio": 1.0,
        "api_event_count_raw": 10,
        "api_event_count_kept": 10,
        "api_known_type_count": 10,
        "api_unknown_type_count": 0,
        "api_truncation_ratio": 0.0,
    }
    truncated = {
        **clean,
        "api_event_count_kept": 5,
        "api_known_type_count": 5,
        "api_truncation_ratio": 0.5,
    }
    partial = {**clean, "dex_parse_success_ratio": 0.5}
    assert compute_api_integrity_v2(clean) > compute_api_integrity_v2(truncated)
    assert compute_api_integrity_v2(clean) > compute_api_integrity_v2(partial)
    fully_truncated = {
        **clean,
        "api_event_count_kept": 0,
        "api_known_type_count": 0,
        "api_truncation_ratio": 1.0,
    }
    assert compute_api_integrity_v2(fully_truncated) < 0.5


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
    assert support == pytest.approx(1.0)
    assert manifest_to_code == pytest.approx(0.0)
    assert code_to_manifest == pytest.approx(2.0 / 3.0)


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
    assert evidence.shape == (1, 11)
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
