from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from fusion.constants import AvailabilityIndex
from fusion.dataset import (
    FatalDatasetConfigError,
    RobustTriModalDataset,
    robust_collate_fn,
)
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.evidence import build_fusion_availability_and_diagnostics
from fusion.perturbations import (
    apply_api_event_dropout,
    apply_api_missing,
    apply_graph_missing,
    apply_manifest_missing,
)
from fusion.quality import (
    OBSERVABLE_REQUIRED_FIELDS,
    OBSERVABLE_SCHEMA_VERSION,
    compute_raw_alive,
    refresh_hard_availability,
)
from tests.pt_factory import save_current_pt


def _dex() -> dict:
    return {
        "call_x": torch.ones((2, 8), dtype=torch.float32),
        "call_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
        "api_ids": torch.tensor([11, 12, 13], dtype=torch.long),
        "api_type_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "api_sensitive_mask": torch.zeros(3),
        "api_method_index": torch.tensor([0, 1, 1], dtype=torch.long),
        "api_in_graph_mask": torch.ones(3),
        "method_api_edge_index": torch.tensor(
            [[0, 1, 1], [0, 1, 2]], dtype=torch.long
        ),
    }


def _write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for sid, label in rows:
            writer.writerow({"id": sid, "label": label})


def _old_list_pt(tmp_path: Path, sid: str = "sample") -> tuple[Path, Path]:
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    torch.save([_dex()], pt_dir / f"{sid}.pt")
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, [(sid, 1)])
    return pt_dir, csv_path


def test_observable_schema_strict_missing_fields(tmp_path: Path):
    pt_dir, csv_path = _old_list_pt(tmp_path, "strict_missing")
    with pytest.raises(FatalDatasetConfigError, match="top-level mapping"):
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=16,
        )


def test_persisted_observable_schema_contract_is_unchanged():
    assert OBSERVABLE_SCHEMA_VERSION == "observable-v2"
    for field in (
        "api_parse_ok",
        "dex_parse_ok",
        "graph_parse_ok",
        "graph_timeout",
        "manifest_parse_ok",
        "manifest_has_content",
        "schema_version",
    ):
        assert field in OBSERVABLE_REQUIRED_FIELDS


def test_current_pt_loads_but_runtime_data_contains_only_hard_availability(
    tmp_path: Path,
):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "current"
    save_current_pt(pt_dir / f"{sid}.pt", _dex())
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, [(sid, 1)])

    sample = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )[0]
    assert sample.api_alive.item() == 1.0
    assert sample.graph_alive.item() == 1.0
    assert sample.manifest_alive.item() == 1.0
    for removed in (
        "q_api",
        "q_graph",
        "q_manifest",
        "q_align",
        "pert_api",
        "pert_graph",
        "pert_manifest",
        "api_integrity",
        "graph_integrity",
        "manifest_integrity",
        "api_encoder_coverage",
        "graph_encoder_coverage",
    ):
        assert not hasattr(sample, removed)


def test_historical_quality_payload_cannot_change_availability_or_fusion_output(
    tmp_path: Path,
):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    low_sid = "quality_low"
    high_sid = "quality_high"
    low_observable = {
        "api_extractor_coverage": 0.01,
        "api_truncation_ratio": 0.99,
        "graph_largest_component_ratio": 0.01,
        "manifest_vocab_coverage": 0.01,
    }
    high_observable = {
        "api_extractor_coverage": 1.0,
        "api_truncation_ratio": 0.0,
        "graph_largest_component_ratio": 1.0,
        "manifest_vocab_coverage": 1.0,
    }
    save_current_pt(
        pt_dir / f"{low_sid}.pt",
        _dex(),
        top_level={
            "observable_metadata": low_observable,
            "q_manifest": torch.tensor([-1000.0]),
            "pert_manifest": torch.tensor([1000.0]),
        },
    )
    save_current_pt(
        pt_dir / f"{high_sid}.pt",
        _dex(),
        top_level={
            "observable_metadata": high_observable,
            "q_manifest": torch.tensor([1000.0]),
            "pert_manifest": torch.tensor([-1000.0]),
        },
    )
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, [(low_sid, 1), (high_sid, 1)])
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )
    first, second = dataset[0], dataset[1]
    assert torch.equal(first.api_ids, second.api_ids)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.manifest_x, second.manifest_x)

    batch = robust_collate_fn([first, second])["graph_batch"]
    logits = torch.tensor([[0.4, -0.2], [0.4, -0.2]], dtype=torch.float32)
    emb = torch.zeros((2, 4), dtype=torch.float32)
    availability, diagnostics = build_fusion_availability_and_diagnostics(
        batch,
        logits,
        logits,
        logits,
        emb,
        emb,
        emb,
    )
    assert tuple(availability.shape) == (2, AvailabilityIndex.BASE_DIM)
    assert torch.equal(availability, torch.ones_like(availability))
    assert set(diagnostics) == {"api_alive", "graph_alive", "manifest_alive"}

    fusion = DiscountProbabilityFusion({"combination": "cumulative"})
    outputs = fusion(logits, logits, logits, availability)
    assert torch.allclose(outputs["final_prob"][0], outputs["final_prob"][1])


def test_compute_raw_alive_uses_only_parse_state_and_current_content():
    source = {
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "api_event_count_kept": 2,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "graph_node_count_raw": 3,
        "manifest_parse_ok": True,
        "manifest_has_content": True,
        # Deliberately contradictory historical proxy values.
        "api_integrity": 0.0,
        "graph_encoder_coverage": 0.0,
        "q_manifest": 0.0,
        "pert_api": 1.0,
    }
    assert compute_raw_alive(source) == (1.0, 1.0, 1.0)


def test_refresh_hard_availability_tracks_current_tensors_only():
    data = {
        **_dex(),
        "x": torch.ones((2, 8)),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "real_node_mask": torch.ones(2, dtype=torch.bool),
        "manifest_x": torch.ones(16),
        "manifest_permission_ids": torch.tensor([1]),
        "manifest_intent_ids": torch.tensor([1]),
        "manifest_category_counts": torch.ones(12),
        "manifest_stats": torch.ones(11),
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "manifest_parse_ok": True,
        "manifest_component_count": 0,
    }
    refresh_hard_availability(data)
    assert (data["api_alive"], data["graph_alive"], data["manifest_alive"]) == (
        1.0,
        1.0,
        1.0,
    )
    assert not any(
        key in data
        for key in (
            "api_integrity",
            "graph_integrity",
            "manifest_integrity",
            "api_encoder_coverage",
            "manifest_code_support",
        )
    )


def test_partial_and_missing_perturbations_refresh_hard_availability():
    data = {
        **_dex(),
        "x": torch.ones((2, 8)),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "real_node_mask": torch.ones(2, dtype=torch.bool),
        "manifest_x": torch.ones(16),
        "manifest_permission_ids": torch.tensor([1]),
        "manifest_intent_ids": torch.tensor([1]),
        "manifest_category_counts": torch.ones(12),
        "manifest_component_category_counts": torch.zeros(12),
        "manifest_stats": torch.ones(11),
        "manifest_permission_dim": 2,
        "manifest_intent_dim": 1,
        "manifest_feature_dim": 0,
        "manifest_permission_category_map": torch.zeros((2, 12)),
        "manifest_intent_category_map": torch.zeros((1, 12)),
        "api_semantic_category_counts": torch.ones(12),
        "graph_semantic_category_counts": torch.ones(12),
        "graph_semantic_source": "alignment",
        "mask": torch.empty((2, 0)),
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "graph_timeout": False,
        "manifest_parse_ok": True,
        "manifest_component_count": 0,
    }
    refresh_hard_availability(data)
    partial = apply_api_event_dropout(dict(data), 0.5)
    assert partial["api_alive"] == 1.0
    assert int(partial["api_ids"].numel()) < int(data["api_ids"].numel())

    api_missing = apply_api_missing(dict(data))
    graph_missing = apply_graph_missing(dict(data))
    manifest_missing = apply_manifest_missing(dict(data))
    assert api_missing["api_alive"] == 0.0
    assert graph_missing["graph_alive"] == 0.0
    assert manifest_missing["manifest_alive"] == 0.0


@pytest.mark.parametrize("field", ["api_alive", "graph_alive", "manifest_alive"])
@pytest.mark.parametrize("bad", [None, 0.5, float("nan")])
def test_fusion_availability_rejects_missing_or_nonbinary_alive(field, bad):
    data = Data()
    for name in ("api_alive", "graph_alive", "manifest_alive"):
        setattr(data, name, torch.ones((1, 1)))
    if bad is None:
        delattr(data, field)
    else:
        setattr(data, field, torch.tensor([[bad]], dtype=torch.float32))
    logits = torch.zeros((1, 2))
    emb = torch.zeros((1, 4))
    with pytest.raises(ValueError, match="availability"):
        build_fusion_availability_and_diagnostics(
            data, logits, logits, logits, emb, emb, emb
        )
