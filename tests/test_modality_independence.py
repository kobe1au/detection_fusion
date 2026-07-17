"""Modality-independence regression tests for input-level truncation + guardrails.

The evidential fusion (I2) treats API / Graph / Manifest as independent evidence
sources. These tests pin that the budget truncation of one modality does not
depend on another modality's fields. Observable relation evidence is allowed as
I1 metadata, while switches that feed one modality's features into another
encoder are rejected for evidential combination rules.
"""

import pytest
import torch
from torch_geometric.data import Data

from fusion.dataset import RobustTriModalDataset
from fusion.graph_encoders import truncate_per_graph
from fusion.train import build_model


def _api_parts(method_api_edge_index, api_in_graph_mask):
    return {
        "api_ids": torch.tensor([10, 11, 12, 13, 14, 15]),
        "api_type_ids": torch.tensor([1, 1, 1, 1, 2, 2]),
        "api_sensitive_mask": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),  # sensitive late
        "api_method_index": torch.arange(6),
        "api_in_graph_mask": api_in_graph_mask,
        "method_api_edge_index": method_api_edge_index,
    }


def test_limit_api_events_keeps_sensitive_and_preserves_order():
    ds = RobustTriModalDataset.__new__(RobustTriModalDataset)
    ds.max_api_events_per_sample = 3
    parts = _api_parts(
        method_api_edge_index=torch.tensor([[0, 4], [0, 4]], dtype=torch.long),
        api_in_graph_mask=torch.ones(6),
    )
    out = ds._limit_api_events(parts)
    kept = out["api_ids"].tolist()
    assert len(kept) == 3
    # Both sensitive events (ids 14, 15) survive even though they are last.
    assert 14 in kept and 15 in kept
    assert kept == sorted(kept)  # temporal order preserved
    # Encoder-budget bookkeeping unchanged.
    assert out["api_event_count_before_encoder_budget"] == 6
    assert out["api_event_count_after_encoder_budget"] == 3


def test_limit_api_events_is_independent_of_graph_fields():
    ds = RobustTriModalDataset.__new__(RobustTriModalDataset)
    ds.max_api_events_per_sample = 3
    a = ds._limit_api_events(
        _api_parts(torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long), torch.ones(6))
    )["api_ids"].tolist()
    # Completely different graph alignment / in-graph mask.
    b = ds._limit_api_events(
        _api_parts(torch.empty((2, 0), dtype=torch.long), torch.zeros(6))
    )["api_ids"].tolist()
    assert a == b  # kept API set must not depend on the graph


def _graph(num_nodes=6, sensitive=(4, 5), with_edges=True):
    data = Data(
        x=torch.arange(num_nodes).float().view(num_nodes, 1).repeat(1, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long) if with_edges else torch.empty((2, 0), dtype=torch.long),
    )
    mask = torch.zeros(num_nodes, dtype=torch.uint8)
    for i in sensitive:
        mask[i] = 1
    data.sensitive_mask = mask
    return data


def test_truncate_per_graph_keeps_sensitive_nodes():
    data = _graph()
    x, _, _, keep_local = truncate_per_graph(data, max_nodes=3)
    kept_rows = set(int(v) for v in x[:, 0].tolist())
    assert {4, 5}.issubset(kept_rows)  # sensitive nodes survive
    assert len(kept_rows) == 3
    assert set(keep_local[0].tolist()) == kept_rows


def test_truncate_per_graph_is_independent_of_api_alignment():
    base = _graph()
    base.method_api_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    base.api_sensitive_mask = torch.ones(2)
    base.api_type_ids = torch.ones(2, dtype=torch.long)
    x1, _, _, _ = truncate_per_graph(base, max_nodes=3)

    other = _graph()
    other.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    x2, _, _, _ = truncate_per_graph(other, max_nodes=3)
    assert x1[:, 0].tolist() == x2[:, 0].tolist()  # graph truncation ignores API links


def _cfg(combination, *, relation_evidence=False, behavior_hint=False):
    return {
        "model": {
            "fusion_mode": "discount_probability",
            "graph_encoder": {"type": "gatv2", "use_behavior_hint": behavior_hint},
        },
        "fusion": {
            "mode": "discount_probability",
            "combination": combination,
            "reliability_calibration": {"enabled": True, "use_relation_evidence": relation_evidence},
        },
    }


def test_build_model_allows_observable_relation_evidence_under_evidential_combination():
    model = build_model(_cfg("dempster", relation_evidence=True), feature_dim=515)
    assert model is not None


def test_build_model_rejects_behavior_hint_under_evidential_combination():
    with pytest.raises(ValueError, match="non-duplicated branch inputs"):
        build_model(_cfg("dempster", behavior_hint=True), feature_dim=515)


def test_build_model_allows_coupling_switches_under_linear_combination():
    # Linear (legacy) fusion is explicitly allowed to couple modalities.
    model = build_model(_cfg("linear", relation_evidence=True, behavior_hint=True), feature_dim=515)
    assert model is not None
