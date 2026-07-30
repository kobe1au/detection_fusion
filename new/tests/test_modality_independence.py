"""Modality-independence regressions for encoder input truncation.

Budget truncation of one modality must not depend on another modality.
"""

import copy

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from fusion import graph_encoders
from fusion.dataset import RobustTriModalDataset, apply_graph_encoder_budget
from fusion.graph_encoders import GraphEncoderGCN, truncate_per_graph


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
    assert kept == sorted(kept)  # original API event order preserved
    assert "api_encoder_coverage" not in out
    assert "api_event_count_before_encoder_budget" not in out


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


def test_implicit_encoder_truncation_warns_once_per_process(monkeypatch):
    monkeypatch.setattr(
        graph_encoders, "_IMPLICIT_ENCODER_TRUNCATION_WARNED", False
    )
    with pytest.warns(
        RuntimeWarning,
        match="without graph_encoder_budget_max_nodes metadata",
    ) as recorded:
        truncate_per_graph(_graph(), max_nodes=3)
        truncate_per_graph(_graph(), max_nodes=3)

    assert len(recorded) == 1


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


def test_dataset_graph_budget_is_exposed_as_encoder_contract():
    data = {
        "x": torch.zeros((2, 3), dtype=torch.float32),
    }

    out = apply_graph_encoder_budget(data, 5)

    assert out["graph_encoder_budget_max_nodes"] == 5


def test_truncate_per_graph_rejects_dataset_encoder_budget_mismatch():
    data = _graph(num_nodes=3, sensitive=(1, 2), with_edges=False)
    data.graph_encoder_budget_max_nodes = torch.tensor([4], dtype=torch.long)

    with pytest.raises(ValueError, match="must match to avoid double truncation"):
        truncate_per_graph(data, max_nodes=3)


def test_truncate_per_graph_rejects_violated_dataset_budget_contract():
    data = _graph(num_nodes=6)
    data.graph_encoder_budget_max_nodes = torch.tensor([3], dtype=torch.long)

    with pytest.raises(RuntimeError, match="contract was violated"):
        truncate_per_graph(data, max_nodes=3)


def _budgeted_graph_batch(*, include_fast_contract: bool) -> Batch:
    first = Data(
        x=torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    first.sensitive_mask = torch.tensor([0, 1, 0], dtype=torch.uint8)
    first.graph_encoder_budget_max_nodes = torch.tensor([4], dtype=torch.long)
    second = Data(
        x=torch.tensor(
            [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=torch.float32
        ),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    second.sensitive_mask = torch.tensor([1, 0], dtype=torch.uint8)
    second.graph_encoder_budget_max_nodes = torch.tensor([4], dtype=torch.long)
    batch = Batch.from_data_list([first, second])
    if include_fast_contract:
        batch.graph_encoder_budget_contract = [1, 4, [3, 2]]
    return batch


def test_prevalidated_graph_budget_fast_path_skips_noop_index_construction(monkeypatch):
    batch = _budgeted_graph_batch(include_fast_contract=True)

    def forbidden_bincount(*args, **kwargs):
        raise AssertionError("the prevalidated fast path must not count graphs on GPU")

    monkeypatch.setattr(graph_encoders.torch, "bincount", forbidden_bincount)
    x, edge_index, returned_batch, keep_local = truncate_per_graph(
        batch, max_nodes=4
    )

    assert x is batch.x
    assert edge_index is batch.edge_index
    assert returned_batch is batch.batch
    assert keep_local is None


def test_prevalidated_graph_budget_fast_path_preserves_contract_guards():
    mismatch = _budgeted_graph_batch(include_fast_contract=True)
    with pytest.raises(ValueError, match="must match to avoid double truncation"):
        truncate_per_graph(mismatch, max_nodes=3)

    violated = _budgeted_graph_batch(include_fast_contract=True)
    violated.graph_encoder_budget_contract = [1, 4, [5, 0]]
    with pytest.raises(RuntimeError, match="contract was violated"):
        truncate_per_graph(violated, max_nodes=4)


def test_graph_encoder_fast_path_matches_guarded_path_outputs_and_gradients():
    torch.manual_seed(11)
    guarded_data = _budgeted_graph_batch(include_fast_contract=False)
    fast_data = _budgeted_graph_batch(include_fast_contract=True)
    guarded_data.x = guarded_data.x.detach().clone().requires_grad_(True)
    fast_data.x = fast_data.x.detach().clone().requires_grad_(True)

    guarded_encoder = GraphEncoderGCN(
        in_dim=3,
        out_dim=4,
        hidden=5,
        max_nodes=4,
        use_behavior_hint=True,
    ).eval()
    fast_encoder = copy.deepcopy(guarded_encoder).eval()

    guarded_h, guarded_emb, _, guarded_keep = guarded_encoder(guarded_data)
    fast_h, fast_emb, _, fast_keep = fast_encoder(fast_data)
    guarded_loss = guarded_h.square().sum() + guarded_emb.square().sum()
    fast_loss = fast_h.square().sum() + fast_emb.square().sum()
    guarded_loss.backward()
    fast_loss.backward()

    torch.testing.assert_close(fast_h, guarded_h, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fast_emb, guarded_emb, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fast_data.x.grad, guarded_data.x.grad, rtol=0.0, atol=0.0)
    assert guarded_keep is None
    assert fast_keep is None
    for guarded_parameter, fast_parameter in zip(
        guarded_encoder.parameters(), fast_encoder.parameters()
    ):
        torch.testing.assert_close(
            fast_parameter.grad,
            guarded_parameter.grad,
            rtol=0.0,
            atol=0.0,
        )


def test_graph_encoder_behavior_hint_mask_paths_are_not_reconstructed(monkeypatch):
    def forbidden_recovery(*args, **kwargs):
        raise AssertionError("sensitive-mask recovery must not run on this path")

    monkeypatch.setattr(
        graph_encoders, "_recover_truncated_sensitive_mask", forbidden_recovery
    )

    fast_data = _budgeted_graph_batch(include_fast_contract=True)
    captured = {}

    class CaptureReadout(nn.Module):
        def forward(self, node_emb, batch, num_graphs, sensitive_mask=None):
            captured["mask"] = sensitive_mask
            return node_emb.new_zeros((num_graphs, node_emb.size(-1)))

    hinted = GraphEncoderGCN(
        in_dim=3,
        out_dim=4,
        hidden=5,
        max_nodes=4,
        use_behavior_hint=True,
    ).eval()
    hinted.readout = CaptureReadout()
    hinted(fast_data)
    assert captured["mask"] is fast_data.sensitive_mask

    # Even when real truncation occurs, a disabled behavior hint must not
    # recover or feed the sensitive mask into readout.
    truncated = _graph(num_nodes=5, sensitive=(3, 4), with_edges=False)
    unhinted = GraphEncoderGCN(
        in_dim=3,
        out_dim=4,
        hidden=5,
        max_nodes=3,
        use_behavior_hint=False,
    ).eval()
    monkeypatch.setattr(graph_encoders, "_IMPLICIT_ENCODER_TRUNCATION_WARNED", True)
    unhinted(truncated)
