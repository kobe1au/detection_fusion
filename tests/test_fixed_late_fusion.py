from types import SimpleNamespace

import torch

import fusion.model as model_module
from fusion.constants import AvailabilityIndex


def test_fixed_late_fusion_uses_exactly_three_source_branches(monkeypatch):
    batch_size = 2
    availability = torch.ones(batch_size, AvailabilityIndex.BASE_DIM)
    monkeypatch.setattr(
        model_module,
        "build_fusion_availability_and_diagnostics",
        lambda *args, **kwargs: (availability, {}),
    )
    model = SimpleNamespace(
        fusion_mode="tri_modal_fixed_gate",
        training=False,
    )
    tensors = {
        "graph_data": object(),
        "api_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        "graph_logits": torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
        "manifest_logits": torch.tensor([[1.5, 1.5], [1.5, 1.5]]),
        "api_emb": torch.zeros(batch_size, 2),
        "graph_emb": torch.zeros(batch_size, 2),
        "manifest_emb": torch.zeros(batch_size, 2),
    }

    logits, weights, _ = model_module._fusion_fixed_gate(
        model,
        batch_size,
        torch.device("cpu"),
        torch.float32,
        tensors,
        {},
    )

    expected_weights = torch.full((batch_size, 3), 1 / 3)
    assert torch.allclose(weights, expected_weights)
    expected_logits = (
        tensors["api_logits"]
        + tensors["graph_logits"]
        + tensors["manifest_logits"]
    ) / 3.0
    assert torch.allclose(logits, expected_logits)


def test_fixed_late_fusion_neutralizes_unavailable_branch_logits(monkeypatch):
    availability = torch.tensor([[0.0, 1.0, 1.0]])
    monkeypatch.setattr(
        model_module,
        "build_fusion_availability_and_diagnostics",
        lambda *args, **kwargs: (availability, {}),
    )
    tensors = {
        "graph_data": object(),
        "api_logits": torch.tensor([[100.0, -100.0]]),
        "graph_logits": torch.tensor([[0.0, 3.0]]),
        "manifest_logits": torch.tensor([[0.0, 3.0]]),
        "api_emb": torch.zeros(1, 2),
        "graph_emb": torch.zeros(1, 2),
        "manifest_emb": torch.zeros(1, 2),
    }

    logits, weights, _ = model_module._fusion_fixed_gate(
        SimpleNamespace(training=False),
        1,
        torch.device("cpu"),
        torch.float32,
        tensors,
        {},
    )

    assert torch.allclose(weights, torch.full((1, 3), 1.0 / 3.0))
    assert torch.allclose(logits, torch.tensor([[0.0, 2.0]]))


def test_fixed_late_fusion_all_dead_returns_zero_logits(monkeypatch):
    availability = torch.zeros(1, AvailabilityIndex.BASE_DIM)
    monkeypatch.setattr(
        model_module,
        "build_fusion_availability_and_diagnostics",
        lambda *args, **kwargs: (availability, {}),
    )
    tensors = {
        "graph_data": object(),
        "api_logits": torch.tensor([[100.0, -100.0]]),
        "graph_logits": torch.tensor([[-50.0, 50.0]]),
        "manifest_logits": torch.tensor([[25.0, -25.0]]),
        "api_emb": torch.zeros(1, 2),
        "graph_emb": torch.zeros(1, 2),
        "manifest_emb": torch.zeros(1, 2),
    }

    logits, weights, _ = model_module._fusion_fixed_gate(
        SimpleNamespace(training=False),
        1,
        torch.device("cpu"),
        torch.float32,
        tensors,
        {},
    )

    assert torch.equal(weights, torch.full((1, 3), 1.0 / 3.0))
    assert torch.equal(logits, torch.zeros(1, 2))
