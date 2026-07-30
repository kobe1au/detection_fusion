from __future__ import annotations

import torch

from fusion.model import quality_aware_logit_fusion


def test_quality_aware_fusion_matches_qmf_energy_rule():
    api = torch.tensor([[2.0, -1.0], [0.5, 0.1]])
    graph = torch.tensor([[0.2, 0.8], [1.0, -0.5]])
    manifest = torch.tensor([[0.3, 0.1], [-0.2, 0.7]])
    alive = torch.ones(2, 3)

    fused, weights, energy = quality_aware_logit_fusion(
        [api, graph, manifest], alive, temperature=10.0
    )
    stacked = torch.stack([api, graph, manifest], dim=1)
    expected_energy = torch.logsumexp(stacked, dim=-1) / 10.0
    expected = (stacked * expected_energy.detach().unsqueeze(-1)).sum(dim=1)

    assert torch.allclose(energy, expected_energy)
    assert torch.allclose(fused, expected)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_quality_aware_fusion_masks_unavailable_branch():
    api = torch.tensor([[3.0, -3.0]])
    graph = torch.tensor([[-3.0, 3.0]])
    manifest = torch.tensor([[1.0, 0.0]])
    alive = torch.tensor([[1.0, 0.0, 1.0]])

    fused, weights, _ = quality_aware_logit_fusion(
        [api, graph, manifest], alive, temperature=10.0
    )
    expected = (
        api * (torch.logsumexp(api, dim=-1) / 10.0).view(-1, 1)
        + manifest * (torch.logsumexp(manifest, dim=-1) / 10.0).view(-1, 1)
    )
    assert torch.allclose(fused, expected)
    assert weights[0, 1].item() == 0.0


def test_quality_aware_fusion_all_dead_returns_explicit_uniform_prediction():
    branch_logits = [
        torch.tensor([[9.0, -4.0]]),
        torch.tensor([[-3.0, 7.0]]),
        torch.tensor([[5.0, 2.0]]),
    ]

    fused, weights, _ = quality_aware_logit_fusion(
        branch_logits,
        torch.zeros(1, 3),
        temperature=10.0,
    )

    assert torch.equal(fused, torch.zeros(1, 2))
    assert torch.equal(torch.softmax(fused, dim=-1), torch.full((1, 2), 0.5))
    assert torch.allclose(weights, torch.full((1, 3), 1.0 / 3.0))
