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
