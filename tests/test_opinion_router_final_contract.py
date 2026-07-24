from __future__ import annotations

import pytest
import torch

from fusion.opinion_router import GlobalOpinionRouter


def _branch_inputs(
    *,
    manifest_alive: float = 0.0,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    probabilities = {
        "api": torch.tensor([[0.9, 0.1]], dtype=torch.float32),
        "graph": torch.tensor([[0.1, 0.9]], dtype=torch.float32),
        "manifest": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    }
    reliability = {
        name: torch.tensor([0.8], dtype=torch.float32)
        for name in probabilities
    }
    alive = {
        "api": torch.tensor([1.0]),
        "graph": torch.tensor([1.0]),
        "manifest": torch.tensor([manifest_alive]),
    }
    return probabilities, reliability, alive


def test_global_conflict_is_normalized_over_alive_branches_only() -> None:
    router = GlobalOpinionRouter()
    probabilities, reliability, alive = _branch_inputs()

    prepared = router.prepare_route_inputs(probabilities, reliability, alive)
    per_branch = prepared["reliability_weighted_cross_modal_conflict"]
    global_conflict = prepared["global_cross_modal_conflict"]

    assert isinstance(per_branch, torch.Tensor)
    assert isinstance(global_conflict, torch.Tensor)
    assert per_branch[0, 2].item() == pytest.approx(0.0)
    assert global_conflict.item() == pytest.approx(
        per_branch[0, :2].mean().item()
    )


def test_router_rejects_soft_availability_values() -> None:
    router = GlobalOpinionRouter()
    probabilities, reliability, alive = _branch_inputs(manifest_alive=0.5)

    with pytest.raises(ValueError, match="hard binary"):
        router.prepare_route_inputs(probabilities, reliability, alive)
