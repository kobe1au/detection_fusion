import pytest
import torch
import torch.nn.functional as F

from fusion.constants import AvailabilityIndex
from fusion.losses import (
    compute_branch_auxiliary_loss,
    resolve_auxiliary_weight_mode,
)


def _outputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        f"{name}_logits_aux": torch.randn(batch_size, 2, requires_grad=True)
        for name in ("api", "graph", "manifest")
    }


def _availability(batch_size: int = 2) -> torch.Tensor:
    return torch.ones(batch_size, AvailabilityIndex.BASE_DIM)


def test_branch_auxiliary_loss_missing_modality_zero():
    availability = _availability()
    availability[:, AvailabilityIndex.API_ALIVE] = 0.0
    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(),
        torch.tensor([0, 1]),
        availability,
        {"auxiliary_weight_mode": "alive_masked_uniform"},
    )
    assert torch.equal(diagnostics["aux_weight_api"], torch.zeros(2))


def test_branch_auxiliary_loss_averages_active_branches():
    outputs = _outputs(2)
    labels = torch.tensor([0, 1])

    loss, diagnostics = compute_branch_auxiliary_loss(
        outputs,
        labels,
        _availability(2),
        {"auxiliary_weight_mode": "alive_masked_uniform"},
    )

    expected = torch.stack(
        [F.cross_entropy(logits, labels) for logits in outputs.values()]
    ).mean()
    assert torch.allclose(loss, expected)
    assert diagnostics["aux_active_branch_count"].item() == 3.0


def test_branch_auxiliary_loss_exposes_exactly_three_branch_weights():
    outputs = _outputs(2)
    labels = torch.tensor([0, 1])

    loss, diagnostics = compute_branch_auxiliary_loss(
        outputs,
        labels,
        _availability(2),
        {"auxiliary_weight_mode": "alive_masked_uniform"},
    )

    expected = torch.stack(
        [
            F.cross_entropy(outputs[f"{name}_logits_aux"], labels)
            for name in ("api", "graph", "manifest")
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    assert diagnostics["aux_active_branch_count"].item() == 3.0
    assert {
        key for key in diagnostics if key.startswith("aux_weight_")
    } == {"aux_weight_api", "aux_weight_graph", "aux_weight_manifest"}


def test_branch_aux_weights_change_relative_branch_contribution():
    labels = torch.tensor([0, 1])
    good = torch.tensor([[5.0, -5.0], [-5.0, 5.0]], requires_grad=True)
    bad = torch.tensor([[-5.0, 5.0], [5.0, -5.0]], requires_grad=True)
    outputs = {
        "api_logits_aux": good,
        "graph_logits_aux": bad,
    }
    loss, _ = compute_branch_auxiliary_loss(
        outputs,
        labels,
        _availability(2),
        {
            "branch_aux_weights": {
                "api": 3.0,
                "graph": 1.0,
                "manifest": 0.0,
            }
        },
    )
    expected = (
        3.0 * F.cross_entropy(good, labels)
        + F.cross_entropy(bad, labels)
    ) / 4.0
    assert torch.allclose(loss, expected)


def test_alive_masked_uniform_uses_only_hard_availability():
    availability = _availability(2)
    availability[0, AvailabilityIndex.API_ALIVE] = 0.0

    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(2),
        torch.tensor([0, 1]),
        availability,
        {"auxiliary_weight_mode": "alive_masked_uniform"},
    )

    assert torch.equal(diagnostics["aux_weight_api"], torch.tensor([0.0, 1.0]))
    assert diagnostics["aux_uses_alive_mask"].item() == 1.0


def test_unmasked_uniform_is_an_explicit_separate_control():
    availability = _availability(1)
    availability[:, AvailabilityIndex.API_ALIVE] = 0.0

    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(1),
        torch.tensor([0]),
        availability,
        {"auxiliary_weight_mode": "unmasked_uniform"},
    )

    assert torch.equal(diagnostics["aux_weight_api"], torch.ones(1))
    assert diagnostics["aux_uses_alive_mask"].item() == 0.0


def test_auxiliary_weight_mode_resolution_uses_only_explicit_mode():
    assert (
        resolve_auxiliary_weight_mode(
            {"auxiliary_weight_mode": "alive_masked_uniform"}
        )
        == "alive_masked_uniform"
    )
    assert resolve_auxiliary_weight_mode({}) == "unmasked_uniform"
    with pytest.raises(ValueError, match="auxiliary_weight_mode"):
        resolve_auxiliary_weight_mode(
            {"auxiliary_weight_mode": "integrity"}
        )
    with pytest.raises(ValueError, match="Unsupported loss configuration"):
        resolve_auxiliary_weight_mode({"unregistered_option": False})
