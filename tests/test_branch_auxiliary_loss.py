import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.losses import (
    compute_branch_auxiliary_loss,
    resolve_auxiliary_weight_mode,
)


def _outputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        f"{name}_logits_aux": torch.randn(batch_size, 2, requires_grad=True)
        for name in ("api", "graph", "manifest")
    }


def _evidence(batch_size: int = 2) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def test_branch_auxiliary_loss_missing_modality_zero():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(), torch.tensor([0, 1]), evidence, {"min_aux_weight": 0.2}
    )
    assert torch.equal(diagnostics["aux_weight_api"], torch.zeros(2))


def test_branch_auxiliary_loss_low_integrity_has_min_weight():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(1), torch.tensor([0]), evidence, {"min_aux_weight": 0.2}
    )
    assert torch.allclose(diagnostics["aux_weight_graph"], torch.tensor([0.2]))


def test_branch_auxiliary_loss_detaches_integrity():
    evidence = _evidence(2).requires_grad_()
    loss, _ = compute_branch_auxiliary_loss(
        _outputs(), torch.tensor([0, 1]), evidence, {"detach_reliability_for_aux": True}
    )
    loss.backward()
    assert evidence.grad is None


def test_branch_auxiliary_loss_averages_active_branches():
    outputs = _outputs(2)
    labels = torch.tensor([0, 1])

    loss, diagnostics = compute_branch_auxiliary_loss(
        outputs, labels, _evidence(2), {"min_aux_weight": 0.2}
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
        _evidence(2),
        {"min_aux_weight": 0.2},
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
        _evidence(2),
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


def test_alive_masked_uniform_removes_only_continuous_integrity_weight():
    evidence = _evidence(2)
    evidence[:, EvidenceIndex.API_INTEGRITY] = torch.tensor([0.0, 0.7])
    evidence[0, EvidenceIndex.API_ALIVE] = 0.0

    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(2),
        torch.tensor([0, 1]),
        evidence,
        {"auxiliary_weight_mode": "alive_masked_uniform"},
    )

    assert torch.equal(diagnostics["aux_weight_api"], torch.tensor([0.0, 1.0]))
    assert diagnostics["aux_uses_integrity_weight"].item() == 0.0
    assert diagnostics["aux_uses_alive_mask"].item() == 1.0


def test_unmasked_uniform_is_an_explicit_separate_control():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0

    _, diagnostics = compute_branch_auxiliary_loss(
        _outputs(1),
        torch.tensor([0]),
        evidence,
        {"auxiliary_weight_mode": "unmasked_uniform"},
    )

    assert torch.equal(diagnostics["aux_weight_api"], torch.ones(1))
    assert diagnostics["aux_uses_integrity_weight"].item() == 0.0
    assert diagnostics["aux_uses_alive_mask"].item() == 0.0


def test_auxiliary_weight_mode_resolution_uses_only_explicit_mode():
    assert (
        resolve_auxiliary_weight_mode(
            {"auxiliary_weight_mode": "alive_masked_uniform"}
        )
        == "alive_masked_uniform"
    )
    assert resolve_auxiliary_weight_mode({}) == "unmasked_uniform"
    for removed_key in ("reliability_weighted_aux", "integrity_weighted_aux"):
        try:
            resolve_auxiliary_weight_mode({removed_key: False})
        except ValueError as exc:
            assert "Removed loss configuration keys" in str(exc)
        else:
            raise AssertionError(f"removed key {removed_key} was accepted")
