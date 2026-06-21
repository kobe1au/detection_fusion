import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.losses import compute_reliability_weighted_aux_loss


def _outputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        f"{name}_logits_aux": torch.randn(batch_size, 2, requires_grad=True)
        for name in ("api", "graph", "manifest", "joint")
    }


def _evidence(batch_size: int = 2) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def test_reliability_weighted_aux_loss_missing_modality_zero():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    _, diagnostics = compute_reliability_weighted_aux_loss(
        _outputs(), torch.tensor([0, 1]), evidence, {"min_aux_weight": 0.2}
    )
    assert torch.equal(diagnostics["aux_weight_api"], torch.zeros(2))


def test_reliability_weighted_aux_loss_low_integrity_has_min_weight():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    _, diagnostics = compute_reliability_weighted_aux_loss(
        _outputs(1), torch.tensor([0]), evidence, {"min_aux_weight": 0.2}
    )
    assert torch.allclose(diagnostics["aux_weight_graph"], torch.tensor([0.2]))


def test_reliability_weighted_aux_loss_detaches_reliability():
    evidence = _evidence(2).requires_grad_()
    loss, _ = compute_reliability_weighted_aux_loss(
        _outputs(), torch.tensor([0, 1]), evidence, {"detach_reliability_for_aux": True}
    )
    loss.backward()
    assert evidence.grad is None


def test_reliability_weighted_aux_loss_averages_active_branches():
    outputs = _outputs(2)
    labels = torch.tensor([0, 1])

    loss, diagnostics = compute_reliability_weighted_aux_loss(
        outputs, labels, _evidence(2), {"min_aux_weight": 0.2}
    )

    expected = torch.stack(
        [F.cross_entropy(logits, labels) for logits in outputs.values()]
    ).mean()
    assert torch.allclose(loss, expected)
    assert diagnostics["aux_active_branch_count"].item() == 4.0

