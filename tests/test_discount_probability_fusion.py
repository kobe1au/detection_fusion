import torch

from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion


def _evidence(batch_size: int = 2) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _logits(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor([[3.0, -3.0]] * batch_size, requires_grad=True) for _ in range(4))


def _run(evidence: torch.Tensor, logits=None, config=None):
    logits = logits or _logits(evidence.size(0))
    return DiscountProbabilityFusion(config)(*logits, evidence)


def test_discount_fusion_weights_sum_to_one():
    outputs = _run(_evidence())
    assert torch.allclose(outputs["fusion_weights"].sum(dim=-1), torch.ones(2))


def test_missing_api_branch_weight_is_zero():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    assert torch.equal(_run(evidence)["fusion_weight_api"], torch.zeros(2))


def test_missing_graph_branch_weight_is_zero():
    evidence = _evidence()
    evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    assert torch.equal(_run(evidence)["fusion_weight_graph"], torch.zeros(2))


def test_missing_manifest_branch_weight_is_zero():
    evidence = _evidence()
    evidence[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    assert torch.equal(_run(evidence)["fusion_weight_manifest"], torch.zeros(2))


def test_unavailable_api_graph_relation_does_not_penalize_api_discount():
    low_support = _evidence(1)
    low_support[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    low_support[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 0.0
    high_support = low_support.clone()
    high_support[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 1.0

    assert torch.allclose(
        _run(low_support)["discount_api"],
        _run(high_support)["discount_api"],
    )


def test_unavailable_manifest_code_relation_does_not_penalize_manifest_discount():
    low_support = _evidence(1)
    low_support[:, EvidenceIndex.API_ALIVE] = 0.0
    low_support[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    low_support[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0
    high_support = low_support.clone()
    high_support[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 1.0

    assert torch.allclose(
        _run(low_support)["discount_manifest"],
        _run(high_support)["discount_manifest"],
    )


def test_no_applicable_relation_does_not_apply_joint_support_penalty():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    evidence[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0

    outputs = _run(evidence, config={"use_confidence_proxy": False})
    assert outputs["discount_joint"].item() == outputs["total_reliability"].item()


def test_high_manifest_conflict_reduces_manifest_discount():
    low = _evidence(1)
    high = low.clone()
    high[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 1.0
    assert _run(high)["discount_manifest"].item() < _run(low)["discount_manifest"].item()


def test_high_entropy_reduces_branch_discount():
    evidence = _evidence(1)
    peaked = list(_logits(1))
    uniform = list(_logits(1))
    uniform[0] = torch.zeros(1, 2, requires_grad=True)
    assert _run(evidence, tuple(uniform))["discount_api"].item() < _run(evidence, tuple(peaked))["discount_api"].item()


def test_probability_fusion_outputs_valid_distribution():
    outputs = _run(_evidence())
    assert torch.all(outputs["final_prob"] >= 0)
    assert torch.allclose(outputs["final_prob"].sum(dim=-1), torch.ones(2))
    assert torch.allclose(outputs["final_logits"].exp(), outputs["final_prob"], atol=1e-6)


def test_discount_detach_blocks_gradient_through_weights():
    evidence = _evidence(1).requires_grad_()
    outputs = _run(evidence, config={"detach_discount": True, "detach_confidence_proxy": True})
    assert outputs["fusion_weights"].requires_grad is False
    outputs["final_logits"].sum().backward()
    assert evidence.grad is None


def test_fallback_used_when_all_discounts_zero():
    evidence = _evidence(1)
    evidence[:, :4] = 0.0
    outputs = _run(evidence)
    assert outputs["fallback_used"].item() == 1.0
    assert torch.allclose(outputs["fusion_weights"], torch.full((1, 4), 0.25))


def test_fallback_preserves_missing_branch_zero_weight():
    evidence = _evidence(1)
    evidence[:, :4] = 0.0
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    outputs = _run(evidence)
    assert outputs["fallback_used"].item() == 1.0
    assert outputs["fusion_weight_api"].item() == 0.0
    assert torch.allclose(outputs["fusion_weights"].sum(dim=-1), torch.ones(1))
