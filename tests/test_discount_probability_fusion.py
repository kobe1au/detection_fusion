import torch
import torch.nn as nn

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


def test_inactive_calibrator_fallback_features_use_configured_missing_support():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    evidence[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0
    outputs = _run(
        evidence,
        config={
            "reliability_calibration": {
                "enabled": True,
                "missing_relation_support": 0.75,
                "use_relation_evidence": True,
            }
        },
    )

    features = outputs["reliability_features_api"]
    assert features[0, 2].item() == 0.75
    assert features[0, 3].item() == 0.75
    assert features[0, 4].item() == 0.75


def test_high_manifest_conflict_reduces_manifest_discount():
    low = _evidence(1)
    high = low.clone()
    high[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 1.0
    assert _run(high)["discount_manifest"].item() < _run(low)["discount_manifest"].item()


class _ConstantReliabilityCalibrator(nn.Module):
    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        value = torch.full(
            (evidence.size(0),),
            0.8,
            dtype=evidence.dtype,
            device=evidence.device,
        )
        return {
            f"predicted_reliability_{name}": value
            for name in ("api", "graph", "manifest", "joint")
        }


def test_calibrated_reliability_uses_explicit_relation_factors_once():
    fusion = DiscountProbabilityFusion(
        {
            "use_confidence_proxy": False,
            "use_support_discount": True,
            "use_conflict_discount": True,
            "reliability_calibration": {"enabled": True},
        }
    )
    fusion.reliability_calibrator = _ConstantReliabilityCalibrator()
    fusion.set_calibration_active(True)

    favorable = _evidence(1)
    unfavorable = favorable.clone()
    unfavorable[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 0.1
    unfavorable[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.1
    unfavorable[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.9
    unfavorable[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.9

    favorable_out = fusion(*_logits(1), favorable)
    unfavorable_out = fusion(*_logits(1), unfavorable)

    assert torch.all(favorable_out["discounts"] > unfavorable_out["discounts"])
    assert favorable_out["explicit_relation_factors_active"].item() == 1.0
    assert unfavorable_out["explicit_relation_factors_active"].item() == 1.0


def test_high_manifest_conflict_reduces_joint_discount():
    low = _evidence(1)
    high = low.clone()
    high[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.9
    high[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.9

    low_out = _run(low, config={"use_confidence_proxy": False})
    high_out = _run(high, config={"use_confidence_proxy": False})

    assert high_out["discount_joint"].item() < low_out["discount_joint"].item()
    assert high_out["joint_conflict_factor"].item() < 1.0


def test_support_discount_can_be_disabled():
    low = _evidence(1)
    low[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 0.0
    low[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0
    high = low.clone()
    high[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 1.0
    high[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 1.0

    low_out = _run(
        low,
        config={"use_support_discount": False, "use_confidence_proxy": False},
    )
    high_out = _run(
        high,
        config={"use_support_discount": False, "use_confidence_proxy": False},
    )
    assert torch.allclose(low_out["discounts"], high_out["discounts"])


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


def test_reliability_discount_exponent_tempers_fusion_discount():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.25
    common = {
        "use_support_discount": False,
        "use_conflict_discount": False,
        "use_confidence_proxy": False,
    }

    strict = _run(evidence, config={**common, "reliability_discount_exponent": 1.0})
    tempered = _run(evidence, config={**common, "reliability_discount_exponent": 0.5})

    assert torch.allclose(strict["discount_api"], torch.tensor([0.25]))
    assert torch.allclose(tempered["discount_api"], torch.tensor([0.5]))
    assert tempered["discount_api"].item() > strict["discount_api"].item()
    assert tempered["reliability_discount_exponent"].item() == 0.5


def test_reliability_can_drive_acceptance_without_fusion_discount():
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.25
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.25
    evidence[:, EvidenceIndex.MANIFEST_INTEGRITY] = 0.25
    evidence[:, EvidenceIndex.CODE_INTEGRITY] = 0.25

    outputs = _run(
        evidence,
        config={
            "use_reliability_discount": False,
            "use_reliability_acceptance": True,
            "use_support_discount": False,
            "use_conflict_discount": False,
            "use_confidence_proxy": False,
        },
    )

    assert torch.allclose(outputs["fusion_weights"], torch.full((1, 4), 0.25))
    assert outputs["total_reliability"].item() == 0.25
    assert outputs["reliability_discount_active"].item() == 0.0
    assert outputs["reliability_acceptance_active"].item() == 1.0


def test_branch_competence_prior_scales_fusion_weights_after_fit():
    fusion = DiscountProbabilityFusion(
        {
            "branch_competence_prior": {"enabled": True},
            "use_reliability_discount": False,
            "use_support_discount": False,
            "use_conflict_discount": False,
            "use_confidence_proxy": False,
        }
    )
    fusion.set_branch_competence_prior([1.0, 0.5, 0.5, 0.5])

    outputs = fusion(*_logits(1), _evidence(1))

    assert outputs["branch_competence_active"].item() == 1.0
    assert outputs["branch_competence_prior_api"].item() == 1.0
    assert torch.allclose(
        outputs["fusion_weights"],
        torch.tensor([[0.4, 0.2, 0.2, 0.2]]),
        atol=1e-6,
    )


def test_weight_sharpening_gamma_emphasizes_larger_discounts():
    base_cfg = {
        "branch_competence_prior": {"enabled": True},
        "use_reliability_discount": False,
        "use_support_discount": False,
        "use_conflict_discount": False,
        "use_confidence_proxy": False,
    }
    flat = DiscountProbabilityFusion(base_cfg)
    flat.set_branch_competence_prior([1.0, 0.5, 0.5, 0.5])
    sharp = DiscountProbabilityFusion({**base_cfg, "weight_sharpening_gamma": 2.0})
    sharp.set_branch_competence_prior([1.0, 0.5, 0.5, 0.5])

    flat_out = flat(*_logits(1), _evidence(1))
    sharp_out = sharp(*_logits(1), _evidence(1))

    assert sharp_out["fusion_weight_api"].item() > flat_out["fusion_weight_api"].item()
    assert sharp_out["weight_sharpening_gamma"].item() == 2.0


def test_visible_integrity_modifier_scales_evidential_trust_after_reference():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "yager",
            "use_reliability_discount": False,
            "visible_integrity_modifier": {
                "enabled": True,
                "beta": 1.0,
                "min_value": 0.5,
            },
        }
    )
    fusion.set_visible_integrity_reference([1.0, 1.0, 1.0])
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.5
    evidence[:, EvidenceIndex.API_ENCODER_COVERAGE] = 0.5

    outputs = fusion(*_logits(1), evidence)

    assert outputs["visible_integrity_modifier_active"].item() == 1.0
    assert torch.allclose(outputs["effective_api_integrity"], torch.tensor([0.25]))
    assert torch.allclose(outputs["visible_modifier_api"], torch.tensor([0.25]))
    assert torch.allclose(outputs["visible_modifier_factor_api"], torch.tensor([0.625]))
    assert torch.allclose(outputs["discount_api"], torch.tensor([0.625]))
    assert outputs["discount_graph"].item() == 1.0
    assert outputs["discount_manifest"].item() == 1.0
