from __future__ import annotations

import pytest
import torch

from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import compute_posthoc_calibration_loss, routing_soft_oracle_loss
from fusion.train import build_run_identity


BRANCHES = ("api", "graph", "manifest")


class _FixedReliabilityCalibrator(torch.nn.Module):
    def __init__(self, values: tuple[float, float, float]):
        super().__init__()
        self.values = values

    def forward(
        self, evidence: torch.Tensor, *_args, **_kwargs
    ) -> dict[str, torch.Tensor]:
        outputs = {
            f"predicted_reliability_{name}": evidence.new_full(
                (evidence.size(0),), value
            )
            for name, value in zip(BRANCHES, self.values)
        }
        return outputs


def _evidence(batch_size: int = 2, *, all_missing: bool = False) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    if all_missing:
        for index in (
            EvidenceIndex.API_ALIVE,
            EvidenceIndex.GRAPH_ALIVE,
            EvidenceIndex.MANIFEST_ALIVE,
        ):
            evidence[:, index] = 0.0
        for index in (
            EvidenceIndex.API_INTEGRITY,
            EvidenceIndex.GRAPH_INTEGRITY,
            EvidenceIndex.MANIFEST_INTEGRITY,
        ):
            evidence[:, index] = 0.0
    return evidence


def _logits() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[4.0, -4.0], [0.2, -0.2]]),
        torch.tensor([[-0.2, 0.2], [-4.0, 4.0]]),
        torch.tensor([[0.3, -0.3], [-0.3, 0.3]]),
    )


def _fusion(**routing_overrides) -> DiscountProbabilityFusion:
    routing = {
        "enabled": True,
        "mode": "learned",
        "risk_mode": "learned",
        "train_end_to_end": False,
        "posthoc_refine": True,
        "prediction_loss_weight": 1.0,
        "risk_loss_weight": 1.0,
        "acceptance_score_mode": "fused_risk",
        **routing_overrides,
    }
    return DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": routing,
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )


def _with_aux(outputs: dict[str, torch.Tensor], logits: tuple[torch.Tensor, ...]):
    outputs.update(
        {f"{name}_logits_aux": value for name, value in zip(BRANCHES, logits)}
    )
    return outputs


def _stage_config(
    *,
    prediction: float,
    risk: float,
    route_oracle: float = 0.0,
    route_oracle_temperature: float = 1.0,
) -> dict:
    return {
        "reliability_calibration": {"weight": 0.0},
        "probability_calibration": {"weight": 0.0},
        "routing": {
            "enabled": True,
            "posthoc_refine": True,
            "calibration_weight": 1.0,
            "prediction_loss_weight": prediction,
            "route_oracle_loss_weight": route_oracle,
            "route_oracle_temperature": route_oracle_temperature,
            "risk_loss_weight": risk,
            "risk_loss": "bce",
        },
    }


def test_final_classifier_is_mixture_and_is_independent_of_risk_probability():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    with torch.no_grad():
        fusion.opinion_router.risk_bias.fill_(-8.0)
        low_risk = fusion(*logits, _evidence())
        fusion.opinion_router.risk_bias.fill_(8.0)
        high_risk = fusion(*logits, _evidence())

    assert torch.allclose(low_risk["final_prob"], low_risk["routing_mixture_prob"])
    assert torch.allclose(high_risk["final_prob"], high_risk["routing_mixture_prob"])
    assert torch.allclose(low_risk["final_prob"], high_risk["final_prob"])
    assert torch.all(high_risk["routing_risk_probability"] > low_risk["routing_risk_probability"])
    assert torch.all(high_risk["acceptance_score"] < low_risk["acceptance_score"])


def test_route_nll_does_not_update_risk_parameters():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    outputs = _with_aux(fusion(*logits, _evidence()), logits)
    loss, diagnostics = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 1]),
        _evidence(),
        _stage_config(prediction=1.0, risk=0.0),
    )
    loss.backward()

    assert diagnostics["routing_prediction_loss"] > 0.0
    assert any(parameter.grad is not None for parameter in fusion.routing_distribution_parameters())
    assert all(parameter.grad is None for parameter in fusion.routing_risk_parameters())


def test_route_prior_uses_positive_learnable_common_log_odds_scale():
    fusion = _fusion(route_conflict_enabled=False)
    router = fusion.opinion_router
    assert router is not None
    assert torch.nn.functional.softplus(router.raw_route_prior_beta).item() == pytest.approx(
        1.0
    )
    assert any(
        parameter is router.raw_route_prior_beta
        for parameter in fusion.routing_distribution_parameters()
    )
    assert all(
        parameter is not router.raw_route_prior_beta
        for parameter in fusion.routing_risk_parameters()
    )

    evidence = _evidence()
    reliability = torch.tensor([0.90, 0.60, 0.20])
    for index, value in zip(
        (
            EvidenceIndex.API_INTEGRITY,
            EvidenceIndex.GRAPH_INTEGRITY,
            EvidenceIndex.MANIFEST_INTEGRITY,
        ),
        reliability,
    ):
        evidence[:, index] = value
    fusion.set_calibration_active(True)
    outputs = fusion(*_logits(), evidence)
    expected = torch.softmax(torch.logit(reliability), dim=-1).expand(2, -1)
    assert torch.allclose(
        outputs["routing_prior_branch_distribution"], expected, atol=1.0e-6
    )
    assert outputs["routing_prefit_uniform_prior_active"].sum().item() == 0.0
    (-outputs["routing_branch_distribution"][:, 0].log().mean()).backward()
    assert router.raw_route_prior_beta.grad is not None
    assert router.raw_route_prior_beta.grad.abs().item() > 0.0


def test_encoder_training_route_is_uniform_over_alive_branches():
    fusion = _fusion()
    evidence = _evidence()
    for index, value in zip(
        (
            EvidenceIndex.API_INTEGRITY,
            EvidenceIndex.GRAPH_INTEGRITY,
            EvidenceIndex.MANIFEST_INTEGRITY,
        ),
        (0.95, 0.35, 0.05),
    ):
        evidence[:, index] = value

    outputs = fusion(*_logits(), evidence)
    expected = torch.full((2, 3), 1.0 / 3.0)
    assert torch.allclose(outputs["routing_prior_branch_distribution"], expected)
    assert torch.allclose(outputs["routing_branch_distribution"], expected)
    assert outputs["fusion_weights"].shape == (2, 3)
    assert torch.allclose(outputs["fusion_weights"], expected)
    assert torch.all(
        outputs["routing_prefit_uniform_prior_active"] == 1.0
    )

    missing = evidence.clone()
    missing[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    missing[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    missing_outputs = fusion(*_logits(), missing)
    expected_missing = torch.tensor([[0.5, 0.0, 0.5]]).expand(2, -1)
    assert torch.allclose(
        missing_outputs["routing_prior_branch_distribution"], expected_missing
    )
    assert torch.allclose(
        missing_outputs["routing_branch_distribution"], expected_missing
    )
    assert torch.allclose(
        missing_outputs["fusion_weights"], expected_missing
    )


def test_posthoc_route_uses_fitted_i1_instead_of_raw_integrity():
    fusion = _fusion(route_conflict_enabled=False)
    fitted_reliability = (0.20, 0.50, 0.80)
    fusion.reliability_calibrator = _FixedReliabilityCalibrator(
        fitted_reliability
    )
    fusion.set_calibration_active(True)

    evidence = _evidence()
    # Reverse the raw-integrity ordering so this test fails if the routed path
    # bypasses I1 after calibration.
    for index, value in zip(
        (
            EvidenceIndex.API_INTEGRITY,
            EvidenceIndex.GRAPH_INTEGRITY,
            EvidenceIndex.MANIFEST_INTEGRITY,
        ),
        (0.95, 0.50, 0.05),
    ):
        evidence[:, index] = value

    outputs = fusion(*_logits(), evidence)
    expected = torch.softmax(
        torch.logit(torch.tensor(fitted_reliability)), dim=-1
    ).expand(2, -1)
    assert torch.allclose(
        outputs["routing_prior_branch_distribution"], expected, atol=1.0e-6
    )
    assert torch.allclose(outputs["routing_branch_distribution"], expected)
    assert outputs["routing_common_scale_reliability_active"].sum().item() == 2.0
    assert outputs["routing_prefit_uniform_prior_active"].sum().item() == 0.0


def test_route_soft_oracle_is_detached_and_respects_alive_mask():
    route_scores = torch.tensor(
        [[0.60, 0.25, 0.15], [0.60, 0.25, 0.15]], requires_grad=True
    )
    route_distribution = torch.softmax(route_scores, dim=-1)
    branch_probabilities = {
        "api": torch.tensor([[0.90, 0.10], [0.01, 0.99]]),
        "graph": torch.tensor([[0.20, 0.80], [0.20, 0.80]]),
        "manifest": torch.tensor([[0.10, 0.90], [0.80, 0.20]]),
    }
    outputs = {
        "routing_branch_distribution": route_distribution,
        "routing_scores": route_scores,
    }
    branch_log_probabilities = []
    for name, probability in branch_probabilities.items():
        log_probability = probability.log().requires_grad_()
        outputs[f"calibrated_log_prob_{name}"] = log_probability
        branch_log_probabilities.append(log_probability)
    evidence = _evidence()
    evidence[1, EvidenceIndex.API_ALIVE] = 0.0

    loss, diagnostics = routing_soft_oracle_loss(
        outputs,
        torch.tensor([0, 1]),
        evidence,
        temperature=0.5,
    )
    loss.backward()

    assert diagnostics["routing_route_oracle_valid_sample_count"].item() == 2
    assert diagnostics["routing_route_oracle_top1_agreement"].item() == pytest.approx(
        0.5
    )
    assert route_scores.grad is not None
    assert route_scores.grad[0, 0] < 0.0
    assert route_scores.grad[1, 1] < 0.0
    assert all(value.grad is None for value in branch_log_probabilities)


def test_route_soft_oracle_retains_gradient_for_collapsed_branch():
    route_scores = torch.tensor([[20.0, -20.0, -20.0]], requires_grad=True)
    outputs = {
        "routing_scores": route_scores,
        "routing_branch_distribution": torch.softmax(route_scores, dim=-1),
        "calibrated_log_prob_api": torch.tensor([[-0.01, -5.0]]),
        "calibrated_log_prob_graph": torch.tensor([[-5.0, -0.01]]),
        "calibrated_log_prob_manifest": torch.tensor([[-0.01, -5.0]]),
    }

    loss, _ = routing_soft_oracle_loss(
        outputs,
        torch.tensor([1]),
        _evidence(batch_size=1),
        temperature=0.01,
    )
    loss.backward()

    assert route_scores.grad is not None
    assert route_scores.grad[0, 1].item() < -0.99


def test_route_soft_oracle_updates_only_route_parameters():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    outputs = _with_aux(fusion(*logits, _evidence()), logits)
    loss, diagnostics = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 1]),
        _evidence(),
        _stage_config(prediction=0.0, route_oracle=0.5, risk=0.0),
    )
    loss.backward()

    assert diagnostics["routing_route_oracle_loss"] > 0.0
    assert diagnostics["routing_route_oracle_loss_weight"] == pytest.approx(0.5)
    assert any(
        parameter.grad is not None
        for parameter in fusion.routing_distribution_parameters()
    )
    assert all(parameter.grad is None for parameter in fusion.routing_risk_parameters())


def test_risk_bce_does_not_update_route_parameters():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    outputs = _with_aux(fusion(*logits, _evidence()), logits)
    loss, diagnostics = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 1]),
        _evidence(),
        _stage_config(prediction=0.0, risk=1.0),
    )
    loss.backward()

    assert diagnostics["routing_risk_loss"] > 0.0
    assert all(parameter.grad is None for parameter in fusion.routing_distribution_parameters())
    assert any(parameter.grad is not None for parameter in fusion.routing_risk_parameters())


def test_all_missing_samples_are_masked_from_proper_losses_but_forced_rejected():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    evidence = _evidence(all_missing=True)
    outputs = _with_aux(fusion(*logits, evidence), logits)
    loss, diagnostics = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 1]),
        evidence,
        _stage_config(prediction=1.0, risk=1.0),
    )
    loss.backward()

    assert diagnostics["routing_valid_sample_count"] == 0
    assert loss.item() == pytest.approx(0.0)
    assert torch.equal(outputs["routing_has_available"], torch.zeros(2))
    assert torch.equal(outputs["routing_risk_probability"], torch.ones(2))
    assert torch.equal(outputs["acceptance_score"], torch.zeros(2))


def test_prefit_encoder_path_uses_neutral_prior_and_no_learned_risk():
    fusion = _fusion()
    outputs = fusion(*_logits(), _evidence())
    assert torch.equal(outputs["routing_learned_components_active"], torch.zeros(2))
    assert torch.equal(outputs["routing_risk_probability"], torch.zeros(2))
    assert torch.allclose(
        outputs["routing_branch_distribution"],
        outputs["routing_prior_branch_distribution"],
    )
    assert torch.allclose(
        outputs["routing_branch_distribution"], torch.full((2, 3), 1.0 / 3.0)
    )


def test_static_route_and_learned_risk_are_independent_modes():
    fusion = _fusion(mode="prior_only", risk_mode="learned")
    assert fusion.routing_distribution_parameters() == []
    assert len(fusion.routing_risk_parameters()) == 2


@pytest.mark.parametrize(
    "removed_key,value",
    [
        ("use_disagreement", True),
        ("risk_enabled", True),
        ("target_loss_weight", 1.0),
        ("mass_constraint", "hard"),
        ("hidden_dim", 8),
    ],
)
def test_removed_i2_v1_keys_fail_fast(removed_key, value):
    with pytest.raises(ValueError, match="Removed I2-v1"):
        _fusion(**{removed_key: value})


def test_learned_components_cannot_disable_posthoc_refinement():
    with pytest.raises(ValueError, match="posthoc_refine=true"):
        _fusion(posthoc_refine=False)


def test_run_identity_reports_v2_route_and_risk_semantics():
    cfg = {
        "method": {"name": "v2"},
        "model": {"fusion_mode": "discount_probability"},
        "fusion": {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "mode": "learned",
                "risk_mode": "learned",
                "route_conflict_enabled": True,
                "risk_conflict_enabled": False,
                "prediction_loss_weight": 1.0,
                "risk_loss_weight": 1.0,
                "risk_loss": "bce",
                "acceptance_score_mode": "fused_risk",
            },
            "reliability_calibration": {
                "enabled": True,
                "use_prediction_margin": True,
                "use_predicted_class_feature": True,
            },
        },
        "calibration": {"enabled": True},
    }
    identity = build_run_identity(cfg, "v2", 42)
    assert identity["routing_mode"] == "learned"
    assert identity["routing_risk_mode"] == "learned"
    assert identity["routing_route_conflict_enabled"] is True
    assert identity["routing_risk_conflict_enabled"] is False
    assert identity["routing_acceptance_score_mode"] == "fused_risk"
    assert "routing_mass_constraint" not in identity
