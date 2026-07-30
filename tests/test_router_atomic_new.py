from __future__ import annotations

import pytest
import torch

from fusion.constants import AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import (
    routing_mixture_log_prob,
    routing_risk_per_sample_loss,
    routing_risk_target,
)
from fusion.train import build_run_identity


BRANCHES = ("api", "graph", "manifest")


class _FixedReliabilityCalibrator(torch.nn.Module):
    def __init__(self, values: tuple[float, float, float]):
        super().__init__()
        self.values = values

    def forward(
        self,
        branch_features: dict[str, torch.Tensor],
        *_args,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        reference = branch_features[BRANCHES[0]]
        outputs = {
            f"predicted_reliability_{name}": reference.new_full(
                (reference.size(0),), value
            )
            for name, value in zip(BRANCHES, self.values)
        }
        return outputs


def _evidence(batch_size: int = 2, *, all_missing: bool = False) -> torch.Tensor:
    evidence = torch.ones(batch_size, AvailabilityIndex.BASE_DIM)
    if all_missing:
        for index in (
            AvailabilityIndex.API_ALIVE,
            AvailabilityIndex.GRAPH_ALIVE,
            AvailabilityIndex.MANIFEST_ALIVE,
        ):
            evidence[:, index] = 0.0
    return evidence


def _logits() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[4.0, -4.0], [0.2, -0.2]]),
        torch.tensor([[-0.2, 0.2], [-4.0, 4.0]]),
        torch.tensor([[0.3, -0.3], [-0.3, 0.3]]),
    )


def _fusion(
    *,
    use_i1_reliability: bool = True,
    **routing_overrides,
) -> DiscountProbabilityFusion:
    routing = {
        "enabled": True,
        "mode": "learned",
        "risk_mode": "learned",
        "posthoc_refine": True,
        "prediction_loss_weight": 1.0,
        "risk_loss_weight": 1.0,
        **routing_overrides,
    }
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "use_i1_reliability": use_i1_reliability,
            "routing": routing,
            "reliability_calibration": {"enabled": False},
        }
    )
    assert fusion.opinion_router is not None
    fusion.opinion_router.set_risk_decision_threshold(0.0)
    return fusion


def _with_aux(outputs: dict[str, torch.Tensor], logits: tuple[torch.Tensor, ...]):
    outputs.update(
        {f"{name}_logits_aux": value for name, value in zip(BRANCHES, logits)}
    )
    return outputs


def _stage_config(
    *,
    prediction: float,
    risk: float,
) -> dict:
    return {
        "routing": {
            "enabled": True,
            "posthoc_refine": True,
            "prediction_loss_weight": prediction,
            "risk_loss_weight": risk,
            "risk_loss": "bce",
            "risk_target": "threshold_malware_false_negative",
            "classification_log_odds_threshold": 0.0,
        },
    }


def _direct_stage_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    routing = config["routing"]
    mixture_log_prob = routing_mixture_log_prob(outputs)
    valid = outputs["routing_has_available"].view(-1).bool()
    prediction_weight = float(routing["prediction_loss_weight"])
    risk_weight = float(routing["risk_loss_weight"])
    prediction_loss = (
        mixture_log_prob.sum() * 0.0
        if prediction_weight > 0.0
        else mixture_log_prob.new_zeros(())
    )
    if prediction_weight > 0.0:
        per_row = torch.nn.functional.nll_loss(
            mixture_log_prob, labels, reduction="none"
        )
        prediction_loss = (
            per_row[valid].mean() if bool(valid.any()) else per_row.sum() * 0.0
        )
    risk_training_logit = outputs["routing_risk_training_logit"]
    risk_loss = (
        risk_training_logit.sum() * 0.0
        if risk_weight > 0.0
        else mixture_log_prob.new_zeros(())
    )
    risk_valid = torch.zeros_like(valid)
    if risk_weight > 0.0:
        target, risk_valid, loss_type, _ = routing_risk_target(
            outputs,
            labels,
            routing,
            mixture_log_prob=mixture_log_prob,
            valid_routing=valid,
        )
        per_row = routing_risk_per_sample_loss(
            outputs["routing_risk_probability"],
            risk_training_logit,
            target,
            risk_valid,
            loss_type=loss_type,
        )
        risk_loss = (
            per_row[risk_valid].mean()
            if bool(risk_valid.any())
            else per_row.sum() * 0.0
        )
    total = prediction_weight * prediction_loss + risk_weight * risk_loss
    return total, {
        "routing_prediction_loss": float(prediction_loss.detach()),
        "routing_risk_loss": float(risk_loss.detach()),
        "routing_valid_sample_count": int(valid.sum()),
        "routing_risk_training_sample_count": int(risk_valid.sum()),
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
    predicted_malware = high_risk["routing_risk_predicted_malware"].bool()
    assert torch.equal(predicted_malware, torch.tensor([False, True]))
    assert torch.all(
        high_risk["routing_risk_probability"][~predicted_malware]
        > low_risk["routing_risk_probability"][~predicted_malware]
    )
    assert torch.equal(
        high_risk["routing_risk_probability"][predicted_malware],
        torch.zeros(1),
    )
    assert torch.all(
        high_risk["acceptance_score"][~predicted_malware]
        < low_risk["acceptance_score"][~predicted_malware]
    )


def test_route_nll_does_not_update_risk_parameters():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    outputs = _with_aux(fusion(*logits, _evidence()), logits)
    loss, diagnostics = _direct_stage_loss(
        outputs,
        torch.tensor([0, 1]),
        _stage_config(prediction=1.0, risk=0.0),
    )
    loss.backward()

    assert diagnostics["routing_prediction_loss"] > 0.0
    assert any(parameter.grad is not None for parameter in fusion.routing_distribution_parameters())
    assert all(parameter.grad is None for parameter in fusion.routing_risk_parameters())


def test_route_prior_uses_positive_learnable_common_log_odds_scale():
    fusion = _fusion()
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

    reliability_values = (0.90, 0.60, 0.20)
    fusion.reliability_calibrator = _FixedReliabilityCalibrator(
        reliability_values
    )
    fusion.set_calibration_active(True)
    outputs = fusion(*_logits(), _evidence())
    reliability = torch.tensor(reliability_values)
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
    missing[:, AvailabilityIndex.GRAPH_ALIVE] = 0.0
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


def test_posthoc_route_uses_fitted_i1_reliability():
    fusion = _fusion()
    fitted_reliability = (0.20, 0.50, 0.80)
    fusion.reliability_calibrator = _FixedReliabilityCalibrator(
        fitted_reliability
    )
    fusion.set_calibration_active(True)

    outputs = fusion(*_logits(), _evidence())
    expected = torch.softmax(
        torch.logit(torch.tensor(fitted_reliability)), dim=-1
    ).expand(2, -1)
    assert torch.allclose(
        outputs["routing_prior_branch_distribution"], expected, atol=1.0e-6
    )
    assert torch.allclose(outputs["routing_branch_distribution"], expected)
    assert outputs["routing_common_scale_reliability_active"].sum().item() == 2.0
    assert outputs["routing_prefit_uniform_prior_active"].sum().item() == 0.0


def test_router_consumes_normalized_branch_probabilities_without_uncertainty_api():
    fusion = _fusion()
    outputs = fusion(*_logits(), _evidence())

    for name in BRANCHES:
        probability = outputs[f"routing_input_probability_{name}"]
        torch.testing.assert_close(
            probability.sum(dim=-1),
            torch.ones(probability.size(0)),
        )
        assert f"routing_input_belief_{name}" not in outputs
        assert f"routing_input_uncertainty_{name}" not in outputs


def test_fusion_exports_exactly_the_three_risk_feature_semantics():
    fitted_reliability = (0.90, 0.55, 0.25)
    fusion = _fusion()
    fusion.reliability_calibrator = _FixedReliabilityCalibrator(
        fitted_reliability
    )
    fusion.set_calibration_active(True)
    outputs = fusion(*_logits(), _evidence())

    distribution = outputs["routing_branch_distribution"].detach()
    reliability = distribution.new_tensor(fitted_reliability).unsqueeze(0)
    expected_deficit = 1.0 - (distribution * reliability).sum(dim=-1)
    torch.testing.assert_close(
        outputs["routing_risk_reliability_deficit"],
        expected_deficit,
    )

    mixture = outputs["routing_mixture_prob"].detach()
    raw_log_odds = mixture[:, 1].log() - mixture[:, 0].log()
    expected_boundary = torch.where(
        raw_log_odds >= 0.0,
        torch.zeros_like(raw_log_odds),
        2.0 * torch.sigmoid(raw_log_odds),
    )
    torch.testing.assert_close(
        outputs["routing_risk_decision_boundary_proximity"],
        expected_boundary,
    )

    risk_conflict = outputs["routing_risk_global_cross_modal_conflict"]
    assert risk_conflict.shape == (2,)
    assert torch.isfinite(risk_conflict).all()
    assert torch.all((risk_conflict >= 0.0) & (risk_conflict <= 1.0))
    for removed_key in (
        "routing_conflict_penalty_mean",
        "routing_route_conflict_feature_active",
        "routing_route_conflict_feature_configured",
        "routing_risk_uncertainty_burden",
        "routing_risk_structural_conflict",
        "routing_risk_missing_fraction",
        "routing_mean_disagreement",
    ):
        assert removed_key not in outputs
    for name in BRANCHES:
        assert f"routing_conflict_penalty_{name}" not in outputs


def test_i1_disabled_route_is_alive_masked_uniform_and_has_no_parameters():
    fusion = _fusion(use_i1_reliability=False)
    fusion.set_calibration_active(True)

    assert fusion.routing_distribution_parameters() == []
    outputs = fusion(*_logits(), _evidence())
    torch.testing.assert_close(
        outputs["routing_branch_distribution"],
        torch.full((2, 3), 1.0 / 3.0),
    )

    evidence = _evidence()
    evidence[:, AvailabilityIndex.GRAPH_ALIVE] = 0.0
    missing_outputs = fusion(*_logits(), evidence)
    torch.testing.assert_close(
        missing_outputs["routing_branch_distribution"],
        torch.tensor([[0.5, 0.0, 0.5]]).expand(2, -1),
    )


def test_disabled_risk_features_share_effective_weights_in_loss_and_deployment():
    fusion = _fusion(
        use_i1_reliability=False,
        risk_conflict_enabled=False,
    )
    fusion.set_calibration_active(True)
    router = fusion.opinion_router
    assert router is not None
    logits = _logits()
    outputs = _with_aux(fusion(*logits, _evidence()), logits)

    effective_weights = router.effective_risk_feature_weights()
    assert effective_weights.numel() == 3
    assert effective_weights[0].item() == pytest.approx(0.0)
    assert effective_weights[2].item() == pytest.approx(0.0)
    assert effective_weights[1].item() > 0.0
    assert torch.equal(
        outputs["routing_risk_reliability_deficit"],
        torch.zeros(2),
    )
    assert torch.equal(
        outputs["routing_risk_global_cross_modal_conflict"],
        torch.zeros(2),
    )
    expected_logit = (
        router.risk_bias
        + effective_weights[1]
        * outputs["routing_risk_decision_boundary_proximity"]
    )
    torch.testing.assert_close(
        outputs["routing_risk_training_logit"],
        expected_logit,
    )

    loss, diagnostics = _direct_stage_loss(
        outputs,
        torch.tensor([0, 1]),
        _stage_config(prediction=0.0, risk=1.0),
    )
    loss.backward()
    assert diagnostics["routing_risk_loss"] > 0.0
    gradient = router.raw_risk_feature_weights.grad
    assert gradient is not None
    assert gradient[0].item() == pytest.approx(0.0, abs=1.0e-12)
    assert gradient[2].item() == pytest.approx(0.0, abs=1.0e-12)
    assert gradient[1].item() != pytest.approx(0.0, abs=1.0e-12)


def test_risk_bce_does_not_update_route_parameters():
    fusion = _fusion()
    fusion.set_calibration_active(True)
    logits = _logits()
    outputs = _with_aux(fusion(*logits, _evidence()), logits)
    loss, diagnostics = _direct_stage_loss(
        outputs,
        torch.tensor([0, 1]),
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
    loss, diagnostics = _direct_stage_loss(
        outputs,
        torch.tensor([0, 1]),
        _stage_config(prediction=1.0, risk=1.0),
    )
    loss.backward()

    assert diagnostics["routing_valid_sample_count"] == 0
    assert loss.item() == pytest.approx(0.0)
    assert torch.equal(outputs["routing_has_available"], torch.zeros(2))
    assert torch.equal(
        outputs["routing_branch_distribution"],
        torch.zeros(2, 3),
    )
    torch.testing.assert_close(
        outputs["routing_mixture_prob"],
        torch.full((2, 2), 0.5),
    )
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
        ("train_end_to_end", False),
        ("acceptance_score_mode", "fused_risk"),
        ("route_conflict_enabled", True),
    ],
)
def test_removed_i2_v1_keys_fail_fast(removed_key, value):
    with pytest.raises(ValueError, match="Removed or retired I2"):
        _fusion(**{removed_key: value})


def test_learned_components_cannot_disable_posthoc_refinement():
    with pytest.raises(ValueError, match="posthoc_refine=true"):
        _fusion(posthoc_refine=False)


def test_run_identity_reports_route_and_risk_semantics():
    cfg = {
        "method": {"name": "v2"},
        "model": {"fusion_mode": "discount_probability"},
        "fusion": {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "mode": "learned",
                "risk_mode": "learned",
                "risk_conflict_enabled": False,
                "prediction_loss_weight": 1.0,
                "risk_loss_weight": 1.0,
                "risk_loss": "bce",
            },
            "reliability_calibration": {
                "enabled": True,
                "use_evidential_certainty": True,
                "use_prediction_margin": True,
                "use_predicted_class_intercept": True,
            },
        },
        "calibration": {
            "enabled": True,
            "fit_perturbations": [
                "api_event_dropout",
                "graph_sparsify",
                "manifest_permission_mask",
            ],
            "perturb_strengths": [0.3, 0.5, 0.7],
        },
    }
    identity = build_run_identity(cfg, "v2", 42)
    assert identity["routing_mode"] == "learned"
    assert identity["routing_risk_mode"] == "learned"
    assert "routing_route_conflict_enabled" not in identity
    assert identity["routing_risk_conflict_enabled"] is False
    assert "routing_acceptance_score_mode" not in identity
    assert "routing_mass_constraint" not in identity
