from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import compute_posthoc_calibration_loss, compute_robust_loss
from fusion.opinion_router import GlobalOpinionRouter
from fusion.train import (
    GATE_DIAGNOSTIC_KEYS,
    build_run_identity,
    uses_routing_calibration_scenarios,
)


BRANCHES = ("api", "graph", "manifest")


def _router_inputs(batch_size: int = 1):
    beliefs = {
        "api": torch.tensor([[0.7, 0.1]]).repeat(batch_size, 1),
        "graph": torch.tensor([[0.2, 0.6]]).repeat(batch_size, 1),
        "manifest": torch.tensor([[0.4, 0.4]]).repeat(batch_size, 1),
    }
    uncertainties = {
        name: torch.full((batch_size,), 0.2) for name in BRANCHES
    }
    reliability = {
        "api": torch.full((batch_size,), 0.9),
        "graph": torch.full((batch_size,), 0.6),
        "manifest": torch.full((batch_size,), 0.3),
    }
    alive = {name: torch.ones(batch_size) for name in BRANCHES}
    visible = {name: torch.ones(batch_size) for name in BRANCHES}
    return beliefs, uncertainties, reliability, alive, visible


def _evidence(batch_size: int = 2) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.6
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.6
    evidence[:, EvidenceIndex.MANIFEST_INTEGRITY] = 0.6
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _logits(batch_size: int = 2):
    return (
        torch.tensor([[2.0, -1.0]]).repeat(batch_size, 1),
        torch.tensor([[-1.0, 2.0]]).repeat(batch_size, 1),
        torch.tensor([[0.5, -0.2]]).repeat(batch_size, 1),
        torch.tensor([[0.1, -0.1]]).repeat(batch_size, 1),
    )


def test_prior_only_matches_mean_correctness_mass_exactly():
    router = GlobalOpinionRouter(hidden_dim=8, mode="prior_only")
    outputs = router(*_router_inputs())

    assert outputs["prior_known_mass"].item() == pytest.approx(0.6)
    assert outputs["known_mass"].item() == pytest.approx(0.6)
    assert outputs["weights"][0].tolist() == pytest.approx(
        [0.3, 0.2, 0.1, 0.4]
    )
    assert outputs["weights"].sum(dim=-1).item() == pytest.approx(1.0)


def test_learned_router_cannot_exceed_prior_known_mass():
    router = GlobalOpinionRouter(hidden_dim=8, mode="learned")
    with torch.no_grad():
        final = router.residual[-1]
        final.weight.zero_()
        final.bias.copy_(
            torch.tensor(
                [40.0, -40.0, 20.0, math.log(0.25 / 0.75)]
            )
        )

    inputs = _router_inputs()
    outputs = router(*inputs)
    branch_mass = outputs["weights"][:, :3].sum(dim=-1)

    assert outputs["known_retention"].item() == pytest.approx(0.25)
    assert branch_mass.item() == pytest.approx(0.15)
    assert torch.all(branch_mass <= outputs["prior_known_mass"] + 1.0e-7)
    assert outputs["weights"].sum(dim=-1).item() == pytest.approx(1.0)

    beliefs, uncertainties, reliability, alive, visible = inputs
    low_visible = {name: value * 0.1 for name, value in visible.items()}
    degraded = router(
        beliefs, uncertainties, reliability, alive, low_visible
    )
    degraded_branch_mass = degraded["weights"][:, :3].sum(dim=-1)
    assert degraded["prior_known_mass"].item() == pytest.approx(0.06)
    assert degraded_branch_mass.item() == pytest.approx(0.015)
    assert torch.all(
        degraded_branch_mass <= degraded["prior_known_mass"] + 1.0e-7
    )


def test_learned_retention_gate_keeps_gradient_after_opposite_updates():
    router = GlobalOpinionRouter(
        hidden_dim=8, mode="learned", initial_known_retention=0.9
    )
    optimizer = torch.optim.SGD(router.parameters(), lr=1.0)

    # First move toward keeping more known mass (the all-correct direction).
    first = router(*_router_inputs())
    (-first["known_retention"].mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # The opposite, all-wrong direction must still reach the gate after that
    # update; a clamp-based gate used to become permanently gradient-dead here.
    second = router(*_router_inputs())
    second["known_retention"].mean().backward()
    gate_grad = router.residual[-1].bias.grad[-1]

    assert torch.isfinite(gate_grad)
    assert gate_grad.abs().item() > 0.0


def test_known_only_removes_unknown_but_all_missing_remains_vacuous():
    router = GlobalOpinionRouter(hidden_dim=8, mode="known_only")
    inputs = _router_inputs()
    outputs = router(*inputs)

    assert outputs["prior_known_mass"].item() == pytest.approx(1.0)
    assert outputs["reliability_prior_known_mass"].item() == pytest.approx(0.6)
    assert outputs["weights"][0, -1].item() == pytest.approx(0.0)
    assert outputs["weights"][0, :3].sum().item() == pytest.approx(1.0)

    beliefs, uncertainties, reliability, _alive, visible = inputs
    missing = {name: torch.zeros(1) for name in BRANCHES}
    all_missing = router(
        beliefs, uncertainties, reliability, missing, visible
    )
    assert torch.equal(all_missing["weights"][0, :3], torch.zeros(3))
    assert all_missing["weights"][0, -1].item() == pytest.approx(1.0)
    assert all_missing["uncertainty"].item() == pytest.approx(1.0)


def test_disagreement_switch_keeps_observation_but_zeros_router_feature():
    router = GlobalOpinionRouter(
        hidden_dim=8, mode="prior_only", use_disagreement=False
    )
    outputs = router(*_router_inputs())

    assert torch.any(outputs["observed_pairwise_disagreement"] > 0.0)
    assert torch.equal(
        outputs["routing_pairwise_disagreement"],
        torch.zeros_like(outputs["routing_pairwise_disagreement"]),
    )
    assert outputs["disagreement_feature_active"].item() == 0.0


def test_router_training_and_posthoc_parameter_switches_are_atomic():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "train_end_to_end": False,
                "posthoc_refine": False,
            },
        }
    )
    router_ids = {id(parameter) for parameter in fusion.opinion_router.parameters()}
    frozen_ids = {
        id(parameter) for parameter in fusion.encoder_training_frozen_parameters()
    }

    assert fusion.routing_encoder_training_parameters() == []
    assert fusion.routing_calibration_parameters() == []
    assert router_ids <= frozen_ids

    posthoc_only = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "train_end_to_end": False,
                "posthoc_refine": True,
            },
        }
    )
    assert posthoc_only.routing_encoder_training_parameters() == []
    assert {
        id(parameter)
        for parameter in posthoc_only.routing_calibration_parameters()
    } == {id(parameter) for parameter in posthoc_only.opinion_router.parameters()}


@pytest.mark.parametrize(
    ("mode", "field"),
    [
        ("unknown_only", "acceptance_score_unknown_only"),
        ("fused_certainty", "acceptance_score_fused_certainty"),
        ("conflict_only", "acceptance_score_conflict_only"),
        ("product", "acceptance_score_product"),
        ("current_product", "acceptance_score_product"),
    ],
)
def test_atomic_acceptance_score_modes_keep_backward_compatible_output(mode, field):
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "mode": "prior_only",
                "acceptance_score_mode": mode,
            },
        }
    )
    outputs = fusion(*_logits(), _evidence())

    assert torch.allclose(outputs["acceptance_score"], outputs[field])
    assert torch.allclose(
        outputs["acceptance_score_product"],
        outputs["acceptance_score_fused_certainty"]
        * outputs["acceptance_score_conflict_only"],
    )
    assert torch.allclose(
        outputs["acceptance_score_unknown_only"],
        1.0 - outputs["routing_weight_unknown"],
    )


def test_unknown_only_acceptance_rejects_non_routed_fusion():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "cumulative",
            "routing": {
                "enabled": False,
                "acceptance_score_mode": "unknown_only",
            },
        }
    )

    with pytest.raises(ValueError, match="requires fusion.combination=routed"):
        fusion(*_logits(), _evidence())


def test_posthoc_routing_loss_weights_support_prediction_only():
    labels = torch.tensor([0, 1])
    final_raw_logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 1.5]], requires_grad=True
    )
    final_log_prob = F.log_softmax(
        final_raw_logits, dim=-1
    )
    routing_weights = torch.tensor(
        [[0.3, 0.2, 0.1, 0.4], [0.2, 0.3, 0.1, 0.4]],
        requires_grad=True,
    )
    outputs = {
        "routing_active": torch.ones(2),
        "routing_weights": routing_weights,
        "final_logits": final_log_prob,
        "api_logits_aux": torch.tensor([[2.0, -1.0], [2.0, -1.0]]),
        "graph_logits_aux": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
        "manifest_logits_aux": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
    }
    config = {
        "reliability_calibration": {"weight": 0.0},
        "probability_calibration": {"weight": 0.0},
        "routing": {
            "enabled": True,
            "calibration_weight": 1.0,
            "target_loss_weight": 0.0,
            "prediction_loss_weight": 2.0,
        },
    }

    loss, diagnostics = compute_posthoc_calibration_loss(
        outputs, labels, _evidence(), config
    )
    expected = 2.0 * F.nll_loss(final_log_prob, labels)

    assert torch.allclose(loss, expected)
    assert diagnostics["routing_calibration_loss"] == pytest.approx(0.0)
    assert diagnostics["routing_target_loss_weight"] == pytest.approx(0.0)
    assert diagnostics["routing_prediction_loss_weight"] == pytest.approx(2.0)
    loss.backward()
    assert final_raw_logits.grad is not None

    incompatible = {
        **config,
        "routing": {
            **config["routing"],
            "mode": "known_only",
            "target_loss_weight": 1.0,
        },
    }
    with pytest.raises(ValueError, match="known_only"):
        compute_posthoc_calibration_loss(
            outputs, labels, _evidence(), incompatible
        )


def test_new_integrity_aux_key_and_legacy_precedence():
    labels = torch.tensor([0, 1])
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    extra = {
        f"{name}_logits_aux": logits.clone()
        for name in ("api", "graph", "manifest", "joint")
    }
    evidence = _evidence()

    _loss, new_parts = compute_robust_loss(
        logits,
        labels,
        extra,
        {
            "branch_aux_weight": 1.0,
            "integrity_weighted_aux": True,
            "reliability_weighted_aux": None,
        },
        evidence=evidence,
    )
    assert "integrity_weighted_aux_loss" in new_parts

    _loss, legacy_parts = compute_robust_loss(
        logits,
        labels,
        extra,
        {
            "branch_aux_weight": 1.0,
            "integrity_weighted_aux": True,
            "reliability_weighted_aux": False,
        },
        evidence=evidence,
    )
    assert "integrity_weighted_aux_loss" not in legacy_parts


def test_router_identity_records_atomic_switches_and_effective_weights():
    cfg = {
        "fusion": {
            "combination": "routed",
            "use_reliability_discount": False,
            "visible_integrity_modifier": {"enabled": True},
            "routing": {
                "enabled": True,
                "mode": "prior_only",
                "use_disagreement": False,
                "train_end_to_end": False,
                "posthoc_refine": False,
                "target_loss_weight": 0.0,
                "prediction_loss_weight": 2.0,
                "acceptance_score_mode": "unknown_only",
            },
        },
        "loss": {
            "integrity_weighted_aux": True,
            "reliability_weighted_aux": None,
        },
        "calibration": {"enabled": True},
        "classification_threshold": {"enabled": False},
        "selective_prediction": {"enabled": False},
    }

    identity = build_run_identity(cfg, "atomic_router", 42)

    assert identity["routing_mode"] == "prior_only"
    assert identity["routing_disagreement_enabled"] is False
    assert identity["router_trained_end_to_end"] is False
    assert identity["router_posthoc_refinement_enabled"] is False
    assert identity["routing_target_loss_weight"] == pytest.approx(0.0)
    assert identity["routing_prediction_loss_weight"] == pytest.approx(2.0)
    assert identity["routing_acceptance_score_mode"] == "unknown_only"
    assert identity["router_encoder_training_reliability_source"] == "neutral_constant"
    assert identity["integrity_weighted_aux_enabled"] is True
    assert uses_routing_calibration_scenarios(cfg) is False


def test_router_audit_outputs_are_persisted_as_gate_diagnostics():
    expected = {
        "routing_known_mass",
        "routing_prior_known_mass",
        "routing_known_retention",
        "routing_disagreement_feature_active",
        "routing_mode_learned",
        "routing_train_end_to_end",
        "routing_posthoc_refine",
        "acceptance_score_unknown_only",
        "acceptance_score_fused_certainty",
        "acceptance_score_conflict_only",
        "acceptance_score_product",
    }

    assert expected <= set(GATE_DIAGNOSTIC_KEYS)
