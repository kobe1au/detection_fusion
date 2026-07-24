import math
from itertools import permutations

import pytest
import torch

from fusion.constants import AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.evidential import (
    combine_conflict_weighted_opinions,
    combine_ecml_opinions,
    predictive_opinion_conflict,
    combine_opinions,
    combine_opinions_with_diagnostics,
    evidential_loss,
    logits_to_opinion,
    opinion_to_prob,
    trust_discount,
)


def test_opinion_belief_and_uncertainty_sum_to_one():
    logits = torch.tensor([[3.0, -3.0], [0.0, 0.0], [-2.0, 5.0]])
    op = logits_to_opinion(logits)
    total = op["belief"].sum(dim=-1) + op["uncertainty"]
    assert torch.allclose(total, torch.ones(3), atol=1e-5)
    # Uniform logits -> maximum uncertainty for K=2 (u = 2 / (2+2) = 0.5).
    assert op["uncertainty"][1].item() > op["uncertainty"][0].item()
    assert op["expected_prob"].sum(dim=-1).allclose(torch.ones(3), atol=1e-5)


def test_trust_discount_moves_belief_to_uncertainty():
    logits = torch.tensor([[4.0, -4.0]])
    op = logits_to_opinion(logits)
    full_b, full_u = trust_discount(op["belief"], op["uncertainty"], torch.tensor([1.0]))
    half_b, half_u = trust_discount(op["belief"], op["uncertainty"], torch.tensor([0.5]))
    zero_b, zero_u = trust_discount(op["belief"], op["uncertainty"], torch.tensor([0.0]))
    assert torch.allclose(full_b, op["belief"], atol=1e-5)
    assert (half_b < full_b).all()
    assert half_u.item() > full_u.item()
    # Zero trust -> vacuous opinion (all uncertainty), the fusion identity.
    assert torch.allclose(zero_b, torch.zeros_like(zero_b), atol=1e-6)
    assert zero_u.item() == 1.0


def test_combination_diagnostics_expose_raw_conflict():
    beliefs = [
        torch.tensor([[0.8, 0.0]]),
        torch.tensor([[0.0, 0.8]]),
    ]
    uncertainties = [torch.tensor([0.2]), torch.tensor([0.2])]

    _belief, _uncertainty, diagnostics = combine_opinions_with_diagnostics(
        beliefs, uncertainties, rule="cumulative"
    )

    assert diagnostics["raw_conflict"].item() > 0.0


def test_vacuous_opinion_is_fusion_identity():
    b_a = torch.tensor([[0.7, 0.1]])
    u_a = torch.tensor([0.2])
    vac_b = torch.tensor([[0.0, 0.0]])
    vac_u = torch.tensor([1.0])
    fused_b, fused_u = combine_opinions(
        [b_a, vac_b], [u_a, vac_u], rule="dempster"
    )
    assert torch.allclose(fused_b, b_a, atol=1e-5)
    assert math.isclose(fused_u.item(), u_a.item(), abs_tol=1e-5)


def test_combine_opinions_three_sources_valid_distribution():
    beliefs = [
        torch.tensor([[0.6, 0.1], [0.2, 0.5]]),
        torch.tensor([[0.5, 0.2], [0.4, 0.3]]),
        torch.tensor([[0.7, 0.0], [0.1, 0.6]]),
    ]
    us = [torch.tensor([0.3, 0.3]), torch.tensor([0.3, 0.3]), torch.tensor([0.3, 0.3])]
    for rule in ("dempster", "cumulative", "log_pool"):
        b, u = combine_opinions(beliefs, us, rule=rule)
        prob = opinion_to_prob(b, u)
        assert torch.all(prob >= 0)
        assert prob.sum(dim=-1).allclose(torch.ones(2), atol=1e-5)


def test_symmetric_combination_rules_are_order_invariant_for_three_sources():
    beliefs = [
        torch.tensor([[0.70, 0.10]]),
        torch.tensor([[0.20, 0.65]]),
        torch.tensor([[0.45, 0.25]]),
    ]
    us = [torch.tensor([0.20]), torch.tensor([0.15]), torch.tensor([0.30])]
    for rule in ("dempster", "cumulative", "log_pool"):
        base_b, base_u = combine_opinions(beliefs, us, rule=rule)
        for order in permutations(range(3)):
            b, u = combine_opinions([beliefs[i] for i in order], [us[i] for i in order], rule=rule)
            assert torch.allclose(b, base_b, atol=1e-6), (rule, order)
            assert torch.allclose(u, base_u, atol=1e-6), (rule, order)


def test_cumulative_combination_is_finite_at_dogmatic_limit():
    beliefs = [
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.6, 0.4]], dtype=torch.float32),
    ]
    uncertainties = [
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([1.0e-30], dtype=torch.float32),
        torch.tensor([1.0e-20], dtype=torch.float32),
    ]

    belief, uncertainty = combine_opinions(
        beliefs, uncertainties, rule="cumulative"
    )

    assert torch.isfinite(belief).all()
    assert torch.isfinite(uncertainty).all()
    assert torch.all(belief >= 0.0)
    assert torch.allclose(
        belief.sum(dim=-1) + uncertainty,
        torch.ones(1),
        atol=1e-6,
    )


def test_conflict_weighted_opinion_is_valid_and_conflict_sensitive():
    aligned_beliefs = [
        torch.tensor([[0.8, 0.0]]),
        torch.tensor([[0.7, 0.1]]),
        torch.tensor([[0.75, 0.05]]),
    ]
    conflict_beliefs = [
        torch.tensor([[0.8, 0.0]]),
        torch.tensor([[0.0, 0.8]]),
        torch.tensor([[0.75, 0.05]]),
    ]
    us = [torch.tensor([0.2]), torch.tensor([0.2]), torch.tensor([0.2])]

    aligned_b, aligned_u, aligned_diag = combine_conflict_weighted_opinions(
        aligned_beliefs, us
    )
    conflict_b, conflict_u, conflict_diag = combine_conflict_weighted_opinions(
        conflict_beliefs, us
    )

    assert torch.allclose(aligned_b.sum(dim=-1) + aligned_u, torch.ones(1), atol=1e-5)
    assert torch.allclose(conflict_b.sum(dim=-1) + conflict_u, torch.ones(1), atol=1e-5)
    assert conflict_diag["raw_conflict"].item() > aligned_diag["raw_conflict"].item()
    assert conflict_u.item() > aligned_u.item()


def _opinion_from_evidence(evidence: torch.Tensor):
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1)
    return evidence / strength.unsqueeze(-1), evidence.size(-1) / strength


def test_ecml_aggregation_is_exact_binary_mean_evidence_opinion():
    evidence_a = torch.tensor([[2.0, 0.0]])
    evidence_b = torch.tensor([[0.0, 4.0]])
    belief_a, uncertainty_a = _opinion_from_evidence(evidence_a)
    belief_b, uncertainty_b = _opinion_from_evidence(evidence_b)

    belief, uncertainty, diagnostics = combine_ecml_opinions(
        [belief_a, belief_b], [uncertainty_a, uncertainty_b]
    )

    # mean evidence = [1, 2], alpha = [2, 3], S = 5.
    assert torch.allclose(belief, torch.tensor([[0.2, 0.4]]), atol=1e-6)
    assert torch.allclose(uncertainty, torch.tensor([0.4]), atol=1e-6)
    assert torch.allclose(
        opinion_to_prob(belief, uncertainty), torch.tensor([[0.4, 0.6]]), atol=1e-6
    )
    assert diagnostics["ecml_available_views"].item() == 2.0
    assert diagnostics["ecml_folded_evidence"].item() == pytest.approx(3.0)


def test_ecml_three_view_aggregation_matches_reference_ordered_fold():
    evidence_api = torch.tensor([[8.0, 0.0]])
    evidence_graph = torch.tensor([[0.0, 4.0]])
    evidence_manifest = torch.tensor([[2.0, 6.0]])
    opinions = [
        _opinion_from_evidence(evidence)
        for evidence in (evidence_api, evidence_graph, evidence_manifest)
    ]

    belief, uncertainty, diagnostics = combine_ecml_opinions(
        [opinion[0] for opinion in opinions],
        [opinion[1] for opinion in opinions],
    )

    # Official RCML fold: ((api + graph) / 2 + manifest) / 2.
    expected_evidence = torch.tensor([[3.0, 4.0]])
    expected_belief, expected_uncertainty = _opinion_from_evidence(
        expected_evidence
    )
    assert torch.allclose(belief, expected_belief, atol=1e-6)
    assert torch.allclose(uncertainty, expected_uncertainty, atol=1e-6)
    assert diagnostics["ecml_folded_evidence"].item() == pytest.approx(7.0)


def test_ecml_aggregation_excludes_unavailable_view_per_sample():
    evidence_a = torch.tensor([[4.0, 2.0], [4.0, 2.0]])
    evidence_b = torch.tensor([[100.0, 0.0], [0.0, 6.0]])
    belief_a, uncertainty_a = _opinion_from_evidence(evidence_a)
    belief_b, uncertainty_b = _opinion_from_evidence(evidence_b)

    belief, uncertainty, diagnostics = combine_ecml_opinions(
        [belief_a, belief_b],
        [uncertainty_a, uncertainty_b],
        availability_masks=torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
    )

    expected_evidence = torch.stack([evidence_a[0], evidence_b[1]])
    expected_belief, expected_uncertainty = _opinion_from_evidence(expected_evidence)
    assert torch.allclose(belief, expected_belief, atol=1e-6)
    assert torch.allclose(uncertainty, expected_uncertainty, atol=1e-6)
    assert torch.equal(diagnostics["ecml_available_views"], torch.ones(2))


def test_ecml_aggregation_treats_pipeline_vacuous_sentinel_as_missing():
    evidence = torch.tensor([[4.0, 2.0]])
    available_belief, available_uncertainty = _opinion_from_evidence(evidence)
    vacuous_belief = torch.zeros_like(available_belief)
    vacuous_uncertainty = torch.ones_like(available_uncertainty)

    belief, uncertainty, diagnostics = combine_ecml_opinions(
        [available_belief, vacuous_belief],
        [available_uncertainty, vacuous_uncertainty],
    )

    assert torch.allclose(belief, available_belief, atol=1e-6)
    assert torch.allclose(uncertainty, available_uncertainty, atol=1e-6)
    assert diagnostics["ecml_available_views"].item() == 1.0


def test_ecml_aggregation_all_missing_is_vacuous_and_dogmatic_limit_is_finite():
    vacuous_belief = torch.zeros(1, 2)
    vacuous_uncertainty = torch.ones(1)
    belief, uncertainty, diagnostics = combine_ecml_opinions(
        [vacuous_belief, vacuous_belief],
        [vacuous_uncertainty, vacuous_uncertainty],
    )
    assert torch.equal(belief, torch.zeros_like(belief))
    assert torch.equal(uncertainty, torch.ones_like(uncertainty))
    assert diagnostics["ecml_available_views"].item() == 0.0

    dogmatic_belief = torch.tensor([[1.0, 0.0]])
    dogmatic_uncertainty = torch.zeros(1)
    belief, uncertainty, _diagnostics = combine_ecml_opinions(
        [dogmatic_belief], [dogmatic_uncertainty]
    )
    assert torch.isfinite(belief).all() and torch.isfinite(uncertainty).all()
    assert torch.allclose(
        belief.sum(dim=-1) + uncertainty, torch.ones(1), atol=1e-6
    )
    assert belief[0, 0] > 0.999999


def test_predictive_opinion_conflict_detects_confident_disagreement():
    aligned = [
        torch.tensor([[0.85, 0.05]]),
        torch.tensor([[0.80, 0.10]]),
        torch.tensor([[0.82, 0.08]]),
    ]
    conflicting = [aligned[0], torch.tensor([[0.05, 0.85]]), aligned[2]]
    uncertainties = [torch.tensor([0.10])] * 3

    aligned_mean, aligned_max = predictive_opinion_conflict(aligned, uncertainties)
    conflict_mean, conflict_max = predictive_opinion_conflict(conflicting, uncertainties)

    assert conflict_mean.item() > aligned_mean.item()
    assert conflict_max.item() > aligned_max.item()
    assert 0.0 <= conflict_mean.item() <= 1.0


def test_inactive_reliability_calibrator_uses_neutral_prefit_trust():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "dempster",
            "reliability_calibration": {"enabled": True},
        }
    )
    availability = torch.ones(1, AvailabilityIndex.BASE_DIM)
    # The API certainty changes substantially, but before post-hoc calibration
    # its prefit trust must remain the Stage-1-neutral unit value.
    uncertain_api = torch.tensor([[0.0, 0.0]])
    certain_api = torch.tensor([[8.0, -8.0]])
    other = torch.tensor([[2.0, -2.0]])

    uncertain = fusion(uncertain_api, other, other, availability)
    certain = fusion(certain_api, other, other, availability)

    assert fusion.calibration_active is False
    assert uncertain["total_reliability"].item() == pytest.approx(
        certain["total_reliability"].item()
    )
    assert uncertain["total_reliability"].item() == pytest.approx(1.0)
    assert uncertain["evidential_certainty_api"].item() < certain[
        "evidential_certainty_api"
    ].item()


def test_evidential_loss_is_finite_and_differentiable():
    logits = torch.tensor([[2.0, -1.0], [-3.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = evidential_loss(logits, labels, anneal_coef=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_evidential_loss_anneal_zero_has_no_kl():
    logits = torch.tensor([[2.0, -1.0]])
    labels = torch.tensor([0])
    no_kl = evidential_loss(logits, labels, anneal_coef=0.0)
    with_kl = evidential_loss(logits, labels, anneal_coef=1.0)
    assert with_kl.item() >= no_kl.item()


def test_compute_robust_loss_edl_class_weight_balanced_is_finite():
    from fusion.losses import compute_robust_loss

    aux = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [1.5, -1.0]], requires_grad=True)
    extra = {
        "api_logits_aux": aux,
        "graph_logits_aux": aux,
        "manifest_logits_aux": aux,
        "final_is_log_probability": False,
    }
    labels = torch.tensor([0, 0, 1])  # imbalanced
    availability = torch.ones(3, AvailabilityIndex.BASE_DIM)
    loss, parts = compute_robust_loss(
        aux,
        labels,
        extra,
        {
            "branch_aux_weight": 0.0,
            "evidential_loss_weight": 0.5,
            "evidential": {"anneal_epochs": 5, "class_weight": "balanced"},
        },
        availability=availability,
        epoch=2,
    )
    assert torch.isfinite(loss)
    assert parts["evidential_loss"] >= 0.0
    loss.backward()
    assert aux.grad is not None and torch.isfinite(aux.grad).all()


def _availability(batch_size: int = 2) -> torch.Tensor:
    return torch.ones(batch_size, AvailabilityIndex.BASE_DIM)


def _logits(batch_size: int = 2):
    return tuple(
        torch.tensor([[3.0, -3.0]] * batch_size, requires_grad=True)
        for _ in range(3)
    )


def test_evidential_fusion_path_outputs_valid_distribution():
    fusion = DiscountProbabilityFusion({"combination": "dempster"})
    out = fusion(*_logits(), _availability())
    assert torch.all(out["final_prob"] >= 0)
    assert out["final_prob"].sum(dim=-1).allclose(torch.ones(2), atol=1e-5)
    assert torch.allclose(out["final_logits"].exp(), out["final_prob"], atol=1e-5)
    assert out["fusion_weights"].shape == (2, 3)
    assert torch.allclose(out["fusion_weights"].sum(dim=-1), torch.ones(2))


def test_routed_fusion_is_active_before_and_after_posthoc_calibration():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {"enabled": True},
            "reliability_calibration": {
                "enabled": True,
                "hidden_dim": 8,
            },
        }
    )
    training_output = fusion(*_logits(), _availability())
    assert torch.equal(training_output["routing_active"], torch.ones(2))
    assert torch.equal(training_output["calibration_active"], torch.zeros(2))
    assert torch.allclose(
        training_output["final_prob"].sum(dim=-1), torch.ones(2), atol=1e-6
    )

    fusion.set_calibration_active(True)
    routed = fusion(*_logits(), _availability())
    assert torch.equal(routed["routing_active"], torch.ones(2))
    assert torch.allclose(routed["final_prob"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(routed["fusion_weights"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert routed["fusion_weights"].shape == (2, 3)
    assert "routing_weight_risk" not in routed
    assert torch.all(
        (routed["routing_risk_probability"] >= 0.0)
        & (routed["routing_risk_probability"] <= 1.0)
    )


def test_routed_fusion_keeps_router_frozen_before_posthoc_calibration():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {"enabled": True},
            "reliability_calibration": {"enabled": True, "hidden_dim": 8},
        }
    )
    output = fusion(*_logits(), _availability())
    torch.nn.functional.nll_loss(
        output["final_logits"], torch.tensor([0, 1])
    ).backward()

    assert output["routing_active"].sum().item() == 2.0
    assert torch.equal(
        output["routing_learned_components_active"], torch.zeros(2)
    )
    assert all(
        parameter.grad is None
        for parameter in fusion.routing_calibration_parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in fusion.encoder_training_frozen_parameters()
    )


def test_routed_fusion_masks_unavailable_modality_exactly():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {"enabled": True},
            "reliability_calibration": {"enabled": True},
        }
    )
    fusion.set_calibration_active(True)
    availability = _availability()
    availability[:, AvailabilityIndex.API_ALIVE] = 0.0
    routed = fusion(*_logits(), availability)
    assert torch.equal(routed["fusion_weight_api"], torch.zeros(2))


def test_missing_modality_logits_do_not_change_routed_fusion_or_conflict():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {"enabled": True},
            "reliability_calibration": {"enabled": True},
        }
    )
    fusion.set_calibration_active(True)
    availability = _availability()
    availability[:, AvailabilityIndex.API_ALIVE] = 0.0
    base_logits = list(_logits())
    changed_logits = list(_logits())
    changed_logits[0] = torch.tensor(
        [[-12.0, 12.0], [-12.0, 12.0]], requires_grad=True
    )

    reference = fusion(*base_logits, availability)
    changed = fusion(*changed_logits, availability)

    for key in (
        "final_prob",
        "fusion_weights",
        "predictive_conflict",
        "predictive_conflict_max",
    ):
        assert torch.allclose(reference[key], changed[key], atol=1.0e-7)


@pytest.mark.parametrize("rule", ["ecml", "conflict_weighted_opinion"])
def test_ecml_evidential_fusion_outputs_valid_distribution(rule):
    fusion = DiscountProbabilityFusion({"combination": rule})
    logits = (
        torch.tensor([[-4.0, 4.0], [-4.0, 4.0]], requires_grad=True),
        torch.tensor([[4.0, -4.0], [4.0, -4.0]], requires_grad=True),
        torch.tensor([[-4.0, 4.0], [-4.0, 4.0]], requires_grad=True),
    )
    out = fusion(*logits, _availability())
    assert torch.all(out["final_prob"] >= 0.0)
    assert out["final_prob"].sum(dim=-1).allclose(torch.ones(2), atol=1e-5)
    assert torch.all(out["raw_conflict"] >= 0.0)
    assert torch.all(out["fused_uncertainty"] >= 0.0)


def test_evidential_dead_modality_gets_zero_weight():
    availability = _availability()
    availability[:, AvailabilityIndex.API_ALIVE] = 0.0
    out = DiscountProbabilityFusion({"combination": "dempster"})(
        *_logits(), availability
    )
    assert torch.equal(out["fusion_weight_api"], torch.zeros(2))


def test_evidential_fp32_for_half_inputs():
    fusion = DiscountProbabilityFusion({"combination": "dempster"})
    logits = tuple(value.half() for value in _logits())
    out = fusion(*logits, _availability().half())
    assert out["final_logits"].dtype == torch.float32
    assert out["acceptance_score"].dtype == torch.float32


def _toy_batch():
    from torch_geometric.data import Batch, Data

    items = []
    for i in range(2):
        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor(i % 2),
        )
        data.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        items.append(data)
    batch = Batch.from_data_list(items)
    batch.api_ids = torch.randint(1, 32, (12,), dtype=torch.long)
    batch.api_type_ids = torch.randint(0, 4, (12,), dtype=torch.long)
    batch.api_sensitive_mask = torch.zeros(12)
    batch.api_batch = torch.cat([torch.full((6,), i, dtype=torch.long) for i in range(2)])
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_semantic_category_counts = torch.rand(2, 12)
    batch.graph_semantic_category_counts = torch.rand(2, 12)
    batch.api_category_counts = batch.api_semantic_category_counts
    batch.graph_category_counts = batch.graph_semantic_category_counts
    batch.manifest_x = torch.rand(2, 32)
    batch.manifest_category_counts = torch.rand(2, 12)
    batch.manifest_stats = torch.rand(2, 11)
    for name in ("q_api", "q_graph", "q_manifest"):
        setattr(batch, name, torch.ones(2, 1))
    batch.q_align = torch.ones(2, 1) * 0.8
    for name in ("api_alive", "graph_alive", "manifest_alive"):
        setattr(batch, name, torch.ones(2, 1))
    for name in (
        "api_integrity",
        "graph_integrity",
        "manifest_integrity",
        "code_integrity",
    ):
        setattr(batch, name, torch.ones(2, 1))
    return batch


def test_full_evidential_pipeline_forward_loss_backward():
    """End-to-end: model -> evidential fusion -> EDL loss -> backward."""
    from fusion.losses import compute_robust_loss
    from fusion.model import TriModalRobustModel

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="discount_probability",
        api_num_hash_buckets=64,
        api_type_vocab_size=16,
        api_emb_dim=32,
        api_hidden_dim=64,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=32,
        graph_hidden=32,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=64,
        manifest_in_dim=32,
        manifest_emb_dim=32,
        manifest_hidden_dim=64,
        discount_fusion_config={
            "combination": "dempster",
            "reliability_calibration": {"enabled": True},
        },
    )
    batch = _toy_batch()
    logits, extra = model(batch)
    assert logits.shape == (2, 2)
    assert extra["fusion_weights"].shape == (2, 3)
    assert torch.allclose(extra["fusion_weights"].sum(dim=-1), torch.ones(2))
    assert torch.allclose(logits.exp().sum(dim=-1), torch.ones(2), atol=1e-4)  # log-prob
    assert "acceptance_score" in extra and "fused_uncertainty" in extra

    loss, parts = compute_robust_loss(
        logits,
        torch.tensor([0, 1]),
        extra,
        {
            "branch_aux_weight": 0.25,
            "auxiliary_weight_mode": "alive_masked_uniform",
            "evidential_loss_weight": 0.1,
            "evidential": {"anneal_epochs": 10},
        },
        availability=extra.get("fusion_availability"),
        epoch=3,
    )
    assert torch.isfinite(loss)
    assert parts["evidential_loss"] >= 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
