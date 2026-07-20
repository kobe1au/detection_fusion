from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.evidential import (
    combine_ecml_opinions,
    dirichlet_expected_ce_loss,
    opinion_to_dirichlet_alpha,
)
from fusion.losses import _ecml_conflict_consistency_loss, compute_robust_loss


VIEW_NAMES = ("api", "graph", "manifest")


def _observable_evidence(batch_size: int) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _paper_loss_config(objective: str, *, mask_unavailable: bool) -> dict:
    return {
        "objective": objective,
        "branch_aux_weight": 0.0,
        "reliability_calibration_weight": 0.0,
        "probability_calibration_weight": 0.0,
        "evidential_loss_weight": 0.0,
        "label_smoothing": 0.0,
        objective: {
            "anneal_epochs": 10,
            "mask_unavailable_views": mask_unavailable,
            **({"consistency_weight": 0.7} if objective == "ecml" else {}),
        },
    }


def _fusion(combination: str) -> DiscountProbabilityFusion:
    return DiscountProbabilityFusion(
        {
            "combination": combination,
            "opinion_source": "evidential",
            "evidence_activation": "softplus",
            "use_reliability_discount": False,
            "use_hard_alive_mask": True,
            "use_confidence_proxy": False,
            "use_support_discount": False,
            "use_conflict_discount": False,
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
            "routing": {"enabled": False},
        }
    )


def _opinion_from_evidence(evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    belief = evidence / strength
    uncertainty = float(evidence.size(-1)) / strength.view(-1)
    return belief, uncertainty


def _official_ecml_sequential_average(
    evidences: list[torch.Tensor], availability: torch.Tensor
) -> torch.Tensor:
    """Reference the fixed-order fold used by the official RCML implementation."""
    result = torch.zeros_like(evidences[0])
    has_result = torch.zeros(
        evidences[0].size(0), 1, dtype=torch.bool, device=evidences[0].device
    )
    for index, view_evidence in enumerate(evidences):
        available = availability[:, index : index + 1].bool()
        first = available & ~has_result
        subsequent = available & has_result
        result = torch.where(first, view_evidence, result)
        result = torch.where(subsequent, (result + view_evidence) / 2.0, result)
        has_result = has_result | available
    return result


def _manual_ecml_conflict(alphas: list[torch.Tensor]) -> torch.Tensor:
    probabilities = [alpha / alpha.sum(dim=-1, keepdim=True) for alpha in alphas]
    certainties = [
        1.0 - float(alpha.size(-1)) / alpha.sum(dim=-1) for alpha in alphas
    ]
    pair_sum = probabilities[0].new_zeros(probabilities[0].size(0))
    for left in range(len(alphas)):
        for right in range(left + 1, len(alphas)):
            projected_distance = 0.5 * (
                probabilities[left] - probabilities[right]
            ).abs().sum(dim=-1)
            pair_sum = (
                pair_sum
                + projected_distance * certainties[left] * certainties[right]
            )
    # The official expression sums both pair orders and divides by V - 1.
    return (2.0 * pair_sum / float(len(alphas) - 1)).mean()


def _official_tmc_ds_pair(alpha_a: torch.Tensor, alpha_b: torch.Tensor) -> torch.Tensor:
    num_classes = alpha_a.size(-1)
    strength_a = alpha_a.sum(dim=-1, keepdim=True)
    strength_b = alpha_b.sum(dim=-1, keepdim=True)
    belief_a = (alpha_a - 1.0) / strength_a
    belief_b = (alpha_b - 1.0) / strength_b
    uncertainty_a = float(num_classes) / strength_a
    uncertainty_b = float(num_classes) / strength_b
    outer = belief_a.unsqueeze(-1) * belief_b.unsqueeze(-2)
    conflict = outer.sum(dim=(-2, -1)) - outer.diagonal(dim1=-2, dim2=-1).sum(-1)
    normalizer = (1.0 - conflict).unsqueeze(-1)
    fused_belief = (
        belief_a * belief_b
        + belief_a * uncertainty_b
        + uncertainty_a * belief_b
    ) / normalizer
    fused_uncertainty = uncertainty_a * uncertainty_b / normalizer
    fused_strength = float(num_classes) / fused_uncertainty
    return fused_belief * fused_strength + 1.0


def test_dirichlet_expected_ce_matches_paper_formula() -> None:
    alpha = torch.tensor(
        [[4.5, 1.2, 2.1], [1.1, 3.8, 2.4]], dtype=torch.float64
    )
    labels = torch.tensor([0, 1])
    anneal = 0.35

    target = F.one_hot(labels, num_classes=alpha.size(-1)).to(alpha.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    expected_ce = (
        target * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    wrong_class_alpha = target + (1.0 - target) * alpha
    expected_kl = kl_divergence(
        Dirichlet(wrong_class_alpha), Dirichlet(torch.ones_like(alpha))
    )
    expected = (expected_ce + anneal * expected_kl).mean()

    actual = dirichlet_expected_ce_loss(
        alpha, labels, anneal_coef=anneal
    )

    assert actual == pytest.approx(expected.item(), rel=1.0e-12, abs=1.0e-12)


def test_tmc_fused_alpha_matches_official_sequential_dempster_rule() -> None:
    fusion = _fusion("dempster")
    logits = [
        torch.tensor([[2.2, -0.7], [-0.4, 1.8]]),
        torch.tensor([[0.8, 1.1], [1.5, -0.2]]),
        torch.tensor([[1.7, 0.3], [-0.8, 2.4]]),
    ]
    branch_alphas = [F.softplus(value) + 1.0 for value in logits[:3]]
    expected = _official_tmc_ds_pair(branch_alphas[0], branch_alphas[1])
    expected = _official_tmc_ds_pair(expected, branch_alphas[2])

    outputs = fusion(*logits, _observable_evidence(2))

    assert torch.allclose(
        outputs["dirichlet_alpha_fused"], expected, rtol=2.0e-5, atol=2.0e-6
    )


def test_tmc_objective_is_exact_sum_of_three_views_and_fused_loss() -> None:
    labels = torch.tensor([0, 1])
    alphas = {
        "api": torch.tensor([[4.0, 1.2], [1.3, 3.5]], requires_grad=True),
        "graph": torch.tensor([[3.2, 1.1], [1.2, 2.8]], requires_grad=True),
        "manifest": torch.tensor([[2.6, 1.4], [1.1, 4.0]], requires_grad=True),
        "fused": torch.tensor([[6.0, 1.2], [1.1, 6.5]], requires_grad=True),
    }
    extra = {
        f"dirichlet_alpha_{name}": value for name, value in alphas.items()
    }
    anneal = 0.5
    expected_terms = [
        dirichlet_expected_ce_loss(alpha, labels, anneal_coef=anneal)
        for alpha in alphas.values()
    ]
    expected = torch.stack(expected_terms).sum()

    actual, parts = compute_robust_loss(
        torch.zeros(2, 2),
        labels,
        extra,
        _paper_loss_config("tmc", mask_unavailable=False),
        evidence=_observable_evidence(2),
        epoch=5,
        materialize_diagnostics=False,
    )

    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    assert parts["tmc_objective_active"].item() == 1.0
    assert parts["ecml_objective_active"].item() == 0.0
    assert parts["ce"].item() == 0.0

    actual.backward()
    for alpha in alphas.values():
        assert alpha.grad is not None
        assert torch.isfinite(alpha.grad).all()
        assert alpha.grad.abs().sum().item() > 0.0


def test_ecml_uses_official_sequential_evidence_average_with_missing_views() -> None:
    api = torch.tensor(
        [[8.0, 1.0], [7.0, 2.0], [6.0, 3.0], [5.0, 4.0], [4.0, 5.0], [3.0, 6.0]]
    )
    graph = torch.tensor(
        [[1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, 9.0], [5.0, 10.0], [6.0, 11.0]]
    )
    manifest = torch.tensor(
        [[4.0, 3.0], [5.0, 2.0], [6.0, 1.0], [7.0, 3.0], [8.0, 2.0], [9.0, 1.0]]
    )
    availability = torch.tensor(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )
    evidences = [api, graph, manifest]
    opinions = [_opinion_from_evidence(value) for value in evidences]

    belief, uncertainty, diagnostics = combine_ecml_opinions(
        [item[0] for item in opinions],
        [item[1] for item in opinions],
        availability_masks=availability,
    )
    actual_alpha = opinion_to_dirichlet_alpha(belief, uncertainty)
    expected_evidence = _official_ecml_sequential_average(evidences, availability)

    assert torch.allclose(actual_alpha, expected_evidence + 1.0, atol=1.0e-6)
    assert torch.equal(
        diagnostics["ecml_available_views"], availability.sum(dim=-1).float()
    )
    # With no available source, ECML must return the vacuous Dirichlet.
    assert torch.equal(actual_alpha[-1], torch.ones_like(actual_alpha[-1]))


def test_ecml_objective_matches_accuracy_average_plus_consistency() -> None:
    labels = torch.tensor([0, 1])
    view_evidences = [
        torch.tensor([[4.0, 0.4], [0.6, 3.2]], requires_grad=True),
        torch.tensor([[2.8, 0.9], [1.1, 2.6]], requires_grad=True),
        torch.tensor([[3.3, 1.2], [0.8, 4.1]], requires_grad=True),
    ]
    view_alphas = [value + 1.0 for value in view_evidences]
    fused_evidence = (view_evidences[0] + view_evidences[1]) / 2.0
    fused_evidence = (fused_evidence + view_evidences[2]) / 2.0
    fused_alpha = fused_evidence + 1.0
    extra = {
        "dirichlet_alpha_api": view_alphas[0],
        "dirichlet_alpha_graph": view_alphas[1],
        "dirichlet_alpha_manifest": view_alphas[2],
        "dirichlet_alpha_fused": fused_alpha,
    }
    anneal = 0.5
    accuracy_terms = [
        dirichlet_expected_ce_loss(alpha, labels, anneal_coef=anneal)
        for alpha in [*view_alphas, fused_alpha]
    ]
    expected_consistency = _manual_ecml_conflict(view_alphas)
    expected = torch.stack(accuracy_terms).mean() + 0.7 * expected_consistency

    actual, parts = compute_robust_loss(
        torch.zeros(2, 2),
        labels,
        extra,
        _paper_loss_config("ecml", mask_unavailable=False),
        evidence=_observable_evidence(2),
        epoch=5,
        materialize_diagnostics=False,
    )

    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    assert torch.allclose(
        parts["ecml_conflict_consistency_loss"],
        expected_consistency.detach(),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    actual.backward()
    for evidence in view_evidences:
        assert evidence.grad is not None
        assert torch.isfinite(evidence.grad).all()
        assert evidence.grad.abs().sum().item() > 0.0


def test_ecml_missing_view_consistency_renormalizes_remaining_pairs() -> None:
    alphas = {
        "api": torch.tensor([[6.0, 1.0]]),
        "graph": torch.tensor([[1.0, 7.0]]),
        "manifest": torch.tensor([[1.5, 5.0]]),
        "fused": torch.tensor([[2.0, 3.0]]),
    }
    alive = {
        "api": torch.ones(1),
        "graph": torch.zeros(1),
        "manifest": torch.ones(1),
    }
    api_prob = alphas["api"] / alphas["api"].sum(dim=-1, keepdim=True)
    manifest_prob = alphas["manifest"] / alphas["manifest"].sum(
        dim=-1, keepdim=True
    )
    projected_distance = 0.5 * (api_prob - manifest_prob).abs().sum(dim=-1)
    api_certainty = 1.0 - 2.0 / alphas["api"].sum(dim=-1)
    manifest_certainty = 1.0 - 2.0 / alphas["manifest"].sum(dim=-1)
    # Two available views produce two ordered pairs and V_eff - 1 == 1.
    expected = 2.0 * projected_distance * api_certainty * manifest_certainty

    actual = _ecml_conflict_consistency_loss(
        alphas, alive, mask_unavailable=True
    )

    assert torch.allclose(actual, expected.mean(), rtol=1.0e-6, atol=1.0e-7)


@pytest.mark.parametrize(
    ("combination", "objective"), [("dempster", "tmc"), ("ecml", "ecml")]
)
def test_missing_view_has_no_direct_or_fused_gradient(
    combination: str, objective: str
) -> None:
    fusion = _fusion(combination)
    logits = [
        torch.tensor([[2.5, -1.0], [2.0, -0.5]], requires_grad=True),
        torch.tensor([[-0.5, 2.0], [-1.0, 3.0]], requires_grad=True),
        torch.tensor([[1.5, -0.2], [-0.3, 2.2]], requires_grad=True),
    ]
    labels = torch.tensor([0, 1])
    evidence = _observable_evidence(2)
    evidence[1, EvidenceIndex.GRAPH_ALIVE] = 0.0

    outputs = fusion(*logits, evidence)
    loss, _parts = compute_robust_loss(
        outputs["final_logits"],
        labels,
        outputs,
        _paper_loss_config(objective, mask_unavailable=True),
        evidence=evidence,
        epoch=5,
        materialize_diagnostics=False,
    )
    loss.backward()

    graph_grad = logits[1].grad
    assert graph_grad is not None and torch.isfinite(graph_grad).all()
    assert graph_grad[0].abs().sum().item() > 0.0
    assert torch.equal(graph_grad[1], torch.zeros_like(graph_grad[1]))
    for branch_logits in (logits[0], logits[2]):
        assert branch_logits.grad is not None
        assert torch.isfinite(branch_logits.grad).all()
        assert branch_logits.grad[1].abs().sum().item() > 0.0


@pytest.mark.parametrize(
    ("combination", "objective"), [("dempster", "tmc"), ("ecml", "ecml")]
)
def test_conflicting_opinions_have_finite_nonzero_gradients(
    combination: str, objective: str
) -> None:
    fusion = _fusion(combination)
    logits = [
        torch.tensor([[40.0, -40.0]], requires_grad=True),
        torch.tensor([[-40.0, 40.0]], requires_grad=True),
        torch.tensor([[30.0, -30.0]], requires_grad=True),
    ]
    labels = torch.tensor([0])
    evidence = _observable_evidence(1)

    outputs = fusion(*logits, evidence)
    loss, _parts = compute_robust_loss(
        outputs["final_logits"],
        labels,
        outputs,
        _paper_loss_config(objective, mask_unavailable=True),
        evidence=evidence,
        epoch=10,
        materialize_diagnostics=False,
    )

    assert torch.isfinite(outputs["dirichlet_alpha_fused"]).all()
    assert torch.isfinite(loss)
    loss.backward()
    for branch_logits in logits[:3]:
        assert branch_logits.grad is not None
        assert torch.isfinite(branch_logits.grad).all()
        assert branch_logits.grad.abs().sum().item() > 0.0
