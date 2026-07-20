from __future__ import annotations

import math

import pytest
import torch

from fusion.train import (
    ROUTING_ROBUSTNESS_FAMILIES,
    ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION,
    _compile_group_robust_row_weights,
    _entropic_soft_worst_group,
)


def _item(name: str, *, strength: float = 0.0) -> dict:
    return {"scenario_name": name, "strength": strength}


def test_group_robust_weights_preserve_family_mechanism_source_hierarchy():
    clean = [_item("clean_a"), _item("clean_b")]
    local_a = [_item("a_s1", strength=0.1), _item("a_s2", strength=0.9)]
    local_b = [_item("b_s1", strength=0.5)]
    joint = [_item("joint_s1", strength=0.5)]
    items = [*clean, *local_a, *local_b, *joint]
    segments = [(index, index + 1) for index in range(len(items))]
    groups = [
        {
            "name": "router:local_completeness",
            "objective_family": "local_completeness",
            "clean": clean,
            "scenario": [*local_a, *local_b],
            "mechanisms": [
                {"name": "mechanism_a", "items": local_a},
                {"name": "mechanism_b", "items": local_b},
            ],
        },
        {
            "name": "router:joint_semantic",
            "objective_family": "joint_semantic",
            "clean": clean,
            "scenario": joint,
            "mechanisms": [{"name": "joint", "items": joint}],
        },
    ]
    (
        clean_weights,
        family_weights,
        family_priors,
        resolved,
    ) = _compile_group_robust_row_weights(
        groups,
        items,
        segments,
        torch.ones(len(items), dtype=torch.bool),
    )

    assert clean_weights.tolist() == pytest.approx([0.5, 0.5, 0, 0, 0, 0])
    # local family: mechanisms are 1/2 each; mechanism_a's two strengths split
    # its half, while mechanism_b's only source retains the other half.
    assert family_weights[0].tolist() == pytest.approx(
        [0, 0, 0.25, 0.25, 0.5, 0]
    )
    assert family_weights[1].tolist() == pytest.approx([0, 0, 0, 0, 0, 1.0])
    assert resolved[0]["num_mechanisms"] == 2
    assert resolved[0]["num_sources"] == 3
    assert family_priors.tolist() == pytest.approx([0.5, 0.5])


def test_strength_cells_preserve_equal_parent_perturbation_mass():
    clean = [_item("clean")]
    type_a = [_item(f"a_s{s}", strength=s) for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
    type_b = [_item("b_missing", strength=1.0)]
    items = [*clean, *type_a, *type_b]
    segments = [(index, index + 1) for index in range(len(items))]
    groups = [
        {
            "name": f"router:type_a/s{item['strength']:g}",
            "objective_family": f"type_a/s{item['strength']:g}",
            "prior_group": "type_a",
            "clean": clean,
            "scenario": [item],
            "mechanisms": [{"name": item["scenario_name"], "items": [item]}],
        }
        for item in type_a
    ] + [
        {
            "name": "router:type_b/missing",
            "objective_family": "type_b/missing",
            "prior_group": "type_b",
            "clean": clean,
            "scenario": type_b,
            "mechanisms": [{"name": "type_b/missing", "items": type_b}],
        }
    ]
    _clean, _rows, priors, resolved = _compile_group_robust_row_weights(
        groups,
        items,
        segments,
        torch.ones(len(items), dtype=torch.bool),
    )
    assert priors[:5].sum().item() == pytest.approx(0.5)
    assert priors[5].item() == pytest.approx(0.5)
    assert [entry["prior_group"] for entry in resolved[:5]] == ["type_a"] * 5


def test_entropic_soft_worst_rho_zero_is_exact_mean_and_rho_one_favors_max():
    base = torch.tensor([0.2, 0.4, 0.9], requires_grad=True)
    mean_value, mean_weights = _entropic_soft_worst_group(
        base,
        soft_worst_weight=0.0,
        temperature=0.1,
    )
    torch.testing.assert_close(mean_value, base.mean())
    torch.testing.assert_close(mean_weights, torch.full((3,), 1.0 / 3.0))

    worst_value, effective_weights = _entropic_soft_worst_group(
        base,
        soft_worst_weight=1.0,
        temperature=0.1,
    )
    assert base.mean().item() < worst_value.item() < base.max().item()
    assert effective_weights.argmax().item() == 2
    assert effective_weights[2] > 0.98
    worst_value.backward()
    assert base.grad is not None
    assert base.grad.argmax().item() == 2
    assert torch.isfinite(base.grad).all()


def test_entropic_soft_worst_respects_hierarchical_group_priors():
    losses = torch.tensor([0.1, 0.2, 0.3])
    priors = torch.tensor([0.25, 0.25, 0.5])
    reduced, effective = _entropic_soft_worst_group(
        losses,
        soft_worst_weight=0.0,
        temperature=0.1,
        group_priors=priors,
    )
    torch.testing.assert_close(reduced, torch.dot(losses, priors))
    torch.testing.assert_close(effective, priors)


def test_robustness_taxonomy_covers_every_calibration_perturbation_once():
    assert set(ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION.values()) == set(
        ROUTING_ROBUSTNESS_FAMILIES
    )
    assert len(ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION) == len(
        set(ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION)
    )
    assert (
        ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION["all_semantic_corrupted"]
        == "joint_semantic"
    )
    for perturb_type in (
        "api_graph_degraded",
        "api_manifest_degraded",
        "graph_manifest_degraded",
        "all_degraded",
    ):
        assert (
            ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION[perturb_type]
            == "combined_completeness"
        )


@pytest.mark.parametrize(
    "rho,tau",
    [(-0.1, 0.1), (1.1, 0.1), (0.5, 0.0), (0.5, math.inf)],
)
def test_entropic_soft_worst_rejects_invalid_configuration(rho: float, tau: float):
    with pytest.raises(ValueError):
        _entropic_soft_worst_group(
            torch.tensor([0.2, 0.3]),
            soft_worst_weight=rho,
            temperature=tau,
        )
