from __future__ import annotations

import math

import pytest
import torch

from fusion.opinion_router import GlobalOpinionRouter


BRANCHES = ("api", "graph", "manifest")


def _inputs(
    malware_probabilities=(0.2, 0.2, 0.2),
    *,
    reliabilities=(0.8, 0.8, 0.8),
    alive=(1.0, 1.0, 1.0),
):
    probabilities = {}
    reliability_map = {}
    alive_map = {}
    for index, name in enumerate(BRANCHES):
        p1 = float(malware_probabilities[index])
        probabilities[name] = torch.tensor([[1.0 - p1, p1]], dtype=torch.float32)
        reliability_map[name] = torch.tensor([float(reliabilities[index])])
        alive_map[name] = torch.tensor([float(alive[index])])
    return probabilities, reliability_map, alive_map


def test_route_conflict_is_an_explicit_non_positive_score_term():
    with_conflict = GlobalOpinionRouter(
        route_conflict_enabled=True,
        risk_conflict_enabled=False,
    )
    without_conflict = GlobalOpinionRouter(
        route_conflict_enabled=False,
        risk_conflict_enabled=False,
    )
    without_conflict.load_state_dict(with_conflict.state_dict())

    inputs = _inputs((0.2, 0.2, 0.8))
    penalized = with_conflict(*inputs, learned_active=True)
    unpenalized = without_conflict(*inputs, learned_active=True)

    assert torch.all(penalized["conflict_penalty"] >= 0.0)
    assert penalized["reliability_weighted_cross_modal_conflict"][
        0, 2
    ] > penalized[
        "reliability_weighted_cross_modal_conflict"
    ][0, 0]
    assert torch.allclose(
        penalized["routing_scores"],
        unpenalized["routing_scores"] - penalized["conflict_penalty"],
        atol=1.0e-6,
    )
    assert penalized["branch_distribution"][0, 2] < unpenalized[
        "branch_distribution"
    ][0, 2]


def test_peer_consensus_is_equivariant_to_modality_permutation():
    router = GlobalOpinionRouter()
    probabilities = (0.13, 0.47, 0.84)
    reliabilities = (0.91, 0.58, 0.24)
    base = router.prepare_route_inputs(
        *_inputs(
            probabilities,
            reliabilities=reliabilities,
        )
    )

    permutation = (2, 0, 1)
    permuted = router.prepare_route_inputs(
        *_inputs(
            tuple(probabilities[index] for index in permutation),
            reliabilities=tuple(reliabilities[index] for index in permutation),
        )
    )

    for key in (
        "peer_consensus_probability",
        "peer_consensus_support",
        "peer_consensus_js",
        "reliability_weighted_cross_modal_conflict",
    ):
        torch.testing.assert_close(
            permuted[key],
            base[key][:, permutation, ...],
            rtol=1.0e-6,
            atol=1.0e-7,
        )


def test_low_reliability_bad_peer_has_less_consensus_influence():
    router = GlobalOpinionRouter()
    probabilities = (0.2, 0.2, 0.9)
    low_bad_peer = router.prepare_route_inputs(
        *_inputs(probabilities, reliabilities=(0.8, 0.9, 0.05))
    )
    trusted_bad_peer = router.prepare_route_inputs(
        *_inputs(probabilities, reliabilities=(0.8, 0.9, 0.9))
    )

    target = 0
    target_probability = probabilities[target]
    low_consensus = low_bad_peer["peer_consensus_probability"][0, target, 1]
    trusted_consensus = trusted_bad_peer["peer_consensus_probability"][
        0, target, 1
    ]
    assert abs(float(low_consensus) - target_probability) < abs(
        float(trusted_consensus) - target_probability
    )
    assert low_bad_peer["peer_consensus_js"][0, target] < trusted_bad_peer[
        "peer_consensus_js"
    ][0, target]
    assert low_bad_peer["reliability_weighted_cross_modal_conflict"][
        0, target
    ] < trusted_bad_peer["reliability_weighted_cross_modal_conflict"][0, target]


def test_identical_branch_opinions_have_zero_peer_consensus_js():
    router = GlobalOpinionRouter()
    prepared = router.prepare_route_inputs(
        *_inputs(
            (0.37, 0.37, 0.37),
            reliabilities=(0.95, 0.61, 0.17),
        )
    )

    torch.testing.assert_close(
        prepared["peer_consensus_js"],
        torch.zeros(1, 3),
        rtol=0.0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        prepared["reliability_weighted_cross_modal_conflict"],
        torch.zeros(1, 3),
        rtol=0.0,
        atol=1.0e-7,
    )


@pytest.mark.parametrize(
    "alive",
    [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_dead_or_no_peer_placeholders_do_not_create_route_conflict(alive):
    router = GlobalOpinionRouter(route_conflict_enabled=True)
    outputs = router(
        *_inputs(
            (0.12, 0.88, 0.51),
            reliabilities=(0.9, 0.7, 0.4),
            alive=alive,
        ),
        learned_active=True,
    )

    torch.testing.assert_close(
        outputs["peer_consensus_js"],
        torch.zeros(1, 3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        outputs["reliability_weighted_cross_modal_conflict"],
        torch.zeros(1, 3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        outputs["conflict_penalty"],
        torch.zeros(1, 3),
        rtol=0.0,
        atol=0.0,
    )


def test_effective_regularizers_and_scale_diagnostics_are_finite():
    router = GlobalOpinionRouter()
    with torch.no_grad():
        router.raw_route_prior_beta.fill_(2.5)
        router.raw_conflict_scale.copy_(torch.tensor([-3.0, 0.0, 1.5]))
        router.raw_risk_feature_weights.copy_(
            torch.linspace(-1.5, 1.5, router.raw_risk_feature_weights.numel())
        )
        router.risk_bias.fill_(-0.75)

    route_l2 = router.route_effective_l2()
    risk_l2 = router.risk_effective_l2()
    assert route_l2.ndim == 0 and torch.isfinite(route_l2) and route_l2 >= 0.0
    assert risk_l2.ndim == 0 and torch.isfinite(risk_l2) and risk_l2 >= 0.0

    diagnostics = router.effective_parameter_diagnostics()
    assert diagnostics.keys() == {
        "route_prior_beta",
        "route_conflict_scale_max",
        "risk_feature_weight_max",
        "risk_bias_abs",
    }
    assert all(math.isfinite(value) and value >= 0.0 for value in diagnostics.values())

    details = router.effective_parameter_details()
    assert details["route_conflict_active"] is True
    assert details["risk_conflict_active"] is True
    assert details["route_score_semantics"] == (
        "beta_logit_reliability_minus_nonnegative_consensus_conflict"
    )
    assert set(details["route_conflict_scale"]) == set(BRANCHES)
    assert set(details["risk_feature_weights"]) == {
        "reliability_deficit",
        "decision_boundary_proximity",
        "global_cross_modal_conflict",
    }
    assert details["risk_bias"] == pytest.approx(-0.75)


def test_effective_parameter_details_report_operative_ablation_semantics():
    router = GlobalOpinionRouter(
        mode="prior_only",
        fixed_prior_beta=0.5,
        route_conflict_enabled=False,
        risk_mode="reliability_prior",
        risk_conflict_enabled=False,
    )
    with torch.no_grad():
        router.raw_route_prior_beta.fill_(8.0)
        router.raw_conflict_scale.fill_(8.0)
        router.raw_risk_feature_weights.fill_(8.0)
        router.risk_bias.fill_(8.0)

    details = router.effective_parameter_details()
    assert details["route_mode"] == "prior_only"
    assert details["route_conflict_enabled"] is False
    assert details["route_conflict_active"] is False
    assert details["route_prior_beta"] == pytest.approx(0.5)
    assert all(
        value == pytest.approx(0.0)
        for value in details["route_conflict_scale"].values()
    )
    assert details["route_score_semantics"] == "beta_logit_reliability"
    assert details["risk_mode"] == "reliability_prior"
    assert details["risk_conflict_enabled"] is False
    assert details["risk_conflict_active"] is False
    assert details["risk_head_semantics"] == "reliability_deficit_probability"
    assert details["risk_feature_weights"] == {}
    assert details["risk_bias"] is None
    diagnostics = router.effective_parameter_diagnostics()
    assert diagnostics["route_prior_beta"] == pytest.approx(0.5)
    assert diagnostics["route_conflict_scale_max"] == pytest.approx(0.0)
    assert diagnostics["risk_feature_weight_max"] == pytest.approx(0.0)
    assert diagnostics["risk_bias_abs"] == pytest.approx(0.0)


def test_route_and_risk_conflict_switches_are_independent():
    base = GlobalOpinionRouter(
        route_conflict_enabled=True,
        risk_conflict_enabled=True,
    )
    no_risk_conflict = GlobalOpinionRouter(
        route_conflict_enabled=True,
        risk_conflict_enabled=False,
    )
    no_risk_conflict.load_state_dict(base.state_dict())
    inputs = _inputs((0.1, 0.1, 0.9))

    full = base(*inputs, learned_active=True)
    ablated = no_risk_conflict(*inputs, learned_active=True)
    assert torch.allclose(
        full["branch_distribution"], ablated["branch_distribution"], atol=1.0e-7
    )
    assert full["risk_global_cross_modal_conflict"].item() > 0.0
    assert ablated["risk_features"][0, 2].item() == pytest.approx(0.0)
    assert ablated["risk_global_cross_modal_conflict"].item() == pytest.approx(0.0)
    assert ablated["risk_probability"].item() < full["risk_probability"].item()


def test_alive_masking_and_all_dead_contract_are_explicit():
    router = GlobalOpinionRouter()
    one_dead = router(
        *_inputs((0.2, 0.8, 0.5), alive=(1.0, 1.0, 0.0)),
        learned_active=True,
    )
    assert one_dead["branch_distribution"][0, 2].item() == pytest.approx(0.0)
    assert one_dead["branch_distribution"].sum().item() == pytest.approx(1.0)
    assert one_dead["reliability_weighted_cross_modal_conflict"][
        0, 2
    ].item() == pytest.approx(0.0)
    assert one_dead["global_cross_modal_conflict"].item() >= 0.0
    assert "missing_fraction" not in one_dead
    assert "risk_missing_fraction" not in one_dead

    all_dead = router(*_inputs(alive=(0.0, 0.0, 0.0)), learned_active=True)
    assert all_dead["has_available"].item() == pytest.approx(0.0)
    assert torch.allclose(
        all_dead["branch_distribution"], torch.zeros(1, 3), atol=0.0
    )
    assert torch.allclose(
        all_dead["mixture_probability"], torch.full((1, 2), 0.5), atol=1.0e-7
    )
    assert all_dead["risk_probability"].item() == pytest.approx(1.0)
    assert all_dead["committed_mass"].item() == pytest.approx(0.0)


def test_three_risk_features_are_aligned_with_detached_route_distribution():
    router = GlobalOpinionRouter()
    outputs = router(
        *_inputs(
            (0.1, 0.4, 0.9),
            reliabilities=(0.9, 0.6, 0.2),
        ),
        learned_active=True,
    )
    pi = outputs["branch_distribution"]
    expected_reliability_deficit = 1.0 - (
        pi * torch.tensor([[0.9, 0.6, 0.2]])
    ).sum(dim=-1)
    assert torch.allclose(
        outputs["risk_reliability_deficit"], expected_reliability_deficit
    )
    assert outputs["risk_features"].shape == (1, 3)
    torch.testing.assert_close(
        outputs["risk_features"][:, 0],
        outputs["risk_reliability_deficit"],
    )
    torch.testing.assert_close(
        outputs["risk_features"][:, 1],
        outputs["risk_decision_boundary_proximity"],
    )
    torch.testing.assert_close(
        outputs["risk_features"][:, 2],
        outputs["risk_global_cross_modal_conflict"],
    )
    torch.testing.assert_close(
        outputs["global_cross_modal_conflict"],
        outputs["reliability_weighted_cross_modal_conflict"].mean(dim=-1),
    )


@pytest.mark.parametrize(
    ("alive", "expected"),
    [
        ((1.0, 1.0, 1.0), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        ((1.0, 0.0, 1.0), (0.5, 0.0, 0.5)),
    ],
)
@pytest.mark.parametrize("mode", ["learned", "prior_only"])
def test_prefit_path_uses_alive_only_uniform_prior_with_zero_learned_risk(
    alive,
    expected,
    mode,
):
    router = GlobalOpinionRouter(mode=mode)
    outputs = router(
        *_inputs(reliabilities=(0.95, 0.35, 0.05), alive=alive),
        learned_active=False,
    )
    expected_distribution = torch.tensor([expected], dtype=torch.float32)
    assert torch.allclose(
        outputs["prior_branch_distribution"], expected_distribution
    )
    assert torch.allclose(outputs["branch_distribution"], expected_distribution)
    assert outputs["route_prior_beta"].item() == pytest.approx(1.0)
    assert outputs["risk_probability"].item() == pytest.approx(0.0)
    assert outputs["learned_components_active"].item() == pytest.approx(0.0)
    assert outputs["prefit_uniform_prior_active"].item() == pytest.approx(1.0)


def test_route_mode_and_risk_mode_are_independent():
    router = GlobalOpinionRouter(
        mode="prior_only",
        risk_mode="learned",
    )
    assert router.route_parameters() == []
    assert len(router.risk_parameters()) == 2
    reliability = torch.tensor([0.9, 0.5, 0.1])
    outputs = router(
        *_inputs(reliabilities=tuple(reliability.tolist())), learned_active=True
    )
    expected = torch.softmax(torch.logit(reliability), dim=-1).unsqueeze(0)
    assert torch.allclose(outputs["prior_branch_distribution"], expected)
    assert torch.allclose(
        outputs["branch_distribution"], outputs["prior_branch_distribution"]
    )
    assert not torch.allclose(
        outputs["branch_distribution"], torch.full((1, 3), 1.0 / 3.0)
    )
    assert outputs["prefit_uniform_prior_active"].item() == pytest.approx(0.0)
    assert outputs["risk_mode_learned"].item() == pytest.approx(1.0)


def test_binary_router_rejects_multiclass_inputs():
    router = GlobalOpinionRouter()
    probabilities, reliabilities, alive = _inputs()
    probabilities = {name: torch.full((1, 3), 0.2) for name in BRANCHES}
    with pytest.raises(ValueError, match="binary"):
        router(probabilities, reliabilities, alive)


def test_router_rejects_raw_evidential_belief_mass_as_probability_input():
    router = GlobalOpinionRouter()
    probabilities, reliabilities, alive = _inputs()
    probabilities["graph"] = torch.tensor([[0.3, 0.4]])

    with pytest.raises(ValueError, match="must sum to one.*raw evidential belief"):
        router(probabilities, reliabilities, alive)


def test_one_hot_probabilities_remain_finite_in_float16_with_small_eps():
    router = GlobalOpinionRouter()
    probabilities = {
        name: torch.tensor([[1.0, 0.0]], dtype=torch.float16)
        for name in BRANCHES
    }
    reliabilities = {
        name: torch.tensor([1.0], dtype=torch.float16)
        for name in BRANCHES
    }
    alive = {
        name: torch.tensor([1.0], dtype=torch.float16)
        for name in BRANCHES
    }

    outputs = router(
        probabilities,
        reliabilities,
        alive,
        eps=1.0e-8,
    )
    for key in (
        "routing_scores",
        "branch_distribution",
        "mixture_probability",
        "peer_consensus_js",
        "risk_probability",
        "risk_training_logit",
    ):
        assert torch.isfinite(outputs[key]).all(), key


def test_learned_route_parameter_set_contains_only_beta_and_conflict_scales():
    router = GlobalOpinionRouter()

    assert not hasattr(router, "route_residual")
    route_parameters = router.route_parameters()
    assert len(route_parameters) == 2
    assert route_parameters[0] is router.raw_route_prior_beta
    assert route_parameters[1] is router.raw_conflict_scale
    assert sum(parameter.numel() for parameter in router.route_parameters()) == 4
    assert sum(parameter.numel() for parameter in router.risk_parameters()) == 4

    no_conflict = GlobalOpinionRouter(route_conflict_enabled=False)
    no_conflict_parameters = no_conflict.route_parameters()
    assert len(no_conflict_parameters) == 1
    assert no_conflict_parameters[0] is no_conflict.raw_route_prior_beta
    expected_l2 = torch.nn.functional.softplus(
        no_conflict.raw_route_prior_beta
    ).square() / 4.0
    torch.testing.assert_close(no_conflict.route_effective_l2(), expected_l2)


def test_no_i1_route_is_conflict_only_and_beta_is_not_optimized():
    router = GlobalOpinionRouter(reliability_input_enabled=False)
    route_parameters = router.route_parameters()
    assert len(route_parameters) == 1
    assert route_parameters[0] is router.raw_conflict_scale

    first = router(
        *_inputs(
            (0.1, 0.5, 0.9),
            reliabilities=(0.99, 0.55, 0.10),
        ),
        learned_active=True,
        compute_risk=False,
    )
    second = router(
        *_inputs(
            (0.1, 0.5, 0.9),
            reliabilities=(0.10, 0.55, 0.99),
        ),
        learned_active=True,
        compute_risk=False,
    )
    torch.testing.assert_close(
        first["branch_distribution"], second["branch_distribution"]
    )
    assert float(first["route_prior_beta"]) == pytest.approx(0.0)
    assert router.effective_parameter_details()["route_score_semantics"] == (
        "negative_nonnegative_consensus_conflict"
    )

    loss = -first["mixture_probability"][0, 1].log()
    loss = loss + 0.1 * router.route_effective_l2()
    loss.backward()
    assert router.raw_route_prior_beta.grad is None
    assert router.raw_conflict_scale.grad is not None


def test_fully_ablated_route_is_static_alive_uniform_with_no_parameters():
    router = GlobalOpinionRouter(
        reliability_input_enabled=False,
        route_conflict_enabled=False,
        risk_mode="disabled",
    )
    assert router.route_parameters() == []
    outputs = router(
        *_inputs(
            (0.1, 0.5, 0.9),
            reliabilities=(0.99, 0.55, 0.10),
            alive=(1.0, 0.0, 1.0),
        ),
        learned_active=True,
    )

    torch.testing.assert_close(
        outputs["branch_distribution"],
        torch.tensor([[0.5, 0.0, 0.5]]),
    )
    assert outputs["learned_components_active"].item() == pytest.approx(0.0)
    assert outputs["route_prior_beta"].item() == pytest.approx(0.0)
    assert router.route_effective_l2().item() == pytest.approx(0.0)


def test_disabled_risk_features_have_zero_gradient_and_no_regularization():
    router = GlobalOpinionRouter(
        reliability_input_enabled=False,
        risk_conflict_enabled=False,
    )
    outputs = router(*_inputs((0.2, 0.5, 0.8)), learned_active=True)
    loss = outputs["risk_probability"].sum() + router.risk_effective_l2()
    loss.backward()

    gradient = router.raw_risk_feature_weights.grad
    assert gradient is not None
    assert float(gradient[0]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(gradient[2]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(gradient[1]) != pytest.approx(0.0, abs=1.0e-12)
    weights = outputs["risk_feature_weights"]
    assert float(weights[0].detach()) == pytest.approx(0.0)
    assert float(weights[2].detach()) == pytest.approx(0.0)

    boundary_weight = torch.nn.functional.softplus(
        router.raw_risk_feature_weights[1]
    )
    expected_l2 = (
        boundary_weight.square() + router.risk_bias.square()
    ) / 4.0
    torch.testing.assert_close(router.risk_effective_l2(), expected_l2)


def test_legacy_free_residual_checkpoint_is_rejected_strictly():
    router = GlobalOpinionRouter()
    legacy_state = dict(router.state_dict())
    legacy_state["route_residual.weight"] = torch.zeros(2, 6)

    with pytest.raises(RuntimeError, match="Unexpected key.*route_residual.weight"):
        router.load_state_dict(legacy_state, strict=True)


def test_legacy_five_feature_risk_checkpoint_is_rejected():
    router = GlobalOpinionRouter()
    legacy_state = dict(router.state_dict())
    legacy_state["raw_risk_feature_weights"] = torch.zeros(5)

    with pytest.raises(RuntimeError, match="size mismatch.*raw_risk_feature_weights"):
        router.load_state_dict(legacy_state, strict=True)


def test_route_weight_is_monotone_in_own_reliability_without_conflict():
    router = GlobalOpinionRouter(route_conflict_enabled=False)
    lower = router(
        *_inputs(reliabilities=(0.60, 0.70, 0.80)), learned_active=True
    )
    higher = router(
        *_inputs(reliabilities=(0.75, 0.70, 0.80)), learned_active=True
    )

    assert higher["routing_scores"][0, 0] > lower["routing_scores"][0, 0]
    assert higher["branch_distribution"][0, 0] > lower["branch_distribution"][0, 0]


def test_router_public_api_excludes_raw_edl_uncertainty_and_old_risk_features():
    router = GlobalOpinionRouter(route_conflict_enabled=True)
    outputs = router(
        *_inputs(
            (0.2, 0.5, 0.8),
            reliabilities=(0.85, 0.65, 0.45),
        ),
        learned_active=True,
    )

    assert outputs["risk_features"].shape[-1] == 3
    for removed_key in (
        "risk_uncertainty_burden",
        "risk_structural_conflict",
        "risk_missing_fraction",
        "missing_fraction",
        "observed_outlier_distance",
        "routing_outlier_distance",
    ):
        assert removed_key not in outputs


def test_oof_branch_distribution_override_controls_mixture_and_respects_alive_mask():
    router = GlobalOpinionRouter()
    override = torch.tensor([[0.2, 0.3, 0.5]])
    outputs = router(
        *_inputs(reliabilities=(0.9, 0.5, 0.2)),
        learned_active=True,
        branch_distribution_override=override,
    )

    assert torch.allclose(outputs["branch_distribution"], override)
    assert outputs["branch_distribution_override_active"].item() == 1.0

    missing = router(
        *_inputs(reliabilities=(0.9, 0.5, 0.2), alive=(1.0, 0.0, 1.0)),
        learned_active=True,
        branch_distribution_override=torch.tensor([[0.2, 0.7, 0.1]]),
    )
    assert missing["branch_distribution"][0, 1].item() == 0.0
    assert torch.allclose(
        missing["branch_distribution"][0, [0, 2]],
        torch.tensor([2.0 / 3.0, 1.0 / 3.0]),
    )


def _parameter_gradients(
    router: GlobalOpinionRouter,
    outputs: dict[str, torch.Tensor],
    *,
    include_risk: bool,
) -> list[torch.Tensor]:
    objective = (
        outputs["mixture_probability"].square().mean()
        + outputs["branch_distribution"].square().mean()
    )
    parameters = list(router.route_parameters())
    if include_risk:
        objective = objective + outputs["risk_training_logit"].mean()
        parameters.extend(router.risk_parameters())
    objective.backward()
    return [
        torch.zeros_like(parameter)
        if parameter.grad is None
        else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def test_prepared_full_path_preserves_every_output_and_parameter_gradient():
    router = GlobalOpinionRouter(
        route_conflict_enabled=True,
        risk_conflict_enabled=True,
        risk_mode="learned",
    )
    inputs = _inputs(
        (0.08, 0.57, 0.91),
        reliabilities=(0.93, 0.62, 0.24),
    )

    router.zero_grad(set_to_none=True)
    wrapper_outputs = router(*inputs, learned_active=True)
    wrapper_gradients = _parameter_gradients(
        router, wrapper_outputs, include_risk=True
    )

    router.zero_grad(set_to_none=True)
    prepared = router.prepare_route_inputs(*inputs)
    prepared_outputs = router.forward_prepared(
        prepared, learned_active=True
    )
    prepared_gradients = _parameter_gradients(
        router, prepared_outputs, include_risk=True
    )

    assert prepared_outputs.keys() == wrapper_outputs.keys()
    for key in wrapper_outputs:
        torch.testing.assert_close(
            prepared_outputs[key],
            wrapper_outputs[key],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            msg=lambda message, key=key: f"output {key!r} differs: {message}",
        )
    assert len(prepared_gradients) == len(wrapper_gradients)
    for prepared_gradient, wrapper_gradient in zip(
        prepared_gradients, wrapper_gradients
    ):
        torch.testing.assert_close(
            prepared_gradient,
            wrapper_gradient,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "alive",
    [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_prepared_route_only_path_preserves_alive_mask_outputs_and_gradients(
    alive,
):
    router = GlobalOpinionRouter(route_conflict_enabled=True)
    inputs = _inputs(
        (0.1, 0.6, 0.88),
        reliabilities=(0.9, 0.55, 0.2),
        alive=alive,
    )

    router.zero_grad(set_to_none=True)
    wrapper_outputs = router(
        *inputs, learned_active=True, compute_risk=False
    )
    wrapper_gradients = _parameter_gradients(
        router, wrapper_outputs, include_risk=False
    )

    router.zero_grad(set_to_none=True)
    with torch.no_grad():
        prepared = router.prepare_route_inputs(*inputs)
    prepared_outputs = router.forward_prepared(
        prepared, learned_active=True, compute_risk=False
    )
    prepared_gradients = _parameter_gradients(
        router, prepared_outputs, include_risk=False
    )

    assert prepared_outputs.keys() == wrapper_outputs.keys()
    for key in wrapper_outputs:
        torch.testing.assert_close(
            prepared_outputs[key],
            wrapper_outputs[key],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    for prepared_gradient, wrapper_gradient in zip(
        prepared_gradients, wrapper_gradients
    ):
        torch.testing.assert_close(
            prepared_gradient,
            wrapper_gradient,
            rtol=0.0,
            atol=0.0,
        )
