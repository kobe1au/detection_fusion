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
    uncertainties=(0.2, 0.2, 0.2),
    alive=(1.0, 1.0, 1.0),
):
    beliefs = {}
    uncertainty_map = {}
    reliability_map = {}
    alive_map = {}
    for index, name in enumerate(BRANCHES):
        uncertainty = float(uncertainties[index])
        p1 = float(malware_probabilities[index])
        beliefs[name] = torch.tensor(
            [[1.0 - p1 - uncertainty / 2.0, p1 - uncertainty / 2.0]],
            dtype=torch.float32,
        )
        uncertainty_map[name] = torch.tensor([uncertainty])
        reliability_map[name] = torch.tensor([float(reliabilities[index])])
        alive_map[name] = torch.tensor([float(alive[index])])
    return beliefs, uncertainty_map, reliability_map, alive_map


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
    assert penalized["observed_outlier_distance"][0, 2] > penalized[
        "observed_outlier_distance"
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
    uncertainties = (0.12, 0.26, 0.44)
    base = router.prepare_route_inputs(
        *_inputs(
            probabilities,
            reliabilities=reliabilities,
            uncertainties=uncertainties,
        )
    )

    permutation = (2, 0, 1)
    permuted = router.prepare_route_inputs(
        *_inputs(
            tuple(probabilities[index] for index in permutation),
            reliabilities=tuple(reliabilities[index] for index in permutation),
            uncertainties=tuple(uncertainties[index] for index in permutation),
        )
    )

    for key in (
        "peer_consensus_probability",
        "peer_consensus_support",
        "peer_consensus_js",
        "observed_outlier_distance",
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
    assert low_bad_peer["observed_outlier_distance"][
        0, target
    ] < trusted_bad_peer["observed_outlier_distance"][0, target]


def test_identical_branch_opinions_have_zero_peer_consensus_js():
    router = GlobalOpinionRouter()
    prepared = router.prepare_route_inputs(
        *_inputs(
            (0.37, 0.37, 0.37),
            reliabilities=(0.95, 0.61, 0.17),
            uncertainties=(0.12, 0.34, 0.48),
        )
    )

    torch.testing.assert_close(
        prepared["peer_consensus_js"],
        torch.zeros(1, 3),
        rtol=0.0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        prepared["observed_outlier_distance"],
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
        outputs["observed_outlier_distance"],
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
        router.route_residual.weight.copy_(
            torch.linspace(-2.0, 2.0, router.route_residual.weight.numel()).view_as(
                router.route_residual.weight
            )
        )
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
        "route_residual_abs_max",
        "risk_feature_weight_max",
        "risk_bias_abs",
    }
    assert all(math.isfinite(value) and value >= 0.0 for value in diagnostics.values())


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
    assert full["risk_structural_conflict"].item() > 0.0
    assert ablated["risk_features"][0, 3].item() == pytest.approx(0.0)
    assert ablated["risk_probability"].item() < full["risk_probability"].item()


def test_missingness_uses_fixed_slots_and_all_missing_is_forced_rejection():
    router = GlobalOpinionRouter()
    one_missing = router(
        *_inputs((0.2, 0.8, 0.5), alive=(1.0, 1.0, 0.0)),
        learned_active=True,
    )
    assert one_missing["branch_distribution"][0, 2].item() == pytest.approx(0.0)
    assert one_missing["branch_distribution"].sum().item() == pytest.approx(1.0)
    assert one_missing["missing_fraction"].item() == pytest.approx(1.0 / 3.0)
    # One observed pair out of the fixed three comparisons.
    assert one_missing["risk_structural_conflict"].item() == pytest.approx(0.6 / 3.0)

    all_missing = router(*_inputs(alive=(0.0, 0.0, 0.0)), learned_active=True)
    assert all_missing["has_available"].item() == pytest.approx(0.0)
    assert torch.allclose(
        all_missing["branch_distribution"], torch.zeros(1, 3), atol=0.0
    )
    assert torch.allclose(
        all_missing["mixture_probability"], torch.full((1, 2), 0.5), atol=1.0e-7
    )
    assert all_missing["risk_probability"].item() == pytest.approx(1.0)
    assert all_missing["committed_mass"].item() == pytest.approx(0.0)


def test_risk_summaries_are_aligned_with_detached_route_distribution():
    router = GlobalOpinionRouter()
    outputs = router(
        *_inputs(
            (0.1, 0.4, 0.9),
            reliabilities=(0.9, 0.6, 0.2),
            uncertainties=(0.1, 0.3, 0.5),
        ),
        learned_active=True,
    )
    pi = outputs["branch_distribution"]
    expected_reliability_deficit = 1.0 - (
        pi * torch.tensor([[0.9, 0.6, 0.2]])
    ).sum(dim=-1)
    expected_uncertainty = (
        pi * torch.tensor([[0.1, 0.3, 0.5]])
    ).sum(dim=-1)
    assert torch.allclose(
        outputs["risk_reliability_deficit"], expected_reliability_deficit
    )
    assert torch.allclose(outputs["risk_uncertainty_burden"], expected_uncertainty)


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
    beliefs, uncertainties, reliabilities, alive = _inputs()
    beliefs = {name: torch.full((1, 3), 0.2) for name in BRANCHES}
    with pytest.raises(ValueError, match="binary"):
        router(beliefs, uncertainties, reliabilities, alive)


def test_route_residual_is_low_capacity_and_has_no_static_branch_bias():
    router = GlobalOpinionRouter()

    assert isinstance(router.route_residual, torch.nn.Linear)
    assert router.route_residual.in_features == 6
    assert router.route_residual.out_features == 2
    assert router.route_residual.bias is None
    assert sum(parameter.numel() for parameter in router.route_parameters()) == 16
    assert sum(parameter.numel() for parameter in router.risk_parameters()) == 6


def test_availability_block_cannot_create_an_all_present_static_branch_bias():
    router = GlobalOpinionRouter(route_conflict_enabled=False)
    with torch.no_grad():
        router.route_residual.weight.zero_()
        router.route_residual.weight[:, 3:] = torch.tensor(
            [[3.0, -2.0, 1.0], [-4.0, 5.0, 2.0]]
        )

    all_present = router(*_inputs(), learned_active=True)
    assert torch.allclose(
        all_present["route_residual"], torch.zeros(1, 3), atol=0.0
    )
    assert torch.allclose(
        all_present["branch_distribution"], torch.full((1, 3), 1.0 / 3.0)
    )

    one_missing = router(
        *_inputs(alive=(1.0, 0.0, 1.0)), learned_active=True
    )
    assert not torch.allclose(
        one_missing["route_residual"], torch.zeros(1, 3)
    )


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
        uncertainties=(0.12, 0.31, 0.48),
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
def test_prepared_route_only_path_preserves_missingness_outputs_and_gradients(
    alive,
):
    router = GlobalOpinionRouter(route_conflict_enabled=True)
    inputs = _inputs(
        (0.1, 0.6, 0.88),
        reliabilities=(0.9, 0.55, 0.2),
        uncertainties=(0.1, 0.3, 0.5),
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
