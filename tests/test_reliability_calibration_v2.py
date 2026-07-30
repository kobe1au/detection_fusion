from __future__ import annotations

import math

import pytest
import torch

import fusion.reliability_calibration as reliability_module
from fusion.reliability_calibration import (
    API_SUPPORT_RELIABILITY_FEATURE_LAYOUT,
    API_SUPPORT_RELIABILITY_FEATURE_NAMES,
    BRANCH_NAMES,
    MONOTONIC_CORRECTNESS_METHOD,
    RELIABILITY_FEATURE_LAYOUT,
    RELIABILITY_FEATURE_NAMES,
    TEMPERATURE_SCALING_CONFIDENCE_METHOD,
    BranchTemperatureScalingConfidenceCalibrator,
    MonotonicBranchCorrectnessCalibrator,
    MonotonicReliabilityCalibrator,
    build_reliability_features,
    normalize_reliability_calibration_method,
    reliability_feature_layout,
)


def _alphas(
    alpha: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    value = (
        torch.tensor([[5.0, 1.0], [1.0, 5.0]])
        if alpha is None
        else alpha
    )
    return {branch: value.clone() for branch in BRANCH_NAMES}


def _features(
    alpha: torch.Tensor | None = None,
    *,
    api_observed_support: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return build_reliability_features(
        _alphas(alpha),
        api_observed_support=api_observed_support,
    )


def _alive(
    batch_size: int = 2,
    *,
    dead_branch: str | None = None,
) -> dict[str, torch.Tensor]:
    result = {
        branch: torch.ones(batch_size, dtype=torch.float32)
        for branch in BRANCH_NAMES
    }
    if dead_branch is not None:
        result[dead_branch] = torch.zeros(batch_size, dtype=torch.float32)
    return result


def _raw_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def test_formal_i1_feature_layout_is_exact_and_branch_local():
    expected = (
        "evidential_certainty",
        "prediction_margin",
        "predicted_malware_indicator",
    )
    assert RELIABILITY_FEATURE_NAMES == expected
    assert RELIABILITY_FEATURE_LAYOUT == {
        branch: expected for branch in BRANCH_NAMES
    }


def test_api_support_layout_changes_only_the_api_branch():
    expected_api = (
        "evidential_certainty",
        "prediction_margin",
        "observed_support",
        "predicted_malware_indicator",
    )
    assert API_SUPPORT_RELIABILITY_FEATURE_NAMES == expected_api
    assert API_SUPPORT_RELIABILITY_FEATURE_LAYOUT == {
        "api": expected_api,
        "graph": RELIABILITY_FEATURE_NAMES,
        "manifest": RELIABILITY_FEATURE_NAMES,
    }
    assert reliability_feature_layout(
        use_api_observed_support=True
    ) == API_SUPPORT_RELIABILITY_FEATURE_LAYOUT
    assert reliability_feature_layout(
        use_api_observed_support=False
    ) == RELIABILITY_FEATURE_LAYOUT


def test_feature_builder_uses_exact_binary_dirichlet_definitions():
    alpha = torch.tensor(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [1.0, 5.0],
        ]
    )
    features = build_reliability_features(_alphas(alpha))
    for branch in BRANCH_NAMES:
        assert features[branch].shape == (3, 3)
        assert features[branch][:, 0].tolist() == pytest.approx(
            [0.0, 2.0 / 3.0, 2.0 / 3.0]
        )
        assert features[branch][:, 1].tolist() == pytest.approx(
            [0.0, 2.0 / 3.0, 2.0 / 3.0]
        )
        assert features[branch][:, 2].tolist() == [0.0, 0.0, 1.0]


def test_feature_builder_appends_only_explicit_api_observed_support():
    support = torch.tensor([0.0, 0.4, 1.0])
    alpha = torch.tensor(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [1.0, 5.0],
        ]
    )
    features = build_reliability_features(
        _alphas(alpha),
        api_observed_support=support,
    )

    assert features["api"].shape == (3, 4)
    assert features["graph"].shape == (3, 3)
    assert features["manifest"].shape == (3, 3)
    assert features["api"][:, 2].tolist() == pytest.approx(
        support.tolist()
    )
    assert features["api"][:, 3].tolist() == [0.0, 0.0, 1.0]
    assert torch.equal(
        features["graph"],
        build_reliability_features(_alphas(alpha))["graph"],
    )
    assert torch.equal(
        features["manifest"],
        build_reliability_features(_alphas(alpha))["manifest"],
    )


@pytest.mark.parametrize(
    ("support", "message"),
    [
        (torch.tensor([[0.5], [0.5]]), "shape"),
        (torch.tensor([0.5]), "shape"),
        (torch.tensor([0, 1]), "floating point"),
        (torch.tensor([-0.1, 0.5]), r"\[0, 1\]"),
        (torch.tensor([0.5, 1.1]), r"\[0, 1\]"),
        (torch.tensor([0.5, float("nan")]), "non-finite"),
    ],
)
def test_feature_builder_strictly_validates_api_observed_support(
    support: torch.Tensor,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        build_reliability_features(
            _alphas(),
            api_observed_support=support,
        )


def test_feature_builder_has_no_cross_branch_dependency():
    base = _alphas()
    changed = _alphas()
    changed["graph"] = torch.tensor([[50.0, 1.0], [1.0, 50.0]])
    base_features = build_reliability_features(base)
    changed_features = build_reliability_features(changed)

    assert torch.equal(base_features["api"], changed_features["api"])
    assert torch.equal(base_features["manifest"], changed_features["manifest"])
    assert not torch.equal(base_features["graph"], changed_features["graph"])


@pytest.mark.parametrize(
    ("bad_alpha", "message"),
    [
        (torch.ones(2, 3), "binary shape"),
        (torch.tensor([[0.9, 1.1]]), "concentration >= 1"),
        (torch.tensor([[1.0, float("nan")]]), "non-finite"),
    ],
)
def test_feature_builder_rejects_non_evidential_or_nonbinary_alpha(
    bad_alpha: torch.Tensor,
    message: str,
):
    alpha = _alphas(torch.ones(bad_alpha.size(0), 2))
    alpha["api"] = bad_alpha
    with pytest.raises(ValueError, match=message):
        build_reliability_features(alpha)


def test_feature_builder_requires_exact_three_branch_mapping():
    alpha = _alphas()
    alpha.pop("manifest")
    with pytest.raises(ValueError, match="exactly"):
        build_reliability_features(alpha)

    alpha = _alphas()
    alpha["other"] = alpha["api"]
    with pytest.raises(ValueError, match="unknown"):
        build_reliability_features(alpha)


def test_each_branch_owns_disjoint_correctness_parameters():
    calibrator = MonotonicReliabilityCalibrator()
    branch_parameter_ids = {
        branch: {id(parameter) for parameter in calibrator.branch_parameters(branch)}
        for branch in BRANCH_NAMES
    }
    assert all(branch_parameter_ids.values())
    for left_index, left in enumerate(BRANCH_NAMES):
        for right in BRANCH_NAMES[left_index + 1 :]:
            assert branch_parameter_ids[left].isdisjoint(
                branch_parameter_ids[right]
            )
    assert all(
        isinstance(
            calibrator.branches[branch],
            MonotonicBranchCorrectnessCalibrator,
        )
        for branch in BRANCH_NAMES
    )


def test_continuous_coefficients_are_positive_and_reliability_is_monotone():
    branch = MonotonicBranchCorrectnessCalibrator(
        use_predicted_class_intercept=False
    )
    weights = branch.effective_continuous_weights()
    assert set(weights) == {"evidential_certainty", "prediction_margin"}
    assert all(float(weight.detach()) > 0.0 for weight in weights.values())

    low = torch.tensor([[0.1, 0.2, 0.0]])
    higher_certainty = torch.tensor([[0.9, 0.2, 0.0]])
    higher_margin = torch.tensor([[0.1, 0.8, 0.0]])
    assert branch(higher_certainty).item() > branch(low).item()
    assert branch(higher_margin).item() > branch(low).item()


def test_continuous_feature_gradients_are_nonnegative_by_construction():
    branch = MonotonicBranchCorrectnessCalibrator()
    features = torch.tensor(
        [[0.2, 0.3, 0.0], [0.6, 0.8, 1.0]],
        requires_grad=True,
    )
    branch.forward_logit(features).sum().backward()
    assert features.grad is not None
    assert torch.all(features.grad[:, :2] > 0.0)


def test_api_observed_support_weight_is_positive_and_monotone():
    branch = MonotonicBranchCorrectnessCalibrator(
        use_evidential_certainty=False,
        use_prediction_margin=False,
        use_observed_support=True,
        use_predicted_class_intercept=False,
    )
    weights = branch.effective_continuous_weights()
    assert set(weights) == {
        "evidential_certainty",
        "prediction_margin",
        "observed_support",
    }
    assert weights["evidential_certainty"].item() == 0.0
    assert weights["prediction_margin"].item() == 0.0
    assert weights["observed_support"].item() > 0.0

    low = torch.tensor([[0.5, 0.5, 0.1, 0.0]])
    high = torch.tensor([[0.5, 0.5, 0.9, 0.0]])
    assert branch(high).item() > branch(low).item()

    features = torch.tensor(
        [[0.2, 0.3, 0.4, 0.0], [0.6, 0.8, 0.9, 1.0]],
        requires_grad=True,
    )
    branch.forward_logit(features).sum().backward()
    assert features.grad is not None
    assert torch.all(features.grad[:, 2] > 0.0)


def test_support_aware_calibrator_keeps_graph_and_manifest_intrinsic():
    intrinsic = MonotonicReliabilityCalibrator(
        use_api_observed_support=False
    )
    calibrator = MonotonicReliabilityCalibrator(
        use_api_observed_support=True
    )
    assert calibrator.feature_layout == API_SUPPORT_RELIABILITY_FEATURE_LAYOUT
    assert calibrator.branches["api"].raw_continuous_weights.shape == (3,)
    assert calibrator.branches["graph"].raw_continuous_weights.shape == (2,)
    assert (
        calibrator.branches["manifest"].raw_continuous_weights.shape == (2,)
    )

    support = torch.tensor([0.25, 0.75])
    features = _features(api_observed_support=support)
    outputs = calibrator(features, alive=_alive())
    intrinsic_outputs = intrinsic(_features(), alive=_alive())
    assert outputs["reliability_features_api"].shape == (2, 4)
    assert outputs["reliability_features_graph"].shape == (2, 3)
    assert outputs["reliability_features_manifest"].shape == (2, 3)
    assert torch.equal(outputs["observed_support_api"], support)
    assert outputs["api_observed_support_feature_active"].eq(1).all()
    for branch in ("graph", "manifest"):
        assert torch.equal(
            calibrator.branches[branch].raw_continuous_weights,
            intrinsic.branches[branch].raw_continuous_weights,
        )
        assert torch.equal(
            outputs[f"predicted_reliability_{branch}"],
            intrinsic_outputs[f"predicted_reliability_{branch}"],
        )


def test_support_layout_mismatch_fails_closed_per_branch():
    support_aware = MonotonicReliabilityCalibrator(
        use_api_observed_support=True
    )
    with pytest.raises(ValueError, match=r"features\['api'\].*shape"):
        support_aware(_features(), alive=_alive())

    intrinsic = MonotonicReliabilityCalibrator(
        use_api_observed_support=False
    )
    with pytest.raises(ValueError, match=r"features\['api'\].*shape"):
        intrinsic(
            _features(api_observed_support=torch.tensor([0.3, 0.7])),
            alive=_alive(),
        )

    invalid_graph = _features(
        api_observed_support=torch.tensor([0.3, 0.7])
    )
    invalid_graph["graph"] = invalid_graph["api"].clone()
    with pytest.raises(ValueError, match=r"features\['graph'\].*shape"):
        support_aware(invalid_graph, alive=_alive())

    invalid_support = _features(
        api_observed_support=torch.tensor([0.3, 0.7])
    )
    invalid_support["api"][0, 2] = 1.01
    with pytest.raises(ValueError, match="within"):
        support_aware(invalid_support, alive=_alive())


def test_predicted_class_intercept_is_optional_and_signed():
    enabled = MonotonicBranchCorrectnessCalibrator(
        use_predicted_class_intercept=True
    )
    assert enabled.predicted_class_intercept is not None
    with torch.no_grad():
        enabled.predicted_class_intercept.fill_(-1.25)
    benign = torch.tensor([[0.5, 0.5, 0.0]])
    malware = torch.tensor([[0.5, 0.5, 1.0]])
    assert (
        enabled.forward_logit(malware) - enabled.forward_logit(benign)
    ).item() == pytest.approx(-1.25)

    disabled = MonotonicBranchCorrectnessCalibrator(
        use_predicted_class_intercept=False
    )
    assert disabled.predicted_class_intercept is None
    assert torch.equal(
        disabled.forward_logit(benign),
        disabled.forward_logit(malware),
    )
    assert "predicted_class_intercept" not in disabled.state_dict()


def test_alive_is_a_hard_output_mask_not_a_learned_feature():
    calibrator = MonotonicReliabilityCalibrator()
    features = _features()
    alive = _alive()
    alive["api"] = torch.tensor([1.0, 0.0])
    outputs = calibrator(features, alive=alive)

    raw_probability = torch.sigmoid(
        outputs["predicted_reliability_logit_api"]
    )
    assert raw_probability[1].item() > 0.0
    assert outputs["predicted_reliability_api"][1].item() == 0.0
    assert torch.allclose(
        outputs["predicted_reliability_api"],
        raw_probability * alive["api"],
    )
    assert outputs["reliability_features_api"].shape == (2, 3)


def test_calibrator_outputs_sigmoid_correctness_probabilities_for_all_branches():
    calibrator = MonotonicReliabilityCalibrator()
    outputs = calibrator(_features(), alive=_alive())
    for branch in BRANCH_NAMES:
        probability = outputs[f"predicted_reliability_{branch}"]
        raw_logit = outputs[f"predicted_reliability_logit_{branch}"]
        assert torch.allclose(probability, torch.sigmoid(raw_logit))
        assert torch.all((probability >= 0.0) & (probability <= 1.0))
        assert torch.equal(
            outputs[f"reliability_features_{branch}"],
            _features()[branch],
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda feature, alive: feature["api"].__setitem__(
                (0, 0), 1.1
            ),
            "within",
        ),
        (
            lambda feature, alive: feature["api"].__setitem__(
                (0, 2), 0.5
            ),
            "binary",
        ),
        (
            lambda feature, alive: alive["api"].__setitem__(0, 0.5),
            "hard binary",
        ),
    ],
)
def test_calibrator_rejects_invalid_features_and_soft_alive_masks(
    mutator,
    message: str,
):
    feature = _features()
    alive = _alive()
    mutator(feature, alive)
    with pytest.raises(ValueError, match=message):
        MonotonicReliabilityCalibrator()(feature, alive=alive)


def test_nonfinite_parameters_fail_closed():
    branch = MonotonicBranchCorrectnessCalibrator()
    with torch.no_grad():
        branch.raw_continuous_weights[0] = float("nan")
    with pytest.raises(RuntimeError, match="remain finite"):
        branch(torch.tensor([[0.5, 0.5, 0.0]]))


def test_new_checkpoint_round_trip_is_strict_and_old_topology_is_rejected():
    source = MonotonicReliabilityCalibrator()
    with torch.no_grad():
        source.branches["api"].bias.fill_(1.5)
        source.branches["graph"].raw_continuous_weights.fill_(
            _raw_softplus(0.75)
        )
        assert (
            source.branches["manifest"].predicted_class_intercept is not None
        )
        source.branches["manifest"].predicted_class_intercept.fill_(-0.4)

    restored = MonotonicReliabilityCalibrator()
    incompatible = restored.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    old_state = dict(source.state_dict())
    old_state["branches.api.competence.bias"] = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(old_state, strict=True)


def test_api_support_checkpoint_requires_the_support_aware_topology():
    source = MonotonicReliabilityCalibrator(
        use_api_observed_support=True
    )
    with torch.no_grad():
        source.branches["api"].raw_continuous_weights[2].fill_(
            _raw_softplus(0.9)
        )
    restored = MonotonicReliabilityCalibrator(
        use_api_observed_support=True
    )
    incompatible = restored.load_state_dict(
        source.state_dict(),
        strict=True,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(
        restored.branches["api"].raw_continuous_weights,
        source.branches["api"].raw_continuous_weights,
    )

    intrinsic = MonotonicReliabilityCalibrator(
        use_api_observed_support=False
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        intrinsic.load_state_dict(source.state_dict(), strict=True)


def test_effective_parameter_details_expose_deployed_api_support_weight():
    calibrator = MonotonicReliabilityCalibrator(
        use_api_observed_support=True
    )
    with torch.no_grad():
        calibrator.branches["api"].raw_continuous_weights[2].fill_(
            _raw_softplus(0.9)
        )

    details = calibrator.effective_parameter_details()

    assert details["api_observed_support_enabled"] is True
    assert details["feature_layout"]["api"] == [
        "evidential_certainty",
        "prediction_margin",
        "observed_support",
        "predicted_malware_indicator",
    ]
    assert details["branches"]["api"]["continuous_weights"][
        "observed_support"
    ] == pytest.approx(0.9)
    assert "observed_support" not in details["branches"]["graph"][
        "continuous_weights"
    ]
    assert "observed_support" not in details["branches"]["manifest"][
        "continuous_weights"
    ]


def test_removed_quality_and_two_stage_apis_do_not_exist():
    removed = (
        "CleanCompetenceHead",
        "NonnegativeDegradationPenalty",
        "build_monotonic_reliability_features",
        "fit_nonnegative_observable_degradation",
        "fit_class_conditional_nonnegative_observable_degradation",
    )
    assert all(
        not hasattr(reliability_module, name)
        for name in removed
    )


def _branch_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "api": logits,
        "graph": logits.flip(dims=(-1,)),
        "manifest": logits * 0.5,
    }


def test_temperature_comparator_is_exact_max_softmax_and_alive_masked():
    logits = torch.tensor([[3.0, -1.0], [-0.5, 1.5]])
    calibrator = BranchTemperatureScalingConfidenceCalibrator()
    with torch.no_grad():
        for branch in BRANCH_NAMES:
            calibrator.log_temperatures[branch].fill_(math.log(2.0))
    alive = _alive(2, dead_branch="manifest")
    outputs = calibrator(_branch_logits(logits), alive=alive)

    expected = torch.softmax(logits / 2.0, dim=-1).amax(dim=-1)
    assert torch.allclose(outputs["predicted_reliability_api"], expected)
    assert torch.equal(
        outputs["predicted_reliability_manifest"],
        torch.zeros(2),
    )
    assert torch.allclose(
        outputs["reliability_temperature_api"],
        torch.full((2,), 2.0),
    )
    assert torch.equal(
        outputs["temperature_scaling_confidence_baseline_active"],
        torch.ones(2),
    )


def test_temperature_comparator_branch_nll_fits_one_positive_scalar():
    logits = torch.tensor(
        [
            [8.0, -8.0],
            [8.0, -8.0],
            [8.0, -8.0],
            [8.0, -8.0],
            [-8.0, 8.0],
            [-8.0, 8.0],
            [-8.0, 8.0],
            [-8.0, 8.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0])
    alive = torch.ones(labels.numel())
    calibrator = BranchTemperatureScalingConfidenceCalibrator()
    before = float(
        calibrator.branch_nll("api", logits, labels, alive).detach()
    )
    optimizer = torch.optim.LBFGS(
        calibrator.branch_parameters("api"),
        lr=1.0,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = calibrator.branch_nll("api", logits, labels, alive)
        loss.backward()
        return loss

    optimizer.step(closure)
    after = float(
        calibrator.branch_nll("api", logits, labels, alive).detach()
    )
    assert after < before
    assert float(calibrator.temperature("api").detach()) > 1.0


def test_temperature_comparator_rejects_branch_with_no_alive_rows():
    calibrator = BranchTemperatureScalingConfidenceCalibrator()
    with pytest.raises(ValueError, match="no alive rows"):
        calibrator.branch_nll(
            "api",
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([0]),
            torch.tensor([0.0]),
        )


def test_temperature_comparator_fails_closed_on_underflowed_temperature():
    calibrator = BranchTemperatureScalingConfidenceCalibrator()
    with torch.no_grad():
        calibrator.log_temperatures["api"].fill_(-1.0e30)
    with pytest.raises(ValueError, match="finite positive"):
        calibrator.branch_nll(
            "api",
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([0]),
            torch.tensor([1.0]),
        )


@pytest.mark.parametrize(
    "value",
    [MONOTONIC_CORRECTNESS_METHOD, TEMPERATURE_SCALING_CONFIDENCE_METHOD],
)
def test_canonical_method_names_are_stable(value: str):
    assert normalize_reliability_calibration_method(value) == value


@pytest.mark.parametrize(
    "removed_alias",
    ["monotonic", "learned_correctness", "temperature"],
)
def test_removed_method_aliases_fail_closed(removed_alias: str):
    with pytest.raises(ValueError, match="must be one of"):
        normalize_reliability_calibration_method(removed_alias)
