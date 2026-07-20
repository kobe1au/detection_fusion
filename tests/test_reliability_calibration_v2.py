import pytest
import torch

from fusion.constants import EvidenceIndex
from fusion.reliability_calibration import (
    ClassConditionalEmbeddingDensity,
    CleanCompetenceHead,
    MonotonicReliabilityCalibrator,
    NonnegativeDegradationPenalty,
    RELIABILITY_FEATURE_LAYOUT,
)


BRANCHES = ("api", "graph", "manifest")


def _evidence(batch_size: int = 1) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _probabilities(value=(0.8, 0.2)) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([value], dtype=torch.float32)
        for name in BRANCHES
    }


def _branch_logits(value=(2.0, -2.0)) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([value], dtype=torch.float32)
        for name in BRANCHES
    }


def _calibrator(**kwargs) -> MonotonicReliabilityCalibrator:
    return MonotonicReliabilityCalibrator(
        use_model_visibility=True,
        **kwargs,
    )


def _embedding_density_calibrator() -> MonotonicReliabilityCalibrator:
    return MonotonicReliabilityCalibrator(
        use_model_visibility=True,
        use_embedding_density=True,
        embedding_dims={name: 2 for name in BRANCHES},
        embedding_density_min_class_samples=2,
    )


def _clean_embedding_reference():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [4.0, 4.0],
            [4.1, 3.9],
            [3.9, 4.1],
        ],
        dtype=torch.float32,
    )
    return labels, {name: embeddings.clone() for name in BRANCHES}


def test_i1_separates_clean_competence_and_degradation_parameters():
    calibrator = _calibrator()
    competence = calibrator.competence_parameters()
    degradation = calibrator.degradation_parameters()
    assert competence
    assert degradation
    assert {id(value) for value in competence}.isdisjoint(
        {id(value) for value in degradation}
    )
    for branch in BRANCHES:
        module = calibrator.branches[branch]
        assert isinstance(module.competence, CleanCompetenceHead)
        assert isinstance(module.degradation, NonnegativeDegradationPenalty)
        assert [id(value) for value in calibrator.branch_parameters(branch)] == [
            *[
                id(value)
                for value in calibrator.branch_competence_parameters(branch)
            ],
            *[
                id(value)
                for value in calibrator.branch_degradation_parameters(branch)
            ],
        ]


def test_i1_has_exact_branch_local_feature_layouts():
    assert RELIABILITY_FEATURE_LAYOUT == {
        "api": (
            "effective_quality_deficit",
            "embedding_tail_q50",
            "embedding_tail_q80",
            "embedding_tail_q95",
            "prediction_margin",
            "predicted_malware_indicator",
        ),
        "graph": (
            "effective_quality_deficit",
            "embedding_tail_q50",
            "embedding_tail_q80",
            "embedding_tail_q95",
            "prediction_margin",
            "predicted_malware_indicator",
        ),
        "manifest": (
            "effective_quality_deficit",
            "embedding_tail_q50",
            "embedding_tail_q80",
            "embedding_tail_q95",
            "prediction_margin",
            "predicted_malware_indicator",
        ),
    }
    assert {name: len(layout) for name, layout in RELIABILITY_FEATURE_LAYOUT.items()} == {
        "api": 6,
        "graph": 6,
        "manifest": 6,
    }


def test_i1_requires_branch_probabilities_and_bounds_outputs():
    calibrator = _calibrator()
    with pytest.raises(ValueError, match="branch_probabilities"):
        calibrator(_evidence())

    outputs = calibrator(_evidence(), branch_probabilities=_probabilities())
    for name in BRANCHES:
        reliability = outputs[f"predicted_reliability_{name}"]
        assert reliability.shape == (1,)
        assert ((0.0 <= reliability) & (reliability <= 1.0)).all()
        assert outputs[f"reliability_features_superset_{name}"].shape[-1] == len(
            RELIABILITY_FEATURE_LAYOUT[name]
        )
        assert outputs[f"competence_design_{name}"].shape == (1, 4)
        assert outputs[f"degradation_design_{name}"].shape == (1, 5)
        assert torch.all(
            outputs[f"predicted_reliability_logit_{name}"]
            <= outputs[f"clean_competence_logit_{name}"]
        )
        assert outputs[f"degradation_penalty_{name}"].item() == pytest.approx(0.0)
        assert torch.equal(
            outputs[f"predicted_reliability_{name}"],
            outputs[f"clean_competence_{name}"],
        )


def test_reliability_probability_is_sigmoid_of_exported_raw_logit():
    calibrator = _calibrator()
    outputs = calibrator(_evidence(), branch_probabilities=_probabilities())

    for name in BRANCHES:
        expected = torch.sigmoid(outputs[f"predicted_reliability_logit_{name}"])
        assert torch.allclose(
            outputs[f"predicted_reliability_{name}"],
            expected * outputs[f"alive_{name}"],
        )


def test_i1_prediction_margin_is_strictly_positive_monotone():
    calibrator = _calibrator()
    low = _probabilities((0.55, 0.45))
    high = _probabilities((0.90, 0.10))

    low_output = calibrator(_evidence(), branch_probabilities=low)
    high_output = calibrator(_evidence(), branch_probabilities=high)

    for name in BRANCHES:
        assert high_output[f"prediction_margin_{name}"].item() == pytest.approx(
            0.8
        )
        assert low_output[f"prediction_margin_{name}"].item() == pytest.approx(
            0.1
        )
        assert (
            high_output[f"predicted_reliability_{name}"]
            > low_output[f"predicted_reliability_{name}"]
        ).all()


def test_i1_supports_predicted_class_conditional_correctness():
    calibrator = _calibrator()
    with torch.no_grad():
        calibrator.branches["api"].competence.predicted_class_weight.fill_(1.0)

    predicted_benign = _probabilities((0.8, 0.2))
    predicted_malware = _probabilities((0.2, 0.8))
    benign_output = calibrator(
        _evidence(), branch_probabilities=predicted_benign
    )
    malware_output = calibrator(
        _evidence(), branch_probabilities=predicted_malware
    )

    # Both examples have the same margin. Only the learned signed class offset
    # differs, proving that one scalar output can be class-conditional.
    assert benign_output["prediction_margin_api"].item() == pytest.approx(
        malware_output["prediction_margin_api"].item()
    )
    assert benign_output["predicted_malware_indicator_api"].item() == 0.0
    assert malware_output["predicted_malware_indicator_api"].item() == 1.0
    assert (
        malware_output["predicted_reliability_api"]
        > benign_output["predicted_reliability_api"]
    ).all()


def test_i1_api_features_are_independent_of_graph_availability():
    calibrator = _calibrator()
    complete = _evidence()
    graph_missing = complete.clone()
    graph_missing[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    graph_missing[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    graph_missing[:, EvidenceIndex.CODE_INTEGRITY] = 0.0
    probabilities = _probabilities()

    complete_output = calibrator(
        complete, branch_probabilities=probabilities
    )
    missing_output = calibrator(
        graph_missing, branch_probabilities=probabilities
    )

    assert torch.equal(
        complete_output["reliability_features_superset_api"],
        missing_output["reliability_features_superset_api"],
    )
    assert torch.equal(
        complete_output["predicted_reliability_api"],
        missing_output["predicted_reliability_api"],
    )


def test_effective_quality_deficit_can_only_reduce_reliability():
    calibrator = _calibrator()
    evidence = _evidence(2)
    evidence[1, EvidenceIndex.API_INTEGRITY] = 0.5
    evidence[1, EvidenceIndex.API_ENCODER_COVERAGE] = 0.5
    probabilities = {
        name: value.expand(2, -1).clone()
        for name, value in _probabilities((0.8, 0.2)).items()
    }
    outputs = calibrator(evidence, branch_probabilities=probabilities)

    assert outputs["clean_competence_api"][0].item() == pytest.approx(
        outputs["clean_competence_api"][1].item()
    )
    assert outputs["effective_quality_deficit_api"].tolist() == pytest.approx(
        [0.0, 0.75]
    )
    assert (
        outputs["degradation_quality_penalty_api"][1]
        > outputs["degradation_quality_penalty_api"][0]
    )
    assert (
        outputs["predicted_reliability_api"][1]
        < outputs["predicted_reliability_api"][0]
    )


def test_degradation_penalty_has_no_bias_and_detaches_clean_competence():
    calibrator = _calibrator()
    branch = calibrator.branches["api"]
    assert not hasattr(branch.degradation, "bias")
    features = torch.tensor([[0.5, 0.2, 0.4, 0.8, 0.9, 1.0]])
    outputs = branch.forward_components(features)
    outputs["degradation_penalty"].sum().backward()

    assert all(parameter.grad is None for parameter in branch.competence_parameters())
    assert all(
        parameter.grad is not None for parameter in branch.degradation_parameters()
    )
    weights = branch.degradation.effective_weights()
    assert weights["quality"].item() > 0.0
    assert (weights["tail"] > 0.0).all()
    assert weights["high_confidence_ood"].item() > 0.0


def test_embedding_density_requires_a_fitted_reference():
    calibrator = _embedding_density_calibrator()
    with pytest.raises(RuntimeError, match="reference is not fitted"):
        calibrator(
            _evidence(),
            branch_probabilities=_probabilities(),
            branch_logits=_branch_logits(),
            branch_embeddings={
                name: torch.zeros(1, 2) for name in BRANCHES
            },
        )


def test_embedding_density_is_branch_local_and_monotone_in_typicality():
    calibrator = _embedding_density_calibrator()
    labels, clean_embeddings = _clean_embedding_reference()
    calibrator.fit_embedding_references(
        clean_embeddings,
        labels,
        _evidence(labels.numel()),
    )

    evidence = _evidence(2)
    probabilities = _probabilities((0.8, 0.2))
    probabilities = {
        name: value.expand(2, -1).clone()
        for name, value in probabilities.items()
    }
    query = {
        name: torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        for name in BRANCHES
    }
    query["api"][1] = torch.tensor([20.0, 20.0])
    outputs = calibrator(
        evidence,
        branch_probabilities=probabilities,
        branch_logits={
            name: torch.tensor([[2.0, -2.0], [2.0, -2.0]])
            for name in BRANCHES
        },
        branch_embeddings=query,
    )

    assert (
        outputs["embedding_in_distribution_score_api"][0]
        > outputs["embedding_in_distribution_score_api"][1]
    )
    assert (
        outputs["predicted_reliability_api"][0]
        > outputs["predicted_reliability_api"][1]
    )
    assert outputs["embedding_tail_q95_api"][0] < outputs[
        "embedding_tail_q95_api"
    ][1]
    assert outputs["degradation_tail_penalty_api"][0] < outputs[
        "degradation_tail_penalty_api"
    ][1]
    assert outputs["degradation_high_confidence_ood_penalty_api"][0] < outputs[
        "degradation_high_confidence_ood_penalty_api"
    ][1]
    for name in ("graph", "manifest"):
        assert torch.equal(
            outputs[f"embedding_in_distribution_score_{name}"][0],
            outputs[f"embedding_in_distribution_score_{name}"][1],
        )
        assert torch.equal(
            outputs[f"predicted_reliability_{name}"][0],
            outputs[f"predicted_reliability_{name}"][1],
        )


def test_embedding_density_selects_reference_with_raw_logit_argmax():
    calibrator = _embedding_density_calibrator()
    labels, clean_embeddings = _clean_embedding_reference()
    calibrator.fit_embedding_references(
        clean_embeddings,
        labels,
        _evidence(labels.numel()),
    )
    # A ReLU evidential transform would turn both negative logits into zero
    # evidence and produce a 0.5/0.5 opinion tie. I1 correctness is defined by
    # raw-logit argmax, so density must still select class 1's clean reference.
    tied_opinion = _probabilities((0.5, 0.5))
    raw_class_one = _branch_logits((-2.0, -1.0))
    class_one_query = {
        name: torch.tensor([[4.0, 4.0]]) for name in BRANCHES
    }
    outputs = calibrator(
        _evidence(),
        branch_probabilities=tied_opinion,
        branch_logits=raw_class_one,
        branch_embeddings=class_one_query,
    )
    for name in BRANCHES:
        assert outputs[f"predicted_malware_indicator_{name}"].item() == 1.0
        assert outputs[f"embedding_in_distribution_score_{name}"].item() > 0.9


def test_embedding_density_constant_reference_is_finite_and_checkpointable():
    reference = ClassConditionalEmbeddingDensity(
        2,
        min_class_samples=2,
    )
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    reference.fit(embeddings, labels, torch.ones(4, dtype=torch.bool))
    score, distance = reference(
        torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        torch.tensor([0, 1]),
    )
    assert torch.isfinite(score).all()
    assert torch.isfinite(distance).all()
    assert ((0.0 <= score) & (score <= 1.0)).all()
    tail = reference.tail_basis(distance, torch.tensor([0, 1]))
    assert torch.isfinite(tail).all()
    assert tail.shape == (2, 3)
    assert ((0.0 <= tail) & (tail <= 1.0)).all()
    assert torch.all(
        reference.distance_quantiles[:, 1:]
        >= reference.distance_quantiles[:, :-1]
    )

    clone = ClassConditionalEmbeddingDensity(2, min_class_samples=2)
    clone.load_state_dict(reference.state_dict(), strict=True)
    clone_score, clone_distance = clone(
        torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        torch.tensor([0, 1]),
    )
    assert torch.equal(score, clone_score)
    assert torch.equal(distance, clone_distance)
    clone_tail = clone.tail_basis(clone_distance, torch.tensor([0, 1]))
    assert torch.equal(tail, clone_tail)
