import json
import math

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from fusion.constants import EvidenceIndex
from fusion.dataset import (
    RobustTriModalDataset,
    apply_graph_encoder_budget,
    robust_collate_fn,
)
from fusion.evidence import build_evidence
from fusion.quality import (
    OBSERVABLE_NUMERIC_FIELDS,
    OBSERVABLE_SIGNAL_FIELDS,
    refresh_observable_signals,
)
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import (
    compute_posthoc_calibration_loss,
    compute_reliability_calibration_loss,
)
from fusion.reliability_calibration import MonotonicReliabilityCalibrator
from fusion.train import (
    _branch_prediction_row,
    _selective_metrics,
    _selective_ranking_metrics,
    _write_metrics_json,
    compute_branch_reliability_metrics,
    fit_posthoc_calibration,
    fit_rejection_threshold,
    split_posthoc_conformal_dataset,
    split_validation_dataset,
)


def _evidence(batch_size: int = 1) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _logits(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor([[2.0, -2.0]] * batch_size) for _ in range(3))


def _branch_probabilities(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "api": torch.tensor([[0.8, 0.2]]).repeat(batch_size, 1),
        "graph": torch.tensor([[0.1, 0.9]]).repeat(batch_size, 1),
        "manifest": torch.tensor([[0.7, 0.3]]).repeat(batch_size, 1),
    }


def test_monotonic_calibrator_respects_intrinsic_integrity_direction():
    calibrator = MonotonicReliabilityCalibrator()
    low = _evidence()
    high = low.clone()
    low[:, EvidenceIndex.API_INTEGRITY] = 0.2
    high[:, EvidenceIndex.API_INTEGRITY] = 0.8
    probabilities = _branch_probabilities()
    assert (
        calibrator(high, branch_probabilities=probabilities)[
            "predicted_reliability_api"
        ]
        >= calibrator(low, branch_probabilities=probabilities)[
            "predicted_reliability_api"
        ]
    ).all()


def test_i1_uses_only_the_formal_branch_local_feature_layout():
    calibrator = MonotonicReliabilityCalibrator(use_model_visibility=True)
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.2
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.4
    evidence[:, EvidenceIndex.MANIFEST_INTEGRITY] = 0.6
    evidence[:, EvidenceIndex.API_ENCODER_COVERAGE] = 0.5
    evidence[:, EvidenceIndex.GRAPH_ENCODER_COVERAGE] = 0.75

    outputs = calibrator(
        evidence,
        branch_probabilities=_branch_probabilities(),
    )

    assert outputs["reliability_features_superset_api"].shape == (1, 6)
    assert outputs["reliability_features_superset_graph"].shape == (1, 6)
    assert outputs["reliability_features_superset_manifest"].shape == (1, 6)
    assert outputs["reliability_features_superset_api"][0].tolist() == pytest.approx(
        [0.9, 0.0, 0.0, 0.0, 0.6, 0.0]
    )
    assert outputs["reliability_features_superset_graph"][0].tolist() == pytest.approx(
        [0.7, 0.0, 0.0, 0.0, 0.8, 1.0]
    )
    assert outputs["reliability_features_superset_manifest"][0].tolist() == pytest.approx(
        [0.4, 0.0, 0.0, 0.0, 0.4, 0.0]
    )


def test_pairwise_semantics_remain_diagnostic_only_for_i1():
    calibrator = MonotonicReliabilityCalibrator()
    first = _evidence()
    second = first.clone()
    first[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.1
    first[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.9
    first[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.9
    second[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.9
    second[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.1
    second[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.1

    probabilities = _branch_probabilities()
    first_out = calibrator(first, branch_probabilities=probabilities)
    second_out = calibrator(second, branch_probabilities=probabilities)
    for name in ("api", "graph", "manifest"):
        assert torch.allclose(
            first_out[f"predicted_reliability_{name}"],
            second_out[f"predicted_reliability_{name}"],
        )

def test_encoder_visibility_is_a_monotone_reliability_source():
    calibrator = MonotonicReliabilityCalibrator(
        use_model_visibility=True,
    )
    low = _evidence()
    high = low.clone()
    low[:, EvidenceIndex.API_ENCODER_COVERAGE] = 0.2
    high[:, EvidenceIndex.API_ENCODER_COVERAGE] = 0.8

    probabilities = _branch_probabilities()
    low_outputs = calibrator(low, branch_probabilities=probabilities)
    high_outputs = calibrator(high, branch_probabilities=probabilities)
    low_features = low_outputs["reliability_features_api"]
    high_features = high_outputs["reliability_features_api"]
    # Density is disabled, so the active view contains only quality deficit,
    # margin and predicted class. Greater visibility reduces the first value.
    assert low_features.shape[-1] == 3
    assert low_features[0, 0].item() == pytest.approx(0.8)
    assert high_features[0, 0].item() == pytest.approx(0.2)
    assert (
        high_outputs["predicted_reliability_api"]
        > low_outputs["predicted_reliability_api"]
    ).all()


def test_intrinsic_single_modality_features_use_self_integrity_once():
    calibrator = MonotonicReliabilityCalibrator(
        use_prediction_margin=False,
        use_predicted_class_feature=False,
    )
    evidence = _evidence()
    integrity = {"api": 0.2, "graph": 0.4, "manifest": 0.6}
    evidence[:, EvidenceIndex.API_INTEGRITY] = integrity["api"]
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = integrity["graph"]
    evidence[:, EvidenceIndex.MANIFEST_INTEGRITY] = integrity["manifest"]
    evidence[:, EvidenceIndex.CODE_INTEGRITY] = 0.9

    outputs = calibrator(evidence, branch_probabilities=_branch_probabilities())
    for name, value in integrity.items():
        expected = torch.tensor([[1.0 - value]])
        assert torch.allclose(outputs[f"reliability_features_{name}"], expected)


def test_intrinsic_api_reliability_is_neutral_to_missing_graph():
    calibrator = MonotonicReliabilityCalibrator()
    complete = _evidence()
    missing_graph = complete.clone()
    missing_graph[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    missing_graph[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    missing_graph[:, EvidenceIndex.CODE_INTEGRITY] = 0.0

    probabilities = _branch_probabilities()
    assert torch.allclose(
        calibrator(complete, branch_probabilities=probabilities)[
            "predicted_reliability_api"
        ],
        calibrator(missing_graph, branch_probabilities=probabilities)[
            "predicted_reliability_api"
        ],
    )


def test_intrinsic_api_reliability_is_neutral_to_degraded_alive_graph():
    calibrator = MonotonicReliabilityCalibrator()
    complete = _evidence()
    degraded_graph = complete.clone()
    degraded_graph[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.01
    degraded_graph[:, EvidenceIndex.CODE_INTEGRITY] = 0.1

    probabilities = _branch_probabilities()
    complete_out = calibrator(complete, branch_probabilities=probabilities)
    degraded_out = calibrator(degraded_graph, branch_probabilities=probabilities)
    assert torch.allclose(
        complete_out["predicted_reliability_api"],
        degraded_out["predicted_reliability_api"],
    )
    assert degraded_out["predicted_reliability_graph"].item() < complete_out[
        "predicted_reliability_graph"
    ].item()

def test_calibrator_alive_mask_can_be_disabled_for_ablation():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    masked = MonotonicReliabilityCalibrator(apply_alive_mask=True)
    unmasked = MonotonicReliabilityCalibrator(apply_alive_mask=False)

    probabilities = _branch_probabilities()
    assert (
        masked(evidence, branch_probabilities=probabilities)[
            "predicted_reliability_api"
        ].item()
        == 0.0
    )
    assert (
        unmasked(evidence, branch_probabilities=probabilities)[
            "predicted_reliability_api"
        ].item()
        > 0.0
    )


def test_missing_manifest_disables_code_manifest_conflict_penalty():
    base = _evidence()
    base[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    high_conflict = base.clone()
    high_conflict[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 1.0
    fusion = DiscountProbabilityFusion()
    base_out = fusion(*_logits(), base)
    conflict_out = fusion(*_logits(), high_conflict)
    assert conflict_out["manifest_code_conflict_applicable"].item() == 0.0
    assert conflict_out["effective_conflict"].item() == 0.0
    assert conflict_out["discount_api"].item() == pytest.approx(base_out["discount_api"].item())
    assert conflict_out["discount_graph"].item() == pytest.approx(base_out["discount_graph"].item())


def test_empty_manifest_code_semantic_relation_does_not_apply_explicit_discount():
    evidence = _evidence()
    evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0

    output = DiscountProbabilityFusion()(*_logits(), evidence)

    assert output["manifest_code_relation_applicable"].item() == 0.0
    assert output["effective_conflict"].item() == 0.0
    assert output["discount_manifest"].item() == pytest.approx(
        output["confidence_factor_manifest"].item()
    )


def test_route_distribution_override_rejects_non_routed_combination():
    fusion = DiscountProbabilityFusion({"combination": "dempster"})

    with pytest.raises(ValueError, match="only valid for the routed"):
        fusion(
            *_logits(),
            _evidence(),
            branch_distribution_override=torch.full((1, 3), 1.0 / 3.0),
        )


def test_routing_risk_bce_uses_raw_logit_without_saturation():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "risk_mode": "learned",
                "posthoc_refine": True,
            },
            "reliability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    assert fusion.opinion_router is not None
    with torch.no_grad():
        fusion.opinion_router.risk_bias.fill_(-20.0)
        fusion.opinion_router.raw_risk_feature_weights.fill_(-30.0)

    evidence = _evidence()
    outputs = fusion(*_logits(), evidence)
    loss, _ = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([1]),
        evidence,
        {
            "reliability_calibration": {"weight": 0.0},
            "probability_calibration": {"weight": 0.0},
            "routing": {
                "enabled": True,
                "posthoc_refine": True,
                "prediction_loss_weight": 0.0,
                "route_oracle_loss_weight": 0.0,
                "risk_loss_weight": 1.0,
                "risk_loss": "bce",
            },
        },
    )
    loss.backward()

    assert outputs["routing_risk_probability"].item() < 1.0e-7
    assert fusion.opinion_router.risk_bias.grad is not None
    assert fusion.opinion_router.risk_bias.grad.item() < -0.99


def test_posthoc_calibration_loss_updates_calibration_parameters():
    fusion = DiscountProbabilityFusion(
        {
            "reliability_calibration": {"enabled": True},
            "probability_calibration": {"enabled": True},
        }
    )
    fusion.set_calibration_active(True)
    evidence = _evidence(2)
    logits = tuple(
        torch.tensor([[3.0, -3.0], [-3.0, 3.0]], dtype=torch.float32)
        for _ in range(3)
    )
    outputs = fusion(*logits, evidence)
    outputs.update(
        {
            f"{name}_logits_aux": branch_logits
            for name, branch_logits in zip(
                ("api", "graph", "manifest"), logits
            )
        }
    )
    loss, parts = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 0]),
        evidence,
        {
            "reliability_calibration": {"weight": 1.0},
            "probability_calibration": {"weight": 1.0},
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["reliability_calibration_loss"] > 0.0
    assert parts["probability_calibration_loss"] > 0.0
    assert all(parameter.grad is not None for parameter in fusion.calibration_parameters())


def test_formal_reliability_fit_uses_exact_three_independent_branches():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "dempster",
            "reliability_calibration": {
                "enabled": True,
                "branches": ["api", "graph", "manifest"],
            },
            "probability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    evidence = _evidence(2)
    logits = tuple(
        torch.tensor([[3.0, -3.0], [-3.0, 3.0]], dtype=torch.float32)
        for _ in range(3)
    )
    outputs = fusion(*logits, evidence)
    outputs.update(
        {
            f"{name}_logits_aux": branch_logits
            for name, branch_logits in zip(
                ("api", "graph", "manifest"), logits
            )
        }
    )
    loss, diagnostics = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 1]),
        evidence,
        {
            "reliability_calibration": {
                "weight": 1.0,
                "branches": ["api", "graph", "manifest"],
            },
            "probability_calibration": {"weight": 0.0},
        },
    )
    loss.backward()

    assert set(fusion.reliability_calibrator.branches) == {
        "api",
        "graph",
        "manifest",
    }
    for name in ("api", "graph", "manifest"):
        assert all(
            parameter.grad is not None
            for parameter in fusion.reliability_calibrator.branches[name].parameters()
        )


def test_posthoc_reliability_loss_is_safe_under_autocast():
    reliability = torch.tensor([0.8, 0.2], dtype=torch.float32, requires_grad=True)
    outputs = {
        "predicted_reliability_api": reliability,
        "api_logits_aux": torch.tensor([[3.0, -3.0], [3.0, -3.0]], dtype=torch.float32),
    }
    evidence = _evidence(2)

    with torch.amp.autocast(device_type="cpu", enabled=True):
        loss, parts = compute_posthoc_calibration_loss(
            outputs,
            torch.tensor([0, 1]),
            evidence,
            {
                "reliability_calibration": {"weight": 1.0, "branches": ["api"]},
                "probability_calibration": {"weight": 0.0},
            },
        )

    loss.backward()
    assert torch.isfinite(loss)
    assert parts["reliability_calibration_loss"] > 0.0
    assert reliability.grad is not None


class _CalibrationGraph:
    def __init__(
        self,
        evidence: torch.Tensor,
        embeddings: dict[str, torch.Tensor] | None = None,
    ):
        self.evidence = evidence
        self.embeddings = embeddings

    def to(self, device, non_blocking=True):
        self.evidence = self.evidence.to(device)
        if isinstance(self.embeddings, dict):
            self.embeddings = {
                name: value.to(device)
                for name, value in self.embeddings.items()
            }
        return self


class _PosthocCalibrationModel(nn.Module):
    fusion_mode = "discount_probability"

    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "use_confidence_proxy": True,
                "reliability_calibration": {
                    "enabled": True,
                },
                "probability_calibration": {"enabled": True},
            }
        )

    def calibration_parameters(self):
        return self.discount_fusion.calibration_parameters()

    def set_calibration_active(self, enabled: bool):
        self.discount_fusion.set_calibration_active(enabled)

    def forward(self, graph, return_features=False):
        self.forward_calls += 1
        batch_size = graph.evidence.size(0)
        branch_logits = tuple(
            graph.evidence.new_tensor([[2.0, -2.0], [-2.0, 2.0]])[:batch_size]
            for _ in range(3)
        )
        outputs = self.discount_fusion(*branch_logits, graph.evidence)
        for name, logits in zip(
            ("api", "graph", "manifest"),
            branch_logits,
        ):
            outputs[f"{name}_logits_aux"] = logits
        outputs["gate_evidence"] = graph.evidence
        return outputs["final_logits"], outputs


class _PosthocRoutedModel(_PosthocCalibrationModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.forward_calls = 0
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "combination": "routed",
                "routing": {
                    "enabled": True,
                    "train_end_to_end": False,
                    "posthoc_refine": True,
                    "prediction_loss_weight": 1.0,
                    "risk_mode": "learned",
                    "risk_loss_weight": 1.0,
                    "final_temperature_scaling": True,
                },
                "reliability_calibration": {
                    "enabled": True,
                    "branches": ["api", "graph", "manifest"],
                },
                "probability_calibration": {"enabled": False},
            }
        )


class _PosthocRouterOnlyModel(_PosthocCalibrationModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.forward_calls = 0
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "combination": "routed",
                "routing": {
                    "enabled": True,
                    "train_end_to_end": False,
                    "posthoc_refine": True,
                    "prediction_loss_weight": 1.0,
                    "risk_mode": "learned",
                    "risk_loss_weight": 1.0,
                    "final_temperature_scaling": False,
                },
                "reliability_calibration": {"enabled": False},
                "probability_calibration": {"enabled": False},
            }
        )


class _PosthocEmbeddingDensityModel(_PosthocCalibrationModel):
    def __init__(self, *, routed: bool = False):
        nn.Module.__init__(self)
        self.forward_calls = 0
        routing = (
            {
                "enabled": True,
                "mode": "prior_only",
                "train_end_to_end": False,
                "posthoc_refine": False,
                "prediction_loss_weight": 0.0,
                "risk_mode": "reliability_prior",
                "risk_target": "reliability_deficit_score",
                "risk_loss_weight": 0.0,
                "final_temperature_scaling": False,
            }
            if routed
            else {"enabled": False}
        )
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "combination": "routed" if routed else "linear",
                "routing": routing,
                "use_support_discount": False,
                "use_conflict_discount": False,
                "reliability_calibration": {
                    "enabled": True,
                    "branches": ["api", "graph", "manifest"],
                    "use_embedding_density": True,
                    "embedding_dims": {
                        "api": 2,
                        "graph": 2,
                        "manifest": 2,
                    },
                    "embedding_density_min_class_samples": 2,
                },
                "probability_calibration": {"enabled": False},
            }
        )

    def forward(self, graph, return_features=False):
        self.forward_calls += 1
        batch_size = graph.evidence.size(0)
        base_logits = graph.evidence.new_tensor(
            [
                [2.0, -2.0],
                [1.5, -1.5],
                [-2.0, 2.0],
                [-1.5, 1.5],
            ]
        )[:batch_size]
        embeddings = graph.embeddings
        assert isinstance(embeddings, dict)
        outputs = self.discount_fusion(
            base_logits,
            base_logits,
            base_logits,
            graph.evidence,
            embeddings=embeddings,
        )
        for name in ("api", "graph", "manifest"):
            outputs[f"{name}_logits_aux"] = base_logits
            if return_features:
                outputs[f"{name}_emb"] = embeddings[name]
        outputs["gate_evidence"] = graph.evidence
        return outputs["final_logits"], outputs


def test_posthoc_i1_caches_embeddings_and_fits_clean_fold_local_reference():
    evidence = _evidence(4)
    labels = torch.tensor([0, 0, 1, 1])
    clean = torch.tensor(
        [[0.0, 0.0], [0.1, -0.1], [4.0, 4.0], [4.1, 3.9]]
    )
    clean_embeddings = {
        name: clean.clone() for name in ("api", "graph", "manifest")
    }
    semantic_embeddings = {
        name: clean.clone() for name in ("api", "graph", "manifest")
    }
    semantic_embeddings["api"] = clean + 20.0

    def _loader(embeddings):
        return [
            {
                "graph_batch": _CalibrationGraph(evidence.clone(), embeddings),
                "labels": labels.clone(),
                "sids": ["a", "b", "c", "d"],
                "quality": {},
                "num_failed": 0,
            }
        ]

    model = _PosthocEmbeddingDensityModel()
    summary = fit_posthoc_calibration(
        model,
        [
            {
                "name": "clean",
                "scenario_group": "clean",
                "reliability_branches": ["api", "graph", "manifest"],
                "loader": _loader(clean_embeddings),
            },
            {
                "name": "calibration_api_semantic_corrupted_s0.9",
                "scenario_group": "api_semantic_corrupted",
                "perturb_type": "api_semantic_corrupted",
                "objective_family": "single_semantic",
                "strength": 0.9,
                "reliability_branches": ["api"],
                "loader": _loader(semantic_embeddings),
            },
        ],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "epochs": 1,
                "stage_optimization": {
                    "default": {
                        "optimizer": "adam",
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "convergence_tolerance": 1.0e-8,
                        "gradient_tolerance": 1.0e-8,
                        "lr": 1.0e-2,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "routing": {
                    "enabled": False,
                    "subset_oracle_loss_weight": 0.0,
                    "subset_oracle_temperature": 1.0,
                    "group_robust_objective": {
                        "enabled": False,
                        "soft_worst_weight": 0.0,
                    },
                },
                "reliability_calibration": {
                    "method": "monotonic_correctness",
                    "weight": 1.0,
                },
                "probability_calibration": {"weight": 0.0},
            },
        },
    )

    references = summary["reliability_embedding_references"]
    assert set(references) == {"api", "graph", "manifest"}
    assert all(item["fitted"] for item in references.values())
    assert all(item["class_count"] == [2, 2] for item in references.values())

    _, clean_outputs = model(
        _CalibrationGraph(evidence.clone(), clean_embeddings),
        return_features=True,
    )
    _, semantic_outputs = model(
        _CalibrationGraph(evidence.clone(), semantic_embeddings),
        return_features=True,
    )
    assert (
        semantic_outputs["embedding_in_distribution_score_api"].mean()
        < clean_outputs["embedding_in_distribution_score_api"].mean()
    )
    for name in ("graph", "manifest"):
        assert torch.equal(
            semantic_outputs[f"embedding_in_distribution_score_{name}"],
            clean_outputs[f"embedding_in_distribution_score_{name}"],
        )


def test_embedding_references_exclude_each_oof_holdout_then_full_refit():
    loader = []
    for source_fold in range(3):
        benign_center = float(source_fold * 5)
        malware_center = float(40 + source_fold * 7)
        embeddings = torch.tensor(
            [
                [benign_center, benign_center],
                [benign_center + 0.1, benign_center - 0.1],
                [malware_center, malware_center],
                [malware_center + 0.1, malware_center - 0.1],
            ]
        )
        loader.append(
            {
                "graph_batch": _CalibrationGraph(
                    _evidence(4),
                    {
                        name: embeddings.clone()
                        for name in ("api", "graph", "manifest")
                    },
                ),
                "labels": torch.tensor([0, 0, 1, 1]),
                "sids": [
                    f"benign-{source_fold}-0",
                    f"benign-{source_fold}-1",
                    f"malware-{source_fold}-0",
                    f"malware-{source_fold}-1",
                ],
                "quality": {},
                "num_failed": 0,
            }
        )

    model = _PosthocEmbeddingDensityModel(routed=True)
    summary = fit_posthoc_calibration(
        model,
        [loader],
        torch.device("cpu"),
        False,
        {
            "train": {"seed": 42},
            "calibration": {
                "enabled": True,
                "cross_fitting": {
                    "required": True,
                    "enabled": True,
                    "mode": "nested",
                    "num_folds": 3,
                },
                "stage_optimization": {
                    "default": {
                        "optimizer": "adam",
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "convergence_tolerance": 1.0e-8,
                        "gradient_tolerance": 1.0e-8,
                        "lr": 1.0e-2,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "routing": {
                    "enabled": True,
                    "mode": "prior_only",
                    "posthoc_refine": False,
                    "risk_mode": "reliability_prior",
                    "risk_target": "reliability_deficit_score",
                    "risk_loss_weight": 0.0,
                    "subset_oracle_loss_weight": 0.0,
                    "subset_oracle_temperature": 1.0,
                    "group_robust_objective": {
                        "enabled": False,
                        "soft_worst_weight": 0.0,
                    },
                },
                "reliability_calibration": {
                    "method": "monotonic_correctness",
                    "weight": 1.0,
                },
                "probability_calibration": {"weight": 0.0},
            },
        },
    )

    fold_references = [
        outer["holdout_reliability_fit"]["embedding_references"]["api"]
        for outer in summary["cross_fitting"]["outer_folds"]
    ]
    assert all(item["class_count"] == [4, 4] for item in fold_references)
    assert len({item["reference_sha256"] for item in fold_references}) == 3
    final_references = summary["reliability_embedding_references"]
    assert all(
        item["class_count"] == [6, 6]
        and len(item["reference_sha256"]) == 64
        for item in final_references.values()
    )


def test_i1_diagnostic_free_loss_preserves_value_and_gradients():
    labels = torch.tensor([0, 1, 1, 0])
    evidence = _evidence(4)
    auxiliary_logits = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0], [0.7, 0.2], [0.1, 0.8]]
    )
    reliability_logit = torch.nn.Parameter(torch.tensor([0.4, 0.7, -0.2, 0.1]))

    def _value_and_grad(materialize_diagnostics: bool):
        reliability_logit.grad = None
        outputs = {
            "predicted_reliability_api": torch.sigmoid(reliability_logit),
            "predicted_reliability_logit_api": reliability_logit,
            "api_logits_aux": auxiliary_logits,
        }
        loss, diagnostics = compute_reliability_calibration_loss(
            outputs,
            labels,
            evidence,
            {"branches": ["api"]},
            materialize_diagnostics=materialize_diagnostics,
        )
        loss.backward()
        return loss.detach(), diagnostics, reliability_logit.grad.detach().clone()

    full_loss, full_diagnostics, full_gradient = _value_and_grad(True)
    lean_loss, lean_diagnostics, lean_gradient = _value_and_grad(False)
    assert full_diagnostics
    assert lean_diagnostics == {}
    assert torch.allclose(lean_loss, full_loss, atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(
        lean_gradient, full_gradient, atol=1.0e-7, rtol=1.0e-7
    )


@pytest.mark.parametrize("stage", ["route", "risk"])
def test_posthoc_diagnostic_free_loss_preserves_value_and_gradients(stage: str):
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "mode": "learned",
                "train_end_to_end": False,
                "posthoc_refine": True,
                "prediction_loss_weight": 1.0,
                "route_oracle_loss_weight": 0.25,
                "risk_mode": "learned",
                "risk_loss_weight": 1.0,
            },
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    evidence = _evidence(4)
    labels = torch.tensor([0, 1, 1, 0])
    logits = (
        torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.2, 0.7], [1.0, -0.2]]),
        torch.tensor([[1.0, -0.3], [0.1, 0.8], [-0.4, 1.2], [0.5, 0.2]]),
        torch.tensor([[0.7, 0.1], [-0.5, 1.1], [0.3, 0.9], [1.4, -0.6]]),
    )
    if stage == "route":
        routing = {
            "enabled": True,
            "posthoc_refine": True,
            "prediction_loss_weight": 1.0,
            "route_oracle_loss_weight": 0.25,
            "risk_loss_weight": 0.0,
        }
        parameters = fusion.routing_distribution_parameters()
    else:
        routing = {
            "enabled": True,
            "posthoc_refine": True,
            "prediction_loss_weight": 0.0,
            "route_oracle_loss_weight": 0.0,
            "risk_loss_weight": 1.0,
            "risk_target": "mixture_argmax_error",
        }
        parameters = fusion.routing_risk_parameters()
    config = {
        "reliability_calibration": {"weight": 0.0},
        "probability_calibration": {"weight": 0.0},
        "routing": routing,
    }

    def _value_and_grad(materialize_diagnostics: bool):
        fusion.zero_grad(set_to_none=True)
        outputs = fusion(*logits, evidence)
        loss, diagnostics = compute_posthoc_calibration_loss(
            outputs,
            labels,
            evidence,
            config,
            materialize_diagnostics=materialize_diagnostics,
        )
        loss.backward()
        gradients = [parameter.grad.detach().clone() for parameter in parameters]
        return loss.detach(), diagnostics, gradients

    full_loss, full_diagnostics, full_gradients = _value_and_grad(True)
    lean_loss, lean_diagnostics, lean_gradients = _value_and_grad(False)

    assert full_diagnostics
    assert lean_diagnostics == {}
    assert torch.allclose(lean_loss, full_loss, atol=1.0e-7, rtol=1.0e-7)
    assert len(lean_gradients) == len(full_gradients)
    for lean_gradient, full_gradient in zip(lean_gradients, full_gradients):
        assert torch.allclose(
            lean_gradient, full_gradient, atol=1.0e-7, rtol=1.0e-7
        )


@pytest.mark.parametrize(
    "opinion_source", ["evidential", "softmax_fixed_uncertainty"]
)
@pytest.mark.parametrize("availability", ["all_alive", "mixed", "all_dead"])
def test_route_only_kernel_preserves_full_route_loss_and_gradients(
    opinion_source: str,
    availability: str,
):
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "opinion_source": opinion_source,
            "softmax_opinion": {"uncertainty": 0.4, "temperature": 1.0},
            "routing": {
                "enabled": True,
                "mode": "learned",
                "train_end_to_end": False,
                "posthoc_refine": True,
                "prediction_loss_weight": 1.0,
                "route_oracle_loss_weight": 0.25,
                "risk_mode": "learned",
                "risk_loss_weight": 1.0,
            },
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    labels = torch.tensor([0, 1, 1, 0])
    evidence = _evidence(4)
    if availability == "mixed":
        evidence[:, EvidenceIndex.API_ALIVE] = torch.tensor([1.0, 1.0, 0.0, 1.0])
        evidence[:, EvidenceIndex.GRAPH_ALIVE] = torch.tensor([1.0, 0.0, 1.0, 1.0])
        evidence[:, EvidenceIndex.MANIFEST_ALIVE] = torch.tensor(
            [0.0, 1.0, 1.0, 1.0]
        )
    elif availability == "all_dead":
        evidence[:, EvidenceIndex.API_ALIVE] = 0.0
        evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
        evidence[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    logits = (
        torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.2, 0.7], [1.0, -0.2]]),
        torch.tensor([[1.0, -0.3], [0.1, 0.8], [-0.4, 1.2], [0.5, 0.2]]),
        torch.tensor([[0.7, 0.1], [-0.5, 1.1], [0.3, 0.9], [1.4, -0.6]]),
    )
    config = {
        "reliability_calibration": {"weight": 0.0},
        "probability_calibration": {"weight": 0.0},
        "routing": {
            "enabled": True,
            "posthoc_refine": True,
            "prediction_loss_weight": 1.0,
            "route_oracle_loss_weight": 0.25,
            "risk_loss_weight": 0.0,
        },
    }
    route_parameters = fusion.routing_distribution_parameters()

    fusion.zero_grad(set_to_none=True)
    full_outputs = fusion(*logits, evidence)
    full_loss, _ = compute_posthoc_calibration_loss(
        full_outputs,
        labels,
        evidence,
        config,
        materialize_diagnostics=False,
    )
    full_loss.backward()
    full_gradients = [parameter.grad.detach().clone() for parameter in route_parameters]

    fusion.zero_grad(set_to_none=True)
    with torch.no_grad():
        static = fusion(*logits, evidence)
    branches = ("api", "graph", "manifest")
    routed = fusion.opinion_router(
        beliefs={
            name: static[f"routing_input_belief_{name}"] for name in branches
        },
        uncertainties={
            name: static[f"routing_input_uncertainty_{name}"] for name in branches
        },
        reliability={
            name: static[f"routing_input_reliability_{name}"] for name in branches
        },
        alive={name: static[f"routing_input_alive_{name}"] for name in branches},
        learned_active=True,
        compute_risk=False,
        eps=float(fusion.config.get("min_discount", 1.0e-8)),
    )
    route_only_outputs = {
        "routing_active": torch.ones_like(routed["has_available"]),
        "routing_has_available": routed["has_available"],
        "routing_mixture_prob": routed["mixture_probability"],
        "routing_branch_distribution": routed["branch_distribution"],
        "routing_scores": routed["routing_scores"],
        **{
            f"calibrated_log_prob_{name}": static[
                f"calibrated_log_prob_{name}"
            ]
            for name in branches
        },
    }
    route_only_loss, _ = compute_posthoc_calibration_loss(
        route_only_outputs,
        labels,
        evidence,
        config,
        materialize_diagnostics=False,
    )
    route_only_loss.backward()
    route_only_gradients = [
        parameter.grad.detach().clone() for parameter in route_parameters
    ]

    assert torch.allclose(route_only_loss, full_loss, atol=1.0e-7, rtol=1.0e-7)
    for route_only_gradient, full_gradient in zip(
        route_only_gradients, full_gradients
    ):
        assert torch.allclose(
            route_only_gradient, full_gradient, atol=1.0e-7, rtol=1.0e-7
        )


def test_posthoc_calibration_reports_bounded_numerical_optimization():
    evidence = _evidence(2)
    loader = [
        {
            "graph_batch": _CalibrationGraph(evidence),
            "labels": torch.tensor([0, 1]),
            "sids": ["a", "b"],
            "quality": {},
            "num_failed": 0,
        }
    ]
    model = _PosthocCalibrationModel()
    summary = fit_posthoc_calibration(
        model,
        [loader],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "stage_optimization": {
                    "default": {
                        "max_steps": 5,
                        "min_steps": 5,
                        "convergence_patience": 2,
                        "convergence_tolerance": 1.0e-12,
                        "lr": 1.0e-3,
                    }
                },
            },
            "fusion": {
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 1.0},
            },
        },
    )

    assert summary["best_epoch"] == 5
    assert summary["epochs_ran"] == 5
    assert summary["loss_evaluations_ran"] == 6
    reliability_stage = summary["stages"]["reliability"]
    assert reliability_stage["lifecycle"] == (
        "clean_competence_then_nonnegative_degradation_v1"
    )
    assert reliability_stage["degradation_never_increases_reliability"] is True
    assert set(reliability_stage["phases"]) == {"competence", "degradation"}
    phase_steps = sum(
        phase["total_steps"] for phase in reliability_stage["phases"].values()
    )
    assert reliability_stage["total_steps"] == phase_steps
    assert all(
        phase["total_steps"] <= phase["max_steps"]
        for phase in reliability_stage["phases"].values()
    )
    assert reliability_stage["stopped_early"] is any(
        phase["stopped_early"]
        for phase in reliability_stage["phases"].values()
    )
    assert summary["stopped_early"] is any(
        stage["stopped_early"]
        for stage in summary["stages"].values()
        if stage.get("enabled")
    )
    assert summary["parameter_selection"] == "stage_numerical_convergence"
    assert summary["final_loss"] == min(summary["losses"])
    assert summary["num_input_loaders"] == 1
    assert summary["num_cached_batches"] == 1
    assert summary["num_cached_samples"] == 2
    assert model.forward_calls == 1


def test_posthoc_cache_iterates_shared_scenario_loader_once_and_splits_sources():
    def _batch(*, source_index=None):
        batch = {
            "graph_batch": _CalibrationGraph(_evidence(2)),
            "labels": torch.tensor([0, 1]),
            "sids": ["a", "b"],
            "quality": {},
            "num_failed": 0,
        }
        if source_index is not None:
            batch["calibration_source_index"] = source_index
        return batch

    combined_loader = [_batch(source_index=0), _batch(source_index=1)]
    sources = [
        {
            "name": "clean",
            "scenario_group": "clean",
            "loader": [_batch()],
            "reliability_branches": ["api", "graph", "manifest"],
        },
        {
            "name": "api_dropout",
            "scenario_group": "api_event_dropout",
            "loader": combined_loader,
            "combined_source_index": 0,
            "reliability_branches": ["api"],
        },
        {
            "name": "api_missing",
            "scenario_group": "missing",
            "loader": combined_loader,
            "combined_source_index": 1,
            "reliability_branches": [],
        },
    ]
    model = _PosthocCalibrationModel()
    summary = fit_posthoc_calibration(
        model,
        sources,
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "stage_optimization": {
                    "default": {
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "convergence_tolerance": 1.0e-12,
                        "lr": 1.0e-3,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 1.0},
            },
        },
    )

    assert model.forward_calls == 3
    assert summary["num_input_loaders"] == 3
    assert summary["num_unique_input_loaders"] == 2
    assert summary["num_encoder_batches_cached"] == 3
    assert summary["num_cached_batches"] == 3
    assert [source["name"] for source in summary["calibration_sources"]] == [
        "clean",
        "api_dropout",
        "api_missing",
    ]
    assert [source["reliability_branches"] for source in summary["calibration_sources"]] == [
        ["api", "graph", "manifest"],
        ["api"],
        [],
    ]


def test_posthoc_calibration_fits_routed_decision_module_from_cached_logits():
    evidence = _evidence(2)
    evidence[:, EvidenceIndex.API_INTEGRITY] = torch.tensor([0.5, 1.0])
    evidence[:, EvidenceIndex.API_ENCODER_COVERAGE] = torch.tensor([0.5, 1.0])
    loader = [
        {
            "graph_batch": _CalibrationGraph(evidence),
            "labels": torch.tensor([0, 1]),
            "sids": ["a", "b"],
            "quality": {},
            "num_failed": 0,
        }
    ]
    model = _PosthocRoutedModel()
    before = [parameter.detach().clone() for parameter in model.discount_fusion.opinion_router.parameters()]
    summary = fit_posthoc_calibration(
        model,
        [loader],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "epochs": 2,
                "patience": 0,
                "lr": 1.0e-2,
            },
            "fusion": {
                "routing": {"enabled": True, "calibration_weight": 1.0},
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 0.0},
            },
        },
    )
    after = list(model.discount_fusion.opinion_router.parameters())
    assert "routing_visible_reference" not in summary
    assert summary["final_temperature"]["enabled"] is True
    assert summary["final_temperature"]["temperature"] > 0.0
    assert summary["final_temperature"]["nll_after"] <= (
        summary["final_temperature"]["nll_before"] + 1.0e-5
    )
    assert model.discount_fusion.calibration_active is True
    assert model.forward_calls == 1
    assert summary["strategy"] == "full_posthoc_fit_configured_group_reduction"
    assert summary["stages"]["reliability"]["objective_groups"] == [
        "api:clean",
        "graph:clean",
        "manifest:clean",
    ]
    assert summary["stages"]["routing_distribution"]["objective_groups"] == [
        "router:clean"
    ]
    assert summary["stages"]["routing_risk"]["objective_groups"] == ["risk:clean"]
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(before, after))


def test_nested_crossfit_uses_every_identity_and_restores_deployment_models():
    loader = [
        {
            "graph_batch": _CalibrationGraph(_evidence(2)),
            "labels": torch.tensor([0, 1]),
            "sids": [f"benign-{fold}", f"malware-{fold}"],
            "quality": {},
            "num_failed": 0,
        }
        for fold in range(3)
    ]
    model = _PosthocRoutedModel()
    summary = fit_posthoc_calibration(
        model,
        [loader],
        torch.device("cpu"),
        False,
        {
            "train": {"seed": 42},
            "calibration": {
                "enabled": True,
                "cross_fitting": {
                    "enabled": True,
                    "mode": "nested",
                    "num_folds": 3,
                },
                "stage_optimization": {
                    "default": {
                        "optimizer": "adam",
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "convergence_tolerance": 1.0e-8,
                        "gradient_tolerance": 1.0e-8,
                        "lr": 1.0e-2,
                        "require_convergence": False,
                    },
                    "reliability": {
                        "optimizer": "lbfgs",
                        "lr": 1.0,
                        "max_steps": 1,
                        "min_steps": 1,
                    },
                },
            },
            "fusion": {
                "routing": {"enabled": True, "calibration_weight": 1.0},
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 0.0},
            },
        },
    )

    assert summary["strategy"] == "identity_grouped_nested_crossfit_staged_refit"
    assert summary["cross_fitting"]["strictly_nested"] is True
    assert summary["cross_fitting"]["oof_reliability_coverage"] == 1.0
    assert summary["cross_fitting"]["oof_route_coverage"] == 1.0
    assert len(summary["cross_fitting"]["outer_folds"]) == 3
    assert summary["stage_clean_sample_counts"] == {
        "reliability": 6,
        "routing_distribution": 6,
        "routing_risk": 6,
        "final_temperature": 6,
    }
    assert len(summary["_oof_clean_rows"]) == 6
    # Every executed I1 fit contains an ordered competence phase followed by a
    # degradation phase. Derive the cross-fit budget from those phase records
    # so this test remains valid if either phase budget changes independently.
    expected_cross_fit_steps = 0
    for outer in summary["cross_fitting"]["outer_folds"]:
        for inner in outer["inner_reliability_fits"]:
            fit = inner["fit"]
            phase_steps = sum(
                phase["total_steps"] for phase in fit["phases"].values()
            )
            assert fit["total_steps"] == phase_steps
            expected_cross_fit_steps += int(
                fit.get("executed_total_steps", phase_steps)
            )
        holdout_fit = outer["holdout_reliability_fit"]
        holdout_phase_steps = sum(
            phase["total_steps"]
            for phase in holdout_fit["phases"].values()
        )
        assert holdout_fit["total_steps"] == holdout_phase_steps
        expected_cross_fit_steps += holdout_phase_steps
        expected_cross_fit_steps += outer["route_fit"]["total_steps"]
    assert summary["cross_fit_optimization_steps"] == expected_cross_fit_steps
    assert summary["cross_fitting"]["unique_inner_reliability_fits"] == 3
    assert summary["cross_fitting"]["reused_inner_reliability_fits"] == 3
    reused = [
        inner["fit"]
        for outer in summary["cross_fitting"]["outer_folds"]
        for inner in outer["inner_reliability_fits"]
        if inner["fit"].get("optimization_reused")
    ]
    assert len(reused) == 3
    assert all(fit["executed_total_steps"] == 0 for fit in reused)
    assert model.forward_calls == 3
    assert model.discount_fusion.calibration_active is True


def test_staged_calibration_keeps_reliability_clean_only_and_routes_degradation():
    evidence = _evidence(2)

    def _loader():
        return [
            {
                "graph_batch": _CalibrationGraph(evidence),
                "labels": torch.tensor([0, 1]),
                "sids": ["a", "b"],
                "quality": {},
                "num_failed": 0,
            }
        ]

    sources = [
        {
            "name": "clean",
            "scenario_group": "clean",
            "reliability_branches": ["api", "graph", "manifest"],
            "loader": _loader(),
        },
        {
            "name": "calibration_api_degraded_s0.9",
            "scenario_group": "api_degraded",
            "reliability_branches": ["api"],
            "loader": _loader(),
        },
        {
            "name": "calibration_api_degraded_s0.5",
            "scenario_group": "api_degraded",
            "reliability_branches": ["api"],
            "loader": _loader(),
        },
        {
            "name": "calibration_api_missing",
            "scenario_group": "missing",
            "reliability_branches": [],
            "loader": _loader(),
        },
    ]
    model = _PosthocRoutedModel()
    summary = fit_posthoc_calibration(
        model,
        sources,
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "epochs": 1,
                "patience": 0,
                "lr": 1.0e-2,
            },
            "fusion": {
                "routing": {"enabled": True, "calibration_weight": 1.0},
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 0.0},
            },
        },
    )

    assert model.forward_calls == 4
    assert summary["stages"]["reliability"]["objective_groups"] == [
        "api:observable",
        "graph:clean",
        "manifest:clean",
    ]
    assert summary["stages"]["routing_distribution"]["objective_groups"] == [
        "router:api_degraded",
        "router:missing",
    ]
    assert summary["stages"]["routing_risk"]["objective_groups"] == [
        "risk:api_degraded",
        "risk:missing",
    ]
    # I1 materializes one fixed design per ordered phase. The degradation
    # phase can only subtract a non-negative penalty from clean competence.
    reliability_stage = summary["stages"]["reliability"]
    assert reliability_stage["lifecycle"] == (
        "clean_competence_then_nonnegative_degradation_v1"
    )
    assert reliability_stage["degradation_never_increases_reliability"] is True
    assert reliability_stage["decision_forward_evaluations"] == 2
    assert reliability_stage["decision_forward_evaluations"] == sum(
        phase["decision_forward_evaluations"]
        for phase in reliability_stage["phases"].values()
    )
    # Fixed I2 router inputs and the risk features are each materialized once;
    # optimization thereafter executes only their lightweight heads.
    route_stage = summary["stages"]["routing_distribution"]
    assert route_stage["decision_forward_evaluations"] == 1
    assert route_stage["lightweight_forward_evaluations"] == route_stage[
        "function_evaluations"
    ]
    assert summary["stages"]["routing_risk"]["decision_forward_evaluations"] == 1
    missing_source = next(
        source
        for source in summary["calibration_sources"]
        if source["scenario_group"] == "missing"
    )
    assert missing_source["reliability_branches"] == []


def test_posthoc_route_executes_subset_oracle_and_soft_worst_group_objective():
    evidence = _evidence(2)

    def _loader():
        return [
            {
                "graph_batch": _CalibrationGraph(evidence),
                "labels": torch.tensor([0, 1]),
                # Every transformed source represents the same two packages.
                "sids": ["a", "b"],
                "quality": {},
                "num_failed": 0,
            }
        ]

    sources = [
        {
            "name": "clean",
            "scenario_group": "clean",
            "loader": _loader(),
        },
        {
            "name": "calibration_api_semantic_corrupted_s0.5",
            "scenario_group": "api_semantic_corrupted",
            "perturb_type": "api_semantic_corrupted",
            "objective_family": "single_semantic",
            "strength": 0.5,
            "reliability_branches": [],
            "loader": _loader(),
        },
        {
            "name": "calibration_api_graph_degraded_s0.5",
            "scenario_group": "api_graph_degraded",
            "perturb_type": "api_graph_degraded",
            "objective_family": "combined_completeness",
            "strength": 0.5,
            "reliability_branches": [],
            "loader": _loader(),
        },
        {
            "name": "calibration_api_missing",
            "scenario_group": "missing",
            "perturb_type": "api_missing",
            "objective_family": "missing",
            "strength": 1.0,
            "reliability_branches": [],
            "loader": _loader(),
        },
        {
            "name": "calibration_graph_missing",
            "scenario_group": "missing",
            "perturb_type": "graph_missing",
            "objective_family": "missing",
            "strength": 1.0,
            "reliability_branches": [],
            "loader": _loader(),
        },
    ]
    model = _PosthocRouterOnlyModel()
    summary = fit_posthoc_calibration(
        model,
        sources,
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "stage_optimization": {
                    "default": {
                        "optimizer": "adam",
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "convergence_tolerance": 1.0e-12,
                        "gradient_tolerance": 1.0e-12,
                        "lr": 1.0e-3,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "routing": {
                    "enabled": True,
                    "calibration_weight": 1.0,
                    "prediction_loss_weight": 1.0,
                    "route_oracle_loss_weight": 0.0,
                    "subset_oracle_loss_weight": 0.25,
                    "subset_oracle_temperature": 0.25,
                    "scenario_objective_weights": {
                        "clean": 0.5,
                        "perturb": 0.5,
                    },
                    "group_robust_objective": {
                        "enabled": True,
                        "taxonomy": "perturb_type_v1",
                        "soft_worst_weight": 0.25,
                        "temperature": 0.1,
                        "apply_to": ["routing_distribution"],
                    },
                },
                "reliability_calibration": {"weight": 0.0},
                "probability_calibration": {"weight": 0.0},
            },
        },
    )

    route = summary["stages"]["routing_distribution"]
    diagnostics = route["objective_diagnostics"]
    assert route["objective_groups"] == [
        "router:api_graph_degraded",
        "router:api_missing",
        "router:api_semantic_corrupted",
        "router:graph_missing",
    ]
    assert math.isfinite(route["initial_loss"])
    assert math.isfinite(route["final_loss"])
    assert route["decision_forward_evaluations"] == 1
    assert route["lightweight_forward_evaluations"] == route[
        "function_evaluations"
    ]
    assert diagnostics["group_robust_objective"]["enabled"] is True
    assert diagnostics["normalized_route_component_weights"] == pytest.approx(
        {
            "prediction": 0.8,
            "row_oracle": 0.0,
            "source_subset_oracle": 0.2,
        }
    )
    assert [item["name"] for item in diagnostics["resolved_families"]] == [
        "api_graph_degraded",
        "api_missing",
        "api_semantic_corrupted",
        "graph_missing",
    ]

    subset = diagnostics["subset_oracle"]
    assert subset["enabled"] is True
    assert len(subset["candidate_subsets"]) == 7
    assert [item["name"] for item in subset["sources"]] == [
        "clean",
        "calibration_api_graph_degraded_s0.5",
        "calibration_api_missing",
        "calibration_api_semantic_corrupted_s0.5",
        "calibration_graph_missing",
    ]
    assert sum(subset["hard_best_subset_counts"].values()) == 5
    assert all(
        sum(item["target_branch_mass"]) == pytest.approx(1.0)
        for item in subset["sources"]
    )
    assert all(item["num_eligible_subsets"] == 7 for item in subset["sources"])
    # Pairwise completeness views close only the route-coverage gap. They must
    # not silently redefine the already threshold-aligned I2 risk protocol.
    assert summary["stages"]["routing_risk"]["objective_groups"] == [
        "risk:api_semantic_corrupted",
        "risk:missing",
    ]


def test_nonrouted_i1_keeps_observable_views_but_probability_fit_is_clean_only():
    evidence = _evidence(2)

    def _loader():
        return [
            {
                "graph_batch": _CalibrationGraph(evidence),
                "labels": torch.tensor([0, 1]),
                "sids": ["a", "b"],
                "quality": {},
                "num_failed": 0,
            }
        ]

    model = _PosthocCalibrationModel()
    summary = fit_posthoc_calibration(
        model,
        [
            {
                "name": "clean",
                "scenario_group": "clean",
                "loader": _loader(),
            },
            {
                "name": "calibration_api_degraded_s0.9",
                "scenario_group": "api_degraded",
                "reliability_branches": ["api"],
                "loader": _loader(),
            },
        ],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "epochs": 1,
                "patience": 0,
                "lr": 1.0e-2,
            },
            "fusion": {
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 1.0},
            },
        },
    )

    assert model.forward_calls == 2
    assert set(summary["stages"]) == {"reliability", "probability"}
    assert summary["stages"]["reliability"]["objective_groups"] == [
        "api:observable",
        "graph:clean",
        "manifest:clean",
    ]
    assert summary["stages"]["probability"]["objective_groups"] == [
        "probability:clean"
    ]


def test_router_only_ablation_keeps_balanced_scenario_protocol():
    evidence = _evidence(2)

    def _loader():
        return [
            {
                "graph_batch": _CalibrationGraph(evidence),
                "labels": torch.tensor([0, 1]),
                "sids": ["a", "b"],
                "quality": {},
                "num_failed": 0,
            }
        ]

    model = _PosthocRouterOnlyModel()
    summary = fit_posthoc_calibration(
        model,
        [
            {
                "name": "clean",
                "scenario_group": "clean",
                "loader": _loader(),
            },
            {
                "name": "calibration_api_degraded_s0.9",
                "scenario_group": "api_degraded",
                "loader": _loader(),
            },
            {
                "name": "calibration_api_missing",
                "scenario_group": "missing",
                "reliability_branches": [],
                "loader": _loader(),
            },
        ],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "epochs": 1,
                "patience": 0,
                "lr": 1.0e-2,
            },
            "fusion": {
                "routing": {
                    "enabled": True,
                    "calibration_weight": 1.0,
                    "prediction_loss_weight": 1.0,
                    "risk_loss_weight": 1.0,
                },
                "reliability_calibration": {"weight": 0.0},
                "probability_calibration": {"weight": 0.0},
            },
        },
    )

    assert model.forward_calls == 3
    assert set(summary["stages"]) == {"routing_distribution", "routing_risk"}
    assert summary["stages"]["routing_distribution"]["objective_groups"] == [
        "router:api_degraded",
        "router:missing",
    ]
    assert summary["stages"]["routing_risk"]["objective_groups"] == [
        "risk:api_degraded",
        "risk:missing",
    ]



def test_reliability_and_conflict_switches_remove_their_decision_signals():
    fusion = DiscountProbabilityFusion(
        {
            "use_reliability_discount": False,
            "use_support_discount": False,
            "use_conflict_discount": False,
            "use_confidence_proxy": False,
            "use_hard_alive_mask": False,
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )
    evidence = _evidence(2)
    evidence[0, EvidenceIndex.API_INTEGRITY] = 0.0
    evidence[0, EvidenceIndex.GRAPH_INTEGRITY] = 0.1
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 1.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 1.0

    outputs = fusion(*_logits(batch_size=2), evidence)

    assert torch.allclose(
        outputs["fusion_weights"], torch.full((2, 3), 1.0 / 3.0)
    )
    assert torch.equal(outputs["effective_conflict"], torch.zeros(2))

def test_all_posthoc_parameters_are_inactive_during_main_training():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {"enabled": True},
            "detach_discount": True,
            "detach_confidence_proxy": True,
            "reliability_calibration": {"enabled": True},
            "probability_calibration": {"enabled": False},
        }
    )
    evidence = _evidence(2)
    logits = tuple(torch.randn(2, 2, requires_grad=True) for _ in range(3))
    outputs = fusion(*logits, evidence)
    torch.nn.functional.nll_loss(outputs["final_logits"], torch.tensor([0, 1])).backward()
    assert outputs["calibration_active"].sum().item() == 0.0
    assert all(
        parameter.grad is None
        for parameter in fusion.encoder_training_frozen_parameters()
    )


def test_validation_split_is_deterministic_and_group_isolated():
    class Dataset:
        sample_sids = ["a", "b", "c", "d", "e"]
        sample_groups = ["same", "same", "g2", "g3", "g4"]
        sample_labels = [0, 0, 1, 0, 1]

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    cfg = {"train": {"seed": 42}, "calibration": {"validation_fraction": 0.4}}
    selection_a, calibration_a, meta_a = split_validation_dataset(cfg, Dataset())
    selection_b, calibration_b, meta_b = split_validation_dataset(cfg, Dataset())
    assert selection_a.indices == selection_b.indices
    assert calibration_a.indices == calibration_b.indices
    assert meta_a == meta_b
    selection_groups = {Dataset.sample_groups[index] for index in selection_a.indices}
    calibration_groups = {Dataset.sample_groups[index] for index in calibration_a.indices}
    assert selection_groups.isdisjoint(calibration_groups)



def test_validation_split_is_label_stratified_for_singleton_groups():
    class Dataset:
        sample_sids = [f"s{index}" for index in range(20)]
        sample_groups = [f"g{index}" for index in range(20)]
        sample_labels = [0] * 10 + [1] * 10

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    cfg = {"train": {"seed": 42}, "calibration": {"validation_fraction": 0.5}}
    selection, calibration, meta = split_validation_dataset(cfg, Dataset())

    assert len(selection) == 10
    assert len(calibration) == 10
    assert meta["selection_label_counts"] == {0: 5, 1: 5}
    assert meta["calibration_label_counts"] == {0: 5, 1: 5}


def test_three_way_calibration_split_is_year_label_stratified():
    sids = [
        f"sample-{year}-{label}-{index}"
        for year in range(2018, 2025)
        for label in (0, 1)
        for index in range(8)
    ]

    class Dataset:
        sample_sids = sids
        sample_groups = [f"group-{sid}" for sid in sids]
        sample_years = [
            year
            for year in range(2018, 2025)
            for _label in (0, 1)
            for _index in range(8)
        ]
        sample_labels = [
            label
            for _year in range(2018, 2025)
            for label in (0, 1)
            for _index in range(8)
        ]

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    dataset = Dataset()
    cfg = {
        "train": {"seed": 42},
        "calibration": {
            "validation_fraction": 0.5,
            "conformal_fraction": 0.5,
            "split_seed": 42,
        },
    }
    selection, holdout, selection_meta = split_validation_dataset(cfg, dataset)
    posthoc, conformal, calibration_meta = split_posthoc_conformal_dataset(
        cfg, dataset, list(holdout.indices)
    )

    assert len(selection) == 56
    assert len(posthoc) == 28
    assert len(conformal) == 28
    assert all(
        count == 4 for count in selection_meta["selection_year_label_counts"].values()
    )
    assert all(
        count == 2 for count in calibration_meta["posthoc_year_label_counts"].values()
    )
    assert all(
        count == 2 for count in calibration_meta["conformal_year_label_counts"].values()
    )


def test_posthoc_and_conformal_calibration_subsets_are_disjoint_and_stratified():
    class Dataset:
        sample_sids = [f"sample-{index}" for index in range(40)]
        sample_groups = [f"group-{index}" for index in range(40)]
        sample_labels = [index % 2 for index in range(40)]

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    dataset = Dataset()
    cfg = {
        "train": {"seed": 42},
        "calibration": {
            "validation_fraction": 0.5,
            "conformal_fraction": 0.5,
            "split_seed": 42,
        },
    }
    selection, holdout, _ = split_validation_dataset(cfg, dataset)
    posthoc, conformal, meta = split_posthoc_conformal_dataset(
        cfg, dataset, list(holdout.indices)
    )

    selection_indices = set(selection.indices)
    posthoc_indices = set(posthoc.indices)
    conformal_indices = set(conformal.indices)
    assert selection_indices.isdisjoint(posthoc_indices)
    assert selection_indices.isdisjoint(conformal_indices)
    assert posthoc_indices.isdisjoint(conformal_indices)
    assert selection_indices | posthoc_indices | conformal_indices == set(range(40))
    assert meta["posthoc_label_counts"] == {0: 5, 1: 5}
    assert meta["conformal_label_counts"] == {0: 5, 1: 5}


def test_discount_probability_fusion_uses_fp32_for_half_inputs():
    fusion = DiscountProbabilityFusion(
        {
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )
    logits = tuple(value.half() for value in _logits(batch_size=2))
    outputs = fusion(*logits, _evidence(batch_size=2).half())

    assert outputs["final_logits"].dtype == torch.float32
    assert outputs["acceptance_score"].dtype == torch.float32
    assert outputs["fusion_weights"].dtype == torch.float32


def test_graph_encoder_budget_refreshes_alignment_and_effective_integrity():
    api_types = torch.tensor([1, 2], dtype=torch.long)
    data = {
        "x": torch.ones((5, 3)),
        "edge_index": torch.tensor([[0, 1, 3, 4], [1, 2, 4, 3]], dtype=torch.long),
        "sensitive_mask": torch.zeros(5, dtype=torch.uint8),
        "real_node_mask": torch.ones(5, dtype=torch.bool),
        "mask": torch.empty((5, 0)),
        "method_api_edge_index": torch.tensor([[0, 4], [0, 1]], dtype=torch.long),
        "api_ids": torch.tensor([10, 11], dtype=torch.long),
        "api_type_ids": api_types,
        "api_method_index": torch.tensor([0, 4], dtype=torch.long),
        "api_in_graph_mask": torch.ones(2),
        "api_semantic_category_counts": torch.ones(12),
        "graph_semantic_category_counts": torch.ones(12),
        "manifest_category_counts": torch.ones(12),
        "api_integrity": 1.0,
        "graph_integrity": 1.0,
        "q_graph": 1.0,
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "manifest_parse_ok": True,
        "api_event_count_raw": 2,
        "api_event_count_kept": 2,
        "api_known_type_count": 2,
        "api_unknown_type_count": 0,
        "graph_edge_count_reference": 4,
        "manifest_permission_ids": torch.tensor([1]),
        "manifest_intent_ids": torch.tensor([1]),
        "manifest_stats": torch.ones(11),
    }

    out = apply_graph_encoder_budget(data, 3, "alignment")

    assert out["x"].size(0) == 3
    assert out["graph_encoder_coverage"] == pytest.approx(0.6)
    assert out["graph_truncated_by_encoder_budget"] == 1.0
    assert out["method_api_edge_index"].tolist() == [[0], [0]]
    assert out["api_method_index"].tolist() == [0, -1]
    assert out["api_in_graph_mask"].tolist() == [1.0, 0.0]
    assert out["graph_encoder_coverage"] == pytest.approx(0.6)
    assert out["graph_truncated_by_encoder_budget"] == 1.0

    # Parse integrity remains pre-budget; visibility is represented separately.
    assert out["graph_integrity"] == pytest.approx(
        out["graph_integrity_before_encoder_budget"]
    )
    assert out["q_graph"] == pytest.approx(out["graph_integrity"])

    # effective_graph_integrity is where coverage correction happens.
    assert out["effective_graph_integrity"] == pytest.approx(
        out["graph_integrity"] * out["graph_encoder_coverage"]
    )
    assert out["effective_graph_integrity"] <= out["graph_integrity"]


def test_graph_encoder_budget_prioritizes_sensitive_nodes_across_the_sample():
    data = {
        "x": torch.arange(15, dtype=torch.float32).view(5, 3) + 1.0,
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "sensitive_mask": torch.tensor([0, 0, 0, 0, 1], dtype=torch.uint8),
        "real_node_mask": torch.ones(5, dtype=torch.bool),
        "mask": torch.empty((5, 0)),
        "method_api_edge_index": torch.empty((2, 0), dtype=torch.long),
        "api_ids": torch.tensor([10], dtype=torch.long),
        "api_type_ids": torch.tensor([1], dtype=torch.long),
        "api_method_index": torch.tensor([4], dtype=torch.long),
        "api_in_graph_mask": torch.ones(1),
        "api_semantic_category_counts": torch.ones(12),
        "graph_semantic_category_counts": torch.ones(12),
        "manifest_category_counts": torch.ones(12),
        "manifest_component_category_counts": torch.zeros(12),
        "manifest_x": torch.ones(16),
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "manifest_parse_ok": True,
        "api_event_count_raw": 1,
        "api_event_count_kept": 1,
        "api_known_type_count": 1,
        "api_unknown_type_count": 0,
        "manifest_permission_ids": torch.tensor([1]),
        "manifest_intent_ids": torch.tensor([1]),
        "manifest_stats": torch.ones(11),
    }
    refresh_observable_signals(data)
    pre_budget_integrity = data["graph_integrity"]

    out = apply_graph_encoder_budget(data, 3, "alignment")

    assert out["x"].tolist() == (torch.arange(15).view(5, 3)[[0, 1, 4]] + 1).tolist()
    assert out["sensitive_mask"].tolist() == [0, 0, 1]
    assert out["api_method_index"].tolist() == [2]
    assert out["graph_integrity"] == pytest.approx(pre_budget_integrity)


def test_graph_encoder_budget_never_lets_multidex_ghosts_displace_real_nodes():
    dataset = RobustTriModalDataset.__new__(RobustTriModalDataset)
    dataset.feature_dim = 2
    dataset.drop_graph_behavior_hints = False
    dataset.max_api_events_per_sample = None
    dataset.graph_semantic_source = "alignment"
    empty_api = {
        "api_ids": torch.empty((0,), dtype=torch.long),
        "api_type_ids": torch.empty((0,), dtype=torch.long),
        "api_sensitive_mask": torch.empty((0,), dtype=torch.float32),
        "api_method_index": torch.empty((0,), dtype=torch.long),
        "api_in_graph_mask": torch.empty((0,), dtype=torch.float32),
        "method_api_edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    data = dataset._aggregate_api_graph(
        [
            {
                "call_x": torch.empty((0, 2), dtype=torch.float32),
                "call_edge_index": torch.empty((2, 0), dtype=torch.long),
                "call_sensitive_mask": torch.empty((0,), dtype=torch.uint8),
                **empty_api,
            },
            {
                "call_x": torch.tensor(
                    [[10.0, 1.0], [20.0, 1.0], [30.0, 1.0]]
                ),
                "call_edge_index": torch.tensor(
                    [[0, 1], [1, 2]], dtype=torch.long
                ),
                "call_sensitive_mask": torch.tensor(
                    [0, 1, 0], dtype=torch.uint8
                ),
                **empty_api,
            },
        ]
    )

    assert data is not None
    assert data["real_node_mask"].tolist() == [False, True, True, True]

    out = apply_graph_encoder_budget(data, 2, "alignment")

    # Select the late sensitive real node and the first non-sensitive real
    # node, then restore their original order. The leading empty-DEX ghost is
    # retained only when every real node already fits in the budget.
    assert out["x"].tolist() == [[10.0, 1.0], [20.0, 1.0]]
    assert out["real_node_mask"].tolist() == [True, True]
    assert out["sensitive_mask"].tolist() == [0, 1]
    assert out["real_num_nodes"] == 2
    assert out["graph_alive"] == 1.0

def test_graph_encoder_budget_diagnostics_survive_collate_and_evidence_building():
    data = Data(
        x=torch.ones((3, 3), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor(1, dtype=torch.long),
    )

    data.sid = "sample-graph-budget"
    data.year = torch.tensor(2024, dtype=torch.long)
    data.sensitive_mask = torch.zeros((3,), dtype=torch.uint8)

    data.api_ids = torch.tensor([10, 11], dtype=torch.long)
    data.api_type_ids = torch.tensor([1, 2], dtype=torch.long)
    data.api_sensitive_mask = torch.zeros((2,), dtype=torch.float32)
    data.api_method_index = torch.tensor([0, 1], dtype=torch.long)
    data.api_in_graph_mask = torch.ones((2,), dtype=torch.float32)
    data.method_api_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    data.api_semantic_category_counts = torch.ones((12,), dtype=torch.float32)
    data.graph_semantic_category_counts = torch.ones((12,), dtype=torch.float32)
    data.api_category_counts = data.api_semantic_category_counts
    data.graph_category_counts = data.graph_semantic_category_counts

    data.manifest_x = torch.zeros((1, 256), dtype=torch.float32)
    data.manifest_permission_ids = torch.tensor([1], dtype=torch.long)
    data.manifest_intent_ids = torch.tensor([1], dtype=torch.long)
    data.manifest_category_counts = torch.ones((12,), dtype=torch.float32)
    data.manifest_stats = torch.ones((11,), dtype=torch.float32)

    data.q_api = torch.tensor([0.9], dtype=torch.float32)
    data.q_graph = torch.tensor([0.8], dtype=torch.float32)
    data.q_manifest = torch.tensor([0.95], dtype=torch.float32)
    data.q_align = torch.tensor([0.7], dtype=torch.float32)
    data.pert_api = torch.tensor([0.0], dtype=torch.float32)
    data.pert_graph = torch.tensor([0.0], dtype=torch.float32)
    data.pert_manifest = torch.tensor([0.0], dtype=torch.float32)

    # Populate all observable fields required by robust_collate_fn().
    for key in OBSERVABLE_NUMERIC_FIELDS:
        setattr(data, key, torch.tensor([0.0], dtype=torch.float32))
    for key in OBSERVABLE_SIGNAL_FIELDS:
        setattr(data, key, torch.tensor([1.0], dtype=torch.float32))

    data.api_integrity = torch.tensor([0.9], dtype=torch.float32)
    data.graph_integrity = torch.tensor([0.8], dtype=torch.float32)
    data.manifest_integrity = torch.tensor([0.95], dtype=torch.float32)
    data.code_integrity = torch.tensor([0.85], dtype=torch.float32)
    data.api_alive = torch.tensor([1.0], dtype=torch.float32)
    data.graph_alive = torch.tensor([1.0], dtype=torch.float32)
    data.manifest_alive = torch.tensor([1.0], dtype=torch.float32)

    # The three fields that were previously dropped during collate.
    data.graph_encoder_coverage = torch.tensor([0.5], dtype=torch.float32)
    data.graph_truncated_by_encoder_budget = torch.tensor([1.0], dtype=torch.float32)
    data.graph_integrity_before_encoder_budget = torch.tensor([0.8], dtype=torch.float32)

    batch = robust_collate_fn([data])
    graph_batch = batch["graph_batch"]

    assert hasattr(graph_batch, "graph_encoder_coverage")
    assert hasattr(graph_batch, "graph_truncated_by_encoder_budget")
    assert hasattr(graph_batch, "graph_integrity_before_encoder_budget")

    assert graph_batch.graph_encoder_coverage.view(-1).item() == pytest.approx(0.5)
    assert graph_batch.graph_truncated_by_encoder_budget.view(-1).item() == pytest.approx(1.0)
    assert graph_batch.graph_integrity_before_encoder_budget.view(-1).item() == pytest.approx(0.8)

    api_logits, graph_logits, manifest_logits = _logits(batch_size=1)
    _, diagnostics = build_evidence(
        graph_batch,
        api_logits,
        graph_logits,
        manifest_logits,
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        use_consistency_evidence=False,
        use_conflict_evidence=False,
        diagnostics_only=True,
    )

    assert diagnostics["graph_encoder_coverage"].item() == pytest.approx(0.5)
    assert diagnostics["graph_truncated_by_encoder_budget"].item() == pytest.approx(1.0)
    assert diagnostics["graph_integrity_before_encoder_budget"].item() == pytest.approx(0.8)
    assert diagnostics["effective_graph_integrity"].item() == pytest.approx(0.4)

def test_selective_metrics_and_validation_threshold():
    rows = [
        {"acceptance_score": 0.9},
        {"acceptance_score": 0.8},
        {"acceptance_score": 0.2},
        {"acceptance_score": 0.1},
    ]
    threshold = fit_rejection_threshold(
        rows, {"enabled": True, "target_coverage": 0.5}
    )
    assert threshold == pytest.approx(0.8)
    metrics = _selective_metrics(
        labels=[0, 1, 0, 1],
        preds=[0, 1, 1, 0],
        acceptance_scores=[0.9, 0.8, 0.2, 0.1],
        threshold=threshold,
    )
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["selective_risk"] == pytest.approx(0.0)
    assert metrics["selective_metrics_defined"] is True


def test_selective_metrics_are_undefined_when_every_sample_is_rejected():
    metrics = _selective_metrics(
        labels=[0, 1],
        preds=[0, 0],
        acceptance_scores=[0.1, 0.2],
        threshold=0.9,
    )

    assert metrics["coverage"] == pytest.approx(0.0)
    assert metrics["selective_metrics_defined"] is False
    assert metrics["selective_risk"] is None
    assert metrics["selective_acc"] is None
    assert metrics["selective_macro_f1"] is None
    assert metrics["aurc"] >= 0.0


def test_threshold_free_selective_metrics_report_only_ranking_quality():
    metrics = _selective_ranking_metrics(
        labels=[0, 1],
        preds=[0, 0],
        acceptance_scores=[0.9, 0.1],
    )

    assert metrics["aurc"] >= 0.0
    assert "coverage" not in metrics
    assert "selective_risk" not in metrics


def test_branch_prediction_row_records_per_branch_correctness():
    extra = {
        "api_logits_aux": torch.tensor([[-2.0, 2.0], [3.0, -3.0]]),
        "graph_logits_aux": torch.tensor([[3.0, -3.0], [-2.0, 2.0]]),
        "manifest_logits_aux": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }
    labels = torch.tensor([1, 0])

    first = _branch_prediction_row(extra, labels, 0)
    second = _branch_prediction_row(extra, labels, 1)

    assert first["api_pred"] == 1
    assert first["api_correct"] == 1
    assert first["graph_pred"] == 0
    assert first["graph_correct"] == 0
    assert first["api_prob"] > 0.9
    assert first["api_confidence"] > 0.9
    assert second["manifest_pred"] == 0
    assert second["manifest_correct"] == 1


def test_branch_reliability_metrics_compare_reliability_to_branch_correctness():
    rows = [
        {"api_correct": 1, "predicted_reliability_api": 0.9},
        {"api_correct": 1, "predicted_reliability_api": 0.8},
        {"api_correct": 0, "predicted_reliability_api": 0.2},
        {"api_correct": 0, "predicted_reliability_api": 0.1},
        {
            "api_alive": 0,
            "api_correct": 1,
            "predicted_reliability_api": 0.0,
        },
        {"graph_correct": 1, "predicted_reliability_graph": 0.4},
    ]

    metrics = compute_branch_reliability_metrics(rows)

    assert metrics["api_reliability_count"] == 4
    assert metrics["api_reliability_auc_defined"] == 1
    assert metrics["api_reliability_ap_defined"] == 1
    assert metrics["api_reliability_auc"] == pytest.approx(1.0)
    assert metrics["api_reliability_brier"] == pytest.approx(0.025)
    assert metrics["api_branch_accuracy"] == pytest.approx(0.5)
    assert metrics["api_reliability_mean"] == pytest.approx(0.5)
    assert metrics["graph_reliability_count"] == 1
    assert metrics["graph_reliability_auc_defined"] == 0
    assert metrics["graph_reliability_ap_defined"] == 0
    assert math.isnan(metrics["graph_reliability_auc"])
    assert math.isnan(metrics["graph_reliability_ap"])
    assert "joint_reliability_count" not in metrics


def test_metrics_json_replaces_nonfinite_values_with_null(tmp_path):
    path = tmp_path / "metrics.json"
    _write_metrics_json(
        path,
        {"auc": float("nan"), "nested": [float("inf"), 1.0]},
    )

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {"auc": None, "nested": [None, 1.0]}
