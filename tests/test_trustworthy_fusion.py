import json
import math

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from fusion.constants import AvailabilityIndex
from fusion.dataset import (
    RobustTriModalDataset,
    apply_graph_encoder_budget,
    robust_collate_fn,
)
from fusion.evidence import build_fusion_availability_and_diagnostics
from fusion.quality import refresh_hard_availability
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import (
    routing_mixture_log_prob,
    routing_risk_per_sample_loss,
    routing_risk_target,
)
from fusion.reliability_calibration import (
    MonotonicReliabilityCalibrator,
    build_reliability_features,
)
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


def _availability(batch_size: int = 1) -> torch.Tensor:
    return torch.ones(batch_size, AvailabilityIndex.BASE_DIM)


def _logits(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor([[2.0, -2.0]] * batch_size) for _ in range(3))


def _branch_probabilities(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "api": torch.tensor([[0.8, 0.2]]).repeat(batch_size, 1),
        "graph": torch.tensor([[0.1, 0.9]]).repeat(batch_size, 1),
        "manifest": torch.tensor([[0.7, 0.3]]).repeat(batch_size, 1),
    }


def _branch_alpha() -> dict[str, torch.Tensor]:
    return {
        "api": torch.tensor([[5.0, 1.0], [1.0, 3.0]]),
        "graph": torch.tensor([[2.0, 4.0], [3.0, 1.0]]),
        "manifest": torch.tensor([[1.0, 2.0], [6.0, 1.0]]),
    }


def _alive(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones(batch_size)
        for name in ("api", "graph", "manifest")
    }


def test_i1_uses_exactly_certainty_margin_and_predicted_class():
    features = build_reliability_features(_branch_alpha())

    # API row 0: S=6, p=(5/6,1/6), predicted benign.
    assert features["api"][0].tolist() == pytest.approx(
        [1.0 - 2.0 / 6.0, 4.0 / 6.0, 0.0]
    )
    # API row 1: S=4, p=(1/4,3/4), predicted malware.
    assert features["api"][1].tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_i1_monotonic_weights_increase_with_certainty_and_margin():
    calibrator = MonotonicReliabilityCalibrator(
        use_predicted_class_intercept=False,
    )
    low = {
        name: torch.tensor([[0.1, 0.2, 0.0]])
        for name in ("api", "graph", "manifest")
    }
    high = {
        name: torch.tensor([[0.8, 0.9, 0.0]])
        for name in ("api", "graph", "manifest")
    }
    alive = _alive(batch_size=1)

    low_out = calibrator(low, alive=alive)
    high_out = calibrator(high, alive=alive)
    for name in ("api", "graph", "manifest"):
        assert (
            high_out[f"predicted_reliability_{name}"]
            > low_out[f"predicted_reliability_{name}"]
        ).all()


def test_i1_is_branch_local_and_has_no_quality_or_perturbation_input():
    calibrator = MonotonicReliabilityCalibrator()
    features = build_reliability_features(_branch_alpha())
    changed = {name: value.clone() for name, value in features.items()}
    changed["graph"][:, :2] = 0.0
    changed["manifest"][:, :2] = 1.0

    first = calibrator(features, alive=_alive())
    second = calibrator(changed, alive=_alive())
    torch.testing.assert_close(
        first["predicted_reliability_api"],
        second["predicted_reliability_api"],
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(TypeError):
        calibrator(  # type: ignore[call-arg]
            features,
            alive=_alive(),
            quality_diagnostics={"api_integrity": torch.ones(2)},
        )


def test_i1_alive_is_a_mandatory_hard_mask():
    calibrator = MonotonicReliabilityCalibrator()
    features = build_reliability_features(_branch_alpha())
    alive = _alive()
    alive["api"][0] = 0.0
    outputs = calibrator(features, alive=alive)

    assert outputs["predicted_reliability_api"][0].item() == 0.0
    assert outputs["predicted_reliability_api"][1].item() > 0.0
    with pytest.raises(TypeError):
        calibrator(features)  # type: ignore[call-arg]


def test_route_distribution_override_rejects_non_routed_combination():
    fusion = DiscountProbabilityFusion({"combination": "dempster"})

    with pytest.raises(ValueError, match="only valid for the routed"):
        fusion(
            *_logits(),
            _availability(),
            branch_distribution_override=torch.full((1, 3), 1.0 / 3.0),
        )


def test_routing_risk_bce_uses_raw_logit_without_saturation():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "risk_mode": "learned",
                "risk_target": "threshold_malware_false_negative",
                "posthoc_refine": True,
            },
            "reliability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    assert fusion.opinion_router is not None
    fusion.opinion_router.set_risk_decision_threshold(0.0)
    with torch.no_grad():
        fusion.opinion_router.risk_bias.fill_(-20.0)
        fusion.opinion_router.raw_risk_feature_weights.fill_(-30.0)

    evidence = _availability()
    outputs = fusion(*_logits(), evidence)
    routing_cfg = {
        "risk_loss": "bce",
        "risk_target": "threshold_malware_false_negative",
        "classification_log_odds_threshold": 0.0,
    }
    labels = torch.tensor([1])
    target, valid, loss_type, _ = routing_risk_target(
        outputs,
        labels,
        routing_cfg,
    )
    per_row = routing_risk_per_sample_loss(
        outputs["routing_risk_probability"],
        outputs["routing_risk_training_logit"],
        target,
        valid,
        loss_type=loss_type,
    )
    loss = per_row[valid].mean()
    loss.backward()

    assert outputs["routing_risk_probability"].item() < 1.0e-7
    assert fusion.opinion_router.risk_bias.grad is not None
    assert fusion.opinion_router.risk_bias.grad.item() < -0.99


class _CalibrationGraph:
    def __init__(
        self,
        evidence: torch.Tensor,
        embeddings: dict[str, torch.Tensor] | None = None,
        degraded_branch: str | None = None,
    ):
        self.evidence = evidence
        self.embeddings = embeddings
        self.degraded_branch = degraded_branch

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
                "combination": "cumulative",
                "reliability_calibration": {
                    "enabled": True,
                },
            }
        )

    def calibration_parameters(self):
        return self.discount_fusion.calibration_parameters()

    def set_calibration_active(self, enabled: bool):
        self.discount_fusion.set_calibration_active(enabled)

    def forward(self, graph):
        self.forward_calls += 1
        batch_size = graph.evidence.size(0)
        clean_logits = graph.evidence.new_tensor(
            [[2.0, -2.0], [-2.0, 2.0]]
        )[:batch_size]
        branch_logits = tuple(
            -clean_logits if graph.degraded_branch == name else clean_logits
            for name in ("api", "graph", "manifest")
        )
        outputs = self.discount_fusion(*branch_logits, graph.evidence)
        for name, logits in zip(
            ("api", "graph", "manifest"),
            branch_logits,
        ):
            outputs[f"{name}_logits_aux"] = logits
        outputs["fusion_availability"] = graph.evidence
        # The production model derives this from the current encoder-visible
        # API sequence. This encoder-free fixture uses full observed support.
        outputs["api_observed_support"] = graph.evidence.new_ones(batch_size)
        outputs["selective_eligible"] = graph.evidence[:, :3].gt(0).any(dim=-1)
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
                    "posthoc_refine": True,
                    "prediction_loss_weight": 1.0,
                    "risk_mode": "learned",
                    "risk_target": "threshold_malware_false_negative",
                    "risk_loss_weight": 1.0,
                    "final_temperature_scaling": True,
                },
                "reliability_calibration": {
                    "enabled": True,
                    "branches": ["api", "graph", "manifest"],
                },
            }
        )
        assert self.discount_fusion.opinion_router is not None
        self.discount_fusion.opinion_router.set_risk_decision_threshold(5.0)


class _PosthocRouterOnlyModel(_PosthocCalibrationModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.forward_calls = 0
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "combination": "routed",
                "routing": {
                    "enabled": True,
                    "posthoc_refine": True,
                    "prediction_loss_weight": 1.0,
                    "risk_mode": "learned",
                    "risk_target": "threshold_malware_false_negative",
                    "risk_loss_weight": 1.0,
                    "final_temperature_scaling": False,
                },
                "reliability_calibration": {"enabled": False},
            }
        )
        assert self.discount_fusion.opinion_router is not None
        self.discount_fusion.opinion_router.set_risk_decision_threshold(5.0)


def test_removed_branch_probability_calibration_config_is_rejected():
    with pytest.raises(ValueError, match="probability_calibration was removed"):
        DiscountProbabilityFusion(
            {
                "combination": "cumulative",
                "probability_calibration": {"enabled": False},
            }
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
                "posthoc_refine": True,
                "prediction_loss_weight": 1.0,
                "risk_mode": "learned",
                "risk_loss_weight": 1.0,
            },
            "reliability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    labels = torch.tensor([0, 1, 1, 0])
    evidence = _availability(4)
    if availability == "mixed":
        evidence[:, AvailabilityIndex.API_ALIVE] = torch.tensor([1.0, 1.0, 0.0, 1.0])
        evidence[:, AvailabilityIndex.GRAPH_ALIVE] = torch.tensor([1.0, 0.0, 1.0, 1.0])
        evidence[:, AvailabilityIndex.MANIFEST_ALIVE] = torch.tensor(
            [0.0, 1.0, 1.0, 1.0]
        )
    elif availability == "all_dead":
        evidence[:, AvailabilityIndex.API_ALIVE] = 0.0
        evidence[:, AvailabilityIndex.GRAPH_ALIVE] = 0.0
        evidence[:, AvailabilityIndex.MANIFEST_ALIVE] = 0.0
    logits = (
        torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.2, 0.7], [1.0, -0.2]]),
        torch.tensor([[1.0, -0.3], [0.1, 0.8], [-0.4, 1.2], [0.5, 0.2]]),
        torch.tensor([[0.7, 0.1], [-0.5, 1.1], [0.3, 0.9], [1.4, -0.6]]),
    )
    route_parameters = fusion.routing_distribution_parameters()

    fusion.zero_grad(set_to_none=True)
    full_outputs = fusion(*logits, evidence)
    if availability == "all_dead":
        # The explicit 0.5/0.5 fallback uses the same conservative binary tie
        # policy as evaluate(): p(malware) >= 0.5 predicts malware.  I3 still
        # marks these rows ineligible and forces rejection.
        assert torch.equal(
            full_outputs["routing_mixture_pred"],
            torch.ones_like(full_outputs["routing_mixture_pred"]),
        )
    full_per_row = torch.nn.functional.nll_loss(
        routing_mixture_log_prob(full_outputs),
        labels,
        reduction="none",
    )
    full_valid = full_outputs["routing_has_available"].bool()
    full_loss = (
        full_per_row[full_valid].mean()
        if bool(full_valid.any())
        else full_per_row.sum() * 0.0
    )
    full_loss.backward()
    full_gradients = [parameter.grad.detach().clone() for parameter in route_parameters]

    fusion.zero_grad(set_to_none=True)
    with torch.no_grad():
        static = fusion(*logits, evidence)
    branches = ("api", "graph", "manifest")
    routed = fusion.opinion_router(
        branch_probabilities={
            name: static[f"routing_input_probability_{name}"]
            for name in branches
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
    }
    route_only_per_row = torch.nn.functional.nll_loss(
        routing_mixture_log_prob(route_only_outputs),
        labels,
        reduction="none",
    )
    route_only_valid = route_only_outputs["routing_has_available"].bool()
    route_only_loss = (
        route_only_per_row[route_only_valid].mean()
        if bool(route_only_valid.any())
        else route_only_per_row.sum() * 0.0
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
    evidence = _availability(2)
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
                "reliability_calibration": {
                    "scenario_objective_weights": {
                        "clean": 1.0,
                        "perturb": 0.0,
                    },
                },
            },
        },
    )

    assert summary["best_epoch"] == 5
    assert summary["epochs_ran"] == 5
    assert summary["loss_evaluations_ran"] == 6
    reliability_stage = summary["stages"]["reliability"]
    assert reliability_stage["total_steps"] == 5
    assert reliability_stage["max_steps"] == 5
    assert reliability_stage["objective_groups"] == [
        "api:clean",
        "graph:clean",
        "manifest:clean",
    ]
    diagnostics = reliability_stage["objective_diagnostics"]
    assert diagnostics["semantics"] == (
        "global_source_balanced_branch_correctness_bce"
    )
    assert diagnostics["scenario_objective_weights"] == {
        "clean": 1.0,
        "perturb": 0.0,
    }
    assert diagnostics["missing_sources_excluded"] is True
    assert set(diagnostics["branches"]) == {
        "api",
        "graph",
        "manifest",
    }
    assert all(
        branch["clean_source_count"] == 1
        and branch["perturb_source_count"] == 0
        for branch in diagnostics["branches"].values()
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
            "graph_batch": _CalibrationGraph(_availability(2)),
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
            "perturb_type": "api_event_dropout",
            "strength": 0.5,
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
                "reliability_calibration": {
                    "scenario_objective_weights": {
                        "clean": 1.0,
                        "perturb": 0.0,
                    },
                },
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


def test_nested_crossfit_uses_every_identity_and_restores_deployment_models():
    loader = [
            {
                "graph_batch": _CalibrationGraph(_availability(2)),
                "labels": torch.tensor([1, 0] if fold == 0 else [0, 1]),
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
                "routing": {
                    "enabled": True,
                    "risk_target": "threshold_malware_false_negative",
                    "classification_log_odds_threshold": 0.0,
                    "scenario_objective_weights": {
                        "clean": 1.0,
                        "perturb": 0.0,
                    },
                },
                "reliability_calibration": {
                    "scenario_objective_weights": {
                        "clean": 1.0,
                        "perturb": 0.0,
                    },
                },
            },
            "classification_threshold": {
                "enabled": True,
                "objective": "macro_f1",
                "selection_rule": "macro_f1_unconstrained_v1",
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
    risk_target = summary["routing_risk_target"]
    assert risk_target["classification_threshold_source"] == (
        "upstream_nested_oof_raw_score"
    )
    assert risk_target["risk_training_cutoff_mode"] == (
        "fold_excluded_upstream_oof_clean"
    )
    assert set(
        risk_target["risk_training_raw_log_odds_threshold_by_fold"]
    ) == {"0", "1", "2"}
    assert len(risk_target["risk_training_cutoff_provenance"]) == 3
    risk_objective = summary["stages"]["routing_risk"][
        "objective_diagnostics"
    ]
    assert risk_objective["classification_cutoff_training_mode"] == (
        "fold_excluded_upstream_oof_clean"
    )
    assert (
        risk_objective["classification_log_odds_threshold_by_fold"]
        == risk_target["risk_training_raw_log_odds_threshold_by_fold"]
    )
    router = model.discount_fusion.opinion_router
    assert router.risk_decision_threshold_active is True
    assert float(
        router._risk_decision_log_odds_threshold.detach().cpu().item()
    ) == pytest.approx(risk_target["raw_log_odds_threshold"])
    for provenance in risk_target["risk_training_cutoff_provenance"]:
        assert provenance["held_out_fold"] not in provenance["fit_folds"]
        assert provenance["num_fit_rows"] == 4
        assert provenance["num_held_out_rows"] == 2
    # Final I1 is a single joint clean/partial correctness-calibration fit.
    # Reused inner fits execute zero additional optimization steps.
    expected_cross_fit_steps = 0
    for outer in summary["cross_fitting"]["outer_folds"]:
        for inner in outer["inner_reliability_fits"]:
            fit = inner["fit"]
            expected_cross_fit_steps += int(
                fit.get("executed_total_steps", fit["total_steps"])
            )
        holdout_fit = outer["holdout_reliability_fit"]
        expected_cross_fit_steps += holdout_fit["total_steps"]
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


def test_staged_calibration_fits_i1_jointly_on_clean_and_branch_local_partial_views():
    clean_evidence = _availability(2)

    def _loader(evidence, labels=(0, 1), degraded_branch=None):
        return [
            {
                "graph_batch": _CalibrationGraph(
                    evidence, degraded_branch=degraded_branch
                ),
                "labels": torch.tensor(
                    [1, 0] if fold == 0 else labels
                ),
                "sids": [f"a-{fold}", f"b-{fold}"],
                "quality": {},
                "num_failed": 0,
            }
            for fold in range(3)
        ]

    sources = [
        {
            "name": "clean",
            "scenario_group": "clean",
            "reliability_branches": ["api", "graph", "manifest"],
            "loader": _loader(clean_evidence),
        },
        {
            "name": "calibration_api_event_dropout_s0.5",
            "scenario_group": "api_event_dropout",
            "perturb_type": "api_event_dropout",
            "strength": 0.5,
            "reliability_branches": ["api"],
            "loader": _loader(clean_evidence, degraded_branch="api"),
        },
        {
            "name": "calibration_graph_sparsify_s0.5",
            "scenario_group": "graph_sparsify",
            "perturb_type": "graph_sparsify",
            "strength": 0.5,
            "reliability_branches": ["graph"],
            "loader": _loader(clean_evidence, degraded_branch="graph"),
        },
        {
            "name": "calibration_manifest_permission_mask_s0.5",
            "scenario_group": "manifest_permission_mask",
            "perturb_type": "manifest_permission_mask",
            "strength": 0.5,
            "reliability_branches": ["manifest"],
            "loader": _loader(clean_evidence),
        },
        {
            "name": "calibration_api_missing",
            "scenario_group": "missing",
            "perturb_type": "api_missing",
            "strength": 1.0,
            "reliability_branches": [],
            "loader": _loader(clean_evidence),
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
                "cross_fitting": {
                    "enabled": True,
                    "mode": "nested",
                    "num_folds": 3,
                },
                "stage_optimization": {
                    "default": {
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "lr": 1.0e-2,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "routing": {
                    "enabled": True,
                    "risk_target": "threshold_malware_false_negative",
                    "classification_log_odds_threshold": 0.0,
                    "scenario_objective_weights": {
                        "clean": 0.5,
                        "perturb": 0.5,
                    },
                },
                "reliability_calibration": {
                    "scenario_objective_weights": {
                        "clean": 0.5,
                        "perturb": 0.5,
                    },
                },
            },
        },
    )

    assert model.forward_calls == 15
    assert summary["stages"]["reliability"]["objective_groups"] == [
        "api:clean_plus_branch_local_partial",
        "graph:clean_plus_branch_local_partial",
        "manifest:clean_plus_branch_local_partial",
    ]
    assert summary["stages"]["routing_distribution"]["objective_groups"] == [
        "router:api_event_dropout",
        "router:graph_sparsify",
        "router:manifest_permission_mask",
        "router:missing",
    ]
    assert summary["stages"]["routing_risk"]["objective_groups"] == [
        "risk:api_event_dropout",
        "risk:graph_sparsify",
        "risk:manifest_permission_mask",
        "risk:missing",
    ]
    reliability_stage = summary["stages"]["reliability"]
    diagnostics = reliability_stage["objective_diagnostics"]
    assert diagnostics["scenario_objective_weights"] == {
        "clean": 0.5,
        "perturb": 0.5,
    }
    expected_mechanism = {
        "api": "api_event_dropout",
        "graph": "graph_sparsify",
        "manifest": "manifest_permission_mask",
    }
    for branch, mechanism in expected_mechanism.items():
        branch_diag = diagnostics["branches"][branch]
        assert branch_diag["clean_source_count"] == 1
        assert branch_diag["perturb_source_count"] == 1
        assert set(branch_diag["perturbation_hierarchy"]) == {mechanism}
    assert reliability_stage["decision_forward_evaluations"] == 1
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


def test_nonrouted_i1_fit_can_be_clean_only():
    evidence = _availability(2)

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
        ],
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
                        "lr": 1.0e-2,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "reliability_calibration": {
                    "scenario_objective_weights": {
                        "clean": 1.0,
                        "perturb": 0.0,
                    },
                },
            },
        },
    )

    assert model.forward_calls == 1
    assert set(summary["stages"]) == {"reliability"}
    assert summary["stages"]["reliability"]["objective_groups"] == [
        "api:clean",
        "graph:clean",
        "manifest:clean",
    ]
def test_router_only_ablation_keeps_balanced_scenario_protocol():
    evidence = _availability(2)

    def _loader():
        return [
            {
                "graph_batch": _CalibrationGraph(evidence),
                "labels": torch.tensor([1, 0] if fold == 0 else [0, 1]),
                "sids": [f"a-{fold}", f"b-{fold}"],
                "quality": {},
                "num_failed": 0,
            }
            for fold in range(3)
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
                "name": "calibration_api_event_dropout_s0.5",
                "scenario_group": "api_event_dropout",
                "perturb_type": "api_event_dropout",
                "strength": 0.5,
                "reliability_branches": ["api"],
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
                "cross_fitting": {
                    "enabled": True,
                    "mode": "nested",
                    "num_folds": 3,
                },
                "stage_optimization": {
                    "default": {
                        "max_steps": 1,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "lr": 1.0e-2,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "routing": {
                    "enabled": True,
                    "prediction_loss_weight": 1.0,
                    "risk_loss_weight": 1.0,
                    "risk_target": "threshold_malware_false_negative",
                    "classification_log_odds_threshold": 0.0,
                    "scenario_objective_weights": {
                        "clean": 0.5,
                        "perturb": 0.5,
                    },
                },
            },
        },
    )

    assert model.forward_calls == 9
    assert set(summary["stages"]) == {"routing_distribution", "routing_risk"}
    assert summary["stages"]["routing_distribution"]["objective_groups"] == [
        "router:api_event_dropout",
        "router:missing",
    ]
    assert summary["stages"]["routing_risk"]["objective_groups"] == [
        "risk:api_event_dropout",
        "risk:missing",
    ]



def test_all_posthoc_parameters_are_inactive_during_main_training():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {"enabled": True},
            "reliability_calibration": {"enabled": True},
        }
    )
    evidence = _availability(2)
    logits = tuple(torch.randn(2, 2, requires_grad=True) for _ in range(3))
    outputs = fusion(*logits, evidence)
    torch.nn.functional.nll_loss(outputs["final_logits"], torch.tensor([0, 1])).backward()
    assert outputs["calibration_active"].sum().item() == 0.0
    assert all(
        parameter.grad is None
        for parameter in fusion.encoder_training_frozen_parameters()
    )


def test_routed_conflict_is_risk_only_and_never_changes_route_scores():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "risk_conflict_enabled": True,
            },
            "reliability_calibration": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    logits = (
        torch.tensor([[3.0, -3.0]]),
        torch.tensor([[3.0, -3.0]]),
        torch.tensor([[-3.0, 3.0]]),
    )
    outputs = fusion(*logits, _availability())

    # I1 is disabled, so the only valid route is uniform over the three alive
    # branches even though the third branch strongly disagrees.
    torch.testing.assert_close(
        outputs["routing_branch_distribution"],
        torch.full((1, 3), 1.0 / 3.0),
    )
    assert outputs["fusion_weights"].shape == (1, 3)
    torch.testing.assert_close(
        outputs["fusion_weights"].sum(dim=-1), torch.ones(1)
    )
    risk_conflict = outputs["routing_risk_global_cross_modal_conflict"]
    assert risk_conflict.shape == (1,)
    assert torch.isfinite(risk_conflict).all()
    assert 0.0 < risk_conflict.item() <= 1.0

    # Route-conflict coefficients, penalties, and diagnostics were deleted.
    for removed_key in (
        "routing_conflict_penalty_mean",
        "routing_route_conflict_feature_active",
        "routing_route_conflict_feature_configured",
    ):
        assert removed_key not in outputs
    for name in ("api", "graph", "manifest"):
        assert f"routing_conflict_penalty_{name}" not in outputs


def test_validation_split_is_deterministic_and_group_isolated():
    class Dataset:
        sample_sids = ["a", "b", "c", "d", "e"]
        sample_groups = ["same", "same", "g2", "g3", "g4"]
        sample_labels = [0, 0, 1, 0, 1]
        sample_years = [2020, 2020, 2021, 2021, 2021]

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
        sample_years = [2020] * 20

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


def test_formal_three_way_validation_budget_is_40_35_25_and_group_disjoint():
    sids = [
        f"sample-{year}-{label}-{index}"
        for year in range(2018, 2025)
        for label in (0, 1)
        for index in range(20)
    ]

    class Dataset:
        sample_sids = sids
        sample_groups = [f"group-{sid}" for sid in sids]
        sample_years = [
            year
            for year in range(2018, 2025)
            for _label in (0, 1)
            for _index in range(20)
        ]
        sample_labels = [
            label
            for _year in range(2018, 2025)
            for label in (0, 1)
            for _index in range(20)
        ]

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    dataset = Dataset()
    cfg = {
        "train": {"seed": 42},
        "calibration": {
            "validation_fraction": 0.60,
            "conformal_fraction": 5.0 / 12.0,
            "split_seed": 42,
            "stratified_group_split": True,
        },
    }
    selection, holdout, outer = split_validation_dataset(cfg, dataset)
    posthoc, decision, inner = split_posthoc_conformal_dataset(
        cfg, dataset, list(holdout.indices)
    )

    assert (len(selection), len(posthoc), len(decision)) == (112, 98, 70)
    assert outer["selection_fraction_of_validation"] == pytest.approx(0.40)
    assert inner["posthoc_fraction_of_validation"] == pytest.approx(0.35)
    assert inner["decision_fraction_of_validation"] == pytest.approx(0.25)
    assert all(
        count == 8 for count in outer["selection_year_label_counts"].values()
    )
    assert all(
        count == 7 for count in inner["posthoc_year_label_counts"].values()
    )
    assert all(
        count == 5 for count in inner["conformal_year_label_counts"].values()
    )
    partitions = [set(selection.indices), set(posthoc.indices), set(decision.indices)]
    assert all(
        partitions[left].isdisjoint(partitions[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    assert set.union(*partitions) == set(range(len(dataset)))


def test_three_way_validation_split_keeps_repeated_packages_in_one_partition():
    class Dataset:
        sample_sids = [f"sample-{index}" for index in range(72)]
        # Every package contributes two rows. The second split operates on a
        # remapped holdout view, so this specifically guards against losing the
        # original package grouping between the outer and inner split.
        sample_groups = [f"package-{index // 2}" for index in range(72)]
        sample_labels = [(index // 2) % 2 for index in range(72)]
        sample_years = [2020 + ((index // 2) % 3) for index in range(72)]

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    dataset = Dataset()
    cfg = {
        "train": {"seed": 42},
        "calibration": {
            "validation_fraction": 0.60,
            "conformal_fraction": 5.0 / 12.0,
            "split_seed": 42,
            "stratified_group_split": True,
        },
    }
    selection, holdout, _ = split_validation_dataset(cfg, dataset)
    posthoc, decision, _ = split_posthoc_conformal_dataset(
        cfg, dataset, list(holdout.indices)
    )

    partition_by_index = {}
    for name, subset in (
        ("selection", selection),
        ("posthoc", posthoc),
        ("decision", decision),
    ):
        for index in subset.indices:
            partition_by_index[int(index)] = name
    assert len(partition_by_index) == len(dataset)
    for group in set(dataset.sample_groups):
        group_partitions = {
            partition_by_index[index]
            for index, value in enumerate(dataset.sample_groups)
            if value == group
        }
        assert len(group_partitions) == 1


def test_validation_split_rejects_disabled_stratified_group_protocol():
    class Dataset:
        sample_sids = ["a", "b"]
        sample_groups = ["a", "b"]
        sample_labels = [0, 1]

        def __len__(self):
            return 2

    with pytest.raises(ValueError, match="stratified_group_split must be true"):
        split_validation_dataset(
            {
                "calibration": {
                    "validation_fraction": 0.5,
                    "stratified_group_split": False,
                }
            },
            Dataset(),
        )


def test_validation_split_rejects_missing_formal_group_or_year_metadata():
    class Dataset:
        sample_sids = ["a", "b", "c", "d"]
        sample_labels = [0, 1, 0, 1]

        def __len__(self):
            return len(self.sample_sids)

    with pytest.raises(
        ValueError,
        match="requires complete package-group and year-label metadata",
    ):
        split_validation_dataset(
            {
                "calibration": {
                    "validation_fraction": 0.5,
                    "stratified_group_split": True,
                }
            },
            Dataset(),
        )


def test_posthoc_and_conformal_calibration_subsets_are_disjoint_and_stratified():
    class Dataset:
        sample_sids = [f"sample-{index}" for index in range(40)]
        sample_groups = [f"group-{index}" for index in range(40)]
        sample_labels = [index % 2 for index in range(40)]
        sample_years = [2020 + index % 2 for index in range(40)]

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
            "combination": "dempster",
            "reliability_calibration": {"enabled": False},
        }
    )
    logits = tuple(value.half() for value in _logits(batch_size=2))
    outputs = fusion(*logits, _availability(batch_size=2).half())

    assert outputs["final_logits"].dtype == torch.float32
    assert outputs["acceptance_score"].dtype == torch.float32
    assert outputs["fusion_weights"].dtype == torch.float32


def test_graph_encoder_budget_truncates_only_graph_encoder_tensors():
    data = {
        "x": torch.ones((5, 3)),
        "edge_index": torch.tensor([[0, 1, 3, 4], [1, 2, 4, 3]], dtype=torch.long),
        "sensitive_mask": torch.zeros(5, dtype=torch.uint8),
        "real_node_mask": torch.ones(5, dtype=torch.bool),
        "dex_parse_ok": True,
        "graph_parse_ok": True,
    }

    out = apply_graph_encoder_budget(data, 3)

    assert out["x"].size(0) == 3
    assert out["edge_index"].tolist() == [[0, 1], [1, 2]]
    assert out["sensitive_mask"].numel() == 3
    assert out["real_node_mask"].numel() == 3
    assert out["graph_encoder_budget_max_nodes"] == 3
    assert out["graph_alive"] == 1.0
    assert "graph_encoder_coverage" not in out
    assert "graph_integrity" not in out


def test_graph_encoder_budget_prioritizes_sensitive_nodes_across_the_sample():
    data = {
        "x": torch.arange(15, dtype=torch.float32).view(5, 3) + 1.0,
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "sensitive_mask": torch.tensor([0, 0, 0, 0, 1], dtype=torch.uint8),
        "real_node_mask": torch.ones(5, dtype=torch.bool),
        "dex_parse_ok": True,
        "graph_parse_ok": True,
    }
    refresh_hard_availability(data)

    out = apply_graph_encoder_budget(data, 3)

    assert out["x"].tolist() == (torch.arange(15).view(5, 3)[[0, 1, 4]] + 1).tolist()
    assert out["sensitive_mask"].tolist() == [0, 0, 1]
    assert out["graph_alive"] == 1.0


def test_graph_encoder_budget_never_lets_multidex_ghosts_displace_real_nodes():
    dataset = RobustTriModalDataset.__new__(RobustTriModalDataset)
    dataset.feature_dim = 2
    dataset.drop_graph_behavior_hints = False
    dataset.max_api_events_per_sample = None
    empty_api = {
        "api_ids": torch.empty((0,), dtype=torch.long),
        "api_type_ids": torch.empty((0,), dtype=torch.long),
        "api_sensitive_mask": torch.empty((0,), dtype=torch.float32),
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

    out = apply_graph_encoder_budget(data, 2)

    # Select the late sensitive real node and the first non-sensitive real
    # node, then restore their original order. The leading empty-DEX ghost is
    # retained only when every real node already fits in the budget.
    assert out["x"].tolist() == [[10.0, 1.0], [20.0, 1.0]]
    assert out["real_node_mask"].tolist() == [True, True]
    assert out["sensitive_mask"].tolist() == [0, 1]
    assert out["graph_alive"] == 1.0

def test_old_runtime_quality_fields_are_dropped_during_collate():
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

    # Contradictory historical fields may still exist in an old in-memory
    # object, but the runtime batch must not transport them.
    data.q_api = torch.tensor([0.9], dtype=torch.float32)
    data.api_integrity = torch.tensor([0.9], dtype=torch.float32)
    data.graph_encoder_coverage = torch.tensor([0.5], dtype=torch.float32)
    data.api_alive = torch.tensor([1.0], dtype=torch.float32)
    data.graph_alive = torch.tensor([1.0], dtype=torch.float32)
    data.manifest_alive = torch.tensor([1.0], dtype=torch.float32)

    batch = robust_collate_fn([data])
    graph_batch = batch["graph_batch"]

    assert not hasattr(graph_batch, "q_api")
    assert not hasattr(graph_batch, "api_integrity")
    assert not hasattr(graph_batch, "graph_encoder_coverage")
    assert graph_batch.api_alive.item() == 1.0
    assert graph_batch.graph_alive.item() == 1.0
    assert graph_batch.manifest_alive.item() == 1.0

    api_logits, graph_logits, manifest_logits = _logits(batch_size=1)
    _, diagnostics = build_fusion_availability_and_diagnostics(
        graph_batch,
        api_logits,
        graph_logits,
        manifest_logits,
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        materialize_diagnostics=True,
    )

    assert set(diagnostics) == {"api_alive", "graph_alive", "manifest_alive"}

def test_selective_metrics_and_validation_threshold():
    rows = [
        {"acceptance_score": 0.9, "selective_eligible": 1},
        {"acceptance_score": 0.8, "selective_eligible": 1},
        {"acceptance_score": 0.2, "selective_eligible": 1},
        {"acceptance_score": 0.1, "selective_eligible": 1},
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
        "joint_logits_aux": torch.tensor([[-3.0, 3.0], [2.0, -2.0]]),
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
    assert first["joint_pred"] == 1
    assert first["joint_correct"] == 1
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


def test_branch_reliability_metrics_report_predicted_class_cells():
    rows = [
        {"api_pred": 0, "api_correct": 1, "predicted_reliability_api": 0.8},
        {"api_pred": 0, "api_correct": 0, "predicted_reliability_api": 0.4},
        {"api_pred": 1, "api_correct": 1, "predicted_reliability_api": 0.9},
        {"api_pred": 1, "api_correct": 0, "predicted_reliability_api": 0.3},
    ]

    metrics = compute_branch_reliability_metrics(rows)

    assert metrics["api_predicted_benign_reliability_count"] == 2
    assert metrics["api_predicted_malware_reliability_count"] == 2
    assert metrics["api_predicted_benign_reliability_branch_accuracy"] == 0.5
    assert metrics["api_predicted_malware_reliability_branch_accuracy"] == 0.5


def test_anchored_competence_metrics_use_tcp_and_include_joint():
    rows = [
        {
            "label": 0,
            "joint_alive": 1,
            "joint_prob": 0.1,
            "joint_correct": 1,
            "predicted_competence_joint": 0.9,
        },
        {
            "label": 1,
            "joint_alive": 1,
            "joint_prob": 0.8,
            "joint_correct": 1,
            "predicted_competence_joint": 0.8,
        },
        {
            "label": 1,
            "joint_alive": 1,
            "joint_prob": 0.4,
            "joint_correct": 0,
            "predicted_competence_joint": 0.3,
        },
        {
            "label": 0,
            "joint_alive": 0,
            "joint_prob": 0.9,
            "joint_correct": 0,
            "predicted_competence_joint": 0.1,
        },
    ]

    metrics = compute_branch_reliability_metrics(rows)

    assert metrics["joint_competence_count"] == 3
    assert metrics["joint_competence_tcp_mse"] == pytest.approx(0.01 / 3.0)
    assert metrics["joint_competence_tcp_mae"] == pytest.approx(0.1 / 3.0)
    assert metrics["joint_competence_correctness_auc_defined"] == 1
    assert metrics["joint_competence_correctness_ap_defined"] == 1
    assert metrics["joint_competence_correctness_auc"] == pytest.approx(1.0)
    assert "joint_competence_brier" not in metrics


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
