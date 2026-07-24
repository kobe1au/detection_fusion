from __future__ import annotations

import math
from pathlib import Path

import torch

from fusion.constants import AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.reliability_calibration import (
    BRANCH_NAMES,
    TEMPERATURE_SCALING_CONFIDENCE_METHOD,
    BranchTemperatureScalingConfidenceCalibrator,
)
from fusion.train import fit_posthoc_calibration, load_config_path


ROOT = Path("config/experiments/tri_modal_robust")


class _CalibrationGraph:
    def __init__(self, evidence: torch.Tensor):
        self.evidence = evidence

    def to(self, device, non_blocking=True):
        self.evidence = self.evidence.to(device)
        return self


class _TemperatureScalingPosthocModel(torch.nn.Module):
    fusion_mode = "discount_probability"

    def __init__(self):
        super().__init__()
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "combination": "cumulative",
                "use_i1_reliability": True,
                "use_hard_alive_mask": True,
                "reliability_calibration": {
                    "enabled": True,
                    "method": TEMPERATURE_SCALING_CONFIDENCE_METHOD,
                    "branches": ["api", "graph", "manifest"],
                },
                "routing": {"enabled": False},
            }
        )

    def calibration_parameters(self):
        return self.discount_fusion.calibration_parameters()

    def set_calibration_active(self, enabled: bool):
        self.discount_fusion.set_calibration_active(enabled)

    def forward(self, graph):
        batch_size = graph.evidence.size(0)
        api = graph.evidence.new_tensor(
            [[5.0, -5.0], [-5.0, 5.0]]
        )[:batch_size]
        branches = _branch_logits(api)
        outputs = self.discount_fusion(
            branches["api"],
            branches["graph"],
            branches["manifest"],
            graph.evidence,
        )
        for name, logits in branches.items():
            outputs[f"{name}_logits_aux"] = logits
        outputs["fusion_availability"] = graph.evidence
        outputs["selective_eligible"] = graph.evidence[:, :3].gt(0).any(dim=-1)
        return outputs["final_logits"], outputs


def _evidence(batch_size: int, *, manifest_alive: bool = True) -> torch.Tensor:
    availability = torch.ones(batch_size, AvailabilityIndex.BASE_DIM)
    availability[:, AvailabilityIndex.MANIFEST_ALIVE] = float(manifest_alive)
    return availability


def _branch_logits(api_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "api": api_logits,
        "graph": api_logits.flip(dims=(-1,)),
        "manifest": api_logits * 0.5,
    }


def test_temperature_confidence_is_exact_max_scaled_softmax_and_masks_dead_branch():
    logits = torch.tensor([[3.0, -1.0], [-0.5, 1.5]])
    calibrator = BranchTemperatureScalingConfidenceCalibrator()
    with torch.no_grad():
        for branch in BRANCH_NAMES:
            calibrator.log_temperatures[branch].fill_(math.log(2.0))

    outputs = calibrator(
        _branch_logits(logits),
        alive={
            "api": torch.ones(2),
            "graph": torch.ones(2),
            "manifest": torch.zeros(2),
        },
    )

    expected = torch.softmax(logits / 2.0, dim=-1).amax(dim=-1)
    assert torch.allclose(outputs["predicted_reliability_api"], expected)
    assert torch.equal(
        outputs["predicted_reliability_manifest"], torch.zeros(2)
    )
    assert torch.allclose(
        outputs["reliability_temperature_api"], torch.full((2,), 2.0)
    )
    assert torch.equal(
        outputs["temperature_scaling_confidence_baseline_active"],
        torch.ones(2),
    )


def test_temperature_calibrator_state_dict_round_trip_is_strict_and_exact():
    source = BranchTemperatureScalingConfidenceCalibrator()
    with torch.no_grad():
        for index, branch in enumerate(BRANCH_NAMES, start=1):
            source.log_temperatures[branch].fill_(math.log(float(index + 1)))

    restored = BranchTemperatureScalingConfidenceCalibrator()
    incompatible = restored.load_state_dict(source.state_dict(), strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for branch in BRANCH_NAMES:
        assert torch.equal(
            source.log_temperatures[branch],
            restored.log_temperatures[branch],
        )


def test_branch_temperature_nll_fits_one_positive_scalar_and_improves_nll():
    # Deliberately overconfident logits with two wrong labels require T > 1.
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
        calibrator.branch_nll("api", logits, labels, alive).detach().item()
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
        calibrator.branch_nll("api", logits, labels, alive).detach().item()
    )
    assert after < before
    assert float(calibrator.temperature("api").detach().item()) > 1.0
    assert float(calibrator.temperature("api").detach().item()) > 0.0


def test_discount_fusion_routes_raw_logit_temperature_confidence_into_i1():
    fusion = DiscountProbabilityFusion(
        {
            "combination": "cumulative",
            "use_i1_reliability": True,
            "use_hard_alive_mask": True,
            "reliability_calibration": {
                "enabled": True,
                "method": TEMPERATURE_SCALING_CONFIDENCE_METHOD,
                "branches": ["api", "graph", "manifest"],
            },
            "routing": {"enabled": False},
        }
    )
    fusion.set_calibration_active(True)
    api_logits = torch.tensor([[2.0, -1.0], [0.5, 1.0]])
    branch_logits = _branch_logits(api_logits)

    outputs = fusion(
        branch_logits["api"],
        branch_logits["graph"],
        branch_logits["manifest"],
        _evidence(2),
    )

    expected = torch.softmax(api_logits, dim=-1).amax(dim=-1)
    assert torch.allclose(outputs["predicted_reliability_api"], expected)
    assert torch.equal(
        outputs["temperature_scaling_confidence_baseline_active"],
        torch.ones(2),
    )


def test_temperature_baseline_configs_separate_matched_and_clean_fit_sources():
    matched = load_config_path(
        ROOT / "ablations/i1/temperature_scaling_confidence.yaml"
    )
    clean = load_config_path(
        ROOT / "ablations/i1/temperature_scaling_confidence_clean_only.yaml"
    )

    for cfg in (matched, clean):
        reliability = cfg["fusion"]["reliability_calibration"]
        assert reliability["method"] == TEMPERATURE_SCALING_CONFIDENCE_METHOD
        assert reliability["branches"] == ["api", "graph", "manifest"]
        assert "use_model_visibility" not in reliability
        assert "use_predicted_class_feature" not in reliability
        assert cfg["calibration"]["cross_fitting"]["enabled"] is True
        assert cfg["fusion"]["routing"]["risk_target"] == (
            "threshold_malware_false_negative"
        )
        assert cfg["selective_prediction"]["mode"] == "risk_control"
        resolved_fusion = DiscountProbabilityFusion(cfg["fusion"])
        assert isinstance(
            resolved_fusion.reliability_calibrator,
            BranchTemperatureScalingConfidenceCalibrator,
        )

    assert (
        matched["fusion"]["reliability_calibration"]["temperature_fit_source"]
        == "clean_plus_branch_local_partial"
    )
    assert (
        clean["fusion"]["reliability_calibration"]["temperature_fit_source"]
        == "clean_only"
    )


def test_temperature_baseline_uses_grouped_oof_i1_lifecycle_and_full_refit():
    def _loader():
        return [
            {
                "graph_batch": _CalibrationGraph(_evidence(2)),
                "labels": torch.tensor([0, 1]),
                "sids": [f"benign-{fold}", f"malware-{fold}"],
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
            "loader": _loader(),
        },
        {
            "name": "api_degraded",
            "scenario_group": "api_degraded",
            "reliability_branches": ["api"],
            "loader": _loader(),
        },
    ]
    model = _TemperatureScalingPosthocModel()
    summary = fit_posthoc_calibration(
        model,
        sources,
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
                    "reliability": {
                        "optimizer": "lbfgs",
                        "lr": 1.0,
                        "max_steps": 2,
                        "min_steps": 1,
                        "convergence_patience": 1,
                        "convergence_tolerance": 1.0e-8,
                        "gradient_tolerance": 1.0e-8,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {
                "reliability_calibration": {
                    "enabled": True,
                    "method": TEMPERATURE_SCALING_CONFIDENCE_METHOD,
                    "temperature_fit_source": "clean_plus_branch_local_partial",
                },
                "routing": {"enabled": False},
            },
        },
    )

    assert summary["strategy"] == "identity_grouped_nested_crossfit_staged_refit"
    assert summary["reliability_calibration_method"] == (
        TEMPERATURE_SCALING_CONFIDENCE_METHOD
    )
    assert summary["reliability_calibration_fit_source"] == (
        "clean_plus_branch_local_partial"
    )
    assert summary["cross_fitting"]["oof_reliability_coverage"] == 1.0
    assert len(summary["_oof_clean_rows"]) == 6
    assert set(summary["reliability_temperatures"]) == {
        "api",
        "graph",
        "manifest",
    }
    assert summary["stages"]["reliability"]["objective_groups"] == [
        "api:clean_plus_branch_local_partial",
        "graph:clean_only",
        "manifest:clean_only",
    ]


def test_temperature_baseline_fits_only_configured_evidence_branches():
    cfg = load_config_path(
        ROOT / "ablations/i1/temperature_scaling_confidence.yaml"
    )
    fusion = DiscountProbabilityFusion(cfg["fusion"])
    calibrator = fusion.reliability_calibrator
    assert isinstance(calibrator, BranchTemperatureScalingConfidenceCalibrator)
    fitted = fusion.reliability_calibration_parameters()
    assert len(fitted) == 3
    assert {id(value) for value in fitted} == {
        id(calibrator.log_temperatures[name])
        for name in ("api", "graph", "manifest")
    }
    assert set(calibrator.log_temperatures) == {"api", "graph", "manifest"}


def test_explicit_main_i1_method_does_not_change_state_or_rng_consumption():
    base = {
        "combination": "cumulative",
        "reliability_calibration": {
            "enabled": True,
            "branches": ["api", "graph", "manifest"],
        },
        "routing": {"enabled": False},
    }
    torch.manual_seed(917)
    implicit = DiscountProbabilityFusion(base)
    implicit_next_random = torch.rand(4)

    explicit_cfg = {
        **base,
        "reliability_calibration": {
            **base["reliability_calibration"],
            "method": "monotonic_correctness",
        },
    }
    torch.manual_seed(917)
    explicit = DiscountProbabilityFusion(explicit_cfg)
    explicit_next_random = torch.rand(4)

    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    assert torch.equal(implicit_next_random, explicit_next_random)
