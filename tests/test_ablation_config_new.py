from pathlib import Path

import pytest
import torch

import run
from fusion.constants import AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.reliability_calibration import (
    MonotonicReliabilityCalibrator,
    RELIABILITY_FEATURE_LAYOUT,
)
from fusion.train import load_config


ROOT = Path("config/experiments/tri_modal_robust")


def _resolved(relative: str) -> dict:
    return load_config([str(ROOT / relative)])


def _evidence(batch_size: int = 2) -> torch.Tensor:
    return torch.ones(batch_size, AvailabilityIndex.BASE_DIM)


def _i1_features(batch_size: int = 2) -> dict[str, torch.Tensor]:
    values = torch.tensor([[0.7, 0.6, 0.0], [0.4, 0.2, 1.0]])
    if batch_size != values.size(0):
        values = values[:1].expand(batch_size, -1).clone()
    return {name: values.clone() for name in RELIABILITY_FEATURE_LAYOUT}


def _alive(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones(batch_size)
        for name in RELIABILITY_FEATURE_LAYOUT
    }


def test_i1_feature_flags_keep_fixed_intrinsic_layout_and_mask_only_weights():
    torch.manual_seed(1701)
    full = MonotonicReliabilityCalibrator(
        use_evidential_certainty=True,
        use_prediction_margin=True,
        use_predicted_class_intercept=True,
    )
    torch.manual_seed(1701)
    no_certainty = MonotonicReliabilityCalibrator(
        use_evidential_certainty=False,
        use_prediction_margin=True,
        use_predicted_class_intercept=True,
    )
    torch.manual_seed(1701)
    no_margin = MonotonicReliabilityCalibrator(
        use_evidential_certainty=True,
        use_prediction_margin=False,
        use_predicted_class_intercept=True,
    )
    no_class = MonotonicReliabilityCalibrator(
        use_evidential_certainty=True,
        use_prediction_margin=True,
        use_predicted_class_intercept=False,
    )

    assert RELIABILITY_FEATURE_LAYOUT == {
        "api": (
            "evidential_certainty",
            "prediction_margin",
            "predicted_malware_indicator",
        ),
        "graph": (
            "evidential_certainty",
            "prediction_margin",
            "predicted_malware_indicator",
        ),
        "manifest": (
            "evidential_certainty",
            "prediction_margin",
            "predicted_malware_indicator",
        ),
    }
    for branch in RELIABILITY_FEATURE_LAYOUT:
        branch_calibrator = full.branches[branch]
        assert branch_calibrator.raw_continuous_weights.shape == (2,)
        assert branch_calibrator.predicted_class_intercept is not None
        assert not hasattr(branch_calibrator, "competence")
        assert not hasattr(branch_calibrator, "degradation")
        torch.testing.assert_close(
            branch_calibrator.raw_continuous_weights,
            no_certainty.branches[branch].raw_continuous_weights,
        )
        torch.testing.assert_close(
            branch_calibrator.raw_continuous_weights,
            no_margin.branches[branch].raw_continuous_weights,
        )
        assert no_certainty.branches[branch].active_continuous_mask.tolist() == [
            0.0,
            1.0,
        ]
        assert no_margin.branches[branch].active_continuous_mask.tolist() == [
            1.0,
            0.0,
        ]
        assert no_class.branches[branch].predicted_class_intercept is None


def test_disabled_i1_features_remain_observable_but_have_zero_effective_weight():
    calibrator = MonotonicReliabilityCalibrator(
        use_evidential_certainty=False,
        use_prediction_margin=True,
        use_predicted_class_intercept=False,
    )
    outputs = calibrator(
        _i1_features(),
        alive=_alive(),
    )

    for branch in RELIABILITY_FEATURE_LAYOUT:
        torch.testing.assert_close(
            outputs[f"reliability_features_{branch}"],
            _i1_features()[branch],
        )
        weights = calibrator.branches[
            branch
        ].effective_continuous_weights()
        assert weights["evidential_certainty"].item() == 0.0
        assert weights["prediction_margin"].item() > 0.0
    assert outputs["evidential_certainty_feature_active"].eq(0).all()
    assert outputs["prediction_margin_feature_active"].eq(1).all()
    assert outputs["predicted_class_intercept_active"].eq(0).all()


def test_auxiliary_supervision_is_alive_masked_in_main_and_i1_module_ablation():
    main = _resolved("evidential_trusted_fusion.yaml")
    no_i1 = _resolved("ablations/modules/no_i1_reliability.yaml")

    assert main["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
    assert no_i1["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"


def test_i1_atomic_configs_toggle_only_the_declared_axis():
    main = _resolved("evidential_trusted_fusion.yaml")
    main_reliability = main["fusion"]["reliability_calibration"]
    assert main_reliability["enabled"] is True
    assert main_reliability["use_evidential_certainty"] is True
    assert main_reliability["use_prediction_margin"] is True
    assert main_reliability["use_predicted_class_intercept"] is True
    assert not {
        "use_model_visibility",
        "use_predicted_class_feature",
        "feature_schema",
        "missing_relation_support",
        "use_relation_evidence",
        "use_edl_certainty_feature",
        "use_evidential_uncertainty",
        "group_mean_alignment",
    } & main_reliability.keys()
    assert not {
        "branch_competence_prior",
        "visible_integrity_modifier",
    } & main["fusion"].keys()

    expected = {
        "ablations/i1/no_evidential_certainty.yaml": (
            True,
            False,
            True,
            True,
        ),
        "ablations/i1/no_prediction_margin.yaml": (
            True,
            True,
            False,
            True,
        ),
        "ablations/i1/no_predicted_class_intercept.yaml": (
            True,
            True,
            True,
            False,
        ),
    }
    for relative, signature in expected.items():
        cfg = _resolved(relative)
        reliability = cfg["fusion"]["reliability_calibration"]
        actual = (
            reliability["enabled"],
            reliability["use_evidential_certainty"],
            reliability["use_prediction_margin"],
            reliability["use_predicted_class_intercept"],
        )
        assert actual == signature
        assert not {
            "use_model_visibility",
            "use_predicted_class_feature",
            "feature_schema",
            "missing_relation_support",
            "use_relation_evidence",
            "use_edl_certainty_feature",
            "use_evidential_uncertainty",
            "group_mean_alignment",
        } & reliability.keys()
        assert cfg["fusion"]["use_i1_reliability"] is True
        assert not {
            "branch_competence_prior",
            "visible_integrity_modifier",
        } & cfg["fusion"].keys()
        assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"

    assert len(run.resolve_targets("i1_atomic")) == 3


def test_removing_i1_calibrator_does_not_change_shared_router_initialization():
    full_cfg = _resolved("evidential_trusted_fusion.yaml")["fusion"]
    no_i1_cfg = _resolved(
        "ablations/modules/no_i1_reliability.yaml"
    )["fusion"]

    torch.manual_seed(1907)
    full = DiscountProbabilityFusion(full_cfg)
    torch.manual_seed(1907)
    no_i1 = DiscountProbabilityFusion(no_i1_cfg)

    assert full.opinion_router is not None
    assert no_i1.opinion_router is not None
    assert full.opinion_router.reliability_input_enabled is True
    assert no_i1.opinion_router.reliability_input_enabled is False
    assert no_i1.opinion_router.route_parameters() == [
        no_i1.opinion_router.raw_conflict_scale
    ]
    for name, value in full.opinion_router.state_dict().items():
        assert torch.equal(value, no_i1.opinion_router.state_dict()[name])


def test_i2_v2_atomic_configs_toggle_route_risk_and_conflict_independently():
    main = _resolved("evidential_trusted_fusion.yaml")
    main_routing = main["fusion"]["routing"]
    assert main_routing["mode"] == "learned"
    assert main_routing["risk_mode"] == "learned"
    assert main_routing["route_conflict_enabled"] is True
    assert main_routing["risk_conflict_enabled"] is True
    assert main_routing["prediction_loss_weight"] == 1.0
    assert main_routing["risk_loss_weight"] == 1.0
    assert not {
        "route_oracle_loss_weight",
        "route_oracle_temperature",
        "subset_oracle_loss_weight",
        "subset_oracle_temperature",
        "group_robust_objective",
    } & main_routing.keys()
    assert "train_end_to_end" not in main_routing
    assert main_routing["posthoc_refine"] is True

    expected = {
        "ablations/i2/router_prior_only.yaml": (
            "prior_only",
            "learned",
            True,
            True,
            0.0,
            1.0,
        ),
        "ablations/i2/router_risk_prior.yaml": (
            "learned",
            "reliability_prior",
            True,
            True,
            1.0,
            0.0,
        ),
        "ablations/i2/router_no_route_conflict.yaml": (
            "learned",
            "learned",
            False,
            True,
            1.0,
            1.0,
        ),
        "ablations/i2/router_no_risk_conflict.yaml": (
            "learned",
            "learned",
            True,
            False,
            1.0,
            1.0,
        ),
    }
    for relative, signature in expected.items():
        cfg = _resolved(relative)
        routing = cfg["fusion"]["routing"]
        actual = (
            routing["mode"],
            routing["risk_mode"],
            routing["route_conflict_enabled"],
            routing["risk_conflict_enabled"],
            routing["prediction_loss_weight"],
            routing["risk_loss_weight"],
        )
        assert actual == signature
        assert routing["enabled"] is True
        assert "train_end_to_end" not in routing
        assert routing["posthoc_refine"] is True
        assert routing["risk_target"] == "threshold_malware_false_negative"

        fusion = DiscountProbabilityFusion(cfg["fusion"])
        assert bool(fusion.routing_distribution_parameters()) is (
            routing["mode"] == "learned"
        )
        assert bool(fusion.routing_risk_parameters()) is (
            routing["risk_mode"] == "learned"
        )

    assert len(run.resolve_targets("i2_atomic")) == 4
    assert len(run.resolve_targets("fusion_rules")) == 4
    with pytest.raises(ValueError, match="full fusion-rule comparisons"):
        run.resolve_targets("i2_rules")


def test_i2_uses_only_compact_registered_views_and_has_no_oracle_grid():
    main = _resolved("evidential_trusted_fusion.yaml")
    assert main["calibration"]["fit_perturbations"] == [
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
    ]
    assert main["calibration"]["perturb_strengths"] == [0.3, 0.5, 0.7]
    assert "i2_robust_route" not in run.GROUPS
    assert not any(
        token in relative
        for relative in run.I2_MECHANISM_ABLATIONS
        for token in ("oracle", "group_robust", "pairwise")
    )


def test_no_i2_learned_components_keeps_i1_and_removes_both_i2_fits():
    cfg = _resolved("ablations/modules/no_i2_learned_components.yaml")
    routing = cfg["fusion"]["routing"]

    assert cfg["fusion"]["combination"] == "routed"
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["use_i1_reliability"] is True
    assert "visible_integrity_modifier" not in cfg["fusion"]
    assert routing["enabled"] is True
    assert routing["mode"] == "prior_only"
    assert routing["risk_mode"] == "reliability_prior"
    assert routing["prediction_loss_weight"] == 0.0
    assert routing["risk_loss_weight"] == 0.0
    assert "train_end_to_end" not in routing
    assert routing["posthoc_refine"] is False
    assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"

    fusion = DiscountProbabilityFusion(cfg["fusion"])
    assert fusion.routing_distribution_parameters() == []
    assert fusion.routing_risk_parameters() == []
    assert len(run.resolve_targets("module")) == 3


def test_i1_i2_factorial_changes_only_learned_i1_and_i2_axes():
    cells = {
        (True, True): _resolved("seeds/seed_42.yaml"),
        (False, True): _resolved(
            "ablations/modules/no_i1_reliability.yaml"
        ),
        (True, False): _resolved(
            "ablations/modules/no_i2_learned_components.yaml"
        ),
        (False, False): _resolved(
            "ablations/factorial/i1_i2/i1_off_i2_off.yaml"
        ),
    }
    for (i1_enabled, i2_enabled), cfg in cells.items():
        reliability = cfg["fusion"]["reliability_calibration"]
        routing = cfg["fusion"]["routing"]
        assert reliability["enabled"] is i1_enabled
        assert cfg["fusion"]["use_i1_reliability"] is i1_enabled
        assert "visible_integrity_modifier" not in cfg["fusion"]
        assert "branch_competence_prior" not in cfg["fusion"]
        assert routing["enabled"] is True
        assert cfg["fusion"]["combination"] == "routed"
        assert routing["mode"] == ("learned" if i2_enabled else "prior_only")
        assert routing["risk_mode"] == (
            "learned" if i2_enabled else "reliability_prior"
        )
        assert routing["prediction_loss_weight"] == (1.0 if i2_enabled else 0.0)
        assert routing["risk_loss_weight"] == (1.0 if i2_enabled else 0.0)
        assert routing["posthoc_refine"] is i2_enabled
        assert "train_end_to_end" not in routing
        assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
        fusion = DiscountProbabilityFusion(cfg["fusion"])
        assert fusion.opinion_router is not None
        assert fusion.opinion_router.reliability_input_enabled is i1_enabled
        if not i1_enabled and i2_enabled:
            assert fusion.opinion_router.route_parameters() == [
                fusion.opinion_router.raw_conflict_scale
            ]
        if not i1_enabled and not i2_enabled:
            assert fusion.opinion_router.route_parameters() == []
        assert cfg["selective_prediction"]["mode"] == "risk_control"

    assert len(run.resolve_targets("i1_i2_2x2")) == 4


def test_no_i3_is_an_atomic_selective_decision_ablation():
    main = _resolved("seeds/seed_42.yaml")
    no_i3 = _resolved("ablations/modules/no_i3_decision_layer.yaml")

    # The macro-F1 operating point is a shared classifier protocol interface;
    # I3 contributes only the downstream selective malware-FN risk control.
    assert main["classification_threshold"] == no_i3["classification_threshold"]
    assert main["classification_threshold"] == {
        "enabled": True,
        "objective": "macro_f1",
        "selection_rule": "macro_f1_unconstrained_v1",
        "constraint": "none",
    }
    assert main["selective_prediction"]["enabled"] is True
    assert no_i3["selective_prediction"]["enabled"] is False

    for section in ("data", "model", "fusion", "loss", "calibration"):
        assert no_i3[section] == main[section], section
    main_train = {
        key: value for key, value in main["train"].items() if key != "exp_name"
    }
    no_i3_train = {
        key: value for key, value in no_i3["train"].items() if key != "exp_name"
    }
    assert no_i3_train == main_train
    assert no_i3["eval"]["eval_only"] is True
    assert no_i3["eval"]["refit_posthoc_calibration"] is False
    assert no_i3["eval"]["refit_decision_calibration"] is False
    assert run.ALIASES["no_i3"] == "ablations/modules/no_i3_decision_layer.yaml"
    assert run.ALIASES["i3_off"] == "ablations/modules/no_i3_decision_layer.yaml"
    assert run.ALIASES["i3_on"] == run.PRIMARY_SEED
    assert "i3_2x2" not in run.GROUPS
    assert run.resolve_targets("factorial") == [
        run.CONFIG_DIR / item for item in run.I1_I2_FACTORIAL
    ]
