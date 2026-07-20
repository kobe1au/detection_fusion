from pathlib import Path

import torch

import run
from fusion.constants import EvidenceIndex
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
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _branch_probabilities(batch_size: int = 2) -> dict[str, torch.Tensor]:
    probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    if batch_size != probabilities.size(0):
        probabilities = probabilities[:1].expand(batch_size, -1).clone()
    return {
        name: probabilities.clone()
        for name in ("api", "graph", "manifest")
    }


def test_i1_feature_flags_preserve_calibrator_topology_and_initialization():
    torch.manual_seed(1701)
    full = MonotonicReliabilityCalibrator(
        use_model_visibility=True,
        use_prediction_margin=True,
        use_predicted_class_feature=True,
    )
    torch.manual_seed(1701)
    masked = MonotonicReliabilityCalibrator(
        use_model_visibility=False,
        use_prediction_margin=False,
        use_predicted_class_feature=False,
    )

    assert full.state_dict().keys() == masked.state_dict().keys()
    for name, value in full.state_dict().items():
        assert value.shape == masked.state_dict()[name].shape
        assert torch.equal(value, masked.state_dict()[name])

    assert {name: len(layout) for name, layout in RELIABILITY_FEATURE_LAYOUT.items()} == {
        "api": 6,
        "graph": 6,
        "manifest": 6,
    }
    for branch, layout in RELIABILITY_FEATURE_LAYOUT.items():
        branch_calibrator = full.branches[branch]
        assert branch_calibrator.competence is not None
        assert branch_calibrator.degradation is not None
        assert len(branch_calibrator.competence.raw_margin_weights) == 3
        assert len(branch_calibrator.degradation.raw_tail_weights) == 3
        competence_ids = {
            id(parameter) for parameter in branch_calibrator.competence_parameters()
        }
        degradation_ids = {
            id(parameter) for parameter in branch_calibrator.degradation_parameters()
        }
        assert competence_ids
        assert degradation_ids
        assert competence_ids.isdisjoint(degradation_ids)


def test_disabled_i1_features_are_zero_masked_in_fixed_slots():
    calibrator = MonotonicReliabilityCalibrator(
        use_model_visibility=False,
        use_prediction_margin=False,
        use_predicted_class_feature=False,
    )
    outputs = calibrator(
        _evidence(),
        branch_probabilities=_branch_probabilities(),
    )

    masked_feature_names = {
        "embedding_tail_q50",
        "embedding_tail_q80",
        "embedding_tail_q95",
        "prediction_margin",
        "predicted_malware_indicator",
    }
    for branch, layout in RELIABILITY_FEATURE_LAYOUT.items():
        features = outputs[f"reliability_features_superset_{branch}"]
        assert features.shape[-1] == len(layout)
        for index, feature_name in enumerate(layout):
            if feature_name in masked_feature_names:
                assert torch.equal(
                    features[:, index], torch.zeros_like(features[:, index])
                )


def test_auxiliary_supervision_is_alive_masked_in_main_and_i1_module_ablation():
    main = _resolved("evidential_trusted_fusion.yaml")
    no_i1 = _resolved("ablations/modules/no_reliability_discount.yaml")

    assert main["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
    assert no_i1["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"


def test_i1_atomic_configs_toggle_only_the_declared_axis():
    main = _resolved("evidential_trusted_fusion.yaml")
    main_reliability = main["fusion"]["reliability_calibration"]
    assert main_reliability["enabled"] is True
    assert main_reliability["use_model_visibility"] is True
    assert main_reliability["use_embedding_density"] is True
    assert main_reliability["use_prediction_margin"] is True
    assert main_reliability["use_predicted_class_feature"] is True
    assert not {
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
        "ablations/i1/no_model_visibility_feature.yaml": (
            True,
            False,
            True,
            True,
            True,
        ),
        "ablations/i1/no_embedding_density.yaml": (
            True,
            True,
            False,
            True,
            True,
        ),
        "ablations/i1/no_prediction_margin.yaml": (
            True,
            True,
            True,
            False,
            True,
        ),
        "ablations/i1/no_predicted_class_intercept.yaml": (
            True,
            True,
            True,
            True,
            False,
        ),
        "ablations/i1/no_learned_reliability_calibration.yaml": (
            False,
            True,
            True,
            True,
            True,
        ),
    }
    for relative, signature in expected.items():
        cfg = _resolved(relative)
        reliability = cfg["fusion"]["reliability_calibration"]
        actual = (
            reliability["enabled"],
            reliability["use_model_visibility"],
            reliability["use_embedding_density"],
            reliability["use_prediction_margin"],
            reliability["use_predicted_class_feature"],
        )
        assert actual == signature
        assert not {
            "feature_schema",
            "missing_relation_support",
            "use_relation_evidence",
            "use_edl_certainty_feature",
            "use_evidential_uncertainty",
            "group_mean_alignment",
        } & reliability.keys()
        assert cfg["fusion"]["use_reliability_discount"] is True
        assert not {
            "branch_competence_prior",
            "visible_integrity_modifier",
        } & cfg["fusion"].keys()
        assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"

    assert len(run.resolve_targets("i1_atomic")) == 5


def test_removing_i1_calibrator_does_not_change_shared_router_initialization():
    full_cfg = _resolved("evidential_trusted_fusion.yaml")["fusion"]
    no_i1_cfg = _resolved(
        "ablations/modules/no_reliability_discount.yaml"
    )["fusion"]

    torch.manual_seed(1907)
    full = DiscountProbabilityFusion(full_cfg)
    torch.manual_seed(1907)
    no_i1 = DiscountProbabilityFusion(no_i1_cfg)

    assert full.opinion_router is not None
    assert no_i1.opinion_router is not None
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
    assert main_routing["route_oracle_loss_weight"] == 0.0
    assert main_routing["subset_oracle_loss_weight"] == 0.0
    assert main_routing["subset_oracle_temperature"] == 1.0
    assert main_routing["risk_loss_weight"] == 1.0
    assert main_routing["train_end_to_end"] is False
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
        assert routing["train_end_to_end"] is False
        assert routing["posthoc_refine"] is True

        fusion = DiscountProbabilityFusion(cfg["fusion"])
        assert bool(fusion.routing_distribution_parameters()) is (
            routing["mode"] == "learned"
        )
        assert bool(fusion.routing_risk_parameters()) is (
            routing["risk_mode"] == "learned"
        )

    assert len(run.resolve_targets("i2_atomic")) == 4
    assert len(run.resolve_targets("i2_rules")) == 4


def test_i2_robust_route_configs_form_independent_objective_and_view_cells():
    main = _resolved("evidential_trusted_fusion.yaml")
    cells = {
        (False, True): main,
        (True, True): _resolved("ablations/i2/with_source_subset_oracle.yaml"),
        (False, False): _resolved("ablations/i2/group_robust_rho_0.yaml"),
        (True, False): _resolved(
            "ablations/i2/with_source_subset_oracle_group_robust_rho_0.yaml"
        ),
    }
    for (subset_enabled, soft_worst_enabled), cfg in cells.items():
        routing = cfg["fusion"]["routing"]
        robust = routing["group_robust_objective"]
        assert (routing["subset_oracle_loss_weight"] > 0.0) is subset_enabled
        assert (robust["soft_worst_weight"] > 0.0) is soft_worst_enabled
        assert robust["enabled"] is True
        assert robust["taxonomy"] == "perturb_type_strength_v1"

    no_pairwise = _resolved(
        "ablations/i2/no_pairwise_completeness_views.yaml"
    )
    assert main["calibration"]["include_pairwise_completeness_views"] is True
    assert (
        no_pairwise["calibration"]["include_pairwise_completeness_views"]
        is False
    )
    assert len(run.resolve_targets("i2_robust_route")) == 5


def test_no_i2_learned_components_keeps_i1_and_removes_both_i2_fits():
    cfg = _resolved("ablations/modules/no_i2_learned_components.yaml")
    routing = cfg["fusion"]["routing"]

    assert cfg["fusion"]["combination"] == "routed"
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["use_reliability_discount"] is True
    assert "visible_integrity_modifier" not in cfg["fusion"]
    assert routing["enabled"] is True
    assert routing["mode"] == "prior_only"
    assert routing["risk_mode"] == "reliability_prior"
    assert routing["prediction_loss_weight"] == 0.0
    assert routing["risk_loss_weight"] == 0.0
    assert routing["train_end_to_end"] is False
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
            "ablations/modules/no_reliability_discount.yaml"
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
        assert cfg["fusion"]["use_reliability_discount"] is i1_enabled
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
        assert routing["train_end_to_end"] is False
        assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
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
