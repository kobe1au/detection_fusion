from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import run
from fusion.train import (
    _canonicalize_selective_prediction_config,
    _extra_eval_paths,
    _dataset_common_kwargs,
    _reject_removed_config_keys,
    build_run_identity,
    build_model,
    load_config,
    load_config_path,
    reliability_calibration_scenarios,
    resolve_posthoc_cross_fitting,
    uses_routing_calibration_scenarios,
    validate_checkpoint_implementation,
    validate_eval_checkpoint_config,
)


ROOT = Path("config/experiments/tri_modal_robust")
MAIN = ROOT / "evidential_trusted_fusion.yaml"
REMOVED_FUSION_KEYS = {
    "branch_competence_prior",
    "confidence_proxy",
    "conflict_factor",
    "detach_confidence_proxy",
    "detach_discount",
    "fallback",
    "probability_calibration",
    "reliability_discount_exponent",
    "support_factor",
    "use_confidence_proxy",
    "use_conflict_discount",
    "use_reliability_discount",
    "visible_integrity_modifier",
    "use_support_discount",
    "weight_sharpening_gamma",
}
REMOVED_I1_KEYS = {
    "apply_alive_mask",
    "use_model_visibility",
    "use_predicted_class_feature",
    "feature_schema",
    "missing_relation_support",
    "use_relation_evidence",
    "use_edl_certainty_feature",
    "use_evidential_uncertainty",
    "group_mean_alignment",
}


def _resolved(path: Path) -> dict:
    return load_config_path(path)


def _semantic_key(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.pop("method", None)
    normalized.get("train", {}).pop("exp_name", None)
    normalized.get("train", {}).pop("seed", None)
    normalized.get("eval", {}).pop("output_name", None)
    return json.dumps(normalized, sort_keys=True, default=str)


def _leaf_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = set()
        for key in set(left) | set(right):
            path = (*prefix, key)
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_leaf_differences(left[key], right[key], path))
        return differences
    return set() if left == right else {prefix}


def _mechanism_differences(reference: dict, candidate: dict):
    normalized = []
    for cfg in (reference, candidate):
        item = copy.deepcopy(cfg)
        item.pop("method", None)
        item.get("train", {}).pop("exp_name", None)
        item.get("train", {}).pop("seed", None)
        item.get("eval", {}).pop("output_name", None)
        normalized.append(item)
    return _leaf_differences(*normalized)


def _decision_only_differences(reference: dict, candidate: dict):
    normalized = []
    for cfg in (reference, candidate):
        item = copy.deepcopy(cfg)
        item.pop("method", None)
        eval_cfg = item.get("eval", {})
        for key in (
            "checkpoint_path",
            "eval_only",
            "output_name",
            "refit_decision_calibration",
        ):
            eval_cfg.pop(key, None)
        normalized.append(item)
    return _leaf_differences(*normalized)


# ── Generic resolve/build/protocol checks ─────────────────────────────────────

def test_all_runnable_experiment_configs_resolve_and_build():
    paths = run.resolve_targets("all")
    assert paths
    for path in paths:
        cfg = _resolved(path)
        _dataset_common_kwargs(cfg, is_train=False)
        build_model(cfg, feature_dim=515)


def test_unordered_all_catalog_is_dry_run_only():
    run.validate_execution_target_order(["all"], dry_run=True)
    with pytest.raises(ValueError, match="catalog-only"):
        run.validate_execution_target_order(["all"], dry_run=False)
    with pytest.raises(ValueError, match="catalog-only"):
        run.validate_execution_target_order(["final,all"], dry_run=False)


def test_runner_groups_and_aliases_reference_existing_configs():
    for name in run.GROUPS:
        assert run.resolve_targets(name)
    for name in run.ALIASES:
        assert run.resolve_targets(name)


def test_redundant_no_learned_i1_cell_and_alias_are_removed():
    assert "no_learned_reliability" not in run.ALIASES
    assert not (
        ROOT / "ablations/i1/no_learned_reliability_calibration.yaml"
    ).exists()
    assert not (
        ROOT / "ablations/modules/no_reliability_discount.yaml"
    ).exists()
    assert (
        ROOT / "ablations/modules/no_i1_reliability.yaml"
    ).is_file()
    with pytest.raises(ValueError, match="Unknown robust experiment target"):
        run.resolve_targets("no_learned_reliability")


def test_runner_groups_execute_each_physical_config_at_most_once():
    for name in run.GROUPS:
        paths = run.resolve_targets(name)
        resolved = [path.resolve() for path in paths]
        assert len(resolved) == len(set(resolved)), name


@pytest.mark.parametrize(
    "removed_key",
    (
        "reliability_weighted_aux",
        "integrity_weighted_aux",
        "semantic_reconstruction_weight",
        "cross_source_consistency_weight",
        "gate_prior_weight",
    ),
)
def test_removed_loss_switches_fail_even_when_neutral(removed_key):
    with pytest.raises(ValueError, match="Removed loss configuration keys"):
        _reject_removed_config_keys(
            {
                "loss": {
                    "auxiliary_weight_mode": "alive_masked_uniform",
                    removed_key: 0.0,
                }
            }
        )


@pytest.mark.parametrize(
    ("section", "removed_key", "value"),
    [
        *(('fusion', key, {"enabled": False}) for key in sorted(REMOVED_FUSION_KEYS)),
        ("reliability_calibration", "feature_schema", "intrinsic_v2"),
        ("reliability_calibration", "missing_relation_support", 0.0),
        ("reliability_calibration", "use_relation_evidence", False),
        ("reliability_calibration", "use_edl_certainty_feature", False),
        ("reliability_calibration", "use_evidential_uncertainty", False),
        ("reliability_calibration", "group_mean_alignment", False),
        ("reliability_calibration", "apply_alive_mask", True),
    ],
)
def test_build_model_rejects_removed_i1_keys_even_when_disabled(
    section, removed_key, value
):
    cfg = _resolved(MAIN)
    target = (
        cfg["fusion"]
        if section == "fusion"
        else cfg["fusion"]["reliability_calibration"]
    )
    target[removed_key] = value

    with pytest.raises(ValueError, match="[Rr]emoved"):
        build_model(cfg, feature_dim=16)


def test_formal_configs_use_clean_validation_selection():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        eval_cfg = cfg.get("eval", {}) or {}
        if eval_cfg.get("eval_only", False):
            continue
        assert "checkpoint_metric" not in cfg["train"], path
        assert "tuning_mode" not in cfg["train"], path
        assert "robust_val" not in eval_cfg, path


def test_runnable_configs_share_validation_protocol():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        calibration = cfg.get("calibration", {}) or {}
        assert cfg["train"]["strict_deterministic"] is False, path
        assert calibration.get("validation_fraction") == pytest.approx(0.60), path
        assert calibration.get("conformal_fraction") == pytest.approx(5.0 / 12.0), path
        assert calibration.get("split_seed") == 42, path
        assert calibration.get("stratified_group_split") is True, path
        assert int(cfg["model"]["max_nodes_gnn"]) > 0, path
        assert "account_for_encoder_budget" not in cfg["model"]["graph_encoder"], path
        assert cfg["fusion"]["force_fp32_decision"] is True, path


def test_main_posthoc_fit_uses_compact_explicit_scenario_protocol():
    cfg = _resolved(MAIN)
    scenarios = reliability_calibration_scenarios(cfg)
    assert cfg["calibration"]["fit_perturbations"] == [
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
    ]
    assert cfg["calibration"]["perturb_strengths"] == [0.3, 0.5, 0.7]
    assert len(scenarios) == 12
    assert {item["perturb_type"] for item in scenarios} == {
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
        "api_missing",
        "graph_missing",
        "manifest_missing",
    }
    assert {item["strength"] for item in scenarios} == {
        0.3,
        0.5,
        0.7,
        1.0,
    }
    branch_local_targets = {
        "api_event_dropout": ["api"],
        "graph_sparsify": ["graph"],
        "manifest_permission_mask": ["manifest"],
    }
    for perturb_type, branches in branch_local_targets.items():
        views = [item for item in scenarios if item["perturb_type"] == perturb_type]
        assert len(views) == 3
        assert all(item["reliability_branches"] == branches for item in views)

    missing_views = [item for item in scenarios if item["perturb_type"].endswith("_missing")]
    assert len(missing_views) == 3
    assert all(item["strength"] == 1.0 for item in missing_views)
    assert all(item["scenario_group"] == "missing" for item in missing_views)
    assert all(item["reliability_branches"] == [] for item in missing_views)

    assert "robust" not in cfg or "train_aug" not in cfg["robust"]


def test_formal_eval_protocol_matches_the_quality_mismatch_scope():
    cfg = _resolved(MAIN)
    scenarios = reliability_calibration_scenarios(cfg)
    fit_types = {item["perturb_type"] for item in scenarios}
    excluded_from_formal_protocol = {
        "api_category_dropout",
        "graph_node_feature_mask",
        "manifest_component_mask",
        "api_semantic_corrupted",
        "graph_semantic_corrupted",
        "manifest_semantic_corrupted",
        "api_graph_degraded",
        "api_manifest_degraded",
        "graph_manifest_degraded",
    }
    assert fit_types.isdisjoint(excluded_from_formal_protocol)
    assert set(cfg["eval"]["perturb_tests"]).isdisjoint(
        excluded_from_formal_protocol
    )
    assert cfg["eval"]["perturb_tests"] == [
        "clean",
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
        "api_missing",
        "graph_missing",
        "manifest_missing",
    ]
    assert cfg["eval"]["perturb_strengths"] == [0.1, 0.3, 0.5, 0.7, 0.9]
    graded_eval = [
        name
        for name in cfg["eval"]["perturb_tests"]
        if name != "clean" and not name.endswith("_missing")
    ]
    missing_eval = [
        name for name in cfg["eval"]["perturb_tests"] if name.endswith("_missing")
    ]
    assert len(graded_eval) == 3
    assert len(missing_eval) == 3
    assert 1 + len(graded_eval) * len(cfg["eval"]["perturb_strengths"]) + len(
        missing_eval
    ) == 19


def test_primary_method_and_baselines_share_the_same_robust_eval_matrix():
    reference = _resolved(ROOT / run.PRIMARY_SEED)["eval"]
    comparison_paths = [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]

    for relative_path in comparison_paths:
        eval_cfg = _resolved(ROOT / relative_path)["eval"]
        assert eval_cfg["perturb_tests"] == reference["perturb_tests"], relative_path
        assert eval_cfg["perturb_strengths"] == reference["perturb_strengths"], (
            relative_path
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fit_perturbations", [], "must be a non-empty sequence"),
        (
            "fit_perturbations",
            ["api_event_dropout", "api_event_dropout"],
            "contains duplicates",
        ),
        ("fit_perturbations", ["clean"], "remove reserved"),
        ("fit_perturbations", ["api_missing"], "remove reserved"),
        ("fit_perturbations", ["not_a_view"], "unsupported mechanisms"),
        ("perturb_strengths", [], "must be a non-empty sequence"),
        ("perturb_strengths", [0.3, 0.3], "contains duplicates"),
        ("perturb_strengths", [float("nan")], "must lie within"),
        ("perturb_strengths", [0.0], "must lie within"),
        ("perturb_strengths", [True], "not booleans"),
    ],
)
def test_posthoc_fit_scenario_protocol_fails_closed(field, value, message):
    cfg = _resolved(MAIN)
    cfg["calibration"][field] = value
    with pytest.raises(ValueError, match=message):
        reliability_calibration_scenarios(cfg)


def test_removed_pairwise_training_switch_is_rejected():
    cfg = _resolved(MAIN)
    cfg["calibration"]["include_pairwise_completeness_views"] = True
    with pytest.raises(ValueError, match="was removed"):
        reliability_calibration_scenarios(cfg)


def test_formal_method_refuses_in_sample_i1_to_i2_stacking():
    cfg = _resolved(MAIN)
    cfg["calibration"]["cross_fitting"]["enabled"] = False
    with pytest.raises(ValueError, match="forbids full-data I1 -> I2 stacking"):
        resolve_posthoc_cross_fitting(cfg)


def test_posthoc_views_are_used_by_either_i1_reliability_or_i2_routing():
    assert uses_routing_calibration_scenarios(_resolved(MAIN)) is True
    assert uses_routing_calibration_scenarios(
        _resolved(ROOT / "ablations/modules/no_i1_reliability.yaml")
    ) is True
    for relative in (
        "ablations/i2/combination_dempster.yaml",
        "ablations/i2/combination_cumulative.yaml",
        "ablations/i2/combination_log_pool.yaml",
        "ablations/i2/combination_conflict_weighted_opinion.yaml",
    ):
        # These cells remove routed I2, but retain learned I1 reliability.
        assert uses_routing_calibration_scenarios(_resolved(ROOT / relative)) is True
    for relative in (
        "baselines/tri_modal_concat.yaml",
        "baselines/trusted/tmc_style_adapted.yaml",
        "baselines/trusted/ecml_style_adapted.yaml",
    ):
        assert uses_routing_calibration_scenarios(_resolved(ROOT / relative)) is False


def test_posthoc_view_construction_matches_the_active_calibration_stage():
    main = _resolved(MAIN)
    i1_only = _resolved(ROOT / "ablations/i2/combination_dempster.yaml")
    i1_scenarios = reliability_calibration_scenarios(i1_only)
    assert len(i1_scenarios) == 9
    assert all(item["reliability_branches"] for item in i1_scenarios)

    risk_only_route = _resolved(ROOT / "ablations/i2/router_prior_only.yaml")
    risk_scenarios = reliability_calibration_scenarios(risk_only_route)
    assert len(risk_scenarios) == 12
    assert len(reliability_calibration_scenarios(main)) == 12


@pytest.mark.parametrize(
    ("removed_key", "value"),
    [
        ("route_oracle_loss_weight", 0.0),
        ("route_oracle_temperature", 1.0),
        ("subset_oracle_loss_weight", 0.0),
        ("subset_oracle_temperature", 1.0),
        (
            "group_robust_objective",
            {"enabled": False, "soft_worst_weight": 0.0},
        ),
    ],
)
def test_removed_oracle_and_group_robust_objectives_fail_closed(
    removed_key: str,
    value: object,
):
    cfg = _resolved(MAIN)
    cfg["fusion"]["routing"][removed_key] = value

    with pytest.raises(ValueError, match="removed from the final method"):
        build_model(cfg, feature_dim=515)


def test_autodl_overlay_uses_stable_tensor_sharing_settings():
    cfg = load_config(
        [
            str(ROOT / "seeds/seed_42.yaml"),
            str(ROOT / "_autodl_paths.yaml"),
        ]
    )
    assert cfg["train"]["multiprocessing_sharing_strategy"] == "file_system"
    assert cfg["train"]["prefetch_factor"] == 1
    assert {
        cfg["data"]["train_pt_dir"],
        cfg["data"]["val_pt_dir"],
        cfg["data"]["test_pt_dir"],
    } == {"/root/autodl-tmp/pts_all"}


def test_seed_only_overlays_change_only_the_seed():
    base = load_config([str(MAIN)])
    for seed in (2024, 3407):
        resolved = load_config(
            [
                str(MAIN),
                str(ROOT / f"_seed_{seed}_overlay.yaml"),
            ]
        )
        expected = copy.deepcopy(base)
        expected["train"]["seed"] = seed
        assert resolved == expected


def test_experiment_pt_paths_match_current_builder_output():
    build_cfg = yaml.safe_load(Path("config/build_pts.yaml").read_text(encoding="utf-8"))
    out_root = str(build_cfg["data"]["out_root"]).replace("\\", "/").rstrip("/")
    cfg = _resolved(MAIN)
    for split in ("train", "val", "test"):
        configured = str(cfg["data"][f"{split}_pt_dir"]).replace("\\", "/").rstrip("/")
        assert configured == f"{out_root}/{split}"


def test_curated_csv_can_select_a_strict_subset_of_current_pts():
    kwargs = _dataset_common_kwargs(_resolved(MAIN), is_train=True)
    assert kwargs["strict_split_integrity"] is True
    assert kwargs["allow_pt_superset"] is True


# ── Main evidential method ────────────────────────────────────────────────────

def test_evidential_main_method_configuration():
    cfg = _resolved(MAIN)
    assert cfg["method"]["protocol_id"] == (
        "intrinsic_opinion_reliability_router_crc_v2"
    )
    assert cfg["encoder_stage"]["protocol_id"] == (
        "neutral_alive_uniform_clean_stage1_v2"
    )
    assert cfg["model"]["fusion_mode"] == "discount_probability"
    # I1: one-stage branch correctness from intrinsic branch state only.
    reliability = cfg["fusion"]["reliability_calibration"]
    assert reliability["enabled"] is True
    assert reliability["branches"] == [
        "api",
        "graph",
        "manifest",
    ]
    assert not REMOVED_I1_KEYS & reliability.keys()
    assert reliability["method"] == "monotonic_correctness"
    assert reliability["use_evidential_certainty"] is True
    assert reliability["use_prediction_margin"] is True
    assert reliability["use_predicted_class_intercept"] is True
    assert "apply_alive_mask" not in reliability
    assert reliability["scenario_objective_weights"] == {
        "clean": 0.5,
        "perturb": 0.5,
    }
    assert cfg["fusion"]["use_i1_reliability"] is True
    assert not REMOVED_FUSION_KEYS & cfg["fusion"].keys()
    assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
    assert cfg["loss"]["evidential_loss_weight"] > 0.0
    assert cfg["loss"]["evidential"]["class_weight"] == "balanced"

    # I2: nested cross-fitted conditional routing and threshold-aligned FN risk.
    assert cfg["fusion"]["combination"] == "routed"
    routing = cfg["fusion"]["routing"]
    assert routing["enabled"] is True
    assert routing["mode"] == "learned"
    assert routing["route_conflict_enabled"] is True
    assert routing["prediction_loss_weight"] == 1.0
    assert routing["risk_mode"] == "learned"
    assert routing["risk_conflict_enabled"] is True
    assert routing["risk_loss_weight"] == 1.0
    assert routing["risk_loss"] == "bce"
    assert routing["risk_target"] == "threshold_malware_false_negative"
    assert "train_end_to_end" not in routing
    assert routing["posthoc_refine"] is True
    assert routing["final_temperature_scaling"] is True
    for obsolete_key in (
        "acceptance_score_mode",
        "route_oracle_loss_weight",
        "route_oracle_temperature",
        "subset_oracle_loss_weight",
        "subset_oracle_temperature",
        "group_robust_objective",
        "use_fused_prediction_loss",
        "target_loss_weight",
        "mass_constraint",
        "known_mass_excess_penalty_weight",
        "initial_known_retention",
        "use_disagreement",
    ):
        assert obsolete_key not in routing
    assert cfg["calibration"]["weight_decay"] == 0.0
    assert cfg["calibration"]["cross_fitting"] == {
        "required": True,
        "enabled": True,
        "mode": "nested",
        "num_folds": 5,
    }
    assert "hidden_dim" not in routing
    assert cfg["calibration"]["stage_optimization"]["reliability"][
        "optimizer"
    ] == "lbfgs"
    assert "reliability_competence" not in cfg["calibration"]["stage_optimization"]
    assert "reliability_degradation" not in cfg["calibration"]["stage_optimization"]
    assert cfg["calibration"]["fit_perturbations"] == [
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
    ]
    assert cfg["calibration"]["perturb_strengths"] == [0.3, 0.5, 0.7]
    assert cfg["eval"]["perturb_strengths"] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert not REMOVED_FUSION_KEYS & cfg["fusion"].keys()
    # I3: calibration-set malware false-negative risk control.
    assert cfg["classification_threshold"]["enabled"] is True
    assert cfg["classification_threshold"]["objective"] == "macro_f1"
    assert cfg["classification_threshold"]["selection_rule"] == (
        "macro_f1_unconstrained_v1"
    )
    assert cfg["classification_threshold"]["constraint"] == "none"
    assert "min_malware_recall" not in cfg["classification_threshold"]
    assert cfg["selective_prediction"]["enabled"] is True
    assert cfg["selective_prediction"]["mode"] == "risk_control"
    assert cfg["selective_prediction"]["threshold_score"] == "model_acceptance"
    assert cfg["selective_prediction"]["risk_level"] == 0.05


def test_run_identity_captures_decision_critical_configuration():
    cfg = _resolved(MAIN)
    identity = build_run_identity(cfg, "evidential_seed_42", 42)
    assert len(identity["resolved_config_sha256"]) == 64
    assert len(identity["method_protocol_sha256"]) == 64
    assert len(identity["method_implementation_sha256"]) == 64
    assert identity["method_protocol_id"] == (
        "intrinsic_opinion_reliability_router_crc_v2"
    )
    assert identity["combination_rule"] == "routed"
    assert identity["global_opinion_routing_enabled"] is True
    assert identity["routing_mode"] == "learned"
    assert identity["routing_route_conflict_enabled"] is True
    assert identity["routing_risk_conflict_enabled"] is True
    assert identity["routing_prediction_loss_weight"] == 1.0
    assert not any("oracle" in key for key in identity)
    assert not any("group_robust" in key for key in identity)
    assert identity["posthoc_fit_perturbations"] == [
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
    ]
    assert identity["posthoc_fit_perturbation_strengths"] == [0.3, 0.5, 0.7]
    assert identity["posthoc_fit_transformed_source_count"] == 12
    assert identity["robust_eval_perturbations"] == cfg["eval"]["perturb_tests"]
    assert identity["robust_eval_perturbation_strengths"] == [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ]
    assert identity["robust_eval_expected_result_count"] == 19
    assert identity["routing_risk_enabled"] is True
    assert identity["routing_risk_mode"] == "learned"
    assert identity["routing_risk_loss_weight"] == 1.0
    assert identity["routing_posthoc_distribution_loss_enabled"] is True
    assert identity["routing_posthoc_risk_loss_enabled"] is True
    assert "routing_acceptance_score_mode" not in identity
    assert identity["routed_final_temperature_enabled"] is True
    assert identity["reliability_calibration_branches"] == [
        "api",
        "graph",
        "manifest",
    ]
    assert identity["reliability_calibrator_architecture"] == (
        "per_branch_monotonic_logistic_correctness"
    )
    assert identity["i1_reliability_input_enabled"] is True
    assert "reliability_use_enabled" not in identity
    assert identity["reliability_lifecycle"] == (
        "single_stage_global_branch_correctness_proper_loss"
    )
    assert identity["router_trained_end_to_end"] is False
    assert identity["router_posthoc_refinement_enabled"] is True
    assert (
        identity["router_encoder_training_reliability_source"]
        == "alive_masked_uniform"
    )
    assert identity["router_posthoc_reliability_source"] == "calibrated_branch_correctness"
    assert identity["semantic_supervision_in_i1"] is False
    assert identity["evidential_certainty_reliability_enabled"] is True
    assert identity["prediction_margin_reliability_enabled"] is True
    assert identity["predicted_class_intercept_enabled"] is True
    assert identity["classification_threshold_enabled"] is True
    assert identity["classification_threshold_objective"] == "macro_f1"
    assert identity["classification_threshold_selection_rule"] == (
        "macro_f1_unconstrained_v1"
    )
    assert identity["classification_threshold_constraint"] == "none"
    assert "classification_min_malware_recall" not in identity
    assert identity["risk_control_level"] == 0.05
    assert identity["selective_score_type"] == "model_acceptance"


def test_disabled_baseline_components_are_reported_as_inactive():
    cfg = _resolved(ROOT / "baselines/tri_modal_concat.yaml")
    identity = build_run_identity(cfg, "baseline_tri_modal_concat", 42)

    assert identity["reliability_calibration_enabled"] is False
    assert identity["i1_reliability_input_enabled"] is False
    assert "reliability_use_enabled" not in identity
    assert identity["reliability_calibration_branches"] == []
    assert identity["semantic_supervision_in_i1"] is False
    assert identity["evidential_certainty_reliability_enabled"] is False
    assert identity["prediction_margin_reliability_enabled"] is False
    assert identity["predicted_class_intercept_enabled"] is False
    assert identity["classification_threshold_objective"] == "disabled"
    assert identity["selective_prediction_mode"] == "disabled"
    assert identity["selective_score_type"] == "disabled"


def test_non_routed_fusion_comparison_does_not_report_routed_components_as_active():
    cfg = _resolved(ROOT / "ablations/i2/combination_dempster.yaml")
    identity = build_run_identity(cfg, "fusion_rule_dempster", 42)
    assert identity["global_opinion_routing_enabled"] is False
    assert identity["routing_mode"] == "disabled"
    assert identity["routing_risk_enabled"] is False
    assert identity["routing_posthoc_distribution_loss_enabled"] is False
    assert identity["routing_posthoc_risk_loss_enabled"] is False
    assert "routing_acceptance_score_mode" not in identity
    assert identity["routed_final_temperature_enabled"] is False


def test_fusion_rule_comparisons_keep_main_i1_without_removed_i1_paths():
    for relative in (
        "ablations/i2/combination_dempster.yaml",
        "ablations/i2/combination_cumulative.yaml",
        "ablations/i2/combination_log_pool.yaml",
        "ablations/i2/combination_conflict_weighted_opinion.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert (
            cfg["fusion"]["reliability_calibration"]
            == _resolved(MAIN)["fusion"]["reliability_calibration"]
        )
        assert not REMOVED_FUSION_KEYS & cfg["fusion"].keys()
        assert not REMOVED_I1_KEYS & cfg["fusion"]["reliability_calibration"].keys()
        assert "acceptance_score_mode" not in cfg["fusion"]["routing"]
        assert cfg["fusion"]["routing"]["final_temperature_scaling"] is True


def test_i2_router_atomic_ablations_are_independent_v2_axes():
    assert run.resolve_targets("i2_atomic") == [
        run.CONFIG_DIR / item for item in run.I2_ROUTER_ATOMIC_ABLATIONS
    ]
    expected_changes = {
        "ablations/i2/router_prior_only.yaml": {
            ("fusion", "routing", "mode"),
            ("fusion", "routing", "fixed_prior_beta"),
            ("fusion", "routing", "prediction_loss_weight"),
        },
        "ablations/i2/router_risk_prior.yaml": {
            ("fusion", "routing", "risk_mode"),
            ("fusion", "routing", "risk_loss_weight"),
        },
        "ablations/i2/router_no_route_conflict.yaml": {
            ("fusion", "routing", "route_conflict_enabled"),
        },
        "ablations/i2/router_no_risk_conflict.yaml": {
            ("fusion", "routing", "risk_conflict_enabled"),
        },
    }
    main = _resolved(MAIN)
    for relative, expected in expected_changes.items():
        cfg = _resolved(ROOT / relative)
        assert _mechanism_differences(main, cfg) == expected, relative
        routing = cfg["fusion"]["routing"]
        assert "train_end_to_end" not in routing
        assert routing["posthoc_refine"] is True
        assert "acceptance_score_mode" not in routing
        assert routing["risk_target"] == "threshold_malware_false_negative"
        assert "mass_constraint" not in routing
        assert "known_mass_excess_penalty_weight" not in routing

    prior_identity = build_run_identity(
        _resolved(ROOT / "ablations/i2/router_prior_only.yaml"),
        "i2_router_prior_only",
        42,
    )
    assert prior_identity["routing_route_conflict_enabled"] is False
    assert prior_identity["routing_risk_conflict_enabled"] is True


def test_i3_acceptance_score_ablation_uses_standard_score_interfaces_only():
    assert run.resolve_targets("i3_acceptance_score") == [
        run.CONFIG_DIR / item for item in run.I3_ACCEPTANCE_SCORE_COMPARISONS
    ]
    expected_differences = {
        "ablations/i3/acceptance_msp_risk_control.yaml": {
            ("selective_prediction", "threshold_score")
        },
        "ablations/i3/acceptance_deployed_class_probability_risk_control.yaml": {
            ("selective_prediction", "threshold_score")
        },
    }
    reference = _resolved(ROOT / run.PRIMARY_SEED)
    assert "acceptance_score_mode" not in reference["fusion"]["routing"]
    assert reference["selective_prediction"]["threshold_score"] == "model_acceptance"
    assert reference["eval"].get("eval_only", False) is False
    observed_scores = {reference["selective_prediction"]["threshold_score"]}
    for relative in run.I3_ACCEPTANCE_SCORE_COMPARISONS:
        cfg = _resolved(ROOT / relative)
        assert "acceptance_score_mode" not in cfg["fusion"]["routing"]
        observed_scores.add(cfg["selective_prediction"]["threshold_score"])
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_decision_calibration"] is True
        assert cfg["eval"].get("refit_posthoc_calibration", False) is False
        assert cfg["selective_prediction"]["mode"] == "risk_control"
        assert cfg["selective_prediction"]["risk_level"] == reference[
            "selective_prediction"
        ]["risk_level"]
        assert _decision_only_differences(reference, cfg) == expected_differences[
            relative
        ], relative
    assert observed_scores == {
        "model_acceptance",
        "msp",
        "deployed_class_probability",
    }
    assert all(
        item in run.I3_MECHANISM_ABLATIONS
        for item in run.I3_ACCEPTANCE_SCORE_COMPARISONS
    )


def test_method_protocol_identity_is_seed_invariant():
    seed_42 = _resolved(ROOT / "seeds/seed_42.yaml")
    seed_2024 = _resolved(ROOT / "seeds/seed_2024.yaml")
    identity_42 = build_run_identity(seed_42, "evidential_seed_42", 42)
    identity_2024 = build_run_identity(seed_2024, "evidential_seed_2024", 2024)

    assert identity_42["resolved_config_sha256"] != identity_2024["resolved_config_sha256"]
    assert identity_42["method_protocol_sha256"] == identity_2024["method_protocol_sha256"]


# ── Baselines ─────────────────────────────────────────────────────────────────

def test_baselines_disable_reliability_calibration_and_rejection():
    for relative in ("baselines/api_only.yaml", "baselines/graph_only.yaml",
                     "baselines/manifest_only.yaml", "baselines/tri_modal_concat.yaml",
                     "baselines/fixed_logit_fusion.yaml", "baselines/api_graph_concat.yaml"):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["mode"] == "model_dispatch", relative
        assert not REMOVED_FUSION_KEYS & cfg["fusion"].keys(), relative
        assert not REMOVED_I1_KEYS & cfg["fusion"]["reliability_calibration"].keys(), relative
        assert cfg["calibration"]["enabled"] is False, relative
        assert cfg["calibration"]["holdout_enabled"] is True, relative
        assert cfg["selective_prediction"]["enabled"] is False, relative
        assert cfg["loss"]["auxiliary_weight_mode"] == "unmasked_uniform", relative
        assert "robust" not in cfg or "train_aug" not in cfg["robust"], relative


def test_recent_trusted_fusion_baselines_do_not_reuse_ours_i1_or_i3():
    expected = {
        "baselines/trusted/tmc_style_adapted.yaml": (
            "discount_probability", "dempster", "tmc"
        ),
        "baselines/trusted/ecml_style_adapted.yaml": (
            "discount_probability", "ecml", "ecml"
        ),
        "baselines/trusted/qmf_energy.yaml": (
            "tri_modal_quality_fusion", None, "standard"
        ),
    }
    assert run.resolve_targets("trusted_baselines") == [
        run.CONFIG_DIR / relative for relative in run.TRUSTED_FUSION_BASELINES
    ]
    for relative, (mode, combination, objective) in expected.items():
        cfg = _resolved(ROOT / relative)
        assert cfg["model"]["fusion_mode"] == mode, relative
        assert cfg["fusion"].get("combination") == combination, relative
        assert cfg["loss"].get("objective", "standard") == objective, relative
        assert cfg["fusion"]["reliability_calibration"]["enabled"] is False, relative
        assert not REMOVED_FUSION_KEYS & cfg["fusion"].keys(), relative
        assert not REMOVED_I1_KEYS & cfg["fusion"]["reliability_calibration"].keys(), relative
        assert cfg["selective_prediction"]["enabled"] is False, relative


# ── Module / mechanism ablations ──────────────────────────────────────────────

def test_module_ablations_remove_whole_innovations():
    no_i1 = _resolved(ROOT / "ablations/modules/no_i1_reliability.yaml")
    assert no_i1["fusion"].get("opinion_source", "evidential") == "evidential"
    assert no_i1["fusion"]["combination"] == "routed"
    assert no_i1["fusion"]["routing"]["enabled"] is True
    assert no_i1["fusion"]["use_i1_reliability"] is False
    assert not REMOVED_FUSION_KEYS & no_i1["fusion"].keys()
    assert not REMOVED_I1_KEYS & no_i1["fusion"]["reliability_calibration"].keys()
    assert no_i1["fusion"]["reliability_calibration"]["enabled"] is False
    assert no_i1["calibration"]["enabled"] is True
    assert no_i1["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
    assert no_i1["loss"]["evidential_loss_weight"] == 0.05
    no_i1_identity = build_run_identity(no_i1, "module_no_i1_reliability", 42)
    assert no_i1_identity["i1_reliability_input_enabled"] is False

    no_i2 = _resolved(ROOT / "ablations/modules/no_i2_learned_components.yaml")
    assert no_i2["fusion"].get("opinion_source", "evidential") == "evidential"
    assert no_i2["fusion"]["combination"] == "routed"
    assert no_i2["fusion"]["routing"]["enabled"] is True
    assert no_i2["fusion"]["routing"]["mode"] == "prior_only"
    assert no_i2["fusion"]["routing"]["prediction_loss_weight"] == 0.0
    assert no_i2["fusion"]["routing"]["risk_mode"] == "reliability_prior"
    assert no_i2["fusion"]["routing"]["risk_loss_weight"] == 0.0
    assert "train_end_to_end" not in no_i2["fusion"]["routing"]
    assert no_i2["fusion"]["routing"]["posthoc_refine"] is False
    assert no_i2["loss"]["evidential_loss_weight"] == 0.05

    no_i3 = _resolved(ROOT / "ablations/modules/no_i3_decision_layer.yaml")
    assert no_i3["selective_prediction"]["enabled"] is False
    assert no_i3["classification_threshold"] == _resolved(MAIN)[
        "classification_threshold"
    ]
    assert no_i3["fusion"]["combination"] == "routed"
    assert no_i3["eval"]["eval_only"] is True
    assert no_i3["eval"]["refit_posthoc_calibration"] is False
    assert no_i3["eval"]["refit_decision_calibration"] is False

def test_i1_atomic_ablations_change_exactly_one_current_feature_axis():
    assert run.resolve_targets("i1_atomic") == [
        run.CONFIG_DIR / item for item in run.I1_ATOMIC_ABLATIONS
    ]
    expected_changes = {
        "ablations/i1/no_evidential_certainty.yaml": {
            ("fusion", "reliability_calibration", "use_evidential_certainty"),
        },
        "ablations/i1/no_prediction_margin.yaml": {
            ("fusion", "reliability_calibration", "use_prediction_margin"),
        },
        "ablations/i1/no_predicted_class_intercept.yaml": {
            ("fusion", "reliability_calibration", "use_predicted_class_intercept"),
        },
    }
    main = _resolved(MAIN)
    for relative, expected in expected_changes.items():
        cfg = _resolved(ROOT / relative)
        assert _mechanism_differences(main, cfg) == expected, relative
        reliability = cfg["fusion"]["reliability_calibration"]
        assert not REMOVED_I1_KEYS & reliability.keys()
        assert not REMOVED_FUSION_KEYS & cfg["fusion"].keys()


def test_i1_temperature_scaling_is_a_comparator_not_an_atomic_ablation():
    assert run.resolve_targets("i1_comparator") == [
        run.CONFIG_DIR / item for item in run.I1_COMPARATORS
    ]
    assert not set(run.I1_COMPARATORS) & set(run.I1_ATOMIC_ABLATIONS)
    assert set(run.I1_COMPARATORS) <= set(run.MECHANISM_ABLATIONS)


def test_training_ablation_removes_edl_supervision_not_i1_reliability():
    cfg = _resolved(ROOT / "ablations/training/no_edl_supervision.yaml")
    assert cfg["loss"]["evidential_loss_weight"] == 0.0
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert not REMOVED_I1_KEYS & cfg["fusion"]["reliability_calibration"].keys()
    assert cfg["fusion"]["combination"] == "routed"


def test_complete_fusion_rule_comparisons_switch_stage1_rule_explicitly():
    assert not set(run.FUSION_RULE_COMPARISONS) & set(
        run.I2_MECHANISM_ABLATIONS
    )
    for relative, rule in (
        ("ablations/i2/combination_dempster.yaml", "dempster"),
        ("ablations/i2/combination_cumulative.yaml", "cumulative"),
        ("ablations/i2/combination_log_pool.yaml", "log_pool"),
        (
            "ablations/i2/combination_conflict_weighted_opinion.yaml",
            "conflict_weighted_opinion",
        ),
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["combination"] == rule, relative
        assert cfg["fusion"]["routing"]["enabled"] is False, relative
        assert (
            cfg["encoder_stage"]["protocol_id"]
            == "fixed_evidential_rule_stage1_v1"
        ), relative


def test_comparison_configs_do_not_claim_main_uniform_stage1_protocol():
    main = _resolved(MAIN)
    assert (
        main["encoder_stage"]["protocol_id"]
        == "neutral_alive_uniform_clean_stage1_v2"
    )
    for relative in [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]:
        cfg = _resolved(ROOT / relative)
        assert cfg["encoder_stage"]["protocol_id"] != (
            "neutral_alive_uniform_clean_stage1_v2"
        ), relative


def test_i3_mechanism_ablations_are_decision_only():
    class_conditional = _resolved(
        ROOT / "ablations/i3/class_conditional_conformal.yaml"
    )
    assert class_conditional["selective_prediction"]["mode"] == "conformal"
    assert class_conditional["selective_prediction"]["class_conditional"] is True
    assert "risk_level" not in class_conditional["selective_prediction"]
    assert "threshold_score" not in class_conditional["selective_prediction"]
    assert class_conditional["eval"]["eval_only"] is True
    assert class_conditional["eval"]["refit_decision_calibration"] is True

    marginal = _resolved(ROOT / "ablations/i3/marginal_conformal.yaml")
    assert marginal["selective_prediction"]["class_conditional"] is False
    assert marginal["eval"]["eval_only"] is True
    assert marginal["eval"]["refit_decision_calibration"] is True

    conflict = _resolved(ROOT / "ablations/i3/conflict_augmented_conformal.yaml")
    assert conflict["selective_prediction"]["use_raw_conflict"] is True
    assert conflict["eval"]["eval_only"] is True
    assert conflict["eval"]["refit_decision_calibration"] is True

    deployed = _resolved(
        ROOT / "ablations/i3/deployed_class_probability_threshold.yaml"
    )
    assert deployed["selective_prediction"]["mode"] == "threshold"
    assert (
        deployed["selective_prediction"]["threshold_score"]
        == "deployed_class_probability"
    )
    assert deployed["eval"]["eval_only"] is True
    assert deployed["selective_prediction"]["target_coverage"] == pytest.approx(0.90)

    msp = _resolved(ROOT / "ablations/i3/msp_threshold.yaml")
    assert msp["selective_prediction"]["mode"] == "threshold"
    assert msp["selective_prediction"]["threshold_score"] == "msp"
    assert msp["eval"]["eval_only"] is True
    assert msp["selective_prediction"]["target_coverage"] == pytest.approx(0.90)
    assert "risk_level" not in msp["selective_prediction"]
    assert "class_conditional" not in msp["selective_prediction"]

    entropy = _resolved(ROOT / "ablations/i3/predictive_entropy_threshold.yaml")
    assert entropy["selective_prediction"]["mode"] == "threshold"
    assert (
        entropy["selective_prediction"]["threshold_score"]
        == "predictive_entropy_certainty"
    )
    assert entropy["eval"]["eval_only"] is True
    assert entropy["selective_prediction"]["target_coverage"] == pytest.approx(0.90)

    uncertainty = _resolved(ROOT / "ablations/i3/uncertainty_threshold.yaml")
    assert uncertainty["selective_prediction"]["mode"] == "threshold"
    assert uncertainty["selective_prediction"]["threshold_score"] == "mixture_certainty"
    assert uncertainty["selective_prediction"]["target_coverage"] == pytest.approx(0.90)
    assert uncertainty["eval"]["eval_only"] is True

    risk_control = _resolved(MAIN)["selective_prediction"]
    assert risk_control["mode"] == "risk_control"
    assert "target_coverage" not in risk_control
    assert "class_conditional" not in risk_control


def test_selective_mode_canonicalization_stabilizes_method_identity():
    cfg = _resolved(ROOT / "ablations/i3/msp_threshold.yaml")
    with_inactive_crc_key = copy.deepcopy(cfg)
    with_inactive_crc_key["selective_prediction"]["risk_level"] = 0.123

    canonical = build_run_identity(cfg, "msp_threshold", 42)
    ignored = build_run_identity(with_inactive_crc_key, "msp_threshold", 42)

    assert canonical["resolved_config_sha256"] == ignored["resolved_config_sha256"]
    assert canonical["method_protocol_sha256"] == ignored["method_protocol_sha256"]

    active_change = copy.deepcopy(cfg)
    active_change["selective_prediction"]["target_coverage"] = 0.8
    changed = build_run_identity(active_change, "msp_threshold", 42)
    assert changed["method_protocol_sha256"] != canonical["method_protocol_sha256"]


@pytest.mark.parametrize(
    "relative",
    (
        "evidential_trusted_fusion.yaml",
        "ablations/i1/temperature_scaling_confidence.yaml",
    ),
)
def test_active_single_stage_i1_optimizer_budget_is_protocol_critical(relative):
    cfg = _resolved(ROOT / relative)
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg["calibration"]["stage_optimization"]["reliability"][
        "max_steps"
    ] += 17

    canonical = build_run_identity(cfg, relative, 42)
    changed = build_run_identity(changed_cfg, relative, 42)

    assert canonical["resolved_config_sha256"] != changed["resolved_config_sha256"]
    assert canonical["method_protocol_sha256"] != changed["method_protocol_sha256"]
    with pytest.raises(ValueError, match="changes model/data semantics"):
        validate_eval_checkpoint_config(changed_cfg, cfg)


def test_unknown_selective_mode_is_rejected_during_config_resolution(tmp_path):
    path = tmp_path / "invalid_selective_mode.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "defaults": [str(MAIN.resolve())],
                "selective_prediction": {"enabled": True, "mode": "typo"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selective_prediction.mode"):
        load_config_path(path)


@pytest.mark.parametrize("value", [[], "", 1])
def test_selective_config_must_be_a_mapping(value):
    with pytest.raises(ValueError, match="must be a mapping"):
        _canonicalize_selective_prediction_config(
            {"selective_prediction": value}
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [("enabled", "false"), ("require_feasible", "false")],
)
def test_selective_boolean_switches_are_not_truthy_strings(key, value):
    config = {
        "enabled": True,
        "mode": "risk_control",
        key: value,
    }
    with pytest.raises(ValueError, match="must be boolean"):
        _canonicalize_selective_prediction_config(
            {"selective_prediction": config}
        )


@pytest.mark.parametrize("value", [True, 0, 1.5, float("inf")])
def test_selective_crc_minimum_requires_a_strict_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        _canonicalize_selective_prediction_config(
            {
                "selective_prediction": {
                    "enabled": True,
                    "mode": "risk_control",
                    "min_calibration_malware": value,
                }
            }
        )


def test_conformal_alpha_and_coverage_cannot_disagree():
    with pytest.raises(ValueError, match="alpha conflicts"):
        _canonicalize_selective_prediction_config(
            {
                "selective_prediction": {
                    "enabled": True,
                    "mode": "conformal",
                    "alpha": 0.2,
                    "target_coverage": 0.9,
                }
            }
        )


def test_mechanism_group_matches_declared_splits():
    paths = run.resolve_targets("mechanism")
    expected = [run.CONFIG_DIR / relative for relative in run.MECHANISM_ABLATIONS]
    assert paths == expected


# ── Appendix sensitivity ─────────────────────────────────────────────────────

def test_appendix_sensitivity_configs_have_distinct_meanings():
    main = _resolved(MAIN)
    assert main["loss"]["evidential_loss_weight"] == 0.05
    edl = _resolved(ROOT / "appendix/edl_weight_0_10.yaml")
    assert edl["loss"]["evidential_loss_weight"] == 0.10
    risk_03 = _resolved(ROOT / "appendix/risk_level_0_03_eval.yaml")
    assert risk_03["selective_prediction"]["risk_level"] == 0.03
    assert risk_03["eval"]["eval_only"] is True
    assert risk_03["eval"]["refit_decision_calibration"] is True
    risk_10 = _resolved(ROOT / "appendix/risk_level_0_10_eval.yaml")
    assert risk_10["selective_prediction"]["risk_level"] == 0.10
    assert risk_10["eval"]["eval_only"] is True
    assert risk_10["eval"]["refit_decision_calibration"] is True


def test_paper_evidential_plan_excludes_sensitivity():
    paper = {p.resolve() for p in run.resolve_targets("paper_evidential")}
    appendix = {p.resolve() for p in run.resolve_targets("appendix")}
    assert not (paper & appendix)
    paper_all = {p.resolve() for p in run.resolve_targets("paper_evidential_all")}
    assert not (paper_all & appendix)


# ── Seeds / external / eval-only reuse ────────────────────────────────────────

def test_seed_overlays_change_only_method_and_seed():
    base = _resolved(MAIN)
    for seed in (42, 2024, 3407):
        resolved = _resolved(ROOT / f"seeds/seed_{seed}.yaml")
        expected = copy.deepcopy(base)
        expected["method"] = {
            **base["method"],
            "name": f"evidential_seed_{seed}",
        }
        expected["train"]["exp_name"] = f"evidential_seed_{seed}"
        expected["train"]["seed"] = seed
        assert resolved == expected, seed


def test_formal_runner_has_no_obfuscation_experiment_entrypoints():
    formal_names = [*run.GROUPS, *run.ALIASES]
    assert all("obfus" not in name.lower() for name in formal_names)
    for group in ("all", "paper_main", "paper_ablation", "paper_evidential_all"):
        assert all(
            "obfus" not in path.as_posix().lower()
            for path in run.resolve_targets(group)
        )


def test_natural_subset_eval_configs_share_the_same_curated_csvs():
    expected_csvs = [
        f"labels/natural_subsets/{name}"
        for name in run.NATURAL_SUBSET_FILES
    ]
    assert run.resolve_targets("natural_subsets") == [
        run.CONFIG_DIR / item for item in run.NATURAL_SUBSET_EVAL
    ]
    for relative in run.NATURAL_SUBSET_EVAL:
        cfg = _resolved(ROOT / relative)
        eval_cfg = cfg["eval"]
        assert eval_cfg["eval_only"] is True, relative
        assert eval_cfg["run_test"] is False, relative
        assert eval_cfg["run_robust_test"] is False, relative
        assert eval_cfg["refit_decision_calibration"] is False, relative
        extra_sets = eval_cfg.get("extra_sets") or []
        if isinstance(extra_sets, dict):
            extra_items = list(extra_sets.values())
        else:
            extra_items = list(extra_sets)
        actual_csvs = [item["csv"] for item in extra_items]
        assert actual_csvs == expected_csvs, relative
        assert all("pt_dir" not in item for item in extra_items), relative
        assert all(
            _extra_eval_paths(cfg, item)[0].replace("\\", "/").rstrip("/")
            == str(cfg["data"]["test_pt_dir"]).replace("\\", "/").rstrip("/")
            for item in extra_items
        ), relative
        assert all(item.get("allow_pt_superset", True) is True for item in extra_items), relative


def test_natural_subset_preflight_rejects_stale_source_hashes(tmp_path):
    subset_dir = tmp_path / "labels" / "natural_subsets"
    subset_dir.mkdir(parents=True)
    diagnostics = tmp_path / "results" / "gate_diagnostics.csv"
    diagnostics.parent.mkdir(parents=True)
    diagnostics.write_text("sid,split\na,test_clean\n", encoding="utf-8")
    test_csv = tmp_path / "labels" / "test.csv"
    test_csv.write_text("sha256,label\na,0\n", encoding="utf-8")
    subset_records = []
    for name in run.NATURAL_SUBSET_FILES:
        path = subset_dir / name
        path.write_text("sha256,label\na,0\n", encoding="utf-8")
        subset_records.append(
            {
                "csv": f"labels/natural_subsets/{name}",
                "csv_sha256": run._sha256(path),
            }
        )
    manifest = {
        "schema_version": run.NATURAL_SUBSET_SCHEMA_VERSION,
        "protocol_id": run.NATURAL_SUBSET_PROTOCOL_ID,
        "protocol_guarantees": {
            "thresholds_fit_on_validation_only": True,
            "target_split_used_for_threshold_selection": False,
            "calibration_split_unseen_by_i1_i2": True,
            "i1_success_is_not_defined_by_i1_reliability": True,
            "label_dependent_subsets_are_diagnostic_only": True,
        },
        "calibration_split": "val_selection",
        "target_split": "test_clean",
        "diagnostics": "results/gate_diagnostics.csv",
        "diagnostics_sha256": run._sha256(diagnostics),
        "test_csv": "labels/test.csv",
        "test_csv_sha256": run._sha256(test_csv),
        "subsets": subset_records,
    }
    (subset_dir / "subset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    run.validate_natural_subset_artifacts(tmp_path)
    subset_path = subset_dir / run.NATURAL_SUBSET_FILES[0]
    original_subset = subset_path.read_text(encoding="utf-8")
    subset_path.write_text("sha256,label\na,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CSV changed after generation"):
        run.validate_natural_subset_artifacts(tmp_path)
    subset_path.write_text(original_subset, encoding="utf-8")

    test_csv.write_text("sha256,label\na,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed"):
        run.validate_natural_subset_artifacts(tmp_path)


def test_decision_only_configs_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    for relative in (
        "appendix/risk_level_0_03_eval.yaml",
        "appendix/risk_level_0_10_eval.yaml",
        "ablations/i3/marginal_conformal.yaml",
        "ablations/i3/deployed_class_probability_threshold.yaml",
        "ablations/i3/msp_threshold.yaml",
        "ablations/i3/predictive_entropy_threshold.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_decision_calibration"] is True
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_catalog_excludes_the_inheritance_only_main_template():
    template = (
        run.CONFIG_DIR / "evidential_trusted_fusion.yaml"
    ).resolve()
    assert template not in {
        path.resolve() for path in run.resolve_targets("all")
    }
    assert "template" not in run.ALIASES


def test_non_seed_experiments_have_unique_resolved_behavior():
    seen: dict[str, Path] = {}
    seed_paths = {path.resolve() for path in run.resolve_targets("seed")}
    for path in run.resolve_targets("all"):
        if path.resolve() in seed_paths:
            continue
        key = _semantic_key(_resolved(path))
        previous = seen.get(key)
        assert previous is None, f"{path} duplicates {previous}"
        seen[key] = path


def test_legacy_graph_budget_checkpoint_config_is_rejected():
    current = _resolved(ROOT / "seeds/seed_42.yaml")
    old = yaml.safe_load(yaml.safe_dump(current))
    old["model"]["graph_encoder"]["account_for_encoder_budget"] = True
    old["fusion"].pop("force_fp32_decision")
    old["calibration"].pop("stratified_group_split")
    with pytest.raises(ValueError, match="model/data semantics"):
        validate_eval_checkpoint_config(current, old)


def test_checkpoint_signature_rejects_different_training_objective_or_protocol():
    current = _resolved(ROOT / "baselines/trusted/tmc_style_adapted.yaml")

    wrong_objective = yaml.safe_load(yaml.safe_dump(current))
    wrong_objective["loss"]["objective"] = "standard"
    with pytest.raises(ValueError, match="model/data semantics"):
        validate_eval_checkpoint_config(current, wrong_objective)

    wrong_protocol = yaml.safe_load(yaml.safe_dump(current))
    wrong_protocol["method"]["protocol_id"] = "legacy_generic_edl"
    with pytest.raises(ValueError, match="model/data semantics"):
        validate_eval_checkpoint_config(current, wrong_protocol)


def test_style_adapted_baseline_identity_exposes_the_declared_protocol():
    tmc = _resolved(ROOT / "baselines/trusted/tmc_style_adapted.yaml")
    ecml = _resolved(ROOT / "baselines/trusted/ecml_style_adapted.yaml")
    tmc_identity = build_run_identity(tmc, tmc["train"]["exp_name"], 42)
    ecml_identity = build_run_identity(ecml, ecml["train"]["exp_name"], 42)

    assert tmc_identity["method_name"] == "tmc_style_adapted"
    assert tmc_identity["method_protocol_id"] == "tmc_style_adapted_v1"
    assert tmc_identity["training_objective"] == "tmc"
    assert tmc_identity["evidential_anneal_epochs"] == 10
    assert tmc_identity["ecml_consistency_weight"] is None
    assert ecml_identity["method_name"] == "ecml_style_adapted"
    assert ecml_identity["method_protocol_id"] == "ecml_style_adapted_v1"
    assert ecml_identity["training_objective"] == "ecml"
    assert ecml_identity["ecml_consistency_weight"] == pytest.approx(1.0)


def test_method_level_baseline_guards_reject_semantic_drift():
    tmc = _resolved(ROOT / "baselines/trusted/tmc_style_adapted.yaml")
    tmc["selective_prediction"]["enabled"] = True
    with pytest.raises(ValueError, match="style-adapted fusion path"):
        build_model(tmc, feature_dim=16)

    qmf = _resolved(ROOT / "baselines/trusted/qmf_energy.yaml")
    qmf["model"]["quality_fusion_temperature"] = 5.0
    with pytest.raises(ValueError, match="QMF-Energy component baseline"):
        build_model(qmf, feature_dim=16)


@pytest.mark.parametrize(
    "removed_alias",
    (
        "tmc_faithful",
        "tmc_style",
        "ecml_faithful",
        "ecml_style",
        "ecml_adapted",
        "ecml_inspired",
        "qmf_style",
    ),
)
def test_obsolete_literature_baseline_aliases_fail_explicitly(removed_alias):
    with pytest.raises(ValueError, match="method identity.*ambiguous"):
        run.resolve_targets(removed_alias)


def test_style_adapted_baseline_aliases_resolve_explicitly():
    assert run.resolve_targets("tmc_style_adapted") == [
        ROOT / "baselines/trusted/tmc_style_adapted.yaml"
    ]
    assert run.resolve_targets("ecml_style_adapted") == [
        ROOT / "baselines/trusted/ecml_style_adapted.yaml"
    ]


def test_eval_checkpoint_requires_current_implementation_fingerprint():
    current_hash = build_run_identity(
        _resolved(ROOT / "seeds/seed_42.yaml"), "seed", 42
    )["method_implementation_sha256"]
    validate_checkpoint_implementation(
        {"method_implementation_sha256": current_hash}
    )
    with pytest.raises(ValueError, match="predates implementation fingerprinting"):
        validate_checkpoint_implementation({})
    with pytest.raises(ValueError, match="different model/fusion implementation"):
        validate_checkpoint_implementation(
            {"method_implementation_sha256": "0" * 64}
        )
