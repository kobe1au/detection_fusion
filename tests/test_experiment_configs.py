from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import run
from fusion.train import (
    _extra_eval_paths,
    _dataset_common_kwargs,
    build_run_identity,
    build_model,
    load_config,
    load_config_path,
    reliability_calibration_scenarios,
    uses_routing_calibration_scenarios,
    validate_checkpoint_implementation,
    validate_eval_checkpoint_config,
)


ROOT = Path("config/experiments/tri_modal_robust")
MAIN = ROOT / "evidential_trusted_fusion.yaml"


def _resolved(path: Path) -> dict:
    return load_config_path(path)


def _semantic_key(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.pop("method", None)
    normalized.get("train", {}).pop("exp_name", None)
    normalized.get("train", {}).pop("seed", None)
    normalized.get("eval", {}).pop("output_name", None)
    return json.dumps(normalized, sort_keys=True, default=str)


# ── Generic resolve/build/protocol checks ─────────────────────────────────────

def test_all_runnable_experiment_configs_resolve_and_build():
    paths = run.resolve_targets("all")
    assert paths
    for path in paths:
        cfg = _resolved(path)
        _dataset_common_kwargs(cfg, is_train=False)
        build_model(cfg, feature_dim=515)


def test_runner_groups_and_aliases_reference_existing_configs():
    for name in run.GROUPS:
        assert run.resolve_targets(name)
    for name in run.ALIASES:
        assert run.resolve_targets(name)


def test_formal_configs_use_clean_validation_selection():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        eval_cfg = cfg.get("eval", {}) or {}
        if eval_cfg.get("eval_only", False):
            continue
        assert cfg["train"]["checkpoint_metric"] == "clean_macro_f1", path
        assert eval_cfg["robust_val"]["enabled"] is False, path


def test_runnable_configs_share_validation_protocol():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        calibration = cfg.get("calibration", {}) or {}
        assert cfg["train"]["strict_deterministic"] is False, path
        assert calibration.get("validation_fraction") == 0.5, path
        assert calibration.get("conformal_fraction") == 0.5, path
        assert calibration.get("split_seed") == 42, path
        assert calibration.get("stratified_group_split") is True, path
        assert cfg["model"]["graph_encoder"]["account_for_encoder_budget"] is True, path
        assert cfg["fusion"]["force_fp32_decision"] is True, path


def test_main_reliability_calibration_covers_declared_deployment_degradation_range():
    cfg = _resolved(MAIN)
    scenarios = reliability_calibration_scenarios(cfg)
    assert len(scenarios) == 23
    assert {item["perturb_type"] for item in scenarios} == {
        "api_degraded",
        "graph_degraded",
        "manifest_degraded",
        "all_degraded",
        "api_missing",
        "graph_missing",
        "manifest_missing",
    }
    assert {item["strength"] for item in scenarios} == {
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        1.0,
    }
    api_views = [item for item in scenarios if item["perturb_type"] == "api_degraded"]
    assert all(item["reliability_branches"] == ["api"] for item in api_views)
    missing_views = [item for item in scenarios if item["perturb_type"].endswith("_missing")]
    assert len(missing_views) == 3
    assert all(item["strength"] == 1.0 for item in missing_views)
    assert all(item["scenario_group"] == "missing" for item in missing_views)
    assert all(item["reliability_branches"] == [] for item in missing_views)

    # The no-training-augmentation ablation changes only encoder training.
    # Its post-hoc router receives the same transformed views so that this
    # ablation does not silently remove two mechanisms at once.
    no_aug = copy.deepcopy(cfg)
    no_aug["robust"]["train_aug"] = False
    assert reliability_calibration_scenarios(no_aug) == scenarios


def test_only_routed_methods_consume_transformed_posthoc_views():
    assert uses_routing_calibration_scenarios(_resolved(MAIN)) is True
    assert uses_routing_calibration_scenarios(
        _resolved(ROOT / "ablations/modules/no_reliability_discount.yaml")
    ) is True
    for relative in (
        "ablations/modules/no_i2_conflict_aware_fusion.yaml",
        "ablations/i2/combination_dempster.yaml",
        "ablations/i2/combination_cumulative.yaml",
        "ablations/i2/combination_log_pool.yaml",
        "ablations/i2/combination_ecml_style.yaml",
    ):
        assert uses_routing_calibration_scenarios(_resolved(ROOT / relative)) is False


def test_seed_42_posthoc_refit_reuses_staged_encoder_and_refits_decisions():
    cfg = _resolved(ROOT / "appendix/refit_seed_42_posthoc.yaml")
    assert run.resolve_targets("posthoc_pilot") == [
        ROOT / "appendix/refit_seed_42_posthoc.yaml"
    ]
    assert cfg["eval"]["eval_only"] is True
    assert cfg["eval"]["refit_posthoc_calibration"] is True
    assert cfg["eval"]["refit_decision_calibration"] is True
    assert cfg["eval"].get("allow_checkpoint_config_mismatch", False) is False
    assert cfg["eval"]["output_name"] == "refit_seed_42_posthoc"
    assert cfg["calibration"]["epochs"] == 60
    assert cfg["fusion"]["reliability_calibration"]["group_mean_alignment"] is True
    assert cfg["fusion"]["visible_integrity_modifier"]["enabled"] is True
    assert (
        cfg["fusion"]["visible_integrity_modifier"]["mode"]
        == "relative_effective"
    )


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
    assert cfg["model"]["fusion_mode"] == "discount_probability"
    # I1: observable reliability/evidence comparability
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["reliability_calibration"]["branches"] == [
        "api",
        "graph",
        "manifest",
    ]
    assert cfg["fusion"]["reliability_calibration"]["use_evidential_uncertainty"] is True
    assert cfg["loss"]["evidential_loss_weight"] > 0.0
    assert cfg["loss"]["evidential"]["class_weight"] == "balanced"
    # I2: post-hoc global opinion routing with an explicit unknown mass.
    assert cfg["fusion"]["combination"] == "routed"
    assert cfg["fusion"]["routing"]["enabled"] is True
    assert cfg["fusion"]["routing"]["use_fused_prediction_loss"] is True
    assert cfg["fusion"]["routing"]["final_temperature_scaling"] is True
    assert cfg["fusion"]["probability_calibration"]["enabled"] is False
    assert cfg["fusion"]["reliability_calibration"]["use_model_visibility"] is True
    assert cfg["fusion"]["visible_integrity_modifier"]["enabled"] is True
    assert cfg["fusion"]["visible_integrity_modifier"]["mode"] == "relative_effective"
    assert cfg["calibration"]["perturb_strengths"] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert cfg["fusion"]["use_confidence_proxy"] is False
    # I3: calibration-set malware false-negative risk control.
    assert cfg["classification_threshold"]["enabled"] is True
    assert cfg["classification_threshold"]["objective"] == "macro_f1"
    assert cfg["classification_threshold"]["min_malware_recall"] == 0.90
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
    assert identity["combination_rule"] == "routed"
    assert identity["global_opinion_routing_enabled"] is True
    assert identity["routing_posthoc_fused_prediction_loss_enabled"] is True
    assert identity["routed_final_temperature_enabled"] is True
    assert identity["reliability_calibration_branches"] == [
        "api",
        "graph",
        "manifest",
    ]
    assert identity["router_trained_end_to_end"] is True
    assert identity["router_posthoc_refinement_enabled"] is True
    assert identity["router_encoder_training_reliability_source"] == "observable_integrity"
    assert identity["router_posthoc_reliability_source"] == "calibrated_branch_correctness"
    assert identity["model_visibility_reliability_enabled"] is True
    assert identity["evidential_certainty_enabled"] is True
    assert identity["classification_threshold_enabled"] is True
    assert identity["classification_threshold_objective"] == "macro_f1"
    assert identity["classification_min_malware_recall"] == 0.90
    assert identity["risk_control_level"] == 0.05
    assert identity["selective_score_type"] == "model_acceptance"


def test_disabled_baseline_components_are_reported_as_inactive():
    cfg = _resolved(ROOT / "baselines/tri_modal_concat.yaml")
    identity = build_run_identity(cfg, "baseline_tri_modal_concat", 42)

    assert identity["reliability_calibration_enabled"] is False
    assert identity["reliability_calibration_branches"] == []
    assert identity["relation_evidence_enabled"] is False
    assert identity["evidential_certainty_enabled"] is False
    assert identity["model_visibility_reliability_enabled"] is False
    assert identity["classification_threshold_objective"] == "disabled"
    assert identity["selective_prediction_mode"] == "disabled"
    assert identity["selective_score_type"] == "disabled"


def test_non_routed_i2_ablation_does_not_report_routed_components_as_active():
    cfg = _resolved(ROOT / "ablations/i2/combination_dempster.yaml")
    identity = build_run_identity(cfg, "i2_combination_dempster", 42)
    assert identity["global_opinion_routing_enabled"] is False
    assert identity["routing_posthoc_fused_prediction_loss_enabled"] is False
    assert identity["routed_final_temperature_enabled"] is False


def test_i2_rule_replacements_keep_main_i1_without_competence_prior():
    for relative in (
        "ablations/i2/combination_dempster.yaml",
        "ablations/i2/combination_cumulative.yaml",
        "ablations/i2/combination_log_pool.yaml",
        "ablations/i2/combination_ecml_style.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["branch_competence_prior"]["enabled"] is False
        assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
        assert cfg["fusion"]["visible_integrity_modifier"]["enabled"] is True


def test_method_protocol_identity_is_seed_invariant():
    seed_42 = _resolved(ROOT / "seeds/seed_42.yaml")
    seed_2024 = _resolved(ROOT / "seeds/seed_2024.yaml")
    identity_42 = build_run_identity(seed_42, "evidential_seed_42", 42)
    identity_2024 = build_run_identity(seed_2024, "evidential_seed_2024", 2024)

    assert identity_42["resolved_config_sha256"] != identity_2024["resolved_config_sha256"]
    assert identity_42["method_protocol_sha256"] == identity_2024["method_protocol_sha256"]


# ── Baselines ─────────────────────────────────────────────────────────────────

def test_baselines_disable_reliability_calibration_and_rejection():
    main = _resolved(MAIN)
    for relative in ("baselines/api_only.yaml", "baselines/graph_only.yaml",
                     "baselines/manifest_only.yaml", "baselines/tri_modal_concat.yaml",
                     "baselines/fixed_logit_fusion.yaml", "baselines/api_graph_concat.yaml"):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["mode"] == "model_dispatch", relative
        assert cfg["fusion"]["branch_competence_prior"]["enabled"] is False, relative
        assert cfg["fusion"]["visible_integrity_modifier"]["enabled"] is False, relative
        assert cfg["calibration"]["enabled"] is False, relative
        assert cfg["calibration"]["holdout_enabled"] is True, relative
        assert cfg["selective_prediction"]["enabled"] is False, relative
        # Baselines share the same training augmentation as the main method.
        assert cfg["robust"] == main["robust"], relative


def test_recent_trusted_fusion_baselines_do_not_reuse_ours_i1_or_i3():
    expected = {
        "baselines/trusted/tmc_style.yaml": ("discount_probability", "dempster"),
        "baselines/trusted/ecml_style.yaml": ("discount_probability", "ecml_style"),
        "baselines/trusted/qmf_style.yaml": ("tri_modal_quality_fusion", "linear"),
    }
    assert run.resolve_targets("trusted_baselines") == [
        run.CONFIG_DIR / relative for relative in run.TRUSTED_FUSION_BASELINES
    ]
    for relative, (mode, combination) in expected.items():
        cfg = _resolved(ROOT / relative)
        assert cfg["model"]["fusion_mode"] == mode, relative
        assert cfg["fusion"].get("combination", "linear") == combination, relative
        assert cfg["fusion"]["reliability_calibration"]["enabled"] is False, relative
        assert cfg["fusion"]["branch_competence_prior"]["enabled"] is False, relative
        assert cfg["fusion"]["visible_integrity_modifier"]["enabled"] is False, relative
        assert cfg["selective_prediction"]["enabled"] is False, relative


# ── Module / mechanism ablations ──────────────────────────────────────────────

def test_module_ablations_remove_whole_innovations():
    no_i1 = _resolved(ROOT / "ablations/modules/no_reliability_discount.yaml")
    assert no_i1["fusion"].get("opinion_source", "evidential") == "evidential"
    assert no_i1["fusion"]["combination"] == "routed"
    assert no_i1["fusion"]["routing"]["enabled"] is True
    assert no_i1["fusion"]["use_reliability_discount"] is False
    assert no_i1["fusion"]["branch_competence_prior"]["enabled"] is False
    assert no_i1["fusion"]["visible_integrity_modifier"]["enabled"] is False
    assert no_i1["fusion"]["reliability_calibration"]["enabled"] is False
    assert no_i1["calibration"]["enabled"] is True
    assert no_i1["loss"]["integrity_weighted_aux"] is True
    assert no_i1["loss"]["reliability_weighted_aux"] is None
    assert no_i1["loss"]["evidential_loss_weight"] == 0.05

    no_i2 = _resolved(ROOT / "ablations/i2/router_prior_only.yaml")
    assert no_i2["fusion"].get("opinion_source", "evidential") == "evidential"
    assert no_i2["fusion"]["combination"] == "routed"
    assert no_i2["fusion"]["routing"]["enabled"] is True
    assert no_i2["fusion"]["routing"]["mode"] == "prior_only"
    assert no_i2["fusion"]["routing"]["train_end_to_end"] is False
    assert no_i2["fusion"]["routing"]["posthoc_refine"] is False
    assert no_i2["loss"]["evidential_loss_weight"] == 0.05

    no_i3 = _resolved(ROOT / "ablations/modules/no_i3_selective_rejection.yaml")
    assert no_i3["selective_prediction"]["enabled"] is False
    assert no_i3["classification_threshold"]["enabled"] is False
    assert no_i3["fusion"]["combination"] == "routed"
    assert no_i3["eval"]["eval_only"] is True
    assert no_i3["eval"]["refit_rejection_threshold"] is False

    no_rel = _resolved(ROOT / "ablations/modules/no_reliability_discount.yaml")
    assert no_rel["fusion"].get("opinion_source", "evidential") == "evidential"
    assert no_rel["fusion"]["combination"] == "routed"
    assert no_rel["fusion"]["routing"]["enabled"] is True
    assert no_rel["fusion"]["use_reliability_discount"] is False
    assert no_rel["fusion"]["reliability_calibration"]["enabled"] is False
    assert no_rel["loss"]["evidential_loss_weight"] == 0.05


def test_i1_mechanism_ablation_removes_relation_evidence():
    cfg = _resolved(ROOT / "ablations/i1/no_relation_evidence.yaml")
    assert cfg["fusion"]["reliability_calibration"]["use_relation_evidence"] is False
    assert cfg["fusion"]["combination"] == "routed"
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True


def test_i1_observable_only_ablation_removes_learned_certainty_source():
    cfg = _resolved(ROOT / "ablations/i1/observable_only_reliability.yaml")
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["reliability_calibration"]["use_relation_evidence"] is True
    assert cfg["fusion"]["reliability_calibration"]["use_evidential_uncertainty"] is False


def test_training_ablation_removes_edl_supervision_not_i1_reliability():
    cfg = _resolved(ROOT / "ablations/training/no_edl_supervision.yaml")
    assert cfg["loss"]["evidential_loss_weight"] == 0.0
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["reliability_calibration"]["use_evidential_uncertainty"] is False
    assert cfg["fusion"]["combination"] == "routed"


def test_i2_mechanism_ablations_switch_combination_rule():
    for relative, rule in (
        ("ablations/i2/combination_dempster.yaml", "dempster"),
        ("ablations/i2/combination_cumulative.yaml", "cumulative"),
        ("ablations/i2/combination_log_pool.yaml", "log_pool"),
        ("ablations/i2/combination_ecml_style.yaml", "ecml_style"),
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["combination"] == rule, relative
        assert cfg["fusion"]["routing"]["enabled"] is False, relative


def test_i3_mechanism_ablations_are_decision_only():
    class_conditional = _resolved(
        ROOT / "ablations/i3/class_conditional_conformal.yaml"
    )
    assert class_conditional["selective_prediction"]["mode"] == "conformal"
    assert class_conditional["selective_prediction"]["class_conditional"] is True
    assert class_conditional["eval"]["eval_only"] is True
    assert class_conditional["eval"]["refit_rejection_threshold"] is True

    marginal = _resolved(ROOT / "ablations/i3/marginal_conformal.yaml")
    assert marginal["selective_prediction"]["class_conditional"] is False
    assert marginal["eval"]["eval_only"] is True
    assert marginal["eval"]["refit_rejection_threshold"] is True

    conflict = _resolved(ROOT / "ablations/i3/conflict_augmented_conformal.yaml")
    assert conflict["selective_prediction"]["use_raw_conflict"] is True
    assert conflict["eval"]["eval_only"] is True
    assert conflict["eval"]["refit_rejection_threshold"] is True

    threshold = _resolved(ROOT / "ablations/i3/threshold_rejection.yaml")
    assert threshold["selective_prediction"]["mode"] == "threshold"
    assert threshold["selective_prediction"]["threshold_score"] == "max_probability"
    assert threshold["eval"]["eval_only"] is True

    uncertainty = _resolved(ROOT / "ablations/i3/uncertainty_threshold.yaml")
    assert uncertainty["selective_prediction"]["mode"] == "threshold"
    assert uncertainty["selective_prediction"]["threshold_score"] == "evidential_certainty"
    assert uncertainty["eval"]["eval_only"] is True


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
    assert risk_03["eval"]["refit_rejection_threshold"] is True
    risk_05 = _resolved(ROOT / "appendix/risk_level_0_05_eval.yaml")
    assert risk_05["selective_prediction"]["risk_level"] == 0.05
    assert risk_05["eval"]["eval_only"] is True
    assert risk_05["eval"]["refit_rejection_threshold"] is True


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
        expected["method"] = {"name": f"evidential_seed_{seed}"}
        expected["train"]["exp_name"] = f"evidential_seed_{seed}"
        expected["train"]["seed"] = seed
        assert resolved == expected, seed


def test_external_obfuscapk_eval_configs_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    assert run.resolve_targets("external") == [run.CONFIG_DIR / item for item in run.EXTERNAL_EVAL]
    for relative in run.EXTERNAL_EVAL:
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["run_test"] is False
        assert cfg["eval"]["run_robust_test"] is False
        # External Obfuscapk evaluation should reuse the thresholds stored in
        # the seed-42 checkpoint. Re-fitting here would make the external
        # configs depend on local clean val paths, which are not needed for
        # eval-only robustness audits.
        assert cfg["eval"]["refit_rejection_threshold"] is False
        assert "evidential_seed_42/42/best_tri_modal_robust.pt" in cfg["eval"]["checkpoint_path"]
        assert cfg["eval"].get("extra_sets"), relative
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_natural_subset_eval_configs_share_the_same_curated_csvs():
    expected_csvs = [
        "labels/natural_subsets/test_api_low_effective_integrity.csv",
        "labels/natural_subsets/test_api_graph_low_support.csv",
        "labels/natural_subsets/test_predictive_high_conflict.csv",
        "labels/natural_subsets/test_low_acceptance.csv",
    ]
    assert run.resolve_targets("natural_subset") == [
        run.CONFIG_DIR / item for item in run.NATURAL_SUBSET_EVAL
    ]
    for relative in run.NATURAL_SUBSET_EVAL:
        cfg = _resolved(ROOT / relative)
        eval_cfg = cfg["eval"]
        assert eval_cfg["eval_only"] is True, relative
        assert eval_cfg["run_test"] is False, relative
        assert eval_cfg["run_robust_test"] is False, relative
        assert eval_cfg["refit_rejection_threshold"] is False, relative
        extra_sets = eval_cfg.get("extra_sets") or []
        if isinstance(extra_sets, dict):
            extra_items = list(extra_sets.values())
        else:
            extra_items = list(extra_sets)
        actual_csvs = [item["csv"] for item in extra_items]
        # Low acceptance is defined by the proposed method and is therefore an
        # I3 diagnostic, not a neutral baseline-comparison subset. Predictive
        # conflict is also fixed once from seed 42 before any baseline is
        # evaluated and must never be re-selected per method.
        expected_for_method = (
            expected_csvs
            if relative in run.NATURAL_SUBSET_OURS_EVAL
            else expected_csvs[:3]
        )
        assert actual_csvs == expected_for_method, relative
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
        "appendix/risk_level_0_05_eval.yaml",
        "ablations/i3/marginal_conformal.yaml",
        "ablations/i3/threshold_rejection.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_rejection_threshold"] is True
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_temperature_scaling_ablation_reuses_router_and_refits_decision_thresholds():
    cfg = _resolved(ROOT / "seeds/temperature_scaling_false_eval.yaml")
    assert cfg["fusion"]["routing"]["final_temperature_scaling"] is True
    assert cfg["eval"]["eval_only"] is True
    assert cfg["eval"]["refit_posthoc_calibration"] is False
    assert cfg["eval"]["final_temperature_override"] == 1.0
    assert cfg["eval"]["allow_checkpoint_config_mismatch"] is True
    assert cfg["eval"]["refit_rejection_threshold"] is True


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


def test_pre_fix_checkpoint_config_is_rejected():
    current = _resolved(ROOT / "seeds/seed_42.yaml")
    old = yaml.safe_load(yaml.safe_dump(current))
    old["model"]["graph_encoder"].pop("account_for_encoder_budget")
    old["fusion"].pop("force_fp32_decision")
    old["calibration"].pop("stratified_group_split")
    with pytest.raises(ValueError, match="model/data semantics"):
        validate_eval_checkpoint_config(current, old)


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
    validate_checkpoint_implementation({}, allow_mismatch=True)
