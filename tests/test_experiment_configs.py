from __future__ import annotations

import copy
import json
from pathlib import Path

import run
import pytest
import yaml
from fusion.train import (
    _dataset_common_kwargs,
    build_model,
    load_config,
    load_config_path,
    validate_eval_checkpoint_config,
)


ROOT = Path("config/experiments/tri_modal_robust")


def _resolved(path: Path) -> dict:
    return load_config_path(path)


def _semantic_key(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.pop("method", None)
    normalized.get("train", {}).pop("exp_name", None)
    normalized.get("train", {}).pop("seed", None)
    normalized.get("eval", {}).pop("output_name", None)
    return json.dumps(normalized, sort_keys=True, default=str)


def _run_key(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.get("train", {}).pop("exp_name", None)
    return json.dumps(normalized, sort_keys=True, default=str)


def test_all_runnable_experiment_configs_resolve_and_build():
    paths = run.resolve_targets("all")
    assert paths
    for path in paths:
        cfg = _resolved(path)
        _dataset_common_kwargs(cfg, is_train=False)
        build_model(cfg, feature_dim=515)


def test_experiment_pt_paths_match_current_builder_output():
    build_cfg = yaml.safe_load(Path("config/build_pts.yaml").read_text(encoding="utf-8"))
    out_root = str(build_cfg["data"]["out_root"]).replace("\\", "/").rstrip("/")
    cfg = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    for split in ("train", "val", "test"):
        configured = str(cfg["data"][f"{split}_pt_dir"]).replace("\\", "/").rstrip("/")
        assert configured == f"{out_root}/{split}"


def test_curated_csv_can_select_a_strict_subset_of_current_pts():
    cfg = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    kwargs = _dataset_common_kwargs(cfg, is_train=True)
    assert kwargs["strict_split_integrity"] is True
    assert kwargs["allow_pt_superset"] is True


def test_runner_groups_and_aliases_reference_existing_configs():
    for name in run.GROUPS:
        assert run.resolve_targets(name)
    for name in run.ALIASES:
        assert run.resolve_targets(name)


def test_tuning_groups_are_validation_only():
    for group in ("tuning_base", "tuning_i1", "tuning_i2", "tuning_i3", "tuning_objective"):
        for path in run.resolve_targets(group):
            cfg = _resolved(path)
            eval_cfg = cfg.get("eval", {}) or {}
            assert eval_cfg.get("run_test") is False, path
            assert eval_cfg.get("run_robust_test") is False, path
            if eval_cfg.get("eval_only", False):
                expected = (
                    "tuning_full_candidate/42/best_tri_modal_robust.pt"
                    if group == "tuning_i3"
                    else "final_seed_42/42/best_tri_modal_robust.pt"
                )
                assert expected in eval_cfg["checkpoint_path"], path
                assert eval_cfg.get("refit_rejection_threshold") is True, path
            else:
                assert cfg["train"].get("tuning_mode") is True, path


def test_formal_training_configs_use_clean_validation_selection():
    for path in run.resolve_targets("all"):
        relative = path.relative_to(run.CONFIG_DIR).as_posix()
        cfg = _resolved(path)
        eval_cfg = cfg.get("eval", {}) or {}
        if relative.startswith("tuning/") or eval_cfg.get("eval_only", False):
            continue
        assert cfg["train"]["checkpoint_metric"] == "clean_macro_f1", path
        assert eval_cfg["robust_val"]["enabled"] is False, path


def test_tuning_training_configs_use_clean_checkpoint_then_final_robust_validation():
    for group in ("tuning_base", "tuning_i1", "tuning_i2", "tuning_objective"):
        for path in run.resolve_targets(group):
            cfg = _resolved(path)
            assert cfg["train"]["checkpoint_metric"] == "clean_macro_f1", path
            assert cfg["eval"]["robust_val"]["enabled"] is True, path


def test_baselines_share_full_training_augmentation():
    full = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    expected = full["robust"]
    for relative in run.BASELINES:
        cfg = _resolved(ROOT / relative)
        assert cfg["robust"]["train_aug"] is True, relative
        assert cfg["robust"]["perturb_prob"] == expected["perturb_prob"], relative
        assert cfg["robust"]["perturb_strengths"] == expected["perturb_strengths"], relative


def test_decision_eval_configs_have_unique_output_names():
    paths = [
        *run.resolve_targets("tuning_i3"),
        *run.resolve_targets("sensitivity"),
        *run.resolve_targets("external"),
    ]
    output_names: dict[str, Path] = {}
    for path in paths:
        cfg = _resolved(path)
        eval_cfg = cfg.get("eval", {}) or {}
        if not eval_cfg.get("eval_only", False):
            continue
        output_name = str(eval_cfg.get("output_name", "")).strip()
        assert output_name, path
        assert output_name not in output_names, f"{path} duplicates {output_names.get(output_name)}"
        output_names[output_name] = path

def test_section_groups_do_not_rerun_full_method():
    final_path = (run.CONFIG_DIR / run.FINAL).resolve()
    section_groups = (
        "module",
        "i1",
        "i1_appendix",
        "i1_full",
        "i2",
        "i2_appendix",
        "i2_full",
        "i3",
        "i3_appendix",
        "i3_full",
        "component",
        "training_ablation",
        "training_ablation_appendix",
        "training_ablation_full",
        "sensitivity",
        "external",
    )
    for group in section_groups:
        paths = {path.resolve() for path in run.resolve_targets(group)}
        assert final_path not in paths, group


def test_per_module_tuning_groups_do_not_rerun_tuning_base():
    tuning_base = (run.CONFIG_DIR / run.TUNING_FULL).resolve()
    for group in ("tuning_i1", "tuning_i2", "tuning_i3"):
        paths = {path.resolve() for path in run.resolve_targets(group)}
        assert tuning_base not in paths, group


def test_paper_plan_excludes_tuning_configs():
    tuning_paths = {
        path.resolve()
        for group in ("tuning_base", "tuning_i1", "tuning_i2", "tuning_i3", "tuning_objective")
        for path in run.resolve_targets(group)
    }
    for group in ("paper", "paper_main", "paper_appendix", "paper_all"):
        paper_paths = {path.resolve() for path in run.resolve_targets(group)}
        assert not (tuning_paths & paper_paths)


def test_runner_groups_do_not_repeat_equivalent_runs():
    for name in run.GROUPS:
        seen: dict[str, Path] = {}
        for path in run.resolve_targets(name):
            key = _run_key(_resolved(path))
            assert key not in seen, f"group {name}: {path} duplicates {seen.get(key)}"
            seen[key] = path



def test_tuning_i3_reuses_refreeze_full_candidate_checkpoint():
    candidate_cfg = _resolved(ROOT / "tuning/full_candidate.yaml")
    for path in run.resolve_targets("tuning_i3"):
        cfg = _resolved(path)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_rejection_threshold"] is True
        assert "tuning_full_candidate/42/best_tri_modal_robust.pt" in cfg["eval"]["checkpoint_path"]
        validate_eval_checkpoint_config(cfg, candidate_cfg)

def test_decision_only_sensitivities_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    for relative in (
        "sensitivity/i3/acceptance_min.yaml",
        "sensitivity/i3/coverage_80.yaml",
        "sensitivity/i3/coverage_95.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_rejection_threshold"] is True
        assert "final_seed_42/42/best_tri_modal_robust.pt" in cfg["eval"]["checkpoint_path"]
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_external_obfuscapk_eval_configs_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    for relative in run.EXTERNAL_EVAL:
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["run_test"] is False
        assert cfg["eval"]["run_robust_test"] is False
        assert cfg["eval"]["refit_rejection_threshold"] is True
        assert "final_seed_42/42/best_tri_modal_robust.pt" in cfg["eval"]["checkpoint_path"]
        assert cfg["eval"].get("extra_sets"), relative
        extra = cfg["eval"]["extra_sets"][0]
        assert extra["skip_if_empty"] is False
        assert extra["pt_dir"].startswith("../../pts_obfuscapk/")
        assert extra["csv"].startswith("labels/obfuscapk_")
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_runnable_configs_share_validation_selection_protocol():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        calibration = cfg.get("calibration", {}) or {}
        assert cfg["train"]["strict_deterministic"] is False, path
        assert calibration.get("validation_fraction") == 0.5, path
        assert calibration.get("split_seed") == 42, path
        assert calibration.get("stratified_group_split") is True, path
        assert cfg["model"]["graph_encoder"]["account_for_encoder_budget"] is True, path
        assert cfg["fusion"]["force_fp32_decision"] is True, path
        assert (
            calibration.get("enabled")
            or calibration.get("holdout_enabled")
            or (cfg.get("selective_prediction", {}) or {}).get("enabled")
        ), path



def test_pre_fix_checkpoint_config_is_rejected():
    current = _resolved(ROOT / "seeds/seed_42.yaml")
    old = yaml.safe_load(yaml.safe_dump(current))
    old["model"]["graph_encoder"].pop("account_for_encoder_budget")
    old["fusion"].pop("force_fp32_decision")
    old["calibration"].pop("stratified_group_split")

    with pytest.raises(ValueError, match="model/data semantics"):
        validate_eval_checkpoint_config(current, old)

def test_non_discount_baselines_do_not_enable_calibration_or_rejection():
    for relative in run.BASELINES:
        cfg = _resolved(ROOT / relative)
        if cfg["model"]["fusion_mode"] == "discount_probability":
            continue
        assert cfg["fusion"]["mode"] == "legacy_learned_gate"
        assert cfg["calibration"]["enabled"] is False
        assert cfg["calibration"]["holdout_enabled"] is True
        assert cfg["selective_prediction"]["enabled"] is False


def test_module_ablations_match_paper_claims():
    no_i1 = _resolved(ROOT / "ablations/modules/no_i1_observable_reliability.yaml")
    assert no_i1["fusion"]["reliability_calibration"]["enabled"] is False
    assert no_i1["fusion"]["use_reliability_discount"] is False
    assert no_i1["fusion"]["use_support_discount"] is False
    assert no_i1["fusion"]["use_conflict_discount"] is False
    assert no_i1["fusion"]["use_hard_alive_mask"] is False
    assert no_i1["semantic_cross_attention"]["use_reliability_bias"] is False
    assert no_i1["semantic_cross_attention"]["use_support_bias"] is False
    assert no_i1["semantic_cross_attention"]["use_conflict_bias"] is False
    assert no_i1["semantic_cross_attention"]["use_relation_mask"] is False
    assert no_i1["semantic_reconstruction"]["use_integrity_conditioning"] is False
    assert no_i1["loss"]["reliability_weighted_aux"] is False

    no_i2 = _resolved(ROOT / "ablations/modules/no_i2_semantic_interaction.yaml")
    assert no_i2["semantic_cross_attention"]["enabled"] is False
    assert no_i2["semantic_cross_attention"]["attach_to_joint"] is False
    assert no_i2["semantic_cross_attention"]["attach_to_reconstruction"] is False
    assert no_i2["semantic_reconstruction"]["enabled"] is False
    assert no_i2["semantic_reconstruction"]["weight"] == 0.0

    no_i3 = _resolved(ROOT / "ablations/modules/no_i3_discount_rejection.yaml")
    assert no_i3["method"]["name"] == "module_no_i3_discount_rejection"
    assert no_i3["model"]["fusion_mode"] == "tri_modal_ours"
    assert no_i3["fusion"]["mode"] == "legacy_learned_gate"
    assert no_i3["calibration"]["enabled"] is False
    assert no_i3["selective_prediction"]["enabled"] is False


def test_calibration_configs_have_trainable_posthoc_parameters():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        if not (cfg.get("calibration", {}) or {}).get("enabled", False):
            continue
        reliability = (cfg.get("fusion", {}) or {}).get("reliability_calibration", {}) or {}
        probability = (cfg.get("fusion", {}) or {}).get("probability_calibration", {}) or {}
        assert reliability.get("enabled", False) or probability.get("enabled", False), path


def test_enabled_masked_reconstruction_has_positive_weight():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        reconstruction = cfg.get("semantic_reconstruction", {}) or {}
        if reconstruction.get("enabled", False):
            assert float(reconstruction.get("weight", 0.0)) > 0.0, path


def test_full_method_enables_reliability_aware_semantic_cross_attention():
    cfg = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    cross_attention = cfg["semantic_cross_attention"]
    assert cross_attention["enabled"] is True
    assert cross_attention["attach_to_joint"] is True
    assert cross_attention["attach_to_reconstruction"] is True
    assert cross_attention["use_reliability_bias"] is True
    assert cross_attention["use_support_bias"] is True
    assert cross_attention["use_conflict_bias"] is True
    assert cross_attention["use_relation_mask"] is True
    assert cross_attention["num_security_tokens"] == 12
    assert cross_attention["num_residual_tokens"] > 0
    assert cross_attention["dropout"] == 0.0
    assert cross_attention["residual_gate_init"] == 0.0
    assert cross_attention["joint_residual_gate_init"] < 0.0
    assert cfg["fusion"]["reliability_calibration"]["missing_relation_support"] == 0.0
    assert cfg["fusion"]["reliability_calibration"]["use_relation_evidence"] is False
    assert cfg["calibration"]["epochs"] >= 10
    assert cfg["calibration"]["patience"] > 0


def test_i2_cross_attention_ablations_change_the_intended_mechanism():
    disabled = _resolved(ROOT / "ablations/i2/no_semantic_cross_attention.yaml")
    assert disabled["semantic_cross_attention"]["enabled"] is False

    plain = _resolved(ROOT / "ablations/i2/plain_semantic_cross_attention.yaml")
    for key in ("use_reliability_bias", "use_support_bias", "use_conflict_bias"):
        assert plain["semantic_cross_attention"][key] is False

    no_residual = _resolved(ROOT / "ablations/i2/no_cross_attention_residual_tokens.yaml")
    assert no_residual["semantic_cross_attention"]["num_residual_tokens"] == 0

    no_presence = _resolved(ROOT / "ablations/i2/no_semantic_presence_prior.yaml")
    assert no_presence["semantic_cross_attention"]["use_semantic_presence_prior"] is False

    joint_only = _resolved(ROOT / "ablations/i2/joint_only_cross_attention.yaml")
    assert joint_only["semantic_cross_attention"]["attach_to_joint"] is True
    assert joint_only["semantic_cross_attention"]["attach_to_reconstruction"] is False


def test_i1_observable_reliability_ablations_change_the_intended_mechanism():
    no_calibration = _resolved(ROOT / "ablations/i1/no_reliability_calibration.yaml")
    assert no_calibration["fusion"]["reliability_calibration"]["enabled"] is False

    integrity_only = _resolved(ROOT / "ablations/i1/integrity_alive_only.yaml")
    assert integrity_only["model"]["gate"]["use_consistency_evidence"] is False
    assert integrity_only["model"]["gate"]["use_conflict_evidence"] is False
    assert integrity_only["fusion"]["use_support_discount"] is False
    assert integrity_only["fusion"]["use_conflict_discount"] is False
    assert integrity_only["semantic_cross_attention"]["use_support_bias"] is False
    assert integrity_only["semantic_cross_attention"]["use_conflict_bias"] is False

    no_support = _resolved(ROOT / "ablations/i1/no_support_evidence.yaml")
    assert no_support["model"]["gate"]["use_consistency_evidence"] is False
    assert no_support["fusion"]["use_support_discount"] is False
    assert no_support["semantic_cross_attention"]["use_support_bias"] is False

    no_conflict = _resolved(ROOT / "ablations/i1/no_conflict_evidence.yaml")
    assert no_conflict["model"]["gate"]["use_conflict_evidence"] is False
    assert no_conflict["fusion"]["use_conflict_discount"] is False
    assert no_conflict["semantic_cross_attention"]["use_conflict_bias"] is False

    no_alive = _resolved(ROOT / "ablations/i1/no_alive_applicability_mask.yaml")
    assert no_alive["model"]["gate"]["apply_alive_mask"] is False
    assert no_alive["fusion"]["use_hard_alive_mask"] is False
    assert no_alive["fusion"]["reliability_calibration"]["apply_alive_mask"] is False
    assert no_alive["semantic_cross_attention"]["use_relation_mask"] is False


def test_i3_main_ablations_use_compact_mechanism_splits():
    no_prob = _resolved(ROOT / "ablations/i3/no_probability_calibration.yaml")
    assert no_prob["fusion"]["probability_calibration"]["enabled"] is False

    no_support_conflict = _resolved(ROOT / "ablations/i3/no_support_conflict_discount.yaml")
    assert no_support_conflict["fusion"]["use_support_discount"] is False
    assert no_support_conflict["fusion"]["use_conflict_discount"] is False

    no_confidence = _resolved(ROOT / "ablations/i3/no_confidence_proxy_discount.yaml")
    assert no_confidence["fusion"]["use_confidence_proxy"] is False

    no_rejection = _resolved(ROOT / "ablations/i3/no_selective_rejection.yaml")
    assert no_rejection["selective_prediction"]["enabled"] is False


def test_no_reconstruction_ablation_disconnects_reconstruction_path():
    cfg = _resolved(ROOT / "ablations/training/no_masked_semantic_reconstruction.yaml")
    assert cfg["semantic_reconstruction"]["enabled"] is False
    assert cfg["semantic_reconstruction"]["weight"] == 0.0
    assert cfg["semantic_cross_attention"]["attach_to_reconstruction"] is False


def test_paper_plan_has_expected_unique_runs():
    paths = run.resolve_targets("paper_all")
    expected = [
        *run.PAPER_BASELINES,
        *run.MODULE_ABLATIONS,
        *run.I1_ABLATIONS,
        *run.I1_APPENDIX_ABLATIONS,
        *run.I2_ABLATIONS,
        *run.I2_APPENDIX_ABLATIONS,
        *run.I3_ABLATIONS,
        *run.I3_APPENDIX_ABLATIONS,
        *run.TRAINING_ABLATIONS,
        *run.TRAINING_APPENDIX_ABLATIONS,
        *run.SEEDS,
        *run.SENSITIVITY,
        *run.EXTERNAL_EVAL,
    ]
    assert len(paths) == len(expected)
    assert len({path.resolve() for path in paths}) == len(paths)


def test_main_paper_plan_is_compact_and_excludes_sensitivity():
    paths = run.resolve_targets("paper")
    expected = [
        *run.PAPER_BASELINES,
        *run.MODULE_ABLATIONS,
        *run.TRAINING_ABLATIONS,
        *run.SEEDS,
    ]
    assert len(paths) == len(expected)
    assert len({path.resolve() for path in paths}) == len(paths)
    sensitivity_paths = {
        (run.CONFIG_DIR / relative).resolve() for relative in run.SENSITIVITY
    }
    assert not ({path.resolve() for path in paths} & sensitivity_paths)


def test_tuning_must_be_run_in_explicit_stages():
    assert "tuning" not in run.GROUPS
    with pytest.raises(ValueError, match="Unknown robust experiment target"):
        run.resolve_targets("tuning")
    assert "tuning_final" not in run.GROUPS
    for required in ("tuning_base", "tuning_i1", "tuning_i2", "tuning_i3", "tuning_objective"):
        assert required in run.GROUPS


def test_objective_tuning_changes_one_weight_at_a_time():
    paths = run.resolve_targets("tuning_objective")
    assert paths[0] == run.CONFIG_DIR / run.TUNING_FULL
    baseline = _resolved(paths[0])
    expected = {
        "branch_aux_weight_0_02.yaml": ("loss", "branch_aux_weight", 0.02),
        "branch_aux_weight_0_10.yaml": ("loss", "branch_aux_weight", 0.10),
        "branch_aux_weight_0_15.yaml": ("loss", "branch_aux_weight", 0.15),
        "branch_aux_weight_0_25.yaml": ("loss", "branch_aux_weight", 0.25),
        "semantic_reconstruction_weight_0_01.yaml": (
            "semantic_reconstruction", "weight", 0.01
        ),
        "semantic_reconstruction_weight_0_04.yaml": (
            "semantic_reconstruction", "weight", 0.04
        ),
        "semantic_reconstruction_weight_0_05.yaml": (
            "semantic_reconstruction", "weight", 0.05
        ),
        "semantic_reconstruction_weight_0_06.yaml": (
            "semantic_reconstruction", "weight", 0.06
        ),
    }
    for path in paths[1:]:
        cfg = _resolved(path)
        section, key, value = expected[path.name]
        assert cfg[section][key] == value
        normalized = copy.deepcopy(cfg)
        normalized[section][key] = baseline[section][key]
        normalized["method"] = baseline["method"]
        normalized["train"]["exp_name"] = baseline["train"]["exp_name"]
        assert normalized == baseline


def test_combined_objective_config_names_match_weights():
    for reconstruction_weight in (0.02, 0.03):
        suffix = f"0.2_{reconstruction_weight:.2f}"
        path = (
            ROOT
            / "tuning"
            / "objective"
            / f"tuning_objective_combined_{suffix}.yaml"
        )
        cfg = _resolved(path)
        assert cfg["method"]["name"] == f"tuning_objective_combined_{suffix}"
        assert cfg["train"]["exp_name"] == f"tuning_objective_combined_{suffix}"
        assert cfg["loss"]["branch_aux_weight"] == 0.20
        assert cfg["semantic_reconstruction"]["weight"] == reconstruction_weight


def test_paper_mechanism_group_contains_only_main_component_ablations():
    paths = run.resolve_targets("paper_mechanism")
    expected = [
        *run.I1_ABLATIONS,
        *run.I2_ABLATIONS,
        *run.I3_ABLATIONS,
    ]
    assert paths == [run.CONFIG_DIR / relative for relative in expected]
    appendix_paths = {
        (run.CONFIG_DIR / relative).resolve()
        for relative in (
            *run.I1_APPENDIX_ABLATIONS,
            *run.I2_APPENDIX_ABLATIONS,
            *run.I3_APPENDIX_ABLATIONS,
        )
    }
    assert not ({path.resolve() for path in paths} & appendix_paths)


def test_standalone_sensitivity_group_runs_seed_checkpoint_before_sensitivity():
    paths = run.resolve_targets("sensitivity_with_seed")
    seed_42 = (run.CONFIG_DIR / run.SEEDS[0]).resolve()
    resolved_paths = [path.resolve() for path in paths]
    sensitivity_paths = {
        (run.CONFIG_DIR / relative).resolve() for relative in run.SENSITIVITY
    }
    first_sensitivity = min(
        index
        for index, path in enumerate(resolved_paths)
        if path in sensitivity_paths
    )

    assert seed_42 in resolved_paths
    assert resolved_paths.index(seed_42) < first_sensitivity


def test_default_appendix_group_does_not_rerun_seed_checkpoint():
    seed_42 = (run.CONFIG_DIR / run.SEEDS[0]).resolve()
    paths = {path.resolve() for path in run.resolve_targets("paper_appendix")}
    assert seed_42 not in paths


def test_standalone_appendix_group_runs_seed_checkpoint_before_sensitivity():
    paths = run.resolve_targets("paper_appendix_with_seed")
    seed_42 = (run.CONFIG_DIR / run.SEEDS[0]).resolve()
    resolved_paths = [path.resolve() for path in paths]
    sensitivity_paths = {
        (run.CONFIG_DIR / relative).resolve() for relative in run.SENSITIVITY
    }
    first_sensitivity = min(
        index
        for index, path in enumerate(resolved_paths)
        if path in sensitivity_paths
    )

    assert seed_42 in resolved_paths
    assert resolved_paths.index(seed_42) < first_sensitivity


def test_non_seed_experiments_have_unique_resolved_behavior():
    seen: dict[str, Path] = {}
    seed_paths = {path.resolve() for path in run.resolve_targets("seed")}
    allowed_duplicate = {
        (run.CONFIG_DIR / "baselines/learned_evidence_logit_fusion.yaml").resolve(),
        (run.CONFIG_DIR / "ablations/modules/no_i3_discount_rejection.yaml").resolve(),
    }
    for path in run.resolve_targets("all"):
        if path.resolve() in seed_paths:
            continue
        key = _semantic_key(_resolved(path))
        previous = seen.get(key)
        if previous is not None:
            assert {path.resolve(), previous.resolve()} == allowed_duplicate
        else:
            seen[key] = path


def test_obsolete_configs_are_hidden_from_runnable_catalog():
    available = {path.resolve() for path in run.available_configs().values()}
    for relative in run.OBSOLETE_CONFIGS:
        assert (run.CONFIG_DIR / relative).resolve() not in available


def test_seed_overlays_change_only_the_training_seed():
    base_path = ROOT / "baselines/api_only.yaml"
    base = _resolved(base_path)
    for seed in (2024, 3407):
        overlay = ROOT / f"_seed_{seed}_overlay.yaml"
        resolved = load_config([str(base_path), str(overlay)])
        expected = copy.deepcopy(base)
        expected["train"]["seed"] = seed
        assert resolved == expected
