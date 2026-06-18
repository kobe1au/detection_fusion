from __future__ import annotations

import copy
import json
from pathlib import Path

import run
import yaml
from fusion.train import (
    _dataset_common_kwargs,
    build_model,
    load_config_path,
    validate_eval_checkpoint_config,
)


ROOT = Path("config/experiments/tri_modal_robust")


def _resolved(path: Path) -> dict:
    return load_config_path(path)


def _semantic_key(cfg: dict) -> str:
    normalized = copy.deepcopy(cfg)
    normalized.get("train", {}).pop("exp_name", None)
    normalized.get("train", {}).pop("seed", None)
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
    for group in ("tuning_i1", "tuning_i2", "tuning_i3", "tuning"):
        for path in run.resolve_targets(group):
            cfg = _resolved(path)
            eval_cfg = cfg.get("eval", {}) or {}
            assert eval_cfg.get("run_test") is False, path
            assert eval_cfg.get("run_robust_test") is False, path
            if eval_cfg.get("eval_only", False):
                assert "tuning_full_candidate/42/best_tri_modal_robust.pt" in eval_cfg[
                    "checkpoint_path"
                ], path
                assert eval_cfg.get("refit_rejection_threshold") is True, path
            else:
                assert cfg["train"].get("tuning_mode") is True, path


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
    tuning_paths = {path.resolve() for path in run.resolve_targets("tuning")}
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


def test_decision_only_sensitivities_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    for relative in (
        "sensitivity/i3/acceptance_product.yaml",
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
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_runnable_configs_share_validation_selection_protocol():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        calibration = cfg.get("calibration", {}) or {}
        assert calibration.get("validation_fraction") == 0.5, path
        assert calibration.get("split_seed") == 42, path
        assert (
            calibration.get("enabled")
            or calibration.get("holdout_enabled")
            or (cfg.get("selective_prediction", {}) or {}).get("enabled")
        ), path


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
    assert no_i1["fusion"]["use_support_discount"] is False
    assert no_i1["fusion"]["use_conflict_discount"] is False
    assert no_i1["fusion"]["use_hard_alive_mask"] is False
    assert no_i1["semantic_cross_attention"]["use_reliability_bias"] is False
    assert no_i1["semantic_cross_attention"]["use_support_bias"] is False
    assert no_i1["semantic_cross_attention"]["use_conflict_bias"] is False

    no_i2 = _resolved(ROOT / "ablations/modules/no_i2_semantic_interaction.yaml")
    assert no_i2["semantic_cross_attention"]["enabled"] is False
    assert no_i2["semantic_cross_attention"]["attach_to_joint"] is False
    assert no_i2["semantic_cross_attention"]["attach_to_reconstruction"] is False

    no_i3 = _resolved(ROOT / "baselines/learned_evidence_logit_fusion.yaml")
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
        *run.BASELINES,
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
        *run.BASELINES,
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
    for path in run.resolve_targets("all"):
        if path.resolve() in seed_paths:
            continue
        key = _semantic_key(_resolved(path))
        assert key not in seen, f"{path} duplicates {seen.get(key)}"
        seen[key] = path
