from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import run
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


def test_formal_configs_use_clean_validation_selection():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        eval_cfg = cfg.get("eval", {}) or {}
        if eval_cfg.get("eval_only", False):
            continue
        assert cfg["train"]["checkpoint_metric"] == "clean_macro_f1", path
        assert eval_cfg["robust_val"]["enabled"] is False, path


def test_baselines_share_train_augmentation_but_disable_posthoc_rejection():
    full = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    for relative in run.BASELINES:
        cfg = _resolved(ROOT / relative)
        assert cfg["robust"] == full["robust"], relative
        assert cfg["calibration"]["enabled"] is False, relative
        assert cfg["selective_prediction"]["enabled"] is False, relative
        assert cfg["fusion"]["mode"] == "legacy_learned_gate", relative


def test_lean_full_method_excludes_semantic_interaction_modules():
    cfg = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    assert "semantic_cross_attention" not in cfg
    assert "semantic_reconstruction" not in cfg
    assert cfg["model"]["fusion_mode"] == "discount_probability"
    assert cfg["fusion"]["mode"] == "discount_probability"
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["probability_calibration"]["enabled"] is True
    assert cfg["fusion"]["reliability_discount_exponent"] == 0.5
    assert cfg["loss"]["reliability_weighted_aux"] is True
    assert cfg["selective_prediction"]["enabled"] is True
    assert cfg["selective_prediction"]["target_coverage"] == 0.95


def test_module_ablations_match_lean_paper_claims():
    no_i1 = _resolved(ROOT / "ablations/modules/no_i1_observable_reliability.yaml")
    assert no_i1["fusion"]["reliability_calibration"]["enabled"] is False
    assert no_i1["fusion"]["branch_competence_prior"]["enabled"] is False
    assert no_i1["fusion"]["use_reliability_discount"] is False
    assert no_i1["fusion"]["use_support_discount"] is True
    assert no_i1["fusion"]["use_conflict_discount"] is True

    reliability_only = _resolved(ROOT / "ablations/modules/reliability_only_discount_fusion.yaml")
    assert reliability_only["fusion"]["use_support_discount"] is False
    assert reliability_only["fusion"]["use_conflict_discount"] is False
    assert reliability_only["fusion"]["use_confidence_proxy"] is False
    assert reliability_only["fusion"]["use_hard_alive_mask"] is False
    assert reliability_only["fusion"]["use_reliability_discount"] is True

    no_i3 = _resolved(ROOT / "ablations/modules/no_i3_selective_rejection.yaml")
    assert no_i3["selective_prediction"]["enabled"] is False
    assert no_i3["fusion"]["mode"] == "discount_probability"


def test_reliability_sensitivity_configs_have_distinct_meanings():
    full = _resolved(ROOT / "observable_reliability_discount_fusion.yaml")
    assert full["fusion"]["use_reliability_discount"] is True
    assert full["fusion"]["reliability_discount_exponent"] == 0.5

    exp_025 = _resolved(ROOT / "sensitivity/i1/reliability_exponent_0_25.yaml")
    assert exp_025["fusion"]["use_reliability_discount"] is True
    assert exp_025["fusion"]["reliability_discount_exponent"] == 0.25

    acceptance_only = _resolved(ROOT / "sensitivity/i1/reliability_acceptance_only.yaml")
    assert acceptance_only["fusion"]["use_reliability_discount"] is False
    assert acceptance_only["fusion"]["use_reliability_acceptance"] is True
    assert acceptance_only["fusion"]["reliability_calibration"]["enabled"] is True


def test_weight_sharpening_sensitivity_configs_are_registered():
    gamma_15 = _resolved(ROOT / "sensitivity/i2/weight_sharpening_gamma_1_5.yaml")
    gamma_20 = _resolved(ROOT / "sensitivity/i2/weight_sharpening_gamma_2_0.yaml")
    assert gamma_15["fusion"]["weight_sharpening_gamma"] == 1.5
    assert gamma_20["fusion"]["weight_sharpening_gamma"] == 2.0
    sensitivity_paths = {path.as_posix() for path in run.resolve_targets("sensitivity")}
    assert (ROOT / "sensitivity/i2/weight_sharpening_gamma_1_5.yaml").as_posix() in sensitivity_paths
    assert (ROOT / "sensitivity/i2/weight_sharpening_gamma_2_0.yaml").as_posix() in sensitivity_paths

def test_mechanism_group_contains_only_lean_mechanism_splits():
    paths = run.resolve_targets("mechanism")
    expected = [run.CONFIG_DIR / relative for relative in run.MECHANISM_ABLATIONS]
    assert paths == expected


def test_paper_plan_excludes_sensitivity_configs():
    paper_paths = {path.resolve() for path in run.resolve_targets("paper")}
    sensitivity_paths = {path.resolve() for path in run.resolve_targets("sensitivity")}
    assert not (paper_paths & sensitivity_paths)


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


def test_decision_only_sensitivities_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    for relative in ("sensitivity/i3/acceptance_min.yaml", "sensitivity/i3/coverage_80.yaml"):
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_rejection_threshold"] is True
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_runnable_configs_share_validation_protocol():
    for path in run.resolve_targets("all"):
        cfg = _resolved(path)
        calibration = cfg.get("calibration", {}) or {}
        assert cfg["train"]["strict_deterministic"] is False, path
        assert calibration.get("validation_fraction") == 0.5, path
        assert calibration.get("split_seed") == 42, path
        assert calibration.get("stratified_group_split") is True, path
        assert cfg["model"]["graph_encoder"]["account_for_encoder_budget"] is True, path
        assert cfg["fusion"]["force_fp32_decision"] is True, path


def test_seed_overlays_change_only_the_training_seed():
    base_path = ROOT / "observable_reliability_discount_fusion.yaml"
    base = _resolved(base_path)
    for seed in (42, 2024, 3407):
        seed_path = ROOT / f"seeds/seed_{seed}.yaml"
        resolved = _resolved(seed_path)
        expected = copy.deepcopy(base)
        expected["method"] = {"name": f"final_seed_{seed}"}
        expected["train"]["exp_name"] = f"final_seed_{seed}"
        expected["train"]["seed"] = seed
        assert resolved == expected


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
