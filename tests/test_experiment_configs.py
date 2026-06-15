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
        "sensitivity/acceptance_product.yaml",
        "sensitivity/coverage_80.yaml",
        "sensitivity/coverage_95.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_rejection_threshold"] is True
        assert "final_seed_42/42/best_tri_modal_robust.pt" in cfg["eval"]["checkpoint_path"]
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

    joint_only = _resolved(ROOT / "ablations/i2/joint_only_cross_attention.yaml")
    assert joint_only["semantic_cross_attention"]["attach_to_joint"] is True
    assert joint_only["semantic_cross_attention"]["attach_to_reconstruction"] is False


def test_no_reconstruction_ablation_disconnects_reconstruction_path():
    cfg = _resolved(ROOT / "ablations/training/no_masked_semantic_reconstruction.yaml")
    assert cfg["semantic_reconstruction"]["enabled"] is False
    assert cfg["semantic_reconstruction"]["weight"] == 0.0
    assert cfg["semantic_cross_attention"]["attach_to_reconstruction"] is False


def test_paper_plan_has_42_unique_runs():
    paths = run.resolve_targets("paper")
    assert len(paths) == 42
    assert len({path.resolve() for path in paths}) == len(paths)


def test_non_seed_experiments_have_unique_resolved_behavior():
    seen: dict[str, Path] = {}
    seed_paths = {path.resolve() for path in run.resolve_targets("seed")}
    for path in run.resolve_targets("all"):
        if path.resolve() in seed_paths:
            continue
        key = _semantic_key(_resolved(path))
        assert key not in seen, f"{path} duplicates {seen.get(key)}"
        seen[key] = path
