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
MAIN = ROOT / "evidential_trusted_fusion.yaml"
PRIOR = ROOT / "observable_reliability_discount_fusion.yaml"


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
        assert calibration.get("split_seed") == 42, path
        assert calibration.get("stratified_group_split") is True, path
        assert cfg["model"]["graph_encoder"]["account_for_encoder_budget"] is True, path
        assert cfg["fusion"]["force_fp32_decision"] is True, path


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
    # I1: dual-source reliability
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["reliability_calibration"]["use_evidential_uncertainty"] is True
    assert cfg["loss"]["evidential_loss_weight"] > 0.0
    assert cfg["loss"]["evidential"]["class_weight"] == "balanced"
    # I2: conflict-aware Yager evidential fusion; legacy temperature off
    assert cfg["fusion"]["combination"] == "yager"
    assert cfg["fusion"]["probability_calibration"]["enabled"] is False
    assert cfg["fusion"]["use_confidence_proxy"] is False
    # I3: class-conditional conformal rejection
    assert cfg["selective_prediction"]["enabled"] is True
    assert cfg["selective_prediction"]["mode"] == "conformal"
    assert cfg["selective_prediction"]["class_conditional"] is True
    assert cfg["selective_prediction"]["target_coverage"] == 0.95


def test_prior_method_is_linear_discount_baseline():
    cfg = _resolved(PRIOR)
    assert cfg["model"]["fusion_mode"] == "discount_probability"
    # The prior method uses linear discount fusion (no evidential combination).
    assert cfg["fusion"].get("combination", "linear") == "linear"
    assert PRIOR.name == run.PRIOR_METHOD


# ── Baselines ─────────────────────────────────────────────────────────────────

def test_baselines_disable_reliability_calibration_and_rejection():
    main = _resolved(MAIN)
    for relative in ("baselines/api_only.yaml", "baselines/graph_only.yaml",
                     "baselines/manifest_only.yaml", "baselines/tri_modal_concat.yaml",
                     "baselines/fixed_logit_fusion.yaml", "baselines/api_graph_concat.yaml"):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["mode"] == "legacy_learned_gate", relative
        assert cfg["calibration"]["enabled"] is False, relative
        assert cfg["selective_prediction"]["enabled"] is False, relative
        # Baselines share the same training augmentation as the main method.
        assert cfg["robust"] == main["robust"], relative


# ── Module / mechanism ablations ──────────────────────────────────────────────

def test_module_ablations_remove_whole_innovations():
    no_i1 = _resolved(ROOT / "ablations/modules/no_i1_reliability.yaml")
    assert no_i1["fusion"]["use_reliability_discount"] is False
    assert no_i1["fusion"]["branch_competence_prior"]["enabled"] is False
    assert no_i1["fusion"]["reliability_calibration"]["enabled"] is False
    assert no_i1["loss"]["evidential_loss_weight"] == 0.0

    no_i3 = _resolved(ROOT / "ablations/modules/no_i3_selective_rejection.yaml")
    assert no_i3["selective_prediction"]["enabled"] is False
    assert no_i3["fusion"]["combination"] == "yager"


def test_i1_mechanism_ablation_removes_only_edl_source():
    cfg = _resolved(ROOT / "ablations/i1/no_edl_reliability_source.yaml")
    assert cfg["fusion"]["reliability_calibration"]["use_evidential_uncertainty"] is False
    assert cfg["loss"]["evidential_loss_weight"] == 0.0
    # The Yager evidential fusion itself is still active.
    assert cfg["fusion"]["combination"] == "yager"
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True


def test_i2_mechanism_ablations_switch_combination_rule():
    for relative, rule in (
        ("ablations/i2/combination_dempster.yaml", "dempster"),
        ("ablations/i2/combination_cumulative.yaml", "cumulative"),
        ("ablations/i2/combination_log_pool.yaml", "log_pool"),
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["fusion"]["combination"] == rule, relative


def test_i3_mechanism_ablations_are_decision_only():
    marginal = _resolved(ROOT / "ablations/i3/marginal_conformal.yaml")
    assert marginal["selective_prediction"]["class_conditional"] is False
    assert marginal["eval"]["eval_only"] is True
    assert marginal["eval"]["refit_rejection_threshold"] is True

    threshold = _resolved(ROOT / "ablations/i3/threshold_rejection.yaml")
    assert threshold["selective_prediction"]["mode"] == "threshold"
    assert threshold["eval"]["eval_only"] is True


def test_mechanism_group_matches_declared_splits():
    paths = run.resolve_targets("mechanism")
    expected = [run.CONFIG_DIR / relative for relative in run.MECHANISM_ABLATIONS]
    assert paths == expected


# ── Sensitivity ───────────────────────────────────────────────────────────────

def test_i1_sensitivity_configs_have_distinct_meanings():
    main = _resolved(MAIN)
    assert main["loss"]["evidential_loss_weight"] == 0.1
    w05 = _resolved(ROOT / "sensitivity/i1/evidential_weight_0_05.yaml")
    w20 = _resolved(ROOT / "sensitivity/i1/evidential_weight_0_2.yaml")
    assert w05["loss"]["evidential_loss_weight"] == 0.05
    assert w20["loss"]["evidential_loss_weight"] == 0.2
    a5 = _resolved(ROOT / "sensitivity/i1/anneal_epochs_5.yaml")
    assert a5["loss"]["evidential"]["anneal_epochs"] == 5
    hd8 = _resolved(ROOT / "sensitivity/i1/reliability_hidden_dim_8.yaml")
    assert hd8["fusion"]["reliability_calibration"]["hidden_dim"] == 8


def test_i2_sensitivity_reliability_exponent():
    e025 = _resolved(ROOT / "sensitivity/i2/reliability_exponent_0_25.yaml")
    e100 = _resolved(ROOT / "sensitivity/i2/reliability_exponent_1_0.yaml")
    assert e025["fusion"]["reliability_discount_exponent"] == 0.25
    assert e100["fusion"]["reliability_discount_exponent"] == 1.0


def test_paper_evidential_plan_excludes_sensitivity():
    paper = {p.resolve() for p in run.resolve_targets("paper_evidential")}
    sensitivity = {p.resolve() for p in run.resolve_targets("sensitivity")}
    assert not (paper & sensitivity)


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
    for relative in run.EXTERNAL_EVAL:
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["run_test"] is False
        assert cfg["eval"]["run_robust_test"] is False
        assert cfg["eval"]["refit_rejection_threshold"] is True
        assert "evidential_seed_42/42/best_tri_modal_robust.pt" in cfg["eval"]["checkpoint_path"]
        assert cfg["eval"].get("extra_sets"), relative
        validate_eval_checkpoint_config(cfg, seed_cfg)


def test_decision_only_configs_reuse_seed_42_checkpoint_safely():
    seed_cfg = _resolved(ROOT / "seeds/seed_42.yaml")
    for relative in (
        "sensitivity/i3/coverage_90.yaml",
        "sensitivity/i3/coverage_99.yaml",
        "ablations/i3/marginal_conformal.yaml",
        "ablations/i3/threshold_rejection.yaml",
    ):
        cfg = _resolved(ROOT / relative)
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"]["refit_rejection_threshold"] is True
        validate_eval_checkpoint_config(cfg, seed_cfg)


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
