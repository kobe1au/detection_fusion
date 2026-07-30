from __future__ import annotations

import copy
from pathlib import Path

import pytest

import run
from fusion.baseline_train import (
    _build_model,
    validate_registered_baseline_config,
)
from fusion.runtime import (
    _dataset_common_kwargs,
    load_config,
    load_config_path,
    load_yaml,
)


ROOT = Path("config/experiments/tri_modal_robust")
PRIMARY = ROOT / run.PRIMARY_SEED


def _resolved(path: Path | str) -> dict:
    return load_config_path(Path(path))


def _normalized_for_diff(cfg: dict) -> dict:
    normalized = copy.deepcopy(cfg)
    normalized.pop("method", None)
    normalized.get("train", {}).pop("exp_name", None)
    normalized.get("eval", {}).pop("output_name", None)
    return normalized


def _leaf_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = set()
        for key in set(left) | set(right):
            path = (*prefix, key)
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], path)
                )
        return differences
    return set() if left == right else {prefix}


def test_formal_catalog_resolves_without_duplicate_paths() -> None:
    paths = run.resolve_targets("all")
    assert paths
    assert len(paths) == len({path.resolve() for path in paths})
    for path in paths:
        cfg = _resolved(path)
        _dataset_common_kwargs(cfg, is_train=False)


def test_every_baseline_resolves_and_builds_with_closed_runner() -> None:
    for relative in [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]:
        path = ROOT / relative
        assert run.resolve_runner_module(path) == "fusion.baseline_train"
        cfg = _resolved(path)
        validate_registered_baseline_config(cfg)
        _build_model(cfg, feature_dim=515)


def test_care_configs_dispatch_only_to_the_care_runner() -> None:
    care_paths = [
        *(ROOT / relative for relative in run.SEEDS),
        *(ROOT / relative for relative in run.CARE_ABLATIONS),
    ]
    for path in care_paths:
        assert run.resolve_runner_module(path) == "fusion.care_train"
        cfg = _resolved(path)
        assert cfg["method"]["protocol_id"] == "care_droid_v1"
        assert cfg["model"]["fusion_mode"] == "care_droid"
        assert "adaptive_fusion" not in cfg
        assert "competence" not in cfg["model"]
        assert "token_fusion" not in cfg["model"]


def test_runner_groups_and_aliases_reference_existing_unique_configs() -> None:
    for name in run.GROUPS:
        paths = run.resolve_targets(name)
        assert paths, name
        assert len(paths) == len({path.resolve() for path in paths}), name
        assert all(path.is_file() for path in paths), name
    for name in run.ALIASES:
        paths = run.resolve_targets(name)
        assert len(paths) == 1, name
        assert paths[0].is_file(), name


def test_unordered_all_catalog_is_dry_run_only() -> None:
    run.validate_execution_target_order(["all"], dry_run=True)
    with pytest.raises(ValueError, match="catalog-only"):
        run.validate_execution_target_order(["all"], dry_run=False)


def test_comparison_methods_share_the_registered_eval_matrix() -> None:
    reference = _resolved(PRIMARY)["eval"]
    for relative in [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]:
        candidate = _resolved(ROOT / relative)["eval"]
        assert candidate["perturb_tests"] == reference["perturb_tests"], relative
        assert (
            candidate["perturb_strengths"] == reference["perturb_strengths"]
        ), relative


def test_baselines_train_their_own_stage_a_without_care_lifecycle() -> None:
    for relative in [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]:
        cfg = _resolved(ROOT / relative)
        validate_registered_baseline_config(cfg)
        assert cfg["encoder_stage"]["mode"] == "fit", relative
        assert cfg["model"]["fusion_mode"] != "care_droid", relative
        assert cfg["calibration"]["expert_split_seed"] == 4242, relative
        assert cfg["calibration"]["expert_val_fraction"] == pytest.approx(
            0.10
        ), relative
        assert cfg["classification_threshold"] == {
            "enabled": False
        }, relative
        assert "care" not in cfg, relative
        assert "adaptive_fusion" not in cfg, relative


def test_baseline_runner_rejects_retired_learned_reliability_lifecycle() -> None:
    cfg = _resolved(ROOT / run.BASELINES[0])
    cfg["fusion"]["use_i1_reliability"] = True
    with pytest.raises(ValueError, match="forbids"):
        validate_registered_baseline_config(cfg)


def test_seed_configs_change_only_model_randomness_and_names() -> None:
    seed_42 = _resolved(ROOT / "seeds/seed_42.yaml")
    for seed in (2024, 3407):
        candidate = _resolved(ROOT / f"seeds/seed_{seed}.yaml")
        differences = _leaf_differences(
            _normalized_for_diff(seed_42),
            _normalized_for_diff(candidate),
        )
        assert differences <= {("train", "seed")}
        assert candidate["train"]["seed"] == seed
        assert candidate["care"]["protocol_seed"] == 424242
        assert candidate["care"]["roles"]["expert_split_seed"] == 4242


def test_autodl_overlay_keeps_one_deduplicated_pt_pool() -> None:
    cfg = load_config([str(PRIMARY), str(ROOT / "_autodl_paths.yaml")])
    assert cfg["train"]["multiprocessing_sharing_strategy"] == "file_system"
    assert cfg["train"]["prefetch_factor"] == 1
    assert {
        cfg["data"]["train_pt_dir"],
        cfg["data"]["val_pt_dir"],
        cfg["data"]["test_pt_dir"],
    } == {"/root/autodl-tmp/pts_all"}


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "care:\n"
        "  risk_training:\n"
        "    folds: 3\n"
        "    folds: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_yaml(path)
