from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import run
from fusion.model import ANCHORED_JOINT_LATE_FUSION_MODES
from fusion.train import (
    _dataset_common_kwargs,
    _stage_b_config,
    build_model,
    build_run_identity,
    load_config,
    load_config_path,
    load_yaml,
)


ROOT = Path("config/experiments/tri_modal_robust")
MAIN = ROOT / "competence_anchored_fusion.yaml"
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


def test_formal_catalog_resolves_builds_and_has_no_duplicate_paths() -> None:
    paths = run.resolve_targets("all")
    assert paths
    assert len(paths) == len({path.resolve() for path in paths})
    for path in paths:
        cfg = _resolved(path)
        _dataset_common_kwargs(cfg, is_train=False)
        build_model(cfg, feature_dim=515)


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


def test_primary_method_has_closed_competence_anchored_contract() -> None:
    cfg = _resolved(MAIN)
    assert cfg["method"]["protocol_id"] == "tcp_joint_anchor_crc_v1"
    assert cfg["encoder_stage"]["protocol_id"] == "joint_atomic_clean_stage1_v1"
    assert cfg["model"]["fusion_mode"] in ANCHORED_JOINT_LATE_FUSION_MODES
    assert cfg["fusion"]["mode"] == "model_dispatch"
    assert set(cfg["fusion"]) == {"mode", "competence", "anchored_router"}

    competence = cfg["fusion"]["competence"]
    assert competence == {
        "projection_dim": 32,
        "hidden_dim": 16,
        "dropout": pytest.approx(0.10),
    }
    router = cfg["fusion"]["anchored_router"]
    assert router["initial_atomic_competence_scale"] > 0.0
    assert router["initial_joint_late_scale"] > 0.0
    assert 0.0 < router["initial_late_gate"] < 1.0

    assert cfg["loss"]["objective"] == "anchored_stage_a"
    assert cfg["loss"]["atomic_aux_weight"] == pytest.approx(0.25)
    assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
    assert "evidential_loss_weight" not in cfg["loss"]
    assert "branch_aux_weight" not in cfg["loss"]

    stage_b = _stage_b_config(cfg)
    assert stage_b["degradation"]["mechanisms"] == [
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
    ]
    assert stage_b["degradation"]["strength_min"] == pytest.approx(0.1)
    assert stage_b["degradation"]["strength_max"] == pytest.approx(0.9)
    assert stage_b["competence"]["regression"] == "mse"
    assert stage_b["competence"]["degraded_loss_weight"] == pytest.approx(0.25)
    assert stage_b["competence"][
        "clean_noninferiority_relative_tolerance"
    ] == pytest.approx(0.01)
    assert stage_b["competence"][
        "clean_noninferiority_absolute_tolerance"
    ] == pytest.approx(0.0)
    assert stage_b["competence"]["ranking_weight"] == pytest.approx(0.10)
    assert stage_b["router"]["degradation_loss_weights"] == [
        0.0,
        0.1,
        0.25,
        0.5,
    ]
    assert stage_b["router"]["clean_anchor_kl_weight"] == pytest.approx(0.10)
    assert stage_b["router"]["clean_noninferiority_tolerance"] == pytest.approx(
        0.0
    )
    assert stage_b["router"][
        "degraded_source_noninferiority_tolerance"
    ] == pytest.approx(0.0)
    assert stage_b["router"]["minimum_robust_gain"] == pytest.approx(0.0)


def test_primary_model_instantiates_exact_stage_a_and_stage_b_partition() -> None:
    model = build_model(_resolved(MAIN), feature_dim=16)
    assert model.joint_expert is not None
    assert model.competence_estimator is not None
    assert model.anchored_fusion is not None
    assert model.discount_fusion is None
    stage_a_keys = model.encoder_stage_state_keys()
    assert any(key.startswith("joint_expert.") for key in stage_a_keys)
    assert not any(key.startswith("competence_estimator.") for key in stage_a_keys)
    assert not any(key.startswith("anchored_fusion.") for key in stage_a_keys)
    frozen_ids = {
        id(parameter) for parameter in model.encoder_training_frozen_parameters()
    }
    expected_ids = {
        id(parameter)
        for module in (model.competence_estimator, model.anchored_fusion)
        for parameter in module.parameters()
    }
    assert frozen_ids == expected_ids


def test_primary_uses_immutable_two_role_validation_and_one_sided_i3() -> None:
    cfg = _resolved(MAIN)
    calibration = cfg["calibration"]
    assert calibration["enabled"] is False
    assert calibration["validation_fraction"] == pytest.approx(0.25)
    assert calibration["stratified_group_split"] is True
    assert calibration["require_role_assignment"] is True
    assert calibration["role_assignment_path"].endswith(
        "validation_roles_protocol_v2.json"
    )
    assert "conformal_fraction" not in calibration
    assert "cross_fitting" not in calibration
    assert "fit_perturbations" not in calibration

    classification = cfg["classification_threshold"]
    assert classification == {
        "enabled": True,
        "objective": "macro_f1",
        "selection_rule": "macro_f1_unconstrained_v1",
        "constraint": "none",
    }
    selective = cfg["selective_prediction"]
    assert selective["enabled"] is True
    assert selective["mode"] == "risk_control"
    assert selective["threshold_score"] == "malware_fn_probability_anchor"
    assert selective["risk_target"] == "accepted_fn_risk_among_malware"
    assert selective["risk_level"] == pytest.approx(0.05)
    assert selective["require_feasible"] is True


def test_primary_eval_protocol_is_scoped_to_three_quality_axes() -> None:
    cfg = _resolved(MAIN)
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
    assert 1 + 3 * 5 + 3 == 19


def test_all_comparison_methods_share_the_registered_eval_matrix() -> None:
    reference = _resolved(PRIMARY)["eval"]
    for relative in [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]:
        candidate = _resolved(ROOT / relative)["eval"]
        assert candidate["perturb_tests"] == reference["perturb_tests"], relative
        assert (
            candidate["perturb_strengths"] == reference["perturb_strengths"]
        ), relative


def test_baselines_train_their_own_stage_a_and_do_not_enable_stage_b() -> None:
    for relative in [*run.BASELINES, *run.TRUSTED_FUSION_BASELINES]:
        cfg = _resolved(ROOT / relative)
        assert cfg["encoder_stage"]["mode"] == "fit", relative
        assert cfg["encoder_stage"]["protocol_id"] != (
            "joint_atomic_clean_stage1_v1"
        ), relative
        assert cfg["model"]["fusion_mode"] != "anchored_joint_late", relative
        assert "stage_b" not in cfg, relative
        assert cfg["classification_threshold"]["enabled"] is True, relative
        assert cfg["classification_threshold"]["objective"] == "macro_f1", relative
        assert cfg["calibration"]["role_assignment_path"].endswith(
            "validation_roles_protocol_v2.json"
        ), relative


def test_primary_run_identity_records_the_new_method_semantics() -> None:
    cfg = _resolved(PRIMARY)
    identity = build_run_identity(cfg, cfg["train"]["exp_name"], 42)
    assert identity["method_protocol_id"] == "tcp_joint_anchor_crc_v1"
    assert identity["model_fusion_mode"] == "anchored_joint_late"
    assert identity["training_objective"] == "anchored_stage_a"
    assert identity["stage_a_primary_expert"] == "joint"
    assert identity["i1_target"] == "continuous_true_class_probability"
    assert "competence_weighted_atomic" in identity["i2_formula"]
    assert identity["selective_score_type"] == (
        "malware_fn_probability_anchor"
    )


def test_seed_configs_change_only_registered_randomness_and_names() -> None:
    seed_42 = _resolved(ROOT / "seeds/seed_42.yaml")
    for seed in (2024, 3407):
        candidate = _resolved(ROOT / f"seeds/seed_{seed}.yaml")
        left = _normalized_for_diff(seed_42)
        right = _normalized_for_diff(candidate)
        differences = _leaf_differences(left, right)
        assert differences <= {
            ("train", "seed"),
            ("train", "stage1_seed"),
            ("train", "stage_b_seed"),
        }
        assert candidate["train"]["seed"] == seed


def test_autodl_overlay_keeps_one_deduplicated_pt_pool() -> None:
    cfg = load_config(
        [str(PRIMARY), str(ROOT / "_autodl_paths.yaml")]
    )
    assert cfg["train"]["multiprocessing_sharing_strategy"] == "file_system"
    assert cfg["train"]["prefetch_factor"] == 1
    assert {
        cfg["data"]["train_pt_dir"],
        cfg["data"]["val_pt_dir"],
        cfg["data"]["test_pt_dir"],
    } == {"/root/autodl-tmp/pts_all"}


def test_removed_main_files_and_old_factorial_catalog_are_absent() -> None:
    removed = [
        ROOT / "evidential_trusted_fusion.yaml",
        ROOT / "ablations/modules/no_i1_reliability.yaml",
        ROOT / "ablations/modules/no_i2_learned_components.yaml",
        ROOT / "ablations/factorial/i1_i2/i1_off_i2_off.yaml",
    ]
    assert all(not path.exists() for path in removed)
    assert "i1_i2_2x2" not in run.GROUPS
    assert "factorial" not in run.GROUPS
    serialized = json.dumps(run.GROUPS, sort_keys=True)
    assert "prior_only" not in serialized
    assert "risk_conflict" not in serialized


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "stage_b:\n"
        "  competence:\n"
        "    degraded_loss_weight: 0.25\n"
        "    degraded_loss_weight: 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="[Dd]uplicate YAML key"):
        load_yaml(path)
