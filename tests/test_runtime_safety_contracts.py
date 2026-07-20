from __future__ import annotations

import copy
from pathlib import Path

import pytest

from fusion.train import (
    _dataset_common_kwargs,
    build_model,
    load_config,
    load_config_path,
    prepare_output_directory,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "config" / "experiments" / "tri_modal_robust"


def test_output_directory_collision_requires_explicit_overwrite(tmp_path: Path):
    out_dir = tmp_path / "experiment" / "42"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.yaml").write_text("finished: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_output_directory(out_dir)

    assert prepare_output_directory(out_dir, overwrite=True) == out_dir


def test_overlay_semantics_do_not_depend_on_name_or_directory(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "method:\n"
        "  name: path_independent_test\n"
        "fusion:\n"
        "  routing:\n"
        "    enabled: true\n"
        "train:\n"
        "  seed: 42\n",
        encoding="utf-8",
    )
    underscored = tmp_path / "_seed_overlay.yaml"
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    ordinary = moved_dir / "seed_overlay.yaml"
    overlay_text = "train:\n  seed: 2024\n"
    underscored.write_text(overlay_text, encoding="utf-8")
    ordinary.write_text(overlay_text, encoding="utf-8")

    first = load_config([str(base), str(underscored)])
    second = load_config([str(base), str(ordinary)])

    assert first == second
    assert first["train"]["seed"] == 2024
    assert first["fusion"]["routing"]["enabled"] is True


@pytest.mark.parametrize(
    "relative_config",
    [
        "baselines/trusted/tmc_style_adapted.yaml",
        "baselines/trusted/ecml_style_adapted.yaml",
    ],
)
def test_evidential_objectives_reject_softmax_fixed_opinions(relative_config: str):
    cfg = load_config_path(EXPERIMENT_ROOT / relative_config)
    cfg = copy.deepcopy(cfg)
    cfg["fusion"]["opinion_source"] = "softmax_fixed_uncertainty"

    with pytest.raises(
        ValueError,
        match="softmax_fixed_uncertainty.*incompatible.*loss.objective",
    ):
        build_model(cfg, feature_dim=16)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("use_support_discount",), True),
        (("detach_discount",), False),
        (("detach_confidence_proxy",), False),
        (("acceptance_aggregation",), "min"),
        (("fallback",), "bogus"),
        (("use_reliability_acceptance",), False),
        (("confidence_proxy", "temperature_api"), 2.0),
        (("support_factor", "manifest_support_base"), 0.8),
        (("conflict_factor", "min_value"), 0.2),
    ],
)
def test_routed_mode_rejects_non_neutral_linear_only_settings(
    path: tuple[str, ...], value: object
):
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    target = cfg["fusion"]
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value

    with pytest.raises(ValueError, match="does not consume"):
        build_model(cfg, feature_dim=16)


@pytest.mark.parametrize(
    ("section", "path", "value"),
    [
        ("model", ("joint_emb_dim",), 32),
        ("fusion", ("linear_use_joint_branch",), False),
        ("loss", ("branch_aux_weights", "joint"), 0.0),
        ("model", ("gate", "apply_alive_mask"), True),
    ],
)
def test_removed_joint_configuration_fails_fast(
    section: str, path: tuple[str, ...], value: object
):
    cfg = copy.deepcopy(
        load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    )
    target = cfg.setdefault(section, {})
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value

    with pytest.raises(ValueError, match="Joint branch.*removed"):
        build_model(cfg, feature_dim=16)


def test_graph_encoder_budget_accounting_cannot_be_disabled():
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["graph_encoder"]["account_for_encoder_budget"] = False

    with pytest.raises(ValueError, match="encoder-only truncation"):
        build_model(cfg, feature_dim=16)


def test_nested_graph_budget_is_shared_by_dataset_and_encoder():
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    del cfg["model"]["max_nodes_gnn"]
    cfg["model"]["graph_encoder"]["max_nodes"] = 64

    kwargs = _dataset_common_kwargs(cfg, is_train=False)
    model = build_model(cfg, feature_dim=16)

    assert kwargs["max_graph_nodes_per_sample"] == 64
    assert model.graph_encoder.max_nodes == 64


def test_conflicting_graph_budget_aliases_fail_at_startup():
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["graph_encoder"]["max_nodes"] = 64

    with pytest.raises(ValueError, match="Conflicting graph budgets"):
        _dataset_common_kwargs(cfg, is_train=False)
    with pytest.raises(ValueError, match="Conflicting graph budgets"):
        build_model(cfg, feature_dim=16)


def test_dataset_api_budget_cannot_exceed_encoder_capacity():
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["data"]["max_api_events_per_sample"] = 8
    cfg["model"]["api_encoder"]["max_seq_len"] = 4

    with pytest.raises(ValueError, match="only API truncation point"):
        _dataset_common_kwargs(cfg, is_train=False)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), True])
def test_formal_config_requires_positive_integral_api_budget(value: object):
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["data"]["max_api_events_per_sample"] = value

    with pytest.raises(ValueError, match="max_api_events_per_sample"):
        _dataset_common_kwargs(cfg, is_train=False)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), True])
def test_model_requires_positive_integral_api_encoder_capacity(value: object):
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["api_encoder"]["max_seq_len"] = value

    with pytest.raises(ValueError, match="max_seq_len"):
        build_model(cfg, feature_dim=16)


@pytest.mark.parametrize("value", [3, 2.5, float("nan"), True])
def test_training_pipeline_rejects_non_binary_model_contract(value: object):
    cfg = load_config_path(EXPERIMENT_ROOT / "evidential_trusted_fusion.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["num_classes"] = value

    with pytest.raises(ValueError, match="num_classes|binary-only"):
        build_model(cfg, feature_dim=16)
