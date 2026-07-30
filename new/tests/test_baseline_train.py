from __future__ import annotations

import copy
from pathlib import Path

import pytest

import run
from fusion.baseline_train import (
    BASELINE_CLASSIFICATION_PROTOCOL_ID,
    BASELINE_EXPERT_SPLIT_SEED,
    BASELINE_EXPERT_VAL_FRACTION,
    FORMAL_ROBUST_CELL_COUNT,
    FORMAL_PERTURBATIONS,
    FORMAL_STRENGTHS,
    REGISTERED_BASELINES,
    _split_baseline_training_roles,
    validate_registered_baseline_config,
)
from fusion.runtime import load_config_path


ROOT = Path("config/experiments/tri_modal_robust")
BASELINE_PATHS = [
    *(ROOT / relative for relative in run.BASELINES),
    *(ROOT / relative for relative in run.TRUSTED_FUSION_BASELINES),
]


def _resolved(path: Path) -> dict:
    return load_config_path(path)


def test_formal_registry_is_exactly_the_run_catalog() -> None:
    assert len(BASELINE_PATHS) == 14
    assert len(REGISTERED_BASELINES) == 14
    observed = {
        validate_registered_baseline_config(_resolved(path)).protocol_id
        for path in BASELINE_PATHS
    }
    assert observed == set(REGISTERED_BASELINES)
    assert FORMAL_ROBUST_CELL_COUNT == 19
    assert BASELINE_CLASSIFICATION_PROTOCOL_ID == (
        "binary_argmax_fixed_0_5_v1"
    )
    assert BASELINE_EXPERT_SPLIT_SEED == 4242
    assert BASELINE_EXPERT_VAL_FRACTION == pytest.approx(0.10)


def test_every_formal_baseline_dispatches_to_closed_runner() -> None:
    for path in BASELINE_PATHS:
        assert run.resolve_runner_module(path) == "fusion.baseline_train"


@pytest.mark.parametrize(
    ("path", "mutator", "message"),
    [
        (
            ROOT / "baselines/trusted/tmc_style_adapted.yaml",
            lambda cfg: cfg["fusion"].__setitem__("combination", "cumulative"),
            "fusion.combination",
        ),
        (
            ROOT / "baselines/trusted/ecml_style_adapted.yaml",
            lambda cfg: cfg["fusion"].__setitem__("combination", "dempster"),
            "fusion.combination",
        ),
        (
            ROOT / "baselines/trusted/qmf_energy.yaml",
            lambda cfg: cfg["model"].__setitem__(
                "quality_fusion_temperature", 7.0
            ),
            "quality_fusion_temperature",
        ),
        (
            ROOT / "baselines/trusted/dempster.yaml",
            lambda cfg: cfg["fusion"].__setitem__(
                "evidence_activation", "relu"
            ),
            "evidence_activation",
        ),
        (
            ROOT / "baselines/trusted/dempster.yaml",
            lambda cfg: cfg["loss"]["evidential"].__setitem__(
                "class_weight", None
            ),
            "class_weight",
        ),
        (
            ROOT / "baselines/api_only.yaml",
            lambda cfg: cfg["model"].__setitem__("fusion_mode", "graph_only"),
            "model.fusion_mode",
        ),
        (
            ROOT / "baselines/tri_modal_concat.yaml",
            lambda cfg: cfg["calibration"].__setitem__("enabled", True),
            "calibration.enabled",
        ),
        (
            ROOT / "baselines/tri_modal_concat.yaml",
            lambda cfg: cfg["calibration"].__setitem__(
                "expert_split_seed", 42
            ),
            "expert_split_seed",
        ),
        (
            ROOT / "baselines/tri_modal_concat.yaml",
            lambda cfg: cfg["calibration"].__setitem__(
                "expert_val_fraction", 0.20
            ),
            "expert_val_fraction",
        ),
        (
            ROOT / "baselines/tri_modal_concat.yaml",
            lambda cfg: cfg["classification_threshold"].__setitem__(
                "enabled", True
            ),
            "classification_threshold.enabled",
        ),
        (
            ROOT / "baselines/tri_modal_concat.yaml",
            lambda cfg: cfg["encoder_stage"].__setitem__("mode", "reuse"),
            "encoder_stage.mode",
        ),
        (
            ROOT / "baselines/tri_modal_concat.yaml",
            lambda cfg: cfg["eval"].__setitem__("eval_only", True),
            "eval-only",
        ),
    ],
)
def test_identity_matrix_rejects_mislabeled_or_retired_lifecycle(
    path: Path,
    mutator,
    message: str,
) -> None:
    cfg = copy.deepcopy(_resolved(path))
    mutator(cfg)
    with pytest.raises(ValueError, match=message):
        validate_registered_baseline_config(cfg)


def test_identity_matrix_rejects_non_registered_method() -> None:
    cfg = _resolved(ROOT / "baselines/api_only.yaml")
    cfg["method"]["protocol_id"] = "invented_baseline"
    with pytest.raises(ValueError, match="14 registered"):
        validate_registered_baseline_config(cfg)


def test_identity_matrix_freezes_formal_19_cell_protocol() -> None:
    cfg = _resolved(ROOT / "baselines/api_only.yaml")
    assert tuple(cfg["eval"]["perturb_tests"]) == FORMAL_PERTURBATIONS
    assert tuple(cfg["eval"]["perturb_strengths"]) == FORMAL_STRENGTHS
    cfg["eval"]["perturb_strengths"] = [0.5]
    with pytest.raises(ValueError, match="eval.perturb_strengths"):
        validate_registered_baseline_config(cfg)


class _MetadataOnlyDataset:
    def __init__(self) -> None:
        self.sample_sids = [f"sid-{index:02d}" for index in range(40)]
        self.sample_groups = [
            f"group-{index // 2:02d}" for index in range(40)
        ]
        self.sample_labels = [index % 2 for index in range(40)]
        self.sample_years = [2020 + (index % 4) for index in range(40)]

    def __len__(self) -> int:
        return len(self.sample_sids)


def test_baseline_training_roles_match_care_and_are_group_disjoint() -> None:
    cfg = _resolved(ROOT / "baselines/tri_modal_concat.yaml")
    dataset = _MetadataOnlyDataset()
    expert_train, expert_val, summary = _split_baseline_training_roles(
        cfg, dataset
    )
    repeated_train, repeated_val, repeated_summary = (
        _split_baseline_training_roles(cfg, dataset)
    )

    assert expert_train.indices == repeated_train.indices
    assert expert_val.indices == repeated_val.indices
    assert summary == repeated_summary
    assert summary["expert_split_seed"] == 4242
    assert summary["expert_val_fraction"] == pytest.approx(0.10)
    assert summary["identity_disjoint"] is True
    assert summary["group_disjoint"] is True
    assert (
        summary["roles"]["expert_train"]["role_usage"]
        == "fit_model_parameters_only"
    )
    assert summary["roles"]["expert_val"]["role_usage"] == (
        "checkpoint_selection_clean_macro_f1_at_fixed_0_5_only"
    )
    train_groups = {
        dataset.sample_groups[index] for index in expert_train.indices
    }
    val_groups = {
        dataset.sample_groups[index] for index in expert_val.indices
    }
    assert train_groups.isdisjoint(val_groups)
    assert sorted(expert_train.indices + expert_val.indices) == list(
        range(len(dataset))
    )
