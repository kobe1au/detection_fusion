from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import run
from fusion.losses import compute_robust_loss
from fusion.model import TriModalRobustModel
from fusion.train import (
    _stage_b_config,
    build_model,
    build_run_identity,
    fit_anchored_stage_b,
    load_config_path,
)


ROOT = Path("config/experiments/tri_modal_robust")
PRIMARY = ROOT / "seeds/seed_42.yaml"


def _resolved(relative: str | Path) -> dict:
    path = Path(relative)
    return load_config_path(path if path.is_absolute() else ROOT / path)


def _without_names(cfg: dict) -> dict:
    value = copy.deepcopy(cfg)
    value.pop("method", None)
    value.get("train", {}).pop("exp_name", None)
    value.get("eval", {}).pop("output_name", None)
    return value


def _leaf_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        result = set()
        for key in set(left) | set(right):
            path = (*prefix, key)
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(_leaf_differences(left[key], right[key], path))
        return result
    return set() if left == right else {prefix}


def _availability(rows: list[tuple[int, int, int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_stage_a_objective_is_joint_primary_plus_alive_atomic_auxiliary() -> None:
    labels = torch.tensor([0, 1])
    joint_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    outputs = {
        "api_logits_aux": torch.tensor(
            [[1.5, -0.5], [-0.5, 1.5]], requires_grad=True
        ),
        "graph_logits_aux": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        ),
        "manifest_logits_aux": torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]], requires_grad=True
        ),
    }
    total, parts = compute_robust_loss(
        joint_logits,
        labels,
        outputs,
        {
            "objective": "anchored_stage_a",
            "atomic_aux_weight": 1.0,
            "auxiliary_weight_mode": "alive_masked_uniform",
        },
        availability=_availability([(1, 1, 1), (1, 1, 1)]),
    )
    assert total.item() == pytest.approx(
        parts["joint_ce"] + parts["atomic_ce"],
        rel=1.0e-6,
    )
    total.backward()
    assert joint_logits.grad is not None
    assert all(value.grad is not None for value in outputs.values())


def test_stage_a_all_dead_rows_do_not_train_joint_or_atomic_biases() -> None:
    labels = torch.tensor([1])
    joint = torch.tensor([[2.0, -1.0]], requires_grad=True)
    branches = {
        f"{name}_logits_aux": torch.tensor(
            [[2.0, -1.0]], requires_grad=True
        )
        for name in ("api", "graph", "manifest")
    }
    loss, parts = compute_robust_loss(
        joint,
        labels,
        branches,
        {
            "objective": "anchored_stage_a",
            "atomic_aux_weight": 1.0,
        },
        availability=_availability([(0, 0, 0)]),
    )
    assert loss.item() == pytest.approx(0.0)
    assert parts["joint_active_fraction"] == pytest.approx(0.0)
    assert parts["atomic_active_fraction"] == pytest.approx(0.0)
    loss.backward()
    assert torch.equal(joint.grad, torch.zeros_like(joint))
    assert all(
        torch.equal(value.grad, torch.zeros_like(value))
        for value in branches.values()
    )


def test_stage_a_missing_modality_trains_only_alive_atomic_fallback() -> None:
    label = torch.tensor([1])
    joint = torch.tensor([[2.0, -1.0]], requires_grad=True)
    branches = {
        f"{name}_logits_aux": torch.tensor(
            [[2.0, -1.0]], requires_grad=True
        )
        for name in ("api", "graph", "manifest")
    }
    loss, parts = compute_robust_loss(
        joint,
        label,
        branches,
        {
            "objective": "anchored_stage_a",
            "atomic_aux_weight": 0.25,
        },
        availability=_availability([(1, 0, 1)]),
    )
    loss.backward()

    assert parts["joint_active_fraction"] == pytest.approx(0.0)
    assert torch.equal(joint.grad, torch.zeros_like(joint))
    assert bool((branches["api_logits_aux"].grad != 0).any())
    assert torch.equal(
        branches["graph_logits_aux"].grad,
        torch.zeros_like(branches["graph_logits_aux"]),
    )
    assert bool((branches["manifest_logits_aux"].grad != 0).any())


@pytest.mark.parametrize(
    ("relative", "changed_path", "expected"),
    [
        (
            "ablations/i1/no_tcp_ranking.yaml",
            ("stage_b", "competence", "ranking_weight"),
            0.0,
        ),
        (
            "ablations/i1/no_degraded_competence.yaml",
            ("stage_b", "competence", "degraded_loss_weight"),
            0.0,
        ),
    ],
)
def test_i1_ablations_change_one_declared_competence_axis(
    relative: str,
    changed_path: tuple[str, ...],
    expected: float,
) -> None:
    main = _resolved("seeds/seed_42.yaml")
    ablation = _resolved(relative)
    assert _leaf_differences(
        _without_names(main),
        _without_names(ablation),
    ) == {
        changed_path,
        ("encoder_stage", "mode"),
        ("encoder_stage", "checkpoint_path"),
    }
    assert ablation["encoder_stage"] == {
        "mode": "reuse",
        "protocol_id": "joint_atomic_clean_stage1_v1",
        "checkpoint_path": (
            "results/tri_modal_robust/competence_anchored_seed_42/42/"
            "best_encoder_selected.pt"
        ),
        "expected_sha256": None,
        "strict_identity": True,
    }
    value = ablation
    for key in changed_path:
        value = value[key]
    assert value == expected


@pytest.mark.parametrize(
    ("relative", "changed_path", "expected"),
    [
        (
            "ablations/i2/clean_only_router.yaml",
            ("stage_b", "router", "degradation_loss_weights"),
            [0.0],
        ),
        (
            "ablations/i2/no_clean_anchor_kl.yaml",
            ("stage_b", "router", "clean_anchor_kl_weight"),
            0.0,
        ),
    ],
)
def test_i2_ablations_change_one_declared_router_axis(
    relative: str,
    changed_path: tuple[str, ...],
    expected,
) -> None:
    main = _resolved("seeds/seed_42.yaml")
    ablation = _resolved(relative)
    assert _leaf_differences(
        _without_names(main),
        _without_names(ablation),
    ) == {
        changed_path,
        ("encoder_stage", "mode"),
        ("encoder_stage", "checkpoint_path"),
    }
    assert ablation["encoder_stage"] == {
        "mode": "reuse",
        "protocol_id": "joint_atomic_clean_stage1_v1",
        "checkpoint_path": (
            "results/tri_modal_robust/competence_anchored_seed_42/42/"
            "best_encoder_selected.pt"
        ),
        "expected_sha256": None,
        "strict_identity": True,
    }
    value = ablation
    for key in changed_path:
        value = value[key]
    assert value == expected


def test_training_ablation_removes_only_atomic_auxiliary() -> None:
    main = _resolved("seeds/seed_42.yaml")
    ablation = _resolved("ablations/training/no_atomic_auxiliary.yaml")
    assert _leaf_differences(
        _without_names(main),
        _without_names(ablation),
    ) == {("loss", "atomic_aux_weight")}
    assert ablation["loss"]["atomic_aux_weight"] == 0.0
    assert ablation["encoder_stage"]["mode"] == "fit"
    assert ablation["encoder_stage"]["checkpoint_path"] is None


def test_no_i3_is_a_pure_decision_layer_removal() -> None:
    main = _resolved("seeds/seed_42.yaml")
    no_i3 = _resolved("ablations/modules/no_i3_decision_layer.yaml")
    assert no_i3["classification_threshold"] == main["classification_threshold"]
    assert main["selective_prediction"]["enabled"] is True
    assert no_i3["selective_prediction"]["enabled"] is False
    for section in (
        "data",
        "encoder_stage",
        "model",
        "fusion",
        "loss",
        "calibration",
        "stage_b",
    ):
        assert no_i3[section] == main[section], section
    assert no_i3["eval"]["eval_only"] is True
    assert no_i3["eval"]["refit_posthoc_calibration"] is False
    assert no_i3["eval"]["refit_decision_calibration"] is False


def test_no_i1_i2_reuses_stage_a_and_disables_entire_stage_b() -> None:
    main = _resolved("seeds/seed_42.yaml")
    ablation = _resolved(
        "ablations/modules/no_i1_i2_joint_anchor.yaml"
    )
    assert _leaf_differences(
        _without_names(main),
        _without_names(ablation),
    ) == {
        ("encoder_stage", "mode"),
        ("encoder_stage", "checkpoint_path"),
        ("stage_b", "enabled"),
    }
    assert ablation["encoder_stage"] == {
        "mode": "reuse",
        "protocol_id": "joint_atomic_clean_stage1_v1",
        "checkpoint_path": (
            "results/tri_modal_robust/competence_anchored_seed_42/42/"
            "best_encoder_selected.pt"
        ),
        "expected_sha256": None,
        "strict_identity": True,
    }
    assert _stage_b_config(ablation)["enabled"] is False
    identity = build_run_identity(
        ablation, ablation["train"]["exp_name"], 42
    )
    assert identity["stage_b_enabled"] is False
    assert identity["stage_b_fit_population"] == "none"
    assert identity["i1_target"] == "disabled"
    assert identity["i2_formula"] == (
        "joint_anchor_with_uniform_missing_fallback"
    )


def test_disabled_stage_b_is_fail_closed_if_fit_is_called() -> None:
    ablation = _resolved(
        "ablations/modules/no_i1_i2_joint_anchor.yaml"
    )
    with pytest.raises(
        ValueError, match=r"stage_b\.enabled=false"
    ):
        fit_anchored_stage_b(
            None,
            train_dataset=None,
            validation_dataset=None,
            validation_loader=None,
            selection_indices=None,
            device=torch.device("cpu"),
            use_amp=False,
            cfg=ablation,
            seed=42,
        )


def test_chain_structure_does_not_offer_invalid_i1_off_i2_on_cell() -> None:
    assert set(run.I1_ABLATIONS) == {
        "ablations/i1/no_tcp_ranking.yaml",
        "ablations/i1/no_degraded_competence.yaml",
    }
    assert set(run.I2_ABLATIONS) == {
        "ablations/i2/clean_only_router.yaml",
        "ablations/i2/no_clean_anchor_kl.yaml",
    }
    assert "i1_i2_2x2" not in run.GROUPS
    assert not hasattr(run, "I1_I2_FACTORIAL")
    assert "no_i1" not in run.ALIASES
    assert "no_i2" not in run.ALIASES
    assert (
        "ablations/modules/no_i1_i2_joint_anchor.yaml"
        in run.MODULE_ABLATIONS
    )
    assert run.ALIASES["no_i1_i2"] == (
        "ablations/modules/no_i1_i2_joint_anchor.yaml"
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("stage_b", "cross_fitting", {"enabled": True}),
        ("stage_b", "reliability_prior", {"enabled": False}),
        ("competence", "use_evidential_uncertainty", True),
        ("competence", "use_prediction_margin", True),
        ("router", "risk_conflict_enabled", False),
        ("router", "prior_only", True),
    ],
)
def test_stage_b_rejects_retired_evidential_prior_and_risk_keys(
    section: str,
    key: str,
    value,
) -> None:
    cfg = _resolved("seeds/seed_42.yaml")
    if section == "stage_b":
        cfg["stage_b"][key] = value
    else:
        cfg["stage_b"][section][key] = value
    with pytest.raises(ValueError, match="Unsupported stage_b"):
        _stage_b_config(cfg)


def test_anchored_model_rejects_legacy_discount_fusion_sections() -> None:
    cfg = _resolved("seeds/seed_42.yaml")
    cfg["fusion"]["routing"] = {"enabled": False}
    with pytest.raises(ValueError, match="accepts only"):
        build_model(cfg, feature_dim=16)


def test_anchored_model_has_no_evidential_or_old_router_modules() -> None:
    model = build_model(_resolved("seeds/seed_42.yaml"), feature_dim=16)
    assert isinstance(model, TriModalRobustModel)
    assert model.discount_fusion is None
    assert not hasattr(model, "opinion_router")
    assert model.competence_estimator is not None
    assert model.anchored_fusion is not None
    assert model.joint_expert is not None


def test_i3_comparison_grid_keeps_the_fitted_classifier_fixed() -> None:
    main = _resolved("seeds/seed_42.yaml")
    for relative in run.I3_MECHANISM_ABLATIONS:
        cfg = _resolved(relative)
        assert cfg["classification_threshold"] == main["classification_threshold"]
        assert cfg["model"] == main["model"]
        assert cfg["fusion"] == main["fusion"]
        assert cfg["stage_b"] == main["stage_b"]
        assert cfg["eval"]["eval_only"] is True
        assert cfg["eval"].get("refit_posthoc_calibration", False) is False
        assert cfg["eval"]["checkpoint_path"].endswith(
            "best_tri_modal_robust.pt"
        )
