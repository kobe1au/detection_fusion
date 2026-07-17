from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fusion.train import (
    CHECKPOINT_STAGE_ENCODER_SELECTED,
    CHECKPOINT_STAGE_PIPELINE_FITTED,
    _fit_routed_final_temperature,
    _decision_calibration_signature,
    _file_sha256,
    _load_eval_checkpoint,
    _resolve_refit_decision_calibration,
    build_run_identity,
    fit_malware_classification_threshold,
    fit_risk_control_thresholds,
    validate_checkpoint_decision_signature,
    validate_checkpoint_stage,
)


def test_final_temperature_temporarily_enables_grad_and_restores_freeze_state():
    log_temperature = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
    raw_log_prob = F.log_softmax(
        torch.tensor(
            [
                [4.0, -4.0],
                [4.0, -4.0],
                [-4.0, 4.0],
                [4.0, -4.0],
            ]
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])

    summary = _fit_routed_final_temperature(
        log_temperature,
        raw_log_prob,
        labels,
    )

    assert log_temperature.requires_grad is False
    assert log_temperature.grad is None
    assert summary["temperature"] > 0.0
    assert summary["nll_after"] <= summary["nll_before"] + 1.0e-8


def test_final_temperature_restores_freeze_state_when_lbfgs_fails(monkeypatch):
    log_temperature = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
    raw_log_prob = F.log_softmax(torch.tensor([[2.0, -2.0]]), dim=-1)
    labels = torch.tensor([0])

    def fail_step(_self, _closure):
        raise RuntimeError("synthetic LBFGS failure")

    monkeypatch.setattr(torch.optim.LBFGS, "step", fail_step)
    with pytest.raises(RuntimeError, match="synthetic LBFGS failure"):
        _fit_routed_final_temperature(log_temperature, raw_log_prob, labels)

    assert log_temperature.requires_grad is False
    assert log_temperature.grad is None


def _write_staged_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    encoder_path = tmp_path / "best_encoder_selected.pt"
    pipeline_path = tmp_path / "best_tri_modal_robust.pt"
    torch.save(
        {
            "checkpoint_stage": CHECKPOINT_STAGE_ENCODER_SELECTED,
            "model": {"opinion_router.weight": torch.tensor([3.0])},
        },
        encoder_path,
    )
    torch.save(
        {
            "checkpoint_stage": CHECKPOINT_STAGE_PIPELINE_FITTED,
            "encoder_checkpoint_path": encoder_path.name,
            "model": {"opinion_router.weight": torch.tensor([9.0])},
        },
        pipeline_path,
    )
    return encoder_path, pipeline_path


def test_posthoc_refit_resolves_encoder_stage_and_preserves_router_start(tmp_path):
    encoder_path, pipeline_path = _write_staged_checkpoints(tmp_path)

    loaded_path, checkpoint = _load_eval_checkpoint(
        pipeline_path,
        refit_posthoc_calibration=True,
        map_location="cpu",
    )

    assert loaded_path == encoder_path
    assert checkpoint["checkpoint_stage"] == CHECKPOINT_STAGE_ENCODER_SELECTED
    assert checkpoint["model"]["opinion_router.weight"].item() == 3.0


def test_ordinary_eval_requires_pipeline_stage(tmp_path):
    encoder_path, pipeline_path = _write_staged_checkpoints(tmp_path)

    loaded_path, checkpoint = _load_eval_checkpoint(
        pipeline_path,
        refit_posthoc_calibration=False,
        map_location="cpu",
    )
    assert loaded_path == pipeline_path
    assert checkpoint["model"]["opinion_router.weight"].item() == 9.0

    with pytest.raises(ValueError, match="requires 'pipeline_fitted'"):
        _load_eval_checkpoint(
            encoder_path,
            refit_posthoc_calibration=False,
            map_location="cpu",
        )


def test_checkpoint_stage_and_pipeline_link_are_mandatory(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_stage"):
        validate_checkpoint_stage({"model": {}})

    pipeline_path = tmp_path / "pipeline_without_encoder_link.pt"
    torch.save(
        {
            "checkpoint_stage": CHECKPOINT_STAGE_PIPELINE_FITTED,
            "model": {},
        },
        pipeline_path,
    )
    with pytest.raises(ValueError, match="does not link"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=True,
            map_location="cpu",
        )


def test_pipeline_encoder_link_verifies_portable_artifact_hash(tmp_path):
    encoder_path, pipeline_path = _write_staged_checkpoints(tmp_path)
    checkpoint = torch.load(pipeline_path, map_location="cpu", weights_only=True)
    checkpoint["encoder_checkpoint_sha256"] = _file_sha256(encoder_path)
    torch.save(checkpoint, pipeline_path)

    _load_eval_checkpoint(
        pipeline_path,
        refit_posthoc_calibration=True,
        map_location="cpu",
    )
    torch.save(
        {
            "checkpoint_stage": CHECKPOINT_STAGE_ENCODER_SELECTED,
            "model": {"opinion_router.weight": torch.tensor([4.0])},
        },
        encoder_path,
    )
    with pytest.raises(ValueError, match="hash does not match"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=True,
            map_location="cpu",
        )


def test_refit_decision_calibration_supports_legacy_alias_and_rejects_conflict():
    assert _resolve_refit_decision_calibration({"refit_decision_calibration": True})
    assert _resolve_refit_decision_calibration({"refit_rejection_threshold": True})
    assert _resolve_refit_decision_calibration(
        {
            "refit_decision_calibration": True,
            "refit_rejection_threshold": True,
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        _resolve_refit_decision_calibration(
            {
                "refit_decision_calibration": False,
                "refit_rejection_threshold": True,
            }
        )


def _decision_cfg(risk_level: float = 0.05) -> dict:
    return {
        "classification_threshold": {
            "enabled": True,
            "objective": "macro_f1",
            "min_malware_recall": 0.9,
        },
        "selective_prediction": {
            "enabled": True,
            "mode": "risk_control",
            "threshold_score": "model_acceptance",
            "risk_target": "malware_fn_rate_after_rejection",
            "risk_level": risk_level,
            "min_calibration_malware": 1,
            "require_feasible": True,
        },
    }


def test_decision_signature_rejects_reusing_threshold_at_new_risk_level():
    checkpoint = {
        "decision_calibration_signature": _decision_calibration_signature(
            _decision_cfg(0.05)
        )
    }
    validate_checkpoint_decision_signature(
        _decision_cfg(0.05),
        checkpoint,
        refit_decision_calibration=False,
    )
    with pytest.raises(ValueError, match="Selective-decision settings differ"):
        validate_checkpoint_decision_signature(
            _decision_cfg(0.03),
            checkpoint,
            refit_decision_calibration=False,
        )
    changed_score = {
        **_decision_cfg(0.05),
        "fusion": {"acceptance_aggregation": "min"},
    }
    with pytest.raises(ValueError, match="Selective-decision settings differ"):
        validate_checkpoint_decision_signature(
            changed_score,
            checkpoint,
            refit_decision_calibration=False,
        )
    validate_checkpoint_decision_signature(
        _decision_cfg(0.03),
        checkpoint,
        refit_decision_calibration=True,
    )


def test_method_protocol_hash_ignores_run_location_and_worker_settings():
    base = {
        "method": {"name": "same_method"},
        "data": {
            "out_dir": "results/a",
            "train_pt_dir": "D:/pts/a",
            "val_pt_dir": "D:/pts/a",
            "test_pt_dir": "D:/pts/a",
        },
        "train": {"eval_num_workers": 2},
        "eval": {"output_name": "first"},
    }
    moved = {
        **base,
        "data": {
            **base["data"],
            "out_dir": "/root/results/b",
            "train_pt_dir": "/root/pts",
            "val_pt_dir": "/root/pts",
            "test_pt_dir": "/root/pts",
        },
        "train": {"eval_num_workers": 12},
        "eval": {"output_name": "second"},
    }

    first = build_run_identity(base, "first", 42)
    second = build_run_identity(moved, "second", 42)

    assert first["resolved_config_sha256"] != second["resolved_config_sha256"]
    assert first["method_protocol_sha256"] == second["method_protocol_sha256"]


@pytest.mark.parametrize("label", [0, 1])
def test_classification_threshold_requires_both_classes(label):
    rows = [
        {"prob_malware": probability, "label": label}
        for probability in (0.1, 0.4, 0.8)
    ]
    with pytest.raises(ValueError, match="both benign and malware"):
        fit_malware_classification_threshold(rows, {"enabled": True})


def _risk_rows(num_malware: int) -> list[dict]:
    return [
        {"acceptance_score": 0.9, "label": 1, "pred": 1}
        for _ in range(num_malware)
    ]


def test_risk_control_enforces_configured_malware_sample_floor():
    with pytest.raises(ValueError, match="min_calibration_malware=6"):
        fit_risk_control_thresholds(
            _risk_rows(5),
            {
                "enabled": True,
                "mode": "risk_control",
                "risk_level": 0.05,
                "min_calibration_malware": 6,
            },
        )


def test_risk_control_require_feasible_preflights_finite_sample_budget():
    with pytest.raises(ValueError, match="at least 19 malware samples"):
        fit_risk_control_thresholds(
            _risk_rows(5),
            {
                "enabled": True,
                "mode": "risk_control",
                "risk_level": 0.05,
                "require_feasible": True,
            },
        )

    fitted = fit_risk_control_thresholds(
        _risk_rows(19),
        {
            "enabled": True,
            "mode": "risk_control",
            "risk_level": 0.05,
            "require_feasible": True,
        },
    )
    assert fitted is not None
    assert fitted["feasible"] is True
    assert fitted["corrected_risk"] == pytest.approx(0.05)
    assert fitted["minimum_malware_for_feasibility"] == 19
    assert fitted["guarantee_type"] == "expected_crc"
    build_run_identity,
