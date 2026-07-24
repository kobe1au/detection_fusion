import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fusion.temperature import (
    FINAL_TEMPERATURE_MAX,
    FINAL_TEMPERATURE_MIN,
    bounded_final_temperature,
    raw_final_temperature_coordinate,
)

from fusion.train import (
    CHECKPOINT_STAGE_ENCODER_SELECTED,
    CHECKPOINT_STAGE_PIPELINE_FITTED,
    PIPELINE_ARTIFACT_SCHEMA_VERSION,
    _calibration_subset_identity,
    _fit_routed_final_temperature,
    _resolve_restored_stage_convergence,
    _decision_calibration_signature,
    _decision_calibration_data_identity,
    _file_sha256,
    _load_eval_checkpoint,
    _pipeline_decision_metadata_sha256,
    _state_dict_sha256,
    _resolve_refit_decision_calibration,
    build_run_identity,
    compute_branch_reliability_metrics,
    deterministic_stratified_fold_ids,
    evaluate,
    fit_malware_classification_threshold,
    fit_oof_malware_classification_threshold,
    fit_posthoc_calibration,
    fit_risk_control_thresholds,
    run as run_training,
    validate_checkpoint_decision_signature,
    validate_checkpoint_manifest_vocab_provenance,
    validate_checkpoint_stage,
    validate_posthoc_oof_rows,
    validate_posthoc_oof_rows_identity,
)


class _CalibrationGraph:
    def __init__(self, evidence: torch.Tensor):
        self.evidence = evidence

    def to(self, device, non_blocking=True):
        self.evidence = self.evidence.to(device)
        return self


class _FinalTemperatureOnlyFusion(torch.nn.Module):
    combination = "routed"

    def __init__(self):
        super().__init__()
        self.log_final_temperature = torch.nn.Parameter(
            torch.tensor(0.0), requires_grad=False
        )

    def reliability_calibration_parameters(self):
        return []

    def routing_calibration_parameters(self):
        return []

    def final_temperature_parameters(self):
        return [self.log_final_temperature]

    def forward(self, api, graph, manifest, evidence):
        raw = F.log_softmax(
            (api + graph + manifest) / 3.0,
            dim=-1,
        )
        temperature = bounded_final_temperature(
            self.log_final_temperature
        )
        calibrated = F.log_softmax(
            raw / temperature,
            dim=-1,
        )
        return {
            "uncalibrated_final_log_prob": raw,
            "final_log_prob": calibrated,
            "final_logits": calibrated,
        }


class _FinalTemperatureOnlyModel(torch.nn.Module):
    fusion_mode = "discount_probability"

    def __init__(self):
        super().__init__()
        self.discount_fusion = _FinalTemperatureOnlyFusion()
        self.calibration_active = False

    def calibration_parameters(self):
        return []

    def set_calibration_active(self, enabled: bool):
        self.calibration_active = bool(enabled)

    def forward(self, graph):
        batch_size = graph.evidence.size(0)
        logits = graph.evidence.new_tensor(
            [[4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, -4.0]]
        )[:batch_size]
        outputs = self.discount_fusion(logits, logits, logits, graph.evidence)
        for name in ("api", "graph", "manifest"):
            outputs[f"{name}_logits_aux"] = logits
        outputs["fusion_availability"] = graph.evidence
        outputs["selective_eligible"] = graph.evidence[:, :3].gt(0).any(dim=-1)
        return outputs["final_logits"], outputs


def test_restored_gradient_rejects_nonstationary_plateau():
    converged, reason = _resolve_restored_stage_convergence(
        provisional_stop_reason="objective_plateau",
        final_grad_inf_norm=1.0e-3,
        gradient_tolerance=1.0e-5,
    )

    assert converged is False
    assert reason == "objective_plateau_nonstationary"


def test_restored_gradient_accepts_stationary_best():
    converged, reason = _resolve_restored_stage_convergence(
        provisional_stop_reason="max_steps",
        final_grad_inf_norm=1.0e-7,
        gradient_tolerance=1.0e-5,
    )

    assert converged is True
    assert reason == "restored_best_gradient_tolerance"


@pytest.mark.parametrize(
    ("final_grad_inf_norm", "gradient_tolerance"),
    [
        (float("nan"), 1.0e-5),
        (float("inf"), 1.0e-5),
        (1.0e-4, 0.0),
        (1.0e-4, -1.0e-5),
    ],
)
def test_restored_convergence_rejects_invalid_gradient_state(
    final_grad_inf_norm: float,
    gradient_tolerance: float,
):
    converged, reason = _resolve_restored_stage_convergence(
        provisional_stop_reason="max_steps",
        final_grad_inf_norm=final_grad_inf_norm,
        gradient_tolerance=gradient_tolerance,
    )

    assert converged is False
    assert reason == "restored_best_invalid_gradient"


def test_provisional_gradient_is_rechecked_after_restore():
    converged, reason = _resolve_restored_stage_convergence(
        provisional_stop_reason="gradient_tolerance",
        final_grad_inf_norm=1.0e-3,
        gradient_tolerance=1.0e-5,
    )

    assert converged is False
    assert reason == "provisional_gradient_not_valid_at_restored_best"


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


def test_raw_oof_cutoff_maps_to_identical_decisions_at_any_positive_temperature():
    rows = [
        {
            "sid": f"s{index}",
            "group": f"g{index}",
            "label": label,
            "raw_log_prob": F.log_softmax(torch.tensor([0.0, score]), dim=0).tolist(),
        }
        for index, (score, label) in enumerate(
            [(-3.0, 0), (-0.5, 0), (0.4, 1), (2.0, 1)]
        )
    ]
    config = {
        "enabled": True,
        "objective": "macro_f1",
        "selection_rule": "macro_f1_unconstrained_v1",
    }
    summaries = [
        fit_oof_malware_classification_threshold(
            rows, config, deployment_temperature=temperature
        )
        for temperature in (0.25, 1.0, 4.0)
    ]
    assert all(summary is not None for summary in summaries)
    raw_cutoffs = {
        float(summary["raw_log_odds_threshold"])
        for summary in summaries
        if summary is not None
    }
    assert len(raw_cutoffs) == 1
    raw_cutoff = next(iter(raw_cutoffs))
    scores = torch.tensor([-3.0, -0.5, 0.4, 2.0])
    expected = scores >= raw_cutoff
    for temperature, summary in zip((0.25, 1.0, 4.0), summaries):
        assert summary is not None
        probabilities = torch.sigmoid(scores / temperature)
        assert torch.equal(
            probabilities >= float(summary["threshold"]), expected
        )


def test_oof_payload_validation_rejects_truncation_against_identity():
    rows = validate_posthoc_oof_rows(
        [
            {
                "sid": "a",
                "group": "pkg.a",
                "label": 0,
                "raw_log_prob": F.log_softmax(torch.tensor([2.0, -2.0]), dim=0).tolist(),
            },
            {
                "sid": "b",
                "group": "pkg.b",
                "label": 1,
                "raw_log_prob": F.log_softmax(torch.tensor([-2.0, 2.0]), dim=0).tolist(),
            },
        ]
    )
    class Dataset:
        sample_sids = ["a", "b"]
        sample_groups = ["pkg.a", "pkg.b"]
        sample_labels = [0, 1]

        def __len__(self):
            return 2

    identity = {
        "posthoc_calibration": _calibration_subset_identity(Dataset(), None)
    }
    validate_posthoc_oof_rows_identity(rows, identity)
    with pytest.raises(ValueError, match="incomplete"):
        validate_posthoc_oof_rows_identity(rows[:1], identity)
    tampered = [dict(row) for row in rows]
    tampered[1]["sid"] = "c"
    with pytest.raises(ValueError, match="disagrees"):
        validate_posthoc_oof_rows_identity(tampered, identity)


@pytest.mark.parametrize("label", [0.5, 1.5, 2, float("nan"), float("inf"), True])
def test_oof_payload_rejects_lossy_or_nonfinite_labels(label: object):
    with pytest.raises(ValueError, match="invalid label"):
        validate_posthoc_oof_rows(
            [
                {
                    "sid": "sample",
                    "group": "pkg.sample",
                    "label": label,
                    "raw_log_prob": F.log_softmax(
                        torch.tensor([1.0, -1.0]), dim=0
                    ).tolist(),
                }
            ]
        )


def test_evaluate_uses_raw_cutoff_when_temperature_probability_saturates():
    class SaturatedBinaryModel(torch.nn.Module):
        def forward(self, graph):
            raw = F.log_softmax(graph.evidence.new_tensor([[0.0, 0.5]]), dim=-1)
            final = F.log_softmax(raw / 1.0e-8, dim=-1)
            return final, {
                "uncalibrated_final_log_prob": raw,
                "selective_eligible": torch.ones(
                    raw.size(0), dtype=torch.bool, device=raw.device
                ),
            }

    loader = [
        {
            "graph_batch": _CalibrationGraph(torch.ones(1, 24)),
            "labels": torch.tensor([0]),
            "sids": ["a"],
            "quality": {},
            "num_failed": 0,
        }
    ]
    metrics, _ = evaluate(
        SaturatedBinaryModel(),
        loader,
        torch.device("cpu"),
        False,
        "saturated",
        classification_threshold=1.0,
        classification_log_odds_threshold=1.0,
    )

    assert metrics["acc"] == pytest.approx(1.0)


def test_posthoc_calibration_supports_final_temperature_only_stage():
    evidence = torch.ones(4, 24)
    loader = [
        {
            "graph_batch": _CalibrationGraph(evidence),
            "labels": torch.tensor([0, 0, 1, 1]),
            "sids": ["a", "b", "c", "d"],
            "quality": {},
            "num_failed": 0,
        }
    ]
    model = _FinalTemperatureOnlyModel()

    summary = fit_posthoc_calibration(
        model,
        [loader],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "stage_optimization": {
                    "default": {
                        "max_steps": 1,
                        "min_steps": 1,
                        "require_convergence": False,
                    }
                },
            },
            "fusion": {},
        },
    )

    assert summary["enabled"] is True
    assert summary["final_temperature"]["enabled"] is True
    assert summary["stages"]["final_temperature"]["enabled"] is True
    assert summary["stages"]["final_temperature"]["total_steps"] == 1
    assert summary["total_optimization_steps"] == 1
    assert summary["stage_grouping"] == "sample_id_fallback"
    assert model.discount_fusion.log_final_temperature.requires_grad is False


def test_posthoc_fold_assignment_keeps_package_group_together():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    package_groups = [
        "pkg.a",
        "pkg.a",
        "pkg.b",
        "pkg.c",
        "pkg.x",
        "pkg.x",
        "pkg.y",
        "pkg.z",
    ]

    folds = deterministic_stratified_fold_ids(
        labels,
        package_groups,
        num_folds=3,
        seed=42,
    )

    assert folds[0].item() == folds[1].item()
    assert folds[4].item() == folds[5].item()


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
    pipeline_model = {"opinion_router.weight": torch.tensor([9.0])}
    pipeline = {
        "checkpoint_stage": CHECKPOINT_STAGE_PIPELINE_FITTED,
        "pipeline_artifact_schema_version": PIPELINE_ARTIFACT_SCHEMA_VERSION,
        "encoder_checkpoint_path": encoder_path.name,
        "encoder_checkpoint_sha256": _file_sha256(encoder_path),
        "model": pipeline_model,
        "pipeline_model_state_sha256": _state_dict_sha256(pipeline_model),
        "classification_threshold": {"threshold": 0.5},
    }
    pipeline["pipeline_decision_metadata_sha256"] = (
        _pipeline_decision_metadata_sha256(pipeline)
    )
    torch.save(pipeline, pipeline_path)
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
    pipeline_model = {"placeholder.weight": torch.tensor([1.0])}
    torch.save(
        {
            "checkpoint_stage": CHECKPOINT_STAGE_PIPELINE_FITTED,
            "pipeline_artifact_schema_version": PIPELINE_ARTIFACT_SCHEMA_VERSION,
            "model": pipeline_model,
            "pipeline_model_state_sha256": _state_dict_sha256(pipeline_model),
        },
        pipeline_path,
    )
    with pytest.raises(ValueError, match="does not link"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=True,
            map_location="cpu",
        )

    encoder_path = tmp_path / "encoder_without_link_hash.pt"
    torch.save(
        {"checkpoint_stage": CHECKPOINT_STAGE_ENCODER_SELECTED, "model": {}},
        encoder_path,
    )
    pipeline_path = tmp_path / "pipeline_without_encoder_hash.pt"
    pipeline_model = {"placeholder.weight": torch.tensor([1.0])}
    torch.save(
        {
            "checkpoint_stage": CHECKPOINT_STAGE_PIPELINE_FITTED,
            "pipeline_artifact_schema_version": PIPELINE_ARTIFACT_SCHEMA_VERSION,
            "encoder_checkpoint_path": encoder_path.name,
            "model": pipeline_model,
            "pipeline_model_state_sha256": _state_dict_sha256(pipeline_model),
        },
        pipeline_path,
    )
    with pytest.raises(ValueError, match="encoder_checkpoint_sha256"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=True,
            map_location="cpu",
        )


def test_checkpoint_manifest_vocab_provenance_is_strict():
    current = {
        "required": True,
        "verified": True,
        "manifest_vocab_sha256": "a" * 64,
        "train_csv_sha256": "b" * 64,
        "train_sample_ids_sha256": "c" * 64,
        "num_train_samples": 17,
    }
    validate_checkpoint_manifest_vocab_provenance(
        {"manifest_vocab_provenance": dict(current)}, current
    )

    with pytest.raises(ValueError, match="no verified Manifest"):
        validate_checkpoint_manifest_vocab_provenance({}, current)

    stale = dict(current)
    stale["manifest_vocab_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="different Manifest vocabulary"):
        validate_checkpoint_manifest_vocab_provenance(
            {"manifest_vocab_provenance": stale}, current
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


def test_pipeline_model_state_hash_is_mandatory_and_verified(tmp_path):
    _, pipeline_path = _write_staged_checkpoints(tmp_path)
    checkpoint = torch.load(pipeline_path, map_location="cpu", weights_only=True)
    checkpoint["model"]["opinion_router.weight"] = torch.tensor([10.0])
    torch.save(checkpoint, pipeline_path)

    with pytest.raises(ValueError, match="model-state hash mismatch"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=False,
            map_location="cpu",
        )


def test_pipeline_decision_metadata_hash_is_mandatory_and_verified(tmp_path):
    _, pipeline_path = _write_staged_checkpoints(tmp_path)
    checkpoint = torch.load(pipeline_path, map_location="cpu", weights_only=True)
    checkpoint.pop("pipeline_decision_metadata_sha256")
    torch.save(checkpoint, pipeline_path)

    with pytest.raises(ValueError, match="pipeline_decision_metadata_sha256"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=False,
            map_location="cpu",
        )

    _, pipeline_path = _write_staged_checkpoints(tmp_path)
    checkpoint = torch.load(pipeline_path, map_location="cpu", weights_only=True)
    checkpoint["classification_threshold"]["threshold"] = 0.75
    torch.save(checkpoint, pipeline_path)

    with pytest.raises(ValueError, match="decision-metadata hash mismatch"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=False,
            map_location="cpu",
        )


def test_pipeline_decision_metadata_hash_covers_oof_rows(tmp_path):
    _, pipeline_path = _write_staged_checkpoints(tmp_path)
    checkpoint = torch.load(pipeline_path, map_location="cpu", weights_only=True)
    checkpoint["posthoc_oof_clean_rows"] = [
        {
            "sid": "sample-a",
            "group": "package:a",
            "label": 1,
            "raw_log_prob": [-2.0, -0.14541345786885906],
        }
    ]
    checkpoint["posthoc_oof_clean_rows_schema_version"] = 1
    checkpoint["pipeline_decision_metadata_sha256"] = (
        _pipeline_decision_metadata_sha256(checkpoint)
    )
    torch.save(checkpoint, pipeline_path)

    checkpoint["posthoc_oof_clean_rows"][0]["label"] = 0
    torch.save(checkpoint, pipeline_path)
    with pytest.raises(ValueError, match="decision-metadata hash mismatch"):
        _load_eval_checkpoint(
            pipeline_path,
            refit_posthoc_calibration=False,
            map_location="cpu",
        )


def test_pipeline_decision_metadata_hash_distinguishes_infinity_and_none():
    infinite = {"conformal_thresholds": {"q_malware": float("inf")}}
    absent = {"conformal_thresholds": {"q_malware": None}}
    assert _pipeline_decision_metadata_sha256(infinite) != (
        _pipeline_decision_metadata_sha256(absent)
    )

    with pytest.raises(ValueError, match="not canonically serializable"):
        _pipeline_decision_metadata_sha256(
            {"conformal_thresholds": {"q_malware": float("nan")}}
        )


def test_refit_decision_calibration_rejects_removed_alias():
    assert _resolve_refit_decision_calibration({"refit_decision_calibration": True})
    assert not _resolve_refit_decision_calibration({})
    with pytest.raises(ValueError, match="was removed"):
        _resolve_refit_decision_calibration(
            {"refit_rejection_threshold": True}
        )


def test_final_temperature_override_requires_decision_refit_when_thresholds_are_used():
    cfg = {
        "train": {"seed": 42, "device": "cpu"},
        "data": {},
        "model": {"fusion_mode": "discount_probability"},
        "fusion": {"mode": "discount_probability"},
        "classification_threshold": {"enabled": True},
        "selective_prediction": {"enabled": False},
        "eval": {
            "eval_only": True,
            "checkpoint_path": "unused.pt",
            "final_temperature_override": 1.0,
            "refit_decision_calibration": False,
        },
    }
    with pytest.raises(
        ValueError,
        match="Final-temperature override.*refit_decision_calibration=true",
    ):
        run_training(cfg)


def _decision_cfg(risk_level: float = 0.05) -> dict:
    return {
        "classification_threshold": {
            "enabled": True,
            "objective": "macro_f1",
            "selection_rule": "macro_f1_unconstrained_v1",
        },
        "selective_prediction": {
            "enabled": True,
            "mode": "risk_control",
            "threshold_score": "model_acceptance",
            "risk_target": "accepted_fn_risk_among_malware",
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
    changed_score = copy.deepcopy(_decision_cfg(0.05))
    changed_score["selective_prediction"]["threshold_score"] = "msp"
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


def test_decision_signature_records_strict_acceptance_comparison():
    signature = _decision_calibration_signature(_decision_cfg())
    assert signature["selective"]["acceptance_comparison"] == (
        "selective_eligible and score > threshold"
    )


def test_decision_data_identity_hashes_rows_groups_classes_and_csv(tmp_path):
    class Dataset:
        sample_sids = ["a", "b", "c", "d"]
        sample_groups = ["pkg.a", "pkg.b", "pkg.c", "pkg.d"]
        sample_labels = [0, 1, 0, 1]

        def __len__(self):
            return 4

    val_csv = tmp_path / "val.csv"
    val_csv.write_text("id,label\na,0\nb,1\nc,0\nd,1\n", encoding="utf-8")
    cfg = {"data": {"root": str(tmp_path), "val_csv": val_csv.name}}
    identity = _decision_calibration_data_identity(
        cfg,
        Dataset(),
        posthoc_indices=[0, 1],
        decision_indices=[2, 3],
    )

    assert identity["posthoc_calibration"]["class_counts"] == {"0": 1, "1": 1}
    assert identity["decision_calibration"]["class_counts"] == {"0": 1, "1": 1}
    checkpoint = {
        "decision_calibration_signature": _decision_calibration_signature(
            _decision_cfg()
        ),
        "decision_calibration_data_identity": identity,
    }
    validate_checkpoint_decision_signature(
        _decision_cfg(),
        checkpoint,
        refit_decision_calibration=False,
        current_data_identity=identity,
    )
    checkpoint_with_oof = {
        **checkpoint,
        "posthoc_oof_clean_rows": [{"sid": "a"}],
    }
    validate_checkpoint_decision_signature(
        _decision_cfg(),
        checkpoint_with_oof,
        refit_decision_calibration=True,
        current_data_identity=identity,
    )
    changed = {
        **identity,
        "decision_calibration": {
            **identity["decision_calibration"],
            "row_identity_sha256": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="data identity differs"):
        validate_checkpoint_decision_signature(
            _decision_cfg(),
            checkpoint,
            refit_decision_calibration=False,
            current_data_identity=changed,
        )
    changed_posthoc = {
        **identity,
        "posthoc_calibration": {
            **identity["posthoc_calibration"],
            "row_identity_sha256": "1" * 64,
        },
    }
    with pytest.raises(ValueError, match="Post-hoc data identity differs"):
        validate_checkpoint_decision_signature(
            _decision_cfg(),
            checkpoint_with_oof,
            refit_decision_calibration=True,
            current_data_identity=changed_posthoc,
        )
    with pytest.raises(ValueError, match="has no decision_calibration_data_identity"):
        validate_checkpoint_decision_signature(
            _decision_cfg(),
            {
                "decision_calibration_signature": _decision_calibration_signature(
                    _decision_cfg()
                )
            },
            refit_decision_calibration=False,
            current_data_identity=identity,
        )


def test_i1_correctness_and_i2_threshold_fn_risk_metrics_are_reported():
    rows = [
        {
            "routing_active": 1,
            "routing_has_available": 1,
            "routing_risk_probability": 0.2,
            "routing_mixture_pred": 1,
            "label": 0,
            "pred": 0,
            "routing_risk_decision_threshold_active": 1,
            "routing_risk_target_threshold_malware_false_negative": 1,
            "api_alive": 1,
            "graph_alive": 1,
            "manifest_alive": 1,
            "api_correct": 1,
            "graph_correct": 0,
            "manifest_correct": 1,
            "predicted_reliability_api": 0.8,
            "predicted_reliability_graph": 0.3,
            "predicted_reliability_manifest": 0.7,
        },
        {
            "routing_active": 1,
            "routing_has_available": 1,
            "routing_risk_probability": 0.8,
            "routing_mixture_pred": 1,
            "label": 1,
            "pred": 0,
            "routing_risk_decision_threshold_active": 1,
            "routing_risk_target_threshold_malware_false_negative": 1,
            "api_alive": 1,
            "graph_alive": 0,
            "manifest_alive": 1,
            "api_correct": 0,
            "graph_correct": 1,
            "manifest_correct": 1,
            "predicted_reliability_api": 0.2,
            "predicted_reliability_graph": 0.9,
            "predicted_reliability_manifest": 0.9,
        },
    ]

    metrics = compute_branch_reliability_metrics(rows)

    assert metrics["api_reliability_count"] == 2
    assert metrics["graph_reliability_count"] == 1
    assert metrics["manifest_reliability_count"] == 2
    assert metrics["api_reliability_brier"] == pytest.approx(0.04)
    assert metrics["graph_reliability_brier"] == pytest.approx(0.09)
    assert metrics["manifest_reliability_brier"] == pytest.approx(0.05)
    assert metrics["routing_risk_count"] == 2
    assert metrics["routing_risk_target"] == "threshold_malware_false_negative"
    assert metrics["routing_risk_brier"] == pytest.approx(0.04)
    assert metrics["routing_risk_mean"] == pytest.approx(0.5)
    assert metrics["routing_mixture_error_rate"] == pytest.approx(0.5)
    assert metrics["routing_risk_error_rate_gap"] == pytest.approx(0.0)
    assert metrics["routing_risk_auc"] == pytest.approx(1.0)
    assert metrics["routing_risk_ap"] == pytest.approx(1.0)
    assert metrics["routing_all_missing_count"] == 0
    assert metrics["routing_all_missing_forced_rejection_count"] == 0


def test_all_missing_rows_are_excluded_from_i2_risk_metrics_and_counted():
    metrics = compute_branch_reliability_metrics(
        [
            {
                "routing_active": 1,
                "routing_has_available": 1,
                "routing_risk_probability": 0.2,
                "routing_mixture_pred": 0,
                "label": 0,
                "pred": 0,
                "routing_risk_decision_threshold_active": 1,
                "routing_risk_target_threshold_malware_false_negative": 1,
            },
            {
                "routing_active": 1,
                "routing_has_available": 1,
                "routing_risk_probability": 0.8,
                "routing_mixture_pred": 0,
                "label": 1,
                "pred": 0,
                "routing_risk_decision_threshold_active": 1,
                "routing_risk_target_threshold_malware_false_negative": 1,
            },
            {
                "routing_active": 1,
                "routing_has_available": 0,
                "routing_risk_probability": 1.0,
                # This row would add an error if all-missing cases were folded
                # into calibration metrics instead of being audited separately.
                "routing_mixture_pred": 1,
                "label": 0,
                "api_alive": 0,
                "graph_alive": 0,
                "manifest_alive": 0,
            },
            {
                "routing_active": 0,
                "routing_has_available": 0,
                "routing_risk_probability": 1.0,
                "routing_mixture_pred": 1,
                "label": 0,
                "api_alive": 1,
                "graph_alive": 1,
                "manifest_alive": 1,
            }
        ]
    )

    assert metrics["routing_risk_count"] == 2
    assert metrics["routing_risk_brier"] == pytest.approx(0.04)
    assert metrics["routing_mixture_error_rate"] == pytest.approx(0.5)
    assert metrics["routing_all_missing_count"] == 1
    assert metrics["routing_all_missing_forced_rejection_count"] == 1


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
        {
            "acceptance_score": 0.9,
            "label": 1,
            "pred": 1,
            "selective_eligible": 1,
        }
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
    assert fitted["guarantee_scope"] == "exchangeable_expected_risk"
    assert fitted["risk_denominator"] == "all_malware"
    assert fitted["risk_numerator"] == "accepted_and_predicted_benign_malware"
