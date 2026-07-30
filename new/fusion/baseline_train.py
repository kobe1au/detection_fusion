from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from fusion.dataset import prepare_robust_batch
from fusion.losses import compute_robust_loss
from fusion.model import TriModalRobustModel
from fusion.runtime import (
    _atomic_torch_save,
    _file_sha256,
    _json_compatible,
    _load_fixed_validation_roles,
    _state_dict_sha256,
    build_dataset,
    build_loader,
    configure_determinism,
    configure_multiprocessing_sharing,
    enforce_failed_ratio,
    iter_robust_test_loaders,
    load_config,
    select_device,
    set_seed,
    split_validation_dataset,
    STAGE_A_EXPERT_TRAIN_LOADER_NAMESPACE,
    STAGE_A_EXPERT_VAL_LOADER_NAMESPACE,
    validate_manifest_vocab_provenance,
    validate_split_partitions,
)
from fusion.utils import (
    build_grad_scaler,
    get_amp_context,
    strict_finite_integer,
)
from fusion.view_protocol import (
    CONTROLLED_TEST_VIEW_PROTOCOL_SEED,
    CONTROLLED_VIEW_MECHANISM_VERSION,
    CONTROLLED_VIEW_SEED_FORMULA,
    canonical_manifest_sha256,
)


logger = logging.getLogger("registered_baseline")

BASELINE_ARTIFACT_SCHEMA_VERSION = 3
BASELINE_METRIC_SCHEMA_VERSION = 16
BASELINE_RUNNER_PROTOCOL_ID = "registered_baseline_runner_v2"
BASELINE_CHECKPOINT_NAME = "best_baseline.pt"
BASELINE_TRAINING_ROLE_PROTOCOL_ID = (
    "baseline_expert_train_expert_val_package_group_disjoint_v1"
)
BASELINE_CLASSIFICATION_PROTOCOL_ID = "binary_argmax_fixed_0_5_v1"
BASELINE_EXPERT_SPLIT_SEED = 4242
BASELINE_EXPERT_VAL_FRACTION = 0.10

FORMAL_PERTURBATIONS = (
    "clean",
    "api_event_dropout",
    "graph_sparsify",
    "manifest_permission_mask",
    "api_missing",
    "graph_missing",
    "manifest_missing",
)
FORMAL_STRENGTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
FORMAL_ROBUST_CELL_COUNT = 19
BRANCHES = ("api", "graph", "manifest")
BRANCH_LOGIT_KEYS = {
    "api": "api_logits_aux",
    "graph": "graph_logits_aux",
    "manifest": "manifest_logits_aux",
}
DIAGNOSTIC_KEYS = (
    "fusion_weight_api",
    "fusion_weight_graph",
    "fusion_weight_manifest",
    "qmf_energy_api",
    "qmf_energy_graph",
    "qmf_energy_manifest",
    "uncertainty_proxy_api",
    "uncertainty_proxy_graph",
    "uncertainty_proxy_manifest",
    "raw_conflict",
)


@dataclass(frozen=True)
class BaselineIdentity:
    method_name: str
    protocol_id: str
    encoder_protocol_id: str
    fusion_mode: str
    fusion_dispatch: str
    combination: str
    opinion_source: str
    hard_alive: bool
    objective: str
    branch_aux_weight: float
    evidential_loss_weight: float
    auxiliary_weight_mode: str
    branch_aux_weights: tuple[float, float, float]
    quality_temperature: float | None = None
    gate_hidden_dim: int = 128
    gate_detach: bool = True
    anneal_epochs: int | None = None
    ecml_consistency_weight: float | None = None


def _identity(
    method_name: str,
    protocol_id: str,
    fusion_mode: str,
    *,
    encoder_protocol_id: str = "comparison_method_specific_stage1_v1",
    fusion_dispatch: str = "model_dispatch",
    combination: str = "",
    opinion_source: str = "",
    hard_alive: bool = False,
    objective: str = "standard",
    branch_aux_weight: float = 0.0,
    evidential_loss_weight: float = 0.0,
    auxiliary_weight_mode: str = "unmasked_uniform",
    branch_aux_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    quality_temperature: float | None = None,
    gate_hidden_dim: int = 128,
    gate_detach: bool = True,
    anneal_epochs: int | None = None,
    ecml_consistency_weight: float | None = None,
) -> BaselineIdentity:
    return BaselineIdentity(
        method_name=method_name,
        protocol_id=protocol_id,
        encoder_protocol_id=encoder_protocol_id,
        fusion_mode=fusion_mode,
        fusion_dispatch=fusion_dispatch,
        combination=combination,
        opinion_source=opinion_source,
        hard_alive=hard_alive,
        objective=objective,
        branch_aux_weight=branch_aux_weight,
        evidential_loss_weight=evidential_loss_weight,
        auxiliary_weight_mode=auxiliary_weight_mode,
        branch_aux_weights=branch_aux_weights,
        quality_temperature=quality_temperature,
        gate_hidden_dim=gate_hidden_dim,
        gate_detach=gate_detach,
        anneal_epochs=anneal_epochs,
        ecml_consistency_weight=ecml_consistency_weight,
    )


REGISTERED_BASELINES: dict[str, BaselineIdentity] = {
    item.protocol_id: item
    for item in (
        _identity(
            "baseline_api_only",
            "baseline_api_only_v1",
            "api_only",
        ),
        _identity(
            "baseline_graph_only",
            "baseline_graph_only_v1",
            "graph_only",
        ),
        _identity(
            "baseline_manifest_only",
            "baseline_manifest_only_v1",
            "manifest_only",
        ),
        _identity(
            "baseline_api_graph_concat",
            "baseline_api_graph_concat_v1",
            "api_graph_concat",
            branch_aux_weight=0.25,
            branch_aux_weights=(1.0, 1.0, 0.0),
        ),
        _identity(
            "baseline_tri_modal_concat",
            "baseline_tri_modal_concat_v1",
            "tri_modal_concat",
            branch_aux_weight=0.25,
        ),
        _identity(
            "baseline_fixed_logit_fusion",
            "baseline_fixed_logit_fusion_v2",
            "tri_modal_fixed_gate",
            branch_aux_weight=0.25,
        ),
        _identity(
            "dense_embedding_gate_adapted",
            "dense_embedding_gate_adapted_v1",
            "tri_modal_dense_embedding_gate",
            branch_aux_weight=0.25,
            auxiliary_weight_mode="alive_masked_uniform",
            gate_detach=False,
        ),
        _identity(
            "fusion_rule_dempster",
            "fixed_evidential_dempster_v1",
            "discount_probability",
            encoder_protocol_id="fixed_evidential_dempster_stage1_v1",
            fusion_dispatch="discount_probability",
            combination="dempster",
            opinion_source="evidential",
            hard_alive=True,
            branch_aux_weight=0.25,
            evidential_loss_weight=0.05,
        ),
        _identity(
            "fusion_rule_cumulative",
            "fixed_evidential_cumulative_v1",
            "discount_probability",
            encoder_protocol_id="fixed_evidential_cumulative_stage1_v1",
            fusion_dispatch="discount_probability",
            combination="cumulative",
            opinion_source="evidential",
            hard_alive=True,
            branch_aux_weight=0.25,
            evidential_loss_weight=0.05,
        ),
        _identity(
            "fusion_rule_log_pool",
            "fixed_evidential_log_pool_v1",
            "discount_probability",
            encoder_protocol_id="fixed_evidential_log_pool_stage1_v1",
            fusion_dispatch="discount_probability",
            combination="log_pool",
            opinion_source="evidential",
            hard_alive=True,
            branch_aux_weight=0.25,
            evidential_loss_weight=0.05,
        ),
        _identity(
            "fusion_rule_conflict_weighted_opinion",
            "fixed_evidential_conflict_weighted_opinion_v1",
            "discount_probability",
            encoder_protocol_id=(
                "fixed_evidential_conflict_weighted_stage1_v1"
            ),
            fusion_dispatch="discount_probability",
            combination="conflict_weighted_opinion",
            opinion_source="evidential",
            hard_alive=True,
            branch_aux_weight=0.25,
            evidential_loss_weight=0.05,
        ),
        _identity(
            "tmc_style_adapted",
            "tmc_style_adapted_v1",
            "discount_probability",
            encoder_protocol_id="trusted_comparison_specific_stage1_v1",
            fusion_dispatch="discount_probability",
            combination="dempster",
            opinion_source="evidential",
            hard_alive=True,
            objective="tmc",
            anneal_epochs=10,
        ),
        _identity(
            "qmf_energy_component",
            "qmf_energy_v1",
            "tri_modal_quality_fusion",
            encoder_protocol_id="qmf_energy_component_stage1_v1",
            hard_alive=True,
            branch_aux_weight=0.25,
            quality_temperature=10.0,
        ),
        _identity(
            "ecml_style_adapted",
            "ecml_style_adapted_v1",
            "discount_probability",
            encoder_protocol_id="trusted_comparison_specific_stage1_v1",
            fusion_dispatch="discount_probability",
            combination="ecml",
            opinion_source="evidential",
            hard_alive=True,
            objective="ecml",
            anneal_epochs=10,
            ecml_consistency_weight=1.0,
        ),
    )
}

if len(REGISTERED_BASELINES) != 14:
    raise RuntimeError("The formal baseline registry must contain exactly 14 methods")


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _expect_equal(field: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        try:
            numeric = float(actual)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Registered baseline identity mismatch: {field} must be "
                f"{expected!r}, got {actual!r}"
            ) from exc
        if not math.isfinite(numeric) or not math.isclose(
            numeric, expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Registered baseline identity mismatch: {field} must be "
                f"{expected!r}, got {actual!r}"
            )
        return
    if actual != expected:
        raise ValueError(
            f"Registered baseline identity mismatch: {field} must be "
            f"{expected!r}, got {actual!r}"
        )


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean, got {value!r}")
    return value


def _reject_any_enabled(cfg: Mapping[str, Any]) -> None:
    fusion = _mapping(cfg.get("fusion", {}), field="fusion")
    reliability = _mapping(
        fusion.get("reliability_calibration", {}),
        field="fusion.reliability_calibration",
    )
    routing = _mapping(
        fusion.get("routing", {}),
        field="fusion.routing",
    )
    calibration = _mapping(cfg.get("calibration", {}), field="calibration")
    selective = _mapping(
        cfg.get("selective_prediction", {}),
        field="selective_prediction",
    )
    eval_cfg = _mapping(cfg.get("eval", {}), field="eval")
    active = []
    for field, value in (
        ("fusion.use_i1_reliability", fusion.get("use_i1_reliability", False)),
        (
            "fusion.reliability_calibration.enabled",
            reliability.get("enabled", False),
        ),
        ("fusion.routing.enabled", routing.get("enabled", False)),
        ("fusion.routing.posthoc_refine", routing.get("posthoc_refine", False)),
        ("fusion.routing.final_temperature_scaling", routing.get("final_temperature_scaling", False)),
        ("calibration.enabled", calibration.get("enabled", False)),
        ("selective_prediction.enabled", selective.get("enabled", False)),
        ("eval.eval_only", eval_cfg.get("eval_only", False)),
        (
            "eval.refit_posthoc_calibration",
            eval_cfg.get("refit_posthoc_calibration", False),
        ),
        (
            "eval.refit_decision_calibration",
            eval_cfg.get("refit_decision_calibration", False),
        ),
    ):
        if _strict_bool(value, field=field):
            active.append(field)
    forbidden_present = [
        field
        for field in (
            "checkpoint_path",
            "final_temperature_override",
            "extra_sets",
        )
        if eval_cfg.get(field) not in (None, "", [], {})
    ]
    active.extend(f"eval.{field}" for field in forbidden_present)
    if active:
        raise ValueError(
            "The registered baseline runner forbids eval-only, encoder reuse, "
            "post-hoc reliability/routing, selective prediction, and external "
            f"evaluation lifecycles; active={active}"
        )


def validate_registered_baseline_config(
    cfg: Mapping[str, Any],
) -> BaselineIdentity:
    """Fail closed unless the resolved config is one registered paper baseline."""

    method = _mapping(cfg.get("method", {}), field="method")
    protocol_id = str(method.get("protocol_id") or "").strip()
    if protocol_id not in REGISTERED_BASELINES:
        raise ValueError(
            "The baseline runner accepts only the 14 registered method "
            f"protocols; got method.protocol_id={protocol_id!r}"
        )
    expected = REGISTERED_BASELINES[protocol_id]
    model = _mapping(cfg.get("model", {}), field="model")
    fusion = _mapping(cfg.get("fusion", {}), field="fusion")
    loss = _mapping(cfg.get("loss", {}), field="loss")
    train = _mapping(cfg.get("train", {}), field="train")
    encoder_stage = _mapping(
        cfg.get("encoder_stage", {}), field="encoder_stage"
    )
    calibration = _mapping(cfg.get("calibration", {}), field="calibration")
    threshold = _mapping(
        cfg.get("classification_threshold", {}),
        field="classification_threshold",
    )
    eval_cfg = _mapping(cfg.get("eval", {}), field="eval")
    gate = _mapping(model.get("gate", {}), field="model.gate")
    routing = _mapping(fusion.get("routing", {}), field="fusion.routing")
    branch_weights = _mapping(
        loss.get("branch_aux_weights", {}),
        field="loss.branch_aux_weights",
    )
    for key, default in (
        ("use_amp", True),
        ("deterministic", True),
        ("strict_deterministic", False),
        ("pin_memory", False),
        ("allow_pyg_pin_memory", False),
        ("persistent_workers", True),
    ):
        _strict_bool(
            train.get(key, default),
            field=f"train.{key}",
        )

    _expect_equal("method.name", str(method.get("name") or ""), expected.method_name)
    _expect_equal("method.protocol_id", protocol_id, expected.protocol_id)
    _expect_equal(
        "encoder_stage.protocol_id",
        str(encoder_stage.get("protocol_id") or ""),
        expected.encoder_protocol_id,
    )
    _expect_equal("encoder_stage.mode", str(encoder_stage.get("mode", "")), "fit")
    _expect_equal(
        "encoder_stage.checkpoint_path",
        encoder_stage.get("checkpoint_path"),
        None,
    )
    _expect_equal(
        "encoder_stage.expected_sha256",
        encoder_stage.get("expected_sha256"),
        None,
    )
    _expect_equal(
        "encoder_stage.strict_identity",
        _strict_bool(
            encoder_stage.get("strict_identity", True),
            field="encoder_stage.strict_identity",
        ),
        True,
    )
    _expect_equal(
        "model.fusion_mode",
        str(model.get("fusion_mode", "")).strip().lower(),
        expected.fusion_mode,
    )
    _expect_equal(
        "fusion.mode",
        str(fusion.get("mode", "")).strip().lower(),
        expected.fusion_dispatch,
    )
    _expect_equal(
        "fusion.combination",
        str(fusion.get("combination") or "").strip().lower(),
        expected.combination,
    )
    _expect_equal(
        "fusion.opinion_source",
        str(fusion.get("opinion_source") or "").strip().lower(),
        expected.opinion_source,
    )
    _expect_equal(
        "fusion.use_hard_alive_mask",
        _strict_bool(
            fusion.get("use_hard_alive_mask", False),
            field="fusion.use_hard_alive_mask",
        ),
        expected.hard_alive,
    )
    _expect_equal(
        "fusion.routing.enabled",
        _strict_bool(
            routing.get("enabled", False),
            field="fusion.routing.enabled",
        ),
        False,
    )
    _expect_equal(
        "fusion.routing.final_temperature_scaling",
        _strict_bool(
            routing.get("final_temperature_scaling", False),
            field="fusion.routing.final_temperature_scaling",
        ),
        False,
    )
    if expected.fusion_mode == "discount_probability":
        _expect_equal(
            "fusion.evidence_activation",
            str(fusion.get("evidence_activation", "")).strip().lower(),
            "softplus",
        )
        _expect_equal(
            "fusion.force_fp32_decision",
            _strict_bool(
                fusion.get("force_fp32_decision", False),
                field="fusion.force_fp32_decision",
            ),
            True,
        )
        _expect_equal(
            "fusion.min_discount",
            fusion.get("min_discount"),
            1.0e-6,
        )
        _expect_equal(
            "fusion.base_rate",
            fusion.get("base_rate", 0.5),
            0.5,
        )
    _expect_equal(
        "loss.objective",
        str(loss.get("objective", "standard")).strip().lower(),
        expected.objective,
    )
    _expect_equal(
        "loss.branch_aux_weight",
        loss.get("branch_aux_weight", 0.0),
        expected.branch_aux_weight,
    )
    _expect_equal(
        "loss.evidential_loss_weight",
        loss.get("evidential_loss_weight", 0.0),
        expected.evidential_loss_weight,
    )
    _expect_equal(
        "loss.auxiliary_weight_mode",
        str(loss.get("auxiliary_weight_mode", "")).strip().lower(),
        expected.auxiliary_weight_mode,
    )
    for index, branch in enumerate(BRANCHES):
        _expect_equal(
            f"loss.branch_aux_weights.{branch}",
            branch_weights.get(branch),
            expected.branch_aux_weights[index],
        )
    _expect_equal(
        "train.label_smoothing",
        train.get("label_smoothing", 0.0),
        0.0,
    )
    _expect_equal(
        "loss.label_smoothing",
        loss.get("label_smoothing", 0.0),
        0.0,
    )
    if expected.evidential_loss_weight > 0.0:
        evidential_cfg = _mapping(
            loss.get("evidential", {}),
            field="loss.evidential",
        )
        _expect_equal(
            "loss.evidential.anneal_epochs",
            strict_finite_integer(
                evidential_cfg.get("anneal_epochs"),
                field_name="loss.evidential.anneal_epochs",
            ),
            10,
        )
        _expect_equal(
            "loss.evidential.branches",
            tuple(evidential_cfg.get("branches", ())),
            BRANCHES,
        )
        _expect_equal(
            "loss.evidential.class_weight",
            evidential_cfg.get("class_weight"),
            "balanced",
        )
    _expect_equal(
        "model.gate.hidden_dim",
        strict_finite_integer(
            gate.get("hidden_dim", 128),
            field_name="model.gate.hidden_dim",
        ),
        expected.gate_hidden_dim,
    )
    _expect_equal(
        "model.gate.detach",
        _strict_bool(
            gate.get("detach", True),
            field="model.gate.detach",
        ),
        expected.gate_detach,
    )
    if expected.quality_temperature is not None:
        _expect_equal(
            "model.quality_fusion_temperature",
            model.get("quality_fusion_temperature"),
            expected.quality_temperature,
        )
    if expected.anneal_epochs is not None:
        objective_cfg = _mapping(
            loss.get(expected.objective, {}),
            field=f"loss.{expected.objective}",
        )
        _expect_equal(
            f"loss.{expected.objective}.anneal_epochs",
            strict_finite_integer(
                objective_cfg.get("anneal_epochs"),
                field_name=f"loss.{expected.objective}.anneal_epochs",
            ),
            expected.anneal_epochs,
        )
        _expect_equal(
            f"loss.{expected.objective}.mask_unavailable_views",
            _strict_bool(
                objective_cfg.get("mask_unavailable_views", False),
                field=(
                    f"loss.{expected.objective}.mask_unavailable_views"
                ),
            ),
            True,
        )
        if expected.ecml_consistency_weight is not None:
            _expect_equal(
                "loss.ecml.consistency_weight",
                objective_cfg.get("consistency_weight"),
                expected.ecml_consistency_weight,
            )

    _expect_equal(
        "calibration.enabled",
        _strict_bool(
            calibration.get("enabled", False),
            field="calibration.enabled",
        ),
        False,
    )
    _expect_equal(
        "calibration.require_role_assignment",
        _strict_bool(
            calibration.get("require_role_assignment", False),
            field="calibration.require_role_assignment",
        ),
        True,
    )
    _expect_equal(
        "calibration.expert_split_seed",
        strict_finite_integer(
            calibration.get("expert_split_seed"),
            field_name="calibration.expert_split_seed",
        ),
        BASELINE_EXPERT_SPLIT_SEED,
    )
    _expect_equal(
        "calibration.expert_val_fraction",
        calibration.get("expert_val_fraction"),
        BASELINE_EXPERT_VAL_FRACTION,
    )
    if not str(calibration.get("role_assignment_path") or "").strip():
        raise ValueError(
            "Registered baselines require calibration.role_assignment_path"
        )
    _expect_equal(
        "classification_threshold.enabled",
        _strict_bool(
            threshold.get("enabled", False),
            field="classification_threshold.enabled",
        ),
        False,
    )
    _expect_equal(
        "eval.run_test",
        _strict_bool(
            eval_cfg.get("run_test", True),
            field="eval.run_test",
        ),
        True,
    )
    _expect_equal(
        "eval.run_robust_test",
        _strict_bool(
            eval_cfg.get("run_robust_test", True),
            field="eval.run_robust_test",
        ),
        True,
    )
    _expect_equal(
        "eval.perturb_tests",
        tuple(str(value).strip().lower() for value in eval_cfg.get("perturb_tests", ())),
        FORMAL_PERTURBATIONS,
    )
    _expect_equal(
        "eval.perturb_strengths",
        tuple(float(value) for value in eval_cfg.get("perturb_strengths", ())),
        FORMAL_STRENGTHS,
    )
    _expect_equal(
        "eval.controlled_view_protocol_seed",
        strict_finite_integer(
            eval_cfg.get("controlled_view_protocol_seed"),
            field_name="eval.controlled_view_protocol_seed",
        ),
        CONTROLLED_TEST_VIEW_PROTOCOL_SEED,
    )
    if _strict_bool(
        (_mapping(
            model.get("graph_encoder", {}),
            field="model.graph_encoder",
        )).get("use_behavior_hint", False),
        field="model.graph_encoder.use_behavior_hint",
    ):
        raise ValueError(
            "Registered baselines require model.graph_encoder.use_behavior_hint=false"
        )
    if strict_finite_integer(
        model.get("num_classes", 2), field_name="model.num_classes"
    ) != 2:
        raise ValueError("Registered baselines are binary-only")
    _reject_any_enabled(cfg)
    return expected


def _build_model(
    cfg: Mapping[str, Any],
    *,
    feature_dim: int,
) -> TriModalRobustModel:
    model_cfg = _mapping(cfg.get("model", {}), field="model")
    fusion_cfg = _mapping(cfg.get("fusion", {}), field="fusion")
    api_cfg = _mapping(model_cfg.get("api_encoder", {}), field="model.api_encoder")
    graph_cfg = _mapping(
        model_cfg.get("graph_encoder", {}), field="model.graph_encoder"
    )
    manifest_cfg = _mapping(
        model_cfg.get("manifest_encoder", {}),
        field="model.manifest_encoder",
    )
    gate_cfg = _mapping(model_cfg.get("gate", {}), field="model.gate")
    graph_budget = strict_finite_integer(
        model_cfg.get("max_nodes_gnn"),
        field_name="model.max_nodes_gnn",
    )
    if graph_budget <= 0:
        raise ValueError("model.max_nodes_gnn must be positive")
    return TriModalRobustModel(
        in_feat_dim=int(feature_dim),
        num_classes=2,
        fusion_mode=str(model_cfg["fusion_mode"]),
        api_num_hash_buckets=int(api_cfg.get("num_hash_buckets", 8192)),
        api_type_vocab_size=int(api_cfg.get("type_vocab_size", 16)),
        api_emb_dim=int(api_cfg.get("emb_dim", 128)),
        api_hidden_dim=int(api_cfg.get("hidden_dim", 256)),
        api_dropout=float(api_cfg.get("dropout", 0.15)),
        api_encoder_type=str(api_cfg.get("type", "transformer")),
        api_layers=int(api_cfg.get("layers", 2)),
        api_heads=int(api_cfg.get("heads", 4)),
        api_max_seq_len=strict_finite_integer(
            api_cfg.get("max_seq_len", 2048),
            field_name="model.api_encoder.max_seq_len",
        ),
        graph_emb_dim=int(graph_cfg.get("emb_dim", 128)),
        graph_hidden=int(graph_cfg.get("hidden", 128)),
        graph_heads=int(graph_cfg.get("heads", 4)),
        graph_layers=int(graph_cfg.get("layers", 2)),
        graph_encoder_type=str(graph_cfg.get("type", "gatv2")),
        max_nodes_gnn=graph_budget,
        use_graph_behavior_hint=False,
        manifest_in_dim=int(manifest_cfg.get("in_dim", 256)),
        manifest_emb_dim=int(manifest_cfg.get("emb_dim", 128)),
        manifest_hidden_dim=int(manifest_cfg.get("hidden_dim", 256)),
        manifest_dropout=float(manifest_cfg.get("dropout", 0.1)),
        quality_fusion_temperature=float(
            model_cfg.get("quality_fusion_temperature", 10.0)
        ),
        gate_hidden_dim=int(gate_cfg.get("hidden_dim", 128)),
        gate_detach=bool(gate_cfg.get("detach", True)),
        discount_fusion_config=dict(fusion_cfg),
    )


def _binary_metrics(
    labels: list[int],
    probabilities: list[float],
    predictions: list[int],
) -> dict[str, Any]:
    if not labels:
        raise RuntimeError("Cannot report an empty evaluation split")
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.int64)
    if (
        p.shape != y.shape
        or pred.shape != y.shape
        or not np.isfinite(p).all()
        or not bool(((p >= 0.0) & (p <= 1.0)).all())
    ):
        raise ValueError("Evaluation predictions are invalid or misaligned")
    confidence = np.maximum(p, 1.0 - p)
    fixed_pred = (p >= 0.5).astype(np.int64)
    confidence_correct = (fixed_pred == y).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (
            (confidence >= lower) & (confidence <= upper)
            if upper >= 1.0
            else (confidence >= lower) & (confidence < upper)
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean())
                - float(confidence_correct[mask].mean())
            )
    metrics: dict[str, Any] = {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(
            f1_score(
                y,
                pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "f1_pos": float(
            f1_score(y, pred, average="binary", pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y,
                pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "recall_pos": float(
            recall_score(y, pred, pos_label=1, zero_division=0)
        ),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_10": float(ece),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_confidence": float(confidence.mean()),
        "confidence_accuracy_gap": float(
            confidence.mean() - confidence_correct.mean()
        ),
    }
    if len(set(labels)) > 1:
        metrics.update(
            {
                "auc_defined": 1,
                "auc": float(roc_auc_score(y, p)),
                "ap_defined": 1,
                "ap": float(average_precision_score(y, p)),
            }
        )
    else:
        metrics.update(
            {
                "auc_defined": 0,
                "auc": None,
                "ap_defined": 0,
                "ap": None,
            }
        )
    return metrics


def _availability_column(
    graph: Any,
    branch: str,
    *,
    batch_size: int,
) -> list[bool]:
    value = getattr(graph, f"{branch}_alive", None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Evaluation batch is missing mandatory {branch}_alive"
        )
    flat = value.detach().view(batch_size, -1)[:, 0]
    valid = torch.isfinite(flat.float()) & ((flat == 0) | (flat == 1))
    if not bool(valid.all().cpu().item()):
        raise ValueError(f"{branch}_alive must be hard binary")
    return [bool(item) for item in flat.bool().cpu().tolist()]


@torch.inference_mode()
def _evaluate(
    model: TriModalRobustModel,
    loader: Any,
    device: torch.device,
    *,
    use_amp: bool,
    split_name: str,
    dump_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    predictions_all: list[int] = []
    rows: list[dict[str, Any]] = []
    failed_total = 0
    branch_labels: dict[str, list[int]] = {name: [] for name in BRANCHES}
    branch_probs: dict[str, list[float]] = {name: [] for name in BRANCHES}

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, sids, failed = prepare_robust_batch(batch, device)
        failed_total += int(failed)
        if graph is None:
            continue
        if not isinstance(sids, list) or len(sids) != int(labels.numel()):
            raise RuntimeError(
                f"{split_name}: SID count must match the valid batch size; "
                f"got {0 if sids is None else len(sids)} SIDs for "
                f"{int(labels.numel())} samples"
            )
        output_digests = [
            str(value)
            for value in batch.get(
                "care_view_output_digests",
                [""] * int(labels.numel()),
            )
        ]
        if len(output_digests) != int(labels.numel()):
            raise RuntimeError(
                f"{split_name}: controlled-view output digest count "
                "must match the valid batch size"
            )
        with get_amp_context(device, use_amp):
            logits, extra = model(graph)
        if logits.ndim != 2 or logits.shape != (labels.numel(), 2):
            raise RuntimeError(
                f"{split_name}: model logits must be [B,2], got {tuple(logits.shape)}"
            )
        probability = torch.softmax(logits.float(), dim=-1)[:, 1]
        if not bool(torch.isfinite(probability).all().cpu().item()):
            raise FloatingPointError(f"{split_name}: non-finite model probability")
        prediction = probability.ge(0.5).long()
        labels_cpu = [int(value) for value in labels.detach().long().cpu().tolist()]
        probabilities_cpu = [
            float(value) for value in probability.detach().cpu().tolist()
        ]
        predictions_cpu = [
            int(value) for value in prediction.detach().cpu().tolist()
        ]
        labels_all.extend(labels_cpu)
        probabilities_all.extend(probabilities_cpu)
        predictions_all.extend(predictions_cpu)

        alive_by_branch = {
            branch: _availability_column(
                graph, branch, batch_size=int(labels.numel())
            )
            for branch in BRANCHES
        }
        branch_prob_batch: dict[str, list[float]] = {}
        for branch, key in BRANCH_LOGIT_KEYS.items():
            branch_logits = extra.get(key)
            if not isinstance(branch_logits, torch.Tensor):
                raise RuntimeError(
                    f"{split_name}: registered baseline omitted {key}"
                )
            if branch_logits.shape != logits.shape:
                raise RuntimeError(
                    f"{split_name}: {key} shape disagrees with final logits"
                )
            values = [
                float(value)
                for value in torch.softmax(
                    branch_logits.float(), dim=-1
                )[:, 1].detach().cpu().tolist()
            ]
            branch_prob_batch[branch] = values
            for label, value, alive in zip(
                labels_cpu, values, alive_by_branch[branch]
            ):
                if alive:
                    branch_labels[branch].append(label)
                    branch_probs[branch].append(value)

        if dump_rows:
            gate = extra.get("gate_weights")
            if not isinstance(gate, torch.Tensor) or gate.shape != (
                labels.numel(),
                3,
            ):
                raise RuntimeError(
                    f"{split_name}: gate_weights must be [B,3]"
                )
            gate_cpu = gate.detach().float().cpu()
            for index, sid in enumerate(sids):
                row: dict[str, Any] = {
                    "split": split_name,
                    "sid": str(sid),
                    "label": labels_cpu[index],
                    "prob_malware": probabilities_cpu[index],
                    "pred": predictions_cpu[index],
                    "correct": int(
                        predictions_cpu[index] == labels_cpu[index]
                    ),
                    "classification_protocol": (
                        BASELINE_CLASSIFICATION_PROTOCOL_ID
                    ),
                    "classification_threshold": 0.5,
                    "controlled_view_output_digest": (
                        output_digests[index]
                    ),
                    "w_api": float(gate_cpu[index, 0].item()),
                    "w_graph": float(gate_cpu[index, 1].item()),
                    "w_manifest": float(gate_cpu[index, 2].item()),
                }
                for branch in BRANCHES:
                    branch_prob = branch_prob_batch[branch][index]
                    branch_pred = int(branch_prob >= 0.5)
                    row[f"{branch}_alive"] = int(
                        alive_by_branch[branch][index]
                    )
                    row[f"{branch}_prob"] = branch_prob
                    row[f"{branch}_pred"] = branch_pred
                    row[f"{branch}_correct"] = int(
                        branch_pred == labels_cpu[index]
                    )
                for key in DIAGNOSTIC_KEYS:
                    value = extra.get(key)
                    if (
                        isinstance(value, torch.Tensor)
                        and value.numel() > index
                    ):
                        row[key] = float(
                            value.detach().float().view(-1)[index].cpu().item()
                        )
                rows.append(row)

    metrics = _binary_metrics(
        labels_all, probabilities_all, predictions_all
    )
    metrics.update(
        {
            "classification_protocol": (
                BASELINE_CLASSIFICATION_PROTOCOL_ID
            ),
            "classification_threshold": 0.5,
            "num_eval": int(len(labels_all)),
            "num_failed": int(failed_total),
            "num_requested": int(len(labels_all) + failed_total),
        }
    )
    branch_metrics: dict[str, Any] = {}
    for branch in BRANCHES:
        if not branch_labels[branch]:
            continue
        branch_prediction = [
            int(value >= 0.5) for value in branch_probs[branch]
        ]
        current = _binary_metrics(
            branch_labels[branch],
            branch_probs[branch],
            branch_prediction,
        )
        current["num_eval"] = len(branch_labels[branch])
        current["eligible_fraction"] = (
            len(branch_labels[branch]) / max(len(labels_all), 1)
        )
        branch_metrics[branch] = current
    metrics["branch_metrics"] = branch_metrics
    return metrics, rows


def _train_one_epoch(
    model: TriModalRobustModel,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> float:
    model.train()
    train_cfg = _mapping(cfg.get("train", {}), field="train")
    use_amp = bool(train_cfg.get("use_amp", True))
    grad_accum = strict_finite_integer(
        train_cfg.get("grad_accum_steps", 1),
        field_name="train.grad_accum_steps",
    )
    if grad_accum <= 0:
        raise ValueError("train.grad_accum_steps must be positive")
    num_batches = len(loader)
    if num_batches <= 0:
        raise RuntimeError("Training loader is empty")
    loss_cfg = dict(_mapping(cfg.get("loss", {}), field="loss"))
    loss_cfg["label_smoothing"] = 0.0
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    valid_samples = 0
    failed_samples = 0
    part_sums: dict[str, float] = {}
    part_counts: dict[str, int] = {}

    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"train {epoch}", leave=False)
    ):
        graph, labels, _sids, failed = prepare_robust_batch(batch, device)
        failed_samples += int(failed)
        if graph is None:
            continue
        batch_size = int(labels.numel())
        group_start = (batch_index // grad_accum) * grad_accum
        accumulation_divisor = min(
            grad_accum, num_batches - group_start
        )
        with get_amp_context(device, use_amp):
            logits, extra = model(graph)
            loss, parts = compute_robust_loss(
                logits,
                labels,
                extra,
                loss_cfg,
                availability=extra.get("fusion_availability"),
                epoch=epoch,
                evidence_activation=str(
                    (_mapping(cfg.get("fusion", {}), field="fusion")).get(
                        "evidence_activation", "softplus"
                    )
                ),
                materialize_diagnostics=False,
            )
        if not bool(torch.isfinite(loss.detach()).all().cpu().item()):
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch}, "
                f"batch={batch_index}"
            )
        scaler.scale(loss / float(accumulation_divisor)).backward()
        is_group_end = (
            (batch_index + 1) % grad_accum == 0
            or batch_index + 1 == num_batches
        )
        if is_group_end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg.get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scalar_loss = float(loss.detach().cpu().item())
        weighted_loss += scalar_loss * batch_size
        valid_samples += batch_size
        for key, value in parts.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                numeric = float(value.detach().cpu().item())
            elif isinstance(value, (float, int)):
                numeric = float(value)
            else:
                continue
            if math.isfinite(numeric):
                part_sums[key] = part_sums.get(key, 0.0) + numeric
                part_counts[key] = part_counts.get(key, 0) + 1

    enforce_failed_ratio(
        {"num_eval": valid_samples, "num_failed": failed_samples},
        dict(cfg),
        f"train_epoch_{epoch}",
    )
    if valid_samples <= 0:
        raise RuntimeError("No valid training samples were processed")
    logger.info(
        "train_loss_parts epoch=%s %s",
        epoch,
        " ".join(
            f"{key}={part_sums[key] / max(part_counts[key], 1):.4f}"
            for key in sorted(part_sums)
            if key
            in {
                "loss",
                "ce",
                "branch_aux",
                "branch_aux_weight",
                "evidential_loss",
                "evidential_loss_weight",
                "dirichlet_fused_loss",
                "dirichlet_view_loss",
                "ecml_conflict_consistency_loss",
            }
        ),
    )
    return weighted_loss / float(valid_samples)


def _baseline_output_artifact_names() -> tuple[set[str], set[str]]:
    collision_names = {
        "resolved_config.yaml",
        "summary.yaml",
        BASELINE_CHECKPOINT_NAME,
    }
    generated_names = collision_names | {"gate_diagnostics.csv"}
    return collision_names, generated_names


def _preflight_output_collision(path: Path, *, overwrite: bool) -> None:
    collision_names, _ = _baseline_output_artifact_names()
    collisions = sorted(
        child.name
        for child in (path.iterdir() if path.is_dir() else ())
        if child.name in collision_names
    )
    if collisions and not overwrite:
        raise FileExistsError(
            f"Output directory {path} already contains run artifacts "
            f"{collisions}. Choose a new train.exp_name/seed or pass "
            "--overwrite explicitly."
        )


def _prepare_output_directory(
    path: Path,
    *,
    overwrite: bool,
) -> Path:
    _preflight_output_collision(path, overwrite=overwrite)
    _, generated_names = _baseline_output_artifact_names()
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in sorted(generated_names):
            artifact = path / name
            if not artifact.exists():
                continue
            if not artifact.is_file() and not artifact.is_symlink():
                raise ValueError(
                    f"Refusing to overwrite non-file run artifact: {artifact}"
                )
            artifact.unlink()
    return path


def _write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            _json_compatible(dict(value)),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Diagnostic row dump must not be empty")
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _implementation_sha256() -> str:
    source_dir = Path(__file__).resolve().parent
    files = (
        "baseline_train.py",
        "runtime.py",
        "constants.py",
        "dataset.py",
        "evidence.py",
        "evidential.py",
        "gates.py",
        "graph_encoders.py",
        "model.py",
        "discount_fusion.py",
        "losses.py",
        "perturbations.py",
        "pt_schema.py",
        "quality.py",
        "semantic_categories.py",
        "utils.py",
    )
    digest = hashlib.sha256()
    for name in files:
        path = source_dir / name
        source = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_compatible(dict(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_sequence(values: list[str]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_role_record(
    dataset: Any,
    indices: list[int],
    *,
    usage: str,
) -> dict[str, Any]:
    sids = [str(dataset.sample_sids[index]) for index in indices]
    groups = [str(dataset.sample_groups[index]) for index in indices]
    labels = [int(dataset.sample_labels[index]) for index in indices]
    years = [int(dataset.sample_years[index]) for index in indices]
    label_values = sorted(set(labels))
    year_label_values = sorted(set(zip(years, labels)))
    semantic_rows = sorted(
        (
            sids[offset],
            groups[offset],
            labels[offset],
            years[offset],
        )
        for offset in range(len(indices))
    )
    return {
        "source_split": "train",
        "role_usage": usage,
        "num_samples": len(indices),
        "num_groups": len(set(groups)),
        "sample_ids_sha256": _sha256_sequence(sorted(sids)),
        "groups_sha256": _sha256_sequence(sorted(set(groups))),
        "semantic_rows_sha256": _sha256_mapping(
            {"rows": semantic_rows}
        ),
        "label_counts": {
            str(label): labels.count(label) for label in label_values
        },
        "year_label_counts": {
            f"{year}:{label}": sum(
                int(
                    current_year == year
                    and current_label == label
                )
                for current_year, current_label in zip(years, labels)
            )
            for year, label in year_label_values
        },
    }


def _split_baseline_training_roles(
    cfg: Mapping[str, Any],
    train_dataset: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Create the same immutable 90/10 Stage-A roles used by CARE."""

    calibration = _mapping(
        cfg.get("calibration", {}), field="calibration"
    )
    split_seed = strict_finite_integer(
        calibration.get("expert_split_seed"),
        field_name="calibration.expert_split_seed",
    )
    fraction = float(calibration.get("expert_val_fraction"))
    split_cfg = copy.deepcopy(dict(cfg))
    split_cfg["calibration"] = {
        "validation_fraction": fraction,
        "split_seed": split_seed,
        "stratified_group_split": True,
    }
    expert_train, expert_val, raw_summary = split_validation_dataset(
        split_cfg, train_dataset
    )
    train_indices = [int(value) for value in expert_train.indices]
    val_indices = [int(value) for value in expert_val.indices]
    size = len(train_dataset)
    if (
        set(train_indices) & set(val_indices)
        or sorted(train_indices + val_indices) != list(range(size))
    ):
        raise RuntimeError(
            "Baseline expert_train/expert_val must be a disjoint full "
            "partition of train"
        )
    sids = list(getattr(train_dataset, "sample_sids", ()))
    groups = list(getattr(train_dataset, "sample_groups", ()))
    labels = list(getattr(train_dataset, "sample_labels", ()))
    years = list(getattr(train_dataset, "sample_years", ()))
    if not (
        len(sids) == len(groups) == len(labels) == len(years) == size
    ):
        raise ValueError(
            "Baseline expert splitting requires complete "
            "SID/group/label/year metadata"
        )
    train_sids = {str(sids[index]) for index in train_indices}
    val_sids = {str(sids[index]) for index in val_indices}
    train_groups = {str(groups[index]) for index in train_indices}
    val_groups = {str(groups[index]) for index in val_indices}
    if train_sids & val_sids:
        raise RuntimeError("Baseline expert roles share sample identities")
    if train_groups & val_groups:
        raise RuntimeError("Baseline expert roles split package groups")
    val_index_set = set(val_indices)
    assignment_rows = sorted(
        (
            str(sids[index]),
            "expert_val" if index in val_index_set else "expert_train",
        )
        for index in range(size)
    )
    public_summary = {
        "protocol_id": BASELINE_TRAINING_ROLE_PROTOCOL_ID,
        "expert_split_seed": split_seed,
        "expert_val_fraction": fraction,
        "identity_disjoint": True,
        "group_disjoint": True,
        "assignment_semantic_sha256": _sha256_mapping(
            {"assignments": assignment_rows}
        ),
        "roles": {
            "expert_train": _training_role_record(
                train_dataset,
                train_indices,
                usage="fit_model_parameters_only",
            ),
            "expert_val": _training_role_record(
                train_dataset,
                val_indices,
                usage=(
                    "checkpoint_selection_clean_macro_f1_at_fixed_0_5_only"
                ),
            ),
        },
        "split_diagnostics": {
            key: value
            for key, value in raw_summary.items()
            if key not in {"selection_indices", "calibration_indices"}
        },
    }
    return expert_train, expert_val, public_summary


def _baseline_validation_audit_summary(
    role_summary: Mapping[str, Any],
) -> dict[str, Any]:
    public = {
        str(key): copy.deepcopy(value)
        for key, value in role_summary.items()
        if key
        not in {
            "selection_indices",
            "calibration_indices",
            "decision_calibration_indices",
        }
    }
    public["roles"] = {
        "model_selection": {
            "source_split": "val",
            "role_usage": (
                "audit_only_no_fitting_no_checkpoint_selection"
            ),
            "num_samples": int(role_summary["num_selection"]),
            "num_groups": int(role_summary["num_selection_groups"]),
        },
        "decision_calibration": {
            "source_split": "val",
            "role_usage": (
                "audit_only_no_fitting_no_checkpoint_selection"
            ),
            "num_samples": int(role_summary["num_calibration"]),
            "num_groups": int(role_summary["num_calibration_groups"]),
        },
    }
    public["labels_consumed_for_fitting"] = False
    public["labels_consumed_for_checkpoint_selection"] = False
    return public


def _controlled_output_manifest_sha256(
    rows: list[dict[str, Any]],
) -> str:
    payload = [
        {
            "sid": str(row["sid"]).strip().lower(),
            "output_digest": str(
                row.get("controlled_view_output_digest", "")
            ),
        }
        for row in rows
    ]
    if not payload or any(not row["output_digest"] for row in payload):
        raise RuntimeError(
            "Controlled test view omitted one or more input digests"
        )
    return canonical_manifest_sha256(payload)


def _run_identity(
    cfg: Mapping[str, Any],
    method: BaselineIdentity,
    *,
    experiment_name: str,
    seed: int,
) -> dict[str, Any]:
    protocol_payload = {
        "runner_protocol_id": BASELINE_RUNNER_PROTOCOL_ID,
        "method": asdict(method),
        "model": cfg.get("model", {}),
        "fusion": cfg.get("fusion", {}),
        "loss": cfg.get("loss", {}),
        "training_roles": {
            "protocol_id": BASELINE_TRAINING_ROLE_PROTOCOL_ID,
            "expert_split_seed": BASELINE_EXPERT_SPLIT_SEED,
            "expert_val_fraction": BASELINE_EXPERT_VAL_FRACTION,
        },
        "classification_rule": {
            "protocol_id": BASELINE_CLASSIFICATION_PROTOCOL_ID,
            "threshold": 0.5,
            "fitted": False,
        },
        "eval": {
            "perturb_tests": list(FORMAL_PERTURBATIONS),
            "perturb_strengths": list(FORMAL_STRENGTHS),
        },
    }
    return {
        "experiment_name": str(experiment_name),
        "method_name": method.method_name,
        "method_protocol_id": method.protocol_id,
        "runner_protocol_id": BASELINE_RUNNER_PROTOCOL_ID,
        "seed": int(seed),
        "resolved_config_sha256": _sha256_mapping(cfg),
        "method_protocol_sha256": _sha256_mapping(protocol_payload),
        "method_implementation_sha256": _implementation_sha256(),
        "model_fusion_mode": method.fusion_mode,
        "combination_rule": method.combination or None,
        "training_objective": method.objective,
        "robust_eval_expected_result_count": FORMAL_ROBUST_CELL_COUNT,
        "classification_protocol_id": BASELINE_CLASSIFICATION_PROTOCOL_ID,
        "classification_threshold": 0.5,
        "classification_threshold_fitted": False,
        "selective_prediction_enabled": False,
    }


def _apply_manifest_provenance(
    cfg: dict[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if not bool(provenance.get("verified", False)):
        raise ValueError(
            "Formal registered baselines require verified train-only "
            "Manifest-vocabulary provenance"
        )
    data = cfg.setdefault("data", {})
    data["expected_manifest_vocab_sha256"] = provenance[
        "manifest_vocab_sha256"
    ]
    data["expected_manifest_train_csv_sha256"] = provenance[
        "train_csv_sha256"
    ]
    data["expected_manifest_train_sample_ids_sha256"] = provenance[
        "train_sample_ids_sha256"
    ]


def _validate_checkpoint(
    payload: Any,
    *,
    method: BaselineIdentity,
    config_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Baseline checkpoint must be a mapping")
    _expect_equal(
        "checkpoint.schema_version",
        payload.get("baseline_artifact_schema_version"),
        BASELINE_ARTIFACT_SCHEMA_VERSION,
    )
    _expect_equal(
        "checkpoint.method_protocol_id",
        payload.get("method_protocol_id"),
        method.protocol_id,
    )
    _expect_equal(
        "checkpoint.resolved_config_sha256",
        payload.get("resolved_config_sha256"),
        config_sha256,
    )
    classification_rule = _mapping(
        payload.get("classification_rule", {}),
        field="checkpoint.classification_rule",
    )
    _expect_equal(
        "checkpoint.classification_rule.protocol_id",
        classification_rule.get("protocol_id"),
        BASELINE_CLASSIFICATION_PROTOCOL_ID,
    )
    _expect_equal(
        "checkpoint.classification_rule.threshold",
        classification_rule.get("threshold"),
        0.5,
    )
    _expect_equal(
        "checkpoint.classification_rule.fitted",
        bool(classification_rule.get("fitted", True)),
        False,
    )
    training_roles = _mapping(
        payload.get("training_roles", {}),
        field="checkpoint.training_roles",
    )
    _expect_equal(
        "checkpoint.training_roles.protocol_id",
        training_roles.get("protocol_id"),
        BASELINE_TRAINING_ROLE_PROTOCOL_ID,
    )
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Baseline checkpoint is missing model_state")
    actual_state_sha = _state_dict_sha256(dict(state))
    _expect_equal(
        "checkpoint.model_state_sha256",
        payload.get("model_state_sha256"),
        actual_state_sha,
    )
    return payload


def run(cfg: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    method = validate_registered_baseline_config(cfg)
    logging.basicConfig(
        level=getattr(
            logging,
            str(cfg.get("log_level", "INFO")).upper(),
            logging.INFO,
        )
    )
    train_cfg = dict(_mapping(cfg.get("train", {}), field="train"))
    data_cfg = dict(_mapping(cfg.get("data", {}), field="data"))
    cfg["train"] = train_cfg
    cfg["data"] = data_cfg
    seed = strict_finite_integer(
        train_cfg.get("seed", 42), field_name="train.seed"
    )
    experiment_name = str(train_cfg.get("exp_name") or method.method_name)
    out_dir = (
        Path(data_cfg.get("out_dir", "results"))
        / experiment_name
        / str(seed)
    )
    # A non-overwrite collision is knowable without touching PT files. Fail
    # immediately instead of spending minutes on data provenance preflights.
    _preflight_output_collision(out_dir, overwrite=overwrite)
    epochs = strict_finite_integer(
        train_cfg.get("epochs"), field_name="train.epochs"
    )
    patience = strict_finite_integer(
        train_cfg.get("patience"), field_name="train.patience"
    )
    if epochs <= 0 or patience <= 0:
        raise ValueError("train.epochs and train.patience must be positive")
    set_seed(seed)
    configure_determinism(
        bool(train_cfg.get("deterministic", True)),
        strict=bool(train_cfg.get("strict_deterministic", False)),
    )
    configure_multiprocessing_sharing(cfg)
    device = select_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("use_amp", True))

    manifest_provenance = validate_manifest_vocab_provenance(cfg)
    _apply_manifest_provenance(cfg, manifest_provenance)
    validate_split_partitions(cfg, include_test=True)

    # Complete every scientific preflight before --overwrite may mutate an
    # existing result directory.
    train_dataset = build_dataset(cfg, "train", is_train=True)
    val_dataset = build_dataset(cfg, "val", is_train=False)
    test_dataset = build_dataset(cfg, "test", is_train=False)
    # Hash the exact clean model inputs as well as every controlled view so
    # CARE and baselines can prove that paired test cells are identical.
    test_dataset.care_digest_view = True
    feature_dims = {
        int(train_dataset.feature_dim),
        int(val_dataset.feature_dim),
        int(test_dataset.feature_dim),
    }
    if len(feature_dims) != 1:
        raise ValueError(
            f"Train/validation/test graph feature dimensions differ: {feature_dims}"
        )
    (
        expert_train_dataset,
        expert_val_dataset,
        training_role_summary,
    ) = _split_baseline_training_roles(cfg, train_dataset)
    roles = _load_fixed_validation_roles(cfg, val_dataset)
    if roles is None:
        raise ValueError(
            "Registered baselines require the fixed validation role assignment"
        )
    model_selection_dataset, decision_dataset, role_summary = roles
    if set(model_selection_dataset.indices) & set(decision_dataset.indices):
        raise RuntimeError("Validation model-selection and decision roles overlap")
    validation_audit_summary = _baseline_validation_audit_summary(
        role_summary
    )

    model = _build_model(
        cfg, feature_dim=next(iter(feature_dims))
    ).to(device)
    # Architecture-specific initialization consumes a different number of
    # random draws. Reset before Stage A so dropout and loader randomness use
    # the same seed protocol as CARE.
    set_seed(seed)
    for parameter in model.encoder_training_frozen_parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Registered baseline has no trainable parameters")

    out_dir = _prepare_output_directory(out_dir, overwrite=overwrite)
    _write_yaml_atomic(out_dir / "resolved_config.yaml", cfg)

    train_loader = build_loader(
        cfg,
        expert_train_dataset,
        is_train=True,
        seed_namespace=STAGE_A_EXPERT_TRAIN_LOADER_NAMESPACE,
    )
    expert_val_loader = build_loader(
        cfg,
        expert_val_dataset,
        is_train=False,
        seed_namespace=STAGE_A_EXPERT_VAL_LOADER_NAMESPACE,
    )
    model_selection_audit_loader = build_loader(
        cfg,
        model_selection_dataset,
        is_train=False,
        seed_namespace=(
            f"baseline/{method.protocol_id}/model_selection_audit"
        ),
    )
    decision_loader = build_loader(
        cfg,
        decision_dataset,
        is_train=False,
        seed_namespace=f"baseline/{method.protocol_id}/decision_audit",
    )
    test_loader = build_loader(
        cfg,
        test_dataset,
        is_train=False,
        seed_namespace=f"baseline/{method.protocol_id}/test",
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("eta_min", 1.0e-6)),
    )
    scaler = build_grad_scaler(device, use_amp)
    run_identity = _run_identity(
        cfg,
        method,
        experiment_name=experiment_name,
        seed=seed,
    )
    checkpoint_path = out_dir / BASELINE_CHECKPOINT_NAME
    best_score = float("-inf")
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    stale = 0
    epochs_ran = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch=epoch,
        )
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        expert_val_metrics, _ = _evaluate(
            model,
            expert_val_loader,
            device,
            use_amp=use_amp,
            split_name="expert_val_checkpoint_selection",
            dump_rows=False,
        )
        enforce_failed_ratio(
            expert_val_metrics, cfg, "expert_val_checkpoint_selection"
        )
        validation_seconds = time.perf_counter() - validation_started
        score = float(expert_val_metrics["macro_f1"])
        if not math.isfinite(score):
            raise FloatingPointError(
                f"Non-finite checkpoint score at epoch={epoch}"
            )
        scheduler.step()
        epochs_ran = epoch
        logger.info(
            "epoch=%s train_loss=%.4f expert_val_macro_f1=%.4f "
            "expert_val_auc=%s expert_val_acc=%.4f "
            "checkpoint_score=%.4f train_wall_seconds=%.2f "
            "expert_val_wall_seconds=%.2f",
            epoch,
            train_loss,
            score,
            expert_val_metrics.get("auc"),
            expert_val_metrics["acc"],
            score,
            train_seconds,
            validation_seconds,
        )
        # Match CARE exactly: strict improvement; exact ties keep the earliest
        # checkpoint.
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = copy.deepcopy(expert_val_metrics)
            stale = 0
            state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            state_sha = _state_dict_sha256(state)
            _atomic_torch_save(
                {
                    "baseline_artifact_schema_version": (
                        BASELINE_ARTIFACT_SCHEMA_VERSION
                    ),
                    "runner_protocol_id": BASELINE_RUNNER_PROTOCOL_ID,
                    "method_protocol_id": method.protocol_id,
                    "method_name": method.method_name,
                    "resolved_config_sha256": run_identity[
                        "resolved_config_sha256"
                    ],
                    "method_protocol_sha256": run_identity[
                        "method_protocol_sha256"
                    ],
                    "method_implementation_sha256": run_identity[
                        "method_implementation_sha256"
                    ],
                    "model_state": state,
                    "model_state_sha256": state_sha,
                    "epoch": epoch,
                    "checkpoint_metric": (
                        "expert_val_clean_macro_f1_at_fixed_0_5"
                    ),
                    "exact_tie_policy": (
                        "strict_improvement_keep_earliest_epoch"
                    ),
                    "checkpoint_score": score,
                    "expert_val_checkpoint_selection": (
                        expert_val_metrics
                    ),
                    "training_roles": training_role_summary,
                    "validation_audit_roles": (
                        validation_audit_summary
                    ),
                    "manifest_vocab_provenance": manifest_provenance,
                    "classification_rule": {
                        "protocol_id": (
                            BASELINE_CLASSIFICATION_PROTOCOL_ID
                        ),
                        "threshold": 0.5,
                        "fitted": False,
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info(
                    "early_stop epoch=%s best_epoch=%s best_score=%.6f",
                    epoch,
                    best_epoch,
                    best_score,
                )
                break

    if best_epoch is None or best_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("Training finished without a baseline checkpoint")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    checkpoint = _validate_checkpoint(
        checkpoint,
        method=method,
        config_sha256=run_identity["resolved_config_sha256"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)

    expert_val_selected_metrics, expert_val_rows = _evaluate(
        model,
        expert_val_loader,
        device,
        use_amp=use_amp,
        split_name="expert_val_checkpoint_selection",
        dump_rows=True,
    )
    enforce_failed_ratio(
        expert_val_selected_metrics,
        cfg,
        "expert_val_checkpoint_selection",
    )
    expert_val_selected_metrics["role_usage"] = (
        "checkpoint_selection_clean_macro_f1_at_fixed_0_5_only"
    )
    if not math.isclose(
        float(expert_val_selected_metrics["macro_f1"]),
        float(best_score),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Reloaded baseline checkpoint does not reproduce its "
            "expert_val selection score"
        )

    model_selection_metrics, model_selection_rows = _evaluate(
        model,
        model_selection_audit_loader,
        device,
        use_amp=use_amp,
        split_name="val_model_selection_audit_only",
        dump_rows=True,
    )
    enforce_failed_ratio(
        model_selection_metrics, cfg, "val_model_selection_audit_only"
    )
    model_selection_metrics["role_usage"] = (
        "audit_only_no_fitting_no_checkpoint_selection"
    )

    decision_metrics, decision_rows = _evaluate(
        model,
        decision_loader,
        device,
        use_amp=use_amp,
        split_name="val_decision_calibration_audit_only",
        dump_rows=True,
    )
    enforce_failed_ratio(
        decision_metrics, cfg, "val_decision_calibration_audit_only"
    )
    decision_metrics["role_usage"] = (
        "audit_only_no_fitting_no_model_selection"
    )

    test_metrics, test_rows = _evaluate(
        model,
        test_loader,
        device,
        use_amp=use_amp,
        split_name="test_clean",
        dump_rows=True,
    )
    enforce_failed_ratio(test_metrics, cfg, "test_clean")
    clean_output_manifest_sha256 = _controlled_output_manifest_sha256(
        test_rows
    )
    robust_results: dict[str, Any] = {}
    robust_rows: list[dict[str, Any]] = []
    for item in iter_robust_test_loaders(cfg, test_dataset):
        key = str(item["result_key"])
        if key in robust_results:
            raise RuntimeError(f"Duplicate robust evaluation key: {key}")
        if str(item["perturb_type"]) == "clean":
            clean_metrics = copy.deepcopy(test_metrics)
            clean_audit = copy.deepcopy(item["controlled_view_audit"])
            clean_audit["output_manifest_sha256"] = (
                clean_output_manifest_sha256
            )
            clean_metrics["controlled_view_audit"] = clean_audit
            robust_results[key] = clean_metrics
            continue
        loader = item.get("loader")
        if loader is None:
            raise RuntimeError(f"Robust evaluation {key} has no loader")
        metrics, rows = _evaluate(
            model,
            loader,
            device,
            use_amp=use_amp,
            split_name=f"test_{key}",
            dump_rows=True,
        )
        enforce_failed_ratio(metrics, cfg, f"test_{key}")
        metrics["perturb_type"] = str(item["perturb_type"])
        metrics["perturb_strength"] = float(item["strength"])
        controlled_view_audit = copy.deepcopy(
            item["controlled_view_audit"]
        )
        controlled_view_audit["output_manifest_sha256"] = (
            _controlled_output_manifest_sha256(rows)
        )
        metrics["controlled_view_audit"] = controlled_view_audit
        robust_results[key] = metrics
        robust_rows.extend(rows)
    if len(robust_results) != FORMAL_ROBUST_CELL_COUNT:
        raise RuntimeError(
            "Registered baseline robust matrix must contain exactly "
            f"{FORMAL_ROBUST_CELL_COUNT} cells, got {len(robust_results)}"
        )
    if "clean" not in robust_results:
        raise RuntimeError("Registered robust matrix omitted clean")

    all_rows = (
        expert_val_rows
        + model_selection_rows
        + decision_rows
        + test_rows
        + robust_rows
    )
    diagnostic_path = out_dir / "gate_diagnostics.csv"
    _write_rows(diagnostic_path, all_rows)

    classification_rule = {
        "protocol_id": BASELINE_CLASSIFICATION_PROTOCOL_ID,
        "prediction": "argmax_binary_softmax",
        "threshold": 0.5,
        "fitted": False,
        "label_consumption": "none",
    }
    data_roles = {
        "training": training_role_summary,
        "validation_audit": validation_audit_summary,
        "test": {
            "source_split": "test",
            "role_usage": "final_reporting_only",
            "num_samples": len(test_dataset),
        },
    }
    summary: dict[str, Any] = {
        "metric_schema_version": BASELINE_METRIC_SCHEMA_VERSION,
        "runner_protocol_id": BASELINE_RUNNER_PROTOCOL_ID,
        "run_identity": run_identity,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "best_checkpoint_score": float(best_score),
        "best_expert_val_macro_f1": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "checkpoint_metric": (
            "expert_val_clean_macro_f1_at_fixed_0_5"
        ),
        "exact_tie_policy": "strict_improvement_keep_earliest_epoch",
        "checkpoint_selection_role": "expert_val",
        "classification_rule": classification_rule,
        "controlled_test_view_protocol": {
            "seed_formula": CONTROLLED_VIEW_SEED_FORMULA,
            "protocol_seed": CONTROLLED_TEST_VIEW_PROTOCOL_SEED,
            "mechanism_version": CONTROLLED_VIEW_MECHANISM_VERSION,
        },
        "data_roles": data_roles,
        "training_split": training_role_summary,
        "validation_split": validation_audit_summary,
        "manifest_vocab_provenance": manifest_provenance,
        "expert_val_checkpoint_selection": (
            expert_val_selected_metrics
        ),
        "val_model_selection": model_selection_metrics,
        "val_decision_calibration": decision_metrics,
        "test": test_metrics,
        "robust": robust_results,
        "extra_eval": {},
        "diagnostic_artifacts": {
            "gate_diagnostics": {
                "path": str(diagnostic_path.resolve()),
                "sha256": _file_sha256(diagnostic_path),
                "num_rows": len(all_rows),
            }
        },
    }
    _write_yaml_atomic(out_dir / "summary.yaml", summary)
    logger.info("finished: %s", out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate one of the 14 registered comparison baselines."
        )
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML configs, applied left to right.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace artifacts in the selected output directory.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
