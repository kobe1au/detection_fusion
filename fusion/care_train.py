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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from fusion.care_fusion import (
    PATH_NAMES,
    hard_predict,
    path_availability,
    route_with_agm_anchor,
)
from fusion.care_training import (
    CareRiskCalibrationCache,
    deterministic_view_spec,
    fit_atomic_crc_correctness_threshold,
    fit_care_risk_crossfit,
    tensor_state_dict_sha256,
)
from fusion.care_losses import compute_care_stage_a_loss
from fusion.care_model import CareDroidModel
from fusion.dataset import (
    RobustTriModalDataset,
    prepare_robust_batch,
)
from fusion.perturbations import (
    GRADED_PERTURBATIONS,
    MISSING_PERTURBATIONS,
)
from fusion.runtime import (
    _atomic_torch_save,
    _file_sha256,
    _load_fixed_validation_roles,
    _sample_ids_sha256,
    _state_dict_sha256,
    build_dataset,
    build_loader,
    configure_determinism,
    configure_multiprocessing_sharing,
    enforce_failed_ratio,
    load_config,
    select_device,
    set_seed,
    split_validation_dataset,
    STAGE_A_EXPERT_TRAIN_LOADER_NAMESPACE,
    STAGE_A_EXPERT_VAL_LOADER_NAMESPACE,
    validate_manifest_vocab_provenance,
    validate_split_partitions,
)
from fusion.utils import build_grad_scaler, get_amp_context
from fusion.view_protocol import (
    CONTROLLED_VIEW_MECHANISM_VERSION,
    CONTROLLED_VIEW_SEED_FORMULA,
    canonical_manifest_sha256,
    fixed_test_view_plan,
    seed_manifest_sha256,
)


logger = logging.getLogger("care_droid")

CARE_PATHS = ("agm", "ag", "am", "gm")
CARE_PATH_INDEX = {name: index for index, name in enumerate(CARE_PATHS)}
CARE_PROTOCOL_ID = "care_droid_v1"
CARE_STAGE_A_SCHEMA_VERSION = 2
CARE_PIPELINE_SCHEMA_VERSION = 4
CARE_SUMMARY_SCHEMA_VERSION = 2
CARE_HARD_PREDICT_RULE = "malware_log_odds_greater_than_or_equal_to_zero"
CARE_CRC_RISK_NAME = "malware_conditional_accepted_false_negative_risk"
CARE_EVAL_STRENGTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
CARE_ROUTING_VIEW_MECHANISMS = (
    "clean",
    *GRADED_PERTURBATIONS,
    *MISSING_PERTURBATIONS,
)
CARE_DECISION_VIEW_MECHANISMS = ("clean",)
CARE_TEST_VIEW_MECHANISMS = CARE_ROUTING_VIEW_MECHANISMS


@dataclass(frozen=True)
class CareRoleDatasets:
    expert_train: Subset
    expert_val: Subset
    routing_cal: Subset
    decision_cal: Subset
    summary: dict[str, Any]


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if converted != value and not (
        isinstance(value, str) and str(converted) == value.strip()
    ):
        raise ValueError(f"{name} must be an exact integer")
    if converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return converted


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        raise ValueError(
            f"{name} has an invalid closed schema: "
            f"missing={missing}, unknown={unknown}"
        )


def validate_care_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the closed, fully materialized CARE-Droid protocol."""

    method = cfg.get("method", {}) or {}
    model = cfg.get("model", {}) or {}
    loss = cfg.get("loss", {}) or {}
    care = cfg.get("care", {}) or {}
    selective = cfg.get("selective_prediction", {}) or {}
    classification = cfg.get("classification_threshold", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    if not isinstance(train_cfg, Mapping):
        raise ValueError("train must be a mapping")
    if not isinstance(eval_cfg, Mapping):
        raise ValueError("eval must be a mapping")
    for key, default in (
        ("use_amp", True),
        ("deterministic", True),
        ("strict_deterministic", False),
        ("pin_memory", False),
        ("allow_pyg_pin_memory", False),
        ("persistent_workers", True),
    ):
        _strict_bool(
            train_cfg.get(key, default),
            name=f"train.{key}",
        )
    eval_only = _strict_bool(
        eval_cfg.get("eval_only", False),
        name="eval.eval_only",
    )
    run_test = _strict_bool(
        eval_cfg.get("run_test", True),
        name="eval.run_test",
    )
    run_robust_test = _strict_bool(
        eval_cfg.get("run_robust_test", True),
        name="eval.run_robust_test",
    )
    refit_decision = _strict_bool(
        eval_cfg.get("refit_decision_calibration", False),
        name="eval.refit_decision_calibration",
    )
    if not run_test or not run_robust_test:
        raise ValueError(
            "CARE formal runs require eval.run_test=true and "
            "eval.run_robust_test=true"
        )
    if eval_only:
        if not refit_decision:
            raise ValueError(
                "CARE eval-only ablations require "
                "eval.refit_decision_calibration=true"
            )
        if not str(eval_cfg.get("output_name") or "").strip():
            raise ValueError("CARE eval-only requires eval.output_name")
        if not str(eval_cfg.get("checkpoint_path") or "").strip():
            raise ValueError("CARE eval-only requires eval.checkpoint_path")
    elif refit_decision:
        raise ValueError(
            "eval.refit_decision_calibration is valid only for eval-only "
            "CARE ablations"
        )
    if str(method.get("name", "")).strip().lower() != "care_droid":
        raise ValueError("CARE runner requires method.name='care_droid'")
    if str(method.get("protocol_id", "")).strip() != CARE_PROTOCOL_ID:
        raise ValueError(
            f"CARE method.protocol_id must be {CARE_PROTOCOL_ID!r}"
        )
    if str(model.get("fusion_mode", "")).strip().lower() != "care_droid":
        raise ValueError("CARE runner requires model.fusion_mode='care_droid'")
    if str(loss.get("objective", "")).strip().lower() != "care_stage_a_clean":
        raise ValueError(
            "CARE runner requires loss.objective='care_stage_a_clean'"
        )
    if set(loss) - {"objective", "label_smoothing"}:
        raise ValueError(
            "CARE Stage-A loss has a closed schema; unsupported keys="
            f"{sorted(set(loss) - {'objective', 'label_smoothing'})}"
        )
    if not math.isclose(
        float(loss.get("label_smoothing", 0.0)),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("CARE Stage-A uses ordinary CE without label smoothing")
    train_label_smoothing = _finite_float(
        (cfg.get("train", {}) or {}).get("label_smoothing", 0.0),
        name="train.label_smoothing",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(
        train_label_smoothing,
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "CARE Stage-A uses ordinary CE; train.label_smoothing must be zero"
        )
    if classification != {"enabled": False}:
        raise ValueError(
            "CARE v1 fixes the binary hard decision at zero log-odds; "
            "classification_threshold must be exactly {'enabled': false}"
        )
    try:
        num_classes = int(model.get("num_classes", 2))
        raw_num_classes = float(model.get("num_classes", 2))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CARE model.num_classes must be exactly 2") from exc
    if (
        isinstance(model.get("num_classes", 2), bool)
        or not math.isfinite(raw_num_classes)
        or raw_num_classes != 2.0
        or num_classes != 2
    ):
        raise ValueError("CARE model.num_classes must be exactly 2")

    _require_exact_keys(
        care,
        name="care",
        required={
            "protocol_seed",
            "paths",
            "roles",
            "stage_a",
            "risk_training",
            "views",
            "routing",
            "decision",
        },
    )
    paths = tuple(str(value).strip().lower() for value in care["paths"])
    if paths != CARE_PATHS:
        raise ValueError(
            f"care.paths must be exactly {list(CARE_PATHS)} in that order"
        )

    roles = care["roles"]
    _require_exact_keys(
        roles,
        name="care.roles",
        required={
            "expert_val_fraction",
            "expert_split_seed",
            "validation_role_assignment_path",
            "validation_role_schema_version",
            "routing_cal_source_role",
            "decision_cal_source_role",
            "require_fixed_assignment",
        },
    )
    expert_val_fraction = _finite_float(
        roles["expert_val_fraction"],
        name="care.roles.expert_val_fraction",
        minimum=0.05,
        maximum=0.40,
    )
    if not expert_val_fraction < 1.0:
        raise ValueError("care.roles.expert_val_fraction must be below one")
    if roles["require_fixed_assignment"] is not True:
        raise ValueError(
            "CARE formal runs require a fixed validation role assignment"
        )
    if str(roles["routing_cal_source_role"]) != "model_selection":
        raise ValueError(
            "care.roles.routing_cal_source_role must be 'model_selection'"
        )
    if str(roles["decision_cal_source_role"]) != "decision_calibration":
        raise ValueError(
            "care.roles.decision_cal_source_role must be "
            "'decision_calibration'"
        )

    stage_a = care["stage_a"]
    _require_exact_keys(
        stage_a,
        name="care.stage_a",
        required={"clean_only"},
    )
    if stage_a["clean_only"] is not True:
        raise ValueError(
            "CARE-Droid v1 Stage A is frozen to clean-only training"
        )

    risk = care["risk_training"]
    _require_exact_keys(
        risk,
        name="care.risk_training",
        required={
            "enabled",
            "folds",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "hidden_dim",
            "grad_clip",
            "fixed_epochs",
            "early_stopping",
        },
    )
    if risk["enabled"] is not True:
        raise ValueError("CARE main method requires risk_training.enabled=true")
    if risk["fixed_epochs"] is not True or risk["early_stopping"] is not False:
        raise ValueError(
            "CARE OOF holdouts cannot select epochs; use fixed epochs without "
            "early stopping"
        )
    folds = _strict_int(
        risk["folds"], name="care.risk_training.folds", minimum=2
    )
    if folds != 3:
        raise ValueError("CARE v1 fixes risk cross-fitting to three folds")
    risk_epochs = _strict_int(
        risk["epochs"], name="care.risk_training.epochs", minimum=1
    )
    risk_batch_size = _strict_int(
        risk["batch_size"],
        name="care.risk_training.batch_size",
        minimum=1,
    )
    hidden_dim = _strict_int(
        risk["hidden_dim"],
        name="care.risk_training.hidden_dim",
        minimum=1,
    )

    views = care["views"]
    _require_exact_keys(
        views,
        name="care.views",
        required={
            "deterministic",
            "protocol_seed",
            "clean",
            "graded_mechanisms",
            "graded_strength_min",
            "graded_strength_max",
            "strength_assignment",
            "missing_mechanisms",
            "expose_view_metadata_to_model",
        },
    )
    if (
        views["deterministic"] is not True
        or views["clean"] is not True
        or views["expose_view_metadata_to_model"] is not False
    ):
        raise ValueError(
            "CARE views must be deterministic, include clean, and keep view "
            "metadata out of the model"
        )
    if str(views["strength_assignment"]) != "sid_mechanism_hash_uniform":
        raise ValueError(
            "care.views.strength_assignment must be "
            "'sid_mechanism_hash_uniform'"
        )
    if tuple(views["graded_mechanisms"]) != tuple(GRADED_PERTURBATIONS):
        raise ValueError(
            "care.views.graded_mechanisms must match the registered canonical "
            f"order {list(GRADED_PERTURBATIONS)}"
        )
    if tuple(views["missing_mechanisms"]) != tuple(MISSING_PERTURBATIONS):
        raise ValueError(
            "care.views.missing_mechanisms must match the registered canonical "
            f"order {list(MISSING_PERTURBATIONS)}"
        )
    strength_min = _finite_float(
        views["graded_strength_min"],
        name="care.views.graded_strength_min",
        minimum=0.0,
        maximum=1.0,
    )
    strength_max = _finite_float(
        views["graded_strength_max"],
        name="care.views.graded_strength_max",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < strength_min <= strength_max < 1.0:
        raise ValueError(
            "CARE graded strengths must satisfy 0 < min <= max < 1; missing "
            "is represented by separate endpoint views"
        )
    if int(views["protocol_seed"]) != int(care["protocol_seed"]):
        raise ValueError(
            "care.views.protocol_seed must equal care.protocol_seed"
        )

    routing = care["routing"]
    _require_exact_keys(
        routing,
        name="care.routing",
        required={
            "enabled",
            "route_on_all_samples",
            "anchor_path",
            "exactly_two_alive_policy",
            "at_most_one_alive_policy",
        },
        optional={"inference_path"},
    )
    if str(routing["anchor_path"]) != "agm":
        raise ValueError("CARE routing anchor_path must be 'agm'")
    if str(routing["exactly_two_alive_policy"]) != "unique_pair":
        raise ValueError(
            "CARE exactly_two_alive_policy must be 'unique_pair'"
        )
    if str(routing["at_most_one_alive_policy"]) != "reject":
        raise ValueError(
            "CARE at_most_one_alive_policy must be 'reject'"
        )
    for key in ("enabled", "route_on_all_samples"):
        if not isinstance(routing[key], bool):
            raise ValueError(f"care.routing.{key} must be boolean")
    if routing["enabled"] is False:
        if routing["route_on_all_samples"] is not False:
            raise ValueError(
                "Disabled CARE routing requires "
                "route_on_all_samples=false"
            )
        if str(routing.get("inference_path", "")) != "agm":
            raise ValueError(
                "The no-routing ablation must declare inference_path='agm'"
            )
    elif "inference_path" in routing:
        raise ValueError(
            "care.routing.inference_path is valid only when routing is disabled"
        )

    decision = care["decision"]
    _require_exact_keys(
        decision,
        name="care.decision",
        required={"natural_only_crc"},
    )
    if decision["natural_only_crc"] is not True:
        raise ValueError("CARE v1 requires natural-only decision calibration")
    if str(selective.get("mode", "")).strip().lower() != "risk_control":
        raise ValueError("CARE requires selective_prediction.mode='risk_control'")
    score_name = str(selective.get("threshold_score", "")).strip().lower()
    allowed_score = {
        "care_selected_path_correctness",
        "msp",
    }
    if score_name not in allowed_score:
        raise ValueError(
            "CARE threshold_score must be the proposed selected-path score or "
            "the registered MSP ablation"
        )
    if str(selective.get("risk_target", "")).strip().lower() != (
        "malware_conditional_accepted_fn"
    ):
        raise ValueError(
            "CARE risk_target must be 'malware_conditional_accepted_fn'"
        )
    risk_level = _finite_float(
        selective.get("risk_level", 0.05),
        name="selective_prediction.risk_level",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < risk_level < 1.0:
        raise ValueError("CARE risk_level must lie strictly within (0, 1)")
    if selective.get("require_feasible") is not True:
        raise ValueError("CARE formal runs require a feasible CRC certificate")

    configured_mechanisms = tuple(
        str(value).strip().lower()
        for value in eval_cfg.get("perturb_tests", ())
    )
    if configured_mechanisms != CARE_TEST_VIEW_MECHANISMS:
        raise ValueError(
            "CARE formal evaluation must contain exactly the frozen ordered "
            f"mechanisms {list(CARE_TEST_VIEW_MECHANISMS)}"
        )
    configured_strengths = tuple(
        _finite_float(
            value,
            name=f"eval.perturb_strengths[{index}]",
            minimum=0.0,
            maximum=1.0,
        )
        for index, value in enumerate(
            eval_cfg.get("perturb_strengths", ())
        )
    )
    if configured_strengths != CARE_EVAL_STRENGTHS:
        raise ValueError(
            "CARE formal evaluation must contain exactly the frozen ordered "
            f"strengths {list(CARE_EVAL_STRENGTHS)}"
        )

    frozen_values = {
        "care.protocol_seed": (
            int(care["protocol_seed"]),
            424242,
        ),
        "eval.controlled_view_protocol_seed": (
            int(eval_cfg.get("controlled_view_protocol_seed", -1)),
            424242,
        ),
        "care.roles.expert_val_fraction": (
            expert_val_fraction,
            0.10,
        ),
        "care.roles.expert_split_seed": (
            int(roles["expert_split_seed"]),
            4242,
        ),
        "care.roles.validation_role_schema_version": (
            int(roles["validation_role_schema_version"]),
            2,
        ),
        "care.risk_training.epochs": (risk_epochs, 20),
        "care.risk_training.batch_size": (risk_batch_size, 256),
        "care.risk_training.hidden_dim": (hidden_dim, 16),
        "care.risk_training.lr": (float(risk["lr"]), 1.0e-3),
        "care.risk_training.weight_decay": (
            float(risk["weight_decay"]),
            1.0e-4,
        ),
        "care.risk_training.grad_clip": (
            float(risk["grad_clip"]),
            5.0,
        ),
        "care.views.graded_strength_min": (strength_min, 0.1),
        "care.views.graded_strength_max": (strength_max, 0.9),
        "selective_prediction.risk_level": (risk_level, 0.05),
        "selective_prediction.min_calibration_malware": (
            int(selective.get("min_calibration_malware", 1)),
            100,
        ),
    }
    changed = {
        name: {"actual": actual, "required": required}
        for name, (actual, required) in frozen_values.items()
        if (
            not math.isclose(
                float(actual),
                float(required),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    }
    if changed:
        raise ValueError(
            "CARE-Droid v1 frozen hyperparameters were changed: "
            f"{changed}"
        )

    calibration_cfg = cfg.get("calibration", {}) or {}
    if calibration_cfg not in ({}, {"enabled": False}):
        raise ValueError(
            "CARE calibration has the closed schema {'enabled': false}; "
            f"unsupported keys={sorted(calibration_cfg)}"
        )
    fusion_cfg = cfg.get("fusion", {}) or {}
    if fusion_cfg not in ({}, {"mode": "care_droid"}):
        raise ValueError(
            "CARE fusion has the closed schema {'mode': 'care_droid'}; "
            f"unsupported keys={sorted(fusion_cfg)}"
        )

    return {
        "protocol_seed": _strict_int(
            care["protocol_seed"], name="care.protocol_seed", minimum=0
        ),
        "paths": CARE_PATHS,
        "roles": {
            **dict(roles),
            "expert_val_fraction": expert_val_fraction,
            "expert_split_seed": _strict_int(
                roles["expert_split_seed"],
                name="care.roles.expert_split_seed",
                minimum=0,
            ),
            "validation_role_schema_version": _strict_int(
                roles["validation_role_schema_version"],
                name="care.roles.validation_role_schema_version",
                minimum=1,
            ),
        },
        "stage_a": dict(stage_a),
        "risk_training": {
            **dict(risk),
            "folds": folds,
            "epochs": risk_epochs,
            "batch_size": risk_batch_size,
            "hidden_dim": hidden_dim,
            "lr": _finite_float(
                risk["lr"],
                name="care.risk_training.lr",
                minimum=0.0,
            ),
            "weight_decay": _finite_float(
                risk["weight_decay"],
                name="care.risk_training.weight_decay",
                minimum=0.0,
            ),
            "grad_clip": _finite_float(
                risk["grad_clip"],
                name="care.risk_training.grad_clip",
                minimum=0.0,
            ),
        },
        "views": {
            **dict(views),
            "protocol_seed": _strict_int(
                views["protocol_seed"],
                name="care.views.protocol_seed",
                minimum=0,
            ),
            "graded_strength_min": strength_min,
            "graded_strength_max": strength_max,
        },
        "routing": dict(routing),
        "decision": dict(decision),
        "selective": {
            **dict(selective),
            "threshold_score": score_name,
            "risk_level": risk_level,
            "min_calibration_malware": _strict_int(
                selective.get("min_calibration_malware", 1),
                name="selective_prediction.min_calibration_malware",
                minimum=1,
            ),
        },
    }


def _dataset_metadata(dataset: RobustTriModalDataset) -> dict[str, list[Any]]:
    size = len(dataset)
    values = {
        "sids": [
            str(value).strip().lower()
            for value in getattr(dataset, "sample_sids", [])
        ],
        "groups": [
            str(value) for value in getattr(dataset, "sample_groups", [])
        ],
        "labels": [
            int(value) for value in getattr(dataset, "sample_labels", [])
        ],
        "years": [
            int(value) for value in getattr(dataset, "sample_years", [])
        ],
    }
    mismatched = {
        name: len(items)
        for name, items in values.items()
        if len(items) != size
    }
    if mismatched:
        raise ValueError(
            "CARE requires complete dataset identity metadata: "
            f"size={size}, observed={mismatched}"
        )
    if len(set(values["sids"])) != size:
        raise ValueError("CARE datasets must not contain duplicate SIDs")
    return values


def _role_summary(
    dataset: RobustTriModalDataset,
    indices: Iterable[int],
) -> dict[str, Any]:
    meta = _dataset_metadata(dataset)
    normalized = sorted({int(index) for index in indices})
    if not normalized or normalized[-1] >= len(dataset) or normalized[0] < 0:
        raise ValueError("CARE role indices must form a non-empty valid subset")
    sids = [meta["sids"][index] for index in normalized]
    groups = sorted({meta["groups"][index] for index in normalized})
    label_counts = {
        str(label): sum(
            int(meta["labels"][index] == label) for index in normalized
        )
        for label in sorted(set(meta["labels"]))
    }
    year_label_counts = {
        f"{year}:{label}": sum(
            int(
                meta["years"][index] == year
                and meta["labels"][index] == label
            )
            for index in normalized
        )
        for year in sorted(set(meta["years"]))
        for label in sorted(set(meta["labels"]))
    }
    semantic_rows = [
        {
            "sid": meta["sids"][index],
            "group": meta["groups"][index],
            "label": int(meta["labels"][index]),
            "year": int(meta["years"][index]),
        }
        for index in normalized
    ]
    semantic_rows_sha256 = hashlib.sha256(
        json.dumps(
            semantic_rows,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "num_samples": len(normalized),
        "num_groups": len(groups),
        "sample_ids_sha256": _sample_ids_sha256(sids),
        "groups_sha256": _sample_ids_sha256(groups),
        "semantic_rows_sha256": semantic_rows_sha256,
        "label_counts": label_counts,
        "year_label_counts": year_label_counts,
        "indices": normalized,
    }


def split_care_roles(
    cfg: dict[str, Any],
    care_cfg: dict[str, Any],
    train_dataset: RobustTriModalDataset,
    val_dataset: RobustTriModalDataset,
    test_dataset: RobustTriModalDataset | None,
) -> CareRoleDatasets:
    """Build the five immutable, group-disjoint CARE data roles."""

    split_cfg = copy.deepcopy(cfg)
    split_cfg["calibration"] = {
        "validation_fraction": care_cfg["roles"]["expert_val_fraction"],
        "split_seed": care_cfg["roles"]["expert_split_seed"],
        "stratified_group_split": True,
    }
    expert_train, expert_val, train_meta = split_validation_dataset(
        split_cfg, train_dataset
    )

    role_cfg = copy.deepcopy(cfg)
    role_cfg["calibration"] = {
        "role_assignment_path": care_cfg["roles"][
            "validation_role_assignment_path"
        ],
        "require_role_assignment": True,
        "split_seed": int(
            (cfg.get("train", {}) or {}).get("seed", 42)
        ),
        "stratified_group_split": True,
    }
    fixed = _load_fixed_validation_roles(role_cfg, val_dataset)
    if fixed is None:
        raise RuntimeError(
            "CARE formal runs require the immutable validation role assignment"
        )
    routing_cal, decision_cal, val_meta = fixed
    if int(val_meta["role_assignment_schema_version"]) != int(
        care_cfg["roles"]["validation_role_schema_version"]
    ):
        raise ValueError(
            "CARE validation role schema differs from its frozen config"
        )

    role_specs: dict[str, tuple[RobustTriModalDataset, list[int]]] = {
        "expert_train": (
            train_dataset,
            [int(value) for value in expert_train.indices],
        ),
        "expert_val": (
            train_dataset,
            [int(value) for value in expert_val.indices],
        ),
        "routing_cal": (
            val_dataset,
            [int(value) for value in routing_cal.indices],
        ),
        "decision_cal": (
            val_dataset,
            [int(value) for value in decision_cal.indices],
        ),
    }
    if test_dataset is not None:
        role_specs["test"] = (
            test_dataset,
            list(range(len(test_dataset))),
        )
    role_summaries = {
        name: _role_summary(dataset, indices)
        for name, (dataset, indices) in role_specs.items()
    }

    sid_roles: dict[str, str] = {}
    group_roles: dict[str, str] = {}
    for role_name, (dataset, indices) in role_specs.items():
        meta = _dataset_metadata(dataset)
        for index in indices:
            sid = meta["sids"][index]
            group = meta["groups"][index]
            previous_sid = sid_roles.setdefault(sid, role_name)
            if previous_sid != role_name:
                raise RuntimeError(
                    f"CARE SID {sid!r} crosses roles "
                    f"{previous_sid!r}/{role_name!r}"
                )
            previous_group = group_roles.setdefault(group, role_name)
            if previous_group != role_name:
                raise RuntimeError(
                    f"CARE group {group!r} crosses roles "
                    f"{previous_group!r}/{role_name!r}"
                )

    summary = {
        "protocol": (
            "expert_train_expert_val_routing_cal_decision_cal_test_"
            "package_group_disjoint_v1"
        ),
        "expert_split_seed": care_cfg["roles"]["expert_split_seed"],
        "expert_val_fraction": care_cfg["roles"]["expert_val_fraction"],
        "validation_role_assignment_path": val_meta[
            "role_assignment_path"
        ],
        "validation_role_assignment_sha256": val_meta[
            "role_assignment_sha256"
        ],
        "validation_role_assignment_semantic_sha256": val_meta[
            "role_assignment_semantic_sha256"
        ],
        "identity_disjoint": True,
        "group_disjoint": True,
        "roles": role_summaries,
    }
    # Indices are required only in memory and must not bloat summary.yaml.
    public_summary = copy.deepcopy(summary)
    for role in public_summary["roles"].values():
        role.pop("indices", None)
    summary["public"] = public_summary
    return CareRoleDatasets(
        expert_train=expert_train,
        expert_val=expert_val,
        routing_cal=routing_cal,
        decision_cal=decision_cal,
        summary=summary,
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _care_role_identity_payload(
    public_role_summary: Mapping[str, Any],
) -> dict[str, Any]:
    roles = public_role_summary.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("CARE public role summary omits roles")
    role_identity: dict[str, Any] = {}
    for name in (
        "expert_train",
        "expert_val",
        "routing_cal",
        "decision_cal",
        "test",
    ):
        row = roles.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"CARE public role summary omits {name!r}")
        role_identity[name] = {
            "num_samples": int(row["num_samples"]),
            "num_groups": int(row["num_groups"]),
            "sample_ids_sha256": str(row["sample_ids_sha256"]),
            "groups_sha256": str(row["groups_sha256"]),
            "semantic_rows_sha256": str(
                row["semantic_rows_sha256"]
            ),
        }
    return {
        "protocol": str(public_role_summary.get("protocol", "")),
        "expert_split_seed": int(
            public_role_summary["expert_split_seed"]
        ),
        "expert_val_fraction": float(
            public_role_summary["expert_val_fraction"]
        ),
        "validation_role_assignment_semantic_sha256": str(
            public_role_summary[
                "validation_role_assignment_semantic_sha256"
            ]
        ),
        "roles": role_identity,
    }


def _care_role_identity_sha256(
    public_role_summary: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        _care_role_identity_payload(public_role_summary),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _care_data_lineage_payload(
    cfg: Mapping[str, Any],
    manifest_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a pipeline to the PT build and train-only Manifest vocabulary."""

    data_cfg = cfg.get("data", {}) or {}
    if not isinstance(data_cfg, Mapping):
        raise ValueError("CARE data configuration must be a mapping")
    if not bool(manifest_provenance.get("verified", False)):
        raise ValueError(
            "CARE formal artifacts require verified train-only Manifest "
            "vocabulary provenance"
        )
    pt_build_fingerprint = str(
        data_cfg.get("expected_pt_build_fingerprint") or ""
    ).strip().lower()
    if len(pt_build_fingerprint) != 64 or any(
        character not in "0123456789abcdef"
        for character in pt_build_fingerprint
    ):
        raise ValueError(
            "CARE data.expected_pt_build_fingerprint must be a SHA-256 digest"
        )
    certificate_value = str(
        data_cfg.get("pt_audit_certificate") or ""
    ).strip()
    if not certificate_value:
        raise ValueError("CARE formal artifacts require a PT audit certificate")
    certificate_path = Path(certificate_value).expanduser()
    if not certificate_path.is_absolute():
        data_root = Path(str(data_cfg.get("root") or ".")).expanduser()
        certificate_path = data_root / certificate_path
    certificate_path = certificate_path.resolve()
    if not certificate_path.is_file():
        raise FileNotFoundError(
            f"CARE PT audit certificate not found: {certificate_path}"
        )
    manifest_digests = {
        "manifest_vocab_sha256": str(
            manifest_provenance.get("manifest_vocab_sha256") or ""
        ).strip().lower(),
        "manifest_train_csv_sha256": str(
            manifest_provenance.get("train_csv_sha256") or ""
        ).strip().lower(),
        "manifest_train_sample_ids_sha256": str(
            manifest_provenance.get("train_sample_ids_sha256") or ""
        ).strip().lower(),
    }
    for name, digest in manifest_digests.items():
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"CARE {name} must be a SHA-256 digest")
    num_train_samples = int(
        manifest_provenance.get("num_train_samples", -1)
    )
    if num_train_samples <= 0:
        raise ValueError(
            "CARE Manifest provenance must contain a positive train count"
        )
    return {
        "domain": "care_droid_data_lineage_v1",
        "pt_build_fingerprint": pt_build_fingerprint,
        "pt_audit_certificate_sha256": _file_sha256(certificate_path),
        **manifest_digests,
        "manifest_num_train_samples": num_train_samples,
    }


def _care_data_lineage_sha256(
    cfg: Mapping[str, Any],
    manifest_provenance: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        _care_data_lineage_payload(cfg, manifest_provenance),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _care_semantic_protocol_payload(
    care_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove machine-local locators from the algorithm identity."""

    payload = _json_compatible(dict(care_cfg))
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("CARE protocol identity requires care.roles")
    roles = dict(roles)
    # Membership is bound separately by role_identity_sha256.  Including the
    # locator here would make byte-identical role assignments appear to be
    # different algorithms merely because two machines mount the repository
    # at different paths.
    roles.pop("validation_role_assignment_path", None)
    payload["roles"] = roles
    return payload


def _care_protocol_sha256(care_cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _care_semantic_protocol_payload(care_cfg),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _care_upstream_protocol_sha256(
    care_cfg: Mapping[str, Any],
) -> str:
    """Hash only the stages that produce Stage A and the path-risk head.

    Registered eval-only ablations may change routing or the CRC acceptance
    score, but they must reuse exactly the same experts, roles, views, and risk
    fitting protocol.
    """

    semantic = _care_semantic_protocol_payload(care_cfg)
    payload = {
        key: semantic[key]
        for key in (
            "protocol_seed",
            "paths",
            "roles",
            "stage_a",
            "risk_training",
            "views",
            "decision",
        )
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _care_upstream_runtime_payload(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe inference semantics that state-dict shapes cannot prove.

    Eval-only ablations reuse a fitted path-risk head.  A strict state load
    does not detect changes such as a different API/graph runtime budget,
    attention dropout configuration, or AMP policy.  Bind those semantics
    without binding machine-local data/output paths.
    """

    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    if not isinstance(model_cfg, Mapping):
        raise ValueError("CARE model configuration must be a mapping")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("CARE data configuration must be a mapping")
    if not isinstance(train_cfg, Mapping):
        raise ValueError("CARE train configuration must be a mapping")
    return {
        "domain": "care_droid_upstream_runtime_v1",
        "model": _json_compatible(dict(model_cfg)),
        "data_input_semantics": {
            "max_api_events_per_sample": _json_compatible(
                data_cfg.get("max_api_events_per_sample")
            ),
        },
        "numeric_inference": {
            "use_amp": _strict_bool(
                train_cfg.get("use_amp", True),
                name="train.use_amp",
            ),
            "deterministic": _strict_bool(
                train_cfg.get("deterministic", True),
                name="train.deterministic",
            ),
            "strict_deterministic": _strict_bool(
                train_cfg.get("strict_deterministic", False),
                name="train.strict_deterministic",
            ),
        },
    }


def _care_upstream_runtime_sha256(
    cfg: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        _care_upstream_runtime_payload(cfg),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            _json_compatible(dict(payload)),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_compatible(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _care_output_directory_path(cfg: dict[str, Any]) -> Path:
    train_cfg = cfg.get("train", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    seed = _strict_int(train_cfg.get("seed", 42), name="train.seed")
    eval_cfg = cfg.get("eval", {}) or {}
    raw_exp_name = (
        eval_cfg.get("output_name")
        if _strict_bool(
            eval_cfg.get("eval_only", False),
            name="eval.eval_only",
        )
        else train_cfg.get("exp_name", "care_droid")
    )
    exp_name = str(raw_exp_name or "").strip()
    if not exp_name:
        raise ValueError("train.exp_name must be non-empty")
    out_dir = Path(data_cfg.get("out_dir", "results/tri_modal_robust"))
    return out_dir / exp_name / str(seed)


def _resolve_eval_checkpoint_path(cfg: Mapping[str, Any]) -> Path:
    eval_cfg = cfg.get("eval", {}) or {}
    checkpoint_value = str(eval_cfg.get("checkpoint_path") or "").strip()
    if not checkpoint_value:
        raise ValueError("CARE eval_only requires eval.checkpoint_path")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    return checkpoint_path.resolve()


def _validate_source_training_seed(
    source: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> int:
    """Bind an eval-only artifact to the model-training seed it claims."""

    current_train = cfg.get("train", {}) or {}
    source_cfg = source.get("cfg")
    if not isinstance(current_train, Mapping):
        raise ValueError("CARE train configuration must be a mapping")
    if not isinstance(source_cfg, Mapping):
        raise ValueError("CARE source pipeline omits its resolved config")
    source_train = source_cfg.get("train", {}) or {}
    if not isinstance(source_train, Mapping):
        raise ValueError("CARE source pipeline has invalid train config")
    current_seed = _strict_int(
        current_train.get("seed", 42),
        name="train.seed",
    )
    stored_seed = _strict_int(
        source.get("training_seed"),
        name="source.training_seed",
    )
    source_cfg_seed = _strict_int(
        source_train.get("seed"),
        name="source.cfg.train.seed",
    )
    if stored_seed != source_cfg_seed:
        raise ValueError(
            "CARE source pipeline training_seed disagrees with its resolved "
            "config"
        )
    if current_seed != stored_seed:
        raise ValueError(
            "CARE eval-only config train.seed must match the source model "
            f"training seed: current={current_seed} source={stored_seed}"
        )
    return stored_seed


def _preflight_care_output_target(
    cfg: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Fail before PT preflight when an output request is already unsafe."""

    out_dir = _care_output_directory_path(dict(cfg))
    generated = {
        "best_care_stage_a.pt",
        "best_care_pipeline.pt",
        "resolved_config.yaml",
        "summary.yaml",
        "care_view_manifest.json",
        "care_oof_predictions.csv",
        "care_route_diagnostics.csv",
        "care_decision_calibration.csv",
        "risk_coverage_curve.csv",
    }
    collisions = sorted(
        name for name in generated if (out_dir / name).exists()
    )
    if collisions and not overwrite:
        raise FileExistsError(
            f"Output directory {out_dir} already contains CARE artifacts "
            f"{collisions}; choose a new exp_name/seed or pass --overwrite"
        )
    if _strict_bool(
        (cfg.get("eval", {}) or {}).get("eval_only", False),
        name="eval.eval_only",
    ):
        source_pipeline = _resolve_eval_checkpoint_path(cfg)
        if not source_pipeline.is_file():
            raise FileNotFoundError(
                f"CARE source pipeline not found: {source_pipeline}"
            )
        destination_pipeline = (
            out_dir / "best_care_pipeline.pt"
        ).resolve()
        if source_pipeline == destination_pipeline:
            raise ValueError(
                "CARE eval_only source pipeline equals its output artifact; "
                "--overwrite would delete the source before loading it"
            )


def _prepare_care_output_directory(
    cfg: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    out_dir = _care_output_directory_path(cfg)
    generated = {
        "best_care_stage_a.pt",
        "best_care_pipeline.pt",
        "resolved_config.yaml",
        "summary.yaml",
        "care_view_manifest.json",
        "care_oof_predictions.csv",
        "care_route_diagnostics.csv",
        "care_decision_calibration.csv",
        "risk_coverage_curve.csv",
    }
    collisions = sorted(
        name for name in generated if (out_dir / name).exists()
    )
    if collisions and not overwrite:
        raise FileExistsError(
            f"Output directory {out_dir} already contains CARE artifacts "
            f"{collisions}; choose a new exp_name/seed or pass --overwrite"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in generated:
            target = out_dir / name
            if target.is_file() or target.is_symlink():
                target.unlink()
            elif target.exists():
                raise ValueError(
                    f"Refusing to overwrite non-file CARE artifact: {target}"
                )
    return out_dir


# The model/training/evaluation implementation is defined below the protocol
# helpers.  Keeping the protocol validation at module scope makes it directly
# unit-testable without loading any APK tensors.


@dataclass(frozen=True)
class CareCachedView:
    sids: tuple[str, ...]
    groups: tuple[str, ...]
    labels: torch.Tensor
    path_logits: torch.Tensor
    modality_alive: torch.Tensor
    path_available: torch.Tensor
    output_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.sids)
        if (
            size <= 0
            or len(self.groups) != size
            or len(self.output_digests) != size
        ):
            raise ValueError("CARE cached view identity lengths disagree")
        if self.labels.shape != (size,):
            raise ValueError("CARE cached labels must have shape [N]")
        if self.labels.is_floating_point() or self.labels.dtype == torch.bool:
            raise TypeError("CARE cached labels must be an integer tensor")
        if self.path_logits.shape != (size, len(CARE_PATHS), 2):
            raise ValueError("CARE cached path logits must have shape [N, 4, 2]")
        if self.modality_alive.shape != (size, 3):
            raise ValueError("CARE cached modality_alive must have shape [N, 3]")
        if self.path_available.shape != (size, len(CARE_PATHS)):
            raise ValueError(
                "CARE cached path_available must have shape [N, 4]"
            )
        if self.modality_alive.dtype != torch.bool:
            raise TypeError("CARE cached modality_alive must be boolean")
        if self.path_available.dtype != torch.bool:
            raise TypeError("CARE cached path_available must be boolean")
        if len(set(self.sids)) != size:
            raise ValueError("CARE cached SIDs must be unique")
        if any(not str(value) for value in self.groups):
            raise ValueError("CARE cached groups must be non-empty")
        if not bool(
            ((self.labels == 0) | (self.labels == 1)).all().item()
        ):
            raise ValueError("CARE cached labels must be binary")
        if not bool(torch.isfinite(self.path_logits).all().item()):
            raise ValueError("CARE cached path logits must be finite")
        expected_available = path_availability(self.modality_alive)
        if not torch.equal(self.path_available, expected_available):
            raise ValueError(
                "CARE cached path availability disagrees with modality alive"
            )

    @property
    def path_log_odds(self) -> torch.Tensor:
        return self.path_logits[..., 1] - self.path_logits[..., 0]


@dataclass(frozen=True)
class CareRoutedBatch:
    selected_path_index: torch.Tensor
    selected_logits: torch.Tensor
    selected_score: torch.Tensor
    eligible: torch.Tensor
    disagreement_with_agm: torch.Tensor

    @property
    def prediction(self) -> torch.Tensor:
        original_shape = self.selected_logits.shape[:-1]
        prediction = hard_predict(
            self.selected_logits.reshape(-1, 2)
        ).reshape(original_shape)
        return torch.where(
            self.eligible,
            prediction,
            torch.full_like(prediction, -1),
        )


def _cpu_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in state.items()
    }


def _care_source_code_sha256() -> str:
    digest = hashlib.sha256()
    source_dir = Path(__file__).resolve().parent
    # Hash the complete runtime package, not a hand-maintained shortlist.
    # CARE depends transitively on encoders, evidence/availability helpers and
    # PT schema validation; omitting one of those files would make the
    # reproducibility digest claim stronger than the artifact can support.
    for path in sorted(source_dir.rglob("*.py")):
        name = path.relative_to(source_dir).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_care_model(
    cfg: dict[str, Any],
    feature_dim: int,
    care_cfg: dict[str, Any],
) -> CareDroidModel:
    model_cfg = cfg.get("model", {}) or {}
    api_cfg = model_cfg.get("api_encoder", {}) or {}
    graph_cfg = model_cfg.get("graph_encoder", {}) or {}
    manifest_cfg = model_cfg.get("manifest_encoder", {}) or {}
    if int(care_cfg["risk_training"]["hidden_dim"]) != 16:
        raise ValueError("CARE v1 fixes the shared risk-head hidden width to 16")
    graph_budget = int(
        graph_cfg.get(
            "max_nodes",
            model_cfg.get("max_nodes_gnn", 12288),
        )
    )
    model = CareDroidModel(
        in_feat_dim=int(feature_dim),
        num_classes=int(model_cfg.get("num_classes", 2)),
        api_num_hash_buckets=int(api_cfg.get("num_hash_buckets", 8192)),
        api_type_vocab_size=int(api_cfg.get("type_vocab_size", 16)),
        api_emb_dim=int(api_cfg.get("emb_dim", 128)),
        api_hidden_dim=int(api_cfg.get("hidden_dim", 256)),
        api_dropout=float(api_cfg.get("dropout", 0.15)),
        api_encoder_type=str(api_cfg.get("type", "transformer")),
        api_layers=int(api_cfg.get("layers", 2)),
        api_heads=int(api_cfg.get("heads", 4)),
        api_max_seq_len=int(api_cfg.get("max_seq_len", 2048)),
        graph_emb_dim=int(graph_cfg.get("emb_dim", 128)),
        graph_hidden=int(graph_cfg.get("hidden", 128)),
        graph_heads=int(graph_cfg.get("heads", 4)),
        graph_layers=int(graph_cfg.get("layers", 2)),
        graph_encoder_type=str(graph_cfg.get("type", "gatv2")),
        max_nodes_gnn=graph_budget,
        use_graph_behavior_hint=bool(
            graph_cfg.get("use_behavior_hint", False)
        ),
        manifest_in_dim=int(manifest_cfg.get("in_dim", 256)),
        manifest_emb_dim=int(manifest_cfg.get("emb_dim", 128)),
        manifest_hidden_dim=int(manifest_cfg.get("hidden_dim", 256)),
        manifest_dropout=float(manifest_cfg.get("dropout", 0.1)),
    )
    if model.care_path_heads is None or model.care_risk_head is None:
        raise RuntimeError("CARE model did not instantiate its four heads/risk head")
    if int(model.care_risk_head.hidden_dim) != 16:
        raise RuntimeError("CARE model risk-head width differs from protocol")
    return model


def _activate_care_risk_training_partition(
    model: CareDroidModel,
) -> None:
    """Freeze Stage A and make only the shared correctness head trainable."""

    if model.care_risk_head is None:
        raise RuntimeError("CARE risk head is unavailable before cross-fitting")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.care_risk_head.parameters():
        parameter.requires_grad_(True)
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    expected_ids = {
        id(parameter) for parameter in model.care_risk_head.parameters()
    }
    if trainable_ids != expected_ids or not expected_ids:
        raise RuntimeError(
            "CARE risk-training parameter partition is not exactly the risk head"
        )


def _binary_metrics_from_logits(
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> dict[str, Any]:
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    logits = logits.detach().float().cpu()
    probabilities = torch.softmax(logits, dim=-1)[:, 1].numpy()
    predictions = hard_predict(logits).numpy().astype(np.int64)
    result: dict[str, Any] = {
        "num_samples": int(labels_np.size),
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "macro_f1": float(
            f1_score(
                labels_np,
                predictions,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        "malware_recall": float(
            recall_score(labels_np, predictions, pos_label=1, zero_division=0)
        ),
        "malware_f1": float(
            f1_score(
                labels_np,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        ),
        "brier": float(brier_score_loss(labels_np, probabilities)),
    }
    if np.unique(labels_np).size == 2:
        result["auc"] = float(roc_auc_score(labels_np, probabilities))
        result["ap"] = float(
            average_precision_score(labels_np, probabilities)
        )
        result["average_precision"] = float(
            average_precision_score(labels_np, probabilities)
        )
        result["log_loss"] = float(
            log_loss(labels_np, probabilities, labels=[0, 1])
        )
    else:
        result.update(
            {
                "auc": float("nan"),
                "average_precision": float("nan"),
                "log_loss": float("nan"),
            }
        )
    return result


def _train_care_stage_a_epoch(
    model: CareDroidModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    model.set_care_risk_active(False)
    use_amp = bool((cfg.get("train", {}) or {}).get("use_amp", True))
    grad_clip = float((cfg.get("train", {}) or {}).get("grad_clip", 1.0))
    label_smoothing = float(
        (cfg.get("train", {}) or {}).get("label_smoothing", 0.0)
    )
    total_loss = 0.0
    totals = {f"care_ce_{name}": 0.0 for name in CARE_PATHS}
    steps = 0
    valid_seen = 0
    failed_seen = 0
    for batch in tqdm(loader, desc=f"care stage-a {epoch}", leave=False):
        graph, labels, _sids, failed = prepare_robust_batch(batch, device)
        failed_seen += int(failed)
        if graph is None:
            continue
        valid_seen += int(labels.numel())
        optimizer.zero_grad(set_to_none=True)
        with get_amp_context(device, use_amp):
            _logits, extra = model(graph)
            loss, parts = compute_care_stage_a_loss(
                extra["care_path_logits"],
                labels,
                extra["care_path_available"],
                label_smoothing=label_smoothing,
                materialize_diagnostics=False,
            )
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError(
                f"CARE Stage-A loss became non-finite at epoch={epoch}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item())
        for name in totals:
            value = parts.get(name)
            if isinstance(value, torch.Tensor):
                totals[name] += float(value.detach().item())
            elif value is not None:
                totals[name] += float(value)
        steps += 1
    enforce_failed_ratio(
        {"num_eval": valid_seen, "num_failed": failed_seen},
        cfg,
        f"care_stage_a_train_epoch_{epoch}",
    )
    if steps <= 0:
        raise RuntimeError("CARE Stage-A training loader produced no batch")
    return (
        total_loss / float(steps),
        {name: value / float(steps) for name, value in totals.items()},
    )


@torch.no_grad()
def _cache_care_view(
    model: CareDroidModel,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    *,
    split_name: str,
) -> CareCachedView:
    model.eval()
    model.set_care_risk_active(False)
    sids: list[str] = []
    groups: list[str] = []
    labels_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    alive_parts: list[torch.Tensor] = []
    available_parts: list[torch.Tensor] = []
    output_digests: list[str] = []
    failed_seen = 0
    valid_seen = 0
    use_amp = bool((cfg.get("train", {}) or {}).get("use_amp", True))
    for batch in tqdm(loader, desc=f"cache {split_name}", leave=False):
        graph, labels, batch_sids, failed = prepare_robust_batch(batch, device)
        failed_seen += int(failed)
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            _primary, extra = model(graph)
        path_logits = extra.get("care_path_logits")
        if not isinstance(path_logits, dict) or set(path_logits) != set(
            CARE_PATHS
        ):
            raise RuntimeError("CARE model did not expose all path logits")
        stacked = torch.stack(
            [path_logits[name].detach().float().cpu() for name in CARE_PATHS],
            dim=1,
        )
        alive = extra.get("care_modality_alive")
        available = extra.get("care_path_available")
        if not isinstance(alive, torch.Tensor) or not isinstance(
            available, torch.Tensor
        ):
            raise RuntimeError("CARE model omitted availability tensors")
        current_groups = [
            str(value) for value in batch.get("sample_groups", [])
        ]
        current_sids = [str(value).strip().lower() for value in batch_sids or []]
        if (
            len(current_sids) != int(labels.numel())
            or len(current_groups) != int(labels.numel())
        ):
            raise RuntimeError("CARE cache lost SID/package-group alignment")
        current_digests = [
            str(value)
            for value in batch.get(
                "care_view_output_digests",
                [""] * int(labels.numel()),
            )
        ]
        if len(current_digests) != int(labels.numel()):
            raise RuntimeError("CARE view output-digest count is invalid")
        sids.extend(current_sids)
        groups.extend(current_groups)
        output_digests.extend(current_digests)
        labels_parts.append(labels.detach().long().cpu())
        logits_parts.append(stacked)
        alive_parts.append(alive.detach().bool().cpu())
        available_parts.append(available.detach().bool().cpu())
        valid_seen += int(labels.numel())
    enforce_failed_ratio(
        {"num_eval": valid_seen, "num_failed": failed_seen},
        cfg,
        split_name,
    )
    if not labels_parts:
        raise RuntimeError(f"{split_name} produced no CARE predictions")
    cached = CareCachedView(
        sids=tuple(sids),
        groups=tuple(groups),
        labels=torch.cat(labels_parts, dim=0),
        path_logits=torch.cat(logits_parts, dim=0),
        modality_alive=torch.cat(alive_parts, dim=0),
        path_available=torch.cat(available_parts, dim=0),
        output_digests=tuple(output_digests),
    )
    if len(set(cached.sids)) != len(cached.sids):
        raise RuntimeError(f"{split_name} contains duplicate SIDs")
    return cached


def _evaluate_stage_a(
    model: CareDroidModel,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    *,
    split_name: str,
) -> dict[str, Any]:
    cached = _cache_care_view(
        model,
        loader,
        device,
        cfg,
        split_name=split_name,
    )
    if not bool(cached.path_available.all().item()):
        unavailable = (~cached.path_available).sum(dim=0).tolist()
        raise ValueError(
            "CARE clean expert_val requires all four paths; unavailable "
            f"counts={dict(zip(CARE_PATHS, map(int, unavailable)))}"
        )
    paths: dict[str, Any] = {}
    for index, name in enumerate(CARE_PATHS):
        mask = cached.path_available[:, index]
        if not bool(mask.any().item()):
            paths[name] = {
                "num_samples": 0,
                "available_fraction": 0.0,
                "macro_f1": float("nan"),
            }
            continue
        metrics = _binary_metrics_from_logits(
            cached.labels[mask],
            cached.path_logits[mask, index],
        )
        metrics["available_fraction"] = float(mask.float().mean().item())
        paths[name] = metrics
    agm_score = float(paths["agm"]["macro_f1"])
    if not math.isfinite(agm_score):
        raise RuntimeError("CARE Stage-A expert_val has no valid AGM prediction")
    return {
        "checkpoint_metric": "clean_agm_macro_f1",
        "checkpoint_score": agm_score,
        "paths": paths,
    }


def _fit_stage_a(
    model: CareDroidModel,
    roles: CareRoleDatasets,
    cfg: dict[str, Any],
    care_cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    role_summary: dict[str, Any],
) -> dict[str, Any]:
    train_loader = build_loader(
        cfg,
        roles.expert_train,
        is_train=True,
        seed_namespace=STAGE_A_EXPERT_TRAIN_LOADER_NAMESPACE,
    )
    val_loader = build_loader(
        cfg,
        roles.expert_val,
        is_train=False,
        seed_namespace=STAGE_A_EXPERT_VAL_LOADER_NAMESPACE,
    )
    for parameter in model.care_risk_head.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"].get("lr", 3.0e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1.0e-2)),
    )
    epochs = int(cfg["train"].get("epochs", 60))
    patience = int(cfg["train"].get("patience", 8))
    if epochs <= 0 or patience <= 0:
        raise ValueError("CARE Stage-A epochs/patience must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(cfg["train"].get("eta_min", 1.0e-6)),
    )
    scaler = build_grad_scaler(
        device,
        bool(cfg["train"].get("use_amp", True)),
    )
    best_score = float("-inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = out_dir / "best_care_stage_a.pt"
    training_seed = _strict_int(
        cfg["train"].get("seed", 42),
        name="train.seed",
    )
    method_protocol_sha256 = _care_protocol_sha256(care_cfg)
    source_code_sha256 = _care_source_code_sha256()
    resolved_cfg = _json_compatible(cfg)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_parts = _train_care_stage_a_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch,
        )
        validation = _evaluate_stage_a(
            model,
            val_loader,
            device,
            cfg,
            split_name=f"care_expert_val_epoch_{epoch}",
        )
        score = float(validation["checkpoint_score"])
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_path_ce": train_parts,
            "validation": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "wall_seconds": float(time.perf_counter() - started),
        }
        history.append(epoch_row)
        logger.info(
            "care_stage_a epoch=%d loss=%.6f agm_macro_f1=%.6f "
            "path_macro_f1=%s wall_seconds=%.2f",
            epoch,
            train_loss,
            score,
            {
                name: validation["paths"][name]["macro_f1"]
                for name in CARE_PATHS
            },
            epoch_row["wall_seconds"],
        )
        # Strict comparison makes exact ties retain the earliest epoch.
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            stage_state = _cpu_state_dict(model.care_stage_a_state_dict())
            artifact = {
                "schema_version": CARE_STAGE_A_SCHEMA_VERSION,
                "artifact_type": "care_droid_stage_a",
                "protocol_id": CARE_PROTOCOL_ID,
                "training_seed": training_seed,
                "method_protocol_sha256": method_protocol_sha256,
                "hard_prediction_rule": CARE_HARD_PREDICT_RULE,
                "selection_rule": (
                    "maximum_clean_agm_macro_f1_exact_tie_earliest_epoch"
                ),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                "stage_a_state": stage_state,
                "stage_a_state_sha256": _state_dict_sha256(stage_state),
                "role_summary": role_summary,
                "source_code_sha256": source_code_sha256,
                "cfg": resolved_cfg,
            }
            _atomic_torch_save(artifact, checkpoint_path)
        else:
            stale += 1
        if stale >= patience:
            break
    if best_epoch < 0 or not checkpoint_path.is_file():
        raise RuntimeError("CARE Stage-A did not produce a checkpoint")
    artifact = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifact, Mapping) or (
        artifact.get("schema_version") != CARE_STAGE_A_SCHEMA_VERSION
        or artifact.get("artifact_type") != "care_droid_stage_a"
        or artifact.get("protocol_id") != CARE_PROTOCOL_ID
        or artifact.get("training_seed") != training_seed
        or artifact.get("method_protocol_sha256")
        != method_protocol_sha256
        or artifact.get("source_code_sha256") != source_code_sha256
        or artifact.get("role_summary") != role_summary
        or artifact.get("cfg") != resolved_cfg
    ):
        raise RuntimeError(
            "CARE Stage-A checkpoint failed schema/protocol/code/role audit"
        )
    stage_state = artifact.get("stage_a_state")
    if not isinstance(stage_state, dict):
        raise RuntimeError("CARE Stage-A checkpoint omits stage_a_state")
    if _state_dict_sha256(stage_state) != artifact.get(
        "stage_a_state_sha256"
    ):
        raise RuntimeError("CARE Stage-A checkpoint state digest is invalid")
    model.load_care_stage_a_state_dict(stage_state)
    actual_hash = _state_dict_sha256(
        _cpu_state_dict(model.care_stage_a_state_dict())
    )
    if actual_hash != artifact["stage_a_state_sha256"]:
        raise RuntimeError("CARE Stage-A checkpoint failed strict reload audit")
    summary = {
        "objective": "equal_mean_clean_ce_over_agm_ag_am_gm",
        "loss_formula": "(CE_AGM + CE_AG + CE_AM + CE_GM) / 4",
        "path_head": {
            "independent_parameters": True,
            "hidden_dim": 128,
            "activation": "relu",
            "dropout": 0.1,
        },
        "checkpoint_selection": (
            "clean AGM Macro-F1 on group-disjoint expert_val"
        ),
        "exact_tie_policy": "earliest_epoch",
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "epochs_ran": len(history),
        "history": history,
        "stage_a_state_sha256": actual_hash,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    return summary


def _subset_base_and_indices(
    subset: Subset,
) -> tuple[RobustTriModalDataset, list[int]]:
    base: Dataset = subset.dataset
    indices = [int(value) for value in subset.indices]
    while isinstance(base, Subset):
        parent_indices = [int(value) for value in base.indices]
        indices = [parent_indices[index] for index in indices]
        base = base.dataset
    if not isinstance(base, RobustTriModalDataset):
        raise TypeError("CARE roles must resolve to RobustTriModalDataset")
    return base, indices


def _make_deterministic_view(
    base: RobustTriModalDataset,
    *,
    mechanism: str,
    care_cfg: dict[str, Any],
) -> tuple[RobustTriModalDataset, list[dict[str, Any]]]:
    view = copy.copy(base)
    view.is_train = False
    view.care_digest_view = True
    protocol_seed = int(care_cfg["views"]["protocol_seed"])
    minimum = float(care_cfg["views"]["graded_strength_min"])
    maximum = float(care_cfg["views"]["graded_strength_max"])
    records: list[dict[str, Any]] = []
    plan: list[tuple[str, float, int]] = []
    for sid in view.sample_sids:
        if mechanism == "clean":
            spec = deterministic_view_spec(
                sid,
                "clean",
                protocol_seed,
                0.0,
                0.0,
            )
        elif mechanism in MISSING_PERTURBATIONS:
            spec = deterministic_view_spec(
                sid,
                mechanism,
                protocol_seed,
                1.0,
                1.0,
            )
            plan.append(
                (mechanism, float(spec["strength"]), int(spec["seed"]))
            )
        else:
            spec = deterministic_view_spec(
                sid,
                mechanism,
                protocol_seed,
                minimum,
                maximum,
            )
            plan.append(
                (mechanism, float(spec["strength"]), int(spec["seed"]))
            )
        records.append(
            {
                "sid": str(sid),
                "view_name": mechanism,
                "mechanism": mechanism,
                "mechanism_version": CONTROLLED_VIEW_MECHANISM_VERSION,
                "sampled_strength": float(spec["strength"]),
                "view_seed": int(spec["seed"]),
            }
        )
    view.eval_perturb_plan = None if mechanism == "clean" else tuple(plan)
    return view, records


def _source_digests_for_indices(
    base: RobustTriModalDataset,
    indices: Sequence[int],
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for index in indices:
        path, _label, sid, _year = base.samples[int(index)]
        sid = str(sid).strip().lower()
        if sid in digests:
            raise RuntimeError("CARE source-digest population repeats a SID")
        digests[sid] = _file_sha256(path)
    return digests


def _cache_routing_views(
    model: CareDroidModel,
    routing_subset: Subset,
    cfg: dict[str, Any],
    care_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[CareRiskCalibrationCache, list[dict[str, Any]]]:
    base, indices = _subset_base_and_indices(routing_subset)
    mechanisms = (
        "clean",
        *tuple(care_cfg["views"]["graded_mechanisms"]),
        *tuple(care_cfg["views"]["missing_mechanisms"]),
    )
    if mechanisms != CARE_ROUTING_VIEW_MECHANISMS:
        raise RuntimeError(
            "CARE routing-cal view registry drifted from the frozen protocol: "
            f"observed={list(mechanisms)} "
            f"expected={list(CARE_ROUTING_VIEW_MECHANISMS)}"
        )
    source_digests = _source_digests_for_indices(base, indices)
    cached_views: list[CareCachedView] = []
    manifest: list[dict[str, Any]] = []
    reference_sids: tuple[str, ...] | None = None
    reference_groups: tuple[str, ...] | None = None
    reference_labels: torch.Tensor | None = None
    for mechanism in mechanisms:
        full_view, records = _make_deterministic_view(
            base,
            mechanism=mechanism,
            care_cfg=care_cfg,
        )
        view_subset = Subset(full_view, indices)
        loader = build_loader(
            cfg,
            view_subset,
            is_train=False,
            seed_namespace=f"care/routing_cal/{mechanism}",
            persistent_workers_override=False,
        )
        cached = _cache_care_view(
            model,
            loader,
            device,
            cfg,
            split_name=f"care_routing_cal_{mechanism}",
        )
        if reference_sids is None:
            reference_sids = cached.sids
            reference_groups = cached.groups
            reference_labels = cached.labels
        elif (
            cached.sids != reference_sids
            or cached.groups != reference_groups
            or not torch.equal(cached.labels, reference_labels)
        ):
            raise RuntimeError(
                "CARE deterministic views changed routing-cal identity/order"
            )
        record_by_sid = {
            str(record["sid"]).strip().lower(): record for record in records
        }
        for sid, output_digest in zip(
            cached.sids,
            cached.output_digests,
        ):
            if not output_digest:
                raise RuntimeError(
                    f"CARE view {mechanism!r} omitted output digest for {sid}"
                )
            record = dict(record_by_sid[sid])
            record["source_digest"] = source_digests[sid]
            record["output_digest"] = output_digest
            manifest.append(record)
        cached_views.append(cached)
    if (
        reference_sids is None
        or reference_groups is None
        or reference_labels is None
    ):
        raise RuntimeError("CARE routing-cal view registry is empty")
    path_logits = torch.stack(
        [cached.path_logits for cached in cached_views],
        dim=1,
    )
    modality_alive = torch.stack(
        [cached.modality_alive for cached in cached_views],
        dim=1,
    )
    cache = CareRiskCalibrationCache.from_path_logits(
        sids=reference_sids,
        groups=reference_groups,
        labels=reference_labels,
        view_names=mechanisms,
        path_logits=path_logits,
        modality_alive=modality_alive,
    )
    expected_manifest_rows = (
        len(reference_sids) * len(mechanisms)
    )
    if len(manifest) != expected_manifest_rows:
        raise RuntimeError("CARE view manifest is incomplete")
    clean_digest_by_sid = {
        str(row["sid"]): str(row["output_digest"])
        for row in manifest
        if row["view_name"] == "clean"
    }
    for row in manifest:
        row["effective_change_from_clean"] = (
            False
            if row["view_name"] == "clean"
            else str(row["output_digest"])
            != clean_digest_by_sid[str(row["sid"])]
        )
    return cache, manifest


def _score_paths(
    model: CareDroidModel,
    path_logits: torch.Tensor,
    modality_alive: torch.Tensor,
    path_available: torch.Tensor,
) -> torch.Tensor:
    if model.care_risk_head is None:
        raise RuntimeError("CARE risk head is unavailable")
    reference_device = next(model.care_risk_head.parameters()).device
    flat_logits = path_logits.reshape(-1, len(CARE_PATHS), 2).to(
        reference_device
    )
    flat_alive = modality_alive.reshape(-1, 3).to(reference_device)
    flat_available = path_available.reshape(-1, len(CARE_PATHS)).to(
        reference_device
    )
    flat_odds = flat_logits[..., 1] - flat_logits[..., 0]
    with torch.no_grad():
        normalized = model.care_risk_head.normalize(
            flat_odds,
            flat_available,
        )
        score = model.care_risk_head.score_all(normalized, flat_alive)
        score = torch.where(
            flat_available,
            score,
            torch.zeros_like(score),
        )
    return score.cpu().reshape(*path_logits.shape[:-1])


def _route_cached(
    path_logits: torch.Tensor,
    path_available: torch.Tensor,
    correctness: torch.Tensor,
    routing_cfg: Mapping[str, Any],
) -> CareRoutedBatch:
    original_shape = path_logits.shape[:-2]
    if path_logits.shape[-2:] != (len(CARE_PATHS), 2):
        raise ValueError("CARE routing requires path_logits [..., 4, 2]")
    if path_available.shape != path_logits.shape[:-1]:
        raise ValueError("CARE path availability shape is invalid")
    if correctness.shape != path_logits.shape[:-1]:
        raise ValueError("CARE correctness score shape is invalid")
    flat_logits = path_logits.reshape(-1, len(CARE_PATHS), 2)
    flat_available = path_available.reshape(-1, len(CARE_PATHS)).bool()
    flat_correctness = correctness.reshape(-1, len(CARE_PATHS))
    paths = {
        name: flat_logits[:, index]
        for index, name in enumerate(CARE_PATHS)
    }
    enabled = routing_cfg.get("enabled", True)
    route_all = routing_cfg.get("route_on_all_samples", False)
    if not isinstance(enabled, bool) or not isinstance(route_all, bool):
        raise ValueError("CARE routing flags must be boolean")
    if enabled and not route_all:
        routed = route_with_agm_anchor(
            paths,
            flat_available,
            flat_correctness,
        )
        selected = routed.selected_path_index
        selected_logits = routed.selected_logits
        selected_score = routed.selected_score
        eligible = ~routed.reject
        disagreement = routed.disagreement_with_agm
    else:
        predictions = torch.stack(
            [hard_predict(paths[name]) for name in CARE_PATHS],
            dim=-1,
        )
        disagreement = predictions[:, 1:].ne(predictions[:, :1])
        all_three = flat_available[:, 0]
        pair_available = flat_available[:, 1:]
        exactly_two = (~all_three) & pair_available.sum(dim=-1).eq(1)
        selected = torch.full(
            (flat_logits.size(0),),
            -1,
            dtype=torch.long,
            device=flat_logits.device,
        )
        if not enabled:
            selected = torch.where(
                all_three,
                torch.zeros_like(selected),
                selected,
            )
            unique_pair = pair_available.long().argmax(dim=-1) + 1
            selected = torch.where(exactly_two, unique_pair, selected)
        else:
            all_candidate = flat_available & all_three.unsqueeze(-1)
            candidate_score = flat_correctness.masked_fill(
                ~all_candidate,
                torch.finfo(flat_correctness.dtype).min,
            )
            selected = torch.where(
                all_three,
                candidate_score.argmax(dim=-1),
                selected,
            )
            unique_pair = pair_available.long().argmax(dim=-1) + 1
            selected = torch.where(exactly_two, unique_pair, selected)
        eligible = selected.ge(0)
        safe = selected.clamp_min(0)
        row = torch.arange(flat_logits.size(0), device=flat_logits.device)
        selected_logits = flat_logits[row, safe]
        selected_score = flat_correctness[row, safe]
        selected_logits = torch.where(
            eligible.unsqueeze(-1),
            selected_logits,
            torch.zeros_like(selected_logits),
        )
        selected_score = torch.where(
            eligible,
            selected_score,
            torch.zeros_like(selected_score),
        )
    return CareRoutedBatch(
        selected_path_index=selected.reshape(*original_shape),
        selected_logits=selected_logits.reshape(*original_shape, 2),
        selected_score=selected_score.reshape(*original_shape),
        eligible=eligible.reshape(*original_shape),
        disagreement_with_agm=disagreement.reshape(*original_shape, 3),
    )


def _ece_binary(
    correctness_probability: torch.Tensor,
    correctness_target: torch.Tensor,
    valid: torch.Tensor,
    bins: int = 10,
) -> float:
    score = correctness_probability[valid].float()
    target = correctness_target[valid].float()
    if score.numel() == 0:
        return float("nan")
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    total = float(score.numel())
    ece = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (score >= lower) & (
            score <= upper if index == bins - 1 else score < upper
        )
        count = int(mask.sum().item())
        if count:
            ece += (
                count
                / total
                * abs(
                    float(score[mask].mean().item())
                    - float(target[mask].mean().item())
                )
            )
    return float(ece)


def _correctness_diagnostics(
    score: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    score = score[valid].float().cpu()
    target = target[valid].float().cpu()
    if score.numel() == 0:
        return {"num_samples": 0}
    error = 1.0 - target.numpy()
    error_score = 1.0 - score.numpy()
    result: dict[str, Any] = {
        "num_samples": int(score.numel()),
        "brier": float(torch.mean((score - target) ** 2).item()),
        "ece_10": _ece_binary(score, target, torch.ones_like(target).bool()),
        "mean_predicted_correctness": float(score.mean().item()),
        "empirical_accuracy": float(target.mean().item()),
    }
    if np.unique(error).size == 2:
        result["error_auroc"] = float(roc_auc_score(error, error_score))
        result["error_auprc"] = float(
            average_precision_score(error, error_score)
        )
    else:
        result["error_auroc"] = float("nan")
        result["error_auprc"] = float("nan")
    return result


def _oof_diagnostics(
    cache: CareRiskCalibrationCache,
    oof_score: torch.Tensor,
    fold_assignment: torch.Tensor,
    routing_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # The risk cache intentionally persists only binary log-odds.  Canonical
    # logits [0, g] preserve every hard prediction, disagreement, and softmax
    # probability while avoiding a second redundant logits tensor.
    diagnostic_path_logits = torch.stack(
        [
            torch.zeros_like(cache.path_log_odds),
            cache.path_log_odds,
        ],
        dim=-1,
    )
    routed = _route_cached(
        diagnostic_path_logits,
        cache.valid_paths,
        oof_score,
        routing_cfg,
    )
    path_prediction = hard_predict(
        diagnostic_path_logits.reshape(-1, 2)
    ).reshape(diagnostic_path_logits.shape[:-1])
    labels = cache.labels[:, None, None]
    correctness = path_prediction.eq(labels)
    selected_prediction = routed.prediction
    selected_correct = selected_prediction.eq(cache.labels[:, None])
    agm_correct = correctness[..., 0]
    eligible = routed.eligible
    # A unique pair chosen because one modality is absent is not a "switch
    # from AGM": AGM is invalid in that state.  Mixing those structural
    # fallbacks into repair/destruction rates compares the pair against AGM's
    # zero-logit placeholder and can manufacture both repairs and damage.
    agm_valid = cache.valid_paths[..., 0]
    switched = (
        eligible
        & agm_valid
        & routed.selected_path_index.ne(0)
    )
    structural_pair_selection = (
        eligible
        & (~agm_valid)
        & routed.selected_path_index.gt(0)
    )
    repair = switched & (~agm_correct) & selected_correct
    destruction = switched & agm_correct & (~selected_correct)
    available_correct = correctness & cache.valid_paths
    fallback_correct_when_agm_wrong = (
        (~agm_correct)
        & available_correct[..., 1:].any(dim=-1)
        & cache.valid_paths[..., 0]
    )
    all_valid_agm = cache.valid_paths[..., 0]
    oracle_denominator = int(all_valid_agm.sum().item())
    risk_by_path = {
        name: _correctness_diagnostics(
            oof_score[..., index],
            cache.correctness_targets[..., index],
            cache.valid_paths[..., index],
        )
        for index, name in enumerate(CARE_PATHS)
    }
    selected_target = torch.where(
        eligible,
        selected_correct.float(),
        torch.zeros_like(routed.selected_score),
    )
    selected_risk = _correctness_diagnostics(
        routed.selected_score,
        selected_target,
        eligible,
    )
    agm_probability = torch.softmax(
        diagnostic_path_logits[..., 0, :],
        dim=-1,
    )
    agm_msp = agm_probability.max(dim=-1).values
    msp_diagnostic = _correctness_diagnostics(
        agm_msp,
        cache.correctness_targets[..., 0],
        cache.valid_paths[..., 0],
    )
    summary = {
        "path_correctness": risk_by_path,
        "selected_path_correctness": selected_risk,
        "agm_msp_reference": msp_diagnostic,
        "oracle_path_diversity": {
            "num_all_paths_valid": oracle_denominator,
            "agm_wrong_fallback_correct_count": int(
                fallback_correct_when_agm_wrong.sum().item()
            ),
            "agm_wrong_fallback_correct_rate_over_all_valid": (
                float(fallback_correct_when_agm_wrong.sum().item())
                / float(max(oracle_denominator, 1))
            ),
        },
        "routing_switch": {
            "eligible_count": int(eligible.sum().item()),
            "switch_count": int(switched.sum().item()),
            "structural_pair_selection_count": int(
                structural_pair_selection.sum().item()
            ),
            "repair_count": int(repair.sum().item()),
            "destruction_count": int(destruction.sum().item()),
            "repair_rate_given_switch": float(repair.sum().item())
            / float(max(int(switched.sum().item()), 1)),
            "destruction_rate_given_switch": float(destruction.sum().item())
            / float(max(int(switched.sum().item()), 1)),
        },
    }
    rows: list[dict[str, Any]] = []
    for sid_index, sid in enumerate(cache.sids):
        for view_index, view_name in enumerate(cache.view_names):
            selected_index = int(
                routed.selected_path_index[sid_index, view_index].item()
            )
            row: dict[str, Any] = {
                "sid": sid,
                "group": cache.groups[sid_index],
                "label": int(cache.labels[sid_index].item()),
                "view_name": view_name,
                "fold": int(fold_assignment[sid_index].item()),
                "selected_path": (
                    CARE_PATHS[selected_index]
                    if selected_index >= 0
                    else "reject"
                ),
                "selected_score": float(
                    routed.selected_score[sid_index, view_index].item()
                ),
                "selected_prediction": int(
                    selected_prediction[sid_index, view_index].item()
                ),
                "eligible": bool(eligible[sid_index, view_index].item()),
                "switched_from_agm": bool(
                    switched[sid_index, view_index].item()
                ),
                "structural_pair_selection": bool(
                    structural_pair_selection[
                        sid_index, view_index
                    ].item()
                ),
                "repair": bool(repair[sid_index, view_index].item()),
                "destruction": bool(
                    destruction[sid_index, view_index].item()
                ),
                "relative_advantage_vs_agm": (
                    float(
                        routed.selected_score[sid_index, view_index].item()
                        - oof_score[sid_index, view_index, 0].item()
                    )
                    if (
                        selected_index >= 0
                        and bool(
                            agm_valid[sid_index, view_index].item()
                        )
                    )
                    else float("nan")
                ),
            }
            for path_index, path_name in enumerate(CARE_PATHS):
                row[f"g_{path_name}"] = float(
                    cache.path_log_odds[
                        sid_index, view_index, path_index
                    ].item()
                )
                row[f"q_{path_name}"] = float(
                    oof_score[sid_index, view_index, path_index].item()
                )
                row[f"valid_{path_name}"] = bool(
                    cache.valid_paths[
                        sid_index, view_index, path_index
                    ].item()
                )
                row[f"correct_{path_name}"] = bool(
                    cache.correctness_targets[
                        sid_index, view_index, path_index
                    ].item()
                )
            rows.append(row)
    return summary, rows


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: _json_compatible(row.get(key))
                        for key in fieldnames
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cache_natural_view(
    model: CareDroidModel,
    subset: Subset,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    split_name: str,
) -> CareCachedView:
    base, indices = _subset_base_and_indices(subset)
    if bool(getattr(base, "is_train", False)):
        raise RuntimeError(
            f"CARE {split_name} must use a non-training natural dataset"
        )
    if getattr(base, "eval_perturb_plan", None) is not None:
        raise RuntimeError(
            f"CARE {split_name} must contain only "
            f"{list(CARE_DECISION_VIEW_MECHANISMS)}; a controlled-view "
            "perturbation plan was attached"
        )
    sample_ids = [
        str(base.samples[index][2]).strip().lower() for index in indices
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(
            f"CARE {split_name} contains duplicate SIDs; natural "
            "decision calibration requires exactly one row per source"
        )
    loader = build_loader(
        cfg,
        subset,
        is_train=False,
        seed_namespace=f"care/{split_name}",
        persistent_workers_override=False,
    )
    return _cache_care_view(
        model,
        loader,
        device,
        cfg,
        split_name=split_name,
    )


def _routed_from_cached(
    model: CareDroidModel,
    cached: CareCachedView,
    routing_cfg: Mapping[str, Any],
) -> tuple[CareRoutedBatch, torch.Tensor]:
    score = _score_paths(
        model,
        cached.path_logits,
        cached.modality_alive,
        cached.path_available,
    )
    routed = _route_cached(
        cached.path_logits,
        cached.path_available,
        score,
        routing_cfg,
    )
    return routed, score


def _selection_score(
    routed: CareRoutedBatch,
    score_type: str,
) -> torch.Tensor:
    if score_type == "care_selected_path_correctness":
        return routed.selected_score
    if score_type != "msp":
        raise ValueError(f"Unsupported CARE decision score {score_type!r}")
    probability = torch.softmax(routed.selected_logits.float(), dim=-1)
    return probability.max(dim=-1).values


def _fit_care_crc(
    cached: CareCachedView,
    routed: CareRoutedBatch,
    decision_score: torch.Tensor,
    selective_cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    labels = cached.labels.bool()
    prediction = routed.prediction
    predicted_benign = routed.eligible & prediction.eq(0)
    false_negative = labels & predicted_benign
    n_malware = int(labels.sum().item())
    if n_malware == 0:
        raise RuntimeError(
            "CARE decision_cal contains no malware; "
            "crc_status=failure_no_malware and no threshold is generated"
        )
    minimum = int(selective_cfg["min_calibration_malware"])
    if n_malware < minimum:
        raise RuntimeError(
            "CARE decision_cal has insufficient malware samples: "
            f"observed={n_malware} required={minimum}"
        )
    crc = fit_atomic_crc_correctness_threshold(
        decision_score.float().cpu(),
        false_negative.bool().cpu(),
        labels.bool().cpu(),
        alpha=float(selective_cfg["risk_level"]),
        eligible=predicted_benign.bool().cpu(),
    )
    if crc.crc_status == "failure_no_malware":
        raise AssertionError(
            "CARE zero-malware precondition was not enforced before CRC fit"
        )
    if (
        not crc.feasible
        and bool(selective_cfg.get("require_feasible", True))
    ):
        raise RuntimeError(
            "CARE CRC is infeasible: "
            f"crc_status={crc.crc_status} N_malware={crc.n_malware} "
            f"alpha={crc.alpha}"
        )
    accepted = (
        routed.eligible
        & (
            prediction.eq(1)
            | (
                predicted_benign
                & crc.accepted.to(predicted_benign.device)
            )
        )
    )
    accepted_fn = accepted & labels & prediction.eq(0)
    decision_summary = {
        **crc.as_summary(),
        "population": "natural_decision_cal_one_row_per_sid",
        "score_name": str(selective_cfg["threshold_score"]),
        "malware_predictions_are_always_accepted": True,
        "benign_acceptance_rule": "q >= lambda",
        "structural_reject_rule": "fewer_than_two_modalities_alive",
        "overall_accepted_count": int(accepted.sum().item()),
        "overall_coverage": float(accepted.float().mean().item()),
        "overall_reject_count": int((~accepted).sum().item()),
        "accepted_fn_count_audit": int(accepted_fn.sum().item()),
        "corrected_risk_audit": float(
            (accepted_fn.sum().item() + 1)
            / float(n_malware + 1)
        ),
        "guarantee_type": "expected_conformal_risk_control",
        "guarantee_scope": "natural_distribution_only",
    }
    rows: list[dict[str, Any]] = []
    for index, sid in enumerate(cached.sids):
        selected_index = int(routed.selected_path_index[index].item())
        rows.append(
            {
                "sid": sid,
                "group": cached.groups[index],
                "label": int(cached.labels[index].item()),
                "selected_path": (
                    CARE_PATHS[selected_index]
                    if selected_index >= 0
                    else "reject"
                ),
                "prediction": int(prediction[index].item()),
                "decision_score": float(decision_score[index].item()),
                "predicted_benign": bool(predicted_benign[index].item()),
                "accepted": bool(accepted[index].item()),
                "accepted_fn": bool(accepted_fn[index].item()),
            }
        )
    return crc, decision_summary, rows


def _risk_coverage_curve(
    cached: CareCachedView,
    routed: CareRoutedBatch,
    decision_score: torch.Tensor,
) -> list[dict[str, Any]]:
    labels = cached.labels.bool()
    prediction = routed.prediction
    predicted_benign = routed.eligible & prediction.eq(0)
    malware_predictions = routed.eligible & prediction.eq(1)
    candidates = sorted(
        {
            float(value)
            for value in decision_score[predicted_benign].tolist()
        },
        reverse=True,
    )
    thresholds = [float("inf"), *candidates, float("-inf")]
    n_malware = int(labels.sum().item())
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = malware_predictions | (
            predicted_benign & decision_score.ge(threshold)
        )
        accepted_fn = accepted & labels & prediction.eq(0)
        rows.append(
            {
                "lambda": threshold,
                "accepted_count": int(accepted.sum().item()),
                "coverage": float(accepted.float().mean().item()),
                "accepted_fn_count": int(accepted_fn.sum().item()),
                "empirical_malware_fn_risk": (
                    float(accepted_fn.sum().item()) / float(max(n_malware, 1))
                ),
                "corrected_malware_fn_risk": (
                    float(accepted_fn.sum().item() + 1)
                    / float(max(n_malware + 1, 1))
                ),
            }
        )
    return rows


def _selective_metrics(
    cached: CareCachedView,
    routed: CareRoutedBatch,
    decision_score: torch.Tensor,
    lambda_threshold: float,
    *,
    guarantee_scope: str,
) -> dict[str, Any]:
    labels = cached.labels
    prediction = routed.prediction
    predicted_malware = routed.eligible & prediction.eq(1)
    predicted_benign = routed.eligible & prediction.eq(0)
    accepted = predicted_malware | (
        predicted_benign & decision_score.ge(float(lambda_threshold))
    )
    accepted_labels = labels[accepted]
    accepted_prediction = prediction[accepted]
    n_malware = int(labels.eq(1).sum().item())
    accepted_fn = accepted & labels.eq(1) & prediction.eq(0)
    result: dict[str, Any] = {
        "num_samples": int(labels.numel()),
        "structural_reject_count": int((~routed.eligible).sum().item()),
        "accepted_count": int(accepted.sum().item()),
        "rejected_count": int((~accepted).sum().item()),
        "coverage": float(accepted.float().mean().item()),
        "malware_count": n_malware,
        "accepted_fn_count": int(accepted_fn.sum().item()),
        "empirical_malware_accepted_fn_risk": float(
            accepted_fn.sum().item()
        )
        / float(max(n_malware, 1)),
        "corrected_malware_accepted_fn_risk": float(
            accepted_fn.sum().item() + 1
        )
        / float(max(n_malware + 1, 1)),
        "threshold_lambda": float(lambda_threshold),
        "guarantee_scope": guarantee_scope,
    }
    if accepted_labels.numel() > 0:
        result["accepted_accuracy"] = float(
            accepted_prediction.eq(accepted_labels).float().mean().item()
        )
        result["accepted_macro_f1"] = float(
            f1_score(
                accepted_labels.numpy(),
                accepted_prediction.numpy(),
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        )
    else:
        result["accepted_accuracy"] = float("nan")
        result["accepted_macro_f1"] = float("nan")
    return result


def _evaluate_care_cached(
    model: CareDroidModel,
    cached: CareCachedView,
    routing_cfg: Mapping[str, Any],
    *,
    score_type: str,
    lambda_threshold: float,
    guarantee_scope: str,
) -> dict[str, Any]:
    routed, path_scores = _routed_from_cached(
        model,
        cached,
        routing_cfg,
    )
    decision_score = _selection_score(routed, score_type)
    raw_mask = routed.eligible
    raw_metrics = (
        _binary_metrics_from_logits(
            cached.labels[raw_mask],
            routed.selected_logits[raw_mask],
        )
        if bool(raw_mask.any().item())
        else {"num_samples": 0}
    )
    path_metrics: dict[str, Any] = {}
    for index, name in enumerate(CARE_PATHS):
        valid = cached.path_available[:, index]
        path_metrics[name] = (
            _binary_metrics_from_logits(
                cached.labels[valid],
                cached.path_logits[valid, index],
            )
            if bool(valid.any().item())
            else {"num_samples": 0}
        )
    return {
        "raw_selected_classification": raw_metrics,
        "path_classification": path_metrics,
        "selective": _selective_metrics(
            cached,
            routed,
            decision_score,
            lambda_threshold,
            guarantee_scope=guarantee_scope,
        ),
        "routing": {
            "selected_path_counts": {
                (
                    CARE_PATHS[index] if index >= 0 else "reject"
                ): int(routed.selected_path_index.eq(index).sum().item())
                for index in (-1, 0, 1, 2, 3)
            },
            "mean_selected_score": float(
                routed.selected_score[routed.eligible].mean().item()
            )
            if bool(routed.eligible.any().item())
            else float("nan"),
        },
        "path_score_diagnostics": {
            name: _correctness_diagnostics(
                path_scores[:, index],
                hard_predict(cached.path_logits[:, index])
                .eq(cached.labels)
                .float(),
                cached.path_available[:, index],
            )
            for index, name in enumerate(CARE_PATHS)
        },
    }


def _test_view_registry(cfg: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    eval_cfg = cfg.get("eval", {}) or {}
    configured = tuple(eval_cfg.get("perturb_tests", ()))
    strengths = tuple(float(value) for value in eval_cfg.get(
        "perturb_strengths", (0.1, 0.3, 0.5, 0.7, 0.9)
    ))
    registry: list[tuple[str, float]] = []
    for mechanism in configured:
        mechanism = str(mechanism).strip().lower()
        if mechanism == "clean":
            registry.append(("clean", 0.0))
        elif mechanism in GRADED_PERTURBATIONS:
            registry.extend((mechanism, strength) for strength in strengths)
        elif mechanism in MISSING_PERTURBATIONS:
            registry.append((mechanism, 1.0))
        else:
            raise ValueError(
                f"Unregistered CARE evaluation mechanism {mechanism!r}"
            )
    if not registry or registry[0] != ("clean", 0.0):
        raise ValueError("CARE evaluation registry must begin with clean")
    if len(registry) != len(set(registry)):
        raise ValueError("CARE evaluation registry contains duplicate cells")
    return tuple(registry)


def _make_fixed_deterministic_test_view(
    test_dataset: RobustTriModalDataset,
    *,
    mechanism: str,
    strength: float,
    care_cfg: Mapping[str, Any],
) -> tuple[RobustTriModalDataset, tuple[dict[str, Any], ...]]:
    """Build one fixed-strength test cell with the frozen CARE seed rule."""

    view = copy.copy(test_dataset)
    view.is_train = False
    # Produce an input digest for every test cell so repeated model seeds can
    # audit that their controlled inputs, rather than only their labels, match.
    view.care_digest_view = True
    protocol_seed = int(care_cfg["views"]["protocol_seed"])
    plan, records = fixed_test_view_plan(
        view.sample_sids,
        mechanism=str(mechanism),
        strength=float(strength),
        protocol_seed=protocol_seed,
    )
    view.eval_perturb_plan = (
        None if mechanism == "clean" else tuple(plan)
    )
    return view, records


def _evaluate_test_suite(
    model: CareDroidModel,
    test_dataset: RobustTriModalDataset,
    cfg: dict[str, Any],
    care_cfg: dict[str, Any],
    device: torch.device,
    *,
    lambda_threshold: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mechanism, strength in _test_view_registry(cfg):
        view, view_records = _make_fixed_deterministic_test_view(
            test_dataset,
            mechanism=mechanism,
            strength=float(strength),
            care_cfg=care_cfg,
        )
        loader = build_loader(
            cfg,
            view,
            is_train=False,
            seed_namespace=f"care/test/{mechanism}/{strength:.6f}",
            persistent_workers_override=False,
        )
        cached = _cache_care_view(
            model,
            loader,
            device,
            cfg,
            split_name=f"care_test_{mechanism}_{strength:.3f}",
        )
        key = (
            "clean"
            if mechanism == "clean"
            else (
                mechanism
                if mechanism in MISSING_PERTURBATIONS
                else f"{mechanism}@{strength:.1f}"
            )
        )
        evaluated = _evaluate_care_cached(
            model,
            cached,
            care_cfg["routing"],
            score_type=care_cfg["selective"]["threshold_score"],
            lambda_threshold=lambda_threshold,
            guarantee_scope=(
                "natural_test_expected_crc"
                if mechanism == "clean"
                else "empirical_only_distribution_shift_no_crc_guarantee"
            ),
        )
        output_payload = [
            {
                "sid": sid,
                "output_digest": digest,
            }
            for sid, digest in zip(
                cached.sids,
                cached.output_digests,
                strict=True,
            )
        ]
        if any(not row["output_digest"] for row in output_payload):
            raise RuntimeError(
                f"CARE test view {key!r} omitted one or more input digests"
            )
        evaluated["controlled_view_audit"] = {
            "seed_formula": CONTROLLED_VIEW_SEED_FORMULA,
            "protocol_seed": int(care_cfg["views"]["protocol_seed"]),
            "mechanism": mechanism,
            "mechanism_version": CONTROLLED_VIEW_MECHANISM_VERSION,
            "strength": float(strength),
            "num_samples": len(output_payload),
            "seed_manifest_sha256": seed_manifest_sha256(view_records),
            "output_manifest_sha256": canonical_manifest_sha256(
                output_payload
            ),
        }
        results[key] = evaluated
    return results


def _oof_path_rows(
    cache: CareRiskCalibrationCache,
    oof_score: torch.Tensor,
    fold_assignment: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid_index, sid in enumerate(cache.sids):
        for view_index, view_name in enumerate(cache.view_names):
            for path_index, path_name in enumerate(CARE_PATHS):
                rows.append(
                    {
                        "sid": sid,
                        "group": cache.groups[sid_index],
                        "label": int(cache.labels[sid_index].item()),
                        "view_name": view_name,
                        "fold": int(fold_assignment[sid_index].item()),
                        "path": path_name,
                        "valid": bool(
                            cache.valid_paths[
                                sid_index, view_index, path_index
                            ].item()
                        ),
                        "log_odds": float(
                            cache.path_log_odds[
                                sid_index, view_index, path_index
                            ].item()
                        ),
                        "correctness_target": float(
                            cache.correctness_targets[
                                sid_index, view_index, path_index
                            ].item()
                        ),
                        "oof_correctness_score": float(
                            oof_score[
                                sid_index, view_index, path_index
                            ].item()
                        ),
                    }
                )
    return rows


def _prepare_data_and_roles(
    cfg: dict[str, Any],
    care_cfg: dict[str, Any],
) -> tuple[
    RobustTriModalDataset,
    RobustTriModalDataset,
    RobustTriModalDataset,
    CareRoleDatasets,
    dict[str, Any],
]:
    validate_split_partitions(cfg, include_test=True)
    manifest_provenance = validate_manifest_vocab_provenance(cfg)
    data_cfg = cfg["data"]
    if bool(manifest_provenance.get("verified", False)):
        data_cfg["expected_manifest_vocab_sha256"] = manifest_provenance[
            "manifest_vocab_sha256"
        ]
        data_cfg["expected_manifest_train_csv_sha256"] = manifest_provenance[
            "train_csv_sha256"
        ]
        data_cfg[
            "expected_manifest_train_sample_ids_sha256"
        ] = manifest_provenance["train_sample_ids_sha256"]
    train_dataset = build_dataset(cfg, "train", is_train=True)
    val_dataset = build_dataset(cfg, "val", is_train=False)
    test_dataset = build_dataset(cfg, "test", is_train=False)
    feature_dims = {
        int(train_dataset.feature_dim),
        int(val_dataset.feature_dim),
        int(test_dataset.feature_dim),
    }
    if len(feature_dims) != 1:
        raise ValueError(
            f"CARE splits disagree on graph feature dimension: {feature_dims}"
        )
    roles = split_care_roles(
        cfg,
        care_cfg,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        roles,
        manifest_provenance,
    )


def _run_care_eval_only(
    cfg: dict[str, Any],
    care_cfg: dict[str, Any],
    model: CareDroidModel,
    roles: CareRoleDatasets,
    test_dataset: RobustTriModalDataset,
    manifest_provenance: dict[str, Any],
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    eval_cfg = cfg.get("eval", {}) or {}
    if eval_cfg.get("refit_decision_calibration") is not True:
        raise ValueError(
            "CARE routing/score ablations must refit their own natural-only "
            "decision CRC on the fixed decision_cal role"
        )
    checkpoint_path = _resolve_eval_checkpoint_path(cfg)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"CARE source pipeline not found: {checkpoint_path}"
        )
    source = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        source.get("schema_version") != CARE_PIPELINE_SCHEMA_VERSION
        or
        source.get("artifact_type") != "care_droid_pipeline"
        or source.get("protocol_id") != CARE_PROTOCOL_ID
    ):
        raise ValueError(
            "CARE eval_only source is not a current-schema care_droid_v1 "
            "pipeline matching the current artifact identity"
        )
    source_training_seed = _validate_source_training_seed(source, cfg)
    current_protocol_sha256 = _care_protocol_sha256(care_cfg)
    current_upstream_protocol_sha256 = _care_upstream_protocol_sha256(
        care_cfg
    )
    if (
        source.get("upstream_protocol_sha256")
        != current_upstream_protocol_sha256
    ):
        raise ValueError(
            "CARE eval_only source uses different experts, roles, views, or "
            "path-risk fitting protocol"
        )
    current_upstream_runtime = _care_upstream_runtime_payload(cfg)
    current_upstream_runtime_sha256 = _care_upstream_runtime_sha256(cfg)
    if (
        source.get("upstream_runtime") != current_upstream_runtime
        or source.get("upstream_runtime_sha256")
        != current_upstream_runtime_sha256
    ):
        raise ValueError(
            "CARE eval_only source uses different encoder, input-budget, or "
            "numeric inference semantics"
        )
    current_source_code_sha256 = _care_source_code_sha256()
    if source.get("source_code_sha256") != current_source_code_sha256:
        raise ValueError(
            "CARE eval_only source was produced by a different implementation; "
            "retrain the primary pipeline before running ablations"
        )
    current_data_lineage = _care_data_lineage_payload(
        cfg,
        manifest_provenance,
    )
    current_data_lineage_sha256 = hashlib.sha256(
        json.dumps(
            current_data_lineage,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        source.get("data_lineage") != current_data_lineage
        or source.get("data_lineage_sha256")
        != current_data_lineage_sha256
    ):
        raise ValueError(
            "CARE eval_only source was produced from a different PT build or "
            "Manifest vocabulary lineage"
        )
    model_state = source.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("CARE source pipeline omits model_state")
    if _state_dict_sha256(model_state) != source.get(
        "model_state_sha256"
    ):
        raise ValueError("CARE source pipeline state digest is invalid")
    current_role_identity = _care_role_identity_sha256(
        roles.summary["public"]
    )
    if source.get("role_identity_sha256") != current_role_identity:
        raise ValueError(
            "CARE eval_only source was fitted on a different immutable "
            "expert/routing/decision/test role population"
        )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.set_care_risk_active(True)
    stage_hash = _state_dict_sha256(
        _cpu_state_dict(model.care_stage_a_state_dict())
    )
    if stage_hash != source.get("stage_a_state_sha256"):
        raise ValueError("CARE source pipeline Stage-A identity is invalid")

    decision_cached = _cache_natural_view(
        model,
        roles.decision_cal,
        cfg,
        device,
        split_name="decision_cal_natural_eval_only",
    )
    decision_routed, _path_scores = _routed_from_cached(
        model,
        decision_cached,
        care_cfg["routing"],
    )
    decision_score = _selection_score(
        decision_routed,
        care_cfg["selective"]["threshold_score"],
    )
    crc, decision_summary, decision_rows = _fit_care_crc(
        decision_cached,
        decision_routed,
        decision_score,
        care_cfg["selective"],
    )
    _write_csv_rows(
        out_dir / "care_decision_calibration.csv",
        decision_rows,
    )
    _write_csv_rows(
        out_dir / "risk_coverage_curve.csv",
        _risk_coverage_curve(
            decision_cached,
            decision_routed,
            decision_score,
        ),
    )
    derived = {
        **{
            key: value
            for key, value in source.items()
            if key
            not in {
                "decision_calibration",
                "lambda_threshold",
                "score_name",
                "cfg",
            }
        },
        "decision_calibration": decision_summary,
        "lambda_threshold": float(crc.lambda_threshold),
        "score_name": care_cfg["selective"]["threshold_score"],
        "method_protocol_sha256": current_protocol_sha256,
        "upstream_protocol_sha256": current_upstream_protocol_sha256,
        "upstream_runtime": current_upstream_runtime,
        "upstream_runtime_sha256": current_upstream_runtime_sha256,
        "source_method_protocol_sha256": source.get(
            "method_protocol_sha256"
        ),
        "source_code_sha256": current_source_code_sha256,
        "data_lineage": current_data_lineage,
        "data_lineage_sha256": current_data_lineage_sha256,
        "cfg": _json_compatible(cfg),
        "source_pipeline_path": str(checkpoint_path.resolve()),
        "source_pipeline_sha256": _file_sha256(checkpoint_path),
    }
    _atomic_torch_save(derived, out_dir / "best_care_pipeline.pt")
    test_results = _evaluate_test_suite(
        model,
        test_dataset,
        cfg,
        care_cfg,
        device,
        lambda_threshold=float(crc.lambda_threshold),
    )
    summary = {
        "summary_schema_version": CARE_SUMMARY_SCHEMA_VERSION,
        "experiment_name": str(eval_cfg.get("output_name") or ""),
        "method_name": "care_droid",
        "method_protocol_id": CARE_PROTOCOL_ID,
        "method_protocol_sha256": current_protocol_sha256,
        "upstream_protocol_sha256": current_upstream_protocol_sha256,
        "upstream_runtime_sha256": current_upstream_runtime_sha256,
        "method_implementation_sha256": current_source_code_sha256,
        "data_lineage_sha256": current_data_lineage_sha256,
        "seed": int((cfg.get("train", {}) or {}).get("seed", 42)),
        "training_seed": int(source_training_seed),
        "eval_only": True,
        "source_pipeline": str(checkpoint_path.resolve()),
        "source_pipeline_sha256": _file_sha256(checkpoint_path),
        "routing": care_cfg["routing"],
        "decision_score": care_cfg["selective"]["threshold_score"],
        "data_roles": roles.summary["public"],
        "role_identity_sha256": current_role_identity,
        "manifest_vocab_provenance": manifest_provenance,
        "decision_calibration": decision_summary,
        "test": test_results,
        "audit": {
            "upstream_stage_a_reused_without_mutation": True,
            "upstream_risk_head_reused_without_refit": True,
            "decision_crc_refitted_on_fixed_natural_role": True,
            "classification_threshold_fitted": False,
        },
    }
    _write_yaml(out_dir / "summary.yaml", summary)
    return summary


def run(
    cfg: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    care_cfg = validate_care_config(cfg)
    eval_only = _strict_bool(
        (cfg.get("eval", {}) or {}).get("eval_only", False),
        name="eval.eval_only",
    )
    _preflight_care_output_target(cfg, overwrite=overwrite)
    logging.basicConfig(
        level=getattr(
            logging,
            str(cfg.get("log_level", "INFO")).upper(),
            logging.INFO,
        )
    )
    train_cfg = cfg.get("train", {}) or {}
    seed = _strict_int(train_cfg.get("seed", 42), name="train.seed")
    set_seed(seed)
    configure_determinism(
        _strict_bool(
            train_cfg.get("deterministic", True),
            name="train.deterministic",
        ),
        strict=_strict_bool(
            train_cfg.get("strict_deterministic", False),
            name="train.strict_deterministic",
        ),
    )
    configure_multiprocessing_sharing(cfg)
    device = select_device(str(train_cfg.get("device", "auto")))
    logger.info(
        "CARE-Droid protocol=%s seed=%d device=%s paths=%s",
        CARE_PROTOCOL_ID,
        seed,
        device,
        CARE_PATHS,
    )
    (
        train_dataset,
        val_dataset,
        test_dataset,
        roles,
        manifest_provenance,
    ) = _prepare_data_and_roles(cfg, care_cfg)
    role_summary = roles.summary["public"]
    # Output mutation starts only after every data/role/provenance preflight.
    out_dir = _prepare_care_output_directory(cfg, overwrite=overwrite)
    _write_yaml(out_dir / "resolved_config.yaml", cfg)

    model = build_care_model(
        cfg,
        int(train_dataset.feature_dim),
        care_cfg,
    ).to(device)
    # Decouple Stage-A stochasticity from architecture initialization.
    set_seed(seed)
    if eval_only:
        return _run_care_eval_only(
            cfg,
            care_cfg,
            model,
            roles,
            test_dataset,
            manifest_provenance,
            device,
            out_dir,
        )
    stage_summary = _fit_stage_a(
        model,
        roles,
        cfg,
        care_cfg,
        device,
        out_dir,
        role_summary,
    )
    stage_hash_before_risk = _state_dict_sha256(
        _cpu_state_dict(model.care_stage_a_state_dict())
    )
    routing_cache, view_manifest = _cache_routing_views(
        model,
        roles.routing_cal,
        cfg,
        care_cfg,
        device,
    )
    view_manifest_payload = {
        "schema_version": 1,
        "protocol_id": CARE_PROTOCOL_ID,
        "protocol_seed": int(care_cfg["protocol_seed"]),
        "view_identity_formula": (
            "H(SID, mechanism, protocol_seed)"
        ),
        "metadata_is_model_input": False,
        "num_rows": len(view_manifest),
        "rows": view_manifest,
    }
    _write_json(out_dir / "care_view_manifest.json", view_manifest_payload)
    view_manifest_sha256 = _file_sha256(
        out_dir / "care_view_manifest.json"
    )

    risk_cfg = care_cfg["risk_training"]
    # Stage A deliberately froze the risk head. Re-establish its exact
    # training partition: encoders/path heads remain immutable and only the
    # shared path-correctness head is trainable.
    _activate_care_risk_training_partition(model)
    risk_result = fit_care_risk_crossfit(
        model,
        routing_cache,
        device=device,
        folds=int(risk_cfg["folds"]),
        epochs=int(risk_cfg["epochs"]),
        batch_size=int(risk_cfg["batch_size"]),
        learning_rate=float(risk_cfg["lr"]),
        weight_decay=float(risk_cfg["weight_decay"]),
        gradient_clip=float(risk_cfg["grad_clip"]),
        protocol_seed=int(care_cfg["protocol_seed"]),
    )
    stage_hash_after_risk = _state_dict_sha256(
        _cpu_state_dict(model.care_stage_a_state_dict())
    )
    if stage_hash_after_risk != stage_hash_before_risk:
        raise RuntimeError(
            "CARE risk fitting mutated Stage-A encoders/path heads"
        )
    model.set_care_risk_active(True)
    oof_summary, route_rows = _oof_diagnostics(
        routing_cache,
        risk_result.oof_correctness_probability,
        risk_result.fold_assignment,
        care_cfg["routing"],
    )
    oof_predictions_path = out_dir / "care_oof_predictions.csv"
    route_diagnostics_path = out_dir / "care_route_diagnostics.csv"
    _write_csv_rows(
        oof_predictions_path,
        _oof_path_rows(
            routing_cache,
            risk_result.oof_correctness_probability,
            risk_result.fold_assignment,
        ),
    )
    _write_csv_rows(
        route_diagnostics_path,
        route_rows,
    )

    decision_cached = _cache_natural_view(
        model,
        roles.decision_cal,
        cfg,
        device,
        split_name="decision_cal_natural",
    )
    decision_routed, _decision_path_scores = _routed_from_cached(
        model,
        decision_cached,
        care_cfg["routing"],
    )
    decision_score = _selection_score(
        decision_routed,
        care_cfg["selective"]["threshold_score"],
    )
    crc, decision_summary, decision_rows = _fit_care_crc(
        decision_cached,
        decision_routed,
        decision_score,
        care_cfg["selective"],
    )
    _write_csv_rows(
        out_dir / "care_decision_calibration.csv",
        decision_rows,
    )
    _write_csv_rows(
        out_dir / "risk_coverage_curve.csv",
        _risk_coverage_curve(
            decision_cached,
            decision_routed,
            decision_score,
        ),
    )

    method_protocol_sha256 = _care_protocol_sha256(care_cfg)
    upstream_protocol_sha256 = _care_upstream_protocol_sha256(care_cfg)
    upstream_runtime = _care_upstream_runtime_payload(cfg)
    upstream_runtime_sha256 = _care_upstream_runtime_sha256(cfg)
    implementation_source_sha256 = _care_source_code_sha256()
    data_lineage = _care_data_lineage_payload(
        cfg,
        manifest_provenance,
    )
    data_lineage_sha256 = _care_data_lineage_sha256(
        cfg,
        manifest_provenance,
    )
    pipeline_state = _cpu_state_dict(model.state_dict())
    pipeline_checkpoint = {
        "schema_version": CARE_PIPELINE_SCHEMA_VERSION,
        "artifact_type": "care_droid_pipeline",
        "protocol_id": CARE_PROTOCOL_ID,
        "training_seed": int(seed),
        "method_protocol_sha256": method_protocol_sha256,
        "upstream_protocol_sha256": upstream_protocol_sha256,
        "upstream_runtime": upstream_runtime,
        "upstream_runtime_sha256": upstream_runtime_sha256,
        "hard_prediction_rule": CARE_HARD_PREDICT_RULE,
        "stage_a_state_sha256": stage_hash_after_risk,
        "stage_a_checkpoint_sha256": _file_sha256(
            out_dir / "best_care_stage_a.pt"
        ),
        "model_state": pipeline_state,
        "model_state_sha256": _state_dict_sha256(pipeline_state),
        "risk_crossfit": risk_result.summary,
        "risk_fold_state_dicts": tuple(
            {
                key: value.detach().cpu().clone()
                for key, value in state.items()
            }
            for state in risk_result.fold_state_dicts
        ),
        "risk_fold_assignment": (
            risk_result.fold_assignment.detach().cpu().clone()
        ),
        "risk_oof_correctness_probability": (
            risk_result.oof_correctness_probability.detach().cpu().clone()
        ),
        "risk_oof_artifacts": {
            "predictions_csv_sha256": _file_sha256(oof_predictions_path),
            "route_diagnostics_csv_sha256": _file_sha256(
                route_diagnostics_path
            ),
        },
        "decision_calibration": decision_summary,
        "lambda_threshold": float(crc.lambda_threshold),
        "score_name": care_cfg["selective"]["threshold_score"],
        "role_summary": role_summary,
        "role_identity_sha256": _care_role_identity_sha256(role_summary),
        "view_manifest_sha256": view_manifest_sha256,
        "source_code_sha256": implementation_source_sha256,
        "data_lineage": data_lineage,
        "data_lineage_sha256": data_lineage_sha256,
        "cfg": _json_compatible(cfg),
    }
    pipeline_path = out_dir / "best_care_pipeline.pt"
    _atomic_torch_save(pipeline_checkpoint, pipeline_path)
    reloaded = torch.load(
        pipeline_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        reloaded.get("schema_version") != CARE_PIPELINE_SCHEMA_VERSION
        or reloaded.get("training_seed") != seed
        or reloaded.get("method_protocol_sha256")
        != method_protocol_sha256
        or reloaded.get("upstream_protocol_sha256")
        != upstream_protocol_sha256
        or reloaded.get("upstream_runtime") != upstream_runtime
        or reloaded.get("upstream_runtime_sha256")
        != upstream_runtime_sha256
        or reloaded.get("source_code_sha256")
        != implementation_source_sha256
        or reloaded.get("data_lineage") != data_lineage
        or reloaded.get("data_lineage_sha256")
        != data_lineage_sha256
    ):
        raise RuntimeError(
            "CARE pipeline checkpoint failed protocol/code/data lineage audit"
        )
    if _state_dict_sha256(reloaded["model_state"]) != reloaded[
        "model_state_sha256"
    ]:
        raise RuntimeError("CARE pipeline checkpoint failed state hash audit")
    reloaded_fold_states = tuple(reloaded.get("risk_fold_state_dicts", ()))
    reloaded_fold_rows = tuple(
        (reloaded.get("risk_crossfit", {}) or {}).get(
            "folds_summary", ()
        )
    )
    if (
        len(reloaded_fold_states) != 3
        or len(reloaded_fold_rows) != 3
    ):
        raise RuntimeError(
            "CARE pipeline checkpoint omitted one or more OOF fold artifacts"
        )
    for fold_state, fold_row in zip(
        reloaded_fold_states,
        reloaded_fold_rows,
        strict=True,
    ):
        if tensor_state_dict_sha256(fold_state) != fold_row.get(
            "state_dict_sha256"
        ):
            raise RuntimeError(
                "CARE pipeline checkpoint failed OOF fold-state hash audit"
            )
    reloaded_assignment = reloaded.get("risk_fold_assignment")
    reloaded_oof = reloaded.get("risk_oof_correctness_probability")
    if (
        not isinstance(reloaded_assignment, torch.Tensor)
        or reloaded_assignment.shape
        != risk_result.fold_assignment.shape
        or not torch.equal(
            reloaded_assignment,
            risk_result.fold_assignment.cpu(),
        )
        or not isinstance(reloaded_oof, torch.Tensor)
        or reloaded_oof.shape
        != risk_result.oof_correctness_probability.shape
        or not torch.equal(
            reloaded_oof,
            risk_result.oof_correctness_probability.cpu(),
        )
    ):
        raise RuntimeError(
            "CARE pipeline checkpoint failed OOF prediction/assignment audit"
        )

    test_results = _evaluate_test_suite(
        model,
        test_dataset,
        cfg,
        care_cfg,
        device,
        lambda_threshold=float(crc.lambda_threshold),
    )
    summary: dict[str, Any] = {
        "summary_schema_version": CARE_SUMMARY_SCHEMA_VERSION,
        "experiment_name": str(
            (cfg.get("train", {}) or {}).get(
                "exp_name",
                "care_droid",
            )
        ),
        "method_name": "care_droid",
        "method_protocol_id": CARE_PROTOCOL_ID,
        "method_protocol_sha256": method_protocol_sha256,
        "upstream_protocol_sha256": upstream_protocol_sha256,
        "upstream_runtime_sha256": upstream_runtime_sha256,
        "method_implementation_sha256": implementation_source_sha256,
        "data_lineage": data_lineage,
        "data_lineage_sha256": data_lineage_sha256,
        "seed": seed,
        "training_seed": seed,
        "algorithm_frozen": True,
        "pipeline": (
            "clean_four_path_experts -> three_fold_group_oof_path_"
            "correctness -> disagreement_aware_agm_anchored_routing -> "
            "natural_distribution_accepted_fn_crc"
        ),
        "paths": list(CARE_PATHS),
        "data_roles": role_summary,
        "role_identity_sha256": _care_role_identity_sha256(role_summary),
        "manifest_vocab_provenance": manifest_provenance,
        "stage_a": stage_summary,
        "risk_crossfit": risk_result.summary,
        "oof_diagnostics": oof_summary,
        "decision_calibration": decision_summary,
        "test": test_results,
        "artifacts": {
            "stage_a": str(
                (out_dir / "best_care_stage_a.pt").resolve()
            ),
            "pipeline": str(pipeline_path.resolve()),
            "view_manifest": str(
                (out_dir / "care_view_manifest.json").resolve()
            ),
            "pipeline_sha256": _file_sha256(pipeline_path),
            "stage_a_state_sha256": stage_hash_after_risk,
            "view_manifest_sha256": view_manifest_sha256,
            "oof_predictions": str(oof_predictions_path.resolve()),
            "oof_predictions_sha256": _file_sha256(
                oof_predictions_path
            ),
            "route_diagnostics": str(route_diagnostics_path.resolve()),
            "route_diagnostics_sha256": _file_sha256(
                route_diagnostics_path
            ),
        },
        "audit": {
            "stage_a_unchanged_by_risk_fit": True,
            "oof_fold_count": int(risk_cfg["folds"]),
            "oof_fixed_epochs": int(risk_cfg["epochs"]),
            "oof_holdout_used_for_selection": False,
            "view_metadata_used_as_model_input": False,
            "classification_threshold_fitted": False,
            "crc_guarantee_scope": "natural_test_only",
        },
    }
    _write_yaml(out_dir / "summary.yaml", summary)
    logger.info(
        "CARE-Droid complete clean_macro_f1=%.6f coverage=%.6f "
        "corrected_fn_risk=%.6f output=%s",
        float(
            test_results["clean"]["raw_selected_classification"].get(
                "macro_f1",
                float("nan"),
            )
        ),
        float(test_results["clean"]["selective"]["coverage"]),
        float(
            test_results["clean"]["selective"][
                "corrected_malware_accepted_fn_risk"
            ]
        ),
        out_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the frozen CARE-Droid protocol"
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="CARE root config followed by optional explicit overlays",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known CARE artifacts in the output directory",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
