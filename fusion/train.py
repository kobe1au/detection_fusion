from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fusion.losses import (
    compute_probability_calibration_loss,
    compute_posthoc_calibration_loss,
    compute_reliability_calibration_loss,
    compute_robust_loss,
)
from fusion.dataset import (
    RobustTriModalDataset,
    prepare_robust_batch,
    robust_collate_fn,
)
from fusion.model import TriModalRobustModel
from fusion.perturbations import EVAL_PERTURB_TYPES
from fusion.reliability_calibration import BRANCH_NAMES
from fusion.utils import build_grad_scaler, get_amp_context
from fusion.constants import EvidenceIndex, TriModalConfigDefaults


logger = logging.getLogger("tri_modal_robust")

DEFAULT_ROBUST_VAL_SCENARIOS = (
    {"name": "api_graph_degraded_s0.5", "perturb_type": "api_graph_degraded", "strength": 0.5, "weight": 0.25},
    {"name": "manifest_degraded_s0.5", "perturb_type": "manifest_degraded", "strength": 0.5, "weight": 0.15},
    {"name": "all_degraded_s0.5", "perturb_type": "all_degraded", "strength": 0.5, "weight": 0.10},
    {"name": "api_missing", "perturb_type": "api_missing", "strength": 1.0, "weight": 1.0 / 30.0},
    {"name": "graph_missing", "perturb_type": "graph_missing", "strength": 1.0, "weight": 1.0 / 30.0},
    {"name": "manifest_missing", "perturb_type": "manifest_missing", "strength": 1.0, "weight": 1.0 / 30.0},
)


BRANCH_EVAL_LOGIT_KEYS = {
    "api": "api_logits_aux",
    "graph": "graph_logits_aux",
    "manifest": "manifest_logits_aux",
    "joint": "joint_logits_aux",
}


class EmptyExtraEvalSetError(RuntimeError):
    """Raised when an optional external eval set has no usable samples."""


CHECKPOINT_STAGE_ENCODER_SELECTED = "encoder_selected"
CHECKPOINT_STAGE_PIPELINE_FITTED = "pipeline_fitted"
CHECKPOINT_STAGES = frozenset(
    {
        CHECKPOINT_STAGE_ENCODER_SELECTED,
        CHECKPOINT_STAGE_PIPELINE_FITTED,
    }
)


GATE_DIAGNOSTIC_KEYS = (
    "api_integrity",
    "api_encoder_coverage",
    "api_total_pipeline_coverage",
    "api_extractor_coverage",
    "api_runtime_encoder_coverage",
    "effective_api_integrity",
    "api_truncated_by_extractor_budget",
    "api_truncated_by_encoder_budget",
    "api_integrity_before_encoder_budget",
    "graph_integrity",
    "graph_encoder_coverage",
    "graph_feature_valid_ratio",
    "effective_graph_integrity",
    "graph_truncated_by_encoder_budget",
    "graph_integrity_before_encoder_budget",
    "manifest_integrity",
    "effective_manifest_integrity",
    "code_integrity",
    "api_graph_anchor_support",
    "manifest_code_support",
    "manifest_to_code_conflict",
    "code_to_manifest_conflict",
    "api_alive",
    "graph_alive",
    "manifest_alive",
    # Compatibility aliases and model-confidence diagnostics.
    "q_api",
    "q_graph",
    "q_manifest",
    "q_align",
    "pert_api",
    "pert_graph",
    "pert_manifest",
    "r_api",
    "r_graph",
    "r_manifest",
    "api_manifest_consistency",
    "graph_manifest_consistency",
    "api_confidence",
    "graph_confidence",
    "manifest_confidence",
    "joint_confidence",
    "discount_api",
    "discount_graph",
    "discount_manifest",
    "discount_joint",
    "fusion_weight_api",
    "fusion_weight_graph",
    "fusion_weight_manifest",
    "fusion_weight_joint",
    "qmf_energy_api",
    "qmf_energy_graph",
    "qmf_energy_manifest",
    "entropy_api",
    "entropy_graph",
    "entropy_manifest",
    "entropy_joint",
    "margin_api",
    "margin_graph",
    "margin_manifest",
    "margin_joint",
    "uncertainty_proxy_api",
    "uncertainty_proxy_graph",
    "uncertainty_proxy_manifest",
    "uncertainty_proxy_joint",
    "zero_weight_fallback_used",
    "predicted_reliability_api",
    "predicted_reliability_graph",
    "predicted_reliability_manifest",
    "predicted_reliability_joint",
    "branch_competence_active",
    "branch_competence_prior_api",
    "branch_competence_prior_graph",
    "branch_competence_prior_manifest",
    "branch_competence_prior_joint",
    "visible_integrity_modifier_active",
    "visible_integrity_modifier_beta",
    "visible_integrity_modifier_min_value",
    "visible_modifier_api",
    "visible_modifier_graph",
    "visible_modifier_manifest",
    "visible_modifier_factor_api",
    "visible_modifier_factor_graph",
    "visible_modifier_factor_manifest",
    "visible_integrity_reference_api",
    "visible_integrity_reference_graph",
    "visible_integrity_reference_manifest",
    "weight_sharpening_gamma",
    "temperature_api",
    "temperature_graph",
    "temperature_manifest",
    "temperature_joint",
    "effective_manifest_to_code_conflict",
    "effective_code_to_manifest_conflict",
    "api_graph_support_applicable",
    "manifest_code_conflict_applicable",
    "manifest_code_relation_applicable",
    "api_manifest_relation_applicable",
    "graph_manifest_relation_applicable",
    "total_reliability",
    "final_uncertainty_proxy",
    "effective_conflict",
    "raw_conflict",
    "predictive_conflict",
    "predictive_conflict_max",
    "routing_active",
    "routing_weight_api",
    "routing_weight_graph",
    "routing_weight_manifest",
    "routing_weight_joint",
    "routing_weight_unknown",
    "routing_known_mass",
    "routing_prior_known_mass",
    "routing_prior_unknown_mass",
    "routing_reliability_prior_known_mass",
    "routing_known_retention",
    "routing_mean_disagreement",
    "routing_disagreement_feature_active",
    "routing_common_scale_reliability_active",
    "routing_mode_learned",
    "routing_mode_prior_only",
    "routing_mode_known_only",
    "routing_train_end_to_end",
    "routing_posthoc_refine",
    "final_temperature",
    "acceptance_score",
    "acceptance_score_unknown_only",
    "acceptance_score_fused_certainty",
    "acceptance_score_conflict_only",
    "acceptance_score_product",
    "calibration_active",
    "gate_uses_perturbation_evidence",
    "explicit_relation_factors_active",
    "joint_conflict_factor",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_determinism(enabled: bool, strict: bool = False) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    if enabled and strict and torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(enabled, warn_only=not strict)


def configure_multiprocessing_sharing(cfg: dict) -> None:
    train_cfg = cfg.get("train", {}) or {}
    strategy = str(train_cfg.get("multiprocessing_sharing_strategy", "") or "").strip()
    if not strategy or strategy.lower() in {"default", "none", "false"}:
        return
    mp = torch.multiprocessing
    try:
        available = set(mp.get_all_sharing_strategies())
    except (AttributeError, RuntimeError):
        available = set()
    if available and strategy not in available:
        logger.warning(
            "train.multiprocessing_sharing_strategy=%s is unavailable; available=%s",
            strategy,
            sorted(available),
        )
        return
    try:
        mp.set_sharing_strategy(strategy)
        logger.info("torch_multiprocessing_sharing_strategy=%s", strategy)
    except (AttributeError, RuntimeError) as exc:
        logger.warning("Unable to set torch multiprocessing sharing strategy %s: %s", strategy, exc)

def select_device(value: str) -> torch.device:
    value = str(value or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device=cuda requested but CUDA is unavailable")
    return torch.device(value)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _should_apply_tri_modal_defaults(path: Path, cfg: dict) -> bool:
    if path.name.startswith("_"):
        return False
    try:
        normalized = path.resolve().as_posix()
    except OSError:
        normalized = path.as_posix()
    return "config/experiments/tri_modal_robust" in normalized and bool(cfg)


def _apply_tri_modal_defaults(path: Path, cfg: dict) -> dict:
    if not _should_apply_tri_modal_defaults(path, cfg):
        return cfg
    return deep_update(TriModalConfigDefaults.CONFIG, cfg)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_path(path: str | Path, seen: set[Path] | None = None) -> dict:
    path = Path(path)
    seen = set(seen or ())
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Recursive config defaults detected: {path}")
    seen.add(resolved)
    raw = load_yaml(path)
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    cfg: dict[str, Any] = {}
    for item in defaults:
        item_path = Path(item)
        if not item_path.is_absolute():
            item_path = path.parent / item_path
        cfg = deep_update(cfg, load_config_path(item_path, seen))
    return _apply_tri_modal_defaults(path, deep_update(cfg, raw))


def load_config(paths: list[str]) -> dict:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_update(cfg, load_config_path(path))
    return cfg


def resolve(root: str | Path, path: str | Path) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    return str(Path(root) / path)


def _read_split_identities(
    csv_path: str | Path,
    expected_split: str,
) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    packages: set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        id_col = next((name for name in ("id", "ID", "Id", "sha256") if name in fields), None)
        pkg_col = next((name for name in ("pkg_name", "package_name", "package") if name in fields), None)
        if id_col is None:
            raise ValueError(f"CSV {csv_path} must contain id or sha256")
        for row_idx, row in enumerate(reader, start=2):
            sid = str(row.get(id_col, "") or "").strip().lower()
            if not sid:
                raise ValueError(f"CSV {csv_path} has empty {id_col} at row {row_idx}")
            ids.add(sid)
            if "split" in fields:
                split_value = str(row.get("split", "") or "").strip().lower()
                if split_value and split_value != expected_split:
                    raise ValueError(
                        f"CSV {csv_path} row {row_idx} declares split={split_value!r}, "
                        f"expected {expected_split!r}"
                    )
            if pkg_col is not None:
                package = str(row.get(pkg_col, "") or "").strip().lower()
                if package and package not in {"nan", "none", "null"}:
                    packages.add(package)
    return ids, packages


def validate_split_partitions(cfg: dict, include_test: bool) -> None:
    data_cfg = cfg.get("data", {})
    if not bool(data_cfg.get("strict_partition_isolation", True)):
        return
    data_root = data_cfg.get("root", "")
    split_names = ["train", "val"] + (["test"] if include_test else [])
    identities: dict[str, tuple[set[str], set[str]]] = {}
    for split in split_names:
        csv_path = resolve(data_root, data_cfg[f"{split}_csv"])
        identities[split] = _read_split_identities(csv_path, split)

    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            id_overlap = sorted(identities[left][0] & identities[right][0])
            pkg_overlap = sorted(identities[left][1] & identities[right][1])
            if id_overlap or pkg_overlap:
                raise ValueError(
                    f"Split leakage between {left} and {right}: "
                    f"id_overlap={len(id_overlap)} examples={id_overlap[:10]}; "
                    f"package_overlap={len(pkg_overlap)} examples={pkg_overlap[:10]}"
                )


def _checkpoint_semantic_signature(cfg: dict) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    fusion_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
    # Acceptance aggregation changes rejection ranking only, not model state.
    fusion_cfg.pop("acceptance_aggregation", None)
    return {
        "model": copy.deepcopy(cfg.get("model", {}) or {}),
        "fusion": fusion_cfg,
        "calibration": copy.deepcopy(cfg.get("calibration", {}) or {}),
        "data": {
            key: copy.deepcopy(data_cfg.get(key))
            for key in (
                "graph_semantic_source",
                "max_api_events_per_sample",
                "label_map",
            )
        },
    }


def validate_eval_checkpoint_config(
    current_cfg: dict,
    checkpoint_cfg: Any,
    *,
    allow_mismatch: bool = False,
) -> None:
    if allow_mismatch:
        return
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(
            "Evaluation checkpoint does not contain its training config. "
            "Set eval.allow_checkpoint_config_mismatch=true only for an explicitly audited legacy checkpoint."
        )
    current = _checkpoint_semantic_signature(current_cfg)
    saved = _checkpoint_semantic_signature(checkpoint_cfg)
    if current != saved:
        raise ValueError(
            "Evaluation config changes model/data semantics relative to the checkpoint. "
            "Use the checkpoint's training config and override only eval paths/settings, or set "
            "eval.allow_checkpoint_config_mismatch=true for an explicitly labelled compatibility audit."
        )


def validate_checkpoint_implementation(
    checkpoint: dict[str, Any],
    *,
    allow_mismatch: bool = False,
) -> None:
    """Reject checkpoints produced by a different decision implementation."""
    if allow_mismatch:
        return
    saved = str(checkpoint.get("method_implementation_sha256", ""))
    current = _method_implementation_sha256()
    if not saved:
        raise ValueError(
            "Evaluation checkpoint predates implementation fingerprinting. "
            "Retrain it with the current code, or set "
            "eval.allow_checkpoint_config_mismatch=true only for an explicitly "
            "labelled compatibility audit."
        )
    if saved != current:
        raise ValueError(
            "Evaluation checkpoint was produced by a different model/fusion "
            "implementation. Retrain it with the current code, or set "
            "eval.allow_checkpoint_config_mismatch=true only for an explicitly "
            "labelled compatibility audit."
        )


def validate_checkpoint_stage(
    checkpoint: dict[str, Any],
    *,
    expected: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> str:
    """Validate the explicit lifecycle stage attached to a checkpoint."""
    location = f" {checkpoint_path}" if checkpoint_path is not None else ""
    stage = str(checkpoint.get("checkpoint_stage", "")).strip().lower()
    if stage not in CHECKPOINT_STAGES:
        raise ValueError(
            f"Checkpoint{location} is missing a valid checkpoint_stage; "
            f"expected one of {sorted(CHECKPOINT_STAGES)}. Retrain the model "
            "with staged checkpointing instead of reusing an ambiguous checkpoint."
        )
    if expected is not None and stage != expected:
        if expected not in CHECKPOINT_STAGES:
            raise ValueError(f"Unknown expected checkpoint stage: {expected!r}")
        raise ValueError(
            f"Checkpoint{location} has checkpoint_stage={stage!r}, "
            f"but this operation requires {expected!r}."
        )
    return stage


def _load_eval_checkpoint(
    checkpoint_path: str | Path,
    *,
    refit_posthoc_calibration: bool,
    map_location: Any,
) -> tuple[Path, dict[str, Any]]:
    """Load the only checkpoint stage valid for the requested eval lifecycle.

    Ordinary evaluation consumes a pipeline-fitted artifact. A post-hoc refit
    consumes the encoder-selected artifact so that the jointly trained router
    is preserved while none of the previous post-hoc labels are inherited.
    A pipeline checkpoint may link to that artifact for config compatibility.
    """
    requested_path = Path(checkpoint_path)
    requested_checkpoint = torch.load(
        requested_path, map_location=map_location, weights_only=True
    )
    requested_stage = validate_checkpoint_stage(
        requested_checkpoint, checkpoint_path=requested_path
    )
    if not refit_posthoc_calibration:
        validate_checkpoint_stage(
            requested_checkpoint,
            expected=CHECKPOINT_STAGE_PIPELINE_FITTED,
            checkpoint_path=requested_path,
        )
        return requested_path, requested_checkpoint

    if requested_stage == CHECKPOINT_STAGE_ENCODER_SELECTED:
        return requested_path, requested_checkpoint

    encoder_reference = requested_checkpoint.get("encoder_checkpoint_path")
    if not str(encoder_reference or "").strip():
        raise ValueError(
            f"Pipeline checkpoint {requested_path} does not link to its "
            "encoder-selected checkpoint; post-hoc refitting cannot safely "
            "reconstruct the training lifecycle."
        )
    encoder_path = Path(str(encoder_reference))
    if not encoder_path.is_absolute():
        encoder_path = requested_path.parent / encoder_path
    if not encoder_path.exists():
        raise FileNotFoundError(
            "Encoder-selected checkpoint linked by the pipeline artifact was "
            f"not found: {encoder_path}"
        )
    expected_encoder_sha256 = str(
        requested_checkpoint.get("encoder_checkpoint_sha256", "")
    ).strip().lower()
    if expected_encoder_sha256:
        actual_encoder_sha256 = _file_sha256(encoder_path)
        if actual_encoder_sha256 != expected_encoder_sha256:
            raise ValueError(
                "Encoder-selected checkpoint hash does not match the pipeline "
                f"artifact: {encoder_path}"
            )
    encoder_checkpoint = torch.load(
        encoder_path, map_location=map_location, weights_only=True
    )
    validate_checkpoint_stage(
        encoder_checkpoint,
        expected=CHECKPOINT_STAGE_ENCODER_SELECTED,
        checkpoint_path=encoder_path,
    )
    return encoder_path, encoder_checkpoint


def _resolve_refit_decision_calibration(eval_cfg: dict | None) -> bool:
    """Resolve the decision-calibration refit flag and its legacy alias."""
    eval_cfg = eval_cfg or {}
    current_key = "refit_decision_calibration"
    legacy_key = "refit_rejection_threshold"
    has_current = current_key in eval_cfg
    has_legacy = legacy_key in eval_cfg
    current = bool(eval_cfg.get(current_key, False))
    legacy = bool(eval_cfg.get(legacy_key, False))
    if has_current and has_legacy and current != legacy:
        raise ValueError(
            f"eval.{current_key} and legacy eval.{legacy_key} disagree; "
            "set only the new key or give both the same value."
        )
    return current if has_current else legacy


def _decision_calibration_signature(cfg: dict) -> dict[str, Any]:
    """Canonicalize every setting that changes fitted decision artifacts."""
    classification_cfg = cfg.get("classification_threshold", {}) or {}
    classification_enabled = bool(classification_cfg.get("enabled", False))
    classification = {
        "enabled": classification_enabled,
        "objective": (
            str(classification_cfg.get("objective", "macro_f1")).lower()
            if classification_enabled
            else "disabled"
        ),
        "min_malware_recall": (
            float(classification_cfg.get("min_malware_recall", 0.0))
            if classification_enabled
            else None
        ),
    }

    selective_cfg = cfg.get("selective_prediction", {}) or {}
    selective_enabled = bool(selective_cfg.get("enabled", False))
    mode = (
        str(selective_cfg.get("mode", "threshold")).lower()
        if selective_enabled
        else "disabled"
    )
    selective: dict[str, Any] = {
        "enabled": selective_enabled,
        "mode": mode,
        "score_type": (
            _selective_score_type(selective_cfg)
            if selective_enabled
            else "disabled"
        ),
    }
    if selective_enabled and selective["score_type"] == "model_acceptance":
        fusion_cfg = cfg.get("fusion", {}) or {}
        if str(fusion_cfg.get("combination", "linear")).lower() == "routed":
            routing_cfg = fusion_cfg.get("routing", {}) or {}
            score_definition = str(
                routing_cfg.get("acceptance_score_mode", "product")
            ).lower()
            if score_definition == "current_product":
                score_definition = "product"
            selective["model_acceptance_definition"] = (
                f"routed:{score_definition}"
            )
        else:
            selective["model_acceptance_definition"] = "fusion:" + str(
                fusion_cfg.get("acceptance_aggregation", "product")
            ).lower()
    if selective_enabled and mode == "risk_control":
        risk_target = str(
            selective_cfg.get(
                "risk_target", "malware_fn_rate_after_rejection"
            )
        ).lower()
        if risk_target == "accepted_malware_false_negative":
            risk_target = "malware_fn_rate_after_rejection"
        selective.update(
            {
                "risk_level": float(selective_cfg.get("risk_level", 0.05)),
                "risk_target": risk_target,
                "min_calibration_malware": int(
                    selective_cfg.get("min_calibration_malware", 1)
                ),
                "require_feasible": bool(
                    selective_cfg.get("require_feasible", False)
                ),
            }
        )
    elif selective_enabled and mode == "conformal":
        target_coverage = float(
            selective_cfg.get("target_coverage", 0.9)
        )
        selective.update(
            {
                "alpha": float(
                    selective_cfg.get("alpha", 1.0 - target_coverage)
                ),
                "class_conditional": bool(
                    selective_cfg.get("class_conditional", True)
                ),
                "use_raw_conflict": bool(
                    selective_cfg.get("use_raw_conflict", False)
                ),
            }
        )
    elif selective_enabled:
        selective["target_coverage"] = float(
            selective_cfg.get("target_coverage", 0.9)
        )
    return {"classification": classification, "selective": selective}


def validate_checkpoint_decision_signature(
    current_cfg: dict,
    checkpoint: dict[str, Any],
    *,
    refit_decision_calibration: bool,
) -> None:
    """Prevent fitted thresholds from being reused under different semantics."""
    if refit_decision_calibration:
        return
    current = _decision_calibration_signature(current_cfg)
    classification_is_used = bool(
        current["classification"]["enabled"]
        or current["selective"]["enabled"]
    )
    selective_is_used = bool(current["selective"]["enabled"])
    if not classification_is_used and not selective_is_used:
        return
    saved = checkpoint.get("decision_calibration_signature")
    if not isinstance(saved, dict):
        raise ValueError(
            "Pipeline checkpoint has no decision_calibration_signature; "
            "retrain it or set eval.refit_decision_calibration=true"
        )
    if (
        classification_is_used
        and current["classification"] != saved.get("classification")
    ):
        raise ValueError(
            "Classification-threshold settings differ from the fitted "
            "checkpoint artifact; set eval.refit_decision_calibration=true"
        )
    if selective_is_used and current["selective"] != saved.get("selective"):
        raise ValueError(
            "Selective-decision settings differ from the fitted checkpoint "
            "artifact; set eval.refit_decision_calibration=true"
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_common_kwargs(
    cfg: dict,
    is_train: bool,
    perturb_type: str | None = None,
    perturb_strength: float = 0.0,
) -> dict[str, Any]:
    data_cfg = cfg["data"]
    allowed_data_keys = {
        "root",
        "train_pt_dir",
        "val_pt_dir",
        "test_pt_dir",
        "train_csv",
        "val_csv",
        "test_csv",
        "out_dir",
        "max_failed_ratio",
        "max_api_events_per_sample",
        "graph_semantic_source",
        "strict_split_integrity",
        "strict_partition_isolation",
        "allow_pt_superset",
        "label_map",
    }
    unknown_data_keys = sorted(set(data_cfg) - allowed_data_keys)
    if unknown_data_keys:
        raise ValueError(f"Unsupported data settings: {unknown_data_keys}")
    robust_cfg = cfg.get("robust", {})
    model_cfg = cfg.get("model", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    return {
        "is_train": is_train,
        "robust_aug": bool(robust_cfg.get("train_aug", False)) if is_train else False,
        "perturb_prob": float(robust_cfg.get("perturb_prob", 0.5)),
        "perturb_strengths": list(robust_cfg.get("perturb_strengths", [0.1, 0.3, 0.5])),
        "eval_perturb_type": perturb_type,
        "eval_perturb_strength": perturb_strength,
        "manifest_dim": int(manifest_cfg.get("in_dim", 256)),
        "manifest_category_dim": int(manifest_cfg.get("category_dim", 12)),
        "manifest_stats_dim": int(manifest_cfg.get("stats_dim", 11)),
        "manifest_permission_dim": int(manifest_cfg.get("permission_dim", 128)),
        "manifest_intent_dim": int(manifest_cfg.get("intent_dim", 64)),
        "manifest_feature_dim": int(manifest_cfg.get("feature_dim", 32)),
        "max_api_events_per_sample": data_cfg.get("max_api_events_per_sample"),
        "max_graph_nodes_per_sample": (
            int(model_cfg.get("max_nodes_gnn", 12288))
            if bool(model_cfg.get("graph_encoder", {}).get("account_for_encoder_budget", True))
            else None
        ),
        "drop_graph_behavior_hints": bool(model_cfg.get("graph_encoder", {}).get("drop_extracted_behavior_hints", False)),
        "graph_semantic_source": str(data_cfg.get("graph_semantic_source", "alignment")),
        "num_classes": int(model_cfg.get("num_classes", 2)),
        "label_map": data_cfg.get("label_map"),
        "strict_split_integrity": bool(data_cfg.get("strict_split_integrity", True)),
        "allow_pt_superset": bool(data_cfg.get("allow_pt_superset", False)),
    }


def build_dataset_from_paths(
    cfg: dict,
    pt_dir: str | Path,
    csv_path: str | Path,
    is_train: bool,
    perturb_type: str | None = None,
    perturb_strength: float = 0.0,
    dataset_overrides: dict[str, Any] | None = None,
):
    kwargs = _dataset_common_kwargs(
        cfg,
        is_train=is_train,
        perturb_type=perturb_type,
        perturb_strength=perturb_strength,
    )
    kwargs.update(dataset_overrides or {})
    try:
        return RobustTriModalDataset(
            pt_dir=str(pt_dir),
            csv_path=str(csv_path),
            **kwargs,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "No matching .pt samples found" in msg:
            raise EmptyExtraEvalSetError(msg) from exc
        raise


def build_dataset(cfg: dict, split: str, is_train: bool, perturb_type: str | None = None, perturb_strength: float = 0.0):
    data_cfg = cfg["data"]
    data_root = data_cfg.get("root", "")
    pt_dir = resolve(data_root, data_cfg[f"{split}_pt_dir"])
    csv_path = resolve(data_root, data_cfg[f"{split}_csv"])
    return build_dataset_from_paths(
        cfg,
        pt_dir,
        csv_path,
        is_train=is_train,
        perturb_type=perturb_type,
        perturb_strength=perturb_strength,
    )


def build_loader(
    cfg: dict,
    dataset,
    is_train: bool,
    *,
    persistent_workers_override: bool | None = None,
):
    train_cfg = cfg["train"]
    worker_key = "num_workers" if is_train else "eval_num_workers"
    workers = int(train_cfg.get(worker_key, train_cfg.get("num_workers", 0)))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    if pin_memory and not bool(train_cfg.get("allow_pyg_pin_memory", False)):
        logger.warning(
            "train.pin_memory=true is unsafe for PyG Data/Batch on some CUDA runtimes; "
            "forcing pin_memory=false. Set train.allow_pyg_pin_memory=true to override."
        )
        pin_memory = False
    persistent_workers = bool(train_cfg.get("persistent_workers", False))
    if persistent_workers_override is not None:
        persistent_workers = bool(persistent_workers_override)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(train_cfg.get("batch_size" if is_train else "eval_batch_size", train_cfg.get("batch_size", 32))),
        "shuffle": is_train,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and workers > 0,
        "collate_fn": robust_collate_fn,
    }
    if workers > 0 and train_cfg.get("prefetch_factor") is not None:
        prefetch_factor = int(train_cfg.get("prefetch_factor"))
        if prefetch_factor <= 0:
            raise ValueError("train.prefetch_factor must be positive when num_workers > 0")
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def split_validation_dataset(cfg: dict, dataset) -> tuple[Subset, Subset, dict[str, Any]]:
    """Deterministically separate checkpoint selection from calibration.

    Package/sample groups remain intact while a deterministic greedy assignment
    keeps both halves close to the full validation year-label distribution.
    """
    calibration_cfg = cfg.get("calibration", {}) or {}
    fraction = float(calibration_cfg.get("validation_fraction", 0.5))
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration.validation_fraction must be within (0, 1)")
    size = len(dataset)
    if size < 2:
        raise ValueError("Validation dataset needs at least two samples for selection/calibration split")
    seed = int(calibration_cfg.get("split_seed", cfg.get("train", {}).get("seed", 42)))
    sids = list(getattr(dataset, "sample_sids", []))
    if len(sids) != size:
        sids = [str(index) for index in range(size)]
    groups = list(getattr(dataset, "sample_groups", []))
    if len(groups) != size:
        groups = list(sids)
    labels = list(getattr(dataset, "sample_labels", []))
    if len(labels) != size:
        samples = list(getattr(dataset, "samples", []))
        if len(samples) == size:
            labels = [int(sample[1]) for sample in samples]
        else:
            raise ValueError(
                "Validation dataset must expose sample_labels for stratified calibration split"
            )
    labels = [int(label) for label in labels]
    years = list(getattr(dataset, "sample_years", []))
    if len(years) != size:
        # Synthetic/legacy datasets without year metadata retain the previous
        # label-stratified behavior through a single sentinel year.
        years = [0] * size
    years = [int(year) for year in years]

    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(str(group), []).append(index)
    if len(group_to_indices) < 2:
        raise ValueError(
            "Validation dataset needs at least two package/sample groups for leakage-free calibration split"
        )

    label_values = sorted(set(labels))
    year_values = sorted(set(years))
    strata = list(zip(years, labels))
    stratum_values = sorted(set(strata))
    total_label_counts = {
        label: sum(int(value == label) for value in labels) for label in label_values
    }
    total_year_counts = {
        year: sum(int(value == year) for value in years) for year in year_values
    }
    total_stratum_counts = {
        stratum: sum(int(value == stratum) for value in strata)
        for stratum in stratum_values
    }
    target_stratum_counts = {
        stratum: float(total_stratum_counts[stratum]) * fraction
        for stratum in stratum_values
    }
    target_calibration_size = min(size - 1, max(1, int(round(size * fraction))))
    ranked_groups = sorted(
        group_to_indices,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    rank = {group: index for index, group in enumerate(ranked_groups)}
    group_stratum_counts: dict[str, dict[tuple[int, int], int]] = {}
    for group, indices in group_to_indices.items():
        group_stratum_counts[group] = {
            stratum: sum(int(strata[index] == stratum) for index in indices)
            for stratum in stratum_values
        }

    def split_error(
        candidate_size: int,
        candidate_counts: dict[tuple[int, int], int],
    ) -> float:
        size_error = abs(candidate_size - target_calibration_size) / max(size, 1)
        stratum_error = sum(
            abs(candidate_counts[stratum] - target_stratum_counts[stratum])
            / max(total_stratum_counts[stratum], 1)
            for stratum in stratum_values
        )
        return size_error + stratum_error

    remaining = set(ranked_groups)
    calibration_groups: list[str] = []
    calibration_indices: list[int] = []
    calibration_stratum_counts = {stratum: 0 for stratum in stratum_values}
    while remaining:
        current_size = len(calibration_indices)
        current_error = split_error(current_size, calibration_stratum_counts)
        candidates = []
        for group in remaining:
            indices = group_to_indices[group]
            new_size = current_size + len(indices)
            if new_size >= size:
                continue
            new_counts = {
                stratum: calibration_stratum_counts[stratum]
                + group_stratum_counts[group][stratum]
                for stratum in stratum_values
            }
            candidates.append(
                (split_error(new_size, new_counts), rank[group], group, new_counts)
            )
        if not candidates:
            break
        best_error, _, best_group, best_counts = min(candidates)
        if current_size >= target_calibration_size and best_error >= current_error:
            break
        calibration_groups.append(best_group)
        calibration_indices.extend(group_to_indices[best_group])
        calibration_stratum_counts = best_counts
        remaining.remove(best_group)

    calibration_indices = sorted(calibration_indices)
    if not calibration_indices or len(calibration_indices) >= size:
        raise ValueError("Unable to build non-empty stratified validation subsets")
    calibration_set = set(calibration_indices)
    selection_indices = [index for index in range(size) if index not in calibration_set]
    selection_groups = sorted(set(groups[index] for index in selection_indices))
    selection_label_counts = {
        label: sum(int(labels[index] == label) for index in selection_indices)
        for label in label_values
    }
    calibration_label_counts = {
        label: sum(int(labels[index] == label) for index in calibration_indices)
        for label in label_values
    }
    selection_year_counts = {
        year: sum(int(years[index] == year) for index in selection_indices)
        for year in year_values
    }
    calibration_year_counts = {
        year: sum(int(years[index] == year) for index in calibration_indices)
        for year in year_values
    }
    selection_year_label_counts = {
        f"{year}:{label}": sum(
            int(years[index] == year and labels[index] == label)
            for index in selection_indices
        )
        for year, label in stratum_values
    }
    calibration_year_label_counts = {
        f"{year}:{label}": sum(
            int(years[index] == year and labels[index] == label)
            for index in calibration_indices
        )
        for year, label in stratum_values
    }
    return (
        Subset(dataset, selection_indices),
        Subset(dataset, calibration_indices),
        {
            "split_seed": seed,
            "validation_fraction": fraction,
            "num_selection": len(selection_indices),
            "num_calibration": len(calibration_indices),
            "num_selection_groups": len(selection_groups),
            "num_calibration_groups": len(calibration_groups),
            "selection_label_counts": selection_label_counts,
            "calibration_label_counts": calibration_label_counts,
            "selection_year_counts": selection_year_counts,
            "calibration_year_counts": calibration_year_counts,
            "selection_year_label_counts": selection_year_label_counts,
            "calibration_year_label_counts": calibration_year_label_counts,
            "selection_indices": selection_indices,
            "calibration_indices": calibration_indices,
        },
    )


def split_posthoc_conformal_dataset(
    cfg: dict,
    dataset,
    holdout_indices: list[int],
) -> tuple[Subset, Subset, dict[str, Any]]:
    """Split validation holdout into model-fitting and decision-calibration subsets.

    The final subset must remain untouched by reliability calibration,
    visibility-reference fitting, routing, and classification-threshold
    selection. Reusing it in those label-dependent steps would invalidate the
    held-out calibration argument used by conformal and risk-control rules.
    """
    calibration_cfg = cfg.get("calibration", {}) or {}
    fraction = float(calibration_cfg.get("conformal_fraction", 0.5))
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration.conformal_fraction must be within (0, 1)")
    original_indices = [int(index) for index in holdout_indices]
    if len(original_indices) < 4:
        raise ValueError(
            "Validation holdout needs at least four samples for disjoint "
            "post-hoc and conformal calibration"
        )

    class HoldoutView:
        def __init__(self):
            all_sids = list(getattr(dataset, "sample_sids", []))
            all_groups = list(getattr(dataset, "sample_groups", []))
            all_labels = list(getattr(dataset, "sample_labels", []))
            all_years = list(getattr(dataset, "sample_years", []))
            if not (
                len(all_sids) == len(dataset)
                and len(all_groups) == len(dataset)
                and len(all_labels) == len(dataset)
            ):
                raise ValueError(
                    "Validation dataset must expose sample_sids, sample_groups and "
                    "sample_labels for three-way calibration splitting"
                )
            self.sample_sids = [all_sids[index] for index in original_indices]
            self.sample_groups = [all_groups[index] for index in original_indices]
            self.sample_labels = [int(all_labels[index]) for index in original_indices]
            self.sample_years = (
                [int(all_years[index]) for index in original_indices]
                if len(all_years) == len(dataset)
                else [0] * len(original_indices)
            )

        def __len__(self):
            return len(original_indices)

        def __getitem__(self, index):
            return dataset[original_indices[index]]

    inner_cfg = copy.deepcopy(cfg)
    inner_cfg.setdefault("calibration", {})["validation_fraction"] = fraction
    inner_cfg["calibration"]["split_seed"] = int(
        calibration_cfg.get("split_seed", cfg.get("train", {}).get("seed", 42))
    ) + 104729
    posthoc_local, conformal_local, inner_meta = split_validation_dataset(
        inner_cfg, HoldoutView()
    )
    posthoc_indices = [original_indices[index] for index in posthoc_local.indices]
    conformal_indices = [original_indices[index] for index in conformal_local.indices]
    if set(posthoc_indices) & set(conformal_indices):
        raise RuntimeError("Post-hoc and conformal calibration subsets overlap")
    return (
        Subset(dataset, posthoc_indices),
        Subset(dataset, conformal_indices),
        {
            "conformal_fraction": fraction,
            "num_posthoc_calibration": len(posthoc_indices),
            "num_conformal_calibration": len(conformal_indices),
            "posthoc_calibration_indices": posthoc_indices,
            "conformal_calibration_indices": conformal_indices,
            "posthoc_label_counts": inner_meta["selection_label_counts"],
            "conformal_label_counts": inner_meta["calibration_label_counts"],
            "posthoc_year_counts": inner_meta["selection_year_counts"],
            "conformal_year_counts": inner_meta["calibration_year_counts"],
            "posthoc_year_label_counts": inner_meta[
                "selection_year_label_counts"
            ],
            "conformal_year_label_counts": inner_meta[
                "calibration_year_label_counts"
            ],
        },
    )


def enforce_failed_ratio(
    metrics: dict[str, Any],
    cfg: dict,
    split_name: str,
    max_failed_ratio: float | None = None,
) -> None:
    total = int(metrics.get("num_eval", 0)) + int(metrics.get("num_failed", 0))
    if total <= 0:
        raise RuntimeError(f"{split_name}: no valid or failed samples were seen")
    failed_ratio = float(metrics.get("num_failed", 0)) / float(total)
    if max_failed_ratio is None:
        max_failed_ratio = float(cfg.get("data", {}).get("max_failed_ratio", 0.0))
    else:
        max_failed_ratio = float(max_failed_ratio)
    if failed_ratio > max_failed_ratio:
        raise RuntimeError(
            f"{split_name}: failed sample ratio {failed_ratio:.4f} exceeds "
            f"data.max_failed_ratio={max_failed_ratio:.4f}"
        )


def _normalize_robust_val_scenarios(raw: Any) -> list[dict[str, Any]]:
    scenarios = list(DEFAULT_ROBUST_VAL_SCENARIOS) if raw is None else raw
    if not isinstance(scenarios, list):
        raise ValueError("eval.robust_val.scenarios must be a list")
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for idx, item in enumerate(scenarios):
        if not isinstance(item, dict):
            raise ValueError(f"eval.robust_val.scenarios[{idx}] must be a mapping")
        perturb_type = str(item.get("perturb_type") or "").strip()
        if not perturb_type or perturb_type == "clean":
            raise ValueError(f"eval.robust_val.scenarios[{idx}] requires a non-clean perturb_type")
        if perturb_type not in EVAL_PERTURB_TYPES:
            raise ValueError(
                f"eval.robust_val.scenarios[{idx}] has unsupported perturb_type={perturb_type!r}"
            )
        strength = float(item.get("strength", 0.5))
        weight = float(item.get("weight", 0.0))
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(f"eval.robust_val.scenarios[{idx}].strength must be within [0, 1]")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"eval.robust_val.scenarios[{idx}].weight must be non-negative")
        name = str(item.get("name") or f"{perturb_type}_s{strength:g}").strip()
        if not name:
            raise ValueError(f"eval.robust_val.scenarios[{idx}].name must not be empty")
        if name in names:
            raise ValueError(f"Duplicate eval.robust_val scenario name: {name}")
        names.add(name)
        out.append(
            {
                "name": name,
                "perturb_type": perturb_type,
                "strength": strength,
                "weight": weight,
            }
        )
    if not out:
        raise ValueError("eval.robust_val.scenarios must not be empty")
    return out


def build_robust_val_loaders(
    cfg: dict,
    subset_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    robust_val_cfg = cfg.get("eval", {}).get("robust_val", {}) or {}
    if not bool(robust_val_cfg.get("enabled", False)):
        return []
    out: list[dict[str, Any]] = []
    for item in _normalize_robust_val_scenarios(robust_val_cfg.get("scenarios")):
        dataset = build_dataset(
            cfg,
            "val",
            is_train=False,
            perturb_type=item["perturb_type"],
            perturb_strength=item["strength"],
        )
        if subset_indices is not None:
            dataset = Subset(dataset, subset_indices)
        out.append({**item, "loader": build_loader(cfg, dataset, is_train=False)})
    return out


RELIABILITY_CALIBRATION_PERTURBATIONS = {
    "api_degraded": ("api",),
    "graph_degraded": ("graph",),
    "manifest_degraded": ("manifest",),
    "all_degraded": ("api", "graph", "manifest"),
}
RELIABILITY_CALIBRATION_MISSING = (
    "api_missing",
    "graph_missing",
    "manifest_missing",
)

# Training logs only consume this small subset. Keeping the other diagnostics
# on the evaluation path avoids synchronizing dozens of GPU tensors per batch.
TRAIN_LOG_DIAGNOSTIC_KEYS = tuple(
    key
    for key in GATE_DIAGNOSTIC_KEYS
    if key.startswith(
        ("discount_", "fusion_weight_", "entropy_", "margin_", "uncertainty_proxy_")
    )
    or key == "zero_weight_fallback_used"
)


def reliability_calibration_scenarios(cfg: dict) -> list[dict[str, Any]]:
    """Build transformed post-hoc views used by the global opinion router.

    These views are built only from the post-hoc calibration subset. They never
    touch checkpoint selection or the disjoint decision-calibration subset.
    Branch-correctness reliability is fitted on clean rows only; transformed
    views teach the router how to allocate mass under degraded and missing
    modalities. ``reliability_branches`` is retained as scenario provenance for
    summaries and compatibility with existing experiment records.
    """
    robust_cfg = cfg.get("robust", {}) or {}
    calibration_cfg = cfg.get("calibration", {}) or {}
    strengths = sorted(
        {
            float(value)
            for value in (
                calibration_cfg.get("perturb_strengths")
                or robust_cfg.get("perturb_strengths")
                or []
            )
            if math.isfinite(float(value)) and 0.0 < float(value) <= 1.0
        }
    )
    graded = [
        {
            "name": f"calibration_{perturb_type}_s{strength:g}",
            "perturb_type": perturb_type,
            "scenario_group": perturb_type,
            "strength": strength,
            "reliability_branches": list(branches),
        }
        for perturb_type, branches in RELIABILITY_CALIBRATION_PERTURBATIONS.items()
        for strength in strengths
    ]
    missing = [
        {
            "name": f"calibration_{perturb_type}",
            "perturb_type": perturb_type,
            "scenario_group": "missing",
            "strength": 1.0,
            # Missing views train the router's unknown outcome. The absent
            # branch has no correctness target, while the surviving branch
            # logits are unchanged and must not be counted as extra clean
            # reliability observations.
            "reliability_branches": [],
        }
        for perturb_type in RELIABILITY_CALIBRATION_MISSING
    ]
    return [*graded, *missing]


def build_reliability_calibration_loaders(
    cfg: dict,
    subset_indices: list[int] | None,
) -> list[dict[str, Any]]:
    """Build deterministic degraded views of the post-hoc calibration subset."""
    out: list[dict[str, Any]] = []
    for item in reliability_calibration_scenarios(cfg):
        dataset = build_dataset(
            cfg,
            "val",
            is_train=False,
            perturb_type=item["perturb_type"],
            perturb_strength=item["strength"],
        )
        if subset_indices is not None:
            dataset = Subset(dataset, subset_indices)
        # These loaders are consumed once while caching frozen branch outputs.
        # Keeping workers alive for every degraded view would accumulate worker
        # processes and file descriptors across all calibration scenarios.
        loader = build_loader(
            cfg,
            dataset,
            is_train=False,
            persistent_workers_override=False,
        )
        out.append({**item, "loader": loader})
    return out


def uses_routing_calibration_scenarios(cfg: dict) -> bool:
    """Return whether post-hoc fitting consumes transformed modality views."""
    fusion_cfg = cfg.get("fusion", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    return bool(
        str(fusion_cfg.get("combination", "linear")).lower() == "routed"
        and routing_cfg.get("enabled", False)
        and routing_cfg.get("posthoc_refine", True)
    )


@torch.no_grad()
def evaluate_robust_validation(
    model,
    loaders: list[dict[str, Any]],
    device,
    use_amp: bool,
    cfg: dict,
    selective_threshold: float | None = None,
    selective_score_type: str = "max_probability",
    classification_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for item in loaders:
        name = str(item["name"])
        metrics, _ = evaluate(
            model,
            item["loader"],
            device,
            use_amp,
            f"val_{name}",
            dump_rows=False,
            selective_threshold=selective_threshold,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )
        enforce_failed_ratio(metrics, cfg, f"val_{name}")
        results[name] = metrics
    return results


def checkpoint_score(
    cfg: dict,
    clean_metrics: dict[str, float],
    robust_metrics: dict[str, dict[str, float]],
    robust_val_loaders: list[dict[str, Any]],
) -> tuple[float, str]:
    metric_name = str(cfg.get("train", {}).get("checkpoint_metric", "clean_macro_f1")).strip().lower()
    clean_f1 = float(clean_metrics["macro_f1"])
    if metric_name in {"clean", "clean_macro_f1", "macro_f1", "val_macro_f1"}:
        return clean_f1, "clean_macro_f1"
    if metric_name != "robust_composite":
        raise ValueError(f"Unsupported train.checkpoint_metric: {metric_name}")
    if not robust_val_loaders:
        raise ValueError("train.checkpoint_metric=robust_composite requires eval.robust_val.enabled=true")

    robust_val_cfg = cfg.get("eval", {}).get("robust_val", {}) or {}
    clean_weight = float(robust_val_cfg.get("clean_weight", 0.4))
    if not math.isfinite(clean_weight) or clean_weight < 0.0:
        raise ValueError("eval.robust_val.clean_weight must be non-negative")
    weighted_sum = clean_weight * clean_f1
    weight_sum = clean_weight
    for item in robust_val_loaders:
        weight = float(item["weight"])
        if weight <= 0.0:
            continue
        name = str(item["name"])
        if name not in robust_metrics:
            raise KeyError(f"Missing robust validation metrics for scenario: {name}")
        weighted_sum += weight * float(robust_metrics[name]["macro_f1"])
        weight_sum += weight
    if weight_sum <= 0.0:
        raise ValueError("Robust validation composite weights must sum to a positive value")
    return weighted_sum / weight_sum, "robust_composite"


def checkpoint_requires_robust_validation(cfg: dict) -> bool:
    metric_name = str(
        cfg.get("train", {}).get("checkpoint_metric", "clean_macro_f1")
    ).strip().lower()
    return metric_name == "robust_composite"


def build_model(cfg: dict, feature_dim: int) -> TriModalRobustModel:
    removed_sections = [
        key for key in ("semantic_cross_attention", "semantic_reconstruction") if key in cfg
    ]
    if removed_sections:
        raise ValueError(
            "The formal lean pipeline no longer accepts semantic interaction/reconstruction "
            f"config sections: {removed_sections}. Remove these legacy sections from the YAML."
        )
    model_cfg = cfg.get("model", {})
    api_cfg = model_cfg.get("api_encoder", {})
    graph_cfg = model_cfg.get("graph_encoder", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    gate_cfg = model_cfg.get("gate", {})
    fusion_cfg = cfg.get("fusion", {}) or {}
    fusion_mode = str(model_cfg.get("fusion_mode", "tri_modal_ours"))
    configured_fusion_mode = str(fusion_cfg.get("mode", "")).lower()
    if configured_fusion_mode == "discount_probability":
        fusion_mode = "discount_probability"
    elif configured_fusion_mode not in {"", "model_dispatch"}:
        raise ValueError(f"Unsupported fusion.mode: {configured_fusion_mode}")

    # Input-duplication guardrail: Graph may be structurally selected around API
    # methods, but fine-grained API behavior hints must not be copied directly
    # into graph node features when the branches are treated as distinct views.
    combination = str(fusion_cfg.get("combination", "linear")).lower()
    if combination in {
        "routed",
        "dempster",
        "cumulative",
        "log_pool",
        "ecml_style",
    }:
        coupling = []
        if bool(graph_cfg.get("use_behavior_hint", False)):
            coupling.append("model.graph_encoder.use_behavior_hint")
        if coupling:
            raise ValueError(
                f"fusion.combination={combination} requires non-duplicated branch inputs, "
                f"but these settings copy cross-modal hints into a branch: {coupling}. "
                "Disable them, or use fusion.combination=linear for a coupled comparison pipeline."
            )

    return TriModalRobustModel(
        in_feat_dim=feature_dim,
        num_classes=int(model_cfg.get("num_classes", 2)),
        fusion_mode=fusion_mode,
        api_num_hash_buckets=int(api_cfg.get("num_hash_buckets", 8192)),
        api_type_vocab_size=int(api_cfg.get("type_vocab_size", 16)),
        api_emb_dim=int(api_cfg.get("emb_dim", 128)),
        api_hidden_dim=int(api_cfg.get("hidden_dim", 256)),
        api_dropout=float(api_cfg.get("dropout", 0.15)),
        api_encoder_type=str(api_cfg.get("type", "transformer")),
        api_layers=int(api_cfg.get("layers", 2)),
        api_heads=int(api_cfg.get("heads", 4)),
        api_max_seq_len=int(api_cfg.get("max_seq_len", 1024)),
        graph_emb_dim=int(graph_cfg.get("emb_dim", 128)),
        graph_hidden=int(graph_cfg.get("hidden", 128)),
        graph_heads=int(graph_cfg.get("heads", 4)),
        graph_layers=int(graph_cfg.get("layers", 2)),
        graph_encoder_type=str(graph_cfg.get("type", "gatv2")),
        max_nodes_gnn=int(model_cfg.get("max_nodes_gnn", graph_cfg.get("max_nodes", 12288))),
        use_graph_behavior_hint=bool(graph_cfg.get("use_behavior_hint", False)),
        manifest_in_dim=int(manifest_cfg.get("in_dim", 256)),
        manifest_emb_dim=int(manifest_cfg.get("emb_dim", 128)),
        manifest_hidden_dim=int(manifest_cfg.get("hidden_dim", 256)),
        manifest_dropout=float(manifest_cfg.get("dropout", 0.1)),
        joint_emb_dim=int(model_cfg.get("joint_emb_dim", 128)),
        quality_fusion_temperature=float(model_cfg.get("quality_fusion_temperature", 10.0)),
        gate_hidden_dim=int(gate_cfg.get("hidden_dim", 128)),
        gate_detach=bool(gate_cfg.get("detach", True)),
        use_consistency_evidence=bool(gate_cfg.get("use_consistency_evidence", True)),
        use_conflict_evidence=bool(gate_cfg.get("use_conflict_evidence", True)),
        use_perturbation_evidence=bool(gate_cfg.get("use_perturbation_evidence", False)),
        apply_alive_mask_to_learned_gate=bool(gate_cfg.get("apply_alive_mask", True)),
        discount_fusion_config=fusion_cfg,
    )


def _metrics(labels: list[int], probs: list[float], preds: list[int]) -> dict[str, float]:
    if not labels:
        return {
            "acc": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "f1_pos": 0.0,
            "recall": 0.0,
            "macro_recall": 0.0,
            "recall_pos": 0.0,
            "auc_defined": 0,
            "auc": float("nan"),
            "ap_defined": 0,
            "ap": float("nan"),
            "brier": 0.0,
            "ece_10": 0.0,
            "mean_confidence": 0.0,
            "confidence_accuracy_gap": 0.0,
        }
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    pred_arr = np.asarray(preds, dtype=np.int64)
    confidence = np.maximum(p, 1.0 - p)
    # Probability calibration is a property of the predictive distribution,
    # independent of a downstream operating threshold selected for malware
    # recall.  Pair max-class confidence with max-class correctness so ECE
    # remains comparable between fixed-0.5 and threshold-tuned evaluations.
    probability_pred = (p >= 0.5).astype(np.int64)
    correct = (probability_pred == y.astype(np.int64)).astype(np.float64)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        if hi >= 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_pos = float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0))
    macro_recall = float(recall_score(labels, preds, average="macro", zero_division=0))
    recall_pos = float(recall_score(labels, preds, average="binary", pos_label=1, zero_division=0))
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": macro_f1,
        "macro_f1": macro_f1,
        "f1_pos": f1_pos,
        "recall": macro_recall,
        "macro_recall": macro_recall,
        "recall_pos": recall_pos,
        "brier": brier,
        "ece_10": float(ece),
        "mean_confidence": float(confidence.mean()),
        "confidence_accuracy_gap": float(confidence.mean() - correct.mean()),
    }
    if len(set(labels)) > 1:
        out["auc_defined"] = 1
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap_defined"] = 1
        out["ap"] = float(average_precision_score(labels, probs))
    else:
        out["auc_defined"] = 0
        out["auc"] = float("nan")
        out["ap_defined"] = 0
        out["ap"] = float("nan")
    return out


def _calibration_ece(
    scores: np.ndarray,
    correctness: np.ndarray,
    bins: int = 10,
) -> float:
    if scores.size == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(scores[mask].mean()) - float(correctness[mask].mean())
        )
    return float(ece)


def _branch_prediction_row(
    extra: dict[str, Any],
    labels: torch.Tensor,
    index: int,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    label = int(labels[index].detach().cpu().item())
    for branch, key in BRANCH_EVAL_LOGIT_KEYS.items():
        branch_logits = extra.get(key)
        if not isinstance(branch_logits, torch.Tensor):
            continue
        if branch_logits.ndim != 2 or branch_logits.size(0) <= index or branch_logits.size(1) < 2:
            continue
        logits_i = branch_logits.float()[index]
        prob_i = torch.softmax(logits_i, dim=-1)
        pred_i = int(logits_i.argmax(dim=-1).detach().cpu().item())
        row[f"{branch}_prob"] = float(prob_i[1].detach().cpu().item())
        row[f"{branch}_pred"] = pred_i
        row[f"{branch}_correct"] = int(pred_i == label)
        row[f"{branch}_confidence"] = float(prob_i.max().detach().cpu().item())
    return row


def _finite_row_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def compute_branch_reliability_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Compare calibrated branch reliability against branch correctness."""
    out: dict[str, float | int] = {}
    for branch in BRANCH_EVAL_LOGIT_KEYS:
        reliability_values: list[float] = []
        correctness_values: list[float] = []
        for row in rows:
            alive = _finite_row_float(row, f"{branch}_alive")
            if alive is not None and alive < 0.5:
                continue
            reliability = _finite_row_float(row, f"predicted_reliability_{branch}")
            correctness = _finite_row_float(row, f"{branch}_correct")
            if reliability is None or correctness is None:
                continue
            reliability_values.append(min(1.0, max(0.0, reliability)))
            correctness_values.append(1.0 if correctness >= 0.5 else 0.0)
        count = len(reliability_values)
        if count == 0:
            continue
        out[f"{branch}_reliability_count"] = count
        reliability_arr = np.asarray(reliability_values, dtype=np.float64)
        correctness_arr = np.asarray(correctness_values, dtype=np.float64)
        out[f"{branch}_reliability_brier"] = float(
            np.mean((reliability_arr - correctness_arr) ** 2)
        )
        out[f"{branch}_reliability_ece_10"] = _calibration_ece(
            reliability_arr,
            correctness_arr,
            bins=10,
        )
        out[f"{branch}_reliability_mean"] = float(reliability_arr.mean())
        out[f"{branch}_branch_accuracy"] = float(correctness_arr.mean())
        out[f"{branch}_reliability_accuracy_gap"] = float(
            reliability_arr.mean() - correctness_arr.mean()
        )
        if len(set(correctness_values)) > 1:
            out[f"{branch}_reliability_auc_defined"] = 1
            out[f"{branch}_reliability_ap_defined"] = 1
            out[f"{branch}_reliability_auc"] = float(
                roc_auc_score(correctness_arr, reliability_arr)
            )
            out[f"{branch}_reliability_ap"] = float(
                average_precision_score(correctness_arr, reliability_arr)
            )
        else:
            out[f"{branch}_reliability_auc_defined"] = 0
            out[f"{branch}_reliability_ap_defined"] = 0
            out[f"{branch}_reliability_auc"] = float("nan")
            out[f"{branch}_reliability_ap"] = float("nan")
    return out

def estimate_branch_competence_prior(
    rows: list[dict[str, Any]],
    cfg: dict,
) -> dict[str, Any]:
    """Estimate a validation-calibrated global competence prior for each branch."""
    prior_cfg = (cfg.get("fusion", {}) or {}).get("branch_competence_prior", {}) or {}
    if not bool(prior_cfg.get("enabled", False)):
        return {"enabled": False}
    metric = str(prior_cfg.get("metric", "macro_f1")).lower()
    normalization = str(prior_cfg.get("normalization", "best")).lower()
    min_value = float(prior_cfg.get("min_value", 0.5))
    if metric not in {"macro_f1", "accuracy", "inverse_brier"}:
        raise ValueError("fusion.branch_competence_prior.metric must be macro_f1, accuracy, or inverse_brier")
    if normalization not in {"best", "direct"}:
        raise ValueError("fusion.branch_competence_prior.normalization must be best or direct")
    if not math.isfinite(min_value) or not 0.0 <= min_value <= 1.0:
        raise ValueError("fusion.branch_competence_prior.min_value must be within [0, 1]")

    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for branch in BRANCH_EVAL_LOGIT_KEYS:
        labels: list[int] = []
        preds: list[int] = []
        probs: list[float] = []
        for row in rows:
            try:
                label = int(row["label"])
                pred = int(row[f"{branch}_pred"])
            except (KeyError, TypeError, ValueError):
                continue
            labels.append(label)
            preds.append(pred)
            prob = _finite_row_float(row, f"{branch}_prob")
            if prob is not None:
                probs.append(min(1.0, max(0.0, prob)))
        counts[branch] = len(labels)
        if not labels:
            scores[branch] = 0.0
            continue
        if metric == "accuracy":
            scores[branch] = float(np.mean(np.asarray(labels) == np.asarray(preds)))
        elif metric == "inverse_brier":
            if len(probs) != len(labels):
                raise ValueError(
                    f"branch {branch} is missing probabilities required for inverse_brier competence"
                )
            y = np.asarray(labels, dtype=np.float64)
            p = np.asarray(probs, dtype=np.float64)
            scores[branch] = float(1.0 - np.mean((p - y) ** 2))
        else:
            scores[branch] = float(f1_score(labels, preds, average="macro", zero_division=0))

    fusion_cfg = cfg.get("fusion", {}) or {}
    combination = str(fusion_cfg.get("combination", "linear")).lower()
    if combination != "linear":
        normalization_branches = ("api", "graph", "manifest")
    elif bool(fusion_cfg.get("linear_use_joint_branch", True)):
        normalization_branches = tuple(BRANCH_EVAL_LOGIT_KEYS)
    else:
        normalization_branches = ("api", "graph", "manifest")
    max_score = max(
        (scores[name] for name in normalization_branches if name in scores),
        default=0.0,
    )
    prior: dict[str, float] = {}
    for branch, score in scores.items():
        value = score
        if normalization == "best" and max_score > 0.0:
            value = score / max_score
        prior[branch] = float(min(1.0, max(min_value, value)))
    values = [prior[name] for name in BRANCH_EVAL_LOGIT_KEYS]
    return {
        "enabled": True,
        "metric": metric,
        "normalization": normalization,
        "min_value": min_value,
        "scores": scores,
        "counts": counts,
        "normalization_branches": list(normalization_branches),
        "prior": prior,
        "values": values,
    }


def apply_branch_competence_prior(model, summary: dict[str, Any]) -> None:
    if not bool((summary or {}).get("enabled", False)):
        return
    values = summary.get("values")
    if values is None:
        prior = summary.get("prior") or {}
        values = [prior[name] for name in BRANCH_EVAL_LOGIT_KEYS]
    model.discount_fusion.set_branch_competence_prior(values, enabled=True)


def estimate_model_visible_integrity_reference(
    rows: list[dict[str, Any]],
    cfg: dict,
) -> dict[str, Any]:
    """Fit clean calibration references for the configured integrity modifier."""
    modifier_cfg = (cfg.get("fusion", {}) or {}).get("visible_integrity_modifier", {}) or {}
    if not bool(modifier_cfg.get("enabled", False)):
        return {"enabled": False}
    min_reference = float(modifier_cfg.get("min_reference", 1.0e-6))
    mode = str(modifier_cfg.get("mode", "bounded_visibility")).lower()
    if mode not in {"bounded_visibility", "relative_effective"}:
        raise ValueError(
            "fusion.visible_integrity_modifier.mode must be "
            "'bounded_visibility' or 'relative_effective'"
        )
    if mode == "relative_effective":
        beta = 1.0
        min_value = 0.0
    else:
        beta = float(modifier_cfg.get("beta", 1.0))
        min_value = float(modifier_cfg.get("min_value", 0.5))
    if not math.isfinite(min_reference) or not 0.0 < min_reference <= 1.0:
        raise ValueError("fusion.visible_integrity_modifier.min_reference must be within (0, 1]")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("fusion.visible_integrity_modifier.beta must be finite and positive")
    if not math.isfinite(min_value) or not 0.0 <= min_value <= 1.0:
        raise ValueError("fusion.visible_integrity_modifier.min_value must be within [0, 1]")

    def _reference_value(row: dict[str, Any], branch: str) -> float | None:
        if mode == "relative_effective":
            value = _finite_row_float(row, f"effective_{branch}_integrity")
            if value is None:
                integrity = _finite_row_float(row, f"{branch}_integrity")
                if integrity is None:
                    return None
                coverage = (
                    _finite_row_float(row, f"{branch}_encoder_coverage")
                    if branch in {"api", "graph"}
                    else 1.0
                )
                if coverage is None:
                    return None
                value = integrity * coverage
        elif branch == "api":
            value = _finite_row_float(row, "api_encoder_coverage")
        elif branch == "graph":
            value = _finite_row_float(row, "graph_encoder_coverage")
        else:
            value = 1.0
        if value is None:
            return None
        return min(1.0, max(0.0, value))

    references: dict[str, float] = {}
    counts: dict[str, int] = {}
    for branch in ("api", "graph", "manifest"):
        values = [
            value
            for row in rows
            if (value := _reference_value(row, branch)) is not None and math.isfinite(value)
        ]
        counts[branch] = len(values)
        if not values:
            references[branch] = 1.0
            continue
        references[branch] = float(min(1.0, max(min_reference, float(np.median(values)))))
    return {
        "enabled": True,
        "mode": mode,
        "metric": (
            "median_clean_effective_integrity"
            if mode == "relative_effective"
            else "median_clean_encoder_coverage"
        ),
        "beta": beta,
        "min_value": min_value,
        "min_reference": min_reference,
        "counts": counts,
        "reference": references,
        "values": [references[name] for name in ("api", "graph", "manifest")],
    }


def apply_model_visible_integrity_reference(model, summary: dict[str, Any]) -> None:
    if not bool((summary or {}).get("enabled", False)):
        return
    values = summary.get("values")
    if values is None:
        reference = summary.get("reference") or {}
        values = [reference[name] for name in ("api", "graph", "manifest")]
    model.discount_fusion.set_visible_integrity_reference(values, enabled=True)


def _selective_metrics(
    labels: list[int],
    preds: list[int],
    acceptance_scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    if not labels or len(acceptance_scores) != len(labels):
        return {}
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    score = np.nan_to_num(
        np.asarray(acceptance_scores, dtype=np.float64),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    accepted = score >= float(threshold)
    coverage = float(accepted.mean())
    errors = (pred != y).astype(np.float64)
    out = {
        "rejection_threshold": float(threshold),
        "coverage": coverage,
        "rejection_rate": float(1.0 - coverage),
        "num_accepted": int(accepted.sum()),
        "num_rejected": int((~accepted).sum()),
    }
    out.update(_selective_ranking_metrics(labels, preds, acceptance_scores))
    if accepted.any():
        out.update(
            {
                "selective_metrics_defined": True,
                "selective_risk": float(errors[accepted].mean()),
                "selective_acc": float(1.0 - errors[accepted].mean()),
                "selective_macro_f1": float(
                    f1_score(y[accepted], pred[accepted], average="macro", zero_division=0)
                ),
            }
        )
    else:
        out.update(
            {
                "selective_metrics_defined": False,
                "selective_risk": None,
                "selective_acc": None,
                "selective_macro_f1": None,
            }
        )
    return out


def _selective_ranking_metrics(
    labels: list[int],
    preds: list[int],
    acceptance_scores: list[float],
) -> dict[str, float]:
    """Report threshold-free selective-ranking quality."""
    if not labels or len(acceptance_scores) != len(labels):
        return {}
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    score = np.nan_to_num(
        np.asarray(acceptance_scores, dtype=np.float64),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    errors = (pred != y).astype(np.float64)
    order = np.argsort(-score, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return {"aurc": float(cumulative_risk.mean())}


def fit_malware_classification_threshold(
    rows: list[dict[str, Any]], config: dict | None = None
) -> dict[str, Any] | None:
    """Fit a binary decision threshold without consulting the test set.

    The threshold maximizes calibration-set macro-F1 subject to a minimum
    malware recall. It is fitted before the disjoint decision-calibration
    subset is used for selective risk control.
    """
    config = config or {}
    if not bool(config.get("enabled", False)):
        return None
    objective = str(config.get("objective", "macro_f1")).strip().lower()
    if objective != "macro_f1":
        raise ValueError("classification_threshold.objective currently supports only 'macro_f1'")
    min_malware_recall = float(config.get("min_malware_recall", 0.90))
    if not 0.0 <= min_malware_recall <= 1.0:
        raise ValueError(
            "classification_threshold.min_malware_recall must be within [0, 1]"
        )

    valid: list[tuple[float, int]] = []
    for row in rows:
        probability = _finite_row_float(row, "prob_malware")
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if probability is not None and label in {0, 1}:
            valid.append((float(probability), label))
    if not valid:
        raise ValueError("Classification-threshold fitting requires finite probabilities")
    labels = np.asarray([label for _probability, label in valid], dtype=np.int64)
    present_labels = set(int(label) for label in labels.tolist())
    if present_labels != {0, 1}:
        raise ValueError(
            "Classification-threshold fitting requires both benign and malware samples"
        )

    unique_probabilities = sorted({probability for probability, _label in valid})
    candidates = [unique_probabilities[0]]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(unique_probabilities[:-1], unique_probabilities[1:])
    )
    candidates.append(math.nextafter(unique_probabilities[-1], float("inf")))
    probabilities = np.asarray(
        [probability for probability, _label in valid], dtype=np.float64
    )

    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        predictions = (probabilities >= float(threshold)).astype(np.int64)
        malware_recall = float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        )
        if malware_recall + 1e-12 < min_malware_recall:
            continue
        macro_f1 = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(labels, predictions))
        # The first two terms encode the declared objective and constraint.
        # Remaining terms make exact ties deterministic without a test-set choice.
        key = (macro_f1, malware_recall, -abs(float(threshold) - 0.5), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "macro_f1": macro_f1,
                "malware_recall": malware_recall,
                "accuracy": accuracy,
            }
    if best is None:
        raise RuntimeError(
            "No classification threshold satisfies the malware-recall constraint"
        )

    fixed_predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "enabled": True,
        "objective": objective,
        "min_malware_recall": min_malware_recall,
        "calibration_split": "val_posthoc_calibration",
        "num_calibration": int(len(valid)),
        "num_calibration_benign": int((labels == 0).sum()),
        "num_calibration_malware": int((labels == 1).sum()),
        "num_candidates": int(len(candidates)),
        "fixed_0_5_macro_f1": float(
            f1_score(labels, fixed_predictions, average="macro", zero_division=0)
        ),
        "fixed_0_5_malware_recall": float(
            recall_score(labels, fixed_predictions, pos_label=1, zero_division=0)
        ),
        **best,
    }


def fit_rejection_threshold(rows: list[dict[str, Any]], config: dict | None = None) -> float | None:
    """Choose an acceptance threshold on validation data for target coverage."""
    config = config or {}
    if not bool(config.get("enabled", False)):
        return None
    if str(config.get("mode", "threshold")).lower() in {"conformal", "risk_control"}:
        return None
    target_coverage = float(config.get("target_coverage", 0.9))
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("selective_prediction.target_coverage must be within (0, 1]")
    scores = sorted(
        (
            float(row["acceptance_score"])
            for row in rows
            if row.get("acceptance_score") is not None
            and math.isfinite(float(row["acceptance_score"]))
        ),
        reverse=True,
    )
    if not scores:
        raise ValueError(
            "selective_prediction.enabled=true requires acceptance_score from discount fusion"
        )
    accepted_count = max(1, int(math.ceil(target_coverage * len(scores))))
    return float(scores[accepted_count - 1])


def _selective_metrics_from_rows(
    rows: list[dict[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    if threshold is None:
        return {}
    valid = [row for row in rows if row.get("acceptance_score") is not None]
    return _selective_metrics(
        [int(row["label"]) for row in valid],
        [int(row["pred"]) for row in valid],
        [float(row["acceptance_score"]) for row in valid],
        threshold,
    )


def _selective_score_type(config: dict | None = None) -> str:
    config = config or {}
    mode = str(config.get("mode", "threshold")).lower()
    if mode == "conformal":
        score_type = "max_probability"
    elif mode == "risk_control":
        score_type = str(config.get("threshold_score", "model_acceptance")).lower()
    else:
        score_type = str(config.get("threshold_score", "max_probability")).lower()
    allowed = {"max_probability", "evidential_certainty", "model_acceptance"}
    if score_type not in allowed:
        raise ValueError(
            "selective_prediction.threshold_score must be one of "
            f"{sorted(allowed)}, got {score_type!r}"
        )
    return score_type


def _batch_selective_score(
    prob_malware: torch.Tensor,
    extra: dict[str, Any],
    score_type: str,
    classification_threshold: float = 0.5,
) -> torch.Tensor:
    """Return a larger-is-safer score for thresholding and AURC."""
    score_type = str(score_type).lower()
    if score_type == "max_probability":
        predicted_malware = prob_malware >= float(classification_threshold)
        return torch.where(
            predicted_malware,
            prob_malware,
            1.0 - prob_malware,
        ).clamp(0.0, 1.0)
    if score_type == "evidential_certainty":
        uncertainty = extra.get("fused_uncertainty")
        if not isinstance(uncertainty, torch.Tensor):
            raise ValueError(
                "evidential_certainty threshold requires fused_uncertainty"
            )
        return (1.0 - uncertainty.float().view(-1)).clamp(0.0, 1.0)
    if score_type == "model_acceptance":
        acceptance = extra.get("acceptance_score")
        if not isinstance(acceptance, torch.Tensor):
            raise ValueError("model_acceptance threshold requires acceptance_score")
        return acceptance.float().view(-1).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported selective score type: {score_type}")


# Selective-decision baselines and calibrated decision rules. Conformal modes
# control prediction-set coverage; the risk-control mode bounds its declared
# malware false-negative loss. Acceptance rate is not guaranteed by either.

def _conformal_quantile(scores: list[float], alpha: float) -> float:
    finite = sorted(float(s) for s in scores if s is not None and math.isfinite(float(s)))
    n = len(finite)
    if n == 0:
        return float("inf")  # no calibration evidence -> never reject on this class
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    if rank < 1:
        rank = 1
    return float(finite[rank - 1])


def _row_raw_conflict(row: dict[str, Any]) -> float:
    for key in ("raw_conflict", "raw_conflict_mass", "effective_conflict"):
        value = _finite_row_float(row, key)
        if value is not None:
            return min(1.0, max(0.0, float(value)))
    return 0.0


def _conformal_nonconformity(
    p1: float,
    label: int,
    raw_conflict: float = 0.0,
    *,
    use_raw_conflict: bool = False,
) -> float:
    base = (1.0 - p1) if int(label) == 1 else p1
    if use_raw_conflict:
        base += min(1.0, max(0.0, float(raw_conflict)))
    return float(base)


def fit_conformal_thresholds(
    rows: list[dict[str, Any]], config: dict | None = None
) -> dict[str, Any] | None:
    """Fit per-class conformal nonconformity thresholds on calibration rows."""
    config = config or {}
    if not bool(config.get("enabled", False)):
        return None
    if str(config.get("mode", "threshold")).lower() != "conformal":
        return None
    target_coverage = float(config.get("target_coverage", 0.9))
    alpha = float(config.get("alpha", 1.0 - target_coverage))
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal selective_prediction.alpha must be within (0, 1)")
    class_conditional = bool(config.get("class_conditional", True))
    use_raw_conflict = bool(config.get("use_raw_conflict", False))
    scores: dict[int, list[float]] = {0: [], 1: []}
    for row in rows:
        prob = row.get("prob_malware")
        label = row.get("label")
        if prob is None or label is None:
            continue
        p1 = float(prob)
        label = int(label)
        # Nonconformity of the true class, computed identically to the
        # prediction-set test below. Probability-only and raw-conflict-augmented
        # scores are separate conformal baselines; the main method uses the
        # disjoint malware false-negative risk-control rule below.
        nonconformity = _conformal_nonconformity(
            p1,
            label,
            _row_raw_conflict(row),
            use_raw_conflict=use_raw_conflict,
        )
        scores.setdefault(label, []).append(nonconformity)
    if class_conditional:
        q_benign = _conformal_quantile(scores.get(0, []), alpha)
        q_malware = _conformal_quantile(scores.get(1, []), alpha)
    else:
        pooled = scores.get(0, []) + scores.get(1, [])
        q_benign = q_malware = _conformal_quantile(pooled, alpha)
    return {
        "mode": "conformal",
        "alpha": alpha,
        "class_conditional": class_conditional,
        "use_raw_conflict": use_raw_conflict,
        "q_benign": q_benign,
        "q_malware": q_malware,
        "num_calibration": int(len(scores.get(0, [])) + len(scores.get(1, []))),
    }


def _uses_conformal_selective(config: dict | None = None) -> bool:
    config = config or {}
    return bool(config.get("enabled", False)) and str(config.get("mode", "threshold")).lower() == "conformal"


def _conformal_prediction_set(
    p1: float,
    thresholds: dict[str, Any],
    raw_conflict: float = 0.0,
) -> tuple[bool, bool]:
    """Return (include_benign, include_malware) for the conformal prediction set."""
    q_benign = float(thresholds.get("q_benign", float("inf")))
    q_malware = float(thresholds.get("q_malware", float("inf")))
    use_raw_conflict = bool(thresholds.get("use_raw_conflict", False))
    include_malware = (
        _conformal_nonconformity(p1, 1, raw_conflict, use_raw_conflict=use_raw_conflict)
        <= q_malware
    )
    include_benign = (
        _conformal_nonconformity(p1, 0, raw_conflict, use_raw_conflict=use_raw_conflict)
        <= q_benign
    )
    return bool(include_benign), bool(include_malware)


def conformal_selective_metrics(
    rows: list[dict[str, Any]], thresholds: dict[str, Any] | None
) -> dict[str, Any]:
    """Security-oriented selective metrics under class-conditional conformal."""
    if not thresholds:
        return {}
    valid = [r for r in rows if r.get("prob_malware") is not None and r.get("label") is not None]
    if not valid:
        return {}
    n = len(valid)
    accepted = 0
    per_class_total = {0: 0, 1: 0}
    per_class_accepted = {0: 0, 1: 0}
    true_in_set = {0: 0, 1: 0}        # empirical conformal coverage check
    accepted_errors = 0
    malware_fn_accepted = 0           # accepted samples that are malware but predicted benign
    empty_sets = 0
    ambiguous_sets = 0
    for row in valid:
        p1 = float(row["prob_malware"])
        y = int(row["label"])
        include_benign, include_malware = _conformal_prediction_set(
            p1,
            thresholds,
            _row_raw_conflict(row),
        )
        size = int(include_benign) + int(include_malware)
        empty_sets += int(size == 0)
        ambiguous_sets += int(size == 2)
        is_accept = size == 1
        pred = 1 if (include_malware and not include_benign) else 0
        per_class_total[y] = per_class_total.get(y, 0) + 1
        true_in_set[y] = true_in_set.get(y, 0) + int(include_malware if y == 1 else include_benign)
        if is_accept:
            accepted += 1
            per_class_accepted[y] = per_class_accepted.get(y, 0) + 1
            if pred != y:
                accepted_errors += 1
            if y == 1 and pred == 0:
                malware_fn_accepted += 1

    def _ratio(num: int, den: int) -> float | None:
        return float(num) / float(den) if den > 0 else None

    out: dict[str, Any] = {
        "conformal_alpha": float(thresholds.get("alpha", 0.0)),
        "conformal_class_conditional": bool(thresholds.get("class_conditional", True)),
        # Acceptance rate = fraction whose prediction set is a singleton. This is
        # NOT conformal coverage -- see conformal_empirical_coverage_* below for
        # the actual guaranteed quantity.
        "conformal_acceptance_rate": _ratio(accepted, n),
        "conformal_rejection_rate": _ratio(n - accepted, n),
        "conformal_empty_set_rate": _ratio(empty_sets, n),
        "conformal_ambiguous_set_rate": _ratio(ambiguous_sets, n),
        "conformal_benign_acceptance_rate": _ratio(per_class_accepted[0], per_class_total[0]),
        "conformal_malware_acceptance_rate": _ratio(per_class_accepted[1], per_class_total[1]),
        "conformal_malware_rejection_rate": (
            None if per_class_total[1] == 0
            else 1.0 - (per_class_accepted[1] / per_class_total[1])
        ),
        # Compatibility metric: accepted false negatives divided by all malware.
        "conformal_malware_fn_after_rejection": _ratio(malware_fn_accepted, per_class_total[1]),
        # Operational false-negative rate among malware samples that were
        # actually accepted for automatic classification.
        "conformal_accepted_malware_fn_rate": _ratio(
            malware_fn_accepted, per_class_accepted[1]
        ),
        "conformal_malware_fn_count": int(malware_fn_accepted),
        "conformal_accepted_malware_count": int(per_class_accepted[1]),
        "conformal_selective_risk": _ratio(accepted_errors, accepted),
        "conformal_selective_acc": (None if accepted == 0 else 1.0 - accepted_errors / accepted),
        # TRUE conformal coverage: P(true label in prediction set | label = c).
        # Should be >= 1 - alpha on exchangeable calibration/test data.
        "conformal_empirical_coverage_benign": _ratio(true_in_set[0], per_class_total[0]),
        "conformal_empirical_coverage_malware": _ratio(true_in_set[1], per_class_total[1]),
        "conformal_num_accepted": int(accepted),
        "conformal_num_rejected": int(n - accepted),
        "conformal_num_empty_sets": int(empty_sets),
        "conformal_num_ambiguous_sets": int(ambiguous_sets),
    }
    return out


def _uses_risk_control_selective(config: dict | None = None) -> bool:
    config = config or {}
    return bool(config.get("enabled", False)) and str(
        config.get("mode", "threshold")
    ).lower() == "risk_control"


def fit_risk_control_thresholds(
    rows: list[dict[str, Any]], config: dict | None = None
) -> dict[str, Any] | None:
    """Maximize automatic acceptance under a corrected malware-FN risk bound.

    The bounded loss is one only when a malware sample is both accepted and
    predicted benign. Following conformal risk-control calibration, the finite
    sample correction is ``(errors + 1) / (n_malware + 1)``.
    """
    config = config or {}
    if not _uses_risk_control_selective(config):
        return None
    risk_level = float(config.get("risk_level", 0.05))
    if not 0.0 < risk_level < 1.0:
        raise ValueError("selective_prediction.risk_level must be within (0, 1)")
    require_feasible = bool(config.get("require_feasible", False))
    minimum_malware_raw = config.get("min_calibration_malware", 1)
    try:
        minimum_malware = int(minimum_malware_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "selective_prediction.min_calibration_malware must be a positive integer"
        ) from exc
    if (
        isinstance(minimum_malware_raw, bool)
        or minimum_malware < 1
        or float(minimum_malware_raw) != float(minimum_malware)
    ):
        raise ValueError(
            "selective_prediction.min_calibration_malware must be a positive integer"
        )
    minimum_malware_for_feasibility = max(
        1, math.ceil(1.0 / risk_level) - 1
    )
    risk_target = str(
        config.get("risk_target", "malware_fn_rate_after_rejection")
    ).lower()
    # Read legacy experiment files, but always emit the denominator-explicit
    # canonical name. The bounded risk is FN among all malware, not FN among
    # only the accepted malware samples.
    if risk_target == "accepted_malware_false_negative":
        risk_target = "malware_fn_rate_after_rejection"
    if risk_target != "malware_fn_rate_after_rejection":
        raise ValueError(
            "selective_prediction.risk_target currently supports only "
            "'malware_fn_rate_after_rejection'"
        )

    valid: list[tuple[float, int, int]] = []
    for row in rows:
        score = _finite_row_float(row, "acceptance_score")
        try:
            label = int(row["label"])
            pred = int(row["pred"])
        except (KeyError, TypeError, ValueError):
            continue
        if score is not None:
            valid.append((float(score), label, pred))
    if not valid:
        raise ValueError("Risk-control calibration requires finite acceptance scores")

    malware_count = sum(label == 1 for _score, label, _pred in valid)
    if malware_count == 0:
        raise ValueError("Risk-control calibration requires at least one malware sample")
    if malware_count < minimum_malware:
        raise ValueError(
            "Risk-control calibration has "
            f"{malware_count} malware samples, fewer than configured "
            f"min_calibration_malware={minimum_malware}"
        )
    if malware_count < minimum_malware_for_feasibility:
        message = (
            "Risk-control calibration needs at least "
            f"{minimum_malware_for_feasibility} malware samples for the finite-sample "
            f"corrected bound to be feasible at risk_level={risk_level}; "
            f"received {malware_count}"
        )
        if require_feasible:
            raise ValueError(message)
        logger.warning(message)

    scores = sorted({score for score, _label, _pred in valid})
    reject_all_threshold = math.nextafter(max(scores), float("inf"))
    candidates = scores + [reject_all_threshold]
    best: dict[str, Any] | None = None
    for threshold in candidates:
        accepted = [item for item in valid if item[0] >= threshold]
        false_negatives = sum(
            label == 1 and pred == 0 for _score, label, pred in accepted
        )
        corrected_risk = (false_negatives + 1.0) / (malware_count + 1.0)
        if corrected_risk > risk_level:
            continue
        candidate = {
            "threshold": float(threshold),
            "num_accepted": int(len(accepted)),
            "malware_false_negatives": int(false_negatives),
            "empirical_risk": float(false_negatives / malware_count),
            "corrected_risk": float(corrected_risk),
        }
        if best is None or candidate["num_accepted"] > best["num_accepted"]:
            best = candidate

    feasible = best is not None
    if best is None:
        if require_feasible:
            raise RuntimeError(
                "No risk-control threshold satisfies the finite-sample corrected "
                "risk bound while selective_prediction.require_feasible=true"
            )
        best = {
            "threshold": float(reject_all_threshold),
            "num_accepted": 0,
            "malware_false_negatives": 0,
            "empirical_risk": 0.0,
            "corrected_risk": float(1.0 / (malware_count + 1.0)),
        }
    return {
        "mode": "risk_control",
        "risk_target": risk_target,
        "risk_level": risk_level,
        "score_type": str(config.get("threshold_score", "model_acceptance")),
        "num_calibration": int(len(valid)),
        "num_calibration_malware": int(malware_count),
        "min_calibration_malware": int(minimum_malware),
        "minimum_malware_for_feasibility": int(
            minimum_malware_for_feasibility
        ),
        "require_feasible": bool(require_feasible),
        "guarantee_type": "expected_crc",
        "feasible": bool(feasible),
        **best,
    }


def risk_control_selective_metrics(
    rows: list[dict[str, Any]], thresholds: dict[str, Any] | None
) -> dict[str, Any]:
    """Evaluate automatic decisions made by a fitted risk-control threshold."""
    if not thresholds:
        return {}
    threshold = float(thresholds["threshold"])
    valid = []
    for row in rows:
        score = _finite_row_float(row, "acceptance_score")
        try:
            label = int(row["label"])
            pred = int(row["pred"])
        except (KeyError, TypeError, ValueError):
            continue
        if score is not None:
            valid.append((float(score), label, pred))
    if not valid:
        return {}

    accepted = [item for item in valid if item[0] >= threshold]
    accepted_errors = sum(label != pred for _score, label, pred in accepted)
    malware_count = sum(label == 1 for _score, label, _pred in valid)
    accepted_malware = sum(label == 1 for _score, label, _pred in accepted)
    malware_false_negatives = sum(
        label == 1 and pred == 0 for _score, label, pred in accepted
    )

    def _ratio(num: int, den: int) -> float | None:
        return float(num) / float(den) if den > 0 else None

    empirical_risk = _ratio(malware_false_negatives, malware_count)
    risk_level = float(thresholds.get("risk_level", 0.0))
    return {
        "risk_control_threshold": threshold,
        "risk_control_risk_level": risk_level,
        "risk_control_risk_target": str(thresholds.get("risk_target", "")),
        "risk_control_calibration_feasible": bool(thresholds.get("feasible", False)),
        "risk_control_calibration_corrected_risk": float(
            thresholds.get("corrected_risk", 0.0)
        ),
        "risk_control_acceptance_rate": _ratio(len(accepted), len(valid)),
        "risk_control_rejection_rate": _ratio(len(valid) - len(accepted), len(valid)),
        "risk_control_selective_risk": _ratio(accepted_errors, len(accepted)),
        "risk_control_selective_acc": (
            None if not accepted else 1.0 - accepted_errors / len(accepted)
        ),
        "risk_control_malware_fn_rate_after_rejection": empirical_risk,
        "risk_control_accepted_malware_fn_rate": _ratio(
            malware_false_negatives, accepted_malware
        ),
        "risk_control_malware_fn_count": int(malware_false_negatives),
        "risk_control_accepted_malware_count": int(accepted_malware),
        "risk_control_num_accepted": int(len(accepted)),
        "risk_control_num_rejected": int(len(valid) - len(accepted)),
        "risk_control_target_met_empirically": (
            None if empirical_risk is None else bool(empirical_risk <= risk_level)
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp: bool,
    split_name: str,
    dump_rows: bool = False,
    selective_threshold: float | None = None,
    selective_score_type: str = "max_probability",
    classification_threshold: float = 0.5,
):
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    preds_all: list[int] = []
    fixed_preds_all: list[int] = []
    acceptance_all: list[float] = []
    rows: list[dict[str, Any]] = []
    num_failed = 0
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, sids, quality, failed = prepare_robust_batch(batch, device)
        num_failed += failed
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            logits, extra = model(graph, return_features=False)
        prob = torch.softmax(logits.float(), dim=-1)[:, 1]
        pred_fixed = (prob >= 0.5).long()
        pred = (prob >= float(classification_threshold)).long()
        labels_all.extend(labels.detach().cpu().long().tolist())
        probs_all.extend(prob.detach().cpu().tolist())
        preds_all.extend(pred.detach().cpu().long().tolist())
        fixed_preds_all.extend(pred_fixed.detach().cpu().long().tolist())
        model_acceptance = extra.get("acceptance_score")
        acceptance = _batch_selective_score(
            prob,
            extra,
            selective_score_type,
            classification_threshold=classification_threshold,
        )
        acceptance_all.extend(acceptance.detach().float().cpu().view(-1).tolist())
        for key in GATE_DIAGNOSTIC_KEYS:
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                finite = value.detach().float().view(-1)
                finite = finite[torch.isfinite(finite)]
                if finite.numel() > 0:
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(finite.sum().cpu().item())
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(finite.numel())

        if dump_rows:
            gate = extra.get("gate_weights")
            for i, sid in enumerate(sids or []):
                row = {
                    "split": split_name,
                    "sid": sid,
                    "label": int(labels[i].detach().cpu().item()),
                    "prob_malware": float(prob[i].detach().cpu().item()),
                    "pred": int(pred[i].detach().cpu().item()),
                    "pred_fixed_0_5": int(pred_fixed[i].detach().cpu().item()),
                    "classification_threshold": float(classification_threshold),
                    "final_confidence": float(torch.softmax(logits.float(), dim=-1)[i].max().detach().cpu().item()),
                    "correct": int(pred[i].detach().cpu().item() == labels[i].detach().cpu().item()),
                    "correct_fixed_0_5": int(
                        pred_fixed[i].detach().cpu().item()
                        == labels[i].detach().cpu().item()
                    ),
                    "year": int(batch.get("years")[i].detach().cpu().item()) if batch.get("years") is not None else 0,
                }
                row.update(_branch_prediction_row(extra, labels, i))
                acceptance_i = float(acceptance.view(-1)[i].detach().cpu().item())
                if isinstance(model_acceptance, torch.Tensor) and model_acceptance.numel() > i:
                    row["model_acceptance_score"] = float(
                        model_acceptance.view(-1)[i].detach().cpu().item()
                    )
                for row_key, batch_key in (
                    ("api_aug_type", "api_aug_types"),
                    ("graph_aug_type", "graph_aug_types"),
                    ("manifest_aug_type", "manifest_aug_types"),
                ):
                    values = batch.get(batch_key) or []
                    row[row_key] = str(values[i]) if i < len(values) else "none"
                if isinstance(gate, torch.Tensor) and gate.size(0) > i:
                    gate_i = gate[i].detach().cpu()
                    row.update({
                        "w_api": float(gate_i[0].item()),
                        "w_graph": float(gate_i[1].item()),
                        "w_manifest": float(gate_i[2].item()),
                        "w_joint": float(gate_i[3].item()),
                    })
                for key in GATE_DIAGNOSTIC_KEYS:
                    value = extra.get(key)
                    if not isinstance(value, torch.Tensor):
                        value = quality.get(key)
                    if isinstance(value, torch.Tensor) and value.numel() > i:
                        row[key] = float(value.view(-1)[i].detach().cpu().item())
                # GATE_DIAGNOSTIC_KEYS contains the model's internal acceptance
                # diagnostic. Keep the experiment-selected score under the
                # canonical column used for threshold fitting and subset cuts.
                row["acceptance_score"] = acceptance_i
                row["selective_score_type"] = str(selective_score_type)
                if selective_threshold is not None:
                    row["rejected"] = int(acceptance_i < selective_threshold)
                rows.append(row)

    metrics = _metrics(labels_all, probs_all, preds_all)
    fixed_metrics = _metrics(labels_all, probs_all, fixed_preds_all)
    metrics.update(
        {
            "classification_threshold": float(classification_threshold),
            "fixed_0_5_acc": fixed_metrics["acc"],
            "fixed_0_5_macro_f1": fixed_metrics["macro_f1"],
            "fixed_0_5_f1_pos": fixed_metrics["f1_pos"],
            "fixed_0_5_recall_pos": fixed_metrics["recall_pos"],
        }
    )
    metrics["num_failed"] = int(num_failed)
    metrics["num_eval"] = int(len(labels_all))
    if len(acceptance_all) == len(labels_all):
        metrics.update(_selective_ranking_metrics(labels_all, preds_all, acceptance_all))
        if selective_threshold is not None:
            metrics.update(
                _selective_metrics(
                    labels_all,
                    preds_all,
                    acceptance_all,
                    selective_threshold,
                )
            )
    for key, total in diagnostic_sums.items():
        metrics[f"mean_{key}"] = total / max(diagnostic_counts.get(key, 0), 1)
    if rows:
        metrics.update(compute_branch_reliability_metrics(rows))
    return metrics, rows


def _fit_routed_final_temperature(
    log_temperature: torch.nn.Parameter,
    raw_log_prob: torch.Tensor,
    temperature_labels: torch.Tensor,
) -> dict[str, Any]:
    """Fit the final scalar temperature without changing its freeze policy."""
    if log_temperature.numel() != 1:
        raise ValueError("Routed final-temperature fitting requires one scalar")
    with torch.no_grad():
        log_temperature.zero_()
        nll_before = float(F.nll_loss(raw_log_prob, temperature_labels).item())

    previous_requires_grad = bool(log_temperature.requires_grad)
    try:
        # Encoder training intentionally freezes post-hoc-only parameters. LBFGS
        # still needs this scalar to require gradients during its own stage.
        log_temperature.requires_grad_(True)
        temperature_optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=1.0,
            max_iter=50,
            line_search_fn="strong_wolfe",
        )

        def _temperature_closure() -> torch.Tensor:
            temperature_optimizer.zero_grad(set_to_none=True)
            calibrated = F.log_softmax(
                raw_log_prob / log_temperature.exp(),
                dim=-1,
            )
            objective = F.nll_loss(calibrated, temperature_labels)
            if not bool(torch.isfinite(objective.detach()).item()):
                raise FloatingPointError(
                    "Non-finite routed final-temperature objective"
                )
            objective.backward()
            return objective

        temperature_optimizer.step(_temperature_closure)
    finally:
        log_temperature.grad = None
        log_temperature.requires_grad_(previous_requires_grad)

    with torch.no_grad():
        fitted_temperature = float(log_temperature.exp().item())
        calibrated = F.log_softmax(
            raw_log_prob / log_temperature.exp(),
            dim=-1,
        )
        nll_after = float(F.nll_loss(calibrated, temperature_labels).item())
        if nll_after > nll_before + 1.0e-8:
            # Temperature scaling is calibration-only. Retain identity when
            # numerical optimization does not improve its fitting objective.
            log_temperature.zero_()
            fitted_temperature = 1.0
            nll_after = nll_before
    if not math.isfinite(fitted_temperature) or fitted_temperature <= 0.0:
        raise RuntimeError("Routed final temperature is non-finite or non-positive")
    return {
        "enabled": True,
        "temperature": fitted_temperature,
        "num_clean_samples": int(temperature_labels.numel()),
        "nll_before": nll_before,
        "nll_after": nll_after,
    }


def fit_posthoc_calibration(
    model,
    loaders: list,
    device,
    use_amp: bool,
    cfg: dict,
) -> dict[str, Any]:
    """Fit reliability, routing, and optional temperature parameters post hoc."""
    calibration_cfg = cfg.get("calibration", {}) or {}
    if not bool(calibration_cfg.get("enabled", False)):
        return {"enabled": False}
    if str(getattr(model, "fusion_mode", "")) != "discount_probability":
        raise ValueError(
            "calibration.enabled=true requires model fusion_mode=discount_probability"
        )
    parameters = list(model.calibration_parameters())
    if not parameters:
        raise ValueError(
            "calibration.enabled=true requires fusion reliability/probability calibration modules"
        )
    epochs = int(calibration_cfg.get("epochs", 20))
    patience = int(calibration_cfg.get("patience", 4))
    min_delta = float(calibration_cfg.get("min_delta", 1.0e-5))
    if epochs <= 0:
        raise ValueError("calibration.epochs must be positive")
    if patience < 0:
        raise ValueError("calibration.patience must be non-negative")
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError("calibration.min_delta must be finite and non-negative")

    def _scenario_group(item: dict[str, Any], name: str, index: int) -> str:
        explicit = item.get("scenario_group")
        if explicit:
            return str(explicit).lower()
        perturb_type = item.get("perturb_type")
        if perturb_type:
            perturb_name = str(perturb_type).lower()
            return "missing" if perturb_name.endswith("_missing") else perturb_name
        return "clean" if index == 0 or name == "clean" else "other"

    calibration_sources: list[dict[str, Any]] = []
    for loader_index, item in enumerate(loaders):
        if isinstance(item, dict):
            loader = item.get("loader")
            if loader is None:
                raise ValueError("calibration source is missing its loader")
            source_name = str(item.get("name") or f"calibration_{loader_index}")
            configured_branches = (
                item["reliability_branches"]
                if "reliability_branches" in item
                else ("api", "graph", "manifest")
            )
            source = {
                "loader": loader,
                "name": source_name,
                "scenario_group": _scenario_group(
                    item, source_name, loader_index
                ),
                "reliability_branches": tuple(
                    str(name).lower()
                    for name in configured_branches
                ),
            }
        else:
            source = {
                "loader": item,
                "name": "clean" if loader_index == 0 else f"calibration_{loader_index}",
                "scenario_group": "clean" if loader_index == 0 else "other",
                "reliability_branches": ("api", "graph", "manifest"),
            }
        calibration_sources.append(source)

    # Cache the frozen encoders' branch logits and observable evidence once.
    # Calibration then optimizes only the small decision module instead of
    # repeating the API Transformer and GNN forward pass for every epoch.
    cached_batches: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for loader_index, source in enumerate(calibration_sources):
            loader = source["loader"]
            for batch in tqdm(loader, desc="cache calibration", leave=False):
                graph, labels, _, _quality, _failed = prepare_robust_batch(batch, device)
                if graph is None:
                    continue
                with get_amp_context(device, use_amp):
                    _logits, extra = model(graph, return_features=False)
                evidence = extra.get("gate_evidence")
                branch_logits = {
                    name: extra.get(f"{name}_logits_aux")
                    for name in ("api", "graph", "manifest", "joint")
                }
                if not isinstance(evidence, torch.Tensor) or any(
                    not isinstance(value, torch.Tensor) for value in branch_logits.values()
                ):
                    raise RuntimeError(
                        "Post-hoc calibration cache requires observable evidence and all branch logits"
                    )
                cached_batches.append(
                    {
                        "labels": labels.detach(),
                        "evidence": evidence.detach().float(),
                        "loader_index": int(loader_index),
                        "scenario_name": source["name"],
                        "scenario_group": source["scenario_group"],
                        "reliability_branches": source["reliability_branches"],
                        "branch_logits": {
                            name: value.detach().float()
                            for name, value in branch_logits.items()
                        },
                    }
                )
    if not cached_batches:
        raise RuntimeError("Post-hoc calibration received no valid batches")
    num_encoder_batches_cached = len(cached_batches)
    merged_cached_batches: list[dict[str, Any]] = []
    for loader_index, source in enumerate(calibration_sources):
        source_batches = [
            item for item in cached_batches if item["loader_index"] == loader_index
        ]
        if not source_batches:
            continue
        merged_cached_batches.append(
            {
                "labels": torch.cat(
                    [item["labels"] for item in source_batches], dim=0
                ),
                "evidence": torch.cat(
                    [item["evidence"] for item in source_batches], dim=0
                ),
                "loader_index": loader_index,
                "scenario_name": source["name"],
                "scenario_group": source["scenario_group"],
                "reliability_branches": source["reliability_branches"],
                "branch_logits": {
                    name: torch.cat(
                        [item["branch_logits"][name] for item in source_batches],
                        dim=0,
                    )
                    for name in ("api", "graph", "manifest", "joint")
                },
            }
        )
    cached_batches = merged_cached_batches

    # The routed path consumes the integrity modifier during calibration. Fit
    # its reference from the clean loader before optimizing the router.
    routing_visible_reference: list[float] | None = None
    routing_visible_reference_mode: str | None = None
    routing_visible_reference_count = 0
    modifier_cfg = ((cfg.get("fusion", {}) or {}).get("visible_integrity_modifier", {}) or {})
    discount_fusion = getattr(model, "discount_fusion", None)
    if (
        bool(modifier_cfg.get("enabled", False))
        and str(getattr(discount_fusion, "combination", "")) == "routed"
    ):
        min_reference = float(modifier_cfg.get("min_reference", 1.0e-6))
        if not math.isfinite(min_reference) or not 0.0 < min_reference <= 1.0:
            raise ValueError(
                "fusion.visible_integrity_modifier.min_reference must be within (0, 1]"
            )
        mode = str(modifier_cfg.get("mode", "bounded_visibility")).lower()
        if mode not in {"bounded_visibility", "relative_effective"}:
            raise ValueError(
                "fusion.visible_integrity_modifier.mode must be "
                "'bounded_visibility' or 'relative_effective'"
            )
        clean_evidence = [
            item["evidence"] for item in cached_batches if item["loader_index"] == 0
        ]
        if clean_evidence:
            merged = torch.cat(clean_evidence, dim=0)
            routing_visible_reference_count = int(merged.size(0))

            def _evidence_column(index: int) -> torch.Tensor:
                if merged.size(-1) <= index:
                    return torch.ones(merged.size(0), device=merged.device)
                return merged[:, index].float().clamp(0.0, 1.0)

            def _median(values: torch.Tensor) -> float:
                values = values[torch.isfinite(values)]
                value = (
                    float(torch.quantile(values, 0.5).item())
                    if values.numel()
                    else 1.0
                )
                return min(1.0, max(min_reference, value))

            if mode == "relative_effective":
                routing_visible_reference = [
                    _median(
                        _evidence_column(EvidenceIndex.API_INTEGRITY)
                        * _evidence_column(EvidenceIndex.API_ENCODER_COVERAGE)
                    ),
                    _median(
                        _evidence_column(EvidenceIndex.GRAPH_INTEGRITY)
                        * _evidence_column(EvidenceIndex.GRAPH_ENCODER_COVERAGE)
                    ),
                    _median(_evidence_column(EvidenceIndex.MANIFEST_INTEGRITY)),
                ]
            else:
                routing_visible_reference = [
                    _median(_evidence_column(EvidenceIndex.API_ENCODER_COVERAGE)),
                    _median(_evidence_column(EvidenceIndex.GRAPH_ENCODER_COVERAGE)),
                    1.0,
                ]
            routing_visible_reference_mode = mode
            discount_fusion.set_visible_integrity_reference(
                routing_visible_reference, enabled=True
            )

    model.set_calibration_active(True)
    previous_requires_grad = {id(param): param.requires_grad for param in model.parameters()}
    learning_rate = float(calibration_cfg.get("lr", 1.0e-3))
    weight_decay = float(calibration_cfg.get("weight_decay", 0.0))
    grad_clip = float(calibration_cfg.get("grad_clip", 5.0))

    def _set_trainable(stage_parameters: list[torch.nn.Parameter]) -> None:
        trainable_ids = {id(param) for param in stage_parameters}
        for param in model.parameters():
            param.requires_grad_(id(param) in trainable_ids)

    def _forward_cached(cached: dict[str, Any]) -> dict[str, torch.Tensor]:
        branch_logits = cached["branch_logits"]
        evidence = cached["evidence"]
        outputs = model.discount_fusion(
            branch_logits["api"],
            branch_logits["graph"],
            branch_logits["manifest"],
            branch_logits["joint"],
            evidence,
        )
        outputs.update(
            {
                f"{name}_logits_aux": value
                for name, value in branch_logits.items()
            }
        )
        outputs["gate_evidence"] = evidence
        return outputs

    def _mean_cached_loss(
        items: list[dict[str, Any]],
        loss_fn,
    ) -> torch.Tensor:
        if not items:
            raise RuntimeError("Calibration objective group contains no scenarios")
        values = [loss_fn(item) for item in items]
        return torch.stack(values).mean()

    def _balanced_group_loss(
        group: dict[str, Any],
        loss_fn,
    ) -> torch.Tensor:
        clean_loss = _mean_cached_loss(group["clean"], loss_fn)
        scenario_items = group.get("scenario") or []
        if not scenario_items:
            return clean_loss
        scenario_loss = _mean_cached_loss(scenario_items, loss_fn)
        # Clean and one degradation family form the two equally important
        # deployment conditions. Averaging them prevents a large perturbation
        # grid from changing the implicit deployment prior.
        return 0.5 * (clean_loss + scenario_loss)

    def _optimize_stage(
        stage_name: str,
        stage_parameters: list[torch.nn.Parameter],
        objective_groups: list[dict[str, Any]],
        objective_fn,
    ) -> dict[str, Any]:
        if not stage_parameters or not objective_groups:
            return {
                "enabled": False,
                "name": stage_name,
                "epochs_ran": 0,
                "best_epoch": 0,
                "stopped_early": False,
                "losses": [],
                "final_loss": None,
                "total_steps": 0,
                "objective_groups": [],
            }
        optimizer = torch.optim.Adam(
            stage_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        _set_trainable(stage_parameters)
        epoch_losses: list[float] = []
        total_steps = 0
        best_loss = float("inf")
        best_epoch = 0
        best_parameters: list[torch.Tensor] | None = None
        stale_epochs = 0
        stopped_early = False
        for epoch in range(1, epochs + 1):
            total = 0.0
            steps = 0
            for group in objective_groups:
                optimizer.zero_grad(set_to_none=True)
                loss = objective_fn(group)
                if not bool(torch.isfinite(loss.detach()).all().item()):
                    raise FloatingPointError(
                        f"Non-finite {stage_name} loss at "
                        f"epoch={epoch} step={steps + 1}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stage_parameters, grad_clip)
                optimizer.step()
                total += float(loss.detach().item())
                steps += 1
                total_steps += 1
            epoch_loss = total / max(steps, 1)
            epoch_losses.append(epoch_loss)
            logger.info(
                "posthoc_calibration stage=%s epoch=%s loss=%.6f",
                stage_name,
                epoch,
                epoch_loss,
            )
            if steps > 0 and epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                best_epoch = epoch
                best_parameters = [
                    param.detach().clone() for param in stage_parameters
                ]
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    stopped_early = True
                    break
        if total_steps == 0 or best_parameters is None:
            raise RuntimeError(
                f"Post-hoc calibration stage {stage_name} produced no valid step"
            )
        with torch.no_grad():
            for parameter, best_value in zip(stage_parameters, best_parameters):
                parameter.copy_(best_value)
        return {
            "enabled": True,
            "name": stage_name,
            "epochs_ran": len(epoch_losses),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "losses": epoch_losses,
            "final_loss": best_loss,
            "total_steps": total_steps,
            "objective_groups": [str(group["name"]) for group in objective_groups],
        }

    clean_cached = [
        item for item in cached_batches if item["scenario_group"] == "clean"
    ]
    if not clean_cached:
        raise RuntimeError("Post-hoc calibration requires a clean calibration source")
    nonclean_by_group: dict[str, list[dict[str, Any]]] = {}
    for item in cached_batches:
        group_name = str(item["scenario_group"])
        if group_name != "clean":
            nonclean_by_group.setdefault(group_name, []).append(item)

    reliability_parameters = (
        list(discount_fusion.reliability_calibration_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "reliability_calibration_parameters")
        else []
    )
    probability_parameters = (
        list(discount_fusion.probability_calibration_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "probability_calibration_parameters")
        else []
    )
    routing_parameters = (
        list(discount_fusion.routing_calibration_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "routing_calibration_parameters")
        else []
    )
    stage_summaries: dict[str, Any] = {}
    try:
        model.eval()
        final_temperature_parameters = (
            discount_fusion.final_temperature_parameters()
            if discount_fusion is not None
            and hasattr(discount_fusion, "final_temperature_parameters")
            else []
        )
        with torch.no_grad():
            for parameter in final_temperature_parameters:
                parameter.zero_()

        if reliability_parameters:
            reliability_cfg = dict(
                ((cfg.get("fusion", {}) or {}).get("reliability_calibration", {}) or {})
            )
            # Estimate each branch's base correctness on the natural clean
            # distribution. Deployment degradation is handled separately by
            # the explicit relative-effective-integrity modifier and router.
            reliability_branches = tuple(
                str(name).lower()
                for name in getattr(
                    discount_fusion,
                    "reliability_calibration_branches",
                    ("api", "graph", "manifest"),
                )
            )
            reliability_groups = [
                {
                    "name": f"{branch}:clean",
                    "branch": branch,
                    "clean": clean_cached,
                    "scenario": [],
                }
                for branch in reliability_branches
            ]

            def _reliability_objective(group: dict[str, Any]) -> torch.Tensor:
                branch_cfg = {**reliability_cfg, "branches": [group["branch"]]}

                def _loss(cached: dict[str, Any]) -> torch.Tensor:
                    outputs = _forward_cached(cached)
                    loss, _ = compute_reliability_calibration_loss(
                        outputs,
                        cached["labels"],
                        cached["evidence"],
                        branch_cfg,
                    )
                    return loss

                return _balanced_group_loss(group, _loss)

            stage_summaries["reliability"] = _optimize_stage(
                "reliability",
                reliability_parameters,
                reliability_groups,
                _reliability_objective,
            )

        if probability_parameters:
            probability_groups = [
                {
                    "name": "probability:clean",
                    "clean": clean_cached,
                    "scenario": [],
                }
            ]

            def _probability_objective(group: dict[str, Any]) -> torch.Tensor:
                def _loss(cached: dict[str, Any]) -> torch.Tensor:
                    outputs = _forward_cached(cached)
                    loss, _ = compute_probability_calibration_loss(
                        outputs,
                        cached["labels"],
                        cached["evidence"],
                    )
                    return loss

                return _balanced_group_loss(group, _loss)

            stage_summaries["probability"] = _optimize_stage(
                "probability",
                probability_parameters,
                probability_groups,
                _probability_objective,
            )

        if routing_parameters:
            routing_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
            routing_cfg.setdefault("reliability_calibration", {})["weight"] = 0.0
            routing_cfg.setdefault("probability_calibration", {})["weight"] = 0.0
            routing_groups = [
                {
                    "name": f"router:{group_name}",
                    "clean": clean_cached,
                    "scenario": items,
                }
                for group_name, items in sorted(nonclean_by_group.items())
            ]
            if not routing_groups:
                routing_groups = [
                    {
                        "name": "router:clean",
                        "clean": clean_cached,
                        "scenario": [],
                    }
                ]

            def _routing_objective(group: dict[str, Any]) -> torch.Tensor:
                def _loss(cached: dict[str, Any]) -> torch.Tensor:
                    outputs = _forward_cached(cached)
                    loss, _ = compute_posthoc_calibration_loss(
                        outputs,
                        cached["labels"],
                        cached["evidence"],
                        routing_cfg,
                        reliability_branches=cached["reliability_branches"],
                    )
                    return loss

                return _balanced_group_loss(group, _loss)

            stage_summaries["routing"] = _optimize_stage(
                "routing",
                routing_parameters,
                routing_groups,
                _routing_objective,
            )

        handled_parameter_ids = {
            id(parameter)
            for stage_parameters in (
                reliability_parameters,
                probability_parameters,
                routing_parameters,
            )
            for parameter in stage_parameters
        }
        unhandled_parameters = [
            parameter
            for parameter in parameters
            if id(parameter) not in handled_parameter_ids
        ]
        if unhandled_parameters:
            raise RuntimeError(
                "Post-hoc calibration contains parameters that are not assigned "
                "to reliability, probability, or routing stages"
            )
    finally:
        for param in model.parameters():
            param.requires_grad_(previous_requires_grad[id(param)])

    enabled_stages = [
        summary for summary in stage_summaries.values() if summary.get("enabled")
    ]
    if not enabled_stages:
        model.set_calibration_active(False)
        raise RuntimeError("Post-hoc calibration did not run any optimization stage")
    epoch_losses = list(enabled_stages[-1]["losses"])
    total_steps = int(sum(stage["total_steps"] for stage in enabled_stages))
    best_loss = float(enabled_stages[-1]["final_loss"])
    aggregate_final_loss = float(
        sum(float(stage["final_loss"]) for stage in enabled_stages)
    )
    best_epoch = int(enabled_stages[-1]["best_epoch"])
    stopped_early = bool(any(stage["stopped_early"] for stage in enabled_stages))

    final_temperature_summary: dict[str, Any] = {"enabled": False}
    final_temperature_parameters = (
        discount_fusion.final_temperature_parameters()
        if discount_fusion is not None
        and hasattr(discount_fusion, "final_temperature_parameters")
        else []
    )
    if final_temperature_parameters:
        # Fit one scalar only on the clean post-hoc subset (loader 0). The
        # reported clean probabilities must not be calibrated on synthetic
        # degradation scenarios.
        clean_log_probs: list[torch.Tensor] = []
        clean_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for cached in cached_batches:
                if int(cached["loader_index"]) != 0:
                    continue
                branch_logits = cached["branch_logits"]
                routed = model.discount_fusion(
                    branch_logits["api"],
                    branch_logits["graph"],
                    branch_logits["manifest"],
                    branch_logits["joint"],
                    cached["evidence"],
                )
                raw_log_prob = routed.get("uncalibrated_final_log_prob")
                if not isinstance(raw_log_prob, torch.Tensor):
                    raise RuntimeError(
                        "Routed final-temperature calibration requires "
                        "uncalibrated_final_log_prob"
                    )
                clean_log_probs.append(raw_log_prob.detach().float())
                clean_labels.append(cached["labels"].detach().long())
        if not clean_log_probs:
            raise RuntimeError(
                "Routed final-temperature calibration received no clean samples"
            )
        raw_log_prob = torch.cat(clean_log_probs, dim=0)
        temperature_labels = torch.cat(clean_labels, dim=0)
        log_temperature = final_temperature_parameters[0]
        final_temperature_summary = _fit_routed_final_temperature(
            log_temperature,
            raw_log_prob,
            temperature_labels,
        )
    temperatures = {}
    for name in ("api", "graph", "manifest", "joint"):
        if model.discount_fusion.temperature_parameters is not None:
            temperatures[name] = float(
                (torch.nn.functional.softplus(
                    model.discount_fusion.temperature_parameters[name].detach()
                ) + 1.0e-4).cpu().item()
            )
    if bool(final_temperature_summary.get("enabled", False)):
        temperatures["final"] = float(final_temperature_summary["temperature"])
    return {
        "enabled": True,
        "strategy": "staged_scenario_balanced",
        "epochs": epochs,
        "epochs_ran": len(epoch_losses),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "losses": epoch_losses,
        "final_loss": best_loss,
        "aggregate_final_loss": aggregate_final_loss,
        "total_optimization_steps": total_steps,
        "stages": stage_summaries,
        "num_input_loaders": len(calibration_sources),
        "calibration_sources": [
            {
                "name": source["name"],
                "scenario_group": source["scenario_group"],
                "reliability_branches": list(source["reliability_branches"]),
            }
            for source in calibration_sources
        ],
        "num_cached_batches": len(cached_batches),
        "num_encoder_batches_cached": num_encoder_batches_cached,
        "num_cached_samples": int(
            sum(int(item["labels"].numel()) for item in cached_batches)
        ),
        "routing_visible_reference": routing_visible_reference,
        "routing_visible_reference_mode": routing_visible_reference_mode,
        "routing_visible_reference_count": routing_visible_reference_count,
        "temperatures": temperatures,
        "final_temperature": final_temperature_summary,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int):
    model.train()
    use_amp = bool(cfg["train"].get("use_amp", True))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg["label_smoothing"] = float(cfg["train"].get("label_smoothing", loss_cfg.get("label_smoothing", 0.0)))
    optimizer.zero_grad(set_to_none=True)
    total_loss: torch.Tensor | None = None
    loss_part_sums: dict[str, torch.Tensor] = {}
    diagnostic_sums: dict[str, torch.Tensor] = {}
    diagnostic_counts: dict[str, int] = {}
    steps = 0
    failed_seen = 0
    valid_seen = 0

    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        graph, labels, _, _quality, failed = prepare_robust_batch(batch, device)
        failed_seen += int(failed)
        if graph is None:
            continue
        valid_seen += int(labels.size(0))
        with get_amp_context(device, use_amp):
            logits, extra = model(graph, return_features=False)
            loss, parts = compute_robust_loss(
                logits,
                labels,
                extra,
                loss_cfg,
                evidence=extra.get("gate_evidence"),
                epoch=epoch,
                materialize_diagnostics=False,
            )
            loss = loss / max(grad_accum, 1)
        if not bool(torch.isfinite(loss.detach()).all().item()):
            diagnostic_parts = {}
            for key, value in parts.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    diagnostic_parts[key] = float(value.detach().item())
                elif isinstance(value, (int, float)):
                    diagnostic_parts[key] = float(value)
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch} step={steps + 1}; "
                f"loss_parts={diagnostic_parts}"
            )
        for key in TRAIN_LOG_DIAGNOSTIC_KEYS:
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                finite = value.detach().float().view(-1)
                finite = finite[torch.isfinite(finite)]
                if finite.numel() > 0:
                    value_sum = finite.sum()
                    if key in diagnostic_sums:
                        diagnostic_sums[key] = diagnostic_sums[key] + value_sum
                    else:
                        diagnostic_sums[key] = value_sum
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(finite.numel())
        for key, value in parts.items():
            if key not in {"loss", "ce", "branch_aux", "branch_aux_weight"}:
                continue
            value_tensor = (
                value.detach()
                if isinstance(value, torch.Tensor)
                else loss.detach().new_tensor(float(value))
            )
            if key in loss_part_sums:
                loss_part_sums[key] = loss_part_sums[key] + value_tensor
            else:
                loss_part_sums[key] = value_tensor
        steps += 1
        scaler.scale(loss).backward()
        if steps % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        unscaled_loss = loss.detach() * max(grad_accum, 1)
        total_loss = unscaled_loss if total_loss is None else total_loss + unscaled_loss

    if steps > 0 and steps % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    enforce_failed_ratio({"num_eval": valid_seen, "num_failed": failed_seen}, cfg, f"train_epoch_{epoch}")
    if steps > 0:
        logger.info(
            "train_loss_parts epoch=%s %s",
            epoch,
            " ".join(
                f"{key}={float((value / steps).item()):.4f}"
                for key, value in sorted(loss_part_sums.items())
                if key in {"loss", "ce", "branch_aux", "branch_aux_weight"}
            ),
        )
        logger.info(
            "train_fusion_diagnostics epoch=%s %s",
            epoch,
            " ".join(
                f"mean_{key}={float((value / max(diagnostic_counts.get(key, 0), 1)).item()):.4f}"
                for key, value in sorted(diagnostic_sums.items())
                if key.startswith(("discount_", "fusion_weight_", "entropy_", "margin_", "uncertainty_proxy_"))
                or key == "zero_weight_fallback_used"
            ),
        )
    if total_loss is None:
        return 0.0
    return float((total_loss / max(steps, 1)).item())


def write_gate_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_risk_coverage_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build exact threshold-achievable risk/coverage operating points by split."""
    grouped: dict[str, list[tuple[float, int, int]]] = {}
    for row in rows:
        score = _finite_row_float(row, "acceptance_score")
        try:
            label = int(row["label"])
            pred = int(row["pred"])
        except (KeyError, TypeError, ValueError):
            continue
        if score is None or label not in {0, 1} or pred not in {0, 1}:
            continue
        split = str(row.get("split") or "unknown")
        grouped.setdefault(split, []).append((float(score), label, pred))

    curve: list[dict[str, Any]] = []
    for split, items in sorted(grouped.items()):
        items.sort(key=lambda item: item[0], reverse=True)
        total = len(items)
        total_malware = sum(label == 1 for _score, label, _pred in items)
        tp = fp = tn = fn = 0
        accepted = 0
        index = 0
        while index < total:
            threshold = items[index][0]
            while index < total and items[index][0] == threshold:
                _score, label, pred = items[index]
                accepted += 1
                if label == 1 and pred == 1:
                    tp += 1
                elif label == 1 and pred == 0:
                    fn += 1
                elif label == 0 and pred == 1:
                    fp += 1
                else:
                    tn += 1
                index += 1

            errors = fp + fn
            accepted_malware = tp + fn
            f1_malware_den = 2 * tp + fp + fn
            f1_benign_den = 2 * tn + fp + fn
            f1_malware = 2 * tp / f1_malware_den if f1_malware_den else 0.0
            f1_benign = 2 * tn / f1_benign_den if f1_benign_den else 0.0
            curve.append(
                {
                    "split": split,
                    "acceptance_threshold": float(threshold),
                    "num_total": int(total),
                    "num_accepted": int(accepted),
                    "coverage": float(accepted / total),
                    "selective_risk": float(errors / accepted),
                    "selective_accuracy": float(1.0 - errors / accepted),
                    "selective_macro_f1": float((f1_benign + f1_malware) / 2.0),
                    "malware_fn_rate_after_rejection": (
                        float(fn / total_malware) if total_malware else None
                    ),
                    "accepted_malware_fn_rate": (
                        float(fn / accepted_malware) if accepted_malware else None
                    ),
                }
            )
    return curve


def write_risk_coverage_curve(path: Path, rows: list[dict[str, Any]]) -> int:
    curve = build_risk_coverage_curve(rows)
    if not curve:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(curve[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)
    return len(curve)


def _normalize_extra_eval_sets(raw_sets: Any) -> list[dict[str, Any]]:
    if not raw_sets:
        return []
    if isinstance(raw_sets, dict):
        return [
            {"name": str(name), **(value if isinstance(value, dict) else {})}
            for name, value in raw_sets.items()
        ]
    if isinstance(raw_sets, list):
        out = []
        for idx, item in enumerate(raw_sets):
            if not isinstance(item, dict):
                raise ValueError(f"eval.extra_sets[{idx}] must be a mapping")
            out.append(dict(item))
        return out
    raise ValueError("eval.extra_sets must be a list or mapping")


def _extra_eval_paths(cfg: dict, item: dict[str, Any]) -> tuple[str, str]:
    data_cfg = cfg.get("data", {}) or {}
    root = str(item.get("root", data_cfg.get("root", "")) or "")
    # Curated subsets of the ordinary test split should inherit the configured
    # test PT pool. External datasets can still override this with item.pt_dir.
    pt_value = (
        item.get("pt_dir")
        or item.get("test_pt_dir")
        or data_cfg.get("test_pt_dir")
    )
    csv_value = item.get("csv") or item.get("csv_path") or item.get("test_csv")
    if not pt_value:
        raise ValueError(
            f"extra eval set {item.get('name', '<unnamed>')} is missing pt_dir "
            "and data.test_pt_dir is not configured"
        )
    if not csv_value:
        raise ValueError(f"extra eval set {item.get('name', '<unnamed>')} is missing csv")
    return resolve(root, pt_value), resolve(root, csv_value)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


_RUN_SPECIFIC_CONFIG_KEYS = {
    "seed",
    "exp_name",
    "output_dir",
    "out_dir",
    "output_name",
    "checkpoint_path",
    "resume_checkpoint",
    "device",
    "num_workers",
    "eval_num_workers",
    "root",
    "train_pt_dir",
    "val_pt_dir",
    "test_pt_dir",
    "multiprocessing_sharing_strategy",
    "prefetch_factor",
    "persistent_workers",
    "pin_memory",
}

_METHOD_IMPLEMENTATION_FILES = (
    "constants.py",
    "dataset.py",
    "evidence.py",
    "gates.py",
    "graph_encoders.py",
    "manifest_features.py",
    "perturbations.py",
    "pt_schema.py",
    "quality.py",
    "reliability_calibration.py",
    "semantic_categories.py",
    "evidential.py",
    "opinion_router.py",
    "discount_fusion.py",
    "model.py",
    "losses.py",
    "train.py",
    "utils.py",
)


def _method_protocol_config(value: Any) -> Any:
    """Remove run-location fields while retaining method/protocol choices."""
    if isinstance(value, dict):
        return {
            key: _method_protocol_config(item)
            for key, item in value.items()
            if key not in _RUN_SPECIFIC_CONFIG_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_method_protocol_config(item) for item in value]
    return _json_compatible(value)


def _method_implementation_sha256() -> str:
    """Fingerprint the model, fusion, loss, and evaluation implementation."""
    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _METHOD_IMPLEMENTATION_FILES:
        path = source_dir / name
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_identity(cfg: dict, experiment_name: str, seed: int) -> dict[str, Any]:
    """Record the resolved method identity beside every summary.

    Result directories can outlive later YAML edits. A stable fingerprint and
    the decision-critical switches prevent stale summaries from being mistaken
    for results produced by the current method definition.
    """
    normalized = _json_compatible(cfg)
    serialized = yaml.safe_dump(
        normalized,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    protocol_config = _method_protocol_config(normalized)
    if isinstance(protocol_config, dict):
        # Seed wrappers use distinct display names for the same method.
        protocol_config.pop("method", None)
    protocol_serialized = yaml.safe_dump(
        protocol_config,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    fusion_cfg = cfg.get("fusion", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    calibration_cfg = cfg.get("calibration", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    selective_cfg = cfg.get("selective_prediction", {}) or {}
    classification_cfg = cfg.get("classification_threshold", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    reliability_enabled = bool(reliability_cfg.get("enabled", False))
    combination_rule = str(fusion_cfg.get("combination", "linear")).lower()
    routing_enabled = bool(
        combination_rule == "routed" and routing_cfg.get("enabled", False)
    )
    routing_target_loss_weight = float(
        routing_cfg.get("target_loss_weight", 1.0)
    )
    routing_prediction_loss_weight = float(
        routing_cfg.get(
            "prediction_loss_weight",
            1.0
            if bool(routing_cfg.get("use_fused_prediction_loss", False))
            else 0.0,
        )
    )
    routing_acceptance_mode = str(
        routing_cfg.get("acceptance_score_mode", "product")
    ).lower()
    if routing_acceptance_mode == "current_product":
        routing_acceptance_mode = "product"
    use_reliability_discount = bool(
        fusion_cfg.get("use_reliability_discount", True)
    )
    visible_modifier_enabled = bool(
        (fusion_cfg.get("visible_integrity_modifier", {}) or {}).get(
            "enabled", False
        )
    )
    legacy_integrity_aux = loss_cfg.get("reliability_weighted_aux")
    integrity_weighted_aux_enabled = (
        bool(legacy_integrity_aux)
        if legacy_integrity_aux is not None
        else bool(loss_cfg.get("integrity_weighted_aux", False))
    )
    selective_enabled = bool(selective_cfg.get("enabled", False))
    classification_enabled = bool(classification_cfg.get("enabled", False))
    selective_mode = (
        str(selective_cfg.get("mode", "threshold")).lower()
        if selective_enabled
        else "disabled"
    )
    return {
        "method_name": str((cfg.get("method", {}) or {}).get("name", experiment_name)),
        "experiment_name": str(experiment_name),
        "seed": int(seed),
        "resolved_config_sha256": hashlib.sha256(serialized).hexdigest(),
        "method_protocol_sha256": hashlib.sha256(protocol_serialized).hexdigest(),
        "method_implementation_sha256": _method_implementation_sha256(),
        "model_fusion_mode": str((cfg.get("model", {}) or {}).get("fusion_mode", "")),
        "combination_rule": combination_rule,
        "global_opinion_routing_enabled": routing_enabled,
        "routing_mode": (
            str(routing_cfg.get("mode", "learned")).lower()
            if routing_enabled
            else "disabled"
        ),
        "routing_disagreement_enabled": bool(
            routing_enabled and routing_cfg.get("use_disagreement", True)
        ),
        "routing_target_loss_weight": (
            routing_target_loss_weight if routing_enabled else None
        ),
        "routing_prediction_loss_weight": (
            routing_prediction_loss_weight if routing_enabled else None
        ),
        "routing_acceptance_score_mode": (
            routing_acceptance_mode if routing_enabled else "disabled"
        ),
        "routing_initial_known_retention": (
            float(routing_cfg.get("initial_known_retention", 0.99))
            if routing_enabled
            else None
        ),
        "routing_posthoc_fused_prediction_loss_enabled": bool(
            routing_enabled and routing_prediction_loss_weight > 0.0
        ),
        "routed_final_temperature_enabled": bool(
            routing_enabled
            and routing_cfg.get("final_temperature_scaling", False)
        ),
        "routed_final_temperature_override": (
            None
            if eval_cfg.get("final_temperature_override") is None
            else float(eval_cfg["final_temperature_override"])
        ),
        "reliability_calibration_enabled": reliability_enabled,
        "reliability_use_enabled": use_reliability_discount,
        "visible_integrity_modifier_enabled": visible_modifier_enabled,
        "integrity_weighted_aux_enabled": integrity_weighted_aux_enabled,
        "reliability_calibration_branches": (
            list(reliability_cfg.get("branches", BRANCH_NAMES))
            if reliability_enabled
            else []
        ),
        "router_trained_end_to_end": bool(
            routing_enabled and routing_cfg.get("train_end_to_end", True)
        ),
        "router_posthoc_refinement_enabled": bool(
            routing_enabled
            and calibration_cfg.get("enabled", False)
            and routing_cfg.get("posthoc_refine", True)
        ),
        "router_encoder_training_reliability_source": (
            "observable_integrity"
            if routing_enabled and use_reliability_discount
            else ("neutral_constant" if routing_enabled else "disabled")
        ),
        "router_posthoc_reliability_source": (
            "calibrated_branch_correctness"
            if routing_enabled
            and use_reliability_discount
            and reliability_enabled
            else (
                "observable_integrity"
                if routing_enabled and use_reliability_discount
                else ("neutral_constant" if routing_enabled else "disabled")
            )
        ),
        "relation_evidence_enabled": bool(
            reliability_enabled
            and reliability_cfg.get("use_relation_evidence", False)
        ),
        "evidential_certainty_enabled": bool(
            reliability_enabled
            and reliability_cfg.get("use_evidential_uncertainty", False)
        ),
        "model_visibility_reliability_enabled": bool(
            reliability_enabled and reliability_cfg.get("use_model_visibility", False)
        ),
        "classification_threshold_enabled": classification_enabled,
        "classification_threshold_objective": (
            str(classification_cfg.get("objective", "macro_f1"))
            if classification_enabled
            else "disabled"
        ),
        "classification_min_malware_recall": (
            float(classification_cfg.get("min_malware_recall", 0.0))
            if classification_enabled
            else None
        ),
        "selective_prediction_enabled": selective_enabled,
        "selective_prediction_mode": selective_mode,
        "selective_score_type": (
            _selective_score_type(selective_cfg) if selective_enabled else "disabled"
        ),
        "target_coverage": (
            float(selective_cfg.get("target_coverage", 1.0))
            if selective_enabled and selective_mode in {"threshold", "conformal"}
            else None
        ),
        "risk_control_level": (
            float(selective_cfg.get("risk_level", 0.0))
            if selective_enabled and selective_mode == "risk_control"
            else None
        ),
        "risk_control_require_feasible": bool(
            selective_enabled
            and selective_mode == "risk_control"
            and selective_cfg.get("require_feasible", False)
        ),
        "risk_control_min_calibration_malware": (
            int(selective_cfg.get("min_calibration_malware", 1))
            if selective_enabled and selective_mode == "risk_control"
            else None
        ),
        "conformal_uses_raw_conflict": bool(
            selective_enabled
            and selective_mode == "conformal"
            and selective_cfg.get("use_raw_conflict", False)
        ),
    }


def _write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _json_compatible(payload),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def run(cfg: dict) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO))
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    configure_determinism(
        bool(train_cfg.get("deterministic", False)),
        strict=bool(train_cfg.get("strict_deterministic", False)),
    )
    configure_multiprocessing_sharing(cfg)
    device = select_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("use_amp", True))
    run_test = bool(eval_cfg.get("run_test", True))
    run_robust_test = bool(eval_cfg.get("run_robust_test", True))
    eval_only = bool(eval_cfg.get("eval_only", False))
    refit_posthoc_calibration = bool(
        eval_cfg.get("refit_posthoc_calibration", False)
    )
    refit_decision_calibration = _resolve_refit_decision_calibration(eval_cfg)
    final_temperature_override_raw = eval_cfg.get("final_temperature_override")
    final_temperature_override = (
        None
        if final_temperature_override_raw is None
        else float(final_temperature_override_raw)
    )
    extra_eval_sets = _normalize_extra_eval_sets(eval_cfg.get("extra_sets"))
    pure_external_eval_only = (
        eval_only
        and not refit_posthoc_calibration
        and not refit_decision_calibration
        and not run_test
        and bool(extra_eval_sets)
    )
    tuning_mode = bool(train_cfg.get("tuning_mode", False))
    calibration_enabled = bool((cfg.get("calibration", {}) or {}).get("enabled", False))
    classification_cfg = cfg.get("classification_threshold", {}) or {}
    classification_threshold_enabled = bool(classification_cfg.get("enabled", False))
    selective_enabled = bool((cfg.get("selective_prediction", {}) or {}).get("enabled", False))
    selective_score_type = _selective_score_type(
        cfg.get("selective_prediction", {}) or {}
    )
    configured_fusion_mode = str((cfg.get("fusion", {}) or {}).get("mode", "")).lower()
    model_fusion_mode = str((cfg.get("model", {}) or {}).get("fusion_mode", "")).lower()
    discount_probability_mode = (
        configured_fusion_mode == "discount_probability"
        or (not configured_fusion_mode and model_fusion_mode == "discount_probability")
    )
    if (calibration_enabled or selective_enabled) and not discount_probability_mode:
        raise ValueError(
            "Post-hoc calibration and selective prediction require discount_probability fusion"
        )
    gate_cfg = cfg.get("model", {}).get("gate", {}) or {}
    if bool(gate_cfg.get("use_perturbation_evidence", False)):
        raise ValueError(
            "model.gate.use_perturbation_evidence=true is no longer supported: "
            "synthetic pert_* values are diagnostic-only."
        )
    if run_robust_test and not run_test:
        raise ValueError("eval.run_robust_test=true requires eval.run_test=true")
    if tuning_mode:
        if run_test or run_robust_test:
            raise ValueError("train.tuning_mode=true forbids test evaluation")
        if extra_eval_sets:
            raise ValueError("train.tuning_mode=true forbids eval.extra_sets")
        if not bool((eval_cfg.get("robust_val", {}) or {}).get("enabled", False)):
            raise ValueError("train.tuning_mode=true requires eval.robust_val.enabled=true")
        tuning_checkpoint_metric = str(
            train_cfg.get("checkpoint_metric", "clean_macro_f1")
        ).strip().lower()
        if tuning_checkpoint_metric not in {
            "clean",
            "clean_macro_f1",
            "macro_f1",
            "val_macro_f1",
            "robust_composite",
        }:
            raise ValueError(
                "train.tuning_mode=true requires a supported clean or robust checkpoint metric"
            )
    if eval_only:
        if tuning_mode:
            raise ValueError("eval.eval_only=true is incompatible with train.tuning_mode=true")
        if not str(eval_cfg.get("checkpoint_path") or "").strip():
            raise ValueError("eval.eval_only=true requires eval.checkpoint_path")
    else:
        if refit_posthoc_calibration:
            raise ValueError(
                "eval.refit_posthoc_calibration=true requires eval.eval_only=true"
            )
        if refit_decision_calibration:
            raise ValueError(
                "eval.refit_decision_calibration=true requires eval.eval_only=true"
            )
    if final_temperature_override is not None:
        if not eval_only:
            raise ValueError(
                "eval.final_temperature_override requires eval.eval_only=true"
            )
        if refit_posthoc_calibration:
            raise ValueError(
                "eval.final_temperature_override isolates the fitted temperature "
                "and is incompatible with eval.refit_posthoc_calibration=true"
            )
        if (
            not math.isfinite(final_temperature_override)
            or final_temperature_override <= 0.0
        ):
            raise ValueError(
                "eval.final_temperature_override must be finite and positive"
            )
    if refit_posthoc_calibration:
        if not calibration_enabled:
            raise ValueError(
                "eval.refit_posthoc_calibration=true requires calibration.enabled=true"
            )
        if (
            classification_threshold_enabled or selective_enabled
        ) and not refit_decision_calibration:
            raise ValueError(
                "Post-hoc refitting changes predictions and therefore requires "
                "eval.refit_decision_calibration=true when classification "
                "thresholding or selective prediction is enabled "
                "(legacy alias: eval.refit_rejection_threshold)"
            )

    train_ds = None
    train_loader = None
    val_loader = None
    val_posthoc_calibration_loader = None
    val_conformal_calibration_loader = None
    robust_val_loaders: list[dict[str, Any]] = []
    robust_calibration_loaders: list[dict[str, Any]] = []
    validation_split_summary: dict[str, Any] = {
        "split_seed": None,
        "validation_fraction": None,
        "num_selection": 0,
        "num_calibration": 0,
        "num_posthoc_calibration": 0,
        "num_conformal_calibration": 0,
        "external_eval_only": bool(pure_external_eval_only),
    }
    prebuilt_extra_eval_sets: list[dict[str, Any]] = []

    if pure_external_eval_only:
        for idx, extra in enumerate(extra_eval_sets):
            name = str(extra.get("name") or f"extra_{idx}")
            pt_dir, csv_path = _extra_eval_paths(cfg, extra)
            perturb_type = extra.get("perturb_type")
            perturb_strength = float(extra.get("perturb_strength", 0.0))
            try:
                extra_ds = build_dataset_from_paths(
                    cfg,
                    pt_dir=pt_dir,
                    csv_path=csv_path,
                    is_train=False,
                    perturb_type=str(perturb_type) if perturb_type else None,
                    perturb_strength=perturb_strength,
                    dataset_overrides={
                        "allow_pt_superset": bool(extra.get("allow_pt_superset", True)),
                        "strict_split_integrity": bool(extra.get("strict_split_integrity", True)),
                    },
                )
            except EmptyExtraEvalSetError as exc:
                if bool(extra.get("skip_if_empty", True)):
                    prebuilt_extra_eval_sets.append(
                        {
                            "name": name,
                            "extra": extra,
                            "pt_dir": pt_dir,
                            "csv_path": csv_path,
                            "dataset": None,
                            "skipped": True,
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                raise
            prebuilt_extra_eval_sets.append(
                {
                    "name": name,
                    "extra": extra,
                    "pt_dir": pt_dir,
                    "csv_path": csv_path,
                    "dataset": extra_ds,
                    "skipped": False,
                    "perturb_type": perturb_type,
                    "perturb_strength": perturb_strength,
                }
            )
        first_extra_ds = next(
            (item["dataset"] for item in prebuilt_extra_eval_sets if item.get("dataset") is not None),
            None,
        )
        if first_extra_ds is None:
            raise RuntimeError("eval.extra_sets did not provide any loadable external dataset")
        feature_dim = int(first_extra_ds.feature_dim)
    else:
        validate_split_partitions(cfg, include_test=run_test)
        val_ds = build_dataset(cfg, "val", is_train=False)
        holdout_enabled = bool((cfg.get("calibration", {}) or {}).get("holdout_enabled", False))
        needs_validation_holdout = (
            calibration_enabled
            or classification_threshold_enabled
            or selective_enabled
            or holdout_enabled
        )
        if needs_validation_holdout:
            val_selection_ds, val_holdout_ds, validation_split = split_validation_dataset(cfg, val_ds)
            selection_indices = list(validation_split["selection_indices"])
            calibration_indices = list(validation_split["calibration_indices"])
            # Every selective method receives the same decision-calibration
            # sample budget. This also keeps threshold baselines from reusing
            # rows that fitted post-hoc model parameters.
            needs_decision_calibration_split = (
                classification_threshold_enabled or selective_enabled
            )
            if needs_decision_calibration_split:
                (
                    val_posthoc_calibration_ds,
                    val_conformal_calibration_ds,
                    conformal_split,
                ) = split_posthoc_conformal_dataset(cfg, val_ds, calibration_indices)
                validation_split.update(conformal_split)
                posthoc_calibration_indices = list(
                    conformal_split["posthoc_calibration_indices"]
                )
            else:
                val_posthoc_calibration_ds = val_holdout_ds
                val_conformal_calibration_ds = val_holdout_ds
                posthoc_calibration_indices = calibration_indices
                validation_split.update(
                    {
                        "num_posthoc_calibration": len(calibration_indices),
                        "num_conformal_calibration": (
                            len(calibration_indices)
                            if _uses_conformal_selective(cfg.get("selective_prediction", {}) or {})
                            else 0
                        ),
                        "posthoc_calibration_indices": calibration_indices,
                        "conformal_calibration_indices": calibration_indices,
                    }
                )
        else:
            val_selection_ds = val_ds
            val_posthoc_calibration_ds = val_ds
            val_conformal_calibration_ds = val_ds
            selection_indices = None
            calibration_indices = None
            posthoc_calibration_indices = None
            validation_split = {
                "split_seed": None,
                "validation_fraction": 1.0,
                "num_selection": len(val_ds),
                "num_calibration": len(val_ds),
                "num_posthoc_calibration": len(val_ds),
                "num_conformal_calibration": 0,
                "selection_indices": None,
                "calibration_indices": None,
                "posthoc_calibration_indices": None,
                "conformal_calibration_indices": None,
            }
        validation_split_summary = {
            key: value
            for key, value in validation_split.items()
            if key not in {
                "selection_indices",
                "calibration_indices",
                "posthoc_calibration_indices",
                "conformal_calibration_indices",
            }
        }
        val_loader = build_loader(cfg, val_selection_ds, is_train=False)
        val_posthoc_calibration_loader = build_loader(
            cfg, val_posthoc_calibration_ds, is_train=False
        )
        val_conformal_calibration_loader = build_loader(
            cfg, val_conformal_calibration_ds, is_train=False
        )
        if not eval_only:
            train_ds = build_dataset(cfg, "train", is_train=True)
            train_loader = build_loader(cfg, train_ds, is_train=True)
        robust_val_loaders = build_robust_val_loaders(cfg, selection_indices)
        robust_calibration_loaders = (
            build_reliability_calibration_loaders(cfg, posthoc_calibration_indices)
            if calibration_enabled
            and (not eval_only or refit_posthoc_calibration)
            and uses_routing_calibration_scenarios(cfg)
            else []
        )
        feature_dim = train_ds.feature_dim if train_ds is not None else val_ds.feature_dim
    if not eval_only and train_ds is None:
        train_ds = build_dataset(cfg, "train", is_train=True)
        train_loader = build_loader(cfg, train_ds, is_train=True)
    test_loader = None
    if run_test:
        test_ds = build_dataset(cfg, "test", is_train=False)
        test_loader = build_loader(cfg, test_ds, is_train=False)

    model = build_model(cfg, feature_dim).to(device)
    source_checkpoint_path: Path | None = None
    requested_checkpoint_path: Path | None = None

    exp_name = str(train_cfg.get("exp_name", "tri_modal_robust"))
    if eval_only:
        exp_name = str(eval_cfg.get("output_name") or f"{exp_name}_eval_only")
    out_dir = Path(data_cfg.get("out_dir", "experiments")) / exp_name / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_checkpoint_path = out_dir / "best_encoder_selected.pt"
    pipeline_checkpoint_path = out_dir / "best_tri_modal_robust.pt"
    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if eval_only:
        requested_checkpoint_path = Path(str(eval_cfg["checkpoint_path"]))
        if not requested_checkpoint_path.is_absolute():
            requested_checkpoint_path = Path.cwd() / requested_checkpoint_path
        if not requested_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {requested_checkpoint_path}"
            )
        best_path, ckpt = _load_eval_checkpoint(
            requested_checkpoint_path,
            refit_posthoc_calibration=refit_posthoc_calibration,
            map_location=device,
        )
        source_checkpoint_path = best_path
        if requested_checkpoint_path != best_path:
            logger.info(
                "posthoc_refit_resolved_encoder_checkpoint requested=%s encoder=%s",
                requested_checkpoint_path,
                best_path,
            )
        allow_checkpoint_mismatch = bool(
            eval_cfg.get("allow_checkpoint_config_mismatch", False)
        )
        validate_eval_checkpoint_config(
            cfg,
            ckpt.get("cfg"),
            allow_mismatch=allow_checkpoint_mismatch,
        )
        validate_checkpoint_implementation(
            ckpt,
            allow_mismatch=allow_checkpoint_mismatch,
        )
        validate_checkpoint_decision_signature(
            cfg,
            ckpt,
            refit_decision_calibration=refit_decision_calibration,
        )
        model.load_state_dict(ckpt["model"])
        if final_temperature_override is not None:
            discount_fusion = getattr(model, "discount_fusion", None)
            parameters = (
                discount_fusion.final_temperature_parameters()
                if discount_fusion is not None
                and hasattr(discount_fusion, "final_temperature_parameters")
                else []
            )
            if len(parameters) != 1:
                raise ValueError(
                    "eval.final_temperature_override requires a checkpoint model "
                    "with routed final-temperature scaling enabled"
                )
            with torch.no_grad():
                parameters[0].fill_(math.log(final_temperature_override))
            logger.info(
                "eval_final_temperature_override=%.6f",
                final_temperature_override,
            )
        branch_competence_summary = dict(ckpt.get("branch_competence_prior") or {"enabled": False})
        apply_branch_competence_prior(model, branch_competence_summary)
        if (
            bool(((cfg.get("fusion", {}) or {}).get("branch_competence_prior", {}) or {}).get("enabled", False))
            and not bool(branch_competence_summary.get("enabled", False))
        ):
            logger.warning(
                "fusion.branch_competence_prior.enabled=true but checkpoint has no fitted prior; using neutral priors"
            )
        visible_integrity_summary = dict(ckpt.get("model_visible_integrity_reference") or {"enabled": False})
        apply_model_visible_integrity_reference(model, visible_integrity_summary)
        if (
            bool(((cfg.get("fusion", {}) or {}).get("visible_integrity_modifier", {}) or {}).get("enabled", False))
            and not bool(visible_integrity_summary.get("enabled", False))
            and not refit_posthoc_calibration
        ):
            logger.warning(
                "fusion.visible_integrity_modifier.enabled=true but checkpoint has no fitted reference; using neutral modifiers"
            )
        best_score = float(ckpt.get("checkpoint_score", -1.0))
        best_val_f1 = float((ckpt.get("val") or {}).get("macro_f1", -1.0))
        checkpoint_metric_name = str(ckpt.get("checkpoint_metric", "loaded_checkpoint"))
        calibration_summary = dict(ckpt.get("calibration") or {"enabled": False})
        if final_temperature_override is not None:
            original_final_temperature = dict(
                calibration_summary.get("final_temperature") or {}
            )
            calibration_summary["final_temperature"] = {
                "enabled": True,
                "overridden_for_ablation": True,
                "temperature": float(final_temperature_override),
                "source_temperature": original_final_temperature.get("temperature"),
            }
            temperatures = dict(calibration_summary.get("temperatures") or {})
            temperatures["final"] = float(final_temperature_override)
            calibration_summary["temperatures"] = temperatures
        classification_threshold_summary = ckpt.get("classification_threshold")
        if classification_threshold_enabled:
            if classification_threshold_summary is None and not refit_decision_calibration:
                raise ValueError(
                    "Classification-threshold eval-only mode requires a fitted "
                    "classification_threshold in the checkpoint or "
                    "eval.refit_decision_calibration=true"
                )
            classification_threshold = float(
                (classification_threshold_summary or {}).get("threshold", 0.5)
            )
        else:
            classification_threshold_summary = None
            classification_threshold = 0.5
        selective_cfg = cfg.get("selective_prediction", {}) or {}
        rejection_threshold = ckpt.get("rejection_threshold")
        conformal_thresholds = ckpt.get("conformal_thresholds")
        risk_control_thresholds = ckpt.get("risk_control_thresholds")
        if bool(selective_cfg.get("enabled", False)) and not refit_decision_calibration:
            if _uses_risk_control_selective(selective_cfg):
                if risk_control_thresholds is None:
                    raise ValueError(
                        "Risk-control eval-only mode requires risk_control_thresholds "
                        "saved in the checkpoint"
                    )
                if (
                    bool(selective_cfg.get("require_feasible", False))
                    and not bool(risk_control_thresholds.get("feasible", False))
                ):
                    raise ValueError(
                        "Risk-control checkpoint threshold is marked infeasible "
                        "while selective_prediction.require_feasible=true; "
                        "set eval.refit_decision_calibration=true"
                    )
            elif _uses_conformal_selective(selective_cfg):
                if conformal_thresholds is None:
                    raise ValueError(
                        "Conformal eval-only mode requires conformal_thresholds saved in the checkpoint"
                    )
            elif rejection_threshold is None:
                raise ValueError(
                    "Threshold eval-only mode requires rejection_threshold saved in the checkpoint"
                )
        rejection_threshold = (
            float(rejection_threshold) if rejection_threshold is not None else None
        )
        if not bool(selective_cfg.get("enabled", False)):
            # A full-coverage ablation must not silently inherit and report the
            # source checkpoint's rejection decisions.
            rejection_threshold = None
            conformal_thresholds = None
            risk_control_thresholds = None
        if refit_posthoc_calibration:
            if val_posthoc_calibration_loader is None:
                raise RuntimeError(
                    "Post-hoc refitting requires the validation calibration loader"
                )
            calibration_loaders = [
                {
                    "name": "clean",
                    "loader": val_posthoc_calibration_loader,
                    "reliability_branches": ["api", "graph", "manifest"],
                },
                *robust_calibration_loaders,
            ]
            calibration_summary = fit_posthoc_calibration(
                model,
                calibration_loaders,
                device,
                use_amp,
                cfg,
            )
            routing_reference = calibration_summary.get(
                "routing_visible_reference"
            )
            if routing_reference is not None:
                modifier_cfg = (
                    (cfg.get("fusion", {}) or {}).get(
                        "visible_integrity_modifier", {}
                    )
                    or {}
                )
                reference_mode = str(
                    calibration_summary.get("routing_visible_reference_mode")
                    or modifier_cfg.get("mode", "bounded_visibility")
                ).lower()
                reference_names = ("api", "graph", "manifest")
                reference = {
                    name: float(value)
                    for name, value in zip(reference_names, routing_reference)
                }
                reference_count = int(
                    calibration_summary.get("routing_visible_reference_count", 0)
                )
                visible_integrity_summary = {
                    "enabled": True,
                    "mode": reference_mode,
                    "metric": (
                        "median_clean_effective_integrity"
                        if reference_mode == "relative_effective"
                        else "median_clean_encoder_coverage"
                    ),
                    "beta": 1.0
                    if reference_mode == "relative_effective"
                    else float(modifier_cfg.get("beta", 1.0)),
                    "min_value": 0.0
                    if reference_mode == "relative_effective"
                    else float(modifier_cfg.get("min_value", 0.5)),
                    "min_reference": float(
                        modifier_cfg.get("min_reference", 1.0e-6)
                    ),
                    "counts": {
                        name: reference_count for name in reference_names
                    },
                    "reference": reference,
                    "values": [reference[name] for name in reference_names],
                    "source": "posthoc_clean_cache",
                }
            if bool(calibration_summary.get("enabled", False)):
                calibration_summary["degradation_scenarios"] = [
                    {
                        "name": str(item["name"]),
                        "perturb_type": str(item["perturb_type"]),
                        "strength": float(item["strength"]),
                        "reliability_branches": list(
                            item["reliability_branches"]
                        ),
                    }
                    for item in robust_calibration_loaders
                ]
            classification_threshold_summary = None
            classification_threshold = 0.5
            rejection_threshold = None
            conformal_thresholds = None
            risk_control_thresholds = None
        logger.info("eval-only mode loaded checkpoint: %s", best_path)
    else:
        assert train_loader is not None
        model.set_calibration_active(False)
        posthoc_only_parameters = (
            model.encoder_training_frozen_parameters()
            if hasattr(model, "encoder_training_frozen_parameters")
            else model.calibration_parameters()
        )
        for parameter in posthoc_only_parameters:
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(train_cfg.get("epochs", 1))),
            eta_min=float(train_cfg.get("eta_min", 1e-6)),
        )
        scaler = build_grad_scaler(device, use_amp)
        best_score = -1.0
        best_val_f1 = -1.0
        checkpoint_metric_name = ""
        best_path = encoder_checkpoint_path
        saved_checkpoint_this_run = False
        patience = int(train_cfg.get("patience", 10))
        stale = 0
        run_epoch_robust_val = checkpoint_requires_robust_validation(cfg)

        for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
            val_metrics, _ = evaluate(
                model,
                val_loader,
                device,
                use_amp,
                "val",
                dump_rows=False,
                selective_score_type=selective_score_type,
            )
            enforce_failed_ratio(val_metrics, cfg, "val")
            val_robust_metrics = (
                evaluate_robust_validation(
                    model,
                    robust_val_loaders,
                    device,
                    use_amp,
                    cfg,
                    selective_score_type=selective_score_type,
                )
                if run_epoch_robust_val
                else {}
            )
            score, checkpoint_metric_name = checkpoint_score(
                cfg,
                val_metrics,
                val_robust_metrics,
                robust_val_loaders,
            )
            if not math.isfinite(float(score)):
                raise FloatingPointError(
                    f"Non-finite checkpoint score at epoch={epoch}: {score}"
                )
            scheduler.step()
            logger.info(
                "epoch=%s train_loss=%.4f val_macro_f1=%.4f val_auc=%.4f val_acc=%.4f checkpoint_score=%.4f",
                epoch,
                train_loss,
                val_metrics["f1"],
                val_metrics["auc"],
                val_metrics["acc"],
                score,
            )
            if score > best_score + float(train_cfg.get("min_delta", 1e-4)):
                best_score = score
                best_val_f1 = float(val_metrics["macro_f1"])
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "val": val_metrics,
                        "val_robust": val_robust_metrics,
                        "checkpoint_score": score,
                        "checkpoint_metric": checkpoint_metric_name,
                        "method_implementation_sha256": _method_implementation_sha256(),
                        "checkpoint_stage": CHECKPOINT_STAGE_ENCODER_SELECTED,
                        "epoch": epoch,
                    },
                    best_path,
                )
                saved_checkpoint_this_run = True
            else:
                stale += 1
                if stale >= patience:
                    break

        if not saved_checkpoint_this_run:
            raise RuntimeError(
                "Training completed without writing a checkpoint in this run; "
                "refusing to load a potentially stale best.pt"
            )
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        validate_checkpoint_stage(
            ckpt,
            expected=CHECKPOINT_STAGE_ENCODER_SELECTED,
            checkpoint_path=best_path,
        )
        model.load_state_dict(ckpt["model"])
        calibration_loaders = [
            {
                "name": "clean",
                "loader": val_posthoc_calibration_loader,
                "reliability_branches": ["api", "graph", "manifest"],
            },
            *robust_calibration_loaders,
        ]
        calibration_summary = fit_posthoc_calibration(
            model,
            calibration_loaders,
            device,
            use_amp,
            cfg,
        )
        if bool(calibration_summary.get("enabled", False)):
            calibration_summary["degradation_scenarios"] = [
                {
                    "name": str(item["name"]),
                    "perturb_type": str(item["perturb_type"]),
                    "strength": float(item["strength"]),
                    "reliability_branches": list(item["reliability_branches"]),
                }
                for item in robust_calibration_loaders
            ]
        rejection_threshold = None
        conformal_thresholds = None
        risk_control_thresholds = None
        classification_threshold_summary = None
        classification_threshold = 0.5

    if pure_external_eval_only:
        val_calibration_metrics: dict[str, Any] = {}
        val_calibration_rows: list[dict[str, Any]] = []
        val_conformal_metrics: dict[str, Any] = {}
        val_conformal_rows: list[dict[str, Any]] = []
        val_metrics: dict[str, Any] = {}
        val_rows: list[dict[str, Any]] = []
        val_robust_results: dict[str, Any] = {}
    else:
        if (
            val_posthoc_calibration_loader is None
            or val_conformal_calibration_loader is None
            or val_loader is None
        ):
            raise RuntimeError("Internal error: validation loaders are required outside pure external eval-only mode")
        val_calibration_metrics, val_calibration_rows = evaluate(
            model,
            val_posthoc_calibration_loader,
            device,
            use_amp,
            "val_posthoc_calibration",
            dump_rows=True,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )
        enforce_failed_ratio(val_calibration_metrics, cfg, "val_posthoc_calibration")
        if not eval_only or refit_posthoc_calibration:
            branch_competence_summary = estimate_branch_competence_prior(
                val_calibration_rows,
                cfg,
            )
            apply_branch_competence_prior(model, branch_competence_summary)
            if bool(branch_competence_summary.get("enabled", False)):
                logger.info(
                    "branch_competence_prior %s",
                    " ".join(
                        f"{name}={value:.4f}"
                        for name, value in branch_competence_summary["prior"].items()
                    ),
                )
                val_calibration_metrics, val_calibration_rows = evaluate(
                    model,
                    val_posthoc_calibration_loader,
                    device,
                    use_amp,
                    "val_posthoc_calibration",
                    dump_rows=True,
                    selective_score_type=selective_score_type,
                    classification_threshold=classification_threshold,
                )
                enforce_failed_ratio(val_calibration_metrics, cfg, "val_posthoc_calibration")
        if not eval_only:
            visible_integrity_summary = estimate_model_visible_integrity_reference(
                val_calibration_rows,
                cfg,
            )
            apply_model_visible_integrity_reference(model, visible_integrity_summary)
            if bool(visible_integrity_summary.get("enabled", False)):
                logger.info(
                    "model_visible_integrity_reference %s",
                    " ".join(
                        f"{name}={value:.4f}"
                        for name, value in visible_integrity_summary["reference"].items()
                    ),
                )
                val_calibration_metrics, val_calibration_rows = evaluate(
                    model,
                    val_posthoc_calibration_loader,
                    device,
                    use_amp,
                    "val_posthoc_calibration",
                    dump_rows=True,
                    selective_score_type=selective_score_type,
                    classification_threshold=classification_threshold,
                )
                enforce_failed_ratio(val_calibration_metrics, cfg, "val_posthoc_calibration")
        if classification_threshold_enabled and (
            not eval_only or refit_decision_calibration
        ):
            classification_threshold_summary = fit_malware_classification_threshold(
                val_calibration_rows, classification_cfg
            )
            if classification_threshold_summary is None:
                raise RuntimeError("Enabled classification-threshold fitting returned no result")
            classification_threshold = float(
                classification_threshold_summary["threshold"]
            )
            logger.info(
                "classification_threshold=%.6f calibration_macro_f1=%.4f "
                "calibration_malware_recall=%.4f",
                classification_threshold,
                classification_threshold_summary["macro_f1"],
                classification_threshold_summary["malware_recall"],
            )
            # Keep the reported post-hoc metrics and row predictions consistent
            # with the now-fixed decision rule used by risk calibration.
            val_calibration_metrics, val_calibration_rows = evaluate(
                model,
                val_posthoc_calibration_loader,
                device,
                use_amp,
                "val_posthoc_calibration",
                dump_rows=True,
                selective_score_type=selective_score_type,
                classification_threshold=classification_threshold,
            )
            enforce_failed_ratio(
                val_calibration_metrics, cfg, "val_posthoc_calibration"
            )
        val_conformal_metrics, val_conformal_rows = evaluate(
            model,
            val_conformal_calibration_loader,
            device,
            use_amp,
            "val_conformal_calibration",
            dump_rows=True,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )
        enforce_failed_ratio(val_conformal_metrics, cfg, "val_conformal_calibration")
        if not eval_only or refit_decision_calibration:
            rejection_threshold = fit_rejection_threshold(
                val_conformal_rows, cfg.get("selective_prediction", {}) or {}
            )
            conformal_thresholds = fit_conformal_thresholds(
                val_conformal_rows, cfg.get("selective_prediction", {}) or {}
            )
            risk_control_thresholds = fit_risk_control_thresholds(
                val_conformal_rows, cfg.get("selective_prediction", {}) or {}
            )
        if (not eval_only or refit_posthoc_calibration) and best_path.exists():
            checkpoint_source = (
                source_checkpoint_path
                if eval_only and source_checkpoint_path is not None
                else best_path
            )
            ckpt = torch.load(
                checkpoint_source, map_location="cpu", weights_only=True
            )
            validate_checkpoint_stage(
                ckpt,
                expected=CHECKPOINT_STAGE_ENCODER_SELECTED,
                checkpoint_path=checkpoint_source,
            )
            ckpt["model"] = model.state_dict()
            ckpt["cfg"] = cfg
            ckpt["method_implementation_sha256"] = (
                _method_implementation_sha256()
            )
            ckpt["checkpoint_stage"] = CHECKPOINT_STAGE_PIPELINE_FITTED
            portable_encoder_path = (
                pipeline_checkpoint_path.parent / "best_encoder_selected.pt"
            )
            if (
                Path(checkpoint_source).resolve()
                != portable_encoder_path.resolve()
            ):
                shutil.copy2(checkpoint_source, portable_encoder_path)
            ckpt["encoder_checkpoint_path"] = portable_encoder_path.name
            ckpt["encoder_checkpoint_sha256"] = _file_sha256(
                portable_encoder_path
            )
            ckpt["calibration"] = calibration_summary
            ckpt["branch_competence_prior"] = branch_competence_summary
            ckpt["model_visible_integrity_reference"] = visible_integrity_summary
            ckpt["classification_threshold"] = classification_threshold_summary
            if rejection_threshold is not None:
                ckpt["rejection_threshold"] = rejection_threshold
            else:
                ckpt.pop("rejection_threshold", None)
            ckpt["conformal_thresholds"] = conformal_thresholds
            ckpt["risk_control_thresholds"] = risk_control_thresholds
            ckpt["decision_calibration_signature"] = (
                _decision_calibration_signature(cfg)
            )
            ckpt["validation_split"] = validation_split_summary
            if eval_only:
                ckpt["source_checkpoint_path"] = str(checkpoint_source)
            best_path = pipeline_checkpoint_path
            torch.save(ckpt, best_path)
        val_conformal_metrics.update(
            _selective_metrics_from_rows(val_conformal_rows, rejection_threshold)
        )
        val_conformal_metrics.update(
            conformal_selective_metrics(val_conformal_rows, conformal_thresholds)
        )
        val_conformal_metrics.update(
            risk_control_selective_metrics(
                val_conformal_rows, risk_control_thresholds
            )
        )

        val_metrics, val_rows = evaluate(
            model,
            val_loader,
            device,
            use_amp,
            "val_selection",
            dump_rows=True,
            selective_threshold=rejection_threshold,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )
        enforce_failed_ratio(val_metrics, cfg, "val_selection")
        val_metrics.update(conformal_selective_metrics(val_rows, conformal_thresholds))
        val_metrics.update(
            risk_control_selective_metrics(val_rows, risk_control_thresholds)
        )
        val_robust_results = evaluate_robust_validation(
            model,
            robust_val_loaders,
            device,
            use_amp,
            cfg,
            selective_threshold=rejection_threshold,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )

    test_metrics: dict[str, Any] = {}
    test_rows: list[dict[str, Any]] = []
    robust_results: dict[str, Any] = {}
    if run_test:
        assert test_loader is not None
        test_metrics, test_rows = evaluate(
            model,
            test_loader,
            device,
            use_amp,
            "test_clean",
            dump_rows=True,
            selective_threshold=rejection_threshold,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )
        enforce_failed_ratio(test_metrics, cfg, "test_clean")
        # Conformal selective metrics on clean test (test_rows is still clean
        # here -- robust rows are appended only inside the loop below).
        test_metrics.update(conformal_selective_metrics(test_rows, conformal_thresholds))
        test_metrics.update(
            risk_control_selective_metrics(test_rows, risk_control_thresholds)
        )
        if run_robust_test:
            perturb_tests = list(eval_cfg.get("perturb_tests", ["clean"]))
            if eval_cfg.get("perturb_strengths") is not None:
                perturb_strengths = [float(v) for v in eval_cfg.get("perturb_strengths") or []]
            else:
                perturb_strengths = [float(eval_cfg.get("perturb_strength", 0.5))]
            perturb_strengths = perturb_strengths or [0.5]
            for perturb in perturb_tests:
                if perturb == "clean":
                    robust_results[perturb] = test_metrics
                    continue
                # *_missing / modality_dropout_* perturbations ignore strength;
                # running them once per strength wastes time on identical results.
                is_strength_invariant = perturb.endswith("_missing") or perturb.startswith("modality_dropout_")
                strengths = [1.0] if is_strength_invariant else perturb_strengths
                for strength in strengths:
                    result_key = perturb if len(strengths) == 1 else f"{perturb}_s{strength:g}"
                    robust_ds = build_dataset(cfg, "test", is_train=False, perturb_type=perturb, perturb_strength=strength)
                    robust_loader = build_loader(cfg, robust_ds, is_train=False)
                    metrics, rows = evaluate(
                        model,
                        robust_loader,
                        device,
                        use_amp,
                        f"test_{result_key}",
                        dump_rows=True,
                        selective_threshold=rejection_threshold,
                        selective_score_type=selective_score_type,
                        classification_threshold=classification_threshold,
                    )
                    enforce_failed_ratio(metrics, cfg, f"test_{result_key}")
                    metrics.update(conformal_selective_metrics(rows, conformal_thresholds))
                    metrics.update(
                        risk_control_selective_metrics(rows, risk_control_thresholds)
                    )
                    robust_results[result_key] = metrics
                    test_rows.extend(rows)

    # Preserve all three disjoint validation roles in the audit dump. In
    # particular, val_conformal_rows are the exact rows used to fit I3.
    all_rows = (
        val_rows
        + val_calibration_rows
        + val_conformal_rows
        + test_rows
    )

    extra_results = {}
    extra_rows: list[dict[str, Any]] = []
    extra_iter = prebuilt_extra_eval_sets if pure_external_eval_only else [
        {"extra": extra, "name": str(extra.get("name") or f"extra_{idx}")}
        for idx, extra in enumerate(extra_eval_sets)
    ]
    for item in extra_iter:
        extra = item["extra"]
        name = str(item.get("name") or extra.get("name") or "extra")
        if item.get("skipped"):
            extra_results[name] = {
                "skipped": True,
                "reason": str(item.get("skip_reason") or ""),
                "pt_dir": str(item.get("pt_dir") or ""),
                "csv": str(item.get("csv_path") or ""),
            }
            continue
        if item.get("dataset") is not None:
            extra_ds = item["dataset"]
            pt_dir = item["pt_dir"]
            csv_path = item["csv_path"]
            perturb_type = item.get("perturb_type")
            perturb_strength = float(item.get("perturb_strength", 0.0))
        else:
            pt_dir, csv_path = _extra_eval_paths(cfg, extra)
            perturb_type = extra.get("perturb_type")
            perturb_strength = float(extra.get("perturb_strength", 0.0))
            try:
                extra_ds = build_dataset_from_paths(
                    cfg,
                    pt_dir=pt_dir,
                    csv_path=csv_path,
                    is_train=False,
                    perturb_type=str(perturb_type) if perturb_type else None,
                    perturb_strength=perturb_strength,
                    dataset_overrides={
                        "allow_pt_superset": bool(extra.get("allow_pt_superset", True)),
                        "strict_split_integrity": bool(extra.get("strict_split_integrity", True)),
                    },
                )
            except EmptyExtraEvalSetError as exc:
                if bool(extra.get("skip_if_empty", True)):
                    extra_results[name] = {
                        "skipped": True,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "pt_dir": str(pt_dir),
                        "csv": str(csv_path),
                    }
                    continue
                raise
        if extra_ds is None:
            if bool(extra.get("skip_if_empty", True)):
                extra_results[name] = {
                    "skipped": True,
                    "reason": "Empty extra dataset",
                    "pt_dir": str(pt_dir),
                    "csv": str(csv_path),
                }
                continue
            raise RuntimeError(f"Extra eval set {name} did not produce a dataset")
        extra_loader = build_loader(cfg, extra_ds, is_train=False)
        split_name = str(extra.get("split_name") or name)
        metrics, rows = evaluate(
            model,
            extra_loader,
            device,
            use_amp,
            split_name,
            dump_rows=True,
            selective_threshold=rejection_threshold,
            selective_score_type=selective_score_type,
            classification_threshold=classification_threshold,
        )
        enforce_failed_ratio(metrics, cfg, split_name, max_failed_ratio=extra.get("max_failed_ratio"))
        metrics.update(conformal_selective_metrics(rows, conformal_thresholds))
        metrics.update(risk_control_selective_metrics(rows, risk_control_thresholds))
        metrics = {
            **metrics,
            "pt_dir": str(pt_dir),
            "csv": str(csv_path),
            "perturb_type": str(perturb_type or ""),
            "perturb_strength": perturb_strength,
        }
        extra_results[name] = metrics
        # all_rows already has val+test; extra_rows keeps extra-eval separate
        # so gate_diagnostics.csv and gate_diagnostics_extra_eval.csv are disjoint.
        extra_rows.extend(rows)

    write_gate_dump(out_dir / "gate_diagnostics.csv", all_rows)
    write_gate_dump(out_dir / "gate_diagnostics_extra_eval.csv", extra_rows)
    risk_coverage_path = out_dir / "risk_coverage_curve.csv"
    extra_risk_coverage_path = out_dir / "risk_coverage_curve_extra_eval.csv"
    risk_coverage_points = write_risk_coverage_curve(
        risk_coverage_path, test_rows
    )
    extra_risk_coverage_points = write_risk_coverage_curve(
        extra_risk_coverage_path, extra_rows
    )
    if extra_results:
        _write_metrics_json(out_dir / "metrics_extra_eval.json", extra_results)
    tuning_robust_composite_score = None
    if tuning_mode:
        robust_score_cfg = copy.deepcopy(cfg)
        robust_score_cfg.setdefault("train", {})["checkpoint_metric"] = "robust_composite"
        tuning_robust_composite_score, _ = checkpoint_score(
            robust_score_cfg,
            val_metrics,
            val_robust_results,
            robust_val_loaders,
        )
    summary = {
        "run_identity": build_run_identity(cfg, exp_name, seed),
        "eval_only": eval_only,
        "refit_posthoc_calibration": refit_posthoc_calibration,
        "refit_decision_calibration": refit_decision_calibration,
        # Retain the old summary key while configs migrate to the clearer name.
        "refit_rejection_threshold": refit_decision_calibration,
        "requested_checkpoint_path": (
            str(requested_checkpoint_path)
            if requested_checkpoint_path is not None
            else None
        ),
        "source_checkpoint_path": (
            str(source_checkpoint_path) if source_checkpoint_path is not None else None
        ),
        "checkpoint_path": str(best_path),
        "checkpoint_stage": CHECKPOINT_STAGE_PIPELINE_FITTED,
        "best_checkpoint_score": best_score,
        "best_val_f1": best_val_f1,
        "best_val_macro_f1": best_val_f1,
        "checkpoint_metric": checkpoint_metric_name,
        "tuning_robust_composite_score": tuning_robust_composite_score,
        "calibration": calibration_summary,
        "classification_threshold": classification_threshold_summary,
        "branch_competence_prior": branch_competence_summary,
        "model_visible_integrity_reference": visible_integrity_summary,
        "conformal_thresholds": conformal_thresholds,
        "risk_control_thresholds": risk_control_thresholds,
        "val": val_metrics,
        "val_selection": val_metrics,
        "val_calibration": val_calibration_metrics,
        "val_posthoc_calibration": val_calibration_metrics,
        "val_conformal_calibration": val_conformal_metrics,
        "validation_split": validation_split_summary,
        "val_robust": val_robust_results,
        "test": test_metrics,
        "robust": robust_results,
        "extra_eval": extra_results,
        "risk_coverage_curves": {
            "test_path": str(risk_coverage_path) if risk_coverage_points else None,
            "test_points": int(risk_coverage_points),
            "extra_eval_path": (
                str(extra_risk_coverage_path)
                if extra_risk_coverage_points
                else None
            ),
            "extra_eval_points": int(extra_risk_coverage_points),
        },
    }
    if rejection_threshold is not None:
        summary["rejection_threshold"] = rejection_threshold
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_json_compatible(summary), f, sort_keys=False)
    logger.info("finished: %s", out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train API+Graph+Manifest robust tri-modal fusion.")
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML configs, applied left to right.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
