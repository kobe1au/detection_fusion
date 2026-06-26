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
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

from fusion.losses import compute_posthoc_calibration_loss, compute_robust_loss
from fusion.dataset import (
    RobustTriModalDataset,
    prepare_robust_batch,
    robust_collate_fn,
)
from fusion.model import TriModalRobustModel
from fusion.perturbations import EVAL_PERTURB_TYPES
from fusion.utils import build_grad_scaler, get_amp_context
from fusion.constants import TriModalConfigDefaults


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


GATE_DIAGNOSTIC_KEYS = (
    "api_integrity",
    "api_encoder_coverage",
    "effective_api_integrity",
    "api_truncated_by_encoder_budget",
    "api_integrity_before_encoder_budget",
    "graph_integrity",
    "graph_encoder_coverage",
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
    "fallback_used",
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
    "acceptance_score",
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
    selective_cfg = cfg.get("selective_prediction", {}) or {}
    return {
        "model": copy.deepcopy(cfg.get("model", {}) or {}),
        "fusion": fusion_cfg,
        "calibration": copy.deepcopy(cfg.get("calibration", {}) or {}),
        # Coverage changes only the validation-fitted rejection threshold.
        "selective_prediction": {"enabled": bool(selective_cfg.get("enabled", False))},
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

        # Cache
        "cache_mode",
        "cache_dir",
        "cache_tag",
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
        # Cache
        "cache_mode": str(data_cfg.get("cache_mode", "none")),
        "cache_dir": (
            resolve(data_cfg.get("root", ""), data_cfg["cache_dir"])
            if str(data_cfg.get("cache_dir", "") or "").strip()
            else None
        ),
        "cache_tag": str(data_cfg.get("cache_tag", "base_processed")),
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


def build_loader(cfg: dict, dataset, is_train: bool):
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
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(train_cfg.get("batch_size" if is_train else "eval_batch_size", train_cfg.get("batch_size", 32))),
        "shuffle": is_train,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(train_cfg.get("persistent_workers", False)) and workers > 0,
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
    keeps both halves close to the full validation label distribution.
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

    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(str(group), []).append(index)
    if len(group_to_indices) < 2:
        raise ValueError(
            "Validation dataset needs at least two package/sample groups for leakage-free calibration split"
        )

    label_values = sorted(set(labels))
    total_label_counts = {
        label: sum(int(value == label) for value in labels) for label in label_values
    }
    target_label_counts = {
        label: float(total_label_counts[label]) * fraction for label in label_values
    }
    target_calibration_size = min(size - 1, max(1, int(round(size * fraction))))
    ranked_groups = sorted(
        group_to_indices,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    rank = {group: index for index, group in enumerate(ranked_groups)}
    group_label_counts: dict[str, dict[int, int]] = {}
    for group, indices in group_to_indices.items():
        group_label_counts[group] = {
            label: sum(int(labels[index] == label) for index in indices)
            for label in label_values
        }

    def split_error(candidate_size: int, candidate_counts: dict[int, int]) -> float:
        size_error = abs(candidate_size - target_calibration_size) / max(size, 1)
        label_error = sum(
            abs(candidate_counts[label] - target_label_counts[label])
            / max(total_label_counts[label], 1)
            for label in label_values
        )
        return size_error + label_error

    remaining = set(ranked_groups)
    calibration_groups: list[str] = []
    calibration_indices: list[int] = []
    calibration_label_counts = {label: 0 for label in label_values}
    while remaining:
        current_size = len(calibration_indices)
        current_error = split_error(current_size, calibration_label_counts)
        candidates = []
        for group in remaining:
            indices = group_to_indices[group]
            new_size = current_size + len(indices)
            if new_size >= size:
                continue
            new_counts = {
                label: calibration_label_counts[label]
                + group_label_counts[group][label]
                for label in label_values
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
        calibration_label_counts = best_counts
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
            "selection_indices": selection_indices,
            "calibration_indices": calibration_indices,
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


@torch.no_grad()
def evaluate_robust_validation(
    model,
    loaders: list[dict[str, Any]],
    device,
    use_amp: bool,
    cfg: dict,
    selective_threshold: float | None = None,
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
    elif configured_fusion_mode not in {"", "legacy_learned_gate"}:
        raise ValueError(f"Unsupported fusion.mode: {configured_fusion_mode}")

    # Independence guardrail: evidential combination (subjective-logic / DS /
    # cumulative) treats API, Graph and Manifest as INDEPENDENT evidence sources.
    # Two switches would break that premise by making one branch's input or
    # reliability depend on another modality, so they are incompatible with the
    # evidential combination rules and must be turned off there.
    combination = str(fusion_cfg.get("combination", "linear")).lower()
    if combination in {"yager", "dempster", "cumulative"}:
        coupling = []
        if bool((fusion_cfg.get("reliability_calibration", {}) or {}).get("use_relation_evidence", False)):
            coupling.append("fusion.reliability_calibration.use_relation_evidence")
        if bool(graph_cfg.get("use_behavior_hint", False)):
            coupling.append("model.graph_encoder.use_behavior_hint")
        if coupling:
            raise ValueError(
                f"fusion.combination={combination} assumes independent modality evidence, "
                f"but these settings couple modalities at the input/reliability level: {coupling}. "
                "Disable them, or use fusion.combination=linear for a coupled (legacy) pipeline."
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
    correct = (pred_arr == y.astype(np.int64)).astype(np.float64)
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
            reliability = _finite_row_float(row, f"predicted_reliability_{branch}")
            correctness = _finite_row_float(row, f"{branch}_correct")
            if reliability is None or correctness is None:
                continue
            reliability_values.append(min(1.0, max(0.0, reliability)))
            correctness_values.append(1.0 if correctness >= 0.5 else 0.0)
        count = len(reliability_values)
        out[f"{branch}_reliability_count"] = count
        if count == 0:
            out[f"{branch}_reliability_auc_defined"] = 0
            out[f"{branch}_reliability_ap_defined"] = 0
            continue
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

    max_score = max(scores.values()) if scores else 0.0
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
    """Fit clean calibration references for model-visible evidence integrity."""
    modifier_cfg = (cfg.get("fusion", {}) or {}).get("visible_integrity_modifier", {}) or {}
    if not bool(modifier_cfg.get("enabled", False)):
        return {"enabled": False}
    min_reference = float(modifier_cfg.get("min_reference", 1.0e-6))
    beta = float(modifier_cfg.get("beta", 1.0))
    min_value = float(modifier_cfg.get("min_value", 0.5))
    if not math.isfinite(min_reference) or not 0.0 < min_reference <= 1.0:
        raise ValueError("fusion.visible_integrity_modifier.min_reference must be within (0, 1]")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("fusion.visible_integrity_modifier.beta must be finite and positive")
    if not math.isfinite(min_value) or not 0.0 <= min_value <= 1.0:
        raise ValueError("fusion.visible_integrity_modifier.min_value must be within [0, 1]")

    def _effective_value(row: dict[str, Any], branch: str) -> float | None:
        direct = _finite_row_float(row, f"effective_{branch}_integrity")
        if direct is not None:
            return min(1.0, max(0.0, direct))
        integrity = _finite_row_float(row, f"{branch}_integrity")
        if integrity is None:
            return None
        if branch == "api":
            coverage = _finite_row_float(row, "api_encoder_coverage")
        elif branch == "graph":
            coverage = _finite_row_float(row, "graph_encoder_coverage")
        else:
            coverage = 1.0
        if coverage is None:
            coverage = 1.0
        return min(1.0, max(0.0, integrity * coverage))

    references: dict[str, float] = {}
    counts: dict[str, int] = {}
    for branch in ("api", "graph", "manifest"):
        values = [
            value
            for row in rows
            if (value := _effective_value(row, branch)) is not None and math.isfinite(value)
        ]
        counts[branch] = len(values)
        if not values:
            references[branch] = 1.0
            continue
        references[branch] = float(min(1.0, max(min_reference, float(np.median(values)))))
    return {
        "enabled": True,
        "metric": "median_clean_effective_integrity",
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


def fit_rejection_threshold(rows: list[dict[str, Any]], config: dict | None = None) -> float | None:
    """Choose an acceptance threshold on validation data for target coverage."""
    config = config or {}
    if not bool(config.get("enabled", False)):
        return None
    if str(config.get("mode", "threshold")).lower() == "conformal":
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


# I3: class-conditional (Mondrian) conformal selective prediction
# Replaces the heuristic coverage threshold with split-conformal calibration so
# the abstention has a finite-sample guarantee: on exchangeable data,
# P(true label in prediction set | label = c) >= 1 - alpha. This abstains on
# low-evidence / high-conflict degraded samples rather than forcing a confident
# decision. Note the guarantee is on coverage, NOT a bound on the rejection rate.

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
    scores: dict[int, list[float]] = {0: [], 1: []}
    for row in rows:
        prob = row.get("prob_malware")
        label = row.get("label")
        if prob is None or label is None:
            continue
        p1 = float(prob)
        label = int(label)
        # Nonconformity of the TRUE class, computed identically to the
        # prediction-set test below (malware: 1 - p1, benign: p1) so calibration
        # and inference scores match exactly and avoid float knife-edges.
        nonconformity = (1.0 - p1) if label == 1 else p1
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
        "q_benign": q_benign,
        "q_malware": q_malware,
        "num_calibration": int(len(scores.get(0, [])) + len(scores.get(1, []))),
    }


def _uses_conformal_selective(config: dict | None = None) -> bool:
    config = config or {}
    return bool(config.get("enabled", False)) and str(config.get("mode", "threshold")).lower() == "conformal"


def _conformal_prediction_set(p1: float, thresholds: dict[str, Any]) -> tuple[bool, bool]:
    """Return (include_benign, include_malware) for the conformal prediction set."""
    q_benign = float(thresholds.get("q_benign", float("inf")))
    q_malware = float(thresholds.get("q_malware", float("inf")))
    include_malware = (1.0 - p1) <= q_malware  # nonconformity of class malware
    include_benign = p1 <= q_benign            # nonconformity of class benign
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
    for row in valid:
        p1 = float(row["prob_malware"])
        y = int(row["label"])
        include_benign, include_malware = _conformal_prediction_set(p1, thresholds)
        size = int(include_benign) + int(include_malware)
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
        "conformal_benign_acceptance_rate": _ratio(per_class_accepted[0], per_class_total[0]),
        "conformal_malware_acceptance_rate": _ratio(per_class_accepted[1], per_class_total[1]),
        "conformal_malware_rejection_rate": (
            None if per_class_total[1] == 0
            else 1.0 - (per_class_accepted[1] / per_class_total[1])
        ),
        # Malware missed despite being accepted -- the dangerous failure mode.
        "conformal_malware_fn_after_rejection": _ratio(malware_fn_accepted, per_class_total[1]),
        "conformal_selective_risk": _ratio(accepted_errors, accepted),
        "conformal_selective_acc": (None if accepted == 0 else 1.0 - accepted_errors / accepted),
        # TRUE conformal coverage: P(true label in prediction set | label = c).
        # Should be >= 1 - alpha on exchangeable calibration/test data.
        "conformal_empirical_coverage_benign": _ratio(true_in_set[0], per_class_total[0]),
        "conformal_empirical_coverage_malware": _ratio(true_in_set[1], per_class_total[1]),
        "conformal_num_accepted": int(accepted),
        "conformal_num_rejected": int(n - accepted),
    }
    return out


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp: bool,
    split_name: str,
    dump_rows: bool = False,
    selective_threshold: float | None = None,
):
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    preds_all: list[int] = []
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
        pred = logits.argmax(dim=-1)
        labels_all.extend(labels.detach().cpu().long().tolist())
        probs_all.extend(prob.detach().cpu().tolist())
        preds_all.extend(pred.detach().cpu().long().tolist())
        acceptance = extra.get("acceptance_score")
        if isinstance(acceptance, torch.Tensor):
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
                    "final_confidence": float(torch.softmax(logits.float(), dim=-1)[i].max().detach().cpu().item()),
                    "correct": int(pred[i].detach().cpu().item() == labels[i].detach().cpu().item()),
                    "year": int(batch.get("years")[i].detach().cpu().item()) if batch.get("years") is not None else 0,
                }
                row.update(_branch_prediction_row(extra, labels, i))
                if isinstance(acceptance, torch.Tensor) and acceptance.numel() > i:
                    acceptance_i = float(acceptance.view(-1)[i].detach().cpu().item())
                    row["acceptance_score"] = acceptance_i
                    if selective_threshold is not None:
                        row["rejected"] = int(acceptance_i < selective_threshold)
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
                rows.append(row)

    metrics = _metrics(labels_all, probs_all, preds_all)
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

@torch.no_grad()
def collect_posthoc_calibration_cache(
    model,
    loaders: list,
    device,
    use_amp: bool,
) -> list[dict[str, torch.Tensor]]:
    """Cache frozen-model branch logits and evidence for post-hoc calibration."""
    model.eval()
    cached: list[dict[str, torch.Tensor]] = []
    required = (
        "api_logits_aux",
        "graph_logits_aux",
        "manifest_logits_aux",
        "joint_logits_aux",
        "gate_evidence",
    )

    for loader in loaders:
        for batch in tqdm(loader, desc="collect calibration cache", leave=False):
            graph, labels, _, _quality, _failed = prepare_robust_batch(batch, device)
            if graph is None:
                continue
            with get_amp_context(device, use_amp):
                _logits, extra = model(graph, return_features=False)

            missing = [key for key in required if key not in extra]
            if missing:
                raise RuntimeError(f"Missing calibration cache fields: {missing}")

            cached.append(
                {
                    "labels": labels.detach().cpu(),
                    "api_logits_aux": extra["api_logits_aux"].detach().cpu(),
                    "graph_logits_aux": extra["graph_logits_aux"].detach().cpu(),
                    "manifest_logits_aux": extra["manifest_logits_aux"].detach().cpu(),
                    "joint_logits_aux": extra["joint_logits_aux"].detach().cpu(),
                    "gate_evidence": extra["gate_evidence"].detach().cpu(),
                }
            )

    if not cached:
        raise RuntimeError("Post-hoc calibration cache collected no valid batches")
    return cached


def calibration_step_from_cached_batch(
    model,
    item: dict[str, torch.Tensor],
    device,
    cfg: dict,
):
    labels = item["labels"].to(device, non_blocking=True)
    api_logits = item["api_logits_aux"].to(device, non_blocking=True)
    graph_logits = item["graph_logits_aux"].to(device, non_blocking=True)
    manifest_logits = item["manifest_logits_aux"].to(device, non_blocking=True)
    joint_logits = item["joint_logits_aux"].to(device, non_blocking=True)
    evidence = item["gate_evidence"].to(device, non_blocking=True)

    outputs = model.discount_fusion(
        api_logits,
        graph_logits,
        manifest_logits,
        joint_logits,
        evidence,
    )
    outputs.update(
        {
            "api_logits_aux": api_logits,
            "graph_logits_aux": graph_logits,
            "manifest_logits_aux": manifest_logits,
            "joint_logits_aux": joint_logits,
            "gate_evidence": evidence,
        }
    )
    return compute_posthoc_calibration_loss(
        outputs,
        labels,
        evidence,
        cfg.get("fusion", {}) or {},
    )

def fit_posthoc_calibration(
    model,
    loaders: list,
    device,
    use_amp: bool,
    cfg: dict,
) -> dict[str, Any]:
    """Fit only monotonic reliability and temperature parameters on validation."""
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
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(calibration_cfg.get("lr", 1.0e-3)),
        weight_decay=float(calibration_cfg.get("weight_decay", 0.0)),
    )
    model.set_calibration_active(True)
    previous_requires_grad = {id(param): param.requires_grad for param in model.parameters()}
    calibration_ids = {id(param) for param in parameters}
    for param in model.parameters():
        param.requires_grad_(id(param) in calibration_ids)

    # Cache
    cached_calibration_batches = None
    if bool(calibration_cfg.get("cache_branch_outputs", False)):
        cached_calibration_batches = collect_posthoc_calibration_cache(
            model,
            loaders,
            device,
            use_amp,
        )

    epoch_losses: list[float] = []
    total_steps = 0
    best_loss = float("inf")
    best_epoch = 0
    best_parameters: list[torch.Tensor] | None = None
    stale_epochs = 0
    stopped_early = False
    try:
        model.eval()
        for epoch in range(1, epochs + 1):
            total = 0.0
            steps = 0
            if cached_calibration_batches is not None:
                for item in tqdm(cached_calibration_batches, desc=f"calibrate {epoch}", leave=False):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _parts = calibration_step_from_cached_batch(
                        model,
                        item,
                        device,
                        cfg,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        float(calibration_cfg.get("grad_clip", 5.0)),
                    )
                    optimizer.step()
                    total += float(loss.detach().item())
                    steps += 1
                    total_steps += 1
            else:
                for loader in loaders:
                    for batch in tqdm(loader, desc=f"calibrate {epoch}", leave=False):
                        graph, labels, _, _quality, _failed = prepare_robust_batch(batch, device)
                        if graph is None:
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        with get_amp_context(device, use_amp):
                            _logits, extra = model(graph, return_features=False)
                        loss, _parts = compute_posthoc_calibration_loss(
                            extra,
                            labels,
                            extra.get("gate_evidence"),
                            cfg.get("fusion", {}) or {},
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            float(calibration_cfg.get("grad_clip", 5.0)),
                        )
                        optimizer.step()
                        total += float(loss.detach().item())
                        steps += 1
                        total_steps += 1
            epoch_loss = total / max(steps, 1)
            epoch_losses.append(epoch_loss)
            logger.info("posthoc_calibration epoch=%s loss=%.6f", epoch, epoch_loss)
            if steps > 0 and epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                best_epoch = epoch
                best_parameters = [param.detach().clone() for param in parameters]
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    stopped_early = True
                    break
    finally:
        for param in model.parameters():
            param.requires_grad_(previous_requires_grad[id(param)])
    if total_steps == 0:
        model.set_calibration_active(False)
        raise RuntimeError(
            "Post-hoc calibration received no valid batches; refusing to use unfitted "
            "reliability and temperature parameters"
        )
    if best_parameters is None:
        raise RuntimeError("Post-hoc calibration did not produce a valid optimization state")
    with torch.no_grad():
        for parameter, best_value in zip(parameters, best_parameters):
            parameter.copy_(best_value)
    temperatures = {}
    for name in ("api", "graph", "manifest", "joint"):
        if model.discount_fusion.temperature_parameters is not None:
            temperatures[name] = float(
                (torch.nn.functional.softplus(
                    model.discount_fusion.temperature_parameters[name].detach()
                ) + 1.0e-4).cpu().item()
            )
    return {
        "enabled": True,
        "epochs": epochs,
        "epochs_ran": len(epoch_losses),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "losses": epoch_losses,
        "final_loss": best_loss,
        "temperatures": temperatures,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int):
    model.train()
    use_amp = bool(cfg["train"].get("use_amp", True))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg["label_smoothing"] = float(cfg["train"].get("label_smoothing", loss_cfg.get("label_smoothing", 0.0)))
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    loss_part_sums: dict[str, float] = {}
    diagnostic_sums: dict[str, float] = {}
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
            )
            loss = loss / max(grad_accum, 1)
        for key in GATE_DIAGNOSTIC_KEYS:
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                finite = value.detach().float().view(-1)
                finite = finite[torch.isfinite(finite)]
                if finite.numel() > 0:
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(finite.sum().cpu().item())
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(finite.numel())
        for key, value in parts.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                loss_part_sums[key] = loss_part_sums.get(key, 0.0) + float(value)
        steps += 1
        scaler.scale(loss).backward()
        if steps % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().item()) * max(grad_accum, 1)

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
                f"{key}={value / steps:.4f}"
                for key, value in sorted(loss_part_sums.items())
                if key in {"loss", "ce", "branch_aux", "branch_aux_weight"}
            ),
        )
        logger.info(
            "train_fusion_diagnostics epoch=%s %s",
            epoch,
            " ".join(
                f"mean_{key}={value / max(diagnostic_counts.get(key, 0), 1):.4f}"
                for key, value in sorted(diagnostic_sums.items())
                if key.startswith(("discount_", "fusion_weight_", "entropy_", "margin_", "uncertainty_proxy_"))
                or key == "fallback_used"
            ),
        )
    return total_loss / max(steps, 1)


def write_gate_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    root = str(item.get("root", cfg.get("data", {}).get("root", "")) or "")
    pt_value = item.get("pt_dir") or item.get("test_pt_dir")
    csv_value = item.get("csv") or item.get("csv_path") or item.get("test_csv")
    if not pt_value:
        raise ValueError(f"extra eval set {item.get('name', '<unnamed>')} is missing pt_dir")
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
    refit_rejection_threshold = bool(eval_cfg.get("refit_rejection_threshold", False))
    tuning_mode = bool(train_cfg.get("tuning_mode", False))
    calibration_enabled = bool((cfg.get("calibration", {}) or {}).get("enabled", False))
    selective_enabled = bool((cfg.get("selective_prediction", {}) or {}).get("enabled", False))
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
        if _normalize_extra_eval_sets(eval_cfg.get("extra_sets")):
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

    validate_split_partitions(cfg, include_test=run_test)
    val_ds = build_dataset(cfg, "val", is_train=False)
    holdout_enabled = bool((cfg.get("calibration", {}) or {}).get("holdout_enabled", False))
    needs_validation_holdout = calibration_enabled or selective_enabled or holdout_enabled
    if needs_validation_holdout:
        val_selection_ds, val_calibration_ds, validation_split = split_validation_dataset(cfg, val_ds)
        selection_indices = list(validation_split["selection_indices"])
        calibration_indices = list(validation_split["calibration_indices"])
    else:
        val_selection_ds = val_ds
        val_calibration_ds = val_ds
        selection_indices = None
        calibration_indices = None
        validation_split = {
            "split_seed": None,
            "validation_fraction": 1.0,
            "num_selection": len(val_ds),
            "num_calibration": len(val_ds),
            "selection_indices": None,
            "calibration_indices": None,
        }
    validation_split_summary = {
        key: value
        for key, value in validation_split.items()
        if key not in {"selection_indices", "calibration_indices"}
    }
    val_loader = build_loader(cfg, val_selection_ds, is_train=False)
    val_calibration_loader = build_loader(cfg, val_calibration_ds, is_train=False)
    train_ds = None
    train_loader = None
    if not eval_only:
        train_ds = build_dataset(cfg, "train", is_train=True)
        train_loader = build_loader(cfg, train_ds, is_train=True)
    robust_val_loaders = build_robust_val_loaders(cfg, selection_indices)
    robust_calibration_loaders = (
        build_robust_val_loaders(cfg, calibration_indices)
        if calibration_enabled
        and bool((cfg.get("calibration", {}) or {}).get("include_robust_val", True))
        else []
    )
    test_loader = None
    if run_test:
        test_ds = build_dataset(cfg, "test", is_train=False)
        test_loader = build_loader(cfg, test_ds, is_train=False)

    feature_dim = train_ds.feature_dim if train_ds is not None else val_ds.feature_dim
    model = build_model(cfg, feature_dim).to(device)

    exp_name = str(train_cfg.get("exp_name", "tri_modal_robust"))
    if eval_only:
        exp_name = str(eval_cfg.get("output_name") or f"{exp_name}_eval_only")
    out_dir = Path(data_cfg.get("out_dir", "experiments")) / exp_name / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if eval_only:
        best_path = Path(str(eval_cfg["checkpoint_path"]))
        if not best_path.is_absolute():
            best_path = Path.cwd() / best_path
        if not best_path.exists():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        validate_eval_checkpoint_config(
            cfg,
            ckpt.get("cfg"),
            allow_mismatch=bool(eval_cfg.get("allow_checkpoint_config_mismatch", False)),
        )
        model.load_state_dict(ckpt["model"])
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
        ):
            logger.warning(
                "fusion.visible_integrity_modifier.enabled=true but checkpoint has no fitted reference; using neutral modifiers"
            )
        best_score = float(ckpt.get("checkpoint_score", -1.0))
        best_val_f1 = float((ckpt.get("val") or {}).get("macro_f1", -1.0))
        checkpoint_metric_name = str(ckpt.get("checkpoint_metric", "loaded_checkpoint"))
        calibration_summary = dict(ckpt.get("calibration") or {"enabled": False})
        selective_cfg = cfg.get("selective_prediction", {}) or {}
        rejection_threshold = ckpt.get("rejection_threshold")
        conformal_thresholds = ckpt.get("conformal_thresholds")
        if bool(selective_cfg.get("enabled", False)) and not refit_rejection_threshold:
            if _uses_conformal_selective(selective_cfg):
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
        logger.info("eval-only mode loaded checkpoint: %s", best_path)
    else:
        assert train_loader is not None
        model.set_calibration_active(False)
        for parameter in model.calibration_parameters():
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
        best_path = out_dir / "best_tri_modal_robust.pt"
        patience = int(train_cfg.get("patience", 10))
        stale = 0
        run_epoch_robust_val = checkpoint_requires_robust_validation(cfg)

        for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
            val_metrics, _ = evaluate(model, val_loader, device, use_amp, "val", dump_rows=False)
            enforce_failed_ratio(val_metrics, cfg, "val")
            val_robust_metrics = (
                evaluate_robust_validation(model, robust_val_loaders, device, use_amp, cfg)
                if run_epoch_robust_val
                else {}
            )
            score, checkpoint_metric_name = checkpoint_score(
                cfg,
                val_metrics,
                val_robust_metrics,
                robust_val_loaders,
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
                        "epoch": epoch,
                    },
                    best_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
        calibration_loaders = [val_calibration_loader]
        calibration_loaders.extend(item["loader"] for item in robust_calibration_loaders)
        calibration_summary = fit_posthoc_calibration(
            model,
            calibration_loaders,
            device,
            use_amp,
            cfg,
        )
        rejection_threshold = None

    val_calibration_metrics, val_calibration_rows = evaluate(
        model,
        val_calibration_loader,
        device,
        use_amp,
        "val_calibration",
        dump_rows=True,
    )
    enforce_failed_ratio(val_calibration_metrics, cfg, "val_calibration")
    if not eval_only:
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
                val_calibration_loader,
                device,
                use_amp,
                "val_calibration",
                dump_rows=True,
            )
            enforce_failed_ratio(val_calibration_metrics, cfg, "val_calibration")
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
                val_calibration_loader,
                device,
                use_amp,
                "val_calibration",
                dump_rows=True,
            )
            enforce_failed_ratio(val_calibration_metrics, cfg, "val_calibration")
    if not eval_only or refit_rejection_threshold:
        rejection_threshold = fit_rejection_threshold(
            val_calibration_rows, cfg.get("selective_prediction", {}) or {}
        )
        conformal_thresholds = fit_conformal_thresholds(
            val_calibration_rows, cfg.get("selective_prediction", {}) or {}
        )
        if not eval_only and best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
            ckpt["model"] = model.state_dict()
            ckpt["calibration"] = calibration_summary
            ckpt["branch_competence_prior"] = branch_competence_summary
            ckpt["model_visible_integrity_reference"] = visible_integrity_summary
            if rejection_threshold is not None:
                ckpt["rejection_threshold"] = rejection_threshold
            else:
                ckpt.pop("rejection_threshold", None)
            ckpt["conformal_thresholds"] = conformal_thresholds
            ckpt["validation_split"] = validation_split_summary
            torch.save(ckpt, best_path)
    val_calibration_metrics.update(
        _selective_metrics_from_rows(val_calibration_rows, rejection_threshold)
    )
    val_calibration_metrics.update(
        conformal_selective_metrics(val_calibration_rows, conformal_thresholds)
    )

    val_metrics, val_rows = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        "val_selection",
        dump_rows=True,
        selective_threshold=rejection_threshold,
    )
    enforce_failed_ratio(val_metrics, cfg, "val_selection")
    val_metrics.update(conformal_selective_metrics(val_rows, conformal_thresholds))
    val_robust_results = evaluate_robust_validation(
        model,
        robust_val_loaders,
        device,
        use_amp,
        cfg,
        selective_threshold=rejection_threshold,
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
        )
        enforce_failed_ratio(test_metrics, cfg, "test_clean")
        # Conformal selective metrics on clean test (test_rows is still clean
        # here -- robust rows are appended only inside the loop below).
        test_metrics.update(conformal_selective_metrics(test_rows, conformal_thresholds))
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
                    )
                    enforce_failed_ratio(metrics, cfg, f"test_{result_key}")
                    metrics.update(conformal_selective_metrics(rows, conformal_thresholds))
                    robust_results[result_key] = metrics
                    test_rows.extend(rows)

    all_rows = val_rows + val_calibration_rows + test_rows

    extra_results = {}
    extra_rows: list[dict[str, Any]] = []
    for idx, extra in enumerate(_normalize_extra_eval_sets(eval_cfg.get("extra_sets"))):
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
                extra_results[name] = {
                    "skipped": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "pt_dir": str(pt_dir),
                    "csv": str(csv_path),
                }
                continue
            raise
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
        )
        enforce_failed_ratio(metrics, cfg, split_name, max_failed_ratio=extra.get("max_failed_ratio"))
        metrics.update(conformal_selective_metrics(rows, conformal_thresholds))
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
        "eval_only": eval_only,
        "checkpoint_path": str(best_path),
        "best_checkpoint_score": best_score,
        "best_val_f1": best_val_f1,
        "best_val_macro_f1": best_val_f1,
        "checkpoint_metric": checkpoint_metric_name,
        "tuning_robust_composite_score": tuning_robust_composite_score,
        "calibration": calibration_summary,
        "branch_competence_prior": branch_competence_summary,
        "model_visible_integrity_reference": visible_integrity_summary,
        "conformal_thresholds": conformal_thresholds,
        "val": val_metrics,
        "val_selection": val_metrics,
        "val_calibration": val_calibration_metrics,
        "validation_split": validation_split_summary,
        "val_robust": val_robust_results,
        "test": test_metrics,
        "robust": robust_results,
        "extra_eval": extra_results,
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
