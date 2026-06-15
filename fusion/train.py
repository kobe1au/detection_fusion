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


logger = logging.getLogger("tri_modal_robust")

DEFAULT_ROBUST_VAL_SCENARIOS = (
    {"name": "api_graph_degraded_s0.5", "perturb_type": "api_graph_degraded", "strength": 0.5, "weight": 0.25},
    {"name": "manifest_degraded_s0.5", "perturb_type": "manifest_degraded", "strength": 0.5, "weight": 0.15},
    {"name": "all_degraded_s0.5", "perturb_type": "all_degraded", "strength": 0.5, "weight": 0.10},
    {"name": "api_missing", "perturb_type": "api_missing", "strength": 1.0, "weight": 1.0 / 30.0},
    {"name": "graph_missing", "perturb_type": "graph_missing", "strength": 1.0, "weight": 1.0 / 30.0},
    {"name": "manifest_missing", "perturb_type": "manifest_missing", "strength": 1.0, "weight": 1.0 / 30.0},
)


class EmptyExtraEvalSetError(RuntimeError):
    """Raised when an optional external eval set has no usable samples."""


GATE_DIAGNOSTIC_KEYS = (
    "api_integrity",
    "graph_integrity",
    "manifest_integrity",
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
    "temperature_api",
    "temperature_graph",
    "temperature_manifest",
    "temperature_joint",
    "effective_manifest_to_code_conflict",
    "effective_code_to_manifest_conflict",
    "manifest_code_conflict_applicable",
    "total_reliability",
    "final_uncertainty_proxy",
    "effective_conflict",
    "acceptance_score",
    "calibration_active",
    "gate_uses_perturbation_evidence",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_determinism(enabled: bool) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)


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
    return deep_update(cfg, raw)


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
) -> tuple[set[str], set[str], set[str]]:
    ids: set[str] = set()
    packages: set[str] = set()
    families: set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        id_col = next((name for name in ("id", "ID", "Id", "sha256") if name in fields), None)
        pkg_col = next((name for name in ("pkg_name", "package_name", "package") if name in fields), None)
        family_col = next(
            (
                name
                for name in ("family", "malware_family", "family_name", "avclass_family")
                if name in fields
            ),
            None,
        )
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
            if family_col is not None:
                family = str(row.get(family_col, "") or "").strip().lower()
                if family and family not in {"nan", "none", "null", "benign", "goodware"}:
                    families.add(family)
    return ids, packages, families


def validate_split_partitions(cfg: dict, include_test: bool) -> None:
    data_cfg = cfg.get("data", {})
    if not bool(data_cfg.get("strict_partition_isolation", True)):
        return
    data_root = data_cfg.get("root", "")
    split_names = ["train", "val"] + (["test"] if include_test else [])
    identities: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for split in split_names:
        csv_path = resolve(data_root, data_cfg[f"{split}_csv"])
        identities[split] = _read_split_identities(csv_path, split)

    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            id_overlap = sorted(identities[left][0] & identities[right][0])
            pkg_overlap = sorted(identities[left][1] & identities[right][1])
            family_overlap = sorted(identities[left][2] & identities[right][2])
            if id_overlap or pkg_overlap or family_overlap:
                raise ValueError(
                    f"Split leakage between {left} and {right}: "
                    f"id_overlap={len(id_overlap)} examples={id_overlap[:10]}; "
                    f"package_overlap={len(pkg_overlap)} examples={pkg_overlap[:10]}; "
                    f"family_overlap={len(family_overlap)} examples={family_overlap[:10]}"
                )


def _checkpoint_semantic_signature(cfg: dict) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    return {
        "model": copy.deepcopy(cfg.get("model", {}) or {}),
        "fusion": copy.deepcopy(cfg.get("fusion", {}) or {}),
        "semantic_reconstruction": copy.deepcopy(cfg.get("semantic_reconstruction", {}) or {}),
        "calibration": copy.deepcopy(cfg.get("calibration", {}) or {}),
        "selective_prediction": copy.deepcopy(cfg.get("selective_prediction", {}) or {}),
        "data": {
            key: copy.deepcopy(data_cfg.get(key))
            for key in (
                "graph_semantic_source",
                "max_api_events_per_sample",
                "min_pt_schema_version",
                "allow_legacy_pt_compat",
                "strict_observable_schema",
                "manifest_vocab_path",
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
    robust_cfg = cfg.get("robust", {})
    model_cfg = cfg.get("model", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    gate_cfg = model_cfg.get("gate", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    require_manifest_semantic_maps = bool(
        gate_cfg.get("use_consistency_evidence", False)
        or gate_cfg.get("use_conflict_evidence", False)
        or float(loss_cfg.get("cross_source_consistency_weight", 0.0)) > 0.0
        or float(loss_cfg.get("semantic_reconstruction_weight", 0.0)) > 0.0
        or bool((cfg.get("semantic_reconstruction", {}) or {}).get("enabled", False))
    )
    data_root = data_cfg.get("root", "")
    # Resolve the manifest vocab path so old PTs can derive semantic maps at
    # dataset init time.  Explicitly set data.manifest_vocab_path="" to disable.
    manifest_vocab_path = str(data_cfg.get("manifest_vocab_path", ""))
    if manifest_vocab_path == "" and "manifest_vocab_path" not in data_cfg:
        manifest_vocab_path = "config/manifest_vocab.yaml"
    if manifest_vocab_path:
        manifest_vocab_path = resolve(data_root, manifest_vocab_path)
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
        "drop_graph_behavior_hints": bool(model_cfg.get("graph_encoder", {}).get("drop_extracted_behavior_hints", False)),
        "graph_semantic_source": str(data_cfg.get("graph_semantic_source", "alignment")),
        "num_classes": int(model_cfg.get("num_classes", 2)),
        "label_map": data_cfg.get("label_map"),
        "strict_split_integrity": bool(data_cfg.get("strict_split_integrity", True)),
        "allow_pt_superset": False,
        "require_manifest_semantic_maps": require_manifest_semantic_maps,
        "allow_legacy_pt_compat": bool(data_cfg.get("allow_legacy_pt_compat", False)),
        "min_pt_schema_version": int(data_cfg.get("min_pt_schema_version", 0)),
        "strict_observable_schema": bool(data_cfg.get("strict_observable_schema", False)),
        "manifest_vocab_path": manifest_vocab_path,
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
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size" if is_train else "eval_batch_size", train_cfg.get("batch_size", 32))),
        shuffle=is_train,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=bool(train_cfg.get("persistent_workers", False)) and workers > 0,
        collate_fn=robust_collate_fn,
    )


def split_validation_dataset(cfg: dict, dataset) -> tuple[Subset, Subset, dict[str, Any]]:
    """Deterministically separate checkpoint selection from calibration."""
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
    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(str(group), []).append(index)
    if len(group_to_indices) < 2:
        raise ValueError(
            "Validation dataset needs at least two package/sample groups for leakage-free calibration split"
        )
    ranked_groups = sorted(
        group_to_indices,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    target_calibration_size = min(size - 1, max(1, int(round(size * fraction))))
    calibration_groups: list[str] = []
    calibration_indices: list[int] = []
    for group in ranked_groups:
        indices = group_to_indices[group]
        if calibration_indices and len(calibration_indices) >= target_calibration_size:
            break
        if len(calibration_indices) + len(indices) >= size:
            continue
        calibration_groups.append(group)
        calibration_indices.extend(indices)
    calibration_indices = sorted(calibration_indices)
    calibration_set = set(calibration_indices)
    selection_indices = [index for index in range(size) if index not in calibration_set]
    selection_groups = sorted(set(groups[index] for index in selection_indices))
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


def build_model(cfg: dict, feature_dim: int) -> TriModalRobustModel:
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
        semantic_reconstruction_config=cfg.get("semantic_reconstruction", {}) or {},
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
            "auc": 0.0,
            "ap": 0.0,
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
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
    else:
        out["auc"] = 0.0
        out["ap"] = 0.0
    return out


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
    return metrics, rows


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
    parameters = list(model.calibration_parameters())
    if not parameters:
        raise ValueError(
            "calibration.enabled=true requires fusion reliability/probability calibration modules"
        )
    epochs = int(calibration_cfg.get("epochs", 5))
    if epochs <= 0:
        raise ValueError("calibration.epochs must be positive")
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

    epoch_losses: list[float] = []
    total_steps = 0
    try:
        model.eval()
        for epoch in range(1, epochs + 1):
            total = 0.0
            steps = 0
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
                        parameters, float(calibration_cfg.get("grad_clip", 5.0))
                    )
                    optimizer.step()
                    total += float(loss.detach().item())
                    steps += 1
                    total_steps += 1
            epoch_loss = total / max(steps, 1)
            epoch_losses.append(epoch_loss)
            logger.info("posthoc_calibration epoch=%s loss=%.6f", epoch, epoch_loss)
    finally:
        for param in model.parameters():
            param.requires_grad_(previous_requires_grad[id(param)])
    if total_steps == 0:
        model.set_calibration_active(False)
        raise RuntimeError(
            "Post-hoc calibration received no valid batches; refusing to use unfitted "
            "reliability and temperature parameters"
        )
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
        "losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
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
                batch=graph,
                evidence=extra.get("gate_evidence"),
                semantic_reconstruction_cfg=cfg.get("semantic_reconstruction", {}) or {},
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
                if key.startswith(("loss_recon_", "mask_rate_", "valid_recon_"))
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


def _write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run(cfg: dict) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO))
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    configure_determinism(bool(train_cfg.get("deterministic", False)))
    device = select_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("use_amp", True))
    run_test = bool(eval_cfg.get("run_test", True))
    run_robust_test = bool(eval_cfg.get("run_robust_test", True))
    eval_only = bool(eval_cfg.get("eval_only", False))
    tuning_mode = bool(train_cfg.get("tuning_mode", False))
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
        if str(train_cfg.get("checkpoint_metric", "")).strip().lower() != "robust_composite":
            raise ValueError("train.tuning_mode=true requires train.checkpoint_metric=robust_composite")
    if eval_only:
        if tuning_mode:
            raise ValueError("eval.eval_only=true is incompatible with train.tuning_mode=true")
        if not str(eval_cfg.get("checkpoint_path") or "").strip():
            raise ValueError("eval.eval_only=true requires eval.checkpoint_path")

    validate_split_partitions(cfg, include_test=run_test)
    val_ds = build_dataset(cfg, "val", is_train=False)
    calibration_enabled = bool((cfg.get("calibration", {}) or {}).get("enabled", False))
    selective_enabled = bool((cfg.get("selective_prediction", {}) or {}).get("enabled", False))
    needs_validation_holdout = calibration_enabled or selective_enabled
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
        best_score = float(ckpt.get("checkpoint_score", -1.0))
        best_val_f1 = float((ckpt.get("val") or {}).get("macro_f1", -1.0))
        checkpoint_metric_name = str(ckpt.get("checkpoint_metric", "loaded_checkpoint"))
        calibration_summary = dict(ckpt.get("calibration") or {"enabled": False})
        rejection_threshold = ckpt.get("rejection_threshold")
        if bool((cfg.get("selective_prediction", {}) or {}).get("enabled", False)) and rejection_threshold is None:
            raise ValueError(
                "Selective eval-only mode requires rejection_threshold saved in the checkpoint"
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

        for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
            val_metrics, _ = evaluate(model, val_loader, device, use_amp, "val", dump_rows=False)
            enforce_failed_ratio(val_metrics, cfg, "val")
            val_robust_metrics = evaluate_robust_validation(model, robust_val_loaders, device, use_amp, cfg)
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
        rejection_threshold = fit_rejection_threshold(
            val_calibration_rows, cfg.get("selective_prediction", {}) or {}
        )
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
            ckpt["model"] = model.state_dict()
            ckpt["calibration"] = calibration_summary
            ckpt["rejection_threshold"] = rejection_threshold
            ckpt["validation_split"] = validation_split_summary
            torch.save(ckpt, best_path)
    val_calibration_metrics.update(
        _selective_metrics_from_rows(val_calibration_rows, rejection_threshold)
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
    summary = {
        "eval_only": eval_only,
        "checkpoint_path": str(best_path),
        "best_checkpoint_score": best_score,
        "best_val_f1": best_val_f1,
        "best_val_macro_f1": best_val_f1,
        "checkpoint_metric": checkpoint_metric_name,
        "calibration": calibration_summary,
        "rejection_threshold": rejection_threshold,
        "val": val_metrics,
        "val_selection": val_metrics,
        "val_calibration": val_calibration_metrics,
        "validation_split": validation_split_summary,
        "val_robust": val_robust_results,
        "test": test_metrics,
        "robust": robust_results,
        "extra_eval": extra_results,
    }
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
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
