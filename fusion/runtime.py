"""Shared runtime primitives for CARE-Droid and registered baselines.

This module deliberately contains no model-specific calibration, routing, or
post-hoc training logic.  It owns only configuration loading, deterministic
runtime setup, dataset/loader construction, immutable split protocols,
artifact hashing/saving, failure accounting, and controlled evaluation views.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fusion.constants import (
    VALIDATION_HOLDOUT_FRACTION,
    TriModalConfigDefaults,
)
from fusion.dataset import RobustTriModalDataset, robust_collate_fn
from fusion.perturbations import EVAL_PERTURB_TYPES
from fusion.utils import strict_finite_integer
from fusion.view_protocol import (
    CONTROLLED_TEST_VIEW_PROTOCOL_SEED,
    CONTROLLED_VIEW_MECHANISM_VERSION,
    CONTROLLED_VIEW_SEED_FORMULA,
    fixed_test_view_plan,
    seed_manifest_sha256,
)


logger = logging.getLogger("tri_modal_runtime")

VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION = 2
STAGE_A_EXPERT_TRAIN_LOADER_NAMESPACE = "stage_a/expert_train"
STAGE_A_EXPERT_VAL_LOADER_NAMESPACE = "stage_a/expert_val"


class EmptyExtraEvalSetError(RuntimeError):
    """Raised when an optional external evaluation set has no usable sample."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strict_bool_value(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean, got {value!r}")
    return value


def _namespaced_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable positive torch seed without consuming global RNG."""

    payload = f"{int(base_seed)}|{str(namespace)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (
        2**63 - 1
    )


def seed_data_loader_worker(_worker_id: int) -> None:
    """Seed worker-local RNGs from PyTorch's explicit loader seed."""

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


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


def configure_multiprocessing_sharing(cfg: Mapping[str, Any]) -> None:
    train_cfg = cfg.get("train", {}) or {}
    strategy = str(
        train_cfg.get("multiprocessing_sharing_strategy", "") or ""
    ).strip()
    if not strategy or strategy.lower() in {"default", "none", "false"}:
        return
    mp = torch.multiprocessing
    try:
        available = set(mp.get_all_sharing_strategies())
    except (AttributeError, RuntimeError):
        available = set()
    if available and strategy not in available:
        logger.warning(
            "train.multiprocessing_sharing_strategy=%s is unavailable; "
            "available=%s",
            strategy,
            sorted(available),
        )
        return
    try:
        mp.set_sharing_strategy(strategy)
        logger.info("torch_multiprocessing_sharing_strategy=%s", strategy)
    except (AttributeError, RuntimeError) as exc:
        logger.warning(
            "Unable to set torch multiprocessing sharing strategy %s: %s",
            strategy,
            exc,
        )


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


def _uses_tri_modal_defaults(cfg: dict) -> bool:
    method_cfg = (cfg or {}).get("method") or {}
    if not isinstance(method_cfg, dict):
        raise ValueError("method must be a mapping when provided")
    return bool(
        str(method_cfg.get("name") or "").strip()
        or str(method_cfg.get("protocol_id") or "").strip()
    )


_SELECTIVE_PREDICTION_KNOWN_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "threshold_score",
        "risk_level",
        "risk_target",
        "min_calibration_malware",
        "require_feasible",
    }
)
_SELECTIVE_PREDICTION_MODES = frozenset({"risk_control"})


def _selective_prediction_mode(config: dict | None = None) -> str:
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("selective_prediction must be a mapping")
    mode = str(config.get("mode", "risk_control")).strip().lower()
    if mode not in _SELECTIVE_PREDICTION_MODES:
        raise ValueError(
            "selective_prediction.mode must be one of "
            f"{sorted(_SELECTIVE_PREDICTION_MODES)}, got {mode!r}"
        )
    return mode


def _strict_config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"selective_prediction.{key} must be boolean")
    return value


def _canonicalize_selective_prediction_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg or {})
    raw_value = out.get("selective_prediction", {})
    raw = {} if raw_value is None else raw_value
    if not isinstance(raw, dict):
        raise ValueError("selective_prediction must be a mapping")
    unknown = set(raw) - _SELECTIVE_PREDICTION_KNOWN_KEYS
    if unknown:
        raise ValueError(
            "Unsupported selective_prediction keys: "
            + ", ".join(sorted(unknown))
        )

    mode = _selective_prediction_mode(raw)
    enabled = _strict_config_bool(raw, "enabled", False)
    if not enabled:
        stale = sorted(set(raw) - {"enabled"})
        if stale:
            raise ValueError(
                "Disabled selective_prediction accepts only 'enabled'; "
                f"remove stale settings: {stale}"
            )
        out["selective_prediction"] = {"enabled": False}
        return out
    minimum_raw = raw.get("min_calibration_malware", 1)
    try:
        minimum = int(minimum_raw)
        minimum_float = float(minimum_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "selective_prediction.min_calibration_malware must be a "
            "positive integer"
        ) from exc
    if (
        isinstance(minimum_raw, bool)
        or minimum < 1
        or not math.isfinite(minimum_float)
        or minimum_float != float(minimum)
    ):
        raise ValueError(
            "selective_prediction.min_calibration_malware must be a "
            "positive integer"
        )
    canonical = {
        "enabled": True,
        "mode": mode,
        "threshold_score": str(
            raw.get("threshold_score", "care_selected_path_correctness")
        ).strip().lower(),
        "risk_level": float(raw.get("risk_level", 0.05)),
        "risk_target": str(
            raw.get("risk_target", "malware_conditional_accepted_fn")
        ).strip().lower(),
        "min_calibration_malware": minimum,
        "require_feasible": _strict_config_bool(
            raw, "require_feasible", False
        ),
    }
    out["selective_prediction"] = canonical
    return out


def _canonicalize_fixed_classification_config(cfg: dict) -> dict:
    """Enforce the only current classification rule: binary argmax/0.5."""

    out = copy.deepcopy(cfg or {})
    raw = out.get("classification_threshold", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("classification_threshold must be a mapping")
    unknown = sorted(set(raw) - {"enabled"})
    if unknown:
        raise ValueError(
            "classification_threshold supports only {'enabled': false}; "
            f"unsupported keys: {unknown}"
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "classification_threshold.enabled must be boolean"
        )
    if enabled:
        raise ValueError(
            "The fixed binary decision protocol requires "
            "classification_threshold.enabled=false"
        )
    out["classification_threshold"] = {"enabled": False}
    return out


def _apply_resolved_config_defaults(cfg: dict) -> dict:
    if not _uses_tri_modal_defaults(cfg):
        return copy.deepcopy(cfg)
    model_cfg = (cfg or {}).get("model") or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model must be a mapping")
    resolved = deep_update(copy.deepcopy(TriModalConfigDefaults.CONFIG), cfg)
    method_cfg = resolved.get("method", {}) or {}
    protocol_id = (
        str(method_cfg.get("protocol_id", "")).strip()
        if isinstance(method_cfg, dict)
        else ""
    )
    if protocol_id == "care_droid_v1":
        resolved_model = resolved.get("model", {}) or {}
        if not isinstance(resolved_model, dict):
            raise ValueError("CARE model must be a mapping")
        if (
            str(resolved_model.get("fusion_mode", "")).strip().lower()
            != "care_droid"
        ):
            raise ValueError(
                "care_droid_v1 requires model.fusion_mode='care_droid'"
            )
        raw_loss = resolved.get("loss", {}) or {}
        if not isinstance(raw_loss, dict):
            raise ValueError("CARE loss must be a mapping")
        if (
            str(raw_loss.get("objective", "")).strip().lower()
            != "care_stage_a_clean"
        ):
            raise ValueError(
                "care_droid_v1 requires "
                "loss.objective='care_stage_a_clean'"
            )
        label_smoothing = float(raw_loss.get("label_smoothing", 0.0))
        if (
            not math.isfinite(label_smoothing)
            or not 0.0 <= label_smoothing < 1.0
        ):
            raise ValueError(
                "CARE loss.label_smoothing must be finite and lie in [0, 1)"
            )
        classification = resolved.get("classification_threshold", {}) or {}
        if not isinstance(classification, dict):
            raise ValueError(
                "CARE classification_threshold must be a mapping"
            )
        if bool(classification.get("enabled", False)):
            raise ValueError(
                "CARE-Droid fixes HardPredict at zero log-odds and rejects "
                "classification_threshold.enabled=true"
            )
        resolved.pop("encoder_stage", None)
        resolved["fusion"] = {"mode": "care_droid"}
        resolved["calibration"] = {"enabled": False}
        resolved["loss"] = {
            "objective": "care_stage_a_clean",
            "label_smoothing": label_smoothing,
        }
        resolved["classification_threshold"] = {"enabled": False}
        resolved.setdefault("train", {})["label_smoothing"] = label_smoothing
        return _canonicalize_selective_prediction_config(resolved)
    return _canonicalize_fixed_classification_config(
        _canonicalize_selective_prediction_config(resolved)
    )


_RAW_CONFIG_SECTION_KEYS = {
    "runner": {"module"},
    "method": {"name", "protocol_id"},
    "data": {
        "root",
        "train_pt_dir",
        "val_pt_dir",
        "test_pt_dir",
        "train_csv",
        "val_csv",
        "test_csv",
        "out_dir",
        "manifest_vocab_path",
        "require_manifest_vocab_provenance",
        "expected_pt_build_fingerprint",
        "pt_audit_certificate",
        "require_pt_audit_certificate",
        "max_failed_ratio",
        "max_api_events_per_sample",
        "strict_split_integrity",
        "strict_partition_isolation",
        "allow_pt_superset",
    },
    "train": {
        "seed",
        "epochs",
        "patience",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "eval_num_workers",
        "lr",
        "weight_decay",
        "exp_name",
        "device",
        "multiprocessing_sharing_strategy",
        "prefetch_factor",
        "deterministic",
        "strict_deterministic",
        "pin_memory",
        "allow_pyg_pin_memory",
        "persistent_workers",
        "use_amp",
        "eta_min",
        "grad_clip",
        "grad_accum_steps",
        "label_smoothing",
    },
    "encoder_stage": {"mode", "protocol_id", "strict_identity"},
    "model": {
        "num_classes",
        "fusion_mode",
        "max_nodes_gnn",
        "api_encoder",
        "graph_encoder",
        "manifest_encoder",
        "gate",
        "quality_fusion_temperature",
    },
    "fusion": {
        "mode",
        "combination",
        "opinion_source",
        "evidence_activation",
        "use_hard_alive_mask",
        "force_fp32_decision",
        "min_discount",
        "base_rate",
    },
    "loss": {
        "objective",
        "label_smoothing",
        "branch_aux_weight",
        "branch_aux_weights",
        "auxiliary_weight_mode",
        "evidential_loss_weight",
        "evidential",
        "tmc",
        "ecml",
    },
    "calibration": {
        "enabled",
        "validation_fraction",
        "split_seed",
        "stratified_group_split",
        "expert_val_fraction",
        "expert_split_seed",
        "role_assignment_path",
        "require_role_assignment",
    },
    "classification_threshold": {"enabled"},
    "selective_prediction": {
        "enabled",
        "mode",
        "threshold_score",
        "risk_target",
        "risk_level",
        "min_calibration_malware",
        "require_feasible",
    },
    "care": {
        "protocol_seed",
        "paths",
        "roles",
        "stage_a",
        "risk_training",
        "views",
        "routing",
        "decision",
    },
    "eval": {
        "run_test",
        "run_robust_test",
        "controlled_view_protocol_seed",
        "perturb_strengths",
        "perturb_tests",
        "eval_only",
        "output_name",
        "checkpoint_path",
        "refit_decision_calibration",
    },
}
_RAW_CONFIG_NESTED_KEYS = {
    ("model", "api_encoder"): {
        "type",
        "num_hash_buckets",
        "type_vocab_size",
        "emb_dim",
        "hidden_dim",
        "dropout",
        "layers",
        "heads",
        "max_seq_len",
    },
    ("model", "graph_encoder"): {
        "type",
        "emb_dim",
        "hidden",
        "heads",
        "layers",
        "use_behavior_hint",
        "drop_extracted_behavior_hints",
    },
    ("model", "manifest_encoder"): {
        "in_dim",
        "emb_dim",
        "hidden_dim",
        "dropout",
        "category_dim",
        "stats_dim",
        "permission_dim",
        "intent_dim",
        "feature_dim",
    },
    ("model", "gate"): {
        "hidden_dim",
        "detach",
    },
    ("loss", "branch_aux_weights"): {
        "api",
        "graph",
        "manifest",
    },
    ("loss", "evidential"): {
        "anneal_epochs",
        "branches",
        "class_weight",
    },
    ("loss", "tmc"): {
        "anneal_epochs",
        "mask_unavailable_views",
    },
    ("loss", "ecml"): {
        "anneal_epochs",
        "consistency_weight",
        "mask_unavailable_views",
    },
}
_RAW_CONFIG_TOP_LEVEL_KEYS = frozenset(
    {
        "defaults",
        "log_level",
        *_RAW_CONFIG_SECTION_KEYS,
    }
)


def _validate_raw_config_schema(cfg: dict) -> dict:
    """Reject every field outside the closed current experiment schema."""

    cfg = copy.deepcopy(cfg or {})
    unknown_top_level = sorted(set(cfg) - _RAW_CONFIG_TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ValueError(
            f"Unsupported top-level configuration keys: {unknown_top_level}"
        )
    for section, allowed_keys in _RAW_CONFIG_SECTION_KEYS.items():
        if section not in cfg:
            continue
        value = cfg[section]
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = sorted(set(value) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unsupported {section} configuration keys: {unknown}"
            )
    for (section, nested_key), allowed_keys in (
        _RAW_CONFIG_NESTED_KEYS.items()
    ):
        section_value = cfg.get(section)
        if not isinstance(section_value, dict) or nested_key not in section_value:
            continue
        nested_value = section_value[nested_key]
        field = f"{section}.{nested_key}"
        if not isinstance(nested_value, dict):
            raise ValueError(f"{field} must be a mapping")
        unknown = sorted(set(nested_value) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unsupported {field} configuration keys: {unknown}"
            )
    return cfg


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silently shadowed keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError(
                f"Unhashable YAML mapping key at {key_node.start_mark}"
            ) from exc
        if duplicate:
            raise ValueError(
                f"Duplicate YAML key {key!r} at {key_node.start_mark}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_UniqueKeySafeLoader) or {}


def _load_explicit_config_path(
    path: str | Path,
    seen: set[Path] | None = None,
) -> dict:
    path = Path(path)
    seen = set(seen or ())
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Recursive config defaults detected: {path}")
    seen.add(resolved)
    raw = _validate_raw_config_schema(load_yaml(path))
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ValueError(f"Config defaults must be a list: {path}")
    cfg: dict[str, Any] = {}
    for item in defaults:
        item_path = Path(item)
        if not item_path.is_absolute():
            item_path = path.parent / item_path
        cfg = deep_update(
            cfg, _load_explicit_config_path(item_path, seen)
        )
    return deep_update(cfg, raw)


def load_config_path(
    path: str | Path,
    seen: set[Path] | None = None,
) -> dict:
    return _apply_resolved_config_defaults(
        _load_explicit_config_path(path, seen)
    )


def load_config(paths: list[str]) -> dict:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_update(cfg, _load_explicit_config_path(path))
    return _apply_resolved_config_defaults(cfg)


def resolve(root: str | Path, path: str | Path) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    return str(Path(root) / path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor bytes, dtypes, and shapes without pickle semantics."""

    if not isinstance(state, dict) or not state:
        raise ValueError("State must be a non-empty tensor mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State entry {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, destination: str | Path) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical_mapping_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_split_identities(
    csv_path: str | Path,
    expected_split: str,
) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    packages: set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        id_col = next(
            (
                name
                for name in ("id", "ID", "Id", "sha256")
                if name in fields
            ),
            None,
        )
        pkg_col = next(
            (
                name
                for name in ("pkg_name", "package_name", "package")
                if name in fields
            ),
            None,
        )
        if id_col is None:
            raise ValueError(f"CSV {csv_path} must contain id or sha256")
        for row_idx, row in enumerate(reader, start=2):
            sid = str(row.get(id_col, "") or "").strip().lower()
            if not sid:
                raise ValueError(
                    f"CSV {csv_path} has empty {id_col} at row {row_idx}"
                )
            ids.add(sid)
            if "split" in fields:
                split_value = str(
                    row.get("split", "") or ""
                ).strip().lower()
                if split_value and split_value != expected_split:
                    raise ValueError(
                        f"CSV {csv_path} row {row_idx} declares "
                        f"split={split_value!r}, expected {expected_split!r}"
                    )
            if pkg_col is not None:
                package = str(row.get(pkg_col, "") or "").strip().lower()
                if package and package not in {"nan", "none", "null"}:
                    packages.add(package)
    return ids, packages


def _sample_ids_sha256(sample_ids: set[str] | list[str]) -> str:
    digest = hashlib.sha256()
    for sid in sorted(str(value).strip().lower() for value in sample_ids):
        encoded = sid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_vocab_sha256(vocab: dict[str, Any]) -> str:
    encoded = json.dumps(
        vocab,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_vocab_provenance(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    if not _strict_bool_value(
        data_cfg.get("require_manifest_vocab_provenance", False),
        field_name="data.require_manifest_vocab_provenance",
    ):
        return {"required": False, "verified": False}
    data_root = data_cfg.get("root", "")
    vocab_value = str(
        data_cfg.get("manifest_vocab_path") or ""
    ).strip()
    if not vocab_value:
        raise ValueError(
            "data.require_manifest_vocab_provenance=true requires "
            "data.manifest_vocab_path"
        )
    vocab_path = Path(resolve(data_root, vocab_value))
    train_csv_path = Path(
        resolve(data_root, data_cfg.get("train_csv", ""))
    )
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Manifest vocabulary provenance file not found: {vocab_path}"
        )
    if not train_csv_path.is_file():
        raise FileNotFoundError(
            "Train CSV required for Manifest provenance not found: "
            f"{train_csv_path}"
        )
    with vocab_path.open("r", encoding="utf-8-sig") as handle:
        vocab = yaml.safe_load(handle) or {}
    if not isinstance(vocab, dict):
        raise ValueError(
            f"Manifest vocabulary must be a mapping: {vocab_path}"
        )
    metadata = vocab.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Manifest vocabulary metadata must be a mapping")
    if (
        metadata.get("source_split") != "train"
        or metadata.get("leakage_guard") != "train_only"
    ):
        raise ValueError(
            "Manifest vocabulary must declare source_split=train and "
            "leakage_guard=train_only"
        )
    train_ids, _ = _read_split_identities(train_csv_path, "train")
    expected_csv_sha = _file_sha256(train_csv_path)
    expected_ids_sha = _sample_ids_sha256(train_ids)
    actual_csv_sha = str(metadata.get("train_csv_sha256") or "")
    actual_ids_sha = str(
        metadata.get("train_sample_ids_sha256") or ""
    )
    if (
        actual_csv_sha != expected_csv_sha
        or actual_ids_sha != expected_ids_sha
    ):
        raise ValueError(
            "Manifest vocabulary was not built from the current train split. "
            "Run scripts/migrate_manifest_vocab_pts.py before training. "
            f"csv_match={actual_csv_sha == expected_csv_sha} "
            f"ids_match={actual_ids_sha == expected_ids_sha}"
        )
    if int(metadata.get("num_records", -1)) != len(train_ids):
        raise ValueError(
            "Manifest vocabulary num_records differs from current train rows"
        )
    return {
        "required": True,
        "verified": True,
        "vocab_path": str(vocab_path.resolve()),
        "manifest_vocab_sha256": _manifest_vocab_sha256(vocab),
        "train_csv_path": str(train_csv_path.resolve()),
        "train_csv_sha256": expected_csv_sha,
        "train_sample_ids_sha256": expected_ids_sha,
        "num_train_samples": int(len(train_ids)),
    }


def validate_split_partitions(
    cfg: Mapping[str, Any],
    include_test: bool,
) -> None:
    data_cfg = cfg.get("data", {}) or {}
    if not _strict_bool_value(
        data_cfg.get("strict_partition_isolation", True),
        field_name="data.strict_partition_isolation",
    ):
        return
    data_root = data_cfg.get("root", "")
    split_names = ["train", "val"] + (["test"] if include_test else [])
    identities: dict[str, tuple[set[str], set[str]]] = {}
    for split in split_names:
        csv_path = resolve(data_root, data_cfg[f"{split}_csv"])
        identities[split] = _read_split_identities(csv_path, split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            id_overlap = sorted(
                identities[left][0] & identities[right][0]
            )
            pkg_overlap = sorted(
                identities[left][1] & identities[right][1]
            )
            if id_overlap or pkg_overlap:
                raise ValueError(
                    f"Split leakage between {left} and {right}: "
                    f"id_overlap={len(id_overlap)} "
                    f"examples={id_overlap[:10]}; "
                    f"package_overlap={len(pkg_overlap)} "
                    f"examples={pkg_overlap[:10]}"
                )


def _validated_max_failed_ratio(value: Any) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(
            "max_failed_ratio must be a finite number in [0, 1), "
            "not boolean"
        )
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "max_failed_ratio must be a finite number in [0, 1)"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError(
            "max_failed_ratio must be finite and within [0, 1), "
            f"got {value!r}"
        )
    if threshold != 0.0:
        raise ValueError(
            "The formal pipeline requires max_failed_ratio=0.0: dropping "
            "failed samples would bias reported metrics"
        )
    return threshold


def _resolve_graph_node_budget(model_cfg: Mapping[str, Any]) -> int:
    graph_cfg = model_cfg.get("graph_encoder", {}) or {}
    removed = sorted(
        set(graph_cfg) & {"account_for_encoder_budget", "max_nodes"}
    )
    if removed:
        raise ValueError(
            "Removed model.graph_encoder settings are unsupported: "
            f"{removed}. Declare model.max_nodes_gnn only."
        )
    if "max_nodes_gnn" not in model_cfg:
        raise ValueError(
            "model.max_nodes_gnn is required as the single graph-node budget"
        )
    budget = strict_finite_integer(
        model_cfg["max_nodes_gnn"],
        field_name="model.max_nodes_gnn",
    )
    if budget <= 0:
        raise ValueError("model.max_nodes_gnn must be a positive integer")
    return budget


def _dataset_common_kwargs(
    cfg: Mapping[str, Any],
    is_train: bool,
) -> dict[str, Any]:
    is_train = _strict_bool_value(is_train, field_name="is_train")
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
        "strict_split_integrity",
        "strict_partition_isolation",
        "allow_pt_superset",
        "manifest_vocab_path",
        "require_manifest_vocab_provenance",
        "expected_manifest_vocab_sha256",
        "expected_manifest_train_csv_sha256",
        "expected_manifest_train_sample_ids_sha256",
        "expected_pt_build_fingerprint",
        "pt_audit_certificate",
        "require_pt_audit_certificate",
    }
    unknown_data_keys = sorted(set(data_cfg) - allowed_data_keys)
    if unknown_data_keys:
        raise ValueError(f"Unsupported data settings: {unknown_data_keys}")
    model_cfg = cfg.get("model", {}) or {}
    api_cfg = model_cfg.get("api_encoder", {}) or {}
    api_event_budget = strict_finite_integer(
        data_cfg.get("max_api_events_per_sample"),
        field_name="data.max_api_events_per_sample",
    )
    api_encoder_capacity = strict_finite_integer(
        api_cfg.get("max_seq_len", 2048),
        field_name="model.api_encoder.max_seq_len",
    )
    if api_event_budget <= 0:
        raise ValueError(
            "data.max_api_events_per_sample must be positive"
        )
    if api_encoder_capacity <= 0:
        raise ValueError("model.api_encoder.max_seq_len must be positive")
    if api_event_budget > api_encoder_capacity:
        raise ValueError(
            "data.max_api_events_per_sample must be <= "
            "model.api_encoder.max_seq_len so Dataset is the only API "
            f"truncation point; got {api_event_budget} > "
            f"{api_encoder_capacity}"
        )
    _validated_max_failed_ratio(
        data_cfg.get("max_failed_ratio", 0.0)
    )
    num_classes = strict_finite_integer(
        model_cfg.get("num_classes", 2),
        field_name="model.num_classes",
    )
    if num_classes != 2:
        raise ValueError(
            "The current tri-modal pipeline is binary-only: "
            "model.num_classes must be 2"
        )
    manifest_cfg = model_cfg.get("manifest_encoder", {}) or {}
    graph_node_budget = _resolve_graph_node_budget(model_cfg)
    certificate_value = str(
        data_cfg.get("pt_audit_certificate") or ""
    ).strip()
    pt_audit_certificate = (
        resolve(data_cfg.get("root", ""), certificate_value)
        if certificate_value
        else None
    )
    return {
        "is_train": is_train,
        "manifest_dim": int(manifest_cfg.get("in_dim", 256)),
        "manifest_category_dim": int(
            manifest_cfg.get("category_dim", 12)
        ),
        "manifest_stats_dim": int(manifest_cfg.get("stats_dim", 11)),
        "manifest_permission_dim": int(
            manifest_cfg.get("permission_dim", 128)
        ),
        "manifest_intent_dim": int(
            manifest_cfg.get("intent_dim", 64)
        ),
        "manifest_feature_dim": int(
            manifest_cfg.get("feature_dim", 32)
        ),
        "max_api_events_per_sample": api_event_budget,
        "max_graph_nodes_per_sample": graph_node_budget,
        "drop_graph_behavior_hints": _strict_bool_value(
            (model_cfg.get("graph_encoder", {}) or {}).get(
                "drop_extracted_behavior_hints", False
            ),
            field_name=(
                "model.graph_encoder.drop_extracted_behavior_hints"
            ),
        ),
        "num_classes": num_classes,
        "strict_split_integrity": _strict_bool_value(
            data_cfg.get("strict_split_integrity", True),
            field_name="data.strict_split_integrity",
        ),
        "allow_pt_superset": _strict_bool_value(
            data_cfg.get("allow_pt_superset", False),
            field_name="data.allow_pt_superset",
        ),
        "expected_manifest_vocab_sha256": data_cfg.get(
            "expected_manifest_vocab_sha256"
        ),
        "expected_manifest_train_csv_sha256": data_cfg.get(
            "expected_manifest_train_csv_sha256"
        ),
        "expected_manifest_train_sample_ids_sha256": data_cfg.get(
            "expected_manifest_train_sample_ids_sha256"
        ),
        "expected_pt_build_fingerprint": data_cfg.get(
            "expected_pt_build_fingerprint"
        ),
        "pt_audit_certificate": pt_audit_certificate,
        "require_pt_audit_certificate": _strict_bool_value(
            data_cfg.get("require_pt_audit_certificate", False),
            field_name="data.require_pt_audit_certificate",
        ),
    }


def build_dataset_from_paths(
    cfg: Mapping[str, Any],
    pt_dir: str | Path,
    csv_path: str | Path,
    is_train: bool,
    dataset_overrides: Mapping[str, Any] | None = None,
) -> RobustTriModalDataset:
    kwargs = _dataset_common_kwargs(
        cfg,
        is_train=is_train,
    )
    kwargs.update(dict(dataset_overrides or {}))
    try:
        return RobustTriModalDataset(
            pt_dir=str(pt_dir),
            csv_path=str(csv_path),
            **kwargs,
        )
    except RuntimeError as exc:
        if "No matching .pt samples found" in str(exc):
            raise EmptyExtraEvalSetError(str(exc)) from exc
        raise


def build_dataset(
    cfg: Mapping[str, Any],
    split: str,
    is_train: bool,
) -> RobustTriModalDataset:
    data_cfg = cfg["data"]
    data_root = data_cfg.get("root", "")
    pt_dir = resolve(data_root, data_cfg[f"{split}_pt_dir"])
    csv_path = resolve(data_root, data_cfg[f"{split}_csv"])
    return build_dataset_from_paths(
        cfg,
        pt_dir,
        csv_path,
        is_train=is_train,
    )


def build_loader(
    cfg: Mapping[str, Any],
    dataset,
    is_train: bool,
    *,
    seed_namespace: str | None = None,
    persistent_workers_override: bool | None = None,
    batch_sampler_override=None,
    collate_fn_override=None,
) -> DataLoader:
    train_cfg = cfg["train"]
    worker_key = "num_workers" if is_train else "eval_num_workers"
    workers = int(
        train_cfg.get(worker_key, train_cfg.get("num_workers", 0))
    )
    pin_memory = _strict_bool_value(
        train_cfg.get("pin_memory", False),
        field_name="train.pin_memory",
    )
    allow_pyg_pin_memory = _strict_bool_value(
        train_cfg.get("allow_pyg_pin_memory", False),
        field_name="train.allow_pyg_pin_memory",
    )
    if pin_memory and not allow_pyg_pin_memory:
        logger.warning(
            "train.pin_memory=true is unsafe for PyG Data/Batch on some "
            "CUDA runtimes; forcing pin_memory=false"
        )
        pin_memory = False
    persistent_workers = _strict_bool_value(
        train_cfg.get("persistent_workers", False),
        field_name="train.persistent_workers",
    )
    if persistent_workers_override is not None:
        persistent_workers = _strict_bool_value(
            persistent_workers_override,
            field_name="persistent_workers_override",
        )
    base_loader_seed = strict_finite_integer(
        train_cfg.get("seed", 42),
        field_name="train.seed",
    )
    resolved_namespace = str(
        seed_namespace or ("train" if is_train else "evaluation")
    )
    generator = torch.Generator()
    generator.manual_seed(
        _namespaced_seed(base_loader_seed, resolved_namespace)
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and workers > 0,
        "collate_fn": collate_fn_override or robust_collate_fn,
        "generator": generator,
        "worker_init_fn": seed_data_loader_worker,
    }
    if batch_sampler_override is None:
        loader_kwargs.update(
            {
                "batch_size": int(
                    train_cfg.get(
                        "batch_size" if is_train else "eval_batch_size",
                        train_cfg.get("batch_size", 32),
                    )
                ),
                "shuffle": bool(is_train),
            }
        )
    else:
        if is_train:
            raise ValueError(
                "A fixed batch sampler is valid only for evaluation loaders"
            )
        loader_kwargs["batch_sampler"] = batch_sampler_override
    if workers > 0 and train_cfg.get("prefetch_factor") is not None:
        prefetch_factor = int(train_cfg["prefetch_factor"])
        if prefetch_factor <= 0:
            raise ValueError(
                "train.prefetch_factor must be positive when workers > 0"
            )
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def _require_stratified_group_validation_split(
    calibration_cfg: Mapping[str, Any],
) -> None:
    value = calibration_cfg.get("stratified_group_split", True)
    if not isinstance(value, bool) or not value:
        raise ValueError(
            "calibration.stratified_group_split must be true"
        )


def split_validation_dataset(
    cfg: Mapping[str, Any],
    dataset,
) -> tuple[Subset, Subset, dict[str, Any]]:
    """Create a deterministic group-disjoint year/label-stratified split."""

    calibration_cfg = cfg.get("calibration", {}) or {}
    _require_stratified_group_validation_split(calibration_cfg)
    fraction = float(
        calibration_cfg.get(
            "validation_fraction", VALIDATION_HOLDOUT_FRACTION
        )
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "calibration.validation_fraction must be within (0, 1)"
        )
    size = len(dataset)
    if size < 2:
        raise ValueError(
            "Dataset needs at least two samples for a role split"
        )
    seed = int(
        calibration_cfg.get(
            "split_seed", (cfg.get("train", {}) or {}).get("seed", 42)
        )
    )
    sids = list(getattr(dataset, "sample_sids", []))
    groups = list(getattr(dataset, "sample_groups", []))
    labels = list(getattr(dataset, "sample_labels", []))
    years = list(getattr(dataset, "sample_years", []))
    missing_metadata = [
        name
        for name, values in (
            ("sample_sids", sids),
            ("sample_groups", groups),
            ("sample_labels", labels),
            ("sample_years", years),
        )
        if len(values) != size
    ]
    if missing_metadata:
        raise ValueError(
            "Formal role splitting requires complete package-group and "
            f"year-label metadata; missing={missing_metadata}"
        )
    labels = [int(label) for label in labels]
    years = [int(year) for year in years]
    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(str(group), []).append(index)
    if len(group_to_indices) < 2:
        raise ValueError(
            "Dataset needs at least two package/sample groups"
        )

    label_values = sorted(set(labels))
    year_values = sorted(set(years))
    strata = list(zip(years, labels))
    stratum_values = sorted(set(strata))
    total_stratum_counts = {
        stratum: sum(int(value == stratum) for value in strata)
        for stratum in stratum_values
    }
    target_stratum_counts = {
        stratum: float(total_stratum_counts[stratum]) * fraction
        for stratum in stratum_values
    }
    target_holdout_size = min(
        size - 1, max(1, int(round(size * fraction)))
    )
    ranked_groups = sorted(
        group_to_indices,
        key=lambda group: hashlib.sha256(
            f"{seed}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    rank = {
        group: index for index, group in enumerate(ranked_groups)
    }
    group_stratum_counts = {
        group: {
            stratum: sum(
                int(strata[index] == stratum) for index in indices
            )
            for stratum in stratum_values
        }
        for group, indices in group_to_indices.items()
    }

    def split_error(
        candidate_size: int,
        candidate_counts: dict[tuple[int, int], int],
    ) -> float:
        size_error = (
            abs(candidate_size - target_holdout_size) / max(size, 1)
        )
        stratum_error = sum(
            abs(
                candidate_counts[stratum]
                - target_stratum_counts[stratum]
            )
            / max(total_stratum_counts[stratum], 1)
            for stratum in stratum_values
        )
        return size_error + stratum_error

    remaining = set(ranked_groups)
    holdout_groups: list[str] = []
    holdout_indices: list[int] = []
    holdout_stratum_counts = {
        stratum: 0 for stratum in stratum_values
    }
    while remaining:
        current_size = len(holdout_indices)
        current_error = split_error(
            current_size, holdout_stratum_counts
        )
        candidates = []
        for group in remaining:
            indices = group_to_indices[group]
            new_size = current_size + len(indices)
            if new_size >= size:
                continue
            new_counts = {
                stratum: (
                    holdout_stratum_counts[stratum]
                    + group_stratum_counts[group][stratum]
                )
                for stratum in stratum_values
            }
            candidates.append(
                (
                    split_error(new_size, new_counts),
                    rank[group],
                    group,
                    new_counts,
                )
            )
        if not candidates:
            break
        best_error, _, best_group, best_counts = min(candidates)
        if (
            current_size >= target_holdout_size
            and best_error >= current_error
        ):
            break
        holdout_groups.append(best_group)
        holdout_indices.extend(group_to_indices[best_group])
        holdout_stratum_counts = best_counts
        remaining.remove(best_group)

    holdout_indices = sorted(holdout_indices)
    if not holdout_indices or len(holdout_indices) >= size:
        raise ValueError(
            "Unable to build non-empty stratified role subsets"
        )
    holdout_set = set(holdout_indices)
    selection_indices = [
        index for index in range(size) if index not in holdout_set
    ]
    selection_groups = sorted(
        set(groups[index] for index in selection_indices)
    )

    def counts(
        indices: list[int], values: list[int], source: list[int]
    ) -> dict[int, int]:
        return {
            value: sum(
                int(source[index] == value) for index in indices
            )
            for value in values
        }

    def joint_counts(indices: list[int]) -> dict[str, int]:
        return {
            f"{year}:{label}": sum(
                int(
                    years[index] == year and labels[index] == label
                )
                for index in indices
            )
            for year, label in stratum_values
        }

    return (
        Subset(dataset, selection_indices),
        Subset(dataset, holdout_indices),
        {
            "split_seed": seed,
            "validation_fraction": fraction,
            "num_validation": size,
            "num_selection": len(selection_indices),
            "num_calibration": len(holdout_indices),
            "selection_fraction_of_validation": (
                len(selection_indices) / float(size)
            ),
            "calibration_fraction_of_validation": (
                len(holdout_indices) / float(size)
            ),
            "num_selection_groups": len(selection_groups),
            "num_calibration_groups": len(holdout_groups),
            "selection_label_counts": counts(
                selection_indices, label_values, labels
            ),
            "calibration_label_counts": counts(
                holdout_indices, label_values, labels
            ),
            "selection_year_counts": counts(
                selection_indices, year_values, years
            ),
            "calibration_year_counts": counts(
                holdout_indices, year_values, years
            ),
            "selection_year_label_counts": joint_counts(
                selection_indices
            ),
            "calibration_year_label_counts": joint_counts(
                holdout_indices
            ),
            "selection_indices": selection_indices,
            "calibration_indices": holdout_indices,
        },
    )


def _load_fixed_validation_roles(
    cfg: Mapping[str, Any],
    dataset,
) -> tuple[Subset, Subset, dict[str, Any]] | None:
    """Load the immutable model-selection/decision-calibration assignment."""

    calibration_cfg = cfg.get("calibration", {}) or {}
    raw_path = str(
        calibration_cfg.get("role_assignment_path") or ""
    ).strip()
    required = bool(
        calibration_cfg.get("require_role_assignment", False)
    )
    if not raw_path:
        if required:
            raise ValueError(
                "calibration.require_role_assignment=true requires "
                "calibration.role_assignment_path"
            )
        return None
    data_cfg = cfg.get("data", {}) or {}
    path = Path(resolve(data_cfg.get("root", ""), raw_path))
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Validation role assignment not found: {path}"
            )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read validation role assignment {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "Validation role assignment must be a JSON mapping"
        )
    if (
        int(payload.get("schema_version", -1))
        != VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Validation role assignment schema mismatch: "
            f"expected={VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION} "
            f"actual={payload.get('schema_version')!r}"
        )

    val_csv = Path(
        resolve(data_cfg.get("root", ""), data_cfg.get("val_csv", ""))
    )
    expected_csv_sha = str(
        payload.get("validation_csv_sha256") or ""
    ).lower()
    actual_csv_sha = _file_sha256(val_csv)
    if expected_csv_sha != actual_csv_sha:
        raise ValueError(
            "Validation role assignment was built for a different "
            f"validation CSV: expected={expected_csv_sha!r} "
            f"actual={actual_csv_sha!r}"
        )

    role_names = ("model_selection", "decision_calibration")
    roles = payload.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(role_names):
        raise ValueError(
            "Validation role assignment must contain exactly roles "
            f"{list(role_names)}"
        )
    dataset_sids = [
        str(value).strip().lower()
        for value in getattr(dataset, "sample_sids", [])
    ]
    dataset_groups = [
        str(value) for value in getattr(dataset, "sample_groups", [])
    ]
    dataset_labels = [
        int(value) for value in getattr(dataset, "sample_labels", [])
    ]
    dataset_years = [
        int(value) for value in getattr(dataset, "sample_years", [])
    ]
    size = len(dataset)
    if not (
        len(dataset_sids)
        == len(dataset_groups)
        == len(dataset_labels)
        == len(dataset_years)
        == size
    ):
        raise ValueError(
            "Fixed validation roles require complete "
            "sid/group/label/year metadata"
        )
    if len(set(dataset_sids)) != size:
        raise ValueError(
            "Validation dataset contains duplicate sample identities"
        )
    index_by_sid = {
        sid: index for index, sid in enumerate(dataset_sids)
    }
    role_ids: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role_name in role_names:
        values = roles[role_name]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Validation role {role_name!r} must be non-empty"
            )
        normalized = [
            str(value).strip().lower() for value in values
        ]
        if any(not value for value in normalized):
            raise ValueError(
                f"Validation role {role_name!r} contains an empty id"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"Validation role {role_name!r} contains duplicate ids"
            )
        overlap = seen.intersection(normalized)
        if overlap:
            raise ValueError(
                "Validation roles overlap; "
                f"examples={sorted(overlap)[:10]}"
            )
        unknown = sorted(set(normalized) - set(index_by_sid))
        if unknown:
            raise ValueError(
                f"Validation role {role_name!r} contains unknown ids: "
                f"{unknown[:10]}"
            )
        seen.update(normalized)
        role_ids[role_name] = normalized
    missing = sorted(set(dataset_sids) - seen)
    if missing:
        raise ValueError(
            "Validation role assignment does not cover the full validation "
            f"set; missing={len(missing)} examples={missing[:10]}"
        )

    group_roles: dict[str, set[str]] = {}
    for role_name, values in role_ids.items():
        for sid in values:
            group = dataset_groups[index_by_sid[sid]]
            group_roles.setdefault(group, set()).add(role_name)
    crossed_groups = sorted(
        group
        for group, assigned in group_roles.items()
        if len(assigned) > 1
    )
    if crossed_groups:
        raise ValueError(
            "Validation role assignment splits package groups; "
            f"examples={crossed_groups[:10]}"
        )

    role_indices = {
        name: sorted(index_by_sid[sid] for sid in role_ids[name])
        for name in role_names
    }
    label_values = sorted(set(dataset_labels))
    year_values = sorted(set(dataset_years))

    def counts(
        indices: list[int],
        values: list[int],
        source: list[int],
    ) -> dict[int, int]:
        return {
            value: sum(
                int(source[index] == value) for index in indices
            )
            for value in values
        }

    def joint_counts(indices: list[int]) -> dict[str, int]:
        return {
            f"{year}:{label}": sum(
                int(
                    dataset_years[index] == year
                    and dataset_labels[index] == label
                )
                for index in indices
            )
            for year in year_values
            for label in label_values
        }

    selection_indices = role_indices["model_selection"]
    decision_indices = role_indices["decision_calibration"]
    assignment_sha = _file_sha256(path)
    assignment_semantic_sha = _canonical_mapping_sha256(
        {
            "schema_version": (
                VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION
            ),
            "validation_csv_sha256": actual_csv_sha,
            "roles": {
                name: sorted(role_ids[name]) for name in role_names
            },
        }
    )
    summary = {
        "role_assignment_path": str(path.resolve()),
        "role_assignment_sha256": assignment_sha,
        "role_assignment_semantic_sha256": assignment_semantic_sha,
        "role_assignment_schema_version": (
            VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION
        ),
        "validation_csv_sha256": actual_csv_sha,
        "split_seed": int(
            payload.get(
                "split_seed",
                calibration_cfg.get("split_seed", 42),
            )
        ),
        "validation_fraction": len(decision_indices) / float(size),
        "num_validation": size,
        "num_selection": len(selection_indices),
        "num_calibration": len(decision_indices),
        "selection_fraction_of_validation": (
            len(selection_indices) / float(size)
        ),
        "calibration_fraction_of_validation": (
            len(decision_indices) / float(size)
        ),
        "decision_fraction_of_validation": (
            len(decision_indices) / float(size)
        ),
        "selection_label_counts": counts(
            selection_indices, label_values, dataset_labels
        ),
        "decision_label_counts": counts(
            decision_indices, label_values, dataset_labels
        ),
        "selection_year_counts": counts(
            selection_indices, year_values, dataset_years
        ),
        "decision_year_counts": counts(
            decision_indices, year_values, dataset_years
        ),
        "selection_year_label_counts": joint_counts(selection_indices),
        "decision_year_label_counts": joint_counts(decision_indices),
        "num_selection_groups": len(
            {dataset_groups[index] for index in selection_indices}
        ),
        "num_calibration_groups": len(
            {dataset_groups[index] for index in decision_indices}
        ),
        "selection_indices": selection_indices,
        "calibration_indices": decision_indices,
        "decision_calibration_indices": decision_indices,
    }
    selection = Subset(dataset, selection_indices)
    decision = Subset(dataset, decision_indices)
    return selection, decision, summary


def enforce_failed_ratio(
    metrics: Mapping[str, Any],
    cfg: Mapping[str, Any],
    split_name: str,
    max_failed_ratio: float | None = None,
) -> None:
    threshold = _validated_max_failed_ratio(
        (cfg.get("data", {}) or {}).get("max_failed_ratio", 0.0)
        if max_failed_ratio is None
        else max_failed_ratio
    )
    num_eval = strict_finite_integer(
        metrics.get("num_eval", 0),
        field_name=f"{split_name}.num_eval",
    )
    num_failed = strict_finite_integer(
        metrics.get("num_failed", 0),
        field_name=f"{split_name}.num_failed",
    )
    if num_eval < 0 or num_failed < 0:
        raise ValueError(
            f"{split_name}: num_eval and num_failed must be non-negative"
        )
    num_requested = num_eval + num_failed
    if num_requested <= 0:
        raise RuntimeError(
            f"{split_name}: no requested samples were seen"
        )
    if num_eval <= 0:
        raise RuntimeError(
            f"{split_name}: no samples were evaluated successfully"
        )
    failed_ratio = float(num_failed) / float(num_requested)
    if failed_ratio > threshold:
        raise RuntimeError(
            f"{split_name}: failed sample ratio {failed_ratio:.4f} "
            f"exceeds data.max_failed_ratio={threshold:.4f}"
        )


def _build_eval_perturbation_view(
    base_dataset: RobustTriModalDataset,
    *,
    perturb_type: str,
    perturb_strength: float,
    protocol_seed: int,
) -> tuple[RobustTriModalDataset, tuple[dict[str, Any], ...]]:
    """Clone a validated evaluation dataset without rescanning its PT pool."""

    if not isinstance(base_dataset, RobustTriModalDataset):
        raise TypeError(
            "Evaluation views require a RobustTriModalDataset base"
        )
    if bool(base_dataset.is_train):
        raise ValueError(
            "Evaluation views cannot be built from a train dataset"
        )
    if (
        perturb_type not in EVAL_PERTURB_TYPES
        or perturb_type == "clean"
    ):
        raise ValueError(
            f"Unsupported evaluation perturbation: {perturb_type!r}"
        )
    strength = float(perturb_strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            "Evaluation perturbation strength must be within [0, 1]"
        )
    view = copy.copy(base_dataset)
    view.is_train = False
    view.care_digest_view = True
    plan, records = fixed_test_view_plan(
        view.sample_sids,
        mechanism=perturb_type,
        strength=strength,
        protocol_seed=protocol_seed,
    )
    if len(plan) != len(view.sample_sids):
        raise RuntimeError(
            "Controlled evaluation plan lost sample identities"
        )
    view.eval_perturb_plan = plan
    return view, records


def _normalize_robust_test_protocol(
    eval_cfg: Mapping[str, Any],
) -> tuple[list[str], list[float]]:
    raw_tests = eval_cfg.get("perturb_tests", ["clean"])
    if not isinstance(raw_tests, (list, tuple)) or not raw_tests:
        raise ValueError(
            "eval.perturb_tests must be a non-empty sequence"
        )
    perturb_tests: list[str] = []
    for raw_test in raw_tests:
        if not isinstance(raw_test, str):
            raise ValueError(
                "eval.perturb_tests must contain non-empty strings"
            )
        perturb = raw_test.strip().lower()
        if not perturb:
            raise ValueError(
                "eval.perturb_tests contains an empty name"
            )
        perturb_tests.append(perturb)
    duplicates = sorted(
        {
            value
            for value in perturb_tests
            if perturb_tests.count(value) > 1
        }
    )
    if duplicates:
        raise ValueError(
            "eval.perturb_tests contains duplicates: "
            f"{duplicates}"
        )
    unsupported = sorted(
        value
        for value in set(perturb_tests)
        if value not in EVAL_PERTURB_TYPES
    )
    if unsupported:
        raise ValueError(
            "eval.perturb_tests contains unsupported mechanisms: "
            f"{unsupported}"
        )

    raw_strengths = eval_cfg.get("perturb_strengths")
    if not isinstance(raw_strengths, (list, tuple)) or not raw_strengths:
        raise ValueError(
            "eval.perturb_strengths must be a non-empty sequence"
        )
    strengths: list[float] = []
    for raw_strength in raw_strengths:
        if isinstance(raw_strength, bool):
            raise ValueError(
                "eval.perturb_strengths must contain numbers, not booleans"
            )
        try:
            strength = float(raw_strength)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "eval.perturb_strengths must contain finite numbers"
            ) from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(
                "eval.perturb_strengths values must be within [0, 1]"
            )
        strengths.append(strength)
    if len(set(strengths)) != len(strengths):
        raise ValueError(
            "eval.perturb_strengths contains duplicates"
        )
    rendered = [f"{value:.1f}" for value in strengths]
    if len(set(rendered)) != len(rendered):
        raise ValueError(
            "eval.perturb_strengths collide after result-key formatting"
        )
    return perturb_tests, strengths


def _robust_test_result_count(
    perturb_tests: list[str],
    perturb_strengths: list[float],
) -> int:
    return sum(
        1
        if perturb == "clean" or perturb.endswith("_missing")
        else len(perturb_strengths)
        for perturb in perturb_tests
    )


def iter_robust_test_loaders(
    cfg: Mapping[str, Any],
    base_dataset: RobustTriModalDataset,
) -> Iterator[dict[str, Any]]:
    """Yield independent, lazy controlled-degradation test loaders."""

    if not isinstance(base_dataset, RobustTriModalDataset):
        raise TypeError(
            "Robust-test views require a RobustTriModalDataset"
        )
    if bool(base_dataset.is_train):
        raise ValueError(
            "Robust-test views cannot be built from a train dataset"
        )
    perturb_tests, strengths = _normalize_robust_test_protocol(
        cfg.get("eval", {}) or {}
    )
    eval_cfg = cfg.get("eval", {}) or {}
    protocol_seed = strict_finite_integer(
        eval_cfg.get("controlled_view_protocol_seed"),
        field_name="eval.controlled_view_protocol_seed",
    )
    if protocol_seed != CONTROLLED_TEST_VIEW_PROTOCOL_SEED:
        raise ValueError(
            "eval.controlled_view_protocol_seed must equal the frozen "
            f"value {CONTROLLED_TEST_VIEW_PROTOCOL_SEED}"
        )
    for perturb in perturb_tests:
        if perturb == "clean":
            _, records = fixed_test_view_plan(
                base_dataset.sample_sids,
                mechanism="clean",
                strength=0.0,
                protocol_seed=protocol_seed,
            )
            yield {
                "result_key": "clean",
                "perturb_type": "clean",
                "strength": 0.0,
                "loader": None,
                "controlled_view_audit": {
                    "seed_formula": CONTROLLED_VIEW_SEED_FORMULA,
                    "protocol_seed": protocol_seed,
                    "mechanism": "clean",
                    "mechanism_version": (
                        CONTROLLED_VIEW_MECHANISM_VERSION
                    ),
                    "strength": 0.0,
                    "num_samples": len(records),
                    "seed_manifest_sha256": (
                        seed_manifest_sha256(records)
                    ),
                },
            }
            continue
        cell_strengths = (
            [1.0] if perturb.endswith("_missing") else strengths
        )
        for strength in cell_strengths:
            result_key = (
                perturb
                if len(cell_strengths) == 1
                else f"{perturb}@{strength:.1f}"
            )
            view, records = _build_eval_perturbation_view(
                base_dataset,
                perturb_type=perturb,
                perturb_strength=strength,
                protocol_seed=protocol_seed,
            )
            yield {
                "result_key": result_key,
                "perturb_type": perturb,
                "strength": float(strength),
                "loader": build_loader(
                    cfg,
                    view,
                    is_train=False,
                    seed_namespace=(
                        f"care/test/{perturb}/{strength:.6f}"
                    ),
                    persistent_workers_override=False,
                ),
                "controlled_view_audit": {
                    "seed_formula": CONTROLLED_VIEW_SEED_FORMULA,
                    "protocol_seed": protocol_seed,
                    "mechanism": perturb,
                    "mechanism_version": (
                        CONTROLLED_VIEW_MECHANISM_VERSION
                    ),
                    "strength": float(strength),
                    "num_samples": len(records),
                    "seed_manifest_sha256": (
                        seed_manifest_sha256(records)
                    ),
                },
            }


__all__ = [
    "CONTROLLED_TEST_VIEW_PROTOCOL_SEED",
    "CONTROLLED_VIEW_MECHANISM_VERSION",
    "CONTROLLED_VIEW_SEED_FORMULA",
    "STAGE_A_EXPERT_TRAIN_LOADER_NAMESPACE",
    "STAGE_A_EXPERT_VAL_LOADER_NAMESPACE",
    "EmptyExtraEvalSetError",
    "VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION",
    "_atomic_torch_save",
    "_build_eval_perturbation_view",
    "_dataset_common_kwargs",
    "_file_sha256",
    "_load_fixed_validation_roles",
    "_normalize_robust_test_protocol",
    "_robust_test_result_count",
    "_sample_ids_sha256",
    "_state_dict_sha256",
    "build_dataset",
    "build_dataset_from_paths",
    "build_loader",
    "configure_determinism",
    "configure_multiprocessing_sharing",
    "deep_update",
    "enforce_failed_ratio",
    "iter_robust_test_loaders",
    "load_config",
    "load_config_path",
    "load_yaml",
    "resolve",
    "seed_data_loader_worker",
    "select_device",
    "set_seed",
    "split_validation_dataset",
    "validate_manifest_vocab_provenance",
    "validate_split_partitions",
]
