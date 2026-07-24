from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from collections.abc import Iterator
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from fusion.losses import (
    REMOVED_LOSS_CONFIG_KEYS,
    compute_robust_loss,
    reliability_alive_mask,
    reliability_correctness_target,
    reliability_per_sample_loss,
    resolve_auxiliary_weight_mode,
    routing_mixture_log_prob,
    routing_risk_per_sample_loss,
    routing_risk_target,
)
from fusion.dataset import (
    RobustTriModalDataset,
    prepare_robust_batch,
    robust_collate_fn,
)
from fusion.temperature import (
    FINAL_TEMPERATURE_MAX,
    FINAL_TEMPERATURE_MIN,
    bounded_final_temperature,
    raw_final_temperature_coordinate,
)
from fusion.model import TriModalRobustModel
from fusion.perturbations import EVAL_PERTURB_TYPES
from fusion.reliability_calibration import (
    BRANCH_NAMES,
    MONOTONIC_CORRECTNESS_METHOD,
    TEMPERATURE_SCALING_CONFIDENCE_METHOD,
    normalize_reliability_calibration_method,
)
from fusion.utils import (
    build_grad_scaler,
    get_amp_context,
    strict_binary_integer,
    strict_finite_integer,
)
from fusion.constants import (
    CONFORMAL_WITHIN_HOLDOUT_FRACTION,
    VALIDATION_HOLDOUT_FRACTION,
    TriModalConfigDefaults,
)


logger = logging.getLogger("tri_modal_robust")

BRANCH_EVAL_LOGIT_KEYS = {
    "api": "api_logits_aux",
    "graph": "graph_logits_aux",
    "manifest": "manifest_logits_aux",
}


class EmptyExtraEvalSetError(RuntimeError):
    """Raised when an optional external eval set has no usable samples."""


CHECKPOINT_STAGE_ENCODER_SELECTED = "encoder_selected"
CHECKPOINT_STAGE_PIPELINE_FITTED = "pipeline_fitted"
ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION = 2
PIPELINE_ARTIFACT_SCHEMA_VERSION = 4

_PIPELINE_VALIDATION_IDENTITY_FIELDS = (
    "role_assignment_schema_version",
    "role_assignment_semantic_sha256",
    "validation_csv_sha256",
)
_PIPELINE_MANIFEST_IDENTITY_FIELDS = (
    "required",
    "verified",
    "manifest_vocab_sha256",
    "train_csv_sha256",
    "train_sample_ids_sha256",
    "num_train_samples",
)
# Bump whenever the inline optimizer/scheduler/save lifecycle inside run()
# changes in a way not represented by the source-hashed Stage-1 functions.
ENCODER_STAGE_TRAINING_PROTOCOL_REVISION = 2
VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION = 1
POSTHOC_OOF_ROWS_SCHEMA_VERSION = 5
METRIC_SUMMARY_SCHEMA_VERSION = 10
ACCEPTANCE_THRESHOLD_COMPARISON = "selective_eligible and score > threshold"
CLASSIFICATION_THRESHOLD_SELECTION_RULE = "macro_f1_unconstrained_v1"
CHECKPOINT_STAGES = frozenset(
    {
        CHECKPOINT_STAGE_ENCODER_SELECTED,
        CHECKPOINT_STAGE_PIPELINE_FITTED,
    }
)


GATE_DIAGNOSTIC_KEYS = (
    # I1 diagnostics
    "evidential_certainty_api",
    "evidential_certainty_graph",
    "evidential_certainty_manifest",
    "prediction_margin_api",
    "prediction_margin_graph",
    "prediction_margin_manifest",
    "predicted_malware_indicator_api",
    "predicted_malware_indicator_graph",
    "predicted_malware_indicator_manifest",
    "api_alive",
    "graph_alive",
    "manifest_alive",
    "fusion_weight_api",
    "fusion_weight_graph",
    "fusion_weight_manifest",
    "qmf_energy_api",
    "qmf_energy_graph",
    "qmf_energy_manifest",
    "uncertainty_proxy_api",
    "uncertainty_proxy_graph",
    "uncertainty_proxy_manifest",
    "predicted_reliability_api",
    "predicted_reliability_graph",
    "predicted_reliability_manifest",
    "total_reliability",
    "raw_conflict",
    "predictive_conflict",
    "predictive_conflict_max",
    "routing_active",
    "routing_risk_probability",
    "routing_mixture_prob_malware",
    "routing_mixture_pred",
    "routing_risk_mode_learned",
    "routing_risk_mode_reliability_prior",
    "routing_risk_mode_disabled",
    "routing_learned_components_active",
    "routing_has_available",
    "routing_risk_reliability_deficit",
    "routing_risk_decision_boundary_proximity",
    "routing_risk_predicted_malware",
    "routing_risk_decision_log_odds_threshold",
    "routing_risk_decision_threshold_active",
    "routing_risk_target_threshold_malware_false_negative",
    "routing_risk_global_cross_modal_conflict",
    "routing_route_prior_beta",
    "routing_prior_only_odds_beta",
    "routing_prior_only_odds_beta_active",
    "routing_conflict_penalty_mean",
    "routing_cross_modal_conflict_api",
    "routing_cross_modal_conflict_graph",
    "routing_cross_modal_conflict_manifest",
    "routing_conflict_penalty_api",
    "routing_conflict_penalty_graph",
    "routing_conflict_penalty_manifest",
    "routing_route_conflict_feature_active",
    "routing_route_conflict_feature_configured",
    "routing_risk_conflict_feature_active",
    "routing_risk_conflict_feature_configured",
    "routing_common_scale_reliability_active",
    "routing_prefit_uniform_prior_active",
    "routing_mode_learned",
    "routing_mode_prior_only",
    "routing_posthoc_refine",
    "final_temperature",
    "acceptance_score",
    "acceptance_score_mixture_certainty",
    "mixture_uncertainty_burden",
    "calibration_active",
)

def _resolve_restored_stage_convergence(
    *,
    provisional_stop_reason: str,
    final_grad_inf_norm: float,
    gradient_tolerance: float,
) -> tuple[bool, str]:
    """Determine convergence at the exact restored deployment state.

    Objective plateau is a stopping heuristic, not sufficient evidence of
    stationarity. The final convergence flag is decided from the gradient at
    the restored minimum-objective parameters.
    """

    if (
        not math.isfinite(final_grad_inf_norm)
        or not math.isfinite(gradient_tolerance)
        or gradient_tolerance <= 0.0
    ):
        return False, "restored_best_invalid_gradient"

    if final_grad_inf_norm <= gradient_tolerance:
        return True, "restored_best_gradient_tolerance"

    if provisional_stop_reason == "objective_plateau":
        return False, "objective_plateau_nonstationary"

    if provisional_stop_reason == "gradient_tolerance":
        return False, "provisional_gradient_not_valid_at_restored_best"

    return False, provisional_stop_reason


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _namespaced_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable positive torch seed without consuming global RNG state."""

    payload = f"{int(base_seed)}|{str(namespace)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def seed_data_loader_worker(_worker_id: int) -> None:
    """Seed every worker-local RNG from PyTorch's explicit loader seed.

    This function must remain at module scope so spawn-based DataLoaders can
    pickle it.  The worker id is already folded into ``torch.initial_seed()``.
    """

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


def _uses_tri_modal_defaults(cfg: dict) -> bool:
    """Resolve the config family from content, never from its filesystem path.

    Runnable tri-modal experiment roots declare a method identity (directly or
    through ``defaults``).  Plain overlays deliberately do not.  This keeps an
    overlay's meaning identical after renaming or moving it and, critically,
    prevents a non-underscore overlay from being expanded into a second full
    default config that silently resets earlier explicit choices.
    """
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
        "target_coverage",
        "alpha",
        "class_conditional",
        "use_raw_conflict",
        "risk_level",
        "risk_target",
        "min_calibration_malware",
        "require_feasible",
    }
)
_SELECTIVE_PREDICTION_MODES = frozenset(
    {"threshold", "conformal", "risk_control"}
)


def _selective_prediction_mode(config: dict | None = None) -> str:
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("selective_prediction must be a mapping")
    mode = str(config.get("mode", "threshold")).strip().lower()
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
    """Keep only the settings consumed by the selected I3 protocol.

    Family defaults necessarily cover all three decision modes.  Without this
    final canonicalization, a threshold overlay also inherits conformal and CRC
    keys (and vice versa), so changing an inactive default changes the resolved
    method fingerprint even though no prediction changes.  Unknown keys and
    modes fail here instead of silently behaving like fixed-coverage threshold
    selection.
    """

    out = copy.deepcopy(cfg or {})
    raw_value = out.get("selective_prediction", {})
    raw = {} if raw_value is None else raw_value
    if not isinstance(raw, dict):
        raise ValueError("selective_prediction must be a mapping")
    unknown = set(raw) - _SELECTIVE_PREDICTION_KNOWN_KEYS
    if unknown:
        raise ValueError(
            "Unsupported selective_prediction keys: " + ", ".join(sorted(unknown))
        )

    mode = _selective_prediction_mode(raw)
    enabled = _strict_config_bool(raw, "enabled", False)
    if not enabled:
        out["selective_prediction"] = {"enabled": False}
        return out

    if mode == "threshold":
        canonical = {
            "enabled": True,
            "mode": mode,
            "threshold_score": str(raw.get("threshold_score", "msp")).strip().lower(),
            "target_coverage": float(raw.get("target_coverage", 0.90)),
        }
    elif mode == "conformal":
        target_coverage = float(raw.get("target_coverage", 0.90))
        alpha = float(raw.get("alpha", 1.0 - target_coverage))
        if "alpha" in raw and "target_coverage" in raw and not math.isclose(
            alpha,
            1.0 - target_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "selective_prediction.alpha conflicts with target_coverage; "
                "require alpha == 1 - target_coverage"
            )
        canonical = {
            "enabled": True,
            "mode": mode,
            # Store both consistent forms because alpha is fitted while target
            # coverage is part of the public experiment report.
            "target_coverage": 1.0 - alpha,
            "alpha": alpha,
            "class_conditional": _strict_config_bool(
                raw, "class_conditional", True
            ),
            "use_raw_conflict": _strict_config_bool(
                raw, "use_raw_conflict", False
            ),
        }
    else:
        minimum_raw = raw.get("min_calibration_malware", 1)
        try:
            minimum = int(minimum_raw)
            minimum_float = float(minimum_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "selective_prediction.min_calibration_malware must be a positive integer"
            ) from exc
        if (
            isinstance(minimum_raw, bool)
            or minimum < 1
            or not math.isfinite(minimum_float)
            or minimum_float != float(minimum)
        ):
            raise ValueError(
                "selective_prediction.min_calibration_malware must be a positive integer"
            )
        canonical = {
            "enabled": True,
            "mode": mode,
            "threshold_score": str(
                raw.get("threshold_score", "model_acceptance")
            ).strip().lower(),
            "risk_level": float(raw.get("risk_level", 0.05)),
            "risk_target": str(
                raw.get("risk_target", "accepted_fn_risk_among_malware")
            ).strip().lower(),
            "min_calibration_malware": minimum,
            "require_feasible": _strict_config_bool(
                raw, "require_feasible", False
            ),
        }
    out["selective_prediction"] = canonical
    return out


def _canonicalize_classification_threshold_config(cfg: dict) -> dict:
    """Freeze the shared classification operating-point protocol.

    Malware-FN control belongs exclusively to I3's disjoint decision layer.
    A validation-only recall constraint would create a second risk controller,
    obscure the ablation boundary, and was not stable in the observed runs.
    Therefore every current run uses unconstrained macro-F1 with a deterministic
    neutral-boundary tie break. The removed ``min_malware_recall`` key is
    rejected even when set to null; old threshold protocols are not accepted.
    """

    out = copy.deepcopy(cfg or {})
    raw = out.get("classification_threshold", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("classification_threshold must be a mapping")
    raw = copy.deepcopy(raw)
    if "min_malware_recall" in raw:
        raise ValueError(
            "classification_threshold.min_malware_recall was removed: "
            "classification selects unconstrained macro-F1 and I3 alone "
            "controls malware false-negative risk"
        )
    selection_rule = str(
        raw.get(
            "selection_rule", CLASSIFICATION_THRESHOLD_SELECTION_RULE
        )
    ).strip().lower()
    if selection_rule != CLASSIFICATION_THRESHOLD_SELECTION_RULE:
        raise ValueError(
            "classification_threshold.selection_rule must be "
            f"{CLASSIFICATION_THRESHOLD_SELECTION_RULE!r}"
        )
    objective = str(raw.get("objective", "macro_f1")).strip().lower()
    if objective != "macro_f1":
        raise ValueError(
            "classification_threshold.objective currently supports only "
            "'macro_f1'"
        )
    raw["objective"] = objective
    raw["selection_rule"] = selection_rule
    raw["constraint"] = "none"
    out["classification_threshold"] = raw
    return out


def _apply_resolved_config_defaults(cfg: dict) -> dict:
    if not _uses_tri_modal_defaults(cfg):
        # A standalone overlay must remain explicit-only. It is canonicalized
        # after being merged with a method root by load_config().
        return copy.deepcopy(cfg)
    resolved = deep_update(TriModalConfigDefaults.CONFIG, cfg)
    return _canonicalize_classification_threshold_config(
        _canonicalize_selective_prediction_config(resolved)
    )


def _reject_removed_config_keys(cfg: dict) -> dict:
    """Reject removed keys in each explicit YAML layer before defaults merge."""

    cfg = copy.deepcopy(cfg or {})
    if "robust" in cfg:
        raise ValueError(
            "The Stage-1 augmentation section 'robust' was removed. Stage 1 is "
            "always clean-only; controlled degradations belong only to frozen-"
            "encoder post-hoc fitting and evaluation."
        )
    train_cfg = cfg.get("train")
    if isinstance(train_cfg, dict):
        removed_train = sorted(
            set(train_cfg) & {"checkpoint_metric", "tuning_mode"}
        )
        if removed_train:
            raise ValueError(
                "Removed Stage-1 tuning settings are unsupported: "
                f"{removed_train}. Stage 1 is selected only by clean validation "
                "macro-F1."
            )
    data_cfg = cfg.get("data")
    if isinstance(data_cfg, dict) and "graph_semantic_source" in data_cfg:
        raise ValueError(
            "Removed key data.graph_semantic_source is unsupported; runtime "
            "fusion no longer materializes API-Graph semantic alignment."
        )
    model_cfg = cfg.get("model")
    graph_cfg = (
        model_cfg.get("graph_encoder")
        if isinstance(model_cfg, dict)
        else None
    )
    if isinstance(graph_cfg, dict):
        removed_graph = sorted(
            set(graph_cfg) & {"account_for_encoder_budget", "max_nodes"}
        )
        if removed_graph:
            raise ValueError(
                "Removed model.graph_encoder settings are unsupported: "
                f"{removed_graph}. Declare the single positive graph budget "
                "only as model.max_nodes_gnn."
            )
    eval_cfg = cfg.get("eval")
    if isinstance(eval_cfg, dict) and "robust_val" in eval_cfg:
        raise ValueError(
            "eval.robust_val was removed. Stage 1 is selected only by clean "
            "validation macro-F1; controlled degradations are reserved for "
            "frozen-encoder post-hoc fitting and final evaluation."
        )
    fusion_cfg = cfg.get("fusion")
    if isinstance(fusion_cfg, dict) and "use_reliability_discount" in fusion_cfg:
        raise ValueError(
            "Removed key fusion.use_reliability_discount is no longer "
            "accepted; use fusion.use_i1_reliability"
        )
    loss_cfg = cfg.get("loss")
    if isinstance(loss_cfg, dict):
        removed = sorted(set(loss_cfg) & REMOVED_LOSS_CONFIG_KEYS)
        if removed:
            raise ValueError(
                "Removed loss configuration keys are unsupported: "
                f"{removed}. Use loss.auxiliary_weight_mode for branch weighting."
            )
    calibration_cfg = cfg.get("calibration")
    if isinstance(calibration_cfg, dict) and "epochs" in calibration_cfg:
        raise ValueError(
            "calibration.epochs was removed. Configure "
            "calibration.stage_optimization.<stage>.max_steps explicitly."
        )

    return cfg


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    raw = _reject_removed_config_keys(load_yaml(path))
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    cfg: dict[str, Any] = {}
    for item in defaults:
        item_path = Path(item)
        if not item_path.is_absolute():
            item_path = path.parent / item_path
        cfg = deep_update(cfg, _load_explicit_config_path(item_path, seen))
    return deep_update(cfg, raw)


def load_config_path(path: str | Path, seen: set[Path] | None = None) -> dict:
    """Resolve inheritance and apply family defaults exactly once at the root."""
    return _apply_resolved_config_defaults(
        _load_explicit_config_path(path, seen)
    )


def load_config(paths: list[str]) -> dict:
    cfg: dict[str, Any] = {}
    for path in paths:
        # Each command-line layer contributes only keys it explicitly declares
        # (plus its YAML ``defaults`` chain).  Family defaults are applied once
        # after every overlay has been merged.
        cfg = deep_update(cfg, _load_explicit_config_path(path))
    return _apply_resolved_config_defaults(cfg)


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


def _sample_ids_sha256(sample_ids: set[str] | list[str]) -> str:
    digest = hashlib.sha256()
    for sid in sorted(str(value).strip().lower() for value in sample_ids):
        encoded = sid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_vocab_sha256(vocab: dict[str, Any]) -> str:
    """Return the canonical digest written into every migrated Manifest PT."""
    encoded = json.dumps(
        vocab,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_vocab_provenance(cfg: dict) -> dict[str, Any]:
    """Bind the formal Manifest vocabulary to the current train split."""
    data_cfg = cfg.get("data", {}) or {}
    if not bool(data_cfg.get("require_manifest_vocab_provenance", False)):
        return {"required": False, "verified": False}
    data_root = data_cfg.get("root", "")
    vocab_value = str(data_cfg.get("manifest_vocab_path") or "").strip()
    if not vocab_value:
        raise ValueError(
            "data.require_manifest_vocab_provenance=true requires "
            "data.manifest_vocab_path"
        )
    vocab_path = Path(resolve(data_root, vocab_value))
    train_csv_path = Path(resolve(data_root, data_cfg.get("train_csv", "")))
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Manifest vocabulary provenance file not found: {vocab_path}"
        )
    if not train_csv_path.is_file():
        raise FileNotFoundError(
            f"Train CSV required for Manifest provenance not found: "
            f"{train_csv_path}"
        )
    with vocab_path.open("r", encoding="utf-8-sig") as handle:
        vocab = yaml.safe_load(handle) or {}
    if not isinstance(vocab, dict):
        raise ValueError(f"Manifest vocabulary must be a mapping: {vocab_path}")
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
    train_ids, _packages = _read_split_identities(train_csv_path, "train")
    expected_csv_sha = _file_sha256(train_csv_path)
    expected_ids_sha = _sample_ids_sha256(train_ids)
    actual_csv_sha = str(metadata.get("train_csv_sha256") or "")
    actual_ids_sha = str(metadata.get("train_sample_ids_sha256") or "")
    if actual_csv_sha != expected_csv_sha or actual_ids_sha != expected_ids_sha:
        raise ValueError(
            "Manifest vocabulary was not built from the current train split. "
            "Run scripts/migrate_manifest_vocab_pts.py (dry-run, then "
            "--apply) before training. "
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


_ENCODER_STAGE_IMPLEMENTATION_FILES = (
    "constants.py",
    "dataset.py",
    "evidence.py",
    "gates.py",
    "graph_encoders.py",
    "manifest_features.py",
    "perturbations.py",
    "pt_schema.py",
    "quality.py",
    "semantic_categories.py",
    "evidential.py",
    "discount_fusion.py",
    "opinion_router.py",
    "reliability_calibration.py",
    "model.py",
    "losses.py",
    "utils.py",
)


def _source_files_sha256(names: tuple[str, ...]) -> str:
    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in names:
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


def _encoder_stage_implementation_sha256() -> str:
    """Fingerprint code that constructs/trains Stage-1 experts.

    Configuration-only I1/I2/I3 experiments can reuse an artifact. Source
    changes in the shared forward path remain a hard incompatibility—even when
    they are intended to be post-hoc-only—because the neutral prefit path must
    be re-audited rather than assumed unchanged.
    """

    digest = hashlib.sha256()
    digest.update(
        _source_files_sha256(_ENCODER_STAGE_IMPLEMENTATION_FILES).encode(
            "ascii"
        )
    )
    # Stage-1 orchestration lives in this otherwise post-hoc-heavy module. Hash
    # only its operative functions, plus an explicit revision for the small
    # optimizer/scheduler block that remains inline in run().
    stage1_functions = (
        set_seed,
        configure_determinism,
        _namespaced_seed,
        seed_data_loader_worker,
        _dataset_common_kwargs,
        build_dataset_from_paths,
        build_dataset,
        build_loader,
        _build_eval_perturbation_view,
        build_model,
        train_one_epoch,
        evaluate_checkpoint_selection,
    )
    for function in stage1_functions:
        digest.update(function.__name__.encode("utf-8"))
        digest.update(b"\0")
        source = (
            inspect.getsource(function)
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    digest.update(
        str(ENCODER_STAGE_TRAINING_PROTOCOL_REVISION).encode("ascii")
    )
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash named tensor bytes, dtypes and shapes without pickle semantics."""

    if not isinstance(state, dict) or not state:
        raise ValueError("Encoder-stage state must be a non-empty tensor mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Encoder-stage state {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        # Flatten first: PyTorch cannot reinterpret a zero-dimensional scalar
        # as uint8 directly, while Stage-1 state may legitimately contain
        # scalar counters/buffers.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _encoder_stage_semantic_signature(
    cfg: dict,
    validation_split: dict[str, Any],
    manifest_vocab_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return only settings that can change the neutral Stage-1 artifact."""

    train_cfg = cfg.get("train", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    fusion_cfg = cfg.get("fusion", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    encoder_cfg = cfg.get("encoder_stage", {}) or {}
    train_fields = (
        "stage1_seed",
        "loader_seed",
        "epochs",
        "patience",
        "batch_size",
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "pin_memory",
        "allow_pyg_pin_memory",
        "lr",
        "weight_decay",
        "eta_min",
        "grad_clip",
        "grad_accum_steps",
        "label_smoothing",
        "use_amp",
        "deterministic",
        "strict_deterministic",
        "min_delta",
    )
    data_fields = (
        "max_api_events_per_sample",
        "label_map",
        "expected_manifest_vocab_sha256",
        "expected_manifest_train_csv_sha256",
        "expected_manifest_train_sample_ids_sha256",
        "expected_pt_build_fingerprint",
    )
    role_identity = {
        key: copy.deepcopy(validation_split.get(key))
        for key in (
            "role_assignment_semantic_sha256",
            "validation_csv_sha256",
            "num_selection",
        )
    }
    return {
        "schema_version": ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION,
        "training_protocol_revision": ENCODER_STAGE_TRAINING_PROTOCOL_REVISION,
        "protocol_id": str(
            encoder_cfg.get(
                "protocol_id", "neutral_alive_uniform_clean_stage1_v2"
            )
        ),
        "model": copy.deepcopy(cfg.get("model", {}) or {}),
        "neutral_fusion": {
            "mode": str(fusion_cfg.get("mode", "")),
            "combination": str(fusion_cfg.get("combination", "")),
            "routing_enabled": bool(routing_cfg.get("enabled", False)),
            "prefit_prior": "alive_masked_uniform",
            # These fields alter branch opinions or the neutral prefit result
            # even though I1/I2 fitted parameters are inactive.
            "evidence_activation": str(
                fusion_cfg.get("evidence_activation", "softplus")
            ),
            "opinion_source": str(
                fusion_cfg.get("opinion_source", "evidential")
            ),
            "softmax_opinion": copy.deepcopy(
                fusion_cfg.get("softmax_opinion", {}) or {}
            ),
            "min_discount": float(
                fusion_cfg.get("min_discount", 1.0e-8)
            ),
            "base_rate": float(fusion_cfg.get("base_rate", 0.5)),
            "use_hard_alive_mask": bool(
                fusion_cfg.get("use_hard_alive_mask", True)
            ),
            "force_fp32_decision": bool(
                fusion_cfg.get("force_fp32_decision", True)
            ),
        },
        "loss": copy.deepcopy(cfg.get("loss", {}) or {}),
        "train": {
            **{
                key: copy.deepcopy(train_cfg.get(key))
                for key in train_fields
                if key not in {"stage1_seed", "loader_seed"}
            },
            "stage1_seed": int(
                train_cfg.get("stage1_seed", train_cfg.get("seed", 42))
            ),
            "loader_seed": int(
                train_cfg.get("loader_seed", train_cfg.get("seed", 42))
            ),
        },
        "data": {key: copy.deepcopy(data_cfg.get(key)) for key in data_fields},
        "validation_selection": role_identity,
        "manifest_vocab": {
            key: copy.deepcopy(manifest_vocab_provenance.get(key))
            for key in (
                "manifest_vocab_sha256",
                "train_csv_sha256",
                "train_sample_ids_sha256",
                "num_train_samples",
            )
        },
    }


def _canonical_mapping_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_artifact_canonical_tree(value: Any) -> list[Any]:
    """Encode artifact metadata without lossy NaN/Inf or type coercion."""

    if value is None:
        return ["none"]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)):
        return ["int", str(int(value))]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            raise ValueError("NaN is forbidden in pipeline decision metadata")
        # float.hex() is deterministic, exact, and keeps +Inf/-Inf distinct.
        return ["float", numeric.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (list, tuple)):
        return [
            "list",
            [_strict_artifact_canonical_tree(item) for item in value],
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(
                "Pipeline decision metadata mappings require string keys"
            )
        return [
            "dict",
            [
                [key, _strict_artifact_canonical_tree(value[key])]
                for key in sorted(value)
            ],
        ]
    raise TypeError(
        "Unsupported pipeline decision metadata type: "
        f"{type(value).__name__}"
    )


def _pipeline_decision_metadata_sha256(
    checkpoint: dict[str, Any],
) -> str:
    """Hash deploy-relevant fitted metadata, excluding relocatable paths."""

    cfg = checkpoint.get("cfg")
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Pipeline checkpoint cfg must be a mapping")
    validation_split = checkpoint.get("validation_split") or {}
    manifest_provenance = checkpoint.get("manifest_vocab_provenance") or {}
    if not isinstance(validation_split, dict):
        raise ValueError("Pipeline validation_split must be a mapping")
    if not isinstance(manifest_provenance, dict):
        raise ValueError("Pipeline manifest provenance must be a mapping")
    payload = {
        "domain": "tri_modal_pipeline_decision_metadata_v1",
        "checkpoint_stage": checkpoint.get("checkpoint_stage"),
        "pipeline_artifact_schema_version": checkpoint.get(
            "pipeline_artifact_schema_version"
        ),
        "pipeline_model_state_sha256": checkpoint.get(
            "pipeline_model_state_sha256"
        ),
        "encoder_checkpoint_sha256": checkpoint.get(
            "encoder_checkpoint_sha256"
        ),
        "method_implementation_sha256": checkpoint.get(
            "method_implementation_sha256"
        ),
        "checkpoint_semantic_signature": _checkpoint_semantic_signature(cfg),
        "calibration": checkpoint.get("calibration"),
        "posthoc_oof_clean_rows": checkpoint.get("posthoc_oof_clean_rows"),
        "posthoc_oof_clean_rows_schema_version": checkpoint.get(
            "posthoc_oof_clean_rows_schema_version"
        ),
        "classification_threshold": checkpoint.get(
            "classification_threshold"
        ),
        "rejection_threshold": checkpoint.get("rejection_threshold"),
        "conformal_thresholds": checkpoint.get("conformal_thresholds"),
        "risk_control_thresholds": checkpoint.get(
            "risk_control_thresholds"
        ),
        "decision_calibration_signature": checkpoint.get(
            "decision_calibration_signature"
        ),
        "decision_calibration_data_identity": checkpoint.get(
            "decision_calibration_data_identity"
        ),
        "validation_split_identity": {
            key: validation_split.get(key)
            for key in _PIPELINE_VALIDATION_IDENTITY_FIELDS
        },
        "manifest_vocab_provenance": {
            key: manifest_provenance.get(key)
            for key in _PIPELINE_MANIFEST_IDENTITY_FIELDS
        },
    }
    try:
        canonical = _strict_artifact_canonical_tree(payload)
        encoded = json.dumps(
            canonical,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Pipeline decision metadata is not canonically serializable"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_encoder_stage_checkpoint(
    checkpoint: dict[str, Any],
    *,
    current_cfg: dict,
    validation_split: dict[str, Any],
    manifest_vocab_provenance: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, torch.Tensor]:
    validate_checkpoint_stage(
        checkpoint,
        expected=CHECKPOINT_STAGE_ENCODER_SELECTED,
        checkpoint_path=checkpoint_path,
    )
    if int(checkpoint.get("encoder_stage_artifact_schema_version", -1)) != ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Encoder artifact {checkpoint_path} uses an unsupported schema"
        )
    state = checkpoint.get("encoder_stage_state")
    if not isinstance(state, dict):
        raise ValueError(
            f"Encoder artifact {checkpoint_path} has no encoder_stage_state"
        )
    expected_state_sha = str(checkpoint.get("encoder_stage_state_sha256") or "")
    actual_state_sha = _state_dict_sha256(state)
    if expected_state_sha != actual_state_sha:
        raise ValueError(
            f"Encoder artifact state hash mismatch: {checkpoint_path}"
        )
    saved_implementation = str(
        checkpoint.get("encoder_stage_implementation_sha256") or ""
    )
    current_implementation = _encoder_stage_implementation_sha256()
    if saved_implementation != current_implementation:
        raise ValueError(
            "Encoder artifact was produced by a different Stage-1 "
            "implementation; retrain only the encoder artifact"
        )
    current_identity = _encoder_stage_semantic_signature(
        current_cfg, validation_split, manifest_vocab_provenance
    )
    saved_identity = checkpoint.get("encoder_stage_identity")
    if saved_identity != current_identity:
        raise ValueError(
            "Encoder artifact Stage-1 semantics differ from the current run"
        )
    expected_identity_sha = str(
        checkpoint.get("encoder_stage_identity_sha256") or ""
    )
    if expected_identity_sha != _canonical_mapping_sha256(current_identity):
        raise ValueError("Encoder artifact identity hash mismatch")
    return state


def _checkpoint_semantic_signature(cfg: dict) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    method_cfg = cfg.get("method", {}) or {}
    fusion_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
    calibration_cfg = copy.deepcopy(cfg.get("calibration", {}) or {})
    active_method_cfg = _canonicalize_inactive_stage_optimization(
        {"fusion": fusion_cfg, "calibration": calibration_cfg}
    )
    fusion_cfg = active_method_cfg["fusion"]
    calibration_cfg = active_method_cfg["calibration"]
    return {
        # The display name may intentionally change for eval-only wrappers, but
        # protocol_id and the loss definition determine what was actually
        # learned.  Keeping them here prevents a generic CE/Brier checkpoint
        # from being silently reused as a TMC/ECML-style adapted checkpoint merely
        # because both share the same encoders and fusion module.
        "method_protocol_id": copy.deepcopy(method_cfg.get("protocol_id")),
        "model": copy.deepcopy(cfg.get("model", {}) or {}),
        "fusion": fusion_cfg,
        "loss": copy.deepcopy(cfg.get("loss", {}) or {}),
        "calibration": calibration_cfg,
        "data": {
            key: copy.deepcopy(data_cfg.get(key))
            for key in (
                "max_api_events_per_sample",
                "label_map",
                "require_manifest_vocab_provenance",
                "expected_manifest_vocab_sha256",
                "expected_manifest_train_csv_sha256",
                "expected_manifest_train_sample_ids_sha256",
                "expected_pt_build_fingerprint",
            )
        },
    }


def validate_eval_checkpoint_config(
    current_cfg: dict,
    checkpoint_cfg: Any,
) -> None:
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(
            "Evaluation checkpoint does not contain its training config. "
            "Retrain it with the current staged pipeline."
        )
    current = _checkpoint_semantic_signature(current_cfg)
    saved = _checkpoint_semantic_signature(checkpoint_cfg)
    if current != saved:
        raise ValueError(
            "Evaluation config changes model/data semantics relative to the checkpoint. "
            "Use the checkpoint's training config and override only eval paths/settings, "
            "or retrain the checkpoint."
        )


def validate_checkpoint_implementation(
    checkpoint: dict[str, Any],
) -> None:
    """Reject checkpoints produced by a different decision implementation."""
    saved = str(checkpoint.get("method_implementation_sha256", ""))
    current = _method_implementation_sha256()
    if not saved:
        raise ValueError(
            "Evaluation checkpoint predates implementation fingerprinting. "
            "Retrain it with the current code."
        )
    if saved != current:
        raise ValueError(
            "Evaluation checkpoint was produced by a different model/fusion "
            "implementation. Retrain it with the current code."
        )


def validate_checkpoint_manifest_vocab_provenance(
    checkpoint: dict[str, Any],
    current_provenance: dict[str, Any],
) -> None:
    """Reject encoder weights whose Manifest columns have different semantics."""
    if not bool(current_provenance.get("required", False)):
        return
    if not bool(current_provenance.get("verified", False)):
        raise ValueError("Current Manifest vocabulary provenance is not verified")
    saved = checkpoint.get("manifest_vocab_provenance")
    if not isinstance(saved, dict) or not bool(saved.get("verified", False)):
        raise ValueError(
            "Evaluation checkpoint has no verified Manifest vocabulary "
            "provenance. Retrain it with the current train-only vocabulary."
        )
    fields = (
        "manifest_vocab_sha256",
        "train_csv_sha256",
        "train_sample_ids_sha256",
        "num_train_samples",
    )
    mismatches = {
        field: {
            "checkpoint": saved.get(field),
            "current": current_provenance.get(field),
        }
        for field in fields
        if saved.get(field) != current_provenance.get(field)
    }
    if mismatches:
        raise ValueError(
            "Evaluation checkpoint was trained with a different Manifest "
            "vocabulary/train split. Manifest feature columns cannot be reused; "
            f"retrain the checkpoint. mismatches={mismatches}"
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


def _load_linked_encoder_checkpoint(
    pipeline_path: str | Path,
    pipeline_checkpoint: dict[str, Any],
    *,
    map_location: Any,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify the encoder artifact linked by a pipeline."""
    pipeline_path = Path(pipeline_path)
    validate_pipeline_checkpoint_artifact(
        pipeline_checkpoint, checkpoint_path=pipeline_path
    )
    encoder_reference = pipeline_checkpoint.get("encoder_checkpoint_path")
    if not str(encoder_reference or "").strip():
        raise ValueError(
            f"Pipeline checkpoint {pipeline_path} does not link to its "
            "encoder-selected checkpoint; post-hoc refitting cannot safely "
            "reconstruct the training lifecycle."
        )
    encoder_path = Path(str(encoder_reference))
    if not encoder_path.is_absolute():
        encoder_path = pipeline_path.parent / encoder_path
    if not encoder_path.exists():
        raise FileNotFoundError(
            "Encoder-selected checkpoint linked by the pipeline artifact was "
            f"not found: {encoder_path}"
        )
    expected_encoder_sha256 = str(
        pipeline_checkpoint.get("encoder_checkpoint_sha256", "")
    ).strip().lower()
    if len(expected_encoder_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_encoder_sha256
    ):
        raise ValueError(
            f"Pipeline checkpoint {pipeline_path} is missing a valid "
            "encoder_checkpoint_sha256; old or ambiguous pipeline artifacts "
            "cannot be used for post-hoc refitting"
        )
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


def validate_pipeline_checkpoint_artifact(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: str | Path,
) -> None:
    """Validate the self-contained state and provenance link of a pipeline."""

    validate_checkpoint_stage(
        checkpoint,
        expected=CHECKPOINT_STAGE_PIPELINE_FITTED,
        checkpoint_path=checkpoint_path,
    )
    if int(checkpoint.get("pipeline_artifact_schema_version", -1)) != (
        PIPELINE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Pipeline checkpoint {checkpoint_path} uses an unsupported or "
            "missing pipeline artifact schema; retrain with the current code"
        )
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(
            f"Pipeline checkpoint {checkpoint_path} has no model state"
        )
    expected_model_sha256 = str(
        checkpoint.get("pipeline_model_state_sha256", "")
    ).strip().lower()
    if len(expected_model_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_model_sha256
    ):
        raise ValueError(
            f"Pipeline checkpoint {checkpoint_path} is missing a valid "
            "pipeline_model_state_sha256"
        )
    if _state_dict_sha256(model_state) != expected_model_sha256:
        raise ValueError(
            f"Pipeline checkpoint model-state hash mismatch: {checkpoint_path}"
        )
    if not str(checkpoint.get("encoder_checkpoint_path") or "").strip():
        raise ValueError(
            f"Pipeline checkpoint {checkpoint_path} does not link to its "
            "encoder-selected checkpoint"
        )
    encoder_sha256 = str(
        checkpoint.get("encoder_checkpoint_sha256", "")
    ).strip().lower()
    if len(encoder_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in encoder_sha256
    ):
        raise ValueError(
            f"Pipeline checkpoint {checkpoint_path} is missing a valid "
            "encoder_checkpoint_sha256"
        )
    expected_metadata_sha256 = str(
        checkpoint.get("pipeline_decision_metadata_sha256", "")
    ).strip().lower()
    if len(expected_metadata_sha256) != 64 or any(
        char not in "0123456789abcdef"
        for char in expected_metadata_sha256
    ):
        raise ValueError(
            f"Pipeline checkpoint {checkpoint_path} is missing a valid "
            "pipeline_decision_metadata_sha256"
        )
    actual_metadata_sha256 = _pipeline_decision_metadata_sha256(checkpoint)
    if actual_metadata_sha256 != expected_metadata_sha256:
        raise ValueError(
            "Pipeline checkpoint decision-metadata hash mismatch: "
            f"{checkpoint_path}"
        )


def _load_eval_checkpoint(
    checkpoint_path: str | Path,
    *,
    refit_posthoc_calibration: bool,
    map_location: Any,
) -> tuple[Path, dict[str, Any]]:
    """Load the only checkpoint stage valid for the requested eval lifecycle.

    Ordinary evaluation consumes a pipeline-fitted artifact. A post-hoc refit
    consumes the encoder-selected artifact so the selected encoders and branch
    heads are preserved while I1, I2, and I3 are fitted again without inheriting
    any previous post-hoc labels. A pipeline checkpoint may link to that
    encoder-selected artifact.
    """
    requested_path = Path(checkpoint_path)
    requested_checkpoint = torch.load(
        requested_path, map_location=map_location, weights_only=True
    )
    requested_stage = validate_checkpoint_stage(
        requested_checkpoint, checkpoint_path=requested_path
    )
    if requested_stage == CHECKPOINT_STAGE_PIPELINE_FITTED:
        validate_pipeline_checkpoint_artifact(
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
    return _load_linked_encoder_checkpoint(
        requested_path,
        requested_checkpoint,
        map_location=map_location,
    )


def _resolve_refit_decision_calibration(eval_cfg: dict | None) -> bool:
    """Resolve the current decision-calibration refit flag."""
    eval_cfg = eval_cfg or {}
    if "refit_rejection_threshold" in eval_cfg:
        raise ValueError(
            "eval.refit_rejection_threshold was removed; use "
            "eval.refit_decision_calibration"
        )
    return bool(eval_cfg.get("refit_decision_calibration", False))


def _decision_calibration_signature(cfg: dict) -> dict[str, Any]:
    """Canonicalize every setting that changes fitted decision artifacts."""
    canonical_cfg = _canonicalize_classification_threshold_config(
        _canonicalize_selective_prediction_config(cfg)
    )
    classification_cfg = canonical_cfg.get("classification_threshold", {}) or {}
    classification_enabled = bool(classification_cfg.get("enabled", False))
    classification = {
        "enabled": classification_enabled,
        "objective": (
            str(classification_cfg.get("objective", "macro_f1")).lower()
            if classification_enabled
            else "disabled"
        ),
        "selection_rule": (
            str(classification_cfg["selection_rule"])
            if classification_enabled
            else "disabled"
        ),
        "constraint": "none" if classification_enabled else "disabled",
    }

    selective_cfg = canonical_cfg.get("selective_prediction", {}) or {}
    selective_enabled = bool(selective_cfg.get("enabled", False))
    mode = _selective_prediction_mode(selective_cfg) if selective_enabled else "disabled"
    selective: dict[str, Any] = {
        "enabled": selective_enabled,
        "mode": mode,
        "score_type": (
            _selective_score_type(selective_cfg)
            if selective_enabled
            else "disabled"
        ),
        "acceptance_comparison": (
            ACCEPTANCE_THRESHOLD_COMPARISON
            if selective_enabled and mode in {"threshold", "risk_control"}
            else None
        ),
    }
    if selective_enabled and selective["score_type"] == "model_acceptance":
        fusion_cfg = cfg.get("fusion", {}) or {}
        if str(fusion_cfg.get("combination", "")).lower() == "routed":
            selective["model_acceptance_definition"] = (
                "routed:one_minus_threshold_aligned_fn_risk"
            )
        else:
            selective["model_acceptance_definition"] = (
                "fusion:mixture_certainty"
            )
    if selective_enabled and mode == "risk_control":
        risk_target = str(
            selective_cfg.get(
                "risk_target", "accepted_fn_risk_among_malware"
            )
        ).lower()
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
    refit_posthoc_calibration: bool = False,
    current_data_identity: dict[str, Any] | None = None,
) -> None:
    """Prevent fitted thresholds from being reused under different semantics."""
    if refit_decision_calibration:
        current_signature = _decision_calibration_signature(current_cfg)
        saved_signature = checkpoint.get("decision_calibration_signature")
        current_fusion = current_cfg.get("fusion", {}) or {}
        current_routing = current_fusion.get("routing", {}) or {}
        threshold_aligned_risk = (
            str(current_routing.get("risk_target", "")).strip().lower()
            == "threshold_malware_false_negative"
            and str(current_routing.get("risk_mode", "learned")).lower()
            == "learned"
        )
        if (
            threshold_aligned_risk
            and not refit_posthoc_calibration
        ):
            if not isinstance(saved_signature, dict):
                raise ValueError(
                    "Threshold-aligned I2 risk cannot be reused without the "
                    "checkpoint decision signature; rerun with "
                    "eval.refit_posthoc_calibration=true"
                )
            if current_signature["classification"] != saved_signature.get(
                "classification"
            ):
                raise ValueError(
                    "Classification-threshold semantics changed the malware-FN "
                    "target used by I2 risk; rerun with "
                    "eval.refit_posthoc_calibration=true, not a decision-only "
                    "refit"
                )
        # Decision settings may intentionally change, but a decision-only
        # refit still reuses the checkpoint's upstream OOF predictions. Those
        # rows must describe exactly the current post-hoc identity pool.
        if checkpoint.get("posthoc_oof_clean_rows"):
            saved_identity = checkpoint.get("decision_calibration_data_identity")
            if not isinstance(saved_identity, dict):
                raise ValueError(
                    "Checkpoint OOF rows have no decision-calibration data "
                    "identity; rerun with eval.refit_posthoc_calibration=true"
                )
            if not isinstance(current_data_identity, dict):
                raise ValueError(
                    "Current post-hoc data identity is required when reusing "
                    "checkpoint OOF rows"
                )
            saved_posthoc_identity = {
                "validation_csv_sha256": saved_identity.get(
                    "validation_csv_sha256"
                ),
                "posthoc_calibration": saved_identity.get(
                    "posthoc_calibration"
                ),
            }
            current_posthoc_identity = {
                "validation_csv_sha256": current_data_identity.get(
                    "validation_csv_sha256"
                ),
                "posthoc_calibration": current_data_identity.get(
                    "posthoc_calibration"
                ),
            }
            if saved_posthoc_identity != current_posthoc_identity:
                raise ValueError(
                    "Post-hoc data identity differs from the checkpoint OOF "
                    "rows; rerun with eval.refit_posthoc_calibration=true"
                )
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
    if classification_is_used and current_data_identity is not None:
        saved_identity = checkpoint.get("decision_calibration_data_identity")
        if not isinstance(saved_identity, dict):
            raise ValueError(
                "Pipeline checkpoint has no decision_calibration_data_identity; "
                "retrain it or set eval.refit_decision_calibration=true"
            )
        if saved_identity != current_data_identity:
            raise ValueError(
                "Decision-calibration data identity differs from the fitted "
                "checkpoint artifact; set eval.refit_decision_calibration=true"
            )


def validate_threshold_aligned_risk_cutoff(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    classification_log_odds_threshold: float | None,
) -> None:
    """Keep a consumed threshold-trained risk head tied to its classifier cutoff."""
    fusion_cfg = cfg.get("fusion", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    selective_enabled = bool(
        (cfg.get("selective_prediction", {}) or {}).get("enabled", False)
    )
    # With no accept/reject decision, the saved risk head cannot affect forced
    # classification.  Its historical target cutoff is therefore irrelevant
    # to a genuine selective-off factorial cell.
    if not selective_enabled:
        return
    if (
        str(fusion_cfg.get("combination", "")).lower() != "routed"
        or not bool(routing_cfg.get("enabled", False))
        or str(routing_cfg.get("risk_mode", "learned")).lower() != "learned"
        or str(routing_cfg.get("risk_target", "")).lower()
        != "threshold_malware_false_negative"
    ):
        return
    classification_enabled = bool(
        (cfg.get("classification_threshold", {}) or {}).get("enabled", False)
    )
    if classification_enabled:
        if classification_log_odds_threshold is None:
            raise RuntimeError(
                "Threshold-aligned routed risk requires the fitted raw "
                "classification cutoff"
            )
        expected = float(np.float32(classification_log_odds_threshold))
    else:
        expected = 0.0
    discount_fusion = getattr(model, "discount_fusion", None)
    router = getattr(discount_fusion, "opinion_router", None)
    if router is None or not bool(router.risk_decision_threshold_active):
        raise RuntimeError(
            "Threshold-aligned routed risk has no active decision cutoff; "
            "rerun post-hoc calibration"
        )
    actual = float(
        router._risk_decision_log_odds_threshold.detach().cpu().item()
    )
    if actual != expected:
        raise RuntimeError(
            "Threshold-aligned routed risk cutoff differs from the deployed "
            f"classifier cutoff: risk={actual} classifier={expected}. Rerun "
            "post-hoc calibration before saving/evaluating this artifact."
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file_verified_atomic(source: str | Path, destination: str | Path) -> str:
    """Copy an artifact without exposing a partial or silently changed file."""

    source_path = Path(source)
    destination_path = Path(destination)
    source_sha256 = _file_sha256(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        shutil.copy2(source_path, temporary_path)
        copied_sha256 = _file_sha256(temporary_path)
        if copied_sha256 != source_sha256:
            raise RuntimeError(
                "Copied artifact hash differs from its source: "
                f"source={source_path} destination={destination_path}"
            )
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    if _file_sha256(destination_path) != source_sha256:
        raise RuntimeError(
            f"Artifact changed after atomic installation: {destination_path}"
        )
    return source_sha256


def _atomic_torch_save(payload: Any, destination: str | Path) -> None:
    """Install a torch artifact atomically so interrupted runs leave no partial PT."""

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


def _calibration_subset_identity(
    dataset,
    indices: list[int] | tuple[int, ...] | None,
) -> dict[str, Any]:
    """Fingerprint one validation subset without persisting sample identities."""
    resolved_indices = (
        list(range(len(dataset)))
        if indices is None
        else [int(index) for index in indices]
    )
    sids = list(getattr(dataset, "sample_sids", []))
    if len(sids) != len(dataset):
        sids = [str(index) for index in range(len(dataset))]
    groups = list(getattr(dataset, "sample_groups", []))
    if len(groups) != len(dataset):
        groups = list(sids)
    labels = list(getattr(dataset, "sample_labels", []))
    if len(labels) != len(dataset):
        samples = list(getattr(dataset, "samples", []))
        if len(samples) != len(dataset):
            raise ValueError(
                "Validation dataset must expose sample_labels to fingerprint "
                "decision-calibration subsets"
            )
        labels = [int(sample[1]) for sample in samples]

    identities = sorted(
        (
            str(sids[index]).strip().lower(),
            str(groups[index]).strip().lower(),
            int(labels[index]),
        )
        for index in resolved_indices
    )
    digest = hashlib.sha256()
    class_counts: dict[str, int] = {}
    for sid, group, label in identities:
        for value in (sid, group):
            encoded_value = value.encode("utf-8")
            digest.update(len(encoded_value).to_bytes(8, byteorder="big"))
            digest.update(encoded_value)
            digest.update(b"\0")
        digest.update(str(label).encode("ascii"))
        digest.update(b"\n")
        key = str(label)
        class_counts[key] = class_counts.get(key, 0) + 1
    return {
        "num_rows": int(len(identities)),
        "class_counts": dict(sorted(class_counts.items())),
        "row_identity_sha256": digest.hexdigest(),
    }


def _decision_calibration_data_identity(
    cfg: dict,
    dataset,
    *,
    posthoc_indices: list[int] | tuple[int, ...] | None,
    decision_indices: list[int] | tuple[int, ...] | None,
) -> dict[str, Any]:
    """Record the exact rows and source CSV used to fit decision artifacts."""
    data_cfg = cfg.get("data", {}) or {}
    data_root = data_cfg.get("root", "")
    val_csv = resolve(data_root, data_cfg.get("val_csv", ""))
    val_csv_path = Path(val_csv)
    if not val_csv_path.is_file():
        raise FileNotFoundError(
            f"Validation CSV required for decision identity was not found: {val_csv_path}"
        )
    return {
        "schema_version": 1,
        "validation_csv_sha256": _file_sha256(val_csv_path),
        "posthoc_calibration": _calibration_subset_identity(
            dataset, posthoc_indices
        ),
        "decision_calibration": _calibration_subset_identity(
            dataset, decision_indices
        ),
    }


def _resolve_graph_node_budget(model_cfg: dict[str, Any]) -> int:
    graph_cfg = model_cfg.get("graph_encoder", {}) or {}
    removed = sorted(
        set(graph_cfg) & {"account_for_encoder_budget", "max_nodes"}
    )
    if removed:
        raise ValueError(
            "Removed model.graph_encoder settings are unsupported: "
            f"{removed}. Declare the single graph budget only as "
            "model.max_nodes_gnn."
        )
    if "max_nodes_gnn" not in model_cfg:
        raise ValueError(
            "model.max_nodes_gnn is required as the single dataset/encoder "
            "graph-node budget"
        )
    budget = strict_finite_integer(
        model_cfg["max_nodes_gnn"],
        field_name="model.max_nodes_gnn",
    )
    if budget <= 0:
        raise ValueError("model.max_nodes_gnn must be a positive integer")
    return budget


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
        "strict_split_integrity",
        "strict_partition_isolation",
        "allow_pt_superset",
        "label_map",
        "manifest_vocab_path",
        "require_manifest_vocab_provenance",
        "expected_manifest_vocab_sha256",
        "expected_manifest_train_csv_sha256",
        "expected_manifest_train_sample_ids_sha256",
        "expected_pt_build_fingerprint",
    }
    unknown_data_keys = sorted(set(data_cfg) - allowed_data_keys)
    if unknown_data_keys:
        raise ValueError(f"Unsupported data settings: {unknown_data_keys}")
    model_cfg = cfg.get("model", {})
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
        raise ValueError("data.max_api_events_per_sample must be positive")
    if api_encoder_capacity <= 0:
        raise ValueError("model.api_encoder.max_seq_len must be positive")
    if api_event_budget > api_encoder_capacity:
        raise ValueError(
            "data.max_api_events_per_sample must be <= "
            "model.api_encoder.max_seq_len so the dataset is the only API "
            f"truncation point; got {api_event_budget} > {api_encoder_capacity}"
        )
    _validated_max_failed_ratio(data_cfg.get("max_failed_ratio", 0.0))
    num_classes = strict_finite_integer(
        model_cfg.get("num_classes", 2), field_name="model.num_classes"
    )
    if num_classes != 2:
        raise ValueError(
            "The current tri-modal pipeline is binary-only: model.num_classes "
            "must be 2 because probability calibration, classification "
            "thresholding, malware-FN risk, and I3 all use the benign/malware "
            "contract explicitly."
        )
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    graph_node_budget = _resolve_graph_node_budget(model_cfg)
    return {
        "is_train": is_train,
        "eval_perturb_type": perturb_type,
        "eval_perturb_strength": perturb_strength,
        "manifest_dim": int(manifest_cfg.get("in_dim", 256)),
        "manifest_category_dim": int(manifest_cfg.get("category_dim", 12)),
        "manifest_stats_dim": int(manifest_cfg.get("stats_dim", 11)),
        "manifest_permission_dim": int(manifest_cfg.get("permission_dim", 128)),
        "manifest_intent_dim": int(manifest_cfg.get("intent_dim", 64)),
        "manifest_feature_dim": int(manifest_cfg.get("feature_dim", 32)),
        "max_api_events_per_sample": api_event_budget,
        "max_graph_nodes_per_sample": graph_node_budget,
        "drop_graph_behavior_hints": bool(model_cfg.get("graph_encoder", {}).get("drop_extracted_behavior_hints", False)),
        "num_classes": num_classes,
        "label_map": data_cfg.get("label_map"),
        "strict_split_integrity": bool(data_cfg.get("strict_split_integrity", True)),
        "allow_pt_superset": bool(data_cfg.get("allow_pt_superset", False)),
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
    seed_namespace: str | None = None,
    persistent_workers_override: bool | None = None,
    batch_sampler_override=None,
    collate_fn_override=None,
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
    base_loader_seed = int(
        train_cfg.get("loader_seed", train_cfg.get("seed", 42))
    )
    resolved_namespace = str(
        seed_namespace or ("train" if is_train else "evaluation")
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        _namespaced_seed(base_loader_seed, resolved_namespace)
    )
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and workers > 0,
        "collate_fn": collate_fn_override or robust_collate_fn,
        "generator": loader_generator,
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
                "shuffle": is_train,
            }
        )
    else:
        if is_train:
            raise ValueError("A fixed batch sampler is only valid for evaluation loaders")
        loader_kwargs["batch_sampler"] = batch_sampler_override
    if workers > 0 and train_cfg.get("prefetch_factor") is not None:
        prefetch_factor = int(train_cfg.get("prefetch_factor"))
        if prefetch_factor <= 0:
            raise ValueError("train.prefetch_factor must be positive when num_workers > 0")
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def _require_stratified_group_validation_split(calibration_cfg: dict) -> None:
    value = calibration_cfg.get("stratified_group_split", True)
    if not isinstance(value, bool) or not value:
        raise ValueError(
            "calibration.stratified_group_split must be true; disabling the "
            "implemented year-label/package-group split is unsupported"
        )


def split_validation_dataset(cfg: dict, dataset) -> tuple[Subset, Subset, dict[str, Any]]:
    """Deterministically separate checkpoint selection from calibration.

    Package/sample groups remain intact while a deterministic greedy assignment
    keeps both halves close to the full validation year-label distribution.
    """
    calibration_cfg = cfg.get("calibration", {}) or {}
    _require_stratified_group_validation_split(calibration_cfg)
    fraction = float(
        calibration_cfg.get(
            "validation_fraction", VALIDATION_HOLDOUT_FRACTION
        )
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration.validation_fraction must be within (0, 1)")
    size = len(dataset)
    if size < 2:
        raise ValueError("Validation dataset needs at least two samples for selection/calibration split")
    seed = int(calibration_cfg.get("split_seed", cfg.get("train", {}).get("seed", 42)))
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
            "Formal validation splitting requires complete package-group and "
            "year-label metadata; missing or length-mismatched fields: "
            f"{missing_metadata}"
        )
    labels = [int(label) for label in labels]
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
            "num_validation": size,
            "num_selection": len(selection_indices),
            "num_calibration": len(calibration_indices),
            "selection_fraction_of_validation": (
                len(selection_indices) / float(size)
            ),
            "calibration_fraction_of_validation": (
                len(calibration_indices) / float(size)
            ),
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

    The final subset must remain untouched by reliability calibration, routing,
    and classification-threshold selection. Reusing it in those label-dependent
    steps would invalidate the held-out calibration argument used by conformal
    and risk-control rules.
    """
    calibration_cfg = cfg.get("calibration", {}) or {}
    fraction = float(
        calibration_cfg.get(
            "conformal_fraction", CONFORMAL_WITHIN_HOLDOUT_FRACTION
        )
    )
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
            "posthoc_fraction_of_validation": (
                len(posthoc_indices) / float(len(dataset))
            ),
            "decision_fraction_of_validation": (
                len(conformal_indices) / float(len(dataset))
            ),
            "posthoc_fraction_of_holdout": (
                len(posthoc_indices) / float(len(original_indices))
            ),
            "decision_fraction_of_holdout": (
                len(conformal_indices) / float(len(original_indices))
            ),
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


def _load_fixed_validation_roles(
    cfg: dict,
    dataset,
) -> tuple[Subset, Subset, Subset, dict[str, Any]] | None:
    """Load an immutable three-way validation assignment by sample identity.

    Fractions remain useful when initially generating a protocol, but formal
    reruns must not silently reshuffle checkpoint-selection samples merely
    because a post-hoc budget changes.  A checked-in identity manifest makes
    those three statistical roles explicit and independently auditable.
    """

    calibration_cfg = cfg.get("calibration", {}) or {}
    raw_path = str(calibration_cfg.get("role_assignment_path") or "").strip()
    required = bool(calibration_cfg.get("require_role_assignment", False))
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
        raise ValueError("Validation role assignment must be a JSON mapping")
    if int(payload.get("schema_version", -1)) != VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError(
            "Validation role assignment schema mismatch: "
            f"expected={VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION} "
            f"actual={payload.get('schema_version')!r}"
        )

    val_csv = Path(
        resolve(data_cfg.get("root", ""), data_cfg.get("val_csv", ""))
    )
    expected_csv_sha = str(payload.get("validation_csv_sha256") or "").lower()
    actual_csv_sha = _file_sha256(val_csv)
    if expected_csv_sha != actual_csv_sha:
        raise ValueError(
            "Validation role assignment was built for a different validation "
            f"CSV: expected={expected_csv_sha!r} actual={actual_csv_sha!r}"
        )

    role_names = (
        "checkpoint_selection",
        "posthoc_calibration",
        "decision_calibration",
    )
    roles = payload.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(role_names):
        raise ValueError(
            "Validation role assignment must contain exactly roles "
            f"{list(role_names)}"
        )
    dataset_sids = [str(value).strip().lower() for value in getattr(dataset, "sample_sids", [])]
    dataset_groups = [str(value) for value in getattr(dataset, "sample_groups", [])]
    dataset_labels = [int(value) for value in getattr(dataset, "sample_labels", [])]
    dataset_years = [int(value) for value in getattr(dataset, "sample_years", [])]
    size = len(dataset)
    if not (
        len(dataset_sids)
        == len(dataset_groups)
        == len(dataset_labels)
        == len(dataset_years)
        == size
    ):
        raise ValueError(
            "Fixed validation roles require complete sid/group/label/year metadata"
        )
    if len(set(dataset_sids)) != size:
        raise ValueError("Validation dataset contains duplicate sample identities")
    index_by_sid = {sid: index for index, sid in enumerate(dataset_sids)}
    role_ids: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role_name in role_names:
        values = roles[role_name]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Validation role {role_name!r} must be a non-empty list"
            )
        normalized = [str(value).strip().lower() for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"Validation role {role_name!r} contains an empty id")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"Validation role {role_name!r} contains duplicate ids"
            )
        overlap = seen.intersection(normalized)
        if overlap:
            raise ValueError(
                "Validation roles overlap; examples="
                f"{sorted(overlap)[:10]}"
            )
        unknown = sorted(set(normalized) - set(index_by_sid))
        if unknown:
            raise ValueError(
                f"Validation role {role_name!r} contains unknown ids: {unknown[:10]}"
            )
        seen.update(normalized)
        role_ids[role_name] = normalized
    missing = sorted(set(dataset_sids) - seen)
    if missing:
        raise ValueError(
            "Validation role assignment does not cover the full validation set; "
            f"missing={len(missing)} examples={missing[:10]}"
        )

    group_roles: dict[str, set[str]] = {}
    for role_name, values in role_ids.items():
        for sid in values:
            group_roles.setdefault(dataset_groups[index_by_sid[sid]], set()).add(
                role_name
            )
    crossed_groups = sorted(
        group for group, assigned in group_roles.items() if len(assigned) > 1
    )
    if crossed_groups:
        raise ValueError(
            "Validation role assignment splits package groups across roles; "
            f"examples={crossed_groups[:10]}"
        )

    role_indices = {
        name: sorted(index_by_sid[sid] for sid in role_ids[name])
        for name in role_names
    }
    label_values = sorted(set(dataset_labels))
    year_values = sorted(set(dataset_years))

    def _counts(indices: list[int], values: list[int], source: list[int]) -> dict[int, int]:
        return {
            value: sum(int(source[index] == value) for index in indices)
            for value in values
        }

    def _joint_counts(indices: list[int]) -> dict[str, int]:
        return {
            f"{year}:{label}": sum(
                int(dataset_years[index] == year and dataset_labels[index] == label)
                for index in indices
            )
            for year in year_values
            for label in label_values
        }

    selection_indices = role_indices["checkpoint_selection"]
    posthoc_indices = role_indices["posthoc_calibration"]
    decision_indices = role_indices["decision_calibration"]
    assignment_sha = _file_sha256(path)
    assignment_semantic_sha = _canonical_mapping_sha256(
        {
            "schema_version": VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION,
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
        "role_assignment_schema_version": VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION,
        "validation_csv_sha256": actual_csv_sha,
        "split_seed": int(payload.get("split_seed", calibration_cfg.get("split_seed", 42))),
        "validation_fraction": (len(posthoc_indices) + len(decision_indices)) / float(size),
        "num_validation": size,
        "num_selection": len(selection_indices),
        "num_calibration": len(posthoc_indices) + len(decision_indices),
        "num_posthoc_calibration": len(posthoc_indices),
        "num_conformal_calibration": len(decision_indices),
        "selection_fraction_of_validation": len(selection_indices) / float(size),
        "calibration_fraction_of_validation": (len(posthoc_indices) + len(decision_indices)) / float(size),
        "posthoc_fraction_of_validation": len(posthoc_indices) / float(size),
        "decision_fraction_of_validation": len(decision_indices) / float(size),
        "selection_label_counts": _counts(selection_indices, label_values, dataset_labels),
        "posthoc_label_counts": _counts(posthoc_indices, label_values, dataset_labels),
        "conformal_label_counts": _counts(decision_indices, label_values, dataset_labels),
        "selection_year_counts": _counts(selection_indices, year_values, dataset_years),
        "posthoc_year_counts": _counts(posthoc_indices, year_values, dataset_years),
        "conformal_year_counts": _counts(decision_indices, year_values, dataset_years),
        "selection_year_label_counts": _joint_counts(selection_indices),
        "posthoc_year_label_counts": _joint_counts(posthoc_indices),
        "conformal_year_label_counts": _joint_counts(decision_indices),
        "num_selection_groups": len({dataset_groups[index] for index in selection_indices}),
        "num_calibration_groups": len({dataset_groups[index] for index in [*posthoc_indices, *decision_indices]}),
        "selection_indices": selection_indices,
        "calibration_indices": sorted([*posthoc_indices, *decision_indices]),
        "posthoc_calibration_indices": posthoc_indices,
        "conformal_calibration_indices": decision_indices,
    }
    return (
        Subset(dataset, selection_indices),
        Subset(dataset, posthoc_indices),
        Subset(dataset, decision_indices),
        summary,
    )


def _validated_max_failed_ratio(value: Any) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError("max_failed_ratio must be a finite number in [0, 1), not boolean")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_failed_ratio must be a finite number in [0, 1)") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError(
            f"max_failed_ratio must be finite and within [0, 1), got {value!r}"
        )
    if threshold != 0.0:
        raise ValueError(
            "The formal pipeline requires max_failed_ratio=0.0: dropping failed "
            "samples would make classification, selective coverage, conformal, "
            "and risk-control metrics describe only the successful subset"
        )
    return threshold


def enforce_failed_ratio(
    metrics: dict[str, Any],
    cfg: dict,
    split_name: str,
    max_failed_ratio: float | None = None,
) -> None:
    threshold = _validated_max_failed_ratio(
        cfg.get("data", {}).get("max_failed_ratio", 0.0)
        if max_failed_ratio is None
        else max_failed_ratio
    )
    num_eval = strict_finite_integer(
        metrics.get("num_eval", 0), field_name=f"{split_name}.num_eval"
    )
    num_failed = strict_finite_integer(
        metrics.get("num_failed", 0), field_name=f"{split_name}.num_failed"
    )
    if num_eval < 0 or num_failed < 0:
        raise ValueError(
            f"{split_name}: num_eval and num_failed must be non-negative; "
            f"got num_eval={num_eval}, num_failed={num_failed}"
        )
    num_requested = num_eval + num_failed
    if num_requested <= 0:
        raise RuntimeError(f"{split_name}: no requested samples were seen")
    if num_eval <= 0:
        raise RuntimeError(
            f"{split_name}: no samples were evaluated successfully "
            f"(num_requested={num_requested}, num_failed={num_failed})"
        )
    failed_ratio = float(num_failed) / float(num_requested)
    if failed_ratio > threshold:
        raise RuntimeError(
            f"{split_name}: failed sample ratio {failed_ratio:.4f} exceeds "
            f"data.max_failed_ratio={threshold:.4f}"
        )


def _build_eval_perturbation_view(
    base_dataset: RobustTriModalDataset,
    *,
    perturb_type: str,
    perturb_strength: float,
) -> RobustTriModalDataset:
    """Clone an already validated eval dataset without rescanning its PT pool.

    ``RobustTriModalDataset.__getitem__`` does not mutate dataset state, so the
    sample/index/path metadata can be shared safely across deterministic eval
    views. Only evaluation mode, perturbation type, and strength differ.
    """
    if not isinstance(base_dataset, RobustTriModalDataset):
        raise TypeError(
            "Evaluation perturbation views require a validated "
            "RobustTriModalDataset base"
        )
    if bool(base_dataset.is_train):
        raise ValueError("Evaluation perturbation views cannot be built from a train dataset")
    if perturb_type not in EVAL_PERTURB_TYPES or perturb_type == "clean":
        raise ValueError(f"Unsupported evaluation perturbation type: {perturb_type!r}")
    strength = float(perturb_strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"Evaluation perturbation strength must be within [0, 1], got {perturb_strength!r}"
        )

    view = copy.copy(base_dataset)
    view.is_train = False
    view.eval_perturb_type = perturb_type
    view.eval_perturb_strength = strength
    return view


def _normalize_robust_test_protocol(
    eval_cfg: dict[str, Any],
) -> tuple[list[str], list[float]]:
    """Validate the named robust-test matrix before any result key is emitted.

    Duplicate perturbations or strengths would generate duplicate ``result_key``
    values and let a later evaluation silently overwrite an earlier one in the
    summary.  Treat the matrix as a scientific protocol and fail before loading
    data instead.
    """
    raw_tests = eval_cfg.get("perturb_tests", ["clean"])
    if not isinstance(raw_tests, (list, tuple)) or not raw_tests:
        raise ValueError("eval.perturb_tests must be a non-empty sequence")
    perturb_tests: list[str] = []
    for raw_test in raw_tests:
        if not isinstance(raw_test, str):
            raise ValueError("eval.perturb_tests must contain non-empty strings")
        perturb = raw_test.strip().lower()
        if not perturb:
            raise ValueError("eval.perturb_tests contains an empty name")
        perturb_tests.append(perturb)
    duplicate_tests = sorted(
        {value for value in perturb_tests if perturb_tests.count(value) > 1}
    )
    if duplicate_tests:
        raise ValueError(
            "eval.perturb_tests contains duplicates that would overwrite result "
            f"keys: {duplicate_tests}"
        )
    unsupported_tests = sorted(
        value for value in set(perturb_tests) if value not in EVAL_PERTURB_TYPES
    )
    if unsupported_tests:
        raise ValueError(
            "eval.perturb_tests contains unsupported mechanisms: "
            f"{unsupported_tests}"
        )

    if eval_cfg.get("perturb_strengths") is not None:
        raw_strengths = eval_cfg.get("perturb_strengths")
        if not isinstance(raw_strengths, (list, tuple)) or not raw_strengths:
            raise ValueError(
                "eval.perturb_strengths must be a non-empty sequence when set"
            )
    else:
        raw_strengths = [eval_cfg.get("perturb_strength", 0.5)]
    perturb_strengths: list[float] = []
    for raw_strength in raw_strengths:
        if isinstance(raw_strength, bool):
            raise ValueError(
                "eval.perturb_strengths must contain finite numbers, not booleans"
            )
        try:
            strength = float(raw_strength)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "eval.perturb_strengths must contain finite numbers"
            ) from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(
                "eval.perturb_strengths values must lie within [0, 1]"
            )
        perturb_strengths.append(strength)
    duplicate_strengths = sorted(
        {
            value
            for value in perturb_strengths
            if perturb_strengths.count(value) > 1
        }
    )
    if duplicate_strengths:
        raise ValueError(
            "eval.perturb_strengths contains duplicates that would overwrite "
            f"result keys: {duplicate_strengths}"
        )
    rendered_strengths = [f"{value:g}" for value in perturb_strengths]
    duplicate_rendered_strengths = sorted(
        {
            value
            for value in rendered_strengths
            if rendered_strengths.count(value) > 1
        }
    )
    if duplicate_rendered_strengths:
        raise ValueError(
            "eval.perturb_strengths contains values that collide after result-key "
            f"formatting: {duplicate_rendered_strengths}"
        )
    return perturb_tests, perturb_strengths


def _robust_test_result_count(
    perturb_tests: list[str], perturb_strengths: list[float]
) -> int:
    """Return the number of unique summary cells produced by a valid matrix."""
    count = 0
    for perturb in perturb_tests:
        if perturb == "clean" or perturb.endswith("_missing"):
            count += 1
        else:
            count += len(perturb_strengths)
    return count


def iter_robust_test_loaders(
    cfg: dict,
    base_dataset: RobustTriModalDataset,
) -> Iterator[dict[str, Any]]:
    """Yield robust-test loaders backed by independent views of ``base_dataset``.

    The clean test dataset has already paid for CSV/PT discovery, partition
    isolation checks, and manifest-vocabulary provenance validation. A shallow
    copy preserves that validated, read-only sample index while keeping each
    view's perturbation selector independent. Loaders are yielded lazily so
    persistent evaluation workers from many scenarios are not retained at once.
    """
    if not isinstance(base_dataset, RobustTriModalDataset):
        raise TypeError(
            "Robust-test views require the validated RobustTriModalDataset "
            "used for clean test evaluation"
        )
    if bool(base_dataset.is_train):
        raise ValueError("Robust-test views cannot be built from a train dataset")

    eval_cfg = cfg.get("eval", {}) or {}
    perturb_tests, perturb_strengths = _normalize_robust_test_protocol(eval_cfg)

    for perturb in perturb_tests:
        if perturb == "clean":
            yield {
                "result_key": "clean",
                "perturb_type": "clean",
                "strength": 0.0,
                "loader": None,
            }
            continue
        # Missing transforms ignore requested strength and are evaluated once.
        is_strength_invariant = perturb.endswith("_missing")
        strengths = [1.0] if is_strength_invariant else perturb_strengths
        for strength in strengths:
            result_key = (
                perturb if len(strengths) == 1 else f"{perturb}_s{strength:g}"
            )
            dataset = _build_eval_perturbation_view(
                base_dataset,
                perturb_type=perturb,
                perturb_strength=strength,
            )
            yield {
                "result_key": result_key,
                "perturb_type": perturb,
                "strength": float(strength),
                "loader": build_loader(cfg, dataset, is_train=False),
            }


RELIABILITY_CALIBRATION_PERTURBATIONS = {
    # I1 sees clean and branch-local partial-degradation outputs. The mechanism
    # name and requested strength only balance objective rows; its input is the
    # transformed branch opinion itself.
    "api_event_dropout": ("api",),
    "graph_sparsify": ("graph",),
    "manifest_permission_mask": ("manifest",),
}
RELIABILITY_CALIBRATION_MISSING = (
    "api_missing",
    "graph_missing",
    "manifest_missing",
)

ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION = {
    "api_event_dropout": "partial_degradation",
    "graph_sparsify": "partial_degradation",
    "manifest_permission_mask": "partial_degradation",
    "api_missing": "missing",
    "graph_missing": "missing",
    "manifest_missing": "missing",
}
ROUTING_ROBUSTNESS_FAMILIES = (
    "partial_degradation",
    "missing",
)


class _TaggedCalibrationScenarioDataset(Dataset):
    """Attach an immutable scenario id without mutating a shared eval view."""

    def __init__(self, dataset, source_index: int):
        self.dataset = dataset
        self.source_index = int(source_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.source_index, self.dataset[index]


class _ScenarioBoundaryBatchSampler:
    """Yield sequential batches that never cross a ConcatDataset boundary."""

    def __init__(self, source_lengths: list[int], batch_size: int):
        self.source_lengths = tuple(int(value) for value in source_lengths)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Calibration batch size must be positive")
        if any(value < 0 for value in self.source_lengths):
            raise ValueError("Calibration source lengths must be non-negative")

    def __iter__(self):
        offset = 0
        for source_length in self.source_lengths:
            source_end = offset + source_length
            for start in range(offset, source_end, self.batch_size):
                yield list(range(start, min(start + self.batch_size, source_end)))
            offset = source_end

    def __len__(self) -> int:
        return sum(
            (source_length + self.batch_size - 1) // self.batch_size
            for source_length in self.source_lengths
        )


def _robust_calibration_collate_fn(tagged_items):
    if not tagged_items:
        raise RuntimeError("Calibration collate received an empty batch")
    source_indices = {int(source_index) for source_index, _item in tagged_items}
    if len(source_indices) != 1:
        raise RuntimeError(
            "Calibration batch crossed a scenario boundary: "
            f"{sorted(source_indices)}"
        )
    batch = robust_collate_fn([item for _source_index, item in tagged_items])
    if not isinstance(batch, dict):
        raise RuntimeError("Calibration collate must return a mapping")
    batch["calibration_source_index"] = next(iter(source_indices))
    return batch

# Training logs only consume this small subset. Keeping the other diagnostics
# on the evaluation path avoids synchronizing dozens of GPU tensors per batch.
TRAIN_LOG_DIAGNOSTIC_KEYS = (
    "fusion_weight_api",
    "fusion_weight_graph",
    "fusion_weight_manifest",
    "uncertainty_proxy_api",
    "uncertainty_proxy_graph",
    "uncertainty_proxy_manifest",
    "routing_prefit_uniform_prior_active",
)
TRAIN_LOG_LOSS_PART_KEYS = (
    "loss",
    "ce",
    "branch_aux",
    "branch_aux_weight",
    "evidential_loss",
    "evidential_loss_weight",
)


def reliability_calibration_scenarios(cfg: dict) -> list[dict[str, Any]]:
    """Build transformed post-hoc views used by the global opinion router.

    These views are built only from the post-hoc calibration subset. They never
    touch checkpoint selection or the disjoint decision-calibration subset.
    ``reliability_branches`` declares the single branch changed by each
    controlled partial-degradation view. These views provide correctness
    supervision only: perturbation name, strength, pre-transform count,
    integrity/coverage metadata, and other-modality signals are never exposed
    as I1 inputs.
    """
    calibration_cfg = cfg.get("calibration", {}) or {}
    fusion_cfg = cfg.get("fusion", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    if "probability_calibration" in fusion_cfg:
        raise ValueError(
            "fusion.probability_calibration was removed; use the I1 "
            "temperature-scaling confidence comparator or "
            "fusion.routing.final_temperature_scaling as appropriate"
        )
    routed_posthoc = bool(
        str(fusion_cfg.get("combination", "")).strip().lower() == "routed"
        and routing_cfg.get("enabled", False)
        and routing_cfg.get("posthoc_refine", True)
    )
    route_distribution_active = bool(
        routed_posthoc
        and str(routing_cfg.get("mode", "learned")).strip().lower() == "learned"
        and float(routing_cfg.get("prediction_loss_weight", 1.0)) > 0.0
    )
    risk_fit_active = bool(
        routed_posthoc
        and str(routing_cfg.get("risk_mode", "learned")).strip().lower()
        == "learned"
        and float(routing_cfg.get("risk_loss_weight", 1.0)) > 0.0
    )
    i1_fit_active = bool(reliability_cfg.get("enabled", False))
    if "include_pairwise_completeness_views" in calibration_cfg:
        raise ValueError(
            "calibration.include_pairwise_completeness_views was removed; "
            "declare every post-hoc mechanism explicitly in "
            "calibration.fit_perturbations"
        )
    any_posthoc_fit = bool(
        i1_fit_active or route_distribution_active or risk_fit_active
    )
    if not any_posthoc_fit:
        return []

    raw_perturbations = calibration_cfg.get("fit_perturbations")
    if not isinstance(raw_perturbations, (list, tuple)) or not raw_perturbations:
        raise ValueError(
            "calibration.fit_perturbations must be a non-empty sequence of "
            "explicit graded post-hoc mechanisms"
        )
    fit_perturbations = [str(value).strip().lower() for value in raw_perturbations]
    if any(not value for value in fit_perturbations):
        raise ValueError("calibration.fit_perturbations contains an empty name")
    duplicate_perturbations = sorted(
        {
            value
            for value in fit_perturbations
            if fit_perturbations.count(value) > 1
        }
    )
    if duplicate_perturbations:
        raise ValueError(
            "calibration.fit_perturbations contains duplicates: "
            f"{duplicate_perturbations}"
        )
    reserved = sorted(
        set(fit_perturbations)
        & ({"clean"} | set(RELIABILITY_CALIBRATION_MISSING))
    )
    if reserved:
        raise ValueError(
            "clean is supplied separately and *_missing views are added once "
            "automatically; remove reserved calibration.fit_perturbations: "
            f"{reserved}"
        )
    unsupported = sorted(
        set(fit_perturbations) - set(RELIABILITY_CALIBRATION_PERTURBATIONS)
    )
    if unsupported:
        raise ValueError(
            "calibration.fit_perturbations contains unsupported mechanisms: "
            f"{unsupported}"
        )

    raw_strengths = calibration_cfg.get("perturb_strengths")
    if not isinstance(raw_strengths, (list, tuple)) or not raw_strengths:
        raise ValueError(
            "calibration.perturb_strengths must be a non-empty sequence"
        )
    strengths: list[float] = []
    for raw_strength in raw_strengths:
        if isinstance(raw_strength, bool):
            raise ValueError(
                "calibration.perturb_strengths must contain finite numbers, not booleans"
            )
        try:
            strength = float(raw_strength)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "calibration.perturb_strengths must contain finite numbers"
            ) from exc
        if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
            raise ValueError(
                "calibration.perturb_strengths values must lie within (0, 1]"
            )
        strengths.append(strength)
    duplicate_strengths = sorted(
        {value for value in strengths if strengths.count(value) > 1}
    )
    if duplicate_strengths:
        raise ValueError(
            "calibration.perturb_strengths contains duplicates: "
            f"{duplicate_strengths}"
        )
    strengths = sorted(strengths)

    graded = [
        {
            "name": f"calibration_{perturb_type}_s{strength:g}",
            "perturb_type": perturb_type,
            "scenario_group": perturb_type,
            "objective_family": ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION[
                perturb_type
            ],
            "strength": strength,
            "reliability_branches": list(branches),
        }
        for perturb_type in fit_perturbations
        for branches in (RELIABILITY_CALIBRATION_PERTURBATIONS[perturb_type],)
        if (
            (i1_fit_active and bool(branches))
            or route_distribution_active
            or risk_fit_active
        )
        for strength in strengths
    ]
    missing = [
        {
            "name": f"calibration_{perturb_type}",
            "perturb_type": perturb_type,
            "scenario_group": "missing",
            "objective_family": "missing",
            "strength": 1.0,
            # Missing views train I2 route/risk robustness. The absent branch
            # has no I1 correctness target, while the surviving branch
            # logits are unchanged and must not be counted as extra clean
            # reliability observations.
            "reliability_branches": [],
        }
        for perturb_type in RELIABILITY_CALIBRATION_MISSING
    ] if (route_distribution_active or risk_fit_active) else []
    return [*graded, *missing]


def build_reliability_calibration_loaders(
    cfg: dict,
    base_dataset: RobustTriModalDataset,
    subset_indices: list[int] | None,
) -> list[dict[str, Any]]:
    """Build deterministic degraded views backed by one evaluation worker pool."""
    scenarios = reliability_calibration_scenarios(cfg)
    if not scenarios:
        raise ValueError(
            "No transformed post-hoc views are consumed by the active I1/I2 stages"
        )
    tagged_datasets = []
    source_lengths = []
    for source_index, item in enumerate(scenarios):
        dataset = _build_eval_perturbation_view(
            base_dataset,
            perturb_type=item["perturb_type"],
            perturb_strength=item["strength"],
        )
        if subset_indices is not None:
            dataset = Subset(dataset, subset_indices)
        tagged_datasets.append(
            _TaggedCalibrationScenarioDataset(dataset, source_index)
        )
        source_lengths.append(len(dataset))

    combined_dataset = ConcatDataset(tagged_datasets)
    eval_batch_size = int(
        cfg["train"].get(
            "eval_batch_size",
            cfg["train"].get("batch_size", 32),
        )
    )
    batch_sampler = _ScenarioBoundaryBatchSampler(
        source_lengths,
        eval_batch_size,
    )
    # The loader is iterated once by fit_posthoc_calibration. Each returned
    # source descriptor retains its own scientific metadata but points to the
    # same loader and immutable local source id.
    loader = build_loader(
        cfg,
        combined_dataset,
        is_train=False,
        persistent_workers_override=False,
        batch_sampler_override=batch_sampler,
        collate_fn_override=_robust_calibration_collate_fn,
    )
    return [
        {**item, "loader": loader, "combined_source_index": source_index}
        for source_index, item in enumerate(scenarios)
    ]


def uses_routing_calibration_scenarios(cfg: dict) -> bool:
    """Return whether any post-hoc stage consumes transformed modality views."""
    return bool(
        (cfg.get("calibration", {}) or {}).get("enabled", False)
        and reliability_calibration_scenarios(cfg)
    )


def build_model(cfg: dict, feature_dim: int) -> TriModalRobustModel:
    cfg = _reject_removed_config_keys(cfg)
    removed_sections = [
        key for key in ("semantic_cross_attention", "semantic_reconstruction") if key in cfg
    ]
    if removed_sections:
        raise ValueError(
            "The formal lean pipeline no longer accepts semantic interaction/reconstruction "
            f"config sections: {removed_sections}. Remove these legacy sections from the YAML."
        )
    model_cfg = cfg.get("model", {})
    num_classes = strict_finite_integer(
        model_cfg.get("num_classes", 2), field_name="model.num_classes"
    )
    if num_classes != 2:
        raise ValueError(
            "The current tri-modal pipeline is binary-only: model.num_classes "
            "must be 2 because classification thresholding, malware-FN risk, "
            "and I3 all use the benign/malware "
            "contract explicitly."
        )
    api_cfg = model_cfg.get("api_encoder", {})
    graph_cfg = model_cfg.get("graph_encoder", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    gate_cfg = model_cfg.get("gate", {})
    graph_node_budget = _resolve_graph_node_budget(model_cfg)
    fusion_cfg = cfg.get("fusion", {}) or {}
    removed_joint_config: list[str] = []
    if "joint_emb_dim" in model_cfg:
        removed_joint_config.append("model.joint_emb_dim")
    if "linear_use_joint_branch" in fusion_cfg:
        removed_joint_config.append("fusion.linear_use_joint_branch")
    branch_aux_weights = ((cfg.get("loss", {}) or {}).get("branch_aux_weights", {}) or {})
    if isinstance(branch_aux_weights, dict) and "joint" in branch_aux_weights:
        removed_joint_config.append("loss.branch_aux_weights.joint")
    if removed_joint_config:
        raise ValueError(
            "The Joint branch and its learned four-branch gate were removed; "
            "delete these obsolete settings: " + ", ".join(removed_joint_config)
        )
    fusion_mode = str(model_cfg.get("fusion_mode", "discount_probability"))
    configured_fusion_mode = str(fusion_cfg.get("mode", "")).lower()
    if configured_fusion_mode == "discount_probability":
        fusion_mode = "discount_probability"
    elif configured_fusion_mode not in {"", "model_dispatch"}:
        raise ValueError(f"Unsupported fusion.mode: {configured_fusion_mode}")
    removed_gate_keys = sorted(
        set(gate_cfg)
        & {
            "apply_alive_mask",
            "use_consistency_evidence",
            "use_conflict_evidence",
            "use_perturbation_evidence",
        }
    )
    if removed_gate_keys:
        raise ValueError(
            "Removed model.gate input switches are unsupported; fusion "
            "handlers now have fixed, explicit inputs: "
            f"{removed_gate_keys}"
        )

    # Input-duplication guardrail: Graph may be structurally selected around API
    # methods, but fine-grained API behavior hints must not be copied directly
    # into graph node features when the branches are treated as distinct views.
    combination = str(fusion_cfg.get("combination", "")).lower()
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    removed_fusion_keys = sorted(
        set(fusion_cfg)
        & {
            "acceptance_aggregation",
            "branch_competence_prior",
            "confidence_proxy",
            "conflict_factor",
            "detach_confidence_proxy",
            "detach_discount",
            "fallback",
            "reliability_discount_exponent",
            "support_factor",
            "use_confidence_proxy",
            "use_conflict_discount",
            "use_reliability_acceptance",
            "use_support_discount",
            "visible_integrity_modifier",
            "weight_sharpening_gamma",
        }
    )
    removed_reliability_keys = sorted(
        set(reliability_cfg)
        & {
            "use_model_visibility",
            "use_predicted_class_feature",
            "degradation_conditioning",
            "degradation_min_rows_per_predicted_class",
            "degradation_require_both_correctness_outcomes",
            "objective_weights",
            "require_all_objective_families",
            "feature_schema",
            "missing_relation_support",
            "use_relation_evidence",
            "use_edl_certainty_feature",
            "use_evidential_uncertainty",
            "group_mean_alignment",
            "apply_alive_mask",
            "weight",
        }
    )
    if removed_fusion_keys or removed_reliability_keys:
        raise ValueError(
            "Removed fusion/I1 configuration keys are unsupported; delete them "
            "instead of setting them to false: "
            f"fusion={removed_fusion_keys}, reliability_calibration="
            f"{removed_reliability_keys}"
        )
    normalize_reliability_calibration_method(
        reliability_cfg.get("method", MONOTONIC_CORRECTNESS_METHOD)
    )
    if combination == "routed" and bool(routing_cfg.get("enabled", False)):
        learned_route = str(routing_cfg.get("mode", "learned")).lower() == "learned"
        learned_risk = str(routing_cfg.get("risk_mode", "learned")).lower() == "learned"
        if (learned_route or learned_risk) and not bool(
            (cfg.get("calibration", {}) or {}).get("enabled", False)
        ):
            raise ValueError(
                "learned I2 route/risk components require calibration.enabled=true"
            )
        risk_mode = str(routing_cfg.get("risk_mode", "learned")).lower()
        if risk_mode == "learned" and "risk_target" not in routing_cfg:
            raise ValueError(
                "learned routed risk requires an explicit "
                "fusion.routing.risk_target; no implicit legacy target is "
                "accepted"
            )
        risk_target = str(
            routing_cfg.get(
                "risk_target", "threshold_malware_false_negative"
            )
        ).strip().lower()
        if risk_target != "threshold_malware_false_negative":
            raise ValueError(
                "The final routed protocol supports only "
                "fusion.routing.risk_target="
                "'threshold_malware_false_negative'."
            )
        weights = routing_cfg.get("scenario_objective_weights", {}) or {}
        if not isinstance(weights, dict):
            raise ValueError(
                "fusion.routing.scenario_objective_weights must be a mapping"
            )
        unknown_weights = set(weights) - {"clean", "perturb"}
        if unknown_weights:
            raise ValueError(
                "Unsupported fusion.routing.scenario_objective_weights keys: "
                f"{sorted(unknown_weights)}"
            )
        clean_weight = float(weights.get("clean", 0.5))
        perturb_weight = float(weights.get("perturb", 0.5))
        if (
            not math.isfinite(clean_weight)
            or not math.isfinite(perturb_weight)
            or clean_weight < 0.0
            or perturb_weight < 0.0
            or clean_weight + perturb_weight <= 0.0
        ):
            raise ValueError(
                "fusion.routing.scenario_objective_weights must contain "
                "finite non-negative clean/perturb mass with positive sum"
            )
        removed_router_objectives = sorted(
            set(routing_cfg)
            & {
                "route_oracle_loss_weight",
                "route_oracle_temperature",
                "subset_oracle_loss_weight",
                "subset_oracle_temperature",
                "group_robust_objective",
                "acceptance_score_mode",
                "train_end_to_end",
                "calibration_weight",
            }
        )
        if removed_router_objectives:
            raise ValueError(
                "Source-oracle and soft-worst routing objectives were removed "
                "from the final method; remove these keys: "
                f"{removed_router_objectives}"
            )
        risk_support_cfg = routing_cfg.get("risk_support", {}) or {}
        if not isinstance(risk_support_cfg, dict):
            raise ValueError("fusion.routing.risk_support must be a mapping")
        unknown_support = set(risk_support_cfg) - {
            "enabled",
            "min_positive_events",
            "min_negative_events",
            "min_positive_groups",
        }
        if unknown_support:
            raise ValueError(
                "Unsupported fusion.routing.risk_support keys: "
                f"{sorted(unknown_support)}"
            )
        for key in (
            "min_positive_events",
            "min_negative_events",
            "min_positive_groups",
        ):
            value = int(risk_support_cfg.get(key, 1))
            if value < 1:
                raise ValueError(f"fusion.routing.risk_support.{key} must be positive")
        if learned_risk and risk_target.startswith("threshold_"):
            cross_fit_cfg = (
                (cfg.get("calibration", {}) or {}).get("cross_fitting", {})
                or {}
            )
            if not bool(cross_fit_cfg.get("enabled", False)):
                raise ValueError(
                    f"fusion.routing.risk_target={risk_target!r} requires "
                    "calibration.cross_fitting.enabled=true"
                )
            selective_cfg = cfg.get("selective_prediction", {}) or {}
            if (
                bool(selective_cfg.get("enabled", False))
                and str(selective_cfg.get("mode", "")).lower()
                == "risk_control"
                and str(
                    selective_cfg.get(
                        "risk_target", "accepted_fn_risk_among_malware"
                    )
                ).lower()
                != "accepted_fn_risk_among_malware"
            ):
                raise ValueError(
                    "threshold-aligned routed risk requires I3 "
                    "risk_target='accepted_fn_risk_among_malware'"
                )
    if combination in {
        "routed",
        "dempster",
        "cumulative",
        "log_pool",
        "ecml",
        "conflict_weighted_opinion",
    }:
        coupling = []
        if bool(graph_cfg.get("use_behavior_hint", False)):
            coupling.append("model.graph_encoder.use_behavior_hint")
        if coupling:
            raise ValueError(
                f"fusion.combination={combination} requires non-duplicated branch inputs, "
                f"but these settings copy cross-modal hints into a branch: {coupling}. "
                "Disable the cross-modal hint or use a separately declared "
                "comparison model."
            )

    loss_cfg = cfg.get("loss", {}) or {}
    training_objective = str(loss_cfg.get("objective", "standard")).strip().lower()
    expected_combination = {"tmc": "dempster", "ecml": "ecml"}.get(
        training_objective
    )
    if expected_combination is not None:
        violations: list[str] = []
        opinion_source = str(
            fusion_cfg.get("opinion_source", "evidential")
        ).strip().lower()
        if fusion_mode != "discount_probability":
            violations.append("model/fusion mode must be discount_probability")
        if combination != expected_combination:
            violations.append(
                f"fusion.combination must be {expected_combination!r}"
            )
        if opinion_source == "softmax_fixed_uncertainty":
            violations.append(
                "fusion.opinion_source='softmax_fixed_uncertainty' is "
                f"incompatible with loss.objective={training_objective!r}; "
                "TMC/ECML require differentiable Dirichlet evidence"
            )
        elif opinion_source != "evidential":
            violations.append("fusion.opinion_source must be 'evidential'")
        if bool((fusion_cfg.get("routing", {}) or {}).get("enabled", False)):
            violations.append("fusion.routing.enabled must be false")
        if bool(fusion_cfg.get("use_i1_reliability", False)):
            violations.append("fusion.use_i1_reliability must be false")
        if not bool(fusion_cfg.get("use_hard_alive_mask", True)):
            violations.append("fusion.use_hard_alive_mask must be true")
        if bool((fusion_cfg.get("reliability_calibration", {}) or {}).get("enabled", False)):
            violations.append("fusion.reliability_calibration.enabled must be false")
        if bool((cfg.get("calibration", {}) or {}).get("enabled", False)):
            violations.append("calibration.enabled must be false")
        if bool((cfg.get("selective_prediction", {}) or {}).get("enabled", False)):
            violations.append("selective_prediction.enabled must be false")
        if violations:
            raise ValueError(
                f"loss.objective={training_objective} requires an isolated "
                "style-adapted fusion path: " + "; ".join(violations)
            )

    method_protocol_id = str(
        (cfg.get("method", {}) or {}).get("protocol_id", "")
    ).strip().lower()
    if method_protocol_id == "qmf_energy_v1":
        qmf_violations: list[str] = []
        if fusion_mode != "tri_modal_quality_fusion":
            qmf_violations.append(
                "model.fusion_mode must be 'tri_modal_quality_fusion'"
            )
        quality_temperature = float(
            model_cfg.get("quality_fusion_temperature", 10.0)
        )
        if not math.isfinite(quality_temperature) or quality_temperature != 10.0:
            qmf_violations.append(
                "model.quality_fusion_temperature must be 10.0"
            )
        if training_objective != "standard":
            qmf_violations.append("loss.objective must be 'standard'")
        if bool((cfg.get("calibration", {}) or {}).get("enabled", False)):
            qmf_violations.append("calibration.enabled must be false")
        if bool((cfg.get("selective_prediction", {}) or {}).get("enabled", False)):
            qmf_violations.append("selective_prediction.enabled must be false")
        if qmf_violations:
            raise ValueError(
                "method.protocol_id=qmf_energy_v1 identifies the isolated "
                "QMF-Energy component baseline: " + "; ".join(qmf_violations)
            )

    return TriModalRobustModel(
        in_feat_dim=feature_dim,
        num_classes=num_classes,
        fusion_mode=fusion_mode,
        api_num_hash_buckets=int(api_cfg.get("num_hash_buckets", 8192)),
        api_type_vocab_size=int(api_cfg.get("type_vocab_size", 16)),
        api_emb_dim=int(api_cfg.get("emb_dim", 128)),
        api_hidden_dim=int(api_cfg.get("hidden_dim", 256)),
        api_dropout=float(api_cfg.get("dropout", 0.15)),
        api_encoder_type=str(api_cfg.get("type", "transformer")),
        api_layers=int(api_cfg.get("layers", 2)),
        api_heads=int(api_cfg.get("heads", 4)),
        api_max_seq_len=strict_finite_integer(
            api_cfg.get("max_seq_len", 1024),
            field_name="model.api_encoder.max_seq_len",
        ),
        graph_emb_dim=int(graph_cfg.get("emb_dim", 128)),
        graph_hidden=int(graph_cfg.get("hidden", 128)),
        graph_heads=int(graph_cfg.get("heads", 4)),
        graph_layers=int(graph_cfg.get("layers", 2)),
        graph_encoder_type=str(graph_cfg.get("type", "gatv2")),
        max_nodes_gnn=graph_node_budget,
        use_graph_behavior_hint=bool(graph_cfg.get("use_behavior_hint", False)),
        manifest_in_dim=int(manifest_cfg.get("in_dim", 256)),
        manifest_emb_dim=int(manifest_cfg.get("emb_dim", 128)),
        manifest_hidden_dim=int(manifest_cfg.get("hidden_dim", 256)),
        manifest_dropout=float(manifest_cfg.get("dropout", 0.1)),
        quality_fusion_temperature=float(model_cfg.get("quality_fusion_temperature", 10.0)),
        gate_hidden_dim=int(gate_cfg.get("hidden_dim", 128)),
        gate_detach=bool(gate_cfg.get("detach", True)),
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
    macro_f1 = float(
        f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)
    )
    f1_pos = float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0))
    macro_recall = float(
        recall_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)
    )
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


def _row_selective_eligible(row: dict[str, Any]) -> bool:
    """Read the mandatory hard-availability gate used by selective rules."""

    if "selective_eligible" not in row:
        raise ValueError(
            "selective evaluation row is missing mandatory selective_eligible"
        )
    raw = row["selective_eligible"]
    if isinstance(raw, bool):
        return raw
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("selective_eligible must be boolean or binary") from exc
    if not math.isfinite(value) or value not in {0.0, 1.0}:
        raise ValueError("selective_eligible must be boolean or binary")
    return bool(value)


def compute_branch_reliability_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit I1 correctness and the explicitly configured I2 risk event."""
    out: dict[str, float | int] = {}
    for branch in BRANCH_EVAL_LOGIT_KEYS:
        reliability_values: list[float] = []
        correctness_values: list[float] = []
        predicted_class_values: list[int] = []
        for row in rows:
            alive = _finite_row_float(row, f"{branch}_alive")
            if alive is not None and alive < 0.5:
                continue
            reliability = _finite_row_float(row, f"predicted_reliability_{branch}")
            correctness = _finite_row_float(row, f"{branch}_correct")
            predicted_class = _finite_row_float(row, f"{branch}_pred")
            if reliability is None or correctness is None:
                continue
            reliability_values.append(min(1.0, max(0.0, reliability)))
            correctness_values.append(1.0 if correctness >= 0.5 else 0.0)
            predicted_class_values.append(
                int(predicted_class)
                if predicted_class in {0.0, 1.0}
                else -1
            )
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

        # Formal I1 may include a signed predicted-class intercept. Report
        # calibration in both predicted-class cells so a pooled score
        # cannot hide a fragile predicted-benign (FN-relevant) region.
        predicted_arr = np.asarray(predicted_class_values, dtype=np.int64)
        for class_index, class_name in (
            (0, "predicted_benign"),
            (1, "predicted_malware"),
        ):
            mask = predicted_arr == class_index
            subgroup_count = int(mask.sum())
            prefix = f"{branch}_{class_name}_reliability"
            out[f"{prefix}_count"] = subgroup_count
            if subgroup_count == 0:
                continue
            subgroup_reliability = reliability_arr[mask]
            subgroup_correctness = correctness_arr[mask]
            subgroup_mean = float(subgroup_reliability.mean())
            subgroup_accuracy = float(subgroup_correctness.mean())
            out[f"{prefix}_brier"] = float(
                np.mean((subgroup_reliability - subgroup_correctness) ** 2)
            )
            out[f"{prefix}_ece_10"] = _calibration_ece(
                subgroup_reliability,
                subgroup_correctness,
                bins=10,
            )
            out[f"{prefix}_mean"] = subgroup_mean
            out[f"{prefix}_branch_accuracy"] = subgroup_accuracy
            out[f"{prefix}_accuracy_gap"] = subgroup_mean - subgroup_accuracy

    risk_values: list[float] = []
    risk_target_values: list[float] = []
    mixture_error_values: list[float] = []
    threshold_aligned_count = 0
    all_missing_count = 0
    all_missing_forced_rejection_count = 0
    for row in rows:
        routing_active = _finite_row_float(row, "routing_active")
        if routing_active is None or routing_active < 0.5:
            continue
        has_available = _finite_row_float(row, "routing_has_available")
        if has_available is None:
            raise ValueError("routed evaluation row is missing routing_has_available")
        if has_available < 0.5:
            all_missing_count += 1
            risk = _finite_row_float(row, "routing_risk_probability")
            if risk is not None and risk >= 1.0 - 1.0e-6:
                all_missing_forced_rejection_count += 1
            continue
        risk = _finite_row_float(row, "routing_risk_probability")
        mixture_pred = _finite_row_float(row, "routing_mixture_pred")
        label = _finite_row_float(row, "label")
        if risk is None or mixture_pred is None or label is None:
            continue
        threshold_active = _finite_row_float(
            row, "routing_risk_decision_threshold_active"
        )
        final_pred = _finite_row_float(row, "pred")
        target_active = _finite_row_float(
            row, "routing_risk_target_threshold_malware_false_negative"
        )
        if target_active is None or target_active < 0.5:
            raise ValueError(
                "Routed evaluation row does not declare the final "
                "threshold-aligned malware-FN risk target"
            )
        risk_values.append(min(1.0, max(0.0, risk)))
        mixture_error = float(
            int(round(mixture_pred)) != int(round(label))
        )
        mixture_error_values.append(mixture_error)
        if (
            threshold_active is None
            or threshold_active < 0.5
            or final_pred is None
        ):
            raise ValueError(
                "Threshold-aligned malware-FN risk requires an active "
                "decision threshold and final predictions in evaluation rows"
            )
        threshold_aligned_count += 1
        risk_target_values.append(
            float(
                int(round(label)) == 1
                and int(round(final_pred)) == 0
            )
        )
    if risk_values:
        risk_arr = np.asarray(risk_values, dtype=np.float64)
        mixture_error_arr = np.asarray(
            mixture_error_values, dtype=np.float64
        )
        out["routing_risk_count"] = int(risk_arr.size)
        out["routing_risk_target"] = "threshold_malware_false_negative"
        out["routing_risk_mean"] = float(risk_arr.mean())
        out["routing_mixture_error_rate"] = float(
            mixture_error_arr.mean()
        )
        out["routing_risk_threshold_aligned_count"] = int(
            threshold_aligned_count
        )
        target_arr = np.asarray(risk_target_values, dtype=np.float64)
        if target_arr.size != risk_arr.size:
            raise RuntimeError(
                "Risk score and target counts disagree during calibration audit"
            )
        out["routing_risk_calibration_defined"] = 1
        out["routing_risk_brier"] = float(
            np.mean((risk_arr - target_arr) ** 2)
        )
        out["routing_risk_ece_10"] = _calibration_ece(
            risk_arr, target_arr, bins=10
        )
        out["routing_risk_target_event_rate"] = float(target_arr.mean())
        out["routing_risk_error_rate_gap"] = float(
            risk_arr.mean() - target_arr.mean()
        )
        if len(set(risk_target_values)) > 1:
            out["routing_risk_auc_defined"] = 1
            out["routing_risk_ap_defined"] = 1
            out["routing_risk_auc"] = float(
                roc_auc_score(target_arr, risk_arr)
            )
            out["routing_risk_ap"] = float(
                average_precision_score(target_arr, risk_arr)
            )
        else:
            out["routing_risk_auc_defined"] = 0
            out["routing_risk_ap_defined"] = 0
            out["routing_risk_auc"] = float("nan")
            out["routing_risk_ap"] = float("nan")
    out["routing_all_missing_count"] = int(all_missing_count)
    out["routing_all_missing_forced_rejection_count"] = int(
        all_missing_forced_rejection_count
    )
    return out

def _validated_selective_arrays(
    labels: list[int],
    preds: list[int],
    acceptance_scores: list[float],
    selective_eligibility: list[bool] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the binary selective-evaluation contract without imputation."""

    if len(preds) != len(labels) or len(acceptance_scores) != len(labels):
        raise ValueError("selective labels, predictions, and scores disagree in length")
    if selective_eligibility is not None and len(selective_eligibility) != len(labels):
        raise ValueError("selective eligibility and labels disagree in length")
    try:
        y_raw = np.asarray(labels, dtype=np.float64)
        pred_raw = np.asarray(preds, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("selective labels and predictions must be binary") from exc
    score = np.asarray(acceptance_scores, dtype=np.float64)
    if (
        not np.isfinite(y_raw).all()
        or not np.isfinite(pred_raw).all()
        or not np.isin(y_raw, [0.0, 1.0]).all()
        or not np.isin(pred_raw, [0.0, 1.0]).all()
    ):
        raise ValueError("selective labels and predictions must be binary")
    y = y_raw.astype(np.int64)
    pred = pred_raw.astype(np.int64)
    if not np.isfinite(score).all() or not ((score >= 0.0) & (score <= 1.0)).all():
        raise ValueError("selective acceptance scores must be finite within [0, 1]")
    eligible = (
        np.ones(len(labels), dtype=bool)
        if selective_eligibility is None
        else np.asarray(selective_eligibility, dtype=bool)
    )
    return y, pred, score, eligible


def _selective_metrics(
    labels: list[int],
    preds: list[int],
    acceptance_scores: list[float],
    threshold: float,
    selective_eligibility: list[bool] | None = None,
) -> dict[str, Any]:
    if not labels:
        if preds or acceptance_scores or selective_eligibility:
            raise ValueError("selective arrays disagree in length")
        return {}
    if not math.isfinite(float(threshold)):
        raise ValueError("selective threshold must be finite")
    y, pred, score, eligible = _validated_selective_arrays(
        labels,
        preds,
        acceptance_scores,
        selective_eligibility,
    )
    accepted = eligible & (score > float(threshold))
    coverage = float(accepted.mean())
    errors = (pred != y).astype(np.float64)
    out = {
        "rejection_threshold": float(threshold),
        "acceptance_comparison": ACCEPTANCE_THRESHOLD_COMPARISON,
        "coverage": coverage,
        "rejection_rate": float(1.0 - coverage),
        "num_accepted": int(accepted.sum()),
        "num_rejected": int((~accepted).sum()),
        "num_ineligible_forced_reject": int((~eligible).sum()),
        "selective_eligible_rate": float(eligible.mean()),
    }
    out.update(
        _selective_ranking_metrics(
            labels,
            preds,
            acceptance_scores,
            selective_eligibility=selective_eligibility,
        )
    )
    if accepted.any():
        out.update(
            {
                "selective_metrics_defined": True,
                "selective_risk": float(errors[accepted].mean()),
                "selective_acc": float(1.0 - errors[accepted].mean()),
                "selective_macro_f1": float(
                    f1_score(
                        y[accepted],
                        pred[accepted],
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
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
    selective_eligibility: list[bool] | None = None,
) -> dict[str, Any]:
    """Report ranking quality over the actually selectable population.

    Availability is a hard gate, so ineligible samples cannot appear in any
    achievable accepted prefix. ``aurc`` is therefore normalized over eligible
    prefixes; ``selective_max_achievable_coverage`` reports their fraction of
    the full evaluation set.
    """
    if not labels:
        if preds or acceptance_scores or selective_eligibility:
            raise ValueError("selective arrays disagree in length")
        return {}
    y, pred, score, eligible = _validated_selective_arrays(
        labels,
        preds,
        acceptance_scores,
        selective_eligibility,
    )
    eligible_count = int(eligible.sum())
    out: dict[str, Any] = {
        "selective_ranking_num_eligible": eligible_count,
        "selective_max_achievable_coverage": float(eligible_count / len(labels)),
        "malware_fn_risk_aurc_target": "accepted_fn_risk_among_malware",
        "malware_fn_risk_aurc_denominator": "all_malware",
        "malware_fn_risk_aurc_coverage_normalization": "eligible_population",
    }
    if eligible_count == 0:
        out.update(
            {
                "aurc_defined": False,
                "aurc": None,
                "malware_fn_risk_aurc_defined": False,
                "malware_fn_risk_aurc": None,
            }
        )
        return out
    eligible_errors = (pred[eligible] != y[eligible]).astype(np.float64)
    eligible_malware_fn = (
        (y[eligible] == 1) & (pred[eligible] == 0)
    ).astype(np.float64)
    total_malware = int((y == 1).sum())
    eligible_scores = score[eligible]
    order = np.argsort(-eligible_scores, kind="stable")
    sorted_scores = eligible_scores[order]
    sorted_errors = eligible_errors[order]
    sorted_malware_fn = eligible_malware_fn[order]
    accepted = 0
    cumulative_errors = 0.0
    cumulative_malware_fn = 0.0
    weighted_area = 0.0
    weighted_malware_fn_area = 0.0
    index = 0
    # A strict score threshold cannot split equal-score samples. Aggregate each
    # tie atomically and weight its endpoint risk by the coverage increment;
    # this makes AURC invariant to row order and aligned with achievable rules.
    while index < eligible_count:
        group_end = index + 1
        while (
            group_end < eligible_count
            and sorted_scores[group_end] == sorted_scores[index]
        ):
            group_end += 1
        group_size = group_end - index
        cumulative_errors += float(sorted_errors[index:group_end].sum())
        cumulative_malware_fn += float(
            sorted_malware_fn[index:group_end].sum()
        )
        accepted += group_size
        weighted_area += group_size * (cumulative_errors / accepted)
        if total_malware > 0:
            # Match I3 exactly: the ordinate is accepted malware false
            # negatives divided by every malware sample, while the integration
            # axis spans only score-eligible prefixes. This evaluates the event
            # that u is trained to rank without silently charging benign false
            # positives to an FN-only risk head.
            weighted_malware_fn_area += group_size * (
                cumulative_malware_fn / total_malware
            )
        index = group_end
    out.update(
        {
            "aurc_defined": True,
            "aurc": float(weighted_area / eligible_count),
            "aurc_tie_policy": "atomic_score_groups",
            "malware_fn_risk_aurc_defined": bool(total_malware > 0),
            "malware_fn_risk_aurc": (
                float(weighted_malware_fn_area / eligible_count)
                if total_malware > 0
                else None
            ),
            "malware_fn_risk_aurc_tie_policy": "atomic_score_groups",
        }
    )
    return out


def validate_posthoc_oof_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate raw binary log probabilities from the upstream OOF stack."""
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"OOF row {index} must be a mapping")
        sid = str(row.get("sid", "")).strip()
        if not sid:
            raise ValueError(f"OOF row {index} has no sample identity")
        sid_key = sid.lower()
        if sid_key in seen_ids:
            raise ValueError(f"OOF rows contain duplicate sample identity {sid!r}")
        seen_ids.add(sid_key)
        group = str(row.get("group", "")).strip()
        if not group:
            raise ValueError(f"OOF row {index} has no package-group identity")
        try:
            label = strict_binary_integer(
                row["label"], field_name=f"OOF row {index} label"
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"OOF row {index} has an invalid label") from exc
        raw_value = row.get("raw_log_prob")
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
            raise ValueError(
                f"OOF row {index} must contain two raw_log_prob values; "
                "retrain the pipeline checkpoint"
            )
        try:
            raw = (float(raw_value[0]), float(raw_value[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OOF row {index} has malformed numeric fields") from exc
        if not all(math.isfinite(value) for value in raw):
            raise ValueError(f"OOF row {index} raw_log_prob must be finite")
        raw_max = max(raw)
        log_normalizer = raw_max + math.log(
            math.exp(raw[0] - raw_max) + math.exp(raw[1] - raw_max)
        )
        if abs(log_normalizer) > 1.0e-4:
            raise ValueError(
                f"OOF row {index} raw_log_prob is not a normalized log probability"
            )
        validated.append(
            {
                "sid": sid,
                "group": group,
                "label": label,
                "raw_log_prob": [raw[0], raw[1]],
            }
        )
    return validated


def validate_posthoc_oof_rows_identity(
    rows: list[dict[str, Any]],
    data_identity: dict[str, Any] | None,
) -> None:
    """Close the loop between a checkpoint OOF payload and its fingerprint."""
    if not rows:
        return
    if not isinstance(data_identity, dict):
        raise ValueError("OOF rows require a saved decision-calibration identity")
    posthoc_identity = data_identity.get("posthoc_calibration")
    if not isinstance(posthoc_identity, dict):
        raise ValueError("OOF rows require a posthoc_calibration identity")
    expected_rows = int(posthoc_identity.get("num_rows", -1))
    actual_counts: dict[str, int] = {}
    for row in rows:
        key = str(
            strict_binary_integer(row["label"], field_name="OOF identity label")
        )
        actual_counts[key] = actual_counts.get(key, 0) + 1
    actual_counts = dict(sorted(actual_counts.items()))
    expected_counts = {
        str(key): int(value)
        for key, value in dict(
            posthoc_identity.get("class_counts") or {}
        ).items()
    }
    identities = sorted(
        (
            str(row["sid"]).strip().lower(),
            str(row["group"]).strip().lower(),
            strict_binary_integer(row["label"], field_name="OOF identity label"),
        )
        for row in rows
    )
    digest = hashlib.sha256()
    for sid, group, label in identities:
        for value in (sid, group):
            encoded_value = value.encode("utf-8")
            digest.update(len(encoded_value).to_bytes(8, byteorder="big"))
            digest.update(encoded_value)
            digest.update(b"\0")
        digest.update(str(label).encode("ascii"))
        digest.update(b"\n")
    expected_digest = str(posthoc_identity.get("row_identity_sha256", ""))
    if (
        len(rows) != expected_rows
        or actual_counts != expected_counts
        or not expected_digest
        or digest.hexdigest() != expected_digest
    ):
        raise ValueError(
            "Checkpoint OOF payload is incomplete or disagrees with its "
            "post-hoc data identity"
        )


def fit_malware_classification_threshold(
    rows: list[dict[str, Any]], config: dict | None = None
) -> dict[str, Any] | None:
    """Fit a binary decision threshold without consulting the test set.

    The shared operating point maximizes calibration-set macro-F1. Malware-FN
    control is deliberately absent here and belongs to the later, disjoint I3
    selective-risk calibration step.
    """
    config = _canonicalize_classification_threshold_config(
        {"classification_threshold": config or {}}
    )["classification_threshold"]
    if not bool(config.get("enabled", False)):
        return None
    objective = str(config.get("objective", "macro_f1")).strip().lower()

    valid = [
        (probability, label)
        for _row, probability, label in _validated_conformal_rows(
            rows, context="Classification-threshold fitting"
        )
    ]
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
    best_key: tuple[float, float, float] | None = None
    for threshold in candidates:
        predictions = (probabilities >= float(threshold)).astype(np.int64)
        malware_recall = float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        )
        macro_f1 = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(labels, predictions))
        # Exact macro-F1 ties prefer the neutral boundary and then the smaller
        # threshold. Accuracy/recall remain diagnostics, never hidden objectives.
        key = (
            macro_f1,
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "macro_f1": macro_f1,
                "malware_recall": malware_recall,
                "accuracy": accuracy,
            }
    if best is None:
        raise RuntimeError("No finite classification threshold candidate exists")

    fixed_predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "enabled": True,
        "objective": objective,
        "selection_rule": CLASSIFICATION_THRESHOLD_SELECTION_RULE,
        "constraint": "none",
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


def _stable_sigmoid_scalar(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def fit_oof_malware_classification_threshold(
    rows: list[dict[str, Any]],
    config: dict | None,
    *,
    deployment_temperature: float,
) -> dict[str, Any] | None:
    """Fit a raw-log-odds cutoff and map it through scalar temperature.

    In binary classification, ``sigmoid(z / T)`` is strictly monotone for
    every positive scalar ``T``. Selecting ``z`` on upstream-OOF predictions
    and mapping only the final cutoff therefore keeps the selected decisions
    independent of the full-data temperature fit.
    """
    config = _canonicalize_classification_threshold_config(
        {"classification_threshold": config or {}}
    )["classification_threshold"]
    if not bool(config.get("enabled", False)):
        return None
    objective = str(config.get("objective", "macro_f1")).strip().lower()
    temperature = float(deployment_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("deployment temperature must be finite and positive")

    validated = validate_posthoc_oof_rows(rows)
    if not validated:
        raise ValueError("OOF classification-threshold fitting requires rows")
    # Router inference and threshold comparison are FP32. Quantize the fitting
    # domain to that exact deployable lattice so an FP64 midpoint cannot round
    # onto a different partition when copied into the router buffer.
    scores = np.asarray(
        [
            float(row["raw_log_prob"][1])
            - float(row["raw_log_prob"][0])
            for row in validated
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            strict_binary_integer(row["label"], field_name="validated OOF label")
            for row in validated
        ],
        dtype=np.int64,
    )
    if set(int(label) for label in labels.tolist()) != {0, 1}:
        raise ValueError(
            "OOF classification-threshold fitting requires both classes"
        )
    unique_scores = np.unique(scores)
    # With the deployed ``score >= cutoff`` rule, each observed upper score
    # directly represents the partition that an FP64 midpoint used to encode.
    # One FP32 successor of max represents predict-all-benign.
    candidates = [np.float32(value) for value in unique_scores]
    candidates.append(
        np.nextafter(
            np.float32(unique_scores[-1]),
            np.float32(np.inf),
        )
    )

    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for raw_threshold_value in candidates:
        raw_threshold = float(np.float32(raw_threshold_value))
        predictions = (scores >= raw_threshold).astype(np.int64)
        malware_recall = float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        )
        macro_f1 = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(labels, predictions))
        key = (macro_f1, -abs(raw_threshold), -raw_threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "raw_log_odds_threshold": float(raw_threshold),
                "threshold": _stable_sigmoid_scalar(
                    float(raw_threshold) / temperature
                ),
                "macro_f1": macro_f1,
                "malware_recall": malware_recall,
                "accuracy": accuracy,
            }
    if best is None:
        raise RuntimeError("No finite OOF classification threshold candidate exists")

    fixed_predictions = (scores >= 0.0).astype(np.int64)
    return {
        "enabled": True,
        "objective": objective,
        "selection_rule": CLASSIFICATION_THRESHOLD_SELECTION_RULE,
        "constraint": "none",
        "calibration_split": "val_posthoc_calibration",
        "num_calibration": int(labels.size),
        "num_calibration_benign": int((labels == 0).sum()),
        "num_calibration_malware": int((labels == 1).sum()),
        "num_candidates": int(len(candidates)),
        "fixed_0_5_macro_f1": float(
            f1_score(
                labels, fixed_predictions, average="macro", zero_division=0
            )
        ),
        "fixed_0_5_malware_recall": float(
            recall_score(
                labels, fixed_predictions, pos_label=1, zero_division=0
            )
        ),
        "deployment_temperature": temperature,
        "prediction_source": "upstream_nested_oof_raw_score",
        **best,
    }


def fit_rejection_threshold(rows: list[dict[str, Any]], config: dict | None = None) -> float | None:
    """Choose an acceptance threshold on validation data for target coverage."""
    config = {} if config is None else config
    mode = _selective_prediction_mode(config)
    if not _strict_config_bool(config, "enabled", False):
        return None
    if mode in {"conformal", "risk_control"}:
        return None
    _selective_score_type(config)
    target_coverage = float(config.get("target_coverage", 0.9))
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("selective_prediction.target_coverage must be within (0, 1]")
    scores: list[float] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Threshold calibration row {row_index} must be a mapping")
        score = _finite_row_float(row, "acceptance_score")
        if score is None or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Threshold calibration row {row_index} requires an acceptance "
                "score within [0, 1]"
            )
        if _row_selective_eligible(row):
            scores.append(float(score))
    scores.sort(reverse=True)
    if not scores:
        raise ValueError(
            "selective_prediction.enabled=true produced no finite configured "
            "selective scores on the decision-calibration rows"
        )
    requested_accepted = max(1, int(math.ceil(target_coverage * len(rows))))
    if requested_accepted > len(scores):
        raise ValueError(
            "selective_prediction.target_coverage is infeasible because the "
            "hard availability gate leaves fewer eligible samples"
        )
    accepted_count = requested_accepted
    # Among hard-eligible rows, the public rule is strictly score > threshold.
    # Move the boundary one
    # representable float below the cutoff so every row tied at the cutoff is
    # accepted together, matching the previous tie-conservative behavior.
    return float(math.nextafter(scores[accepted_count - 1], float("-inf")))


def _selective_metrics_from_rows(
    rows: list[dict[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    if threshold is None:
        return {}
    valid: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Selective evaluation row {row_index} must be a mapping")
        missing = [key for key in ("label", "pred", "acceptance_score") if key not in row]
        if missing:
            raise ValueError(
                f"Selective evaluation row {row_index} is missing fields: {missing}"
            )
        valid.append(row)
    return _selective_metrics(
        [row["label"] for row in valid],
        [row["pred"] for row in valid],
        [float(row["acceptance_score"]) for row in valid],
        threshold,
        [_row_selective_eligible(row) for row in valid],
    )


def _selective_score_type(config: dict | None = None) -> str:
    config = {} if config is None else config
    mode = _selective_prediction_mode(config)
    if mode == "conformal":
        # Conformal prediction sets are fitted from class nonconformity below;
        # this score is only retained in dumped rows/ranking diagnostics.
        score_type = "msp"
    elif mode == "risk_control":
        score_type = str(config.get("threshold_score", "model_acceptance")).lower()
    else:
        score_type = str(config.get("threshold_score", "msp")).lower()
    if score_type == "max_probability":
        raise ValueError(
            "selective_prediction.threshold_score='max_probability' was removed "
            "because it ambiguously referred to deployed-class confidence. Use "
            "'deployed_class_probability' for that threshold-aware score or "
            "'msp' for strict max(p, 1-p)."
        )
    allowed = {
        "deployed_class_probability",
        "msp",
        "predictive_entropy_certainty",
        "evidential_certainty",
        "mixture_certainty",
        "model_acceptance",
    }
    if score_type not in allowed:
        raise ValueError(
            "selective_prediction.threshold_score must be one of "
            f"{sorted(allowed)}, got {score_type!r}"
        )
    return score_type


def _validate_selective_score_fusion_compatibility(
    *,
    selective_enabled: bool,
    score_type: str,
    discount_probability_mode: bool,
) -> None:
    """Fail early only when a score needs discount-fusion-only diagnostics.

    Probability-only scores (strict MSP, deployed-class probability, predictive
    entropy) and conformal prediction sets are valid for any classifier.  I3 is
    a decision layer and must not be coupled to the proposed fusion mechanism.
    The remaining score types consume tensors emitted only by
    ``DiscountProbabilityFusion`` and therefore retain an explicit guard.
    """

    if not bool(selective_enabled):
        return
    discount_only_scores = {
        "evidential_certainty",
        "mixture_certainty",
        "model_acceptance",
    }
    normalized = str(score_type).strip().lower()
    if normalized in discount_only_scores and not bool(discount_probability_mode):
        raise ValueError(
            f"selective score {normalized!r} requires discount_probability "
            "fusion diagnostics; use 'msp', 'deployed_class_probability', or "
            "'predictive_entropy_certainty' for a model-agnostic score"
        )


def _batch_selective_score(
    prob_malware: torch.Tensor,
    extra: dict[str, Any],
    score_type: str,
    classification_threshold: float = 0.5,
    predicted_malware_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a larger-is-safer score for thresholding and AURC."""
    score_type = str(score_type).lower()
    if score_type == "max_probability":
        raise ValueError(
            "selective score 'max_probability' was removed because it is "
            "ambiguous; use 'deployed_class_probability' or strict 'msp'"
        )
    probability = prob_malware.float().view(-1).clamp(0.0, 1.0)
    if score_type == "deployed_class_probability":
        predicted_malware = (
            predicted_malware_override.bool().view(-1)
            if isinstance(predicted_malware_override, torch.Tensor)
            else probability >= float(classification_threshold)
        )
        return torch.where(
            predicted_malware,
            probability,
            1.0 - probability,
        ).clamp(0.0, 1.0)
    if score_type == "msp":
        # Strict maximum softmax probability for the binary classifier. Unlike
        # deployed_class_probability, this baseline is independent of a fitted
        # deployment threshold that may differ from 0.5.
        return torch.maximum(probability, 1.0 - probability)
    if score_type == "predictive_entropy_certainty":
        # Binary predictive entropy normalised to [0, 1], then inverted so all
        # selective scores retain the common larger-is-safer convention.
        complement = 1.0 - probability
        tiny = torch.finfo(probability.dtype).tiny
        entropy = -(
            probability * torch.log(probability.clamp_min(tiny))
            + complement * torch.log(complement.clamp_min(tiny))
        )
        return (1.0 - entropy / math.log(2.0)).clamp(0.0, 1.0)
    if score_type == "evidential_certainty":
        uncertainty = extra.get("fused_uncertainty")
        if not isinstance(uncertainty, torch.Tensor):
            raise ValueError(
                "evidential_certainty threshold requires fused_uncertainty"
            )
        return (1.0 - uncertainty.float().view(-1)).clamp(0.0, 1.0)
    if score_type == "mixture_certainty":
        certainty = extra.get("acceptance_score_mixture_certainty")
        if not isinstance(certainty, torch.Tensor):
            raise ValueError(
                "mixture_certainty threshold requires "
                "acceptance_score_mixture_certainty"
            )
        return certainty.float().view(-1).clamp(0.0, 1.0)
    if score_type == "model_acceptance":
        acceptance = extra.get("acceptance_score")
        if not isinstance(acceptance, torch.Tensor):
            raise ValueError("model_acceptance threshold requires acceptance_score")
        return acceptance.float().view(-1).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported selective score type: {score_type}")


def _batch_selective_eligibility(
    extra: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return whether at least one primary modality is available per sample."""

    explicit = extra.get("selective_eligible")
    if not isinstance(explicit, torch.Tensor):
        raise ValueError(
            "model output is missing mandatory tensor selective_eligible"
        )
    flattened = explicit.detach().to(device=device).view(-1)
    if flattened.numel() != int(batch_size) or not bool(
        torch.isfinite(flattened.float()).all().item()
    ):
        raise ValueError(
            "selective_eligible must contain one finite value per sample"
        )
    if not bool(((flattened == 0) | (flattened == 1)).all().item()):
        raise ValueError("selective_eligible must contain binary values")
    return flattened.bool()


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
    if "raw_conflict" not in row:
        raise ValueError(
            "conflict-aware conformal row is missing mandatory raw_conflict"
        )
    value = _finite_row_float(row, "raw_conflict")
    if value is None or not 0.0 <= value <= 1.0:
        raise ValueError("raw_conflict must be finite within [0, 1]")
    return float(value)


def _validated_conformal_rows(
    rows: list[dict[str, Any]],
    *,
    context: str,
) -> list[tuple[dict[str, Any], float, int]]:
    validated: list[tuple[dict[str, Any], float, int]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{context} row {row_index} must be a mapping")
        probability = _finite_row_float(row, "prob_malware")
        if probability is None or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{context} row {row_index} requires prob_malware within [0, 1]"
            )
        try:
            label = strict_binary_integer(
                row["label"], field_name=f"{context} row {row_index} label"
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{context} row {row_index} has an invalid label") from exc
        validated.append((row, float(probability), label))
    return validated


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
    config = {} if config is None else config
    if not _uses_conformal_selective(config):
        return None
    target_coverage = float(config.get("target_coverage", 0.9))
    alpha = float(config.get("alpha", 1.0 - target_coverage))
    if "alpha" in config and "target_coverage" in config and not math.isclose(
        alpha,
        1.0 - target_coverage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "selective_prediction.alpha conflicts with target_coverage"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal selective_prediction.alpha must be within (0, 1)")
    class_conditional = _strict_config_bool(config, "class_conditional", True)
    use_raw_conflict = _strict_config_bool(config, "use_raw_conflict", False)
    validated = _validated_conformal_rows(
        rows, context="Conformal calibration"
    )
    if not validated:
        raise ValueError("Conformal calibration requires at least one valid row")
    scores: dict[int, list[float]] = {0: [], 1: []}
    for row, p1, label in validated:
        # Nonconformity of the true class, computed identically to the
        # prediction-set test below. Probability-only and raw-conflict-augmented
        # scores are separate conformal baselines; the main method uses the
        # disjoint malware false-negative risk-control rule below.
        nonconformity = _conformal_nonconformity(
            p1,
            label,
            _row_raw_conflict(row) if use_raw_conflict else 0.0,
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
    config = {} if config is None else config
    mode = _selective_prediction_mode(config)
    return _strict_config_bool(config, "enabled", False) and mode == "conformal"


def _conformal_prediction_set(
    p1: float,
    thresholds: dict[str, Any],
    raw_conflict: float = 0.0,
) -> tuple[bool, bool]:
    """Return (include_benign, include_malware) for the conformal prediction set."""
    q_benign = float(thresholds.get("q_benign", float("inf")))
    q_malware = float(thresholds.get("q_malware", float("inf")))
    if (
        math.isnan(q_benign)
        or math.isnan(q_malware)
        or q_benign < 0.0
        or q_malware < 0.0
    ):
        raise ValueError("Conformal thresholds must be non-negative and not NaN")
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
    alpha = float(thresholds.get("alpha", float("nan")))
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("Conformal thresholds require alpha within (0, 1)")
    valid = _validated_conformal_rows(rows, context="Conformal evaluation")
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
    ineligible_forced_rejects = 0
    use_raw_conflict = bool(thresholds.get("use_raw_conflict", False))
    for row, p1, y in valid:
        eligible = _row_selective_eligible(row)
        if eligible:
            include_benign, include_malware = _conformal_prediction_set(
                p1,
                thresholds,
                _row_raw_conflict(row) if use_raw_conflict else 0.0,
            )
        else:
            # With no observable modality, abstain by returning the full label
            # set. This remains non-singleton (hence rejected) while preserving
            # conformal coverage instead of manufacturing a false empty set.
            include_benign, include_malware = True, True
            ineligible_forced_rejects += 1
        size = int(include_benign) + int(include_malware)
        empty_sets += int(size == 0)
        ambiguous_sets += int(size == 2)
        is_accept = size == 1
        pred = 1 if (include_malware and not include_benign) else 0
        row["conformal_include_benign"] = int(include_benign)
        row["conformal_include_malware"] = int(include_malware)
        row["conformal_set_size"] = int(size)
        row["accepted"] = int(is_accept)
        row["rejected"] = int(not is_accept)
        row["selective_decision_mode"] = "conformal"
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
        "conformal_alpha": alpha,
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
        # CRC-aligned risk: accepted false negatives divided by all malware.
        "conformal_accepted_fn_risk_among_malware": _ratio(
            malware_fn_accepted, per_class_total[1]
        ),
        # Distinct operational conditional rate among accepted malware only.
        "conformal_fn_rate_given_accepted_malware": _ratio(
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
        "conformal_num_ineligible_forced_reject": int(
            ineligible_forced_rejects
        ),
        "conformal_ineligible_set_policy": "full_ambiguous_set",
    }
    return out


def _uses_risk_control_selective(config: dict | None = None) -> bool:
    config = {} if config is None else config
    mode = _selective_prediction_mode(config)
    return _strict_config_bool(config, "enabled", False) and mode == "risk_control"


def fit_risk_control_thresholds(
    rows: list[dict[str, Any]], config: dict | None = None
) -> dict[str, Any] | None:
    """Maximize acceptance under corrected accepted-FN risk among malware.

    The bounded loss is one only when a malware sample is both accepted and
    predicted benign. Following conformal risk-control calibration, the finite
    sample correction is ``(errors + 1) / (n_malware + 1)``. Therefore the
    controlled denominator is every malware sample, not only accepted malware;
    the latter conditional FNR is reported separately and is not guaranteed by
    this rule. Under the stated exchangeability assumptions, the guarantee is
    in expectation over the calibration sample and a new malware example.
    """
    config = {} if config is None else config
    if not _uses_risk_control_selective(config):
        return None
    score_type = _selective_score_type(config)
    risk_level = float(config.get("risk_level", 0.05))
    if not 0.0 < risk_level < 1.0:
        raise ValueError("selective_prediction.risk_level must be within (0, 1)")
    require_feasible = _strict_config_bool(config, "require_feasible", False)
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
        config.get("risk_target", "accepted_fn_risk_among_malware")
    ).lower()
    if risk_target != "accepted_fn_risk_among_malware":
        raise ValueError(
            "selective_prediction.risk_target currently supports only "
            "'accepted_fn_risk_among_malware'"
        )

    valid: list[tuple[float, int, int, bool]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Risk-control calibration row {row_index} must be a mapping")
        score = _finite_row_float(row, "acceptance_score")
        label_raw = _finite_row_float(row, "label")
        pred_raw = _finite_row_float(row, "pred")
        if label_raw is None or pred_raw is None:
            raise ValueError(
                f"Risk-control calibration row {row_index} has invalid label/pred"
            )
        if label_raw not in {0.0, 1.0} or pred_raw not in {0.0, 1.0}:
            raise ValueError(
                f"Risk-control calibration row {row_index} label/pred must be binary"
            )
        label = int(label_raw)
        pred = int(pred_raw)
        if score is None:
            raise ValueError(
                f"Risk-control calibration row {row_index} has no finite acceptance score"
            )
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Risk-control calibration row {row_index} acceptance score must be within [0, 1]"
            )
        valid.append(
            (float(score), label, pred, _row_selective_eligible(row))
        )
    if not valid:
        raise ValueError("Risk-control calibration requires finite acceptance scores")

    malware_count = sum(label == 1 for _score, label, _pred, _eligible in valid)
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

    scores = sorted(
        {score for score, _label, _pred, eligible in valid if eligible}
    )
    # Under the declared strict rule, a threshold equal to ``score`` rejects
    # that complete tie group. The one threshold just below the minimum score
    # represents accept-all; the exact observed scores then enumerate every
    # successively more conservative set. Using the exact rejected boundary is
    # important: placing the threshold just below the *next accepted* score has
    # the same calibration set but can needlessly reject an arbitrary interval
    # of future scores.
    candidates = (
        [math.nextafter(scores[0], float("-inf")), *scores]
        if scores
        else [1.0]
    )
    # All public acceptance scores are bounded in [0, 1].  A strict ``> 1``
    # rule is therefore a deployment-safe reject-all fallback, unlike the
    # largest calibration score, which says nothing about future examples.
    reject_all_threshold = 1.0
    best: dict[str, Any] | None = None
    for threshold in candidates:
        accepted = [
            item for item in valid if item[3] and item[0] > threshold
        ]
        false_negatives = sum(
            label == 1 and pred == 0
            for _score, label, pred, _eligible in accepted
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
        "score_type": score_type,
        "num_calibration": int(len(valid)),
        "num_calibration_malware": int(malware_count),
        "min_calibration_malware": int(minimum_malware),
        "minimum_malware_for_feasibility": int(
            minimum_malware_for_feasibility
        ),
        "require_feasible": bool(require_feasible),
        "guarantee_type": "expected_crc",
        "guarantee_scope": "exchangeable_expected_risk",
        "risk_numerator": "accepted_and_predicted_benign_malware",
        "risk_denominator": "all_malware",
        "risk_definition": (
            "E[1{accepted and predicted_benign} | label=malware]"
        ),
        "finite_sample_correction": "(accepted_fn_count + 1) / (n_malware + 1)",
        "acceptance_comparison": ACCEPTANCE_THRESHOLD_COMPARISON,
        "eligibility_rule": "at_least_one_primary_modality_alive",
        "num_ineligible_forced_reject": int(
            sum(not eligible for _score, _label, _pred, eligible in valid)
        ),
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
    minimum_threshold = math.nextafter(0.0, float("-inf"))
    if (
        not math.isfinite(threshold)
        or threshold < minimum_threshold
        or threshold > 1.0
    ):
        raise ValueError(
            "Risk-control threshold must be finite within the score boundary"
        )
    comparison = str(
        thresholds.get(
            "acceptance_comparison", ACCEPTANCE_THRESHOLD_COMPARISON
        )
    )
    if comparison != ACCEPTANCE_THRESHOLD_COMPARISON:
        raise ValueError(
            "Risk-control checkpoint uses an incompatible acceptance comparison; "
            "retrain the pipeline checkpoint"
        )
    valid: list[tuple[float, int, int, bool]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Risk-control evaluation row {row_index} must be a mapping")
        score = _finite_row_float(row, "acceptance_score")
        label_raw = _finite_row_float(row, "label")
        pred_raw = _finite_row_float(row, "pred")
        if label_raw is None or pred_raw is None:
            raise ValueError(
                f"Risk-control evaluation row {row_index} has invalid label/pred"
            )
        if label_raw not in {0.0, 1.0} or pred_raw not in {0.0, 1.0}:
            raise ValueError(
                f"Risk-control evaluation row {row_index} label/pred must be binary"
            )
        label = int(label_raw)
        pred = int(pred_raw)
        if score is None:
            raise ValueError(
                f"Risk-control evaluation row {row_index} has no finite acceptance score"
            )
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Risk-control evaluation row {row_index} acceptance score must be within [0, 1]"
            )
        eligible = _row_selective_eligible(row)
        accepted_by_risk_control = bool(eligible and float(score) > threshold)
        row["risk_control_threshold"] = threshold
        row["risk_control_acceptance_comparison"] = comparison
        row["selective_eligible"] = int(eligible)
        row["risk_control_accepted"] = int(accepted_by_risk_control)
        row["risk_control_rejected"] = int(not accepted_by_risk_control)
        # Canonical per-row decision fields are consumed by the analysis
        # scripts; the prefixed copies above retain explicit audit provenance.
        row["accepted"] = row["risk_control_accepted"]
        row["rejected"] = row["risk_control_rejected"]
        row["selective_decision_mode"] = "risk_control"
        valid.append((float(score), label, pred, eligible))
    if not valid:
        return {}

    accepted = [item for item in valid if item[3] and item[0] > threshold]
    accepted_errors = sum(
        label != pred for _score, label, pred, _eligible in accepted
    )
    malware_count = sum(
        label == 1 for _score, label, _pred, _eligible in valid
    )
    accepted_malware = sum(
        label == 1 for _score, label, _pred, _eligible in accepted
    )
    malware_false_negatives = sum(
        label == 1 and pred == 0
        for _score, label, pred, _eligible in accepted
    )

    def _ratio(num: int, den: int) -> float | None:
        return float(num) / float(den) if den > 0 else None

    empirical_risk = _ratio(malware_false_negatives, malware_count)
    risk_level = float(thresholds.get("risk_level", 0.0))
    return {
        "risk_control_threshold": threshold,
        "risk_control_acceptance_comparison": comparison,
        "risk_control_risk_level": risk_level,
        "risk_control_risk_target": str(thresholds.get("risk_target", "")),
        "risk_control_guarantee_type": str(
            thresholds.get("guarantee_type", "expected_crc")
        ),
        "risk_control_guarantee_scope": str(
            thresholds.get("guarantee_scope", "exchangeable_expected_risk")
        ),
        "risk_control_risk_numerator": str(
            thresholds.get(
                "risk_numerator", "accepted_and_predicted_benign_malware"
            )
        ),
        "risk_control_risk_denominator": str(
            thresholds.get("risk_denominator", "all_malware")
        ),
        "risk_control_eligibility_rule": str(
            thresholds.get(
                "eligibility_rule", "at_least_one_primary_modality_alive"
            )
        ),
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
        "risk_control_accepted_fn_risk_among_malware": empirical_risk,
        "risk_control_fn_rate_given_accepted_malware": _ratio(
            malware_false_negatives, accepted_malware
        ),
        "risk_control_malware_fn_count": int(malware_false_negatives),
        "risk_control_accepted_malware_count": int(accepted_malware),
        "risk_control_num_accepted": int(len(accepted)),
        "risk_control_num_rejected": int(len(valid) - len(accepted)),
        "risk_control_num_ineligible_forced_reject": int(
            sum(not eligible for _score, _label, _pred, eligible in valid)
        ),
        "risk_control_target_met_empirically": (
            None if empirical_risk is None else bool(empirical_risk <= risk_level)
        ),
    }


@torch.inference_mode()
def evaluate_checkpoint_selection(
    model,
    loader,
    device,
    use_amp: bool,
    split_name: str = "val_checkpoint_selection",
) -> dict[str, Any]:
    """Evaluate only the clean metrics consumed by encoder checkpoint selection.

    This profile deliberately excludes row dumps, selective-prediction scores,
    and fusion diagnostics. Predictions and labels stay on the accelerator for
    the whole validation pass and are transferred together once, avoiding the
    per-batch synchronizations of the full reporting evaluator.
    """
    model.eval()
    label_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    branch_probability_batches: dict[str, list[torch.Tensor]] = {
        branch: [] for branch in BRANCH_EVAL_LOGIT_KEYS
    }
    branch_outputs_available = {
        branch: True for branch in BRANCH_EVAL_LOGIT_KEYS
    }
    num_failed = 0

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, _sids, failed = prepare_robust_batch(
            batch, device
        )
        num_failed += int(failed)
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            logits, extra = model(graph)
        probability_batches.append(
            torch.softmax(logits.float(), dim=-1)[:, 1]
        )
        for branch, logit_key in BRANCH_EVAL_LOGIT_KEYS.items():
            branch_logits = extra.get(logit_key)
            if not isinstance(branch_logits, torch.Tensor):
                branch_outputs_available[branch] = False
                branch_probability_batches[branch].clear()
                continue
            if branch_outputs_available[branch]:
                branch_probability_batches[branch].append(
                    torch.softmax(branch_logits.float(), dim=-1)[:, 1]
                )
        label_batches.append(labels.long().view(-1))

    if not label_batches:
        metrics = _metrics([], [], [])
        metrics["num_failed"] = int(num_failed)
        metrics["num_eval"] = 0
        return metrics

    labels = torch.cat(label_batches, dim=0)
    probabilities = torch.cat(probability_batches, dim=0)
    predictions = probabilities.ge(0.5)
    packed = torch.stack(
        [
            labels.to(dtype=torch.float32),
            probabilities,
            predictions.to(dtype=torch.float32),
        ],
        dim=-1,
    ).cpu()
    packed_rows = packed.tolist()
    labels_cpu = [int(row[0]) for row in packed_rows]
    probabilities_cpu = [float(row[1]) for row in packed_rows]
    predictions_cpu = [int(row[2]) for row in packed_rows]
    metrics = _metrics(labels_cpu, probabilities_cpu, predictions_cpu)
    branch_metrics: dict[str, dict[str, Any]] = {}
    for branch, batches in branch_probability_batches.items():
        if not branch_outputs_available[branch] or len(batches) != len(label_batches):
            continue
        branch_probabilities = torch.cat(batches, dim=0).cpu()
        if branch_probabilities.numel() != labels.numel():
            raise RuntimeError(
                f"Checkpoint-selection {branch} branch rows disagree with labels"
            )
        branch_probs_cpu = [float(value) for value in branch_probabilities.tolist()]
        branch_preds_cpu = [int(value >= 0.5) for value in branch_probs_cpu]
        current = _metrics(labels_cpu, branch_probs_cpu, branch_preds_cpu)
        branch_metrics[branch] = current
        for key in ("macro_f1", "f1", "auc", "acc"):
            metrics[f"{branch}_{key}"] = current[key]
    if branch_metrics:
        metrics["branch_metrics"] = branch_metrics
    metrics["num_failed"] = int(num_failed)
    metrics["num_eval"] = int(len(labels_cpu))
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp: bool,
    split_name: str,
    dump_rows: bool = False,
    selective_threshold: float | None = None,
    selective_score_type: str = "msp",
    classification_threshold: float = 0.5,
    classification_log_odds_threshold: float | None = None,
):
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    preds_all: list[int] = []
    fixed_preds_all: list[int] = []
    acceptance_all: list[float] = []
    selective_eligibility_all: list[bool] = []
    rows: list[dict[str, Any]] = []
    num_failed = 0
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}
    fusion_weight_square_sums: dict[str, float] = {}

    if classification_log_odds_threshold is not None and not math.isfinite(
        float(classification_log_odds_threshold)
    ):
        raise ValueError("classification_log_odds_threshold must be finite")

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, sids, failed = prepare_robust_batch(batch, device)
        num_failed += failed
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            logits, extra = model(graph)
        prob = torch.softmax(logits.float(), dim=-1)[:, 1]
        pred_fixed = (prob >= 0.5).long()
        if classification_log_odds_threshold is None:
            pred = (prob >= float(classification_threshold)).long()
        else:
            raw_log_prob = extra.get("uncalibrated_final_log_prob")
            if not isinstance(raw_log_prob, torch.Tensor) or (
                raw_log_prob.ndim != 2
                or raw_log_prob.size(0) != labels.numel()
                or raw_log_prob.size(1) != 2
            ):
                raise ValueError(
                    "Raw-log-odds classification requires "
                    "uncalibrated_final_log_prob with shape [B, 2]"
                )
            raw_log_odds = (
                raw_log_prob.float()[:, 1] - raw_log_prob.float()[:, 0]
            )
            pred = (
                raw_log_odds >= float(classification_log_odds_threshold)
            ).long()
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
            predicted_malware_override=pred.bool(),
        )
        selective_eligibility = _batch_selective_eligibility(
            extra,
            batch_size=int(labels.numel()),
            device=labels.device,
        )
        acceptance_all.extend(acceptance.detach().float().cpu().view(-1).tolist())
        selective_eligibility_all.extend(
            selective_eligibility.detach().cpu().bool().tolist()
        )
        for key in GATE_DIAGNOSTIC_KEYS:
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                finite = value.detach().float().view(-1)
                finite = finite[torch.isfinite(finite)]
                if finite.numel() > 0:
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(finite.sum().cpu().item())
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(finite.numel())

                    if key in {
                        "fusion_weight_api",
                        "fusion_weight_graph",
                        "fusion_weight_manifest",
                    }:
                        fusion_weight_square_sums[key] = (
                            fusion_weight_square_sums.get(key, 0.0)
                            + float(finite.square().sum().cpu().item())
                        )

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
                    "classification_log_odds_threshold": (
                        None
                        if classification_log_odds_threshold is None
                        else float(classification_log_odds_threshold)
                    ),
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
                selective_eligible_i = bool(
                    selective_eligibility.view(-1)[i].detach().cpu().item()
                )
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
                    })
                for key in GATE_DIAGNOSTIC_KEYS:
                    value = extra.get(key)
                    if isinstance(value, torch.Tensor) and value.numel() > i:
                        row[key] = float(value.view(-1)[i].detach().cpu().item())
                # GATE_DIAGNOSTIC_KEYS contains the model's internal acceptance
                # diagnostic. Keep the experiment-selected score under the
                # canonical column used for threshold fitting and subset cuts.
                row["acceptance_score"] = acceptance_i
                row["selective_eligible"] = int(selective_eligible_i)
                row["selective_score_type"] = str(selective_score_type)
                if selective_threshold is not None:
                    accepted_i = bool(
                        selective_eligible_i
                        and acceptance_i > selective_threshold
                    )
                    row["accepted"] = int(accepted_i)
                    row["rejected"] = int(not accepted_i)
                    row["selective_decision_mode"] = "threshold"
                    row["acceptance_comparison"] = (
                        ACCEPTANCE_THRESHOLD_COMPARISON
                    )
                rows.append(row)

    metrics = _metrics(labels_all, probs_all, preds_all)
    fixed_metrics = _metrics(labels_all, probs_all, fixed_preds_all)
    metrics.update(
        {
            "classification_threshold": float(classification_threshold),
            "classification_log_odds_threshold": (
                None
                if classification_log_odds_threshold is None
                else float(classification_log_odds_threshold)
            ),
            "fixed_0_5_acc": fixed_metrics["acc"],
            "fixed_0_5_macro_f1": fixed_metrics["macro_f1"],
            "fixed_0_5_f1_pos": fixed_metrics["f1_pos"],
            "fixed_0_5_recall_pos": fixed_metrics["recall_pos"],
        }
    )
    metrics["num_failed"] = int(num_failed)
    metrics["num_eval"] = int(len(labels_all))
    if len(acceptance_all) == len(labels_all):
        metrics.update(
            _selective_ranking_metrics(
                labels_all,
                preds_all,
                acceptance_all,
                selective_eligibility=selective_eligibility_all,
            )
        )
        metrics["selective_eligible_rate"] = float(
            sum(selective_eligibility_all) / max(len(selective_eligibility_all), 1)
        )
        metrics["num_ineligible_forced_reject"] = int(
            len(selective_eligibility_all) - sum(selective_eligibility_all)
        )
        if selective_threshold is not None:
            metrics.update(
                _selective_metrics(
                    labels_all,
                    preds_all,
                    acceptance_all,
                    selective_threshold,
                    selective_eligibility_all,
                )
            )
    for key, total in diagnostic_sums.items():
        metrics[f"mean_{key}"] = total / max(
            diagnostic_counts.get(key, 0), 1
        )

    # Sample-level fusion-weight variation. In the proposed routed method these
    # are exactly pi; comparison rules expose their own attributable weights.
    for key, square_sum in fusion_weight_square_sums.items():
        count = diagnostic_counts.get(key, 0)

        if count > 0:
            mean = diagnostic_sums[key] / count
            variance = max(
                square_sum / count - mean * mean,
                0.0,
            )
            metrics[f"std_{key}"] = float(
                math.sqrt(variance)
            )

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
            # calibrated = F.log_softmax(
            #     raw_log_prob / log_temperature.exp(),
            #     dim=-1,
            # )
            temperature = bounded_final_temperature(log_temperature)
            calibrated = F.log_softmax(
                raw_log_prob / temperature,
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
        # fitted_temperature = float(log_temperature.exp().item())
        # calibrated = F.log_softmax(
        #     raw_log_prob / log_temperature.exp(),
        #     dim=-1,
        # )
        temperature = bounded_final_temperature(log_temperature)
        fitted_temperature = float(temperature.item())
        calibrated = F.log_softmax(
            raw_log_prob / temperature,
            dim=-1,
        )
        nll_after = float(F.nll_loss(calibrated, temperature_labels).item())
        if nll_after > nll_before + 1.0e-8:
            # Temperature scaling is calibration-only. Retain identity when
            # numerical optimization does not improve its fitting objective.
            log_temperature.zero_()
            fitted_temperature = 1.0
            nll_after = nll_before
    # if not math.isfinite(fitted_temperature) or fitted_temperature <= 0.0:
    #     raise RuntimeError("Routed final temperature is non-finite or non-positive")
    if not math.isfinite(fitted_temperature):
        raise RuntimeError("Routed final temperature is non-finite")

    # bound_tolerance = 1.0e-6
    if not (
        FINAL_TEMPERATURE_MIN
        <= fitted_temperature
        <= FINAL_TEMPERATURE_MAX
    ):
        raise RuntimeError(
            "Routed final temperature escaped its declared bounds: "
            f"temperature={fitted_temperature:.9g}, "
            f"bounds=[{FINAL_TEMPERATURE_MIN}, "
            f"{FINAL_TEMPERATURE_MAX}]"
        )
    return {
        "enabled": True,
        "temperature": fitted_temperature,
        "num_clean_samples": int(temperature_labels.numel()),
        "nll_before": nll_before,
        "nll_after": nll_after,
    }


def resolve_posthoc_cross_fitting(cfg: dict) -> dict[str, Any]:
    """Resolve identity-grouped nested cross-fitting for the post-hoc stack.

    The former role-disjoint 2/2/1/1 fold allocation was leakage-safe but left
    every small decision module trained on only a fraction of the available
    package identities.  The formal protocol now uses outer-route/inner-I1
    cross-fitting and refits deployable I1 on the full post-hoc identity pool.
    """
    calibration_cfg = cfg.get("calibration", {}) or {}
    if "stage_split" in calibration_cfg:
        raise ValueError(
            "calibration.stage_split was removed; use calibration.cross_fitting"
        )
    raw = calibration_cfg.get("cross_fitting", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("calibration.cross_fitting must be a mapping")
    required = bool(raw.get("required", False))
    enabled = bool(raw.get("enabled", False))
    split_seed = int(
        raw.get(
            "split_seed",
            calibration_cfg.get(
                "split_seed", cfg.get("train", {}).get("seed", 42)
            ),
        )
    )
    if not enabled:
        if required:
            raise ValueError(
                "calibration.cross_fitting.required=true forbids full-data "
                "I1 -> I2 stacking; enable nested cross-fitting"
            )
        return {
            "required": False,
            "enabled": False,
            "mode": "full_fit",
            "num_folds": 1,
            "split_seed": split_seed,
            "identity_disjoint": False,
            "strictly_nested": False,
        }

    mode = str(raw.get("mode", "nested")).strip().lower()
    if mode != "nested":
        raise ValueError(
            "calibration.cross_fitting.mode must be 'nested'; a non-nested "
            "route fold can indirectly see its holdout labels through I1"
        )
    num_folds = int(raw.get("num_folds", 5))
    if num_folds < 3:
        raise ValueError(
            "calibration.cross_fitting.num_folds must be at least 3 so each "
            "outer route fold has a non-empty inner I1 training complement"
        )
    return {
        "required": required,
        "enabled": True,
        "mode": "nested",
        "num_folds": num_folds,
        "split_seed": split_seed,
        "identity_disjoint": True,
        "strictly_nested": True,
    }


def deterministic_stratified_fold_ids(
    labels: torch.Tensor,
    sample_ids: list[str] | tuple[str, ...],
    *,
    num_folds: int,
    seed: int,
) -> torch.Tensor:
    """Assign stable class-stratified folds, grouping duplicate identities."""
    labels_cpu = labels.detach().long().view(-1).cpu().tolist()
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(labels_cpu):
        raise ValueError("sample_ids and labels must have the same length")
    if num_folds <= 0:
        raise ValueError("num_folds must be positive")
    identity_labels: dict[str, int] = {}
    for sample_id, label in zip(ids, labels_cpu):
        label = int(label)
        previous = identity_labels.setdefault(sample_id, label)
        if previous != label:
            raise ValueError(
                f"sample identity {sample_id!r} has conflicting labels {previous} and {label}"
            )
    identity_folds: dict[str, int] = {}
    by_label: dict[int, list[str]] = {}
    for sample_id, label in identity_labels.items():
        by_label.setdefault(label, []).append(sample_id)
    for label, label_ids in sorted(by_label.items()):
        ordered = sorted(
            label_ids,
            key=lambda sample_id: (
                hashlib.sha256(
                    f"{int(seed)}|{label}|{sample_id}".encode("utf-8")
                ).hexdigest(),
                sample_id,
            ),
        )
        for rank, sample_id in enumerate(ordered):
            identity_folds[sample_id] = rank % num_folds
    fold_ids = [identity_folds[sample_id] for sample_id in ids]
    return torch.tensor(fold_ids, device=labels.device, dtype=torch.long)


def _slice_cached_calibration_item(
    cached: dict[str, Any],
    folds: list[int],
) -> dict[str, Any] | None:
    fold_ids = cached.get("fold_ids")
    if not isinstance(fold_ids, torch.Tensor):
        raise ValueError("cached calibration item is missing fold_ids")
    mask = torch.zeros_like(fold_ids, dtype=torch.bool)
    for fold in folds:
        mask |= fold_ids.eq(int(fold))
    if not bool(mask.any().item()):
        return None
    indices = mask.nonzero(as_tuple=False).view(-1)
    sliced = dict(cached)
    row_count = int(fold_ids.numel())
    for key, value in cached.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.size(0)) == row_count
            and key != "fold_ids"
        ):
            sliced[key] = value.index_select(0, indices)
    sliced["fold_ids"] = fold_ids.index_select(0, indices)
    sliced["branch_logits"] = {
        name: value.index_select(0, indices)
        for name, value in cached["branch_logits"].items()
    }
    sample_ids = cached.get("sample_ids") or []
    sample_groups = cached.get("sample_groups") or sample_ids
    cpu_indices = indices.detach().cpu().tolist()
    sliced["sample_ids"] = [sample_ids[index] for index in cpu_indices]
    sliced["sample_groups"] = [sample_groups[index] for index in cpu_indices]
    sliced["_cache_index"] = int(cached["_cache_index"])
    sliced["_row_indices"] = indices
    return sliced


def _compile_posthoc_source_masses(
    objective_groups: list[dict[str, Any]],
    items: list[dict[str, Any]],
    scenario_weights: dict[str, float],
) -> dict[int, float]:
    """Compile the exact group -> source coefficients used post hoc.

    The existing hierarchy averages objective groups, gives clean/perturb the
    configured deployment prior inside every non-clean family, and finally
    averages sources within each side.  Compiling those coefficients once lets
    packed objectives reduce per-row losses without allowing source row counts
    or grid density to change objective mass. I1 compiles one such hierarchy
    per branch before applying the outer equal-branch average.
    """
    if not objective_groups:
        raise RuntimeError("Cannot compile an empty post-hoc objective")
    if not items:
        raise RuntimeError("Cannot compile post-hoc weights without sources")
    item_ids = {id(item) for item in items}
    if len(item_ids) != len(items):
        raise RuntimeError("Post-hoc source selection contains duplicate objects")
    clean_weight = float(scenario_weights["clean"])
    perturb_weight = float(scenario_weights["perturb"])
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (clean_weight, perturb_weight)
    ) or not math.isclose(clean_weight + perturb_weight, 1.0, abs_tol=1.0e-9):
        raise ValueError("Post-hoc scenario weights must be normalized and non-negative")

    masses = {item_id: 0.0 for item_id in item_ids}
    group_scale = 1.0 / float(len(objective_groups))

    def _add(source_items: list[dict[str, Any]], mass: float) -> None:
        if not source_items:
            raise RuntimeError("Post-hoc objective group contains no clean sources")
        per_source = group_scale * float(mass) / float(len(source_items))
        for item in source_items:
            item_id = id(item)
            if item_id not in masses:
                raise RuntimeError(
                    "Post-hoc objective group references a source outside its "
                    "packed stage"
                )
            masses[item_id] += per_source

    for group in objective_groups:
        clean_items = list(group.get("clean") or [])
        scenario_items = list(group.get("scenario") or [])
        if scenario_items:
            _add(clean_items, clean_weight)
            _add(scenario_items, perturb_weight)
        else:
            # This is the historical clean-only behaviour: when there is no
            # perturb side the configured prior is not allowed to discard clean
            # objective mass.
            _add(clean_items, 1.0)

    total_mass = sum(masses.values())
    if not math.isclose(total_mass, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise RuntimeError(
            "Compiled post-hoc source masses do not sum to one: "
            f"{total_mass:.12g}"
        )
    return masses


def _compile_posthoc_row_weights(
    items: list[dict[str, Any]],
    segments: list[tuple[int, int]],
    source_masses: dict[int, float],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Expand source mass into source-normalized packed row weights.

    Every source retains its own ``max(valid_count, 1)`` denominator.  An
    all-dead source therefore contributes zero and its mass is deliberately not
    redistributed to other sources, matching the previous nested reduction.
    """
    if len(items) != len(segments):
        raise RuntimeError("Post-hoc source/segment counts disagree")
    valid = valid_mask.detach().view(-1).bool()
    weights = torch.zeros(valid.numel(), device=valid.device, dtype=torch.float32)
    expected_start = 0
    for item, (raw_start, raw_end) in zip(items, segments):
        start, end = int(raw_start), int(raw_end)
        if start != expected_start or end <= start or end > valid.numel():
            raise RuntimeError("Post-hoc packed source segments are not contiguous")
        expected_start = end
        item_id = id(item)
        if item_id not in source_masses:
            raise RuntimeError("Post-hoc source is missing its objective mass")
        source_valid = valid[start:end].to(dtype=weights.dtype)
        denominator = source_valid.sum().clamp_min(1.0)
        weights[start:end] = (
            float(source_masses[item_id]) * source_valid / denominator
        )
    if expected_start != valid.numel():
        raise RuntimeError("Post-hoc packed segments do not cover the valid mask")
    return weights.detach()


def fit_posthoc_calibration(
    model,
    loaders: list,
    device,
    use_amp: bool,
    cfg: dict,
) -> dict[str, Any]:
    """Fit I1 reliability, I2 routing, and optional final temperature post hoc."""
    calibration_cfg = cfg.get("calibration", {}) or {}
    if not bool(calibration_cfg.get("enabled", False)):
        return {"enabled": False}
    fit_started_at = time.perf_counter()
    if str(getattr(model, "fusion_mode", "")) != "discount_probability":
        raise ValueError(
            "calibration.enabled=true requires model fusion_mode=discount_probability"
        )
    discount_fusion = getattr(model, "discount_fusion", None)
    parameters = list(model.calibration_parameters())
    final_temperature_parameters = (
        list(discount_fusion.final_temperature_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "final_temperature_parameters")
        else []
    )
    if not parameters and not final_temperature_parameters:
        raise ValueError(
            "calibration.enabled=true requires reliability, routing, or "
            "final-temperature calibration parameters"
        )
    stage_optimization_cfg = calibration_cfg.get("stage_optimization", {}) or {}
    if not isinstance(stage_optimization_cfg, dict):
        raise ValueError("calibration.stage_optimization must be a mapping")
    supported_stage_optimization = {
        "default",
        "reliability",
        "routing_distribution",
        "routing_risk",
        "final_temperature",
    }
    unknown_stage_optimization = sorted(
        set(stage_optimization_cfg) - supported_stage_optimization
    )
    if unknown_stage_optimization:
        raise ValueError(
            "calibration.stage_optimization contains unsupported stage blocks: "
            f"{unknown_stage_optimization}"
        )
    default_max_steps = 300

    def _scenario_group(item: dict[str, Any], name: str, index: int) -> str:
        explicit = item.get("scenario_group")
        if explicit:
            return str(explicit).lower()
        perturb_type = item.get("perturb_type")
        if perturb_type:
            perturb_name = str(perturb_type).lower()
            return "missing" if perturb_name.endswith("_missing") else perturb_name
        return "clean" if index == 0 or name == "clean" else "other"

    def _objective_family(item: dict[str, Any], name: str, index: int) -> str:
        explicit = item.get("objective_family")
        if explicit:
            return str(explicit).strip().lower()
        perturb_type = str(item.get("perturb_type") or "").strip().lower()
        if perturb_type:
            family = ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION.get(perturb_type)
            if family is None:
                raise ValueError(
                    f"Calibration source {name!r} has no routing objective family "
                    f"for perturb_type={perturb_type!r}"
                )
            return family
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
                "objective_family": _objective_family(
                    item, source_name, loader_index
                ),
                "perturb_type": str(item.get("perturb_type") or "clean").lower(),
                "strength": float(item.get("strength", 0.0)),
                "reliability_branches": tuple(
                    str(name).lower()
                    for name in configured_branches
                ),
                "combined_source_index": (
                    int(item["combined_source_index"])
                    if "combined_source_index" in item
                    else None
                ),
            }
        else:
            source = {
                "loader": item,
                "name": "clean" if loader_index == 0 else f"calibration_{loader_index}",
                "scenario_group": "clean" if loader_index == 0 else "other",
                "objective_family": "clean" if loader_index == 0 else "other",
                "perturb_type": "clean" if loader_index == 0 else "other",
                "strength": 0.0,
                "reliability_branches": ("api", "graph", "manifest"),
                "combined_source_index": None,
            }
        perturb_type = str(source["perturb_type"])
        declared_i1_branches = tuple(source["reliability_branches"])
        if perturb_type == "clean":
            expected_i1_branches = None
        else:
            expected_i1_branches = RELIABILITY_CALIBRATION_PERTURBATIONS.get(
                perturb_type
            )
        if (
            expected_i1_branches is not None
            and declared_i1_branches != tuple(expected_i1_branches)
        ):
            raise ValueError(
                f"Calibration source {source['name']!r} assigns I1 branches "
                f"{declared_i1_branches}, expected {tuple(expected_i1_branches)} "
                "from the registered branch-local partial-degradation protocol"
            )
        calibration_sources.append(source)

    # Cache the frozen encoders' branch logits and the alive-mask carrier once.
    # Calibration then optimizes only the small decision module instead of
    # repeating the API Transformer and GNN forward pass for every epoch.
    cached_batches: list[dict[str, Any]] = []
    source_offsets = [0 for _source in calibration_sources]
    source_failed_counts = [0 for _source in calibration_sources]
    processed_combined_loaders: set[int] = set()
    cache_started_at = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for declared_loader_index, declared_source in enumerate(calibration_sources):
            loader = declared_source["loader"]
            combined_source_index = declared_source["combined_source_index"]
            if combined_source_index is None:
                combined_source_map = None
            else:
                loader_identity = id(loader)
                if loader_identity in processed_combined_loaders:
                    continue
                processed_combined_loaders.add(loader_identity)
                combined_source_map: dict[int, int] = {}
                for candidate_index, candidate in enumerate(calibration_sources):
                    if candidate["loader"] is not loader:
                        continue
                    local_source_index = candidate["combined_source_index"]
                    if local_source_index is None:
                        raise RuntimeError(
                            "A combined calibration loader cannot also be declared "
                            "as an untagged source"
                        )
                    if local_source_index in combined_source_map:
                        raise RuntimeError(
                            "Combined calibration loader has duplicate source index "
                            f"{local_source_index}"
                        )
                    combined_source_map[local_source_index] = candidate_index

            for batch in tqdm(loader, desc="cache calibration", leave=False):
                if combined_source_map is None:
                    loader_index = declared_loader_index
                    source = declared_source
                    if batch.get("calibration_source_index") is not None:
                        raise RuntimeError(
                            "An untagged calibration source emitted combined-loader metadata"
                        )
                else:
                    raw_source_index = batch.get("calibration_source_index")
                    if raw_source_index is None:
                        raise RuntimeError(
                            "Combined calibration loader batch is missing its source index"
                        )
                    local_source_index = int(raw_source_index)
                    if local_source_index not in combined_source_map:
                        raise RuntimeError(
                            "Combined calibration loader emitted unknown source index "
                            f"{local_source_index}"
                        )
                    loader_index = combined_source_map[local_source_index]
                    source = calibration_sources[loader_index]

                graph, labels, sids, failed = prepare_robust_batch(
                    batch, device
                )
                source_failed_counts[loader_index] += int(failed)
                if graph is None:
                    continue
                batch_size = int(labels.numel())
                if sids is None:
                    sample_ids = [
                        f"row-{source_offsets[loader_index] + index}"
                        for index in range(batch_size)
                    ]
                else:
                    sample_ids = [str(value) for value in sids]
                    if len(sample_ids) != batch_size:
                        raise RuntimeError(
                            "Post-hoc calibration cache received a sample-id "
                            "count that differs from labels"
                        )
                batch_groups = batch.get("sample_groups")
                group_source = (
                    "package_isolation_group"
                    if batch_groups is not None
                    else "sample_id_fallback"
                )
                sample_groups = (
                    [str(value) for value in batch_groups]
                    if batch_groups is not None
                    else list(sample_ids)
                )
                if len(sample_groups) != batch_size:
                    raise RuntimeError(
                        "Post-hoc calibration cache received a package-group "
                        "count that differs from labels"
                    )
                source_offsets[loader_index] += batch_size
                with get_amp_context(device, use_amp):
                    _logits, extra = model(graph)
                availability = extra.get("fusion_availability")
                branch_logits = {
                    name: extra.get(f"{name}_logits_aux")
                    for name in ("api", "graph", "manifest")
                }
                if not isinstance(availability, torch.Tensor) or any(
                    not isinstance(value, torch.Tensor) for value in branch_logits.values()
                ):
                    raise RuntimeError(
                        "Post-hoc calibration cache requires the alive-mask "
                        "carrier and all branch logits"
                    )
                cached_batches.append(
                    {
                        "labels": labels.detach(),
                        "availability": availability.detach().float(),
                        "loader_index": int(loader_index),
                        "scenario_name": source["name"],
                        "scenario_group": source["scenario_group"],
                        "objective_family": source["objective_family"],
                        "perturb_type": source["perturb_type"],
                        "strength": source["strength"],
                        "reliability_branches": source["reliability_branches"],
                        "sample_ids": sample_ids,
                        "sample_groups": sample_groups,
                        "group_source": group_source,
                        "branch_logits": {
                            name: value.detach().float()
                            for name, value in branch_logits.items()
                        },
                    }
                )
    for loader_index, source in enumerate(calibration_sources):
        source_failed = source_failed_counts[loader_index]
        if source_failed:
            raise RuntimeError(
                "Post-hoc calibration source "
                f"{source['name']!r} dropped {source_failed} samples; "
                "identity-disjoint fitting refuses silent scenario deletion"
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
                "availability": torch.cat(
                    [item["availability"] for item in source_batches], dim=0
                ),
                "loader_index": loader_index,
                "scenario_name": source["name"],
                "scenario_group": source["scenario_group"],
                "objective_family": source["objective_family"],
                "perturb_type": source["perturb_type"],
                "strength": source["strength"],
                "reliability_branches": source["reliability_branches"],
                "sample_ids": [
                    sample_id
                    for item in source_batches
                    for sample_id in item["sample_ids"]
                ],
                "sample_groups": [
                    sample_group
                    for item in source_batches
                    for sample_group in item["sample_groups"]
                ],
                "group_source": source_batches[0]["group_source"],
                "branch_logits": {
                    name: torch.cat(
                        [item["branch_logits"][name] for item in source_batches],
                        dim=0,
                    )
                    for name in ("api", "graph", "manifest")
                },
            }
        )
    cached_batches = merged_cached_batches
    cache_wall_time_seconds = float(time.perf_counter() - cache_started_at)
    logger.info(
        "posthoc_cache_complete sources=%s unique_loaders=%s "
        "encoder_batches=%s samples=%s wall_seconds=%.2f",
        len(calibration_sources),
        len({id(source["loader"]) for source in calibration_sources}),
        num_encoder_batches_cached,
        sum(int(item["labels"].numel()) for item in cached_batches),
        cache_wall_time_seconds,
    )
    for cache_index, cached in enumerate(cached_batches):
        cached["_cache_index"] = int(cache_index)
    group_sources = {str(item["group_source"]) for item in cached_batches}
    if len(group_sources) != 1:
        raise RuntimeError(
            "Post-hoc calibration sources disagree on package-group metadata: "
            f"{sorted(group_sources)}"
        )
    stage_grouping = next(iter(group_sources))

    cross_fitting = resolve_posthoc_cross_fitting(cfg)
    clean_for_identity = [
        item for item in cached_batches if item["scenario_group"] == "clean"
    ]
    if not clean_for_identity:
        raise RuntimeError("Post-hoc cross-fitting requires a clean identity source")
    clean_labels = torch.cat([item["labels"] for item in clean_for_identity], dim=0)
    clean_sample_groups = [
        sample_group
        for item in clean_for_identity
        for sample_group in item["sample_groups"]
    ]
    clean_fold_ids = deterministic_stratified_fold_ids(
        clean_labels,
        clean_sample_groups,
        num_folds=int(cross_fitting["num_folds"]),
        seed=int(cross_fitting["split_seed"]),
    )
    group_to_fold: dict[str, int] = {}
    group_to_label: dict[str, int] = {}
    for sample_group, fold, label in zip(
        clean_sample_groups,
        clean_fold_ids.detach().cpu().tolist(),
        clean_labels.detach().cpu().tolist(),
    ):
        group_key = str(sample_group)
        previous = group_to_fold.setdefault(group_key, int(fold))
        if previous != int(fold):
            raise RuntimeError(
                f"package isolation group {sample_group!r} was assigned multiple folds"
            )
        previous_label = group_to_label.setdefault(group_key, int(label))
        if previous_label != int(label):
            raise RuntimeError(
                "A clean package isolation group has conflicting labels: "
                f"{sample_group!r}"
            )
    for cached in cached_batches:
        unknown_groups = [
            sample_group
            for sample_group in cached["sample_groups"]
            if str(sample_group) not in group_to_fold
        ]
        if unknown_groups:
            raise RuntimeError(
                "A calibration scenario contains package isolation groups absent "
                f"from clean: {unknown_groups[:5]}"
            )
        label_mismatches = [
            (sample_group, int(label), group_to_label[str(sample_group)])
            for sample_group, label in zip(
                cached["sample_groups"],
                cached["labels"].detach().cpu().tolist(),
            )
            if int(label) != group_to_label[str(sample_group)]
        ]
        if label_mismatches:
            raise RuntimeError(
                "A calibration scenario changed the clean package label: "
                f"{label_mismatches[:5]}"
            )
        cached["fold_ids"] = torch.tensor(
            [group_to_fold[str(group)] for group in cached["sample_groups"]],
            device=cached["labels"].device,
            dtype=torch.long,
        )

    all_folds = list(range(int(cross_fitting["num_folds"])))

    def _fold_cached(folds: list[int], *, role: str) -> list[dict[str, Any]]:
        selected = [
            sliced
            for cached in cached_batches
            if (
                sliced := _slice_cached_calibration_item(cached, folds)
            ) is not None
        ]
        if not selected:
            raise RuntimeError(
                f"Post-hoc calibration selection {role!r} received no samples"
            )
        return selected

    full_cached = _fold_cached(all_folds, role="full_posthoc")
    full_clean_cached = [
        item for item in full_cached if item["scenario_group"] == "clean"
    ]
    if not full_clean_cached:
        raise RuntimeError("Post-hoc calibration requires clean samples")

    # I1 consumes only intrinsic branch opinion state plus the hard alive mask.
    # No extraction-quality, coverage, or validation-global reference is
    # exposed to any outer holdout.

    model.set_calibration_active(True)
    previous_requires_grad = {id(param): param.requires_grad for param in model.parameters()}
    default_learning_rate = float(calibration_cfg.get("lr", 1.0e-3))
    default_weight_decay = float(calibration_cfg.get("weight_decay", 0.0))
    default_grad_clip = float(calibration_cfg.get("grad_clip", 5.0))

    def _stage_optimization(stage_name: str) -> dict[str, Any]:
        shared = stage_optimization_cfg.get("default", {}) or {}
        specific = stage_optimization_cfg.get(stage_name, {}) or {}
        if not isinstance(shared, dict) or not isinstance(specific, dict):
            raise ValueError(
                "calibration.stage_optimization.default and each stage must be mappings"
            )
        resolved = {**shared, **specific}
        max_steps = int(resolved.get("max_steps", default_max_steps))
        optimizer_name = str(resolved.get("optimizer", "adam")).strip().lower()
        result = {
            "optimizer": optimizer_name,
            "max_steps": max_steps,
            "min_steps": int(resolved.get("min_steps", min(50, max_steps))),
            "convergence_patience": int(
                resolved.get("convergence_patience", 20)
            ),
            "convergence_tolerance": float(
                resolved.get("convergence_tolerance", 1.0e-7)
            ),
            "minimum_relative_improvement": float(
                resolved.get("minimum_relative_improvement", 0.0)
            ),
            "lr": float(resolved.get("lr", default_learning_rate)),
            "weight_decay": float(
                resolved.get("weight_decay", default_weight_decay)
            ),
            "grad_clip": float(resolved.get("grad_clip", default_grad_clip)),
            "log_every": int(resolved.get("log_every", 25)),
            "require_convergence": bool(
                resolved.get("require_convergence", False)
            ),
            "gradient_tolerance": float(
                resolved.get("gradient_tolerance", 1.0e-5)
            ),
            "lbfgs_history_size": int(
                resolved.get("lbfgs_history_size", 20)
            ),
            "lr_scheduler_patience": int(
                resolved.get("lr_scheduler_patience", 50)
            ),
            "lr_scheduler_factor": float(
                resolved.get("lr_scheduler_factor", 0.5)
            ),
            "min_lr": float(resolved.get("min_lr", 1.0e-6)),
        }
        if result["optimizer"] not in {"adam", "lbfgs"}:
            raise ValueError(
                f"{stage_name} optimizer must be 'adam' or 'lbfgs'"
            )
        if result["max_steps"] <= 0:
            raise ValueError(f"{stage_name} max_steps must be positive")
        if not 0 <= result["min_steps"] <= result["max_steps"]:
            raise ValueError(
                f"{stage_name} min_steps must be within [0, max_steps]"
            )
        if result["convergence_patience"] <= 0:
            raise ValueError(f"{stage_name} convergence_patience must be positive")
        for key in (
            "convergence_tolerance",
            "gradient_tolerance",
            "lr",
            "grad_clip",
        ):
            if not math.isfinite(result[key]) or result[key] <= 0.0:
                raise ValueError(f"{stage_name} {key} must be finite and positive")
        if (
            not math.isfinite(result["minimum_relative_improvement"])
            or result["minimum_relative_improvement"] < 0.0
        ):
            raise ValueError(
                f"{stage_name} minimum_relative_improvement must be finite and non-negative"
            )
        if (
            not math.isfinite(result["weight_decay"])
            or result["weight_decay"] < 0.0
        ):
            raise ValueError(f"{stage_name} weight_decay must be finite and non-negative")
        if result["log_every"] <= 0:
            raise ValueError(f"{stage_name} log_every must be positive")
        if result["lbfgs_history_size"] <= 0:
            raise ValueError(f"{stage_name} lbfgs_history_size must be positive")
        if result["lr_scheduler_patience"] <= 0:
            raise ValueError(f"{stage_name} lr_scheduler_patience must be positive")
        if not 0.0 < result["lr_scheduler_factor"] < 1.0:
            raise ValueError(
                f"{stage_name} lr_scheduler_factor must be within (0, 1)"
            )
        if not math.isfinite(result["min_lr"]) or result["min_lr"] <= 0.0:
            raise ValueError(f"{stage_name} min_lr must be finite and positive")
        if result["optimizer"] == "lbfgs" and result["weight_decay"] != 0.0:
            raise ValueError(
                f"{stage_name} LBFGS requires weight_decay=0; regularising raw "
                "softplus parameters does not shrink their effective weights"
            )
        return result

    def _set_trainable(stage_parameters: list[torch.nn.Parameter]) -> None:
        trainable_ids = {id(param) for param in stage_parameters}
        for param in model.parameters():
            param.requires_grad_(id(param) in trainable_ids)

    def _forward_cached(
        cached: dict[str, Any],
        *,
        reliability_key: str | None = None,
        route_key: str | None = None,
    ) -> dict[str, torch.Tensor]:
        branch_logits = cached["branch_logits"]
        availability = cached["availability"]
        reliability_override = (
            cached.get(reliability_key) if reliability_key is not None else None
        )
        route_override = cached.get(route_key) if route_key is not None else None
        if reliability_key is not None and not isinstance(
            reliability_override, torch.Tensor
        ):
            raise RuntimeError(
                f"cached calibration item is missing {reliability_key!r}"
            )
        if route_key is not None and not isinstance(route_override, torch.Tensor):
            raise RuntimeError(f"cached calibration item is missing {route_key!r}")
        override_kwargs: dict[str, torch.Tensor] = {}
        if isinstance(reliability_override, torch.Tensor):
            override_kwargs["reliability_override"] = reliability_override
        if isinstance(route_override, torch.Tensor):
            override_kwargs["branch_distribution_override"] = route_override
        outputs = model.discount_fusion(
            branch_logits["api"],
            branch_logits["graph"],
            branch_logits["manifest"],
            availability,
            **override_kwargs,
        )
        outputs.update(
            {
                f"{name}_logits_aux": value
                for name, value in branch_logits.items()
            }
        )
        outputs["fusion_availability"] = availability
        return outputs

    def _pack_cached_selection(
        items: list[dict[str, Any]],
        *,
        reliability_key: str | None,
        route_key: str | None,
    ) -> dict[str, Any]:
        """Pack scenario sources for one decision-module forward.

        Losses are still reduced source-by-source after the forward and then
        combined by the existing source -> scenario-family hierarchy.  Packing
        therefore changes only tensor scheduling; it does not let a scenario
        with more rows or strengths acquire additional objective mass.
        """
        if not items:
            raise RuntimeError("Cannot pack an empty calibration selection")
        segments: list[tuple[int, int]] = []
        offset = 0
        for item in items:
            rows = int(item["labels"].numel())
            if rows <= 0:
                raise RuntimeError("Calibration cache contains an empty scenario")
            segments.append((offset, offset + rows))
            offset += rows

        if len(items) == 1:
            packed = items[0]
        else:
            packed = {
                "labels": torch.cat([item["labels"] for item in items], dim=0),
                "availability": torch.cat(
                    [item["availability"] for item in items], dim=0
                ),
                "branch_logits": {
                    name: torch.cat(
                        [item["branch_logits"][name] for item in items], dim=0
                    )
                    for name in ("api", "graph", "manifest")
                },
            }
            for key in (reliability_key, route_key):
                if key is None:
                    continue
                values = [item.get(key) for item in items]
                if any(not isinstance(value, torch.Tensor) for value in values):
                    raise RuntimeError(
                        f"Packed calibration selection is missing {key!r}"
                    )
                packed[key] = torch.cat(values, dim=0)
        return {
            "items": items,
            "packed": packed,
            "segments": segments,
            "segment_by_item_id": {
                id(item): segment for item, segment in zip(items, segments)
            },
            "total_rows": offset,
        }

    def _slice_packed_outputs(
        outputs: dict[str, Any],
        start: int,
        end: int,
        total_rows: int,
    ) -> dict[str, Any]:
        # Decision-module outputs are predominantly batch tensors. Scalars and
        # fixed parameter diagnostics are shared without copying.
        return {
            key: (
                value[start:end]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.size(0)) == total_rows
                else value
            )
            for key, value in outputs.items()
        }

    def _mean_cached_loss(
        items: list[dict[str, Any]],
        loss_fn,
        context: dict[str, Any],
        *,
        reliability_key: str | None = None,
        route_key: str | None = None,
        requires_forward: bool = True,
    ) -> torch.Tensor:
        if not items:
            raise RuntimeError("Calibration objective group contains no scenarios")
        if not requires_forward:
            values = [loss_fn(item, None) for item in items]
            return torch.stack(values).mean()

        # Every source in a stage consists only of frozen encoder outputs and
        # small post-hoc tensors.  Pack the complete stage selection once per
        # override signature, then slice it back into sources for the original
        # source -> scenario-family reduction.  This preserves every objective
        # weight while reducing I1/route/risk to one decision forward per
        # objective evaluation instead of one forward per scenario family.
        selection_key = ("all_objective_items", reliability_key, route_key)
        selections = context["selections"]
        selection = selections.get(selection_key)
        if selection is None:
            selection = _pack_cached_selection(
                context["all_items"],
                reliability_key=reliability_key,
                route_key=route_key,
            )
            selections[selection_key] = selection

        forwards = context["forwards"]
        outputs = forwards.get(selection_key)
        if outputs is None:
            outputs = _forward_cached(
                selection["packed"],
                reliability_key=reliability_key,
                route_key=route_key,
            )
            forwards[selection_key] = outputs
            context["forward_count"] = int(context["forward_count"]) + 1

        total_rows = int(selection["total_rows"])
        values = []
        segment_by_item_id = selection["segment_by_item_id"]
        for item in items:
            segment = segment_by_item_id.get(id(item))
            if segment is None:
                raise RuntimeError(
                    "Calibration objective requested an item outside its packed stage"
                )
            start, end = segment
            item_outputs = _slice_packed_outputs(
                outputs, int(start), int(end), total_rows
            )
            values.append(loss_fn(item, item_outputs))
        return torch.stack(values).mean()

    def _normalized_scenario_weights(raw: Any) -> dict[str, float]:
        raw = raw or {"clean": 0.5, "perturb": 0.5}
        if not isinstance(raw, dict):
            raise ValueError(
                "fusion.routing.scenario_objective_weights must be a mapping"
            )
        unknown = set(raw) - {"clean", "perturb"}
        if unknown:
            raise ValueError(
                "Unsupported fusion.routing.scenario_objective_weights keys: "
                f"{sorted(unknown)}"
            )
        clean_weight = float(raw.get("clean", 0.5))
        perturb_weight = float(raw.get("perturb", 0.5))
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (clean_weight, perturb_weight)
        ):
            raise ValueError(
                "fusion.routing.scenario_objective_weights values must be "
                "finite and non-negative"
            )
        total = clean_weight + perturb_weight
        if total <= 0.0:
            raise ValueError(
                "fusion.routing.scenario_objective_weights must have positive mass"
            )
        return {
            "clean": clean_weight / total,
            "perturb": perturb_weight / total,
        }

    routing_scenario_weights = _normalized_scenario_weights(
        (((cfg.get("fusion", {}) or {}).get("routing", {}) or {}).get(
            "scenario_objective_weights"
        ))
    )

    def _balanced_group_loss(
        group: dict[str, Any],
        loss_fn,
        context: dict[str, Any],
        *,
        weights: dict[str, float] | None = None,
        reliability_key: str | None = None,
        route_key: str | None = None,
        requires_forward: bool = True,
    ) -> torch.Tensor:
        clean_loss = _mean_cached_loss(
            group["clean"],
            loss_fn,
            context,
            reliability_key=reliability_key,
            route_key=route_key,
            requires_forward=requires_forward,
        )
        scenario_items = group.get("scenario") or []
        if not scenario_items:
            return clean_loss
        scenario_loss = _mean_cached_loss(
            scenario_items,
            loss_fn,
            context,
            reliability_key=reliability_key,
            route_key=route_key,
            requires_forward=requires_forward,
        )
        resolved = weights or {"clean": 0.5, "perturb": 0.5}
        # Scenario families are averaged separately so grid density cannot
        # change their mass. The clean/perturb deployment prior is an
        # explicit, pre-registered configuration and can be ablated.
        return (
            float(resolved["clean"]) * clean_loss
            + float(resolved["perturb"]) * scenario_loss
        )

    def _optimize_stage(
        stage_name: str,
        stage_parameters: list[torch.nn.Parameter],
        objective_groups: list[dict[str, Any]],
        objective_fn,
        *,
        config_stage_name: str | None = None,
        global_objective: bool = False,
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
                "decision_forward_evaluations": 0,
                "lightweight_forward_evaluations": 0,
                "converged": False,
                "objective_groups": [],
            }
        optimization = _stage_optimization(config_stage_name or stage_name)
        _set_trainable(stage_parameters)
        initial_parameters = [
            parameter.detach().clone() for parameter in stage_parameters
        ]
        best_parameters = [value.clone() for value in initial_parameters]
        step_losses: list[float] = []
        total_steps = 0
        best_step = 0
        best_loss = float("inf")
        previous_loss: float | None = None
        stable_steps = 0
        converged = False
        stop_reason = "max_steps"
        max_grad_norm = 0.0
        final_grad_inf_norm = float("inf")
        function_evaluations = 0
        decision_forward_evaluations = 0
        lightweight_forward_evaluations = 0

        packed_selection_cache: dict[Any, dict[str, Any]] = {}
        persistent_objective_cache: dict[str, Any] = {}
        all_objective_items: list[dict[str, Any]] = []
        seen_objective_item_ids: set[int] = set()
        for group in objective_groups:
            for item in [*(group.get("clean") or []), *(group.get("scenario") or [])]:
                item_id = id(item)
                if item_id in seen_objective_item_ids:
                    continue
                seen_objective_item_ids.add(item_id)
                all_objective_items.append(item)
        if not all_objective_items:
            raise RuntimeError(f"Post-hoc stage {stage_name} has no cached source items")

        def _objective() -> torch.Tensor:
            nonlocal decision_forward_evaluations
            nonlocal lightweight_forward_evaluations
            # Packed inputs persist across steps, while differentiable outputs
            # live for exactly one objective evaluation.  In particular, the
            # clean forward is shared by all scenario groups without retaining
            # an autograd graph across optimizer steps or LBFGS closures.
            context = {
                "selections": packed_selection_cache,
                "forwards": {},
                "forward_count": 0,
                "lightweight_forward_count": 0,
                "all_items": all_objective_items,
                "persistent": persistent_objective_cache,
            }
            if global_objective:
                objective = objective_fn(objective_groups, context)
            else:
                group_losses = [
                    objective_fn(group, context) for group in objective_groups
                ]
                objective = torch.stack(group_losses).mean()
            decision_forward_evaluations += int(context["forward_count"])
            lightweight_forward_evaluations += int(
                context["lightweight_forward_count"]
            )
            if not bool(torch.isfinite(objective.detach()).all().item()):
                raise FloatingPointError(
                    f"Non-finite {stage_name} loss"
                )
            return objective

        def _raw_gradient_norms() -> tuple[float, float]:
            squared = 0.0
            infinity = 0.0
            for parameter in stage_parameters:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError(
                        f"Non-finite {stage_name} gradient"
                    )
                squared += float(gradient.square().sum().item())
                if gradient.numel():
                    infinity = max(infinity, float(gradient.abs().max().item()))
            return math.sqrt(squared), infinity

        if optimization["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                stage_parameters,
                lr=optimization["lr"],
                weight_decay=optimization["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=optimization["lr_scheduler_factor"],
                patience=optimization["lr_scheduler_patience"],
                min_lr=optimization["min_lr"],
            )
        else:
            # One accepted quasi-Newton iteration per outer call keeps the
            # trajectory and convergence diagnostics interpretable.  The
            # strong-Wolfe closure may evaluate several trial points.
            optimizer = torch.optim.LBFGS(
                stage_parameters,
                lr=optimization["lr"],
                max_iter=1,
                history_size=optimization["lbfgs_history_size"],
                tolerance_grad=optimization["gradient_tolerance"],
                tolerance_change=optimization["convergence_tolerance"],
                line_search_fn="strong_wolfe",
            )
            scheduler = None

        for step in range(0, optimization["max_steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = _objective()
            function_evaluations += 1
            step_loss = float(loss.detach().item())
            step_losses.append(step_loss)
            if step_loss < best_loss:
                best_loss = step_loss
                best_step = step
                best_parameters = [
                    parameter.detach().clone() for parameter in stage_parameters
                ]

            if previous_loss is not None and step >= optimization["min_steps"]:
                relative_change = abs(previous_loss - step_loss) / max(
                    1.0e-12, abs(previous_loss)
                )
                stable_steps = (
                    stable_steps + 1
                    if relative_change <= optimization["convergence_tolerance"]
                    else 0
                )
                if stable_steps >= optimization["convergence_patience"]:
                    converged = True
                    stop_reason = "objective_plateau"
            if optimization["optimizer"] == "adam":
                # Evaluate convergence with the gradient at the current
                # parameters. Reusing the previous pre-update gradient is
                # stale under Adam momentum and can stop one iterate early.
                loss.backward()
                grad_norm_value, final_grad_inf_norm = _raw_gradient_norms()
                max_grad_norm = max(max_grad_norm, grad_norm_value)
            if (
                step >= optimization["min_steps"]
                and final_grad_inf_norm <= optimization["gradient_tolerance"]
            ):
                converged = True
                stop_reason = "gradient_tolerance"
            if (
                step == 0
                or step % optimization["log_every"] == 0
                or converged
                or step == optimization["max_steps"]
            ):
                logger.info(
                    "posthoc_calibration stage=%s step=%s loss=%.6f best=%.6f stable=%s",
                    stage_name,
                    step,
                    step_loss,
                    best_loss,
                    stable_steps,
                )
            if converged or step == optimization["max_steps"]:
                break

            if optimization["optimizer"] == "adam":
                torch.nn.utils.clip_grad_norm_(
                    stage_parameters, optimization["grad_clip"]
                )
                optimizer.step()
                assert scheduler is not None
                scheduler.step(step_loss)
            else:
                closure_evaluations = 0

                def _closure() -> torch.Tensor:
                    nonlocal closure_evaluations
                    nonlocal function_evaluations
                    nonlocal max_grad_norm
                    nonlocal final_grad_inf_norm
                    optimizer.zero_grad(set_to_none=True)
                    closure_loss = _objective()
                    closure_loss.backward()
                    closure_evaluations += 1
                    function_evaluations += 1
                    grad_norm_value, grad_inf_norm = _raw_gradient_norms()
                    max_grad_norm = max(max_grad_norm, grad_norm_value)
                    final_grad_inf_norm = grad_inf_norm
                    return closure_loss

                optimizer.step(_closure)
                if closure_evaluations <= 0:
                    raise RuntimeError(
                        f"LBFGS stage {stage_name} performed no closure evaluation"
                    )
            total_steps += 1
            previous_loss = step_loss
        if total_steps == 0 and not converged:
            raise RuntimeError(
                f"Post-hoc calibration stage {stage_name} produced no valid step"
            )
        with torch.no_grad():
            for parameter, best_value in zip(stage_parameters, best_parameters):
                parameter.copy_(best_value)
        # LBFGS can finish on a rejected line-search trial while the selected
        # parameters are restored from an earlier accepted minimum. Recompute
        # at the exact restored state so robust-family diagnostics describe the
        # parameters that will actually be deployed.
        restored_loss_tensor = _objective()
        function_evaluations += 1
        restored_best_loss = float(restored_loss_tensor.detach().item())
        optimizer.zero_grad(set_to_none=True)
        restored_loss_tensor.backward()
        restored_grad_norm, final_grad_inf_norm = _raw_gradient_norms()
        max_grad_norm = max(max_grad_norm, restored_grad_norm)
        optimizer.zero_grad(set_to_none=True)

        provisional_stop_reason = stop_reason
        plateau_detected = provisional_stop_reason == "objective_plateau"

        converged, stop_reason = _resolve_restored_stage_convergence(
            provisional_stop_reason=provisional_stop_reason,
            final_grad_inf_norm=final_grad_inf_norm,
            gradient_tolerance=optimization["gradient_tolerance"],
        )

        stopped_early = total_steps < optimization["max_steps"]

        if not math.isclose(
            restored_best_loss,
            best_loss,
            rel_tol=1.0e-5,
            abs_tol=1.0e-7,
        ):
            raise RuntimeError(
                f"Post-hoc stage {stage_name} best-state restore changed its "
                f"objective: tracked={best_loss:.9g}, "
                f"restored={restored_best_loss:.9g}"
            )
        parameter_delta_l2 = math.sqrt(
            sum(
                float((current.detach() - initial).square().sum().item())
                for current, initial in zip(stage_parameters, initial_parameters)
            )
        )
        relative_loss_improvement = (
            step_losses[0] - best_loss
        ) / max(1.0e-12, abs(step_losses[0]))
        if optimization["require_convergence"] and not converged:
            raise RuntimeError(
                f"Post-hoc calibration stage {stage_name} did not satisfy the "
                f"restored-state convergence requirement; "
                f"provisional_stop_reason={provisional_stop_reason}, "
                f"final_stop_reason={stop_reason}, "
                f"final_grad_inf_norm={final_grad_inf_norm:.6g}, "
                f"gradient_tolerance={optimization['gradient_tolerance']:.6g}, "
                f"best_loss={best_loss:.9g}, "
                f"total_steps={total_steps}, "
                f"max_steps={optimization['max_steps']}"
            )
        if (
            optimization["require_convergence"]
            and relative_loss_improvement
            < optimization["minimum_relative_improvement"]
        ):
            raise RuntimeError(
                f"Post-hoc calibration stage {stage_name} converged without the "
                "pre-registered minimum improvement; "
                f"relative_improvement={relative_loss_improvement:.6g}"
            )
        result = {
            "enabled": True,
            "name": stage_name,
            "epochs_ran": total_steps,
            "best_epoch": best_step,
            "stopped_early": bool(stopped_early),
            "plateau_detected": bool(plateau_detected),
            "restored_gradient_converged": bool(converged),
            "provisional_stop_reason": provisional_stop_reason,
            "converged": bool(converged),
            "stop_reason": stop_reason,
            "parameter_selection": "minimum_training_objective",
            "losses": step_losses,
            "initial_loss": step_losses[0],
            "final_loss": best_loss,
            "restored_best_loss": restored_best_loss,
            "total_steps": total_steps,
            "max_steps": optimization["max_steps"],
            "min_steps": optimization["min_steps"],
            "convergence_patience": optimization["convergence_patience"],
            "convergence_tolerance": optimization["convergence_tolerance"],
            "max_grad_norm": max_grad_norm,
            "final_grad_inf_norm": final_grad_inf_norm,
            "function_evaluations": function_evaluations,
            "decision_forward_evaluations": decision_forward_evaluations,
            "lightweight_forward_evaluations": lightweight_forward_evaluations,
            "parameter_delta_l2": parameter_delta_l2,
            "relative_loss_improvement": relative_loss_improvement,
            "optimization": optimization,
            "objective_groups": [str(group["name"]) for group in objective_groups],
        }
        summary_metadata = getattr(objective_fn, "summary_metadata", None)
        if callable(summary_metadata):
            metadata = summary_metadata()
            if metadata:
                result["objective_diagnostics"] = metadata
        if (config_stage_name or stage_name) in {
            "routing_distribution",
            "routing_risk",
        }:
            fitted_component = (
                "route"
                if (config_stage_name or stage_name) == "routing_distribution"
                else "risk"
            )
            result["fitted_component"] = fitted_component
            router = getattr(discount_fusion, "opinion_router", None)
            diagnostic_builder = getattr(
                router, "effective_parameter_diagnostics", None
            )
            if callable(diagnostic_builder):
                all_diagnostics = diagnostic_builder()
                diagnostics = {
                    name: value
                    for name, value in all_diagnostics.items()
                    if name.startswith(f"{fitted_component}_")
                }
                result["effective_parameter_diagnostics"] = diagnostics
                detail_builder = getattr(
                    router, "effective_parameter_details", None
                )
                if callable(detail_builder):
                    all_details = detail_builder()
                    result["effective_parameter_details"] = {
                        name: value
                        for name, value in all_details.items()
                        if name.startswith(f"{fitted_component}_")
                    }
                routing_options = (
                    (cfg.get("fusion", {}) or {}).get("routing", {}) or {}
                )
                max_effective_parameter = float(
                    routing_options.get("max_effective_parameter", 25.0)
                )
                if (
                    not math.isfinite(max_effective_parameter)
                    or max_effective_parameter <= 0.0
                ):
                    raise ValueError(
                        "fusion.routing.max_effective_parameter must be finite "
                        "and positive"
                    )
                excessive = {
                    name: value
                    for name, value in diagnostics.items()
                    if float(value) > max_effective_parameter
                }
                if excessive:
                    raise RuntimeError(
                        f"Post-hoc stage {stage_name} learned an unstable "
                        f"effective scale: {excessive}"
                    )
        return result

    reliability_parameters = (
        list(discount_fusion.reliability_calibration_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "reliability_calibration_parameters")
        else []
    )
    routing_distribution_parameters = (
        list(discount_fusion.routing_distribution_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "routing_distribution_parameters")
        else []
    )
    routing_risk_parameters = (
        list(discount_fusion.routing_risk_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "routing_risk_parameters")
        else []
    )
    routing_parameters = [
        *routing_distribution_parameters,
        *routing_risk_parameters,
    ]
    configured_risk_target = str(
        (((cfg.get("fusion", {}) or {}).get("routing", {}) or {}).get(
            "risk_target", "threshold_malware_false_negative"
        ))
    ).strip().lower()
    if (
        routing_risk_parameters
        and configured_risk_target.startswith("threshold_")
        and not bool(cross_fitting["enabled"])
    ):
        raise ValueError(
            f"fusion.routing.risk_target={configured_risk_target!r} requires "
            "strict upstream cross-fitting before the risk stage"
        )
    reliability_cfg = dict(
        ((cfg.get("fusion", {}) or {}).get("reliability_calibration", {}) or {})
    )
    reliability_method = normalize_reliability_calibration_method(
        reliability_cfg.get("method", MONOTONIC_CORRECTNESS_METHOD)
    )
    configured_i1_loss = str(
        reliability_cfg.get("loss", "bce")
    ).strip().lower()
    if (
        reliability_method == MONOTONIC_CORRECTNESS_METHOD
        and configured_i1_loss != "bce"
    ):
        raise ValueError(
            "monotonic_correctness I1 uses one global BCE objective; "
            "fusion.reliability_calibration.loss must be 'bce'"
        )
    raw_i1_scenario_weights = reliability_cfg.get(
        "scenario_objective_weights",
        {"clean": 0.50, "perturb": 0.50},
    )
    if not isinstance(raw_i1_scenario_weights, dict):
        raise ValueError(
            "fusion.reliability_calibration.scenario_objective_weights must "
            "be a mapping"
        )
    unknown_i1_scenario_weights = sorted(
        set(raw_i1_scenario_weights) - {"clean", "perturb"}
    )
    if unknown_i1_scenario_weights:
        raise ValueError(
            "fusion.reliability_calibration.scenario_objective_weights "
            f"contains unsupported keys: {unknown_i1_scenario_weights}"
        )
    i1_scenario_weights = {
        name: float(raw_i1_scenario_weights.get(name, 0.0))
        for name in ("clean", "perturb")
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in i1_scenario_weights.values()
    ) or not math.isclose(
        sum(i1_scenario_weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            "fusion.reliability_calibration.scenario_objective_weights must "
            "contain non-negative clean/perturb masses summing to one"
        )
    temperature_fit_source = str(
        reliability_cfg.get("temperature_fit_source", "clean_only")
    ).strip().lower()
    if temperature_fit_source not in {
        "clean_only",
        "clean_plus_branch_local_partial",
    }:
        raise ValueError(
            "fusion.reliability_calibration.temperature_fit_source must be "
            "'clean_only' or 'clean_plus_branch_local_partial'"
        )
    reliability_branches = tuple(
        str(name).lower()
        for name in getattr(
            discount_fusion,
            "reliability_calibration_branches",
            ("api", "graph", "manifest"),
        )
    )
    routing_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
    routing_cfg.setdefault("routing", {})["prediction_loss_weight"] = 1.0
    routing_cfg.setdefault("routing", {})["risk_loss_weight"] = 0.0
    risk_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
    risk_cfg.setdefault("routing", {})["prediction_loss_weight"] = 0.0
    risk_cfg.setdefault("routing", {})["risk_loss_weight"] = 1.0

    def _clean_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in items if item["scenario_group"] == "clean"]

    def _nonclean_by_group(
        items: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            group_name = str(item["scenario_group"])
            if group_name != "clean":
                result.setdefault(group_name, []).append(item)
        return result

    def _build_reliability_groups(
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clean = _clean_items(items)
        if not clean:
            raise RuntimeError("I1 fitting selection has no clean samples")
        nonclean = _nonclean_by_group(items)
        groups: list[dict[str, Any]] = []
        for branch in reliability_branches:
            use_partial_scenarios = (
                reliability_method != TEMPERATURE_SCALING_CONFIDENCE_METHOD
                or temperature_fit_source == "clean_plus_branch_local_partial"
            )
            partial_scenarios = (
                [
                    item
                    for scenario_items in nonclean.values()
                    for item in scenario_items
                    if branch in item.get("reliability_branches", ())
                ]
                if use_partial_scenarios
                else []
            )
            if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD:
                source_label = (
                    "clean_plus_branch_local_partial"
                    if partial_scenarios
                    else "clean_only"
                )
            else:
                source_label = (
                    "clean_plus_branch_local_partial"
                    if partial_scenarios
                    else "clean"
                )
            groups.append(
                {
                    "name": f"{branch}:{source_label}",
                    "branch": branch,
                    "clean": clean,
                    "scenario": partial_scenarios,
                }
            )
        return groups

    def _build_routing_groups(
        items: list[dict[str, Any]],
        *,
        prefix: str,
    ) -> list[dict[str, Any]]:
        clean = _clean_items(items)
        if not clean:
            raise RuntimeError(f"{prefix} fitting selection has no clean samples")
        nonclean = _nonclean_by_group(items)
        return [
            {
                "name": f"{prefix}:{group_name}",
                "objective_family": group_name,
                "clean": clean,
                "scenario": scenario_items,
                "mechanisms": [
                    {"name": group_name, "items": list(scenario_items)}
                ],
            }
            for group_name, scenario_items in sorted(nonclean.items())
        ] or [{"name": f"{prefix}:clean", "clean": clean, "scenario": []}]

    i1_objective_diagnostics: dict[str, Any] = {}

    def _reliability_objective(
        groups_or_group: list[dict[str, Any]] | dict[str, Any],
        context: dict[str, Any],
    ) -> torch.Tensor:
        if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD:
            if not isinstance(groups_or_group, dict):
                raise RuntimeError(
                    "temperature-scaling I1 expects one objective group at a time"
                )
            group = groups_or_group
            calibrator = getattr(discount_fusion, "reliability_calibrator", None)
            branch_nll = getattr(calibrator, "branch_nll", None)
            if not callable(branch_nll):
                raise RuntimeError(
                    "temperature_scaling_confidence requires a calibrator with "
                    "a branch_nll objective"
                )

            def _temperature_nll(
                cached: dict[str, Any], _outputs: dict[str, Any] | None
            ) -> torch.Tensor:
                branch = str(group["branch"])
                alive = reliability_alive_mask(
                    cached["availability"], branch
                )
                return branch_nll(
                    branch,
                    cached["branch_logits"][branch],
                    cached["labels"],
                    alive,
                )

            return _balanced_group_loss(
                group,
                _temperature_nll,
                context,
                weights=i1_scenario_weights,
                requires_forward=False,
            )

        if reliability_method != MONOTONIC_CORRECTNESS_METHOD:
            raise RuntimeError(
                f"Unsupported I1 method {reliability_method!r}"
            )
        if isinstance(groups_or_group, dict):
            raise RuntimeError("monotonic I1 requires the global packed objective")
        objective_groups = list(groups_or_group)
        if not objective_groups:
            raise RuntimeError("monotonic I1 objective contains no branch groups")

        calibrator = getattr(discount_fusion, "reliability_calibrator", None)
        branch_modules = getattr(calibrator, "branches", None)
        if branch_modules is None:
            raise RuntimeError("Monotonic I1 fitting requires branch calibrators")

        signature = tuple(id(item) for item in context["all_items"])
        objective_signature = tuple(
            (
                str(group["name"]),
                str(group["branch"]),
                tuple(id(item) for item in (group.get("clean") or [])),
                tuple(id(item) for item in (group.get("scenario") or [])),
            )
            for group in objective_groups
        )
        persistent = context["persistent"]
        static = persistent.get("i1_global_static")
        if static is None:
            selection = _pack_cached_selection(
                context["all_items"],
                reliability_key=None,
                route_key=None,
            )
            with torch.no_grad():
                packed_outputs = _forward_cached(selection["packed"])
            context["forward_count"] = int(context["forward_count"]) + 1

            total_rows = int(selection["total_rows"])
            required_keys = tuple(
                key
                for branch in reliability_branches
                for key in (
                    f"reliability_features_{branch}",
                    f"alive_{branch}",
                )
            )
            missing = [
                key
                for key in required_keys
                if not isinstance(packed_outputs.get(key), torch.Tensor)
            ]
            if missing:
                raise RuntimeError(
                    "I1 feature precomputation is missing calibrator outputs: "
                    f"{missing}"
                )
            for branch in reliability_branches:
                features = packed_outputs[f"reliability_features_{branch}"]
                alive = packed_outputs[f"alive_{branch}"]
                if tuple(features.shape) != (total_rows, 3):
                    raise RuntimeError(
                        f"I1 reliability_features_{branch} must have shape "
                        f"[{total_rows}, 3], got {tuple(features.shape)}"
                    )
                if alive.numel() != total_rows:
                    raise RuntimeError(
                        f"I1 alive_{branch} must contain {total_rows} rows"
                    )

            global_segment_by_id = {
                id(item): (int(start), int(end))
                for item, (start, end) in zip(
                    context["all_items"], selection["segments"]
                )
            }
            group_count = len(objective_groups)
            branch_static: dict[str, dict[str, Any]] = {}
            branch_diagnostics: dict[str, Any] = {}

            for group in objective_groups:
                branch = str(group["branch"]).strip().lower()
                if branch in branch_static:
                    raise RuntimeError(
                        f"monotonic I1 contains duplicate branch group {branch!r}"
                    )
                if branch not in branch_modules:
                    raise RuntimeError(
                        f"monotonic I1 has no calibrator for branch {branch!r}"
                    )

                clean_items = list(group.get("clean") or [])
                perturb_items = list(group.get("scenario") or [])
                branch_items = [*clean_items, *perturb_items]
                branch_item_ids = [id(item) for item in branch_items]
                if not branch_items:
                    raise RuntimeError(
                        f"monotonic I1 branch {branch!r} has no sources"
                    )
                if len(set(branch_item_ids)) != len(branch_item_ids):
                    raise RuntimeError(
                        f"monotonic I1 branch {branch!r} contains duplicate sources"
                    )
                if any(item_id not in global_segment_by_id for item_id in branch_item_ids):
                    raise RuntimeError(
                        f"monotonic I1 branch {branch!r} references an unknown source"
                    )

                clean_mass = float(i1_scenario_weights["clean"])
                perturb_mass = float(i1_scenario_weights["perturb"])
                if clean_mass > 0.0 and not clean_items:
                    raise RuntimeError(
                        f"I1 branch {branch!r} has positive clean mass but no clean source"
                    )
                if perturb_mass > 0.0 and not perturb_items:
                    raise RuntimeError(
                        f"I1 branch {branch!r} has positive perturb mass but no "
                        "branch-local partial-degradation source"
                    )

                source_masses = {id(item): 0.0 for item in branch_items}
                if clean_items:
                    per_clean_source = clean_mass / float(len(clean_items))
                    for item in clean_items:
                        if str(item.get("perturb_type") or "clean").lower() != "clean":
                            raise RuntimeError(
                                f"I1 branch {branch!r} clean side contains a "
                                "non-clean source"
                            )
                        source_masses[id(item)] = per_clean_source

                perturb_hierarchy: dict[
                    str, dict[float, list[dict[str, Any]]]
                ] = {}
                for item in perturb_items:
                    perturb_type = str(
                        item.get("perturb_type") or ""
                    ).strip().lower()
                    scenario_group = str(
                        item.get("scenario_group") or ""
                    ).strip().lower()
                    if (
                        not perturb_type
                        or perturb_type == "clean"
                        or perturb_type.endswith("_missing")
                        or scenario_group == "missing"
                    ):
                        raise RuntimeError(
                            "Missing/clean sources must not enter the I1 "
                            f"perturbation objective ({branch}/{perturb_type})"
                        )
                    expected_branches = RELIABILITY_CALIBRATION_PERTURBATIONS.get(
                        perturb_type
                    )
                    if expected_branches is None or branch not in expected_branches:
                        raise RuntimeError(
                            f"I1 source {perturb_type!r} is not a registered "
                            f"partial-degradation view for branch {branch!r}"
                        )
                    strength = float(item.get("strength", 0.0))
                    if not math.isfinite(strength):
                        raise RuntimeError(
                            f"I1 source {perturb_type!r} has non-finite strength"
                        )
                    perturb_hierarchy.setdefault(perturb_type, {}).setdefault(
                        strength, []
                    ).append(item)

                if perturb_hierarchy:
                    per_mechanism = perturb_mass / float(len(perturb_hierarchy))
                    for strengths in perturb_hierarchy.values():
                        per_strength = per_mechanism / float(len(strengths))
                        for strength_items in strengths.values():
                            per_source = per_strength / float(len(strength_items))
                            for item in strength_items:
                                source_masses[id(item)] = per_source

                total_source_mass = sum(source_masses.values())
                if not math.isclose(
                    total_source_mass,
                    1.0,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                ):
                    raise RuntimeError(
                        f"I1 branch {branch!r} source masses sum to "
                        f"{total_source_mass:.12g}, expected one"
                    )

                features_parts: list[torch.Tensor] = []
                alive_parts: list[torch.Tensor] = []
                labels_parts: list[torch.Tensor] = []
                logits_parts: list[torch.Tensor] = []
                branch_segments: list[tuple[int, int]] = []
                offset = 0
                for item in branch_items:
                    start, end = global_segment_by_id[id(item)]
                    rows = end - start
                    features_parts.append(
                        packed_outputs[
                            f"reliability_features_{branch}"
                        ][start:end].detach()
                    )
                    alive_parts.append(
                        packed_outputs[f"alive_{branch}"][start:end].detach()
                    )
                    labels_parts.append(item["labels"])
                    logits_parts.append(item["branch_logits"][branch])
                    branch_segments.append((offset, offset + rows))
                    offset += rows

                features = torch.cat(features_parts, dim=0)
                alive = torch.cat(alive_parts, dim=0).view(-1).float()
                correctness = reliability_correctness_target(
                    torch.cat(logits_parts, dim=0),
                    torch.cat(labels_parts, dim=0),
                )
                row_weights = _compile_posthoc_row_weights(
                    branch_items,
                    branch_segments,
                    source_masses,
                    alive.gt(0.0),
                ) / float(group_count)

                source_details: list[dict[str, Any]] = []
                for item, (start, end) in zip(branch_items, branch_segments):
                    source_details.append(
                        {
                            "name": str(item.get("scenario_name") or "unnamed"),
                            "mechanism": str(
                                item.get("perturb_type") or "clean"
                            ).strip().lower(),
                            "strength": float(item.get("strength", 0.0)),
                            "configured_mass": float(source_masses[id(item)]),
                            "rows": int(end - start),
                            "alive_rows": int(
                                alive[start:end].gt(0.0).sum().item()
                            ),
                        }
                    )

                branch_static[branch] = {
                    "features": features,
                    "alive": alive,
                    "correctness": correctness,
                    "row_weights": row_weights,
                }
                branch_diagnostics[branch] = {
                    "clean_source_count": len(clean_items),
                    "perturb_source_count": len(perturb_items),
                    "perturbation_hierarchy": {
                        mechanism: {
                            f"{strength:g}": len(strength_items)
                            for strength, strength_items in sorted(strengths.items())
                        }
                        for mechanism, strengths in sorted(
                            perturb_hierarchy.items()
                        )
                    },
                    "effective_alive_objective_mass": float(
                        row_weights.sum().item() * float(group_count)
                    ),
                    "sources": source_details,
                }

            static = {
                "signature": signature,
                "objective_signature": objective_signature,
                "branches": branch_static,
                "diagnostics": {
                    "semantics": (
                        "global_source_balanced_branch_correctness_bce"
                    ),
                    "loss": "bce",
                    "scenario_objective_weights": dict(i1_scenario_weights),
                    "missing_sources_excluded": True,
                    "branch_reduction": "equal_mean",
                    "perturb_reduction": (
                        "equal_mechanism_then_equal_strength_then_equal_source"
                    ),
                    "branches": branch_diagnostics,
                },
            }
            persistent["i1_global_static"] = static
            i1_objective_diagnostics.clear()
            i1_objective_diagnostics.update(
                copy.deepcopy(static["diagnostics"])
            )
        elif (
            static["signature"] != signature
            or static["objective_signature"] != objective_signature
        ):
            raise RuntimeError(
                "Monotonic I1 persistent cache was reused with different sources"
            )

        weighted_branch_losses: list[torch.Tensor] = []
        for branch, branch_data in static["branches"].items():
            module = branch_modules[branch]
            raw_logit = module.forward_logit(branch_data["features"])
            alive = branch_data["alive"].to(raw_logit)
            reliability = torch.sigmoid(raw_logit) * alive
            per_row = reliability_per_sample_loss(
                reliability,
                raw_logit,
                branch_data["correctness"],
                loss_type="bce",
            )
            row_weights = branch_data["row_weights"].to(per_row)
            if per_row.numel() != row_weights.numel():
                raise RuntimeError(
                    f"I1 branch {branch!r} loss/weight row counts disagree"
                )
            weighted_branch_losses.append((per_row * row_weights).sum())

        context["lightweight_forward_count"] = int(
            context["lightweight_forward_count"]
        ) + len(weighted_branch_losses)
        return torch.stack(weighted_branch_losses).sum()

    def _i1_summary_metadata() -> dict[str, Any]:
        return copy.deepcopy(i1_objective_diagnostics)

    _reliability_objective.summary_metadata = _i1_summary_metadata

    def _fit_reliability_stage(
        stage_name: str,
        objective_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if reliability_method not in {
            MONOTONIC_CORRECTNESS_METHOD,
            TEMPERATURE_SCALING_CONFIDENCE_METHOD,
        }:
            raise RuntimeError(
                f"Unsupported I1 method {reliability_method!r}"
            )
        return _optimize_stage(
            stage_name,
            reliability_parameters,
            objective_groups,
            _reliability_objective,
            config_stage_name="reliability",
            global_objective=(
                reliability_method == MONOTONIC_CORRECTNESS_METHOD
            ),
        )

    def _routing_objective(reliability_key: str | None):
        # Encoder opinions, I1 reliability and availability are immutable while
        # pi is fitted. Materialize the exact router inputs once and optimize
        # only the conditional-mixture NLL. Clean/perturb masses are explicit
        # protocol weights; no label-derived oracle or soft-worst objective is
        # part of the final method.
        route_options = routing_cfg.get("routing", {}) or {}
        route_enabled = bool(route_options.get("enabled", False)) and bool(
            route_options.get("posthoc_refine", True)
        )
        route_prediction_weight = float(
            route_options.get("prediction_loss_weight", 1.0)
        )
        route_effective_l2 = float(
            route_options.get("route_effective_l2", 0.0)
        )
        if not math.isfinite(route_effective_l2) or route_effective_l2 < 0.0:
            raise ValueError(
                "fusion.routing.route_effective_l2 must be finite and non-negative"
            )
        for name, value in (
            ("prediction_loss_weight", route_prediction_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"fusion.routing.{name} must be finite and non-negative")
        if (
            route_enabled
            and routing_distribution_parameters
            and route_prediction_weight <= 0.0
        ):
            raise ValueError(
                "The final I2 route requires fusion.routing."
                "prediction_loss_weight > 0"
            )

        static_signature: tuple[int, ...] | None = None
        static_inputs: dict[str, Any] = {}

        def _ensure_static_route_inputs(
            objective_groups: list[dict[str, Any]],
            context: dict[str, Any],
        ) -> None:
            nonlocal static_signature
            all_items = context["all_items"]
            signature = tuple(id(item) for item in all_items)
            if static_signature is not None:
                if signature != static_signature:
                    raise RuntimeError(
                        "Route-stage cached source identity changed during optimization"
                    )
                return

            selection = _pack_cached_selection(
                all_items,
                reliability_key=reliability_key,
                route_key=None,
            )
            with torch.no_grad():
                packed_outputs = _forward_cached(
                    selection["packed"],
                    reliability_key=reliability_key,
                    route_key=None,
                )
            context["forward_count"] = int(context["forward_count"]) + 1

            required_keys = tuple(
                key
                for name in ("api", "graph", "manifest")
                for key in (
                    f"routing_input_probability_{name}",
                    f"routing_input_reliability_{name}",
                    f"routing_input_alive_{name}",
                )
            )
            missing = [
                key
                for key in required_keys
                if not isinstance(packed_outputs.get(key), torch.Tensor)
            ]
            if missing:
                raise RuntimeError(
                    "Route input precomputation is missing routed outputs: "
                    f"{missing}"
                )
            router = getattr(discount_fusion, "opinion_router", None)
            prepare_route_inputs = getattr(router, "prepare_route_inputs", None)
            if not callable(prepare_route_inputs):
                raise RuntimeError(
                    "Routing fitting requires a router with prepared-input support"
                )
            branches = ("api", "graph", "manifest")
            with torch.no_grad():
                prepared = prepare_route_inputs(
                    branch_probabilities={
                        name: packed_outputs[
                            f"routing_input_probability_{name}"
                        ].detach()
                        for name in branches
                    },
                    reliability={
                        name: packed_outputs[
                            f"routing_input_reliability_{name}"
                        ].detach()
                        for name in branches
                    },
                    alive={
                        name: packed_outputs[
                            f"routing_input_alive_{name}"
                        ].detach()
                        for name in branches
                    },
                    eps=float(
                        getattr(discount_fusion, "config", {}).get(
                            "min_discount", 1.0e-8
                        )
                    ),
                )
            static_inputs["prepared"] = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in prepared.items()
            }
            static_inputs["learned_active"] = bool(
                getattr(discount_fusion, "calibration_active", False)
            )
            labels = selection["packed"]["labels"].detach().long().view(-1)
            prepared_has_available = prepared.get("has_available")
            if not isinstance(prepared_has_available, torch.Tensor):
                raise RuntimeError("Prepared routing inputs are missing availability")
            static_inputs["labels"] = labels
            source_masses = _compile_posthoc_source_masses(
                objective_groups,
                all_items,
                routing_scenario_weights,
            )
            static_inputs["prediction_row_weights"] = (
                _compile_posthoc_row_weights(
                    all_items,
                    selection["segments"],
                    source_masses,
                    prepared_has_available,
                )
            )
            static_signature = signature

        def _route_outputs(
            objective_groups: list[dict[str, Any]],
            context: dict[str, Any],
        ) -> dict[str, torch.Tensor]:
            _ensure_static_route_inputs(objective_groups, context)
            cached_outputs = context["forwards"].get("route_only_kernel")
            if cached_outputs is not None:
                return cached_outputs
            router = getattr(discount_fusion, "opinion_router", None)
            if router is None:
                raise RuntimeError("Routing fitting requires an opinion router")
            forward_prepared = getattr(router, "forward_prepared", None)
            if not callable(forward_prepared):
                raise RuntimeError(
                    "Routing fitting requires a router with prepared-input support"
                )
            branches = ("api", "graph", "manifest")
            routed = forward_prepared(
                static_inputs["prepared"],
                learned_active=bool(static_inputs["learned_active"]),
                compute_risk=False,
            )
            has_available = routed["has_available"]
            cached_outputs = {
                "routing_active": torch.ones_like(has_available),
                "routing_has_available": has_available,
                "routing_mixture_prob": routed["mixture_probability"],
                "routing_branch_distribution": routed["branch_distribution"],
                "routing_scores": routed["routing_scores"],
            }
            context["forwards"]["route_only_kernel"] = cached_outputs
            context["lightweight_forward_count"] = int(
                context.get("lightweight_forward_count", 0)
            ) + 1
            return cached_outputs

        def _objective(
            objective_groups: list[dict[str, Any]],
            context: dict[str, Any],
        ) -> torch.Tensor:
            packed_outputs = _route_outputs(objective_groups, context)
            static_inputs["last_branch_distribution"] = packed_outputs[
                "routing_branch_distribution"
            ].detach()
            mixture_log_prob = routing_mixture_log_prob(packed_outputs)
            prediction_per_row = F.nll_loss(
                mixture_log_prob,
                static_inputs["labels"],
                reduction="none",
            )
            static_inputs["last_prediction_per_row"] = (
                prediction_per_row.detach()
            )
            data_nll = torch.dot(
                prediction_per_row,
                static_inputs["prediction_row_weights"].to(
                    dtype=prediction_per_row.dtype
                ),
            )
            static_inputs["last_data_nll"] = data_nll.detach()
            objective = route_prediction_weight * data_nll
            router = getattr(discount_fusion, "opinion_router", None)
            regularizer = getattr(router, "route_effective_l2", None)
            if route_effective_l2 > 0.0:
                if not callable(regularizer):
                    raise RuntimeError(
                        "Configured route effective L2 requires router support"
                    )
                objective = objective + route_effective_l2 * regularizer()
            return objective

        def _summary_metadata() -> dict[str, Any]:
            metadata: dict[str, Any] = {
                "objective": "conditional_mixture_nll",
                "prediction_loss_weight": route_prediction_weight,
                "scenario_objective_weights": dict(routing_scenario_weights),
                "route_effective_l2": route_effective_l2,
            }
            data_nll = static_inputs.get("last_data_nll")
            if isinstance(data_nll, torch.Tensor):
                metadata["final_weighted_nll"] = float(data_nll.cpu())
            final_distribution = static_inputs.get("last_branch_distribution")
            if isinstance(final_distribution, torch.Tensor):
                distribution = final_distribution.detach().float().clamp_min(
                    1.0e-12
                )
                entropy = -(
                    distribution * distribution.log()
                ).sum(dim=-1) / math.log(float(distribution.size(-1)))
                maximum = distribution.max(dim=-1).values
                metadata["route_distribution_diagnostics"] = {
                    "mean_normalized_entropy": float(entropy.mean().cpu()),
                    "mean_max_weight": float(maximum.mean().cpu()),
                    "fraction_max_weight_above_0_99": float(
                        maximum.gt(0.99).float().mean().cpu()
                    ),
                }
            return metadata

        setattr(_objective, "summary_metadata", _summary_metadata)

        return _objective

    def _risk_objective(
        reliability_key: str | None,
        route_key: str | None,
    ):
        # With OOF reliability and route fixed, I2's three risk features are
        # immutable.  Materialize them once for the complete stage and optimize
        # only the monotone logistic head thereafter.  The previous path rebuilt
        # opinions, routing and diagnostics on every one of the hundreds of risk
        # optimizer steps even though none of those tensors depended on the risk
        # parameters. Source-normalized masks are likewise compiled once, so
        # the learned head is evaluated exactly once over the packed stage.
        risk_options = risk_cfg.get("routing", {}) or {}
        risk_objective_weight = float(risk_options.get("risk_loss_weight", 1.0))
        risk_effective_l2 = float(risk_options.get("risk_effective_l2", 0.0))
        for name, value in (
            ("risk_loss_weight", risk_objective_weight),
            ("risk_effective_l2", risk_effective_l2),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"fusion.routing.{name} must be finite and non-negative")

        static_inputs: dict[str, Any] = {}
        static_item_signature: tuple[int, ...] | None = None

        def _ensure_static_risk_features(
            objective_groups: list[dict[str, Any]],
            context: dict[str, Any],
        ) -> None:
            nonlocal static_item_signature
            all_items = context["all_items"]
            signature = tuple(id(item) for item in all_items)
            if static_item_signature is not None:
                if signature != static_item_signature:
                    raise RuntimeError(
                        "Risk-stage cached source identity changed during optimization"
                    )
                return

            selection = _pack_cached_selection(
                all_items,
                reliability_key=reliability_key,
                route_key=route_key,
            )
            with torch.no_grad():
                packed_outputs = _forward_cached(
                    selection["packed"],
                    reliability_key=reliability_key,
                    route_key=route_key,
                )
            context["forward_count"] = int(context["forward_count"]) + 1
            feature_keys = (
                "routing_risk_reliability_deficit",
                "routing_risk_decision_boundary_proximity",
                "routing_risk_global_cross_modal_conflict",
            )
            required_keys = (
                *feature_keys,
                "routing_has_available",
                "routing_mixture_prob",
                "uncalibrated_final_log_prob",
            )
            missing = [
                key
                for key in required_keys
                if not isinstance(packed_outputs.get(key), torch.Tensor)
            ]
            if missing:
                raise RuntimeError(
                    "Risk feature precomputation is missing routed outputs: "
                    f"{missing}"
                )
            labels = selection["packed"]["labels"].detach().long().view(-1)
            (
                error_target,
                risk_valid,
                risk_loss_type,
                risk_target_type,
            ) = routing_risk_target(
                packed_outputs,
                labels,
                risk_options,
            )
            source_masses = _compile_posthoc_source_masses(
                objective_groups,
                all_items,
                routing_scenario_weights,
            )
            static_inputs.update(
                {
                    "features": torch.stack(
                        [packed_outputs[key].view(-1) for key in feature_keys],
                        dim=-1,
                    ).detach(),
                    "error_target": error_target,
                    "risk_valid": risk_valid,
                    "risk_loss_type": risk_loss_type,
                    "risk_target_type": risk_target_type,
                    "row_weights": _compile_posthoc_row_weights(
                        all_items,
                        selection["segments"],
                        source_masses,
                        risk_valid,
                    ),
                    "source_items": list(all_items),
                    "source_segments": list(selection["segments"]),
                    "source_masses": dict(source_masses),
                }
            )
            static_item_signature = signature

        def _objective(
            objective_groups: list[dict[str, Any]],
            context: dict[str, Any],
        ) -> torch.Tensor:
            _ensure_static_risk_features(objective_groups, context)
            router = getattr(discount_fusion, "opinion_router", None)
            raw_weights = getattr(router, "raw_risk_feature_weights", None)
            effective_weight_builder = getattr(
                router, "effective_risk_feature_weights", None
            )
            risk_bias = getattr(router, "risk_bias", None)
            if not isinstance(raw_weights, torch.Tensor) or not isinstance(
                risk_bias, torch.Tensor
            ) or not callable(effective_weight_builder):
                raise RuntimeError(
                    "Learned I2 risk fitting requires the monotone risk head"
                )
            features = static_inputs["features"]
            risk_weights = effective_weight_builder().to(
                device=features.device, dtype=features.dtype
            )
            if risk_weights.numel() != features.size(-1):
                raise RuntimeError(
                    "Risk feature width disagrees with the learned risk head"
                )
            risk_logit = risk_bias.to(
                device=features.device, dtype=features.dtype
            ) + (features * risk_weights.view(1, -1)).sum(dim=-1)
            static_inputs["last_risk_logit"] = risk_logit.detach()
            per_row = routing_risk_per_sample_loss(
                torch.sigmoid(risk_logit),
                risk_logit,
                static_inputs["error_target"],
                static_inputs["risk_valid"],
                loss_type=static_inputs["risk_loss_type"],
            )
            data_risk_loss = torch.dot(
                per_row,
                static_inputs["row_weights"].to(dtype=per_row.dtype),
            )
            static_inputs["last_risk_per_row_loss"] = per_row.detach()
            static_inputs["last_data_risk_loss"] = data_risk_loss.detach()
            regularization_loss = data_risk_loss.new_zeros(())
            router_regularizer = getattr(router, "risk_effective_l2", None)
            if risk_effective_l2 > 0.0:
                if not callable(router_regularizer):
                    raise RuntimeError(
                        "Configured risk effective L2 requires router support"
                    )
                regularization_loss = (
                    risk_effective_l2 * router_regularizer()
                )
            static_inputs["last_risk_regularization_loss"] = (
                regularization_loss.detach()
            )
            objective = data_risk_loss + regularization_loss
            scaled_objective = risk_objective_weight * objective
            static_inputs["last_scaled_risk_objective"] = (
                scaled_objective.detach()
            )
            return scaled_objective

        def _summary_metadata() -> dict[str, Any]:
            metadata: dict[str, Any] = {
                "risk_effective_l2": risk_effective_l2,
                "risk_target": static_inputs.get("risk_target_type"),
            }
            risk_logit = static_inputs.get("last_risk_logit")
            if isinstance(risk_logit, torch.Tensor):
                probability = torch.sigmoid(risk_logit.detach().float())
                valid = static_inputs.get("risk_valid")
                target = static_inputs.get("error_target")
                metadata["risk_prediction_diagnostics"] = {
                    "minimum": float(probability.min().cpu()),
                    "mean": float(probability.mean().cpu()),
                    "maximum": float(probability.max().cpu()),
                    "num_valid": int(
                        valid.detach().bool().sum().cpu()
                    )
                    if isinstance(valid, torch.Tensor)
                    else None,
                    "valid_positive_rate": float(
                        target.detach()[valid.detach().bool()].float().mean().cpu()
                    )
                    if isinstance(valid, torch.Tensor)
                    and isinstance(target, torch.Tensor)
                    and bool(valid.detach().bool().any().cpu())
                    else None,
                }
            per_row_loss = static_inputs.get("last_risk_per_row_loss")
            valid = static_inputs.get("risk_valid")
            row_weights = static_inputs.get("row_weights")
            source_items = static_inputs.get("source_items")
            source_segments = static_inputs.get("source_segments")
            source_masses = static_inputs.get("source_masses")
            if all(
                (
                    isinstance(per_row_loss, torch.Tensor),
                    isinstance(valid, torch.Tensor),
                    isinstance(row_weights, torch.Tensor),
                    isinstance(source_items, list),
                    isinstance(source_segments, list),
                    isinstance(source_masses, dict),
                )
            ):
                assert isinstance(per_row_loss, torch.Tensor)
                assert isinstance(valid, torch.Tensor)
                assert isinstance(row_weights, torch.Tensor)
                assert isinstance(source_items, list)
                assert isinstance(source_segments, list)
                assert isinstance(source_masses, dict)
                if len(source_items) != len(source_segments):
                    raise RuntimeError(
                        "Risk source diagnostics disagree on source/segment count"
                    )
                loss_values = per_row_loss.detach().float()
                valid_mask = valid.detach().bool()
                weights = row_weights.detach().float()
                diagnostic_loss_values = loss_values.double()
                diagnostic_weights = weights.double()
                source_rows: list[dict[str, Any]] = []
                family_rows: dict[str, dict[str, Any]] = {}
                contribution_sum = 0.0
                for item, (raw_start, raw_end) in zip(
                    source_items, source_segments
                ):
                    start, end = int(raw_start), int(raw_end)
                    segment_valid = valid_mask[start:end]
                    segment_losses = loss_values[start:end]
                    segment_weights = weights[start:end]
                    num_valid = int(segment_valid.sum().cpu())
                    mean_loss = (
                        float(segment_losses[segment_valid].mean().cpu())
                        if num_valid > 0
                        else None
                    )
                    contribution = float(
                        torch.dot(
                            diagnostic_loss_values[start:end],
                            diagnostic_weights[start:end],
                        ).cpu()
                    )
                    effective_mass = float(
                        diagnostic_weights[start:end].sum().cpu()
                    )
                    configured_mass = float(source_masses[id(item)])
                    contribution_sum += contribution
                    objective_family = str(
                        item.get("objective_family") or "other"
                    )
                    source_row = {
                        "name": str(
                            item.get("scenario_name")
                            or item.get("name")
                            or "source"
                        ),
                        "objective_family": objective_family,
                        "perturb_type": str(
                            item.get("perturb_type") or "clean"
                        ),
                        "strength": float(item.get("strength", 0.0)),
                        "num_rows": end - start,
                        "num_valid": num_valid,
                        "configured_mass": configured_mass,
                        "effective_mass": effective_mass,
                        "mean_risk_loss": mean_loss,
                        "weighted_risk_loss_contribution": contribution,
                    }
                    source_rows.append(source_row)
                    family = family_rows.setdefault(
                        objective_family,
                        {
                            "objective_family": objective_family,
                            "num_sources": 0,
                            "num_rows": 0,
                            "num_valid": 0,
                            "configured_mass": 0.0,
                            "effective_mass": 0.0,
                            "weighted_risk_loss_contribution": 0.0,
                        },
                    )
                    family["num_sources"] += 1
                    family["num_rows"] += end - start
                    family["num_valid"] += num_valid
                    family["configured_mass"] += configured_mass
                    family["effective_mass"] += effective_mass
                    family["weighted_risk_loss_contribution"] += contribution
                for family in family_rows.values():
                    effective_mass = float(family["effective_mass"])
                    family["mean_risk_loss"] = (
                        float(
                            family["weighted_risk_loss_contribution"]
                        )
                        / effective_mass
                        if effective_mass > 0.0
                        else None
                    )
                data_risk_loss = float(
                    static_inputs["last_data_risk_loss"].detach().cpu()
                )
                diagnostic_data_risk_loss = float(
                    torch.dot(
                        diagnostic_loss_values, diagnostic_weights
                    ).cpu()
                )
                if not all(
                    math.isfinite(value)
                    for value in (
                        contribution_sum,
                        data_risk_loss,
                        diagnostic_data_risk_loss,
                    )
                ):
                    raise RuntimeError(
                        "Risk source loss diagnostics contain a non-finite value"
                    )
                if not math.isclose(
                    contribution_sum,
                    diagnostic_data_risk_loss,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                ):
                    raise RuntimeError(
                        "Risk source loss contributions do not reconstruct the "
                        "float64 diagnostic objective: "
                        f"{contribution_sum} != {diagnostic_data_risk_loss}"
                    )
                if not math.isclose(
                    diagnostic_data_risk_loss,
                    data_risk_loss,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-6,
                ):
                    raise RuntimeError(
                        "Risk float64 diagnostic objective materially disagrees "
                        f"with the trained float32 objective: "
                        f"{diagnostic_data_risk_loss} != {data_risk_loss}"
                    )
                metadata["risk_loss_diagnostics"] = {
                    "loss_type": static_inputs.get("risk_loss_type"),
                    "risk_target": static_inputs.get("risk_target_type"),
                    "data_risk_loss": data_risk_loss,
                    "diagnostic_float64_data_risk_loss": (
                        diagnostic_data_risk_loss
                    ),
                    "regularization_loss": float(
                        static_inputs["last_risk_regularization_loss"]
                        .detach()
                        .cpu()
                    ),
                    "scaled_final_objective": float(
                        static_inputs["last_scaled_risk_objective"]
                        .detach()
                        .cpu()
                    ),
                    "risk_loss_weight": risk_objective_weight,
                    "effective_total_mass": float(
                        diagnostic_weights.sum().cpu()
                    ),
                    "sources": source_rows,
                    "objective_families": [
                        family_rows[name] for name in sorted(family_rows)
                    ],
                }
            return metadata

        setattr(_objective, "summary_metadata", _summary_metadata)
        return _objective

    def _parameter_snapshot(
        stage_parameters: list[torch.nn.Parameter],
    ) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in stage_parameters]

    def _restore_parameter_snapshot(
        stage_parameters: list[torch.nn.Parameter],
        snapshot: list[torch.Tensor],
    ) -> None:
        if len(stage_parameters) != len(snapshot):
            raise RuntimeError("Post-hoc parameter snapshot length changed")
        with torch.no_grad():
            for parameter, value in zip(stage_parameters, snapshot):
                parameter.copy_(value)

    def _compact_stage_summary(summary: dict[str, Any]) -> dict[str, Any]:
        compact = {
            key: value
            for key, value in summary.items()
            # Fold-local objective diagnostics include one row for every
            # calibration source. They are useful for the final full-data fit,
            # but repeating them for every inner/outer fold makes the summary
            # needlessly large and hard to inspect.
            if key not in {"losses", "optimization", "objective_diagnostics"}
        }
        phases = compact.get("phases")
        if isinstance(phases, dict):
            compact["phases"] = {
                phase_name: {
                    key: value
                    for key, value in phase.items()
                    if key
                    not in {"losses", "optimization", "objective_diagnostics"}
                }
                for phase_name, phase in phases.items()
                if isinstance(phase, dict)
            }
        return compact

    def _write_cached_prediction(
        selected: list[dict[str, Any]],
        *,
        destination_key: str,
        count_key: str | None,
        reliability_key: str | None,
        route_key: str | None,
        output_builder,
        additional_destinations: tuple[tuple[str, str | None], ...] = (),
    ) -> None:
        with torch.no_grad():
            selection = _pack_cached_selection(
                selected,
                reliability_key=reliability_key,
                route_key=route_key,
            )
            outputs = _forward_cached(
                selection["packed"],
                reliability_key=reliability_key,
                route_key=route_key,
            )
            packed_value = output_builder(outputs).detach().float()
            total_rows = int(selection["total_rows"])
            if packed_value.ndim <= 0 or int(packed_value.size(0)) != total_rows:
                raise RuntimeError(
                    "OOF output builder must return one leading row per packed "
                    f"sample, got {tuple(packed_value.shape)} for {total_rows} rows"
                )
            destinations = (
                (destination_key, count_key),
                *additional_destinations,
            )
            for item, (start, end) in zip(selected, selection["segments"]):
                value = packed_value[int(start) : int(end)]
                parent = cached_batches[int(item["_cache_index"])]
                row_indices = item["_row_indices"]
                for target_key, target_count_key in destinations:
                    expected = parent[target_key].index_select(0, row_indices)
                    if value.shape != expected.shape:
                        raise RuntimeError(
                            f"OOF value shape {tuple(value.shape)} does not match "
                            f"destination {target_key} shape {tuple(expected.shape)}"
                        )
                    parent[target_key].index_copy_(0, row_indices, value)
                    if target_count_key is not None:
                        increment = torch.ones_like(row_indices, dtype=torch.long)
                        parent[target_count_key].index_add_(
                            0, row_indices, increment
                        )

    def _reliability_output(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        values = []
        for name in ("api", "graph", "manifest"):
            value = outputs.get(f"predicted_reliability_{name}")
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Nested cross-fitting requires predicted_reliability_{name}"
                )
            values.append(value.view(-1))
        return torch.stack(values, dim=-1)

    def _route_output(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        value = outputs.get("routing_branch_distribution")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "Nested cross-fitting requires routing_branch_distribution"
            )
        return value

    def _validate_oof_field(value_key: str, count_key: str) -> None:
        for cached in cached_batches:
            count = cached[count_key]
            value = cached[value_key]
            if not bool(count.eq(1).all().item()):
                counts = torch.unique(count.detach().cpu()).tolist()
                raise RuntimeError(
                    f"{value_key} must be written exactly once per row; counts={counts}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"{value_key} contains an unfilled or non-finite row")

    def _build_upstream_oof_clean_rows(
        items: list[dict[str, Any]],
        *,
        reliability_key: str | None,
        route_key: str | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for cached in items:
                outputs = _forward_cached(
                    cached,
                    reliability_key=reliability_key,
                    route_key=route_key,
                )
                raw_log_prob = outputs.get("uncalibrated_final_log_prob")
                if not isinstance(raw_log_prob, torch.Tensor) or (
                    raw_log_prob.ndim != 2
                    or raw_log_prob.size(0) != cached["labels"].numel()
                    or raw_log_prob.size(1) != 2
                ):
                    raise RuntimeError(
                        "Threshold-aligned I2 risk fitting requires upstream "
                        "OOF uncalibrated_final_log_prob with shape [B, 2]"
                    )
                labels = cached["labels"].detach().cpu().long().tolist()
                sample_ids = [str(value) for value in cached["sample_ids"]]
                sample_groups = [
                    str(value) for value in cached["sample_groups"]
                ]
                raw_rows = raw_log_prob.detach().cpu().float().tolist()
                if not (
                    len(labels)
                    == len(sample_ids)
                    == len(sample_groups)
                    == len(raw_rows)
                ):
                    raise RuntimeError(
                        "OOF clean IDs, groups, labels, and scores disagree"
                    )
                rows.extend(
                    {
                        "sid": sid,
                        "group": group,
                        "label": int(label),
                        "raw_log_prob": [float(raw[0]), float(raw[1])],
                    }
                    for sid, group, label, raw in zip(
                        sample_ids, sample_groups, labels, raw_rows
                    )
                )
        return validate_posthoc_oof_rows(rows)

    def _threshold_fn_risk_support(
        items: list[dict[str, Any]],
        *,
        reliability_key: str | None,
        route_key: str | None,
        raw_threshold: float,
    ) -> dict[str, Any]:
        """Audit whether threshold-aligned FN risk has identifiable labels."""
        scenario_stats: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for cached in items:
                outputs = _forward_cached(
                    cached,
                    reliability_key=reliability_key,
                    route_key=route_key,
                )
                raw_log_prob = outputs.get("uncalibrated_final_log_prob")
                available = outputs.get("routing_has_available")
                if not isinstance(raw_log_prob, torch.Tensor) or not isinstance(
                    available, torch.Tensor
                ):
                    raise RuntimeError(
                        "Threshold-FN support audit requires raw routed scores "
                        "and routing_has_available"
                    )
                labels = cached["labels"].detach().long().view(-1)
                predicted_benign = (
                    raw_log_prob.detach()[:, 1]
                    - raw_log_prob.detach()[:, 0]
                    < float(raw_threshold)
                )
                valid = available.detach().view(-1).bool() & predicted_benign
                positive = valid & labels.eq(1)
                negative = valid & labels.eq(0)
                groups = [str(value) for value in cached["sample_groups"]]
                sample_ids = [str(value) for value in cached["sample_ids"]]
                positive_cpu = positive.detach().cpu().tolist()
                negative_cpu = negative.detach().cpu().tolist()
                valid_cpu = valid.detach().cpu().tolist()
                name = str(cached["scenario_name"])
                scenario_stats[name] = {
                    "scenario_group": str(cached["scenario_group"]),
                    "num_rows": int(labels.numel()),
                    "num_predicted_benign": int(valid.sum().item()),
                    "num_positive_fn_events": int(positive.sum().item()),
                    "num_negative_events": int(negative.sum().item()),
                    "num_positive_groups": len(
                        {group for group, keep in zip(groups, positive_cpu) if keep}
                    ),
                    "num_negative_groups": len(
                        {group for group, keep in zip(groups, negative_cpu) if keep}
                    ),
                    "num_predicted_benign_groups": len(
                        {group for group, keep in zip(groups, valid_cpu) if keep}
                    ),
                    "num_positive_sample_ids": len(
                        {sid for sid, keep in zip(sample_ids, positive_cpu) if keep}
                    ),
                    "positive_groups": sorted(
                        {group for group, keep in zip(groups, positive_cpu) if keep}
                    ),
                    "negative_groups": sorted(
                        {group for group, keep in zip(groups, negative_cpu) if keep}
                    ),
                }

        def _aggregate(family: str) -> dict[str, Any] | None:
            selected = [
                value
                for value in scenario_stats.values()
                if (
                    value["scenario_group"] == "clean"
                    if family == "clean"
                    else value["scenario_group"] != "clean"
                )
            ]
            if not selected:
                return None
            return {
                "num_scenarios": len(selected),
                "num_rows": sum(value["num_rows"] for value in selected),
                "num_predicted_benign": sum(
                    value["num_predicted_benign"] for value in selected
                ),
                "num_positive_fn_events": sum(
                    value["num_positive_fn_events"] for value in selected
                ),
                "num_negative_events": sum(
                    value["num_negative_events"] for value in selected
                ),
                "num_positive_groups": len(
                    {
                        group
                        for value in selected
                        for group in value["positive_groups"]
                    }
                ),
                "num_negative_groups": len(
                    {
                        group
                        for value in selected
                        for group in value["negative_groups"]
                    }
                ),
            }

        support_cfg = (
            (((cfg.get("fusion", {}) or {}).get("routing", {}) or {}).get(
                "risk_support", {}
            ))
            or {}
        )
        enabled = bool(support_cfg.get("enabled", True))
        minima = {
            "num_positive_fn_events": int(
                support_cfg.get("min_positive_events", 1)
            ),
            "num_negative_events": int(
                support_cfg.get("min_negative_events", 1)
            ),
            "num_positive_groups": int(
                support_cfg.get("min_positive_groups", 1)
            ),
        }
        if any(value < 1 for value in minima.values()):
            raise ValueError(
                "fusion.routing.risk_support minima must all be positive"
            )
        aggregates = {
            "clean": _aggregate("clean"),
            "perturb": _aggregate("perturb"),
        }
        failures: list[str] = []
        if enabled:
            for family, weight in routing_scenario_weights.items():
                if weight <= 0.0:
                    continue
                stats = aggregates.get(family)
                if stats is None:
                    failures.append(f"{family}: no scenarios")
                    continue
                for key, minimum in minima.items():
                    if int(stats[key]) < minimum:
                        failures.append(
                            f"{family}.{key}={stats[key]} < {minimum}"
                        )
        result = {
            "enabled": enabled,
            "minima": minima,
            "scenario_objective_weights": dict(routing_scenario_weights),
            "families": aggregates,
            "scenarios": {
                name: {
                    key: value
                    for key, value in stats.items()
                    if key not in {"positive_groups", "negative_groups"}
                }
                for name, stats in scenario_stats.items()
            },
            "passed": not failures,
            "failures": failures,
        }
        if failures:
            raise RuntimeError(
                "Threshold-aligned I2 risk is not identifiable on the current "
                "post-hoc pool; adjust the split/threshold or preregister lower "
                "support minima instead of fitting an all-negative risk head: "
                + "; ".join(failures)
            )
        return result

    stage_summaries: dict[str, Any] = {}
    routing_risk_target_summary: dict[str, Any] = {
        "target": str(
            (((cfg.get("fusion", {}) or {}).get("routing", {}) or {}).get(
                "risk_target", "threshold_malware_false_negative"
            ))
        ).strip().lower(),
        "classification_threshold_source": None,
    }
    oof_clean_rows: list[dict[str, Any]] = []
    cross_fit_summary: dict[str, Any] = {
        **cross_fitting,
        "outer_folds": [],
    }
    uses_routed_output = (
        str(getattr(discount_fusion, "combination", "")) == "routed"
    )
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

        if bool(cross_fitting["enabled"]):
            has_reliability = bool(reliability_parameters)
            has_learned_route = bool(routing_distribution_parameters)
            cross_fit_summary["variant"] = (
                "outer_route_inner_i1"
                if has_reliability and has_learned_route
                else "oof_i1_fixed_route"
                if has_reliability
                else "fixed_i1_oof_route"
                if has_learned_route
                else "fixed_upstream"
            )
            reliability_initial = _parameter_snapshot(reliability_parameters)
            route_initial = _parameter_snapshot(routing_distribution_parameters)
            for cached in cached_batches:
                rows = int(cached["labels"].numel())
                device_for_cache = cached["labels"].device
                cached["working_reliability"] = torch.full(
                    (rows, 3),
                    float("nan"),
                    device=device_for_cache,
                    dtype=torch.float32,
                )
                cached["oof_reliability"] = torch.full_like(
                    cached["working_reliability"], float("nan")
                )
                cached["oof_route_distribution"] = torch.full_like(
                    cached["working_reliability"], float("nan")
                )
                cached["oof_reliability_count"] = torch.zeros(
                    rows, device=device_for_cache, dtype=torch.long
                )
                cached["oof_route_count"] = torch.zeros(
                    rows, device=device_for_cache, dtype=torch.long
                )

            if has_reliability and has_learned_route:
                # Strict outer-route / inner-I1 nesting. Every route-training
                # row is predicted by an I1 model that saw neither that row nor
                # the outer route holdout.
                # For an unordered pair of excluded folds {a, b}, the two
                # ordered uses (outer=a, inner=b) and (outer=b, inner=a) have
                # exactly the same I1 training rows and the same deterministic
                # initial state. Fit that model once and use it to predict each
                # excluded fold when its ordered role is visited.
                inner_i1_fit_cache: dict[tuple[int, ...], dict[str, Any]] = {}
                reused_inner_i1_fits = 0
                for outer_fold in all_folds:
                    outer_train_folds = [
                        fold for fold in all_folds if fold != outer_fold
                    ]
                    for cached in cached_batches:
                        cached["working_reliability"].fill_(float("nan"))
                    inner_summaries: list[dict[str, Any]] = []
                    for inner_fold in outer_train_folds:
                        inner_train_folds = [
                            fold
                            for fold in outer_train_folds
                            if fold != inner_fold
                        ]
                        inner_train = _fold_cached(
                            inner_train_folds,
                            role=f"outer_{outer_fold}_inner_{inner_fold}_train",
                        )
                        inner_holdout = _fold_cached(
                            [inner_fold],
                            role=f"outer_{outer_fold}_inner_{inner_fold}_holdout",
                        )
                        inner_fit_key = tuple(sorted(inner_train_folds))
                        cached_inner_fit = inner_i1_fit_cache.get(inner_fit_key)
                        if cached_inner_fit is None:
                            _restore_parameter_snapshot(
                                reliability_parameters, reliability_initial
                            )
                            inner_summary = _fit_reliability_stage(
                                f"reliability/outer_{outer_fold}/inner_{inner_fold}",
                                _build_reliability_groups(inner_train),
                            )
                            inner_summary["optimization_reused"] = False
                            inner_summary["executed_total_steps"] = int(
                                inner_summary["total_steps"]
                            )
                            inner_i1_fit_cache[inner_fit_key] = {
                                "parameters": _parameter_snapshot(
                                    reliability_parameters
                                ),
                                "summary": copy.deepcopy(inner_summary),
                                "source_outer_fold": int(outer_fold),
                                "source_inner_fold": int(inner_fold),
                            }
                        else:
                            _restore_parameter_snapshot(
                                reliability_parameters,
                                cached_inner_fit["parameters"],
                            )
                            inner_summary = copy.deepcopy(
                                cached_inner_fit["summary"]
                            )
                            inner_summary["name"] = (
                                f"reliability/outer_{outer_fold}/inner_{inner_fold}"
                            )
                            inner_summary["optimization_reused"] = True
                            inner_summary["executed_total_steps"] = 0
                            inner_summary["reused_from_outer_fold"] = int(
                                cached_inner_fit["source_outer_fold"]
                            )
                            inner_summary["reused_from_inner_fold"] = int(
                                cached_inner_fit["source_inner_fold"]
                            )
                            reused_inner_i1_fits += 1
                        _write_cached_prediction(
                            inner_holdout,
                            destination_key="working_reliability",
                            count_key=None,
                            reliability_key=None,
                            route_key=None,
                            output_builder=_reliability_output,
                        )
                        inner_summaries.append(
                            {
                                "holdout_fold": int(inner_fold),
                                "train_clean_samples": int(
                                    sum(
                                        item["labels"].numel()
                                        for item in _clean_items(inner_train)
                                    )
                                ),
                                "fit": _compact_stage_summary(inner_summary),
                            }
                        )

                    outer_train = _fold_cached(
                        outer_train_folds,
                        role=f"outer_{outer_fold}_train",
                    )
                    outer_holdout = _fold_cached(
                        [outer_fold],
                        role=f"outer_{outer_fold}_holdout",
                    )
                    _restore_parameter_snapshot(
                        reliability_parameters, reliability_initial
                    )
                    outer_i1_summary = _fit_reliability_stage(
                        f"reliability/outer_{outer_fold}/holdout_model",
                        _build_reliability_groups(outer_train),
                    )
                    _write_cached_prediction(
                        outer_holdout,
                        destination_key="working_reliability",
                        count_key=None,
                        reliability_key=None,
                        route_key=None,
                        output_builder=_reliability_output,
                        additional_destinations=(
                            ("oof_reliability", "oof_reliability_count"),
                        ),
                    )
                    for cached in cached_batches:
                        if not bool(
                            torch.isfinite(
                                cached["working_reliability"]
                            ).all().item()
                        ):
                            raise RuntimeError(
                                f"outer fold {outer_fold} has incomplete nested I1 predictions"
                            )

                    _restore_parameter_snapshot(
                        routing_distribution_parameters, route_initial
                    )
                    outer_route_summary = _optimize_stage(
                        f"routing_distribution/outer_{outer_fold}",
                        routing_distribution_parameters,
                        _build_routing_groups(
                            outer_train,
                            prefix="router",
                        ),
                        _routing_objective("working_reliability"),
                        config_stage_name="routing_distribution",
                        global_objective=True,
                    )
                    outer_holdout_with_reliability = _fold_cached(
                        [outer_fold],
                        role=f"outer_{outer_fold}_route_prediction",
                    )
                    _write_cached_prediction(
                        outer_holdout_with_reliability,
                        destination_key="oof_route_distribution",
                        count_key="oof_route_count",
                        reliability_key="working_reliability",
                        route_key=None,
                        output_builder=_route_output,
                    )
                    cross_fit_summary["outer_folds"].append(
                        {
                            "holdout_fold": int(outer_fold),
                            "train_clean_samples": int(
                                sum(
                                    item["labels"].numel()
                                    for item in _clean_items(outer_train)
                                )
                            ),
                            "holdout_clean_samples": int(
                                sum(
                                    item["labels"].numel()
                                    for item in _clean_items(outer_holdout)
                                )
                            ),
                            "inner_reliability_fits": inner_summaries,
                            "holdout_reliability_fit": _compact_stage_summary(
                                outer_i1_summary
                            ),
                            "route_fit": _compact_stage_summary(
                                outer_route_summary
                            ),
                        }
                    )
                cross_fit_summary["unique_inner_reliability_fits"] = int(
                    len(inner_i1_fit_cache)
                )
                cross_fit_summary["reused_inner_reliability_fits"] = int(
                    reused_inner_i1_fits
                )
            elif has_reliability:
                # With no learned route, ordinary OOF I1 is sufficient: every
                # downstream route/rule is a fixed transformation of that OOF
                # reliability and contains no fitted label-dependent module.
                for outer_fold in all_folds:
                    train_folds = [fold for fold in all_folds if fold != outer_fold]
                    train_items = _fold_cached(
                        train_folds, role=f"reliability_fold_{outer_fold}_train"
                    )
                    holdout_items = _fold_cached(
                        [outer_fold], role=f"reliability_fold_{outer_fold}_holdout"
                    )
                    _restore_parameter_snapshot(
                        reliability_parameters, reliability_initial
                    )
                    fit_summary = _fit_reliability_stage(
                        f"reliability/fold_{outer_fold}",
                        _build_reliability_groups(train_items),
                    )
                    _write_cached_prediction(
                        holdout_items,
                        destination_key="oof_reliability",
                        count_key="oof_reliability_count",
                        reliability_key=None,
                        route_key=None,
                        output_builder=_reliability_output,
                    )
                    cross_fit_summary["outer_folds"].append(
                        {
                            "holdout_fold": int(outer_fold),
                            "holdout_reliability_fit": _compact_stage_summary(
                                fit_summary
                            ),
                            "inner_reliability_fits": [],
                        }
                    )
            elif has_learned_route:
                # I1-off ablation: the route's upstream features are fixed, so
                # a direct grouped route cross-fit is strictly out of fold.
                for outer_fold in all_folds:
                    train_folds = [fold for fold in all_folds if fold != outer_fold]
                    train_items = _fold_cached(
                        train_folds, role=f"route_fold_{outer_fold}_train"
                    )
                    holdout_items = _fold_cached(
                        [outer_fold], role=f"route_fold_{outer_fold}_holdout"
                    )
                    _restore_parameter_snapshot(
                        routing_distribution_parameters, route_initial
                    )
                    fit_summary = _optimize_stage(
                        f"routing_distribution/fold_{outer_fold}",
                        routing_distribution_parameters,
                        _build_routing_groups(
                            train_items,
                            prefix="router",
                        ),
                        _routing_objective(None),
                        config_stage_name="routing_distribution",
                        global_objective=True,
                    )
                    _write_cached_prediction(
                        holdout_items,
                        destination_key="oof_route_distribution",
                        count_key="oof_route_count",
                        reliability_key=None,
                        route_key=None,
                        output_builder=_route_output,
                    )
                    cross_fit_summary["outer_folds"].append(
                        {
                            "holdout_fold": int(outer_fold),
                            "inner_reliability_fits": [],
                            "route_fit": _compact_stage_summary(fit_summary),
                        }
                    )

            if has_reliability:
                _validate_oof_field("oof_reliability", "oof_reliability_count")
                cross_fit_summary["oof_reliability_coverage"] = 1.0
            else:
                cross_fit_summary["oof_reliability_coverage"] = None

            if uses_routed_output and not has_learned_route:
                # Prior-only/static routed ablations still need an OOF-aligned
                # route tensor for risk and threshold fitting. Since the route
                # has no fitted parameters, evaluating the fixed transformation
                # on OOF I1 (or fixed raw inputs when I1 is off) is leakage-safe.
                fixed_route_items = _fold_cached(
                    all_folds, role="fixed_route_oof_projection"
                )
                _write_cached_prediction(
                    fixed_route_items,
                    destination_key="oof_route_distribution",
                    count_key="oof_route_count",
                    reliability_key=("oof_reliability" if has_reliability else None),
                    route_key=None,
                    output_builder=_route_output,
                )

            if uses_routed_output:
                _validate_oof_field("oof_route_distribution", "oof_route_count")
                cross_fit_summary["oof_route_coverage"] = 1.0
            else:
                cross_fit_summary["oof_route_coverage"] = None

            full_cached = _fold_cached(all_folds, role="full_posthoc_with_oof")
            full_clean_cached = _clean_items(full_cached)

            configured_risk_target = str(
                risk_cfg["routing"].get(
                    "risk_target", "threshold_malware_false_negative"
                )
            ).strip().lower()
            if (
                routing_risk_parameters
                and configured_risk_target
                == "threshold_malware_false_negative"
            ):
                oof_clean_rows = _build_upstream_oof_clean_rows(
                    full_clean_cached,
                    reliability_key=(
                        "oof_reliability" if has_reliability else None
                    ),
                    route_key=(
                        "oof_route_distribution"
                        if uses_routed_output
                        else None
                    ),
                )
                classification_enabled_for_risk = bool(
                    (cfg.get("classification_threshold", {}) or {}).get(
                        "enabled", False
                    )
                )
                if classification_enabled_for_risk:
                    threshold_for_risk = (
                        fit_oof_malware_classification_threshold(
                            oof_clean_rows,
                            cfg.get("classification_threshold", {}) or {},
                            # Only the raw cutoff is consumed here.  Its
                            # probability representation is remapped after the
                            # final scalar temperature is fitted below.
                            deployment_temperature=1.0,
                        )
                    )
                    if not threshold_for_risk:
                        raise RuntimeError(
                            "Threshold-aligned I2 risk requested but no OOF "
                            "classification threshold was fitted"
                        )
                    raw_threshold = float(
                        threshold_for_risk["raw_log_odds_threshold"]
                    )
                    threshold_source = "upstream_nested_oof_raw_score"
                    threshold_macro_f1 = float(
                        threshold_for_risk["macro_f1"]
                    )
                    threshold_malware_recall = float(
                        threshold_for_risk["malware_recall"]
                    )
                else:
                    # Explicit classification-off factorial cell: refit u for
                    # the protocol-neutral binary argmax boundary instead of
                    # silently reusing a risk head trained for another cutoff.
                    raw_threshold = 0.0
                    threshold_source = "protocol_neutral_fixed_0_5"
                    threshold_macro_f1 = None
                    threshold_malware_recall = None
                risk_cfg["routing"][
                    "classification_log_odds_threshold"
                ] = raw_threshold
                discount_fusion.set_routing_risk_decision_threshold(
                    raw_threshold
                )
                routing_risk_target_summary.update(
                    {
                        "classification_threshold_source": threshold_source,
                        "raw_log_odds_threshold": raw_threshold,
                        "num_threshold_rows": int(len(oof_clean_rows)),
                        "threshold_macro_f1": threshold_macro_f1,
                        "threshold_malware_recall": (
                            threshold_malware_recall
                        ),
                    }
                )
                if configured_risk_target == "threshold_malware_false_negative":
                    routing_risk_target_summary["support"] = (
                        _threshold_fn_risk_support(
                            full_cached,
                            reliability_key=(
                                "oof_reliability" if has_reliability else None
                            ),
                            route_key=(
                                "oof_route_distribution"
                                if uses_routed_output
                                else None
                            ),
                            raw_threshold=raw_threshold,
                        )
                    )

            if has_learned_route:
                _restore_parameter_snapshot(
                    routing_distribution_parameters, route_initial
                )
                stage_summaries["routing_distribution"] = _optimize_stage(
                    "routing_distribution",
                    routing_distribution_parameters,
                    _build_routing_groups(
                        full_cached,
                        prefix="router",
                    ),
                    _routing_objective(
                        "oof_reliability" if has_reliability else None
                    ),
                    config_stage_name="routing_distribution",
                    global_objective=True,
                )

            if has_reliability:
                _restore_parameter_snapshot(
                    reliability_parameters, reliability_initial
                )
                stage_summaries["reliability"] = _fit_reliability_stage(
                    "reliability",
                    _build_reliability_groups(full_cached),
                )

            if routing_risk_parameters:
                stage_summaries["routing_risk"] = _optimize_stage(
                    "routing_risk",
                    routing_risk_parameters,
                    _build_routing_groups(full_cached, prefix="risk"),
                    _risk_objective(
                        "oof_reliability" if has_reliability else None,
                        "oof_route_distribution" if uses_routed_output else None,
                    ),
                    config_stage_name="routing_risk",
                    global_objective=True,
                )
        else:
            if reliability_parameters:
                stage_summaries["reliability"] = _fit_reliability_stage(
                    "reliability",
                    _build_reliability_groups(full_cached),
                )
            if routing_distribution_parameters:
                stage_summaries["routing_distribution"] = _optimize_stage(
                    "routing_distribution",
                    routing_distribution_parameters,
                    _build_routing_groups(
                        full_cached,
                        prefix="router",
                    ),
                    _routing_objective(None),
                    config_stage_name="routing_distribution",
                    global_objective=True,
                )
            if routing_risk_parameters:
                stage_summaries["routing_risk"] = _optimize_stage(
                    "routing_risk",
                    routing_risk_parameters,
                    _build_routing_groups(full_cached, prefix="risk"),
                    _risk_objective(None, None),
                    config_stage_name="routing_risk",
                    global_objective=True,
                )

        handled_parameter_ids = {
            id(parameter)
            for stage_parameters in (
                reliability_parameters,
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
                "to reliability or routing stages"
            )
    finally:
        for param in model.parameters():
            param.requires_grad_(previous_requires_grad[id(param)])

    final_temperature_summary: dict[str, Any] = {"enabled": False}
    final_temperature_parameters = (
        discount_fusion.final_temperature_parameters()
        if discount_fusion is not None
        and hasattr(discount_fusion, "final_temperature_parameters")
        else []
    )
    needs_oof_rows = bool(cross_fitting["enabled"])
    clean_log_probs: list[torch.Tensor] = []
    clean_labels_for_temperature: list[torch.Tensor] = []
    clean_ids_for_rows: list[str] = []
    clean_groups_for_rows: list[str] = []
    if final_temperature_parameters or needs_oof_rows:
        # The scalar temperature sees only clean post-hoc rows. Its input is
        # already strict OOF with respect to every learned upstream I1/I2 stage.
        with torch.no_grad():
            for cached in full_clean_cached:
                routed = _forward_cached(
                    cached,
                    reliability_key=(
                        "oof_reliability"
                        if needs_oof_rows and reliability_parameters
                        else None
                    ),
                    route_key=(
                        "oof_route_distribution"
                        if needs_oof_rows and uses_routed_output
                        else None
                    ),
                )
                raw_log_prob = routed.get("uncalibrated_final_log_prob")
                if not isinstance(raw_log_prob, torch.Tensor):
                    raise RuntimeError(
                        "Final-temperature calibration requires "
                        "uncalibrated_final_log_prob"
                    )
                clean_log_probs.append(raw_log_prob.detach().float())
                clean_labels_for_temperature.append(
                    cached["labels"].detach().long()
                )
                clean_ids_for_rows.extend(
                    str(value) for value in cached["sample_ids"]
                )
                clean_groups_for_rows.extend(
                    str(value) for value in cached["sample_groups"]
                )
        if not clean_log_probs:
            raise RuntimeError(
                "Final-temperature calibration received no clean samples"
            )

    merged_raw = (
        torch.cat(clean_log_probs, dim=0) if clean_log_probs else None
    )
    merged_labels = (
        torch.cat(clean_labels_for_temperature, dim=0)
        if clean_labels_for_temperature
        else None
    )
    if final_temperature_parameters:
        assert merged_raw is not None
        assert merged_labels is not None
        # Refit the deployable scalar once on every upstream-OOF clean row.
        log_temperature = final_temperature_parameters[0]
        final_temperature_summary = _fit_routed_final_temperature(
            log_temperature,
            merged_raw,
            merged_labels,
        )
        final_temperature_summary["parameter_selection"] = (
            "full_upstream_oof_refit"
            if needs_oof_rows
            else "full_clean_refit"
        )
        stage_summaries["final_temperature"] = {
            "enabled": True,
            "name": "final_temperature",
            "epochs_ran": 1,
            "best_epoch": 1,
            "stopped_early": False,
            "converged": True,
            "parameter_selection": final_temperature_summary[
                "parameter_selection"
            ],
            "losses": [float(final_temperature_summary["nll_after"])],
            "final_loss": float(final_temperature_summary["nll_after"]),
            "total_steps": 1,
            "objective_groups": ["final_temperature:clean"],
            **final_temperature_summary,
        }

    if needs_oof_rows:
        assert merged_raw is not None
        assert merged_labels is not None
        if (
            len(clean_ids_for_rows) != int(merged_labels.numel())
            or len(clean_groups_for_rows) != int(merged_labels.numel())
        ):
            raise RuntimeError(
                "OOF clean IDs, package groups, and labels have different lengths"
            )
        # Binary temperature scaling is strictly monotone. Persist only the
        # upstream-OOF raw score and choose the classification cutoff on that
        # scale; run() maps the cutoff through the deployable temperature.
        # This avoids pretending a full-data fitted temperature is itself OOF.
        oof_clean_rows = [
            {
                "sid": sample_id,
                "group": sample_group,
                "label": int(label),
                "raw_log_prob": [float(raw[0]), float(raw[1])],
            }
            for sample_id, sample_group, label, raw in zip(
                clean_ids_for_rows,
                clean_groups_for_rows,
                merged_labels.detach().cpu().tolist(),
                merged_raw.detach().cpu().tolist(),
            )
        ]
        cross_fit_summary["classification_score_source"] = (
            "upstream_oof_raw_log_probability"
        )

    enabled_stages = [
        summary for summary in stage_summaries.values() if summary.get("enabled")
    ]
    if not enabled_stages:
        model.set_calibration_active(False)
        raise RuntimeError("Post-hoc calibration did not run any optimization stage")
    epoch_losses = list(enabled_stages[-1]["losses"])
    cross_fit_optimization_steps = 0
    cross_fit_stage_summaries: list[dict[str, Any]] = []
    for outer in cross_fit_summary.get("outer_folds", []):
        for inner in outer.get("inner_reliability_fits", []):
            fit = inner["fit"]
            cross_fit_optimization_steps += int(
                fit.get("executed_total_steps", fit["total_steps"])
            )
            if bool(fit.get("enabled", False)):
                cross_fit_stage_summaries.append(fit)
        for fit_key in ("holdout_reliability_fit", "route_fit"):
            fit = outer.get(fit_key)
            if isinstance(fit, dict):
                cross_fit_optimization_steps += int(fit["total_steps"])
                if bool(fit.get("enabled", False)):
                    cross_fit_stage_summaries.append(fit)
    total_steps = int(
        sum(stage["total_steps"] for stage in enabled_stages)
        + cross_fit_optimization_steps
    )
    best_loss = float(enabled_stages[-1]["final_loss"])
    aggregate_final_loss = float(
        sum(float(stage["final_loss"]) for stage in enabled_stages)
    )
    best_epoch = int(enabled_stages[-1]["best_epoch"])
    stopped_early = bool(
        any(stage["stopped_early"] for stage in enabled_stages)
        or any(
            stage.get("stopped_early", False)
            for stage in cross_fit_stage_summaries
        )
    )
    numerical_stages = [
        stage
        for stage in enabled_stages
        if stage.get("name") != "final_temperature"
    ]
    all_crossfit_stages_converged = all(
        bool(stage.get("converged", False))
        for stage in cross_fit_stage_summaries
    )
    cross_fit_summary["all_fitted_stages_converged"] = (
        all_crossfit_stages_converged
        if cross_fit_stage_summaries
        else None
    )
    all_numerical_stages_converged = bool(
        numerical_stages or cross_fit_stage_summaries
    ) and all(
        bool(stage.get("converged", False)) for stage in numerical_stages
    ) and all_crossfit_stages_converged
    temperatures = {}
    if bool(final_temperature_summary.get("enabled", False)):
        temperatures["final"] = float(final_temperature_summary["temperature"])
    reliability_temperatures: dict[str, float] = {}
    reliability_calibrator = getattr(
        model.discount_fusion, "reliability_calibrator", None
    )
    reliability_temperature = getattr(
        reliability_calibrator, "temperature", None
    )
    if (
        reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD
        and callable(reliability_temperature)
    ):
        reliability_temperatures = {
            name: float(reliability_temperature(name).detach().cpu().item())
            for name in reliability_branches
        }
    full_clean_sample_count = int(
        sum(item["labels"].numel() for item in full_clean_cached)
    )
    final_i2_parameter_diagnostics: dict[str, Any] = {}
    final_i2_parameter_details: dict[str, Any] = {}
    final_router = getattr(discount_fusion, "opinion_router", None)
    final_diagnostic_builder = getattr(
        final_router, "effective_parameter_diagnostics", None
    )
    if callable(final_diagnostic_builder):
        final_i2_parameter_diagnostics = final_diagnostic_builder()
    final_detail_builder = getattr(
        final_router, "effective_parameter_details", None
    )
    if callable(final_detail_builder):
        final_i2_parameter_details = final_detail_builder()

    def _active_stage_clean_count(stage_name: str) -> int:
        stage = stage_summaries.get(stage_name) or {}
        return full_clean_sample_count if bool(stage.get("enabled", False)) else 0

    total_wall_time_seconds = float(time.perf_counter() - fit_started_at)
    logger.info(
        "posthoc_calibration_complete wall_seconds=%.2f cache_seconds=%.2f "
        "optimization_steps=%s",
        total_wall_time_seconds,
        cache_wall_time_seconds,
        total_steps,
    )
    return {
        "enabled": True,
        "i2_diagnostics_schema_version": 2,
        "strategy": (
            "identity_grouped_nested_crossfit_staged_refit"
            if cross_fitting["enabled"]
            else "full_posthoc_fit_configured_group_reduction"
        ),
        "stage_grouping": stage_grouping,
        "reliability_calibration_method": reliability_method,
        "reliability_calibration_fit_source": (
            temperature_fit_source
            if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD
            else "clean_plus_branch_local_partial_degradations"
        ),
        "reliability_scenario_objective_weights": dict(
            i1_scenario_weights
        ),
        "reliability_temperatures": reliability_temperatures,
        "posthoc_fit_perturbations": [
            str(value).strip().lower()
            for value in (calibration_cfg.get("fit_perturbations") or [])
        ],
        "posthoc_fit_perturbation_strengths": [
            float(value)
            for value in (calibration_cfg.get("perturb_strengths") or [])
        ],
        "posthoc_fit_logical_source_count": len(calibration_sources),
        "posthoc_fit_transformed_source_count": max(
            len(calibration_sources) - 1, 0
        ),
        "routing_scenario_objective_weights": dict(
            routing_scenario_weights
        ),
        "routing_robustness_family_mapping": dict(
            sorted(ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION.items())
        ),
        "routing_risk_target": routing_risk_target_summary,
        # This is the only parameter block describing the fully fitted,
        # deployable I2. Stage-local blocks intentionally expose only the
        # component fitted at that stage, so route summaries cannot be
        # mistaken for final risk-head coefficients (or vice versa).
        "final_i2_parameter_diagnostics": final_i2_parameter_diagnostics,
        "final_i2_parameter_details": final_i2_parameter_details,
        "parameter_selection": (
            "strict_oof_upstream_then_full_stage_fit"
            if cross_fitting["enabled"]
            else "stage_numerical_convergence"
        ),
        "cross_fitting": cross_fit_summary,
        "stage_clean_sample_counts": {
            "reliability": _active_stage_clean_count("reliability"),
            "routing_distribution": _active_stage_clean_count(
                "routing_distribution"
            ),
            "routing_risk": _active_stage_clean_count("routing_risk"),
            "final_temperature": _active_stage_clean_count("final_temperature"),
        },
        "max_steps_default": default_max_steps,
        "epochs_ran": int(enabled_stages[-1]["total_steps"]),
        "loss_evaluations_ran": len(epoch_losses),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "all_numerical_stages_converged": all_numerical_stages_converged,
        "stage_optimization": stage_optimization_cfg,
        "losses": epoch_losses,
        "final_loss": best_loss,
        "aggregate_final_loss": aggregate_final_loss,
        "total_optimization_steps": total_steps,
        "cross_fit_optimization_steps": cross_fit_optimization_steps,
        "stages": stage_summaries,
        "num_input_loaders": len(calibration_sources),
        "num_unique_input_loaders": len(
            {id(source["loader"]) for source in calibration_sources}
        ),
        "calibration_sources": [
            {
                "name": source["name"],
                "scenario_group": source["scenario_group"],
                "objective_family": source["objective_family"],
                "perturb_type": source["perturb_type"],
                "strength": source["strength"],
                "reliability_branches": list(source["reliability_branches"]),
            }
            for source in calibration_sources
        ],
        "num_cached_batches": len(cached_batches),
        "num_encoder_batches_cached": num_encoder_batches_cached,
        "cache_wall_time_seconds": cache_wall_time_seconds,
        "wall_time_seconds": total_wall_time_seconds,
        "num_cached_samples": int(
            sum(int(item["labels"].numel()) for item in cached_batches)
        ),
        "temperatures": temperatures,
        "final_temperature": final_temperature_summary,
        # Consumed immediately by run() for classification-threshold fitting
        # and removed before checkpoint serialization.
        "_oof_clean_rows": oof_clean_rows,
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
        graph, labels, _, failed = prepare_robust_batch(batch, device)
        failed_seen += int(failed)
        if graph is None:
            continue
        valid_seen += int(labels.size(0))
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
                    (cfg.get("fusion", {}) or {}).get(
                        "evidence_activation", "softplus"
                    )
                ),
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
            if key not in TRAIN_LOG_LOSS_PART_KEYS:
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
                if key in TRAIN_LOG_LOSS_PART_KEYS
            ),
        )
        logger.info(
            "train_fusion_diagnostics epoch=%s %s",
            epoch,
            " ".join(
                f"mean_{key}={float((value / max(diagnostic_counts.get(key, 0), 1)).item()):.4f}"
                for key, value in sorted(diagnostic_sums.items())
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
    grouped: dict[str, list[tuple[float, int, int, bool]]] = {}
    for row_index, row in enumerate(rows):
        score = _finite_row_float(row, "acceptance_score")
        label_raw = _finite_row_float(row, "label")
        pred_raw = _finite_row_float(row, "pred")
        if label_raw is None or pred_raw is None:
            raise ValueError(
                f"Risk-coverage row {row_index} has invalid label/pred"
            )
        if (
            score is None
            or label_raw not in {0.0, 1.0}
            or pred_raw not in {0.0, 1.0}
        ):
            raise ValueError(
                f"Risk-coverage row {row_index} requires a finite score and binary label/pred"
            )
        label = int(label_raw)
        pred = int(pred_raw)
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Risk-coverage row {row_index} acceptance score must be within [0, 1]"
            )
        split = str(row.get("split") or "unknown")
        grouped.setdefault(split, []).append(
            (float(score), label, pred, _row_selective_eligible(row))
        )

    curve: list[dict[str, Any]] = []
    for split, items in sorted(grouped.items()):
        eligible_items = [item for item in items if item[3]]
        eligible_items.sort(key=lambda item: item[0], reverse=True)
        total = len(items)
        total_malware = sum(
            label == 1 for _score, label, _pred, _eligible in items
        )
        ineligible_count = total - len(eligible_items)
        tp = fp = tn = fn = 0
        accepted = 0
        index = 0
        curve.append(
            {
                "split": split,
                "acceptance_threshold": 1.0,
                "acceptance_comparison": ACCEPTANCE_THRESHOLD_COMPARISON,
                "num_total": int(total),
                "num_accepted": 0,
                "num_ineligible_forced_reject": int(ineligible_count),
                "coverage": 0.0,
                "selective_metrics_defined": False,
                "selective_risk": None,
                "selective_accuracy": None,
                "selective_macro_f1": None,
                "accepted_fn_risk_among_malware": (
                    0.0 if total_malware else None
                ),
                "fn_rate_given_accepted_malware": None,
            }
        )
        while index < len(eligible_items):
            tied_score = eligible_items[index][0]
            while (
                index < len(eligible_items)
                and eligible_items[index][0] == tied_score
            ):
                _score, label, pred, _eligible = eligible_items[index]
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
                    "acceptance_threshold": float(
                        math.nextafter(tied_score, float("-inf"))
                    ),
                    "acceptance_comparison": ACCEPTANCE_THRESHOLD_COMPARISON,
                    "num_total": int(total),
                    "num_accepted": int(accepted),
                    "num_ineligible_forced_reject": int(ineligible_count),
                    "coverage": float(accepted / total),
                    "selective_metrics_defined": True,
                    "selective_risk": float(errors / accepted),
                    "selective_accuracy": float(1.0 - errors / accepted),
                    "selective_macro_f1": float((f1_benign + f1_malware) / 2.0),
                    "accepted_fn_risk_among_malware": (
                        float(fn / total_malware) if total_malware else None
                    ),
                    "fn_rate_given_accepted_malware": (
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
    "temperature.py", 
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


def _canonicalize_inactive_stage_optimization(
    protocol_config: dict[str, Any],
) -> dict[str, Any]:
    """Remove the I1 stage block when I1 itself is disabled."""

    fusion_cfg = protocol_config.get("fusion", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    reliability_enabled = bool(reliability_cfg.get("enabled", False))
    calibration_cfg = protocol_config.get("calibration", {}) or {}
    stage_cfg = calibration_cfg.get("stage_optimization", {}) or {}
    if not isinstance(stage_cfg, dict):
        return protocol_config
    if not reliability_enabled:
        stage_cfg.pop("reliability", None)
    return protocol_config


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
    normalized = _canonicalize_classification_threshold_config(
        _canonicalize_selective_prediction_config(_json_compatible(cfg))
    )
    serialized = yaml.safe_dump(
        normalized,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    protocol_config = _method_protocol_config(normalized)
    if isinstance(protocol_config, dict):
        encoder_protocol = protocol_config.get("encoder_stage")
        if isinstance(encoder_protocol, dict):
            for lifecycle_key in (
                "mode",
                "expected_sha256",
                "strict_identity",
            ):
                encoder_protocol.pop(lifecycle_key, None)
        # Seed wrappers use distinct display names for the same method. Keep
        # method.protocol_id in the protocol hash: deleting the whole mapping
        # previously made two explicitly different protocols hash-identical.
        protocol_method = protocol_config.get("method")
        if isinstance(protocol_method, dict):
            protocol_method.pop("name", None)
            if not protocol_method:
                protocol_config.pop("method", None)
        protocol_config = _canonicalize_inactive_stage_optimization(
            protocol_config
        )
    protocol_serialized = yaml.safe_dump(
        protocol_config,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    fusion_cfg = normalized.get("fusion", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    calibration_cfg = normalized.get("calibration", {}) or {}
    eval_cfg = normalized.get("eval", {}) or {}
    selective_cfg = normalized.get("selective_prediction", {}) or {}
    classification_cfg = normalized.get("classification_threshold", {}) or {}
    loss_cfg = normalized.get("loss", {}) or {}
    reliability_enabled = bool(reliability_cfg.get("enabled", False))
    reliability_method = normalize_reliability_calibration_method(
        reliability_cfg.get("method", MONOTONIC_CORRECTNESS_METHOD)
    )
    reliability_scenario_weights_cfg = (
        reliability_cfg.get("scenario_objective_weights", {}) or {}
    )
    reliability_scenario_weights = {
        "clean": float(reliability_scenario_weights_cfg.get("clean", 0.5)),
        "perturb": float(
            reliability_scenario_weights_cfg.get("perturb", 0.5)
        ),
    }
    combination_rule = str(fusion_cfg.get("combination", "")).lower()
    routing_enabled = bool(
        combination_rule == "routed" and routing_cfg.get("enabled", False)
    )
    routing_prediction_loss_weight = float(
        routing_cfg.get("prediction_loss_weight", 1.0)
    )
    posthoc_fit_perturbations = [
        str(value).strip().lower()
        for value in (calibration_cfg.get("fit_perturbations") or [])
    ]
    posthoc_fit_strengths = [
        float(value)
        for value in (calibration_cfg.get("perturb_strengths") or [])
    ]
    robust_eval_perturbations, robust_eval_strengths = (
        _normalize_robust_test_protocol(eval_cfg)
    )
    routing_risk_loss_weight = float(routing_cfg.get("risk_loss_weight", 1.0))
    routing_risk_mode = str(routing_cfg.get("risk_mode", "learned")).strip().lower()
    routing_risk_enabled = routing_risk_mode != "disabled"
    routing_risk_loss = str(routing_cfg.get("risk_loss", "bce")).strip().lower()
    routing_mode = str(routing_cfg.get("mode", "learned")).strip().lower()
    routing_fixed_prior_beta = float(routing_cfg.get("fixed_prior_beta", 1.0))
    routing_risk_target = str(
        routing_cfg.get("risk_target", "threshold_malware_false_negative")
    ).strip().lower()
    routing_scenario_weights_cfg = (
        routing_cfg.get("scenario_objective_weights", {}) or {}
    )
    routing_scenario_clean_weight = float(
        routing_scenario_weights_cfg.get("clean", 0.5)
    )
    routing_scenario_perturb_weight = float(
        routing_scenario_weights_cfg.get("perturb", 0.5)
    )
    use_i1_reliability = bool(
        fusion_cfg.get("use_i1_reliability", True)
    )
    routing_reliability_prior_active = bool(
        routing_enabled and use_i1_reliability
    )
    auxiliary_weight_mode = resolve_auxiliary_weight_mode(loss_cfg)
    selective_enabled = bool(selective_cfg.get("enabled", False))
    classification_enabled = bool(classification_cfg.get("enabled", False))
    selective_mode = (
        str(selective_cfg.get("mode", "threshold")).lower()
        if selective_enabled
        else "disabled"
    )
    final_temperature_enabled = bool(
        routing_cfg.get("final_temperature_scaling", False)
    )
    training_objective = str(loss_cfg.get("objective", "standard")).strip().lower()
    method_cfg = normalized.get("method", {}) or {}
    objective_cfg = loss_cfg.get(training_objective, {}) or {}
    return {
        "method_name": str(method_cfg.get("name", experiment_name)),
        "method_protocol_id": method_cfg.get("protocol_id"),
        "experiment_name": str(experiment_name),
        "seed": int(seed),
        "resolved_config_sha256": hashlib.sha256(serialized).hexdigest(),
        "method_protocol_sha256": hashlib.sha256(protocol_serialized).hexdigest(),
        "method_implementation_sha256": _method_implementation_sha256(),
        "model_fusion_mode": str(
            (normalized.get("model", {}) or {}).get("fusion_mode", "")
        ),
        "combination_rule": combination_rule,
        "training_objective": training_objective,
        "evidential_anneal_epochs": (
            int(objective_cfg.get("anneal_epochs", 10))
            if training_objective in {"tmc", "ecml"}
            else None
        ),
        "ecml_consistency_weight": (
            float(objective_cfg.get("consistency_weight", 1.0))
            if training_objective == "ecml"
            else None
        ),
        "global_opinion_routing_enabled": routing_enabled,
        "routing_mode": (
            routing_mode
            if routing_enabled
            else "disabled"
        ),
        "routing_prior_semantics": (
            f"odds_prior_beta{routing_fixed_prior_beta:g}"
            if routing_reliability_prior_active and routing_mode == "prior_only"
            else (
                "learned_positive_odds_beta"
                if routing_reliability_prior_active and routing_mode == "learned"
                else "alive_masked_uniform_no_reliability_prior"
                if routing_enabled
                else "disabled"
            )
        ),
        "routing_fixed_prior_beta": (
            routing_fixed_prior_beta
            if routing_reliability_prior_active and routing_mode == "prior_only"
            else None
        ),
        "routing_prior_beta_trainable": bool(
            routing_reliability_prior_active and routing_mode == "learned"
        ),
        "routing_reliability_input_enabled": routing_reliability_prior_active,
        "routing_distribution_formula": (
            "beta_logit_reliability_minus_nonnegative_consensus_conflict"
            if routing_reliability_prior_active
            and routing_mode == "learned"
            and routing_cfg.get("route_conflict_enabled", True)
            else "beta_logit_reliability"
            if routing_reliability_prior_active and routing_mode == "learned"
            else "negative_nonnegative_consensus_conflict"
            if routing_enabled
            and routing_mode == "learned"
            and routing_cfg.get("route_conflict_enabled", True)
            else f"fixed_odds_prior_beta_{routing_fixed_prior_beta:g}"
            if routing_reliability_prior_active and routing_mode == "prior_only"
            else "alive_masked_uniform"
            if routing_enabled
            else "disabled"
        ),
        "routing_free_residual_enabled": False,
        "routing_route_conflict_enabled": bool(
            routing_enabled
            and routing_mode == "learned"
            and routing_cfg.get("route_conflict_enabled", True)
        ),
        "routing_route_conflict_metric": (
            "reliability_weighted_leave_one_out_normalized_js_v1"
            if routing_enabled
            and routing_mode == "learned"
            and routing_cfg.get("route_conflict_enabled", True)
            and use_i1_reliability
            else "alive_uniform_leave_one_out_normalized_js_v1"
            if routing_enabled
            and routing_mode == "learned"
            and routing_cfg.get("route_conflict_enabled", True)
            else "disabled"
        ),
        "routing_risk_conflict_enabled": bool(
            routing_enabled
            and routing_risk_mode == "learned"
            and routing_cfg.get("risk_conflict_enabled", True)
        ),
        "routing_prediction_loss_weight": (
            routing_prediction_loss_weight if routing_enabled else None
        ),
        "routing_route_effective_l2": (
            float(routing_cfg.get("route_effective_l2", 0.0))
            if routing_enabled
            else None
        ),
        "routing_risk_effective_l2": (
            float(routing_cfg.get("risk_effective_l2", 0.0))
            if routing_enabled and routing_risk_enabled
            else None
        ),
        "posthoc_fit_perturbations": posthoc_fit_perturbations,
        "posthoc_fit_perturbation_strengths": posthoc_fit_strengths,
        "posthoc_fit_transformed_source_count": (
            len(reliability_calibration_scenarios(normalized))
            if calibration_cfg.get("enabled", False)
            else 0
        ),
        "robust_eval_perturbations": robust_eval_perturbations,
        "robust_eval_perturbation_strengths": robust_eval_strengths,
        "robust_eval_expected_result_count": _robust_test_result_count(
            robust_eval_perturbations,
            robust_eval_strengths,
        ),
        "routing_risk_enabled": bool(routing_enabled and routing_risk_enabled),
        "routing_risk_mode": routing_risk_mode if routing_enabled else "disabled",
        "routing_risk_loss_weight": (
            routing_risk_loss_weight if routing_enabled and routing_risk_enabled else None
        ),
        "routing_risk_loss": (
            routing_risk_loss if routing_enabled and routing_risk_enabled else "disabled"
        ),
        "routing_risk_target": (
            routing_risk_target
            if routing_enabled and routing_risk_enabled
            else "disabled"
        ),
        "routing_scenario_objective_weights": (
            {
                "clean": routing_scenario_clean_weight,
                "perturb": routing_scenario_perturb_weight,
            }
            if routing_enabled
            else None
        ),
        "routing_initial_risk": (
            float(routing_cfg.get("initial_risk", 0.10))
            if routing_enabled and routing_risk_enabled
            else None
        ),
        "routing_posthoc_distribution_loss_enabled": bool(
            routing_enabled
            and routing_prediction_loss_weight > 0.0
        ),
        "routing_posthoc_risk_loss_enabled": bool(
            routing_enabled
            and routing_risk_mode == "learned"
            and routing_risk_loss_weight > 0.0
        ),
        "routed_final_temperature_enabled": bool(
            routing_enabled and final_temperature_enabled
        ),
        "final_temperature_enabled": final_temperature_enabled,
        "routed_final_temperature_override": (
            None
            if eval_cfg.get("final_temperature_override") is None
            else float(eval_cfg["final_temperature_override"])
        ),
        "reliability_calibration_enabled": reliability_enabled,
        "reliability_calibration_method": (
            reliability_method if reliability_enabled else "disabled"
        ),
        "reliability_calibrator_architecture": (
            (
                "per_branch_scalar_temperature_max_softmax"
                if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD
                else "per_branch_monotonic_logistic_correctness"
            )
            if reliability_enabled
            else "disabled"
        ),
        "reliability_calibration_fit_source": (
            str(
                reliability_cfg.get("temperature_fit_source", "clean_only")
            ).strip().lower()
            if reliability_enabled
            and reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD
            else (
                "clean_plus_branch_local_partial_degradations"
                if reliability_enabled
                else "disabled"
            )
        ),
        "reliability_lifecycle": (
            "single_stage_global_branch_correctness_proper_loss"
            if reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            else "single_stage_branch_temperature_nll"
            if reliability_enabled
            else "disabled"
        ),
        "reliability_scenario_objective_weights": (
            dict(reliability_scenario_weights)
            if reliability_enabled
            else None
        ),
        "reliability_calibration_loss": (
            (
                "branch_nll"
                if reliability_method
                == TEMPERATURE_SCALING_CONFIDENCE_METHOD
                else str(reliability_cfg.get("loss", "bce")).strip().lower()
            )
            if reliability_enabled
            else "disabled"
        ),
        "i1_reliability_input_enabled": use_i1_reliability,
        "auxiliary_weight_mode": auxiliary_weight_mode,
        "reliability_calibration_branches": (
            list(reliability_cfg.get("branches", BRANCH_NAMES))
            if reliability_enabled
            else []
        ),
        "router_trained_end_to_end": False,
        "router_posthoc_refinement_enabled": bool(
            routing_enabled
            and calibration_cfg.get("enabled", False)
            and routing_cfg.get("posthoc_refine", True)
        ),
        "router_encoder_training_reliability_source": (
            "alive_masked_uniform" if routing_enabled else "disabled"
        ),
        "router_posthoc_reliability_source": (
            (
                "temperature_scaled_max_softmax_confidence"
                if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD
                else "calibrated_branch_correctness"
            )
            if routing_enabled
            and use_i1_reliability
            and reliability_enabled
            else ("unit_prior" if routing_enabled else "disabled")
        ),
        "semantic_supervision_in_i1": False,
        "evidential_certainty_reliability_enabled": bool(
            reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            and reliability_cfg.get("use_evidential_certainty", True)
        ),
        "prediction_margin_reliability_enabled": bool(
            reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            and reliability_cfg.get("use_prediction_margin", True)
        ),
        "predicted_class_intercept_enabled": bool(
            reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            and reliability_cfg.get("use_predicted_class_intercept", True)
        ),
        "classification_threshold_enabled": classification_enabled,
        "classification_threshold_objective": (
            str(classification_cfg.get("objective", "macro_f1"))
            if classification_enabled
            else "disabled"
        ),
        "classification_threshold_selection_rule": (
            str(classification_cfg.get("selection_rule"))
            if classification_enabled
            else "disabled"
        ),
        "classification_threshold_constraint": (
            "none" if classification_enabled else "disabled"
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
        "risk_control_guarantee_type": (
            "expected_crc"
            if selective_enabled and selective_mode == "risk_control"
            else "disabled"
        ),
        "risk_control_risk_target": (
            str(selective_cfg.get("risk_target"))
            if selective_enabled and selective_mode == "risk_control"
            else "disabled"
        ),
        "risk_control_risk_denominator": (
            "all_malware"
            if selective_enabled and selective_mode == "risk_control"
            else "disabled"
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


_OUTPUT_COLLISION_ARTIFACTS = frozenset(
    {
        "resolved_config.yaml",
        "summary.yaml",
        "best_encoder_selected.pt",
        "best_tri_modal_robust.pt",
        "pipeline_decision_refit.pt",
    }
)
_OUTPUT_GENERATED_ARTIFACTS = _OUTPUT_COLLISION_ARTIFACTS | frozenset(
    {
        "gate_diagnostics.csv",
        "gate_diagnostics_extra_eval.csv",
        "risk_coverage_curve.csv",
        "risk_coverage_curve_extra_eval.csv",
        "metrics_extra_eval.json",
    }
)


def prepare_output_directory(
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    preserve_paths: set[Path] | None = None,
) -> Path:
    """Create a run directory without silently reusing experiment artifacts."""
    path = Path(out_dir)
    collisions = sorted(
        child.name
        for child in (path.iterdir() if path.is_dir() else ())
        if child.name in _OUTPUT_COLLISION_ARTIFACTS
    )
    if collisions and not overwrite:
        raise FileExistsError(
            f"Output directory {path} already contains run artifacts "
            f"{collisions}. Choose a new train.exp_name/seed or pass "
            "--overwrite explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        preserved = {
            candidate.resolve() for candidate in (preserve_paths or set())
        }
        for name in sorted(_OUTPUT_GENERATED_ARTIFACTS):
            artifact = path / name
            if not artifact.exists() or artifact.resolve() in preserved:
                continue
            if not artifact.is_file() and not artifact.is_symlink():
                raise ValueError(
                    f"Refusing to overwrite non-file run artifact: {artifact}"
                )
            artifact.unlink()
    return path


def run(cfg: dict, *, overwrite: bool = False) -> dict[str, Any]:
    # CLI loading already canonicalizes these sections. Repeating the
    # idempotent steps protects direct programmatic callers and keeps
    # resolved_config.yaml identical to the run-identity fingerprint.
    cfg = _canonicalize_classification_threshold_config(
        _canonicalize_selective_prediction_config(cfg)
    )
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
    encoder_stage_cfg = cfg.get("encoder_stage", {}) or {}
    encoder_stage_mode = str(
        encoder_stage_cfg.get("mode", "fit")
    ).strip().lower()
    if encoder_stage_mode not in {"fit", "reuse"}:
        raise ValueError("encoder_stage.mode must be 'fit' or 'reuse'")
    if not bool(encoder_stage_cfg.get("strict_identity", True)):
        raise ValueError(
            "encoder_stage.strict_identity=false is unsupported; a reused "
            "Stage-1 artifact must match its data/model/training identity"
        )
    configured_encoder_checkpoint = str(
        encoder_stage_cfg.get("checkpoint_path") or ""
    ).strip()
    if not eval_only and encoder_stage_mode == "reuse" and not configured_encoder_checkpoint:
        raise ValueError(
            "encoder_stage.mode=reuse requires encoder_stage.checkpoint_path"
        )
    if not eval_only and encoder_stage_mode == "fit" and configured_encoder_checkpoint:
        raise ValueError(
            "encoder_stage.checkpoint_path is only valid when mode=reuse"
        )
    if eval_only and configured_encoder_checkpoint:
        raise ValueError(
            "--encoder-checkpoint/encoder_stage.checkpoint_path is not valid "
            "for eval.eval_only runs; use eval.checkpoint_path and the explicit "
            "refit_posthoc_calibration lifecycle instead"
        )
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
    calibration_cfg = cfg.get("calibration", {}) or {}
    _require_stratified_group_validation_split(calibration_cfg)
    calibration_enabled = bool(calibration_cfg.get("enabled", False))
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
    if calibration_enabled and not discount_probability_mode:
        raise ValueError(
            "Post-hoc I1/I2 calibration modules require discount_probability fusion"
        )
    _validate_selective_score_fusion_compatibility(
        selective_enabled=selective_enabled,
        score_type=selective_score_type,
        discount_probability_mode=discount_probability_mode,
    )
    if run_robust_test and not run_test:
        raise ValueError("eval.run_robust_test=true requires eval.run_test=true")
    if eval_only:
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
        if (
            classification_threshold_enabled or selective_enabled
        ) and not refit_decision_calibration:
            raise ValueError(
                "Final-temperature override changes the probability scale and "
                "therefore requires eval.refit_decision_calibration=true when "
                "classification thresholding or selective prediction is enabled "
                "(set eval.refit_decision_calibration=true)"
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
                "(set eval.refit_decision_calibration=true)"
            )

    manifest_vocab_provenance = validate_manifest_vocab_provenance(cfg)
    manifest_runtime_fields = {
        "expected_manifest_vocab_sha256": "manifest_vocab_sha256",
        "expected_manifest_train_csv_sha256": "train_csv_sha256",
        "expected_manifest_train_sample_ids_sha256": (
            "train_sample_ids_sha256"
        ),
    }
    if bool(manifest_vocab_provenance.get("verified", False)):
        for data_key, provenance_key in manifest_runtime_fields.items():
            data_cfg[data_key] = manifest_vocab_provenance[provenance_key]
    else:
        # A resolved config from a provenance-enforced run may contain these
        # derived fields. Do not keep enforcing stale digests after the guard is
        # explicitly disabled.
        for data_key in manifest_runtime_fields:
            data_cfg.pop(data_key, None)

    exp_name = str(train_cfg.get("exp_name", "tri_modal_robust"))
    if eval_only:
        exp_name = str(
            eval_cfg.get("output_name") or f"{exp_name}_eval_only"
        )
    requested_out_dir = (
        Path(data_cfg.get("out_dir", "experiments")) / exp_name / str(seed)
    )
    preserve_on_overwrite: set[Path] = set()
    if configured_encoder_checkpoint:
        configured_path = Path(configured_encoder_checkpoint)
        preserve_on_overwrite.add(
            configured_path
            if configured_path.is_absolute()
            else Path.cwd() / configured_path
        )
    configured_eval_checkpoint = str(
        eval_cfg.get("checkpoint_path") or ""
    ).strip()
    if configured_eval_checkpoint:
        configured_eval_path = Path(configured_eval_checkpoint)
        configured_eval_path = (
            configured_eval_path
            if configured_eval_path.is_absolute()
            else Path.cwd() / configured_eval_path
        )
        preserve_on_overwrite.add(configured_eval_path)
        # A pipeline may live inside the very output directory being replaced.
        # Resolve and verify its immutable encoder dependency before cleanup so
        # --overwrite cannot delete the source required by a later refit (or
        # leave an otherwise portable pipeline with a broken provenance link).
        if configured_eval_path.is_file():
            configured_eval_artifact = torch.load(
                configured_eval_path,
                map_location="cpu",
                weights_only=True,
            )
            configured_eval_stage = validate_checkpoint_stage(
                configured_eval_artifact,
                checkpoint_path=configured_eval_path,
            )
            if configured_eval_stage == CHECKPOINT_STAGE_PIPELINE_FITTED:
                linked_encoder_path, _ = _load_linked_encoder_checkpoint(
                    configured_eval_path,
                    configured_eval_artifact,
                    map_location="cpu",
                )
                preserve_on_overwrite.add(linked_encoder_path)
    # Keep output mutation behind all dataset/split/role/model preflight below.
    # In particular, ``--overwrite`` must not delete a previous run before a
    # missing validation-role manifest or invalid PT pool is discovered.
    out_dir = requested_out_dir
    encoder_checkpoint_path = out_dir / "best_encoder_selected.pt"
    pipeline_checkpoint_path = out_dir / (
        "pipeline_decision_refit.pt"
        if eval_only
        and refit_decision_calibration
        and not refit_posthoc_calibration
        else "best_tri_modal_robust.pt"
    )
    train_ds = None
    train_loader = None
    val_loader = None
    val_posthoc_calibration_loader = None
    val_conformal_calibration_loader = None
    robust_calibration_loaders: list[dict[str, Any]] = []
    validation_split_summary: dict[str, Any] = {
        "split_seed": None,
        "validation_fraction": None,
        "num_validation": 0,
        "num_selection": 0,
        "num_calibration": 0,
        "num_posthoc_calibration": 0,
        "num_conformal_calibration": 0,
        "selection_fraction_of_validation": None,
        "posthoc_fraction_of_validation": None,
        "decision_fraction_of_validation": None,
        "external_eval_only": bool(pure_external_eval_only),
    }
    decision_calibration_data_identity: dict[str, Any] | None = None
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
            fixed_roles = _load_fixed_validation_roles(cfg, val_ds)
            if fixed_roles is not None:
                (
                    val_selection_ds,
                    val_posthoc_calibration_ds,
                    val_conformal_calibration_ds,
                    validation_split,
                ) = fixed_roles
                selection_indices = list(validation_split["selection_indices"])
                calibration_indices = list(validation_split["calibration_indices"])
                posthoc_calibration_indices = list(
                    validation_split["posthoc_calibration_indices"]
                )
                decision_calibration_indices = list(
                    validation_split["conformal_calibration_indices"]
                )
            else:
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
                    decision_calibration_indices = list(
                        conformal_split["conformal_calibration_indices"]
                    )
                else:
                    val_posthoc_calibration_ds = val_holdout_ds
                    val_conformal_calibration_ds = val_holdout_ds
                    posthoc_calibration_indices = calibration_indices
                    decision_calibration_indices = calibration_indices
                    validation_split.update(
                        {
                            "num_posthoc_calibration": len(calibration_indices),
                            "num_conformal_calibration": (
                                len(calibration_indices)
                                if _uses_conformal_selective(cfg.get("selective_prediction", {}) or {})
                                else 0
                            ),
                            "posthoc_fraction_of_validation": (
                                len(calibration_indices) / float(len(val_ds))
                            ),
                            "decision_fraction_of_validation": (
                                len(calibration_indices) / float(len(val_ds))
                                if _uses_conformal_selective(
                                    cfg.get("selective_prediction", {}) or {}
                                )
                                else 0.0
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
            decision_calibration_indices = None
            validation_split = {
                "split_seed": None,
                "validation_fraction": 1.0,
                "num_validation": len(val_ds),
                "num_selection": len(val_ds),
                "num_calibration": len(val_ds),
                "num_posthoc_calibration": len(val_ds),
                "num_conformal_calibration": 0,
                "selection_fraction_of_validation": 1.0,
                "calibration_fraction_of_validation": 1.0,
                "posthoc_fraction_of_validation": 1.0,
                "decision_fraction_of_validation": 0.0,
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
        if classification_threshold_enabled or selective_enabled:
            decision_calibration_data_identity = (
                _decision_calibration_data_identity(
                    cfg,
                    val_ds,
                    posthoc_indices=posthoc_calibration_indices,
                    decision_indices=decision_calibration_indices,
                )
            )
            validation_split_summary["decision_calibration_data_identity"] = (
                decision_calibration_data_identity
            )
        logger.info(
            "validation_protocol split_seed=%s selection=%d (%.4f) "
            "posthoc=%d (%.4f) decision=%d (%.4f) total=%d",
            validation_split_summary.get("split_seed"),
            int(validation_split_summary["num_selection"]),
            float(validation_split_summary["selection_fraction_of_validation"]),
            int(validation_split_summary["num_posthoc_calibration"]),
            float(validation_split_summary["posthoc_fraction_of_validation"]),
            int(validation_split_summary["num_conformal_calibration"]),
            float(validation_split_summary["decision_fraction_of_validation"]),
            int(validation_split_summary["num_validation"]),
        )
        val_loader = build_loader(cfg, val_selection_ds, is_train=False)
        val_posthoc_calibration_loader = build_loader(
            cfg, val_posthoc_calibration_ds, is_train=False
        )
        val_conformal_calibration_loader = build_loader(
            cfg, val_conformal_calibration_ds, is_train=False
        )
        if not eval_only and encoder_stage_mode == "fit":
            train_ds = build_dataset(cfg, "train", is_train=True)
            train_loader = build_loader(
                cfg,
                train_ds,
                is_train=True,
                seed_namespace="stage1_train",
            )
        robust_calibration_loaders = (
            build_reliability_calibration_loaders(
                cfg, val_ds, posthoc_calibration_indices
            )
            if calibration_enabled
            and (not eval_only or refit_posthoc_calibration)
            and uses_routing_calibration_scenarios(cfg)
            else []
        )
        feature_dim = train_ds.feature_dim if train_ds is not None else val_ds.feature_dim
    if (
        not eval_only
        and encoder_stage_mode == "fit"
        and train_ds is None
    ):
        train_ds = build_dataset(cfg, "train", is_train=True)
        train_loader = build_loader(
            cfg,
            train_ds,
            is_train=True,
            seed_namespace="stage1_train",
        )
    test_ds: RobustTriModalDataset | None = None
    test_loader = None
    if run_test:
        test_ds = build_dataset(cfg, "test", is_train=False)
        test_loader = build_loader(cfg, test_ds, is_train=False)

    if not eval_only or refit_posthoc_calibration:
        # A fresh model uses the same declared initialization stream in fit,
        # encoder-reuse, and eval-refit lifecycles. Reuse then strictly replaces
        # Stage-1 tensors, leaving I1/I2 at the same pre-fit starting state.
        set_seed(int(train_cfg.get("stage1_seed", seed)))
    model = build_model(cfg, feature_dim).to(device)
    source_checkpoint_path: Path | None = None
    requested_checkpoint_path: Path | None = None
    posthoc_oof_clean_rows: list[dict[str, Any]] = []
    classification_log_odds_threshold: float | None = None
    encoder_state_sha_before_posthoc: str | None = None
    encoder_state_sha_after_posthoc: str | None = None
    encoder_stage_source_path: Path | None = None

    out_dir = prepare_output_directory(
        requested_out_dir,
        overwrite=overwrite,
        preserve_paths=preserve_on_overwrite,
    )
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
        if "allow_checkpoint_config_mismatch" in eval_cfg:
            raise ValueError(
                "eval.allow_checkpoint_config_mismatch was removed; retrain "
                "checkpoints whose configuration or implementation differs"
            )
        if refit_posthoc_calibration:
            encoder_state = validate_encoder_stage_checkpoint(
                ckpt,
                current_cfg=cfg,
                validation_split=validation_split_summary,
                manifest_vocab_provenance=manifest_vocab_provenance,
                checkpoint_path=best_path,
            )
            model.load_encoder_stage_state_dict(encoder_state)
            encoder_stage_source_path = best_path
        else:
            validate_eval_checkpoint_config(cfg, ckpt.get("cfg"))
            validate_checkpoint_implementation(ckpt)
            validate_checkpoint_manifest_vocab_provenance(
                ckpt, manifest_vocab_provenance
            )
            validate_checkpoint_decision_signature(
                cfg,
                ckpt,
                refit_decision_calibration=refit_decision_calibration,
                refit_posthoc_calibration=refit_posthoc_calibration,
                current_data_identity=decision_calibration_data_identity,
            )
            model.load_state_dict(ckpt["model"])
            encoder_state_sha_before_posthoc = _state_dict_sha256(
                model.encoder_stage_state_dict()
            )
            encoder_state_sha_after_posthoc = encoder_state_sha_before_posthoc
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
                    "with opinion-fusion final-temperature scaling enabled"
                )
            with torch.no_grad():
                # parameters[0].fill_(math.log(final_temperature_override))
                parameters[0].fill_(
                    raw_final_temperature_coordinate(final_temperature_override)
                )
            logger.info(
                "eval_final_temperature_override=%.6f",
                final_temperature_override,
            )
        best_score = float(ckpt.get("checkpoint_score", -1.0))
        best_val_f1 = float((ckpt.get("val") or {}).get("macro_f1", -1.0))
        checkpoint_metric_name = str(ckpt.get("checkpoint_metric", "loaded_checkpoint"))
        calibration_summary = dict(ckpt.get("calibration") or {"enabled": False})
        posthoc_oof_clean_rows = list(
            ckpt.get("posthoc_oof_clean_rows") or []
        )
        if posthoc_oof_clean_rows:
            if (
                ckpt.get("posthoc_oof_clean_rows_schema_version")
                != POSTHOC_OOF_ROWS_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Checkpoint OOF rows use an unsupported schema; retrain "
                    "the pipeline checkpoint"
                )
            posthoc_oof_clean_rows = validate_posthoc_oof_rows(
                posthoc_oof_clean_rows
            )
            validate_posthoc_oof_rows_identity(
                posthoc_oof_clean_rows,
                ckpt.get("decision_calibration_data_identity"),
            )
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
            raw_cutoff = (classification_threshold_summary or {}).get(
                "raw_log_odds_threshold"
            )
            classification_log_odds_threshold = (
                None if raw_cutoff is None else float(raw_cutoff)
            )
            if classification_log_odds_threshold is not None and not math.isfinite(
                classification_log_odds_threshold
            ):
                raise ValueError(
                    "Checkpoint classification raw-log-odds threshold must be finite"
                )
        else:
            classification_threshold_summary = None
            classification_threshold = 0.5
            classification_log_odds_threshold = None
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
            encoder_state_sha_before_posthoc = _state_dict_sha256(
                model.encoder_stage_state_dict()
            )
            calibration_summary = fit_posthoc_calibration(
                model,
                calibration_loaders,
                device,
                use_amp,
                cfg,
            )
            encoder_state_sha_after_posthoc = _state_dict_sha256(
                model.encoder_stage_state_dict()
            )
            if encoder_state_sha_after_posthoc != encoder_state_sha_before_posthoc:
                raise RuntimeError(
                    "Post-hoc fitting mutated the frozen Stage-1 encoder artifact"
                )
            posthoc_oof_clean_rows = list(
                calibration_summary.pop("_oof_clean_rows", [])
            )
            posthoc_oof_clean_rows = validate_posthoc_oof_rows(
                posthoc_oof_clean_rows
            )
            if bool(calibration_summary.get("enabled", False)):
                calibration_summary["posthoc_perturbation_scenarios"] = [
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
            classification_log_odds_threshold = None
            rejection_threshold = None
            conformal_thresholds = None
            risk_control_thresholds = None
        logger.info("eval-only mode loaded checkpoint: %s", best_path)
    else:
        model.set_calibration_active(False)
        best_path = encoder_checkpoint_path
        if encoder_stage_mode == "reuse":
            source_encoder = Path(configured_encoder_checkpoint)
            if not source_encoder.is_absolute():
                source_encoder = Path.cwd() / source_encoder
            if not source_encoder.is_file():
                raise FileNotFoundError(
                    f"Reusable encoder artifact not found: {source_encoder}"
                )
            expected_sha = str(
                encoder_stage_cfg.get("expected_sha256") or ""
            ).strip().lower()
            actual_sha = _file_sha256(source_encoder)
            if expected_sha and expected_sha != actual_sha:
                raise ValueError(
                    "Reusable encoder artifact file hash mismatch: "
                    f"expected={expected_sha} actual={actual_sha}"
                )
            ckpt = torch.load(
                source_encoder, map_location=device, weights_only=True
            )
            encoder_state = validate_encoder_stage_checkpoint(
                ckpt,
                current_cfg=cfg,
                validation_split=validation_split_summary,
                manifest_vocab_provenance=manifest_vocab_provenance,
                checkpoint_path=source_encoder,
            )
            model.load_encoder_stage_state_dict(encoder_state)
            if source_encoder.resolve() != best_path.resolve():
                _copy_file_verified_atomic(source_encoder, best_path)
            encoder_stage_source_path = source_encoder
            best_score = float(ckpt.get("checkpoint_score", -1.0))
            best_val_f1 = float(
                (ckpt.get("val") or {}).get("macro_f1", -1.0)
            )
            checkpoint_metric_name = str(
                ckpt.get("checkpoint_metric", "loaded_encoder_artifact")
            )
            logger.info(
                "encoder_stage_reused source=%s sha256=%s epoch=%s score=%.6f",
                source_encoder,
                actual_sha,
                ckpt.get("epoch"),
                best_score,
            )
        else:
            assert train_loader is not None
            posthoc_only_parameters = (
                model.encoder_training_frozen_parameters()
                if hasattr(model, "encoder_training_frozen_parameters")
                else model.calibration_parameters()
            )
            for parameter in posthoc_only_parameters:
                parameter.requires_grad_(False)

            # Reset the main-process RNG immediately before Stage-1. Changes to
            # frozen I1/I2 module construction can no longer alter dropout or
            # augmentation draws; the DataLoader owns a separate generator.
            stage1_seed = int(train_cfg.get("stage1_seed", seed))
            set_seed(stage1_seed)
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
            checkpoint_metric_name = "clean_macro_f1"
            saved_checkpoint_this_run = False
            patience = int(train_cfg.get("patience", 10))
            stale = 0
            encoder_identity = _encoder_stage_semantic_signature(
                cfg, validation_split_summary, manifest_vocab_provenance
            )
            encoder_identity_sha = _canonical_mapping_sha256(encoder_identity)
            encoder_implementation_sha = _encoder_stage_implementation_sha256()

            for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
                train_started_at = time.perf_counter()
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    cfg,
                    epoch,
                )
                train_wall_seconds = float(
                    time.perf_counter() - train_started_at
                )
                val_started_at = time.perf_counter()
                val_metrics = evaluate_checkpoint_selection(
                    model,
                    val_loader,
                    device,
                    use_amp,
                    "val_checkpoint_selection",
                )
                enforce_failed_ratio(val_metrics, cfg, "val")
                val_wall_seconds = float(
                    time.perf_counter() - val_started_at
                )
                score = float(val_metrics["macro_f1"])
                if not math.isfinite(float(score)):
                    raise FloatingPointError(
                        f"Non-finite checkpoint score at epoch={epoch}: {score}"
                    )
                scheduler.step()
                branch_f1 = {
                    branch: values.get("macro_f1")
                    for branch, values in (
                        val_metrics.get("branch_metrics") or {}
                    ).items()
                }
                logger.info(
                    "epoch=%s train_loss=%.4f val_macro_f1=%.4f "
                    "val_auc=%.4f val_acc=%.4f checkpoint_score=%.4f "
                    "branch_macro_f1=%s train_wall_seconds=%.2f "
                    "val_wall_seconds=%.2f",
                    epoch,
                    train_loss,
                    val_metrics["macro_f1"],
                    val_metrics["auc"],
                    val_metrics["acc"],
                    score,
                    branch_f1,
                    train_wall_seconds,
                    val_wall_seconds,
                )
                if score > best_score + float(
                    train_cfg.get("min_delta", 1e-4)
                ):
                    best_score = score
                    best_val_f1 = float(val_metrics["macro_f1"])
                    stale = 0
                    encoder_state = model.encoder_stage_state_dict()
                    encoder_state_sha = _state_dict_sha256(encoder_state)
                    _atomic_torch_save(
                        {
                            "encoder_stage_artifact_schema_version": (
                                ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION
                            ),
                            "encoder_stage_state": encoder_state,
                            "encoder_stage_state_sha256": encoder_state_sha,
                            "encoder_stage_identity": encoder_identity,
                            "encoder_stage_identity_sha256": encoder_identity_sha,
                            "encoder_stage_implementation_sha256": (
                                encoder_implementation_sha
                            ),
                            "cfg": cfg,
                            "val": val_metrics,
                            "checkpoint_score": score,
                            "checkpoint_metric": checkpoint_metric_name,
                            "manifest_vocab_provenance": (
                                manifest_vocab_provenance
                            ),
                            "validation_split": validation_split_summary,
                            "checkpoint_stage": (
                                CHECKPOINT_STAGE_ENCODER_SELECTED
                            ),
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
                    "Training completed without writing a checkpoint in this "
                    "run; refusing to load an earlier encoder artifact"
                )
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            encoder_state = validate_encoder_stage_checkpoint(
                ckpt,
                current_cfg=cfg,
                validation_split=validation_split_summary,
                manifest_vocab_provenance=manifest_vocab_provenance,
                checkpoint_path=best_path,
            )
            model.load_encoder_stage_state_dict(encoder_state)
            encoder_stage_source_path = best_path

        encoder_state_sha_before_posthoc = _state_dict_sha256(
            model.encoder_stage_state_dict()
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
        encoder_state_sha_after_posthoc = _state_dict_sha256(
            model.encoder_stage_state_dict()
        )
        if encoder_state_sha_after_posthoc != encoder_state_sha_before_posthoc:
            raise RuntimeError(
                "Post-hoc fitting mutated the frozen Stage-1 encoder artifact"
            )
        posthoc_oof_clean_rows = list(
            calibration_summary.pop("_oof_clean_rows", [])
        )
        posthoc_oof_clean_rows = validate_posthoc_oof_rows(
            posthoc_oof_clean_rows
        )
        if bool(calibration_summary.get("enabled", False)):
            calibration_summary["posthoc_perturbation_scenarios"] = [
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
        classification_log_odds_threshold = None

    if pure_external_eval_only:
        val_calibration_metrics: dict[str, Any] = {}
        val_calibration_rows: list[dict[str, Any]] = []
        val_conformal_metrics: dict[str, Any] = {}
        val_conformal_rows: list[dict[str, Any]] = []
        val_metrics: dict[str, Any] = {}
        val_rows: list[dict[str, Any]] = []
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
        )
        enforce_failed_ratio(val_calibration_metrics, cfg, "val_posthoc_calibration")
        if classification_threshold_enabled and (
            not eval_only or refit_decision_calibration
        ):
            cross_fit_enabled = bool(
                (calibration_summary.get("cross_fitting") or {}).get(
                    "enabled", False
                )
            )
            if cross_fit_enabled and not posthoc_oof_clean_rows:
                raise RuntimeError(
                    "Nested cross-fitted classification-threshold fitting requires "
                    "the clean OOF predictions from the current post-hoc fit; rerun "
                    "with eval.refit_posthoc_calibration=true"
                )
            if cross_fit_enabled:
                validate_posthoc_oof_rows_identity(
                    posthoc_oof_clean_rows,
                    decision_calibration_data_identity,
                )
            if cross_fit_enabled:
                decision_fusion = getattr(model, "discount_fusion", None)
                temperature_value = (
                    decision_fusion.final_temperature()
                    if decision_fusion is not None
                    and hasattr(decision_fusion, "final_temperature")
                    else None
                )
                deployment_temperature = (
                    float(temperature_value.detach().cpu().item())
                    if isinstance(temperature_value, torch.Tensor)
                    else 1.0
                )
                classification_threshold_summary = (
                    fit_oof_malware_classification_threshold(
                        posthoc_oof_clean_rows,
                        classification_cfg,
                        deployment_temperature=deployment_temperature,
                    )
                )
            else:
                classification_threshold_summary = (
                    fit_malware_classification_threshold(
                        val_calibration_rows, classification_cfg
                    )
                )
            if classification_threshold_summary is None:
                raise RuntimeError("Enabled classification-threshold fitting returned no result")
            classification_threshold = float(
                classification_threshold_summary["threshold"]
            )
            raw_cutoff = classification_threshold_summary.get(
                "raw_log_odds_threshold"
            )
            classification_log_odds_threshold = (
                None if raw_cutoff is None else float(raw_cutoff)
            )
            fitted_risk_target = (
                calibration_summary.get("routing_risk_target") or {}
            )
            risk_aligned_cutoff = fitted_risk_target.get(
                "raw_log_odds_threshold"
            )
            if risk_aligned_cutoff is not None:
                if classification_log_odds_threshold is None or not math.isclose(
                    float(risk_aligned_cutoff),
                    float(classification_log_odds_threshold),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise RuntimeError(
                        "The final classification cutoff differs from the OOF "
                        "cutoff that supervised I2 risk; rerun post-hoc "
                        "calibration instead of reusing this risk head"
                    )
                classification_threshold_summary[
                    "routing_risk_alignment_verified"
                ] = True
                classification_threshold_summary[
                    "routing_risk_target"
                ] = fitted_risk_target.get("target")
            if not cross_fit_enabled:
                classification_threshold_summary["prediction_source"] = (
                    "full_posthoc_predictions"
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
                classification_log_odds_threshold=(
                    classification_log_odds_threshold
                ),
            )
            enforce_failed_ratio(
                val_calibration_metrics, cfg, "val_posthoc_calibration"
            )
        validate_threshold_aligned_risk_cutoff(
            model,
            cfg,
            classification_log_odds_threshold,
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
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
        if (
            not eval_only
            or refit_posthoc_calibration
            or refit_decision_calibration
        ) and best_path.exists():
            checkpoint_source = (
                source_checkpoint_path
                if eval_only and source_checkpoint_path is not None
                else best_path
            )
            ckpt = torch.load(
                checkpoint_source, map_location="cpu", weights_only=True
            )
            source_stage = validate_checkpoint_stage(
                ckpt, checkpoint_path=checkpoint_source
            )
            if source_stage == CHECKPOINT_STAGE_ENCODER_SELECTED:
                source_encoder_path = Path(checkpoint_source)
            else:
                source_encoder_path, _source_encoder_checkpoint = (
                    _load_linked_encoder_checkpoint(
                        checkpoint_source,
                        ckpt,
                        map_location="cpu",
                    )
                )
            # The pipeline artifact carries the full deployable model and a
            # hash-checked link to the immutable encoder-only artifact. Do not
            # duplicate the Stage-1 tensor payload inside both files.
            ckpt.pop("encoder_stage_state", None)
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
                source_encoder_path.resolve()
                != portable_encoder_path.resolve()
            ):
                _copy_file_verified_atomic(
                    source_encoder_path, portable_encoder_path
                )
            ckpt["encoder_checkpoint_path"] = portable_encoder_path.name
            ckpt["encoder_checkpoint_sha256"] = _file_sha256(
                portable_encoder_path
            )
            ckpt["pipeline_artifact_schema_version"] = (
                PIPELINE_ARTIFACT_SCHEMA_VERSION
            )
            ckpt["pipeline_model_state_sha256"] = _state_dict_sha256(
                ckpt["model"]
            )
            ckpt["calibration"] = calibration_summary
            if (
                posthoc_oof_clean_rows
                and decision_calibration_data_identity is not None
            ):
                # Compact clean OOF raw log probabilities are retained so
                # protocol appendix runs can refit the classification operating
                # point without
                # rerunning the nested post-hoc stack or using in-sample rows.
                validate_posthoc_oof_rows_identity(
                    posthoc_oof_clean_rows,
                    decision_calibration_data_identity,
                )
                ckpt["posthoc_oof_clean_rows"] = posthoc_oof_clean_rows
                ckpt["posthoc_oof_clean_rows_schema_version"] = (
                    POSTHOC_OOF_ROWS_SCHEMA_VERSION
                )
            else:
                ckpt.pop("posthoc_oof_clean_rows", None)
                ckpt.pop("posthoc_oof_clean_rows_schema_version", None)
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
            ckpt["decision_calibration_data_identity"] = (
                decision_calibration_data_identity
            )
            ckpt["validation_split"] = validation_split_summary
            ckpt["manifest_vocab_provenance"] = manifest_vocab_provenance
            if eval_only:
                ckpt["source_checkpoint_path"] = str(checkpoint_source)
            ckpt["pipeline_decision_metadata_sha256"] = (
                _pipeline_decision_metadata_sha256(ckpt)
            )
            best_path = pipeline_checkpoint_path
            _atomic_torch_save(ckpt, best_path)
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
        )
        enforce_failed_ratio(val_metrics, cfg, "val_selection")
        val_metrics.update(conformal_selective_metrics(val_rows, conformal_thresholds))
        val_metrics.update(
            risk_control_selective_metrics(val_rows, risk_control_thresholds)
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
        )
        enforce_failed_ratio(test_metrics, cfg, "test_clean")
        # Conformal selective metrics on clean test (test_rows is still clean
        # here -- robust rows are appended only inside the loop below).
        test_metrics.update(conformal_selective_metrics(test_rows, conformal_thresholds))
        test_metrics.update(
            risk_control_selective_metrics(test_rows, risk_control_thresholds)
        )
        if run_robust_test:
            assert test_ds is not None
            for robust_item in iter_robust_test_loaders(cfg, test_ds):
                result_key = robust_item["result_key"]
                if robust_item["perturb_type"] == "clean":
                    robust_results[result_key] = test_metrics
                    continue
                robust_loader = robust_item["loader"]
                assert robust_loader is not None
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
                    classification_log_odds_threshold=(
                        classification_log_odds_threshold
                    ),
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
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
    summary = {
        "metric_schema_version": METRIC_SUMMARY_SCHEMA_VERSION,
        "run_identity": build_run_identity(cfg, exp_name, seed),
        "eval_only": eval_only,
        "refit_posthoc_calibration": refit_posthoc_calibration,
        "refit_decision_calibration": refit_decision_calibration,
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
        "encoder_stage": {
            "mode": (
                "eval_checkpoint"
                if eval_only and not refit_posthoc_calibration
                else "eval_refit"
                if eval_only
                else encoder_stage_mode
            ),
            "protocol_id": str(
                encoder_stage_cfg.get(
                    "protocol_id",
                    "neutral_alive_uniform_clean_stage1_v2",
                )
            ),
            "source_path": (
                str(encoder_stage_source_path)
                if encoder_stage_source_path is not None
                else None
            ),
            "portable_artifact_path": (
                str(encoder_checkpoint_path)
                if encoder_checkpoint_path.is_file()
                else None
            ),
            "selected_epoch": ckpt.get("epoch"),
            "selection_metrics": copy.deepcopy(ckpt.get("val") or {}),
            "selection_score": ckpt.get("checkpoint_score"),
            "selection_metric": ckpt.get("checkpoint_metric"),
            "artifact_state_sha256": ckpt.get(
                "encoder_stage_state_sha256"
            ),
            "artifact_identity_sha256": ckpt.get(
                "encoder_stage_identity_sha256"
            ),
            "artifact_implementation_sha256": ckpt.get(
                "encoder_stage_implementation_sha256"
            ),
            "state_sha256_before_posthoc": (
                encoder_state_sha_before_posthoc
            ),
            "state_sha256_after_posthoc": (
                encoder_state_sha_after_posthoc
            ),
            "posthoc_preserved_encoder_state": bool(
                encoder_state_sha_before_posthoc
                and encoder_state_sha_before_posthoc
                == encoder_state_sha_after_posthoc
            ),
            "selection_to_posthoc_clean_macro_f1_delta": (
                float(val_metrics.get("macro_f1", 0.0))
                - float((ckpt.get("val") or {}).get("macro_f1", 0.0))
                if val_metrics and (ckpt.get("val") or {}).get("macro_f1") is not None
                else None
            ),
        },
        "decision_calibration_signature": _decision_calibration_signature(cfg),
        "decision_calibration_data_identity": decision_calibration_data_identity,
        "acceptance_comparison": _decision_calibration_signature(cfg)[
            "selective"
        ]["acceptance_comparison"],
        "calibration": calibration_summary,
        "classification_threshold": classification_threshold_summary,
        "conformal_thresholds": conformal_thresholds,
        "risk_control_thresholds": risk_control_thresholds,
        "val_selection": val_metrics,
        "val_posthoc_calibration": val_calibration_metrics,
        "val_conformal_calibration": val_conformal_metrics,
        "validation_split": validation_split_summary,
        "manifest_vocab_provenance": manifest_vocab_provenance,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing artifacts in an existing run directory.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help=(
            "Reuse a strict encoder-only Stage-1 artifact and run only "
            "post-hoc I1/I2/I3 fitting plus evaluation."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.encoder_checkpoint:
        cfg.setdefault("encoder_stage", {})["mode"] = "reuse"
        cfg["encoder_stage"]["checkpoint_path"] = args.encoder_checkpoint
    run(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
