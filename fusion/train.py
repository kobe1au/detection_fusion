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
    ROUTING_PROBABILITY_SUBSETS,
    compute_probability_calibration_loss,
    compute_reliability_calibration_loss,
    compute_robust_loss,
    reliability_alive_mask,
    reliability_correctness_target,
    reliability_per_sample_loss,
    resolve_auxiliary_weight_mode,
    routing_mixture_log_prob,
    routing_risk_per_sample_loss,
    routing_risk_target,
    routing_source_subset_oracle_target,
    routing_soft_oracle_per_sample_loss,
    routing_soft_oracle_target,
    routing_subset_oracle_per_sample_loss,
)
from fusion.dataset import (
    RobustTriModalDataset,
    prepare_robust_batch,
    robust_collate_fn,
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
}


class EmptyExtraEvalSetError(RuntimeError):
    """Raised when an optional external eval set has no usable samples."""


CHECKPOINT_STAGE_ENCODER_SELECTED = "encoder_selected"
CHECKPOINT_STAGE_PIPELINE_FITTED = "pipeline_fitted"
POSTHOC_OOF_ROWS_SCHEMA_VERSION = 3
METRIC_SUMMARY_SCHEMA_VERSION = 5
ACCEPTANCE_THRESHOLD_COMPARISON = "selective_eligible and score > threshold"
CLASSIFICATION_THRESHOLD_SELECTION_RULE = "macro_f1_unconstrained_v1"
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
    # PT-quality fields and model-confidence diagnostics.
    "q_api",
    "q_graph",
    "q_manifest",
    "q_align",
    "pert_api",
    "pert_graph",
    "pert_manifest",
    "api_manifest_consistency",
    "graph_manifest_consistency",
    "api_confidence",
    "graph_confidence",
    "manifest_confidence",
    "discount_api",
    "discount_graph",
    "discount_manifest",
    "fusion_weight_api",
    "fusion_weight_graph",
    "fusion_weight_manifest",
    "qmf_energy_api",
    "qmf_energy_graph",
    "qmf_energy_manifest",
    "entropy_api",
    "entropy_graph",
    "entropy_manifest",
    "margin_api",
    "margin_graph",
    "margin_manifest",
    "uncertainty_proxy_api",
    "uncertainty_proxy_graph",
    "uncertainty_proxy_manifest",
    "zero_weight_fallback_used",
    "predicted_reliability_api",
    "predicted_reliability_graph",
    "predicted_reliability_manifest",
    "embedding_in_distribution_score_api",
    "embedding_in_distribution_score_graph",
    "embedding_in_distribution_score_manifest",
    "embedding_mahalanobis_distance_api",
    "embedding_mahalanobis_distance_graph",
    "embedding_mahalanobis_distance_manifest",
    "embedding_density_feature_active",
    "effective_trust_cap_api",
    "effective_trust_cap_graph",
    "effective_trust_cap_manifest",
    "weight_sharpening_gamma",
    "temperature_api",
    "temperature_graph",
    "temperature_manifest",
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
    "routing_weight_risk",
    "routing_risk_probability",
    "routing_committed_mass",
    "routing_mixture_prob_malware",
    "routing_mixture_pred",
    "routing_risk_mode_learned",
    "routing_risk_mode_reliability_prior",
    "routing_risk_mode_disabled",
    "routing_learned_components_active",
    "routing_has_available",
    "routing_risk_reliability_deficit",
    "routing_risk_uncertainty_burden",
    "routing_risk_decision_boundary_proximity",
    "routing_risk_predicted_malware",
    "routing_risk_decision_log_odds_threshold",
    "routing_risk_decision_threshold_active",
    "routing_risk_target_mixture_argmax_error",
    "routing_risk_target_threshold_classification_error",
    "routing_risk_target_threshold_malware_false_negative",
    "routing_risk_target_reliability_deficit_score",
    "routing_risk_structural_conflict",
    "routing_risk_missing_fraction",
    "routing_route_prior_beta",
    "routing_prior_only_odds_beta",
    "routing_prior_only_odds_beta_active",
    "routing_conflict_penalty_mean",
    "routing_mean_disagreement",
    "routing_route_conflict_feature_active",
    "routing_route_conflict_feature_configured",
    "routing_risk_conflict_feature_active",
    "routing_risk_conflict_feature_configured",
    "routing_common_scale_reliability_active",
    "routing_prefit_uniform_prior_active",
    "routing_mode_learned",
    "routing_mode_prior_only",
    "routing_train_end_to_end",
    "routing_posthoc_refine",
    "final_temperature",
    "acceptance_score",
    "acceptance_score_fused_risk",
    "acceptance_score_mixture_certainty",
    "mixture_uncertainty_burden",
    "acceptance_score_product",
    "acceptance_score_pretrust_conflict",
    "acceptance_score_trusted_conflict",
    "calibration_active",
    "gate_uses_perturbation_evidence",
    "explicit_relation_factors_active",
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
    neutral-boundary tie break.  ``min_malware_recall: null`` is accepted only
    so inherited defaults can be normalized away; any non-null value is a hard
    configuration error.
    """

    out = copy.deepcopy(cfg or {})
    raw = out.get("classification_threshold", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("classification_threshold must be a mapping")
    raw = copy.deepcopy(raw)
    if raw.get("min_malware_recall") is not None:
        raise ValueError(
            "classification_threshold.min_malware_recall was removed: "
            "classification selects unconstrained macro-F1 and I3 alone "
            "controls malware false-negative risk"
        )
    raw.pop("min_malware_recall", None)
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
    loss_cfg = cfg.get("loss")
    if isinstance(loss_cfg, dict):
        removed = sorted(set(loss_cfg) & REMOVED_LOSS_CONFIG_KEYS)
        if removed:
            raise ValueError(
                "Removed loss configuration keys are unsupported: "
                f"{removed}. Use loss.auxiliary_weight_mode for branch weighting."
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


def _checkpoint_semantic_signature(cfg: dict) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    method_cfg = cfg.get("method", {}) or {}
    fusion_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
    # Acceptance-score choices change only the downstream decision artifact,
    # not learned model state.  They are guarded by the decision signature so
    # an eval-only run can intentionally refit thresholds without pretending
    # the encoder/fusion checkpoint is incompatible.
    fusion_cfg.pop("acceptance_aggregation", None)
    routing_cfg = fusion_cfg.get("routing")
    if isinstance(routing_cfg, dict):
        routing_cfg.pop("acceptance_score_mode", None)
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
        "calibration": copy.deepcopy(cfg.get("calibration", {}) or {}),
        "data": {
            key: copy.deepcopy(data_cfg.get(key))
            for key in (
                "graph_semantic_source",
                "max_api_events_per_sample",
                "label_map",
                "require_manifest_vocab_provenance",
                "expected_manifest_vocab_sha256",
                "expected_manifest_train_csv_sha256",
                "expected_manifest_train_sample_ids_sha256",
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
        if str(fusion_cfg.get("combination", "linear")).lower() == "routed":
            routing_cfg = fusion_cfg.get("routing", {}) or {}
            score_definition = str(
                routing_cfg.get("acceptance_score_mode", "product")
            ).lower()
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
        threshold_aligned_risk = str(
            current_routing.get("risk_target", "")
        ).strip().lower() in {
            "threshold_classification_error",
            "threshold_malware_false_negative",
        } and str(current_routing.get("risk_mode", "learned")).lower() == "learned"
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
        str(fusion_cfg.get("combination", "linear")).lower() != "routed"
        or not bool(routing_cfg.get("enabled", False))
        or str(routing_cfg.get("risk_mode", "learned")).lower() != "learned"
        or str(routing_cfg.get("risk_target", "")).lower()
        not in {
            "threshold_classification_error",
            "threshold_malware_false_negative",
        }
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
    top_level = model_cfg.get("max_nodes_gnn")
    nested = graph_cfg.get("max_nodes")
    if top_level is None and nested is None:
        return 12288
    top_value = (
        strict_finite_integer(top_level, field_name="model.max_nodes_gnn")
        if top_level is not None
        else None
    )
    nested_value = (
        strict_finite_integer(
            nested, field_name="model.graph_encoder.max_nodes"
        )
        if nested is not None
        else None
    )
    if top_value is not None and nested_value is not None and top_value != nested_value:
        raise ValueError(
            "Conflicting graph budgets: model.max_nodes_gnn and "
            "model.graph_encoder.max_nodes must match when both are set; "
            f"got {top_value} and {nested_value}"
        )
    budget = top_value if top_value is not None else nested_value
    assert budget is not None
    if budget <= 0:
        raise ValueError("The graph node budget must be a positive integer")
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
        "graph_semantic_source",
        "strict_split_integrity",
        "strict_partition_isolation",
        "allow_pt_superset",
        "label_map",
        "manifest_vocab_path",
        "require_manifest_vocab_provenance",
        "expected_manifest_vocab_sha256",
        "expected_manifest_train_csv_sha256",
        "expected_manifest_train_sample_ids_sha256",
    }
    unknown_data_keys = sorted(set(data_cfg) - allowed_data_keys)
    if unknown_data_keys:
        raise ValueError(f"Unsupported data settings: {unknown_data_keys}")
    robust_cfg = cfg.get("robust", {})
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
    account_for_graph_budget = bool(
        model_cfg.get("graph_encoder", {}).get(
            "account_for_encoder_budget", True
        )
    )
    if not account_for_graph_budget:
        raise ValueError(
            "model.graph_encoder.account_for_encoder_budget=false is unsupported: "
            "encoder-only truncation would leave graph coverage/reliability stale"
        )
    graph_node_budget = _resolve_graph_node_budget(model_cfg)
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
        "max_api_events_per_sample": api_event_budget,
        "max_graph_nodes_per_sample": (
            graph_node_budget
            if account_for_graph_budget
            else None
        ),
        "drop_graph_behavior_hints": bool(model_cfg.get("graph_encoder", {}).get("drop_extracted_behavior_hints", False)),
        "graph_semantic_source": str(data_cfg.get("graph_semantic_source", "alignment")),
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
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and workers > 0,
        "collate_fn": collate_fn_override or robust_collate_fn,
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


def _build_eval_perturbation_view(
    base_dataset: RobustTriModalDataset,
    *,
    perturb_type: str,
    perturb_strength: float,
) -> RobustTriModalDataset:
    """Clone an already validated eval dataset without rescanning its PT pool.

    ``RobustTriModalDataset.__getitem__`` does not mutate dataset state, so the
    sample/index/path metadata can be shared safely across deterministic eval
    views. Only the four execution-mode scalars differ between views.
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
    view.robust_aug = False
    view.eval_perturb_type = perturb_type
    view.eval_perturb_strength = strength
    return view


def build_robust_val_loaders(
    cfg: dict,
    base_dataset: RobustTriModalDataset,
    subset_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    robust_val_cfg = cfg.get("eval", {}).get("robust_val", {}) or {}
    if not bool(robust_val_cfg.get("enabled", False)):
        return []
    out: list[dict[str, Any]] = []
    for item in _normalize_robust_val_scenarios(robust_val_cfg.get("scenarios")):
        dataset = _build_eval_perturbation_view(
            base_dataset,
            perturb_type=item["perturb_type"],
            perturb_strength=item["strength"],
        )
        if subset_indices is not None:
            dataset = Subset(dataset, subset_indices)
        out.append({**item, "loader": build_loader(cfg, dataset, is_train=False)})
    return out


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
    perturb_tests = list(eval_cfg.get("perturb_tests", ["clean"]))
    if eval_cfg.get("perturb_strengths") is not None:
        perturb_strengths = [
            float(value) for value in eval_cfg.get("perturb_strengths") or []
        ]
    else:
        perturb_strengths = [float(eval_cfg.get("perturb_strength", 0.5))]
    perturb_strengths = perturb_strengths or [0.5]

    for perturb in perturb_tests:
        if perturb == "clean":
            yield {
                "result_key": "clean",
                "perturb_type": "clean",
                "strength": 0.0,
                "loader": None,
            }
            continue
        # Missing/dropout transforms ignore their requested strength. Keep the
        # existing one-run contract so identical views are never reevaluated.
        is_strength_invariant = perturb.endswith("_missing") or perturb.startswith(
            "modality_dropout_"
        )
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
    # I1 receives explicit, branch-local completeness perturbations. The model
    # sees only signals recomputed from the transformed view; requested severity,
    # augmentation names, and pert_* markers are never inputs. Single-branch
    # semantic views provide ordinary correctness supervision only to the
    # affected branch; joint semantic corruption remains I2-only.
    "api_event_dropout": ("api",),
    "api_category_dropout": ("api",),
    "graph_sparsify": ("graph",),
    "graph_node_feature_mask": ("graph",),
    "manifest_permission_mask": ("manifest",),
    "manifest_intent_mask": ("manifest",),
    "manifest_component_mask": ("manifest",),
    # Pairwise and all-branch completeness views are I2-only.  Including all
    # three pairs avoids targeting a single weakness observed on one test cell.
    "api_graph_degraded": (),
    "api_manifest_degraded": (),
    "graph_manifest_degraded": (),
    "all_degraded": (),
    "api_semantic_corrupted": ("api",),
    "graph_semantic_corrupted": ("graph",),
    "manifest_semantic_corrupted": ("manifest",),
    "all_semantic_corrupted": (),
}
RELIABILITY_CALIBRATION_MISSING = (
    "api_missing",
    "graph_missing",
    "manifest_missing",
)

ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION = {
    "api_event_dropout": "local_completeness",
    "api_category_dropout": "local_completeness",
    "graph_sparsify": "local_completeness",
    "graph_node_feature_mask": "local_completeness",
    "manifest_permission_mask": "local_completeness",
    "manifest_intent_mask": "local_completeness",
    "manifest_component_mask": "local_completeness",
    "api_graph_degraded": "combined_completeness",
    "api_manifest_degraded": "combined_completeness",
    "graph_manifest_degraded": "combined_completeness",
    "all_degraded": "combined_completeness",
    "api_semantic_corrupted": "single_semantic",
    "graph_semantic_corrupted": "single_semantic",
    "manifest_semantic_corrupted": "single_semantic",
    "all_semantic_corrupted": "joint_semantic",
    "api_missing": "missing",
    "graph_missing": "missing",
    "manifest_missing": "missing",
}
ROUTING_ROBUSTNESS_FAMILIES = (
    "local_completeness",
    "combined_completeness",
    "single_semantic",
    "joint_semantic",
    "missing",
)
ROUTING_PAIRWISE_COMPLETENESS = frozenset(
    {
        "api_graph_degraded",
        "api_manifest_degraded",
        "graph_manifest_degraded",
    }
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
TRAIN_LOG_DIAGNOSTIC_KEYS = tuple(
    key
    for key in GATE_DIAGNOSTIC_KEYS
    if key.startswith(
        ("discount_", "fusion_weight_", "entropy_", "margin_", "uncertainty_proxy_")
    )
    or key in {
        "zero_weight_fallback_used",
        "routing_prefit_uniform_prior_active",
    }
)


def reliability_calibration_scenarios(cfg: dict) -> list[dict[str, Any]]:
    """Build transformed post-hoc views used by the global opinion router.

    These views are built only from the post-hoc calibration subset. They never
    touch checkpoint selection or the disjoint decision-calibration subset.
    ``reliability_branches`` declares only branches whose transformation changes
    deployment-observable quality. I1 may use those rows with a proper
    correctness score; all views remain available to I2. Each single-branch
    semantic view may supervise only its affected I1 branch, including the
    no-density negative-control cell. Joint semantic corruption remains I2-only.
    No perturbation name, strength, pre-transform count, or other-modality
    signal is exposed as an I1 feature.
    """
    robust_cfg = cfg.get("robust", {}) or {}
    calibration_cfg = cfg.get("calibration", {}) or {}
    fusion_cfg = cfg.get("fusion", {}) or {}
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    routed_posthoc = bool(
        str(fusion_cfg.get("combination", "linear")).strip().lower() == "routed"
        and routing_cfg.get("enabled", False)
        and routing_cfg.get("posthoc_refine", True)
    )
    route_distribution_active = bool(
        routed_posthoc
        and str(routing_cfg.get("mode", "learned")).strip().lower() == "learned"
        and (
            float(routing_cfg.get("prediction_loss_weight", 1.0)) > 0.0
            or float(routing_cfg.get("route_oracle_loss_weight", 0.0)) > 0.0
            or float(routing_cfg.get("subset_oracle_loss_weight", 0.0)) > 0.0
        )
    )
    risk_fit_active = bool(
        routed_posthoc
        and str(routing_cfg.get("risk_mode", "learned")).strip().lower()
        == "learned"
        and float(routing_cfg.get("risk_loss_weight", 1.0)) > 0.0
    )
    i1_fit_active = bool(reliability_cfg.get("enabled", False))
    include_pairwise = calibration_cfg.get(
        "include_pairwise_completeness_views", False
    )
    if not isinstance(include_pairwise, bool):
        raise ValueError(
            "calibration.include_pairwise_completeness_views must be boolean"
        )
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
    reliability_perturbations = dict(RELIABILITY_CALIBRATION_PERTURBATIONS)
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
        for perturb_type, branches in reliability_perturbations.items()
        if (
            (i1_fit_active and bool(branches))
            or route_distribution_active
            or risk_fit_active
        )
        if (
            perturb_type not in ROUTING_PAIRWISE_COMPLETENESS
            or (route_distribution_active and include_pairwise)
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


@torch.no_grad()
def evaluate_robust_validation(
    model,
    loaders: list[dict[str, Any]],
    device,
    use_amp: bool,
    cfg: dict,
    selective_threshold: float | None = None,
    selective_score_type: str = "msp",
    classification_threshold: float = 0.5,
    classification_log_odds_threshold: float | None = None,
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
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
            "must be 2 because probability calibration, classification "
            "thresholding, malware-FN risk, and I3 all use the benign/malware "
            "contract explicitly."
        )
    api_cfg = model_cfg.get("api_encoder", {})
    graph_cfg = model_cfg.get("graph_encoder", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    gate_cfg = model_cfg.get("gate", {})
    if not bool(graph_cfg.get("account_for_encoder_budget", True)):
        raise ValueError(
            "model.graph_encoder.account_for_encoder_budget=false is unsupported: "
            "encoder-only truncation would leave graph coverage/reliability stale"
        )
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
    if "apply_alive_mask" in gate_cfg:
        removed_joint_config.append("model.gate.apply_alive_mask")
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
    if fusion_mode == "tri_modal_dense_embedding_gate":
        unused_dense_gate_flags = [
            key
            for key in (
                "use_consistency_evidence",
                "use_conflict_evidence",
                "use_perturbation_evidence",
            )
            if bool(gate_cfg.get(key, False))
        ]
        if unused_dense_gate_flags:
            raise ValueError(
                "tri_modal_dense_embedding_gate consumes only branch embeddings "
                "and the mandatory alive mask; these evidence flags would be "
                f"silent no-ops: {unused_dense_gate_flags}. Set them to false."
            )

    # Input-duplication guardrail: Graph may be structurally selected around API
    # methods, but fine-grained API behavior hints must not be copied directly
    # into graph node features when the branches are treated as distinct views.
    combination = str(fusion_cfg.get("combination", "linear")).lower()
    routing_cfg = fusion_cfg.get("routing", {}) or {}
    reliability_cfg = fusion_cfg.get("reliability_calibration", {}) or {}
    removed_fusion_keys = sorted(
        set(fusion_cfg) & {"branch_competence_prior", "visible_integrity_modifier"}
    )
    removed_reliability_keys = sorted(
        set(reliability_cfg)
        & {
            "feature_schema",
            "missing_relation_support",
            "use_relation_evidence",
            "use_edl_certainty_feature",
            "use_evidential_uncertainty",
            "group_mean_alignment",
        }
    )
    if removed_fusion_keys or removed_reliability_keys:
        raise ValueError(
            "Removed fusion/I1 configuration keys are unsupported; delete them "
            "instead of setting them to false: "
            f"fusion={removed_fusion_keys}, reliability_calibration="
            f"{removed_reliability_keys}"
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
        ignored_non_neutral: list[str] = []
        for key in (
            "use_support_discount",
            "use_conflict_discount",
            "use_confidence_proxy",
        ):
            if bool(fusion_cfg.get(key, False)):
                ignored_non_neutral.append(f"fusion.{key}=true")
        for key in (
            "reliability_discount_exponent",
            "weight_sharpening_gamma",
        ):
            value = float(fusion_cfg.get(key, 1.0))
            if not math.isfinite(value) or value != 1.0:
                ignored_non_neutral.append(f"fusion.{key}={value!r}")
        for key, neutral in (
            ("detach_discount", True),
            ("detach_confidence_proxy", True),
            ("acceptance_aggregation", "product"),
            ("fallback", "uniform"),
        ):
            value = fusion_cfg.get(key, neutral)
            if isinstance(neutral, bool):
                is_neutral = isinstance(value, bool) and value is neutral
            else:
                is_neutral = str(value).strip().lower() == str(neutral)
            if not is_neutral:
                ignored_non_neutral.append(f"fusion.{key}={value!r}")
        if "use_reliability_acceptance" in fusion_cfg:
            ignored_non_neutral.append(
                "fusion.use_reliability_acceptance is linear-path-only"
            )

        # These mappings are injected by the shared defaults for linear
        # comparisons.  Routed fusion does not read them; permit only their
        # canonical neutral/default contents so an edited value cannot be
        # silently mistaken for part of the routed method.
        routed_ignored_mapping_defaults = {
            "confidence_proxy": {
                "type": "entropy_margin",
                "temperature_api": 1.0,
                "temperature_graph": 1.0,
                "temperature_manifest": 1.0,
            },
            "support_factor": {
                "manifest_support_base": 0.5,
                "code_anchor_base": 0.5,
            },
            "conflict_factor": {"min_value": 0.05},
        }
        for section, neutral_values in routed_ignored_mapping_defaults.items():
            values = fusion_cfg.get(section, {}) or {}
            if not isinstance(values, dict):
                ignored_non_neutral.append(f"fusion.{section} must be a mapping")
                continue
            unknown = sorted(set(values) - set(neutral_values))
            if unknown:
                ignored_non_neutral.append(
                    f"fusion.{section} has unsupported routed keys {unknown}"
                )
            for key, neutral in neutral_values.items():
                if key not in values:
                    continue
                value = values[key]
                if isinstance(neutral, str):
                    is_neutral = str(value).strip().lower() == neutral
                else:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        is_neutral = False
                    else:
                        is_neutral = math.isfinite(numeric) and numeric == neutral
                if not is_neutral:
                    ignored_non_neutral.append(
                        f"fusion.{section}.{key}={value!r}"
                    )
        if ignored_non_neutral:
            raise ValueError(
                "fusion.combination='routed' does not consume these legacy "
                "linear-path settings; remove or neutralize them: "
                + "; ".join(ignored_non_neutral)
            )

        risk_mode = str(routing_cfg.get("risk_mode", "learned")).lower()
        if risk_mode == "learned" and "risk_target" not in routing_cfg:
            raise ValueError(
                "learned routed risk requires an explicit "
                "fusion.routing.risk_target; no implicit legacy target is "
                "accepted"
            )
        risk_target = str(
            routing_cfg.get("risk_target", "mixture_argmax_error")
        ).strip().lower()
        supported_risk_targets = {
            "mixture_argmax_error",
            "threshold_classification_error",
            "threshold_malware_false_negative",
            "reliability_deficit_score",
        }
        if risk_target not in supported_risk_targets:
            raise ValueError(
                "fusion.routing.risk_target must be one of "
                f"{sorted(supported_risk_targets)}"
            )
        if learned_risk and risk_target == "reliability_deficit_score":
            raise ValueError(
                "learned routed risk cannot use the unfitted "
                "reliability_deficit_score target"
            )
        if risk_mode == "reliability_prior" and risk_target != "reliability_deficit_score":
            raise ValueError(
                "fusion.routing.risk_mode='reliability_prior' requires "
                "risk_target='reliability_deficit_score' so summaries do not "
                "mislabel an uncalibrated control score"
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
        subset_oracle_weight = float(
            routing_cfg.get("subset_oracle_loss_weight", 0.0)
        )
        subset_oracle_temperature = float(
            routing_cfg.get("subset_oracle_temperature", 1.0)
        )
        if not math.isfinite(subset_oracle_weight) or subset_oracle_weight < 0.0:
            raise ValueError(
                "fusion.routing.subset_oracle_loss_weight must be finite and non-negative"
            )
        if (
            not math.isfinite(subset_oracle_temperature)
            or subset_oracle_temperature <= 0.0
        ):
            raise ValueError(
                "fusion.routing.subset_oracle_temperature must be finite and positive"
            )
        if subset_oracle_weight == 0.0 and subset_oracle_temperature != 1.0:
            raise ValueError(
                "fusion.routing.subset_oracle_temperature is inactive when "
                "subset_oracle_loss_weight=0; set it to the neutral value 1.0"
            )
        if subset_oracle_weight > 0.0 and (
            not learned_route or not bool(routing_cfg.get("posthoc_refine", True))
        ):
            raise ValueError(
                "fusion.routing.subset_oracle_loss_weight>0 requires the learned "
                "post-hoc routing distribution"
            )

        group_robust_cfg = routing_cfg.get("group_robust_objective", {}) or {}
        if not isinstance(group_robust_cfg, dict):
            raise ValueError(
                "fusion.routing.group_robust_objective must be a mapping"
            )
        unknown_group_robust = set(group_robust_cfg) - {
            "enabled",
            "taxonomy",
            "soft_worst_weight",
            "temperature",
            "apply_to",
        }
        if unknown_group_robust:
            raise ValueError(
                "Unsupported fusion.routing.group_robust_objective keys: "
                f"{sorted(unknown_group_robust)}"
            )
        group_robust_enabled = bool(group_robust_cfg.get("enabled", False))
        group_taxonomy = str(
            group_robust_cfg.get("taxonomy", "perturb_type_v1")
        ).strip().lower()
        if group_taxonomy not in {
            "perturb_type_v1",
            "perturb_type_strength_v1",
            "robustness_family_v1",
        }:
            raise ValueError(
                "fusion.routing.group_robust_objective.taxonomy must be "
                "'perturb_type_v1', 'perturb_type_strength_v1', or "
                "'robustness_family_v1'"
            )
        soft_worst_weight = float(
            group_robust_cfg.get("soft_worst_weight", 0.0)
        )
        soft_worst_temperature = float(group_robust_cfg.get("temperature", 0.1))
        if (
            not math.isfinite(soft_worst_weight)
            or not 0.0 <= soft_worst_weight <= 1.0
        ):
            raise ValueError(
                "fusion.routing.group_robust_objective.soft_worst_weight "
                "must be within [0, 1]"
            )
        if (
            not math.isfinite(soft_worst_temperature)
            or soft_worst_temperature <= 0.0
        ):
            raise ValueError(
                "fusion.routing.group_robust_objective.temperature must be "
                "finite and positive"
            )
        apply_to = group_robust_cfg.get(
            "apply_to", ["routing_distribution"]
        )
        if not isinstance(apply_to, list) or apply_to != ["routing_distribution"]:
            raise ValueError(
                "fusion.routing.group_robust_objective.apply_to currently must be "
                "['routing_distribution']"
            )
        if group_robust_enabled and (
            not learned_route
            or not bool(routing_cfg.get("posthoc_refine", True))
            or perturb_weight <= 0.0
        ):
            raise ValueError(
                "enabled group-robust routing requires a learned post-hoc route "
                "and positive perturb objective mass"
            )
        if not group_robust_enabled and soft_worst_weight != 0.0:
            raise ValueError(
                "fusion.routing.group_robust_objective.soft_worst_weight is "
                "inactive when enabled=false; set it to 0.0"
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
                "Disable them, or use fusion.combination=linear for a coupled comparison pipeline."
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
        if bool(fusion_cfg.get("use_reliability_discount", False)):
            violations.append("fusion.use_reliability_discount must be false")
        if bool(fusion_cfg.get("use_support_discount", False)):
            violations.append("fusion.use_support_discount must be false")
        if bool(fusion_cfg.get("use_conflict_discount", False)):
            violations.append("fusion.use_conflict_discount must be false")
        if not bool(fusion_cfg.get("use_hard_alive_mask", True)):
            violations.append("fusion.use_hard_alive_mask must be true")
        if bool((fusion_cfg.get("reliability_calibration", {}) or {}).get("enabled", False)):
            violations.append("fusion.reliability_calibration.enabled must be false")
        if bool((fusion_cfg.get("probability_calibration", {}) or {}).get("enabled", False)):
            violations.append("fusion.probability_calibration.enabled must be false")
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
        use_consistency_evidence=bool(gate_cfg.get("use_consistency_evidence", True)),
        use_conflict_evidence=bool(gate_cfg.get("use_conflict_evidence", True)),
        use_perturbation_evidence=bool(gate_cfg.get("use_perturbation_evidence", False)),
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
    """Return the hard availability gate used by every selective rule.

    Newly evaluated rows always carry ``selective_eligible``.  The alive-field
    fallback keeps analysis helpers useful for explicitly constructed rows,
    while malformed values fail instead of being silently interpreted as an
    acceptance score.
    """

    if "selective_eligible" in row:
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

    alive_keys = [f"{name}_alive" for name in ("api", "graph", "manifest")]
    present = [key in row for key in alive_keys]
    if any(present):
        if not all(present):
            raise ValueError(
                "selective eligibility requires all three modality alive fields"
            )
        alive_values = [_finite_row_float(row, key) for key in alive_keys]
        if any(value not in {0.0, 1.0} for value in alive_values):
            raise ValueError(
                "modality alive fields must contain finite binary values"
            )
        return any(bool(value) for value in alive_values)
    # Hand-built unit/analysis rows predate the explicit field and represent
    # ordinary available samples. Formal evaluate() output never takes this
    # fallback because it writes selective_eligible unconditionally.
    return True


def compute_branch_reliability_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit I1 correctness and the explicitly configured I2 risk event."""
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

    risk_values: list[float] = []
    risk_target_values: list[float] = []
    mixture_error_values: list[float] = []
    threshold_aligned_count = 0
    observed_risk_targets: set[str] = set()
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
        target_flags = {
            "mixture_argmax_error": _finite_row_float(
                row, "routing_risk_target_mixture_argmax_error"
            ),
            "threshold_classification_error": _finite_row_float(
                row,
                "routing_risk_target_threshold_classification_error",
            ),
            "threshold_malware_false_negative": _finite_row_float(
                row,
                "routing_risk_target_threshold_malware_false_negative",
            ),
            "reliability_deficit_score": _finite_row_float(
                row, "routing_risk_target_reliability_deficit_score"
            ),
        }
        active_targets = [
            name
            for name, value in target_flags.items()
            if value is not None and value >= 0.5
        ]
        if len(active_targets) > 1:
            raise ValueError(
                f"routed evaluation row declares multiple risk targets: {active_targets}"
            )
        risk_target = active_targets[0] if active_targets else "mixture_argmax_error"
        observed_risk_targets.add(risk_target)
        risk_values.append(min(1.0, max(0.0, risk)))
        mixture_error = float(
            int(round(mixture_pred)) != int(round(label))
        )
        mixture_error_values.append(mixture_error)
        if risk_target == "reliability_deficit_score":
            continue
        if risk_target.startswith("threshold_"):
            if (
                threshold_active is None
                or threshold_active < 0.5
                or final_pred is None
            ):
                raise ValueError(
                    f"risk_target={risk_target!r} requires an active decision "
                    "threshold and final predictions in evaluation rows"
                )
            threshold_aligned_count += 1
            if risk_target == "threshold_classification_error":
                risk_target_values.append(
                    float(int(round(final_pred)) != int(round(label)))
                )
            else:
                risk_target_values.append(
                    float(
                        int(round(label)) == 1
                        and int(round(final_pred)) == 0
                    )
                )
        else:
            risk_target_values.append(mixture_error)
    if risk_values:
        if len(observed_risk_targets) != 1:
            raise ValueError(
                "Evaluation rows mix incompatible routed risk targets: "
                f"{sorted(observed_risk_targets)}"
            )
        risk_target = next(iter(observed_risk_targets))
        risk_arr = np.asarray(risk_values, dtype=np.float64)
        mixture_error_arr = np.asarray(
            mixture_error_values, dtype=np.float64
        )
        out["routing_risk_count"] = int(risk_arr.size)
        out["routing_risk_target"] = risk_target
        out["routing_risk_mean"] = float(risk_arr.mean())
        out["routing_mixture_error_rate"] = float(
            mixture_error_arr.mean()
        )
        out["routing_risk_threshold_aligned_count"] = int(
            threshold_aligned_count
        )
        if risk_target == "reliability_deficit_score":
            out["routing_risk_calibration_defined"] = 0
        else:
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
    if isinstance(explicit, torch.Tensor):
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

    alive_columns: list[torch.Tensor] = []
    present_alive_keys: list[str] = []
    for name in ("api", "graph", "manifest"):
        key = f"{name}_alive"
        value = extra.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        present_alive_keys.append(key)
        flattened = value.detach().to(device=device).view(-1)
        if flattened.numel() != int(batch_size):
            raise ValueError(
                f"{key} must contain one availability value per sample"
            )
        if not bool(torch.isfinite(flattened.float()).all().item()):
            raise ValueError(f"{key} contains non-finite availability values")
        alive_columns.append(flattened > 0.0)
    if alive_columns:
        if len(alive_columns) != 3:
            raise ValueError(
                "selective eligibility requires API, Graph, and Manifest alive "
                f"diagnostics together; received {present_alive_keys}"
            )
        return torch.stack(alive_columns, dim=-1).any(dim=-1)

    routing_available = extra.get("routing_has_available")
    if isinstance(routing_available, torch.Tensor):
        flattened = routing_available.detach().to(device=device).view(-1)
        if flattened.numel() != int(batch_size) or not bool(
            torch.isfinite(flattened.float()).all().item()
        ):
            raise ValueError(
                "routing_has_available must contain one finite value per sample"
            )
        return flattened > 0.0

    # Non-TriModal test doubles may expose no availability diagnostics. They
    # represent ordinary eligible classifier outputs; every production model
    # path emits all three alive fields through build_evidence().
    return torch.ones(int(batch_size), dtype=torch.bool, device=device)


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
        if key not in row:
            continue
        value = _finite_row_float(row, key)
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be finite within [0, 1]")
        return float(value)
    return 0.0


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
    num_failed = 0

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, _sids, _quality, failed = prepare_robust_batch(
            batch, device
        )
        num_failed += int(failed)
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            logits, _extra = model(graph, return_features=False)
        probability_batches.append(
            torch.softmax(logits.float(), dim=-1)[:, 1]
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
    if classification_log_odds_threshold is not None and not math.isfinite(
        float(classification_log_odds_threshold)
    ):
        raise ValueError("classification_log_odds_threshold must be finite")

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, sids, quality, failed = prepare_robust_batch(batch, device)
        num_failed += failed
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            logits, extra = model(graph, return_features=False)
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
                    if not isinstance(value, torch.Tensor):
                        value = quality.get(key)
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
    if isinstance(cached.get("branch_embeddings"), dict):
        sliced["branch_embeddings"] = {
            name: value.index_select(0, indices)
            for name, value in cached["branch_embeddings"].items()
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


def _compile_group_robust_row_weights(
    objective_groups: list[dict[str, Any]],
    items: list[dict[str, Any]],
    segments: list[tuple[int, int]],
    valid_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]:
    """Compile clean and hierarchical family row reductions for soft worst-group.

    The hierarchy is family -> perturbation mechanism -> strength/source ->
    valid rows.  Consequently, adding another strength to one mechanism cannot
    silently increase that mechanism's objective mass.
    """
    if not objective_groups:
        raise RuntimeError("Group-robust routing requires objective groups")
    item_ids = {id(item) for item in items}
    if len(item_ids) != len(items):
        raise RuntimeError("Group-robust routing received duplicate sources")

    reference_clean = list(objective_groups[0].get("clean") or [])
    if not reference_clean:
        raise RuntimeError("Group-robust routing requires clean sources")
    reference_clean_ids = {id(item) for item in reference_clean}
    if any(
        {id(item) for item in (group.get("clean") or [])} != reference_clean_ids
        for group in objective_groups
    ):
        raise RuntimeError("Group-robust routing groups disagree on clean sources")

    clean_masses = {id(item): 0.0 for item in items}
    for item in reference_clean:
        if id(item) not in clean_masses:
            raise RuntimeError("Group-robust clean source is outside packed selection")
        clean_masses[id(item)] = 1.0 / float(len(reference_clean))
    clean_row_weights = _compile_posthoc_row_weights(
        items, segments, clean_masses, valid_mask
    )

    family_rows: list[torch.Tensor] = []
    resolved: list[dict[str, Any]] = []
    for group in objective_groups:
        mechanisms = list(group.get("mechanisms") or [])
        if not mechanisms:
            raise RuntimeError(
                f"Group-robust routing group {group.get('name')!r} has no mechanisms"
            )
        family_masses = {id(item): 0.0 for item in items}
        mechanism_scale = 1.0 / float(len(mechanisms))
        resolved_mechanisms: list[dict[str, Any]] = []
        seen_source_ids: set[int] = set()
        for mechanism in mechanisms:
            mechanism_name = str(mechanism.get("name") or "").strip().lower()
            mechanism_items = list(mechanism.get("items") or [])
            if not mechanism_name or not mechanism_items:
                raise RuntimeError("Group-robust mechanism metadata is incomplete")
            source_scale = mechanism_scale / float(len(mechanism_items))
            for item in mechanism_items:
                item_id = id(item)
                if item_id not in family_masses:
                    raise RuntimeError(
                        "Group-robust mechanism references a source outside the packed stage"
                    )
                if item_id in seen_source_ids:
                    raise RuntimeError(
                        "A non-clean source appears more than once inside one robust family"
                    )
                seen_source_ids.add(item_id)
                family_masses[item_id] = source_scale
            resolved_mechanisms.append(
                {
                    "name": mechanism_name,
                    "num_sources": len(mechanism_items),
                    "strengths": sorted(
                        {float(item.get("strength", 0.0)) for item in mechanism_items}
                    ),
                }
            )
        scenario_ids = {id(item) for item in (group.get("scenario") or [])}
        if seen_source_ids != scenario_ids:
            raise RuntimeError(
                "Group-robust family mechanisms do not partition its scenario sources"
            )
        family_rows.append(
            _compile_posthoc_row_weights(
                items, segments, family_masses, valid_mask
            )
        )
        resolved.append(
            {
                "name": str(group.get("objective_family") or group.get("name")),
                "prior_group": str(
                    group.get("prior_group")
                    or group.get("objective_family")
                    or group.get("name")
                ),
                "num_mechanisms": len(resolved_mechanisms),
                "num_sources": len(seen_source_ids),
                "mechanisms": resolved_mechanisms,
            }
        )
    # A strength-resolved taxonomy must not give a five-strength perturbation
    # type five times the *mean-objective* prior mass of a one-cell missingness
    # type. First balance parent perturbation types, then balance cells within
    # each parent. The same prior also anchors the entropic soft maximum.
    prior_counts: dict[str, int] = {}
    for item in resolved:
        prior_group = str(item["prior_group"])
        prior_counts[prior_group] = prior_counts.get(prior_group, 0) + 1
    parent_mass = 1.0 / float(len(prior_counts))
    family_priors = torch.tensor(
        [
            parent_mass / float(prior_counts[str(item["prior_group"])])
            for item in resolved
        ],
        device=clean_row_weights.device,
        dtype=clean_row_weights.dtype,
    )
    for item, prior_mass in zip(resolved, family_priors.detach().cpu().tolist()):
        item["prior_mass"] = float(prior_mass)
    return (
        clean_row_weights,
        torch.stack(family_rows, dim=0),
        family_priors,
        resolved,
    )


def _entropic_soft_worst_group(
    group_losses: torch.Tensor,
    *,
    soft_worst_weight: float,
    temperature: float,
    group_priors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend mean family risk with a stateless log-mean-exp worst-family risk."""
    if group_losses.ndim != 1 or group_losses.numel() <= 0:
        raise ValueError("group_losses must be a non-empty vector")
    rho = float(soft_worst_weight)
    tau = float(temperature)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("soft_worst_weight must be within [0, 1]")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("soft-worst temperature must be finite and positive")
    if group_priors is None:
        priors = torch.full_like(
            group_losses, 1.0 / float(group_losses.numel())
        )
    else:
        priors = group_priors.to(
            device=group_losses.device, dtype=group_losses.dtype
        ).view(-1)
        if priors.shape != group_losses.shape:
            raise ValueError("group_priors must match group_losses")
        if (
            not bool(torch.isfinite(priors).all().item())
            or bool((priors <= 0.0).any().item())
        ):
            raise ValueError("group_priors must be finite and strictly positive")
        priors = priors / priors.sum()
    mean_risk = torch.dot(priors, group_losses)
    log_mean_exp = tau * torch.logsumexp(
        group_losses / tau + priors.log(), dim=0
    )
    reduced = (1.0 - rho) * mean_risk + rho * log_mean_exp
    effective_weights = (
        (1.0 - rho) * priors
        + rho
        * torch.softmax(group_losses.detach() / tau + priors.log(), dim=0)
    )
    return reduced, effective_weights.detach()


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
            "calibration.enabled=true requires reliability, probability, routing, "
            "or final-temperature calibration parameters"
        )
    stage_optimization_cfg = calibration_cfg.get("stage_optimization", {}) or {}
    if not isinstance(stage_optimization_cfg, dict):
        raise ValueError("calibration.stage_optimization must be a mapping")
    legacy_default_steps = int(calibration_cfg.get("epochs", 300))
    if legacy_default_steps <= 0:
        raise ValueError("calibration.epochs must be positive when provided")

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
        calibration_sources.append(source)

    reliability_calibrator_for_cache = getattr(
        discount_fusion, "reliability_calibrator", None
    )
    cache_branch_embeddings = bool(
        getattr(reliability_calibrator_for_cache, "use_embedding_density", False)
    )

    # Cache the frozen encoders' branch logits and observable evidence once.
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

                graph, labels, sids, _quality, failed = prepare_robust_batch(
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
                    _logits, extra = model(
                        graph,
                        return_features=cache_branch_embeddings,
                    )
                evidence = extra.get("gate_evidence")
                branch_logits = {
                    name: extra.get(f"{name}_logits_aux")
                    for name in ("api", "graph", "manifest")
                }
                branch_embeddings = {
                    name: extra.get(f"{name}_emb")
                    for name in ("api", "graph", "manifest")
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
                        "branch_embeddings": (
                            {
                                name: value.detach().float()
                                for name, value in branch_embeddings.items()
                            }
                            if cache_branch_embeddings
                            and all(
                                isinstance(value, torch.Tensor)
                                for value in branch_embeddings.values()
                            )
                            else None
                        ),
                    }
                )
                if (
                    cache_branch_embeddings
                    and cached_batches[-1]["branch_embeddings"] is None
                ):
                    raise RuntimeError(
                        "I1 embedding density requires all branch embeddings in "
                        "the post-hoc cache"
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
                "evidence": torch.cat(
                    [item["evidence"] for item in source_batches], dim=0
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
                "branch_embeddings": (
                    {
                        name: torch.cat(
                            [item["branch_embeddings"][name] for item in source_batches],
                            dim=0,
                        )
                        for name in ("api", "graph", "manifest")
                    }
                    if cache_branch_embeddings
                    else None
                ),
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

    # Validation-global visibility modifiers were removed from the strict OOF
    # routed path.  They duplicate I1 quality and cannot be applied to an outer
    # holdout without a fold-local reference fit (guarded in build_model).

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
        max_steps = int(resolved.get("max_steps", legacy_default_steps))
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
        evidence = cached["evidence"]
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
        embedding_kwargs: dict[str, Any] = {}
        if isinstance(cached.get("branch_embeddings"), dict):
            embedding_kwargs["embeddings"] = cached["branch_embeddings"]
        outputs = model.discount_fusion(
            branch_logits["api"],
            branch_logits["graph"],
            branch_logits["manifest"],
            evidence,
            **embedding_kwargs,
            **override_kwargs,
        )
        outputs.update(
            {
                f"{name}_logits_aux": value
                for name, value in branch_logits.items()
            }
        )
        outputs["gate_evidence"] = evidence
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
                "evidence": torch.cat(
                    [item["evidence"] for item in items], dim=0
                ),
                "branch_logits": {
                    name: torch.cat(
                        [item["branch_logits"][name] for item in items], dim=0
                    )
                    for name in ("api", "graph", "manifest")
                },
            }
            branch_embedding_values = [
                item.get("branch_embeddings") for item in items
            ]
            if any(value is not None for value in branch_embedding_values):
                if any(not isinstance(value, dict) for value in branch_embedding_values):
                    raise RuntimeError(
                        "Packed calibration sources disagree on branch embeddings"
                    )
                packed["branch_embeddings"] = {
                    name: torch.cat(
                        [value[name] for value in branch_embedding_values], dim=0
                    )
                    for name in ("api", "graph", "manifest")
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

    def _normalized_group_robust_objective(raw: Any) -> dict[str, Any]:
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(
                "fusion.routing.group_robust_objective must be a mapping"
            )
        enabled = bool(raw.get("enabled", False))
        taxonomy = str(raw.get("taxonomy", "perturb_type_v1")).strip().lower()
        soft_worst_weight = float(raw.get("soft_worst_weight", 0.0))
        temperature = float(raw.get("temperature", 0.1))
        apply_to = list(raw.get("apply_to", ["routing_distribution"]))
        if taxonomy not in {
            "perturb_type_v1",
            "perturb_type_strength_v1",
            "robustness_family_v1",
        }:
            raise ValueError("Unsupported route group-robust taxonomy")
        if (
            not math.isfinite(soft_worst_weight)
            or not 0.0 <= soft_worst_weight <= 1.0
        ):
            raise ValueError("route group-robust soft_worst_weight must be within [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("route group-robust temperature must be finite and positive")
        if not enabled and soft_worst_weight != 0.0:
            raise ValueError(
                "disabled route group-robust objective requires "
                "soft_worst_weight=0"
            )
        if apply_to != ["routing_distribution"]:
            raise ValueError(
                "route group-robust apply_to currently must be ['routing_distribution']"
            )
        return {
            "enabled": enabled,
            "taxonomy": taxonomy,
            "soft_worst_weight": soft_worst_weight,
            "temperature": temperature,
            "apply_to": apply_to,
        }

    routing_group_robust = _normalized_group_robust_objective(
        (((cfg.get("fusion", {}) or {}).get("routing", {}) or {}).get(
            "group_robust_objective"
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
        # change their mass.  The clean/degradation deployment prior is an
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
                f"Post-hoc calibration stage {stage_name} failed to converge within "
                f"{optimization['max_steps']} steps; best_loss={best_loss:.6f}"
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
            "stopped_early": bool(converged),
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
        if (config_stage_name or stage_name) == "reliability":
            calibrator = getattr(discount_fusion, "reliability_calibrator", None)
            reference_summary = getattr(
                calibrator, "embedding_reference_summary", None
            )
            if callable(reference_summary):
                references = reference_summary()
                if references:
                    result["embedding_references"] = references
        if (config_stage_name or stage_name) in {
            "routing_distribution",
            "routing_risk",
        }:
            router = getattr(discount_fusion, "opinion_router", None)
            diagnostic_builder = getattr(
                router, "effective_parameter_diagnostics", None
            )
            if callable(diagnostic_builder):
                diagnostics = diagnostic_builder()
                result["effective_parameter_diagnostics"] = diagnostics
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
    reliability_competence_parameters = (
        list(discount_fusion.reliability_competence_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "reliability_competence_parameters")
        else []
    )
    reliability_degradation_parameters = (
        list(discount_fusion.reliability_degradation_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "reliability_degradation_parameters")
        else []
    )
    probability_parameters = (
        list(discount_fusion.probability_calibration_parameters())
        if discount_fusion is not None
        and hasattr(discount_fusion, "probability_calibration_parameters")
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
            "risk_target", "mixture_argmax_error"
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
    if reliability_method == MONOTONIC_CORRECTNESS_METHOD:
        split_parameter_ids = {
            id(parameter)
            for parameter in [
                *reliability_competence_parameters,
                *reliability_degradation_parameters,
            ]
        }
        all_parameter_ids = {id(parameter) for parameter in reliability_parameters}
        if split_parameter_ids != all_parameter_ids:
            raise RuntimeError(
                "Monotonic I1 parameter partition must cover every calibrator "
                "parameter exactly once"
            )
        if (
            len(split_parameter_ids)
            != len(reliability_competence_parameters)
            + len(reliability_degradation_parameters)
        ):
            raise RuntimeError(
                "Monotonic I1 competence/degradation parameter sets overlap"
            )
    elif reliability_competence_parameters or reliability_degradation_parameters:
        raise RuntimeError(
            "Only monotonic_correctness may expose the two-stage I1 parameter split"
        )

    raw_i1_objective_weights = reliability_cfg.get(
        "objective_weights",
        {"clean": 0.50, "completeness": 0.25, "semantic": 0.25},
    )
    if not isinstance(raw_i1_objective_weights, dict):
        raise ValueError(
            "fusion.reliability_calibration.objective_weights must be a mapping"
        )
    i1_objective_weights = {
        name: float(raw_i1_objective_weights.get(name, 0.0))
        for name in ("clean", "completeness", "semantic")
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in i1_objective_weights.values()
    ) or not math.isclose(
        sum(i1_objective_weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            "fusion.reliability_calibration.objective_weights must contain "
            "non-negative clean/completeness/semantic masses summing to one"
        )
    i1_require_all_objective_families = bool(
        reliability_cfg.get("require_all_objective_families", False)
    )
    temperature_fit_source = str(
        reliability_cfg.get("temperature_fit_source", "clean_only")
    ).strip().lower()
    if temperature_fit_source not in {"clean_only", "matched_observable"}:
        raise ValueError(
            "fusion.reliability_calibration.temperature_fit_source must be "
            "'clean_only' or 'matched_observable'"
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
    routing_cfg.setdefault("reliability_calibration", {})["weight"] = 0.0
    routing_cfg.setdefault("probability_calibration", {})["weight"] = 0.0
    routing_cfg.setdefault("routing", {})["prediction_loss_weight"] = 1.0
    routing_cfg.setdefault("routing", {})["risk_loss_weight"] = 0.0
    risk_cfg = copy.deepcopy(cfg.get("fusion", {}) or {})
    risk_cfg.setdefault("reliability_calibration", {})["weight"] = 0.0
    risk_cfg.setdefault("probability_calibration", {})["weight"] = 0.0
    risk_cfg.setdefault("routing", {})["prediction_loss_weight"] = 0.0
    risk_cfg.setdefault("routing", {})["route_oracle_loss_weight"] = 0.0
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
            use_observable_scenarios = (
                reliability_method != TEMPERATURE_SCALING_CONFIDENCE_METHOD
                or temperature_fit_source == "matched_observable"
            )
            observable_scenarios = (
                [
                    item
                    for scenario_items in nonclean.values()
                    for item in scenario_items
                    if branch in item.get("reliability_branches", ())
                ]
                if use_observable_scenarios
                else []
            )
            if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD:
                source_label = (
                    "clean_plus_matched_observable"
                    if observable_scenarios
                    else "clean_only"
                )
            else:
                source_label = "observable" if observable_scenarios else "clean"
            groups.append(
                {
                    "name": f"{branch}:{source_label}",
                    "branch": branch,
                    "clean": clean,
                    "scenario": observable_scenarios,
                }
            )
        return groups

    def _build_routing_groups(
        items: list[dict[str, Any]],
        *,
        prefix: str,
        use_group_robust_taxonomy: bool = False,
    ) -> list[dict[str, Any]]:
        clean = _clean_items(items)
        if not clean:
            raise RuntimeError(f"{prefix} fitting selection has no clean samples")
        nonclean = _nonclean_by_group(items)
        if prefix == "risk":
            # Pairwise completeness views were added specifically to close a
            # route-training coverage gap.  Keep the already successful FN-risk
            # head's source protocol unchanged until a dedicated risk ablation
            # justifies expanding it.
            nonclean = {
                group_name: [
                    item
                    for item in scenario_items
                    if str(item.get("perturb_type") or "").lower()
                    not in ROUTING_PAIRWISE_COMPLETENESS
                ]
                for group_name, scenario_items in nonclean.items()
            }
            nonclean = {
                group_name: scenario_items
                for group_name, scenario_items in nonclean.items()
                if scenario_items
            }
        if use_group_robust_taxonomy:
            taxonomy = str(routing_group_robust["taxonomy"])
            if taxonomy in {"perturb_type_v1", "perturb_type_strength_v1"}:
                # Group by the declared perturbation itself, not by the legacy
                # ``scenario_group`` reduction (which intentionally collapses
                # api/graph/manifest missing into one "missing" bucket).
                # The taxonomy name and the objective mass therefore agree.
                grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for scenario_items in nonclean.values():
                    for item in scenario_items:
                        perturb_type = str(
                            item.get("perturb_type") or ""
                        ).strip().lower()
                        if not perturb_type or perturb_type in {"clean", "other"}:
                            raise RuntimeError(
                                "Non-clean routing source is missing perturb_type metadata"
                            )
                        group_key = perturb_type
                        mechanism_key = perturb_type
                        if taxonomy == "perturb_type_strength_v1":
                            strength = float(item.get("strength", 0.0))
                            if not math.isfinite(strength) or strength < 0.0:
                                raise RuntimeError(
                                    "Non-clean routing source has invalid strength metadata"
                                )
                            # Missing-modality views have no continuous severity.
                            # Detect them from their declared mechanism rather
                            # than from the stored strength (currently 1.0).
                            strength_label = (
                                "missing"
                                if perturb_type.endswith("_missing")
                                else f"s{strength:g}"
                            )
                            group_key = f"{perturb_type}/{strength_label}"
                            mechanism_key = group_key
                        grouped.setdefault(group_key, {}).setdefault(
                            mechanism_key, []
                        ).append(item)
            else:
                grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for scenario_items in nonclean.values():
                    for item in scenario_items:
                        family = str(item.get("objective_family") or "").strip().lower()
                        perturb_type = str(item.get("perturb_type") or "").strip().lower()
                        if family not in ROUTING_ROBUSTNESS_FAMILIES:
                            raise RuntimeError(
                                f"Routing source {item.get('scenario_name')!r} has invalid "
                                f"objective_family={family!r}"
                            )
                        if not perturb_type or perturb_type in {"clean", "other"}:
                            raise RuntimeError(
                                "Non-clean routing source is missing perturb_type metadata"
                            )
                        grouped.setdefault(family, {}).setdefault(
                            perturb_type, []
                        ).append(item)
            groups = []
            for family, mechanisms in sorted(grouped.items()):
                scenario_items = [
                    item
                    for mechanism_items in mechanisms.values()
                    for item in mechanism_items
                ]
                groups.append(
                    {
                        "name": f"{prefix}:{family}",
                        "objective_family": family,
                        "prior_group": (
                            family.split("/", 1)[0]
                            if taxonomy == "perturb_type_strength_v1"
                            else family
                        ),
                        "clean": clean,
                        "scenario": scenario_items,
                        "mechanisms": [
                            {
                                "name": mechanism,
                                "items": list(mechanism_items),
                            }
                            for mechanism, mechanism_items in sorted(
                                mechanisms.items()
                            )
                        ],
                    }
                )
            if groups:
                return groups
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

    def _reliability_objective(
        groups_or_group: list[dict[str, Any]] | dict[str, Any],
        context: dict[str, Any],
        *,
        component: str = "full",
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
                return branch_nll(
                    group["branch"],
                    cached["branch_logits"][group["branch"]],
                    cached["labels"],
                    cached["evidence"],
                )

            return _balanced_group_loss(
                group,
                _temperature_nll,
                context,
                requires_forward=False,
            )

        if isinstance(groups_or_group, dict):
            raise RuntimeError(
                "monotonic I1 requires the global packed objective"
            )
        if component not in {"competence", "degradation"}:
            raise RuntimeError(
                "monotonic I1 objective component must be 'competence' or "
                "'degradation'"
            )
        objective_groups = list(groups_or_group)
        if not objective_groups:
            raise RuntimeError("monotonic I1 objective contains no branch groups")

        persistent = context["persistent"]
        static_cache_key = f"i1_global_static/{component}"
        static = persistent.get(static_cache_key)
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
        if static is None:
            selection = _pack_cached_selection(
                context["all_items"], reliability_key=None, route_key=None
            )
            with torch.no_grad():
                packed_outputs = _forward_cached(selection["packed"])
            context["forward_count"] = int(context["forward_count"]) + 1
            required_keys = tuple(
                key
                for name in ("api", "graph", "manifest")
                for key in (
                    f"reliability_features_superset_{name}",
                    f"alive_{name}",
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
            global_segment_by_id = {
                id(item): (int(start), int(end))
                for item, (start, end) in zip(
                    context["all_items"], selection["segments"]
                )
            }
            group_count = len(objective_groups)
            branch_static: dict[str, dict[str, torch.Tensor]] = {}
            for group in objective_groups:
                branch = str(group["branch"])
                if branch in branch_static:
                    raise RuntimeError(
                        f"monotonic I1 contains duplicate branch group {branch!r}"
                    )
                raw_items = [
                    *(group.get("clean") or []),
                    *(group.get("scenario") or []),
                ]
                raw_ids = [id(item) for item in raw_items]
                if len(set(raw_ids)) != len(raw_ids):
                    raise RuntimeError(
                        f"monotonic I1 branch {branch!r} contains duplicate sources"
                    )
                selected_ids = set(raw_ids)
                branch_items = [
                    item
                    for item in context["all_items"]
                    if id(item) in selected_ids
                ]
                if len(branch_items) != len(raw_items):
                    raise RuntimeError(
                        f"monotonic I1 branch {branch!r} references an unknown source"
                    )

                features_parts: list[torch.Tensor] = []
                alive_parts: list[torch.Tensor] = []
                labels_parts: list[torch.Tensor] = []
                logits_parts: list[torch.Tensor] = []
                evidence_parts: list[torch.Tensor] = []
                branch_segments: list[tuple[int, int]] = []
                offset = 0
                for item in branch_items:
                    start, end = global_segment_by_id[id(item)]
                    rows = end - start
                    features_parts.append(
                        packed_outputs[
                            f"reliability_features_superset_{branch}"
                        ][start:end].detach()
                    )
                    alive_parts.append(
                        packed_outputs[f"alive_{branch}"][start:end].detach()
                    )
                    labels_parts.append(item["labels"])
                    logits_parts.append(item["branch_logits"][branch])
                    evidence_parts.append(item["evidence"])
                    branch_segments.append((offset, offset + rows))
                    offset += rows
                features = torch.cat(features_parts, dim=0)
                prediction_alive = torch.cat(alive_parts, dim=0).view(-1)
                labels = torch.cat(labels_parts, dim=0)
                branch_logits = torch.cat(logits_parts, dim=0)
                evidence = torch.cat(evidence_parts, dim=0)
                correctness = reliability_correctness_target(
                    branch_logits, labels
                )
                valid = reliability_alive_mask(evidence, branch)
                if component == "competence":
                    if group.get("scenario"):
                        raise RuntimeError(
                            "I1 clean-competence phase must not consume "
                            "degradation rows"
                        )
                    source_masses = {
                        id(item): 1.0 / float(len(branch_items))
                        for item in branch_items
                    }
                else:
                    categorized: dict[str, dict[str, list[dict[str, Any]]]] = {
                        "clean": {},
                        "completeness": {},
                        "semantic": {},
                    }
                    for item in branch_items:
                        perturb_type = str(
                            item.get("perturb_type") or "clean"
                        ).strip().lower()
                        if perturb_type == "clean":
                            category = "clean"
                        elif perturb_type.endswith("_semantic_corrupted"):
                            category = "semantic"
                        else:
                            category = "completeness"
                        categorized[category].setdefault(
                            perturb_type, []
                        ).append(item)
                    missing_families = [
                        name
                        for name, mass in i1_objective_weights.items()
                        if mass > 0.0 and not categorized[name]
                    ]
                    if missing_families and i1_require_all_objective_families:
                        raise RuntimeError(
                            "I1 degradation objective is missing configured "
                            f"families for branch {branch!r}: {missing_families}"
                        )
                    active_mass = sum(
                        i1_objective_weights[name]
                        for name in categorized
                        if categorized[name]
                    )
                    if active_mass <= 0.0:
                        raise RuntimeError(
                            f"I1 branch {branch!r} has no positive objective mass"
                        )
                    source_masses = {id(item): 0.0 for item in branch_items}
                    for category, mechanisms in categorized.items():
                        if not mechanisms:
                            continue
                        category_mass = (
                            i1_objective_weights[category] / active_mass
                        )
                        mechanism_mass = category_mass / float(len(mechanisms))
                        for mechanism_items in mechanisms.values():
                            per_source = mechanism_mass / float(
                                len(mechanism_items)
                            )
                            for item in mechanism_items:
                                source_masses[id(item)] += per_source
                # The previous objective first reduced every branch group and
                # then averaged the three groups.  Folding that final mean into
                # the immutable row weights preserves both branch equality and
                # every source's own alive denominator.
                row_weights = _compile_posthoc_row_weights(
                    branch_items,
                    branch_segments,
                    source_masses,
                    valid,
                ) / float(group_count)
                branch_static[branch] = {
                    "features": features,
                    "prediction_alive": prediction_alive,
                    "correctness": correctness,
                    "row_weights": row_weights,
                }
            static = {
                "signature": signature,
                "objective_signature": objective_signature,
                "branches": branch_static,
            }
            persistent[static_cache_key] = static
        elif static["signature"] != signature:
            raise RuntimeError(
                "I1 static feature cache was reused with a different source selection"
            )
        elif static["objective_signature"] != objective_signature:
            raise RuntimeError(
                "I1 static feature cache was reused with different objective groups"
            )

        loss_type = str(reliability_cfg.get("loss", "bce")).strip().lower()
        branch_losses: list[torch.Tensor] = []
        for group in objective_groups:
            branch = str(group["branch"])
            if branch not in branch_modules or branch not in static["branches"]:
                raise RuntimeError(
                    f"Monotonic I1 fitting is missing branch {branch!r}"
                )
            branch_data = static["branches"][branch]
            component_outputs = branch_modules[branch].forward_components(
                branch_data["features"]
            )
            reliability_logit = component_outputs[
                "clean_competence_logit"
                if component == "competence"
                else "reliability_logit"
            ]
            reliability = torch.sigmoid(reliability_logit)
            if bool(getattr(calibrator, "apply_alive_mask", True)):
                reliability = reliability * branch_data["prediction_alive"]
            per_row = reliability_per_sample_loss(
                reliability,
                reliability_logit,
                branch_data["correctness"],
                loss_type=loss_type,
            )
            branch_losses.append(torch.dot(per_row, branch_data["row_weights"]))
        context["lightweight_forward_count"] = int(
            context["lightweight_forward_count"]
        ) + len(branch_losses)
        return torch.stack(branch_losses).sum()

    def _fit_i1_embedding_reference(
        objective_groups: list[dict[str, Any]],
    ) -> None:
        """Fit one fold-local clean density reference before both I1 phases."""

        calibrator = getattr(discount_fusion, "reliability_calibrator", None)
        if not bool(getattr(calibrator, "use_embedding_density", False)):
            return
        clean_reference_items: list[dict[str, Any]] = []
        seen_clean_ids: set[int] = set()
        for group in objective_groups:
            for item in group.get("clean") or []:
                if id(item) not in seen_clean_ids:
                    clean_reference_items.append(item)
                    seen_clean_ids.add(id(item))
        if not clean_reference_items:
            raise RuntimeError(
                "I1 embedding density requires clean fold-local reference rows"
            )
        clean_reference = _pack_cached_selection(
            clean_reference_items,
            reliability_key=None,
            route_key=None,
        )["packed"]
        branch_embeddings = clean_reference.get("branch_embeddings")
        if not isinstance(branch_embeddings, dict):
            raise RuntimeError(
                "I1 embedding density clean cache has no branch embeddings"
            )
        fit_reference = getattr(calibrator, "fit_embedding_references", None)
        if not callable(fit_reference):
            raise RuntimeError(
                "Configured I1 density calibrator cannot fit embedding references"
            )
        fit_reference(
            branch_embeddings,
            clean_reference["labels"],
            clean_reference["evidence"],
        )

    def _reliability_competence_objective(
        groups: list[dict[str, Any]], context: dict[str, Any]
    ) -> torch.Tensor:
        return _reliability_objective(
            groups, context, component="competence"
        )

    def _reliability_degradation_objective(
        groups: list[dict[str, Any]], context: dict[str, Any]
    ) -> torch.Tensor:
        return _reliability_objective(
            groups, context, component="degradation"
        )

    def _fit_reliability_stage(
        stage_name: str,
        objective_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Fit I1 with an explicit clean-then-degradation lifecycle.

        Temperature scaling remains a one-stage comparator.  The proposed I1
        first estimates clean branch competence from margin/class only, freezes
        it, and then learns a bias-free non-negative degradation penalty from
        clean plus branch-local transformed views.  This makes the inequality
        ``reliability <= clean_competence`` structural rather than empirical.
        """

        if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD:
            return _optimize_stage(
                stage_name,
                reliability_parameters,
                objective_groups,
                _reliability_objective,
                config_stage_name="reliability",
                global_objective=False,
            )
        if reliability_method != MONOTONIC_CORRECTNESS_METHOD:
            raise RuntimeError(
                f"Unsupported two-stage I1 method {reliability_method!r}"
            )
        if not reliability_competence_parameters:
            raise RuntimeError("Monotonic I1 has no clean-competence parameters")
        if not reliability_degradation_parameters:
            raise RuntimeError("Monotonic I1 has no degradation parameters")

        _fit_i1_embedding_reference(objective_groups)
        clean_groups = [
            {
                **group,
                "name": f"{group['branch']}:clean_competence",
                "scenario": [],
            }
            for group in objective_groups
        ]
        competence_summary = _optimize_stage(
            f"{stage_name}/competence",
            reliability_competence_parameters,
            clean_groups,
            _reliability_competence_objective,
            config_stage_name="reliability_competence",
            global_objective=True,
        )
        degradation_summary = _optimize_stage(
            f"{stage_name}/degradation",
            reliability_degradation_parameters,
            objective_groups,
            _reliability_degradation_objective,
            config_stage_name="reliability_degradation",
            global_objective=True,
        )

        total_steps = int(
            competence_summary["total_steps"]
            + degradation_summary["total_steps"]
        )
        initial_loss = float(
            competence_summary["initial_loss"]
            + degradation_summary["initial_loss"]
        )
        final_loss = float(
            competence_summary["final_loss"]
            + degradation_summary["final_loss"]
        )
        reference_summary = getattr(
            getattr(discount_fusion, "reliability_calibrator", None),
            "embedding_reference_summary",
            None,
        )
        result = {
            "enabled": True,
            "name": stage_name,
            "lifecycle": "clean_competence_then_nonnegative_degradation_v1",
            "degradation_never_increases_reliability": True,
            "epochs_ran": total_steps,
            "best_epoch": total_steps,
            "stopped_early": bool(
                competence_summary["stopped_early"]
                or degradation_summary["stopped_early"]
            ),
            "converged": bool(
                competence_summary["converged"]
                and degradation_summary["converged"]
            ),
            "stop_reason": (
                "both_phases_converged"
                if competence_summary["converged"]
                and degradation_summary["converged"]
                else "competence_converged_degradation_budget_ended"
                if competence_summary["converged"]
                else "competence_budget_ended_degradation_converged"
                if degradation_summary["converged"]
                else "both_phase_budgets_ended"
            ),
            "parameter_selection": "ordered_phase_minima",
            "losses": [
                *competence_summary["losses"],
                *degradation_summary["losses"],
            ],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "restored_best_loss": final_loss,
            "total_steps": total_steps,
            "function_evaluations": int(
                competence_summary["function_evaluations"]
                + degradation_summary["function_evaluations"]
            ),
            "decision_forward_evaluations": int(
                competence_summary["decision_forward_evaluations"]
                + degradation_summary["decision_forward_evaluations"]
            ),
            "lightweight_forward_evaluations": int(
                competence_summary["lightweight_forward_evaluations"]
                + degradation_summary["lightweight_forward_evaluations"]
            ),
            "parameter_delta_l2": math.sqrt(
                float(competence_summary["parameter_delta_l2"]) ** 2
                + float(degradation_summary["parameter_delta_l2"]) ** 2
            ),
            "relative_loss_improvement": (
                (initial_loss - final_loss) / max(1.0e-12, abs(initial_loss))
            ),
            "objective_groups": [
                str(group["name"]) for group in objective_groups
            ],
            "objective_weights": dict(i1_objective_weights),
            "phases": {
                "competence": competence_summary,
                "degradation": degradation_summary,
            },
            "optimization": {
                "competence": competence_summary["optimization"],
                "degradation": degradation_summary["optimization"],
            },
        }
        if callable(reference_summary):
            references = reference_summary()
            if references:
                result["embedding_references"] = references
        return result

    def _probability_objective(
        group: dict[str, Any], context: dict[str, Any]
    ) -> torch.Tensor:
        def _loss(
            cached: dict[str, Any], outputs: dict[str, Any] | None
        ) -> torch.Tensor:
            if outputs is None:
                raise RuntimeError("Probability packed forward produced no outputs")
            loss, _ = compute_probability_calibration_loss(
                outputs,
                cached["labels"],
                cached["evidence"],
            )
            return loss

        return _balanced_group_loss(group, _loss, context)

    def _routing_objective(reliability_key: str | None):
        # Encoder opinions, I1 reliability and availability do not change
        # while pi is fitted.  Materialize those exact router inputs once for
        # the complete stage. Source/family reductions are compiled into row
        # weights, so every optimizer evaluation executes one router kernel and
        # two packed reductions rather than one loss call per scenario source.
        route_options = routing_cfg.get("routing", {}) or {}
        route_enabled = bool(route_options.get("enabled", False)) and bool(
            route_options.get("posthoc_refine", True)
        )
        route_calibration_weight = (
            float(route_options.get("calibration_weight", 1.0))
            if route_enabled
            else 0.0
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
        route_oracle_weight = float(
            route_options.get("route_oracle_loss_weight", 0.0)
        )
        route_oracle_temperature = float(
            route_options.get("route_oracle_temperature", 1.0)
        )
        subset_oracle_weight = float(
            route_options.get("subset_oracle_loss_weight", 0.0)
        )
        subset_oracle_temperature = float(
            route_options.get("subset_oracle_temperature", 1.0)
        )
        for name, value in (
            ("calibration_weight", route_calibration_weight),
            ("prediction_loss_weight", route_prediction_weight),
            ("route_oracle_loss_weight", route_oracle_weight),
            ("subset_oracle_loss_weight", subset_oracle_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"fusion.routing.{name} must be finite and non-negative")
        if (
            not math.isfinite(route_oracle_temperature)
            or route_oracle_temperature <= 0.0
        ):
            raise ValueError(
                "fusion.routing.route_oracle_temperature must be finite and positive"
            )
        if (
            not math.isfinite(subset_oracle_temperature)
            or subset_oracle_temperature <= 0.0
        ):
            raise ValueError(
                "fusion.routing.subset_oracle_temperature must be finite and positive"
            )
        if subset_oracle_weight == 0.0 and subset_oracle_temperature != 1.0:
            raise ValueError(
                "disabled source-subset oracle requires temperature=1.0"
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
                    f"routing_input_belief_{name}",
                    f"routing_input_uncertainty_{name}",
                    f"routing_input_reliability_{name}",
                    f"routing_input_alive_{name}",
                    f"calibrated_log_prob_{name}",
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
                    beliefs={
                        name: packed_outputs[
                            f"routing_input_belief_{name}"
                        ].detach()
                        for name in branches
                    },
                    uncertainties={
                        name: packed_outputs[
                            f"routing_input_uncertainty_{name}"
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
            for name in branches:
                static_inputs[f"calibrated_log_prob_{name}"] = packed_outputs[
                    f"calibrated_log_prob_{name}"
                ].detach()
            static_inputs["learned_active"] = bool(
                getattr(discount_fusion, "calibration_active", False)
            )
            labels = selection["packed"]["labels"].detach().long().view(-1)
            evidence = selection["packed"]["evidence"].detach()
            static_inputs["source_names"] = [
                str(item.get("scenario_name") or item.get("name") or "source")
                for item in all_items
            ]
            prepared_has_available = prepared.get("has_available")
            if not isinstance(prepared_has_available, torch.Tensor):
                raise RuntimeError("Prepared routing inputs are missing availability")
            static_inputs["labels"] = labels

            source_masses = None
            if not bool(routing_group_robust["enabled"]):
                source_masses = _compile_posthoc_source_masses(
                    objective_groups,
                    all_items,
                    routing_scenario_weights,
                )

            def _store_component_weights(
                prefix: str, valid_mask: torch.Tensor
            ) -> None:
                if bool(routing_group_robust["enabled"]):
                    (
                        clean_weights,
                        family_weights,
                        family_priors,
                        resolved,
                    ) = (
                        _compile_group_robust_row_weights(
                            objective_groups,
                            all_items,
                            selection["segments"],
                            valid_mask,
                        )
                    )
                    static_inputs[f"{prefix}_clean_row_weights"] = clean_weights
                    static_inputs[f"{prefix}_family_row_weights"] = family_weights
                    static_inputs[f"{prefix}_family_priors"] = family_priors
                    static_inputs["resolved_group_robust_families"] = resolved
                else:
                    assert source_masses is not None
                    static_inputs[f"{prefix}_row_weights"] = (
                        _compile_posthoc_row_weights(
                            all_items,
                            selection["segments"],
                            source_masses,
                            valid_mask,
                        )
                    )

            _store_component_weights("prediction", prepared_has_available)
            if route_oracle_weight > 0.0:
                oracle_target, oracle_valid = routing_soft_oracle_target(
                    packed_outputs,
                    labels,
                    evidence,
                    temperature=route_oracle_temperature,
                )
                static_inputs["oracle_target"] = oracle_target.detach()
                _store_component_weights("oracle", oracle_valid)
            if subset_oracle_weight > 0.0:
                subset_target, subset_valid, subset_diagnostics = (
                    routing_source_subset_oracle_target(
                        packed_outputs,
                        labels,
                        evidence,
                        source_segments=selection["segments"],
                        source_names=static_inputs["source_names"],
                        temperature=subset_oracle_temperature,
                    )
                )
                static_inputs["subset_oracle_target"] = subset_target.detach()
                static_inputs["subset_oracle_diagnostics"] = {
                    key: value.detach()
                    for key, value in subset_diagnostics.items()
                }
                _store_component_weights("subset_oracle", subset_valid)
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
                **{
                    f"calibrated_log_prob_{name}": static_inputs[
                        f"calibrated_log_prob_{name}"
                    ]
                    for name in branches
                },
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
            component_rows: list[tuple[str, float, torch.Tensor]] = [
                ("prediction", route_prediction_weight, prediction_per_row)
            ]
            if route_oracle_weight > 0.0:
                oracle_per_row = routing_soft_oracle_per_sample_loss(
                    packed_outputs,
                    static_inputs["oracle_target"],
                )
                component_rows.append(
                    ("oracle", route_oracle_weight, oracle_per_row)
                )
            if subset_oracle_weight > 0.0:
                subset_oracle_per_row = routing_subset_oracle_per_sample_loss(
                    packed_outputs,
                    static_inputs["subset_oracle_target"],
                )
                component_rows.append(
                    (
                        "subset_oracle",
                        subset_oracle_weight,
                        subset_oracle_per_row,
                    )
                )
            component_weight_sum = sum(
                component_weight
                for _prefix, component_weight, _per_row in component_rows
                if component_weight > 0.0
            )
            if component_weight_sum <= 0.0:
                raise RuntimeError(
                    "Routing-distribution fit has no positive objective component"
                )
            # Normalize active route components so adding/removing an auxiliary
            # term changes only their relative trade-off.  Otherwise gradient
            # clipping, scheduler thresholds, and convergence tolerances would
            # give the subset-oracle ablation a different optimization budget.
            component_rows = [
                (prefix, component_weight / component_weight_sum, per_row)
                for prefix, component_weight, per_row in component_rows
                if component_weight > 0.0
            ]

            if bool(routing_group_robust["enabled"]):
                clean_objective = prediction_per_row.new_zeros(())
                family_objective = None
                for prefix, component_weight, per_row in component_rows:
                    if component_weight <= 0.0:
                        continue
                    clean_weights = static_inputs[
                        f"{prefix}_clean_row_weights"
                    ].to(dtype=per_row.dtype)
                    family_weights = static_inputs[
                        f"{prefix}_family_row_weights"
                    ].to(dtype=per_row.dtype)
                    clean_objective = clean_objective + component_weight * torch.dot(
                        per_row, clean_weights
                    )
                    family_values = torch.mv(family_weights, per_row)
                    weighted_family_values = component_weight * family_values
                    family_objective = (
                        weighted_family_values
                        if family_objective is None
                        else family_objective + weighted_family_values
                    )
                if family_objective is None:
                    raise RuntimeError("Group-robust route has no active objective component")
                perturb_objective, effective_family_weights = (
                    _entropic_soft_worst_group(
                        family_objective,
                        soft_worst_weight=float(
                            routing_group_robust["soft_worst_weight"]
                        ),
                        temperature=float(routing_group_robust["temperature"]),
                        group_priors=static_inputs[
                            "prediction_family_priors"
                        ],
                    )
                )
                static_inputs["last_family_losses"] = family_objective.detach()
                static_inputs["last_effective_family_weights"] = (
                    effective_family_weights.detach()
                )
                objective = (
                    float(routing_scenario_weights["clean"]) * clean_objective
                    + float(routing_scenario_weights["perturb"])
                    * perturb_objective
                )
            else:
                objective = prediction_per_row.new_zeros(())
                for prefix, component_weight, per_row in component_rows:
                    if component_weight <= 0.0:
                        continue
                    objective = objective + component_weight * torch.dot(
                        per_row,
                        static_inputs[f"{prefix}_row_weights"].to(
                            dtype=per_row.dtype
                        ),
                    )
            router = getattr(discount_fusion, "opinion_router", None)
            regularizer = getattr(router, "route_effective_l2", None)
            if route_effective_l2 > 0.0:
                if not callable(regularizer):
                    raise RuntimeError(
                        "Configured route effective L2 requires router support"
                    )
                objective = objective + route_effective_l2 * regularizer()
            return route_calibration_weight * objective

        def _summary_metadata() -> dict[str, Any]:
            subset_enabled = bool(subset_oracle_weight > 0.0)
            metadata: dict[str, Any] = {
                "group_robust_objective": {
                    **dict(routing_group_robust),
                    "hierarchical_group_balancing_enabled": bool(
                        routing_group_robust["enabled"]
                    ),
                    "soft_worst_enabled": bool(
                        routing_group_robust["enabled"]
                        and float(routing_group_robust["soft_worst_weight"]) > 0.0
                    ),
                },
                "resolved_families": list(
                    static_inputs.get("resolved_group_robust_families") or []
                ),
                "subset_oracle": {
                    "enabled": subset_enabled,
                    "semantics": (
                        "source_soft_subset_probability"
                        if subset_enabled
                        else "disabled"
                    ),
                    "loss_weight": subset_oracle_weight,
                    "temperature": (
                        subset_oracle_temperature if subset_enabled else None
                    ),
                    "candidate_subsets": (
                        [
                            "+".join(subset)
                            for subset in ROUTING_PROBABILITY_SUBSETS
                        ]
                        if subset_enabled
                        else []
                    ),
                },
                "normalized_route_component_weights": {
                    "prediction": route_prediction_weight
                    / max(
                        route_prediction_weight
                        + route_oracle_weight
                        + subset_oracle_weight,
                        1.0e-12,
                    ),
                    "row_oracle": route_oracle_weight
                    / max(
                        route_prediction_weight
                        + route_oracle_weight
                        + subset_oracle_weight,
                        1.0e-12,
                    ),
                    "source_subset_oracle": subset_oracle_weight
                    / max(
                        route_prediction_weight
                        + route_oracle_weight
                        + subset_oracle_weight,
                        1.0e-12,
                    ),
                },
                "route_effective_l2": route_effective_l2,
            }
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
            diagnostics = static_inputs.get("subset_oracle_diagnostics")
            if not isinstance(diagnostics, dict):
                return metadata
            hard_best = diagnostics["hard_best_subset_index"].detach().cpu().tolist()
            candidate_nll = diagnostics["candidate_nll"].detach().cpu().tolist()
            candidate_mass = diagnostics["candidate_mass"].detach().cpu().tolist()
            target_mass = diagnostics["target_branch_mass"].detach().cpu().tolist()
            valid_counts = diagnostics["source_valid_count"].detach().cpu().tolist()
            eligible_counts = diagnostics[
                "eligible_candidate_count"
            ].detach().cpu().tolist()
            gaps = diagnostics["best_second_gap"].detach().cpu().tolist()
            source_names = list(static_inputs.get("source_names") or [])
            subset_names = ["+".join(value) for value in ROUTING_PROBABILITY_SUBSETS]
            best_counts = {name: 0 for name in subset_names}
            source_rows = []
            for index, best_index in enumerate(hard_best):
                best_name = subset_names[best_index] if int(best_index) >= 0 else None
                if best_name is not None:
                    best_counts[best_name] += 1
                source_rows.append(
                    {
                        "name": source_names[index],
                        "num_valid": int(valid_counts[index]),
                        "num_eligible_subsets": int(eligible_counts[index]),
                        "hard_best_subset": best_name,
                        "best_second_nll_gap": (
                            float(gaps[index])
                            if math.isfinite(float(gaps[index]))
                            else None
                        ),
                        "candidate_nll": [
                            float(value) if math.isfinite(float(value)) else None
                            for value in candidate_nll[index]
                        ],
                        "candidate_mass": [
                            float(value) for value in candidate_mass[index]
                        ],
                        "target_branch_mass": [
                            float(value) for value in target_mass[index]
                        ],
                    }
                )
            metadata["subset_oracle"].update(
                {
                    "hard_best_subset_counts": best_counts,
                    "sources": source_rows,
                }
            )
            return metadata

        setattr(_objective, "summary_metadata", _summary_metadata)

        return _objective

    def _risk_objective(
        reliability_key: str | None,
        route_key: str | None,
    ):
        # With OOF reliability and route fixed, I2's five risk features are
        # immutable.  Materialize them once for the complete stage and optimize
        # only the monotone logistic head thereafter.  The previous path rebuilt
        # opinions, routing and diagnostics on every one of the hundreds of risk
        # optimizer steps even though none of those tensors depended on the risk
        # parameters. Source-normalized masks are likewise compiled once, so
        # the learned head is evaluated exactly once over the packed stage.
        risk_options = risk_cfg.get("routing", {}) or {}
        risk_enabled = bool(risk_options.get("enabled", False)) and bool(
            risk_options.get("posthoc_refine", True)
        )
        risk_calibration_weight = (
            float(risk_options.get("calibration_weight", 1.0))
            if risk_enabled
            else 0.0
        )
        risk_objective_weight = float(risk_options.get("risk_loss_weight", 1.0))
        risk_effective_l2 = float(risk_options.get("risk_effective_l2", 0.0))
        for name, value in (
            ("calibration_weight", risk_calibration_weight),
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
                "routing_risk_uncertainty_burden",
                "routing_risk_decision_boundary_proximity",
                "routing_risk_structural_conflict",
                "routing_risk_missing_fraction",
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
            risk_bias = getattr(router, "risk_bias", None)
            if not isinstance(raw_weights, torch.Tensor) or not isinstance(
                risk_bias, torch.Tensor
            ):
                raise RuntimeError(
                    "Learned I2 risk fitting requires the monotone risk head"
                )
            features = static_inputs["features"]
            risk_weights = F.softplus(raw_weights).to(
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
            objective = torch.dot(
                per_row,
                static_inputs["row_weights"].to(dtype=per_row.dtype),
            )
            router_regularizer = getattr(router, "risk_effective_l2", None)
            if risk_effective_l2 > 0.0:
                if not callable(router_regularizer):
                    raise RuntimeError(
                        "Configured risk effective L2 requires router support"
                    )
                objective = objective + risk_effective_l2 * router_regularizer()
            return risk_calibration_weight * risk_objective_weight * objective

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

    def _embedding_reference_snapshot() -> dict[str, torch.Tensor]:
        calibrator = getattr(discount_fusion, "reliability_calibrator", None)
        snapshot = getattr(calibrator, "embedding_reference_snapshot", None)
        return snapshot() if callable(snapshot) else {}

    def _restore_embedding_reference_snapshot(
        state: dict[str, torch.Tensor],
    ) -> None:
        calibrator = getattr(discount_fusion, "reliability_calibrator", None)
        restore = getattr(
            calibrator, "restore_embedding_reference_snapshot", None
        )
        if callable(restore):
            restore(state)
        elif state:
            raise RuntimeError(
                "Cached I1 embedding reference cannot be restored by this calibrator"
            )

    def _compact_stage_summary(summary: dict[str, Any]) -> dict[str, Any]:
        compact = {
            key: value
            for key, value in summary.items()
            # Fold-local route diagnostics include one row for every calibration
            # source and all seven subset candidates.  They are useful for the
            # final full-data fit, but repeating them for every inner/outer fold
            # makes the summary needlessly large and hard to inspect.
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
                if (
                    str(cached.get("perturb_type") or "").lower()
                    in ROUTING_PAIRWISE_COMPLETENESS
                ):
                    continue
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
                "risk_target", "mixture_argmax_error"
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
            if probability_parameters:
                raise ValueError(
                    "Nested cross-fitting does not support a separate branch "
                    "probability-calibration stage"
                )
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
                                "embedding_reference": (
                                    _embedding_reference_snapshot()
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
                            _restore_embedding_reference_snapshot(
                                cached_inner_fit.get(
                                    "embedding_reference", {}
                                )
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
                            use_group_robust_taxonomy=bool(
                                routing_group_robust["enabled"]
                            ),
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
                            use_group_robust_taxonomy=bool(
                                routing_group_robust["enabled"]
                            ),
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
                    "risk_target", "mixture_argmax_error"
                )
            ).strip().lower()
            if (
                routing_risk_parameters
                and configured_risk_target
                in {
                    "threshold_classification_error",
                    "threshold_malware_false_negative",
                }
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
                        use_group_robust_taxonomy=bool(
                            routing_group_robust["enabled"]
                        ),
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
            if probability_parameters:
                stage_summaries["probability"] = _optimize_stage(
                    "probability",
                    probability_parameters,
                    [
                        {
                            "name": "probability:clean",
                            "clean": full_clean_cached,
                            "scenario": [],
                        }
                    ],
                    _probability_objective,
                    config_stage_name="probability",
                )
            if routing_distribution_parameters:
                stage_summaries["routing_distribution"] = _optimize_stage(
                    "routing_distribution",
                    routing_distribution_parameters,
                    _build_routing_groups(
                        full_cached,
                        prefix="router",
                        use_group_robust_taxonomy=bool(
                            routing_group_robust["enabled"]
                        ),
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
    for name in ("api", "graph", "manifest"):
        if model.discount_fusion.temperature_parameters is not None:
            temperatures[name] = float(
                (torch.nn.functional.softplus(
                    model.discount_fusion.temperature_parameters[name].detach()
                ) + 1.0e-4).cpu().item()
            )
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
    embedding_reference_summary_fn = getattr(
        reliability_calibrator, "embedding_reference_summary", None
    )
    embedding_reference_summary = (
        embedding_reference_summary_fn()
        if callable(embedding_reference_summary_fn)
        else {}
    )
    full_clean_sample_count = int(
        sum(item["labels"].numel() for item in full_clean_cached)
    )

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
            else (
                (
                    "fold_local_clean_embedding_reference_plus_branch_local_scenarios"
                    if cross_fitting["enabled"]
                    else "full_posthoc_clean_embedding_reference_plus_branch_local_scenarios"
                )
                if bool(
                    getattr(
                        reliability_calibrator,
                        "use_embedding_density",
                        False,
                    )
                )
                else "clean_plus_branch_local_scenarios"
            )
        ),
        "reliability_temperatures": reliability_temperatures,
        "reliability_embedding_references": embedding_reference_summary,
        "routing_scenario_objective_weights": dict(
            routing_scenario_weights
        ),
        "routing_group_robust_objective": dict(routing_group_robust),
        "routing_robustness_family_mapping": dict(
            sorted(ROUTING_ROBUSTNESS_FAMILY_BY_PERTURBATION.items())
        ),
        "routing_risk_target": routing_risk_target_summary,
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
        "max_steps_default": legacy_default_steps,
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
                or key == "routing_prefit_uniform_prior_active"
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
        # Seed wrappers use distinct display names for the same method. Keep
        # method.protocol_id in the protocol hash: deleting the whole mapping
        # previously made two explicitly different protocols hash-identical.
        protocol_method = protocol_config.get("method")
        if isinstance(protocol_method, dict):
            protocol_method.pop("name", None)
            if not protocol_method:
                protocol_config.pop("method", None)
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
    combination_rule = str(fusion_cfg.get("combination", "linear")).lower()
    routing_enabled = bool(
        combination_rule == "routed" and routing_cfg.get("enabled", False)
    )
    routing_prediction_loss_weight = float(
        routing_cfg.get("prediction_loss_weight", 1.0)
    )
    routing_route_oracle_loss_weight = float(
        routing_cfg.get("route_oracle_loss_weight", 0.0)
    )
    routing_route_oracle_temperature = float(
        routing_cfg.get("route_oracle_temperature", 1.0)
    )
    routing_subset_oracle_loss_weight = float(
        routing_cfg.get("subset_oracle_loss_weight", 0.0)
    )
    routing_subset_oracle_temperature = float(
        routing_cfg.get("subset_oracle_temperature", 1.0)
    )
    routing_group_robust_objective = dict(
        routing_cfg.get("group_robust_objective", {}) or {}
    )
    include_pairwise_completeness_views = bool(
        calibration_cfg.get("include_pairwise_completeness_views", False)
    )
    routing_risk_loss_weight = float(routing_cfg.get("risk_loss_weight", 1.0))
    routing_risk_mode = str(routing_cfg.get("risk_mode", "learned")).strip().lower()
    routing_risk_enabled = routing_risk_mode != "disabled"
    routing_risk_loss = str(routing_cfg.get("risk_loss", "bce")).strip().lower()
    routing_mode = str(routing_cfg.get("mode", "learned")).strip().lower()
    routing_fixed_prior_beta = float(routing_cfg.get("fixed_prior_beta", 1.0))
    routing_risk_target = str(
        routing_cfg.get("risk_target", "mixture_argmax_error")
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
    routing_acceptance_mode = str(
        routing_cfg.get("acceptance_score_mode", "product")
    ).lower()
    use_reliability_discount = bool(
        fusion_cfg.get("use_reliability_discount", True)
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
            if routing_enabled and routing_mode == "prior_only"
            else (
                "learned_positive_odds_beta"
                if routing_enabled and routing_mode == "learned"
                else "disabled"
            )
        ),
        "routing_fixed_prior_beta": (
            routing_fixed_prior_beta
            if routing_enabled and routing_mode == "prior_only"
            else None
        ),
        "routing_prior_beta_trainable": bool(
            routing_enabled and routing_mode == "learned"
        ),
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
        "routing_route_oracle_loss_weight": (
            routing_route_oracle_loss_weight if routing_enabled else None
        ),
        "routing_route_oracle_temperature": (
            routing_route_oracle_temperature if routing_enabled else None
        ),
        "routing_subset_oracle_semantics": (
            "source_soft_subset_probability"
            if routing_enabled and routing_subset_oracle_loss_weight > 0.0
            else "disabled"
        ),
        "routing_subset_oracle_candidate_count": (
            7 if routing_enabled and routing_subset_oracle_loss_weight > 0.0 else 0
        ),
        "routing_subset_oracle_loss_weight": (
            routing_subset_oracle_loss_weight if routing_enabled else None
        ),
        "routing_subset_oracle_temperature": (
            routing_subset_oracle_temperature
            if routing_enabled and routing_subset_oracle_loss_weight > 0.0
            else None
        ),
        "routing_group_robust_objective": (
            {
                **routing_group_robust_objective,
                "hierarchical_group_balancing_enabled": bool(
                    routing_group_robust_objective.get("enabled", False)
                ),
                "soft_worst_enabled": bool(
                    routing_group_robust_objective.get("enabled", False)
                    and float(
                        routing_group_robust_objective.get(
                            "soft_worst_weight", 0.0
                        )
                    )
                    > 0.0
                ),
            }
            if routing_enabled
            else None
        ),
        "routing_pairwise_completeness_views_enabled": bool(
            routing_enabled and include_pairwise_completeness_views
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
        "routing_acceptance_score_mode": (
            routing_acceptance_mode if routing_enabled else "disabled"
        ),
        "routing_posthoc_distribution_loss_enabled": bool(
            routing_enabled
            and (
                routing_prediction_loss_weight > 0.0
                or routing_route_oracle_loss_weight > 0.0
                or routing_subset_oracle_loss_weight > 0.0
            )
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
                else "clean_competence_minus_nonnegative_degradation"
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
                "clean_competence_then_balanced_branch_degradation"
                if reliability_enabled
                else "disabled"
            )
        ),
        "reliability_lifecycle": (
            "clean_competence_then_nonnegative_degradation_v1"
            if reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            else "single_stage_baseline"
            if reliability_enabled
            else "disabled"
        ),
        "reliability_objective_weights": (
            dict(
                reliability_cfg.get(
                    "objective_weights",
                    {"clean": 0.50, "completeness": 0.25, "semantic": 0.25},
                )
            )
            if reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            else None
        ),
        "reliability_use_enabled": use_reliability_discount,
        "auxiliary_weight_mode": auxiliary_weight_mode,
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
            "alive_masked_uniform" if routing_enabled else "disabled"
        ),
        "router_posthoc_reliability_source": (
            (
                "temperature_scaled_max_softmax_confidence"
                if reliability_method == TEMPERATURE_SCALING_CONFIDENCE_METHOD
                else "calibrated_branch_correctness"
            )
            if routing_enabled
            and use_reliability_discount
            and reliability_enabled
            else (
                "observable_integrity"
                if routing_enabled and use_reliability_discount
                else ("unit_prior" if routing_enabled else "disabled")
            )
        ),
        "model_visibility_reliability_enabled": bool(
            reliability_enabled and reliability_cfg.get("use_model_visibility", False)
        ),
        "embedding_density_reliability_enabled": bool(
            reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            and reliability_cfg.get("use_embedding_density", False)
        ),
        "embedding_density_reference_semantics": (
            "fold_local_clean_class_conditional_diagonal_mahalanobis"
            if reliability_enabled
            and reliability_method == MONOTONIC_CORRECTNESS_METHOD
            and reliability_cfg.get("use_embedding_density", False)
            else "disabled"
        ),
        "prediction_margin_reliability_enabled": bool(
            reliability_enabled and reliability_cfg.get("use_prediction_margin", True)
        ),
        "predicted_class_reliability_enabled": bool(
            reliability_enabled
            and reliability_cfg.get("use_predicted_class_feature", True)
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


def prepare_output_directory(
    out_dir: str | Path,
    *,
    overwrite: bool = False,
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
    if calibration_enabled and not discount_probability_mode:
        raise ValueError(
            "Post-hoc I1/I2 calibration modules require discount_probability fusion"
        )
    _validate_selective_score_fusion_compatibility(
        selective_enabled=selective_enabled,
        score_type=selective_score_type,
        discount_probability_mode=discount_probability_mode,
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
    out_dir = prepare_output_directory(
        Path(data_cfg.get("out_dir", "experiments")) / exp_name / str(seed),
        overwrite=overwrite,
    )
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
        robust_val_loaders = build_robust_val_loaders(
            cfg, val_ds, selection_indices
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
    if not eval_only and train_ds is None:
        train_ds = build_dataset(cfg, "train", is_train=True)
        train_loader = build_loader(cfg, train_ds, is_train=True)
    test_ds: RobustTriModalDataset | None = None
    test_loader = None
    if run_test:
        test_ds = build_dataset(cfg, "test", is_train=False)
        test_loader = build_loader(cfg, test_ds, is_train=False)

    model = build_model(cfg, feature_dim).to(device)
    source_checkpoint_path: Path | None = None
    requested_checkpoint_path: Path | None = None
    posthoc_oof_clean_rows: list[dict[str, Any]] = []
    classification_log_odds_threshold: float | None = None

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
                parameters[0].fill_(math.log(final_temperature_override))
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
            calibration_summary = fit_posthoc_calibration(
                model,
                calibration_loaders,
                device,
                use_amp,
                cfg,
            )
            posthoc_oof_clean_rows = list(
                calibration_summary.pop("_oof_clean_rows", [])
            )
            posthoc_oof_clean_rows = validate_posthoc_oof_rows(
                posthoc_oof_clean_rows
            )
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
            classification_log_odds_threshold = None
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
            train_started_at = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
            train_wall_seconds = float(time.perf_counter() - train_started_at)
            val_started_at = time.perf_counter()
            val_metrics = evaluate_checkpoint_selection(
                model,
                val_loader,
                device,
                use_amp,
                "val_checkpoint_selection",
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
            val_wall_seconds = float(time.perf_counter() - val_started_at)
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
                "epoch=%s train_loss=%.4f val_macro_f1=%.4f val_auc=%.4f val_acc=%.4f checkpoint_score=%.4f train_wall_seconds=%.2f val_wall_seconds=%.2f",
                epoch,
                train_loss,
                val_metrics["f1"],
                val_metrics["auc"],
                val_metrics["acc"],
                score,
                train_wall_seconds,
                val_wall_seconds,
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
                        "manifest_vocab_provenance": manifest_vocab_provenance,
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
                "refusing to load an encoder-selected artifact from an earlier run"
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
        posthoc_oof_clean_rows = list(
            calibration_summary.pop("_oof_clean_rows", [])
        )
        posthoc_oof_clean_rows = validate_posthoc_oof_rows(
            posthoc_oof_clean_rows
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
        classification_log_odds_threshold = None

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
                shutil.copy2(source_encoder_path, portable_encoder_path)
            ckpt["encoder_checkpoint_path"] = portable_encoder_path.name
            ckpt["encoder_checkpoint_sha256"] = _file_sha256(
                portable_encoder_path
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
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
            classification_log_odds_threshold=(
                classification_log_odds_threshold
            ),
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
        "decision_calibration_signature": _decision_calibration_signature(cfg),
        "decision_calibration_data_identity": decision_calibration_data_identity,
        "acceptance_comparison": _decision_calibration_signature(cfg)[
            "selective"
        ]["acceptance_comparison"],
        "tuning_robust_composite_score": tuning_robust_composite_score,
        "calibration": calibration_summary,
        "classification_threshold": classification_threshold_summary,
        "conformal_thresholds": conformal_thresholds,
        "risk_control_thresholds": risk_control_thresholds,
        "val_selection": val_metrics,
        "val_posthoc_calibration": val_calibration_metrics,
        "val_conformal_calibration": val_conformal_metrics,
        "validation_split": validation_split_summary,
        "manifest_vocab_provenance": manifest_vocab_provenance,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing artifacts in an existing run directory.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
