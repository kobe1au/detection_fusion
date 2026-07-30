from __future__ import annotations

import math

import torch

from fusion.manifest_features import (
    category_counts_from_strings,
    normalize_manifest_permissions,
)
from fusion.quality import refresh_hard_availability
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM
from fusion.utils import clamp_strength


# Formal controlled-degradation surface.  Keep this list intentionally small:
# one graded mechanism per modality plus the three missing-modality endpoints.
GRADED_PERTURBATIONS = (
    "api_event_dropout",
    "graph_sparsify",
    "manifest_permission_mask",
)
MISSING_PERTURBATIONS = (
    "api_missing",
    "graph_missing",
    "manifest_missing",
)
EVAL_PERTURB_TYPES = {
    None,
    "clean",
    *GRADED_PERTURBATIONS,
    *MISSING_PERTURBATIONS,
}

# manifest_features.py defines the first Manifest statistic as
# log1p(permission_count) / 6.  A permission mask must update both copies of
# that statistic or the clean count remains available to the Manifest encoder.
_MANIFEST_PERMISSION_COUNT_LOG_NORM = 6.0
_API_EVENT_VECTOR_KEYS = (
    "api_ids",
    "api_type_ids",
    "api_sensitive_mask",
)


def _num_to_perturb(total: int, strength: float) -> int:
    total = max(0, int(total))
    strength = clamp_strength(strength)
    if total <= 0 or strength <= 0.0:
        return 0
    return min(total, max(1, int(round(total * strength))))


def _require_api_encoder_vector(
    data: dict,
    key: str,
    expected_length: int,
) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Current PT is missing aligned API tensor {key!r}; regenerate it "
            "with build_tri_modal_pts_direct.py"
        )
    flattened = value.reshape(-1)
    if flattened.numel() != expected_length:
        raise ValueError(
            f"Aligned API tensor {key!r} has length {flattened.numel()}, "
            f"expected {expected_length}"
        )
    return flattened


def _select_api_events(data: dict, keep: torch.Tensor) -> None:
    """Apply one selector to the three tensors consumed by the API encoder."""

    keep = keep.bool().reshape(-1)
    n_api = int(keep.numel())
    if n_api <= 0:
        return

    aligned = {
        key: _require_api_encoder_vector(data, key, n_api)
        for key in _API_EVENT_VECTOR_KEYS
    }
    keep_idx = torch.where(keep)[0]

    for key, value in aligned.items():
        data[key] = value[keep_idx.to(value.device)].clone()


def apply_api_event_dropout(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    api_ids = data.get("api_ids")
    if (
        not isinstance(api_ids, torch.Tensor)
        or api_ids.numel() == 0
        or strength <= 0.0
    ):
        return data

    n_api = int(api_ids.numel())
    n_drop = _num_to_perturb(n_api, strength)
    if n_drop <= 0:
        return data
    drop_idx = torch.randperm(n_api, device=api_ids.device)[:n_drop]
    keep = torch.ones((n_api,), dtype=torch.bool, device=api_ids.device)
    keep[drop_idx] = False

    _select_api_events(data, keep)
    data["api_aug_type"] = "api_event_dropout"
    refresh_hard_availability(data)
    return data


def apply_api_missing(data: dict) -> dict:
    for key in _API_EVENT_VECTOR_KEYS:
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            data[key] = value.reshape(-1)[:0].clone()
    data["api_parse_ok"] = False
    data["api_aug_type"] = "api_missing"
    refresh_hard_availability(data)
    return data


def apply_graph_sparsify(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    edge = data.get("edge_index")
    if not isinstance(edge, torch.Tensor) or edge.ndim != 2 or edge.size(1) == 0:
        return data

    # Remove an exact strength-controlled number of edges.  Independent
    # Bernoulli draws can leave a small graph unchanged even when the view is
    # labelled degraded, which weakens the interpretation of the evaluation
    # x-axis.
    num_edges = int(edge.size(1))
    num_to_remove = _num_to_perturb(num_edges, strength)
    remove_positions = torch.randperm(
        num_edges,
        device=edge.device,
    )[:num_to_remove]
    keep = torch.ones((num_edges,), dtype=torch.bool, device=edge.device)
    keep[remove_positions] = False
    data["edge_index"] = edge[:, keep]
    data["graph_aug_type"] = "graph_sparsify"
    refresh_hard_availability(data)
    return data


def apply_graph_missing(data: dict) -> dict:
    x = data.get("x")
    if isinstance(x, torch.Tensor):
        data["x"] = torch.zeros_like(x)
    edge = data.get("edge_index")
    if isinstance(edge, torch.Tensor):
        data["edge_index"] = edge.new_empty((2, 0), dtype=torch.long)
    sensitive = data.get("sensitive_mask")
    if isinstance(sensitive, torch.Tensor):
        data["sensitive_mask"] = torch.zeros_like(sensitive)

    data["graph_parse_ok"] = False
    data["graph_aug_type"] = "graph_missing"
    refresh_hard_availability(data)
    return data


def _set_manifest_semantic_counts(data: dict, updated: torch.Tensor) -> None:
    updated = updated.float().clamp_min(0.0)
    data["manifest_category_counts"] = updated
    vec = data.get("manifest_x")
    if not isinstance(vec, torch.Tensor) or vec.numel() == 0:
        return
    start = (
        int(data.get("manifest_permission_dim", 0))
        + int(data.get("manifest_intent_dim", 0))
        + int(data.get("manifest_feature_dim", 0))
    )
    end = start + SEMANTIC_CATEGORY_DIM
    if end > vec.size(-1):
        raise ValueError(
            "manifest_x is too short for its declared semantic-category layout: "
            f"width={vec.size(-1)} required={end}"
        )
    normalized = updated / updated.sum().clamp_min(1.0)
    out = vec.clone()
    if out.ndim == 1:
        out[start:end] = normalized.to(device=out.device, dtype=out.dtype)
    else:
        out[:, start:end] = normalized.to(device=out.device, dtype=out.dtype)
    data["manifest_x"] = out


def _validated_manifest_permission_state(
    data: dict,
) -> tuple[
    list[str],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Validate the CPU-only token alignment used by permission masking."""

    tokens = data.get("manifest_permission_tokens")
    if not isinstance(tokens, list):
        raise ValueError(
            "Current PT is missing canonical manifest permission tokens; run "
            "scripts/migrate_manifest_vocab_pts.py"
        )
    canonical_tokens = normalize_manifest_permissions(tokens)
    if tokens != canonical_tokens:
        raise ValueError(
            "manifest permission tokens must be lower-case, de-duplicated, and sorted"
        )
    token_ids = data.get("manifest_permission_token_ids")
    ids = data.get("manifest_permission_ids")
    counts = data.get("manifest_category_counts")
    stats = data.get("manifest_stats")
    vec = data.get("manifest_x")
    permission_dim = int(data.get("manifest_permission_dim", -1))
    if (
        not isinstance(token_ids, torch.Tensor)
        or not isinstance(ids, torch.Tensor)
        or not isinstance(counts, torch.Tensor)
        or not isinstance(stats, torch.Tensor)
        or not isinstance(vec, torch.Tensor)
    ):
        raise ValueError(
            "Current PT is missing Manifest permission-alignment tensors; run "
            "scripts/migrate_manifest_vocab_pts.py"
        )
    token_ids = token_ids.long().reshape(-1)
    ids = ids.long().reshape(-1)
    counts = counts.float().reshape(-1)
    stats = stats.float().reshape(-1)
    if token_ids.numel() != len(tokens):
        raise ValueError(
            "manifest_permission_token_ids must align one-to-one with permission tokens"
        )
    if vec.ndim not in {1, 2}:
        raise ValueError(
            f"manifest_x must be one- or two-dimensional, got {tuple(vec.shape)}"
        )
    if permission_dim < 0 or permission_dim > vec.size(-1):
        raise ValueError(
            "manifest_permission_dim is outside manifest_x: "
            f"permission_dim={permission_dim} width={vec.size(-1)}"
        )
    if (
        counts.numel() != SEMANTIC_CATEGORY_DIM
        or not bool(torch.isfinite(counts).all().item())
        or bool((counts < 0.0).any().item())
    ):
        raise ValueError(
            "manifest_category_counts must contain exactly "
            f"{SEMANTIC_CATEGORY_DIM} finite non-negative values"
        )
    if stats.numel() < 1 or not bool(torch.isfinite(stats).all().item()):
        raise ValueError("manifest_stats must be a non-empty finite tensor")
    if bool(
        ((token_ids < 0) | (token_ids > permission_dim)).any().item()
    ):
        raise ValueError(
            f"manifest_permission_token_ids must be within [0, {permission_dim}]"
        )
    expected_ids = token_ids[token_ids > 0].unique(sorted=True)
    if int((token_ids > 0).sum().item()) != int(expected_ids.numel()):
        raise ValueError(
            "non-zero manifest_permission_token_ids must be unique"
        )
    if not torch.equal(ids.cpu(), expected_ids.cpu()):
        raise ValueError(
            "manifest_permission_ids disagrees with manifest_permission_token_ids"
        )
    if permission_dim == 0:
        active_ids = torch.empty((0,), dtype=torch.long, device=vec.device)
    else:
        permission_segment = vec[..., :permission_dim].reshape(
            -1, permission_dim
        )
        active_ids = (
            torch.where((permission_segment.abs() > 1.0e-8).any(dim=0))[0].long()
            + 1
        )
    if not torch.equal(active_ids.cpu(), expected_ids.cpu()):
        raise ValueError(
            "Manifest permission bits disagree with manifest_permission_token_ids"
        )
    expected_count_stat = math.log1p(len(tokens)) / _MANIFEST_PERMISSION_COUNT_LOG_NORM
    actual_count_stat = float(stats[0].item())
    if not math.isclose(
        actual_count_stat,
        expected_count_stat,
        rel_tol=1.0e-5,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            "manifest_stats permission count disagrees with "
            "manifest_meta.permissions; run scripts/migrate_manifest_vocab_pts.py"
        )
    return tokens, token_ids, ids, counts, stats, vec, permission_dim


def _set_manifest_permission_count(data: dict, count: int) -> None:
    """Synchronize the raw count statistic and its embedded feature."""

    stats = data["manifest_stats"]
    vec = data["manifest_x"]
    if not isinstance(stats, torch.Tensor) or stats.numel() < 1:
        raise ValueError(
            "Current PT is missing manifest_stats; regenerate it with "
            "build_tri_modal_pts_direct.py"
        )
    if not isinstance(vec, torch.Tensor) or vec.ndim not in {1, 2}:
        raise ValueError(
            "Current PT is missing a valid manifest_x; regenerate it with "
            "build_tri_modal_pts_direct.py"
        )

    count = max(0, int(count))
    if not math.isfinite(float(stats.detach().float().reshape(-1)[0].item())):
        raise ValueError("manifest_stats permission count must be finite")
    normalized = math.log1p(count) / _MANIFEST_PERMISSION_COUNT_LOG_NORM
    stats_new = stats.clone()
    stats_new.reshape(-1)[0] = normalized
    data["manifest_stats"] = stats_new
    data["manifest_permission_count"] = count

    stats_start = (
        int(data.get("manifest_permission_dim", 0))
        + int(data.get("manifest_intent_dim", 0))
        + int(data.get("manifest_feature_dim", 0))
        + SEMANTIC_CATEGORY_DIM
    )
    if stats_start + stats.numel() > vec.size(-1):
        raise ValueError(
            "manifest_x is too short for its declared statistics layout: "
            f"width={vec.size(-1)} required={stats_start + stats.numel()}"
        )
    vec_new = vec.clone()
    if vec_new.ndim == 1:
        vec_new[stats_start] = normalized
    else:
        vec_new[:, stats_start] = normalized
    data["manifest_x"] = vec_new


def apply_manifest_permission_mask(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    (
        tokens,
        token_ids,
        ids,
        original_counts,
        _stats,
        vec,
        permission_dim,
    ) = _validated_manifest_permission_state(data)
    # A Manifest can be valid yet contain no permissions.  Do not label that
    # sample as degraded when the operator cannot change any model input.
    if not tokens:
        return data

    num_to_remove = _num_to_perturb(len(tokens), strength)
    remove_positions = (
        torch.randperm(len(tokens), device=token_ids.device)[:num_to_remove]
        if num_to_remove > 0
        else torch.empty((0,), dtype=torch.long, device=token_ids.device)
    )
    keep = torch.ones(
        (len(tokens),), dtype=torch.bool, device=token_ids.device
    )
    keep[remove_positions] = False
    remaining_tokens = [
        token for index, token in enumerate(tokens) if bool(keep[index].item())
    ]
    remaining_token_ids = token_ids[keep].clone()
    remaining_known_ids = remaining_token_ids[
        remaining_token_ids > 0
    ].unique(sorted=True)

    original_permission_counts = category_counts_from_strings(tokens).to(
        device=original_counts.device,
        dtype=original_counts.dtype,
    )
    non_permission_counts = original_counts - original_permission_counts
    if bool((non_permission_counts < -1.0e-5).any().item()):
        raise ValueError(
            "manifest_category_counts is inconsistent with manifest_meta.permissions; "
            "run scripts/migrate_manifest_vocab_pts.py"
        )
    remaining_permission_counts = category_counts_from_strings(
        remaining_tokens
    ).to(
        device=original_counts.device,
        dtype=original_counts.dtype,
    )
    updated_counts = (
        non_permission_counts.clamp_min(0.0) + remaining_permission_counts
    )

    updated_vec = vec.clone()
    updated_vec[..., :permission_dim] = 0.0
    if remaining_known_ids.numel() > 0:
        positions = (remaining_known_ids - 1).to(updated_vec.device)
        if updated_vec.ndim == 1:
            updated_vec[positions] = 1.0
        else:
            updated_vec[:, positions] = 1.0
    data["manifest_x"] = updated_vec
    data["manifest_permission_tokens"] = remaining_tokens
    data["manifest_permission_token_ids"] = remaining_token_ids
    data["manifest_permission_ids"] = remaining_known_ids.to(
        device=ids.device
    )
    _set_manifest_semantic_counts(data, updated_counts)
    _set_manifest_permission_count(data, len(remaining_tokens))
    data["manifest_aug_type"] = "manifest_permission_mask"
    refresh_hard_availability(data)
    return data


def apply_manifest_missing(data: dict) -> dict:
    for key in (
        "manifest_x",
        "manifest_permission_ids",
        "manifest_permission_token_ids",
        "manifest_category_counts",
        "manifest_stats",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            if key in {
                "manifest_permission_ids",
                "manifest_permission_token_ids",
            }:
                data[key] = value.reshape(-1)[:0].clone()
            else:
                data[key] = torch.zeros_like(value)
    data["manifest_permission_tokens"] = []
    data["manifest_has_content"] = False
    data["manifest_permission_count"] = 0
    data["manifest_component_count"] = 0
    data["manifest_intent_count"] = 0
    data["manifest_parse_ok"] = False
    data["manifest_aug_type"] = "manifest_missing"
    refresh_hard_availability(data)
    return data


_PERTURB_REGISTRY = {
    "api_event_dropout": apply_api_event_dropout,
    "graph_sparsify": apply_graph_sparsify,
    "manifest_permission_mask": apply_manifest_permission_mask,
    "api_missing": lambda data, _strength: apply_api_missing(data),
    "graph_missing": lambda data, _strength: apply_graph_missing(data),
    "manifest_missing": lambda data, _strength: apply_manifest_missing(data),
}


def apply_perturbation(
    data: dict,
    perturb_type: str | None,
    strength: float,
) -> dict:
    if perturb_type in {None, "clean"}:
        return data
    handler = _PERTURB_REGISTRY.get(perturb_type)
    if handler is None:
        raise ValueError(f"Unsupported perturb_type: {perturb_type}")
    return handler(data, clamp_strength(strength))
