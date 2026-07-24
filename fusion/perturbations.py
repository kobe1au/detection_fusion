from __future__ import annotations

import math

import torch

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

    keep = torch.rand(edge.size(1), device=edge.device) > strength
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


def _choose_active_permission_positions(
    vec: torch.Tensor,
    permission_dim: int,
    strength: float,
) -> torch.Tensor:
    if vec.ndim not in {1, 2}:
        raise ValueError(
            f"manifest_x must be one- or two-dimensional, got shape={tuple(vec.shape)}"
        )
    width = int(vec.size(-1))
    permission_dim = max(0, min(int(permission_dim), width))
    if permission_dim <= 0:
        return torch.empty((0,), dtype=torch.long, device=vec.device)
    segment = vec[..., :permission_dim].reshape(-1, permission_dim)
    active = torch.where((segment.abs() > 1e-8).any(dim=0))[0]
    count = _num_to_perturb(active.numel(), strength)
    if count <= 0:
        return active[:0]
    chosen = active[
        torch.randperm(active.numel(), device=vec.device)[:count]
    ]
    return chosen


def _mask_vector_positions(
    vec: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    out = vec.clone()
    if out.ndim == 1:
        out[positions] = 0.0
    else:
        out[:, positions] = 0.0
    return out


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


def _remove_manifest_permission_semantics(
    data: dict,
    positions: torch.Tensor,
) -> None:
    counts = data.get("manifest_category_counts")
    mapping = data.get("manifest_permission_category_map")
    if (
        not isinstance(counts, torch.Tensor)
        or not isinstance(mapping, torch.Tensor)
        or mapping.ndim != 2
        or mapping.size(1) != counts.numel()
        or mapping.size(0) < int(data.get("manifest_permission_dim", 0))
    ):
        raise ValueError(
            "Current PT is missing a valid manifest_permission_category_map; "
            "regenerate it with build_tri_modal_pts_direct.py"
        )
    selected = positions.to(mapping.device).long()
    delta = mapping[selected].sum(dim=0).to(
        device=counts.device,
        dtype=counts.dtype,
    )
    _set_manifest_semantic_counts(data, (counts - delta).clamp_min(0.0))


def _sync_manifest_permission_count(data: dict) -> None:
    """Make IDs, manifest_stats and the embedded stats segment agree."""

    ids = data.get("manifest_permission_ids")
    stats = data.get("manifest_stats")
    vec = data.get("manifest_x")
    if not isinstance(ids, torch.Tensor):
        raise ValueError(
            "Current PT is missing manifest_permission_ids; regenerate it with "
            "build_tri_modal_pts_direct.py"
        )
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

    count = int(torch.unique(ids.long().reshape(-1)).numel())
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
    vec = data.get("manifest_x")
    if not isinstance(vec, torch.Tensor) or vec.numel() == 0:
        return data

    permission_dim = int(data.get("manifest_permission_dim", 0))
    if permission_dim < 0 or permission_dim > vec.size(-1):
        raise ValueError(
            "manifest_permission_dim is outside manifest_x: "
            f"permission_dim={permission_dim} width={vec.size(-1)}"
        )
    positions = _choose_active_permission_positions(
        vec,
        permission_dim,
        strength,
    )
    ids = data.get("manifest_permission_ids")
    if not isinstance(ids, torch.Tensor):
        raise ValueError(
            "Current PT is missing manifest_permission_ids; regenerate it with "
            "build_tri_modal_pts_direct.py"
        )

    active_ids = torch.unique(ids.long().reshape(-1)).sort().values
    if permission_dim == 0:
        active_positions = torch.empty(
            (0,), dtype=torch.long, device=vec.device
        )
    else:
        active_positions = (
            torch.where(
                (
                    vec[..., :permission_dim]
                    .reshape(-1, permission_dim)
                    .abs()
                    > 1e-8
                ).any(dim=0)
            )[0]
            + 1
        )
    if not torch.equal(active_ids.cpu(), active_positions.long().cpu()):
        raise ValueError(
            "manifest_permission_ids and active permission bits disagree; "
            "regenerate the PT"
        )

    if positions.numel() > 0:
        data["manifest_x"] = _mask_vector_positions(vec, positions)
        _remove_manifest_permission_semantics(data, positions)
        removed_ids = positions.to(ids.device).long() + 1
        keep = ~torch.isin(ids.long().reshape(-1), removed_ids)
        data["manifest_permission_ids"] = ids.reshape(-1)[keep].clone()

    _sync_manifest_permission_count(data)
    data["manifest_aug_type"] = "manifest_permission_mask"
    refresh_hard_availability(data)
    return data


def apply_manifest_missing(data: dict) -> dict:
    for key in (
        "manifest_x",
        "manifest_permission_ids",
        "manifest_category_counts",
        "manifest_stats",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            if key == "manifest_permission_ids":
                data[key] = value.reshape(-1)[:0].clone()
            else:
                data[key] = torch.zeros_like(value)
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
