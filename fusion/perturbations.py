from __future__ import annotations

import random
from typing import Iterable

import torch

from fusion.semantic_categories import (
    SEMANTIC_CATEGORY_DIM,
    api_semantic_counts_from_type_ids,
)

from fusion.quality import (
    refresh_api_quality,
    refresh_graph_quality,
    refresh_align_quality,
    refresh_observable_signals,
)
from fusion.utils import clamp_strength, scalar_float


API_PERTURBATIONS = {
    "api_event_dropout",
    "api_sensitive_event_dropout",
    "api_category_dropout",
    "api_feature_noise",
    "modality_dropout_api",
    "api_degraded",
    "api_missing",
}

GRAPH_PERTURBATIONS = {
    "graph_sparsify",
    "graph_local_break",
    "graph_feature_obfuscation",
    "graph_node_feature_mask",
    "modality_dropout_graph",
    "graph_degraded",
    "graph_missing",
}

MANIFEST_PERTURBATIONS = {
    "manifest_permission_mask",
    "manifest_permission_injection",
    "manifest_intent_mask",
    "manifest_component_mask",
    "manifest_feature_noise",
    "modality_dropout_manifest",
    "manifest_degraded",
    "manifest_missing",
}

COMBINED_PERTURBATIONS = {
    "clean",
    "api_graph_degraded",
    "api_manifest_degraded",
    "graph_manifest_degraded",
    "all_degraded",
}

EVAL_PERTURB_TYPES = (
    {None}
    | API_PERTURBATIONS
    | GRAPH_PERTURBATIONS
    | MANIFEST_PERTURBATIONS
    | COMBINED_PERTURBATIONS
)


def _num_to_perturb(total: int, strength: float) -> int:
    total = max(0, int(total))
    strength = clamp_strength(strength)
    if total <= 0 or strength <= 0.0:
        return 0
    return min(total, max(1, int(round(total * strength))))


def _set_min_pert(data: dict, key: str, strength: float) -> None:
    current = scalar_float(data.get(key), 0.0)
    data[key] = max(current, clamp_strength(strength))


def _zero_tensor_like(value):
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return value


def recompute_api_category_counts(data: dict) -> None:
    ids = data.get("api_type_ids")
    counts = api_semantic_counts_from_type_ids(ids)
    device = ids.device if isinstance(ids, torch.Tensor) else None
    counts = counts.to(device=device) if device is not None else counts
    data["api_semantic_category_counts"] = counts
    data["api_category_counts"] = counts


def _select_api_events(data: dict, keep: torch.Tensor) -> None:
    keep = keep.bool().view(-1)
    n_api = int(keep.numel())
    if n_api <= 0:
        return
    data.setdefault("api_event_count_raw", n_api)
    # Freeze every pre-perturbation budget boundary before mutating the event
    # tensors.  refresh_observable_metadata() can then distinguish synthetic
    # dropout from extractor/runtime truncation instead of treating the
    # post-dropout length as a new encoder budget.
    data.setdefault("api_event_count_before_method_budget", n_api)
    data.setdefault("api_event_count_after_method_budget", n_api)
    data.setdefault("api_event_count_before_extractor_budget", n_api)
    data.setdefault("api_event_count_after_extractor_budget", n_api)
    data.setdefault("api_event_count_before_encoder_budget", n_api)
    data.setdefault("api_event_count_after_encoder_budget", n_api)
    keep_idx = torch.where(keep)[0]

    for key in (
        "api_ids",
        "api_type_ids",
        "api_sensitive_mask",
        "api_method_index",
        "api_in_graph_mask",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == n_api:
            data[key] = value[keep_idx.to(value.device)].clone()

    mask = data.get("mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.size(1) == n_api:
        data["mask"] = mask[:, keep_idx.to(mask.device)].clone()

    edge = data.get("method_api_edge_index")
    if not (
        isinstance(edge, torch.Tensor)
        and edge.ndim == 2
        and edge.size(0) == 2
        and edge.numel() > 0
    ):
        return

    mapping = torch.full((n_api,), -1, dtype=torch.long, device=edge.device)
    keep_edge = keep_idx.to(edge.device)
    mapping[keep_edge] = torch.arange(keep_edge.numel(), dtype=torch.long, device=edge.device)
    dst = edge[1].long()
    valid = (dst >= 0) & (dst < n_api)
    if valid.any():
        valid = valid & (mapping[dst.clamp(0, max(n_api - 1, 0))] >= 0)
    edge = edge[:, valid].clone()
    if edge.numel() > 0:
        edge[1] = mapping[edge[1].long()]
    data["method_api_edge_index"] = edge


def apply_api_event_dropout(data: dict, strength: float, sensitive_only: bool = False) -> dict:
    strength = clamp_strength(strength)
    api_ids = data.get("api_ids")
    if not isinstance(api_ids, torch.Tensor) or api_ids.numel() == 0 or strength <= 0.0:
        return data

    n_api = int(api_ids.numel())
    device = api_ids.device
    if sensitive_only and isinstance(data.get("api_sensitive_mask"), torch.Tensor):
        candidate = torch.where(data["api_sensitive_mask"].to(device).float() > 0.5)[0]
        if candidate.numel() == 0:
            return data
    else:
        candidate = torch.arange(n_api, device=device)

    n_drop = _num_to_perturb(candidate.numel(), strength)
    if n_drop <= 0:
        return data
    drop_idx = candidate[torch.randperm(candidate.numel(), device=device)[:n_drop]]
    keep = torch.ones((n_api,), dtype=torch.bool, device=device)
    keep[drop_idx] = False

    actual = float(drop_idx.numel()) / max(n_api, 1)
    _select_api_events(data, keep)
    _set_min_pert(data, "pert_api", actual)
    data["api_aug_type"] = "api_sensitive_event_dropout" if sensitive_only else "api_event_dropout"
    recompute_api_category_counts(data)
    refresh_api_quality(data)
    refresh_align_quality(data)
    return data


def apply_api_category_dropout(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    ids = data.get("api_type_ids")
    if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
        return data
    non_other = torch.where(ids.long().view(-1) > 0)[0]
    if non_other.numel() == 0:
        return apply_api_event_dropout(data, strength, sensitive_only=False)
    n_drop = _num_to_perturb(non_other.numel(), strength)
    if n_drop <= 0:
        return data
    chosen = non_other[torch.randperm(non_other.numel(), device=ids.device)[:n_drop]]
    ids_new = ids.clone()
    ids_new[chosen] = 0
    data["api_type_ids"] = ids_new
    actual = float(n_drop) / max(int(ids.numel()), 1)
    _set_min_pert(data, "pert_api", actual)
    data["api_aug_type"] = "api_category_dropout"
    recompute_api_category_counts(data)
    refresh_api_quality(data)
    refresh_align_quality(data)
    return data


def apply_api_feature_noise(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    ids = data.get("api_ids")
    if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
        return data
    n = int(ids.numel())
    n_noise = _num_to_perturb(n, strength)
    if n_noise <= 0:
        return data
    idx = torch.randperm(n, device=ids.device)[:n_noise]
    ids_new = ids.clone()
    replacement = torch.randint(1, int(ids.max().item()) + 2, (n_noise,), device=ids.device)
    same = replacement == ids_new[idx]
    replacement[same] = replacement[same] % (int(ids.max().item()) + 1) + 1
    ids_new[idx] = replacement
    data["api_ids"] = ids_new
    actual = float((ids_new != ids).float().mean().item())
    _set_min_pert(data, "pert_api", actual)
    data["api_aug_type"] = "api_feature_noise"
    refresh_api_quality(data)
    refresh_align_quality(data)
    return data


def apply_api_missing(data: dict) -> dict:
    for key in (
        "api_ids",
        "api_type_ids",
        "api_sensitive_mask",
        "api_method_index",
        "api_in_graph_mask",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            data[key] = value[:0].clone()
    edge = data.get("method_api_edge_index")
    if isinstance(edge, torch.Tensor):
        data["method_api_edge_index"] = edge.new_empty((2, 0), dtype=torch.long)
    mask = data.get("mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 2:
        data["mask"] = mask.new_empty((mask.size(0), 0), dtype=torch.float32)
    template = data.get("api_semantic_category_counts", data.get("api_category_counts"))
    if isinstance(template, torch.Tensor) and template.numel() == SEMANTIC_CATEGORY_DIM:
        counts = torch.zeros_like(template)
    else:
        device = template.device if isinstance(template, torch.Tensor) else None
        counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32, device=device)
    data["api_semantic_category_counts"] = counts
    data["api_category_counts"] = counts
    data["q_api"] = 0.0
    data["pert_api"] = 1.0
    data["api_parse_ok"] = False
    data["api_parse_error"] = "synthetic modality missing"
    data["api_aug_type"] = "modality_dropout_api"
    refresh_align_quality(data)
    return data


def apply_graph_sparsify(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    edge = data.get("edge_index")
    if not isinstance(edge, torch.Tensor) or edge.ndim != 2 or edge.size(1) == 0:
        # No edge exists to remove, so this is an actual no-op rather than a
        # fabricated full-strength degradation.
        return data
    data.setdefault("graph_edge_count_reference", int(edge.size(1)))
    keep = torch.rand(edge.size(1), device=edge.device) > strength
    data["edge_index"] = edge[:, keep]
    actual = 1.0 - float(keep.float().mean().item())
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_sparsify"
    refresh_graph_quality(data)
    refresh_align_quality(data)
    return data


def apply_graph_local_break(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    edge = data.get("edge_index")
    if not isinstance(edge, torch.Tensor) or edge.ndim != 2 or edge.size(1) == 0:
        return apply_graph_sparsify(data, 1.0)
    data.setdefault("graph_edge_count_reference", int(edge.size(1)))
    num_nodes = int(data.get("x").size(0)) if isinstance(data.get("x"), torch.Tensor) else int(edge.max().item()) + 1
    sensitive = data.get("sensitive_mask")
    if isinstance(sensitive, torch.Tensor) and sensitive.numel() > 0 and sensitive.bool().any():
        candidates = torch.where(sensitive.to(edge.device).bool())[0]
        center = int(candidates[random.randrange(candidates.numel())].item())
    else:
        center = random.randrange(max(num_nodes, 1))
    local = (edge[0] == center) | (edge[1] == center)
    drop = local & (torch.rand(edge.size(1), device=edge.device) < strength)
    keep = ~drop
    data["edge_index"] = edge[:, keep]
    actual = 1.0 - float(keep.float().mean().item())
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_local_break"
    refresh_graph_quality(data)
    refresh_align_quality(data)
    return data


def apply_graph_feature_obfuscation(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(0) == 0:
        return data
    n = int(x.size(0))
    real_mask = data.get("real_node_mask")
    candidates = (
        torch.where(real_mask.to(x.device).view(-1).bool())[0]
        if isinstance(real_mask, torch.Tensor) and real_mask.numel() == n
        else torch.arange(n, device=x.device)
    )
    n_mask = _num_to_perturb(candidates.numel(), strength)
    if n_mask <= 0:
        return data
    idx = candidates[torch.randperm(candidates.numel(), device=x.device)[:n_mask]]
    x_new = x.clone()
    x_new[idx] = 0.15 * x_new[idx] + 0.03 * torch.randn_like(x_new[idx])
    data["x"] = x_new
    actual = float(n_mask) / max(int(candidates.numel()), 1)
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_feature_obfuscation"
    refresh_graph_quality(data)
    refresh_align_quality(data)
    return data


def apply_graph_node_feature_mask(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(0) == 0:
        return data
    n = int(x.size(0))
    real_mask = data.get("real_node_mask")
    candidates = (
        torch.where(real_mask.to(x.device).view(-1).bool())[0]
        if isinstance(real_mask, torch.Tensor) and real_mask.numel() == n
        else torch.arange(n, device=x.device)
    )
    n_mask = _num_to_perturb(candidates.numel(), strength)
    if n_mask <= 0:
        return data
    idx = candidates[torch.randperm(candidates.numel(), device=x.device)[:n_mask]]
    x_new = x.clone()
    x_new[idx] = 0.0
    data["x"] = x_new
    actual = float(n_mask) / max(int(candidates.numel()), 1)
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_node_feature_mask"
    refresh_graph_quality(data)
    refresh_align_quality(data)
    return data


def apply_graph_missing(data: dict) -> dict:
    x = data.get("x")
    if isinstance(x, torch.Tensor):
        data["x"] = torch.zeros_like(x)
    edge = data.get("edge_index")
    if isinstance(edge, torch.Tensor):
        data["edge_index"] = edge.new_empty((2, 0), dtype=torch.long)
    method_edge = data.get("method_api_edge_index")
    if isinstance(method_edge, torch.Tensor):
        data["method_api_edge_index"] = method_edge.new_empty((2, 0), dtype=torch.long)
    api_in_graph = data.get("api_in_graph_mask")
    if isinstance(api_in_graph, torch.Tensor):
        data["api_in_graph_mask"] = torch.zeros_like(api_in_graph)
    sensitive = data.get("sensitive_mask")
    if isinstance(sensitive, torch.Tensor):
        data["sensitive_mask"] = torch.zeros_like(sensitive)
    graph_counts = data.get("graph_category_counts")
    if isinstance(graph_counts, torch.Tensor):
        data["graph_category_counts"] = torch.zeros_like(graph_counts)
    graph_semantic_counts = data.get("graph_semantic_category_counts")
    if isinstance(graph_semantic_counts, torch.Tensor):
        data["graph_semantic_category_counts"] = torch.zeros_like(graph_semantic_counts)
    data["q_graph"] = 0.0
    data["pert_graph"] = 1.0
    data["graph_parse_ok"] = False
    data["graph_parse_error"] = "synthetic modality missing"
    data["graph_aug_type"] = "modality_dropout_graph"
    refresh_observable_signals(data)
    return data


def _mask_vector_positions(vec: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = vec.clone()
    if out.ndim == 1:
        out[positions] = 0.0
    elif out.ndim == 2:
        out[:, positions] = 0.0
    return out


def _inject_vector_positions(vec: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = vec.clone()
    if out.ndim == 1:
        out[positions] = 1.0
    elif out.ndim == 2:
        out[:, positions] = 1.0
    return out


def _choose_vector_positions(
    vec: torch.Tensor,
    strength: float,
    start: int = 0,
    end: int | None = None,
    *,
    active: bool | None = None,
) -> tuple[torch.Tensor, int]:
    width = int(vec.size(-1))
    start = max(0, min(int(start), width))
    end = width if end is None else max(start, min(int(end), width))
    span = end - start
    strength = clamp_strength(strength)
    if span <= 0 or strength <= 0.0:
        return torch.empty((0,), dtype=torch.long, device=vec.device), 0
    segment = vec[..., start:end].reshape(-1, span)
    if active is True:
        candidates = torch.where((segment.abs() > 1e-8).any(dim=0))[0]
    elif active is False:
        candidates = torch.where(~(segment.abs() > 1e-8).any(dim=0))[0]
    else:
        candidates = torch.arange(span, device=vec.device)
    n = _num_to_perturb(candidates.numel(), strength)
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=vec.device), int(candidates.numel())
    chosen = candidates[torch.randperm(candidates.numel(), device=vec.device)[:n]]
    return start + chosen, int(candidates.numel())


def _actual_fraction(changed: int, eligible: int) -> float:
    return float(changed) / max(int(eligible), 1)


def _update_manifest_perturbation(data: dict, actual: float) -> None:
    _set_min_pert(data, "pert_manifest", actual)


def _update_manifest_semantic_counts(
    data: dict,
    map_key: str,
    relative_positions: torch.Tensor,
    sign: float,
) -> bool:
    counts = data.get("manifest_category_counts")
    mapping = data.get(map_key)
    if (
        not isinstance(counts, torch.Tensor)
        or not isinstance(mapping, torch.Tensor)
        or mapping.ndim != 2
        or mapping.size(1) != counts.numel()
        or relative_positions.numel() == 0
    ):
        return False
    relative_positions = relative_positions.to(mapping.device).long()
    relative_positions = relative_positions[
        (relative_positions >= 0) & (relative_positions < mapping.size(0))
    ]
    if relative_positions.numel() == 0:
        return False
    delta = mapping[relative_positions].sum(dim=0).to(device=counts.device, dtype=counts.dtype)
    updated = (counts + float(sign) * delta).clamp_min(0.0)
    _set_manifest_semantic_counts(data, updated)
    return True


def _require_manifest_semantic_update(changed: bool, map_key: str) -> None:
    if not changed:
        raise ValueError(
            f"Current PT is missing a valid {map_key}; regenerate it with "
            "build_tri_modal_pts_direct.py"
        )


def _set_manifest_semantic_counts(data: dict, updated: torch.Tensor) -> None:
    updated = updated.float().clamp_min(0.0)
    data["manifest_category_counts"] = updated
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        start = (
            int(data.get("manifest_permission_dim", 0))
            + int(data.get("manifest_intent_dim", 0))
            + int(data.get("manifest_feature_dim", 0))
        )
        end = start + SEMANTIC_CATEGORY_DIM
        if end <= vec.size(-1):
            normalized = updated.float() / updated.float().sum().clamp_min(1.0)
            out = vec.clone()
            if out.ndim == 1:
                out[start:end] = normalized.to(device=out.device, dtype=out.dtype)
            else:
                out[:, start:end] = normalized.to(device=out.device, dtype=out.dtype)
            data["manifest_x"] = out


def apply_manifest_permission_mask(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    actual = 0.0
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", max(1, vec.size(-1) // 2)))
        pos, eligible = _choose_vector_positions(vec, strength, 0, perm_dim, active=True)
        if pos.numel() > 0:
            data["manifest_x"] = _mask_vector_positions(vec, pos)
            actual = _actual_fraction(pos.numel(), eligible)
            changed = _update_manifest_semantic_counts(
                data,
                "manifest_permission_category_map",
                pos,
                -1.0,
            )
            _require_manifest_semantic_update(changed, "manifest_permission_category_map")
    ids = data.get("manifest_permission_ids")
    if isinstance(ids, torch.Tensor) and ids.numel() > 0 and "pos" in locals() and pos.numel() > 0:
        removed_ids = pos.to(ids.device).long() + 1
        keep = ~torch.isin(ids.long(), removed_ids)
        data["manifest_permission_ids"] = ids[keep].clone()
    _update_manifest_perturbation(data, actual)
    data["manifest_aug_type"] = "manifest_permission_mask"
    refresh_observable_signals(data)
    return data


def apply_manifest_permission_injection(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    actual = 0.0
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", max(1, vec.size(-1) // 2)))
        pos, eligible = _choose_vector_positions(vec, strength, 0, perm_dim, active=False)
        if pos.numel() > 0:
            data["manifest_x"] = _inject_vector_positions(vec, pos)
            actual = _actual_fraction(pos.numel(), eligible)
            changed = _update_manifest_semantic_counts(
                data,
                "manifest_permission_category_map",
                pos,
                1.0,
            )
            _require_manifest_semantic_update(changed, "manifest_permission_category_map")
            ids = data.get("manifest_permission_ids")
            if isinstance(ids, torch.Tensor):
                injected = pos.to(ids.device).long() + 1
                data["manifest_permission_ids"] = torch.unique(torch.cat([ids.long(), injected])).sort().values
    _update_manifest_perturbation(data, actual)
    data["manifest_aug_type"] = "manifest_permission_injection"
    refresh_observable_signals(data)
    return data


def apply_manifest_intent_mask(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    actual = 0.0
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", max(1, vec.size(-1) // 2)))
        intent_dim = int(data.get("manifest_intent_dim", max(1, vec.size(-1) // 4)))
        pos, eligible = _choose_vector_positions(
            vec, strength, perm_dim, perm_dim + intent_dim, active=True
        )
        if pos.numel() > 0:
            data["manifest_x"] = _mask_vector_positions(vec, pos)
            actual = _actual_fraction(pos.numel(), eligible)
            changed = _update_manifest_semantic_counts(
                data,
                "manifest_intent_category_map",
                pos - perm_dim,
                -1.0,
            )
            _require_manifest_semantic_update(changed, "manifest_intent_category_map")
    ids = data.get("manifest_intent_ids")
    if isinstance(ids, torch.Tensor) and ids.numel() > 0 and "pos" in locals() and pos.numel() > 0:
        removed_ids = (pos.to(ids.device).long() - perm_dim) + 1
        keep = ~torch.isin(ids.long(), removed_ids)
        data["manifest_intent_ids"] = ids[keep].clone()
    _update_manifest_perturbation(data, actual)
    data["manifest_aug_type"] = "manifest_intent_mask"
    refresh_observable_signals(data)
    return data


def apply_manifest_component_mask(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    actual = 0.0
    selected_relative = torch.empty((0,), dtype=torch.long)
    stats = data.get("manifest_stats")
    if isinstance(stats, torch.Tensor) and stats.numel() > 0:
        flat_stats = stats.view(-1)
        component_stat_indices = torch.tensor(
            [1, 2, 3, 4, 9, 10],
            dtype=torch.long,
            device=flat_stats.device,
        )
        component_stat_indices = component_stat_indices[
            component_stat_indices < flat_stats.numel()
        ]
        candidates = component_stat_indices[
            flat_stats[component_stat_indices].abs() > 1e-8
        ]
        n = _num_to_perturb(candidates.numel(), strength)
        if n > 0:
            stats_new = stats.clone()
            selected_relative = candidates[
                torch.randperm(candidates.numel(), device=stats.device)[:n]
            ]
            stats_new.view(-1)[selected_relative] = 0.0
            data["manifest_stats"] = stats_new
            actual = _actual_fraction(selected_relative.numel(), candidates.numel())
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0 and selected_relative.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", 0))
        intent_dim = int(data.get("manifest_intent_dim", 0))
        feature_dim = int(data.get("manifest_feature_dim", 0))
        stats_dim = int(stats.numel()) if isinstance(stats, torch.Tensor) else max(1, vec.size(-1) // 8)
        start = perm_dim + intent_dim + feature_dim + SEMANTIC_CATEGORY_DIM
        if start >= vec.size(-1):
            start = max(0, vec.size(-1) - stats_dim)
        pos = start + selected_relative.to(vec.device)
        pos = pos[pos < vec.size(-1)]
        data["manifest_x"] = _mask_vector_positions(vec, pos)
    component_counts = data.get("manifest_component_category_counts")
    counts = data.get("manifest_category_counts")
    if (
        isinstance(component_counts, torch.Tensor)
        and component_counts.numel() == SEMANTIC_CATEGORY_DIM
        and isinstance(counts, torch.Tensor)
        and counts.numel() == SEMANTIC_CATEGORY_DIM
        and component_counts.float().abs().sum() > 0
    ):
        removed = component_counts.to(device=counts.device, dtype=counts.dtype) * strength
        _set_manifest_semantic_counts(data, (counts - removed).clamp_min(0.0))
        data["manifest_component_category_counts"] = (
            component_counts.float() * (1.0 - strength)
        ).to(device=component_counts.device, dtype=component_counts.dtype)
        actual = max(actual, strength)
    current_components = int(data.get("manifest_component_count", 0) or 0)
    data["manifest_component_count"] = max(0, int(round(current_components * (1.0 - strength))))
    _update_manifest_perturbation(data, actual)
    data["manifest_aug_type"] = "manifest_component_mask"
    refresh_observable_signals(data)
    return data


def apply_manifest_feature_noise(data: dict, strength: float) -> dict:
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return data
    actual = 0.0
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        raw_dim = (
            int(data.get("manifest_permission_dim", 0))
            + int(data.get("manifest_intent_dim", 0))
            + int(data.get("manifest_feature_dim", 0))
            + SEMANTIC_CATEGORY_DIM
            + int(data.get("manifest_stats").numel() if isinstance(data.get("manifest_stats"), torch.Tensor) else 0)
        )
        raw_dim = min(int(vec.size(-1)), raw_dim if raw_dim > 0 else int(vec.size(-1)))
        pos, eligible = _choose_vector_positions(vec, strength, 0, raw_dim)
        if pos.numel() > 0:
            out = vec.float().clone()
            noise_shape = (out.size(0), pos.numel()) if out.ndim == 2 else (pos.numel(),)
            noise = torch.randn(noise_shape, dtype=out.dtype, device=out.device) * (0.05 + 0.20 * strength)
            if out.ndim == 1:
                out[pos] = (out[pos] + noise).clamp(0.0, 1.0)
            else:
                out[:, pos] = (out[:, pos] + noise).clamp(0.0, 1.0)
            data["manifest_x"] = out
            actual = _actual_fraction(pos.numel(), eligible)
    counts = data.get("manifest_category_counts")
    if isinstance(counts, torch.Tensor) and counts.numel() == SEMANTIC_CATEGORY_DIM:
        scale = counts.float().mean().clamp_min(1.0)
        noise = torch.randn_like(counts.float()) * scale * (0.05 + 0.20 * strength)
        _set_manifest_semantic_counts(data, (counts.float() + noise).clamp_min(0.0))
    _update_manifest_perturbation(data, actual)
    data["manifest_aug_type"] = "manifest_feature_noise"
    refresh_observable_signals(data)
    return data


def apply_manifest_missing(data: dict) -> dict:
    for key in (
        "manifest_x",
        "manifest_permission_ids",
        "manifest_intent_ids",
        "manifest_category_counts",
        "manifest_component_category_counts",
        "manifest_stats",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            if key in {"manifest_permission_ids", "manifest_intent_ids"}:
                data[key] = value.new_empty((0,))
            else:
                data[key] = torch.zeros_like(value)
    data["q_manifest"] = 0.0
    data["pert_manifest"] = 1.0
    data["manifest_has_content"] = False
    data["manifest_permission_count"] = 0
    data["manifest_component_count"] = 0
    data["manifest_intent_count"] = 0
    data["manifest_parse_ok"] = False
    data["manifest_parse_error"] = "synthetic modality missing"
    data["manifest_aug_type"] = "modality_dropout_manifest"
    refresh_observable_signals(data)
    return data


def apply_manifest_degraded(data: dict, strength: float) -> dict:
    if clamp_strength(strength) <= 0.0:
        return data
    return _apply_first_effective(
        data,
        strength,
        [
            apply_manifest_permission_mask,
            apply_manifest_permission_injection,
            apply_manifest_intent_mask,
            apply_manifest_component_mask,
            apply_manifest_feature_noise,
        ],
        "pert_manifest",
    )


def _apply_first_effective(
    data: dict,
    strength: float,
    operations: list,
    pert_key: str,
) -> dict:
    before = scalar_float(data.get(pert_key), 0.0)
    operations = list(operations)
    random.shuffle(operations)
    for operation in operations:
        data = operation(data, strength)
        if scalar_float(data.get(pert_key), 0.0) > before:
            return data
    return data


def apply_graph_degraded(data: dict, strength: float) -> dict:
    if clamp_strength(strength) <= 0.0:
        return data
    return _apply_first_effective(
        data,
        strength,
        [
            apply_graph_sparsify,
            apply_graph_local_break,
            apply_graph_feature_obfuscation,
            apply_graph_node_feature_mask,
        ],
        "pert_graph",
    )


# ── perturbation dispatch registry ────────────────────────────────────
# Maps perturb_type → handler(data, strength).  Handlers that discard
# strength (e.g. *_missing) are wrapped in a lambda for a uniform
# two-argument signature.

_PERTURB_REGISTRY: dict = {
    "api_degraded": lambda d, s: random.choice(
        [apply_api_event_dropout, apply_api_category_dropout, apply_api_feature_noise]
    )(d, s),
    "api_missing": lambda d, _s: apply_api_missing(d),
    "modality_dropout_api": lambda d, _s: apply_api_missing(d),
    "api_event_dropout": lambda d, s: apply_api_event_dropout(d, s, sensitive_only=False),
    "api_sensitive_event_dropout": lambda d, s: apply_api_event_dropout(d, s, sensitive_only=True),
    "api_category_dropout": apply_api_category_dropout,
    "api_feature_noise": apply_api_feature_noise,
    "graph_degraded": apply_graph_degraded,
    "graph_missing": lambda d, _s: apply_graph_missing(d),
    "modality_dropout_graph": lambda d, _s: apply_graph_missing(d),
    "graph_sparsify": apply_graph_sparsify,
    "graph_local_break": apply_graph_local_break,
    "graph_feature_obfuscation": apply_graph_feature_obfuscation,
    "graph_node_feature_mask": apply_graph_node_feature_mask,
    "manifest_degraded": apply_manifest_degraded,
    "manifest_missing": lambda d, _s: apply_manifest_missing(d),
    "modality_dropout_manifest": lambda d, _s: apply_manifest_missing(d),
    "manifest_permission_mask": apply_manifest_permission_mask,
    "manifest_permission_injection": apply_manifest_permission_injection,
    "manifest_intent_mask": apply_manifest_intent_mask,
    "manifest_component_mask": apply_manifest_component_mask,
    "manifest_feature_noise": apply_manifest_feature_noise,
}


def _apply_one(data: dict, perturb_type: str, strength: float) -> dict:
    if perturb_type in {None, "clean"}:
        return data
    handler = _PERTURB_REGISTRY.get(perturb_type)
    if handler is None:
        raise ValueError(f"Unsupported perturb_type: {perturb_type}")
    return handler(data, strength)


def apply_perturbation(data: dict, perturb_type: str | None, strength: float) -> dict:
    strength = clamp_strength(strength)
    if perturb_type in {None, "clean"}:
        return data
    if perturb_type == "api_graph_degraded":
        data = _apply_one(data, "api_degraded", strength)
        return _apply_one(data, "graph_degraded", strength)
    if perturb_type == "api_manifest_degraded":
        data = _apply_one(data, "api_degraded", strength)
        return _apply_one(data, "manifest_degraded", strength)
    if perturb_type == "graph_manifest_degraded":
        data = _apply_one(data, "graph_degraded", strength)
        return _apply_one(data, "manifest_degraded", strength)
    if perturb_type == "all_degraded":
        data = _apply_one(data, "api_degraded", strength)
        data = _apply_one(data, "graph_degraded", strength)
        return _apply_one(data, "manifest_degraded", strength)
    return _apply_one(data, perturb_type, strength)


def sample_training_perturbation(
    perturb_prob: float,
    perturb_strengths: Iterable[float],
) -> tuple[str | None, float]:
    if random.random() >= max(0.0, min(1.0, float(perturb_prob))):
        return None, 0.0
    strengths = list(perturb_strengths) or [0.3]
    perturb_type = random.choice(
        [
            "api_degraded",
            "graph_degraded",
            "manifest_degraded",
            "api_graph_degraded",
            "api_manifest_degraded",
            "graph_manifest_degraded",
            "all_degraded",
        ]
    )
    return perturb_type, float(random.choice(strengths))
