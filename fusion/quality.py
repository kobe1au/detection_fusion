from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch

from fusion.utils import clamp01


# Persisted APK -> PT schema.  These fields remain part of the immutable PT
# payload so existing extraction artifacts and the Manifest-only migration keep
# their provenance contract.  The runtime Dataset validates the schema but does
# not feed these historical quality/coverage values to the model.
OBSERVABLE_SCHEMA_VERSION = "observable-v2"

OBSERVABLE_BOOL_FIELDS = (
    "api_parse_ok",
    "api_truncated",
    "dex_parse_ok",
    "graph_parse_ok",
    "graph_timeout",
    "manifest_parse_ok",
    "manifest_has_content",
)

OBSERVABLE_COUNT_FIELDS = (
    "api_event_count_raw",
    "api_event_count_kept",
    "api_known_type_count",
    "api_unknown_type_count",
    "dex_file_count",
    "class_count",
    "method_count",
    "graph_node_count_raw",
    "graph_edge_count_raw",
    "graph_isolated_node_count",
    "manifest_permission_count",
    "manifest_component_count",
    "manifest_intent_count",
)

OBSERVABLE_RATIO_FIELDS = (
    "api_truncation_ratio",
    "graph_largest_component_ratio",
    "manifest_vocab_coverage",
)

OBSERVABLE_PERSISTED_BUDGET_FIELDS = (
    "dex_parse_success_ratio",
    "api_event_count_before_method_budget",
    "api_event_count_after_method_budget",
    "api_event_count_before_extractor_budget",
    "api_event_count_after_extractor_budget",
    "api_extractor_coverage",
    "api_truncated_by_extractor_budget",
)

OBSERVABLE_ERROR_FIELDS = (
    "api_parse_error",
    "graph_parse_error",
    "manifest_parse_error",
)

OBSERVABLE_REQUIRED_FIELDS = (
    *OBSERVABLE_BOOL_FIELDS,
    *OBSERVABLE_COUNT_FIELDS,
    *OBSERVABLE_RATIO_FIELDS,
    *OBSERVABLE_PERSISTED_BUDGET_FIELDS,
    *OBSERVABLE_ERROR_FIELDS,
    "schema_version",
)


def _value(source: Any, key: str, default: Any = 0.0) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _bool(source: Any, key: str, default: bool = False) -> bool:
    value = _value(source, key, default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return bool(default)
        value = value.detach().view(-1)[0].item()
    return bool(value)


def _number(source: Any, key: str, default: float = 0.0) -> float:
    value = _value(source, key, default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        value = value.detach().float().view(-1)[0].item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _count(source: Any, key: str, default: int = 0) -> int:
    return max(0, int(round(_number(source, key, float(default)))))


def graph_structure_stats(edge_index: Any, num_nodes: int) -> tuple[int, float]:
    """Return isolated-node count and largest weak component ratio for PT build."""

    num_nodes = max(0, int(num_nodes))
    if num_nodes <= 0:
        return 0, 0.0
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    if (
        isinstance(edge_index, torch.Tensor)
        and edge_index.ndim == 2
        and edge_index.size(0) == 2
    ):
        for src, dst in edge_index.detach().long().t().cpu().tolist():
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
    isolated = sum(1 for neighbors in adjacency if not neighbors)
    seen: set[int] = set()
    largest = 0
    for start in range(num_nodes):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        largest = max(largest, size)
    return isolated, clamp01(largest / max(num_nodes, 1))


def compute_raw_alive(source: Any) -> tuple[float, float, float]:
    """Return the three hard availability bits from current content and parse state."""

    api_alive = float(
        _bool(source, "api_parse_ok")
        and _bool(source, "dex_parse_ok")
        and _count(source, "api_event_count_kept") > 0
    )
    graph_alive = float(
        _bool(source, "dex_parse_ok")
        and _bool(source, "graph_parse_ok")
        and not _bool(source, "graph_timeout")
        and _count(source, "graph_node_count_raw") > 0
    )
    manifest_alive = float(
        _bool(source, "manifest_parse_ok")
        and _bool(source, "manifest_has_content")
    )
    return api_alive, graph_alive, manifest_alive


def refresh_hard_availability(data: dict[str, Any]) -> None:
    """Refresh only current-content counts and the three binary alive fields.

    This is the complete runtime metadata path used by fusion.  It deliberately
    does not reconstruct extraction integrity, visibility/coverage ratios,
    cross-modal support, perturbation strength, or any PT quality alias.
    """

    api_ids = data.get("api_ids")
    api_count = int(api_ids.numel()) if isinstance(api_ids, torch.Tensor) else 0
    data.setdefault("api_parse_ok", isinstance(api_ids, torch.Tensor))
    data.setdefault("dex_parse_ok", True)
    data["api_event_count_kept"] = api_count

    x = data.get("x")
    real_mask = data.get("real_node_mask")
    if isinstance(real_mask, torch.Tensor):
        graph_nodes = int(real_mask.view(-1).bool().sum().item())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        graph_nodes = int(x.size(0))
    else:
        graph_nodes = 0
    data.setdefault("graph_parse_ok", isinstance(x, torch.Tensor))
    data.setdefault("graph_timeout", False)
    data["graph_node_count_raw"] = graph_nodes

    permission_ids = data.get("manifest_permission_ids")
    intent_ids = data.get("manifest_intent_ids")
    permission_count = (
        int(permission_ids.numel())
        if isinstance(permission_ids, torch.Tensor)
        else _count(data, "manifest_permission_count")
    )
    intent_count = (
        int(intent_ids.numel())
        if isinstance(intent_ids, torch.Tensor)
        else _count(data, "manifest_intent_count")
    )
    component_count = _count(data, "manifest_component_count")
    manifest_counts = data.get("manifest_category_counts")
    manifest_stats = data.get("manifest_stats")
    has_semantics = (
        isinstance(manifest_counts, torch.Tensor)
        and bool(manifest_counts.detach().float().abs().sum().item() > 0.0)
    )
    has_stats = (
        isinstance(manifest_stats, torch.Tensor)
        and bool(manifest_stats.detach().float().abs().sum().item() > 0.0)
    )
    data.setdefault(
        "manifest_parse_ok", isinstance(data.get("manifest_x"), torch.Tensor)
    )
    data["manifest_permission_count"] = permission_count
    data["manifest_intent_count"] = intent_count
    data["manifest_has_content"] = bool(
        permission_count + component_count + intent_count > 0
        or has_semantics
        or has_stats
    )

    api_alive, graph_alive, manifest_alive = compute_raw_alive(data)
    data["api_alive"] = api_alive
    data["graph_alive"] = graph_alive
    data["manifest_alive"] = manifest_alive
