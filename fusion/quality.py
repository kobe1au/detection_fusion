from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch

from fusion.constants import QualityConstants
from fusion.utils import clamp01, scalar_float


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

OBSERVABLE_OPTIONAL_NUMERIC_FIELDS = (
    *OBSERVABLE_PERSISTED_BUDGET_FIELDS,
    "api_runtime_encoder_coverage",
    "api_event_count_before_encoder_budget",
    "api_event_count_after_encoder_budget",
    "api_encoder_coverage",
    "api_truncated_by_extractor_budget",
    "api_truncated_by_encoder_budget",
    "api_integrity_before_encoder_budget",

     # Graph encoder-budget observable fields.
    "graph_feature_valid_ratio",
    "graph_encoder_coverage",
    "graph_truncated_by_encoder_budget",
    "graph_integrity_before_encoder_budget",
)

OBSERVABLE_ERROR_FIELDS = (
    "api_parse_error",
    "graph_parse_error",
    "manifest_parse_error",
)

OBSERVABLE_NUMERIC_FIELDS = (
    *OBSERVABLE_BOOL_FIELDS,
    *OBSERVABLE_COUNT_FIELDS,
    *OBSERVABLE_RATIO_FIELDS,
    *OBSERVABLE_OPTIONAL_NUMERIC_FIELDS,
)

OBSERVABLE_REQUIRED_FIELDS = (
    *OBSERVABLE_BOOL_FIELDS,
    *OBSERVABLE_COUNT_FIELDS,
    *OBSERVABLE_RATIO_FIELDS,
    *OBSERVABLE_PERSISTED_BUDGET_FIELDS,
    *OBSERVABLE_ERROR_FIELDS,
    "schema_version",
)

OBSERVABLE_SIGNAL_FIELDS = (
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
    """Return isolated-node count and largest weak component ratio."""
    num_nodes = max(0, int(num_nodes))
    if num_nodes <= 0:
        return 0, 0.0
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and edge_index.size(0) == 2:
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


def compute_api_integrity_v2(source: Any) -> float:
    """Observable API extraction completeness, independent of availability.

    Completeness reflects (a) a successful parse, (b) the survival of extracted
    API events through synthetic degradation, and (c) API-type coverage. It
    deliberately does not penalise method, DEX or runtime budget truncation.
    Extraction and runtime limits are retained as separate diagnostics. The
    formal model-visible modifier uses runtime encoder coverage only, so fixed
    PT construction budgets are not mistaken for sample-specific degradation.
    Synthetic API dropout still reduces integrity through ``visible_retention``
    (kept events after perturbation relative to the post-budget event count).
    """
    if not _bool(source, "api_parse_ok") or not _bool(source, "dex_parse_ok"):
        return 0.0
    kept = _count(source, "api_event_count_kept")
    extracted = _count(source, "api_event_count_before_encoder_budget", kept)
    after_budget = _count(source, "api_event_count_after_encoder_budget", extracted)
    extracted = max(extracted, kept)
    after_budget = max(min(after_budget, extracted), kept)
    known = _count(source, "api_known_type_count")
    unknown = _count(source, "api_unknown_type_count")
    dex_ratio = clamp01(_number(source, "dex_parse_success_ratio", 1.0))
    if extracted <= 0 and kept <= 0:
        # Successfully parsed but legitimately no framework API events.
        return dex_ratio
    # Visible retention: how many post-budget events survived synthetic
    # degradation. Clean data -> 1.0; API dropout lowers it.
    visible_retention = clamp01(kept / after_budget) if after_budget > 0 else 1.0
    count_consistency = 1.0 if kept <= after_budget else 0.0
    type_total = known + unknown
    type_coverage = 0.0 if kept <= 0 else clamp01(known / max(type_total, kept))
    integrity = clamp01(
        0.20
        + 0.55 * visible_retention
        + 0.10 * count_consistency
        + 0.15 * type_coverage
    )
    return clamp01(integrity * dex_ratio)


def compute_graph_integrity_v2(source: Any) -> float:
    """Observable graph extraction completeness, independent of availability.

    A successful empty graph can be legitimate for the selected extraction
    scope. ``graph_alive`` separately represents whether graph content exists.
    """
    if (
        not _bool(source, "dex_parse_ok")
        or not _bool(source, "graph_parse_ok")
        or _bool(source, "graph_timeout")
    ):
        return 0.0
    nodes = _count(source, "graph_node_count_raw")
    edges = _count(source, "graph_edge_count_raw")
    isolated = _count(source, "graph_isolated_node_count")
    lcc = _number(source, "graph_largest_component_ratio")
    structure_valid = float(isolated <= nodes and 0.0 <= lcc <= 1.0 and edges >= 0)
    method_count = _count(source, "method_count")
    count_valid = float(method_count <= 0 or nodes <= method_count)
    dex_ratio = clamp01(_number(source, "dex_parse_success_ratio", 1.0))
    edge_retention = clamp01(_number(source, "graph_edge_retention_ratio", 1.0))
    feature_valid_ratio = clamp01(_number(source, "graph_feature_valid_ratio", 1.0))
    if nodes <= 0:
        return clamp01((0.85 + 0.10 * structure_valid + 0.05 * count_valid) * dex_ratio)

    # Sparse or disconnected graphs can be legitimate APK structure. Only
    # penalize extreme fragmentation as an observable extraction anomaly;
    # never reward a sample merely for having a large or highly connected graph.
    isolated_ratio = clamp01(isolated / nodes)
    severe_isolation = clamp01((isolated_ratio - 0.80) / 0.20)
    severe_fragmentation = (
        clamp01((0.05 - lcc) / 0.05)
        if nodes > 1 and edges > 0
        else 0.0
    )
    anomaly_penalty = max(
        0.20 * max(severe_isolation, severe_fragmentation),
        0.35 * (1.0 - edge_retention),
        0.35 * (1.0 - feature_valid_ratio),
    )
    base = 0.85 + 0.10 * structure_valid + 0.05 * count_valid
    return clamp01(dex_ratio * base * (1.0 - anomaly_penalty))


def compute_manifest_integrity_v2(source: Any) -> float:
    """Observable Manifest parse/vectorization completeness, not availability.

    A successfully parsed minimal Manifest is complete even when no modeled
    content is present. ``manifest_alive`` carries that availability signal.
    """
    if not _bool(source, "manifest_parse_ok"):
        return 0.0
    coverage = clamp01(_number(source, "manifest_vocab_coverage", 1.0))
    counts_valid = float(
        all(
            _number(source, key, -1.0) >= 0.0
            for key in (
                "manifest_permission_count",
                "manifest_component_count",
                "manifest_intent_count",
            )
        )
    )
    return clamp01(0.75 + 0.20 * coverage + 0.05 * counts_valid)


def compute_api_graph_anchor_support(
    method_api_edge_index: Any,
    num_api_events: int,
    num_graph_nodes: int,
    num_graph_storage_nodes: int | None = None,
) -> float:
    """How much of the API representation is structurally anchored to Graph."""
    num_api_events = max(0, int(num_api_events))
    num_graph_nodes = max(0, int(num_graph_nodes))
    num_graph_storage_nodes = max(
        num_graph_nodes,
        int(num_graph_storage_nodes) if num_graph_storage_nodes is not None else num_graph_nodes,
    )
    if (
        num_api_events <= 0
        or num_graph_nodes <= 0
        or not isinstance(method_api_edge_index, torch.Tensor)
        or method_api_edge_index.ndim != 2
        or method_api_edge_index.size(0) != 2
        or method_api_edge_index.numel() == 0
    ):
        return 0.0
    edge = method_api_edge_index.detach().long()
    valid = (
        (edge[0] >= 0)
        & (edge[0] < num_graph_storage_nodes)
        & (edge[1] >= 0)
        & (edge[1] < num_api_events)
    )
    if not valid.any():
        return 0.0
    edge = edge[:, valid]
    api_coverage = clamp01(edge[1].unique().numel() / num_api_events)
    node_coverage = clamp01(edge[0].unique().numel() / num_graph_nodes)
    if api_coverage <= 0.0 or node_coverage <= 0.0:
        return 0.0
    return clamp01(2.0 * api_coverage * node_coverage / (api_coverage + node_coverage))


def compute_manifest_code_support_and_conflict(
    api_counts: Any,
    graph_counts: Any,
    manifest_counts: Any,
) -> tuple[float, float, float]:
    """Return symmetric Manifest-Code support and directional conflicts."""
    def clean(value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            return torch.empty((0,), dtype=torch.float32)
        return torch.nan_to_num(value.detach().float().view(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    api = clean(api_counts)
    graph = clean(graph_counts)
    manifest = clean(manifest_counts)
    width = max(api.numel(), graph.numel(), manifest.numel())
    if width <= 0:
        return 0.0, 0.0, 0.0

    def pad(value: torch.Tensor) -> torch.Tensor:
        if value.numel() < width:
            value = torch.cat([value, value.new_zeros(width - value.numel())])
        return value

    code_present = torch.maximum(pad(api), pad(graph)) > 0
    manifest_present = pad(manifest) > 0
    code_total = int(code_present.sum().item())
    manifest_total = int(manifest_present.sum().item())
    intersection = int((manifest_present & code_present).sum().item())
    union = int((manifest_present | code_present).sum().item())
    support = intersection / union if union > 0 else 0.0
    if manifest_total <= 0:
        manifest_to_code = 0.0
    else:
        manifest_to_code = int((manifest_present & ~code_present).sum().item()) / manifest_total
    code_to_manifest = (
        int((code_present & ~manifest_present).sum().item()) / code_total
        if code_total > 0
        else 0.0
    )
    return clamp01(support), clamp01(manifest_to_code), clamp01(code_to_manifest)


def compute_raw_alive(source: Any) -> tuple[float, float, float]:
    """Raw availability derived from parse state and extracted content only."""
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


def refresh_observable_metadata(data: dict[str, Any]) -> None:
    """Refresh mutable observable metadata after synthetic degradation."""
    api_ids = data.get("api_ids")
    api_types = data.get("api_type_ids")
    kept = int(api_ids.numel()) if isinstance(api_ids, torch.Tensor) else 0
    data.setdefault("api_parse_ok", isinstance(api_ids, torch.Tensor))
    data.setdefault("api_parse_error", "")
    data.setdefault("dex_parse_ok", True)
    data.setdefault("dex_file_count", 1)
    data.setdefault("dex_parse_success_ratio", 1.0)
    data.setdefault("class_count", 0)
    stored_extractor_kept = _count(data, "api_event_count_kept", kept)
    before_method_budget = _count(
        data,
        "api_event_count_before_method_budget",
        _count(data, "api_event_count_raw", stored_extractor_kept),
    )
    after_method_budget = _count(
        data,
        "api_event_count_after_method_budget",
        _count(data, "api_event_count_raw", stored_extractor_kept),
    )
    before_extractor_budget = _count(
        data,
        "api_event_count_before_extractor_budget",
        after_method_budget,
    )
    after_extractor_budget = _count(
        data,
        "api_event_count_after_extractor_budget",
        stored_extractor_kept,
    )
    before_method_budget = max(before_method_budget, after_method_budget, kept)
    after_method_budget = max(min(after_method_budget, before_method_budget), kept)
    before_extractor_budget = max(
        min(before_extractor_budget, after_method_budget), after_extractor_budget, kept
    )
    after_extractor_budget = max(
        min(after_extractor_budget, before_extractor_budget), kept
    )

    extracted_before_budget = _count(
        data, "api_event_count_before_encoder_budget", after_extractor_budget
    )
    extracted_before_budget = max(
        min(extracted_before_budget, after_extractor_budget), kept
    )
    after_budget_default = min(extracted_before_budget, kept) if extracted_before_budget > 0 else kept
    after_budget = _count(data, "api_event_count_after_encoder_budget", after_budget_default)
    after_budget = max(min(after_budget, extracted_before_budget), kept)
    raw = max(_count(data, "api_event_count_raw", before_extractor_budget), before_extractor_budget)
    data["api_event_count_kept"] = kept
    data["api_event_count_before_method_budget"] = before_method_budget
    data["api_event_count_after_method_budget"] = after_method_budget
    data["api_event_count_before_extractor_budget"] = before_extractor_budget
    data["api_event_count_after_extractor_budget"] = after_extractor_budget
    data["api_event_count_before_encoder_budget"] = extracted_before_budget
    data["api_event_count_after_encoder_budget"] = after_budget
    data["api_event_count_raw"] = raw
    data["api_truncated"] = bool(after_extractor_budget < before_method_budget)
    data["api_truncation_ratio"] = (
        clamp01(1.0 - after_extractor_budget / before_method_budget)
        if before_method_budget > 0
        else 0.0
    )
    data["api_extractor_coverage"] = (
        clamp01(after_extractor_budget / before_method_budget)
        if before_method_budget > 0
        else 1.0
    )
    data["api_runtime_encoder_coverage"] = (
        clamp01(after_budget / extracted_before_budget)
        if extracted_before_budget > 0
        else 1.0
    )
    data["api_encoder_coverage"] = (
        clamp01(after_budget / before_method_budget)
        if before_method_budget > 0
        else 1.0
    )
    data["api_truncated_by_extractor_budget"] = float(
        before_method_budget > 0 and after_extractor_budget < before_method_budget
    )
    data["api_truncated_by_encoder_budget"] = float(
        extracted_before_budget > 0 and after_budget < extracted_before_budget
    )
    if isinstance(api_types, torch.Tensor) and api_types.numel() == kept:
        data["api_known_type_count"] = int((api_types.view(-1).long() > 0).sum().item())
        data["api_unknown_type_count"] = kept - int(data["api_known_type_count"])
    else:
        data["api_known_type_count"] = 0
        data["api_unknown_type_count"] = kept

    x = data.get("x")
    real_mask = data.get("real_node_mask")
    if isinstance(real_mask, torch.Tensor):
        nodes = int(real_mask.view(-1).bool().sum().item())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        nodes = int(x.size(0))
    else:
        nodes = 0
    if isinstance(x, torch.Tensor) and x.ndim == 2 and x.size(0) > 0:
        if isinstance(real_mask, torch.Tensor) and real_mask.numel() == x.size(0):
            real_x = x[real_mask.view(-1).bool()]
        else:
            real_x = x[:nodes]
        if real_x.numel() > 0:
            finite_nonzero = torch.isfinite(real_x).all(dim=1) & (
                real_x.abs().sum(dim=1) > 1.0e-8
            )
            data["graph_feature_valid_ratio"] = float(
                finite_nonzero.float().mean().item()
            )
        else:
            data["graph_feature_valid_ratio"] = 1.0
    else:
        data["graph_feature_valid_ratio"] = 1.0
    edge_index = data.get("edge_index")
    edges = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 else 0
    isolated, lcc = graph_structure_stats(edge_index, nodes)
    data["graph_node_count_raw"] = nodes
    previous_edges = _count(data, "graph_edge_count_raw", edges)
    reference_edges = max(_count(data, "graph_edge_count_reference", previous_edges), edges)
    data["graph_edge_count_reference"] = reference_edges
    data["graph_edge_count_raw"] = edges
    data["graph_edge_retention_ratio"] = (
        clamp01(edges / reference_edges) if reference_edges > 0 else 1.0
    )
    data["graph_isolated_node_count"] = isolated
    data["graph_largest_component_ratio"] = lcc
    data.setdefault("graph_parse_ok", isinstance(x, torch.Tensor))
    data.setdefault("graph_parse_error", "")
    data.setdefault("graph_timeout", False)
    data.setdefault("method_count", nodes)

    permission_ids = data.get("manifest_permission_ids")
    intent_ids = data.get("manifest_intent_ids")
    manifest_counts = data.get("manifest_category_counts")
    manifest_stats = data.get("manifest_stats")
    data.setdefault("manifest_parse_ok", isinstance(data.get("manifest_x"), torch.Tensor))
    data.setdefault("manifest_parse_error", "")
    data.setdefault("manifest_vocab_coverage", 1.0)
    data.setdefault("manifest_component_count", 0)
    data["manifest_permission_count"] = int(permission_ids.numel()) if isinstance(permission_ids, torch.Tensor) else _count(data, "manifest_permission_count")
    data["manifest_intent_count"] = int(intent_ids.numel()) if isinstance(intent_ids, torch.Tensor) else _count(data, "manifest_intent_count")
    has_semantics = isinstance(manifest_counts, torch.Tensor) and bool(manifest_counts.detach().float().abs().sum() > 0)
    has_stats = isinstance(manifest_stats, torch.Tensor) and bool(manifest_stats.detach().float().abs().sum() > 0)
    data["manifest_has_content"] = bool(
        _count(data, "manifest_permission_count")
        + _count(data, "manifest_component_count")
        + _count(data, "manifest_intent_count")
        > 0
        or has_semantics
        or has_stats
    )
    data.setdefault("schema_version", "runtime-fallback")


def refresh_observable_signals(data: dict[str, Any]) -> None:
    refresh_observable_metadata(data)
    api_integrity = compute_api_integrity_v2(data)
    graph_integrity = compute_graph_integrity_v2(data)
    manifest_integrity = compute_manifest_integrity_v2(data)
    code_integrity = clamp01(math.sqrt(api_integrity * graph_integrity))
    anchor = compute_api_graph_anchor_support(
        data.get("method_api_edge_index"),
        _count(data, "api_event_count_kept"),
        _count(data, "graph_node_count_raw"),
        int(data.get("x").size(0)) if isinstance(data.get("x"), torch.Tensor) else None,
    )
    support, manifest_conflict, code_conflict = compute_manifest_code_support_and_conflict(
        data.get("api_semantic_category_counts", data.get("api_category_counts")),
        data.get("graph_semantic_category_counts", data.get("graph_category_counts")),
        data.get("manifest_category_counts"),
    )
    api_alive, graph_alive, manifest_alive = compute_raw_alive(data)
    data.update(
        {
            "api_integrity": api_integrity,
            "api_integrity_before_encoder_budget": api_integrity,
            "graph_integrity": graph_integrity,
            "manifest_integrity": manifest_integrity,
            "code_integrity": code_integrity,
            "api_graph_anchor_support": anchor,
            "manifest_code_support": support,
            "manifest_to_code_conflict": manifest_conflict,
            "code_to_manifest_conflict": code_conflict,
            "api_alive": api_alive,
            "graph_alive": graph_alive,
            "manifest_alive": manifest_alive,
            # Compatibility aliases. These no longer encode synthetic strength.
            "q_api": api_integrity,
            "q_graph": graph_integrity,
            "q_manifest": manifest_integrity,
            "q_align": anchor,
            "r_api": api_integrity,
            "r_graph": graph_integrity,
            "r_manifest": manifest_integrity,
        }
    )


# Legacy quality helpers retained for explicit baseline/compatibility runs.
def compute_api_quality(api_ids, api_type_ids=None, api_in_graph_mask=None) -> float:
    if not isinstance(api_ids, torch.Tensor):
        return 0.0
    api_ids = api_ids.view(-1)
    n = int(api_ids.numel())
    if n <= 0:
        return 0.0
    count_score = min(1.0, n / QualityConstants.API_COUNT_NORM)
    diversity_score = min(1.0, float(api_ids.unique().numel()) / max(n, 1) * QualityConstants.API_DIVERSITY_SCALE)
    coverage_score = (
        float(api_in_graph_mask.float().view(-1).mean().item())
        if isinstance(api_in_graph_mask, torch.Tensor) and api_in_graph_mask.numel() == n
        else 0.0
    )
    type_score = (
        float((api_type_ids.long().view(-1) > 0).float().mean().item())
        if isinstance(api_type_ids, torch.Tensor) and api_type_ids.numel() == n
        else 0.0
    )
    return clamp01(
        QualityConstants.API_COUNT_WEIGHT * count_score
        + QualityConstants.API_DIVERSITY_WEIGHT * diversity_score
        + QualityConstants.API_COVERAGE_WEIGHT * coverage_score
        + QualityConstants.API_TYPE_WEIGHT * type_score
    )


def compute_graph_quality(edge_index, num_nodes: int, node_features=None, real_node_mask=None) -> float:
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return 0.0
    num_edges = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 else 0
    node_score = 1.0 - math.exp(-num_nodes / QualityConstants.GRAPH_NODE_NORM)
    edge_score = 1.0 - math.exp(-num_edges / max(num_nodes, 1))
    feature_score = 1.0
    if isinstance(node_features, torch.Tensor) and node_features.ndim == 2:
        real = node_features[real_node_mask.view(-1).bool()].float() if isinstance(real_node_mask, torch.Tensor) and real_node_mask.numel() == node_features.size(0) else node_features[:num_nodes].float()
        feature_score = float((torch.isfinite(real).all(dim=1) & (real.abs().sum(dim=1) > 1e-8)).float().mean().item()) if real.numel() > 0 else 0.0
    return clamp01(QualityConstants.GRAPH_NODE_WEIGHT * node_score + QualityConstants.GRAPH_EDGE_WEIGHT * edge_score + QualityConstants.GRAPH_FEATURE_WEIGHT * feature_score)


def compute_align_quality(
    q_api: float,
    q_graph: float,
    method_api_edge_index,
    num_nodes: int,
    num_api: int,
    num_storage_nodes: int | None = None,
) -> float:
    del q_api, q_graph
    return compute_api_graph_anchor_support(
        method_api_edge_index, num_api, num_nodes, num_storage_nodes
    )


def compute_manifest_quality(manifest_x, manifest_category_counts=None, manifest_stats=None, manifest_meta=None) -> float:
    meta = manifest_meta if isinstance(manifest_meta, dict) else {}
    if meta.get("parse_error"):
        return 0.0
    stored = meta.get("quality_score")
    if stored is not None:
        return clamp01(scalar_float(stored, 0.0))
    return 1.0 if any(isinstance(v, torch.Tensor) and v.numel() > 0 for v in (manifest_x, manifest_category_counts, manifest_stats)) else 0.0


def refresh_api_quality(data: dict) -> None:
    refresh_observable_signals(data)


def refresh_graph_quality(data: dict) -> None:
    refresh_observable_signals(data)


def refresh_align_quality(data: dict) -> None:
    refresh_observable_signals(data)


def refresh_code_quality(data: dict) -> None:
    refresh_observable_signals(data)
