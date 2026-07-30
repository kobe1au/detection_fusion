from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fusion.pt_schema import PT_SCHEMA_VERSION
from fusion.quality import OBSERVABLE_REQUIRED_FIELDS, OBSERVABLE_SCHEMA_VERSION


def current_pt_payload(
    dexes: dict[str, Any] | list[dict[str, Any]],
    *,
    top_level: dict[str, Any] | None = None,
    manifest_dim: int = 32,
    sid: str | None = None,
) -> dict[str, Any]:
    raw_dex_list = [dexes] if isinstance(dexes, dict) else list(dexes)
    dex_list: list[dict[str, Any]] = []
    for raw_dex in raw_dex_list:
        dex = dict(raw_dex)
        call_x = dex.get("call_x")
        if call_x is None:
            call_x = torch.empty((0, 515), dtype=torch.float32)
        call_x = torch.as_tensor(call_x)
        num_nodes = int(call_x.shape[0]) if call_x.ndim == 2 else 0
        api_ids = torch.as_tensor(
            dex.get("api_ids", torch.empty((0,), dtype=torch.long))
        )
        api_count = int(api_ids.numel())
        dex.setdefault("call_x", call_x)
        dex.setdefault(
            "call_edge_index",
            torch.empty((2, 0), dtype=torch.long),
        )
        dex.setdefault(
            "call_sensitive_mask",
            torch.zeros((num_nodes,), dtype=torch.uint8),
        )
        dex.setdefault("api_ids", api_ids)
        dex.setdefault(
            "api_type_ids",
            torch.zeros((api_count,), dtype=torch.uint8),
        )
        dex.setdefault(
            "api_sensitive_mask",
            torch.zeros((api_count,), dtype=torch.uint8),
        )
        dex_list.append(dex)
    source = {**(dex_list[0] if dex_list else {}), **(top_level or {})}
    sample_id = str(sid or source.get("sha256") or "test-sample").strip().lower()
    permission_dim = int(source.get("manifest_permission_dim", 2))
    intent_dim = int(source.get("manifest_intent_dim", 1))
    feature_dim = int(source.get("manifest_feature_dim", 0))
    permission_ids = torch.as_tensor(
        source.get("manifest_permission_ids", [1]),
        dtype=torch.long,
    ).view(-1)
    intent_ids = torch.as_tensor(
        source.get("manifest_intent_ids", [1]),
        dtype=torch.long,
    ).view(-1)
    source_meta = dict(source.get("manifest_meta") or {})
    raw_permissions = source_meta.get("permissions")
    if raw_permissions is None:
        permissions = [
            f"permission.test.{int(permission_id)}"
            for permission_id in permission_ids.tolist()
        ]
    else:
        permissions = sorted(
            {
                str(value).strip().lower()
                for value in raw_permissions
                if str(value).strip()
            }
        )
    permission_token_ids = torch.as_tensor(
        source.get(
            "manifest_permission_token_ids",
            [
                int(permission_ids[index])
                if index < int(permission_ids.numel())
                else 0
                for index in range(len(permissions))
            ],
        ),
        dtype=torch.long,
    ).view(-1)
    category_counts = torch.as_tensor(
        source.get("manifest_category_counts", torch.ones(12))
    ).float().view(-1)
    manifest_stats = torch.as_tensor(
        source.get("manifest_stats", torch.ones(11))
    ).float().view(-1)
    required_manifest_dim = (
        permission_dim + intent_dim + feature_dim + 12 + 11
    )
    requested_x = torch.as_tensor(
        source.get("manifest_x", torch.empty((0,)))
    ).float().view(-1)
    manifest_x = torch.zeros(
        max(manifest_dim, required_manifest_dim, int(requested_x.numel())),
        dtype=torch.float32,
    )
    active = permission_ids[
        (permission_ids >= 1) & (permission_ids <= permission_dim)
    ]
    if active.numel():
        manifest_x[active - 1] = 1.0
    active_intents = intent_ids[
        (intent_ids >= 1) & (intent_ids <= intent_dim)
    ]
    if active_intents.numel():
        manifest_x[permission_dim + active_intents - 1] = 1.0
    category_start = permission_dim + intent_dim + feature_dim
    manifest_x[category_start : category_start + 12] = (
        category_counts / category_counts.sum().clamp_min(1.0)
    )
    manifest_x[
        category_start + 12 : category_start + 12 + 11
    ] = manifest_stats

    api_count = sum(
        int(torch.as_tensor(dex.get("api_ids", [])).numel()) for dex in dex_list
    )
    graph_nodes = sum(
        int(torch.as_tensor(dex.get("call_x", torch.empty((0, 0)))).shape[0])
        for dex in dex_list
    )
    graph_edges = sum(
        int(torch.as_tensor(dex.get("call_edge_index", torch.empty((2, 0)))).shape[-1])
        for dex in dex_list
    )
    observable = {key: 0 for key in OBSERVABLE_REQUIRED_FIELDS}
    observable.update(
        {
            "api_parse_ok": True,
            "api_parse_error": "",
            "api_event_count_raw": api_count,
            "api_event_count_kept": api_count,
            "api_event_count_before_method_budget": api_count,
            "api_event_count_after_method_budget": api_count,
            "api_event_count_before_extractor_budget": api_count,
            "api_event_count_after_extractor_budget": api_count,
            "api_extractor_coverage": 1.0,
            "api_truncated_by_extractor_budget": 0.0,
            "api_truncated": False,
            "api_truncation_ratio": 0.0,
            "api_known_type_count": api_count,
            "api_unknown_type_count": 0,
            "dex_parse_ok": True,
            "dex_file_count": len(dex_list),
            "dex_parse_success_ratio": 1.0,
            "class_count": max(graph_nodes, 1),
            "method_count": graph_nodes,
            "graph_parse_ok": True,
            "graph_parse_error": "",
            "graph_timeout": False,
            "graph_node_count_raw": graph_nodes,
            "graph_edge_count_raw": graph_edges,
            "graph_isolated_node_count": 0,
            "graph_largest_component_ratio": 1.0 if graph_nodes else 0.0,
            "manifest_parse_ok": True,
            "manifest_parse_error": "",
            "manifest_has_content": bool(manifest_x.abs().sum() > 0),
            "manifest_vocab_coverage": 1.0,
            "manifest_permission_count": int(
                torch.as_tensor(source.get("manifest_permission_ids", [1])).numel()
            ),
            "manifest_component_count": 0,
            "manifest_intent_count": int(
                torch.as_tensor(source.get("manifest_intent_ids", [1])).numel()
            ),
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        }
    )
    observable.update(dict(source.get("observable_metadata") or {}))
    observable["schema_version"] = OBSERVABLE_SCHEMA_VERSION

    return {
        "dex_list": dex_list,
        "observable_metadata": observable,
        "direct_build_meta": {
            "pt_schema_version": PT_SCHEMA_VERSION,
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
            "build_fingerprint": "test-current-schema",
            "dex_success_ratio": 1.0,
            "sha256": sample_id,
        },
        "manifest_x": manifest_x,
        "manifest_permission_ids": permission_ids,
        "manifest_permission_token_ids": permission_token_ids,
        "manifest_intent_ids": intent_ids,
        "manifest_category_counts": category_counts,
        "manifest_component_category_counts": torch.as_tensor(
            source.get("manifest_component_category_counts", torch.zeros(12))
        ).float().view(-1),
        "manifest_stats": manifest_stats,
        "manifest_meta": {
            **source_meta,
            "permissions": permissions,
            "sha256": sample_id,
        },
        "manifest_permission_dim": permission_dim,
        "manifest_intent_dim": intent_dim,
        "manifest_feature_dim": feature_dim,
    }


def save_current_pt(
    path: str | Path,
    dexes: dict[str, Any] | list[dict[str, Any]],
    *,
    top_level: dict[str, Any] | None = None,
    manifest_dim: int = 32,
) -> None:
    torch.save(
        current_pt_payload(
            dexes,
            top_level=top_level,
            manifest_dim=manifest_dim,
            sid=Path(path).stem,
        ),
        path,
    )
