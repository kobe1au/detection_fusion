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
    manifest_dim: int = 16,
) -> dict[str, Any]:
    dex_list = [dexes] if isinstance(dexes, dict) else list(dexes)
    source = {**(dex_list[0] if dex_list else {}), **(top_level or {})}
    permission_dim = int(source.get("manifest_permission_dim", 2))
    intent_dim = int(source.get("manifest_intent_dim", 1))
    feature_dim = int(source.get("manifest_feature_dim", 0))
    manifest_x = source.get("manifest_x", torch.ones(manifest_dim))
    manifest_x = torch.as_tensor(manifest_x).float().view(-1)

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
            "api_truncated": False,
            "api_truncation_ratio": 0.0,
            "api_known_type_count": api_count,
            "api_unknown_type_count": 0,
            "dex_parse_ok": True,
            "dex_file_count": len(dex_list),
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
        },
        "manifest_x": manifest_x,
        "manifest_permission_ids": torch.as_tensor(
            source.get("manifest_permission_ids", [1]), dtype=torch.long
        ).view(-1),
        "manifest_intent_ids": torch.as_tensor(
            source.get("manifest_intent_ids", [1]), dtype=torch.long
        ).view(-1),
        "manifest_category_counts": torch.as_tensor(
            source.get("manifest_category_counts", torch.ones(12))
        ).float().view(-1),
        "manifest_component_category_counts": torch.as_tensor(
            source.get("manifest_component_category_counts", torch.zeros(12))
        ).float().view(-1),
        "manifest_permission_category_map": torch.as_tensor(
            source.get(
                "manifest_permission_category_map",
                torch.zeros((permission_dim, 12)),
            )
        ).float(),
        "manifest_intent_category_map": torch.as_tensor(
            source.get(
                "manifest_intent_category_map",
                torch.zeros((intent_dim, 12)),
            )
        ).float(),
        "manifest_stats": torch.as_tensor(
            source.get("manifest_stats", torch.ones(11))
        ).float().view(-1),
        "manifest_meta": dict(source.get("manifest_meta") or {}),
        "manifest_permission_dim": permission_dim,
        "manifest_intent_dim": intent_dim,
        "manifest_feature_dim": feature_dim,
        "q_manifest": torch.as_tensor(source.get("q_manifest", [1.0])).float().view(-1),
        "pert_manifest": torch.as_tensor(source.get("pert_manifest", [0.0])).float().view(-1),
    }


def save_current_pt(
    path: str | Path,
    dexes: dict[str, Any] | list[dict[str, Any]],
    *,
    top_level: dict[str, Any] | None = None,
    manifest_dim: int = 16,
) -> None:
    torch.save(
        current_pt_payload(dexes, top_level=top_level, manifest_dim=manifest_dim),
        path,
    )
