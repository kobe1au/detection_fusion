from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from fusion.quality import (
    OBSERVABLE_BOOL_FIELDS,
    OBSERVABLE_COUNT_FIELDS,
    OBSERVABLE_ERROR_FIELDS,
    OBSERVABLE_PERSISTED_BUDGET_FIELDS,
    OBSERVABLE_RATIO_FIELDS,
    OBSERVABLE_REQUIRED_FIELDS,
    OBSERVABLE_SCHEMA_VERSION,
)


# Schema v5 makes two previously implicit contracts explicit:
#   * every DEX owns a complete, internally aligned Graph/API tensor bundle;
#   * every canonical Manifest permission owns an aligned vocabulary token id.
# Old payloads must be migrated in place; the runtime intentionally has no
# compatibility/repair path for them.
PT_SCHEMA_VERSION = 5
PT_AUDIT_CERTIFICATE_VERSION = 1

CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS = (
    "dex_list",
    "observable_metadata",
    "direct_build_meta",
    "manifest_x",
    "manifest_permission_ids",
    "manifest_permission_token_ids",
    "manifest_intent_ids",
    "manifest_category_counts",
    "manifest_component_category_counts",
    "manifest_stats",
    "manifest_meta",
    "manifest_permission_dim",
    "manifest_intent_dim",
    "manifest_feature_dim",
)

CURRENT_DEX_REQUIRED_FIELDS = (
    "call_x",
    "call_edge_index",
    "call_sensitive_mask",
    "api_ids",
    "api_type_ids",
    "api_sensitive_mask",
)

# These fields belonged to retired quality-proxy/masking implementations.
# Schema-v5 formal payloads have one source of truth and reject them rather
# than silently carrying unused values into future experiments.
RETIRED_PT_TOP_LEVEL_FIELDS = (
    "manifest_permission_category_map",
    "manifest_intent_category_map",
    "q_manifest",
    "pert_manifest",
)

# Frozen extractor taxonomy: 0=other/unknown and 1..15 are named API types.
MAX_API_TYPE_ID = 15

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_MASK_DTYPES = {*_INTEGER_DTYPES, torch.bool}


class PTSchemaValidationError(ValueError):
    """A persisted PT violates the current, non-repairing data contract."""


def pt_audit_entries_sha256(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _python_scalar(value: Any, field: str, *, path: str | Path) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.numel() != 1:
            raise PTSchemaValidationError(
                f"{field} must be one scalar value; path={path}"
            )
        return value.detach().cpu().item()
    return value


def _finite_number(value: Any, field: str, *, path: str | Path) -> float:
    value = _python_scalar(value, field, path=path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PTSchemaValidationError(
            f"{field} must be a finite number; path={path}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PTSchemaValidationError(
            f"{field} must be finite; path={path}"
        )
    return number


def validate_observable_metadata(
    observable: Mapping[str, Any],
    *,
    path: str | Path,
) -> None:
    """Reject truthy strings, fractional counts, and invalid quality ratios."""

    for field in OBSERVABLE_BOOL_FIELDS:
        value = _python_scalar(observable[field], field, path=path)
        if not (
            isinstance(value, bool)
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) in {0.0, 1.0}
            )
        ):
            raise PTSchemaValidationError(
                f"observable_metadata.{field} must be bool or numeric 0/1; "
                f"path={path}"
            )

    for field in OBSERVABLE_COUNT_FIELDS:
        number = _finite_number(
            observable[field],
            f"observable_metadata.{field}",
            path=path,
        )
        if number < 0 or not number.is_integer():
            raise PTSchemaValidationError(
                f"observable_metadata.{field} must be a non-negative integer; "
                f"path={path}"
            )

    for field in OBSERVABLE_RATIO_FIELDS:
        number = _finite_number(
            observable[field],
            f"observable_metadata.{field}",
            path=path,
        )
        if not 0.0 <= number <= 1.0:
            raise PTSchemaValidationError(
                f"observable_metadata.{field} must be within [0,1]; path={path}"
            )

    budget_counts = {
        "api_event_count_before_method_budget",
        "api_event_count_after_method_budget",
        "api_event_count_before_extractor_budget",
        "api_event_count_after_extractor_budget",
    }
    budget_ratios = {"dex_parse_success_ratio", "api_extractor_coverage"}
    budget_bits = {"api_truncated_by_extractor_budget"}
    if (
        budget_counts | budget_ratios | budget_bits
        != set(OBSERVABLE_PERSISTED_BUDGET_FIELDS)
    ):
        raise RuntimeError(
            "Observable persisted-budget schema changed; update strict PT validation"
        )
    for field in budget_counts:
        number = _finite_number(
            observable[field],
            f"observable_metadata.{field}",
            path=path,
        )
        if number < 0 or not number.is_integer():
            raise PTSchemaValidationError(
                f"observable_metadata.{field} must be a non-negative integer; "
                f"path={path}"
            )
    for field in budget_ratios | budget_bits:
        number = _finite_number(
            observable[field],
            f"observable_metadata.{field}",
            path=path,
        )
        if not 0.0 <= number <= 1.0:
            raise PTSchemaValidationError(
                f"observable_metadata.{field} must be within [0,1]; path={path}"
            )
    if float(observable["api_truncated_by_extractor_budget"]) not in {0.0, 1.0}:
        raise PTSchemaValidationError(
            "observable_metadata.api_truncated_by_extractor_budget must be 0/1; "
            f"path={path}"
        )
    for field in OBSERVABLE_ERROR_FIELDS:
        if not isinstance(observable[field], str):
            raise PTSchemaValidationError(
                f"observable_metadata.{field} must be a string; path={path}"
            )


def _context(path: str | Path, *, dex_index: int | None = None) -> str:
    value = f"path={path}"
    if dex_index is not None:
        value += f" dex_index={dex_index}"
    return value


def _require_tensor(
    owner: Mapping[str, Any],
    field: str,
    *,
    path: str | Path,
    dex_index: int | None = None,
) -> torch.Tensor:
    value = owner.get(field)
    if not isinstance(value, torch.Tensor):
        raise PTSchemaValidationError(
            f"{field} must be a torch.Tensor; {_context(path, dex_index=dex_index)}"
        )
    if value.layout != torch.strided:
        raise PTSchemaValidationError(
            f"{field} must use dense strided storage, got {value.layout}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    return value


def _require_vector(
    owner: Mapping[str, Any],
    field: str,
    *,
    path: str | Path,
    dex_index: int | None = None,
) -> torch.Tensor:
    value = _require_tensor(owner, field, path=path, dex_index=dex_index)
    if value.ndim != 1:
        raise PTSchemaValidationError(
            f"{field} must have shape [N], got {tuple(value.shape)}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    return value


def _require_integral(
    value: torch.Tensor,
    field: str,
    *,
    path: str | Path,
    dex_index: int | None = None,
    allow_bool: bool = False,
) -> None:
    accepted = _MASK_DTYPES if allow_bool else _INTEGER_DTYPES
    if value.dtype not in accepted:
        raise PTSchemaValidationError(
            f"{field} must use an integer"
            f"{'/bool' if allow_bool else ''} dtype, got {value.dtype}; "
            f"{_context(path, dex_index=dex_index)}"
        )


def _require_binary(
    value: torch.Tensor,
    field: str,
    *,
    path: str | Path,
    dex_index: int | None = None,
) -> None:
    if value.dtype not in _MASK_DTYPES and not value.dtype.is_floating_point:
        raise PTSchemaValidationError(
            f"{field} must use a real numeric/bool dtype, got {value.dtype}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    if (
        value.dtype.is_floating_point
        and value.numel()
        and not bool(torch.isfinite(value).all().item())
    ):
        raise PTSchemaValidationError(
            f"{field} contains non-finite values; "
            f"{_context(path, dex_index=dex_index)}"
        )
    if value.numel() and bool(((value != 0) & (value != 1)).any().item()):
        raise PTSchemaValidationError(
            f"{field} must contain only 0/1 values; "
            f"{_context(path, dex_index=dex_index)}"
        )


def validate_current_dex(
    dex: Any,
    *,
    path: str | Path,
    dex_index: int,
) -> None:
    """Validate one schema-v5 DEX without filling, clipping, or filtering it."""

    if not isinstance(dex, Mapping):
        raise PTSchemaValidationError(
            f"dex entry must be a mapping; {_context(path, dex_index=dex_index)}"
        )
    missing = [field for field in CURRENT_DEX_REQUIRED_FIELDS if field not in dex]
    if missing:
        raise PTSchemaValidationError(
            f"dex entry is missing required fields {missing}; "
            f"{_context(path, dex_index=dex_index)}"
        )

    call_x = _require_tensor(dex, "call_x", path=path, dex_index=dex_index)
    if call_x.ndim != 2 or int(call_x.shape[1]) <= 0:
        raise PTSchemaValidationError(
            f"call_x must have shape [N,D] with D>0, got {tuple(call_x.shape)}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    if not call_x.dtype.is_floating_point or call_x.dtype.is_complex:
        raise PTSchemaValidationError(
            f"call_x must use a real floating dtype, got {call_x.dtype}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    if call_x.numel() and not bool(torch.isfinite(call_x).all().item()):
        raise PTSchemaValidationError(
            f"call_x contains non-finite values; "
            f"{_context(path, dex_index=dex_index)}"
        )
    num_nodes = int(call_x.shape[0])

    edge_index = _require_tensor(
        dex,
        "call_edge_index",
        path=path,
        dex_index=dex_index,
    )
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise PTSchemaValidationError(
            "call_edge_index must have shape [2,E], "
            f"got {tuple(edge_index.shape)}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    _require_integral(
        edge_index,
        "call_edge_index",
        path=path,
        dex_index=dex_index,
    )
    if edge_index.numel():
        edge_min = int(edge_index.min().item())
        edge_max = int(edge_index.max().item())
        if edge_min < 0 or edge_max >= num_nodes:
            raise PTSchemaValidationError(
                "call_edge_index contains node indices outside "
                f"[0,{max(num_nodes - 1, -1)}]: min={edge_min} max={edge_max}; "
                f"{_context(path, dex_index=dex_index)}"
            )

    node_mask = _require_vector(
        dex,
        "call_sensitive_mask",
        path=path,
        dex_index=dex_index,
    )
    if int(node_mask.numel()) != num_nodes:
        raise PTSchemaValidationError(
            "call_sensitive_mask length must equal call_x rows: "
            f"actual={node_mask.numel()} expected={num_nodes}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    _require_binary(
        node_mask,
        "call_sensitive_mask",
        path=path,
        dex_index=dex_index,
    )

    api_ids = _require_vector(
        dex,
        "api_ids",
        path=path,
        dex_index=dex_index,
    )
    api_type_ids = _require_vector(
        dex,
        "api_type_ids",
        path=path,
        dex_index=dex_index,
    )
    api_sensitive = _require_vector(
        dex,
        "api_sensitive_mask",
        path=path,
        dex_index=dex_index,
    )
    _require_integral(api_ids, "api_ids", path=path, dex_index=dex_index)
    _require_integral(
        api_type_ids,
        "api_type_ids",
        path=path,
        dex_index=dex_index,
    )
    _require_binary(
        api_sensitive,
        "api_sensitive_mask",
        path=path,
        dex_index=dex_index,
    )
    api_count = int(api_ids.numel())
    lengths = {
        "api_ids": api_count,
        "api_type_ids": int(api_type_ids.numel()),
        "api_sensitive_mask": int(api_sensitive.numel()),
    }
    if len(set(lengths.values())) != 1:
        raise PTSchemaValidationError(
            f"API tensors must have equal lengths, got {lengths}; "
            f"{_context(path, dex_index=dex_index)}"
        )
    if api_ids.numel() and int(api_ids.min().item()) < 0:
        raise PTSchemaValidationError(
            f"api_ids must be non-negative; {_context(path, dex_index=dex_index)}"
        )
    if api_type_ids.numel():
        type_min = int(api_type_ids.min().item())
        type_max = int(api_type_ids.max().item())
        if type_min < 0 or type_max > MAX_API_TYPE_ID:
            raise PTSchemaValidationError(
                f"api_type_ids must be within [0,{MAX_API_TYPE_ID}], "
                f"got min={type_min} max={type_max}; "
                f"{_context(path, dex_index=dex_index)}"
            )


def validate_current_dex_list(
    dex_list: Any,
    *,
    path: str | Path,
) -> list[dict[str, Any]]:
    """Validate every DEX in a persisted payload and return it unchanged."""

    if not isinstance(dex_list, list) or not dex_list:
        raise PTSchemaValidationError(
            f"dex_list must be a non-empty list of mappings; path={path}"
        )
    for dex_index, dex in enumerate(dex_list):
        validate_current_dex(dex, path=path, dex_index=dex_index)
    return dex_list


def _strict_nonnegative_float_vector(
    payload: Mapping[str, Any],
    field: str,
    *,
    path: str | Path,
    length: int | None = None,
) -> torch.Tensor:
    value = _require_vector(payload, field, path=path)
    if not value.dtype.is_floating_point:
        raise PTSchemaValidationError(
            f"{field} must use a floating dtype, got {value.dtype}; path={path}"
        )
    if length is not None and int(value.numel()) != int(length):
        raise PTSchemaValidationError(
            f"{field} must have length {length}, got {value.numel()}; path={path}"
        )
    if value.numel() and (
        not bool(torch.isfinite(value).all().item())
        or bool((value < 0).any().item())
    ):
        raise PTSchemaValidationError(
            f"{field} must contain finite non-negative values; path={path}"
        )
    return value


def _strict_dimension(
    payload: Mapping[str, Any],
    field: str,
    *,
    path: str | Path,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PTSchemaValidationError(
            f"{field} must be a non-negative integer; path={path}"
        )
    return int(value)


def validate_manifest_payload_contract(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
) -> None:
    """Validate the exact vectorizer layout consumed by the Manifest branch."""

    permission_dim = _strict_dimension(
        payload, "manifest_permission_dim", path=path
    )
    intent_dim = _strict_dimension(payload, "manifest_intent_dim", path=path)
    feature_dim = _strict_dimension(payload, "manifest_feature_dim", path=path)
    manifest_x = _require_vector(payload, "manifest_x", path=path)
    if not manifest_x.dtype.is_floating_point:
        raise PTSchemaValidationError(
            f"manifest_x must use a floating dtype, got {manifest_x.dtype}; path={path}"
        )
    if manifest_x.numel() and not bool(torch.isfinite(manifest_x).all().item()):
        raise PTSchemaValidationError(
            f"manifest_x must contain only finite values; path={path}"
        )

    category_counts = _strict_nonnegative_float_vector(
        payload,
        "manifest_category_counts",
        path=path,
        length=12,
    )
    _strict_nonnegative_float_vector(
        payload,
        "manifest_component_category_counts",
        path=path,
        length=12,
    )
    manifest_stats = _strict_nonnegative_float_vector(
        payload,
        "manifest_stats",
        path=path,
        length=11,
    )
    required_dim = permission_dim + intent_dim + feature_dim + 12 + 11
    if int(manifest_x.numel()) < required_dim:
        raise PTSchemaValidationError(
            "manifest_x is shorter than its declared vectorizer layout: "
            f"actual={manifest_x.numel()} required={required_dim}; path={path}"
        )

    permission_bits = manifest_x[:permission_dim]
    intent_start = permission_dim
    intent_bits = manifest_x[intent_start : intent_start + intent_dim]
    feature_start = intent_start + intent_dim
    feature_bits = manifest_x[feature_start : feature_start + feature_dim]
    category_start = feature_start + feature_dim
    category_values = manifest_x[category_start : category_start + 12]
    stats_start = category_start + 12
    stats_values = manifest_x[stats_start : stats_start + 11]
    padding = manifest_x[required_dim:]

    for field, bits in (
        ("manifest_x permission segment", permission_bits),
        ("manifest_x intent segment", intent_bits),
        ("manifest_x feature segment", feature_bits),
    ):
        if bits.numel() and bool(((bits != 0.0) & (bits != 1.0)).any().item()):
            raise PTSchemaValidationError(
                f"{field} must contain only 0/1 values; path={path}"
            )
    if padding.numel() and bool((padding != 0.0).any().item()):
        raise PTSchemaValidationError(
            f"manifest_x padding beyond the declared layout must be zero; path={path}"
        )

    expected_category = category_counts / category_counts.sum().clamp_min(1.0)
    if not torch.allclose(
        category_values.float(),
        expected_category.float(),
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise PTSchemaValidationError(
            "manifest_x category segment disagrees with "
            f"manifest_category_counts; path={path}"
        )
    if not torch.allclose(
        stats_values.float(),
        manifest_stats.float(),
        rtol=0.0,
        atol=0.0,
    ):
        raise PTSchemaValidationError(
            f"manifest_x stats segment disagrees with manifest_stats; path={path}"
        )

    intent_ids = _require_vector(payload, "manifest_intent_ids", path=path)
    if intent_ids.dtype != torch.long:
        raise PTSchemaValidationError(
            "manifest_intent_ids must use torch.long, "
            f"got {intent_ids.dtype}; path={path}"
        )
    if intent_ids.numel():
        if (
            int(intent_ids.min().item()) < 1
            or int(intent_ids.max().item()) > intent_dim
        ):
            raise PTSchemaValidationError(
                f"manifest_intent_ids must be within [1,{intent_dim}]; path={path}"
            )
        if intent_ids.numel() > 1 and bool(
            (intent_ids[1:] <= intent_ids[:-1]).any().item()
        ):
            raise PTSchemaValidationError(
                "manifest_intent_ids must be strictly increasing and unique; "
                f"path={path}"
            )
    active_intent_ids = (
        torch.nonzero(intent_bits > 0.5, as_tuple=False).view(-1).long() + 1
    )
    if not torch.equal(active_intent_ids, intent_ids):
        raise PTSchemaValidationError(
            "manifest_x active intent bits must exactly match "
            f"manifest_intent_ids; path={path}"
        )

def validate_manifest_permission_contract(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
) -> None:
    """Validate canonical permission/token/vocabulary alignment in schema v5."""

    meta = payload.get("manifest_meta")
    if not isinstance(meta, Mapping):
        raise PTSchemaValidationError(
            f"manifest_meta must be a mapping; path={path}"
        )
    raw_permissions = meta.get("permissions")
    if not isinstance(raw_permissions, list) or any(
        not isinstance(value, str) for value in raw_permissions
    ):
        raise PTSchemaValidationError(
            f"manifest_meta.permissions must be a list of strings; path={path}"
        )
    canonical_permissions = sorted(
        {
            value.strip().lower()
            for value in raw_permissions
            if value.strip()
        }
    )
    if raw_permissions != canonical_permissions:
        raise PTSchemaValidationError(
            "manifest_meta.permissions must be normalized, de-duplicated, "
            f"and deterministically sorted; path={path}"
        )

    permission_dim = _strict_dimension(
        payload, "manifest_permission_dim", path=path
    )

    token_ids = _require_vector(
        payload,
        "manifest_permission_token_ids",
        path=path,
    )
    permission_ids = _require_vector(
        payload,
        "manifest_permission_ids",
        path=path,
    )
    if token_ids.dtype != torch.long:
        raise PTSchemaValidationError(
            "manifest_permission_token_ids must use torch.long, "
            f"got {token_ids.dtype}; path={path}"
        )
    if permission_ids.dtype != torch.long:
        raise PTSchemaValidationError(
            "manifest_permission_ids must use torch.long, "
            f"got {permission_ids.dtype}; path={path}"
        )
    if int(token_ids.numel()) != len(canonical_permissions):
        raise PTSchemaValidationError(
            "manifest_permission_token_ids length must equal canonical "
            "manifest_meta.permissions length: "
            f"actual={token_ids.numel()} expected={len(canonical_permissions)}; "
            f"path={path}"
        )
    if token_ids.numel():
        token_min = int(token_ids.min().item())
        token_max = int(token_ids.max().item())
        if token_min < 0 or token_max > permission_dim:
            raise PTSchemaValidationError(
                "manifest_permission_token_ids must be within "
                f"[0,{permission_dim}], got min={token_min} max={token_max}; "
                f"path={path}"
            )
    if permission_ids.numel():
        ids_min = int(permission_ids.min().item())
        ids_max = int(permission_ids.max().item())
        if ids_min < 1 or ids_max > permission_dim:
            raise PTSchemaValidationError(
                f"manifest_permission_ids must be within [1,{permission_dim}], "
                f"got min={ids_min} max={ids_max}; path={path}"
            )
        if permission_ids.numel() > 1 and bool(
            (permission_ids[1:] <= permission_ids[:-1]).any().item()
        ):
            raise PTSchemaValidationError(
                "manifest_permission_ids must be strictly increasing and unique; "
                f"path={path}"
            )

    nonzero_unique = token_ids[token_ids > 0].unique(sorted=True)
    if not torch.equal(nonzero_unique, permission_ids):
        raise PTSchemaValidationError(
            "non-zero manifest_permission_token_ids must exactly match "
            f"manifest_permission_ids; path={path}"
        )
    nonzero = token_ids[token_ids > 0]
    if int(nonzero.unique().numel()) != int(nonzero.numel()):
        raise PTSchemaValidationError(
            "non-zero manifest_permission_token_ids must be unique; "
            f"path={path}"
        )

    manifest_x = _require_tensor(payload, "manifest_x", path=path)
    if manifest_x.ndim != 1 or int(manifest_x.numel()) < permission_dim:
        raise PTSchemaValidationError(
            "manifest_x must be a vector containing the full permission bit "
            f"segment of length {permission_dim}, got shape={tuple(manifest_x.shape)}; "
            f"path={path}"
        )
    permission_bits = manifest_x[:permission_dim]
    if permission_bits.numel() and (
        not permission_bits.dtype.is_floating_point
        or not bool(torch.isfinite(permission_bits).all().item())
        or bool(
            ((permission_bits != 0.0) & (permission_bits != 1.0)).any().item()
        )
    ):
        raise PTSchemaValidationError(
            "manifest_x permission segment must contain finite binary float "
            f"values; path={path}"
        )
    active_ids = (
        torch.nonzero(permission_bits > 0.5, as_tuple=False).view(-1).long() + 1
    )
    if not torch.equal(active_ids, permission_ids):
        raise PTSchemaValidationError(
            "manifest_x active permission bits must exactly match "
            f"manifest_permission_ids; path={path}"
        )


def validate_current_pt_payload(
    payload: Any,
    path: str | Path,
    *,
    expected_sid: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the complete current PT payload without legacy repair."""

    if not isinstance(payload, dict):
        raise PTSchemaValidationError(
            f"PT must be a schema-{PT_SCHEMA_VERSION} top-level mapping; path={path}"
        )
    retired = [
        field for field in RETIRED_PT_TOP_LEVEL_FIELDS if field in payload
    ]
    if retired:
        raise PTSchemaValidationError(
            f"PT schema-{PT_SCHEMA_VERSION} contains retired top-level fields "
            f"{retired}; migrate the PT pool before training; path={path}"
        )
    missing = [
        field for field in CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS if field not in payload
    ]
    if missing:
        raise PTSchemaValidationError(
            f"PT schema-{PT_SCHEMA_VERSION} is missing top-level fields {missing}; "
            f"path={path}"
        )

    direct_meta = payload.get("direct_build_meta")
    if not isinstance(direct_meta, Mapping):
        raise PTSchemaValidationError(
            f"direct_build_meta must be a mapping; path={path}"
        )
    version = direct_meta.get("pt_schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PT_SCHEMA_VERSION
    ):
        raise PTSchemaValidationError(
            "PT schema version does not match required current version: "
            f"expected={PT_SCHEMA_VERSION} actual={version!r}; path={path}"
        )
    if direct_meta.get("schema_version") != OBSERVABLE_SCHEMA_VERSION:
        raise PTSchemaValidationError(
            "direct_build_meta.schema_version mismatch: "
            f"expected={OBSERVABLE_SCHEMA_VERSION!r} "
            f"actual={direct_meta.get('schema_version')!r}; path={path}"
        )
    if not str(direct_meta.get("build_fingerprint") or "").strip():
        raise PTSchemaValidationError(
            f"PT is missing direct build fingerprint; path={path}"
        )

    sid = str(expected_sid or Path(path).stem).strip().lower()
    direct_sid = str(direct_meta.get("sha256") or "").strip().lower()
    if not sid or direct_sid != sid:
        raise PTSchemaValidationError(
            "PT filename/direct_build_meta.sha256 identity mismatch: "
            f"expected={sid!r} actual={direct_sid!r}; path={path}"
        )
    manifest_meta = payload.get("manifest_meta")
    if not isinstance(manifest_meta, Mapping):
        raise PTSchemaValidationError(
            f"manifest_meta must be a mapping; path={path}"
        )
    manifest_sid = str(manifest_meta.get("sha256") or "").strip().lower()
    if manifest_sid != sid:
        raise PTSchemaValidationError(
            "PT filename/manifest_meta.sha256 identity mismatch: "
            f"expected={sid!r} actual={manifest_sid!r}; path={path}"
        )

    observable = payload.get("observable_metadata")
    if not isinstance(observable, Mapping):
        raise PTSchemaValidationError(
            f"observable_metadata must be a mapping; path={path}"
        )
    missing_observable = [
        field for field in OBSERVABLE_REQUIRED_FIELDS if field not in observable
    ]
    if (
        observable.get("schema_version") != OBSERVABLE_SCHEMA_VERSION
        or missing_observable
    ):
        raise PTSchemaValidationError(
            "observable_metadata schema is incomplete: "
            f"schema_version={observable.get('schema_version')!r} "
            f"missing={missing_observable}; path={path}"
        )
    validate_observable_metadata(observable, path=path)

    dex_list = validate_current_dex_list(payload.get("dex_list"), path=path)
    validate_manifest_payload_contract(payload, path=path)
    validate_manifest_permission_contract(payload, path=path)
    return dex_list, [payload, *dex_list]
