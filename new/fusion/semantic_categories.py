from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fusion.manifest_features import DEFAULT_CATEGORIES


SEMANTIC_CATEGORIES = tuple(DEFAULT_CATEGORIES)
SEMANTIC_CATEGORY_DIM = len(SEMANTIC_CATEGORIES)
CATEGORY_TO_INDEX = {name: idx for idx, name in enumerate(SEMANTIC_CATEGORIES)}

EXTRACTOR_API_CATEGORY_NAMES = (
    "other",
    "telephony",
    "sms",
    "location",
    "contacts_content",
    "camera_media",
    "network",
    "runtime_exec",
    "reflection",
    "dynamic_loading",
    "file_io",
    "package_info",
    "crypto",
    "webview",
    "system_settings",
    "account",
)

# Must match extract/extract_graph_api.py::API_CATEGORY_NAMES:
#   0 other, 1 telephony, 2 sms, 3 location, 4 contacts_content,
#   5 camera_media, 6 network, 7 runtime_exec, 8 reflection,
#   9 dynamic_loading, 10 file_io, 11 package_info, 12 crypto,
#   13 webview, 14 system_settings, 15 account.
# Keep id 0 as unknown/background. If extractor taxonomy changes, this table
# and its regression test must change together.
DEFAULT_API_TYPE_ID_TO_CATEGORY: dict[int, str] = {
    1: "telephony",
    2: "sms",
    3: "location",
    4: "contacts",
    5: "camera_media",
    6: "network",
    7: "dynamic_loading",
    8: "dynamic_loading",
    9: "dynamic_loading",
    10: "storage",
    11: "component_exposure",
    12: "crypto",
    13: "network",
    14: "system_settings",
    15: "contacts",
}


def validate_api_type_mapping(
    mapping: Mapping[int, str] | None = None,
    api_category_names: Sequence[str] | None = None,
    target_categories: Sequence[str] | None = None,
) -> None:
    """Validate `DEFAULT_API_TYPE_ID_TO_CATEGORY` against the extractor taxonomy.

    Raises ``ValueError`` with a precise diff when:

    * a mapping key falls outside the extractor's ``API_CATEGORY_NAMES``
      range (id 0 is reserved for ``other`` / unknown);
    * a mapping value is not in the shared 12-D taxonomy
      (``SEMANTIC_CATEGORIES`` / ``DEFAULT_CATEGORIES``).

    This is a defensive runtime check meant to catch silent drift between
    ``extract/extract_graph_api.py::API_CATEGORY_NAMES`` and this module's
    mapping table. It does NOT validate semantic correctness of each
    individual mapping (e.g. that id=10 should map to ``storage``) — that
    remains the author's responsibility.
    """
    if mapping is None:
        mapping = DEFAULT_API_TYPE_ID_TO_CATEGORY
    if api_category_names is None:
        # Training/inference deployments may omit the extractor package. The
        # model still needs to run, so fall back to the frozen taxonomy used
        # when tri-modal .pt files were generated.
        try:
            from extract.extract_graph_api import API_CATEGORY_NAMES as _names
            api_category_names = tuple(_names)
        except ModuleNotFoundError:
            api_category_names = EXTRACTOR_API_CATEGORY_NAMES
    if target_categories is None:
        target_categories = SEMANTIC_CATEGORIES

    target_set = set(target_categories)
    n_names = len(api_category_names)
    if n_names < 2:
        raise ValueError(
            f"api_category_names must contain at least the reserved 'other' "
            f"slot plus one real category; got {list(api_category_names)!r}"
        )

    def _key_is_valid(k: Any) -> bool:
        # bool is a subclass of int in Python; reject explicitly to avoid
        # `True`/`False` keys silently passing as id=1/id=0.
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        return 1 <= k < n_names

    bad_keys_raw = [k for k in mapping if not _key_is_valid(k)]
    # Stringify before sorting so mixed-type keys (e.g. an accidental str
    # key in the mapping) don't blow up sorted() with a TypeError.
    bad_keys = sorted(repr(k) for k in bad_keys_raw)
    bad_values = sorted({v for v in mapping.values() if v not in target_set})

    if not bad_keys and not bad_values:
        return

    parts: list[str] = []
    if bad_keys:
        parts.append(
            f"keys outside extractor range [1, {n_names - 1}]: {bad_keys}"
        )
    if bad_values:
        parts.append(
            f"values not in 12-D taxonomy {sorted(target_set)}: {bad_values}"
        )
    raise ValueError(
        "DEFAULT_API_TYPE_ID_TO_CATEGORY is inconsistent with extractor / "
        "shared taxonomy: "
        + "; ".join(parts)
        + ". Update fusion.semantic_categories.DEFAULT_API_TYPE_ID_TO_CATEGORY "
        "together with extract.extract_graph_api.API_CATEGORY_NAMES and "
        "fusion.manifest_features.DEFAULT_CATEGORIES."
    )


def sanitize_semantic_counts(value: Any, *, require_exact: bool = False) -> torch.Tensor:
    """Return a clean 12-D semantic count vector.

    `require_exact=True` is used for graph counts loaded from existing .pt
    files. If a source vector is not already in the shared 12-D taxonomy, it is
    treated as unavailable instead of being silently trimmed or padded.
    """
    if isinstance(value, torch.Tensor):
        out = value.detach().float().view(-1)
    elif value is None:
        out = torch.empty((0,), dtype=torch.float32)
    else:
        out = torch.as_tensor(value, dtype=torch.float32).view(-1)

    if require_exact and out.numel() not in {0, SEMANTIC_CATEGORY_DIM}:
        out = torch.empty((0,), dtype=torch.float32)

    if out.numel() < SEMANTIC_CATEGORY_DIM:
        pad = torch.zeros((SEMANTIC_CATEGORY_DIM - out.numel(),), dtype=torch.float32)
        out = torch.cat([out, pad], dim=0)
    elif out.numel() > SEMANTIC_CATEGORY_DIM:
        out = out[:SEMANTIC_CATEGORY_DIM]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# Cached vectorized lookup: type_id → category_index for the default mapping.
# Rebuilt lazily on first use; the default mapping is frozen after startup.
_DEFAULT_TYPE_ID_TABLE: torch.Tensor | None = None


def _build_type_id_table(mapping: Mapping[int, str]) -> torch.Tensor:
    """Build a (max_id+1,)-shaped tensor mapping type_id → category_index."""
    max_id = max(mapping.keys()) if mapping else -1
    if max_id < 0:
        return torch.empty((0,), dtype=torch.long)
    table = torch.full((max_id + 1,), -1, dtype=torch.long)
    for type_id, category in mapping.items():
        cat_idx = CATEGORY_TO_INDEX.get(category or "")
        if cat_idx is not None:
            table[type_id] = cat_idx
    return table


def api_semantic_counts_from_type_ids(
    api_type_ids: torch.Tensor | None,
    mapping: Mapping[int, str] | None = None,
) -> torch.Tensor:
    counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    if not isinstance(api_type_ids, torch.Tensor) or api_type_ids.numel() == 0:
        return counts

    flat = api_type_ids.detach().long().view(-1).cpu()

    # Use cached default table for the common case.
    if mapping is None or mapping is DEFAULT_API_TYPE_ID_TO_CATEGORY:
        global _DEFAULT_TYPE_ID_TABLE
        if _DEFAULT_TYPE_ID_TABLE is None:
            _DEFAULT_TYPE_ID_TABLE = _build_type_id_table(DEFAULT_API_TYPE_ID_TO_CATEGORY)
        table = _DEFAULT_TYPE_ID_TABLE
    else:
        table = _build_type_id_table(mapping)

    if table.numel() == 0:
        return counts

    valid = (flat >= 0) & (flat < table.size(0))
    if not valid.any():
        return counts
    cat_idx = table[flat[valid].clamp(0, table.size(0) - 1)]
    valid_cat = cat_idx >= 0
    if valid_cat.any():
        counts.scatter_add_(
            0,
            cat_idx[valid_cat],
            torch.ones(int(valid_cat.sum()), dtype=torch.float32),
        )
    return counts


def graph_semantic_counts_from_method_api_edges(
    api_type_ids: torch.Tensor | None,
    method_api_edge_index: torch.Tensor | None,
    *,
    mapping: Mapping[int, str] | None = None,
) -> torch.Tensor:
    """Aggregate API semantic categories carried by graph-aligned methods.

    The graph branch is structural, so its semantic category distribution is
    derived only from API events that are anchored to a graph method through
    `method_api_edge_index`. This gives Graph-Manifest consistency a real
    structural-context basis instead of reusing the full API histogram.
    """
    counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    if (
        not isinstance(api_type_ids, torch.Tensor)
        or not isinstance(method_api_edge_index, torch.Tensor)
        or method_api_edge_index.ndim != 2
        or method_api_edge_index.size(0) != 2
        or api_type_ids.numel() == 0
        or method_api_edge_index.numel() == 0
    ):
        return counts

    api_idx = method_api_edge_index[1].detach().long().view(-1).cpu()
    valid = (api_idx >= 0) & (api_idx < int(api_type_ids.numel()))
    if not valid.any():
        return counts
    aligned_types = api_type_ids.detach().long().view(-1).cpu()[api_idx[valid]]
    return api_semantic_counts_from_type_ids(aligned_types, mapping=mapping)
