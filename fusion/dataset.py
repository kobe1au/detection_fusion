from __future__ import annotations

from contextlib import contextmanager
import hashlib
import logging
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from fusion.perturbations import (
    EVAL_PERTURB_TYPES,
    apply_perturbation,
    sample_training_perturbation,
)
from fusion.semantic_categories import (
    SEMANTIC_CATEGORY_DIM,
    api_semantic_counts_from_type_ids,
    graph_semantic_counts_from_method_api_edges,
    sanitize_semantic_counts,
)
from fusion.quality import (
    OBSERVABLE_ERROR_FIELDS,
    OBSERVABLE_NUMERIC_FIELDS,
    OBSERVABLE_REQUIRED_FIELDS,
    OBSERVABLE_SCHEMA_VERSION,
    OBSERVABLE_SIGNAL_FIELDS,
    compute_api_quality,
    compute_graph_quality,
    compute_align_quality,
    compute_manifest_quality,
    graph_structure_stats,
    refresh_observable_signals,
)
from fusion.utils import scalar_float

logger = logging.getLogger(__name__)


VALID_GRAPH_SEMANTIC_SOURCES = ("alignment", "full_api", "zero")

class FatalDatasetConfigError(RuntimeError):
    """Configuration/data schema error that must not be converted to a dummy sample."""


def build_isolation_groups(
    sample_sids: list[str],
    packages: dict[str, str],
    families: dict[str, str],
) -> list[str]:
    """Build connected groups so neither package nor family crosses a split."""
    ignored_packages = {"", "nan", "none", "null"}
    ignored_families = ignored_packages | {"benign", "goodware"}
    parent = {sid: sid for sid in sample_sids}

    def find(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for values, ignored in ((packages, ignored_packages), (families, ignored_families)):
        first_by_value: dict[str, str] = {}
        for sid in sample_sids:
            value = str(values.get(sid, "") or "").strip().lower()
            if value in ignored:
                continue
            if value in first_by_value:
                union(sid, first_by_value[value])
            else:
                first_by_value[value] = sid

    return [f"isolation:{find(sid)}" for sid in sample_sids]


def _precompute_category_maps_from_vocab(vocab_path: str | Path) -> dict | None:
    """Pre-compute Manifest semantic category maps from vocab for old-PT fallback.

    Old tri-modal .pt files do not contain ``manifest_permission_category_map``
    or ``manifest_intent_category_map``.  Those maps are a pure function of the
    Manifest vocab term list and the keyword→category table, so they can be
    reconstructed at dataset init time without re-extracting APKs.
    """
    from fusion.manifest_features import (
        load_manifest_vocab,
        category_counts_from_strings,
        DEFAULT_CATEGORIES,
    )

    try:
        vocab = load_manifest_vocab(vocab_path, require_train_metadata=False, allow_empty=True)
    except Exception as exc:
        logger.warning("Cannot load manifest vocab for old-PT map fallback (%s): %s", vocab_path, exc)
        return None

    permission_vocab = list(vocab.get("permission_vocab") or [])
    intent_vocab = list(vocab.get("intent_vocab") or [])
    feature_vocab = list(vocab.get("feature_vocab") or [])
    categories = list(vocab.get("categories") or DEFAULT_CATEGORIES)

    perm_dim = len(permission_vocab)
    intent_dim = len(intent_vocab)
    feat_dim = len(feature_vocab)

    if perm_dim + intent_dim + feat_dim + SEMANTIC_CATEGORY_DIM + 11 == 0:
        return None

    perm_map = (
        torch.stack(
            [category_counts_from_strings([token], categories) for token in permission_vocab],
            dim=0,
        ).float()
        if permission_vocab
        else torch.zeros((0, len(categories)), dtype=torch.float32)
    )

    intent_map = (
        torch.stack(
            [category_counts_from_strings([token], categories) for token in intent_vocab],
            dim=0,
        ).float()
        if intent_vocab
        else torch.zeros((0, len(categories)), dtype=torch.float32)
    )

    return {
        "perm_map": perm_map,
        "intent_map": intent_map,
        "perm_dim": perm_dim,
        "intent_dim": intent_dim,
        "feat_dim": feat_dim,
    }


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


@contextmanager
def _temporary_random_seed(seed: int):
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)


def _as_float_tensor(value, length: int, default: float = 0.0) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().float().view(-1)
    elif value is None:
        out = torch.empty((0,), dtype=torch.float32)
    else:
        out = torch.as_tensor(value, dtype=torch.float32).view(-1)
    if out.numel() < length:
        pad = torch.full((length - out.numel(),), float(default), dtype=torch.float32)
        out = torch.cat([out, pad], dim=0)
    elif out.numel() > length:
        out = out[:length]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _flat_numel(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().view(-1).numel())
    if value is None:
        return 0
    try:
        return int(torch.as_tensor(value).view(-1).numel())
    except (TypeError, ValueError):
        return 0


def _as_long_tensor(value, length: int | None = None, fill_value: int = 0) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().long().view(-1)
    elif value is None:
        out = torch.empty((0,), dtype=torch.long)
    else:
        out = torch.as_tensor(value, dtype=torch.long).view(-1)
    if length is not None:
        if out.numel() < length:
            out = torch.cat([out, torch.full((length - out.numel(),), int(fill_value), dtype=torch.long)])
        elif out.numel() > length:
            out = out[:length]
    return out


def _as_category_map(value, rows: int, columns: int = SEMANTIC_CATEGORY_DIM) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.ndim == 2 and value.size(1) == columns:
        out = value.detach().float()
    else:
        out = torch.zeros((0, columns), dtype=torch.float32)
    if out.size(0) < rows:
        out = torch.cat([out, torch.zeros((rows - out.size(0), columns), dtype=torch.float32)], dim=0)
    elif out.size(0) > rows:
        out = out[:rows]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _first_present(sources: list[dict[str, Any]], key: str):
    for src in sources:
        if isinstance(src, dict) and key in src and src[key] is not None:
            return src[key]
    return None


def _first_int(sources: list[dict[str, Any]], key: str, default: int) -> int:
    value = _first_present(sources, key)
    if value is None:
        return int(default)
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().view(-1)[0].item()
        elif isinstance(value, (list, tuple)):
            value = value[0]
        return max(0, int(value))
    except (IndexError, TypeError, ValueError):
        return int(default)


def _normalize_loaded_pt(raw) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(raw, list):
        dex_list = [d for d in raw if isinstance(d, dict)]
        return dex_list, dex_list
    if isinstance(raw, dict):
        if isinstance(raw.get("dex_list"), list):
            dex_list = [d for d in raw["dex_list"] if isinstance(d, dict)]
            return dex_list, [raw] + dex_list
        if isinstance(raw.get("dexes"), list):
            dex_list = [d for d in raw["dexes"] if isinstance(d, dict)]
            return dex_list, [raw] + dex_list
        return [raw], [raw]
    return [], []


def _merged_observable_metadata(sources: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    # Sources are [top_level, dex...]. Let the top-level sample metadata win.
    for source in reversed(sources):
        value = source.get("observable_metadata") if isinstance(source, dict) else None
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _sum_nested_meta(dex_list: list[dict[str, Any]], section: str, key: str) -> int:
    total = 0
    for dex in dex_list:
        meta = dex.get("meta", {}) if isinstance(dex, dict) else {}
        nested = meta.get(section, {}) if isinstance(meta, dict) else {}
        try:
            total += max(0, int(nested.get(key, 0))) if isinstance(nested, dict) else 0
        except (TypeError, ValueError):
            pass
    return total


def _fallback_observable_metadata(
    data: dict[str, Any],
    sources: list[dict[str, Any]],
    dex_list: list[dict[str, Any]],
    saved: dict[str, Any],
) -> dict[str, Any]:
    out = dict(saved)
    api_ids = data.get("api_ids")
    api_types = data.get("api_type_ids")
    kept = int(api_ids.numel()) if isinstance(api_ids, torch.Tensor) else 0
    raw = int(out.get("api_event_count_raw", 0) or 0)
    if raw <= 0:
        raw = _sum_nested_meta(dex_list, "api", "num_api_events_raw") or kept
    known = int((api_types.long().view(-1) > 0).sum().item()) if isinstance(api_types, torch.Tensor) else 0

    x = data.get("x")
    real_nodes = int(data.get("real_num_nodes", 0))
    if real_nodes <= 0 and isinstance(x, torch.Tensor) and x.ndim == 2:
        real_nodes = int(x.size(0))
    edge = data.get("edge_index")
    edge_count = int(edge.size(1)) if isinstance(edge, torch.Tensor) and edge.ndim == 2 else 0
    isolated, lcc = graph_structure_stats(edge, real_nodes)

    direct_meta = _first_present(sources, "direct_build_meta")
    direct_meta = direct_meta if isinstance(direct_meta, dict) else {}
    dex_file_count = int(out.get("dex_file_count", direct_meta.get("dex_file_count", len(dex_list))) or 0)
    success_ratio = scalar_float(direct_meta.get("dex_success_ratio", 1.0), 1.0)

    manifest_meta = data.get("manifest_meta")
    manifest_meta = manifest_meta if isinstance(manifest_meta, dict) else {}
    manifest_error = str(out.get("manifest_parse_error", manifest_meta.get("parse_error", "")) or "")
    coverage_values = [
        scalar_float(manifest_meta.get(key), 1.0)
        for key in (
            "permission_vocab_coverage",
            "intent_vocab_coverage",
            "feature_vocab_coverage",
        )
        if key in manifest_meta
    ]
    manifest_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 1.0

    fallback = {
        "api_parse_ok": bool(dex_list),
        "api_parse_error": "",
        "api_event_count_raw": raw,
        "api_event_count_kept": kept,
        "api_truncated": kept < raw,
        "api_truncation_ratio": (1.0 - kept / raw) if raw > 0 else 0.0,
        "api_known_type_count": known,
        "api_unknown_type_count": max(0, kept - known),
        "dex_parse_ok": bool(dex_list),
        "dex_file_count": max(dex_file_count, len(dex_list)),
        "dex_parse_success_ratio": max(0.0, min(1.0, success_ratio)),
        "class_count": int(out.get("class_count", 0) or 0),
        "method_count": int(out.get("method_count", _sum_nested_meta(dex_list, "graph", "num_methods_all")) or 0),
        "graph_parse_ok": bool(dex_list),
        "graph_parse_error": "",
        "graph_timeout": False,
        "graph_node_count_raw": real_nodes,
        "graph_edge_count_raw": edge_count,
        "graph_isolated_node_count": isolated,
        "graph_largest_component_ratio": lcc,
        "manifest_parse_ok": not bool(manifest_error),
        "manifest_parse_error": manifest_error,
        "manifest_has_content": bool(
            data.get("manifest_x") is not None
            and scalar_float(data.get("q_manifest"), 0.0) > 0.0
        ),
        "manifest_vocab_coverage": manifest_coverage,
        "manifest_permission_count": int(data.get("manifest_permission_ids").numel()) if isinstance(data.get("manifest_permission_ids"), torch.Tensor) else 0,
        "manifest_component_count": int(manifest_meta.get("component_count", 0) or 0),
        "manifest_intent_count": int(data.get("manifest_intent_ids").numel()) if isinstance(data.get("manifest_intent_ids"), torch.Tensor) else 0,
        "schema_version": str(out.get("schema_version") or "legacy-fallback"),
    }
    fallback.update(out)
    return fallback


def apply_dex_success_ratio(data: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    """Penalize code-side quality when only part of a multi-DEX APK parsed."""
    direct_meta = _first_present(sources, "direct_build_meta")
    if not isinstance(direct_meta, dict):
        return
    success_ratio = float(direct_meta.get("dex_success_ratio", 1.0))
    if not math.isfinite(success_ratio):
        success_ratio = 0.0
    success_ratio = max(0.0, min(1.0, success_ratio))
    if success_ratio >= 1.0:
        return
    data["q_api"] *= success_ratio
    data["q_graph"] *= success_ratio
    data["q_align"] = compute_align_quality(
        data["q_api"],
        data["q_graph"],
        data["method_api_edge_index"],
        int(data["real_num_nodes"]),
        int(data["api_ids"].numel()),
    )


class RobustTriModalDataset(Dataset):
    """Standalone API + Graph + Manifest dataset for robust fusion."""

    def __init__(
        self,
        pt_dir: str,
        csv_path: str,
        is_train: bool = True,
        robust_aug: bool = False,
        perturb_prob: float = 0.5,
        perturb_strengths: list[float] | tuple[float, ...] | None = None,
        eval_perturb_type: str | None = None,
        eval_perturb_strength: float = 0.0,
        max_api_events_per_sample: int | None = None,
        manifest_dim: int = 256,
        manifest_category_dim: int = 12,
        manifest_stats_dim: int = 11,
        manifest_permission_dim: int = 128,
        manifest_intent_dim: int = 64,
        manifest_feature_dim: int = 32,
        drop_graph_behavior_hints: bool = False,
        graph_semantic_source: str = "alignment",
        num_classes: int = 2,
        label_map: dict | None = None,
        strict_split_integrity: bool = True,
        allow_pt_superset: bool = False,
        require_manifest_semantic_maps: bool = False,
        allow_legacy_pt_compat: bool = False,
        min_pt_schema_version: int = 0,
        manifest_vocab_path: str = "",
        strict_observable_schema: bool = False,
        **_unused,
    ):
        # Pre-compute semantic category maps from the Manifest vocab so that
        # old tri-modal .pt files (which lack the maps) can still benefit from
        # coherent semantic-count updates during perturbation.
        self._vocab_maps: dict | None = None
        if manifest_vocab_path:
            self._vocab_maps = _precompute_category_maps_from_vocab(manifest_vocab_path)
        else:
            self._vocab_maps = None
        if eval_perturb_type not in EVAL_PERTURB_TYPES:
            raise ValueError(f"Unsupported eval_perturb_type: {eval_perturb_type}")
        self.pt_dir = Path(pt_dir)
        self.is_train = bool(is_train)
        self.robust_aug = bool(robust_aug)
        self.perturb_prob = float(perturb_prob)
        self.perturb_strengths = list(perturb_strengths or [0.1, 0.3, 0.5])
        self.eval_perturb_type = eval_perturb_type
        self.eval_perturb_strength = float(eval_perturb_strength)
        if not math.isfinite(self.perturb_prob) or not 0.0 <= self.perturb_prob <= 1.0:
            raise ValueError(f"perturb_prob must be within [0, 1], got {self.perturb_prob}")
        if not self.perturb_strengths or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in self.perturb_strengths
        ):
            raise ValueError(f"perturb_strengths must be a non-empty list within [0, 1], got {self.perturb_strengths}")
        if not math.isfinite(self.eval_perturb_strength) or not 0.0 <= self.eval_perturb_strength <= 1.0:
            raise ValueError(
                f"eval_perturb_strength must be within [0, 1], got {self.eval_perturb_strength}"
            )
        self.max_api_events_per_sample = (
            int(max_api_events_per_sample) if max_api_events_per_sample is not None else None
        )
        self.manifest_dim = int(manifest_dim)
        if int(manifest_category_dim) != SEMANTIC_CATEGORY_DIM:
            raise ValueError(
                f"Robust semantic category space must be {SEMANTIC_CATEGORY_DIM}-D; "
                f"got manifest_category_dim={manifest_category_dim}"
            )
        self.manifest_category_dim = SEMANTIC_CATEGORY_DIM
        self.manifest_stats_dim = int(manifest_stats_dim)
        self.manifest_permission_dim = int(manifest_permission_dim)
        self.manifest_intent_dim = int(manifest_intent_dim)
        self.manifest_feature_dim = int(manifest_feature_dim)
        self.drop_graph_behavior_hints = bool(drop_graph_behavior_hints)
        src = str(graph_semantic_source or "alignment").lower()
        if src not in VALID_GRAPH_SEMANTIC_SOURCES:
            raise ValueError(
                f"Unsupported graph_semantic_source={graph_semantic_source!r}; "
                f"must be one of {VALID_GRAPH_SEMANTIC_SOURCES}"
            )
        self.graph_semantic_source = src
        self.strict_split_integrity = bool(strict_split_integrity)
        self.allow_pt_superset = bool(allow_pt_superset)
        self.require_manifest_semantic_maps = bool(require_manifest_semantic_maps)
        self.allow_legacy_pt_compat = bool(allow_legacy_pt_compat)
        self.min_pt_schema_version = max(0, int(min_pt_schema_version))
        self.strict_observable_schema = bool(strict_observable_schema)
        self._warned_legacy_schema = False
        self._warned_manifest_map_fallback = False
        self._warned_observable_schema = False

        df = pd.read_csv(csv_path)
        id_col = next((c for c in ["id", "ID", "Id", "sha256"] if c in df.columns), None)
        if id_col is None:
            raise ValueError("CSV must contain id or sha256")
        if "label" not in df.columns:
            raise ValueError("CSV must contain label")
        year_col = next((c for c in ["year", "Year", "vt_year", "dex_year"] if c in df.columns), None)
        package_col = next(
            (c for c in ["pkg_name", "package_name", "package"] if c in df.columns),
            None,
        )
        family_col = next(
            (c for c in ["family", "malware_family", "family_name", "avclass_family"] if c in df.columns),
            None,
        )

        sid_series = df[id_col].astype(str).str.strip().str.lower()
        duplicate_csv_ids = sorted(sid_series[sid_series.duplicated(keep=False)].unique().tolist())
        if duplicate_csv_ids:
            raise ValueError(
                f"CSV {csv_path} contains duplicate sample IDs; "
                f"count={len(duplicate_csv_ids)} examples={duplicate_csv_ids[:10]}"
            )
        raw_labels = df["label"].astype(str).str.strip()
        if label_map:
            normalized_map = {str(k).strip(): int(v) for k, v in label_map.items()}
            mapped = raw_labels.map(normalized_map)
            if mapped.isna().any():
                bad = df.loc[mapped.isna(), [id_col, "label"]].head(10).to_dict("records")
                raise ValueError(
                    f"CSV {csv_path} contains labels not covered by data.label_map; "
                    f"examples={bad}"
                )
            label_series = mapped.astype(int)
        else:
            label_series = pd.to_numeric(df["label"], errors="coerce")
            if label_series.isna().any():
                bad = df.loc[label_series.isna(), [id_col, "label"]].head(10).to_dict("records")
                raise ValueError(f"CSV {csv_path} contains non-integer labels; examples={bad}")
            label_series = label_series.astype(int)
        num_classes = int(num_classes)
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        invalid = ~label_series.between(0, num_classes - 1)
        if invalid.any():
            bad = df.loc[invalid, [id_col, "label"]].head(20).to_dict("records")
            counts = label_series.value_counts().sort_index().to_dict()
            raise ValueError(
                f"CSV {csv_path} contains labels outside [0, {num_classes - 1}] "
                f"for num_classes={num_classes}; label_counts={counts}; examples={bad}. "
                "Fix the CSV labels or set data.label_map in the config."
            )
        labels = dict(zip(sid_series, label_series))
        years = (
            dict(zip(sid_series, pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)))
            if year_col
            else {sid: 0 for sid in sid_series}
        )
        groups = (
            dict(
                zip(
                    sid_series,
                    df[package_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower(),
                )
            )
            if package_col
            else {sid: "" for sid in sid_series}
        )
        families = (
            dict(
                zip(
                    sid_series,
                    df[family_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower(),
                )
            )
            if family_col
            else {sid: "" for sid in sid_series}
        )

        pt_files = sorted(self.pt_dir.rglob("*.pt"))
        pt_by_sid: dict[str, Path] = {}
        duplicate_pt_ids: list[str] = []
        for pt_file in pt_files:
            sid = pt_file.stem.lower()
            if sid in pt_by_sid:
                duplicate_pt_ids.append(sid)
            else:
                pt_by_sid[sid] = pt_file
        if duplicate_pt_ids:
            raise ValueError(
                f"PT directory {self.pt_dir} contains duplicate filename stems; "
                f"count={len(set(duplicate_pt_ids))} examples={sorted(set(duplicate_pt_ids))[:10]}"
            )

        csv_ids = set(labels)
        pt_ids = set(pt_by_sid)
        csv_only = sorted(csv_ids - pt_ids)
        pt_only = sorted(pt_ids - csv_ids)
        rejected_pt_only = pt_only if not self.allow_pt_superset else []
        if csv_only or rejected_pt_only:
            message = (
                f"Split integrity mismatch for CSV={csv_path} PT={self.pt_dir}: "
                f"csv_only={len(csv_only)} examples={csv_only[:10]}; "
                f"pt_only={len(pt_only)} examples={pt_only[:10]}"
            )
            if self.strict_split_integrity:
                raise ValueError(message)
            logger.warning(message)

        self.samples: list[tuple[Path, int, str, int]] = []
        for sid in sorted(csv_ids & pt_ids):
            self.samples.append((pt_by_sid[sid], int(labels[sid]), sid, int(years.get(sid, 0))))
        if not self.samples:
            raise RuntimeError(f"No matching .pt samples found in {self.pt_dir} for {csv_path}")
        self.sample_sids = [sid for _, _, sid, _ in self.samples]
        self.sample_years = [year for _, _, _, year in self.samples]
        self.sample_groups = build_isolation_groups(self.sample_sids, groups, families)
        self.feature_dim = self._infer_feature_dim(default_dim=515)
        logger.info("Loaded %d robust tri-modal samples from %s", len(self.samples), self.pt_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def _infer_feature_dim(self, default_dim: int) -> int:
        for pt_file, _, _, _ in self.samples:
            try:
                raw = torch.load(pt_file, map_location="cpu", weights_only=False)
                dex_list, _ = _normalize_loaded_pt(raw)
                for dex in dex_list:
                    x = dex.get("call_x") if isinstance(dex, dict) else None
                    if isinstance(x, torch.Tensor) and x.ndim == 2 and x.size(1) > 0:
                        dim = int(x.size(1))
                        if self.drop_graph_behavior_hints and dim == 519:
                            return 515
                        return dim
            except Exception as exc:
                logger.warning("feature_dim inference failed for %s: %s", pt_file, exc)
        return int(default_dim)

    def _dummy(self, label: int, sid: str, year: int, reason: str, pt_path: Path | None = None) -> Data:
        data = Data(
            x=torch.zeros((1, self.feature_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor(label, dtype=torch.long),
        )
        data.sensitive_mask = torch.zeros((1,), dtype=torch.uint8)
        data.real_num_nodes = torch.tensor([0], dtype=torch.long)
        data.real_node_mask = torch.zeros((1,), dtype=torch.bool)
        data.api_ids = torch.empty((0,), dtype=torch.long)
        data.api_type_ids = torch.empty((0,), dtype=torch.long)
        data.api_sensitive_mask = torch.empty((0,), dtype=torch.float32)
        data.api_method_index = torch.empty((0,), dtype=torch.long)
        data.api_in_graph_mask = torch.empty((0,), dtype=torch.float32)
        data.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.api_semantic_category_counts = torch.zeros((self.manifest_category_dim,), dtype=torch.float32)
        data.graph_semantic_category_counts = torch.zeros((self.manifest_category_dim,), dtype=torch.float32)
        data.api_category_counts = data.api_semantic_category_counts.clone()
        data.graph_category_counts = data.graph_semantic_category_counts.clone()
        data.manifest_x = torch.zeros((1, self.manifest_dim), dtype=torch.float32)
        data.manifest_permission_ids = torch.empty((0,), dtype=torch.long)
        data.manifest_intent_ids = torch.empty((0,), dtype=torch.long)
        data.manifest_category_counts = torch.zeros((self.manifest_category_dim,), dtype=torch.float32)
        data.manifest_stats = torch.zeros((self.manifest_stats_dim,), dtype=torch.float32)
        data.q_api = torch.tensor([0.0], dtype=torch.float32)
        data.q_graph = torch.tensor([0.0], dtype=torch.float32)
        data.q_manifest = torch.tensor([0.0], dtype=torch.float32)
        data.q_align = torch.tensor([0.0], dtype=torch.float32)
        data.pert_api = torch.tensor([1.0], dtype=torch.float32)
        data.pert_graph = torch.tensor([1.0], dtype=torch.float32)
        data.pert_manifest = torch.tensor([1.0], dtype=torch.float32)
        for key in OBSERVABLE_NUMERIC_FIELDS:
            setattr(data, key, torch.tensor([0.0], dtype=torch.float32))
        for key in OBSERVABLE_SIGNAL_FIELDS:
            setattr(data, key, torch.tensor([0.0], dtype=torch.float32))
        for key in OBSERVABLE_ERROR_FIELDS:
            setattr(data, key, reason)
        data.schema_version = "dummy"
        data.sid = sid
        data.year = torch.tensor(int(year), dtype=torch.long)
        data.is_dummy = True
        data.fail_reason = reason
        data.fail_path = str(pt_path) if pt_path else ""
        return data

    def _sanitize_call_x(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32)
        x = x.float()
        if self.drop_graph_behavior_hints and x.size(1) == 519:
            x = x[:, :515]
        if x.size(1) > self.feature_dim:
            x = x[:, : self.feature_dim]
        elif x.size(1) < self.feature_dim:
            x = torch.cat([x, torch.zeros((x.size(0), self.feature_dim - x.size(1)), dtype=x.dtype)], dim=1)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _sanitize_edge_index(edge_index, num_nodes: int) -> torch.Tensor:
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = edge_index.long()
        if edge_index.numel() == 0 or num_nodes <= 0:
            return torch.empty((2, 0), dtype=torch.long)
        valid = (
            (edge_index[0] >= 0)
            & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0)
            & (edge_index[1] < num_nodes)
        )
        return edge_index[:, valid]

    @staticmethod
    def _sanitize_mask(mask, length: int, dtype=torch.float32) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            return torch.zeros((length,), dtype=dtype)
        out = mask.to(dtype=dtype).view(-1)
        if out.numel() < length:
            out = torch.cat([out, torch.zeros((length - out.numel(),), dtype=dtype)])
        elif out.numel() > length:
            out = out[:length]
        return out

    def _limit_api_events(self, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.max_api_events_per_sample is None:
            return parts
        n = int(parts["api_ids"].numel())
        if n <= self.max_api_events_per_sample:
            return parts
        limit = max(0, int(self.max_api_events_per_sample))
        keep = torch.arange(limit, device=parts["api_ids"].device)
        for key in ("api_ids", "api_type_ids", "api_sensitive_mask", "api_method_index", "api_in_graph_mask"):
            value = parts[key]
            parts[key] = value[keep.to(value.device)] if value.numel() > 0 else value[:0]
        edge = parts["method_api_edge_index"]
        if edge.numel() > 0:
            mapping = torch.full((n,), -1, dtype=torch.long, device=edge.device)
            keep_edge = keep.to(edge.device)
            mapping[keep_edge] = torch.arange(keep_edge.numel(), dtype=torch.long, device=edge.device)
            dst = edge[1].long()
            valid = (dst >= 0) & (dst < n) & (mapping[dst.clamp(0, max(n - 1, 0))] >= 0)
            edge = edge[:, valid].clone()
            if edge.numel() > 0:
                edge[1] = mapping[edge[1].long()]
            parts["method_api_edge_index"] = edge
        return parts

    @staticmethod
    def _api_semantic_category_counts(api_type_ids: torch.Tensor) -> torch.Tensor:
        return api_semantic_counts_from_type_ids(api_type_ids)


    def _process_dex(self, dex: dict[str, Any], node_offset: int, api_offset: int):
        x = self._sanitize_call_x(dex.get("call_x"))
        orig_size = int(x.size(0))
        if orig_size == 0:
            # Ghost node: keeps GNN message-passing alive for dex files with
            # zero call-graph nodes.  All-zero features carry no signal, but
            # the node still participates in readout attention — for samples
            # where EVERY dex is empty the graph embedding degenerates to a
            # uniform average of ghosts.
            x = torch.zeros((1, self.feature_dim), dtype=torch.float32)
        n = int(x.size(0))
        edge_index = self._sanitize_edge_index(dex.get("call_edge_index"), orig_size)
        if edge_index.numel() > 0:
            edge_index = edge_index + node_offset
        sensitive = self._sanitize_mask(dex.get("call_sensitive_mask"), n, dtype=torch.uint8)
        real_node_mask = torch.ones((n,), dtype=torch.bool)
        if orig_size == 0:
            real_node_mask.zero_()

        api_ids = _as_long_tensor(dex.get("api_ids")).clamp_min(0)
        num_api = int(api_ids.numel())
        api_type_ids = _as_long_tensor(dex.get("api_type_ids"), num_api, fill_value=0).clamp_min(0)
        api_sensitive = self._sanitize_mask(dex.get("api_sensitive_mask"), num_api, dtype=torch.float32).clamp(0.0, 1.0)
        api_method_index = _as_long_tensor(dex.get("api_method_index"), num_api, fill_value=-1)
        valid_method = (api_method_index >= 0) & (api_method_index < orig_size)
        api_method_index = torch.where(api_method_index >= 0, api_method_index + node_offset, api_method_index)
        api_method_index = torch.where(valid_method, api_method_index, torch.full_like(api_method_index, -1))
        api_in_graph = self._sanitize_mask(dex.get("api_in_graph_mask"), num_api, dtype=torch.float32).clamp(0.0, 1.0)

        method_api_edge_index = dex.get("method_api_edge_index")
        if isinstance(method_api_edge_index, torch.Tensor) and method_api_edge_index.ndim == 2 and method_api_edge_index.size(0) == 2:
            local_edge = method_api_edge_index.long()
            valid = (
                (local_edge[0] >= 0)
                & (local_edge[0] < orig_size)
                & (local_edge[1] >= 0)
                & (local_edge[1] < num_api)
            )
            method_api_edge_index = local_edge[:, valid]
            if method_api_edge_index.numel() > 0:
                method_api_edge_index = method_api_edge_index.clone()
                method_api_edge_index[0] += node_offset
                method_api_edge_index[1] += api_offset
        else:
            method_api_edge_index = torch.empty((2, 0), dtype=torch.long)

        parts = {
            "api_ids": api_ids,
            "api_type_ids": api_type_ids,
            "api_sensitive_mask": api_sensitive,
            "api_method_index": api_method_index,
            "api_in_graph_mask": api_in_graph,
            "method_api_edge_index": method_api_edge_index,
        }
        return {
            "x": x,
            "edge_index": edge_index,
            "sensitive_mask": sensitive,
            "real_node_mask": real_node_mask,
            "num_nodes": n,
            "real_nodes": orig_size,
            "num_api": int(parts["api_ids"].numel()),
            **parts,
        }

    def _aggregate_api_graph(self, dex_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        xs, edges, sens, real_masks = [], [], [], []
        api_ids, api_types, api_sensitive, api_methods, api_in_graph, method_edges = [], [], [], [], [], []
        node_offset = 0
        api_offset = 0
        total_real_nodes = 0
        for dex in dex_list:
            if not isinstance(dex, dict):
                continue
            part = self._process_dex(dex, node_offset, api_offset)
            xs.append(part["x"])
            edges.append(part["edge_index"])
            sens.append(part["sensitive_mask"])
            real_masks.append(part["real_node_mask"])
            api_ids.append(part["api_ids"])
            api_types.append(part["api_type_ids"])
            api_sensitive.append(part["api_sensitive_mask"])
            api_methods.append(part["api_method_index"])
            api_in_graph.append(part["api_in_graph_mask"])
            method_edges.append(part["method_api_edge_index"])
            node_offset += int(part["num_nodes"])
            api_offset += int(part["num_api"])
            total_real_nodes += int(part.get("real_nodes", int(part["num_nodes"])))
        if not xs:
            return None

        x = torch.cat(xs, dim=0)
        edge_index = torch.cat([e for e in edges if e.numel() > 0], dim=1) if any(e.numel() > 0 for e in edges) else torch.empty((2, 0), dtype=torch.long)
        sensitive_mask = torch.cat(sens, dim=0).to(torch.uint8)
        real_node_mask = torch.cat(real_masks, dim=0).bool()
        final_api_ids = torch.cat([v for v in api_ids if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_ids) else torch.empty((0,), dtype=torch.long)
        final_api_types = torch.cat([v for v in api_types if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_types) else torch.empty((0,), dtype=torch.long)
        final_api_sensitive = torch.cat([v for v in api_sensitive if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_sensitive) else torch.empty((0,), dtype=torch.float32)
        final_api_methods = torch.cat([v for v in api_methods if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_methods) else torch.empty((0,), dtype=torch.long)
        final_api_in_graph = torch.cat([v for v in api_in_graph if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_in_graph) else torch.empty((0,), dtype=torch.float32)
        final_method_edges = torch.cat([e for e in method_edges if e.numel() > 0], dim=1) if any(e.numel() > 0 for e in method_edges) else torch.empty((2, 0), dtype=torch.long)

        api_parts = self._limit_api_events({
            "api_ids": final_api_ids,
            "api_type_ids": final_api_types,
            "api_sensitive_mask": final_api_sensitive,
            "api_method_index": final_api_methods,
            "api_in_graph_mask": final_api_in_graph,
            "method_api_edge_index": final_method_edges,
        })
        final_api_ids = api_parts["api_ids"]
        final_api_types = api_parts["api_type_ids"]
        final_api_sensitive = api_parts["api_sensitive_mask"]
        final_api_methods = api_parts["api_method_index"]
        final_api_in_graph = api_parts["api_in_graph_mask"]
        final_method_edges = api_parts["method_api_edge_index"]

        q_api = compute_api_quality(final_api_ids, final_api_types, final_api_in_graph)
        q_graph = compute_graph_quality(edge_index, total_real_nodes, x, real_node_mask)
        if total_real_nodes <= 0:
            q_graph = 0.0  # all nodes are zero-feature ghosts — no real graph signal
        q_align = compute_align_quality(
            q_api,
            q_graph,
            final_method_edges,
            total_real_nodes,
            int(final_api_ids.numel()),
            int(x.size(0)),
        )
        api_semantic_counts = self._api_semantic_category_counts(final_api_types)
        if self.graph_semantic_source == "alignment":
            graph_semantic_counts = graph_semantic_counts_from_method_api_edges(
                final_api_types,
                final_method_edges,
            )
        elif self.graph_semantic_source == "full_api":
            graph_semantic_counts = api_semantic_counts.clone()
        else:  # "zero"
            graph_semantic_counts = torch.zeros(
                (SEMANTIC_CATEGORY_DIM,), dtype=torch.float32
            )
        return {
            "x": x,
            "edge_index": edge_index,
            "sensitive_mask": sensitive_mask,
            "real_num_nodes": total_real_nodes,
            "real_node_mask": real_node_mask,
            "api_ids": final_api_ids,
            "api_type_ids": final_api_types,
            "api_sensitive_mask": final_api_sensitive,
            "api_method_index": final_api_methods,
            "api_in_graph_mask": final_api_in_graph,
            "method_api_edge_index": final_method_edges,
            "api_semantic_category_counts": api_semantic_counts,
            "api_category_counts": api_semantic_counts,
            "graph_semantic_category_counts": graph_semantic_counts,
            "graph_category_counts": graph_semantic_counts,
            "q_api": q_api,
            "q_graph": q_graph,
            "q_align": q_align,
            "pert_api": 0.0,
            "pert_graph": 0.0,
            "api_aug_type": "none",
            "graph_aug_type": "none",
            "mask": torch.empty((x.size(0), 0), dtype=torch.float32),
        }

    def _manifest_payload(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
        manifest_x_raw = _first_present(sources, "manifest_x")
        has_manifest = manifest_x_raw is not None or _first_present(sources, "manifest_category_counts") is not None
        raw_manifest_dim = _flat_numel(manifest_x_raw)
        if raw_manifest_dim > self.manifest_dim:
            raise FatalDatasetConfigError(
                f"manifest_x dimension {raw_manifest_dim} exceeds configured manifest_dim={self.manifest_dim}; "
                "regenerate tri-modal .pt files or increase model.manifest_encoder.in_dim"
            )
        manifest_x = _as_float_tensor(manifest_x_raw, self.manifest_dim)
        manifest_counts = sanitize_semantic_counts(_first_present(sources, "manifest_category_counts"))
        manifest_component_counts = sanitize_semantic_counts(
            _first_present(sources, "manifest_component_category_counts")
        )
        graph_raw = _first_present(sources, "graph_semantic_category_counts")
        if graph_raw is None:
            graph_raw = _first_present(sources, "graph_category_counts")
        graph_counts = sanitize_semantic_counts(graph_raw, require_exact=True)
        manifest_stats = _as_float_tensor(_first_present(sources, "manifest_stats"), self.manifest_stats_dim)
        permission_dim = _first_int(sources, "manifest_permission_dim", self.manifest_permission_dim)
        intent_dim = _first_int(sources, "manifest_intent_dim", self.manifest_intent_dim)
        permission_map_raw = _first_present(sources, "manifest_permission_category_map")
        intent_map_raw = _first_present(sources, "manifest_intent_category_map")
        maps_available = (
            isinstance(permission_map_raw, torch.Tensor)
            and permission_map_raw.ndim == 2
            and permission_map_raw.shape == (permission_dim, SEMANTIC_CATEGORY_DIM)
            and isinstance(intent_map_raw, torch.Tensor)
            and intent_map_raw.ndim == 2
            and intent_map_raw.shape == (intent_dim, SEMANTIC_CATEGORY_DIM)
        )

        # ── old-PT fallback: reconstruct maps from vocab ──────────────────
        vocab_feature_dim: int | None = None
        if not maps_available and self._vocab_maps is not None:
            vm = self._vocab_maps
            # Old PTs do not store dims; use vocab-derived values.
            if _first_present(sources, "manifest_permission_dim") is None and vm["perm_dim"] > 0:
                permission_dim = vm["perm_dim"]
            if _first_present(sources, "manifest_intent_dim") is None and vm["intent_dim"] > 0:
                intent_dim = vm["intent_dim"]
            if _first_present(sources, "manifest_feature_dim") is None and vm["feat_dim"] > 0:
                vocab_feature_dim = vm["feat_dim"]
            permission_map_raw = vm["perm_map"]
            intent_map_raw = vm["intent_map"]
            maps_available = (
                isinstance(permission_map_raw, torch.Tensor)
                and permission_map_raw.ndim == 2
                and permission_map_raw.shape == (permission_dim, SEMANTIC_CATEGORY_DIM)
                and isinstance(intent_map_raw, torch.Tensor)
                and intent_map_raw.ndim == 2
                and intent_map_raw.shape == (intent_dim, SEMANTIC_CATEGORY_DIM)
            )
        meta = _first_present(sources, "manifest_meta")
        meta = meta if isinstance(meta, dict) else {}
        q_raw = _first_present(sources, "q_manifest")
        p_raw = _first_present(sources, "pert_manifest")
        if q_raw is None:
            q_manifest = compute_manifest_quality(
                manifest_x,
                manifest_counts,
                manifest_stats,
                meta,
            ) if has_manifest else 0.0
        else:
            q_manifest = float(torch.as_tensor(q_raw).float().view(-1)[0].item())
        if not math.isfinite(q_manifest):
            q_manifest = 0.0
        manifest_available = has_manifest and q_manifest > 0.0
        if self.require_manifest_semantic_maps and manifest_available and not maps_available:
            if not self.allow_legacy_pt_compat:
                raise FatalDatasetConfigError(
                    "PT is missing Manifest term-to-category maps required by the configured "
                    "consistency evidence/loss. Regenerate PT files with the current direct builder."
                )
            if not self._warned_manifest_map_fallback:
                logger.warning(
                    "Using legacy PT compatibility mode: Manifest term-to-category maps are "
                    "missing, so Manifest perturbations will use approximate semantic-count "
                    "updates. Prefer schema-3 PTs for final formal experiments."
                )
                self._warned_manifest_map_fallback = True
        if p_raw is None:
            pert_manifest = 0.0
        else:
            pert_manifest = float(torch.as_tensor(p_raw).float().view(-1)[0].item())
        if not math.isfinite(pert_manifest):
            pert_manifest = 0.0

        # ── derive permission/intent IDs from manifest_x bits (old-PT fallback) ──
        permission_ids_raw = _first_present(sources, "manifest_permission_ids")
        if permission_ids_raw is None and manifest_available and permission_dim > 0:
            perm_vec = manifest_x[:permission_dim]
            active = (perm_vec > 0.5).nonzero(as_tuple=True)[0]
            permission_ids_raw = active + 1  # +1: 0 = padding / unmatched

        intent_ids_raw = _first_present(sources, "manifest_intent_ids")
        if intent_ids_raw is None and manifest_available and intent_dim > 0:
            intent_vec = manifest_x[permission_dim : permission_dim + intent_dim]
            active = (intent_vec > 0.5).nonzero(as_tuple=True)[0]
            intent_ids_raw = active + 1

        return {
            "manifest_x": manifest_x,
            "manifest_permission_ids": _as_long_tensor(permission_ids_raw),
            "manifest_intent_ids": _as_long_tensor(intent_ids_raw),
            "manifest_permission_category_map": _as_category_map(
                permission_map_raw,
                permission_dim,
            ),
            "manifest_intent_category_map": _as_category_map(
                intent_map_raw,
                intent_dim,
            ),
            "manifest_category_counts": manifest_counts,
            "manifest_component_category_counts": manifest_component_counts,
            "manifest_stats": manifest_stats,
            "manifest_meta": meta,
            "graph_semantic_category_counts": graph_counts,
            "graph_category_counts": graph_counts,
            "q_manifest": max(0.0, min(1.0, q_manifest)),
            "pert_manifest": max(0.0, min(1.0, pert_manifest)),
            "manifest_aug_type": "none",
            "manifest_permission_dim": permission_dim,
            "manifest_intent_dim": intent_dim,
            "manifest_feature_dim": vocab_feature_dim if vocab_feature_dim is not None else _first_int(sources, "manifest_feature_dim", self.manifest_feature_dim),
        }

    def _to_data_object(self, data: dict[str, Any], label: int, sid: str, year: int) -> Data:
        obj = Data(x=data["x"], edge_index=data["edge_index"], y=torch.tensor(label, dtype=torch.long))
        obj.sensitive_mask = data["sensitive_mask"]
        obj.real_num_nodes = torch.tensor([int(data.get("real_num_nodes", 0))], dtype=torch.long)
        obj.real_node_mask = data["real_node_mask"].bool()
        obj.api_ids = data["api_ids"]
        obj.api_type_ids = data["api_type_ids"]
        obj.api_sensitive_mask = data["api_sensitive_mask"]
        obj.api_method_index = data["api_method_index"]
        obj.api_in_graph_mask = data["api_in_graph_mask"]
        obj.method_api_edge_index = data["method_api_edge_index"]
        obj.api_semantic_category_counts = data["api_semantic_category_counts"].float()
        obj.graph_semantic_category_counts = data["graph_semantic_category_counts"].float()
        obj.api_category_counts = obj.api_semantic_category_counts
        obj.graph_category_counts = obj.graph_semantic_category_counts
        obj.manifest_x = data["manifest_x"].float().view(1, -1)
        obj.manifest_permission_ids = data["manifest_permission_ids"].long().view(-1)
        obj.manifest_intent_ids = data["manifest_intent_ids"].long().view(-1)
        obj.manifest_category_counts = data["manifest_category_counts"].float().view(-1)
        obj.manifest_stats = data["manifest_stats"].float().view(-1)
        obj.q_api = torch.tensor([data["q_api"]], dtype=torch.float32)
        obj.q_graph = torch.tensor([data["q_graph"]], dtype=torch.float32)
        obj.q_manifest = torch.tensor([data["q_manifest"]], dtype=torch.float32)
        obj.q_align = torch.tensor([data["q_align"]], dtype=torch.float32)
        obj.pert_api = torch.tensor([data["pert_api"]], dtype=torch.float32)
        obj.pert_graph = torch.tensor([data["pert_graph"]], dtype=torch.float32)
        obj.pert_manifest = torch.tensor([data["pert_manifest"]], dtype=torch.float32)
        for key in OBSERVABLE_NUMERIC_FIELDS:
            setattr(obj, key, torch.tensor([scalar_float(data.get(key), 0.0)], dtype=torch.float32))
        for key in OBSERVABLE_SIGNAL_FIELDS:
            setattr(obj, key, torch.tensor([scalar_float(data.get(key), 0.0)], dtype=torch.float32))
        for key in OBSERVABLE_ERROR_FIELDS:
            setattr(obj, key, str(data.get(key, "") or ""))
        obj.schema_version = str(data.get("schema_version", "") or "")
        obj.sid = sid
        obj.year = torch.tensor(int(year), dtype=torch.long)
        obj.is_dummy = False
        obj.api_aug_type = data.get("api_aug_type", "none")
        obj.graph_aug_type = data.get("graph_aug_type", "none")
        obj.manifest_aug_type = data.get("manifest_aug_type", "none")
        return obj

    def __getitem__(self, idx: int):
        pt_path, label, sid, year = self.samples[idx]
        try:
            raw = torch.load(pt_path, map_location="cpu", weights_only=False)
            if self.min_pt_schema_version > 0:
                meta = raw.get("direct_build_meta", {}) if isinstance(raw, dict) else {}
                try:
                    version = int(meta.get("pt_schema_version", 0)) if isinstance(meta, dict) else 0
                except (TypeError, ValueError):
                    version = 0
                if version < self.min_pt_schema_version:
                    if not self.allow_legacy_pt_compat:
                        raise FatalDatasetConfigError(
                            f"PT schema version {version} is below required version "
                            f"{self.min_pt_schema_version}: {pt_path}. Regenerate PT files."
                        )
                    if not self._warned_legacy_schema:
                        logger.warning(
                            "Using legacy PT compatibility mode: observed PT schema version "
                            "%s below configured min_pt_schema_version=%s. This avoids APK "
                            "re-extraction but should be reported as a compatibility run.",
                            version,
                            self.min_pt_schema_version,
                        )
                        self._warned_legacy_schema = True
                if self.min_pt_schema_version >= 3:
                    required_fields = (
                        "manifest_permission_category_map",
                        "manifest_intent_category_map",
                        "manifest_component_category_counts",
                    )
                    missing_fields = [
                        key for key in required_fields
                        if not isinstance(raw, dict) or key not in raw
                    ]
                    if missing_fields:
                        if not self.allow_legacy_pt_compat:
                            raise FatalDatasetConfigError(
                                f"PT declares schema version {version} but is missing "
                                f"required fields {missing_fields}: {pt_path}. Regenerate PT files."
                            )
                        if not self._warned_legacy_schema:
                            logger.warning(
                                "Using legacy PT compatibility mode: current-schema Manifest "
                                "fields are missing from at least one PT. Missing fields from "
                                "first observed sample: %s",
                                missing_fields,
                            )
                            self._warned_legacy_schema = True
            dex_list, sources = _normalize_loaded_pt(raw)
            data = self._aggregate_api_graph(dex_list)
            if data is None:
                return self._dummy(label, sid, year, "empty valid sample", pt_path)
            # Graph counts computed via the configured source (alignment /
            # full_api / zero) always win over whatever the .pt happens to
            # carry — manifest_payload only provides a fallback for legacy
            # files.
            graph_counts_from_source = data.get("graph_semantic_category_counts")
            apply_dex_success_ratio(data, sources)
            manifest_payload = self._manifest_payload(sources)
            data.update(manifest_payload)
            if isinstance(graph_counts_from_source, torch.Tensor):
                data["graph_semantic_category_counts"] = graph_counts_from_source
                data["graph_category_counts"] = graph_counts_from_source
            saved_observable = _merged_observable_metadata(sources)
            strict_observable = (
                raw.get("observable_metadata", {})
                if isinstance(raw, dict) and isinstance(raw.get("dex_list"), list)
                else {}
            )
            checked_observable = strict_observable if self.strict_observable_schema else saved_observable
            missing_observable = [
                key for key in OBSERVABLE_REQUIRED_FIELDS
                if key not in checked_observable
            ]
            invalid_version = checked_observable.get("schema_version") != OBSERVABLE_SCHEMA_VERSION
            if missing_observable or invalid_version:
                detail = (
                    f"PT observable schema is incomplete for {pt_path}: "
                    f"schema_version={checked_observable.get('schema_version')!r}, "
                    f"missing={missing_observable}"
                )
                if self.strict_observable_schema:
                    raise FatalDatasetConfigError(detail)
                if not self._warned_observable_schema:
                    logger.warning("%s. Deriving compatibility fallbacks.", detail)
                    self._warned_observable_schema = True
            data.update(_fallback_observable_metadata(data, sources, dex_list, saved_observable))
            refresh_observable_signals(data)
            if self.robust_aug and self.is_train:
                perturb_type, strength = sample_training_perturbation(self.perturb_prob, self.perturb_strengths)
                data = apply_perturbation(data, perturb_type, strength)
            elif not self.is_train and self.eval_perturb_type:
                # Keep aggregate perturbation subtypes stable across strength sweeps.
                seed = _stable_seed(sid, self.eval_perturb_type)
                with _temporary_random_seed(seed):
                    data = apply_perturbation(data, self.eval_perturb_type, self.eval_perturb_strength)
            return self._to_data_object(data, label, sid, year)
        except FatalDatasetConfigError:
            raise
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            return self._dummy(label, sid, year, f"{type(exc).__name__}: {exc}", pt_path)


def robust_collate_fn(data_list):
    failed_items = []
    valid_items = []
    for d in data_list:
        if d is None:
            failed_items.append({"sid": None, "path": None, "reason": "data is None"})
        elif getattr(d, "is_dummy", False):
            failed_items.append({
                "sid": getattr(d, "sid", None),
                "path": getattr(d, "fail_path", None),
                "reason": getattr(d, "fail_reason", "dummy sample"),
            })
        else:
            valid_items.append(d)

    if not valid_items:
        return {
            "graph_batch": None,
            "labels": None,
            "sids": None,
            "years": None,
            "quality": None,
            "failed_items": failed_items,
            "num_failed": len(failed_items),
            "num_valid": 0,
        }

    sids = [d.sid for d in valid_items]
    api_aug_types = [getattr(d, "api_aug_type", "none") for d in valid_items]
    graph_aug_types = [getattr(d, "graph_aug_type", "none") for d in valid_items]
    manifest_aug_types = [getattr(d, "manifest_aug_type", "none") for d in valid_items]
    years = torch.stack([d.year for d in valid_items]).view(-1)
    labels = torch.stack([d.y for d in valid_items])
    graph_list = []
    api_ids_all, api_type_all, api_sensitive_all, api_batch_all = [], [], [], []
    api_method_all, api_in_graph_all, method_edges_all = [], [], []
    api_counts, graph_counts, manifest_counts, manifest_xs, manifest_stats = [], [], [], [], []
    perm_ids_all, perm_batch_all, intent_ids_all, intent_batch_all = [], [], [], []
    observable_numeric = {key: [] for key in OBSERVABLE_NUMERIC_FIELDS}
    observable_signals = {key: [] for key in OBSERVABLE_SIGNAL_FIELDS}
    observable_errors = {key: [] for key in OBSERVABLE_ERROR_FIELDS}
    observable_versions: list[str] = []
    node_offset = 0
    api_offset = 0

    for sample_idx, d in enumerate(valid_items):
        gd = Data(x=d.x, edge_index=d.edge_index, y=d.y)
        gd.sensitive_mask = d.sensitive_mask
        graph_list.append(gd)

        n_api = int(d.api_ids.numel())
        if n_api > 0:
            api_ids_all.append(d.api_ids.long())
            api_type_all.append(d.api_type_ids.long())
            api_sensitive_all.append(d.api_sensitive_mask.float())
            api_batch_all.append(torch.full((n_api,), sample_idx, dtype=torch.long, device=d.api_ids.device))
            method_idx = d.api_method_index.long().clone()
            valid_method = method_idx >= 0
            method_idx[valid_method] += node_offset
            api_method_all.append(method_idx)
            api_in_graph_all.append(d.api_in_graph_mask.float())
            edge = d.method_api_edge_index
            if isinstance(edge, torch.Tensor) and edge.numel() > 0:
                edge = edge.long().clone()
                edge[0] += node_offset
                edge[1] += api_offset
                method_edges_all.append(edge)

        api_counts.append(getattr(d, "api_semantic_category_counts", d.api_category_counts).float())
        graph_counts.append(getattr(d, "graph_semantic_category_counts", d.graph_category_counts).float())
        manifest_counts.append(d.manifest_category_counts.float())
        manifest_xs.append(d.manifest_x.float().view(1, -1))
        manifest_stats.append(d.manifest_stats.float().view(-1))
        for key in OBSERVABLE_NUMERIC_FIELDS:
            observable_numeric[key].append(getattr(d, key).float().view(-1)[:1])
        for key in OBSERVABLE_SIGNAL_FIELDS:
            observable_signals[key].append(getattr(d, key).float().view(-1)[:1])
        for key in OBSERVABLE_ERROR_FIELDS:
            observable_errors[key].append(str(getattr(d, key, "") or ""))
        observable_versions.append(str(getattr(d, "schema_version", "") or ""))

        if d.manifest_permission_ids.numel() > 0:
            perm_ids_all.append(d.manifest_permission_ids.long())
            perm_batch_all.append(torch.full((d.manifest_permission_ids.numel(),), sample_idx, dtype=torch.long, device=d.x.device))
        if d.manifest_intent_ids.numel() > 0:
            intent_ids_all.append(d.manifest_intent_ids.long())
            intent_batch_all.append(torch.full((d.manifest_intent_ids.numel(),), sample_idx, dtype=torch.long, device=d.x.device))

        node_offset += int(d.x.size(0))
        api_offset += n_api

    graph_batch = Batch.from_data_list(graph_list)
    device = graph_batch.x.device
    graph_batch.api_ids = torch.cat(api_ids_all).long() if api_ids_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.api_type_ids = torch.cat(api_type_all).long() if api_type_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.api_sensitive_mask = torch.cat(api_sensitive_all).float() if api_sensitive_all else torch.empty((0,), dtype=torch.float32, device=device)
    graph_batch.api_batch = torch.cat(api_batch_all).long() if api_batch_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.api_method_index = torch.cat(api_method_all).long() if api_method_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.api_in_graph_mask = torch.cat(api_in_graph_all).float() if api_in_graph_all else torch.empty((0,), dtype=torch.float32, device=device)
    graph_batch.method_api_edge_index = torch.cat(method_edges_all, dim=1).long() if method_edges_all else torch.empty((2, 0), dtype=torch.long, device=device)
    graph_batch.api_semantic_category_counts = torch.stack(api_counts).float()
    graph_batch.graph_semantic_category_counts = torch.stack(graph_counts).float()
    graph_batch.api_category_counts = graph_batch.api_semantic_category_counts
    graph_batch.graph_category_counts = graph_batch.graph_semantic_category_counts
    graph_batch.manifest_x = torch.cat(manifest_xs, dim=0).float()
    graph_batch.manifest_category_counts = torch.stack(manifest_counts).float()
    graph_batch.manifest_stats = torch.stack(manifest_stats).float()
    graph_batch.manifest_permission_ids = torch.cat(perm_ids_all).long() if perm_ids_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.manifest_permission_batch = torch.cat(perm_batch_all).long() if perm_batch_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.manifest_intent_ids = torch.cat(intent_ids_all).long() if intent_ids_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.manifest_intent_batch = torch.cat(intent_batch_all).long() if intent_batch_all else torch.empty((0,), dtype=torch.long, device=device)

    q_api = torch.stack([d.q_api for d in valid_items])
    q_graph = torch.stack([d.q_graph for d in valid_items])
    q_manifest = torch.stack([d.q_manifest for d in valid_items])
    q_align = torch.stack([d.q_align for d in valid_items])
    pert_api = torch.stack([d.pert_api for d in valid_items])
    pert_graph = torch.stack([d.pert_graph for d in valid_items])
    pert_manifest = torch.stack([d.pert_manifest for d in valid_items])
    graph_batch.q_api = q_api
    graph_batch.q_graph = q_graph
    graph_batch.q_manifest = q_manifest
    graph_batch.q_align = q_align
    graph_batch.pert_api = pert_api
    graph_batch.pert_graph = pert_graph
    graph_batch.pert_manifest = pert_manifest
    for key, values in observable_numeric.items():
        setattr(graph_batch, key, torch.stack(values).float())
    for key, values in observable_signals.items():
        setattr(graph_batch, key, torch.stack(values).float())
    graph_batch.observable_errors = observable_errors
    graph_batch.observable_schema_versions = observable_versions
    graph_batch.years = years

    return {
        "graph_batch": graph_batch,
        "labels": labels,
        "sids": sids,
        "api_aug_types": api_aug_types,
        "graph_aug_types": graph_aug_types,
        "manifest_aug_types": manifest_aug_types,
        "years": years,
        "quality": {
            "q_api": q_api,
            "q_graph": q_graph,
            "q_manifest": q_manifest,
            "q_align": q_align,
            "pert_api": pert_api,
            "pert_graph": pert_graph,
            "pert_manifest": pert_manifest,
            **{
                key: getattr(graph_batch, key)
                for key in OBSERVABLE_SIGNAL_FIELDS
            },
        },
        "observable_metadata": {
            **{
                key: getattr(graph_batch, key)
                for key in OBSERVABLE_NUMERIC_FIELDS
            },
            **observable_errors,
            "schema_version": observable_versions,
        },
        "failed_items": failed_items,
        "num_failed": len(failed_items),
        "num_valid": len(valid_items),
    }


def prepare_robust_batch(batch: dict[str, Any], device: torch.device):
    graph = batch.get("graph_batch")
    labels = batch.get("labels")
    if graph is None or labels is None:
        return None, None, None, None, int(batch.get("num_failed", 0))
    graph = graph.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    quality = batch.get("quality") or {}
    quality = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in quality.items()}
    return graph, labels, batch.get("sids"), quality, int(batch.get("num_failed", 0))
