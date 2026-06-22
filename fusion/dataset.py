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
    refresh_observable_signals,
)
from fusion.pt_schema import CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS, PT_SCHEMA_VERSION
from fusion.utils import scalar_float

logger = logging.getLogger(__name__)


VALID_GRAPH_SEMANTIC_SOURCES = ("alignment", "full_api", "zero")


def apply_graph_encoder_budget(
    data: dict[str, Any],
    max_nodes: int | None,
    graph_semantic_source: str,
) -> dict[str, Any]:
    """Apply the model's per-sample graph budget before evidence is refreshed."""
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        return data
    storage_nodes = int(x.size(0))
    raw_integrity = float(data.get("graph_integrity", data.get("q_graph", 0.0)))
    if max_nodes is None or max_nodes <= 0 or storage_nodes <= max_nodes:
        data["graph_encoder_coverage"] = 1.0
        data["graph_truncated_by_encoder_budget"] = 0.0
        data["graph_integrity_before_encoder_budget"] = raw_integrity
        return data

    keep = torch.arange(max_nodes, dtype=torch.long)
    mapping = torch.full((storage_nodes,), -1, dtype=torch.long)
    mapping[keep] = torch.arange(max_nodes, dtype=torch.long)
    data["x"] = x[keep]
    for key in ("sensitive_mask", "real_node_mask", "mask"):
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.size(0) == storage_nodes:
            data[key] = value[keep]

    edge = data.get("edge_index")
    if isinstance(edge, torch.Tensor) and edge.ndim == 2 and edge.size(0) == 2:
        edge = edge.long()
        valid = (
            (edge[0] >= 0)
            & (edge[0] < storage_nodes)
            & (edge[1] >= 0)
            & (edge[1] < storage_nodes)
        )
        edge = edge[:, valid]
        src = mapping[edge[0]]
        dst = mapping[edge[1]]
        retained = (src >= 0) & (dst >= 0)
        data["edge_index"] = torch.stack([src[retained], dst[retained]], dim=0)

    method_edge = data.get("method_api_edge_index")
    if isinstance(method_edge, torch.Tensor) and method_edge.ndim == 2 and method_edge.size(0) == 2:
        method_edge = method_edge.long()
        valid = (method_edge[0] >= 0) & (method_edge[0] < storage_nodes)
        method_edge = method_edge[:, valid]
        retained_src = mapping[method_edge[0]]
        retained = retained_src >= 0
        data["method_api_edge_index"] = torch.stack(
            [retained_src[retained], method_edge[1, retained]], dim=0
        )

    api_method_index = data.get("api_method_index")
    if isinstance(api_method_index, torch.Tensor):
        api_method_index = api_method_index.long().view(-1)
        valid = (api_method_index >= 0) & (api_method_index < storage_nodes)
        mapped = torch.full_like(api_method_index, -1)
        mapped[valid] = mapping[api_method_index[valid]]
        data["api_method_index"] = mapped

    api_ids = data.get("api_ids")
    num_api = int(api_ids.numel()) if isinstance(api_ids, torch.Tensor) else 0
    api_in_graph = torch.zeros((num_api,), dtype=torch.float32)
    retained_method_edge = data.get("method_api_edge_index")
    if (
        isinstance(retained_method_edge, torch.Tensor)
        and retained_method_edge.ndim == 2
        and retained_method_edge.size(0) == 2
        and retained_method_edge.numel() > 0
    ):
        dst = retained_method_edge[1].long()
        dst = dst[(dst >= 0) & (dst < num_api)]
        if dst.numel() > 0:
            api_in_graph[dst.unique()] = 1.0
    data["api_in_graph_mask"] = api_in_graph

    api_types = data.get("api_type_ids")
    if graph_semantic_source == "alignment":
        data["graph_semantic_category_counts"] = graph_semantic_counts_from_method_api_edges(
            api_types, data.get("method_api_edge_index")
        )
    elif graph_semantic_source == "full_api":
        data["graph_semantic_category_counts"] = data["api_semantic_category_counts"].clone()
    else:
        data["graph_semantic_category_counts"] = torch.zeros(
            (SEMANTIC_CATEGORY_DIM,), dtype=torch.float32
        )
    data["graph_category_counts"] = data["graph_semantic_category_counts"]

    real_mask = data.get("real_node_mask")
    data["real_num_nodes"] = (
        int(real_mask.bool().sum().item())
        if isinstance(real_mask, torch.Tensor)
        else int(data["x"].size(0))
    )
    data["graph_encoder_coverage"] = float(max_nodes / storage_nodes)
    data["graph_truncated_by_encoder_budget"] = 1.0
    data["graph_integrity_before_encoder_budget"] = raw_integrity
    refresh_observable_signals(data)
    effective_graph_integrity = max(
        0.0,
        min(1.0, float(data["graph_integrity"]) * data["graph_encoder_coverage"]),
    )
    data["graph_integrity"] = effective_graph_integrity
    data["code_integrity"] = math.sqrt(
        max(0.0, float(data["api_integrity"]) * effective_graph_integrity)
    )
    data["q_graph"] = effective_graph_integrity
    data["r_graph"] = effective_graph_integrity
    return data

class FatalDatasetConfigError(RuntimeError):
    """Configuration/data schema error that must not be converted to a dummy sample."""


def build_package_isolation_groups(
    sample_sids: list[str],
    packages: dict[str, str],
) -> list[str]:
    """Group samples by package when available, otherwise by sample ID."""
    ignored_packages = {"", "nan", "none", "null"}
    groups = []
    for sid in sample_sids:
        package = str(packages.get(sid, "") or "").strip().lower()
        groups.append(
            f"package:{package}" if package not in ignored_packages else f"sample:{sid}"
        )
    return groups


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


def _validate_current_pt_payload(
    raw: Any,
    pt_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and expose only the current direct-builder PT structure."""
    if not isinstance(raw, dict):
        raise FatalDatasetConfigError(
            f"PT must be a current schema-{PT_SCHEMA_VERSION} top-level mapping: {pt_path}"
        )
    missing_top_level = [
        key for key in CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS if key not in raw
    ]
    if missing_top_level:
        raise FatalDatasetConfigError(
            f"PT schema-{PT_SCHEMA_VERSION} payload is missing top-level fields "
            f"{missing_top_level}: {pt_path}"
        )

    direct_meta = raw["direct_build_meta"]
    if not isinstance(direct_meta, dict):
        raise FatalDatasetConfigError(f"PT direct_build_meta must be a mapping: {pt_path}")
    try:
        version = int(direct_meta.get("pt_schema_version", 0))
    except (TypeError, ValueError):
        version = 0
    if version != PT_SCHEMA_VERSION:
        raise FatalDatasetConfigError(
            f"PT schema version {version} does not match required current version "
            f"{PT_SCHEMA_VERSION}: {pt_path}"
        )
    if direct_meta.get("schema_version") != OBSERVABLE_SCHEMA_VERSION:
        raise FatalDatasetConfigError(
            f"PT direct_build_meta.schema_version must be {OBSERVABLE_SCHEMA_VERSION!r}: "
            f"{pt_path}"
        )
    if not str(direct_meta.get("build_fingerprint") or "").strip():
        raise FatalDatasetConfigError(f"PT is missing direct build fingerprint: {pt_path}")

    observable = raw["observable_metadata"]
    if not isinstance(observable, dict):
        raise FatalDatasetConfigError(f"PT observable_metadata must be a mapping: {pt_path}")
    missing_observable = [key for key in OBSERVABLE_REQUIRED_FIELDS if key not in observable]
    if observable.get("schema_version") != OBSERVABLE_SCHEMA_VERSION or missing_observable:
        raise FatalDatasetConfigError(
            f"PT observable schema is incomplete for {pt_path}: "
            f"schema_version={observable.get('schema_version')!r}, "
            f"missing={missing_observable}"
        )

    dex_list = raw["dex_list"]
    if not isinstance(dex_list, list) or any(not isinstance(dex, dict) for dex in dex_list):
        raise FatalDatasetConfigError(f"PT dex_list must be a list of mappings: {pt_path}")
    return dex_list, [raw, *dex_list]


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
        max_graph_nodes_per_sample: int | None = None,
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
    ):
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
        self.max_graph_nodes_per_sample = (
            int(max_graph_nodes_per_sample)
            if max_graph_nodes_per_sample is not None
            else None
        )
        if self.max_graph_nodes_per_sample is not None and self.max_graph_nodes_per_sample <= 0:
            raise ValueError("max_graph_nodes_per_sample must be positive")
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
        self.sample_labels = [label for _, label, _, _ in self.samples]
        self.sample_years = [year for _, _, _, year in self.samples]
        self.sample_groups = build_package_isolation_groups(self.sample_sids, groups)
        self.feature_dim = self._infer_feature_dim(default_dim=515)
        logger.info("Loaded %d robust tri-modal samples from %s", len(self.samples), self.pt_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def _infer_feature_dim(self, default_dim: int) -> int:
        for pt_file, _, _, _ in self.samples:
            try:
                raw = torch.load(pt_file, map_location="cpu", weights_only=False)
                dex_list, _ = _validate_current_pt_payload(raw, pt_file)
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

    def _manifest_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_x_raw = payload["manifest_x"]
        required_tensors = (
            "manifest_x",
            "manifest_permission_ids",
            "manifest_intent_ids",
            "manifest_category_counts",
            "manifest_component_category_counts",
            "manifest_permission_category_map",
            "manifest_intent_category_map",
            "manifest_stats",
            "q_manifest",
            "pert_manifest",
        )
        invalid_tensors = [
            key for key in required_tensors if not isinstance(payload.get(key), torch.Tensor)
        ]
        if invalid_tensors:
            raise FatalDatasetConfigError(
                f"Current PT Manifest fields must be tensors: {invalid_tensors}"
            )
        for key in ("manifest_category_counts", "manifest_component_category_counts"):
            if payload[key].numel() != SEMANTIC_CATEGORY_DIM:
                raise FatalDatasetConfigError(
                    f"Current PT {key} must contain exactly {SEMANTIC_CATEGORY_DIM} values"
                )
        if payload["manifest_stats"].numel() != self.manifest_stats_dim:
            raise FatalDatasetConfigError(
                f"Current PT manifest_stats must contain exactly {self.manifest_stats_dim} values"
            )
        raw_manifest_dim = _flat_numel(manifest_x_raw)
        if raw_manifest_dim > self.manifest_dim:
            raise FatalDatasetConfigError(
                f"manifest_x dimension {raw_manifest_dim} exceeds configured manifest_dim={self.manifest_dim}; "
                "regenerate tri-modal .pt files or increase model.manifest_encoder.in_dim"
            )
        manifest_x = _as_float_tensor(manifest_x_raw, self.manifest_dim)
        manifest_counts = sanitize_semantic_counts(
            payload["manifest_category_counts"], require_exact=True
        )
        manifest_component_counts = sanitize_semantic_counts(
            payload["manifest_component_category_counts"], require_exact=True
        )
        manifest_stats = _as_float_tensor(payload["manifest_stats"], self.manifest_stats_dim)
        permission_dim = int(payload["manifest_permission_dim"])
        intent_dim = int(payload["manifest_intent_dim"])
        feature_dim = int(payload["manifest_feature_dim"])
        permission_map_raw = payload["manifest_permission_category_map"]
        intent_map_raw = payload["manifest_intent_category_map"]
        maps_available = (
            isinstance(permission_map_raw, torch.Tensor)
            and permission_map_raw.ndim == 2
            and permission_map_raw.shape == (permission_dim, SEMANTIC_CATEGORY_DIM)
            and isinstance(intent_map_raw, torch.Tensor)
            and intent_map_raw.ndim == 2
            and intent_map_raw.shape == (intent_dim, SEMANTIC_CATEGORY_DIM)
        )
        if not maps_available:
            raise FatalDatasetConfigError(
                "Current PT is missing valid Manifest term-to-category maps. "
                "Regenerate it with build_tri_modal_pts_direct.py."
            )
        meta = payload["manifest_meta"]
        if not isinstance(meta, dict):
            raise FatalDatasetConfigError("Current PT manifest_meta must be a mapping")
        q_manifest = float(torch.as_tensor(payload["q_manifest"]).float().view(-1)[0].item())
        if not math.isfinite(q_manifest):
            raise FatalDatasetConfigError("Current PT q_manifest must be finite")
        pert_manifest = float(torch.as_tensor(payload["pert_manifest"]).float().view(-1)[0].item())
        if not math.isfinite(pert_manifest):
            raise FatalDatasetConfigError("Current PT pert_manifest must be finite")

        return {
            "manifest_x": manifest_x,
            "manifest_permission_ids": _as_long_tensor(payload["manifest_permission_ids"]),
            "manifest_intent_ids": _as_long_tensor(payload["manifest_intent_ids"]),
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
            "q_manifest": max(0.0, min(1.0, q_manifest)),
            "pert_manifest": max(0.0, min(1.0, pert_manifest)),
            "manifest_aug_type": "none",
            "manifest_permission_dim": permission_dim,
            "manifest_intent_dim": intent_dim,
            "manifest_feature_dim": feature_dim,
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
        obj.graph_encoder_coverage = torch.tensor(
            [scalar_float(data.get("graph_encoder_coverage"), 1.0)], dtype=torch.float32
        )
        obj.graph_truncated_by_encoder_budget = torch.tensor(
            [scalar_float(data.get("graph_truncated_by_encoder_budget"), 0.0)],
            dtype=torch.float32,
        )
        obj.graph_integrity_before_encoder_budget = torch.tensor(
            [
                scalar_float(
                    data.get("graph_integrity_before_encoder_budget"),
                    data.get("graph_integrity", 0.0),
                )
            ],
            dtype=torch.float32,
        )
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
            dex_list, sources = _validate_current_pt_payload(raw, pt_path)
            data = self._aggregate_api_graph(dex_list)
            if data is None:
                return self._dummy(label, sid, year, "empty valid sample", pt_path)
            apply_dex_success_ratio(data, sources)
            data.update(self._manifest_payload(raw))
            data.update(raw["observable_metadata"])
            refresh_observable_signals(data)
            if self.robust_aug and self.is_train:
                perturb_type, strength = sample_training_perturbation(self.perturb_prob, self.perturb_strengths)
                data = apply_perturbation(data, perturb_type, strength)
            elif not self.is_train and self.eval_perturb_type:
                # Keep aggregate perturbation subtypes stable across strength sweeps.
                seed = _stable_seed(sid, self.eval_perturb_type)
                with _temporary_random_seed(seed):
                    data = apply_perturbation(data, self.eval_perturb_type, self.eval_perturb_strength)
            data = apply_graph_encoder_budget(
                data,
                self.max_graph_nodes_per_sample,
                self.graph_semantic_source,
            )
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
