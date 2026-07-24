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
)
from fusion.semantic_categories import (
    SEMANTIC_CATEGORY_DIM,
    sanitize_semantic_counts,
)
from fusion.quality import (
    OBSERVABLE_REQUIRED_FIELDS,
    OBSERVABLE_SCHEMA_VERSION,
    refresh_hard_availability,
)
from fusion.pt_schema import CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS, PT_SCHEMA_VERSION
from fusion.utils import strict_binary_integer, strict_finite_integer

logger = logging.getLogger(__name__)


def apply_graph_encoder_budget(
    data: dict[str, Any],
    max_nodes: int,
) -> dict[str, Any]:
    """Apply the one declared dataset/encoder graph-node budget."""

    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_nodes must be one positive integer")
    runtime_budget = int(max_nodes)
    # Runtime-only contract consumed by the graph encoder.  It prevents a
    # silently different second truncation when dataset and encoder budgets
    # drift apart; no APK-derived PT field is changed or required.
    data["graph_encoder_budget_max_nodes"] = runtime_budget
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        return data
    storage_nodes = int(x.size(0))
    if storage_nodes <= max_nodes:
        return data

    # Graph message passing is order-insensitive, so retain security-relevant
    # real nodes across all DEX files before filling the remaining budget from
    # the original real-node prefix. Ghost nodes keep empty DEX files
    # representable, but must never consume budget while a real node is still
    # available. Within each priority tier the original order is stable.
    sensitive = data.get("sensitive_mask")
    if isinstance(sensitive, torch.Tensor) and sensitive.numel() == storage_nodes:
        sensitive_flag = sensitive.view(-1).bool().to(x.device)
    else:
        sensitive_flag = torch.zeros(
            storage_nodes, dtype=torch.bool, device=x.device
        )
    real_mask = data.get("real_node_mask")
    if isinstance(real_mask, torch.Tensor) and real_mask.numel() == storage_nodes:
        real_flag = real_mask.view(-1).bool().to(x.device)
    else:
        real_flag = torch.ones(
            storage_nodes, dtype=torch.bool, device=x.device
        )
    priority_parts = (
        torch.nonzero(real_flag & sensitive_flag, as_tuple=False).view(-1),
        torch.nonzero(real_flag & ~sensitive_flag, as_tuple=False).view(-1),
        torch.nonzero(~real_flag, as_tuple=False).view(-1),
    )
    keep = torch.cat(priority_parts, dim=0)[:max_nodes]
    keep = keep.sort().values.long()
    mapping = torch.full(
        (storage_nodes,), -1, dtype=torch.long, device=x.device
    )
    mapping[keep] = torch.arange(
        max_nodes, dtype=torch.long, device=x.device
    )
    data["x"] = x[keep]
    for key in ("sensitive_mask", "real_node_mask"):
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.size(0) == storage_nodes:
            data[key] = value[keep.to(value.device)]

    edge = data.get("edge_index")
    if isinstance(edge, torch.Tensor) and edge.ndim == 2 and edge.size(0) == 2:
        edge = edge.long().to(x.device)
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

    refresh_hard_availability(data)
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
    expected_sid = Path(pt_path).stem.strip().lower()
    direct_sid = str(direct_meta.get("sha256") or "").strip().lower()
    if not direct_sid or direct_sid != expected_sid:
        raise FatalDatasetConfigError(
            "PT filename/direct_build_meta.sha256 identity mismatch: "
            f"path={pt_path} expected={expected_sid!r} actual={direct_sid!r}"
        )
    manifest_meta = raw.get("manifest_meta")
    if not isinstance(manifest_meta, dict):
        raise FatalDatasetConfigError(f"PT manifest_meta must be a mapping: {pt_path}")
    manifest_sid = str(manifest_meta.get("sha256") or "").strip().lower()
    if not manifest_sid or manifest_sid != expected_sid:
        raise FatalDatasetConfigError(
            "PT filename/manifest_meta.sha256 identity mismatch: "
            f"path={pt_path} expected={expected_sid!r} actual={manifest_sid!r}"
        )

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


class RobustTriModalDataset(Dataset):
    """Standalone API + Graph + Manifest dataset for robust fusion."""

    def __init__(
        self,
        pt_dir: str,
        csv_path: str,
        is_train: bool = True,
        eval_perturb_type: str | None = None,
        eval_perturb_strength: float = 0.0,
        max_api_events_per_sample: int | None = None,
        max_graph_nodes_per_sample: int = 12288,
        manifest_dim: int = 256,
        manifest_category_dim: int = 12,
        manifest_stats_dim: int = 11,
        manifest_permission_dim: int = 128,
        manifest_intent_dim: int = 64,
        manifest_feature_dim: int = 32,
        drop_graph_behavior_hints: bool = False,
        num_classes: int = 2,
        label_map: dict | None = None,
        strict_split_integrity: bool = True,
        allow_pt_superset: bool = False,
        expected_manifest_vocab_sha256: str | None = None,
        expected_manifest_train_csv_sha256: str | None = None,
        expected_manifest_train_sample_ids_sha256: str | None = None,
        expected_pt_build_fingerprint: str | None = None,
    ):
        if eval_perturb_type not in EVAL_PERTURB_TYPES:
            raise ValueError(f"Unsupported eval_perturb_type: {eval_perturb_type}")
        self.pt_dir = Path(pt_dir)
        self.is_train = bool(is_train)
        self.eval_perturb_type = eval_perturb_type
        self.eval_perturb_strength = float(eval_perturb_strength)
        if not math.isfinite(self.eval_perturb_strength) or not 0.0 <= self.eval_perturb_strength <= 1.0:
            raise ValueError(
                f"eval_perturb_strength must be within [0, 1], got {self.eval_perturb_strength}"
            )
        self.max_api_events_per_sample = (
            strict_finite_integer(
                max_api_events_per_sample,
                field_name="max_api_events_per_sample",
            )
            if max_api_events_per_sample is not None
            else None
        )
        self.max_graph_nodes_per_sample = strict_finite_integer(
            max_graph_nodes_per_sample,
            field_name="max_graph_nodes_per_sample",
        )
        if self.max_api_events_per_sample is not None and self.max_api_events_per_sample <= 0:
            raise ValueError("max_api_events_per_sample must be positive; use None for no limit")
        if self.max_graph_nodes_per_sample <= 0:
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
        self.strict_split_integrity = bool(strict_split_integrity)
        self.allow_pt_superset = bool(allow_pt_superset)
        self.expected_pt_build_fingerprint = str(
            expected_pt_build_fingerprint or ""
        ).strip().lower()
        if self.expected_pt_build_fingerprint and (
            len(self.expected_pt_build_fingerprint) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.expected_pt_build_fingerprint
            )
        ):
            raise ValueError(
                "expected_pt_build_fingerprint must be a lowercase SHA-256 digest"
            )
        manifest_provenance = {
            "manifest_vocab_sha256": expected_manifest_vocab_sha256,
            "manifest_train_csv_sha256": expected_manifest_train_csv_sha256,
            "manifest_train_sample_ids_sha256": (
                expected_manifest_train_sample_ids_sha256
            ),
        }
        self.expected_manifest_provenance = {
            key: str(value or "").strip().lower()
            for key, value in manifest_provenance.items()
            if str(value or "").strip()
        }
        if self.expected_manifest_provenance and len(
            self.expected_manifest_provenance
        ) != len(manifest_provenance):
            missing = sorted(
                set(manifest_provenance) - set(self.expected_manifest_provenance)
            )
            raise ValueError(
                "Manifest PT provenance must be supplied as one complete tuple; "
                f"missing={missing}"
            )
        for key, value in self.expected_manifest_provenance.items():
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{key} must be a lowercase SHA-256 digest")

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
        if label_map is not None:
            if not isinstance(label_map, dict) or not label_map:
                raise ValueError("data.label_map must be a non-empty mapping when provided")
            normalized_map: dict[str, int] = {}
            for raw_key, raw_value in label_map.items():
                key = str(raw_key).strip()
                if not key:
                    raise ValueError("data.label_map contains an empty normalized key")
                if key in normalized_map:
                    raise ValueError(
                        f"data.label_map contains duplicate normalized key {key!r}"
                    )
                normalized_map[key] = strict_binary_integer(
                    raw_value,
                    field_name=f"data.label_map[{key!r}]",
                )
            if df["label"].isna().any():
                bad = df.loc[df["label"].isna(), [id_col, "label"]].head(10).to_dict("records")
                raise ValueError(f"CSV {csv_path} contains missing labels; examples={bad}")
            mapped = raw_labels.map(normalized_map)
            if mapped.isna().any():
                bad = df.loc[mapped.isna(), [id_col, "label"]].head(10).to_dict("records")
                raise ValueError(
                    f"CSV {csv_path} contains labels not covered by data.label_map; "
                    f"examples={bad}"
                )
            label_series = mapped.astype("int64")
        else:
            parsed_labels: list[int] = []
            invalid_label_rows: list[Any] = []
            for row_index, raw_value in df["label"].items():
                try:
                    parsed_labels.append(
                        strict_binary_integer(
                            raw_value,
                            field_name=f"CSV label at row {row_index}",
                        )
                    )
                except ValueError:
                    invalid_label_rows.append(row_index)
            if invalid_label_rows:
                bad = (
                    df.loc[invalid_label_rows, [id_col, "label"]]
                    .head(10)
                    .to_dict("records")
                )
                raise ValueError(
                    f"CSV {csv_path} contains labels that are not finite binary integers; "
                    f"examples={bad}"
                )
            label_series = pd.Series(parsed_labels, index=df.index, dtype="int64")
        num_classes = strict_finite_integer(num_classes, field_name="num_classes")
        if num_classes != 2:
            raise ValueError(
                f"The current dataset contract is binary-only; num_classes must be 2, got {num_classes}"
            )
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
        if year_col:
            parsed_years: list[int] = []
            invalid_year_rows: list[Any] = []
            for row_index, raw_value in df[year_col].items():
                try:
                    parsed_years.append(
                        strict_finite_integer(
                            raw_value,
                            field_name=f"CSV year at row {row_index}",
                        )
                    )
                except ValueError:
                    invalid_year_rows.append(row_index)
            if invalid_year_rows:
                bad = (
                    df.loc[invalid_year_rows, [id_col, year_col]]
                    .head(10)
                    .to_dict("records")
                )
                raise ValueError(
                    f"CSV {csv_path} contains years that are not finite integers; examples={bad}"
                )
            years = dict(zip(sid_series, parsed_years))
        else:
            years = {sid: 0 for sid in sid_series}
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

    def _validate_manifest_pt_provenance(
        self,
        raw: dict[str, Any],
        pt_path: str | Path,
    ) -> None:
        """Reject stale or partially migrated Manifest payloads."""
        if not self.expected_manifest_provenance:
            return
        direct_meta = raw.get("direct_build_meta") or {}
        mismatches = {
            key: {
                "expected": expected,
                "actual": str(direct_meta.get(key) or "").strip().lower(),
            }
            for key, expected in self.expected_manifest_provenance.items()
            if str(direct_meta.get(key) or "").strip().lower() != expected
        }
        if mismatches:
            raise FatalDatasetConfigError(
                "PT Manifest provenance does not match the current train-only "
                "vocabulary. Run scripts/migrate_manifest_vocab_pts.py "
                f"(dry-run, then --apply) before training: path={pt_path} "
                f"mismatches={mismatches}"
            )

    def _validate_pt_build_fingerprint(
        self,
        raw: dict[str, Any],
        pt_path: str | Path,
    ) -> None:
        """Bind every consumed PT to the pre-registered extraction build."""

        if not self.expected_pt_build_fingerprint:
            return
        direct_meta = raw.get("direct_build_meta") or {}
        actual = str(direct_meta.get("build_fingerprint") or "").strip().lower()
        if actual != self.expected_pt_build_fingerprint:
            raise FatalDatasetConfigError(
                "PT build fingerprint does not match "
                "data.expected_pt_build_fingerprint: "
                f"path={pt_path} expected={self.expected_pt_build_fingerprint!r} "
                f"actual={actual!r}"
            )

    def _infer_feature_dim(self, default_dim: int) -> int:
        for pt_file, _, _, _ in self.samples:
            try:
                # Current APK payloads use PyTorch's zip format.  Memory mapping
                # keeps unused tensor storages lazy instead of copying the whole
                # multi-DEX payload on every epoch; tensors touched below retain
                # exactly the same dtype and values.
                raw = torch.load(
                    str(pt_file),
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                dex_list, _ = _validate_current_pt_payload(raw, pt_file)
                self._validate_pt_build_fingerprint(raw, pt_file)
                self._validate_manifest_pt_provenance(raw, pt_file)
                for dex in dex_list:
                    x = dex.get("call_x") if isinstance(dex, dict) else None
                    if isinstance(x, torch.Tensor) and x.ndim == 2 and x.size(1) > 0:
                        dim = int(x.size(1))
                        if self.drop_graph_behavior_hints and dim == 519:
                            return 515
                        return dim
            except FatalDatasetConfigError:
                raise
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
        data.api_ids = torch.empty((0,), dtype=torch.long)
        data.api_type_ids = torch.empty((0,), dtype=torch.long)
        data.api_sensitive_mask = torch.empty((0,), dtype=torch.float32)
        data.graph_encoder_budget_max_nodes = torch.tensor(
            [int(self.max_graph_nodes_per_sample)],
            dtype=torch.long,
        )
        data.manifest_x = torch.zeros((1, self.manifest_dim), dtype=torch.float32)
        data.api_alive = torch.tensor([0.0], dtype=torch.float32)
        data.graph_alive = torch.tensor([0.0], dtype=torch.float32)
        data.manifest_alive = torch.tensor([0.0], dtype=torch.float32)
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
        n = int(parts["api_ids"].numel())
        if self.max_api_events_per_sample is None:
            return parts
        if n <= self.max_api_events_per_sample:
            return parts
        limit = max(0, int(self.max_api_events_per_sample))
        device = parts["api_ids"].device
        # Intrinsic, modality-independent priority: keep all sensitive API events
        # (a property of the API call itself), then fill the remaining budget
        # with a CONTIGUOUS prefix of the non-sensitive events. Contiguity is
        # essential -- the API branch is a sequence model that learns local
        # behavioural n-grams, so an even/strided subsample would shatter those
        # patterns. Selection is based only on API-local tensors.
        sensitive = parts.get("api_sensitive_mask")
        if isinstance(sensitive, torch.Tensor) and sensitive.numel() == n:
            sens_flag = sensitive.to(device).view(-1) > 0.5
            sens_idx = torch.nonzero(sens_flag, as_tuple=False).view(-1)
            if int(sens_idx.numel()) >= limit:
                keep = sens_idx[:limit]
            else:
                non_idx = torch.nonzero(~sens_flag, as_tuple=False).view(-1)
                remaining = limit - int(sens_idx.numel())
                keep = torch.cat([sens_idx, non_idx[:remaining]])
            keep = keep.sort().values
        else:
            keep = torch.arange(limit, device=device)
        for key in ("api_ids", "api_type_ids", "api_sensitive_mask"):
            value = parts[key]
            parts[key] = value[keep.to(value.device)] if value.numel() > 0 else value[:0]
        return parts

    def _process_dex(self, dex: dict[str, Any], node_offset: int):
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

        parts = {
            "api_ids": api_ids,
            "api_type_ids": api_type_ids,
            "api_sensitive_mask": api_sensitive,
        }
        return {
            "x": x,
            "edge_index": edge_index,
            "sensitive_mask": sensitive,
            "real_node_mask": real_node_mask,
            "num_nodes": n,
            **parts,
        }

    def _aggregate_api_graph(self, dex_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        xs, edges, sens, real_masks = [], [], [], []
        api_ids, api_types, api_sensitive = [], [], []
        node_offset = 0
        for dex in dex_list:
            if not isinstance(dex, dict):
                continue
            part = self._process_dex(dex, node_offset)
            xs.append(part["x"])
            edges.append(part["edge_index"])
            sens.append(part["sensitive_mask"])
            real_masks.append(part["real_node_mask"])
            api_ids.append(part["api_ids"])
            api_types.append(part["api_type_ids"])
            api_sensitive.append(part["api_sensitive_mask"])
            node_offset += int(part["num_nodes"])
        if not xs:
            return None

        x = torch.cat(xs, dim=0)
        edge_index = torch.cat([e for e in edges if e.numel() > 0], dim=1) if any(e.numel() > 0 for e in edges) else torch.empty((2, 0), dtype=torch.long)
        sensitive_mask = torch.cat(sens, dim=0).to(torch.uint8)
        real_node_mask = torch.cat(real_masks, dim=0).bool()
        final_api_ids = torch.cat([v for v in api_ids if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_ids) else torch.empty((0,), dtype=torch.long)
        final_api_types = torch.cat([v for v in api_types if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_types) else torch.empty((0,), dtype=torch.long)
        final_api_sensitive = torch.cat([v for v in api_sensitive if v.numel() > 0], dim=0) if any(v.numel() > 0 for v in api_sensitive) else torch.empty((0,), dtype=torch.float32)

        api_parts = self._limit_api_events({
            "api_ids": final_api_ids,
            "api_type_ids": final_api_types,
            "api_sensitive_mask": final_api_sensitive,
        })
        final_api_ids = api_parts["api_ids"]
        final_api_types = api_parts["api_type_ids"]
        final_api_sensitive = api_parts["api_sensitive_mask"]
        return {
            "x": x,
            "edge_index": edge_index,
            "sensitive_mask": sensitive_mask,
            "real_node_mask": real_node_mask,
            "api_ids": final_api_ids,
            "api_type_ids": final_api_types,
            "api_sensitive_mask": final_api_sensitive,
            "api_aug_type": "none",
            "graph_aug_type": "none",
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
        sanitize_semantic_counts(
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
        permission_ids = _as_long_tensor(payload["manifest_permission_ids"])
        _as_long_tensor(payload["manifest_intent_ids"])
        permission_map = _as_category_map(
            permission_map_raw,
            permission_dim,
        )
        _as_category_map(intent_map_raw, intent_dim)
        return {
            "manifest_x": manifest_x,
            # CPU-only state required to keep manifest_x internally consistent
            # under manifest_permission_mask. None of these tensors reaches the
            # model batch.
            "manifest_permission_ids": permission_ids,
            "manifest_permission_category_map": permission_map,
            "manifest_category_counts": manifest_counts,
            "manifest_stats": manifest_stats,
            "manifest_aug_type": "none",
            "manifest_permission_dim": permission_dim,
            "manifest_intent_dim": intent_dim,
            "manifest_feature_dim": feature_dim,
        }

    def _to_data_object(self, data: dict[str, Any], label: int, sid: str, year: int) -> Data:
        obj = Data(x=data["x"], edge_index=data["edge_index"], y=torch.tensor(label, dtype=torch.long))
        obj.sensitive_mask = data["sensitive_mask"]
        obj.api_ids = data["api_ids"]
        obj.api_type_ids = data["api_type_ids"]
        obj.api_sensitive_mask = data["api_sensitive_mask"]
        obj.graph_encoder_budget_max_nodes = torch.tensor(
            [int(data["graph_encoder_budget_max_nodes"])],
            dtype=torch.long,
        )
        obj.manifest_x = data["manifest_x"].float().view(1, -1)
        obj.api_alive = torch.tensor([float(data["api_alive"])], dtype=torch.float32)
        obj.graph_alive = torch.tensor(
            [float(data["graph_alive"])], dtype=torch.float32
        )
        obj.manifest_alive = torch.tensor(
            [float(data["manifest_alive"])], dtype=torch.float32
        )
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
            # PT files are intentionally left unchanged.  mmap only changes how
            # their storages are paged into CPU memory and avoids eagerly reading
            # large extraction fields that this runtime does not consume.
            raw = torch.load(
                str(pt_path),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            dex_list, _sources = _validate_current_pt_payload(raw, pt_path)
            self._validate_pt_build_fingerprint(raw, pt_path)
            self._validate_manifest_pt_provenance(raw, pt_path)
            data = self._aggregate_api_graph(dex_list)
            if data is None:
                return self._dummy(label, sid, year, "empty valid sample", pt_path)
            data.update(self._manifest_payload(raw))
            # Persisted quality/coverage fields remain schema-validated for PT
            # provenance. Runtime fusion reads only parse state plus counts
            # recomputed from the tensors that actually enter the encoders.
            observable = raw["observable_metadata"]
            for key in (
                "api_parse_ok",
                "dex_parse_ok",
                "graph_parse_ok",
                "graph_timeout",
                "manifest_parse_ok",
                "manifest_has_content",
                "manifest_permission_count",
                "manifest_component_count",
                "manifest_intent_count",
            ):
                data[key] = observable[key]
            refresh_hard_availability(data)
            if not self.is_train and self.eval_perturb_type:
                # Keep aggregate perturbation subtypes stable across strength sweeps.
                seed = _stable_seed(sid, self.eval_perturb_type)
                with _temporary_random_seed(seed):
                    data = apply_perturbation(data, self.eval_perturb_type, self.eval_perturb_strength)
            data = apply_graph_encoder_budget(
                data,
                self.max_graph_nodes_per_sample,
            )
            obj = self._to_data_object(data, label, sid, year)
            # Runtime-only isolation metadata from the split CSV. It is not
            # persisted into, or required from, the APK-derived PT payload.
            obj.sample_group = str(self.sample_groups[idx])
            return obj
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
            "sample_groups": None,
            "years": None,
            "failed_items": failed_items,
            "num_failed": len(failed_items),
            "num_valid": 0,
        }

    sids = [d.sid for d in valid_items]
    sample_groups = [
        str(getattr(d, "sample_group", d.sid)) for d in valid_items
    ]
    api_aug_types = [getattr(d, "api_aug_type", "none") for d in valid_items]
    graph_aug_types = [getattr(d, "graph_aug_type", "none") for d in valid_items]
    manifest_aug_types = [getattr(d, "manifest_aug_type", "none") for d in valid_items]
    years = torch.stack([d.year for d in valid_items]).view(-1)
    labels = torch.stack([d.y for d in valid_items])
    graph_list = []
    api_ids_all, api_type_all, api_sensitive_all, api_batch_all = [], [], [], []
    manifest_xs = []
    availability_values = {
        "api_alive": [],
        "graph_alive": [],
        "manifest_alive": [],
    }
    graph_budget_values: list[int] = []
    graph_node_counts: list[int] = []
    for sample_idx, d in enumerate(valid_items):
        gd = Data(x=d.x, edge_index=d.edge_index)
        gd.sensitive_mask = d.sensitive_mask
        graph_budget = torch.as_tensor(
            getattr(d, "graph_encoder_budget_max_nodes", 0),
            dtype=torch.long,
            device=d.x.device,
        ).view(-1)[:1]
        gd.graph_encoder_budget_max_nodes = graph_budget
        graph_budget_values.append(
            int(graph_budget.detach().cpu().item())
            if graph_budget.numel() == 1
            else 0
        )
        graph_node_counts.append(int(d.x.size(0)))
        graph_list.append(gd)

        n_api = int(d.api_ids.numel())
        if n_api > 0:
            api_ids_all.append(d.api_ids.long())
            api_type_all.append(d.api_type_ids.long())
            api_sensitive_all.append(d.api_sensitive_mask.float())
            api_batch_all.append(torch.full((n_api,), sample_idx, dtype=torch.long, device=d.api_ids.device))

        manifest_xs.append(d.manifest_x.float().view(1, -1))
        for key in availability_values:
            availability_values[key].append(getattr(d, key).float().view(-1)[:1])

    graph_batch = Batch.from_data_list(graph_list)
    if (
        graph_budget_values
        and graph_budget_values[0] > 0
        and all(value == graph_budget_values[0] for value in graph_budget_values)
        and all(
            count <= graph_budget_values[0] for count in graph_node_counts
        )
    ):
        # CPU-only proof that every graph was already constrained by the same
        # dataset budget. The encoder consumes this runtime marker after the
        # batch moves to GPU, avoiding per-graph scalar synchronization and
        # identity keep-index construction. PT payloads are unchanged.
        graph_batch.graph_encoder_budget_contract = [
            1,
            int(graph_budget_values[0]),
            list(graph_node_counts),
        ]
    device = graph_batch.x.device
    graph_batch.api_ids = torch.cat(api_ids_all).long() if api_ids_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.api_type_ids = torch.cat(api_type_all).long() if api_type_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.api_sensitive_mask = torch.cat(api_sensitive_all).float() if api_sensitive_all else torch.empty((0,), dtype=torch.float32, device=device)
    graph_batch.api_batch = torch.cat(api_batch_all).long() if api_batch_all else torch.empty((0,), dtype=torch.long, device=device)
    graph_batch.manifest_x = torch.cat(manifest_xs, dim=0).float()

    for key, values in availability_values.items():
        setattr(graph_batch, key, torch.stack(values).float())

    return {
        "graph_batch": graph_batch,
        "labels": labels,
        "sids": sids,
        "sample_groups": sample_groups,
        "api_aug_types": api_aug_types,
        "graph_aug_types": graph_aug_types,
        "manifest_aug_types": manifest_aug_types,
        "years": years,
        "failed_items": failed_items,
        "num_failed": len(failed_items),
        "num_valid": len(valid_items),
    }


def prepare_robust_batch(batch: dict[str, Any], device: torch.device):
    graph = batch.get("graph_batch")
    labels = batch.get("labels")
    if graph is None or labels is None:
        return None, None, None, int(batch.get("num_failed", 0))
    graph = graph.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return graph, labels, batch.get("sids"), int(batch.get("num_failed", 0))
