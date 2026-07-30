from __future__ import annotations

"""Rebuild a train-only Manifest vocabulary and migrate an existing PT pool.

The migration deliberately does not parse APKs or touch Graph/API extraction.  It
reuses persisted Manifest JSONL records, rewrites only Manifest-owned PT fields,
and updates the direct-build fingerprint.  Running without ``--apply`` is a
read-only preflight.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract.extract_graph_api import atomic_torch_save
from fusion.manifest_features import (
    build_manifest_vocab,
    normalize_manifest_permissions,
    validate_manifest_vocab,
    vectorize_manifest_record,
)
from fusion.pt_schema import (
    CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS,
    PT_AUDIT_CERTIFICATE_VERSION,
    PT_SCHEMA_VERSION,
    PTSchemaValidationError,
    RETIRED_PT_TOP_LEVEL_FIELDS,
    pt_audit_entries_sha256,
    validate_observable_metadata,
    validate_current_dex_list,
    validate_current_pt_payload,
)
from fusion.quality import OBSERVABLE_REQUIRED_FIELDS, OBSERVABLE_SCHEMA_VERSION
from scripts.build_tri_modal_pts_direct import (
    _build_fingerprint,
    _fingerprint_config,
    _load_direct_config,
)


logger = logging.getLogger("migrate_manifest_vocab_pts")


MANIFEST_TOP_LEVEL_FIELDS = (
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

# Schema v4 persisted vocabulary-wide term/category maps solely to support the
# retired known-token masking implementation. Schema v5 rebuilds permission
# semantics from canonical raw tokens, so carrying these tensors forward would
# waste space and preserve a misleading second source of truth.
REMOVED_MANIFEST_TOP_LEVEL_FIELDS = (
    *RETIRED_PT_TOP_LEVEL_FIELDS,
)

MANIFEST_OBSERVABLE_FIELDS = (
    "manifest_parse_ok",
    "manifest_parse_error",
    "manifest_has_content",
    "manifest_vocab_coverage",
    "manifest_permission_count",
    "manifest_component_count",
    "manifest_intent_count",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_digest(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sid in sorted(str(value).strip().lower() for value in sample_ids):
        encoded = sid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()

_NON_MANIFEST_REQUIRED_FIELDS = {
    "dex_list",
    "observable_metadata",
    "direct_build_meta",
}
_CURRENT_MANIFEST_SCHEMA_FIELDS = (
    set(CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS) - _NON_MANIFEST_REQUIRED_FIELDS
)
if _CURRENT_MANIFEST_SCHEMA_FIELDS != set(MANIFEST_TOP_LEVEL_FIELDS):
    raise RuntimeError(
        "PT schema changed; review the Manifest migration field ownership before running"
    )


def _normalized_fieldnames(fieldnames: list[str] | None) -> dict[str, str]:
    return {
        str(name).strip().lstrip("\ufeff").strip('"').strip("'").lower(): name
        for name in (fieldnames or [])
    }


def read_csv_ids(csv_path: Path, id_column: str | None = None) -> list[str]:
    """Read unique sample IDs using the same auto-column priority as the dataset."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = _normalized_fieldnames(reader.fieldnames)
        if id_column:
            requested = str(id_column).strip().lower()
            if requested not in fields:
                raise ValueError(
                    f"{csv_path} does not contain ID column {id_column!r}; "
                    f"available columns={sorted(fields)}"
                )
            source_column = fields[requested]
        else:
            source_column = next(
                (fields[name] for name in ("id", "sha256") if name in fields),
                None,
            )
            if source_column is None:
                raise ValueError(f"{csv_path} must contain id or sha256")

        ids: list[str] = []
        seen: set[str] = set()
        duplicates: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            sid = str(row.get(source_column) or "").strip().lower()
            if not sid:
                raise ValueError(f"Empty sample ID at {csv_path}:{line_number}")
            if sid in seen:
                duplicates.add(sid)
            seen.add(sid)
            ids.append(sid)

    if duplicates:
        raise ValueError(
            f"Train CSV contains duplicate sample IDs: count={len(duplicates)} "
            f"examples={sorted(duplicates)[:10]}"
        )
    if not ids:
        raise ValueError(f"Train CSV is empty: {csv_path}")
    return ids


def list_pt_files(pt_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted(pt_dir.rglob("*.pt")):
        if not path.is_file():
            continue
        sid = path.stem.strip().lower()
        if not sid:
            raise ValueError(f"PT file has an empty basename: {path}")
        if sid in files:
            duplicates.add(sid)
        else:
            files[sid] = path
    if duplicates:
        raise ValueError(
            f"PT pool contains duplicate filename stems: count={len(duplicates)} "
            f"examples={sorted(duplicates)[:10]}"
        )
    if not files:
        raise ValueError(f"PT pool contains no .pt files: {pt_dir}")
    return files


def resolve_manifest_jsonl_files(
    explicit_files: Iterable[str | Path],
    directories: Iterable[str | Path],
    *,
    default_directory: Path | None,
) -> list[Path]:
    paths: list[Path] = []
    requested_files = [Path(value).expanduser().resolve() for value in explicit_files]
    requested_dirs = [Path(value).expanduser().resolve() for value in directories]
    if not requested_files and not requested_dirs and default_directory is not None:
        requested_dirs = [default_directory.expanduser().resolve()]

    for path in requested_files:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest JSONL does not exist: {path}")
        paths.append(path)
    for directory in requested_dirs:
        if not directory.is_dir():
            raise NotADirectoryError(f"Manifest JSONL directory does not exist: {directory}")
        paths.extend(path for path in sorted(directory.rglob("*.jsonl")) if path.is_file())

    unique: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        unique.setdefault(resolved, resolved)
    result = sorted(unique.values())
    if not result:
        raise ValueError("No Manifest JSONL files were found")
    return result


def read_manifest_jsonl_strict(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"Manifest record must be an object at {path}:{line_number}")
                sid_value = str(raw.get("sid") or "").strip().lower()
                sha_value = str(raw.get("sha256") or "").strip().lower()
                if sid_value and sha_value and sid_value != sha_value:
                    raise ValueError(
                        f"Manifest sid/sha256 mismatch at {path}:{line_number}: "
                        f"sid={sid_value!r} sha256={sha_value!r}"
                    )
                sid = sid_value or sha_value
                if not sid:
                    raise ValueError(f"Manifest record has no sid/sha256 at {path}:{line_number}")
                if sid in records:
                    raise ValueError(
                        f"Duplicate Manifest record for {sid!r}: "
                        f"first={origins[sid]}, duplicate={path}:{line_number}"
                    )
                record = dict(raw)
                record["sid"] = sid
                record["sha256"] = sid
                records[sid] = record
                origins[sid] = f"{path}:{line_number}"
    if not records:
        raise ValueError("Manifest JSONL inputs contain no records")
    return records


def _load_pt(path: Path) -> dict[str, Any]:
    # This migration intentionally has no unsafe/legacy pickle fallback.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"PT payload must be a top-level mapping: {path}")
    return payload


def _validate_current_pt(
    payload: dict[str, Any],
    path: Path,
    sid: str,
    *,
    allow_legacy_permission_mapping: bool = False,
) -> None:
    if not allow_legacy_permission_mapping:
        try:
            validate_current_pt_payload(payload, path, expected_sid=sid)
        except PTSchemaValidationError as exc:
            raise ValueError(str(exc)) from exc
        return

    direct_meta = payload.get("direct_build_meta")
    if not isinstance(direct_meta, dict):
        raise ValueError(f"PT direct_build_meta must be a mapping: {path}")
    raw_pt_version = direct_meta.get("pt_schema_version")
    if (
        isinstance(raw_pt_version, bool)
        or not isinstance(raw_pt_version, int)
    ):
        raise ValueError(
            f"PT schema version must be an integer at {path}: "
            f"actual={raw_pt_version!r}"
        )
    actual_pt_version = int(raw_pt_version)
    if actual_pt_version == PT_SCHEMA_VERSION:
        # A current payload must always satisfy the complete current contract.
        # The migration may remove explicitly retired top-level fields from a
        # briefly generated schema-v5 payload, but it cannot waive any other
        # part of the current contract.
        validation_payload = dict(payload)
        for field in RETIRED_PT_TOP_LEVEL_FIELDS:
            validation_payload.pop(field, None)
        try:
            validate_current_pt_payload(
                validation_payload,
                path,
                expected_sid=sid,
            )
        except PTSchemaValidationError as exc:
            raise ValueError(str(exc)) from exc
        return
    if actual_pt_version != PT_SCHEMA_VERSION - 1:
        raise ValueError(
            f"PT schema version mismatch at {path}: "
            f"expected one of {[PT_SCHEMA_VERSION - 1, PT_SCHEMA_VERSION]} "
            f"actual={actual_pt_version!r}"
        )

    missing = [key for key in CURRENT_PT_REQUIRED_TOP_LEVEL_FIELDS if key not in payload]
    allowed_missing = {"manifest_permission_token_ids"}
    unsupported_missing = sorted(set(missing) - allowed_missing)
    if unsupported_missing:
        raise ValueError(
            f"PT is not current schema-{PT_SCHEMA_VERSION}; "
            f"missing={unsupported_missing}: {path}"
        )
    try:
        validate_current_dex_list(payload.get("dex_list"), path=path)
    except PTSchemaValidationError as exc:
        raise ValueError(str(exc)) from exc

    if direct_meta.get("schema_version") != OBSERVABLE_SCHEMA_VERSION:
        raise ValueError(
            f"PT observable schema mismatch at {path}: "
            f"expected={OBSERVABLE_SCHEMA_VERSION!r} "
            f"actual={direct_meta.get('schema_version')!r}"
        )
    if not str(direct_meta.get("build_fingerprint") or "").strip():
        raise ValueError(f"PT is missing its direct build fingerprint: {path}")
    meta_sid = str(direct_meta.get("sha256") or "").strip().lower()
    if not meta_sid or meta_sid != sid:
        raise ValueError(
            f"PT filename/direct_build_meta.sha256 mismatch at {path}: "
            f"filename={sid!r} metadata={meta_sid!r}"
        )
    manifest_meta = payload.get("manifest_meta")
    if not isinstance(manifest_meta, dict):
        raise ValueError(f"PT manifest_meta must be a mapping: {path}")
    manifest_sid = str(manifest_meta.get("sha256") or "").strip().lower()
    if not manifest_sid or manifest_sid != sid:
        raise ValueError(
            f"PT filename/manifest_meta.sha256 mismatch at {path}: "
            f"filename={sid!r} metadata={manifest_sid!r}"
        )

    observable = payload.get("observable_metadata")
    if not isinstance(observable, dict):
        raise ValueError(f"PT observable_metadata must be a mapping: {path}")
    if observable.get("schema_version") != OBSERVABLE_SCHEMA_VERSION:
        raise ValueError(
            f"PT observable_metadata schema mismatch at {path}: "
            f"expected={OBSERVABLE_SCHEMA_VERSION!r} "
            f"actual={observable.get('schema_version')!r}"
        )
    missing_observable = [key for key in OBSERVABLE_REQUIRED_FIELDS if key not in observable]
    if missing_observable:
        raise ValueError(
            f"PT observable_metadata is incomplete; missing={missing_observable}: {path}"
        )
    try:
        validate_observable_metadata(observable, path=path)
    except PTSchemaValidationError as exc:
        raise ValueError(str(exc)) from exc

    token_ids = payload.get("manifest_permission_token_ids")
    if token_ids is None and allow_legacy_permission_mapping:
        if actual_pt_version != PT_SCHEMA_VERSION - 1:
            raise ValueError(
                "Only schema-v4 payloads may omit "
                f"manifest_permission_token_ids; schema-v{actual_pt_version} "
                f"is corrupt and must not use the legacy migration path: {path}"
            )
        return
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError(
            f"PT manifest_permission_token_ids must be a tensor: {path}"
        )
    raw_permissions = manifest_meta.get("permissions")
    if not isinstance(raw_permissions, list):
        raise ValueError(
            f"PT manifest_meta.permissions must be a canonical list: {path}"
        )
    permission_tokens = normalize_manifest_permissions(raw_permissions)
    if raw_permissions != permission_tokens:
        raise ValueError(
            "PT manifest_meta.permissions must be lower-case, de-duplicated, "
            f"and sorted: {path}"
        )
    token_ids = token_ids.detach().long().reshape(-1)
    if token_ids.numel() != len(permission_tokens):
        raise ValueError(
            "PT manifest_permission_token_ids does not align one-to-one with "
            f"manifest_meta.permissions: {path}"
        )
    permission_dim = int(payload.get("manifest_permission_dim", -1))
    if permission_dim < 0 or bool(
        ((token_ids < 0) | (token_ids > permission_dim)).any().item()
    ):
        raise ValueError(
            "PT manifest_permission_token_ids contains an ID outside "
            f"[0, {permission_dim}]: {path}"
        )
    permission_ids = payload.get("manifest_permission_ids")
    if not isinstance(permission_ids, torch.Tensor):
        raise ValueError(f"PT manifest_permission_ids must be a tensor: {path}")
    expected_known_ids = token_ids[token_ids > 0].unique(sorted=True)
    actual_known_ids = permission_ids.detach().long().reshape(-1)
    if not torch.equal(actual_known_ids.cpu(), expected_known_ids.cpu()):
        raise ValueError(
            "PT manifest_permission_ids disagrees with "
            f"manifest_permission_token_ids: {path}"
        )


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return set(left) == set(right) and all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def _manifest_matches(
    payload: dict[str, Any],
    expected: dict[str, Any],
    build_fingerprint: str,
    provenance: dict[str, str],
) -> tuple[bool, bool]:
    manifest_match = all(
        _values_equal(payload.get(key), expected.get(key))
        for key in MANIFEST_TOP_LEVEL_FIELDS
    )
    manifest_match = manifest_match and not any(
        key in payload for key in REMOVED_MANIFEST_TOP_LEVEL_FIELDS
    )
    observable = payload.get("observable_metadata") or {}
    expected_observable = expected.get("observable_metadata") or {}
    manifest_match = manifest_match and all(
        _values_equal(observable.get(key), expected_observable.get(key))
        for key in MANIFEST_OBSERVABLE_FIELDS
    )
    direct_meta = payload.get("direct_build_meta") or {}
    fingerprint_match = (
        str(direct_meta.get("build_fingerprint") or "")
        == build_fingerprint
        and all(
            str(direct_meta.get(key) or "") == value
            for key, value in provenance.items()
        )
    )
    return manifest_match, fingerprint_match


def _updated_payload(
    payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    build_fingerprint: str,
    provenance: dict[str, str],
) -> dict[str, Any]:
    """Return a shallow copy with only explicitly owned fields replaced."""
    updated = dict(payload)
    for key in REMOVED_MANIFEST_TOP_LEVEL_FIELDS:
        updated.pop(key, None)
    for key in MANIFEST_TOP_LEVEL_FIELDS:
        updated[key] = manifest_payload[key]

    observable = dict(payload["observable_metadata"])
    manifest_observable = manifest_payload["observable_metadata"]
    for key in MANIFEST_OBSERVABLE_FIELDS:
        observable[key] = manifest_observable[key]
    updated["observable_metadata"] = observable

    direct_meta = dict(payload["direct_build_meta"])
    direct_meta["pt_schema_version"] = PT_SCHEMA_VERSION
    direct_meta["build_fingerprint"] = str(build_fingerprint)
    direct_meta.update(provenance)
    updated["direct_build_meta"] = direct_meta
    return updated


def _vocab_digest(vocab: dict[str, Any]) -> str:
    encoded = json.dumps(vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vocab_summary(vocab: dict[str, Any]) -> dict[str, Any]:
    return {
        "permission_dim": len(vocab.get("permission_vocab") or []),
        "intent_dim": len(vocab.get("intent_vocab") or []),
        "feature_dim": len(vocab.get("feature_vocab") or []),
        "category_dim": len(vocab.get("categories") or []),
        "metadata": dict(vocab.get("metadata") or {}),
        "sha256": _vocab_digest(vocab),
    }


def _vocab_diff(old_vocab: dict[str, Any] | None, new_vocab: dict[str, Any]) -> dict[str, Any] | None:
    if old_vocab is None:
        return None
    result: dict[str, Any] = {}
    for key in ("permission_vocab", "intent_vocab", "feature_vocab"):
        old_items = [str(value) for value in (old_vocab.get(key) or [])]
        new_items = [str(value) for value in (new_vocab.get(key) or [])]
        old_set = set(old_items)
        new_set = set(new_items)
        result[key] = {
            "old_count": len(old_items),
            "new_count": len(new_items),
            "added_count": len(new_set - old_set),
            "removed_count": len(old_set - new_set),
            "added_examples": sorted(new_set - old_set)[:10],
            "removed_examples": sorted(old_set - new_set)[:10],
            "order_changed": old_items != new_items,
        }
    return result


def _atomic_write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_optional_vocab(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Existing Manifest vocab must be a mapping: {path}")
    return value


def _layout_key(payload: dict[str, Any]) -> str:
    manifest_x = payload.get("manifest_x")
    x_dim = int(manifest_x.numel()) if isinstance(manifest_x, torch.Tensor) else -1
    return (
        f"permission={int(payload.get('manifest_permission_dim', -1))},"
        f"intent={int(payload.get('manifest_intent_dim', -1))},"
        f"feature={int(payload.get('manifest_feature_dim', -1))},"
        f"manifest_x={x_dim}"
    )


def _mean_or_none(total: float, count: int) -> float | None:
    return float(total / count) if count else None


def _scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


def _audit_pool(
    pt_files: dict[str, Path],
    records: dict[str, dict[str, Any]],
    vocab: dict[str, Any],
    manifest_dim: int,
    build_fingerprint: str,
    provenance: dict[str, str],
    *,
    verify_expected: bool,
    certificate_root: Path | None = None,
    allowed_existing_fingerprints: set[str] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, tuple[int, int, int]]]:
    fingerprint_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    coverage_total = 0.0
    coverage_count = 0
    manifest_mismatch: list[str] = []
    fingerprint_mismatch: list[str] = []
    file_stats: dict[str, tuple[int, int, int]] = {}
    certificate_entries: list[dict[str, Any]] = []

    for index, (sid, path) in enumerate(sorted(pt_files.items()), start=1):
        payload = _load_pt(path)
        _validate_current_pt(
            payload,
            path,
            sid,
            allow_legacy_permission_mapping=not verify_expected,
        )
        expected = vectorize_manifest_record(records[sid], vocab, manifest_dim=manifest_dim)
        manifest_match, fingerprint_match = _manifest_matches(
            payload,
            expected,
            build_fingerprint,
            provenance,
        )
        if not manifest_match:
            manifest_mismatch.append(sid)
        if not fingerprint_match:
            fingerprint_mismatch.append(sid)
        if verify_expected and (not manifest_match or not fingerprint_match):
            reasons = []
            if not manifest_match:
                reasons.append("Manifest payload")
            if not fingerprint_match:
                reasons.append("build fingerprint")
            raise RuntimeError(f"Post-migration verification failed for {path}: {', '.join(reasons)} mismatch")

        meta = payload["direct_build_meta"]
        existing_fingerprint = str(meta.get("build_fingerprint") or "")
        if (
            allowed_existing_fingerprints is not None
            and existing_fingerprint not in allowed_existing_fingerprints
        ):
            raise RuntimeError(
                "PT code-side build fingerprint is incompatible with the "
                "supplied build config and previous Manifest vocabulary; "
                "refusing to relabel unchanged API/Graph tensors: "
                f"path={path} fingerprint={existing_fingerprint!r} "
                f"allowed={sorted(allowed_existing_fingerprints)}"
            )
        observable = payload["observable_metadata"]
        fingerprint_counts[existing_fingerprint or "<missing>"] += 1
        layout_counts[_layout_key(payload)] += 1
        coverage_total += float(observable["manifest_vocab_coverage"])
        coverage_count += 1
        stat = path.stat()
        current_stat = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        file_stats[sid] = current_stat
        if verify_expected:
            if certificate_root is None:
                raise RuntimeError(
                    "certificate_root is required for the post-migration audit"
                )
            relative_path = path.resolve().relative_to(
                certificate_root.resolve()
            ).as_posix()
            certificate_entries.append(
                {
                    "sid": sid,
                    "relative_path": relative_path,
                    "size": current_stat[0],
                    "mtime_ns": current_stat[1],
                    "ctime_ns": current_stat[2],
                    "call_x_dims": [
                        int(dex["call_x"].shape[1])
                        for dex in payload["dex_list"]
                    ],
                }
            )
        if index % 500 == 0:
            logger.info("Audited %d/%d PT files", index, len(pt_files))

    needs_update = sorted(set(manifest_mismatch) | set(fingerprint_mismatch))
    audit = {
        "pt_count": len(pt_files),
        "build_fingerprint_counts": dict(sorted(fingerprint_counts.items())),
        "manifest_layout_counts": dict(sorted(layout_counts.items())),
        "mean_manifest_vocab_coverage": _mean_or_none(coverage_total, coverage_count),
        "manifest_mismatch_count": len(manifest_mismatch),
        "manifest_mismatch_examples": manifest_mismatch[:10],
        "fingerprint_mismatch_count": len(fingerprint_mismatch),
        "fingerprint_mismatch_examples": fingerprint_mismatch[:10],
        "needs_update_count": len(needs_update),
        "needs_update_examples": needs_update[:10],
    }
    if verify_expected:
        audit["pt_audit_certificate"] = {
            "certificate_version": PT_AUDIT_CERTIFICATE_VERSION,
            "pt_schema_version": PT_SCHEMA_VERSION,
            "pool_root": str(certificate_root.resolve()),
            "build_fingerprint": build_fingerprint,
            "entries": certificate_entries,
            "certificate_sha256": pt_audit_entries_sha256(
                certificate_entries
            ),
        }
    return audit, needs_update, file_stats


def _assert_files_unchanged(
    pt_files: dict[str, Path],
    expected_stats: dict[str, tuple[int, int, int]],
) -> None:
    changed: list[str] = []
    for sid, path in sorted(pt_files.items()):
        stat = path.stat()
        current = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        if current != expected_stats[sid]:
            changed.append(sid)
    if changed:
        raise RuntimeError(
            "PT files changed after preflight; refusing to start migration: "
            f"count={len(changed)} examples={changed[:10]}"
        )


def _assert_file_unchanged(
    path: Path,
    expected_stat: tuple[int, int, int],
) -> None:
    stat = path.stat()
    current = (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    if current != expected_stat:
        raise RuntimeError(
            "PT changed after migration preflight; refusing to overwrite a "
            f"concurrent extractor result: path={path} "
            f"expected={expected_stat} actual={current}"
        )


def run_migration(
    *,
    train_csv: str | Path,
    pt_dir: str | Path,
    build_config: str | Path,
    vocab_out: str | Path | None = None,
    manifest_jsonl: Iterable[str | Path] = (),
    manifest_jsonl_dirs: Iterable[str | Path] = (),
    id_column: str | None = None,
    apply: bool = False,
    audit_json: str | Path | None = None,
) -> dict[str, Any]:
    train_csv_path = Path(train_csv).expanduser().resolve()
    pt_dir_path = Path(pt_dir).expanduser().resolve()
    build_config_path = Path(build_config).expanduser().resolve()
    if not train_csv_path.is_file():
        raise FileNotFoundError(f"Train CSV does not exist: {train_csv_path}")
    if not pt_dir_path.is_dir():
        raise NotADirectoryError(f"PT pool does not exist: {pt_dir_path}")
    if not build_config_path.is_file():
        raise FileNotFoundError(f"Direct-build config does not exist: {build_config_path}")

    cfg = _load_direct_config(build_config_path)
    vocab_out_path = (
        Path(vocab_out).expanduser().resolve()
        if vocab_out is not None
        else Path(cfg["manifest_vocab_path"]).expanduser().resolve()
    )
    jsonl_files = resolve_manifest_jsonl_files(
        manifest_jsonl,
        manifest_jsonl_dirs,
        default_directory=Path(cfg["manifest_jsonl_dir"]),
    )

    train_ids = read_csv_ids(train_csv_path, id_column=id_column)
    train_id_set = set(train_ids)
    pt_files = list_pt_files(pt_dir_path)
    pt_ids = set(pt_files)
    records = read_manifest_jsonl_strict(jsonl_files)
    record_ids = set(records)

    train_missing_pt = sorted(train_id_set - pt_ids)
    train_missing_manifest = sorted(train_id_set - record_ids)
    pt_missing_manifest = sorted(pt_ids - record_ids)
    if train_missing_pt or train_missing_manifest or pt_missing_manifest:
        raise RuntimeError(
            "Manifest migration preflight failed; no files were modified. "
            f"train_missing_pt={len(train_missing_pt)} examples={train_missing_pt[:10]}; "
            f"train_missing_manifest={len(train_missing_manifest)} examples={train_missing_manifest[:10]}; "
            f"pt_missing_manifest={len(pt_missing_manifest)} examples={pt_missing_manifest[:10]}"
        )

    train_records = [records[sid] for sid in sorted(train_id_set)]
    vocab = build_manifest_vocab(
        train_records,
        max_permissions=int(cfg["max_permissions"]),
        max_intents=int(cfg["max_intents"]),
        max_features=int(cfg["max_features"]),
    )
    vocab["metadata"] = {
        "source_split": "train",
        "leakage_guard": "train_only",
        "num_records": len(train_records),
        "train_csv_sha256": _sha256_file(train_csv_path),
        "train_sample_ids_sha256": _sample_id_digest(train_id_set),
    }
    validate_manifest_vocab(
        vocab,
        require_train_metadata=True,
        allow_empty=bool(cfg["allow_empty_vocab"]),
    )
    build_fingerprint = _build_fingerprint(_fingerprint_config(cfg), vocab)
    legacy_build_fingerprint = _build_fingerprint(
        _fingerprint_config(cfg),
        vocab,
        pt_schema_version=PT_SCHEMA_VERSION - 1,
    )
    manifest_provenance = {
        "manifest_vocab_sha256": _vocab_digest(vocab),
        "manifest_train_csv_sha256": str(
            vocab["metadata"]["train_csv_sha256"]
        ),
        "manifest_train_sample_ids_sha256": str(
            vocab["metadata"]["train_sample_ids_sha256"]
        ),
    }
    old_vocab = _load_optional_vocab(vocab_out_path)
    previous_build_fingerprint = (
        _build_fingerprint(_fingerprint_config(cfg), old_vocab)
        if old_vocab is not None
        else None
    )
    legacy_previous_build_fingerprint = (
        _build_fingerprint(
            _fingerprint_config(cfg),
            old_vocab,
            pt_schema_version=PT_SCHEMA_VERSION - 1,
        )
        if old_vocab is not None
        else None
    )
    allowed_existing_fingerprints = {
        build_fingerprint,
        legacy_build_fingerprint,
    }
    if previous_build_fingerprint is not None:
        allowed_existing_fingerprints.add(previous_build_fingerprint)
    if legacy_previous_build_fingerprint is not None:
        allowed_existing_fingerprints.add(legacy_previous_build_fingerprint)

    before, needs_update, file_stats = _audit_pool(
        pt_files,
        records,
        vocab,
        int(cfg["manifest_dim"]),
        build_fingerprint,
        manifest_provenance,
        verify_expected=False,
        allowed_existing_fingerprints=allowed_existing_fingerprints,
    )
    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "inputs": {
            "train_csv": str(train_csv_path),
            "pt_dir": str(pt_dir_path),
            "build_config": str(build_config_path),
            "vocab_out": str(vocab_out_path),
            "manifest_jsonl_files": [str(path) for path in jsonl_files],
        },
        "coverage": {
            "train_csv_samples": len(train_ids),
            "pt_pool_samples": len(pt_files),
            "manifest_records": len(records),
            "unused_manifest_records": len(record_ids - pt_ids),
            "unused_manifest_record_examples": sorted(record_ids - pt_ids)[:10],
            "train_missing_pt": 0,
            "train_missing_manifest": 0,
            "pt_missing_manifest": 0,
        },
        "vocab": {
            "old": _vocab_summary(old_vocab) if old_vocab is not None else None,
            "new": _vocab_summary(vocab),
            "diff": _vocab_diff(old_vocab, vocab),
        },
        "proposed_build_fingerprint": build_fingerprint,
        "legacy_proposed_build_fingerprint": legacy_build_fingerprint,
        "previous_build_fingerprint": previous_build_fingerprint,
        "legacy_previous_build_fingerprint": (
            legacy_previous_build_fingerprint
        ),
        "manifest_provenance": manifest_provenance,
        "before": before,
        "apply": None,
        "after": None,
    }

    if apply:
        # A full logical preflight has completed.  Recheck every file before the
        # first write so a concurrent extractor/training job cannot be silently
        # mixed into this batch.
        _assert_files_unchanged(pt_files, file_stats)
        needs_update_set = set(needs_update)
        updated_count = 0
        for index, sid in enumerate(sorted(needs_update_set), start=1):
            path = pt_files[sid]
            _assert_file_unchanged(path, file_stats[sid])
            payload = _load_pt(path)
            _validate_current_pt(
                payload,
                path,
                sid,
                allow_legacy_permission_mapping=True,
            )
            current_fingerprint = str(
                payload["direct_build_meta"].get("build_fingerprint") or ""
            )
            if current_fingerprint not in allowed_existing_fingerprints:
                raise RuntimeError(
                    "PT build fingerprint changed after preflight; refusing "
                    "to relabel unchanged API/Graph tensors: "
                    f"path={path} fingerprint={current_fingerprint!r}"
                )
            manifest_payload = vectorize_manifest_record(
                records[sid],
                vocab,
                manifest_dim=int(cfg["manifest_dim"]),
            )
            updated = _updated_payload(
                payload,
                manifest_payload,
                build_fingerprint,
                manifest_provenance,
            )
            _validate_current_pt(updated, path, sid)
            _assert_file_unchanged(path, file_stats[sid])
            atomic_torch_save(updated, path)
            updated_count += 1
            if index % 500 == 0:
                logger.info("Updated %d/%d PT files", index, len(needs_update_set))

        # Verify the complete pool against the in-memory new vocabulary before
        # publishing it.  If a tensor write or read-back is bad, the old vocab
        # remains in place and formal training still rejects the partially
        # migrated pool.  A retry can safely accept both old/new fingerprints.
        after, remaining, _ = _audit_pool(
            pt_files,
            records,
            vocab,
            int(cfg["manifest_dim"]),
            build_fingerprint,
            manifest_provenance,
            verify_expected=True,
            certificate_root=pt_dir_path,
            allowed_existing_fingerprints={build_fingerprint},
        )
        if remaining:
            raise RuntimeError(
                f"Post-migration audit found {len(remaining)} stale PT files: {remaining[:10]}"
            )
        _atomic_write_yaml(vocab, vocab_out_path)
        report["apply"] = {
            "updated_pt_count": updated_count,
            "unchanged_pt_count": len(pt_files) - updated_count,
            "vocab_written": str(vocab_out_path),
        }
        report["after"] = after

    if audit_json is not None:
        _atomic_write_json(report, Path(audit_json).expanduser().resolve())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a train-only Manifest vocab and migrate only Manifest fields "
            "inside a unified current-schema PT pool. Defaults to dry-run."
        )
    )
    parser.add_argument("--train-csv", required=True, help="Current train split CSV.")
    parser.add_argument("--pt-dir", required=True, help="Unified pool containing all target .pt files.")
    parser.add_argument(
        "--build-config",
        required=True,
        help="Direct PT build YAML used to reproduce extraction dimensions/fingerprint settings.",
    )
    parser.add_argument(
        "--vocab-out",
        default=None,
        help="Output vocab YAML. Defaults to manifest.vocab_path in --build-config.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        action="append",
        default=[],
        help="Manifest JSONL file; may be repeated.",
    )
    parser.add_argument(
        "--manifest-jsonl-dir",
        action="append",
        default=[],
        help=(
            "Directory recursively containing Manifest JSONL files; may be repeated. "
            "Defaults to manifest.manifest_jsonl_dir in --build-config."
        ),
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional train CSV ID column. Auto-detects id, then sha256.",
    )
    parser.add_argument(
        "--audit-json",
        default=None,
        help="Optional path for the complete pre/post audit JSON.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply atomic per-PT updates. Without this flag the command is read-only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_migration(
        train_csv=args.train_csv,
        pt_dir=args.pt_dir,
        build_config=args.build_config,
        vocab_out=args.vocab_out,
        manifest_jsonl=args.manifest_jsonl,
        manifest_jsonl_dirs=args.manifest_jsonl_dir,
        id_column=args.id_column,
        apply=bool(args.apply),
        audit_json=args.audit_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
