from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract.extract_graph_api import atomic_torch_save, process_apk, sha256_file
from fusion.manifest_features import (
    build_manifest_vocab,
    extract_manifest_record,
    load_manifest_vocab,
    save_manifest_vocab,
    validate_manifest_vocab,
    vectorize_manifest_record,
)
from fusion.quality import (
    OBSERVABLE_REQUIRED_FIELDS,
    OBSERVABLE_SCHEMA_VERSION,
)


PT_SCHEMA_VERSION = 4
logger = logging.getLogger("build_tri_modal_pts_direct")


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _build_fingerprint(cfg: dict, vocab: dict) -> str:
    payload = {
        "pt_schema_version": PT_SCHEMA_VERSION,
        "observable_schema_version": OBSERVABLE_SCHEMA_VERSION,
        "config": _canonical(cfg),
        "manifest_vocab": _canonical(vocab),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_unique_hashes(jobs: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for job in jobs:
        sha = str(job.get("sha256") or "").strip().lower()
        if not sha:
            continue
        if sha in seen:
            duplicates.append(sha)
        else:
            seen[sha] = str(job.get("split") or "")
    if duplicates:
        raise RuntimeError(
            f"Duplicate APK hashes in extraction jobs: count={len(set(duplicates))}, "
            f"examples={sorted(set(duplicates))[:10]}"
        )


def _resume_existing(job: dict[str, Any], cfg: dict, build_fingerprint: str) -> tuple[bool, dict[str, Any]]:
    split = str(job.get("split") or "")
    sha = str(job.get("sha256") or "").strip().lower()
    out_dir = Path((cfg.get("out_dirs") or {})[split])
    path = out_dir / f"{sha}.pt"
    row = {"split": split, "sha256": sha, "path": str(path)}
    if not bool(cfg.get("resume", False)) or not path.exists():
        return False, {**row, "status": "pending"}
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        meta = raw.get("direct_build_meta", {}) if isinstance(raw, dict) else {}
        version = int(meta.get("pt_schema_version", 0)) if isinstance(meta, dict) else 0
        fingerprint = str(meta.get("build_fingerprint") or "") if isinstance(meta, dict) else ""
        observable = raw.get("observable_metadata", {}) if isinstance(raw, dict) else {}
        observable_complete = (
            isinstance(observable, dict)
            and observable.get("schema_version") == OBSERVABLE_SCHEMA_VERSION
            and all(key in observable for key in OBSERVABLE_REQUIRED_FIELDS)
        )
        if version == PT_SCHEMA_VERSION and fingerprint == build_fingerprint and observable_complete:
            return True, {**row, "status": "ok"}
        if bool(cfg.get("allow_legacy_resume", False)):
            return True, {**row, "status": "ok", "compatibility": "legacy"}
        return False, {**row, "status": "failed", "reason": "schema or build fingerprint mismatch"}
    except Exception as exc:
        return False, {**row, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def merge_observable_metadata(*parts: dict[str, Any]) -> dict[str, Any]:
    """Merge code and Manifest metadata into one strict sample-level schema."""
    merged: dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict):
            value = part.get("observable_metadata", part)
            if isinstance(value, dict):
                merged.update(value)
    merged["schema_version"] = OBSERVABLE_SCHEMA_VERSION
    missing = [key for key in OBSERVABLE_REQUIRED_FIELDS if key not in merged]
    if missing:
        raise ValueError(f"Cannot build {OBSERVABLE_SCHEMA_VERSION} PT; missing observable fields: {missing}")
    return merged


def build_observable_payload(
    dex_list: list[dict[str, Any]],
    code_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    *,
    build_fingerprint: str,
) -> dict[str, Any]:
    """Build the top-level structure required by strict_observable_schema."""
    observable = merge_observable_metadata(code_payload, manifest_payload)
    payload = {
        **manifest_payload,
        "dex_list": dex_list,
        "observable_metadata": observable,
        "direct_build_meta": {
            **dict(code_payload.get("direct_build_meta") or {}),
            "pt_schema_version": PT_SCHEMA_VERSION,
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
            "build_fingerprint": str(build_fingerprint),
        },
    }
    return payload


def _load_direct_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    data = raw.get("data") or {}
    hyperparameters = raw.get("hyperparameters") or {}
    graph = raw.get("graph") or {}
    api = raw.get("api") or {}
    manifest = raw.get("manifest") or {}
    storage = raw.get("storage") or {}
    execution = raw.get("execution") or {}

    splits = [str(value) for value in data.get("splits", [])]
    split_dirs = {str(key): Path(value) for key, value in (data.get("split_dirs") or {}).items()}
    if not splits:
        splits = list(split_dirs)
    if not splits or any(split not in split_dirs for split in splits):
        raise ValueError("data.splits and data.split_dirs must define every requested split")

    out_root = Path(data.get("out_root") or "pts_tri")
    configured_out_dirs = data.get("out_dirs") or {}
    out_dirs = {
        split: Path(configured_out_dirs.get(split) or out_root / split)
        for split in splits
    }
    manifest_vocab_path = Path(manifest.get("vocab_path") or out_root / "manifest_vocab.yaml")
    manifest_jsonl_dir = Path(manifest.get("manifest_jsonl_dir") or out_root / "_manifest_jsonl")
    failed_json = Path(execution.get("failed_json") or out_root / "failed_tri_modal_direct.json")

    cfg = {
        "splits": splits,
        "split_dirs": split_dirs,
        "out_root": out_root,
        "out_dirs": out_dirs,
        "vocab_size": int(hyperparameters.get("vocab_size", 256)),
        "sensitive_hops": int(graph.get("sensitive_hops", 1)),
        "max_methods_per_dex": int(graph.get("max_methods_per_dex", 4096)),
        "fallback_max_methods": int(graph.get("fallback_max_methods", 512)),
        "fallback_policy": str(graph.get("fallback_policy", "api_rich")),
        "use_graph_behavior_hints": bool(graph.get("use_behavior_hints", False)),
        "num_api_buckets": int(api.get("num_hash_buckets", 8192)),
        "max_api_events_per_dex": int(api.get("max_events_per_dex", 1024)),
        "max_api_events_per_method": int(api.get("max_events_per_method", 32)),
        "api_event_scope": str(api.get("event_scope", "all_methods")),
        "framework_only": bool(api.get("framework_only", True)),
        "include_descriptor": bool(api.get("include_descriptor", False)),
        "manifest_vocab_path": manifest_vocab_path,
        "rebuild_vocab": bool(manifest.get("rebuild_vocab", False)),
        "allow_empty_vocab": bool(manifest.get("allow_empty_vocab", False)),
        "manifest_dim": int(manifest.get("manifest_dim", 256)),
        "max_permissions": int(manifest.get("max_permissions", 128)),
        "max_intents": int(manifest.get("max_intents", 64)),
        "max_features": int(manifest.get("max_features", 32)),
        "save_manifest_jsonl": bool(manifest.get("save_manifest_jsonl", True)),
        "manifest_jsonl_dir": manifest_jsonl_dir,
        "keep_method_names": bool(storage.get("keep_method_names", False)),
        "keep_api_tokens": bool(storage.get("keep_api_tokens", False)),
        "workers": max(1, int(execution.get("workers", 1))),
        "resume": bool(execution.get("resume", False)),
        "allow_legacy_resume": bool(execution.get("allow_legacy_resume", False)),
        "allow_empty_splits": bool(execution.get("allow_empty_splits", False)),
        "fail_on_error": bool(execution.get("fail_on_error", False)),
        "failed_json": failed_json,
    }
    if cfg["rebuild_vocab"] and cfg["resume"]:
        raise ValueError("manifest.rebuild_vocab=true is incompatible with execution.resume=true")
    return cfg


def _fingerprint_config(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "vocab_size",
        "sensitive_hops",
        "max_methods_per_dex",
        "fallback_max_methods",
        "fallback_policy",
        "use_graph_behavior_hints",
        "num_api_buckets",
        "max_api_events_per_dex",
        "max_api_events_per_method",
        "api_event_scope",
        "framework_only",
        "include_descriptor",
        "manifest_dim",
        "max_permissions",
        "max_intents",
        "max_features",
        "keep_method_names",
        "keep_api_tokens",
    )
    return {key: cfg[key] for key in keys}


def _collect_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    apk_items: list[tuple[str, Path]] = []

    for split in cfg["splits"]:
        split_dir = Path(cfg["split_dirs"][split])
        paths = sorted(split_dir.glob("*.apk")) if split_dir.exists() else []

        if not paths and not cfg["allow_empty_splits"]:
            raise RuntimeError(f"No APK files found for split={split!r} in {split_dir}")

        logger.info("Found %d APK(s) for split=%r in %s", len(paths), split, split_dir)
        apk_items.extend((split, apk_path) for apk_path in paths)

    logger.info("Hashing %d APK file(s) before extraction", len(apk_items))

    for split, apk_path in tqdm(
        apk_items,
        desc="Hash APKs",
        unit="apk",
        dynamic_ncols=True,
    ):
        jobs.append(
            {
                "split": split,
                "apk_path": str(apk_path),
                "sha256": sha256_file(apk_path).lower(),
            }
        )

    _validate_unique_hashes(jobs)
    return jobs


def _extract_manifest_records(
    jobs: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in cfg["splits"]}
    for job in tqdm(jobs, desc="Extract Manifest", unit="apk"):
        record = extract_manifest_record(job["apk_path"], sid=job["sha256"]).to_json()
        record["sha256"] = job["sha256"]
        records[job["sha256"]] = record
        by_split[job["split"]].append(record)

    if cfg["save_manifest_jsonl"]:
        jsonl_dir = Path(cfg["manifest_jsonl_dir"])
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        for split, split_records in by_split.items():
            with open(jsonl_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
                for record in split_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _load_or_build_vocab(
    records: dict[str, dict[str, Any]],
    jobs: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    vocab_path = Path(cfg["manifest_vocab_path"])
    if cfg["rebuild_vocab"] or not vocab_path.exists():
        train_records = [
            records[job["sha256"]]
            for job in jobs
            if job["split"] == "train"
        ]
        if not train_records:
            raise RuntimeError("Manifest vocabulary must be built from a non-empty train split")
        vocab = build_manifest_vocab(
            train_records,
            max_permissions=cfg["max_permissions"],
            max_intents=cfg["max_intents"],
            max_features=cfg["max_features"],
        )
        vocab["metadata"] = {
            "source_split": "train",
            "leakage_guard": "train_only",
            "num_records": len(train_records),
        }
        validate_manifest_vocab(
            vocab,
            require_train_metadata=True,
            allow_empty=cfg["allow_empty_vocab"],
        )
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        save_manifest_vocab(vocab, vocab_path)
        return vocab
    return load_manifest_vocab(
        vocab_path,
        require_train_metadata=True,
        allow_empty=cfg["allow_empty_vocab"],
    )


def _build_one(
    job: dict[str, Any],
    manifest_record: dict[str, Any],
    cfg: dict[str, Any],
    vocab: dict[str, Any],
    build_fingerprint: str,
) -> dict[str, Any]:
    resumed, row = _resume_existing(job, cfg, build_fingerprint)
    if resumed:
        return row

    split = job["split"]
    sha = job["sha256"]
    out_dir = Path(cfg["out_dirs"][split])
    out_dir.mkdir(parents=True, exist_ok=True)
    code_cfg = dict(cfg)
    code_cfg["resume"] = False
    ok, reason = process_apk(Path(job["apk_path"]), out_dir, split, code_cfg)
    if not ok:
        return {**row, "status": "failed", "reason": reason}

    path = out_dir / f"{sha}.pt"
    try:
        code_payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(code_payload, dict) or not isinstance(code_payload.get("dex_list"), list):
            raise ValueError("Graph/API extractor did not produce a current top-level dex_list payload")
        manifest_payload = vectorize_manifest_record(
            manifest_record,
            vocab,
            manifest_dim=cfg["manifest_dim"],
        )
        payload = build_observable_payload(
            code_payload["dex_list"],
            code_payload,
            manifest_payload,
            build_fingerprint=build_fingerprint,
        )
        payload["direct_build_meta"].update(
            {
                "split": split,
                "sha256": sha,
                "apk_name": Path(job["apk_path"]).name,
            }
        )
        atomic_torch_save(payload, path)
        return {**row, "status": "ok"}
    except Exception as exc:
        return {**row, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _write_index(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "sha256", "path", "status", "reason", "compatibility"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_direct_config(Path(config_path))
    Path(cfg["out_root"]).mkdir(parents=True, exist_ok=True)
    for out_dir in cfg["out_dirs"].values():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    jobs = _collect_jobs(cfg)
    records = _extract_manifest_records(jobs, cfg)
    vocab = _load_or_build_vocab(records, jobs, cfg)
    build_fingerprint = _build_fingerprint(_fingerprint_config(cfg), vocab)
    rows: list[dict[str, Any]] = []

    if cfg["workers"] == 1:
        for job in tqdm(jobs, desc="Build tri-modal PT", unit="apk"):
            rows.append(_build_one(job, records[job["sha256"]], cfg, vocab, build_fingerprint))
    else:
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as executor:
            futures = {
                executor.submit(
                    _build_one,
                    job,
                    records[job["sha256"]],
                    cfg,
                    vocab,
                    build_fingerprint,
                ): job
                for job in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Build tri-modal PT", unit="apk"):
                job = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            "split": job["split"],
                            "sha256": job["sha256"],
                            "path": str(Path(cfg["out_dirs"][job["split"]]) / f"{job['sha256']}.pt"),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    rows.sort(key=lambda row: (str(row.get("split", "")), str(row.get("sha256", ""))))
    _write_index(rows, Path(cfg["out_root"]) / "tri_modal_direct_index.csv")
    failed = [row for row in rows if row.get("status") != "ok"]
    Path(cfg["failed_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["failed_json"], "w", encoding="utf-8") as handle:
        json.dump(failed, handle, ensure_ascii=False, indent=2)
    summary = {
        "total": len(rows),
        "ok": len(rows) - len(failed),
        "failed": len(failed),
        "pt_schema_version": PT_SCHEMA_VERSION,
        "observable_schema_version": OBSERVABLE_SCHEMA_VERSION,
        "build_fingerprint": build_fingerprint,
    }
    if failed and cfg["fail_on_error"]:
        raise RuntimeError(f"Tri-modal extraction failed for {len(failed)} APK(s)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict graph + API + Manifest PT files")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
