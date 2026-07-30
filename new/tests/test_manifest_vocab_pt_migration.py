from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import migrate_manifest_vocab_pts as migration
from tests.pt_factory import current_pt_payload


def _sid(char: str) -> str:
    return char * 64


def _write_build_config(tmp_path: Path) -> Path:
    path = tmp_path / "build_pts.yaml"
    raw = {
        "data": {
            "splits": ["train", "val", "test"],
            "split_dirs": {
                "train": str(tmp_path / "apks" / "train"),
                "val": str(tmp_path / "apks" / "val"),
                "test": str(tmp_path / "apks" / "test"),
            },
            "out_root": str(tmp_path / "unused_out"),
        },
        "hyperparameters": {"vocab_size": 256},
        "graph": {
            "sensitive_hops": 1,
            "max_methods_per_dex": 64,
            "fallback_max_methods": 16,
            "fallback_policy": "api_rich",
            "use_behavior_hints": True,
        },
        "api": {
            "num_hash_buckets": 128,
            "max_events_per_dex": 64,
            "max_events_per_method": 8,
            "event_scope": "all_methods",
            "framework_only": True,
            "include_descriptor": False,
        },
        "manifest": {
            "vocab_path": str(tmp_path / "default_vocab.yaml"),
            "rebuild_vocab": False,
            "allow_empty_vocab": False,
            "manifest_dim": 32,
            "max_permissions": 2,
            "max_intents": 1,
            "max_features": 1,
            "save_manifest_jsonl": True,
            "manifest_jsonl_dir": str(tmp_path / "jsonl"),
        },
        "storage": {"keep_method_names": False, "keep_api_tokens": False},
        "execution": {
            "workers": 1,
            "resume": False,
            "allow_empty_splits": True,
            "fail_on_error": True,
        },
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _write_train_csv(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sha256", "label"])
        writer.writeheader()
        for index, sid in enumerate(ids):
            writer.writerow({"id": sid, "sha256": sid, "label": index % 2})


def _record(
    sid: str,
    *,
    permission: str,
    intent: str,
    feature: str,
) -> dict[str, object]:
    return {
        "sid": sid,
        "sha256": sid,
        "apk_name": f"{sid}.apk",
        "permissions": [permission],
        "intent_actions": [intent],
        "intent_categories": [],
        "uses_features": [feature],
        "activities": ["com.example.MainActivity"],
        "services": [],
        "receivers": [],
        "providers": [],
        "component_count": 1,
        "exported_component_count": 1,
        "min_sdk": 23,
        "target_sdk": 33,
        "debuggable": False,
        "parse_error": "",
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_pt(path: Path, sid: str, marker: float) -> None:
    dex = {
        "call_x": torch.tensor([[marker, marker + 1.0]], dtype=torch.float32),
        "call_edge_index": torch.empty((2, 0), dtype=torch.long),
        "call_sensitive_mask": torch.zeros(1, dtype=torch.uint8),
        "api_ids": torch.tensor([1, 2], dtype=torch.long),
        "api_type_ids": torch.tensor([3, 4], dtype=torch.long),
        "api_sensitive_mask": torch.ones(2, dtype=torch.uint8),
    }
    payload = current_pt_payload(dex, manifest_dim=32, sid=sid)
    # Exercise the supported fast upgrade from the immediately preceding PT
    # schema: the new token alignment is rebuilt from Manifest JSONL + vocab,
    # without reparsing the APK or touching Graph/API tensors.
    payload.pop("manifest_permission_token_ids", None)
    payload["manifest_permission_category_map"] = torch.zeros((2, 12))
    payload["manifest_intent_category_map"] = torch.zeros((1, 12))
    payload["q_manifest"] = torch.tensor([1.0])
    payload["pert_manifest"] = torch.tensor([0.0])
    payload["direct_build_meta"].update(
        {
            "pt_schema_version": max(1, migration.PT_SCHEMA_VERSION - 1),
            "sha256": sid,
            "build_fingerprint": "old-fingerprint",
            "code_marker": f"keep-{marker}",
        }
    )
    payload["observable_metadata"]["code_only_metric"] = marker
    payload["extra_code_payload"] = {
        "marker": marker,
        "tensor": torch.tensor([marker + 10.0]),
    }
    torch.save(payload, path)


def _fixture_pool(tmp_path: Path) -> tuple[Path, Path, Path, Path, str, str]:
    train_sid = _sid("a")
    heldout_sid = _sid("b")
    train_csv = tmp_path / "train.csv"
    _write_train_csv(train_csv, [train_sid])
    pt_dir = tmp_path / "pts_all"
    pt_dir.mkdir()
    _write_pt(pt_dir / f"{train_sid}.pt", train_sid, 1.0)
    _write_pt(pt_dir / f"{heldout_sid}.pt", heldout_sid, 2.0)
    jsonl = tmp_path / "jsonl" / "all.jsonl"
    _write_jsonl(
        jsonl,
        [
            _record(
                train_sid,
                permission="permission.train.only",
                intent="intent.train.only",
                feature="feature.train.only",
            ),
            _record(
                heldout_sid,
                permission="permission.heldout.only",
                intent="intent.heldout.only",
                feature="feature.heldout.only",
            ),
        ],
    )
    config = _write_build_config(tmp_path)
    return train_csv, pt_dir, jsonl, config, train_sid, heldout_sid


def _write_previous_vocab_and_bind_pts(
    config: Path,
    vocab_out: Path,
    pt_dir: Path,
) -> bytes:
    previous_vocab = {
        "permission_vocab": ["permission.legacy"],
        "intent_vocab": ["intent.legacy"],
        "feature_vocab": ["feature.legacy"],
        "categories": [f"category_{index}" for index in range(12)],
        "metadata": {
            "source_split": "train",
            "leakage_guard": "train_only",
            "num_records": 2,
        },
    }
    vocab_out.write_text(
        yaml.safe_dump(previous_vocab, sort_keys=False), encoding="utf-8"
    )
    cfg = migration._load_direct_config(config)
    previous_fingerprint = migration._build_fingerprint(
        migration._fingerprint_config(cfg),
        previous_vocab,
        pt_schema_version=migration.PT_SCHEMA_VERSION - 1,
    )
    for path in pt_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["direct_build_meta"]["build_fingerprint"] = previous_fingerprint
        torch.save(payload, path)
    return vocab_out.read_bytes()


def _declare_previous_fingerprint(config: Path, fingerprint: str) -> None:
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["migration"] = {
        "allowed_previous_build_fingerprints": [fingerprint],
    }
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )


def test_parser_defaults_to_dry_run():
    args = migration.build_parser().parse_args(
        [
            "--train-csv",
            "train.csv",
            "--pt-dir",
            "pts",
            "--build-config",
            "build.yaml",
        ]
    )
    assert args.apply is False


def test_dry_run_is_read_only_and_every_pt_load_is_weights_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    train_csv, pt_dir, jsonl, config, _, _ = _fixture_pool(tmp_path)
    vocab_out = tmp_path / "new_vocab.yaml"
    previous_vocab_bytes = _write_previous_vocab_and_bind_pts(
        config, vocab_out, pt_dir
    )
    before = {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}

    original_load = torch.load
    calls: list[bool] = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(migration.torch, "load", checked_load)
    report = migration.run_migration(
        train_csv=train_csv,
        pt_dir=pt_dir,
        build_config=config,
        vocab_out=vocab_out,
        manifest_jsonl=[jsonl],
    )

    assert report["mode"] == "dry-run"
    assert report["before"]["needs_update_count"] == 2
    assert report["after"] is None
    assert calls and all(value is True for value in calls)
    assert vocab_out.read_bytes() == previous_vocab_bytes
    assert before == {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}


def test_apply_rebuilds_vocab_from_train_only_and_preserves_code_payload(tmp_path: Path):
    train_csv, pt_dir, jsonl, config, train_sid, heldout_sid = _fixture_pool(tmp_path)
    vocab_out = tmp_path / "new_vocab.yaml"
    _write_previous_vocab_and_bind_pts(config, vocab_out, pt_dir)
    audit_path = tmp_path / "audit.json"
    original_train = torch.load(
        pt_dir / f"{train_sid}.pt", map_location="cpu", weights_only=True
    )
    original_heldout = torch.load(
        pt_dir / f"{heldout_sid}.pt", map_location="cpu", weights_only=True
    )

    report = migration.run_migration(
        train_csv=train_csv,
        pt_dir=pt_dir,
        build_config=config,
        vocab_out=vocab_out,
        manifest_jsonl_dirs=[jsonl.parent],
        apply=True,
        audit_json=audit_path,
    )

    vocab = yaml.safe_load(vocab_out.read_text(encoding="utf-8"))
    assert vocab["permission_vocab"] == ["permission.train.only"]
    assert vocab["intent_vocab"] == ["intent.train.only"]
    assert vocab["feature_vocab"] == ["feature.train.only"]
    assert vocab["metadata"]["source_split"] == "train"
    assert vocab["metadata"]["leakage_guard"] == "train_only"
    assert vocab["metadata"]["num_records"] == 1
    assert len(vocab["metadata"]["train_csv_sha256"]) == 64
    assert len(vocab["metadata"]["train_sample_ids_sha256"]) == 64

    migrated_train = torch.load(
        pt_dir / f"{train_sid}.pt", map_location="cpu", weights_only=True
    )
    migrated_heldout = torch.load(
        pt_dir / f"{heldout_sid}.pt", map_location="cpu", weights_only=True
    )
    for original, migrated in (
        (original_train, migrated_train),
        (original_heldout, migrated_heldout),
    ):
        assert torch.equal(original["dex_list"][0]["call_x"], migrated["dex_list"][0]["call_x"])
        assert torch.equal(original["dex_list"][0]["api_ids"], migrated["dex_list"][0]["api_ids"])
        assert original["extra_code_payload"]["marker"] == migrated["extra_code_payload"]["marker"]
        assert torch.equal(
            original["extra_code_payload"]["tensor"],
            migrated["extra_code_payload"]["tensor"],
        )
        assert (
            original["observable_metadata"]["code_only_metric"]
            == migrated["observable_metadata"]["code_only_metric"]
        )
        assert (
            original["direct_build_meta"]["code_marker"]
            == migrated["direct_build_meta"]["code_marker"]
        )
        assert (
            migrated["direct_build_meta"]["build_fingerprint"]
            == report["proposed_build_fingerprint"]
        )

    assert migrated_train["manifest_permission_ids"].tolist() == [1]
    assert migrated_train["manifest_permission_token_ids"].tolist() == [1]
    assert migrated_heldout["manifest_permission_ids"].numel() == 0
    assert migrated_heldout["manifest_permission_token_ids"].tolist() == [0]
    for migrated in (migrated_train, migrated_heldout):
        assert "manifest_permission_category_map" not in migrated
        assert "manifest_intent_category_map" not in migrated
        assert "q_manifest" not in migrated
        assert "pert_manifest" not in migrated
    assert migrated_heldout["manifest_meta"]["permissions"] == ["permission.heldout.only"]
    assert (
        migrated_train["direct_build_meta"]["pt_schema_version"]
        == migration.PT_SCHEMA_VERSION
    )
    assert (
        migrated_heldout["direct_build_meta"]["pt_schema_version"]
        == migration.PT_SCHEMA_VERSION
    )
    assert migrated_heldout["observable_metadata"]["manifest_vocab_coverage"] == pytest.approx(0.0)
    assert report["apply"] == {
        "updated_pt_count": 2,
        "unchanged_pt_count": 0,
        "vocab_written": str(vocab_out.resolve()),
    }
    assert report["after"]["needs_update_count"] == 0
    certificate = report["after"]["pt_audit_certificate"]
    assert certificate["pt_schema_version"] == migration.PT_SCHEMA_VERSION
    assert certificate["pool_root"] == str(pt_dir.resolve())
    assert certificate["build_fingerprint"] == report["proposed_build_fingerprint"]
    assert [entry["sid"] for entry in certificate["entries"]] == sorted(
        [train_sid, heldout_sid]
    )
    assert all(
        set(entry)
        == {
            "sid",
            "relative_path",
            "size",
            "mtime_ns",
            "ctime_ns",
            "call_x_dims",
        }
        for entry in certificate["entries"]
    )
    assert certificate["certificate_sha256"] == migration.pt_audit_entries_sha256(
        certificate["entries"]
    )
    assert json.loads(audit_path.read_text(encoding="utf-8"))["after"]["needs_update_count"] == 0
    assert not list(pt_dir.rglob("*.tmp"))


def test_incompatible_code_fingerprint_hard_fails_before_write(tmp_path: Path):
    train_csv, pt_dir, jsonl, config, _, _ = _fixture_pool(tmp_path)
    vocab_out = tmp_path / "new_vocab.yaml"
    previous_vocab_bytes = _write_previous_vocab_and_bind_pts(
        config, vocab_out, pt_dir
    )
    first_pt = next(pt_dir.glob("*.pt"))
    payload = torch.load(first_pt, map_location="cpu", weights_only=True)
    payload["direct_build_meta"]["build_fingerprint"] = "f" * 64
    torch.save(payload, first_pt)
    before = {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}

    with pytest.raises(RuntimeError, match="refusing to relabel unchanged API/Graph"):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=config,
            vocab_out=vocab_out,
            manifest_jsonl=[jsonl],
            apply=True,
        )

    assert vocab_out.read_bytes() == previous_vocab_bytes
    assert before == {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}


def test_declared_previous_fingerprint_allows_manifest_only_lineage_migration(
    tmp_path: Path,
):
    train_csv, pt_dir, jsonl, config, _, _ = _fixture_pool(tmp_path)
    vocab_out = tmp_path / "new_vocab.yaml"
    _write_previous_vocab_and_bind_pts(config, vocab_out, pt_dir)
    first_pt = next(pt_dir.glob("*.pt"))
    previous_fingerprint = str(
        torch.load(
            first_pt,
            map_location="cpu",
            weights_only=True,
        )["direct_build_meta"]["build_fingerprint"]
    )

    # Reproduce deployment after the repository has already installed the new
    # train-only vocab: that file can no longer be used to derive the PTs'
    # previous combined fingerprint.
    installed_vocab = yaml.safe_load(vocab_out.read_text(encoding="utf-8"))
    installed_vocab["permission_vocab"] = ["permission.replacement"]
    vocab_out.write_text(
        yaml.safe_dump(installed_vocab, sort_keys=False),
        encoding="utf-8",
    )
    installed_vocab_bytes = vocab_out.read_bytes()
    before = {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}

    with pytest.raises(
        RuntimeError,
        match="declared migration lineage",
    ):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=config,
            vocab_out=vocab_out,
            manifest_jsonl=[jsonl],
            apply=True,
        )
    assert before == {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}
    assert vocab_out.read_bytes() == installed_vocab_bytes

    target_vocab = yaml.safe_load(vocab_out.read_text(encoding="utf-8"))
    cfg_without_lineage = migration._load_direct_config(config)
    target_fingerprint_without_lineage = migration._build_fingerprint(
        migration._fingerprint_config(cfg_without_lineage),
        target_vocab,
    )
    _declare_previous_fingerprint(config, previous_fingerprint)
    cfg_with_lineage = migration._load_direct_config(config)
    target_fingerprint_with_lineage = migration._build_fingerprint(
        migration._fingerprint_config(cfg_with_lineage),
        target_vocab,
    )
    assert target_fingerprint_with_lineage == target_fingerprint_without_lineage

    report = migration.run_migration(
        train_csv=train_csv,
        pt_dir=pt_dir,
        build_config=config,
        vocab_out=vocab_out,
        manifest_jsonl=[jsonl],
        apply=True,
    )
    assert report["declared_previous_build_fingerprints"] == [
        previous_fingerprint
    ]
    assert report["after"]["build_fingerprint_counts"] == {
        report["proposed_build_fingerprint"]: 2
    }
    for path in pt_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert (
            payload["direct_build_meta"]["build_fingerprint"]
            == report["proposed_build_fingerprint"]
        )


@pytest.mark.parametrize(
    "value",
    [
        "not-a-digest",
        "A" * 64,
        ["f" * 64, "f" * 64],
    ],
)
def test_declared_previous_fingerprint_config_is_strict(
    tmp_path: Path,
    value,
):
    config = _write_build_config(tmp_path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["migration"] = {
        "allowed_previous_build_fingerprints": (
            value if isinstance(value, list) else [value]
        )
    }
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allowed_previous_build_fingerprints"):
        migration._load_direct_config(config)


@pytest.mark.parametrize(
    ("migration_settings", "message"),
    [
        ([], "migration settings must be a mapping"),
        (
            {"unknown_setting": []},
            "Unsupported migration settings",
        ),
        (
            {"allowed_previous_build_fingerprints": "f" * 64},
            "allowed_previous_build_fingerprints must be a list",
        ),
    ],
)
def test_migration_settings_schema_is_strict(
    tmp_path: Path,
    migration_settings,
    message: str,
):
    config = _write_build_config(tmp_path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["migration"] = migration_settings
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        migration._load_direct_config(config)


def test_late_v4_observable_failure_hard_fails_before_any_write(tmp_path: Path):
    train_csv, pt_dir, jsonl, config, _, heldout_sid = _fixture_pool(tmp_path)
    vocab_out = tmp_path / "new_vocab.yaml"
    previous_vocab_bytes = _write_previous_vocab_and_bind_pts(
        config, vocab_out, pt_dir
    )
    # The held-out SID sorts after the train SID, so this catches a
    # write-as-you-audit implementation that would migrate the first PT before
    # discovering an invalid observable value in the second.
    late_pt = pt_dir / f"{heldout_sid}.pt"
    payload = torch.load(late_pt, map_location="cpu", weights_only=True)
    payload["observable_metadata"]["api_parse_ok"] = "false"
    torch.save(payload, late_pt)
    before = {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}

    with pytest.raises(
        ValueError,
        match=r"observable_metadata\.api_parse_ok must be bool or numeric 0/1",
    ):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=config,
            vocab_out=vocab_out,
            manifest_jsonl=[jsonl],
            apply=True,
        )

    assert vocab_out.read_bytes() == previous_vocab_bytes
    assert before == {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}


def test_current_schema_missing_permission_alignment_is_not_treated_as_legacy(
    tmp_path: Path,
):
    train_csv, pt_dir, jsonl, config, _, _ = _fixture_pool(tmp_path)
    vocab_out = tmp_path / "new_vocab.yaml"
    _write_previous_vocab_and_bind_pts(config, vocab_out, pt_dir)
    first_pt = next(pt_dir.glob("*.pt"))
    payload = torch.load(first_pt, map_location="cpu", weights_only=True)
    payload["direct_build_meta"]["pt_schema_version"] = (
        migration.PT_SCHEMA_VERSION
    )
    torch.save(payload, first_pt)
    before = {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}

    with pytest.raises(
        ValueError,
        match="schema-5 is missing top-level fields.*manifest_permission_token_ids",
    ):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=config,
            vocab_out=vocab_out,
            manifest_jsonl=[jsonl],
            apply=True,
        )

    assert before == {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}


def test_missing_manifest_hard_fails_before_any_write(tmp_path: Path):
    train_csv, pt_dir, jsonl, config, train_sid, _ = _fixture_pool(tmp_path)
    # Deliberately remove the held-out Manifest record while retaining its PT.
    _write_jsonl(
        jsonl,
        [
            _record(
                train_sid,
                permission="permission.train.only",
                intent="intent.train.only",
                feature="feature.train.only",
            )
        ],
    )
    vocab_out = tmp_path / "must_not_exist.yaml"
    before = {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}

    with pytest.raises(RuntimeError, match="pt_missing_manifest=1"):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=config,
            vocab_out=vocab_out,
            manifest_jsonl=[jsonl],
            apply=True,
        )

    assert not vocab_out.exists()
    assert before == {path.name: path.read_bytes() for path in pt_dir.glob("*.pt")}


def test_train_sample_missing_pt_hard_fails(tmp_path: Path):
    train_csv, pt_dir, jsonl, config, train_sid, heldout_sid = _fixture_pool(tmp_path)
    missing_sid = _sid("c")
    _write_train_csv(train_csv, [train_sid, missing_sid])
    records = [
        _record(
            train_sid,
            permission="permission.train.only",
            intent="intent.train.only",
            feature="feature.train.only",
        ),
        _record(
            heldout_sid,
            permission="permission.heldout.only",
            intent="intent.heldout.only",
            feature="feature.heldout.only",
        ),
        _record(
            missing_sid,
            permission="permission.missing",
            intent="intent.missing",
            feature="feature.missing",
        ),
    ]
    _write_jsonl(jsonl, records)

    with pytest.raises(RuntimeError, match="train_missing_pt=1"):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=config,
            vocab_out=tmp_path / "must_not_exist.yaml",
            manifest_jsonl=[jsonl],
            apply=True,
        )
