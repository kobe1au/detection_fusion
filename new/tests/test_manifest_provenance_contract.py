from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from fusion.dataset import FatalDatasetConfigError, RobustTriModalDataset
from fusion.manifest_features import load_manifest_vocab
from scripts.build_tri_modal_pts_direct import (
    _build_fingerprint,
    _fingerprint_config,
    _load_direct_config,
)
from scripts import migrate_manifest_vocab_pts as migration
from tests.pt_factory import save_current_pt


EXPECTED_PROVENANCE = {
    "manifest_vocab_sha256": "a" * 64,
    "manifest_train_csv_sha256": "b" * 64,
    "manifest_train_sample_ids_sha256": "c" * 64,
}

ROOT = Path(__file__).resolve().parents[1]


def _sid(char: str) -> str:
    return char * 64


def _dex_payload(marker: float) -> dict[str, torch.Tensor]:
    return {
        "call_x": torch.tensor(
            [[marker, marker + 1.0], [marker + 2.0, marker + 3.0]],
            dtype=torch.float32,
        ),
        "call_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
        "api_ids": torch.tensor([10, 20], dtype=torch.long),
        "api_type_ids": torch.tensor([1, 6], dtype=torch.long),
        "api_sensitive_mask": torch.ones(2, dtype=torch.float32),
        "api_method_index": torch.tensor([0, 1], dtype=torch.long),
        "api_in_graph_mask": torch.ones(2, dtype=torch.float32),
        "method_api_edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    }


def _write_current_pt(
    path: Path,
    *,
    marker: float,
    provenance: dict[str, str] = EXPECTED_PROVENANCE,
) -> None:
    save_current_pt(path, _dex_payload(marker), manifest_dim=32)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["direct_build_meta"].update(provenance)
    torch.save(payload, path)


def _write_csv(path: Path, sample_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "year"])
        writer.writeheader()
        for index, sid in enumerate(sample_ids):
            writer.writerow({"id": sid, "label": index % 2, "year": 2024})


def _dataset(pt_dir: Path, csv_path: Path) -> RobustTriModalDataset:
    return RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        expected_manifest_vocab_sha256=EXPECTED_PROVENANCE[
            "manifest_vocab_sha256"
        ],
        expected_manifest_train_csv_sha256=EXPECTED_PROVENANCE[
            "manifest_train_csv_sha256"
        ],
        expected_manifest_train_sample_ids_sha256=EXPECTED_PROVENANCE[
            "manifest_train_sample_ids_sha256"
        ],
    )


def test_later_pt_provenance_mismatch_fails_during_split_preflight(tmp_path: Path):
    """A bad later PT must fail before any sample reaches the model."""
    first_sid = _sid("1")
    second_sid = _sid("2")
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    _write_current_pt(pt_dir / f"{first_sid}.pt", marker=1.0)
    stale = dict(EXPECTED_PROVENANCE)
    stale["manifest_vocab_sha256"] = "d" * 64
    _write_current_pt(
        pt_dir / f"{second_sid}.pt",
        marker=2.0,
        provenance=stale,
    )
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, [first_sid, second_sid])

    with pytest.raises(
        FatalDatasetConfigError,
        match="PT Manifest provenance does not match",
    ):
        _dataset(pt_dir, csv_path)


def test_expected_pt_build_fingerprint_is_strict(tmp_path: Path):
    sid = _sid("f")
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    _write_current_pt(pt_dir / f"{sid}.pt", marker=1.0)
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, [sid])

    with pytest.raises(FatalDatasetConfigError, match="build fingerprint"):
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=32,
            expected_pt_build_fingerprint="f" * 64,
        )


def test_main_config_pt_fingerprint_matches_canonical_build_inputs():
    build_cfg = _load_direct_config(ROOT / "config" / "build_pts.yaml")
    vocab = load_manifest_vocab(
        ROOT / "config" / "manifest_vocab.yaml",
        require_train_metadata=True,
        allow_empty=False,
    )
    actual = _build_fingerprint(_fingerprint_config(build_cfg), vocab)
    with (ROOT / "config" / "experiments" / "tri_modal_robust" /
          "base_tri_modal_robust.yaml").open(encoding="utf-8") as handle:
        expected = str(
            (yaml.safe_load(handle) or {})["data"][
                "expected_pt_build_fingerprint"
            ]
        )

    assert expected == actual


def test_renamed_pt_is_rejected_by_migration_and_dataset(tmp_path: Path):
    source_sid = _sid("a")
    renamed_sid = _sid("b")
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    source = pt_dir / f"{source_sid}.pt"
    renamed = pt_dir / f"{renamed_sid}.pt"
    _write_current_pt(source, marker=3.0)
    source.rename(renamed)

    payload = torch.load(renamed, map_location="cpu", weights_only=True)
    with pytest.raises(
        ValueError,
        match=r"direct_build_meta\.sha256 identity mismatch",
    ):
        migration._validate_current_pt(payload, renamed, renamed_sid)

    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, [renamed_sid])
    with pytest.raises(FatalDatasetConfigError, match="identity mismatch"):
        _dataset(pt_dir, csv_path)


def test_per_file_stat_guard_rejects_post_preflight_change(tmp_path: Path):
    sid = _sid("3")
    path = tmp_path / f"{sid}.pt"
    _write_current_pt(path, marker=4.0)
    before = path.stat()
    expected = (int(before.st_size), int(before.st_mtime_ns))

    # Change only mtime so the file remains a readable current-schema PT.
    os.utime(
        path,
        ns=(int(before.st_atime_ns), int(before.st_mtime_ns) + 2_000_000_000),
    )

    with pytest.raises(RuntimeError, match="changed after migration preflight"):
        migration._assert_file_unchanged(path, expected)


def test_migration_rejects_train_identity_missing_from_pt_pool(tmp_path: Path):
    present_sid = _sid("4")
    missing_sid = _sid("5")
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    _write_current_pt(pt_dir / f"{present_sid}.pt", marker=5.0)
    train_csv = tmp_path / "train.csv"
    _write_csv(train_csv, [present_sid, missing_sid])

    jsonl = tmp_path / "manifest.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for sid in (present_sid, missing_sid):
            handle.write(json.dumps({"sid": sid, "sha256": sid}) + "\n")

    build_config = tmp_path / "build.yaml"
    build_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "splits": ["train"],
                    "split_dirs": {"train": str(tmp_path / "unused_apks")},
                    "out_root": str(tmp_path / "unused_out"),
                },
                "manifest": {
                    "vocab_path": str(tmp_path / "old_vocab.yaml"),
                    "manifest_jsonl_dir": str(tmp_path),
                    "manifest_dim": 256,
                    "max_permissions": 128,
                    "max_intents": 64,
                    "max_features": 32,
                },
                "execution": {
                    "resume": False,
                    "allow_empty_splits": True,
                    "workers": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="train_missing_pt=1"):
        migration.run_migration(
            train_csv=train_csv,
            pt_dir=pt_dir,
            build_config=build_config,
            vocab_out=tmp_path / "old_vocab.yaml",
            manifest_jsonl=[jsonl],
        )
