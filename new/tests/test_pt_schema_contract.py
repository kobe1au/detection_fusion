from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from fusion.dataset import (
    FatalDatasetConfigError,
    RobustTriModalDataset,
    robust_collate_fn,
)
from fusion.pt_schema import (
    PT_AUDIT_CERTIFICATE_VERSION,
    PT_SCHEMA_VERSION,
    PTSchemaValidationError,
    pt_audit_entries_sha256,
    validate_current_pt_payload,
)
from scripts.build_tri_modal_pts_direct import _resume_existing
from tests.pt_factory import current_pt_payload, save_current_pt


def _valid_dex(*, nodes: int = 2, events: int = 2) -> dict[str, torch.Tensor]:
    return {
        "call_x": torch.ones((nodes, 8), dtype=torch.float16),
        "call_edge_index": (
            torch.tensor([[0], [1]], dtype=torch.int32)
            if nodes >= 2
            else torch.empty((2, 0), dtype=torch.int32)
        ),
        "call_sensitive_mask": torch.zeros((nodes,), dtype=torch.uint8),
        "api_ids": torch.arange(2, 2 + events, dtype=torch.long),
        "api_type_ids": torch.ones((events,), dtype=torch.uint8),
        "api_sensitive_mask": torch.zeros((events,), dtype=torch.uint8),
    }


def _payload(dex: dict[str, torch.Tensor], sid: str = "sample"):
    return current_pt_payload(dex, manifest_dim=32, sid=sid)


def test_current_schema_accepts_explicit_empty_graph_and_api_tensors():
    dex = _valid_dex(nodes=0, events=0)
    payload = _payload(dex)

    dex_list, sources = validate_current_pt_payload(payload, "sample.pt")

    assert dex_list[0]["call_x"].shape == (0, 8)
    assert dex_list[0]["call_edge_index"].shape == (2, 0)
    assert dex_list[0]["api_ids"].shape == (0,)
    assert sources == [payload, dex_list[0]]


def test_current_schema_rejects_missing_dex_field_instead_of_filling_it():
    payload = _payload(_valid_dex())
    del payload["dex_list"][0]["api_type_ids"]

    with pytest.raises(
        PTSchemaValidationError,
        match=r"missing required fields .*api_type_ids.*dex_index=0",
    ):
        validate_current_pt_payload(payload, "sample.pt")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "call_x",
            torch.tensor([[float("nan")] * 8]),
            "call_x contains non-finite",
        ),
        (
            "call_edge_index",
            torch.tensor([[0.0], [1.0]]),
            "call_edge_index must use an integer",
        ),
        (
            "call_edge_index",
            torch.tensor([[0], [2]], dtype=torch.long),
            "outside",
        ),
        (
            "call_sensitive_mask",
            torch.tensor([0], dtype=torch.uint8),
            "length must equal",
        ),
        (
            "api_ids",
            torch.tensor([2, -1], dtype=torch.long),
            "api_ids must be non-negative",
        ),
        (
            "api_type_ids",
            torch.tensor([1, 16], dtype=torch.long),
            r"within \[0,15\]",
        ),
        (
            "api_sensitive_mask",
            torch.tensor([0, 2], dtype=torch.uint8),
            "only 0/1",
        ),
    ],
)
def test_current_schema_rejects_corrupt_dex_tensors(
    field: str,
    value: torch.Tensor,
    message: str,
):
    payload = _payload(_valid_dex())
    payload["dex_list"][0][field] = value

    with pytest.raises(PTSchemaValidationError, match=message):
        validate_current_pt_payload(payload, "sample.pt")


def test_current_schema_rejects_misaligned_api_vectors():
    payload = _payload(_valid_dex())
    payload["dex_list"][0]["api_type_ids"] = torch.tensor([1], dtype=torch.uint8)

    with pytest.raises(PTSchemaValidationError, match="equal lengths"):
        validate_current_pt_payload(payload, "sample.pt")


def test_current_schema_rejects_truthy_string_observable_boolean():
    payload = _payload(_valid_dex())
    payload["observable_metadata"]["api_parse_ok"] = "false"

    with pytest.raises(PTSchemaValidationError, match="must be bool or numeric 0/1"):
        validate_current_pt_payload(payload, "sample.pt")


def test_current_schema_rejects_nonfinite_or_inconsistent_manifest_layout():
    payload = _payload(_valid_dex())
    payload["manifest_x"][-1] = float("nan")
    with pytest.raises(PTSchemaValidationError, match="manifest_x.*finite"):
        validate_current_pt_payload(payload, "sample.pt")

    payload = _payload(_valid_dex())
    category_start = (
        payload["manifest_permission_dim"]
        + payload["manifest_intent_dim"]
        + payload["manifest_feature_dim"]
    )
    payload["manifest_x"][category_start] = 0.9
    with pytest.raises(PTSchemaValidationError, match="category segment disagrees"):
        validate_current_pt_payload(payload, "sample.pt")


@pytest.mark.parametrize("field", ["q_manifest", "pert_manifest"])
def test_current_schema_rejects_retired_manifest_quality_fields(field: str):
    payload = _payload(_valid_dex())
    payload[field] = torch.tensor([0.0])

    with pytest.raises(PTSchemaValidationError, match="retired top-level fields"):
        validate_current_pt_payload(payload, "sample.pt")


def test_dataset_preflight_reports_later_corrupt_pt_before_iteration(
    tmp_path: Path,
):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    save_current_pt(pt_dir / "good.pt", _valid_dex(), manifest_dim=32)
    save_current_pt(pt_dir / "bad.pt", _valid_dex(), manifest_dim=32)
    bad_path = pt_dir / "bad.pt"
    bad = torch.load(bad_path, map_location="cpu", weights_only=True)
    bad["dex_list"][0]["call_edge_index"] = torch.tensor(
        [[0], [99]],
        dtype=torch.long,
    )
    torch.save(bad, bad_path)

    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "good", "label": 0})
        writer.writerow({"id": "bad", "label": 1})

    with pytest.raises(FatalDatasetConfigError) as exc_info:
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=32,
        )
    message = str(exc_info.value)
    assert "PT preflight failed before model construction" in message
    assert "sid=bad" in message
    assert str(bad_path) in message
    assert "outside" in message


def test_builder_resume_revalidates_dex_tensor_contract(tmp_path: Path):
    out_dir = tmp_path / "pts"
    out_dir.mkdir()
    sid = "a" * 64
    path = out_dir / f"{sid}.pt"
    save_current_pt(path, _valid_dex(), manifest_dim=32)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fingerprint = "f" * 64
    payload["direct_build_meta"]["build_fingerprint"] = fingerprint
    payload["dex_list"][0]["api_sensitive_mask"] = torch.tensor(
        [0, 2],
        dtype=torch.uint8,
    )
    torch.save(payload, path)

    resumed, row = _resume_existing(
        {"split": "train", "sha256": sid},
        {
            "resume": True,
            "out_dirs": {"train": out_dir},
        },
        fingerprint,
    )

    assert resumed is False
    assert row["status"] == "failed"
    assert "api_sensitive_mask must contain only 0/1" in row["reason"]


def test_collate_never_silently_drops_failed_samples():
    with pytest.raises(FatalDatasetConfigError, match="never dropped from metrics"):
        robust_collate_fn([None])


def test_dataset_certificate_fast_path_uses_stats_without_loading_pts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    fingerprint = "f" * 64
    entries = []
    for sid in ("first", "second"):
        path = pt_dir / f"{sid}.pt"
        save_current_pt(path, _valid_dex(), manifest_dim=32)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["direct_build_meta"]["build_fingerprint"] = fingerprint
        torch.save(payload, path)
        stat = path.stat()
        entries.append(
            {
                "sid": sid,
                "relative_path": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "call_x_dims": [8],
            }
        )
    certificate_path = tmp_path / "migration_apply.json"
    certificate = {
        "certificate_version": PT_AUDIT_CERTIFICATE_VERSION,
        "pt_schema_version": PT_SCHEMA_VERSION,
        "pool_root": str(pt_dir.resolve()),
        "build_fingerprint": fingerprint,
        "entries": entries,
        "certificate_sha256": pt_audit_entries_sha256(entries),
    }
    certificate_path.write_text(
        json.dumps(
            {
                "mode": "apply",
                "after": {"pt_audit_certificate": certificate},
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "first", "label": 0})
        writer.writerow({"id": "second", "label": 1})

    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("certificate fast path must not torch.load at init")
        ),
    )
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        expected_pt_build_fingerprint=fingerprint,
        pt_audit_certificate=str(certificate_path),
        require_pt_audit_certificate=True,
    )

    assert dataset.feature_dim == 8
