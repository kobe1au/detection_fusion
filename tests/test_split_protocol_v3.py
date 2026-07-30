from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from fusion.dataset import build_package_isolation_groups
from scripts.build_tri_modal_pts_direct import (
    _build_fingerprint,
    _fingerprint_config,
    _load_direct_config,
)
from scripts.repartition_labels_by_year import build_assignment, read_inputs


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels"
EXPECTED_TEST_SHA256 = (
    "2481ec95d07d78f5f080464ab54d6cc310fe00e7a18e2a986cf20c7b4bd7954b"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(split: str) -> list[dict[str, str]]:
    with (LABELS / f"{split}.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _identity_digest(identities: list[str]) -> str:
    digest = hashlib.sha256()
    for sid in sorted(value.strip().lower() for value in identities):
        encoded = sid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _groups(rows: list[dict[str, str]]) -> set[str]:
    sids = [row["sha256"].strip().lower() for row in rows]
    packages = {
        sid: row.get("pkg_name", "")
        for sid, row in zip(sids, rows)
    }
    return set(build_package_isolation_groups(sids, packages))


def test_formal_7_1_2_split_matches_frozen_protocol() -> None:
    metadata = json.loads(
        (LABELS / "split_metadata_7_1_2.json").read_text(encoding="utf-8")
    )
    assert metadata["protocol"] == (
        "year_label_stratified_package_group_fixed_test_7_1_2_v1"
    )
    assert metadata["ratios"] == {"train": 0.7, "val": 0.1, "test": 0.2}
    assert metadata["frozen_split"] == "test"
    assert metadata["moved_samples"] == 1461

    expected_counts = {"train": 10223, "val": 1460, "test": 2921}
    split_rows = {split: _rows(split) for split in expected_counts}
    assert {
        split: len(rows) for split, rows in split_rows.items()
    } == expected_counts

    identities = {
        split: {row["sha256"].strip().lower() for row in rows}
        for split, rows in split_rows.items()
    }
    assert all(len(identities[split]) == expected_counts[split] for split in identities)
    assert identities["train"].isdisjoint(identities["val"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["val"].isdisjoint(identities["test"])
    assert len(set().union(*identities.values())) == 14604

    groups = {split: _groups(rows) for split, rows in split_rows.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

    for split, rows in split_rows.items():
        actual_joint = Counter(
            f"{int(row['year'])}:{int(row['label'])}" for row in rows
        )
        assert dict(sorted(actual_joint.items())) == metadata["summary"][split][
            "year_labels"
        ]
        assert _sha256(LABELS / f"{split}.csv") == metadata[
            "output_csv_sha256"
        ][split]

    assert metadata["input_csv_sha256"]["test"] == EXPECTED_TEST_SHA256
    assert metadata["output_csv_sha256"]["test"] == EXPECTED_TEST_SHA256
    assert metadata["input_identity_sha256"]["test"] == (
        "0b096750d6d18d7bb23c2035380fac90d4cbf52e7e2a27f67902a619cfdf6460"
    )
    assert (
        metadata["output_identity_sha256"]["test"]
        == metadata["input_identity_sha256"]["test"]
    )
    assert _sha256(LABELS / "test.csv") == EXPECTED_TEST_SHA256
    assert metadata["old_to_new"] == {
        "test->test": 2921,
        "train->train": 8762,
        "val->train": 1461,
        "val->val": 1460,
    }

    all_rows, _ = read_inputs(LABELS)
    assignment, _ = build_assignment(
        all_rows,
        (0.7, 0.1, 0.2),
        42,
        freeze_test=True,
    )
    assert all(
        assignment[row["_group_key"]] == row["_old_split"] for row in all_rows
    )


def test_validation_roles_v3_are_complete_disjoint_and_sufficient_for_i3() -> None:
    rows = _rows("val")
    role_path = LABELS / "validation_roles_protocol_v3.json"
    payload = json.loads(role_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["protocol"] == (
        "year_label_stratified_package_group_2to1_v3"
    )
    assert payload["validation_csv_sha256"] == _sha256(LABELS / "val.csv")
    assert payload["counts"] == {
        "decision_calibration": 487,
        "model_selection": 973,
    }

    ids = {row["sha256"].strip().lower() for row in rows}
    selection = set(payload["roles"]["model_selection"])
    decision = set(payload["roles"]["decision_calibration"])
    assert selection.isdisjoint(decision)
    assert selection | decision == ids

    row_by_sid = {
        row["sha256"].strip().lower(): row
        for row in rows
    }
    selection_rows = [row_by_sid[sid] for sid in selection]
    decision_rows = [row_by_sid[sid] for sid in decision]
    assert _groups(selection_rows).isdisjoint(_groups(decision_rows))
    assert sum(int(row["label"]) == 1 for row in decision_rows) == 240
    assert sum(int(row["label"]) == 1 for row in decision_rows) >= 100


def test_manifest_vocab_is_bound_to_the_new_train_split() -> None:
    train_rows = _rows("train")
    vocab = yaml.safe_load(
        (ROOT / "config" / "manifest_vocab.yaml").read_text(encoding="utf-8")
    )
    metadata = vocab["metadata"]
    assert metadata["source_split"] == "train"
    assert metadata["leakage_guard"] == "train_only"
    assert metadata["num_records"] == 10223
    assert metadata["train_csv_sha256"] == _sha256(LABELS / "train.csv")
    assert metadata["train_sample_ids_sha256"] == _identity_digest(
        [row["sha256"] for row in train_rows]
    )
    build_cfg = _load_direct_config(ROOT / "config" / "build_pts.yaml")
    assert build_cfg["migration_allowed_previous_build_fingerprints"] == [
        "7825ef23db92b98a7055b72275317f045e234d305f1d5803a063e4b0edbf25da"
    ]
    expected_fingerprint = _build_fingerprint(
        _fingerprint_config(build_cfg),
        vocab,
    )
    experiment_cfg = yaml.safe_load(
        (
            ROOT
            / "config"
            / "experiments"
            / "tri_modal_robust"
            / "base_tri_modal_robust.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        experiment_cfg["data"]["expected_pt_build_fingerprint"]
        == expected_fingerprint
    )
