from __future__ import annotations

import copy
import hashlib
import json

import torch
from torch.utils.data import Dataset

from fusion.train import (
    _encoder_stage_semantic_signature,
    _load_fixed_validation_roles,
    _state_dict_sha256,
    build_loader,
)


class _IntegerDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index


def _loader_order(cfg):
    loader = build_loader(
        cfg,
        _IntegerDataset(),
        is_train=True,
        seed_namespace="stage1_train",
        collate_fn_override=lambda values: list(values),
    )
    return [value for batch in loader for value in batch]


def test_stage1_loader_shuffle_is_independent_of_global_rng_consumption():
    cfg = {
        "train": {
            "seed": 42,
            "loader_seed": 31415,
            "batch_size": 3,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    }
    torch.manual_seed(1)
    torch.rand(1000)
    first = _loader_order(cfg)
    torch.manual_seed(999)
    torch.rand(7)
    second = _loader_order(cfg)

    assert first == second
    assert sorted(first) == list(range(12))


def test_encoder_identity_ignores_posthoc_configuration_but_not_stage1():
    cfg = {
        "encoder_stage": {"protocol_id": "stage1-v1", "mode": "fit"},
        "train": {
            "seed": 42,
            "stage1_seed": 101,
            "loader_seed": 202,
            "epochs": 3,
            "batch_size": 4,
        },
        "model": {"hidden": 8},
        "fusion": {
            "mode": "discount_probability",
            "combination": "routed",
            "routing": {
                "enabled": True,
                "risk_mode": "learned",
            },
            "reliability_calibration": {"enabled": True},
        },
        "loss": {"branch_aux_weight": 0.25},
        "robust": {"train_aug": True},
        "data": {
            "max_api_events_per_sample": 32,
            "expected_pt_build_fingerprint": "f" * 64,
        },
    }
    roles = {
        "role_assignment_semantic_sha256": "a" * 64,
        "validation_csv_sha256": "b" * 64,
        "num_selection": 10,
    }
    manifest = {
        "manifest_vocab_sha256": "c" * 64,
        "train_csv_sha256": "d" * 64,
        "train_sample_ids_sha256": "e" * 64,
        "num_train_samples": 100,
    }
    original = _encoder_stage_semantic_signature(cfg, roles, manifest)
    posthoc_changed = copy.deepcopy(cfg)
    posthoc_changed["encoder_stage"] = {
        "protocol_id": "stage1-v1",
        "mode": "reuse",
        "checkpoint_path": "elsewhere.pt",
    }
    posthoc_changed["fusion"]["routing"]["risk_mode"] = "reliability_prior"
    posthoc_changed["fusion"]["reliability_calibration"]["enabled"] = False
    assert (
        _encoder_stage_semantic_signature(posthoc_changed, roles, manifest)
        == original
    )

    stage1_changed = copy.deepcopy(cfg)
    stage1_changed["model"]["hidden"] = 16
    assert (
        _encoder_stage_semantic_signature(stage1_changed, roles, manifest)
        != original
    )
    fusion_changed = copy.deepcopy(cfg)
    fusion_changed["fusion"]["evidence_activation"] = "exp"
    assert (
        _encoder_stage_semantic_signature(fusion_changed, roles, manifest)
        != original
    )
    roles_changed = dict(roles)
    roles_changed["role_assignment_semantic_sha256"] = "9" * 64
    assert (
        _encoder_stage_semantic_signature(cfg, roles_changed, manifest)
        != original
    )
    pt_build_changed = copy.deepcopy(cfg)
    pt_build_changed["data"]["expected_pt_build_fingerprint"] = "8" * 64
    assert (
        _encoder_stage_semantic_signature(pt_build_changed, roles, manifest)
        != original
    )
    loader_changed = copy.deepcopy(cfg)
    loader_changed["train"]["num_workers"] = 7
    assert (
        _encoder_stage_semantic_signature(loader_changed, roles, manifest)
        != original
    )


def test_encoder_state_hash_accepts_scalar_buffers_and_detects_changes():
    state = {
        "weight": torch.tensor([1.0, 2.0]),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    first = _state_dict_sha256(state)
    second = _state_dict_sha256({key: value.clone() for key, value in state.items()})
    changed = {key: value.clone() for key, value in state.items()}
    changed["counter"].add_(1)

    assert first == second
    assert first != _state_dict_sha256(changed)


class _RoleDataset:
    sample_sids = ["a", "b", "c", "d"]
    sample_groups = ["package:a", "package:b", "package:c", "package:d"]
    sample_labels = [0, 1, 0, 1]
    sample_years = [2020, 2020, 2021, 2021]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index


def test_fixed_validation_roles_are_identity_complete_and_disjoint(tmp_path):
    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "id,label,year,split,pkg_name\n"
        "a,0,2020,val,a\n"
        "b,1,2020,val,b\n"
        "c,0,2021,val,c\n"
        "d,1,2021,val,d\n",
        encoding="utf-8",
    )
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    role_path = tmp_path / "roles.json"
    role_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation_csv_sha256": csv_sha,
                "split_seed": 42,
                "roles": {
                    "checkpoint_selection": ["a", "b"],
                    "posthoc_calibration": ["c"],
                    "decision_calibration": ["d"],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "data": {"root": "", "val_csv": str(csv_path)},
        "calibration": {
            "role_assignment_path": str(role_path),
            "require_role_assignment": True,
        },
    }

    selection, posthoc, decision, summary = _load_fixed_validation_roles(
        cfg, _RoleDataset()
    )

    assert list(selection.indices) == [0, 1]
    assert list(posthoc.indices) == [2]
    assert list(decision.indices) == [3]
    assert summary["num_selection"] == 2
    assert summary["num_posthoc_calibration"] == 1
    assert summary["num_conformal_calibration"] == 1
