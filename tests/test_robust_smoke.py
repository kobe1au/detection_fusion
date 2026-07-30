from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

import fusion.perturbations as perturbations_module
from fusion.constants import AvailabilityIndex
from fusion.dataset import (
    FatalDatasetConfigError,
    RobustTriModalDataset,
    build_package_isolation_groups,
    robust_collate_fn,
)
from fusion.losses import compute_robust_loss
from fusion.manifest_features import (
    DEFAULT_CATEGORIES,
    category_counts_from_strings,
    load_manifest_vocab,
    normalize_manifest_permissions,
    vectorize_manifest_record,
)
from fusion.model import ApiSequenceEncoder, TriModalRobustModel
from fusion.train import (
    _dataset_common_kwargs,
    _json_compatible,
    _metrics,
    enforce_failed_ratio,
    load_config_path,
    run as run_training,
    validate_eval_checkpoint_config,
    validate_split_partitions,
)
from fusion.semantic_categories import (
    CATEGORY_TO_INDEX,
    DEFAULT_API_TYPE_ID_TO_CATEGORY,
    SEMANTIC_CATEGORIES,
    api_semantic_counts_from_type_ids,
    validate_api_type_mapping,
)
from scripts.build_tri_modal_pts_direct import (
    PT_SCHEMA_VERSION,
    _build_one,
    _build_fingerprint,
    _load_direct_config,
    _resume_existing,
    _validate_unique_hashes,
)
from fusion.quality import OBSERVABLE_REQUIRED_FIELDS, OBSERVABLE_SCHEMA_VERSION
from fusion.perturbations import (
    EVAL_PERTURB_TYPES,
    apply_api_event_dropout,
    apply_api_missing,
    apply_graph_sparsify,
    apply_perturbation,
    apply_graph_missing,
    apply_manifest_missing,
    apply_manifest_permission_mask,
)
from tests.pt_factory import current_pt_payload, save_current_pt


def test_metrics_report_macro_f1_as_primary_f1():
    labels = [0, 0, 1, 1]
    preds = [1, 1, 1, 1]
    probs = [0.7, 0.8, 0.9, 0.6]

    metrics = _metrics(labels, probs, preds)

    assert metrics["acc"] == pytest.approx(0.5)
    assert metrics["auc_defined"] == 1
    assert metrics["ap_defined"] == 1
    assert metrics["f1_pos"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_f1"] == pytest.approx(1.0 / 3.0)
    assert metrics["f1"] == pytest.approx(metrics["macro_f1"])
    assert metrics["recall_pos"] == pytest.approx(1.0)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert "brier" in metrics
    assert "ece_10" in metrics
    assert "mean_confidence" in metrics


def test_metrics_mark_auc_and_ap_undefined_for_single_class():
    metrics = _metrics([1, 1], [0.8, 0.9], [1, 1])

    assert metrics["auc_defined"] == 0
    assert metrics["auc"] != metrics["auc"]
    assert metrics["ap_defined"] == 0
    assert metrics["ap"] != metrics["ap"]


def test_confidence_metrics_are_independent_of_operating_threshold():
    metrics = _metrics(
        labels=[0, 1],
        probs=[0.4, 0.4],
        # A tuned operating threshold can label the second sample malware even
        # though benign remains the maximum-probability class.
        preds=[0, 1],
    )

    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["mean_confidence"] == pytest.approx(0.6)
    assert metrics["ece_10"] == pytest.approx(0.1)
    assert metrics["confidence_accuracy_gap"] == pytest.approx(0.1)


def test_metric_serialization_converts_nonfinite_values_to_null():
    cleaned = _json_compatible(
        {"nan": float("nan"), "nested": [float("inf"), 1.0]}
    )

    assert cleaned == {"nan": None, "nested": [None, 1.0]}
    dumped = yaml.safe_dump(cleaned)
    assert ".nan" not in dumped.lower()
    assert ".inf" not in dumped.lower()


def test_internal_isolation_groups_use_package_or_sample_id():
    sids = ["a", "b", "c", "d"]
    packages = {"a": "pkg-one", "b": "pkg-one", "c": "pkg-two", "d": "pkg-three"}

    groups = build_package_isolation_groups(sids, packages)

    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert groups[3] != groups[0]


def test_config_loader_allows_shared_parent_defaults(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("model:\n  hidden: 64\n", encoding="utf-8")
    left.write_text("defaults: [parent.yaml]\nloss:\n  left: 1\n", encoding="utf-8")
    right.write_text("defaults: [parent.yaml]\nloss:\n  right: 2\n", encoding="utf-8")
    child.write_text("defaults: [left.yaml, right.yaml]\n", encoding="utf-8")

    cfg = load_config_path(child)

    assert cfg["model"]["hidden"] == 64
    assert cfg["loss"] == {"left": 1, "right": 2}


def test_unknown_data_and_builder_settings_fail_fast(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported data settings"):
        _dataset_common_kwargs(
            {
                "data": {"removed_option": True},
                "model": {},
            },
            is_train=False,
        )

    path = tmp_path / "build.yaml"
    path.write_text("execution:\n  removed_option: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported execution settings"):
        _load_direct_config(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"train": {"tuning_mode": True}},
            "Removed Stage-1 tuning settings",
        ),
        (
            {"train": {"checkpoint_metric": "robust_composite"}},
            "Removed Stage-1 tuning settings",
        ),
        (
            {"eval": {"robust_val": {"enabled": True}}},
            "eval.robust_val was removed",
        ),
    ],
)
def test_removed_stage1_tuning_settings_fail_at_config_load(
    tmp_path: Path,
    payload: dict,
    message: str,
):
    path = tmp_path / "removed_stage1_tuning.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config_path(path)


def test_posthoc_calibration_requires_discount_probability_fusion():
    with pytest.raises(ValueError, match="require discount_probability fusion"):
        run_training(
            {
                "train": {"device": "cpu"},
                "data": {},
                "model": {"fusion_mode": "tri_modal_fixed_gate"},
                "fusion": {"mode": "model_dispatch"},
                "calibration": {"enabled": True},
                "eval": {"run_test": False, "run_robust_test": False},
            }
        )


def test_eval_only_requires_checkpoint_path():
    with pytest.raises(ValueError, match="requires eval.checkpoint_path"):
        run_training(
            {
                "train": {"device": "cpu"},
                "data": {},
                "model": {"gate": {}},
                "eval": {
                    "eval_only": True,
                    "run_test": False,
                    "run_robust_test": False,
                },
            }
        )


def test_eval_only_checkpoint_config_rejects_semantic_mismatch():
    saved = {
        "model": {"fusion_mode": "tri_modal_fixed_gate", "graph_encoder": {"use_behavior_hint": False}},
        "data": {},
    }
    current = {
        "model": {"fusion_mode": "tri_modal_fixed_gate", "graph_encoder": {"use_behavior_hint": True}},
        "data": {},
    }
    with pytest.raises(ValueError, match="changes model/data semantics"):
        validate_eval_checkpoint_config(current, saved)


def test_partition_validation_rejects_cross_split_package_leakage(tmp_path: Path):
    for split, sid in (("train", "a"), ("val", "b")):
        path = tmp_path / f"{split}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label", "split", "pkg_name"])
            writer.writeheader()
            writer.writerow({"id": sid, "label": 0, "split": split, "pkg_name": "same.package"})
    cfg = {
        "data": {
            "root": str(tmp_path),
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "strict_partition_isolation": True,
        }
    }
    with pytest.raises(ValueError, match="package_overlap=1"):
        validate_split_partitions(cfg, include_test=False)


def test_api_type_id_mapping_matches_extractor_taxonomy():
    expected = {
        1: "telephony",
        2: "sms",
        3: "location",
        4: "contacts",
        5: "camera_media",
        6: "network",
        7: "dynamic_loading",
        8: "dynamic_loading",
        9: "dynamic_loading",
        10: "storage",
        11: "component_exposure",
        12: "crypto",
        13: "network",
        14: "system_settings",
        15: "contacts",
    }
    for type_id, category in expected.items():
        counts = api_semantic_counts_from_type_ids(torch.tensor([type_id], dtype=torch.long))
        assert counts[CATEGORY_TO_INDEX[category]].item() == 1.0


def test_validate_api_type_mapping_accepts_current_default():
    # Default mapping must pass against the live extractor taxonomy.
    validate_api_type_mapping()


def test_validate_api_type_mapping_rejects_out_of_range_key():
    bad = dict(DEFAULT_API_TYPE_ID_TO_CATEGORY)
    # 99 is far outside any realistic API_CATEGORY_NAMES length.
    bad[99] = "network"
    with pytest.raises(ValueError, match="outside extractor range"):
        validate_api_type_mapping(mapping=bad)


def test_validate_api_type_mapping_rejects_unknown_category_value():
    bad = dict(DEFAULT_API_TYPE_ID_TO_CATEGORY)
    bad[1] = "not_a_real_category"
    with pytest.raises(ValueError, match="not in 12-D taxonomy"):
        validate_api_type_mapping(mapping=bad)


def test_validate_api_type_mapping_accepts_injected_taxonomy():
    # Allow callers to inject taxonomies (e.g. in unit tests for hypothetical
    # extractor revisions) without touching module-level defaults.
    validate_api_type_mapping(
        mapping={1: "network"},
        api_category_names=("other", "network"),
        target_categories=SEMANTIC_CATEGORIES,
    )


def test_validate_api_type_mapping_rejects_non_int_key():
    # Mixed-type keys must not crash sorted() with a TypeError.
    bad = {1: "network", "not_an_int": "network"}
    with pytest.raises(ValueError, match="outside extractor range"):
        validate_api_type_mapping(mapping=bad)


def test_validate_api_type_mapping_rejects_bool_key():
    # bool is an int subclass; reject explicitly so True/False can't masquerade
    # as id=1/id=0.
    bad = {True: "network"}
    with pytest.raises(ValueError, match="outside extractor range"):
        validate_api_type_mapping(mapping=bad)


def test_validate_api_type_mapping_rejects_empty_taxonomy():
    # n_names < 2 means the extractor has no real categories beyond 'other'.
    with pytest.raises(ValueError, match="reserved 'other' slot"):
        validate_api_type_mapping(
            mapping={},
            api_category_names=("other",),
        )


def test_robust_model_forward_and_loss():
    items = []
    for i in range(2):
        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor(i % 2),
        )
        data.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        items.append(data)

    batch = Batch.from_data_list(items)
    batch.api_ids = torch.randint(1, 32, (12,), dtype=torch.long)
    batch.api_type_ids = torch.randint(0, 4, (12,), dtype=torch.long)
    batch.api_sensitive_mask = torch.zeros(12)
    batch.api_batch = torch.cat([torch.full((6,), i, dtype=torch.long) for i in range(2)])
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_semantic_category_counts = torch.rand(2, 12)
    batch.graph_semantic_category_counts = torch.rand(2, 12)
    batch.api_category_counts = batch.api_semantic_category_counts
    batch.graph_category_counts = batch.graph_semantic_category_counts
    batch.manifest_x = torch.rand(2, 32)
    batch.manifest_category_counts = torch.rand(2, 12)
    batch.manifest_stats = torch.rand(2, 11)
    batch.api_alive = torch.ones(2, 1)
    batch.graph_alive = torch.ones(2, 1)
    batch.manifest_alive = torch.ones(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="discount_probability",
        api_num_hash_buckets=64,
        api_type_vocab_size=16,
        api_emb_dim=32,
        api_hidden_dim=64,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=32,
        graph_hidden=32,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=64,
        manifest_in_dim=32,
        manifest_emb_dim=32,
        manifest_hidden_dim=64,
        discount_fusion_config={
            "combination": "dempster",
            "reliability_calibration": {
                "enabled": True,
                "method": "monotonic_correctness",
                "use_api_observed_support": True,
            },
        },
    )
    model.eval()
    logits, extra = model(batch)
    loss, parts = compute_robust_loss(
        logits,
        torch.tensor([0, 1]),
        extra,
        {"branch_aux_weight": 0.05},
    )
    assert logits.shape == (2, 2)
    assert extra["fusion_weights"].shape == (2, 3)
    assert torch.allclose(extra["fusion_weights"].sum(dim=-1), torch.ones(2))
    assert extra["fusion_availability"].shape == (
        2,
        AvailabilityIndex.BASE_DIM,
    )
    assert torch.equal(extra["fusion_availability"], torch.ones(2, 3))
    expected_api_support = torch.full(
        (2,),
        math.log1p(6.0) / math.log1p(16.0),
    )
    assert torch.allclose(
        extra["api_observed_support"],
        expected_api_support,
    )
    assert torch.allclose(
        extra["observed_support_api"],
        expected_api_support,
    )
    assert extra["reliability_features_api"].shape == (2, 4)
    assert extra["reliability_features_graph"].shape == (2, 3)
    assert torch.equal(
        extra["api_observed_support_feature_active"],
        torch.ones(2),
    )
    assert "api_integrity" not in extra
    assert "api_encoder_coverage" not in extra
    assert "api_graph_anchor_support" not in extra
    assert "api_semantic_logits" not in extra
    assert "manifest_to_code_conflict" not in extra
    assert "code_to_manifest_conflict" not in extra
    assert torch.isfinite(loss)
    assert parts["branch_aux_weight"] == 0.05
    assert "cross_source_consistency_weight" not in parts
    assert "gate_prior_weight" not in parts

    missing_api_batch = batch.clone()
    missing_api_batch.api_alive = torch.zeros(2, 1)
    _, missing_api_extra = model(missing_api_batch)
    assert torch.allclose(
        missing_api_extra["fusion_weights"][:, 0],
        torch.zeros(2),
        atol=1e-6,
    )
    assert torch.allclose(
        missing_api_extra["fusion_weights"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    missing_graph_batch = batch.clone()
    missing_graph_batch.graph_alive = torch.zeros(2, 1)
    _, missing_graph_extra = model(missing_graph_batch)
    assert torch.equal(missing_graph_extra["fusion_weights"][:, 1], torch.zeros(2))

    missing_manifest_batch = batch.clone()
    missing_manifest_batch.manifest_alive = torch.zeros(2, 1)
    _, missing_manifest_extra = model(missing_manifest_batch)
    assert torch.equal(
        missing_manifest_extra["fusion_weights"][:, 2], torch.zeros(2)
    )


@pytest.mark.parametrize(
    "removed_mode",
    ("tri_modal_ours", "tri_modal_confidence_gate", "tri_modal_reliability_gate"),
)
def test_removed_four_branch_gate_modes_fail_fast(removed_mode):
    with pytest.raises(ValueError, match="Unsupported tri-modal fusion_mode"):
        TriModalRobustModel(in_feat_dim=16, fusion_mode=removed_mode)


def test_api_encoder_rejects_invalid_batch_assignments():
    encoder = ApiSequenceEncoder(
        num_hash_buckets=16,
        type_vocab_size=4,
        emb_dim=8,
        hidden_dim=8,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,
    )
    data = Data()
    data.api_ids = torch.tensor([1, 2], dtype=torch.long)
    data.api_batch = torch.tensor([0, 2], dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        encoder(data, num_graphs=2, device=torch.device("cpu"), dtype=torch.float32)


def test_api_encoder_rejects_silent_second_truncation():
    encoder = ApiSequenceEncoder(
        num_hash_buckets=16,
        type_vocab_size=4,
        emb_dim=8,
        hidden_dim=8,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        max_seq_len=2,
    )
    data = Data()
    data.api_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    data.api_type_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    data.api_sensitive_mask = torch.zeros(3)
    data.api_batch = torch.zeros(3, dtype=torch.long)

    with pytest.raises(RuntimeError, match="budget contract was violated"):
        encoder(data, num_graphs=1, device=torch.device("cpu"), dtype=torch.float32)


def test_api_hash_bucket_last_valid_id_is_not_overflow():
    encoder = ApiSequenceEncoder(
        num_hash_buckets=16,
        type_vocab_size=4,
        emb_dim=8,
        hidden_dim=8,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,
    )
    assert encoder.max_valid_api_id == 17
    assert encoder.overflow_id == 18
    assert encoder.api_embedding.num_embeddings == 19

    ids = torch.tensor([-1, 0, 1, 2, 17, 18, 99], dtype=torch.long)
    normalized = encoder._normalize_api_ids(ids, torch.device("cpu"))
    assert normalized.tolist() == [0, 0, 1, 2, 17, 18, 18]


def test_robust_dataset_collate(tmp_path: Path, monkeypatch):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample1"
    save_current_pt(
        pt_dir / f"{sid}.pt",
        [
            {
                "call_x": torch.randn(3, 8),
                "call_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "call_sensitive_mask": torch.tensor([0, 1, 0], dtype=torch.uint8),
                "api_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "api_type_ids": torch.tensor([1, 2, 0], dtype=torch.long),
                "api_sensitive_mask": torch.tensor([1.0, 0.0, 0.0]),
                "api_method_index": torch.tensor([0, 1, 2], dtype=torch.long),
                "api_in_graph_mask": torch.tensor([1.0, 1.0, 1.0]),
                "method_api_edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
                "manifest_x": torch.ones(16),
                "manifest_category_counts": torch.ones(12),
                "manifest_stats": torch.ones(11),
                "q_manifest": torch.tensor([1.0]),
                "pert_manifest": torch.tensor([0.0]),
            }
        ],
        manifest_dim=32,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})

    # Some PyTorch releases accept pathlib.Path for ordinary torch.load calls
    # but require a string filename when mmap=True.  Enforce the stricter
    # contract here so the AutoDL runtime cannot regress silently.
    original_torch_load = torch.load
    mmap_paths = []

    def strict_mmap_load(path, *args, **kwargs):
        if kwargs.get("mmap", False):
            assert isinstance(path, str)
            mmap_paths.append(path)
        return original_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", strict_mmap_load)

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=True,
        manifest_dim=32,
        manifest_category_dim=12,
        manifest_stats_dim=11,
        max_graph_nodes_per_sample=8,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=robust_collate_fn)))
    graph = batch["graph_batch"]
    assert set(graph.keys()) == {
        "x",
        "edge_index",
        "sensitive_mask",
        "batch",
        "ptr",
        "graph_encoder_budget_max_nodes",
        "graph_encoder_budget_contract",
        "api_ids",
        "api_type_ids",
        "api_sensitive_mask",
        "api_batch",
        "manifest_x",
        "api_alive",
        "graph_alive",
        "manifest_alive",
    }
    assert graph.manifest_x.shape == (1, 32)
    assert not hasattr(graph, "q_manifest")
    assert graph.api_ids.numel() == 3
    assert graph.api_type_ids.numel() == 3
    assert graph.api_sensitive_mask.numel() == 3
    for removed_field in (
        "api_method_index",
        "api_in_graph_mask",
        "method_api_edge_index",
        "api_semantic_category_counts",
        "graph_semantic_category_counts",
        "api_category_counts",
        "graph_category_counts",
        "real_node_mask",
        "real_num_nodes",
        "manifest_permission_ids",
        "manifest_intent_ids",
        "manifest_category_counts",
        "manifest_stats",
    ):
        assert not hasattr(graph, removed_field)
    assert graph.graph_encoder_budget_max_nodes.view(-1).tolist() == [8]
    assert graph.graph_encoder_budget_contract == [1, 8, [3]]
    assert graph.to(torch.device("cpu")).graph_encoder_budget_contract == [1, 8, [3]]
    if os.name == "nt":
        assert mmap_paths == []
    else:
        assert len(mmap_paths) >= 2  # split preflight and __getitem__


def test_dataset_strict_split_integrity_rejects_csv_pt_mismatch(tmp_path: Path):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    torch.save({}, pt_dir / "pt_only.pt")
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "csv_only", "label": 0})

    with pytest.raises(ValueError, match="Split integrity mismatch"):
        RobustTriModalDataset(str(pt_dir), str(csv_path), is_train=False)


def _make_label_contract_fixture(
    tmp_path: Path,
    *,
    label: object,
    year: object | None = None,
) -> tuple[Path, Path]:
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample"
    save_current_pt(pt_dir / f"{sid}.pt", {}, manifest_dim=32)
    csv_path = tmp_path / "labels.csv"
    fieldnames = ["id", "label"] + (["year"] if year is not None else [])
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row: dict[str, object] = {"id": sid, "label": label}
        if year is not None:
            row["year"] = year
        writer.writerow(row)
    return pt_dir, csv_path


@pytest.mark.parametrize("label", [0.5, 1.5, 2, "nan", "inf", "true"])
def test_dataset_rejects_non_finite_or_non_binary_labels(tmp_path: Path, label: object):
    pt_dir, csv_path = _make_label_contract_fixture(tmp_path, label=label)

    with pytest.raises(ValueError, match="finite binary integers"):
        RobustTriModalDataset(
            str(pt_dir), str(csv_path), is_train=False, manifest_dim=32
        )


@pytest.mark.parametrize("mapped_value", [0.5, 2, float("nan"), float("inf"), True])
def test_dataset_rejects_invalid_label_map_values(tmp_path: Path, mapped_value: object):
    pt_dir, csv_path = _make_label_contract_fixture(tmp_path, label="malware")

    with pytest.raises(ValueError, match=r"data\.label_map"):
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=32,
            label_map={"malware": mapped_value},
        )


def test_dataset_accepts_explicit_string_label_map(tmp_path: Path):
    pt_dir, csv_path = _make_label_contract_fixture(tmp_path, label="malware")

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        label_map={"malware": 1},
    )

    assert dataset.sample_labels == [1]


@pytest.mark.parametrize("year", [2020.5, "nan", "inf", "true"])
def test_dataset_rejects_non_finite_or_fractional_years(tmp_path: Path, year: object):
    pt_dir, csv_path = _make_label_contract_fixture(tmp_path, label=1, year=year)

    with pytest.raises(ValueError, match="years that are not finite integers"):
        RobustTriModalDataset(
            str(pt_dir), str(csv_path), is_train=False, manifest_dim=32
        )


@pytest.mark.parametrize("budget", [0, -1, 1.5, float("nan"), True])
def test_dataset_requires_a_positive_integral_api_budget(tmp_path: Path, budget: object):
    with pytest.raises(ValueError, match="max_api_events_per_sample"):
        RobustTriModalDataset(
            str(tmp_path / "pts"),
            str(tmp_path / "labels.csv"),
            is_train=False,
            max_api_events_per_sample=budget,
        )


def test_dataset_allows_pt_superset_for_real_failure_slice(tmp_path: Path):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    for sid in ("selected", "not_selected"):
        save_current_pt(
            pt_dir / f"{sid}.pt",
            {
                "call_x": torch.ones(1, 8),
                "call_edge_index": torch.empty((2, 0), dtype=torch.long),
                "api_ids": torch.tensor([1]),
                "api_type_ids": torch.tensor([1]),
                "manifest_x": torch.ones(16),
            },
            manifest_dim=32,
        )
    csv_path = tmp_path / "slice.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "selected", "label": 1})

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        allow_pt_superset=True,
    )
    assert len(dataset) == 1
    assert dataset.sample_sids == ["selected"]


def test_all_ghost_graph_is_not_alive_and_transports_no_runtime_quality_fields(
    tmp_path: Path,
):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "ghost"
    save_current_pt(
        pt_dir / f"{sid}.pt",
        {
            "call_x": torch.empty((0, 8)),
            "call_edge_index": torch.empty((2, 0), dtype=torch.long),
            "api_ids": torch.tensor([1]),
            "api_type_ids": torch.tensor([1]),
            "api_method_index": torch.tensor([0]),
            "method_api_edge_index": torch.tensor([[0], [0]], dtype=torch.long),
            "manifest_x": torch.ones(16),
        },
        manifest_dim=32,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 0})

    data = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        eval_perturb_type="graph_sparsify",
        eval_perturb_strength=0.5,
    )[0]
    assert data.graph_alive.item() == 0.0
    assert not hasattr(data, "graph_integrity")
    assert not hasattr(data, "q_align")
    assert not hasattr(data, "real_num_nodes")
    assert not hasattr(data, "real_node_mask")
    assert not hasattr(data, "method_api_edge_index")


def test_multidex_api_limit_is_sample_level_and_keeps_encoder_vectors_aligned(
    tmp_path: Path,
):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample2"
    save_current_pt(
        pt_dir / f"{sid}.pt",
        [
            {
                "call_x": torch.randn(2, 8),
                "call_edge_index": torch.empty((2, 0), dtype=torch.long),
                "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
                "api_ids": torch.tensor([1, 2], dtype=torch.long),
                "api_type_ids": torch.tensor([1, 2], dtype=torch.long),
                "api_sensitive_mask": torch.ones(2),
                "api_method_index": torch.tensor([0, 1], dtype=torch.long),
                "api_in_graph_mask": torch.ones(2),
                "method_api_edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                "manifest_x": torch.ones(16),
                "manifest_category_counts": torch.ones(12),
                "manifest_stats": torch.ones(11),
                "q_manifest": torch.tensor([1.0]),
                "pert_manifest": torch.tensor([0.0]),
            },
            {
                "call_x": torch.randn(2, 8),
                "call_edge_index": torch.empty((2, 0), dtype=torch.long),
                "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
                "api_ids": torch.tensor([3, 4], dtype=torch.long),
                "api_type_ids": torch.tensor([3, 4], dtype=torch.long),
                "api_sensitive_mask": torch.ones(2),
                "api_method_index": torch.tensor([0, 1], dtype=torch.long),
                "api_in_graph_mask": torch.ones(2),
                "method_api_edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            },
        ],
        manifest_dim=32,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 0, "year": 2024})

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        max_api_events_per_sample=3,
    )
    graph = next(iter(DataLoader(dataset, batch_size=1, collate_fn=robust_collate_fn)))["graph_batch"]
    # All 4 events are sensitive and the budget (3) forces a drop. The three
    # API-encoder vectors retain the same contiguous prefix.
    assert graph.api_ids.tolist() == [1, 2, 3]
    assert graph.api_type_ids.tolist() == [1, 2, 3]
    assert graph.api_sensitive_mask.tolist() == [1.0, 1.0, 1.0]
    assert not hasattr(graph, "method_api_edge_index")
    assert not hasattr(graph, "graph_semantic_category_counts")
    assert not hasattr(graph, "api_runtime_encoder_coverage")
    assert not hasattr(graph, "api_encoder_coverage")


def test_manifest_perturbation_uses_payload_vocab_dims(tmp_path: Path):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample_manifest_dims"
    manifest_stats = torch.ones(11)
    manifest_stats[0] = math.log1p(2) / 6.0
    manifest_x = torch.zeros(32)
    manifest_x[:3] = 1.0
    manifest_x[3:15] = 1.0 / 12.0
    manifest_x[15:26] = manifest_stats
    save_current_pt(
        pt_dir / f"{sid}.pt",
        {
            "call_x": torch.randn(2, 8),
            "call_edge_index": torch.empty((2, 0), dtype=torch.long),
            "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
            "api_ids": torch.tensor([1, 2], dtype=torch.long),
            "api_type_ids": torch.tensor([1, 2], dtype=torch.long),
            "api_sensitive_mask": torch.ones(2),
            "api_method_index": torch.tensor([0, 1], dtype=torch.long),
            "api_in_graph_mask": torch.ones(2),
            "method_api_edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            "manifest_x": manifest_x,
            "manifest_permission_dim": 2,
            "manifest_intent_dim": 1,
            "manifest_permission_ids": torch.tensor([1, 2]),
            "manifest_intent_ids": torch.tensor([1]),
            "manifest_category_counts": torch.ones(12),
            "manifest_stats": manifest_stats,
            "q_manifest": torch.tensor([1.0]),
            "pert_manifest": torch.tensor([0.0]),
        },
        manifest_dim=32,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 0, "year": 2024})

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        manifest_permission_dim=128,
        manifest_intent_dim=64,
        eval_perturb_type="manifest_permission_mask",
        eval_perturb_strength=1.0,
    )
    data = dataset[0]
    manifest_x = data.manifest_x.view(-1)
    assert manifest_x[:2].sum().item() == 0.0
    assert manifest_x[2:].sum().item() > 0.0


def test_dataset_rejects_manifest_x_larger_than_configured_dim(tmp_path: Path):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample_manifest_too_wide"
    save_current_pt(
        pt_dir / f"{sid}.pt",
        {
            "call_x": torch.randn(2, 8),
            "call_edge_index": torch.empty((2, 0), dtype=torch.long),
            "call_sensitive_mask": torch.zeros(2, dtype=torch.uint8),
            "api_ids": torch.tensor([1], dtype=torch.long),
            "api_type_ids": torch.tensor([1], dtype=torch.long),
            "api_sensitive_mask": torch.ones(1),
            "api_method_index": torch.tensor([0], dtype=torch.long),
            "api_in_graph_mask": torch.ones(1),
            "method_api_edge_index": torch.tensor([[0], [0]], dtype=torch.long),
            "manifest_x": torch.ones(40),
            "manifest_category_counts": torch.ones(12),
            "manifest_stats": torch.ones(11),
        },
        manifest_dim=40,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 0, "year": 2024})

    with pytest.raises(FatalDatasetConfigError, match="manifest_x dimension"):
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=32,
        )


def test_dataset_rejects_non_current_pt_schema_version(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="old_schema")
    path = pt_dir / "old_schema.pt"
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["direct_build_meta"]["pt_schema_version"] = PT_SCHEMA_VERSION - 1
    torch.save(raw, path)
    with pytest.raises(FatalDatasetConfigError, match="does not match required current version"):
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=32,
        )


def test_removed_legacy_loss_weights_are_rejected():
    logits = torch.zeros(2, 2, requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    extra = {}
    for cfg in (
        {"gate_prior_weight": 0.0},
        {"cross_source_consistency_weight": 0.0},
        {"semantic_reconstruction_weight": 0.0},
    ):
        with pytest.raises(ValueError, match="Removed loss configuration keys"):
            compute_robust_loss(logits, labels, extra, cfg)


_DEFAULT_PERMISSION_TOKENS = normalize_manifest_permissions(
    [
        "android.permission.READ_SMS",
        "android.permission.INTERNET",
        "android.permission.CAMERA",
        "android.permission.ACCESS_FINE_LOCATION",
    ]
)
_NON_PERMISSION_CATEGORY_COUNTS = torch.ones(12)


def _perturbation_sample(
    permission_tokens: list[str] | None = None,
    permission_token_ids: list[int] | None = None,
    *,
    permission_dim: int = 4,
):
    permission_tokens = (
        list(_DEFAULT_PERMISSION_TOKENS)
        if permission_tokens is None
        else list(permission_tokens)
    )
    assert permission_tokens == normalize_manifest_permissions(permission_tokens)
    permission_token_ids = (
        list(range(1, len(permission_tokens) + 1))
        if permission_token_ids is None
        else list(permission_token_ids)
    )
    assert len(permission_token_ids) == len(permission_tokens)
    known_ids = sorted({value for value in permission_token_ids if value > 0})
    assert all(value <= permission_dim for value in known_ids)

    manifest_category_counts = (
        _NON_PERMISSION_CATEGORY_COUNTS
        + category_counts_from_strings(permission_tokens)
    )
    manifest_stats = torch.ones(11)
    manifest_stats[0] = math.log1p(len(permission_tokens)) / 6.0
    intent_dim = 2
    feature_dim = 0
    category_start = permission_dim + intent_dim + feature_dim
    stats_start = category_start + 12
    manifest_x = torch.zeros(max(64, stats_start + manifest_stats.numel()))
    if known_ids:
        manifest_x[torch.tensor(known_ids) - 1] = 1.0
    manifest_x[permission_dim : permission_dim + intent_dim] = 1.0
    manifest_x[category_start : category_start + 12] = (
        manifest_category_counts / manifest_category_counts.sum()
    )
    manifest_x[stats_start : stats_start + 11] = manifest_stats
    return {
        "api_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "api_type_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "api_sensitive_mask": torch.ones(3),
        "x": torch.randn(3, 8),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "sensitive_mask": torch.ones(3, dtype=torch.uint8),
        "manifest_x": manifest_x,
        "manifest_permission_tokens": permission_tokens,
        "manifest_permission_token_ids": torch.tensor(
            permission_token_ids, dtype=torch.long
        ),
        "manifest_permission_ids": torch.tensor(known_ids, dtype=torch.long),
        "manifest_category_counts": manifest_category_counts,
        "manifest_stats": manifest_stats,
        "manifest_permission_dim": permission_dim,
        "manifest_intent_dim": intent_dim,
        "manifest_feature_dim": feature_dim,
    }


def test_each_nonclean_perturbation_refreshes_hard_availability_once(monkeypatch):
    original_refresh = perturbations_module.refresh_hard_availability
    refresh_count = 0

    def counted_refresh(data):
        nonlocal refresh_count
        refresh_count += 1
        return original_refresh(data)

    monkeypatch.setattr(
        perturbations_module, "refresh_hard_availability", counted_refresh
    )
    perturb_types = sorted(
        value for value in EVAL_PERTURB_TYPES if value not in {None, "clean"}
    )
    for index, perturb_type in enumerate(perturb_types):
        random.seed(7000 + index)
        torch.manual_seed(7000 + index)
        before = refresh_count
        apply_perturbation(_perturbation_sample(), perturb_type, 0.5)
        assert refresh_count - before == 1, perturb_type


def test_api_missing_sets_hard_availability_zero():
    original = _perturbation_sample()
    graph_x = original["x"].clone()
    graph_edges = original["edge_index"].clone()
    data = apply_api_missing(original)
    assert data["api_alive"] == 0.0
    assert data["api_ids"].numel() == 0
    assert data["api_type_ids"].numel() == 0
    assert data["api_sensitive_mask"].numel() == 0
    assert torch.equal(data["x"], graph_x)
    assert torch.equal(data["edge_index"], graph_edges)


def test_api_event_dropout_keeps_only_api_encoder_vectors_aligned():
    torch.manual_seed(0)
    data = _perturbation_sample()
    graph_x = data["x"].clone()
    graph_edges = data["edge_index"].clone()
    data["api_ids"] = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    data["api_type_ids"] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    data["api_sensitive_mask"] = torch.tensor([0.0, 1.0, 0.0, 1.0])

    out = apply_api_event_dropout(data, 0.5)
    for key in ("api_ids", "api_type_ids", "api_sensitive_mask"):
        assert out[key].numel() == 2
    expected_by_id = {
        10: (1, 0.0),
        11: (2, 1.0),
        12: (3, 0.0),
        13: (4, 1.0),
    }
    for api_id, api_type, sensitive in zip(
        out["api_ids"].tolist(),
        out["api_type_ids"].tolist(),
        out["api_sensitive_mask"].tolist(),
    ):
        assert (api_type, sensitive) == expected_by_id[api_id]
    assert out["api_alive"] == 1.0
    assert torch.equal(out["x"], graph_x)
    assert torch.equal(out["edge_index"], graph_edges)


def test_api_event_dropout_rejects_misaligned_side_tensors():
    data = _perturbation_sample()
    data["api_type_ids"] = data["api_type_ids"][:2]
    with pytest.raises(ValueError, match="api_type_ids.*length 2.*expected 3"):
        apply_api_event_dropout(data, 0.5)


def test_graph_missing_sets_hard_availability_zero():
    data = apply_graph_missing(_perturbation_sample())
    assert data["graph_alive"] == 0.0


def test_graph_sparsify_strength_one_removes_all_edges():
    data = _perturbation_sample()
    out = apply_graph_sparsify(data, 1.0)
    assert out["edge_index"].numel() == 0
    assert out["graph_alive"] == 1.0


def test_manifest_perturbation_updates_feature_vector_and_semantic_counts():
    """Known permission bits, counts, and semantics change together."""
    data = _perturbation_sample()
    before = data["manifest_category_counts"].clone()
    before_x = data["manifest_x"].clone()
    data = apply_manifest_permission_mask(data, 0.5)
    assert not torch.equal(before, data["manifest_category_counts"])
    assert not torch.equal(before_x, data["manifest_x"])
    assert data["manifest_alive"] == 1.0
    expected_count_stat = (
        math.log1p(len(data["manifest_permission_tokens"])) / 6.0
    )
    stats_start = 4 + 2 + 0 + 12
    assert data["manifest_stats"][0].item() == pytest.approx(expected_count_stat)
    assert data["manifest_x"][stats_start].item() == pytest.approx(
        expected_count_stat
    )
    assert data["manifest_permission_count"] == len(
        data["manifest_permission_tokens"]
    )


def _assert_manifest_permission_state_consistent(data: dict) -> None:
    tokens = data["manifest_permission_tokens"]
    token_ids = data["manifest_permission_token_ids"].long()
    permission_dim = int(data["manifest_permission_dim"])
    expected_known_ids = token_ids[token_ids > 0].unique(sorted=True)
    active_ids = (
        torch.where(data["manifest_x"][:permission_dim].abs() > 1.0e-8)[0]
        + 1
    )
    assert tokens == normalize_manifest_permissions(tokens)
    assert len(tokens) == token_ids.numel()
    assert torch.equal(data["manifest_permission_ids"], expected_known_ids)
    assert torch.equal(active_ids, expected_known_ids)
    assert data["manifest_permission_count"] == len(tokens)

    expected_stat = math.log1p(len(tokens)) / 6.0
    stats_start = (
        permission_dim
        + int(data["manifest_intent_dim"])
        + int(data["manifest_feature_dim"])
        + 12
    )
    assert data["manifest_stats"][0].item() == pytest.approx(expected_stat)
    assert data["manifest_x"][stats_start].item() == pytest.approx(expected_stat)

    expected_counts = (
        _NON_PERMISSION_CATEGORY_COUNTS
        + category_counts_from_strings(tokens)
    )
    assert torch.equal(data["manifest_category_counts"], expected_counts)
    category_start = stats_start - 12
    expected_normalized = expected_counts / expected_counts.sum()
    assert torch.allclose(
        data["manifest_x"][category_start:stats_start],
        expected_normalized,
    )


def test_permission_mask_strengths_cover_all_oov_permissions_proportionally():
    tokens = [
        f"oov.permission.network_{index:02d}" for index in range(10)
    ]
    weak = _perturbation_sample(tokens, [0] * 10)
    medium = _perturbation_sample(tokens, [0] * 10)
    torch.manual_seed(781)
    weak = apply_manifest_permission_mask(weak, 0.1)
    torch.manual_seed(781)
    medium = apply_manifest_permission_mask(medium, 0.3)

    assert len(weak["manifest_permission_tokens"]) == 9
    assert len(medium["manifest_permission_tokens"]) == 7
    assert set(medium["manifest_permission_tokens"]).issubset(
        weak["manifest_permission_tokens"]
    )
    assert weak["manifest_permission_ids"].numel() == 0
    assert medium["manifest_permission_ids"].numel() == 0
    _assert_manifest_permission_state_consistent(weak)
    _assert_manifest_permission_state_consistent(medium)


def test_permission_mask_keeps_mixed_known_oov_alignment_consistent():
    tokens = normalize_manifest_permissions(
        [
            "android.permission.INTERNET",
            "custom.permission.OOV_NETWORK",
            "android.permission.READ_SMS",
            "custom.permission.OOV_SMS",
        ]
    )
    token_to_id = {
        "android.permission.internet": 2,
        "android.permission.read_sms": 8,
    }
    token_ids = [token_to_id.get(token, 0) for token in tokens]
    data = _perturbation_sample(
        tokens,
        token_ids,
        permission_dim=8,
    )
    torch.manual_seed(991)
    masked = apply_manifest_permission_mask(data, 0.5)

    assert len(masked["manifest_permission_tokens"]) == 2
    _assert_manifest_permission_state_consistent(masked)


def test_permission_mask_strength_one_removes_known_and_oov_permissions():
    tokens = normalize_manifest_permissions(
        [
            "android.permission.INTERNET",
            "custom.permission.OOV_NETWORK",
            "android.permission.READ_SMS",
        ]
    )
    token_ids = [
        1 if token == "android.permission.internet" else 0
        for token in tokens
    ]
    data = _perturbation_sample(tokens, token_ids)
    out = apply_manifest_permission_mask(data, 1.0)

    assert out["manifest_permission_tokens"] == []
    assert out["manifest_permission_token_ids"].numel() == 0
    assert out["manifest_permission_ids"].numel() == 0
    assert torch.equal(
        out["manifest_category_counts"],
        _NON_PERMISSION_CATEGORY_COUNTS,
    )
    _assert_manifest_permission_state_consistent(out)


def test_permission_mask_rejects_payload_without_token_alignment():
    data = _perturbation_sample()
    del data["manifest_permission_token_ids"]
    with pytest.raises(
        ValueError,
        match="missing Manifest permission-alignment tensors",
    ):
        apply_manifest_permission_mask(data, 0.3)


def test_zero_strength_degradation_is_noop():
    perturb_types = [
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
    ]
    for perturb_type in perturb_types:
        data = _perturbation_sample()
        before = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
        out = apply_perturbation(data, perturb_type, 0.0)
        for key, expected in before.items():
            actual = out[key]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected), perturb_type
            else:
                assert actual == expected, perturb_type


def test_only_formal_controlled_perturbations_are_public():
    assert EVAL_PERTURB_TYPES == {
        None,
        "clean",
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
        "api_missing",
        "graph_missing",
        "manifest_missing",
    }


def test_eval_perturbation_is_deterministic_per_sample(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="deterministic_eval")
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        manifest_stats_dim=11,
        eval_perturb_type="api_event_dropout",
        eval_perturb_strength=0.5,
    )
    first = dataset[0]
    second = dataset[0]
    assert first.api_aug_type == second.api_aug_type
    assert torch.equal(first.api_ids, second.api_ids)
    stronger = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        manifest_stats_dim=11,
        eval_perturb_type="api_event_dropout",
        eval_perturb_strength=0.9,
    )[0]
    assert first.api_aug_type == stronger.api_aug_type


def test_manifest_permission_alignment_state_never_enters_model_batch(
    tmp_path: Path,
):
    sid = "manifest_mask_hidden_state"
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    manifest_payload = vectorize_manifest_record(
        {
            "sha256": sid,
            "permissions": [
                "android.permission.INTERNET",
                "custom.permission.OOV_NETWORK",
            ],
            "component_count": 1,
        },
        {
            "categories": list(DEFAULT_CATEGORIES),
            "permission_vocab": ["android.permission.internet"],
            "intent_vocab": [],
            "feature_vocab": [],
        },
        manifest_dim=32,
    )
    save_current_pt(
        pt_dir / f"{sid}.pt",
        {
            "call_x": torch.ones((1, 8)),
            "call_edge_index": torch.empty((2, 0), dtype=torch.long),
            "call_sensitive_mask": torch.zeros(1, dtype=torch.uint8),
            "api_ids": torch.tensor([1], dtype=torch.long),
            "api_type_ids": torch.tensor([1], dtype=torch.uint8),
            "api_sensitive_mask": torch.zeros(1, dtype=torch.uint8),
        },
        top_level=manifest_payload,
        manifest_dim=32,
    )
    csv_path = tmp_path / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})

    sample = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=32,
        manifest_permission_dim=1,
        manifest_intent_dim=0,
        manifest_feature_dim=0,
        eval_perturb_type="manifest_permission_mask",
        eval_perturb_strength=0.5,
    )[0]

    assert sample.manifest_aug_type == "manifest_permission_mask"
    assert not hasattr(sample, "manifest_permission_tokens")
    assert not hasattr(sample, "manifest_permission_token_ids")


def test_manifest_missing_zeroes_manifest_counts_and_availability():
    data = apply_manifest_missing(_perturbation_sample())
    assert data["manifest_category_counts"].sum().item() == 0.0
    assert data["manifest_alive"] == 0.0


def test_removed_cross_source_and_semantic_losses_are_not_supported():
    logits = torch.zeros(1, 2, requires_grad=True)
    labels = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="Removed loss configuration keys"):
        compute_robust_loss(
            logits,
            labels,
            {},
            {"semantic_reconstruction_weight": 0.1, "cross_source_consistency_weight": 0.1},
        )


def test_loss_rejects_removed_cross_source_weight_regardless_of_value():
    with pytest.raises(ValueError, match="Removed loss configuration keys"):
        compute_robust_loss(
            torch.zeros(1, 2),
            torch.tensor([0]),
            {},
            {"cross_source_consistency_weight": -0.1},
        )


def test_empty_manifest_vocab_rejected_by_default(tmp_path: Path):
    path = tmp_path / "manifest_vocab.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "categories": list(DEFAULT_CATEGORIES),
                "permission_vocab": [],
                "intent_vocab": [],
                "feature_vocab": [],
                "metadata": {"source_split": "train", "leakage_guard": "train_only"},
            },
            f,
            sort_keys=False,
        )

    with pytest.raises(ValueError, match="Manifest vocab is empty"):
        load_manifest_vocab(path, require_train_metadata=True)

    vocab = load_manifest_vocab(path, require_train_metadata=True, allow_empty=True)
    assert vocab["permission_vocab"] == []


def test_manifest_vectorization_rejects_layout_truncation():
    vocab = {
        "categories": list(DEFAULT_CATEGORIES),
        "permission_vocab": ["android.permission.INTERNET"] * 4,
        "intent_vocab": ["android.intent.action.MAIN"] * 3,
        "feature_vocab": ["android.hardware.camera"] * 2,
    }
    record = {
        "permissions": ["android.permission.INTERNET"],
        "intent_actions": ["android.intent.action.MAIN"],
        "uses_features": ["android.hardware.camera"],
        "component_count": 1,
    }
    required = 4 + 3 + 2 + len(DEFAULT_CATEGORIES) + 11
    with pytest.raises(ValueError, match="manifest_dim is too small"):
        vectorize_manifest_record(record, vocab, manifest_dim=required - 1)


def test_manifest_vectorization_stores_permission_alignment_and_audit_coverage():
    vocab = {
        "categories": list(DEFAULT_CATEGORIES),
        "permission_vocab": ["android.permission.internet"],
        "intent_vocab": [],
        "feature_vocab": [],
    }
    record = {
        "permissions": [
            "UNKNOWN.PERMISSION",
            "android.permission.internet",
            "android.permission.INTERNET",
        ],
        "component_count": 1,
    }
    required = 1 + len(DEFAULT_CATEGORIES) + 11
    payload = vectorize_manifest_record(record, vocab, manifest_dim=required)
    assert payload["manifest_component_category_counts"].shape == (12,)
    assert "q_manifest" not in payload
    assert "pert_manifest" not in payload
    assert payload["observable_metadata"]["manifest_vocab_coverage"] == pytest.approx(
        0.5
    )
    assert payload["manifest_meta"]["permissions"] == [
        "android.permission.internet",
        "unknown.permission",
    ]
    assert payload["manifest_permission_token_ids"].tolist() == [1, 0]
    assert payload["manifest_permission_ids"].tolist() == [1]
    assert payload["observable_metadata"]["manifest_permission_count"] == 2
    assert payload["manifest_stats"][0].item() == pytest.approx(
        math.log1p(2) / 6.0
    )


def test_direct_build_fingerprint_changes_with_extraction_schema():
    cfg = {
        "vocab_size": 256,
        "sensitive_hops": 1,
        "max_methods_per_dex": 4096,
        "fallback_max_methods": 512,
        "fallback_policy": "api_rich",
        "use_graph_behavior_hints": False,
        "num_api_buckets": 8192,
        "max_api_events_per_dex": 1024,
        "max_api_events_per_method": 32,
        "api_event_scope": "all_methods",
        "framework_only": True,
        "include_descriptor": False,
        "manifest_dim": 256,
    }
    vocab = {
        "permission_vocab": ["p"],
        "intent_vocab": ["i"],
        "feature_vocab": ["f"],
        "categories": list(DEFAULT_CATEGORIES),
    }
    first = _build_fingerprint(cfg, vocab)
    cfg["use_graph_behavior_hints"] = True
    assert _build_fingerprint(cfg, vocab) != first


def test_direct_build_rejects_duplicate_hashes_across_splits():
    jobs = [
        {"split": "train", "apk_path": "a.apk", "sha256": "same"},
        {"split": "test", "apk_path": "b.apk", "sha256": "same"},
    ]
    with pytest.raises(RuntimeError, match="Duplicate APK hashes"):
        _validate_unique_hashes(jobs)


def test_direct_resume_requires_matching_current_schema(tmp_path: Path):
    out_dir = tmp_path / "train"
    out_dir.mkdir()
    job = {"split": "train", "apk_path": "a.apk", "sha256": "abc"}
    cfg = {
        "resume": True,
        "out_dirs": {"train": out_dir},
    }
    path = out_dir / "abc.pt"
    observable = {key: 0 for key in OBSERVABLE_REQUIRED_FIELDS}
    observable.update(
        {
            "api_parse_error": "",
            "graph_parse_error": "",
            "manifest_parse_error": "",
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        }
    )
    payload = current_pt_payload(
        {
            "call_x": torch.ones(1, 8),
            "call_edge_index": torch.empty((2, 0), dtype=torch.long),
            "api_ids": torch.tensor([1]),
            "api_type_ids": torch.tensor([1]),
        },
        sid="abc",
    )
    payload["observable_metadata"] = observable
    payload["direct_build_meta"]["build_fingerprint"] = "match"
    torch.save(payload, path)
    status, row = _resume_existing(job, cfg, "match")
    assert status is True
    assert row["status"] == "ok"
    assert row["reason"] == ""

    status, row = _resume_existing(job, cfg, "different")
    assert status is False
    assert row["status"] == "failed"

    del payload["manifest_permission_token_ids"]
    torch.save(payload, path)
    status, row = _resume_existing(job, cfg, "match")
    assert status is False
    assert row["status"] == "failed"


def test_direct_build_clears_resume_mismatch_reason_after_success(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "train"
    out_dir.mkdir()
    path = out_dir / "abc.pt"
    job = {"split": "train", "apk_path": str(tmp_path / "a.apk"), "sha256": "abc"}
    cfg = {
        "resume": True,
        "out_dirs": {"train": out_dir},
        "manifest_dim": 32,
    }

    stale = current_pt_payload(
        {
            "call_x": torch.ones(1, 8),
            "call_edge_index": torch.empty((2, 0), dtype=torch.long),
            "api_ids": torch.tensor([1]),
            "api_type_ids": torch.tensor([1]),
        },
        sid="abc",
    )
    stale["direct_build_meta"]["build_fingerprint"] = "stale"
    torch.save(stale, path)

    def fake_process_apk(*_args, **_kwargs):
        current = current_pt_payload(
            stale["dex_list"], manifest_dim=32, sid="abc"
        )
        current["direct_build_meta"]["build_fingerprint"] = "temporary-code-stage"
        torch.save(current, path)
        return True, ""

    monkeypatch.setattr(
        "scripts.build_tri_modal_pts_direct.process_apk",
        fake_process_apk,
    )
    result = _build_one(
        job,
        {"sha256": "abc", "component_count": 1},
        cfg,
        {
            "categories": list(DEFAULT_CATEGORIES),
            "permission_vocab": [],
            "intent_vocab": [],
            "feature_vocab": [],
        },
        "current",
    )

    assert result["status"] == "ok"
    assert result["reason"] == ""


def test_failed_ratio_guard_rejects_silent_bad_sample_rate():
    with pytest.raises(RuntimeError, match="failed sample ratio"):
        enforce_failed_ratio({"num_eval": 9, "num_failed": 1}, {"data": {"max_failed_ratio": 0.0}}, "train")


@pytest.mark.parametrize(
    "threshold", [float("nan"), float("inf"), -0.1, 1.0, 1.1, 0.1, True]
)
def test_failed_ratio_guard_rejects_invalid_or_nonzero_tolerance(threshold: object):
    with pytest.raises(ValueError, match="max_failed_ratio"):
        enforce_failed_ratio(
            {"num_eval": 1, "num_failed": 0},
            {"data": {"max_failed_ratio": threshold}},
            "test",
        )


def test_failed_ratio_guard_rejects_empty_and_all_failed_evaluation():
    cfg = {"data": {"max_failed_ratio": 0.0}}
    with pytest.raises(RuntimeError, match="no requested samples"):
        enforce_failed_ratio({"num_eval": 0, "num_failed": 0}, cfg, "empty")
    with pytest.raises(RuntimeError, match="no samples were evaluated successfully"):
        enforce_failed_ratio({"num_eval": 0, "num_failed": 3}, cfg, "all_failed")


def _make_graph_source_pt(tmp_path: Path, sid: str = "graph_src_sample"):
    """Build a tiny current-schema PT used by dataset contract tests."""
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir(parents=True, exist_ok=True)
    save_current_pt(
        pt_dir / f"{sid}.pt",
        [
            {
                "call_x": torch.randn(4, 8),
                "call_edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                "call_sensitive_mask": torch.zeros(4, dtype=torch.uint8),
                "api_ids": torch.tensor([10, 20, 30, 40], dtype=torch.long),
                "api_type_ids": torch.tensor([1, 2, 3, 6], dtype=torch.long),
                "api_sensitive_mask": torch.ones(4),
                "api_method_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "api_in_graph_mask": torch.ones(4),
                "method_api_edge_index": torch.tensor([[0, 3], [0, 3]], dtype=torch.long),
                "manifest_x": torch.ones(16),
                "manifest_category_counts": torch.ones(12),
                "manifest_stats": torch.ones(11),
                "q_manifest": torch.tensor([1.0]),
                "pert_manifest": torch.tensor([0.0]),
            }
        ],
        manifest_dim=32,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})
    return pt_dir, csv_path


# ── gradient-flow & empty-input integration tests ──────────────────────


def test_dense_embedding_gate_receives_gradient():
    """Backward pass must deliver non-zero gradients to the three-way gate."""
    items = []
    for i in range(2):
        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor(i % 2),
        )
        data.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        items.append(data)

    batch = Batch.from_data_list(items)
    batch.api_ids = torch.randint(1, 32, (12,), dtype=torch.long)
    batch.api_type_ids = torch.randint(0, 4, (12,), dtype=torch.long)
    batch.api_sensitive_mask = torch.zeros(12)
    batch.api_batch = torch.cat([torch.full((6,), i, dtype=torch.long) for i in range(2)])
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_semantic_category_counts = torch.rand(2, 12)
    batch.graph_semantic_category_counts = torch.rand(2, 12)
    batch.api_category_counts = batch.api_semantic_category_counts
    batch.graph_category_counts = batch.graph_semantic_category_counts
    batch.manifest_x = torch.rand(2, 32)
    batch.manifest_category_counts = torch.rand(2, 12)
    batch.manifest_stats = torch.rand(2, 11)
    batch.q_api = torch.ones(2, 1)
    batch.q_graph = torch.ones(2, 1)
    batch.q_manifest = torch.ones(2, 1)
    batch.q_align = torch.ones(2, 1) * 0.8
    batch.api_alive = torch.ones(2, 1)
    batch.graph_alive = torch.ones(2, 1)
    batch.manifest_alive = torch.ones(2, 1)
    batch.pert_api = torch.zeros(2, 1)
    batch.pert_graph = torch.zeros(2, 1)
    batch.pert_manifest = torch.zeros(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_dense_embedding_gate",
        api_num_hash_buckets=64,
        api_type_vocab_size=16,
        api_emb_dim=32,
        api_hidden_dim=64,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=32,
        graph_hidden=32,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=64,
        manifest_in_dim=32,
        manifest_emb_dim=32,
        manifest_hidden_dim=64,
    )

    gate_params = list(model.dense_embedding_gate.parameters())
    assert gate_params

    logits, extra = model(batch)
    loss, _ = compute_robust_loss(
        logits,
        torch.tensor([0, 1]),
        extra,
        {"branch_aux_weight": 0.05},
    )
    loss.backward()

    nonzero_grads = 0
    for p in gate_params:
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            nonzero_grads += 1
    assert nonzero_grads > 0, "dense embedding gate received zero gradients"


def test_empty_api_and_zero_node_graph_forward():
    """Model forward must handle empty API sequences and zero-node graphs
    without producing NaN or Inf."""
    items = []
    for i in range(2):
        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor(i % 2),
        )
        data.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        items.append(data)

    batch = Batch.from_data_list(items)
    # Empty API sequence.
    batch.api_ids = torch.empty((0,), dtype=torch.long)
    batch.api_type_ids = torch.empty((0,), dtype=torch.long)
    batch.api_sensitive_mask = torch.empty((0,), dtype=torch.float32)
    batch.api_batch = torch.empty((0,), dtype=torch.long)
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_semantic_category_counts = torch.rand(2, 12)
    batch.graph_semantic_category_counts = torch.rand(2, 12)
    batch.api_category_counts = batch.api_semantic_category_counts
    batch.graph_category_counts = batch.graph_semantic_category_counts
    batch.manifest_x = torch.rand(2, 32)
    batch.manifest_category_counts = torch.rand(2, 12)
    batch.manifest_stats = torch.rand(2, 11)
    batch.q_api = torch.ones(2, 1)
    batch.q_graph = torch.ones(2, 1)
    batch.q_manifest = torch.ones(2, 1)
    batch.q_align = torch.ones(2, 1) * 0.8
    batch.api_alive = torch.zeros(2, 1)
    batch.graph_alive = torch.ones(2, 1)
    batch.manifest_alive = torch.ones(2, 1)
    batch.pert_api = torch.zeros(2, 1)
    batch.pert_graph = torch.zeros(2, 1)
    batch.pert_manifest = torch.zeros(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_fixed_gate",
        api_num_hash_buckets=64,
        api_type_vocab_size=16,
        api_emb_dim=32,
        api_hidden_dim=64,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=32,
        graph_hidden=32,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=64,
        manifest_in_dim=32,
        manifest_emb_dim=32,
        manifest_hidden_dim=64,
    )

    logits, extra = model(batch)
    assert logits.shape == (2, 2), "output shape must be (batch, num_classes)"
    assert torch.isfinite(logits).all(), "logits must be finite with empty API"
    assert extra["gate_weights"].shape == (2, 3)
    assert torch.isfinite(extra["gate_weights"]).all()

    # Also test with zero-node graphs.
    batch_zero = batch.clone()
    batch_zero.x = torch.empty((0, 16))
    batch_zero.edge_index = torch.empty((2, 0), dtype=torch.long)
    batch_zero.batch = torch.empty((0,), dtype=torch.long)
    batch_zero.sensitive_mask = torch.empty((0,), dtype=torch.uint8)
    batch_zero.graph_alive = torch.zeros(2, 1)

    logits_zero, extra_zero = model(batch_zero)
    assert logits_zero.shape == (2, 2)
    assert torch.isfinite(logits_zero).all(), "logits must be finite with zero-node graph"
    assert extra_zero["gate_weights"].shape == (2, 3)
    assert torch.isfinite(extra_zero["gate_weights"]).all()
