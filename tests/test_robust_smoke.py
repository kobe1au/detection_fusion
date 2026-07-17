from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from fusion.constants import EvidenceIndex
from fusion.dataset import (
    FatalDatasetConfigError,
    RobustTriModalDataset,
    build_package_isolation_groups,
    robust_collate_fn,
)
from fusion.gates import heuristic_reliability_gate
from fusion.losses import compute_robust_loss
from fusion.quality import compute_align_quality
from fusion.manifest_features import DEFAULT_CATEGORIES, load_manifest_vocab, vectorize_manifest_record
from fusion.model import ApiSequenceEncoder, TriModalRobustModel
from fusion.train import (
    _dataset_common_kwargs,
    _json_compatible,
    _metrics,
    _normalize_robust_val_scenarios,
    checkpoint_score,
    checkpoint_requires_robust_validation,
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
    apply_api_event_dropout,
    apply_api_missing,
    apply_graph_feature_obfuscation,
    apply_graph_sparsify,
    apply_perturbation,
    apply_graph_missing,
    apply_manifest_component_mask,
    apply_manifest_feature_noise,
    apply_manifest_missing,
    apply_manifest_permission_injection,
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


def test_probability_calibration_is_independent_of_operating_threshold():
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


def test_checkpoint_score_clean_and_robust_composite():
    clean = {"macro_f1": 0.9}
    robust = {
        "api_graph": {"macro_f1": 0.8},
        "manifest": {"macro_f1": 0.7},
    }
    loaders = [
        {"name": "api_graph", "weight": 0.4},
        {"name": "manifest", "weight": 0.2},
    ]

    clean_score, clean_name = checkpoint_score(
        {"train": {"checkpoint_metric": "clean_macro_f1"}},
        clean,
        robust,
        loaders,
    )
    assert clean_name == "clean_macro_f1"
    assert clean_score == pytest.approx(0.9)

    robust_score, robust_name = checkpoint_score(
        {
            "train": {"checkpoint_metric": "robust_composite"},
            "eval": {"robust_val": {"clean_weight": 0.4}},
        },
        clean,
        robust,
        loaders,
    )
    assert robust_name == "robust_composite"
    assert robust_score == pytest.approx((0.4 * 0.9 + 0.4 * 0.8 + 0.2 * 0.7) / 1.0)


def test_epoch_robust_validation_is_only_required_for_robust_checkpoint_metric():
    assert checkpoint_requires_robust_validation(
        {"train": {"checkpoint_metric": "robust_composite"}}
    )
    assert not checkpoint_requires_robust_validation(
        {
            "train": {"checkpoint_metric": "clean_macro_f1"},
            "eval": {"robust_val": {"enabled": True}},
        }
    )


def test_checkpoint_score_robust_composite_requires_robust_loaders():
    with pytest.raises(ValueError, match="robust_val.enabled=true"):
        checkpoint_score(
            {"train": {"checkpoint_metric": "robust_composite"}},
            {"macro_f1": 0.9},
            {},
            [],
        )


def test_robust_val_scenarios_reject_duplicates_and_invalid_strength():
    with pytest.raises(ValueError, match="Duplicate"):
        _normalize_robust_val_scenarios(
            [
                {"name": "same", "perturb_type": "api_degraded", "strength": 0.5, "weight": 0.5},
                {"name": "same", "perturb_type": "graph_degraded", "strength": 0.5, "weight": 0.5},
            ]
        )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        _normalize_robust_val_scenarios(
            [{"name": "bad", "perturb_type": "api_degraded", "strength": 1.5, "weight": 1.0}]
        )
    with pytest.raises(ValueError, match="unsupported perturb_type"):
        _normalize_robust_val_scenarios(
            [{"name": "bad", "perturb_type": "unknown_degradation", "strength": 0.5, "weight": 1.0}]
        )


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


def test_tuning_mode_forbids_test_evaluation():
    with pytest.raises(ValueError, match="forbids test evaluation"):
        run_training(
            {
                "train": {
                    "tuning_mode": True,
                    "checkpoint_metric": "robust_composite",
                    "device": "cpu",
                },
                "data": {},
                "eval": {
                    "run_test": True,
                    "run_robust_test": True,
                    "robust_val": {"enabled": True},
                },
            }
        )


def test_posthoc_calibration_requires_discount_probability_fusion():
    with pytest.raises(ValueError, match="require discount_probability fusion"):
        run_training(
            {
                "train": {"device": "cpu"},
                "data": {},
                "model": {"fusion_mode": "tri_modal_ours"},
                "fusion": {"mode": "model_dispatch"},
                "calibration": {"enabled": True},
                "eval": {"run_test": False, "run_robust_test": False},
            }
        )


def test_perturbation_evidence_is_forbidden_in_model_path():
    with pytest.raises(ValueError, match="diagnostic-only"):
        run_training(
            {
                "train": {"device": "cpu"},
                "data": {},
                "model": {"gate": {"use_perturbation_evidence": True}},
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
        "model": {"fusion_mode": "tri_modal_ours", "graph_encoder": {"use_behavior_hint": False}},
        "data": {"graph_semantic_source": "alignment"},
    }
    current = {
        "model": {"fusion_mode": "tri_modal_ours", "graph_encoder": {"use_behavior_hint": True}},
        "data": {"graph_semantic_source": "alignment"},
    }
    with pytest.raises(ValueError, match="changes model/data semantics"):
        validate_eval_checkpoint_config(current, saved)
    validate_eval_checkpoint_config(current, saved, allow_mismatch=True)


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
    batch.q_api = torch.ones(2, 1)
    batch.q_graph = torch.ones(2, 1)
    batch.q_manifest = torch.ones(2, 1)
    batch.q_align = torch.ones(2, 1) * 0.8
    batch.pert_api = torch.zeros(2, 1)
    batch.pert_graph = torch.zeros(2, 1)
    batch.pert_manifest = torch.zeros(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_ours",
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
        joint_emb_dim=32,
    )
    logits, extra = model(batch, return_features=True)
    loss, parts = compute_robust_loss(
        logits,
        torch.tensor([0, 1]),
        extra,
        {"branch_aux_weight": 0.05},
    )
    assert logits.shape == (2, 2)
    assert extra["gate_weights"].shape == (2, 4)
    assert extra["gate_evidence"].shape == (2, EvidenceIndex.BASE_DIM)
    assert torch.allclose(extra["gate_evidence"][:, 4], extra["api_graph_anchor_support"])
    assert "api_graph_consistency" not in extra
    assert extra["gate_prior_enabled"] is True
    assert "api_semantic_logits" not in extra
    assert extra["manifest_to_code_conflict"].shape == (2,)
    assert extra["code_to_manifest_conflict"].shape == (2,)
    assert torch.isfinite(loss)
    assert parts["branch_aux_weight"] == 0.05
    assert "cross_source_consistency_weight" not in parts
    assert "gate_prior_weight" not in parts

    batch.q_api = torch.full((2, 1), 0.8)
    batch.pert_api = torch.full((2, 1), 0.5)
    _, observable_extra = model(batch, return_features=False)
    assert torch.allclose(observable_extra["r_api"], torch.full((2,), 0.8))
    assert "pert_api" not in observable_extra
    assert torch.allclose(observable_extra["gate_uses_perturbation_evidence"], torch.zeros(2))

    with pytest.raises(ValueError, match="diagnostic-only"):
        TriModalRobustModel(
            in_feat_dim=16,
            fusion_mode="tri_modal_ours",
            use_perturbation_evidence=True,
        )

    missing_api_batch = batch.clone()
    missing_api_batch.q_api = torch.zeros(2, 1)
    _, missing_api_extra = model(missing_api_batch, return_features=True)
    assert torch.allclose(
        missing_api_extra["gate_weights"][:, 0],
        torch.zeros(2),
        atol=1e-6,
    )
    assert torch.allclose(
        missing_api_extra["gate_weights"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert torch.allclose(
        missing_api_extra["api_emb_for_joint"],
        torch.zeros_like(missing_api_extra["api_emb_for_joint"]),
        atol=1e-7,
    )
    assert torch.isfinite(missing_api_extra["joint_emb"]).all()

    missing_graph_batch = batch.clone()
    missing_graph_batch.q_graph = torch.zeros(2, 1)
    _, missing_graph_extra = model(missing_graph_batch, return_features=True)
    assert torch.allclose(
        missing_graph_extra["graph_emb_for_joint"],
        torch.zeros_like(missing_graph_extra["graph_emb_for_joint"]),
        atol=1e-7,
    )

    missing_manifest_batch = batch.clone()
    missing_manifest_batch.q_manifest = torch.zeros(2, 1)
    _, missing_manifest_extra = model(missing_manifest_batch, return_features=True)
    assert torch.allclose(
        missing_manifest_extra["manifest_emb_for_joint"],
        torch.zeros_like(missing_manifest_extra["manifest_emb_for_joint"]),
        atol=1e-7,
    )

    confidence_model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_confidence_gate",
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
        joint_emb_dim=32,
    )
    _, confidence_extra = confidence_model(batch, return_features=False)
    assert confidence_extra["gate_weights"].shape == (2, 4)
    assert torch.allclose(
        confidence_extra["gate_weights"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert confidence_extra["gate_prior_enabled"] is False


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


def test_robust_dataset_collate(tmp_path: Path):
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
        manifest_dim=16,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=True,
        manifest_dim=16,
        manifest_category_dim=12,
        manifest_stats_dim=11,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=robust_collate_fn)))
    graph = batch["graph_batch"]
    assert graph.manifest_x.shape == (1, 16)
    assert graph.q_manifest.shape == (1, 1)
    assert graph.api_ids.numel() == 3
    assert graph.api_semantic_category_counts.shape == (1, 12)
    assert graph.graph_semantic_category_counts.shape == (1, 12)
    assert graph.api_category_counts.shape == (1, 12)
    assert graph.graph_semantic_category_counts.sum().item() == 2.0


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
            manifest_dim=16,
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
        manifest_dim=16,
        allow_pt_superset=True,
    )
    assert len(dataset) == 1
    assert dataset.sample_sids == ["selected"]


def test_all_ghost_graph_has_integrity_but_is_not_alive(tmp_path: Path):
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
        manifest_dim=16,
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
        manifest_dim=16,
        eval_perturb_type="graph_node_feature_mask",
        eval_perturb_strength=0.5,
    )[0]
    assert data.real_num_nodes.item() == 0
    assert data.graph_integrity.item() > 0.0
    assert data.graph_alive.item() == 0.0
    assert data.q_align.item() == 0.0
    assert data.method_api_edge_index.numel() == 0


def test_multidex_api_limit_is_sample_level_and_preserves_alignment(tmp_path: Path):
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
        manifest_dim=16,
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
        manifest_dim=16,
        max_api_events_per_sample=3,
    )
    graph = next(iter(DataLoader(dataset, batch_size=1, collate_fn=robust_collate_fn)))["graph_batch"]
    # All 4 events are sensitive and the budget (3) forces a drop; the sensitive
    # tier keeps a contiguous prefix (indices 0,1,2 -> ids 1,2,3) and the
    # API<->graph alignment is remapped consistently.
    assert graph.api_ids.tolist() == [1, 2, 3]
    assert graph.method_api_edge_index[1].tolist() == [0, 1, 2]
    assert graph.method_api_edge_index[0].tolist() == [0, 1, 2]
    assert graph.graph_semantic_category_counts.sum().item() == 3.0
    assert graph.api_event_count_before_encoder_budget.item() == pytest.approx(4.0)
    assert graph.api_event_count_after_encoder_budget.item() == pytest.approx(3.0)
    assert graph.api_runtime_encoder_coverage.item() == pytest.approx(0.75)
    assert graph.api_encoder_coverage.item() == pytest.approx(0.75)
    assert graph.api_truncated_by_encoder_budget.item() == pytest.approx(1.0)


def test_manifest_perturbation_uses_payload_vocab_dims(tmp_path: Path):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample_manifest_dims"
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
            "manifest_x": torch.ones(16),
            "manifest_permission_dim": 2,
            "manifest_intent_dim": 1,
            "manifest_category_counts": torch.ones(12),
            "manifest_stats": torch.ones(11),
            "q_manifest": torch.tensor([1.0]),
            "pert_manifest": torch.tensor([0.0]),
        },
        manifest_dim=16,
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
        manifest_dim=16,
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
            "manifest_x": torch.ones(20),
            "manifest_category_counts": torch.ones(12),
            "manifest_stats": torch.ones(11),
        },
        manifest_dim=20,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 0, "year": 2024})

    dataset = RobustTriModalDataset(str(pt_dir), str(csv_path), is_train=False, manifest_dim=16)
    with pytest.raises(FatalDatasetConfigError, match="manifest_x dimension"):
        dataset[0]


def test_dataset_requires_current_manifest_semantic_maps(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="missing_manifest_maps")
    path = pt_dir / "missing_manifest_maps.pt"
    raw = torch.load(path, map_location="cpu", weights_only=False)
    del raw["manifest_permission_category_map"]
    torch.save(raw, path)
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )
    with pytest.raises(FatalDatasetConfigError, match="missing top-level fields"):
        dataset[0]


def test_dataset_rejects_non_current_pt_schema_version(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="old_schema")
    path = pt_dir / "old_schema.pt"
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["direct_build_meta"]["pt_schema_version"] = PT_SCHEMA_VERSION - 1
    torch.save(raw, path)
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        manifest_dim=16,
    )
    with pytest.raises(FatalDatasetConfigError, match="does not match required current version"):
        dataset[0]


def test_heuristic_joint_gate_uses_manifest_reliability():
    evidence = torch.zeros(2, 11)
    evidence[:, 0] = 1.0
    evidence[:, 1] = 1.0
    evidence[:, 3] = 1.0
    evidence[:, 4] = 1.0
    evidence[:, 5] = 1.0
    evidence[:, 8] = 1.0
    evidence[:, 9] = 1.0
    evidence[:, 10] = 1.0
    evidence[0, 2] = 1.0
    evidence[1, 2] = 0.0

    weights = heuristic_reliability_gate(evidence)
    assert weights[0, 3] > weights[1, 3]


def test_removed_legacy_loss_weights_are_rejected():
    logits = torch.zeros(2, 2, requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    extra = {"gate_weights_train": torch.full((2, 4), 0.25, requires_grad=True)}
    for cfg in (
        {"gate_prior_weight": 0.01},
        {"cross_source_consistency_weight": 0.05},
        {"semantic_reconstruction_weight": 0.1},
    ):
        with pytest.raises(ValueError, match="removed from the formal lean pipeline"):
            compute_robust_loss(logits, labels, extra, cfg)


def _perturbation_sample():
    return {
        "api_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "api_type_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "api_sensitive_mask": torch.ones(3),
        "api_method_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "api_in_graph_mask": torch.ones(3),
        "method_api_edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        "mask": torch.ones(3, 3),
        "api_semantic_category_counts": torch.ones(12),
        "api_category_counts": torch.ones(12),
        "x": torch.randn(3, 8),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "sensitive_mask": torch.ones(3, dtype=torch.uint8),
        "graph_semantic_category_counts": torch.ones(12),
        "graph_category_counts": torch.ones(12),
        "manifest_x": torch.ones(16),
        "manifest_permission_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "manifest_intent_ids": torch.tensor([1, 2], dtype=torch.long),
        "manifest_permission_category_map": torch.ones((4, 12)),
        "manifest_intent_category_map": torch.ones((2, 12)),
        "manifest_component_category_counts": torch.zeros(12),
        "manifest_category_counts": torch.ones(12),
        "manifest_stats": torch.ones(11),
        "manifest_permission_dim": 4,
        "manifest_intent_dim": 2,
        "manifest_feature_dim": 0,
        "q_api": 1.0,
        "q_graph": 1.0,
        "q_manifest": 1.0,
        "q_align": 0.8,
        "pert_api": 0.0,
        "pert_graph": 0.0,
        "pert_manifest": 0.0,
        "real_num_nodes": 3,
    }


def test_api_missing_sets_q_align_zero():
    data = apply_api_missing(_perturbation_sample())
    assert data["q_api"] == 0.0
    assert data["api_alive"] == 0.0
    assert data["api_integrity"] == 0.0
    assert data["code_integrity"] == 0.0
    assert data["pert_api"] == 1.0
    assert data["q_align"] == 0.0
    assert data["graph_semantic_category_counts"].sum().item() == 12.0


def test_api_event_dropout_removes_tokens_and_remaps_edges():
    torch.manual_seed(0)
    data = _perturbation_sample()
    data["api_ids"] = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    data["api_type_ids"] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    data["api_sensitive_mask"] = torch.ones(4)
    data["api_method_index"] = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    data["api_in_graph_mask"] = torch.ones(4)
    data["method_api_edge_index"] = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]], dtype=torch.long)
    data["mask"] = torch.ones(3, 4)

    out = apply_api_event_dropout(data, 0.5)
    assert out["api_ids"].numel() == 2
    assert out["api_type_ids"].numel() == 2
    assert out["mask"].shape == (3, 2)
    assert out["method_api_edge_index"].size(1) == 2
    assert out["method_api_edge_index"][1].max().item() < out["api_ids"].numel()
    assert out["q_api"] < 1.0
    assert out["q_api"] > 0.0
    assert out["q_align"] <= 0.8
    assert out["graph_semantic_category_counts"].sum().item() == 12.0


def test_sensitive_api_dropout_is_noop_when_no_sensitive_events_exist():
    data = _perturbation_sample()
    data["api_sensitive_mask"] = torch.zeros_like(data["api_sensitive_mask"])
    before = data["api_ids"].clone()
    out = apply_api_event_dropout(data, 1.0, sensitive_only=True)
    assert torch.equal(out["api_ids"], before)
    assert out["pert_api"] == 0.0


def test_graph_missing_sets_q_align_zero():
    data = apply_graph_missing(_perturbation_sample())
    assert data["q_graph"] == 0.0
    assert data["graph_alive"] == 0.0
    assert data["graph_integrity"] == 0.0
    assert data["code_integrity"] == 0.0
    assert data["pert_graph"] == 1.0
    assert data["q_align"] == 0.0


def test_align_quality_requires_explicit_method_api_edges():
    assert compute_align_quality(
        1.0,
        1.0,
        torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        num_api=4,
    ) == 0.0
    aligned = compute_align_quality(
        1.0,
        1.0,
        torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        num_nodes=4,
        num_api=4,
    )
    assert 0.0 < aligned < 1.0


def test_graph_degradation_preserves_category_counts():
    """Graph structural perturbations (obfuscation, sparsify, etc.) no longer
    artificially degrade semantic category counts.  Only graph quality drops;
    the category counts are tied to method_api_edge_index which is unchanged
    by structural edits."""
    torch.manual_seed(0)
    data = _perturbation_sample()
    before = data["graph_semantic_category_counts"].clone()
    out = apply_graph_feature_obfuscation(data, 0.5)
    # Category counts must NOT change — only structural quality is affected.
    assert torch.equal(before, out["graph_semantic_category_counts"])
    assert torch.equal(out["graph_semantic_category_counts"], out["graph_category_counts"])


def test_graph_sparsify_strength_one_removes_all_edges():
    data = _perturbation_sample()
    out = apply_graph_sparsify(data, 1.0)
    assert out["edge_index"].numel() == 0
    assert out["pert_graph"] == pytest.approx(1.0)


def test_aggregate_graph_degradation_retries_noop_operations():
    data = _perturbation_sample()
    data["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    out = apply_perturbation(data, "graph_degraded", 0.5)
    assert out["pert_graph"] > 0.0


def test_manifest_perturbation_updates_feature_vector_and_semantic_counts():
    """Manifest perturbations must keep feature vectors and category-count
    evidence in sync. Otherwise gate diagnostics would see stale consistency
    evidence under manifest_degraded tests."""
    data = _perturbation_sample()
    before = data["manifest_category_counts"].clone()
    before_x = data["manifest_x"].clone()
    data = apply_manifest_permission_mask(data, 0.5)
    assert not torch.equal(before, data["manifest_category_counts"])
    # The feature vector IS modified.
    assert not torch.equal(before_x, data["manifest_x"])
    assert data["q_manifest"] == 1.0
    assert data["pert_manifest"] == pytest.approx(0.5)


def test_sparse_manifest_mask_and_injection_always_change_eligible_positions():
    data = _perturbation_sample()
    data["manifest_x"] = torch.zeros(32)
    data["manifest_x"][[1, 7]] = 1.0
    data["manifest_permission_dim"] = 16
    data["manifest_permission_ids"] = torch.tensor([2, 8], dtype=torch.long)
    data["manifest_permission_category_map"] = torch.ones((16, 12))
    masked = apply_manifest_permission_mask(data, 0.5)
    assert int((masked["manifest_x"][:16] > 0).sum().item()) == 1
    assert masked["manifest_permission_ids"].numel() == 1
    assert masked["pert_manifest"] == pytest.approx(0.5)

    injected = apply_manifest_permission_injection(masked, 0.5)
    assert int((injected["manifest_x"][:16] > 0).sum().item()) > 1
    assert injected["manifest_permission_ids"].numel() > 1


def test_manifest_permission_mask_updates_semantic_counts_when_mapping_is_available():
    data = _perturbation_sample()
    data["manifest_x"] = torch.zeros(8)
    data["manifest_x"][0] = 1.0
    data["manifest_permission_dim"] = 2
    data["manifest_permission_ids"] = torch.tensor([1], dtype=torch.long)
    data["manifest_category_counts"] = torch.zeros(12)
    data["manifest_category_counts"][0] = 1.0
    mapping = torch.zeros(2, 12)
    mapping[0, 0] = 1.0
    data["manifest_permission_category_map"] = mapping
    out = apply_manifest_permission_mask(data, 1.0)
    assert out["manifest_category_counts"][0].item() == 0.0


def test_manifest_feature_noise_does_not_modify_padding():
    data = _perturbation_sample()
    data["manifest_x"] = torch.zeros(64)
    data["manifest_permission_dim"] = 4
    data["manifest_intent_dim"] = 3
    data["manifest_feature_dim"] = 2
    data["manifest_stats"] = torch.ones(2)
    raw_dim = 4 + 3 + 2 + 12 + 2
    out = apply_manifest_feature_noise(data, 1.0)
    assert torch.equal(out["manifest_x"][raw_dim:], torch.zeros(64 - raw_dim))


def test_zero_strength_degradation_is_noop():
    perturb_types = [
        "api_event_dropout",
        "api_sensitive_event_dropout",
        "api_category_dropout",
        "api_feature_noise",
        "graph_sparsify",
        "graph_local_break",
        "graph_feature_obfuscation",
        "graph_node_feature_mask",
        "manifest_permission_mask",
        "manifest_permission_injection",
        "manifest_intent_mask",
        "manifest_component_mask",
        "manifest_feature_noise",
        "api_degraded",
        "graph_degraded",
        "manifest_degraded",
        "api_graph_degraded",
        "api_manifest_degraded",
        "graph_manifest_degraded",
        "all_degraded",
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


def test_eval_perturbation_is_deterministic_per_sample(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="deterministic_eval")
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        robust_aug=False,
        manifest_dim=16,
        manifest_stats_dim=11,
        eval_perturb_type="all_degraded",
        eval_perturb_strength=0.5,
    )
    first = dataset[0]
    second = dataset[0]
    assert first.api_aug_type == second.api_aug_type
    assert first.graph_aug_type == second.graph_aug_type
    assert first.manifest_aug_type == second.manifest_aug_type
    assert torch.equal(first.api_ids, second.api_ids)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.manifest_x, second.manifest_x)
    stronger = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        robust_aug=False,
        manifest_dim=16,
        manifest_stats_dim=11,
        eval_perturb_type="all_degraded",
        eval_perturb_strength=0.9,
    )[0]
    assert first.api_aug_type == stronger.api_aug_type
    assert first.graph_aug_type == stronger.graph_aug_type
    assert first.manifest_aug_type == stronger.manifest_aug_type


def test_manifest_component_mask_uses_vector_layout_stats_offset():
    data = _perturbation_sample()
    data["manifest_x"] = torch.ones(256)
    data["manifest_stats"] = torch.ones(11)
    data["manifest_permission_dim"] = 128
    data["manifest_intent_dim"] = 64
    data["manifest_feature_dim"] = 32
    out = apply_manifest_component_mask(data, 1.0)
    stats_start = 128 + 64 + 32 + 12
    component_indices = torch.tensor([1, 2, 3, 4, 9, 10])
    non_component_indices = torch.tensor([0, 5, 6, 7, 8])
    stats_segment = out["manifest_x"][stats_start : stats_start + 11]
    assert stats_segment[component_indices].sum().item() == 0.0
    assert stats_segment[non_component_indices].sum().item() == 5.0
    assert out["manifest_x"][247:].sum().item() == 9.0


def test_manifest_component_mask_updates_component_semantic_counts():
    data = _perturbation_sample()
    data["manifest_x"] = torch.ones(256)
    data["manifest_stats"] = torch.ones(11)
    data["manifest_permission_dim"] = 128
    data["manifest_intent_dim"] = 64
    data["manifest_feature_dim"] = 32
    component_counts = torch.zeros(12)
    component_counts[7] = 1.0
    component_counts[8] = 1.0
    data["manifest_component_category_counts"] = component_counts
    data["manifest_category_counts"] = component_counts.clone()
    out = apply_manifest_component_mask(data, 1.0)
    assert out["manifest_category_counts"].sum().item() == 0.0
    assert out["manifest_component_category_counts"].sum().item() == 0.0


def test_manifest_missing_zeroes_manifest_counts_and_q_manifest():
    data = apply_manifest_missing(_perturbation_sample())
    assert data["manifest_category_counts"].sum().item() == 0.0
    assert data["q_manifest"] == 0.0
    assert data["manifest_alive"] == 0.0
    assert data["manifest_integrity"] == 0.0
    assert data["pert_manifest"] == 1.0


def test_removed_cross_source_and_semantic_losses_are_not_supported():
    logits = torch.zeros(1, 2, requires_grad=True)
    labels = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="removed from the formal lean pipeline"):
        compute_robust_loss(
            logits,
            labels,
            {},
            {"semantic_reconstruction_weight": 0.1, "cross_source_consistency_weight": 0.1},
        )


def test_loss_rejects_negative_auxiliary_weight():
    with pytest.raises(ValueError, match="cross_source_consistency_weight"):
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


def test_manifest_vectorization_stores_semantic_maps_and_coverage_quality():
    vocab = {
        "categories": list(DEFAULT_CATEGORIES),
        "permission_vocab": ["android.permission.internet"],
        "intent_vocab": [],
        "feature_vocab": [],
    }
    record = {
        "permissions": ["android.permission.internet", "unknown.permission"],
        "component_count": 1,
    }
    required = 1 + len(DEFAULT_CATEGORIES) + 11
    payload = vectorize_manifest_record(record, vocab, manifest_dim=required)
    assert payload["manifest_permission_category_map"].shape == (1, 12)
    assert payload["manifest_permission_category_map"][:, CATEGORY_TO_INDEX["network"]].sum().item() > 0
    assert payload["manifest_component_category_counts"].shape == (12,)
    assert payload["q_manifest"].item() == pytest.approx(0.75)


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
        }
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

    del payload["manifest_permission_category_map"]
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
        }
    )
    stale["direct_build_meta"]["build_fingerprint"] = "stale"
    torch.save(stale, path)

    def fake_process_apk(*_args, **_kwargs):
        current = current_pt_payload(stale["dex_list"], manifest_dim=32)
        current["direct_build_meta"]["build_fingerprint"] = "temporary-code-stage"
        torch.save(current, path)
        return True, ""

    monkeypatch.setattr(
        "scripts.build_tri_modal_pts_direct.process_apk",
        fake_process_apk,
    )
    result = _build_one(
        job,
        {"component_count": 1},
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


# ---------------------------------------------------------------------------
# P0.2: graph_semantic_source ablation switch
# ---------------------------------------------------------------------------


def _make_graph_source_pt(tmp_path: Path, sid: str = "graph_src_sample"):
    """Build a tiny .pt with 4 API events but only 2 of them aligned to the
    graph via method_api_edge_index. The alignment vs. full-API distinction
    must be observable.

    API type_ids: [1=telephony, 2=sms, 3=location, 6=network]
    method_api_edge_index[1] = [0, 3]  -> only telephony + network are aligned.
    """
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
        manifest_dim=16,
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})
    return pt_dir, csv_path


def _load_single_sample(pt_dir: Path, csv_path: Path, graph_semantic_source: str):
    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=False,
        robust_aug=False,
        manifest_dim=16,
        manifest_category_dim=12,
        manifest_stats_dim=11,
        graph_semantic_source=graph_semantic_source,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=robust_collate_fn)))
    return batch["graph_batch"]


def test_graph_semantic_source_alignment_uses_method_api_edges(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="graph_align")
    graph = _load_single_sample(pt_dir, csv_path, "alignment")
    api_counts = graph.api_semantic_category_counts[0]
    graph_counts = graph.graph_semantic_category_counts[0]
    # API sees telephony, sms, location, network (4 categories).
    assert api_counts[CATEGORY_TO_INDEX["telephony"]].item() == 1.0
    assert api_counts[CATEGORY_TO_INDEX["sms"]].item() == 1.0
    assert api_counts[CATEGORY_TO_INDEX["location"]].item() == 1.0
    assert api_counts[CATEGORY_TO_INDEX["network"]].item() == 1.0
    # Graph alignment only retains the two anchored events: telephony, network.
    assert graph_counts[CATEGORY_TO_INDEX["telephony"]].item() == 1.0
    assert graph_counts[CATEGORY_TO_INDEX["network"]].item() == 1.0
    assert graph_counts[CATEGORY_TO_INDEX["sms"]].item() == 0.0
    assert graph_counts[CATEGORY_TO_INDEX["location"]].item() == 0.0
    # Alignment and full-API must NOT agree on this sample.
    assert not torch.equal(api_counts, graph_counts)


def test_graph_semantic_source_full_api_copies_api_counts(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="graph_full_api")
    graph = _load_single_sample(pt_dir, csv_path, "full_api")
    api_counts = graph.api_semantic_category_counts[0]
    graph_counts = graph.graph_semantic_category_counts[0]
    assert torch.equal(api_counts, graph_counts)
    # Sanity: with the same sample, alignment yields a different distribution.
    graph_align = _load_single_sample(pt_dir, csv_path, "alignment").graph_semantic_category_counts[0]
    assert not torch.equal(graph_counts, graph_align)


def test_graph_semantic_source_zero_returns_all_zeros(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="graph_zero")
    graph = _load_single_sample(pt_dir, csv_path, "zero")
    assert torch.all(graph.graph_semantic_category_counts == 0.0)
    # API counts must still be populated — only the graph branch is zeroed.
    assert graph.api_semantic_category_counts.abs().sum().item() > 0.0


def test_graph_semantic_source_rejects_invalid_value(tmp_path: Path):
    pt_dir, csv_path = _make_graph_source_pt(tmp_path, sid="graph_bad_src")
    with pytest.raises(ValueError, match="Unsupported graph_semantic_source"):
        RobustTriModalDataset(
            str(pt_dir),
            str(csv_path),
            is_train=False,
            manifest_dim=16,
            manifest_category_dim=12,
            manifest_stats_dim=11,
            graph_semantic_source="not_a_real_source",
        )


# ── gradient-flow & empty-input integration tests ──────────────────────


def test_learned_gate_receives_gradient():
    """Backward pass must deliver non-zero gradients to gate_net parameters."""
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
    batch.pert_api = torch.zeros(2, 1)
    batch.pert_graph = torch.zeros(2, 1)
    batch.pert_manifest = torch.zeros(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_ours",
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
        joint_emb_dim=32,
    )

    # Verify gate_net has trainable parameters.
    gate_params = list(model.gate_net.parameters())
    assert len(gate_params) > 0, "gate_net must have parameters"

    logits, extra = model(batch, return_features=True)
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
    assert nonzero_grads > 0, "gate_net received zero gradients — possible detach bug"


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
    batch.pert_api = torch.zeros(2, 1)
    batch.pert_graph = torch.zeros(2, 1)
    batch.pert_manifest = torch.zeros(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_ours",
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
        joint_emb_dim=32,
    )

    logits, extra = model(batch, return_features=True)
    assert logits.shape == (2, 2), "output shape must be (batch, num_classes)"
    assert torch.isfinite(logits).all(), "logits must be finite with empty API"
    assert extra["gate_weights"].shape == (2, 4)
    assert torch.isfinite(extra["gate_weights"]).all()

    # Also test with zero-node graphs.
    batch_zero = batch.clone()
    batch_zero.x = torch.empty((0, 16))
    batch_zero.edge_index = torch.empty((2, 0), dtype=torch.long)
    batch_zero.batch = torch.empty((0,), dtype=torch.long)
    batch_zero.sensitive_mask = torch.empty((0,), dtype=torch.uint8)

    logits_zero, extra_zero = model(batch_zero, return_features=True)
    assert logits_zero.shape == (2, 2)
    assert torch.isfinite(logits_zero).all(), "logits must be finite with zero-node graph"
    assert extra_zero["gate_weights"].shape == (2, 4)
    assert torch.isfinite(extra_zero["gate_weights"]).all()
