from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Batch, Data

import fusion.dataset as dataset_module
from fusion.anchored_training import (
    CachedExpertBatch,
    _competence_loss_statistics,
    _normalized_clean_degraded_loss,
    _select_competence_candidate,
    cache_expert_outputs,
    competence_diagnostics,
    evaluate_cached_fusion,
    fit_anchored_router,
    fit_competence_heads,
)
from fusion.competence_fusion import EXPERT_NAMES
from fusion.dataset import RobustTriModalDataset
from fusion.model import TriModalRobustModel
from fusion.train import (
    _batch_selective_score,
    _build_train_single_degradation_view,
    _load_fixed_validation_roles,
)
from tests.pt_factory import save_current_pt


def _small_anchored_model() -> TriModalRobustModel:
    return TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="anchored_joint_late",
        api_num_hash_buckets=64,
        api_type_vocab_size=16,
        api_emb_dim=8,
        api_hidden_dim=8,
        api_dropout=0.0,
        api_layers=1,
        api_heads=1,
        api_max_seq_len=16,
        graph_emb_dim=8,
        graph_hidden=8,
        graph_heads=1,
        graph_layers=1,
        max_nodes_gnn=32,
        manifest_in_dim=32,
        manifest_emb_dim=8,
        manifest_hidden_dim=8,
        manifest_dropout=0.0,
        joint_emb_dim=8,
        joint_hidden_dim=12,
        joint_dropout=0.0,
        competence_projection_dim=4,
        competence_hidden_dim=4,
        competence_dropout=0.0,
    )


def _model_batch(
    *,
    api_alive: tuple[bool, bool] = (True, True),
    graph_alive: tuple[bool, bool] = (True, True),
    manifest_alive: tuple[bool, bool] = (True, True),
) -> Batch:
    graphs: list[Data] = []
    for label in (0, 1):
        graph = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 1], [1, 2, 3, 0, 3]],
                dtype=torch.long,
            ),
            y=torch.tensor(label, dtype=torch.long),
        )
        graph.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)
    batch.api_ids = torch.randint(1, 32, (12,), dtype=torch.long)
    batch.api_type_ids = torch.randint(0, 4, (12,), dtype=torch.long)
    batch.api_sensitive_mask = torch.zeros(12)
    batch.api_batch = torch.cat(
        [torch.full((6,), index, dtype=torch.long) for index in range(2)]
    )
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_semantic_category_counts = torch.rand(2, 12)
    batch.graph_semantic_category_counts = torch.rand(2, 12)
    batch.api_category_counts = batch.api_semantic_category_counts
    batch.graph_category_counts = batch.graph_semantic_category_counts
    batch.manifest_x = torch.rand(2, 32)
    batch.manifest_category_counts = torch.rand(2, 12)
    batch.manifest_stats = torch.rand(2, 11)
    batch.api_alive = torch.tensor(api_alive, dtype=torch.float32).view(2, 1)
    batch.graph_alive = torch.tensor(graph_alive, dtype=torch.float32).view(2, 1)
    batch.manifest_alive = torch.tensor(
        manifest_alive, dtype=torch.float32
    ).view(2, 1)
    return batch


def _cached_batch(*, degraded: bool = False) -> CachedExpertBatch:
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    generator = torch.Generator().manual_seed(101 if degraded else 100)
    embeddings = {
        name: torch.randn(4, 8, generator=generator) for name in EXPERT_NAMES
    }
    clean_logits = {
        "api": torch.tensor([[2.0, -1.0], [-1.0, 2.0], [1.5, -0.5], [-0.5, 1.5]]),
        "graph": torch.tensor([[1.2, -0.2], [-0.3, 1.3], [1.0, 0.0], [0.1, 0.9]]),
        "manifest": torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]]),
        "joint": torch.tensor([[2.2, -1.2], [-1.1, 2.1], [1.8, -0.8], [-0.7, 1.7]]),
    }
    logits = {name: value.float().clone() for name, value in clean_logits.items()}
    if degraded:
        # One deterministic current-view degradation: the API expert becomes
        # misleading while all cached tensors remain finite and immutable.
        logits["api"] = logits["api"].flip(dims=(-1,))
        embeddings["api"] = embeddings["api"] * 0.25
    alive = {
        name: torch.ones(labels.numel(), dtype=torch.bool)
        for name in EXPERT_NAMES
    }
    return CachedExpertBatch(
        labels=labels,
        embeddings=embeddings,
        logits=logits,
        alive=alive,
    )


def _slice_cached_batch(
    batch: CachedExpertBatch,
    indices: list[int],
) -> CachedExpertBatch:
    index = torch.tensor(indices, dtype=torch.long)
    return CachedExpertBatch(
        labels=batch.labels[index],
        embeddings={
            name: value[index] for name, value in batch.embeddings.items()
        },
        logits={name: value[index] for name, value in batch.logits.items()},
        alive={name: value[index] for name, value in batch.alive.items()},
    )


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _assert_state_equal(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    assert tuple(before) == tuple(after)
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_stage_a_uses_joint_expert_and_excludes_stage_b_parameters() -> None:
    torch.manual_seed(11)
    model = _small_anchored_model()
    model.train()
    graph = _model_batch()

    frozen = model.encoder_training_frozen_parameters()
    frozen_ids = {id(parameter) for parameter in frozen}
    expected_frozen_ids = {
        id(parameter)
        for module in (model.competence_estimator, model.anchored_fusion)
        for parameter in module.parameters()
    }
    assert frozen_ids == expected_frozen_ids
    for parameter in frozen:
        parameter.requires_grad_(False)

    logits, extra = model(graph)
    assert extra["final_is_log_probability"] is True
    assert torch.equal(
        extra["joint_alive"],
        torch.ones(2, dtype=torch.bool),
    )
    assert torch.allclose(
        logits,
        F.log_softmax(extra["joint_logits_aux"], dim=-1),
        atol=1.0e-6,
    )
    assert model.joint_expert.representation[1].in_features == 8 * 3 + 3

    stage_a_keys = model.encoder_stage_state_keys()
    assert any(key.startswith("joint_expert.") for key in stage_a_keys)
    assert not any(key.startswith("competence_estimator.") for key in stage_a_keys)
    assert not any(key.startswith("anchored_fusion.") for key in stage_a_keys)

    loss = F.cross_entropy(logits, graph.y.view(-1))
    loss.backward()
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.joint_expert.parameters()
    )
    assert all(parameter.grad is None for parameter in frozen)


def test_stage_b_fits_only_its_declared_parameter_partition() -> None:
    torch.manual_seed(17)
    device = torch.device("cpu")
    model = _small_anchored_model().to(device)
    clean = _cached_batch()
    degraded = _cached_batch(degraded=True)
    validation = replace(clean)

    expert_before = _clone_state(model.encoder_stage_state_dict())
    fit_competence_heads(
        model,
        train_clean=[clean],
        train_degraded=[degraded],
        validation_sources={
            "clean": [validation],
            "api_event_dropout": [degraded],
            "graph_sparsify": [degraded],
            "manifest_permission_mask": [degraded],
        },
        clean_validation_source="clean",
        device=device,
        config={
            "epochs": 2,
            "patience": 1,
            "lr": 1.0e-2,
            "weight_decay": 0.0,
            "degraded_loss_weight": 0.25,
            "ranking_weight": 0.1,
            "ranking_tie_tolerance": 0.02,
            "regression": "mse",
            "grad_clip": 5.0,
        },
    )
    _assert_state_equal(expert_before, model.encoder_stage_state_dict())
    competence_before_router = _clone_state(
        model.competence_estimator.state_dict()
    )

    fit_anchored_router(
        model,
        train_clean=[clean],
        train_degraded=[degraded],
        validation_sources={
            "clean": [validation],
            "degraded": [degraded],
        },
        clean_validation_source="clean",
        device=device,
        config={
            "degradation_loss_weights": [0.1],
            "epochs": 1,
            "patience": 1,
            "lr": 1.0e-2,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
            "clean_anchor_kl_weight": 0.1,
            "clean_noninferiority_tolerance": 1.0,
        },
    )
    _assert_state_equal(expert_before, model.encoder_stage_state_dict())
    _assert_state_equal(
        competence_before_router,
        model.competence_estimator.state_dict(),
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.competence_estimator.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.anchored_fusion.parameters()
    )


def test_stage_b_competence_diagnostics_cover_all_four_tcp_heads() -> None:
    model = _small_anchored_model()
    diagnostics = competence_diagnostics(
        model,
        {"train_clean": [_cached_batch()]},
        torch.device("cpu"),
    )

    assert set(diagnostics) == {"train_clean"}
    assert set(diagnostics["train_clean"]) == set(EXPERT_NAMES)
    for expert in EXPERT_NAMES:
        current = diagnostics["train_clean"][expert]
        assert current["defined"] is True
        assert current["num_rows"] == 4
        assert current["tcp_mse"] >= 0.0
        assert current["tcp_mae"] >= 0.0


def test_i1_source_loss_is_invariant_to_cached_batch_boundaries() -> None:
    torch.manual_seed(18)
    model = _small_anchored_model()
    source = _cached_batch()
    # Make valid expert-row and ranking-pair counts differ between the two
    # batches so a mean-of-batch-means implementation would change the result.
    source.alive["api"][0] = False
    source.alive["graph"][1] = False
    source.alive["manifest"][2] = False
    whole = _competence_loss_statistics(
        model,
        [source],
        torch.device("cpu"),
        regression="mse",
        ranking_weight=0.1,
        tie_tolerance=0.02,
    )
    split = _competence_loss_statistics(
        model,
        [
            _slice_cached_batch(source, [0]),
            _slice_cached_batch(source, [1, 2, 3]),
        ],
        torch.device("cpu"),
        regression="mse",
        ranking_weight=0.1,
        tie_tolerance=0.02,
    )

    assert split["valid_expert_rows"] == whole["valid_expert_rows"]
    assert split["valid_atomic_pairs"] == whole["valid_atomic_pairs"]
    assert split["regression_loss"] == pytest.approx(
        whole["regression_loss"], abs=1.0e-7
    )
    assert split["ranking_loss"] == pytest.approx(
        whole["ranking_loss"], abs=1.0e-7
    )
    assert split["total_loss"] == pytest.approx(
        whole["total_loss"], abs=1.0e-7
    )


def test_i1_selection_uses_relative_clean_band_then_degraded_losses() -> None:
    def candidate(epoch: int, clean: float, degraded: tuple[float, float, float]):
        return {
            "epoch": epoch,
            "state": {"epoch": torch.tensor(epoch)},
            "validation_sources": {
                "clean": {"total_loss": clean},
                "api": {"total_loss": degraded[0]},
                "graph": {"total_loss": degraded[1]},
                "manifest": {"total_loss": degraded[2]},
            },
        }

    selected, public, clean_reference = _select_competence_candidate(
        [
            candidate(1, 1.000, (0.9, 0.9, 0.9)),
            # Inside the 1% clean band and more robust.
            candidate(2, 1.005, (0.4, 0.5, 0.6)),
            # Best degraded loss but outside the clean-safe band.
            candidate(3, 1.020, (0.1, 0.1, 0.1)),
        ],
        clean_source="clean",
        clean_noninferiority_relative_tolerance=0.01,
        clean_noninferiority_absolute_tolerance=0.0,
    )

    assert clean_reference == pytest.approx(1.0)
    assert selected["epoch"] == 2
    assert [row["clean_noninferior"] for row in public] == [True, True, False]


def test_i1_diagnostics_remain_exported_when_i2_guard_falls_back() -> None:
    torch.manual_seed(20)
    model = _small_anchored_model()
    cache = _cached_batch()
    fit_competence_heads(
        model,
        train_clean=[cache],
        train_degraded=[],
        validation_sources={"clean": [cache]},
        clean_validation_source="clean",
        device=torch.device("cpu"),
        config={
            "epochs": 1,
            "patience": 1,
            "degraded_loss_weight": 0.0,
        },
    )
    model.set_anchored_fusion_active(False)
    model.eval()
    _logits, extra = model(_model_batch())

    assert model.competence_diagnostics_active is True
    for expert in EXPERT_NAMES:
        assert f"predicted_competence_{expert}" in extra
    assert "late_competence" not in extra


def test_cached_fusion_fits_separate_candidate_and_joint_clean_thresholds() -> None:
    torch.manual_seed(21)
    model = _small_anchored_model()
    cache = _cached_batch()
    metrics = evaluate_cached_fusion(
        model,
        {"clean": [cache], "api_event_dropout": [_cached_batch(degraded=True)]},
        torch.device("cpu"),
        clean_source="clean",
    )

    assert metrics["classification_threshold"]["num_calibration"] == 4
    assert metrics["joint_anchor_classification_threshold"][
        "num_calibration"
    ] == 4
    assert len(metrics["sources"]["clean"]["rows"]["probability"]) == 4
    assert len(
        metrics["sources"]["clean"]["rows"]["joint_anchor_probability"]
    ) == 4
    candidate_threshold = metrics["classification_threshold"]["threshold"]
    joint_threshold = metrics["joint_anchor_classification_threshold"][
        "threshold"
    ]
    clean_rows = metrics["sources"]["clean"]["rows"]
    labels = torch.tensor(clean_rows["labels"]).numpy()
    candidate_probability = torch.tensor(clean_rows["probability"]).numpy()
    joint_probability = torch.tensor(
        clean_rows["joint_anchor_probability"]
    ).numpy()
    assert metrics["sources"]["clean"]["macro_f1"] == pytest.approx(
        f1_score(
            labels,
            candidate_probability >= candidate_threshold,
            average="macro",
        )
    )
    assert metrics["sources"]["clean"][
        "joint_anchor_macro_f1"
    ] == pytest.approx(
        f1_score(
            labels,
            joint_probability >= joint_threshold,
            average="macro",
        )
    )


def test_i2_scenario_weight_changes_relative_not_total_loss_scale() -> None:
    clean = torch.tensor(2.0, requires_grad=True)
    degraded = torch.tensor(4.0, requires_grad=True)

    loss = _normalized_clean_degraded_loss(
        clean,
        degraded,
        degraded_weight=3.0,
    )
    loss.backward()

    assert loss.item() == pytest.approx(3.5)
    assert clean.grad.item() == pytest.approx(0.25)
    assert degraded.grad.item() == pytest.approx(0.75)
    assert _normalized_clean_degraded_loss(
        clean.detach(),
        None,
        degraded_weight=0.0,
    ).item() == pytest.approx(2.0)


def test_router_selection_guard_fails_closed_to_joint_only() -> None:
    torch.manual_seed(19)
    device = torch.device("cpu")
    model = _small_anchored_model().to(device)
    clean = _cached_batch()
    degraded = _cached_batch(degraded=True)

    summary = fit_anchored_router(
        model,
        train_clean=[clean],
        train_degraded=[degraded],
        validation_sources={
            "clean": [clean],
            "degraded": [degraded],
        },
        clean_validation_source="clean",
        device=device,
        config={
            "degradation_loss_weights": [0.0],
            "epochs": 1,
            "patience": 1,
            "lr": 1.0e-2,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
            "clean_anchor_kl_weight": 0.1,
            "clean_noninferiority_tolerance": 0.0,
            # Macro-F1 gain cannot exceed 1, so no candidate can pass.
            "minimum_robust_gain": 2.0,
        },
    )

    assert (
        summary["deployment"]
        == "joint_with_uniform_missing_fallback_selection_guard"
    )
    assert summary["selected"] is None
    assert model.anchored_fusion_active is False
    assert summary["classification_threshold"]["locked_by_stage_b"] is True
    assert summary["classification_threshold"]["prediction_source"] == (
        "joint_anchor_selection_guard_fallback"
    )
    assert summary["classification_threshold"]["threshold"] == pytest.approx(
        summary["joint_anchor_classification_threshold"]["threshold"]
    )


def test_cache_expert_outputs_is_detached_cpu_only_and_restores_lifecycle() -> None:
    torch.manual_seed(23)
    device = torch.device("cpu")
    model = _small_anchored_model().to(device)
    model.set_anchored_fusion_active(True)
    graph = _model_batch()
    raw_loader = [
        {
            "graph_batch": graph,
            "labels": graph.y.view(-1).clone(),
            "sids": ["sample-0", "sample-1"],
            "num_failed": 0,
        }
    ]
    expert_before = _clone_state(model.encoder_stage_state_dict())

    cache, summary = cache_expert_outputs(
        model,
        raw_loader,
        device,
        use_amp=False,
        source_name="unit-clean",
    )

    assert model.anchored_fusion_active is True
    assert summary["num_batches"] == 1
    assert summary["num_samples"] == 2
    assert len(cache) == 1
    assert tuple(cache[0].embeddings) == EXPERT_NAMES
    assert tuple(cache[0].logits) == EXPERT_NAMES
    assert tuple(cache[0].alive) == EXPERT_NAMES
    for name in EXPERT_NAMES:
        assert cache[0].embeddings[name].device.type == "cpu"
        assert cache[0].logits[name].device.type == "cpu"
        assert cache[0].alive[name].device.type == "cpu"
        assert not cache[0].embeddings[name].requires_grad
        assert not cache[0].logits[name].requires_grad
    _assert_state_equal(expert_before, model.encoder_stage_state_dict())


def _write_train_dataset(tmp_path: Path) -> RobustTriModalDataset:
    sid = "sample-001"
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    num_nodes = 8
    num_api = 20
    edge_sources = torch.arange(0, 16, dtype=torch.long) % num_nodes
    edge_targets = (edge_sources + 1) % num_nodes
    save_current_pt(
        pt_dir / f"{sid}.pt",
        {
            "call_x": torch.randn(num_nodes, 16),
            "call_edge_index": torch.stack([edge_sources, edge_targets]),
            "call_sensitive_mask": torch.zeros(num_nodes, dtype=torch.uint8),
            "api_ids": torch.arange(1, num_api + 1, dtype=torch.long),
            "api_type_ids": torch.arange(num_api, dtype=torch.long) % 4,
            "api_sensitive_mask": torch.zeros(num_api),
            "manifest_permission_dim": 10,
            "manifest_intent_dim": 1,
            "manifest_feature_dim": 0,
            "manifest_permission_ids": torch.arange(1, 9, dtype=torch.long),
            "manifest_permission_token_ids": torch.arange(1, 9, dtype=torch.long),
            "manifest_stats": torch.tensor(
                [math.log1p(8) / 6.0, *([1.0] * 10)],
                dtype=torch.float32,
            ),
            "manifest_meta": {
                "permissions": [f"permission.test.{index}" for index in range(1, 9)]
            },
        },
        manifest_dim=40,
    )
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})
    return RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=True,
        manifest_dim=40,
        manifest_permission_dim=10,
        manifest_intent_dim=1,
        manifest_feature_dim=0,
        max_graph_nodes_per_sample=32,
    )


@pytest.mark.parametrize(
    ("mechanism", "expected_fields"),
    [
        (
            "api_event_dropout",
            ("api_event_dropout", "none", "none"),
        ),
        (
            "graph_sparsify",
            ("none", "graph_sparsify", "none"),
        ),
        (
            "manifest_permission_mask",
            ("none", "none", "manifest_permission_mask"),
        ),
    ],
)
def test_train_degradation_is_single_modality_and_identity_deterministic(
    tmp_path: Path,
    mechanism: str,
    expected_fields: tuple[str, str, str],
) -> None:
    base = _write_train_dataset(tmp_path)
    view = _build_train_single_degradation_view(
        base,
        {
            "mechanisms": [mechanism],
            "strength_min": 0.37,
            "strength_max": 0.37,
        },
        seed=321,
    )

    first = view[0]
    second = view[0]
    assert (
        first.api_aug_type,
        first.graph_aug_type,
        first.manifest_aug_type,
    ) == expected_fields
    assert (
        second.api_aug_type,
        second.graph_aug_type,
        second.manifest_aug_type,
    ) == expected_fields
    for tensor_name in (
        "x",
        "edge_index",
        "api_ids",
        "api_type_ids",
        "api_sensitive_mask",
        "manifest_x",
    ):
        assert torch.equal(getattr(first, tensor_name), getattr(second, tensor_name))

    clean = base[0]
    assert (
        clean.api_aug_type,
        clean.graph_aug_type,
        clean.manifest_aug_type,
    ) == ("none", "none", "none")
    assert base.train_perturbations == ()


def test_train_degradation_draws_continuous_strength_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _write_train_dataset(tmp_path)
    view = _build_train_single_degradation_view(
        base,
        {
            "mechanisms": ["api_event_dropout"],
            "strength_min": 0.13,
            "strength_max": 0.87,
        },
        seed=1234,
    )
    real_apply = dataset_module.apply_perturbation
    observed: list[tuple[str, float]] = []

    def recording_apply(
        data: dict,
        perturb_type: str,
        strength: float,
    ) -> dict:
        observed.append((perturb_type, float(strength)))
        return real_apply(data, perturb_type, strength)

    monkeypatch.setattr(dataset_module, "apply_perturbation", recording_apply)
    first = view[0]
    second = view[0]

    assert len(observed) == 2
    assert observed[0] == observed[1]
    assert observed[0][0] == "api_event_dropout"
    assert 0.13 < observed[0][1] < 0.87
    assert torch.equal(first.api_ids, second.api_ids)


def test_anchored_model_all_dead_is_normalized_uniform_log_probability() -> None:
    torch.manual_seed(29)
    model = _small_anchored_model().eval()
    all_dead = _model_batch(
        api_alive=(False, False),
        graph_alive=(False, False),
        manifest_alive=(False, False),
    )

    model.set_anchored_fusion_active(True)
    with torch.no_grad():
        logits, extra = model(all_dead)

    assert extra["final_is_log_probability"] is True
    assert extra["selective_eligible"].tolist() == [False, False]
    assert extra["joint_alive"].tolist() == [False, False]
    assert torch.equal(extra["joint_weight"], torch.zeros(2))
    assert torch.equal(extra["fusion_weight_joint"], torch.zeros(2))
    assert torch.equal(extra["gate_weights"], torch.zeros(2, 3))
    assert torch.allclose(
        logits.exp(),
        torch.full_like(logits, 0.5),
        atol=1.0e-6,
    )
    assert torch.allclose(
        torch.logsumexp(logits, dim=-1),
        torch.zeros(2),
        atol=1.0e-6,
    )

    model.set_anchored_fusion_active(False)
    with torch.no_grad():
        fallback_logits, fallback = model(all_dead)
    assert torch.allclose(
        fallback_logits.exp(),
        torch.full_like(fallback_logits, 0.5),
        atol=1.0e-6,
    )
    assert torch.equal(fallback["late_gate"], torch.zeros(2))
    assert torch.equal(fallback["fusion_weight_joint"], torch.zeros(2))
    assert torch.equal(fallback["gate_weights"], torch.zeros(2, 3))


def test_joint_ineligible_uses_explicit_atomic_missing_modality_fallback() -> None:
    torch.manual_seed(31)
    model = _small_anchored_model().eval()
    graph_missing = _model_batch(
        graph_alive=(False, False),
    )

    model.set_anchored_fusion_active(False)
    with torch.no_grad():
        logits, extra = model(graph_missing)
    expected = 0.5 * (
        extra["expert_probabilities"]["api"]
        + extra["expert_probabilities"]["manifest"]
    )
    assert torch.allclose(logits.exp(), expected, atol=1.0e-6)
    assert torch.equal(extra["joint_weight"], torch.zeros(2))
    assert torch.allclose(extra["late_weight_api"], torch.full((2,), 0.5))
    assert torch.allclose(extra["late_weight_graph"], torch.zeros(2))
    assert torch.allclose(
        extra["late_weight_manifest"], torch.full((2,), 0.5)
    )

    model.set_anchored_fusion_active(True)
    with torch.no_grad():
        routed_logits, routed_extra = model(graph_missing)
    assert torch.equal(routed_extra["late_gate"], torch.ones(2))
    assert torch.equal(routed_extra["fusion_weight_joint"], torch.zeros(2))
    assert torch.allclose(
        torch.logsumexp(routed_logits, dim=-1),
        torch.zeros(2),
        atol=1.0e-6,
    )


class _ValidationMetadata:
    def __init__(self) -> None:
        self.sample_sids = ["a", "b", "c", "d"]
        self.sample_groups = ["group-a", "group-b", "group-c", "group-d"]
        self.sample_labels = [0, 1, 0, 1]
        self.sample_years = [2022, 2022, 2023, 2023]

    def __len__(self) -> int:
        return len(self.sample_sids)

    def __getitem__(self, index: int) -> str:
        return self.sample_sids[index]


def test_validation_roles_v2_are_two_disjoint_roles_and_v1_is_rejected(
    tmp_path: Path,
) -> None:
    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "id,label,year\n"
        "a,0,2022\n"
        "b,1,2022\n"
        "c,0,2023\n"
        "d,1,2023\n",
        encoding="utf-8",
    )
    csv_sha = hashlib.sha256(val_csv.read_bytes()).hexdigest()
    role_path = tmp_path / "roles.json"
    role_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "validation_csv_sha256": csv_sha,
                "split_seed": 42,
                "roles": {
                    "model_selection": ["a", "b", "c"],
                    "decision_calibration": ["d"],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "data": {"root": str(tmp_path), "val_csv": "val.csv"},
        "calibration": {
            "role_assignment_path": "roles.json",
            "require_role_assignment": True,
        },
    }

    model_selection, upstream_selection, decision, summary = (
        _load_fixed_validation_roles(cfg, _ValidationMetadata())
    )
    assert list(model_selection.indices) == [0, 1, 2]
    assert list(upstream_selection.indices) == [0, 1, 2]
    assert list(decision.indices) == [3]
    assert set(model_selection.indices).isdisjoint(decision.indices)
    assert summary["role_assignment_schema_version"] == 2
    assert summary["num_posthoc_calibration"] == 0

    role_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation_csv_sha256": csv_sha,
                "roles": {
                    "selection": ["a", "b"],
                    "posthoc_calibration": ["c"],
                    "decision_calibration": ["d"],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        _load_fixed_validation_roles(cfg, _ValidationMetadata())


def test_i3_malware_fn_probability_anchor_matches_deployed_decision() -> None:
    probability = torch.tensor([0.05, 0.30, 0.399, 0.40, 0.80])
    predicted_malware = torch.tensor([False, False, False, True, True])

    score = _batch_selective_score(
        probability,
        {},
        "malware_fn_probability_anchor",
        classification_threshold=0.40,
        predicted_malware_override=predicted_malware,
    )

    assert torch.allclose(score[:3], 1.0 - probability[:3])
    assert torch.equal(score[3:], torch.ones(2))
    assert score[0] > score[1] > score[2]
    assert bool(torch.isfinite(score).all())
    assert bool(((score >= 0.0) & (score <= 1.0)).all())
