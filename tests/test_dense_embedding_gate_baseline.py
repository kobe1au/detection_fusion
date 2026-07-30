from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from fusion.gates import (
    DenseTriModalEmbeddingGate,
    dense_embedding_late_fusion_logits,
)
from fusion.opinion_router import GlobalOpinionRouter
from fusion.model import TriModalRobustModel
from fusion.train import (
    _validate_selective_score_fusion_compatibility,
    build_model,
    load_config_path,
)


ROOT = Path("config/experiments/tri_modal_robust")


def _embeddings(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    return {
        "api": torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            requires_grad=requires_grad,
        ),
        "graph": torch.tensor(
            [[-1.0, 1.0], [4.0, 2.0], [0.0, 0.0]],
            requires_grad=requires_grad,
        ),
        "manifest": torch.tensor(
            [
                [0.5, 1.0, 1.5, 2.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            requires_grad=requires_grad,
        ),
    }


def _gate(*, detach_embeddings: bool = False) -> DenseTriModalEmbeddingGate:
    return DenseTriModalEmbeddingGate(
        {"api": 3, "graph": 2, "manifest": 4},
        hidden_dim=8,
        detach_embeddings=detach_embeddings,
    )


def _model_batch() -> Batch:
    items = [
        Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor(
                [[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long
            ),
            y=torch.tensor(index),
            sensitive_mask=torch.zeros(4, dtype=torch.uint8),
        )
        for index in (0, 1)
    ]
    batch = Batch.from_data_list(items)
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
    batch.q_api = torch.ones(2, 1)
    batch.q_graph = torch.ones(2, 1)
    batch.q_manifest = torch.ones(2, 1)
    batch.q_align = torch.ones(2, 1)
    batch.api_alive = torch.ones(2, 1)
    batch.graph_alive = torch.ones(2, 1)
    batch.manifest_alive = torch.ones(2, 1)
    return batch


def test_dense_embedding_gate_starts_uniform_and_masks_dead_branches():
    gate = _gate()
    alive = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )

    weights, scores = gate(_embeddings(), alive)

    assert scores.shape == (3, 3)
    assert torch.allclose(weights[0], torch.full((3,), 1.0 / 3.0))
    assert torch.allclose(weights[1], torch.tensor([0.5, 0.0, 0.5]))
    assert torch.equal(weights[2], torch.zeros(3))
    assert weights[1, 1].item() == 0.0
    assert torch.allclose(weights[:2].sum(dim=-1), torch.ones(2))

    logits = dense_embedding_late_fusion_logits(
        {
            "api": torch.tensor([[4.0, -1.0], [8.0, -3.0], [20.0, -20.0]]),
            "graph": torch.tensor([[-3.0, 2.0], [-9.0, 7.0], [-30.0, 30.0]]),
            "manifest": torch.tensor([[1.0, 0.0], [2.0, 3.0], [40.0, -40.0]]),
        },
        weights,
    )
    assert torch.equal(logits[2], torch.zeros(2))
    assert torch.equal(torch.softmax(logits[2], dim=-1), torch.full((2,), 0.5))


def test_dense_embedding_gate_is_trainable_and_detach_switch_is_explicit():
    embeddings = _embeddings(requires_grad=True)
    gate = _gate(detach_embeddings=False)
    with torch.no_grad():
        gate.net[-1].weight.normal_(mean=0.0, std=0.1)
    weights, _ = gate(embeddings, torch.ones(3, 3))
    (-weights[:, 0].log().mean()).backward()

    assert gate.net[-1].weight.grad is not None
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in embeddings.values())

    detached_embeddings = _embeddings(requires_grad=True)
    detached_gate = _gate(detach_embeddings=True)
    with torch.no_grad():
        detached_gate.net[-1].weight.normal_(mean=0.0, std=0.1)
    detached_weights, _ = detached_gate(detached_embeddings, torch.ones(3, 3))
    (-detached_weights[:, 0].log().mean()).backward()

    assert detached_gate.net[-1].weight.grad is not None
    assert all(value.grad is None for value in detached_embeddings.values())


def test_dense_embedding_model_mode_fuses_only_alive_branches():
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
        gate_detach=False,
    ).eval()
    batch = _model_batch()
    logits, extra = model(batch)
    assert logits.shape == (2, 2)
    assert extra["gate_weights"].shape == (2, 3)
    assert torch.allclose(extra["gate_weights"].sum(dim=-1), torch.ones(2))

    batch.graph_alive.zero_()
    _, graph_dead = model(batch)
    assert torch.equal(graph_dead["gate_weights"][:, 1], torch.zeros(2))
    assert torch.allclose(
        graph_dead["gate_weights"].sum(dim=-1), torch.ones(2)
    )

    batch.api_alive.zero_()
    batch.manifest_alive.zero_()
    all_dead_logits, all_dead = model(batch)
    assert torch.equal(all_dead["gate_weights"], torch.zeros(2, 3))
    assert torch.equal(all_dead_logits, torch.zeros(2, 2))
    assert torch.equal(
        torch.softmax(all_dead_logits, dim=-1), torch.full((2, 2), 0.5)
    )

    # Eligibility follows the branches actually consumed by a baseline, not
    # unrelated modalities that happen to remain present in the APK payload.
    model.fusion_mode = "api_only"
    batch.api_alive.zero_()
    batch.graph_alive.fill_(1.0)
    batch.manifest_alive.fill_(1.0)
    api_only_logits, api_only = model(batch)
    assert torch.equal(api_only_logits, torch.zeros(2, 2))
    assert torch.equal(
        api_only["selective_eligible"], torch.zeros(2, dtype=torch.bool)
    )


def test_dense_embedding_gate_adapted_config_has_independent_identity():
    cfg = load_config_path(ROOT / "baselines/dense_embedding_gate_adapted.yaml")

    assert cfg["method"]["name"] == "dense_embedding_gate_adapted"
    assert cfg["method"]["protocol_id"] == "dense_embedding_gate_adapted_v1"
    assert cfg["model"]["fusion_mode"] == "tri_modal_dense_embedding_gate"
    assert cfg["model"]["gate"]["detach"] is False
    assert "use_consistency_evidence" not in cfg["model"]["gate"]
    assert "use_conflict_evidence" not in cfg["model"]["gate"]
    assert "use_perturbation_evidence" not in cfg["model"]["gate"]
    assert "apply_alive_mask" not in cfg["model"]["gate"]
    assert cfg["loss"]["auxiliary_weight_mode"] == "alive_masked_uniform"
    assert cfg["calibration"]["enabled"] is False
    assert cfg["calibration"]["holdout_enabled"] is True
    # Every main-table fusion method receives the same model-selection
    # macro-F1 threshold budget. I3 remains a separate decision layer.
    assert cfg["classification_threshold"]["enabled"] is True
    assert cfg["selective_prediction"]["enabled"] is False

    # I3 is a model-agnostic decision layer when it consumes only final class
    # probabilities.  The baseline must not be rejected merely because it does
    # not instantiate the proposed evidential fusion path.
    _validate_selective_score_fusion_compatibility(
        selective_enabled=True,
        score_type="msp",
        discount_probability_mode=False,
    )

    model = build_model(cfg, feature_dim=16)
    assert model.dense_embedding_gate is not None
    assert model.dense_embedding_gate.detach_embeddings is False

    pseudo_switch = copy.deepcopy(cfg)
    pseudo_switch["model"]["gate"]["apply_alive_mask"] = False
    with pytest.raises(ValueError, match="Removed model.gate input switches"):
        build_model(pseudo_switch, feature_dim=16)

    silent_evidence_switch = copy.deepcopy(cfg)
    silent_evidence_switch["model"]["gate"]["use_conflict_evidence"] = True
    with pytest.raises(ValueError, match="Removed model.gate input switches"):
        build_model(silent_evidence_switch, feature_dim=16)


def test_non_discount_models_reject_only_discount_specific_selective_scores():
    for score_type in (
        "msp",
        "deployed_class_probability",
        "predictive_entropy_certainty",
    ):
        _validate_selective_score_fusion_compatibility(
            selective_enabled=True,
            score_type=score_type,
            discount_probability_mode=False,
        )

    for score_type in (
        "evidential_certainty",
        "mixture_certainty",
        "model_acceptance",
    ):
        with pytest.raises(ValueError, match="requires discount_probability"):
            _validate_selective_score_fusion_compatibility(
                selective_enabled=True,
                score_type=score_type,
                discount_probability_mode=False,
            )


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_prior_only_uses_explicit_fixed_odds_beta_without_trainable_route(beta: float):
    router = GlobalOpinionRouter(mode="prior_only", fixed_prior_beta=beta)
    reliability = torch.tensor([0.90, 0.60, 0.20])
    branch_probabilities = {
        name: torch.tensor([[0.50, 0.50]])
        for name in ("api", "graph", "manifest")
    }
    reliabilities = {
        name: reliability[index : index + 1]
        for index, name in enumerate(("api", "graph", "manifest"))
    }
    alive = {
        name: torch.ones(1) for name in ("api", "graph", "manifest")
    }

    outputs = router(
        branch_probabilities,
        reliabilities,
        alive,
        learned_active=True,
    )
    expected = torch.softmax(beta * torch.logit(reliability), dim=-1).unsqueeze(0)

    assert router.route_parameters() == []
    assert outputs["route_prior_beta"].item() == pytest.approx(beta)
    assert outputs["prior_only_odds_beta_active"].item() == pytest.approx(1.0)
    assert torch.allclose(outputs["branch_distribution"], expected)


def test_fixed_prior_beta_is_restricted_to_the_prior_only_sensitivity():
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            GlobalOpinionRouter(mode="prior_only", fixed_prior_beta=invalid)
    with pytest.raises(ValueError, match="prior_only sensitivity"):
        GlobalOpinionRouter(mode="learned", fixed_prior_beta=0.5)


def test_retired_prior_beta_configs_are_absent_from_formal_catalog():
    retired = (
        ROOT / "appendix/prior_beta_0_5.yaml",
        ROOT / "ablations/i2/router_prior_only.yaml",
        ROOT / "appendix/prior_beta_2_0.yaml",
    )
    assert all(not path.exists() for path in retired)
