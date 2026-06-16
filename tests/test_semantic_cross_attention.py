from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from fusion.constants import EvidenceIndex, GateConstants
from fusion.model import TriModalRobustModel
from fusion.semantic_cross_attention import ReliabilityAwareSemanticCrossAttention


def _evidence(batch_size: int = 4) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _module(**overrides) -> ReliabilityAwareSemanticCrossAttention:
    config = {
        "dim": 128,
        "num_heads": 4,
        "num_security_tokens": 12,
        "num_residual_tokens": 4,
        "dropout": 0.0,
        "residual_gate_init": -3.0,
    }
    config.update(overrides)
    module = ReliabilityAwareSemanticCrossAttention(**config)
    module.eval()
    return module


def _embeddings(batch_size: int = 4, dim: int = 128):
    return tuple(torch.randn(batch_size, dim) for _ in range(3))


def test_semantic_cross_attention_shapes():
    output = _module()(*_embeddings(), _evidence())
    assert output["enhanced_tokens"].shape == (4, 3, 16, 128)
    assert output["semantic_attention"].shape == (4, 4, 48, 48)
    assert output["enhanced_joint"].shape == (4, 128)
    assert output["semantic_reliability_prior"].shape == (4, 3, 16)
    assert output["semantic_relation_applicable"].shape == (4, 3, 3)


def test_semantic_cross_attention_does_not_require_semantic_presence_or_pt_fields():
    output = _module()(*_embeddings(), _evidence(), semantic_presence=None)
    assert torch.isfinite(output["enhanced_joint"]).all()
    assert torch.isfinite(output["semantic_attention"]).all()


def test_unavailable_graph_never_propagates_as_attention_source():
    evidence = _evidence()
    evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    output = _module()(*_embeddings(), evidence)
    graph_source = slice(16, 32)
    assert output["semantic_attention"][..., graph_source].abs().max().item() < 1.0e-7


def test_unobserved_manifest_code_relation_blocks_cross_modal_propagation():
    evidence = _evidence()
    evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0
    output = _module()(*_embeddings(), evidence)
    attention = output["semantic_attention"]
    code_tokens = slice(0, 32)
    manifest_tokens = slice(32, 48)
    assert attention[:, :, code_tokens, manifest_tokens].abs().max().item() < 1.0e-7
    assert attention[:, :, manifest_tokens, code_tokens].abs().max().item() < 1.0e-7


def test_manifest_code_applicability_is_pair_specific():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.0
    output = _module()(*_embeddings(), evidence)
    applicable = output["semantic_relation_applicable"]
    support = output["semantic_support_matrix"]
    conflict = output["semantic_conflict_matrix"]

    assert not applicable[:, 0, 2].any()
    assert not applicable[:, 2, 0].any()
    assert applicable[:, 1, 2].all()
    assert applicable[:, 2, 1].all()
    assert torch.equal(support[:, 0, 2], torch.zeros(4))
    assert torch.equal(conflict[:, 0, 2], torch.zeros(4))
    assert output["api_manifest_relation_applicable"].sum().item() == 0
    assert output["graph_manifest_relation_applicable"].sum().item() == 4
    assert output["manifest_code_relation_applicable"].sum().item() == 4


def test_semantic_presence_modulates_security_tokens_but_not_residual_tokens():
    presence = {
        name: torch.ones(4, 12)
        for name in ("api", "graph", "manifest")
    }
    presence["api"][:, 0] = 0.0
    output = _module()(*_embeddings(), _evidence(), semantic_presence=presence)
    prior = output["semantic_reliability_prior"]
    assert torch.all(prior[:, 0, 0] < prior[:, 0, 1])
    assert torch.allclose(prior[:, 0, 12:], prior[:, 0, 12:13].expand(-1, 4))


def test_attention_diagnostics_average_only_alive_target_queries():
    evidence = _evidence()
    evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    output = _module(use_relation_mask=False)(*_embeddings(), evidence)
    attention = output["semantic_attention"]
    alive_targets = torch.cat([attention[:, :, :16], attention[:, :, 32:]], dim=2)
    expected_to_api = alive_targets[..., :16].mean(dim=(1, 2, 3))

    cross_mask = (
        ~torch.eye(3, dtype=torch.bool)
    ).repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)
    alive_target_mask = torch.cat(
        [torch.ones(16, dtype=torch.bool), torch.zeros(16, dtype=torch.bool), torch.ones(16, dtype=torch.bool)]
    )
    expected_cross = attention[
        :, :, alive_target_mask
    ][..., cross_mask[alive_target_mask]].mean(dim=(1, 2))
    entropy = -(
        attention.clamp_min(GateConstants.EPS)
        * torch.log(attention.clamp_min(GateConstants.EPS))
    ).sum(dim=-1)
    expected_entropy = entropy[:, :, alive_target_mask].mean(dim=(1, 2))

    assert torch.allclose(
        output["mean_semantic_attention_to_api"], expected_to_api, atol=1.0e-7
    )
    assert torch.allclose(
        output["mean_cross_modal_attention"], expected_cross, atol=1.0e-7
    )
    assert torch.allclose(
        output["mean_semantic_attention_entropy"], expected_entropy, atol=1.0e-7
    )


def test_all_unavailable_modalities_produce_zero_joint_and_diagnostics():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    evidence[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    output = _module()(*_embeddings(), evidence)

    assert torch.equal(output["enhanced_joint"], torch.zeros(4, 128))
    assert torch.equal(output["mean_cross_modal_attention"], torch.zeros(4))
    assert torch.equal(output["mean_semantic_attention_entropy"], torch.zeros(4))
    for name in ("api", "graph", "manifest"):
        assert torch.equal(output[f"mean_semantic_attention_to_{name}"], torch.zeros(4))


def test_model_masks_cross_joint_projection_when_all_modalities_unavailable():
    model = TriModalRobustModel(
        in_feat_dim=8,
        fusion_mode="discount_probability",
        api_num_hash_buckets=32,
        api_type_vocab_size=16,
        api_emb_dim=16,
        api_hidden_dim=32,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=16,
        graph_hidden=16,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=32,
        manifest_in_dim=16,
        manifest_emb_dim=16,
        manifest_hidden_dim=32,
        joint_emb_dim=32,
        semantic_cross_attention_config={
            "enabled": True,
            "dim": 16,
            "num_heads": 4,
            "num_security_tokens": 12,
            "num_residual_tokens": 4,
            "dropout": 0.0,
            "attach_to_joint": True,
            "attach_to_reconstruction": False,
        },
    )
    batch = _model_batch()
    for name in (
        "api_alive",
        "graph_alive",
        "manifest_alive",
        "api_integrity",
        "graph_integrity",
        "manifest_integrity",
    ):
        setattr(batch, name, torch.zeros(2, 1))

    _, output = model(batch, return_features=True)

    assert torch.equal(output["joint_emb"], torch.zeros(2, 32))


def test_high_manifest_code_conflict_reduces_cross_modal_attention():
    module = _module()
    embeddings = _embeddings()
    low = _evidence()
    high = low.clone()
    high[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 1.0
    high[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 1.0

    low_attention = module(*embeddings, low)["semantic_attention"]
    high_attention = module(*embeddings, high)["semantic_attention"]
    code_tokens = slice(0, 32)
    manifest_tokens = slice(32, 48)
    low_cross = (
        low_attention[:, :, code_tokens, manifest_tokens].mean()
        + low_attention[:, :, manifest_tokens, code_tokens].mean()
    )
    high_cross = (
        high_attention[:, :, code_tokens, manifest_tokens].mean()
        + high_attention[:, :, manifest_tokens, code_tokens].mean()
    )
    assert high_cross.item() < low_cross.item()


def test_initial_residual_gate_keeps_enhanced_tokens_close_to_base_tokens():
    output = _module(residual_gate_init=-3.0)(*_embeddings(), _evidence())
    mean_delta = (
        output["enhanced_tokens"] - output["base_semantic_tokens"]
    ).abs().mean()
    assert mean_delta.item() < 0.1
    assert output["semantic_residual_gate"][0].item() < 0.1


def test_leave_one_out_reconstruction_sources_do_not_leak_target_modality():
    module = _module()
    api, graph, manifest = _embeddings()
    first = module(api, graph, manifest, _evidence())
    second = module(api + 100.0, graph, manifest, _evidence())
    excluding_api_first = first["enhanced_semantic_excluding_api"]
    excluding_api_second = second["enhanced_semantic_excluding_api"]
    assert torch.allclose(
        excluding_api_first[:, 1:],
        excluding_api_second[:, 1:],
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def _model_batch() -> Batch:
    batch = Batch.from_data_list(
        [
            Data(
                x=torch.randn(3, 8),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                sensitive_mask=torch.zeros(3, dtype=torch.uint8),
            )
            for _ in range(2)
        ]
    )
    batch.api_ids = torch.randint(1, 16, (8,))
    batch.api_type_ids = torch.zeros(8, dtype=torch.long)
    batch.api_sensitive_mask = torch.zeros(8)
    batch.api_batch = torch.tensor([0] * 4 + [1] * 4)
    batch.manifest_x = torch.rand(2, 16)
    batch.api_semantic_category_counts = torch.ones(2, 12)
    batch.graph_semantic_category_counts = torch.ones(2, 12)
    batch.manifest_category_counts = torch.ones(2, 12)
    for name in (
        "api_integrity",
        "graph_integrity",
        "manifest_integrity",
        "code_integrity",
        "api_graph_anchor_support",
        "manifest_code_support",
        "api_alive",
        "graph_alive",
        "manifest_alive",
    ):
        setattr(batch, name, torch.ones(2, 1))
    batch.manifest_to_code_conflict = torch.zeros(2, 1)
    batch.code_to_manifest_conflict = torch.zeros(2, 1)
    return batch


def _model(cross_attention_enabled: bool) -> TriModalRobustModel:
    return TriModalRobustModel(
        in_feat_dim=8,
        fusion_mode="discount_probability",
        api_num_hash_buckets=32,
        api_type_vocab_size=16,
        api_emb_dim=16,
        api_hidden_dim=32,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=16,
        graph_hidden=16,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=32,
        manifest_in_dim=16,
        manifest_emb_dim=16,
        manifest_hidden_dim=32,
        joint_emb_dim=16,
        semantic_reconstruction_config={
            "enabled": True,
            "semantic_dim": 16,
            "mask_prob_api": 0.0,
            "mask_prob_graph": 0.0,
            "mask_prob_manifest": 0.0,
        },
        semantic_cross_attention_config={
            "enabled": cross_attention_enabled,
            "dim": 16,
            "num_heads": 4,
            "num_security_tokens": 12,
            "num_residual_tokens": 4,
            "dropout": 0.0,
            "attach_to_joint": True,
            "attach_to_reconstruction": True,
        },
    )


def test_model_disabled_uses_previous_joint_path_and_discount_fusion():
    logits, output = _model(False)(_model_batch())
    assert logits.shape == (2, 2)
    assert "final_prob" in output
    assert "semantic_attention" not in output
    assert torch.equal(output["semantic_cross_attention_enabled"], torch.zeros(2))


def test_model_enabled_enhances_joint_and_reconstruction_sources():
    logits, output = _model(True)(_model_batch())
    assert logits.shape == (2, 2)
    assert output["semantic_attention"].shape == (2, 4, 48, 48)
    assert output["enhanced_joint"].shape == (2, 16)
    assert output["recon_api_semantic_logits"].shape == (2, 12)
    assert torch.equal(output["semantic_cross_attention_enabled"], torch.ones(2))


def test_cross_attention_changes_joint_without_changing_single_modality_branches():
    model = _model(True).eval()
    batch = _model_batch()
    with torch.no_grad():
        _, first = model(batch)
        joint_projection = model.semantic_cross_attention.joint_projection[1]
        joint_projection.weight.zero_()
        joint_projection.bias.zero_()
        _, second = model(batch)

    for name in ("api", "graph", "manifest"):
        assert torch.allclose(
            first[f"{name}_logits_aux"],
            second[f"{name}_logits_aux"],
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    assert not torch.allclose(first["joint_logits_aux"], second["joint_logits_aux"])
