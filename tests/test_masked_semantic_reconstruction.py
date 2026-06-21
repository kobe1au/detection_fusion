from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from fusion.constants import EvidenceIndex
from fusion.losses import compute_masked_semantic_reconstruction_loss, compute_robust_loss
from fusion.model import TriModalRobustModel


def _evidence(batch_size: int = 2) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _loss_inputs(batch_size: int = 2):
    outputs = {}
    for name in ("api", "graph", "manifest"):
        outputs[f"recon_{name}_semantic_logits"] = torch.randn(batch_size, 12, requires_grad=True)
        outputs[f"mask_{name}_semantic"] = torch.ones(batch_size)
    batch = SimpleNamespace(
        api_semantic_category_counts=torch.randint(0, 2, (batch_size, 12)).float(),
        graph_semantic_category_counts=torch.randint(0, 2, (batch_size, 12)).float(),
        manifest_category_counts=torch.randint(0, 2, (batch_size, 12)).float(),
    )
    config = {
        "enabled": True,
        "loss": "bce",
        "min_target_integrity": 0.2,
        "min_input_integrity": 0.2,
        "detach_reliability": True,
    }
    return outputs, batch, config


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


def _model(mask_probs=(1.0, 1.0, 1.0)) -> TriModalRobustModel:
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
            "semantic_dim": 20,
            "mask_prob_api": mask_probs[0],
            "mask_prob_graph": mask_probs[1],
            "mask_prob_manifest": mask_probs[2],
        },
    )


def test_semantic_projectors_output_shape():
    _, outputs = _model()(_model_batch())
    assert outputs["z_api"].shape == (2, 20)
    assert outputs["z_graph"].shape == (2, 20)
    assert outputs["z_manifest"].shape == (2, 20)


def test_masked_reconstruction_logits_shape():
    _, outputs = _model()(_model_batch())
    assert outputs["recon_api_semantic_logits"].shape == (2, 12)
    assert outputs["recon_graph_semantic_logits"].shape == (2, 12)
    assert outputs["recon_manifest_semantic_logits"].shape == (2, 12)


def test_semantic_reconstruction_scales_each_source_by_its_integrity():
    model = _model(mask_probs=(0.0, 0.0, 0.0)).eval()
    batch = _model_batch()
    with torch.no_grad():
        _, baseline = model(batch)
        batch.graph_integrity[0] = 0.0
        _, outputs = model(batch)

    assert outputs["semantic_source_weight_graph"][0].item() == 0.0
    assert outputs["semantic_source_weight_graph"][1].item() == 1.0
    assert not torch.allclose(
        baseline["recon_api_semantic_logits"][0],
        outputs["recon_api_semantic_logits"][0],
    )


def test_masking_only_enabled_during_training():
    model = _model()
    model.train()
    _, train_outputs = model(_model_batch())
    assert torch.equal(train_outputs["mask_api_semantic"], torch.ones(2))
    model.eval()
    _, eval_outputs = model(_model_batch())
    assert torch.equal(eval_outputs["mask_api_semantic"], torch.zeros(2))


def test_masked_reconstruction_loss_finite():
    outputs, batch, config = _loss_inputs(3)
    outputs["mask_api_semantic"] = torch.tensor([1.0, 0.0, 0.0])
    outputs["mask_graph_semantic"] = torch.tensor([0.0, 1.0, 0.0])
    outputs["mask_manifest_semantic"] = torch.tensor([0.0, 0.0, 1.0])
    loss, diagnostics = compute_masked_semantic_reconstruction_loss(
        outputs, batch, _evidence(3), config
    )
    loss.backward()
    assert torch.isfinite(loss)
    expected = torch.stack(
        [diagnostics[f"loss_recon_{name}"] for name in ("api", "graph", "manifest")]
    ).mean()
    assert torch.allclose(loss.detach(), expected)
    assert diagnostics["active_recon_target_count"].item() == 3.0


def test_masked_reconstruction_averages_only_active_targets():
    outputs, batch, config = _loss_inputs(1)
    outputs["mask_graph_semantic"] = torch.zeros(1)
    outputs["mask_manifest_semantic"] = torch.zeros(1)

    loss, diagnostics = compute_masked_semantic_reconstruction_loss(
        outputs, batch, _evidence(1), config
    )

    assert torch.allclose(loss.detach(), diagnostics["loss_recon_api"])
    assert diagnostics["active_recon_target_count"].item() == 1.0


def test_masked_reconstruction_skips_low_integrity_target():
    outputs, batch, config = _loss_inputs(1)
    evidence = _evidence(1)
    evidence[:, EvidenceIndex.API_INTEGRITY] = 0.1
    _, diagnostics = compute_masked_semantic_reconstruction_loss(outputs, batch, evidence, config)
    assert diagnostics["loss_recon_api"].item() == 0.0
    assert diagnostics["valid_recon_api_rate"].item() == 0.0


def test_masked_reconstruction_uses_multilabel_targets():
    outputs, batch, config = _loss_inputs(1)
    outputs["mask_graph_semantic"] = torch.zeros(1)
    outputs["mask_manifest_semantic"] = torch.zeros(1)
    outputs["recon_api_semantic_logits"] = torch.tensor([[6.0] + [-6.0] * 11], requires_grad=True)
    batch.api_semantic_category_counts = torch.tensor([[2.0] + [0.0] * 11])
    _, matching = compute_masked_semantic_reconstruction_loss(outputs, batch, _evidence(1), config)
    batch.api_semantic_category_counts = torch.tensor([[0.0, 3.0] + [0.0] * 10])
    _, mismatching = compute_masked_semantic_reconstruction_loss(outputs, batch, _evidence(1), config)
    assert matching["loss_recon_api"].item() < mismatching["loss_recon_api"].item()


def test_no_nan_when_one_input_modality_missing():
    outputs, batch, config = _loss_inputs()
    evidence = _evidence()
    evidence[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    loss, _ = compute_masked_semantic_reconstruction_loss(outputs, batch, evidence, config)
    assert torch.isfinite(loss)


def test_main_discount_model_outputs_and_loss_backward():
    semantic_config = {
        "enabled": True,
        "loss": "bce",
        "weight": 0.02,
        "min_target_integrity": 0.2,
        "min_input_integrity": 0.2,
        "detach_reliability": True,
    }
    model = _model(mask_probs=(1.0, 0.0, 0.0))
    batch = _model_batch()
    logits, outputs = model(batch)
    for key in (
        "final_prob",
        "final_logits",
        "discounts",
        "fusion_weights",
        "entropy_api",
        "margin_api",
        "uncertainty_proxy_api",
        "recon_api_semantic_logits",
    ):
        assert key in outputs
    loss, parts = compute_robust_loss(
        logits,
        torch.tensor([0, 1]),
        outputs,
        {
            "branch_aux_weight": 0.05,
            "reliability_weighted_aux": True,
            "min_aux_weight": 0.2,
            "detach_reliability_for_aux": True,
        },
        batch=batch,
        evidence=outputs["gate_evidence"],
        semantic_reconstruction_cfg=semantic_config,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["masked_semantic_reconstruction"] > 0.0
