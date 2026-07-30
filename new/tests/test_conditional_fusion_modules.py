from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

import fusion.model as model_module
from fusion.model import TRI_MODAL_FUSION_MODES, TriModalRobustModel


def _batch() -> Batch:
    items = []
    for label in (0, 1):
        graph = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor(
                [[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long
            ),
            y=torch.tensor(label),
        )
        graph.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        items.append(graph)

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
    for name in ("api_alive", "graph_alive", "manifest_alive"):
        setattr(batch, name, torch.ones(2, 1))
    for name in (
        "api_integrity",
        "graph_integrity",
        "manifest_integrity",
        "code_integrity",
    ):
        setattr(batch, name, torch.ones(2, 1))
    return batch


def _model(fusion_mode: str) -> TriModalRobustModel:
    return TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode=fusion_mode,
        api_num_hash_buckets=64,
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
        max_nodes_gnn=64,
        manifest_in_dim=32,
        manifest_emb_dim=16,
        manifest_hidden_dim=32,
        discount_fusion_config=(
            {"combination": "dempster"}
            if fusion_mode == "discount_probability"
            else None
        ),
    )


@pytest.mark.parametrize("fusion_mode", sorted(TRI_MODAL_FUSION_MODES))
def test_every_formal_fusion_mode_owns_only_its_specialized_module_and_forwards(
    fusion_mode: str,
):
    model = _model(fusion_mode)
    state_keys = tuple(model.state_dict())

    needs_api_graph_concat = fusion_mode in model_module.API_GRAPH_CONCAT_FUSION_MODES
    needs_tri_concat = fusion_mode in model_module.TRI_MODAL_CONCAT_FUSION_MODES
    needs_dense_gate = fusion_mode == "tri_modal_dense_embedding_gate"
    needs_discount = fusion_mode == "discount_probability"

    assert (model.api_graph_concat_head is not None) is needs_api_graph_concat
    assert (model.tri_concat_head is not None) is needs_tri_concat
    assert (model.dense_embedding_gate is not None) is needs_dense_gate
    assert (model.discount_fusion is not None) is needs_discount
    assert any(key.startswith("api_graph_concat_head.") for key in state_keys) is needs_api_graph_concat
    assert any(key.startswith("tri_concat_head.") for key in state_keys) is needs_tri_concat
    assert any(key.startswith("dense_embedding_gate.") for key in state_keys) is needs_dense_gate
    # Fixed evidential fusion is intentionally parameter-free. Its presence is
    # represented by the module attribute, not by a synthetic state-dict key.
    assert not any(key.startswith("discount_fusion.") for key in state_keys)

    with torch.no_grad():
        logits, extra = model(_batch())
    assert logits.shape == (2, 2)
    assert extra["gate_weights"].shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert "api_emb_for_concat" not in extra
    assert "graph_emb_for_concat" not in extra
    assert "manifest_emb_for_concat" not in extra
    assert "api_graph_concat_input" not in extra
    assert "tri_modal_concat_input" not in extra


@pytest.mark.parametrize(
    "removed_alias",
    ("api", "graph", "manifest", "api_graph", "api_graph_manifest_concat"),
)
def test_removed_fusion_mode_aliases_are_rejected(removed_alias: str):
    with pytest.raises(ValueError, match="Unsupported tri-modal fusion_mode"):
        _model(removed_alias)


@pytest.mark.parametrize(
    ("handler", "attribute", "message", "tensors"),
    (
        (
            model_module._fusion_api_graph_concat,
            "api_graph_concat_head",
            "initialized API/Graph concat head",
            {},
        ),
        (
            model_module._fusion_tri_concat,
            "tri_concat_head",
            "initialized tri-modal concat head",
            {},
        ),
        (
            model_module._fusion_discount_probability,
            "discount_fusion",
            "initialized discount fusion module",
            {},
        ),
    ),
)
def test_specialized_handlers_fail_clearly_when_their_module_is_absent(
    handler,
    attribute: str,
    message: str,
    tensors: dict,
):
    model = _model("api_only")
    assert getattr(model, attribute) is None

    with pytest.raises(RuntimeError, match=message):
        handler(
            model,
            2,
            torch.device("cpu"),
            torch.float32,
            tensors,
            {},
        )


def test_concat_inputs_mask_unavailable_branch_embeddings():
    batch = _batch()
    batch.api_alive.zero_()
    api_graph_model = _model("api_graph_concat")
    tri_model = _model("tri_modal_concat")

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, args):
            captured[name] = args[0].detach().clone()

        return hook

    api_handle = api_graph_model.api_graph_concat_head.register_forward_pre_hook(
        capture("api_graph")
    )
    tri_handle = tri_model.tri_concat_head.register_forward_pre_hook(
        capture("tri")
    )
    try:
        with torch.no_grad():
            api_graph_model(batch)
            tri_model(batch)
    finally:
        api_handle.remove()
        tri_handle.remove()

    assert torch.equal(
        captured["api_graph"][:, :16], torch.zeros(2, 16)
    )
    assert torch.equal(captured["tri"][:, :16], torch.zeros(2, 16))
