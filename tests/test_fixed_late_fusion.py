from types import SimpleNamespace

import torch

import fusion.model as model_module
from fusion.constants import EvidenceIndex


def test_fixed_late_fusion_uses_three_source_branches_without_joint(monkeypatch):
    batch_size = 2
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    monkeypatch.setattr(
        model_module,
        "build_evidence",
        lambda *args, **kwargs: (evidence, {}),
    )
    model = SimpleNamespace(
        fusion_mode="tri_modal_fixed_gate",
        use_consistency_evidence=False,
        use_conflict_evidence=False,
        use_perturbation_evidence=False,
    )
    tensors = {
        "graph_data": object(),
        "api_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        "graph_logits": torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
        "manifest_logits": torch.tensor([[1.5, 1.5], [1.5, 1.5]]),
        "joint_logits": torch.full((batch_size, 2), 100.0),
        "api_emb": torch.zeros(batch_size, 2),
        "graph_emb": torch.zeros(batch_size, 2),
        "manifest_emb": torch.zeros(batch_size, 2),
    }

    logits, weights, _ = model_module._fusion_evidence_based(
        model,
        batch_size,
        torch.device("cpu"),
        torch.float32,
        tensors,
        {},
    )

    expected_weights = torch.tensor(
        [[1 / 3, 1 / 3, 1 / 3, 0.0]]
    ).repeat(batch_size, 1)
    assert torch.allclose(weights, expected_weights)
    expected_logits = (
        tensors["api_logits"]
        + tensors["graph_logits"]
        + tensors["manifest_logits"]
    ) / 3.0
    assert torch.allclose(logits, expected_logits)
