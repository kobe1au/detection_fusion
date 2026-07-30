from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from fusion.care_model import CareDroidModel


def _small_model() -> CareDroidModel:
    return CareDroidModel(
        in_feat_dim=16,
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
    )


def test_care_model_owns_no_comparison_fusion_modules() -> None:
    model = _small_model()
    for name in (
        "api_head",
        "graph_head",
        "manifest_head",
        "api_graph_concat_head",
        "tri_concat_head",
        "dense_embedding_gate",
        "discount_fusion",
    ):
        assert not hasattr(model, name)
    assert model.fusion_mode == "care_droid"
    assert set(model.care_path_heads.heads) == {"agm", "ag", "am", "gm"}


def test_stage_a_artifact_is_strict_and_excludes_risk_head() -> None:
    model = _small_model()
    stage_state = model.care_stage_a_state_dict()
    assert stage_state
    assert not any(key.startswith("care_risk_head.") for key in stage_state)
    risk_before = {
        key: value.detach().clone()
        for key, value in model.care_risk_head.state_dict().items()
    }

    first_parameter = next(model.api_encoder.parameters())
    with torch.no_grad():
        first_parameter.add_(1.0)
    model.load_care_stage_a_state_dict(stage_state)

    for key, value in risk_before.items():
        assert torch.equal(model.care_risk_head.state_dict()[key], value)
    with pytest.raises(ValueError, match="unexpected"):
        model.load_care_stage_a_state_dict(
            {**stage_state, "care_risk_head.invalid": torch.zeros(1)}
        )


def test_risk_routing_cannot_start_before_normalization_fit() -> None:
    model = _small_model()
    with pytest.raises(RuntimeError, match="before log-odds normalization"):
        model.set_care_risk_active(True)
    model.care_risk_head.set_log_odds_normalization(
        torch.zeros(4),
        torch.ones(4),
    )
    model.set_care_risk_active(True)
    assert model.care_risk_active


def test_importing_care_runner_does_not_load_comparison_stack() -> None:
    banned = (
        "fusion.model",
        "fusion.losses",
        "fusion.evidential",
        "fusion.discount_fusion",
        "fusion.gates",
    )
    script = (
        "import json, sys; "
        "import fusion.care_train; "
        f"print(json.dumps([name for name in {banned!r} if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []
