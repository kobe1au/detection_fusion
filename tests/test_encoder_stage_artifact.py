from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from fusion.model import TriModalRobustModel
from fusion.train import (
    CHECKPOINT_STAGE_ENCODER_SELECTED,
    ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION,
    POSTHOC_ONLY_COMPATIBLE_ENCODER_IMPLEMENTATION_TRANSITIONS,
    _canonical_mapping_sha256,
    _encoder_stage_implementation_sha256,
    _encoder_stage_semantic_signature,
    _state_dict_sha256,
    validate_encoder_stage_checkpoint,
)


def _small_model(fusion_mode: str = "discount_probability") -> TriModalRobustModel:
    discount_config = None
    if fusion_mode == "discount_probability":
        discount_config = {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "mode": "learned",
                "posthoc_refine": True,
                "prediction_loss_weight": 1.0,
                "risk_mode": "learned",
                "risk_target": "threshold_malware_false_negative",
                "risk_loss_weight": 1.0,
                "final_temperature_scaling": True,
            },
            "reliability_calibration": {
                "enabled": True,
                "method": "monotonic_correctness",
            },
        }
    return TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode=fusion_mode,
        api_num_hash_buckets=32,
        api_type_vocab_size=8,
        api_emb_dim=8,
        api_hidden_dim=8,
        api_layers=1,
        api_heads=1,
        api_max_seq_len=16,
        graph_emb_dim=8,
        graph_hidden=8,
        graph_heads=1,
        graph_layers=1,
        max_nodes_gnn=32,
        manifest_in_dim=16,
        manifest_emb_dim=8,
        manifest_hidden_dim=8,
        gate_hidden_dim=8,
        discount_fusion_config=discount_config,
    )


def _cloned_state(model: TriModalRobustModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def test_encoder_stage_state_roundtrip_preserves_posthoc_state():
    torch.manual_seed(7)
    source = _small_model()
    encoder_state = source.encoder_stage_state_dict()

    torch.manual_seed(19)
    target = _small_model()
    posthoc_keys = [
        key for key in target.state_dict() if key.startswith("discount_fusion.")
    ]
    assert posthoc_keys
    posthoc_before = {
        key: target.state_dict()[key].detach().clone() for key in posthoc_keys
    }
    assert any(
        not torch.equal(target.state_dict()[key], value)
        for key, value in encoder_state.items()
        if value.is_floating_point()
    )

    target.load_encoder_stage_state_dict(encoder_state)

    assert tuple(encoder_state) == target.encoder_stage_state_keys()
    for key, value in encoder_state.items():
        assert torch.equal(target.state_dict()[key], value)
    for key, value in posthoc_before.items():
        assert torch.equal(target.state_dict()[key], value)


def test_encoder_artifact_contract_validates_identity_and_tensor_hash(tmp_path):
    model = _small_model()
    state = model.encoder_stage_state_dict()
    cfg = {
        "encoder_stage": {"protocol_id": "unit-stage-v1"},
        "train": {"seed": 7, "epochs": 1, "batch_size": 2},
        "model": {"unit": True},
        "fusion": {
            "mode": "discount_probability",
            "combination": "routed",
            "routing": {
                "enabled": True,
            },
        },
        "loss": {},
        "robust": {},
        "data": {},
    }
    roles = {
        "role_assignment_sha256": "a" * 64,
        "validation_csv_sha256": "b" * 64,
        "num_selection": 4,
    }
    manifest = {
        "manifest_vocab_sha256": "c" * 64,
        "train_csv_sha256": "d" * 64,
        "train_sample_ids_sha256": "e" * 64,
        "num_train_samples": 8,
    }
    identity = _encoder_stage_semantic_signature(cfg, roles, manifest)
    artifact = {
        "checkpoint_stage": CHECKPOINT_STAGE_ENCODER_SELECTED,
        "encoder_stage_artifact_schema_version": (
            ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION
        ),
        "encoder_stage_state": state,
        "encoder_stage_state_sha256": _state_dict_sha256(state),
        "encoder_stage_identity": identity,
        "encoder_stage_identity_sha256": _canonical_mapping_sha256(identity),
        "encoder_stage_implementation_sha256": (
            _encoder_stage_implementation_sha256()
        ),
    }

    validated = validate_encoder_stage_checkpoint(
        artifact,
        current_cfg=cfg,
        validation_split=roles,
        manifest_vocab_provenance=manifest,
        checkpoint_path=tmp_path / "encoder.pt",
    )
    assert validated is state

    # The redesigned architecture intentionally supports no old encoder
    # implementation fingerprints. A compatibility allow-list would silently
    # reintroduce the best.pt lifecycle the current protocol removed.
    assert POSTHOC_ONLY_COMPATIBLE_ENCODER_IMPLEMENTATION_TRANSITIONS == frozenset()

    unknown_implementation = dict(artifact)
    unknown_implementation["encoder_stage_implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different Stage-1 implementation"):
        validate_encoder_stage_checkpoint(
            unknown_implementation,
            current_cfg=cfg,
            validation_split=roles,
            manifest_vocab_provenance=manifest,
            checkpoint_path=tmp_path / "unknown_encoder.pt",
        )

    tampered = dict(artifact)
    tampered_state = OrderedDict(
        (key, value.detach().clone()) for key, value in state.items()
    )
    first_float_key = next(
        key for key, value in tampered_state.items() if value.is_floating_point()
    )
    tampered_state[first_float_key].view(-1)[0].add_(1.0)
    tampered["encoder_stage_state"] = tampered_state
    with pytest.raises(ValueError, match="state hash mismatch"):
        validate_encoder_stage_checkpoint(
            tampered,
            current_cfg=cfg,
            validation_split=roles,
            manifest_vocab_provenance=manifest,
            checkpoint_path=tmp_path / "encoder.pt",
        )


def test_encoder_implementation_fingerprint_has_no_legacy_transition(
    tmp_path,
    monkeypatch,
):
    model = _small_model()
    state = model.encoder_stage_state_dict()
    cfg = {
        "encoder_stage": {"protocol_id": "unit-stage-v1"},
        "train": {"seed": 7, "epochs": 1, "batch_size": 2},
        "model": {"unit": True},
        "fusion": {
            "mode": "discount_probability",
            "combination": "routed",
            "routing": {"enabled": True},
        },
        "loss": {},
        "robust": {},
        "data": {},
    }
    roles = {
        "role_assignment_sha256": "a" * 64,
        "validation_csv_sha256": "b" * 64,
        "num_selection": 4,
    }
    manifest = {
        "manifest_vocab_sha256": "c" * 64,
        "train_csv_sha256": "d" * 64,
        "train_sample_ids_sha256": "e" * 64,
        "num_train_samples": 8,
    }
    identity = _encoder_stage_semantic_signature(cfg, roles, manifest)
    current_implementation = _encoder_stage_implementation_sha256()
    artifact = {
        "checkpoint_stage": CHECKPOINT_STAGE_ENCODER_SELECTED,
        "encoder_stage_artifact_schema_version": (
            ENCODER_STAGE_ARTIFACT_SCHEMA_VERSION
        ),
        "encoder_stage_state": state,
        "encoder_stage_state_sha256": _state_dict_sha256(state),
        "encoder_stage_identity": identity,
        "encoder_stage_identity_sha256": _canonical_mapping_sha256(identity),
        "encoder_stage_implementation_sha256": current_implementation,
    }

    future_implementation = "f" * 64
    assert future_implementation != current_implementation
    monkeypatch.setattr(
        "fusion.train._encoder_stage_implementation_sha256",
        lambda: future_implementation,
    )
    with pytest.raises(ValueError, match="different Stage-1 implementation"):
        validate_encoder_stage_checkpoint(
            artifact,
            current_cfg=cfg,
            validation_split=roles,
            manifest_vocab_provenance=manifest,
            checkpoint_path=tmp_path / "future_encoder.pt",
        )


def test_encoder_stage_state_excludes_i1_i2_and_temperature():
    model = _small_model()
    full_keys = tuple(model.state_dict())
    stage_keys = model.encoder_stage_state_keys()
    stage_state = model.encoder_stage_state_dict()

    assert any(key.startswith("discount_fusion.") for key in full_keys)
    assert not any(key.startswith("discount_fusion.") for key in stage_keys)
    assert not any("reliability_calibrator" in key for key in stage_keys)
    assert not any("opinion_router" in key for key in stage_keys)
    assert not any("temperature" in key for key in stage_keys)
    assert tuple(stage_state) == stage_keys
    for prefix in (
        "api_encoder.",
        "graph_encoder.",
        "manifest_encoder.",
        "api_head.",
        "graph_head.",
        "manifest_head.",
    ):
        assert any(key.startswith(prefix) for key in stage_keys)


@pytest.mark.parametrize(
    ("fusion_mode", "specialized_prefix"),
    [
        ("api_only", None),
        ("api_graph_concat", "api_graph_concat_head."),
        ("tri_modal_concat", "tri_concat_head."),
        ("tri_modal_dense_embedding_gate", "dense_embedding_gate."),
        ("discount_probability", None),
    ],
)
def test_encoder_stage_state_includes_only_the_active_stage1_fusion_module(
    fusion_mode: str,
    specialized_prefix: str | None,
):
    model = _small_model(fusion_mode)
    keys = model.encoder_stage_state_keys()

    optional_prefixes = {
        "api_graph_concat_head.",
        "tri_concat_head.",
        "dense_embedding_gate.",
    }
    observed = {
        prefix for prefix in optional_prefixes if any(key.startswith(prefix) for key in keys)
    }
    assert observed == ({specialized_prefix} if specialized_prefix else set())
    assert not any(key.startswith("discount_fusion.") for key in keys)


def test_encoder_stage_load_rejects_missing_and_unexpected_keys():
    model = _small_model()
    state = model.encoder_stage_state_dict()
    missing_key = next(iter(state))

    missing = OrderedDict(state)
    missing.pop(missing_key)
    with pytest.raises(ValueError, match="missing="):
        model.load_encoder_stage_state_dict(missing)

    unexpected = OrderedDict(state)
    unexpected["discount_fusion.forbidden"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected="):
        model.load_encoder_stage_state_dict(unexpected)


def test_encoder_stage_load_rejects_shape_mismatch_before_mutation():
    model = _small_model()
    original = _cloned_state(model)
    invalid = model.encoder_stage_state_dict()
    first_key = next(iter(invalid))
    invalid[first_key] = torch.zeros(
        (*invalid[first_key].shape, 1),
        dtype=invalid[first_key].dtype,
        device=invalid[first_key].device,
    )

    with pytest.raises(ValueError, match="tensor shapes disagree"):
        model.load_encoder_stage_state_dict(invalid)

    for key, value in original.items():
        assert torch.equal(model.state_dict()[key], value)


def test_encoder_stage_load_rejects_non_tensor_values():
    model = _small_model()
    invalid = model.encoder_stage_state_dict()
    first_key = next(iter(invalid))
    invalid[first_key] = "not-a-tensor"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="values must be tensors"):
        model.load_encoder_stage_state_dict(invalid)
