from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import fusion.anchored_training as anchored_training
from fusion.anchored_training import (
    CachedExpertBatch,
    fit_anchored_router,
    fit_competence_heads,
)
from fusion.competence_fusion import (
    ATOMIC_EXPERT_NAMES,
    EXPERT_NAMES,
    AnchoredCompetenceFusion,
    ContentConditionedCompetence,
    atomic_pairwise_ranking_loss,
    competence_learning_loss,
    true_class_probability_targets,
)


def _alive(
    api: list[bool],
    graph: list[bool],
    manifest: list[bool],
    joint: list[bool],
) -> dict[str, torch.Tensor]:
    return {
        "api": torch.tensor(api),
        "graph": torch.tensor(graph),
        "manifest": torch.tensor(manifest),
        "joint": torch.tensor(joint),
    }


def _probabilities(
    rows: dict[str, list[list[float]]],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(rows[name], dtype=torch.float32)
        for name in EXPERT_NAMES
    }


def test_competence_heads_are_expert_local_detached_and_hard_masked() -> None:
    torch.manual_seed(7)
    dimensions = {"api": 5, "graph": 4, "manifest": 3, "joint": 6}
    estimator = ContentConditionedCompetence(
        dimensions,
        projection_dim=4,
        hidden_dim=5,
        dropout=0.0,
    )
    embeddings = {
        name: torch.randn(3, width, requires_grad=True)
        for name, width in dimensions.items()
    }
    logits = {
        name: torch.randn(3, 2, requires_grad=True) for name in EXPERT_NAMES
    }
    alive = _alive(
        [True, False, True],
        [True, True, True],
        [False, False, True],
        [True, True, True],
    )

    output = estimator(embeddings, logits, alive)
    assert tuple(output.competence) == EXPERT_NAMES
    for name in EXPERT_NAMES:
        assert output.competence[name].shape == (3,)
        assert torch.all(output.unmasked_competence[name] > 0.0)
        assert torch.all(output.unmasked_competence[name] < 1.0)
        assert torch.equal(
            output.competence[name][~alive[name]],
            torch.zeros_like(output.competence[name][~alive[name]]),
        )

    sum(value.sum() for value in output.competence.values()).backward()
    assert all(value.grad is None for value in embeddings.values())
    assert all(value.grad is None for value in logits.values())
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool((parameter.grad != 0).any())
        for parameter in estimator.parameters()
    )


def test_competence_is_invariant_to_per_row_additive_logit_shift() -> None:
    torch.manual_seed(11)
    dimensions = {"api": 5, "graph": 4, "manifest": 3, "joint": 6}
    estimator = ContentConditionedCompetence(
        dimensions,
        projection_dim=4,
        hidden_dim=5,
        dropout=0.0,
    ).eval()
    embeddings = {
        name: torch.randn(4, width) for name, width in dimensions.items()
    }
    logits = {name: torch.randn(4, 2) for name in EXPERT_NAMES}
    shifted = {
        name: value + torch.randn(4, 1) * 20.0
        for name, value in logits.items()
    }
    alive = {name: torch.ones(4, dtype=torch.bool) for name in EXPERT_NAMES}

    original = estimator(embeddings, logits, alive)
    offset = estimator(embeddings, shifted, alive)

    for name in EXPERT_NAMES:
        assert torch.allclose(
            original.unmasked_competence[name],
            offset.unmasked_competence[name],
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        assert torch.allclose(
            original.competence_logits[name],
            offset.competence_logits[name],
            atol=2.0e-6,
            rtol=2.0e-6,
        )


def test_tcp_targets_are_continuous_true_class_probabilities() -> None:
    probability = _probabilities(
        {
            "api": [[0.8, 0.2], [0.1, 0.9]],
            "graph": [[0.6, 0.4], [0.3, 0.7]],
            "manifest": [[0.55, 0.45], [0.8, 0.2]],
            "joint": [[0.9, 0.1], [0.2, 0.8]],
        }
    )
    labels = torch.tensor([0, 1])
    targets = true_class_probability_targets(probability, labels)

    assert torch.allclose(targets["api"], torch.tensor([0.8, 0.9]))
    assert torch.allclose(targets["graph"], torch.tensor([0.6, 0.7]))
    assert torch.allclose(targets["manifest"], torch.tensor([0.55, 0.2]))
    assert torch.allclose(targets["joint"], torch.tensor([0.9, 0.8]))
    assert all(not target.requires_grad for target in targets.values())


def test_competence_loss_supports_mse_soft_bce_and_atomic_ranking() -> None:
    targets = {
        "api": torch.tensor([0.9, 0.2]),
        "graph": torch.tensor([0.6, 0.5]),
        "manifest": torch.tensor([0.3, 0.8]),
        "joint": torch.tensor([0.8, 0.7]),
    }
    ordered = {name: value.clone().requires_grad_() for name, value in targets.items()}
    reversed_atomic = {
        "api": targets["manifest"].clone(),
        "graph": targets["graph"].clone(),
        "manifest": targets["api"].clone(),
        "joint": targets["joint"].clone(),
    }
    alive = {name: torch.ones(2, dtype=torch.bool) for name in EXPERT_NAMES}

    ordered_rank, ordered_count = atomic_pairwise_ranking_loss(
        ordered,
        targets,
        alive,
    )
    reversed_rank, reversed_count = atomic_pairwise_ranking_loss(
        reversed_atomic,
        targets,
        alive,
    )
    assert int(ordered_count) == 6
    assert int(reversed_count) == 6
    assert ordered_rank < reversed_rank

    mse = competence_learning_loss(
        ordered,
        targets,
        alive,
        regression="mse",
        ranking_weight=0.1,
    )
    soft_bce = competence_learning_loss(
        ordered,
        targets,
        alive,
        regression="bce",
        ranking_weight=0.0,
    )
    assert mse.regression.item() == pytest.approx(0.0)
    assert mse.total.item() > 0.0
    assert soft_bce.regression.item() > 0.0
    assert int(mse.valid_expert_rows) == 8
    mse.total.backward()
    assert ordered["api"].grad is not None


def test_competence_loss_never_backpropagates_into_tcp_targets() -> None:
    competence = {
        name: torch.full((2,), 0.5, requires_grad=True)
        for name in EXPERT_NAMES
    }
    targets = {
        name: torch.tensor([0.8, 0.2], requires_grad=True)
        for name in EXPERT_NAMES
    }
    alive = {name: torch.ones(2, dtype=torch.bool) for name in EXPERT_NAMES}

    output = competence_learning_loss(
        competence,
        targets,
        alive,
        ranking_weight=0.1,
    )
    output.total.backward()

    assert all(value.grad is not None for value in competence.values())
    assert all(value.grad is None for value in targets.values())


def test_atomic_weights_and_joint_late_gate_are_monotone_in_competence() -> None:
    fusion = AnchoredCompetenceFusion(
        initial_atomic_competence_scale=1.0,
        initial_joint_late_scale=2.0,
        initial_late_gate=0.5,
    )
    probability = _probabilities(
        {
            "api": [[0.1, 0.9]],
            "graph": [[0.8, 0.2]],
            "manifest": [[0.7, 0.3]],
            "joint": [[0.75, 0.25]],
        }
    )
    alive = {name: torch.ones(1, dtype=torch.bool) for name in EXPERT_NAMES}
    high_api = {
        "api": torch.tensor([0.9]),
        "graph": torch.tensor([0.2]),
        "manifest": torch.tensor([0.1]),
        "joint": torch.tensor([0.8]),
    }
    low_api = {
        "api": torch.tensor([0.1]),
        "graph": torch.tensor([0.2]),
        "manifest": torch.tensor([0.1]),
        "joint": torch.tensor([0.8]),
    }
    output_high = fusion(probability, high_api, alive)
    output_low = fusion(probability, low_api, alive)
    low_joint = {name: value.clone() for name, value in high_api.items()}
    low_joint["joint"] = torch.tensor([0.2])
    output_low_joint = fusion(probability, low_joint, alive)

    assert output_high.atomic_weights[0, 0] > output_high.atomic_weights[0, 1]
    assert output_high.atomic_weights[0, 1] > output_high.atomic_weights[0, 2]
    assert output_high.atomic_weights[0, 0] > output_low.atomic_weights[0, 0]
    assert output_high.late_competence > output_low.late_competence
    assert output_high.late_gate > output_low.late_gate
    assert output_low_joint.late_gate > output_high.late_gate
    assert fusion.effective_atomic_competence_scale().item() > 0.0
    assert fusion.effective_joint_late_scale().item() > 0.0
    assert torch.allclose(output_high.probability.sum(dim=-1), torch.ones(1))


def test_availability_fallbacks_and_all_dead_output_are_explicit() -> None:
    fusion = AnchoredCompetenceFusion(initial_late_gate=0.5)
    probability = _probabilities(
        {
            "api": [[0.1, 0.9], [0.2, 0.8], [0.1, 0.9]],
            "graph": [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8]],
            "manifest": [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]],
            "joint": [[0.75, 0.25], [0.4, 0.6], [0.9, 0.1]],
        }
    )
    competence = {
        "api": torch.tensor([0.9, 0.8, 0.7]),
        "graph": torch.tensor([0.2, 0.7, 0.6]),
        "manifest": torch.tensor([0.1, 0.6, 0.5]),
        "joint": torch.tensor([0.8, 0.9, 0.4]),
    }
    # Row 0: joint only. Row 1: atomic only. Row 2: all dead.
    alive = _alive(
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [True, False, False],
    )
    output = fusion(probability, competence, alive)

    assert torch.allclose(output.probability[0], probability["joint"][0])
    assert output.late_gate[0].item() == 0.0
    assert torch.allclose(output.probability[1], output.late_probability[1])
    assert output.late_gate[1].item() == 1.0
    assert torch.allclose(output.probability[2], torch.tensor([0.5, 0.5]))
    expected_uniform_atomic = torch.stack(
        [
            probability["api"][1],
            probability["graph"][1],
            probability["manifest"][1],
        ]
    ).mean(dim=0)
    assert torch.allclose(
        output.uniform_atomic_probability[1],
        expected_uniform_atomic,
    )
    assert torch.allclose(output.joint_probability[1], expected_uniform_atomic)
    assert torch.allclose(output.joint_probability[2], torch.tensor([0.5, 0.5]))
    assert output.joint_competence[1].item() == 0.0
    assert output.joint_competence[2].item() == 0.0
    assert torch.equal(output.atomic_competence[0], torch.zeros(3))
    assert torch.equal(output.atomic_weights[2], torch.zeros(3))
    assert output.all_dead.tolist() == [False, False, True]
    assert bool(torch.isfinite(output.probability).all())
    assert bool(torch.isfinite(output.atomic_weights).all())


def test_i2_default_detachment_prevents_expert_or_i1_pollution() -> None:
    fusion = AnchoredCompetenceFusion()
    logits = {
        name: torch.randn(2, 2, requires_grad=True) for name in EXPERT_NAMES
    }
    probability = {name: torch.softmax(value, dim=-1) for name, value in logits.items()}
    competence = {
        name: torch.full((2,), 0.6, requires_grad=True) for name in EXPERT_NAMES
    }
    alive = {name: torch.ones(2, dtype=torch.bool) for name in EXPERT_NAMES}

    output = fusion(probability, competence, alive)
    loss = -output.probability[:, 1].clamp_min(1.0e-6).log().mean()
    loss.backward()

    assert all(value.grad is None for value in logits.values())
    assert all(value.grad is None for value in competence.values())
    assert fusion.atomic_relative_bias.grad is not None
    assert fusion.raw_atomic_competence_scale.grad is not None
    assert fusion.raw_joint_late_scale.grad is not None
    assert fusion.joint_late_bias.grad is not None


def test_rejects_nonbinary_availability_and_incomplete_expert_mappings() -> None:
    estimator = ContentConditionedCompetence(
        {"api": 2, "graph": 2, "manifest": 2, "joint": 2},
        dropout=0.0,
    )
    embeddings = {name: torch.randn(1, 2) for name in EXPERT_NAMES}
    logits = {name: torch.randn(1, 2) for name in EXPERT_NAMES}
    bad_alive = {name: torch.ones(1) for name in EXPERT_NAMES}
    bad_alive["api"] = torch.tensor([0.25])
    with pytest.raises(ValueError, match="must be boolean"):
        estimator(embeddings, logits, bad_alive)

    incomplete = dict(embeddings)
    incomplete.pop("joint")
    with pytest.raises(ValueError, match="exactly"):
        estimator(
            incomplete,
            logits,
            {name: torch.ones(1, dtype=torch.bool) for name in EXPERT_NAMES},
        )


def test_rejects_nonfinite_or_unnormalized_stage_b_inputs() -> None:
    estimator = ContentConditionedCompetence(
        {"api": 2, "graph": 2, "manifest": 2, "joint": 2},
        dropout=0.0,
    )
    embeddings = {name: torch.randn(1, 2) for name in EXPERT_NAMES}
    logits = {name: torch.randn(1, 2) for name in EXPERT_NAMES}
    alive = {name: torch.ones(1, dtype=torch.bool) for name in EXPERT_NAMES}
    bad_logits = dict(logits)
    bad_logits["api"] = torch.tensor([[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="finite"):
        estimator(embeddings, bad_logits, alive)

    probability = {
        name: torch.tensor([[0.5, 0.5]]) for name in EXPERT_NAMES
    }
    bad_probability = dict(probability)
    bad_probability["graph"] = torch.tensor([[0.8, 0.8]])
    labels = torch.tensor([0])
    with pytest.raises(ValueError, match="normalized probabilities"):
        true_class_probability_targets(bad_probability, labels)

    competence = {
        name: torch.tensor([0.5]) for name in EXPERT_NAMES
    }
    competence["manifest"] = torch.tensor([float("inf")])
    router = AnchoredCompetenceFusion()
    with pytest.raises(ValueError, match="finite"):
        router(probability, competence, alive)


class _TinyStageBModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dimensions = {name: 2 for name in EXPERT_NAMES}
        self.expert_probe = nn.Linear(2, 2)
        self.competence_estimator = ContentConditionedCompetence(
            dimensions,
            projection_dim=3,
            hidden_dim=3,
            dropout=0.0,
        )
        self.anchored_fusion = AnchoredCompetenceFusion(
            initial_late_gate=0.2,
        )
        self.anchored_fusion_active = False

    def set_anchored_fusion_active(self, active: bool) -> None:
        self.anchored_fusion_active = bool(active)


def _cached_batch() -> CachedExpertBatch:
    labels = torch.tensor([0, 1, 0, 1])
    logits = {
        "api": torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]]),
        "graph": torch.tensor([[1.5, 0.0], [0.0, 1.5], [0.8, 0.0], [0.0, 0.8]]),
        "manifest": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.4, 0.0], [0.0, 0.4]]),
        "joint": torch.tensor([[2.5, 0.0], [0.0, 2.5], [1.2, 0.0], [0.0, 1.2]]),
    }
    return CachedExpertBatch(
        labels=labels,
        embeddings={
            name: torch.tensor(
                [[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]]
            )
            for name in EXPERT_NAMES
        },
        logits=logits,
        alive={name: torch.ones(4, dtype=torch.bool) for name in EXPERT_NAMES},
    )


def test_competence_fit_does_not_update_router_or_expert_parameters() -> None:
    torch.manual_seed(19)
    model = _TinyStageBModel()
    cache = _cached_batch()
    router_before = {
        name: value.detach().clone()
        for name, value in model.anchored_fusion.state_dict().items()
    }
    expert_before = {
        name: value.detach().clone()
        for name, value in model.expert_probe.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    summary = fit_competence_heads(
        model,
        train_clean=[cache],
        train_degraded=[cache],
        validation_sources={"clean": [cache]},
        clean_validation_source="clean",
        device=torch.device("cpu"),
        config={
            "epochs": 1,
            "patience": 1,
            "lr": 1.0e-3,
            "regression": "mse",
            "degraded_loss_weight": 0.25,
            "ranking_weight": 0.1,
        },
    )

    assert summary["best_epoch"] == 1
    assert summary["early_stopping_population"] == (
        "val_model_selection_clean_only"
    )
    for name, value in model.anchored_fusion.state_dict().items():
        assert torch.equal(value, router_before[name])
    for name, value in model.expert_probe.state_dict().items():
        assert torch.equal(value, expert_before[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.competence_estimator.parameters()
    )


def test_router_grid_uses_fresh_optimizers_and_enforces_clean_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    model = _TinyStageBModel()
    cache = _cached_batch()
    fit_competence_heads(
        model,
        train_clean=[cache],
        train_degraded=[cache],
        validation_sources={"clean": [cache]},
        clean_validation_source="clean",
        device=torch.device("cpu"),
        config={"epochs": 1, "patience": 1},
    )

    calls = {"evaluation": 0}

    def fake_evaluation(_model, _sources, _device, **_kwargs):
        calls["evaluation"] += 1
        # candidate 0: initial, trained; candidate 1: initial, trained.
        scenarios = (
            (0.900, 0.900, 0.50),
            (0.899, 0.900, 0.60),
            (0.900, 0.900, 0.50),
            (0.890, 0.900, 0.99),
        )
        clean_f1, joint_f1, robust_f1 = scenarios[calls["evaluation"] - 1]
        return {
            "sources": {
                "clean": {
                    "macro_f1": clean_f1,
                    "joint_anchor_macro_f1": joint_f1,
                    "nll": 0.20,
                },
                "degraded": {
                    "macro_f1": robust_f1,
                    "joint_anchor_macro_f1": 0.50,
                    "nll": 0.40,
                },
            },
            "classification_threshold": {
                "threshold": 0.40 + 0.01 * calls["evaluation"],
            },
            "joint_anchor_classification_threshold": {"threshold": 0.45},
        }

    optimizer_instances = []
    real_adamw = torch.optim.AdamW

    class _SpyAdamW(real_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_state_size = len(self.state)
            self.step_calls = 0
            optimizer_instances.append(self)

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    monkeypatch.setattr(
        anchored_training,
        "evaluate_cached_fusion",
        fake_evaluation,
    )
    monkeypatch.setattr(torch.optim, "AdamW", _SpyAdamW)

    summary = fit_anchored_router(
        model,
        train_clean=[cache],
        train_degraded=[cache],
        validation_sources={"clean": [cache], "degraded": [cache]},
        clean_validation_source="clean",
        device=torch.device("cpu"),
        config={
            "epochs": 1,
            "patience": 1,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "degradation_loss_weights": [0.0, 0.5],
            "clean_noninferiority_tolerance": 0.003,
            "degraded_source_noninferiority_tolerance": 0.0,
        },
    )

    assert summary["deployment"] == "anchored_joint_late"
    assert summary["selected"]["degradation_loss_weight"] == 0.0
    assert summary["selected"]["robust_mean_macro_f1"] == pytest.approx(0.60)
    assert summary["classification_threshold"]["threshold"] == pytest.approx(
        0.42
    )
    assert summary["classification_threshold"]["locked_by_stage_b"] is True
    assert summary["classification_threshold"]["prediction_source"] == (
        "anchored_joint_late"
    )
    assert len(optimizer_instances) == 2
    assert all(item.initial_state_size == 0 for item in optimizer_instances)
    # Weight zero completely skips the degraded update; each candidate gets
    # exactly one paired optimizer step, not two sequential AdamW steps.
    assert [item.step_calls for item in optimizer_instances] == [1, 1]
    assert all(
        parameter.grad is None
        for parameter in model.competence_estimator.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.anchored_fusion.parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_router_selection_prefers_any_clean_safe_candidate() -> None:
    safe = anchored_training._router_selection_tuple(
        {
            "sources": {
                "clean": {
                    "macro_f1": 0.898,
                    "joint_anchor_macro_f1": 0.900,
                    "nll": 0.2,
                },
                "degraded": {
                    "macro_f1": 0.55,
                    "joint_anchor_macro_f1": 0.50,
                    "nll": 0.4,
                },
            }
        },
        clean_source="clean",
        clean_noninferiority_tolerance=0.003,
    )
    unsafe = anchored_training._router_selection_tuple(
        {
            "sources": {
                "clean": {
                    "macro_f1": 0.890,
                    "joint_anchor_macro_f1": 0.900,
                    "nll": 0.2,
                },
                "degraded": {
                    "macro_f1": 0.99,
                    "joint_anchor_macro_f1": 0.50,
                    "nll": 0.4,
                },
            }
        },
        clean_source="clean",
        clean_noninferiority_tolerance=0.003,
    )
    clean_safe_without_robust_gain = anchored_training._router_selection_tuple(
        {
            "sources": {
                "clean": {
                    "macro_f1": 0.900,
                    "joint_anchor_macro_f1": 0.900,
                    "nll": 0.2,
                },
                "degraded": {
                    "macro_f1": 0.50,
                    "joint_anchor_macro_f1": 0.50,
                    "nll": 0.4,
                },
            }
        },
        clean_source="clean",
        clean_noninferiority_tolerance=0.003,
    )
    assert safe[0] is True
    assert unsafe[0] is False
    assert clean_safe_without_robust_gain[0] is False
    assert safe > unsafe


def test_router_guard_rejects_mean_gain_when_one_degraded_source_is_harmed():
    state = anchored_training._router_selection_state(
        {
            "sources": {
                "clean": {
                    "macro_f1": 0.90,
                    "joint_anchor_macro_f1": 0.90,
                    "nll": 0.20,
                },
                "api_event_dropout": {
                    "macro_f1": 0.70,
                    "joint_anchor_macro_f1": 0.60,
                    "nll": 0.30,
                },
                "graph_sparsify": {
                    "macro_f1": 0.70,
                    "joint_anchor_macro_f1": 0.60,
                    "nll": 0.30,
                },
                "manifest_permission_mask": {
                    "macro_f1": 0.59,
                    "joint_anchor_macro_f1": 0.60,
                    "nll": 0.30,
                },
            }
        },
        clean_source="clean",
        clean_noninferiority_tolerance=0.0,
        degraded_source_noninferiority_tolerance=0.0,
        minimum_robust_gain=0.0,
    )

    assert state["robust_mean_gain"] > 0.0
    assert state["minimum_degraded_source_delta"] == pytest.approx(-0.01)
    assert state["every_degraded_source_noninferior"] is False
    assert state["eligible"] is False
