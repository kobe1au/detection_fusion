from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fusion.losses import routing_risk_per_sample_loss, routing_risk_target
from fusion.opinion_router import GlobalOpinionRouter
from fusion.train import (
    compute_branch_reliability_metrics,
    fit_oof_malware_classification_threshold,
    validate_threshold_aligned_risk_cutoff,
)


BRANCHES = ("api", "graph", "manifest")


def _risk_only_config(
    target: str,
    *,
    raw_cutoff: float,
) -> dict:
    return {
        "routing": {
            "enabled": True,
            "posthoc_refine": True,
            "prediction_loss_weight": 0.0,
            "risk_loss_weight": 1.0,
            "risk_loss": "bce",
            "risk_target": target,
            "classification_log_odds_threshold": raw_cutoff,
        },
    }


def test_threshold_fn_loss_masks_malware_predictions_and_has_finite_gradients():
    # Scores at indices 0 and 2 are on the predicted-benign side. Only those
    # two samples may supervise P(malware | fixed classifier predicts benign).
    raw_scores = torch.tensor([-1.0, 1.0, -2.0, 2.0])
    raw_log_prob = F.log_softmax(
        torch.stack([torch.zeros_like(raw_scores), raw_scores], dim=-1),
        dim=-1,
    )
    labels = torch.tensor([1, 0, 0, 1])
    risk_training_logit = torch.tensor(
        [0.2, -0.3, 0.4, -0.5], requires_grad=True
    )
    outputs = {
        "routing_active": torch.ones(4),
        "routing_has_available": torch.ones(4),
        "routing_mixture_prob": raw_log_prob.exp(),
        "uncalibrated_final_log_prob": raw_log_prob,
        "routing_risk_probability": torch.sigmoid(risk_training_logit),
        "routing_risk_training_logit": risk_training_logit,
    }

    config = _risk_only_config(
        "threshold_malware_false_negative",
        raw_cutoff=0.0,
    )["routing"]
    target, valid, loss_type, target_type = routing_risk_target(
        outputs,
        labels,
        config,
    )
    per_row = routing_risk_per_sample_loss(
        outputs["routing_risk_probability"],
        risk_training_logit,
        target,
        valid,
        loss_type=loss_type,
    )
    loss = per_row[valid].mean()
    expected = F.binary_cross_entropy_with_logits(
        risk_training_logit[[0, 2]],
        torch.tensor([1.0, 0.0]),
    )

    assert torch.allclose(loss, expected)
    assert target_type == "threshold_malware_false_negative"
    assert int(valid.sum()) == 2
    assert target[valid].mean().item() == pytest.approx(0.5)

    loss.backward()
    assert risk_training_logit.grad is not None
    assert torch.isfinite(risk_training_logit.grad).all()
    assert torch.all(risk_training_logit.grad[[0, 2]] != 0.0)
    assert torch.equal(
        risk_training_logit.grad[[1, 3]],
        torch.zeros(2),
    )


def _router_inputs(
    malware_probabilities: tuple[float, ...],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    malware = torch.tensor(malware_probabilities, dtype=torch.float32)
    probability = torch.stack([1.0 - malware, malware], dim=-1)
    return (
        {name: probability.clone() for name in BRANCHES},
        {name: torch.full_like(malware, 0.8) for name in BRANCHES},
        {name: torch.ones_like(malware) for name in BRANCHES},
    )


def test_threshold_fn_router_gates_malware_side_risk_to_zero():
    router = GlobalOpinionRouter(
        risk_mode="learned",
        risk_target="threshold_malware_false_negative",
    )
    router.set_risk_decision_threshold(0.0)

    outputs = router(
        *_router_inputs((0.8, 0.2)),
        learned_active=True,
    )

    assert torch.equal(
        outputs["risk_predicted_malware"],
        torch.tensor([1.0, 0.0]),
    )
    assert outputs["risk_probability"][0].item() == pytest.approx(0.0)
    assert torch.isneginf(outputs["risk_logit"][0])
    assert outputs["risk_probability"][1].item() > 0.0
    assert torch.isfinite(outputs["risk_training_logit"]).all()


def _risk_metric_row(
    *,
    label: int,
    pred: int,
    mixture_pred: int,
    risk: float,
    target: str,
) -> dict:
    return {
        "routing_active": 1,
        "routing_has_available": 1,
        "routing_risk_probability": risk,
        "routing_mixture_pred": mixture_pred,
        "routing_risk_decision_threshold_active": 1,
        "pred": pred,
        "label": label,
        f"routing_risk_target_{target}": 1,
    }


@pytest.mark.parametrize(
    ("target", "rows"),
    [
        (
            "threshold_malware_false_negative",
            [
                # The same false positive is not a malware-FN event.
                _risk_metric_row(
                    label=0,
                    pred=1,
                    mixture_pred=1,
                    risk=0.2,
                    target="threshold_malware_false_negative",
                ),
                _risk_metric_row(
                    label=1,
                    pred=0,
                    mixture_pred=1,
                    risk=0.8,
                    target="threshold_malware_false_negative",
                ),
            ],
        ),
    ],
)
def test_threshold_risk_metrics_use_the_declared_event_semantics(target, rows):
    metrics = compute_branch_reliability_metrics(rows)

    assert metrics["routing_risk_target"] == target
    assert metrics["routing_risk_threshold_aligned_count"] == 2
    assert metrics["routing_risk_target_event_rate"] == pytest.approx(0.5)
    assert metrics["routing_risk_brier"] == pytest.approx(0.04)
    assert metrics["routing_risk_auc"] == pytest.approx(1.0)


def _threshold_aligned_cfg(*, classification_enabled: bool) -> dict:
    return {
        "fusion": {
            "combination": "routed",
            "routing": {
                "enabled": True,
                "risk_mode": "learned",
                "risk_target": "threshold_malware_false_negative",
            },
        },
        "classification_threshold": {"enabled": classification_enabled},
        "selective_prediction": {"enabled": True},
    }


def test_classification_off_requires_protocol_neutral_zero_risk_cutoff():
    router = GlobalOpinionRouter(
        risk_mode="learned",
        risk_target="threshold_malware_false_negative",
    )
    model = SimpleNamespace(
        discount_fusion=SimpleNamespace(opinion_router=router)
    )
    cfg = _threshold_aligned_cfg(classification_enabled=False)

    router.set_risk_decision_threshold(0.0)
    validate_threshold_aligned_risk_cutoff(model, cfg, None)

    router.set_risk_decision_threshold(0.25)
    with pytest.raises(RuntimeError, match="differs from the deployed classifier cutoff"):
        validate_threshold_aligned_risk_cutoff(model, cfg, None)


def test_selective_off_ignores_unused_threshold_aligned_risk_head():
    router = GlobalOpinionRouter(
        risk_mode="learned",
        risk_target="threshold_malware_false_negative",
    )
    router.set_risk_decision_threshold(0.25)
    model = SimpleNamespace(
        discount_fusion=SimpleNamespace(opinion_router=router)
    )
    cfg = _threshold_aligned_cfg(classification_enabled=False)
    cfg["selective_prediction"] = {"enabled": False}

    # With no acceptance/rejection decision, u is diagnostic-only and cannot
    # affect forced classification.  Requiring a costly post-hoc refit merely
    # to change this unused head would confound the genuine off/off cell.
    validate_threshold_aligned_risk_cutoff(model, cfg, None)


def _normalized_binary_log_prob(raw_log_odds: float) -> list[float]:
    log_p0 = -math.log1p(math.exp(float(raw_log_odds)))
    return [log_p0, log_p0 + float(raw_log_odds)]


def test_oof_cutoff_preserves_partition_for_adjacent_fp32_scores():
    lower = np.float32(1.0)
    upper = np.nextafter(lower, np.float32(np.inf))
    assert np.nextafter(lower, np.float32(np.inf)) == upper
    rows = [
        {
            "sid": "lower",
            "group": "lower",
            "label": 0,
            "raw_log_prob": _normalized_binary_log_prob(float(lower)),
        },
        {
            "sid": "upper",
            "group": "upper",
            "label": 1,
            "raw_log_prob": _normalized_binary_log_prob(float(upper)),
        },
    ]

    summary = fit_oof_malware_classification_threshold(
        rows,
        {
            "enabled": True,
            "objective": "macro_f1",
            "selection_rule": "macro_f1_unconstrained_v1",
        },
        deployment_temperature=3.0,
    )

    assert summary is not None
    cutoff = np.float32(summary["raw_log_odds_threshold"])
    assert cutoff == upper
    deploy_predictions = torch.tensor(
        [lower, upper], dtype=torch.float32
    ) >= float(cutoff)
    assert torch.equal(deploy_predictions, torch.tensor([False, True]))
    assert summary["macro_f1"] == pytest.approx(1.0)
