from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fusion.constants import AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import compute_robust_loss
from fusion.opinion_router import GlobalOpinionRouter


def _availability(rows: int) -> torch.Tensor:
    return torch.ones(rows, AvailabilityIndex.BASE_DIM)


def _fusion_inputs() -> list[torch.Tensor]:
    values = (
        ((0.4, -0.2), (-0.3, 0.8), (0.1, -0.4)),
        ((0.2, -0.1), (-0.5, 0.6), (-0.2, 0.7)),
        ((0.5, -0.4), (-0.1, 0.2), (0.3, -0.6)),
    )
    return [torch.tensor(value, requires_grad=True) for value in values]


def _run_fusion(module: DiscountProbabilityFusion):
    inputs = _fusion_inputs()
    outputs = module(*inputs, _availability(3))
    loss = F.nll_loss(outputs["final_logits"], torch.tensor([0, 1, 0]))
    loss.backward()
    return outputs, [value.grad.detach().clone() for value in inputs]


@pytest.mark.parametrize("active", (False, True))
def test_calibration_active_state_dict_roundtrip_restores_shadow_and_gradients(
    active: bool,
):
    config = {
        "combination": "dempster",
        "reliability_calibration": {"enabled": True},
    }
    source = DiscountProbabilityFusion(config)
    restored = DiscountProbabilityFusion(config)
    if active:
        source.set_calibration_active(True)
    else:
        # Exercise restoration in both directions: loading an inactive source
        # must clear a destination whose Python shadow was already active.
        restored.set_calibration_active(True)

    restored.load_state_dict(source.state_dict(), strict=True)

    assert restored.calibration_active is active
    assert bool(restored._calibration_active) is active
    source_outputs, source_gradients = _run_fusion(source)
    restored_outputs, restored_gradients = _run_fusion(restored)
    assert source_outputs["final_is_log_probability"] is True
    assert restored_outputs["final_is_log_probability"] is True
    torch.testing.assert_close(
        restored_outputs["final_logits"], source_outputs["final_logits"]
    )
    torch.testing.assert_close(
        restored_outputs["calibration_active"],
        source_outputs["calibration_active"],
    )
    for restored_gradient, source_gradient in zip(
        restored_gradients, source_gradients
    ):
        torch.testing.assert_close(restored_gradient, source_gradient)


def _router_inputs():
    malware_probability = torch.tensor((0.2, 0.45, 0.8))
    probability = torch.stack(
        (1.0 - malware_probability, malware_probability), dim=-1
    )
    return (
        {
            name: probability.clone().detach().requires_grad_(True)
            for name in ("api", "graph", "manifest")
        },
        {
            name: torch.full_like(malware_probability, 0.8)
            for name in ("api", "graph", "manifest")
        },
        {
            name: torch.ones_like(malware_probability)
            for name in ("api", "graph", "manifest")
        },
    )


def _run_router(module: GlobalOpinionRouter):
    probabilities, reliabilities, alive = _router_inputs()
    outputs = module(
        probabilities,
        reliabilities,
        alive,
        learned_active=True,
    )
    loss = outputs["risk_probability"].sum() + 0.1 * outputs[
        "mixture_probability"
    ][:, 1].sum()
    loss.backward()
    input_gradients = {
        name: value.grad.detach().clone() for name, value in probabilities.items()
    }
    parameter_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }
    return outputs, input_gradients, parameter_gradients


@pytest.mark.parametrize("active", (False, True))
def test_risk_threshold_state_dict_roundtrip_restores_shadow_and_gradients(
    active: bool,
):
    kwargs = {
        "risk_mode": "learned",
        "risk_target": "threshold_malware_false_negative",
    }
    source = GlobalOpinionRouter(**kwargs)
    restored = GlobalOpinionRouter(**kwargs)
    if active:
        source.set_risk_decision_threshold(0.25)
    else:
        restored.set_risk_decision_threshold(0.9)

    restored.load_state_dict(source.state_dict(), strict=True)

    assert restored.risk_decision_threshold_active is active
    assert bool(restored._risk_decision_threshold_active) is active
    source_outputs, source_input_gradients, source_parameter_gradients = (
        _run_router(source)
    )
    restored_outputs, restored_input_gradients, restored_parameter_gradients = (
        _run_router(restored)
    )
    for key in (
        "risk_probability",
        "risk_decision_boundary_proximity",
        "risk_decision_log_odds_threshold",
        "risk_decision_threshold_active",
        "mixture_probability",
    ):
        torch.testing.assert_close(restored_outputs[key], source_outputs[key])
    for name in source_input_gradients:
        torch.testing.assert_close(
            restored_input_gradients[name], source_input_gradients[name]
        )
    for name, source_gradient in source_parameter_gradients.items():
        restored_gradient = restored_parameter_gradients[name]
        if source_gradient is None:
            assert restored_gradient is None
        else:
            torch.testing.assert_close(restored_gradient, source_gradient)


def test_python_log_probability_flag_preserves_nll_value_and_gradient():
    labels = torch.tensor((0, 1, 1))
    raw = torch.tensor(
        ((0.4, -0.3), (-0.2, 0.7), (0.1, 0.8)), dtype=torch.float32
    )
    actual_logits = F.log_softmax(raw, dim=-1).detach().requires_grad_(True)
    reference_logits = actual_logits.detach().clone().requires_grad_(True)

    actual, _ = compute_robust_loss(
        actual_logits,
        labels,
        {"final_is_log_probability": True},
        {"branch_aux_weight": 0.0},
        materialize_diagnostics=False,
    )
    reference = F.nll_loss(reference_logits, labels)
    actual.backward()
    reference.backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_logits.grad, reference_logits.grad)


def test_tensor_log_probability_flag_is_rejected_before_hot_path_sync():
    logits = F.log_softmax(torch.tensor(((0.2, -0.1),)), dim=-1)
    with pytest.raises(TypeError, match="must be a Python bool"):
        compute_robust_loss(
            logits,
            torch.tensor((0,)),
            {"final_is_log_probability": torch.tensor(True)},
            {"branch_aux_weight": 0.0},
            materialize_diagnostics=False,
        )
