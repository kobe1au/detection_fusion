from __future__ import annotations

import math

import torch


FINAL_TEMPERATURE_MIN = 1.0e-2
FINAL_TEMPERATURE_MAX = 1.0e2
FINAL_TEMPERATURE_IDENTITY = 1.0

_LOG_TEMPERATURE_CENTER = 0.5 * (
    math.log(FINAL_TEMPERATURE_MIN)
    + math.log(FINAL_TEMPERATURE_MAX)
)
_LOG_TEMPERATURE_RADIUS = 0.5 * (
    math.log(FINAL_TEMPERATURE_MAX)
    - math.log(FINAL_TEMPERATURE_MIN)
)

if not math.isclose(
    math.sqrt(FINAL_TEMPERATURE_MIN * FINAL_TEMPERATURE_MAX),
    FINAL_TEMPERATURE_IDENTITY,
    rel_tol=0.0,
    abs_tol=1.0e-12,
):
    raise RuntimeError(
        "Final-temperature bounds must have geometric mean equal to 1.0 "
        "so the zero raw coordinate remains the identity temperature"
    )


def bounded_final_temperature(
    raw_temperature: torch.Tensor,
) -> torch.Tensor:
    if raw_temperature.numel() != 1:
        raise ValueError(
            "bounded_final_temperature expects exactly one scalar"
        )

    log_temperature = (
        _LOG_TEMPERATURE_CENTER
        + _LOG_TEMPERATURE_RADIUS
        * torch.tanh(raw_temperature)
    )

    return torch.exp(log_temperature).clamp(
        min=FINAL_TEMPERATURE_MIN,
        max=FINAL_TEMPERATURE_MAX,
    )

def raw_final_temperature_coordinate(temperature: float) -> float:
    """Convert a deployable temperature to its unconstrained coordinate."""

    value = float(temperature)
    if not math.isfinite(value):
        raise ValueError("Final temperature must be finite")

    # tanh 的精确端点对应无穷大坐标，因此 override 使用开区间。
    if not FINAL_TEMPERATURE_MIN < value < FINAL_TEMPERATURE_MAX:
        raise ValueError(
            "Final-temperature override must lie strictly within "
            f"({FINAL_TEMPERATURE_MIN}, {FINAL_TEMPERATURE_MAX})"
        )

    normalized = (
        math.log(value) - _LOG_TEMPERATURE_CENTER
    ) / _LOG_TEMPERATURE_RADIUS

    if not -1.0 < normalized < 1.0:
        raise ValueError("Final-temperature inverse coordinate is invalid")

    return math.atanh(normalized)