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
    """Map an unconstrained scalar to a bounded positive temperature.

    raw_temperature == 0 maps exactly to T == 1.  The mapping is smooth
    and its deployed range is [FINAL_TEMPERATURE_MIN,
    FINAL_TEMPERATURE_MAX] up to floating-point saturation.
    """

    if raw_temperature.numel() != 1:
        raise ValueError(
            "bounded_final_temperature expects exactly one scalar"
        )

    log_temperature = (
        _LOG_TEMPERATURE_CENTER
        + _LOG_TEMPERATURE_RADIUS * torch.tanh(raw_temperature)
    )
    return torch.exp(log_temperature)