from __future__ import annotations

import logging
import math
import numbers
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation

import torch

logger = logging.getLogger(__name__)


# ── shared numeric helpers ────────────────────────────────────────────

def scalar_float(value, default: float = 0.0) -> float:
    """Extract a Python float from a tensor / scalar / number, with a safe fallback."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().view(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clamp_strength(strength: float) -> float:
    return max(0.0, min(1.0, float(strength)))


def strict_finite_integer(value, *, field_name: str) -> int:
    """Parse an integer without silently truncating floats or booleans."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{field_name} must be a finite integer, got boolean {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Rational):
        if value.denominator == 1:
            return int(value)
        raise ValueError(f"{field_name} must be a finite integer, got {value!r}")
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(f"{field_name} must be a finite integer, got {value!r}")
    try:
        numeric = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite integer, got {value!r}") from exc
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError(f"{field_name} must be a finite integer, got {value!r}")
    return int(numeric)


def strict_binary_integer(value, *, field_name: str = "label") -> int:
    """Parse the project's binary label contract without lossy coercion."""
    parsed = strict_finite_integer(value, field_name=field_name)
    if parsed not in {0, 1}:
        raise ValueError(f"{field_name} must be binary (0 or 1), got {value!r}")
    return parsed


# ── AMP helpers ───────────────────────────────────────────────────────

def get_amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def build_grad_scaler(device: torch.device, enabled: bool):
    use_scaler = bool(enabled) and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, RuntimeError):
        try:
            return torch.cuda.amp.GradScaler(enabled=use_scaler)
        except Exception:
            if use_scaler:
                # User explicitly asked for AMP but we cannot deliver it —
                # warn loudly and re-raise so the run stops instead of
                # silently training in FP32.
                msg = (
                    "AMP was requested (train.use_amp=true) but GradScaler "
                    "creation failed on this PyTorch build.  Set use_amp=false "
                    "or fix the CUDA / PyTorch installation."
                )
                logger.error(msg)
                raise RuntimeError(msg) from None
            logger.warning("GradScaler creation failed, AMP already disabled — continuing in FP32.")
            return torch.cuda.amp.GradScaler(enabled=False)
