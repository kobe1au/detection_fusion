from __future__ import annotations

import logging
import warnings
from contextlib import nullcontext

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
