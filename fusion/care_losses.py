from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from fusion.care_fusion import PATH_NAMES


CARE_STAGE_A_OBJECTIVE = "care_stage_a_clean"


def compute_care_stage_a_loss(
    path_logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    path_available: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Train AGM/AG/AM/GM with one equally weighted clean CE per path."""

    if (
        not isinstance(path_logits, Mapping)
        or set(path_logits) != set(PATH_NAMES)
    ):
        raise ValueError(
            "CARE Stage A requires path logits for AGM, AG, AM, and GM"
        )
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("CARE Stage A labels must have shape [B]")
    label_smoothing = float(label_smoothing)
    if (
        not math.isfinite(label_smoothing)
        or not 0.0 <= label_smoothing < 1.0
    ):
        raise ValueError("label_smoothing must lie within [0, 1)")

    if (
        not isinstance(path_available, torch.Tensor)
        or path_available.shape != (labels.numel(), len(PATH_NAMES))
    ):
        raise ValueError(
            "CARE Stage A requires care_path_available with shape [B, 4]"
        )
    available_valid = (
        (path_available == 0) | (path_available == 1)
    ).all()
    if path_available.device.type == "cpu":
        if not bool(available_valid.item()):
            raise ValueError("CARE path availability must be binary")
    else:
        torch._assert_async(
            available_valid, "CARE path availability must be binary"
        )
    all_paths_available = path_available.bool().all()
    if path_available.device.type == "cpu":
        if not bool(all_paths_available.item()):
            raise ValueError(
                "CARE clean Stage A requires all four paths for every sample; "
                "missing-modality views belong only to routing calibration"
            )
    else:
        torch._assert_async(
            all_paths_available,
            "CARE clean Stage A requires all four paths for every sample",
        )

    per_path_loss: list[torch.Tensor] = []
    diagnostics: dict[str, torch.Tensor] = {}
    for path_index, path_name in enumerate(PATH_NAMES):
        current_logits = path_logits[path_name]
        if (
            not isinstance(current_logits, torch.Tensor)
            or current_logits.shape != (labels.numel(), 2)
        ):
            raise ValueError(
                f"CARE {path_name.upper()} logits must have shape [B, 2]"
            )
        sample_loss = F.cross_entropy(
            current_logits,
            labels.long(),
            reduction="none",
            label_smoothing=label_smoothing,
        )
        mask = path_available[:, path_index].to(dtype=sample_loss.dtype)
        path_loss = sample_loss.mean()
        per_path_loss.append(path_loss)
        diagnostics[f"care_ce_{path_name}"] = path_loss.detach()
        diagnostics[f"care_available_fraction_{path_name}"] = (
            mask.mean().detach()
        )

    total = torch.stack(per_path_loss).mean()
    diagnostics["loss"] = total.detach()
    diagnostics["care_stage_a_ce"] = total.detach()
    diagnostics["care_active_path_count"] = total.new_tensor(
        float(len(PATH_NAMES))
    )
    if materialize_diagnostics:
        return total, {
            key: float(value.item())
            if isinstance(value, torch.Tensor)
            else float(value)
            for key, value in diagnostics.items()
        }
    return total, diagnostics
