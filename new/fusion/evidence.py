from __future__ import annotations

import torch

from fusion.constants import AvailabilityIndex


def hard_alive_attr(
    graph_data,
    name: str,
    batch_size: int,
    device,
    dtype,
) -> torch.Tensor:
    """Read one mandatory binary availability field without any fallback."""

    value = getattr(graph_data, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"fusion availability requires explicit tensor field {name!r}"
        )
    if value.numel() != batch_size:
        raise ValueError(
            f"fusion availability field {name!r} must contain exactly "
            f"{batch_size} values, got shape {tuple(value.shape)}"
        )
    return value.to(device=device, dtype=dtype).view(batch_size, 1)


def validate_binary_availability(availability: torch.Tensor) -> None:
    valid = (
        torch.isfinite(availability)
        & ((availability == 0.0) | (availability == 1.0))
    ).all()
    message = "fusion availability must be finite and binary (0 or 1)"
    if availability.device.type == "cpu":
        if not bool(valid.item()):
            raise ValueError(message)
    else:
        # Preserve strict validation without synchronizing each GPU batch.
        torch._assert_async(valid, message)


def build_fusion_availability_and_diagnostics(
    graph_data,
    api_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    manifest_logits: torch.Tensor,
    api_emb: torch.Tensor,
    graph_emb: torch.Tensor,
    manifest_emb: torch.Tensor,
    *,
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the complete model-visible metadata: exactly three alive bits."""

    del graph_logits, manifest_logits, api_emb, graph_emb, manifest_emb
    batch_size = api_logits.size(0)
    device = api_logits.device
    dtype = api_logits.dtype
    api_alive = hard_alive_attr(
        graph_data, "api_alive", batch_size, device, dtype
    )
    graph_alive = hard_alive_attr(
        graph_data, "graph_alive", batch_size, device, dtype
    )
    manifest_alive = hard_alive_attr(
        graph_data, "manifest_alive", batch_size, device, dtype
    )
    availability = torch.cat(
        [api_alive, graph_alive, manifest_alive],
        dim=-1,
    )
    validate_binary_availability(availability)
    if availability.size(-1) != AvailabilityIndex.BASE_DIM:
        raise RuntimeError(
            "Fusion availability dimension mismatch: built "
            f"{availability.size(-1)}, expected {AvailabilityIndex.BASE_DIM}"
        )
    if not materialize_diagnostics:
        return availability, {}
    return availability, {
        "api_alive": api_alive.detach().view(batch_size),
        "graph_alive": graph_alive.detach().view(batch_size),
        "manifest_alive": manifest_alive.detach().view(batch_size),
    }
