from __future__ import annotations

import torch
import torch.nn as nn

from fusion.constants import ArchitectureConstants, GateConstants


TRI_MODAL_EMBEDDING_BRANCHES = ("api", "graph", "manifest")


class DenseTriModalEmbeddingGate(nn.Module):
    """Small dense late-fusion gate over the three modality embeddings.

    This is an intentionally generic black-box gating baseline, not a strict
    reproduction of a sparse mixture-of-experts model.  Each branch embedding
    is normalised, unavailable branches are zeroed before concatenation, and a
    dense MLP produces one routing logit per modality.  Softmax normalisation is
    restricted to alive branches, so an unavailable encoder can never receive
    decision mass through a learned bias.

    ``detach_embeddings`` controls only the gate-input path.  The fused branch
    logits remain differentiable in either case.  The adapted baseline config
    leaves it disabled, matching an ordinary end-to-end dense gate trained only
    on the training split.
    """

    def __init__(
        self,
        embedding_dims: dict[str, int],
        *,
        hidden_dim: int | None = None,
        detach_embeddings: bool = False,
    ):
        super().__init__()
        if set(embedding_dims) != set(TRI_MODAL_EMBEDDING_BRANCHES):
            raise ValueError(
                "DenseTriModalEmbeddingGate embedding_dims must contain exactly "
                f"{TRI_MODAL_EMBEDDING_BRANCHES}"
            )
        self.embedding_dims = {
            name: int(embedding_dims[name]) for name in TRI_MODAL_EMBEDDING_BRANCHES
        }
        if any(value <= 0 for value in self.embedding_dims.values()):
            raise ValueError("all dense embedding-gate dimensions must be positive")
        hidden_dim = int(hidden_dim or ArchitectureConstants.GATE_HIDDEN_DIM)
        if hidden_dim <= 0:
            raise ValueError("dense embedding-gate hidden_dim must be positive")
        self.detach_embeddings = bool(detach_embeddings)
        self.input_norms = nn.ModuleDict(
            {
                name: nn.LayerNorm(self.embedding_dims[name])
                for name in TRI_MODAL_EMBEDDING_BRANCHES
            }
        )
        input_dim = sum(self.embedding_dims.values()) + len(
            TRI_MODAL_EMBEDDING_BRANCHES
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(TRI_MODAL_EMBEDDING_BRANCHES)),
        )
        # Begin from an alive-only uniform route.  This avoids assigning a
        # branch-specific competence prior before the data have trained it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if set(embeddings) != set(TRI_MODAL_EMBEDDING_BRANCHES):
            raise ValueError(
                "dense embedding gate requires exactly API, Graph, and Manifest "
                "embeddings"
            )
        reference = embeddings[TRI_MODAL_EMBEDDING_BRANCHES[0]]
        if reference.ndim != 2:
            raise ValueError("dense embedding-gate inputs must have shape [B, D]")
        batch_size = int(reference.size(0))
        if alive.shape != (batch_size, len(TRI_MODAL_EMBEDDING_BRANCHES)):
            raise ValueError(
                "dense embedding-gate alive mask must have shape "
                f"[B, {len(TRI_MODAL_EMBEDDING_BRANCHES)}], got {tuple(alive.shape)}"
            )
        alive_mask = alive.to(
            device=reference.device, dtype=reference.dtype
        ).gt(0.0)
        alive_float = alive_mask.to(dtype=reference.dtype)

        features: list[torch.Tensor] = []
        for index, name in enumerate(TRI_MODAL_EMBEDDING_BRANCHES):
            value = embeddings[name]
            expected_shape = (batch_size, self.embedding_dims[name])
            if value.shape != expected_shape:
                raise ValueError(
                    f"dense embedding gate expected {name} shape {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )
            value = value.to(device=reference.device, dtype=reference.dtype)
            if self.detach_embeddings:
                value = value.detach()
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            value = self.input_norms[name](value)
            features.append(value * alive_float[:, index : index + 1])
        gate_input = torch.cat([*features, alive_float], dim=-1)
        scores = self.net(gate_input)

        has_available = alive_mask.any(dim=-1, keepdim=True)
        unavailable_floor = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~alive_mask, unavailable_floor)
        # Avoid an all-(-inf) softmax for the explicit all-dead fallback.
        stable_scores = torch.where(
            has_available, masked_scores, torch.zeros_like(masked_scores)
        )
        weights = torch.softmax(stable_scores, dim=-1) * alive_float
        weights = torch.where(
            has_available,
            weights / weights.sum(dim=-1, keepdim=True).clamp_min(GateConstants.EPS),
            torch.zeros_like(weights),
        )
        return weights, masked_scores


def dense_embedding_late_fusion_logits(
    branch_logits: dict[str, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Fuse the three branch logits, with zero logits as the all-dead fallback."""
    if set(branch_logits) != set(TRI_MODAL_EMBEDDING_BRANCHES):
        raise ValueError(
            "dense embedding late fusion requires exactly API, Graph, and "
            "Manifest logits"
        )
    reference = branch_logits[TRI_MODAL_EMBEDDING_BRANCHES[0]]
    if reference.ndim != 2:
        raise ValueError("dense embedding late-fusion logits must have shape [B, C]")
    expected_weight_shape = (reference.size(0), len(TRI_MODAL_EMBEDDING_BRANCHES))
    if weights.shape != expected_weight_shape:
        raise ValueError(
            f"dense embedding late-fusion weights must have shape {expected_weight_shape}, "
            f"got {tuple(weights.shape)}"
        )
    ordered: list[torch.Tensor] = []
    for name in TRI_MODAL_EMBEDDING_BRANCHES:
        value = branch_logits[name]
        if value.shape != reference.shape:
            raise ValueError("all dense embedding late-fusion logits must agree in shape")
        ordered.append(value.to(device=reference.device, dtype=reference.dtype))
    stacked = torch.stack(ordered, dim=1)
    operative_weights = weights.to(device=reference.device, dtype=reference.dtype)
    # When all sources are dead, DenseTriModalEmbeddingGate returns all-zero
    # weights. The weighted sum is therefore explicit zero logits and softmax is
    # the uniform class distribution, independent of placeholder branch heads.
    return (stacked * operative_weights.unsqueeze(-1)).sum(dim=1)
