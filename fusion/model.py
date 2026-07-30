from __future__ import annotations

import math

import torch
import torch.nn as nn
from fusion.evidence import build_fusion_availability_and_diagnostics

from fusion.constants import ArchitectureConstants, AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.gates import (
    DenseTriModalEmbeddingGate,
    dense_embedding_late_fusion_logits,
)
from fusion.modality_encoders import TriModalEncoderBackbone


TRI_MODAL_FUSION_MODES = {
    "api_only",
    "graph_only",
    "manifest_only",
    "api_graph_concat",
    "tri_modal_concat",
    "tri_modal_fixed_gate",
    "tri_modal_quality_fusion",
    "tri_modal_dense_embedding_gate",
    "discount_probability",
}

API_GRAPH_CONCAT_FUSION_MODES = {"api_graph_concat"}
TRI_MODAL_CONCAT_FUSION_MODES = {"tri_modal_concat"}


# ── fusion-mode dispatch helpers ──────────────────────────────────────
# Each handler receives (model, batch_size, device, dtype, tensors, extra)
# and returns (logits, gate_weights, extra).

def quality_aware_logit_fusion(
    logits_by_branch: list[torch.Tensor],
    alive: torch.Tensor,
    *,
    temperature: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QMF-Energy detached weighting component for a shared-label task.

    Final logits follow QMF's late-fusion energy component. This helper does
    not implement QMF's history-based confidence-ranking objective, so callers
    must not identify it as the complete QMF method. The normalized weights
    returned here are diagnostics only and do not replace the unnormalized
    energy-weighted decision rule.
    """
    if not logits_by_branch:
        raise ValueError("quality_aware_logit_fusion requires at least one branch")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("quality fusion temperature must be finite and positive")
    logits = torch.stack(logits_by_branch, dim=1)
    if alive.shape != logits.shape[:2]:
        raise ValueError(
            f"alive mask shape {tuple(alive.shape)} does not match logits {tuple(logits.shape[:2])}"
        )
    alive = alive.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    energy = torch.logsumexp(logits, dim=-1) / float(temperature)
    quality = energy.detach() * alive
    fused = (logits * quality.unsqueeze(-1)).sum(dim=1)

    has_source = alive.sum(dim=-1, keepdim=True) > 0.0
    masked_energy = energy.masked_fill(alive <= 0.0, torch.finfo(energy.dtype).min)
    diagnostic_weights = torch.softmax(masked_energy, dim=-1) * alive
    diagnostic_weights = diagnostic_weights / diagnostic_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    uniform = torch.full_like(diagnostic_weights, 1.0 / float(len(logits_by_branch)))
    diagnostic_weights = torch.where(has_source, diagnostic_weights, uniform)
    # No branch is observable in the all-dead endpoint. Returning head-bias
    # logits would manufacture class evidence from absent modalities.
    fused = torch.where(has_source, fused, torch.zeros_like(fused))
    return fused, diagnostic_weights, energy


def _fusion_single_api(_model, batch_size, device, dtype, tensors, extra):
    gate = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    gate[:, 0] = 1.0
    return tensors["api_logits"], gate, extra


def _fusion_single_graph(_model, batch_size, device, dtype, tensors, extra):
    gate = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    gate[:, 1] = 1.0
    return tensors["graph_logits"], gate, extra


def _fusion_single_manifest(_model, batch_size, device, dtype, tensors, extra):
    gate = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    gate[:, 2] = 1.0
    return tensors["manifest_logits"], gate, extra


def _fusion_api_graph_concat(model, batch_size, device, dtype, tensors, extra):
    if model.api_graph_concat_head is None:
        raise RuntimeError(
            "api_graph_concat fusion requires an initialized API/Graph concat head"
        )
    logits = model.api_graph_concat_head(tensors["api_graph_concat_input"])
    gate = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    gate[:, :2] = 0.5
    return logits, gate, extra


def _fusion_tri_concat(model, batch_size, device, dtype, tensors, extra):
    if model.tri_concat_head is None:
        raise RuntimeError(
            "tri_modal_concat fusion requires an initialized tri-modal concat head"
        )
    logits = model.tri_concat_head(tensors["tri_modal_concat_input"])
    gate = torch.full(
        (batch_size, 3), 1.0 / 3.0, device=device, dtype=dtype
    )
    return logits, gate, extra


def _fusion_quality_aware(model, batch_size, device, dtype, tensors, extra):
    """Adapt QMF energy-aware late fusion to the three APK modalities."""
    availability, diagnostics = build_fusion_availability_and_diagnostics(
        tensors["graph_data"],
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["api_emb"],
        tensors["graph_emb"],
        tensors["manifest_emb"],
        materialize_diagnostics=not model.training,
    )
    extra.update(diagnostics)
    extra["fusion_availability"] = availability.detach()
    alive = torch.stack(
        [
            availability[:, AvailabilityIndex.API_ALIVE],
            availability[:, AvailabilityIndex.GRAPH_ALIVE],
            availability[:, AvailabilityIndex.MANIFEST_ALIVE],
        ],
        dim=-1,
    )
    logits, weights3, energy = quality_aware_logit_fusion(
        [
            tensors["api_logits"],
            tensors["graph_logits"],
            tensors["manifest_logits"],
        ],
        alive,
        temperature=model.quality_fusion_temperature,
    )
    gate = weights3.to(dtype=dtype)
    for index, name in enumerate(("api", "graph", "manifest")):
        extra[f"qmf_energy_{name}"] = energy[:, index]
        extra[f"fusion_weight_{name}"] = weights3[:, index]
    return logits, gate, extra


def _fusion_dense_embedding_gate(model, batch_size, device, dtype, tensors, extra):
    """Adapted dense embedding-gated three-branch late-fusion baseline."""
    availability, diagnostics = build_fusion_availability_and_diagnostics(
        tensors["graph_data"],
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["api_emb"],
        tensors["graph_emb"],
        tensors["manifest_emb"],
        materialize_diagnostics=not model.training,
    )
    extra.update(diagnostics)
    extra["fusion_availability"] = availability.detach()
    alive = torch.stack(
        [
            availability[:, AvailabilityIndex.API_ALIVE],
            availability[:, AvailabilityIndex.GRAPH_ALIVE],
            availability[:, AvailabilityIndex.MANIFEST_ALIVE],
        ],
        dim=-1,
    )
    if model.dense_embedding_gate is None:
        raise RuntimeError(
            "tri_modal_dense_embedding_gate requires an initialized dense gate"
        )
    weights3, gate_scores = model.dense_embedding_gate(
        {
            "api": tensors["api_emb"],
            "graph": tensors["graph_emb"],
            "manifest": tensors["manifest_emb"],
        },
        alive,
    )
    logits = dense_embedding_late_fusion_logits(
        {
            "api": tensors["api_logits"],
            "graph": tensors["graph_logits"],
            "manifest": tensors["manifest_logits"],
        },
        weights3,
    )
    gate = weights3.to(dtype=dtype)
    extra["dense_embedding_gate_scores"] = gate_scores
    for index, name in enumerate(("api", "graph", "manifest")):
        extra[f"fusion_weight_{name}"] = weights3[:, index]
    return logits, gate, extra


def _fusion_fixed_gate(model, batch_size, device, dtype, tensors, extra):
    """Equal-weight late fusion over the modalities alive in each sample."""
    availability, diagnostics = build_fusion_availability_and_diagnostics(
        tensors["graph_data"],
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["api_emb"],
        tensors["graph_emb"],
        tensors["manifest_emb"],
        materialize_diagnostics=not model.training,
    )
    extra.update(diagnostics)
    extra["fusion_availability"] = availability.detach()

    alive = torch.stack(
        [
            availability[:, AvailabilityIndex.API_ALIVE],
            availability[:, AvailabilityIndex.GRAPH_ALIVE],
            availability[:, AvailabilityIndex.MANIFEST_ALIVE],
        ],
        dim=-1,
    ).to(device=device, dtype=dtype)
    alive_count = alive.sum(dim=-1, keepdim=True)
    gate_weights = torch.where(
        alive_count > 0,
        alive / alive_count.clamp_min(1.0),
        torch.zeros_like(alive),
    )
    logits = (
        gate_weights[:, 0:1] * tensors["api_logits"]
        + gate_weights[:, 1:2] * tensors["graph_logits"]
        + gate_weights[:, 2:3] * tensors["manifest_logits"]
    )
    return logits, gate_weights, extra


def _fusion_discount_probability(model, batch_size, device, dtype, tensors, extra):
    del batch_size, device, dtype
    if model.discount_fusion is None:
        raise RuntimeError(
            "discount_probability fusion requires an initialized discount fusion module"
        )
    availability, diagnostics = build_fusion_availability_and_diagnostics(
        tensors["graph_data"],
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["api_emb"],
        tensors["graph_emb"],
        tensors["manifest_emb"],
        materialize_diagnostics=not model.training,
    )
    extra.update(diagnostics)
    extra["fusion_availability"] = availability.detach()
    fusion_outputs = model.discount_fusion(
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        availability,
    )
    extra.update(fusion_outputs)
    return fusion_outputs["final_logits"], fusion_outputs["fusion_weights"], extra


FUSION_DISPATCH: dict[str, callable] = {
    "api_only": _fusion_single_api,
    "graph_only": _fusion_single_graph,
    "manifest_only": _fusion_single_manifest,
    "api_graph_concat": _fusion_api_graph_concat,
    "tri_modal_concat": _fusion_tri_concat,
    "tri_modal_fixed_gate": _fusion_fixed_gate,
    "tri_modal_quality_fusion": _fusion_quality_aware,
    "tri_modal_dense_embedding_gate": _fusion_dense_embedding_gate,
    "discount_probability": _fusion_discount_probability,
}


def build_main_head(in_dim: int, num_classes: int) -> nn.Sequential:
    hidden = ArchitectureConstants.HEAD_HIDDEN_DIMS[-1]
    drop = ArchitectureConstants.HEAD_DROPOUT_RATES[-1]
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(drop),
        nn.Linear(hidden, num_classes),
    )


class TriModalRobustModel(TriModalEncoderBackbone):
    def __init__(
        self,
        in_feat_dim: int = 515,
        num_classes: int = 2,
        fusion_mode: str = "discount_probability",
        api_num_hash_buckets: int = 8192,
        api_type_vocab_size: int = 16,
        api_emb_dim: int = 128,
        api_hidden_dim: int = 256,
        api_dropout: float = 0.15,
        api_encoder_type: str = "transformer",
        api_layers: int = 2,
        api_heads: int = 4,
        api_max_seq_len: int = 1024,
        graph_emb_dim: int = 128,
        graph_hidden: int = 128,
        graph_heads: int = 4,
        graph_layers: int = 2,
        graph_encoder_type: str = "gatv2",
        max_nodes_gnn: int = 12288,
        use_graph_behavior_hint: bool = False,
        manifest_in_dim: int = 256,
        manifest_emb_dim: int = 128,
        manifest_hidden_dim: int = 256,
        manifest_dropout: float = 0.1,
        quality_fusion_temperature: float = 10.0,
        gate_hidden_dim: int = 128,
        gate_detach: bool = True,
        discount_fusion_config: dict | None = None,
    ):
        super().__init__(
            in_feat_dim=in_feat_dim,
            api_num_hash_buckets=api_num_hash_buckets,
            api_type_vocab_size=api_type_vocab_size,
            api_emb_dim=api_emb_dim,
            api_hidden_dim=api_hidden_dim,
            api_dropout=api_dropout,
            api_encoder_type=api_encoder_type,
            api_layers=api_layers,
            api_heads=api_heads,
            api_max_seq_len=api_max_seq_len,
            graph_emb_dim=graph_emb_dim,
            graph_hidden=graph_hidden,
            graph_heads=graph_heads,
            graph_layers=graph_layers,
            graph_encoder_type=graph_encoder_type,
            max_nodes_gnn=max_nodes_gnn,
            use_graph_behavior_hint=use_graph_behavior_hint,
            manifest_in_dim=manifest_in_dim,
            manifest_emb_dim=manifest_emb_dim,
            manifest_hidden_dim=manifest_hidden_dim,
            manifest_dropout=manifest_dropout,
        )
        fusion_mode = str(fusion_mode or "discount_probability")
        if fusion_mode not in TRI_MODAL_FUSION_MODES:
            raise ValueError(f"Unsupported tri-modal fusion_mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.num_classes = int(num_classes)
        if not math.isfinite(float(quality_fusion_temperature)) or float(quality_fusion_temperature) <= 0.0:
            raise ValueError("quality_fusion_temperature must be finite and positive")
        self.quality_fusion_temperature = float(quality_fusion_temperature)
        self.api_head = build_main_head(api_emb_dim, num_classes)
        self.graph_head = build_main_head(graph_emb_dim, num_classes)
        self.manifest_head = build_main_head(manifest_emb_dim, num_classes)
        self.api_graph_concat_head = (
            build_main_head(api_emb_dim + graph_emb_dim, num_classes)
            if self.fusion_mode in API_GRAPH_CONCAT_FUSION_MODES
            else None
        )
        self.tri_concat_head = (
            build_main_head(
                api_emb_dim + graph_emb_dim + manifest_emb_dim,
                num_classes,
            )
            if self.fusion_mode in TRI_MODAL_CONCAT_FUSION_MODES
            else None
        )
        # Independent learned embedding-gate comparison, instantiated only for
        # its registered fusion mode.
        self.dense_embedding_gate = (
            DenseTriModalEmbeddingGate(
                {
                    "api": self.api_emb_dim,
                    "graph": self.graph_emb_dim,
                    "manifest": self.manifest_emb_dim,
                },
                hidden_dim=gate_hidden_dim,
                detach_embeddings=gate_detach,
            )
            if self.fusion_mode == "tri_modal_dense_embedding_gate"
            else None
        )
        self.discount_fusion = (
            DiscountProbabilityFusion(discount_fusion_config)
            if self.fusion_mode == "discount_probability"
            else None
        )

    def forward(
        self,
        graph_data,
    ):
        encoded = self.encode_modalities(graph_data)
        device = encoded["device"]
        dtype = encoded["dtype"]
        batch_size = int(encoded["batch_size"])
        api_emb = encoded["api_emb"]
        graph_emb = encoded["graph_emb"]
        manifest_emb = encoded["manifest_emb"]
        api_available = encoded["api_available"]
        graph_available = encoded["graph_available"]
        manifest_available = encoded["manifest_available"]
        primary_available = encoded["modality_alive"]

        api_logits = self.api_head(api_emb)
        graph_logits = self.graph_head(graph_emb)
        manifest_logits = self.manifest_head(manifest_emb)
        extra = {
            "api_logits_aux": api_logits,
            "graph_logits_aux": graph_logits,
            "manifest_logits_aux": manifest_logits,
        }
        fusion_tensors = {
            "api_logits": api_logits,
            "graph_logits": graph_logits,
            "manifest_logits": manifest_logits,
            "api_emb": api_emb,
            "graph_emb": graph_emb,
            "manifest_emb": manifest_emb,
            "graph_data": graph_data,
        }
        if self.fusion_mode in API_GRAPH_CONCAT_FUSION_MODES:
            fusion_tensors["api_graph_concat_input"] = torch.cat(
                [
                    api_emb * api_available,
                    graph_emb * graph_available,
                ],
                dim=-1,
            )
        elif self.fusion_mode in TRI_MODAL_CONCAT_FUSION_MODES:
            fusion_tensors["tri_modal_concat_input"] = torch.cat(
                [
                    api_emb * api_available,
                    graph_emb * graph_available,
                    manifest_emb * manifest_available,
                ],
                dim=-1,
            )
        handler = FUSION_DISPATCH[self.fusion_mode]
        logits, gate_weights, extra = handler(
            self,
            batch_size,
            device,
            dtype,
            fusion_tensors,
            extra,
        )
        if self.fusion_mode == "api_only":
            selective_eligible = primary_available[:, 0]
        elif self.fusion_mode == "graph_only":
            selective_eligible = primary_available[:, 1]
        elif self.fusion_mode == "manifest_only":
            selective_eligible = primary_available[:, 2]
        elif self.fusion_mode == "api_graph_concat":
            selective_eligible = primary_available[:, :2].any(dim=-1)
        else:
            selective_eligible = primary_available.any(dim=-1)
        if (
            self.fusion_mode != "discount_probability"
            and not bool(extra.get("final_is_log_probability", False))
        ):
            # Canonical logit baselines use cross entropy, so zero logits are
            # the explicit uniform-predictive fallback when every branch that
            # the method consumes is unavailable. Fixed evidential fusion
            # already returns normalized log probabilities and retains its own
            # log(1 / K) fallback.
            logits = torch.where(
                selective_eligible.unsqueeze(-1),
                logits,
                torch.zeros_like(logits),
            )
        # Explicitly describe availability of the branches consumed by this
        # fusion mode. Evaluation must not infer it from unrelated live inputs.
        extra["selective_eligible"] = selective_eligible

        extra["gate_weights"] = gate_weights.detach()

        if "api_alive" not in extra and not self.training:
            _, diagnostics = build_fusion_availability_and_diagnostics(
                graph_data,
                api_logits,
                graph_logits,
                manifest_logits,
                api_emb,
                graph_emb,
                manifest_emb,
                materialize_diagnostics=True,
            )
            extra.update(diagnostics)

        return logits, extra
