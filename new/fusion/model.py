from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn as nn
from fusion.care_fusion import (
    MODALITY_NAMES,
    PATH_INDEX,
    PATH_MODALITIES,
    PATH_NAMES,
    CAREPathHeads,
    CAREPathRiskHead,
    binary_log_odds,
    route_with_agm_anchor,
)
from fusion.evidence import build_fusion_availability_and_diagnostics

from fusion.constants import ArchitectureConstants, AvailabilityIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.gates import (
    DenseTriModalEmbeddingGate,
    dense_embedding_late_fusion_logits,
)
from fusion.graph_encoders import GraphEncoderGAT, GraphEncoderGCN
from fusion.semantic_categories import validate_api_type_mapping
from fusion.utils import strict_finite_integer
from torch_geometric.utils import softmax


TRI_MODAL_FUSION_MODES = {
    "api_only",
    "graph_only",
    "manifest_only",
    "api_graph_concat",
    "tri_modal_concat",
    "tri_modal_fixed_gate",
    "tri_modal_quality_fusion",
    "tri_modal_dense_embedding_gate",
    "care_droid",
    "discount_probability",
}

API_GRAPH_CONCAT_FUSION_MODES = {"api_graph_concat"}
TRI_MODAL_CONCAT_FUSION_MODES = {"tri_modal_concat"}
CARE_FUSION_MODES = {"care_droid"}


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
        api_observed_support=tensors["api_observed_support"],
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


class ApiSequenceEncoder(nn.Module):
    """API event encoder for the robust tri-modal pipeline."""

    def __init__(
        self,
        num_hash_buckets: int,
        type_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        encoder_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.num_hash_buckets = int(num_hash_buckets)
        self.type_vocab_size = int(type_vocab_size)
        self.emb_dim = int(emb_dim)
        self.encoder_type = str(encoder_type).lower()
        self.max_seq_len = strict_finite_integer(
            max_seq_len, field_name="ApiSequenceEncoder.max_seq_len"
        )
        if self.max_seq_len <= 0:
            raise ValueError("ApiSequenceEncoder.max_seq_len must be positive")
        self.max_valid_api_id = self.num_hash_buckets + 1
        self.overflow_id = self.num_hash_buckets + 2
        self.api_embedding = nn.Embedding(self.overflow_id + 1, emb_dim, padding_idx=0)
        # Extracted API hash ids occupy 2..N+1.  Keep N+1 as a valid bucket and
        # reserve N+2 exclusively for out-of-range values.
        nn.init.zeros_(self.api_embedding.weight[self.overflow_id])
        self.type_embedding = nn.Embedding(self.type_vocab_size, emb_dim)
        self.sensitive_embedding = nn.Embedding(2, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_dropout = nn.Dropout(dropout)

        if self.encoder_type == "bigru":
            if emb_dim % 2 != 0:
                raise ValueError("emb_dim must be even for bigru API encoder")
            self.sequence_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim // 2,
                num_layers=max(1, int(num_layers)),
                batch_first=True,
                bidirectional=True,
                dropout=dropout if int(num_layers) > 1 else 0.0,
            )
            self.pos_embedding = None
        elif self.encoder_type == "transformer":
            if emb_dim % int(num_heads) != 0:
                raise ValueError("api emb_dim must be divisible by api heads")
            self.pos_embedding = nn.Embedding(self.max_seq_len, emb_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=int(num_heads),
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        else:
            raise ValueError(f"Unsupported API encoder type: {encoder_type}")

        self.out_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.pool_score = nn.Sequential(
            nn.Linear(emb_dim, max(hidden_dim // 2, 32)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 32), 1),
        )

    def _empty_output(self, num_graphs: int, device, dtype):
        return (
            torch.zeros((0, self.emb_dim), device=device, dtype=dtype),
            torch.zeros((num_graphs, self.emb_dim), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    @staticmethod
    def _padded_batch(event_emb: torch.Tensor, api_batch: torch.Tensor, num_graphs: int):
        device = event_emb.device
        lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len <= 0:
            return None, None, lengths, None
        padded = event_emb.new_zeros((num_graphs, max_len, event_emb.size(-1)))
        key_padding_mask = torch.ones((num_graphs, max_len), device=device, dtype=torch.bool)
        offsets = torch.zeros((num_graphs + 1,), device=device, dtype=torch.long)
        offsets[1:] = lengths.cumsum(dim=0)
        restore_pos = torch.arange(event_emb.size(0), device=device) - offsets[api_batch]
        padded[api_batch, restore_pos] = event_emb
        key_padding_mask[api_batch, restore_pos] = False
        empty_rows = lengths == 0
        if empty_rows.any():
            key_padding_mask[empty_rows, 0] = False
        return padded, key_padding_mask, lengths, restore_pos

    def _normalize_api_ids(self, api_ids: torch.Tensor, device) -> torch.Tensor:
        api_ids = api_ids.to(device=device, dtype=torch.long)
        overflow = torch.full_like(api_ids, self.overflow_id)
        api_ids = torch.where(api_ids > self.max_valid_api_id, overflow, api_ids)
        return api_ids.clamp_min(0)

    def forward(self, graph_data, num_graphs: int, device, dtype):
        api_ids = getattr(graph_data, "api_ids", None)
        api_batch = getattr(graph_data, "api_batch", None)
        if api_ids is None or api_batch is None or api_ids.numel() == 0:
            return self._empty_output(num_graphs, device, dtype)

        api_ids = self._normalize_api_ids(api_ids, device)
        api_batch = api_batch.to(device=device, dtype=torch.long).view(-1)
        if api_batch.numel() != api_ids.numel():
            raise ValueError(
                f"api_batch length {api_batch.numel()} does not match api_ids length {api_ids.numel()}"
            )
        if num_graphs <= 0 or (api_batch < 0).any() or (api_batch >= num_graphs).any():
            raise ValueError(
                f"api_batch contains indices outside [0, {max(num_graphs - 1, 0)}]"
            )
        if api_batch.numel() > 1 and (api_batch[1:] < api_batch[:-1]).any():
            raise ValueError("api_batch must be grouped in non-decreasing sample order")
        raw_lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
        observed_max = int(raw_lengths.max().item()) if raw_lengths.numel() > 0 else 0
        if observed_max > self.max_seq_len:
            raise RuntimeError(
                "API dataset/encoder budget contract was violated: the encoder "
                "must never truncate silently because reliability evidence and "
                "semantic counts were already computed from the dataset output; "
                f"observed per-sample length {observed_max} exceeds max_seq_len="
                f"{self.max_seq_len}"
            )
        keep = slice(None)

        raw_type_ids = getattr(graph_data, "api_type_ids", None)
        api_type_ids = (
            torch.zeros_like(api_ids)
            if raw_type_ids is None
            else raw_type_ids.to(device=device, dtype=torch.long)[keep].clamp(0, self.type_vocab_size - 1)
        )
        raw_sensitive = getattr(graph_data, "api_sensitive_mask", None)
        api_sensitive = (
            torch.zeros_like(api_ids)
            if raw_sensitive is None
            else raw_sensitive.to(device=device)[keep].float().gt(0.5).long().clamp(0, 1)
        )
        event_emb = self.api_embedding(api_ids) + self.type_embedding(api_type_ids) + self.sensitive_embedding(api_sensitive)
        event_emb = self.input_dropout(self.input_norm(event_emb))
        padded, key_padding_mask, lengths, restore_pos = self._padded_batch(event_emb, api_batch, num_graphs)
        if padded is None:
            return self._empty_output(num_graphs, device, dtype)

        if self.encoder_type == "transformer":
            pos = torch.arange(padded.size(1), device=device).clamp(max=self.max_seq_len - 1)
            encoded = self.sequence_encoder(padded + self.pos_embedding(pos).unsqueeze(0), src_key_padding_mask=key_padding_mask)
        else:
            encoded, _ = self.sequence_encoder(padded)
        token_emb = encoded[api_batch, restore_pos]
        token_emb = self.out_proj(token_emb)
        scores = self.pool_score(token_emb).view(-1)
        scores = scores.masked_fill(~torch.isfinite(scores), 0.0)
        weights = softmax(scores, api_batch, num_nodes=num_graphs).view(-1, 1)
        pooled = token_emb.new_zeros((num_graphs, self.emb_dim))
        pooled.index_add_(0, api_batch, token_emb * weights)
        return token_emb, pooled.to(dtype=dtype), api_batch


class ManifestEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.emb_dim = int(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class TriModalRobustModel(nn.Module):
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
        super().__init__()
        # Defensive check: ensure DEFAULT_API_TYPE_ID_TO_CATEGORY stays
        # consistent with the extractor taxonomy and the 12-D shared space.
        validate_api_type_mapping()
        fusion_mode = str(fusion_mode or "discount_probability")
        if fusion_mode not in TRI_MODAL_FUSION_MODES:
            raise ValueError(f"Unsupported tri-modal fusion_mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.num_classes = int(num_classes)
        self.api_emb_dim = int(api_emb_dim)
        self.graph_emb_dim = int(graph_emb_dim)
        self.manifest_in_dim = int(manifest_in_dim)
        self.manifest_emb_dim = int(manifest_emb_dim)
        if not math.isfinite(float(quality_fusion_temperature)) or float(quality_fusion_temperature) <= 0.0:
            raise ValueError("quality_fusion_temperature must be finite and positive")
        self.quality_fusion_temperature = float(quality_fusion_temperature)
        self.gate_detach = bool(gate_detach)
        self.api_encoder = ApiSequenceEncoder(
            num_hash_buckets=api_num_hash_buckets,
            type_vocab_size=api_type_vocab_size,
            emb_dim=api_emb_dim,
            hidden_dim=api_hidden_dim,
            dropout=api_dropout,
            encoder_type=api_encoder_type,
            num_layers=api_layers,
            num_heads=api_heads,
            max_seq_len=api_max_seq_len,
        )

        graph_encoder_type = str(graph_encoder_type or "gatv2").lower()
        if graph_encoder_type in {"gat", "gatv2"}:
            self.graph_encoder = GraphEncoderGAT(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                heads=graph_heads,
                num_layers=graph_layers,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        elif graph_encoder_type == "gcn":
            self.graph_encoder = GraphEncoderGCN(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        else:
            raise ValueError(f"Unsupported graph_encoder_type: {graph_encoder_type}")

        self.manifest_encoder = ManifestEncoder(
            in_dim=manifest_in_dim,
            emb_dim=manifest_emb_dim,
            hidden_dim=manifest_hidden_dim,
            dropout=manifest_dropout,
        )

        # Baselines use branch-local heads. CARE-Droid instead owns exactly
        # four fixed path heads.
        use_care = self.fusion_mode in CARE_FUSION_MODES
        self.api_head = (
            None if use_care else build_main_head(api_emb_dim, num_classes)
        )
        self.graph_head = (
            None if use_care else build_main_head(graph_emb_dim, num_classes)
        )
        self.manifest_head = (
            None if use_care else build_main_head(manifest_emb_dim, num_classes)
        )
        self.care_path_heads = (
            CAREPathHeads(
                {
                    "api": self.api_emb_dim,
                    "graph": self.graph_emb_dim,
                    "manifest": self.manifest_emb_dim,
                },
                class_count=self.num_classes,
            )
            if use_care
            else None
        )
        self.care_risk_head = CAREPathRiskHead() if use_care else None
        self._care_risk_active = False
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

    def set_care_risk_active(self, enabled: bool) -> None:
        if self.fusion_mode not in CARE_FUSION_MODES:
            if enabled:
                raise RuntimeError(
                    "CARE risk routing requires fusion_mode='care_droid'"
                )
            return
        if self.care_risk_head is None:
            raise RuntimeError("CARE risk head was not initialized")
        if enabled and not bool(
            self.care_risk_head.normalization_is_fitted.item()
        ):
            raise RuntimeError(
                "CARE risk routing cannot be activated before log-odds "
                "normalization is fitted"
            )
        self._care_risk_active = bool(enabled)

    @property
    def care_risk_active(self) -> bool:
        return bool(self._care_risk_active)

    def _encoder_stage_modules(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return the modules whose learned state belongs to clean training.

        The three encoders and method-specific classification heads belong to
        the saved CARE Stage-A state.
        Fusion-specific trainable modules are included only for comparison
        modes that optimize them in this stage.
        ``discount_fusion`` is deliberately absent because every registered
        evidential combination rule is parameter-free.

        Keeping this as an explicit module partition also preserves required
        module buffers without admitting unrelated state.
        """

        modules: list[tuple[str, nn.Module]] = [
            ("api_encoder", self.api_encoder),
            ("graph_encoder", self.graph_encoder),
            ("manifest_encoder", self.manifest_encoder),
        ]
        for name, head in (
            ("api_head", self.api_head),
            ("graph_head", self.graph_head),
            ("manifest_head", self.manifest_head),
        ):
            if head is not None:
                modules.append((name, head))
        if self.care_path_heads is not None:
            modules.append(("care_path_heads", self.care_path_heads))
        if self.api_graph_concat_head is not None:
            modules.append(("api_graph_concat_head", self.api_graph_concat_head))
        if self.tri_concat_head is not None:
            modules.append(("tri_concat_head", self.tri_concat_head))
        if self.dense_embedding_gate is not None:
            modules.append(("dense_embedding_gate", self.dense_embedding_gate))

        # Fail closed if a future fusion mode adds a Stage-1 parameter without
        # declaring its owning module above.  The training loop freezes exactly
        # ``encoder_training_frozen_parameters``; the complement must therefore
        # equal this artifact partition rather than being silently discarded.
        frozen_ids = {id(parameter) for parameter in self.encoder_training_frozen_parameters()}
        expected_stage_parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if id(parameter) not in frozen_ids
        }
        included_parameter_ids = {
            id(parameter)
            for _module_name, module in modules
            for parameter in module.parameters()
        }
        missing = sorted(
            name
            for name, parameter in expected_stage_parameters.items()
            if id(parameter) not in included_parameter_ids
        )
        included_frozen = sorted(
            name
            for name, parameter in self.named_parameters()
            if id(parameter) in included_parameter_ids and id(parameter) in frozen_ids
        )
        if missing or included_frozen:
            raise RuntimeError(
                "Encoder-stage module partition disagrees with the training "
                "freeze policy: "
                f"missing_trainable={missing}, included_frozen={included_frozen}"
            )
        return tuple(modules)

    def encoder_stage_state_keys(self) -> tuple[str, ...]:
        """Return the exact, ordered key contract for a Stage-1 artifact."""

        keys: list[str] = []
        for module_name, module in self._encoder_stage_modules():
            keys.extend(
                f"{module_name}.{local_key}" for local_key in module.state_dict()
            )
        if len(keys) != len(set(keys)):
            raise RuntimeError("Encoder-stage state contains duplicate keys")
        return tuple(keys)

    def encoder_stage_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Materialize an independent Stage-1-only state dictionary.

        Values are cloned so the artifact cannot change later if the live model
        is modified.  Device and dtype are preserved, matching ``state_dict``;
        callers may serialize it directly or move tensors to CPU explicitly.
        """

        state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for module_name, module in self._encoder_stage_modules():
            for local_key, value in module.state_dict().items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "Encoder-stage artifacts support tensor parameters and "
                        f"buffers only; {module_name}.{local_key} has type "
                        f"{type(value).__name__}"
                    )
                state[f"{module_name}.{local_key}"] = value.detach().clone()
        expected = self.encoder_stage_state_keys()
        if tuple(state) != expected:
            raise RuntimeError("Encoder-stage state order disagrees with its key contract")
        return state

    def load_encoder_stage_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        """Strictly load an encoder-stage artifact into the current model.

        Missing, extra, non-tensor, and shape-incompatible values are rejected
        before any live parameter is changed.  Each declared module is then
        loaded with PyTorch's own ``strict=True`` contract; this method never
        relies on ``strict=False`` to hide lifecycle or architecture drift.
        """

        if not isinstance(state, Mapping):
            raise TypeError("encoder-stage state must be a mapping")
        if any(not isinstance(key, str) for key in state):
            raise TypeError("encoder-stage state keys must be strings")

        expected_keys = self.encoder_stage_state_keys()
        expected_set = set(expected_keys)
        actual_set = set(state)
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        if missing or unexpected:
            raise ValueError(
                "Encoder-stage state keys disagree with the current model: "
                f"missing={missing}, unexpected={unexpected}"
            )

        current_state = self.state_dict()
        invalid_types: list[str] = []
        shape_mismatches: dict[str, dict[str, tuple[int, ...]]] = {}
        for key in expected_keys:
            value = state[key]
            if not isinstance(value, torch.Tensor):
                invalid_types.append(f"{key}:{type(value).__name__}")
                continue
            expected_shape = tuple(current_state[key].shape)
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                shape_mismatches[key] = {
                    "expected": expected_shape,
                    "actual": actual_shape,
                }
        if invalid_types:
            raise TypeError(
                "Encoder-stage state values must be tensors: "
                + ", ".join(invalid_types)
            )
        if shape_mismatches:
            raise ValueError(
                "Encoder-stage state tensor shapes disagree with the current "
                f"model: {shape_mismatches}"
            )

        # All global validation is complete, so a failure cannot leave a
        # partially loaded artifact merely because a later module had a bad key
        # or shape.  Local dictionaries have their prefixes removed and retain
        # the exact key set returned by each owning module.
        for module_name, module in self._encoder_stage_modules():
            prefix = f"{module_name}."
            local_state = OrderedDict(
                (key[len(prefix) :], state[key])
                for key in expected_keys
                if key.startswith(prefix)
            )
            module.load_state_dict(local_state, strict=True)

    def encoder_training_frozen_parameters(self) -> list[nn.Parameter]:
        """CARE risk-head parameters excluded from clean Stage-A training."""
        if self.fusion_mode in CARE_FUSION_MODES:
            if self.care_risk_head is None:
                raise RuntimeError("CARE risk head was not initialized")
            return list(self.care_risk_head.parameters())
        return []

    def care_stage_a_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return the strict CARE Stage-A artifact (encoders plus four heads)."""

        if self.fusion_mode not in CARE_FUSION_MODES:
            raise RuntimeError(
                "care_stage_a_state_dict requires fusion_mode='care_droid'"
            )
        return self.encoder_stage_state_dict()

    def load_care_stage_a_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        """Strictly load a CARE Stage-A artifact, excluding the risk head."""

        if self.fusion_mode not in CARE_FUSION_MODES:
            raise RuntimeError(
                "load_care_stage_a_state_dict requires fusion_mode='care_droid'"
            )
        self.load_encoder_stage_state_dict(state)

    def _manifest_input(self, graph_data, batch_size: int, device, dtype) -> torch.Tensor:
        x = getattr(graph_data, "manifest_x", None)
        if not isinstance(x, torch.Tensor):
            return torch.zeros((batch_size, self.manifest_in_dim), device=device, dtype=dtype)
        x = x.to(device=device, dtype=dtype).view(batch_size, -1)
        if x.size(1) < self.manifest_in_dim:
            x = torch.cat([x, x.new_zeros((batch_size, self.manifest_in_dim - x.size(1)))], dim=-1)
        elif x.size(1) > self.manifest_in_dim:
            # Truncation should not happen in normal flow (dataset guards this).
            # Warn here so silent information loss never goes unnoticed.
            warnings.warn(
                f"manifest_x dim {x.size(1)} > configured {self.manifest_in_dim}; "
                f"truncating trailing features. Check dataset or model config."
            )
            x = x[:, : self.manifest_in_dim]
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode_api(self, graph_data, batch_size: int, device, dtype):
        """Encode the API stream and expose support from the stream actually used.

        ``api_observed_support`` is deliberately computed from ``api_batch``
        returned by the encoder, not from extraction-time or pre-perturbation
        counts stored in a PT payload.  It therefore measures only how much of
        the fixed encoder capacity is occupied by the input visible at
        inference time:

            log(1 + n_visible) / log(1 + encoder_capacity)

        This is a bounded amount-of-current-evidence signal, not an estimate
        of an APK's unknown original completeness.
        """

        _, pooled, encoded_api_batch = self.api_encoder(
            graph_data, batch_size, device, dtype
        )
        if batch_size <= 0:
            raise ValueError("API observed support requires a positive batch size")
        if (
            encoded_api_batch.ndim != 1
            or encoded_api_batch.dtype != torch.long
        ):
            raise RuntimeError(
                "ApiSequenceEncoder must return a one-dimensional long "
                "api_batch index tensor"
            )
        counts = torch.bincount(
            encoded_api_batch,
            minlength=batch_size,
        ).to(device=device, dtype=torch.float32)
        if counts.numel() != batch_size:
            raise RuntimeError(
                "API observed-support counts disagree with the model batch size"
            )
        capacity = int(self.api_encoder.max_seq_len)
        if capacity <= 0:
            raise RuntimeError("API encoder capacity must be positive")
        count_contract_valid = (counts <= float(capacity)).all()
        count_contract_message = (
            "API observed-support count exceeds the encoder capacity; "
            "the dataset/encoder budget contract was violated"
        )
        if counts.device.type == "cpu":
            if not bool(count_contract_valid.item()):
                raise RuntimeError(count_contract_message)
        else:
            torch._assert_async(count_contract_valid, count_contract_message)
        denominator = math.log1p(float(capacity))
        support = torch.log1p(counts).div(denominator).clamp(0.0, 1.0)
        return pooled, support

    @staticmethod
    def _availability_mask(
        graph_data,
        name: str,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Current dataset batch is missing mandatory availability mask {name!r}"
            )
        value = value.to(device=device, dtype=dtype).view(batch_size, -1)
        if value.size(1) == 0:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        return value[:, :1].gt(0.0).to(dtype=dtype)

    def forward(
        self,
        graph_data,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        batch_size = int(getattr(graph_data, "num_graphs", 1))

        api_emb, api_observed_support = self._encode_api(
            graph_data, batch_size, device, dtype
        )
        _, graph_emb, _, _ = self.graph_encoder(graph_data)
        manifest_x = self._manifest_input(graph_data, batch_size, device, dtype)
        manifest_emb = self.manifest_encoder(manifest_x)

        api_available = self._availability_mask(
            graph_data, "api_alive", batch_size, device, dtype
        )
        graph_available = self._availability_mask(
            graph_data, "graph_alive", batch_size, device, dtype
        )
        manifest_available = self._availability_mask(
            graph_data, "manifest_alive", batch_size, device, dtype
        )
        primary_available = torch.cat(
            [api_available, graph_available, manifest_available], dim=-1
        ).bool()
        # CARE uses only fixed-path log-odds and the three hard alive bits.
        # API support remains available to registered comparison methods but
        # is deliberately absent from CARE outputs as well as its risk head.
        extra = (
            {}
            if self.fusion_mode in CARE_FUSION_MODES
            else {"api_observed_support": api_observed_support.detach()}
        )

        if self.fusion_mode in CARE_FUSION_MODES:
            if self.care_path_heads is None or self.care_risk_head is None:
                raise RuntimeError("CARE-Droid modules are incomplete")
            placeholder_logits = api_emb.new_zeros((batch_size, 2))
            availability, diagnostics = build_fusion_availability_and_diagnostics(
                graph_data,
                placeholder_logits,
                placeholder_logits,
                placeholder_logits,
                api_emb,
                graph_emb,
                manifest_emb,
                materialize_diagnostics=not self.training,
            )
            modality_alive = availability.bool()
            path_logits, path_available = self.care_path_heads(
                {
                    "api": api_emb,
                    "graph": graph_emb,
                    "manifest": manifest_emb,
                },
                modality_alive,
            )
            path_log_odds = torch.stack(
                [binary_log_odds(path_logits[name]) for name in PATH_NAMES],
                dim=-1,
            )
            extra.update(diagnostics)
            extra.update(
                {
                    "fusion_availability": availability.detach(),
                    "care_path_logits": path_logits,
                    "care_path_available": path_available.detach(),
                    "care_modality_alive": modality_alive.detach(),
                    "care_path_log_odds": path_log_odds.detach(),
                }
            )
            selective_eligible = modality_alive.sum(dim=-1).ge(2)
            if self.care_risk_active:
                normalized_log_odds = self.care_risk_head.normalize(
                    path_log_odds,
                    path_available,
                )
                correctness = self.care_risk_head.score_all(
                    normalized_log_odds,
                    modality_alive,
                )
                correctness = torch.where(
                    path_available,
                    correctness,
                    torch.zeros_like(correctness),
                )
                routing = route_with_agm_anchor(
                    path_logits,
                    path_available,
                    correctness,
                )
                logits = routing.selected_logits
                extra.update(
                    {
                        "care_path_correctness": correctness,
                        "care_selected_path_index": routing.selected_path_index,
                        "care_selected_score": routing.selected_score,
                        "care_reject": routing.reject,
                        "care_disagreement_with_agm": (
                            routing.disagreement_with_agm
                        ),
                    }
                )
                selected_path_index = routing.selected_path_index
            else:
                # Stage A is selected only by the clean AGM classifier. The
                # other three equal-weight path losses shape the shared
                # encoders but do not create a random, unfitted router.
                logits = path_logits["agm"]
                selected_path_index = torch.where(
                    path_available[:, PATH_INDEX["agm"]],
                    torch.full(
                        (batch_size,),
                        PATH_INDEX["agm"],
                        device=device,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (batch_size,), -1, device=device, dtype=torch.long
                    ),
                )

            gate_weights = logits.new_zeros((batch_size, len(MODALITY_NAMES)))
            for path_index, path_name in enumerate(PATH_NAMES):
                selected_mask = selected_path_index.eq(path_index).to(dtype=dtype)
                contribution = 1.0 / float(len(PATH_MODALITIES[path_name]))
                for modality_name in PATH_MODALITIES[path_name]:
                    modality_index = MODALITY_NAMES.index(modality_name)
                    gate_weights[:, modality_index] = (
                        gate_weights[:, modality_index]
                        + selected_mask * contribution
                    )
        else:
            if self.api_head is None or self.graph_head is None or self.manifest_head is None:
                raise RuntimeError("Baseline branch heads were not initialized")
            api_logits = self.api_head(api_emb)
            graph_logits = self.graph_head(graph_emb)
            manifest_logits = self.manifest_head(manifest_emb)
            extra.update(
                {
                    "api_logits_aux": api_logits,
                    "graph_logits_aux": graph_logits,
                    "manifest_logits_aux": manifest_logits,
                }
            )
            fusion_tensors = {
                "api_logits": api_logits,
                "graph_logits": graph_logits,
                "manifest_logits": manifest_logits,
                "api_emb": api_emb,
                "graph_emb": graph_emb,
                "manifest_emb": manifest_emb,
                "graph_data": graph_data,
                "api_observed_support": api_observed_support,
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

        if (
            self.fusion_mode not in CARE_FUSION_MODES
            and "api_alive" not in extra
            and not self.training
        ):
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
