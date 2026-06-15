from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.evidence import build_evidence

from fusion.constants import ArchitectureConstants, EvidenceIndex, GateConstants
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.gates import (
    FourBranchEvidenceGate,
    heuristic_reliability_gate,
    confidence_gate,
    apply_alive_mask,
)
from fusion.graph_encoders import GraphEncoderGAT, GraphEncoderGCN
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM, validate_api_type_mapping
from fusion.semantic_cross_attention import ReliabilityAwareSemanticCrossAttention
from torch_geometric.utils import softmax


TRI_MODAL_FUSION_MODES = {
    "api",
    "api_only",
    "graph",
    "graph_only",
    "manifest",
    "manifest_only",
    "api_graph",
    "api_graph_concat",
    "api_graph_manifest_concat",
    "tri_modal_concat",
    "tri_modal_fixed_gate",
    "tri_modal_reliability_gate",
    "tri_modal_confidence_gate",
    "tri_modal_ours",
    "discount_probability",
}


# ── fusion-mode dispatch helpers ──────────────────────────────────────
# Each handler receives (model, batch_size, device, dtype, tensors, extra)
# and returns (logits, gate_weights, extra).

def _fusion_single_api(_model, batch_size, device, dtype, tensors, extra):
    gate = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    gate[:, 0] = 1.0
    return tensors["api_logits"], gate, extra


def _fusion_single_graph(_model, batch_size, device, dtype, tensors, extra):
    gate = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    gate[:, 1] = 1.0
    return tensors["graph_logits"], gate, extra


def _fusion_single_manifest(_model, batch_size, device, dtype, tensors, extra):
    gate = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    gate[:, 2] = 1.0
    return tensors["manifest_logits"], gate, extra


def _fusion_api_graph_concat(model, batch_size, device, dtype, tensors, extra):
    logits = model.api_graph_concat_head(
        torch.cat([tensors["api_emb"], tensors["graph_emb"]], dim=-1)
    )
    gate = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    gate[:, 3] = 1.0
    extra["joint_logits_aux"] = logits
    return logits, gate, extra


def _fusion_tri_concat(model, batch_size, device, dtype, tensors, extra):
    logits = model.tri_concat_head(tensors["joint_input"])
    gate = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    gate[:, 3] = 1.0
    extra["joint_logits_aux"] = logits
    return logits, gate, extra


def _fusion_evidence_based(model, batch_size, device, dtype, tensors, extra):
    """Evidence-based gate fusion (fixed / reliability / confidence / learned)."""
    evidence, diagnostics = build_evidence(
        tensors["graph_data"],
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["joint_logits"],
        tensors["api_emb"],
        tensors["graph_emb"],
        tensors["manifest_emb"],
        use_consistency_evidence=model.use_consistency_evidence,
        use_conflict_evidence=model.use_conflict_evidence,
        use_perturbation_evidence=model.use_perturbation_evidence,
    )
    extra.update(diagnostics)
    extra["gate_evidence"] = evidence.detach()

    mode = model.fusion_mode
    if mode == "tri_modal_fixed_gate":
        gate_weights = torch.full((batch_size, 4), 0.25, device=device, dtype=dtype)
        extra["gate_prior_enabled"] = False
    elif mode == "tri_modal_reliability_gate":
        gate_weights = heuristic_reliability_gate(evidence).to(dtype=dtype)
        extra["gate_prior_enabled"] = False
    elif mode == "tri_modal_confidence_gate":
        gate_weights = confidence_gate(
            torch.stack(
                [
                    diagnostics["api_confidence"],
                    diagnostics["graph_confidence"],
                    diagnostics["manifest_confidence"],
                    diagnostics["joint_confidence"],
                ],
                dim=-1,
            )
        ).to(dtype=dtype)
        extra["gate_prior_enabled"] = False
    else:
        gate_input = evidence.detach() if model.gate_detach else evidence
        gate_weights = model.gate_net(gate_input)
        if model.apply_alive_mask_to_learned_gate:
            gate_weights = apply_alive_mask(gate_weights, evidence).to(dtype=dtype)
        extra["gate_prior_enabled"] = True

    logits = (
        gate_weights[:, 0:1] * tensors["api_logits"]
        + gate_weights[:, 1:2] * tensors["graph_logits"]
        + gate_weights[:, 2:3] * tensors["manifest_logits"]
        + gate_weights[:, 3:4] * tensors["joint_logits"]
    )
    return logits, gate_weights, extra


def _fusion_discount_probability(model, batch_size, device, dtype, tensors, extra):
    del batch_size, device, dtype
    evidence, diagnostics = build_evidence(
        tensors["graph_data"],
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["joint_logits"],
        tensors["api_emb"],
        tensors["graph_emb"],
        tensors["manifest_emb"],
        use_consistency_evidence=model.use_consistency_evidence,
        use_conflict_evidence=model.use_conflict_evidence,
        use_perturbation_evidence=False,
    )
    extra.update(diagnostics)
    extra["gate_evidence"] = evidence.detach()
    fusion_outputs = model.discount_fusion(
        tensors["api_logits"],
        tensors["graph_logits"],
        tensors["manifest_logits"],
        tensors["joint_logits"],
        evidence,
    )
    extra.update(fusion_outputs)
    extra["gate_prior_enabled"] = False
    return fusion_outputs["final_logits"], fusion_outputs["fusion_weights"], extra


FUSION_DISPATCH: dict[str, callable] = {
    "api": _fusion_single_api,
    "api_only": _fusion_single_api,
    "graph": _fusion_single_graph,
    "graph_only": _fusion_single_graph,
    "manifest": _fusion_single_manifest,
    "manifest_only": _fusion_single_manifest,
    "api_graph": _fusion_api_graph_concat,
    "api_graph_concat": _fusion_api_graph_concat,
    "api_graph_manifest_concat": _fusion_tri_concat,
    "tri_modal_concat": _fusion_tri_concat,
    "tri_modal_fixed_gate": _fusion_evidence_based,
    "tri_modal_reliability_gate": _fusion_evidence_based,
    "tri_modal_confidence_gate": _fusion_evidence_based,
    "tri_modal_ours": _fusion_evidence_based,
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


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def sample_semantic_mask(batch_size: int, mask_prob: float, device) -> torch.Tensor:
    mask_prob = float(mask_prob)
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError(f"semantic mask probability must be within [0, 1], got {mask_prob}")
    return torch.rand(batch_size, device=device) < mask_prob


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
        self.max_seq_len = int(max_seq_len)
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
        if self.max_seq_len > 0:
            raw_lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
            offsets = torch.zeros((num_graphs + 1,), device=device, dtype=torch.long)
            offsets[1:] = raw_lengths.cumsum(dim=0)
            local_pos = torch.arange(api_batch.numel(), device=device) - offsets[api_batch]
            keep = local_pos < self.max_seq_len
            api_ids = api_ids[keep]
            api_batch = api_batch[keep]
            if api_ids.numel() == 0:
                return self._empty_output(num_graphs, device, dtype)
        else:
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
        fusion_mode: str = "tri_modal_ours",
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
        joint_emb_dim: int = 128,
        gate_hidden_dim: int = 128,
        gate_detach: bool = True,
        use_consistency_evidence: bool = True,
        use_conflict_evidence: bool = True,
        use_perturbation_evidence: bool = False,
        apply_alive_mask_to_learned_gate: bool = True,
        semantic_reconstruction_config: dict | None = None,
        semantic_cross_attention_config: dict | None = None,
        discount_fusion_config: dict | None = None,
    ):
        super().__init__()
        # Defensive check: ensure DEFAULT_API_TYPE_ID_TO_CATEGORY stays
        # consistent with the extractor taxonomy and the 12-D shared space.
        validate_api_type_mapping()
        fusion_mode = str(fusion_mode or "tri_modal_ours")
        if fusion_mode not in TRI_MODAL_FUSION_MODES:
            raise ValueError(f"Unsupported tri-modal fusion_mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.num_classes = int(num_classes)
        self.api_emb_dim = int(api_emb_dim)
        self.graph_emb_dim = int(graph_emb_dim)
        self.manifest_in_dim = int(manifest_in_dim)
        self.manifest_emb_dim = int(manifest_emb_dim)
        self.joint_emb_dim = int(joint_emb_dim)
        self.gate_detach = bool(gate_detach)
        self.use_consistency_evidence = bool(use_consistency_evidence)
        self.use_conflict_evidence = bool(use_conflict_evidence)
        if use_perturbation_evidence:
            raise ValueError(
                "Synthetic pert_* metadata is diagnostic-only and cannot be enabled in TriModalRobustModel."
            )
        self.use_perturbation_evidence = False
        self.apply_alive_mask_to_learned_gate = bool(apply_alive_mask_to_learned_gate)
        self.semantic_reconstruction_config = dict(semantic_reconstruction_config or {})
        self.semantic_reconstruction_enabled = bool(self.semantic_reconstruction_config.get("enabled", False))
        self.semantic_dim = int(self.semantic_reconstruction_config.get("semantic_dim", 128))
        if self.semantic_dim <= 0:
            raise ValueError("semantic_reconstruction.semantic_dim must be positive")
        self.semantic_cross_attention_config = dict(semantic_cross_attention_config or {})
        self.semantic_cross_attention_enabled = bool(
            self.semantic_cross_attention_config.get("enabled", False)
        )
        self.attach_cross_attention_to_joint = bool(
            self.semantic_cross_attention_config.get("attach_to_joint", True)
        )
        self.attach_cross_attention_to_reconstruction = bool(
            self.semantic_cross_attention_config.get("attach_to_reconstruction", True)
        )

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

        self.joint_encoder = nn.Sequential(
            nn.Linear(api_emb_dim + graph_emb_dim + manifest_emb_dim, max(joint_emb_dim * 2, 128)),
            nn.GELU(),
            nn.Dropout(ArchitectureConstants.HEAD_DROPOUT),
            nn.Linear(max(joint_emb_dim * 2, 128), joint_emb_dim),
            nn.LayerNorm(joint_emb_dim),
        )
        if self.semantic_cross_attention_enabled:
            cross_dim = int(self.semantic_cross_attention_config.get("dim", joint_emb_dim))
            if cross_dim <= 0:
                raise ValueError("semantic_cross_attention.dim must be positive")
            cross_security_tokens = int(
                self.semantic_cross_attention_config.get(
                    "num_security_tokens", SEMANTIC_CATEGORY_DIM
                )
            )
            if cross_security_tokens != SEMANTIC_CATEGORY_DIM:
                raise ValueError(
                    "semantic_cross_attention.num_security_tokens must match the "
                    f"current {SEMANTIC_CATEGORY_DIM}-D security taxonomy"
                )
            self.semantic_cross_attention_dim = cross_dim
            self.cross_attention_input_projections = nn.ModuleDict(
                {
                    "api": nn.Identity() if api_emb_dim == cross_dim else nn.Linear(api_emb_dim, cross_dim),
                    "graph": nn.Identity() if graph_emb_dim == cross_dim else nn.Linear(graph_emb_dim, cross_dim),
                    "manifest": (
                        nn.Identity()
                        if manifest_emb_dim == cross_dim
                        else nn.Linear(manifest_emb_dim, cross_dim)
                    ),
                }
            )
            self.semantic_cross_attention = ReliabilityAwareSemanticCrossAttention(
                dim=cross_dim,
                num_heads=int(self.semantic_cross_attention_config.get("num_heads", 4)),
                num_security_tokens=cross_security_tokens,
                num_residual_tokens=int(
                    self.semantic_cross_attention_config.get("num_residual_tokens", 4)
                ),
                dropout=float(self.semantic_cross_attention_config.get("dropout", 0.1)),
                residual_gate_init=float(
                    self.semantic_cross_attention_config.get("residual_gate_init", -3.0)
                ),
                use_reliability_bias=bool(
                    self.semantic_cross_attention_config.get("use_reliability_bias", True)
                ),
                use_support_bias=bool(
                    self.semantic_cross_attention_config.get("use_support_bias", True)
                ),
                use_conflict_bias=bool(
                    self.semantic_cross_attention_config.get("use_conflict_bias", True)
                ),
                use_relation_mask=bool(
                    self.semantic_cross_attention_config.get("use_relation_mask", True)
                ),
                use_semantic_presence_prior=bool(
                    self.semantic_cross_attention_config.get(
                        "use_semantic_presence_prior", True
                    )
                ),
            )
            self.cross_attention_joint_projection = (
                nn.Identity()
                if cross_dim == joint_emb_dim
                else nn.Linear(cross_dim, joint_emb_dim)
            )
            self.cross_attention_reconstruction_projections = nn.ModuleDict(
                {
                    "api": nn.Identity() if cross_dim == api_emb_dim else nn.Linear(cross_dim, api_emb_dim),
                    "graph": (
                        nn.Identity()
                        if cross_dim == graph_emb_dim
                        else nn.Linear(cross_dim, graph_emb_dim)
                    ),
                    "manifest": (
                        nn.Identity()
                        if cross_dim == manifest_emb_dim
                        else nn.Linear(cross_dim, manifest_emb_dim)
                    ),
                }
            )
        else:
            self.semantic_cross_attention_dim = 0

        self.api_head = build_main_head(api_emb_dim, num_classes)
        self.graph_head = build_main_head(graph_emb_dim, num_classes)
        self.manifest_head = build_main_head(manifest_emb_dim, num_classes)
        self.joint_head = build_main_head(joint_emb_dim, num_classes)
        self.api_semantic_head = nn.Linear(api_emb_dim, SEMANTIC_CATEGORY_DIM)
        self.graph_semantic_head = nn.Linear(graph_emb_dim, SEMANTIC_CATEGORY_DIM)
        self.manifest_semantic_head = nn.Linear(manifest_emb_dim, SEMANTIC_CATEGORY_DIM)
        self.api_graph_concat_head = build_main_head(api_emb_dim + graph_emb_dim, num_classes)
        self.tri_concat_head = build_main_head(api_emb_dim + graph_emb_dim + manifest_emb_dim, num_classes)
        self.gate_net = FourBranchEvidenceGate(
            evidence_dim=EvidenceIndex.BASE_DIM,
            hidden_dim=gate_hidden_dim,
        )
        if self.semantic_reconstruction_enabled:
            semantic_hidden = max(self.semantic_dim, 128)
            self.api_semantic_projector = build_mlp(api_emb_dim, semantic_hidden, self.semantic_dim)
            self.graph_semantic_projector = build_mlp(graph_emb_dim, semantic_hidden, self.semantic_dim)
            self.manifest_semantic_projector = build_mlp(manifest_emb_dim, semantic_hidden, self.semantic_dim)
            self.reconstruct_api_semantic = build_mlp(
                2 * self.semantic_dim, semantic_hidden, SEMANTIC_CATEGORY_DIM
            )
            self.reconstruct_graph_semantic = build_mlp(
                2 * self.semantic_dim, semantic_hidden, SEMANTIC_CATEGORY_DIM
            )
            self.reconstruct_manifest_semantic = build_mlp(
                2 * self.semantic_dim, semantic_hidden, SEMANTIC_CATEGORY_DIM
            )
        self.discount_fusion = DiscountProbabilityFusion(discount_fusion_config)

    def calibration_parameters(self) -> list[nn.Parameter]:
        """Parameters fitted after model selection without changing encoders."""
        return self.discount_fusion.calibration_parameters()

    def set_calibration_active(self, enabled: bool) -> None:
        self.discount_fusion.set_calibration_active(enabled)

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
        _, pooled, _ = self.api_encoder(graph_data, batch_size, device, dtype)
        return pooled

    def _semantic_reconstruction_outputs(
        self,
        api_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        manifest_emb: torch.Tensor,
        api_alive: torch.Tensor,
        graph_alive: torch.Tensor,
        manifest_alive: torch.Tensor,
        api_source_weight: torch.Tensor,
        graph_source_weight: torch.Tensor,
        manifest_source_weight: torch.Tensor,
        semantic_source_embeddings_by_target: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.semantic_reconstruction_enabled:
            return {}
        batch_size = api_emb.size(0)
        device = api_emb.device
        original_embeddings = {
            "api": api_emb,
            "graph": graph_emb,
            "manifest": manifest_emb,
        }
        projectors = {
            "api": self.api_semantic_projector,
            "graph": self.graph_semantic_projector,
            "manifest": self.manifest_semantic_projector,
        }
        base_z = {
            name: F.normalize(projectors[name](embedding), dim=-1)
            for name, embedding in original_embeddings.items()
        }
        semantic_source_embeddings_by_target = semantic_source_embeddings_by_target or {}

        def source_z(target_name: str, source_name: str) -> torch.Tensor:
            target_sources = semantic_source_embeddings_by_target.get(target_name, {})
            embedding = target_sources.get(source_name, original_embeddings[source_name])
            return F.normalize(projectors[source_name](embedding), dim=-1)

        if self.training and self.semantic_reconstruction_enabled:
            mask_api = sample_semantic_mask(
                batch_size, self.semantic_reconstruction_config.get("mask_prob_api", 0.15), device
            )
            mask_graph = sample_semantic_mask(
                batch_size, self.semantic_reconstruction_config.get("mask_prob_graph", 0.15), device
            )
            mask_manifest = sample_semantic_mask(
                batch_size, self.semantic_reconstruction_config.get("mask_prob_manifest", 0.15), device
            )
        else:
            mask_api = torch.zeros(batch_size, device=device, dtype=torch.bool)
            mask_graph = torch.zeros(batch_size, device=device, dtype=torch.bool)
            mask_manifest = torch.zeros(batch_size, device=device, dtype=torch.bool)

        def masked_input(
            z: torch.Tensor,
            alive: torch.Tensor,
            source_weight: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            available = alive.view(batch_size, 1).bool() & ~mask.view(batch_size, 1)
            weighted = z * source_weight.view(batch_size, 1).clamp(0.0, 1.0)
            return torch.where(available, weighted, torch.zeros_like(z))

        alive = {"api": api_alive, "graph": graph_alive, "manifest": manifest_alive}
        source_weight = {
            "api": api_source_weight,
            "graph": graph_source_weight,
            "manifest": manifest_source_weight,
        }
        masks = {"api": mask_api, "graph": mask_graph, "manifest": mask_manifest}

        def reconstruction_input(target_name: str, source_name: str) -> torch.Tensor:
            return masked_input(
                source_z(target_name, source_name),
                alive[source_name],
                source_weight[source_name],
                masks[source_name],
            )

        return {
            "z_api": base_z["api"],
            "z_graph": base_z["graph"],
            "z_manifest": base_z["manifest"],
            "recon_api_semantic_logits": self.reconstruct_api_semantic(
                torch.cat(
                    [
                        reconstruction_input("api", "graph"),
                        reconstruction_input("api", "manifest"),
                    ],
                    dim=-1,
                )
            ),
            "recon_graph_semantic_logits": self.reconstruct_graph_semantic(
                torch.cat(
                    [
                        reconstruction_input("graph", "api"),
                        reconstruction_input("graph", "manifest"),
                    ],
                    dim=-1,
                )
            ),
            "recon_manifest_semantic_logits": self.reconstruct_manifest_semantic(
                torch.cat(
                    [
                        reconstruction_input("manifest", "api"),
                        reconstruction_input("manifest", "graph"),
                    ],
                    dim=-1,
                )
            ),
            "mask_api_semantic": mask_api.to(dtype=api_emb.dtype),
            "mask_graph_semantic": mask_graph.to(dtype=api_emb.dtype),
            "mask_manifest_semantic": mask_manifest.to(dtype=api_emb.dtype),
            "semantic_source_weight_api": api_source_weight.view(-1),
            "semantic_source_weight_graph": graph_source_weight.view(-1),
            "semantic_source_weight_manifest": manifest_source_weight.view(-1),
        }

    @staticmethod
    def _availability_mask(
        graph_data,
        name: str,
        legacy_quality_name: str,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if not isinstance(value, torch.Tensor):
            value = getattr(graph_data, legacy_quality_name, None)
        if not isinstance(value, torch.Tensor):
            return torch.ones((batch_size, 1), device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).view(batch_size, -1)
        if value.size(1) == 0:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        return value[:, :1].gt(0.0).to(dtype=dtype)

    @staticmethod
    def _observable_weight(
        graph_data,
        name: str,
        legacy_quality_name: str,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if not isinstance(value, torch.Tensor):
            value = getattr(graph_data, legacy_quality_name, None)
        if not isinstance(value, torch.Tensor):
            return torch.ones((batch_size, 1), device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).view(batch_size, -1)
        if value.size(1) == 0:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        return torch.nan_to_num(
            value[:, :1].clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0
        )

    @staticmethod
    def _semantic_presence(
        graph_data,
        batch_size: int,
        device,
        dtype,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, attribute in (
            ("api", "api_semantic_category_counts"),
            ("graph", "graph_semantic_category_counts"),
            ("manifest", "manifest_category_counts"),
        ):
            value = getattr(graph_data, attribute, None)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device=device, dtype=dtype)
            value = value.view(1, -1).expand(batch_size, -1) if value.ndim == 1 else value.view(batch_size, -1)
            if value.size(1) == SEMANTIC_CATEGORY_DIM:
                out[name] = value.gt(0.0).to(dtype=dtype)
        return out

    def forward(
        self,
        graph_data,
        return_features: bool = False,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        batch_size = int(getattr(graph_data, "num_graphs", 1))

        api_emb = self._encode_api(graph_data, batch_size, device, dtype)
        _, graph_emb, _, _ = self.graph_encoder(graph_data)
        manifest_x = self._manifest_input(graph_data, batch_size, device, dtype)
        manifest_emb = self.manifest_encoder(manifest_x)

        api_joint_mask = self._availability_mask(
            graph_data, "api_alive", "q_api", batch_size, device, dtype
        )
        graph_joint_mask = self._availability_mask(
            graph_data, "graph_alive", "q_graph", batch_size, device, dtype
        )
        manifest_joint_mask = self._availability_mask(
            graph_data, "manifest_alive", "q_manifest", batch_size, device, dtype
        )
        api_semantic_weight = self._observable_weight(
            graph_data, "api_integrity", "q_api", batch_size, device, dtype
        )
        graph_semantic_weight = self._observable_weight(
            graph_data, "graph_integrity", "q_graph", batch_size, device, dtype
        )
        manifest_semantic_weight = self._observable_weight(
            graph_data, "manifest_integrity", "q_manifest", batch_size, device, dtype
        )
        api_emb_for_joint = api_emb * api_joint_mask
        graph_emb_for_joint = graph_emb * graph_joint_mask
        manifest_emb_for_joint = manifest_emb * manifest_joint_mask

        joint_input = torch.cat([api_emb_for_joint, graph_emb_for_joint, manifest_emb_for_joint], dim=-1)
        api_logits = self.api_head(api_emb)
        graph_logits = self.graph_head(graph_emb)
        manifest_logits = self.manifest_head(manifest_emb)
        base_joint_emb = self.joint_encoder(joint_input)
        base_joint_logits = self.joint_head(base_joint_emb)

        cross_attention_outputs: dict[str, torch.Tensor] = {}
        semantic_source_embeddings_by_target: dict[str, dict[str, torch.Tensor]] | None = None
        joint_emb = base_joint_emb
        if self.semantic_cross_attention_enabled:
            observable_evidence, _ = build_evidence(
                graph_data,
                api_logits,
                graph_logits,
                manifest_logits,
                base_joint_logits,
                api_emb,
                graph_emb,
                manifest_emb,
                use_consistency_evidence=self.use_consistency_evidence,
                use_conflict_evidence=self.use_conflict_evidence,
                use_perturbation_evidence=False,
            )
            cross_attention_outputs = self.semantic_cross_attention(
                self.cross_attention_input_projections["api"](api_emb),
                self.cross_attention_input_projections["graph"](graph_emb),
                self.cross_attention_input_projections["manifest"](manifest_emb),
                observable_evidence,
                semantic_presence=self._semantic_presence(
                    graph_data, batch_size, device, dtype
                ),
            )
            if self.attach_cross_attention_to_joint:
                joint_emb = self.cross_attention_joint_projection(
                    cross_attention_outputs["enhanced_joint"]
                )
            if self.attach_cross_attention_to_reconstruction:
                semantic_source_embeddings_by_target = {}
                modality_names = ("api", "graph", "manifest")
                for target_name in modality_names:
                    target_excluded = cross_attention_outputs[
                        f"enhanced_semantic_excluding_{target_name}"
                    ]
                    semantic_source_embeddings_by_target[target_name] = {
                        source_name: self.cross_attention_reconstruction_projections[
                            source_name
                        ](target_excluded[:, source_index])
                        for source_index, source_name in enumerate(modality_names)
                        if source_name != target_name
                    }
        joint_logits = self.joint_head(joint_emb)

        extra = {
            "api_logits_aux": api_logits,
            "graph_logits_aux": graph_logits,
            "manifest_logits_aux": manifest_logits,
            "joint_logits_aux": joint_logits,
            "api_semantic_logits": self.api_semantic_head(api_emb),
            "graph_semantic_logits": self.graph_semantic_head(graph_emb),
            "manifest_semantic_logits": self.manifest_semantic_head(manifest_emb),
            "semantic_cross_attention_enabled": torch.full(
                (batch_size,),
                float(self.semantic_cross_attention_enabled),
                device=device,
                dtype=dtype,
            ),
        }
        extra.update(cross_attention_outputs)
        extra.update(
            self._semantic_reconstruction_outputs(
                api_emb,
                graph_emb,
                manifest_emb,
                api_joint_mask,
                graph_joint_mask,
                manifest_joint_mask,
                api_semantic_weight,
                graph_semantic_weight,
                manifest_semantic_weight,
                semantic_source_embeddings_by_target=semantic_source_embeddings_by_target,
            )
        )

        handler = FUSION_DISPATCH[self.fusion_mode]
        logits, gate_weights, extra = handler(
            self,
            batch_size,
            device,
            dtype,
            {
                "api_logits": api_logits,
                "graph_logits": graph_logits,
                "manifest_logits": manifest_logits,
                "joint_logits": joint_logits,
                "api_emb": api_emb,
                "graph_emb": graph_emb,
                "manifest_emb": manifest_emb,
                "api_emb_for_joint": api_emb_for_joint,
                "graph_emb_for_joint": graph_emb_for_joint,
                "manifest_emb_for_joint": manifest_emb_for_joint,
                "joint_input": joint_input,
                "graph_data": graph_data,
            },
            extra,
        )

        extra["gate_weights_train"] = gate_weights
        extra["gate_weights"] = gate_weights.detach()
        extra.setdefault("gate_prior_enabled", False)

        if "api_confidence" not in extra and not self.training:
            _, diagnostics = build_evidence(
                graph_data,
                api_logits,
                graph_logits,
                manifest_logits,
                joint_logits,
                api_emb,
                graph_emb,
                manifest_emb,
                use_consistency_evidence=self.use_consistency_evidence,
                use_conflict_evidence=self.use_conflict_evidence,
                use_perturbation_evidence=self.use_perturbation_evidence,
                diagnostics_only=True,
            )
            extra.update(diagnostics)

        if return_features:
            extra["api_emb"] = api_emb
            extra["graph_emb"] = graph_emb
            extra["manifest_emb"] = manifest_emb
            extra["api_emb_for_joint"] = api_emb_for_joint
            extra["graph_emb_for_joint"] = graph_emb_for_joint
            extra["manifest_emb_for_joint"] = manifest_emb_for_joint
            extra["joint_emb"] = joint_emb
            extra["base_joint_emb"] = base_joint_emb

        return logits, extra
