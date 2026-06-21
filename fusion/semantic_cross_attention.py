from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex, GateConstants


MODALITY_NAMES = ("api", "graph", "manifest")


def _column(evidence: torch.Tensor, index: int) -> torch.Tensor:
    return evidence[:, index].clamp(0.0, 1.0)


def _presence(
    semantic_presence: dict[str, torch.Tensor] | None,
    name: str,
    *,
    batch_size: int,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if not semantic_presence:
        return None
    value = semantic_presence.get(name)
    if not isinstance(value, torch.Tensor):
        return None
    value = value.to(device=reference.device, dtype=reference.dtype)
    value = value.view(1, -1).expand(batch_size, -1) if value.ndim == 1 else value.view(batch_size, -1)
    if value.size(1) != width:
        return None
    return value.gt(0.0).to(dtype=reference.dtype)


def build_semantic_reliability_priors(
    evidence: torch.Tensor,
    num_security_tokens: int,
    num_residual_tokens: int,
    semantic_presence: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build observable token priors and modality-relation matrices.

    These priors deliberately use only evidence available during representation
    training. Post-hoc calibrated branch reliability is not used here.
    """
    if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError(
            f"Expected [B, >= {EvidenceIndex.BASE_DIM}] evidence, got {tuple(evidence.shape)}"
        )
    num_security_tokens = int(num_security_tokens)
    num_residual_tokens = int(num_residual_tokens)
    if num_security_tokens <= 0:
        raise ValueError("num_security_tokens must be positive")
    if num_residual_tokens < 0:
        raise ValueError("num_residual_tokens must be non-negative")

    api_integrity = _column(evidence, EvidenceIndex.API_INTEGRITY)
    graph_integrity = _column(evidence, EvidenceIndex.GRAPH_INTEGRITY)
    manifest_integrity = _column(evidence, EvidenceIndex.MANIFEST_INTEGRITY)
    code_integrity = _column(evidence, EvidenceIndex.CODE_INTEGRITY)
    anchor_support = _column(evidence, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT)
    manifest_support = _column(evidence, EvidenceIndex.MANIFEST_CODE_SUPPORT)
    manifest_conflict = _column(evidence, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT)
    code_conflict = _column(evidence, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT)
    api_alive = _column(evidence, EvidenceIndex.API_ALIVE).bool()
    graph_alive = _column(evidence, EvidenceIndex.GRAPH_ALIVE).bool()
    manifest_alive = _column(evidence, EvidenceIndex.MANIFEST_ALIVE).bool()

    api_graph_applicable = api_alive & graph_alive
    manifest_code_relation_observed = (
        (manifest_support > 0.0)
        | (manifest_conflict > 0.0)
        | (code_conflict > 0.0)
    )
    api_manifest_applicable = (
        api_alive & manifest_alive & manifest_code_relation_observed
    )
    graph_manifest_applicable = (
        graph_alive & manifest_alive & manifest_code_relation_observed
    )
    manifest_code_applicable = api_manifest_applicable | graph_manifest_applicable

    zero = torch.zeros_like(anchor_support)
    # Modality priors describe source quality only. A missing code counterpart
    # must not reduce the surviving branch through geometric-mean code quality.
    api_code_context = torch.where(graph_alive, code_integrity, api_integrity)
    graph_code_context = torch.where(api_alive, code_integrity, graph_integrity)
    r_api = api_alive.to(evidence.dtype) * torch.stack(
        [
            api_integrity,
            api_code_context,
        ],
        dim=-1,
    ).mean(dim=-1)
    r_graph = graph_alive.to(evidence.dtype) * torch.stack(
        [
            graph_integrity,
            graph_code_context,
        ],
        dim=-1,
    ).mean(dim=-1)
    r_manifest = manifest_alive.to(evidence.dtype) * manifest_integrity
    modality_reliability = torch.stack(
        [r_api, r_graph, r_manifest], dim=1
    ).clamp(0.0, 1.0)
    modality_alive = torch.stack(
        [api_alive, graph_alive, manifest_alive], dim=1
    )

    security_priors = []
    for index, name in enumerate(MODALITY_NAMES):
        base = modality_reliability[:, index : index + 1].expand(
            -1, num_security_tokens
        )
        presence = _presence(
            semantic_presence,
            name,
            batch_size=evidence.size(0),
            width=num_security_tokens,
            reference=evidence,
        )
        security_priors.append(
            base if presence is None else base * (0.5 + 0.5 * presence)
        )
    security_prior = torch.stack(security_priors, dim=1)
    residual_prior = modality_reliability.unsqueeze(-1).expand(
        -1, -1, num_residual_tokens
    )
    semantic_reliability_prior = torch.cat(
        [security_prior, residual_prior], dim=-1
    ).clamp(0.0, 1.0)

    support_matrix = evidence.new_zeros((evidence.size(0), 3, 3))
    conflict_matrix = evidence.new_zeros((evidence.size(0), 3, 3))
    applicable_matrix = torch.zeros(
        (evidence.size(0), 3, 3), device=evidence.device, dtype=torch.bool
    )
    for index in range(3):
        support_matrix[:, index, index] = 1.0
        applicable_matrix[:, index, index] = modality_alive[:, index]

    support_matrix[:, 0, 1] = support_matrix[:, 1, 0] = torch.where(
        api_graph_applicable, anchor_support, zero
    )
    applicable_matrix[:, 0, 1] = applicable_matrix[:, 1, 0] = api_graph_applicable

    manifest_code_conflict = torch.maximum(manifest_conflict, code_conflict)
    for code_index, pair_applicable in (
        (0, api_manifest_applicable),
        (1, graph_manifest_applicable),
    ):
        pair_support = torch.where(pair_applicable, manifest_support, zero)
        pair_conflict = torch.where(pair_applicable, manifest_code_conflict, zero)
        support_matrix[:, code_index, 2] = pair_support
        support_matrix[:, 2, code_index] = pair_support
        conflict_matrix[:, code_index, 2] = pair_conflict
        conflict_matrix[:, 2, code_index] = pair_conflict
        applicable_matrix[:, code_index, 2] = pair_applicable
        applicable_matrix[:, 2, code_index] = pair_applicable

    return {
        "semantic_reliability_prior": semantic_reliability_prior,
        "modality_reliability_prior": modality_reliability,
        "modality_alive": modality_alive,
        "semantic_support_matrix": support_matrix,
        "semantic_conflict_matrix": conflict_matrix,
        "semantic_relation_applicable": applicable_matrix,
        "api_graph_relation_applicable": api_graph_applicable,
        "api_manifest_relation_applicable": api_manifest_applicable,
        "graph_manifest_relation_applicable": graph_manifest_applicable,
        "manifest_code_relation_applicable": manifest_code_applicable,
    }


class ReliabilityAwareSemanticCrossAttention(nn.Module):
    """Cross-modal semantic-token interaction controlled by observable priors."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_security_tokens: int = 12,
        num_residual_tokens: int = 4,
        dropout: float = 0.1,
        residual_gate_init: float = 0.0,
        use_reliability_bias: bool = True,
        use_support_bias: bool = True,
        use_conflict_bias: bool = True,
        use_relation_mask: bool = True,
        use_semantic_presence_prior: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.num_security_tokens = int(num_security_tokens)
        self.num_residual_tokens = int(num_residual_tokens)
        self.num_tokens = self.num_security_tokens + self.num_residual_tokens
        if self.dim <= 0 or self.num_heads <= 0 or self.dim % self.num_heads != 0:
            raise ValueError("semantic cross-attention dim must be positive and divisible by num_heads")
        if self.num_security_tokens <= 0 or self.num_residual_tokens < 0:
            raise ValueError("semantic token counts must be positive/non-negative")
        if self.num_tokens <= 0:
            raise ValueError("semantic cross-attention requires at least one token")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("semantic cross-attention dropout must be within [0, 1)")

        self.use_reliability_bias = bool(use_reliability_bias)
        self.use_support_bias = bool(use_support_bias)
        self.use_conflict_bias = bool(use_conflict_bias)
        self.use_relation_mask = bool(use_relation_mask)
        self.use_semantic_presence_prior = bool(use_semantic_presence_prior)
        self.head_dim = self.dim // self.num_heads

        self.modality_projections = nn.ModuleList(
            [nn.Linear(self.dim, self.dim) for _ in MODALITY_NAMES]
        )
        self.security_token_embed = nn.Parameter(
            torch.empty(3, self.num_security_tokens, self.dim)
        )
        self.residual_token_embed = nn.Parameter(
            torch.empty(3, self.num_residual_tokens, self.dim)
        )
        nn.init.normal_(self.security_token_embed, std=0.02)
        if self.num_residual_tokens > 0:
            nn.init.normal_(self.residual_token_embed, std=0.02)

        self.token_norm = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.output_norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(float(dropout))
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim),
        )

        self.alpha_reliability_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_support_raw = nn.Parameter(torch.tensor(0.0))
        self.gamma_conflict_raw = nn.Parameter(torch.tensor(0.0))
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def _tokens(
        self,
        api_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        manifest_emb: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = (api_emb, graph_emb, manifest_emb)
        if any(value.ndim != 2 or value.size(-1) != self.dim for value in embeddings):
            shapes = [tuple(value.shape) for value in embeddings]
            raise ValueError(
                f"semantic cross-attention expects three [B, {self.dim}] embeddings, got {shapes}"
            )
        if len({value.size(0) for value in embeddings}) != 1:
            raise ValueError("semantic cross-attention embeddings must share batch size")

        learned = torch.cat(
            [self.security_token_embed, self.residual_token_embed], dim=1
        )
        tokens = torch.stack(
            [
                projection(embedding).unsqueeze(1) + learned[index].unsqueeze(0)
                for index, (projection, embedding) in enumerate(
                    zip(self.modality_projections, embeddings)
                )
            ],
            dim=1,
        )
        return self.token_norm(tokens)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(
            value.size(0), value.size(1), self.num_heads, self.head_dim
        ).transpose(1, 2)

    @staticmethod
    def _masked_softmax(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = score.masked_fill(~mask, -1.0e4)
        attention = torch.softmax(score, dim=-1) * mask.to(dtype=score.dtype)
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            GateConstants.EPS
        )

    def forward(
        self,
        api_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        manifest_emb: torch.Tensor,
        evidence: torch.Tensor,
        semantic_presence: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self._tokens(api_emb, graph_emb, manifest_emb)
        priors = build_semantic_reliability_priors(
            evidence,
            self.num_security_tokens,
            self.num_residual_tokens,
            semantic_presence=semantic_presence if self.use_semantic_presence_prior else None,
        )
        flat_tokens = tokens.view(tokens.size(0), 3 * self.num_tokens, self.dim)
        q = self._split_heads(self.q_proj(flat_tokens))
        k = self._split_heads(self.k_proj(flat_tokens))
        v = self._split_heads(self.v_proj(flat_tokens))
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        token_reliability = priors["semantic_reliability_prior"].view(
            tokens.size(0), 3 * self.num_tokens
        )
        support_token_matrix = priors["semantic_support_matrix"].repeat_interleave(
            self.num_tokens, dim=1
        ).repeat_interleave(self.num_tokens, dim=2)
        conflict_token_matrix = priors["semantic_conflict_matrix"].repeat_interleave(
            self.num_tokens, dim=1
        ).repeat_interleave(self.num_tokens, dim=2)
        relation_token_matrix = priors["semantic_relation_applicable"].repeat_interleave(
            self.num_tokens, dim=1
        ).repeat_interleave(self.num_tokens, dim=2)
        source_alive = priors["modality_alive"].repeat_interleave(
            self.num_tokens, dim=1
        )

        if self.use_reliability_bias:
            score = score + F.softplus(self.alpha_reliability_raw) * torch.log(
                token_reliability.clamp_min(GateConstants.EPS)
            ).unsqueeze(1).unsqueeze(1)
        if self.use_support_bias:
            score = score + F.softplus(self.beta_support_raw) * support_token_matrix.unsqueeze(1)
        if self.use_conflict_bias:
            score = score - F.softplus(self.gamma_conflict_raw) * conflict_token_matrix.unsqueeze(1)

        # Unavailable sources never propagate. Relation masking additionally
        # blocks unobserved cross-modal relations without treating them as conflict.
        attention_mask = source_alive.unsqueeze(1).unsqueeze(1).expand_as(score)
        if self.use_relation_mask:
            attention_mask = attention_mask & relation_token_matrix.unsqueeze(1)
        attention = self._masked_softmax(score, attention_mask)

        def enhance(attention_weights: torch.Tensor) -> torch.Tensor:
            attended_value = torch.matmul(attention_weights, v).transpose(1, 2).contiguous()
            attended_value = attended_value.view(
                tokens.size(0), 3 * self.num_tokens, self.dim
            )
            attended_value = self.out_proj(attended_value).view_as(tokens)
            return self.output_norm(
                tokens
                + torch.sigmoid(self.residual_gate) * self.dropout(attended_value)
            )

        enhanced = enhance(attention)
        excluded_source_outputs: dict[str, torch.Tensor] = {}
        for excluded_index, excluded_name in enumerate(MODALITY_NAMES):
            keep_source = torch.ones(
                (3 * self.num_tokens,),
                device=tokens.device,
                dtype=attention.dtype,
            )
            start = excluded_index * self.num_tokens
            keep_source[start : start + self.num_tokens] = 0.0
            excluded_attention = attention * keep_source.view(1, 1, 1, -1)
            excluded_attention = excluded_attention / excluded_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(GateConstants.EPS)
            excluded_source_outputs[
                f"enhanced_semantic_excluding_{excluded_name}"
            ] = enhance(excluded_attention).mean(dim=2)

        modality_alive = priors["modality_alive"].to(dtype=enhanced.dtype)
        token_alive = modality_alive.unsqueeze(-1).expand(-1, -1, self.num_tokens)
        pooled = (enhanced * token_alive.unsqueeze(-1)).sum(dim=(1, 2))
        denominator = token_alive.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        pooled = torch.where(
            denominator > 0,
            pooled / denominator.clamp_min(1.0),
            torch.zeros_like(pooled),
        )
        enhanced_joint = torch.where(
            denominator > 0,
            self.joint_projection(pooled),
            torch.zeros_like(pooled),
        )

        attention_entropy = -(
            attention.clamp_min(GateConstants.EPS)
            * torch.log(attention.clamp_min(GateConstants.EPS))
        ).sum(dim=-1)
        target_alive = priors["modality_alive"].repeat_interleave(
            self.num_tokens, dim=1
        )
        target_alive_float = target_alive.to(dtype=attention.dtype)
        target_row_mask = target_alive_float.unsqueeze(1)

        def alive_target_mean(
            value: torch.Tensor,
            extra_denominator: int = 1,
        ) -> torch.Tensor:
            numerator = (value * target_row_mask.unsqueeze(-1)).sum(
                dim=tuple(range(1, value.ndim))
            )
            denominator = (
                target_alive_float.sum(dim=-1)
                * self.num_heads
                * int(extra_denominator)
            )
            return torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(1.0),
                torch.zeros_like(numerator),
            )

        mean_attention_entropy = alive_target_mean(
            attention_entropy.unsqueeze(-1)
        )
        cross_modal_mask = (
            ~torch.eye(3, device=tokens.device, dtype=torch.bool)
        ).repeat_interleave(self.num_tokens, dim=0).repeat_interleave(
            self.num_tokens, dim=1
        )
        cross_modal_attention = alive_target_mean(
            attention * cross_modal_mask.view(1, 1, 3 * self.num_tokens, -1),
            extra_denominator=2 * self.num_tokens,
        )

        outputs: dict[str, torch.Tensor] = {
            "base_semantic_tokens": tokens,
            "enhanced_tokens": enhanced,
            "enhanced_api_semantic": enhanced[:, 0].mean(dim=1),
            "enhanced_graph_semantic": enhanced[:, 1].mean(dim=1),
            "enhanced_manifest_semantic": enhanced[:, 2].mean(dim=1),
            "enhanced_joint": enhanced_joint,
            "semantic_attention": attention,
            "semantic_reliability_prior": priors["semantic_reliability_prior"],
            "semantic_relation_applicable": priors["semantic_relation_applicable"],
            "semantic_support_matrix": priors["semantic_support_matrix"],
            "semantic_conflict_matrix": priors["semantic_conflict_matrix"],
            "mean_semantic_attention_entropy": mean_attention_entropy,
            "mean_cross_modal_attention": cross_modal_attention,
            "semantic_residual_gate": torch.sigmoid(self.residual_gate).view(1).expand(
                tokens.size(0)
            ),
        }
        for index, name in enumerate(MODALITY_NAMES):
            source_slice = slice(
                index * self.num_tokens, (index + 1) * self.num_tokens
            )
            outputs[f"mean_semantic_reliability_prior_{name}"] = priors[
                "semantic_reliability_prior"
            ][:, index].mean(dim=-1)
            outputs[f"mean_semantic_attention_to_{name}"] = alive_target_mean(
                attention[..., source_slice],
                extra_denominator=self.num_tokens,
            )
        outputs.update(excluded_source_outputs)
        outputs.update(
            {
                key: value
                for key, value in priors.items()
                if key
                in {
                    "modality_reliability_prior",
                    "modality_alive",
                    "api_graph_relation_applicable",
                    "api_manifest_relation_applicable",
                    "graph_manifest_relation_applicable",
                    "manifest_code_relation_applicable",
                }
            }
        )
        return outputs
