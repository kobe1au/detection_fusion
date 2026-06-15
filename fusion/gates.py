from __future__ import annotations

import torch
import torch.nn as nn

from fusion.constants import ArchitectureConstants, EvidenceIndex, GateConstants


class FourBranchEvidenceGate(nn.Module):
    """Evidence-only four-way gate for API, Graph, Manifest, and Joint branches."""

    def __init__(self, evidence_dim: int = EvidenceIndex.BASE_DIM, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or ArchitectureConstants.GATE_HIDDEN_DIM
        self.evidence_dim = int(evidence_dim)
        self.net = nn.Sequential(
            nn.Linear(self.evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, GateConstants.NUM_BRANCHES),
        )
        nn.init.constant_(self.net[-1].bias, ArchitectureConstants.GATE_INIT_BIAS)
        with torch.no_grad():
            self.net[-1].bias[3] = ArchitectureConstants.GATE_JOINT_INIT_BIAS

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.size(-1) != self.evidence_dim:
            raise ValueError(
                f"FourBranchEvidenceGate expected {self.evidence_dim} features, got {evidence.size(-1)}"
            )
        return torch.softmax(self.net(evidence), dim=-1)


def heuristic_reliability_gate(evidence: torch.Tensor) -> torch.Tensor:
    if evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError(
            f"heuristic_reliability_gate expected ≥{EvidenceIndex.BASE_DIM} evidence dims, "
            f"got {evidence.size(-1)}"
        )
    r_api = evidence[:, EvidenceIndex.R_API : EvidenceIndex.R_API + 1]
    r_graph = evidence[:, EvidenceIndex.R_GRAPH : EvidenceIndex.R_GRAPH + 1]
    r_manifest = evidence[:, EvidenceIndex.R_MANIFEST : EvidenceIndex.R_MANIFEST + 1]

    api_graph = evidence[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT : EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT + 1]
    manifest_code = evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT : EvidenceIndex.MANIFEST_CODE_SUPPORT + 1]

    api_alive = evidence[:, EvidenceIndex.API_ALIVE : EvidenceIndex.API_ALIVE + 1]
    graph_alive = evidence[:, EvidenceIndex.GRAPH_ALIVE : EvidenceIndex.GRAPH_ALIVE + 1]
    manifest_alive = evidence[:, EvidenceIndex.MANIFEST_ALIVE : EvidenceIndex.MANIFEST_ALIVE + 1]

    alive_sum = (api_alive + graph_alive + manifest_alive).clamp_min(1.0)
    reliability_support = (
        r_api * api_alive + r_graph * graph_alive + r_manifest * manifest_alive
    ) / alive_sum

    code_alive = torch.maximum(api_alive, graph_alive)
    pair_support = api_alive * graph_alive + code_alive * manifest_alive
    pair_consistency = (
        api_graph * api_alive * graph_alive
        + manifest_code * code_alive * manifest_alive
    ) / pair_support.clamp_min(1.0)

    joint_score = (reliability_support.square() * (0.5 + 0.5 * pair_consistency)).clamp(0.0, 1.0)
    joint_availability = (alive_sum / 3.0).clamp(0.0, 1.0)

    scores = torch.cat(
        [
            r_api * api_alive,
            r_graph * graph_alive,
            r_manifest * manifest_alive,
            joint_score * joint_availability,
        ],
        dim=-1,
    )
    denom = scores.sum(dim=-1, keepdim=True)
    normalized = scores / denom.clamp_min(GateConstants.EPS)
    fallback = torch.full_like(scores, GateConstants.UNIFORM_BRANCH_WEIGHT)
    return torch.where(denom > GateConstants.EPS, normalized, fallback)


def confidence_gate(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.size(-1) != GateConstants.NUM_BRANCHES:
        raise ValueError(f"confidence_gate expected [B, 4] confidence scores, got {tuple(scores.shape)}")
    scores = scores.clamp(0.0, 1.0)

    denom = scores.sum(dim=-1, keepdim=True)
    normalized = scores / denom.clamp_min(GateConstants.EPS)
    fallback = torch.full_like(scores, GateConstants.UNIFORM_BRANCH_WEIGHT)
    return torch.where(denom > GateConstants.EPS, normalized, fallback)


def apply_alive_mask(gate_weights: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    api_alive = evidence[:, EvidenceIndex.API_ALIVE : EvidenceIndex.API_ALIVE + 1].clamp(0.0, 1.0)
    graph_alive = evidence[:, EvidenceIndex.GRAPH_ALIVE : EvidenceIndex.GRAPH_ALIVE + 1].clamp(0.0, 1.0)
    manifest_alive = evidence[:, EvidenceIndex.MANIFEST_ALIVE : EvidenceIndex.MANIFEST_ALIVE + 1].clamp(0.0, 1.0)

    joint_alive = torch.maximum(torch.maximum(api_alive, graph_alive), manifest_alive)
    support = torch.cat([api_alive, graph_alive, manifest_alive, joint_alive], dim=-1)

    masked = gate_weights * support
    denom = masked.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(masked, GateConstants.UNIFORM_BRANCH_WEIGHT)
    return torch.where(denom > GateConstants.EPS, masked / denom.clamp_min(GateConstants.EPS), fallback)
