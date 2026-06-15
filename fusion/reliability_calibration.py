from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex


BRANCH_NAMES = ("api", "graph", "manifest", "joint")


def _column(evidence: torch.Tensor, index: int) -> torch.Tensor:
    return evidence[:, index].clamp(0.0, 1.0)


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights for monotonic mappings."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.raw_weight, mean=-2.0, std=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.raw_weight), self.bias)


class MonotonicBranchCalibrator(nn.Module):
    """Estimate branch correctness probability from all-positive evidence."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            PositiveLinear(input_dim, hidden_dim),
            nn.Softplus(),
            PositiveLinear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).view(-1)


def build_monotonic_reliability_features(
    evidence: torch.Tensor,
    *,
    missing_relation_support: float = 0.0,
    neutral_support: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build branch-specific features where larger always means more reliable."""
    if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError(
            f"Expected [B, >= {EvidenceIndex.BASE_DIM}] evidence, got {tuple(evidence.shape)}"
        )

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
    code_alive = api_alive | graph_alive
    joint_alive = code_alive | manifest_alive
    api_graph_applicable = api_alive & graph_alive
    manifest_code_relation_observed = (
        (manifest_support > 0.0)
        | (manifest_conflict > 0.0)
        | (code_conflict > 0.0)
    )
    manifest_code_applicable = manifest_alive & code_alive & manifest_code_relation_observed

    if neutral_support is not None:
        # Backward-compatible alias for older experiment configs/callers.
        missing_relation_support = float(neutral_support)
    missing_support = torch.full_like(anchor_support, float(missing_relation_support))
    # Unavailable relations contribute no positive evidence. Applicability is
    # kept as a diagnostic mask rather than treating missing counterparts as
    # either maximum support or maximum conflict.
    anchor_good = torch.where(api_graph_applicable, anchor_support, missing_support)
    manifest_support_good = torch.where(
        manifest_code_applicable, manifest_support, missing_support
    )
    manifest_conflict_good = torch.where(
        manifest_code_applicable, 1.0 - manifest_conflict, missing_support
    )
    code_conflict_good = torch.where(
        manifest_code_applicable, 1.0 - code_conflict, missing_support
    )

    alive_float = torch.stack([api_alive, graph_alive, manifest_alive], dim=-1).float()
    integrities = torch.stack(
        [api_integrity, graph_integrity, manifest_integrity], dim=-1
    )
    mean_alive_integrity = (
        integrities * alive_float
    ).sum(dim=-1) / alive_float.sum(dim=-1).clamp_min(1.0)
    alive_fraction = alive_float.mean(dim=-1)
    pair_values = torch.stack([anchor_good, manifest_support_good], dim=-1)
    conflict_good = torch.minimum(manifest_conflict_good, code_conflict_good)

    features = {
        "api": torch.stack(
            [
                api_integrity,
                code_integrity,
                anchor_good,
                manifest_support_good,
                code_conflict_good,
            ],
            dim=-1,
        ),
        "graph": torch.stack(
            [
                graph_integrity,
                code_integrity,
                anchor_good,
                manifest_support_good,
                code_conflict_good,
            ],
            dim=-1,
        ),
        "manifest": torch.stack(
            [
                manifest_integrity,
                code_integrity,
                manifest_support_good,
                manifest_conflict_good,
                anchor_good,
            ],
            dim=-1,
        ),
        "joint": torch.stack(
            [
                mean_alive_integrity,
                code_integrity,
                pair_values.mean(dim=-1),
                conflict_good,
                alive_fraction,
            ],
            dim=-1,
        ),
    }
    diagnostics = {
        "api_graph_support_applicable": api_graph_applicable.float(),
        "manifest_code_conflict_applicable": manifest_code_applicable.float(),
        "manifest_code_relation_applicable": manifest_code_applicable.float(),
        "effective_manifest_to_code_conflict": torch.where(
            manifest_code_applicable, manifest_conflict, torch.zeros_like(manifest_conflict)
        ),
        "effective_code_to_manifest_conflict": torch.where(
            manifest_code_applicable, code_conflict, torch.zeros_like(code_conflict)
        ),
        "alive_api": api_alive.float(),
        "alive_graph": graph_alive.float(),
        "alive_manifest": manifest_alive.float(),
        "alive_joint": joint_alive.float(),
    }
    return features, diagnostics


class MonotonicReliabilityCalibrator(nn.Module):
    """Four branch-specific monotonic correctness-probability estimators."""

    def __init__(
        self,
        hidden_dim: int = 16,
        missing_relation_support: float = 0.0,
        neutral_support: float | None = None,
        apply_alive_mask: bool = True,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("reliability_calibration.hidden_dim must be positive")
        if neutral_support is not None:
            missing_relation_support = float(neutral_support)
        if not 0.0 <= float(missing_relation_support) <= 1.0:
            raise ValueError("reliability_calibration.missing_relation_support must be within [0, 1]")
        self.missing_relation_support = float(missing_relation_support)
        self.apply_alive_mask = bool(apply_alive_mask)
        self.branches = nn.ModuleDict(
            {
                name: MonotonicBranchCalibrator(input_dim=5, hidden_dim=hidden_dim)
                for name in BRANCH_NAMES
            }
        )

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        features, diagnostics = build_monotonic_reliability_features(
            evidence, missing_relation_support=self.missing_relation_support
        )
        outputs = dict(diagnostics)
        for name in BRANCH_NAMES:
            alive = diagnostics[f"alive_{name}"]
            reliability = self.branches[name](features[name])
            if self.apply_alive_mask:
                reliability = reliability * alive
            outputs[f"reliability_features_{name}"] = features[name]
            outputs[f"predicted_reliability_{name}"] = reliability
        return outputs
