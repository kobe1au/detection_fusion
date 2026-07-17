from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex


BRANCH_NAMES = ("api", "graph", "manifest", "joint")

# Keep the learned calibrator topology invariant across mechanism ablations.
# Feature flags below mask their slot to zero instead of changing a branch
# network's input dimension.  This prevents an I1 feature ablation from
# consuming a different number of RNG draws and thereby changing the global
# opinion router's initialization.
RELIABILITY_FEATURE_SUPERSET_LAYOUT = {
    "api": ("integrity", "model_visibility", "api_graph_support", "evidential_certainty"),
    "graph": ("integrity", "model_visibility", "api_graph_support", "evidential_certainty"),
    "manifest": ("integrity", "evidential_certainty"),
    "joint": (
        "mean_alive_integrity",
        "effective_code_integrity",
        "api_graph_support",
        "alive_fraction",
        "evidential_certainty",
    ),
}


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
    use_relation_evidence: bool = True,
    use_model_visibility: bool = False,
    evidential_certainty: dict[str, torch.Tensor] | None = None,
    fixed_superset_layout: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build branch-specific monotone reliability features.

    API-Graph anchor support is retained because it has a stable positive
    interpretation for the two code branches. Manifest-Code semantic
    support/conflict remains available in diagnostics, but is not forced into
    the monotone calibrator: its empirical direction can vary with declarations,
    libraries, and code visibility. Learned evidential certainty can be appended
    as a complementary sample-difficulty signal.
    """
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
    api_visibility = _column(evidence, EvidenceIndex.API_ENCODER_COVERAGE)
    graph_visibility = _column(evidence, EvidenceIndex.GRAPH_ENCODER_COVERAGE)

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

    missing_support = torch.full_like(anchor_support, float(missing_relation_support))
    # Relation evidence is an optional positive contribution. When the
    # counterpart is unavailable, no relation contribution is added; missing
    # support must never be encoded as maximal agreement.
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
    if not use_relation_evidence:
        no_relation_evidence = torch.zeros_like(anchor_support)
        anchor_good = no_relation_evidence
        manifest_support_good = no_relation_evidence
        manifest_conflict_good = no_relation_evidence
        code_conflict_good = no_relation_evidence

    # Single-modality reliability uses intrinsic quality exactly once.
    # Manifest-Code relations remain diagnostic because their direction is not
    # stable enough to impose as a universally positive monotone feature.
    effective_code_integrity = torch.where(
        api_alive & graph_alive,
        code_integrity,
        torch.where(api_alive, api_integrity, torch.where(graph_alive, graph_integrity, torch.zeros_like(code_integrity))),
    )

    alive_float = torch.stack([api_alive, graph_alive, manifest_alive], dim=-1).float()
    integrities = torch.stack(
        [api_integrity, graph_integrity, manifest_integrity], dim=-1
    )
    mean_alive_integrity = (
        integrities * alive_float
    ).sum(dim=-1) / alive_float.sum(dim=-1).clamp_min(1.0)
    alive_fraction = alive_float.mean(dim=-1)
    legacy_features = {
        "api": torch.stack(
            [
                api_integrity,
                anchor_good,
            ],
            dim=-1,
        ),
        "graph": torch.stack(
            [
                graph_integrity,
                anchor_good,
            ],
            dim=-1,
        ),
        "manifest": manifest_integrity.unsqueeze(-1),
        "joint": torch.stack(
            [
                mean_alive_integrity,
                effective_code_integrity,
                anchor_good,
                alive_fraction,
            ],
            dim=-1,
        ),
    }
    if use_model_visibility:
        # Visibility helps estimate branch correctness on clean calibration
        # data. The main routed method may additionally apply a relative
        # effective-integrity modifier as an explicit deployment-shift guard;
        # that separate constraint is reported in the diagnostics.
        legacy_features["api"] = torch.cat(
            [
                legacy_features["api"][:, :1],
                api_visibility.unsqueeze(-1),
                legacy_features["api"][:, 1:],
            ],
            dim=-1,
        )
        legacy_features["graph"] = torch.cat(
            [
                legacy_features["graph"][:, :1],
                graph_visibility.unsqueeze(-1),
                legacy_features["graph"][:, 1:],
            ],
            dim=-1,
        )
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
    certainty_by_branch: dict[str, torch.Tensor] = {}
    if evidential_certainty is not None:
        # Append learned evidential certainty (1 - u) as a complementary
        # monotone-positive feature. Missing entries default to neutral 1.0;
        # the explicit alive mask still removes unavailable branches.
        for name in BRANCH_NAMES:
            cert = evidential_certainty.get(name)
            if cert is None:
                cert = torch.ones_like(api_integrity)
            cert = (
                cert.to(device=evidence.device, dtype=evidence.dtype)
                .view(-1)
                .clamp(0.0, 1.0)
            )
            certainty_by_branch[name] = cert
            legacy_features[name] = torch.cat(
                [legacy_features[name], cert.unsqueeze(-1)], dim=-1
            )

    if not fixed_superset_layout:
        return legacy_features, diagnostics

    zeros = torch.zeros_like(api_integrity)
    api_visibility_feature = api_visibility if use_model_visibility else zeros
    graph_visibility_feature = graph_visibility if use_model_visibility else zeros
    certainty = {
        name: certainty_by_branch.get(name, zeros) for name in BRANCH_NAMES
    }
    features = {
        "api": torch.stack(
            [api_integrity, api_visibility_feature, anchor_good, certainty["api"]],
            dim=-1,
        ),
        "graph": torch.stack(
            [
                graph_integrity,
                graph_visibility_feature,
                anchor_good,
                certainty["graph"],
            ],
            dim=-1,
        ),
        "manifest": torch.stack(
            [manifest_integrity, certainty["manifest"]], dim=-1
        ),
        "joint": torch.stack(
            [
                mean_alive_integrity,
                effective_code_integrity,
                anchor_good,
                alive_fraction,
                certainty["joint"],
            ],
            dim=-1,
        ),
    }
    return features, diagnostics


class MonotonicReliabilityCalibrator(nn.Module):
    """Four branch-specific monotonic correctness-probability estimators."""

    def __init__(
        self,
        hidden_dim: int = 16,
        missing_relation_support: float = 0.0,
        use_relation_evidence: bool = True,
        use_model_visibility: bool = False,
        apply_alive_mask: bool = True,
        use_evidential_uncertainty: bool = False,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("reliability_calibration.hidden_dim must be positive")
        if not 0.0 <= float(missing_relation_support) <= 1.0:
            raise ValueError("reliability_calibration.missing_relation_support must be within [0, 1]")
        self.missing_relation_support = float(missing_relation_support)
        self.use_relation_evidence = bool(use_relation_evidence)
        self.use_model_visibility = bool(use_model_visibility)
        self.apply_alive_mask = bool(apply_alive_mask)
        self.use_evidential_uncertainty = bool(use_evidential_uncertainty)
        self.branches = nn.ModuleDict(
            {
                name: MonotonicBranchCalibrator(
                    input_dim=len(RELIABILITY_FEATURE_SUPERSET_LAYOUT[name]),
                    hidden_dim=hidden_dim,
                )
                for name in BRANCH_NAMES
            }
        )

    def forward(
        self,
        evidence: torch.Tensor,
        evidential_certainty: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.use_evidential_uncertainty and evidential_certainty is None:
            raise ValueError(
                "reliability calibrator was built with use_evidential_uncertainty=true "
                "but no evidential certainty was provided"
            )
        features, diagnostics = build_monotonic_reliability_features(
            evidence,
            missing_relation_support=self.missing_relation_support,
            use_relation_evidence=self.use_relation_evidence,
            use_model_visibility=self.use_model_visibility,
            evidential_certainty=evidential_certainty if self.use_evidential_uncertainty else None,
            fixed_superset_layout=True,
        )
        outputs = dict(diagnostics)
        for name in BRANCH_NAMES:
            alive = diagnostics[f"alive_{name}"]
            reliability = self.branches[name](features[name])
            if self.apply_alive_mask:
                reliability = reliability * alive
            # Preserve the historical active-feature diagnostic for downstream
            # reports while exposing the invariant network input explicitly.
            # The learned branch always consumes the fixed superset tensor.
            outputs[f"reliability_features_superset_{name}"] = features[name]
            if name in {"api", "graph"}:
                active_indices = [0]
                if self.use_model_visibility:
                    active_indices.append(1)
                active_indices.append(2)
                if self.use_evidential_uncertainty:
                    active_indices.append(3)
            elif name == "manifest":
                active_indices = [0] + (
                    [1] if self.use_evidential_uncertainty else []
                )
            else:
                active_indices = [0, 1, 2, 3] + (
                    [4] if self.use_evidential_uncertainty else []
                )
            outputs[f"reliability_features_{name}"] = features[name][
                :, active_indices
            ]
            outputs[f"predicted_reliability_{name}"] = reliability
        return outputs
