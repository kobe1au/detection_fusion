from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex, GateConstants
from fusion.reliability_calibration import (
    BRANCH_NAMES,
    MonotonicReliabilityCalibrator,
    build_monotonic_reliability_features,
)


def compute_branch_confidence_proxy(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
    eps: float = GateConstants.EPS,
) -> dict[str, torch.Tensor]:
    """Compute an entropy/margin softmax confidence proxy."""
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("temperature tensor must be scalar")
        temperature = temperature.to(device=logits.device, dtype=logits.dtype)
        if not bool(torch.isfinite(temperature).item()) or not bool((temperature > 0).item()):
            raise ValueError(f"temperature must be finite and positive, got {temperature.item()}")
    else:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if logits.ndim != 2 or logits.size(-1) < 2:
        raise ValueError(f"confidence proxy expects [B, C] logits with C >= 2, got {tuple(logits.shape)}")

    prob = F.softmax(logits / temperature, dim=-1)
    confidence = prob.max(dim=-1).values
    entropy = -(prob * torch.log(prob + eps)).sum(dim=-1)
    normalized_entropy = entropy / math.log(prob.size(-1))
    top2 = prob.topk(k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    uncertainty_proxy = (normalized_entropy * (1.0 - margin)).clamp(0.0, 1.0)
    confidence_factor = (1.0 - uncertainty_proxy).clamp(0.0, 1.0)
    return {
        "prob": prob,
        "confidence": confidence,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "margin": margin,
        "uncertainty_proxy": uncertainty_proxy,
        "confidence_factor": confidence_factor,
    }


def _column(evidence: torch.Tensor, index: int) -> torch.Tensor:
    return evidence[:, index].clamp(0.0, 1.0)


def _mean_alive_integrity(
    integrities: torch.Tensor,
    alive: torch.Tensor,
) -> torch.Tensor:
    alive_float = alive.to(dtype=integrities.dtype)
    return (integrities * alive_float).sum(dim=-1) / alive_float.sum(dim=-1).clamp_min(1.0)


def _available_pair_support(
    alive_api: torch.Tensor,
    alive_graph: torch.Tensor,
    alive_manifest: torch.Tensor,
    anchor_support: torch.Tensor,
    manifest_support: torch.Tensor,
    neutral_value: float,
) -> torch.Tensor:
    api_graph_available = alive_api & alive_graph
    manifest_code_available = alive_manifest & (alive_api | alive_graph)
    pair_count = api_graph_available.to(anchor_support.dtype) + manifest_code_available.to(anchor_support.dtype)
    pair_sum = (
        anchor_support * api_graph_available.to(anchor_support.dtype)
        + manifest_support * manifest_code_available.to(anchor_support.dtype)
    )
    neutral = torch.full_like(pair_sum, float(neutral_value))
    return torch.where(pair_count > 0, pair_sum / pair_count.clamp_min(1.0), neutral)


class DiscountProbabilityFusion(nn.Module):
    """Calibrated monotonic probability discount fusion with rejection support."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = dict(config or {})
        self.register_buffer("_calibration_active", torch.tensor(False, dtype=torch.bool))
        reliability_cfg = self.config.get("reliability_calibration", {}) or {}
        self.reliability_calibrator = (
            MonotonicReliabilityCalibrator(
                hidden_dim=int(reliability_cfg.get("hidden_dim", 16)),
                missing_relation_support=float(
                    reliability_cfg.get(
                        "missing_relation_support",
                        reliability_cfg.get("neutral_support", 0.0),
                    )
                ),
            )
            if bool(reliability_cfg.get("enabled", False))
            else None
        )
        probability_cfg = self.config.get("probability_calibration", {}) or {}
        self.temperature_parameters = None
        if bool(probability_cfg.get("enabled", False)):
            confidence_cfg = self.config.get("confidence_proxy", {}) or {}
            raw = {}
            for name in BRANCH_NAMES:
                initial = float(
                    probability_cfg.get(
                        f"initial_temperature_{name}",
                        confidence_cfg.get(f"temperature_{name}", 1.0),
                    )
                )
                if not math.isfinite(initial) or initial <= 0.0:
                    raise ValueError(f"Initial temperature for {name} must be positive")
                raw[name] = nn.Parameter(
                    torch.tensor(math.log(math.expm1(max(initial - 1.0e-4, 1.0e-4))))
                )
            self.temperature_parameters = nn.ParameterDict(raw)

    def calibration_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.reliability_calibrator is not None:
            parameters.extend(self.reliability_calibrator.parameters())
        if self.temperature_parameters is not None:
            parameters.extend(self.temperature_parameters.parameters())
        return parameters

    @property
    def calibration_active(self) -> bool:
        return bool(self._calibration_active.item())

    def set_calibration_active(self, enabled: bool) -> None:
        self._calibration_active.fill_(bool(enabled))

    def _temperature(self, name: str, confidence_cfg: dict) -> float | torch.Tensor:
        if self.temperature_parameters is None or not self.calibration_active:
            return float(confidence_cfg.get(f"temperature_{name}", 1.0))
        return F.softplus(self.temperature_parameters[name]) + 1.0e-4

    def forward(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        joint_logits: torch.Tensor,
        evidence: torch.Tensor,
        config: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = dict(self.config)
        cfg.update(config or {})
        if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
            raise ValueError(
                f"DiscountProbabilityFusion expected [B, >= {EvidenceIndex.BASE_DIM}] evidence, "
                f"got {tuple(evidence.shape)}"
            )

        eps = float(cfg.get("min_discount", GateConstants.EPS))
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"fusion.min_discount must be finite and positive, got {eps}")
        detach_confidence = bool(cfg.get("detach_confidence_proxy", True))
        detach_discount = bool(cfg.get("detach_discount", True))
        use_hard_alive = bool(cfg.get("use_hard_alive_mask", True))
        use_confidence = bool(cfg.get("use_confidence_proxy", True))
        use_conflict = bool(cfg.get("use_conflict_discount", True))

        confidence_cfg = cfg.get("confidence_proxy", {}) or {}
        logits_by_branch = (api_logits, graph_logits, manifest_logits, joint_logits)
        proxies: dict[str, dict[str, torch.Tensor]] = {}
        for name, logits in zip(BRANCH_NAMES, logits_by_branch):
            proxies[name] = compute_branch_confidence_proxy(
                logits,
                temperature=self._temperature(name, confidence_cfg),
                eps=eps,
            )

        confidence_factors = []
        for name in BRANCH_NAMES:
            factor = proxies[name]["confidence_factor"]
            if not use_confidence:
                factor = torch.ones_like(factor)
            elif detach_confidence:
                factor = factor.detach()
            confidence_factors.append(factor)

        api_integrity = _column(evidence, EvidenceIndex.API_INTEGRITY)
        graph_integrity = _column(evidence, EvidenceIndex.GRAPH_INTEGRITY)
        manifest_integrity = _column(evidence, EvidenceIndex.MANIFEST_INTEGRITY)
        anchor_support = _column(evidence, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT)
        manifest_support = _column(evidence, EvidenceIndex.MANIFEST_CODE_SUPPORT)
        manifest_conflict = _column(evidence, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT)
        code_conflict = _column(evidence, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT)
        alive_api = _column(evidence, EvidenceIndex.API_ALIVE).bool()
        alive_graph = _column(evidence, EvidenceIndex.GRAPH_ALIVE).bool()
        alive_manifest = _column(evidence, EvidenceIndex.MANIFEST_ALIVE).bool()
        alive_joint = alive_api | alive_graph | alive_manifest
        code_alive = alive_api | alive_graph
        api_graph_applicable = alive_api & alive_graph
        manifest_code_applicable = alive_manifest & code_alive

        support_cfg = cfg.get("support_factor", {}) or {}
        neutral_value = float(support_cfg.get("neutral_value", 0.5))
        code_anchor_base = float(support_cfg.get("code_anchor_base", 0.5))
        manifest_support_base = float(support_cfg.get("manifest_support_base", 0.5))
        code_anchor_factor = torch.where(
            api_graph_applicable,
            code_anchor_base + (1.0 - code_anchor_base) * anchor_support,
            torch.ones_like(anchor_support),
        )
        manifest_support_factor = torch.where(
            manifest_code_applicable,
            manifest_support_base + (1.0 - manifest_support_base) * manifest_support,
            torch.ones_like(manifest_support),
        )

        conflict_min = float((cfg.get("conflict_factor", {}) or {}).get("min_value", 0.05))
        if use_conflict:
            manifest_conflict_factor = torch.where(
                manifest_code_applicable,
                (1.0 - manifest_conflict).clamp_min(conflict_min),
                torch.ones_like(manifest_conflict),
            )
            code_conflict_factor = torch.where(
                manifest_code_applicable,
                (1.0 - code_conflict).clamp_min(conflict_min),
                torch.ones_like(code_conflict),
            )
        else:
            manifest_conflict_factor = torch.ones_like(manifest_conflict)
            code_conflict_factor = torch.ones_like(code_conflict)

        mean_alive_integrity = _mean_alive_integrity(
            torch.stack([api_integrity, graph_integrity, manifest_integrity], dim=-1),
            torch.stack([alive_api, alive_graph, alive_manifest], dim=-1),
        )
        pair_support = _available_pair_support(
            alive_api,
            alive_graph,
            alive_manifest,
            anchor_support,
            manifest_support,
            neutral_value,
        )
        pair_support_applicable = api_graph_applicable | manifest_code_applicable
        joint_support_factor = torch.where(
            pair_support_applicable,
            0.5 + 0.5 * pair_support,
            torch.ones_like(pair_support),
        )
        reliability_outputs: dict[str, torch.Tensor] = {}
        if self.reliability_calibrator is not None and self.calibration_active:
            reliability_outputs = self.reliability_calibrator(evidence)
            base_reliability = [
                reliability_outputs[f"predicted_reliability_{name}"]
                for name in BRANCH_NAMES
            ]
        else:
            features, feature_diagnostics = build_monotonic_reliability_features(evidence)
            reliability_outputs.update(feature_diagnostics)
            for name, value in features.items():
                reliability_outputs[f"reliability_features_{name}"] = value
            base_reliability = [
                api_integrity,
                graph_integrity,
                manifest_integrity,
                mean_alive_integrity,
            ]
        raw_discounts = torch.stack(
            [
                base_reliability[0] * code_anchor_factor * code_conflict_factor * confidence_factors[0],
                base_reliability[1] * code_anchor_factor * code_conflict_factor * confidence_factors[1],
                base_reliability[2] * manifest_support_factor * manifest_conflict_factor * confidence_factors[2],
                base_reliability[3] * joint_support_factor * confidence_factors[3],
            ],
            dim=-1,
        )
        alive_mask = torch.stack([alive_api, alive_graph, alive_manifest, alive_joint], dim=-1)
        discounts = raw_discounts * alive_mask.to(raw_discounts.dtype) if use_hard_alive else raw_discounts
        discounts_for_weight = discounts.detach() if detach_discount else discounts
        if use_hard_alive:
            discounts_for_weight = discounts_for_weight * alive_mask.to(discounts_for_weight.dtype)

        valid_sum = discounts_for_weight.sum(dim=-1, keepdim=True)
        fallback_used = valid_sum <= eps
        fallback = str(cfg.get("fallback", "uniform")).lower()
        if fallback != "uniform":
            raise ValueError(f"Unsupported discount fusion fallback: {fallback}")
        fallback_weights = torch.full_like(discounts_for_weight, 1.0 / len(BRANCH_NAMES))
        if use_hard_alive:
            alive_fallback = alive_mask.to(discounts_for_weight.dtype)
            alive_count = alive_fallback.sum(dim=-1, keepdim=True)
            fallback_weights = torch.where(
                alive_count > 0,
                alive_fallback / alive_count.clamp_min(1.0),
                fallback_weights,
            )
        fusion_weights = torch.where(
            fallback_used,
            fallback_weights,
            discounts_for_weight / valid_sum.clamp_min(eps),
        )

        branch_prob = torch.stack([proxies[name]["prob"] for name in BRANCH_NAMES], dim=1)
        final_prob = (fusion_weights.unsqueeze(-1) * branch_prob).sum(dim=1)
        final_prob = final_prob / final_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
        final_logits = torch.log(final_prob.clamp_min(eps))
        final_proxy = compute_branch_confidence_proxy(final_logits, temperature=1.0, eps=eps)
        reliability_matrix = torch.stack(base_reliability, dim=-1)
        total_reliability = (fusion_weights * reliability_matrix).sum(dim=-1).clamp(0.0, 1.0)
        effective_conflict = torch.maximum(
            torch.where(
                manifest_code_applicable, manifest_conflict, torch.zeros_like(manifest_conflict)
            ),
            torch.where(
                manifest_code_applicable, code_conflict, torch.zeros_like(code_conflict)
            ),
        )
        acceptance_components = torch.stack(
            [
                total_reliability,
                1.0 - final_proxy["uncertainty_proxy"],
                1.0 - effective_conflict,
            ],
            dim=-1,
        )
        acceptance_aggregation = str(cfg.get("acceptance_aggregation", "min")).lower()
        if acceptance_aggregation == "min":
            # A threshold on the minimum implements an explicit OR rejection:
            # low reliability OR high uncertainty OR high conflict.
            acceptance_score = acceptance_components.min(dim=-1).values
        elif acceptance_aggregation == "product":
            acceptance_score = acceptance_components.prod(dim=-1)
        else:
            raise ValueError(
                "fusion.acceptance_aggregation must be 'min' or 'product'"
            )
        acceptance_score = acceptance_score.clamp(0.0, 1.0)

        outputs: dict[str, torch.Tensor] = {
            "discounts": discounts,
            "fusion_weights": fusion_weights,
            "fallback_used": fallback_used.to(dtype=discounts.dtype).view(-1),
            "final_prob": final_prob,
            "final_logits": final_logits,
            "final_is_log_probability": torch.ones((), device=final_logits.device, dtype=torch.bool),
            "total_reliability": total_reliability,
            "final_uncertainty_proxy": final_proxy["uncertainty_proxy"],
            "effective_conflict": effective_conflict,
            "acceptance_score": acceptance_score,
            "calibration_active": torch.full(
                (final_logits.size(0),),
                float(self.calibration_active),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "api_graph_conflict_applicable": api_graph_applicable.to(dtype=discounts.dtype),
            "manifest_code_conflict_applicable": manifest_code_applicable.to(dtype=discounts.dtype),
        }
        outputs.update(reliability_outputs)
        for index, name in enumerate(BRANCH_NAMES):
            outputs[f"discount_{name}"] = discounts[:, index]
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            outputs[f"calibrated_log_prob_{name}"] = torch.log(
                proxies[name]["prob"].clamp_min(eps)
            )
            outputs[f"temperature_{name}"] = torch.as_tensor(
                self._temperature(name, confidence_cfg),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ).view(1).expand(final_logits.size(0))
            for key in ("confidence", "entropy", "normalized_entropy", "margin", "uncertainty_proxy", "confidence_factor"):
                outputs[f"{key}_{name}"] = proxies[name][key]
        return outputs
