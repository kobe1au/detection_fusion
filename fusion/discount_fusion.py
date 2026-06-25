from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex, GateConstants
from fusion.evidential import (
    EVIDENCE_BRANCHES,
    COMBINATION_RULES,
    combine_opinions,
    logits_to_opinion,
    opinion_to_prob,
    trust_discount,
)
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


def _optional_column(evidence: torch.Tensor, index: int, default: float = 1.0) -> torch.Tensor:
    if evidence.size(-1) > index:
        return _column(evidence, index)
    return torch.full_like(evidence[:, 0], float(default)).clamp(0.0, 1.0)


def _mean_alive_integrity(
    integrities: torch.Tensor,
    alive: torch.Tensor,
) -> torch.Tensor:
    alive_float = alive.to(dtype=integrities.dtype)
    return (integrities * alive_float).sum(dim=-1) / alive_float.sum(dim=-1).clamp_min(1.0)


def _available_pair_support(
    api_graph_available: torch.Tensor,
    manifest_code_available: torch.Tensor,
    anchor_support: torch.Tensor,
    manifest_support: torch.Tensor,
) -> torch.Tensor:
    pair_count = api_graph_available.to(anchor_support.dtype) + manifest_code_available.to(anchor_support.dtype)
    pair_sum = (
        anchor_support * api_graph_available.to(anchor_support.dtype)
        + manifest_support * manifest_code_available.to(anchor_support.dtype)
    )
    return torch.where(
        pair_count > 0,
        pair_sum / pair_count.clamp_min(1.0),
        torch.zeros_like(pair_sum),
    )


class DiscountProbabilityFusion(nn.Module):
    """Calibrated monotonic probability discount fusion with rejection support."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = dict(config or {})
        if not bool(self.config.get("force_fp32_decision", True)):
            raise ValueError("fusion.force_fp32_decision=false is unsupported")
        self.register_buffer("_calibration_active", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "_branch_competence_active",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_branch_competence_prior",
            torch.ones(len(BRANCH_NAMES), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_visible_integrity_reference_active",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_visible_integrity_reference",
            torch.ones(len(EVIDENCE_BRANCHES), dtype=torch.float32),
            persistent=False,
        )
        reliability_cfg = self.config.get("reliability_calibration", {}) or {}
        self.reliability_calibrator = (
            MonotonicReliabilityCalibrator(
                hidden_dim=int(reliability_cfg.get("hidden_dim", 16)),
                missing_relation_support=float(reliability_cfg.get("missing_relation_support", 0.0)),
                use_relation_evidence=bool(reliability_cfg.get("use_relation_evidence", False)),
                apply_alive_mask=bool(reliability_cfg.get("apply_alive_mask", True)),
                use_evidential_uncertainty=bool(
                    reliability_cfg.get("use_evidential_uncertainty", False)
                ),
            )
            if bool(reliability_cfg.get("enabled", False))
            else None
        )
        # I2 combination rule. "linear" keeps the legacy reliability-discount
        # probability average (an ablation baseline); the evidential rules build
        # subjective-logic opinions and combine them conflict-awarely.
        combination = str(self.config.get("combination", "linear")).lower()
        if combination != "linear" and combination not in COMBINATION_RULES:
            raise ValueError(
                f"fusion.combination must be 'linear' or one of {COMBINATION_RULES}, got {combination}"
            )
        self.combination = combination
        self.evidence_activation = str(self.config.get("evidence_activation", "softplus")).lower()
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

    @property
    def branch_competence_active(self) -> bool:
        return bool(self._branch_competence_active.item())

    def set_branch_competence_prior(self, values, enabled: bool = True) -> None:
        tensor = torch.as_tensor(
            values,
            dtype=self._branch_competence_prior.dtype,
            device=self._branch_competence_prior.device,
        ).view(-1)
        if tensor.numel() != len(BRANCH_NAMES):
            raise ValueError(
                f"branch competence prior expects {len(BRANCH_NAMES)} values, got {tensor.numel()}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("branch competence prior contains non-finite values")
        if bool((tensor < 0.0).any()) or bool((tensor > 1.0).any()):
            raise ValueError("branch competence prior values must be within [0, 1]")
        self._branch_competence_prior.copy_(tensor.clamp(0.0, 1.0))
        self._branch_competence_active.fill_(bool(enabled))

    def branch_competence_prior_values(self) -> dict[str, float]:
        values = self._branch_competence_prior.detach().cpu().tolist()
        return {name: float(value) for name, value in zip(BRANCH_NAMES, values)}

    @property
    def visible_integrity_reference_active(self) -> bool:
        return bool(self._visible_integrity_reference_active.item())

    def set_visible_integrity_reference(self, values, enabled: bool = True) -> None:
        tensor = torch.as_tensor(
            values,
            dtype=self._visible_integrity_reference.dtype,
            device=self._visible_integrity_reference.device,
        ).view(-1)
        if tensor.numel() != len(EVIDENCE_BRANCHES):
            raise ValueError(
                f"visible integrity reference expects {len(EVIDENCE_BRANCHES)} values, got {tensor.numel()}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("visible integrity reference contains non-finite values")
        if bool((tensor <= 0.0).any()) or bool((tensor > 1.0).any()):
            raise ValueError("visible integrity reference values must be within (0, 1]")
        self._visible_integrity_reference.copy_(tensor.clamp(1.0e-6, 1.0))
        self._visible_integrity_reference_active.fill_(bool(enabled))

    def visible_integrity_reference_values(self) -> dict[str, float]:
        values = self._visible_integrity_reference.detach().cpu().tolist()
        return {name: float(value) for name, value in zip(EVIDENCE_BRANCHES, values)}

    def _visible_integrity_terms(
        self,
        evidence: torch.Tensor,
        cfg: dict,
        eps: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, float, float]:
        modifier_cfg = cfg.get("visible_integrity_modifier", {}) or {}
        enabled = bool(modifier_cfg.get("enabled", False))
        beta = float(modifier_cfg.get("beta", 1.0))
        min_value = float(modifier_cfg.get("min_value", 0.5))
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("fusion.visible_integrity_modifier.beta must be finite and positive")
        if not math.isfinite(min_value) or not 0.0 <= min_value <= 1.0:
            raise ValueError("fusion.visible_integrity_modifier.min_value must be within [0, 1]")

        api_integrity = _column(evidence, EvidenceIndex.API_INTEGRITY)
        graph_integrity = _column(evidence, EvidenceIndex.GRAPH_INTEGRITY)
        manifest_integrity = _column(evidence, EvidenceIndex.MANIFEST_INTEGRITY)
        api_coverage = _optional_column(evidence, EvidenceIndex.API_ENCODER_COVERAGE, 1.0)
        graph_coverage = _optional_column(evidence, EvidenceIndex.GRAPH_ENCODER_COVERAGE, 1.0)
        effective = {
            "api": (api_integrity * api_coverage).clamp(0.0, 1.0),
            "graph": (graph_integrity * graph_coverage).clamp(0.0, 1.0),
            "manifest": manifest_integrity.clamp(0.0, 1.0),
        }
        device = evidence.device
        dtype = evidence.dtype
        if enabled and self.visible_integrity_reference_active:
            reference = self._visible_integrity_reference.to(device=device, dtype=dtype).clamp_min(eps)
            reference_by_branch = {name: reference[i] for i, name in enumerate(EVIDENCE_BRANCHES)}
            relative = {
                name: (effective[name] / reference_by_branch[name]).clamp(0.0, 1.0)
                for name in EVIDENCE_BRANCHES
            }
            factor = {
                name: (min_value + (1.0 - min_value) * relative[name].pow(beta)).clamp(0.0, 1.0)
                for name in EVIDENCE_BRANCHES
            }
            active = torch.full((evidence.size(0),), 1.0, device=device, dtype=dtype)
        else:
            reference_by_branch = {
                name: torch.ones((), device=device, dtype=dtype) for name in EVIDENCE_BRANCHES
            }
            relative = {name: torch.ones_like(effective[name]) for name in EVIDENCE_BRANCHES}
            factor = {name: torch.ones_like(effective[name]) for name in EVIDENCE_BRANCHES}
            active = torch.zeros((evidence.size(0),), device=device, dtype=dtype)
        return effective, relative, factor, active, beta, min_value
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
        # Encoders can remain under AMP, but calibration, probability discounting,
        # and rejection scores need FP32 to avoid quantized temperatures/thresholds.
        with torch.autocast(device_type=api_logits.device.type, enabled=False):
            return self._forward_fp32(
                api_logits.float(),
                graph_logits.float(),
                manifest_logits.float(),
                joint_logits.float(),
                evidence.float(),
                config,
            )

    def _forward_evidential_fp32(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        joint_logits: torch.Tensor,
        evidence: torch.Tensor,
        cfg: dict,
        rule: str,
    ) -> dict[str, torch.Tensor]:
        """I2: conflict-aware evidential fusion of the three real modalities.

        Each modality emits a Dirichlet/subjective-logic opinion (I1). The
        opinion is trust-discounted by its reliability (competence x observable
        quality x evidential certainty) and the three are combined with a
        conflict-aware rule (Yager by default). The joint branch is NOT combined
        -- it is a learned interaction of the same three embeddings and would
        double-count -- it stays an auxiliary head only. Conflict mass becomes
        fused uncertainty, which is the natural selective-rejection signal (I3).
        """
        eps = float(cfg.get("min_discount", GateConstants.EPS))
        use_hard_alive = bool(cfg.get("use_hard_alive_mask", True))
        use_reliability = bool(cfg.get("use_reliability_discount", True))
        reliability_exponent = float(cfg.get("reliability_discount_exponent", 1.0))
        if not math.isfinite(reliability_exponent) or reliability_exponent <= 0.0:
            raise ValueError("fusion.reliability_discount_exponent must be finite and positive")
        competence_cfg = cfg.get("branch_competence_prior", {}) or {}
        use_competence = bool(competence_cfg.get("enabled", False))
        base_rate = float(cfg.get("base_rate", 0.5))

        logits_by_branch = {
            "api": api_logits,
            "graph": graph_logits,
            "manifest": manifest_logits,
            "joint": joint_logits,
        }
        opinions = {
            name: logits_to_opinion(
                logits, evidence_activation=self.evidence_activation, eps=eps
            )
            for name, logits in logits_by_branch.items()
        }
        evidential_certainty = {
            name: (1.0 - opinions[name]["uncertainty"]).clamp(0.0, 1.0) for name in BRANCH_NAMES
        }

        api_integrity = _column(evidence, EvidenceIndex.API_INTEGRITY)
        graph_integrity = _column(evidence, EvidenceIndex.GRAPH_INTEGRITY)
        manifest_integrity = _column(evidence, EvidenceIndex.MANIFEST_INTEGRITY)
        integrity_by_branch = {
            "api": api_integrity,
            "graph": graph_integrity,
            "manifest": manifest_integrity,
        }
        alive_by_branch = {
            "api": _column(evidence, EvidenceIndex.API_ALIVE).clamp(0.0, 1.0),
            "graph": _column(evidence, EvidenceIndex.GRAPH_ALIVE).clamp(0.0, 1.0),
            "manifest": _column(evidence, EvidenceIndex.MANIFEST_ALIVE).clamp(0.0, 1.0),
        }

        # Observable reliability: calibrated dual-source estimate when active,
        # otherwise raw observable integrity (optionally multiplied by evidential
        # certainty so the second source still participates in ablations).
        reliability_outputs: dict[str, torch.Tensor] = {}
        calibrated = self.reliability_calibrator is not None and self.calibration_active
        if calibrated:
            reliability_outputs = self.reliability_calibrator(evidence, evidential_certainty)
            observable_reliability = {
                name: reliability_outputs[f"predicted_reliability_{name}"].clamp(0.0, 1.0)
                for name in EVIDENCE_BRANCHES
            }
        else:
            use_evidential = bool(
                (cfg.get("reliability_calibration", {}) or {}).get("use_evidential_uncertainty", False)
            )
            observable_reliability = {}
            for name in EVIDENCE_BRANCHES:
                base = integrity_by_branch[name].clamp(0.0, 1.0)
                if use_evidential:
                    base = base * evidential_certainty[name]
                observable_reliability[name] = base.clamp(0.0, 1.0)

        if use_competence and self.branch_competence_active:
            competence_prior = self._branch_competence_prior.to(
                device=api_integrity.device, dtype=api_integrity.dtype
            )
        else:
            competence_prior = torch.ones(
                len(BRANCH_NAMES), device=api_integrity.device, dtype=api_integrity.dtype
            )
        competence_by_branch = {name: competence_prior[i] for i, name in enumerate(BRANCH_NAMES)}
        (
            visible_effective_integrity,
            visible_modifier,
            visible_modifier_factor,
            visible_modifier_active,
            visible_modifier_beta,
            visible_modifier_min_value,
        ) = self._visible_integrity_terms(evidence, cfg, eps)

        beliefs: list[torch.Tensor] = []
        uncertainties: list[torch.Tensor] = []
        trust_by_branch: dict[str, torch.Tensor] = {}
        for name in EVIDENCE_BRANCHES:
            reliability = observable_reliability[name]
            if use_reliability:
                reliability = reliability.clamp(0.0, 1.0).pow(reliability_exponent)
            else:
                reliability = torch.ones_like(reliability)
            trust = (competence_by_branch[name] * reliability * visible_modifier_factor[name]).clamp(0.0, 1.0)
            if use_hard_alive:
                # Dead modalities become vacuous opinions (the identity element of
                # the fusion), contributing nothing instead of voting.
                trust = trust * alive_by_branch[name]
            trust_by_branch[name] = trust
            discounted_belief, discounted_u = trust_discount(
                opinions[name]["belief"], opinions[name]["uncertainty"], trust
            )
            beliefs.append(discounted_belief)
            uncertainties.append(discounted_u)

        fused_belief, fused_uncertainty = combine_opinions(beliefs, uncertainties, rule=rule, eps=eps)
        final_prob = opinion_to_prob(fused_belief, fused_uncertainty, base_rate=base_rate, eps=eps)
        final_logits = torch.log(final_prob.clamp_min(eps))

        # Pseudo fusion weights for diagnostics / gate dump: how much each
        # modality's (trusted) belief contributed. Joint is fixed at 0.
        contribution = torch.stack(
            [trust_by_branch[name] * (1.0 - opinions[name]["uncertainty"]) for name in EVIDENCE_BRANCHES],
            dim=-1,
        )
        contribution_sum = contribution.sum(dim=-1, keepdim=True).clamp_min(eps)
        weights3 = contribution / contribution_sum
        fusion_weights = torch.cat([weights3, torch.zeros_like(weights3[:, :1])], dim=-1)

        total_reliability = (weights3 * torch.stack(
            [trust_by_branch[name] for name in EVIDENCE_BRANCHES], dim=-1
        )).sum(dim=-1).clamp(0.0, 1.0)
        acceptance_score = (1.0 - fused_uncertainty).clamp(0.0, 1.0)

        batch = final_logits.size(0)
        outputs: dict[str, torch.Tensor] = {
            "final_prob": final_prob,
            "final_logits": final_logits,
            "final_is_log_probability": torch.ones((), device=final_logits.device, dtype=torch.bool),
            "fusion_weights": fusion_weights,
            "fused_uncertainty": fused_uncertainty,
            "fused_belief_malware": fused_belief[:, 1] if fused_belief.size(-1) > 1 else fused_belief[:, 0],
            "total_reliability": total_reliability,
            "acceptance_score": acceptance_score,
            "combination_rule_is_evidential": torch.ones((batch,), device=final_logits.device, dtype=final_logits.dtype),
            "branch_competence_active": torch.full(
                (batch,),
                float(use_competence and self.branch_competence_active),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "visible_integrity_modifier_active": visible_modifier_active,
            "visible_integrity_modifier_beta": torch.full(
                (batch,),
                visible_modifier_beta,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "visible_integrity_modifier_min_value": torch.full(
                (batch,),
                visible_modifier_min_value,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "calibration_active": torch.full(
                (batch,), float(self.calibration_active), device=final_logits.device, dtype=final_logits.dtype
            ),
            "fallback_used": torch.zeros((batch,), device=final_logits.device, dtype=final_logits.dtype),
        }
        outputs.update(reliability_outputs)
        for index, name in enumerate(BRANCH_NAMES):
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            outputs[f"uncertainty_proxy_{name}"] = opinions[name]["uncertainty"]
            outputs[f"evidential_certainty_{name}"] = evidential_certainty[name]
            outputs[f"branch_competence_prior_{name}"] = torch.full(
                (batch,),
                float(competence_prior[index].detach().cpu().item()),
                device=final_logits.device,
                dtype=final_logits.dtype,
            )
            if name in visible_effective_integrity:
                ref_index = EVIDENCE_BRANCHES.index(name)
                outputs[f"effective_{name}_integrity"] = visible_effective_integrity[name]
                outputs[f"visible_modifier_{name}"] = visible_modifier[name]
                outputs[f"visible_modifier_factor_{name}"] = visible_modifier_factor[name]
                outputs[f"visible_integrity_reference_{name}"] = torch.full(
                    (batch,),
                    float(self._visible_integrity_reference[ref_index].detach().cpu().item()),
                    device=final_logits.device,
                    dtype=final_logits.dtype,
                )
            outputs[f"calibrated_log_prob_{name}"] = torch.log(
                opinions[name]["expected_prob"].clamp_min(eps)
            )
            if name in trust_by_branch:
                outputs[f"discount_{name}"] = trust_by_branch[name]
        return outputs

    def _forward_fp32(
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

        combination = str(cfg.get("combination", self.combination)).lower()
        if combination in COMBINATION_RULES:
            return self._forward_evidential_fp32(
                api_logits, graph_logits, manifest_logits, joint_logits, evidence, cfg, combination
            )

        eps = float(cfg.get("min_discount", GateConstants.EPS))
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"fusion.min_discount must be finite and positive, got {eps}")
        detach_confidence = bool(cfg.get("detach_confidence_proxy", True))
        detach_discount = bool(cfg.get("detach_discount", True))
        use_hard_alive = bool(cfg.get("use_hard_alive_mask", True))
        use_confidence = bool(cfg.get("use_confidence_proxy", True))
        use_reliability = bool(cfg.get("use_reliability_discount", True))
        use_reliability_acceptance = bool(
            cfg.get("use_reliability_acceptance", use_reliability)
        )
        reliability_exponent = float(cfg.get("reliability_discount_exponent", 1.0))
        if not math.isfinite(reliability_exponent) or reliability_exponent <= 0.0:
            raise ValueError(
                "fusion.reliability_discount_exponent must be finite and positive"
            )
        use_conflict = bool(cfg.get("use_conflict_discount", True))
        use_support = bool(cfg.get("use_support_discount", True))
        competence_cfg = cfg.get("branch_competence_prior", {}) or {}
        use_competence = bool(competence_cfg.get("enabled", False))
        weight_gamma = float(cfg.get("weight_sharpening_gamma", 1.0))
        if not math.isfinite(weight_gamma) or weight_gamma <= 0.0:
            raise ValueError("fusion.weight_sharpening_gamma must be finite and positive")

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
        manifest_code_relation_observed = (
            (manifest_support > 0.0)
            | (manifest_conflict > 0.0)
            | (code_conflict > 0.0)
        )
        manifest_code_applicable = alive_manifest & code_alive & manifest_code_relation_observed

        support_cfg = cfg.get("support_factor", {}) or {}
        code_anchor_base = float(support_cfg.get("code_anchor_base", 0.5))
        manifest_support_base = float(support_cfg.get("manifest_support_base", 0.5))
        for name, value in (
            ("code_anchor_base", code_anchor_base),
            ("manifest_support_base", manifest_support_base),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"fusion.support_factor.{name} must be within [0, 1]")
        calibrated_reliability_active = (
            self.reliability_calibrator is not None and self.calibration_active
        )
        # Relation evidence is applied exactly once through these explicit
        # factors. The formal calibrator uses intrinsic observable quality only.
        apply_explicit_relation_factors = use_support or use_conflict
        if use_support:
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
        else:
            code_anchor_factor = torch.ones_like(anchor_support)
            manifest_support_factor = torch.ones_like(manifest_support)

        conflict_min = float((cfg.get("conflict_factor", {}) or {}).get("min_value", 0.05))
        if not math.isfinite(conflict_min) or not 0.0 <= conflict_min <= 1.0:
            raise ValueError("fusion.conflict_factor.min_value must be within [0, 1]")
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
        joint_conflict_factor = torch.minimum(
            manifest_conflict_factor,
            code_conflict_factor,
        )

        mean_alive_integrity = _mean_alive_integrity(
            torch.stack([api_integrity, graph_integrity, manifest_integrity], dim=-1),
            torch.stack([alive_api, alive_graph, alive_manifest], dim=-1),
        )
        pair_support = _available_pair_support(
            api_graph_applicable,
            manifest_code_applicable,
            anchor_support,
            manifest_support,
        )
        pair_support_applicable = api_graph_applicable | manifest_code_applicable
        joint_support_factor = (
            torch.where(
                pair_support_applicable,
                0.5 + 0.5 * pair_support,
                torch.ones_like(pair_support),
            )
            if use_support
            else torch.ones_like(pair_support)
        )
        reliability_outputs: dict[str, torch.Tensor] = {}
        if calibrated_reliability_active:
            reliability_outputs = self.reliability_calibrator(evidence)
            base_reliability = [
                reliability_outputs[f"predicted_reliability_{name}"]
                for name in BRANCH_NAMES
            ]
        else:
            reliability_cfg = cfg.get("reliability_calibration", {}) or {}
            features, feature_diagnostics = build_monotonic_reliability_features(
                evidence,
                missing_relation_support=float(
                    reliability_cfg.get("missing_relation_support", 0.0)
                ),
                use_relation_evidence=bool(
                    reliability_cfg.get("use_relation_evidence", False)
                ),
            )
            reliability_outputs.update(feature_diagnostics)
            for name, value in features.items():
                reliability_outputs[f"reliability_features_{name}"] = value
            base_reliability = [
                api_integrity,
                graph_integrity,
                manifest_integrity,
                mean_alive_integrity,
            ]
        estimated_reliability = [
            value.clamp(0.0, 1.0) for value in base_reliability
        ]
        if use_reliability:
            reliability_for_fusion = [
                value.pow(reliability_exponent).clamp(0.0, 1.0)
                for value in estimated_reliability
            ]
        else:
            reliability_for_fusion = [
                torch.ones_like(api_integrity) for _ in BRANCH_NAMES
            ]
        if use_reliability_acceptance:
            reliability_for_acceptance = estimated_reliability
        else:
            reliability_for_acceptance = [
                torch.ones_like(api_integrity) for _ in BRANCH_NAMES
            ]
        if use_competence and self.branch_competence_active:
            competence_prior = self._branch_competence_prior.to(
                device=api_integrity.device,
                dtype=api_integrity.dtype,
            )
        else:
            competence_prior = torch.ones(
                len(BRANCH_NAMES),
                device=api_integrity.device,
                dtype=api_integrity.dtype,
            )
        (
            visible_effective_integrity,
            visible_modifier,
            visible_modifier_factor,
            visible_modifier_active,
            visible_modifier_beta,
            visible_modifier_min_value,
        ) = self._visible_integrity_terms(evidence, cfg, eps)
        raw_discounts = torch.stack(
            [
                competence_prior[0] * reliability_for_fusion[0] * visible_modifier_factor["api"] * code_anchor_factor * code_conflict_factor * confidence_factors[0],
                competence_prior[1] * reliability_for_fusion[1] * visible_modifier_factor["graph"] * code_anchor_factor * code_conflict_factor * confidence_factors[1],
                competence_prior[2] * reliability_for_fusion[2] * visible_modifier_factor["manifest"] * manifest_support_factor * manifest_conflict_factor * confidence_factors[2],
                competence_prior[3] * reliability_for_fusion[3] * joint_support_factor * joint_conflict_factor * confidence_factors[3],
            ],
            dim=-1,
        )
        alive_mask = torch.stack([alive_api, alive_graph, alive_manifest, alive_joint], dim=-1)
        discounts = raw_discounts * alive_mask.to(raw_discounts.dtype) if use_hard_alive else raw_discounts
        discounts_for_weight = discounts.detach() if detach_discount else discounts
        if use_hard_alive:
            discounts_for_weight = discounts_for_weight * alive_mask.to(discounts_for_weight.dtype)
        if weight_gamma != 1.0:
            discounts_for_weight = discounts_for_weight.clamp_min(0.0).pow(weight_gamma)

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
        reliability_matrix = torch.stack(reliability_for_acceptance, dim=-1)
        total_reliability = (fusion_weights * reliability_matrix).sum(dim=-1).clamp(0.0, 1.0)
        effective_conflict = (
            torch.maximum(
                torch.where(
                    manifest_code_applicable,
                    manifest_conflict,
                    torch.zeros_like(manifest_conflict),
                ),
                torch.where(
                    manifest_code_applicable,
                    code_conflict,
                    torch.zeros_like(code_conflict),
                ),
            )
            if use_conflict
            else torch.zeros_like(manifest_conflict)
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

        batch = final_logits.size(0)
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
            "visible_integrity_modifier_active": visible_modifier_active,
            "visible_integrity_modifier_beta": torch.full(
                (batch,),
                visible_modifier_beta,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "visible_integrity_modifier_min_value": torch.full(
                (batch,),
                visible_modifier_min_value,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "calibration_active": torch.full(
                (final_logits.size(0),),
                float(self.calibration_active),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "explicit_relation_factors_active": torch.full(
                (final_logits.size(0),),
                float(apply_explicit_relation_factors),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "reliability_discount_active": torch.full(
                (final_logits.size(0),),
                float(use_reliability),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "reliability_acceptance_active": torch.full(
                (final_logits.size(0),),
                float(use_reliability_acceptance),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "reliability_discount_exponent": torch.full(
                (final_logits.size(0),),
                reliability_exponent,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "branch_competence_active": torch.full(
                (final_logits.size(0),),
                float(use_competence and self.branch_competence_active),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "weight_sharpening_gamma": torch.full(
                (final_logits.size(0),),
                weight_gamma,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "joint_conflict_factor": joint_conflict_factor,
            "api_graph_support_applicable": api_graph_applicable.to(dtype=discounts.dtype),
            "manifest_code_conflict_applicable": manifest_code_applicable.to(dtype=discounts.dtype),
            "manifest_code_relation_applicable": manifest_code_applicable.to(dtype=discounts.dtype),
        }
        outputs.update(reliability_outputs)
        for index, name in enumerate(BRANCH_NAMES):
            outputs[f"discount_{name}"] = discounts[:, index]
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            outputs[f"branch_competence_prior_{name}"] = torch.full(
                (final_logits.size(0),),
                float(competence_prior[index].detach().cpu().item()),
                device=final_logits.device,
                dtype=final_logits.dtype,
            )
            if name in visible_effective_integrity:
                ref_index = EVIDENCE_BRANCHES.index(name)
                outputs[f"effective_{name}_integrity"] = visible_effective_integrity[name]
                outputs[f"visible_modifier_{name}"] = visible_modifier[name]
                outputs[f"visible_modifier_factor_{name}"] = visible_modifier_factor[name]
                outputs[f"visible_integrity_reference_{name}"] = torch.full(
                    (batch,),
                    float(self._visible_integrity_reference[ref_index].detach().cpu().item()),
                    device=final_logits.device,
                    dtype=final_logits.dtype,
                )
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
