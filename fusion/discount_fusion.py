from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex, GateConstants
from fusion.evidential import (
    EVIDENCE_BRANCHES,
    COMBINATION_RULES,
    combine_opinions_with_diagnostics,
    logits_to_opinion,
    multisource_conflict,
    opinion_to_dirichlet_alpha,
    opinion_to_prob,
    predictive_opinion_conflict,
    trust_discount,
    logits_to_softmax_opinion,
)
from fusion.opinion_router import GlobalOpinionRouter
from fusion.reliability_calibration import (
    BRANCH_NAMES,
    MONOTONIC_CORRECTNESS_METHOD,
    TEMPERATURE_SCALING_CONFIDENCE_METHOD,
    BranchTemperatureScalingConfidenceCalibrator,
    MonotonicReliabilityCalibrator,
    build_monotonic_reliability_features,
    normalize_reliability_calibration_method,
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


def _effective_integrity_terms(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return observable integrity after the actual encoder coverage budget."""
    return {
        "api": (
            _column(evidence, EvidenceIndex.API_INTEGRITY)
            * _column(evidence, EvidenceIndex.API_ENCODER_COVERAGE)
        ).clamp(0.0, 1.0),
        "graph": (
            _column(evidence, EvidenceIndex.GRAPH_INTEGRITY)
            * _column(evidence, EvidenceIndex.GRAPH_ENCODER_COVERAGE)
        ).clamp(0.0, 1.0),
        "manifest": _column(evidence, EvidenceIndex.MANIFEST_INTEGRITY),
    }


class DiscountProbabilityFusion(nn.Module):
    """Calibrated monotonic probability discount fusion with rejection support."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        embedding_dims: dict[str, int] | None = None,
    ):
        super().__init__()
        self.config = dict(config or {})
        if "final_temperature_scaling" in self.config:
            raise ValueError(
                "fusion.final_temperature_scaling is not a valid top-level "
                "switch; use fusion.routing.final_temperature_scaling"
            )
        if not bool(self.config.get("force_fp32_decision", True)):
            raise ValueError("fusion.force_fp32_decision=false is unsupported")
        self.register_buffer("_calibration_active", torch.tensor(False, dtype=torch.bool))
        # The buffer is the checkpoint source of truth; the Python shadow is
        # the hot-path value. Reading a CUDA scalar buffer with ``.item()`` in
        # every forward would otherwise serialize the device.
        self._calibration_active_shadow = False
        removed_fusion_keys = sorted(
            set(self.config)
            & {"branch_competence_prior", "visible_integrity_modifier"}
        )
        if removed_fusion_keys:
            raise ValueError(
                "Removed validation-global fusion keys are unsupported: "
                f"{removed_fusion_keys}"
            )
        reliability_cfg = self.config.get("reliability_calibration", {}) or {}
        configured_embedding_dims = reliability_cfg.get("embedding_dims")
        if configured_embedding_dims is not None and not isinstance(
            configured_embedding_dims, dict
        ):
            raise ValueError(
                "fusion.reliability_calibration.embedding_dims must be a mapping"
            )
        if embedding_dims is None and isinstance(configured_embedding_dims, dict):
            embedding_dims = {
                str(name): int(value)
                for name, value in configured_embedding_dims.items()
            }
        elif embedding_dims is not None and isinstance(configured_embedding_dims, dict):
            normalized_configured_dims = {
                str(name): int(value)
                for name, value in configured_embedding_dims.items()
            }
            normalized_runtime_dims = {
                str(name): int(value) for name, value in embedding_dims.items()
            }
            if normalized_configured_dims != normalized_runtime_dims:
                raise ValueError(
                    "Configured I1 embedding_dims disagree with encoder dimensions: "
                    f"configured={normalized_configured_dims}, "
                    f"runtime={normalized_runtime_dims}"
                )
        removed_reliability_keys = sorted(
            set(reliability_cfg)
            & {
                "feature_schema",
                "missing_relation_support",
                "use_relation_evidence",
                "use_edl_certainty_feature",
                "use_evidential_uncertainty",
                "group_mean_alignment",
            }
        )
        if removed_reliability_keys:
            raise ValueError(
                "Removed I1 reliability keys are unsupported: "
                f"{removed_reliability_keys}"
            )
        # I2 combination rule. "linear" provides a reliability-discount
        # probability comparison; the evidence rules operate on opinions.
        combination = str(self.config.get("combination", "linear")).lower()
        if combination not in {"linear", "routed"} and combination not in COMBINATION_RULES:
            raise ValueError(
                "fusion.combination must be 'linear', 'routed', or one of "
                f"{COMBINATION_RULES}, got {combination}"
            )
        self.combination = combination
        routing_cfg = self.config.get("routing", {}) or {}
        removed_routing_keys = sorted(
            set(routing_cfg)
            & {
                "use_disagreement",
                "risk_enabled",
                "target_loss_weight",
                "use_fused_prediction_loss",
                "mass_constraint",
                "known_mass_excess_penalty_weight",
                "initial_known_retention",
                "acceptance_mode",
                "hidden_dim",
            }
        )
        if removed_routing_keys:
            raise ValueError(
                "Removed I2-v1 routing keys are not supported: "
                f"{removed_routing_keys}"
            )
        self.routing_mode = str(routing_cfg.get("mode", "learned")).strip().lower()
        if self.routing_mode not in GlobalOpinionRouter.MODES:
            raise ValueError(
                "fusion.routing.mode must be one of "
                f"{GlobalOpinionRouter.MODES}, got {self.routing_mode!r}"
            )
        routing_enabled = bool(routing_cfg.get("enabled", False))
        self.routing_route_conflict_enabled = bool(
            routing_cfg.get("route_conflict_enabled", True)
        )
        self.routing_risk_conflict_enabled = bool(
            routing_cfg.get("risk_conflict_enabled", True)
        )
        self.routing_risk_mode = str(
            routing_cfg.get("risk_mode", "learned")
        ).strip().lower()
        self.routing_risk_target = str(
            routing_cfg.get("risk_target", "mixture_argmax_error")
        ).strip().lower()
        self.routing_train_end_to_end = bool(
            routing_cfg.get("train_end_to_end", False)
        )
        if routing_enabled and self.routing_train_end_to_end:
            raise ValueError(
                "I2-v2 requires fusion.routing.train_end_to_end=false; pi and u "
                "must be fitted post-hoc with their own proper losses"
            )
        self.routing_posthoc_refine = bool(
            routing_cfg.get("posthoc_refine", True)
        )
        if (
            routing_enabled
            and not self.routing_posthoc_refine
            and (
                self.routing_mode == "learned"
                or self.routing_risk_mode == "learned"
            )
        ):
            raise ValueError(
                "learned I2 components require fusion.routing.posthoc_refine=true"
            )
        self.opinion_router = (
            GlobalOpinionRouter(
                mode=self.routing_mode,
                route_conflict_enabled=self.routing_route_conflict_enabled,
                risk_conflict_enabled=self.routing_risk_conflict_enabled,
                risk_mode=self.routing_risk_mode,
                risk_target=self.routing_risk_target,
                initial_risk=float(routing_cfg.get("initial_risk", 0.10)),
                fixed_prior_beta=float(routing_cfg.get("fixed_prior_beta", 1.0)),
            )
            if routing_enabled
            else None
        )
        # Instantiate the shared router before the optional I1 calibrator. This
        # keeps router initialization identical when the calibrator itself is
        # removed in an I1 ablation; only the ablated module then consumes a
        # different number of RNG draws.
        self.reliability_calibration_method = (
            normalize_reliability_calibration_method(
                reliability_cfg.get("method", MONOTONIC_CORRECTNESS_METHOD)
            )
        )
        reliability_enabled = bool(reliability_cfg.get("enabled", False))
        if (
            reliability_enabled
            and self.reliability_calibration_method
            == TEMPERATURE_SCALING_CONFIDENCE_METHOD
        ):
            unsupported_active_features = [
                name
                for name, default in (
                    ("use_model_visibility", False),
                    ("use_embedding_density", False),
                    ("use_prediction_margin", False),
                    ("use_predicted_class_feature", False),
                )
                if bool(reliability_cfg.get(name, default))
            ]
            if unsupported_active_features:
                raise ValueError(
                    "temperature_scaling_confidence is a raw-logit-only I1 "
                    "baseline; disable unused feature-calibrator switches: "
                    f"{unsupported_active_features}"
                )
            self.reliability_calibrator = (
                BranchTemperatureScalingConfidenceCalibrator(
                    apply_alive_mask=bool(
                        reliability_cfg.get("apply_alive_mask", True)
                    ),
                    initial_temperature=float(
                        reliability_cfg.get("initial_temperature", 1.0)
                    ),
                )
            )
        elif reliability_enabled:
            self.reliability_calibrator = MonotonicReliabilityCalibrator(
                use_model_visibility=bool(
                    reliability_cfg.get("use_model_visibility", False)
                ),
                use_embedding_density=bool(
                    reliability_cfg.get("use_embedding_density", False)
                ),
                use_prediction_margin=bool(
                    reliability_cfg.get("use_prediction_margin", True)
                ),
                use_predicted_class_feature=bool(
                    reliability_cfg.get("use_predicted_class_feature", True)
                ),
                apply_alive_mask=bool(
                    reliability_cfg.get("apply_alive_mask", True)
                ),
                embedding_dims=embedding_dims,
                embedding_density_variance_shrinkage=float(
                    reliability_cfg.get(
                        "embedding_density_variance_shrinkage", 0.10
                    )
                ),
                embedding_density_reference_quantile=float(
                    reliability_cfg.get(
                        "embedding_density_reference_quantile", 0.95
                    )
                ),
                embedding_density_min_class_samples=int(
                    reliability_cfg.get(
                        "embedding_density_min_class_samples", 8
                    )
                ),
            )
        else:
            self.reliability_calibrator = None
        uses_opinion_combination = (
            self.combination == "routed" or self.combination in COMBINATION_RULES
        )
        configured_reliability_branches = reliability_cfg.get(
            "branches", BRANCH_NAMES
        )
        if not isinstance(configured_reliability_branches, (list, tuple)) or not configured_reliability_branches:
            raise ValueError(
                "fusion.reliability_calibration.branches must be a non-empty list"
            )
        reliability_branches = tuple(
            str(name).lower() for name in configured_reliability_branches
        )
        invalid_reliability_branches = [
            name for name in reliability_branches if name not in BRANCH_NAMES
        ]
        if invalid_reliability_branches:
            raise ValueError(
                "fusion.reliability_calibration.branches contains unsupported "
                f"branches: {invalid_reliability_branches}"
            )
        if (
            bool(getattr(self.reliability_calibrator, "use_embedding_density", False))
            and set(reliability_branches) != set(BRANCH_NAMES)
        ):
            raise ValueError(
                "I1 embedding density currently requires all three formal "
                "branches so every cached inference path has a fitted reference"
            )
        if len(set(reliability_branches)) != len(reliability_branches):
            raise ValueError(
                "fusion.reliability_calibration.branches must not contain duplicates"
            )
        self.reliability_calibration_branches = reliability_branches
        if (
            self.reliability_calibrator is not None
            and uses_opinion_combination
        ):
            missing_routed_branches = sorted(
                set(EVIDENCE_BRANCHES) - set(reliability_branches)
            )
            if missing_routed_branches:
                raise ValueError(
                    "opinion fusion requires calibrated reliability branches "
                    f"{list(EVIDENCE_BRANCHES)}; missing {missing_routed_branches}"
                )
        if self.combination == "routed" and self.opinion_router is None:
            raise ValueError(
                "fusion.combination=routed requires fusion.routing.enabled=true"
            )
        self.log_final_temperature = (
            nn.Parameter(torch.zeros(()))
            if uses_opinion_combination
            and bool(routing_cfg.get("final_temperature_scaling", False))
            else None
        )
        self.evidence_activation = str(self.config.get("evidence_activation", "softplus")).lower()
        probability_cfg = self.config.get("probability_calibration", {}) or {}
        if bool(probability_cfg.get("enabled", False)) and uses_opinion_combination:
            raise ValueError(
                "branch probability calibration is unsupported for opinion "
                "fusion because it does not parameterize the opinion path; "
                "use fusion.routing.final_temperature_scaling instead"
            )
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

    def reliability_calibration_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.reliability_calibrator is not None:
            for name in self.reliability_calibration_branches:
                parameters.extend(
                    self.reliability_calibrator.branch_parameters(name)
                )
        return parameters

    def reliability_competence_parameters(self) -> list[nn.Parameter]:
        """Return only I1's clean-competence parameters.

        The proposed calibrator is fitted in two ordered phases.  Keeping this
        boundary at the fusion module prevents the training lifecycle from
        depending on the calibrator's internal attribute layout.  Simpler I1
        baselines (for example per-branch temperature scaling) intentionally
        expose no such split and continue to use
        :meth:`reliability_calibration_parameters`.
        """

        calibrator = self.reliability_calibrator
        branch_parameters = getattr(
            calibrator, "branch_competence_parameters", None
        )
        if not callable(branch_parameters):
            return []
        return [
            parameter
            for name in self.reliability_calibration_branches
            for parameter in branch_parameters(name)
        ]

    def reliability_degradation_parameters(self) -> list[nn.Parameter]:
        """Return only I1's non-negative degradation parameters."""

        calibrator = self.reliability_calibrator
        branch_parameters = getattr(
            calibrator, "branch_degradation_parameters", None
        )
        if not callable(branch_parameters):
            return []
        return [
            parameter
            for name in self.reliability_calibration_branches
            for parameter in branch_parameters(name)
        ]

    def probability_calibration_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.temperature_parameters is not None:
            parameters.extend(self.temperature_parameters.parameters())
        return parameters

    def routing_calibration_parameters(self) -> list[nn.Parameter]:
        return [
            *self.routing_distribution_parameters(),
            *self.routing_risk_parameters(),
        ]

    def routing_distribution_parameters(self) -> list[nn.Parameter]:
        """Parameters of the conditional class router pi."""
        if (
            self.combination == "routed"
            and self.opinion_router is not None
            and self.routing_posthoc_refine
        ):
            return list(self.opinion_router.route_parameters())
        return []

    def routing_risk_parameters(self) -> list[nn.Parameter]:
        """Parameters of the independent configured decision-risk calibrator."""
        if (
            self.combination == "routed"
            and self.opinion_router is not None
            and self.routing_posthoc_refine
        ):
            return list(self.opinion_router.risk_parameters())
        return []

    def set_routing_risk_decision_threshold(
        self,
        raw_log_odds_threshold: float,
    ) -> None:
        if self.opinion_router is None:
            raise ValueError(
                "routing risk decision threshold requires an active router"
            )
        self.opinion_router.set_risk_decision_threshold(
            raw_log_odds_threshold
        )

    def routing_encoder_training_parameters(self) -> list[nn.Parameter]:
        """Return router parameters that participate in encoder training."""
        if (
            self.combination != "routed"
            or self.opinion_router is None
            or not self.routing_train_end_to_end
            or self.routing_mode == "prior_only"
        ):
            return []
        return [
            *self.opinion_router.route_parameters(),
            *self.opinion_router.risk_parameters(),
        ]

    def calibration_parameters(self) -> list[nn.Parameter]:
        return [
            *self.reliability_calibration_parameters(),
            *self.probability_calibration_parameters(),
            *self.routing_calibration_parameters(),
        ]

    def encoder_training_frozen_parameters(self) -> list[nn.Parameter]:
        """Parameters reserved exclusively for independent post-hoc fitting.

        Reliability calibration, branch temperature calibration, and the final
        scalar temperature remain isolated from the encoder-training split.
        The router is included only when ``routing.train_end_to_end=false`` (or
        when ``prior_only`` makes its residual intentionally inactive), which
        lets the existing training loop honor the atomic routing switch without
        special-case optimizer code.
        """
        parameters = [
            *self.reliability_calibration_parameters(),
            *self.probability_calibration_parameters(),
            *self.final_temperature_parameters(),
        ]
        if (
            self.opinion_router is not None
            and (
                not self.routing_train_end_to_end
                or self.routing_mode == "prior_only"
            )
        ):
            parameters.extend(self.opinion_router.parameters())
        return parameters

    def final_temperature_parameters(self) -> list[nn.Parameter]:
        """Return the scalar opinion-output calibration parameter, if enabled."""
        return [] if self.log_final_temperature is None else [self.log_final_temperature]

    def final_temperature(self) -> torch.Tensor | None:
        if self.log_final_temperature is None:
            return None
        return self.log_final_temperature.exp()

    @property
    def calibration_active(self) -> bool:
        return self._calibration_active_shadow

    def set_calibration_active(self, enabled: bool) -> None:
        resolved = bool(enabled)
        self._calibration_active.fill_(resolved)
        self._calibration_active_shadow = resolved

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # State loading is a lifecycle boundary where one scalar device sync is
        # acceptable and necessary to restore the hot-path shadow exactly.
        self._calibration_active_shadow = bool(
            self._calibration_active.detach().item()
        )

    def _temperature(self, name: str, confidence_cfg: dict) -> float | torch.Tensor:
        if self.temperature_parameters is None or not self.calibration_active:
            return float(confidence_cfg.get(f"temperature_{name}", 1.0))
        return F.softplus(self.temperature_parameters[name]) + 1.0e-4

    def forward(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        evidence: torch.Tensor,
        embeddings: dict[str, torch.Tensor] | None = None,
        config: dict | None = None,
        *,
        reliability_override: torch.Tensor | None = None,
        branch_distribution_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Encoders can remain under AMP, but calibration, probability discounting,
        # and rejection scores need FP32 to avoid quantized temperatures/thresholds.
        with torch.autocast(device_type=api_logits.device.type, enabled=False):
            return self._forward_fp32(
                api_logits.float(),
                graph_logits.float(),
                manifest_logits.float(),
                evidence.float(),
                embeddings,
                config,
                None if reliability_override is None else reliability_override.float(),
                (
                    None
                    if branch_distribution_override is None
                    else branch_distribution_override.float()
                ),
            )

    def _forward_evidential_fp32(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        evidence: torch.Tensor,
        embeddings: dict[str, torch.Tensor] | None,
        cfg: dict,
        rule: str,
        reliability_override: torch.Tensor | None = None,
        branch_distribution_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse the three independent modality opinions in FP32.

        The main path freezes the encoders, then fits a conditional branch
        distribution with identity-grouped nested cross-fitting and fits the
        separate configured decision-event risk score from strictly out-of-fold upstream
        predictions. Classical evidential rules remain available only as named
        comparison methods.
        """
        eps = float(cfg.get("min_discount", GateConstants.EPS))
        use_hard_alive = bool(cfg.get("use_hard_alive_mask", True))
        use_reliability = bool(cfg.get("use_reliability_discount", True))
        base_rate = float(cfg.get("base_rate", 0.5))

        logits_by_branch = {
            "api": api_logits,
            "graph": graph_logits,
            "manifest": manifest_logits,
        }
        opinion_source = str(cfg.get("opinion_source", "evidential")).lower()

        if opinion_source == "evidential":
            opinions = {
                name: logits_to_opinion(
                    logits, evidence_activation=self.evidence_activation, eps=eps
                )
                for name, logits in logits_by_branch.items()
            }
        elif opinion_source == "softmax_fixed_uncertainty":
            softmax_cfg = cfg.get("softmax_opinion", {}) or {}
            fixed_u = float(softmax_cfg.get("uncertainty", 0.5))
            temperature = float(softmax_cfg.get("temperature", 1.0))
            opinions = {
                name: logits_to_softmax_opinion(
                    logits,
                    uncertainty=fixed_u,
                    temperature=temperature,
                    eps=eps,
                )
                for name, logits in logits_by_branch.items()
            }
        else:
            raise ValueError(f"Unsupported fusion.opinion_source: {opinion_source}")
        
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
        # Predictive conflict is measured before trust discounting, but a
        # missing modality is first converted to a vacuous opinion. Otherwise
        # arbitrary placeholder logits from an unavailable encoder could create
        # a false disagreement signal.
        predictive_beliefs = []
        predictive_uncertainties = []
        for name in EVIDENCE_BRANCHES:
            alive = alive_by_branch[name]
            predictive_beliefs.append(opinions[name]["belief"] * alive.unsqueeze(-1))
            predictive_uncertainties.append(
                1.0 - alive * (1.0 - opinions[name]["uncertainty"])
            )
        predictive_conflict, predictive_conflict_max = predictive_opinion_conflict(
            predictive_beliefs,
            predictive_uncertainties,
            eps=eps,
        )

        # I1 reliability: the fitted post-hoc calibrator can combine observable
        # evidence with branch-local prediction features. Before that fit,
        # observable integrity remains available for diagnostics/non-routed
        # comparison rules, but the routed encoder-training path uses an
        # alive-only neutral prior inside GlobalOpinionRouter. Raw integrity is
        # not a common-scale correctness probability and must not be logit-routed.
        reliability_outputs: dict[str, torch.Tensor] = {}
        # Strict OOF route/risk fitting supplies already materialized I1 values.
        # Re-running the calibrator here and immediately overwriting its output
        # is both wasteful and, with dozens of scenario views, a dominant source
        # of tiny GPU kernels and validation synchronizations.
        calibrated = (
            reliability_override is None
            and self.reliability_calibrator is not None
            and self.calibration_active
        )
        routing_active = (
            rule == "routed"
            and self.opinion_router is not None
        )
        if branch_distribution_override is not None and not routing_active:
            raise ValueError(
                "branch_distribution_override is only valid for the routed "
                "opinion-combination rule"
            )
        if calibrated:
            branch_probabilities_for_reliability = {
                name: (
                    opinions[name]["belief"]
                    + opinions[name]["uncertainty"].unsqueeze(-1)
                    / float(opinions[name]["belief"].size(-1))
                ).clamp(0.0, 1.0)
                for name in BRANCH_NAMES
            }
            all_reliability_outputs = self.reliability_calibrator(
                evidence,
                branch_probabilities=branch_probabilities_for_reliability,
                branch_logits=logits_by_branch,
                branch_embeddings=embeddings,
            )
            reliability_outputs = dict(all_reliability_outputs)
            observable_reliability = {
                name: all_reliability_outputs[
                    f"predicted_reliability_{name}"
                ].clamp(0.0, 1.0)
                for name in EVIDENCE_BRANCHES
            }
        else:
            observable_reliability = {
                name: integrity_by_branch[name].clamp(0.0, 1.0)
                for name in EVIDENCE_BRANCHES
            }

        reliability_override_active = reliability_override is not None
        if reliability_override_active:
            if not self.calibration_active:
                raise ValueError(
                    "reliability_override is restricted to the active post-hoc lifecycle"
                )
            expected_shape = (evidence.size(0), len(EVIDENCE_BRANCHES))
            if reliability_override.shape != expected_shape:
                raise ValueError(
                    "reliability_override must have shape "
                    f"{expected_shape}, got {tuple(reliability_override.shape)}"
                )
            if not bool(torch.isfinite(reliability_override).all().item()) or bool(
                ((reliability_override < 0.0) | (reliability_override > 1.0)).any().item()
            ):
                raise ValueError(
                    "reliability_override must contain finite values within [0, 1]"
                )
            observable_reliability = {
                name: reliability_override[:, index]
                for index, name in enumerate(EVIDENCE_BRANCHES)
            }
            for name in EVIDENCE_BRANCHES:
                value = observable_reliability[name]
                reliability_outputs[f"predicted_reliability_{name}"] = value
                reliability_outputs[f"predicted_reliability_logit_{name}"] = torch.logit(
                    value.clamp(eps, 1.0 - eps)
                )

        effective_integrity = _effective_integrity_terms(evidence)
        beliefs: list[torch.Tensor] = []
        uncertainties: list[torch.Tensor] = []
        trust_by_branch: dict[str, torch.Tensor] = {}
        routing_reliability: dict[str, torch.Tensor] = {}
        for name in EVIDENCE_BRANCHES:
            reliability = observable_reliability[name]
            if use_reliability:
                reliability = reliability.clamp(0.0, 1.0)
            else:
                reliability = torch.ones_like(reliability)
            # I1 already estimates the operative correctness probability. It
            # is therefore the sole learned trust signal for every fusion rule.
            trust = reliability
            if use_hard_alive:
                # Dead modalities become vacuous opinions (the identity element of
                # the fusion), contributing nothing instead of voting.
                trust = trust * alive_by_branch[name]
            trust_by_branch[name] = trust
            routing_reliability[name] = reliability
            discounted_belief, discounted_u = trust_discount(
                opinions[name]["belief"], opinions[name]["uncertainty"], trust
            )
            beliefs.append(discounted_belief)
            uncertainties.append(discounted_u)

        routing_outputs: dict[str, torch.Tensor] = {}
        if routing_active:
            raw_conflict = multisource_conflict(
                beliefs, uncertainties, eps=eps
            ).clamp(0.0, 1.0)
            # During encoder training there is no fitted I1 probability yet.
            # Pass a common unit value so both route and risk diagnostics are
            # neutral apart from the explicit alive mask. The active post-hoc
            # path keeps the configured calibrated/raw-ablation reliability.
            router_reliability = {
                name: (
                    routing_reliability[name].detach()
                    if self.calibration_active
                    else torch.ones_like(routing_reliability[name])
                )
                for name in EVIDENCE_BRANCHES
            }
            routing_outputs = self.opinion_router(
                beliefs={name: opinions[name]["belief"] for name in EVIDENCE_BRANCHES},
                uncertainties={
                    name: opinions[name]["uncertainty"] for name in EVIDENCE_BRANCHES
                },
                # Reliability supervision and routing supervision remain
                # separate: routing NLL must not distort correctness calibration.
                reliability=router_reliability,
                alive=alive_by_branch,
                learned_active=self.calibration_active,
                branch_distribution_override=branch_distribution_override,
                eps=eps,
            )
            fused_belief = routing_outputs["belief"]
            fused_uncertainty = routing_outputs["uncertainty"]
        else:
            fused_belief, fused_uncertainty, combination_diagnostics = (
                combine_opinions_with_diagnostics(
                    beliefs,
                    uncertainties,
                    rule=rule,
                    availability_masks=(
                        [alive_by_branch[name] for name in EVIDENCE_BRANCHES]
                        if rule == "ecml"
                        else None
                    ),
                    eps=eps,
                )
            )
            raw_conflict = combination_diagnostics["raw_conflict"].clamp(0.0, 1.0)
        fused_alpha = opinion_to_dirichlet_alpha(
            fused_belief, fused_uncertainty, eps=eps
        )
        # I2-v2 keeps the proper targets separate. pi's conditional mixture is
        # the classifier; u is an independently fitted decision-event risk
        # score for I3 rejection. Mapping u back into class probability would make the
        # deployed classifier differ from the one optimized by mixture NLL.
        uncalibrated_final_prob = (
            routing_outputs["mixture_probability"]
            if routing_active
            else opinion_to_prob(
                fused_belief, fused_uncertainty, base_rate=base_rate, eps=eps
            )
        )
        uncalibrated_final_log_prob = torch.log(
            uncalibrated_final_prob.clamp_min(eps)
        )
        final_temperature = (
            self.final_temperature()
            if self.calibration_active
            else None
        )
        if final_temperature is not None:
            final_logits = F.log_softmax(
                uncalibrated_final_log_prob / final_temperature.clamp_min(eps),
                dim=-1,
            )
            final_prob = final_logits.exp()
        else:
            final_logits = uncalibrated_final_log_prob
            final_prob = uncalibrated_final_prob

        if routing_active:
            routed_weights3 = routing_outputs["weights"][:, : len(EVIDENCE_BRANCHES)]
            routed_weight_sum = routed_weights3.sum(dim=-1, keepdim=True)
            weights3 = torch.where(
                routed_weight_sum > eps,
                routed_weights3 / routed_weight_sum.clamp_min(eps),
                torch.zeros_like(routed_weights3),
            )
            actual_routing_weights = routing_outputs["weights"]
        else:
            # Pseudo fusion weights for comparison rules: how much each
            # modality's trusted belief contributed.
            contribution = torch.stack(
                [
                    trust_by_branch[name] * (1.0 - opinions[name]["uncertainty"])
                    for name in EVIDENCE_BRANCHES
                ],
                dim=-1,
            )
            contribution_sum = contribution.sum(dim=-1, keepdim=True).clamp_min(eps)
            weights3 = contribution / contribution_sum
            actual_routing_weights = torch.cat(
                [weights3, torch.zeros_like(weights3[:, :1])], dim=-1
            )
        fusion_weights = weights3

        total_reliability = (weights3 * torch.stack(
            [trust_by_branch[name] for name in EVIDENCE_BRANCHES], dim=-1
        )).sum(dim=-1).clamp(0.0, 1.0)
        routing_risk_probability = (
            routing_outputs["risk_probability"]
            if routing_active
            else fused_uncertainty
        ).clamp(0.0, 1.0)
        routing_risk_logit = (
            routing_outputs["risk_logit"]
            if routing_active
            else torch.logit(
                routing_risk_probability.clamp(eps, 1.0 - eps)
            )
        )
        acceptance_score_fused_risk = (1.0 - routing_risk_probability).clamp(
            0.0, 1.0
        )
        mixture_uncertainty_burden = (
            routing_outputs["risk_uncertainty_burden"]
            if routing_active
            else fused_uncertainty
        ).clamp(0.0, 1.0)
        acceptance_score_mixture_certainty = (
            1.0 - mixture_uncertainty_burden
        ).clamp(0.0, 1.0)
        # ``predictive_conflict`` is computed from the raw (alive-masked)
        # opinions before I1 trust discounting. ``raw_conflict`` denotes
        # conflict after trust discounting. Expose both semantics so
        # an acceptance ablation cannot silently switch between them.
        acceptance_score_pretrust_conflict = (1.0 - predictive_conflict).clamp(
            0.0, 1.0
        )
        acceptance_score_trusted_conflict = (1.0 - raw_conflict).clamp(0.0, 1.0)
        acceptance_score_product = (
            acceptance_score_mixture_certainty * acceptance_score_trusted_conflict
        ).clamp(0.0, 1.0)
        routing_cfg = cfg.get("routing", {}) or {}
        acceptance_score_mode = str(
            routing_cfg.get("acceptance_score_mode", "product")
        ).strip().lower()
        acceptance_scores = {
            "fused_risk": acceptance_score_fused_risk,
            "mixture_certainty": acceptance_score_mixture_certainty,
            "pretrust_conflict": acceptance_score_pretrust_conflict,
            "trusted_conflict": acceptance_score_trusted_conflict,
            "product": acceptance_score_product,
        }
        if acceptance_score_mode not in acceptance_scores:
            raise ValueError(
                "fusion.routing.acceptance_score_mode must be one of "
                f"{sorted(acceptance_scores)}, got {acceptance_score_mode!r}"
            )
        if acceptance_score_mode == "fused_risk" and not routing_active:
            raise ValueError(
                "fusion.routing.acceptance_score_mode=fused_risk requires "
                "fusion.combination=routed with routing enabled"
            )
        acceptance_score = acceptance_scores[acceptance_score_mode]

        batch = final_logits.size(0)
        outputs: dict[str, torch.Tensor] = {
            "final_prob": final_prob,
            "final_logits": final_logits,
            "uncalibrated_final_log_prob": uncalibrated_final_log_prob,
            "final_temperature": torch.full(
                (batch,),
                1.0,
                device=final_logits.device,
                dtype=final_logits.dtype,
            )
            if final_temperature is None
            else final_temperature.expand(batch),
            "final_is_log_probability": True,
            "fusion_weights": fusion_weights,
            # Preserve full, differentiable Dirichlet parameters for adapted
            # TMC/ECML training objectives. Final probabilities alone do not
            # retain evidence strength and cannot reconstruct these losses.
            "dirichlet_alpha_fused": fused_alpha,
            "fused_uncertainty": fused_uncertainty,
            "fused_belief_malware": fused_belief[:, 1] if fused_belief.size(-1) > 1 else fused_belief[:, 0],
            "raw_conflict": raw_conflict,
            "pretrust_conflict": predictive_conflict,
            "trusted_conflict": raw_conflict,
            "predictive_conflict": predictive_conflict,
            "predictive_conflict_max": predictive_conflict_max,
            "total_reliability": total_reliability,
            "acceptance_score": acceptance_score,
            "acceptance_score_fused_risk": acceptance_score_fused_risk,
            "acceptance_score_mixture_certainty": acceptance_score_mixture_certainty,
            "mixture_uncertainty_burden": mixture_uncertainty_burden,
            "acceptance_score_pretrust_conflict": acceptance_score_pretrust_conflict,
            "acceptance_score_trusted_conflict": acceptance_score_trusted_conflict,
            "acceptance_score_product": acceptance_score_product,
            "routing_active": torch.full(
                (batch,),
                float(routing_active),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_mode_learned": torch.full(
                (batch,),
                float(routing_active and self.routing_mode == "learned"),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_mode_prior_only": torch.full(
                (batch,),
                float(routing_active and self.routing_mode == "prior_only"),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_train_end_to_end": torch.full(
                (batch,),
                float(routing_active and self.routing_train_end_to_end),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_posthoc_refine": torch.full(
                (batch,),
                float(routing_active and self.routing_posthoc_refine),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_weights": actual_routing_weights,
            "routing_risk_probability": routing_risk_probability,
            "routing_risk_logit": routing_risk_logit,
            "routing_risk_training_logit": (
                routing_outputs["risk_training_logit"]
                if routing_active
                else routing_risk_logit
            ),
            "routing_weight_risk": routing_risk_probability,
            "routing_risk_mode_learned": (
                routing_outputs["risk_mode_learned"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_mode_reliability_prior": (
                routing_outputs["risk_mode_reliability_prior"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_mode_disabled": (
                routing_outputs["risk_mode_disabled"]
                if routing_active
                else torch.ones_like(routing_risk_probability)
            ),
            "routing_learned_components_active": (
                routing_outputs["learned_components_active"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_has_available": (
                routing_outputs["has_available"]
                if routing_active
                else torch.ones_like(routing_risk_probability)
            ),
            "routing_committed_mass": (1.0 - routing_risk_probability).clamp(0.0, 1.0),
            "routing_mixture_prob": (
                routing_outputs["mixture_probability"]
                if routing_active
                else final_prob
            ),
            "routing_mixture_prob_malware": (
                routing_outputs["mixture_probability"][:, 1]
                if routing_active
                else final_prob[:, 1]
            ),
            "routing_mixture_pred": (
                routing_outputs["mixture_probability"].argmax(dim=-1)
                if routing_active
                else final_prob.argmax(dim=-1)
            ).to(dtype=final_logits.dtype),
            "routing_branch_distribution": (
                routing_outputs["branch_distribution"]
                if routing_active
                else weights3
            ),
            "routing_scores": (
                routing_outputs["routing_scores"]
                if routing_active
                else torch.log(weights3.clamp_min(eps))
            ),
            "routing_prior_branch_distribution": (
                routing_outputs["prior_branch_distribution"]
                if routing_active
                else weights3
            ),
            "routing_route_prior_beta": (
                routing_outputs["route_prior_beta"].expand(batch)
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_prior_only_odds_beta": (
                routing_outputs["prior_only_odds_beta"].expand(batch)
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_prior_only_odds_beta_active": (
                routing_outputs["prior_only_odds_beta_active"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_reliability_deficit": (
                routing_outputs["risk_reliability_deficit"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_uncertainty_burden": (
                routing_outputs["risk_uncertainty_burden"]
                if routing_active
                else fused_uncertainty
            ),
            "routing_risk_decision_boundary_proximity": (
                routing_outputs["risk_decision_boundary_proximity"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_predicted_malware": (
                routing_outputs["risk_predicted_malware"]
                if routing_active
                else final_prob.argmax(dim=-1).to(final_prob.dtype)
            ),
            "routing_risk_decision_log_odds_threshold": (
                routing_outputs["risk_decision_log_odds_threshold"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_decision_threshold_active": (
                routing_outputs["risk_decision_threshold_active"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_risk_target_mixture_argmax_error": torch.full_like(
                routing_risk_probability,
                float(self.routing_risk_target == "mixture_argmax_error"),
            ),
            "routing_risk_target_threshold_classification_error": torch.full_like(
                routing_risk_probability,
                float(
                    self.routing_risk_target
                    == "threshold_classification_error"
                ),
            ),
            "routing_risk_target_threshold_malware_false_negative": torch.full_like(
                routing_risk_probability,
                float(
                    self.routing_risk_target
                    == "threshold_malware_false_negative"
                ),
            ),
            "routing_risk_target_reliability_deficit_score": torch.full_like(
                routing_risk_probability,
                float(self.routing_risk_target == "reliability_deficit_score"),
            ),
            "routing_risk_structural_conflict": (
                routing_outputs["risk_structural_conflict"]
                if routing_active
                else predictive_conflict
            ),
            "routing_risk_missing_fraction": (
                routing_outputs["risk_missing_fraction"]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_conflict_penalty_mean": (
                routing_outputs["conflict_penalty"].mean(dim=-1)
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_route_conflict_feature_active": (
                routing_outputs["route_conflict_feature_active"]
                if routing_active
                else torch.zeros_like(actual_routing_weights[:, -1])
            ),
            "routing_route_conflict_feature_configured": (
                routing_outputs["route_conflict_feature_configured"]
                if routing_active
                else torch.zeros_like(actual_routing_weights[:, -1])
            ),
            "routing_risk_conflict_feature_active": (
                routing_outputs["risk_conflict_feature_active"]
                if routing_active
                else torch.zeros_like(actual_routing_weights[:, -1])
            ),
            "routing_risk_conflict_feature_configured": (
                routing_outputs["risk_conflict_feature_configured"]
                if routing_active
                else torch.zeros_like(actual_routing_weights[:, -1])
            ),
            "routing_mean_disagreement": (
                routing_outputs["mean_disagreement"]
                if routing_active
                else predictive_conflict
            ),
            "routing_common_scale_reliability_active": torch.full(
                (batch,),
                float(routing_active and calibrated and use_reliability),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_prefit_uniform_prior_active": (
                routing_outputs["prefit_uniform_prior_active"]
                if routing_active
                else torch.zeros_like(actual_routing_weights[:, -1])
            ),
            "reliability_override_active": torch.full(
                (batch,),
                float(reliability_override_active),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "combination_rule_is_evidential": torch.ones((batch,), device=final_logits.device, dtype=final_logits.dtype),
            "calibration_active": torch.full(
                (batch,), float(self.calibration_active), device=final_logits.device, dtype=final_logits.dtype
            ),
        }
        outputs.update(reliability_outputs)
        for name in BRANCH_NAMES:
            if name not in self.reliability_calibration_branches:
                # An unfitted branch must not be exported as a calibrated
                # correctness probability.
                outputs.pop(f"predicted_reliability_{name}", None)
                outputs.pop(f"predicted_reliability_logit_{name}", None)
        for index, name in enumerate(BRANCH_NAMES):
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            outputs[f"routing_weight_{name}"] = (
                actual_routing_weights[:, index]
                if name in EVIDENCE_BRANCHES
                else torch.zeros_like(actual_routing_weights[:, 0])
            )
            outputs[f"uncertainty_proxy_{name}"] = opinions[name]["uncertainty"]
            if routing_active:
                # Exact, branch-local inputs to the conditional router.  The
                # post-hoc route optimizer caches these tensors once because
                # encoder logits, I1 reliability and availability are frozen
                # throughout that stage.  Exporting the actual inputs avoids
                # reconstructing subjective opinions from rounded diagnostics
                # or duplicating the reliability policy in train.py.
                outputs[f"routing_input_belief_{name}"] = opinions[name]["belief"]
                outputs[f"routing_input_uncertainty_{name}"] = opinions[name][
                    "uncertainty"
                ]
                outputs[f"routing_input_reliability_{name}"] = router_reliability[
                    name
                ]
                outputs[f"routing_input_alive_{name}"] = alive_by_branch[name]
            outputs[f"evidential_certainty_{name}"] = evidential_certainty[name]
            if name in effective_integrity:
                outputs[f"effective_{name}_integrity"] = effective_integrity[name]
            outputs[f"calibrated_log_prob_{name}"] = torch.log(
                opinions[name]["expected_prob"].clamp_min(eps)
            )
            if name in EVIDENCE_BRANCHES:
                outputs[f"dirichlet_alpha_{name}"] = opinions[name]["alpha"]
            if name in trust_by_branch:
                if calibrated and name in self.reliability_calibration_branches:
                    outputs[f"clean_correctness_probability_{name}"] = (
                        observable_reliability[name]
                    )
                outputs[f"effective_trust_cap_{name}"] = trust_by_branch[name]
                # Historical operational-trust alias.
                outputs[f"discount_{name}"] = trust_by_branch[name]
        return outputs

    def _forward_fp32(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        evidence: torch.Tensor,
        embeddings: dict[str, torch.Tensor] | None = None,
        config: dict | None = None,
        reliability_override: torch.Tensor | None = None,
        branch_distribution_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = dict(self.config)
        cfg.update(config or {})
        if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
            raise ValueError(
                f"DiscountProbabilityFusion expected [B, >= {EvidenceIndex.BASE_DIM}] evidence, "
                f"got {tuple(evidence.shape)}"
            )

        combination = str(cfg.get("combination", self.combination)).lower()
        if combination == "routed" or combination in COMBINATION_RULES:
            return self._forward_evidential_fp32(
                api_logits,
                graph_logits,
                manifest_logits,
                evidence,
                embeddings,
                cfg,
                combination,
                reliability_override,
                branch_distribution_override,
            )

        if reliability_override is not None or branch_distribution_override is not None:
            raise ValueError(
                "post-hoc reliability/route overrides require fusion.combination=routed"
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
        if "linear_use_joint_branch" in cfg:
            raise ValueError(
                "fusion.linear_use_joint_branch was removed; linear fusion now "
                "always uses exactly API, Graph, and Manifest"
            )
        weight_gamma = float(cfg.get("weight_sharpening_gamma", 1.0))
        if not math.isfinite(weight_gamma) or weight_gamma <= 0.0:
            raise ValueError("fusion.weight_sharpening_gamma must be finite and positive")

        confidence_cfg = cfg.get("confidence_proxy", {}) or {}
        reliability_cfg = cfg.get("reliability_calibration", {}) or {}
        logits_by_branch = (api_logits, graph_logits, manifest_logits)

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
        any_alive = alive_api | alive_graph | alive_manifest
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
        reliability_outputs: dict[str, torch.Tensor] = {}
        observable_reliability_fallback = [
            api_integrity,
            graph_integrity,
            manifest_integrity,
        ]
        if calibrated_reliability_active:
            reliability_outputs = self.reliability_calibrator(
                evidence,
                branch_probabilities={
                    name: proxies[name]["prob"] for name in BRANCH_NAMES
                },
                branch_logits={
                    name: logits for name, logits in zip(BRANCH_NAMES, logits_by_branch)
                },
                branch_embeddings=embeddings,
            )
            base_reliability = [
                (
                    reliability_outputs[f"predicted_reliability_{name}"]
                    if name in self.reliability_calibration_branches
                    else observable_reliability_fallback[index]
                )
                for index, name in enumerate(BRANCH_NAMES)
            ]
        else:
            features, feature_diagnostics = build_monotonic_reliability_features(
                evidence,
                use_model_visibility=bool(
                    reliability_cfg.get("use_model_visibility", False)
                ),
                # Encoder training has no fitted fold-local embedding
                # reference. The fixed topology therefore exposes a neutral
                # zero slot until the post-hoc lifecycle becomes active.
                use_embedding_density=False,
                use_prediction_margin=bool(
                    reliability_cfg.get("use_prediction_margin", True)
                ),
                use_predicted_class_feature=bool(
                    reliability_cfg.get("use_predicted_class_feature", True)
                ),
                branch_probabilities={
                    name: proxies[name]["prob"] for name in BRANCH_NAMES
                },
                branch_logits={
                    name: logits
                    for name, logits in zip(BRANCH_NAMES, logits_by_branch)
                },
            )
            reliability_outputs.update(feature_diagnostics)
            for name, value in features.items():
                reliability_outputs[f"reliability_features_{name}"] = value
            base_reliability = observable_reliability_fallback
        estimated_reliability = [
            value.clamp(0.0, 1.0) for value in base_reliability
        ]
        if use_reliability:
            reliability_for_fusion = [
                (
                    value
                    if calibrated_reliability_active
                    else value.pow(reliability_exponent)
                ).clamp(0.0, 1.0)
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
        effective_integrity = _effective_integrity_terms(evidence)
        raw_discounts = torch.stack(
            [
                reliability_for_fusion[0] * code_anchor_factor * code_conflict_factor * confidence_factors[0],
                reliability_for_fusion[1] * code_anchor_factor * code_conflict_factor * confidence_factors[1],
                reliability_for_fusion[2] * manifest_support_factor * manifest_conflict_factor * confidence_factors[2],
            ],
            dim=-1,
        )
        alive_mask = torch.stack([alive_api, alive_graph, alive_manifest], dim=-1)
        discounts = raw_discounts * alive_mask.to(raw_discounts.dtype) if use_hard_alive else raw_discounts
        discounts_for_weight = discounts.detach() if detach_discount else discounts
        if use_hard_alive:
            discounts_for_weight = discounts_for_weight * alive_mask.to(discounts_for_weight.dtype)
        if weight_gamma != 1.0:
            discounts_for_weight = discounts_for_weight.clamp_min(0.0).pow(weight_gamma)

        valid_sum = discounts_for_weight.sum(dim=-1, keepdim=True)
        zero_weight_fallback_used = valid_sum <= eps
        fallback = str(cfg.get("fallback", "uniform")).lower()
        if fallback != "uniform":
            raise ValueError(f"Unsupported discount fusion fallback: {fallback}")
        fallback_weights = torch.full_like(
            discounts_for_weight, 1.0 / len(BRANCH_NAMES)
        )
        if use_hard_alive:
            alive_fallback = alive_mask.to(discounts_for_weight.dtype)
            alive_count = alive_fallback.sum(dim=-1, keepdim=True)
            fallback_weights = torch.where(
                alive_count > 0,
                alive_fallback / alive_count.clamp_min(1.0),
                fallback_weights,
            )
        fusion_weights = torch.where(
            zero_weight_fallback_used,
            fallback_weights,
            discounts_for_weight / valid_sum.clamp_min(eps),
        )

        branch_prob = torch.stack([proxies[name]["prob"] for name in BRANCH_NAMES], dim=1)
        final_prob = (fusion_weights.unsqueeze(-1) * branch_prob).sum(dim=1)
        final_prob = final_prob / final_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
        # Placeholder branch logits are implementation details, not evidence.
        # When every primary modality is unavailable, expose an explicit
        # maximum-uncertainty class distribution instead of averaging those
        # arbitrary placeholders through the fallback weights.
        all_modalities_dead = ~any_alive
        uniform_prob = torch.full_like(
            final_prob,
            1.0 / float(final_prob.size(-1)),
        )
        final_prob = torch.where(
            all_modalities_dead.view(-1, 1),
            uniform_prob,
            final_prob,
        )
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
            "zero_weight_fallback_used": zero_weight_fallback_used.to(
                dtype=discounts.dtype
            ).view(-1),
            "final_prob": final_prob,
            "final_logits": final_logits,
            "final_is_log_probability": True,
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
            "weight_sharpening_gamma": torch.full(
                (final_logits.size(0),),
                weight_gamma,
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "api_graph_support_applicable": api_graph_applicable.to(dtype=discounts.dtype),
            "manifest_code_conflict_applicable": manifest_code_applicable.to(dtype=discounts.dtype),
            "manifest_code_relation_applicable": manifest_code_applicable.to(dtype=discounts.dtype),
        }
        outputs.update(reliability_outputs)
        for name in BRANCH_NAMES:
            if name not in self.reliability_calibration_branches:
                outputs.pop(f"predicted_reliability_{name}", None)
                outputs.pop(f"predicted_reliability_logit_{name}", None)
        for index, name in enumerate(BRANCH_NAMES):
            if (
                calibrated_reliability_active
                and name in self.reliability_calibration_branches
            ):
                outputs[f"clean_correctness_probability_{name}"] = (
                    estimated_reliability[index]
                )
            outputs[f"effective_trust_cap_{name}"] = discounts[:, index]
            # Historical operational-trust alias.
            outputs[f"discount_{name}"] = discounts[:, index]
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            if name in effective_integrity:
                outputs[f"effective_{name}_integrity"] = effective_integrity[name]
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
