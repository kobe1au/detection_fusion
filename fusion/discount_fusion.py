from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import AvailabilityIndex, GateConstants
from fusion.temperature import bounded_final_temperature
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
    build_reliability_features,
    normalize_reliability_calibration_method,
)


def _availability_column(
    availability: torch.Tensor, index: int
) -> torch.Tensor:
    return availability[:, index].clamp(0.0, 1.0)


def _validate_binary_availability(availability: torch.Tensor) -> None:
    valid = (
        torch.isfinite(availability)
        & ((availability == 0.0) | (availability == 1.0))
    ).all()
    message = (
        "DiscountProbabilityFusion availability must contain only finite "
        "binary values (0 or 1)"
    )
    if availability.device.type == "cpu":
        if not bool(valid.item()):
            raise ValueError(message)
    else:
        torch._assert_async(valid, message)


class DiscountProbabilityFusion(nn.Module):
    """Calibrated monotonic probability discount fusion with rejection support."""

    def __init__(
        self,
        config: dict | None = None,
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
        removed_linear_keys = sorted(
            set(self.config)
            & {
                "confidence_proxy",
                "conflict_factor",
                "detach_confidence_proxy",
                "detach_discount",
                "fallback",
                "linear_use_joint_branch",
                "reliability_discount_exponent",
                "support_factor",
                "use_confidence_proxy",
                "use_conflict_discount",
                "use_support_discount",
                "weight_sharpening_gamma",
            }
        )
        if removed_linear_keys:
            raise ValueError(
                "The legacy linear discount-probability path was removed; "
                "delete its obsolete fusion keys: "
                f"{removed_linear_keys}"
            )
        if "use_reliability_discount" in self.config:
            raise ValueError(
                "Removed key fusion.use_reliability_discount is no longer "
                "accepted; use fusion.use_i1_reliability"
            )
        if "probability_calibration" in self.config:
            raise ValueError(
                "fusion.probability_calibration was removed; use the I1 "
                "temperature-scaling confidence comparator or "
                "fusion.routing.final_temperature_scaling as appropriate"
            )
        reliability_cfg = self.config.get("reliability_calibration", {}) or {}
        removed_reliability_keys = sorted(
            set(reliability_cfg)
            & {
                "use_model_visibility",
                "use_predicted_class_feature",
                "degradation_conditioning",
                "degradation_min_rows_per_predicted_class",
                "degradation_require_both_correctness_outcomes",
                "objective_weights",
                "require_all_objective_families",
                "feature_schema",
                "missing_relation_support",
                "use_relation_evidence",
                "use_edl_certainty_feature",
                "use_evidential_uncertainty",
                "group_mean_alignment",
                "apply_alive_mask",
                "weight",
            }
        )
        if removed_reliability_keys:
            raise ValueError(
                "Removed I1 reliability keys are unsupported: "
                f"{removed_reliability_keys}"
            )
        if "combination" not in self.config:
            raise ValueError(
                "fusion.combination is required for discount_probability "
                "fusion; choose 'routed' or an explicit evidential rule"
            )
        combination = str(self.config["combination"]).lower()
        if combination != "routed" and combination not in COMBINATION_RULES:
            raise ValueError(
                "fusion.combination must be 'routed' or one of "
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
                "acceptance_score_mode",
                "hidden_dim",
                "train_end_to_end",
                "calibration_weight",
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
        self.i1_reliability_input_enabled = bool(
            self.config.get("use_i1_reliability", True)
        )
        self.routing_risk_conflict_enabled = bool(
            routing_cfg.get("risk_conflict_enabled", True)
        )
        self.routing_risk_mode = str(
            routing_cfg.get("risk_mode", "learned")
        ).strip().lower()
        self.routing_risk_target = str(
            routing_cfg.get("risk_target", "threshold_malware_false_negative")
        ).strip().lower()
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
                reliability_input_enabled=(
                    self.i1_reliability_input_enabled
                ),
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
        removed_i1_density_keys = sorted(
            key
            for key in (
                "use_embedding_density",
                "embedding_dims",
                "embedding_density_variance_shrinkage",
                "embedding_density_reference_quantile",
                "embedding_density_min_class_samples",
            )
            if key in reliability_cfg
        )
        if removed_i1_density_keys:
            raise ValueError(
                "I1 no longer consumes encoder-space embedding density/tail "
                "signals; remove obsolete reliability_calibration keys: "
                f"{removed_i1_density_keys}"
            )
        reliability_enabled = bool(reliability_cfg.get("enabled", False))
        if (
            reliability_enabled
            and self.reliability_calibration_method
            == TEMPERATURE_SCALING_CONFIDENCE_METHOD
        ):
            self.reliability_calibrator = (
                BranchTemperatureScalingConfidenceCalibrator(
                    initial_temperature=float(
                        reliability_cfg.get("initial_temperature", 1.0)
                    ),
                )
            )
        elif reliability_enabled:
            self.reliability_calibrator = MonotonicReliabilityCalibrator(
                use_evidential_certainty=bool(
                    reliability_cfg.get("use_evidential_certainty", True)
                ),
                use_prediction_margin=bool(
                    reliability_cfg.get("use_prediction_margin", True)
                ),
                use_predicted_class_intercept=bool(
                    reliability_cfg.get(
                        "use_predicted_class_intercept", True
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
        # self.log_final_temperature = (
        #     nn.Parameter(torch.zeros(()))
        #     if uses_opinion_combination
        #     and bool(routing_cfg.get("final_temperature_scaling", False))
        #     else None
        # )
        self.log_final_temperature = (
            nn.Parameter(torch.zeros(()))
            if uses_opinion_combination
            and bool(routing_cfg.get("final_temperature_scaling", False))
            else None
        )
        self.evidence_activation = str(self.config.get("evidence_activation", "softplus")).lower()
    def reliability_calibration_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.reliability_calibrator is not None:
            for name in self.reliability_calibration_branches:
                parameters.extend(
                    self.reliability_calibrator.branch_parameters(name)
                )
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
        """I1/I2 are always isolated from clean Stage-1 encoder training."""
        return []

    def calibration_parameters(self) -> list[nn.Parameter]:
        return [
            *self.reliability_calibration_parameters(),
            *self.routing_calibration_parameters(),
        ]

    def encoder_training_frozen_parameters(self) -> list[nn.Parameter]:
        """Parameters reserved exclusively for independent post-hoc fitting.

        Reliability calibration and the final scalar temperature remain
        isolated from the encoder-training split. The router is always
        included because I1/I2 are fitted only after the clean Stage-1
        checkpoint has been selected.
        """
        parameters = [
            *self.reliability_calibration_parameters(),
            *self.final_temperature_parameters(),
        ]
        if self.opinion_router is not None:
            parameters.extend(self.opinion_router.parameters())
        return parameters

    def final_temperature_parameters(self) -> list[nn.Parameter]:
        """Return the scalar opinion-output calibration parameter, if enabled."""
        return [] if self.log_final_temperature is None else [self.log_final_temperature]

    # def final_temperature(self) -> torch.Tensor | None:
    #     if self.log_final_temperature is None:
    #         return None
    #     return self.log_final_temperature.exp()
    def final_temperature(self) -> torch.Tensor | None:
        if self.log_final_temperature is None:
            return None
        return bounded_final_temperature(self.log_final_temperature)

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

    def forward(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        availability: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
        branch_distribution_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if (
            availability.ndim != 2
            or availability.size(-1) != AvailabilityIndex.BASE_DIM
        ):
            raise ValueError(
                "DiscountProbabilityFusion expected exact binary fusion "
                f"availability [B, {AvailabilityIndex.BASE_DIM}], got "
                f"{tuple(availability.shape)}"
            )
        _validate_binary_availability(availability)
        combination = str(cfg.get("combination", self.combination)).lower()
        if combination != "routed" and combination not in COMBINATION_RULES:
            raise ValueError(
                "fusion.combination must be 'routed' or one of "
                f"{COMBINATION_RULES}, got {combination}"
            )
        # Encoders can remain under AMP, but opinion fusion, calibration and
        # rejection scores need FP32.
        with torch.autocast(device_type=api_logits.device.type, enabled=False):
            return self._forward_evidential_fp32(
                api_logits.float(),
                graph_logits.float(),
                manifest_logits.float(),
                availability.float(),
                cfg,
                combination,
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
        availability: torch.Tensor,
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
        use_i1_reliability = bool(cfg.get("use_i1_reliability", True))
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
        
        alive_by_branch = {
            "api": _availability_column(
                availability, AvailabilityIndex.API_ALIVE
            ),
            "graph": _availability_column(
                availability, AvailabilityIndex.GRAPH_ALIVE
            ),
            "manifest": _availability_column(
                availability, AvailabilityIndex.MANIFEST_ALIVE
            ),
        }
        reliability_features = build_reliability_features(
            {name: opinions[name]["alpha"] for name in BRANCH_NAMES}
        )
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

        # I1 has exactly one branch-local input tensor: evidential certainty,
        # decision margin and predicted-class indicator. Extraction integrity,
        # perturbation metadata and cross-modal information never enter it.
        reliability_outputs: dict[str, torch.Tensor] = {
            f"reliability_features_{name}": reliability_features[name]
            for name in BRANCH_NAMES
        }
        for name in BRANCH_NAMES:
            reliability_outputs[f"evidential_certainty_{name}"] = (
                reliability_features[name][:, 0]
            )
            reliability_outputs[f"prediction_margin_{name}"] = (
                reliability_features[name][:, 1]
            )
            reliability_outputs[f"predicted_malware_indicator_{name}"] = (
                reliability_features[name][:, 2]
            )
            reliability_outputs[f"alive_{name}"] = alive_by_branch[name]
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
            if (
                self.reliability_calibration_method
                == TEMPERATURE_SCALING_CONFIDENCE_METHOD
            ):
                all_reliability_outputs = self.reliability_calibrator(
                    logits_by_branch,
                    alive=alive_by_branch,
                )
            else:
                all_reliability_outputs = self.reliability_calibrator(
                    reliability_features,
                    alive=alive_by_branch,
                )
            reliability_outputs.update(all_reliability_outputs)
            branch_reliability = {
                name: all_reliability_outputs[
                    f"predicted_reliability_{name}"
                ].clamp(0.0, 1.0)
                for name in EVIDENCE_BRANCHES
            }
        else:
            branch_reliability = {
                name: torch.ones_like(alive_by_branch[name])
                for name in EVIDENCE_BRANCHES
            }

        reliability_override_active = reliability_override is not None
        if reliability_override_active:
            if not self.calibration_active:
                raise ValueError(
                    "reliability_override is restricted to the active post-hoc lifecycle"
                )
            expected_shape = (availability.size(0), len(EVIDENCE_BRANCHES))
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
            branch_reliability = {
                name: reliability_override[:, index]
                for index, name in enumerate(EVIDENCE_BRANCHES)
            }
            for name in EVIDENCE_BRANCHES:
                value = branch_reliability[name]
                reliability_outputs[f"predicted_reliability_{name}"] = value
                reliability_outputs[f"predicted_reliability_logit_{name}"] = torch.logit(
                    value.clamp(eps, 1.0 - eps)
                )

        beliefs: list[torch.Tensor] = []
        uncertainties: list[torch.Tensor] = []
        trust_by_branch: dict[str, torch.Tensor] = {}
        routing_reliability: dict[str, torch.Tensor] = {}
        for name in EVIDENCE_BRANCHES:
            reliability = branch_reliability[name]
            if use_i1_reliability:
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
                branch_probabilities={
                    name: opinions[name]["expected_prob"]
                    for name in EVIDENCE_BRANCHES
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
            routed_weights3 = routing_outputs["branch_distribution"]
            routed_weight_sum = routed_weights3.sum(dim=-1, keepdim=True)
            weights3 = torch.where(
                routed_weight_sum > eps,
                routed_weights3 / routed_weight_sum.clamp_min(eps),
                torch.zeros_like(routed_weights3),
            )
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
        routed_acceptance_score = (1.0 - routing_risk_probability).clamp(
            0.0, 1.0
        )
        mixture_uncertainty_burden = (
            (
                weights3
                * torch.stack(
                    [
                        opinions[name]["uncertainty"]
                        for name in EVIDENCE_BRANCHES
                    ],
                    dim=-1,
                )
            ).sum(dim=-1)
            if routing_active
            else fused_uncertainty
        ).clamp(0.0, 1.0)
        acceptance_score_mixture_certainty = (
            1.0 - mixture_uncertainty_burden
        ).clamp(0.0, 1.0)
        # I3's proposed score is fixed by the method definition: one minus the
        # threshold-aligned FN risk. Classical comparison rules have no learned
        # FN head, so their generic model-acceptance diagnostic is evidential
        # mixture certainty; MSP/entropy baselines are selected by evaluation.
        acceptance_score = (
            routed_acceptance_score
            if routing_active
            else acceptance_score_mixture_certainty
        )

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
            "raw_conflict": raw_conflict,
            "predictive_conflict": predictive_conflict,
            "predictive_conflict_max": predictive_conflict_max,
            "total_reliability": total_reliability,
            "acceptance_score": acceptance_score,
            "acceptance_score_mixture_certainty": acceptance_score_mixture_certainty,
            "mixture_uncertainty_burden": mixture_uncertainty_burden,
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
            "routing_posthoc_refine": torch.full(
                (batch,),
                float(routing_active and self.routing_posthoc_refine),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_risk_probability": routing_risk_probability,
            "routing_risk_logit": routing_risk_logit,
            "routing_risk_training_logit": (
                routing_outputs["risk_training_logit"]
                if routing_active
                else routing_risk_logit
            ),
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
            "routing_risk_target_threshold_malware_false_negative": torch.full_like(
                routing_risk_probability,
                float(
                    self.routing_risk_target
                    == "threshold_malware_false_negative"
                ),
            ),
            "routing_risk_global_cross_modal_conflict": (
                routing_outputs["risk_global_cross_modal_conflict"]
                if routing_active
                else predictive_conflict
            ),
            "routing_conflict_penalty_mean": (
                routing_outputs["conflict_penalty"].mean(dim=-1)
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            ),
            "routing_route_conflict_feature_active": (
                routing_outputs["route_conflict_feature_active"]
                if routing_active
                else torch.zeros_like(fusion_weights[:, -1])
            ),
            "routing_route_conflict_feature_configured": (
                routing_outputs["route_conflict_feature_configured"]
                if routing_active
                else torch.zeros_like(fusion_weights[:, -1])
            ),
            "routing_risk_conflict_feature_active": (
                routing_outputs["risk_conflict_feature_active"]
                if routing_active
                else torch.zeros_like(fusion_weights[:, -1])
            ),
            "routing_risk_conflict_feature_configured": (
                routing_outputs["risk_conflict_feature_configured"]
                if routing_active
                else torch.zeros_like(fusion_weights[:, -1])
            ),
            "routing_common_scale_reliability_active": torch.full(
                (batch,),
                float(
                    routing_active
                    and use_i1_reliability
                    and (calibrated or reliability_override_active)
                ),
                device=final_logits.device,
                dtype=final_logits.dtype,
            ),
            "routing_prefit_uniform_prior_active": (
                routing_outputs["prefit_uniform_prior_active"]
                if routing_active
                else torch.zeros_like(fusion_weights[:, -1])
            ),
            "calibration_active": torch.full(
                (batch,), float(self.calibration_active), device=final_logits.device, dtype=final_logits.dtype
            ),
        }
        for branch_index, name in enumerate(BRANCH_NAMES):
            outputs[f"routing_cross_modal_conflict_{name}"] = (
                routing_outputs[
                    "reliability_weighted_cross_modal_conflict"
                ][:, branch_index]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            )
            outputs[f"routing_conflict_penalty_{name}"] = (
                routing_outputs["conflict_penalty"][:, branch_index]
                if routing_active
                else torch.zeros_like(routing_risk_probability)
            )
        outputs.update(reliability_outputs)
        for name in BRANCH_NAMES:
            if name not in self.reliability_calibration_branches:
                # An unfitted branch must not be exported as a calibrated
                # correctness probability.
                outputs.pop(f"predicted_reliability_{name}", None)
                outputs.pop(f"predicted_reliability_logit_{name}", None)
        for index, name in enumerate(BRANCH_NAMES):
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            outputs[f"uncertainty_proxy_{name}"] = opinions[name]["uncertainty"]
            if routing_active:
                # Exact, frozen I2 inputs used by the post-hoc route optimizer.
                outputs[f"routing_input_probability_{name}"] = opinions[name][
                    "expected_prob"
                ]
                outputs[f"routing_input_reliability_{name}"] = router_reliability[
                    name
                ]
                outputs[f"routing_input_alive_{name}"] = alive_by_branch[name]
            if name in EVIDENCE_BRANCHES:
                outputs[f"dirichlet_alpha_{name}"] = opinions[name]["alpha"]
        return outputs
