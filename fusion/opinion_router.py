from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTING_BRANCHES = ("api", "graph", "manifest")
RISK_FEATURE_NAMES = (
    "reliability_deficit",
    "decision_boundary_proximity",
    "global_cross_modal_conflict",
)

RISK_TARGETS = ("threshold_malware_false_negative",)


def threshold_aligned_fn_risk_state(
    raw_log_odds: torch.Tensor,
    raw_log_odds_threshold: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the deployed class side and benign-side boundary proximity.

    ``raw_log_odds_threshold`` may be the scalar deployment cutoff used at
    inference or one cutoff per row during fold-excluded risk-head fitting.
    Keeping this transformation shared prevents the I2 target, training
    feature, and deployed gating rule from drifting apart.
    """
    if not isinstance(raw_log_odds, torch.Tensor) or raw_log_odds.ndim != 1:
        raise ValueError("raw_log_odds must be a rank-one tensor")
    threshold = torch.as_tensor(
        raw_log_odds_threshold,
        device=raw_log_odds.device,
        dtype=raw_log_odds.dtype,
    )
    if threshold.ndim == 0:
        threshold = threshold.expand_as(raw_log_odds)
    elif threshold.shape != raw_log_odds.shape:
        raise ValueError(
            "raw_log_odds_threshold must be scalar or match raw_log_odds"
        )
    predicted_malware = raw_log_odds >= threshold
    # Conditional malware-FN risk is defined only on the predicted-benign
    # side. Its boundary feature rises monotonically from zero towards one as
    # a benign decision approaches the fitted malware cutoff.
    decision_boundary_proximity = torch.where(
        predicted_malware,
        torch.zeros_like(raw_log_odds),
        (2.0 * torch.sigmoid(raw_log_odds - threshold)).clamp(0.0, 1.0),
    )
    return predicted_malware, decision_boundary_proximity


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive")
    return math.log(math.expm1(value))


class GlobalOpinionRouter(nn.Module):
    """Reliability-log-odds router with an independent decision-risk head.

    The class router ``pi`` and the decision-event risk score ``u`` have different
    statistical targets and are deliberately parameterised separately. Before
    post-hoc fitting, ``pi`` is uniform over alive branches because hard
    availability is not a calibrated correctness probability:

    * ``pi`` is trained only by conditional-mixture NLL.
    * ``u`` is trained by BCE/Brier against its explicitly configured event.

    The class-routing score is deliberately restricted to
    ``beta * logit(reliability_m)`` over alive branches. Cross-modal conflict
    does not identify which branch is wrong, so it cannot change branch
    weights. It remains an input to the independent monotone risk head, whose
    three fixed features are routed reliability deficit, proximity to the
    deployed classification boundary, and global cross-modal conflict. Raw
    evidential uncertainty and missing fractions are intentionally not router
    inputs.
    """

    MODES = ("learned", "prior_only")

    def __init__(
        self,
        *,
        mode: str = "learned",
        risk_conflict_enabled: bool = True,
        risk_mode: str = "learned",
        risk_target: str = "threshold_malware_false_negative",
        initial_risk: float = 0.10,
        fixed_prior_beta: float = 1.0,
        reliability_input_enabled: bool = True,
    ):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError(f"routing mode must be one of {self.MODES}, got {mode!r}")
        if not 0.0 < float(initial_risk) < 1.0:
            raise ValueError("routing initial_risk must be within (0, 1)")
        self.mode = mode
        # This flag is owned by the fusion-level I1 ablation. When disabled,
        # reliability must disappear from every I2 path (prior, peer consensus,
        # and risk), rather than being replaced by an unidentifiable constant
        # coefficient that is still optimized.
        self.reliability_input_enabled = bool(reliability_input_enabled)
        fixed_prior_beta = float(fixed_prior_beta)
        if not math.isfinite(fixed_prior_beta) or fixed_prior_beta <= 0.0:
            raise ValueError("routing fixed_prior_beta must be finite and positive")
        if mode != "prior_only" and fixed_prior_beta != 1.0:
            raise ValueError(
                "routing fixed_prior_beta is a prior_only sensitivity; learned "
                "routing fits its own positive beta and requires fixed_prior_beta=1"
            )
        self.register_buffer(
            "_fixed_prior_beta",
            torch.tensor(fixed_prior_beta, dtype=torch.float32),
        )
        risk_mode = str(risk_mode).strip().lower()
        if risk_mode not in {"learned", "reliability_prior", "disabled"}:
            raise ValueError(
                "routing risk_mode must be learned, reliability_prior, or disabled"
            )
        self.risk_conflict_enabled = bool(risk_conflict_enabled)
        self.risk_mode = risk_mode
        risk_target = str(risk_target).strip().lower()
        if risk_target not in RISK_TARGETS:
            raise ValueError(
                f"routing risk_target must be one of {RISK_TARGETS}, "
                f"got {risk_target!r}"
            )
        self.risk_target = risk_target
        self.register_buffer(
            "_risk_decision_log_odds_threshold",
            torch.zeros((), dtype=torch.float32),
        )
        self.register_buffer(
            "_risk_decision_threshold_active",
            torch.tensor(False, dtype=torch.bool),
        )
        # Persist the lifecycle flag in the buffer, but branch on this Python
        # shadow in forward to avoid a CUDA scalar ``.item()`` synchronization.
        self._risk_decision_threshold_active_shadow = False

        # I1 estimates P(branch prediction is correct). Its odds, rather than
        # the old log-probability, express the relative evidence for correctness
        # versus error. A single positive scale preserves the common I1 scale
        # and cannot silently learn branch-specific competence priors.
        self.raw_route_prior_beta = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )

        # A compact monotone risk calibrator. Every input is oriented so a larger
        # value means more risk; positive weights preserve that interpretation.
        self.raw_risk_feature_weights = nn.Parameter(
            torch.full((len(RISK_FEATURE_NAMES),), _inverse_softplus(0.10))
        )
        initial_logit = math.log(float(initial_risk) / (1.0 - float(initial_risk)))
        self.risk_bias = nn.Parameter(torch.tensor(initial_logit))

    @property
    def risk_decision_threshold_active(self) -> bool:
        return self._risk_decision_threshold_active_shadow

    def set_risk_decision_threshold(self, raw_log_odds_threshold: float) -> None:
        value = float(raw_log_odds_threshold)
        if not math.isfinite(value):
            raise ValueError("routing risk decision threshold must be finite")
        self._risk_decision_log_odds_threshold.fill_(value)
        self._risk_decision_threshold_active.fill_(True)
        self._risk_decision_threshold_active_shadow = True

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
        self._risk_decision_threshold_active_shadow = bool(
            self._risk_decision_threshold_active.detach().item()
        )

    def route_parameters(self) -> list[nn.Parameter]:
        if self.mode == "prior_only":
            return []
        return (
            [self.raw_route_prior_beta]
            if self.reliability_input_enabled
            else []
        )

    def risk_parameters(self) -> list[nn.Parameter]:
        if self.risk_mode != "learned":
            return []
        return [self.raw_risk_feature_weights, self.risk_bias]

    def effective_risk_feature_weights(self) -> torch.Tensor:
        """Return the exact deployed non-negative risk coefficients."""

        weights = F.softplus(self.raw_risk_feature_weights)
        feature_mask = weights.new_tensor(
            [
                float(self.reliability_input_enabled),
                1.0,
                float(self.risk_conflict_enabled),
            ]
        )
        return weights * feature_mask

    def route_effective_l2(self) -> torch.Tensor:
        """L2 penalty on the operative positive reliability-odds exponent."""

        if self.mode != "learned" or not self.reliability_input_enabled:
            return self.raw_route_prior_beta.new_zeros(())
        return F.softplus(self.raw_route_prior_beta).square()

    def risk_effective_l2(self) -> torch.Tensor:
        """Fixed-four-slot L2 on effective risk coefficients and intercept."""

        if self.risk_mode != "learned":
            return self.risk_bias.new_zeros(())
        effective_weights = self.effective_risk_feature_weights()
        return torch.cat(
            [effective_weights.view(-1), self.risk_bias.view(-1)]
        ).square().mean()

    def effective_parameter_diagnostics(self) -> dict[str, float]:
        """Return cold-path scale diagnostics for convergence audits."""

        with torch.no_grad():
            learned_route = self.mode == "learned"
            learned_risk = self.risk_mode == "learned"
            reliability_prior_active = bool(
                self.reliability_input_enabled
                and self.mode in {"learned", "prior_only"}
            )
            beta = float(
                (
                    F.softplus(self.raw_route_prior_beta)
                    if learned_route and reliability_prior_active
                    else self._fixed_prior_beta
                    if reliability_prior_active
                    else self.raw_route_prior_beta.new_zeros(())
                )
                .detach()
                .cpu()
            )
            risk = (
                self.effective_risk_feature_weights().detach().cpu()
                if learned_risk
                else torch.zeros_like(
                    self.raw_risk_feature_weights.detach().cpu()
                )
            )
            return {
                "route_prior_beta": beta,
                "risk_feature_weight_max": float(risk.max()),
                "risk_bias_abs": (
                    float(self.risk_bias.detach().abs().cpu())
                    if learned_risk
                    else 0.0
                ),
            }

    def effective_parameter_details(self) -> dict[str, object]:
        """Return named deployed I2 coefficients for scientific diagnostics.

        ``effective_parameter_diagnostics`` intentionally stays flat because it
        is consumed by the numerical scale guard. This richer view records the
        exact deployed route equation and every remaining monotone coefficient.
        """

        with torch.no_grad():
            learned_route = self.mode == "learned"
            reliability_prior_active = bool(self.reliability_input_enabled)
            learned_risk = self.risk_mode == "learned"
            risk_tensor = (
                self.effective_risk_feature_weights().detach().cpu()
                if learned_risk
                else torch.zeros_like(
                    self.raw_risk_feature_weights.detach().cpu()
                )
            )
            if reliability_prior_active:
                route_semantics = "beta_logit_reliability"
            else:
                route_semantics = "alive_masked_uniform"
            return {
                "route_mode": self.mode,
                "route_reliability_input_enabled": reliability_prior_active,
                "route_prior_beta": float(
                    (
                        F.softplus(self.raw_route_prior_beta)
                        if learned_route and reliability_prior_active
                        else self._fixed_prior_beta
                        if reliability_prior_active
                        else self.raw_route_prior_beta.new_zeros(())
                    )
                    .detach()
                    .cpu()
                ),
                "route_score_semantics": route_semantics,
                "risk_mode": self.risk_mode,
                "risk_conflict_enabled": bool(self.risk_conflict_enabled),
                "risk_conflict_active": bool(
                    learned_risk and self.risk_conflict_enabled
                ),
                "risk_head_semantics": (
                    "monotone_logistic_features"
                    if learned_risk
                    else "reliability_deficit_probability"
                    if self.risk_mode == "reliability_prior"
                    else "disabled"
                ),
                "risk_feature_weights": (
                    {
                        feature: float(value)
                        for feature, value in zip(
                            RISK_FEATURE_NAMES, risk_tensor.tolist()
                        )
                    }
                    if learned_risk
                    else {}
                ),
                "risk_bias": (
                    float(self.risk_bias.detach().cpu())
                    if learned_risk
                    else None
                ),
            }

    def prepare_route_inputs(
        self,
        branch_probabilities: dict[str, torch.Tensor],
        reliability: dict[str, torch.Tensor],
        alive: dict[str, torch.Tensor],
        *,
        eps: float = 1.0e-6,
    ) -> dict[str, torch.Tensor | float]:
        """Prepare parameter-independent tensors shared by route/risk steps.

        The returned tensors contain no router-parameter computation.  A
        post-hoc caller may therefore prepare them once under ``torch.no_grad``
        and reuse them across optimizer evaluations through
        :meth:`forward_prepared`.  Ordinary end-to-end callers should keep
        using :meth:`forward`, which preserves gradients to the supplied
        branch probabilities and reliability values.
        """
        if not 0.0 < float(eps) < 0.5:
            raise ValueError("routing eps must be within (0, 0.5)")

        missing_probability = [
            name for name in ROUTING_BRANCHES if name not in branch_probabilities
        ]
        missing_reliability = [
            name for name in ROUTING_BRANCHES if name not in reliability
        ]
        missing_alive = [name for name in ROUTING_BRANCHES if name not in alive]
        if missing_probability or missing_reliability or missing_alive:
            raise ValueError(
                "routing inputs must contain api/graph/manifest; "
                f"missing_probabilities={missing_probability}, "
                f"missing_reliability={missing_reliability}, "
                f"missing_alive={missing_alive}"
            )

        reference = branch_probabilities[ROUTING_BRANCHES[0]]
        if not reference.is_floating_point():
            raise ValueError("routing branch probabilities must be floating point")
        if reference.ndim != 2 or reference.size(-1) != 2:
            raise ValueError(
                "I2 routing is defined for binary branch probabilities [B, 2]"
            )
        batch_size, num_classes = reference.shape
        probability_eps = max(
            float(eps),
            float(torch.finfo(reference.dtype).tiny),
        )

        probability_values: list[torch.Tensor] = []
        for name in ROUTING_BRANCHES:
            probability = branch_probabilities[name].to(
                device=reference.device, dtype=reference.dtype
            )
            if probability.shape != reference.shape:
                raise ValueError(
                    "all routed branch probabilities must have the same [B, 2] shape"
                )
            if not bool(torch.isfinite(probability).all().item()) or bool(
                (probability < 0.0).any().item()
            ):
                raise ValueError(
                    f"branch probability {name!r} must be finite and non-negative"
                )
            mass = probability.sum(dim=-1, keepdim=True)
            if bool((mass <= float(eps)).any().item()):
                raise ValueError(
                    f"branch probability {name!r} must have positive class mass"
                )
            if not bool(
                torch.isclose(
                    mass,
                    torch.ones_like(mass),
                    rtol=1.0e-4,
                    atol=max(float(eps), 1.0e-6),
                )
                .all()
                .item()
            ):
                raise ValueError(
                    f"branch probability {name!r} must sum to one; "
                    "raw evidential belief mass is not a routing probability"
                )
            # Remove harmless floating-point drift after enforcing the semantic
            # probability contract. Deliberately do not turn arbitrary
            # non-unit masses into probabilities.
            probability_values.append(probability / mass)
        probability_stack = torch.stack(probability_values, dim=1)

        input_reliability_stack = torch.stack(
            [reliability[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype)
        alive_stack = torch.stack(
            [alive[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype)
        expected_stack_shape = (batch_size, len(ROUTING_BRANCHES))
        if input_reliability_stack.shape != expected_stack_shape:
            raise ValueError("routing reliability batch shape disagrees with probabilities")
        if alive_stack.shape != expected_stack_shape:
            raise ValueError("routing alive batch shape disagrees with probabilities")
        if not bool(torch.isfinite(input_reliability_stack).all().item()) or bool(
            (
                (input_reliability_stack < 0.0)
                | (input_reliability_stack > 1.0)
            ).any().item()
        ):
            raise ValueError("routing reliability must be finite and within [0, 1]")
        if not bool(torch.isfinite(alive_stack).all().item()) or bool(
            ((alive_stack < 0.0) | (alive_stack > 1.0)).any().item()
        ):
            raise ValueError("routing alive values must be finite and within [0, 1]")
        if bool(((alive_stack != 0.0) & (alive_stack != 1.0)).any().item()):
            raise ValueError(
                "routing alive values must be hard binary availability masks"
            )

        reliability_stack = (
            input_reliability_stack
            if self.reliability_input_enabled
            else torch.ones_like(input_reliability_stack)
        )

        # Each branch is compared only with the reliability-weighted consensus
        # of its alive peers. Dead placeholder probabilities are masked before
        # either the consensus or the conflict is constructed.
        eye = torch.eye(
            len(ROUTING_BRANCHES), device=reference.device, dtype=reference.dtype
        ).unsqueeze(0)
        peer_availability = (
            alive_stack.unsqueeze(-1) * alive_stack.unsqueeze(-2) * (1.0 - eye)
        )
        peer_count = peer_availability.sum(dim=-1)

        peer_reliability = reliability_stack.unsqueeze(1) * peer_availability
        peer_reliability_mass = peer_reliability.sum(dim=-1)
        peer_consensus_probability = (
            peer_reliability.unsqueeze(-1) * probability_stack.unsqueeze(1)
        ).sum(dim=2) / peer_reliability_mass.unsqueeze(-1).clamp_min(
            probability_eps
        )
        peer_consensus_probability = torch.where(
            peer_reliability_mass.unsqueeze(-1) > 0.0,
            peer_consensus_probability,
            torch.full_like(peer_consensus_probability, 1.0 / float(num_classes)),
        )
        branch_probability = probability_stack.clamp_min(probability_eps)
        branch_probability = branch_probability / branch_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(probability_eps)
        consensus_probability = peer_consensus_probability.clamp_min(
            probability_eps
        )
        consensus_probability = consensus_probability / consensus_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(probability_eps)
        js_midpoint = (0.5 * (branch_probability + consensus_probability)).clamp_min(
            probability_eps
        )
        peer_consensus_js = 0.5 * (
            (
                branch_probability
                * (branch_probability.log() - js_midpoint.log())
            ).sum(dim=-1)
            + (
                consensus_probability
                * (consensus_probability.log() - js_midpoint.log())
            ).sum(dim=-1)
        ) / math.log(2.0)
        peer_consensus_defined = (alive_stack > 0.0) & (
            peer_reliability_mass > 0.0
        )
        peer_consensus_js = torch.where(
            peer_consensus_defined,
            peer_consensus_js,
            torch.zeros_like(peer_consensus_js),
        )
        # Scale normalized JS by mean peer reliability. With no reliable peer,
        # cross-modal conflict is undefined and contributes exactly zero.
        peer_consensus_support = (
            peer_reliability_mass / peer_count.clamp_min(1.0)
        ).clamp(0.0, 1.0)
        reliability_weighted_cross_modal_conflict = (
            peer_consensus_js.clamp(0.0, 1.0)
            * peer_consensus_support
            * alive_stack
        )
        # Normalize over the currently alive branches, not the fixed three-slot
        # tensor. Dividing by three when a branch is missing would make this
        # feature an implicit missing-ratio proxy, which is outside the final
        # three-feature I2 risk contract.
        global_cross_modal_conflict = (
            reliability_weighted_cross_modal_conflict.sum(dim=-1)
            / alive_stack.sum(dim=-1).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        routing_reliability = reliability_stack * alive_stack
        has_available = alive_stack.sum(dim=-1) > 0.0
        unavailable = alive_stack <= 0.0
        # ``fusion.min_discount`` may be smaller than the representable gap to
        # one in the routing dtype. Use a dtype-aware bound so r=1 never turns
        # into infinite log-odds and an undefined all-inf softmax.
        logit_eps = max(float(eps), float(torch.finfo(reference.dtype).eps))
        reliability_log_odds = torch.logit(
            reliability_stack.clamp(logit_eps, 1.0 - logit_eps)
        )
        uniform_class = torch.full(
            (batch_size, num_classes),
            1.0 / float(num_classes),
            device=reference.device,
            dtype=reference.dtype,
        )

        return {
            "eps": float(eps),
            "probability_stack": probability_stack,
            "input_reliability_stack": input_reliability_stack,
            "reliability_stack": reliability_stack,
            "alive_stack": alive_stack,
            "routing_reliability": routing_reliability,
            "has_available": has_available,
            "unavailable": unavailable,
            "reliability_log_odds": reliability_log_odds,
            "uniform_class": uniform_class,
            "reliability_weighted_cross_modal_conflict": (
                reliability_weighted_cross_modal_conflict
            ),
            "global_cross_modal_conflict": global_cross_modal_conflict,
            "peer_consensus_probability": peer_consensus_probability,
            "peer_consensus_support": peer_consensus_support,
            "peer_consensus_js": peer_consensus_js,
        }

    def forward_prepared(
        self,
        prepared: dict[str, torch.Tensor | float],
        *,
        learned_active: bool = True,
        branch_distribution_override: torch.Tensor | None = None,
        compute_risk: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Execute the parameter-dependent route/risk path on prepared input."""
        required_tensor_keys = (
            "probability_stack",
            "input_reliability_stack",
            "reliability_stack",
            "alive_stack",
            "routing_reliability",
            "has_available",
            "unavailable",
            "reliability_log_odds",
            "uniform_class",
            "reliability_weighted_cross_modal_conflict",
            "global_cross_modal_conflict",
            "peer_consensus_probability",
            "peer_consensus_support",
            "peer_consensus_js",
        )
        missing = [
            key
            for key in required_tensor_keys
            if not isinstance(prepared.get(key), torch.Tensor)
        ]
        if missing:
            raise ValueError(f"prepared routing inputs are missing tensors: {missing}")
        eps = float(prepared.get("eps", 0.0))
        if not 0.0 < eps < 0.5:
            raise ValueError("prepared routing eps must be within (0, 0.5)")

        probability_stack = prepared["probability_stack"]
        assert isinstance(probability_stack, torch.Tensor)
        if (
            probability_stack.ndim != 3
            or probability_stack.size(1) != len(ROUTING_BRANCHES)
            or probability_stack.size(2) != 2
        ):
            raise ValueError(
                "prepared probability_stack must have shape "
                f"[B, {len(ROUTING_BRANCHES)}, 2]"
            )
        batch_size, _, num_classes = probability_stack.shape
        reference = probability_stack[:, 0, :]
        probability_eps = max(eps, float(torch.finfo(reference.dtype).tiny))
        logit_eps = max(eps, float(torch.finfo(reference.dtype).eps))

        input_reliability_stack = prepared["input_reliability_stack"]
        reliability_stack = prepared["reliability_stack"]
        alive_stack = prepared["alive_stack"]
        routing_reliability = prepared["routing_reliability"]
        has_available = prepared["has_available"]
        unavailable = prepared["unavailable"]
        reliability_log_odds = prepared["reliability_log_odds"]
        uniform_class = prepared["uniform_class"]
        reliability_weighted_cross_modal_conflict = prepared[
            "reliability_weighted_cross_modal_conflict"
        ]
        global_cross_modal_conflict = prepared["global_cross_modal_conflict"]
        peer_consensus_probability = prepared["peer_consensus_probability"]
        peer_consensus_support = prepared["peer_consensus_support"]
        peer_consensus_js = prepared["peer_consensus_js"]
        assert all(
            isinstance(value, torch.Tensor)
            for value in (
                input_reliability_stack,
                reliability_stack,
                alive_stack,
                routing_reliability,
                has_available,
                unavailable,
                reliability_log_odds,
                uniform_class,
                reliability_weighted_cross_modal_conflict,
                global_cross_modal_conflict,
                peer_consensus_probability,
                peer_consensus_support,
                peer_consensus_js,
            )
        )

        expected_stack_shape = (batch_size, len(ROUTING_BRANCHES))
        if any(
            value.shape != expected_stack_shape
            for value in (
                input_reliability_stack,
                reliability_stack,
                alive_stack,
                routing_reliability,
                unavailable,
                reliability_log_odds,
                reliability_weighted_cross_modal_conflict,
            )
        ):
            raise ValueError("prepared routing stack shapes disagree")
        if has_available.shape != (batch_size,):
            raise ValueError("prepared has_available has an invalid shape")
        if global_cross_modal_conflict.shape != (batch_size,):
            raise ValueError(
                "prepared global_cross_modal_conflict has an invalid shape"
            )

        learned_route_active = bool(
            learned_active
            and self.mode == "learned"
            and self.reliability_input_enabled
        )
        learned_route_prior_beta = (
            F.softplus(self.raw_route_prior_beta).to(
                device=reference.device, dtype=reference.dtype
            )
            if self.reliability_input_enabled
            else torch.zeros((), device=reference.device, dtype=reference.dtype)
        )
        route_prior_beta = (
            self._fixed_prior_beta.to(
                device=reference.device, dtype=reference.dtype
            )
            if self.mode == "prior_only" and self.reliability_input_enabled
            else learned_route_prior_beta
        )
        operative_route_prior_beta = (
            route_prior_beta if learned_route_active else route_prior_beta.detach()
        )
        # ``learned_active`` is the post-hoc lifecycle switch. Before I1 is
        # fitted, Stage-1 must use an alive-only neutral prior.
        if bool(learned_active):
            unmasked_prior_scores = (
                operative_route_prior_beta * reliability_log_odds
            )
        else:
            unmasked_prior_scores = torch.zeros_like(reliability_log_odds)
        prior_scores = unmasked_prior_scores.masked_fill(
            unavailable, torch.finfo(reference.dtype).min
        )
        prior_branch_distribution = F.softmax(prior_scores, dim=-1)
        prior_branch_distribution = torch.where(
            has_available.unsqueeze(-1),
            prior_branch_distribution,
            torch.zeros_like(prior_branch_distribution),
        )
        routing_scores = prior_scores
        branch_distribution = prior_branch_distribution

        override_active = branch_distribution_override is not None
        if override_active:
            override = branch_distribution_override.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            if override.shape != (batch_size, len(ROUTING_BRANCHES)):
                raise ValueError(
                    "branch_distribution_override must have shape "
                    f"[B, {len(ROUTING_BRANCHES)}], got {tuple(override.shape)}"
                )
            if not bool(torch.isfinite(override).all().item()) or bool(
                (override < 0.0).any().item()
            ):
                raise ValueError(
                    "branch_distribution_override must contain finite non-negative values"
                )
            override = override * alive_stack
            override_sum = override.sum(dim=-1, keepdim=True)
            invalid_available = has_available & override_sum.view(-1).le(float(eps))
            if bool(invalid_available.any().item()):
                raise ValueError(
                    "branch_distribution_override assigns no mass to an available branch"
                )
            branch_distribution = torch.where(
                has_available.unsqueeze(-1),
                override / override_sum.clamp_min(probability_eps),
                torch.zeros_like(override),
            )

        mixture_probability = (
            branch_distribution.unsqueeze(-1) * probability_stack
        ).sum(dim=1)
        mixture_probability = torch.where(
            has_available.unsqueeze(-1), mixture_probability, uniform_class
        )
        mixture_probability = mixture_probability.clamp_min(probability_eps)
        mixture_probability = mixture_probability / mixture_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(probability_eps)

        if not compute_risk:
            # Route fitting has no gradient path through the independent risk
            # head. Return immediately after the conditional class mixture so
            # optimizer evaluations do not rebuild the three static risk
            # features or synchronize the decision-threshold buffers.
            return {
                "branch_distribution": branch_distribution,
                "prior_branch_distribution": prior_branch_distribution,
                "mixture_probability": mixture_probability,
                "routing_scores": routing_scores,
                "routing_reliability": routing_reliability,
                "reliability_log_odds": reliability_log_odds,
                "route_prior_beta": route_prior_beta,
                "reliability_weighted_cross_modal_conflict": (
                    reliability_weighted_cross_modal_conflict
                ),
                "global_cross_modal_conflict": global_cross_modal_conflict,
                "has_available": has_available.to(dtype=reference.dtype),
            }

        # Fused-risk features are aligned with the route that is actually used.
        # Detaching pi/p_mix makes the risk loss incapable of changing the
        # conditional class router.
        risk_distribution = branch_distribution.detach()
        risk_mixture_probability = mixture_probability.detach()
        reliability_deficit = (
            1.0 - (risk_distribution * reliability_stack).sum(dim=-1)
        ).clamp(0.0, 1.0)
        raw_log_odds = (
            risk_mixture_probability[:, 1].clamp_min(probability_eps).log()
            - risk_mixture_probability[:, 0].clamp_min(probability_eps).log()
        )
        if self.risk_decision_threshold_active:
            decision_threshold = self._risk_decision_log_odds_threshold.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            # Argmax in a binary frame is the zero raw-log-odds boundary.
            decision_threshold = torch.zeros(
                (), device=reference.device, dtype=reference.dtype
            )
        (
            predicted_malware,
            decision_boundary_proximity,
        ) = threshold_aligned_fn_risk_state(
            raw_log_odds,
            decision_threshold,
        )

        risk_conflict = (
            global_cross_modal_conflict
            if self.risk_conflict_enabled
            else torch.zeros_like(global_cross_modal_conflict)
        )
        risk_features = torch.stack(
            [
                reliability_deficit,
                decision_boundary_proximity,
                risk_conflict,
            ],
            dim=-1,
        )

        learned_risk_active = bool(learned_active and self.risk_mode == "learned")
        if self.risk_mode == "disabled" or (
            self.risk_mode == "learned" and not learned_risk_active
        ):
            risk_probability = torch.zeros(
                batch_size,
                device=reference.device,
                dtype=reference.dtype,
            )
            risk_logit = torch.full_like(risk_probability, -torch.inf)
            risk_feature_weights = torch.zeros(
                len(RISK_FEATURE_NAMES), device=reference.device, dtype=reference.dtype
            )
        elif self.risk_mode == "reliability_prior":
            # Static no-learned-risk control: routed unreliability under I1,
            # with no fitted risk parameters.
            risk_probability = reliability_deficit
            risk_logit = torch.logit(
                risk_probability.clamp(logit_eps, 1.0 - logit_eps)
            )
            risk_feature_weights = torch.zeros(
                len(RISK_FEATURE_NAMES), device=reference.device, dtype=reference.dtype
            )
            risk_feature_weights[0] = float(self.reliability_input_enabled)
        else:
            risk_feature_weights = self.effective_risk_feature_weights().to(
                device=reference.device, dtype=reference.dtype
            )
            risk_training_logit = self.risk_bias.to(
                device=reference.device, dtype=reference.dtype
            ) + (risk_features * risk_feature_weights.unsqueeze(0)).sum(dim=-1)
            risk_probability = torch.sigmoid(risk_training_logit)
            risk_logit = risk_training_logit
        if self.risk_mode != "learned" or not learned_risk_active:
            risk_training_logit = risk_logit
        if self.risk_decision_threshold_active:
            risk_probability = torch.where(
                predicted_malware,
                torch.zeros_like(risk_probability),
                risk_probability,
            )
            risk_logit = torch.where(
                predicted_malware,
                torch.full_like(risk_logit, -torch.inf),
                risk_logit,
            )
        risk_probability = torch.where(
            has_available, risk_probability, torch.ones_like(risk_probability)
        ).clamp(0.0, 1.0)
        risk_logit = torch.where(
            has_available,
            risk_logit,
            torch.full_like(risk_logit, torch.inf),
        )

        committed_mass = (1.0 - risk_probability).clamp(0.0, 1.0)
        fused_belief = mixture_probability * committed_mass.unsqueeze(-1)
        fused_uncertainty = risk_probability

        return {
            "belief": fused_belief,
            "uncertainty": fused_uncertainty,
            "branch_distribution": branch_distribution,
            "prior_branch_distribution": prior_branch_distribution,
            "mixture_probability": mixture_probability,
            "routing_scores": routing_scores,
            "routing_reliability": routing_reliability,
            "reliability_log_odds": reliability_log_odds,
            "route_prior_beta": route_prior_beta,
            "prior_only_odds_beta": self._fixed_prior_beta.to(
                device=reference.device, dtype=reference.dtype
            ),
            "prior_only_odds_beta_active": torch.full_like(
                risk_probability,
                float(self.mode == "prior_only" and self.reliability_input_enabled),
            ),
            "route_reliability_input_enabled": torch.full_like(
                risk_probability, float(self.reliability_input_enabled)
            ),
            "peer_consensus_probability": peer_consensus_probability,
            "peer_consensus_support": peer_consensus_support,
            "peer_consensus_js": peer_consensus_js,
            "reliability_weighted_cross_modal_conflict": (
                reliability_weighted_cross_modal_conflict
            ),
            "global_cross_modal_conflict": global_cross_modal_conflict,
            "risk_probability": risk_probability,
            "risk_logit": risk_logit,
            "risk_training_logit": risk_training_logit,
            "risk_features": risk_features,
            "risk_feature_weights": risk_feature_weights,
            "risk_reliability_deficit": reliability_deficit,
            "risk_decision_boundary_proximity": decision_boundary_proximity,
            "risk_predicted_malware": predicted_malware.to(reference.dtype),
            "risk_decision_log_odds_threshold": decision_threshold.expand(
                batch_size
            ),
            "risk_decision_threshold_active": torch.full_like(
                risk_probability,
                float(self.risk_decision_threshold_active),
            ),
            "risk_global_cross_modal_conflict": risk_conflict,
            "has_available": has_available.to(dtype=reference.dtype),
            "committed_mass": committed_mass,
            "risk_mode_learned": torch.full_like(
                risk_probability, float(self.risk_mode == "learned")
            ),
            "risk_mode_reliability_prior": torch.full_like(
                risk_probability, float(self.risk_mode == "reliability_prior")
            ),
            "risk_mode_disabled": torch.full_like(
                risk_probability, float(self.risk_mode == "disabled")
            ),
            "learned_components_active": torch.full_like(
                risk_probability, float(learned_route_active or learned_risk_active)
            ),
            "prefit_uniform_prior_active": torch.full_like(
                risk_probability, float(not bool(learned_active))
            ),
            "risk_conflict_feature_configured": torch.full_like(
                risk_probability, float(self.risk_conflict_enabled)
            ),
            "risk_conflict_feature_active": torch.full_like(
                risk_probability,
                float(self.risk_conflict_enabled and learned_risk_active),
            ),
            "branch_distribution_override_active": torch.full_like(
                risk_probability, float(override_active)
            ),
        }

    def forward(
        self,
        branch_probabilities: dict[str, torch.Tensor],
        reliability: dict[str, torch.Tensor],
        alive: dict[str, torch.Tensor],
        *,
        learned_active: bool = True,
        branch_distribution_override: torch.Tensor | None = None,
        compute_risk: bool = True,
        eps: float = 1.0e-6,
    ) -> dict[str, torch.Tensor]:
        prepared = self.prepare_route_inputs(
            branch_probabilities,
            reliability,
            alive,
            eps=eps,
        )
        return self.forward_prepared(
            prepared,
            learned_active=learned_active,
            branch_distribution_override=branch_distribution_override,
            compute_risk=compute_risk,
        )
