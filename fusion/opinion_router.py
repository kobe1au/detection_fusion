from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTING_BRANCHES = ("api", "graph", "manifest")
RISK_FEATURE_NAMES = (
    "reliability_deficit",
    "uncertainty_burden",
    "decision_boundary_proximity",
    "structural_conflict",
    "missing_fraction",
)

RISK_TARGETS = (
    "mixture_argmax_error",
    "threshold_classification_error",
    "threshold_malware_false_negative",
    "reliability_deficit_score",
)


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive")
    return math.log(math.expm1(value))


class GlobalOpinionRouter(nn.Module):
    """Reliability-log-odds router with an independent decision-risk head.

    The class router ``pi`` and the decision-event risk score ``u`` have different
    statistical targets and are deliberately parameterised separately. Before
    post-hoc fitting, ``pi`` is uniform over alive branches because raw extraction
    integrity is not a cross-branch calibrated correctness probability:

    * ``pi`` is trained by conditional-mixture NLL plus an optional detached
      branch-NLL soft-oracle auxiliary objective.
    * ``u`` is trained by BCE/Brier against its explicitly configured event.

    Predictive disagreement can affect a branch score only through the explicit
    term ``-softplus(lambda_m) * d_m``.  The free residual never consumes class
    probabilities or disagreement, so it cannot learn an opposite-sign copy of
    that term.  Risk features all have the orientation "larger means riskier"
    and use non-negative weights. Missing sources occupy fixed slots instead of
    disappearing from an ``N_alive`` denominator.
    """

    MODES = ("learned", "prior_only")

    def __init__(
        self,
        *,
        mode: str = "learned",
        route_conflict_enabled: bool = True,
        risk_conflict_enabled: bool = True,
        risk_mode: str = "learned",
        risk_target: str = "mixture_argmax_error",
        initial_risk: float = 0.10,
        fixed_prior_beta: float = 1.0,
    ):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError(f"routing mode must be one of {self.MODES}, got {mode!r}")
        if not 0.0 < float(initial_risk) < 1.0:
            raise ValueError("routing initial_risk must be within (0, 1)")
        self.mode = mode
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
        self.route_conflict_enabled = bool(route_conflict_enabled)
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

        # Per branch: opinion uncertainty and a missingness indicator. Reliability enters
        # only through the common-scale log-odds prior below, so the free
        # residual cannot cancel I1 as a direct function of that same input.
        # Class probabilities and disagreement are intentionally absent as well.
        residual_input_dim = len(ROUTING_BRANCHES) * 2
        # The post-hoc identity pool is intentionally much smaller than the
        # encoder-training set.  A single zero-initialised linear residual is
        # therefore used instead of the former 6->H->3 MLP.  Omitting a bias is
        # important: branch-specific intercepts would learn a second static
        # competence prior on top of I1's calibrated correctness intercepts.
        # Softmax is invariant to adding the same value to all logits. Learn
        # only K-1 relative logits and fix the Manifest residual logit to zero,
        # removing the otherwise exact per-feature common-row nullspace.
        self.route_residual = nn.Linear(
            residual_input_dim,
            len(ROUTING_BRANCHES) - 1,
            bias=False,
        )
        nn.init.zeros_(self.route_residual.weight)

        # I1 estimates P(branch prediction is correct). Its odds, rather than
        # the old log-probability, express the relative evidence for correctness
        # versus error. A single positive scale preserves the common I1 scale
        # and cannot silently learn branch-specific competence priors.
        self.raw_route_prior_beta = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )

        # Starts near prior-only routing; post-hoc fitting learns how strongly an
        # otherwise comparable outlier should be penalised.
        self.raw_conflict_scale = nn.Parameter(
            torch.full((len(ROUTING_BRANCHES),), _inverse_softplus(0.05))
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
        return [
            *self.route_residual.parameters(),
            self.raw_route_prior_beta,
            self.raw_conflict_scale,
        ]

    def risk_parameters(self) -> list[nn.Parameter]:
        if self.risk_mode != "learned":
            return []
        return [self.raw_risk_feature_weights, self.risk_bias]

    def route_effective_l2(self) -> torch.Tensor:
        """L2 penalty in the router's operative parameterization.

        Penalizing raw softplus coordinates would make the regularizer depend
        on an arbitrary reparameterization and would not prevent the effective
        odds/conflict scales from drifting toward hard routing.
        """

        if self.mode != "learned":
            return self.raw_route_prior_beta.new_zeros(())
        beta = F.softplus(self.raw_route_prior_beta)
        conflict = F.softplus(self.raw_conflict_scale)
        residual = self.route_residual.weight
        return torch.cat(
            [beta.view(-1), conflict.view(-1), residual.reshape(-1)]
        ).square().mean()

    def risk_effective_l2(self) -> torch.Tensor:
        """L2 penalty on effective monotone risk coefficients and intercept."""

        if self.risk_mode != "learned":
            return self.risk_bias.new_zeros(())
        weights = F.softplus(self.raw_risk_feature_weights)
        return torch.cat([weights.view(-1), self.risk_bias.view(-1)]).square().mean()

    def effective_parameter_diagnostics(self) -> dict[str, float]:
        """Return cold-path scale diagnostics for convergence audits."""

        with torch.no_grad():
            beta = float(F.softplus(self.raw_route_prior_beta).detach().cpu())
            conflict = F.softplus(self.raw_conflict_scale).detach().cpu()
            risk = F.softplus(self.raw_risk_feature_weights).detach().cpu()
            residual = self.route_residual.weight.detach().cpu()
            return {
                "route_prior_beta": beta,
                "route_conflict_scale_max": float(conflict.max()),
                "route_residual_abs_max": float(residual.abs().max()),
                "risk_feature_weight_max": float(risk.max()),
                "risk_bias_abs": float(self.risk_bias.detach().abs().cpu()),
            }

    def prepare_route_inputs(
        self,
        beliefs: dict[str, torch.Tensor],
        uncertainties: dict[str, torch.Tensor],
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
        opinions and reliability values.
        """
        if not 0.0 < float(eps) < 0.5:
            raise ValueError("routing eps must be within (0, 0.5)")

        reference = beliefs[ROUTING_BRANCHES[0]]
        if reference.ndim != 2 or reference.size(-1) != 2:
            raise ValueError("I2-v2 routing is defined for binary [B, 2] beliefs")
        batch_size, num_classes = reference.shape

        reliability_stack = torch.stack(
            [reliability[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
        uncertainty_stack = torch.stack(
            [uncertainties[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
        alive_stack = torch.stack(
            [alive[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
        if any(value.shape != (batch_size, len(ROUTING_BRANCHES)) for value in (
            reliability_stack,
            uncertainty_stack,
            alive_stack,
        )):
            raise ValueError("routing reliability/uncertainty/alive batch shapes disagree")

        expected_probabilities: list[torch.Tensor] = []
        for name in ROUTING_BRANCHES:
            belief = beliefs[name].to(device=reference.device, dtype=reference.dtype)
            uncertainty = uncertainties[name].view(-1, 1).to(
                device=reference.device, dtype=reference.dtype
            )
            if belief.shape != reference.shape:
                raise ValueError("all routed beliefs must have the same [B, C] shape")
            expected_probabilities.append(
                (belief + uncertainty / float(num_classes)).clamp(0.0, 1.0)
            )
        probability_stack = torch.stack(expected_probabilities, dim=1)
        malware_probability = probability_stack[:, :, 1]

        # Observed disagreement and branch-local outlier distance. Missing
        # placeholder logits are masked before either quantity is constructed.
        pairwise_matrix = (
            malware_probability.unsqueeze(-1) - malware_probability.unsqueeze(-2)
        ).abs()
        eye = torch.eye(
            len(ROUTING_BRANCHES), device=reference.device, dtype=reference.dtype
        ).unsqueeze(0)
        peer_availability = (
            alive_stack.unsqueeze(-1) * alive_stack.unsqueeze(-2) * (1.0 - eye)
        )
        peer_count = peer_availability.sum(dim=-1)

        # I2 compares each branch with the consensus of its *reliable* peers.
        # The previous unweighted mean absolute disagreement was branch-aware,
        # but treated a weak peer exactly like a well-calibrated one.  That
        # symmetry is undesirable when one modality is semantically corrupted:
        # two trustworthy peers should provide stronger counter-evidence than
        # two already doubtful peers.  I1 values are detached by the post-hoc
        # lifecycle before route fitting, so this construction introduces no
        # circular route -> reliability dependency.
        peer_reliability = reliability_stack.unsqueeze(1) * peer_availability
        peer_reliability_mass = peer_reliability.sum(dim=-1)
        peer_consensus_probability = (
            peer_reliability.unsqueeze(-1) * probability_stack.unsqueeze(1)
        ).sum(dim=2) / peer_reliability_mass.unsqueeze(-1).clamp_min(float(eps))
        peer_consensus_probability = torch.where(
            peer_reliability_mass.unsqueeze(-1) > 0.0,
            peer_consensus_probability,
            torch.full_like(peer_consensus_probability, 1.0 / float(num_classes)),
        )
        branch_probability = probability_stack.clamp_min(float(eps))
        branch_probability = branch_probability / branch_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(float(eps))
        consensus_probability = peer_consensus_probability.clamp_min(float(eps))
        consensus_probability = consensus_probability / consensus_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(float(eps))
        js_midpoint = (0.5 * (branch_probability + consensus_probability)).clamp_min(
            float(eps)
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
        # Scale the normalized JS divergence by mean peer reliability.  With
        # no peer, conflict is undefined and therefore contributes zero; the
        # existing missing-fraction feature handles that case explicitly.
        peer_consensus_support = (
            peer_reliability_mass / peer_count.clamp_min(1.0)
        ).clamp(0.0, 1.0)
        observed_outlier_distance = (
            peer_consensus_js.clamp(0.0, 1.0)
            * peer_consensus_support
            * alive_stack
        )
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
        # Missingness, rather than raw availability, keeps the last three
        # inputs exactly zero for the ordinary all-modalities-present case.
        # Otherwise alive=[1,1,1] would let a bias-free linear layer synthesize
        # an arbitrary static branch intercept and duplicate I1 competence.
        residual_features = torch.cat(
            [uncertainty_stack * alive_stack, 1.0 - alive_stack], dim=-1
        )

        uniform_class = torch.full(
            (batch_size, num_classes),
            1.0 / float(num_classes),
            device=reference.device,
            dtype=reference.dtype,
        )
        fixed_slots = float(len(ROUTING_BRANCHES))
        alive_fraction = alive_stack.sum(dim=-1) / fixed_slots
        missing_fraction = (1.0 - alive_fraction).clamp(0.0, 1.0)
        pair_indices = ((0, 1), (0, 2), (1, 2))
        pairwise_disagreement = torch.stack(
            [pairwise_matrix[:, left, right] for left, right in pair_indices],
            dim=-1,
        )
        pairwise_availability = torch.stack(
            [
                alive_stack[:, left] * alive_stack[:, right]
                for left, right in pair_indices
            ],
            dim=-1,
        )
        observed_pairwise_disagreement = (
            pairwise_disagreement * pairwise_availability
        )
        # Use a fixed three-pair denominator. Missing comparisons contribute
        # neither false agreement nor a second missingness penalty; the
        # explicit missing_fraction feature carries that information.
        structural_pairwise_conflict = observed_pairwise_disagreement
        structural_conflict = structural_pairwise_conflict.mean(dim=-1)

        return {
            "eps": float(eps),
            "probability_stack": probability_stack,
            "reliability_stack": reliability_stack,
            "uncertainty_stack": uncertainty_stack,
            "alive_stack": alive_stack,
            "routing_reliability": routing_reliability,
            "has_available": has_available,
            "unavailable": unavailable,
            "reliability_log_odds": reliability_log_odds,
            "residual_features": residual_features,
            "uniform_class": uniform_class,
            "observed_outlier_distance": observed_outlier_distance,
            "peer_consensus_probability": peer_consensus_probability,
            "peer_consensus_support": peer_consensus_support,
            "peer_consensus_js": peer_consensus_js,
            "alive_fraction": alive_fraction,
            "missing_fraction": missing_fraction,
            "observed_pairwise_disagreement": observed_pairwise_disagreement,
            "structural_pairwise_conflict": structural_pairwise_conflict,
            "structural_conflict": structural_conflict,
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
            "reliability_stack",
            "uncertainty_stack",
            "alive_stack",
            "routing_reliability",
            "has_available",
            "unavailable",
            "reliability_log_odds",
            "residual_features",
            "uniform_class",
            "observed_outlier_distance",
            "peer_consensus_probability",
            "peer_consensus_support",
            "peer_consensus_js",
            "alive_fraction",
            "missing_fraction",
            "observed_pairwise_disagreement",
            "structural_pairwise_conflict",
            "structural_conflict",
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

        reliability_stack = prepared["reliability_stack"]
        uncertainty_stack = prepared["uncertainty_stack"]
        alive_stack = prepared["alive_stack"]
        routing_reliability = prepared["routing_reliability"]
        has_available = prepared["has_available"]
        unavailable = prepared["unavailable"]
        reliability_log_odds = prepared["reliability_log_odds"]
        residual_features = prepared["residual_features"]
        uniform_class = prepared["uniform_class"]
        observed_outlier_distance = prepared["observed_outlier_distance"]
        peer_consensus_probability = prepared["peer_consensus_probability"]
        peer_consensus_support = prepared["peer_consensus_support"]
        peer_consensus_js = prepared["peer_consensus_js"]
        alive_fraction = prepared["alive_fraction"]
        missing_fraction = prepared["missing_fraction"]
        observed_pairwise_disagreement = prepared[
            "observed_pairwise_disagreement"
        ]
        structural_pairwise_conflict = prepared[
            "structural_pairwise_conflict"
        ]
        structural_conflict = prepared["structural_conflict"]
        assert all(
            isinstance(value, torch.Tensor)
            for value in (
                reliability_stack,
                uncertainty_stack,
                alive_stack,
                routing_reliability,
                has_available,
                unavailable,
                reliability_log_odds,
                residual_features,
                uniform_class,
                observed_outlier_distance,
                peer_consensus_probability,
                peer_consensus_support,
                peer_consensus_js,
                alive_fraction,
                missing_fraction,
                observed_pairwise_disagreement,
                structural_pairwise_conflict,
                structural_conflict,
            )
        )

        expected_stack_shape = (batch_size, len(ROUTING_BRANCHES))
        if any(
            value.shape != expected_stack_shape
            for value in (
                reliability_stack,
                uncertainty_stack,
                alive_stack,
                routing_reliability,
                unavailable,
                reliability_log_odds,
                observed_outlier_distance,
            )
        ):
            raise ValueError("prepared routing stack shapes disagree")
        if residual_features.shape != (
            batch_size,
            2 * len(ROUTING_BRANCHES),
        ):
            raise ValueError("prepared residual_features has an invalid shape")
        if has_available.shape != (batch_size,):
            raise ValueError("prepared has_available has an invalid shape")

        learned_route_active = bool(learned_active and self.mode == "learned")
        routing_outlier_distance = (
            observed_outlier_distance
            if self.route_conflict_enabled and learned_route_active
            else torch.zeros_like(observed_outlier_distance)
        )
        learned_route_prior_beta = F.softplus(self.raw_route_prior_beta).to(
            device=reference.device, dtype=reference.dtype
        )
        route_prior_beta = (
            self._fixed_prior_beta.to(
                device=reference.device, dtype=reference.dtype
            )
            if self.mode == "prior_only"
            else learned_route_prior_beta
        )
        operative_route_prior_beta = (
            route_prior_beta if learned_route_active else route_prior_beta.detach()
        )
        # ``learned_active`` is the post-hoc lifecycle switch, not merely a
        # residual-network switch. Before I1 is fitted, the supplied values are
        # raw integrity fallbacks whose near-one values would be explosively
        # separated by log-odds. Use an alive-only neutral prior at that stage.
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
        if not learned_route_active:
            route_residual = torch.zeros_like(routing_reliability)
        else:
            relative_residual = self.route_residual(residual_features)
            route_residual = torch.cat(
                [
                    relative_residual,
                    torch.zeros_like(relative_residual[:, :1]),
                ],
                dim=-1,
            )
        conflict_scale = F.softplus(self.raw_conflict_scale).to(
            device=reference.device, dtype=reference.dtype
        )
        conflict_penalty = routing_outlier_distance * conflict_scale.unsqueeze(0)
        routing_scores = (prior_scores + route_residual - conflict_penalty).masked_fill(
            unavailable, torch.finfo(reference.dtype).min
        )
        if not learned_route_active:
            branch_distribution = prior_branch_distribution
        else:
            branch_distribution = F.softmax(routing_scores, dim=-1)
            branch_distribution = torch.where(
                has_available.unsqueeze(-1),
                branch_distribution,
                torch.zeros_like(branch_distribution),
            )

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
                override / override_sum.clamp_min(float(eps)),
                torch.zeros_like(override),
            )

        mixture_probability = (
            branch_distribution.unsqueeze(-1) * probability_stack
        ).sum(dim=1)
        mixture_probability = torch.where(
            has_available.unsqueeze(-1), mixture_probability, uniform_class
        )
        mixture_probability = mixture_probability.clamp_min(float(eps))
        mixture_probability = mixture_probability / mixture_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(float(eps))

        if not compute_risk:
            # Route fitting has no gradient path through the independent risk
            # head.  Return immediately after the conditional class mixture so
            # thousands of optimizer evaluations do not rebuild five static
            # risk features or synchronize the decision-threshold buffers.
            return {
                "branch_distribution": branch_distribution,
                "prior_branch_distribution": prior_branch_distribution,
                "mixture_probability": mixture_probability,
                "routing_scores": routing_scores,
                "route_residual": route_residual,
                "routing_reliability": routing_reliability,
                "reliability_log_odds": reliability_log_odds,
                "route_prior_beta": route_prior_beta,
                "observed_outlier_distance": observed_outlier_distance,
                "routing_outlier_distance": routing_outlier_distance,
                "conflict_penalty_scale": conflict_scale,
                "conflict_penalty": conflict_penalty,
                "has_available": has_available.to(dtype=reference.dtype),
            }

        # Fused-risk features are aligned with the route that is actually used.
        # Detaching pi/p_mix makes the risk loss incapable of changing the
        # conditional class router. Missingness remains a fixed three-slot
        # feature, so confidence cannot improve merely because N_alive shrank.
        risk_distribution = branch_distribution.detach()
        risk_mixture_probability = mixture_probability.detach()
        reliability_deficit = (
            1.0 - (risk_distribution * reliability_stack).sum(dim=-1)
        ).clamp(0.0, 1.0)
        uncertainty_burden = (
            risk_distribution * uncertainty_stack
        ).sum(dim=-1).clamp(0.0, 1.0)
        raw_log_odds = (
            risk_mixture_probability[:, 1].clamp_min(float(eps)).log()
            - risk_mixture_probability[:, 0].clamp_min(float(eps)).log()
        )
        if (
            self.risk_target
            in {
                "threshold_classification_error",
                "threshold_malware_false_negative",
            }
            and self.risk_decision_threshold_active
        ):
            decision_threshold = self._risk_decision_log_odds_threshold.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            # Argmax in a binary frame is the zero raw-log-odds boundary.
            decision_threshold = torch.zeros(
                (), device=reference.device, dtype=reference.dtype
            )
        predicted_malware = raw_log_odds >= decision_threshold
        if self.risk_target == "threshold_malware_false_negative":
            # On the benign side, proximity increases monotonically towards
            # the fixed malware boundary.  The event is impossible on the
            # malware side and is gated to zero below.
            decision_boundary_proximity = torch.where(
                predicted_malware,
                torch.zeros_like(raw_log_odds),
                (2.0 * torch.sigmoid(raw_log_odds - decision_threshold)).clamp(
                    0.0, 1.0
                ),
            )
        else:
            decision_boundary_proximity = torch.exp(
                -(raw_log_odds - decision_threshold).abs()
            ).clamp(0.0, 1.0)

        risk_conflict = (
            structural_conflict
            if self.risk_conflict_enabled
            else torch.zeros_like(structural_conflict)
        )
        risk_features = torch.stack(
            [
                reliability_deficit,
                uncertainty_burden,
                decision_boundary_proximity,
                risk_conflict,
                missing_fraction,
            ],
            dim=-1,
        )

        learned_risk_active = bool(learned_active and self.risk_mode == "learned")
        if self.risk_mode == "disabled" or (
            self.risk_mode == "learned" and not learned_risk_active
        ):
            risk_probability = torch.zeros(batch_size, device=reference.device, dtype=reference.dtype)
            risk_logit = torch.full_like(risk_probability, -torch.inf)
            risk_feature_weights = torch.zeros(
                len(RISK_FEATURE_NAMES), device=reference.device, dtype=reference.dtype
            )
        elif self.risk_mode == "reliability_prior":
            # Static no-learned-risk control: expected missing-or-incorrect
            # source fraction under I1, with no fitted risk parameters.
            risk_probability = reliability_deficit
            risk_logit = torch.logit(
                risk_probability.clamp(float(eps), 1.0 - float(eps))
            )
            risk_feature_weights = torch.zeros(
                len(RISK_FEATURE_NAMES), device=reference.device, dtype=reference.dtype
            )
            risk_feature_weights[0] = 1.0
        else:
            risk_feature_weights = F.softplus(self.raw_risk_feature_weights).to(
                device=reference.device, dtype=reference.dtype
            )
            risk_training_logit = self.risk_bias.to(
                device=reference.device, dtype=reference.dtype
            ) + (risk_features * risk_feature_weights.unsqueeze(0)).sum(dim=-1)
            risk_probability = torch.sigmoid(risk_training_logit)
            risk_logit = risk_training_logit
        if self.risk_mode != "learned" or not learned_risk_active:
            risk_training_logit = risk_logit
        if (
            self.risk_target == "threshold_malware_false_negative"
            and self.risk_decision_threshold_active
        ):
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
        branch_weights = branch_distribution * committed_mass.unsqueeze(-1)
        weights = torch.cat([branch_weights, risk_probability.unsqueeze(-1)], dim=-1)
        fused_belief = mixture_probability * committed_mass.unsqueeze(-1)
        fused_uncertainty = risk_probability

        return {
            "belief": fused_belief,
            "uncertainty": fused_uncertainty,
            "weights": weights,
            "branch_distribution": branch_distribution,
            "prior_branch_distribution": prior_branch_distribution,
            "mixture_probability": mixture_probability,
            "routing_scores": routing_scores,
            "route_residual": route_residual,
            "routing_reliability": routing_reliability,
            "reliability_log_odds": reliability_log_odds,
            "route_prior_beta": route_prior_beta,
            "prior_only_odds_beta": self._fixed_prior_beta.to(
                device=reference.device, dtype=reference.dtype
            ),
            "prior_only_odds_beta_active": torch.full_like(
                risk_probability, float(self.mode == "prior_only")
            ),
            "observed_outlier_distance": observed_outlier_distance,
            "peer_consensus_probability": peer_consensus_probability,
            "peer_consensus_support": peer_consensus_support,
            "peer_consensus_js": peer_consensus_js,
            "routing_outlier_distance": routing_outlier_distance,
            "conflict_penalty_scale": conflict_scale,
            "conflict_penalty": conflict_penalty,
            "risk_probability": risk_probability,
            "risk_logit": risk_logit,
            "risk_training_logit": risk_training_logit,
            "risk_features": risk_features,
            "risk_feature_weights": risk_feature_weights,
            "risk_reliability_deficit": reliability_deficit,
            "risk_uncertainty_burden": uncertainty_burden,
            "risk_decision_boundary_proximity": decision_boundary_proximity,
            "risk_predicted_malware": predicted_malware.to(reference.dtype),
            "risk_decision_log_odds_threshold": decision_threshold.expand(
                batch_size
            ),
            "risk_decision_threshold_active": torch.full_like(
                risk_probability,
                float(self.risk_decision_threshold_active),
            ),
            "risk_structural_conflict": structural_conflict,
            "risk_missing_fraction": missing_fraction,
            "alive_fraction": alive_fraction,
            "missing_fraction": missing_fraction,
            "observed_pairwise_disagreement": observed_pairwise_disagreement,
            "structural_pairwise_conflict": structural_pairwise_conflict,
            "mean_disagreement": observed_pairwise_disagreement.mean(dim=-1),
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
            "route_conflict_feature_configured": torch.full_like(
                risk_probability, float(self.route_conflict_enabled)
            ),
            "risk_conflict_feature_configured": torch.full_like(
                risk_probability, float(self.risk_conflict_enabled)
            ),
            "route_conflict_feature_active": torch.full_like(
                risk_probability,
                float(self.route_conflict_enabled and learned_route_active),
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
        beliefs: dict[str, torch.Tensor],
        uncertainties: dict[str, torch.Tensor],
        reliability: dict[str, torch.Tensor],
        alive: dict[str, torch.Tensor],
        *,
        learned_active: bool = True,
        branch_distribution_override: torch.Tensor | None = None,
        compute_risk: bool = True,
        eps: float = 1.0e-6,
    ) -> dict[str, torch.Tensor]:
        prepared = self.prepare_route_inputs(
            beliefs,
            uncertainties,
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
