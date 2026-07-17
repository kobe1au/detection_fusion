from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTING_BRANCHES = ("api", "graph", "manifest")


class GlobalOpinionRouter(nn.Module):
    """Route three modality opinions while keeping known mass interpretable.

    For ``learned`` and ``prior_only`` routing, the mean effective reliability
    is a *hard upper bound* on total branch mass.  A learned residual may change
    the distribution within that known mass and may move additional mass to the
    residual-error/unknown outcome, but it cannot create more known mass than
    the reliability prior permits.  ``known_only`` is the explicit no-unknown
    ablation: when at least one branch is available its operative known-mass
    prior is one and the learned residual only routes between branches.

    ``prior_only`` uses the exact reliability prior and does not consume the
    residual network.  Keeping the network instantiated gives all modes the
    same checkpoint shape and makes atomic eval-time substitutions possible.
    """

    MODES = ("learned", "prior_only", "known_only")

    def __init__(
        self,
        hidden_dim: int = 16,
        *,
        mode: str = "learned",
        use_disagreement: bool = True,
        initial_known_retention: float = 0.99,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("routing hidden_dim must be positive")
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError(f"routing mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.use_disagreement = bool(use_disagreement)
        if not 0.0 < float(initial_known_retention) < 1.0:
            raise ValueError(
                "routing initial_known_retention must be within (0, 1)"
            )
        self.initial_known_retention = float(initial_known_retention)
        # Per branch: reliability, uncertainty, malware probability, and
        # availability. Three pairwise probability differences provide the
        # global disagreement context. Model visibility acts directly on the
        # reliability prior instead of becoming another freely learned feature.
        input_dim = len(ROUTING_BRANCHES) * 4 + 3
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(ROUTING_BRANCHES) + 1),
        )
        final = self.residual[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        nn.init.constant_(
            final.bias[-1],
            math.log(
                self.initial_known_retention
                / (1.0 - self.initial_known_retention)
            ),
        )

    def forward(
        self,
        beliefs: dict[str, torch.Tensor],
        uncertainties: dict[str, torch.Tensor],
        reliability: dict[str, torch.Tensor],
        alive: dict[str, torch.Tensor],
        visible_factor: dict[str, torch.Tensor],
        *,
        eps: float = 1.0e-6,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 < float(eps) < 0.5:
            raise ValueError("routing eps must be within (0, 0.5)")

        reference = beliefs[ROUTING_BRANCHES[0]]
        if reference.ndim != 2 or reference.size(-1) < 2:
            raise ValueError("routing expects [B, C] beliefs with C >= 2")

        reliability_stack = torch.stack(
            [reliability[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype)
        uncertainty_stack = torch.stack(
            [uncertainties[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype)
        alive_stack = torch.stack(
            [alive[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype)
        visible_stack = torch.stack(
            [visible_factor[name].view(-1) for name in ROUTING_BRANCHES], dim=-1
        ).to(device=reference.device, dtype=reference.dtype)

        expected_probabilities = []
        num_classes = reference.size(-1)
        for name in ROUTING_BRANCHES:
            belief = beliefs[name].to(device=reference.device, dtype=reference.dtype)
            uncertainty = uncertainties[name].view(-1, 1).to(
                device=reference.device, dtype=reference.dtype
            )
            expected_probabilities.append(
                (belief + uncertainty / float(num_classes)).clamp(0.0, 1.0)
            )
        probability_stack = torch.stack(expected_probabilities, dim=1)
        malware_probability = probability_stack[:, :, 1]
        observed_pairwise_disagreement = torch.stack(
            [
                (malware_probability[:, 0] - malware_probability[:, 1]).abs(),
                (malware_probability[:, 0] - malware_probability[:, 2]).abs(),
                (malware_probability[:, 1] - malware_probability[:, 2]).abs(),
            ],
            dim=-1,
        )
        pairwise_availability = torch.stack(
            [
                alive_stack[:, 0] * alive_stack[:, 1],
                alive_stack[:, 0] * alive_stack[:, 2],
                alive_stack[:, 1] * alive_stack[:, 2],
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        observed_pairwise_disagreement = (
            observed_pairwise_disagreement * pairwise_availability
        )
        routing_pairwise_disagreement = (
            observed_pairwise_disagreement
            if self.use_disagreement
            else torch.zeros_like(observed_pairwise_disagreement)
        )

        effective_reliability = (
            reliability_stack.clamp(0.0, 1.0)
            * visible_stack.clamp(0.0, 1.0)
            * alive_stack.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        # A missing branch is represented only by its availability bit. Its
        # placeholder logits/opinion must not change how surviving branches are
        # routed, because those values do not correspond to observed evidence.
        reliability_feature = reliability_stack * alive_stack
        uncertainty_feature = uncertainty_stack * alive_stack
        malware_probability_feature = malware_probability * alive_stack
        features = torch.cat(
            [
                reliability_feature,
                uncertainty_feature,
                malware_probability_feature,
                alive_stack,
                routing_pairwise_disagreement,
            ],
            dim=-1,
        )

        available_count = alive_stack.sum(dim=-1).clamp_min(1.0)
        reliability_sum = effective_reliability.sum(dim=-1)
        has_available = alive_stack.sum(dim=-1) > 0.0
        reliability_prior_known_mass = (
            reliability_sum / available_count
        ).clamp(0.0, 1.0)
        reliability_prior_known_mass = torch.where(
            has_available,
            reliability_prior_known_mass,
            torch.zeros_like(reliability_prior_known_mass),
        )

        # ``known_only`` deliberately removes the residual-error outcome.  Its
        # operative prior is therefore one whenever any source is available;
        # the reliability-derived prior remains exposed separately for audits.
        prior_known_mass = (
            has_available.to(dtype=reference.dtype)
            if self.mode == "known_only"
            else reliability_prior_known_mass
        )
        prior_unknown_mass = 1.0 - prior_known_mass

        # Reliability supplies the branch-distribution prior.  When all
        # effective reliabilities are zero but branches are alive, the masked
        # softmax below falls back to a uniform available-branch distribution.
        branch_prior_logits = effective_reliability.clamp_min(float(eps)).log()
        unavailable = alive_stack <= 0.0
        branch_prior_logits = branch_prior_logits.masked_fill(
            unavailable, torch.finfo(branch_prior_logits.dtype).min
        )
        uniform_available_distribution = (
            alive_stack / available_count.unsqueeze(-1)
        )
        prior_branch_distribution = torch.where(
            (reliability_sum > 0.0).unsqueeze(-1),
            effective_reliability
            / reliability_sum.clamp_min(float(eps)).unsqueeze(-1),
            uniform_available_distribution,
        )
        prior_branch_distribution = torch.where(
            has_available.unsqueeze(-1),
            prior_branch_distribution,
            torch.zeros_like(prior_branch_distribution),
        )

        if self.mode == "prior_only":
            residual_logits = torch.zeros(
                (reference.size(0), len(ROUTING_BRANCHES) + 1),
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            residual_logits = self.residual(features)

        if self.mode == "prior_only":
            branch_distribution = prior_branch_distribution
        else:
            branch_logits = (
                branch_prior_logits
                + residual_logits[:, : len(ROUTING_BRANCHES)]
            )
            branch_logits = branch_logits.masked_fill(
                unavailable, torch.finfo(branch_logits.dtype).min
            )
            branch_distribution = F.softmax(branch_logits, dim=-1)
            branch_distribution = torch.where(
                has_available.unsqueeze(-1),
                branch_distribution,
                torch.zeros_like(branch_distribution),
            )

        if self.mode == "learned":
            # The fourth residual output is a retention gate, not an independent
            # unknown logit. A sigmoid keeps the hard known-mass upper bound
            # while retaining gradients in both directions across optimizer
            # steps. The positive initial bias starts close to, but below, the
            # exact prior; ``prior_only`` remains the exact-prior ablation.
            known_retention = torch.sigmoid(residual_logits[:, -1])
            known_mass = prior_known_mass * known_retention
        else:
            known_retention = torch.ones_like(prior_known_mass)
            known_mass = prior_known_mass
        known_mass = torch.where(
            has_available, known_mass, torch.zeros_like(known_mass)
        ).clamp(0.0, 1.0)
        branch_weights = branch_distribution * known_mass.unsqueeze(-1)
        unknown_weight = (1.0 - known_mass).clamp(0.0, 1.0)
        weights = torch.cat([branch_weights, unknown_weight.unsqueeze(-1)], dim=-1)
        belief_stack = torch.stack(
            [beliefs[name] for name in ROUTING_BRANCHES], dim=1
        ).to(device=reference.device, dtype=reference.dtype)
        fused_belief = (branch_weights.unsqueeze(-1) * belief_stack).sum(dim=1)
        fused_uncertainty = unknown_weight + (
            branch_weights * uncertainty_stack
        ).sum(dim=-1)

        total = fused_belief.sum(dim=-1) + fused_uncertainty
        fused_belief = fused_belief / total.clamp_min(float(eps)).unsqueeze(-1)
        fused_uncertainty = (fused_uncertainty / total.clamp_min(float(eps))).clamp(
            0.0, 1.0
        )
        return {
            "belief": fused_belief,
            "uncertainty": fused_uncertainty,
            "weights": weights,
            "effective_reliability": effective_reliability,
            "prior_branch_distribution": prior_branch_distribution,
            "branch_distribution": branch_distribution,
            "known_mass": branch_weights.sum(dim=-1),
            "known_retention": known_retention,
            "prior_known_mass": prior_known_mass,
            "prior_unknown_mass": prior_unknown_mass,
            "reliability_prior_known_mass": reliability_prior_known_mass,
            "reliability_prior_unknown_mass": 1.0 - reliability_prior_known_mass,
            "observed_pairwise_disagreement": observed_pairwise_disagreement,
            "routing_pairwise_disagreement": routing_pairwise_disagreement,
            "disagreement_feature_active": torch.full_like(
                prior_known_mass, float(self.use_disagreement)
            ),
            "mean_disagreement": observed_pairwise_disagreement.sum(dim=-1)
            / pairwise_availability.sum(dim=-1).clamp_min(1.0),
        }
