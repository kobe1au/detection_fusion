from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fusion.constants import AvailabilityIndex, GateConstants
from fusion.evidential import (
    COMBINATION_RULES,
    EVIDENCE_BRANCHES,
    combine_opinions_with_diagnostics,
    logits_to_opinion,
    opinion_to_dirichlet_alpha,
    opinion_to_prob,
    trust_discount,
)


def _validate_binary_availability(availability: torch.Tensor) -> None:
    if availability.ndim != 2 or availability.size(1) != AvailabilityIndex.BASE_DIM:
        raise ValueError(
            "DiscountProbabilityFusion availability must have shape "
            f"[B, {AvailabilityIndex.BASE_DIM}], got {tuple(availability.shape)}"
        )
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
    """Fixed evidential fusion used only by registered comparison methods.

    CARE-Droid never calls this module. It contains only the parameter-free
    rules needed by the registered fixed-fusion comparisons.
    """

    _ALLOWED_CONFIG_KEYS = frozenset(
        {
            "base_rate",
            "combination",
            "evidence_activation",
            "force_fp32_decision",
            "min_discount",
            "mode",
            "opinion_source",
            "use_hard_alive_mask",
        }
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        unknown = sorted(set(self.config) - self._ALLOWED_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                "Unsupported fixed evidential-fusion keys: "
                f"{unknown}. Learned calibration/routing options belong to no "
                "current method and must be removed."
            )

        mode = str(self.config.get("mode", "discount_probability")).lower()
        if mode != "discount_probability":
            raise ValueError(
                "DiscountProbabilityFusion requires fusion.mode="
                "'discount_probability'"
            )
        if "combination" not in self.config:
            raise ValueError(
                "fusion.combination is required for every fixed evidential "
                "comparison"
            )
        self.combination = str(self.config["combination"]).lower()
        if self.combination not in COMBINATION_RULES:
            raise ValueError(
                "fusion.combination must be one of "
                f"{COMBINATION_RULES}, got {self.combination!r}"
            )
        self.opinion_source = str(
            self.config.get("opinion_source", "evidential")
        ).lower()
        if self.opinion_source != "evidential":
            raise ValueError(
                "Current fixed comparisons require "
                "fusion.opinion_source='evidential'"
            )
        self.evidence_activation = str(
            self.config.get("evidence_activation", "softplus")
        ).lower()
        if self.evidence_activation not in {"softplus", "relu", "exp"}:
            raise ValueError(
                "fusion.evidence_activation must be softplus, relu, or exp"
            )
        if not bool(self.config.get("use_hard_alive_mask", True)):
            raise ValueError(
                "Current evidential comparisons require "
                "fusion.use_hard_alive_mask=true"
            )
        if not bool(self.config.get("force_fp32_decision", True)):
            raise ValueError(
                "Current evidential comparisons require "
                "fusion.force_fp32_decision=true"
            )

        self.eps = float(self.config.get("min_discount", GateConstants.EPS))
        if not 0.0 < self.eps < 1.0:
            raise ValueError("fusion.min_discount must lie strictly within (0, 1)")
        self.base_rate = float(self.config.get("base_rate", 0.5))
        if not 0.0 <= self.base_rate <= 1.0:
            raise ValueError("fusion.base_rate must lie within [0, 1]")

    def forward(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, torch.Tensor | bool]:
        if not (
            api_logits.shape == graph_logits.shape == manifest_logits.shape
        ):
            raise ValueError("All evidential branch logits must have equal shape")
        if api_logits.ndim != 2 or api_logits.size(-1) < 2:
            raise ValueError(
                "Evidential branch logits must have shape [B, C>=2]"
            )
        if availability.size(0) != api_logits.size(0):
            raise ValueError("Availability batch size disagrees with branch logits")
        _validate_binary_availability(availability)

        with torch.autocast(device_type=api_logits.device.type, enabled=False):
            return self._forward_fp32(
                api_logits.float(),
                graph_logits.float(),
                manifest_logits.float(),
                availability.float(),
            )

    def _forward_fp32(
        self,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, torch.Tensor | bool]:
        logits_by_branch = {
            "api": api_logits,
            "graph": graph_logits,
            "manifest": manifest_logits,
        }
        opinions = {
            name: logits_to_opinion(
                logits,
                evidence_activation=self.evidence_activation,
                eps=self.eps,
            )
            for name, logits in logits_by_branch.items()
        }
        alive = {
            "api": availability[:, AvailabilityIndex.API_ALIVE],
            "graph": availability[:, AvailabilityIndex.GRAPH_ALIVE],
            "manifest": availability[:, AvailabilityIndex.MANIFEST_ALIVE],
        }

        predictive_uncertainties: list[torch.Tensor] = []
        fused_beliefs: list[torch.Tensor] = []
        fused_uncertainties: list[torch.Tensor] = []
        for name in EVIDENCE_BRANCHES:
            branch_alive = alive[name]
            belief, uncertainty = trust_discount(
                opinions[name]["belief"],
                opinions[name]["uncertainty"],
                branch_alive,
            )
            predictive_uncertainties.append(uncertainty)
            fused_beliefs.append(belief)
            fused_uncertainties.append(uncertainty)

        fused_belief, fused_uncertainty, diagnostics = (
            combine_opinions_with_diagnostics(
                fused_beliefs,
                fused_uncertainties,
                rule=self.combination,
                availability_masks=[
                    alive[name] for name in EVIDENCE_BRANCHES
                ],
                eps=self.eps,
            )
        )
        final_prob = opinion_to_prob(
            fused_belief,
            fused_uncertainty,
            base_rate=self.base_rate,
            eps=self.eps,
        )
        final_log_prob = final_prob.clamp_min(self.eps).log()
        fused_alpha = opinion_to_dirichlet_alpha(
            fused_belief,
            fused_uncertainty,
            eps=self.eps,
        )

        contribution = torch.stack(
            [
                alive[name] * (1.0 - opinions[name]["uncertainty"])
                for name in EVIDENCE_BRANCHES
            ],
            dim=-1,
        )
        contribution_sum = contribution.sum(dim=-1, keepdim=True)
        alive_matrix = torch.stack(
            [alive[name] for name in EVIDENCE_BRANCHES], dim=-1
        )
        alive_sum = alive_matrix.sum(dim=-1, keepdim=True)
        fallback_weights = torch.where(
            alive_sum > 0,
            alive_matrix / alive_sum.clamp_min(self.eps),
            torch.zeros_like(alive_matrix),
        )
        fusion_weights = torch.where(
            contribution_sum > self.eps,
            contribution / contribution_sum.clamp_min(self.eps),
            fallback_weights,
        )

        outputs: dict[str, torch.Tensor | bool] = {
            "final_prob": final_prob,
            "final_logits": final_log_prob,
            "final_is_log_probability": True,
            "fusion_weights": fusion_weights,
            "dirichlet_alpha_fused": fused_alpha,
            "raw_conflict": diagnostics["raw_conflict"].clamp(0.0, 1.0),
        }
        for index, name in enumerate(EVIDENCE_BRANCHES):
            outputs[f"fusion_weight_{name}"] = fusion_weights[:, index]
            outputs[f"uncertainty_proxy_{name}"] = (
                predictive_uncertainties[index]
            )
            outputs[f"dirichlet_alpha_{name}"] = torch.where(
                alive[name].bool().unsqueeze(-1),
                opinions[name]["alpha"],
                torch.ones_like(opinions[name]["alpha"]),
            )
            outputs[f"{name}_alive"] = alive[name]
        return outputs
