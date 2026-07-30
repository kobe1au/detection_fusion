"""Evidential (subjective-logic) primitives for the tri-modal robust pipeline.

This module provides the opinion representation, hard availability discount,
conflict diagnostics, and the fixed comparison rules used by the registered
trusted-fusion baselines. CARE-Droid does not call these rules.

Everything here is pure functional tensor code so it can be unit-tested in
isolation and runs under an explicit FP32 context from the caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Source modalities that participate in evidence combination. API and Graph
# are related code views, so the method does not claim statistical
# independence between all three modalities.
EVIDENCE_BRANCHES = ("api", "graph", "manifest")

COMBINATION_RULES = (
    "dempster",
    "cumulative",
    "log_pool",
    "ecml",
    "conflict_weighted_opinion",
)


def _coerce_availability_masks(
    availability_masks: list[torch.Tensor] | torch.Tensor | None,
    *,
    batch_size: int,
    num_sources: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return an explicit ``[B, S]`` Boolean source-availability matrix."""
    if availability_masks is None:
        return None
    if isinstance(availability_masks, torch.Tensor):
        if availability_masks.shape != (batch_size, num_sources):
            raise ValueError(
                "availability tensor must have shape [B, S], got "
                f"{tuple(availability_masks.shape)}"
            )
        return availability_masks.to(device=device, dtype=torch.bool)
    if len(availability_masks) != num_sources:
        raise ValueError("availability mask count must match source count")
    masks: list[torch.Tensor] = []
    for mask in availability_masks:
        if mask.numel() != batch_size:
            raise ValueError(
                "each availability mask must contain one value per sample"
            )
        masks.append(mask.to(device=device).view(-1).to(dtype=torch.bool))
    return torch.stack(masks, dim=1)


def logits_to_opinion(
    logits: torch.Tensor,
    *,
    evidence_activation: str = "softplus",
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Map raw branch logits to a multinomial subjective-logic opinion.

    Returns belief mass per class, uncertainty mass ``u`` (with
    ``sum(belief) + u == 1``), the Dirichlet ``alpha`` and the expected
    probability ``alpha / S``.
    """
    if logits.ndim != 2 or logits.size(-1) < 2:
        raise ValueError(f"logits_to_opinion expects [B, C>=2], got {tuple(logits.shape)}")
    activation = str(evidence_activation).lower()
    if activation == "softplus":
        evidence = F.softplus(logits)
    elif activation == "relu":
        evidence = F.relu(logits)
    elif activation == "exp":
        evidence = torch.exp(logits.clamp(max=80.0))
    else:
        raise ValueError(f"Unsupported evidence_activation: {evidence_activation}")
    evidence = evidence.clamp_min(0.0)
    num_classes = logits.size(-1)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True).clamp_min(eps)  # S = sum(alpha)
    belief = evidence / strength
    uncertainty = (float(num_classes) / strength).clamp(0.0, 1.0)
    expected_prob = alpha / strength
    return {
        "evidence": evidence,
        "alpha": alpha,
        "strength": strength.view(-1),
        "belief": belief,
        "uncertainty": uncertainty.view(-1),
        "expected_prob": expected_prob,
    }


def trust_discount(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Josang trust discounting of an opinion by a source-reliability ``r``.

    ``b'_k = r * b_k`` and ``u' = 1 - r * (1 - u)``. Lower reliability moves
    belief mass into uncertainty, so an unreliable source contributes little to
    the combined opinion rather than voting at full strength.
    """
    r = reliability.clamp(0.0, 1.0).view(-1, 1)
    discounted_belief = belief * r
    base_belief_mass = belief.sum(dim=-1, keepdim=True)
    discounted_uncertainty = (1.0 - r * base_belief_mass).clamp(0.0, 1.0)
    return discounted_belief, discounted_uncertainty.view(-1)


def _as_masses(belief: torch.Tensor, uncertainty: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = uncertainty.view(-1, 1).clamp(0.0, 1.0)
    b = belief.clamp_min(0.0)
    total = b.sum(dim=-1, keepdim=True) + u
    # Renormalise defensively so each opinion is a valid mass assignment.
    b = b / total.clamp_min(1e-8)
    u = u / total.clamp_min(1e-8)
    return b, u


def _combine_log_pool_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    eps: float,
    availability_masks: list[torch.Tensor] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a source-symmetric log pool over the available sources.

    A hard-missing source must be the identity, rather than an extra uniform
    factor that silently flattens the remaining predictions.
    """
    probabilities = []
    uncertainty_values = []
    for belief, uncertainty in zip(beliefs, uncertainties):
        b, u = _as_masses(belief, uncertainty)
        # Uniform base rate on a binary or k-ary frame.
        probabilities.append(
            (b + u / float(b.size(-1))).clamp_min(eps)
        )
        uncertainty_values.append(u.clamp(0.0, 1.0))
    probability_stack = torch.stack(probabilities, dim=1)  # [B, S, C]
    uncertainty_stack = torch.stack(uncertainty_values, dim=1).squeeze(-1)
    available = _coerce_availability_masks(
        availability_masks,
        batch_size=probability_stack.size(0),
        num_sources=probability_stack.size(1),
        device=probability_stack.device,
    )
    if available is None:
        # An exactly vacuous opinion is the pipeline's missing-view sentinel.
        available = uncertainty_stack < 1.0 - eps
    available_float = available.to(dtype=probability_stack.dtype)
    available_count = available_float.sum(dim=1, keepdim=True)
    log_prob = (
        probability_stack.clamp_min(eps).log()
        * available_float.unsqueeze(-1)
    ).sum(dim=1) / available_count.clamp_min(1.0)
    probability = torch.softmax(log_prob, dim=-1)
    uncertainty = (
        (
            uncertainty_stack
            .clamp_min(eps)
            .log()
            * available_float
        ).sum(dim=1, keepdim=True)
        / available_count.clamp_min(1.0)
    ).exp().clamp(0.0, 1.0)
    uncertainty = torch.where(
        available_count > 0.0,
        uncertainty,
        torch.ones_like(uncertainty),
    )
    belief = probability * (1.0 - uncertainty)
    return belief, uncertainty.view(-1)


def _combine_cumulative_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate evidence ratios directly instead of pairwise mass folding.

    For a multinomial opinion, ``b / u`` is proportional to its Dirichlet
    evidence.  Summing those ratios is source symmetric and avoids the tiny
    products and near-zero denominators of repeated pairwise cumulative fusion.
    FP16/BF16/FP32 inputs are accumulated in FP64, then returned in the input
    dtype.  ``eps`` gives dogmatic opinions a finite limiting representation.
    """
    masses = [_as_masses(belief, uncertainty) for belief, uncertainty in zip(beliefs, uncertainties)]
    output_dtype = masses[0][0].dtype
    work_dtype = (
        torch.float64
        if output_dtype in (torch.float16, torch.bfloat16, torch.float32)
        else output_dtype
    )
    belief_stack = torch.stack([belief for belief, _u in masses], dim=0).to(work_dtype)
    uncertainty_stack = torch.stack([u for _belief, u in masses], dim=0).to(work_dtype)
    stable_eps = max(float(eps), float(torch.finfo(work_dtype).eps))
    accumulated_ratio = (
        belief_stack / uncertainty_stack.clamp_min(stable_eps)
    ).sum(dim=0)
    normalizer = 1.0 + accumulated_ratio.sum(dim=-1, keepdim=True)
    belief = (accumulated_ratio / normalizer).to(output_dtype)
    uncertainty = (1.0 / normalizer).to(output_dtype)

    # Preserve a valid mass assignment after casting back to a lower precision.
    total = belief.sum(dim=-1, keepdim=True) + uncertainty
    belief = belief / total.clamp_min(float(eps))
    uncertainty = uncertainty / total.clamp_min(float(eps))
    return belief, uncertainty.view(-1)


def combine_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    rule: str = "dempster",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine N opinions with source-symmetric rules where possible."""
    if not beliefs:
        raise ValueError("combine_opinions requires at least one opinion")
    if len(beliefs) != len(uncertainties):
        raise ValueError("beliefs and uncertainties length mismatch")
    rule = str(rule).lower()
    if rule == "ecml":
        belief, uncertainty, _diagnostics = combine_ecml_opinions(
            beliefs, uncertainties, eps=eps
        )
        return belief, uncertainty
    if rule == "conflict_weighted_opinion":
        belief, uncertainty, _diagnostics = combine_conflict_weighted_opinions(
            beliefs, uncertainties, eps=eps
        )
        return belief, uncertainty
    if len(beliefs) == 1:
        belief, u = _as_masses(beliefs[0], uncertainties[0])
        return belief, u.view(-1)
    if rule == "log_pool":
        return _combine_log_pool_opinions(beliefs, uncertainties, eps)
    if rule == "dempster":
        singleton, theta, conflict, _b_stack, _u_stack = _multisource_intersection(
            beliefs, uncertainties, eps=eps
        )
        norm = (1.0 - conflict).clamp_min(eps)
        belief = singleton / norm.view(-1, 1)
        u = theta / norm
        return belief, u.view(-1).clamp(0.0, 1.0)
    if rule == "cumulative":
        return _combine_cumulative_opinions(beliefs, uncertainties, eps)
    raise ValueError(f"Unsupported combination rule: {rule}")


def _multisource_intersection(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return singleton, theta, conflict, and normalised source masses.

    For sources with singleton class masses ``b_i(k)`` and uncertainty mass
    ``u_i``, the N-way conjunctive singleton mass is

    ``m(k) = prod_i (b_i(k) + u_i) - prod_i u_i``.

    The remaining mass is raw conflict. This is order-independent and is
    reported as a baseline diagnostic.
    """
    if not beliefs:
        raise ValueError("multisource intersection requires at least one opinion")
    if len(beliefs) != len(uncertainties):
        raise ValueError("beliefs and uncertainties length mismatch")

    norm_beliefs: list[torch.Tensor] = []
    norm_uncertainties: list[torch.Tensor] = []
    for belief, uncertainty in zip(beliefs, uncertainties):
        b, u = _as_masses(belief, uncertainty)
        norm_beliefs.append(b)
        norm_uncertainties.append(u.view(-1))

    b_stack = torch.stack(norm_beliefs, dim=1)  # [B, S, C]
    u_stack = torch.stack(norm_uncertainties, dim=1).clamp(0.0, 1.0)  # [B, S]
    theta = u_stack.prod(dim=1).clamp(0.0, 1.0)
    singleton = (b_stack + u_stack.unsqueeze(-1)).prod(dim=1) - theta.view(-1, 1)
    singleton = singleton.clamp_min(0.0)
    conflict = (1.0 - singleton.sum(dim=-1) - theta).clamp_min(0.0)
    return singleton, theta, conflict, b_stack, u_stack


def multisource_conflict(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Order-independent raw conflict mass among all supplied opinions."""
    _singleton, _theta, conflict, _b_stack, _u_stack = _multisource_intersection(
        beliefs, uncertainties, eps=eps
    )
    return conflict


def _opinion_to_dirichlet_evidence(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover non-negative Dirichlet evidence from an opinion.

    For a ``K``-class subjective-logic opinion generated from a Dirichlet,
    ``u = K / S`` and ``b_k = e_k / S``. Therefore
    ``e_k = K b_k / u``. Exact dogmatic opinions (``u == 0``) correspond to
    infinite evidence; the epsilon floor provides their finite numerical limit.
    """
    b, u = _as_masses(belief, uncertainty)
    num_classes = b.size(-1)
    evidence = (
        float(num_classes) * b / u.clamp_min(eps)
    ).clamp_min(0.0)
    return evidence, b, u.view(-1)


def opinion_to_dirichlet_alpha(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Recover Dirichlet ``alpha`` from a multinomial opinion.

    Subjective logic uses ``u = K / S`` and ``b_k = e_k / S``.  The
    conversion therefore preserves both the projected probabilities and the
    evidence strength, which would be lost if a training objective attempted
    to reconstruct ``alpha`` from final class probabilities alone.
    """
    evidence, _belief, _uncertainty = _opinion_to_dirichlet_evidence(
        belief, uncertainty, eps=eps
    )
    return evidence + 1.0


def combine_ecml_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    *,
    availability_masks: list[torch.Tensor] | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """ECML conflictive aggregation via its ordered binary evidence mean.

    The ECML paper defines a binary operator equivalent to averaging two
    evidence vectors and folds it as ``w1 diamond w2 diamond ... diamond wV``.
    The authors' reference implementation follows that fixed-order fold, which
    is not associative for more than two views.  We therefore preserve the
    caller's source order instead of silently replacing the method with a
    symmetric global mean.  For API -> Graph -> Manifest, three available
    views consequently receive weights ``0.25, 0.25, 0.5``.

    After the fold, the fused opinion is reconstructed as
    ``alpha = fused_evidence + 1``, ``b = fused_evidence / S``, and ``u = K/S``.

    ``availability_masks`` may be a ``[B, S]`` tensor or one ``[B]`` tensor per
    source. When it is omitted, an exactly vacuous opinion is treated as an
    unavailable-source sentinel. This matches the pipeline's hard missing-view
    convention; callers that need an *available but vacuous* view to participate
    can pass an explicit all-ones mask. Missing sources are skipped without
    reordering the remaining views. If every source is unavailable, the result
    is the vacuous opinion.
    """
    if not beliefs:
        raise ValueError("combine_ecml_opinions requires at least one opinion")
    if len(beliefs) != len(uncertainties):
        raise ValueError("beliefs and uncertainties length mismatch")

    evidences: list[torch.Tensor] = []
    norm_beliefs: list[torch.Tensor] = []
    norm_uncertainties: list[torch.Tensor] = []
    reference_shape = beliefs[0].shape
    if len(reference_shape) != 2 or reference_shape[-1] < 2:
        raise ValueError(
            "combine_ecml_opinions expects beliefs shaped [B, C>=2], "
            f"got {tuple(reference_shape)}"
        )
    for belief, uncertainty in zip(beliefs, uncertainties):
        if belief.shape != reference_shape:
            raise ValueError("all ECML beliefs must have the same [B, C] shape")
        if uncertainty.numel() != reference_shape[0]:
            raise ValueError("each ECML uncertainty must contain one value per sample")
        evidence, b, u = _opinion_to_dirichlet_evidence(
            belief, uncertainty, eps=eps
        )
        evidences.append(evidence)
        norm_beliefs.append(b)
        norm_uncertainties.append(u)

    evidence_stack = torch.stack(evidences, dim=1)  # [B, S, C]
    belief_stack = torch.stack(norm_beliefs, dim=1)
    uncertainty_stack = torch.stack(norm_uncertainties, dim=1)
    batch_size, num_sources, num_classes = evidence_stack.shape

    if availability_masks is None:
        # Hard-masked missing views arrive as exactly vacuous opinions. Softplus
        # evidence is strictly positive, so a learned available view cannot hit
        # this sentinel in normal model execution.
        available = (
            (belief_stack.sum(dim=-1) > eps)
            | (uncertainty_stack < 1.0 - eps)
        )
    elif isinstance(availability_masks, torch.Tensor):
        if availability_masks.shape != (batch_size, num_sources):
            raise ValueError(
                "ECML availability tensor must have shape [B, S], got "
                f"{tuple(availability_masks.shape)}"
            )
        available = availability_masks.to(
            device=evidence_stack.device, dtype=torch.bool
        )
    else:
        if len(availability_masks) != num_sources:
            raise ValueError("ECML availability mask count must match source count")
        masks: list[torch.Tensor] = []
        for mask in availability_masks:
            if mask.numel() != batch_size:
                raise ValueError(
                    "each ECML availability mask must contain one value per sample"
                )
            masks.append(
                mask.to(device=evidence_stack.device).view(-1).to(dtype=torch.bool)
            )
        available = torch.stack(masks, dim=1)

    available_float = available.to(dtype=evidence_stack.dtype)
    available_count = available_float.sum(dim=1)

    # Match RCML/model.py: initialise from the first available view and then
    # apply (accumulator + next_evidence) / 2 in the declared source order.
    # torch.where keeps the operation batched and differentiable with respect
    # to every participating evidence tensor; availability is observational.
    fused_evidence = torch.zeros_like(evidence_stack[:, 0, :])
    has_accumulator = torch.zeros(
        batch_size, device=evidence_stack.device, dtype=torch.bool
    )
    for source_index in range(num_sources):
        source_available = available[:, source_index]
        source_evidence = evidence_stack[:, source_index, :]
        first_source = source_available & ~has_accumulator
        subsequent_source = source_available & has_accumulator
        fused_evidence = torch.where(
            first_source.unsqueeze(-1), source_evidence, fused_evidence
        )
        fused_evidence = torch.where(
            subsequent_source.unsqueeze(-1),
            0.5 * (fused_evidence + source_evidence),
            fused_evidence,
        )
        has_accumulator = has_accumulator | source_available

    # With no available source fused_evidence remains zero, hence alpha == 1
    # and the result is exactly the desired vacuous opinion.
    alpha = fused_evidence + 1.0
    strength = alpha.sum(dim=-1).clamp_min(eps)
    belief = fused_evidence / strength.unsqueeze(-1)
    uncertainty = (
        float(num_classes) / strength
    ).clamp(0.0, 1.0)

    diagnostic_beliefs = [
        norm_beliefs[i] * available_float[:, i].unsqueeze(-1)
        for i in range(num_sources)
    ]
    diagnostic_uncertainties = [
        1.0
        - available_float[:, i]
        * (1.0 - norm_uncertainties[i])
        for i in range(num_sources)
    ]
    diagnostics = {
        "raw_conflict": multisource_conflict(
            diagnostic_beliefs, diagnostic_uncertainties, eps=eps
        ).clamp(0.0, 1.0),
        "ecml_available_views": available_count,
        "ecml_folded_evidence": fused_evidence.sum(dim=-1),
    }
    return belief, uncertainty, diagnostics


def combine_conflict_weighted_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    *,
    availability_masks: list[torch.Tensor] | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Custom conflict-weighted probability aggregation baseline.

    This mechanism is an internal rule ablation, not an ECML implementation.
    It tests whether a view's contribution should decrease when a confident
    prediction conflicts with other confident predictions:

    * convert each opinion to pignistic probabilities;
    * compute pairwise conflict as probability projection distance multiplied
      by the two views' certainty;
    * weight each view by ``certainty * (1 - view_conflict)``;
    * average probabilities with those weights and return an opinion whose
      uncertainty is the residual conflict-penalised uncertainty.

    The rule is source-order invariant and has no dataset-tuned hyperparameter,
    which makes it suitable as a paper baseline against the routed method and
    other evidence-combination rules.
    """
    if not beliefs:
        raise ValueError("combine_conflict_weighted_opinions requires at least one opinion")
    if len(beliefs) != len(uncertainties):
        raise ValueError("beliefs and uncertainties length mismatch")

    norm_beliefs: list[torch.Tensor] = []
    norm_uncertainties: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    certainties: list[torch.Tensor] = []
    for belief, uncertainty in zip(beliefs, uncertainties):
        b, u = _as_masses(belief, uncertainty)
        u = u.view(-1).clamp(0.0, 1.0)
        norm_beliefs.append(b)
        norm_uncertainties.append(u)
        probs.append(opinion_to_prob(b, u, eps=eps))
        certainties.append((1.0 - u).clamp(0.0, 1.0))

    prob_stack = torch.stack(probs, dim=1)  # [B, S, C]
    certainty_stack = torch.stack(certainties, dim=1)  # [B, S]
    num_sources = prob_stack.size(1)
    available = _coerce_availability_masks(
        availability_masks,
        batch_size=prob_stack.size(0),
        num_sources=num_sources,
        device=prob_stack.device,
    )
    if available is None:
        available = certainty_stack > eps
    available_float = available.to(dtype=prob_stack.dtype)
    if num_sources == 1:
        b, u = _as_masses(norm_beliefs[0], norm_uncertainties[0])
        b = b * available_float[:, :1]
        u = 1.0 - available_float[:, :1] * (1.0 - u)
        zero = torch.zeros_like(u.view(-1))
        return b, u.view(-1), {
            "raw_conflict": zero,
            "conflict_weighted_view_conflict_mean": zero,
        }

    pair_conflicts: list[torch.Tensor] = []
    view_conflict = torch.zeros_like(certainty_stack)
    counts = torch.zeros_like(certainty_stack)
    for i in range(num_sources):
        for j in range(i + 1, num_sources):
            # 0.5 * L1 distance is in [0, 1] for probability vectors.
            distance = 0.5 * (prob_stack[:, i, :] - prob_stack[:, j, :]).abs().sum(dim=-1)
            pair_available = available_float[:, i] * available_float[:, j]
            pair = (
                distance
                * certainty_stack[:, i]
                * certainty_stack[:, j]
                * pair_available
            ).clamp(0.0, 1.0)
            pair_conflicts.append(pair)
            view_conflict[:, i] += pair
            view_conflict[:, j] += pair
            counts[:, i] += pair_available
            counts[:, j] += pair_available

    view_conflict = (view_conflict / counts.clamp_min(1.0)).clamp(0.0, 1.0)
    view_reliability = (
        certainty_stack * (1.0 - view_conflict) * available_float
    ).clamp(0.0, 1.0)
    reliability_sum = view_reliability.sum(dim=1, keepdim=True)
    available_count = available_float.sum(dim=1, keepdim=True)
    available_weights = available_float / available_count.clamp_min(1.0)
    weights = torch.where(
        reliability_sum > eps,
        view_reliability / reliability_sum.clamp_min(eps),
        available_weights,
    )

    fused_prob = (weights.unsqueeze(-1) * prob_stack).sum(dim=1)
    fused_certainty = (weights * view_reliability).sum(dim=1).clamp(0.0, 1.0)
    uncertainty = (1.0 - fused_certainty).clamp(0.0, 1.0)
    belief = fused_prob * (1.0 - uncertainty).view(-1, 1)
    belief, uncertainty = _as_masses(belief, uncertainty)

    if pair_conflicts:
        valid_pair_count = (
            available_count.squeeze(-1)
            * (available_count.squeeze(-1) - 1.0)
            / 2.0
        )
        conflictive_raw = (
            torch.stack(pair_conflicts, dim=1).sum(dim=1)
            / valid_pair_count.clamp_min(1.0)
        ).clamp(0.0, 1.0)
    else:
        conflictive_raw = torch.zeros_like(uncertainty)
    diagnostics = {
        "raw_conflict": conflictive_raw,
        "conflict_weighted_view_conflict_mean": view_conflict.mean(dim=1).clamp(0.0, 1.0),
    }
    return belief, uncertainty.view(-1), diagnostics


def combine_opinions_with_diagnostics(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    *,
    rule: str = "dempster",
    availability_masks: list[torch.Tensor] | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Combine opinions and expose raw conflict diagnostics."""
    rule = str(rule).lower()
    raw_conflict = multisource_conflict(beliefs, uncertainties, eps=eps)
    if rule == "ecml":
        belief, uncertainty, diagnostics = combine_ecml_opinions(
            beliefs,
            uncertainties,
            availability_masks=availability_masks,
            eps=eps,
        )
        diagnostics["raw_conflict"] = raw_conflict.clamp(0.0, 1.0)
        return belief, uncertainty, diagnostics
    if rule == "conflict_weighted_opinion":
        belief, uncertainty, diagnostics = combine_conflict_weighted_opinions(
            beliefs,
            uncertainties,
            availability_masks=availability_masks,
            eps=eps,
        )
        # Report the larger of conjunctive conflict and the custom
        # confidence-weighted disagreement.
        diagnostics["raw_conflict"] = torch.maximum(
            raw_conflict.clamp(0.0, 1.0),
            diagnostics["raw_conflict"].clamp(0.0, 1.0),
        )
        return belief, uncertainty, diagnostics

    if rule == "log_pool":
        belief, uncertainty = _combine_log_pool_opinions(
            beliefs,
            uncertainties,
            eps,
            availability_masks=availability_masks,
        )
    else:
        belief, uncertainty = combine_opinions(
            beliefs, uncertainties, rule=rule, eps=eps
        )
    diagnostics = {
        "raw_conflict": raw_conflict.clamp(0.0, 1.0),
    }
    return belief, uncertainty, diagnostics


def opinion_to_prob(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    base_rate: float | torch.Tensor = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project an opinion to a probability (pignistic transform).

    ``p_k = b_k + a_k * u`` with base rate ``a_k`` (uniform by default).
    """
    u = uncertainty.view(-1, 1).clamp(0.0, 1.0)
    if isinstance(base_rate, torch.Tensor):
        a = base_rate.to(device=belief.device, dtype=belief.dtype).view(1, -1)
    else:
        a = torch.full((1, belief.size(-1)), float(base_rate), device=belief.device, dtype=belief.dtype)
    prob = belief.clamp_min(0.0) + a * u
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)


def predictive_opinion_conflict(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    *,
    availability_masks: list[torch.Tensor] | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and maximum confidence-weighted pairwise disagreement.

    For each source pair the score multiplies total-variation distance between
    projected probabilities by both source certainties. It is diagnostic only.
    """
    if not beliefs:
        raise ValueError("predictive_opinion_conflict requires at least one opinion")
    if len(beliefs) != len(uncertainties):
        raise ValueError("beliefs and uncertainties length mismatch")

    probs: list[torch.Tensor] = []
    certainties: list[torch.Tensor] = []
    for belief, uncertainty in zip(beliefs, uncertainties):
        b, u = _as_masses(belief, uncertainty)
        u = u.view(-1).clamp(0.0, 1.0)
        probs.append(opinion_to_prob(b, u, eps=eps))
        certainties.append((1.0 - u).clamp(0.0, 1.0))

    if len(probs) == 1:
        zero = torch.zeros_like(certainties[0])
        return zero, zero

    available = _coerce_availability_masks(
        availability_masks,
        batch_size=probs[0].size(0),
        num_sources=len(probs),
        device=probs[0].device,
    )
    if available is None:
        available = torch.stack(certainties, dim=1) > eps
    pair_scores: list[torch.Tensor] = []
    pair_validity: list[torch.Tensor] = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            distance = 0.5 * (probs[i] - probs[j]).abs().sum(dim=-1)
            pair_valid = available[:, i] & available[:, j]
            pair_scores.append(
                (
                    distance
                    * certainties[i]
                    * certainties[j]
                    * pair_valid.to(dtype=distance.dtype)
                ).clamp(0.0, 1.0)
            )
            pair_validity.append(pair_valid)
    stacked = torch.stack(pair_scores, dim=-1)
    valid_count = torch.stack(pair_validity, dim=-1).sum(dim=-1)
    mean = stacked.sum(dim=-1) / valid_count.clamp_min(1).to(stacked.dtype)
    return mean, stacked.max(dim=-1).values


# ── EDL training objective ────────────────────────────────────────────────────

def _kl_dirichlet_to_uniform(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1, ..., 1) ), closed form (Sensoy et al., 2018)."""
    num_classes = alpha.size(-1)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    ones = torch.ones_like(alpha)
    sum_ones = ones.sum(dim=-1, keepdim=True)
    term1 = torch.lgamma(sum_alpha).squeeze(-1) - torch.lgamma(alpha).sum(dim=-1)
    term2 = -torch.lgamma(sum_ones).squeeze(-1) + torch.lgamma(ones).sum(dim=-1)
    digamma_diff = torch.digamma(alpha) - torch.digamma(sum_alpha)
    term3 = ((alpha - ones) * digamma_diff).sum(dim=-1)
    return term1 + term2 + term3


def dirichlet_expected_ce_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    *,
    anneal_coef: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """TMC/ECML Dirichlet expected cross-entropy with annealed KL.

    This follows the objective used by the official TMC and ECML
    implementations:

    ``sum_k y_k (digamma(S) - digamma(alpha_k))``

    plus an annealed KL divergence from the wrong-class evidence to the
    uniform Dirichlet.  ``alpha`` is accepted directly so the fused opinion's
    evidence strength remains part of the loss.
    """
    if alpha.ndim != 2 or alpha.size(-1) < 2:
        raise ValueError(
            "dirichlet_expected_ce_loss expects alpha shaped [B, C>=2], "
            f"got {tuple(alpha.shape)}"
        )
    if labels.numel() != alpha.size(0):
        raise ValueError("labels must contain one target per Dirichlet row")
    if not bool(torch.isfinite(alpha).all().item()) or bool((alpha <= 0).any().item()):
        raise ValueError("Dirichlet alpha must be finite and strictly positive")

    strength = alpha.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = F.one_hot(
        labels.long().view(-1), num_classes=alpha.size(-1)
    ).to(dtype=alpha.dtype, device=alpha.device)
    expected_ce = (
        target * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    alpha_tilde = target + (1.0 - target) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde, eps=eps)
    per_sample = expected_ce + float(max(anneal_coef, 0.0)) * kl

    if sample_weight is not None:
        weight = sample_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        ).view(-1)
        if weight.numel() != per_sample.numel():
            raise ValueError("sample_weight must contain one value per sample")
        if not bool(torch.isfinite(weight).all().item()) or bool((weight < 0).any().item()):
            raise ValueError("sample_weight must be finite and non-negative")
        denominator = weight.sum()
        weighted = (
            (per_sample * weight).sum() / denominator.clamp_min(eps)
        )
        return weighted * (denominator > 0).to(dtype=weighted.dtype)
    return per_sample.mean()


def evidential_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    anneal_coef: float = 1.0,
    evidence_activation: str = "softplus",
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """EDL Bayes-risk (Brier) loss + annealed KL evidence regulariser.

    ``L = E[ (y - phat)^2 ] + lambda_t * KL( Dir(alpha_tilde) || Dir(1) )``
    where ``alpha_tilde = y + (1 - y) * alpha`` removes evidence on the true
    class from the KL term, so the regulariser only penalises *misleading*
    evidence. ``anneal_coef`` should ramp from 0 to 1 over training to keep clean
    accuracy from collapsing early.
    """
    opinion = logits_to_opinion(logits, evidence_activation=evidence_activation, eps=eps)
    alpha = opinion["alpha"]
    strength = alpha.sum(dim=-1, keepdim=True).clamp_min(eps)
    prob = alpha / strength
    num_classes = logits.size(-1)
    target = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=prob.dtype)
    # Brier / Bayes-risk: sum_k (y_k - p_k)^2 + p_k(1 - p_k) / (S + 1).
    err = ((target - prob) ** 2).sum(dim=-1)
    var = (prob * (1.0 - prob) / (strength + 1.0)).sum(dim=-1)
    bayes_risk = err + var
    alpha_tilde = target + (1.0 - target) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde, eps=eps)
    per_sample = bayes_risk + float(max(anneal_coef, 0.0)) * kl
    if sample_weight is not None:
        w = sample_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        ).view(-1)
        if w.numel() != per_sample.numel():
            raise ValueError("sample_weight must contain one value per sample")
        if not bool(torch.isfinite(w).all().item()) or bool((w < 0).any().item()):
            raise ValueError("sample_weight must be finite and non-negative")
        denominator = w.sum()
        weighted = (per_sample * w).sum() / denominator.clamp_min(eps)
        return weighted * (denominator > 0).to(dtype=weighted.dtype)
    return per_sample.mean()
