"""Evidential (subjective-logic) primitives for the tri-modal robust pipeline.

This module is the formal core of the upgraded I1/I2:

* I1 provides sample-level modality reliability from observable modality 
quality, cross-modal relation evidence, and optional evidential certainty.

* I2 -- per-branch opinions are discounted by their reliability (Jøsang's
  subjective-logic *trust discounting*) and then combined with a
  **conflict-aware** rule. The default rule is Yager's, which routes conflict
  mass to uncertainty instead of normalising it away (classic Dempster-Shafer
  becomes over-confident exactly on the high-conflict degraded samples that are
  this project's headline scenario). Dempster / cumulative fusion are provided
  as ablation baselines.

Everything here is pure functional tensor code so it can be unit-tested in
isolation and runs under an explicit FP32 context from the caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Real, independent modalities that participate in the evidence combination.
# The joint branch is a *learned interaction* of these three embeddings and is
# therefore NOT an independent evidence source -- combining it here would double
# count. It is kept as an auxiliary / baseline head only.
EVIDENCE_BRANCHES = ("api", "graph", "manifest")

COMBINATION_RULES = ("yager", "dempster", "cumulative", "log_pool")


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
    """Jøsang trust discounting of an opinion by a source-reliability ``r``.

    ``b'_k = r * b_k`` and ``u' = 1 - r * (1 - u)``. Lower reliability moves
    belief mass into uncertainty, so an unreliable source contributes little to
    the combined opinion (and raises the rejection signal) rather than voting at
    full strength.
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


def combine_pair(
    belief_a: torch.Tensor,
    u_a: torch.Tensor,
    belief_b: torch.Tensor,
    u_b: torch.Tensor,
    rule: str = "yager",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two multinomial opinions on a shared binary/k-ary frame.

    All rules return ``(belief, uncertainty)`` with the convention that
    ``conflict`` is either reassigned to uncertainty (Yager / cumulative) or
    normalised away (Dempster).
    """
    rule = str(rule).lower()
    b_a, ua = _as_masses(belief_a, u_a)
    b_b, ub = _as_masses(belief_b, u_b)

    if rule == "log_pool":
        # Geometric / product-of-experts pooling on the pignistic probabilities.
        # Kept as a stable ablation baseline; uncertainty is the geometric mean.
        p_a = b_a + 0.5 * ua  # base rate 0.5 absorbed; caller may override frame
        p_b = b_b + 0.5 * ub
        pooled = (p_a.clamp_min(eps).log() + p_b.clamp_min(eps).log()) * 0.5
        prob = torch.softmax(pooled, dim=-1)
        u = (ua * ub).clamp(0.0, 1.0)
        belief = prob * (1.0 - u)
        return belief, u.view(-1)

    if rule == "cumulative":
        # Jøsang aleatory cumulative belief fusion (accumulates evidence).
        denom = (ua + ub - ua * ub).clamp_min(eps)
        belief = (b_a * ub + b_b * ua) / denom
        u = (ua * ub) / denom
        return belief, u.view(-1).clamp(0.0, 1.0)

    # Dempster-Shafer intersection masses (shared by Dempster and Yager).
    # singletons: m(k) = b_a(k) b_b(k) + b_a(k) u_b + u_a b_b(k)
    inter_singleton = b_a * b_b + b_a * ub + ua * b_b
    inter_theta = ua * ub
    total_singleton = inter_singleton.sum(dim=-1, keepdim=True)
    # Conflict mass = everything assigned to the empty set (k != k').
    conflict = (1.0 - total_singleton - inter_theta).clamp_min(0.0)

    if rule == "dempster":
        norm = (1.0 - conflict).clamp_min(eps)
        belief = inter_singleton / norm
        u = inter_theta / norm
        return belief, u.view(-1).clamp(0.0, 1.0)

    if rule == "yager":
        # Conflict mass is routed to uncertainty (Theta) instead of normalising.
        belief = inter_singleton
        u = inter_theta + conflict
        return belief, u.view(-1).clamp(0.0, 1.0)

    raise ValueError(f"Unsupported combination rule: {rule}")


def combine_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    rule: str = "yager",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-fold combination of N opinions.

    Yager's rule is not associative, so the fold order is fixed (api, graph,
    manifest) and documented. For the 3-source binary frame used here the order
    sensitivity is negligible; cumulative fusion is associative.
    """
    if not beliefs:
        raise ValueError("combine_opinions requires at least one opinion")
    if len(beliefs) != len(uncertainties):
        raise ValueError("beliefs and uncertainties length mismatch")
    belief, u = _as_masses(beliefs[0], uncertainties[0])
    belief, u = belief, u.view(-1)
    for next_belief, next_u in zip(beliefs[1:], uncertainties[1:]):
        belief, u = combine_pair(belief, u, next_belief, next_u, rule=rule, eps=eps)
    return belief, u


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
        w = sample_weight.to(dtype=per_sample.dtype).view(-1)
        denom = w.sum().clamp_min(1.0)
        return (per_sample * w).sum() / denom
    return per_sample.mean()
