"""Evidential (subjective-logic) primitives for the tri-modal robust pipeline.

This module provides the opinion representation, trust discounting, conflict
diagnostics, and the comparison rules used by the pipeline. Dempster,
cumulative fusion, log pooling, and conflictive aggregation are independent
comparison rules; the routed main method does not call them.

Everything here is pure functional tensor code so it can be unit-tested in
isolation and runs under an explicit FP32 context from the caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Primary source modalities that participate in the evidence combination.
# The joint branch is a *learned interaction* of these three embeddings and is
# therefore a deterministic reuse of their information -- combining it here
# would double-count. API and Graph are related code views, so the method does
# not claim statistical independence between all three primary modalities.
EVIDENCE_BRANCHES = ("api", "graph", "manifest")

COMBINATION_RULES = ("dempster", "cumulative", "log_pool", "ecml_style")

def logits_to_softmax_opinion(
    logits: torch.Tensor,
    *,
    uncertainty: float = 0.5,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Map logits to a pseudo subjective-logic opinion without Dirichlet evidence.

    This is used for the no-EDL-opinion ablation: the branch still provides
    class probabilities, but uncertainty is fixed, so conflict-aware fusion can
    be tested without learned Dirichlet evidence.
    """
    if logits.ndim != 2 or logits.size(-1) < 2:
        raise ValueError(f"logits_to_softmax_opinion expects [B, C>=2], got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    u_value = float(uncertainty)
    if not 0.0 <= u_value <= 1.0:
        raise ValueError("uncertainty must be within [0, 1]")

    prob = F.softmax(logits / float(temperature), dim=-1)
    u = torch.full((logits.size(0),), u_value, device=logits.device, dtype=logits.dtype)
    belief = prob * (1.0 - u.view(-1, 1))

    return {
        "evidence": torch.zeros_like(logits),
        "alpha": torch.ones_like(logits),
        "strength": torch.full((logits.size(0),), float(logits.size(-1)), device=logits.device, dtype=logits.dtype),
        "belief": belief,
        "uncertainty": u,
        "expected_prob": prob,
    }

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
    rule: str = "dempster",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two multinomial opinions on a shared binary/k-ary frame.

    All rules return ``(belief, uncertainty)``. Dempster normalises
    conjunctive conflict, while cumulative fusion combines independent
    evidence counts directly.
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
        # Josang aleatory cumulative belief fusion (accumulates evidence).
        denom = (ua + ub - ua * ub).clamp_min(eps)
        belief = (b_a * ub + b_b * ua) / denom
        u = (ua * ub) / denom
        return belief, u.view(-1).clamp(0.0, 1.0)

    # Dempster-Shafer intersection masses.
    # singletons: m(k) = b_a(k) b_b(k) + b_a(k) u_b + u_a b_b(k)
    inter_singleton = b_a * b_b + b_a * ub + ua * b_b
    inter_theta = ua * ub
    total_singleton = inter_singleton.sum(dim=-1, keepdim=True)
    # Conflict mass = everything assigned to the empty set (k != k').
    conflict = (1.0 - total_singleton - inter_theta).clamp_min(0.0)

    if rule == "dempster":
        norm = (1.0 - conflict).clamp_min(eps)
        belief = inter_singleton / norm.view(-1, 1)
        u = inter_theta / norm
        return belief, u.view(-1).clamp(0.0, 1.0)

    raise ValueError(f"Unsupported combination rule: {rule}")


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
    if len(beliefs) == 1:
        belief, u = _as_masses(beliefs[0], uncertainties[0])
        return belief, u.view(-1)
    if rule == "log_pool":
        probs = []
        u_values = []
        for belief, uncertainty in zip(beliefs, uncertainties):
            b, u = _as_masses(belief, uncertainty)
            probs.append(opinion_to_prob(b, u, eps=eps).clamp_min(eps))
            u_values.append(u.clamp(0.0, 1.0))
        log_prob = torch.stack([p.log() for p in probs], dim=0).mean(dim=0)
        prob = torch.softmax(log_prob, dim=-1)
        u = torch.stack(u_values, dim=0).clamp_min(eps).log().mean(dim=0).exp().clamp(0.0, 1.0)
        belief = prob * (1.0 - u.view(-1, 1))
        return belief, u.view(-1)
    if rule == "dempster":
        singleton, theta, conflict, _b_stack, _u_stack = _multisource_intersection(
            beliefs, uncertainties, eps=eps
        )
        norm = (1.0 - conflict).clamp_min(eps)
        belief = singleton / norm.view(-1, 1)
        u = theta / norm
        return belief, u.view(-1).clamp(0.0, 1.0)

    # Cumulative fusion is associative, so pairwise folding is order-stable.
    belief, u = _as_masses(beliefs[0], uncertainties[0])
    belief, u = belief, u.view(-1)
    for next_belief, next_u in zip(beliefs[1:], uncertainties[1:]):
        belief, u = combine_pair(belief, u, next_belief, next_u, rule=rule, eps=eps)
    return belief, u


def _multisource_intersection(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return singleton, theta, conflict, and normalised source masses.

    For sources with singleton class masses ``b_i(k)`` and uncertainty mass
    ``u_i``, the N-way conjunctive singleton mass is

    ``m(k) = prod_i (b_i(k) + u_i) - prod_i u_i``.

    The remaining mass is raw conflict. This is order-independent and is used
    as a diagnostic and rejection signal.
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


def combine_ecml_style_opinions(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """ECML-style conflictive opinion aggregation baseline.

    This is an adapted conflictive multi-view fusion rule rather than a strict
    ECML reproduction. It follows ECML's core intuition that view reliability
    should decrease when a confident view conflicts with other confident views:

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
        raise ValueError("combine_ecml_style_opinions requires at least one opinion")
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
    if num_sources == 1:
        b, u = _as_masses(norm_beliefs[0], norm_uncertainties[0])
        zero = torch.zeros_like(u.view(-1))
        return b, u.view(-1), {
            "raw_conflict": zero,
            "ecml_view_conflict_mean": zero,
        }

    pair_conflicts: list[torch.Tensor] = []
    view_conflict = torch.zeros_like(certainty_stack)
    counts = torch.zeros_like(certainty_stack)
    for i in range(num_sources):
        for j in range(i + 1, num_sources):
            # 0.5 * L1 distance is in [0, 1] for probability vectors.
            distance = 0.5 * (prob_stack[:, i, :] - prob_stack[:, j, :]).abs().sum(dim=-1)
            pair = (distance * certainty_stack[:, i] * certainty_stack[:, j]).clamp(0.0, 1.0)
            pair_conflicts.append(pair)
            view_conflict[:, i] += pair
            view_conflict[:, j] += pair
            counts[:, i] += 1.0
            counts[:, j] += 1.0

    view_conflict = (view_conflict / counts.clamp_min(1.0)).clamp(0.0, 1.0)
    view_reliability = (certainty_stack * (1.0 - view_conflict)).clamp(0.0, 1.0)
    reliability_sum = view_reliability.sum(dim=1, keepdim=True)
    uniform_weights = torch.full_like(view_reliability, 1.0 / float(num_sources))
    weights = torch.where(
        reliability_sum > eps,
        view_reliability / reliability_sum.clamp_min(eps),
        uniform_weights,
    )

    fused_prob = (weights.unsqueeze(-1) * prob_stack).sum(dim=1)
    fused_certainty = (weights * view_reliability).sum(dim=1).clamp(0.0, 1.0)
    uncertainty = (1.0 - fused_certainty).clamp(0.0, 1.0)
    belief = fused_prob * (1.0 - uncertainty).view(-1, 1)
    belief, uncertainty = _as_masses(belief, uncertainty)

    if pair_conflicts:
        conflictive_raw = torch.stack(pair_conflicts, dim=1).mean(dim=1).clamp(0.0, 1.0)
    else:
        conflictive_raw = torch.zeros_like(uncertainty)
    diagnostics = {
        "raw_conflict": conflictive_raw,
        "ecml_view_conflict_mean": view_conflict.mean(dim=1).clamp(0.0, 1.0),
    }
    return belief, uncertainty.view(-1), diagnostics


def combine_opinions_with_diagnostics(
    beliefs: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    *,
    rule: str = "dempster",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Combine opinions and expose raw conflict for rejection diagnostics."""
    rule = str(rule).lower()
    raw_conflict = multisource_conflict(beliefs, uncertainties, eps=eps)
    if rule == "ecml_style":
        belief, uncertainty, diagnostics = combine_ecml_style_opinions(
            beliefs, uncertainties, eps=eps
        )
        # Keep rejection aware of both conjunctive conflict and ECML-style
        # confidence-weighted disagreement.
        diagnostics["raw_conflict"] = torch.maximum(
            raw_conflict.clamp(0.0, 1.0),
            diagnostics["raw_conflict"].clamp(0.0, 1.0),
        )
        return belief, uncertainty, diagnostics

    belief, uncertainty = combine_opinions(beliefs, uncertainties, rule=rule, eps=eps)
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
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and maximum confidence-weighted pairwise disagreement.

    The score is computed from the original branch opinions, before I1 trust
    discounting. For each source pair it multiplies total-variation distance
    between projected probabilities by both source certainties. This exposes
    confident disagreement as a separate routing and diagnostic signal.
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

    pair_scores: list[torch.Tensor] = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            distance = 0.5 * (probs[i] - probs[j]).abs().sum(dim=-1)
            pair_scores.append(
                (distance * certainties[i] * certainties[j]).clamp(0.0, 1.0)
            )
    stacked = torch.stack(pair_scores, dim=-1)
    return stacked.mean(dim=-1), stacked.max(dim=-1).values


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
