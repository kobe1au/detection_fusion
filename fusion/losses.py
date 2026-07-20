from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.evidential import dirichlet_expected_ce_loss, evidential_loss


BRANCH_AUX_KEYS = (
    "api_logits_aux",
    "graph_logits_aux",
    "manifest_logits_aux",
)

BRANCH_AUX_NAMES = {
    "api_logits_aux": "api",
    "graph_logits_aux": "graph",
    "manifest_logits_aux": "manifest",
}
BRANCH_NAMES = ("api", "graph", "manifest")
ROUTING_BRANCH_NAMES = ("api", "graph", "manifest")
ROUTING_PROBABILITY_SUBSETS = (
    ("api",),
    ("graph",),
    ("manifest",),
    ("api", "graph"),
    ("api", "manifest"),
    ("graph", "manifest"),
    ("api", "graph", "manifest"),
)
AUXILIARY_WEIGHT_MODES = (
    "integrity",
    "alive_masked_uniform",
    "unmasked_uniform",
)
PAPER_EVIDENTIAL_OBJECTIVES = ("tmc", "ecml")
REMOVED_LOSS_CONFIG_KEYS = frozenset(
    {
        "reliability_weighted_aux",
        "integrity_weighted_aux",
        "semantic_reconstruction_weight",
        "cross_source_consistency_weight",
        "gate_prior_weight",
    }
)
ROUTING_RISK_TARGETS = (
    "mixture_argmax_error",
    "threshold_classification_error",
    "threshold_malware_false_negative",
    "reliability_deficit_score",
)


def _evidence_column(evidence: torch.Tensor, index: int, detach: bool = False) -> torch.Tensor:
    value = evidence[:, index].clamp(0.0, 1.0)
    return value.detach() if detach else value


def _branch_alive_masks(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
    api = _evidence_column(evidence, EvidenceIndex.API_ALIVE, True)
    graph = _evidence_column(evidence, EvidenceIndex.GRAPH_ALIVE, True)
    manifest = _evidence_column(evidence, EvidenceIndex.MANIFEST_ALIVE, True)
    return {
        "api": api,
        "graph": graph,
        "manifest": manifest,
    }


def resolve_auxiliary_weight_mode(loss_cfg: dict | None) -> str:
    """Resolve the explicit branch-auxiliary weighting mechanism."""

    loss_cfg = loss_cfg or {}
    removed = sorted(set(loss_cfg) & REMOVED_LOSS_CONFIG_KEYS)
    if removed:
        raise ValueError(
            "Removed loss configuration keys are unsupported: "
            f"{removed}. Use loss.auxiliary_weight_mode for branch weighting."
        )
    mode = str(
        loss_cfg.get("auxiliary_weight_mode", "unmasked_uniform")
    ).strip().lower()
    if mode not in AUXILIARY_WEIGHT_MODES:
        raise ValueError(
            "loss.auxiliary_weight_mode must be one of "
            f"{AUXILIARY_WEIGHT_MODES}, got {mode!r}"
        )
    return mode


def _paper_dirichlet_alphas(outputs: dict) -> dict[str, torch.Tensor]:
    """Return the three view alphas and fused alpha required by TMC/ECML."""
    alphas: dict[str, torch.Tensor] = {}
    for name in (*BRANCH_NAMES[:3], "fused"):
        value = outputs.get(f"dirichlet_alpha_{name}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "TMC/ECML-style adapted objectives require differentiable "
                f"dirichlet_alpha_{name} from evidential fusion"
            )
        if value.ndim != 2 or value.size(-1) < 2:
            raise ValueError(
                f"dirichlet_alpha_{name} must have shape [B, C>=2], "
                f"got {tuple(value.shape)}"
            )
        alphas[name] = value
    reference_shape = alphas["fused"].shape
    if any(value.shape != reference_shape for value in alphas.values()):
        raise ValueError("all TMC/ECML Dirichlet alphas must have the same shape")
    return alphas


def _ecml_conflict_consistency_loss(
    alphas: dict[str, torch.Tensor],
    alive: dict[str, torch.Tensor],
    *,
    mask_unavailable: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """ECML projected-distance times conjunctive-certainty regularizer.

    For all views available this is exactly
    ``1/(V-1) * sum_i sum_{j!=i} c(omega_i, omega_j)``. Missing APK views are
    excluded by a detached availability mask; this is the only protocol
    extension beyond ECML's complete-view training setting.
    """
    names = BRANCH_NAMES[:3]
    probabilities: dict[str, torch.Tensor] = {}
    certainties: dict[str, torch.Tensor] = {}
    for name in names:
        alpha = alphas[name]
        strength = alpha.sum(dim=-1, keepdim=True).clamp_min(eps)
        probabilities[name] = alpha / strength
        certainties[name] = (
            1.0 - float(alpha.size(-1)) / strength.view(-1)
        ).clamp(0.0, 1.0)

    pair_sum = alphas["fused"].new_zeros(alphas["fused"].size(0))
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            projected_distance = 0.5 * (
                probabilities[left] - probabilities[right]
            ).abs().sum(dim=-1)
            pair = projected_distance * certainties[left] * certainties[right]
            if mask_unavailable:
                pair = pair * alive[left].view(-1) * alive[right].view(-1)
            pair_sum = pair_sum + pair

    if mask_unavailable:
        available_count = torch.stack(
            [alive[name].view(-1) for name in names], dim=-1
        ).sum(dim=-1)
        valid = (available_count >= 2.0).to(dtype=pair_sum.dtype)
        # Preserve ECML's ordered-pair normalisation after excluding unavailable
        # views: each unordered pair occurs twice and the denominator is the
        # number of *available* alternatives, V_eff - 1.
        per_sample = 2.0 * pair_sum / (available_count - 1.0).clamp_min(1.0)
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)
    # The paper sums both (i,j) and (j,i), then divides by V-1.
    per_sample = (2.0 / float(len(names) - 1)) * pair_sum
    return per_sample.mean()


def _compute_paper_evidential_objective(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor | None,
    loss_cfg: dict,
    *,
    objective: str,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Compute the shared-encoder TMC/ECML-style adapted training loss."""
    if objective not in PAPER_EVIDENTIAL_OBJECTIVES:
        raise ValueError(f"unsupported paper evidential objective: {objective}")
    if not isinstance(evidence, torch.Tensor):
        raise ValueError(f"loss.objective={objective} requires observable evidence masks")

    incompatible = {
        "branch_aux_weight": float(loss_cfg.get("branch_aux_weight", 0.0)),
        "reliability_calibration_weight": float(
            loss_cfg.get("reliability_calibration_weight", 0.0)
        ),
        "probability_calibration_weight": float(
            loss_cfg.get("probability_calibration_weight", 0.0)
        ),
        "evidential_loss_weight": float(loss_cfg.get("evidential_loss_weight", 0.0)),
        "label_smoothing": float(loss_cfg.get("label_smoothing", 0.0)),
    }
    nonzero = {name: value for name, value in incompatible.items() if value != 0.0}
    if nonzero:
        raise ValueError(
            f"loss.objective={objective} replaces generic CE/auxiliary objectives; "
            f"set incompatible weights to zero: {nonzero}"
        )

    method_cfg = loss_cfg.get(objective, {}) or {}
    anneal_epochs = max(int(method_cfg.get("anneal_epochs", 10)), 1)
    anneal_coef = min(1.0, float(max(epoch, 0)) / float(anneal_epochs))
    mask_unavailable = bool(method_cfg.get("mask_unavailable_views", True))
    alphas = _paper_dirichlet_alphas(outputs)
    alive = _branch_alive_masks(evidence)
    view_losses: dict[str, torch.Tensor] = {}
    for name in BRANCH_NAMES[:3]:
        sample_weight = alive[name] if mask_unavailable else None
        view_losses[name] = dirichlet_expected_ce_loss(
            alphas[name],
            labels,
            anneal_coef=anneal_coef,
            sample_weight=sample_weight,
        )

    fused_weight = None
    if mask_unavailable:
        fused_weight = torch.stack(
            [alive[name] for name in BRANCH_NAMES[:3]], dim=-1
        ).sum(dim=-1).gt(0).to(dtype=alphas["fused"].dtype)
    fused_loss = dirichlet_expected_ce_loss(
        alphas["fused"],
        labels,
        anneal_coef=anneal_coef,
        sample_weight=fused_weight,
    )
    view_sum = torch.stack(list(view_losses.values())).sum()

    consistency = fused_loss.new_zeros(())
    if objective == "tmc":
        # Official TMC sums every view objective and the fused objective.
        total = view_sum + fused_loss
    else:
        # Official ECML averages the V view losses plus the fused loss and adds
        # the decision-conflict consistency term with gamma=1 by default.
        gamma = float(method_cfg.get("consistency_weight", 1.0))
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("loss.ecml.consistency_weight must be finite and non-negative")
        consistency = _ecml_conflict_consistency_loss(
            alphas,
            alive,
            mask_unavailable=mask_unavailable,
        )
        total = (view_sum + fused_loss) / float(len(view_losses) + 1)
        total = total + gamma * consistency

    parts: dict[str, float | torch.Tensor] = {
        "loss": total.detach(),
        "ce": total.new_zeros(()),
        "branch_aux": total.new_zeros(()),
        "branch_aux_weight": 0.0,
        "dirichlet_fused_loss": fused_loss.detach(),
        "dirichlet_view_loss": view_sum.detach(),
        "ecml_conflict_consistency_loss": consistency.detach(),
        "evidential_anneal_coef": total.new_tensor(anneal_coef),
        "tmc_objective_active": total.new_tensor(float(objective == "tmc")),
        "ecml_objective_active": total.new_tensor(float(objective == "ecml")),
    }
    for name, value in view_losses.items():
        parts[f"dirichlet_loss_{name}"] = value.detach()
    return total, parts


def routing_mixture_log_prob(
    outputs: dict,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return I2's conditional class mixture, independent of fused risk."""
    if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 0.5:
        raise ValueError("routing conditional-mixture eps must be within (0, 0.5)")
    mixture = outputs.get("routing_mixture_prob")
    if not isinstance(mixture, torch.Tensor):
        raise ValueError("routing mixture loss requires routing_mixture_prob")
    if mixture.ndim != 2 or mixture.size(-1) < 2:
        raise ValueError("routing_mixture_prob must have shape [B, C>=2]")
    mixture = mixture.float()
    mixture = mixture.clamp_min(float(eps))
    mixture = mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return mixture.log()


def routing_soft_oracle_target(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build I2's detached branch oracle and its per-row valid mask.

    The oracle assigns more mass to an available branch when that branch gives
    the labelled class a lower NLL.  Keeping target construction separate from
    reduction lets the post-hoc fitter compile it once and apply the exact
    source/family row weights without duplicating the oracle semantics.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("fusion.routing.route_oracle_temperature must be finite and positive")
    if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 0.5:
        raise ValueError("routing soft-oracle eps must be within (0, 0.5)")
    if not isinstance(evidence, torch.Tensor):
        raise ValueError("routing soft-oracle loss requires observable evidence")

    branch_distribution = outputs.get("routing_branch_distribution")
    if not isinstance(branch_distribution, torch.Tensor):
        raise ValueError(
            "routing soft-oracle loss requires routing_branch_distribution"
        )
    expected_shape = (labels.numel(), len(ROUTING_BRANCH_NAMES))
    if branch_distribution.shape != expected_shape:
        raise ValueError(
            "routing_branch_distribution must have shape "
            f"{expected_shape}, got {tuple(branch_distribution.shape)}"
        )
    alive_by_branch = _branch_alive_masks(evidence)
    alive_stack = torch.stack(
        [alive_by_branch[name].view(-1) for name in ROUTING_BRANCH_NAMES], dim=-1
    ).to(device=branch_distribution.device, dtype=torch.float32)
    if alive_stack.shape != expected_shape:
        raise ValueError("routing alive-mask shape disagrees with labels")
    available = alive_stack > 0.0
    valid = available.any(dim=-1)

    true_class_nll = []
    label_index = labels.long().view(-1, 1)
    for name in ROUTING_BRANCH_NAMES:
        branch_log_prob = outputs.get(f"calibrated_log_prob_{name}")
        if not isinstance(branch_log_prob, torch.Tensor):
            raise ValueError(
                "routing soft-oracle loss requires "
                f"calibrated_log_prob_{name}"
            )
        if branch_log_prob.ndim != 2 or branch_log_prob.size(0) != labels.numel():
            raise ValueError(
                f"calibrated_log_prob_{name} must have shape [B, C]"
            )
        true_class_nll.append(
            -branch_log_prob.detach().float().gather(1, label_index).squeeze(1)
        )
    nll_stack = torch.stack(true_class_nll, dim=-1)

    oracle_logits = (-nll_stack / float(temperature)).masked_fill(
        ~available, torch.finfo(torch.float32).min
    )
    oracle_target = F.softmax(oracle_logits, dim=-1)
    oracle_target = torch.where(
        valid.unsqueeze(-1), oracle_target, torch.zeros_like(oracle_target)
    ).detach()
    return oracle_target, valid.detach()


def routing_source_subset_oracle_target(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    *,
    source_segments: Sequence[tuple[int, int]] | None = None,
    source_names: Sequence[str] | None = None,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build a fold-local soft oracle over the seven probability subsets.

    Unlike a per-row best-subset target (which necessarily collapses to the
    best singleton for an arithmetic probability mixture), this oracle scores
    each subset over a complete calibration source.  It can therefore identify
    complementary branches whose aggregate NLL is lower across that source.
    Targets are recomputed inside every post-hoc fit and are detached from the
    branch predictions; only the route scores receive oracle gradients.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(
            "fusion.routing.subset_oracle_temperature must be finite and positive"
        )
    if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 0.5:
        raise ValueError("routing source-subset oracle eps must be within (0, 0.5)")
    if not isinstance(evidence, torch.Tensor):
        raise ValueError("routing source-subset oracle requires observable evidence")

    routing_scores = outputs.get("routing_scores")
    if not isinstance(routing_scores, torch.Tensor):
        raise ValueError("routing source-subset oracle requires routing_scores")
    batch_size = int(labels.numel())
    expected_route_shape = (batch_size, len(ROUTING_BRANCH_NAMES))
    if tuple(routing_scores.shape) != expected_route_shape:
        raise ValueError(
            "routing_scores must have shape "
            f"{expected_route_shape}, got {tuple(routing_scores.shape)}"
        )

    labels = labels.detach().long().view(-1)
    branch_log_probs: list[torch.Tensor] = []
    class_count: int | None = None
    for name in ROUTING_BRANCH_NAMES:
        value = outputs.get(f"calibrated_log_prob_{name}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "routing source-subset oracle requires "
                f"calibrated_log_prob_{name}"
            )
        value = value.detach().float()
        if value.ndim != 2 or int(value.size(0)) != batch_size:
            raise ValueError(f"calibrated_log_prob_{name} must have shape [B, C]")
        if class_count is None:
            class_count = int(value.size(1))
        elif int(value.size(1)) != class_count:
            raise ValueError("calibrated branch class dimensions disagree")
        branch_log_probs.append(value - torch.logsumexp(value, dim=-1, keepdim=True))
    assert class_count is not None
    if class_count < 2:
        raise ValueError("routing source-subset oracle requires at least two classes")
    if batch_size and bool(((labels < 0) | (labels >= class_count)).any().item()):
        raise ValueError("routing source-subset oracle labels are outside the class frame")

    if source_segments is None:
        segments = ((0, batch_size),)
    else:
        segments = tuple((int(start), int(end)) for start, end in source_segments)
        if not segments:
            raise ValueError("routing source-subset oracle requires non-empty segments")
    expected_start = 0
    for start, end in segments:
        if start != expected_start or end <= start or end > batch_size:
            raise ValueError(
                "routing source-subset oracle segments must be contiguous, "
                "non-empty, and cover the packed batch exactly"
            )
        expected_start = end
    if expected_start != batch_size:
        raise ValueError(
            "routing source-subset oracle segments must cover the packed batch exactly"
        )
    if source_names is None:
        resolved_source_names = tuple(f"source_{index}" for index in range(len(segments)))
    else:
        resolved_source_names = tuple(str(value) for value in source_names)
        if len(resolved_source_names) != len(segments):
            raise ValueError(
                "routing source-subset oracle source_names must match source_segments"
            )

    alive_by_branch = _branch_alive_masks(evidence.detach())
    alive = torch.stack(
        [alive_by_branch[name].view(-1) for name in ROUTING_BRANCH_NAMES], dim=-1
    ).to(device=routing_scores.device).gt(0.0)
    if tuple(alive.shape) != expected_route_shape:
        raise ValueError("routing source-subset alive-mask shape disagrees with labels")

    log_prob_stack = torch.stack(branch_log_probs, dim=1).to(
        device=routing_scores.device
    )
    label_index = labels.to(device=routing_scores.device).view(-1, 1, 1)
    true_class_log_prob = log_prob_stack.gather(
        2, label_index.expand(-1, len(ROUTING_BRANCH_NAMES), 1)
    ).squeeze(-1)
    membership = torch.tensor(
        [
            [name in subset for name in ROUTING_BRANCH_NAMES]
            for subset in ROUTING_PROBABILITY_SUBSETS
        ],
        device=routing_scores.device,
        dtype=torch.bool,
    )

    target = routing_scores.new_zeros(expected_route_shape, dtype=torch.float32)
    valid = torch.zeros(batch_size, device=routing_scores.device, dtype=torch.bool)
    source_count = len(segments)
    subset_count = len(ROUTING_PROBABILITY_SUBSETS)
    candidate_nll = torch.full(
        (source_count, subset_count),
        torch.inf,
        device=routing_scores.device,
        dtype=torch.float32,
    )
    candidate_mass = torch.zeros_like(candidate_nll)
    hard_best_subset_index = torch.full(
        (source_count,), -1, device=routing_scores.device, dtype=torch.long
    )
    source_valid_count = torch.zeros(
        source_count, device=routing_scores.device, dtype=torch.long
    )
    eligible_candidate_count = torch.zeros(
        source_count, device=routing_scores.device, dtype=torch.long
    )
    target_branch_mass = torch.zeros(
        (source_count, len(ROUTING_BRANCH_NAMES)),
        device=routing_scores.device,
        dtype=torch.float32,
    )
    best_second_gap = torch.full(
        (source_count,), torch.nan, device=routing_scores.device, dtype=torch.float32
    )

    for source_index, (start, end) in enumerate(segments):
        source_alive = alive[start:end]
        globally_valid = source_alive.any(dim=-1)
        valid_count = int(globally_valid.sum().item())
        source_valid_count[source_index] = valid_count
        if valid_count <= 0:
            continue

        effective_membership = source_alive.unsqueeze(1) & membership.unsqueeze(0)
        active_count = effective_membership.sum(dim=-1)
        row_supported = active_count > 0
        # A candidate must cover every globally valid row in the source.  It is
        # not allowed to look artificially good by dropping native-missing rows.
        eligible = ((~globally_valid).unsqueeze(-1) | row_supported).all(dim=0)
        eligible_count = int(eligible.sum().item())
        eligible_candidate_count[source_index] = eligible_count
        if eligible_count <= 0:
            raise RuntimeError(
                "routing source-subset oracle found no candidate covering a valid source"
            )
        if eligible_count == 1:
            raise RuntimeError(
                "routing source-subset oracle is not identifiable for source "
                f"{resolved_source_names[source_index]!r} (index {source_index}): "
                "only one of the seven candidates covers "
                "all valid rows. Audit native modality availability instead of "
                "silently treating the forced candidate as an oracle preference."
            )

        selected_log_prob = true_class_log_prob[start:end].unsqueeze(1).masked_fill(
            ~effective_membership, -torch.inf
        )
        subset_true_log_prob = torch.logsumexp(selected_log_prob, dim=-1) - active_count.clamp_min(
            1
        ).to(dtype=torch.float32).log()
        safe_nll = torch.where(
            globally_valid.unsqueeze(-1) & row_supported,
            -subset_true_log_prob,
            torch.zeros_like(subset_true_log_prob),
        )
        source_candidate_nll = safe_nll.sum(dim=0) / float(valid_count)
        source_candidate_nll = source_candidate_nll.masked_fill(~eligible, torch.inf)
        candidate_nll[source_index] = source_candidate_nll

        oracle_logits = (-source_candidate_nll / float(temperature)).masked_fill(
            ~eligible, -torch.inf
        )
        source_candidate_mass = F.softmax(oracle_logits, dim=0)
        candidate_mass[source_index] = source_candidate_mass
        hard_best_subset_index[source_index] = int(
            source_candidate_nll.argmin().item()
        )
        finite_nll = source_candidate_nll[eligible]
        if finite_nll.numel() >= 2:
            best_two = torch.topk(finite_nll, k=2, largest=False).values
            best_second_gap[source_index] = best_two[1] - best_two[0]

        candidate_branch_weights = effective_membership.to(dtype=torch.float32) / (
            active_count.unsqueeze(-1).clamp_min(1).to(dtype=torch.float32)
        )
        source_target = torch.einsum(
            "k,nkm->nm", source_candidate_mass, candidate_branch_weights
        )
        source_target = torch.where(
            globally_valid.unsqueeze(-1),
            source_target
            / source_target.sum(dim=-1, keepdim=True).clamp_min(float(eps)),
            torch.zeros_like(source_target),
        )
        target[start:end] = source_target
        valid[start:end] = globally_valid
        target_branch_mass[source_index] = source_target[globally_valid].mean(dim=0)

    diagnostics = {
        "candidate_nll": candidate_nll.detach(),
        "candidate_mass": candidate_mass.detach(),
        "hard_best_subset_index": hard_best_subset_index.detach(),
        "source_valid_count": source_valid_count.detach(),
        "eligible_candidate_count": eligible_candidate_count.detach(),
        "target_branch_mass": target_branch_mass.detach(),
        "best_second_gap": best_second_gap.detach(),
    }
    return target.detach(), valid.detach(), diagnostics


def routing_subset_oracle_per_sample_loss(
    outputs: dict,
    oracle_target: torch.Tensor,
) -> torch.Tensor:
    """Return score-level CE for the detached source-subset route target."""
    routing_scores = outputs.get("routing_scores")
    if not isinstance(routing_scores, torch.Tensor):
        raise ValueError("routing source-subset oracle loss requires routing_scores")
    if routing_scores.shape != oracle_target.shape:
        raise ValueError(
            "routing_scores and source-subset target must have the same shape, got "
            f"{tuple(routing_scores.shape)} and {tuple(oracle_target.shape)}"
        )
    route_log_prob = F.log_softmax(routing_scores.float(), dim=-1)
    return -(oracle_target.detach().float() * route_log_prob).sum(dim=-1)


def routing_soft_oracle_per_sample_loss(
    outputs: dict,
    oracle_target: torch.Tensor,
) -> torch.Tensor:
    """Return unreduced soft-oracle cross entropy for every packed row."""
    routing_scores = outputs.get("routing_scores")
    if not isinstance(routing_scores, torch.Tensor):
        raise ValueError("routing soft-oracle loss requires routing_scores")
    if routing_scores.shape != oracle_target.shape:
        raise ValueError(
            "routing_scores and oracle_target must have the same shape, got "
            f"{tuple(routing_scores.shape)} and {tuple(oracle_target.shape)}"
        )
    route_log_prob = F.log_softmax(routing_scores.float(), dim=-1)
    return -(oracle_target.detach().float() * route_log_prob).sum(dim=-1)


def routing_soft_oracle_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match ``pi`` to a detached soft oracle built from branch true-class NLL.

    The source-level public reduction is retained for diagnostics and external
    callers.  Training's packed fast path consumes the same target and
    per-sample helper, then applies precompiled source-normalized row weights.
    """
    oracle_target, valid = routing_soft_oracle_target(
        outputs,
        labels,
        evidence,
        temperature=temperature,
        eps=eps,
    )
    per_sample = routing_soft_oracle_per_sample_loss(outputs, oracle_target)

    valid_weight = valid.to(dtype=per_sample.dtype)
    loss = (per_sample * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    if not materialize_diagnostics:
        return loss, {}
    target_entropy = -(oracle_target * oracle_target.clamp_min(float(eps)).log()).sum(
        dim=-1
    )
    top1_agreement = (
        outputs["routing_branch_distribution"].detach().argmax(dim=-1)
        == oracle_target.argmax(dim=-1)
    ).float()
    diagnostics = {
        "routing_route_oracle_loss": loss.detach(),
        "routing_route_oracle_target_entropy": (
            (target_entropy * valid_weight).sum()
            / valid_weight.sum().clamp_min(1.0)
        ),
        "routing_route_oracle_top1_agreement": (
            (top1_agreement * valid_weight).sum()
            / valid_weight.sum().clamp_min(1.0)
        ),
        "routing_route_oracle_valid_sample_count": valid.sum().detach(),
    }
    return loss, diagnostics


def routing_risk_target(
    outputs: dict,
    labels: torch.Tensor,
    routing_config: dict | None = None,
    *,
    mixture_log_prob: torch.Tensor | None = None,
    valid_routing: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Build the detached I2 risk target and its exact training mask.

    In particular, ``threshold_malware_false_negative`` estimates malware
    probability conditional on the fixed classifier predicting benign.  The
    predicted-malware side is excluded from the fitting denominator rather
    than added as abundant zero targets.  Both the public source-level loss and
    the packed post-hoc optimizer call this helper so that this contract cannot
    drift between paths.
    """
    routing_config = routing_config or {}
    risk_loss_type = str(routing_config.get("risk_loss", "bce")).strip().lower()
    risk_target_type = str(
        routing_config.get("risk_target", "mixture_argmax_error")
    ).strip().lower()
    if risk_loss_type not in {"bce", "brier"}:
        raise ValueError("fusion.routing.risk_loss must be 'bce' or 'brier'")
    if risk_target_type not in ROUTING_RISK_TARGETS:
        raise ValueError(
            "fusion.routing.risk_target must be one of "
            f"{list(ROUTING_RISK_TARGETS)}, got {risk_target_type!r}"
        )
    if risk_target_type == "reliability_deficit_score":
        raise ValueError(
            "fusion.routing.risk_target='reliability_deficit_score' is an "
            "unfitted control score and cannot supervise a learned risk head"
        )

    labels = labels.long().view(-1)
    if mixture_log_prob is None:
        mixture_log_prob = routing_mixture_log_prob(outputs)
    if mixture_log_prob.ndim != 2 or int(mixture_log_prob.size(0)) != labels.numel():
        raise ValueError("routing mixture batch shape disagrees with labels")

    if valid_routing is None:
        has_available = outputs.get("routing_has_available")
        if not isinstance(has_available, torch.Tensor):
            raise ValueError("routing calibration requires routing_has_available")
        valid_routing = has_available.detach().view(-1).bool()
    else:
        valid_routing = valid_routing.detach().view(-1).bool()
    if valid_routing.numel() != labels.numel():
        raise ValueError("routing_has_available batch shape disagrees with labels")

    risk_valid = valid_routing.clone()
    if risk_target_type == "mixture_argmax_error":
        error_target = mixture_log_prob.detach().argmax(dim=-1).ne(labels).float()
    else:
        raw_threshold = routing_config.get("classification_log_odds_threshold")
        if raw_threshold is None:
            raise ValueError(
                f"fusion.routing.risk_target={risk_target_type!r} requires a "
                "fitted classification_log_odds_threshold"
            )
        raw_threshold = float(raw_threshold)
        if not math.isfinite(raw_threshold):
            raise ValueError(
                "fusion.routing.classification_log_odds_threshold must be finite"
            )
        raw_log_prob = outputs.get("uncalibrated_final_log_prob")
        if not isinstance(raw_log_prob, torch.Tensor) or (
            raw_log_prob.ndim != 2
            or raw_log_prob.shape != mixture_log_prob.shape
            or raw_log_prob.size(-1) != 2
        ):
            raise ValueError(
                "threshold-aligned routing risk requires "
                "uncalibrated_final_log_prob with shape [B, 2]"
            )
        threshold_prediction = (
            raw_log_prob.detach()[:, 1] - raw_log_prob.detach()[:, 0]
            >= raw_threshold
        )
        if risk_target_type == "threshold_classification_error":
            error_target = threshold_prediction.ne(labels).float()
        else:
            error_target = (labels.eq(1) & ~threshold_prediction).float()
            risk_valid &= ~threshold_prediction

    return (
        error_target.detach(),
        risk_valid.detach(),
        risk_loss_type,
        risk_target_type,
    )


def routing_risk_per_sample_loss(
    predicted_error: torch.Tensor,
    predicted_error_logit: torch.Tensor | None,
    error_target: torch.Tensor,
    risk_valid: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    """Return the unreduced risk-head loss with invalid rows made numerically safe."""
    loss_type = str(loss_type).strip().lower()
    if loss_type not in {"bce", "brier"}:
        raise ValueError("fusion.routing.risk_loss must be 'bce' or 'brier'")
    predicted_error = predicted_error.float().view(-1)
    error_target = error_target.detach().float().view(-1)
    risk_valid = risk_valid.detach().view(-1).bool()
    if not (
        predicted_error.numel() == error_target.numel() == risk_valid.numel()
    ):
        raise ValueError("routing risk tensors disagree on their batch shape")
    safe_target = torch.where(
        risk_valid, error_target, torch.zeros_like(error_target)
    )
    if loss_type == "brier":
        return (predicted_error - safe_target).square()
    if not isinstance(predicted_error_logit, torch.Tensor):
        raise ValueError("routing BCE risk loss requires routing_risk_logit")
    predicted_error_logit = predicted_error_logit.float().view(-1)
    if predicted_error_logit.numel() != error_target.numel():
        raise ValueError("routing risk logit batch shape disagrees with its target")
    safe_logit = torch.where(
        risk_valid,
        predicted_error_logit,
        torch.zeros_like(predicted_error_logit),
    )
    return F.binary_cross_entropy_with_logits(
        safe_logit, safe_target, reduction="none"
    )


def reliability_correctness_target(
    branch_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return the detached branch-correctness target used by I1."""
    if not isinstance(branch_logits, torch.Tensor) or branch_logits.ndim != 2:
        raise ValueError("I1 branch logits must have shape [B, C]")
    labels = labels.long().view(-1)
    if int(branch_logits.size(0)) != int(labels.numel()):
        raise ValueError("I1 branch logits and labels disagree on batch shape")
    return branch_logits.detach().argmax(dim=-1).eq(labels).float()


def reliability_per_sample_loss(
    predicted_reliability: torch.Tensor,
    predicted_reliability_logit: torch.Tensor | None,
    correctness: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    """Return the unreduced proper loss used by both packed and public I1."""
    loss_type = str(loss_type).strip().lower()
    if loss_type not in {"bce", "brier"}:
        raise ValueError("reliability_calibration.loss must be 'bce' or 'brier'")
    reliability = predicted_reliability.view(-1).float().clamp(
        1.0e-6, 1.0 - 1.0e-6
    )
    correctness = correctness.detach().float().view(-1)
    if reliability.numel() != correctness.numel():
        raise ValueError("I1 reliability and correctness disagree on batch shape")
    if loss_type == "brier":
        return (reliability - correctness).square()
    # The formal I1 path exports its raw correctness logit.  Using the fused
    # BCE kernel keeps LBFGS strong-Wolfe line searches stable at extreme
    # logits and avoids sigmoid/clamp saturation.  Fixtures that expose only a
    # probability retain the finite logit-equivalent fallback.
    if isinstance(predicted_reliability_logit, torch.Tensor):
        bce_logit = predicted_reliability_logit.view(-1).float()
        if bce_logit.numel() != correctness.numel():
            raise ValueError("I1 reliability logit batch shape disagrees with target")
    else:
        bce_logit = torch.logit(reliability)
    return F.binary_cross_entropy_with_logits(
        bce_logit,
        correctness,
        reduction="none",
    )


def reliability_alive_mask(evidence: torch.Tensor, branch: str) -> torch.Tensor:
    """Return the exact detached availability mask used by the public I1 loss."""
    branch = str(branch).strip().lower()
    if branch not in BRANCH_NAMES:
        raise ValueError(f"unsupported I1 branch {branch!r}")
    return _branch_alive_masks(evidence)[branch].view(-1)


def compute_reliability_calibration_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
    *,
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise predicted reliability with observed branch correctness."""
    config = config or {}
    if not isinstance(evidence, torch.Tensor):
        raise ValueError("reliability calibration requires observable evidence")
    alive = _branch_alive_masks(evidence)
    loss_type = str(config.get("loss", "bce")).lower()
    if loss_type not in {"bce", "brier"}:
        raise ValueError("reliability_calibration.loss must be 'bce' or 'brier'")
    configured_branches = config.get("branches", BRANCH_NAMES)
    if not isinstance(configured_branches, (list, tuple)) or not configured_branches:
        raise ValueError("reliability_calibration.branches must be a non-empty list")
    branch_names = tuple(str(name).lower() for name in configured_branches)
    invalid = [name for name in branch_names if name not in BRANCH_NAMES]
    if invalid:
        raise ValueError(
            "reliability_calibration.branches contains unsupported branches: "
            f"{invalid}"
        )
    if len(set(branch_names)) != len(branch_names):
        raise ValueError("reliability_calibration.branches must not contain duplicates")
    losses = []
    diagnostics: dict[str, torch.Tensor] = {}
    ref = labels.new_zeros((), dtype=torch.float32)
    for name in branch_names:
        reliability = outputs.get(f"predicted_reliability_{name}")
        reliability_logit = outputs.get(f"predicted_reliability_logit_{name}")
        logits = outputs.get(f"{name}_logits_aux")
        if not isinstance(reliability, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise ValueError(
                f"reliability calibration branch {name!r} is missing its "
                "predicted reliability or auxiliary logits"
            )
        reliability = reliability.view(-1).float().clamp(1.0e-6, 1.0 - 1.0e-6)
        correctness = reliability_correctness_target(logits, labels)
        weight = alive[name].view(-1).float()
        if materialize_diagnostics and not bool((weight.sum() > 0).item()):
            continue
        per_sample = reliability_per_sample_loss(
            reliability,
            reliability_logit,
            correctness,
            loss_type=loss_type,
        )
        denom = weight.sum().clamp_min(1.0)
        proper_loss = (per_sample * weight).sum() / denom
        mean_reliability = (reliability * weight).sum() / denom
        branch_accuracy = (correctness * weight).sum() / denom
        branch_loss = proper_loss
        losses.append(branch_loss)
        diagnostics[f"reliability_loss_{name}"] = branch_loss.detach()
        diagnostics[f"reliability_proper_loss_{name}"] = proper_loss.detach()
        diagnostics[f"mean_predicted_reliability_{name}"] = mean_reliability.detach()
        diagnostics[f"branch_accuracy_{name}"] = branch_accuracy.detach()
    total = torch.stack(losses).mean() if losses else ref
    if not materialize_diagnostics:
        return total, {}
    diagnostics["reliability_calibration_loss"] = total.detach()
    return total, diagnostics


def compute_probability_calibration_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Temperature-scaling NLL over available branch predictions."""
    if not isinstance(evidence, torch.Tensor):
        raise ValueError("probability calibration requires observable evidence")
    alive = _branch_alive_masks(evidence)
    losses = []
    diagnostics: dict[str, torch.Tensor] = {}
    ref = labels.new_zeros((), dtype=torch.float32)
    for name in BRANCH_NAMES:
        log_prob = outputs.get(f"calibrated_log_prob_{name}")
        if not isinstance(log_prob, torch.Tensor):
            continue
        per_sample = F.nll_loss(log_prob.float(), labels.long(), reduction="none")
        weight = alive[name].view(-1).float()
        if not bool((weight.sum() > 0).item()):
            continue
        denom = weight.sum().clamp_min(1.0)
        branch_loss = (per_sample * weight).sum() / denom
        losses.append(branch_loss)
        diagnostics[f"probability_calibration_loss_{name}"] = branch_loss.detach()
    total = torch.stack(losses).mean() if losses else ref
    diagnostics["probability_calibration_loss"] = total.detach()
    return total, diagnostics


def compute_posthoc_calibration_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
    *,
    reliability_branches: tuple[str, ...] | list[str] | None = None,
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit I1, conditional routing pi, or fused risk without target mixing."""
    config = config or {}
    reliability_cfg = dict(config.get("reliability_calibration", {}) or {})
    if reliability_branches is not None:
        reliability_cfg["branches"] = list(reliability_branches)
    probability_cfg = config.get("probability_calibration", {}) or {}
    routing_cfg = config.get("routing", {}) or {}

    reliability_weight = float(reliability_cfg.get("weight", 1.0))
    probability_weight = float(probability_cfg.get("weight", 1.0))
    routing_enabled = bool(routing_cfg.get("enabled", False)) and bool(
        routing_cfg.get("posthoc_refine", True)
    )
    routing_weight = float(routing_cfg.get("calibration_weight", 1.0)) if routing_enabled else 0.0
    prediction_weight = float(routing_cfg.get("prediction_loss_weight", 1.0))
    route_oracle_weight = float(routing_cfg.get("route_oracle_loss_weight", 0.0))
    route_oracle_temperature = float(
        routing_cfg.get("route_oracle_temperature", 1.0)
    )
    subset_oracle_weight = float(
        routing_cfg.get("subset_oracle_loss_weight", 0.0)
    )
    subset_oracle_temperature = float(
        routing_cfg.get("subset_oracle_temperature", 1.0)
    )
    risk_weight = float(routing_cfg.get("risk_loss_weight", 1.0))
    risk_loss_type = str(routing_cfg.get("risk_loss", "bce")).strip().lower()
    risk_target_type = str(
        routing_cfg.get("risk_target", "mixture_argmax_error")
    ).strip().lower()
    if risk_loss_type not in {"bce", "brier"}:
        raise ValueError("fusion.routing.risk_loss must be 'bce' or 'brier'")
    if risk_target_type not in ROUTING_RISK_TARGETS:
        raise ValueError(
            "fusion.routing.risk_target must be one of "
            f"{list(ROUTING_RISK_TARGETS)}, got {risk_target_type!r}"
        )
    if risk_weight > 0.0 and risk_target_type == "reliability_deficit_score":
        raise ValueError(
            "fusion.routing.risk_target='reliability_deficit_score' is an "
            "unfitted control score and cannot supervise a learned risk head"
        )
    for name, value in (
        ("reliability_calibration.weight", reliability_weight),
        ("probability_calibration.weight", probability_weight),
        ("routing.calibration_weight", routing_weight),
        ("routing.prediction_loss_weight", prediction_weight),
        ("routing.route_oracle_loss_weight", route_oracle_weight),
        ("routing.subset_oracle_loss_weight", subset_oracle_weight),
        ("routing.risk_loss_weight", risk_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fusion.{name} must be finite and non-negative")
    if not math.isfinite(route_oracle_temperature) or route_oracle_temperature <= 0.0:
        raise ValueError(
            "fusion.routing.route_oracle_temperature must be finite and positive"
        )
    if (
        not math.isfinite(subset_oracle_temperature)
        or subset_oracle_temperature <= 0.0
    ):
        raise ValueError(
            "fusion.routing.subset_oracle_temperature must be finite and positive"
        )
    if subset_oracle_weight == 0.0 and subset_oracle_temperature != 1.0:
        raise ValueError(
            "disabled source-subset oracle requires temperature=1.0"
        )
    if (
        routing_weight > 0.0
        and prediction_weight == 0.0
        and route_oracle_weight == 0.0
        and subset_oracle_weight == 0.0
        and risk_weight == 0.0
    ):
        raise ValueError(
            "post-hoc routing requires prediction_loss_weight, "
            "route_oracle_loss_weight, subset_oracle_loss_weight, or risk_loss_weight"
        )

    zero = labels.new_zeros((), dtype=torch.float32)
    if reliability_weight > 0.0:
        reliability_loss, reliability_diag = compute_reliability_calibration_loss(
            outputs, labels, evidence, reliability_cfg
        )
    else:
        reliability_loss = zero
        reliability_diag = {"reliability_calibration_loss": zero.detach()}
    if probability_weight > 0.0:
        probability_loss, probability_diag = compute_probability_calibration_loss(
            outputs, labels, evidence
        )
    else:
        probability_loss = zero
        probability_diag = {"probability_calibration_loss": zero.detach()}

    routing_active = outputs.get("routing_active")
    active = (
        routing_weight > 0.0
        and isinstance(routing_active, torch.Tensor)
        and (
            not materialize_diagnostics
            or bool((routing_active.detach().float().max() > 0.0).item())
        )
    )
    mixture_log_prob = routing_mixture_log_prob(outputs) if active else None
    has_available = outputs.get("routing_has_available")
    if active and not isinstance(has_available, torch.Tensor):
        raise ValueError("routing calibration requires routing_has_available")
    if isinstance(has_available, torch.Tensor):
        valid_routing = has_available.detach().view(-1).bool()
        if valid_routing.numel() != labels.numel():
            raise ValueError("routing_has_available batch shape disagrees with labels")
    else:
        valid_routing = torch.ones_like(labels, dtype=torch.bool)
    routing_prediction_loss = zero
    if active and prediction_weight > 0.0 and mixture_log_prob is not None:
        prediction_per_sample = F.nll_loss(
            mixture_log_prob, labels.long(), reduction="none"
        )
        prediction_weight_mask = valid_routing.to(
            dtype=prediction_per_sample.dtype
        )
        routing_prediction_loss = (
            prediction_per_sample * prediction_weight_mask
        ).sum() / prediction_weight_mask.sum().clamp_min(1.0)

    routing_route_oracle_loss = zero
    route_oracle_diag: dict[str, torch.Tensor] = {}
    if active and route_oracle_weight > 0.0:
        routing_route_oracle_loss, route_oracle_diag = routing_soft_oracle_loss(
            outputs,
            labels,
            evidence,
            temperature=route_oracle_temperature,
            materialize_diagnostics=materialize_diagnostics,
        )

    routing_subset_oracle_loss = zero
    subset_oracle_diag: dict[str, torch.Tensor] = {}
    if active and subset_oracle_weight > 0.0:
        subset_target, subset_valid, raw_subset_diag = (
            routing_source_subset_oracle_target(
                outputs,
                labels,
                evidence,
                temperature=subset_oracle_temperature,
            )
        )
        subset_per_sample = routing_subset_oracle_per_sample_loss(
            outputs, subset_target
        )
        subset_valid_weight = subset_valid.to(dtype=subset_per_sample.dtype)
        routing_subset_oracle_loss = (
            subset_per_sample * subset_valid_weight
        ).sum() / subset_valid_weight.sum().clamp_min(1.0)
        subset_gaps = raw_subset_diag["best_second_gap"]
        finite_subset_gaps = subset_gaps[torch.isfinite(subset_gaps)]
        eligible_subset_counts = raw_subset_diag["eligible_candidate_count"]
        subset_oracle_diag = {
            "routing_subset_oracle_valid_sample_count": subset_valid.sum().detach(),
            "routing_subset_oracle_min_eligible_candidate_count": (
                eligible_subset_counts.min()
                if eligible_subset_counts.numel()
                else subset_valid.new_zeros((), dtype=torch.long)
            ).detach(),
            "routing_subset_oracle_mean_best_second_gap": (
                finite_subset_gaps.mean()
                if finite_subset_gaps.numel()
                else routing_subset_oracle_loss.detach().new_zeros(())
            ).detach(),
        }

    routing_risk_loss = zero
    routing_risk_error_rate = zero
    routing_risk_training_count = 0
    if active and risk_weight > 0.0 and mixture_log_prob is not None:
        predicted_error = outputs.get("routing_risk_probability")
        if not isinstance(predicted_error, torch.Tensor):
            raise ValueError("routing risk loss requires routing_risk_probability")
        predicted_error = predicted_error.float().view(-1)
        predicted_error_logit = outputs.get(
            "routing_risk_training_logit",
            outputs.get("routing_risk_logit"),
        )
        if risk_loss_type != "brier" and not isinstance(
            predicted_error_logit, torch.Tensor
        ):
            raise ValueError("routing BCE risk loss requires routing_risk_logit")
        if materialize_diagnostics:
            has_valid_routing = bool(valid_routing.any().item())
        else:
            # Post-hoc fitting only invokes this loss on a routed cache.  The
            # masked reductions below remain well-defined even for an all-dead
            # source and avoid synchronizing the device for every packed
            # scenario segment.
            has_valid_routing = True
        if has_valid_routing:
            (
                error_target,
                risk_valid,
                resolved_risk_loss_type,
                resolved_risk_target_type,
            ) = routing_risk_target(
                outputs,
                labels,
                routing_cfg,
                mixture_log_prob=mixture_log_prob,
                valid_routing=valid_routing,
            )
            if (
                resolved_risk_loss_type != risk_loss_type
                or resolved_risk_target_type != risk_target_type
            ):
                raise RuntimeError("routing risk configuration changed during loss evaluation")
            per_sample_risk = routing_risk_per_sample_loss(
                predicted_error,
                (
                    predicted_error_logit
                    if isinstance(predicted_error_logit, torch.Tensor)
                    else None
                ),
                error_target,
                risk_valid,
                loss_type=risk_loss_type,
            )
            if materialize_diagnostics:
                routing_risk_training_count = int(
                    risk_valid.sum().detach().item()
                )
                if not bool(risk_valid.any().item()):
                    routing_risk_loss = predicted_error.sum() * 0.0
                    routing_risk_error_rate = zero
                else:
                    selected_target = error_target[risk_valid]
                    if risk_loss_type != "brier":
                        assert isinstance(predicted_error_logit, torch.Tensor)
                        valid_logit = predicted_error_logit.float().view(-1)[
                            risk_valid
                        ]
                        if not bool(
                            torch.isfinite(valid_logit.detach()).all().item()
                        ):
                            raise FloatingPointError(
                                "routing_risk_training_logit must be finite for "
                                "risk-training samples"
                            )
                    routing_risk_loss = per_sample_risk[risk_valid].mean()
                    routing_risk_error_rate = selected_target.mean()
            else:
                risk_weight_mask = risk_valid.to(dtype=predicted_error.dtype)
                risk_denominator = risk_weight_mask.sum().clamp_min(1.0)
                safe_target = torch.where(
                    risk_valid, error_target, torch.zeros_like(error_target)
                )
                routing_risk_loss = (
                    per_sample_risk * risk_weight_mask
                ).sum() / risk_denominator
                routing_risk_error_rate = (
                    safe_target * risk_weight_mask
                ).sum() / risk_denominator
        else:
            routing_risk_loss = predicted_error.sum() * 0.0

    routing_distribution_weight_sum = (
        prediction_weight + route_oracle_weight + subset_oracle_weight
    )
    routing_distribution_loss = (
        (
            prediction_weight * routing_prediction_loss
            + route_oracle_weight * routing_route_oracle_loss
            + subset_oracle_weight * routing_subset_oracle_loss
        )
        / routing_distribution_weight_sum
        if routing_distribution_weight_sum > 0.0
        else zero
    )
    total = (
        reliability_weight * reliability_loss
        + probability_weight * probability_loss
        + routing_weight
        * (
            routing_distribution_loss + risk_weight * routing_risk_loss
        )
    )
    if not materialize_diagnostics:
        return total, {}
    diagnostics = {
        "calibration_loss": float(total.detach().item()),
        "reliability_calibration_loss": float(reliability_loss.detach().item()),
        "probability_calibration_loss": float(probability_loss.detach().item()),
        "routing_prediction_loss": float(routing_prediction_loss.detach().item()),
        "routing_route_oracle_loss": float(
            routing_route_oracle_loss.detach().item()
        ),
        "routing_subset_oracle_loss": float(
            routing_subset_oracle_loss.detach().item()
        ),
        "routing_risk_loss": float(routing_risk_loss.detach().item()),
        "routing_risk_error_rate": float(routing_risk_error_rate.detach().item()),
        "routing_prediction_loss_weight": prediction_weight,
        "routing_route_oracle_loss_weight": route_oracle_weight,
        "routing_route_oracle_temperature": route_oracle_temperature,
        "routing_subset_oracle_loss_weight": subset_oracle_weight,
        "routing_subset_oracle_temperature": subset_oracle_temperature,
        "routing_distribution_component_weight_sum": (
            routing_distribution_weight_sum
        ),
        "routing_prediction_loss_weight_normalized": (
            prediction_weight / routing_distribution_weight_sum
            if routing_distribution_weight_sum > 0.0
            else 0.0
        ),
        "routing_route_oracle_loss_weight_normalized": (
            route_oracle_weight / routing_distribution_weight_sum
            if routing_distribution_weight_sum > 0.0
            else 0.0
        ),
        "routing_subset_oracle_loss_weight_normalized": (
            subset_oracle_weight / routing_distribution_weight_sum
            if routing_distribution_weight_sum > 0.0
            else 0.0
        ),
        "routing_risk_loss_weight": risk_weight,
        "routing_risk_loss_type": risk_loss_type,
        "routing_risk_target": risk_target_type,
        "routing_risk_training_sample_count": routing_risk_training_count,
        "routing_valid_sample_count": int(valid_routing.sum().detach().item()),
        "routing_posthoc_refine_enabled": float(routing_enabled),
    }
    for source in (
        reliability_diag,
        probability_diag,
        route_oracle_diag,
        subset_oracle_diag,
    ):
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                diagnostics[key] = float(value.detach().item())
    return total, diagnostics


def _weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    per_sample = F.cross_entropy(
        logits,
        labels.long(),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    denom = sample_weight.sum()
    weighted = (per_sample * sample_weight).sum() / denom.clamp_min(1e-8)
    return weighted * (denom > 0).to(dtype=weighted.dtype)


def compute_branch_auxiliary_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute integrity- or availability-weighted branch supervision.

    ``alive_masked_uniform`` is the atomic no-integrity counterpart: every
    available sample has unit weight while dead branches remain excluded.
    ``unmasked_uniform`` is retained as an explicit diagnostic control.
    """
    config = config or {}
    ref = next((outputs.get(key) for key in BRANCH_AUX_KEYS if isinstance(outputs.get(key), torch.Tensor)), None)
    if not isinstance(ref, torch.Tensor):
        raise ValueError("outputs does not contain branch auxiliary logits")
    if not isinstance(evidence, torch.Tensor) or evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError("integrity-weighted auxiliary loss requires observable evidence")

    weight_mode = str(config.get("auxiliary_weight_mode", "integrity")).strip().lower()
    if weight_mode not in AUXILIARY_WEIGHT_MODES:
        raise ValueError(
            "loss.auxiliary_weight_mode must be one of "
            f"{AUXILIARY_WEIGHT_MODES}, got {weight_mode!r}"
        )
    min_weight = float(config.get("min_aux_weight", 0.2))
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("loss.min_aux_weight must be within [0, 1]")
    detach = bool(config.get("detach_reliability_for_aux", True))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    configured_branch_weights = config.get("branch_aux_weights") or {}
    if not isinstance(configured_branch_weights, dict):
        raise ValueError("loss.branch_aux_weights must be a mapping")
    api_integrity = _evidence_column(evidence, EvidenceIndex.API_INTEGRITY, detach)
    graph_integrity = _evidence_column(evidence, EvidenceIndex.GRAPH_INTEGRITY, detach)
    manifest_integrity = _evidence_column(evidence, EvidenceIndex.MANIFEST_INTEGRITY, detach)
    api_alive = _evidence_column(evidence, EvidenceIndex.API_ALIVE, True).bool()
    graph_alive = _evidence_column(evidence, EvidenceIndex.GRAPH_ALIVE, True).bool()
    manifest_alive = _evidence_column(evidence, EvidenceIndex.MANIFEST_ALIVE, True).bool()
    if weight_mode == "integrity":
        branch_weight = {
            "api": api_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * api_integrity),
            "graph": graph_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * graph_integrity),
            "manifest": manifest_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * manifest_integrity),
        }
    elif weight_mode == "alive_masked_uniform":
        branch_weight = {
            "api": api_alive.to(ref.dtype),
            "graph": graph_alive.to(ref.dtype),
            "manifest": manifest_alive.to(ref.dtype),
        }
    else:
        ones = torch.ones_like(api_integrity, dtype=ref.dtype)
        branch_weight = {name: ones for name in BRANCH_NAMES}
    losses = []
    active_weights = []
    active_flags = []
    diagnostics: dict[str, torch.Tensor] = {}
    for key in BRANCH_AUX_KEYS:
        branch = BRANCH_AUX_NAMES[key]
        logits = outputs.get(key)
        configured_weight = float(configured_branch_weights.get(branch, 1.0))
        if not math.isfinite(configured_weight) or configured_weight < 0.0:
            raise ValueError(
                f"loss.branch_aux_weights.{branch} must be finite and non-negative"
            )
        sample_weight = branch_weight[branch]
        diagnostics[f"aux_weight_{branch}"] = sample_weight * configured_weight
        if configured_weight > 0.0 and isinstance(logits, torch.Tensor) and logits.shape == ref.shape:
            active = (sample_weight.sum() > 0).to(dtype=ref.dtype)
            losses.append(
                _weighted_cross_entropy(
                    logits, labels, sample_weight, label_smoothing
                )
                * configured_weight
            )
            active_weights.append(active * configured_weight)
            active_flags.append(active)
    if losses:
        loss_values = torch.stack(losses)
        normalizers = torch.stack(active_weights)
        active = torch.stack(active_flags)
        total = (loss_values * active).sum() / normalizers.sum().clamp_min(1.0e-8)
        active_branch_count = active.sum().detach()
    else:
        total = ref.sum() * 0.0
        active_branch_count = ref.new_zeros(())
    diagnostics["aux_active_branch_count"] = active_branch_count
    diagnostics["aux_uses_integrity_weight"] = ref.new_tensor(
        float(weight_mode == "integrity")
    )
    diagnostics["aux_uses_alive_mask"] = ref.new_tensor(
        float(weight_mode != "unmasked_uniform")
    )
    diagnostics["branch_auxiliary_loss"] = total.detach()
    return total, diagnostics


def compute_robust_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    extra: dict | None = None,
    loss_cfg: dict | None = None,
    *,
    evidence: torch.Tensor | None = None,
    epoch: int = 0,
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Robust objective with independently attributable auxiliary terms."""
    extra = extra or {}
    loss_cfg = loss_cfg or {}
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    branch_aux_weight = float(loss_cfg.get("branch_aux_weight", 0.05))
    reliability_calibration_weight = float(loss_cfg.get("reliability_calibration_weight", 0.0))
    probability_calibration_weight = float(loss_cfg.get("probability_calibration_weight", 0.0))
    evidential_loss_weight = float(loss_cfg.get("evidential_loss_weight", 0.0))
    named_weights = {
        "branch_aux_weight": branch_aux_weight,
        "reliability_calibration_weight": reliability_calibration_weight,
        "probability_calibration_weight": probability_calibration_weight,
        "evidential_loss_weight": evidential_loss_weight,
    }
    for name, value in named_weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    removed_loss_keys = sorted(set(loss_cfg) & REMOVED_LOSS_CONFIG_KEYS)
    if removed_loss_keys:
        raise ValueError(
            "Removed loss configuration keys are unsupported: "
            f"{removed_loss_keys}. Use loss.auxiliary_weight_mode for branch weighting."
        )

    objective = str(loss_cfg.get("objective", "standard")).strip().lower()
    if objective in PAPER_EVIDENTIAL_OBJECTIVES:
        total, parts = _compute_paper_evidential_objective(
            extra,
            labels,
            evidence,
            loss_cfg,
            objective=objective,
            epoch=epoch,
        )
        if materialize_diagnostics:
            parts = {
                key: float(value.item())
                if isinstance(value, torch.Tensor)
                else float(value)
                for key, value in parts.items()
            }
        return total, parts
    if objective not in {"standard", "default", "generic"}:
        raise ValueError(
            "loss.objective must be standard, tmc, or ecml; "
            f"got {objective!r}"
        )

    final_is_log_probability = extra.get("final_is_log_probability", False)
    if not isinstance(final_is_log_probability, bool):
        raise TypeError(
            "extra.final_is_log_probability must be a Python bool; tensor "
            "flags would force a device synchronization in the loss hot path"
        )
    if final_is_log_probability:
        if label_smoothing > 0.0:
            nll = -logits.gather(1, labels.long().view(-1, 1)).view(-1)
            smooth = -logits.mean(dim=-1)
            ce = ((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean()
        else:
            ce = F.nll_loss(logits, labels.long())
    else:
        ce = F.cross_entropy(logits, labels.long(), label_smoothing=label_smoothing)
    branch_loss = logits.new_tensor(0.0)
    branch_weight_sum = 0.0
    branch_weights = loss_cfg.get("branch_aux_weights") or {}
    if not isinstance(branch_weights, dict):
        branch_weights = {}
    aux_diagnostics: dict[str, torch.Tensor] = {}
    auxiliary_weight_mode = resolve_auxiliary_weight_mode(loss_cfg)
    if auxiliary_weight_mode != "unmasked_uniform":
        if not isinstance(evidence, torch.Tensor):
            raise ValueError(
                f"{auxiliary_weight_mode} auxiliary supervision requires observable evidence"
            )
        branch_loss, aux_diagnostics = compute_branch_auxiliary_loss(
            extra,
            labels,
            evidence,
            {**loss_cfg, "auxiliary_weight_mode": auxiliary_weight_mode},
        )
    else:
        for key in BRANCH_AUX_KEYS:
            aux_logits = extra.get(key)
            if isinstance(aux_logits, torch.Tensor) and aux_logits.shape == logits.shape:
                branch_name = BRANCH_AUX_NAMES.get(key, key)
                weight = float(branch_weights.get(branch_name, branch_weights.get(key, 1.0)))
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(
                        f"loss.branch_aux_weights.{branch_name} must be finite and non-negative"
                    )
                if weight <= 0.0:
                    continue
                branch_loss = branch_loss + weight * F.cross_entropy(
                    aux_logits,
                    labels.long(),
                    label_smoothing=label_smoothing,
                )
                branch_weight_sum += weight
        if branch_weight_sum > 0.0:
            branch_loss = branch_loss / branch_loss.new_tensor(branch_weight_sum)

    reliability_calibration = logits.new_tensor(0.0)
    probability_calibration = logits.new_tensor(0.0)
    calibration_diagnostics: dict[str, torch.Tensor] = {}
    if reliability_calibration_weight > 0.0:
        if not isinstance(evidence, torch.Tensor):
            raise ValueError("reliability calibration loss requires observable evidence")
        reliability_calibration, reliability_diag = compute_reliability_calibration_loss(
            extra,
            labels,
            evidence,
            loss_cfg.get("reliability_calibration", {}) or {},
        )
        calibration_diagnostics.update(reliability_diag)
    if probability_calibration_weight > 0.0:
        if not isinstance(evidence, torch.Tensor):
            raise ValueError("probability calibration loss requires observable evidence")
        probability_calibration, probability_diag = compute_probability_calibration_loss(
            extra,
            labels,
            evidence,
        )
        calibration_diagnostics.update(probability_diag)

    # EDL opinion objective: Bayes-risk + annealed KL on each branch's
    # Dirichlet evidence head. The KL coefficient ramps from 0 to 1 over
    # ``evidential.anneal_epochs`` so clean accuracy is not destroyed early.
    evidential_total = logits.new_tensor(0.0)
    evidential_diagnostics: dict[str, torch.Tensor] = {}
    if evidential_loss_weight > 0.0:
        edl_cfg = loss_cfg.get("evidential", {}) or {}
        anneal_epochs = max(int(edl_cfg.get("anneal_epochs", 10)), 1)
        anneal_coef = min(1.0, float(max(epoch, 0)) / anneal_epochs)
        activation = str(edl_cfg.get("evidence_activation", "softplus"))
        edl_branches = edl_cfg.get("branches") or ["api", "graph", "manifest"]
        # Optional class weighting so EDL evidence does not collapse onto the
        # majority class under malware/benign imbalance. "balanced" uses per-batch
        # inverse frequency; a list/dict gives explicit per-class weights.
        class_weight_cfg = edl_cfg.get("class_weight")
        class_weight_per_sample = None
        if class_weight_cfg:
            num_classes = logits.size(-1)
            label_idx = labels.long().view(-1)
            if isinstance(class_weight_cfg, str) and class_weight_cfg.lower() == "balanced":
                counts = torch.bincount(label_idx, minlength=num_classes).to(logits.dtype)
                present = (counts > 0).sum().clamp_min(1)
                per_class = torch.where(
                    counts > 0,
                    label_idx.numel() / (present * counts.clamp_min(1.0)),
                    torch.ones_like(counts),
                )
            elif isinstance(class_weight_cfg, (list, tuple, dict)):
                values = (
                    [float(class_weight_cfg[i]) for i in range(num_classes)]
                    if isinstance(class_weight_cfg, dict)
                    else [float(v) for v in class_weight_cfg]
                )
                if len(values) != num_classes:
                    raise ValueError("loss.evidential.class_weight length must equal num_classes")
                per_class = logits.new_tensor(values)
            else:
                raise ValueError("loss.evidential.class_weight must be 'balanced' or a list/dict")
            class_weight_per_sample = per_class[label_idx]
        alive = (
            _branch_alive_masks(evidence)
            if isinstance(evidence, torch.Tensor) and evidence.ndim == 2
            and evidence.size(-1) >= EvidenceIndex.BASE_DIM
            else None
        )
        edl_losses = []
        edl_active_flags = []
        for key in BRANCH_AUX_KEYS:
            branch_name = BRANCH_AUX_NAMES[key]
            if branch_name not in edl_branches:
                continue
            aux_logits = extra.get(key)
            if not isinstance(aux_logits, torch.Tensor) or aux_logits.shape != logits.shape:
                continue
            weight = alive[branch_name].view(-1) if alive is not None else None
            if class_weight_per_sample is not None:
                weight = (
                    class_weight_per_sample
                    if weight is None
                    else weight * class_weight_per_sample
                )
            branch_edl = evidential_loss(
                aux_logits,
                labels,
                anneal_coef=anneal_coef,
                evidence_activation=activation,
                sample_weight=weight,
            )
            edl_losses.append(branch_edl)
            edl_active_flags.append(
                logits.new_ones(())
                if weight is None
                else (weight.sum() > 0).to(dtype=logits.dtype)
            )
            evidential_diagnostics[f"evidential_loss_{branch_name}"] = branch_edl.detach()
        if edl_losses:
            edl_values = torch.stack(edl_losses)
            edl_active = torch.stack(edl_active_flags)
            evidential_total = (
                (edl_values * edl_active).sum() / edl_active.sum().clamp_min(1.0)
            )
        evidential_diagnostics["evidential_anneal_coef"] = logits.new_tensor(anneal_coef)
        evidential_diagnostics["evidential_loss"] = evidential_total.detach()

    total = (
        ce
        + branch_aux_weight * branch_loss
        + reliability_calibration_weight * reliability_calibration
        + probability_calibration_weight * probability_calibration
        + evidential_loss_weight * evidential_total
    )
    parts: dict[str, float | torch.Tensor] = {
        "loss": total.detach(),
        "ce": ce.detach(),
        "branch_aux": branch_loss.detach(),
        "branch_aux_weight": branch_aux_weight,
        "reliability_calibration": reliability_calibration.detach(),
        "reliability_calibration_weight": reliability_calibration_weight,
        "probability_calibration": probability_calibration.detach(),
        "probability_calibration_weight": probability_calibration_weight,
        "evidential_loss": evidential_total.detach(),
        "evidential_loss_weight": evidential_loss_weight,
    }
    for source in (aux_diagnostics, calibration_diagnostics, evidential_diagnostics):
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                parts[key] = value.detach()
    if materialize_diagnostics:
        parts = {
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in parts.items()
        }
    return total, parts
