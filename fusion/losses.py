from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from fusion.constants import AvailabilityIndex
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
AUXILIARY_WEIGHT_MODES = (
    "alive_masked_uniform",
    "unmasked_uniform",
)
PAPER_EVIDENTIAL_OBJECTIVES = ("tmc", "ecml")
LOSS_CONFIG_KEYS = frozenset(
    {
        "objective",
        "label_smoothing",
        "branch_aux_weight",
        "branch_aux_weights",
        "auxiliary_weight_mode",
        "evidential_loss_weight",
        "evidential",
        "tmc",
        "ecml",
    }
)


def _validate_loss_config_keys(loss_cfg: dict) -> None:
    unknown = sorted(set(loss_cfg) - LOSS_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported loss configuration keys: {unknown}")


def _availability_column(
    availability: torch.Tensor, index: int, detach: bool = False
) -> torch.Tensor:
    value = availability[:, index].clamp(0.0, 1.0)
    return value.detach() if detach else value


def _validate_availability(
    availability: torch.Tensor,
    *,
    context: str,
) -> None:
    if (
        not isinstance(availability, torch.Tensor)
        or availability.ndim != 2
        or availability.size(-1) != AvailabilityIndex.BASE_DIM
    ):
        raise ValueError(
            f"{context} requires exact fusion availability "
            f"[B, {AvailabilityIndex.BASE_DIM}]"
        )
    valid = (
        torch.isfinite(availability)
        & ((availability == 0.0) | (availability == 1.0))
    ).all()
    message = f"{context} requires finite binary availability values (0 or 1)"
    if availability.device.type == "cpu":
        if not bool(valid.item()):
            raise ValueError(message)
    else:
        torch._assert_async(valid, message)


def _branch_alive_masks(
    availability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_availability(availability, context="branch availability mask")
    api = _availability_column(
        availability, AvailabilityIndex.API_ALIVE, True
    )
    graph = _availability_column(
        availability, AvailabilityIndex.GRAPH_ALIVE, True
    )
    manifest = _availability_column(
        availability, AvailabilityIndex.MANIFEST_ALIVE, True
    )
    return {
        "api": api,
        "graph": graph,
        "manifest": manifest,
    }


def resolve_auxiliary_weight_mode(loss_cfg: dict | None) -> str:
    """Resolve the explicit branch-auxiliary weighting mechanism."""

    loss_cfg = loss_cfg or {}
    _validate_loss_config_keys(loss_cfg)
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
    availability: torch.Tensor | None,
    loss_cfg: dict,
    *,
    objective: str,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Compute the shared-encoder TMC/ECML-style adapted training loss."""
    if objective not in PAPER_EVIDENTIAL_OBJECTIVES:
        raise ValueError(f"unsupported paper evidential objective: {objective}")
    if not isinstance(availability, torch.Tensor):
        raise ValueError(
            f"loss.objective={objective} requires fusion availability masks"
        )

    incompatible = {
        "branch_aux_weight": float(loss_cfg.get("branch_aux_weight", 0.0)),
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
    alive = _branch_alive_masks(availability)
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
    availability: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute availability-masked or unmasked branch supervision."""
    config = config or {}
    ref = next((outputs.get(key) for key in BRANCH_AUX_KEYS if isinstance(outputs.get(key), torch.Tensor)), None)
    if not isinstance(ref, torch.Tensor):
        raise ValueError("outputs does not contain branch auxiliary logits")
    _validate_availability(
        availability, context="availability-masked auxiliary loss"
    )

    weight_mode = str(
        config.get("auxiliary_weight_mode", "alive_masked_uniform")
    ).strip().lower()
    if weight_mode not in AUXILIARY_WEIGHT_MODES:
        raise ValueError(
            "loss.auxiliary_weight_mode must be one of "
            f"{AUXILIARY_WEIGHT_MODES}, got {weight_mode!r}"
        )
    label_smoothing = float(config.get("label_smoothing", 0.0))
    configured_branch_weights = config.get("branch_aux_weights") or {}
    if not isinstance(configured_branch_weights, dict):
        raise ValueError("loss.branch_aux_weights must be a mapping")
    api_alive = _availability_column(
        availability, AvailabilityIndex.API_ALIVE, True
    ).bool()
    graph_alive = _availability_column(
        availability, AvailabilityIndex.GRAPH_ALIVE, True
    ).bool()
    manifest_alive = _availability_column(
        availability, AvailabilityIndex.MANIFEST_ALIVE, True
    ).bool()
    if weight_mode == "alive_masked_uniform":
        branch_weight = {
            "api": api_alive.to(ref.dtype),
            "graph": graph_alive.to(ref.dtype),
            "manifest": manifest_alive.to(ref.dtype),
        }
    else:
        ones = torch.ones_like(api_alive, dtype=ref.dtype)
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
    availability: torch.Tensor | None = None,
    epoch: int = 0,
    evidence_activation: str = "softplus",
    materialize_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Robust objective with independently attributable auxiliary terms."""
    extra = extra or {}
    loss_cfg = loss_cfg or {}
    _validate_loss_config_keys(loss_cfg)
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    # Keep the standalone loss API identical to the registered Stage-1
    # protocol; callers that bypass config resolution must not silently train
    # a different objective.
    branch_aux_weight = float(loss_cfg.get("branch_aux_weight", 0.25))
    evidential_loss_weight = float(loss_cfg.get("evidential_loss_weight", 0.05))
    named_weights = {
        "branch_aux_weight": branch_aux_weight,
        "evidential_loss_weight": evidential_loss_weight,
    }
    for name, value in named_weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    objective = str(loss_cfg.get("objective", "standard")).strip().lower()
    if objective in PAPER_EVIDENTIAL_OBJECTIVES:
        total, parts = _compute_paper_evidential_objective(
            extra,
            labels,
            availability,
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
        if not isinstance(availability, torch.Tensor):
            raise ValueError(
                f"{auxiliary_weight_mode} auxiliary supervision requires "
                "fusion availability"
            )
        branch_loss, aux_diagnostics = compute_branch_auxiliary_loss(
            extra,
            labels,
            availability,
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

    # EDL opinion objective: Bayes-risk + annealed KL on each branch's
    # Dirichlet evidence head. The KL coefficient ramps from 0 to 1 over
    # ``evidential.anneal_epochs`` so clean accuracy is not destroyed early.
    evidential_total = logits.new_tensor(0.0)
    evidential_diagnostics: dict[str, torch.Tensor] = {}
    if evidential_loss_weight > 0.0:
        edl_cfg = loss_cfg.get("evidential", {}) or {}
        if "evidence_activation" in edl_cfg:
            raise ValueError(
                "loss.evidential.evidence_activation was removed; configure "
                "the single shared fusion.evidence_activation instead"
            )
        anneal_epochs = max(int(edl_cfg.get("anneal_epochs", 10)), 1)
        anneal_coef = min(1.0, float(max(epoch, 0)) / anneal_epochs)
        activation = str(evidence_activation).strip().lower()
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
            _branch_alive_masks(availability)
            if isinstance(availability, torch.Tensor)
            and availability.ndim == 2
            and availability.size(-1) == AvailabilityIndex.BASE_DIM
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
        + evidential_loss_weight * evidential_total
    )
    parts: dict[str, float | torch.Tensor] = {
        "loss": total.detach(),
        "ce": ce.detach(),
        "branch_aux": branch_loss.detach(),
        "branch_aux_weight": branch_aux_weight,
        "evidential_loss": evidential_total.detach(),
        "evidential_loss_weight": evidential_loss_weight,
    }
    for source in (aux_diagnostics, evidential_diagnostics):
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                parts[key] = value.detach()
    if materialize_diagnostics:
        parts = {
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in parts.items()
        }
    return total, parts
