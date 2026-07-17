from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.evidential import evidential_loss


BRANCH_AUX_KEYS = (
    "api_logits_aux",
    "graph_logits_aux",
    "manifest_logits_aux",
    "joint_logits_aux",
)

BRANCH_AUX_NAMES = {
    "api_logits_aux": "api",
    "graph_logits_aux": "graph",
    "manifest_logits_aux": "manifest",
    "joint_logits_aux": "joint",
}
BRANCH_NAMES = ("api", "graph", "manifest", "joint")


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
        "joint": torch.maximum(torch.maximum(api, graph), manifest),
    }


def build_routing_calibration_target(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
) -> torch.Tensor:
    """Allocate routing supervision between correct branches and unknown.

    Each available correct branch receives ``1 / num_available`` mass. The
    remaining mass is assigned to unknown, so the unknown target equals the
    observed fraction of available branches that predicted incorrectly.
    """
    alive = _branch_alive_masks(evidence)
    correct_available = []
    available = []
    for name in ("api", "graph", "manifest"):
        branch_logits = outputs.get(f"{name}_logits_aux")
        if not isinstance(branch_logits, torch.Tensor):
            raise ValueError(
                "routing calibration requires API, Graph, and Manifest logits"
            )
        branch_alive = alive[name].view(-1).float()
        correct = branch_logits.detach().argmax(dim=-1).eq(labels.long()).float()
        correct_available.append(correct * branch_alive)
        available.append(branch_alive)

    correct_stack = torch.stack(correct_available, dim=-1)
    available_stack = torch.stack(available, dim=-1)
    available_count = available_stack.sum(dim=-1, keepdim=True)
    safe_available_count = available_count.clamp_min(1.0)
    branch_target = correct_stack / safe_available_count
    correct_fraction = correct_stack.sum(dim=-1, keepdim=True) / safe_available_count
    unknown_target = 1.0 - correct_fraction
    no_available = available_count <= 0.0
    branch_target = torch.where(
        no_available.expand_as(branch_target),
        torch.zeros_like(branch_target),
        branch_target,
    )
    unknown_target = torch.where(
        no_available,
        torch.ones_like(unknown_target),
        unknown_target,
    )
    return torch.cat([branch_target, unknown_target], dim=-1)


def compute_reliability_calibration_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
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
    group_mean_alignment = bool(config.get("group_mean_alignment", False))
    losses = []
    diagnostics: dict[str, torch.Tensor] = {}
    ref = labels.new_zeros((), dtype=torch.float32)
    for name in branch_names:
        reliability = outputs.get(f"predicted_reliability_{name}")
        logits = outputs.get(f"{name}_logits_aux")
        if not isinstance(reliability, torch.Tensor) or not isinstance(logits, torch.Tensor):
            continue
        reliability = reliability.view(-1).float().clamp(1.0e-6, 1.0 - 1.0e-6)
        correctness = logits.detach().argmax(dim=-1).eq(labels.long()).float()
        weight = alive[name].view(-1).float()
        if not bool((weight.sum() > 0).item()):
            continue
        if loss_type == "brier":
            per_sample = (reliability - correctness).square()
        else:
            per_sample = -(
                correctness * reliability.log()
                + (1.0 - correctness) * torch.log1p(-reliability)
            )
        denom = weight.sum().clamp_min(1.0)
        proper_loss = (per_sample * weight).sum() / denom
        mean_reliability = (reliability * weight).sum() / denom
        branch_accuracy = (correctness * weight).sum() / denom
        mean_alignment_loss = (mean_reliability - branch_accuracy).square()
        branch_loss = (
            proper_loss + mean_alignment_loss
            if group_mean_alignment
            else proper_loss
        )
        losses.append(branch_loss)
        diagnostics[f"reliability_loss_{name}"] = branch_loss.detach()
        diagnostics[f"reliability_proper_loss_{name}"] = proper_loss.detach()
        diagnostics[f"reliability_mean_alignment_loss_{name}"] = (
            mean_alignment_loss.detach()
        )
        diagnostics[f"mean_predicted_reliability_{name}"] = mean_reliability.detach()
        diagnostics[f"branch_accuracy_{name}"] = branch_accuracy.detach()
    total = torch.stack(losses).mean() if losses else ref
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Objective used after model selection to fit calibration-only parameters."""
    config = config or {}
    reliability_cfg = dict(config.get("reliability_calibration", {}) or {})
    if reliability_branches is not None:
        reliability_cfg["branches"] = list(reliability_branches)
    probability_cfg = config.get("probability_calibration", {}) or {}
    routing_cfg = config.get("routing", {}) or {}
    routing_posthoc_refine = bool(routing_cfg.get("posthoc_refine", True))
    reliability_weight = float(reliability_cfg.get("weight", 1.0))
    probability_weight = float(probability_cfg.get("weight", 1.0))
    routing_weight = (
        float(routing_cfg.get("calibration_weight", 1.0))
        if bool(routing_cfg.get("enabled", False)) and routing_posthoc_refine
        else 0.0
    )
    routing_target_weight = float(routing_cfg.get("target_loss_weight", 1.0))
    routing_prediction_weight = float(
        routing_cfg.get(
            "prediction_loss_weight",
            1.0 if bool(routing_cfg.get("use_fused_prediction_loss", False)) else 0.0,
        )
    )
    for name, value in (
        ("reliability_calibration.weight", reliability_weight),
        ("probability_calibration.weight", probability_weight),
        ("routing.calibration_weight", routing_weight),
        ("routing.target_loss_weight", routing_target_weight),
        ("routing.prediction_loss_weight", routing_prediction_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fusion.{name} must be finite and non-negative")
    if (
        routing_weight > 0.0
        and routing_target_weight == 0.0
        and routing_prediction_weight == 0.0
    ):
        raise ValueError(
            "post-hoc routing requires a positive target_loss_weight or "
            "prediction_loss_weight"
        )
    if (
        routing_weight > 0.0
        and str(routing_cfg.get("mode", "learned")).lower() == "known_only"
        and routing_target_weight > 0.0
    ):
        raise ValueError(
            "routing.mode=known_only cannot fit targets that reserve mass for "
            "the unknown outcome; set target_loss_weight=0 and use a positive "
            "prediction_loss_weight"
        )
    zero = labels.new_zeros((), dtype=torch.float32)
    if reliability_weight > 0.0:
        reliability_loss, reliability_diag = compute_reliability_calibration_loss(
            outputs, labels, evidence, reliability_cfg
        )
    else:
        reliability_loss = zero
        reliability_diag = {
            "reliability_calibration_loss": zero.detach(),
        }
    if probability_weight > 0.0:
        probability_loss, probability_diag = compute_probability_calibration_loss(
            outputs, labels, evidence
        )
    else:
        probability_loss = zero
        probability_diag = {
            "probability_calibration_loss": zero.detach(),
        }
    routing_active = outputs.get("routing_active")
    routing_weights = outputs.get("routing_weights")
    if (
        routing_weight > 0.0
        and routing_target_weight > 0.0
        and isinstance(routing_active, torch.Tensor)
        and bool((routing_active.detach().float().max() > 0.0).item())
        and isinstance(routing_weights, torch.Tensor)
    ):
        routing_target = build_routing_calibration_target(outputs, labels, evidence)
        routing_loss = -(
            routing_target
            * routing_weights.float().clamp_min(1.0e-6).log()
        ).sum(dim=-1).mean()
    else:
        routing_loss = zero
    routing_prediction_loss = zero
    final_log_prob = outputs.get("final_logits")
    if (
        routing_weight > 0.0
        and routing_prediction_weight > 0.0
        and isinstance(routing_active, torch.Tensor)
        and bool((routing_active.detach().float().max() > 0.0).item())
        and isinstance(final_log_prob, torch.Tensor)
    ):
        routing_prediction_loss = F.nll_loss(
            final_log_prob.float(), labels.long()
        )
    total = (
        reliability_weight * reliability_loss
        + probability_weight * probability_loss
        + routing_weight
        * (
            routing_target_weight * routing_loss
            + routing_prediction_weight * routing_prediction_loss
        )
    )
    diagnostics = {
        "calibration_loss": float(total.detach().item()),
        "reliability_calibration_loss": float(reliability_loss.detach().item()),
        "probability_calibration_loss": float(probability_loss.detach().item()),
        "routing_calibration_loss": float(routing_loss.detach().item()),
        "routing_prediction_loss": float(
            routing_prediction_loss.detach().item()
        ),
        "routing_target_loss_weight": routing_target_weight,
        "routing_prediction_loss_weight": routing_prediction_weight,
        "routing_posthoc_refine_enabled": float(routing_posthoc_refine),
    }
    for source in (reliability_diag, probability_diag):
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


def compute_integrity_weighted_aux_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weight branch auxiliary supervision by observable input integrity."""
    config = config or {}
    ref = next((outputs.get(key) for key in BRANCH_AUX_KEYS if isinstance(outputs.get(key), torch.Tensor)), None)
    if not isinstance(ref, torch.Tensor):
        raise ValueError("outputs does not contain branch auxiliary logits")
    if not isinstance(evidence, torch.Tensor) or evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError("integrity-weighted auxiliary loss requires observable evidence")

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
    joint_alive = api_alive | graph_alive | manifest_alive
    alive_matrix = torch.stack([api_alive, graph_alive, manifest_alive], dim=-1).to(ref.dtype)
    integrity_matrix = torch.stack([api_integrity, graph_integrity, manifest_integrity], dim=-1)
    mean_alive_integrity = (integrity_matrix * alive_matrix).sum(dim=-1) / alive_matrix.sum(dim=-1).clamp_min(1.0)

    branch_weight = {
        "api": api_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * api_integrity),
        "graph": graph_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * graph_integrity),
        "manifest": manifest_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * manifest_integrity),
        "joint": joint_alive.to(ref.dtype) * (min_weight + (1.0 - min_weight) * mean_alive_integrity),
    }
    losses = []
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
        weight = branch_weight[branch] * configured_weight
        diagnostics[f"aux_weight_{branch}"] = weight
        if configured_weight > 0.0 and isinstance(logits, torch.Tensor) and logits.shape == ref.shape:
            losses.append(_weighted_cross_entropy(logits, labels, weight, label_smoothing))
            active_flags.append((weight.sum() > 0).to(dtype=ref.dtype))
    if losses:
        loss_values = torch.stack(losses)
        active = torch.stack(active_flags)
        total = (loss_values * active).sum() / active.sum().clamp_min(1.0)
        active_branch_count = active.sum().detach()
    else:
        total = ref.sum() * 0.0
        active_branch_count = ref.new_zeros(())
    diagnostics["aux_active_branch_count"] = active_branch_count
    diagnostics["integrity_weighted_aux_loss"] = total.detach()
    # Backward-compatible diagnostic consumed by existing result collectors.
    diagnostics["reliability_weighted_aux_loss"] = total.detach()
    return total, diagnostics


def compute_reliability_weighted_aux_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backward-compatible alias for :func:`compute_integrity_weighted_aux_loss`."""
    return compute_integrity_weighted_aux_loss(outputs, labels, evidence, config)


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
    legacy_weights = {
        name: float(loss_cfg.get(name, 0.0))
        for name in (
            "semantic_reconstruction_weight",
            "cross_source_consistency_weight",
            "gate_prior_weight",
        )
    }
    nonzero_legacy = {name: value for name, value in legacy_weights.items() if value != 0.0}
    if nonzero_legacy:
        raise ValueError(
            "Semantic reconstruction/cross-source/gate-prior losses were removed from "
            f"the formal lean pipeline; non-zero legacy weights are unsupported: {nonzero_legacy}"
        )

    final_is_log_probability = extra.get("final_is_log_probability", False)
    if isinstance(final_is_log_probability, torch.Tensor):
        final_is_log_probability = bool(final_is_log_probability.detach().cpu().item())
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
    legacy_integrity_aux = loss_cfg.get("reliability_weighted_aux")
    use_integrity_weighted_aux = (
        bool(legacy_integrity_aux)
        if legacy_integrity_aux is not None
        else bool(loss_cfg.get("integrity_weighted_aux", False))
    )
    if use_integrity_weighted_aux:
        if not isinstance(evidence, torch.Tensor):
            raise ValueError(
                "integrity-weighted auxiliary supervision requires observable evidence"
            )
        branch_loss, aux_diagnostics = compute_integrity_weighted_aux_loss(
            extra, labels, evidence, loss_cfg
        )
    else:
        for key in BRANCH_AUX_KEYS:
            aux_logits = extra.get(key)
            if isinstance(aux_logits, torch.Tensor) and aux_logits.shape == logits.shape:
                branch_name = BRANCH_AUX_NAMES.get(key, key)
                weight = float(branch_weights.get(branch_name, branch_weights.get(key, 1.0)))
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
