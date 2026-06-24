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
    losses = []
    diagnostics: dict[str, torch.Tensor] = {}
    ref = labels.new_zeros((), dtype=torch.float32)
    for name in BRANCH_NAMES:
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
        branch_loss = (per_sample * weight).sum() / denom
        losses.append(branch_loss)
        diagnostics[f"reliability_loss_{name}"] = branch_loss.detach()
        diagnostics[f"mean_predicted_reliability_{name}"] = (
            (reliability.detach() * weight).sum() / denom
        )
        diagnostics[f"branch_accuracy_{name}"] = (correctness * weight).sum() / denom
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Objective used after model selection to fit calibration-only parameters."""
    config = config or {}
    reliability_cfg = config.get("reliability_calibration", {}) or {}
    probability_cfg = config.get("probability_calibration", {}) or {}
    reliability_weight = float(reliability_cfg.get("weight", 1.0))
    probability_weight = float(probability_cfg.get("weight", 1.0))
    for name, value in (
        ("reliability_calibration.weight", reliability_weight),
        ("probability_calibration.weight", probability_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fusion.{name} must be finite and non-negative")
    reliability_loss, reliability_diag = compute_reliability_calibration_loss(
        outputs, labels, evidence, reliability_cfg
    )
    probability_loss, probability_diag = compute_probability_calibration_loss(
        outputs, labels, evidence
    )
    total = reliability_weight * reliability_loss + probability_weight * probability_loss
    diagnostics = {
        "calibration_loss": float(total.detach().item()),
        "reliability_calibration_loss": float(reliability_loss.detach().item()),
        "probability_calibration_loss": float(probability_loss.detach().item()),
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


def compute_reliability_weighted_aux_loss(
    outputs: dict,
    labels: torch.Tensor,
    evidence: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = config or {}
    ref = next((outputs.get(key) for key in BRANCH_AUX_KEYS if isinstance(outputs.get(key), torch.Tensor)), None)
    if not isinstance(ref, torch.Tensor):
        raise ValueError("outputs does not contain branch auxiliary logits")
    if not isinstance(evidence, torch.Tensor) or evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError("reliability-weighted auxiliary loss requires observable evidence")

    min_weight = float(config.get("min_aux_weight", 0.2))
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("loss.min_aux_weight must be within [0, 1]")
    detach = bool(config.get("detach_reliability_for_aux", True))
    label_smoothing = float(config.get("label_smoothing", 0.0))
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
        weight = branch_weight[branch]
        diagnostics[f"aux_weight_{branch}"] = weight
        if isinstance(logits, torch.Tensor) and logits.shape == ref.shape:
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
    diagnostics["reliability_weighted_aux_loss"] = total.detach()
    return total, diagnostics


def compute_robust_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    extra: dict | None = None,
    loss_cfg: dict | None = None,
    *,
    evidence: torch.Tensor | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
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
    if bool(loss_cfg.get("reliability_weighted_aux", False)):
        if not isinstance(evidence, torch.Tensor):
            raise ValueError("loss.reliability_weighted_aux=true requires observable evidence")
        branch_loss, aux_diagnostics = compute_reliability_weighted_aux_loss(
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

    # I1 evidential (EDL) objective: Bayes-risk + annealed KL on each branch's
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
            if weight is not None and not bool((weight.sum() > 0).item()):
                continue
            branch_edl = evidential_loss(
                aux_logits,
                labels,
                anneal_coef=anneal_coef,
                evidence_activation=activation,
                sample_weight=weight,
            )
            edl_losses.append(branch_edl)
            evidential_diagnostics[f"evidential_loss_{branch_name}"] = branch_edl.detach()
        if edl_losses:
            evidential_total = torch.stack(edl_losses).mean()
        evidential_diagnostics["evidential_anneal_coef"] = logits.new_tensor(anneal_coef)
        evidential_diagnostics["evidential_loss"] = evidential_total.detach()

    total = (
        ce
        + branch_aux_weight * branch_loss
        + reliability_calibration_weight * reliability_calibration
        + probability_calibration_weight * probability_calibration
        + evidential_loss_weight * evidential_total
    )
    parts = {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "branch_aux": float(branch_loss.detach().item()),
        "branch_aux_weight": branch_aux_weight,
        "reliability_calibration": float(reliability_calibration.detach().item()),
        "reliability_calibration_weight": reliability_calibration_weight,
        "probability_calibration": float(probability_calibration.detach().item()),
        "probability_calibration_weight": probability_calibration_weight,
        "evidential_loss": float(evidential_total.detach().item()),
        "evidential_loss_weight": evidential_loss_weight,
    }
    for source in (aux_diagnostics, calibration_diagnostics, evidential_diagnostics):
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                parts[key] = float(value.detach().item())
    return total, parts
