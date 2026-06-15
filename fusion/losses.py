from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.gates import heuristic_reliability_gate


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


def _batch_semantic_target(
    batch,
    name: str,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    value = getattr(batch, name, None)
    if not isinstance(value, torch.Tensor):
        return None
    value = value.to(device=ref.device, dtype=ref.dtype)
    value = value.view(1, -1).expand(ref.size(0), -1) if value.ndim == 1 else value.view(ref.size(0), -1)
    if value.size(1) != ref.size(1):
        return None
    return (value > 0).to(dtype=ref.dtype)


def _weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=-1)
    denom = sample_weight.sum()
    return (per_sample * sample_weight).sum() / denom.clamp_min(1e-8) if bool((denom > 0).item()) else logits.sum() * 0.0


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
            per_sample = F.binary_cross_entropy(reliability, correctness, reduction="none")
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


def compute_masked_semantic_reconstruction_loss(
    outputs: dict,
    batch,
    evidence: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct masked 12-D semantic targets from other available modalities."""
    config = config or {}
    ref = outputs.get("recon_api_semantic_logits")
    if not isinstance(ref, torch.Tensor):
        raise ValueError("outputs is missing recon_api_semantic_logits")
    zero = ref.sum() * 0.0
    if not bool(config.get("enabled", False)):
        return zero, {"loss_recon_total": zero}
    if str(config.get("loss", "bce")).lower() != "bce":
        raise ValueError("semantic_reconstruction.loss currently supports only 'bce'")
    if not isinstance(evidence, torch.Tensor) or evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError("masked semantic reconstruction requires observable evidence")

    detach = bool(config.get("detach_reliability", True))
    integrity = {
        "api": _evidence_column(evidence, EvidenceIndex.API_INTEGRITY, detach),
        "graph": _evidence_column(evidence, EvidenceIndex.GRAPH_INTEGRITY, detach),
        "manifest": _evidence_column(evidence, EvidenceIndex.MANIFEST_INTEGRITY, detach),
    }
    use_calibrated_reliability = bool(config.get("use_calibrated_reliability", False))
    if use_calibrated_reliability and not all(
        isinstance(outputs.get(f"predicted_reliability_{name}"), torch.Tensor)
        for name in ("api", "graph", "manifest")
    ):
        raise ValueError(
            "semantic_reconstruction.use_calibrated_reliability=true requires an "
            "already-fitted and active reliability calibrator"
        )
    source_reliability = {}
    for name in ("api", "graph", "manifest"):
        calibrated = outputs.get(f"predicted_reliability_{name}")
        if use_calibrated_reliability and isinstance(calibrated, torch.Tensor):
            calibrated = calibrated.view(-1).to(device=ref.device, dtype=ref.dtype)
            source_reliability[name] = calibrated.detach() if detach else calibrated
        else:
            source_reliability[name] = integrity[name]
    alive = {
        "api": _evidence_column(evidence, EvidenceIndex.API_ALIVE, True).bool(),
        "graph": _evidence_column(evidence, EvidenceIndex.GRAPH_ALIVE, True).bool(),
        "manifest": _evidence_column(evidence, EvidenceIndex.MANIFEST_ALIVE, True).bool(),
    }
    masks = {
        name: outputs.get(f"mask_{name}_semantic", ref.new_zeros(ref.size(0))).view(-1).bool()
        for name in ("api", "graph", "manifest")
    }
    min_target = float(config.get("min_target_integrity", 0.2))
    min_input = float(config.get("min_input_integrity", 0.2))
    for name, value in (("min_target_integrity", min_target), ("min_input_integrity", min_input)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"semantic_reconstruction.{name} must be within [0, 1]")

    target_names = {
        "api": "api_semantic_category_counts",
        "graph": "graph_semantic_category_counts",
        "manifest": "manifest_category_counts",
    }
    source_names = {
        "api": ("graph", "manifest"),
        "graph": ("api", "manifest"),
        "manifest": ("api", "graph"),
    }
    losses: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    for target_name in ("api", "graph", "manifest"):
        logits = outputs.get(f"recon_{target_name}_semantic_logits")
        if not isinstance(logits, torch.Tensor):
            losses[target_name] = zero
            diagnostics[f"valid_recon_{target_name}_rate"] = zero.detach()
            continue
        target = _batch_semantic_target(batch, target_names[target_name], logits)
        if target is None:
            losses[target_name] = logits.sum() * 0.0
            diagnostics[f"valid_recon_{target_name}_rate"] = zero.detach()
            continue

        source_a, source_b = source_names[target_name]
        source_a_available = alive[source_a] & ~masks[source_a]
        source_b_available = alive[source_b] & ~masks[source_b]
        source_a_weight = source_reliability[source_a] * source_a_available.to(ref.dtype)
        source_b_weight = source_reliability[source_b] * source_b_available.to(ref.dtype)
        input_weight = torch.maximum(source_a_weight, source_b_weight)
        valid = (
            masks[target_name]
            & alive[target_name]
            & (integrity[target_name] >= min_target)
            & (input_weight >= min_input)
        )
        sample_weight = valid.to(ref.dtype) * integrity[target_name] * input_weight
        losses[target_name] = _weighted_bce(logits, target, sample_weight)
        diagnostics[f"valid_recon_{target_name}_rate"] = valid.float().mean().detach()
        diagnostics[f"mask_rate_{target_name}"] = masks[target_name].float().mean().detach()

    total = losses["api"] + losses["graph"] + losses["manifest"]
    diagnostics.update(
        {
            "loss_recon_api": losses["api"].detach(),
            "loss_recon_graph": losses["graph"].detach(),
            "loss_recon_manifest": losses["manifest"].detach(),
            "loss_recon_total": total.detach(),
        }
    )
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
    return (per_sample * sample_weight).sum() / denom.clamp_min(1e-8) if bool((denom > 0).item()) else logits.sum() * 0.0


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
    diagnostics: dict[str, torch.Tensor] = {}
    for key in BRANCH_AUX_KEYS:
        branch = BRANCH_AUX_NAMES[key]
        logits = outputs.get(key)
        weight = branch_weight[branch]
        diagnostics[f"aux_weight_{branch}"] = weight
        if isinstance(logits, torch.Tensor) and logits.shape == ref.shape:
            losses.append(_weighted_cross_entropy(logits, labels, weight, label_smoothing))
    total = torch.stack(losses).sum() if losses else ref.sum() * 0.0
    diagnostics["reliability_weighted_aux_loss"] = total.detach()
    return total, diagnostics


def _matrix(extra: dict, key: str, ref: torch.Tensor) -> torch.Tensor | None:
    value = extra.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.ndim == 1:
        out = out.view(1, -1).expand(ref.size(0), -1)
    else:
        out = out.view(ref.size(0), -1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _reliability(extra: dict, key: str, ref: torch.Tensor, default: float) -> torch.Tensor:
    value = extra.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.full((ref.size(0),), float(default), device=ref.device, dtype=ref.dtype)
    out = value.to(device=ref.device, dtype=ref.dtype).view(ref.size(0), -1)[:, 0]
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _weighted_cosine_direction_loss(
    pred_logits: torch.Tensor | None,
    target_counts: torch.Tensor | None,
    weights: torch.Tensor,
    active_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_logits is None or target_counts is None:
        zero = weights.new_tensor(0.0)
        return zero, zero
    pred = F.softplus(pred_logits.float())
    target = target_counts.float().clamp_min(0.0)
    if pred.size(1) != target.size(1):
        zero = weights.new_tensor(0.0)
        return zero, zero
    if active_only:
        active = target > 0
        pred = pred * active.to(dtype=pred.dtype)
        target = target * active.to(dtype=target.dtype)
    valid = (target.abs().sum(dim=-1) > 0) & (weights > 0)
    if not valid.any():
        zero = weights.new_tensor(0.0)
        return zero, zero
    sim = F.cosine_similarity(pred[valid], target[valid], dim=-1).clamp(-1.0, 1.0)
    loss = 1.0 - sim
    w = weights[valid].float()
    return (loss * w).sum() / w.sum().clamp_min(1e-8), w.sum()


def _pair_weight(
    base_weight: torch.Tensor,
    consistency: torch.Tensor | None,
    min_reliability: float,
    min_consistency: float,
) -> torch.Tensor:
    weight = base_weight
    if min_reliability > 0.0:
        weight = torch.where(weight >= min_reliability, weight, torch.zeros_like(weight))
    if consistency is not None and min_consistency > 0.0:
        weight = torch.where(consistency >= min_consistency, weight, torch.zeros_like(weight))
    return weight


def _semantic_losses(
    extra: dict,
    ref_logits: torch.Tensor,
    loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_cfg = loss_cfg or {}
    api_pred = _matrix(extra, "api_semantic_logits", ref_logits)
    graph_pred = _matrix(extra, "graph_semantic_logits", ref_logits)
    manifest_pred = _matrix(extra, "manifest_semantic_logits", ref_logits)
    api_counts = _matrix(extra, "api_semantic_category_counts", ref_logits)
    graph_counts = _matrix(extra, "graph_semantic_category_counts", ref_logits)
    manifest_counts = _matrix(extra, "manifest_category_counts", ref_logits)

    r_api = _reliability(extra, "r_api", ref_logits, 1.0)
    r_graph = _reliability(extra, "r_graph", ref_logits, 1.0)
    r_manifest = _reliability(extra, "r_manifest", ref_logits, 0.0)
    api_manifest = _reliability(extra, "api_manifest_consistency", ref_logits, 0.0)
    graph_manifest = _reliability(extra, "graph_manifest_consistency", ref_logits, 0.0)
    min_reliability = float(loss_cfg.get("cross_source_min_reliability", 0.0))
    min_consistency = float(loss_cfg.get("cross_source_min_consistency", 0.0))
    active_only = bool(loss_cfg.get("semantic_active_only", False))

    reconstruction_terms: list[torch.Tensor] = []
    cross_source_terms: list[torch.Tensor] = []
    for pred, target, weight, consistency in (
        (api_pred, api_counts, r_api, None),
        (graph_pred, graph_counts, r_graph, None),
        (manifest_pred, manifest_counts, r_manifest, None),
    ):
        filtered_weight = _pair_weight(weight, None, min_reliability, 0.0)
        term, used_weight = _weighted_cosine_direction_loss(pred, target, filtered_weight, active_only=active_only)
        if float(used_weight.detach().item()) > 0.0:
            reconstruction_terms.append(term)

    for pred, target, weight, consistency in (
        (api_pred, manifest_counts, r_api * r_manifest, api_manifest),
        (graph_pred, manifest_counts, r_graph * r_manifest, graph_manifest),
        (manifest_pred, api_counts, r_manifest * r_api, api_manifest),
        (manifest_pred, graph_counts, r_manifest * r_graph, graph_manifest),
    ):
        filtered_weight = _pair_weight(weight, consistency, min_reliability, min_consistency)
        term, used_weight = _weighted_cosine_direction_loss(pred, target, filtered_weight, active_only=active_only)
        if float(used_weight.detach().item()) > 0.0:
            cross_source_terms.append(term)

    reconstruction = (
        torch.stack(reconstruction_terms).mean().to(dtype=ref_logits.dtype)
        if reconstruction_terms
        else ref_logits.new_tensor(0.0)
    )
    cross_source = (
        torch.stack(cross_source_terms).mean().to(dtype=ref_logits.dtype)
        if cross_source_terms
        else ref_logits.new_tensor(0.0)
    )
    return reconstruction, cross_source


def _gate_prior_target(extra: dict, ref_logits: torch.Tensor) -> torch.Tensor:
    """Return a heuristic gate-prior target for the learned gate.

    Prefer ``extra["gate_evidence"]`` (already built during the forward pass)
    over reconstructing the evidence tensor from scratch.  Fall back to the
    explicit rebuild only when the cached tensor is missing (e.g. standalone
    loss tests).
    """
    dtype = ref_logits.dtype
    cached = extra.get("gate_evidence")
    if isinstance(cached, torch.Tensor):
        evidence = cached[..., : EvidenceIndex.BASE_DIM].to(device=ref_logits.device, dtype=dtype)
    else:
        batch_size = ref_logits.size(0)
        device = ref_logits.device
        evidence = torch.zeros((batch_size, EvidenceIndex.BASE_DIM), device=device, dtype=dtype)
        evidence[:, EvidenceIndex.R_API] = _reliability(extra, "r_api", ref_logits, 1.0)
        evidence[:, EvidenceIndex.R_GRAPH] = _reliability(extra, "r_graph", ref_logits, 1.0)
        evidence[:, EvidenceIndex.R_MANIFEST] = _reliability(extra, "r_manifest", ref_logits, 0.0)
        evidence[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = _reliability(extra, "api_graph_anchor_support", ref_logits, 0.0)
        evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = _reliability(extra, "manifest_code_support", ref_logits, 0.0)
        evidence[:, EvidenceIndex.API_ALIVE] = _reliability(extra, "api_alive", ref_logits, 1.0)
        evidence[:, EvidenceIndex.GRAPH_ALIVE] = _reliability(extra, "graph_alive", ref_logits, 1.0)
        evidence[:, EvidenceIndex.MANIFEST_ALIVE] = _reliability(extra, "manifest_alive", ref_logits, 0.0)
    return heuristic_reliability_gate(evidence).to(dtype=dtype).detach()


def _gate_prior_loss(extra: dict, ref_logits: torch.Tensor) -> torch.Tensor:
    if not bool(extra.get("gate_prior_enabled", False)):
        return ref_logits.new_tensor(0.0)
    gate_weights = extra.get("gate_weights_train")
    if not isinstance(gate_weights, torch.Tensor):
        return ref_logits.new_tensor(0.0)
    gate_weights = gate_weights.to(device=ref_logits.device, dtype=ref_logits.dtype)
    if gate_weights.ndim != 2 or gate_weights.size(0) != ref_logits.size(0) or gate_weights.size(1) != 4:
        return ref_logits.new_tensor(0.0)
    target = _gate_prior_target(extra, ref_logits).to(device=ref_logits.device, dtype=ref_logits.dtype)
    return F.kl_div(
        gate_weights.clamp_min(1e-8).log(),
        target,
        reduction="batchmean",
    ).to(dtype=ref_logits.dtype)


def compute_robust_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    extra: dict | None = None,
    loss_cfg: dict | None = None,
    *,
    batch=None,
    evidence: torch.Tensor | None = None,
    semantic_reconstruction_cfg: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Robust objective with independently attributable auxiliary terms."""
    extra = extra or {}
    loss_cfg = loss_cfg or {}
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    branch_aux_weight = float(loss_cfg.get("branch_aux_weight", 0.05))
    semantic_reconstruction_weight = float(loss_cfg.get("semantic_reconstruction_weight", 0.0))
    cross_source_consistency_weight = float(loss_cfg.get("cross_source_consistency_weight", 0.0))
    gate_prior_weight = float(loss_cfg.get("gate_prior_weight", 0.0))
    reliability_calibration_weight = float(loss_cfg.get("reliability_calibration_weight", 0.0))
    probability_calibration_weight = float(loss_cfg.get("probability_calibration_weight", 0.0))
    named_weights = {
        "branch_aux_weight": branch_aux_weight,
        "semantic_reconstruction_weight": semantic_reconstruction_weight,
        "cross_source_consistency_weight": cross_source_consistency_weight,
        "gate_prior_weight": gate_prior_weight,
        "reliability_calibration_weight": reliability_calibration_weight,
        "probability_calibration_weight": probability_calibration_weight,
    }
    for name, value in named_weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    for name in ("cross_source_min_reliability", "cross_source_min_consistency"):
        value = float(loss_cfg.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}")

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

    if semantic_reconstruction_weight > 0.0 or cross_source_consistency_weight > 0.0:
        semantic_reconstruction, cross_source_consistency = _semantic_losses(extra, logits, loss_cfg)
    else:
        semantic_reconstruction = logits.new_tensor(0.0)
        cross_source_consistency = logits.new_tensor(0.0)
    gate_prior = (
        _gate_prior_loss(extra, logits)
        if gate_prior_weight > 0.0
        else logits.new_tensor(0.0)
    )
    masked_reconstruction = logits.new_tensor(0.0)
    masked_diagnostics: dict[str, torch.Tensor] = {}
    semantic_reconstruction_cfg = semantic_reconstruction_cfg or {}
    masked_reconstruction_weight = float(semantic_reconstruction_cfg.get("weight", 0.0))
    if not math.isfinite(masked_reconstruction_weight) or masked_reconstruction_weight < 0.0:
        raise ValueError(
            "semantic_reconstruction.weight must be finite and non-negative, "
            f"got {masked_reconstruction_weight}"
        )
    if bool(semantic_reconstruction_cfg.get("enabled", False)) and masked_reconstruction_weight > 0.0:
        if batch is None or not isinstance(evidence, torch.Tensor):
            raise ValueError("enabled masked semantic reconstruction requires batch and observable evidence")
        masked_reconstruction, masked_diagnostics = compute_masked_semantic_reconstruction_loss(
            extra,
            batch,
            evidence,
            semantic_reconstruction_cfg,
        )

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

    total = (
        ce
        + branch_aux_weight * branch_loss
        + masked_reconstruction_weight * masked_reconstruction
        + semantic_reconstruction_weight * semantic_reconstruction
        + cross_source_consistency_weight * cross_source_consistency
        + gate_prior_weight * gate_prior
        + reliability_calibration_weight * reliability_calibration
        + probability_calibration_weight * probability_calibration
    )
    parts = {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "branch_aux": float(branch_loss.detach().item()),
        "branch_aux_weight": branch_aux_weight,
        "semantic_reconstruction": float(semantic_reconstruction.detach().item()),
        "semantic_reconstruction_weight": semantic_reconstruction_weight,
        "cross_source_consistency": float(cross_source_consistency.detach().item()),
        "cross_source_consistency_weight": cross_source_consistency_weight,
        "cross_source_min_reliability": float(loss_cfg.get("cross_source_min_reliability", 0.0)),
        "cross_source_min_consistency": float(loss_cfg.get("cross_source_min_consistency", 0.0)),
        "gate_prior": float(gate_prior.detach().item()),
        "gate_prior_weight": gate_prior_weight,
        "masked_semantic_reconstruction": float(masked_reconstruction.detach().item()),
        "masked_semantic_reconstruction_weight": masked_reconstruction_weight,
        "reliability_calibration": float(reliability_calibration.detach().item()),
        "reliability_calibration_weight": reliability_calibration_weight,
        "probability_calibration": float(probability_calibration.detach().item()),
        "probability_calibration_weight": probability_calibration_weight,
    }
    for source in (aux_diagnostics, masked_diagnostics, calibration_diagnostics):
        for key, value in source.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                parts[key] = float(value.detach().item())
    return total, parts
