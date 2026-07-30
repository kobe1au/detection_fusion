"""Explicit train-only Stage-B lifecycle for competence-anchored fusion.

Stage A owns the encoders, atomic heads, and Joint expert.  This module freezes
that artifact, caches its current-view outputs, then fits:

1. expert-local continuous TCP competence heads; and
2. the small Joint-anchor / reliable-late router.

No validation decision-calibration row and no test row is accepted by this
module.  Callers provide only train caches for parameter fitting and a
model-selection cache for early stopping / hyperparameter choice.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from fusion.competence_fusion import (
    ATOMIC_EXPERT_NAMES,
    EXPERT_NAMES,
    competence_learning_loss,
    true_class_probability_targets,
)
from fusion.dataset import prepare_robust_batch
from fusion.thresholds import fit_binary_macro_f1_threshold
from fusion.utils import get_amp_context


@dataclass(frozen=True)
class CachedExpertBatch:
    labels: torch.Tensor
    embeddings: dict[str, torch.Tensor]
    logits: dict[str, torch.Tensor]
    alive: dict[str, torch.Tensor]

    @property
    def num_samples(self) -> int:
        return int(self.labels.numel())

    def to(self, device: torch.device) -> "CachedExpertBatch":
        return CachedExpertBatch(
            labels=self.labels.to(device=device, non_blocking=True),
            embeddings={
                name: value.to(device=device, non_blocking=True)
                for name, value in self.embeddings.items()
            },
            logits={
                name: value.to(device=device, non_blocking=True)
                for name, value in self.logits.items()
            },
            alive={
                name: value.to(device=device, non_blocking=True)
                for name, value in self.alive.items()
            },
        )


def _require_anchored_modules(model) -> tuple[torch.nn.Module, torch.nn.Module]:
    estimator = getattr(model, "competence_estimator", None)
    router = getattr(model, "anchored_fusion", None)
    if not isinstance(estimator, torch.nn.Module) or not isinstance(
        router, torch.nn.Module
    ):
        raise ValueError(
            "Stage B requires a model with competence_estimator and anchored_fusion"
        )
    estimator_parameter_ids = {id(parameter) for parameter in estimator.parameters()}
    router_parameter_ids = {id(parameter) for parameter in router.parameters()}
    if estimator_parameter_ids & router_parameter_ids:
        raise RuntimeError(
            "competence_estimator and anchored_fusion must not share parameters"
        )
    return estimator, router


def _clear_gradients(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


def _require_nonempty_cache_collection(
    caches: list[list[CachedExpertBatch]],
    *,
    label: str,
) -> None:
    if not isinstance(caches, list) or not caches:
        raise ValueError(f"{label} must contain at least one cache")
    if not any(cache for cache in caches):
        raise ValueError(f"{label} must contain at least one cached batch")


@torch.no_grad()
def cache_expert_outputs(
    model,
    loader,
    device: torch.device,
    *,
    use_amp: bool,
    source_name: str,
) -> tuple[list[CachedExpertBatch], dict[str, Any]]:
    """Cache one immutable expert pass and validate its probability contract."""

    _require_anchored_modules(model)
    previous_active = bool(getattr(model, "anchored_fusion_active", False))
    previous_training = bool(model.training)
    model.set_anchored_fusion_active(False)
    model.eval()
    cached: list[CachedExpertBatch] = []
    failed_total = 0
    started = time.perf_counter()
    try:
        for raw_batch in loader:
            graph, labels, _sids, failed = prepare_robust_batch(raw_batch, device)
            failed_total += int(failed)
            if graph is None:
                continue
            with get_amp_context(device, use_amp):
                _joint_logits, extra = model(graph)
            embeddings = extra.get("expert_embeddings")
            logits = extra.get("expert_logits")
            alive = extra.get("expert_alive")
            if (
                not isinstance(embeddings, dict)
                or not isinstance(logits, dict)
                or not isinstance(alive, dict)
                or set(embeddings) != set(EXPERT_NAMES)
                or set(logits) != set(EXPERT_NAMES)
                or set(alive) != set(EXPERT_NAMES)
            ):
                raise RuntimeError(
                    "Anchored model did not expose the exact four-expert cache contract"
                )
            raw_labels = labels.detach().view(-1).cpu()
            if raw_labels.is_floating_point() and not bool(
                torch.isfinite(raw_labels).all()
            ):
                raise FloatingPointError("Stage-B cache labels must be finite")
            if not bool(((raw_labels == 0) | (raw_labels == 1)).all()):
                raise ValueError("Stage-B cache requires binary labels 0/1")
            labels_cpu = raw_labels.long()
            embeddings_cpu: dict[str, torch.Tensor] = {}
            logits_cpu: dict[str, torch.Tensor] = {}
            alive_cpu: dict[str, torch.Tensor] = {}
            for name in EXPERT_NAMES:
                embedding = embeddings[name].detach().float().cpu()
                expert_logits = logits[name].detach().float().cpu()
                raw_alive = alive[name].detach().view(-1).cpu()
                if raw_alive.is_floating_point() and not bool(
                    torch.isfinite(raw_alive).all()
                ):
                    raise FloatingPointError(
                        f"Non-finite availability for expert {name!r}"
                    )
                if not bool(((raw_alive == 0) | (raw_alive == 1)).all()):
                    raise ValueError(
                        f"Availability for expert {name!r} must be hard 0/1"
                    )
                expert_alive = raw_alive.bool()
                if (
                    embedding.ndim != 2
                    or expert_logits.ndim != 2
                    or int(embedding.size(0)) != labels_cpu.numel()
                    or int(expert_logits.size(0)) != labels_cpu.numel()
                    or int(expert_logits.size(1)) != 2
                    or expert_alive.numel() != labels_cpu.numel()
                ):
                    raise RuntimeError(
                        f"Invalid cached expert tensor shapes for {name!r}"
                    )
                if not bool(torch.isfinite(embedding).all()) or not bool(
                    torch.isfinite(expert_logits).all()
                ):
                    raise FloatingPointError(
                        f"Non-finite Stage-B cache tensors for expert {name!r}"
                    )
                probability = torch.softmax(expert_logits, dim=-1)
                if not bool(torch.isfinite(probability).all()) or not torch.allclose(
                    probability.sum(dim=-1),
                    torch.ones(probability.size(0)),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                ):
                    raise FloatingPointError(
                        f"Invalid normalized probabilities for expert {name!r}"
                    )
                embeddings_cpu[name] = embedding
                logits_cpu[name] = expert_logits
                alive_cpu[name] = expert_alive
            cached.append(
                CachedExpertBatch(
                    labels=labels_cpu,
                    embeddings=embeddings_cpu,
                    logits=logits_cpu,
                    alive=alive_cpu,
                )
            )
    finally:
        model.set_anchored_fusion_active(previous_active)
        model.train(previous_training)
    if failed_total:
        raise RuntimeError(
            f"Stage-B cache {source_name!r} dropped {failed_total} samples"
        )
    if not cached:
        raise RuntimeError(f"Stage-B cache {source_name!r} is empty")
    return cached, {
        "source": str(source_name),
        "num_batches": int(len(cached)),
        "num_samples": int(sum(item.num_samples for item in cached)),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _expert_probabilities(
    batch: CachedExpertBatch,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.softmax(batch.logits[name].float(), dim=-1)
        for name in EXPERT_NAMES
    }


def _competence_forward(model, batch: CachedExpertBatch):
    estimator, _router = _require_anchored_modules(model)
    return estimator(batch.embeddings, batch.logits, batch.alive)


def _competence_loss_statistics(
    model,
    cache: list[CachedExpertBatch],
    device: torch.device,
    *,
    regression: str,
    ranking_weight: float,
    tie_tolerance: float,
) -> dict[str, float | int]:
    """Evaluate one source with exact valid-row/pair aggregation.

    ``competence_learning_loss`` normalizes regression by alive expert rows and
    ranking by valid atomic pairs inside each batch. Averaging those already
    normalized values by the number of APKs would make validation depend on
    batch boundaries. Recovering each numerator and denominator before the
    source-level division makes the model-selection objective invariant to
    batching.
    """

    estimator, _router = _require_anchored_modules(model)
    estimator.eval()
    regression_numerator = 0.0
    ranking_numerator = 0.0
    valid_rows = 0
    valid_pairs = 0
    num_samples = 0
    with torch.no_grad():
        for cpu_batch in cache:
            batch = cpu_batch.to(device)
            output = _competence_forward(model, batch)
            targets = true_class_probability_targets(
                _expert_probabilities(batch),
                batch.labels,
            )
            loss = competence_learning_loss(
                output.competence,
                targets,
                batch.alive,
                regression=regression,
                ranking_weight=ranking_weight,
                tie_tolerance=tie_tolerance,
            )
            batch_valid_rows = int(loss.valid_expert_rows.detach().cpu())
            batch_valid_pairs = int(loss.valid_atomic_pairs.detach().cpu())
            regression_numerator += (
                float(loss.regression.detach().cpu()) * batch_valid_rows
            )
            ranking_numerator += (
                float(loss.ranking.detach().cpu()) * batch_valid_pairs
            )
            valid_rows += batch_valid_rows
            valid_pairs += batch_valid_pairs
            num_samples += int(batch.labels.numel())
    if valid_rows <= 0:
        raise ValueError("competence validation source has no alive expert rows")
    regression_loss = regression_numerator / float(valid_rows)
    ranking_loss = (
        ranking_numerator / float(valid_pairs) if valid_pairs > 0 else 0.0
    )
    total = regression_loss + float(ranking_weight) * ranking_loss
    if not all(
        math.isfinite(value)
        for value in (regression_loss, ranking_loss, total)
    ):
        raise FloatingPointError("Non-finite source-level competence loss")
    return {
        "total_loss": float(total),
        "regression_loss": float(regression_loss),
        "ranking_loss": float(ranking_loss),
        "valid_expert_rows": int(valid_rows),
        "valid_atomic_pairs": int(valid_pairs),
        "num_samples": int(num_samples),
    }


def _select_competence_candidate(
    candidates: list[dict[str, Any]],
    *,
    clean_source: str,
    clean_noninferiority_relative_tolerance: float,
    clean_noninferiority_absolute_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    if not candidates:
        raise ValueError("competence selection requires at least one candidate")
    clean_losses = [
        float(candidate["validation_sources"][clean_source]["total_loss"])
        for candidate in candidates
    ]
    if not all(math.isfinite(value) for value in clean_losses):
        raise FloatingPointError("I1 clean validation losses must be finite")
    clean_reference = float(min(clean_losses))
    relative_tolerance = float(clean_noninferiority_relative_tolerance)
    absolute_tolerance = float(clean_noninferiority_absolute_tolerance)
    clean_upper_bound = (
        clean_reference * (1.0 + relative_tolerance) + absolute_tolerance
    )
    degraded_sources = sorted(
        set(candidates[0]["validation_sources"]) - {clean_source}
    )
    public_candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_key: tuple[float, ...] | None = None
    for candidate in candidates:
        source_losses = candidate["validation_sources"]
        if set(source_losses) != set(candidates[0]["validation_sources"]):
            raise RuntimeError("I1 validation source sets changed across epochs")
        clean_loss = float(source_losses[clean_source]["total_loss"])
        clean_noninferior = clean_loss <= clean_upper_bound
        degraded_values = [
            float(source_losses[name]["total_loss"]) for name in degraded_sources
        ]
        degraded_mean = (
            float(np.mean(degraded_values)) if degraded_values else None
        )
        degraded_worst = (
            float(np.max(degraded_values)) if degraded_values else None
        )
        # Maximize this tuple: clean feasibility is a hard first stage, then
        # minimize degraded mean and worst-source loss. Clean loss and earlier
        # epoch are deterministic final tie-breakers only.
        if degraded_values:
            selection_key = (
                float(clean_noninferior),
                -float(degraded_mean),
                -float(degraded_worst),
                -clean_loss,
                -float(candidate["epoch"]),
            )
        else:
            selection_key = (
                float(clean_noninferior),
                -clean_loss,
                -float(candidate["epoch"]),
            )
        public = {
            "epoch": int(candidate["epoch"]),
            "clean_noninferior": bool(clean_noninferior),
            "clean_loss": float(clean_loss),
            "degraded_mean_loss": degraded_mean,
            "degraded_worst_source_loss": degraded_worst,
            "validation_sources": copy.deepcopy(source_losses),
        }
        public_candidates.append(public)
        if clean_noninferior and (
            selected_key is None or selection_key > selected_key
        ):
            selected = candidate
            selected_key = selection_key
    if selected is None:
        raise RuntimeError("I1 clean non-inferiority selection found no candidate")
    return selected, public_candidates, clean_reference


def fit_competence_heads(
    model,
    *,
    train_clean: list[CachedExpertBatch],
    train_degraded: list[CachedExpertBatch],
    validation_sources: dict[str, list[CachedExpertBatch]],
    clean_validation_source: str,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit I1 and select an epoch without an arbitrary source mixture weight.

    Clean model-selection loss is a hard non-inferiority constraint. Among
    clean-safe epochs, selection minimizes the mean loss over the fixed
    single-modality degraded views and then their worst-source loss. A caller
    may provide only the clean source for the registered clean-only ablation.
    """

    estimator, router = _require_anchored_modules(model)
    _require_nonempty_cache_collection([train_clean], label="train_clean")
    if not isinstance(validation_sources, dict) or not validation_sources:
        raise ValueError("validation_sources must contain named cached sources")
    if clean_validation_source not in validation_sources:
        raise ValueError(
            f"clean validation source {clean_validation_source!r} is not present"
        )
    if any(not cache for cache in validation_sources.values()):
        raise ValueError("every I1 validation source must contain cached batches")
    epochs = int(config.get("epochs", 30))
    patience = int(config.get("patience", 6))
    if epochs <= 0 or patience <= 0:
        raise ValueError("competence epochs and patience must be positive")
    lr = float(config.get("lr", 1.0e-3))
    weight_decay = float(config.get("weight_decay", 1.0e-4))
    ranking_weight = float(config.get("ranking_weight", 0.10))
    tie_tolerance = float(config.get("ranking_tie_tolerance", 0.02))
    degraded_loss_weight = float(config.get("degraded_loss_weight", 0.25))
    clean_noninferiority_relative_tolerance = float(
        config.get("clean_noninferiority_relative_tolerance", 0.01)
    )
    clean_noninferiority_absolute_tolerance = float(
        config.get("clean_noninferiority_absolute_tolerance", 0.0)
    )
    regression = str(config.get("regression", "mse")).strip().lower()
    if regression not in {"mse", "bce"}:
        raise ValueError("competence regression must be 'mse' or 'bce'")
    grad_clip = float(config.get("grad_clip", 5.0))
    for name, value in {
        "lr": lr,
        "weight_decay": weight_decay,
        "ranking_weight": ranking_weight,
        "ranking_tie_tolerance": tie_tolerance,
        "degraded_loss_weight": degraded_loss_weight,
        "clean_noninferiority_relative_tolerance": (
            clean_noninferiority_relative_tolerance
        ),
        "clean_noninferiority_absolute_tolerance": (
            clean_noninferiority_absolute_tolerance
        ),
        "grad_clip": grad_clip,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"competence {name} must be finite and non-negative")
    if lr <= 0.0 or grad_clip <= 0.0:
        raise ValueError("competence lr and grad_clip must be positive")
    if degraded_loss_weight > 0.0:
        _require_nonempty_cache_collection(
            [train_degraded],
            label="train_degraded",
        )

    model.zero_grad(set_to_none=True)
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    for parameter in estimator.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        estimator.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    epoch_candidates: list[dict[str, Any]] = []
    selected_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        estimator.train()
        total = 0.0
        clean_total = 0.0
        degraded_total = 0.0
        steps = 0
        max_batches = (
            max(len(train_clean), len(train_degraded))
            if degraded_loss_weight > 0.0
            else len(train_clean)
        )
        for batch_index in range(max_batches):
            optimizer.zero_grad(set_to_none=True)
            clean_batch = train_clean[batch_index % len(train_clean)].to(device)
            clean_output = _competence_forward(model, clean_batch)
            clean_targets = true_class_probability_targets(
                _expert_probabilities(clean_batch),
                clean_batch.labels,
            )
            clean_losses = competence_learning_loss(
                clean_output.competence,
                clean_targets,
                clean_batch.alive,
                regression=regression,
                ranking_weight=ranking_weight,
                tie_tolerance=tie_tolerance,
            )
            loss = clean_losses.total
            degraded_value = 0.0
            if degraded_loss_weight > 0.0:
                degraded_batch = train_degraded[
                    batch_index % len(train_degraded)
                ].to(device)
                degraded_output = _competence_forward(model, degraded_batch)
                degraded_targets = true_class_probability_targets(
                    _expert_probabilities(degraded_batch),
                    degraded_batch.labels,
                )
                degraded_losses = competence_learning_loss(
                    degraded_output.competence,
                    degraded_targets,
                    degraded_batch.alive,
                    regression=regression,
                    ranking_weight=ranking_weight,
                    tie_tolerance=tie_tolerance,
                )
                # Normalization keeps the optimizer scale stable while the
                # configured coefficient controls only the relative auxiliary
                # contribution.
                loss = (
                    loss + degraded_loss_weight * degraded_losses.total
                ) / (1.0 + degraded_loss_weight)
                degraded_value = float(
                    degraded_losses.total.detach().cpu()
                )
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError("Non-finite competence loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                estimator.parameters(),
                grad_clip,
            )
            if not bool(torch.isfinite(gradient_norm.detach()).all()):
                raise FloatingPointError("Non-finite competence gradient")
            optimizer.step()
            total += float(loss.detach().cpu())
            clean_total += float(clean_losses.total.detach().cpu())
            degraded_total += degraded_value
            steps += 1
        source_statistics = {
            source_name: _competence_loss_statistics(
                model,
                cache,
                device,
                regression=regression,
                ranking_weight=ranking_weight,
                tie_tolerance=tie_tolerance,
            )
            for source_name, cache in validation_sources.items()
        }
        epoch_candidates.append(
            {
                "epoch": int(epoch),
                "state": copy.deepcopy(estimator.state_dict()),
                "validation_sources": source_statistics,
            }
        )
        selected_candidate, _public_candidates, _clean_reference = (
            _select_competence_candidate(
                epoch_candidates,
                clean_source=clean_validation_source,
                clean_noninferiority_relative_tolerance=(
                    clean_noninferiority_relative_tolerance
                ),
                clean_noninferiority_absolute_tolerance=(
                    clean_noninferiority_absolute_tolerance
                ),
            )
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(total / max(steps, 1)),
                "train_clean_loss": float(clean_total / max(steps, 1)),
                "train_degraded_loss": (
                    float(degraded_total / max(steps, 1))
                    if degraded_loss_weight > 0.0
                    else None
                ),
                "validation_sources": copy.deepcopy(source_statistics),
            }
        )
        current_selected_epoch = int(selected_candidate["epoch"])
        if current_selected_epoch != selected_epoch:
            selected_epoch = current_selected_epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    selected, public_candidates, clean_reference = _select_competence_candidate(
        epoch_candidates,
        clean_source=clean_validation_source,
        clean_noninferiority_relative_tolerance=(
            clean_noninferiority_relative_tolerance
        ),
        clean_noninferiority_absolute_tolerance=(
            clean_noninferiority_absolute_tolerance
        ),
    )
    estimator.load_state_dict(selected["state"], strict=True)
    estimator.eval()
    for parameter in estimator.parameters():
        parameter.requires_grad_(False)
    diagnostics_setter = getattr(
        model, "set_competence_diagnostics_active", None
    )
    if callable(diagnostics_setter):
        diagnostics_setter(True)
    model.zero_grad(set_to_none=True)
    return {
        "best_epoch": int(selected["epoch"]),
        "best_validation_loss": float(
            selected["validation_sources"][clean_validation_source][
                "total_loss"
            ]
        ),
        "selection_policy": (
            "clean_noninferiority_then_degraded_mean_then_worst_source_v1"
            if len(validation_sources) > 1
            else "clean_only_minimum_tcp_loss_v1"
        ),
        "early_stopping_population": (
            "val_model_selection_clean_plus_fixed_single_modality_degradations"
            if len(validation_sources) > 1
            else "val_model_selection_clean_only"
        ),
        "clean_validation_source": str(clean_validation_source),
        "clean_loss_reference": float(clean_reference),
        "clean_noninferiority_relative_tolerance": float(
            clean_noninferiority_relative_tolerance
        ),
        "clean_noninferiority_absolute_tolerance": float(
            clean_noninferiority_absolute_tolerance
        ),
        "clean_noninferiority_effective_upper_bound": float(
            clean_reference
            * (1.0 + clean_noninferiority_relative_tolerance)
            + clean_noninferiority_absolute_tolerance
        ),
        "selected_validation_sources": copy.deepcopy(
            selected["validation_sources"]
        ),
        "candidates": public_candidates,
        "regression": regression,
        "ranking_weight": float(ranking_weight),
        "ranking_tie_tolerance": float(tie_tolerance),
        "degraded_loss_weight": float(degraded_loss_weight),
        "history": history,
        "wall_seconds": float(time.perf_counter() - started),
    }


def _binary_macro_f1(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    return float(
        f1_score(
            labels.astype(np.int64),
            (probability >= float(threshold)).astype(np.int64),
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )


def _fit_macro_f1_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    return fit_binary_macro_f1_threshold(labels, probability)


@torch.no_grad()
def evaluate_cached_fusion(
    model,
    sources: dict[str, list[CachedExpertBatch]],
    device: torch.device,
    *,
    clean_source: str = "clean",
) -> dict[str, Any]:
    estimator, router = _require_anchored_modules(model)
    estimator.eval()
    router.eval()
    if not isinstance(sources, dict) or not sources:
        raise ValueError("fusion evaluation requires at least one named source")
    if clean_source not in sources:
        raise ValueError(
            f"clean fusion-evaluation source {clean_source!r} is not present"
        )
    source_rows: dict[str, dict[str, np.ndarray]] = {}
    for source_name, cache in sources.items():
        if not cache:
            raise ValueError(
                f"fusion evaluation source {source_name!r} has no cached batches"
            )
        labels_all: list[np.ndarray] = []
        final_all: list[np.ndarray] = []
        joint_all: list[np.ndarray] = []
        late_gate_all: list[np.ndarray] = []
        for cpu_batch in cache:
            batch = cpu_batch.to(device)
            probabilities = _expert_probabilities(batch)
            competence = _competence_forward(model, batch)
            output = router(
                probabilities,
                competence.competence,
                batch.alive,
            )
            labels_all.append(batch.labels.detach().cpu().numpy())
            final_all.append(output.probability[:, 1].detach().cpu().numpy())
            joint_all.append(
                output.joint_probability[:, 1].detach().cpu().numpy()
            )
            late_gate_all.append(output.late_gate.detach().cpu().numpy())
        labels = np.concatenate(labels_all)
        final = np.concatenate(final_all)
        joint = np.concatenate(joint_all)
        late_gate = np.concatenate(late_gate_all)
        if not (
            np.isfinite(final).all()
            and np.isfinite(joint).all()
            and np.isfinite(late_gate).all()
        ):
            raise FloatingPointError(
                f"Non-finite fusion evaluation output for source {source_name!r}"
            )
        source_rows[source_name] = {
            "labels": labels.astype(np.int64),
            "probability": final.astype(np.float64),
            "joint_anchor_probability": joint.astype(np.float64),
            "late_gate": late_gate.astype(np.float64),
        }
    clean_rows = source_rows[clean_source]
    candidate_threshold = _fit_macro_f1_threshold(
        clean_rows["labels"],
        clean_rows["probability"],
    )
    joint_threshold = _fit_macro_f1_threshold(
        clean_rows["labels"],
        clean_rows["joint_anchor_probability"],
    )
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_name, rows in source_rows.items():
        labels = rows["labels"]
        final = rows["probability"]
        joint = rows["joint_anchor_probability"]
        late_gate = rows["late_gate"]
        source_metrics[source_name] = {
            "num_samples": int(labels.size),
            "macro_f1": _binary_macro_f1(
                labels,
                final,
                threshold=float(candidate_threshold["threshold"]),
            ),
            "joint_anchor_macro_f1": _binary_macro_f1(
                labels,
                joint,
                threshold=float(joint_threshold["threshold"]),
            ),
            "fixed_0_5_macro_f1": _binary_macro_f1(
                labels, final, threshold=0.5
            ),
            "joint_anchor_fixed_0_5_macro_f1": _binary_macro_f1(
                labels, joint, threshold=0.5
            ),
            "nll": float(
                -np.mean(
                    np.log(
                        np.where(labels == 1, final, 1.0 - final).clip(
                            1.0e-8, 1.0
                        )
                    )
                )
            ),
            "joint_anchor_nll": float(
                -np.mean(
                    np.log(
                        np.where(labels == 1, joint, 1.0 - joint).clip(
                            1.0e-8, 1.0
                        )
                    )
                )
            ),
            "mean_late_gate": float(np.mean(late_gate)),
            # Per-row probabilities remain available to selection tests and
            # audits. Callers strip them before serializing checkpoint/summary
            # metadata.
            "rows": {
                "labels": labels.tolist(),
                "probability": final.tolist(),
                "joint_anchor_probability": joint.tolist(),
            },
        }
    return {
        "clean_source": str(clean_source),
        "classification_threshold": candidate_threshold,
        "joint_anchor_classification_threshold": joint_threshold,
        "sources": source_metrics,
    }


def _fusion_metrics_without_rows(metrics: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(metrics)
    for source in (public.get("sources") or {}).values():
        if isinstance(source, dict):
            source.pop("rows", None)
    return public


def _router_selection_state(
    metrics: dict[str, Any],
    *,
    clean_source: str,
    clean_noninferiority_tolerance: float,
    degraded_source_noninferiority_tolerance: float,
    minimum_robust_gain: float,
) -> dict[str, Any]:
    source_metrics = metrics["sources"]
    if clean_source not in source_metrics:
        raise ValueError(
            f"clean validation source {clean_source!r} is not present"
        )
    clean = source_metrics[clean_source]
    degraded = {
        name: value
        for name, value in source_metrics.items()
        if name != clean_source
    }
    if not degraded:
        raise ValueError("Router selection requires at least one degraded source")
    clean_delta = float(clean["macro_f1"]) - float(
        clean["joint_anchor_macro_f1"]
    )
    clean_ok = clean_delta >= -float(clean_noninferiority_tolerance)
    robust_values = [float(value["macro_f1"]) for value in degraded.values()]
    robust_joint_values = [
        float(value["joint_anchor_macro_f1"]) for value in degraded.values()
    ]
    selection_values = [
        float(clean["macro_f1"]),
        float(clean["joint_anchor_macro_f1"]),
        *robust_values,
        *robust_joint_values,
    ]
    if not all(math.isfinite(value) for value in selection_values):
        raise FloatingPointError("Router selection metrics must be finite")
    robust_mean = float(np.mean(robust_values))
    robust_joint_mean = float(np.mean(robust_joint_values))
    robust_gain = robust_mean - robust_joint_mean
    source_deltas = {
        name: float(value["macro_f1"])
        - float(value["joint_anchor_macro_f1"])
        for name, value in degraded.items()
    }
    min_source_delta = float(min(source_deltas.values()))
    every_source_noninferior = all(
        delta >= -float(degraded_source_noninferiority_tolerance)
        for delta in source_deltas.values()
    )
    robust_improved = robust_gain > float(minimum_robust_gain)
    eligible = bool(clean_ok and every_source_noninferior and robust_improved)
    mean_nll = float(
        np.mean([float(value["nll"]) for value in source_metrics.values()])
    )
    return {
        "selection_tuple": (
            eligible,
            min_source_delta,
            robust_gain,
            clean_delta,
            -mean_nll,
        ),
        "eligible": eligible,
        "clean_noninferior": bool(clean_ok),
        "every_degraded_source_noninferior": bool(
            every_source_noninferior
        ),
        "robust_improved": bool(robust_improved),
        "degraded_source_deltas": source_deltas,
        "minimum_degraded_source_delta": min_source_delta,
        "robust_mean_macro_f1": robust_mean,
        "robust_joint_mean_macro_f1": robust_joint_mean,
        "robust_mean_gain": robust_gain,
        "robust_worst_macro_f1": float(np.min(robust_values)),
        "clean_macro_f1": float(clean["macro_f1"]),
        "clean_delta": float(clean_delta),
        "mean_nll": float(mean_nll),
        "joint_anchor_clean_macro_f1": float(
            clean["joint_anchor_macro_f1"]
        ),
    }


def _router_selection_tuple(
    metrics: dict[str, Any],
    *,
    clean_source: str,
    clean_noninferiority_tolerance: float,
    degraded_source_noninferiority_tolerance: float = 0.0,
    minimum_robust_gain: float = 0.0,
) -> tuple[bool, float, float, float, float]:
    """Return the lexicographic router deployment objective.

    A candidate is eligible only when clean performance is non-inferior *and*
    its mean degraded-source macro-F1 strictly improves over the fixed Joint
    anchor by more than ``minimum_robust_gain``.
    """

    return _router_selection_state(
        metrics,
        clean_source=clean_source,
        clean_noninferiority_tolerance=clean_noninferiority_tolerance,
        degraded_source_noninferiority_tolerance=(
            degraded_source_noninferiority_tolerance
        ),
        minimum_robust_gain=minimum_robust_gain,
    )["selection_tuple"]


def _router_training_loss(
    model,
    batch: CachedExpertBatch,
    *,
    anchor_kl_weight: float,
    is_clean: bool,
) -> torch.Tensor:
    estimator, router = _require_anchored_modules(model)
    with torch.no_grad():
        competence = estimator(batch.embeddings, batch.logits, batch.alive)
        probabilities = _expert_probabilities(batch)
    output = router(probabilities, competence.competence, batch.alive)
    nll = F.nll_loss(
        output.probability.clamp_min(1.0e-8).log(),
        batch.labels.long(),
    )
    total = nll
    if is_clean and anchor_kl_weight > 0.0:
        anchor_kl = F.kl_div(
            output.probability.clamp_min(1.0e-8).log(),
            output.joint_probability.detach(),
            reduction="batchmean",
        )
        total = total + float(anchor_kl_weight) * anchor_kl
    return total


def _normalized_clean_degraded_loss(
    clean_loss: torch.Tensor,
    degraded_loss: torch.Tensor | None,
    *,
    degraded_weight: float,
) -> torch.Tensor:
    """Combine scenario losses without changing the optimizer scale.

    The candidate grid is a relative clean/degraded scenario prior. Dividing
    by ``1 + weight`` prevents it from also changing AdamW weight decay,
    gradient clipping, and the effective learning-rate scale.
    """

    weight = float(degraded_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("degraded_weight must be finite and non-negative")
    if (
        not isinstance(clean_loss, torch.Tensor)
        or clean_loss.ndim != 0
        or not clean_loss.is_floating_point()
    ):
        raise ValueError("clean_loss must be a floating-point scalar tensor")
    if weight == 0.0:
        return clean_loss
    if (
        not isinstance(degraded_loss, torch.Tensor)
        or degraded_loss.ndim != 0
        or not degraded_loss.is_floating_point()
    ):
        raise ValueError(
            "positive degraded_weight requires a floating-point scalar "
            "degraded_loss"
        )
    if (
        degraded_loss.device != clean_loss.device
        or degraded_loss.dtype != clean_loss.dtype
    ):
        degraded_loss = degraded_loss.to(
            device=clean_loss.device,
            dtype=clean_loss.dtype,
        )
    return (
        clean_loss + weight * degraded_loss
    ) / (1.0 + weight)


def fit_anchored_router(
    model,
    *,
    train_clean: list[CachedExpertBatch],
    train_degraded: list[CachedExpertBatch],
    validation_sources: dict[str, list[CachedExpertBatch]],
    clean_validation_source: str,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit I2 over a pre-registered degradation-weight grid."""

    estimator, router = _require_anchored_modules(model)
    if not train_clean:
        raise ValueError("train_clean must contain at least one cached batch")
    if not isinstance(validation_sources, dict) or not validation_sources:
        raise ValueError("validation_sources must contain named cached sources")
    if clean_validation_source not in validation_sources:
        raise ValueError(
            f"clean validation source {clean_validation_source!r} is not present"
        )
    if any(not cache for cache in validation_sources.values()):
        raise ValueError("every validation source must contain cached batches")
    raw_weights = config.get("degradation_loss_weights", [0.0, 0.1, 0.25, 0.5])
    if not isinstance(raw_weights, (list, tuple)) or not raw_weights:
        raise ValueError("degradation_loss_weights must be a non-empty sequence")
    degradation_weights = [float(value) for value in raw_weights]
    if any(
        not math.isfinite(value) or value < 0.0 for value in degradation_weights
    ):
        raise ValueError(
            "degradation_loss_weights must contain finite non-negative values"
        )
    if len(degradation_weights) != len(set(degradation_weights)):
        raise ValueError("degradation_loss_weights must not contain duplicates")
    if any(value > 0.0 for value in degradation_weights) and not train_degraded:
        raise ValueError(
            "positive degradation_loss_weights require train_degraded batches"
        )
    epochs = int(config.get("epochs", 40))
    patience = int(config.get("patience", 8))
    lr = float(config.get("lr", 5.0e-3))
    weight_decay = float(config.get("weight_decay", 1.0e-4))
    grad_clip = float(config.get("grad_clip", 5.0))
    anchor_kl_weight = float(config.get("clean_anchor_kl_weight", 0.10))
    tolerance = float(config.get("clean_noninferiority_tolerance", 0.003))
    source_tolerance = float(
        config.get("degraded_source_noninferiority_tolerance", 0.0)
    )
    minimum_robust_gain = float(config.get("minimum_robust_gain", 0.0))
    if epochs <= 0 or patience <= 0:
        raise ValueError("router epochs and patience must be positive")
    for name, value in {
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "clean_anchor_kl_weight": anchor_kl_weight,
        "clean_noninferiority_tolerance": tolerance,
        "degraded_source_noninferiority_tolerance": source_tolerance,
        "minimum_robust_gain": minimum_robust_gain,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"router {name} must be finite and non-negative")
    if lr <= 0.0 or grad_clip <= 0.0:
        raise ValueError("router lr and grad_clip must be positive")

    model.zero_grad(set_to_none=True)
    estimator.eval()
    for parameter in estimator.parameters():
        parameter.requires_grad_(False)
    initial_router_state = copy.deepcopy(router.state_dict())
    candidate_summaries: list[dict[str, Any]] = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected_summary: dict[str, Any] | None = None
    selected_tuple: tuple[bool, float, float, float, float] | None = None
    joint_anchor_threshold: dict[str, Any] | None = None
    started = time.perf_counter()
    for degradation_weight in degradation_weights:
        router.load_state_dict(initial_router_state, strict=True)
        router.train()
        for parameter in router.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        best_candidate_state = copy.deepcopy(router.state_dict())
        best_candidate_metrics = evaluate_cached_fusion(
            model,
            validation_sources,
            device,
            clean_source=clean_validation_source,
        )
        current_anchor_threshold = best_candidate_metrics[
            "joint_anchor_classification_threshold"
        ]
        if joint_anchor_threshold is None:
            joint_anchor_threshold = copy.deepcopy(current_anchor_threshold)
        elif not math.isclose(
            float(joint_anchor_threshold["threshold"]),
            float(current_anchor_threshold["threshold"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "Joint-anchor clean threshold changed across I2 candidates"
            )
        best_candidate_selection = _router_selection_state(
            best_candidate_metrics,
            clean_source=clean_validation_source,
            clean_noninferiority_tolerance=tolerance,
            degraded_source_noninferiority_tolerance=source_tolerance,
            minimum_robust_gain=minimum_robust_gain,
        )
        best_candidate_tuple = best_candidate_selection["selection_tuple"]
        best_epoch = 0
        stale = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            router.train()
            total_loss = 0.0
            steps = 0
            max_batches = (
                max(len(train_clean), len(train_degraded))
                if degradation_weight > 0.0
                else len(train_clean)
            )
            for batch_index in range(max_batches):
                optimizer.zero_grad(set_to_none=True)
                clean_batch = train_clean[
                    batch_index % len(train_clean)
                ].to(device)
                clean_loss = _router_training_loss(
                    model,
                    clean_batch,
                    anchor_kl_weight=anchor_kl_weight,
                    is_clean=True,
                )
                degraded_loss = None
                if degradation_weight > 0.0:
                    degraded_batch = train_degraded[
                        batch_index % len(train_degraded)
                    ].to(device)
                    degraded_loss = _router_training_loss(
                        model,
                        degraded_batch,
                        anchor_kl_weight=anchor_kl_weight,
                        is_clean=False,
                    )
                loss = _normalized_clean_degraded_loss(
                    clean_loss,
                    degraded_loss,
                    degraded_weight=degradation_weight,
                )
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise FloatingPointError("Non-finite anchored-router loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    router.parameters(),
                    grad_clip,
                )
                if not bool(torch.isfinite(gradient_norm.detach()).all()):
                    raise FloatingPointError("Non-finite anchored-router gradient")
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
            metrics = evaluate_cached_fusion(
                model,
                validation_sources,
                device,
                clean_source=clean_validation_source,
            )
            selection = _router_selection_state(
                metrics,
                clean_source=clean_validation_source,
                clean_noninferiority_tolerance=tolerance,
                degraded_source_noninferiority_tolerance=source_tolerance,
                minimum_robust_gain=minimum_robust_gain,
            )
            selection_tuple = selection["selection_tuple"]
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(total_loss / max(steps, 1)),
                    "eligible": bool(selection["eligible"]),
                    "clean_noninferior": bool(selection["clean_noninferior"]),
                    "every_degraded_source_noninferior": bool(
                        selection["every_degraded_source_noninferior"]
                    ),
                    "robust_improved": bool(selection["robust_improved"]),
                    "minimum_degraded_source_delta": float(
                        selection["minimum_degraded_source_delta"]
                    ),
                    "degraded_source_deltas": copy.deepcopy(
                        selection["degraded_source_deltas"]
                    ),
                    "robust_mean_macro_f1": float(
                        selection["robust_mean_macro_f1"]
                    ),
                    "robust_joint_mean_macro_f1": float(
                        selection["robust_joint_mean_macro_f1"]
                    ),
                    "robust_mean_gain": float(selection["robust_mean_gain"]),
                    "robust_worst_macro_f1": float(
                        selection["robust_worst_macro_f1"]
                    ),
                    "clean_macro_f1": float(selection["clean_macro_f1"]),
                    "clean_delta": float(selection["clean_delta"]),
                    "mean_nll": float(selection["mean_nll"]),
                    "classification_threshold": float(
                        metrics["classification_threshold"]["threshold"]
                    ),
                    "joint_anchor_classification_threshold": float(
                        metrics["joint_anchor_classification_threshold"][
                            "threshold"
                        ]
                    ),
                }
            )
            if selection_tuple > best_candidate_tuple:
                best_candidate_tuple = selection_tuple
                best_candidate_selection = selection
                best_candidate_metrics = metrics
                best_candidate_state = copy.deepcopy(router.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        router.load_state_dict(best_candidate_state, strict=True)
        candidate = {
            "degradation_loss_weight": float(degradation_weight),
            "best_epoch": int(best_epoch),
            "eligible": bool(best_candidate_selection["eligible"]),
            "clean_noninferior": bool(
                best_candidate_selection["clean_noninferior"]
            ),
            "every_degraded_source_noninferior": bool(
                best_candidate_selection[
                    "every_degraded_source_noninferior"
                ]
            ),
            "robust_improved": bool(best_candidate_selection["robust_improved"]),
            "minimum_degraded_source_delta": float(
                best_candidate_selection["minimum_degraded_source_delta"]
            ),
            "degraded_source_deltas": copy.deepcopy(
                best_candidate_selection["degraded_source_deltas"]
            ),
            "robust_mean_macro_f1": float(
                best_candidate_selection["robust_mean_macro_f1"]
            ),
            "robust_joint_mean_macro_f1": float(
                best_candidate_selection["robust_joint_mean_macro_f1"]
            ),
            "robust_mean_gain": float(
                best_candidate_selection["robust_mean_gain"]
            ),
            "robust_worst_macro_f1": float(
                best_candidate_selection["robust_worst_macro_f1"]
            ),
            "clean_macro_f1": float(best_candidate_selection["clean_macro_f1"]),
            "clean_delta": float(best_candidate_selection["clean_delta"]),
            "mean_nll": float(best_candidate_selection["mean_nll"]),
            "classification_threshold": copy.deepcopy(
                best_candidate_metrics["classification_threshold"]
            ),
            "joint_anchor_classification_threshold": copy.deepcopy(
                best_candidate_metrics["joint_anchor_classification_threshold"]
            ),
            "validation": _fusion_metrics_without_rows(best_candidate_metrics),
            "history": history,
        }
        candidate_summaries.append(candidate)
        if bool(best_candidate_selection["eligible"]) and (
            selected_tuple is None or best_candidate_tuple > selected_tuple
        ):
            selected_tuple = best_candidate_tuple
            selected_state = copy.deepcopy(best_candidate_state)
            selected_summary = candidate

    if selected_state is None:
        # The pre-registered non-inferiority guard fails closed. A dynamic
        # method is not deployed merely because it was fitted.
        router.load_state_dict(initial_router_state, strict=True)
        model.set_anchored_fusion_active(False)
        deployment = "joint_with_uniform_missing_fallback_selection_guard"
        if joint_anchor_threshold is None:
            raise RuntimeError("I2 did not evaluate a Joint-anchor threshold")
        deployed_threshold = copy.deepcopy(joint_anchor_threshold)
        deployed_threshold["prediction_source"] = (
            "joint_anchor_selection_guard_fallback"
        )
    else:
        router.load_state_dict(selected_state, strict=True)
        model.set_anchored_fusion_active(True)
        deployment = "anchored_joint_late"
        if selected_summary is None:
            raise RuntimeError("I2 selected router state has no summary")
        deployed_threshold = copy.deepcopy(
            selected_summary["classification_threshold"]
        )
        deployed_threshold["prediction_source"] = "anchored_joint_late"
    deployed_threshold.update(
        {
            "calibration_split": "val_model_selection",
            "selection_rule": (
                "per_candidate_clean_model_selection_macro_f1_v1"
            ),
            "locked_by_stage_b": True,
            "joint_anchor_threshold": float(
                (joint_anchor_threshold or {})["threshold"]
            ),
        }
    )
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return {
        "deployment": deployment,
        "selected": selected_summary,
        "candidates": candidate_summaries,
        "clean_noninferiority_tolerance": float(tolerance),
        "degraded_source_noninferiority_tolerance": float(source_tolerance),
        "minimum_robust_gain": float(minimum_robust_gain),
        "classification_threshold": deployed_threshold,
        "joint_anchor_classification_threshold": copy.deepcopy(
            joint_anchor_threshold
        ),
        "selection_rule": (
            "eligible_then_min_source_delta_then_mean_delta_then_clean_delta_"
            "then_negative_mean_nll_v1"
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including exact ties."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return ranks


@torch.no_grad()
def competence_diagnostics(
    model,
    sources: dict[str, list[CachedExpertBatch]],
    device: torch.device,
) -> dict[str, Any]:
    estimator, _router = _require_anchored_modules(model)
    estimator.eval()
    if not isinstance(sources, dict) or not sources:
        raise ValueError("competence diagnostics require named sources")
    result_by_source: dict[str, Any] = {}
    for source_name, cache in sources.items():
        if not cache:
            raise ValueError(
                f"competence diagnostics source {source_name!r} is empty"
            )
        predicted: dict[str, list[np.ndarray]] = {
            name: [] for name in EXPERT_NAMES
        }
        targets: dict[str, list[np.ndarray]] = {
            name: [] for name in EXPERT_NAMES
        }
        correctness: dict[str, list[np.ndarray]] = {
            name: [] for name in EXPERT_NAMES
        }
        for cpu_batch in cache:
            batch = cpu_batch.to(device)
            probabilities = _expert_probabilities(batch)
            output = estimator(batch.embeddings, batch.logits, batch.alive)
            target = true_class_probability_targets(probabilities, batch.labels)
            for name in EXPERT_NAMES:
                valid = batch.alive[name]
                if not bool(valid.any()):
                    continue
                predicted[name].append(
                    output.competence[name][valid].detach().cpu().numpy()
                )
                targets[name].append(target[name][valid].detach().cpu().numpy())
                correctness[name].append(
                    probabilities[name][valid]
                    .argmax(dim=-1)
                    .eq(batch.labels[valid])
                    .detach()
                    .cpu()
                    .numpy()
                )
        source_result: dict[str, Any] = {}
        for name in EXPERT_NAMES:
            if not predicted[name]:
                source_result[name] = {"defined": False, "num_rows": 0}
                continue
            q = np.concatenate(predicted[name]).astype(np.float64)
            target = np.concatenate(targets[name]).astype(np.float64)
            correct = np.concatenate(correctness[name]).astype(np.int64)
            pearson = (
                float(np.corrcoef(q, target)[0, 1])
                if q.size >= 2
                and np.std(q) > 0.0
                and np.std(target) > 0.0
                else None
            )
            q_rank = _rank(q)
            target_rank = _rank(target)
            spearman = (
                float(np.corrcoef(q_rank, target_rank)[0, 1])
                if q.size >= 2
                and np.std(q_rank) > 0.0
                and np.std(target_rank) > 0.0
                else None
            )
            error = 1 - correct
            error_auroc = (
                float(roc_auc_score(error, 1.0 - q))
                if np.unique(error).size == 2
                else None
            )
            source_result[name] = {
                "defined": True,
                "num_rows": int(q.size),
                "tcp_mse": float(np.mean((q - target) ** 2)),
                "tcp_mae": float(np.mean(np.abs(q - target))),
                "tcp_pearson": pearson,
                "tcp_spearman": spearman,
                "error_auroc": error_auroc,
            }
        result_by_source[str(source_name)] = source_result
    return result_by_source


__all__ = [
    "CachedExpertBatch",
    "cache_expert_outputs",
    "competence_diagnostics",
    "evaluate_cached_fusion",
    "fit_anchored_router",
    "fit_competence_heads",
]
