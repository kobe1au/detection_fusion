"""Content-conditioned expert competence and anchored dynamic fusion.

This module implements the redesigned main-path I1/I2 contract.  It is kept
independent from the legacy evidential/opinion router so the new method cannot
silently consume perturbation metadata or historical quality proxies.

I1 predicts a continuous true-class-probability (TCP) surrogate for each of
three atomic experts and one joint expert.  Expert embeddings and logits are
detached at the estimator boundary: competence learning cannot alter the
classification experts.

I2 keeps the joint representation as the clean-data anchor and uses a
competence-weighted late fusion of atomic experts as a fallback.  Its public
forward signature accepts only probabilities, competence, and hard
availability masks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


ATOMIC_EXPERT_NAMES = ("api", "graph", "manifest")
EXPERT_NAMES = (*ATOMIC_EXPERT_NAMES, "joint")


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive")
    return math.log(math.expm1(value))


def _require_exact_mapping(
    values: Mapping[str, torch.Tensor],
    *,
    names: tuple[str, ...],
    label: str,
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be an expert-to-tensor mapping")
    missing = [name for name in names if name not in values]
    unknown = sorted(set(values) - set(names))
    if missing or unknown:
        raise ValueError(
            f"{label} must contain exactly {list(names)}; "
            f"missing={missing}, unknown={unknown}"
        )


def _validate_tensor_condition(
    condition: torch.Tensor,
    *,
    message: str,
) -> None:
    """Validate a scalar tensor without serialising the CUDA hot path."""

    if condition.device.type == "cpu":
        if not bool(condition.item()):
            raise ValueError(message)
    else:
        torch._assert_async(condition, message)


def _validate_matrix(
    value: torch.Tensor,
    *,
    label: str,
    width: int | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{label} must be a floating-point tensor")
    if value.ndim != 2 or value.size(0) <= 0 or value.size(1) <= 0:
        raise ValueError(f"{label} must have non-empty shape [B, D]")
    if width is not None and int(value.size(1)) != int(width):
        raise ValueError(
            f"{label} must have width {width}, got shape {tuple(value.shape)}"
        )
    if batch_size is not None and int(value.size(0)) != int(batch_size):
        raise ValueError(
            f"{label} must have batch size {batch_size}, "
            f"got shape {tuple(value.shape)}"
        )
    _validate_tensor_condition(
        torch.isfinite(value).all(),
        message=f"{label} must contain only finite values",
    )
    return value


def _validate_vector(
    value: torch.Tensor,
    *,
    label: str,
    batch_size: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{label} must be a floating-point tensor")
    if value.ndim != 1 or value.numel() <= 0:
        raise ValueError(f"{label} must have non-empty shape [B]")
    if batch_size is not None and int(value.numel()) != int(batch_size):
        raise ValueError(
            f"{label} must have batch size {batch_size}, "
            f"got shape {tuple(value.shape)}"
        )
    _validate_tensor_condition(
        torch.isfinite(value).all(),
        message=f"{label} must contain only finite values",
    )
    return value


def _validate_alive(
    value: torch.Tensor,
    *,
    label: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label} must be a tensor")
    if value.ndim != 1 or int(value.numel()) != int(batch_size):
        raise ValueError(
            f"{label} must have shape [B]=[{batch_size}], "
            f"got {tuple(value.shape)}"
        )
    value = value.to(device=device)
    if value.dtype != torch.bool:
        raise ValueError(
            f"{label} must be boolean; convert verified hard 0/1 availability "
            "once at the data boundary"
        )
    return value


def _validate_probability_matrix(
    value: torch.Tensor,
    *,
    label: str,
    batch_size: int | None = None,
    class_count: int | None = None,
) -> torch.Tensor:
    value = _validate_matrix(
        value,
        label=label,
        width=class_count,
        batch_size=batch_size,
    )
    valid_probability = (
        ((value >= 0.0) & (value <= 1.0)).all()
        & torch.isclose(
            value.sum(dim=-1),
            torch.ones(value.size(0), device=value.device, dtype=value.dtype),
            atol=1.0e-4,
            rtol=1.0e-4,
        ).all()
    )
    _validate_tensor_condition(
        valid_probability,
        message=f"{label} must contain normalized probabilities in [0, 1]",
    )
    return value


class _ExpertCompetenceHead(nn.Module):
    """Small expert-local head over a detached embedding and normalized logits."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        class_count: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        probability_epsilon: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.class_count = int(class_count)
        self.probability_epsilon = float(probability_epsilon)
        self.embedding_normalization = nn.LayerNorm(
            self.embedding_dim,
            elementwise_affine=False,
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, int(projection_dim)),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(int(projection_dim) + self.class_count, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = _validate_matrix(
            embedding,
            label="expert embedding",
            width=self.embedding_dim,
        )
        logits = _validate_matrix(
            logits,
            label="expert logits",
            width=self.class_count,
            batch_size=int(embedding.size(0)),
        )
        if embedding.device != logits.device:
            raise ValueError("expert embedding and logits must share a device")

        # Both tensors are model-state observations.  Detaching here makes the
        # I1 objective unable to improve itself by changing an expert.
        detached_embedding = embedding.detach()
        detached_logits = logits.detach().to(dtype=detached_embedding.dtype)
        normalized_embedding = self.embedding_normalization(detached_embedding)
        # Probabilities retain the complete binary decision state while being
        # invariant to the additive logit offset that leaves the classifier
        # distribution unchanged. L2-normalising raw logits would let an
        # arbitrary head bias change I1 without changing p(y).
        normalized_logits = torch.softmax(detached_logits.float(), dim=-1).to(
            dtype=detached_embedding.dtype
        )
        projected = self.embedding_projection(normalized_embedding)
        competence_logit = self.predictor(
            torch.cat([projected, normalized_logits], dim=-1)
        ).squeeze(-1)
        dtype_epsilon = max(
            self.probability_epsilon,
            float(torch.finfo(competence_logit.dtype).eps),
        )
        competence = torch.sigmoid(competence_logit).clamp(
            dtype_epsilon,
            1.0 - dtype_epsilon,
        )
        return competence, competence_logit


@dataclass(frozen=True)
class CompetenceOutput:
    """I1 outputs.

    ``competence`` is hard-masked to zero for unavailable experts.
    ``unmasked_competence`` remains strictly inside ``(0, 1)`` and is useful
    for calibration diagnostics.  The logits can be used for ranking without
    applying a numerically unstable inverse sigmoid.
    """

    competence: dict[str, torch.Tensor]
    unmasked_competence: dict[str, torch.Tensor]
    competence_logits: dict[str, torch.Tensor]
    alive: dict[str, torch.Tensor]


class ContentConditionedCompetence(nn.Module):
    """Four independent content-conditioned competence estimators.

    The only inputs are current expert embeddings, current expert logits, and
    hard availability.  No extraction totals, perturbation identities,
    perturbation strengths, or peer-expert features are accepted.
    """

    def __init__(
        self,
        embedding_dims: Mapping[str, int],
        *,
        class_count: int = 2,
        projection_dim: int = 32,
        hidden_dim: int = 16,
        dropout: float = 0.10,
        probability_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if not isinstance(embedding_dims, Mapping):
            raise ValueError("embedding_dims must be an expert-to-width mapping")
        missing = [name for name in EXPERT_NAMES if name not in embedding_dims]
        unknown = sorted(set(embedding_dims) - set(EXPERT_NAMES))
        if missing or unknown:
            raise ValueError(
                f"embedding_dims must contain exactly {list(EXPERT_NAMES)}; "
                f"missing={missing}, unknown={unknown}"
            )
        if int(class_count) < 2:
            raise ValueError("class_count must be at least two")
        if int(projection_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("projection_dim and hidden_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie within [0, 1)")
        if (
            not math.isfinite(float(probability_epsilon))
            or float(probability_epsilon) <= 0.0
            or float(probability_epsilon) >= 0.5
        ):
            raise ValueError("probability_epsilon must lie within (0, 0.5)")

        normalized_dims: dict[str, int] = {}
        for name in EXPERT_NAMES:
            width = int(embedding_dims[name])
            if width <= 0:
                raise ValueError(f"embedding_dims[{name!r}] must be positive")
            normalized_dims[name] = width

        self.embedding_dims = normalized_dims
        self.class_count = int(class_count)
        self.heads = nn.ModuleDict(
            {
                name: _ExpertCompetenceHead(
                    embedding_dim=normalized_dims[name],
                    class_count=self.class_count,
                    projection_dim=int(projection_dim),
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    probability_epsilon=float(probability_epsilon),
                )
                for name in EXPERT_NAMES
            }
        )

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        logits: Mapping[str, torch.Tensor],
        alive: Mapping[str, torch.Tensor],
    ) -> CompetenceOutput:
        _require_exact_mapping(embeddings, names=EXPERT_NAMES, label="embeddings")
        _require_exact_mapping(logits, names=EXPERT_NAMES, label="logits")
        _require_exact_mapping(alive, names=EXPERT_NAMES, label="alive")

        competence: dict[str, torch.Tensor] = {}
        unmasked: dict[str, torch.Tensor] = {}
        competence_logits: dict[str, torch.Tensor] = {}
        normalized_alive: dict[str, torch.Tensor] = {}
        expected_batch_size: int | None = None
        expected_device: torch.device | None = None
        for name in EXPERT_NAMES:
            embedding = embeddings[name]
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
                raise ValueError(f"embeddings[{name!r}] must have shape [B, D]")
            if expected_batch_size is None:
                expected_batch_size = int(embedding.size(0))
                expected_device = embedding.device
            elif int(embedding.size(0)) != expected_batch_size:
                raise ValueError("expert embedding batch sizes disagree")
            if embedding.device != expected_device:
                raise ValueError("expert embeddings must share a device")

            raw_competence, raw_logit = self.heads[name](
                embedding,
                logits[name],
            )
            availability = _validate_alive(
                alive[name],
                label=f"alive[{name!r}]",
                batch_size=int(raw_competence.numel()),
                device=raw_competence.device,
            )
            normalized_alive[name] = availability
            unmasked[name] = raw_competence
            competence_logits[name] = raw_logit
            competence[name] = raw_competence * availability.to(raw_competence)

        return CompetenceOutput(
            competence=competence,
            unmasked_competence=unmasked,
            competence_logits=competence_logits,
            alive=normalized_alive,
        )


def true_class_probability_targets(
    expert_probabilities: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return detached continuous TCP targets ``p_expert(y)``.

    Targets are unconditionally detached: allowing the competence objective to
    improve by changing expert probabilities would destroy its calibration
    meaning and violate the Stage-A/Stage-B boundary.
    """

    _require_exact_mapping(
        expert_probabilities,
        names=EXPERT_NAMES,
        label="expert_probabilities",
    )
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a rank-one tensor")
    if labels.is_floating_point() or labels.dtype == torch.bool:
        raise ValueError("labels must use an integer dtype")

    targets: dict[str, torch.Tensor] = {}
    batch_size = int(labels.numel())
    class_count: int | None = None
    for name in EXPERT_NAMES:
        probability = _validate_probability_matrix(
            expert_probabilities[name],
            label=f"expert_probabilities[{name!r}]",
            batch_size=batch_size,
            class_count=class_count,
        )
        if class_count is None:
            class_count = int(probability.size(1))
            if class_count < 2:
                raise ValueError("expert probabilities require at least two classes")
        if labels.device != probability.device:
            raise ValueError("labels and expert probabilities must share a device")
        _validate_tensor_condition(
            ((labels >= 0) & (labels < class_count)).all(),
            message="labels contain an out-of-range class index",
        )
        target = probability.gather(1, labels.long().unsqueeze(1)).squeeze(1)
        targets[name] = target.detach()
    return targets


def _validate_competence_inputs(
    competence: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    alive: Mapping[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    _require_exact_mapping(competence, names=EXPERT_NAMES, label="competence")
    _require_exact_mapping(targets, names=EXPERT_NAMES, label="targets")
    _require_exact_mapping(alive, names=EXPERT_NAMES, label="alive")
    normalized_competence: dict[str, torch.Tensor] = {}
    normalized_targets: dict[str, torch.Tensor] = {}
    normalized_alive: dict[str, torch.Tensor] = {}
    batch_size: int | None = None
    reference_device: torch.device | None = None
    for name in EXPERT_NAMES:
        predicted = _validate_vector(
            competence[name],
            label=f"competence[{name!r}]",
            batch_size=batch_size,
        )
        if batch_size is None:
            batch_size = int(predicted.numel())
            reference_device = predicted.device
        if predicted.device != reference_device:
            raise ValueError("competence tensors must share a device")
        target = _validate_vector(
            targets[name],
            label=f"targets[{name!r}]",
            batch_size=batch_size,
        ).to(device=predicted.device, dtype=predicted.dtype).detach()
        _validate_tensor_condition(
            (
                (predicted >= 0.0)
                & (predicted <= 1.0)
                & (target >= 0.0)
                & (target <= 1.0)
            ).all(),
            message="competence predictions and TCP targets must lie in [0, 1]",
        )
        normalized_competence[name] = predicted
        normalized_targets[name] = target
        normalized_alive[name] = _validate_alive(
            alive[name],
            label=f"alive[{name!r}]",
            batch_size=batch_size,
            device=predicted.device,
        )
    return normalized_competence, normalized_targets, normalized_alive


def atomic_pairwise_ranking_loss(
    competence: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    alive: Mapping[str, torch.Tensor],
    *,
    tie_tolerance: float = 1.0e-6,
    probability_epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank atomic experts by their TCP target on each sample.

    Returns ``(mean_loss, valid_pair_count)``.  Joint is deliberately excluded:
    its role is handled by the explicit joint-vs-late anchor gate.
    """

    if not math.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    if (
        not math.isfinite(float(probability_epsilon))
        or float(probability_epsilon) <= 0.0
        or float(probability_epsilon) >= 0.5
    ):
        raise ValueError("probability_epsilon must lie within (0, 0.5)")
    predicted, target, availability = _validate_competence_inputs(
        competence,
        targets,
        alive,
    )
    reference = predicted[ATOMIC_EXPERT_NAMES[0]]
    loss_sum = reference.sum() * 0.0
    valid_count = torch.zeros((), device=reference.device, dtype=torch.long)
    dtype_epsilon = max(
        float(probability_epsilon),
        float(torch.finfo(reference.dtype).eps),
    )
    for left_index, left in enumerate(ATOMIC_EXPERT_NAMES):
        for right in ATOMIC_EXPERT_NAMES[left_index + 1 :]:
            target_difference = target[left] - target[right]
            valid = (
                availability[left]
                & availability[right]
                & (target_difference.abs() > float(tie_tolerance))
            )
            left_score = torch.logit(
                predicted[left].clamp(
                    dtype_epsilon,
                    1.0 - dtype_epsilon,
                )
            )
            right_score = torch.logit(
                predicted[right].clamp(
                    dtype_epsilon,
                    1.0 - dtype_epsilon,
                )
            )
            direction = target_difference.sign()
            pair_loss = F.softplus(-direction * (left_score - right_score))
            loss_sum = loss_sum + (pair_loss * valid.to(pair_loss)).sum()
            valid_count = valid_count + valid.sum()
    ranking_loss = loss_sum / valid_count.clamp_min(1).to(loss_sum)
    return ranking_loss, valid_count


@dataclass(frozen=True)
class CompetenceLossOutput:
    total: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    valid_expert_rows: torch.Tensor
    valid_atomic_pairs: torch.Tensor


def competence_learning_loss(
    competence: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    alive: Mapping[str, torch.Tensor],
    *,
    regression: str = "mse",
    ranking_weight: float = 0.10,
    tie_tolerance: float = 1.0e-6,
    probability_epsilon: float = 1.0e-6,
) -> CompetenceLossOutput:
    """Fit continuous TCP and optionally its within-sample atomic ordering.

    ``regression='bce'`` uses binary cross entropy with a *soft* TCP target;
    it is not a binary correctness objective.
    """

    regression = str(regression).strip().lower()
    if regression not in {"mse", "bce"}:
        raise ValueError("regression must be 'mse' or 'bce'")
    if not math.isfinite(float(ranking_weight)) or float(ranking_weight) < 0.0:
        raise ValueError("ranking_weight must be finite and non-negative")
    predicted, target, availability = _validate_competence_inputs(
        competence,
        targets,
        alive,
    )

    row_loss_sum = predicted[EXPERT_NAMES[0]].sum() * 0.0
    valid_rows = torch.zeros(
        (),
        device=predicted[EXPERT_NAMES[0]].device,
        dtype=torch.long,
    )
    for name in EXPERT_NAMES:
        valid = availability[name]
        if regression == "mse":
            row_loss = (predicted[name] - target[name]).square()
        else:
            dtype_epsilon = max(
                float(probability_epsilon),
                float(torch.finfo(predicted[name].dtype).eps),
            )
            row_loss = F.binary_cross_entropy(
                predicted[name].clamp(dtype_epsilon, 1.0 - dtype_epsilon),
                target[name],
                reduction="none",
            )
        row_loss_sum = row_loss_sum + (row_loss * valid.to(row_loss)).sum()
        valid_rows = valid_rows + valid.sum()
    regression_loss = row_loss_sum / valid_rows.clamp_min(1).to(row_loss_sum)

    ranking_loss, pair_count = atomic_pairwise_ranking_loss(
        predicted,
        target,
        availability,
        tie_tolerance=tie_tolerance,
        probability_epsilon=probability_epsilon,
    )
    total = regression_loss + float(ranking_weight) * ranking_loss
    return CompetenceLossOutput(
        total=total,
        regression=regression_loss,
        ranking=ranking_loss,
        valid_expert_rows=valid_rows,
        valid_atomic_pairs=pair_count,
    )


@dataclass(frozen=True)
class AnchoredFusionOutput:
    """Final probability and interpretable I2 diagnostics.

    ``joint_probability`` is the exact safe anchor used by deployment: the
    real Joint expert when it is alive, an alive-uniform atomic mixture when
    Joint is unavailable, and a uniform class distribution when all experts
    are dead.
    """

    probability: torch.Tensor
    joint_probability: torch.Tensor
    uniform_atomic_probability: torch.Tensor
    late_probability: torch.Tensor
    atomic_weights: torch.Tensor
    atomic_scores: torch.Tensor
    atomic_competence: torch.Tensor
    joint_competence: torch.Tensor
    late_competence: torch.Tensor
    late_gate: torch.Tensor
    has_atomic: torch.Tensor
    has_joint: torch.Tensor
    all_dead: torch.Tensor

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            "uniform_atomic_probability": self.uniform_atomic_probability,
            "atomic_weights": self.atomic_weights,
            "atomic_scores": self.atomic_scores,
            "atomic_competence": self.atomic_competence,
            "joint_competence": self.joint_competence,
            "late_competence": self.late_competence,
            "late_gate": self.late_gate,
            "has_atomic": self.has_atomic,
            "has_joint": self.has_joint,
            "all_dead": self.all_dead,
        }


class AnchoredCompetenceFusion(nn.Module):
    """Joint anchor with a competence-weighted atomic late fallback.

    Atomic routing follows ``softmax(relative_bias + scale * log(q_m))`` over
    alive experts.  The final gate is monotone in
    ``log(q_late) - log(q_joint)`` with a strictly positive scale.

    All expert probabilities and competence values are unconditionally
    detached. Thus I2 NLL can train only the small routing parameters; expert
    heads stay frozen and I1 keeps its continuous-TCP semantics.
    """

    def __init__(
        self,
        *,
        initial_atomic_competence_scale: float = 1.0,
        initial_joint_late_scale: float = 1.0,
        initial_late_gate: float = 0.10,
        probability_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < float(initial_late_gate) < 1.0:
            raise ValueError("initial_late_gate must lie within (0, 1)")
        if (
            not math.isfinite(float(probability_epsilon))
            or float(probability_epsilon) <= 0.0
            or float(probability_epsilon) >= 0.5
        ):
            raise ValueError("probability_epsilon must lie within (0, 0.5)")
        self.probability_epsilon = float(probability_epsilon)
        self.atomic_relative_bias = nn.Parameter(
            torch.zeros(len(ATOMIC_EXPERT_NAMES), dtype=torch.float32)
        )
        self.raw_atomic_competence_scale = nn.Parameter(
            torch.tensor(
                _inverse_softplus(float(initial_atomic_competence_scale)),
                dtype=torch.float32,
            )
        )
        self.raw_joint_late_scale = nn.Parameter(
            torch.tensor(
                _inverse_softplus(float(initial_joint_late_scale)),
                dtype=torch.float32,
            )
        )
        gate_logit = math.log(
            float(initial_late_gate) / (1.0 - float(initial_late_gate))
        )
        self.joint_late_bias = nn.Parameter(
            torch.tensor(gate_logit, dtype=torch.float32)
        )

    def effective_atomic_competence_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_atomic_competence_scale).clamp_min(
            self.probability_epsilon
        )

    def effective_joint_late_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_joint_late_scale).clamp_min(
            self.probability_epsilon
        )

    def forward(
        self,
        expert_probabilities: Mapping[str, torch.Tensor],
        competence: Mapping[str, torch.Tensor],
        alive: Mapping[str, torch.Tensor],
    ) -> AnchoredFusionOutput:
        _require_exact_mapping(
            expert_probabilities,
            names=EXPERT_NAMES,
            label="expert_probabilities",
        )
        _require_exact_mapping(competence, names=EXPERT_NAMES, label="competence")
        _require_exact_mapping(alive, names=EXPERT_NAMES, label="alive")

        probabilities: dict[str, torch.Tensor] = {}
        competence_values: dict[str, torch.Tensor] = {}
        availability: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        class_count: int | None = None
        reference_device: torch.device | None = None
        reference_dtype: torch.dtype | None = None
        for name in EXPERT_NAMES:
            probability = _validate_probability_matrix(
                expert_probabilities[name],
                label=f"expert_probabilities[{name!r}]",
                batch_size=batch_size,
                class_count=class_count,
            )
            if batch_size is None:
                batch_size = int(probability.size(0))
                class_count = int(probability.size(1))
                reference_device = probability.device
                reference_dtype = probability.dtype
            if probability.device != reference_device:
                raise ValueError("expert probabilities must share a device")
            if probability.dtype != reference_dtype:
                probability = probability.to(dtype=reference_dtype)
            value = _validate_vector(
                competence[name],
                label=f"competence[{name!r}]",
                batch_size=batch_size,
            ).to(device=reference_device, dtype=reference_dtype)
            mask = _validate_alive(
                alive[name],
                label=f"alive[{name!r}]",
                batch_size=batch_size,
                device=reference_device,
            )
            probabilities[name] = probability.detach()
            competence_values[name] = value.detach()
            availability[name] = mask

        atomic_probability = torch.stack(
            [probabilities[name] for name in ATOMIC_EXPERT_NAMES],
            dim=1,
        )
        atomic_competence = torch.stack(
            [competence_values[name] for name in ATOMIC_EXPERT_NAMES],
            dim=1,
        )
        atomic_alive = torch.stack(
            [availability[name] for name in ATOMIC_EXPERT_NAMES],
            dim=1,
        )
        atomic_competence = atomic_competence * atomic_alive.to(atomic_competence)
        dtype_epsilon = max(
            self.probability_epsilon,
            float(torch.finfo(atomic_probability.dtype).eps),
        )
        centered_bias = self.atomic_relative_bias - self.atomic_relative_bias.mean()
        atomic_scores = centered_bias.to(
            device=atomic_probability.device,
            dtype=atomic_probability.dtype,
        ).unsqueeze(0) + self.effective_atomic_competence_scale().to(
            device=atomic_probability.device,
            dtype=atomic_probability.dtype,
        ) * torch.log(
            atomic_competence.clamp(dtype_epsilon, 1.0 - dtype_epsilon)
        )
        score_floor = torch.finfo(atomic_scores.dtype).min
        masked_scores = atomic_scores.masked_fill(~atomic_alive, score_floor)
        atomic_weights = torch.softmax(masked_scores, dim=-1)
        atomic_weights = atomic_weights * atomic_alive.to(atomic_weights)
        has_atomic = atomic_alive.any(dim=-1)
        atomic_weights = atomic_weights / atomic_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(dtype_epsilon)

        uniform_probability = torch.full(
            (batch_size, class_count),
            1.0 / float(class_count),
            device=atomic_probability.device,
            dtype=atomic_probability.dtype,
        )
        uniform_atomic_weights = atomic_alive.to(atomic_probability) / (
            atomic_alive.sum(dim=-1, keepdim=True)
            .clamp_min(1)
            .to(atomic_probability)
        )
        uniform_atomic_probability = (
            uniform_atomic_weights.unsqueeze(-1) * atomic_probability
        ).sum(dim=1)
        uniform_atomic_probability = torch.where(
            has_atomic.unsqueeze(-1),
            uniform_atomic_probability,
            uniform_probability,
        )
        late_probability = (
            atomic_weights.unsqueeze(-1) * atomic_probability
        ).sum(dim=1)
        late_probability = torch.where(
            has_atomic.unsqueeze(-1),
            late_probability,
            uniform_probability,
        )
        late_competence = (atomic_weights * atomic_competence).sum(dim=-1)
        late_competence = torch.where(
            has_atomic,
            late_competence,
            torch.zeros_like(late_competence),
        )

        has_joint = availability["joint"]
        joint_probability = torch.where(
            has_joint.unsqueeze(-1),
            probabilities["joint"],
            uniform_atomic_probability,
        )
        joint_competence = (
            competence_values["joint"] * has_joint.to(competence_values["joint"])
        )
        both_available = has_atomic & has_joint
        gate_logit = self.joint_late_bias.to(
            device=atomic_probability.device,
            dtype=atomic_probability.dtype,
        ) + self.effective_joint_late_scale().to(
            device=atomic_probability.device,
            dtype=atomic_probability.dtype,
        ) * (
            torch.log(late_competence.clamp(dtype_epsilon, 1.0))
            - torch.log(joint_competence.clamp(dtype_epsilon, 1.0))
        )
        learned_late_gate = torch.sigmoid(gate_logit)
        late_gate = torch.where(
            both_available,
            learned_late_gate,
            torch.where(
                has_atomic,
                torch.ones_like(learned_late_gate),
                torch.zeros_like(learned_late_gate),
            ),
        )
        probability = (
            (1.0 - late_gate).unsqueeze(-1) * joint_probability
            + late_gate.unsqueeze(-1) * late_probability
        )
        all_dead = ~(has_atomic | has_joint)
        probability = torch.where(
            all_dead.unsqueeze(-1),
            uniform_probability,
            probability,
        )
        _validate_tensor_condition(
            (
                torch.isfinite(probability).all()
                & torch.isfinite(atomic_weights).all()
                & torch.isfinite(late_gate).all()
                & torch.isclose(
                    probability.sum(dim=-1),
                    torch.ones(
                        probability.size(0),
                        device=probability.device,
                        dtype=probability.dtype,
                    ),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                ).all()
            ),
            message="anchored fusion produced invalid probabilities or diagnostics",
        )

        return AnchoredFusionOutput(
            probability=probability,
            joint_probability=joint_probability,
            uniform_atomic_probability=uniform_atomic_probability,
            late_probability=late_probability,
            atomic_weights=atomic_weights,
            atomic_scores=atomic_scores,
            atomic_competence=atomic_competence,
            joint_competence=joint_competence,
            late_competence=late_competence,
            late_gate=late_gate,
            has_atomic=has_atomic,
            has_joint=has_joint,
            all_dead=all_dead,
        )


__all__ = [
    "ATOMIC_EXPERT_NAMES",
    "EXPERT_NAMES",
    "AnchoredCompetenceFusion",
    "AnchoredFusionOutput",
    "CompetenceLossOutput",
    "CompetenceOutput",
    "ContentConditionedCompetence",
    "atomic_pairwise_ranking_loss",
    "competence_learning_loss",
    "true_class_probability_targets",
]
