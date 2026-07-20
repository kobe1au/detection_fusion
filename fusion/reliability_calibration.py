from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import EvidenceIndex


BRANCH_NAMES = ("api", "graph", "manifest")

MONOTONIC_CORRECTNESS_METHOD = "monotonic_correctness"
TEMPERATURE_SCALING_CONFIDENCE_METHOD = "temperature_scaling_confidence"
RELIABILITY_CALIBRATION_METHODS = (
    MONOTONIC_CORRECTNESS_METHOD,
    TEMPERATURE_SCALING_CONFIDENCE_METHOD,
)

# Keep the operative I1 topology invariant across atomic feature ablations.
# Optional signals are zero-masked instead of changing a branch module's
# parameterization.  The design deliberately separates the clean-competence
# variables from degradation-only variables; the latter can only subtract from
# the clean log-odds inside ``MonotonicBranchCalibrator``.
RELIABILITY_FEATURE_LAYOUT = {
    "api": (
        "effective_quality_deficit",
        "embedding_tail_q50",
        "embedding_tail_q80",
        "embedding_tail_q95",
        "prediction_margin",
        "predicted_malware_indicator",
    ),
    "graph": (
        "effective_quality_deficit",
        "embedding_tail_q50",
        "embedding_tail_q80",
        "embedding_tail_q95",
        "prediction_margin",
        "predicted_malware_indicator",
    ),
    "manifest": (
        "effective_quality_deficit",
        "embedding_tail_q50",
        "embedding_tail_q80",
        "embedding_tail_q95",
        "prediction_margin",
        "predicted_malware_indicator",
    ),
}

MARGIN_HINGE_KNOTS = (0.50, 0.80)
EMBEDDING_TAIL_QUANTILES = (0.50, 0.80, 0.95)


class ClassConditionalEmbeddingDensity(nn.Module):
    """Fold-local diagonal-Mahalanobis reference for one branch embedding.

    The reference is fitted only on clean rows from the current I1 training
    fold.  At inference the branch's own predicted class selects a clean class
    reference; no perturbation name, severity, other modality, or target label
    is consumed.  A diagonal shrinkage estimate is deliberately used because
    each post-hoc fold contains only a few hundred independent packages.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_classes: int = 2,
        variance_shrinkage: float = 0.10,
        reference_quantile: float = 0.95,
        tail_quantiles: tuple[float, ...] = EMBEDDING_TAIL_QUANTILES,
        min_class_samples: int = 8,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.variance_shrinkage = float(variance_shrinkage)
        self.reference_quantile = float(reference_quantile)
        self.tail_quantiles = tuple(float(value) for value in tail_quantiles)
        self.min_class_samples = int(min_class_samples)
        self.eps = float(eps)
        if self.embedding_dim <= 0:
            raise ValueError("embedding density requires embedding_dim > 0")
        if self.num_classes < 2:
            raise ValueError("embedding density requires at least two classes")
        if not 0.0 <= self.variance_shrinkage <= 1.0:
            raise ValueError("embedding density variance_shrinkage must be within [0, 1]")
        if not 0.5 <= self.reference_quantile < 1.0:
            raise ValueError("embedding density reference_quantile must be within [0.5, 1)")
        if (
            not self.tail_quantiles
            or any(not 0.0 < value < 1.0 for value in self.tail_quantiles)
            or tuple(sorted(set(self.tail_quantiles))) != self.tail_quantiles
        ):
            raise ValueError(
                "embedding density tail_quantiles must be unique, increasing, "
                "and lie within (0, 1)"
            )
        if self.min_class_samples < 2:
            raise ValueError("embedding density min_class_samples must be >= 2")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("embedding density eps must be finite and positive")

        self.register_buffer(
            "class_mean",
            torch.zeros(self.num_classes, self.embedding_dim),
        )
        self.register_buffer(
            "class_variance",
            torch.ones(self.num_classes, self.embedding_dim),
        )
        self.register_buffer(
            "distance_scale",
            torch.ones(self.num_classes),
        )
        self.register_buffer(
            "distance_quantiles",
            torch.ones(self.num_classes, len(self.tail_quantiles)),
        )
        self.register_buffer(
            "class_count",
            torch.zeros(self.num_classes, dtype=torch.long),
        )
        self.register_buffer("reference_fitted", torch.tensor(False, dtype=torch.bool))
        self._reference_fitted_shadow = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Loading is cold-path; mirror the persistent checkpoint flag once so
        # every inference batch does not synchronize a CUDA scalar via item().
        self._reference_fitted_shadow = bool(
            self.reference_fitted.detach().cpu().item()
        )

    def _validate_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        require_finite: bool,
    ) -> torch.Tensor:
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("embedding density requires a tensor embedding")
        if embeddings.ndim != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                "embedding density expected [B, "
                f"{self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        embeddings = embeddings.detach().float()
        if require_finite and not bool(torch.isfinite(embeddings).all().item()):
            raise ValueError("embedding density received non-finite embeddings")
        return embeddings

    @torch.no_grad()
    def fit(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        alive: torch.Tensor,
    ) -> None:
        embeddings = self._validate_embeddings(embeddings, require_finite=True)
        labels = labels.detach().long().view(-1).to(device=embeddings.device)
        alive = alive.detach().bool().view(-1).to(device=embeddings.device)
        if labels.numel() != embeddings.size(0) or alive.numel() != embeddings.size(0):
            raise ValueError("embedding density fit rows disagree")
        if bool(((labels < 0) | (labels >= self.num_classes)).any().item()):
            raise ValueError("embedding density labels are outside the configured classes")
        # Reference fitting is cold-path and uses float64 statistics so finite
        # but large FP32 encoder values cannot overflow while squaring.
        statistics_embeddings = embeddings.double()
        available_embeddings = statistics_embeddings[alive]
        if available_embeddings.size(0) < self.num_classes * self.min_class_samples:
            raise ValueError("embedding density has too few available clean samples")

        global_variance = available_embeddings.var(dim=0, unbiased=False).clamp_min(
            self.eps
        )
        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        quantiles: list[torch.Tensor] = []
        counts: list[int] = []
        for class_index in range(self.num_classes):
            selected = alive & labels.eq(class_index)
            class_embeddings = statistics_embeddings[selected]
            count = int(class_embeddings.size(0))
            if count < self.min_class_samples:
                raise ValueError(
                    "embedding density clean reference class "
                    f"{class_index} has {count} samples; requires "
                    f"{self.min_class_samples}"
                )
            mean = class_embeddings.mean(dim=0)
            empirical_variance = class_embeddings.var(dim=0, unbiased=False)
            variance = (
                (1.0 - self.variance_shrinkage) * empirical_variance
                + self.variance_shrinkage * global_variance
            ).clamp_min(self.eps)
            distance = ((class_embeddings - mean).square() / variance).mean(dim=-1)
            scale = torch.quantile(distance, self.reference_quantile).clamp_min(
                self.eps
            )
            tail_quantiles = torch.quantile(
                distance,
                torch.tensor(
                    self.tail_quantiles,
                    device=distance.device,
                    dtype=distance.dtype,
                ),
            ).clamp_min(0.0)
            means.append(mean)
            variances.append(variance)
            scales.append(scale)
            quantiles.append(tail_quantiles)
            counts.append(count)

        fitted_tensors = [*means, *variances, *scales, *quantiles]
        if any(
            not bool(torch.isfinite(value).all().item())
            for value in fitted_tensors
        ):
            raise ValueError("embedding density fitted non-finite reference statistics")

        self.class_mean.copy_(torch.stack(means).to(self.class_mean))
        self.class_variance.copy_(torch.stack(variances).to(self.class_variance))
        self.distance_scale.copy_(torch.stack(scales).to(self.distance_scale))
        self.distance_quantiles.copy_(
            torch.stack(quantiles).to(self.distance_quantiles)
        )
        self.class_count.copy_(
            torch.tensor(counts, device=self.class_count.device, dtype=torch.long)
        )
        self.reference_fitted.fill_(True)
        self._reference_fitted_shadow = True

    def forward(
        self,
        embeddings: torch.Tensor,
        predicted_class: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder outputs are validated when each clean reference is fitted.
        # Avoid a device synchronization for every inference branch/batch;
        # pathological query distances are conservatively mapped to the
        # maximum finite distance below.
        embeddings = self._validate_embeddings(embeddings, require_finite=False)
        if not self._reference_fitted_shadow:
            raise RuntimeError(
                "embedding density reference is not fitted; fit I1 before inference"
            )
        predicted_class = predicted_class.detach().long().view(-1).to(
            device=embeddings.device
        )
        if predicted_class.numel() != embeddings.size(0):
            raise ValueError("embedding density predicted-class rows disagree")
        mean = self.class_mean.to(embeddings).index_select(0, predicted_class)
        variance = self.class_variance.to(embeddings).index_select(
            0, predicted_class
        )
        scale = self.distance_scale.to(embeddings).index_select(
            0, predicted_class
        )
        distance = ((embeddings - mean).square() / variance).mean(dim=-1)
        distance = torch.nan_to_num(
            distance,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=0.0,
        ).clamp(0.0, 1.0e6)
        score = torch.exp(-math.log(2.0) * distance / scale.clamp_min(self.eps))
        return score.clamp(0.0, 1.0), distance

    def tail_basis(
        self,
        distance: torch.Tensor,
        predicted_class: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded monotone excess-distance bases for clean quantiles.

        Every column is exactly zero below its class-conditional clean
        quantile, then increases smoothly toward one.  The bases are fixed
        fold-local design variables; only their non-negative penalty weights
        are learned later.
        """

        if not self._reference_fitted_shadow:
            raise RuntimeError(
                "embedding density reference is not fitted; fit I1 before inference"
            )
        distance = distance.detach().float().view(-1)
        predicted_class = predicted_class.detach().long().view(-1).to(
            device=distance.device
        )
        if distance.numel() != predicted_class.numel():
            raise ValueError("embedding tail distance and predicted-class rows disagree")
        thresholds = self.distance_quantiles.to(distance).index_select(
            0, predicted_class
        )
        scale = self.distance_scale.to(distance).index_select(
            0, predicted_class
        ).clamp_min(self.eps)
        excess = (distance.unsqueeze(-1) - thresholds).clamp_min(0.0)
        basis = -torch.expm1(-excess / scale.unsqueeze(-1))
        return torch.nan_to_num(basis, nan=1.0, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )


def _column(evidence: torch.Tensor, index: int) -> torch.Tensor:
    return evidence[:, index].clamp(0.0, 1.0)


def normalize_reliability_calibration_method(value: str) -> str:
    """Return the canonical I1 estimator identity.

    Temperature scaling is deliberately a separate confidence baseline, not a
    feature switch inside the proposed monotonic correctness calibrator.
    """

    method = str(value).strip().lower()
    aliases = {
        "monotonic": MONOTONIC_CORRECTNESS_METHOD,
        "learned_correctness": MONOTONIC_CORRECTNESS_METHOD,
        "temperature": TEMPERATURE_SCALING_CONFIDENCE_METHOD,
        "temperature_scaling": TEMPERATURE_SCALING_CONFIDENCE_METHOD,
        "branch_temperature_scaling": TEMPERATURE_SCALING_CONFIDENCE_METHOD,
    }
    method = aliases.get(method, method)
    if method not in RELIABILITY_CALIBRATION_METHODS:
        raise ValueError(
            "reliability_calibration.method must be one of "
            f"{list(RELIABILITY_CALIBRATION_METHODS)}, got {value!r}"
        )
    return method


def _normalize_branch_vector(
    value: torch.Tensor,
    *,
    branch: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"branch_probabilities[{branch!r}] must be a tensor")
    probability = value.to(device=device, dtype=dtype)
    if (
        probability.ndim != 2
        or probability.size(0) != batch_size
        or probability.size(1) < 2
    ):
        raise ValueError(
            f"branch_probabilities[{branch!r}] must have shape [B, C] with C >= 2; "
            f"got {tuple(probability.shape)} for B={batch_size}"
        )
    if not bool(torch.isfinite(probability).all().item()):
        raise ValueError(
            f"branch_probabilities[{branch!r}] contains non-finite values"
        )
    if bool((probability < 0.0).any().item()):
        raise ValueError(f"branch_probabilities[{branch!r}] contains negative values")
    total = probability.sum(dim=-1, keepdim=True)
    if bool((total <= 0.0).any().item()):
        raise ValueError(f"branch_probabilities[{branch!r}] has a zero-mass row")
    return probability / total


def build_branch_prediction_features(
    branch_probabilities: dict[str, torch.Tensor],
    *,
    branch_logits: dict[str, torch.Tensor] | None = None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build intrinsic prediction features for correctness calibration.

    ``prediction_margin`` is the top-1 minus top-2 normalized probability and
    is therefore in ``[0, 1]``. ``predicted_malware_indicator`` is one exactly
    when class index 1 is the branch argmax.  The latter occupies a signed slot
    in the monotone network, giving one shared calibrator a learned
    predicted-class-conditional offset while quality, margin and evidence
    strength remain positive-monotone.
    """

    if not isinstance(branch_probabilities, dict):
        raise ValueError("I1 requires branch_probabilities to be a mapping")
    missing = [
        name for name in BRANCH_NAMES if branch_probabilities.get(name) is None
    ]
    if missing:
        raise ValueError(
            "I1 branch_probabilities is missing required branches: "
            f"{missing}"
        )

    margins: dict[str, torch.Tensor] = {}
    predicted_malware: dict[str, torch.Tensor] = {}
    for name in BRANCH_NAMES:
        probability = _normalize_branch_vector(
            branch_probabilities[name],
            branch=name,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        top_two = probability.topk(k=2, dim=-1).values
        margins[name] = (top_two[:, 0] - top_two[:, 1]).clamp(0.0, 1.0)
        logits = None if branch_logits is None else branch_logits.get(name)
        if logits is not None:
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.size(0) != batch_size
                or logits.size(1) != probability.size(1)
            ):
                raise ValueError(
                    f"branch_logits[{name!r}] must match the corresponding "
                    f"probability shape {tuple(probability.shape)}"
                )
            predicted_class = logits.to(device=device).argmax(dim=-1)
        else:
            predicted_class = probability.argmax(dim=-1)
        predicted_malware[name] = predicted_class.eq(1).to(dtype=dtype)
    return margins, predicted_malware


class PositiveLinear(nn.Module):
    """Linear layer with positive monotone weights and optional signed inputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        signed_input_indices: tuple[int, ...] = (),
    ):
        super().__init__()
        invalid = [index for index in signed_input_indices if not 0 <= index < in_features]
        if invalid:
            raise ValueError(f"signed input indices out of range: {invalid}")
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        signed_mask = torch.zeros((1, in_features), dtype=torch.bool)
        if signed_input_indices:
            signed_mask[:, list(signed_input_indices)] = True
        self.register_buffer("signed_input_mask", signed_mask, persistent=False)
        nn.init.normal_(self.raw_weight, mean=-1.0, std=0.10)
        # Preserve identical RNG consumption across fixed-topology ablations,
        # but give the categorical predicted-class offsets a neutral prior.
        if signed_input_indices:
            with torch.no_grad():
                self.raw_weight[:, list(signed_input_indices)] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive_weight = F.softplus(self.raw_weight)
        weight = torch.where(
            self.signed_input_mask,
            self.raw_weight,
            positive_weight,
        )
        return F.linear(x, weight, self.bias)


def monotone_hinge_basis(
    value: torch.Tensor,
    knots: tuple[float, ...],
) -> torch.Tensor:
    """Build a bounded low-dimensional monotone hinge design on ``[0, 1]``."""

    value = value.view(-1).clamp(0.0, 1.0)
    columns = [value]
    for knot in knots:
        knot = float(knot)
        if not 0.0 < knot < 1.0:
            raise ValueError("monotone hinge knots must lie within (0, 1)")
        columns.append((value - knot).clamp_min(0.0) / (1.0 - knot))
    return torch.stack(columns, dim=-1)


class CleanCompetenceHead(nn.Module):
    """Clean correctness log-odds from margin and a predicted-class offset."""

    def __init__(
        self,
        *,
        margin_hinge_knots: tuple[float, ...] = MARGIN_HINGE_KNOTS,
    ) -> None:
        super().__init__()
        self.margin_hinge_knots = tuple(float(value) for value in margin_hinge_knots)
        if tuple(sorted(set(self.margin_hinge_knots))) != self.margin_hinge_knots:
            raise ValueError("clean-competence margin knots must be unique and increasing")
        self.raw_margin_weights = nn.Parameter(
            torch.full((1 + len(self.margin_hinge_knots),), -1.0)
        )
        self.predicted_class_weight = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def effective_margin_weights(self) -> torch.Tensor:
        return F.softplus(self.raw_margin_weights)

    def design(
        self,
        margin: torch.Tensor,
        predicted_malware: torch.Tensor,
    ) -> torch.Tensor:
        margin_basis = monotone_hinge_basis(margin, self.margin_hinge_knots)
        predicted_malware = predicted_malware.view(-1).to(
            device=margin_basis.device,
            dtype=margin_basis.dtype,
        )
        if predicted_malware.numel() != margin_basis.size(0):
            raise ValueError("clean-competence margin and predicted-class rows disagree")
        return torch.cat([margin_basis, predicted_malware.unsqueeze(-1)], dim=-1)

    def forward_logit(
        self,
        margin: torch.Tensor,
        predicted_malware: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        design = self.design(margin, predicted_malware)
        margin_width = self.raw_margin_weights.numel()
        logit = (
            self.bias.to(design)
            + (design[:, :margin_width] * self.effective_margin_weights().to(design)).sum(
                dim=-1
            )
            + design[:, margin_width] * self.predicted_class_weight.to(design)
        )
        return logit, design


class NonnegativeDegradationPenalty(nn.Module):
    """Bias-free branch-local degradation penalty in clean log-odds space."""

    def __init__(self, *, num_tail_bases: int) -> None:
        super().__init__()
        self.num_tail_bases = int(num_tail_bases)
        if self.num_tail_bases <= 0:
            raise ValueError("degradation penalty requires at least one tail basis")
        # A near-zero prior lets unsupported degradation mechanisms remain
        # effectively inactive while avoiding a saturated softplus gradient.
        self.raw_quality_weight = nn.Parameter(torch.tensor(-3.0))
        self.raw_tail_weights = nn.Parameter(
            torch.full((self.num_tail_bases,), -3.0)
        )
        self.raw_high_confidence_ood_weight = nn.Parameter(torch.tensor(-3.0))

    def effective_weights(self) -> dict[str, torch.Tensor]:
        return {
            "quality": F.softplus(self.raw_quality_weight),
            "tail": F.softplus(self.raw_tail_weights),
            "high_confidence_ood": F.softplus(
                self.raw_high_confidence_ood_weight
            ),
        }

    def forward(
        self,
        quality_deficit: torch.Tensor,
        tail_basis: torch.Tensor,
        clean_competence: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        quality_deficit = quality_deficit.view(-1).clamp(0.0, 1.0)
        if tail_basis.ndim != 2 or tail_basis.size(1) != self.num_tail_bases:
            raise ValueError(
                "degradation tail basis must have shape [B, "
                f"{self.num_tail_bases}], got {tuple(tail_basis.shape)}"
            )
        tail_basis = tail_basis.to(
            device=quality_deficit.device,
            dtype=quality_deficit.dtype,
        ).clamp(0.0, 1.0)
        if tail_basis.size(0) != quality_deficit.numel():
            raise ValueError("degradation quality and tail rows disagree")
        clean_competence = clean_competence.detach().view(-1).to(
            device=quality_deficit.device,
            dtype=quality_deficit.dtype,
        ).clamp(0.0, 1.0)
        if clean_competence.numel() != quality_deficit.numel():
            raise ValueError("degradation competence and quality rows disagree")

        # The q95 excess is the genuinely out-of-reference component.  Using
        # it for the interaction prevents ordinary clean-density variation from
        # being mislabeled as a high-confidence OOD event.
        high_confidence_ood = tail_basis[:, -1] * clean_competence
        design = torch.cat(
            [
                quality_deficit.unsqueeze(-1),
                tail_basis,
                high_confidence_ood.unsqueeze(-1),
            ],
            dim=-1,
        )
        weights = self.effective_weights()
        quality_penalty = quality_deficit * weights["quality"].to(quality_deficit)
        tail_penalty = (
            tail_basis * weights["tail"].to(tail_basis).view(1, -1)
        ).sum(dim=-1)
        interaction_penalty = high_confidence_ood * weights[
            "high_confidence_ood"
        ].to(high_confidence_ood)
        total = quality_penalty + tail_penalty + interaction_penalty
        return total, {
            "design": design,
            "quality": quality_penalty,
            "tail": tail_penalty,
            "high_confidence_ood": interaction_penalty,
        }


class MonotonicBranchCalibrator(nn.Module):
    """Clean competence minus a non-negative degradation penalty."""

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        signed_input_indices: tuple[int, ...] = (),
        margin_hinge_knots: tuple[float, ...] = MARGIN_HINGE_KNOTS,
        num_tail_bases: int = len(EMBEDDING_TAIL_QUANTILES),
    ):
        super().__init__()
        expected_width = 2 + int(num_tail_bases) + 1
        if input_dim is not None and int(input_dim) != expected_width:
            raise ValueError(
                f"branch reliability design width must be {expected_width}, got {input_dim}"
            )
        if signed_input_indices and signed_input_indices != (expected_width - 1,):
            raise ValueError(
                "the only signed branch reliability input is predicted class"
            )
        self.num_tail_bases = int(num_tail_bases)
        self.competence = CleanCompetenceHead(
            margin_hinge_knots=margin_hinge_knots
        )
        self.degradation = NonnegativeDegradationPenalty(
            num_tail_bases=self.num_tail_bases
        )

    def competence_parameters(self) -> list[nn.Parameter]:
        return list(self.competence.parameters())

    def degradation_parameters(self) -> list[nn.Parameter]:
        return list(self.degradation.parameters())

    def forward_components(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        expected_width = 2 + self.num_tail_bases + 1
        if features.ndim != 2 or features.size(1) != expected_width:
            raise ValueError(
                f"branch reliability features must have shape [B, {expected_width}], "
                f"got {tuple(features.shape)}"
            )
        quality_deficit = features[:, 0]
        tail_basis = features[:, 1 : 1 + self.num_tail_bases]
        margin = features[:, 1 + self.num_tail_bases]
        predicted_malware = features[:, 2 + self.num_tail_bases]
        clean_logit, competence_design = self.competence.forward_logit(
            margin, predicted_malware
        )
        clean_competence = torch.sigmoid(clean_logit)
        penalty, penalty_parts = self.degradation(
            quality_deficit,
            tail_basis,
            clean_competence,
        )
        reliability_logit = clean_logit - penalty
        return {
            "clean_competence_logit": clean_logit,
            "clean_competence": clean_competence,
            "degradation_penalty": penalty,
            "degradation_quality_penalty": penalty_parts["quality"],
            "degradation_tail_penalty": penalty_parts["tail"],
            "degradation_high_confidence_ood_penalty": penalty_parts[
                "high_confidence_ood"
            ],
            "competence_design": competence_design,
            "degradation_design": penalty_parts["design"],
            "reliability_logit": reliability_logit,
        }

    def forward_logit(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features)["reliability_logit"]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logit(features))


class BranchTemperatureScalingConfidenceCalibrator(nn.Module):
    """Per-branch temperature-scaling confidence baseline for I1.

    Each branch owns one positive scalar temperature fitted with multiclass
    NLL on its configured, identity-grouped out-of-fold branch logits. The
    canonical clean-only cell uses clean logits; the primary matched-budget
    comparator additionally includes exactly the branch-local observable-
    degradation views used by the proposed I1. At inference the I1 value is the
    conventional maximum
    temperature-scaled softmax probability,
    ``r_m=max softmax(z_m / T_m)``.  The module intentionally consumes neither
    observable quality features nor cross-modal signals; it is the simple
    calibration baseline against which the proposed feature-conditioned
    correctness estimator is compared.
    """

    _MIN_LOG_TEMPERATURE = -8.0
    _MAX_LOG_TEMPERATURE = 8.0

    def __init__(
        self,
        *,
        apply_alive_mask: bool = True,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        initial_temperature = float(initial_temperature)
        if not torch.isfinite(torch.tensor(initial_temperature)) or initial_temperature <= 0.0:
            raise ValueError(
                "temperature-scaling initial_temperature must be finite and positive"
            )
        self.apply_alive_mask = bool(apply_alive_mask)
        initial_log_temperature = float(torch.tensor(initial_temperature).log().item())
        self.log_temperatures = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(initial_log_temperature))
                for name in BRANCH_NAMES
            }
        )

    @staticmethod
    def _alive_by_branch(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
            raise ValueError(
                "temperature-scaling confidence expects evidence with shape "
                f"[B, >= {EvidenceIndex.BASE_DIM}], got {tuple(evidence.shape)}"
            )
        api = _column(evidence, EvidenceIndex.API_ALIVE)
        graph = _column(evidence, EvidenceIndex.GRAPH_ALIVE)
        manifest = _column(evidence, EvidenceIndex.MANIFEST_ALIVE)
        return {
            "api": api,
            "graph": graph,
            "manifest": manifest,
        }

    @staticmethod
    def _validate_logits(
        logits: torch.Tensor,
        *,
        branch: str,
        batch_size: int,
    ) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"branch_logits[{branch!r}] must be a tensor")
        if (
            logits.ndim != 2
            or logits.size(0) != batch_size
            or logits.size(1) < 2
        ):
            raise ValueError(
                f"branch_logits[{branch!r}] must have shape [B, C] with C >= 2; "
                f"got {tuple(logits.shape)} for B={batch_size}"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError(f"branch_logits[{branch!r}] contains non-finite values")
        return logits.float()

    def temperature(self, branch: str) -> torch.Tensor:
        if branch not in self.log_temperatures:
            raise ValueError(f"unsupported temperature-scaling branch {branch!r}")
        bounded_log_temperature = self.log_temperatures[branch].clamp(
            self._MIN_LOG_TEMPERATURE,
            self._MAX_LOG_TEMPERATURE,
        )
        return bounded_log_temperature.exp()

    def branch_parameters(self, branch: str) -> list[nn.Parameter]:
        if branch not in self.log_temperatures:
            raise ValueError(f"unsupported temperature-scaling branch {branch!r}")
        return [self.log_temperatures[branch]]

    def branch_nll(
        self,
        branch: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        """Temperature-scaling NLL for one available branch-logit view."""

        batch_size = int(labels.numel())
        logits = self._validate_logits(
            logits,
            branch=branch,
            batch_size=batch_size,
        )
        if labels.ndim != 1:
            labels = labels.view(-1)
        labels = labels.to(device=logits.device, dtype=torch.long)
        if labels.numel() != batch_size:
            raise ValueError("temperature-scaling labels and logits disagree")
        alive = self._alive_by_branch(
            evidence.to(device=logits.device, dtype=logits.dtype)
        )[branch].view(-1)
        available = alive > 0.0
        if not bool(available.any().item()):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[available] / self.temperature(branch),
            labels[available],
        )

    def forward(
        self,
        evidence: torch.Tensor,
        *,
        branch_probabilities: dict[str, torch.Tensor] | None = None,
        branch_logits: dict[str, torch.Tensor] | None = None,
        branch_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del branch_probabilities, branch_embeddings
        if not isinstance(branch_logits, dict):
            raise ValueError(
                "temperature-scaling confidence requires raw branch_logits"
            )
        missing = [name for name in BRANCH_NAMES if branch_logits.get(name) is None]
        if missing:
            raise ValueError(
                "temperature-scaling confidence is missing branch logits: "
                f"{missing}"
            )
        alive = self._alive_by_branch(evidence)
        outputs: dict[str, torch.Tensor] = {}
        for name in BRANCH_NAMES:
            logits = self._validate_logits(
                branch_logits[name],
                branch=name,
                batch_size=evidence.size(0),
            ).to(device=evidence.device)
            temperature = self.temperature(name).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            log_probability = F.log_softmax(logits / temperature, dim=-1)
            confidence = log_probability.exp().amax(dim=-1)
            reliability_logit = torch.logit(
                confidence.clamp(1.0e-6, 1.0 - 1.0e-6)
            )
            reliability = confidence
            if self.apply_alive_mask:
                reliability = reliability * alive[name].to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
            outputs[f"alive_{name}"] = alive[name].to(
                device=logits.device,
                dtype=logits.dtype,
            )
            outputs[f"predicted_reliability_logit_{name}"] = reliability_logit
            outputs[f"predicted_reliability_{name}"] = reliability
            outputs[f"temperature_scaled_log_prob_{name}"] = log_probability
            outputs[f"reliability_temperature_{name}"] = temperature.expand(
                evidence.size(0)
            )
        outputs["temperature_scaling_confidence_baseline_active"] = torch.ones(
            evidence.size(0),
            device=evidence.device,
            dtype=evidence.dtype,
        )
        return outputs


def build_monotonic_reliability_features(
    evidence: torch.Tensor,
    *,
    use_model_visibility: bool = False,
    use_embedding_density: bool = False,
    use_prediction_margin: bool = True,
    use_predicted_class_feature: bool = True,
    branch_probabilities: dict[str, torch.Tensor] | None = None,
    branch_logits: dict[str, torch.Tensor] | None = None,
    embedding_in_distribution_scores: dict[str, torch.Tensor] | None = None,
    embedding_tail_bases: dict[str, torch.Tensor] | None = None,
    prediction_features: tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]
    | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build the operative branch-local I1 correctness features."""
    if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
        raise ValueError(
            f"Expected [B, >= {EvidenceIndex.BASE_DIM}] evidence, got {tuple(evidence.shape)}"
        )

    api_integrity = _column(evidence, EvidenceIndex.API_INTEGRITY)
    graph_integrity = _column(evidence, EvidenceIndex.GRAPH_INTEGRITY)
    manifest_integrity = _column(evidence, EvidenceIndex.MANIFEST_INTEGRITY)
    api_visibility = _column(evidence, EvidenceIndex.API_ENCODER_COVERAGE)
    graph_visibility = _column(evidence, EvidenceIndex.GRAPH_ENCODER_COVERAGE)

    api_alive = _column(evidence, EvidenceIndex.API_ALIVE).bool()
    graph_alive = _column(evidence, EvidenceIndex.GRAPH_ALIVE).bool()
    manifest_alive = _column(evidence, EvidenceIndex.MANIFEST_ALIVE).bool()

    if prediction_features is None:
        margins, predicted_malware = build_branch_prediction_features(
            branch_probabilities,
            branch_logits=branch_logits,
            batch_size=evidence.size(0),
            device=evidence.device,
            dtype=evidence.dtype,
        )
    else:
        margins, predicted_malware = prediction_features
    zeros = torch.zeros_like(api_integrity)
    zero_tail = torch.zeros(
        evidence.size(0),
        len(EMBEDDING_TAIL_QUANTILES),
        device=evidence.device,
        dtype=evidence.dtype,
    )
    if not use_prediction_margin:
        margins = {name: zeros for name in BRANCH_NAMES}
    if not use_predicted_class_feature:
        predicted_malware = {name: zeros for name in BRANCH_NAMES}
    if use_embedding_density:
        if not isinstance(embedding_in_distribution_scores, dict) or not isinstance(
            embedding_tail_bases, dict
        ):
            raise ValueError(
                "I1 embedding density requires one branch-local score and tail "
                "basis per branch"
            )
        missing_density = [
            name
            for name in BRANCH_NAMES
            if not isinstance(embedding_in_distribution_scores.get(name), torch.Tensor)
        ]
        if missing_density:
            raise ValueError(
                "I1 embedding density is missing branches: "
                f"{missing_density}"
            )
        density = {}
        tails = {}
        for name in BRANCH_NAMES:
            value = embedding_in_distribution_scores[name].to(
                device=evidence.device, dtype=evidence.dtype
            ).view(-1)
            if value.numel() != evidence.size(0):
                raise ValueError(
                    f"I1 embedding density score for {name!r} is invalid"
                )
            density[name] = value.clamp(0.0, 1.0)
            tail = embedding_tail_bases.get(name)
            if (
                not isinstance(tail, torch.Tensor)
                or tail.ndim != 2
                or tail.size(0) != evidence.size(0)
                or tail.size(1) != len(EMBEDDING_TAIL_QUANTILES)
            ):
                raise ValueError(
                    f"I1 embedding tail basis for {name!r} must have shape "
                    f"[B, {len(EMBEDDING_TAIL_QUANTILES)}]"
                )
            tails[name] = tail.to(
                device=evidence.device, dtype=evidence.dtype
            ).clamp(0.0, 1.0)
    else:
        density = {name: zeros for name in BRANCH_NAMES}
        tails = {name: zero_tail for name in BRANCH_NAMES}

    effective_quality = {
        "api": (
            api_integrity * api_visibility
            if use_model_visibility
            else api_integrity
        ).clamp(0.0, 1.0),
        "graph": (
            graph_integrity * graph_visibility
            if use_model_visibility
            else graph_integrity
        ).clamp(0.0, 1.0),
        # Manifest has no independent encoder-coverage measurement.
        "manifest": manifest_integrity,
    }
    quality_deficit = {
        name: (1.0 - value).clamp(0.0, 1.0)
        for name, value in effective_quality.items()
    }
    diagnostics = {
        "alive_api": api_alive.float(),
        "alive_graph": graph_alive.float(),
        "alive_manifest": manifest_alive.float(),
        "prediction_margin_feature_active": torch.full_like(
            api_integrity, float(use_prediction_margin)
        ),
        "predicted_class_feature_active": torch.full_like(
            api_integrity, float(use_predicted_class_feature)
        ),
        "embedding_density_feature_active": torch.full_like(
            api_integrity, float(use_embedding_density)
        ),
        "model_visibility_feature_api": (
            api_visibility if use_model_visibility else torch.ones_like(api_visibility)
        ),
        "model_visibility_feature_graph": (
            graph_visibility
            if use_model_visibility
            else torch.ones_like(graph_visibility)
        ),
    }
    features = {
        "api": torch.stack(
            [
                quality_deficit["api"],
                *tails["api"].unbind(dim=-1),
                margins["api"],
                predicted_malware["api"],
            ],
            dim=-1,
        ),
        "graph": torch.stack(
            [
                quality_deficit["graph"],
                *tails["graph"].unbind(dim=-1),
                margins["graph"],
                predicted_malware["graph"],
            ],
            dim=-1,
        ),
        "manifest": torch.stack(
            [
                quality_deficit["manifest"],
                *tails["manifest"].unbind(dim=-1),
                margins["manifest"],
                predicted_malware["manifest"],
            ],
            dim=-1,
        ),
    }
    for name in BRANCH_NAMES:
        diagnostics[f"prediction_margin_{name}"] = margins[name]
        diagnostics[f"predicted_malware_indicator_{name}"] = predicted_malware[name]
        diagnostics[f"embedding_in_distribution_score_{name}"] = density[name]
        diagnostics[f"effective_quality_{name}"] = effective_quality[name]
        diagnostics[f"effective_quality_deficit_{name}"] = quality_deficit[name]
        for index, quantile in enumerate(EMBEDDING_TAIL_QUANTILES):
            diagnostics[
                f"embedding_tail_q{int(round(100.0 * quantile))}_{name}"
            ] = tails[name][:, index]
    return features, diagnostics


class MonotonicReliabilityCalibrator(nn.Module):
    """Three branch-specific monotonic correctness-probability estimators."""

    def __init__(
        self,
        use_model_visibility: bool = False,
        use_embedding_density: bool = False,
        use_prediction_margin: bool = True,
        use_predicted_class_feature: bool = True,
        apply_alive_mask: bool = True,
        embedding_dims: dict[str, int] | None = None,
        embedding_density_variance_shrinkage: float = 0.10,
        embedding_density_reference_quantile: float = 0.95,
        embedding_density_min_class_samples: int = 8,
    ):
        super().__init__()
        self.use_model_visibility = bool(use_model_visibility)
        self.use_embedding_density = bool(use_embedding_density)
        self.use_prediction_margin = bool(use_prediction_margin)
        self.use_predicted_class_feature = bool(use_predicted_class_feature)
        self.apply_alive_mask = bool(apply_alive_mask)
        if self.use_embedding_density:
            if not isinstance(embedding_dims, dict):
                raise ValueError(
                    "I1 embedding density requires configured branch embedding dimensions"
                )
            missing_dims = [name for name in BRANCH_NAMES if name not in embedding_dims]
            if missing_dims:
                raise ValueError(
                    "I1 embedding density is missing embedding dimensions for "
                    f"{missing_dims}"
                )
            self.embedding_references = nn.ModuleDict(
                {
                    name: ClassConditionalEmbeddingDensity(
                        int(embedding_dims[name]),
                        variance_shrinkage=embedding_density_variance_shrinkage,
                        reference_quantile=embedding_density_reference_quantile,
                        min_class_samples=embedding_density_min_class_samples,
                    )
                    for name in BRANCH_NAMES
                }
            )
        else:
            self.embedding_references = nn.ModuleDict()
        self.branches = nn.ModuleDict(
            {
                name: MonotonicBranchCalibrator(
                    input_dim=len(RELIABILITY_FEATURE_LAYOUT[name]),
                    signed_input_indices=tuple(
                        index
                        for index, feature_name in enumerate(
                            RELIABILITY_FEATURE_LAYOUT[name]
                        )
                        if feature_name == "predicted_malware_indicator"
                    ),
                )
                for name in BRANCH_NAMES
            }
        )

    def branch_parameters(self, branch: str) -> list[nn.Parameter]:
        if branch not in self.branches:
            raise ValueError(f"unsupported monotonic reliability branch {branch!r}")
        return list(self.branches[branch].parameters())

    def branch_competence_parameters(self, branch: str) -> list[nn.Parameter]:
        if branch not in self.branches:
            raise ValueError(f"unsupported monotonic reliability branch {branch!r}")
        return self.branches[branch].competence_parameters()

    def branch_degradation_parameters(self, branch: str) -> list[nn.Parameter]:
        if branch not in self.branches:
            raise ValueError(f"unsupported monotonic reliability branch {branch!r}")
        return self.branches[branch].degradation_parameters()

    def competence_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name in BRANCH_NAMES
            for parameter in self.branch_competence_parameters(name)
        ]

    def degradation_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name in BRANCH_NAMES
            for parameter in self.branch_degradation_parameters(name)
        ]

    @torch.no_grad()
    def fit_embedding_references(
        self,
        branch_embeddings: dict[str, torch.Tensor],
        labels: torch.Tensor,
        evidence: torch.Tensor,
    ) -> None:
        if not self.use_embedding_density:
            return
        if not isinstance(branch_embeddings, dict):
            raise ValueError("I1 embedding density requires branch embeddings")
        if evidence.ndim != 2 or evidence.size(-1) < EvidenceIndex.BASE_DIM:
            raise ValueError("I1 embedding reference requires valid evidence")
        alive = {
            "api": _column(evidence, EvidenceIndex.API_ALIVE).bool(),
            "graph": _column(evidence, EvidenceIndex.GRAPH_ALIVE).bool(),
            "manifest": _column(evidence, EvidenceIndex.MANIFEST_ALIVE).bool(),
        }
        for name in BRANCH_NAMES:
            value = branch_embeddings.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"I1 embedding reference is missing branch {name!r}"
                )
            self.embedding_references[name].fit(value, labels, alive[name])

    def embedding_reference_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().clone()
            for key, value in self.embedding_references.state_dict().items()
        }

    @torch.no_grad()
    def restore_embedding_reference_snapshot(
        self, state: dict[str, torch.Tensor]
    ) -> None:
        if not self.use_embedding_density:
            if state:
                raise ValueError("disabled embedding density received reference state")
            return
        self.embedding_references.load_state_dict(state, strict=True)

    def embedding_reference_summary(self) -> dict[str, dict[str, object]]:
        if not self.use_embedding_density:
            return {}
        result: dict[str, dict[str, object]] = {}
        for name, reference in self.embedding_references.items():
            digest = hashlib.sha256()
            for value in (
                reference.class_mean,
                reference.class_variance,
                reference.distance_scale,
                reference.distance_quantiles,
                reference.class_count,
                reference.reference_fitted,
            ):
                digest.update(
                    value.detach().cpu().contiguous().numpy().tobytes()
                )
            result[name] = {
                "fitted": bool(reference.reference_fitted.detach().cpu().item()),
                "class_count": [
                    int(value)
                    for value in reference.class_count.detach().cpu().tolist()
                ],
                "distance_scale": [
                    float(value)
                    for value in reference.distance_scale.detach().cpu().tolist()
                ],
                "distance_quantiles": [
                    [float(value) for value in row]
                    for row in reference.distance_quantiles.detach().cpu().tolist()
                ],
                "tail_quantiles": [
                    float(value) for value in reference.tail_quantiles
                ],
                "embedding_dim": int(reference.embedding_dim),
                "variance_shrinkage": float(reference.variance_shrinkage),
                "reference_quantile": float(reference.reference_quantile),
                "reference_sha256": digest.hexdigest(),
            }
        return result

    def forward(
        self,
        evidence: torch.Tensor,
        *,
        branch_probabilities: dict[str, torch.Tensor] | None = None,
        branch_logits: dict[str, torch.Tensor] | None = None,
        branch_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        density_scores: dict[str, torch.Tensor] | None = None
        density_distances: dict[str, torch.Tensor] = {}
        density_tail_bases: dict[str, torch.Tensor] | None = None
        if self.use_embedding_density:
            if not isinstance(branch_embeddings, dict) or any(
                not isinstance(branch_embeddings.get(name), torch.Tensor)
                for name in BRANCH_NAMES
            ):
                raise ValueError("I1 embedding density requires branch embeddings")
            if not isinstance(branch_logits, dict) or any(
                not isinstance(branch_logits.get(name), torch.Tensor)
                for name in BRANCH_NAMES
            ):
                raise ValueError(
                    "I1 embedding density requires raw branch_logits so its "
                    "class reference matches the argmax correctness target"
                )
        prediction_features = build_branch_prediction_features(
            branch_probabilities,
            branch_logits=branch_logits,
            batch_size=evidence.size(0),
            device=evidence.device,
            dtype=evidence.dtype,
        )
        if self.use_embedding_density:
            _margins, predicted_malware = prediction_features
            density_scores = {}
            density_tail_bases = {}
            for name in BRANCH_NAMES:
                predicted_class = predicted_malware[name].long()
                score, distance = self.embedding_references[name](
                    branch_embeddings[name], predicted_class
                )
                density_scores[name] = score
                density_distances[name] = distance
                density_tail_bases[name] = self.embedding_references[
                    name
                ].tail_basis(distance, predicted_class)
        features, diagnostics = build_monotonic_reliability_features(
            evidence,
            use_model_visibility=self.use_model_visibility,
            use_embedding_density=self.use_embedding_density,
            use_prediction_margin=self.use_prediction_margin,
            use_predicted_class_feature=self.use_predicted_class_feature,
            branch_probabilities=branch_probabilities,
            branch_logits=branch_logits,
            embedding_in_distribution_scores=density_scores,
            embedding_tail_bases=density_tail_bases,
            prediction_features=prediction_features,
        )
        outputs = dict(diagnostics)
        for name, distance in density_distances.items():
            outputs[f"embedding_mahalanobis_distance_{name}"] = distance
        for name in BRANCH_NAMES:
            alive = diagnostics[f"alive_{name}"]
            branch_outputs = self.branches[name].forward_components(features[name])
            reliability_logit = branch_outputs["reliability_logit"]
            reliability = torch.sigmoid(reliability_logit)
            if self.apply_alive_mask:
                reliability = reliability * alive
            # Expose both the invariant input tensor and its operative columns.
            outputs[f"reliability_features_superset_{name}"] = features[name]
            active_indices = []
            for index, feature_name in enumerate(RELIABILITY_FEATURE_LAYOUT[name]):
                if feature_name.startswith("embedding_tail_"):
                    active = self.use_embedding_density
                elif feature_name == "prediction_margin":
                    active = self.use_prediction_margin
                elif feature_name == "predicted_malware_indicator":
                    active = self.use_predicted_class_feature
                else:
                    active = True
                if active:
                    active_indices.append(index)
            outputs[f"reliability_features_{name}"] = features[name][
                :, active_indices
            ]
            for key, value in branch_outputs.items():
                if key == "reliability_logit":
                    continue
                outputs[f"{key}_{name}"] = value
            outputs[f"predicted_reliability_logit_{name}"] = reliability_logit
            outputs[f"predicted_reliability_{name}"] = reliability
        return outputs
