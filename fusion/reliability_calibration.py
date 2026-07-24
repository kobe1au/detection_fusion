from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


BRANCH_NAMES = ("api", "graph", "manifest")

MONOTONIC_CORRECTNESS_METHOD = "monotonic_correctness"
TEMPERATURE_SCALING_CONFIDENCE_METHOD = "temperature_scaling_confidence"
RELIABILITY_CALIBRATION_METHODS = (
    MONOTONIC_CORRECTNESS_METHOD,
    TEMPERATURE_SCALING_CONFIDENCE_METHOD,
)

RELIABILITY_FEATURE_NAMES = (
    "evidential_certainty",
    "prediction_margin",
    "predicted_malware_indicator",
)
RELIABILITY_FEATURE_LAYOUT = {
    branch: RELIABILITY_FEATURE_NAMES for branch in BRANCH_NAMES
}
CONTINUOUS_RELIABILITY_FEATURES = RELIABILITY_FEATURE_NAMES[:2]


def normalize_reliability_calibration_method(value: str) -> str:
    """Return the canonical I1 estimator identity."""

    method = str(value).strip().lower()
    if method not in RELIABILITY_CALIBRATION_METHODS:
        raise ValueError(
            "reliability_calibration.method must be one of "
            f"{list(RELIABILITY_CALIBRATION_METHODS)}, got {value!r}"
        )
    return method


def _require_exact_branch_mapping(
    values: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a branch-to-tensor mapping")
    missing = [branch for branch in BRANCH_NAMES if branch not in values]
    unknown = sorted(set(values) - set(BRANCH_NAMES))
    if missing or unknown:
        raise ValueError(
            f"{name} must contain exactly {list(BRANCH_NAMES)}; "
            f"missing={missing}, unknown={unknown}"
        )


def _validate_binary_alpha(
    alpha: torch.Tensor,
    *,
    branch: str,
) -> torch.Tensor:
    if not isinstance(alpha, torch.Tensor):
        raise ValueError(f"branch_alpha[{branch!r}] must be a tensor")
    if not alpha.is_floating_point():
        raise ValueError(f"branch_alpha[{branch!r}] must be floating point")
    if alpha.ndim != 2 or alpha.size(0) <= 0 or alpha.size(1) != 2:
        raise ValueError(
            f"branch_alpha[{branch!r}] must have binary shape [B, 2], "
            f"got {tuple(alpha.shape)}"
        )
    if not bool(torch.isfinite(alpha).all().item()):
        raise ValueError(f"branch_alpha[{branch!r}] contains non-finite values")
    if bool((alpha < 1.0).any().item()):
        raise ValueError(
            f"branch_alpha[{branch!r}] must be Dirichlet evidence alpha=e+1 "
            "with every concentration >= 1"
        )
    return alpha


def build_reliability_features(
    branch_alpha: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build the complete intrinsic model-state input to the proposed I1.

    For binary Dirichlet concentration ``alpha`` with strength ``S`` and
    ``K=2`` classes, the feature vector is exactly:

    ``[1 - K/S, top1(alpha/S) - top2(alpha/S), argmax(alpha/S) == 1]``.

    No quality proxy, model-visibility value, perturbation metadata, peer
    modality, or hidden two-stage feature is accepted by this API.
    """

    _require_exact_branch_mapping(branch_alpha, name="branch_alpha")
    features: dict[str, torch.Tensor] = {}
    batch_size: int | None = None
    for branch in BRANCH_NAMES:
        alpha = _validate_binary_alpha(branch_alpha[branch], branch=branch)
        if batch_size is None:
            batch_size = int(alpha.size(0))
        elif int(alpha.size(0)) != batch_size:
            raise ValueError("branch_alpha batch sizes disagree")

        strength = alpha.sum(dim=-1)
        certainty = (1.0 - 2.0 / strength).clamp(0.0, 1.0)
        expected_probability = alpha / strength.unsqueeze(-1)
        top_two = expected_probability.topk(k=2, dim=-1).values
        margin = (top_two[:, 0] - top_two[:, 1]).clamp(0.0, 1.0)
        predicted_malware = expected_probability.argmax(dim=-1).eq(1).to(alpha)
        features[branch] = torch.stack(
            [certainty, margin, predicted_malware],
            dim=-1,
        )
    return features


def _validate_feature_tensor(
    features: torch.Tensor,
    *,
    branch: str,
) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"features[{branch!r}] must be a tensor")
    if not features.is_floating_point():
        raise ValueError(f"features[{branch!r}] must be floating point")
    expected_width = len(RELIABILITY_FEATURE_NAMES)
    if (
        features.ndim != 2
        or features.size(0) <= 0
        or features.size(1) != expected_width
    ):
        raise ValueError(
            f"features[{branch!r}] must have shape [B, {expected_width}], "
            f"got {tuple(features.shape)}"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError(f"features[{branch!r}] contains non-finite values")
    continuous = features[:, :2]
    if bool(((continuous < 0.0) | (continuous > 1.0)).any().item()):
        raise ValueError(
            f"features[{branch!r}] certainty and margin must lie within [0, 1]"
        )
    predicted_malware = features[:, 2]
    if bool(
        ((predicted_malware != 0.0) & (predicted_malware != 1.0)).any().item()
    ):
        raise ValueError(
            f"features[{branch!r}] predicted-class indicator must be binary"
        )
    return features


def _normalize_alive(
    value: torch.Tensor,
    *,
    branch: str,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"alive[{branch!r}] must be a tensor")
    alive = value.to(device=reference.device).reshape(-1)
    if alive.numel() != batch_size:
        raise ValueError(
            f"alive[{branch!r}] must contain {batch_size} values, "
            f"got {alive.numel()}"
        )
    if not bool(torch.isfinite(alive.float()).all().item()):
        raise ValueError(f"alive[{branch!r}] contains non-finite values")
    if alive.dtype == torch.bool:
        return alive.to(dtype=reference.dtype)
    if bool(((alive != 0) & (alive != 1)).any().item()):
        raise ValueError(f"alive[{branch!r}] must be a hard binary mask")
    return alive.to(dtype=reference.dtype)


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("initial monotone weight must be finite and positive")
    return math.log(math.expm1(value))


class MonotonicBranchCorrectnessCalibrator(nn.Module):
    """One branch's structurally monotone correctness-probability model.

    The logit is

    ``b + w_c * certainty + w_m * margin + gamma * predicted_malware``

    where ``w_c`` and ``w_m`` are strictly non-negative through softplus.
    ``gamma`` is an optional signed intercept because predicted class is
    categorical rather than an ordered quality coordinate.
    """

    def __init__(
        self,
        *,
        use_evidential_certainty: bool = True,
        use_prediction_margin: bool = True,
        use_predicted_class_intercept: bool = True,
        initial_continuous_weight: float = 0.25,
        initial_bias: float = 0.0,
    ) -> None:
        super().__init__()
        active_mask = torch.tensor(
            [
                bool(use_evidential_certainty),
                bool(use_prediction_margin),
            ],
            dtype=torch.float32,
        )
        if not bool(active_mask.bool().any().item()):
            raise ValueError(
                "at least one of evidential certainty or prediction margin "
                "must be active"
            )
        self.register_buffer(
            "active_continuous_mask",
            active_mask,
            persistent=True,
        )
        initial_bias = float(initial_bias)
        if not math.isfinite(initial_bias):
            raise ValueError("initial_bias must be finite")
        raw_initial = _inverse_softplus(initial_continuous_weight)
        self.raw_continuous_weights = nn.Parameter(
            torch.full((len(CONTINUOUS_RELIABILITY_FEATURES),), raw_initial)
        )
        self.bias = nn.Parameter(torch.tensor(initial_bias))
        self.use_predicted_class_intercept = bool(
            use_predicted_class_intercept
        )
        if self.use_predicted_class_intercept:
            self.predicted_class_intercept = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("predicted_class_intercept", None)

    def effective_continuous_weights(self) -> dict[str, torch.Tensor]:
        weights = (
            F.softplus(self.raw_continuous_weights)
            * self.active_continuous_mask
        )
        return {
            name: weights[index]
            for index, name in enumerate(CONTINUOUS_RELIABILITY_FEATURES)
        }

    def forward_logit(self, features: torch.Tensor) -> torch.Tensor:
        features = _validate_feature_tensor(features, branch="single_branch")
        weights = (
            F.softplus(self.raw_continuous_weights)
            * self.active_continuous_mask
        ).to(features)
        bias = self.bias.to(features)
        if not bool(torch.isfinite(weights).all().item()) or not bool(
            torch.isfinite(bias).item()
        ):
            raise RuntimeError(
                "monotonic correctness parameters must remain finite"
            )
        logit = bias + (
            features[:, :2] * weights.view(1, -1)
        ).sum(dim=-1)
        if self.predicted_class_intercept is not None:
            class_intercept = self.predicted_class_intercept.to(features)
            if not bool(torch.isfinite(class_intercept).item()):
                raise RuntimeError(
                    "predicted-class intercept must remain finite"
                )
            logit = logit + features[:, 2] * class_intercept
        return logit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logit(features))


class MonotonicReliabilityCalibrator(nn.Module):
    """Independent monotone correctness calibrators for API/Graph/Manifest."""

    def __init__(
        self,
        *,
        use_evidential_certainty: bool = True,
        use_prediction_margin: bool = True,
        use_predicted_class_intercept: bool = True,
        initial_continuous_weight: float = 0.25,
        initial_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_evidential_certainty = bool(use_evidential_certainty)
        self.use_prediction_margin = bool(use_prediction_margin)
        self.use_predicted_class_intercept = bool(
            use_predicted_class_intercept
        )
        self.branches = nn.ModuleDict(
            {
                branch: MonotonicBranchCorrectnessCalibrator(
                    use_evidential_certainty=self.use_evidential_certainty,
                    use_prediction_margin=self.use_prediction_margin,
                    use_predicted_class_intercept=(
                        self.use_predicted_class_intercept
                    ),
                    initial_continuous_weight=initial_continuous_weight,
                    initial_bias=initial_bias,
                )
                for branch in BRANCH_NAMES
            }
        )

    def branch_parameters(self, branch: str) -> list[nn.Parameter]:
        branch = str(branch).strip().lower()
        if branch not in self.branches:
            raise ValueError(f"unsupported reliability branch {branch!r}")
        return list(self.branches[branch].parameters())

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        alive: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        _require_exact_branch_mapping(features, name="features")
        _require_exact_branch_mapping(alive, name="alive")

        validated: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        for branch in BRANCH_NAMES:
            value = _validate_feature_tensor(features[branch], branch=branch)
            if batch_size is None:
                batch_size = int(value.size(0))
            elif int(value.size(0)) != batch_size:
                raise ValueError("reliability feature batch sizes disagree")
            validated[branch] = value
        assert batch_size is not None

        outputs: dict[str, torch.Tensor] = {}
        for branch in BRANCH_NAMES:
            branch_features = validated[branch]
            branch_alive = _normalize_alive(
                alive[branch],
                branch=branch,
                batch_size=batch_size,
                reference=branch_features,
            )
            raw_logit = self.branches[branch].forward_logit(branch_features)
            probability = torch.sigmoid(raw_logit) * branch_alive
            outputs[f"predicted_reliability_{branch}"] = probability
            outputs[f"predicted_reliability_logit_{branch}"] = raw_logit
            outputs[f"reliability_features_{branch}"] = branch_features
            outputs[f"evidential_certainty_{branch}"] = branch_features[:, 0]
            outputs[f"prediction_margin_{branch}"] = branch_features[:, 1]
            outputs[f"predicted_malware_indicator_{branch}"] = (
                branch_features[:, 2]
            )
            outputs[f"alive_{branch}"] = branch_alive

        reference = validated[BRANCH_NAMES[0]][:, 0]
        outputs["monotonic_correctness_calibrator_active"] = torch.ones_like(
            reference
        )
        outputs["predicted_class_intercept_active"] = torch.full_like(
            reference,
            float(self.use_predicted_class_intercept),
        )
        outputs["evidential_certainty_feature_active"] = torch.full_like(
            reference,
            float(self.use_evidential_certainty),
        )
        outputs["prediction_margin_feature_active"] = torch.full_like(
            reference,
            float(self.use_prediction_margin),
        )
        return outputs


def _validate_binary_logits(
    logits: torch.Tensor,
    *,
    branch: str,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError(f"branch_logits[{branch!r}] must be a tensor")
    if not logits.is_floating_point():
        raise ValueError(f"branch_logits[{branch!r}] must be floating point")
    if logits.ndim != 2 or logits.size(0) <= 0 or logits.size(1) != 2:
        raise ValueError(
            f"branch_logits[{branch!r}] must have binary shape [B, 2], "
            f"got {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"branch_logits[{branch!r}] contains non-finite values")
    return logits


class BranchTemperatureScalingConfidenceCalibrator(nn.Module):
    """Per-branch scalar-temperature confidence comparator.

    This is deliberately separate from the proposed I1. It consumes raw binary
    logits only, and its reliability score is max softmax confidence after
    temperature scaling.
    """

    def __init__(self, *, initial_temperature: float = 1.0) -> None:
        super().__init__()
        initial_temperature = float(initial_temperature)
        if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be finite and positive")
        self.log_temperatures = nn.ParameterDict(
            {
                branch: nn.Parameter(
                    torch.tensor(math.log(initial_temperature))
                )
                for branch in BRANCH_NAMES
            }
        )

    def temperature(self, branch: str) -> torch.Tensor:
        branch = str(branch).strip().lower()
        if branch not in self.log_temperatures:
            raise ValueError(f"unsupported temperature branch {branch!r}")
        return self.log_temperatures[branch].exp()

    def branch_parameters(self, branch: str) -> list[nn.Parameter]:
        branch = str(branch).strip().lower()
        if branch not in self.log_temperatures:
            raise ValueError(f"unsupported temperature branch {branch!r}")
        return [self.log_temperatures[branch]]

    def branch_nll(
        self,
        branch: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        logits = _validate_binary_logits(logits, branch=branch)
        if not isinstance(labels, torch.Tensor):
            raise ValueError("temperature-scaling labels must be a tensor")
        labels = labels.to(device=logits.device).long().reshape(-1)
        if labels.numel() != logits.size(0):
            raise ValueError("temperature-scaling labels disagree with logits")
        if bool(((labels < 0) | (labels > 1)).any().item()):
            raise ValueError("temperature-scaling labels must be binary")
        alive_mask = _normalize_alive(
            alive,
            branch=branch,
            batch_size=int(logits.size(0)),
            reference=logits,
        ).bool()
        if not bool(alive_mask.any().item()):
            raise ValueError(
                f"temperature-scaling branch {branch!r} has no alive rows"
            )
        temperature = self.temperature(branch).to(logits)
        if not bool(torch.isfinite(temperature).item()) or not bool(
            (temperature > 0.0).item()
        ):
            raise ValueError(
                f"temperature-scaling branch {branch!r} must have a finite "
                "positive temperature"
            )
        return F.cross_entropy(
            logits[alive_mask] / temperature,
            labels[alive_mask],
        )

    def forward(
        self,
        branch_logits: Mapping[str, torch.Tensor],
        *,
        alive: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        _require_exact_branch_mapping(branch_logits, name="branch_logits")
        _require_exact_branch_mapping(alive, name="alive")
        outputs: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        for branch in BRANCH_NAMES:
            logits = _validate_binary_logits(
                branch_logits[branch],
                branch=branch,
            )
            if batch_size is None:
                batch_size = int(logits.size(0))
            elif int(logits.size(0)) != batch_size:
                raise ValueError("branch_logits batch sizes disagree")
            alive_mask = _normalize_alive(
                alive[branch],
                branch=branch,
                batch_size=int(logits.size(0)),
                reference=logits,
            )
            temperature = self.temperature(branch).to(logits)
            if not bool(torch.isfinite(temperature).item()) or not bool(
                (temperature > 0.0).item()
            ):
                raise ValueError(
                    f"temperature-scaling branch {branch!r} must have a finite "
                    "positive temperature"
                )
            probability = torch.softmax(logits / temperature, dim=-1)
            confidence = probability.amax(dim=-1)
            raw_logit = torch.logit(
                confidence.clamp(
                    torch.finfo(confidence.dtype).eps,
                    1.0 - torch.finfo(confidence.dtype).eps,
                )
            )
            outputs[f"predicted_reliability_{branch}"] = (
                confidence * alive_mask
            )
            outputs[f"predicted_reliability_logit_{branch}"] = raw_logit
            outputs[f"reliability_temperature_{branch}"] = (
                temperature.expand_as(confidence)
            )
            outputs[f"alive_{branch}"] = alive_mask

        assert batch_size is not None
        reference = outputs[f"predicted_reliability_{BRANCH_NAMES[0]}"]
        outputs["temperature_scaling_confidence_baseline_active"] = (
            torch.ones_like(reference)
        )
        return outputs


__all__ = [
    "BRANCH_NAMES",
    "CONTINUOUS_RELIABILITY_FEATURES",
    "MONOTONIC_CORRECTNESS_METHOD",
    "RELIABILITY_CALIBRATION_METHODS",
    "RELIABILITY_FEATURE_LAYOUT",
    "RELIABILITY_FEATURE_NAMES",
    "TEMPERATURE_SCALING_CONFIDENCE_METHOD",
    "BranchTemperatureScalingConfidenceCalibrator",
    "MonotonicBranchCorrectnessCalibrator",
    "MonotonicReliabilityCalibrator",
    "build_reliability_features",
    "normalize_reliability_calibration_method",
]
