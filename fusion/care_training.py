"""Leakage-safe CARE path-risk training and decision-protocol utilities.

The module is deliberately independent from the training entry point.  It
implements the parts of CARE whose statistical semantics must stay fixed:

* four candidate paths: AGM, AG, AM, and GM;
* an unweighted SID -> view -> valid-path binary cross-entropy;
* three-fold, group-disjoint out-of-fold risk fitting;
* fold-local log-odds normalization fitted on valid training paths only;
* a final refit on the complete routing-calibration population;
* deterministic SID/mechanism perturbation seeds; and
* score-tie-atomic CRC threshold selection.

The risk-head boundary follows ``CAREPathRiskHead`` without importing it:
``score_all(normalized_log_odds, modality_alive) -> probability[B, 4]`` and
``set_log_odds_normalization(center, scale)``.  Keeping this module
duck-typed avoids a circular dependency with the model implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from fusion.care_fusion import hard_predict_log_odds
from fusion.view_protocol import (
    deterministic_view_seed,
    deterministic_view_spec,
)


CARE_PATH_NAMES = ("agm", "ag", "am", "gm")
CARE_RISK_PROTOCOL_ID = "care_path_risk_crossfit_v1"
CARE_RISK_FOLDS = 3


def _require_nonempty_unique_strings(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    if not resolved or any(not value for value in resolved):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must be unique")
    return resolved


def _stable_digest(values: Any) -> str:
    payload = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_length_prefixed(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, "big", signed=False))
    hasher.update(payload)


def clone_state_dict_to_cpu(
    state_dict: dict[str, torch.Tensor] | Any,
) -> dict[str, torch.Tensor]:
    """Return an independent CPU tensor clone suitable for a checkpoint."""

    cloned: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"CARE state_dict entry {key!r} is not a tensor"
            )
        cloned[key] = value.detach().cpu().clone()
    return cloned


def tensor_state_dict_sha256(
    state_dict: dict[str, torch.Tensor] | Any,
) -> str:
    """Hash tensor state independently of mapping order and device."""

    hasher = hashlib.sha256()
    for raw_key in sorted(state_dict):
        key = str(raw_key)
        value = state_dict[raw_key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"CARE state_dict entry {key!r} is not a tensor"
            )
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "layout": str(tensor.layout),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _hash_length_prefixed(hasher, metadata)
        if tensor.layout != torch.strided:
            raise TypeError(
                f"CARE state_dict entry {key!r} must use strided layout"
            )
        raw = (
            tensor.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        _hash_length_prefixed(hasher, raw)
    return hasher.hexdigest()


def care_path_validity(modality_alive: torch.Tensor) -> torch.Tensor:
    """Map API/Graph/Manifest availability to AGM/AG/AM/GM validity."""

    if not isinstance(modality_alive, torch.Tensor):
        raise TypeError("modality_alive must be a tensor")
    if modality_alive.ndim < 2 or int(modality_alive.size(-1)) != 3:
        raise ValueError("modality_alive must have shape [..., 3]")
    if modality_alive.dtype != torch.bool:
        raise TypeError("modality_alive must be boolean")
    api = modality_alive[..., 0]
    graph = modality_alive[..., 1]
    manifest = modality_alive[..., 2]
    return torch.stack(
        [
            api & graph & manifest,
            api & graph,
            api & manifest,
            graph & manifest,
        ],
        dim=-1,
    )


def fixed_path_predictions(path_log_odds: torch.Tensor) -> torch.Tensor:
    """Use the frozen binary rule: malware iff logit_1-logit_0 >= 0."""

    if not isinstance(path_log_odds, torch.Tensor):
        raise TypeError("path_log_odds must be a tensor")
    if not path_log_odds.is_floating_point():
        raise TypeError("path_log_odds must be floating point")
    if path_log_odds.ndim < 2 or int(path_log_odds.size(-1)) != 4:
        raise ValueError("path_log_odds must have shape [..., 4]")
    return hard_predict_log_odds(path_log_odds).bool()


def fixed_path_correctness_targets(
    path_log_odds: torch.Tensor,
    labels: torch.Tensor,
    valid_paths: torch.Tensor,
) -> torch.Tensor:
    """Return binary path-correctness targets with zero invalid placeholders."""

    prediction = fixed_path_predictions(path_log_odds)
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.is_floating_point()
        or labels.dtype == torch.bool
    ):
        raise TypeError("labels must be a one-dimensional integer tensor")
    if int(labels.numel()) != int(path_log_odds.size(0)):
        raise ValueError("labels must contain one value per SID")
    if not bool(((labels == 0) | (labels == 1)).all().item()):
        raise ValueError("labels must be binary")
    if (
        not isinstance(valid_paths, torch.Tensor)
        or valid_paths.shape != path_log_odds.shape
        or valid_paths.dtype != torch.bool
    ):
        raise ValueError(
            "valid_paths must be boolean and match path_log_odds"
        )
    label_shape = [int(labels.numel())] + [1] * (
        path_log_odds.ndim - 1
    )
    expanded_labels = labels.view(*label_shape).to(
        device=prediction.device,
        dtype=torch.bool,
    )
    target = prediction.eq(expanded_labels).to(path_log_odds)
    return torch.where(valid_paths, target, torch.zeros_like(target))


@dataclass(frozen=True)
class CareRiskCalibrationData:
    """One routing-calibration population with all deterministic views."""

    sids: tuple[str, ...]
    groups: tuple[str, ...]
    labels: torch.Tensor
    view_names: tuple[str, ...]
    path_log_odds: torch.Tensor
    modality_alive: torch.Tensor
    valid_paths: torch.Tensor
    correctness_targets: torch.Tensor

    def __post_init__(self) -> None:
        sids = _require_nonempty_unique_strings(self.sids, label="sids")
        groups = tuple(str(value) for value in self.groups)
        views = _require_nonempty_unique_strings(
            self.view_names,
            label="view_names",
        )
        if len(groups) != len(sids) or any(not group for group in groups):
            raise ValueError("groups must contain one non-empty value per SID")
        if (
            not isinstance(self.labels, torch.Tensor)
            or self.labels.ndim != 1
            or int(self.labels.numel()) != len(sids)
            or self.labels.is_floating_point()
            or self.labels.dtype == torch.bool
        ):
            raise TypeError("labels must be integer [num_sids]")
        if not bool(
            ((self.labels == 0) | (self.labels == 1)).all().item()
        ):
            raise ValueError("labels must be binary")
        expected_path_shape = (len(sids), len(views), 4)
        expected_alive_shape = (len(sids), len(views), 3)
        if (
            not isinstance(self.path_log_odds, torch.Tensor)
            or not self.path_log_odds.is_floating_point()
            or tuple(self.path_log_odds.shape) != expected_path_shape
        ):
            raise ValueError(
                f"path_log_odds must have shape {expected_path_shape}"
            )
        if (
            not isinstance(self.modality_alive, torch.Tensor)
            or self.modality_alive.dtype != torch.bool
            or tuple(self.modality_alive.shape) != expected_alive_shape
        ):
            raise ValueError(
                f"modality_alive must be boolean {expected_alive_shape}"
            )
        if (
            not isinstance(self.valid_paths, torch.Tensor)
            or self.valid_paths.dtype != torch.bool
            or tuple(self.valid_paths.shape) != expected_path_shape
        ):
            raise ValueError(
                f"valid_paths must be boolean {expected_path_shape}"
            )
        expected_valid = care_path_validity(self.modality_alive)
        if not torch.equal(
            self.valid_paths.to(expected_valid.device),
            expected_valid,
        ):
            raise ValueError(
                "valid_paths disagrees with the frozen AGM/AG/AM/GM "
                "availability rule"
            )
        if (
            not isinstance(self.correctness_targets, torch.Tensor)
            or not self.correctness_targets.is_floating_point()
            or tuple(self.correctness_targets.shape) != expected_path_shape
        ):
            raise ValueError(
                f"correctness_targets must have shape {expected_path_shape}"
            )
        feature_devices = {
            self.path_log_odds.device,
            self.modality_alive.device,
            self.valid_paths.device,
            self.correctness_targets.device,
        }
        if len(feature_devices) != 1:
            raise ValueError(
                "CARE path, availability, validity, and target tensors "
                "must share one device"
            )
        valid = self.valid_paths
        valid_log_odds = self.path_log_odds[valid]
        valid_targets = self.correctness_targets[valid]
        if not bool(torch.isfinite(valid_log_odds).all().item()):
            raise ValueError("valid path log-odds must be finite")
        if not bool(torch.isfinite(valid_targets).all().item()):
            raise ValueError("valid correctness targets must be finite")
        if not bool(
            ((valid_targets == 0.0) | (valid_targets == 1.0)).all().item()
        ):
            raise ValueError("valid correctness targets must be binary")
        object.__setattr__(self, "sids", sids)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "view_names", views)

    @property
    def num_sids(self) -> int:
        return len(self.sids)

    @property
    def num_views(self) -> int:
        return len(self.view_names)

    @classmethod
    def from_path_log_odds(
        cls,
        *,
        sids: Sequence[str],
        groups: Sequence[str],
        labels: torch.Tensor,
        view_names: Sequence[str],
        path_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
    ) -> "CareRiskCalibrationData":
        valid = care_path_validity(modality_alive)
        targets = fixed_path_correctness_targets(
            path_log_odds,
            labels,
            valid,
        )
        return cls(
            sids=tuple(str(value) for value in sids),
            groups=tuple(str(value) for value in groups),
            labels=labels,
            view_names=tuple(str(value) for value in view_names),
            path_log_odds=path_log_odds,
            modality_alive=modality_alive,
            valid_paths=valid,
            correctness_targets=targets,
        )

    @classmethod
    def from_path_logits(
        cls,
        *,
        sids: Sequence[str],
        groups: Sequence[str],
        labels: torch.Tensor,
        view_names: Sequence[str],
        path_logits: torch.Tensor,
        modality_alive: torch.Tensor,
    ) -> "CareRiskCalibrationData":
        if (
            not isinstance(path_logits, torch.Tensor)
            or not path_logits.is_floating_point()
            or path_logits.ndim != 4
            or tuple(path_logits.shape[-2:]) != (4, 2)
        ):
            raise ValueError(
                "path_logits must have shape [SID, view, 4, 2]"
            )
        return cls.from_path_log_odds(
            sids=sids,
            groups=groups,
            labels=labels,
            view_names=view_names,
            path_log_odds=path_logits[..., 1] - path_logits[..., 0],
            modality_alive=modality_alive,
        )


@dataclass(frozen=True)
class ValidPathLogOddsNormalizer:
    """Per-path center/scale fitted without reading invalid placeholders."""

    center: torch.Tensor
    scale: torch.Tensor
    valid_count: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.center, torch.Tensor)
            or not self.center.is_floating_point()
            or tuple(self.center.shape) != (4,)
        ):
            raise ValueError("center must be floating [4]")
        if (
            not isinstance(self.scale, torch.Tensor)
            or not self.scale.is_floating_point()
            or tuple(self.scale.shape) != (4,)
        ):
            raise ValueError("scale must be floating [4]")
        if (
            not isinstance(self.valid_count, torch.Tensor)
            or tuple(self.valid_count.shape) != (4,)
            or self.valid_count.is_floating_point()
            or self.valid_count.dtype == torch.bool
        ):
            raise ValueError("valid_count must be integer [4]")
        if not bool(torch.isfinite(self.center).all().item()):
            raise ValueError("normalization center must be finite")
        if not bool(
            (torch.isfinite(self.scale) & (self.scale > 0.0)).all().item()
        ):
            raise ValueError("normalization scale must be finite and positive")
        if not bool((self.valid_count >= 0).all().item()):
            raise ValueError("valid_count cannot be negative")

    def transform(
        self,
        path_log_odds: torch.Tensor,
        valid_paths: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not isinstance(path_log_odds, torch.Tensor)
            or not path_log_odds.is_floating_point()
            or path_log_odds.ndim < 2
            or int(path_log_odds.size(-1)) != 4
        ):
            raise ValueError("path_log_odds must be floating [..., 4]")
        if (
            not isinstance(valid_paths, torch.Tensor)
            or valid_paths.dtype != torch.bool
            or valid_paths.shape != path_log_odds.shape
        ):
            raise ValueError(
                "valid_paths must be boolean and match path_log_odds"
            )
        valid = valid_paths.to(device=path_log_odds.device)
        center = self.center.to(
            device=path_log_odds.device,
            dtype=path_log_odds.dtype,
        )
        scale = self.scale.to(
            device=path_log_odds.device,
            dtype=path_log_odds.dtype,
        )
        safe = torch.where(valid, path_log_odds, center)
        normalized = (safe - center) / scale
        normalized = torch.where(
            valid,
            normalized,
            torch.zeros_like(normalized),
        )
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError("normalized log-odds contain non-finite values")
        return normalized

    def as_summary(self) -> dict[str, Any]:
        return {
            "path_names": list(CARE_PATH_NAMES),
            "mu": [float(value) for value in self.center.tolist()],
            "sigma": [float(value) for value in self.scale.tolist()],
            "valid_count": [
                int(value) for value in self.valid_count.tolist()
            ],
            "invalid_path_placeholder": 0.0,
        }


def fit_valid_path_log_odds_normalizer(
    path_log_odds: torch.Tensor,
    valid_paths: torch.Tensor,
    *,
    minimum_scale: float = 1.0e-6,
) -> ValidPathLogOddsNormalizer:
    """Fit four independent statistics using valid values only."""

    if (
        not isinstance(path_log_odds, torch.Tensor)
        or not path_log_odds.is_floating_point()
        or path_log_odds.ndim < 2
        or int(path_log_odds.size(-1)) != 4
    ):
        raise ValueError("path_log_odds must be floating [..., 4]")
    if (
        not isinstance(valid_paths, torch.Tensor)
        or valid_paths.dtype != torch.bool
        or valid_paths.shape != path_log_odds.shape
    ):
        raise ValueError(
            "valid_paths must be boolean and match path_log_odds"
        )
    if (
        not math.isfinite(float(minimum_scale))
        or float(minimum_scale) <= 0.0
    ):
        raise ValueError("minimum_scale must be finite and positive")
    flattened = path_log_odds.detach().double().cpu().reshape(-1, 4)
    valid = valid_paths.detach().cpu().reshape(-1, 4)
    center = torch.zeros(4, dtype=torch.float64)
    scale = torch.ones(4, dtype=torch.float64)
    counts = valid.sum(dim=0).to(dtype=torch.long)
    for path_index in range(4):
        values = flattened[valid[:, path_index], path_index]
        if values.numel() == 0:
            continue
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("valid path log-odds must be finite")
        current_center = values.mean()
        current_sigma = torch.sqrt(
            torch.mean((values - current_center).square())
        )
        center[path_index] = current_center
        scale[path_index] = (
            current_sigma
            if float(current_sigma) >= float(minimum_scale)
            else 1.0
        )
    return ValidPathLogOddsNormalizer(
        center=center.float(),
        scale=scale.float(),
        valid_count=counts,
    )


@dataclass(frozen=True)
class HierarchicalBCEResult:
    loss: torch.Tensor
    per_path_loss: torch.Tensor
    per_view_loss: torch.Tensor
    per_sid_loss: torch.Tensor
    valid_views: torch.Tensor
    valid_sids: torch.Tensor
    valid_path_count_per_view: torch.Tensor
    valid_view_count_per_sid: torch.Tensor


def sid_view_valid_path_bce(
    correctness_probability: torch.Tensor,
    correctness_target: torch.Tensor,
    valid_paths: torch.Tensor,
) -> HierarchicalBCEResult:
    """Compute the frozen equal-SID/equal-view/equal-valid-path objective."""

    if (
        not isinstance(correctness_probability, torch.Tensor)
        or not correctness_probability.is_floating_point()
        or correctness_probability.ndim != 3
        or int(correctness_probability.size(-1)) != 4
    ):
        raise ValueError(
            "correctness_probability must be floating [SID, view, 4]"
        )
    if (
        not isinstance(correctness_target, torch.Tensor)
        or not correctness_target.is_floating_point()
        or correctness_target.shape != correctness_probability.shape
    ):
        raise ValueError(
            "correctness_target must be floating and match probability"
        )
    if (
        not isinstance(valid_paths, torch.Tensor)
        or valid_paths.dtype != torch.bool
        or valid_paths.shape != correctness_probability.shape
    ):
        raise ValueError(
            "valid_paths must be boolean and match probability"
        )
    valid = valid_paths.to(device=correctness_probability.device)
    target = correctness_target.to(
        device=correctness_probability.device,
        dtype=correctness_probability.dtype,
    )
    probability = correctness_probability
    if not bool(
        (
            torch.isfinite(probability[valid])
            & (probability[valid] >= 0.0)
            & (probability[valid] <= 1.0)
        ).all().item()
    ):
        raise ValueError("valid correctness probabilities must lie in [0, 1]")
    if not bool(
        (
            torch.isfinite(target[valid])
            & ((target[valid] == 0.0) | (target[valid] == 1.0))
        ).all().item()
    ):
        raise ValueError("valid correctness targets must be binary")
    safe_probability = torch.where(
        valid,
        probability,
        torch.full_like(probability, 0.5),
    )
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    per_path = F.binary_cross_entropy(
        safe_probability,
        safe_target,
        reduction="none",
    )
    path_count = valid.sum(dim=-1)
    valid_views = path_count > 0
    per_view = (
        (per_path * valid.to(per_path)).sum(dim=-1)
        / path_count.clamp_min(1).to(per_path)
    )
    view_count = valid_views.sum(dim=-1)
    valid_sids = view_count > 0
    per_sid = (
        (per_view * valid_views.to(per_view)).sum(dim=-1)
        / view_count.clamp_min(1).to(per_view)
    )
    if not bool(valid_sids.any().item()):
        raise ValueError("CARE risk BCE has no valid SID/view/path")
    loss = per_sid[valid_sids].mean()
    return HierarchicalBCEResult(
        loss=loss,
        per_path_loss=per_path,
        per_view_loss=per_view,
        per_sid_loss=per_sid,
        valid_views=valid_views,
        valid_sids=valid_sids,
        valid_path_count_per_view=path_count,
        valid_view_count_per_sid=view_count,
    )


def _set_head_normalization(
    head: torch.nn.Module,
    normalizer: ValidPathLogOddsNormalizer,
    device: torch.device,
) -> None:
    setter = getattr(head, "set_log_odds_normalization", None)
    if not callable(setter):
        raise TypeError(
            "CARE risk head must implement set_log_odds_normalization"
        )
    setter(
        normalizer.center.to(device=device),
        normalizer.scale.to(device=device),
    )


def _head_score_all(
    head: torch.nn.Module,
    normalized_log_odds: torch.Tensor,
    modality_alive: torch.Tensor,
) -> torch.Tensor:
    scorer = getattr(head, "score_all", None)
    if not callable(scorer):
        raise TypeError("CARE risk head must implement score_all")
    probability = scorer(normalized_log_odds, modality_alive)
    if (
        not isinstance(probability, torch.Tensor)
        or probability.shape != normalized_log_odds.shape
        or not probability.is_floating_point()
    ):
        raise ValueError(
            "CARE risk head score_all must return floating [B, 4]"
        )
    return probability


def _tensor_indices(
    indices: np.ndarray | Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(indices, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def _predict_head(
    head: torch.nn.Module,
    data: CareRiskCalibrationData,
    indices: np.ndarray,
    normalizer: ValidPathLogOddsNormalizer,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    head.eval()
    output = torch.zeros(
        len(indices),
        data.num_views,
        4,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_numpy = indices[start : start + int(batch_size)]
            source_index = _tensor_indices(
                batch_numpy,
                device=data.path_log_odds.device,
            )
            raw = data.path_log_odds.index_select(0, source_index)
            valid = data.valid_paths.index_select(0, source_index)
            alive = data.modality_alive.index_select(0, source_index)
            normalized = normalizer.transform(raw, valid).to(device=device)
            flattened = normalized.reshape(-1, 4)
            flat_alive = alive.to(device=device).reshape(-1, 3)
            probability = _head_score_all(
                head,
                flattened,
                flat_alive,
            ).reshape(len(batch_numpy), data.num_views, 4)
            probability = torch.where(
                valid.to(device=device),
                probability,
                torch.zeros_like(probability),
            )
            if not bool(torch.isfinite(probability).all().item()):
                raise FloatingPointError(
                    "CARE risk head emitted non-finite probabilities"
                )
            output[start : start + len(batch_numpy)] = (
                probability.detach().float().cpu()
            )
    return output


def _fit_head_fixed_epochs(
    head: torch.nn.Module,
    data: CareRiskCalibrationData,
    indices: np.ndarray,
    normalizer: ValidPathLogOddsNormalizer,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
) -> list[dict[str, float]]:
    """Fit exactly ``epochs`` mini-batch passes with no holdout feedback.

    A cache row is one SID and already contains every deterministic view and
    candidate path for that SID.  Consequently, a mini-batch update preserves
    the frozen SID -> view -> valid-path hierarchy while giving
    ``batch_size`` its ordinary optimization meaning.
    """

    if int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if (
        not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
        or not math.isfinite(float(gradient_clip))
        or float(gradient_clip) <= 0.0
    ):
        raise ValueError("CARE optimizer settings are invalid")
    if len(indices) <= 0:
        raise ValueError("CARE risk fit indices cannot be empty")
    head.to(device)
    _set_head_normalization(head, normalizer, device)
    trainable = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("CARE risk head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    valid_sid_mask = (
        data.valid_paths[
            _tensor_indices(indices, device=data.valid_paths.device)
        ]
        .any(dim=-1)
        .any(dim=-1)
    )
    total_valid_sids = int(valid_sid_mask.sum().item())
    if total_valid_sids <= 0:
        raise ValueError("CARE risk fit population has no valid paths")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    history: list[dict[str, float]] = []
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [
            int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        for epoch in range(1, int(epochs) + 1):
            head.train()
            order = torch.randperm(len(indices), generator=generator).numpy()
            observed_valid_sids = 0
            optimizer_steps = 0
            for start in range(0, len(order), int(batch_size)):
                local = order[start : start + int(batch_size)]
                batch_numpy = np.asarray(indices[local], dtype=np.int64)
                source_index = _tensor_indices(
                    batch_numpy,
                    device=data.path_log_odds.device,
                )
                raw = data.path_log_odds.index_select(0, source_index)
                valid = data.valid_paths.index_select(0, source_index)
                alive = data.modality_alive.index_select(0, source_index)
                target = data.correctness_targets.index_select(
                    0, source_index
                )
                batch_valid_sid = valid.any(dim=-1).any(dim=-1)
                if not bool(batch_valid_sid.any().item()):
                    continue
                normalized = normalizer.transform(raw, valid).to(
                    device=device
                )
                probability = _head_score_all(
                    head,
                    normalized.reshape(-1, 4),
                    alive.to(device=device).reshape(-1, 3),
                ).reshape(len(batch_numpy), data.num_views, 4)
                objective = sid_view_valid_path_bce(
                    probability,
                    target.to(device=device),
                    valid.to(device=device),
                )
                batch_valid_sids = int(objective.valid_sids.sum().item())
                optimizer.zero_grad(set_to_none=True)
                objective.loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(gradient_clip),
                )
                if not bool(torch.isfinite(gradient_norm).all().item()):
                    raise FloatingPointError(
                        "CARE risk head produced non-finite gradients"
                    )
                optimizer.step()
                observed_valid_sids += batch_valid_sids
                optimizer_steps += 1
            if observed_valid_sids != total_valid_sids:
                raise RuntimeError(
                    "CARE mini-batch epoch did not consume every valid SID: "
                    f"observed={observed_valid_sids}, "
                    f"expected={total_valid_sids}"
                )
            if optimizer_steps <= 0:
                raise RuntimeError(
                    "CARE mini-batch epoch produced no optimizer updates"
                )
            fitted_probability = _predict_head(
                head,
                data,
                np.asarray(indices, dtype=np.int64),
                normalizer,
                device=device,
                batch_size=batch_size,
            )
            fitted_target = data.correctness_targets[
                _tensor_indices(
                    indices,
                    device=data.correctness_targets.device,
                )
            ].cpu()
            fitted_valid = data.valid_paths[
                _tensor_indices(indices, device=data.valid_paths.device)
            ].cpu()
            fitted_loss = sid_view_valid_path_bce(
                fitted_probability,
                fitted_target,
                fitted_valid,
            ).loss
            history.append(
                {
                    "epoch": float(epoch),
                    "train_sid_view_path_bce": float(fitted_loss),
                    "optimizer_steps": float(optimizer_steps),
                }
            )
    return history


def _validate_group_labels(data: CareRiskCalibrationData) -> None:
    groups_by_label: dict[int, set[str]] = {0: set(), 1: set()}
    for group, raw_label in zip(data.groups, data.labels.tolist()):
        label = int(raw_label)
        groups_by_label[label].add(str(group))
    if min(len(values) for values in groups_by_label.values()) < (
        CARE_RISK_FOLDS
    ):
        raise ValueError(
            "three-fold CARE cross-fitting requires at least three package "
            "groups containing each class"
        )


def _sid_view_identity_payload(
    sids: Sequence[str],
    view_names: Sequence[str],
    protocol_seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "sid": str(sid),
            "view": str(view),
            "view_seed": deterministic_view_seed(
                str(sid),
                str(view),
                int(protocol_seed),
            ),
        }
        for sid in sids
        for view in view_names
    ]


@dataclass(frozen=True)
class CareRiskCrossFitResult:
    risk_head: torch.nn.Module
    oof_correctness_probability: torch.Tensor
    fold_assignment: torch.Tensor
    fold_state_dicts: tuple[dict[str, torch.Tensor], ...]
    final_normalizer: ValidPathLogOddsNormalizer
    summary: dict[str, Any]


def fit_care_path_risk_crossfit(
    risk_head: torch.nn.Module,
    data: CareRiskCalibrationData,
    *,
    device: torch.device | str = "cpu",
    folds: int = CARE_RISK_FOLDS,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    gradient_clip: float = 5.0,
    protocol_seed: int = 42,
) -> CareRiskCrossFitResult:
    """Fit leakage-free OOF CARE risk scores and the final deployable head."""

    if not isinstance(risk_head, torch.nn.Module):
        raise TypeError("risk_head must be a torch.nn.Module")
    if int(folds) != CARE_RISK_FOLDS:
        raise ValueError("CARE risk cross-fitting is frozen to exactly 3 folds")
    if isinstance(protocol_seed, bool) or not isinstance(
        protocol_seed, (int, np.integer)
    ):
        raise TypeError("protocol_seed must be an integer")
    _validate_group_labels(data)
    resolved_device = torch.device(device)
    labels = data.labels.detach().cpu().numpy().astype(np.int64)
    groups = np.asarray(data.groups, dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=CARE_RISK_FOLDS,
        shuffle=True,
        random_state=int(protocol_seed),
    )
    splits = list(
        splitter.split(
            np.zeros(data.num_sids, dtype=np.int64),
            labels,
            groups=groups,
        )
    )
    initial_state = copy.deepcopy(risk_head.state_dict())
    oof = torch.zeros(
        data.num_sids,
        data.num_views,
        4,
        dtype=torch.float32,
    )
    assignment = torch.full((data.num_sids,), -1, dtype=torch.long)
    fold_summaries: list[dict[str, Any]] = []
    fold_state_dicts: list[dict[str, torch.Tensor]] = []
    for fold_index, (train_indices, holdout_indices) in enumerate(splits):
        train_indices = np.asarray(train_indices, dtype=np.int64)
        holdout_indices = np.asarray(holdout_indices, dtype=np.int64)
        train_groups = {data.groups[int(index)] for index in train_indices}
        holdout_groups = {
            data.groups[int(index)] for index in holdout_indices
        }
        if train_groups & holdout_groups:
            raise RuntimeError("CARE OOF fold leaks package groups")
        train_tensor = _tensor_indices(
            train_indices,
            device=data.path_log_odds.device,
        )
        holdout_tensor = _tensor_indices(
            holdout_indices,
            device=data.path_log_odds.device,
        )
        normalizer = fit_valid_path_log_odds_normalizer(
            data.path_log_odds.index_select(0, train_tensor),
            data.valid_paths.index_select(0, train_tensor),
        )
        holdout_valid_count = (
            data.valid_paths.index_select(0, holdout_tensor)
            .reshape(-1, 4)
            .sum(dim=0)
            .cpu()
        )
        unsupported = (
            (normalizer.valid_count == 0) & (holdout_valid_count > 0)
        )
        if bool(unsupported.any().item()):
            names = [
                CARE_PATH_NAMES[index]
                for index in torch.where(unsupported)[0].tolist()
            ]
            raise ValueError(
                "OOF holdout contains valid paths absent from its training "
                f"fold: {names}"
            )
        fold_head = copy.deepcopy(risk_head)
        fold_head.load_state_dict(initial_state, strict=True)
        history = _fit_head_fixed_epochs(
            fold_head,
            data,
            train_indices,
            normalizer,
            device=resolved_device,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            gradient_clip=float(gradient_clip),
            seed=int(protocol_seed) + 1009 * (fold_index + 1),
        )
        holdout_probability = _predict_head(
            fold_head,
            data,
            holdout_indices,
            normalizer,
            device=resolved_device,
            batch_size=int(batch_size),
        )
        fold_state = clone_state_dict_to_cpu(fold_head.state_dict())
        fold_state_digest = tensor_state_dict_sha256(fold_state)
        fold_state_dicts.append(fold_state)
        oof[torch.as_tensor(holdout_indices, dtype=torch.long)] = (
            holdout_probability
        )
        if bool((assignment[holdout_indices] >= 0).any().item()):
            raise RuntimeError("CARE OOF SID received multiple fold predictions")
        assignment[holdout_indices] = int(fold_index)
        train_sids = [data.sids[int(index)] for index in train_indices]
        holdout_sids = [
            data.sids[int(index)] for index in holdout_indices
        ]
        train_sid_views = _sid_view_identity_payload(
            train_sids,
            data.view_names,
            int(protocol_seed),
        )
        holdout_sid_views = _sid_view_identity_payload(
            holdout_sids,
            data.view_names,
            int(protocol_seed),
        )
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "num_train_sids": len(train_sids),
                "num_holdout_sids": len(holdout_sids),
                "num_train_groups": len(train_groups),
                "num_holdout_groups": len(holdout_groups),
                "train_sids": train_sids,
                "holdout_sids": holdout_sids,
                "train_sid_sha256": _stable_digest(train_sids),
                "holdout_sid_sha256": _stable_digest(holdout_sids),
                "train_group_sha256": _stable_digest(
                    sorted(train_groups)
                ),
                "holdout_group_sha256": _stable_digest(
                    sorted(holdout_groups)
                ),
                "train_sid_view_sha256": _stable_digest(train_sid_views),
                "holdout_sid_view_sha256": _stable_digest(
                    holdout_sid_views
                ),
                "view_names": list(data.view_names),
                "path_names": list(CARE_PATH_NAMES),
                "normalization": normalizer.as_summary(),
                "state_dict_sha256": fold_state_digest,
                "fixed_epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "gradient_clip": float(gradient_clip),
                "optimizer": "adamw",
                "optimizer_update_unit": "mini_batch",
                "history": history,
                "holdout_used_for_optimization": False,
                "holdout_used_for_early_stopping": False,
                "holdout_used_for_structure_selection": False,
                "group_disjoint": True,
            }
        )
        del fold_head
    if bool((assignment < 0).any().item()):
        raise RuntimeError("CARE OOF predictions are incomplete")
    if not bool(torch.isfinite(oof).all().item()):
        raise RuntimeError("CARE OOF probabilities are non-finite")

    full_normalizer = fit_valid_path_log_odds_normalizer(
        data.path_log_odds,
        data.valid_paths,
    )
    risk_head.load_state_dict(initial_state, strict=True)
    final_history = _fit_head_fixed_epochs(
        risk_head,
        data,
        np.arange(data.num_sids, dtype=np.int64),
        full_normalizer,
        device=resolved_device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        gradient_clip=float(gradient_clip),
        seed=int(protocol_seed) + 7919,
    )
    risk_head.eval()
    for parameter in risk_head.parameters():
        parameter.requires_grad_(False)
    final_state_digest = tensor_state_dict_sha256(risk_head.state_dict())
    all_sid_views = _sid_view_identity_payload(
        data.sids,
        data.view_names,
        int(protocol_seed),
    )
    summary = {
        "protocol_id": CARE_RISK_PROTOCOL_ID,
        "target": "fixed_path_hard_rule_correctness",
        "hard_prediction_rule": "malware_iff_log_odds_greater_equal_zero",
        "objective": "sid_then_view_then_valid_path_unweighted_bce",
        "folds": CARE_RISK_FOLDS,
        "fixed_epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip": float(gradient_clip),
        "optimizer": "adamw",
        "optimizer_update_unit": "mini_batch",
        "early_stopping": False,
        "structure_selection": False,
        "folds_summary": fold_summaries,
        "oof_population": "routing_cal_fold_holdout_only",
        "oof_purpose": "leakage_free_risk_and_routing_diagnostics_only",
        "oof_used_for_downstream_selection": False,
        "final_refit_population": "all_routing_cal",
        "final_normalization": full_normalizer.as_summary(),
        "final_state_dict_sha256": final_state_digest,
        "final_history": final_history,
        "routing_cal_sids": list(data.sids),
        "routing_cal_sid_sha256": _stable_digest(list(data.sids)),
        "routing_cal_sid_view_sha256": _stable_digest(all_sid_views),
        "view_names": list(data.view_names),
        "path_names": list(CARE_PATH_NAMES),
        "protocol_seed": int(protocol_seed),
    }
    return CareRiskCrossFitResult(
        risk_head=risk_head,
        oof_correctness_probability=oof,
        fold_assignment=assignment,
        fold_state_dicts=tuple(fold_state_dicts),
        final_normalizer=full_normalizer,
        summary=summary,
    )


# Stable integration name used by fusion.care_train.
CareRiskCalibrationCache = CareRiskCalibrationData


def _resolve_care_risk_head(model_or_head: Any) -> torch.nn.Module:
    if isinstance(model_or_head, torch.nn.Module) and callable(
        getattr(model_or_head, "score_all", None)
    ):
        return model_or_head
    direct_names = (
        "care_risk_head",
        "care_path_risk_head",
        "path_risk_head",
        "risk_head",
    )
    for name in direct_names:
        value = getattr(model_or_head, name, None)
        if isinstance(value, torch.nn.Module) and callable(
            getattr(value, "score_all", None)
        ):
            return value
    care_fusion = getattr(model_or_head, "care_fusion", None)
    if care_fusion is not None:
        for name in direct_names:
            value = getattr(care_fusion, name, None)
            if isinstance(value, torch.nn.Module) and callable(
                getattr(value, "score_all", None)
            ):
                return value
    raise TypeError(
        "model_or_head must be a CARE risk head or expose one through "
        "care_risk_head/care_path_risk_head/path_risk_head/risk_head"
    )


def fit_care_risk_crossfit(
    model_or_head: Any,
    cached: CareRiskCalibrationCache,
    **kwargs: Any,
) -> CareRiskCrossFitResult:
    """Stable wrapper returning fitted head, OOF probabilities, and summary."""

    if not isinstance(cached, CareRiskCalibrationData):
        raise TypeError("cached must be a CareRiskCalibrationCache")
    return fit_care_path_risk_crossfit(
        _resolve_care_risk_head(model_or_head),
        cached,
        **kwargs,
    )


@dataclass(frozen=True)
class AtomicScoreGroup:
    score: float
    indices: tuple[int, ...]


def atomic_score_groups(
    scores: torch.Tensor,
    eligible: torch.Tensor | None = None,
) -> tuple[AtomicScoreGroup, ...]:
    """Group exactly equal eligible scores before any threshold search."""

    if (
        not isinstance(scores, torch.Tensor)
        or not scores.is_floating_point()
        or scores.ndim != 1
    ):
        raise ValueError("scores must be a one-dimensional floating tensor")
    if eligible is None:
        eligible = torch.ones_like(scores, dtype=torch.bool)
    if (
        not isinstance(eligible, torch.Tensor)
        or eligible.dtype != torch.bool
        or eligible.shape != scores.shape
    ):
        raise ValueError("eligible must be boolean and match scores")
    selected_indices = torch.where(eligible)[0]
    selected_scores = scores[selected_indices]
    if not bool(torch.isfinite(selected_scores).all().item()):
        raise ValueError("eligible scores must be finite")
    ordered = sorted(
        (
            (float(scores[index].item()), int(index))
            for index in selected_indices.tolist()
        ),
        key=lambda item: (item[0], item[1]),
    )
    groups: list[AtomicScoreGroup] = []
    for score, index in ordered:
        if groups and score == groups[-1].score:
            previous = groups[-1]
            groups[-1] = AtomicScoreGroup(
                score=previous.score,
                indices=(*previous.indices, index),
            )
        else:
            groups.append(AtomicScoreGroup(score=score, indices=(index,)))
    return tuple(groups)


@dataclass(frozen=True)
class AtomicCRCThresholdResult:
    threshold: float
    accepted: torch.Tensor
    feasible: bool
    corrected_risk: float
    empirical_risk: float
    accepted_count: int
    accepted_loss_count: int
    target_population_count: int
    accepted_group_count: int
    total_group_count: int
    risk_level: float

    def as_summary(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "acceptance_comparison": (
                "eligible and path_risk_score <= threshold"
            ),
            "score_ties_are_atomic": True,
            "feasible": bool(self.feasible),
            "corrected_risk": float(self.corrected_risk),
            "empirical_risk": float(self.empirical_risk),
            "accepted_count": int(self.accepted_count),
            "accepted_loss_count": int(self.accepted_loss_count),
            "target_population_count": int(
                self.target_population_count
            ),
            "accepted_group_count": int(self.accepted_group_count),
            "total_group_count": int(self.total_group_count),
            "risk_level": float(self.risk_level),
            "crc_correction": "(accepted_loss_count + 1) / "
            "(target_population_count + 1)",
        }


def fit_atomic_crc_risk_threshold(
    risk_scores: torch.Tensor,
    loss_events: torch.Tensor,
    target_population: torch.Tensor,
    *,
    risk_level: float,
    eligible: torch.Tensor | None = None,
) -> AtomicCRCThresholdResult:
    """Select the largest tie-atomic low-risk acceptance set under CRC."""

    if (
        not isinstance(risk_scores, torch.Tensor)
        or not risk_scores.is_floating_point()
        or risk_scores.ndim != 1
    ):
        raise ValueError("risk_scores must be floating [N]")
    for name, value in (
        ("loss_events", loss_events),
        ("target_population", target_population),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.shape != risk_scores.shape
        ):
            raise ValueError(f"{name} must be boolean and match risk_scores")
    if bool((loss_events & ~target_population).any().item()):
        raise ValueError("loss_events must be inside target_population")
    if (
        not math.isfinite(float(risk_level))
        or not 0.0 < float(risk_level) < 1.0
    ):
        raise ValueError("risk_level must lie strictly between zero and one")
    if eligible is None:
        eligible = torch.ones_like(risk_scores, dtype=torch.bool)
    if (
        not isinstance(eligible, torch.Tensor)
        or eligible.dtype != torch.bool
        or eligible.shape != risk_scores.shape
    ):
        raise ValueError("eligible must be boolean and match risk_scores")
    groups = atomic_score_groups(risk_scores, eligible)
    target_count = int(target_population.sum().item())
    if target_count <= 0:
        raise ValueError("CRC requires a non-empty target population")
    denominator = float(target_count + 1)
    accepted = torch.zeros_like(eligible)
    accepted_losses = 0
    accepted_group_count = 0
    feasible = (1.0 / denominator) <= float(risk_level)
    if feasible:
        for group in groups:
            group_index = torch.as_tensor(
                group.indices,
                dtype=torch.long,
                device=loss_events.device,
            )
            candidate_losses = accepted_losses + int(
                loss_events[group_index].sum().item()
            )
            candidate_corrected = (candidate_losses + 1.0) / denominator
            # Inclusive boundary: exact equality satisfies the CRC constraint.
            if candidate_corrected <= float(risk_level):
                accepted[group_index.to(accepted.device)] = True
                accepted_losses = candidate_losses
                accepted_group_count += 1
            else:
                # The numerator cannot decrease for later nested candidates.
                break
    corrected_risk = (accepted_losses + 1.0) / denominator
    empirical_risk = accepted_losses / float(target_count)
    threshold = (
        groups[accepted_group_count - 1].score
        if accepted_group_count > 0
        else float("-inf")
    )
    return AtomicCRCThresholdResult(
        threshold=float(threshold),
        accepted=accepted,
        feasible=bool(feasible),
        corrected_risk=float(corrected_risk),
        empirical_risk=float(empirical_risk),
        accepted_count=int(accepted.sum().item()),
        accepted_loss_count=int(accepted_losses),
        target_population_count=int(target_count),
        accepted_group_count=int(accepted_group_count),
        total_group_count=len(groups),
        risk_level=float(risk_level),
    )


@dataclass(frozen=True)
class CorrectnessCRCThresholdResult:
    """CARE-facing CRC result for the correctness score ``q``."""

    lambda_threshold: float
    risk_threshold: float
    accepted: torch.Tensor
    feasible: bool
    crc_status: str
    corrected_risk: float
    empirical_risk: float
    accepted_count: int
    accepted_fn_count: int
    n_malware: int
    accepted_group_count: int
    total_group_count: int
    alpha: float

    def as_summary(self) -> dict[str, Any]:
        return {
            "lambda": float(self.lambda_threshold),
            "equivalent_risk_threshold": float(self.risk_threshold),
            "acceptance_comparison": (
                "eligible and correctness_q >= lambda"
            ),
            "risk_conversion": "risk_score = 1 - correctness_q",
            "score_ties_are_atomic": True,
            "crc_status": str(self.crc_status),
            "feasible": bool(self.feasible),
            "corrected_risk": float(self.corrected_risk),
            "empirical_risk": float(self.empirical_risk),
            "accepted_count": int(self.accepted_count),
            "accepted_fn_count": int(self.accepted_fn_count),
            "N_malware": int(self.n_malware),
            "accepted_group_count": int(self.accepted_group_count),
            "total_group_count": int(self.total_group_count),
            "alpha": float(self.alpha),
            "finite_sample_floor": (
                1.0 / float(self.n_malware + 1)
                if self.n_malware > 0
                else None
            ),
            "crc_correction": "(accepted_fn_count + 1) / "
            "(N_malware + 1)",
        }


def fit_atomic_crc_correctness_threshold(
    correctness_scores: torch.Tensor,
    false_negative_events: torch.Tensor,
    malware_population: torch.Tensor,
    *,
    alpha: float,
    eligible: torch.Tensor | None = None,
) -> CorrectnessCRCThresholdResult:
    """Fit CARE's ``benign iff q >= lambda`` tie-atomic CRC rule.

    The lower-is-safer helper remains available for generic risk scores.  CARE
    exposes correctness ``q``, so this wrapper explicitly maps
    ``risk_score = 1 - q`` and returns the equivalent correctness threshold.
    """

    if (
        not isinstance(correctness_scores, torch.Tensor)
        or not correctness_scores.is_floating_point()
        or correctness_scores.ndim != 1
    ):
        raise ValueError("correctness_scores must be floating [N]")
    for name, value in (
        ("false_negative_events", false_negative_events),
        ("malware_population", malware_population),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.shape != correctness_scores.shape
        ):
            raise ValueError(
                f"{name} must be boolean and match correctness_scores"
            )
    if bool((false_negative_events & ~malware_population).any().item()):
        raise ValueError(
            "false_negative_events must be inside malware_population"
        )
    if (
        not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must lie strictly between zero and one")
    if eligible is None:
        eligible = torch.ones_like(correctness_scores, dtype=torch.bool)
    if (
        not isinstance(eligible, torch.Tensor)
        or eligible.dtype != torch.bool
        or eligible.shape != correctness_scores.shape
    ):
        raise ValueError(
            "eligible must be boolean and match correctness_scores"
        )
    eligible_scores = correctness_scores[eligible]
    if not bool(
        (
            torch.isfinite(eligible_scores)
            & (eligible_scores >= 0.0)
            & (eligible_scores <= 1.0)
        ).all().item()
    ):
        raise ValueError(
            "eligible correctness_scores must be finite probabilities"
        )
    n_malware = int(malware_population.sum().item())
    empty_acceptance = torch.zeros_like(eligible)
    if n_malware == 0:
        return CorrectnessCRCThresholdResult(
            lambda_threshold=float("inf"),
            risk_threshold=float("-inf"),
            accepted=empty_acceptance,
            feasible=False,
            crc_status="failure_no_malware",
            corrected_risk=float("nan"),
            empirical_risk=float("nan"),
            accepted_count=0,
            accepted_fn_count=0,
            n_malware=0,
            accepted_group_count=0,
            total_group_count=len(
                atomic_score_groups(correctness_scores, eligible)
            ),
            alpha=float(alpha),
        )
    finite_sample_floor = 1.0 / float(n_malware + 1)
    if finite_sample_floor > float(alpha):
        return CorrectnessCRCThresholdResult(
            lambda_threshold=float("inf"),
            risk_threshold=float("-inf"),
            accepted=empty_acceptance,
            feasible=False,
            crc_status="infeasible_insufficient_malware",
            corrected_risk=float(finite_sample_floor),
            empirical_risk=0.0,
            accepted_count=0,
            accepted_fn_count=0,
            n_malware=n_malware,
            accepted_group_count=0,
            total_group_count=len(
                atomic_score_groups(correctness_scores, eligible)
            ),
            alpha=float(alpha),
        )
    generic = fit_atomic_crc_risk_threshold(
        1.0 - correctness_scores,
        false_negative_events,
        malware_population,
        risk_level=float(alpha),
        eligible=eligible,
    )
    if bool(generic.accepted.any().item()):
        lambda_threshold = float(
            correctness_scores[generic.accepted].min().item()
        )
        risk_threshold = 1.0 - lambda_threshold
    else:
        lambda_threshold = float("inf")
        risk_threshold = float("-inf")
    # Reconstruct the public q-rule and assert it preserves every atomic group.
    public_acceptance = (
        eligible
        & (correctness_scores >= float(lambda_threshold))
    )
    if not torch.equal(public_acceptance, generic.accepted):
        raise RuntimeError(
            "q-to-risk threshold conversion changed the acceptance set"
        )
    return CorrectnessCRCThresholdResult(
        lambda_threshold=float(lambda_threshold),
        risk_threshold=float(risk_threshold),
        accepted=generic.accepted,
        feasible=True,
        crc_status="feasible",
        corrected_risk=float(generic.corrected_risk),
        empirical_risk=float(generic.empirical_risk),
        accepted_count=int(generic.accepted_count),
        accepted_fn_count=int(generic.accepted_loss_count),
        n_malware=n_malware,
        accepted_group_count=int(generic.accepted_group_count),
        total_group_count=int(generic.total_group_count),
        alpha=float(alpha),
    )


__all__ = [
    "CARE_PATH_NAMES",
    "CARE_RISK_FOLDS",
    "CARE_RISK_PROTOCOL_ID",
    "AtomicCRCThresholdResult",
    "AtomicScoreGroup",
    "CareRiskCalibrationCache",
    "CareRiskCalibrationData",
    "CareRiskCrossFitResult",
    "CorrectnessCRCThresholdResult",
    "HierarchicalBCEResult",
    "ValidPathLogOddsNormalizer",
    "atomic_score_groups",
    "care_path_validity",
    "clone_state_dict_to_cpu",
    "deterministic_view_seed",
    "deterministic_view_spec",
    "fit_atomic_crc_risk_threshold",
    "fit_atomic_crc_correctness_threshold",
    "fit_care_path_risk_crossfit",
    "fit_care_risk_crossfit",
    "fit_valid_path_log_odds_normalizer",
    "fixed_path_correctness_targets",
    "fixed_path_predictions",
    "sid_view_valid_path_bce",
    "tensor_state_dict_sha256",
]
