"""Shared binary operating-point fitting for validation-only model selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


MACRO_F1_THRESHOLD_SELECTION_RULE = "macro_f1_unconstrained_v1"


def fit_binary_macro_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Fit the deterministic macro-F1 threshold used by every paper method.

    Exact macro-F1 ties prefer the neutral 0.5 boundary and then the smaller
    threshold. Including 0.5 explicitly is necessary because it may lie inside
    a wide gap between observed probabilities and therefore not be one of the
    adjacent midpoints.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.size == 0 or probabilities.size != labels.size:
        raise ValueError("threshold fitting requires aligned non-empty rows")
    if set(int(value) for value in np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("threshold fitting requires both binary classes")
    if not np.isfinite(probabilities).all() or not bool(
        ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    ):
        raise ValueError(
            "threshold fitting requires finite probabilities in [0, 1]"
        )

    unique = sorted(set(float(value) for value in probabilities.tolist()))
    candidates = [unique[0], 0.5]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(unique[:-1], unique[1:])
    )
    candidates.append(math.nextafter(unique[-1], float("inf")))
    candidates = sorted(set(candidates))

    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in candidates:
        prediction = (probabilities >= float(threshold)).astype(np.int64)
        macro_f1 = float(
            f1_score(
                labels,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        key = (
            macro_f1,
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "macro_f1": macro_f1,
                "malware_recall": float(
                    recall_score(
                        labels,
                        prediction,
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(labels, prediction)),
            }
    if best is None:
        raise RuntimeError("no finite macro-F1 threshold candidate exists")
    fixed_prediction = (probabilities >= 0.5).astype(np.int64)
    return {
        "enabled": True,
        "objective": "macro_f1",
        "selection_rule": MACRO_F1_THRESHOLD_SELECTION_RULE,
        "constraint": "none",
        "num_calibration": int(labels.size),
        "num_calibration_benign": int((labels == 0).sum()),
        "num_calibration_malware": int((labels == 1).sum()),
        "num_candidates": int(len(candidates)),
        "fixed_0_5_macro_f1": float(
            f1_score(
                labels,
                fixed_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "fixed_0_5_malware_recall": float(
            recall_score(
                labels,
                fixed_prediction,
                pos_label=1,
                zero_division=0,
            )
        ),
        **best,
    }


__all__ = [
    "MACRO_F1_THRESHOLD_SELECTION_RULE",
    "fit_binary_macro_f1_threshold",
]
