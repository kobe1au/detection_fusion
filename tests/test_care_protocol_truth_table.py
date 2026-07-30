from __future__ import annotations

import pytest
import torch

from fusion.care_fusion import path_availability
from fusion.care_train import (
    CARE_PATHS,
    CareCachedView,
    _fit_care_crc,
    _route_cached,
    _selection_score,
    _selective_metrics,
)


ROUTING_ENABLED = {
    "enabled": True,
    "route_on_all_samples": False,
}


def _path_logits_from_predictions(
    predictions: list[list[int]],
) -> torch.Tensor:
    """Encode an explicit CARE hard-prediction truth table as logits."""

    prediction_tensor = torch.tensor(predictions, dtype=torch.long)
    if (
        prediction_tensor.ndim != 2
        or prediction_tensor.size(1) != len(CARE_PATHS)
        or not bool(
            (
                (prediction_tensor == 0)
                | (prediction_tensor == 1)
            ).all().item()
        )
    ):
        raise ValueError("predictions must be binary with shape [N, 4]")
    benign = torch.tensor([2.0, -2.0])
    malware = torch.tensor([-2.0, 2.0])
    return torch.where(
        prediction_tensor.unsqueeze(-1).bool(),
        malware,
        benign,
    )


def _all_alive(size: int) -> torch.Tensor:
    return torch.ones((size, 3), dtype=torch.bool)


def test_all_paths_agree_always_selects_agm() -> None:
    path_logits = _path_logits_from_predictions(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    available = path_availability(_all_alive(2))
    # Even overwhelmingly larger pair scores cannot override the AGM anchor
    # when the pair makes the same hard prediction.
    correctness = torch.tensor(
        [
            [0.10, 0.99, 0.98, 0.97],
            [0.20, 0.97, 0.98, 0.99],
        ],
        dtype=torch.float32,
    )

    routed = _route_cached(
        path_logits,
        available,
        correctness,
        ROUTING_ENABLED,
    )

    assert routed.selected_path_index.tolist() == [0, 0]
    assert routed.eligible.tolist() == [True, True]
    assert torch.equal(routed.selected_logits, path_logits[:, 0])
    assert routed.selected_score.tolist() == pytest.approx([0.10, 0.20])
    assert not bool(routed.disagreement_with_agm.any().item())


def test_disagreeing_pair_with_lower_or_equal_score_does_not_switch() -> None:
    path_logits = _path_logits_from_predictions(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    available = path_availability(_all_alive(2))
    correctness = torch.tensor(
        [
            [0.70, 0.69, 0.99, 0.99],
            [0.70, 0.70, 0.99, 0.99],
        ],
        dtype=torch.float32,
    )

    routed = _route_cached(
        path_logits,
        available,
        correctness,
        ROUTING_ENABLED,
    )

    # AM/GM have larger scores but agree with AGM and are therefore not route
    # candidates. AG disagrees, but a switch requires a *strictly* larger
    # correctness score than AGM.
    assert routed.disagreement_with_agm.tolist() == [
        [True, False, False],
        [True, False, False],
    ]
    assert routed.selected_path_index.tolist() == [0, 0]
    assert routed.selected_score.tolist() == pytest.approx([0.70, 0.70])


def test_equal_high_scoring_disagreeing_pairs_use_ag_am_gm_priority() -> None:
    path_logits = _path_logits_from_predictions(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    available = path_availability(_all_alive(3))
    correctness = torch.tensor(
        [
            [0.50, 0.80, 0.80, 0.80],
            [0.50, 0.40, 0.80, 0.80],
            [0.50, 0.40, 0.40, 0.80],
        ],
        dtype=torch.float32,
    )

    routed = _route_cached(
        path_logits,
        available,
        correctness,
        ROUTING_ENABLED,
    )

    # Stable public order is AGM, AG, AM, GM. Once AGM is strictly beaten,
    # equal pair scores resolve AG before AM before GM.
    assert routed.selected_path_index.tolist() == [1, 2, 3]
    assert routed.selected_score.tolist() == pytest.approx([0.80] * 3)


def test_exactly_two_alive_selects_the_unique_available_pair() -> None:
    modality_alive = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )
    available = path_availability(modality_alive)
    path_logits = _path_logits_from_predictions(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Scores are deliberately adversarial: structural fallback must ignore
    # them and select the sole available two-modality path.
    correctness = torch.tensor(
        [
            [0.99, 0.01, 0.99, 0.99],
            [0.99, 0.99, 0.01, 0.99],
            [0.99, 0.99, 0.99, 0.01],
        ],
        dtype=torch.float32,
    )

    routed = _route_cached(
        path_logits,
        available,
        correctness,
        ROUTING_ENABLED,
    )

    assert available.tolist() == [
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]
    assert routed.selected_path_index.tolist() == [1, 2, 3]
    assert routed.eligible.tolist() == [True, True, True]
    assert routed.prediction.tolist() == [1, 1, 1]
    assert routed.selected_score.tolist() == pytest.approx([0.01] * 3)


def test_zero_or_one_alive_is_structurally_rejected_and_zeroed() -> None:
    modality_alive = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        dtype=torch.bool,
    )
    available = path_availability(modality_alive)
    path_logits = _path_logits_from_predictions(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    correctness = torch.full((4, 4), 0.99, dtype=torch.float32)

    routed = _route_cached(
        path_logits,
        available,
        correctness,
        ROUTING_ENABLED,
    )

    assert not bool(available.any().item())
    assert routed.selected_path_index.tolist() == [-1, -1, -1, -1]
    assert routed.eligible.tolist() == [False, False, False, False]
    assert routed.prediction.tolist() == [-1, -1, -1, -1]
    assert torch.equal(
        routed.selected_logits,
        torch.zeros_like(routed.selected_logits),
    )
    assert torch.equal(
        routed.selected_score,
        torch.zeros_like(routed.selected_score),
    )


def test_crc_acceptance_truth_table_matches_selective_metrics() -> None:
    # Rows:
    # 0 malware prediction (always accepted, even with a low q);
    # 1-2 benign predictions at/above the fitted q threshold;
    # 3 malware false negative just below the threshold;
    # 4 lower-q benign prediction;
    # 5 structural rejection with a high placeholder score.
    labels = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.long)
    modality_alive = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, False, False],
        ],
        dtype=torch.bool,
    )
    path_logits = _path_logits_from_predictions(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    path_available = path_availability(modality_alive)
    selected_q = torch.tensor(
        [0.01, 0.90, 0.80, 0.70, 0.60, 0.99],
        dtype=torch.float32,
    )
    correctness = selected_q.unsqueeze(-1).expand(-1, 4).clone()
    routed = _route_cached(
        path_logits,
        path_available,
        correctness,
        ROUTING_ENABLED,
    )
    cached = CareCachedView(
        sids=tuple(f"sid-{index}" for index in range(labels.numel())),
        groups=tuple(f"group-{index}" for index in range(labels.numel())),
        labels=labels,
        path_logits=path_logits,
        modality_alive=modality_alive,
        path_available=path_available,
        output_digests=tuple("" for _ in range(labels.numel())),
    )
    decision_score = _selection_score(
        routed,
        "care_selected_path_correctness",
    )
    selective_cfg = {
        "threshold_score": "care_selected_path_correctness",
        "risk_level": 0.25,
        "min_calibration_malware": 1,
        "require_feasible": True,
    }

    crc, decision_summary, decision_rows = _fit_care_crc(
        cached,
        routed,
        decision_score,
        selective_cfg,
    )
    metrics = _selective_metrics(
        cached,
        routed,
        decision_score,
        crc.lambda_threshold,
        guarantee_scope="natural_distribution_only",
    )

    assert crc.feasible is True
    assert crc.lambda_threshold == pytest.approx(0.80)
    assert routed.prediction.tolist() == [1, 0, 0, 0, 0, -1]
    assert [row["accepted"] for row in decision_rows] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert decision_rows[0]["decision_score"] == pytest.approx(0.01)
    assert decision_rows[0]["accepted"] is True
    assert decision_rows[2]["decision_score"] == pytest.approx(
        crc.lambda_threshold
    )
    assert decision_rows[2]["accepted"] is True
    assert decision_rows[5]["decision_score"] == pytest.approx(0.0)
    assert decision_rows[5]["accepted"] is False

    assert decision_summary["overall_accepted_count"] == 3
    assert decision_summary["overall_reject_count"] == 3
    assert decision_summary["accepted_fn_count_audit"] == 0
    assert decision_summary["corrected_risk_audit"] == pytest.approx(0.25)
    assert metrics["accepted_count"] == decision_summary[
        "overall_accepted_count"
    ]
    assert metrics["rejected_count"] == decision_summary[
        "overall_reject_count"
    ]
    assert metrics["accepted_fn_count"] == decision_summary[
        "accepted_fn_count_audit"
    ]
    assert metrics[
        "corrected_malware_accepted_fn_risk"
    ] == pytest.approx(decision_summary["corrected_risk_audit"])
    assert metrics["structural_reject_count"] == 1
    assert metrics["threshold_lambda"] == pytest.approx(
        crc.lambda_threshold
    )
