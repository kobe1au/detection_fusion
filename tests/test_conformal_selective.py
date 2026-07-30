import numpy as np
import pytest
import torch

from fusion.train import (
    _batch_selective_eligibility,
    _batch_selective_score,
    _row_selective_eligible,
    _selective_ranking_metrics,
    _selective_score_type,
    build_risk_coverage_curve,
    conformal_selective_metrics,
    fit_conformal_thresholds,
    fit_malware_classification_threshold,
    fit_rejection_threshold,
    fit_risk_control_thresholds,
    risk_control_selective_metrics,
)
from fusion.thresholds import fit_binary_macro_f1_threshold


def _rows(probs_labels):
    rows = []
    for p, y in probs_labels:
        rows.append(
            {
                "prob_malware": p,
                "label": y,
                "pred": int(p >= 0.5),
                "selective_eligible": 1,
            }
        )
    return rows


def test_fit_conformal_thresholds_disabled_returns_none():
    assert fit_conformal_thresholds([], {"enabled": False}) is None
    assert fit_conformal_thresholds([], {"enabled": True, "mode": "threshold"}) is None


def test_classification_threshold_maximizes_unconstrained_macro_f1():
    rows = _rows(
        [(0.10, 0), (0.20, 0), (0.30, 0), (0.40, 0)]
        + [(0.15, 1), (0.60, 1), (0.70, 1), (0.80, 1)]
    )
    fitted = fit_malware_classification_threshold(
        rows,
        {
            "enabled": True,
            "objective": "macro_f1",
            "selection_rule": "macro_f1_unconstrained_v1",
        },
    )

    assert fitted is not None
    assert fitted["threshold"] == pytest.approx(0.5)
    assert fitted["malware_recall"] == pytest.approx(0.75)
    assert fitted["macro_f1"] >= fitted["fixed_0_5_macro_f1"]
    assert fitted["selection_rule"] == "macro_f1_unconstrained_v1"
    assert fitted["constraint"] == "none"
    assert fitted["calibration_split"] == "val_posthoc_calibration"


def test_shared_threshold_fitter_explicitly_prefers_neutral_boundary_in_gap():
    # Every threshold in (0.2, 0.8] yields the same perfect predictions. The
    # observed-probability midpoint is 0.5 here, but adding extra outer values
    # creates multiple equivalent midpoint candidates and audits that 0.5 is
    # explicitly retained as the neutral tie-break.
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.05, 0.10, 0.70, 0.95], dtype=np.float64)
    fitted = fit_binary_macro_f1_threshold(labels, probabilities)

    assert fitted["threshold"] == pytest.approx(0.5)
    assert fitted["macro_f1"] == pytest.approx(1.0)


def test_classification_threshold_rejects_removed_recall_constraint():
    with pytest.raises(ValueError, match="min_malware_recall was removed"):
        fit_malware_classification_threshold(
            _rows([(0.1, 0), (0.9, 1)]),
            {
                "enabled": True,
                "objective": "macro_f1",
                "min_malware_recall": 0.9,
            },
        )


def test_risk_coverage_curve_reports_exact_threshold_operating_points():
    rows = [
        {"split": "test_clean", "acceptance_score": 0.9, "label": 1, "pred": 1, "selective_eligible": 1},
        {"split": "test_clean", "acceptance_score": 0.8, "label": 1, "pred": 0, "selective_eligible": 1},
        {"split": "test_clean", "acceptance_score": 0.7, "label": 0, "pred": 0, "selective_eligible": 1},
        {"split": "other", "acceptance_score": 0.6, "label": 0, "pred": 1, "selective_eligible": 1},
    ]
    curve = build_risk_coverage_curve(rows)
    clean = [point for point in curve if point["split"] == "test_clean"]

    assert [point["coverage"] for point in clean] == pytest.approx(
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    assert clean[-1]["selective_risk"] == pytest.approx(1.0 / 3.0)
    assert clean[-1]["accepted_fn_risk_among_malware"] == pytest.approx(0.5)
    assert all(
        point["acceptance_comparison"]
        == "selective_eligible and score > threshold"
        for point in clean
    )


def test_risk_coverage_curve_keeps_an_all_ineligible_split_auditable():
    curve = build_risk_coverage_curve(
        [
            {
                "split": "all_dead",
                "acceptance_score": 0.5,
                "label": 1,
                "pred": 0,
                "selective_eligible": 0,
            }
        ]
    )

    assert len(curve) == 1
    assert curve[0]["split"] == "all_dead"
    assert curve[0]["coverage"] == 0.0
    assert curve[0]["num_ineligible_forced_reject"] == 1
    assert curve[0]["selective_metrics_defined"] is False


def test_fit_conformal_thresholds_class_conditional():
    rows = _rows([(0.6 + 0.01 * i, 1) for i in range(10)] + [(0.4 - 0.01 * i, 0) for i in range(10)])
    th = fit_conformal_thresholds(
        rows, {"enabled": True, "mode": "conformal", "target_coverage": 0.8}
    )
    assert th is not None
    assert th["class_conditional"] is True
    assert th["alpha"] == pytest.approx(0.2)
    assert 0.0 <= th["q_malware"] <= 1.0
    assert 0.0 <= th["q_benign"] <= 1.0


def test_tiny_calibration_set_yields_vacuous_threshold():
    # Conformal is conservative: too few points to certify -> never reject.
    rows = _rows([(0.95, 1), (0.9, 1), (0.85, 1)])
    th = fit_conformal_thresholds(
        rows, {"enabled": True, "mode": "conformal", "target_coverage": 0.8}
    )
    assert th["q_malware"] == float("inf")


def test_conformal_coverage_meets_target_on_calibration_data():
    # Well-separated calibration set -> empirical coverage should hit ~1-alpha.
    rows = _rows([(0.9, 1)] * 20 + [(0.1, 0)] * 20)
    th = fit_conformal_thresholds(
        rows, {"enabled": True, "mode": "conformal", "target_coverage": 0.9}
    )
    metrics = conformal_selective_metrics(rows, th)
    assert metrics["conformal_empirical_coverage_malware"] >= 0.9 - 1e-6
    assert metrics["conformal_empirical_coverage_benign"] >= 0.9 - 1e-6
    assert 0.0 <= metrics["conformal_acceptance_rate"] <= 1.0
    assert metrics["conformal_malware_acceptance_rate"] is not None
    assert metrics["conformal_num_empty_sets"] >= 0
    assert metrics["conformal_num_ambiguous_sets"] >= 0
    assert metrics["conformal_malware_fn_count"] >= 0
    assert metrics["conformal_accepted_malware_count"] >= 0


def test_conformal_all_dead_is_a_rejected_full_set_without_losing_coverage():
    rows = [
        {
            "prob_malware": 0.5,
            "label": 1,
            "pred": 0,
            "selective_eligible": 0,
        }
    ]
    thresholds = {
        "alpha": 0.1,
        "class_conditional": True,
        "use_raw_conflict": False,
        "q_benign": 0.2,
        "q_malware": 0.2,
    }

    metrics = conformal_selective_metrics(rows, thresholds)

    assert metrics["conformal_acceptance_rate"] == 0.0
    assert metrics["conformal_ambiguous_set_rate"] == 1.0
    assert metrics["conformal_empty_set_rate"] == 0.0
    assert metrics["conformal_empirical_coverage_malware"] == 1.0
    assert metrics["conformal_ineligible_set_policy"] == "full_ambiguous_set"
    assert rows[0]["conformal_set_size"] == 2
    assert rows[0]["accepted"] == 0
    assert rows[0]["rejected"] == 1
    assert rows[0]["selective_decision_mode"] == "conformal"


def test_conformal_metrics_empty_without_thresholds():
    assert conformal_selective_metrics(_rows([(0.9, 1)]), None) == {}


def test_conformal_nonconformity_uses_raw_conflict_when_present():
    rows = [
        {"prob_malware": 0.9, "label": 1, "pred": 1, "raw_conflict": 0.0, "selective_eligible": 1},
        {"prob_malware": 0.9, "label": 1, "pred": 1, "raw_conflict": 0.4, "selective_eligible": 1},
        {"prob_malware": 0.1, "label": 0, "pred": 0, "raw_conflict": 0.0, "selective_eligible": 1},
        {"prob_malware": 0.1, "label": 0, "pred": 0, "raw_conflict": 0.4, "selective_eligible": 1},
    ]
    with_conflict = fit_conformal_thresholds(
        rows,
        {
            "enabled": True,
            "mode": "conformal",
            "target_coverage": 0.5,
            "use_raw_conflict": True,
        },
    )
    without_conflict = fit_conformal_thresholds(
        rows,
        {
            "enabled": True,
            "mode": "conformal",
            "target_coverage": 0.5,
            "use_raw_conflict": False,
        },
    )

    assert with_conflict["use_raw_conflict"] is True
    assert with_conflict["q_malware"] > without_conflict["q_malware"]
    assert with_conflict["q_benign"] > without_conflict["q_benign"]


def test_conflict_aware_conformal_requires_current_raw_conflict_field():
    with pytest.raises(ValueError, match="missing mandatory raw_conflict"):
        fit_conformal_thresholds(
            _rows([(0.9, 1), (0.1, 0)]),
            {
                "enabled": True,
                "mode": "conformal",
                "target_coverage": 0.5,
                "use_raw_conflict": True,
            },
        )


def test_conformal_defaults_to_probability_only_nonconformity():
    rows = _rows([(0.9, 1), (0.8, 1), (0.1, 0), (0.2, 0)])
    for row in rows:
        row["raw_conflict"] = 0.8
    thresholds = fit_conformal_thresholds(
        rows, {"enabled": True, "mode": "conformal", "target_coverage": 0.5}
    )
    assert thresholds["use_raw_conflict"] is False


def test_threshold_rejection_is_disabled_in_conformal_mode():
    rows = [
        {"acceptance_score": 0.9, "label": 1, "pred": 1, "selective_eligible": 1},
        {"acceptance_score": 0.4, "label": 0, "pred": 1, "selective_eligible": 1},
    ]
    assert fit_rejection_threshold(rows, {"enabled": True, "mode": "conformal"}) is None
    assert fit_rejection_threshold(rows, {"enabled": True, "mode": "threshold", "target_coverage": 0.5}) == pytest.approx(0.9)
    assert fit_rejection_threshold(rows, {"enabled": True, "mode": "risk_control"}) is None


def test_risk_control_maximizes_acceptance_under_malware_fn_bound():
    rows = []
    rows.extend(
        {
            "acceptance_score": 0.1,
            "label": 1,
            "pred": 0,
            "prob_malware": 0.1,
            "selective_eligible": 1,
        }
        for _ in range(5)
    )
    rows.extend(
        {
            "acceptance_score": 0.9,
            "label": 1,
            "pred": 1,
            "prob_malware": 0.9,
            "selective_eligible": 1,
        }
        for _ in range(45)
    )
    rows.extend(
        {
            "acceptance_score": 0.8,
            "label": 0,
            "pred": 0,
            "prob_malware": 0.1,
            "selective_eligible": 1,
        }
        for _ in range(50)
    )
    thresholds = fit_risk_control_thresholds(
        rows,
        {
            "enabled": True,
            "mode": "risk_control",
            "risk_level": 0.05,
            "threshold_score": "model_acceptance",
        },
    )
    assert thresholds is not None
    assert thresholds["feasible"] is True
    assert thresholds["num_accepted"] == 95
    assert thresholds["corrected_risk"] <= 0.05
    assert thresholds["acceptance_comparison"] == (
        "selective_eligible and score > threshold"
    )

    metrics = risk_control_selective_metrics(rows, thresholds)
    assert metrics["risk_control_acceptance_rate"] == pytest.approx(0.95)
    assert metrics["risk_control_accepted_fn_risk_among_malware"] == 0.0
    assert metrics["risk_control_target_met_empirically"] is True


def test_risk_control_strict_boundary_keeps_repeated_scores_atomic():
    rows = [
        {"acceptance_score": 0.9, "label": 1, "pred": 1, "selective_eligible": 1}
        for _ in range(19)
    ]
    # This complete tie group must either be accepted or rejected.  Accepting
    # it would add a false negative and violate alpha=0.05.
    rows.extend(
        [
            {"acceptance_score": 0.5, "label": 1, "pred": 0, "selective_eligible": 1},
            {"acceptance_score": 0.5, "label": 0, "pred": 0, "selective_eligible": 1},
            {"acceptance_score": 0.5, "label": 0, "pred": 0, "selective_eligible": 1},
        ]
    )
    thresholds = fit_risk_control_thresholds(
        rows,
        {
            "enabled": True,
            "mode": "risk_control",
            "risk_level": 0.05,
            "require_feasible": True,
        },
    )

    assert thresholds is not None
    assert thresholds["num_accepted"] == 19
    assert thresholds["threshold"] == pytest.approx(0.5)
    metrics = risk_control_selective_metrics(rows, thresholds)
    assert metrics["risk_control_num_accepted"] == 19
    assert metrics["risk_control_num_rejected"] == 3
    assert (
        metrics["risk_control_acceptance_comparison"]
        == "selective_eligible and score > threshold"
    )


def test_risk_control_uses_the_minimum_feasible_boundary_not_next_accepted_score():
    rows = [
        {"acceptance_score": 0.9, "label": 1, "pred": 1, "selective_eligible": 1}
        for _ in range(19)
    ]
    rows.append(
        {"acceptance_score": 0.2, "label": 1, "pred": 0, "selective_eligible": 1}
    )

    thresholds = fit_risk_control_thresholds(
        rows,
        {
            "enabled": True,
            "mode": "risk_control",
            "risk_level": 0.05,
            "require_feasible": True,
        },
    )

    assert thresholds is not None
    assert thresholds["threshold"] == pytest.approx(0.2)
    # A future score between the rejected error and the smallest accepted
    # calibration score must remain accepted under the fitted strict rule.
    probe = [
        {"acceptance_score": 0.5, "label": 0, "pred": 0, "selective_eligible": 1}
    ]
    risk_control_selective_metrics(probe, thresholds)
    assert probe[0]["risk_control_accepted"] == 1


def test_risk_control_evaluation_rejects_score_equal_to_threshold():
    rows = [
        {"acceptance_score": 0.5, "label": 0, "pred": 0, "selective_eligible": 1},
        {"acceptance_score": 0.6, "label": 1, "pred": 1, "selective_eligible": 1},
    ]
    metrics = risk_control_selective_metrics(
        rows,
        {
            "threshold": 0.5,
            "risk_level": 0.1,
            "risk_target": "accepted_fn_risk_among_malware",
            "feasible": True,
            "corrected_risk": 0.0,
            "acceptance_comparison": "selective_eligible and score > threshold",
        },
    )
    assert metrics["risk_control_num_accepted"] == 1
    assert metrics["risk_control_num_rejected"] == 1
    assert rows[0]["risk_control_accepted"] == 0
    assert rows[0]["risk_control_rejected"] == 1
    assert rows[1]["risk_control_accepted"] == 1
    assert rows[1]["risk_control_rejected"] == 0
    assert rows[0]["risk_control_acceptance_comparison"] == (
        "selective_eligible and score > threshold"
    )
    assert rows[0]["accepted"] == 0
    assert rows[0]["rejected"] == 1
    assert rows[1]["accepted"] == 1
    assert rows[1]["rejected"] == 0
    assert rows[0]["selective_decision_mode"] == "risk_control"


def test_crc_risk_and_conditional_accepted_malware_fnr_are_not_conflated():
    rows = [
        {"acceptance_score": 0.9, "label": 1, "pred": 1, "selective_eligible": 1},
        {"acceptance_score": 0.8, "label": 1, "pred": 0, "selective_eligible": 1},
        {"acceptance_score": 0.4, "label": 1, "pred": 1, "selective_eligible": 1},
        {"acceptance_score": 0.3, "label": 1, "pred": 0, "selective_eligible": 1},
    ]
    metrics = risk_control_selective_metrics(
        rows,
        {
            "threshold": 0.5,
            "risk_level": 0.3,
            "risk_target": "accepted_fn_risk_among_malware",
            "feasible": True,
            "corrected_risk": 0.3,
            "acceptance_comparison": "selective_eligible and score > threshold",
        },
    )

    # CRC averages the accepted-FN indicator over every malware sample.
    assert metrics["risk_control_accepted_fn_risk_among_malware"] == pytest.approx(0.25)
    # This separate operational statistic conditions on accepted malware only.
    assert metrics["risk_control_fn_rate_given_accepted_malware"] == pytest.approx(0.5)


def test_risk_control_hard_rejects_all_dead_even_when_accept_all_is_feasible():
    rows = [
        {
            "acceptance_score": 0.6,
            "label": 1,
            "pred": 1,
            "selective_eligible": 1,
        }
        for _ in range(19)
    ]
    rows.append(
        {
            # A dense-gate all-dead fallback has uniform logits and strict
            # MSP=0.5. Availability, not an arbitrary score cutoff, must make
            # this a mandatory rejection.
            "acceptance_score": 0.5,
            "label": 1,
            "pred": 0,
            "selective_eligible": 0,
        }
    )
    thresholds = fit_risk_control_thresholds(
        rows,
        {
            "enabled": True,
            "mode": "risk_control",
            "risk_level": 0.10,
            "require_feasible": True,
        },
    )

    assert thresholds is not None
    assert thresholds["num_accepted"] == 19
    assert thresholds["num_ineligible_forced_reject"] == 1
    metrics = risk_control_selective_metrics(rows, thresholds)
    assert rows[-1]["risk_control_accepted"] == 0
    assert rows[-1]["risk_control_rejected"] == 1
    assert metrics["risk_control_num_ineligible_forced_reject"] == 1
    assert metrics["risk_control_accepted_fn_risk_among_malware"] == 0.0


def test_risk_control_rejects_malformed_rows_instead_of_shrinking_denominator():
    config = {
        "enabled": True,
        "mode": "risk_control",
        "risk_level": 0.10,
    }
    with pytest.raises(ValueError, match="no finite acceptance score"):
        fit_risk_control_thresholds(
            [{"acceptance_score": float("nan"), "label": 1, "pred": 0}],
            config,
        )
    with pytest.raises(ValueError, match="label/pred must be binary"):
        fit_risk_control_thresholds(
            [{"acceptance_score": 0.5, "label": 2, "pred": 0}],
            config,
        )
    with pytest.raises(ValueError, match="no finite acceptance score"):
        risk_control_selective_metrics(
            [{"acceptance_score": float("inf"), "label": 1, "pred": 0}],
            {
                "threshold": 0.5,
                "risk_level": 0.10,
                "risk_target": "accepted_fn_risk_among_malware",
            },
        )


def test_risk_control_rejects_removed_ambiguous_target_name():
    rows = [{"acceptance_score": 0.9, "label": 1, "pred": 1}]
    with pytest.raises(ValueError, match="currently supports only"):
        fit_risk_control_thresholds(
            rows,
            {
                "enabled": True,
                "mode": "risk_control",
                "risk_target": "malware_fn_rate_after_rejection",
            },
        )


def test_risk_control_reports_when_finite_sample_target_is_infeasible():
    rows = [
        {
            "acceptance_score": 0.9,
            "label": 1,
            "pred": 1,
            "prob_malware": 0.9,
            "selective_eligible": 1,
        }
        for _ in range(5)
    ]
    thresholds = fit_risk_control_thresholds(
        rows,
        {"enabled": True, "mode": "risk_control", "risk_level": 0.05},
    )
    assert thresholds is not None
    assert thresholds["feasible"] is False
    assert thresholds["num_accepted"] == 0


def test_infeasible_crc_fallback_rejects_unseen_higher_scores():
    calibration = [
        {"acceptance_score": 0.5, "label": 1, "pred": 1, "selective_eligible": 1}
        for _ in range(5)
    ]
    thresholds = fit_risk_control_thresholds(
        calibration,
        {"enabled": True, "mode": "risk_control", "risk_level": 0.01},
    )

    assert thresholds["feasible"] is False
    assert thresholds["threshold"] == 1.0
    probe = [
        {"acceptance_score": 0.9, "label": 1, "pred": 0, "selective_eligible": 1}
    ]
    metrics = risk_control_selective_metrics(probe, thresholds)
    assert metrics["risk_control_num_accepted"] == 0
    assert probe[0]["risk_control_rejected"] == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_conformal_rejects_invalid_probabilities(invalid):
    with pytest.raises(ValueError, match="prob_malware within"):
        fit_conformal_thresholds(
            [{"prob_malware": invalid, "label": 1}],
            {"enabled": True, "mode": "conformal", "target_coverage": 0.9},
        )


def test_every_selective_dispatch_rejects_an_unknown_mode():
    config = {"enabled": True, "mode": "typo"}
    rows = [{"acceptance_score": 0.9, "prob_malware": 0.9, "label": 1, "pred": 1}]
    for fitter in (
        fit_rejection_threshold,
        fit_conformal_thresholds,
        fit_risk_control_thresholds,
    ):
        with pytest.raises(ValueError, match="selective_prediction.mode"):
            fitter(rows, config)


def test_selective_aurc_uses_only_achievable_tie_atomic_prefixes():
    first = _selective_ranking_metrics(
        labels=[0, 1, 1],
        preds=[0, 0, 0],
        acceptance_scores=[0.5, 0.5, 0.9],
        selective_eligibility=[True, True, False],
    )
    swapped = _selective_ranking_metrics(
        labels=[1, 0, 1],
        preds=[0, 0, 0],
        acceptance_scores=[0.5, 0.5, 0.9],
        selective_eligibility=[True, True, False],
    )

    assert first["aurc"] == pytest.approx(0.5)
    assert swapped["aurc"] == pytest.approx(first["aurc"])
    assert first["selective_max_achievable_coverage"] == pytest.approx(2.0 / 3.0)
    assert first["aurc_tie_policy"] == "atomic_score_groups"


def test_target_aligned_aurc_does_not_charge_false_positives_to_fn_risk():
    metrics = _selective_ranking_metrics(
        labels=[0, 1],
        preds=[1, 0],
        acceptance_scores=[0.9, 0.1],
        selective_eligibility=[True, True],
    )

    # Generic selective classification AURC counts both errors. The primary
    # I2/I3 ranking metric counts only the accepted malware-FN event and keeps
    # the CRC denominator fixed to all malware.
    assert metrics["aurc"] == pytest.approx(1.0)
    assert metrics["malware_fn_risk_aurc"] == pytest.approx(0.5)
    assert metrics["malware_fn_risk_aurc_target"] == (
        "accepted_fn_risk_among_malware"
    )
    assert metrics["malware_fn_risk_aurc_denominator"] == "all_malware"


def test_selective_eligible_is_mandatory_and_binary():
    with pytest.raises(ValueError, match="missing mandatory"):
        _row_selective_eligible(
            {"api_alive": 1, "graph_alive": 1, "manifest_alive": 1}
        )
    with pytest.raises(ValueError, match="boolean or binary"):
        _row_selective_eligible({"selective_eligible": float("nan")})
    with pytest.raises(ValueError, match="missing mandatory tensor"):
        _batch_selective_eligibility(
            {},
            batch_size=1,
            device=torch.device("cpu"),
        )


def test_selective_helpers_do_not_truncate_fractional_binary_fields():
    with pytest.raises(ValueError, match="must be binary"):
        _selective_ranking_metrics(
            labels=[0.5],
            preds=[0.5],
            acceptance_scores=[0.5],
        )
    with pytest.raises(ValueError, match="label/pred must be binary"):
        risk_control_selective_metrics(
            [{"acceptance_score": 0.5, "label": 0.5, "pred": 0.5}],
            {
                "threshold": 0.5,
                "risk_level": 0.1,
                "risk_target": "accepted_fn_risk_among_malware",
            },
        )
    with pytest.raises(ValueError, match="binary label/pred"):
        build_risk_coverage_curve(
            [{"acceptance_score": 0.5, "label": 0.5, "pred": 0.5}]
        )


def test_selective_threshold_scores_have_explicit_semantics():
    probability = torch.tensor([0.2, 0.7])
    extra = {
        "fused_uncertainty": torch.tensor([0.4, 0.1]),
        "acceptance_score": torch.tensor([0.3, 0.8]),
    }

    assert torch.allclose(
        _batch_selective_score(
            probability, extra, "deployed_class_probability"
        ),
        torch.tensor([0.8, 0.7]),
    )
    assert torch.allclose(
        _batch_selective_score(
            probability,
            extra,
            "deployed_class_probability",
            classification_threshold=0.75,
        ),
        torch.tensor([0.8, 0.3]),
    )
    assert torch.allclose(
        _batch_selective_score(
            probability,
            extra,
            "msp",
            classification_threshold=0.75,
        ),
        torch.tensor([0.8, 0.7]),
    )
    expected_entropy_certainty = torch.tensor(
        [0.2780719, 0.1187091], dtype=torch.float32
    )
    assert torch.allclose(
        _batch_selective_score(
            probability, extra, "predictive_entropy_certainty"
        ),
        expected_entropy_certainty,
        atol=1e-6,
    )
    assert torch.allclose(
        _batch_selective_score(
            torch.tensor([0.0, 0.5, 1.0]),
            extra,
            "predictive_entropy_certainty",
        ),
        torch.tensor([1.0, 0.0, 1.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        _batch_selective_score(probability, extra, "evidential_certainty"),
        torch.tensor([0.6, 0.9]),
    )
    assert _selective_score_type({"mode": "conformal"}) == "msp"
    assert _selective_score_type(
        {"mode": "threshold", "threshold_score": "evidential_certainty"}
    ) == "evidential_certainty"
    with pytest.raises(ValueError, match="max_probability.*removed"):
        _selective_score_type(
            {"mode": "threshold", "threshold_score": "max_probability"}
        )
    with pytest.raises(ValueError, match="max_probability.*removed"):
        _batch_selective_score(probability, extra, "max_probability")


def test_binary_entropy_certainty_is_rank_and_acceptance_equivalent_to_msp():
    # Use unique distances from 0.5 so the expected order has no tie ambiguity.
    probability = torch.tensor([0.02, 0.11, 0.27, 0.49, 0.58, 0.74, 0.93])
    msp = _batch_selective_score(probability, {}, "msp")
    entropy = _batch_selective_score(
        probability, {}, "predictive_entropy_certainty"
    )
    msp_order = torch.argsort(msp, descending=True)
    entropy_order = torch.argsort(entropy, descending=True)
    assert torch.equal(msp_order, entropy_order)

    for accepted_count in range(1, probability.numel() + 1):
        msp_cutoff = msp[msp_order[accepted_count - 1]]
        entropy_cutoff = entropy[entropy_order[accepted_count - 1]]
        assert torch.equal(msp >= msp_cutoff, entropy >= entropy_cutoff)
