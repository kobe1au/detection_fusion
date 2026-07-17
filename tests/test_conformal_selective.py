import pytest
import torch

from fusion.train import (
    _batch_selective_score,
    _selective_score_type,
    build_risk_coverage_curve,
    conformal_selective_metrics,
    fit_conformal_thresholds,
    fit_malware_classification_threshold,
    fit_rejection_threshold,
    fit_risk_control_thresholds,
    risk_control_selective_metrics,
)


def _rows(probs_labels):
    rows = []
    for p, y in probs_labels:
        rows.append({"prob_malware": p, "label": y, "pred": int(p >= 0.5)})
    return rows


def test_fit_conformal_thresholds_disabled_returns_none():
    assert fit_conformal_thresholds([], {"enabled": False}) is None
    assert fit_conformal_thresholds([], {"enabled": True, "mode": "threshold"}) is None


def test_classification_threshold_maximizes_macro_f1_under_recall_constraint():
    rows = _rows(
        [(0.10, 0), (0.20, 0), (0.40, 0), (0.60, 0)]
        + [(0.35, 1), (0.45, 1), (0.80, 1), (0.90, 1)]
    )
    fitted = fit_malware_classification_threshold(
        rows,
        {
            "enabled": True,
            "objective": "macro_f1",
            "min_malware_recall": 0.75,
        },
    )

    assert fitted is not None
    assert fitted["threshold"] < 0.5
    assert fitted["malware_recall"] >= 0.75
    assert fitted["macro_f1"] >= fitted["fixed_0_5_macro_f1"]
    assert fitted["calibration_split"] == "val_posthoc_calibration"


def test_risk_coverage_curve_reports_exact_threshold_operating_points():
    rows = [
        {"split": "test_clean", "acceptance_score": 0.9, "label": 1, "pred": 1},
        {"split": "test_clean", "acceptance_score": 0.8, "label": 1, "pred": 0},
        {"split": "test_clean", "acceptance_score": 0.7, "label": 0, "pred": 0},
        {"split": "other", "acceptance_score": 0.6, "label": 0, "pred": 1},
    ]
    curve = build_risk_coverage_curve(rows)
    clean = [point for point in curve if point["split"] == "test_clean"]

    assert [point["coverage"] for point in clean] == pytest.approx(
        [1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    assert clean[-1]["selective_risk"] == pytest.approx(1.0 / 3.0)
    assert clean[-1]["malware_fn_rate_after_rejection"] == pytest.approx(0.5)


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


def test_conformal_metrics_empty_without_thresholds():
    assert conformal_selective_metrics(_rows([(0.9, 1)]), None) == {}


def test_conformal_nonconformity_uses_raw_conflict_when_present():
    rows = [
        {"prob_malware": 0.9, "label": 1, "pred": 1, "raw_conflict": 0.0},
        {"prob_malware": 0.9, "label": 1, "pred": 1, "raw_conflict": 0.4},
        {"prob_malware": 0.1, "label": 0, "pred": 0, "raw_conflict": 0.0},
        {"prob_malware": 0.1, "label": 0, "pred": 0, "raw_conflict": 0.4},
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
        {"acceptance_score": 0.9, "label": 1, "pred": 1},
        {"acceptance_score": 0.4, "label": 0, "pred": 1},
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
        }
        for _ in range(5)
    )
    rows.extend(
        {
            "acceptance_score": 0.9,
            "label": 1,
            "pred": 1,
            "prob_malware": 0.9,
        }
        for _ in range(45)
    )
    rows.extend(
        {
            "acceptance_score": 0.8,
            "label": 0,
            "pred": 0,
            "prob_malware": 0.1,
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

    metrics = risk_control_selective_metrics(rows, thresholds)
    assert metrics["risk_control_acceptance_rate"] == pytest.approx(0.95)
    assert metrics["risk_control_malware_fn_rate_after_rejection"] == 0.0
    assert metrics["risk_control_target_met_empirically"] is True


def test_risk_control_reports_when_finite_sample_target_is_infeasible():
    rows = [
        {"acceptance_score": 0.9, "label": 1, "pred": 1, "prob_malware": 0.9}
        for _ in range(5)
    ]
    thresholds = fit_risk_control_thresholds(
        rows,
        {"enabled": True, "mode": "risk_control", "risk_level": 0.05},
    )
    assert thresholds is not None
    assert thresholds["feasible"] is False
    assert thresholds["num_accepted"] == 0


def test_selective_threshold_scores_have_explicit_semantics():
    probability = torch.tensor([0.2, 0.7])
    extra = {
        "fused_uncertainty": torch.tensor([0.4, 0.1]),
        "acceptance_score": torch.tensor([0.3, 0.8]),
    }

    assert torch.allclose(
        _batch_selective_score(probability, extra, "max_probability"),
        torch.tensor([0.8, 0.7]),
    )
    assert torch.allclose(
        _batch_selective_score(
            probability,
            extra,
            "max_probability",
            classification_threshold=0.75,
        ),
        torch.tensor([0.8, 0.3]),
    )
    assert torch.allclose(
        _batch_selective_score(probability, extra, "evidential_certainty"),
        torch.tensor([0.6, 0.9]),
    )
    assert _selective_score_type({"mode": "conformal"}) == "max_probability"
    assert _selective_score_type(
        {"mode": "threshold", "threshold_score": "evidential_certainty"}
    ) == "evidential_certainty"
