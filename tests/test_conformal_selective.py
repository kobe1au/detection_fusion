import pytest

from fusion.train import (
    conformal_selective_metrics,
    fit_conformal_thresholds,
)


def _rows(probs_labels):
    rows = []
    for p, y in probs_labels:
        rows.append({"prob_malware": p, "label": y, "pred": int(p >= 0.5)})
    return rows


def test_fit_conformal_thresholds_disabled_returns_none():
    assert fit_conformal_thresholds([], {"enabled": False}) is None
    assert fit_conformal_thresholds([], {"enabled": True, "mode": "threshold"}) is None


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


def test_conformal_metrics_empty_without_thresholds():
    assert conformal_selective_metrics(_rows([(0.9, 1)]), None) == {}
