import pytest
import torch

from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import compute_posthoc_calibration_loss
from fusion.reliability_calibration import MonotonicReliabilityCalibrator
from fusion.train import (
    _selective_metrics,
    _selective_ranking_metrics,
    fit_rejection_threshold,
    split_validation_dataset,
)


def _evidence(batch_size: int = 1) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def _logits(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor([[2.0, -2.0]] * batch_size) for _ in range(4))


def test_monotonic_calibrator_respects_integrity_and_conflict_directions():
    calibrator = MonotonicReliabilityCalibrator(hidden_dim=8)
    low = _evidence()
    high = low.clone()
    low[:, EvidenceIndex.API_INTEGRITY] = 0.2
    high[:, EvidenceIndex.API_INTEGRITY] = 0.8
    assert (
        calibrator(high)["predicted_reliability_api"]
        >= calibrator(low)["predicted_reliability_api"]
    ).all()

    conflict = high.clone()
    conflict[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.9
    assert (
        calibrator(conflict)["predicted_reliability_api"]
        <= calibrator(high)["predicted_reliability_api"]
    ).all()


def test_missing_counterpart_does_not_increase_api_reliability():
    calibrator = MonotonicReliabilityCalibrator(hidden_dim=8)
    present = _evidence()
    present[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.5
    missing = present.clone()
    missing[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    assert (
        calibrator(missing)["predicted_reliability_api"]
        <= calibrator(present)["predicted_reliability_api"]
    ).all()


def test_missing_manifest_disables_code_manifest_conflict_penalty():
    base = _evidence()
    base[:, EvidenceIndex.MANIFEST_ALIVE] = 0.0
    high_conflict = base.clone()
    high_conflict[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 1.0
    fusion = DiscountProbabilityFusion()
    base_out = fusion(*_logits(), base)
    conflict_out = fusion(*_logits(), high_conflict)
    assert conflict_out["manifest_code_conflict_applicable"].item() == 0.0
    assert conflict_out["effective_conflict"].item() == 0.0
    assert conflict_out["discount_api"].item() == pytest.approx(base_out["discount_api"].item())
    assert conflict_out["discount_graph"].item() == pytest.approx(base_out["discount_graph"].item())


def test_posthoc_calibration_loss_updates_calibration_parameters():
    fusion = DiscountProbabilityFusion(
        {
            "reliability_calibration": {"enabled": True, "hidden_dim": 8},
            "probability_calibration": {"enabled": True},
        }
    )
    fusion.set_calibration_active(True)
    evidence = _evidence(2)
    logits = tuple(
        torch.tensor([[3.0, -3.0], [-3.0, 3.0]], dtype=torch.float32)
        for _ in range(4)
    )
    outputs = fusion(*logits, evidence)
    outputs.update(
        {
            f"{name}_logits_aux": branch_logits
            for name, branch_logits in zip(
                ("api", "graph", "manifest", "joint"), logits
            )
        }
    )
    loss, parts = compute_posthoc_calibration_loss(
        outputs,
        torch.tensor([0, 0]),
        evidence,
        {
            "reliability_calibration": {"weight": 1.0},
            "probability_calibration": {"weight": 1.0},
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["reliability_calibration_loss"] > 0.0
    assert parts["probability_calibration_loss"] > 0.0
    assert all(parameter.grad is not None for parameter in fusion.calibration_parameters())


def test_calibration_parameters_are_inactive_during_main_training():
    fusion = DiscountProbabilityFusion(
        {
            "detach_discount": True,
            "detach_confidence_proxy": True,
            "reliability_calibration": {"enabled": True, "hidden_dim": 8},
            "probability_calibration": {"enabled": True},
        }
    )
    evidence = _evidence(2)
    logits = tuple(torch.randn(2, 2, requires_grad=True) for _ in range(4))
    outputs = fusion(*logits, evidence)
    torch.nn.functional.nll_loss(outputs["final_logits"], torch.tensor([0, 1])).backward()
    assert outputs["calibration_active"].sum().item() == 0.0
    assert all(parameter.grad is None for parameter in fusion.calibration_parameters())


def test_validation_split_is_deterministic_and_group_isolated():
    class Dataset:
        sample_sids = ["a", "b", "c", "d", "e"]
        sample_groups = ["same", "same", "g2", "g3", "g4"]

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    cfg = {"train": {"seed": 42}, "calibration": {"validation_fraction": 0.4}}
    selection_a, calibration_a, meta_a = split_validation_dataset(cfg, Dataset())
    selection_b, calibration_b, meta_b = split_validation_dataset(cfg, Dataset())
    assert selection_a.indices == selection_b.indices
    assert calibration_a.indices == calibration_b.indices
    assert meta_a == meta_b
    selection_groups = {Dataset.sample_groups[index] for index in selection_a.indices}
    calibration_groups = {Dataset.sample_groups[index] for index in calibration_a.indices}
    assert selection_groups.isdisjoint(calibration_groups)


def test_selective_metrics_and_validation_threshold():
    rows = [
        {"acceptance_score": 0.9},
        {"acceptance_score": 0.8},
        {"acceptance_score": 0.2},
        {"acceptance_score": 0.1},
    ]
    threshold = fit_rejection_threshold(
        rows, {"enabled": True, "target_coverage": 0.5}
    )
    assert threshold == pytest.approx(0.8)
    metrics = _selective_metrics(
        labels=[0, 1, 0, 1],
        preds=[0, 1, 1, 0],
        acceptance_scores=[0.9, 0.8, 0.2, 0.1],
        threshold=threshold,
    )
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["selective_risk"] == pytest.approx(0.0)
    assert metrics["selective_metrics_defined"] is True


def test_selective_metrics_are_undefined_when_every_sample_is_rejected():
    metrics = _selective_metrics(
        labels=[0, 1],
        preds=[0, 0],
        acceptance_scores=[0.1, 0.2],
        threshold=0.9,
    )

    assert metrics["coverage"] == pytest.approx(0.0)
    assert metrics["selective_metrics_defined"] is False
    assert metrics["selective_risk"] is None
    assert metrics["selective_acc"] is None
    assert metrics["selective_macro_f1"] is None
    assert metrics["aurc"] >= 0.0


def test_threshold_free_selective_metrics_report_only_ranking_quality():
    metrics = _selective_ranking_metrics(
        labels=[0, 1],
        preds=[0, 0],
        acceptance_scores=[0.9, 0.1],
    )

    assert metrics["aurc"] >= 0.0
    assert "coverage" not in metrics
    assert "selective_risk" not in metrics
