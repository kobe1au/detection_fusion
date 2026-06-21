import pytest
import torch
import torch.nn as nn

from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import compute_posthoc_calibration_loss
from fusion.reliability_calibration import MonotonicReliabilityCalibrator
from fusion.train import (
    _branch_prediction_row,
    _selective_metrics,
    _selective_ranking_metrics,
    compute_branch_reliability_metrics,
    fit_posthoc_calibration,
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


def test_intrinsic_calibrator_ignores_pairwise_relation_values():
    calibrator = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=False,
    )
    favorable = _evidence()
    unfavorable = favorable.clone()
    unfavorable[:, EvidenceIndex.API_GRAPH_ANCHOR_SUPPORT] = 0.1
    unfavorable[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.1
    unfavorable[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.9
    unfavorable[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.9

    favorable_out = calibrator(favorable)
    unfavorable_out = calibrator(unfavorable)
    for name in ("api", "graph", "manifest", "joint"):
        assert torch.allclose(
            favorable_out[f"predicted_reliability_{name}"],
            unfavorable_out[f"predicted_reliability_{name}"],
        )


def test_intrinsic_api_reliability_is_neutral_to_missing_graph():
    calibrator = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=False,
    )
    complete = _evidence()
    missing_graph = complete.clone()
    missing_graph[:, EvidenceIndex.GRAPH_ALIVE] = 0.0
    missing_graph[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.0
    missing_graph[:, EvidenceIndex.CODE_INTEGRITY] = 0.0

    assert torch.allclose(
        calibrator(complete)["predicted_reliability_api"],
        calibrator(missing_graph)["predicted_reliability_api"],
    )


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


def test_calibrator_alive_mask_can_be_disabled_for_ablation():
    evidence = _evidence()
    evidence[:, EvidenceIndex.API_ALIVE] = 0.0
    masked = MonotonicReliabilityCalibrator(hidden_dim=8, apply_alive_mask=True)
    unmasked = MonotonicReliabilityCalibrator(hidden_dim=8, apply_alive_mask=False)

    assert masked(evidence)["predicted_reliability_api"].item() == 0.0
    assert unmasked(evidence)["predicted_reliability_api"].item() > 0.0


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


def test_empty_manifest_code_semantic_relation_does_not_apply_explicit_discount():
    evidence = _evidence()
    evidence[:, EvidenceIndex.MANIFEST_CODE_SUPPORT] = 0.0
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0

    output = DiscountProbabilityFusion()(*_logits(), evidence)

    assert output["manifest_code_relation_applicable"].item() == 0.0
    assert output["effective_conflict"].item() == 0.0
    assert output["discount_manifest"].item() == pytest.approx(
        output["confidence_factor_manifest"].item()
    )


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


def test_posthoc_reliability_loss_is_safe_under_autocast():
    reliability = torch.tensor([0.8, 0.2], dtype=torch.float32, requires_grad=True)
    outputs = {
        "predicted_reliability_api": reliability,
        "api_logits_aux": torch.tensor([[3.0, -3.0], [3.0, -3.0]], dtype=torch.float32),
    }
    evidence = _evidence(2)

    with torch.amp.autocast(device_type="cpu", enabled=True):
        loss, parts = compute_posthoc_calibration_loss(
            outputs,
            torch.tensor([0, 1]),
            evidence,
            {
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 0.0},
            },
        )

    loss.backward()
    assert torch.isfinite(loss)
    assert parts["reliability_calibration_loss"] > 0.0
    assert reliability.grad is not None


class _CalibrationGraph:
    def __init__(self, evidence: torch.Tensor):
        self.evidence = evidence

    def to(self, device, non_blocking=True):
        self.evidence = self.evidence.to(device)
        return self


class _PosthocCalibrationModel(nn.Module):
    fusion_mode = "discount_probability"

    def __init__(self):
        super().__init__()
        self.discount_fusion = DiscountProbabilityFusion(
            {
                "use_confidence_proxy": True,
                "reliability_calibration": {
                    "enabled": True,
                    "hidden_dim": 8,
                    "use_relation_evidence": False,
                },
                "probability_calibration": {"enabled": True},
            }
        )

    def calibration_parameters(self):
        return self.discount_fusion.calibration_parameters()

    def set_calibration_active(self, enabled: bool):
        self.discount_fusion.set_calibration_active(enabled)

    def forward(self, graph, return_features=False):
        batch_size = graph.evidence.size(0)
        branch_logits = tuple(
            graph.evidence.new_tensor([[2.0, -2.0], [-2.0, 2.0]])[:batch_size]
            for _ in range(4)
        )
        outputs = self.discount_fusion(*branch_logits, graph.evidence)
        for name, logits in zip(
            ("api", "graph", "manifest", "joint"),
            branch_logits,
        ):
            outputs[f"{name}_logits_aux"] = logits
        outputs["gate_evidence"] = graph.evidence
        return outputs["final_logits"], outputs


def test_posthoc_calibration_restores_best_epoch_and_stops_early():
    evidence = _evidence(2)
    loader = [
        {
            "graph_batch": _CalibrationGraph(evidence),
            "labels": torch.tensor([0, 1]),
            "sids": ["a", "b"],
            "quality": {},
            "num_failed": 0,
        }
    ]
    model = _PosthocCalibrationModel()
    summary = fit_posthoc_calibration(
        model,
        [loader],
        torch.device("cpu"),
        False,
        {
            "calibration": {
                "enabled": True,
                "epochs": 5,
                "patience": 1,
                "min_delta": 1.0e6,
                "lr": 1.0e-3,
            },
            "fusion": {
                "reliability_calibration": {"weight": 1.0},
                "probability_calibration": {"weight": 1.0},
            },
        },
    )

    assert summary["best_epoch"] == 1
    assert summary["epochs_ran"] == 2
    assert summary["stopped_early"] is True
    assert summary["final_loss"] == summary["losses"][0]


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


def test_branch_prediction_row_records_per_branch_correctness():
    extra = {
        "api_logits_aux": torch.tensor([[-2.0, 2.0], [3.0, -3.0]]),
        "graph_logits_aux": torch.tensor([[3.0, -3.0], [-2.0, 2.0]]),
        "manifest_logits_aux": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "joint_logits_aux": torch.tensor([[-1.0, 1.0], [1.0, -1.0]]),
    }
    labels = torch.tensor([1, 0])

    first = _branch_prediction_row(extra, labels, 0)
    second = _branch_prediction_row(extra, labels, 1)

    assert first["api_pred"] == 1
    assert first["api_correct"] == 1
    assert first["graph_pred"] == 0
    assert first["graph_correct"] == 0
    assert first["api_prob"] > 0.9
    assert first["api_confidence"] > 0.9
    assert second["joint_pred"] == 0
    assert second["joint_correct"] == 1


def test_branch_reliability_metrics_compare_reliability_to_branch_correctness():
    rows = [
        {"api_correct": 1, "predicted_reliability_api": 0.9},
        {"api_correct": 1, "predicted_reliability_api": 0.8},
        {"api_correct": 0, "predicted_reliability_api": 0.2},
        {"api_correct": 0, "predicted_reliability_api": 0.1},
        {"graph_correct": 1, "predicted_reliability_graph": 0.4},
    ]

    metrics = compute_branch_reliability_metrics(rows)

    assert metrics["api_reliability_count"] == 4
    assert metrics["api_reliability_auc"] == pytest.approx(1.0)
    assert metrics["api_reliability_brier"] == pytest.approx(0.025)
    assert metrics["api_branch_accuracy"] == pytest.approx(0.5)
    assert metrics["api_reliability_mean"] == pytest.approx(0.5)
    assert metrics["graph_reliability_count"] == 1
    assert metrics["graph_reliability_auc"] == 0.0
