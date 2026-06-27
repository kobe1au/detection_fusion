import json
import math

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from fusion.constants import EvidenceIndex
from fusion.dataset import apply_graph_encoder_budget, robust_collate_fn
from fusion.evidence import build_evidence
from fusion.quality import OBSERVABLE_NUMERIC_FIELDS, OBSERVABLE_SIGNAL_FIELDS
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.losses import compute_posthoc_calibration_loss
from fusion.reliability_calibration import MonotonicReliabilityCalibrator
from fusion.train import (
    _branch_prediction_row,
    _selective_metrics,
    _selective_ranking_metrics,
    _write_metrics_json,
    compute_branch_reliability_metrics,
    estimate_branch_competence_prior,
    estimate_model_visible_integrity_reference,
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


def test_intrinsic_single_modality_features_use_self_integrity_once():
    calibrator = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=False,
    )
    evidence = _evidence()
    integrity = {"api": 0.2, "graph": 0.4, "manifest": 0.6}
    evidence[:, EvidenceIndex.API_INTEGRITY] = integrity["api"]
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = integrity["graph"]
    evidence[:, EvidenceIndex.MANIFEST_INTEGRITY] = integrity["manifest"]
    evidence[:, EvidenceIndex.CODE_INTEGRITY] = 0.9

    outputs = calibrator(evidence)
    for name, value in integrity.items():
        expected = torch.tensor([[value, 0.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(outputs[f"reliability_features_{name}"], expected)


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


def test_intrinsic_api_reliability_is_neutral_to_degraded_alive_graph():
    calibrator = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=False,
    )
    complete = _evidence()
    degraded_graph = complete.clone()
    degraded_graph[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.01
    degraded_graph[:, EvidenceIndex.CODE_INTEGRITY] = 0.1

    complete_out = calibrator(complete)
    degraded_out = calibrator(degraded_graph)
    assert torch.allclose(
        complete_out["predicted_reliability_api"],
        degraded_out["predicted_reliability_api"],
    )
    assert degraded_out["predicted_reliability_graph"].item() < complete_out[
        "predicted_reliability_graph"
    ].item()


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
                    "use_relation_evidence": True,
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



def test_reliability_and_conflict_switches_remove_their_decision_signals():
    fusion = DiscountProbabilityFusion(
        {
            "use_reliability_discount": False,
            "use_support_discount": False,
            "use_conflict_discount": False,
            "use_confidence_proxy": False,
            "use_hard_alive_mask": False,
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )
    evidence = _evidence(2)
    evidence[0, EvidenceIndex.API_INTEGRITY] = 0.0
    evidence[0, EvidenceIndex.GRAPH_INTEGRITY] = 0.1
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 1.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 1.0

    outputs = fusion(*_logits(batch_size=2), evidence)

    assert torch.allclose(
        outputs["fusion_weights"], torch.full((2, 4), 0.25)
    )
    assert torch.equal(outputs["effective_conflict"], torch.zeros(2))

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
        sample_labels = [0, 0, 1, 0, 1]

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



def test_validation_split_is_label_stratified_for_singleton_groups():
    class Dataset:
        sample_sids = [f"s{index}" for index in range(20)]
        sample_groups = [f"g{index}" for index in range(20)]
        sample_labels = [0] * 10 + [1] * 10

        def __len__(self):
            return len(self.sample_sids)

        def __getitem__(self, index):
            return index

    cfg = {"train": {"seed": 42}, "calibration": {"validation_fraction": 0.5}}
    selection, calibration, meta = split_validation_dataset(cfg, Dataset())

    assert len(selection) == 10
    assert len(calibration) == 10
    assert meta["selection_label_counts"] == {0: 5, 1: 5}
    assert meta["calibration_label_counts"] == {0: 5, 1: 5}


def test_discount_probability_fusion_uses_fp32_for_half_inputs():
    fusion = DiscountProbabilityFusion(
        {
            "reliability_calibration": {"enabled": False},
            "probability_calibration": {"enabled": False},
        }
    )
    logits = tuple(value.half() for value in _logits(batch_size=2))
    outputs = fusion(*logits, _evidence(batch_size=2).half())

    assert outputs["final_logits"].dtype == torch.float32
    assert outputs["acceptance_score"].dtype == torch.float32
    assert outputs["fusion_weights"].dtype == torch.float32


def test_graph_encoder_budget_refreshes_alignment_and_effective_integrity():
    api_types = torch.tensor([1, 2], dtype=torch.long)
    data = {
        "x": torch.ones((5, 3)),
        "edge_index": torch.tensor([[0, 1, 3, 4], [1, 2, 4, 3]], dtype=torch.long),
        "sensitive_mask": torch.zeros(5, dtype=torch.uint8),
        "real_node_mask": torch.ones(5, dtype=torch.bool),
        "mask": torch.empty((5, 0)),
        "method_api_edge_index": torch.tensor([[0, 4], [0, 1]], dtype=torch.long),
        "api_ids": torch.tensor([10, 11], dtype=torch.long),
        "api_type_ids": api_types,
        "api_method_index": torch.tensor([0, 4], dtype=torch.long),
        "api_in_graph_mask": torch.ones(2),
        "api_semantic_category_counts": torch.ones(12),
        "graph_semantic_category_counts": torch.ones(12),
        "manifest_category_counts": torch.ones(12),
        "api_integrity": 1.0,
        "graph_integrity": 1.0,
        "q_graph": 1.0,
        "api_parse_ok": True,
        "dex_parse_ok": True,
        "graph_parse_ok": True,
        "manifest_parse_ok": True,
        "api_event_count_raw": 2,
        "api_event_count_kept": 2,
        "api_known_type_count": 2,
        "api_unknown_type_count": 0,
        "graph_edge_count_reference": 4,
        "manifest_permission_ids": torch.tensor([1]),
        "manifest_intent_ids": torch.tensor([1]),
        "manifest_stats": torch.ones(11),
    }

    out = apply_graph_encoder_budget(data, 3, "alignment")

    assert out["x"].size(0) == 3
    assert out["graph_encoder_coverage"] == pytest.approx(0.6)
    assert out["graph_truncated_by_encoder_budget"] == 1.0
    assert out["method_api_edge_index"].tolist() == [[0], [0]]
    assert out["api_method_index"].tolist() == [0, -1]
    assert out["api_in_graph_mask"].tolist() == [1.0, 0.0]
    # assert out["graph_integrity"] < out["graph_integrity_before_encoder_budget"]
    # assert out["code_integrity"] == pytest.approx(
    #     math.sqrt(out["api_integrity"] * out["graph_integrity"])
    # )
    assert out["graph_encoder_coverage"] == pytest.approx(0.6)
    assert out["graph_truncated_by_encoder_budget"] == 1.0

    # graph_integrity remains the post-budget structural integrity,
    # coverage is not multiplied into graph_integrity itself.
    assert out["q_graph"] == pytest.approx(out["graph_integrity"])
    assert out["r_graph"] == pytest.approx(out["graph_integrity"])

    # effective_graph_integrity is where coverage correction happens.
    assert out["effective_graph_integrity"] == pytest.approx(
        out["graph_integrity"] * out["graph_encoder_coverage"]
    )
    assert out["effective_graph_integrity"] <= out["graph_integrity"]

def test_graph_encoder_budget_diagnostics_survive_collate_and_evidence_building():
    data = Data(
        x=torch.ones((3, 3), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor(1, dtype=torch.long),
    )

    data.sid = "sample-graph-budget"
    data.year = torch.tensor(2024, dtype=torch.long)
    data.sensitive_mask = torch.zeros((3,), dtype=torch.uint8)

    data.api_ids = torch.tensor([10, 11], dtype=torch.long)
    data.api_type_ids = torch.tensor([1, 2], dtype=torch.long)
    data.api_sensitive_mask = torch.zeros((2,), dtype=torch.float32)
    data.api_method_index = torch.tensor([0, 1], dtype=torch.long)
    data.api_in_graph_mask = torch.ones((2,), dtype=torch.float32)
    data.method_api_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    data.api_semantic_category_counts = torch.ones((12,), dtype=torch.float32)
    data.graph_semantic_category_counts = torch.ones((12,), dtype=torch.float32)
    data.api_category_counts = data.api_semantic_category_counts
    data.graph_category_counts = data.graph_semantic_category_counts

    data.manifest_x = torch.zeros((1, 256), dtype=torch.float32)
    data.manifest_permission_ids = torch.tensor([1], dtype=torch.long)
    data.manifest_intent_ids = torch.tensor([1], dtype=torch.long)
    data.manifest_category_counts = torch.ones((12,), dtype=torch.float32)
    data.manifest_stats = torch.ones((11,), dtype=torch.float32)

    data.q_api = torch.tensor([0.9], dtype=torch.float32)
    data.q_graph = torch.tensor([0.8], dtype=torch.float32)
    data.q_manifest = torch.tensor([0.95], dtype=torch.float32)
    data.q_align = torch.tensor([0.7], dtype=torch.float32)
    data.pert_api = torch.tensor([0.0], dtype=torch.float32)
    data.pert_graph = torch.tensor([0.0], dtype=torch.float32)
    data.pert_manifest = torch.tensor([0.0], dtype=torch.float32)

    # Populate all observable fields required by robust_collate_fn().
    for key in OBSERVABLE_NUMERIC_FIELDS:
        setattr(data, key, torch.tensor([0.0], dtype=torch.float32))
    for key in OBSERVABLE_SIGNAL_FIELDS:
        setattr(data, key, torch.tensor([1.0], dtype=torch.float32))

    data.api_integrity = torch.tensor([0.9], dtype=torch.float32)
    data.graph_integrity = torch.tensor([0.8], dtype=torch.float32)
    data.manifest_integrity = torch.tensor([0.95], dtype=torch.float32)
    data.code_integrity = torch.tensor([0.85], dtype=torch.float32)
    data.api_alive = torch.tensor([1.0], dtype=torch.float32)
    data.graph_alive = torch.tensor([1.0], dtype=torch.float32)
    data.manifest_alive = torch.tensor([1.0], dtype=torch.float32)

    # The three fields that were previously dropped during collate.
    data.graph_encoder_coverage = torch.tensor([0.5], dtype=torch.float32)
    data.graph_truncated_by_encoder_budget = torch.tensor([1.0], dtype=torch.float32)
    data.graph_integrity_before_encoder_budget = torch.tensor([0.8], dtype=torch.float32)

    batch = robust_collate_fn([data])
    graph_batch = batch["graph_batch"]

    assert hasattr(graph_batch, "graph_encoder_coverage")
    assert hasattr(graph_batch, "graph_truncated_by_encoder_budget")
    assert hasattr(graph_batch, "graph_integrity_before_encoder_budget")

    assert graph_batch.graph_encoder_coverage.view(-1).item() == pytest.approx(0.5)
    assert graph_batch.graph_truncated_by_encoder_budget.view(-1).item() == pytest.approx(1.0)
    assert graph_batch.graph_integrity_before_encoder_budget.view(-1).item() == pytest.approx(0.8)

    api_logits, graph_logits, manifest_logits, joint_logits = _logits(batch_size=1)
    _, diagnostics = build_evidence(
        graph_batch,
        api_logits,
        graph_logits,
        manifest_logits,
        joint_logits,
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        use_consistency_evidence=False,
        use_conflict_evidence=False,
        diagnostics_only=True,
    )

    assert diagnostics["graph_encoder_coverage"].item() == pytest.approx(0.5)
    assert diagnostics["graph_truncated_by_encoder_budget"].item() == pytest.approx(1.0)
    assert diagnostics["graph_integrity_before_encoder_budget"].item() == pytest.approx(0.8)
    assert diagnostics["effective_graph_integrity"].item() == pytest.approx(0.4)

def test_graph_visible_modifier_does_not_double_count_coverage():
    fusion = DiscountProbabilityFusion({
        "combination": "yager",
        "use_reliability_discount": False,
        "visible_integrity_modifier": {
            "enabled": True,
            "beta": 1.0,
            "min_value": 0.5,
        },
    })
    fusion.set_visible_integrity_reference([1.0, 1.0, 1.0])

    evidence = torch.ones(1, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.GRAPH_INTEGRITY] = 0.8
    evidence[:, EvidenceIndex.GRAPH_ENCODER_COVERAGE] = 0.5

    out = fusion(*_logits(1), evidence)

    assert torch.allclose(out["effective_graph_integrity"], torch.tensor([0.4]))
    assert torch.allclose(out["visible_modifier_graph"], torch.tensor([0.4]))
    assert torch.allclose(out["visible_modifier_factor_graph"], torch.tensor([0.7]))
    assert torch.allclose(out["discount_graph"], torch.tensor([0.7]))

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
    assert metrics["api_reliability_auc_defined"] == 1
    assert metrics["api_reliability_ap_defined"] == 1
    assert metrics["api_reliability_auc"] == pytest.approx(1.0)
    assert metrics["api_reliability_brier"] == pytest.approx(0.025)
    assert metrics["api_branch_accuracy"] == pytest.approx(0.5)
    assert metrics["api_reliability_mean"] == pytest.approx(0.5)
    assert metrics["graph_reliability_count"] == 1
    assert metrics["graph_reliability_auc_defined"] == 0
    assert metrics["graph_reliability_ap_defined"] == 0
    assert math.isnan(metrics["graph_reliability_auc"])
    assert math.isnan(metrics["graph_reliability_ap"])


def test_metrics_json_replaces_nonfinite_values_with_null(tmp_path):
    path = tmp_path / "metrics.json"
    _write_metrics_json(
        path,
        {"auc": float("nan"), "nested": [float("inf"), 1.0]},
    )

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {"auc": None, "nested": [None, 1.0]}



def test_estimate_branch_competence_prior_uses_validation_branch_macro_f1():
    rows = [
        {"label": 0, "api_pred": 0, "graph_pred": 0, "manifest_pred": 1, "joint_pred": 0, "api_prob": 0.1, "graph_prob": 0.2, "manifest_prob": 0.9, "joint_prob": 0.1},
        {"label": 1, "api_pred": 1, "graph_pred": 0, "manifest_pred": 0, "joint_pred": 1, "api_prob": 0.9, "graph_prob": 0.4, "manifest_prob": 0.2, "joint_prob": 0.8},
        {"label": 0, "api_pred": 0, "graph_pred": 0, "manifest_pred": 0, "joint_pred": 0, "api_prob": 0.2, "graph_prob": 0.3, "manifest_prob": 0.4, "joint_prob": 0.2},
        {"label": 1, "api_pred": 1, "graph_pred": 0, "manifest_pred": 0, "joint_pred": 1, "api_prob": 0.8, "graph_prob": 0.2, "manifest_prob": 0.3, "joint_prob": 0.9},
    ]
    summary = estimate_branch_competence_prior(
        rows,
        {
            "fusion": {
                "branch_competence_prior": {
                    "enabled": True,
                    "metric": "macro_f1",
                    "normalization": "best",
                    "min_value": 0.5,
                }
            }
        },
    )

    assert summary["enabled"] is True
    assert summary["prior"]["api"] == pytest.approx(1.0)
    assert summary["prior"]["joint"] == pytest.approx(1.0)
    assert summary["prior"]["manifest"] < summary["prior"]["api"]
    assert summary["prior"]["graph"] == pytest.approx(0.5)
    assert summary["counts"] == {"api": 4, "graph": 4, "manifest": 4, "joint": 4}


def test_estimate_model_visible_integrity_reference_uses_clean_effective_median():
    rows = [
        {
            "api_integrity": 0.8,
            "api_encoder_coverage": 0.5,
            "graph_integrity": 0.9,
            "graph_encoder_coverage": 1.0,
            "manifest_integrity": 0.7,
        },
        {
            "api_integrity": 1.0,
            "api_encoder_coverage": 0.8,
            "graph_integrity": 0.7,
            "graph_encoder_coverage": 0.5,
            "manifest_integrity": 0.9,
        },
    ]
    summary = estimate_model_visible_integrity_reference(
        rows,
        {
            "fusion": {
                "visible_integrity_modifier": {
                    "enabled": True,
                    "beta": 2.0,
                    "min_value": 0.4,
                }
            }
        },
    )

    assert summary["enabled"] is True
    assert summary["reference"]["api"] == pytest.approx(0.6)
    assert summary["reference"]["graph"] == pytest.approx(0.625)
    assert summary["reference"]["manifest"] == pytest.approx(0.8)
    assert summary["values"] == pytest.approx([0.6, 0.625, 0.8])
    assert summary["beta"] == 2.0
    assert summary["min_value"] == 0.4
