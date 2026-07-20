import math

import pytest
import torch

from fusion.train import _metrics, evaluate, evaluate_checkpoint_selection


class _LogitGraph:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def to(self, device, non_blocking=True):
        self.logits = self.logits.to(device)
        return self


class _StaticCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, graph, return_features=False):
        self.forward_calls += 1
        batch_size = graph.logits.size(0)
        extra = {
            "fusion_weight_api": graph.logits.new_full(
                (batch_size,), 1.0 / 3.0
            ),
            "fusion_weight_graph": graph.logits.new_full(
                (batch_size,), 1.0 / 3.0
            ),
            "fusion_weight_manifest": graph.logits.new_full(
                (batch_size,), 1.0 / 3.0
            ),
        }
        return graph.logits, extra


def _loader():
    return [
        {
            "graph_batch": _LogitGraph(
                torch.tensor([[2.0, 0.0], [0.0, 1.0]])
            ),
            "labels": torch.tensor([0, 1]),
            "sids": ["a", "b"],
            "quality": {},
            "num_failed": 1,
        },
        {
            "graph_batch": _LogitGraph(
                torch.tensor([[0.4, 0.2], [0.1, 0.3]])
            ),
            "labels": torch.tensor([1, 0]),
            "sids": ["c", "d"],
            "quality": {},
            "num_failed": 0,
        },
    ]


def test_checkpoint_selection_profile_matches_full_clean_metrics():
    full_model = _StaticCheckpointModel()
    full_metrics, full_rows = evaluate(
        full_model,
        _loader(),
        torch.device("cpu"),
        False,
        "full_validation",
        dump_rows=False,
    )
    lean_model = _StaticCheckpointModel()
    checkpoint_metrics = evaluate_checkpoint_selection(
        lean_model,
        _loader(),
        torch.device("cpu"),
        False,
    )

    core_metric_keys = set(_metrics([], [], []))
    assert set(checkpoint_metrics) == core_metric_keys | {"num_failed", "num_eval"}
    for key in core_metric_keys:
        full_value = full_metrics[key]
        checkpoint_value = checkpoint_metrics[key]
        if isinstance(full_value, float) and math.isnan(full_value):
            assert math.isnan(checkpoint_value)
        else:
            assert checkpoint_value == pytest.approx(full_value, abs=0.0, rel=0.0)
    assert checkpoint_metrics["num_failed"] == full_metrics["num_failed"] == 1
    assert checkpoint_metrics["num_eval"] == full_metrics["num_eval"] == 4
    assert full_rows == []
    assert "mean_fusion_weight_api" in full_metrics
    assert "mean_fusion_weight_api" not in checkpoint_metrics
    assert "aurc" in full_metrics
    assert "aurc" not in checkpoint_metrics
    assert full_model.forward_calls == lean_model.forward_calls == 2

