from __future__ import annotations

import pytest
import torch

from fusion.constants import EvidenceIndex
from fusion.losses import (
    ROUTING_PROBABILITY_SUBSETS,
    routing_source_subset_oracle_target,
    routing_subset_oracle_per_sample_loss,
)


BRANCHES = ("api", "graph", "manifest")


def _evidence(alive_rows: list[tuple[int, int, int]]) -> torch.Tensor:
    value = torch.zeros(len(alive_rows), EvidenceIndex.BASE_DIM)
    value[:, :3] = 1.0
    for row, alive in enumerate(alive_rows):
        value[row, EvidenceIndex.API_ALIVE] = alive[0]
        value[row, EvidenceIndex.GRAPH_ALIVE] = alive[1]
        value[row, EvidenceIndex.MANIFEST_ALIVE] = alive[2]
    return value


def _outputs(true_probabilities: torch.Tensor, *, requires_grad: bool = False) -> dict:
    # All tests use malware labels, so column one is the declared true-class
    # probability. Shape: [B, branch].
    branch_probabilities = torch.stack(
        (1.0 - true_probabilities, true_probabilities), dim=-1
    ).clamp(1.0e-6, 1.0 - 1.0e-6)
    outputs = {
        "routing_scores": torch.zeros(
            true_probabilities.size(0), 3, requires_grad=True
        )
    }
    for index, branch in enumerate(BRANCHES):
        log_prob = branch_probabilities[:, index].log().detach().clone()
        log_prob.requires_grad_(requires_grad)
        outputs[f"calibrated_log_prob_{branch}"] = log_prob
    return outputs


def test_source_subset_oracle_finds_complementary_pair():
    outputs = _outputs(
        torch.tensor(
            [
                [0.99, 0.20, 0.30],
                [0.20, 0.99, 0.30],
            ]
        )
    )
    target, valid, diagnostics = routing_source_subset_oracle_target(
        outputs,
        torch.ones(2, dtype=torch.long),
        _evidence([(1, 1, 1), (1, 1, 1)]),
        temperature=0.01,
    )

    pair_index = ROUTING_PROBABILITY_SUBSETS.index(("api", "graph"))
    assert diagnostics["hard_best_subset_index"].tolist() == [pair_index]
    assert diagnostics["eligible_candidate_count"].tolist() == [7]
    assert diagnostics["candidate_nll"][0, pair_index] == pytest.approx(
        -torch.log(torch.tensor(0.595)).item(), rel=1.0e-5
    )
    assert valid.tolist() == [True, True]
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(2))
    assert float(target[:, 2].max()) < 1.0e-4


def test_source_subset_oracle_respects_source_boundaries():
    outputs = _outputs(
        torch.tensor(
            [
                [0.95, 0.20, 0.20],
                [0.90, 0.25, 0.20],
                [0.20, 0.95, 0.20],
                [0.25, 0.90, 0.20],
            ]
        )
    )
    target, _valid, diagnostics = routing_source_subset_oracle_target(
        outputs,
        torch.ones(4, dtype=torch.long),
        _evidence([(1, 1, 1)] * 4),
        source_segments=[(0, 2), (2, 4)],
        temperature=0.01,
    )

    assert diagnostics["hard_best_subset_index"].tolist() == [0, 1]
    assert float(target[:2, 0].mean()) > 0.99
    assert float(target[2:, 1].mean()) > 0.99


def test_source_subset_oracle_requires_candidate_to_cover_native_missing_rows():
    outputs = _outputs(
        torch.tensor(
            [
                [0.99, 0.80, 0.30],
                [0.99, 0.80, 0.30],
            ]
        )
    )
    target, valid, diagnostics = routing_source_subset_oracle_target(
        outputs,
        torch.ones(2, dtype=torch.long),
        _evidence([(0, 1, 1), (1, 1, 1)]),
        temperature=0.1,
    )

    assert torch.isinf(diagnostics["candidate_nll"][0, 0])
    assert diagnostics["eligible_candidate_count"].tolist() == [6]
    assert diagnostics["candidate_mass"][0, 0].item() == 0.0
    assert target[0, 0].item() == 0.0
    assert valid.tolist() == [True, True]
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(2))


def test_source_subset_oracle_all_dead_is_explicitly_invalid_and_finite():
    outputs = _outputs(torch.tensor([[0.9, 0.8, 0.7]]))
    target, valid, diagnostics = routing_source_subset_oracle_target(
        outputs,
        torch.ones(1, dtype=torch.long),
        _evidence([(0, 0, 0)]),
    )
    assert not valid.any()
    assert target.eq(0.0).all()
    assert diagnostics["hard_best_subset_index"].tolist() == [-1]
    assert diagnostics["eligible_candidate_count"].tolist() == [0]
    assert diagnostics["candidate_mass"].eq(0.0).all()
    assert not torch.isnan(diagnostics["candidate_nll"]).any()


def test_source_subset_oracle_detaches_branches_and_updates_route_scores_only():
    outputs = _outputs(
        torch.tensor([[0.9, 0.7, 0.2], [0.3, 0.8, 0.6]]),
        requires_grad=True,
    )
    target, valid, _diagnostics = routing_source_subset_oracle_target(
        outputs,
        torch.ones(2, dtype=torch.long),
        _evidence([(1, 1, 1), (1, 1, 1)]),
        temperature=0.2,
    )
    per_row = routing_subset_oracle_per_sample_loss(outputs, target)
    loss = per_row[valid].mean()
    loss.backward()

    assert outputs["routing_scores"].grad is not None
    assert torch.isfinite(outputs["routing_scores"].grad).all()
    for branch in BRANCHES:
        assert outputs[f"calibrated_log_prob_{branch}"].grad is None


def test_source_subset_oracle_rejects_a_forced_single_candidate():
    outputs = _outputs(
        torch.tensor(
            [
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
            ]
        )
    )
    with pytest.raises(RuntimeError, match="only one of the seven candidates"):
        routing_source_subset_oracle_target(
            outputs,
            torch.ones(3, dtype=torch.long),
            _evidence([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        )
