from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.care_training import (
    CARE_PATH_NAMES,
    CareRiskCalibrationCache,
    CareRiskCalibrationData,
    atomic_score_groups,
    care_path_validity,
    clone_state_dict_to_cpu,
    deterministic_view_seed,
    deterministic_view_spec,
    fit_atomic_crc_correctness_threshold,
    fit_atomic_crc_risk_threshold,
    fit_care_risk_crossfit,
    fit_valid_path_log_odds_normalizer,
    fixed_path_correctness_targets,
    fixed_path_predictions,
    sid_view_valid_path_bce,
    tensor_state_dict_sha256,
)
from fusion.care_fusion import CAREPathRiskHead


class _StubCarePathRiskHead(nn.Module):
    """Small implementation of the frozen CARE risk-head boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("log_odds_center", torch.zeros(4))
        self.register_buffer("log_odds_scale", torch.ones(4))
        self.linear = nn.Linear(11, 1)

    def set_log_odds_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.log_odds_center.copy_(center.to(self.log_odds_center))
        self.log_odds_scale.copy_(scale.to(self.log_odds_scale))

    def forward(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
        candidate_path_index: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(
            candidate_path_index,
            num_classes=4,
        ).to(normalized_log_odds)
        features = torch.cat(
            [
                normalized_log_odds,
                modality_alive.to(normalized_log_odds),
                one_hot,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.linear(features).squeeze(-1))

    def score_all(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(normalized_log_odds.size(0))
        path_index = torch.arange(
            4,
            device=normalized_log_odds.device,
        ).view(1, 4).expand(batch_size, -1)
        expanded_odds = normalized_log_odds.unsqueeze(1).expand(
            -1, 4, -1
        )
        expanded_alive = modality_alive.unsqueeze(1).expand(-1, 4, -1)
        return self(
            expanded_odds.reshape(-1, 4),
            expanded_alive.reshape(-1, 3),
            path_index.reshape(-1),
        ).view(batch_size, 4)


def _routing_cache(num_sids: int = 12) -> CareRiskCalibrationCache:
    if num_sids % 2:
        raise ValueError("test cache uses a balanced binary population")
    sids = tuple(f"sid-{index:02d}" for index in range(num_sids))
    groups = tuple(f"package-{index:02d}" for index in range(num_sids))
    labels = torch.tensor(
        [index % 2 for index in range(num_sids)],
        dtype=torch.long,
    )
    view_names = ("clean", "api_event_dropout")
    raw = torch.arange(
        num_sids * len(view_names) * 4,
        dtype=torch.float32,
    ).reshape(num_sids, len(view_names), 4)
    path_log_odds = (raw.remainder(11.0) - 5.0) / 2.0
    # All four candidate paths exist, keeping every fold well supported.
    modality_alive = torch.ones(
        num_sids,
        len(view_names),
        3,
        dtype=torch.bool,
    )
    return CareRiskCalibrationData.from_path_log_odds(
        sids=sids,
        groups=groups,
        labels=labels,
        view_names=view_names,
        path_log_odds=path_log_odds,
        modality_alive=modality_alive,
    )


def test_sid_view_valid_path_bce_uses_three_equal_weight_layers() -> None:
    alive = torch.tensor(
        [
            [[True, True, True], [True, True, False]],
            [[True, False, True], [False, False, False]],
        ]
    )
    valid = care_path_validity(alive)
    probability = torch.full((2, 2, 4), float("nan"))
    target = torch.zeros((2, 2, 4))

    # SID 0: view losses 0 and 4 -> SID loss 2.
    probability[0, 0] = 1.0
    target[0, 0] = 1.0
    probability[0, 1, 1] = float(np.exp(-4.0))
    target[0, 1, 1] = 1.0
    # SID 1: one valid AM path with loss 1 -> SID loss 1.
    probability[1, 0, 2] = float(np.exp(-1.0))
    target[1, 0, 2] = 1.0

    result = sid_view_valid_path_bce(probability, target, valid)

    assert result.per_view_loss[0, 0] == pytest.approx(0.0)
    assert result.per_view_loss[0, 1] == pytest.approx(4.0)
    assert result.per_sid_loss.tolist() == pytest.approx([2.0, 1.0])
    assert result.loss == pytest.approx(1.5)
    assert result.valid_path_count_per_view.tolist() == [[4, 1], [1, 0]]
    assert result.valid_view_count_per_sid.tolist() == [2, 1]
    # A flat path average would be 5/6 and is explicitly not the method.
    assert result.loss.item() != pytest.approx(5.0 / 6.0)


def test_log_odds_normalizer_uses_only_valid_paths_and_zero_placeholder() -> None:
    values = torch.tensor(
        [
            [[1.0, 5.0, float("nan"), 0.0], [9.0e8, 5.0, -8.0, 4.0]],
            [[3.0, -9.0e8, 7.0, -7.0], [float("nan"), 2.0, 6.0, 8.0]],
        ]
    )
    valid = torch.tensor(
        [
            [[True, True, False, True], [False, True, False, True]],
            [[True, False, False, False], [False, False, False, False]],
        ]
    )

    normalizer = fit_valid_path_log_odds_normalizer(values, valid)
    normalized = normalizer.transform(values, valid)

    assert normalizer.center.tolist() == pytest.approx([2.0, 5.0, 0.0, 2.0])
    assert normalizer.scale.tolist() == pytest.approx([1.0, 1.0, 1.0, 2.0])
    assert normalizer.valid_count.tolist() == [2, 2, 0, 2]
    assert normalized[0, 0, 0] == pytest.approx(-1.0)
    assert normalized[1, 0, 0] == pytest.approx(1.0)
    assert normalized[0, 0, 3] == pytest.approx(-1.0)
    assert normalized[0, 1, 3] == pytest.approx(1.0)
    assert torch.equal(
        normalized[~valid],
        torch.zeros_like(normalized[~valid]),
    )
    assert torch.isfinite(normalized).all()


def test_deterministic_view_spec_is_stable_and_identity_bound() -> None:
    first = deterministic_view_spec(
        "sample-a",
        "api_event_dropout",
        4242,
        0.1,
        0.9,
    )
    second = deterministic_view_spec(
        "sample-a",
        "api_event_dropout",
        4242,
        0.1,
        0.9,
    )
    changed = deterministic_view_spec(
        "sample-b",
        "api_event_dropout",
        4242,
        0.1,
        0.9,
    )

    assert first == second
    assert first != changed
    assert first["seed"] == deterministic_view_seed(
        "sample-a",
        "api_event_dropout",
        4242,
    )
    assert 0.1 <= first["strength"] <= 0.9
    assert deterministic_view_spec(
        "sample-a",
        "graph_sparsify",
        7,
        0.3,
        0.3,
    )["strength"] == pytest.approx(0.3)


def test_four_path_validity_and_zero_log_odds_hard_rule_are_frozen() -> None:
    alive = torch.tensor(
        [
            [[True, True, True]],
            [[True, True, False]],
            [[True, False, True]],
            [[False, True, True]],
        ]
    )
    expected_valid = torch.tensor(
        [
            [[True, True, True, True]],
            [[False, True, False, False]],
            [[False, False, True, False]],
            [[False, False, False, True]],
        ]
    )
    assert CARE_PATH_NAMES == ("agm", "ag", "am", "gm")
    assert torch.equal(care_path_validity(alive), expected_valid)

    odds = torch.zeros(4, 1, 4)
    assert torch.equal(
        fixed_path_predictions(odds),
        torch.ones_like(odds, dtype=torch.bool),
    )
    labels = torch.tensor([1, 0, 1, 0])
    target = fixed_path_correctness_targets(odds, labels, expected_valid)
    assert target[0, 0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert target[1, 0, 1].item() == 0.0
    assert target[2, 0, 2].item() == 1.0
    assert target[3, 0, 3].item() == 0.0

    logits = torch.stack(
        [torch.zeros_like(odds), odds],
        dim=-1,
    )
    cache = CareRiskCalibrationCache.from_path_logits(
        sids=("a", "b", "c", "d"),
        groups=("ga", "gb", "gc", "gd"),
        labels=labels,
        view_names=("clean",),
        path_logits=logits,
        modality_alive=alive,
    )
    assert torch.equal(cache.path_log_odds, odds)
    assert torch.equal(cache.correctness_targets, target)


def test_three_fold_crossfit_records_provenance_and_full_refit() -> None:
    torch.manual_seed(17)
    cache = _routing_cache()
    head = _StubCarePathRiskHead()
    model = SimpleNamespace(care_risk_head=head)

    result = fit_care_risk_crossfit(
        model,
        cache,
        device="cpu",
        folds=3,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=5.0,
        protocol_seed=73,
    )

    assert result.risk_head is head
    assert result.oof_correctness_probability.shape == (12, 2, 4)
    assert torch.isfinite(result.oof_correctness_probability).all()
    assert bool(
        (
            (result.oof_correctness_probability >= 0.0)
            & (result.oof_correctness_probability <= 1.0)
        ).all()
    )
    assert sorted(result.fold_assignment.unique().tolist()) == [0, 1, 2]
    assert result.summary["folds"] == 3
    assert result.summary["early_stopping"] is False
    assert result.summary["structure_selection"] is False
    assert result.summary["optimizer"] == "adamw"
    assert result.summary["optimizer_update_unit"] == "mini_batch"
    assert result.summary["final_refit_population"] == "all_routing_cal"
    assert len(result.summary["final_history"]) == 2
    assert all(
        row["optimizer_steps"] > 1
        for row in result.summary["final_history"]
    )
    assert len(result.summary["routing_cal_sid_sha256"]) == 64
    assert len(result.summary["routing_cal_sid_view_sha256"]) == 64
    assert result.summary["view_names"] == ["clean", "api_event_dropout"]
    assert result.summary["path_names"] == list(CARE_PATH_NAMES)
    assert len(result.fold_state_dicts) == 3

    all_holdout_sids: list[str] = []
    for fold, fold_state in zip(
        result.summary["folds_summary"],
        result.fold_state_dicts,
        strict=True,
    ):
        assert fold["fixed_epochs"] == 2
        assert len(fold["history"]) == 2
        assert fold["optimizer_update_unit"] == "mini_batch"
        assert all(row["optimizer_steps"] > 0 for row in fold["history"])
        assert fold["holdout_used_for_optimization"] is False
        assert fold["holdout_used_for_early_stopping"] is False
        assert fold["holdout_used_for_structure_selection"] is False
        assert set(fold["train_sids"]).isdisjoint(fold["holdout_sids"])
        assert len(fold["train_sid_sha256"]) == 64
        assert len(fold["holdout_sid_sha256"]) == 64
        assert len(fold["train_sid_view_sha256"]) == 64
        assert len(fold["holdout_sid_view_sha256"]) == 64
        assert len(fold["normalization"]["mu"]) == 4
        assert len(fold["normalization"]["sigma"]) == 4
        assert fold["normalization"]["invalid_path_placeholder"] == 0.0
        assert len(fold["state_dict_sha256"]) == 64
        assert fold["state_dict_sha256"] == tensor_state_dict_sha256(
            fold_state
        )
        assert all(
            value.device.type == "cpu" and not value.requires_grad
            for value in fold_state.values()
        )
        assert torch.allclose(
            fold_state["log_odds_center"],
            torch.tensor(fold["normalization"]["mu"]),
        )
        assert torch.allclose(
            fold_state["log_odds_scale"],
            torch.tensor(fold["normalization"]["sigma"]),
        )
        all_holdout_sids.extend(fold["holdout_sids"])
    assert sorted(all_holdout_sids) == sorted(cache.sids)
    assert torch.allclose(
        head.log_odds_center,
        result.final_normalizer.center,
    )
    assert torch.allclose(
        head.log_odds_scale,
        result.final_normalizer.scale,
    )
    assert result.summary["final_state_dict_sha256"] == (
        tensor_state_dict_sha256(head.state_dict())
    )
    assert all(not parameter.requires_grad for parameter in head.parameters())
    preserved_fold_weight = result.fold_state_dicts[0][
        "linear.weight"
    ].clone()
    with torch.no_grad():
        head.linear.weight.add_(1.0)
    assert torch.equal(
        result.fold_state_dicts[0]["linear.weight"],
        preserved_fold_weight,
    )


def test_actual_care_risk_head_satisfies_crossfit_boundary() -> None:
    torch.manual_seed(23)
    head = CAREPathRiskHead(hidden_dim=8)

    result = fit_care_risk_crossfit(
        head,
        _routing_cache(),
        device="cpu",
        folds=3,
        epochs=1,
        batch_size=6,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=5.0,
        protocol_seed=79,
    )

    assert result.risk_head is head
    assert bool(head.normalization_is_fitted.item())
    assert torch.allclose(
        head.log_odds_center,
        result.final_normalizer.center,
    )
    assert torch.allclose(
        head.log_odds_scale,
        result.final_normalizer.scale,
    )
    assert torch.isfinite(result.oof_correctness_probability).all()


def test_crossfit_skips_a_sid_with_no_valid_path_without_reweighting() -> None:
    original = _routing_cache()
    alive = original.modality_alive.clone()
    alive[0] = False
    cache = CareRiskCalibrationData.from_path_log_odds(
        sids=original.sids,
        groups=original.groups,
        labels=original.labels,
        view_names=original.view_names,
        path_log_odds=original.path_log_odds,
        modality_alive=alive,
    )
    torch.manual_seed(31)

    result = fit_care_risk_crossfit(
        _StubCarePathRiskHead(),
        cache,
        folds=3,
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=5.0,
        protocol_seed=83,
    )

    assert torch.equal(
        result.oof_correctness_probability[0],
        torch.zeros_like(result.oof_correctness_probability[0]),
    )
    assert torch.isfinite(result.oof_correctness_probability).all()


def test_oof_holdout_targets_cannot_change_their_fold_fit() -> None:
    cache = _routing_cache()
    torch.manual_seed(29)
    initial = _StubCarePathRiskHead()
    first_head = copy.deepcopy(initial)
    second_head = copy.deepcopy(initial)
    first = fit_care_risk_crossfit(
        first_head,
        cache,
        device="cpu",
        folds=3,
        epochs=1,
        batch_size=5,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=5.0,
        protocol_seed=101,
    )
    held_out_sids = first.summary["folds_summary"][0]["holdout_sids"]
    held_out_indices = torch.tensor(
        [cache.sids.index(sid) for sid in held_out_sids],
        dtype=torch.long,
    )
    changed_targets = cache.correctness_targets.clone()
    changed_targets[held_out_indices] = (
        1.0 - changed_targets[held_out_indices]
    )
    changed = CareRiskCalibrationData(
        sids=cache.sids,
        groups=cache.groups,
        labels=cache.labels,
        view_names=cache.view_names,
        path_log_odds=cache.path_log_odds,
        modality_alive=cache.modality_alive,
        valid_paths=cache.valid_paths,
        correctness_targets=changed_targets,
    )
    second = fit_care_risk_crossfit(
        second_head,
        changed,
        device="cpu",
        folds=3,
        epochs=1,
        batch_size=5,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=5.0,
        protocol_seed=101,
    )

    assert torch.equal(
        first.fold_assignment[held_out_indices],
        torch.zeros(len(held_out_indices), dtype=torch.long),
    )
    assert torch.equal(
        second.fold_assignment[held_out_indices],
        torch.zeros(len(held_out_indices), dtype=torch.long),
    )
    assert torch.allclose(
        first.oof_correctness_probability[held_out_indices],
        second.oof_correctness_probability[held_out_indices],
        atol=0.0,
        rtol=0.0,
    )
    assert (
        first.summary["folds_summary"][0]["history"]
        == second.summary["folds_summary"][0]["history"]
    )


def test_crossfit_rejects_any_fold_count_other_than_three() -> None:
    with pytest.raises(ValueError, match="exactly 3 folds"):
        fit_care_risk_crossfit(
            _StubCarePathRiskHead(),
            _routing_cache(),
            folds=2,
            epochs=1,
        )


def test_atomic_score_groups_preserve_exact_float32_ties() -> None:
    lower = np.float32(1.0)
    upper = np.nextafter(lower, np.float32(np.inf))
    scores = torch.tensor([upper, lower, upper], dtype=torch.float32)

    groups = atomic_score_groups(scores)

    assert len(groups) == 2
    assert groups[0].score == float(lower)
    assert groups[0].indices == (1,)
    assert groups[1].score == float(upper)
    assert groups[1].indices == (0, 2)


def test_state_dict_digest_is_order_device_and_clone_stable() -> None:
    original = {
        "weight": torch.tensor([[1.0, 2.0]], requires_grad=True),
        "fitted": torch.tensor(True),
    }
    cloned = clone_state_dict_to_cpu(original)
    reversed_mapping = {
        "fitted": cloned["fitted"],
        "weight": cloned["weight"],
    }

    digest = tensor_state_dict_sha256(original)
    assert tensor_state_dict_sha256(cloned) == digest
    assert tensor_state_dict_sha256(reversed_mapping) == digest
    assert cloned["weight"].device.type == "cpu"
    assert cloned["weight"].requires_grad is False
    assert cloned["weight"].data_ptr() != original["weight"].data_ptr()

    cloned["weight"][0, 0] += 1.0
    assert tensor_state_dict_sha256(cloned) != digest


def test_crc_threshold_never_splits_a_tied_score_group() -> None:
    scores = torch.tensor(
        [0.1, 0.1, 0.2, 0.2, 0.2]
        + [0.3 + 0.01 * index for index in range(14)]
    )
    loss_events = torch.zeros(19, dtype=torch.bool)
    loss_events[0] = True
    loss_events[2] = True
    target_population = torch.ones(19, dtype=torch.bool)

    result = fit_atomic_crc_risk_threshold(
        scores,
        loss_events,
        target_population,
        risk_level=0.1,
    )

    # First tie group gives (1 + 1) / (19 + 1) == 0.1 and is feasible.
    assert result.threshold == pytest.approx(0.1)
    assert torch.equal(
        result.accepted[:5],
        torch.tensor([True, True, False, False, False]),
    )
    assert result.corrected_risk == pytest.approx(0.1)
    assert result.accepted_group_count == 1
    assert result.as_summary()["score_ties_are_atomic"] is True


def test_crc_boundary_is_inclusive_and_infeasible_floor_is_explicit() -> None:
    boundary = fit_atomic_crc_risk_threshold(
        torch.tensor([0.1, 0.2, 0.3]),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor(
            [True] * 9,
            dtype=torch.bool,
        )[:3],
        risk_level=0.25,
    )
    # target_count=3 => finite-sample floor 1/(3+1)=0.25.
    assert boundary.feasible is True
    assert boundary.accepted.tolist() == [True, True, True]
    assert boundary.corrected_risk == pytest.approx(0.25)

    infeasible = fit_atomic_crc_risk_threshold(
        torch.tensor([0.1, 0.2]),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([True, False]),
        risk_level=0.1,
    )
    assert infeasible.feasible is False
    assert infeasible.accepted.tolist() == [False, False]
    assert infeasible.threshold == float("-inf")
    assert infeasible.corrected_risk == pytest.approx(0.5)


def test_correctness_crc_uses_q_greater_equal_lambda_and_atomic_ties() -> None:
    correctness = torch.tensor(
        [0.9, 0.9, 0.8, 0.8, 0.8]
        + [0.7 - 0.01 * index for index in range(14)]
    )
    fn_events = torch.zeros(19, dtype=torch.bool)
    fn_events[0] = True
    fn_events[2] = True
    malware = torch.ones(19, dtype=torch.bool)

    result = fit_atomic_crc_correctness_threshold(
        correctness,
        fn_events,
        malware,
        alpha=0.1,
    )

    assert result.crc_status == "feasible"
    assert result.lambda_threshold == pytest.approx(
        float(correctness[0])
    )
    assert result.risk_threshold == pytest.approx(
        1.0 - result.lambda_threshold
    )
    assert torch.equal(
        result.accepted[:5],
        torch.tensor([True, True, False, False, False]),
    )
    assert torch.equal(
        result.accepted,
        correctness >= result.lambda_threshold,
    )
    assert result.corrected_risk == pytest.approx(0.1)
    summary = result.as_summary()
    assert summary["acceptance_comparison"] == (
        "eligible and correctness_q >= lambda"
    )
    assert summary["score_ties_are_atomic"] is True
    assert summary["N_malware"] == 19


def test_correctness_crc_reports_no_malware_and_insufficient_malware() -> None:
    no_malware = fit_atomic_crc_correctness_threshold(
        torch.tensor([0.8, 0.7]),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        alpha=0.05,
    )
    assert no_malware.crc_status == "failure_no_malware"
    assert no_malware.feasible is False
    assert no_malware.n_malware == 0
    assert no_malware.lambda_threshold == float("inf")
    assert no_malware.accepted.tolist() == [False, False]
    assert np.isnan(no_malware.corrected_risk)

    insufficient = fit_atomic_crc_correctness_threshold(
        torch.tensor([0.8, 0.7, 0.6]),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor([True, True, True]),
        alpha=0.2,
    )
    assert insufficient.crc_status == (
        "infeasible_insufficient_malware"
    )
    assert insufficient.feasible is False
    assert insufficient.n_malware == 3
    assert insufficient.corrected_risk == pytest.approx(0.25)
    assert insufficient.accepted.tolist() == [False, False, False]
