from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from paper.baselines import drebin_style_sparse
from paper.baselines import maldozer_inspired_api_sequence
from paper.baselines import mamadroid_inspired_markov
from paper.baselines.maldozer_inspired_api_sequence import load_api_sequences
from paper.baselines.mamadroid_inspired_markov import extract_markov_features
from paper.baselines.common import (
    enforce_formal_split_completeness,
    read_label_csv,
    validation_selection_indices,
)
from paper.run_trusted_fusion_baselines import (
    FUSION_RULE_TARGETS,
    FORMAL_TARGETS,
    METHOD_TARGETS,
)


def test_paper_validation_selection_matches_disjoint_group_holdout():
    frame = pd.DataFrame(
        {
            "sha256": [f"sha-{index}" for index in range(20)],
            "label": [index % 2 for index in range(20)],
            "year": [2020 + index % 2 for index in range(20)],
            "pkg_name": [f"pkg-{index // 2}" for index in range(20)],
        }
    )

    selection, summary = validation_selection_indices(
        frame, calibration_fraction=0.5, seed=42
    )

    calibration = summary["calibration_indices"]
    assert selection == summary["selection_indices"]
    assert set(selection).isdisjoint(calibration)
    assert sorted(selection + calibration) == list(range(len(frame)))
    selection_groups = set(frame.iloc[selection]["pkg_name"])
    calibration_groups = set(frame.iloc[calibration]["pkg_name"])
    assert selection_groups.isdisjoint(calibration_groups)


def test_paper_validation_selection_defaults_to_formal_40_percent_budget():
    frame = pd.DataFrame(
        {
            "sha256": [f"sha-{index}" for index in range(20)],
            "label": [index % 2 for index in range(20)],
            "year": [2020 + index % 2 for index in range(20)],
            "pkg_name": [f"pkg-{index // 2}" for index in range(20)],
        }
    )

    selection, summary = validation_selection_indices(frame, seed=42)

    assert summary["validation_fraction"] == pytest.approx(0.60)
    assert summary["selection_fraction_of_validation"] == pytest.approx(0.40)
    assert len(selection) == 8


def test_maldozer_empty_api_sequence_is_padding_only(tmp_path):
    sid = "a" * 64
    torch.save(
        {"dex_list": [{"api_ids": torch.empty((0,), dtype=torch.long)}]},
        tmp_path / f"{sid}.pt",
    )
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"sha256": [sid], "label": [1]}).to_csv(csv_path, index=False)

    dataset, _frame, failures = load_api_sequences(
        tmp_path, csv_path, max_len=2048, vocab_size=8192, show_progress=False
    )

    assert not failures
    assert dataset.sequences[0].tolist() == [0]


def test_mamadroid_empty_api_sequence_has_no_artificial_state_occupancy():
    features = extract_markov_features(
        {"dex_list": [{"api_type_ids": torch.empty((0,), dtype=torch.long)}]},
        num_states=4,
        max_api_events=16,
        smoothing=1.0e-3,
    )

    occupancy = features[16:20]
    assert occupancy.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert features[-1] == pytest.approx(0.0)


def test_paper_targets_separate_method_baselines_from_fusion_rules():
    assert METHOD_TARGETS == {
        "ours": "final",
        "tmc": "tmc",
        "ecml": "ecml",
        "qmf_energy": "qmf_energy",
    }
    assert FUSION_RULE_TARGETS == {
        "dempster_rule_only": "dempster",
        "cumulative_subjective_logic": "cumulative",
        "log_pool": "log_pool",
        "conflict_weighted_opinion": "conflict_weighted_opinion",
    }
    assert FORMAL_TARGETS == {**METHOD_TARGETS, **FUSION_RULE_TARGETS}


def test_paper_targets_have_no_removed_style_or_adapted_entrypoints():
    formal_tokens = [*FORMAL_TARGETS, *FORMAL_TARGETS.values()]
    assert all(
        "style" not in token and "adapted" not in token
        for token in formal_tokens
    )


def test_paper_label_reader_rejects_duplicate_samples(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"sha256": ["ABC", "abc"], "label": [0, 1]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate sha256"):
        read_label_csv(path)


@pytest.mark.parametrize("label", [0.5, 1.5, 2, float("nan"), float("inf"), True])
def test_paper_label_reader_rejects_lossy_or_nonfinite_labels(tmp_path, label):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"sha256": ["sample"], "label": [label]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="finite binary integers"):
        read_label_csv(path)


def test_validation_selection_defensively_rejects_fractional_labels():
    frame = pd.DataFrame(
        {
            "sha256": ["a", "b"],
            "label": [0.5, 1.0],
            "pkg_name": ["pkg.a", "pkg.b"],
        }
    )

    with pytest.raises(ValueError, match="finite binary integers"):
        validation_selection_indices(frame)


@pytest.mark.parametrize(
    ("num_eval", "failures"),
    [
        (0, []),
        (1, [{"sha256": "bad", "error": "missing PT"}]),
    ],
)
def test_formal_baseline_split_completeness_is_fail_closed(num_eval, failures):
    with pytest.raises(RuntimeError, match="refuses success-subset metrics"):
        enforce_formal_split_completeness(
            "test",
            num_eval=num_eval,
            failures=failures,
        )


def test_formal_baseline_split_completeness_accepts_complete_nonempty_split():
    enforce_formal_split_completeness("test", num_eval=1, failures=[])


class _PredictionMustNotRun:
    def predict_proba(self, _features):
        raise AssertionError("prediction ran before the failure guard")


def test_drebin_evaluation_rejects_partial_split_before_prediction(monkeypatch, tmp_path):
    frame = pd.DataFrame({"sha256": ["ok"], "label": [0]})
    monkeypatch.setattr(
        drebin_style_sparse,
        "build_feature_dicts",
        lambda *_args, **_kwargs: (
            [{"feature": 1.0}],
            np.asarray([0], dtype=np.int64),
            frame,
            [{"sha256": "bad", "error": "missing PT"}],
        ),
    )

    with pytest.raises(RuntimeError, match="refuses success-subset metrics"):
        drebin_style_sparse.evaluate_split(
            _PredictionMustNotRun(),
            tmp_path,
            tmp_path / "labels.csv",
            tmp_path,
            "test",
            max_api_events=8,
            show_progress=False,
        )


def test_maldozer_evaluation_rejects_partial_split_before_prediction(monkeypatch, tmp_path):
    frame = pd.DataFrame({"sha256": ["ok"], "label": [0]})
    dataset = maldozer_inspired_api_sequence.ApiSequenceDataset(
        [torch.tensor([1], dtype=torch.long)],
        np.asarray([0], dtype=np.int64),
        ["ok"],
    )
    monkeypatch.setattr(
        maldozer_inspired_api_sequence,
        "load_api_sequences",
        lambda *_args, **_kwargs: (
            dataset,
            frame,
            [{"sha256": "bad", "error": "missing PT"}],
        ),
    )
    monkeypatch.setattr(
        maldozer_inspired_api_sequence,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prediction ran before the failure guard")
        ),
    )

    with pytest.raises(RuntimeError, match="refuses success-subset metrics"):
        maldozer_inspired_api_sequence.evaluate_split(
            object(),
            tmp_path,
            tmp_path / "labels.csv",
            tmp_path,
            "test",
            max_len=8,
            vocab_size=16,
            batch_size=1,
            device=torch.device("cpu"),
            show_progress=False,
        )


def test_mamadroid_evaluation_rejects_partial_split_before_prediction(monkeypatch, tmp_path):
    frame = pd.DataFrame({"sha256": ["ok"], "label": [0]})
    monkeypatch.setattr(
        mamadroid_inspired_markov,
        "build_features",
        lambda *_args, **_kwargs: (
            np.ones((1, 4), dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            frame,
            [{"sha256": "bad", "error": "missing PT"}],
        ),
    )

    with pytest.raises(RuntimeError, match="refuses success-subset metrics"):
        mamadroid_inspired_markov.evaluate_split(
            _PredictionMustNotRun(),
            tmp_path,
            tmp_path / "labels.csv",
            tmp_path,
            "test",
            num_states=2,
            max_api_events=8,
            smoothing=1.0e-3,
            show_progress=False,
        )
