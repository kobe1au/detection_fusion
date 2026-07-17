from __future__ import annotations

import pandas as pd
import pytest
import torch

from paper.baselines.maldozer_inspired_api_sequence import load_api_sequences
from paper.baselines.mamadroid_inspired_markov import extract_markov_features
from paper.baselines.common import read_label_csv, validation_selection_indices
from paper.run_trusted_fusion_baselines import FORMAL_TARGETS


def test_paper_validation_selection_matches_disjoint_group_holdout():
    frame = pd.DataFrame(
        {
            "sha256": [f"sha-{index}" for index in range(20)],
            "label": [index % 2 for index in range(20)],
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


def test_trusted_fusion_targets_are_all_formal_comparisons():
    assert set(FORMAL_TARGETS) == {
        "ours",
        "tmc_dempster",
        "cumulative_subjective_logic",
        "log_pool",
        "ecml_style",
    }


def test_paper_label_reader_rejects_duplicate_samples(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"sha256": ["ABC", "abc"], "label": [0, 1]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate sha256"):
        read_label_csv(path)
