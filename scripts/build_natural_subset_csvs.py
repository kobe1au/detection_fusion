from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BRANCHES = ("api", "graph", "manifest")
SUBSET_SCHEMA_VERSION = 6
SUBSET_PROTOCOL_ID = "i1_i2_unseen_validation_natural_difficulty_v2"
SUBSET_CALIBRATION_SPLIT = "val_selection"
SUBSET_TARGET_SPLIT = "test_clean"

# These definitions are deliberately small and auditable:
# - disagreement is label-free;
# - exactly-one-wrong subsets are explicitly label-dependent diagnostics;
# - the two continuous difficulty thresholds are fit on validation only.
SUBSET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "branch_disagreement": {
        "label_dependency": "label_free",
        "threshold_source": "none",
        "eligibility": "at_least_two_alive_branches",
        "purpose": "natural_cross_modal_prediction_disagreement",
        "valid_claim_scope": "fusion_behavior_under_natural_disagreement",
    },
    "api_only_wrong": {
        "label_dependency": "uses_ground_truth_label",
        "threshold_source": "none",
        "eligibility": "all_three_branches_alive",
        "purpose": "diagnostic_exactly_one_branch_wrong",
        "valid_claim_scope": "posthoc_branch_failure_diagnosis_only",
    },
    "graph_only_wrong": {
        "label_dependency": "uses_ground_truth_label",
        "threshold_source": "none",
        "eligibility": "all_three_branches_alive",
        "purpose": "diagnostic_exactly_one_branch_wrong",
        "valid_claim_scope": "posthoc_branch_failure_diagnosis_only",
    },
    "manifest_only_wrong": {
        "label_dependency": "uses_ground_truth_label",
        "threshold_source": "none",
        "eligibility": "all_three_branches_alive",
        "purpose": "diagnostic_exactly_one_branch_wrong",
        "valid_claim_scope": "posthoc_branch_failure_diagnosis_only",
    },
    "reliability_imbalance": {
        "label_dependency": "label_free",
        "threshold_source": "validation_only",
        "eligibility": "at_least_two_alive_branches",
        "purpose": "natural_sample_level_branch_reliability_imbalance",
        "valid_claim_scope": "downstream_i2_routing_stress_diagnostic_not_i1_validation",
    },
    "high_cross_modal_conflict": {
        "label_dependency": "label_free",
        "threshold_source": "validation_only",
        "eligibility": "at_least_two_alive_branches",
        "purpose": "natural_probability_conflict",
        "valid_claim_scope": "fusion_behavior_under_validation_frozen_probability_conflict",
    },
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected_columns = ["_sample_id", *columns]
    canonical = (
        frame.loc[:, selected_columns]
        .sort_values("_sample_id", kind="stable")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(canonical).hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_id_column(frame: pd.DataFrame, preferred: str | None) -> str:
    candidates = [preferred] if preferred else []
    candidates.extend(["sid", "sha256", "id"])
    for column in candidates:
        if column and column in frame.columns:
            return column
    raise ValueError(
        "Could not infer sample id column from diagnostics. "
        "Expected one of: sid, sha256, id."
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _finite_numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    source: str,
    lower: float | None = None,
    upper: float | None = None,
    integer: bool = False,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    numeric = values.to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric)
    if lower is not None:
        valid &= numeric >= lower
    if upper is not None:
        valid &= numeric <= upper
    if integer:
        valid &= numeric == np.floor(numeric)
    if not bool(valid.all()):
        bad_ids = frame.loc[~valid, "_sample_id"].astype(str).tolist()[:10]
        bounds = (
            f" in [{lower}, {upper}]"
            if lower is not None and upper is not None
            else ""
        )
        raise ValueError(
            f"{source} column {column!r} must contain finite "
            f"{'integer ' if integer else ''}values{bounds}; examples={bad_ids}"
        )
    return values.astype(np.int64 if integer else np.float64)


def _prepare_split(
    diagnostics: pd.DataFrame,
    *,
    split: str,
    id_column: str,
    require_label: bool,
) -> pd.DataFrame:
    frame = diagnostics[diagnostics["split"].astype(str) == split].copy()
    if frame.empty:
        raise ValueError(f"No diagnostics rows found for split={split!r}")
    frame["_sample_id"] = frame[id_column].astype(str)
    duplicate = frame["_sample_id"].duplicated(keep=False)
    if bool(duplicate.any()):
        examples = sorted(frame.loc[duplicate, "_sample_id"].unique().tolist())[:10]
        raise ValueError(
            f"Diagnostics split={split!r} contains duplicate sample ids: {examples}"
        )

    required = [
        *(f"{branch}_alive" for branch in BRANCHES),
        *(f"{branch}_pred" for branch in BRANCHES),
        *(f"{branch}_prob" for branch in BRANCHES),
        *(f"predicted_reliability_{branch}" for branch in BRANCHES),
    ]
    if require_label:
        required.append("label")
    _require_columns(frame, required, f"diagnostics split={split!r}")

    for branch in BRANCHES:
        frame[f"{branch}_alive"] = _finite_numeric_column(
            frame,
            f"{branch}_alive",
            source=f"diagnostics split={split!r}",
            lower=0.0,
            upper=1.0,
            integer=True,
        )
        frame[f"{branch}_pred"] = _finite_numeric_column(
            frame,
            f"{branch}_pred",
            source=f"diagnostics split={split!r}",
            lower=0.0,
            upper=1.0,
            integer=True,
        )
        frame[f"{branch}_prob"] = _finite_numeric_column(
            frame,
            f"{branch}_prob",
            source=f"diagnostics split={split!r}",
            lower=0.0,
            upper=1.0,
        )
        frame[f"predicted_reliability_{branch}"] = _finite_numeric_column(
            frame,
            f"predicted_reliability_{branch}",
            source=f"diagnostics split={split!r}",
            lower=0.0,
            upper=1.0,
        )
    if require_label:
        frame["label"] = _finite_numeric_column(
            frame,
            "label",
            source=f"diagnostics split={split!r}",
            lower=0.0,
            upper=1.0,
            integer=True,
        )
    return frame


def _bernoulli_js(probability_a: float, probability_b: float) -> float:
    """Jensen-Shannon divergence between two binary predictive distributions."""

    eps = np.finfo(np.float64).eps
    p = np.clip(
        np.asarray([1.0 - probability_a, probability_a], dtype=np.float64),
        eps,
        1.0,
    )
    q = np.clip(
        np.asarray([1.0 - probability_b, probability_b], dtype=np.float64),
        eps,
        1.0,
    )
    middle = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / middle))
        + 0.5 * np.sum(q * np.log(q / middle))
    )


def _derived_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    alive_count: list[int] = []
    disagreement: list[bool] = []
    reliability_imbalance: list[float] = []
    probability_conflict: list[float] = []

    for row in scored.to_dict(orient="records"):
        alive_branches = [
            branch for branch in BRANCHES if int(row[f"{branch}_alive"]) == 1
        ]
        alive_count.append(len(alive_branches))
        predictions = [int(row[f"{branch}_pred"]) for branch in alive_branches]
        disagreement.append(
            len(predictions) >= 2 and len(set(predictions)) > 1
        )
        if len(alive_branches) < 2:
            reliability_imbalance.append(float("nan"))
            probability_conflict.append(float("nan"))
            continue

        reliability = [
            float(row[f"predicted_reliability_{branch}"])
            for branch in alive_branches
        ]
        reliability_imbalance.append(max(reliability) - min(reliability))
        pairwise_conflict = [
            _bernoulli_js(
                float(row[f"{left}_prob"]),
                float(row[f"{right}_prob"]),
            )
            for left_index, left in enumerate(alive_branches)
            for right in alive_branches[left_index + 1 :]
        ]
        probability_conflict.append(float(np.mean(pairwise_conflict)))

    scored["_alive_count"] = alive_count
    scored["_branch_disagreement"] = disagreement
    scored["_reliability_imbalance"] = reliability_imbalance
    scored["_cross_modal_conflict"] = probability_conflict
    return scored


def _fit_validation_threshold(
    values: pd.Series,
    *,
    tail_fraction: float,
    subset_name: str,
    min_calibration_count: int,
) -> tuple[float, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=np.float64))]
    count = int(len(numeric))
    if count < min_calibration_count:
        raise ValueError(
            f"Subset {subset_name} has only {count} eligible validation rows; "
            f"minimum is {min_calibration_count}."
        )
    threshold = float(numeric.quantile(1.0 - tail_fraction))
    if not math.isfinite(threshold):
        raise ValueError(
            f"Validation threshold for subset {subset_name} is not finite"
        )
    return threshold, count


def _join_target_labels(
    target: pd.DataFrame,
    test_csv: pd.DataFrame,
    *,
    test_csv_path: Path,
) -> pd.DataFrame:
    labels = test_csv.copy()
    _require_columns(labels, ["sha256", "label"], str(test_csv_path))
    labels["_sample_id"] = labels["sha256"].astype(str)
    duplicate = labels["_sample_id"].duplicated(keep=False)
    if bool(duplicate.any()):
        examples = sorted(labels.loc[duplicate, "_sample_id"].unique().tolist())[:10]
        raise ValueError(f"{test_csv_path} contains duplicate sha256 values: {examples}")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    valid_labels = (
        np.isfinite(labels["label"].to_numpy(dtype=np.float64))
        & labels["label"].isin([0, 1]).to_numpy()
    )
    if not bool(valid_labels.all()):
        examples = labels.loc[~valid_labels, "_sample_id"].astype(str).tolist()[:10]
        raise ValueError(
            f"{test_csv_path} labels must be finite binary values; examples={examples}"
        )
    labels["label"] = labels["label"].astype(np.int64)

    diagnostic_ids = set(target["_sample_id"])
    label_ids = set(labels["_sample_id"])
    diagnostics_only = sorted(diagnostic_ids - label_ids)
    labels_only = sorted(label_ids - diagnostic_ids)
    if diagnostics_only or labels_only:
        raise ValueError(
            "Natural subsets require one clean diagnostic row for every test sample: "
            f"diagnostics_only={len(diagnostics_only)} examples={diagnostics_only[:10]}; "
            f"labels_only={len(labels_only)} examples={labels_only[:10]}"
        )

    label_by_id = labels.set_index("_sample_id")["label"]
    expected_label = target["_sample_id"].map(label_by_id).astype(np.int64)
    mismatch = target["label"].astype(np.int64) != expected_label
    if bool(mismatch.any()):
        examples = target.loc[mismatch, "_sample_id"].astype(str).tolist()[:10]
        raise ValueError(
            "Diagnostics labels disagree with the target CSV labels; "
            f"examples={examples}"
        )
    return labels


def _target_subset_masks(
    target: pd.DataFrame,
    *,
    reliability_threshold: float,
    conflict_threshold: float,
) -> dict[str, pd.Series]:
    all_alive = target["_alive_count"] == len(BRANCHES)
    masks: dict[str, pd.Series] = {
        "branch_disagreement": target["_branch_disagreement"].astype(bool),
        "reliability_imbalance": (
            (target["_alive_count"] >= 2)
            & (target["_reliability_imbalance"] >= reliability_threshold)
        ),
        "high_cross_modal_conflict": (
            (target["_alive_count"] >= 2)
            & (target["_cross_modal_conflict"] >= conflict_threshold)
        ),
    }
    for target_branch in BRANCHES:
        wrong = (
            target[f"{target_branch}_pred"].astype(np.int64)
            != target["label"].astype(np.int64)
        )
        peers_correct = pd.Series(True, index=target.index)
        for branch in BRANCHES:
            if branch == target_branch:
                continue
            peers_correct &= (
                target[f"{branch}_pred"].astype(np.int64)
                == target["label"].astype(np.int64)
            )
        masks[f"{target_branch}_only_wrong"] = all_alive & wrong & peers_correct
    return masks


def build_subsets(
    diagnostics_path: Path,
    test_csv_path: Path,
    output_dir: Path,
    *,
    calibration_split: str = SUBSET_CALIBRATION_SPLIT,
    target_split: str = SUBSET_TARGET_SPLIT,
    tail_fraction: float = 1.0 / 3.0,
    min_count: int = 10,
    min_calibration_count: int = 10,
    id_column: str | None = None,
) -> list[dict[str, Any]]:
    """Build frozen natural-difficulty subsets without looking at target tails.

    The two continuous thresholds are estimated exclusively on
    ``calibration_split``.  ``target_split`` values are only compared with the
    frozen thresholds.  Exactly-one-wrong subsets use target labels and are
    therefore diagnostic slices, not label-free deployment detectors.
    """

    if not (0.0 < tail_fraction < 0.5):
        raise ValueError("tail_fraction must be in (0, 0.5)")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if min_calibration_count < 1:
        raise ValueError("min_calibration_count must be at least 1")
    if calibration_split != SUBSET_CALIBRATION_SPLIT:
        raise ValueError(
            "The registered natural-subset protocol requires "
            f"calibration_split={SUBSET_CALIBRATION_SPLIT!r}; I1/I2 fitting "
            "identities cannot define their own reliability-imbalance tail."
        )
    if target_split != SUBSET_TARGET_SPLIT:
        raise ValueError(
            "The registered natural-subset protocol requires "
            f"target_split={SUBSET_TARGET_SPLIT!r}."
        )
    if calibration_split == target_split:
        raise ValueError(
            "calibration_split and target_split must be disjoint; target data "
            "cannot define natural-subset thresholds"
        )

    diagnostics = _read_csv(diagnostics_path)
    _require_columns(diagnostics, ["split"], str(diagnostics_path))
    sample_id_column = _resolve_id_column(diagnostics, id_column)
    calibration = _prepare_split(
        diagnostics,
        split=calibration_split,
        id_column=sample_id_column,
        require_label=False,
    )
    target = _prepare_split(
        diagnostics,
        split=target_split,
        id_column=sample_id_column,
        require_label=True,
    )
    calibration = _derived_scores(calibration)
    target = _derived_scores(target)

    reliability_threshold, reliability_calibration_count = (
        _fit_validation_threshold(
            calibration["_reliability_imbalance"],
            tail_fraction=tail_fraction,
            subset_name="reliability_imbalance",
            min_calibration_count=min_calibration_count,
        )
    )
    conflict_threshold, conflict_calibration_count = _fit_validation_threshold(
        calibration["_cross_modal_conflict"],
        tail_fraction=tail_fraction,
        subset_name="high_cross_modal_conflict",
        min_calibration_count=min_calibration_count,
    )
    thresholds = {
        "reliability_imbalance": {
            "evidence": "max_minus_min_predicted_reliability_over_alive_branches",
            "threshold": reliability_threshold,
            "comparison": ">=",
            "source_split": calibration_split,
            "source_split_sha256": _frame_sha256(
                calibration,
                [
                    *(f"{branch}_alive" for branch in BRANCHES),
                    *(f"predicted_reliability_{branch}" for branch in BRANCHES),
                ],
            ),
            "eligible_validation_count": reliability_calibration_count,
            "tail_fraction": tail_fraction,
        },
        "high_cross_modal_conflict": {
            "evidence": "mean_pairwise_bernoulli_jensen_shannon_divergence",
            "threshold": conflict_threshold,
            "comparison": ">=",
            "source_split": calibration_split,
            "source_split_sha256": _frame_sha256(
                calibration,
                [
                    *(f"{branch}_alive" for branch in BRANCHES),
                    *(f"{branch}_prob" for branch in BRANCHES),
                ],
            ),
            "eligible_validation_count": conflict_calibration_count,
            "tail_fraction": tail_fraction,
        },
    }

    test_csv = _read_csv(test_csv_path)
    labels = _join_target_labels(target, test_csv, test_csv_path=test_csv_path)
    masks = _target_subset_masks(
        target,
        reliability_threshold=reliability_threshold,
        conflict_threshold=conflict_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_sha256 = _sha256(diagnostics_path)
    test_csv_sha256 = _sha256(test_csv_path)
    summary: list[dict[str, Any]] = []
    pending_outputs: list[tuple[Path, pd.DataFrame]] = []

    for subset_name, definition in SUBSET_DEFINITIONS.items():
        mask = masks[subset_name].fillna(False).astype(bool)
        subset_ids = set(target.loc[mask, "_sample_id"].astype(str).tolist())
        subset_csv = labels[labels["_sample_id"].isin(subset_ids)].copy()
        subset_csv = subset_csv.drop(columns=["_sample_id"]).reset_index(drop=True)
        if len(subset_csv) < min_count:
            raise ValueError(
                f"Subset {subset_name} only has {len(subset_csv)} examples; "
                f"minimum is {min_count}."
            )
        out_path = output_dir / f"test_{subset_name}.csv"
        pending_outputs.append((out_path, subset_csv))
        label_counts = subset_csv["label"].value_counts().to_dict()
        threshold = thresholds.get(subset_name)
        summary.append(
            {
                "subset": subset_name,
                "csv": str(out_path.as_posix()),
                "target_split": target_split,
                **definition,
                "threshold": (
                    float(threshold["threshold"]) if threshold is not None else None
                ),
                "threshold_comparison": (
                    str(threshold["comparison"]) if threshold is not None else None
                ),
                "calibration_split": (
                    calibration_split if threshold is not None else None
                ),
                "diagnostic_count": int(mask.sum()),
                "csv_count": int(len(subset_csv)),
                "benign_count": int(label_counts.get(0, 0)),
                "malware_count": int(label_counts.get(1, 0)),
                "source_diagnostics_sha256": diagnostics_sha256,
                "source_test_csv_sha256": test_csv_sha256,
            }
        )

    summary_path = output_dir / "subset_summary.csv"
    for (out_path, subset_csv), record in zip(pending_outputs, summary):
        _atomic_write_csv(subset_csv, out_path)
        record["csv_sha256"] = _sha256(out_path)
    _atomic_write_csv(pd.DataFrame.from_records(summary), summary_path)

    manifest = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "protocol_id": SUBSET_PROTOCOL_ID,
        "protocol_dependencies": {
            "diagnostic_source": "one_frozen_main_method_checkpoint",
            "required_splits": [calibration_split, target_split],
            "required_branch_fields": {
                "availability": [f"{branch}_alive" for branch in BRANCHES],
                "prediction": [f"{branch}_pred" for branch in BRANCHES],
                "probability": [f"{branch}_prob" for branch in BRANCHES],
                "i1_reliability": [
                    f"predicted_reliability_{branch}" for branch in BRANCHES
                ],
            },
            "reliability_imbalance_dependency": (
                "depends_on_i1_outputs_and_is_not_independent_evidence_for_i1"
            ),
            "cross_modal_conflict_dependency": (
                "computed_from_branch_probabilities_without_i1_reliability"
            ),
            "exactly_one_wrong_dependency": (
                "uses_target_ground_truth_and_is_diagnostic_only"
            ),
        },
        "protocol_guarantees": {
            "thresholds_fit_on_validation_only": True,
            "target_split_used_for_threshold_selection": False,
            "calibration_split_unseen_by_i1_i2": True,
            "i1_success_is_not_defined_by_i1_reliability": True,
            "label_dependent_subsets_are_diagnostic_only": True,
        },
        "diagnostics": str(diagnostics_path.as_posix()),
        "diagnostics_sha256": diagnostics_sha256,
        "test_csv": str(test_csv_path.as_posix()),
        "test_csv_sha256": test_csv_sha256,
        "calibration_split": calibration_split,
        "target_split": target_split,
        "tail_fraction": tail_fraction,
        "calibration_sample_count": int(len(calibration)),
        "target_sample_count": int(len(test_csv)),
        "thresholds": thresholds,
        "subsets": summary,
    }
    _atomic_write_json(manifest, output_dir / "subset_manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build natural predictive-difficulty subsets with all continuous "
            "thresholds frozen on validation diagnostics."
        )
    )
    parser.add_argument("--diagnostics", default="results/gate_diagnostics.csv")
    parser.add_argument("--test-csv", default="labels/test.csv")
    parser.add_argument("--out-dir", default="labels/natural_subsets")
    parser.add_argument(
        "--calibration-split",
        default=SUBSET_CALIBRATION_SPLIT,
        help=(
            "Validation-only diagnostic split used to freeze tail thresholds. "
            "The default was not used to fit I1/I2."
        ),
    )
    parser.add_argument(
        "--target-split",
        default=SUBSET_TARGET_SPLIT,
        help="Diagnostic split to which frozen rules are applied.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=1.0 / 3.0,
        help="Upper-tail fraction selected using validation diagnostics only.",
    )
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-calibration-count", type=int, default=10)
    parser.add_argument("--id-column", default=None)
    args = parser.parse_args()

    summary = build_subsets(
        diagnostics_path=Path(args.diagnostics),
        test_csv_path=Path(args.test_csv),
        output_dir=Path(args.out_dir),
        calibration_split=str(args.calibration_split),
        target_split=str(args.target_split),
        tail_fraction=float(args.tail_fraction),
        min_count=int(args.min_count),
        min_calibration_count=int(args.min_calibration_count),
        id_column=args.id_column,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
