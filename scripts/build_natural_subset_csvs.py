from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUBSET_SPECS: dict[str, tuple[str, str]] = {
    "api_low_effective_integrity": ("effective_api_integrity", "low"),
    "api_graph_low_support": ("api_graph_anchor_support", "low"),
    # Predictive conflict is measured before I1 trust discounting, so this cut
    # does not select samples merely because Ours already suppressed a view.
    "predictive_high_conflict": ("predictive_conflict", "high"),
    "low_acceptance": ("acceptance_score", "low"),
}

SUBSET_SCHEMA_VERSION = 3


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


def _subset_mask(values: pd.Series, direction: str, quantile: float) -> tuple[pd.Series, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        raise ValueError("Cannot build subset from an all-empty evidence column")
    if direction == "low":
        threshold = float(valid.quantile(quantile))
        return numeric <= threshold, threshold
    if direction == "high":
        threshold = float(valid.quantile(1.0 - quantile))
        return numeric >= threshold, threshold
    raise ValueError(f"Unknown subset direction: {direction}")


def build_subsets(
    diagnostics_path: Path,
    test_csv_path: Path,
    output_dir: Path,
    split: str,
    quantile: float,
    min_count: int,
    id_column: str | None = None,
) -> list[dict[str, Any]]:
    if not (0.0 < quantile < 0.5):
        raise ValueError("--quantile must be in (0, 0.5)")

    diagnostics = _read_csv(diagnostics_path)
    test_csv = _read_csv(test_csv_path)
    if "sha256" not in test_csv.columns:
        raise ValueError(f"{test_csv_path} must contain a sha256 column")
    if "label" not in test_csv.columns:
        raise ValueError(f"{test_csv_path} must contain a label column")
    if "split" not in diagnostics.columns:
        raise ValueError(f"{diagnostics_path} must contain a split column")

    sample_id_column = _resolve_id_column(diagnostics, id_column)
    split_frame = diagnostics[diagnostics["split"].astype(str) == split].copy()
    if split_frame.empty:
        raise ValueError(f"No diagnostics rows found for split={split!r}")
    split_frame[sample_id_column] = split_frame[sample_id_column].astype(str)
    duplicate_diagnostics = split_frame[sample_id_column].duplicated(keep=False)
    if bool(duplicate_diagnostics.any()):
        examples = sorted(
            split_frame.loc[duplicate_diagnostics, sample_id_column].unique().tolist()
        )[:10]
        raise ValueError(
            f"Diagnostics split={split!r} contains duplicate sample ids: {examples}"
        )

    test_csv = test_csv.copy()
    test_csv["sha256"] = test_csv["sha256"].astype(str)
    duplicate_labels = test_csv["sha256"].duplicated(keep=False)
    if bool(duplicate_labels.any()):
        examples = sorted(
            test_csv.loc[duplicate_labels, "sha256"].unique().tolist()
        )[:10]
        raise ValueError(f"{test_csv_path} contains duplicate sha256 values: {examples}")

    diagnostic_ids = set(split_frame[sample_id_column])
    label_ids = set(test_csv["sha256"])
    diagnostics_only = sorted(diagnostic_ids - label_ids)
    labels_only = sorted(label_ids - diagnostic_ids)
    if diagnostics_only or labels_only:
        raise ValueError(
            "Natural subsets require one clean diagnostic row for every test sample: "
            f"diagnostics_only={len(diagnostics_only)} examples={diagnostics_only[:10]}; "
            f"labels_only={len(labels_only)} examples={labels_only[:10]}"
        )

    for evidence, _direction in SUBSET_SPECS.values():
        if evidence not in split_frame.columns:
            raise ValueError(f"Diagnostics file is missing evidence column: {evidence}")
        numeric = pd.to_numeric(split_frame[evidence], errors="coerce")
        invalid = ~numeric.map(lambda value: pd.notna(value) and float("-inf") < value < float("inf"))
        if bool(invalid.any()):
            examples = split_frame.loc[invalid, sample_id_column].astype(str).tolist()[:10]
            raise ValueError(
                f"Diagnostics evidence column {evidence!r} contains non-finite values "
                f"for split={split!r}: examples={examples}"
            )
        split_frame[evidence] = numeric.astype(float)

    label_sha = test_csv["sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    pending_outputs: list[tuple[Path, pd.DataFrame]] = []
    diagnostics_sha256 = _sha256(diagnostics_path)
    test_csv_sha256 = _sha256(test_csv_path)

    for subset_name, (evidence, direction) in SUBSET_SPECS.items():
        mask, threshold = _subset_mask(split_frame[evidence], direction, quantile)
        subset_ids = set(split_frame.loc[mask.fillna(False), sample_id_column].astype(str))
        subset_csv = test_csv[label_sha.isin(subset_ids)].copy()
        if len(subset_csv) < min_count:
            raise ValueError(
                f"Subset {subset_name} only has {len(subset_csv)} examples after "
                f"joining with {test_csv_path}; minimum is {min_count}."
            )
        out_path = output_dir / f"test_{subset_name}.csv"
        pending_outputs.append((out_path, subset_csv))

        label_counts = subset_csv["label"].value_counts().to_dict() if "label" in subset_csv else {}
        evidence_values = pd.to_numeric(
            split_frame.loc[mask.fillna(False), evidence],
            errors="coerce",
        )
        summary.append(
            {
                "subset": subset_name,
                "csv": str(out_path.as_posix()),
                "split": split,
                "evidence": evidence,
                "direction": direction,
                "quantile": quantile,
                "threshold": threshold,
                "diagnostic_count": int(mask.fillna(False).sum()),
                "csv_count": int(len(subset_csv)),
                "evidence_mean": float(evidence_values.mean()),
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
        "diagnostics": str(diagnostics_path.as_posix()),
        "diagnostics_sha256": diagnostics_sha256,
        "test_csv": str(test_csv_path.as_posix()),
        "test_csv_sha256": test_csv_sha256,
        "split": split,
        "quantile": quantile,
        "sample_count": int(len(test_csv)),
        "subsets": summary,
    }
    _atomic_write_json(manifest, output_dir / "subset_manifest.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build natural evidence-quality subset CSVs from Ours gate diagnostics."
    )
    parser.add_argument("--diagnostics", default="results/gate_diagnostics.csv")
    parser.add_argument("--test-csv", default="labels/test.csv")
    parser.add_argument("--out-dir", default="labels/natural_subsets")
    parser.add_argument("--split", default="test_clean")
    parser.add_argument("--quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--id-column", default=None)
    args = parser.parse_args()

    summary = build_subsets(
        diagnostics_path=Path(args.diagnostics),
        test_csv_path=Path(args.test_csv),
        output_dir=Path(args.out_dir),
        split=str(args.split),
        quantile=float(args.quantile),
        min_count=int(args.min_count),
        id_column=args.id_column,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
