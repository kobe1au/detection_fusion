from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.train import conformal_selective_metrics  # noqa: E402


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _load_conformal_thresholds(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = yaml.safe_load(f) or {}
    thresholds = summary.get("conformal_thresholds")
    if not thresholds:
        raise ValueError(f"{summary_path} does not contain conformal_thresholds")
    return thresholds


def _prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [_safe_int(row.get("label")) for row in rows]
    preds = [_safe_int(row.get("pred")) for row in rows]
    total = len(labels)
    correct = sum(int(y == p) for y, p in zip(labels, preds))
    out = {"acc": correct / total if total else 0.0}
    f1s = []
    for cls in (0, 1):
        tp = sum(int(y == cls and p == cls) for y, p in zip(labels, preds))
        fp = sum(int(y != cls and p == cls) for y, p in zip(labels, preds))
        fn = sum(int(y == cls and p != cls) for y, p in zip(labels, preds))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    out["macro_f1"] = sum(f1s) / len(f1s)
    return out


def _mean_column(rows: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(row.get(key), default=float("nan")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate conformal coverage by sample year from gate diagnostics."
    )
    parser.add_argument("--rows", nargs="+", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split-prefix",
        default="test",
        help="Only include rows whose split starts with this prefix. Use --all-splits to disable.",
    )
    parser.add_argument("--all-splits", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(args.rows)
    thresholds = _load_conformal_thresholds(args.summary)

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split", "") or "")
        if not args.all_splits and not split.startswith(str(args.split_prefix)):
            continue
        year = _safe_int(row.get("year"))
        if year <= 0:
            continue
        groups[(split, year)].append(row)

    fieldnames = [
        "split",
        "year",
        "n",
        "n_benign",
        "n_malware",
        "acc",
        "macro_f1",
        "mean_raw_conflict",
        "mean_acceptance_score",
        "conformal_acceptance_rate",
        "conformal_selective_risk",
        "conformal_empirical_coverage_benign",
        "conformal_empirical_coverage_malware",
        "conformal_accepted_fn_risk_among_malware",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (split, year), group_rows in sorted(groups.items()):
            labels = [_safe_int(row.get("label")) for row in group_rows]
            metrics = {
                **_prediction_metrics(group_rows),
                **conformal_selective_metrics(group_rows, thresholds),
            }
            writer.writerow(
                {
                    "split": split,
                    "year": year,
                    "n": len(group_rows),
                    "n_benign": sum(int(label == 0) for label in labels),
                    "n_malware": sum(int(label == 1) for label in labels),
                    "acc": metrics.get("acc", float("nan")),
                    "macro_f1": metrics.get("macro_f1", float("nan")),
                    "mean_raw_conflict": _mean_column(group_rows, "raw_conflict"),
                    "mean_acceptance_score": _mean_column(group_rows, "acceptance_score"),
                    "conformal_acceptance_rate": metrics.get("conformal_acceptance_rate", float("nan")),
                    "conformal_selective_risk": metrics.get("conformal_selective_risk", float("nan")),
                    "conformal_empirical_coverage_benign": metrics.get(
                        "conformal_empirical_coverage_benign", float("nan")
                    ),
                    "conformal_empirical_coverage_malware": metrics.get(
                        "conformal_empirical_coverage_malware", float("nan")
                    ),
                    "conformal_accepted_fn_risk_among_malware": metrics.get(
                        "conformal_accepted_fn_risk_among_malware", float("nan")
                    ),
                }
            )


if __name__ == "__main__":
    main()
