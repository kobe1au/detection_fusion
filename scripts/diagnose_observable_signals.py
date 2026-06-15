from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.dataset import RobustTriModalDataset
from fusion.quality import OBSERVABLE_SIGNAL_FIELDS


DEFAULT_SCENARIOS = (
    "clean",
    "api_degraded",
    "graph_degraded",
    "manifest_degraded",
    "api_graph_degraded",
    "all_degraded",
    "api_missing",
    "graph_missing",
    "manifest_missing",
)

TREND_EXPECTATIONS = {
    ("api_event_dropout", "api_integrity"): "nonincreasing",
    ("api_category_dropout", "api_integrity"): "nonincreasing",
    ("graph_sparsify", "graph_integrity"): "nonincreasing",
    ("manifest_permission_mask", "code_to_manifest_conflict"): "nondecreasing",
}

DISTRIBUTION_REQUIRED_COLUMNS = {
    "split",
    "scenario",
    "strength",
    "signal",
    "count",
    "mean",
    "variance",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
}

TREND_REQUIRED_COLUMNS = {
    "split",
    "scenario",
    "signal",
    "num_strengths",
    "mean_at_min_strength",
    "mean_at_max_strength",
    "endpoint_delta",
    "varies_with_strength",
    "strength_mean_correlation",
    "expected_direction",
    "monotonic_expected",
    "monotonic_nonincreasing",
    "monotonic_nondecreasing",
}


def _signal_rows(dataset: RobustTriModalDataset, split: str, scenario: str, strength: float) -> list[dict]:
    rows: list[dict] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if getattr(item, "is_dummy", False):
            continue
        row = {
            "split": split,
            "scenario": scenario,
            "strength": float(strength),
            "sid": str(item.sid),
            "label": int(item.y.item()),
        }
        for key in OBSERVABLE_SIGNAL_FIELDS:
            row[key] = float(getattr(item, key).view(-1)[0].item())
        rows.append(row)
    return rows


def _distribution_table(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    group_keys = ["split", "scenario", "strength"]
    for group, frame in rows.groupby(group_keys, dropna=False):
        for signal in OBSERVABLE_SIGNAL_FIELDS:
            values = frame[signal].astype(float)
            records.append(
                {
                    **dict(zip(group_keys, group)),
                    "signal": signal,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "variance": float(values.var(ddof=0)),
                    "p05": float(values.quantile(0.05)),
                    "p25": float(values.quantile(0.25)),
                    "p50": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame.from_records(records)


def _label_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for group, frame in rows.groupby(["split", "scenario", "strength"], dropna=False):
        labels = frame["label"].astype(int).to_numpy()
        for signal in OBSERVABLE_SIGNAL_FIELDS:
            values = frame[signal].astype(float).to_numpy()
            corr = float(np.corrcoef(values, labels)[0, 1]) if values.size > 1 and np.std(values) > 0 and np.std(labels) > 0 else 0.0
            auc = float(roc_auc_score(labels, values)) if len(set(labels.tolist())) > 1 else 0.0
            records.append(
                {
                    "split": group[0],
                    "scenario": group[1],
                    "strength": group[2],
                    "signal": signal,
                    "label_correlation": corr,
                    "single_signal_auc": auc,
                }
            )
    return pd.DataFrame.from_records(records)


def _trend_table(distribution: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for group, frame in distribution.groupby(["split", "scenario", "signal"], dropna=False):
        ordered = frame.sort_values("strength")
        strengths = ordered["strength"].astype(float).to_numpy()
        means = ordered["mean"].astype(float).to_numpy()
        corr = (
            float(np.corrcoef(strengths, means)[0, 1])
            if strengths.size > 1 and np.std(strengths) > 0 and np.std(means) > 0
            else 0.0
        )
        monotonic_nonincreasing = bool(np.all(np.diff(means) <= 1e-8))
        monotonic_nondecreasing = bool(np.all(np.diff(means) >= -1e-8))
        expected_direction = TREND_EXPECTATIONS.get((str(group[1]), str(group[2])), "")
        monotonic_expected = (
            monotonic_nondecreasing
            if expected_direction == "nondecreasing"
            else monotonic_nonincreasing
            if expected_direction == "nonincreasing"
            else True
        )
        records.append(
            {
                "split": group[0],
                "scenario": group[1],
                "signal": group[2],
                "num_strengths": int(strengths.size),
                "mean_at_min_strength": float(means[0]),
                "mean_at_max_strength": float(means[-1]),
                "endpoint_delta": float(means[-1] - means[0]),
                "varies_with_strength": bool(np.ptp(means) > 1e-8),
                "strength_mean_correlation": corr,
                "expected_direction": expected_direction,
                "monotonic_expected": monotonic_expected,
                "monotonic_nonincreasing": monotonic_nonincreasing,
                "monotonic_nondecreasing": monotonic_nondecreasing,
            }
        )
    return pd.DataFrame.from_records(records)


def _output_checks(distribution: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("distribution_nonempty", not distribution.empty, f"rows={len(distribution)}")
    missing_distribution = sorted(DISTRIBUTION_REQUIRED_COLUMNS.difference(distribution.columns))
    add(
        "distribution_required_columns",
        not missing_distribution,
        f"missing={missing_distribution}",
    )
    missing_trends = sorted(TREND_REQUIRED_COLUMNS.difference(trends.columns))
    add("trends_required_columns", not missing_trends, f"missing={missing_trends}")

    if not missing_distribution and not distribution.empty:
        numeric_columns = ["count", "mean", "variance", "p05", "p25", "p50", "p75", "p95"]
        finite = np.isfinite(distribution[numeric_columns].to_numpy(dtype=float)).all()
        add("distribution_numeric_finite", bool(finite), f"columns={numeric_columns}")
        signal_values = distribution[["mean", "p05", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float)
        in_range = bool(((signal_values >= -1e-8) & (signal_values <= 1.0 + 1e-8)).all())
        add("distribution_signal_range", in_range, "expected all signal summaries in [0, 1]")
        quantiles = distribution[["p05", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float)
        quantiles_ordered = bool((np.diff(quantiles, axis=1) >= -1e-8).all())
        add("distribution_quantiles_ordered", quantiles_ordered, "expected p05<=p25<=p50<=p75<=p95")
        add(
            "distribution_positive_counts",
            bool((distribution["count"].astype(int) > 0).all()),
            f"min_count={int(distribution['count'].min())}",
        )

    if not missing_trends and not trends.empty:
        duplicate_count = int(trends.duplicated(["split", "scenario", "signal"]).sum())
        add("trends_unique_groups", duplicate_count == 0, f"duplicates={duplicate_count}")
        expected = trends[trends["expected_direction"].astype(str) != ""]
        failures = expected[~expected["monotonic_expected"].astype(bool)]
        add(
            "expected_trends_monotonic",
            failures.empty,
            f"checked={len(expected)}, failures={len(failures)}",
        )

    return pd.DataFrame.from_records(checks)


def _paired_table(rows: pd.DataFrame, pair_csv: Path) -> pd.DataFrame:
    pairs = pd.read_csv(pair_csv)
    required = {"clean_id", "obfuscated_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Paired CSV must contain {sorted(required)}")
    clean = rows[rows["scenario"] == "clean"].drop_duplicates("sid").set_index("sid")
    records: list[dict] = []
    for pair in pairs.to_dict("records"):
        clean_id = str(pair["clean_id"]).lower()
        obfuscated_id = str(pair["obfuscated_id"]).lower()
        if clean_id not in clean.index or obfuscated_id not in clean.index:
            continue
        row = {"clean_id": clean_id, "obfuscated_id": obfuscated_id}
        for signal in OBSERVABLE_SIGNAL_FIELDS:
            row[f"delta_{signal}"] = float(clean.loc[obfuscated_id, signal] - clean.loc[clean_id, signal])
        records.append(row)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose observable reliability signals.")
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--strict-observable-schema", action="store_true")
    parser.add_argument("--paired-csv", default="")
    parser.add_argument("--fail-on-check-error", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for scenario in args.scenarios:
        strengths = [0.0] if scenario == "clean" else ([1.0] if scenario.endswith("_missing") else args.strengths)
        for strength in strengths:
            dataset = RobustTriModalDataset(
                args.pt_dir,
                args.csv,
                is_train=False,
                robust_aug=False,
                eval_perturb_type=None if scenario == "clean" else scenario,
                eval_perturb_strength=float(strength),
                strict_observable_schema=bool(args.strict_observable_schema),
            )
            rows.extend(_signal_rows(dataset, args.split, scenario, float(strength)))

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError("No valid PT samples were available for observable-signal diagnostics.")
    distribution = _distribution_table(frame)
    labels = _label_diagnostics(frame)
    trends = _trend_table(distribution)
    checks = _output_checks(distribution, trends)
    frame.to_csv(out_dir / "observable_signal_rows.csv", index=False)
    distribution.to_csv(out_dir / "observable_signal_distribution.csv", index=False)
    labels.to_csv(out_dir / "observable_signal_label_diagnostics.csv", index=False)
    trends.to_csv(out_dir / "observable_signal_trends.csv", index=False)
    checks.to_csv(out_dir / "observable_signal_checks.csv", index=False)
    summary = {
        "num_rows": int(len(frame)),
        "signals": list(OBSERVABLE_SIGNAL_FIELDS),
        "scenarios": list(args.scenarios),
        "varying_trends": int(trends["varies_with_strength"].astype(bool).sum()),
        "expected_trend_failures": int(
            (
                (trends["expected_direction"].astype(str) != "")
                & ~trends["monotonic_expected"].astype(bool)
            ).sum()
        ),
        "checks_passed": bool(checks["passed"].astype(bool).all()),
    }
    if args.paired_csv:
        paired = _paired_table(frame, Path(args.paired_csv))
        paired.to_csv(out_dir / "observable_signal_paired_shifts.csv", index=False)
        summary["num_pairs"] = int(len(paired))
    with open(out_dir / "observable_signal_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    if args.fail_on_check_error and not summary["checks_passed"]:
        raise RuntimeError(f"Observable diagnostic output checks failed; inspect {out_dir}")


if __name__ == "__main__":
    main()
