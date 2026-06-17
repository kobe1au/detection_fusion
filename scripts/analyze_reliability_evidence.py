from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


BRANCHES = ("api", "graph", "manifest", "joint")

EVIDENCE_FIELDS = (
    "api_integrity",
    "graph_integrity",
    "manifest_integrity",
    "code_integrity",
    "api_graph_anchor_support",
    "manifest_code_support",
    "manifest_to_code_conflict",
    "code_to_manifest_conflict",
    "api_alive",
    "graph_alive",
    "manifest_alive",
    "api_graph_support_applicable",
    "manifest_code_relation_applicable",
    "api_manifest_relation_applicable",
    "graph_manifest_relation_applicable",
)

EVIDENCE_BIN_SPECS = {
    "api_integrity": ("api",),
    "graph_integrity": ("graph",),
    "manifest_integrity": ("manifest",),
    "code_integrity": ("api", "graph", "joint"),
    "api_graph_anchor_support": ("api", "graph", "joint"),
    "manifest_code_support": ("manifest", "joint"),
    "manifest_to_code_conflict": ("manifest", "joint"),
    "code_to_manifest_conflict": ("api", "graph", "joint"),
    "max_manifest_code_conflict": ("api", "graph", "manifest", "joint"),
}

FILTER_COLUMNS = ("experiment", "seed", "split", "diagnostic_file")


def _run_identity(path: Path, results_root: Path | None) -> dict[str, str]:
    if results_root is not None:
        try:
            rel = path.parent.relative_to(results_root)
            parts = rel.parts
        except ValueError:
            parts = path.parent.parts
    else:
        parts = path.parent.parts
    experiment = parts[-2] if len(parts) >= 2 else path.parent.name
    seed = parts[-1] if parts else ""
    return {
        "experiment": experiment,
        "seed": seed,
        "run_dir": str(path.parent),
        "diagnostic_file": path.name,
    }


def _diagnostic_paths(args: argparse.Namespace) -> list[Path]:
    if args.diagnostics:
        return [Path(item) for item in args.diagnostics]
    root = Path(args.results_root)
    patterns = ["gate_diagnostics.csv", "gate_diagnostics_extra_eval.csv"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(paths)


def _read_diagnostics(paths: list[Path], results_root: Path | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        for key, value in _run_identity(path, results_root).items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _parse_filter(values: list[str] | None) -> set[str]:
    return {str(value) for value in values or [] if str(value).strip()}


def _apply_filters(
    frame: pd.DataFrame,
    *,
    experiments: set[str],
    seeds: set[str],
    splits: set[str],
    diagnostic_files: set[str],
) -> pd.DataFrame:
    filtered = frame
    filters = {
        "experiment": experiments,
        "seed": seeds,
        "split": splits,
        "diagnostic_file": diagnostic_files,
    }
    for column, accepted in filters.items():
        if accepted and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str).isin(accepted)]
    return filtered.copy()


def _safe_path_part(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def _ece(scores: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    if scores.size == 0:
        return 0.0
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        if high >= 1.0:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(
            float(scores[mask].mean()) - float(correctness[mask].mean())
        )
    return float(total)


def _finite_binary(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.where(array >= 0.5, 1.0, 0.0)


def _safe_auc(correctness: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(correctness, scores))
        if _auc_defined(correctness)
        else float("nan")
    )


def _auc_defined(correctness: np.ndarray) -> bool:
    return len(set(correctness.tolist())) > 1


def _safe_ap(correctness: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(correctness, scores))
        if _ap_defined(correctness)
        else float("nan")
    )


def _ap_defined(correctness: np.ndarray) -> bool:
    return len(set(correctness.tolist())) > 1


def reliability_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["experiment", "seed", "diagnostic_file", "split"]
    available_groups = [column for column in group_columns if column in frame.columns]
    for group, group_frame in frame.groupby(available_groups, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(available_groups, group_values))
        for branch in BRANCHES:
            reliability_key = f"predicted_reliability_{branch}"
            correct_key = f"{branch}_correct"
            if reliability_key not in group_frame.columns or correct_key not in group_frame.columns:
                continue
            data = group_frame[[reliability_key, correct_key]].dropna()
            if data.empty:
                continue
            scores = data[reliability_key].astype(float).clip(0.0, 1.0).to_numpy()
            correctness = _finite_binary(data[correct_key])
            auc_defined = _auc_defined(correctness)
            ap_defined = _ap_defined(correctness)
            records.append(
                {
                    **base,
                    "branch": branch,
                    "count": int(scores.size),
                    "reliability_mean": float(scores.mean()),
                    "branch_accuracy": float(correctness.mean()),
                    "reliability_accuracy_gap": float(scores.mean() - correctness.mean()),
                    "brier": float(np.mean((scores - correctness) ** 2)),
                    "ece_10": _ece(scores, correctness, bins=10),
                    "auc_defined": int(auc_defined),
                    "auc": _safe_auc(correctness, scores),
                    "ap_defined": int(ap_defined),
                    "ap": _safe_ap(correctness, scores),
                }
            )
    return pd.DataFrame.from_records(records)


def evidence_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    evidence_columns = [column for column in EVIDENCE_FIELDS if column in frame.columns]
    if not evidence_columns:
        return pd.DataFrame.from_records(records)
    group_columns = ["experiment", "seed", "diagnostic_file", "split"]
    available_groups = [column for column in group_columns if column in frame.columns]
    numeric = _to_numeric(frame, evidence_columns)
    for group, group_frame in numeric.groupby(available_groups, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(available_groups, group_values))
        alive_pattern = ""
        if {"api_alive", "graph_alive", "manifest_alive"}.issubset(group_frame.columns):
            alive_means = [
                float(group_frame[column].mean())
                for column in ("api_alive", "graph_alive", "manifest_alive")
            ]
            alive_pattern = "".join("1" if value >= 0.5 else "0" for value in alive_means)
        for column in evidence_columns:
            values = group_frame[column].astype(float).dropna()
            if values.empty:
                continue
            records.append(
                {
                    **base,
                    "alive_pattern_majority": alive_pattern,
                    "evidence": column,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "p10": float(values.quantile(0.10)),
                    "p50": float(values.quantile(0.50)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame.from_records(records)


def _with_derived_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    conflict_columns = [
        column
        for column in ("manifest_to_code_conflict", "code_to_manifest_conflict")
        if column in out.columns
    ]
    if conflict_columns:
        out["max_manifest_code_conflict"] = out[conflict_columns].max(axis=1)
    return out


def _tercile_labels(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return labels
    if valid.nunique(dropna=True) <= 1:
        labels.loc[valid.index] = "all"
        return labels
    q = min(3, len(valid))
    label_values = ["low", "mid", "high"] if q == 3 else ["low", "high"]
    try:
        ranked = valid.rank(method="first")
        labels.loc[valid.index] = pd.qcut(ranked, q=q, labels=label_values).astype(str)
    except ValueError:
        labels.loc[valid.index] = "all"
    return labels


def _assign_evidence_bins(
    frame: pd.DataFrame,
    evidence: str,
    group_columns: list[str],
    bin_scope: str,
) -> pd.Series:
    if bin_scope not in {"global", "group"}:
        raise ValueError("bin_scope must be 'global' or 'group'")
    if bin_scope == "global" or not group_columns:
        return _tercile_labels(frame[evidence])
    labels = pd.Series(index=frame.index, dtype="object")
    for _, group_frame in frame.groupby(group_columns, dropna=False):
        labels.loc[group_frame.index] = _tercile_labels(group_frame[evidence])
    return labels


def evidence_bin_effects_table(frame: pd.DataFrame, bin_scope: str = "global") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    frame = _with_derived_evidence(frame)
    group_columns = ["experiment", "seed", "diagnostic_file", "split"]
    available_groups = [column for column in group_columns if column in frame.columns]
    for evidence, branches in EVIDENCE_BIN_SPECS.items():
        if evidence not in frame.columns:
            continue
        working = frame.copy()
        working["evidence_bin"] = _assign_evidence_bins(
            working,
            evidence,
            available_groups,
            bin_scope,
        )
        working = working.dropna(subset=["evidence_bin"])
        if working.empty:
            continue
        for group, group_frame in working.groupby([*available_groups, "evidence_bin"], dropna=False):
            group_values = group if isinstance(group, tuple) else (group,)
            base = dict(zip([*available_groups, "evidence_bin"], group_values))
            evidence_values = pd.to_numeric(group_frame[evidence], errors="coerce").dropna()
            if evidence_values.empty:
                continue
            for branch in branches:
                correct_key = f"{branch}_correct"
                reliability_key = f"predicted_reliability_{branch}"
                if correct_key not in group_frame.columns and reliability_key not in group_frame.columns:
                    continue
                branch_record: dict[str, Any] = {
                    **base,
                    "evidence": evidence,
                    "branch": branch,
                    "count": int(len(group_frame)),
                    "evidence_mean": float(evidence_values.mean()),
                    "evidence_min": float(evidence_values.min()),
                    "evidence_max": float(evidence_values.max()),
                }
                if correct_key in group_frame.columns:
                    correctness = pd.to_numeric(group_frame[correct_key], errors="coerce").dropna()
                    if not correctness.empty:
                        branch_record["branch_accuracy"] = float((correctness >= 0.5).mean())
                if reliability_key in group_frame.columns:
                    reliability = pd.to_numeric(group_frame[reliability_key], errors="coerce").dropna()
                    if not reliability.empty:
                        branch_record["predicted_reliability_mean"] = float(
                            reliability.clip(0.0, 1.0).mean()
                        )
                if (
                    "branch_accuracy" in branch_record
                    and "predicted_reliability_mean" in branch_record
                ):
                    branch_record["reliability_accuracy_gap"] = float(
                        branch_record["predicted_reliability_mean"]
                        - branch_record["branch_accuracy"]
                    )
                records.append(branch_record)
    return pd.DataFrame.from_records(records)


def _diagram_points(scores: np.ndarray, correctness: np.ndarray, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for idx, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if high >= 1.0:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": idx,
                "bin_low": float(low),
                "bin_high": float(high),
                "count": int(mask.sum()),
                "mean_reliability": float(scores[mask].mean()),
                "empirical_accuracy": float(correctness[mask].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_reliability_diagrams_for_frame(
    frame: pd.DataFrame,
    out_dir: Path,
    title_prefix: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for branch in BRANCHES:
        reliability_key = f"predicted_reliability_{branch}"
        correct_key = f"{branch}_correct"
        if reliability_key not in frame.columns or correct_key not in frame.columns:
            continue
        data = frame[[reliability_key, correct_key]].dropna()
        if data.empty:
            continue
        scores = data[reliability_key].astype(float).clip(0.0, 1.0).to_numpy()
        correctness = _finite_binary(data[correct_key])
        points = _diagram_points(scores, correctness, bins=10)
        if points.empty:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        points.to_csv(out_dir / f"reliability_diagram_{branch}.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0)
        ax.plot(
            points["mean_reliability"],
            points["empirical_accuracy"],
            marker="o",
            linewidth=1.5,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted branch reliability")
        ax.set_ylabel("Empirical branch accuracy")
        title = f"{branch.capitalize()} reliability"
        if title_prefix:
            title = f"{title_prefix} {title}"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"reliability_diagram_{branch}.png", dpi=200)
        plt.close(fig)


def write_reliability_diagrams(frame: pd.DataFrame, out_dir: Path) -> None:
    group_columns = [
        column
        for column in ("experiment", "seed", "split", "diagnostic_file")
        if column in frame.columns
    ]
    if not group_columns:
        _write_reliability_diagrams_for_frame(frame, out_dir)
        return
    for group, group_frame in frame.groupby(group_columns, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        group_map = dict(zip(group_columns, group_values))
        group_dir = out_dir
        for column in group_columns:
            if column == "diagnostic_file":
                group_dir = group_dir / _safe_path_part(Path(str(group_map[column])).stem)
            else:
                group_dir = group_dir / _safe_path_part(group_map[column])
        title_parts = [
            str(group_map[column])
            for column in ("experiment", "seed", "split")
            if column in group_map
        ]
        _write_reliability_diagrams_for_frame(
            group_frame,
            group_dir,
            title_prefix=" ".join(title_parts),
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze observable evidence and branch reliability diagnostics."
    )
    parser.add_argument("--results-root", default="results/tri_modal_robust")
    parser.add_argument("--diagnostics", nargs="*", default=None)
    parser.add_argument("--out-dir", default="tables")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--experiment", nargs="*", default=None)
    parser.add_argument("--seed", nargs="*", default=None)
    parser.add_argument("--split", nargs="*", default=None)
    parser.add_argument("--diagnostic-file", nargs="*", default=None)
    parser.add_argument(
        "--bin-scope",
        choices=("global", "group"),
        default="global",
        help=(
            "Use one evidence-binning threshold globally after filtering, or "
            "separate thresholds inside each experiment/seed/split group."
        ),
    )
    parser.add_argument("--fail-if-empty", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root) if args.results_root else None
    paths = _diagnostic_paths(args)
    frame = _read_diagnostics(paths, results_root)
    if frame.empty:
        if args.fail_if_empty:
            raise RuntimeError("No gate diagnostics were found.")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(Path(args.out_dir) / "i1_reliability_calibration.csv", index=False)
        pd.DataFrame().to_csv(Path(args.out_dir) / "i1_evidence_groups.csv", index=False)
        pd.DataFrame().to_csv(Path(args.out_dir) / "i1_evidence_bin_effects.csv", index=False)
        return

    numeric_columns = [
        *EVIDENCE_FIELDS,
        *[f"predicted_reliability_{branch}" for branch in BRANCHES],
        *[f"{branch}_correct" for branch in BRANCHES],
    ]
    frame = _to_numeric(frame, [column for column in numeric_columns if column in frame.columns])
    frame = _apply_filters(
        frame,
        experiments=_parse_filter(args.experiment),
        seeds=_parse_filter(args.seed),
        splits=_parse_filter(args.split),
        diagnostic_files=_parse_filter(args.diagnostic_file),
    )
    if frame.empty and args.fail_if_empty:
        raise RuntimeError("No diagnostics remained after filtering.")
    _write_csv(reliability_table(frame), Path(args.out_dir) / "i1_reliability_calibration.csv")
    _write_csv(evidence_group_table(frame), Path(args.out_dir) / "i1_evidence_groups.csv")
    _write_csv(
        evidence_bin_effects_table(frame, bin_scope=args.bin_scope),
        Path(args.out_dir) / "i1_evidence_bin_effects.csv",
    )
    write_reliability_diagrams(frame, Path(args.figures_dir))


if __name__ == "__main__":
    main()
