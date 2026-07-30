from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCHES = ("api", "graph", "manifest", "joint")
METRIC_SUMMARY_SCHEMA_VERSION = 14
FILTER_COLUMNS = ("experiment", "seed", "split", "diagnostic_file")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str, *, summary_path: Path) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(
            f"{summary_path} has an invalid {field}; diagnostics require a "
            "complete run-identity binding"
        )
    return text


def _bound_diagnostic_metadata(path: Path) -> tuple[dict[str, str], list[str]]:
    """Verify that one diagnostic CSV is hash-bound to a schema-v14 run."""

    path = path.resolve()
    summary_path = path.parent / "summary.yaml"
    if not summary_path.is_file():
        raise ValueError(f"{path} has no sibling summary.yaml")
    try:
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read {summary_path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path} must contain a mapping")
    if summary.get("metric_schema_version") != METRIC_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"{summary_path} uses metric_schema_version="
            f"{summary.get('metric_schema_version')!r}; expected "
            f"{METRIC_SUMMARY_SCHEMA_VERSION}"
        )

    identity = summary.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{summary_path} is missing run_identity")
    required = (
        "experiment_name",
        "method_name",
        "seed",
        "method_protocol_id",
        "method_protocol_sha256",
        "method_implementation_sha256",
    )
    missing = [
        key
        for key in required
        if key not in identity
        or (key != "method_protocol_id" and not str(identity[key]).strip())
    ]
    if missing:
        raise ValueError(
            f"{summary_path} run_identity is missing required fields: {missing}"
        )
    protocol_sha = _require_sha256(
        identity["method_protocol_sha256"],
        "run_identity.method_protocol_sha256",
        summary_path=summary_path,
    )
    implementation_sha = _require_sha256(
        identity["method_implementation_sha256"],
        "run_identity.method_implementation_sha256",
        summary_path=summary_path,
    )

    artifact_key = path.stem
    artifacts = summary.get("diagnostic_artifacts")
    artifact = artifacts.get(artifact_key) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError(
            f"{summary_path} does not bind diagnostic_artifacts.{artifact_key}"
        )
    artifact_path = Path(str(artifact.get("path") or ""))
    if not artifact_path.is_absolute() or artifact_path.resolve() != path:
        raise ValueError(
            f"{summary_path} does not bind the requested diagnostic path {path}"
        )
    expected_sha = _require_sha256(
        artifact.get("sha256"),
        f"diagnostic_artifacts.{artifact_key}.sha256",
        summary_path=summary_path,
    )
    if _file_sha256(path) != expected_sha:
        raise ValueError(f"{path} changed after its source summary was written")
    for key, expected in (
        ("method_protocol_id", str(identity.get("method_protocol_id") or "")),
        ("method_protocol_sha256", protocol_sha),
        ("method_implementation_sha256", implementation_sha),
    ):
        if str(artifact.get(key) or "") != expected:
            raise ValueError(
                f"{summary_path} diagnostic artifact disagrees with "
                f"run_identity.{key}"
            )
    declared_splits = artifact.get("splits")
    if not isinstance(declared_splits, list):
        raise ValueError(
            f"{summary_path} diagnostic artifact has no split inventory"
        )
    return (
        {
            "experiment": str(identity["experiment_name"]),
            "method": str(identity["method_name"]),
            "seed": str(identity["seed"]),
            "method_protocol_id": str(identity.get("method_protocol_id") or ""),
            "method_protocol_sha256": protocol_sha,
            "method_implementation_sha256": implementation_sha,
            "run_dir": str(path.parent),
            "summary_path": str(summary_path),
            "diagnostic_file": path.name,
        },
        sorted(str(value) for value in declared_splits),
    )


def _diagnostic_paths(args: argparse.Namespace) -> list[Path]:
    if args.diagnostics:
        return [Path(item) for item in args.diagnostics]
    root = Path(args.results_root)
    return sorted(
        [
            *root.rglob("gate_diagnostics.csv"),
            *root.rglob("gate_diagnostics_extra_eval.csv"),
        ]
    )


def _validate_validation_role_identities(
    frame: pd.DataFrame,
    *,
    path: Path,
) -> None:
    """Fail closed on duplicated or overlapping two-role diagnostics."""

    if "split" not in frame.columns:
        return
    role_names = (
        "val_model_selection",
        "val_decision_calibration",
    )
    present = {
        role
        for role in role_names
        if bool((frame["split"].astype(str) == role).any())
    }
    if not present:
        return
    if "sid" not in frame.columns:
        raise ValueError(
            f"{path} contains validation-role rows without sample identities"
        )
    role_ids: dict[str, set[str]] = {}
    for role in role_names:
        rows = frame[frame["split"].astype(str) == role]
        if rows.empty:
            role_ids[role] = set()
            continue
        normalized = rows["sid"].astype(str).str.strip().str.lower()
        if bool((normalized == "").any()):
            raise ValueError(
                f"{path} validation role {role!r} contains an empty sample id"
            )
        duplicated = normalized[normalized.duplicated()].unique().tolist()
        if duplicated:
            raise ValueError(
                f"{path} validation role {role!r} contains duplicate sample "
                f"ids: {duplicated[:10]}"
            )
        role_ids[role] = set(normalized.tolist())
    overlap = sorted(
        role_ids["val_model_selection"]
        & role_ids["val_decision_calibration"]
    )
    if overlap:
        raise ValueError(
            f"{path} validation roles overlap: {overlap[:10]}"
        )


def _read_diagnostics(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        identity, declared_splits = _bound_diagnostic_metadata(path)
        actual_splits = sorted(
            str(value)
            for value in frame.get("split", pd.Series(dtype=object)).dropna().unique()
            if str(value)
        )
        if actual_splits != declared_splits:
            raise ValueError(
                f"{path} split inventory disagrees with its source summary"
            )
        _validate_validation_role_identities(frame, path=path)
        for key, value in identity.items():
            frame[key] = value
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


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
    filters = {
        "experiment": experiments,
        "seed": seeds,
        "split": splits,
        "diagnostic_file": diagnostic_files,
    }
    out = frame
    for column, accepted in filters.items():
        if accepted and column in out.columns:
            out = out[out[column].astype(str).isin(accepted)]
    return out.copy()


def _safe_path_part(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in text
    )


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if (
        left.size < 2
        or np.std(left) <= 0.0
        or np.std(right) <= 0.0
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _branch_rows(frame: pd.DataFrame, branch: str) -> pd.DataFrame:
    required = (
        "label",
        f"{branch}_prob",
        f"{branch}_correct",
        f"predicted_competence_{branch}",
    )
    if any(key not in frame.columns for key in required):
        return pd.DataFrame()
    columns = [*required]
    alive_key = f"{branch}_alive"
    if alive_key in frame.columns:
        columns.append(alive_key)
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna(
        subset=list(required)
    )
    if alive_key in data.columns:
        data = data[data[alive_key] >= 0.5]
    if data.empty:
        return data
    label = data["label"].round().astype(int).to_numpy()
    probability = data[f"{branch}_prob"].clip(0.0, 1.0).to_numpy(dtype=float)
    data = data.copy()
    data["tcp_target"] = np.where(label == 1, probability, 1.0 - probability)
    data["competence"] = data[f"predicted_competence_{branch}"].clip(0.0, 1.0)
    data["correctness"] = (
        data[f"{branch}_correct"].to_numpy(dtype=float) >= 0.5
    ).astype(float)
    return data


def competence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Primary I1 audit against the continuous TCP target."""

    records: list[dict[str, Any]] = []
    group_columns = [
        key
        for key in ("experiment", "seed", "diagnostic_file", "split")
        if key in frame.columns
    ]
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    for group, group_frame in grouped:
        values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(group_columns, values))
        for branch in BRANCHES:
            data = _branch_rows(group_frame, branch)
            if data.empty:
                continue
            competence = data["competence"].to_numpy(dtype=float)
            tcp = data["tcp_target"].to_numpy(dtype=float)
            correctness = data["correctness"].to_numpy(dtype=int)
            error = 1 - correctness
            records.append(
                {
                    **base,
                    "branch": branch,
                    "count": int(competence.size),
                    "competence_mean": float(competence.mean()),
                    "tcp_mean": float(tcp.mean()),
                    "competence_tcp_gap": float(competence.mean() - tcp.mean()),
                    "tcp_mse": float(np.mean((competence - tcp) ** 2)),
                    "tcp_mae": float(np.mean(np.abs(competence - tcp))),
                    "tcp_pearson": _correlation(competence, tcp),
                    "tcp_spearman": _correlation(
                        _rank(competence), _rank(tcp)
                    ),
                    "error_auroc_defined": int(np.unique(error).size == 2),
                    "error_auroc": (
                        float(roc_auc_score(error, 1.0 - competence))
                        if np.unique(error).size == 2
                        else None
                    ),
                    "branch_accuracy": float(correctness.mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def pairwise_ordering_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Check whether competence orders simultaneously alive experts like TCP."""

    records: list[dict[str, Any]] = []
    group_columns = [
        key
        for key in ("experiment", "seed", "diagnostic_file", "split")
        if key in frame.columns
    ]
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    for group, group_frame in grouped:
        values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(group_columns, values))
        for left_index, left in enumerate(BRANCHES):
            for right in BRANCHES[left_index + 1 :]:
                required = [
                    "label",
                    f"{left}_prob",
                    f"{right}_prob",
                    f"predicted_competence_{left}",
                    f"predicted_competence_{right}",
                ]
                if any(key not in group_frame.columns for key in required):
                    continue
                columns = list(required)
                for branch in (left, right):
                    key = f"{branch}_alive"
                    if key in group_frame.columns:
                        columns.append(key)
                data = group_frame[columns].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna(subset=required)
                for branch in (left, right):
                    key = f"{branch}_alive"
                    if key in data.columns:
                        data = data[data[key] >= 0.5]
                if data.empty:
                    continue
                label = data["label"].round().astype(int).to_numpy()
                left_prob = data[f"{left}_prob"].clip(0.0, 1.0).to_numpy()
                right_prob = data[f"{right}_prob"].clip(0.0, 1.0).to_numpy()
                left_tcp = np.where(label == 1, left_prob, 1.0 - left_prob)
                right_tcp = np.where(label == 1, right_prob, 1.0 - right_prob)
                target_delta = left_tcp - right_tcp
                predicted_delta = (
                    data[f"predicted_competence_{left}"].to_numpy()
                    - data[f"predicted_competence_{right}"].to_numpy()
                )
                non_ties = np.abs(target_delta) > 0.02
                if not non_ties.any():
                    continue
                records.append(
                    {
                        **base,
                        "left_branch": left,
                        "right_branch": right,
                        "count": int(non_ties.sum()),
                        "ordering_accuracy": float(
                            np.mean(
                                np.sign(predicted_delta[non_ties])
                                == np.sign(target_delta[non_ties])
                            )
                        ),
                        "mean_absolute_tcp_gap": float(
                            np.mean(np.abs(target_delta[non_ties]))
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def competence_bin_table(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = [
        key
        for key in ("experiment", "seed", "diagnostic_file", "split")
        if key in frame.columns
    ]
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    edges = np.linspace(0.0, 1.0, bins + 1)
    for group, group_frame in grouped:
        values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(group_columns, values))
        for branch in BRANCHES:
            data = _branch_rows(group_frame, branch)
            if data.empty:
                continue
            competence = data["competence"].to_numpy(dtype=float)
            tcp = data["tcp_target"].to_numpy(dtype=float)
            correctness = data["correctness"].to_numpy(dtype=float)
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (
                    (competence >= low) & (competence <= high)
                    if high >= 1.0
                    else (competence >= low) & (competence < high)
                )
                if not mask.any():
                    continue
                records.append(
                    {
                        **base,
                        "branch": branch,
                        "bin": index,
                        "bin_low": float(low),
                        "bin_high": float(high),
                        "count": int(mask.sum()),
                        "mean_competence": float(competence[mask].mean()),
                        "mean_tcp": float(tcp[mask].mean()),
                        "empirical_accuracy": float(correctness[mask].mean()),
                    }
                )
    return pd.DataFrame.from_records(records)


def _write_competence_diagrams(frame: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    group_columns = [
        key
        for key in ("experiment", "seed", "split", "diagnostic_file")
        if key in frame.columns
    ]
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    for group, group_frame in grouped:
        values = group if isinstance(group, tuple) else (group,)
        group_map = dict(zip(group_columns, values))
        group_dir = out_dir
        for key in group_columns:
            value = (
                Path(str(group_map[key])).stem
                if key == "diagnostic_file"
                else group_map[key]
            )
            group_dir /= _safe_path_part(value)
        points = competence_bin_table(group_frame)
        if points.empty or "branch" not in points.columns:
            continue
        for branch in BRANCHES:
            branch_points = points[points["branch"] == branch]
            if branch_points.empty:
                continue
            group_dir.mkdir(parents=True, exist_ok=True)
            branch_points.to_csv(
                group_dir / f"tcp_competence_diagram_{branch}.csv",
                index=False,
            )
            fig, ax = plt.subplots(figsize=(4.0, 4.0))
            ax.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            ax.plot(
                branch_points["mean_competence"],
                branch_points["mean_tcp"],
                marker="o",
                linewidth=1.5,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Predicted competence")
            ax.set_ylabel("Mean true-class probability")
            ax.set_title(f"{branch.capitalize()} TCP competence")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                group_dir / f"tcp_competence_diagram_{branch}.png",
                dpi=200,
            )
            plt.close(fig)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit schema-v14 I1 competence predictions against continuous "
            "true-class-probability targets."
        )
    )
    parser.add_argument("--results-root", default="results/tri_modal_robust")
    parser.add_argument("--diagnostics", nargs="*", default=None)
    parser.add_argument("--out-dir", default="tables")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--experiment", nargs="*", default=None)
    parser.add_argument("--seed", nargs="*", default=None)
    parser.add_argument("--split", nargs="*", default=None)
    parser.add_argument("--diagnostic-file", nargs="*", default=None)
    parser.add_argument("--fail-if-empty", action="store_true")
    args = parser.parse_args()

    frame = _read_diagnostics(_diagnostic_paths(args))
    frame = _apply_filters(
        frame,
        experiments=_parse_filter(args.experiment),
        seeds=_parse_filter(args.seed),
        splits=_parse_filter(args.split),
        diagnostic_files=_parse_filter(args.diagnostic_file),
    )
    if frame.empty and args.fail_if_empty:
        raise RuntimeError("No current-schema competence diagnostics were found")

    out_dir = Path(args.out_dir)
    _write_csv(
        competence_table(frame),
        out_dir / "i1_tcp_competence_summary.csv",
    )
    _write_csv(
        pairwise_ordering_table(frame),
        out_dir / "i1_tcp_pairwise_ordering.csv",
    )
    _write_csv(
        competence_bin_table(frame),
        out_dir / "i1_tcp_competence_bins.csv",
    )
    _write_competence_diagrams(frame, Path(args.figures_dir))


if __name__ == "__main__":
    main()
