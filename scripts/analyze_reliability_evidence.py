from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.utils import strict_binary_integer


MODALITY_BRANCHES = ("api", "graph", "manifest")
BRANCHES = MODALITY_BRANCHES

EVIDENCE_FIELDS = (
    "api_integrity",
    "api_encoder_coverage",
    "api_total_pipeline_coverage",
    "api_extractor_coverage",
    "api_runtime_encoder_coverage",
    "effective_api_integrity",
    "graph_integrity",
    "graph_encoder_coverage",
    "effective_graph_integrity",
    "manifest_integrity",
    "effective_manifest_integrity",
    "code_integrity",
    "api_graph_anchor_support",
    "manifest_code_support",
    "manifest_to_code_conflict",
    "code_to_manifest_conflict",
    "predictive_conflict",
    "predictive_conflict_max",
    "raw_conflict",
    "acceptance_score",
    "discount_api",
    "discount_graph",
    "discount_manifest",
    "fusion_weight_api",
    "fusion_weight_graph",
    "fusion_weight_manifest",
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
    "api_encoder_coverage": ("api",),
    "api_total_pipeline_coverage": ("api",),
    "api_extractor_coverage": ("api",),
    "api_runtime_encoder_coverage": ("api",),
    "effective_api_integrity": ("api",),
    "graph_integrity": ("graph",),
    "graph_encoder_coverage": ("graph",),
    "effective_graph_integrity": ("graph",),
    "manifest_integrity": ("manifest",),
    "effective_manifest_integrity": ("manifest",),
    "code_integrity": ("api", "graph"),
    "api_graph_anchor_support": ("api", "graph"),
    "manifest_code_support": ("manifest",),
    "manifest_to_code_conflict": ("manifest",),
    "code_to_manifest_conflict": ("api", "graph"),
    "max_manifest_code_conflict": ("api", "graph", "manifest"),
    "predictive_conflict": ("api", "graph", "manifest"),
    "predictive_conflict_max": ("api", "graph", "manifest"),
    "raw_conflict": ("api", "graph", "manifest"),
    "discount_api": ("api",),
    "discount_graph": ("graph",),
    "discount_manifest": ("manifest",),
    "min_modality_discount": ("api", "graph", "manifest"),
    "mean_modality_discount": ("api", "graph", "manifest"),
}

NATURAL_SUBSET_SPECS = {
    "api_low_integrity": ("api_integrity", "low"),
    "api_low_encoder_coverage": ("api_encoder_coverage", "low"),
    "api_low_effective_integrity": ("effective_api_integrity", "low"),
    "graph_low_integrity": ("graph_integrity", "low"),
    "graph_low_encoder_coverage": ("graph_encoder_coverage", "low"),
    "graph_low_effective_integrity": ("effective_graph_integrity", "low"),
    "manifest_low_integrity": ("manifest_integrity", "low"),
    "manifest_low_effective_integrity": ("effective_manifest_integrity", "low"),
    "code_low_integrity": ("code_integrity", "low"),
    "api_graph_low_support": ("api_graph_anchor_support", "low"),
    "manifest_code_low_support": ("manifest_code_support", "low"),
    "manifest_to_code_high_conflict": ("manifest_to_code_conflict", "high"),
    "code_to_manifest_high_conflict": ("code_to_manifest_conflict", "high"),
    "max_manifest_code_high_conflict": ("max_manifest_code_conflict", "high"),
    "predictive_high_conflict": ("predictive_conflict", "high"),
    "raw_high_conflict": ("raw_conflict", "high"),
    "api_low_trust": ("discount_api", "low"),
    "graph_low_trust": ("discount_graph", "low"),
    "manifest_low_trust": ("discount_manifest", "low"),
    "min_modality_low_trust": ("min_modality_discount", "low"),
    "mean_modality_low_trust": ("mean_modality_discount", "low"),
    "low_acceptance": ("acceptance_score", "low"),
}

NATURAL_MISSING_SPECS = {
    "api_naturally_missing": "api_alive",
    "graph_naturally_missing": "graph_alive",
    "manifest_naturally_missing": "manifest_alive",
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
        for branch in MODALITY_BRANCHES:
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


def reliability_signal_diagnostics_table(
    frame: pd.DataFrame,
    *,
    permutations: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compare calibrated reliability with its component signals and shuffles."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    intrinsic_keys = {
        "api": "api_integrity",
        "graph": "graph_integrity",
        "manifest": "manifest_integrity",
    }
    records: list[dict[str, Any]] = []
    group_columns = [
        column
        for column in ("experiment", "seed", "diagnostic_file", "split")
        if column in frame.columns
    ]
    rng = np.random.default_rng(int(random_seed))
    for group, group_frame in frame.groupby(group_columns, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(group_columns, group_values))
        for branch in MODALITY_BRANCHES:
            reliability_key = f"predicted_reliability_{branch}"
            correct_key = f"{branch}_correct"
            if reliability_key not in group_frame.columns or correct_key not in group_frame.columns:
                continue
            data = group_frame[[reliability_key, correct_key]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if data.empty:
                continue
            reliability = data[reliability_key].clip(0.0, 1.0).to_numpy(dtype=float)
            correctness = _finite_binary(data[correct_key])
            shuffled_auc = np.asarray(
                [_safe_auc(correctness, rng.permutation(reliability)) for _ in range(permutations)],
                dtype=float,
            )
            native_auc = _safe_auc(correctness, reliability)

            intrinsic_auc = float("nan")
            intrinsic_key = intrinsic_keys[branch]
            if intrinsic_key in group_frame.columns:
                component = group_frame.loc[data.index, intrinsic_key].apply(
                    pd.to_numeric, errors="coerce"
                )
                mask = component.notna().to_numpy()
                if mask.any():
                    intrinsic_auc = _safe_auc(
                        correctness[mask], component.to_numpy(dtype=float)[mask]
                    )

            certainty_auc = float("nan")
            uncertainty_key = f"uncertainty_proxy_{branch}"
            if uncertainty_key in group_frame.columns:
                component = group_frame.loc[data.index, uncertainty_key].apply(
                    pd.to_numeric, errors="coerce"
                )
                mask = component.notna().to_numpy()
                if mask.any():
                    certainty_auc = _safe_auc(
                        correctness[mask], 1.0 - component.to_numpy(dtype=float)[mask]
                    )

            shuffled_mean = float(np.nanmean(shuffled_auc))
            records.append(
                {
                    **base,
                    "branch": branch,
                    "count": int(reliability.size),
                    "reliability_auc": native_auc,
                    "intrinsic_integrity_auc": intrinsic_auc,
                    "evidential_certainty_auc": certainty_auc,
                    "shuffled_reliability_auc_mean": shuffled_mean,
                    "shuffled_reliability_auc_std": float(np.nanstd(shuffled_auc)),
                    "reliability_permutation_gap": float(native_auc - shuffled_mean),
                    "permutations": int(permutations),
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
    discount_columns = [
        column
        for column in ("discount_api", "discount_graph", "discount_manifest")
        if column in out.columns
    ]
    if discount_columns:
        numeric_discounts = out[discount_columns].apply(pd.to_numeric, errors="coerce")
        out["min_modality_discount"] = numeric_discounts.min(axis=1)
        out["mean_modality_discount"] = numeric_discounts.mean(axis=1)
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


def _classification_metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"label", "prob_malware", "pred"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    data = frame[list(required)].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if data.empty:
        return {}
    valid_index = data.index
    labels_arr = np.asarray(
        [
            strict_binary_integer(value, field_name="analysis label")
            for value in data["label"].tolist()
        ],
        dtype=np.int64,
    )
    probs_arr = data["prob_malware"].clip(0.0, 1.0).to_numpy(dtype=float)
    preds_arr = np.asarray(
        [
            strict_binary_integer(value, field_name="analysis prediction")
            for value in data["pred"].tolist()
        ],
        dtype=np.int64,
    )
    confidence = np.maximum(probs_arr, 1.0 - probs_arr)
    correctness = (preds_arr == labels_arr).astype(float)
    auc_defined = _auc_defined(labels_arr.astype(float))
    ap_defined = _ap_defined(labels_arr.astype(float))
    out: dict[str, Any] = {
        "count": int(labels_arr.size),
        "positive_rate": float(labels_arr.mean()) if labels_arr.size else float("nan"),
        "acc": float(accuracy_score(labels_arr, preds_arr)),
        "error_rate": float(1.0 - accuracy_score(labels_arr, preds_arr)),
        "macro_f1": float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)),
        "brier": float(np.mean((probs_arr - labels_arr.astype(float)) ** 2)),
        "ece_10": _ece(confidence, correctness, bins=10),
        "mean_confidence": float(confidence.mean()),
        "confidence_accuracy_gap": float(confidence.mean() - correctness.mean()),
        "auc_defined": int(auc_defined),
        "auc": _safe_auc(labels_arr.astype(float), probs_arr),
        "ap_defined": int(ap_defined),
        "ap": _safe_ap(labels_arr.astype(float), probs_arr),
    }
    malware_mask = labels_arr == 1
    if malware_mask.any():
        out["malware_fn_rate"] = float(((preds_arr == 0) & malware_mask).sum() / malware_mask.sum())
    if "acceptance_score" in frame.columns:
        acceptance = pd.to_numeric(frame.loc[valid_index, "acceptance_score"], errors="coerce")
        if acceptance.notna().any():
            out["acceptance_score_mean"] = float(acceptance.dropna().mean())
    if "rejected" in frame.columns:
        rejected = pd.to_numeric(frame.loc[valid_index, "rejected"], errors="coerce")
        if rejected.notna().any():
            rejected_bool = rejected.fillna(1.0).to_numpy(dtype=float) >= 0.5
            accepted_bool = ~rejected_bool
            out["rejection_rate"] = float(rejected_bool.mean())
            out["accepted_rate"] = float(accepted_bool.mean())
            out["accepted_count"] = int(accepted_bool.sum())
            if accepted_bool.any():
                accepted_errors = (preds_arr[accepted_bool] != labels_arr[accepted_bool]).astype(float)
                out["selective_risk"] = float(accepted_errors.mean())
                out["selective_acc"] = float(1.0 - accepted_errors.mean())
                out["selective_macro_f1"] = float(
                    f1_score(
                        labels_arr[accepted_bool],
                        preds_arr[accepted_bool],
                        average="macro",
                        zero_division=0,
                    )
                )
                accepted_malware = accepted_bool & malware_mask
                if malware_mask.any():
                    out["accepted_fn_risk_among_malware"] = float(
                        ((preds_arr == 0) & accepted_malware).sum()
                        / malware_mask.sum()
                    )
                if accepted_malware.any():
                    out["fn_rate_given_accepted_malware"] = float(
                        ((preds_arr == 0) & accepted_malware).sum() / accepted_malware.sum()
                    )
            else:
                out["selective_risk"] = float("nan")
                out["selective_acc"] = float("nan")
                out["selective_macro_f1"] = float("nan")
    return out


def natural_degradation_subset_table(
    frame: pd.DataFrame,
    quantile: float = 1.0 / 3.0,
    min_count: int = 10,
) -> pd.DataFrame:
    """Evaluate clean, naturally low-quality evidence subsets without perturbing samples."""
    if not (0.0 < quantile < 0.5):
        raise ValueError("quantile must be in (0, 0.5)")
    records: list[dict[str, Any]] = []
    frame = _with_derived_evidence(frame)
    group_columns = ["experiment", "seed", "diagnostic_file", "split"]
    available_groups = [column for column in group_columns if column in frame.columns]
    numeric_columns = [
        *{spec[0] for spec in NATURAL_SUBSET_SPECS.values()},
        *NATURAL_MISSING_SPECS.values(),
        "label",
        "prob_malware",
        "pred",
        "acceptance_score",
        "rejected",
    ]
    numeric = _to_numeric(frame, [column for column in numeric_columns if column in frame.columns])
    for group, group_frame in numeric.groupby(available_groups, dropna=False):
        group_values = group if isinstance(group, tuple) else (group,)
        base = dict(zip(available_groups, group_values))

        for subset_name, (evidence, direction) in NATURAL_SUBSET_SPECS.items():
            if evidence not in group_frame.columns:
                continue
            values = pd.to_numeric(group_frame[evidence], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                continue
            threshold = float(valid.quantile(quantile if direction == "low" else 1.0 - quantile))
            mask = values <= threshold if direction == "low" else values >= threshold
            subset = group_frame[mask.fillna(False)]
            if len(subset) < min_count:
                continue
            metrics = _classification_metrics_for_frame(subset)
            if not metrics:
                continue
            records.append(
                {
                    **base,
                    "subset": subset_name,
                    "subset_type": "natural_evidence_quantile",
                    "evidence": evidence,
                    "direction": direction,
                    "quantile": float(quantile),
                    "threshold": threshold,
                    "evidence_mean": float(values[mask].mean()),
                    **metrics,
                }
            )

        for subset_name, alive_column in NATURAL_MISSING_SPECS.items():
            if alive_column not in group_frame.columns:
                continue
            values = pd.to_numeric(group_frame[alive_column], errors="coerce")
            mask = values < 0.5
            subset = group_frame[mask.fillna(False)]
            if len(subset) < min_count:
                continue
            metrics = _classification_metrics_for_frame(subset)
            if not metrics:
                continue
            records.append(
                {
                    **base,
                    "subset": subset_name,
                    "subset_type": "natural_missing_modality",
                    "evidence": alive_column,
                    "direction": "missing",
                    "quantile": float("nan"),
                    "threshold": 0.5,
                    "evidence_mean": float(values[mask].mean()),
                    **metrics,
                }
            )
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
    for branch in MODALITY_BRANCHES:
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
    parser.add_argument(
        "--natural-subset-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Bottom/top quantile used for natural degradation subset cuts.",
    )
    parser.add_argument(
        "--natural-subset-min-count",
        type=int,
        default=10,
        help="Minimum samples required before writing a natural subset row.",
    )
    parser.add_argument(
        "--reliability-permutations",
        type=int,
        default=100,
        help="Number of score shuffles used for the reliability-alignment diagnostic.",
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
        pd.DataFrame().to_csv(Path(args.out_dir) / "i1_reliability_signal_diagnostics.csv", index=False)
        pd.DataFrame().to_csv(Path(args.out_dir) / "i1_evidence_groups.csv", index=False)
        pd.DataFrame().to_csv(Path(args.out_dir) / "i1_evidence_bin_effects.csv", index=False)
        pd.DataFrame().to_csv(Path(args.out_dir) / "natural_degradation_subsets.csv", index=False)
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
    _write_csv(
        reliability_signal_diagnostics_table(
            frame,
            permutations=int(args.reliability_permutations),
        ),
        Path(args.out_dir) / "i1_reliability_signal_diagnostics.csv",
    )
    _write_csv(evidence_group_table(frame), Path(args.out_dir) / "i1_evidence_groups.csv")
    _write_csv(
        evidence_bin_effects_table(frame, bin_scope=args.bin_scope),
        Path(args.out_dir) / "i1_evidence_bin_effects.csv",
    )
    _write_csv(
        natural_degradation_subset_table(
            frame,
            quantile=float(args.natural_subset_quantile),
            min_count=int(args.natural_subset_min_count),
        ),
        Path(args.out_dir) / "natural_degradation_subsets.csv",
    )
    write_reliability_diagrams(frame, Path(args.figures_dir))


if __name__ == "__main__":
    main()
