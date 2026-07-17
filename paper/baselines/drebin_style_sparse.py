from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from paper.baselines.common import (
    binary_metrics,
    concat_float_from_sources,
    concat_long_from_sources,
    first_float,
    first_long,
    load_pt,
    read_label_csv,
    set_reproducible_seed,
    write_json,
    write_predictions,
)


def _add_count_features(prefix: str, values: np.ndarray, features: dict[str, float], *, max_items: int | None = None) -> None:
    if values.size == 0:
        return
    if max_items is not None and values.size > max_items:
        values = values[:max_items]
    for key, count in Counter(int(v) for v in values if int(v) >= 0).items():
        features[f"{prefix}:{key}"] = float(count)


def _add_dense_features(prefix: str, values: np.ndarray, features: dict[str, float]) -> None:
    for idx, value in enumerate(values):
        v = float(value)
        if v != 0.0 and np.isfinite(v):
            features[f"{prefix}:{idx}"] = v


def extract_drebin_style_features(payload: dict[str, Any], *, max_api_events: int = 4096) -> dict[str, float]:
    """Extract sparse static features from current tri-modal PT payload.

    This is an adapted Drebin-style baseline. It uses static manifest/API
    indicators available in the PT files rather than the exact original Drebin
    feature templates (which require raw APK string/API names).
    """
    features: dict[str, float] = {}

    api_ids = concat_long_from_sources(payload, "api_ids").numpy()
    api_types = concat_long_from_sources(payload, "api_type_ids").numpy()
    api_sensitive = concat_float_from_sources(payload, "api_sensitive_mask").numpy()
    api_in_graph = concat_float_from_sources(payload, "api_in_graph_mask").numpy()
    _add_count_features("api_hash", api_ids, features, max_items=max_api_events)
    _add_count_features("api_type", api_types, features, max_items=max_api_events)
    if api_sensitive.size:
        features["api_sensitive_count"] = float((api_sensitive[:max_api_events] > 0.5).sum())
        features["api_sensitive_ratio"] = float((api_sensitive[:max_api_events] > 0.5).mean())
    if api_in_graph.size:
        features["api_in_graph_count"] = float((api_in_graph[:max_api_events] > 0.5).sum())
        features["api_in_graph_ratio"] = float((api_in_graph[:max_api_events] > 0.5).mean())

    _add_dense_features(
        "api_semantic",
        first_float(payload, "api_semantic_category_counts").numpy(),
        features,
    )
    _add_dense_features(
        "graph_semantic",
        first_float(payload, "graph_semantic_category_counts").numpy(),
        features,
    )
    _add_dense_features(
        "manifest_category",
        first_float(payload, "manifest_category_counts").numpy(),
        features,
    )
    _add_dense_features(
        "manifest_component",
        first_float(payload, "manifest_component_category_counts").numpy(),
        features,
    )
    _add_dense_features("manifest_stats", first_float(payload, "manifest_stats").numpy(), features)
    _add_dense_features("manifest_x", first_float(payload, "manifest_x").numpy(), features)
    _add_count_features("permission", first_long(payload, "manifest_permission_ids").numpy(), features)
    _add_count_features("intent", first_long(payload, "manifest_intent_ids").numpy(), features)

    return features


def build_feature_dicts(
    pt_dir: str | Path,
    csv_path: str | Path,
    *,
    max_api_events: int,
    show_progress: bool = True,
) -> tuple[list[dict[str, float]], np.ndarray, Any, Any]:
    frame = read_label_csv(csv_path)
    feature_dicts: list[dict[str, float]] = []
    labels: list[int] = []
    kept_sha: list[str] = []
    failures: list[dict[str, str]] = []
    rows = list(frame.itertuples(index=False))
    iterator = tqdm(
        rows,
        desc=f"extract {Path(csv_path).stem}",
        unit="apk",
        leave=False,
        disable=not show_progress,
    )
    for row in iterator:
        sha = str(getattr(row, "sha256"))
        try:
            payload = load_pt(pt_dir, sha)
            feature_dicts.append(extract_drebin_style_features(payload, max_api_events=max_api_events))
            labels.append(int(getattr(row, "label")))
            kept_sha.append(sha)
        except Exception as exc:  # noqa: BLE001 - baseline should report bad samples and continue.
            failures.append({"sha256": sha, "error": f"{type(exc).__name__}: {exc}"})
    if not feature_dicts:
        raise RuntimeError(f"No loadable samples from {csv_path}")
    kept_frame = frame[frame["sha256"].astype(str).isin(kept_sha)].copy()
    return feature_dicts, np.asarray(labels, dtype=np.int64), kept_frame, failures


def evaluate_split(
    model: Pipeline,
    pt_dir: str | Path,
    csv_path: str | Path,
    out_dir: Path,
    split_name: str,
    *,
    max_api_events: int,
    show_progress: bool = True,
) -> dict[str, Any]:
    feature_dicts, labels, frame, failures = build_feature_dicts(
        pt_dir,
        csv_path,
        max_api_events=max_api_events,
        show_progress=show_progress,
    )
    prob = model.predict_proba(feature_dicts)[:, 1]
    metrics = binary_metrics(labels, prob)
    metrics["num_failed"] = len(failures)
    metrics["csv"] = str(csv_path)
    metrics["pt_dir"] = str(pt_dir)
    write_predictions(out_dir / f"predictions_{split_name}.csv", frame, prob)
    if failures:
        write_json(out_dir / f"failures_{split_name}.json", {"failures": failures})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a Drebin-style sparse static baseline.")
    parser.add_argument("--train-pt-dir", required=True)
    parser.add_argument("--val-pt-dir", required=True)
    parser.add_argument("--test-pt-dir", required=True)
    parser.add_argument("--train-csv", default="labels/train.csv")
    parser.add_argument("--val-csv", default="labels/val.csv")
    parser.add_argument("--test-csv", default="labels/test.csv")
    parser.add_argument("--extra-test-csv", action="append", default=[], help="Additional CSV to evaluate, e.g. natural subset CSV.")
    parser.add_argument("--out-dir", default="paper/outputs/drebin_style")
    parser.add_argument("--max-api-events", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    show_progress = not bool(args.no_progress)
    set_reproducible_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_labels, _train_frame, train_failures = build_feature_dicts(
        args.train_pt_dir,
        args.train_csv,
        max_api_events=int(args.max_api_events),
        show_progress=show_progress,
    )
    print(f"Training Drebin-style model on {len(train_features)} samples...", flush=True)
    model = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    C=float(args.c),
                    class_weight="balanced",
                    max_iter=int(args.max_iter),
                    n_jobs=-1,
                    solver="saga",
                    penalty="l2",
                    random_state=int(args.seed),
                    verbose=0,
                ),
            ),
        ]
    )
    model.fit(train_features, train_labels)

    summary: dict[str, Any] = {
        "method": "Drebin-style sparse static baseline",
        "train": {"num_eval": int(train_labels.size), "num_failed": len(train_failures)},
        "validation": evaluate_split(
            model,
            args.val_pt_dir,
            args.val_csv,
            out_dir,
            "val",
            max_api_events=int(args.max_api_events),
            show_progress=show_progress,
        ),
        "test": evaluate_split(
            model,
            args.test_pt_dir,
            args.test_csv,
            out_dir,
            "test",
            max_api_events=int(args.max_api_events),
            show_progress=show_progress,
        ),
        "extra_eval": {},
    }
    for csv_path in args.extra_test_csv:
        name = Path(csv_path).stem
        summary["extra_eval"][name] = evaluate_split(
            model,
            args.test_pt_dir,
            csv_path,
            out_dir,
            name,
            max_api_events=int(args.max_api_events),
            show_progress=show_progress,
        )
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
