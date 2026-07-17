from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from paper.baselines.common import (
    binary_metrics,
    concat_long_from_sources,
    load_pt,
    read_label_csv,
    set_reproducible_seed,
    write_json,
    write_predictions,
)


def extract_markov_features(
    payload: dict[str, Any],
    *,
    num_states: int,
    max_api_events: int,
    smoothing: float,
) -> np.ndarray:
    """Extract a MaMaDroid-inspired API abstraction Markov feature vector.

    The original MaMaDroid abstracts API calls into package/family states and
    models their transition matrix. Current PT files do not preserve raw
    package/family API names, so this adapted baseline uses ``api_type_ids`` as
    the API abstraction state sequence. It should therefore be reported as a
    MaMaDroid-inspired baseline, not as a strict reproduction.
    """
    states = concat_long_from_sources(payload, "api_type_ids").numpy().astype(np.int64)
    observed_length = int(states.size)
    states = states[: int(max_api_events)]
    states = np.clip(states, 0, int(num_states) - 1)

    transition = np.full((int(num_states), int(num_states)), float(smoothing), dtype=np.float32)
    if states.size >= 2:
        src = states[:-1]
        dst = states[1:]
        for left, right in zip(src, dst):
            transition[int(left), int(right)] += 1.0
    row_sum = transition.sum(axis=1, keepdims=True)
    transition = transition / np.maximum(row_sum, 1e-12)

    occupancy = np.bincount(states, minlength=int(num_states)).astype(np.float32)
    occupancy = occupancy / max(float(occupancy.sum()), 1.0)
    length_feature = np.asarray(
        [min(float(observed_length) / float(max_api_events), 1.0)],
        dtype=np.float32,
    )
    return np.concatenate([transition.reshape(-1), occupancy, length_feature], axis=0).astype(np.float32)


def build_features(
    pt_dir: str | Path,
    csv_path: str | Path,
    *,
    num_states: int,
    max_api_events: int,
    smoothing: float,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, Any, list[dict[str, str]]]:
    frame = read_label_csv(csv_path)
    features: list[np.ndarray] = []
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
            features.append(
                extract_markov_features(
                    payload,
                    num_states=num_states,
                    max_api_events=max_api_events,
                    smoothing=smoothing,
                )
            )
            labels.append(int(getattr(row, "label")))
            kept_sha.append(sha)
        except Exception as exc:  # noqa: BLE001 - report failures and continue.
            failures.append({"sha256": sha, "error": f"{type(exc).__name__}: {exc}"})
    if not features:
        raise RuntimeError(f"No loadable samples from {csv_path}")
    kept_frame = frame[frame["sha256"].astype(str).isin(kept_sha)].copy()
    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64), kept_frame, failures


def evaluate_split(
    model: Pipeline,
    pt_dir: str | Path,
    csv_path: str | Path,
    out_dir: Path,
    split_name: str,
    *,
    num_states: int,
    max_api_events: int,
    smoothing: float,
    show_progress: bool = True,
) -> dict[str, Any]:
    features, labels, frame, failures = build_features(
        pt_dir,
        csv_path,
        num_states=num_states,
        max_api_events=max_api_events,
        smoothing=smoothing,
        show_progress=show_progress,
    )
    prob = model.predict_proba(features)[:, 1]
    metrics = binary_metrics(labels, prob)
    metrics["num_failed"] = len(failures)
    metrics["csv"] = str(csv_path)
    metrics["pt_dir"] = str(pt_dir)
    write_predictions(out_dir / f"predictions_{split_name}.csv", frame, prob)
    if failures:
        write_json(out_dir / f"failures_{split_name}.json", {"failures": failures})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a MaMaDroid-inspired Markov baseline.")
    parser.add_argument("--train-pt-dir", required=True)
    parser.add_argument("--val-pt-dir", required=True)
    parser.add_argument("--test-pt-dir", required=True)
    parser.add_argument("--train-csv", default="labels/train.csv")
    parser.add_argument("--val-csv", default="labels/val.csv")
    parser.add_argument("--test-csv", default="labels/test.csv")
    parser.add_argument("--extra-test-csv", action="append", default=[])
    parser.add_argument("--out-dir", default="paper/outputs/mamadroid_inspired")
    parser.add_argument("--num-states", type=int, default=16)
    parser.add_argument("--max-api-events", type=int, default=4096)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    show_progress = not bool(args.no_progress)
    set_reproducible_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_labels, _train_frame, train_failures = build_features(
        args.train_pt_dir,
        args.train_csv,
        num_states=int(args.num_states),
        max_api_events=int(args.max_api_events),
        smoothing=float(args.smoothing),
        show_progress=show_progress,
    )
    print(f"Training MaMaDroid-inspired model on {train_features.shape[0]} samples...", flush=True)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(args.c),
                    class_weight="balanced",
                    max_iter=int(args.max_iter),
                    solver="lbfgs",
                    random_state=int(args.seed),
                ),
            ),
        ]
    )
    model.fit(train_features, train_labels)

    summary: dict[str, Any] = {
        "method": "MaMaDroid-inspired API abstraction Markov baseline",
        "note": (
            "Adapted baseline: uses PT api_type_ids as API abstraction states, "
            "not raw package/family names from the original MaMaDroid paper."
        ),
        "feature": {
            "num_states": int(args.num_states),
            "max_api_events": int(args.max_api_events),
            "smoothing": float(args.smoothing),
            "dim": int(train_features.shape[1]),
        },
        "train": {"num_eval": int(train_labels.size), "num_failed": len(train_failures)},
        "validation": evaluate_split(
            model,
            args.val_pt_dir,
            args.val_csv,
            out_dir,
            "val",
            num_states=int(args.num_states),
            max_api_events=int(args.max_api_events),
            smoothing=float(args.smoothing),
            show_progress=show_progress,
        ),
        "test": evaluate_split(
            model,
            args.test_pt_dir,
            args.test_csv,
            out_dir,
            "test",
            num_states=int(args.num_states),
            max_api_events=int(args.max_api_events),
            smoothing=float(args.smoothing),
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
            num_states=int(args.num_states),
            max_api_events=int(args.max_api_events),
            smoothing=float(args.smoothing),
            show_progress=show_progress,
        )
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
