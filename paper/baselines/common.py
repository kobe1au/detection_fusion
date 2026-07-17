from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable paper baselines."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def validation_selection_indices(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[int], dict[str, Any]]:
    """Return the same group-stratified checkpoint-selection split as Full."""
    from fusion.train import split_validation_dataset

    package_col = next(
        (
            name
            for name in ("pkg_name", "package_name", "package")
            if name in frame.columns
        ),
        None,
    )
    sids = frame["sha256"].astype(str).tolist()
    if package_col is None:
        groups = list(sids)
    else:
        groups = []
        for sid, value in zip(sids, frame[package_col].tolist()):
            group = str(value or "").strip().lower()
            groups.append(group if group not in {"", "nan", "none", "null"} else sid)

    class FrameView:
        def __init__(self) -> None:
            self.sample_sids = sids
            self.sample_groups = groups
            self.sample_labels = frame["label"].astype(int).tolist()

        def __len__(self) -> int:
            return len(self.sample_sids)

        def __getitem__(self, index: int) -> int:
            return index

    selection, _calibration, summary = split_validation_dataset(
        {
            "train": {"seed": int(seed)},
            "calibration": {
                "validation_fraction": float(calibration_fraction),
                "split_seed": int(seed),
            },
        },
        FrameView(),
    )
    return [int(index) for index in selection.indices], summary


def read_label_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sha256" not in frame.columns:
        raise ValueError(f"{path} must contain a sha256 column")
    if "label" not in frame.columns:
        raise ValueError(f"{path} must contain a label column")
    frame = frame.copy()
    frame["sha256"] = frame["sha256"].astype(str).str.strip().str.lower()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if bool((frame["sha256"] == "").any()):
        raise ValueError(f"{path} contains an empty sha256")
    if bool(frame["sha256"].duplicated().any()):
        duplicates = frame.loc[frame["sha256"].duplicated(), "sha256"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate sha256 values: {duplicates}")
    invalid_labels = sorted(set(frame["label"].tolist()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{path} contains non-binary labels: {invalid_labels}")
    return frame


def pt_path_for(pt_dir: str | Path, sha256: str) -> Path:
    return Path(pt_dir) / f"{sha256}.pt"


def load_pt(pt_dir: str | Path, sha256: str) -> dict[str, Any]:
    path = pt_path_for(pt_dir, sha256)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected PT dict payload: {path}")
    return obj


def tensor_1d(obj: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if obj is None:
        out = torch.empty((0,))
    elif isinstance(obj, torch.Tensor):
        out = obj.detach().cpu().view(-1)
    else:
        out = torch.as_tensor(obj).view(-1)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def first_present(sources: Iterable[dict[str, Any]], key: str) -> Any:
    for source in sources:
        if isinstance(source, dict) and key in source and source[key] is not None:
            return source[key]
    return None


def dex_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dex_list = payload.get("dex_list")
    if isinstance(dex_list, list):
        return [dex for dex in dex_list if isinstance(dex, dict)]
    return []


def all_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [payload, *dex_sources(payload)]


def concat_long_from_sources(payload: dict[str, Any], key: str) -> torch.Tensor:
    values = []
    for source in all_sources(payload):
        value = source.get(key)
        if value is not None:
            tensor = tensor_1d(value, dtype=torch.long)
            if tensor.numel() > 0:
                values.append(tensor)
    if not values:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(values).long()


def concat_float_from_sources(payload: dict[str, Any], key: str) -> torch.Tensor:
    values = []
    for source in all_sources(payload):
        value = source.get(key)
        if value is not None:
            tensor = tensor_1d(value, dtype=torch.float32)
            if tensor.numel() > 0:
                values.append(tensor)
    if not values:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(values).float()


def first_long(payload: dict[str, Any], key: str) -> torch.Tensor:
    return tensor_1d(first_present(all_sources(payload), key), dtype=torch.long)


def first_float(payload: dict[str, Any], key: str) -> torch.Tensor:
    return tensor_1d(first_present(all_sources(payload), key), dtype=torch.float32)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    if y_true.size == 0:
        return float("nan")
    confidence = np.maximum(prob, 1.0 - prob)
    pred = (prob >= 0.5).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        acc = np.mean(pred[mask] == y_true[mask])
        conf = np.mean(confidence[mask])
        ece += float(np.mean(mask)) * abs(float(acc) - float(conf))
    return float(ece)


def binary_metrics(y_true: np.ndarray, prob_malware: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    prob_malware = np.asarray(prob_malware).astype(float)
    pred = (prob_malware >= 0.5).astype(int)
    out: dict[str, Any] = {
        "num_eval": int(y_true.size),
        "acc": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob_malware)),
        "ece_10": expected_calibration_error(y_true, prob_malware, bins=10),
    }
    precision, recall, _f1, _support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )
    out["recall_benign"] = float(recall[0])
    out["recall_malware"] = float(recall[1])
    out["precision_benign"] = float(precision[0])
    out["precision_malware"] = float(precision[1])
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, prob_malware))
        out["ap"] = float(average_precision_score(y_true, prob_malware))
        out["auc_defined"] = 1
        out["ap_defined"] = 1
    else:
        out["auc"] = None
        out["ap"] = None
        out["auc_defined"] = 0
        out["ap_defined"] = 0
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            return v if math.isfinite(v) else None
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(path: str | Path, frame: pd.DataFrame, prob: np.ndarray) -> None:
    pred_frame = frame[["sha256", "label"]].copy()
    pred_frame["prob_malware"] = np.asarray(prob, dtype=float)
    pred_frame["pred"] = (pred_frame["prob_malware"] >= 0.5).astype(int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(path, index=False)
