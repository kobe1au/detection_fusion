from __future__ import annotations

import hashlib
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

from fusion.utils import strict_binary_integer, strict_finite_integer


VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION = 2
VALIDATION_ROLE_ASSIGNMENT_PROTOCOL = (
    "year_label_stratified_package_group_2to1_v3"
)
DEFAULT_VALIDATION_ROLE_ASSIGNMENT = Path(
    "labels/validation_roles_protocol_v3.json"
)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable paper baselines."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _strict_binary_label_series(series: pd.Series, *, context: str) -> pd.Series:
    parsed: list[int] = []
    invalid_rows: list[Any] = []
    for row_index, raw_value in series.items():
        try:
            parsed.append(
                strict_binary_integer(
                    raw_value,
                    field_name=f"{context} label at row {row_index}",
                )
            )
        except ValueError:
            invalid_rows.append(row_index)
    if invalid_rows:
        examples = [
            {"row": row_index, "label": series.loc[row_index]}
            for row_index in invalid_rows[:10]
        ]
        raise ValueError(
            f"{context} contains labels that are not finite binary integers; "
            f"examples={examples}"
        )
    return pd.Series(parsed, index=series.index, dtype="int64")


def enforce_formal_split_completeness(
    split_name: str,
    *,
    num_eval: int,
    failures: Iterable[Any] | None = None,
) -> None:
    """Refuse paper metrics computed on only the successfully loaded subset."""
    num_eval = strict_finite_integer(
        num_eval, field_name=f"{split_name}.num_eval"
    )
    failure_rows = list(failures or [])
    if num_eval > 0 and not failure_rows:
        return
    raise RuntimeError(
        f"{split_name}: formal baseline refuses success-subset metrics; "
        f"num_eval={num_eval}, num_failed={len(failure_rows)}, "
        f"failure_examples={failure_rows[:3]}"
    )


def validation_selection_indices(
    frame: pd.DataFrame,
    *,
    validation_csv_path: str | Path,
    role_assignment_path: str | Path = DEFAULT_VALIDATION_ROLE_ASSIGNMENT,
) -> tuple[list[int], dict[str, Any]]:
    """Load the immutable v3 ``model_selection`` identities.

    Paper baselines must not independently redraw validation subsets. The
    assignment file is accepted only when it exactly matches the bytes and
    identities of the supplied validation CSV.
    """

    validation_csv = resolve_path(validation_csv_path)
    role_assignment = resolve_path(role_assignment_path)
    if not validation_csv.is_file():
        raise FileNotFoundError(
            f"Validation CSV not found: {validation_csv}"
        )
    if not role_assignment.is_file():
        raise FileNotFoundError(
            f"Validation role assignment not found: {role_assignment}"
        )
    try:
        payload = json.loads(role_assignment.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read validation role assignment {role_assignment}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Validation role assignment must be a JSON mapping")
    if (
        int(payload.get("schema_version", -1))
        != VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Validation role assignment schema mismatch: "
            f"expected={VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION} "
            f"actual={payload.get('schema_version')!r}"
        )
    if payload.get("protocol") != VALIDATION_ROLE_ASSIGNMENT_PROTOCOL:
        raise ValueError(
            "Validation role assignment protocol mismatch: "
            f"expected={VALIDATION_ROLE_ASSIGNMENT_PROTOCOL!r} "
            f"actual={payload.get('protocol')!r}"
        )

    expected_csv_sha = str(payload.get("validation_csv_sha256") or "").lower()
    actual_csv_sha = _file_sha256(validation_csv)
    if expected_csv_sha != actual_csv_sha:
        raise ValueError(
            "Validation role assignment was built for a different validation "
            f"CSV: expected={expected_csv_sha!r} actual={actual_csv_sha!r}"
        )

    required_columns = {"sha256", "label"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Validation frame is missing required columns: "
            f"{missing_columns}"
        )
    sids = frame["sha256"].astype(str).str.strip().str.lower().tolist()
    if any(not sid for sid in sids):
        raise ValueError("Validation frame contains an empty sha256")
    if len(set(sids)) != len(sids):
        raise ValueError("Validation frame contains duplicate sample identities")
    _strict_binary_label_series(
        frame["label"], context="validation selection frame"
    )
    index_by_sid = {sid: index for index, sid in enumerate(sids)}

    role_names = ("model_selection", "decision_calibration")
    roles = payload.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(role_names):
        raise ValueError(
            "Validation role assignment must contain exactly roles "
            f"{list(role_names)}"
        )
    role_ids: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role_name in role_names:
        raw_ids = roles[role_name]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(
                f"Validation role {role_name!r} must be a non-empty list"
            )
        normalized = [str(value).strip().lower() for value in raw_ids]
        if any(not sid for sid in normalized):
            raise ValueError(
                f"Validation role {role_name!r} contains an empty identity"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"Validation role {role_name!r} contains duplicate identities"
            )
        overlap = seen.intersection(normalized)
        if overlap:
            raise ValueError(
                "Validation roles overlap; examples="
                f"{sorted(overlap)[:10]}"
            )
        unknown = sorted(set(normalized) - set(index_by_sid))
        if unknown:
            raise ValueError(
                f"Validation role {role_name!r} contains unknown identities: "
                f"{unknown[:10]}"
            )
        seen.update(normalized)
        role_ids[role_name] = normalized

    missing = sorted(set(sids) - seen)
    if missing:
        raise ValueError(
            "Validation role assignment does not cover the full validation "
            f"set; missing={len(missing)} examples={missing[:10]}"
        )

    declared_counts = payload.get("counts")
    actual_counts = {
        role_name: len(role_ids[role_name]) for role_name in role_names
    }
    if declared_counts != actual_counts:
        raise ValueError(
            "Validation role assignment count mismatch: "
            f"declared={declared_counts!r} actual={actual_counts!r}"
        )

    package_col = next(
        (
            name
            for name in ("pkg_name", "package_name", "package")
            if name in frame.columns
        ),
        None,
    )
    if package_col is None:
        raise ValueError(
            "Formal baseline validation roles require a package-group column"
        )
    group_by_sid: dict[str, str] = {}
    for sid, value in zip(sids, frame[package_col].tolist()):
        group = str(value or "").strip().lower()
        group_by_sid[sid] = (
            sid if group in {"", "nan", "none", "null"} else group
        )
    group_roles: dict[str, set[str]] = {}
    for role_name, identities in role_ids.items():
        for sid in identities:
            group_roles.setdefault(group_by_sid[sid], set()).add(role_name)
    crossed_groups = sorted(
        group for group, assigned_roles in group_roles.items()
        if len(assigned_roles) > 1
    )
    if crossed_groups:
        raise ValueError(
            "Validation role assignment splits package groups across roles; "
            f"examples={crossed_groups[:10]}"
        )

    model_selection_set = set(role_ids["model_selection"])
    model_selection_indices = [
        index for index, sid in enumerate(sids) if sid in model_selection_set
    ]
    decision_set = set(role_ids["decision_calibration"])
    decision_indices = [
        index for index, sid in enumerate(sids) if sid in decision_set
    ]
    size = len(frame)
    summary = {
        "schema_version": VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION,
        "protocol": VALIDATION_ROLE_ASSIGNMENT_PROTOCOL,
        "role_assignment_path": str(role_assignment),
        "role_assignment_sha256": _file_sha256(role_assignment),
        "validation_csv_path": str(validation_csv),
        "validation_csv_sha256": actual_csv_sha,
        "num_validation": size,
        "num_model_selection": len(model_selection_indices),
        "num_decision_calibration": len(decision_indices),
        "model_selection_fraction_of_validation": (
            len(model_selection_indices) / float(size)
        ),
        "decision_calibration_fraction_of_validation": (
            len(decision_indices) / float(size)
        ),
        "model_selection_indices": model_selection_indices,
        "decision_calibration_indices": decision_indices,
    }
    return model_selection_indices, summary


def read_label_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sha256" not in frame.columns:
        raise ValueError(f"{path} must contain a sha256 column")
    if "label" not in frame.columns:
        raise ValueError(f"{path} must contain a label column")
    frame = frame.copy()
    frame["sha256"] = frame["sha256"].astype(str).str.strip().str.lower()
    frame["label"] = _strict_binary_label_series(frame["label"], context=str(path))
    if bool((frame["sha256"] == "").any()):
        raise ValueError(f"{path} contains an empty sha256")
    if bool(frame["sha256"].duplicated().any()):
        duplicates = frame.loc[frame["sha256"].duplicated(), "sha256"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate sha256 values: {duplicates}")
    return frame


def pt_path_for(pt_dir: str | Path, sha256: str) -> Path:
    return Path(pt_dir) / f"{sha256}.pt"


def load_pt(pt_dir: str | Path, sha256: str) -> dict[str, Any]:
    path = pt_path_for(pt_dir, sha256)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
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
