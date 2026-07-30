"""Shared deterministic identities for controlled degradation views."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np


CONTROLLED_TEST_VIEW_PROTOCOL_SEED = 424242
CONTROLLED_VIEW_SEED_FORMULA = "H(SID, mechanism, protocol_seed)"
CONTROLLED_VIEW_MECHANISM_VERSION = "controlled_test_view_v1"


def deterministic_view_seed(
    sid: str,
    mechanism: str,
    protocol_seed: int,
) -> int:
    """Return the frozen process-independent 31-bit view seed."""

    sid = str(sid)
    mechanism = str(mechanism)
    if not sid or not mechanism:
        raise ValueError("sid and mechanism must be non-empty")
    if isinstance(protocol_seed, bool) or not isinstance(
        protocol_seed, (int, np.integer)
    ):
        raise TypeError("protocol_seed must be an integer")
    payload = json.dumps(
        [sid, mechanism, int(protocol_seed)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        2**31 - 1
    )


def deterministic_view_spec(
    sid: str,
    mechanism: str,
    protocol_seed: int,
    strength_min: float,
    strength_max: float,
) -> dict[str, int | float]:
    """Bind one SID/mechanism identity to a seed and deterministic strength."""

    minimum = float(strength_min)
    maximum = float(strength_max)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or not 0.0 <= minimum <= maximum <= 1.0
    ):
        raise ValueError(
            "view strengths must satisfy 0 <= min <= max <= 1"
        )
    sid = str(sid)
    mechanism = str(mechanism)
    seed = deterministic_view_seed(sid, mechanism, protocol_seed)
    payload = json.dumps(
        [sid, mechanism, int(protocol_seed)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    unit_interval = int.from_bytes(digest[8:16], "big") / float(
        2**64 - 1
    )
    return {
        "seed": int(seed),
        "strength": float(
            minimum + (maximum - minimum) * unit_interval
        ),
    }


def fixed_test_view_plan(
    sample_sids: Sequence[str],
    *,
    mechanism: str,
    strength: float,
    protocol_seed: int = CONTROLLED_TEST_VIEW_PROTOCOL_SEED,
) -> tuple[tuple[tuple[str, float, int], ...], tuple[dict[str, Any], ...]]:
    """Build the canonical per-sample fixed-strength test plan."""

    mechanism = str(mechanism).strip().lower()
    if not mechanism:
        raise ValueError("mechanism must be non-empty")
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie within [0, 1]")
    records: list[dict[str, Any]] = []
    plan: list[tuple[str, float, int]] = []
    for raw_sid in sample_sids:
        sid = str(raw_sid).strip().lower()
        if not sid:
            raise ValueError("sample_sids must contain non-empty identities")
        spec = deterministic_view_spec(
            sid,
            mechanism,
            protocol_seed,
            strength,
            strength,
        )
        if mechanism != "clean":
            plan.append((mechanism, strength, int(spec["seed"])))
        records.append(
            {
                "sid": sid,
                "view_name": mechanism,
                "mechanism": mechanism,
                "mechanism_version": (
                    CONTROLLED_VIEW_MECHANISM_VERSION
                ),
                "sampled_strength": strength,
                "view_seed": int(spec["seed"]),
            }
        )
    return tuple(plan), tuple(records)


def canonical_manifest_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_manifest_sha256(records: Sequence[dict[str, Any]]) -> str:
    rows = [
        {
            "sid": str(row["sid"]),
            "view_name": str(row["view_name"]),
            "sampled_strength": float(row["sampled_strength"]),
            "view_seed": int(row["view_seed"]),
        }
        for row in records
    ]
    return canonical_manifest_sha256(rows)


__all__ = [
    "CONTROLLED_TEST_VIEW_PROTOCOL_SEED",
    "CONTROLLED_VIEW_MECHANISM_VERSION",
    "CONTROLLED_VIEW_SEED_FORMULA",
    "canonical_manifest_sha256",
    "deterministic_view_seed",
    "deterministic_view_spec",
    "fixed_test_view_plan",
    "seed_manifest_sha256",
]
