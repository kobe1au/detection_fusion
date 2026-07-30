from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.dataset import build_package_isolation_groups
from fusion.runtime import (
    VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION,
    split_validation_dataset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ValidationMetadata:
    """PT-free view matching RobustTriModalDataset's sorted CSV metadata."""

    def __init__(self, csv_path: Path) -> None:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Validation CSV is empty: {csv_path}")
        id_field = next(
            (name for name in ("id", "ID", "Id", "sha256") if name in rows[0]),
            None,
        )
        if id_field is None or "label" not in rows[0]:
            raise ValueError("Validation CSV requires id/sha256 and label")
        year_field = next(
            (name for name in ("year", "Year", "vt_year", "dex_year") if name in rows[0]),
            None,
        )
        package_field = next(
            (name for name in ("pkg_name", "package_name", "package") if name in rows[0]),
            None,
        )
        normalized: dict[str, tuple[int, int, str]] = {}
        for row in rows:
            sid = str(row[id_field]).strip().lower()
            if not sid or sid in normalized:
                raise ValueError(f"Empty or duplicate validation identity: {sid!r}")
            label = int(row["label"])
            if label not in {0, 1}:
                raise ValueError(f"Non-binary validation label for {sid}: {label}")
            year = int(row[year_field]) if year_field else 0
            package = str(row.get(package_field, "") or "").strip().lower()
            normalized[sid] = (label, year, package)
        self.sample_sids = sorted(normalized)
        self.sample_labels = [normalized[sid][0] for sid in self.sample_sids]
        self.sample_years = [normalized[sid][1] for sid in self.sample_sids]
        packages = {sid: normalized[sid][2] for sid in self.sample_sids}
        self.sample_groups = build_package_isolation_groups(
            self.sample_sids, packages
        )

    def __len__(self) -> int:
        return len(self.sample_sids)

    def __getitem__(self, index: int) -> int:
        return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the two-role model-selection/I3 validation protocol."
    )
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decision-fraction",
        type=float,
        default=1.0 / 3.0,
    )
    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "Protocol identifier written to the assignment. By default it is "
            "derived from the requested model-selection:decision ratio."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < float(args.decision_fraction) < 1.0:
        raise ValueError("--decision-fraction must be within (0, 1)")

    csv_path = args.validation_csv.resolve()
    dataset = ValidationMetadata(csv_path)
    cfg = {
        "train": {"seed": int(args.seed)},
        "calibration": {
            "validation_fraction": float(args.decision_fraction),
            "split_seed": int(args.seed),
            "stratified_group_split": True,
        },
    }
    selection, decision, outer = split_validation_dataset(cfg, dataset)
    role_indices = {
        "model_selection": list(selection.indices),
        "decision_calibration": list(decision.indices),
    }
    flattened = [index for values in role_indices.values() for index in values]
    if len(flattened) != len(dataset) or set(flattened) != set(range(len(dataset))):
        raise RuntimeError("Generated validation roles do not form a partition")
    decision_ratio = Fraction(float(args.decision_fraction)).limit_denominator(1000)
    model_parts = decision_ratio.denominator - decision_ratio.numerator
    decision_parts = decision_ratio.numerator
    protocol = str(args.protocol or "").strip() or (
        "year_label_stratified_package_group_"
        f"{model_parts}to{decision_parts}_v3"
    )
    payload = {
        "schema_version": VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION,
        "protocol": protocol,
        "validation_csv": args.validation_csv.as_posix(),
        "validation_csv_sha256": _sha256(csv_path),
        "split_seed": int(args.seed),
        "decision_fraction": (
            len(role_indices["decision_calibration"]) / float(len(dataset))
        ),
        "migration_source": None,
        "counts": {
            name: len(indices) for name, indices in role_indices.items()
        },
        "roles": {
            name: [dataset.sample_sids[index] for index in indices]
            for name, indices in role_indices.items()
        },
        "generator_summary": {
            key: value
            for key, value in outer.items()
            if not key.endswith("_indices")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "counts": payload["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
