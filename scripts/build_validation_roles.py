from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.dataset import build_package_isolation_groups
from fusion.train import (
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
        default=0.25,
    )
    parser.add_argument(
        "--source-v1",
        type=Path,
        default=None,
        help=(
            "Non-cherry-picked migration: union the v1 checkpoint/posthoc roles "
            "and retain its untouched decision role."
        ),
    )
    args = parser.parse_args()

    csv_path = args.validation_csv.resolve()
    dataset = ValidationMetadata(csv_path)
    migration_source = None
    if args.source_v1 is not None:
        source_path = args.source_v1.resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if str(source.get("validation_csv_sha256") or "") != _sha256(csv_path):
            raise ValueError("source-v1 was built for a different validation CSV")
        source_roles = source.get("roles") or {}
        expected = {
            "checkpoint_selection",
            "posthoc_calibration",
            "decision_calibration",
        }
        if set(source_roles) != expected:
            raise ValueError(
                f"source-v1 must contain exactly roles {sorted(expected)}"
            )
        index_by_sid = {
            sid: index for index, sid in enumerate(dataset.sample_sids)
        }
        model_ids = [
            *source_roles["checkpoint_selection"],
            *source_roles["posthoc_calibration"],
        ]
        decision_ids = list(source_roles["decision_calibration"])
        if (
            len(model_ids) != len(set(model_ids))
            or set(model_ids) & set(decision_ids)
            or set(model_ids) | set(decision_ids) != set(dataset.sample_sids)
        ):
            raise ValueError("source-v1 roles do not form a complete partition")
        role_indices = {
            "model_selection": sorted(index_by_sid[sid] for sid in model_ids),
            "decision_calibration": sorted(
                index_by_sid[sid] for sid in decision_ids
            ),
        }
        outer = {
            "migration": "union_v1_checkpoint_and_posthoc_keep_decision",
            "source_path": args.source_v1.as_posix(),
            "source_sha256": _sha256(source_path),
        }
        migration_source = {
            "path": args.source_v1.as_posix(),
            "sha256": _sha256(source_path),
        }
    else:
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
    payload = {
        "schema_version": VALIDATION_ROLE_ASSIGNMENT_SCHEMA_VERSION,
        "protocol": "year_label_stratified_package_group_75_25_v2",
        "validation_csv": args.validation_csv.as_posix(),
        "validation_csv_sha256": _sha256(csv_path),
        "split_seed": int(args.seed),
        "decision_fraction": (
            len(role_indices["decision_calibration"]) / float(len(dataset))
        ),
        "migration_source": migration_source,
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
