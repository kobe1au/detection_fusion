#!/usr/bin/env python3
"""Repartition existing label CSVs with year/label stratification.

The input universe is exactly the union of labels/train.csv, labels/val.csv,
and labels/test.csv. Package groups are indivisible, so the same package name
cannot leak across output splits. Existing package assignments are retained
whenever possible, and single-row groups fill each year/label quota exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.6, 0.2, 0.2),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Back up and replace train.csv, val.csv, and test.csv.",
    )
    return parser.parse_args()


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_group(row: dict[str, str]) -> str:
    package = (row.get("pkg_name") or "").strip().casefold()
    if package:
        return f"pkg:{package}"
    return f"sha:{(row.get('sha256') or '').strip().lower()}"


def integer_targets(
    count: int,
    ratios: tuple[float, float, float],
    seed: int,
    cell: tuple[int, int],
) -> dict[str, int]:
    raw = [count * ratio for ratio in ratios]
    values = [math.floor(value) for value in raw]
    remainder = count - sum(values)
    order = sorted(
        range(len(SPLITS)),
        key=lambda index: (
            -(raw[index] - values[index]),
            stable_digest(f"{seed}:{cell[0]}:{cell[1]}:{SPLITS[index]}"),
        ),
    )
    for index in order[:remainder]:
        values[index] += 1
    return dict(zip(SPLITS, values))


def read_inputs(labels_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    seen_sha: set[str] = set()

    for split in SPLITS:
        path = labels_dir / f"{split}.csv"
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"CSV schema mismatch: {path}")
            for source_index, row in enumerate(reader, start=2):
                sha = (row.get("sha256") or "").strip().lower()
                if not sha:
                    raise ValueError(f"Missing sha256 at {path}:{source_index}")
                if sha in seen_sha:
                    raise ValueError(f"Duplicate sha256 across input CSVs: {sha}")
                seen_sha.add(sha)
                try:
                    year = int(row.get("year", ""))
                    label = int(row.get("label", ""))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid year/label at {path}:{source_index}"
                    ) from exc
                if label not in (0, 1):
                    raise ValueError(f"Unsupported label {label} for {sha}")
                row["_sha_key"] = sha
                row["_year_key"] = str(year)
                row["_label_key"] = str(label)
                row["_group_key"] = normalized_group(row)
                row["_old_split"] = split
                rows.append(row)

    if fieldnames is None or "split" not in fieldnames:
        raise ValueError("Input CSVs must contain a split column")
    return rows, fieldnames


def build_assignment(
    rows: list[dict[str, str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, str], dict[tuple[int, int], dict[str, int]]]:
    cell_totals: Counter[tuple[int, int]] = Counter(
        (int(row["_year_key"]), int(row["_label_key"])) for row in rows
    )
    targets = {
        cell: integer_targets(count, ratios, seed, cell)
        for cell, count in sorted(cell_totals.items())
    }

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["_group_key"]].append(row)

    current: dict[str, Counter[tuple[int, int]]] = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}
    multi_groups = [item for item in groups.items() if len(item[1]) > 1]
    multi_groups.sort(
        key=lambda item: (-len(item[1]), stable_digest(f"{seed}:{item[0]}"))
    )

    # Existing inputs are already package-disjoint. Keeping multi-row packages
    # in place minimizes PT movement while leaving abundant singleton groups to
    # satisfy the exact year/label quotas.
    for group, group_rows in multi_groups:
        old_splits = {row["_old_split"] for row in group_rows}
        if len(old_splits) != 1:
            raise ValueError(
                f"Existing split leaks package group {group!r}: {sorted(old_splits)}"
            )
        split = next(iter(old_splits))
        vector: Counter[tuple[int, int]] = Counter(
            (int(row["_year_key"]), int(row["_label_key"]))
            for row in group_rows
        )
        if any(
            current[split][cell] + amount > targets[cell][split]
            for cell, amount in vector.items()
        ):
            raise RuntimeError(
                f"Existing multi-row package groups exceed the new quota in {split}: {group}"
            )
        assignment[group] = split
        current[split].update(vector)

    singleton_by_cell: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group, group_rows in groups.items():
        if len(group_rows) != 1:
            continue
        row = group_rows[0]
        cell = (int(row["_year_key"]), int(row["_label_key"]))
        singleton_by_cell[cell].append(group)

    for cell in sorted(targets):
        groups_for_cell = singleton_by_cell[cell]
        deficits = {
            split: targets[cell][split] - current[split][cell]
            for split in SPLITS
        }
        if min(deficits.values()) < 0 or sum(deficits.values()) != len(groups_for_cell):
            raise RuntimeError(f"Invalid singleton deficits for cell {cell}: {deficits}")

        old_groups = {
            split: sorted(
                [
                    group
                    for group in groups_for_cell
                    if groups[group][0]["_old_split"] == split
                ],
                key=lambda group: stable_digest(f"{seed}:{cell}:keep:{group}"),
            )
            for split in SPLITS
        }
        surplus: list[str] = []
        for split in SPLITS:
            keep_count = min(len(old_groups[split]), deficits[split])
            for group in old_groups[split][:keep_count]:
                assignment[group] = split
            current[split][cell] += keep_count
            deficits[split] -= keep_count
            surplus.extend(old_groups[split][keep_count:])

        surplus.sort(key=lambda group: stable_digest(f"{seed}:{cell}:move:{group}"))
        cursor = 0
        for split in SPLITS:
            stop = cursor + deficits[split]
            for group in surplus[cursor:stop]:
                assignment[group] = split
            current[split][cell] += deficits[split]
            cursor = stop
        if cursor != len(surplus):
            raise RuntimeError(f"Unassigned singleton surplus for cell {cell}")

    if len(assignment) != len(groups):
        raise RuntimeError(
            f"Assigned {len(assignment)} of {len(groups)} package groups"
        )
    for cell, split_targets in targets.items():
        actual = {split: current[split][cell] for split in SPLITS}
        if actual != split_targets:
            raise RuntimeError(
                f"Year/label quota mismatch for {cell}: {actual} != {split_targets}"
            )
    return assignment, targets


def summarize(
    rows: list[dict[str, str]], assignment: dict[str, str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLITS:
        selected = [row for row in rows if assignment[row["_group_key"]] == split]
        summary[split] = {
            "count": len(selected),
            "labels": dict(
                sorted(Counter(int(row["_label_key"]) for row in selected).items())
            ),
            "years": dict(
                sorted(Counter(int(row["_year_key"]) for row in selected).items())
            ),
            "year_labels": {
                f"{year}:{label}": count
                for (year, label), count in sorted(
                    Counter(
                        (int(row["_year_key"]), int(row["_label_key"]))
                        for row in selected
                    ).items()
                )
            },
            "unique_packages": len({row["_group_key"] for row in selected}),
        }
    return summary


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    assignment: dict[str, str],
    split: str,
) -> None:
    selected = [row for row in rows if assignment[row["_group_key"]] == split]
    selected.sort(key=lambda row: row["_sha_key"])
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for source in selected:
            row = {field: source.get(field, "") for field in fieldnames}
            row["split"] = split
            writer.writerow(row)
    os.replace(temp_path, path)


def write_mapping(
    path: Path,
    rows: list[dict[str, str]],
    assignment: dict[str, str],
) -> None:
    fields = [
        "sha256",
        "pkg_name",
        "year",
        "label",
        "old_split",
        "new_split",
        "source_split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["_sha_key"]):
            writer.writerow(
                {
                    "sha256": row.get("sha256", ""),
                    "pkg_name": row.get("pkg_name", ""),
                    "year": row.get("year", ""),
                    "label": row.get("label", ""),
                    "old_split": row["_old_split"],
                    "new_split": assignment[row["_group_key"]],
                    "source_split": row.get("source_split", ""),
                }
            )


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir).resolve()
    ratios = tuple(float(value) for value in args.ratios)
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError(f"Ratios must be positive and sum to 1, got {ratios}")

    rows, fieldnames = read_inputs(labels_dir)
    assignment, targets = build_assignment(rows, ratios, args.seed)
    summary = summarize(rows, assignment)
    old_new = Counter(
        (row["_old_split"], assignment[row["_group_key"]]) for row in rows
    )
    moved = sum(
        int(row["_old_split"] != assignment[row["_group_key"]]) for row in rows
    )
    metadata: dict[str, Any] = {
        "protocol": "year-label stratified, package-group disjoint",
        "ratios": dict(zip(SPLITS, ratios)),
        "seed": args.seed,
        "total_samples": len(rows),
        "moved_samples": moved,
        "summary": summary,
        "targets": {
            f"{year}:{label}": split_targets
            for (year, label), split_targets in sorted(targets.items())
        },
        "old_to_new": {
            f"{old}->{new}": count
            for (old, new), count in sorted(old_new.items())
        },
    }

    if args.write:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = labels_dir / f"backup_before_split_6_2_2_{stamp}"
        backup_dir.mkdir(parents=False, exist_ok=False)
        for split in SPLITS:
            shutil.copy2(labels_dir / f"{split}.csv", backup_dir / f"{split}.csv")
        metadata["backup_dir"] = str(backup_dir)

        for split in SPLITS:
            write_csv(
                labels_dir / f"{split}.csv",
                rows,
                fieldnames,
                assignment,
                split,
            )
        write_mapping(labels_dir / "split_reassignment_6_2_2.csv", rows, assignment)
        with (labels_dir / "split_metadata_6_2_2.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not args.write:
        print("Dry run only. Re-run with --write to replace the split CSVs.")


if __name__ == "__main__":
    main()
