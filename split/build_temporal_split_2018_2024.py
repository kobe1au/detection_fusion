#!/usr/bin/env python3
"""Build a 2018-2021 / 2022 / 2023-2024 chronological split.

The script keeps compatible seed rows from existing train/val/test CSVs, then
streams the full AndroZoo metadata CSV to fill missing year-class quotas.
It intentionally avoids package-level blocking by default because package
blocking can remove temporally valid updates and make this split too small.
Package overlaps are reported in metadata for transparency.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict


TRAIN_YEARS = (2018, 2019, 2020, 2021)
VAL_YEARS = (2022,)
TEST_YEARS = (2023, 2024)
FIELDNAMES = ["sha256", "label", "year", "pkg_name", "market"]


@dataclass(frozen=True)
class Sample:
    sha256: str
    label: int
    year: int
    pkg_name: str
    market: str
    source: str

    def to_row(self) -> dict[str, str | int]:
        return {
            "sha256": self.sha256,
            "label": self.label,
            "year": self.year,
            "pkg_name": self.pkg_name,
            "market": self.market,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-csv", default="resouce/latest_with-added-date.csv")
    parser.add_argument("--seed-dir", default="resouce")
    parser.add_argument("--out-dir", default="resource/dataset_split_2018_2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--malware-min", type=int, default=4)
    parser.add_argument("--benign-max", type=int, default=0)
    parser.add_argument("--min-apk-size", type=int, default=50000)
    parser.add_argument("--max-apk-size", type=int, default=100_000_000)
    parser.add_argument(
        "--buffer-factor",
        type=int,
        default=3,
        help="Reservoir buffer multiplier for each missing quota.",
    )
    return parser.parse_args()


def split_for_year(year: int) -> str | None:
    if year in TRAIN_YEARS:
        return "train"
    if year in VAL_YEARS:
        return "val"
    if year in TEST_YEARS:
        return "test"
    return None


def build_quotas(train_size: int, val_size: int, test_size: int) -> dict[str, dict[tuple[int, int], int]]:
    def per_year_label(total: int, years: tuple[int, ...]) -> dict[tuple[int, int], int]:
        slots = [(year, label) for year in years for label in (0, 1)]
        base = total // len(slots)
        rem = total - base * len(slots)
        quotas = {slot: base for slot in slots}
        for slot in slots[:rem]:
            quotas[slot] += 1
        return quotas

    return {
        "train": per_year_label(train_size, TRAIN_YEARS),
        "val": per_year_label(val_size, VAL_YEARS),
        "test": per_year_label(test_size, TEST_YEARS),
    }


def normalize_sha(raw: str) -> str:
    return (raw or "").strip().lower()


def normalize_pkg(raw: str, sha: str) -> str:
    pkg = (raw or "").strip()
    return pkg if pkg else f"__unknown_{sha}"


def normalize_market(raw: str) -> str:
    return "play" if "play" in (raw or "").lower() else "other"


def read_seed_samples(seed_dir: Path) -> tuple[DefaultDict[tuple[str, int, int], list[Sample]], set[str]]:
    groups: DefaultDict[tuple[str, int, int], list[Sample]] = defaultdict(list)
    all_seed_shas: set[str] = set()
    seen: set[str] = set()

    for name in ("train", "val", "test"):
        path = seed_dir / f"{name}.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sha = normalize_sha(row.get("sha256", ""))
                if not sha:
                    continue
                all_seed_shas.add(sha)
                if sha in seen:
                    continue
                seen.add(sha)
                try:
                    year = int(row.get("year", ""))
                    label = int(row.get("label", ""))
                except ValueError:
                    continue
                target = split_for_year(year)
                if target is None or label not in (0, 1):
                    continue
                sample = Sample(
                    sha256=sha,
                    label=label,
                    year=year,
                    pkg_name=normalize_pkg(row.get("pkg_name", ""), sha),
                    market=normalize_market(row.get("market", "")),
                    source=f"seed:{name}",
                )
                groups[(target, year, label)].append(sample)

    return groups, all_seed_shas


def select_seed_anchors(
    seed_groups: dict[tuple[str, int, int], list[Sample]],
    quotas: dict[str, dict[tuple[int, int], int]],
    rng: random.Random,
) -> tuple[DefaultDict[str, list[Sample]], dict[str, dict[str, int]]]:
    selected: DefaultDict[str, list[Sample]] = defaultdict(list)
    stats: dict[str, dict[str, int]] = defaultdict(dict)

    for split, split_quotas in quotas.items():
        for (year, label), quota in split_quotas.items():
            candidates = list(seed_groups.get((split, year, label), []))
            rng.shuffle(candidates)
            keep = candidates[:quota]
            selected[split].extend(keep)
            key = f"{year}:{label}"
            stats[split][key] = len(keep)

    return selected, stats


def parse_year_from_added(raw: str) -> int | None:
    raw = (raw or "").strip()
    if len(raw) < 4:
        return None
    try:
        return int(raw[:4])
    except ValueError:
        return None


def parse_label(vt_raw: str, benign_max: int, malware_min: int) -> int | None:
    try:
        vt = int(vt_raw)
    except (TypeError, ValueError):
        return None
    if vt <= benign_max:
        return 0
    if vt >= malware_min:
        return 1
    return None


def stream_supplements(
    full_csv: Path,
    quotas: dict[str, dict[tuple[int, int], int]],
    selected: dict[str, list[Sample]],
    all_seed_shas: set[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[DefaultDict[tuple[str, int, int], list[Sample]], dict[str, dict[str, int]], int]:
    selected_counts = Counter(
        (split, sample.year, sample.label)
        for split, samples in selected.items()
        for sample in samples
    )
    needs: dict[tuple[str, int, int], int] = {}
    for split, split_quotas in quotas.items():
        for (year, label), quota in split_quotas.items():
            need = quota - selected_counts[(split, year, label)]
            if need > 0:
                needs[(split, year, label)] = need

    buffers = {
        key: max(need * args.buffer_factor, need + 100)
        for key, need in needs.items()
    }
    reservoirs: DefaultDict[tuple[str, int, int], list[Sample]] = defaultdict(list)
    seen_by_group: Counter[tuple[str, int, int]] = Counter()
    progress = 0

    with full_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            progress += 1
            if progress % 2_000_000 == 0:
                print(f"  scanned {progress // 1_000_000}M rows")

            year = parse_year_from_added(row.get("added", ""))
            if year is None:
                continue
            split = split_for_year(year)
            if split is None:
                continue
            label = parse_label(row.get("vt_detection", ""), args.benign_max, args.malware_min)
            if label is None:
                continue
            key = (split, year, label)
            if key not in needs:
                continue

            sha = normalize_sha(row.get("sha256", ""))
            if not sha or sha in all_seed_shas:
                continue

            try:
                apk_size = int(row.get("apk_size", "0"))
            except ValueError:
                continue
            if apk_size < args.min_apk_size or apk_size > args.max_apk_size:
                continue

            sample = Sample(
                sha256=sha,
                label=label,
                year=year,
                pkg_name=normalize_pkg(row.get("pkg_name", ""), sha),
                market=normalize_market(row.get("markets", "")),
                source="full_csv",
            )

            seen_by_group[key] += 1
            reservoir = reservoirs[key]
            limit = buffers[key]
            n_seen = seen_by_group[key]
            if len(reservoir) < limit:
                reservoir.append(sample)
            else:
                j = rng.randrange(n_seen)
                if j < limit:
                    reservoir[j] = sample

    pool_stats: dict[str, dict[str, int]] = defaultdict(dict)
    for key, seen_count in sorted(seen_by_group.items()):
        split, year, label = key
        pool_stats[split][f"{year}:{label}"] = seen_count

    return reservoirs, pool_stats, progress


def fill_from_reservoirs(
    selected: DefaultDict[str, list[Sample]],
    quotas: dict[str, dict[tuple[int, int], int]],
    reservoirs: dict[tuple[str, int, int], list[Sample]],
    rng: random.Random,
) -> tuple[DefaultDict[str, list[Sample]], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    used_shas = {s.sha256 for samples in selected.values() for s in samples}
    supplement_stats: dict[str, dict[str, int]] = defaultdict(dict)
    final_stats: dict[str, dict[str, int]] = defaultdict(dict)

    for split, split_quotas in quotas.items():
        for (year, label), quota in split_quotas.items():
            current = [s for s in selected[split] if s.year == year and s.label == label]
            need = quota - len(current)
            added = 0
            if need > 0:
                candidates = list(reservoirs.get((split, year, label), []))
                rng.shuffle(candidates)
                for sample in candidates:
                    if added >= need:
                        break
                    if sample.sha256 in used_shas:
                        continue
                    used_shas.add(sample.sha256)
                    selected[split].append(sample)
                    added += 1
            key = f"{year}:{label}"
            supplement_stats[split][key] = added
            final_stats[split][key] = sum(
                1 for s in selected[split] if s.year == year and s.label == label
            )

        rng.shuffle(selected[split])

    return selected, supplement_stats, final_stats


def write_split(out_dir: Path, name: str, samples: list[Sample]) -> None:
    path = out_dir / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_row())


def summarize(samples_by_split: dict[str, list[Sample]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split, samples in samples_by_split.items():
        years = Counter(s.year for s in samples)
        labels = Counter(s.label for s in samples)
        year_labels = Counter(f"{s.year}:{s.label}" for s in samples)
        sources = Counter(s.source for s in samples)
        summary[split] = {
            "n": len(samples),
            "years": dict(sorted(years.items())),
            "labels": dict(sorted(labels.items())),
            "year_labels": dict(sorted(year_labels.items())),
            "sources": dict(sorted(sources.items())),
            "unique_sha": len({s.sha256 for s in samples}),
            "unique_pkg": len({s.pkg_name for s in samples}),
        }
    return summary


def overlap_report(samples_by_split: dict[str, list[Sample]]) -> dict[str, object]:
    report: dict[str, object] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sha_a = {s.sha256 for s in samples_by_split[a]}
        sha_b = {s.sha256 for s in samples_by_split[b]}
        pkg_a = {s.pkg_name for s in samples_by_split[a] if not s.pkg_name.startswith("__unknown_")}
        pkg_b = {s.pkg_name for s in samples_by_split[b] if not s.pkg_name.startswith("__unknown_")}
        report[f"{a}_{b}"] = {
            "sha_overlap": len(sha_a & sha_b),
            "pkg_overlap": len(pkg_a & pkg_b),
        }
    return report


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    full_csv = Path(args.full_csv)
    seed_dir = Path(args.seed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quotas = build_quotas(args.train_size, args.val_size, args.test_size)
    print("[1/4] Reading seed splits...")
    seed_groups, all_seed_shas = read_seed_samples(seed_dir)

    print("[2/4] Selecting compatible seed anchors...")
    selected, seed_anchor_stats = select_seed_anchors(seed_groups, quotas, rng)

    print("[3/4] Streaming full CSV for supplements...")
    reservoirs, pool_stats, scanned_rows = stream_supplements(
        full_csv=full_csv,
        quotas=quotas,
        selected=selected,
        all_seed_shas=all_seed_shas,
        args=args,
        rng=rng,
    )

    print("[4/4] Filling quotas and writing files...")
    selected, supplement_stats, final_stats = fill_from_reservoirs(
        selected=selected,
        quotas=quotas,
        reservoirs=reservoirs,
        rng=rng,
    )

    samples_by_split = {name: list(selected.get(name, [])) for name in ("train", "val", "test")}
    for name, samples in samples_by_split.items():
        write_split(out_dir, name, samples)
    for year in TEST_YEARS:
        write_split(out_dir, f"test_{year}", [s for s in samples_by_split["test"] if s.year == year])

    metadata = {
        "protocol": "train=2018-2021,val=2022,test=2023-2024",
        "full_csv": str(full_csv),
        "seed_dir": str(seed_dir),
        "seed": args.seed,
        "vt": {"benign_max": args.benign_max, "malware_min": args.malware_min},
        "apk_size": {"min": args.min_apk_size, "max": args.max_apk_size},
        "quotas": {
            split: {f"{year}:{label}": quota for (year, label), quota in split_quotas.items()}
            for split, split_quotas in quotas.items()
        },
        "seed_anchor_stats": seed_anchor_stats,
        "supplement_stats": supplement_stats,
        "full_csv_pool_seen_after_seed_exclusion": pool_stats,
        "final_year_label_stats": final_stats,
        "summary": summarize(samples_by_split),
        "overlap": overlap_report(samples_by_split),
        "scanned_rows": scanned_rows,
        "samples": {split: [asdict(s) for s in samples] for split, samples in samples_by_split.items()},
    }
    with (out_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(metadata["overlap"], ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
