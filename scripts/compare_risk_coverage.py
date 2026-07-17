from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.train import build_risk_coverage_curve


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=CSV_PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise argparse.ArgumentTypeError("Comparison name must not be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Diagnostics CSV not found: {path}")
    return name, path


def _read_rows(path: Path, split: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [row for row in rows if str(row.get("split") or "") == split]
    if not selected:
        available = sorted({str(row.get("split") or "") for row in rows})
        raise ValueError(f"Split {split!r} not found in {path}; available={available}")
    return selected


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty comparison: {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare selective risk at matched coverage from gate diagnostics."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=_named_path,
        metavar="NAME=CSV_PATH",
        help="Repeat for every method or version to compare.",
    )
    parser.add_argument("--split", default="test_clean")
    parser.add_argument(
        "--coverages",
        default="0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated target acceptance rates.",
    )
    parser.add_argument("--out-dir", default="results/risk_coverage_comparison")
    args = parser.parse_args()

    targets = [float(value) for value in args.coverages.split(",") if value.strip()]
    if not targets or any(not 0.0 < value <= 1.0 for value in targets):
        raise ValueError("Every target coverage must be within (0, 1]")

    combined: list[dict] = []
    curves_by_name: dict[str, list[dict]] = {}
    for name, path in args.input:
        curve = build_risk_coverage_curve(_read_rows(path, args.split))
        curve = [row for row in curve if row["split"] == args.split]
        if not curve:
            raise ValueError(f"No valid risk-coverage points produced for {name}")
        curves_by_name[name] = curve
        combined.extend({"method": name, **row} for row in curve)

    matched: list[dict] = []
    for target in targets:
        for name, curve in curves_by_name.items():
            point = min(
                curve,
                key=lambda row: (
                    abs(float(row["coverage"]) - target),
                    -float(row["coverage"]),
                ),
            )
            matched.append(
                {
                    "target_coverage": target,
                    "method": name,
                    **point,
                }
            )

    out_dir = Path(args.out_dir)
    curve_path = out_dir / "risk_coverage_curves.csv"
    matched_path = out_dir / "risk_coverage_matched.csv"
    _write_csv(curve_path, combined)
    _write_csv(matched_path, matched)
    print(f"Wrote {curve_path}")
    print(f"Wrote {matched_path}")


if __name__ == "__main__":
    main()
