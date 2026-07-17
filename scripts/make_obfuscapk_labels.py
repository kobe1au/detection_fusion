from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch


SPLITS = {
    "nop": "nop",
    "goto": "goto",
    "method_rename": "method_rename",
    "string": "string",
    "combined": "combined",
    "advanced_reflection": "advanced_reflection",
    "call_indirection": "call_indirection",
    "mixed_api_graph_manifest": "mixed_api_graph_manifest",
}

SHA_RE = re.compile(r"^([0-9a-fA-F]{64})_")


def read_source_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {
            str(name).strip().lstrip("\ufeff").strip('"').strip("'"): name
            for name in (reader.fieldnames or [])
        }
        if "sha256" not in fieldnames or "label" not in fieldnames:
            raise ValueError(f"{path} must contain sha256,label columns")
        sha_col = fieldnames["sha256"]
        label_col = fieldnames["label"]
        for row in reader:
            sha = str(row[sha_col]).strip().lower()
            labels[sha] = int(str(row[label_col]).strip())
    return labels


def original_sha_from_apk_name(apk_name: str) -> str | None:
    match = SHA_RE.match(str(apk_name or ""))
    return match.group(1).lower() if match else None


def write_labels_for_split(split: str, pt_dir: Path, out_csv: Path, source_labels: dict[str, int]) -> tuple[int, int]:
    rows: list[tuple[str, int]] = []
    missing = 0
    for pt_path in sorted(pt_dir.glob("*.pt")):
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        meta = payload.get("direct_build_meta") or {}
        apk_name = str(meta.get("apk_name") or "")
        original_sha = original_sha_from_apk_name(apk_name)
        label = source_labels.get(original_sha or "")
        if label is None:
            missing += 1
            continue
        rows.append((pt_path.stem.lower(), int(label)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "label"])
        writer.writerows(rows)
    return len(rows), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labels/obfuscapk_*.csv from aligned Obfuscapk PTs.")
    parser.add_argument("--pt-root", default="D:/pts_obfuscapk")
    parser.add_argument("--source-csv", default="obfuscapk_logits/rebuild_success_1000.csv")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split names to label. Defaults to all known Obfuscapk splits.",
    )
    args = parser.parse_args()

    pt_root = Path(args.pt_root)
    source_csv = Path(args.source_csv)
    labels_dir = Path(args.labels_dir)
    source_labels = read_source_labels(source_csv)

    total_rows = 0
    total_missing = 0
    split_names = list(args.splits) if args.splits else list(SPLITS)
    unknown = [name for name in split_names if name not in SPLITS]
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}. Known: {sorted(SPLITS)}")

    for name in split_names:
        dirname = SPLITS[name]
        pt_dir = pt_root / dirname
        if not pt_dir.is_dir():
            raise FileNotFoundError(f"Missing PT directory for {name}: {pt_dir}")
        out_csv = labels_dir / f"obfuscapk_{name}.csv"
        rows, missing = write_labels_for_split(name, pt_dir, out_csv, source_labels)
        total_rows += rows
        total_missing += missing
        print(f"{name}: wrote {rows} rows to {out_csv}; missing_source_label={missing}")

    if total_missing:
        raise RuntimeError(f"Missing source labels for {total_missing} PT files")
    print(f"done: wrote {total_rows} labels")


if __name__ == "__main__":
    main()
