from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalized_fieldnames(fieldnames: list[str] | None) -> dict[str, str]:
    return {
        str(name).strip().lstrip("\ufeff").strip('"').strip("'").lower(): name
        for name in (fieldnames or [])
    }


def read_csv_ids(csv_path: Path, sha_column: str) -> set[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = _normalized_fieldnames(reader.fieldnames)
        requested = sha_column.strip().lower()
        if requested not in fields:
            raise ValueError(
                f"{csv_path} does not contain column {sha_column!r}; "
                f"available columns: {sorted(fields)}"
            )

        source_column = fields[requested]
        ids: set[str] = set()
        duplicates: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            sha = str(row.get(source_column) or "").strip().lower()
            if not SHA256_RE.fullmatch(sha):
                raise ValueError(
                    f"Invalid SHA-256 at {csv_path}:{line_number}: {sha!r}"
                )
            if sha in ids:
                duplicates.add(sha)
            ids.add(sha)

    if duplicates:
        examples = sorted(duplicates)[:10]
        raise ValueError(
            f"CSV contains {len(duplicates)} duplicate SHA-256 values; examples={examples}"
        )
    return ids


def list_pt_files(pt_dir: Path, extension: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(pt_dir.glob(f"*{extension}")):
        if not path.is_file():
            continue
        sha = path.stem.strip().lower()
        if not SHA256_RE.fullmatch(sha):
            print(f"WARNING: ignoring non-SHA filename: {path.name}")
            continue
        if sha in files:
            raise ValueError(f"Duplicate PT basename ignoring case: {path.name}")
        files[sha] = path
    return files


def write_id_report(path: Path, ids: list[str], column: str = "sha256") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column])
        writer.writerows((value,) for value in ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a label CSV with a PT directory, print CSV samples missing PTs, "
            "and optionally move PTs absent from the CSV into a quarantine directory."
        )
    )
    parser.add_argument("--csv", required=True, help="Label CSV used as the source of truth.")
    parser.add_argument("--pt-dir", required=True, help="Directory containing SHA256.pt files.")
    parser.add_argument(
        "--quarantine-dir",
        required=True,
        help="Destination directory for PT files not present in the CSV.",
    )
    parser.add_argument("--sha-column", default="sha256", help="CSV SHA-256 column name.")
    parser.add_argument("--extension", default=".pt", help="PT filename extension.")
    parser.add_argument(
        "--move-extra",
        action="store_true",
        help="Actually move extra PT files. Without this flag, only report planned actions.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Optional report directory. Defaults to the quarantine directory.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    pt_dir = Path(args.pt_dir).expanduser().resolve()
    quarantine_dir = Path(args.quarantine_dir).expanduser().resolve()
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else quarantine_dir
    )
    extension = str(args.extension)
    if not extension.startswith("."):
        extension = f".{extension}"

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    if not pt_dir.is_dir():
        raise NotADirectoryError(f"PT directory does not exist: {pt_dir}")
    if quarantine_dir == pt_dir:
        raise ValueError("Quarantine directory must differ from the PT directory")
    if pt_dir in quarantine_dir.parents:
        raise ValueError("Quarantine directory must not be inside the PT directory")

    csv_ids = read_csv_ids(csv_path, args.sha_column)
    pt_files = list_pt_files(pt_dir, extension)
    pt_ids = set(pt_files)
    missing = sorted(csv_ids - pt_ids)
    extra = sorted(pt_ids - csv_ids)

    print(f"CSV samples:       {len(csv_ids)}")
    print(f"PT files:          {len(pt_files)}")
    print(f"CSV missing PT:    {len(missing)}")
    print(f"PT absent in CSV:  {len(extra)}")

    if missing:
        print("\nCSV samples missing from the PT directory:")
        for sha in missing:
            print(sha)
    else:
        print("\nNo CSV samples are missing PT files.")

    write_id_report(report_dir / "csv_missing_pt.csv", missing)
    write_id_report(report_dir / "pt_absent_in_csv.csv", extra)

    if not args.move_extra:
        if extra:
            print(
                f"\nDry run: {len(extra)} extra PT files would be moved to "
                f"{quarantine_dir}"
            )
            print("Run again with --move-extra to perform the move.")
        return

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination_conflicts = [
        quarantine_dir / pt_files[sha].name
        for sha in extra
        if (quarantine_dir / pt_files[sha].name).exists()
    ]
    if destination_conflicts:
        examples = [str(path) for path in destination_conflicts[:10]]
        raise FileExistsError(
            f"Quarantine contains {len(destination_conflicts)} conflicting files; "
            f"examples={examples}"
        )

    moved_rows: list[tuple[str, str, str]] = []
    for sha in extra:
        source = pt_files[sha]
        destination = quarantine_dir / source.name
        shutil.move(str(source), str(destination))
        moved_rows.append((sha, str(source), str(destination)))

    moved_manifest = report_dir / "moved_pt_files.csv"
    moved_manifest.parent.mkdir(parents=True, exist_ok=True)
    with moved_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "source", "destination"])
        writer.writerows(moved_rows)

    remaining_ids = set(list_pt_files(pt_dir, extension))
    missing_after = sorted(csv_ids - remaining_ids)
    extra_after = sorted(remaining_ids - csv_ids)
    print(f"\nMoved PT files:    {len(moved_rows)}")
    print(f"Missing after move:{len(missing_after)}")
    print(f"Extra after move:  {len(extra_after)}")
    print(f"Move manifest:     {moved_manifest}")
    if extra_after:
        raise RuntimeError("Extra PT files remain after the move")


if __name__ == "__main__":
    main()
