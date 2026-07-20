from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_MAX_NODES = 12288


def _tensor_rows(value: Any) -> int:
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        return 0
    return int(value.shape[0])


def graph_size_from_payload(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("PT payload must be a dictionary")
    dex_list = payload.get("dex_list")
    if not isinstance(dex_list, list):
        raise ValueError("PT payload is missing the current-schema dex_list")

    real_nodes = 0
    encoder_nodes = 0
    valid_dex_count = 0
    for dex in dex_list:
        if not isinstance(dex, dict):
            continue
        valid_dex_count += 1
        nodes = _tensor_rows(dex.get("call_x"))
        real_nodes += nodes
        # RobustTriModalDataset inserts one zero-feature ghost node for each
        # empty DEX so the graph encoder still receives a valid graph.
        encoder_nodes += max(nodes, 1)

    observable = payload.get("observable_metadata")
    raw_nodes = -1
    if isinstance(observable, dict):
        try:
            raw_nodes = int(observable.get("graph_node_count_raw", -1))
        except (TypeError, ValueError):
            raw_nodes = -1

    return {
        "dex_count": valid_dex_count,
        "real_nodes": real_nodes,
        "encoder_nodes": encoder_nodes,
        "raw_metadata_nodes": raw_nodes,
    }


def inspect_pt(path: Path, max_nodes: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sizes = graph_size_from_payload(payload)
    encoder_nodes = sizes["encoder_nodes"]
    truncated_nodes = max(encoder_nodes - max_nodes, 0)
    retained_ratio = 1.0 if encoder_nodes <= 0 else min(max_nodes / encoder_nodes, 1.0)
    return {
        "sid": path.stem,
        "path": str(path.resolve()),
        **sizes,
        "exceeds_max_nodes": int(encoder_nodes > max_nodes),
        "truncated_nodes": truncated_nodes,
        "retained_ratio": retained_ratio,
    }


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(rows: list[dict[str, Any]], failures: int, max_nodes: int) -> dict[str, Any]:
    node_counts = [int(row["encoder_nodes"]) for row in rows]
    affected = [row for row in rows if int(row["exceeds_max_nodes"]) == 1]
    total_nodes = sum(node_counts)
    truncated_nodes = sum(int(row["truncated_nodes"]) for row in rows)
    loaded = len(rows)
    return {
        "max_nodes": max_nodes,
        "loaded_pt_count": loaded,
        "failed_pt_count": failures,
        "exceeding_pt_count": len(affected),
        "exceeding_pt_ratio": len(affected) / loaded if loaded else 0.0,
        "total_encoder_nodes": total_nodes,
        "would_truncate_nodes": truncated_nodes,
        "would_truncate_node_ratio": truncated_nodes / total_nodes if total_nodes else 0.0,
        "encoder_nodes_min": min(node_counts) if node_counts else 0,
        "encoder_nodes_p50": _percentile(node_counts, 0.50),
        "encoder_nodes_p90": _percentile(node_counts, 0.90),
        "encoder_nodes_p95": _percentile(node_counts, 0.95),
        "encoder_nodes_p99": _percentile(node_counts, 0.99),
        "encoder_nodes_max": max(node_counts) if node_counts else 0,
        "worst_retained_ratio": min(
            (float(row["retained_ratio"]) for row in rows), default=1.0
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(name: str, summary: dict[str, Any]) -> None:
    print(
        f"{name}: loaded={summary['loaded_pt_count']} failed={summary['failed_pt_count']} "
        f"exceed={summary['exceeding_pt_count']} "
        f"({summary['exceeding_pt_ratio']:.4%})"
    )
    print(
        "  encoder_nodes "
        f"p50={summary['encoder_nodes_p50']:.1f} "
        f"p90={summary['encoder_nodes_p90']:.1f} "
        f"p95={summary['encoder_nodes_p95']:.1f} "
        f"p99={summary['encoder_nodes_p99']:.1f} "
        f"max={summary['encoder_nodes_max']}"
    )
    print(
        "  truncation "
        f"nodes={summary['would_truncate_nodes']} "
        f"ratio={summary['would_truncate_node_ratio']:.4%} "
        f"worst_retained={summary['worst_retained_ratio']:.4%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current-schema PT graph sizes against the GNN node limit."
    )
    parser.add_argument(
        "--pt-dir",
        nargs="+",
        required=True,
        type=Path,
        help="One or more split directories containing .pt files.",
    )
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pt_graph_size_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_nodes <= 0:
        raise ValueError("--max-nodes must be positive")

    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, str]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for pt_dir in args.pt_dir:
        split = pt_dir.name
        paths = sorted(pt_dir.rglob("*.pt")) if pt_dir.is_dir() else []
        split_rows: list[dict[str, Any]] = []
        split_failures: list[dict[str, str]] = []
        for path in paths:
            try:
                row = {"split": split, **inspect_pt(path, args.max_nodes)}
                split_rows.append(row)
            except Exception as exc:  # Keep auditing after a corrupt PT.
                split_failures.append(
                    {
                        "split": split,
                        "sid": path.stem,
                        "path": str(path.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        summary = summarize(split_rows, len(split_failures), args.max_nodes)
        summaries[split] = summary
        all_rows.extend(split_rows)
        all_failures.extend(split_failures)
        _print_summary(split, summary)

    summaries["all"] = summarize(all_rows, len(all_failures), args.max_nodes)
    _print_summary("all", summaries["all"])

    detail_fields = [
        "split",
        "sid",
        "path",
        "dex_count",
        "real_nodes",
        "encoder_nodes",
        "raw_metadata_nodes",
        "exceeds_max_nodes",
        "truncated_nodes",
        "retained_ratio",
    ]
    _write_csv(args.output_dir / "graph_size_details.csv", all_rows, detail_fields)
    _write_csv(
        args.output_dir / "graph_size_failures.csv",
        all_failures,
        ["split", "sid", "path", "error"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "graph_size_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Wrote audit files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
