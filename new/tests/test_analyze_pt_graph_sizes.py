from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pt_graph_sizes.py"
SPEC = importlib.util.spec_from_file_location("analyze_pt_graph_sizes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_graph_size_counts_multi_dex_and_empty_ghost_node() -> None:
    payload = {
        "dex_list": [
            {"call_x": torch.ones((4, 3))},
            {"call_x": torch.empty((0, 3))},
        ],
        "observable_metadata": {"graph_node_count_raw": 4},
    }

    result = MODULE.graph_size_from_payload(payload)

    assert result == {
        "dex_count": 2,
        "real_nodes": 4,
        "encoder_nodes": 5,
        "raw_metadata_nodes": 4,
    }


def test_inspect_and_summary_report_model_side_truncation(tmp_path: Path) -> None:
    pt_path = tmp_path / "sample.pt"
    torch.save(
        {
            "dex_list": [{"call_x": torch.ones((7, 2))}],
            "observable_metadata": {"graph_node_count_raw": 7},
        },
        pt_path,
    )

    row = MODULE.inspect_pt(pt_path, max_nodes=5)
    summary = MODULE.summarize([row], failures=1, max_nodes=5)

    assert row["exceeds_max_nodes"] == 1
    assert row["truncated_nodes"] == 2
    assert row["retained_ratio"] == pytest.approx(5 / 7)
    assert summary["exceeding_pt_ratio"] == 1.0
    assert summary["would_truncate_node_ratio"] == pytest.approx(2 / 7)
    assert summary["failed_pt_count"] == 1


def test_current_schema_is_required() -> None:
    with pytest.raises(ValueError, match="dex_list"):
        MODULE.graph_size_from_payload({"call_x": torch.ones((2, 2))})
