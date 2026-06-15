#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API+graph complementarity evaluation wrapper.

The implementation lives in :mod:`fusion.complementarity`.  This wrapper keeps
the historical ``test_m`` entrypoint while using the current API+graph dataset,
model, and batch-preparation logic.

Example:
    python test_m/complementarity.py \
      --base config/base.yaml \
      --split test \
      --model api=experiments/api_baseline/42/best_api_baseline.pt \
      --model graph=experiments/gatv2_baseline/42/best_gatv2_baseline.pt \
      --model concat=experiments/api_graph_concat/42/best_api_graph_concat.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fusion.complementarity import (  # noqa: E402,F401
    DEFAULT_MODELS,
    ModelSpec,
    PredictionPack,
    aligned_arrays,
    collect_predictions,
    load_checkpoint_config,
    load_yaml,
    metric_summary,
    multi_model_oracle,
    pairwise_complementarity,
    parse_model_arg,
    resolve_existing_path,
    select_device,
    validate_full_config,
    write_csv,
)
from fusion.complementarity import main as _main  # noqa: E402


def main() -> None:
    if "--out-dir" not in sys.argv:
        sys.argv.extend(["--out-dir", "test_m/results/complementarity"])
    _main()


if __name__ == "__main__":
    main()
