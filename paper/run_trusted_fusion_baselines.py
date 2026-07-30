from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

METHOD_TARGETS = {
    "ours": "final",
    "tmc": "tmc",
    "ecml": "ecml",
    # Component-level QMF comparison: energy-weighted late fusion only.
    "qmf_energy": "qmf_energy",
}

FUSION_RULE_TARGETS = {
    "dempster_rule_only": "dempster",
    "cumulative_subjective_logic": "cumulative",
    "log_pool": "log_pool",
}

FORMAL_TARGETS = {**METHOD_TARGETS, **FUSION_RULE_TARGETS}


def run_target(target: str, extra_config: str | None, dry_run: bool) -> None:
    cmd = [sys.executable, "run.py", target]
    if extra_config:
        cmd.extend(["--extra-config", extra_config])
    if dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-level method baselines and complete fusion-rule "
            "comparisons under the common protocol."
        )
    )
    parser.add_argument(
        "--method",
        choices=["all", *FORMAL_TARGETS.keys()],
        default="all",
        help="Paper comparison to run.",
    )
    parser.add_argument(
        "--extra-config",
        default="config/experiments/tri_modal_robust/_autodl_paths.yaml",
        help="Path overlay used on AutoDL; pass empty string to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = list(FORMAL_TARGETS) if args.method == "all" else [args.method]
    extra_config = str(args.extra_config).strip() or None
    for method in methods:
        run_target(FORMAL_TARGETS[method], extra_config, bool(args.dry_run))


if __name__ == "__main__":
    main()
