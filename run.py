from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
CONFIG_DIR = Path("config/experiments/tri_modal_robust")

ALIASES = {
    "final": "observable_reliability_discount_fusion.yaml",
    "api": "baselines/api_only.yaml",
    "graph": "baselines/graph_only.yaml",
    "manifest": "baselines/manifest_only.yaml",
    "api_graph": "baselines/api_graph_concat.yaml",
    "concat": "baselines/tri_modal_concat.yaml",
    "fixed": "baselines/fixed_logit_fusion.yaml",
    "confidence": "baselines/confidence_logit_fusion.yaml",
    "heuristic": "baselines/heuristic_reliability_logit_fusion.yaml",
    "learned": "baselines/learned_evidence_logit_fusion.yaml",
}

BASELINES = [
    "baselines/api_only.yaml",
    "baselines/graph_only.yaml",
    "baselines/manifest_only.yaml",
    "baselines/api_graph_concat.yaml",
    "baselines/tri_modal_concat.yaml",
    "baselines/fixed_logit_fusion.yaml",
    "baselines/confidence_logit_fusion.yaml",
    "baselines/heuristic_reliability_logit_fusion.yaml",
    "baselines/learned_evidence_logit_fusion.yaml",
]

I1_ABLATIONS = [
    "ablations/i1/no_reliability_calibration.yaml",
    "ablations/i1/integrity_alive_only.yaml",
]

I2_ABLATIONS = [
    "ablations/i2/no_masked_semantic_reconstruction.yaml",
    "ablations/i2/low_mask_probability.yaml",
    "ablations/i2/high_mask_probability.yaml",
    "ablations/i2/low_reconstruction_weight.yaml",
    "ablations/i2/high_reconstruction_weight.yaml",
]

I3_ABLATIONS = [
    "ablations/i3/no_probability_calibration.yaml",
    "ablations/i3/no_support_discount.yaml",
    "ablations/i3/no_conflict_discount.yaml",
    "ablations/i3/no_confidence_proxy_discount.yaml",
    "ablations/i3/no_hard_alive_mask.yaml",
    "ablations/i3/no_selective_rejection.yaml",
    "ablations/i3/raw_discount_no_posthoc_calibration.yaml",
]

TRAINING_ABLATIONS = [
    "ablations/training/no_train_augmentation.yaml",
    "ablations/training/no_branch_auxiliary.yaml",
    "ablations/training/no_reliability_weighted_aux.yaml",
]

SENSITIVITY = [
    "sensitivity/missing_relation_as_full_support.yaml",
    "sensitivity/acceptance_product.yaml",
    "sensitivity/coverage_80.yaml",
    "sensitivity/coverage_95.yaml",
]

SEEDS = [
    "seeds/seed_42.yaml",
    "seeds/seed_2024.yaml",
    "seeds/seed_3407.yaml",
]

FINAL = "observable_reliability_discount_fusion.yaml"

GROUPS = {
    "main": [FINAL, *BASELINES],
    "baselines": BASELINES,
    "i1": [FINAL, *I1_ABLATIONS],
    "i2": [FINAL, *I2_ABLATIONS],
    "i3": [FINAL, "baselines/learned_evidence_logit_fusion.yaml", *I3_ABLATIONS],
    "ablation": [FINAL, *I1_ABLATIONS, *I2_ABLATIONS, *I3_ABLATIONS, *TRAINING_ABLATIONS],
    "training_ablation": [FINAL, *TRAINING_ABLATIONS],
    "sensitivity": [SEEDS[0], *SENSITIVITY],
    "seed": SEEDS,
    "paper": [
        *BASELINES,
        *I1_ABLATIONS,
        *I2_ABLATIONS,
        *I3_ABLATIONS,
        *TRAINING_ABLATIONS,
        *SEEDS,
        *SENSITIVITY,
    ],
}


def available_configs() -> dict[str, Path]:
    configs: dict[str, Path] = {}
    stem_paths: dict[str, list[Path]] = {}
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        rel = path.relative_to(CONFIG_DIR)
        if path.name == "base_tri_modal_robust.yaml" or path.stem.startswith("_"):
            continue
        key = rel.with_suffix("").as_posix()
        configs[key] = path
        stem_paths.setdefault(path.stem, []).append(path)
    for stem, paths in stem_paths.items():
        if len(paths) == 1:
            configs[stem] = paths[0]
    return configs


def _require_paths(group: str, relative_paths: list[str]) -> list[Path]:
    paths = [CONFIG_DIR / item for item in relative_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Experiment group '{group}' references missing configs: {missing}")
    return paths


def resolve_targets(target: str) -> list[Path]:
    if target in GROUPS:
        return _require_paths(target, GROUPS[target])
    if target == "all":
        return sorted(set(available_configs().values()))
    if target.endswith((".yaml", ".yml")):
        path = Path(target)
        if not path.exists():
            raise ValueError(f"Experiment config not found: {path}")
        return [path]
    target = ALIASES.get(target, target)
    path = CONFIG_DIR / target
    if path.exists():
        return [path]
    configs = available_configs()
    if target in configs:
        return [configs[target]]
    known = ", ".join(["all", *sorted(GROUPS), *sorted(ALIASES), *sorted(configs)])
    raise ValueError(f"Unknown robust experiment target '{target}'. Known: {known}")


def resolve_target_specs(targets: list[str]) -> list[Path]:
    parts = [
        part.strip()
        for target in (targets or ["final"])
        for part in str(target).split(",")
        if part.strip()
    ]
    resolved: list[Path] = []
    seen: set[Path] = set()
    for part in parts:
        for path in resolve_targets(part):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(path)
    return resolved


def run_config(config_path: Path) -> None:
    print(f"==> Running {config_path}", flush=True)
    subprocess.run(
        [PYTHON_BIN, "-m", "fusion.train", "--config", str(config_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust tri-modal fusion experiments.")
    parser.add_argument(
        "target",
        nargs="*",
        default=["final"],
        help="Groups, aliases, YAML paths, or comma-separated targets.",
    )
    parser.add_argument("--list", action="store_true", help="List runnable experiment configs.")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without training.")
    args = parser.parse_args()

    if args.list:
        seen: set[Path] = set()
        for path in sorted(available_configs().values()):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                print(path.relative_to(CONFIG_DIR).with_suffix("").as_posix() + f": {path}")
        return

    targets = resolve_target_specs(args.target)
    if args.dry_run:
        for path in targets:
            print(path)
        return
    for path in targets:
        run_config(path)


if __name__ == "__main__":
    main()
