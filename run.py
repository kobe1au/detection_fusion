from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)
CONFIG_DIR = Path("config/experiments/tri_modal_robust")

# Main method of the thesis: the evidential trusted-fusion pipeline
# (I1 dual-source reliability, I2 conflict-aware Yager fusion, I3 conformal).
#
# The runnable "final" entry is seed_42 rather than the bare template, because
# decision-only configs (I3 coverage variants and external evals) reuse the
# seed-42 checkpoint path. Keeping seed_42 first makes one-shot paper groups
# executable from a clean results directory.
FINAL_TEMPLATE = "evidential_trusted_fusion.yaml"
PRIMARY_SEED = "seeds/seed_42.yaml"
FINAL = PRIMARY_SEED

# The previous lean linear discount-fusion method, kept as a strong prior-method
# baseline (no EDL, no subjective-logic combination, no conformal).
PRIOR_METHOD = "observable_reliability_discount_fusion.yaml"

BASELINES = [
    "baselines/api_only.yaml",
    "baselines/graph_only.yaml",
    "baselines/manifest_only.yaml",
    "baselines/api_graph_concat.yaml",
    "baselines/tri_modal_concat.yaml",
    "baselines/fixed_logit_fusion.yaml",
    PRIOR_METHOD,
]

# Remove a whole innovation from the main method.
MODULE_ABLATIONS = [
    "ablations/modules/no_i1_reliability.yaml",
    "ablations/modules/no_i3_selective_rejection.yaml",
]

# Vary the mechanism within each innovation.
I1_ABLATIONS = [
    "ablations/i1/no_edl_reliability_source.yaml",
    "ablations/i1/no_relation_evidence.yaml",
]
I2_ABLATIONS = [
    "ablations/i2/combination_dempster.yaml",
    "ablations/i2/combination_cumulative.yaml",
    "ablations/i2/combination_log_pool.yaml",
]
I3_ABLATIONS = [
    "ablations/i3/marginal_conformal.yaml",
    "ablations/i3/threshold_rejection.yaml",
]
MECHANISM_ABLATIONS = [*I1_ABLATIONS, *I2_ABLATIONS, *I3_ABLATIONS]

TRAINING_ABLATIONS = [
    "ablations/training/no_train_augmentation.yaml",
    "ablations/training/no_branch_auxiliary.yaml",
    "ablations/training/no_edl_class_weight.yaml",
]

SENSITIVITY = [
    "sensitivity/i1/evidential_weight_0_05.yaml",
    "sensitivity/i1/evidential_weight_0_2.yaml",
    "sensitivity/i1/anneal_epochs_5.yaml",
    "sensitivity/i1/anneal_epochs_20.yaml",
    "sensitivity/i1/reliability_hidden_dim_8.yaml",
    "sensitivity/i1/reliability_hidden_dim_32.yaml",
    "sensitivity/i1/visible_modifier_beta_0_5.yaml",
    "sensitivity/i1/visible_modifier_beta_2_0.yaml",
    "sensitivity/i1/visible_modifier_min_0_3.yaml",
    "sensitivity/i1/visible_modifier_min_0_7.yaml",
    "sensitivity/i2/reliability_exponent_0_25.yaml",
    "sensitivity/i2/reliability_exponent_1_0.yaml",
    "sensitivity/i3/coverage_90.yaml",
    "sensitivity/i3/coverage_99.yaml",
]

EXTERNAL_EVAL = [
    "external/obfuscapk_rename_eval.yaml",
    "external/obfuscapk_code_eval.yaml",
    "external/obfuscapk_encryption_eval.yaml",
    "external/obfuscapk_combined_eval.yaml",
]

SEEDS = [
    PRIMARY_SEED,
    "seeds/seed_2024.yaml",
    "seeds/seed_3407.yaml",
]

ALIASES = {
    "final": FINAL,
    "evidential": FINAL,
    "evidential_template": FINAL_TEMPLATE,
    "prior": PRIOR_METHOD,
    "linear": PRIOR_METHOD,
    "api": "baselines/api_only.yaml",
    "graph": "baselines/graph_only.yaml",
    "manifest": "baselines/manifest_only.yaml",
    "concat": "baselines/tri_modal_concat.yaml",
    "late": "baselines/fixed_logit_fusion.yaml",
    "no_i1": "ablations/modules/no_i1_reliability.yaml",
    "no_i3": "ablations/modules/no_i3_selective_rejection.yaml",
    "no_edl": "ablations/i1/no_edl_reliability_source.yaml",
    "no_relation": "ablations/i1/no_relation_evidence.yaml",
    "dempster": "ablations/i2/combination_dempster.yaml",
    "cumulative": "ablations/i2/combination_cumulative.yaml",
    "log_pool": "ablations/i2/combination_log_pool.yaml",
    "marginal_conformal": "ablations/i3/marginal_conformal.yaml",
    "threshold_rejection": "ablations/i3/threshold_rejection.yaml",
    "visible_beta_0_5": "sensitivity/i1/visible_modifier_beta_0_5.yaml",
    "visible_beta_2_0": "sensitivity/i1/visible_modifier_beta_2_0.yaml",
    "visible_min_0_3": "sensitivity/i1/visible_modifier_min_0_3.yaml",
    "visible_min_0_7": "sensitivity/i1/visible_modifier_min_0_7.yaml",
}

GROUPS = {
    "main": [FINAL, *BASELINES],
    "baselines": BASELINES,
    "module": MODULE_ABLATIONS,
    "mechanism": MECHANISM_ABLATIONS,
    "i1": I1_ABLATIONS,
    "i2": I2_ABLATIONS,
    "i3": [PRIMARY_SEED, *I3_ABLATIONS],
    "training_ablation": TRAINING_ABLATIONS,
    "sensitivity": SENSITIVITY,
    "sensitivity_with_seed": [PRIMARY_SEED, *SENSITIVITY],
    "external": [PRIMARY_SEED, *EXTERNAL_EVAL],
    "obfuscapk": [PRIMARY_SEED, *EXTERNAL_EVAL],
    "seed": SEEDS,
    "full": SEEDS,
    # Paper plans (evidential method is the protagonist).
    "paper_main": [*SEEDS, *BASELINES],
    "paper_ablation": [
        PRIMARY_SEED,
        *MODULE_ABLATIONS,
        *MECHANISM_ABLATIONS,
        *TRAINING_ABLATIONS,
    ],
    "paper_external": [PRIMARY_SEED, *EXTERNAL_EVAL],
    "paper_evidential": [
        *SEEDS,
        *BASELINES,
        *MODULE_ABLATIONS,
        *MECHANISM_ABLATIONS,
    ],
    "paper_evidential_all": [
        *SEEDS,
        *BASELINES,
        *MODULE_ABLATIONS,
        *MECHANISM_ABLATIONS,
        *TRAINING_ABLATIONS,
        *SENSITIVITY,
        *EXTERNAL_EVAL,
    ],
}

def available_configs() -> dict[str, Path]:
    configs: dict[str, Path] = {}
    stem_paths: dict[str, list[Path]] = {}
    if not CONFIG_DIR.is_dir():
        return configs
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        relative = path.relative_to(CONFIG_DIR)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.name == "base_tri_modal_robust.yaml" or path.stem.startswith("_"):
            continue
        key = relative.with_suffix("").as_posix()
        configs[key] = path
        stem_paths.setdefault(path.stem, []).append(path)
    for stem, paths in stem_paths.items():
        if len(paths) == 1:
            configs[stem] = paths[0]
    return configs


def _require_paths(group: str, relative_paths: list[str]) -> list[Path]:
    paths = [CONFIG_DIR / item for item in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
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
        if not path.is_file():
            raise ValueError(f"Experiment config not found: {path}")
        return [path]
    target = ALIASES.get(target, target)
    path = CONFIG_DIR / target
    if path.is_file():
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


def run_config(config_path: Path, extra_configs: list[Path] | None = None) -> None:
    extra_configs = list(extra_configs or [])
    suffix = f" + {' + '.join(str(path) for path in extra_configs)}" if extra_configs else ""
    print(f"==> Running {config_path}{suffix}", flush=True)
    subprocess.run(
        [
            PYTHON_BIN,
            "-m",
            "fusion.train",
            "--config",
            str(config_path),
            *[str(path) for path in extra_configs],
        ],
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
    parser.add_argument(
        "--extra-config",
        nargs="*",
        default=[],
        help="Additional YAML config overlays appended after each selected config.",
    )
    args = parser.parse_args()

    if args.list:
        configs = available_configs()
        print("Groups:")
        for name in sorted(GROUPS):
            print(f"  {name}")
        print("Aliases:")
        for name in sorted(ALIASES):
            print(f"  {name} -> {ALIASES[name]}")
        print("Configs:")
        for key in sorted(configs):
            print(f"  {key}: {configs[key]}")
        return

    extra_configs = [Path(item) for item in args.extra_config]
    missing_extra = [str(path) for path in extra_configs if not path.is_file()]
    if missing_extra:
        raise ValueError(f"Extra config not found: {missing_extra}")

    targets = resolve_target_specs(args.target)
    if args.dry_run:
        for path in targets:
            suffix = f" + {' + '.join(str(p) for p in extra_configs)}" if extra_configs else ""
            print(f"{path}{suffix}")
        return
    for path in targets:
        run_config(path, extra_configs)


if __name__ == "__main__":
    main()
