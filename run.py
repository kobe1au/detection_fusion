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

ALIASES = {
    "final": "observable_reliability_discount_fusion.yaml",
    "api": "baselines/api_only.yaml",
    "graph": "baselines/graph_only.yaml",
    "manifest": "baselines/manifest_only.yaml",
    "concat": "baselines/tri_modal_concat.yaml",
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

MODULE_ABLATIONS = [
    "ablations/modules/no_i1_observable_reliability.yaml",
    "ablations/modules/no_i2_semantic_interaction.yaml",
]

MODULE_ABLATIONS_WITH_I3 = [
    *MODULE_ABLATIONS,
    "baselines/learned_evidence_logit_fusion.yaml",
]

I1_ABLATIONS = [
    "ablations/i1/no_reliability_calibration.yaml",
    "ablations/i1/integrity_alive_only.yaml",
    "ablations/i1/no_alive_applicability_mask.yaml",
]

I1_APPENDIX_ABLATIONS = [
    "ablations/i1/no_support_evidence.yaml",
    "ablations/i1/no_conflict_evidence.yaml",
]

I2_ABLATIONS = [
    "ablations/i2/no_semantic_cross_attention.yaml",
    "ablations/i2/plain_semantic_cross_attention.yaml",
    "ablations/i2/no_cross_attention_relation_mask.yaml",
    "ablations/i2/no_cross_attention_residual_tokens.yaml",
]

I2_APPENDIX_ABLATIONS = [
    "ablations/i2/no_cross_attention_reliability_bias.yaml",
    "ablations/i2/no_cross_attention_support_bias.yaml",
    "ablations/i2/no_cross_attention_conflict_bias.yaml",
    "ablations/i2/no_semantic_presence_prior.yaml",
    "ablations/i2/joint_only_cross_attention.yaml",
]

I3_ABLATIONS = [
    "ablations/i3/no_probability_calibration.yaml",
    "ablations/i3/no_support_conflict_discount.yaml",
    "ablations/i3/no_confidence_proxy_discount.yaml",
    "ablations/i3/no_selective_rejection.yaml",
]

I3_APPENDIX_ABLATIONS = [
    "ablations/i3/no_support_discount.yaml",
    "ablations/i3/no_conflict_discount.yaml",
    "ablations/i3/no_hard_alive_mask.yaml",
    "ablations/i3/raw_discount_no_posthoc_calibration.yaml",
]

TUNING_FULL = "tuning/full_candidate.yaml"

TUNING_I1 = [
    "tuning/i1/reliability_hidden_dim_8.yaml",
    "tuning/i1/reliability_hidden_dim_32.yaml",
    "tuning/i1/missing_relation_support_0_5.yaml",
    "tuning/i1/missing_relation_support_1_0.yaml",
    "tuning/i1/support_base_0_25.yaml",
    "tuning/i1/support_base_0_75.yaml",
    "tuning/i1/conflict_min_0_1.yaml",
    "tuning/i1/conflict_min_0_2.yaml",
]

TUNING_I2 = [
    "tuning/i2/residual_tokens_2.yaml",
    "tuning/i2/residual_tokens_8.yaml",
    "tuning/i2/attention_heads_2.yaml",
    "tuning/i2/attention_heads_8.yaml",
    "tuning/i2/dropout_0_0.yaml",
    "tuning/i2/dropout_0_2.yaml",
]

TUNING_I3 = [
    "tuning/i3/acceptance_product_eval.yaml",
    "tuning/i3/coverage_80_eval.yaml",
    "tuning/i3/coverage_95_eval.yaml",
]

TUNING = [
    TUNING_FULL,
    *TUNING_I1,
    *TUNING_I2,
    *TUNING_I3,
]

TRAINING_ABLATIONS = [
    "ablations/training/no_masked_semantic_reconstruction.yaml",
    "ablations/training/no_train_augmentation.yaml",
]

TRAINING_APPENDIX_ABLATIONS = [
    "ablations/training/no_branch_auxiliary.yaml",
    "ablations/training/no_reliability_weighted_aux.yaml",
]

SENSITIVITY = [
    "sensitivity/i1/reliability_hidden_dim_8.yaml",
    "sensitivity/i1/reliability_hidden_dim_32.yaml",
    "sensitivity/i1/missing_relation_support_0_5.yaml",
    "sensitivity/i1/missing_relation_support_1_0.yaml",
    "sensitivity/i1/support_base_0_25.yaml",
    "sensitivity/i1/support_base_0_75.yaml",
    "sensitivity/i1/conflict_min_0_1.yaml",
    "sensitivity/i1/conflict_min_0_2.yaml",
    "sensitivity/i2/residual_tokens_2.yaml",
    "sensitivity/i2/residual_tokens_8.yaml",
    "sensitivity/i2/attention_heads_2.yaml",
    "sensitivity/i2/attention_heads_8.yaml",
    "sensitivity/i3/acceptance_product.yaml",
    "sensitivity/i3/coverage_80.yaml",
    "sensitivity/i3/coverage_95.yaml",
]

EXTERNAL_EVAL = [
    "external/obfuscapk_rename_eval.yaml",
    "external/obfuscapk_code_eval.yaml",
    "external/obfuscapk_encryption_eval.yaml",
    "external/obfuscapk_combined_eval.yaml",
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
    "module": MODULE_ABLATIONS_WITH_I3,
    "tuning_base": [TUNING_FULL],
    "tuning_i1": TUNING_I1,
    "tuning_i2": TUNING_I2,
    "tuning_i3": TUNING_I3,
    "tuning": TUNING,
    "i1": I1_ABLATIONS,
    "i1_appendix": I1_APPENDIX_ABLATIONS,
    "i1_full": [*I1_ABLATIONS, *I1_APPENDIX_ABLATIONS],
    "i2": I2_ABLATIONS,
    "i2_appendix": I2_APPENDIX_ABLATIONS,
    "i2_full": [*I2_ABLATIONS, *I2_APPENDIX_ABLATIONS],
    "i3": ["baselines/learned_evidence_logit_fusion.yaml", *I3_ABLATIONS],
    "i3_appendix": I3_APPENDIX_ABLATIONS,
    "i3_full": [
        "baselines/learned_evidence_logit_fusion.yaml",
        *I3_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
    ],
    "ablation": [
        *MODULE_ABLATIONS_WITH_I3,
        *TRAINING_ABLATIONS,
    ],
    "component": [
        *I1_ABLATIONS,
        *I1_APPENDIX_ABLATIONS,
        *I2_ABLATIONS,
        *I2_APPENDIX_ABLATIONS,
        *I3_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
    ],
    "ablation_appendix": [
        *I1_APPENDIX_ABLATIONS,
        *I2_APPENDIX_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
        *TRAINING_APPENDIX_ABLATIONS,
    ],
    "ablation_full": [
        *I1_ABLATIONS,
        *I1_APPENDIX_ABLATIONS,
        *I2_ABLATIONS,
        *I2_APPENDIX_ABLATIONS,
        *I3_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
        *TRAINING_ABLATIONS,
        *TRAINING_APPENDIX_ABLATIONS,
    ],
    "training_ablation": TRAINING_ABLATIONS,
    "training_ablation_appendix": TRAINING_APPENDIX_ABLATIONS,
    "training_ablation_full": [*TRAINING_ABLATIONS, *TRAINING_APPENDIX_ABLATIONS],
    "external": EXTERNAL_EVAL,
    "obfuscapk": EXTERNAL_EVAL,
    "sensitivity": SENSITIVITY,
    "sensitivity_with_seed": [SEEDS[0], *SENSITIVITY],
    "seed": SEEDS,
    "full": SEEDS,
    "paper": [
        *BASELINES,
        *MODULE_ABLATIONS,
        *TRAINING_ABLATIONS,
        *SEEDS,
    ],
    "paper_main": [
        *BASELINES,
        *MODULE_ABLATIONS,
        *TRAINING_ABLATIONS,
        *SEEDS,
    ],
    "paper_appendix": [
        *I1_APPENDIX_ABLATIONS,
        *I2_APPENDIX_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
        *TRAINING_APPENDIX_ABLATIONS,
        *SENSITIVITY,
    ],
    "paper_external": EXTERNAL_EVAL,
    "paper_appendix_with_seed": [
        SEEDS[0],
        *I1_APPENDIX_ABLATIONS,
        *I2_APPENDIX_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
        *TRAINING_APPENDIX_ABLATIONS,
        *SENSITIVITY,
    ],
    "paper_all": [
        *BASELINES,
        *MODULE_ABLATIONS,
        *I1_ABLATIONS,
        *I1_APPENDIX_ABLATIONS,
        *I2_ABLATIONS,
        *I2_APPENDIX_ABLATIONS,
        *I3_ABLATIONS,
        *I3_APPENDIX_ABLATIONS,
        *TRAINING_ABLATIONS,
        *TRAINING_APPENDIX_ABLATIONS,
        *SEEDS,
        *SENSITIVITY,
        *EXTERNAL_EVAL,
    ],
}


def available_configs() -> dict[str, Path]:
    configs: dict[str, Path] = {}
    stem_paths: dict[str, list[Path]] = {}
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        relative = path.relative_to(CONFIG_DIR)
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
        help="Additional YAML overlays appended to every selected experiment config.",
    )
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
    extra_configs = [Path(path) for path in args.extra_config]
    missing_extra = [str(path) for path in extra_configs if not path.exists()]
    if missing_extra:
        raise ValueError(f"Extra config overlay not found: {missing_extra}")
    if args.dry_run:
        for path in targets:
            if extra_configs:
                print(str(path) + " + " + " + ".join(str(item) for item in extra_configs))
            else:
                print(path)
        return
    for path in targets:
        run_config(path, extra_configs)


if __name__ == "__main__":
    main()
