from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)
CONFIG_DIR = Path("config/experiments/tri_modal_robust")
DEFAULT_RUNNER_MODULE = "fusion.baseline_train"
ALLOWED_RUNNER_MODULES = {
    DEFAULT_RUNNER_MODULE,
    "fusion.care_train",
}
# Only these directories participate in the formal experiment catalog.
FORMAL_CONFIG_DIRS = {
    "ablations",
    "baselines",
    "seeds",
}
# Paper method: four fixed clean-trained paths, an OOF path-correctness risk
# head, conservative conditional routing, and natural-only malware-FN CRC.
PRIMARY_SEED = "seeds/seed_42.yaml"

BASELINES = [
    "baselines/api_only.yaml",
    "baselines/graph_only.yaml",
    "baselines/manifest_only.yaml",
    "baselines/api_graph_concat.yaml",
    "baselines/tri_modal_concat.yaml",
    "baselines/fixed_logit_fusion.yaml",
    "baselines/dense_embedding_gate_adapted.yaml",
]

TRUSTED_FUSION_BASELINES = [
    "baselines/trusted/dempster.yaml",
    "baselines/trusted/cumulative.yaml",
    "baselines/trusted/log_pool.yaml",
    "baselines/trusted/tmc_style_adapted.yaml",
    "baselines/trusted/qmf_energy.yaml",
    "baselines/trusted/ecml_style_adapted.yaml",
]

FUSION_RULE_COMPARISONS = [
    "baselines/trusted/dempster.yaml",
    "baselines/trusted/cumulative.yaml",
    "baselines/trusted/log_pool.yaml",
]

CARE_ABLATIONS = [
    "ablations/care/no_learned_routing.yaml",
    "ablations/care/route_on_all_samples.yaml",
    "ablations/care/msp_acceptance.yaml",
]

SEEDS = [
    PRIMARY_SEED,
    "seeds/seed_2024.yaml",
    "seeds/seed_3407.yaml",
]

ALIASES = {
    "final": PRIMARY_SEED,
    "ours": PRIMARY_SEED,
    "care": PRIMARY_SEED,
    "care_droid": PRIMARY_SEED,
    "api": "baselines/api_only.yaml",
    "graph": "baselines/graph_only.yaml",
    "manifest": "baselines/manifest_only.yaml",
    "concat": "baselines/tri_modal_concat.yaml",
    "late": "baselines/fixed_logit_fusion.yaml",
    "embedding_gate": "baselines/dense_embedding_gate_adapted.yaml",
    "no_learned_routing": "ablations/care/no_learned_routing.yaml",
    "route_on_all": "ablations/care/route_on_all_samples.yaml",
    "route_on_all_samples": "ablations/care/route_on_all_samples.yaml",
    "msp_acceptance": "ablations/care/msp_acceptance.yaml",
    "dempster": "baselines/trusted/dempster.yaml",
    "cumulative": "baselines/trusted/cumulative.yaml",
    "log_pool": "baselines/trusted/log_pool.yaml",
    "tmc": "baselines/trusted/tmc_style_adapted.yaml",
    "tmc_style_adapted": "baselines/trusted/tmc_style_adapted.yaml",
    "qmf_energy": "baselines/trusted/qmf_energy.yaml",
    "ecml": "baselines/trusted/ecml_style_adapted.yaml",
    "ecml_style_adapted": "baselines/trusted/ecml_style_adapted.yaml",
}

GROUPS = {
    "main": [PRIMARY_SEED],
    "main_comparison": [PRIMARY_SEED, *BASELINES, *TRUSTED_FUSION_BASELINES],
    "baselines": BASELINES,
    "trusted_baselines": TRUSTED_FUSION_BASELINES,
    "care_ablation": CARE_ABLATIONS,
    "method_ablation": CARE_ABLATIONS,
    "fusion_rules": FUSION_RULE_COMPARISONS,
    "seed": SEEDS,
    "paper_main": [*SEEDS, *BASELINES, *TRUSTED_FUSION_BASELINES],
    "paper_ablation": CARE_ABLATIONS,
    "paper_all": [
        *SEEDS,
        *BASELINES,
        *TRUSTED_FUSION_BASELINES,
        *CARE_ABLATIONS,
    ],
}


def available_configs() -> dict[str, Path]:
    configs: dict[str, Path] = {}
    stem_paths: dict[str, list[Path]] = {}
    # Registration is explicit. Merely leaving an auxiliary YAML below a
    # formal-looking directory must never make `run.py all` execute a
    # non-paper lifecycle.
    registered_relatives = {
        Path(item).as_posix()
        for targets in GROUPS.values()
        for item in targets
    }
    if not CONFIG_DIR.is_dir():
        return configs
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        relative = path.relative_to(CONFIG_DIR)
        if any(part.startswith(".") for part in relative.parts):
            continue
        # Top-level YAML files are inheritance templates or machine overlays,
        # never independently runnable paper cells.
        if len(relative.parts) == 1:
            continue
        if relative.parts[0] not in FORMAL_CONFIG_DIRS:
            continue
        if relative.as_posix() not in registered_relatives:
            continue
        if path.name in {"base_tri_modal_robust.yaml", "debug_fast.yaml"} or path.stem.startswith("_"):
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
    # Composite paper groups intentionally reuse reference cells from module
    # and factorial matrices. Execute each physical config once while
    # preserving the declared first-occurrence order.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _resolve_declared_runner_module(
    config_path: Path,
    *,
    _seen: set[Path] | None = None,
) -> str | None:

    path = config_path.resolve()
    seen = set(_seen or ())
    if path in seen:
        raise ValueError(f"Recursive config defaults detected while resolving runner: {path}")
    seen.add(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read experiment config while resolving runner: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")

    module: str | None = None
    defaults = raw.get("defaults", []) or []
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ValueError(f"Config defaults must be a list: {path}")
    for item in defaults:
        parent = Path(str(item))
        if not parent.is_absolute():
            parent = path.parent / parent
        inherited = _resolve_declared_runner_module(parent, _seen=seen)
        if inherited is not None:
            module = inherited

    runner = raw.get("runner")
    if runner is not None:
        if not isinstance(runner, dict) or not str(runner.get("module", "")).strip():
            raise ValueError(f"runner.module must be a non-empty string: {path}")
        module = str(runner["module"]).strip()
    if module is not None and module not in ALLOWED_RUNNER_MODULES:
        raise ValueError(
            f"Unsupported runner.module={module!r} in {path}; "
            f"allowed={sorted(ALLOWED_RUNNER_MODULES)}"
        )
    return module


def resolve_runner_module(config_path: Path) -> str:
    """Resolve the runner declared through a YAML defaults chain.

    CARE-Droid owns a closed lifecycle in ``fusion.care_train``. The 13
    registered comparison methods own a separate closed lifecycle in
    ``fusion.baseline_train``. Runner selection is therefore configuration
    identity, not a command-line option that can accidentally execute one
    method through the other method's lifecycle.
    """

    return _resolve_declared_runner_module(config_path) or DEFAULT_RUNNER_MODULE


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


def validate_execution_target_order(
    targets: list[str],
    *,
    dry_run: bool,
) -> None:
    """Reject the unordered catalog shortcut for real executions.

    ``all`` enumerates every leaf YAML for inspection, but those leaves are
    not an executable DAG: CARE decision-only cells require the primary
    pipeline checkpoint.
    """

    parts = {
        part.strip()
        for target in (targets or ["final"])
        for part in str(target).split(",")
        if part.strip()
    }
    if not dry_run and "all" in parts:
        raise ValueError(
            "Target 'all' is catalog-only and may be used only with --dry-run. "
            "Run explicit ordered groups instead: train the main/baseline "
            "groups first, then run CARE ablations. Use "
            "--list to inspect the full catalog."
        )


def run_config(
    config_path: Path,
    extra_configs: list[Path] | None = None,
    *,
    overwrite: bool = False,
) -> None:
    extra_configs = list(extra_configs or [])
    runner_module = resolve_runner_module(config_path)
    suffix = f" + {' + '.join(str(path) for path in extra_configs)}" if extra_configs else ""
    print(f"==> Running [{runner_module}] {config_path}{suffix}", flush=True)
    subprocess.run(
        [
            PYTHON_BIN,
            "-m",
            runner_module,
            "--config",
            str(config_path),
            *[str(path) for path in extra_configs],
            *(["--overwrite"] if overwrite else []),
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
        "--overwrite",
        action="store_true",
        help="Explicitly allow each selected run to replace existing artifacts.",
    )
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

    try:
        validate_execution_target_order(args.target, dry_run=bool(args.dry_run))
    except ValueError as exc:
        parser.error(str(exc))
    targets = resolve_target_specs(args.target)
    if args.dry_run:
        for path in targets:
            suffix = f" + {' + '.join(str(p) for p in extra_configs)}" if extra_configs else ""
            print(f"[{resolve_runner_module(path)}] {path}{suffix}")
        return
    for path in targets:
        run_config(
            path,
            extra_configs,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
