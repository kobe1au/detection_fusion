from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)
CONFIG_DIR = Path("config/experiments/tri_modal_robust")
# Only these directories participate in the formal experiment catalog.
FORMAL_CONFIG_DIRS = {
    "ablations",
    "appendix",
    "baselines",
    "natural_subsets",
    "seeds",
}
NATURAL_SUBSET_DIR = Path("labels/natural_subsets")
NATURAL_SUBSET_SCHEMA_VERSION = 6
NATURAL_SUBSET_PROTOCOL_ID = "i1_i2_unseen_validation_natural_difficulty_v2"
NATURAL_SUBSET_FILES = (
    "test_branch_disagreement.csv",
    "test_api_only_wrong.csv",
    "test_graph_only_wrong.csv",
    "test_manifest_only_wrong.csv",
    "test_reliability_imbalance.csv",
    "test_high_cross_modal_conflict.csv",
)

# Paper method: I1 calibrated branch-correctness reliability, I2 conditional
# modality routing plus threshold-aligned malware-FN risk, and I3 malware-FN
# risk control.
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
    "baselines/trusted/tmc_style_adapted.yaml",
    "baselines/trusted/qmf_energy.yaml",
    "baselines/trusted/ecml_style_adapted.yaml",
]

NATURAL_SUBSET_OURS_EVAL = [
    "natural_subsets/ours_eval.yaml",
]

NATURAL_SUBSET_BASELINE_EVAL = [
    "natural_subsets/api_only_eval.yaml",
    "natural_subsets/graph_only_eval.yaml",
    "natural_subsets/manifest_only_eval.yaml",
    "natural_subsets/api_graph_concat_eval.yaml",
    "natural_subsets/tri_modal_concat_eval.yaml",
    "natural_subsets/fixed_logit_fusion_eval.yaml",
    "natural_subsets/dense_embedding_gate_adapted_eval.yaml",
]

NATURAL_SUBSET_FUSION_RULE_EVAL = [
    "natural_subsets/dempster_eval.yaml",
    "natural_subsets/cumulative_eval.yaml",
    "natural_subsets/log_pool_eval.yaml",
    "natural_subsets/conflict_weighted_opinion_eval.yaml",
]

NATURAL_SUBSET_TRUSTED_EVAL = [
    "natural_subsets/tmc_style_adapted_eval.yaml",
    "natural_subsets/qmf_energy_eval.yaml",
    "natural_subsets/ecml_style_adapted_eval.yaml",
]

NATURAL_SUBSET_EVAL = [
    *NATURAL_SUBSET_OURS_EVAL,
    *NATURAL_SUBSET_BASELINE_EVAL,
    *NATURAL_SUBSET_FUSION_RULE_EVAL,
    *NATURAL_SUBSET_TRUSTED_EVAL,
]

MODULE_ABLATIONS = [
    "ablations/modules/no_i1_reliability.yaml",
    "ablations/modules/no_i2_learned_components.yaml",
    "ablations/modules/no_i3_decision_layer.yaml",
]

I1_ATOMIC_ABLATIONS = [
    "ablations/i1/no_evidential_certainty.yaml",
    "ablations/i1/no_prediction_margin.yaml",
    "ablations/i1/no_predicted_class_intercept.yaml",
]

I1_COMPARATORS = [
    # Matched-budget simple comparator: same nested OOF identities/views as I1,
    # but only one NLL-fitted scalar temperature per modality branch.
    "ablations/i1/temperature_scaling_confidence.yaml",
]

I2_ROUTER_ATOMIC_ABLATIONS = [
    "ablations/i2/router_prior_only.yaml",
    "ablations/i2/router_risk_prior.yaml",
    "ablations/i2/router_no_route_conflict.yaml",
    "ablations/i2/router_no_risk_conflict.yaml",
]

I2_SCENARIO_WEIGHT_ABLATIONS = [
    "ablations/i2/scenario_weight_clean_0_70.yaml",
    "ablations/i2/scenario_weight_clean_0_30.yaml",
]

I2_PRIOR_BETA_SENSITIVITY = [
    "appendix/prior_beta_0_5.yaml",
    # beta=1 is the already registered prior-only atomic cell.
    "ablations/i2/router_prior_only.yaml",
    "appendix/prior_beta_2_0.yaml",
]

FUSION_RULE_COMPARISONS = [
    "ablations/i2/combination_dempster.yaml",
    "ablations/i2/combination_cumulative.yaml",
    "ablations/i2/combination_log_pool.yaml",
    "ablations/i2/combination_conflict_weighted_opinion.yaml",
]

I2_MECHANISM_ABLATIONS = [
    *I2_ROUTER_ATOMIC_ABLATIONS,
    *I2_SCENARIO_WEIGHT_ABLATIONS,
]

I3_ACCEPTANCE_SCORE_COMPARISONS = [
    "ablations/i3/acceptance_msp_risk_control.yaml",
    "ablations/i3/acceptance_deployed_class_probability_risk_control.yaml",
]

I3_MECHANISM_ABLATIONS = [
    "ablations/i3/class_conditional_conformal.yaml",
    "ablations/i3/marginal_conformal.yaml",
    "ablations/i3/conflict_augmented_conformal.yaml",
    "ablations/i3/deployed_class_probability_threshold.yaml",
    "ablations/i3/msp_threshold.yaml",
    "ablations/i3/uncertainty_threshold.yaml",
    "ablations/i3/model_acceptance_threshold.yaml",
    # fused_risk is already the primary experiment; run only the two
    # decision-only score substitutions here.
    *I3_ACCEPTANCE_SCORE_COMPARISONS,
]

MECHANISM_ABLATIONS = [
    *I1_ATOMIC_ABLATIONS,
    *I1_COMPARATORS,
    *I2_MECHANISM_ABLATIONS,
    *I3_MECHANISM_ABLATIONS,
]

I1_I2_FACTORIAL = [
    PRIMARY_SEED,
    "ablations/modules/no_i1_reliability.yaml",
    "ablations/modules/no_i2_learned_components.yaml",
    "ablations/factorial/i1_i2/i1_off_i2_off.yaml",
]

FACTORIAL_ABLATIONS = [
    *I1_I2_FACTORIAL,
]

FACTORIAL_REMAINING = [
    *I1_I2_FACTORIAL[1:],
]

TRAINING_ABLATIONS = [
    "ablations/training/no_branch_auxiliary.yaml",
    "ablations/training/no_edl_supervision.yaml",
    "ablations/training/no_edl_class_weight.yaml",
]

APPENDIX_SENSITIVITY = [
    "appendix/edl_weight_0_10.yaml",
    # Canonical clean-only temperature scaling is separated from the
    # matched-budget I1 comparator in the formal mechanism comparisons.
    "ablations/i1/temperature_scaling_confidence_clean_only.yaml",
    "appendix/prior_beta_0_5.yaml",
    "appendix/prior_beta_2_0.yaml",
    "appendix/risk_level_0_03_eval.yaml",
    "appendix/risk_level_0_10_eval.yaml",
    # Binary entropy certainty is rank-equivalent to MSP; retain only as a
    # numerical sanity check rather than a formal independent mechanism cell.
    "ablations/i3/predictive_entropy_threshold.yaml",
]

SEEDS = [
    PRIMARY_SEED,
    "seeds/seed_2024.yaml",
    "seeds/seed_3407.yaml",
]

ALIASES = {
    "final": PRIMARY_SEED,
    "evidential": PRIMARY_SEED,
    "api": "baselines/api_only.yaml",
    "graph": "baselines/graph_only.yaml",
    "manifest": "baselines/manifest_only.yaml",
    "concat": "baselines/tri_modal_concat.yaml",
    "late": "baselines/fixed_logit_fusion.yaml",
    "embedding_gate": "baselines/dense_embedding_gate_adapted.yaml",
    "no_i1": "ablations/modules/no_i1_reliability.yaml",
    "no_i2": "ablations/modules/no_i2_learned_components.yaml",
    "no_i3": "ablations/modules/no_i3_decision_layer.yaml",
    "no_evidential_certainty": "ablations/i1/no_evidential_certainty.yaml",
    "no_prediction_margin": "ablations/i1/no_prediction_margin.yaml",
    "no_predicted_class_intercept": "ablations/i1/no_predicted_class_intercept.yaml",
    "i1_temperature": "ablations/i1/temperature_scaling_confidence.yaml",
    "i1_temperature_clean": "ablations/i1/temperature_scaling_confidence_clean_only.yaml",
    "dempster": "ablations/i2/combination_dempster.yaml",
    "cumulative": "ablations/i2/combination_cumulative.yaml",
    "log_pool": "ablations/i2/combination_log_pool.yaml",
    "conflict_weighted_opinion": "ablations/i2/combination_conflict_weighted_opinion.yaml",
    "router_prior_only": "ablations/i2/router_prior_only.yaml",
    "router_risk_prior": "ablations/i2/router_risk_prior.yaml",
    "router_no_route_conflict": "ablations/i2/router_no_route_conflict.yaml",
    "router_no_risk_conflict": "ablations/i2/router_no_risk_conflict.yaml",
    "i2_weight_clean_070": "ablations/i2/scenario_weight_clean_0_70.yaml",
    "i2_weight_clean_030": "ablations/i2/scenario_weight_clean_0_30.yaml",
    "prior_beta_050": "appendix/prior_beta_0_5.yaml",
    "prior_beta_100": "ablations/i2/router_prior_only.yaml",
    "prior_beta_200": "appendix/prior_beta_2_0.yaml",
    "class_conditional_conformal": "ablations/i3/class_conditional_conformal.yaml",
    "marginal_conformal": "ablations/i3/marginal_conformal.yaml",
    "conflict_conformal": "ablations/i3/conflict_augmented_conformal.yaml",
    "deployed_probability_threshold": "ablations/i3/deployed_class_probability_threshold.yaml",
    "msp_threshold": "ablations/i3/msp_threshold.yaml",
    "predictive_entropy_threshold": "ablations/i3/predictive_entropy_threshold.yaml",
    "uncertainty_threshold": "ablations/i3/uncertainty_threshold.yaml",
    "acceptance_threshold": "ablations/i3/model_acceptance_threshold.yaml",
    "accept_risk": PRIMARY_SEED,
    "accept_msp_crc": "ablations/i3/acceptance_msp_risk_control.yaml",
    "accept_deployed_probability_crc": "ablations/i3/acceptance_deployed_class_probability_risk_control.yaml",
    "i1_on_i2_on": PRIMARY_SEED,
    "i1_off_i2_on": "ablations/modules/no_i1_reliability.yaml",
    "i1_on_i2_off": "ablations/modules/no_i2_learned_components.yaml",
    "i1_off_i2_off": "ablations/factorial/i1_i2/i1_off_i2_off.yaml",
    "i3_on": PRIMARY_SEED,
    "i3_off": "ablations/modules/no_i3_decision_layer.yaml",
    "risk_03": "appendix/risk_level_0_03_eval.yaml",
    "tmc": "baselines/trusted/tmc_style_adapted.yaml",
    "tmc_style_adapted": "baselines/trusted/tmc_style_adapted.yaml",
    "qmf_energy": "baselines/trusted/qmf_energy.yaml",
    "ecml": "baselines/trusted/ecml_style_adapted.yaml",
    "ecml_style_adapted": "baselines/trusted/ecml_style_adapted.yaml",
    "no_edl_supervision": "ablations/training/no_edl_supervision.yaml",
    "natural_ours": "natural_subsets/ours_eval.yaml",
    "natural_dempster": "natural_subsets/dempster_eval.yaml",
    "natural_cumulative": "natural_subsets/cumulative_eval.yaml",
    "natural_log_pool": "natural_subsets/log_pool_eval.yaml",
    "natural_conflict_weighted_opinion": "natural_subsets/conflict_weighted_opinion_eval.yaml",
    "natural_embedding_gate": "natural_subsets/dense_embedding_gate_adapted_eval.yaml",
    "natural_tmc": "natural_subsets/tmc_style_adapted_eval.yaml",
    "natural_qmf_energy": "natural_subsets/qmf_energy_eval.yaml",
    "natural_ecml": "natural_subsets/ecml_style_adapted_eval.yaml",
}

# Old aliases named partial mechanisms as if they were complete literature
# methods. Failing explicitly prevents old commands and checkpoints from being
# silently reinterpreted under the current style-adapted baseline protocol.
REMOVED_ALIASES = {
    "no_i2_cumulative": (
        "use 'cumulative'; cumulative is a full fusion-rule comparison, "
        "not an atomic no-I2 cell"
    ),
    "i2_rules": "use 'fusion_rules'; these are full fusion-rule comparisons",
    "natural_subset_i2": "use 'natural_subset_fusion_rules'",
    "natural_i2": "use 'natural_subset_fusion_rules'",
    "tmc_faithful": "use 'tmc' (TMC-style adapted)",
    "tmc_style": "use 'tmc' (TMC-style adapted)",
    "qmf_style": "use 'qmf_energy' (energy-fusion component only)",
    "ecml_faithful": "use 'ecml' (ECML-style adapted)",
    "ecml_adapted": "use 'ecml' (ECML-style adapted) or 'conflict_weighted_opinion'",
    "ecml_style": "use 'ecml' (ECML-style adapted) or 'conflict_weighted_opinion'",
    "ecml_inspired": "use 'conflict_weighted_opinion'; this custom rule is not ECML",
    "natural_tmc_style": "use 'natural_tmc'",
    "natural_qmf_style": "use 'natural_qmf_energy'",
    "natural_ecml_adapted": "use 'natural_ecml'",
    "natural_ecml_style": "use 'natural_conflict_weighted_opinion'",
}

GROUPS = {
    "main": [PRIMARY_SEED, *BASELINES],
    "baselines": BASELINES,
    "trusted_baselines": TRUSTED_FUSION_BASELINES,
    "natural_subsets": NATURAL_SUBSET_EVAL,
    "natural_subset_ours": NATURAL_SUBSET_OURS_EVAL,
    "natural_subset_baselines": NATURAL_SUBSET_BASELINE_EVAL,
    "natural_subset_fusion_rules": NATURAL_SUBSET_FUSION_RULE_EVAL,
    "natural_subset_trusted": NATURAL_SUBSET_TRUSTED_EVAL,
    "module": MODULE_ABLATIONS,
    "mechanism": MECHANISM_ABLATIONS,
    "i1_atomic": I1_ATOMIC_ABLATIONS,
    "i1_comparator": I1_COMPARATORS,
    "i2_atomic": I2_ROUTER_ATOMIC_ABLATIONS,
    "fusion_rules": FUSION_RULE_COMPARISONS,
    "i2_scenario_weights": I2_SCENARIO_WEIGHT_ABLATIONS,
    "i2_prior_beta_sensitivity": I2_PRIOR_BETA_SENSITIVITY,
    "i2_mechanism": I2_MECHANISM_ABLATIONS,
    "i3_mechanism": I3_MECHANISM_ABLATIONS,
    "i3_acceptance_score": I3_ACCEPTANCE_SCORE_COMPARISONS,
    "i1_i2_2x2": I1_I2_FACTORIAL,
    "i1_i2_factorial": I1_I2_FACTORIAL,
    "factorial": FACTORIAL_ABLATIONS,
    "factorial_remaining": FACTORIAL_REMAINING,
    "training_ablation": TRAINING_ABLATIONS,
    "seed": SEEDS,
    "appendix": APPENDIX_SENSITIVITY,
    "paper_main": [*SEEDS, *BASELINES, *TRUSTED_FUSION_BASELINES],
    "paper_ablation": [
        PRIMARY_SEED,
        *MODULE_ABLATIONS,
        *MECHANISM_ABLATIONS,
        *FUSION_RULE_COMPARISONS,
        *FACTORIAL_ABLATIONS,
    ],
    "paper_evidential": [
        *SEEDS,
        *BASELINES,
        *TRUSTED_FUSION_BASELINES,
        *MODULE_ABLATIONS,
        *MECHANISM_ABLATIONS,
        *FUSION_RULE_COMPARISONS,
        *FACTORIAL_ABLATIONS,
    ],
    "paper_natural": NATURAL_SUBSET_EVAL,
    "paper_evidential_all": [
        *SEEDS,
        *BASELINES,
        *TRUSTED_FUSION_BASELINES,
        *MODULE_ABLATIONS,
        *MECHANISM_ABLATIONS,
        *FUSION_RULE_COMPARISONS,
        *FACTORIAL_ABLATIONS,
        *TRAINING_ABLATIONS,
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
        # Top-level YAML files are inheritance templates or machine overlays,
        # never independently runnable paper cells.
        if len(relative.parts) == 1:
            continue
        if relative.parts[0] not in FORMAL_CONFIG_DIRS:
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


def resolve_targets(target: str) -> list[Path]:
    if target in REMOVED_ALIASES:
        raise ValueError(
            f"Experiment alias '{target}' was removed because its method identity "
            f"was ambiguous; {REMOVED_ALIASES[target]}."
        )
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
    not an executable DAG: decision-only cells require the primary checkpoint,
    and natural-subset cells additionally require a frozen subset manifest.
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
            "groups first, build the frozen natural-subset artifacts, then run "
            "paper_natural. Use --list to inspect the full catalog."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_natural_subset_artifacts(root: Path = ROOT) -> None:
    subset_dir = root / NATURAL_SUBSET_DIR
    manifest_path = subset_dir / "subset_manifest.json"
    rebuild = (
        "Rebuild them with scripts/build_natural_subset_csvs.py from the "
        "current seed-42 gate_diagnostics.csv before running natural subsets."
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"Natural-subset manifest is missing: {manifest_path}. {rebuild}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Natural-subset manifest is unreadable: {manifest_path}. {rebuild}") from exc
    if int(manifest.get("schema_version", -1)) != NATURAL_SUBSET_SCHEMA_VERSION:
        raise RuntimeError(
            "Natural-subset manifest uses an obsolete schema "
            f"({manifest.get('schema_version')!r}); expected "
            f"{NATURAL_SUBSET_SCHEMA_VERSION}. {rebuild}"
        )
    if str(manifest.get("protocol_id", "")) != NATURAL_SUBSET_PROTOCOL_ID:
        raise RuntimeError(
            "Natural-subset manifest protocol does not match the registered "
            f"method: found={manifest.get('protocol_id')!r}, expected="
            f"{NATURAL_SUBSET_PROTOCOL_ID!r}. {rebuild}"
        )
    guarantees = manifest.get("protocol_guarantees")
    required_true = (
        "thresholds_fit_on_validation_only",
        "calibration_split_unseen_by_i1_i2",
        "i1_success_is_not_defined_by_i1_reliability",
        "label_dependent_subsets_are_diagnostic_only",
    )
    if (
        not isinstance(guarantees, dict)
        or not all(guarantees.get(key) is True for key in required_true)
        or guarantees.get("target_split_used_for_threshold_selection") is not False
    ):
        raise RuntimeError(
            f"Natural-subset manifest lacks the frozen-validation safeguards. {rebuild}"
        )
    if (
        str(manifest.get("calibration_split", "")) != "val_selection"
        or str(manifest.get("target_split", "")) != "test_clean"
    ):
        raise RuntimeError(
            "Natural-subset manifest uses the wrong lifecycle splits; expected "
            "calibration_split='val_selection' and target_split='test_clean'. "
            f"{rebuild}"
        )
    missing = [
        str(subset_dir / name)
        for name in NATURAL_SUBSET_FILES
        if not (subset_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Natural-subset CSVs are incomplete: {missing}. {rebuild}")

    for path_key, hash_key in (
        ("diagnostics", "diagnostics_sha256"),
        ("test_csv", "test_csv_sha256"),
    ):
        source = Path(str(manifest.get(path_key, "")))
        if not source.is_absolute():
            source = root / source
        expected_hash = str(manifest.get(hash_key, ""))
        if not source.is_file() or len(expected_hash) != 64 or _sha256(source) != expected_hash:
            raise RuntimeError(
                f"Natural-subset source changed or is unavailable: {source}. {rebuild}"
            )

    subset_records = manifest.get("subsets")
    if not isinstance(subset_records, list):
        raise RuntimeError(
            f"Natural-subset manifest has no output provenance. {rebuild}"
        )
    output_hashes: dict[str, str] = {}
    for record in subset_records:
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Natural-subset manifest has malformed output provenance. {rebuild}"
            )
        name = Path(str(record.get("csv", ""))).name
        if not name or name in output_hashes:
            raise RuntimeError(
                f"Natural-subset manifest has duplicate or unnamed outputs. {rebuild}"
            )
        output_hashes[name] = str(record.get("csv_sha256", ""))
    expected_outputs = set(NATURAL_SUBSET_FILES)
    if set(output_hashes) != expected_outputs:
        raise RuntimeError(
            "Natural-subset manifest output set does not match the experiment "
            f"protocol: found={sorted(output_hashes)}, "
            f"expected={sorted(expected_outputs)}. {rebuild}"
        )
    for name, expected_hash in output_hashes.items():
        output_path = subset_dir / name
        if len(expected_hash) != 64 or _sha256(output_path) != expected_hash:
            raise RuntimeError(
                f"Natural-subset CSV changed after generation: {output_path}. {rebuild}"
            )


def run_config(
    config_path: Path,
    extra_configs: list[Path] | None = None,
    *,
    overwrite: bool = False,
    encoder_checkpoint: str | None = None,
) -> None:
    extra_configs = list(extra_configs or [])
    if "natural_subsets" in config_path.parts:
        validate_natural_subset_artifacts()
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
            *(["--overwrite"] if overwrite else []),
            *(
                ["--encoder-checkpoint", str(encoder_checkpoint)]
                if encoder_checkpoint
                else []
            ),
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
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help=(
            "Reuse one strict Stage-1 encoder artifact for each selected "
            "configuration; incompatible identities fail before post-hoc fit."
        ),
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
            print(f"{path}{suffix}")
        return
    for path in targets:
        run_config(
            path,
            extra_configs,
            overwrite=args.overwrite,
            encoder_checkpoint=args.encoder_checkpoint,
        )


if __name__ == "__main__":
    main()
