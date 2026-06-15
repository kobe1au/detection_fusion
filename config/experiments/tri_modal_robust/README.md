# Tri-modal Robust Fusion Experiments

This directory contains the clean experiment plan for the robust API + Graph + Manifest framework.

## Experiment Routes

- `observable_reliability_discount_fusion.yaml`: publication-facing final method.
- `ablations/`: ablations for the final observable-reliability method.
- `final_seed/`: multi-seed runs for the final observable-reliability method.
- `i1/`, `i2/`, `i3/`, and `full/`: legacy development routes for the earlier gate-based method.
- `seed/`: legacy gate-based multi-seed overrides; use `run.py legacy_seed` only for comparison.
- `tune/`: sensitivity checks for innovation-related parameters. Do not mix these with the main ablation tables.

## Recommended Order

Run the core method first:

```bash
python run.py final
python run.py final_ablation --dry-run
python run.py seed --dry-run
```

The helper runner can also select legacy grouped experiments:

```bash
python run.py main --dry-run
python run.py i1 --dry-run
python run.py i2 --dry-run
python run.py i3 --dry-run
python run.py i2,i3 --dry-run
python run.py seed --dry-run
```

`run.py` deliberately excludes `tune/` configs. Use the Optuna driver for tuning so test
evaluation cannot be mixed into parameter selection.

Then run innovation-specific ablations:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/i1/api_graph_concat.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/tri_modal_concat.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i1/reliability_gate.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i2/no_consistency.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/consistency_evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/conflict_evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/evidence_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/loss_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/semantic_reconstruction_only.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i2/evidence_plus_loss.yaml

python -m fusion.train --config config/experiments/tri_modal_robust/i3/fixed_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/confidence_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/reliability_gate.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_no_alive_mask.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_no_prior.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/i3/learned_gate_with_prior.yaml
```

After the best setting is confirmed, run:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/seed/seed_42.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/seed/seed_2024.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/seed/seed_3407.yaml
```

## Stage-wise Optuna Tuning

Optuna tuning uses representative robust validation for checkpoint selection and does not
load or evaluate the test split. Run the stages in order:

The fixed robust-validation checkpoint score is a weighted macro-F1 average:

- clean validation: `0.40`
- API+Graph degraded at strength `0.5`: `0.25`
- Manifest degraded at strength `0.5`: `0.15`
- all modalities degraded at strength `0.5`: `0.10`
- API, Graph, and Manifest missing: `0.0333` each

These scenarios and weights must be frozen before examining final test results. Changing
them after observing test performance would make the final test no longer independent.

Use a new output directory and study name for every protocol version. The study stores a
configuration fingerprint and rejects incompatible resumed trials.

```bash
python scripts/tune_robust_optuna.py --stage i2 --trials 25 \
  --study-name robust_v2_i2 --output-dir results/optuna/robust_v2

python scripts/tune_robust_optuna.py --stage i3 --trials 25 \
  --study-name robust_v2_i3 --output-dir results/optuna/robust_v2 \
  --config config/experiments/tri_modal_robust/tune/optuna_base.yaml \
  results/optuna/robust_v2/best_i2_override.yaml

python scripts/tune_robust_optuna.py --stage aug --trials 9 \
  --study-name robust_v2_aug --output-dir results/optuna/robust_v2 \
  --config config/experiments/tri_modal_robust/tune/optuna_base.yaml \
  results/optuna/robust_v2/best_i2_override.yaml \
  results/optuna/robust_v2/best_i3_override.yaml
```

The augmentation stage is an exact 3-by-3 grid over perturbation probability and strength
profile, so it has nine unique trials. The i2 search includes
`cross_source_consistency_weight=0`; this allows the study to reject the cross-source
loss if it does not improve representative robust validation. Semantic reconstruction
and cross-source consistency are separate loss terms and must be reported separately.

Tuning and final training use the same 60-epoch budget, early-stopping rule, deterministic
mode, and robust-composite checkpoint metric. The only intended difference is that tuning
does not load or evaluate test data.

Use one seed during broad search. Do not use another Optuna search as a substitute for
multi-seed confirmation. After all three stages are fixed, train the exact selected
configuration with the three seed overrides:

```bash
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml config/experiments/tri_modal_robust/seed/seed_42.yaml results/optuna/robust_v2/best_i2_override.yaml results/optuna/robust_v2/best_i3_override.yaml results/optuna/robust_v2/best_aug_override.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml config/experiments/tri_modal_robust/seed/seed_2024.yaml results/optuna/robust_v2/best_i2_override.yaml results/optuna/robust_v2/best_i3_override.yaml results/optuna/robust_v2/best_aug_override.yaml
python -m fusion.train --config config/experiments/tri_modal_robust/full/ours.yaml config/experiments/tri_modal_robust/seed/seed_3407.yaml results/optuna/robust_v2/best_i2_override.yaml results/optuna/robust_v2/best_i3_override.yaml results/optuna/robust_v2/best_aug_override.yaml
```

The seed override must appear before the generated best-parameter overrides because each
seed config inherits `full/ours.yaml`.

After selecting the final parameters, run the complete test protocol from `full/ours.yaml`
instead of `optuna_base.yaml`:

```bash
python scripts/make_post_optuna_configs.py \
  --tag robust_v2 \
  --best-i2 results/optuna/robust_v2/best_i2_override.yaml \
  --best-i3 results/optuna/robust_v2/best_i3_override.yaml \
  --best-aug results/optuna/robust_v2/best_aug_override.yaml

python run.py post_optuna/robust_v2/full --dry-run
python run.py post_optuna/robust_v2/full
```

The generated `post_optuna/<tag>` configs lock the safe override order:
base method, best selected parameters, then the final ablation override. This prevents
best gate parameters from overwriting fixed/reliability gate ablations.
Use the generated configs for the final paper tables:

```bash
python run.py post_optuna/robust_v2/i1 --dry-run
python run.py post_optuna/robust_v2/i2 --dry-run
python run.py post_optuna/robust_v2/i3 --dry-run
python run.py post_optuna/robust_v2/full --dry-run
python run.py post_optuna/robust_v2/seed --dry-run

python run.py post_optuna/robust_v2/i1
python run.py post_optuna/robust_v2/i2
python run.py post_optuna/robust_v2/i3
python run.py post_optuna/robust_v2/full
python run.py post_optuna/robust_v2/seed
```

The original `i1/`, `i2/`, `i3/`, and `full/` configs remain useful for development
and sanity checks.  They are not the frozen post-tuning protocol once Optuna has been
used.

## External-Style Reference Baselines

Use these as reference implementations, not claims of exact paper reproduction.
They provide classical/static comparison points for the final method:

```bash
python scripts/train_static_baselines.py \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
  --out-dir results/static_reference_baselines \
  --run-test \
  --robust-test
```

The script includes Drebin-style sparse static features, MaMaDroid-style API-type
transition features, API bag-of-words, Manifest-only, and tri-modal static linear
baselines. Report them separately from the internal neural ablations.

## Real Failure Slices

Build quality/failure slice CSVs before final robustness evaluation:

```bash
python scripts/build_real_failure_slices.py \
  --config config/experiments/tri_modal_robust/base_tri_modal_robust.yaml \
  --splits val test \
  --out-dir results/robust_slices \
  --extra-eval-yaml results/robust_slices/extra_eval_slices.yaml
```

Then evaluate the selected checkpoint/method on those slices by appending the generated
override:

```bash
python -m fusion.train --config \
  config/experiments/tri_modal_robust/post_optuna/robust_v2/full/full_ours.yaml \
  results/robust_slices/extra_eval_slices.yaml
```

These slices are the main evidence for real extractor failures: low API quality, low
graph quality, low API-Graph alignment, Manifest parse failures, and partial multi-DEX
failures.

## Calibration Diagnostics

Every neural evaluation now reports `brier`, `ece_10`, `mean_confidence`, and
`confidence_accuracy_gap` in `summary.yaml`. Per-sample `gate_diagnostics.csv` also
contains final confidence and correctness, which supports calibration and gate-weight
correlation plots.

The publication-facing trustworthy fusion route is:

```bash
python run.py final
```

`final` resolves to `observable_reliability_discount_fusion.yaml`. It keeps the PT schema
unchanged and adds three runtime/model-level mechanisms:

- branch-specific monotonic reliability calibration from observable parsing evidence;
- observable-integrity-conditioned masked security-semantic reconstruction regularization;
- calibrated probability discount fusion with conflict applicability masks and
  validation-fitted selective rejection.

The reliability calibrator uses observable parsing evidence only. Entropy/margin
confidence proxies are applied later by discount fusion; they are not calibrator inputs.
Masked semantic reconstruction is conditioned by observable integrity and availability,
not by the post-hoc calibrated reliability outputs.

The validation CSV is deterministically separated into disjoint `val_selection` and
`val_calibration` subsets. Checkpoint selection and robust validation use only
`val_selection`; after model selection, only the monotonic calibrators and branch
temperatures are fitted on `val_calibration`. Its acceptance-score quantile determines
the rejection threshold, which is persisted in the checkpoint. Test and robustness
summaries report `coverage`, `selective_risk`, `selective_macro_f1`, and `aurc`.
Because `val_calibration` is used both to fit calibration parameters and to choose the
rejection threshold, metrics reported on that subset are diagnostic only. Publication
claims about calibration and selective prediction must use the untouched test split,
external sets, or a stricter cross-fitted protocol.

## Notes

The default data paths target the AutoDL layout:

- train pt: `/root/autodl-tmp/pts/train`
- val pt: `/root/autodl-tmp/pts/val`
- test pt: `/pts/test`
- labels: `labels/{train,val,test}.csv`

If these paths change, update only `base_tri_modal_robust.yaml`.

The main gate uses observable post-extraction integrity, support/conflict, and raw-alive
signals. Synthetic `pert_*` values are diagnostics only. The former
`full/ours_oracle_perturbation_evidence.yaml` entry is retained as a deprecated diagnostic
config and no longer exposes perturbation strength to the model.

Synthetic degradation augmentation remains enabled during training and robustness
evaluation. The defensible claim is that fusion decisions never read synthetic
perturbation labels or `pert_*` oracle metadata, not that training uses no synthetic
degradation.

Unavailable cross-modal relations do not apply support/conflict discount factors. In the
monotonic calibrator they contribute no positive relation support
(`missing_relation_support=0.0`), which is distinct from treating them as conflict.

The final config currently consumes at most 1024 API events per sample and requires the
observable schema, while retaining legacy PT compatibility. Increase the sequence limit
or require PT schema 4 only after auditing the actual formal PT files; the current direct
extractor is configured for at most 1024 events per DEX, so a blanket claim of 2048 events
would not be justified by configuration alone.

## Observable Reliability Schema

The current main evidence is built from `observable-v1` extraction metadata. It separates:

- extraction integrity: `api_integrity`, `graph_integrity`, `manifest_integrity`, `code_integrity`;
- semantic support/conflict: `api_graph_anchor_support`, `manifest_code_support`,
  `manifest_to_code_conflict`, `code_to_manifest_conflict`;
- raw availability: `api_alive`, `graph_alive`, `manifest_alive`.

`api_graph_anchor_support` is an API-to-call-graph anchor coverage signal, not an
independent cross-modal consistency score. Main evidence never reads `pert_*`.

An empty-but-successfully-parsed modality is not automatically incomplete. When
the saved raw count is also zero, integrity remains high and the corresponding
`*_alive` signal is zero. If raw content was observed but kept content is lost
(for example API truncation), integrity decreases.

Formal runs use `data.strict_observable_schema=true` and reject legacy PTs missing the
saved `observable-v1` fields. Compatibility runs may set it to false, but the fallback is
logged and must not be mixed into formal results.

Build strict tri-modal PT files with:

```bash
python scripts/build_tri_modal_pts_direct.py --config config/extract_tri_model.yaml
```

Signal diagnostics:

```bash
python scripts/diagnose_observable_signals.py \
  --pt-dir /path/to/pts \
  --csv /path/to/labels.csv \
  --split test \
  --out-dir results/observable_diagnostics \
  --strict-observable-schema
```

The diagnostic command writes distribution, trend, label-correlation, and output-check
CSVs. Add `--fail-on-check-error` in automated runs to reject missing columns, non-finite
values, out-of-range signals, invalid quantiles, or violated declared trend expectations.

Because the current test split has already been inspected during development, publication
claims require a newly locked final test or an external real-obfuscation/failure set.
