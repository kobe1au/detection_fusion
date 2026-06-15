# Tri-modal Robust Fusion Experiment Plan

This plan is derived from the current implementation. It does not reuse the
deleted legacy i1/i2/i3 experiment route.

## Fixed Protocol

- Data: strict current schema-4 PTs and `labels/{train,val,test}.csv`.
- CSV membership is authoritative; extra unreferenced PT files are ignored,
  while missing/duplicate PTs still fail validation.
- Split isolation: sample ID and package name must not overlap across splits.
- Model input limit: at most 2048 API events per sample.
- Checkpoint selection: robust-composite macro-F1 on `val_selection`.
- Validation holdout: every runnable experiment uses the same deterministic
  package-isolated 50/50 split of Val.
- Post-hoc calibration: discount-fusion experiments with calibration enabled fit
  reliability and/or branch temperatures on clean `val_calibration`; non-discount
  baselines do not.
- Test: clean test plus degradation strengths 0.1, 0.3, 0.5, 0.7, and 0.9,
  and one run for each missing-modality scenario.
- Synthetic degradation may be used for training and robustness evaluation, but
  model evidence never reads `pert_*` metadata.

## Experiment Groups

Run commands:

```bash
python run.py final
python run.py baselines
python run.py i1
python run.py i2
python run.py i3
python run.py training_ablation
python run.py sensitivity
python run.py seed
python run.py paper --dry-run
```

`paper` contains 33 unique runs: all baselines, innovation/training ablations,
sensitivity analyses, and three final-method seeds. `seeds/seed_42.yaml` is the
final-method reference inside that group, so the root final config is not run
twice.

### Main Method

`observable_reliability_discount_fusion.yaml` is the publication-facing method:

1. observable monotonic branch-reliability calibration;
2. observable-integrity-conditioned cross-modal masked semantic reconstruction;
3. calibrated probability discount fusion with selective rejection.

### Baselines

Pure representation baselines disable final-method reconstruction, branch
auxiliary loss, post-hoc calibration, and rejection:

- API only;
- Graph only;
- Manifest only;
- API + Graph concatenation;
- tri-modal concatenation.

Fusion baselines retain the final method's representation training and replace
only the final fusion/calibration/rejection stage:

- fixed equal-weight logit fusion;
- confidence-weighted logit fusion;
- heuristic observable-reliability logit fusion;
- learned observable-evidence logit fusion.

All baselines use the same `val_selection` subset as the final method.

### Innovation I1

Observable monotonic reliability calibration:

- `no_reliability_calibration`: retain branch-temperature calibration but use
  raw observable integrity as base reliability;
- `integrity_alive_only`: remove cross-source support/conflict evidence and
  explicit relation discounts, retaining integrity, availability, and confidence;

### Innovation I2

Cross-modal masked security-semantic reconstruction:

- remove the reconstruction mechanism;
- mask-probability sensitivity at 0.05 and 0.30;
- reconstruction-weight sensitivity at 0.01 and 0.05.

Masks are sampled only during training. A selected target modality is
reconstructed from the other available, integrity-weighted modalities.

### Innovation I3

Calibrated probability discount fusion and rejection:

- remove branch-temperature calibration;
- remove explicit support discount;
- remove explicit conflict discount;
- remove entropy-margin confidence discount;
- remove hard alive masking;
- remove thresholded rejection;
- remove all post-hoc calibration while retaining raw probability discount;
- compare against learned observable-evidence logit fusion.

`no_support_discount` and `no_conflict_discount` remove only explicit
multipliers; the monotonic reliability calibrator can still use the
corresponding observable evidence.

### Training And Sensitivity

- remove synthetic training degradation;
- remove branch auxiliary supervision;
- replace integrity-weighted branch auxiliary supervision with ordinary branch
  auxiliary supervision;
- treat unavailable relations as full positive support;
- use product instead of minimum acceptance aggregation;
- target selective coverage of 0.80 and 0.95.

The acceptance-aggregation and coverage sensitivities are eval-only runs that
reuse the calibrated `seed_42` checkpoint and refit only the rejection threshold
on `val_calibration`. Run `python run.py seed` before `python run.py sensitivity`;
the `paper` group already enforces this order.

### Multi-seed

The final method is repeated with seeds 42, 2024, and 3407. The validation
holdout split remains fixed with `calibration.split_seed=42`.

## Reporting

Primary classification metrics:

- macro-F1, accuracy, ROC-AUC, AP;
- clean and every robustness scenario;
- mean and standard deviation over the three final seeds.

Calibration and selective metrics:

- Brier score, ECE-10, confidence-accuracy gap;
- coverage, selective risk, selective macro-F1, and AURC.

Do not describe entropy/margin as calibrated uncertainty. It is a confidence
proxy. Do not describe support/conflict-only ablations as removing those signals
from the reliability calibrator unless the config explicitly disables relation
evidence.

Before training, validate observable-signal distributions and degradation trends:

```bash
python scripts/diagnose_observable_signals.py --pt-dir D:/pts_robust/test --csv labels/test.csv --out-dir results/signal_diagnostics --split test --fail-on-check-error
```

Publication gaps not represented by runnable YAMLs yet:

- external published malware-detection baselines;
- paired real Obfuscapk evaluation. `config/extract_obfuscapk.yaml` is aligned
  with the current PT protocol, but the paired APK/PT/CSV data is not present.
