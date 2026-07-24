# Tri-Modal Trusted-Fusion Experiments

This directory is the formal experiment catalog for the current API/Graph/
Manifest pipeline. Stable implementation defaults live in
`fusion/constants.py`; experiment YAML files declare only intentional protocol
differences.

The paper studies one problem: when the three modality branches have unequal
sample-level reliability, a fixed or equal-confidence fusion rule is not
appropriate. The method therefore follows one ordered chain:

1. estimate each branch's intrinsic correctness reliability (I1);
2. use those comparable reliabilities to route and estimate final-decision risk
   (I2);
3. control accepted malware false-negative risk on a disjoint calibration set
   (I3).

I1, I2, and I3 are not three independent classifiers. They are three stages of
one trusted-fusion decision pipeline.

## Final method

### Stage 1: clean branch learning

The primary method has three encoders and three binary branch heads: API,
Graph, and Manifest. It has no concatenation head and no `joint` branch.
All three branches are trained together on clean inputs. Before I1/I2 have
been fitted, available branches are fused with an alive-masked uniform mean:

```text
p_uniform = sum_m alive_m * p_m / sum_m alive_m
```

The Stage-1 objective is:

```text
L = L_uniform_fusion_CE
  + 0.25 * L_alive_uniform_branch_CE
  + 0.05 * L_EDL
```

The checkpoint score is the clean validation Macro-F1 of this neutral uniform
fusion. The selected artifact contains only the three encoders and branch
heads. It does not contain a fitted I1 calibrator, I2 router/risk head,
temperature, or I3 threshold.

Stage 1 is deliberately clean-only. Controlled
partial degradation is reserved for the frozen-encoder post-hoc protocol, so
augmentation cannot obscure whether I1/I2 add value.

### I1: intrinsic branch-correctness calibration

For each alive branch `m`, I1 estimates:

```text
r_m = P(branch m's argmax prediction is correct | its own opinion)
```

The deployed input contains exactly four branch-local quantities:

- evidential certainty `1 - K / sum(alpha_m)`;
- probability margin `top1(p_m) - top2(p_m)`;
- an optional predicted-malware class intercept;
- `alive_m`, used only as a hard availability mask.

The calibrated logit has a branch-specific intercept, non-negative certainty
and margin coefficients, and an optional signed predicted-class intercept.
The non-negative coefficients make reliability monotone in both continuous
confidence signals. The supervision target is branch correctness
`1{argmax(p_m) = y}`.

I1 does **not** consume perturbation type or strength, pre-degradation counts,
missing ratios, integrity heuristics, runtime coverage, other branch outputs,
or cross-modal disagreement. Perturbation identity is used only to select and
balance calibration rows; it is not a deployed feature.

Each branch is fitted on clean OOF rows plus that branch's controlled partial-
degradation OOF rows. A missing branch is assigned reliability zero through
the alive mask and is excluded from correctness fitting. The primary clean/
perturb objective mass is 0.50/0.50. Per-branch temperature-scaled confidence
is the simple matched-budget comparator.

### I2: reliability-conditioned routing and final-FN risk

I2 receives only branch probabilities, I1 reliabilities, and alive masks.
For each available branch it computes a reliability-weighted peer-consensus
conflict, then routes with:

```text
score_m = beta * logit(r_m) - lambda_m * conflict_m
pi = alive-masked softmax(score)
p_mix = sum_m pi_m * p_m
```

`beta` and every conflict scale are non-negative. The route is fitted by the
proper conditional-mixture NLL. There is no feature-embedding gate, free
residual router, subset oracle, per-row oracle, soft worst-group objective, or
Group-DRO state in the primary method.

After the deployable classification boundary is fixed from strict OOF mixture
predictions, I2 fits a separate monotone risk head for the aligned event
`malware predicted as benign`. Its three inputs are:

- reliability deficit `1 - sum_m pi_m r_m`;
- proximity to the fitted classification decision boundary;
- global reliability-weighted cross-modal conflict.

Branch probabilities and I1 reliabilities are detached while fitting I2, so
post-hoc objectives cannot modify the Stage-1 encoders or redefine I1.
The risk head outputs `u`; the primary acceptance score is exactly `1 - u`.
Clean and controlled-perturbation rows have an explicit configurable objective
prior (0.50/0.50 in the main cell, with 0.70/0.30 and 0.30/0.70 sensitivity
cells).

### I3: malware-FN conformal risk control

The classification boundary is selected first for Macro-F1 on strict OOF
post-hoc rows and then frozen. I3 does not change this classifier. On a
disjoint decision-calibration set, it selects an acceptance threshold over
`1 - u` subject to:

```text
(accepted_malware_false_negatives + 1)
----------------------------------------- <= alpha
       (number_of_malware + 1)
```

The primary `alpha` is 0.05. If no non-empty acceptance set is feasible, the
conservative fallback is reject-all.

This is an **expected conformal risk-control guarantee** under the stated
exchangeability assumptions. It is not a high-probability `1-delta` guarantee,
not a guarantee under arbitrary distribution shift, and not a guarantee for
FNR conditional only on accepted malware. Those quantities are reported as
empirical operating metrics without stronger claims.

## Validation and leakage-control lifecycle

The executable protocol never fits on its evaluation set: all learned modules,
classification thresholds, acceptance thresholds, perturbation choices, and
hyperparameters come from train or the declared validation roles. The
validation identities are frozen in
`labels/validation_roles_protocol_v1.json` and divided by package-isolation
group into:

1. 40% for Stage-1 checkpoint selection;
2. 35% for nested I1/I2 fitting and classification-boundary fitting;
3. 25% for the final I3 decision calibration.

The YAML representation is nested:
`calibration.validation_fraction=0.60` reserves the joint post-hoc/I3 holdout,
then `calibration.conformal_fraction=5/12` assigns 25% of the original
validation set to I3.

The 35% post-hoc subset uses five deterministic package-group folds.
Every view of one package remains in the same fold. Nested cross-fitting is
required: every row used to fit a downstream stage is predicted by upstream
modules that did not fit that row. Deployment modules are refitted only after
strict OOF rows and the fixed classification boundary have been produced.

To intentionally create a new role protocol:

```bash
python scripts/build_validation_roles.py \
  --validation-csv labels/val.csv \
  --output labels/validation_roles_protocol_v1.json \
  --seed 42 \
  --validation-fraction 0.60 \
  --decision-fraction-within-holdout 0.4166666666666667
```

Changing this role file changes the experiment protocol and requires rerunning
all affected experiments.

## Controlled evidence-availability protocol

The registered partial-degradation mechanisms are:

- API: `api_event_dropout`;
- Graph: `graph_sparsify`;
- Manifest: `manifest_permission_mask`.

Post-hoc fitting uses strengths 0.3, 0.5, and 0.7. Together with clean and the
three whole-modality missing endpoints, I2 sees 13 logical sources:
`1 + 3*3 + 3`. I1 uses only clean plus the affected branch's own partial views;
whole-modality missing rows supervise neither branch correctness nor a
fabricated quality ratio.

Formal evaluation uses the same three mechanisms at strengths 0.1, 0.3, 0.5,
0.7, and 0.9, plus clean and the three missing endpoints: 19 result cells in
total. These are controlled evidence-availability stress tests, not claims of
unseen attacks, semantic corruption, or external-domain generalization.

The registered curves are evaluation outputs and the code never feeds them
back into fitting. In this project history, however, results on the current
`test.csv` were inspected while the method was redesigned. That split must
therefore be described as a development/diagnostic test, not as an untouched
confirmatory test. A final unbiased paper claim requires a newly locked test
set or an independent external set that is evaluated only after the protocol
is frozen.

## Comparison protocol

Main representation/fusion baselines:

- API only;
- Graph only;
- Manifest only;
- API+Graph concatenation;
- tri-modal concatenation;
- fixed logit fusion;
- dense embedding gate adapted to these encoders.

Trusted-fusion baselines:

- **TMC-style adapted**;
- **ECML-style adapted**;
- **QMF-Energy component baseline**.

The phrases `TMC-style adapted` and `ECML-style adapted` are mandatory. These
cells preserve relevant objectives or fusion mechanisms under this
repository's APK modalities, encoders, split, missing-view handling, and
training budget; they are not strict reproductions of the original papers.

Dempster, cumulative subjective logic, log-pool, and the custom
conflict-weighted opinion rule are complete fusion-rule comparisons. Their
fusion rule is active during Stage 1, they train their own Stage-1 artifact,
and `selective_prediction.enabled` is false. They evaluate forced
classification and are not post-hoc-only I2 ablations.

The formal ablations are:

- module removal: exactly one complete no-I1 cell, learned I2, or I3;
- I1 atomic: evidential certainty, margin, or predicted-class intercept;
- I1 comparator: per-branch temperature-scaled confidence;
- I2 atomic: learned route prior, learned FN-risk head, route conflict, and
  risk conflict;
- I2 scenario-prior sensitivity: clean/perturb 0.70/0.30 and 0.30/0.70;
- I1 x I2 factorial: both on, only I1, only I2, and both off;
- I3: the same CRC rule ranked by learned risk, MSP, or deployed-class
  probability, plus fixed-coverage/conformal mechanism baselines.

Predictive entropy is only a numerical sanity check in this binary task because
it is rank-equivalent to MSP. It must not be presented as an independent
selective-prediction baseline.

## Experiment commands

Use the AutoDL path overlay for every AutoDL run:

```bash
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

python run.py seed_2024 seed_3407 \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

python run.py baselines trusted_baselines \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

python run.py module i1_atomic i1_comparator i2_atomic \
  i2_scenario_weights fusion_rules factorial_remaining \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

python run.py i3_mechanism \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

python run.py training_ablation appendix \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

Use `python run.py --list` to inspect the current catalog and
`python run.py <targets> --dry-run` before a long batch.

After the primary seed-42 run has produced diagnostics, freeze the natural
difficulty subsets on validation diagnostics and only then run their test
evaluations:

```bash
python scripts/build_natural_subset_csvs.py \
  --diagnostics results/tri_modal_robust/evidential_seed_42/42/gate_diagnostics.csv \
  --test-csv labels/test.csv \
  --out-dir labels/natural_subsets

python run.py natural_subsets \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

Main-table baselines must also be repeated with the registered seed overlays
when reporting mean and standard deviation:

```bash
python run.py baselines trusted_baselines \
  --extra-config \
    config/experiments/tri_modal_robust/_autodl_paths.yaml \
    config/experiments/tri_modal_robust/_seed_2024_overlay.yaml

python run.py baselines trusted_baselines \
  --extra-config \
    config/experiments/tri_modal_robust/_autodl_paths.yaml \
    config/experiments/tri_modal_robust/_seed_3407_overlay.yaml
```

An encoder checkpoint may be reused only when architecture, Stage-1 loss,
training data, PT build, loader/RNG protocol, and validation-role identities
are unchanged. Post-hoc I1/I2/I3 ablations normally satisfy that contract.
Representation, fusion-rule, and Stage-1 objective baselines do not.

```bash
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml \
  --encoder-checkpoint /absolute/path/to/best_encoder_selected.pt \
  --overwrite
```

## Manifest vocabulary preflight

After a split change, rebuild the vocabulary from the current train identities
and migrate only Manifest-owned PT fields. This preserves the expensive
API/Graph payload and does not reparse APKs. Do not run extraction, migration,
or training concurrently.

```bash
# Read-only audit
python scripts/migrate_manifest_vocab_pts.py \
  --train-csv labels/train.csv \
  --pt-dir /root/autodl-tmp/pts_all \
  --build-config config/build_pts.yaml \
  --vocab-out config/manifest_vocab.yaml \
  --manifest-jsonl-dir /root/autodl-tmp/pts_all/_manifest_jsonl \
  --audit-json manifest_migration_dry_run.json

# Apply only after the read-only audit reports complete coverage
python scripts/migrate_manifest_vocab_pts.py \
  --train-csv labels/train.csv \
  --pt-dir /root/autodl-tmp/pts_all \
  --build-config config/build_pts.yaml \
  --vocab-out config/manifest_vocab.yaml \
  --manifest-jsonl-dir /root/autodl-tmp/pts_all/_manifest_jsonl \
  --audit-json manifest_migration_apply.json \
  --apply
```

The formal run checks train/vocabulary provenance and the expected PT build
fingerprint before accepting the pool. This migration changes the experiment
identity, so old checkpoints are intentionally not supported.

## Reporting boundary

For I1 report per-branch correctness calibration and discrimination (Brier,
ECE, AUC/AP, and reliability-versus-accuracy diagnostics). For I2 report
forced-classification Macro-F1/AUC/AP, mixture NLL, routing diagnostics, and
threshold-aligned malware-FN risk calibration/ranking. For I3 report coverage,
accepted-malware recall, CRC-aligned accepted-FN risk among all malware,
conditional accepted-malware FNR as an empirical metric, and risk-coverage
curves.

Natural hard subsets are diagnostic analyses generated by
`scripts/build_natural_subset_csvs.py`. Reliability-imbalance and
high-conflict cutoffs are frozen on `val_selection`, whose identities were not
used to fit I1/I2, and only then applied to `test_clean`; test scores never
choose a cutoff. Branch disagreement
is label-free. The three "exactly one branch wrong" sets use test labels to
explain complementarity and must be reported explicitly as post-hoc diagnostic
sets, not deployment-detectable selection rules. Reliability imbalance may
stress I2 routing, but it cannot be used as an independent validation of I1
because I1 reliability defines that subset.
