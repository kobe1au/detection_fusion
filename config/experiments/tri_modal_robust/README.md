# Tri-Modal Robust Experiments

This directory contains the pre-registered experiment catalog for the current
I1/I2/I3 pipeline. Stable defaults live in `fusion/constants.py`; YAML files
declare only meaningful experimental differences.

## Current method identity

- **I1 — intrinsic branch-correctness calibration.** Each alive API, Graph, or
  Manifest branch estimates `P(its own argmax is correct)` using an ordered
  `clean competence - degradation penalty` model. Phase one fits competence
  from a monotone margin spline plus a predicted-class intercept. Phase two
  freezes competence and learns a bias-free non-negative penalty from effective
  quality deficit and fold-local class-conditional diagonal-Mahalanobis tail
  excess, so degradation can never increase reliability. Each I1 fit estimates
  its class references only from clean rows in
  that fit's training selection; at inference, the branch's own predicted class
  selects the reference. The main method does not reuse perturbation identity or
  severity, target labels, other-modality outputs, cross-modal disagreement, or
  EDL certainty in I1, and does not multiply the calibrated probability by a
  second visibility modifier. Branch-local completeness views and the API,
  Graph, and Manifest single-branch semantic views supervise only the affected
  branch. Clean/completeness/semantic objective masses are fixed at
  0.50/0.25/0.25; `all_semantic_corrupted` remains I2-only.
- **I2 — conditional routing plus threshold-aligned malware-FN risk.** `pi` is a conditional
  distribution over available branches. Its common-scale prior is the positive
  scaled log-odds of I1 correctness, and the main route is fitted only with the
  proper mixture NLL. Branch conflict is normalized Jensen-Shannon divergence
  from a reliability-weighted leave-one-out peer consensus; dead/no-peer cases
  add no artificial conflict. On perturbation rows, that same mixture NLL is
  reduced by a perturb-type/strength-balanced mean plus a stateless entropic
  soft worst-group term; the
  latter changes group aggregation, not the per-row supervision target. All
  row/subset-oracle weights are zero in the primary method. The detached
  source-level subset oracle is retained only as an optional add-on experiment,
  not as part of the deployed method or primary objective.
  Encoder training uses an alive-only
  uniform route; the log-odds prior is activated only after I1 fitting. `u` is
  fitted only on samples that the fixed deployable classifier predicts benign,
  with target `1` exactly for malware false negatives. On the predicted-malware
  side that event is impossible, so its deployed risk is fixed to zero. The
  final classifier uses temperature-scaled
  `p_mix = sum_i pi_i p_i`; `u` is only a rejection ranking score. Clean and
  perturbed calibration families receive an explicit configurable objective
  prior (main cell 0.5/0.5, sensitivity cells 0.7/0.3 and 0.3/0.7). Three
  pairwise completeness views are included symmetrically. Within the perturb
  side, the main route combines the mechanism-balanced mean with a stateless
  entropic soft worst-group risk; it does not maintain call-order-dependent
  Group-DRO state. I3 then
  calibrates acceptance against this same thresholded malware-FN event on a
  disjoint decision set.
  There is no unknown semantic class and no known-mass budget.
- **I3 — malware-FN selective risk control.** The binary
  operating threshold is an unconstrained macro-F1 utility boundary fitted
  first and fixed as the classifier boundary that defines I2's FN-risk target;
  it is not part of I3. A disjoint decision-calibration subset then fits
  the acceptance threshold using `1-u` as the primary score. Its bounded loss
  is `1{accepted and predicted benign}` among malware, so the empirical
  denominator is **all malware**, not accepted malware. The finite-sample rule
  `(accepted_FN + 1)/(n_malware + 1) <= alpha` provides expected CRC under the
  stated exchangeability assumptions. It does not provide a high-probability
  guarantee or a guarantee for FNR conditional on accepted malware; that
  operational conditional FNR is reported separately.

Classical Dempster, cumulative subjective logic, log-pool, and the custom
conflict-weighted opinion rule are explicit I2 comparison cells. They never
replace the routed main path silently.

## Validation lifecycle

The validation set is split by canonical package-isolation group into:

1. 50% for checkpoint selection;
2. 25% for post-hoc model fitting and the classification threshold;
3. 25% for the final I3 decision rule.

The post-hoc quarter uses five deterministic package-group folds. Every
clean/degraded view of one package stays in the same fold. For each outer route
holdout, inner I1 fits exclude both the outer holdout and the row being predicted;
this produces strict OOF route predictions for the risk head and temperature.
Every inner, outer, and deployment I1 fit re-estimates its class-conditional
diagonal-Mahalanobis references from only the clean rows in its own training
selection before building branch-local calibration features.
The binary classification cutoff is first selected on the raw log-odds of clean
upstream-OOF route outputs. Risk is fitted on strict OOF-route outputs using that
fixed cutoff, after which deployable I1 and route modules are refitted on all
eligible post-hoc identities. The scalar temperature is monotone and the raw
cutoff is mapped through it exactly; the full-data temperature fit therefore
cannot change which samples are predicted malware or benign.

Unknown or failed calibration samples stop the run. I1, route, and risk stages
use deterministic full-batch LBFGS with a strong-Wolfe line search, explicit
effective-parameter regularization for I2, best-state restoration, and scale
guards. Every fold starts
from the same parameter snapshot, and every OOF cache row must be written exactly
once before a pipeline checkpoint can be saved.

The numerical budgets above do not rerun the complete fusion stack at every
optimizer step. Each fit materializes its frozen tensors once: I1 reuses its
branch-local feature matrix and evaluates three packed branch heads, the route
reuses prepared three-opinion inputs, and the risk stage reuses its
five-dimensional feature matrix. Subsequent steps run only the corresponding
small calibrator, route kernel, or logistic head. All perturbation sources share
one boundary-preserving evaluation worker pool, while their source identities
and objective masses remain separate. Stage summaries report full decision
forwards separately from lightweight forwards, together with cache and total
post-hoc wall time.

## Encoder-stage performance contract

The encoder-selection stage keeps the original candidate epochs, clean
Macro-F1 checkpoint score, patience, optimizer, and augmentation stream. Its
runtime path is optimized without changing those choices: PT tensor storages
are memory-mapped read-only, already budgeted Graph batches carry a CPU-checked
contract so the encoder does not repeat no-op GPU truncation, and materialized
Manifest/code relations are not recomputed as fallbacks. Per-epoch checkpoint
validation uses a lean inference profile that computes only the classification
metrics consumed by checkpoint selection; complete diagnostics are still
computed after the best encoder is restored. Each epoch log reports separate
`train_wall_seconds` and `val_wall_seconds` values so remaining I/O or encoder
cost can be measured rather than inferred from total runtime.

## Experiment groups

- `python run.py final`: primary seed-42 method.
- `python run.py seed`: primary method for seeds 42, 2024, and 3407.
- `python run.py baselines`: representation, fixed-logit, and the adapted dense
  embedding-gated late-fusion baseline. The latter is not a strict sparse-MoE
  or Shazeer reproduction.
- `python run.py trusted_baselines`: TMC-style adapted, ECML-style adapted, and
  the QMF-Energy component baseline.
- `python run.py module`: complete I1, learned-I2, and I3 removals. I3 removal
  retains the shared macro-F1 classifier boundary and disables only selective
  malware-FN acceptance control.
- `python run.py i1_atomic`: five single-axis I1 ablations: model visibility,
  embedding density, margin, predicted-class intercept, and learned calibration.
  Every feature cell keeps the same clean and single-branch semantic supervision
  rows; `no_embedding_density` zero-masks only that slot. The class-intercept
  cell does not remove the predicted-class selector required by class-conditional
  density.
- `python run.py i1_comparator`: matched-budget per-branch temperature-scaling
  comparator. It is a baseline replacement, not an atomic feature ablation.
- `python run.py i2_atomic`: learned-`pi`, learned-`u`, route-conflict, and
  risk-conflict ablations.
- `python run.py i2_rules`: four fixed fusion-rule comparisons.
- `python run.py i2_scenario_weights`: clean/perturb objective-prior sensitivity
  at 0.7/0.3 and 0.3/0.7 (the primary cell is 0.5/0.5).
- `python run.py i2_robust_route`: compare the primary mixture-NLL plus soft
  worst-group route against the optional source-subset-oracle add-on and the
  `rho=0` mean-only reduction, including their joint add-on/mean-only cell;
  independently remove the three pairwise-completeness views and compare
  perturb-type balancing with the five-family taxonomy. These are validation-only
  selection cells; none should be chosen from test perturbation results.
- `python run.py i2_prior_beta_sensitivity`: fixed odds-prior beta 0.5/1/2 for
  the `prior_only` ablation. Beta remains non-trainable in all three cells;
  learned routing retains its separately fitted positive beta. This comparator
  is deliberately a fixed odds rule, not a fitted soft router: because odds
  diverge as reliability approaches one, the beta=1 cell can behave like a
  near-hard selector and is reported with that exact interpretation.

Protocol warning: the existing robust-test summaries were already inspected
while designing the soft-worst and pairwise-view revisions and the optional
source-subset-oracle add-on.
Those rows are therefore development diagnostics, not an untouched final test.
Freeze the new configuration using validation-only robustness cells, then report
one new held-out or external confirmatory test; do not tune the three route
hyperparameters again from the same test table.

- `python run.py i3_acceptance_score`: compare `1-u`, pre-trust conflict,
  trusted conflict, product, strict MSP, and deployment-class probability under
  the same malware-FN CRC rule and the same `alpha`. The MSP/probability cells
  change only `threshold_score`; they do not replace CRC with fixed coverage.
- `python run.py i3_mechanism`: conformal controls plus threshold baselines for
  deployment-class probability, strict MSP `max(p, 1-p)`, routed evidential
  mixture certainty, and learned model acceptance. Deployment-class probability
  is threshold-aware; it is deliberately not labelled MSP when the fitted class
  cutoff is not 0.5. The optional `predictive_entropy_threshold` target implements
  normalized certainty `1-H(p)/log(2)` as a sanity check. In this binary task it
  is a monotone transform of MSP, so it must not be reported as an independent
  ranking baseline or interpreted as additional evidence.
- `python run.py mechanism`: all atomic I1/I2/I3 checks.
- `python run.py factorial_remaining`: non-primary cells of the I1×I2 matrix
  after `final` has completed.
- `python run.py appendix`: small sensitivity runs.

The I1×I2 factorial has exactly four cells: both on, I1 off, learned I2 off,
and both off. Turning learned I2 off fixes `pi` to the I1 reliability prior and
uses the deterministic reliability-deficit risk; it does not change I1.

The fitted unconstrained macro-F1 classification boundary is a shared operating
point, not part of I3. The atomic `no_i3_decision_layer.yaml` experiment keeps
that exact boundary and disables only selective acceptance. There is therefore
no classification×risk 2×2 in the formal plan: toggling the classifier would
change the task protocol rather than isolate I3.

## Literature-baseline wording

Always report the two evidence baselines as **TMC-style adapted** and
**ECML-style adapted**. They retain the relevant evidential objectives and
fusion mechanisms under this repository's APK modalities, encoders, missing-
view handling, data split, and training budget. They are not strict
reproductions of the original papers. Report `qmf_energy` only as the
**QMF-Energy component baseline**.

## Recommended execution order

Use the AutoDL path overlay for every command. Before the first formal run,
rebuild the Manifest vocabulary from the current train split and migrate only
the Manifest-owned PT fields. This preserves the expensive API/Graph tensors
and does not parse APKs. Run the first command without `--apply`, inspect its
coverage/diff report, then repeat it with `--apply`:

```bash
# Read-only preflight
python scripts/migrate_manifest_vocab_pts.py \
  --train-csv labels/train.csv \
  --pt-dir /root/autodl-tmp/pts_all \
  --build-config config/build_pts.yaml \
  --vocab-out config/manifest_vocab.yaml \
  --manifest-jsonl-dir /root/autodl-tmp/pts_all/_manifest_jsonl \
  --audit-json manifest_migration_dry_run.json

# Atomic Manifest-only migration and complete post-audit
python scripts/migrate_manifest_vocab_pts.py \
  --train-csv labels/train.csv \
  --pt-dir /root/autodl-tmp/pts_all \
  --build-config config/build_pts.yaml \
  --vocab-out config/manifest_vocab.yaml \
  --manifest-jsonl-dir /root/autodl-tmp/pts_all/_manifest_jsonl \
  --audit-json manifest_migration_apply.json \
  --apply
```

Do not run extraction, migration, or training concurrently. Formal training
checks the train CSV/vocabulary hashes and every PT's sample identity and
Manifest provenance before accepting the pool. After migration, all checkpoints
must be retrained from scratch.

```bash
# 1. Primary method
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 2. Other primary seeds (avoid rerunning seed 42)
python run.py seed_2024 seed_3407 \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 3. Common and literature-style baselines
python run.py baselines trusted_baselines \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 3a. Repeat every main-table baseline at seed 2024
python run.py baselines trusted_baselines \
  --extra-config \
    config/experiments/tri_modal_robust/_autodl_paths.yaml \
    config/experiments/tri_modal_robust/_seed_2024_overlay.yaml

# 3b. Repeat every main-table baseline at seed 3407
python run.py baselines trusted_baselines \
  --extra-config \
    config/experiments/tri_modal_robust/_autodl_paths.yaml \
    config/experiments/tri_modal_robust/_seed_3407_overlay.yaml

# 4. Module, atomic, rule, and remaining factorial comparisons
python run.py module mechanism factorial_remaining \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 5. Optional sensitivities
python run.py training_ablation appendix \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

`paper_main` includes all three primary-method seeds but only the seed-42
baseline configs. A formal mean/std main table therefore also
requires the two baseline-overlay commands above. Mechanism ablations remain
seed-42 controlled comparisons unless they are explicitly repeated with the
same two seed overlays.

`python run.py all` is a catalog, not a from-scratch schedule: eval-only runs
need their corresponding newly trained checkpoint, and natural-subset runs
need freshly generated manifests.

For forced-classification I1/I2 comparisons, report the common 0.5-boundary
macro-F1, AUC, and AP. Report the fitted classification threshold only with I3.
For I1 report per-branch Brier/ECE/AUC/AP and reliability–accuracy gap. For I2
report mixture NLL plus risk Brier/ECE/AUC/AP and the target-aligned
`malware_fn_risk_aurc`. Keep generic classification-error AURC as a secondary
MSP-comparable selective-classification diagnostic; it is not the primary score
for the FN-only `u` head. For I3 report coverage, accepted-
malware recall, FNR conditional on accepted malware, the CRC-aligned accepted-FN
risk among all malware, risk–coverage curves, generic AURC, and
`malware_fn_risk_aurc`. Both AURCs are computed only
within the hard-eligible population (at least one branch actually available),
with equal-score ties admitted atomically. Always report
`selective_max_achievable_coverage` beside AURC so that this conditional ranking
metric is not mistaken for coverage over all test samples. Hard-ineligible
samples are rejected by threshold/CRC modes; conformal mode assigns the full
two-class prediction set, which is likewise non-singleton and therefore
rejected without breaking set-coverage semantics.

Natural subsets must be rebuilt from the new seed-42 diagnostics before their
eval-only configs are run:

```bash
python scripts/build_natural_subset_csvs.py \
  --diagnostics results/tri_modal_robust/evidential_seed_42/42/gate_diagnostics.csv \
  --test-csv labels/test.csv \
  --out-dir labels/natural_subsets
python run.py natural_subsets \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```
