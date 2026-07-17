# Tri-Modal Robust Experiments

This directory keeps the formal experiment plan intentionally small.
Most stable model constants live in `fusion/constants.py`; YAML files only
express the conceptual experiment differences.

Main groups:

- `python run.py final`: full method with seed 42.
- `python run.py seed`: full method with seeds 42, 2024, and 3407.
- `python run.py baselines`: internal representation/fusion baselines.
- `python run.py trusted_baselines`: adapted recent trusted-fusion baselines
  under the same APK encoders, data split, and training budget.
- `python run.py module`: three innovation-level removals for I1/I2/I3. The
  no-I1 run retains the global router but removes calibrated reliability and
  visibility inputs while keeping integrity-weighted encoder supervision fixed;
  the no-I2 run retains I1 and the routed evidential-opinion path but fixes its
  weights to the exact reliability prior; the no-I3 run evaluates the main
  checkpoint without classification-threshold fitting or rejection.
- `python run.py mechanism`: atomic mechanism checks for I1/I2/I3.
- `python run.py i2_atomic`: prior-only, no-unknown, no-disagreement,
  posthoc-only, and encoder-only router controls. Classical Dempster,
  cumulative, log-pool, and ECML-style substitutions remain under `i2_rules`.
- `python run.py i1_i2_2x2`: factorial I1 reliability x I2 routing matrix.
- `python run.py i3_2x2`: factorial classification-threshold x malware-risk
  rejection matrix, reusing the same seed-42 prediction checkpoint.
- `python run.py factorial_remaining`: the six non-primary cells from both
  2x2 matrices; use this after `final` to avoid rerunning seed 42.
- `python run.py external`: Obfuscapk eval-only sets using the seed-42 checkpoint.
- `python run.py appendix`: small appendix sensitivities only.

Method-identity warning:

- The proposed I2 path is `combination: routed` in both encoder training and
  evaluation. It does not switch to a classical evidence rule at any stage.
- The proposed I3 path is `selective_prediction.mode: risk_control` with a
  malware false-negative target. Conformal variants are explicit comparison
  experiments under `ablations/i3`; they are not the proposed method.
- Thesis drafts, figures, or result files created before this method identity
  was fixed must not be used to describe the current implementation.

The trusted-fusion baselines are controlled adaptations rather than exact
reproductions on their original datasets. In particular, `qmf_style_adapted`
uses QMF's official detached energy-weighted late-fusion rule and fixed
temperature, while retaining this repository's common branch-training
objective instead of QMF's task-specific confidence-ranking loss. Report these
methods as `*-style adapted` in the thesis and state the shared-encoder protocol.

Paper mapping:

- I1: calibrated branch-correctness reliability. Observable extraction and
  relation evidence is combined with branch evidential certainty by a monotone
  calibrator, so API, Graph, and Manifest reliability share the same semantics:
  estimated probability that the frozen branch prediction is correct.
- I2: global opinion routing. During encoder training, a lightweight router
  jointly observes observable integrity, branch opinions, availability, and
  prediction disagreement. On the independent post-hoc subset, the same router
  is refined using the three calibrated branch-correctness probabilities. A
  separately reported relative effective-integrity
  factor constrains the route when model-visible evidence falls below its clean
  calibration reference. The router allocates mass to API, Graph, Manifest, or
  a residual error/abstention outcome; this residual is not a third semantic
  class. Its total known-branch mass cannot exceed the reliability-derived
  prior, so learned routing may move prior mass toward abstention but cannot
  manufacture extra trust. The router participates in encoder training and is
  subsequently refined on the independent post-hoc split. Classical evidence
  rules are used only as explicitly named comparison methods.
- I3: constrained classification and malware false-negative risk control. The
  post-hoc validation subset first selects the malware decision threshold that
  maximizes macro-F1 while keeping malware recall at or above 90%. With that
  classifier fixed, the disjoint decision-calibration subset selects the lowest
  acceptance threshold whose finite-sample corrected expected risk of accepting
  a malware sample as benign meets the configured target. This risk is divided
  by all malware calibration samples, not only accepted malware samples.

I3 comparisons use the same seed-42 checkpoint and the same held-out
calibration subset: no rejection, a maximum-probability threshold, an
evidential-uncertainty threshold, marginal and class-conditional split
conformal prediction, conflict-augmented conformal prediction, and the main
malware false-negative risk-control rule.

Validation protocol for formal runs with selective prediction:

- 50% of validation groups select the checkpoint.
- 25% fit post-hoc reliability/routing parameters and the constrained malware
  classification threshold when needed.
- 25% fit the selective decision rule. Threshold, conformal, and risk-control
  variants use this same final subset, so their comparison has an equal budget.

The post-hoc subset is evaluated once in its clean form and under single- and
all-modality degradation at strengths 0.1, 0.3, 0.5, 0.7, and 0.9, plus three
missing-modality views. Reliability and probability calibration use only clean
rows. Each router objective balances clean rows equally with one degraded or
missing scenario family, so the perturbation-grid size does not set an implicit
deployment prior. Checkpoint selection and the disjoint decision-calibration
subset never consume these transformed views.

Non-selective baselines use the same 50% checkpoint-selection subset and leave
the remaining validation groups unused for model selection.

For the I1+I2 forced-classification table, compare every method at the common
0.5 decision boundary using `fixed_0_5_macro_f1`, AUC, and AP. The
validation-fitted malware threshold belongs to I3 and is reported only with
the risk-control experiments. Brier score and ECE may be reported as
full-pipeline calibration outcomes, but they do not isolate I1/I2 because the
main routed method fits one final temperature while the plain baselines do not.
Do not attribute that cross-method calibration difference to routing alone
unless the same final calibration is fitted for every baseline.

## Recommended execution order

Use the AutoDL path overlay for every command below.

```bash
# 1. Primary full method. Inspect this run before launching the full matrix.
python run.py final --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 2. Remaining full-method seeds. Do not run the `seed` group after `final`,
# because that group also contains seed 42.
python run.py seed_2024 seed_3407 --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 3. Common representation/fusion baselines and recent trusted-fusion
# adaptations (TMC-style, QMF-style, and ECML-style).
python run.py baselines trusted_baselines \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 4. Innovation-level, atomic, and the six remaining factorial cells. The trusted-fusion
# Dempster/cumulative/log-pool rule substitutions remain mechanism ablations;
# they are not substitutes for the independent trusted-fusion baselines above.
python run.py module mechanism factorial_remaining \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# 5. Optional training and appendix sensitivities.
python run.py training_ablation appendix --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# A focused eval-only comparison of the 3% and 5% malware-FN risk levels.
# Reuse the same seed-42 checkpoint and refit both validation-based decision
# thresholds. The two runs differ only in the target residual malware-FN risk.
python run.py risk_05 risk_03 --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

Every test run writes `risk_coverage_curve.csv`, with one threshold-achievable
operating point per acceptance-score level and test scenario. It can be used
directly for risk-coverage plots. To compare two runs at matched acceptance
rates from their gate diagnostics, run:

```bash
python scripts/compare_risk_coverage.py \
  --input old=PATH_TO_OLD/gate_diagnostics.csv \
  --input new=PATH_TO_NEW/gate_diagnostics.csv \
  --split test_clean \
  --out-dir results/risk_coverage_comparison
```

Natural subsets must be rebuilt from the new seed-42 gate diagnostics before
their eval-only configs are run:

```bash
python scripts/build_natural_subset_csvs.py \
  --diagnostics results/tri_modal_robust/evidential_seed_42/42/gate_diagnostics.csv \
  --test-csv labels/test.csv \
  --out-dir labels/natural_subsets
python run.py natural_subsets --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

The builder writes `subset_manifest.json` with source and generated-CSV hashes
and refuses partial, duplicate, or subsequently modified artifacts. The three
common comparison subsets are low API effective integrity, low API-Graph
support, and high predictive conflict. Low acceptance is defined by the
proposed method and is evaluated only as an I3 diagnostic for Ours, not as a
neutral baseline-comparison subset.

The three newer Obfuscapk scenarios are an external robustness supplement and
should run only after their aligned PT directories and CSV files are present:

```bash
python run.py external_new baseline_obfuscapk_new \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```
