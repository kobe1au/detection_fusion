# Reliability-Aware Tri-Modal Experiment Plan

This experiment plan targets the current model with reliability-aware semantic
Cross-Attention. It uses the existing schema-4 PT files and does not require PT
regeneration.

## Fixed Protocol

- Data membership is defined by `labels/{train,val,test}.csv`.
- All splits use strict schema-4 / observable-v1 PTs.
- Sample ID and package name must not overlap across Train, Val, and Test.
- At most 2048 API events are used per APK.
- Val is deterministically split by package/sample group:
  - `val_selection` selects the best checkpoint;
  - `val_calibration` fits reliability, branch temperatures, and rejection threshold.
- Checkpoint selection uses robust-composite Macro-F1 on `val_selection`.
- Test includes clean data, five degradation strengths, and modality-missing cases.
- Synthetic `pert_*` metadata is diagnostic-only and is never model evidence.

## Current Full Method

`observable_reliability_discount_fusion.yaml` contains:

1. observable evidence and monotonic branch-reliability calibration;
2. observable-prior-constrained security-semantic Cross-Attention;
3. calibrated probability discount fusion and selective rejection;
4. masked semantic reconstruction using target-excluded enhanced sources;
5. synthetic degradation training and reliability-weighted branch supervision.

The Cross-Attention module:

- preserves independent API, Graph, and Manifest branch logits;
- enhances only the Joint branch;
- uses 12 category-indexed learned security anchor tokens and 4 learned
  residual tokens per modality;
- injects observable reliability, support, conflict, and relation applicability;
- blocks unavailable modalities as attention sources;
- constructs API-Manifest and Graph-Manifest applicability separately;
- excludes the reconstruction target modality from enhanced reconstruction sources.

## Run Groups

```bash
python run.py final
python run.py baselines
python run.py i1
python run.py i2
python run.py i3
python run.py training_ablation
python run.py seed
python run.py sensitivity
python run.py paper --dry-run
```

`paper` contains 42 unique runs. The final method is represented by the three
seed configs and is not duplicated by the root final config.

## Baselines: 9 Runs

Representation baselines disable Cross-Attention, semantic reconstruction,
branch auxiliary supervision, calibration, and rejection:

- API only;
- Graph only;
- Manifest only;
- API + Graph concatenation;
- tri-modal concatenation.

Fusion baselines retain the full representation-training path, including
Cross-Attention, but replace probability discount fusion:

- fixed equal-weight logit fusion;
- confidence-weighted logit fusion;
- heuristic observable-reliability logit fusion;
- learned observable-evidence logit fusion.

## Innovation I1: Observable Reliability Priors

Question: do observable relation evidence, branch calibration, and semantic
presence priors improve reliability estimation and downstream robustness?

- `no_reliability_calibration`: use raw observable integrity as base fusion reliability;
- `integrity_alive_only`: remove relation support/conflict evidence from both
  the evidence path and explicit final discounts;
- `no_semantic_presence_prior`: use modality-level priors for all security
  tokens instead of presence-modulated token priors.

Report branch reliability Brier/ECE, correctness separation, fusion weights, and
robust classification metrics.

The token-level value is an observable presence-modulated prior, not a
post-hoc calibrated per-category correctness probability.

## Innovation I2: Reliability-Aware Semantic Cross-Attention

Question: does observable-prior-constrained semantic interaction outperform the
old independent-joint path and ordinary Cross-Attention?

- `no_semantic_cross_attention`: restore the previous concatenation Joint branch;
- `plain_semantic_cross_attention`: retain relation masking but remove
  reliability/support/conflict attention biases;
- `no_cross_attention_reliability_bias`;
- `no_cross_attention_support_bias`;
- `no_cross_attention_conflict_bias`;
- `no_cross_attention_relation_mask`: unavailable sources remain blocked;
- `no_cross_attention_residual_tokens`;
- `joint_only_cross_attention`: do not attach target-excluded enhanced sources
  to semantic reconstruction.

Report Joint-branch Macro-F1, final Macro-F1, missing-modality robustness,
attention-to-modality means, attention entropy, and residual-gate value.

## Innovation I3: Calibrated Discount Fusion And Rejection

Question: which final decision components improve calibration and selective risk?

- `no_probability_calibration`;
- `no_support_discount`;
- `no_conflict_discount`;
- `no_confidence_proxy_discount`;
- `no_hard_alive_mask`;
- `no_selective_rejection`;
- `raw_discount_no_posthoc_calibration`;
- comparison with `learned_evidence_logit_fusion`.

`no_support_discount` and `no_conflict_discount` remove explicit final-discount
multipliers only. Cross-Attention and the branch-reliability calibrator may still
consume the corresponding observable evidence.

## Training Ablations: 4 Runs

- remove masked semantic reconstruction;
- remove synthetic degradation training;
- remove branch auxiliary supervision;
- replace reliability-weighted branch supervision with ordinary branch supervision.

## Full Method: 3 Seeds

- seed 42;
- seed 2024;
- seed 3407.

Report mean and standard deviation. The validation holdout remains fixed with
`calibration.split_seed=42`.

## Sensitivity: 8 Runs

Train new models for:

- 2 and 8 residual tokens, compared with the default 4;
- 2 and 8 attention heads, compared with the default 4;
- unavailable relation treated as full support in the post-hoc calibrator.

Reuse the calibrated seed-42 checkpoint and refit only the rejection threshold for:

- product acceptance aggregation;
- target coverage 0.80;
- target coverage 0.95.

Run `python run.py seed` before decision-only sensitivity experiments. The
`paper` group already enforces the required order.

## Reporting

Classification and robustness:

- Macro-F1, accuracy, ROC-AUC, AP;
- clean and every synthetic degradation/missing scenario;
- per-branch and final predictions.

Calibration and selective prediction:

- Brier, ECE-10, confidence-accuracy gap;
- coverage, selective risk, selective Macro-F1, AURC.

Cross-Attention analysis:

- mean semantic reliability prior per modality;
- mean attention received by each source modality over alive target queries;
- mean cross-modal attention and attention entropy;
- behavior under missing modalities and high Manifest-Code conflict.

## Required External Validation

The runnable YAMLs still do not prove real-world robustness. Publication
experiments should additionally include:

- paired original/Obfuscapk APK evaluation;
- naturally packed or partially unparseable APK evaluation;
- external published malware-detection baselines.
