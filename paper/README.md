# Paper Baselines

This folder contains the baselines used for the thesis comparison section. The
implementations are intentionally split into two groups.

`review_draft14/` and `定稿13_统一图表/` are historical rendered exports from an
obsolete protocol. They are not inputs to the current experiment or reporting
workflow; the executable generator for that obsolete method has been removed.

## A. Android Malware Detection Paradigms

These baselines are **adapted / inspired** versions because the exact original
raw feature templates are not fully present in the current tri-modal PT files.
They should be reported as *-style adapted baselines, not strict reproductions.

### Drebin-style sparse static baseline

Uses sparse static indicators from the current PT files:

- Manifest permission / intent ids.
- Manifest category, component, and statistic vectors.
- API hash counts and API semantic category counts.
- Sensitive API and API-in-graph counts.

Example:

```bash
python -m paper.baselines.drebin_style_sparse \
  --train-pt-dir /root/autodl-tmp/pts_all \
  --val-pt-dir /root/autodl-tmp/pts_all \
  --test-pt-dir /root/autodl-tmp/pts_all \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_branch_disagreement.csv \
  --extra-test-csv labels/natural_subsets/test_api_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_graph_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_manifest_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_reliability_imbalance.csv \
  --extra-test-csv labels/natural_subsets/test_high_cross_modal_conflict.csv \
  --out-dir paper/outputs/drebin_style
```

### MalDozer-inspired API sequence baseline

Uses only `api_ids` from PT files and trains a lightweight API sequence CNN. It
represents the API-sequence deep detection family without claiming exact
MalDozer reproduction.

Example:

```bash
python -m paper.baselines.maldozer_inspired_api_sequence \
  --train-pt-dir /root/autodl-tmp/pts_all \
  --val-pt-dir /root/autodl-tmp/pts_all \
  --test-pt-dir /root/autodl-tmp/pts_all \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_branch_disagreement.csv \
  --extra-test-csv labels/natural_subsets/test_api_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_graph_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_manifest_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_reliability_imbalance.csv \
  --extra-test-csv labels/natural_subsets/test_high_cross_modal_conflict.csv \
  --out-dir paper/outputs/maldozer_inspired
```

### MaMaDroid-inspired semantic Markov baseline

Uses `api_type_ids` from PT files as abstract API states and builds a Markov
transition matrix over those states. The original MaMaDroid abstracts raw API
calls into package / family names; those raw names are not preserved in the
current PT schema, so this is an adapted baseline for the same behavioral
Markov modeling idea.

Example:

```bash
python -m paper.baselines.mamadroid_inspired_markov \
  --train-pt-dir /root/autodl-tmp/pts_all \
  --val-pt-dir /root/autodl-tmp/pts_all \
  --test-pt-dir /root/autodl-tmp/pts_all \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_branch_disagreement.csv \
  --extra-test-csv labels/natural_subsets/test_api_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_graph_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_manifest_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_reliability_imbalance.csv \
  --extra-test-csv labels/natural_subsets/test_high_cross_modal_conflict.csv \
  --out-dir paper/outputs/mamadroid_inspired
```

## B. Trusted Fusion / Multi-View Evidence Baselines

All comparisons reuse the same APK modalities, data split, encoder capacity,
and training budget. The method-level baselines and the complete fusion-rule
comparisons are deliberately reported separately.

Current proposed-method identity:

- Stage 1 trains three independent modality encoders and heads on clean
  training samples. It uses alive-masked uniform probability fusion for
  checkpoint selection; there is no concatenated or additional cross-modal
  branch.
- I1 estimates each alive branch's correctness probability from only its
  Dirichlet evidential certainty, top-1/top-2 margin, and an optional
  predicted-class intercept. Perturbation metadata, completeness/coverage
  proxies, and peer-modality signals are not model inputs.
- I2 routes with
  `beta * logit(reliability) - lambda * reliability-weighted conflict` and fits
  the route by conditional mixture NLL. Its separate risk head has exactly
  three inputs: routed reliability deficit, proximity to the deployed
  classification boundary, and global cross-modal conflict. The risk target is
  only the deployed-threshold malware false-negative event.
- I3 calibrates acceptance on a disjoint decision-calibration split using the
  finite-sample CRC correction. The claim is expected malware-FN risk control
  under exchangeability, not a high-probability guarantee.

Old checkpoints and summaries from earlier method protocols do not support
this method and must not be mixed into the current result tables.

Method-level comparisons:

- `tmc`: a TMC-style adapted baseline that retains the per-view and fused
  Dirichlet objectives and Dempster-Shafer opinion combination under the
  repository's common APK encoders and protocol.
- `ecml`: an ECML-style adapted baseline that retains fixed-order binary
  mean-evidence aggregation, per-view/fused evidential objectives, and the
  conflict-consistency regularizer under the common protocol.
- `qmf_energy`: QMF's detached energy-weighted late-fusion component with its
  fixed temperature. It does not include QMF's history-based confidence-ranking
  loss and must not be presented as a complete QMF reimplementation.
- `ours`: the proposed method.

Complete fusion-rule comparisons:

- `dempster_rule_only`: Dempster combination under the common three-branch
  evidential protocol, not the TMC training objective.
- `cumulative_subjective_logic`: cumulative subjective-logic fusion.
- `log_pool`: log-opinion-pool / product-of-experts fusion.
- `conflict_weighted_opinion`: the repository's custom certainty/conflict-
  weighted opinion rule. It is a complete fixed fusion-rule comparison, not
  an atomic I2 ablation and not ECML.

Run all formal trusted-fusion comparisons:

```bash
python paper/run_trusted_fusion_baselines.py --method all
```

Dry run:

```bash
python paper/run_trusted_fusion_baselines.py --method all --dry-run
```

## Reporting Wording

Use the following wording in the paper:

- "Drebin-style" and "MalDozer-inspired" for Android detection paradigms.
- "MaMaDroid-inspired" for API abstraction Markov behavior modeling.
- "TMC-style adapted" and "ECML-style adapted" for the two evidential
  baselines. They retain selected core objectives and fusion mechanisms, but
  are not strict reproductions of the original datasets, input networks,
  missing-view protocol, training procedure, or reported scores.
- "QMF-Energy component baseline" for `qmf_energy`; never shorten this result to
  a complete "QMF" comparison.
- "Dempster rule-only", "Cumulative subjective-logic", and "Log-pool
  evidential fusion" for the controlled fusion-rule baselines.
- "Conflict-weighted opinion" for the custom fixed fusion rule; do not label it
  as an atomic I2 ablation or as ECML.
