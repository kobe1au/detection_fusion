# Paper Baselines

This folder contains the baselines used for the thesis comparison section. The
implementations are intentionally split into two groups.

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
  --train-pt-dir /root/autodl-tmp/pts/train \
  --val-pt-dir /pts/val \
  --test-pt-dir /pts/test \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_api_low_effective_integrity.csv \
  --extra-test-csv labels/natural_subsets/test_api_graph_low_support.csv \
  --extra-test-csv labels/natural_subsets/test_predictive_high_conflict.csv \
  --extra-test-csv labels/natural_subsets/test_low_acceptance.csv \
  --out-dir paper/outputs/drebin_style
```

### MalDozer-inspired API sequence baseline

Uses only `api_ids` from PT files and trains a lightweight API sequence CNN. It
represents the API-sequence deep detection family without claiming exact
MalDozer reproduction.

Example:

```bash
python -m paper.baselines.maldozer_inspired_api_sequence \
  --train-pt-dir /root/autodl-tmp/pts/train \
  --val-pt-dir /pts/val \
  --test-pt-dir /pts/test \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_api_low_effective_integrity.csv \
  --extra-test-csv labels/natural_subsets/test_api_graph_low_support.csv \
  --extra-test-csv labels/natural_subsets/test_predictive_high_conflict.csv \
  --extra-test-csv labels/natural_subsets/test_low_acceptance.csv \
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
  --train-pt-dir /root/autodl-tmp/pts/train \
  --val-pt-dir /pts/val \
  --test-pt-dir /pts/test \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_api_low_effective_integrity.csv \
  --extra-test-csv labels/natural_subsets/test_api_graph_low_support.csv \
  --extra-test-csv labels/natural_subsets/test_predictive_high_conflict.csv \
  --extra-test-csv labels/natural_subsets/test_low_acceptance.csv \
  --out-dir paper/outputs/mamadroid_inspired
```

## B. Trusted Fusion / Multi-View Evidence Baselines

All comparisons reuse the same APK modalities, data split, encoder capacity,
and training budget. The method-level baselines and the controlled I2
mechanism substitutions are deliberately reported separately.

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

Controlled I2 fusion-mechanism comparisons:

- `dempster_rule_only`: Dempster combination under the proposed method's common
  I1 evidence protocol, not the TMC training objective.
- `cumulative_subjective_logic`: cumulative subjective-logic fusion.
- `log_pool`: log-opinion-pool / product-of-experts fusion.
- `conflict_weighted_opinion`: the repository's custom certainty/conflict-
  weighted opinion rule. It is an I2 mechanism ablation, not ECML.

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
- "Conflict-weighted opinion" for the custom I2 mechanism; do not label it as
  ECML.
