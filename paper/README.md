# Paper Baselines

This folder contains the baselines used for the thesis comparison section. The
implementations are intentionally split into two groups.

## A. Android Malware Detection Paradigms

These baselines are **adapted / inspired** versions because the exact original
raw feature templates are not fully present in the current tri-modal PT files.
They should be reported as style baselines, not strict reproductions.

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
  --extra-test-csv labels/natural_subsets/test_raw_high_conflict.csv \
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
  --extra-test-csv labels/natural_subsets/test_raw_high_conflict.csv \
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
  --extra-test-csv labels/natural_subsets/test_raw_high_conflict.csv \
  --extra-test-csv labels/natural_subsets/test_low_acceptance.csv \
  --out-dir paper/outputs/mamadroid_inspired
```

## B. Trusted Fusion / Multi-View Evidence Baselines

These baselines reuse the same tri-modal encoders and differ mainly in the
evidential combination rule. They are the closest comparisons for the paper's
trusted-fusion contribution.

- `tmc_dempster`: TMC-style Dempster-Shafer evidential fusion.
- `cumulative_subjective_logic`: cumulative subjective-logic fusion.
- `log_pool`: log-opinion-pool / product-of-experts fusion.
- `ecml_style`: adapted conflictive multi-view / ECML-style opinion aggregation.
- `ours`: the proposed method.

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
- "TMC-style Dempster" for the trusted multi-view baseline.
- "Cumulative subjective-logic" and "Log-pool evidential fusion" for fusion-rule baselines.
- `ecml_style` is an ECML-style baseline: it uses conflict-weighted view reliability,
  but it is not a strict reproduction of the original ECML objective.
