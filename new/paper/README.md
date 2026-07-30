# CARE-Droid paper experiment notes

The only proposed-method protocol in this project is `care_droid_v1`.
Its executable specification is:

- `fusion/care_fusion.py`;
- `fusion/care_training.py`;
- `fusion/care_train.py`;
- `config/experiments/tri_modal_robust/care_droid.yaml`.

Historical documents and exported figures below this directory predate
CARE-Droid. They are retained only as source material and must not be cited as
the current algorithm, loss, data lifecycle, or experimental result.

## Frozen method

The paper method is:

```text
clean AGM/AG/AM/GM path experts
  -> three-fold group-OOF fixed-path correctness estimation
  -> disagreement-aware AGM-anchored routing
  -> natural-distribution accepted-FN CRC
```

It does not contain EDL, prototypes, reference tokens, competence tokens,
conflict losses, advantage heads, a fitted classification threshold, or
manual routing thresholds.

The public binary prediction is malware iff
`malware_logit - benign_logit >= 0`.

## Formal commands

```bash
# Main method, seed 42.
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Three method seeds.
python run.py seed \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Standard and trusted-fusion baselines.
python run.py baselines trusted_baselines \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Frozen inference/decision ablations.
python run.py paper_ablation \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

All source-paper adaptations must be named `*-style adapted`,
`*-inspired`, or `component baseline` according to their actual implementation;
they must not be described as strict reproductions.

## Result collection

```bash
python scripts/collect_experiment_results.py \
  --results-root results/tri_modal_robust \
  --out-dir tables
```

The collector accepts the frozen CARE summary schema and the current baseline
summary schema. It rejects retired proposed-method sections.

## Interpretation boundary

Artificial degradation cells are controlled stress tests. CRC is calibrated
only on the natural `decision_cal` role, and its expected-risk statement
applies only to exchangeable natural test data. Degradation results are
reported empirically without a CRC guarantee.
