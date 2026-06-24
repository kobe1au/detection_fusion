# Evidential trusted-fusion experiment plan

Single robustness axis: **trustworthy tri-modal detection under modality
degradation / obfuscation**. Three innovations:

- **I1 — dual-source reliability**: observable parse quality + EDL evidential
  certainty `1-u` (Dirichlet evidence head).
- **I2 — conflict-aware evidential fusion**: subjective-logic trust discounting
  + Yager combination (conflict mass → uncertainty, not normalised away).
- **I3 — class-conditional conformal selective rejection**: per-class (Mondrian)
  split conformal; abstains on low-evidence / high-conflict samples.

Main method: `evidential_trusted_fusion.yaml`.
Prior-method baseline (linear discount, no EDL/conformal):
`observable_reliability_discount_fusion.yaml`.

## Run

```bash
python run.py final            # the evidential main method
python run.py paper_evidential # main + baselines + module/mechanism ablations + seeds
python run.py paper_evidential_all   # everything (adds training ablations, sensitivity, external)
python run.py --list           # all groups / aliases / configs
```

## Groups

| group | contents |
|-------|----------|
| `main` | evidential method + baselines |
| `baselines` | single-modal, concat, fixed gate, prior linear method |
| `module` | I1-off, I3-off |
| `mechanism` (`i1`/`i2`/`i3`) | EDL-source, combination rule, conformal variant ablations |
| `training_ablation` | no augmentation / no branch-aux / no EDL class-weight |
| `sensitivity` | I1 (EDL weight, anneal, hidden dim), I2 (reliability exponent), I3 (coverage) |
| `external` (`obfuscapk`) | obfuscapk rename/code/encryption/combined (eval-only) |
| `seed` | seeds 42 / 2024 / 3407 |

## Notes
- I3 ablations/sensitivity and external evals are **eval-only**: they reuse the
  trained `evidential_seed_42` checkpoint and only re-fit the decision threshold
  (rejection is post-hoc, no retraining needed).
- Conformal metric naming is deliberate: `conformal_*_acceptance_rate` is the
  singleton-acceptance fraction; `conformal_empirical_coverage_*` is the actual
  guaranteed coverage P(true label ∈ set | class) ≥ 1−α.
