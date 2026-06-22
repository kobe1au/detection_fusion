# Tri-Modal Robust Experiment Plan

This directory supports the paper structure:

1. overall performance against internal baselines;
2. module-level ablation for I1/I2/I3;
3. static parsing degradation robustness;
4. mechanism analysis;
5. optional comparison with reproduced prior detectors.

## Main Paper

- Frozen method parameters: clean-validation checkpoint selection,
  `branch_aux_weight=0.20`, masked reconstruction `weight=0.02`, semantic
  Cross-Attention `dropout=0.0`, and `acceptance_aggregation=product`.
- Frozen semantics also account for the per-sample GNN node budget, use FP32 calibrated decision math, and split validation with package-group-aware label stratification. Pre-fix checkpoints are intentionally incompatible.
- `python run.py main`: full method plus every internal diagnostic baseline.
- `python run.py paper_baselines`: the compact main-table baselines: API only,
  Graph only, Manifest only, tri-modal Concat, and fixed Late Fusion.
- `python run.py module`: `w/o I1`, `w/o I2`, and `w/o I3` for Section 4.3.
- `python run.py full`: final three-seed full method.
- `python run.py paper_main`: compact main-paper set: five baselines, three
  module ablations, two training ablations, and the full method at three seeds.

`module`, `i1`, `i2`, `i3`, `component`, and `paper_appendix` do not rerun the
full method. Compare their rows against results from `main`, `full`, or
`paper_main`.

## Multi-Seed Protocol

`deterministic=true` with `strict_deterministic=false` is the speed-oriented,
best-effort seeded protocol. Formal conclusions use three seeds; set strict mode
only for a dedicated bitwise-reproducibility audit because it disables faster
CUDA attention kernels.

`paper_main` runs the full method at three seeds, but internal baselines and
ablations at seed 42. For publication-level mean and standard deviation, repeat
the paper baselines and key ablations with the two seed-only overlays:

```bash
python run.py paper_baselines module training_ablation --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml config/experiments/tri_modal_robust/_seed_2024_overlay.yaml
python run.py paper_baselines module training_ablation --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml config/experiments/tri_modal_robust/_seed_3407_overlay.yaml
```

## Robustness

- Controlled degradation is produced by `eval.perturb_tests` in the base config.
- Natural degradation subsets are produced after training:

```bash
python scripts/analyze_reliability_evidence.py \
  --results-root results/tri_modal_robust_new \
  --out-dir tables/natural_degradation \
  --figures-dir figures/natural_degradation \
  --experiment final_seed_42 \
  --seed 42 \
  --split test_clean \
  --diagnostic-file gate_diagnostics.csv \
  --bin-scope group \
  --natural-subset-quantile 0.333 \
  --natural-subset-min-count 30 \
  --fail-if-empty
```

- Obfuscapk eval-only configs live under `external/`. They reuse the trained
  full-method checkpoint and require `pts_obfuscapk/<scenario>/test` plus an
  exact `labels/obfuscapk_<scenario>_test.csv`; missing inputs fail loudly.

## Refreeze And Appendix

- Pre-fix tuning summaries remain provenance only because graph-budget evidence,
  calibrated decision precision, and the validation split changed.
- Run `python run.py refreeze` once before formal experiments. It compares the
  frozen candidate against nearby branch-loss, reconstruction, conflict, and
  Cross-Attention dropout alternatives under the corrected protocol.
- Select by clean validation Macro-F1; use `tuning_robust_composite_score` only
  as a predeclared tie-break. Then update the frozen base config if necessary.
- `python run.py tuning_objective`, `tuning_i1`, and `tuning_i2` are historical
  broad sweeps and do not need to be repeated unless the compact refreeze is
  inconclusive.
- Run `python run.py tuning_i3` after `refreeze`; it reuses the
  `tuning_full_candidate` checkpoint, is eval-only, and tests `min` against
  frozen `product`. On AutoDL, append `_autodl_tuning_i3_checkpoint.yaml` last.
- `python run.py component`: fine-grained component ablations.
- `python run.py sensitivity`: appendix sensitivity runs.
- `python run.py paper_appendix`: component ablations plus sensitivity, assuming
  the seed-42 full checkpoint already exists.
- `python run.py paper_appendix_with_seed`: standalone appendix run that first
  trains the required seed-42 full checkpoint.

## AutoDL

Use the path overlay as the last config:

```bash
python run.py paper_main --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

Run `final_seed_42` before `sensitivity` or `external`. The default AutoDL
overlay redirects those eval-only configs to
`results/tri_modal_robust_new/final_seed_42/42/best_tri_modal_robust.pt`.


## Prior Detector Reproductions

The repository does not currently contain audited Drebin, MaMaDroid, MalDozer,
MsDroid, or DeepCatra reproductions. Internal representation/fusion baselines must
not be described as reproduced prior methods. Obfuscapk PT directories require
scenario-specific CSV files containing exactly the successfully transformed
sample IDs.
