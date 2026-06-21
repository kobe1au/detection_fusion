# Tri-Modal Robust Experiment Plan

This directory supports the paper structure:

1. overall performance against internal baselines;
2. module-level ablation for I1/I2/I3;
3. static parsing degradation robustness;
4. mechanism analysis;
5. optional comparison with reproduced prior detectors.

## Main Paper

- `python run.py main`: full method plus internal baselines for Section 4.2.
- `python run.py module`: `w/o I1`, `w/o I2`, and `w/o I3` for Section 4.3.
- `python run.py full`: final three-seed full method.
- `python run.py paper_main`: compact main-paper runnable set.

`module`, `i1`, `i2`, `i3`, `component`, and `paper_appendix` do not rerun the
full method. Compare their rows against results from `main`, `full`, or
`paper_main`.

## Multi-Seed Protocol

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
  --results-root results/tri_modal_robust \
  --out-dir tables/natural_degradation \
  --figures-dir figures/natural_degradation \
  --split test_clean \
  --diagnostic-file gate_diagnostics.csv \
  --bin-scope group \
  --natural-subset-quantile 0.333 \
  --natural-subset-min-count 30
```

- Obfuscapk eval-only configs live under `external/`. They assume obfuscated PTs
  were built separately and reuse the trained full-method checkpoint.

## Appendix

- `python run.py component`: fine-grained component ablations.
- `python run.py tuning_base`: validation-only default candidate.
- `python run.py tuning_i1`, `python run.py tuning_i2`, `python run.py tuning_i3`:
  validation-only variants for each module.
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


## Prior Detector Reproductions

The repository does not currently contain audited Drebin, MaMaDroid, MalDozer,
MsDroid, or DeepCatra reproductions. Internal representation/fusion baselines must
not be described as reproduced prior methods. Obfuscapk PT directories require a
CSV containing exactly the successfully transformed sample IDs; replace Windows
template paths before AutoDL evaluation.
