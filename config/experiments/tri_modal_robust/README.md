# Tri-Modal Robust Experiment Plan

The layout follows current reproducibility practice: hyperparameters are chosen
on validation data, test data is reserved for frozen final models, mechanism
ablations are separated from numeric sensitivity analyses, and final results are
reported across multiple seeds.

## Files

- `base_tri_modal_robust.yaml`: fixed protocol and the default full method.
- `observable_reliability_discount_fusion.yaml`: the named full-method entry.
- `_autodl_paths.yaml`: optional path overlay for the AutoDL server.
- `tuning/`: validation-only hyperparameter selection.
- `ablations/`: frozen full method with one mechanism removed.
- `sensitivity/`: final/appendix hyperparameter sensitivity.
- `seeds/`: final multi-seed runs.

## Recommended Order

1. `python run.py tuning_i1`: choose reliability-module hyperparameters on
   validation only.
2. Freeze the selected I1 values in `base_tri_modal_robust.yaml`.
3. `python run.py tuning_i2`: choose semantic-interaction hyperparameters on
   validation only.
4. Freeze the selected I2 values.
5. `python run.py tuning_i3`: choose decision/rejection settings on validation
   only.
6. Freeze the final full method.
7. Run compact main-paper experiments:
   - `python run.py main`
   - `python run.py i1`
   - `python run.py i2`
   - `python run.py i3`
   - `python run.py training_ablation`
   - `python run.py full`
8. Run appendix experiments only when needed:
   - `python run.py ablation_appendix`
   - `python run.py sensitivity`

Use AutoDL paths with:

```bash
python run.py tuning_i1 --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

## Interpretation

- `tuning_*` experiments are not paper ablations. They justify selected values.
- `i1`, `i2`, and `i3` are compact mechanism-level ablations for the main paper.
- `i1_appendix`, `i2_appendix`, `i3_appendix`, and `ablation_appendix` keep
  fine-grained component splits for supplementary material.
- `sensitivity` reports whether conclusions are stable under nearby numeric
  hyperparameter changes and belongs in appendix/supplementary material.
- `paper` and `paper_main` exclude tuning and sensitivity runs. They are the
  recommended main-paper runnable set.
- `paper_appendix` contains fine-grained ablations and sensitivity experiments;
  it first runs `seed_42` so decision-only sensitivity configs have a checkpoint.
- `paper_all` contains both main-paper and appendix experiments.

## Main-Paper Ablation Scope

To match common recent ML paper style, the main text uses a compact set of
mechanism ablations:

- I1: reliability calibration, full observable evidence versus
  integrity/alive-only evidence, and alive/applicability masking.
- I2: semantic Cross-Attention, reliability-aware biases, relation masking, and
  residual tokens.
- I3: probability calibration, support/conflict discounting, confidence proxy,
  and selective rejection.
- Training: masked semantic reconstruction and synthetic degradation training.

Fine-grained splits such as support-only versus conflict-only and individual
attention-bias removals are preserved as appendix experiments.
