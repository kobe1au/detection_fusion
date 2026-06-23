# Lean tri-modal robust experiment plan

The formal method is intentionally compact:

1. Observable static-parsing reliability evidence.
2. Reliability-aware probability-level discount fusion.
3. Selective rejection for low-reliability static samples.

Cross-attention and masked semantic reconstruction were removed from the formal
training path; current YAMLs expose only the compact reliability-discount-rejection
pipeline.

Main commands:

```bash
python run.py seed
python run.py baselines
python run.py module
python run.py training_ablation
python run.py mechanism
python run.py sensitivity
python run.py external
```
