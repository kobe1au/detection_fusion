# CARE-Droid Experiments

This directory is the formal experiment catalog for the API/Graph/Manifest
malware detector. The only proposed method is `care_droid_v1`, configured by
`care_droid.yaml`. It runs through `python -m fusion.care_train`; ordinary
paper baselines run through `python -m fusion.baseline_train`.

Earlier proposed-method lifecycles and their dedicated ablations are not
registered or retained. They must not be mixed with CARE-Droid results.

## Frozen method contract

### Four fixed prediction paths

CARE-Droid exposes exactly four paths in a fixed order:

- `agm`: API + Graph + Manifest;
- `ag`: API + Graph;
- `am`: API + Manifest;
- `gm`: Graph + Manifest.

These are not a configurable expert search space. The full `agm` path is the
normal anchor, while the three two-modality paths provide bounded fallbacks
when one modality is unreliable or unavailable.

### Four disjoint data roles

The current train split is divided by package group into:

- `expert_train`: 90% used to fit the four clean path experts;
- `expert_val`: 10% used only to select the Stage-A expert checkpoint.

This assignment is deterministic with `expert_split_seed=4242`. The validation
CSV reuses the immutable SID assignment in
`labels/validation_roles_protocol_v3.json`:

- its 973 `model_selection` SIDs have CARE semantics `routing_cal`;
- its 487 `decision_calibration` SIDs have CARE semantics `decision_cal`.

The physical source-role names are recorded in the configuration so the
semantic rename cannot silently change identities. `routing_cal` never selects
Stage A, and `decision_cal` never fits experts, path risk, or routing.

### Stage A: clean path experts

All four paths are trained from clean `expert_train` samples. Stage A uses
`loss.objective=care_stage_a_clean`, the base 60-epoch budget, and patience 8
on clean `expert_val`. The selected expert artifact is:

```text
best_care_stage_a.pt
```

The primary method does not use paired degraded-view supervision in Stage A.

### Path-risk fitting and routing

After Stage A is frozen, CARE-Droid builds deterministic views of
`routing_cal`:

- one clean view;
- one SID-specific view for each of `api_event_dropout`, `graph_sparsify`, and
  `manifest_permission_mask`;
- `api_missing`, `graph_missing`, and `manifest_missing`.

For a graded view, strength is
`H(SID, mechanism, protocol_seed)` mapped uniformly into `[0.1, 0.9]`.
`protocol_seed=424242`. View identity and strength are audit metadata and are
never model inputs.

The shared path-risk head is fitted with three
`StratifiedGroupKFold` folds. It uses fixed 20-epoch fits, batch size 256,
hidden width 16, AdamW mini-batch updates, and no early stopping. Fold-local
log-odds normalization is fitted only on valid paths. OOF predictions provide
leakage-free path-risk and routing diagnostics; they do not select epochs,
features, architecture, or thresholds. The same fixed risk head is then
refitted on all `routing_cal` rows for deployment.

Training rows are averaged at three levels—SID, view, and valid path—so an SID
with more valid paths or views cannot dominate the BCE objective. Every
artifact records fold membership, SID/view provenance, data digests, and
fold-local normalization statistics.

Routing is deliberately bounded:

- with three live modalities, `agm` remains the anchor; only pair paths whose
  hard prediction disagrees with AGM are candidates, and a pair replaces AGM
  only when its predicted correctness is strictly larger;
- with exactly two live modalities, the unique matching pair path is used;
- with at most one live modality, the sample is rejected.

`route_on_all_samples` is false in the proposed method. No perturbation label,
strength, pre-damage count, or oracle path label is available at inference.

### Natural-only decision control

CARE-Droid does not fit a separate Macro-F1 classification threshold.
`classification_threshold.enabled=false`.

I3 fits only on clean, naturally observed `decision_cal` samples. It ranks
acceptance with `care_selected_path_correctness` and controls
`malware_conditional_accepted_fn` at `risk_level=0.05`. Artificial views do not
enter conformal calibration. The guarantee is expected conformal risk control
under exchangeability; it is not a high-probability guarantee under arbitrary
distribution shift.

The complete deployment artifact is:

```text
best_care_pipeline.pt
```

## Formal comparisons

The main table contains exactly 14 registered comparison baselines: seven
standard cells (API-only, Graph-only, Manifest-only, API+Graph concat,
tri-modal concat, alive-normalized equal-weight logit fusion, and dense
embedding gate) plus seven
trusted-fusion cells (Dempster, cumulative subjective logic, log pool,
conflict-weighted opinion, TMC-style adapted, QMF energy, and ECML-style
adapted).

Every baseline uses the same Stage-A data budget as CARE: only the deterministic
90% `expert_train` role fits parameters and the group-disjoint 10%
`expert_val` role selects the checkpoint by clean Macro-F1 at the fixed 0.5
operating point. The validation `model_selection` and `decision_calibration`
roles are reported only as audits; neither fits parameters, selects an epoch,
nor changes the classifier. CARE and all 14 baselines use the same unfitted
binary argmax/softmax-0.5 classification rule in the primary table.

The minimal CARE-Droid ablation set is:

- `no_learned_routing`: use `agm` whenever all three modalities are alive,
  while retaining the deterministic unique-pair fallback when exactly two
  modalities are alive;
- `route_on_all_samples`: remove the conservative routing restriction while
  keeping the same experts and risk head;
- `msp_acceptance`: keep CARE routing but replace the I3 acceptance score with
  maximum softmax probability.

These three cells reuse the frozen seed-42 pipeline because they change only
inference or decision rules.

No paired-path or anchor-family appendix is registered in v1. Either would
change the Stage-A objective and therefore requires a separately frozen
protocol rather than a half-compatible overlay.

## Commands

Append the AutoDL path overlay on the remote machine:

```bash
# Primary CARE-Droid run.
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Remaining seeds after the primary seed-42 run.
python run.py seed_2024 seed_3407 \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Ordinary paper baselines.
python run.py baselines trusted_baselines \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Three minimal CARE-Droid ablations.
python run.py paper_ablation \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Verify paths and runner dispatch without starting jobs.
python run.py paper_all --dry-run
python run.py --list
```

Dry-run output includes `[fusion.care_train]` or
`[fusion.baseline_train]`, making
lifecycle dispatch auditable before a long experiment starts.

Existing output directories are rejected. Use `--overwrite` only for an
intentional replacement.

## Manifest vocabulary preflight

After changing the dataset split, rebuild the Manifest vocabulary from current
train identities and migrate only Manifest-owned PT fields. API/Graph payloads
remain unchanged:

```bash
python scripts/migrate_manifest_vocab_pts.py \
  --train-csv labels/train.csv \
  --pt-dir /root/autodl-tmp/pts_all \
  --build-config config/build_pts.yaml \
  --vocab-out config/manifest_vocab.yaml \
  --manifest-jsonl-dir /root/autodl-tmp/pts_all/_manifest_jsonl \
  --audit-json manifest_migration_dry_run.json

python scripts/migrate_manifest_vocab_pts.py \
  --train-csv labels/train.csv \
  --pt-dir /root/autodl-tmp/pts_all \
  --build-config config/build_pts.yaml \
  --vocab-out config/manifest_vocab.yaml \
  --manifest-jsonl-dir /root/autodl-tmp/pts_all/_manifest_jsonl \
  --audit-json manifest_migration_apply.json \
  --apply
```

Do not migrate PT files while training reads the same pool.
