# Tri-Modal Competence-Anchored Fusion Experiments

This directory is the formal experiment catalog for the API/Graph/Manifest
malware detector. The paper asks one question: when the useful information in
the three modalities differs from sample to sample, can fusion remain strong
on clean inputs while adapting safely to an unreliable or missing modality?

The registered method is `tcp_joint_anchor_crc_v1`, configured by
`competence_anchored_fusion.yaml`.

## Method lifecycle

### Stage A: Joint anchor plus atomic experts

One training pass learns four experts:

- one Joint expert over the three encoded modalities;
- one API expert;
- one Graph expert;
- one Manifest expert.

The clean Stage-A objective is

```text
L_stage_A = CE(Joint, y)
          + lambda_atomic * mean_alive_m CE(atomic_m, y)
```

with `lambda_atomic=0.25`. Atomic losses are averaged over alive experts within
each sample before the batch mean. The Joint expert is the primary objective
and the clean anchor; the lower atomic weight makes the three atomic heads
deep supervision and missing-modality fallback experts rather than co-equal
primary tasks. It is not an average of atomic predictions. There is no EDL
loss and no uniform-fusion checkpoint proxy in the proposed method.

The checkpoint is selected by clean **Joint-anchor deployment Macro-F1** on
the fixed `model_selection` identities: the real Joint expert is used when all
three atomic inputs are alive, otherwise the explicit alive-uniform atomic
fallback is used. The pure-Joint Macro-F1 and Joint-eligible fraction are
reported separately, so the full-population checkpoint score is not
mislabelled as a pure-Joint result.

### I1: expert-local TCP competence

After Stage A, all encoders and expert heads are frozen. For each expert `m`,
I1 predicts

```text
q_m ~= p_m[y]
```

where `p_m[y]` is that expert's probability assigned to the true class
(true-class probability, TCP). The competence head receives only the current
expert embedding, its current probability vector, and its hard alive mask.

I1 is fitted on train identities under clean and one randomly sampled
single-modality degradation view. The degraded TCP loss is an auxiliary
regularizer with weight `0.25`. Every epoch is evaluated separately on clean
and the three fixed single-modality `model_selection` views. Selection first
keeps epochs whose clean TCP loss is within the pre-registered 1% relative
band of the best clean epoch, then minimizes degraded-source mean loss and
worst-source loss. Thus no arbitrary weighted validation mixture defines the
chosen head. Its loss also contains a small pairwise ranking term.
Perturbation name, perturbation strength,
pre-degradation counts, retained ratios, cross-modal conflict, and test
statistics are not model inputs.

This target is continuous: a confidently correct expert receives a target
near one, a confidently wrong expert a target near zero, and ambiguous
predictions remain intermediate. I1 is therefore a competence estimator, not
the retired branch-argmax correctness calibrator.

### I2: Joint-anchored adaptive late fusion

I2 constructs a competence-weighted late expert from the alive atomic
probabilities and combines it with the Joint anchor:

```text
score_m = b_m + beta_atomic * log(q_m)
w_m = softmax_alive(score_m)
p_late = sum_m w_m * p_m
g = sigmoid(b_gate + beta_gate * (log(q_late) - log(q_joint)))
p_final = (1 - g) * p_joint + g * p_late
```

Both scale parameters are constrained positive, so increasing an expert's
competence cannot reduce its atomic score, and increasing late competence
relative to Joint cannot reduce `g`. The small router is learned after Stage
A. Its training contains:

- clean classification loss;
- single-modality degradation loss with a registered candidate-weight grid;
- a clean KL anchor to the frozen Joint prediction.

Each candidate and the Joint anchor independently fit an unconstrained
Macro-F1 threshold on the same clean `model_selection` rows; those thresholds
are then frozen across clean and all three degraded sources. Deployment is
fail-closed: the adaptive router is enabled only if clean Macro-F1 is not below
Joint, none of the three degraded sources is below Joint, and the mean
degraded-source gain is strictly positive. Candidate ranking is minimum
source delta, mean delta, clean delta, then negative NLL. Otherwise the saved
pipeline uses the clean-thresholded Joint anchor when all
modalities are alive, an alive-uniform atomic fallback when Joint is
ineligible, and a uniform class distribution only when every modality is
dead.

I1 and I2 are coupled parts of one trusted-fusion mechanism. The old I1×I2
factorial, log-odds reliability prior, conflict risk head, source oracle,
scenario-prior grid, and nested five-fold post-hoc stack are not part of this
method.

The causal module ablation disables the complete Stage B and deploys the
unchanged Joint anchor from the same Stage-A checkpoint. It does not pretend
that I2 can operate without the I1 competence values that define its atomic
late expert.

### I3: malware false-negative risk control

The ordinary classification threshold is selected for Macro-F1 on
`model_selection` and then frozen. A disjoint `decision_calibration` role is
used only by I3 to choose an acceptance threshold subject to

```text
(accepted malware false negatives + 1) / (number of malware + 1) <= alpha
```

with `alpha=0.05`. The score
`malware_fn_probability_anchor` ranks samples by the deployed classifier's
probability of making the malware-as-benign error. The code reports an
expected conformal risk-control guarantee under exchangeability; it does not
claim a high-probability guarantee or a guarantee under arbitrary shift.

## Data lifecycle

`labels/validation_roles_protocol_v2.json` freezes a package-group-disjoint
75/25 partition:

- 75% `model_selection`: Stage-A checkpoint selection, Stage-B early stopping
  and candidate selection, and the ordinary classification threshold;
- 25% `decision_calibration`: I3 only.

Stage-B parameters are fitted on train identities. Validation selects fitted
states and hyperparameters; it is not described as unseen by I1/I2. Test data
is never used for fitting, candidate selection, or thresholds.

Protocol warning for the current study: earlier method revisions were informed
by summaries from the existing `labels/test.csv`, so that split must now be
treated as a development test rather than an untouched confirmatory set. The
final paper claim requires a newly locked, previously unseen test set or an
external confirmation dataset. Code-level split isolation cannot undo
researcher-side reuse of already inspected test results.

The controlled degradation registry contains one evidence-availability stress
test per modality:

- API: `api_event_dropout`;
- Graph: `graph_sparsify`;
- Manifest: `manifest_permission_mask`.

Training samples one mechanism and a continuous strength in `[0.1, 0.9]`.
Stage-B selection evaluates the three fixed mechanisms at strength `0.5`.
Formal evaluation reports five strengths `[0.1, 0.3, 0.5, 0.7, 0.9]`, plus
clean and the three whole-modality-missing endpoints. These are controlled
availability tests, not claims about unseen attacks or temporal
generalization.

## Comparisons and ablations

Representation/fusion baselines:

- API only, Graph only, Manifest only;
- API+Graph concat and tri-modal concat;
- fixed logit fusion;
- dense embedding gate adapted.

Trusted-fusion comparisons:

- Dempster, cumulative, log-pool, conflict-weighted opinion;
- TMC-style adapted;
- ECML-style adapted;
- QMF energy component.

`TMC-style adapted` and `ECML-style adapted` are mandatory paper names. These
experiments share the APK inputs and encoders but are not strict
reproductions.

All main fusion baselines receive the same fixed `model_selection` identities
and the same unconstrained Macro-F1 threshold budget as the proposed method.
That fitted-threshold result is the primary classification table. A separate
fixed-0.5 table is retained as a no-threshold-tuning sensitivity analysis; I3
remains disabled for fusion baselines and is compared in its dedicated table.

The compact method ablation set is:

- `no_i1_i2`: disable the complete competence/router Stage B and deploy the
  matched Joint anchor;
- `no_degraded_competence`: fit I1 on clean train outputs only;
- `no_tcp_ranking`: remove only I1's ranking regularizer;
- `router_clean_only`: remove degraded-router training;
- `no_clean_anchor_kl`: remove I2's clean Joint-anchor regularizer;
- `no_atomic_auxiliary`: remove Stage-A atomic supervision;
- `no_i3`: disable only selective risk control;
- I3 score/rule comparisons: MSP, deployed-class probability, fixed coverage,
  marginal conformal, and class-conditional conformal.

There is no I1×I2 2×2 grid because I2 is explicitly defined in terms of I1
competence. “I2 on, I1 off” would be a different model rather than an atomic
ablation.

The I1/I2 ablations, including complete `no_i1_i2`, reuse the hash-checked
seed-42 `best_encoder_selected.pt`; they do not retrain Stage A. The
`no_atomic_auxiliary` cell changes the Stage-A objective and therefore trains
its own artifact. Run `final` before `paper_ablation`; the ablation group does
not rerun or overwrite the primary result.

## Commands

On AutoDL, always append the machine-path overlay:

```bash
# Primary seed
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Repeated seeds
python run.py seed \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Main comparison methods
python run.py baselines trusted_baselines \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Proposed-method ablations and I3 comparisons
python run.py paper_ablation \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml

# Inspect without executing
python run.py paper_all --dry-run
python run.py --list
```

Cross-method mean/std tables require repeating comparison methods with the
registered seed overlays:

```bash
python run.py baselines trusted_baselines \
  --extra-config \
    config/experiments/tri_modal_robust/_autodl_paths.yaml \
    config/experiments/tri_modal_robust/_seed_2024_overlay.yaml

python run.py baselines trusted_baselines \
  --extra-config \
    config/experiments/tri_modal_robust/_autodl_paths.yaml \
    config/experiments/tri_modal_robust/_seed_3407_overlay.yaml
```

An existing output directory is rejected. Use `--overwrite` only when
replacement is intentional. Stage-A reuse is allowed only when the strict
architecture, objective, data, PT provenance, RNG/loader, and validation-role
identity checks pass:

```bash
python run.py final \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml \
  --encoder-checkpoint /absolute/path/to/best_encoder_selected.pt \
  --overwrite
```

Collect schema-v14 results and source-separated I1 diagnostics with:

```bash
python scripts/collect_experiment_results.py
python scripts/analyze_reliability_evidence.py
```

The I1 analysis reports competence-versus-TCP MSE/MAE, Pearson/Spearman
alignment, error AUROC, cross-expert ordering accuracy, and TCP calibration
diagrams. Old EDL/reliability signal tables are intentionally unsupported.

## Natural difficulty subsets

After freezing the final seed-42 method, rebuild the schema-v9 subsets from
its validation diagnostics:

```bash
python scripts/build_natural_subset_csvs.py \
  --diagnostics \
    results/tri_modal_robust/competence_anchored_seed_42/42/gate_diagnostics.csv \
  --test-csv labels/test.csv \
  --out-dir labels/natural_subsets

python run.py natural_subsets \
  --extra-config config/experiments/tri_modal_robust/_autodl_paths.yaml
```

The registered label-free subsets include branch disagreement,
competence imbalance, and high cross-modal conflict. “Exactly one branch
wrong” subsets use test labels and are diagnostic only. Validation chooses
subset cutoffs; target test scores never choose them.

## Manifest vocabulary preflight

After a split change, rebuild the vocabulary from current train identities and
migrate only Manifest-owned PT fields. This preserves the expensive API/Graph
payload:

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

Do not run PT migration and training concurrently. Old checkpoints and
schema-v12 summaries are intentionally incompatible with this protocol.
