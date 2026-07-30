# CARE-Droid v1 frozen specification

## Stage A: four clean prediction paths

Shared API, Graph, and Manifest encoders feed four independent lightweight
heads:

```text
AGM, AG, AM, GM
```

The clean training objective is:

\[
\mathcal L_A =
\frac{
\operatorname{CE}_{AGM}+
\operatorname{CE}_{AG}+
\operatorname{CE}_{AM}+
\operatorname{CE}_{GM}
}{4}.
\]

Stage A is fitted on `expert_train`. The checkpoint is the epoch with maximum
clean AGM Macro-F1 on the group-disjoint `expert_val`; exact ties keep the
earliest epoch.

## Path correctness estimation

For every path \(S\), define the binary log-odds

\[
g_S = \ell_{S,1}-\ell_{S,0}
\]

and the public hard prediction

\[
\hat y_S=\mathbb 1[g_S\ge 0].
\]

One shared 11-16-1 MLP estimates fixed-path correctness from exactly:

- four fold-normalized path log-odds;
- three hard modality-alive bits;
- a four-dimensional candidate-path one-hot.

The target is \(\mathbb 1[\hat y_S=y]\). Training averages BCE in the order
SID, deterministic view, and valid path.

`routing_cal` uses three group-disjoint OOF folds. Every fold fits
normalization only on valid paths in its training population, uses fixed
mini-batch training epochs without holdout feedback, and emits predictions
only for its holdout. A final head is refitted on all `routing_cal` rows with
the same fixed training budget.

## Deterministic controlled views

Each routing-calibration SID has seven fixed views:

- clean;
- API event dropout;
- graph sparsification;
- Manifest permission masking;
- API missing;
- Graph missing;
- Manifest missing.

The seed is \(H(SID, mechanism, protocol\_seed)\). Graded strengths are mapped
into \([0.1,0.9]\). View mechanism, strength, and seed are audit metadata only;
they never enter the risk head or router.

## AGM-anchored routing

- Three alive modalities: retain AGM when all hard predictions agree.
  Otherwise compare AGM only with disagreeing pair paths and choose maximum
  predicted correctness. Exact score ties use `AGM > AG > AM > GM`.
- Exactly two alive modalities: use the unique matching pair path.
- At most one alive modality: reject structurally.

No learned or hand-set routing threshold is used.

## Natural accepted-FN CRC

`decision_cal` contains one clean row per SID. Malware predictions are always
accepted. A benign prediction is accepted iff its selected-path correctness
score \(q\) satisfies \(q\ge\lambda\). Exact score ties are accepted or
rejected as one group.

The threshold maximizes the nested accepted set subject to:

\[
\frac{N_{\mathrm{accepted\ FN}}+1}
{N_{\mathrm{malware}}+1}
\le \alpha,\qquad \alpha=0.05.
\]

If `N_malware=0`, calibration fails without producing a threshold. If
\(\alpha < 1/(N_{\mathrm{malware}}+1)\), the status is
`infeasible_insufficient_malware`; reject-all must not be described as
certified.

This is an expected conformal-risk statement under exchangeability, not a
high-probability guarantee under arbitrary distribution shift.
