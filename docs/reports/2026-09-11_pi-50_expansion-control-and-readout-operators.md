# P-R1 / P-R2 — readout width-invariance and the expansion control

**Author:** pi-50 (gx10-50ef) · **Date:** 2026-09-11 · **Pre-registration:** bus #165 · **Interim report:** bus #167
**Instruments:** `scripts/pi50/r2_expansion_control.py`, `scripts/pi50/r1_readout_operators.py` (both `bdh@0f1763b`)
**Data:** RA2b checkpoints in `out/`, serving matrix `docs/reports/data/2026-09-10_ra2b_matrix.csv`

## 1. What was already established before this pass

Exp-3 (`exp3_delta_m.py`, `exp3_masked_vs_free.py`): old decoder blocks are bit-identical across all 19
ladder transitions, receive zero gradient-active optimizer state, and recover their acquisition-exit PPL
*exactly* when the readout is masked to their own prefix. Damage therefore lives in inference-time
selection, not synaptic erasure. A1 (`exp3_calib_null.py`): activation *energy* carries no selection
signal — every energy-based selector picks base block 0. Mechanism suspected: `k_sparse_ratio = 0.0`, so
`_k_sparse_relu` never runs and the readout sums ReLU candidates over all N units.

Open question posed by operator (#164): is the damage **arithmetic** (more competitors → different winner
set) or does it require **learned** representations from the new language competing for old inputs?

## 2. P-R2 random-expansion control — ANSWERED: mostly arithmetic

Base object: `out/bdh_europarl_ladRA2b-en_last.pt` (latent 8192 = 4 blocks × 2048, 100.9 M params).
Arms: **A** append whole blocks of iid Gaussian weights at the empirical scale of existing weights
(untrained, frozen); **C** append exact zeros (inert); **B** the real ladder checkpoints, read from the
matrix. Evaluation: free-width PPL on English test crops, BB=3–8, bf16 autocast, same crop pairing as the
A2 instrument. Instrument gate: un-grown free PPL must reproduce the matrix exit — measured **2.33 vs
matrix 2.31**, PASS.

| width | arm A (random) | arm C (inert) | arm A masked@8192 | arm B (real ladder) |
|------:|---------------:|--------------:|------------------:|--------------------:|
|  8192 |           2.33 |          2.33 |                 — |                2.31 |
| 10240 |           9.57 |          2.33 |              2.33 |                3.01 (es ckpt) |
| 47104 |          20.16 |          2.33 |              2.33 |               31.07 (lt ckpt) |

Three conclusions:

1. **One random, untrained, never-touched block costs English 2.33 → 9.57 (4.1×).** No new language, no
   gradient, no knowledge competition: only 2048 additional positive terms entering an existing sum.
2. **Pure arithmetic width growth reproduces 83 % of the log-scale damage.** Total damage
   ln(31.07/2.33) = 2.595; random-only damage ln(20.16/2.33) = 2.162; ratio 0.833. The trained Lithuanian
   territory adds the remaining ~17 %, so learned competition is real but second-order.
3. **Arm C is exactly 2.33 at every width** — bit-identical to baseline. This validates the head-major
   row handling (`decoder` rows are h·N+n; see `scripts/verify_masked_forward.py:110` for the same trap
   hit by pi-33) and shows the damage requires *nonzero* contributions, not merely larger tensors. Masking
   back to 8192 restores 2.33 under arm A as well, so the storage/execution split holds for synthetic
   growth too.

Consequence for the manuscript: BDH's "catastrophic forgetting" is predominantly an **unnormalised-sum
artefact of width growth**, not interference between languages. That reframes the finding from a
continual-learning failure into a readout-normalisation defect with a fixable operator.

## 3. P-R1 readout operators — NEGATIVE, plus a structural reason two of the arms were vacuous

Five inference-only operators applied to the shipped `ladRA2b-lt_last` (N=47104, 23 territories), no
training, no external mask: identity control, absolute-K top-k (K=2048, width-independent), F-V5
extreme-value shift `relu(v − σ_b·√(2 ln |b|))` per block, block-average readout (each block's decoder
contribution scaled by 1/n_blocks), log-N scaling. Identity control reproduces the matrix free row
(en 31.14 / de 35.21 / cs 60.86 / bg 226.99 / el 65.24 vs 31.07 / 34.71 / 62.10 / 230.22 / 63.66), so the
sweep measures the same phenomenon the matrix recorded.

| operator | en | de | cs | bg | el | verdict |
|---|---:|---:|---:|---:|---:|---|
| id | 31.14 | 35.21 | 60.86 | 226.99 | 65.24 | gate passed |
| absK (K=2048) | 39.37 | 48.09 | 78.73 | 1160.22 | 287.38 | **worse than doing nothing** |
| evshift (F-V5) | 36.66 | 49.47 | 117.32 | 4122.12 | 3472.78 | **worse, catastrophically on bg/el** |
| blkavg (÷23) | 31.14 | 35.22 | 60.85 | 227.07 | 65.29 | **provably vacuous** (see below) |
| lognorm | 31.14 | 35.22 | 60.86 | 226.92 | 65.27 | **provably vacuous** - confirms the LayerNorm explanation |

**No pre-registered pass condition was met.** The two candidate rescues make damaged languages worse,
in some cases by 5–18×, and the two rescaling arms change nothing measurable.

### Why blkavg/lognorm are vacuous by construction

`bdh.py:279` reads `yMLP = xy_sparse.reshape(...) @ self.decoder` followed immediately by `y = self.ln(yMLP)`.
A LayerNorm on the readout output annihilates any *global* gain change: dividing every block's decoder
rows by 23 (or by ln(N)) leaves the normalised output invariant up to the epsilon term, which is exactly
the ≤0.04 % difference observed. My pre-registration conflated "normalise the sum" with "rescale the
sum": only **relative** reweighting between territories can ever matter here. That is an instrument-design
error on my side, caught because the numbers were suspiciously identical rather than because they looked
interesting — the same class of defect as F-V7 (flags that do nothing), except this time the code told me
why in one line.

### What the negative results mean

1. **Culling candidates hurts more than growth helps.** With `k_sparse_ratio = 0.0` the readout is a
   learned many-term estimator over all N ReLU units; forcing it back to a fixed count (absK) or a fixed
   tail threshold (evshift) removes information the model was trained to expect. The damage from adding
   terms (§2) cannot be undone by removing terms.
2. **The extreme-value story is refuted as a *fix*, though not as a *cause*.** F-V5 predicted winner-set
   drift from growing maxima; P-R2 confirms growth alone suffices to cause damage, but the correction
   that follows from that prediction does not repair it. Cause and remedy are decoupled.
3. **This strengthens the operator's framing (#164).** If no width-invariant arithmetic on the existing
   readout rescues old languages at inference time, then BDH's central problem really is *addressing*
   — selecting which territory's candidates may participate — rather than readout algebra. The only
   mechanism that has worked so far is explicit selection: A2's label-free argmin-NLL prefix choice
   reached 20/20 within ±8 % of oracle PPL (#148/#150), i.e. masking works, normalising does not.

### Not yet tested (P-R1b)

Relative, gain-preserving reweighting that LayerNorm cannot cancel: per-block activation-mass
normalisation (`o_b · c/s_b` with s_b the block's ReLU mass at that position); softmax mixture across
territories with temperature; and calibration-fitted per-block scalars solved on a 64 KB slice (the
strongest remaining inference-side hypothesis, since it subsumes any monotone width compensation).

## 4. Relationship to prior work (do not duplicate)

`scripts/probe_selection_fix_operators.py` (pi-33, commit `6754e83`) already compared base / maskBefore /
absoluteK / maskAfter policies at **operator level in float64** using an activation-retention metric, and
retracted "mask before selection" as equivalent to zero-init and measured to break. P-R1 is not a repeat:
it measures end-to-end language PPL in the `k_sparse_ratio = 0.0` regime BDH actually ships, where none of
those policies apply unless the aggregation itself is changed. Literature anchors for the operators:
capacity-factor top-k (Switch/GShard), load-balancing auxiliary losses, SupSup thresholded supermasks,
maxout-style normalisation, and the extreme-value correction predicted as F-V5 on 2026-08-31.

## 5. Open items

* P-R3 cheap addressing from byte geometry (input-side predictor of the argmin-NLL route) — not started.
* P-R4 conformal threshold freezing for the two-axis reject rule (#163 X3) — not started.
* Arm A uses matched-scale Gaussian init, not the pipeline's exact fresh-BDH scheme; if an operator
  succeeds, re-run arm A with true init before generalising the 83 % figure.
* `exp4b_budget.py` still running (7/20 domains at ~1.6 h/domain, ETA ≈ +30 h); A2 already PASSED 20/20
  at 1 MB, so the budget curve is refinement, not prerequisite. Operator decision requested in #165.
