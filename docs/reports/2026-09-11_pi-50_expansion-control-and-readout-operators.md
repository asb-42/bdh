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

## 6. P-R1b relative reweighting — REFUTED, and the way it fails is the finding

Arms pre-registered in #175, all gain-preserving and relative so `self.ln` at bdh.py:279 cannot cancel them.
Identity gate re-run in this process: en 32.04 vs matrix 31.07, PASS.

| arm | en | de | cs | bg | el | reading |
|---|---:|---:|---:|---:|---:|---|
| id | 32.04 | 35.56 | 60.23 | 228.12 | 64.25 | control |
| massnorm | 31.26 | 39.77 | 91.77 | 215073 | 180510 | catastrophic: equalising mass gives new territories a loud voice |
| softmix tau=0.5 | **18.73** | **27.24** | 68.31 | 4486 | 4093 | first operator to move damaged languages the RIGHT way - and it destroys bg/el |
| softmix tau=1.0 | 46.02 | 48.56 | 95.52 | 965 | 1036 | sharper mixing goes the wrong way |
| calibgain (fit on en only) | see below | | | | | decisive arm |

### calibgain: what an English-only fitted compensation actually learns

One scalar per territory, fitted by Adam on English NLL over a 64 KB calibration slice disjoint from test,
60 steps, then every language evaluated unchanged. Pre-registered discriminator: if gains fitted on one
language rescue the others, a shared width compensation exists.

Loss fell 3.454 -> 0.915 (English exit corresponds to ln 2.31 = 0.84, so English is restored essentially
completely - expected by construction and therefore NOT the interesting part). The fitted vector is:

    base territories 0-3   : 2.41 2.16 2.31 2.53        mean 2.351
    appended territories 4-22: 0.02 ... 0.12 (all <=0.12) mean 0.036   (clamped floor 0.02)

Log-scale recovery relative to identity: en **+0.99**, de **+0.01**, cs **-0.37**, bg **-2.29**, el **-4.19**
(mean -1.17; negative = worse than doing nothing).

Conclusion, stated as the falsification of my own hypothesis: **there is no language-agnostic scalar
reweighting of the BDH readout that repairs growth damage.** Optimal single-language compensation
converges to "silence everything that grew", i.e. it rediscovers oracle masking specialised to the
language it was fitted on - which necessarily damages languages whose territory is among those silenced.
softmix tau=0.5 shows the same split with a different mechanism (helps the oldest languages, wrecks the
newest). Both failures point the same direction: the correct operation is **input-dependent selection of
which territory may speak**, not a fixed reweighting. That is precisely what A2's argmin-NLL prefix choice
does, and it is why selection has now succeeded (20/20, within +-8% of oracle) where culling, rescaling,
mass-equalising, temperature-mixing and gradient-fitted gains have all failed.

Instrument defect found and fixed during this arm: `ALPHA.view(1,1,1,NB)` broadcast against the
`(...,NB,1)` mass tensor produced `(...,NB,NB)` instead of scaling territories, killing the first run
(`RuntimeError: size of tensor a (2048) must match tensor b (23)`). Fixed to `(1,1,1,NB,1)` plus an explicit
shape assert. The three earlier arms were unaffected because their gain tensor was None, so the massnorm
and softmix numbers above stand.

## 7. P-R3 cheap addressing from byte geometry — 76% agreement, and my prediction about the failure mode was wrong

Labels generated by the model itself: 20 domains x 16 held-out test crops, cumulative-prefix NLL scored at
all 23 widths, argmin width = self-supervised route label. Every domain's label set was single-valued and
equal to its OWN territory (en->index 3 = 8192 ... sl->21), independently re-confirming A2's 20/20 selection
result on fresh crops.

Multinomial logistic regression on hashed byte 1-4-gram counts (2^18 buckets, L2-normalised), split BY CROP
(train 240 / test 80), no human task IDs:

* **top-1 agreement with the likelihood router: 0.762, Wilson 95% [0.659, 0.842]**, majority-class baseline 0.087.
* Bimodal by domain: **16 of 20 at 1.00**; the misses are exactly **es 0.00, pl 0.00, sk 0.00, et 0.67**.
* My pre-registered prediction - Latin easy, Cyrillic/Greek hard - is WRONG in direction: **bg 1.00, el 1.00**,
  and every systematic failure is a Latin-script neighbour of an early territory.

Limitation that stops me claiming more: 4 test crops per domain cannot separate "byte geometry genuinely
fails for es/pl/sk" from "sparse-sample confusion between adjacent classes". The honest statement is that a
counting model reproduces the router's decisions well enough (76% overall, perfect on 16/20 domains, 8.7x
chance) to make cheap input-side addressing plausible, while the residual failures are unresolved rather
than explained. Instrument gap: the script did not persist X/Y, so answering that question requires either
re-running label generation (~2.7 h GPU) or adding a dump flag - done next time before relying on it.

Cost accounting, the point of the exercise: the cheap predictor costs one counting pass over the input plus
a 262144x23 matvec, versus 23 masked forwards through the full model. On the RA2b checkpoint that is a
~23x reduction in forward cost for address determination, at the price of ~24% decision disagreement whose
structure is still unexplained.

## 8. exp4b finished while I was writing: selection accuracy is calibration-budget-INSENSITIVE, and binary search over widths FAILS

`scripts/pi50/exp4b_budget.py` reached 20/20 domains after ~17 h. Two questions, two clean answers
(log `~/bdh-review/reports/a2budget.log`, instrument in repo since `dd40e3a`):

| calibration bytes | linear argmin-NLL | forwards/input | binary search | forwards/input |
|---:|---:|---:|---:|---:|
| 4,000 | **20/20** | 23 | 6/20 | ~26 |
| 16,000 | **20/20** | 23 | 4/20 | ~24 |
| 64,000 | **20/20** | 23 | 4/20 | ~26 |
| 256,000 | **20/20** | 23 | 4/20 | ~26 |
| 1,000,000 | **20/20** | 23 | 4/20 | ~26 |

1. **The label-free selector needs almost no calibration data.** 4 KB of held-out bytes - about eight
   512-byte crops - selects the correct own-prefix territory for all 20 domains, identical to the 1 MB
   result that A2 originally reported. The accuracy bar from #148 (>=90% AND within 10% of oracle PPL) is
   met at every budget tested. Caveat kept explicit: the five budgets are nested measurements on the SAME
   20 domains, so they are not independent replicates; what the table establishes is the absence of
   degradation down to 4 KB, not a precise accuracy-vs-bytes slope.
2. **Binary search over prefix widths fails, 4-6 of 20.** It assumes NLL decreases then increases with
   width (unimodality). That assumption is false for BDH, and per the pre-registration the failure itself is
   the finding: the cumulative-prefix loss surface has multiple local minima, so you cannot find the right
   territory by bisection. Cost therefore stays O(K) masked forwards per candidate unless something cheaper
   supplies the address - which is precisely why P-R3/F-1/F-2 matter, and why the byte-geometry predictor
   at 76% agreement is worth pursuing even though it is not yet good enough to trust.

Practical consequence for any deployment claim: self-selection is robust and nearly data-free, but it is
NOT sublinear in the number of accumulated territories. That is the honest state of the cost story, and it
matches the operator's scaling question in #164 rather than contradicting it.
