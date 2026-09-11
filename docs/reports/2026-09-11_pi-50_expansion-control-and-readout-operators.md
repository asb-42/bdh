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

> **SUPERSEDED IN PART — see section 9.** The 76 % figure and the "es/pl/sk/et fail" structure below are
> artifacts of an unbalanced, underfit multinomial, not properties of byte geometry. Arm B of
> `scripts/pi50/r3d_fit_ablation.py` reaches **1.000** on the same held-out crops. The pre-registered
> prediction being wrong in direction still stands as written (Cyrillic/Greek were indeed easy); the
> diagnosis of the four failures did not survive.

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

## 9. Follow-ups F-1/F-2/F-3: the addresser works, and the interesting result is where the two routers DISAGREE

Instruments: `r3b_byte_addressing_density.py` (label generation + feature persistence, 640 crops, 2.3 h GPU),
`r3c_ood_addendum.py` (corrected territory map + unseen-language inference), `r3d_fit_ablation.py`
(fit-design ablation, CPU-only off persisted features). Bus: #186 request, #187 acceptance, #188 iu
contamination, #196 withdrawals; this section closes them.

### 9.1 A defect found in my own newest numbers before they reached the paper

r3b reported es/pl/sk/et at 1.00 for every training density while the pooled headline fell 0.762 → 0.681
and misses migrated to the 16 control domains. Giving four domains 96 crops changed both their own data and
everyone else's share of a fixed-budget L2-penalised fit — two variables at once, so the discriminator
pre-registered in #187 was inapplicable. Consequence: the six-language F-2 table produced by that fit was
also unusable, because its predicted targets were exactly {es, pl, sk, et}, the over-represented classes.
Both withdrawn (#196) and regenerated from persisted features at zero GPU cost.

Also fixed: `lang_of_territory` used `ti-4`; the correct relation is territory = ladder position + 3
(English owns base indices 0–3, so ro→19 is *right* yet logged `own=False`). `r3c` asserts the mapping at import.

### 9.2 Fit ablation: the addresser reproduces the likelihood router perfectly

Same 160 held-out crops, same features, three fitting designs:

| arm | agreement [Wilson 95] | control-domain mean | miss attractor |
|-----|----------------------|---------------------|----------------|
| A reproduce r3b (unweighted softmax, 200 ep, L2 1e-4) | 0.681 [0.606, 0.748] | **0.125** | es ×33, sk ×9, et ×8 |
| B class-balanced softmax (inverse-freq weights, 600 ep, L2 1e-5) | **1.000 [0.977, 1.000]** | 1.000 | none |
| C one-vs-rest logistic per territory, margin abstention | 0.988 [0.956, 0.997] | 0.938 | sk ×2 |

Arm A's control-domain mean of 0.125 is the smoking gun: the starved classes were not merely degraded, they
were collapsed onto the majority classes. F-3 therefore **dissolves** — no disagreement confusion table is
left to build on trained domains (`f3_confusion_corrected.json` is `{}`).

Composition-controlled density curve (controls pinned at 12 train crops, class weights recomputed at every
point): es and et are 1.00 already at 8 crops, pl 1.00, sk 0.97 at 8 → 1.00 at 32, and controls stay at 1.00
at every point — which is what a properly balanced design must do. So the calibration requirement for the
cheap addresser is roughly **8–16 crops ≈ 4–8 KB per language**, the same order as A2's budget-insensitivity.

Method note: all fitting runs in a fixed 4096-dim Johnson–Lindenstrauss projection of the same hashed feature
matrix (full-dimensional epochs cost ~10 min each). Validity check built in: projected arm A reproduces
unprojected arm A's 0.681 exactly, and all arms share one projection, so between-arm comparison cannot be
affected by it.

### 9.3 F-2 redone under the balanced fit — this is where the science is

Cheap predictor (byte n-grams only, never sees a model forward pass) versus Quinn's likelihood router
(`out/logs/ra2b_routdiag_*.txt`, his counts reported in #163):

| input | cheap addresser route | likelihood router route | verdict |
|-------|----------------------|-------------------------|---------|
| zh | el ×36, sk ×2, bg ×1, cs ×1 → 37/40 in {bg,el} | 37/40 to bg+el | **exact count agreement** |
| ja | bg ×34, el ×6 → 40/40 in {bg,el} | 40/40 to bg+el | **agrees** |
| lv | lt ×40 | lt 40/40 (A3) | **agrees** |
| hi | fi ×40 | 40/40 to bg+el | **diverges** |
| iu (syllabic-only) | pl ×27, ro ×13 | TBD pending Quinn's cleaned re-run | open |
| iu (ASCII lines only) | **en(base) ×40** | 12/40 en on the raw file | agrees, and confirms #188 |
| ga | hu ×15, sv ×13, it ×3, others ×9 | ppl 50.54, counts unpublished | open |

Three readings, in decreasing order of confidence:

1. **The contamination confirmation is the cleanest new fact.** Given only the non-syllabic ASCII rows of
   `xscript_iu.txt`, the fitted predictor routes 40/40 to the English base territory — a label it was never
   trained on for those rows. Independent corroboration of #188 using a different instrument than my byte
   census, and of Quinn's reading that part of the en-share in X2 was real text-in-support rather than
   routing behaviour.
2. **Byte geometry genuinely carries the address for scripts adjacent to the training distribution**: zh, ja
   and lv reproduce the likelihood router's choices, including lv→lt Baltic kinship that nobody labelled by
   hand. A linear model on hashed 1–4-grams matching 37/40 against 37/40 is a strong result for cheap
   addressing — one counting pass instead of 23 masked forwards.
3. **hi is the measured boundary.** Devanagari lies outside the convex hull of everything the classifier saw;
   it snaps to an arbitrary Latin territory (fi ×40) instead of bg/el. The likelihood router, which actually
   evaluates the candidates, gets the reference answer by construction. An earlier draft of this paragraph
   called that prediction *confidently* wrong; `r3e_margins.py` (section 9.5) shows the adjective was too
   strong — hi's margin sits below the in-support median, and a threshold catches 45 % of its crops. The
   structural point survives: a cascade should prefilter in-support and escalate the O(K) likelihood scan when
   the input is far from every training territory.

Quinn expected iu-clean to still route bg/el; the cheap predictor says pl/ro. Either the byte model
extrapolates poorly to syllabics as it does to devanagari, or the two addressers really do partition
cross-script space differently. His re-run settles it — and if it lands bg/el while bytes say pl/ro, we have a
pattern (non-Latin scripts outside the hull) rather than two curiosities.

### 9.4 What changes in the manuscript

* The addresser claim upgrades from "76 %, partial" to "perfect reproduction on trained domains at ~4–8 KB
  calibration per language, with a measured out-of-hull failure mode on non-Latin scripts".
* State the fitting-design lesson explicitly: an unbalanced multinomial turns starvation into apparent
  structure, and the signature is misses concentrating on over-represented classes. Any router evaluation —
  including a future semantic/embedding layer — must declare class mix and total sample budget as design
  parameters, otherwise "the cheap router disagrees with the likelihood router" and "the cheap router was
  starved" produce identical-looking numbers.
* Keep the two-axis rejection rule of #163 (ratio AND absolute competence); §9.3 point 3 supplies the
  mechanism for why the ratio axis alone cannot catch hi.

### 9.5 Margin distribution: the stage-1 trigger for Sonde C, calibrated for free

Quinn's #197 design note asked for the balanced arms' margin distribution rather than top-1 only, because
Sonde C's cascade needs an uncertainty signal as its stage-1 trigger. Measured on the same fit
(`scripts/pi50/r3e_margins.py`, artifact `docs/reports/data/r3_followups/r3e_margins.json`):

**In-support reference.** On the 160 held-out trained-domain crops — all 160 correct — the top-1 minus top-2
softmax margin has percentiles p5 = 0.030, p25 = 0.051, p50 = 0.094, p75 = 0.185, p95 = 0.513, and a minimum
of **0.0139**. That minimum is the natural trigger floor: any threshold at or below it never fires on an
in-support crop.

**Out-of-support behaviour, against the likelihood-router reference:**

| input | cheap routes | median margin | fraction below the 0.0139 floor | agrees with likelihood? |
|-------|--------------|---------------|--------------------------------|-------------------------|
| lv | lt ×40 | 0.104 | **0.00** | yes (lt 40/40) |
| iu ASCII lines | en(base) ×40 | 0.090 | **0.00** | yes (English text) |
| hi | fi ×40 | 0.015 | 0.45 | no |
| zh | el ×36, sk ×2, bg ×1 | 0.008 | 0.82 | yes in aggregate |
| ja | bg ×34, el ×6 | 0.007 | 0.85 | yes in aggregate |
| ga | hu ×15, sv ×13, it ×3 | 0.004 | 0.85 | unknown |
| iu syllabic-only | pl ×27, ro ×13 | 0.001 | **1.00** | unknown / likely no |

The structure is better than expected and partly contradicts what I wrote in §9.3 before measuring it:

* The two cases where the byte addresser **agrees** with the likelihood router on unseen material (lv→lt,
  iu-ASCII→en) are exactly the two cases that sit comfortably inside the in-support margin band and would
  never trigger escalation. The addresser is confident precisely where it is right.
* Everything else fires the trigger between 82 % and 100 % of crops — including iu-syllabic, whose margin of
  0.001 is the most extreme value in the table and which is also the case where the two addressers most likely
  disagree.
* **hi is the hard case**, and not for the reason I claimed: it commits to one territory consistently, so its
  margin (0.015) lands just above the in-support minimum even though it is 6× below the in-support median. A
  floor at the minimum escalates 45 % of its crops; a floor at the median would escalate essentially all of
  them plus some legitimate in-support traffic.

Design consequence for Sonde C, stated as a measurement rather than a preference: a single margin threshold
gets you most of the way, and the residual is hi-shaped, so pair the margin with a byte-distance-to-support
axis and let the OR of the two drive escalation. Thresholds should be frozen from the numbers above (floor
0.0139 from n = 160 in-support crops; escalation rates quoted per input), which is the same
freeze-before-evaluate discipline P-R4 applies to the two-axis rejection rule. Caveat to keep visible: the
in-support floor comes from a fit that is 160/160 correct on these very crops, so it is optimistic — the real
trigger will need its margin estimated on crops the fit did not see at all, ideally from a re-fit under the
next growth phase, before anyone quotes a coverage guarantee from it.

### 9.6 Hull geometry test: Quinn's mechanism survives, his single scalar does not, and a second trigger signal falls out

Quinn's #200 synthesis proposed: "the byte addresser interpolates within the hull of trained byte statistics;
it generalizes to scripts that neighbor the hull's high-byte boundary (zh/ja) and snaps arbitrarily for
scripts that don't (hi/iu)." Falsifier stated before running (`scripts/pi50/r3f_hull_geometry.py`): if hi and
iu centroids sit *nearest* bg/el in n-gram space while the model still chooses elsewhere, the failure is not
hull distance. Validity gate first: leave-one-out nearest-centroid classification must reproduce the trained
routes — it does, 640/640 crops, every domain 1.00, so pure distance geometry recovers what the fitted linear
model recovers at 1.000 and OOD distances are interpretable.

| input | share bytes ≥ 0x80 | nearest territory centroids (cosine) | rank of bg / el | what the addresser chose | likelihood router |
|---|---|---|---|---|---|
| lv | 17.1 % | lt 0.841, sl 0.789, et 0.778 | 20 / 19 | lt ×40 | lt 40/40 |
| iu ASCII lines | 0.0 % | **en(base) 0.854**, fr 0.785 | 20 / 19 | en(base) ×40 | English text |
| ga | 11.8 % | it 0.792, sv 0.789, es 0.787 | 20 / 19 | hu/sv/it spread | ppl 50.54 |
| zh | 94.7 % | **el 0.083, bg 0.071** | 1 / 2 | el ×36/bg ×1 | bg+el 37/40 |
| ja | 96.4 % | **bg 0.073, el 0.073** | 1 / 2 | bg ×34/el ×6 | bg+el 40/40 |
| hi | 86.0 % | fi 0.144, sv 0.125, en(base) 0.097 | **19 / 20** | fi ×40 | bg+el 40/40 |
| iu syllabic | 91.3 % | en(base) 0.072, ro 0.071, cs 0.069 | **19 / 20** | pl ×27/ro ×13 | el ×28/bg ×12 |

Three conclusions, in order of strength:

1. **The mechanism is confirmed, but not as "the fit snapped arbitrarily."** For hi and iu the raw centroid
   geometry points at the *same* territories the fitted model chose (fi for hi; ro/en-base for iu) with bg/el
   ranked 19th and 20th of 22. The addresser is not malfunctioning; it is faithfully reporting surface
   proximity. What disagrees with it is the likelihood router, which measures realized fit instead. Two
   different definitions of "similar" produce two different answers, and both are internally consistent.
2. **High-byte fraction alone does not order the outcomes.** iu-syllabic is 91.3 % non-ASCII — *more* than
   hi's 86.0 % — yet diverges more severely (margin 0.001, bg/el rank 19/20, escalation 100 %). What actually
   separates zh/ja from hi/iu is proximity to bg/el specifically, i.e. which UTF-8 ranges the corpus covered,
   not how many bytes fell outside ASCII. Quinn's sentence should say "neighboring the trained multi-byte
   *regions*", not "high-byte boundary", and the scalar version should not appear in the manuscript at all.
3. **A second stage-1 trigger signal falls out for free: maximum centroid cosine.** Across these seven inputs
   it is bimodal with an enormous gap — 0.78–0.85 for material in the Latin training family (lv, ga, iu-ASCII)
   versus 0.07–0.14 for everything else (zh, ja, hi, iu-syllabic). Any threshold in the interval ≈ 0.15–0.78
   separates in-support from out-of-hull perfectly on this sample, requires no fitted parameters, and costs one
   dot product per territory against precomputed centroids. Its cost profile is also honest: it escalates zh
   and ja, where the addresser already agrees with the likelihood router, so it buys certainty about *divergence*
   (hi, iu) by spending escalations on *agreement* — 4 of 7 inputs escalate here, versus 2 of 7 if we only
   needed to catch the disagreements. Pairing it with the margin floor from §9.5 is what makes the OR-gate in
   Sonde C's Arm-3 spec concrete rather than decorative.

Contamination corroboration arrives a third way here: `iu_ascii` scores 0.854 against the English base
centroid, higher than any other input-to-territory similarity in the table, from a distance measure that
never saw a label or a fit. Combined with the fitted predictor's en(base) ×40 (#199) and my line census
(#188), the reading of those 333 lines as English prose is about as well-supported as anything in this set.
