# RA2b serving matrix, and why BDH "forgets": selection, not erasure

**Author:** pi-50 (GX10 seat, `asb@gx10-50ef`) · **Date:** 2026-09-10 · **Status:** measurement complete, one arm in flight
**Bus thread:** `bdh-cl` #127 → #143 · **Code:** `scripts/pi50/` · **Data:** `docs/reports/data/`

## 1. Why this exists

The operator asked (#127) whether we hold pre-phase PPL for all 20 languages at every
checkpoint. We did not: only 50 of 400 cells existed (milestones at phases 5/10/15/20,
seen-languages-only). This work fills the matrix, then follows the single anomaly it exposed
into a mechanism result that changes what the manuscript may claim.

## 2. The 20×20 matrix (`data/2026-09-10_ra2b_matrix.csv`)

* Checkpoints: `out/bdh_europarl_ladRA2b-{lang}_last.pt`, all 20, produced on GX10.
* Eval: `scripts/lang_eval.py`, `--europarl-lang-mb 30`, **`EVAL_BATCH=1` pinned identically for
  all 400 cells**, one invocation per checkpoint covering all 20 domains, crop seed fixed inside
  `lang_eval.py`. Runner: `scripts/pi50/matrix_eval.sh`.
* Hardware: RTX 4090 (`.200`), checkpoints mirrored read-only into a private home dir; nothing
  written under `out/`. Peak VRAM 9.3 GB cold / ~3 GB steady; sweep 33 min.
* Consistency check: the diagonal reproduces the ladder's own exit numbers exactly
  (cs 3.48, da 2.85, de 2.73), so matrix and Appendix A speak the same protocol.

Headline (row = final `lt` checkpoint, free evaluation): every language except the last collapses
— en 2.31→31.07, pl 3.01→62.11, hu 2.88→73.89, bg 6.09→230.22, el 6.36→63.66. Single-phase
examples: French costs Polish 3.01→40.52; Danish costs Czech 3.48→26.49.

Two protocol warnings that must not be repeated: retention computed as `1 − forgetting/initial`
is **position-biased by recency** (quote per-language ratios, never a mean), and apparent
"forward transfer" cells are often **recovery after damage** rather than genuine benefit.

## 3. Experiment 3 — where does the damage live? Nowhere in the weights.

| measurement | result | instrument |
|---|---|---|
| decoder churn ‖ΔW‖/‖W‖, every old block, all 19 transitions | **0.0000** | `exp3_delta_m.py` (per-head row-aligned, rows every 16th) |
| old blocks receiving gradient | **0**, all phases | full nonzero count of `exp_avg_sq` |
| encoder / encoder_v old columns | **0.0000** churn, 0 active elements | `exp3_encoder_churn.py` |
| embed / lm_head | bit-identical (frozen from step 1, `pipeline/train.py:148-149`) | prior identity probe |
| routing (19 `ladRA2b_routdiag_p*.txt`) | each trained domain keeps **100 %** of its own prefix | confusion-grid parse |

Yet serving PPL falls by 10–40×. The resolution is decisive and cheap to test: evaluate the
final checkpoint twice per domain, free vs `neuron_mask` restricted to that domain's own prefix
(`exp3_masked_vs_free.py`).

| domain | free | masked to own prefix | its acquisition exit |
|---|---|---|---|
| en | 31.07 | **2.31** | 2.31 |
| pl | 62.11 | **3.01** | 3.01 |
| hu | 73.89 | **2.88** | 2.88 |
| bg | 230.22 | **6.09** | 6.09 |
| el | 63.66 | **6.36** | 6.36 |
| lt | 3.83 | 3.83 (negative control: own prefix = full width) | 3.83 |

**All 19 damaged domains return to their exit value exactly.** Nothing was destroyed; access was
lost. Under the fixed regime (F-V7/F-V8 repaired ladder) growth is *perfectly* modular in
parameters and in the routed-prefix diagnostic, and still useless at inference without an
external language ID.

## 4. Mechanism

`k_sparse_ratio = 0.0` in these configs, so `_k_sparse_relu` never runs: there is **no top-k**.
The latent path is `relu(x @ encoder)` summed against the decoder over **all N candidates**
(`bdh.py:252-270`), and `bdh.py`'s own header notes that the retained set of old activations
changes with width. Growing N from 8192 to 47104 per head therefore adds terms to an existing sum
while every old weight stays bit-identical — enough to move logits hard enough to wreck PPL by an
order of magnitude. Interference here is **additive superposition into a shared readout**, not
synaptic overwrite.

## 5. Calibration null: energy cannot route (#141 proposal, first arm)

Instrumented copy of the forward path (`exp3_calib_null.py`; faithfulness verified — en free
30.73 / masked 2.29 reproduce §3), measuring P(correct territory | x) over 20 domains:

| selector | accuracy |
|---|---|
| block mean | 1/20 |
| block max | 0/20 |
| cumulative prefix max | 1/20 |
| cumulative prefix mean | 1/20 |

Every selector chooses **block 0 (English base territory) for every input**. Base neurons carry
2–12× the activation mass of any language's own block (own-block mean rises monotonically with
phase, es 0.27 → lt 0.48, tracking recency rather than identity). The extreme-value bias that
normalization is meant to fix is real but second-order here (later prefixes beat en's own max in
0.3/20 cases); the first-order problem is that **activation energy carries no language-localized
signal**. A normalized-energy router of the form `s_j(x) = mean_{i∈N_j} a_i(x)` therefore cannot
work regardless of normalization. Selection must come from **functional contribution** — hence
`exp4_selfnll_selection.py`: label-free argmin-NLL over the 23 cumulative prefixes on a
calibration slice, scored on a disjoint test slice. Pre-registered bar: ≥90 % correct-prefix
accuracy **and** within 10 % of oracle PPL. **Status 2026-09-10 (later): the harness bug is fixed and it now passes its own sanity gate.**
Root cause was pairing inputs from one buffer half with targets from the other, which drove every
NLL to chance level (~e^5.3 ≈ 198) and made argmin correctly choose the *smallest* model — a
silent instrument failure, not a scientific result. Three earlier `str.replace` patches had
no-op'd without complaining, which is why the fix took several rounds; the instrumented debug run
(free en 31.34 / masked 2.28, matching the matrix) localised it immediately. The script now aborts
unless unmasked English lands in 15–60 and oracle-masked English below 4. First cell under the
fixed code: label-free argmin-NLL selects width **8192 for English — its own prefix — reaching PPL
2.36 against an oracle of 2.36** (free eval 31.36).  Full sweep result: **both pre-registered bars PASS** — label-free argmin-NLL selected the
correct own-prefix for **20/20** domains, and the PPL under the selected prefix sits within
**±8 %** of each language's acquisition exit (mean |penalty| 3.31 %, median 2.12 %). Example:
bg 6.18 vs exit 6.09, el 6.45 vs 6.36, hu 3.00 vs 2.88, pl 3.25 vs 3.01. Unconstrained free eval
on the same checkpoint gives 20–76 for those languages. So on a closed set of 20 prefixes, with
~1 MB of calibration bytes per input, internal likelihood identifies the right submodel where
activation energy could not (A1: ≤1/20).

Two self-reported defects found while scoring this: my `ppl_oracle` column used an off-by-one
own-width formula (`(POS+3)*BLK`), which truncated each language's newest block and inflated the
apparent oracle cost (lt 61.36 vs its true 3.83) — hence the nonsense `-89.7 %` summary line in
the raw log; the selection column itself was unaffected, and the table above is scored against
the independent exp-3/matrix exits. And the earlier chance-level pairing bug (§5 above).

Honest limits on this positive result, in descending order of importance: (i) selection consumes
~1 MB of same-domain calibration bytes — a token-budget sweep is required before anyone calls this
"automatic", since real deployment offers far less; (ii) 23 masked forward passes per input is up
to 23× inference cost, so any compute claim must net that out (binary search over widths would
cost ~5 passes and needs its own accuracy check); (iii) there is no rejection option, so a 21st
unseen language will be forced onto some existing prefix rather than flagged as novel — that is
experiment A3, together with code-switched inputs and per-window versus per-sequence decisions.is the next action.

## 6. Cross-checks on the fixed-capacity arm (FCS, `bdh@3c9510b`/`d76ca5f`)

Recomputed independently from the raw 21×20 log attached to bus #138: de 33.06 vs zero-shot 32.05,
bg 18,612.86, el 10,927.74, en final 29.02, acquisition floor 1.54–2.29 — **all reproduce**.

One interpretation correction: "erased language-specific learning entirely" holds for **9 of 19**
domains (es fr de da pt it sv nl fi at ≥95 % of their own zero-shot level). Ten remain clearly
better than zero-shot (pl 0.34×, sl 0.35×, cs 0.37×, sk 0.43×, ro 0.61×, hu 0.78×, et 0.87×).
Erasure is family-structured, not total — which strengthens P-FCS-3 and weakens the sweeping
version of P-FCS-1.

On F6 (two-arm re-acquisition): endpoints after a fixed 2k steps cannot separate "destroyed" from
"intact but slowly re-accessible"; both saturate, and the control arm (en-only base reaching bg
1.70 having never seen bg) shows the European-primed prior does nearly all the work. The reported
delta (1.8 %) sits below the 2–4 % seed floor, i.e. the arms are indistinguishable — supporting
"no measurable difference", not "nothing left to recover". Steps-to-threshold curves would settle
it. Its scoping to FCS is correct and does **not** transfer to RA2b, where §3 shows exact recovery.

## 7. What this changes in the narrative

1. Retention under growth is **lossless in parameters** and **conditional on submodel selection**.
   Any claim of continual learning must say whether a mask was supplied at inference.
2. Freeze-component and weight-protection experiments are answered negatively: everything relevant
   is already frozen, so protection has nothing to damp. Consolidation-style `α_i = 1/(1+c_i)`
   cannot address an additive-readout artifact.
3. The open engineering question is prefix disambiguation at the readout (likelihood-based or
   width-normalized selection), which needs **no retraining**.
4. Prior art is now load-bearing: freeze-append-grow-select-submodel is structurally close to
   Progressive Networks (Rusu 2016), PackNet (Chang 2019), Expert Gate (Mallya & Lazebnik 2018),
   PCANets, SupSup/HSP mask allocation. The novelty claim must be argued against those before
   submission, not after.

## 8. Reproduction

```bash
cd /srv/coding/bdh
.venv/bin/python scripts/pi50/matrix_eval.sh preflight          # runner + guards
.venv/bin/python scripts/pi50/exp3_delta_m.py                   # churn census (CPU, mmap)
.venv/bin/python scripts/pi50/exp3_encoder_churn.py             # encoder-side control
.venv/bin/python scripts/pi50/exp3_masked_vs_free.py            # §3, the decisive run
.venv/bin/python scripts/pi50/exp3_calib_null.py 8              # §5, instrumented energies
.venv/bin/python scripts/pi50/exp4_selfnll_selection.py <ckpt> 25   # §5 follow-up
```

Method notes worth keeping: `awk` field indices on human-readable eval output are a trap — parse
with regex; always state the aggregation operator (block-mean vs block-max differ); verify a
monkeypatch actually fired before quoting numbers derived from it (mine silently did nothing until
the dead-code branch was found); Python block-buffers piped stdout, so use `-u` or `stdbuf -oL`
before concluding a background job died.
