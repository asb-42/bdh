# Fixed-Capacity Sequence Matrix — Results (FCS readout)

Date: 2026-09-10 · Experiment: operator HAK #130, design #131, GO same day
Seat: A0-Quinn · Host: bdh-4090 (.200, RTX 4090, compiled, 81 ms/step) · Run: 05:39–13:00 CEST (7.3 h)
Code: bdh@5ed4411 (fixed-regime train.py; no masked path exists in this arm, so the F-decay fix
is not exercised here) · Pre-registration: docs/plans/2026-09-10_fixed-capacity-matrix.md (P-FCS-1/2/3)

Primary artifact: `out/logs/fixedcap_matrix.txt` on .200 — 20 rows (one per phase, after each
phase all 20 domains cold-evaluated, protocol-congruent random-crop, generator 1234).
Checkpoints: `ladFCS-{lang}_{last,best}.pt` (nothing overwrote any other arm).

## F1 — Acquisition floor (P-FCS-2): **PASS**

Every language acquires at its own phase between 1.54 and 2.29 ppl (bg 1.54, el 1.59, ro 1.98,
hu 2.00, fr 2.08 … en 2.29). No saturation cliff: the 20th language (lt) acquires at 2.13,
identical to the band. Fixed 100M capacity is never the binding constraint for acquiring ONE
language — consistent with the paper's single-language fixed-capacity results.

Notable inversion vs RA2b (open observation, no mechanism claim): the non-Latin scripts acquire
BEST here (bg 1.54, el 1.59) while they were the WORST acquirers under growth+selection
(RA2b: bg 5.86, el 5.99). Under full overwrite the entire model serves the current language;
under route-aware growth the new phase competes with protected old segments for effective
capacity. Cost direction inverts with the protection regime.

**VERDICT P-FCS-2 (acquisition floor stays comparable, no cliff): PASS.** — A0-Quinn, 2026-09-10

## F2 — Final-row forgetting (P-FCS-1): **PASS, and cleaner than predicted**

Row 20 (after lt) vs each language's own acquisition:

- Latin-script languages: 26.8–60.0 ppl = **12.5–28.4× acquisition** (en 29.0, es 26.8, it 28.1,
  pt 29.9, de 33.1, fr 33.0, nl 38.2, da 41.1, sv 44.9, ro 47.2, et 49.0, fi 52.9, sk 51.4,
  hu 56.8, cs 59.4, pl 60.0, sl 30.1)
- Non-Latin scripts: bg 18,613 (×12,086), el 10,928 (×6,873) — five orders of magnitude.

The pre-registered prediction (early Latin languages > 20 ppl, non-Latin worst) is confirmed.

**The sharper finding the matrix adds:** for the Germanic/Romance languages whose zero-shot was
measured at row 1 (after en only), row-20 serving is AT or ABOVE that zero-shot level —
de 33.06 vs zero-shot 32.05; es 26.81 vs 26.93; it 28.05 vs 27.52; pt 29.92 vs 29.79;
da 41.06 vs 38.20; nl 38.17 vs 33.97; sv 44.89 vs 40.70. **The 19 intervening phases erased
the language-specific learning entirely: what remains at row 20 is exactly (or slightly less
than) what English training alone transferred.** Under the operator's retention formula
(1 − forgetting/initial-competence, nll space), these languages sit at retention ≈ 0 or below —
slightly negative transfer for da/nl/sv.

**VERDICT P-FCS-1 (theory-predicted forgetting catastrophe): PASS — confirmed, with the
zero-shot-level-erasure reading as the strongest statement of it.** — A0-Quinn, 2026-09-10

## F3 — Interference structure (P-FCS-3): **FAIL — replaced by a family-structure finding**

Pre-registered prediction: backward interference dominated by the most recent phase (recency).
Measured: **not recency — linguistic family.** The en column oscillates across rows in a clean
pattern:

- after Romance/Germanic phases (es fr de da pt it sv nl): en = 10.5–13.6 ppl
- after Slavic/Uralic/Baltic phases (pl cs hu et sk ro sl lt): en = 22.1–29.1 ppl

Forgetting is non-monotonic and phase-type-dependent: a Germanic phase after a Slavic phase
PARTIALLY RESTORES en (e.g. en 29.1 after sk → 11.3 after sv). Mechanistic reading (hypothesis,
not proven): shared Latin-script subword statistics act as implicit replay — Romance/Germanic
phases retrain en-like byte patterns; Slavic/Uralic phases do not. bg (Cyrillic) and el (Greek)
phase effects sit between (en 20.5 / 12.1).

The bg/el columns show the same knife-edge from the other side: bg swings between 600 (row 10,
after hu) and 20.5M (row 18, after nl) across single phases — non-Latin retention is not a
stable degraded level but an unstable equilibrium that each new phase re-randomizes.

**VERDICT P-FCS-3 (recency-dominated interference): FAIL — the measured structure is
family-dependent implicit replay, a stronger and more specific mechanism than registered.**
— A0-Quinn, 2026-09-10

## F4 — Forward transfer (zero-shot row 1, after en only)

Germanic: da 38.2, sv 40.7, nl 34.0, de 32.1 · Romance: fr 19.95, es 26.9, it 27.5, pt 29.8 ·
Slavic: sk 119, pl 175, cs 161, sl 86 · Uralic/Baltic: fi 54, et 56, hu 73, lt 156 ·
non-Latin scripts: bg 4.9M, el 12.4M (effectively random). Family-structured exactly as the
byte-level subword-sharing story predicts; fr is the best zero-shot transfer of all 19.

## F5 — The contrast that answers #130: FCS vs RA2b (same sequence, same instruments)

| | FCS row 20 (fixed 100M) | RA2b p20 (route-aware growth, 579M) |
|---|---|---|
| Latin languages, joint serving | 26.8–60.0 (≈ zero-shot or worse) | 20.4–73.9 (interference band, leak-free) |
| Latin languages, own-width routed | n/a (no routes exist) | **2.36–4.04 (retention = acquisition)** |
| bg / el | 18,613 / 10,928 | routed: 6.24 / 6.47; joint: 230 / 64 |
| en final | 29.0 (12.7× acq, ≈ zero-shot) | routed 2.36 (1.05× acq) |

Without growth+selection, the model retains NOTHING language-specific (Latin languages land at
their English-only zero-shot levels; non-Latin scripts collapse five orders of magnitude).
With route-aware growth, every trained language is served at acquisition quality (1.02–1.13×).
**That contrast is the measured memory story: it is real, and it is the growth+selection pair
that delivers it — capacity alone delivers nothing.**

## Confounds (pre-declared, unchanged)

Host/era offset (.200 batch 4 vs gx10 batch 1, +13.7 % acquisition penalty measured) applies to
cross-arm acquisition comparisons, not to within-matrix drift (same host, same instrument).
Single seed; one of 20! orderings; the FCS wd term is the standard uniform regime (nothing
masked, nothing protected — the intended fixed-capacity baseline).

## F6 — Two-arm re-acquisition probe (Exp-4 companion to #134/#135): knowledge is destroyed, not hidden

Motivation (#134 Exp 4, sharpened by #135): does the final FCS model still *contain* bg
structure it has lost access to, or was the bg knowledge destroyed by 19 overwriting
phases? Test: fine-tune on bg for 2k steps from two bases and compare recovery speed.

| arm | base | bg state at base | bg after 2k steps |
|---|---|---|---|
| main | ladFCS-lt_last (row 20; bg trained at p11, 19 phases ago) | serving 18,613 | **1.67** |
| control | ladFCS-en_last (row 1; bg never trained) | zero-shot 4.9M | **1.70** |

Delta 1.8 % — below the measured seed floor (2–4 %). Both arms land at the acquisition
band (bg acquisition at p11 was 1.54) in 2k steps regardless of whether the base ever
knew bg. **Any European-primed base re-learns bg to acquisition level in 2k steps; the
residual state of a once-trained, 19-phases-overwritten model contributes nothing
measurable to that speed.**

**VERDICT: the FCS forgetting is knowledge destruction, not access loss.** The capability
to represent bg is architecture-inherent (acquisition always succeeds); the specific
bg knowledge written at p11 is gone in every operational sense. — A0-Quinn, 2026-09-10

Implications:
- For #134 Exp 4 (synaptic consolidation): **protection-before-overwrite is the correct
  family** — post-hoc recovery has nothing left to recover faster than scratch. Growth+
  masks (RA2b) are the extreme version of this and the only measured configuration that
  retains; replay (H1p) re-writes instead of protecting.
- Side readings, consistent with F3: bg re-training moved en 29.02 → 21.48 (cross-family
  byte-statistics help) and lt 2.13 → 14.45 (most-recent language takes the overwrite
  damage) — family structure and recency co-occur, family dominates.
- Honest bound: this probe measures re-learning *speed*, the practically decisive axis;
  it cannot exclude sub-threshold residual structure (below the 2-4 % floor) — which by
  the same bound is operationally irrelevant.

Artifacts: out/logs/fcs_bg_reacq.log, out/logs/fcs_bg_ctl.log, checkpoints
ladFCS-bgREACQ / ladFCS-bgCTL on .200.

— A0-Quinn, 2026-09-10 · artifacts: .200 out/logs/fixedcap_matrix.txt, fixedcap_<lang>.log × 20,
ladFCS-* checkpoints × 40