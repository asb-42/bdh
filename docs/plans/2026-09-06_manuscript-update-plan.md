# Manuscript Update Plan (A7): Decay Corrections Now vs. After RA2b

- Date: 2026-09-06
- Author: A0-Quinn (Agent Zero seat)
- Target: `docs/papers/cl-bdh-manuscript.tex` (rev 2, 815 lines)
- Status: proposal; the "today" items are mechanical corrections of already-superseded numbers; the "after RA2b" items require the readout

## Why not rewrite everything now

Every number in the manuscript that the decay-leak discoveries affect is either (a) already corrected by a repo artifact (report/erratum landed) or (b) dependent on the RA2b readout (fixed-regime retention = the new headline). Rewriting §4/§6 prose now would mean rewriting it again in ~48 h with the real numbers. What CAN and SHOULD happen today is the mechanical part: every stale number in the TeX gets a correction note or the corrected value, so no reviewer or reader can cite a known-wrong figure from the current revision.

## Tier 1 — land TODAY (mechanical, no RA2b dependency)

| # | Location (current TeX) | Problem | Fix |
|---|---|---|---|
| T1 | §6 Arm G subsection (line ~430-441): "×128→×708, ~554M", "EN = 41.2", "BG = 1436, EL = 1268", "peak 3.49 at LT" | Mult geometry wrong (actual 704, not 708 — the 708 was the computed-table error); Arm-G numbers are leaky-regime numbers (that is fine — they are what was run — but the text does not say so); sv 53.5 was never a trained-phase value (zero-shot transfer) | Correct mult to 704; add regime footnote: "All Arm-G/Arm-R/GR numbers were measured under the leaky regime (F-decay, report 2026-09-04); acquisition numbers are unaffected by the leak (S2d/MF nulls), erosion/joint numbers carry the three-way decomposition caveat (report §5/B)"; fix or footnote the sv line |
| T2 | §6 "routing is coarse, not fine-grained" (line ~439-441) | Falsified by the team's own 20-way diagnosis (99.7 % detection, §4 RA2 final report) — this was flagged CRITICAL in the r2 review (2026-08-28) and the paragraph is still in | Replace with the measured 20-way routing result + pointer to the RA2 final report |
| T3 | §6 Arm G "Final erosion" numbers EN 41.2 / ES 31.7 / BG 1436 / EL 1268 | These are JOINT-serving numbers; the routed-serving story (bg 2.50 at own route, G2) shows retention — the paragraph presents only the collapse half | Add the routed-serving half (one sentence + the RA2/G2 routed numbers) or an explicit pointer; full rebalance after RA2b |
| T4 | Limitations (line ~493-501) | Missing: the decay-leak discovery and its implication (all pre-fix arms share a confound); missing: the closed-form correction gives every cross-arm contrast a regime column | Add 3-4 sentences: the leak, the fix, the three-way decomposition, the fork-lookup provenance rule; cite reports 2026-09-03/04 + battle plan |
| T5 | Negative-results register (App., line ~783+) | The F-decay leak is itself a negative result of methodological value ("aggregate metrics cannot see silent weight erosion") + the silent-expert-death parallel (Jin et al. via Marin tracker) | Add one register entry with the Jin et al. pointer and the Marin convergent-validation citation |

## Tier 2 — after RA2b (needs the readout)

| # | Change | Depends on |
|---|---|---|
| U1 | §6 new subsection or promoted §6.4: "The decay confound and its removal" — closed form, two regimes, boundary law, counterfactual-repair lesson | Already measured (report 2026-09-04 + Appendix B); blocked only on where the narrative lives after the full readout restructure |
| U2 | §4/theory bridge: does route-awareness matter once the leak is fixed? — RA2b retention vs G2 retention is THE cell | RA2b retention numbers (H-decay-1/2/3 verdicts) |
| U3 | Three-way decomposition (decay artifact / interference / co-adaptation) replaces the binary "interference vs destruction" framing in §4-adjacent text | RA2b joint/routed milestones; the decomposition is measured but the final shares need the fixed-regime readout |
| U4 | Replicate floors: every single-cell number in the manuscript gets the 2.3-4.4 % seed-floor band or an n-notation | Landed measurements (2×2); applies at rewrite time |
| U5 | Accumulation re-verification: the 20-phase acquisition curve under the FIXED regime (RA2b) replaces or is presented beside the leaky curve | RA2b full curve |
| U6 | External positioning: arXiv 2604.09780 (Myth of Expert Specialization) — the routdiag reading must be defended or rescoped against its claim that routing reflects geometry, not expertise | pi-50's A11 pass |

## Recommended sequence

1. Today: Tier 1 (T1-T5) as a single manuscript commit titled "rev 3: decay-confound corrections (pre-RA2b)" — small, honest, reviewable.
2. RA2b readout (~Sep 7/8): battle plan Phase B — full §4/§6 rewrite with U1-U5, with pi-50's A11 folded in before skeleton freeze.
3. The manuscript skeleton (new section order, figure plan) should be drafted ONLY after A11 lands, per the battle plan A7 dependency.
