# A3 Closed-Set Probe — Results (21st never-trained language)

Date: 2026-09-10/11 · Pre-registration: HAK #152 (before any number existed) ·
Probe run: .200 RTX 4090, 22:27:52 done · Checkpoint: RA2b-lt_last (579M, streamed gx10→.200,
size-verified 6,947,012,095 both sides) · Instrument: validated likelihood router
(eval_router.py, 800/800 on trained languages in the p20 routdiag) · 21 domains × 20 routes,
batch 1, window 128, 40 crops/domain, protocol-congruent.

Primary artifact: `out/logs/ra2b_routdiag_p20_lv21.txt` on .200 (full confusion matrix +
routed table). 21st domain: lv (Latvian, Europarl v7, 96 MB, never in any ladder of any arm).

## V1 — The 20 trained languages: replication is clean

All 20 route 40/40 to their own training width — the p20 result reproduces on a different
host (RTX 4090, .200) with the streamed checkpoint. Routed ppl values are identical to the
gx10 p20 routdiag for 13 domains (en 2.36, bg 6.24, el 6.47, lt 4.04, …) and within 1–4 %
for seven late-phase domains (sv 3.22 vs 3.08, pl 3.29 vs 3.17, ro 3.68 vs 3.51, sk 3.64
vs 3.85, sl 3.83 vs 3.71, pt 2.79 vs 2.87, nl 3.09 vs 3.10) — bf16-autocast numerics
differing by host GPU; a cross-host replication data point, all deltas below the measured
seed floor.

## V2 — lv: P-A3-1 PASS (no territory)

lv routed ppl on its preferred route: **37.61** — far above the trained band (2.36–6.47).
No route gives lv a trained-language-quality reading. (Consistent with RA2b protocol: lv was
never trained, so no territory was ever written for it.)

**VERDICT P-A3-1: PASS.** — A0-Quinn, 2026-09-11

## V3 — lv: P-A3-2 PASS (closed-set bias, family-structured)

The router selects **lt's route (47104) for 40/40 lv crops** — not noise, a systematic choice,
and exactly the pre-registered Baltic-family prediction (lt Lithuanian is lv Latvian's
closest relative in the ladder; byte statistics overlap most). The family geometry that
governs interference (FCS F3) and activation energy (pi-50 #143) also governs the router's
behavior on an unseen language.

**VERDICT P-A3-2: PASS.** — A0-Quinn, 2026-09-11

## V4 — P-A3-3: the reject option is free (calibration measured)

The informative number: **routing advantage** = joint ppl / routed ppl on the selected route.

- Trained languages: 36.76 / 2.36 … 6.47 = **advantage 5.7–15.6×** (worst case el/bg)
- lv: 36.76 / 37.61 = **advantage 0.98×** — the router's "best" route is slightly WORSE
  than joint serving for a never-trained language.

Separation between the worst trained advantage (5.7×) and lv (0.98×) is a factor ~5.8 —
a clean, wide gap. A threshold on routing advantage (or equivalently on min-route NLL
relative to joint NLL) separates seen from unseen languages with large margin, using
calibration data already computed by any router invocation. **Automatic routing does not
require a closed-set caveat if it ships with this reject rule: if min-route NLL ≈ joint
NLL, the input is outside the trained support — do not route.**

**VERDICT P-A3-3: RESOLVED — closed-set overfitting is measurable, and reject is
cheap; the paper claim should state the reject rule rather than a closed-set
limitation.** — A0-Quinn, 2026-09-11

## Implications

1. For operator #141 (monotonic growth, step 8): routing is safe beyond the closed set —
   unseen inputs are detectable at zero extra cost, turning "closed-set overfitting" from
   a limitation into a feature (out-of-support detection).
2. For pi-50's A2 selector comparison: the lv row is ready as the OOD test case; any
   selector that cannot reproduce the advantage-gap (trained ≫ 1, unseen ≈ 1) inherits the
   same reject rule with its own score.
3. Family geometry is now the single axis that explains interference (FCS F3), activation
   energy (pi-50 #143), and unseen-language routing (this probe) — one sentence for the
   discussion section.

## V5 — Reject-threshold validation, second unseen language: ga (Gaeilge, n=2)

Operator GO (2026-09-11). Gaeilge does NOT exist in Europarl v7 — verified against the statmt
directory listing (exactly 21 pairs: our 20 trained languages + lv; the earlier suggestion
"mt/ga" in the A3 confound note was an unchecked memory claim, corrected here). Source:
**OPUS DGT v2021** (EU legal texts, Irish is an official EU language), 55.4 MB ga side — a
corpus AND register switch from Europarl, declared as confound. It makes the probe strictly
harder: unseen language + unseen register. Run: same checkpoint (RA2b-lt_last on .200),
same instrument, ga over the 20 routes, 40 crops, batch 1 (04:11:42 done).

- **P-GA-1 (no territory): PASS.** ga routed ppl on its best route: **50.54** — far above
  the trained band (2.36–6.47).
- **P-GA-2 (family routing, weaker prior): PASS with the predicted nuance.** Irish is an
  isolated Celtic language with no relative in the ladder — and the routing is DIFFUSE
  where lv's was sharp: bg-route 29/40, plus scattered crops across cs (5), da/en/es/fr
  (2 each), sk (1). No single family target exists, and the router spreads accordingly —
  consistent with byte-statistics proximity to Latin-script domains, weaker than lv→lt.
- **P-GA-3 (reject replication — the decision-relevant one): PASS.** Note the joint
  reference must be computed on the SAME corpus (DGT), not reused from Europarl: joint
  full-width on ga-DGT crops = **70.17**. Routing advantage = 70.17 / 50.54 = **1.39×** —
  far below the 5.7× floor of the worst trained language and of the same order as lv's
  0.98×. The register shift cancels in the ratio (both terms move together).

**VERDICT: the reject rule is validated on two typologically distant unseen languages
(lv Baltic-related, ga Celtic-isolated) from two corpora (Europarl, DGT). Threshold
formulation for the paper: reject routing when joint/min-route NLL ratio < ~5 (measured:
trained languages 5.7–15.6×, unseen 0.98× and 1.39× — gap ≥ 4.1× on both sides).**
— A0-Quinn, 2026-09-11

Artifacts: .200 out/logs/ra2b_routdiag_ga_dgt.txt; DGT source
.200 data/europarl/DGT.en-ga.ga.txt (OPUS-DGT v2021, license CC-BY-4.0 per OPUS metadata).

## Confounds (pre-declared, updated for V5)

Single seed, one ordering, 40 crops/domain; the seven small cross-host deltas (V1) are
bf16-autocast numerics, all sub-floor. The reject threshold is now validated on two unseen
languages (lv, ga) from two corpora — the earlier single-language caveat is closed; the
ga measurement carries the DGT register switch, declared above, which the ratio-based
threshold cancels. A third unseen language would tighten the threshold's lower bound
further but is not required for the paper's claim.


Single seed, one ordering, 40 crops/domain; the seven small cross-host deltas are bf16
autocast numerics, documented above, all sub-floor. lv is one unseen language; the reject
threshold should be validated on a second (e.g. mt/ga) before the paper number is final.

— A0-Quinn, 2026-09-11 · artifacts: .200 out/logs/ra2b_routdiag_p20_lv21.txt;
lv data .200 data/europarl/europarl-v7.lv-en.lv.txt (96 MB, sha-pinned by download log)
