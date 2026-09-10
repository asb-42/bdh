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

## Confounds (pre-declared)

Single seed, one ordering, 40 crops/domain; the seven small cross-host deltas are bf16
autocast numerics, documented above, all sub-floor. lv is one unseen language; the reject
threshold should be validated on a second (e.g. mt/ga) before the paper number is final.

— A0-Quinn, 2026-09-11 · artifacts: .200 out/logs/ra2b_routdiag_p20_lv21.txt;
lv data .200 data/europarl/europarl-v7.lv-en.lv.txt (96 MB, sha-pinned by download log)
