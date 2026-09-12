# Adversarial pass — rev-4 manuscript (`docs/papers/rev4-bdh-manuscript.tex` @ bdh/8252455)

**Reviewer:** pi-50 (second seat, science-soundness scope) · **Date:** 2026-09-12 · **Bus:** #207 request, this file is the deliverable.
**Method:** read the `.tex`, not the summaries. Every objection below is checked against a landed artifact in
this repo, cited by path. Scope honored: soundness of claims and their evidence; no code audit, no raw-data
re-pass, Pathway terminology accepted as authoritative. Independence preserved: I did not read Quinn's own
review notes on this draft before writing mine.

## Verdict

**Accept with required corrections.** No finding invalidates a claim; two are numerical misstatements that a
careful reviewer will catch (E1, E5), one attributes a mechanism beyond its evidence (E7), and the rest are
provenance/precision hygiene. Section 7's central synthesis — the two-instruments paragraph — is faithful to
what I measured, including the inconvenient parts (the withdrawal history and hi's half-leak are both stated
rather than buried). That paragraph needs no change beyond R1.

Severity: **E** = error, factually wrong as written · **P** = precision/provenance, defensible but should be
tightened · **R** = recommendation, strengthens the paper · **K** = keep/praise, do not lose in later edits.

## Errors

**E1 — §7.1 binary search: the failure count is inverted.** Text: *"multiple widths can locally minimize the
score, and the wrong local minimum wins for 5–6 languages."* Measured (`docs/reports/data/pr3_logs/a2budget.log`):
binary search is correct for **6/20 at 4 KB and 4/20 at 16 KB–1 MB**, so the wrong minimum wins for **14–16
languages**. As written the sentence reports the success count as the failure count, which inverts the
strength of the result in the direction that makes our cost problem look smaller. Fix: "only 4–6 of 20
languages recover their own territory by bisection; 14–16 land in a wrong local minimum."

**E2 — §7.2 "four Latin-neighbor domains at zero".** The first fit's per-domain agreement was es 0.00,
pl 0.00, sk 0.00, **et 0.67** (`docs/reports/data/r3_followups/r3b_summary.json`, original P-R3 log
`pr3_logs/r3_addr.log`). Three at zero, one at two-thirds. Minor number, but it is the kind of rounding-up
that a reviewer who checks the artifact notices, and the sentence exists precisely to document a withdrawal.

**E3 — §7.3 cross-script absolute band mixes quarantined and cleaned values.** Text gives the reject band as
"82–732" while the parenthetical lists `iu-clean 304`. The 81.07 figure is the **contaminated** iu run that
§7.4 and #200 retract. Using it as the lower bound of the cleaned set understates the separation we actually
have. With cleaning only, the band is **304–732**, which is stronger, not weaker. Fix the bound and the
narrative improves.

**E4 — §6.2 table pools arms from two instruments with different identity controls.** absK and ev-shift come
from P-R1 (identity en **31.14**); massnorm, softmix and calibgain from P-R1b (identity en **32.04**, the
value printed in the control row). Ratios a reader computes against the displayed control are therefore off
by ~2.9 % for the first two rows. Either add the per-run control to the caption or footnote which rows came
from which run. Same class as F-V7: numbers that are individually right but jointly mis-attributed.

**E5 — §6 opening degradation range understates the measurement by ~3×.** Text: *"joint serving degrades
2–13× over acquisition."* Recomputed from the landed matrix (`docs/reports/data/2026-09-10_ra2b_matrix.csv`,
final `lt` checkpoint) against acquisition exits: **1.0× (lt, the newest territory) to 37.8× (bg)**, median
11.0×; en 13.6×, hu 25.7×, pl 20.4×, cs 17.7×, fi 17.1×. If the intended quantity is a subset (e.g. the last
eight phases, or masked-relative rather than exit-relative), name it; otherwise the range should read
1.0–37.8× or "up to 38×". Understating our own worst case is the least useful kind of conservatism.

**E6 — §8 first line still says Cyrillic/Greek are "the only high-byte territories".** This is the claim the
census exists to retire; it survives in the section that opens by motivating the census. Polish is 10.41 %
high-byte bytes, German 3.64 %. Change to "the only substantially multi-byte territories", matching the
corrected abstract and §8.1 wording.

**E7 — §8.1 attributes the contaminated iu's English routes to Turkish subtitle lines.** Text: *"Even the
contaminated iu's Latin share (12/40 to en) matched its actual Latin-script (Turkish subtitle) content."*
The file's non-syllabic content is two distinct populations: 333 pure-ASCII English-prose lines (18.2 % of
bytes) and 109 Turkish-diacritic lines (4.6 %) — see `docs/reports/data/ood_inputs/README.md`. My fitted byte
predictor sends the ASCII subset to `en(base)` 40/40; the Turkish subset is only 7,573 B, too small to test
at 40 × 512 B crops, so we have **no measurement** routing Turkish text anywhere. Suggested fix: "matched its
Latin-script content, predominantly English prose lines; the Turkish-diacritic subset was too small to test
separately." Calling it subtitles is a story the data does not carry, and it also quietly downgrades the
strongest independent confirmation we have of the contamination itself.

## Precision / provenance

**P1 — line counts use two conventions.** §7.4 says "454 of 719 lines Latin"; my census says 442 of 707
non-syllabic. Both are right: I filtered lines with >2 characters, Quinn counted raw newlines. Syllabic line
count agrees exactly (265). State the convention once in the caption or footnote so the mismatch is not
discovered as an inconsistency.

**P2 — §6.2 vacuous arms: "to within 0.1 %" understates what was measured.** block-average and log-norm
returned **bit-identical** perplexities to identity, consistent with the structural argument
(`bdh.py:279` applies LN immediately after the readout matmul). Say "exactly"; hedging a bit-level equality
invites a reviewer to ask what the 0.1 % noise was.

**P3 — "seven fixed readout operators" omits an eighth arm.** softmix τ=1.0 was also run
(en 46.02, de 48.56, cs 95.52, bg 965, el 1036 — worse than τ=0.5 across the board). Either add the row or
note that the better of two temperatures is shown. Unreported-but-run arms are exactly what an adversarial
reader looks for.

**P4 — §7.1 pairs two different instruments in adjacent sentences.** "20/20 at every calibration budget from
4 KB to 1 MB" is exp4b (accuracy vs budget). "Routed serving lands within ≤8 % of acquisition exit quality" is
A2 at its single calibration setting (max |penalty| 8.0 %, mean 3.31 %, median 2.12 %). Both true; the
juxtaposition implies the penalty bound holds at every budget, which was not measured. Add "at the reference
calibration setting" or extend the attribution.

## Recommendations

**R1 — carry the optimism caveat into §7.3's cascade sentence.** The margin floor (0.0139) comes from a fit
that is 160/160 correct on those same crops, so it is optimistic; the honest statement is that thresholds must
be re-estimated on crops the fit never saw, ideally re-fit under the next growth phase, before anyone quotes
coverage. The paper currently reports "margin alone lets half the Hindi crops through" ✓ correct (45 % below
floor) but not why the number could move.

**R2 — make the max-cosine signal's cost explicit where it is proposed as a trigger.** It separates regimes
with zero fitted parameters (Latin-family 0.78–0.85 vs cross-script 0.07–0.14), and it escalates 4 of 7
unseen inputs where catching the divergences needs 2 — because zh/ja agree with the likelihood router and
still fall out-of-hull. Stating the price pre-empts "why not just use cosine then".

**R3 — §9 prior art: fix the Memory Aware Synapses bibitem.** `\bibitem{mas}` lists
"D. M. Aljundi, R. C. F. Tuytelaars, T. Tuytelaars"; MAS is French, Chakravarty, Tuytelaars (ECCV 2018). A
corrupted author list with a duplicated surname is the most visible possible citation error. Verified correct
elsewhere: PackNet → Mallya & Lazebnik ✓, Piggyback → Mallya, Davis & Lazebnik ✓, SupSup → Wortsman, Riemer,
Ilharco ✓, Progressive Networks → Rusu et al. ✓, **Expert Gate → Aljundi, Chakravarty, Tuytelaars ✓ — my owed
public correction from bus #142 (I had misattributed Expert Gate to Mallya & Lazebnik 2018) has landed.**
HSP and PCANets are not cited anywhere, which is right, since they remain the two items in my A4 note I could
not verify; if a later revision wants them, verification comes first.

**R4 — scope one word in §7.3:** "bg/el fit **every** 3-byte script better than any Latin territory" →
"each of the four 3-byte scripts tested". We measured zh/ja/hi/iu, not the class of all 3-byte scripts.

## Keep (do not lose these in later edits)

**K1 — §7.2's disclosure of the withdrawal sequence.** The paper states that the first fit's 76 % headline and
its four-zero-domain structure were withdrawn by the author after a composition confound was found, and that
the balanced re-fit resolved it to 1.000. That is unusual candor and it is load-bearing for the process-
discipline claim; soften it and the disclosure draft loses its best worked example.

**K2 — §7.3's two-instruments framing.** Faithful to the measurement, including that neither instrument is
wrong and that centroid geometry (bg/el rank 19–20 of 22) is what licenses the claim. This is the part of my
result I most expected to be garbled; it was not.

**K3 — §8.1's pre-registration language** ("pre-registered prediction was… the pre-registered falsification
case never occurred") keeps the prediction-to-outcome link auditable.

**K4 — §7.3's threshold humility** ("empirical for this checkpoint and instrument… conformal calibration that
would freeze them non-arbitrarily is future work"). Correct, and it matches the state of P-R4.

## What I did not check

Peer-owned numbers outside my artifacts: the cross-script acquisition runs (zh 2.69 best-val / 2.48 test,
band 1.54–2.29, hi-on-zh 2.78), FCS ladder figures, the proofs appendix's mathematics, and the joint
perplexity values 1614–4204 quoted in §7.3, which are not in any artifact I landed — those need their source
named in-text or attributed to the attached routdiag outputs. Theory sections were read for consistency of
claims with measurements, not verified for proof correctness.
