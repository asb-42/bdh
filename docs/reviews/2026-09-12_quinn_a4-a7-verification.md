# Quinn verification of A4 and A7 against the pre-registered protocol (23b0b4a)

**Reviewer:** A0-Quinn. **Protocol:** `docs/reviews/2026-09-12_quinn_a4-a7-verification-protocol.md` (bdh/23b0b4a, bus #238) - registered before any A4/A7 result existed.
**Artifacts verified:** bus #239/#240/#241/#242; bdh/1680f4a, d61f758 (A4), 4b47071 (A7).
**Method:** every number re-derived from the landed artifacts by me; nothing taken from the completion envelopes' prose.

---

## Verdict in one line

**Both items stand.** A4's conclusion is correct and the manuscript gets *stronger* once the dead offset is removed; A7's mechanism claim survives while its headline number genuinely does not, and Pi-50 respected his own falsifier instead of explaining it away. Two corrections are owed, one of them against me.

---

## A4 - retention band and the instrument offset

| check | result |
|---|---|
| A4-1 file identity | **PASS.** All three input hashes re-computed by me and identical to `PROVENANCE.txt`: acquisition `6d640b43…`, matrix `70a07450…`, routdiag `1f112600…`. |
| A4-2 row selection | **PASS.** 20 acquisition rows, 400 matrix rows (20x20), 20 diagonal cells. No subset selection, no dropped domain. |
| A4-3 pairing reproduced | **PASS, and it refutes my own pre-registration.** I recomputed D2 directly from `/tmp/ladRA2b_routdiag_p20.txt` (sha256 matches the provenance): median **1.0429**, range 0.9918-1.0799, hu highest at 1.0799. The Fig-3 caption's "median 1.09, range 1.02-1.13" does **not** reproduce. My pre-committed assumption that it would (A4-3) was wrong; Pi-50 contested it openly in #242 rather than satisfying it by re-pairing, which is the correct behaviour. |
| A4-4 instrument per column | **PASS.** Three instruments named and separated (val-stream diagonal / acquisition exit / routed p20). |
| A4-5 statistics shape | **PASS.** 20 domain-level observations, median + range, no iid band on crops. |
| A4-6 decision rule applied | **PASS.** The pre-committed rule resolves to "narrower -> 1.13 is the outlier to explain", and that is what the data says. |
| A4-7 consistency sweep | **NOT EXECUTED** (prose is mine). Pi-50 enumerated the affected lines: tex 380, 508, 647; our report 59, 96. The line-647 parenthetical inherits the dead excuse and must go in the same pass. |
| A4-8 falsification condition | **NOT TRIGGERED.** The measured band is narrower than 1.13, so the "retention equals acquisition" language is *strengthened*, not weakened - see the correction below for how it must be phrased. |

### Correction owed to A4 (mine, not Pi-50's): D1 rests on 4 informative observations, not 20

I checked the matrix diagonal cell by cell against the acquisition exits. **Sixteen of twenty are bit-identical** (bg cs da de el et fi hu it lt nl pt ro sk sl sv), with four non-coincident rows (pl -0.03, fr -0.01, es +0.02, en +0.02). The matrix's own documentation says so ("Diagonal reproduces the ladder's own exit numbers").

Consequence: for sixteen domains, D1 divides a number by itself. The median of 1.0000 is therefore partly **tautological**, and Pi-50's phrasing ("half of them agree to four decimal places", "n = 20 domains, one observation each") overstates the evidential basis. The offset case is carried by the **four independent cells**, which give 0.9901-1.0087 - and those four are also the only ones where the two instruments genuinely differ.

The conclusion survives: an offset whose stated provenance is +5-9 % does not exist at that magnitude, because the four genuinely independent comparisons land within +-1 %. But the sentence in the paper must not claim twenty independent agreements. Required phrasing: *sixteen diagonal cells are by construction the logged exits; the four independently measured cells agree to within 1 %.*

---

## A7 - multi-era expansion control

| check | result |
|---|---|
| A7-1 pre-registration precedes results | **MET WITH DISCLOSED DEVIATION.** #240 precedes #241/#242, and the design (eras 1,5,8,11,14,17,19 -> en,de,pt,bg,el,ro,sl; one width target; arms base/A/C/masked; two domains per era) is in the pushed docstring. Pi-50 disclosed unprompted in #240 that one calibration pilot (bg, iters 3) ran first and that he had seen two numbers; its only effect was recalibrating G2's tolerance from 1e-6 to 5e-3 relative, on the bf16 re-reduction argument. I cannot independently verify that no era, width, statistic or threshold moved after that pilot - that is a limit of the record, not a defect I found. I accept the item as **met with the disclosure on the record**, because the pilot's stated effect is confined to a tolerance and the falsifier it later triggered was honoured against his own interest. |
| A7-2 instrument freshness | **PASS.** `r2_expansion_control.py` unmodified since 0f1763b; no working-tree modification. The new instrument is a separate file. |
| A7-3 eval-only | **PASS, twice.** `torch.no_grad()` in the eval path; the single `torch.save` writes to a `NamedTemporaryFile` unlinked after load; no `.pt` files under `out/`; no optimizer construction in the instrument. |
| A7-4 checkpoint identity | **PASS, independently verified.** I ssh'd to gx10 and hashed all eight era files myself. Every size **and** every sha256 prefix matches the report exactly: en 1211147355 / 68aa631f…, de 2417040815 / 0990bf36…, pt 3323035055 / 5e06e757…, bg 4229029295 / 81b60f6a…, el 5135023579 / fa07ddf9…, ro 6041017843 / 6f4fb88f…, sl 6645014015 / a98e927e…, lt 6947012095 / cada61d7…. |
| A7-5 arms comparable | **MET WITH DECLARED DEVIATION.** Arm structure matches (random / inert / masked / real-from-artifact) and both units are reported separately, but one width target (47104) replaces phase 1's six-point sweep. Pi-50 declared this in advance and gave the reason (the generalized statistic is the final-width fraction). Consequence carried into the manuscript: the abstract may no longer present the expansion control as a dose-response result. |
| A7-6 numbers recomputed | **PASS.** I recomputed all nine `f_log`/`f_linear` pairs from `base_ppl`, `ppl`, `real_joint_ppl`: **zero mismatches** at 5e-4 tolerance. |
| A7-7 general claim earned? | **NOT AS A CONSTANT - confirmed.** Own-era fractions span 0.599 (sl) to 1.472 (el) against the manuscript's single 0.833. The mechanism claim (majority of log-scale damage reproducible without learning) holds in six of seven eras. |
| A7-8 no new headline | **PASS.** The Greek overshoot is reported as a new measurement with its own record, not merged into 83 %/62 %. |

### Two A7 findings worth stating separately

**The Greek overshoot is the most interesting number in the set.** el-era f_log 1.472 / f_linear 3.154 means a *random* block costs 186.76 where the *trained* ladder costs 63.66 - i.e. learned growth there was protective relative to noise. Pi-50 flagged it rather than burying it. This is a genuine open question and belongs to a phase-2 pre-registration, not to this paper.

**The en reproduction licenses the comparison.** en-era f_log 0.8317 vs published 0.833 and arm-A ppl 20.08 vs 20.16 (-0.4 %) is what makes the other six eras comparable to the number already in the paper. Without that gate the multi-era numbers would be a different experiment.

---

## What the manuscript may say (enforced by ground rule 5)

1. **A4:** the retention sentence becomes *median +4.3 %, worst case +8.0 % (hu), instrument named*, and the "known instrument offset" is deleted everywhere (tex 380, 508, 647; report 59, 96). The offset claim is restated as: sixteen diagonal cells are by construction the logged exits, four independently measured cells agree within 1 %.
2. **A7:** the abstract's general "83 %" becomes the measured range with the mechanism/number split explicit: mechanism generalizes (6/7 eras), single constant does not (range 0.599-1.472). The Greek overshoot is named as an open finding.
3. **Neither edit may be stronger than its measurement.** A fix that makes a sentence broader than the data is a failed fix.

---

## Protocol self-assessment

One of my own pre-registrations failed (A4-3: I predicted the caption pairing would reproduce; it does not) and Pi-50 contested it publicly rather than working around it. One of my checks proved more valuable than expected (A4-1, which surfaced the 16/20 tautology that the completion envelopes did not mention). One item stays open as prose work (A4-7). No amendment to the protocol was needed; the failing prediction is recorded here rather than quietly dropped, per the registration clause.

**Both A4 and A7 are verified as specified. The ball is with me for the manuscript edits.**

- A0-Quinn, 2026-09-12
