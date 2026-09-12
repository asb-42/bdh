# Assessment of the first external review (Qwen3.7-Plus, relayed at bus #213)

**Reviewer of the reviewer:** pi-50 · 2026-09-12 · scope: science soundness of the review's claims about our
numbers and framing. Manuscript checked at `bdh/7a8ba99` (rev 4.1, Quinn's pass over my #210 findings).

## Headline

The review is broadly sympathetic and mostly accurate as a summary, but **three of its five Critical Q&A
answers are wrong in ways that would damage us if adopted**, and its single numerical verification is
performed with an incorrect formula. Its reproducibility critique is correct and actionable. Most
importantly, checking its arithmetic exposed a defect **I missed in my own adversarial pass** — see §A, which
is the only genuinely valuable thing this exchange produced and is written up first because it is against me.

## A. What the review forced me to find: the 83 % figure is under-qualified (my miss)

The review attempted to confirm our headline (`#213` §3): *"log(20.16)/log(31.07) ≈ 3.00/3.43 ≈ 0.87, which
aligns closely with the 83 % figure."* It does not, and the method is wrong. Our measurement normalizes by
the pre-expansion baseline:

| quantity | value |
|---|---|
| ours, log-scale damage fraction `ln(20.16/2.33) / ln(31.07/2.33)` | **0.833** |
| the review's `ln(20.16)/ln(31.07)` — no baseline subtraction | 0.874 |
| same comparison in **linear** perplexity `(20.16−2.33)/(31.07−2.33)` | **0.620** |

Their formula silently treats base perplexity as 1.0 (in both numerator and denominator), which is why it
lands near ours by coincidence rather than by agreement. Two consequences, in order of severity:

1. **Substantive, and mine to own:** "83 %" appears **nine times** in rev 4.1 and carries the "(log scale)"
   qualifier in exactly **one** place (§6.1 body, line 439). The abstract (34), the introduction arc (47), the
   contribution list (67), the theory section (226), the figure caption (411), the subsection heading (415),
   and the conclusion (851) all assert "83 % of the damage is arithmetic" bare. On linear perplexity the same
   control yields **62 %**. The claim is legitimate — perplexity is logarithmic by construction and
   log-scale is the right unit for additive readout terms — but a reader who computes the ratio off the table
   printed two lines above the sentence will get 62 %, and a reviewer who tries the review's formula gets
   87 %. My #210 pass audited every number I own for *value* correctness and never asked whether the
   *aggregation operator* was stated where it was quoted. That is a real gap in how I reviewed.
2. Recommended fix, in preference order: **(i)** state both fractions wherever the headline appears
   ("83 % of the damage on the log scale, 62 % in perplexity units") and define the damage fraction once in
   §6.1 with its formula; **(ii)** failing that, attach "(log scale)" to every one of the nine occurrences,
   including the heading and caption; **(iii)** do not leave it as is. Option (i) is strictly better than (ii)
   because it removes the appearance of having chosen the favorable unit.

This is a presentation defect, not a result defect: the A/C arms and the masked column are unchanged, and the
conclusion (damage requires nonzero contributions; masking restores 2.33) is unit-independent.

## B. Where the review is wrong

**B1 — Q3 collapses our open problem, and contradicts §7.1.** The answer asserts that the byte addresser
*"already achieves O(1) addressing … entirely bypassing the need to scan or binary-search the NLL surface"*,
thereby solving sublinear addressing. Measured position of the paper: the balanced fit reaches 1.000
[0.977, 1.000] **on in-support domains** (`r3d_fit_ablation.py`, arm B), and §7.3 reports that out-of-hull
inputs break it — Hindi routes to Latin territories under the byte fit while the likelihood router chooses
bg/el, and clean Inuktitut does the same. Our recommended architecture is therefore a **cascade**: cheap
addresser in-support, escalate to the O(K) likelihood scan when margin is low *or* max centroid cosine falls
in the out-of-hull band. If the operator takes Q3's framing, the paper loses its central unresolved question
and §7.1's own sentence ("selection scales linear in territory count unless a cheaper addresser exists below
it") becomes self-contradictory. Do not adopt; if we reply, quote §7.3 back at it.

**B2 — "Routed serving completely solves the degradation" contradicts a section heading.** §5.4 is titled
*"Joint serving: recovered but not solved"* (line 383). Routed serving reproduces acquisition quality
(retention 100.0–100.2 %) but requires the scan, has no rejection rule, and fails informatively out-of-hull.
Related merge-instruments slip in the same sentence: "retention equals acquisition, zero drift across a full
growth phase, 800/800 routing accuracy" stitches §5's fixed-regime routdiag sweep (line 359, peer-owned) onto
RA2b retention language, reading as though one instrument produced all three. Same failure class as my E4/F-V7.

**B3 — Q1's optimizer-generality claim is false.** *"Optimizers without decoupled weight decay (e.g., standard
Adam) … would not exhibit this."* Coupled L2 regularization adds `wd·x` to the gradient of a masked parameter,
so its update is nonzero too: frozen weights erode under coupled regularization as well. What decoupling buys
us is a **clean closed form** — the uniform product factor `c = Π(1 − lr_t·wd)` we measured (0.892636 ±
0.000002 on the G regime) — not the existence of the effect. Under coupled L2 the erosion instead interacts
with second-moment estimates and is less predictable, arguably worse. If this sentence reaches our rebuttal it
invites an easy correction from any optimizer-literate reviewer. Separately, Q1 accepts "minor memory copy
overhead" for the step-end restore; we have never measured the throughput cost of the mask-restore patch, so
our text should either quantify steps/sec with and without it or avoid adjectives like "minor".

**B4 — Q4 upgrades a caveat we deliberately left open.** Our text: fixed-2k endpoints "cannot separate
'destroyed' from 'intact but slowly re-accessible'; the claim is resolution-bounded" (lines 321–324). The
review converts the Δ = 1.8 % (below the 2–4 % seed floor) into "strongly suggesting that any latent access is
practically negligible". Absence of a measurable speed advantage at 2k steps is not evidence that none exists
at 20k — that is precisely why we wrote the caveat. The settling experiment is a longer re-acquisition arm
(≥20k steps, overwritten vs fresh-base, matched budget, ≥2 seeds given the floor), which has not been run.
Either run it or keep the text strict; do not cite the review as support.

**B5 — analogy presented as corroboration.** §6 of the review claims the decay confound "directly mirrors
industry observations of 'silent expert death' in large-scale MoE training". Plausible-sounding, unsourced, and
not something we measured. Silent expert death is normally attributed to load-balancing collapse and router
starvation, not to weight-decay erosion of masked parameters. We should not import it; if we want the
connection, it needs a citation we have actually read.

## C. Where the review is right, and sharper than our current text

- **Reproducibility of envelope citations — valid, and the highest-priority pre-release task.** Line 143–145
  promises "Every number in this paper traces to a committed artifact … cited by envelope number", and then
  makes external verification depend on a private bus. Concrete plan: (1) export the cited envelopes
  (pre-registrations, verdicts, retractions) into `docs/bus-transcript/` as an append-only JSONL so the audit
  trail ships with the source; (2) keep `scripts/pi50/phase1_manifest.py --check` in CI so a cited path cannot
  silently disappear — this guard already caught two real incidents (`#206[2]`, and the still-missing
  `features_sparse.npz` found after it); (3) ship the regeneration scripts with the frozen-instrument README
  so numbers are re-derivable from CSVs plus code; (4) state a checkpoint policy — the 1.21–6.95 GB `_last.pt`
  files and the `.200`-only `ra2b_joint_{zh,ja,hi,iu}.txt` runs are currently outside any public release, so
  either publish them or say plainly which numbers require them.
- **Byte-level specificity / transfer doubt — worth adopting as written.** Our Limitations has the germ of it
  (line 842, "relies on byte distinctness that open-domain abilities may not have"); the review states the
  consequence better: a tokenized or BPE-front-ended model may not present separable input geometry at all, and
  higher-level capabilities (reasoning, world knowledge) are exactly where byte statistics stop being
  discriminative. Promote it from a clause to a numbered limitation.
- **Scale constraint framing** — non-unimodality "may compound unpredictably at larger scales" is fair, and
  consistent with our own binary-search result (14–16 of 20 fail even at 23 widths).
- **Low-resource bound on the iu claims** — correctly reads our disclosure rather than treating 40/40 as
  population-level evidence.

## D. Governance note

This is a machine-generated review, and the empirical trace of that fact is §A: it validated our headline
number with a broken formula and reported agreement. Its verdict ("absolutely ready for public release")
carries no evidential weight and must not be quoted as endorsement in the revision notes or the AI-participation
section. Its criticisms are still worth processing — several are useful — but each has to be checked against
artifacts, which is what this document did. Per our own disclosure policy, if any of its wording ends up in the
manuscript, that goes in the AI-participation record.
