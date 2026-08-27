# Review: cl-bdh-manuscript (Revision 2)

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/papers/cl-bdh-manuscript.tex` at commit `013b559` (revision 2, accumulation study)
**Context:** This review precedes Phase 3 (manuscript update). Phase 2 (growth + routing)
is running; if it changes the conclusion, fold that in before submission.

## Verdict

**Major revision before submission — one factual inconsistency, one reproducibility gap.**
The scientific core is strong and the epistemic discipline is exemplary. Both issues below
are fixable in one work session.

## 1. CRITICAL — Arm-G routing description is superseded by today's Phase 1

Section `Accumulation` / Arm G states:

> "The grown model *does* have routable structure ... but the routing is coarse,
> not fine-grained."

That was the status before the 20-way routing diagnosis (`docs/reports/2026-08-27_routing-diagnosis.md`, commit `fc61fcb`). Phase 1 showed:

- **15/20 routes are singleton** (one domain per route) — that is fine-grained by any definition
- P-Det = 758/760 = **99.7 %** on the fixed `eval_router.py` (report denominator was 800; corrected in `de5025b`)
- Routed retention 4.7–22.8× better than joint (58.39 ppl)

**Action:** update the Arm-G paragraph with the Phase-1 numbers and change the verdict from
"coarse" to "fine-grained under 20-way routing". Also cite the routing-diagnosis report in
Reproducibility. A reader of R2 would otherwise receive a conclusion that the team itself
has already falsified.

## 2. MAJOR — The theory's verification artifact is missing from the repo

The exact-isolation theorem and Corollary (prefix growth constructs the structure) rest on
separability facts **S1/S2** (coordinate separability; LayerNorm as sole cross-neuron
operation), stated as "verified against the implementation (masked-forward reproduction
tests)". I searched `scripts/` and found no script that runs that verification.

The claim is the load-bearing wall of the theory section. Without the script, an
adversarial reader cannot re-run it, and the paper's own reproducibility promise
("all mechanisms ship in the repository") is not met for the most important check.

**Action:** commit a small script, e.g. `scripts/verify_masked_forward.py`, that
(a) builds a small BDH, (b) runs masked-forward vs. specialist-forward on the old phase's
test inputs, (c) asserts tensor identity per level (ε_inv, ε_eq thresholds), and
(d) prints the S1/S2 verification table. Ideally reproduced as a one-line `README`
command in the manuscript's Reproducibility section.

## 3. Authorship — align with the team register

The manuscript declares "ox-alpha, autonomous research agent" as author. Per
`docs/team/IDENTITIES.md`, ox-alpha is a retired preset name. The actual producing seats
are: MiMo (execution: code, experiments, reports), Quinn (review seat: reviews, go/no-go),
maintainer ASB (compute, direction; declined co-authorship).

**Action (maintainer's call, not mine to impose):** decide before submission which name(s)
the AI contributor carries and update the Authorship section accordingly. The
Acknowledgments currently credit "four independent formal critiques by other large language
models" without naming the reviewing seat; naming Quinn as reviewer is a transparency
improvement, not vanity.

## 4. Strengths worth preserving

1. **Epistemic labeling.** Strict `proved / measured / conjectured` tags with the rule
   "measurements never discharge a missing assumption" — this is exactly the discipline
   this project set as an invariant. Keep it.
2. **Negative-results register.** Seven honest failures; especially the learned in-pass
   gates and the magnitude-ranked pruning collapse. Reviewers like this.
3. **Self-correction embedded.** The E6/E8 story (cross-corpus comparisons void, volume
   flat, composition governs sparsity) is visible in O4/O5. That is the project's INV-5
   discipline in paper form.
4. **Limitations are real.** Batch-1 inference at Arm-G scale, expansiveness as a
   directional proxy, no synthetic replay — all disclosed.

## 5. Minor notes (no action required)

- The 0.3-nat falsifier language appears in the accumulation section; consider stating it
  once in the protocol section so reviewers see it pre-registered, not only in results.
- `sequential endpoint` row in Table pareto (11.08/18.28/2.13) matches the H1 report;
  cross-checked, consistent.
- The prior-art triage table is useful; the PR#6 entry (upstream CL fork) is fairly
  positioned as not directly comparable.

## Bottom line

The paper's claim — weight isolation ≠ computation isolation, and prefix selection makes
isolation exact in BDH — is well supported for what it claims. Fix the stale Arm-G routing
paragraph with the Phase-1 result, ship the masked-forward verification script, and sort
authorship. Then revision 3 is submission-ready pending Phase 2.
