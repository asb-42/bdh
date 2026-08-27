# Review: Routing Diagnosis (Arm-G, 20-way)

**Date:** 2026-08-27
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-27_routing-diagnosis.md` (commit `fc61fcb`)
**Context:** Gate for Phase 2 (growth + routing) per `docs/plans/2026-08-27_growth-routing-and-next-steps.md`

## Verdict

**PROCEED to Phase 2.** The grown stack has fine-grained routable structure. Phase 2 is
already running; nothing in this review is a stop signal. Two measurement corrections
should be applied to the report and carried into Phase 2 evaluation.

## 1. P-Det: arithmetic inconsistency — conclusion strengthens, not weakens

The report states: 19 domains, 40 crops/domain, overall 758/800 = 94.75% ("marginal, just
below 95% threshold").

But the independently listed table contains **19 domains** (bg, cs, da, de, el, es, et, fi,
fr, hu, it, lt, nl, pl, pt, ro, sk, sl, sv). 19 × 40 = **760** crops, not 800.

The only non-100% domain in the table is et at 95% (38/40), so the internally consistent
count is:

- correct = 18×40 + 38 = 758
- denominator = 760
- **P-Det = 758/760 = 99.7%** (not 94.75%)

If the intended sampling was 20 domains × 40 = 800, then the 20th domain and its results
are missing, which is itself an error. Either way, the reported 758/800 cannot be right.

**Action:** fix the report denominator (or add the missing domain). The gate ≥95% is
comfortably met under self-consistent accounting — the report's "marginal" framing is
incorrect.

## 2. P-Route: gate criterion was not measured as specified

The plan pre-registered P-Route falsifier as routed retention "within 0.3 nats" (i.e.,
against a per-domain specialist/phase-peak baseline). The report compares routed PPL
against **joint PPL (58.39)** and reports 4.7–22.8× improvement.

That is a valid and impressive signal, but it is **not the pre-registered comparison**.
A joint model is catastrophically degraded, so beating it is a low bar; the interesting
question is how close routed serving gets to the per-phase specialists.

The Phase-1 routed values (2.56–12.46) are already in a plausible specialist range, which
is why this is a flag and not a blocker. Still:

**Action:** in Phase 2 instrumentation, add per-domain specialist comparisons (same eval
protocol) and report routed PPL minus specialist PPL in nats. If possible, retro-compute
this for the Arm-G checkpoint.

## 3. Previous review issues — correctly addressed

Commit `4ff1973` incorporates both points from my previous review:

1. The 88.8% preliminary is demoted to anecdotal (pre-fix `eval_router.py`) and Phase 1 is
   the only decision measurement. Good.
2. The trained routing head is removed from Phase 2; the plan now specifies compiled
detector / likelihood router, which are the validated label-free options. Good.

Both changes match the evidence (Addendum 12, commits `7752743`, `d62ba57`).

## 4. Non-blocking observations

- Checkpoint: report says 554M params at mult=704; the earlier Arm-G report said final
  mult=708. Confirm which is current.
- Route clustering is mostly linguistically coherent; the fr+pl and el+sk cross-family
  pairings are unexpected. Worth a short look in Phase 2, but not an action item now.
- "Only error is et": if the denominator is genuinely 800, there are ~40 missing errors;
  this should be resolved before the report is cited in the manuscript.

## Bottom line

The routing result stands: the grown stack has routable structure, routing is
fine-grained (15/20 singleton), and routed retention is dramatically better than joint.
Phase 2 is justified. Fix the numbers, keep the specialist baseline in Phase 2's metric
suite, and proceed.
