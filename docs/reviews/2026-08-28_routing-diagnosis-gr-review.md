# Review: Routing Diagnosis Arm G+R (20-way)

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-27_routing-diagnosis-gr.md` (commit `b3e10b5`)
**Context:** First intermediate result of Phase 2 (growth + routing combination).

## Verdict

**Strong finding, but the report's numbers need correction and the interpretation needs
one diagnostic step before the next arm is built.**

## 1. The core result is real and important

The current phase (lt) has no routable structure: 32% detection, scattered across 7
routes, routed ppl 55.10 = joint (54.96). Older languages retain routable structure.
This is the **current-phase problem**: the growth rule preserves old prefixes but does
not construct one for the language being trained now.

This is exactly the theory's prediction framed differently: **isolation is constructed,
not emergent.** The old languages are routable because their structure was constructed
when they were trained with (presumably) full attention. The last language could not
construct its prefix under the current training setup.

## 2. CRITICAL — arithmetic inconsistencies (before this enters the manuscript)

- Report: "Overall: 2/760 = 0.3% on diagonal". 2/760 is 0.26%, but more importantly
  the count is wrong. The table lists 16 fully-correct domains (100%) — not 18.
- es is 95% and et is 82%, so "18 older languages route perfectly (100%)" is false.
  Independently: 16×40 + es 38 + et ~33 + lt ~13 = ~724/760 ≈ 95.3%, not 99.7% or 100%.
- Report: "Excluding lt: 744/720 = 100%" — 744/720 is mathematically >1 and internally
  inconsistent (744 ≠ 720, and es/et are not perfect either).

**Action:** recompute the P-Det table carefully (per-domain counts, diagonal, overall).
The qualitative conclusion survives, but the manuscript must not carry broken arithmetic
again (see the routing-diagnosis review from 2026-08-27 on the same class of error).

## 3. Interpretation: freeze_attn is a likely confounder — test before building Arm C

Arm G (without freeze_attn) gave lt a clean route: 3.74 ppl at Route 20. Arm G+R froze
attention for the final phase. Two competing explanations for lt's missing structure:

1. **Current-phase problem (the report's reading):** the current phase fundamentally
   cannot be isolated; the recipe needs explicit prefix allocation during training.
2. **Freeze-attention artifact:** attention was trained before lt and has no capacity
   allocated for it; with attention frozen, lt cannot form a consistent neuron-level
   projection. The missing routable structure is then a property of the experiment
   design (freezing the last phase's only trainable pathway), not of the architecture.

Arm G's lt result (3.74 ppl routed) is direct evidence that explanation 2 is
plausible: without the freeze, lt WAS routable.

**Recommended cheap diagnostic (before option 2 as a full arm):**
Train the final phase with unfrozen attention (or a dedicated lt specialist with its own
attention), then re-run the 20-way routing diagnosis. If lt becomes routable, the
freeze is the cause and routing-aware training is the correct next step. If not, the
current-phase problem is real and option 2 is justified.

**Against option 1 (explicit prefix reservation):** reserving neurons in advance is a
learned-gate-style mechanism; learned in-pass gates were already falsified (Addendum 12,
bcbf022: feature poverty, compiled detector = oracle). Routing-aware training (option 2)
is theoretically consistent with the manuscript's construction thesis.

## 4. What is good

- Comparing Arm G vs Arm G+R is the right frame and gives a clean message: routing
  preserves old languages perfectly once their structure exists.
- The report names the exact gap ("the prefix is an emergent property of training, not
  an explicit allocation"). That sentence is the thesis of the next experiment.
- Compute efficiency: 30 min inference, no new training. Good discipline.

## Bottom line

The current-phase problem is a real, isolatable finding — but before building a new
arm, run the freeze-attention diagnostic on lt. It costs one training run and resolves
the two competing explanations. Fix the arithmetic in the report regardless.
