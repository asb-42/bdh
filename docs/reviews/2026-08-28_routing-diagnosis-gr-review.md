# Review: Routing Diagnosis Arm G+R (20-way)

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-27_routing-diagnosis-gr.md` (commit `b3e10b5`)
**Context:** First intermediate result from Phase 2 (growth + routing combination).

## Verdict

**Core finding is sound — but fix the arithmetic before the report is cited. Most
important next step: rule out the freeze_attn confounder with a targeted diagnostic
run before building a new arm.**

## 1. The finding is real and important

The last-trained language (lt) has no routable structure: 32% detection, scattered
across 7 routes, routed PPL 55.10 = joint (54.96). All older languages route well
(95–100%).

This is the **current-phase problem**: the growth rule preserves old prefixes but
constructs none for the language currently being trained. The report's phrasing
("the prefix is an emergent property of training, not an explicit allocation") is
exactly the right diagnosis.

## 2. Arithmetic: same error class as in the previous routing report

- "Overall: 2/760 = 0.3% on diagonal" — incorrect (2/760 = 0.26%, and 2 is not the
  correct count anyway).
- "Excluding lt: 744/720 = 100%" — mathematically impossible (numerator >
  denominator); also es/et are not 100% per the report's own table.
- Independent count from the table: 16 languages × 40 = 640 + es 38 + et ~33 +
  lt ~13 = **~724/760 = 95.3%**, not 99.7% and not "everything 100%".

**Action:** fix in the report (724/760 = 95.3%, lt as the only major outlier). The
manuscript must not carry faulty percentage arithmetic — the review note from
2026-08-27 flagged the same error class.

## 3. More important than the numbers: freeze_attn is a plausible confounder

The difference between Arm G and Arm G+R is the freeze_attn mechanism (commit
`8fef089`: "freeze attn weights during growth"). Compare:

| Arm | lt detection | lt routed PPL | Attention during lt training |
|---|---|---|---|
| Arm G (no R) | clean route (route 20) | 3.74 | trainable |
| Arm G+R | 32%, scattered | 55.10 (= joint) | **frozen** |

Plausible explanation: lt was trained with frozen attention — the attention was
already optimized for older languages and had no free capacity to form a
projection for the new language. Without trainable attention, lt cannot build a
consistent neuron-level projection pattern; it spreads across existing structure.
The current-phase problem would then be an artifact of the freeze rule, not a
fundamental architecture limit.

**Recommended diagnostic experiment (costs one run, no new arm architecture):**
Retrain the last phase (or a repeat of lt) with **unfrozen attention**, then
re-run the routing diagnosis.
- If lt becomes routable (~3.7 ppl, clean route) → freeze_attn is the cause. Next
  step: keep attention trainable for the current phase and build routing awareness
  into training (option 2 from the report, now with a precise mechanism).
- If lt stays unroutable → the current-phase problem is fundamental; then option 2
  is worth the effort.

## 4. Against option 1 (explicit prefix reservation) — short note

Reserving neurons in advance resembles a learned-gate mechanism; learned in-pass
 gates were already falsified (Addendum 12, bcbf022 — feature poverty, compiled
detector = oracle). If the diagnostic confirms freeze_attn, route-aware training
(option 2) is the theoretically more consistent answer.

## 5. What is good

- Comparing Arm G vs Arm G+R is the right frame and gives a clean message: routing
  preserves older languages perfectly once their structure exists.
- The interpretation names the gap precisely (emergent vs. allocated).
- Cheap method: pure inference, no new training runs in this diagnostic.

## Bottom line

Core finding (lt unroutable, older languages routable) stands. Fix the arithmetic.
Before the next arm: rule out the freeze_attn confounder with one targeted run.
That is cheaper and more precise than jumping straight to a large new experiment.
