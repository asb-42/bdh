# Review: freeze_attn Diagnostic Experiment

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-27_freeze-attn-diagnostic.md` (commit `c7b638f`)
**Context:** Follow-up to the Arm G+R routing diagnosis review (2026-08-28). Tests the
hypothesis that frozen attention caused lt's missing routable structure.

## Verdict

**Core conclusion accepted: freeze_attn alone is not the confounder; the current-phase
problem is real and path-dependent. One arithmetic correction required before the
numbers enter the manuscript. One design caveat on what the experiment actually tested.
Recommend a 2- to 3-phase route-aware-training proof-of-concept before a full ladder.**

## 1. The core finding is accepted

lt remains unroutable under unfrozen attention (5% detection, 55.30 ppl = joint 56.33).
The comparison is instructive:

| Condition | lt routing acc | lt routed PPL |
|---|---|---|
| Arm G (trained from scratch, no freeze history) | 32% | 3.74 |
| Arm G+R (frozen attn), diagnostic: last phase unfrozen | 5% | 55.30 |

The report's diagnosis is plausible and matches the evidence: the attention was locked
by 19 phases of freeze history into a structure optimized for older languages;
unfreezing the last phase cannot reorganize it. The growth recipe creates a path
dependency that the freeze flag alone does not explain.

## 2. CRITICAL — overall P-Det arithmetic is inconsistent with the table

The report claims overall 740/760 = 97.4%. Counting the table independently: 15
languages at 100% × 40 = 600, plus cs 37, es 30, et 34, lt 2 = **703/760 = 92.5%**
under the report's own rows. The reported 740 is off by 37 (suspiciously close to
cs's count). Either the table or the total is wrong; both cannot be true.

This is the third arithmetic inconsistency in this report series (after 758/800 in the
routing diagnosis and 744/720 in the G+R report). It does not change the qualitative
conclusion (lt is the only major failure), but it must be corrected before any of
these numbers are cited in the manuscript. The pattern warrants a protocol addition:
recompute totals directly from per-domain counts in every report before commit.

## 3. Design caveat — what the experiment actually tested

Unfreezing attention on an init from `ladGR-sl_last.pt` (19 phases of frozen attention
history) tests "thaw the last phase after a long frozen history", not "attention
always trainable during growth". The report correctly notes the difference (init state
vs freeze flag), but does not draw the methodological consequence:

- To fully separate "path dependency of freeze history" from "current-phase problem in
the growth recipe", run a minimal control: a 2- or 3-phase ladder with **no
freeze_attn at any phase**, then check whether phase 2 (or 3) is routable. This costs
hours, not a full ladder, and would make the causal claim much stronger.

## 4. Recommendation: route-aware training PoC (2-3 phases, pre-registered)

Given the finding, the next experiment should be a small, pre-registered
proof-of-concept of routing-aware training, not a full 19-phase ladder:

- 2-3 Europarl phases (~30 MB each), width growth +32/phase, attention unfrozen or
  selectively trainable per phase.
- During training of phase n, evaluate the loss on the freshly grown prefix only
  (route-aware objective), not on the whole stack.
- Falsifier: phase n must reach routing detection ≥95% and routed retention within
  0.3 nats of its specialist at the end of the run.
- If the PoC passes, scale to the full ladder. If it fails, the current-phase problem
  is architectural and "accept the limitation" becomes the honest default.

This avoids spending ~40 h on a full ladder before we know whether the mechanism
works at all.

## 5. Non-blocking notes

- "lt accounts for 38/20 misroutes" — almost certainly a typo (should be 38 of 40).
  Fix in the report.
- P-Route is still reported against the degraded joint baseline. For the manuscript,
  keep the specialist-baseline framing required since the routing-diagnosis review:
  lt's specialist ppl is 3.36; routed serving at 55.30 means the current-phase
  specialist is not reachable through the router — that framing makes the result
  sharper than "no benefit vs joint".

## Bottom line

The experiment answers the question it was designed for: freeze_attn is not the
confounder. The current-phase problem is real and path-dependent. Fix the arithmetic,
run the no-freeze control and the small route-aware PoC before committing to a full
ladder, and adopt the specialist-baseline framing for P-Route.
