# Growth + Routing: Next Steps

**Date:** 2026-08-27 · **Status:** active
**Context:** Step-4 (parameter scaling 150M/250M) cancelled — Arm G is already a scaling
experiment (100M → 700M over 19 phases). The open question is whether the grown stack has
routable structure, and whether growth + routing prevents forgetting.

---

## 1. What we know

| Experiment | What it tested | Result |
|---|---|---|
| Arm R (fixed 100M) | Pure interference accumulation | P-Acq PASS, P-Eros FAIL (+12.7/+20.7 nats) |
| Arm G (growth to 700M) | Growth WITHOUT routing | P-Acq FAIL (peak 3.49), P-Eros FAIL (en 41.2, bg 1436) |
| Routing on Arm G stack | Does grown model have routable structure? | Partial: 88.8% accuracy with 5 routes, 7/19 domains >5× |

**Key insight:** Arm G was growth WITHOUT routing. Arm R was routing WITHOUT growth.
The combination experiment (growth + routing) is the missing cell in the matrix.

## 2. Open questions

1. **Does the ~700M Arm-G stack have fine-grained routable structure?**
   - Current routing used 5 routes (k=9k/18k/27k/36k/45k) — too coarse for 20 domains
   - Need 20-way detection (P-Det falsifier: ≥95%)
   - Need per-domain routed retention (P-Route falsifier: within 0.3 nats)

2. **Why do some domains have poor routing?**
   - bg=415.91 (55%), sl=20.76 (50%), ro=39.57 (72%)
   - Is this a capacity issue (these languages were trained last) or a structural issue?

3. **Can growth + routing prevent forgetting?**
   - The combination experiment at ~100M
   - This is the central test of "computation-isolable CL"

## 3. Execution plan

### Phase 1: Routing diagnosis on Arm-G stack (no training)

**Goal:** Complete the routing picture on the existing ~700M checkpoint.

**Tasks:**
1. Run eval_router.py with 20 routes (one per domain) on `out/bdh_europarl_ladG-lt_last.pt`
   - `--routes 4500,9000,13500,18000,22500,27000,31500,36000,40500,45000,49500,54000,58500,63000,67500,72000,76500,81000,85500,90000`
   - All 20 Europarl domains as `--domains`
   - Compute 20-way detection accuracy (P-Det)
   - Compute per-domain routed retention (P-Route)

2. Analyze routing quality:
   - Which domains are well-routed? Which are not?
   - Is routing fine-grained (language-specific) or coarse (family-level)?
   - What's the relationship between training order and routing quality?

3. Report findings in `docs/reports/2026-08-27_routing-diagnosis.md`

**Compute:** ~30 min on 4090 (batch-1 inference, 20 routes × 20 domains × 40 crops)

**Decision gate:**
- If P-Det ≥95% AND P-Route within 0.3 nats → proceed to Phase 2 (growth + routing)
- If P-Det <95% OR P-Route fails → analyze why, consider alternative routing strategies
- If routing is coarse (family-level) → the recipe needs refinement

### Phase 2: Growth + Routing ladder experiment (~100M)

**Goal:** Test whether growth + routing prevents forgetting.

**Design:**
- Same 20-phase Europarl ladder as Arm R/G
- Width growth: +32 neurons/head per phase (×128 → ×708)
- **Addition:** trained routing head (or compiled detector) at each phase
- **Addition:** during training, route to the current phase's prefix; during eval, use the router

**Measurement:**
- P-Acq: does acquisition stay ≤2.6 ppl?
- P-Eros: does erosion stay ≤0.3 nats (vs. phase peak)?
- P-Route: does routing preserve earlier languages?

**Compute:** ~40 h on 4090 (batch 4→1 as width grows)

**Decision gate:**
- If P-Acq PASS + P-Eros PASS → the recipe works; manuscript complete
- If P-Acq PASS + P-Eros PARTIAL → routing helps but doesn't eliminate forgetting; need merge/replay
- If P-Acq FAIL → the combination doesn't work at this scale; reconsider

### Phase 3: Manuscript update (if Phase 2 succeeds)

**Goal:** Add growth + routing results to the manuscript.

**Additions:**
- Update section 6 (accumulation) with routing diagnosis and combination results
- Update decision rule table
- Update limitations (routing quality, scale)

## 4. Files to create/modify

- `docs/plans/2026-08-27_growth-routing-and-next-steps.md` (this file)
- `docs/reports/2026-08-27_routing-diagnosis.md` (Phase 1 output)
- `scripts/eval_router.py` (may need updates for 20-way routing)
- `scripts/ladder_armGR.sh` (Phase 2 training script, if needed)
- `docs/papers/cl-bdh-manuscript.tex` (Phase 3 manuscript update)

## 5. Timeline

| Phase | Est. time | Status |
|---|---|---|
| Phase 1: Routing diagnosis | ~1 h | pending |
| Phase 2: Growth + Routing | ~40 h | pending (after Phase 1) |
| Phase 3: Manuscript update | ~2 h | pending (after Phase 2) |

## 6. Step-4 status

**CANCELLED.** Arm G is already a scaling experiment (100M → 700M). The recipe's
mechanisms are architecture-agnostic. No 150M/250M training needed unless Phase 2
shows the recipe fails at 100M due to capacity (not routing).

---

**Plan authored:** 2026-08-27 by ox-alpha after the 2026-08-26/27 sessions.
Ledger: `.jspace/` goal `ladder`.
