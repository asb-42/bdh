# Phase-Count Ladder Plan — Does computation isolation survive accumulation?

**Version:** 1.0 · **Date:** 2026-08-26 · **Supersedes:** nothing (new program phase)
**Context:** post-manuscript-v1 (`docs/papers/cl-bdh-manuscript.{tex,pdf}`); implements the
agreed sequence *repair theory → operator validation → phase ladder → parameter scaling*.
Steps 1–2 of that sequence are complete (revision-1 manuscript; state-level miscalibration
curves in Appendix A, `app:meas`). This plan is step 3. Parameter scaling (150M/250M) is
step 4 and explicitly deferred until the ladder reports.

---

## 1. Research question

> Does the computation-isolation phenomenon survive continual accumulation?

The manuscript's positive results rest on 3–5 phases. If prefix selection stops being exact,
or acquisition degrades, or erosion accelerates super-linearly by phase 15–20, the central
claim must be scoped to few-phase regimes — better to know before spending compute on scale.

## 2. Design

**Canonical micro-phase**: 30 MB / 10k steps / batch 4 / fresh optimizer / `--init-from`
previous endpoint — byte-identical protocol to the H1 baseline phases (user-approved;
no anchor run needed since this *is* the established protocol).

**Inventory** (verified against statmt listing 2026-08-26): 21 distinct Europarl v7 X-en
language sides. Ladder uses 20; LV held in reserve.

**Fixed phase order** (families interleaved to avoid Germanic/Romance clustering confounds):

```
en*, es, pl, fr, de, it, da, cs, pt, nl, ro, sv, el, hu, bg, fi, sk, sl, et, lt
```

`en*` reuses `out/bdh_europarl_cl-a-en_last.pt` (identical protocol/seed as H1 phase 1);
19 fresh phases follow.

### Arms

| arm | mode | width | purpose | est. wall |
|---|---|---|---|---|
| **R (rewrite)** | `--init-from` chaining, constant ×128 | 100M | pure interference-accumulation curve; H1-family canonical | ~24 h |
| **G (growth+routed)** | `--grow-mult 32` per phase, batch 4→2 beyond ×192 | 160→×736 (~575M) | does routed/isolation structure survive accumulation? | ~40 h |

Arm R launches first; Arm G launches after reviewing R's early trajectory (its gate/routing
measurements depend on G checkpoints anyway).

### Measurements

- **Every phase**: new-language acquisition ppl (lang_eval, current language); running
  log at INFO level.
- **Milestones (phases 5, 10, 15, 20)**: full seen-language interference matrix; xy-sparsity
  (analyze --sparsity).
- **Final (phase 20)**: complete 20×20 acquisition/retention matrix; xy-sparsity; calibrated
  graph snapshot; **growth arm only**: routed-to-k ppl for all k (batch-1 inference),
  20-way likelihood-detector accuracy, leakage budget re-measurement at ×736.

## 3. Pre-registered predictions (falsifiers)

| id | prediction | falsifier |
|---|---|---|
| P-Acq | New-language peak ppl stays ≤ 2.6 through phase 20 | monotone drift > 0.5 nats from phase 5 on (accumulation poisons learning) |
| P-Eros | Rewrite-arm oldest-language ppl decays toward the decorrelation plateau (~15–25 ppl band) by phase ~12, then saturates | stabilizes ≤ 8 ppl (competition capped) or falls ≤ 4 (healing dominates) |
| P-Route | Growth arm: routed-to-k ppl within 0.3 nats of phase-k endpoint quality for **all** k at the final ×736 stack | routed retention degrades monotonically with k-distance (invariant structure does not survive accumulation) |
| P-Det | ≥ 95% 20-way detection accuracy at the final stack (Romance/Slavic clusters are the risk) | < 90% |
| P-Spars | xy-sparsity within 93 ± 2 pp across all phases despite 20-language history | drift beyond ±2 pp |

## 4. Compute policy (DGX-hop triggers)

The 4090 carries Arm R entirely (constant width). Triggers to move to a larger machine:

1. **OOM** at batch 2 on any Arm-G phase (margin thins beyond ~×800 multiplier; expected
   around phase 22+, i.e., beyond this ladder's end — but watch it),
2. **wall-clock**: any single Arm-G phase exceeds 60 min, or projected Arm-G total exceeds
   48 h,
3. **step 4 (parameter scaling 150M/250M)**: batch-4 training at t512 will exceed 24 GB
   (250M ≈ 30 GB at today's activation footprint) — DGX Spark becomes the natural home for
   the entire scaling study, not merely a convenience.

If triggered: freeze state, push, and continue the same protocol on the larger machine —
the plan document travels with the repo.

## 5. Execution checklist

- [x] Language inventory verified (21 sides)
- [x] Loader supports all sides (`sources` map extended)
- [x] Arm R complete: 19/19 phases, P-Acq PASS (peak 2.21), P-Eros FAIL (+12.7/+20.7 nats)
- [x] Milestone reviews: full interference matrix at phases 5/10/15/19 (post-hoc from checkpoints)
- [x] Arm G complete: 19/19 phases, P-Acq FAIL (peak 3.49), P-Eros FAIL (en 41.2, bg 1436)
- [x] Routing measurements on Arm G stack: 5 routes, 88.8% accuracy, 7/19 domains >5×
- [x] Manuscript revision 2: accumulation section, math fixes (commit `013b559`)
- [ ] Routing diagnosis: 20-way detection (P-Det), per-domain retention (P-Route) → `2026-08-27_growth-routing-and-next-steps.md`
- [ ] Growth + Routing combination experiment (~100M) → `2026-08-27_growth-routing-and-next-steps.md`
- [ ] Step-4: CANCELLED — Arm G is already a scaling experiment (100M→700M)

## 6. Provenance

Plan authored by ox-alpha after the 2026-08-25/26 sessions; sequence proposed by the
maintainer (theory-first, phase-count-before-scale), refined in dialogue (micro-phase sizing
kept at the H1-canonical 30 MB/10k; distinct-language inventory verified against the statmt
listing; LV reserved). Ledger: `.jspace/` goal `ladder`.
