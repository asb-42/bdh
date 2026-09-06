# Decay-Aftermath Battle Plan (v1.1)

- Date: 2026-09-05 (v1.1 amendment 2026-09-06)
- Author: A0-Quinn (Agent Zero seat), drafted per operator instruction (#108)
- Review: pi-50 #116 — APPROVE with five amendments; all five folded into this v1.1
- Status: approved; §6 items 2–3 carry drafted rulings awaiting operator sign-off
- Thread: HAK bdh-cl #93–#117; commits 11813b1, 25d1340, 2ce8673, 5422561 (v1.0), 018f126 + 9e52878 (atlas)

## 0. Why plan now, not after RA2b

The decay question is closed: derivation (closed form `p_exit = p_entry · Π(1-lr_t·wd)` for zero-gradient coordinates), f32-scalar realization term (0.892752 → f32 0.892635, measured 0.892636(2)), and triple-instrument cross-validation (Quinn torch c-fits 0.89263; pi-50 weight-norm ratios 0.892636(2); OC-GLM-200 atlas elementwise 0.8926 — four-decimal agreement, three independent code paths). What the RA2b readout decides is a bounded list (H-decay-1/2/3). Everything else we owe is readout-independent. Serializing it behind a 5-day wait would waste the wait.

## 1. Standing rules (ADOPTED per #116)

R1 — **No early RA2b numbers.** No single number from RA2b enters any doc until phase-20 exit + full routdiag + milestones exist. Every number carries F-V6 provenance columns (window/block/batch/split/kernel/init-file).
R2 — **PENDING-RA2b is a greppable token.** Open slots are marked `PENDING-RA2b` so we can prove at merge time nothing leaked in early.
R3 — **Fork band on base-era RA2 figures.** No base-era RA2 decay figure quoted without the ±2.3% `_best`-fork band; per-transition factors 0.5798–0.5929, not a single constant. (Band width set by C5, now in Phase A.)
R4 — **PASS/FAIL on every pre-registered prediction at readout**, signed by whoever registered it.
R5 — **Publish-before-write for GPU claims.** Prefix + config announced on the bus before anything writes to out/.

## 2. Phase A — readout-independent

| # | Task | Owner | Machine | Status |
|---|---|---|---|---|
| A1 | S/M chain: S0a/S0b fixed-seed bg duplicates — DONE: bit-identical on all six tensors (S1 PASS; joint 2.79, routed 2.90 identical) | A0-Quinn | .200 GPU | done 22:29 |
| A1b | S0c+S0d: seed floor + regime contrast (2×2 completion) — DONE: S1d seed floor acq 2.3%/joint 3.8%/routed 4.4%; S2d regime@acq 2.2% = within seed noise, not resolvable at n=1; S4d c=1.000000 (third .200 fix confirmation); S2 final: all controllables excluded (kernel 0, seed 2.3%, batch 1.1%, regime ≤2.5%, compile-mode both eras compiled per warmup signature in Aug-26 log) → residual ~12% = era drift (torch 2.13.0+cu130 today, Aug version unlogged, F-V6 lesson) | A0-Quinn | .200 GPU | done 08:29 |
| A2 | Report v1.1 amendments: closed-form statement (schedule 0.892752 / f32 realization 0.892635 / measured 0.892636(2)), fork band, replicate spread, repaired-joint unreplicated, G2-retention-plainly | A0-Quinn | container | queued |
| A3 | Marin adoption memo: health checklist → BDH, per-segment norm-forecast drift monitor, presentation patterns | A0-Quinn | container | queued |
| A4 | docs/reviews publication of pi-50's Marin/instrument note under operator policy #98 | pi-50 | gx10 | his queue |
| A5 | Moments-census write-up (exp_avg_sq==0 as free learning telemetry) | pi-50 | read-only | his queue |
| A6 | F-V6 extension: literal init filename + script re-runnability check (bash -n + argparse smoke) | pi-50 | container/gx10 | his queue |
| A7 | Manuscript skeleton with PENDING-RA2b slots — **must follow A11** (external positioning first) | A0-Quinn | container | queued |
| A8 | F1 fix: ladder_ra2.sh literal-`\n` corruption — DONE (bash -n OK, zero literal backslash-n) | A0-Quinn | container | done @5422561 |
| A9 | p5_inchain_check.py upstreamed — DONE | A0-Quinn | container | done @5422561 |
| A10 | Eval-side replicate — DONE: A10a PASS (det1 == det2 exactly, 2.64/2.64 — evaluator deterministic, crop RNG Generator(1234) holds); A10b batch-4 vs batch-1 = 2.67 vs 2.64 (+1.1%, small — does not explain S2) | A0-Quinn | .200 GPU | done 07:47 |
| A11 | **NEW (amendment 4): external positioning pass** — M4 prior-art owed; arXiv 2604.09780 (The Myth of Expert Specialization) attacks the routdiag reading directly (cosine≈1 + scale-invariant stable_rank is exactly the pattern it predicts); skeleton must be drafted with it, not restructured later | pi-50 (claimed) | read-only | his queue |
| A12 | **NEW (amendment 5): int-cast screen** — grep repo analysis scripts for `int(x.abs().max())==0`-class zero-checks — DONE: clean; no dangerous idiom in committed scripts; nearest pattern is verify_masked_forward.py:136 `max_ulp` reporting (intended integer ULP count, not a zero-check) | A0-Quinn | container | done 06:17 |
| A13 | **C4 moved from Phase C (amendment 3): F-V8 ruling** — draft in §6, operator sign-off | operator | — | draft ready |
| A14 | es-argmin discrepancy — RESOLVED (#121): en best@9200 triple-confirmed (log argmin 201 rows, en_best.pt step metadata = 9200, closed-form replication 10/10 incl. es: pred 0.584705 = measured); pi-50's 9594 inversion is an artifact of that transition's noise (per-head spread 3.6e-4, ~100x the others — self-flagged); es_best==es_last==10000 explains sv's exact factor. Fork points = log argmins, cross-confirmed by checkpoint metadata | A0-Quinn | .200 read-only | done 08:15 |

Phase A exit criterion: A1–A14 landed or explicitly waived on the bus.

## 3. Phase B — gated on RA2b completion (~Sep 7/8)

| # | Task | Owner | Notes |
|---|---|---|---|
| B1 | Full RA2b readout: acquisition curve, routdiag, milestones, boundary grids | A0-Quinn + pi-50 | H-decay-1/2/3 verdicts, PASS/FAIL lines (R4) |
| B2 | §4 rewrite around fixed-regime readout | A0-Quinn | three-way decomposition with measured numbers |
| B3 | Missing-cell discussion: plain growth + aggressive decay never run at ladder scale; S/M (A1/A1b/A10) phase-level results inform | all | |
| B4 | Replicate sizing from S runs: error bars on single-cell headline numbers | A0-Quinn | S1 done (floor = 0 at fixed seed); S0c prices seed floor |
| B5 | Partial-readout policy — DRAFTED in §6, operator sign-off | operator | before day 5 |

## 4. Phase C — after B

| # | Task | Notes |
|---|---|---|
| C1 | QAT proposal gate revisit (architecturally sound post-fix; expansivity numbers still warn) | docs/plans/2026-08-28_qat-proposal.md |
| C2 | sv-backfill fixed-regime variant (only if 20-phase G2 geometry needed for the paper) | ~6-8 h GPU |
| C3 | Weight-atlas per-phase norm-forecast as standing instrument (design in A3 memo) | catches F-decay-class leaks at phase 1 |

(C4 and C5 moved to Phase A per review #116, amendment 3.)

## 5. Machine allocation (current)

- **gx10**: RA2b only (phase 10/20, hu in progress; ~halfway; ETA Sep 7/8). Read-only Git fine.
- **.200 GPU**: S0c running (A1b); A10 eval-replicate queued after it. Lease system in use per #111 (gpu://rtx4090, claim/renew/release practiced).
- **.200 Git**: first-class pull host since #117 — deploy key registered, origin `git@github.com-quinn4090:asb-42/bdh.git`, fetch + ff-pull to 5422561 verified. scp era over.
- **Container (Quinn)**: A2, A3, A7 (after A11), plan/bus stewardship.
- **pi-50 (gx10 seat)**: A4, A5, A6, A11; read-only on .200; will not self-serve GPU.

## 6. Open decisions (operator)

1. ~~Plan approval~~ — **RESOLVED**: approved with amendments (#116), folded into v1.1.
2. ~~B5 partial-readout ruling~~ — **RESOLVED** (operator sign-off #119 → bus #120): per-phase acquisition quotable at exit; no retention/forgetting headline before phase-20 + full routdiag + milestones. Forward-looking for RA2b; the RA2 final report is not a retroactive violation (its retention numbers were reported at chain completion, which the ruling permits).
3. ~~C4/F-V8 ruling~~ — **RESOLVED** (operator sign-off #119 → bus #120): fresh-optimizer-per-phase is intentional; keep saving optimizer_state as substrate for moments-census telemetry; future true-resume would be a separate explicit flag.
4. ~~Deploy key for .200~~ — **RESOLVED** (#117): registered, origin canonical, fetch + ff verified. NOTE: operator (asb) needs his own GitHub credentials on .200 for pulls — a0-quinn has the deploy key in per-user config; asb needs (a) second deploy key, (b) HTTPS+token, or (c) sudo -u a0-quinn. Operator's call.

### Draft ruling B5 — partial RA2b readout (sign-off pending)

Per-phase acquisition numbers and direction-of-progress statements may be quoted once that phase's exit exists (they describe that phase only). NO retention/forgetting headline numbers — joint milestone cells, routed serving tables, forgetting claims — in any doc or evidence-bearing bus post before the phase-20 exit + full routdiag + milestone set exist. Rationale: retention numbers are chain-state-dependent and change with every subsequent phase; an early quoted number is guaranteed stale by the time it is cited. This adopts pi-50's #107 vote (P1/P3 directionally yes; headline retention numbers no).

### Draft ruling C4 — F-V8, optimizer state (sign-off pending)

`optimizer_state` is saved in checkpoints (train.py:71) but never restored under `--init-from`. Ruling: **intentional** for the ladder protocol — the documented protocol is "fresh optimizer via --init-from"; AdamW moments belong to a phase's data distribution, and carrying them across language phases would violate the per-phase design. Keep saving it: it is the substrate for the moments-census telemetry (exp_avg_sq==0 reads, #92/#96) and a future true-resume would be a separate explicit flag, not a change to `--init-from`. Consequence: checkpoints carry ~2x model bytes of optimizer state by design; storage planning accounts for it. Alternative (NOT recommended while the census is an active instrument): strip optimizer state from `_last` to halve disk at the cost of the telemetry.

## 7. Lessons filed this thread

- Bootstrap overwrites /root/.ssh/config on container rebuild (03:12, Sep 5) — aliases/mappings lost, keys survived; restored manually; bootstrap should append, not overwrite.
- f2f4707 introduced literal-`\n` corruption into ladder_ra2.sh (found by pi-50 #110; fixed @5422561). Base-era RA2 artifacts predate the corruption; exact command-line provenance for those runs is unrecoverable from the file.
- The S/M launcher was reviewed and corrected pre-launch (argv-in-manifest bug) — then relaunching caught a wrong-init (pt vs hu) at step 55 via the growth-line check (#114). R5 working as intended.
- **`--seed` lesson (new, #116 amendment 1):** config.py auto-generates argparse flags over Config dataclass fields — source-grepping an auto-generated CLI for a literal flag always returns zero hits. My #113 "no --seed flag (verified)" claim was wrong; pi-50 fell into the identical trap an hour earlier and caught it. Verification of CLI surface must go through the live parser, not source grep.
- **Bus schema note:** type=review_verdict is rejected outside a task_request context (D13: meta.kind must be from the closed set); reviews posted as chat with verdict in first line (pi-50 #116 practice).
- **Load-bearing caveat (pi-50 #116):** .200's pinned cd89ed7 is verified leaky (11813b1 is NOT an ancestor) — all S/M numbers describe cd89ed7 physics and must not be back-fitted onto the RA2b fixed-regime story.

## Appendix A1 — S/M chain results (2026-09-06, cd89ed7 physics; see caveat above)

Pre-registrations in #113; S0c in #117. All numbers: bg phase, init ladG-hu_last (mult 416→448), 10k steps, batch 1.

| run | best-val ppl | joint ppl | routed ppl (28672) |
|---|---|---|---|
| S0a (seed 1337) | 2.74 | 2.79 | 2.90 |
| S0b (seed 1337) | 2.74 | 2.79 | 2.90 |
| S0c (seed 1338) | PENDING | PENDING | PENDING |
| M (aggressive cosine) | 2.55 | 2.61 | 2.72 |

- S1 (S0a vs S0b bitwise): **PASS** — bit-identical, all six tensors; pipeline fully deterministic at fixed seed on this box. G-vs-GR divergence therefore NOT kernel nondeterminism-today; cause is elsewhere (era/hardware/init-chain — A14 informs).
- S2 (acquisition vs ladG-bg 2.41): **FAIL** — 2.74 is 14% above; not noise (S1); regime marker (candidates: batch-4 reference vs batch-1 S-runs; hardware era). Flagged, not averaged away.
- S3 (routed spread): 0.00 by construction of S1.
- M1 (aggressive acquisition): 2.55 < 2.74 — better, as predicted (band 2.2–3.0).
- M2 (M-joint vs S-joint): 2.61 < 2.79 — **my prediction FALSIFIED**: one aggressive phase does not hurt joint serving at phase level.
- M3 (M-routed vs S-routed): 2.72 < 2.90 — **FALSIFIED**: one aggressive phase does not break routed serving either. RA2-style collapse is a multi-phase cumulative phenomenon (c^k over many later phases), not a single-phase effect. Sharpens the §4 three-way decomposition.
- S0c verdicts (seed 1338): **REGIME-CONFOUNDED** — my deploy-key pull moved .200 from cd89ed7 (leaky) to 5422561 (fixed) between S0b and S0c, so S0c ran under the FIXED regime (growth line: 'bit-exact via step-end restore'). c-factor vs hu_last: 1.000000 (P5 cross-platform PASS on .200/RTX-4090 — first independent confirmation of the fix on this hardware). S1c (s0a vs s0c) is therefore NOT a pure seed floor — it is seed+regime combined (acq 2.74→2.62 = −4.4%; joint 2.79→2.64 = −5.4%; routed 2.90→2.75 = −5.2%). A proper seed floor needs S0d (one variable at a time), now a Phase-A nicety, not a blocker (RA2b carries the fixed regime prospectively). Filed as lesson.
