# Decay-Aftermath Battle Plan (v1.0)

- Date: 2026-09-05
- Author: A0-Quinn (Agent Zero seat), drafted per operator instruction (#108)
- Status: proposal for room review; Phase A is readout-independent and can start on approval
- Thread: HAK bdh-cl #93–#113; commits 11813b1, 25d1340, 2ce8673, 018f126 (atlas), 9e52878 (atlas)

## 0. Why plan now, not after RA2b

The decay question is closed: derivation (closed form `p_exit = p_entry · Π(1-lr_t·wd)` for zero-gradient coordinates), f32-scalar realization term (0.892752 → 0.892635), and triple-instrument cross-validation (Quinn torch c-fits 0.89263, pi-50 weight-norm ratios 0.892636(2), OC-GLM-200 atlas elementwise 0.8926 — four-decimal agreement, three independent code paths). What the RA2b readout decides is a bounded list (H-decay-1/2/3). Everything we owe is readout-independent. Serializing all of it behind a 5-day wait would waste the wait.

## 1. Standing rules (proposed, effective on approval)

R1 — **No early RA2b numbers.** No single number from RA2b enters any doc until phase-20 exit + full routdiag + milestones exist. Every number carries F-V6 provenance columns (window/block/batch/split/kernel/init-file).
R2 — **PENDING-RA2b is a greppable token.** Open slots in docs are marked `PENDING-RA2b` so we can prove at merge time nothing leaked in early.
R3 — **Fork band on base-era RA2 figures.** No base-era RA2 decay figure quoted anywhere without the ±2.3% `_best`-fork band; per-transition factors 0.5798–0.5929, not a single constant.
R4 — **PASS/FAIL on every pre-registered prediction at readout**, signed by whoever registered it. P5 already has one (PASS, Quinn run, pi-50 confirmed independently on es→pl with 8,388,608 nonzero exp_avg_sq cells = exactly the new block).
R5 — **Publish-before-write for GPU claims.** Prefix + config announced on the bus before anything writes to out/ (operator #109 practice; S/M chain announced in #113).

## 2. Phase A — start now, readout-independent

| # | Task | Owner | Machine | Est. |
|---|---|---|---|---|
| A1 | S/M chain on 4090: S0a/S0b fixed-seed bg duplicates (kernel-determinism floor) + M aggressive-cosine plain-growth cell | A0-Quinn (running) | .200 GPU | ~3 h, running |
| A2 | Report v1.1 amendments: closed-form statement (schedule product 0.892752 / f32 realization 0.892635 / measured 0.892636(2)), fork band, replicate spread next to mild-regime retention figures, repaired-joint marked unreplicated, G2-retention-without-route-awareness stated plainly | A0-Quinn | container | ~2 h |
| A3 | Marin adoption memo: 12-point health checklist mapped to BDH, per-segment norm-forecast drift monitor design (Percy Liang's demand), presentation patterns | A0-Quinn | container | ~2 h |
| A4 | docs/reviews publication of pi-50's Marin/instrument note under operator policy #98 (attribution, verbatim caution quotes, measured/external separation) | pi-50 | gx10 | ~1 h |
| A5 | Moments-census write-up: exp_avg_sq==0 as free per-parameter learning telemetry in 340 checkpoints | pi-50 | read-only | ~1 h |
| A6 | F-V6 extension: literal init filename (best-vs-last) + script re-runnability check (bash -n + argparse smoke) as required fields | pi-50 (extension), Quinn (F1 fix landed) | container/gx10 | ~30 min |
| A7 | Manuscript skeleton with PENDING-RA2b slots | A0-Quinn | container | ~3 h |
| A8 | F1 fix: ladder_ra2.sh literal-\n corruption (Quinn's f2f4707) — fixed, verified bash -n + zero literal \n; commit pending | A0-Quinn | container | done |
| A9 | p5_inchain_check.py upstreamed to scripts/ (Quinn authorship per pi-50 #107 offer) | A0-Quinn | container | ~30 min |

Phase A exit criterion: A1–A9 all landed or explicitly waived on the bus; repo clean; no PENDING-RA2b slot accidentally filled.

## 3. Phase B — gated on RA2b completion (~Sep 7/8)

| # | Task | Owner | Notes |
|---|---|---|---|
| B1 | Full RA2b readout: acquisition curve, 20-way routdiag per phase, milestones, boundary grids | A0-Quinn + pi-50 | H-decay-1/2/3 verdicts with PASS/FAIL lines (R4) |
| B2 | §4 rewrite around fixed-regime readout | A0-Quinn | three-way decomposition (decay/interference/co-adaptation) with measured numbers |
| B3 | Missing-cell discussion: plain growth + aggressive decay (M1–M3 from Phase A inform this); never-run at ladder scale, state as limitation | all | |
| B4 | Replicate sizing from S runs: error bars on all single-cell headline numbers | A0-Quinn | S1/S2/S3 pre-registered in #113 |
| B5 | Partial-readout policy decision (pi-50 #107): P1/P3 directionally quotable, headline retention numbers no — adopt as ruling? | operator | decide before day 5 |

## 4. Phase C — after B

| # | Task | Notes |
|---|---|---|
| C1 | QAT proposal gate revisit (architecturally sound post-fix; expansivity numbers still warn) | docs/plans/2026-08-28_qat-proposal.md |
| C2 | sv-backfill fixed-regime variant (only if 20-phase G2 geometry needed for the paper) | ~6-8 h GPU |
| C3 | Weight-atlas per-phase norm-forecast as standing instrument (drift monitor; catches F-decay-class leaks at phase 1) | design in A3 memo |
| C4 | F-V8 ruling (optimizer_state saved at train.py:71, never restored) | operator decision |
| C5 | es argmin discrepancy: pi-50 inversion implies parent-best@9594, Quinn log extraction says en best@9200; 100x per-head spread on this one transition. Resolve via .200-era ladRA2_en.log (base-era phases ran on .200, logs transferred to gx10 — Quinn to re-extract with the exact log actually used) | A0-Quinn | ~30 min |

## 5. Machine allocation (current)

- **gx10**: RA2b only (phase 5/20 area at time of writing; ~96% GPU; days to completion). Read-only Git OK (pi-50 ff-merged 2ce8673 there).
- **.200 GPU**: S/M chain (A1, running, announced #113).
- **.200 CPU**: weight-atlas scans done (20/20 G2-chain coverage, 018f126 + 9e52878); free for more scans on request.
- **Container (Quinn)**: A2, A3, A7, A8, A9 + bus coordination.
- **pi-50 (gx10 seat)**: A4, A5, A6; read-only moments census done; will not self-serve GPU.

## 6. Open decisions (operator)

1. Approve this plan (or amend).
2. B5 partial-readout policy ruling (before day 5).
3. C4 F-V8 ruling.
4. bdh repo on .200 still at cd89ed7 (no deploy key for a0-quinn there; syncs via container clone — acceptable interim).

## 7. Lessons filed this thread (for the record)

- Bootstrap overwrites /root/.ssh/config on container rebuild (03:12 today) — host aliases and GitHub mappings lost; keys survived. Restored manually. Action: note in ops runbook; bootstrap should append, not overwrite.
- f2f4707 introduced literal-\n corruption into ladder_ra2.sh (Quinn's own commit; found by pi-50 #110; same bug class Quinn hit live in the v8 chain). Fixed in A8; the base-era RA2 artifacts predate the corruption and were not produced by the broken text — but exact command-line provenance for those runs is gone.
- The S/M launcher was reviewed once more before launch and corrected (argv-in-manifest bug: manifest must record the literal command, not a description).
