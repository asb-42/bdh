# Decay-Leak Family Measurement: Two Regimes, Counterfactual Repair, Boundary Restoration, sv Backfill

- Date: 2026-09-04 (measurements 2026-09-03/04)
- Author: A0-Quinn (Agent Zero seat)
- Data: .200 `/home/a0-quinn/bdh_work/` (chains v2–v9, all cell outputs, repair sidecars); gx10 `out/logs/` (RA2 repaired evals, routdiag p20 raw/repaired)
- Status: closes the leaky-family measurements. RA2b (fixed regime) is the prospective control, currently training on gx10.

## 1. Root cause of the two decay regimes (closed)

The AdamW decoupled-decay leak (F-decay, 2026-09-03) hit every arm, at two intensities fully explained by the lr-schedule flags:

| family | scripts | schedule | per-phase factor c | en segment after full chain |
|---|---|---|---|---|
| RA2 | ladder_ra2*.sh | `--warmup-iters 1000 --lr-decay-iters 10000` (full cosine) | 0.57978 | ×3.2e-5 (annihilated) |
| G/GR/G2 | ladder_armG*.sh, v7 backfill | pipeline defaults (`warmup 30, lr_decay_iters 300, min_lr 1e-4` → lr = 1e-4 for ~97% of steps) | 0.8927 | ×0.116–0.129 (mild) |

Verified by per-segment c-fits (uniform multiplicative, residuals ~1e-5, 0 flags in every repair):

- G (19 phases): en c = 0.129460 = 0.8927^18
- G2 (20 phases): en c = 0.115561 = 0.8927^19; sv c = 0.634888 = 0.8927^4
- GR (20 phases): sl c = 0.892636 = 0.8927^1
- RA2: 0.57978 schedule-exact (see 2026-09-03 finding report)

**Consequence:** cross-arm contrasts in earlier reports (G vs RA2) are confounded by BOTH route-awareness and the schedule (decay rate). Both arms preserve weight direction (uniform multiplicative decay); neither destroys structure. The clean factorial (fixed leak, single schedule, ± route-aware) is delivered prospectively by RA2b.

## 2. Per-segment repair is a counterfactual, not a restoration

`repair_decay.py` restores each segment to its own phase-exit amplitude. That composite never existed: later phases trained against already-decayed older segments. All cells measured (cold eval, protocol-congruent; routed = own route, 40 crops):

Joint serving (ppl):

| arm (chain) | bg raw | bg repaired | el raw | el repaired |
|---|---|---|---|---|
| G (19 ph, mult 704) | 1435.63 | 577.92 | 1267.75 | 283.46 |
| GR (20 ph, 736) | 1235.58 | 1311.64 | 989.04 | 736.47 |
| G2 (20 ph, 736, sv backfill) | 1430.10 | 1591.34 | 801.35 | 807.11 |
| RA2 (20 ph, 736, aggressive) | 1649.24 | 481.46 | 890.55 | 118.10 |

Routed serving (ppl, own route):

| arm | bg raw | bg repaired | el raw | el repaired |
|---|---|---|---|---|
| G | 2.50 | 41.31 | 2.64 | 35.08 |
| GR | 2.99 | 14.24 | 2.94 | 17.63 |
| G2 | 2.52 | 41.31 | 2.64 | 35.08 |
| RA2 | 15.65 | 37.63 | 16.11 | 62.64 |

Findings:

- Routing decisions themselves shift under per-segment repair (RA2: 9 domains at 40/40-own after repair vs 7 before; sk/sl lose their own routing entirely to cs).
- The joint effect is chain-state-dependent (G −60% bg vs G2 +11% bg) — not predictable a priori.
- RA2 repaired routing splits by segment depth: base languages improve dramatically (en 11.84→2.37 ≈ acquisition 2.29; es 19.40→3.99), late resume languages worsen (bg 15.65→37.63, el 16.11→62.64, lt 10.81→36.94).
- Internal consistency check: repaired routed bg/el are IDENTICAL between G and G2 (41.31 / 35.08) although the two repairs divide by different chain-length factors — expected, because serving at route w only reads rows < w and those rows restore to the same phase-exit composite in both arms. Confirms mask isolation plus the counterfactual nature of the composite.

**Conclusion:** per-segment repair is a measurement instrument for the decay share of joint collapse (it bounds, not restores). Do not serve from it.

## 3. Boundary restoration law (validated exactly)

The leak multiplies every frozen-path element by the same factor in every later phase. Therefore ONE scalar restores the entire checkpoint to its end-of-phase-j state: divide all rows below the phase-j width by c^(final−j). That state existed (co-adaptation preserved).

New tool: `scripts/boundary_repair.py`. Pre-registered predictions, then measured (v9, 2026-09-04):

| restoration | prediction | measured | ground truth |
|---|---|---|---|
| G2-lt → phase 16 (sv), route 38912 | ~3.2 (acquisition 3.22) | 3.27 | sv source checkpoint at own route: 3.27 |
| G2-lt → phase 11 (bg), route 28672 | ~2.4 (acquisition 2.38) | 2.50 | bg source checkpoint at own route: 2.50 |
| RA2-lt → phase 13 (fi), route 32768 | ~8.7 (p13 routdiag 8.72) | 8.68 | p13 routdiag 8.72; earlier splice repair 8.68 |

Three for three, exact to instrument resolution. Boundary undecay is the correct "roll back to phase j" operation for any leaky-ladder checkpoint: it reproduces the source checkpoint's serving value without possessing the source checkpoint.

## 4. sv backfill: Arm G completed to 20 phases (ladG2)

Forensics (closes the 2026-08-26 open item):

- The ladG sv phase was never trained: CUDA OOM at 576→608 (`ladG_sv.log`); no sv checkpoint ever existed; the resume deliberately continued sk→ro (`ladder_armG_resume.sh`, INIT = sk_last).
- Arm G was a VALID 19-phase chain (final mult 704, ≈554M params). "sv 53.5" in the old phase-20 milestone was zero-shot transfer on ro-trained neurons, not erosion. The 2026-08-30 correction header wrongly applied GR's 20-phase/736 geometry to G.
- Backfill (v7, leaky regime deliberately, for homogeneity with phases 1–15): sv→ro→nl→sl→lt from sk_last, run prefix `ladG2-*`, nothing overwritten. DONE 06:41. Acquisition: sv 3.22, ro 3.07, nl 3.27, sl 3.57, lt 3.58. All 20 c-fits uniform 0.8927/phase — the completed chain is a single decay regime end-to-end.
- 20-language joint milestone on ladG2-lt: bg 1430.10, el 801.35, lt 3.58, sv 48.22, remaining languages 30–64. Notable vs the old 19-phase chain: el improves 1267.75→801.35 with the sv phase added; bg unchanged (1435.63→1430.10); hu 80.0→62.64.

## 5. What this means for §4 (three-way decomposition, measured)

1. **Decay artifact:** dominates RA2's joint collapse of the non-Latin pair (bg 1649→481 under repair; the residual is interference + co-adaptation) and RA2 fi's routed degradation (boundary restore 16.36→8.68, exact).
2. **Real interference:** dominates G/GR/G2 joint collapse (repair does not fix it: 1591/1311 vs acquisition ~2.4). Joint forgetting is genuine in every arm.
3. **Co-adaptation:** the third term. It makes per-segment repair destructive exactly where training was joint (routed bg 2.50→41.31) and chain-dependent under joint serving.
4. **Positive result:** under MILD decay, growth + likelihood routing retains languages WITHOUT route-aware training (G2 routed bg 2.52, sv 3.27 vs acquisition 2.38, 3.22). Under AGGRESSIVE decay, RA2's route-aware training held routed serving at acquisition (+0.9% bg). The missing cell (plain growth + aggressive decay) was never run; RA2b delivers the fixed-regime control prospectively.

## 6. Erratum items

Landed with this commit:

- `2026-08-26_arm-g-results.md`: second correction banner — G ran 19 phases, final mult 704 (table cells 676/708 → 672/704); sv 53.5 is zero-shot transfer; the completed 20-phase geometry exists as ladG2.
- Retracted (mine): "ladG without sv is not repairable" (a 19-source repair is arithmetically valid — measured) and "later G phases sit on a false foundation" (the chain is valid, just sv-less).

Stands:

- RA2 final report §4 routing table — re-parsed programmatically against the raw confusion matrix; every cell confirmed (7×40/40 own-prefix; fi 36/40 with 4→et; cs/pl→sk; Romance cluster on ro; sv→et 24/40).
- All acquisition numbers, all routing accuracies, G/GR joint erosion milestones (real interference).

Open (for the manuscript §4 rewrite, after RA2b):

- G-vs-RA2 contrast is schedule-confounded; rewrite around the fixed-regime readout.
- G-vs-G2 repaired-joint asymmetry (577.92 vs 1591.34 on bg) — co-adaptation sensitivity, mechanism open.
- G2 sv joint improves under repair (48.22→16.72) while bg worsens — open.
- G vs GR raw routed bg differ (2.50 vs 2.99) although freeze-attn is a no-op on bdh — exact determinism across runs not established.

## 7. Reproducibility

- Workspace: `.200:/home/a0-quinn/bdh_work/` — chain_v2..v9 logs, all per-cell outputs (`*_raw.txt`, `*_rep.txt`, probe files), repaired and boundary checkpoints, repair JSON sidecars.
- Backfill checkpoints: `.200:/media/data/coding/bdh/out/bdh_europarl_ladG2-*.pt` (group `coding`, g+rw).
- Tools: `scripts/repair_decay.py` (per-segment c-fit), `scripts/boundary_repair.py` (single-scalar boundary restore; `--phase j --c 0.8927|0.57978`), `scripts/eval_router.py` (routes = prefix widths; single-domain probes with `--routes <width>`).
- Staging note: the G2 repair used symlinked sources under a `ladGX` prefix (ladG phases 1–15 + ladG2 phases 16–20).