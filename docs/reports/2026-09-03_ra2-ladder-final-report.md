# RA2 Route-Aware Growth Ladder — Final Report (Phases 1–20)

- Date: 2026-09-03 ~07:45 CEST
- Author: A0-Quinn (Agent Zero seat, backend glm-5.3-flash via B.AI)
- Data: gx10 /srv/coding/bdh/out/logs/ (ladder_ra2_analysis.txt, ladRA2_routdiag_p20.txt, ladRA2_lt.log)
- Status: training complete (20/20); p20 boundary grid still running at writing time

## 1. Run summary

- 20/20 phases complete: base en→ro (phases 1–12, width mult 128→480), resume fi hu bg et el sk sl lt (phases 13–20, 480→736), route-aware (alpha 0.9), no-freeze-attn, zero-init growth.
- Final phase lt: best val_loss 2.2774 (ppl 9.75), test ppl 9.43 at step 10000; checkpoint out/bdh_europarl_ladRA2-lt_best.pt.
- All 20 phase checkpoints and routdiag outputs p13–p20 on disk. Cold evals exist for phases 1 and 13–20; phases 2–12 are covered by training-log best-val instead (Appendix A) — the .200-era logs are complete on gx10 and carry no cold-eval blocks.

## 2. Acquisition: no width tax

Cold eval ppl at each phase's own checkpoint: en 2.29 (p1) · fi 8.29 · hu 8.47 · bg 15.51 · et 9.35 · el 15.24 · sk 10.19 · sl 9.45 · lt 9.94.

The last phase acquires as well as any earlier one — growing to mult 736 does not tax new-language learning. Non-Latin scripts (bg 15.5, el 15.2) acquire ~5–7 ppl worse than the Latin-script languages; §4 shows this is script/tokenizer cost, not interference (they route perfectly and keep acquisition-level quality).

## 3. Joint full-width serving forgets — non-uniformly

p20 milestone eval (all segments active, 19 seen languages; lt from its phase eval):

bg 1649.2 · el 890.6 · hu 57.6 · cs 52.2 · pl 43.8 · sk 42.3 · sv 37.6 · fi 37.5 · ro 35.4 · da 30.5 · de 29.6 · en 29.5 · fr 28.9 · et 28.1 · nl 27.6 · pt 25.4 · es 23.6 · sl 23.5 · it 21.5 · lt 9.94

- The two non-Latin resume languages collapse completely (bg 106x, el 58x its acquisition ppl); Latin-script languages degrade 2–7x.
- Base languages improved between the p15 and p20 milestones (cs 85→52, pl 73→44, es 31→24, it 28→21, pt 34→25, fr 36→29): later family segments (sk, sl) help related languages even at joint serving. Growth does not hurt the base languages.
- Whether the bg/el collapse is interference or segment destruction is decided in §4.

## 4. Likelihood routing restores the forgotten languages

p20 routing diagnosis: 20 prefix routes (8,192→47,104 neurons/head), per-crop argmin of loss on the first 128 tokens, then serve the remaining positions with the chosen prefix.

Route selection (crops per true domain; prefix identified by width = 8192 + 2048·j, i.e. mult 128 + 32·j):

| domain | own-prefix crops | main leaks |
|---|---|---|
| fi | 36/40 (mult 512) | 4 → et prefix (both Finno-Ugric) |
| hu | 40/40 (544) | — |
| bg | 40/40 (576) | — |
| et | 40/40 (608) | — |
| el | 40/40 (640) | — |
| sk | 40/40 (672) | — |
| sl | 40/40 (704) | — |
| lt | 40/40 (736) | — |
| cs | 0/40 | 40 → sk prefix (Slavic→Slavic) |
| pl | 0/40 | 37 → sk prefix |
| es, fr, it | 0/40 | Romance cluster on ro prefix: es 25, fr 35, it 28 of 40 |
| pt | 1/40 | 18 → sk prefix, 13 → ro prefix |
| ro | 19/40 | 14 → sk prefix |
| en, de, nl, da, sv | 0/40 | scattered mid prefixes; sv → et prefix 24/40 (Estonian's heavy Germanic lexicon) |

7 of 8 resume languages route 40/40 crops to their own training prefix; fi routes 36/40, and its 4 misses go to the et prefix (its Finno-Ugric neighbour). No resume-language crop lands on an unrelated family's prefix.

Serving quality (ppl):

| domain | acquisition | joint @ p20 | routed @ p20 | routed vs acq |
|---|---|---|---|---|
| fi | 8.29 | 37.51 | 16.36 | +97% (outlier; 4 crops on et prefix) |
| hu | 8.47 | 57.59 | 10.96 | +29% |
| bg | 15.51 | 1649.24 | 15.65 | +0.9% |
| et | 9.35 | 28.07 | 9.86 | +5% |
| el | 15.24 | 890.55 | 16.11 | +6% |
| sk | 10.19 | 42.29 | 10.85 | +6% |
| sl | 9.45 | 23.52 | 10.07 | +7% |
| lt | 9.94 | (9.94)* | 10.81 | +9% |

*lt is the final phase: joint serving is its own prefix by construction.

- Routed beats joint on all 19 non-final domains; joint aggregate 43.26 (served positions only).
- Headline: old-language knowledge survives growth intact inside its width segment — routed serving is within 10% of acquisition for 6/8 resume languages (bg within 1%), hu +29%, fi +97%. The catastrophic forgetting at joint serving is co-activation interference at inference time, not weight destruction. Route-aware growth (alpha 0.9) plus a 128-token-window likelihood router largely eliminates forgetting for the trained segments at ladder scale.
- Residual: base languages have no protected own segment (phase 1 predates route-aware training); routing still roughly halves their joint-serving ppl (en 29.5→11.8, cs 52.2→16.6) but does not restore base-model quality (en acquisition 2.29).
- Open anomaly: fi routed 16.36 vs acquisition 8.29 (+97%) — the only resume language materially above acquisition under (nearly) its own prefix; needs a per-route serving breakdown before interpretation.

## 5. Instrument finding: eval_router.py header bug (fixed in this commit)

The confusion matrix printed DOMAIN names as column headers, but columns are ROUTE INDICES (prefix widths 8192+2048j). With 20 routes × 20 domains the output masquerades as a language confusion matrix — bg→pl (40/40) actually means bg→prefix width 36,864, which IS bg's own training width. Treacherous coincidence: sk and sl sit exactly on their own names (alphabetical rank equals width index); the other resume languages do not — lt's own-prefix hits land in the column labeled 'sv', fi's in 'lt', bg's in 'pl', el's in 'ro', et's in 'pt', hu's in 'nl'. Row labels (true domains) are correct. All earlier routing outputs from this instrument (poc_ra2 a05/a09/a10, ladG, ladGR routing files) need re-reading with routes-as-columns. This commit fixes the header to print prefix widths.

## 6. Ops notes

- ladRA2_routdiag_p20.txt backed up to /tmp (md5 7e1130f597c7701445ccb0a5c4b28863) before boundary-grid completion; the launcher appends (>>), so no truncation risk.
- Boundary grid p20 (pre-registered P-Det definition) was still running (PID 85798, ~57 min) at writing time; the watcher logs ladder-RA2-resume-done on exit, after which the tree is free for pi-50's ff-only merge of this fix.
- Scan handoff: the weight-atlas series requested in bdh-cl seq 38 (en/de/pt/lt checkpoints) now has a sharper target — verify whether bg/el segment preservation at p20 is structural (stable spectral norms) or functional-only.

## Appendix A: complete acquisition curve, phases 1–20 (added post-push, same day)

Source: each phase's training log (`out/logs/ladRA2_<lang>.log`, transferred from ai with md5 verification on 2026-08-31) — the `done. best val_loss` line exists for all 20 phases. Post-hoc cold evals exist only for phases 1 and 13–20 (the gx10-resumed analysis file). The two instruments agree on all 9 overlap points: cold eval reads +1.0% to +4.9% above training best-val (mean +2.4%) — consistent, no contradiction.

| phase | lang | mult after | best val ppl (log) | cold eval ppl |
|---|---|---|---|---|
| 1 | en | 128 | 2.25 | 2.29 |
| 2 | es | 160 | 2.40 | — |
| 3 | pl | 192 | 3.17 | — |
| 4 | fr | 224 | 2.45 | — |
| 5 | de | 256 | 2.72 | — |
| 6 | nl | 288 | 3.37 | — |
| 7 | it | 320 | 3.43 | — |
| 8 | sv | 352 | 8.39 | — |
| 9 | da | 384 | 7.96 | — |
| 10 | pt | 416 | 7.66 | — |
| 11 | cz (=cs) | 448 | 9.66 | — |
| 12 | ro | 480 | 9.32 | — |
| 13 | fi | 512 | 7.90 | 8.29 |
| 14 | hu | 544 | 8.34 | 8.47 |
| 15 | bg | 576 | 15.06 | 15.51 |
| 16 | et | 608 | 9.08 | 9.35 |
| 17 | el | 640 | 14.99 | 15.24 |
| 18 | sk | 672 | 10.09 | 10.19 |
| 19 | sl | 704 | 9.18 | 9.45 |
| 20 | lt | 736 | 9.75 | 9.94 |

- cz carries cs data per Addendum 1; the cz log's growth line (`416 -> 448 mult`) confirms the phase-11 schedule position exactly.
- **Within-family observation (correlational, cause not isolated):** the later sibling of each language family acquires 3–4x worse than the early one — es 2.40 (phase 2) vs pt 7.66 (phase 10); fr 2.45 (phase 4) vs ro 9.32 (phase 12); pl 3.17 (phase 3) vs cz 9.66 (phase 11); de 2.72 (phase 5) vs sv 8.39 / da 7.96 (phases 8–9). Position in the ladder predicts acquisition cost better than language identity. Confound, unstated elsewhere and unresolved here: width and frozen-segment count grow together in the base ladder, so interference (§3/§4 story) and width cannot be separated from this data alone. Note the resume ladder shows NO such rise (fi 7.90 → lt 9.75 across 480→736), so 'no width tax' (§2) holds within the resume regime; the base-ladder rise is real but its driver is open.
- **Batch provenance (sharpens seq 40 / F-V3):** the primary phase logs contain NO batch marker at all (grep `batch` = 0 hits in every log checked: es, pt, ro, fi, lt). The only machine-written batch values are the analysis-file phase header lines (`batch=4` for phase 2). The progress-report hand table's `2` has no primary source. Machine provenance favors batch=4 for phases 2–4 (matching the committed script's `BS=4 when PREV_MULT<=192`); the hand table's 2 is unsupported by any artifact found on either host.
