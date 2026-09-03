# Correction scope: the frozen-attention narrative (F-V7 null-contrast instance)

**Date:** 2026-09-03 · **Seat:** pi-50 (gx10-50ef, review role) · **Base:** bdh@25d1340
**Provenance:** derived from bus thread #75 (A0-Quinn), #77/#79/#80 (pi-50), #78 item 3 (acceptance at full scope).
Method and classification heuristic are stated inside; the CAUSAL/NARRATIVE split is a first pass for human
confirmation, not a verdict. Landed as an artifact on Quinn's go (#78 item 5) so the reclassification lives in one
place instead of chat scrollback.

freeze history|unfrozen|locked by|resistant to reorganiz|cannot reorganize|not the confounder), then
classified heuristically: CAUSAL = asserts why something happened; NARRATIVE = describes/references;
FLAG = mentions the CLI/config knob only. Classification is a first pass for human confirmation, not a verdict.

Totals: 72 matching lines in 15 files. Classes: CAUSAL=9, FLAG=2, NARRATIVE=61

| file | lines |
|---|---|
| `docs/reviews/2026-08-28_freezattn-diagnostic-review.md` | 13 |
| `docs/reports/2026-08-27_freeze-attn-diagnostic.md` | 11 |
| `docs/reviews/2026-08-28_route-aware-poc-review.md` | 11 |
| `docs/reviews/2026-08-28_routing-diagnosis-gr-review.md` | 9 |
| `docs/reports/2026-08-28_route-aware-poc-v2.md` | 5 |
| `docs/plans/2026-08-28_route-aware-poc.md` | 4 |
| `docs/reports/2026-08-27_routing-diagnosis-gr.md` | 4 |
| `docs/plans/2026-08-28_qat-proposal.md` | 3 |
| `docs/reports/2026-09-03_decay-leak-finding.md` | 3 |
| `docs/reports/2026-08-28_route-aware-poc.md` | 2 |
| `docs/reports/2026-08-30_ladRA3-progress-report.md` | 2 |
| `docs/reviews/2026-08-29_route-aware-poc-v2-review.md` | 2 |
| `docs/reviews/2026-08-29_quinn_pi-review-audit.md` | 1 |
| `docs/reviews/2026-08-31_weight-atlas-bdh-implementation-review.md` | 1 |
| `docs/tasks/2026-08-30_ladder-ra2-validation-checklist.md` | 1 |

## CAUSAL-class assertions (the sentences that cannot stand as evidence)

- `docs/plans/2026-08-28_route-aware-poc.md:13` — `freeze_attn` is NOT the confounder for the current-phase routing problem: with
- `docs/reports/2026-08-27_freeze-attn-diagnostic.md:81` — **freeze_attn is NOT the confounder.** The unfrozen model was initialized from `ladGR-sl_last.pt` — a model that already had 19 phases of frozen atten
- `docs/reports/2026-08-27_freeze-attn-diagnostic.md:85` — **Root cause:** Once attention is frozen for multiple phases, the resulting structure is resistant to reorganization for new languages. The growth rec
- `docs/reports/2026-08-27_freeze-attn-diagnostic.md:94` — 3. **Arm G's success with lt** was because it was trained from scratch, not on top of a model with frozen attention history
- `docs/reports/2026-08-30_ladRA3-progress-report.md:5` — The R3 ladder (α=0.9, unfrozen attention, 20-phase growth) completed **12 of 20 phases** before hitting a GPU memory wall on the RTX 4090 (24 GB). Pha
- `docs/reports/2026-09-03_decay-leak-finding.md:123` — - The 2026-08-27 freeze_attn diagnostic therefore compared two effectively
- `docs/reviews/2026-08-28_freezattn-diagnostic-review.md:11` — **Core conclusion accepted: freeze_attn alone is not the confounder; the current-phase
- `docs/reviews/2026-08-28_freezattn-diagnostic-review.md:28` — unfreezing the last phase cannot reorganize it. The growth recipe creates a path
- `docs/reviews/2026-08-28_route-aware-poc-review.md:23` — frozen-history model; here there is no freeze history because we run from scratch").

## Source-level facts behind this

- `bdh.py:83-91` — `class Attention` contains only `self.freqs = torch.nn.Buffer(...)`; no `nn.Parameter`.
- `bdh.py:173-189` — BDH parameters are `encoder`, `encoder_v`, `decoder` (+ `embed`/`lm_head` wrapper).
- `pipeline/train.py:150-151` — `for p in raw_model.attn.parameters(): p.requires_grad_(False)` iterates an empty generator on `--model bdh`.
- Already recorded by A0-Quinn in `docs/reports/2026-09-03_decay-leak-finding.md:119-128` (no-op, cosmetic log lines, diagnostic compared effectively identical configs, conclusion 'stands trivially').

## Flag-inertness sweep (negative result)

- `dropout` — LIVE on BDH (`bdh.py:188 self.drop = nn.Dropout(config.dropout)`).
- `chunk_size` — referenced only inside `pipeline/config.py:124,144` when building the bdh-linear variant; behaves as its comment scopes it.
- `baseline_n_layer` — used at `config.py:150,207` for the transformer baseline; as commented.
- No additional dead flags found. (Two names I hypothesised, `qk_norm` and `rope_theta`, do not exist anywhere in the tree —
  a `grep` returning zero matches had looked like 'flag with no effect'; corrected before reporting.)

