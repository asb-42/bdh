# Fixed-Capacity Sequence Matrix (FCS) — Pre-registered Protocol

Date: 2026-09-10 · Operator proposal: HAK #130 · Quinn design: HAK #131 · Operator GO: 2026-09-10
Seat: A0-Quinn · Host: bdh-4090 (.200, RTX 4090, compiled, fixed-regime code @5422561+)

## Purpose

The one cell our experimental grid never measured: a pure sequential ladder at FIXED capacity —
no growth, no gradient masks, no route-awareness, no replay. Twenty languages overwrite the same
weights; after every phase, all 20 domains are cold-evaluated. The result is a 21×20 matrix whose
rows are training prefixes and whose columns are domains. This is the measured floor against which
every BDH memory claim (route-aware growth: 100 % routing, zero drift, retention = acquisition,
RA2b readout 072a5a3) is contrasted.

## Pre-registered predictions (before any number exists)

- **P-FCS-1 (theory-predicted catastrophe):** without growth or selection, joint serving degrades
  monotonically per domain; earlier languages erode toward tens-of-ppl (paper's own claim:
  freezing weights does not freeze computation). Concretely: by row 20, most early Latin languages
  sit > 20 ppl (vs RA2b joint 20–60 range), with non-Latin domains worst.
- **P-FCS-2 (acquisition floor):** each new language acquires at its RA2b-comparable floor
  (±batch/host offset: .200 batch 4 vs gx10 batch 1, measured +13.7 % acquisition penalty)
  — i.e. acquisition of language k is NOT substantially harder than its first-phase counterpart
  (no capacity-saturation cliff at 100M).
- **P-FCS-3 (interference asymmetry):** backward interference (I_{A→B}) is dominated by the
  MOST RECENT training phase (recency effect) rather than cumulative phase count — the
  depth-recurrent architecture's shared computation should overwrite the newest serving path.
  If instead damage is spread evenly across all later phases, the shared-computation story needs
  revision.

Falsification of P-FCS-1 would be a MAJOR result (growth doing less work than claimed);
confirmation makes the RA2b contrast the measured memory story.

## Protocol (ladder_fixedcap.sh)

- Sequence: same as RA2b — en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt (comparability;
  ordering bias noted in limitations)
- Capacity: FIXED mult 128 (~100M params, n_embd 512, n_head 8, block 512) — the paper's own
  fixed-capacity scale; no --grow-mult, no --route-aware, no masks
- Phase 1: fresh training (no init). Phases 2-20: --init-from <prev>_last.pt (weights restored,
  fresh optimizer — the existing train.py path at :117-120, verified)
- Steps/phase: 10000, batch 4 (RA2b phase-1 protocol), warmup 1000, cosine 10000 (schedule-
  congruent with RA2b phase 1; the per-phase closed-form leak factor is irrelevant here: nothing
  is masked, wd acts on all weights uniformly, which is the intended behavior at fixed capacity)
- After EVERY phase (including phase 1, and row 0 before phase 1):
  `lang_eval.py <ckpt> 30 <all-20-langs-csv> 4` → one matrix row (random-crop cold eval,
  protocol-congruent, generator 1234)
- Row 0: zero-shot eval of the random-init model is SKIPPED as meaningless (random init has no
  language structure; a zero-shot row would measure tokenizer-free byte noise). Instead, row 1
  (after EN) is the first informative row: EN acquisition + 19 zero-shot transfer cells. This
  deviates from operator #130's row 0; rationale: the random-init row is pure noise, and the
  forward-transfer baseline for every language is its own first-column entry when it is (or was)
  trained — measured, not extrapolated. Forward transfer on never-trained languages uses the
  row-1 zero-shot cells as the 'before' anchor where meaningful.
- Outputs: out/logs/fixedcap_matrix.txt (accumulating rows), per-phase logs
  fixedcap_<lang>.log, checkpoints ladFCS-<lang>_{last,best}.pt (nothing overwrites any other arm)

## Cost estimate (measured basis: .200 S/M-runs at mult 416-448, ~105 ms/step compiled)

- 20 × 10k steps at mult 128 ≈ faster than the 448-cell (smaller model, same IO) — est. 12-20
  min/phase → 4-7 h training total
- 21 matrix evals × 20 domains × 100 crops × batch 4: ~6-10 min each → 2-3.5 h
- **Total: ~7-10 h, one overnight run on .200** (vs ~44 h+ on gx10 eager)

## Confounds (pre-declared)

1. Host/era: RA2b acquisition numbers are gx10 (eager, batch 1); FCS runs on .200 (compiled,
  batch 4). Cross-arm acquisition comparison carries the measured +13.7 % batch penalty and
  compile transparency (S1e ≤ 0.7 %). Within-matrix drift (the primary readout) is same-host,
  same-instrument, exact.
2. Sequence bias: one of 20! orderings (as RA2b); limitations note.
3. Seed floor: single-seed run; cell deltas < 2-4 % not quotable as effects (measured floor).
4. The fixed-capacity wd term: at fixed capacity, weight decay on all weights is the standard
  regime (no masked path to protect) — not a leak; noted for reviewer clarity.

## Success criteria

- Complete 21×20 matrix on disk, protocol-congruent, one commit + bus post with attachment
- P-FCS-1/2/3 verdicts signed PASS/FAIL/UNRESOLVED per R4
- Direct contrast table FCS row-20 vs RA2b p20 (joint and routed) in the readout

— A0-Quinn, 2026-09-10
