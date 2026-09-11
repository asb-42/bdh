# Sonde B — Domain-CL Probe (Code/Math/Prose)

Status: PRE-REGISTERED DRAFT v0.1 · 2026-09-11 · Quinn · Operator GO 2026-09-11
Position in roadmap: between rev-4 paper and PoC full pretrain (see roadmap
conversation 2026-09-11). Sonde A (scaling probe) is drafted separately.

## Purpose (the M3 bet)

All CL evidence to date uses language identity as the domain criterion: Europarl
languages are byte-distinct, objective, and likelihoood-routable (99.7 %). The PoC
thesis ("Amoeba grows via continual learning") requires that growth+selection
works for non-language domains — code, mathematics, general prose — where no
clean identity exists. Sonde B measures whether CL phases on such domains write
separable, preserved, addressable territories, using the same instruments that
worked for languages.

This is the cheapest test of the biggest unknown in the roadmap. If it fails,
the PoC design changes fundamentally; if it passes, the full pretrain proceeds
with evidence.

## Design

Base model: fresh BDH at mult 128 (~100M), block_size 512 (Sonde B does not
need long context — it measures growth, not chat). Trained on a small mixed
corpus (~1.5 GB total, ~300M tokens at byte level) to give the base some shared
structure before growth begins. Data: FineWeb-Edu sample (prose), plus a small
multilingual sample (no Europarl languages used later as CL phases).

CL phases (4, byte-distinct, ascending ambiguity):
1. **Code** — Python source (e.g. The Stack sample, filtered). Byte-distinct:
   indentation, ASCII-heavy, keywords, punctuation patterns.
2. **Math** — LaTeX source (arXiv sample or similar). Byte-distinct: backslashes,
   braces, command words, symbol density.
3. **Prose-2** — a second natural-language corpus from a DIFFERENT register
   (e.g. legal text, medical, or a non-European language — NOT a Europarl
   language we already tested; the point is to include one phase whose byte
   statistics overlap heavily with the base, making it the hard case).
4. **Control: a new Europarl language** (e.g. Irish/Gaeilge via DGT, already
   downloaded) — a known-good phase to validate the protocol still works in
   this run.

Per phase: 10k steps, batch 1 (route-aware, same as RA2b protocol), growth
mult +32, route-aware alpha 0.9, cosine schedule. Fresh optimizer per phase
(C4 ruling).

## Measurements (same instruments, same pre-registrations)

- **Acquisition:** best val ppl per phase (expect code/math to acquire well if
  the architecture generalizes; this is a hypothesis, not a given).
- **Territory:** P5 bit-check per growth transition (encoder/encoder_v/decoder
  slices, embed/lm_head).
- **Routing:** eval_router with per-phase routes + all-domain diagnosis
  (confusion matrix, 40 crops/domain).
- **Retention:** routed ppl at own-width for each earlier phase vs. its
  acquisition.
- **Joint vs. routed:** per-domain joint measurement for the two-axis check.
- **Selection:** exp4-style self-NLL selector (20/20 protocol) over the grown
  model — does the label-free selector find code vs. math vs. prose?

## Pre-registered predictions

- **P-B1 (territory):** All phases write bit-preserved territories (P5 PASS
  expected — the masking mechanism is domain-blind).
- **P-B2 (acquisition):** Code and math acquire (val ppl drops to reasonable
  levels); prose-2 acquires (it's still language). Falsifier: code or math
  fails to acquire — would indicate byte-level BDH cannot model structured
  non-language domains.
- **P-B3 (routing):** Code/math/prose-2 route to their own territories at
  high accuracy — byte statistics differ enough. The prose-2 phase is the
  hard case (prediction: lower routing accuracy, family-adjacent to base).
- **P-B4 (retention):** Routed retention = acquisition for all phases.
- **P-B5 (selection):** The self-NLL selector reaches ≥75 % on the 4-domain
  diagnosis (not 20/20 — only 4 domains, harder base-overlap for prose).

## Gates

- **Gate B-PASS:** P-B1 and P-B4 hold (territory + retention). This alone
  validates the growth mechanism for non-language domains. → PoC pretrain
  proceeds with CL phases planned.
- **Gate B-FAIL:** P-B1 or P-B4 fail. → The growth mechanism is language-
  specific or byte-identity-bound; the PoC pivots to a different CL strategy
  or documents the boundary.
- **Partial:** P-B3/P-B5 fail on prose-2 (routing/selection confusion) but
  hold on code/math. → Domain choice becomes a PoC design constraint (choose
  byte-distinct phases), embedding router (Sonde C) becomes mandatory.

## Budget

- Base pretrain: ~300M tokens × 100M params at 22k tok/s (4090, compiled,
  measured in Stage B) ≈ 4 hours.
- CL phases: 4 × 10k steps × ~150 ms/step (route-aware, eager, batch 1;
  from RA2b phase-1 timing at similar width) ≈ 7 hours.
- Evals + routing: ~2 hours.
- **Total: ~13 hours on the 4090.** One overnight run, or two evenings.

## Non-goals

- No chat instruction tuning (that is post-Sonde-B).
- No scaling beyond 100M (Sonde A covers that).
- No embedding router (Sonde C covers that — but its labels come from this
  probe if it runs).

— Quinn, 2026-09-11
