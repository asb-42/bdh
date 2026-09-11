# Sonde C — Embedding-Router Probe (Self-Distilled Addressing)

Status: PRE-REGISTERED DRAFT v0.1 · 2026-09-11 · Quinn · Operator GO 2026-09-11
Position in roadmap: after Sonde B (labels come from any grown multi-domain
model; Sonde B's model is the natural source). Depends on: nothing else.

## Purpose (the P-R3 gap)

P-R3 measured that byte 1–4-gram geometry addresses 16/20 domains perfectly
but fails on Latin-script neighbors of early territories (es/pl/sk at 0.00,
et at 0.67; Wilson [0.659, 0.842] overall). The failed cells are exactly
where domain identity is not byte-distinct — and the PoC's chat phases will be
byte-IDENTICAL with their host corpus. An input-side router that uses richer
representation than byte counts is therefore not optional for the PoC; it is
the condition for chat-CL.

The operator's question (2026-09-11): why not give the router an embedding-based
semantic layer? Answer: nothing principled stands against it — but the variants
differ in whether they import foreign competence (violating the own-AI constraint)
or distill it from BDH itself. This probe tests the own-AI variant first.

## Design

Three arms, all inference-only or offline-fitting; no ladder training.

### Arm 1 — Self-distilled linear head on BDH early residuals (own-AI default)

Take the grown multi-domain model (Sonde B's, or RA2b-lt as fallback — 20
language territories already exist). For each held-out crop:
1. Forward pass up to an early layer; extract the residual-stream state at a
   mid position (e.g. position 64 of 512).
2. Labels: the argmin-NLL route (the A2 protocol — self-supervised, no human
   task IDs), already validated at 20/20 on RA2b.
3. Fit a small head (logistic regression on the pooled residual, or a 2-layer
   MLP if the linear is insufficient) to predict the route.
4. Report: top-1 agreement on held-out crops, per-domain, with Wilson
   intervals; compare against P-R3's byte-n-gram predictor on the same splits.

Key hypothesis: the residual stream at mid-depth carries territory-relevant
structure that byte counts do not — the A1 energy-null (block-0 dominance in
ACTIVATION STATISTICS) does not preclude this, because a TRAINED head on
residuals is a different object than an untrained energy statistic.

### Arm 2 — External embedding control (upper bound, NOT own-AI)

Same protocol, but features from a small external sentence-embedding model
(e.g. a MiniLM-class model run locally on the 4090). Purpose: measure the
upper bound of what embedding-based addressing could achieve, so Arm 1's
result is interpretable ("self-distilled reaches X % of the external bound").
Explicitly labeled as foreign-competence control in any manuscript use; not
a candidate for the PoC serving path under the own-AI constraint.

### Arm 3 — Byte-n-gram + residual ensemble (cheap-first cascade)

The production shape: byte-n-gram as stage 1 (free, perfect on script-level
clusters), residual head as stage 2 only for inputs the n-gram marks as
uncertain (low-margin predictions). Measure: cascade accuracy vs. full-head
accuracy vs. P-R3 alone, and the fraction of inputs needing stage 2.
This is the PoC addressing architecture if Arms 1–2 work.

## Pre-registered predictions

- **P-C1 (residual head > byte-n-gram on the hard cells):** the self-distilled
  head beats 0.762 overall and specifically rescues the Latin-neighbor failures
  (es/pl/sk). Prediction: ≥0.90 overall; hard cells ≥0.60. Falsifier: head ≤
  byte-n-gram → the residual stream at that depth carries no more address than
  byte counts, and the distillation route needs deeper probes or fails.
- **P-C2 (external bound high):** Arm 2 ≥0.95 overall — embedding space clearly
  contains the address. If Arm 2 is ALSO low, the address genuinely lives
  only in the likelihood computation and input-side routing of any kind is
  limited (the P-R3 falsifier scenario, now measured two ways).
- **P-C3 (cascade economy):** Arm 3 reaches ≥ Arm 1 accuracy with ≤ 30 % of
  inputs routed to stage 2.
- **P-C4 (OOD consistency):** for unseen languages (lv, ga, zh/ja/hi/iu-clean
  artifacts on .200), the cascade's chosen routes match the likelihood router's
  choices at ≥ the P-R3 level, and the two-axis reject still separates (the
  head must not create false in-support confidence on OOD inputs).

## Gates

- **Gate C-PASS:** P-C1 holds → self-distilled addressing works; PoC serving
  architecture = cascade (Arm 3), own-AI intact.
- **Gate C-PARTIAL:** P-C1 fails but P-C2 holds → the address exists in
  representation space but not cheaply self-distilled; decision point for the
  operator: accept external embeddings for the router (pragmatic) or invest in
  a small own embedding model (own-AI, more work).
- **Gate C-FAIL:** P-C1 and P-C2 both fail → input-side addressing is
  fundamentally limited to byte-distinct domains; chat-CL needs the likelihood
  scan (O(K)) or a fundamentally different mechanism. Recorded as a boundary
  result; PoC scope adjusted.

## Budget

- Label generation: 20 domains × 96 crops × 23 widths on RA2b-lt (already
  exists as the r3b density run's persisted features, if pi-50's format is
  reusable) — ~2 h GPU if regenerated.
- Residual extraction + head fitting: minutes (CPU or one GPU pass).
- External embedding control: one MiniLM pass over the same crops, ~minutes
  on the 4090.
- **Total: ≤ 3 h GPU.**

## Relationship to prior work

Expert Gate (Aljundi et al., CVPR 2017) is task-embedding-based addressing;
our variant differs in (i) self-supervised labels (argmin-NLL, no task IDs),
(ii) the measured byte-geometry base layer with its known failure cells, and
(iii) the two-axis OOD rejection coupled to the cascade. Positioned in prior-art
cluster B of the rev-4 outline; this probe provides the measured numbers for
that positioning.

## Non-goals

- No new ladder training.
- No own embedding-model pretraining (that is a Gate C-PARTIAL contingency,
  not this probe).
- No serving integration (cascade design only; integration is PoC work).

— Quinn, 2026-09-11
