# Route-Aware Training PoC — Pre-Registered Design

**Date:** 2026-08-28
**Status:** active (approved by maintainer)
**Author:** Quinn (review seat, Saga AI Labs) — design; MiMo (execution seat) — implementation
**Supersedes:** the unscheduled full-ladder option for testing the current-phase problem

---

## 1. Context

The freeze_attn diagnostic (commit `c7b638f`, review `bdce518`) showed that
`freeze_attn` is NOT the confounder for the current-phase routing problem: with
unfrozen attention, the last-trained language (lt) remains unroutable (5% detection,
routed PPL 55.30 = joint). The current-phase problem is fundamental to the growth
recipe: once attention has been shaped by previous phases, the newest language
cannot form a consistent routable projection.

Before spending ~40 h on a full 19-phase ladder with routing-aware training, we run
a small pre-registered proof of concept. If the mechanism works at 2–3 phases, we
scale; if it fails, "accept the limitation" becomes the honest default for the
current phase.

## 2. Hypothesis

**H-PoC: A route-aware training objective (loss computed only on the freshly grown
prefix during phase n) makes phase n's language routable at the end of the run —
detection ≥95% and routed retention within 0.3 nats of its specialist — at 2–3
phase scale.**

Rationale: the current-phase language fails because its representations mix into
the shared structure. If training pressure explicitly forces the new neurons to
carry the new language's loss, those neurons should form a separable prefix.

## 3. Design

### 3.1 Task setup

- 3 Europarl phases (~30 MB each), contiguous blocks, fixed order, e.g. EN -> DE -> ES
  (same family structure as the H1 experiments, so results are comparable).
- Width growth: +32 neurons/head per phase (same rule as Arm G/R).
- Attention: unfrozen during each phase's training. (The diagnostic showed that
  unfreezing alone does not fix a frozen-history model; here there is no freeze
  history because we run from scratch.)
- Optimizer: AdamW, peak lr 1e-3, cosine decay, fresh optimizer per phase, warmup
  1000, batch 4x512 (same as canonical protocol).

### 3.2 The route-aware objective

During phase n training:

- Instantiate a hard prefix mask that keeps only the neurons grown in phase n active
  for the loss computation (mask others to zero in the loss, or equivalently route
  the forward through the phase-n prefix).
- Compute cross-entropy loss on the prefix-masked forward only. Optionally mix a
  small fraction (10%) of standard full-forward loss to keep the model well-formed
  for the older phases; this mixing must be fixed and reported in the log.
- The router itself (likelihood or compiled detector) is NOT trained during this
  PoC; it stays the validated label-free router from the repo.

### 3.3 Eval protocol

- Use the same cold random-crop protocol as all CL runs; within-corpus; protocol
  congruent with training (INV-1).
- After the final phase, run the 20-way routing diagnosis on the final checkpoint
  (same routes, same domains, 40 crops as in `2026-08-27_routing-diagnosis.md`).
- For each phase language, measure:
  - P-Det (routing detection accuracy on the 20-way task)
  - P-Route: routed PPL vs the phase specialist PPL (trained alone on that phase,
    same protocol). Specialist PPL is computed independently; report both the
    specialist value and routed value.

## 4. Pre-registered falsifiers

| Metric | Pass | Partial | Fail |
|---|---|---|---|
| P-Det for phase 2 and 3 | ≥95% | 80–95% | <80% |
| P-Route (routed PPL - specialist PPL) | ≤0.3 nats | 0.3–1.0 nats | >1.0 nats |
| Acquisition (phase target PPL) | ≤2.6 ppl | 2.6–3.5 ppl | >3.5 ppl |

Decision rule:

- **Pass:** P-Det ≥95% AND P-Route ≤0.3 nats on phases 2 and 3. → proceed to a full
  ladder with route-aware training.
- **Partial:** detection or retention in the partial band. → inspect which phase
  fails and why; consider a second PoC with adjusted objective mixing before a
  ladder.
- **Fail:** either phase below the fail threshold. → current-phase problem is
  architectural under this objective; default to "accept the limitation" and write
  it into the manuscript limitations section; do not spend 40 h on a ladder.

## 5. What we explicitly do NOT do in this PoC

- No explicit prefix reservation / neuron pre-allocation (resembles learned gates,
  already falsified — Addendum 12, bcbf022).
- No trained routing head (same reason).
- No quantization, no QAT (separate draft plan, starts only after R3).
- No full 19-phase run unless the PoC passes.

## 6. Deliverables

- Code changes (if any) to support prefix-masked loss in `pipeline/train.py`,
  committed with logs.
- Report: `docs/reports/2026-08-28_route-aware-poc.md` with the three falsifier
  tables, specialist PPL values, and the router confusion output.
- Checkpoint paths and exact commands for reproducibility.

## 7. Open implementation questions (for MiMo, resolve before training)

1. Does the current `train.py` support a mask on the loss/forward that only touches
   the newly grown neuron block? If not, minimal change needed (a `prefix_mask`
   parameter passed to the loss; keep it separate from eval routing).
2. How exactly does the growth rule assign neuron indices per phase (added range)?
   The mask must match that exact range.
3. Should the 10% full-forward mix be applied from the first step of phase 2, or
   annealed in? Prefer fixed mixing for the PoC; note it for the full ladder.

## 8. Timeline

- Implementation + runs: ~6–10 h GPU time on the 4090 (3 phases, 10k steps each at
  growing width).
- Report + review: same day.

---

**Pre-registered on 2026-08-28 before any runs. Signed: Quinn.**
