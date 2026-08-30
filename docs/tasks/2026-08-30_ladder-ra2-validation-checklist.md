# ladder_ra2.sh — Validation Checklist (run after completion)

**Date:** 2026-08-30 · **Author:** Quinn (review seat) · **Status:** pre-registered; execute when the R3 ladder finishes on the 4090

Validating the R3 ladder output BEFORE any number enters a report or the
manuscript. Checks reference the binding rules in
`docs/reviews/2026-08-28_r3-guidelines.md` (Sections 5–7). Failure classes
where established: F5 = consistency between derived numbers is not
verification.

## 0. Run integrity (do first)

- [ ] Parse `out/logs/ladder_ra2_analysis.txt`: every phase reached `done`;
      no phase silently skipped.
- [ ] **OOM audit (critical given reported occasional OOMs):** collect all
      OOM/retry markers. For every restarted phase, verify INIT was the
      previous phase's `_last` checkpoint (chain integrity), never a re-init
      from BASE. A phase that restarted from BASE invalidates its own and
      all downstream numbers.
- [ ] `step=10000` present in every phase checkpoint (Arm G/R convention).

## 1. Provenance (Section 6)

- [ ] Every run-property column in every derived table cites its artifact
      (checkpoint path / log line) or carries an explicit `computed` tag
      with the formula.
- [ ] No column is silently re-derived from another derived column.

## 2. Parameter/multiplier oracle (F5)

- [ ] For every phase checkpoint: `bytes/12` ≈ P(m), and
      P(m) = 786432·m + 262144 (+64·m), with m read from the checkpoint cfg
      (not from any script or report).
- [ ] All multipliers on the ×32 grid; schedule 20 phases ×128 → ×736;
      final ≈ 579,123,200 parameters.
- [ ] Per-head widths = 64·m (routes are in per-head neuron units).

## 3. Route-grid validity

- [ ] Every routing file contains/prints its route list; routes are
      boundary-aligned (one per expert-cap boundary incl. BASE) and in
      per-head neuron units (64×mult).
- [ ] 20 domains incl. `sv`; no duplicate `en` column in milestone
      matrices; no clamped/phantom route at the top of the grid.
- [ ] Instrument type labeled per table (boundary-aligned vs. historical
      linspace): historical P-Route/P-Det numbers are NOT boundary-comparable.

## 4. Arithmetic re-derivation (F5)

- [ ] Re-derive every derived cell independently. Two mutually consistent
      derived numbers are NOT verification.
- [ ] Falsifier ratios (P-Acq / P-Eros / P-Route / P-Det) recomputed from
      raw artifact values with the pre-registered definitions (α per arm;
      specialist baseline for P-Route per Section 2).

## 5. Protocol congruence (the PoC-v2 lesson)

- [ ] Attention state frozen/thawed exactly as pre-registered (v2 violated
      this; check the run config, not the report's claim).
- [ ] LR schedule congruent with the pre-registration.
- [ ] Eval protocol identical across phases and arms (eval_router.py at the
      fixed version, d10d389 or later).

## 6. Verification claims (Section 7)

- [ ] Exactness claims stated as ULP-bounded (not bit-equality) wherever
      the width changed.
- [ ] Exactness attribution names the operative mechanism (zero-init + RoPE
      frequency preservation, plus the mask in the ReLU regime), never the
      mask alone under a preceding selection operator.
- [ ] Citable S1/S2 evidence comes only from `verify_masked_forward.py` at
      4c8e51e or later (GX10-attested by pi-33). The original d1bc095
      version never produced a verdict — recorded as failure class F6 in
      the team A/B protocol.

## 7. Independence

- [ ] Validation executed by a seat that did not author the run artifacts
      (execution seat validates; Quinn/Pi cross-check).
- [ ] Validator findings enter the record as erratum or confirmation, each
      with its artifact reference.
