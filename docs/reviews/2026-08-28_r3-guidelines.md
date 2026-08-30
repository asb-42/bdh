# Guidelines for CL Manuscript R3

**Date:** 2026-08-28
**Author:** Quinn (review seat, Saga AI Labs)
**Status:** handlungsleitende Richtlinien für R3; Herkunft: externe Long-Horizon-R&D-Evaluation, ausgewertet im lokalen Dark-Factory-Kontext. Nur diese operativen Punkte fließen in BDH-CL ein; die Thematik selbst ist dort nicht Gegenstand.

---

## 1. Novelty positioning

Much of the recipe is composition of known building blocks (growth, routing,
merge/prune/replay). The defensible novelty claim is the **formal core**: the
exact-isolation criterion (invariance + restriction equivalence) and the
construction thesis ("prefix growth constructs the structure"). R3 should state
that as the contribution, not the mechanism list.

## 2. Verifier robustness

External evaluations found more evaluation shortcuts than genuine novelty. Our
falsifiers (P-Acq, P-Eros, P-Route) must be hardened against shortcut solutions.
In particular: keep the specialist-baseline framing for P-Route (routed retention
measured against per-phase specialists, not only against a degraded joint
baseline).

## 3. Experience management as method

Report how process experience was managed in this project: pre-registered
falsifiers, protocol congruence (INV-1), review-gate before merge, self-correction
of superseded measurements (E6/E8 precedent). State explicitly that these are
deliberate countermeasures against negative transfer and over-anchoring.

## 4. Reliability reporting

Where seeds exist, report both average and best behavior (the avg/best gap is
large and informative). Mark single-seed runs clearly. This is already the
convention in Table pareto; make it explicit in the protocol description.

---

These four points are the only operational imports from the external evaluation;
they apply to writing and validating R3, not to the BDH research scope.

---

## 5. Checkpoint-verified facts for Arm G/G+R (2026-08-30)

Measured from the saved checkpoints (read-only inspection; see
`docs/reports/2026-08-30_pi-checkpoint-measurement.md`), superseding all
circulating derived numbers:

- Schedule: **20 phases, ×128 → ×736** (the report's ×708 and its 19-row table
  are wrong — the `sv` row was dropped; MiMo's ×738 is off-grid).
- Growth rate: **+32 multiplier units/phase = +2,048 neurons/head** (one unit =
  64 per-head latent neurons). The manuscript's "+32 neurons/head" understates
  by 64×.
- Final size: **579,123,200 ≈ 579M parameters**. The manuscript's "~554M" is
  `sl`, the penultimate phase (553,955,328) — R3 currently describes Arm G one
  phase short; the report's "~700M" is wrong in the other direction.
- Both script headers corrected in the same commit; the report table should be
  re-derived from artifacts, not formulas (any computed cell must be labeled).

Process rule inherited from this finding: **consistency between two derived
numbers is not verification** — only a value read from an artifact breaks the
chain. Recorded as failure class F5 in the team A/B protocol.

## 6. Reporting rule (binding, 2026-08-30)

Every table column that claims a run property (multiplier, parameter count,
step, loss, ppl) must state its source: the artifact it was read from
(checkpoint path, log line) or an explicit `computed` tag with the formula.
Formulas may not silently substitute for measurements, and row counts are not
phase counts — count the sequence in the generating script. Reviewers should
reject any run-property column without provenance.

## 7. Verification claims (binding, 2026-08-30)

From the ?02 verification round (Pi, pi-33):

- **State the strength of the exactness claim.** Where the width changes,
  claim ULP-bounded, not bit-equality: bit-equality across a changed N
  depends on BLAS reduction order, is stronger than the theorems need, and
  can fail spuriously. Bit-equality is only meaningful at identical shapes
  (the zero-init route).
- **Attribute exactness to its operative mechanism:** append-only growth +
  exact-zero initialization of new neurons + verbatim preservation of old
  RoPE frequencies — not to the mask. With k_sparse_ratio > 0, top-k runs
  before the mask (`_k_sparse_relu` at bdh.py:244 precedes the mask at
  :246), so a mask alone cannot preserve computation; zero-init keeps new
  neurons out of top-k. Full derivation: Pi's ?02 artifact (docs/notes/,
  this date).
- **Frequency lattice note (methods):** `2 ** 16 ** (q/n)` is
  right-associative and differs from `(2 ** 16) ** (q/n)`; new neurons
  therefore enter on a different frequency lattice than a from-scratch
  model of the same width.
- **Independent execution:** the author of a verification artifact may not
  attest its own run. ?02 attestation must come from a non-author (Pi on
  the GX10, or the execution seat).
- **Amendment (pi-33, 2026-08-30; derivation sections 3-4 of the cited ?02
  artifact).** Two refinements to the bullets above, both from reading `bdh.py`.
  1. "Not the mask" is too strong: in the ReLU regime the mask is applied after
     `relu` (`bdh.py:246-247`) and is sufficient by itself, which is exactly what S1
     measures as passing. The accurate rule is *not the mask alone, once a
     selection operator precedes it*.
  2. Under ratio-based top-k, neither the mask nor zero-init preserves
     exactness, because `k = int(ratio * width)` (`bdh.py:13`) grows with the
     width and so enlarges the retained set of *old* neurons at the growth step.
     Sparse growth needs an absolute `k`, or the mask applied before selection.
     Consequence: the negative result at `cl-bdh-manuscript.tex:794` (top-k
     forgets worse than ReLU) is predicted structurally, not empirically.
     **Both halves now measured** (2026-08-30, GX10): dense growth is exact to one
     float32 epsilon, and the sparse gap tracks how binding `k` is. Note for the
     record: `scripts/verify_masked_forward.py` as committed at `d1bc095` could not
     run at all, and its `grow`/`zero_new` did not mirror `train.py`'s head-major
     decoder layout; three repairs were needed before any verdict existed, so no
     S1/S2 result from that file should be cited without reading the diff. See the
     derivation's section 7.
