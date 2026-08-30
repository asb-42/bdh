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
