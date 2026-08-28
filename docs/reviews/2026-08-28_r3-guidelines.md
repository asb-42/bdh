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
