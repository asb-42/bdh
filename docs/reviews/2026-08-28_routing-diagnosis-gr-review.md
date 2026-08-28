# Review: Routing Diagnosis Arm G+R (20-way)

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-27_routing-diagnosis-gr.md` (commit `b3e10b5`)
**Context:** First intermediate result from Phase 2 (growth + routing combination).

## Verdict

**Kernbefund tragfähig — aber Arithmetik korrigieren, bevor der Bericht zitiert wird. Wichtigster Schritt: freeze_attn-Konfounder per Diagnose-Experiment ausschließen, bevor ein neuer Arm gebaut wird.**

## 1. Der Befund ist real und wichtig

Die zuletzt trainierte Sprache (lt) hat keine routbare Struktur: 32 % Detektion, auf 7 Routen verstreut, Routed-PPL 55,10 = Joint (54,96). Alle älteren Sprachen routen gut (95–100 %).

Das ist das **aktuelle-Phasen-Problem**: Die Wachstumsregel erhält alte Präfixe, konstruiert aber keines für die aktuell trainierte Sprache. Die Formulierung des Berichts („the prefix is an emergent property of training, not an explicit allocation") ist exakt die richtige Diagnose.

## 2. Arithmetik: dieselbe Fehlerklasse wie im letzten Routing-Report

- „Overall: 2/760 = 0.3 % on diagonal" — rechnerisch falsch (2/760 = 0,26 %, und 2 ist ohnehin nicht die korrekte Anzahl).
- „Excluding lt: 744/720 = 100 %" — mathematisch unmöglich (Zähler > Nenner); außerdem sind es/et laut eigener Tabelle nicht 100 %.
- Unabhängige Zählung aus der Tabelle: 16 Sprachen à 40 = 640 + es 38 + et ~33 + lt ~13 = **~724/760 = 95,3 %**, nicht 99,7 % und nicht „alles 100 %".

**Aktion:** Im Report korrigieren (724/760 = 95,3 %, lt als einziger Ausreißer). Das Manuskript darf keine fehlerhafte Prozentarithmetik tragen — die Review-Anmerkung vom 2026-08-27 betraf genau dieselbe Klasse.

## 3. Wichtiger als die Zahlen: freeze_attn ist ein plausibler Konfounder

Der Unterschied zwischen Arm G und Arm G+R ist der freeze_attn-Mechanismus (Commit `8fef089`: „freeze attn weights during growth"). Prüfe den Vergleich:

| Arm | lt-Detektion | lt-Routed-PPL | Attention bei lt-Training |
|---|---|---|---|
| Arm G (ohne R) | saubere Route (Route 20) | 3,74 | trainierbar |
| Arm G+R | 32 %, verstreut | 55,10 (= Joint) | **eingefroren** |

Plausible Erklärung: Lt wurde mit eingefrorener Attention trainiert — die Attention war bereits auf ältere Sprachen optimiert und hatte für die neue Sprache keinen freien Gestaltungsraum. Ohne trainierbare Attention kann lt kein konsistentes neuronales Projektionsmuster formen; es verteilt sich über die vorhandenen Strukturen. Das aktuelle-Phasen-Problem wäre dann ein Artefakt der Freeze-Regel, nicht ein fundamentales Architektur-Limit.

**Empfohlenes Diagnose-Experiment (kostet einen Run, keine neue Arm-Architektur):**
Letzte Phase (oder eine Wiederholung von lt) mit **aufgetauter Attention** trainieren, dann Routing-Diagnose erneut messen.
- Wird lt routbar (≈ 3,7 ppl, saubere Route) → freeze_attn ist der Grund. Dann ist der nächste Schritt: Attention für die aktuelle Phase trainierbar lassen und die Routing-Awareness ins Training einbauen (Option 2 aus dem Bericht, aber jetzt mit präzisem Mechanismus).
- Bleibt lt unroutbar → das aktuelle-Phasen-Problem ist fundamental; dann lohnt sich der Aufwand für Option 2.

## 4. Gegen Option 1 (explizite Präfix-Reservierung) — kurze Anmerkung

Neuronen vorab zu reservieren ähnelt einem learned-gate-Mechanismus; learned in-pass gates wurden bereits falsifiziert (Addendum 12, bcbf022 — Feature Poverty, compiled detector = oracle). Wenn die Diagnose freeze_attn bestätigt, ist route-aware training (Option 2) die theoretisch konsistentere Antwort.

## 5. Gut gemacht

- Der Vergleich Arm G vs Arm G+R ist der richtige Rahmen und gibt eine klare Botschaft: Routing erhält alte Sprachen perfekt, sobald deren Struktur existiert.
- Die Interpretation benennt die Baustelle präzise (emergent vs. allokiert).
- Günstige Methode: reine Inference, keine neuen Trainingsläufe in dieser Diagnose.

## Bottom line

Kernbefund (lt unroutbar, ältere routbar) steht. Arithmetik korrigieren. Vor dem nächsten Arm: freeze_attn-Konfounder mit einem gezielten Run ausschließen. Das ist billiger und präziser als direkt ein neues großes Experiment.
