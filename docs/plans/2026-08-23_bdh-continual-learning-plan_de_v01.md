# BDH Continual Learning — Forschungsplan (v0.1)

**Datum:** 2026-08-23 · **Status:** Entwurf zur gemeinsamen Prüfung · **Ledger:** `.jspace/` im Projektroot

## 0. Mission

Wie bleiben statische Gewichte nach Trainingsende neuroplastisch — ohne katastrophales Vergessen?
Konkret: Mechanismus C entwerfen, der akkumulierten Synapsen-State σ_l konsolidiert, und damit den Schuldschein einlösen, den das Paper selbst ausstellt („we do not provide a direct answer“, Zeile 379).

## 1. Problemformalisierung

```
C: (σ_acc, G_alt, optionales Replay-Budget R) → G_neu
```

Anforderungen:
1. **Vorwärts:** Nach dem Training gemachte Erfahrung wird fähigkeitswirksam.
2. **Rückwärts:** Alte Fähigkeiten degradieren begrenzt — BWT ≥ −ε über alle bisherigen Domänen.
3. **Zyklierbarkeit:** Beliebig oft wiederholbar ohne Drift (Stability–Plasticity über viele Zyklen).
4. **Lokalität (weich):** Möglichst lokale Regeln im Wake-Modus; Gradienten allenfalls zyklisch.

**Architektur-Zwang:** Das Paper leitet her, dass σ-Nutzsignal und Schreibraten bei T ≈ 1/ρ kollidieren (ρ ≈ 5 % Sparsity → untere Grenze ≈ 20 Tokens). Konsolidierung muss daher **periodisch** sein — die Architektur verlangt Schlaf. Wir nutzen das als Designprinzip: kein einmaliger Transfer, sondern ein Schlaf-Wach-Zyklus.

## 2. BDH-eigene Hebel

| # | Eigenschaft | Nutzen für Continual Learning |
|---|---|---|
| 1 | 1:1 Ratio Parameter ↔ State (beide O(n²)) | σ und G sind shape-gleich → Konsolidierung ist Residual-Write ohne Projektion |
| 2 | Monosemante Synapsen (< 100M skaliert) | Write-Gates können auditieren, WAS sie ändern, bevor sie es tun |
| 3 | Sparse positive Aktivierungen (ρ ≈ 5 %) | Importanz-Statistiken lokal/billig; Gradient-DAG fast baumförmig für T < 1/ρ |
| 4 | Bewiesenes Zero-Shot-Merging (ES/FR/PT) | Strukturelles CL ohne Joint Training ist plausibel |
| 5 | Per-Edge-Damping u(i,j) | Native Multi-Timescale-Kaskade (STP → LTP abbildbar) |

## 3. Sechs Mechanismen (nach Invasivität)

- **A — Naiver Perioden-Merge (Baseline):** G ← G + λ·ΣΔσ alle W Tokens. Erwartung: messbares Vergessen. Zweck: Floor quantifizieren, Messapparat testen.
- **B — Eligibility-Gating:** Schreiben nur Kanten mit ≥ K Potenzierungen über ≥ D Kontexte; Rate ∝ Potenzierung. Rein lokal, label-frei.
- **C — BDH-EWC:** Per-Kanten-Importanz F_ij aus Koaktivierungsstatistik; Schrittweite η/(F_ij+ε). Lokales Elastic-Weight-Consolidation ohne Backprop.
- **D — Schlaf mit Replay:** Konsolidierung mischt Kandidaten-Writes mit k Replay-Batches (echt/generativ); Distillation auf alten Logits. Gradienten nur im Schlaf.
- **E — Multi-Timescale-σ-Kaskade:** Drei σ-Stufen, u_fast > u_mid > u_slow; nur die langspeisende Stufe konsolidiert. Kosten: 3× State.
- **F — Wachstum & Merge:** Neue Erfahrung → neues Partikel-Subgraph/Zwillingsmodul, Merge nach Paper-Rezept; Pruning zwischen Zyklen über Weight-Atlas-Beitrags-Scores. BWT ≈ 0 konstruktionsbedingt; Preis: Kapazität.

## 4. Experimentierplattform

- **Modelle:** BDH 10M–50M (didaktisches Listing), Char-Level UTF-8, Regime wie Paper-Anhang (AdamW 1e-3, TBPTT 2048).
- **Domänen-Sequenz:** Phase 1 TinyStories-EN → Phase 2 Wikipedia-DE → Phase 3 Python-Code. Variante: Europarl-Sprachphasen als direkter Anschluss ans Paper.
- **Protokoll:** je Phase einfrieren → konsolidieren → ALLE Domänen evaluieren; ≥ 3 Seeds; gleiche Budgets.
- **Metriken:** ACC_avg, BWT, FWT **plus** Anteil geänderter Kanten, Domänen-Synapsen-Overlap, Spektraldrift je Layer (Weight-Atlas), Monosemantizitäts-Retention (Currency-Synapse-Probe).
- **Hardware:** RTX 4090 (≤ 50M Runs), Server für Sweeps.

## 5. Hypothesen

| ID | Aussage | Test | Falsifikation |
|----|---------|------|---------------|
| H1 | Exposuresignal sättigt bei T* ≈ c/ρ; Reset+Konsolidierung bei T* schlägt nie-Reset und immer-Reset | Perplexität über Expositionslänge; Sweep Zykluslänge W | Monotone Besserung ohne Reset oder gar kein Effekt |
| H2 | Eligibility-Gating hält ≥ 90 % Vorwärtsgewinn von A bei halbem Vergessen | B vs. A, gleiche Schreibbudgets | Gating killt Vorwärtsgewinn |
| H3 | Importanzschutz (C) schlägt uniformes Gating; Vorteil wächst mit sinkendem ρ | C vs. B, zwei Modellgrößen | Kein signifikanter Unterschied |
| H4 | Ohne Replay nur Within-Domain-Assoziation; Cross-Domain-Binding (DE↔EN) erst mit Schlaf-Replay | DE-Exposition auf EN-Modell ± Replay; Binding-Probes | Binding entsteht auch ohne Replay |
| H5 | Kaskaden-σ (E) bessere Konsolidierungsqualität bei gleicher Zykluslänge (SNR) | E vs. Single-σ | Fast-Traces genügen bereits |
| H6 | Wachstum&Merge: BWT ≈ 0 über ≥ 3 Phasen; Pruning −30 % Zubau bei < 1 % Verlust | Wiederholte Merges + Pruning | Merge-Qualität degradiert zyklisch |
| H7 | Schlafkosten skalieren mit potentierten Kanten, nicht mit n → Sleep/Wake-Ratio gebunden über 10M→100M | Kostenmessung über zwei Größenordnungen | Ratio wächst mit n |

**Priorität:** H1 → H2/H3 → **H4** → H5 → H6/H7.

## 6. Offene Designfragen

1. σ-Readout-Semantik für den Merge (additiv / Log-Raum / normierte Zähler) — Ledger ?01.
2. Vergessens-Phänotyp: Gewichtüberschreibung ODER Homöostase-Kollaps (Sparsity-Drift)? Beides monitoren.
3. Generatives Replay: Ist BDH gut samplbar? Fallback echter Datenpuffer.
4. Lizenz-Check pathwaycom/bdh (LICENSE.md) vor Ableitung.

## 7. Verbindungen

- **Pipeline-Task:** liefert Phase 0+1 dieser Roadmap mit (gleiche Plattform).
- **Weight Atlas:** Prä-/Post-Konsolidierungs-Fingerprints = Vergessensdetektion; Modulwahl für F.
- **J-Space:** dieses Dokument ist das Projektleger; H-Nummerierung sessions-stabil.

## 8. Paper-Anker

- Z. 373–379: Vertagung des lifelong-Transfers (unser Mandat)
- Z. 1584–1594: No-BPTT-Experiment (Totwasser → H4-Design)
- Z. 1629: Schwelle T ~ 1/ρ (H1-Formel, Schlaf-Zwang)
- Z. 1489 ff.: Model Merging (Grundlage F)
- Z. 1423 ff.: Monosemantic synapses / sparse activations (Grundlage B/C)
