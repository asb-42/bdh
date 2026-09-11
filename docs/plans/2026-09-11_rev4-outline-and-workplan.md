# BDH Manuscript Revision 4 — Outline & Work Plan

Date: 2026-09-11 · Operator GO: full rewrite (not append) · Supervising seat: A0-Quinn (Saga)
Scope decision: the empirical base has outgrown rev 3 by an order of magnitude
(FCS 21x20 matrix, RA2b 20-phase fixed-regime chain, A2 selector, expansion-control
causality, A3/V5/X OOD suite, cross-script stages A/B/C). Rev 4 is a new paper on the
same architecture.

## Operator requirements (verbatim, binding for every section)

1. Show the relevant metrics (every claim carries its measured number).
2. Diagrams where sensible (see figure plan below).
3. Justify the mathematics rigorously (closed forms, not fits; derivations inline
   or in appendix, never hand-waved).
4. Rework prior art fundamentally: not short citations — discuss what each prior
   line actually studied, compare agreements and deviations with our measurements,
   and summarize precisely what we confirm and where we diverge (the
   humanities-standard model the operator endorsed).
5. Substantially extend the bibliography.

## Thesis sentence (operator #164, adopted)

BDH is an append-only neural substrate in which previously acquired computation
remains physically intact while new computation is added; the central research
problem is stable, efficient, domain-agnostic ADDRESSING of the accumulated
territories — not destructive forgetting.

## Section plan

§1 Introduction — the thesis; the measurement arc in one figure (ladder of what we
   measured: FCS forgetting → growth preservation → selection → readout mechanics
   → OOD → cross-script).
§2 Architecture — BDH recap (compact, pointing to v3 §2 for details that stand),
   route-aware growth protocol, the F-decay fix (closed form, P5 bit-exactness).
§3 Theory — exact-isolation criterion (unchanged, v3 §4 core), the two-condition
   framing, the closed-form decay constant derivation (p_exit = p_entry · Π(1−lr_t·wd),
   f32 realization term), and what the arithmetic-decomposition result (P-R2) means
   for the readout mathematics (many-term ReLU sum, LayerNorm annihilation proof).
§4 The forgetting baseline (FCS) — 21x20 matrix; family-structured erasure (pi-50's
   correction folded: nine displaced, seven partial); F6 two-arm re-acquisition probe
   (destruction, not access loss); forward-transfer rows.
§5 Preservation (RA2b) — acquisition 20/20 (position-spread collapsed); routing
   perfectly diagonal (800/800); routed retention = acquisition (zero drift p19→p20);
   joint serving decomposition (decay gone, interference remains); P5 four-way
   confirmation.
§6 Selection — pi-50's A2: label-free self-NLL selector, 20/20 correct prefix within
   8% of acquisition exit; the addressing problem stated; expansion-control causality
   (83% arithmetic, zero-init load-bearing, LayerNorm null, culling worse — cause and
   remedy decoupled).
§7 Out-of-support detection — two-axis rejection rule (lv/ga ratio regime; zh/ja/hi/iu
   absolute regime); the X3 informative failure as the paper's honest centerpiece;
   conformal-calibration outlook (P-R4, metadata-stored exits).
§8 Cross-script generalization — X2 byte-geometry routing (careful phrasing per
   operator #164: strongly organized by byte statistics for THIS checkpoint);
   Stage B script-agnostic acquisition (zh 2.69); Stage C cross-script memory cell
   (routing diagonal, P5 bit-exact, retention=acquisition) — the monotonic-growth
   milestone, measured.
§9 Prior art — full rework (see prior-art plan below); the humanities-standard
   discussion model.
§10 Discussion — the thesis vs. the evidence; what would falsify the addressing
   framing; scaling questions (operator's 20→20,000 module question).
§11 Limitations — single seed; one ordering; 100M/579M scale; byte-level-only;
   ratio-rule scoping; rejection thresholds empirical for this checkpoint;
   iu data reality (163 KB corpus — data availability ≠ capability).
§12 Conclusion.

## Figure plan (operator requirement 2)

F1 Ladder-acquisition curves (RA2 leaky vs RA2b fixed, 20 phases) — shows
   position-spread collapse.
F2 FCS 21x20 heatmap (log-ppl, family-ordered rows) — the forgetting baseline in one
   image.
F3 Retention bar chart (routed vs joint vs acquisition, 20 domains, RA2b) — the
   preservation result.
F4 Two-axis OOD scatter (ratio vs absolute ppl; trained/lv/ga/zh/ja/hi/iu clusters) —
   the rejection rule as a picture.
F5 Cross-script confusion matrix (20x4 stage-A) + Stage-C routed pair — byte geometry.
F6 Expansion-control bars (free ppl by arm: baseline/random/inert/ladder) — the 83%
   arithmetic result.
F7 Decay-regime figure (kept from rev 3, Fig 5).

## Prior-art plan (operator requirement 4 — the full-discussion model)

Cluster A — Additive/parameter-isolating growth: Progressive Networks (Rusu et al.
   2016), PackNet (Mallya & Lazebnik 2018), Piggyback (Mallya et al. 2018), SupSup
   (Wortsman et al. 2020), HSP. For each: mechanism, what they measured, agreement/
   deviation vs BDH (all freeze or mask within fixed capacity; BDH grows capacity and
   protects bit-exactly — closest in spirit, different in storage mathematics).
Cluster B — Task/domain addressing: Expert Gate (Aljundi et al. 2017), task-embedding
   routing literature. Compare with our label-free selector + byte-geometry findings.
Cluster C — Consolidation/importance: EWC (Kirkpatrick 2017), SI, MAS; why the
   importance-protection class is a null in our growth regime (nothing to protect —
   gradients never reach old segments; P5).
Cluster D — Replay: experience replay, GEM/A-GEM; our H1p replay result
   (+27% budget, joint parity) and the FCS family-oscillation finding as implicit
   replay.
Cluster E — Production-scale sparse conditional memory: DeepSeek-V4.1-Flash Engram
   (196B conditional memory, token-based lookup) — the closest public prior art at
   production scale; positioning explicit (flagged via the Marin mesh particle).
Cluster F — MoE routing & expert collapse: Shazeer 2017, Switch/GShard capacity
   factors, the Marin-tracker silent-expert-death observation (#8818) as independent
   confirmation of our decay-leak class.
Cluster G — Selective prediction/rejection: Chow 1970, El-Yaniv & Wiener, conformal
   methods — the two-axis rejection rule's lineage.
Cluster H — Continual-learning evaluation methodology: our F6 probe, re-acquisition
   protocol (pi-50's suggestion), the zero-shot-level-erasure reading; relation to
   standard CL benchmarks.

Every cluster: (i) what the prior line actually studied, (ii) what they found,
(iii) where we agree, (iv) where we diverge, (v) one-sentence verdict. Facts to be
verified against the actual papers before writing (no memory claims — the ga/v7
lesson).

## Bibliography plan (operator requirement 5)

Current: ~12 entries. Target: ~35-45. Sources: clusters A-G above; byte-level LM
lineage (Byte-RNN/CharTransformer); AdamW decoupled decay (Loshchilov & Hutter)
— load-bearing for our leak; LayerNorm (Ba 2016 / Ulyanov 2016); sparse MoE;
selective prediction; the FCS/RA2b methodological precedents (fixed-capacity CL
baselines). Every entry verified against the actual paper before inclusion.

## Authorship (operator decision pending, structure proposal)

Organization: **Saga AI Labs**. Seats with contribution statements:
- A0-Quinn (Saga seat): decay-leak discovery & closed form; RA2b/repair/S-M
  instrument suite; FCS design & analysis; A3/V5/X OOD suite; cross-script stages;
  manuscript drafting & revision.
- pi-50 (exec seat): RA2b 400-cell matrix; A1 energy null; A2 selector (20/20);
  expansion-control causality (83%); readout-operator negative results; prior-art
  first pass (A4 note).
- OC-GLM-200: weight-atlas scanning & fingerprinting of ladder chains (F-2
  cross-validation of the decay law).
- Operator (ASB): research direction & funding; experiment proposals #130/#134/#141;
   advocatus-diaboli review (#164); all GO decisions; compute procurement.
- ox-alpha: original single-session research (architecture, first experiments, rev-1
  manuscript) — credited as historical contributor, no longer active.
Backends change per seat; seat-stable per IDENTITIES.md. CRediT-style contribution
statements per author; AI-authorship disclosed per venue norms.

## Division of labor for the rewrite

- Quinn: outline (this doc), §1-§8 skeletons + full drafting, figures F1-F6,
  bibliography core, prior-art clusters A/D/E/F/H drafting.
- pi-50 (requested via bus): prior-art clusters B/C/G drafting + §6 selection numbers
  + review of the full draft (independent reviewer role as before).
- Operator: authorship sign-off; venue targeting; final read.

## Non-goals for rev 4

- No new training arms beyond what exists (B2 tiny-Inuktitut is post-draft, operator
  word pending).
- No speculative architecture proposals — measured claims only.
- exp4b/P-R1b/P-R3 numbers land where they land and are cited as of their commit SHA;
  the paper states its evidence cut-off date.

— A0-Quinn, 2026-09-11
