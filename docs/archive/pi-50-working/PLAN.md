# BDH / BDH-CL review — working plan

Session goal: build real understanding of (1) the BDH architecture as published by Pathway and
(2) our fork `bdh-cl` + its manuscript, reviewed adversarially. Long-running work on a 14 tok/s
local model, so **notes are the product, context is disposable**.

## Sources (staged locally, do not re-download)
| what | where | notes |
|---|---|---|
| Pathway paper, *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain* | `~/bdh-review/paper-src/paper.tex` | 35 583 words. **arXiv 2509.26507** — NOT 2509.2507 (that is an unrelated RV-exoplanet paper; user's link had a dropped digit) |
| our fork `bdh-cl` | `~/bdh-review/bdh` @ `cb72631` | 24 py files / 3063 LOC; docs/{papers,reports,reviews,notes,plans,tasks,archive} |
| our manuscript | `~/bdh-review/bdh/docs/papers/cl-bdh-manuscript.tex` | 5002 words, rev 2 (2026-08-27), author "ox-alpha (autonomous research agent)" |

## Method / resilience rules (important)
1. Read in **section-sized chunks by line range** (maps below). Never read the whole 35k-word file at once.
2. After each chunk: append notes to the target file, then tick the ledger **in the same write**. If the
   model or pi dies, resume = `cat PLAN.md` + `tail` of the active note file. Do not re-derive from scratch.
3. Append-only notes; never rewrite a note file wholesale (protects llama.cpp prompt cache and history).
4. Annotation discipline in every note: `[FACT]` (traceable to source, give `file:line`),
   `[MATH]` (derivation I performed myself), `[INFERENCE]`, `[GUESS]`, plus confidence `c/10`.
   This is deliberate: the meta-question is whether the model *understands* the math, so fluent
   sentences are worthless without provenance and self-flagged uncertainty.
5. Do not modify `~/bdh-review/bdh` working tree (review artifacts live in `~/bdh-review/notes/`).

## Reading order — Task 1: Dragon Hatchling (`paper-src/paper.tex`)
- [x] L211–395  Intro, motivation, modus-ponens intuition, contribution, notation → notes §1 + issues I1–I6
- [ ] L396–699  §2 BDH as local distributed graph dynamics; edge-reweighting (the core equations);
                attention as micro-inductive bias; oscillator toy model; brain models
- [ ] L700–1002 §3 BDH-GPU: notation, state-space definition (**most important for the fork**),
                particle-system reading, BDH-GPU ↔ BDH parameter/state-size preservation
- [ ] L1003–1085 §4 Implementation + scaling laws; comparison to GPT2-like / other architectures
- [ ] L1086–1334 §5 Modularity + scale-free structure; ReLU-lowrank block and signal propagation
- [ ] L1335–1486 §6 Linear attention, sparse positive activations, monosemantic synapses
- [ ] L1487–1645 §7 Experiments (model merging by concatenation! no-BPTT training) + conclusions
- [ ] L1755–2100 Appendices: formal claims/proofs, protocol equivalence, linear-attention claim,
                desirable properties, PyTorch listing (compare against `bdh/bdh*.py`)

## Reading order — Task 2: our manuscript (`cl-bdh-manuscript.tex`), peer-review mode
- [ ] L31–60   Abstract — record every quantitative claim as a checkable item
- [ ] L61–168  Intro + Setup (architecture primer → **check fidelity against paper §2–3**; protocol;
               state- vs output-level isolation definitions — where the whole argument hinges)
- [ ] L169–214 §3 The phenomenon: four negative results (frozen-weights erosion claim is load-bearing)
- [ ] L215–290 §4 Theory: when does selection give exact isolation? (verify theorem statements + proofs)
- [ ] L291–404 §5 Recipe branch by branch (grow / select / soft-regime budget / merge-prune-replay /
               replay-only / decision rule)
- [ ] L405–460 §6 Accumulation: Arm R vs Arm G across many phases
- [ ] L461–520 §7 Secondary observations, §8 Limitations, §9 Conclusion
- [ ] L635–800 Appendix: separability facts, proofs of Thm dissociation / Thm criterion / Lemma zf /
               Prop soft / Prop amp, measurement remark, prior-art triage, reproducibility, negative register

## Review axes (Task 2 output)
A. **Mathematical correctness** — theorem statements vs proofs; hidden assumptions; the "soft gates can
   never be exact" counterexample class; reachable-set invariance criterion.
B. **Claim ↔ evidence traceability** — every number in abstract/tables must map to an artifact in
   `docs/reports/` or code; flag orphans, seed/variance absence, and effect sizes stated without baselines.
C. **Internal consistency** — notation drift between sections, and between manuscript and the Pathway
   paper's definitions (fork drift is the classic failure mode).
D. **Novelty / positioning** — vs EWC/progressive nets/MoE routing/adapter growth; prior-art triage appendix.
E. **Overclaiming & presentation** — what a hostile reviewer would attack first.

## Notes files
- `notes/10-paper-bdh.md` — Task 1, per section
- `notes/20-manuscript-review.md` — Task 2 review findings by axis
- `notes/30-math-checks.md` — my own independent derivations (the actual comprehension test)
- `notes/40-claims-vs-evidence.md` — claim → artifact → verdict table
- `notes/50-open-questions.md` — things I cannot resolve from sources

## Scope agreed with user (2026-08-29 ~18:15) — these are BINDING
- **Priority: is the science sound**, within reasonable boundaries. Project knowledge gained en route is a
  byproduct and becomes the basis for a later "how do we proceed" discussion.
- Posture: **critical, adversarial if needed, but constructive**. Say plainly when a claim looks wrong,
  together with what would settle it.
- Scope = **the two papers**. `docs/reports/` may be consulted when in doubt. **Raw-data / run-log
  verification is out of scope.** No code audit unless something specific demands investigation.
- Independence: read both papers first and record issues; only afterwards peek at `docs/reviews/` and
  `docs/notes/`. Keep my read as independent as possible.
- Terminology: **Pathway's paper is authoritative** (user notes some of it is debatable; this is still
  mostly a fork). Flag terminology drift, but do not rename their terms in my own writing without noting it.
- Artifacts stay **local** (`~/bdh-review/notes/`) for now; publishing into the repo happens once A2A with
  Quinn is working.
- Confidence ledger: approved, keep it.
- Note on provenance: all quoted numbers come from files on disk with `file:line` (verified e.g.
  `README.md:27-30`, `cl-bdh-manuscript.tex:48,315,360`). Nothing invented.

## Ledger
- 2026-08-29 ~18:05 workspace created; both sources staged; arXiv ID corrected.
- ~18:15 scope agreed with user (see BINDING above).
- ~18:20 Task 1 chunk 1 done (L211–400) → `notes/10-paper-bdh.md`. Next: L396–699 (§2 BDH graph formalism, core equations).
- Task 1: DONE (§1–§8 read; appendix proofs checked; notes complete)
- Task 2: DONE — REVIEW.md written (majors M1–M6, 13 minors, 8 author questions, verification log); notes/20-manuscript-review.md = 5 review rounds w/ hand-checked math; notes/40-claims-vs-evidence.md = 28-row claim ledger.
- Independence protocol kept: docs/reviews/ and docs/notes/ NOT read before writing REVIEW.md. Next: skim them, reconcile (expect overlap on M1/M3 if A0 self-reviewed).
