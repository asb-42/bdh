# BDH Continual Learning Plan — Consolidating σ_l After Training Ends

**Version:** 1.0 (final) · **Date:** 2026-08-24 · **Language:** EN
**Supersedes:** `2026-08-23_bdh-continual-learning-plan_de_v01.md`, `_de_v02.md` (archived in `docs/plans/`)
**Platform:** fork `asb-42/bdh` — parity phases 1–6 complete (`eef5f7e..067fe9f`), MIT
**Validation basis:** RTX 4090 session 2026-08-23 (`docs/reports/2026-08-23_progress-report.md`, ledger ✓01–✓16)
**Process ledger:** `.jspace/WORKSPACE.md` (project root)

---

## 0. Mission

How do frozen weights stay neuroplastic after training ends — without catastrophic forgetting?

Concretely: design and test a consolidation operator

```
C: (σ_acc, G_frozen, optional replay budget R) → G_new
```

that turns accumulated synaptic fast-weight state σ_l into durable parameter change, with bounded backward transfer. This fulfills the promise the paper itself defers — *"we do not provide a direct answer as to how the brain actually handles this effect at longer timescales"* (L379) — and operationalizes the threshold the paper derives: state-to-weight relaxation bottlenecks on synaptic state capacity at `T ~ 1/ρ` (L1629).

## 1. Problem Formalization

Requirements:

1. **Forward transfer:** post-training experience becomes capability-effective.
2. **Backward transfer bounded:** BWT ≥ −ε on all previously seen domains.
3. **Cycleability:** repeatable indefinitely without drift (stability–plasticity over many sleep/wake cycles).
4. **Soft locality:** wake-phase rules local where possible; gradients permitted only during sleep.

**Architectural constraint (design principle, not metaphor):** σ usage signal and write rates collide at `T ≈ 1/ρ` (ρ ≈ 5 % sparsity → lower bound ≈ 20 tokens). Consolidation therefore cannot be a one-shot event; the architecture demands periodic cycles. We build a **sleep–wake architecture**, not a transfer mechanism.

## 2. Empirical Foundation (validated 2026-08-23, RTX 4090)

Six findings from the validation session directly shape this plan:

| # | Finding | Consequence |
|---|---------|-------------|
| F1 | No-BPTT costs only +0.0044 nats on monolingual LM (ctrl 1.1222 → 1.1266) | The paper's degradation lives in cross-domain alignment specifically. Any "learning" claim requires a cross-domain binding probe (→ INV-3); single-domain tests are blind to the known failure mode. |
| F2 | Eval/train protocol congruence is load-bearing: ctrl model degrades +1.75 nats under imposed state | Hard invariant INV-1. All H1–H7 measurements run under congruent protocols or are void. |
| F3 | Neuron-dim concat merge beats both parents without finetuning (1.1941 vs 1.2548/1.2486) | Mechanism F has empirical wind at its back; structural CL is demonstrated, not hypothetical. |
| F4 | ALiBi damping is load-bearing for BDHLinear (undamped collapses to val 2.52; slope 0.05 sweet spot) | Per-edge damping (shared fork follow-up) is a stability prerequisite for Mechanism E, not cosmetics. |
| F5 | Step-efficiency ≠ compute-efficiency: GPT-25M wins at matched compute (1.0911 vs 1.1177; 5.7× step ratio) | Honest reporting: CL gains are stated against compute-matched baselines, never step-matched alone. |
| F6 | bdh-prime: −0.027 nats quality at 60× wall-clock (autograd retains ρ per token per layer; OOM at L=6, fits at L=2) | Cost discipline INV-2. Note: the 60× is an implementation artifact of the eager gated scan, not an architectural property of gating — but the discipline it teaches is general. |

## 3. Invariants (hard rules; violations void results)

- **INV-1 Protocol congruence.** Eval protocol must match training protocol (stateful-eval iff carry-trained; identical stream/block semantics). Rationale: F2.
- **INV-2 Cost gate.** Any mechanism whose steady-state per-step cost exceeds 10× the linear baseline at matched parameters is classified research-only until a fused/compiled implementation exists. Rationale: F6.
- **INV-3 Binding probes.** Every hypothesis claiming learning (not mere association) must include a cross-domain binding measurement. Rationale: F1.
- **INV-4 Coverage-declared checkpoints.** Every checkpoint states what was verified, on which data, under which protocol. A number without coverage is a mood, not a result.
- **INV-5 Surprise handling.** An unexpected result is treated as a measurement-artifact hypothesis until ruled out (the §5.1→§5.2 discipline: the cold-eval anomaly was resolved by protocol correction, not accepted as a finding).

## 4. BDH-Specific Levers

| # | Property | Use for CL | Status |
|---|----------|------------|--------|
| 1 | 1:1 parameter-to-state ratio (both O(n²)) | σ and G are shape-equal → consolidation is a residual write, no projection problem | structural |
| 2 | Monosemantic synapses (<100M scale) | Write-gates can audit WHAT they change before changing it | reproduced (France↔Germany synapse) |
| 3 | Sparse positive activations (~97.4 % zeros, paper-matching) | Cheap local importance statistics; near-tree gradient DAGs for T < 1/ρ | reproduced |
| 4 | Zero-shot merging (ES/FR/PT in paper; disjoint-half here) | Structural CL without joint training | reproduced (F3) |
| 5 | Per-edge damping u(i,j) | Native multi-timescale cascade (STP→LTP mapping) | partial (uniform ALiBi slope); per-edge is the shared fork follow-up |

## 5. Mechanisms (ordered by invasiveness)

- **A — Naive periodic merge (floor baseline).** `G ← G + λ·ΣΔσ` every W tokens. Expected: measurable forgetting. Purpose: quantify the floor, validate the measurement apparatus. Infrastructure ready today (merge CLI, analyze tooling, GPU memory guards).
- **B — Eligibility gating.** Write only edges with ≥ K potentiations across ≥ D contexts; rate ∝ potentiation count. Purely local, label-free.
- **C — BDH-EWC.** Per-edge importance F_ij from co-activation statistics; step size η/(F_ij+ε). Local elastic weight consolidation without backprop.
- **D — Sleep with replay.** Consolidation mixes candidate writes with k replay batches (real or generative); distillation on old logits protects old capabilities. Gradients only during sleep. Prerequisite: INV-1 — without protocol congruence, D is unmeasurable.
- **E — Multi-timescale σ cascade.** Three σ stages with u_fast > u_mid > u_slow; only the slow-fed stage consolidates. Cost: 3× state (memory, not compute). Requires per-edge damping (F4). Not threatened by prime's 60× pathology — that is an autograd-retention artifact of the gated scan; a cascade adds inference state, not retained graph nodes. Still subject to INV-2 like everything else.
- **F — Growth & merge.** New experience → new particle subgraph / twin module; merge per paper recipe (neuron-dim concat); prune between cycles using Weight Atlas contribution scores. BWT ≈ 0 by construction; price: capacity growth. Empirical wind: F3.

## 6. Experimental Platform

- **Code:** fork `asb-42/bdh` (pipeline/ structure; models `bdh.py`, `bdh_linear.py`, `bdh_prime.py`; parity phases 1–6 merged; MIT).
- **Scales:** 25M standard (n_embd=256, n_head=8, n_layer=6, mult=128, block=512); 0.33M pilot (CPU-capable, proven by the prime/linear comparison).
- **Hardware:** RTX 4090 24 GB primary; local CPU pilots for harness bring-up.
- **Data:** wikitext-2 for harness shakedown. CL-sequence corpus decision open (?03):
  - *Europarl language phases* (EN→DE→ES …): direct paper continuity, native cross-lingual binding for H4/INV-3. **Recommended for the H-series.**
  - *TinyStories → WikiDE → Python* (domain diversity): generality beyond languages. **Recommended as secondary check.**
- **Protocol:** per phase: freeze → consolidate → evaluate ALL domains; ≥ 3 seeds; compute-matched budgets (F5).

## 7. Hypotheses

| ID | Claim | Test | Falsification |
|----|-------|------|---------------|
| H1 | Exposure signal saturates at T* ≈ c/ρ; reset+consolidation at T* beats never-reset and always-reset | Perplexity over exposure length; sweep cycle length W | Monotone improvement without reset, or no effect |
| H2 | Eligibility gating retains ≥ 90 % of A's forward gain at half the forgetting | B vs A, equal write budgets | Gating kills forward gain |
| H3 | Importance protection (C) beats uniform gating; advantage grows as ρ drops | C vs B, two model scales | No significant difference |
| H4 | Without replay only within-domain association forms; cross-domain binding (DE↔EN) requires sleep-replay | DE exposure on EN model ± replay; binding probes mandatory (INV-3); `--no-bptt` arm as control (F1) | Binding emerges without replay |
| H5 | Cascade σ (E) consolidates better at equal cycle length (SNR) | E vs single-σ | Fast traces suffice |
| H6 | Growth & merge: BWT ≈ 0 over ≥ 3 phases; pruning removes 30 % growth at < 1 % loss | Repeated merges + pruning | Merge quality degrades cyclically |
| H7 | Sleep cost scales with potentiated edge count, not n → sleep/wake ratio bound holds 10M→100M | Cost measurement across two orders of magnitude | Ratio grows with n |

**Priority with gate:** H1 → H2/H3 → **[Gate: H4 only after green H2+H3]** → H4 → H5 → H6/H7.
Rationale: an H4 null-result on a broken B/C vehicle is confounded ("replay unnecessary" vs "consolidation writes nothing"). H4 remains the thesis centerpiece, not the first experiment.

## 8. Measurement Suite

Standard CL metrics: ACC_avg, BWT, FWT — always compute-matched (F5).

BDH-native diagnostics:

- Edge-change fraction per consolidation (how much did C actually write?)
- Domain synapse overlap (do domains share synapses? predicts interference)
- Spectral drift per layer (Weight Atlas fingerprints, pre/post consolidation — objective forgetting detection)
- Monosemanticity retention (currency-synapse-style probes via `analyze.py synapse_trace`)
- Sparsity drift (ρ stability — homoeostasis collapse is a distinct forgetting phenotype from weight overwrite; monitor both)
- Per-position warm/cold context curves (`scripts/position_curves.py`) — does consolidated state still carry useful context?

## 9. Open Design Questions

- **?01** σ readout semantics for the merge: additive, log-space, or normalized potentiation counters? — settle by studying the rule tables (paper.tex, equations-of-reasoning section), then prototype.
- **?03** CL-sequence corpora: Europarl phases vs domain-diversity sequence (recommendation in §6).
- **?04** Execution venue: SSH access to the 4090 machine vs local CPU pilots at 0.33M.

## 10. Execution Order

1. **Phase 0 (now):** pilot harness locally at 0.33M; implement Baseline A; verify the measurement suite end-to-end on wikitext-2.
2. **Phase 1:** H1 saturation curve + cycle-length sweep (cheapest; calibrates everything).
3. **Phase 2:** B vs C comparison at two scales.
4. **Gate review:** green H2+H3 required.
5. **Phase 3:** H4 binding study with replay arms (Europarl if ?03 resolves accordingly).
6. **Phase 4:** E cascade (blocked on per-edge damping work — shared fork follow-up).
7. **Phase 5:** F growth & merge cycles with Weight Atlas pruning.
8. **Parallel:** H7 cost-scaling measurements ride along every phase.

## 11. Connections

- **Weight Atlas:** pre/post-consolidation topographic fingerprints = objective forgetting detection; contribution scores select prune candidates in F. The terrain maps get their diagnostic purpose.
- **J-Space:** this document anchors the project ledger; H-numbering is stable across sessions and harnesses (proven: de_v01/v02 migrated across agent contexts intact).
- **Paper anchors:** L373–379 (deferred lifelong transfer — our mandate) · L1584–1594 (no-BPTT experiment — dead end mapped, refined by F1) · L1629 (T~1/ρ threshold — H1 formula, sleep necessity) · L1489ff (model merging — basis of F) · L1423ff (monosemantic synapses / sparse activations — basis of B/C).

## 12. Provenance

- **v0.1, v0.2 (German):** authored by ox-alpha (Agent Zero), reviewed by the user; archived in `docs/plans/`.
- **v1.0 (this file, English, final):** incorporates the user review (gate formalization, license closure), the Claude Sonnet 5 review (INV-1 elevation, cost-gate discipline, start-with-A confirmation), and validated findings F1–F6 from the 2026-08-23 RTX 4090 session.
- **One deliberate divergence from the Sonnet-5 review:** its warning that prime's 60× cost threatens Mechanism E is corrected — the cost is an autograd-retention artifact of the eager gated scan, while E adds inference state (3× memory), not retained graph nodes. The discipline (INV-2) is kept; the misattribution is not.
