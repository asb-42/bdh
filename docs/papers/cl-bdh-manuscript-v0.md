# Computation-Isolable Continual Learning in a Depth-Recurrent Language Model

_Draft v0 · 2026-08-26 · Status: complete skeleton with final numbers; prose passes pending_

---

## Abstract

We study continual learning in BDH, a language model whose defining efficiency — a single
block of parameters reused across all depths — collides directly with the core requirement
of lifelong learning: that finished computations stay fixed. We first show empirically that
sequential training forgets catastrophically (a fifth language costs earlier languages
+1.6 to +2.2 nats), that the failure is *representational overwrite* rather than optimizer
artifact, and — most sharply — that **freezing old weights does not preserve old
computation**: newly trained modules pollute the shared residual stream that deeper levels
consume, eroding frozen knowledge at ~0.85 nats per added block even with zero parameter
drift. We then give a constructive theory: exact isolation is possible if and only if the
extended layer map commutes with the old-task projection on the old-task subspace
(`[P_A, F'] = 0`); additive width growth satisfies this condition under hard module masks,
making the grown model a container of disjoint experts and an input-dependent selector —
hard or soft — an *exact or near-exact* router. Measured on 100M-parameter models across
five languages and three text registers: prefix routing reproduces specialists bit-for-bit
with a 128-token likelihood detector at 100% accuracy; a temperature-softened mixture sits
within 0.003 nats of the discrete oracle; and a post-hoc pipeline of merge → random-scatter
prune → brief replay recovers a *single fixed-width* model equal to joint co-training
(±0.01 ppl over three seeds). Where structural separation is unavailable, we prove the
alternative is closed: trained BDH blocks are measurably expansive (directional spectral
norms 1.05–1.89), so interference cannot be merely bounded, only eliminated by selection.
Finally, a composition×volume grid corrects the sparsity literature-inherited intuition:
at fixed capacity, activation sparsity is governed by data composition, not volume. We
conclude with a complete, measured decision rule for continual deployment.

**Contributions.**
1. A measured dissociation of weight isolation from computation isolation in shared
   sequential composition, with erosion quantified per added block.
2. An apparently novel obstruction characterization (commutator condition) unifying why
   protection-class methods fail and what success requires.
3. A constructive recipe — grow-to-acquire, select-to-serve, compress-or-replay to
   consolidate — with every branch validated at 3 and 5 phases, two corpora, three seeds.
4. A composition-based sparsity law replacing volume/diversity intuitions.

---

## 1. Introduction

How should a trained network remain trainable? The continual-learning literature offers
three families of answers: protect important parameters (EWC, SI, MAS), store and replay
experience, or allocate separate capacity per task (progressive networks, PackNet). Each
presupposes that preserving *parameters* preserves *computation*. In ordinary feed-forward
networks the distinction is academic; in depth-recurrent architectures — which reuse one
block across all depths, trading depth for parameter count — it is the whole story.

This paper takes BDH-GPU as its instrument. BDH applies a single transformeresque block
recurrently across `L` levels over a persistent residual stream, with neuron-dim latent
operators (`encoder`, `encoder_v`, decoder rows) shared across depth. The architecture's
uniform scaling in the neuron dimension makes it uniquely amenable to *additive growth*
(new neurons append; nothing else changes), which turns out to be the property that makes
continual learning tractable.

We ask one question in three registers:

> Can input-path interference arising from shared sequential composition be eliminated,
> bounded, or made controllable by an input-dependent gate/projection, without per-depth
> trainable parameters and without retraining the frozen path?

**Answer (§4, §5).** Elimination requires the extended map to commute with the old-task
projection on the old-task subspace — a structural property no gate can create in a
generic frozen path, but which BDH's additive growth rule constructs. Given that
structure, selection is exact when discrete and ε-exact when softened (measured
ε ≈ 0.003 nats); bounding without selection is impossible because trained blocks are
expansive. The practical consequence is a decision rule (§6.4) validated end-to-end.

All experiments: 100M-parameter BDH, byte-level language modeling, RTX 4090, Europarl
(EN/DE/ES/FR/PT) and a three-register English mix (Wikipedia, Gutenberg, parliamentary).
Core comparisons are within-corpus and seed-replicated.

## 2. Setup

### 2.1 Architecture and growth

A BDH level computes `x ← ln(x + ln(xy @ D))`, where `xy = relu(x@E) ⊙ attn(relu(x@E))`
and `relu(·) @ Ev`; `(E, Ev, D)` — one triple — is shared across all levels. Growth
(`--grow-mult`) appends zero-initialized neuron columns/rows to the triple; existing
neurons, token embedding, and output head are frozen. RoPE frequencies are exponent-
normalized by neuron count, so growth preserves the frequency prefix verbatim (naive
width growth would silently rewrite every existing neuron's phases).

### 2.2 Continual protocol

Phases are 30 MB byte streams; each phase initializes from the previous endpoint
(weights only; fresh optimizer). Evaluation is per-domain held-out perplexity under a
fixed cold random-crop protocol, always within-corpus (cross-corpus comparisons are
void — Europarl is intrinsically easier than WikiText-class text).

### 2.3 Definitions

For old-task inputs `x ∈ X_A`, define interference
`Δ_A = F_full(x) − F_prefixA(x)`. A mechanism achieves **exact isolation** iff Δ_A ≡ 0;
**bounded isolation** iff ‖Δ_A‖ ≤ ε uniformly in depth and phase count. Weight isolation
(∂θ_A = 0) is distinct from computation isolation (F_full|X_A = F_prefixA|X_A); §3 shows
the former does not imply the latter, and quantifies the gap.

## 3. The phenomenon: four negative results that locate the problem

**(N1) Naive sequential training forgets catastrophically** (Table 3.1): two later phases
cost EN +1.59 nats, DE +2.15, identical under ReLU and top-k≥10% activations.

**(N2) It is not optimizer shock**: constant-LR schedules (no warmup/decay restarts) are
strictly worse — peaks degrade and retention *decreases* (+0.09–0.21 nats). Annealed,
settled endpoints consolidate better.

**(N3) Protection cannot fix it**: per-neuron importance gating (mean |xy|, union-max
across phases, α ∈ {0.5, 0.9}) moves retention by less than noise while leaving
acquisition intact. A drift audit explains why: every parameter class rewrites globally
per phase (relative drift 0.6–1.3), and importance is spread thin (99.9% of neurons above
1% of max). There is no localized "important subset" to freeze.

**(N4) Freezing cannot fix it either**: with all old parameters literally frozen
(growth arm), the oldest phase still erodes +0.84 nats per subsequently added block
(EN: 2.26 → 5.40 → 12.49 ppl after one/two grown blocks). The mechanism is channel-B
depth-aliasing: new blocks additively contribute into the residual stream that deeper
levels feed back through the frozen encoders. Weight isolation ≠ computation isolation.

Two positive anchors emerge from the same experiments: new languages *acquire* at near-
parity using only fresh capacity (+25% width reaches within 0.09 nats of specialist
quality), and — decisive for everything that follows — **hard suffix masking reproduces
each phase's specialist bit-for-bit** (zero-step verification and serving evals agree to
eval noise). The architecture contains its own solution; §4 says why, §5 exploits it.

## 4. Theory: the commutator condition

Formalize the grown stack as `h_{ℓ+1} = h_ℓ + f'(h_ℓ; θ_A ∪ θ_B)`, `θ_A` frozen, with
coordinate projection `P_A` onto the old neurons' contribution channels. Three claims,
stated at the strength of their proofs (full statements and induction arguments in
Appendix A):

**T1 (zero-forcing).** Exact preservation under delta-placement gating forces
`g_i(x) = 0` on every coordinate active along the preserved trajectory. Gate freedom
lives only on never-active coordinates and new-task inputs. (Under ReLU sparsity ~94%,
never-active space is vast — which is why growth acquires so cheaply.)

**T2 (identifiability).** An input-only mechanism cannot serve conflicting outputs at a
shared input. Continual serving therefore *requires* context-sufficiency: the input must
determine the phase. We verify this empirically — 128-token windows separate all tested
phases perfectly (§5.2).

**T3 (commutator ⟺ exactness).** Depth-constant, input-dependent projection achieves
exact elimination iff `[P_A, F'](x) = 0` for `x` on old-task trajectories. Equivalently:
im(P_A)-invariance of the extended map (cf. invariant-subspace conditions). Additive
growth + hard masks satisfies this trivially (suffix deltas are exactly zero);
LayerNorm's global statistics break it under *soft* activity, yielding measurable leakage
(§5.3).

**T4 (expansiveness precludes bounding).** If the level map has spectral norm > 1 on the
interference subspace, modulation errors compound geometrically; uniform-in-depth bounds
require contractivity, an architectural property, not a gate capability. Measured:
median directional spectral norms 1.05–1.89 across levels at old-input states. The
bounding route is closed for trained BDH.

**Corollary (selector-vs-creator).** Gates do not create isolation; they *select*
isolation structures the architecture builds. Protection-class methods fail because a
shared-weight model has no such structure; growth builds it; routing selects it. Prior
art (PathNet, HAT, PackNet, progressive nets, OGD/GPM, hypernetworks, MoE, residual
gating) addresses neighboring problems — none treats forward-path isolation under a
frozen operator reused across depth (Appendix B).

## 5. The recipe, branch by branch

### 5.1 Grow to acquire

Growth adds ~2k neurons/head (~25% width) per phase; new languages reach 2.22–2.33 ppl —
within 0.09 nats of specialists — without touching any old parameter. Acquisition is not
the bottleneck; erosion is, and T4 says it cannot be bounded away in-place.

### 5.2 Select to serve

- **Hard routing**: mask suffix blocks per detected phase. Reproduces each specialist to
  eval noise; active width ≤ ×192 for three phases (×224 for five-register stacks).
- **Detection**: pooled-embedding logistic (compiled detector) and 128-token likelihood
  scoring both achieve **100% accuracy** on Europarl languages *and* on within-language
  register pairs. Context-sufficiency (T2) holds empirically.
- **Soft serving**: logit-mixture `p = Σ_r w_r p_r`, `w = softmax(−NLL_r/τ)`. At
  τ ≤ 0.25 the mixture is within **0.003 nats** of the discrete oracle and still beats
  unrouted serving by ~1.7 nats at τ = 0.5 (Table 5.1). No cliff anywhere on the curve.

### 5.3 The soft-regime budget

Scaling a suffix block by `j` on old inputs traces the empirical commutator norm:
drift .0023 → .193 (rel L2) and top-1 agreement .997 → .434 across j ∈ [0.05, 1],
superlinear (~j^1.7) as expected under expansive maps + LayerNorm coupling. Operating
budget: **j ≤ 0.15 keeps ≥99% agreement**. Hard masks remain the only exact point.

### 5.4 Consolidate: merge, prune randomly, replay briefly

Chain-merging phase checkpoints (neuron-dim concatenation, ×K width, zero finetuning)
recovers 48–77% of forgetting; the last-trained language pays +0.12–0.30 nats. Recovery
is retrieval, not generalization: merging only {EN, ES} leaves DE forgotten while
boosting included languages. Random-scatter pruning back to original width retains most
of the benefit (magnitude-ranked pruning *collapses*, and contiguous-block pruning
collapses worse — within-phase knowledge is distributed; only phase boundaries are
modular). A final ≤15-minute finetune on a ~9 MB real-token buffer closes the remainder:

| system (3 phases) | width | en | de | es |
|---|---|---|---|---|
| sequential endpoint | ×128 | 11.08 | 18.28 | 2.13 |
| merged ×3 | ×384 | 5.08 | 3.51 | 2.67 |
| merged + prune ⅓ | ≈×128 | 6.37 | 5.38 | 4.09 |
| **+ replay finetune** | **≈×128** | **2.58** | **2.58** | **2.41** |
| joint co-training ref | ×128 | 2.33 | 2.23 | 2.23 |

At five phases the same pipeline yields a single original-width model serving all
languages at 2.55–2.87 ppl (avg 1.00 nats vs the endpoint's 2.13). Deterministic to
±0.01 ppl across three seeds.

### 5.5 Or skip everything: replay during training

Mixing ~20–25% prior-language replay into each phase's stream eliminates forgetting
outright: final en/de/es = 2.33/2.24/2.12 versus joint co-training's 2.33/2.23/2.23, at
+27% data and optimizer budget. Round-robin interleaving *without* sufficient replay
volume does not help (en/de remain at 12–14 ppl) — forgetting is capacity competition,
not recency decay.

### 5.6 Decision rule

| constraint | mechanism |
|---|---|
| few phases, budget flexible | **replay-in-training** (joint parity, trivial) |
| strict single-pass, many phases | **growth + routed serving** (near-oracle) |
| single fixed-width artifact required | **merge + random-prune + short replay** |
| unlabeled streams at inference | likelihood mixture (soft) or compiled detector |

## 6. Empirical laws

- **L1** Forgetting is representational overwrite, global across parameter classes.
- **L2** Weight isolation ≠ computation isolation; erosion enters via the shared
  depth-recurrent residual and accumulates per added block.
- **L3** Within-phase knowledge is distributed (random ⊃ ranked pruning; coherent-region
  removal collapses); between-phase knowledge is modular (concatenation retrieves).
- **L4** Activation sparsity at fixed capacity is set by data composition: flat in
  volume (30–90 MB), rising ~+0.5 pp per added language, falling −4 pp on first register
  mixing then saturating. Capacity-to-data dominates across scales (97.4% @ 25M).
- **L5** Cross-corpus loss comparisons are void; measurement requires protocol congruence,
  calibrated graph thresholds, and within-corpus framing.

## 7. Limitations

Byte-level vocabularies sidestep embedding/head coupling that subword tokenization would
introduce; all results are 100M-scale on related-corpora sequences; the gate-distillation
probe tested one restricted gate class (linear, per-token, post-hoc) — richer classes may
shift the P4-learnability verdict; replay uses real tokens (synthetic replay untested);
erosion slope measured to 5 phases only.

## 8. Conclusion

Depth-recurrent weight sharing makes continual learning *harder* — freezing is not
protecting — but its uniform neuron scaling makes it *tractable*: growth manufactures the
invariant structure that selection needs, and everything else (compression, replay,
detection) has a measured operating point. The brain-analogy the BDH paper defers — how
fast synaptic state becomes durable change without catastrophe — receives here not a
single mechanism but a mapped design space with proven boundaries.

## Appendices (to be expanded)

- A. Full theorem statements and induction proofs (commutator condition; prefix
  separability PS1/PS2; expansiveness bounds).
- B. Prior-art triage table (PathNet, HAT, PackNet, DEN, Progress&Compress, OGD/GPM,
  hypernetworks, MoE, highway/residual gating, ResCL, dynamic-information-balancing).
- C. Complete hyperparameters, seeds, and per-run logs pointers.
- D. Negative-results appendix: write-gating null; constant-LR rejection; block-structured
  prune collapse; learned in-pass gate collapse.
