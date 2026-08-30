# QAT for BDH — Draft Proposal (Reminder)

**Date:** 2026-08-28
**Status:** draft — do NOT start before Revision 3 of the CL manuscript is complete. This
plan exists so we do not miss the right moment; it is to be revised when Phase Q1 is due.
**Owner:** Quinn (review seat) — design; MiMo (execution seat) — implementation.

---

## 1. Why this exists

QAT for BDH is architecturally attractive but must not be launched while the current CL
science (Phase 2/3, manuscript Revision 3) is still moving. Running QAT in parallel would
corrupt every FP32-based metric comparison (protocol-congruence invariant INV-1) and flood
the next diagnosis with a second simultaneous regime (same lesson as the freeze_attn
confounder). This draft is a placeholder so we recognize the right moment and have the
design ready for revision when it arrives.

## 2. Motivation: the Qapdex lesson is the core evidence

The 1.58-bit PTQ attempt collapsed to 0.000 on every benchmark. The failure was not the
target model — it was PTQ instead of QAT. BitNet-style quality only emerges when
quantization is simulated during training (straight-through estimator, fake-quant forward
pass). Any future quantization effort for BDH must therefore be QAT, never PTQ.

## 3. Why BDH is QAT-friendly (architectural fit)

1. **Coordinate separability (manuscript facts S1/S2):** the only cross-neuron coupling is
   LayerNorm. Encoder columns, per-neuron attention, and decoder rows act independently
   per neuron, so quantization can be controlled per neuron/parameter slice. Transformer
   attention (QKV projections coupling across the sequence space) does not have this.
2. **Sparse Positive Activation Rule (~5% active):** most quantization error lands on
   inactive paths and does not reach the output. QAT can exploit the sparsity as a
   noise sieve.
3. **Additive growth / CL compatibility:** quantization scales and fake-quant operators
   must be defined per phase block and must survive the growth rule; the recipe's
   isolation properties (prefix selection, merge/prune/replay) must hold under quantized
   inference.

## 4. The warning to take seriously: expansiveness amplifies injections

Measured directional gains 1.05–1.89 per level over L=6 levels. A per-level quantization
injection gets multiplied across levels — conservative worst case roughly 6^1.5 ≈ 15×,
realistically less because LayerNorm renormalizes, but not negligible. Implication:

- QAT for BDH must simulate quantization **per level** with separate scales
  (per-level fake-quant + scaling), not a single global quantizer.
- The Q1 quality gate must be measured under the exact same evaluation protocol as FP32
  (cold random-crop, within-corpus, congruent protocol).

## 5. Phases and gates

| Phase | Trigger | Content | Exit gate |
|---|---|---|---|
| **R3** | now | CL science: manuscript Revision 3, Phase 2/3 results (growth+routing, freeze_attn diagnostic) | manuscript submission-ready |
| **Q1** | R3 complete | Feasibility on tiny model (~0.3M params, as in the earlier Qapdex-style probe): fake-quant forward, STE, per-level scales; quality vs FP32 baseline | **quant loss ≤ 0.1 nats** vs FP32 on the same data/protocol. If not met → QAT for BDH is a dead end, stop early and cheap |
| **Q2** | Q1 green | Quantize train/infer at 100M scale; measure P-Acq / P-Eros / P-Route under quant with same falsifiers (0.3-nat erosion bound) | routing and isolation metrics match FP32 within protocol tolerance |
| **Q3** | Q2 green | Full CL + QAT combination: consolidation under quant, growth under quant, routing under quant; merged+pruned artifacts quantized end-to-end | near-joint parity of consolidated quantized model vs its FP32 counterpart |

## 6. Hard rules

1. Never PTQ a BDH post-hoc at extreme bit-widths (the Qapdex failure mode).
2. Never measure QAT results against FP32 under a different protocol (INV-1).
3. Never run QAT experiments while the FP32 CL protocol is still the active measurement
   baseline for the manuscript.
4. Per-level quantization scales are mandatory; global-only quantization is rejected.
5. Any selection operator entering via quantization (top-k / sparse positive
   activation, §7 Q1) must be width-invariant under growth: hold k absolute across
   the growth step (freeze the pre-growth k, or equivalently rescale the post-growth
   ratio to rho*N/N'). Masking the new block — before or after selection — does NOT
   restore exactness: it is tensor-equal to zero-init, and k = floor(rho*width)
   still admits extra OLD activations. Ratio-based k silently breaks forward
   exactness at every growth step (docs/reports/2026-08-30_s1s2-exactness-verification.md;
   counting argument and float64 operator probe: derivation §8, commit 6754e83).

## 7. Open design questions (for the Q1 revision)

- Optimal bit-width and whether the Sparse Positive Activation rule permits 1.58-bit
  ternary weights at all, or whether 2–4 bit ranges are the real target.
- Interaction of quant scales with the additive growth rule (do scales need to be per
  phase block and frozen on growth?).
- Whether the compiled detector / likelihood router remain 100% accurate on quantized
  activations, or need quant-aware thresholds.

## 8. Trigger condition for revision

Revise this plan (and turn it active) when ALL of: (a) manuscript R3 is committed,
(b) Phase 2/3 results are reviewed and accepted, (c) the freeze_attn diagnostic has a
final verdict. Until then it stays a reminder draft.
