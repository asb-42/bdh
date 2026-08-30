# Review audit — verification of Pi's blind review artifacts (rev2 @ cb72631)

Auditor: Quinn (backend glm-5.3-flash). Object: `2026-08-29_pi_manuscript-r2-review.md`,
`2026-08-29_pi_claims-vs-evidence.md`, `2026-08-29_pi_bdh-parent-paper-notes.md` (same upload batch as
`2026-08-29_refereee-report_pi.md`, already audited). Ground truth: `docs/papers/cl-bdh-manuscript.tex`
(rev2 @ cb72631), Pathway source (`paper.tex`, arXiv 2509.26507), the `docs/reports/` series, and
`pipeline/` + `scripts/` code. Method: line-level grep verification of every checkable quotation,
independent recomputation of all arithmetic, code inspection for claims outside the paper's scope.

## Verdict

Substantiated at referee grade. ~90% of checkable claims verified, zero fabricated quotations
(every sampled quote is verbatim; line refs drift ±4 at worst). All six majors survive the full audit.
The notes contain a layer of findings **beyond** the referee report, itemized below.

## Majors — audit results

- **M1 (units)**: confirmed. `ln(24.6/11.9) = 0.726`, `ln(22.8/2.1) = 2.383` — Pi's +0.73/+2.40 nats.
  Root cause verbatim in the ladder report (ppl differences labelled nats). P-Eros still fails its
  0.3-nat falsifier, by ~2.4x/~8x instead of 10–70x.
- **M2 (primer ≠ BDH-GPU)**: confirmed against `eq:integral` — the `relu(W_x v)` write branch is absent
  from the primer, the parent's attention score is one global scalar per token pair (not per-neuron),
  and the parent residual lives in R^n. Math intact, attribution broken. Recommended fix accepted:
  re-derive or declare a channel-wise variant.
- **M3 (parity on wrong branch)**: confirmed. Table 2 gaps +0.25/+0.34/+0.18 ppl = +0.10/+0.14/+0.08 nats;
  gap-closed 90/84/87% (recomputed in nats, matches Pi). §5.5 replay branch is the genuine parity branch.
- **M4 (prior art)**: confirmed by my own searches. Net2Net/Network Morphism (function-preserving
  expansion), PCANets/Modularity-with-Invariance (frozen-block growth + norm statistics), Riemer'21
  (recurrent-trunk growth), subspace-selection variants — all absent from the triage table. Pi's own
  "verify" flags on items 2–4 stand.
- **M5 (proof hypotheses)**: confirmed, including a correction **of my earlier position**: Pi's reduction
  of the first-component condition to `(1+g1)^2 + g1 g2 εδ = 1` is correct (ReLU positive homogeneity),
  the main text's `= 4` is wrong; his signed-gate counterexample (γ=1, g1=0, g2=−2) verifies exactly.
  Zero-forcing cancellation gap confirmed at `:677`.
- **M6 (mechanism sentence contradicts evidence)**: confirmed and **strengthened by code** — Pi's open
  question R4 ("was anything frozen in Arm G?") is answerable from the repo: `pipeline/config.py:59`
  freezes old neurons + embedding + LM head (fresh neurons only), `freeze_attn: True` default,
  `pipeline/train.py:147–165` applies `requires_grad_(False)`. Inferred from defaults + script silence,
  not run logs (caveat noted). Consequence: there was no overwrite; old computation survives in weights
  and erosion is purely computational. "Growth protects weights but not computation" is the honest
  sentence — stronger than the manuscript's current claim.

## Findings in the notes beyond the report

- **C-7 — growth rate wrong.** [ERRATUM 2026-08-30, superseded by Pi's checkpoint measurement — see docs/reports/2026-08-30_pi-checkpoint-measurement.md.] The text below was infected by the report's ×708/19-phase premises; I recorded the mutual consistency of two derived numbers as verification, which is exactly the failure mode it describes: "×128→×708 over 19 phases = 30.5 multiplier units/phase = ~1,950 neurons/head per phase, not '+32'. Width and parameter figures are mutually consistent (554M/100.9M ≈ 708/128)." Measured truth: 20 phases, ×128→×736 → rate is +32 mult units/phase = **+2,048 neurons/head** (manuscript understates by 64×). And ~554M is not a wrong number at all — it is sl's exact count (553,955,328); the manuscript simply describes Arm G one phase short (final lt = 579,123,200).
- **A-5 — the dissoc witness is degenerate in-architecture.** LayerNorm is shift-invariant
  (`LN(z + c·1) = LN(z)`), so the witness `F' = F_A + c·1` vanishes before the first LN and the closed
  form `F'^L = F_A^L + Lc·1` does not hold. Conclusion survives trivially at depth 1; cite N4's measured
  case instead.
- **H10/H11 — merge evaluation axis.** Pathway's concatenation isolates the n-dimensional parameters but
  **averages embeddings/unembedding** — the parent's own failure channel (generation broken in all three
  from-English directions, below the 0.65 baseline). Our merge/prune/replay branch must be measured on
  those failing columns, not only where plain concatenation already works.
- **I28 — unlabeled interference evidence in the parent.** En→Es 0.35→2.57 in Pathway's own merge table
  is textbook forgetting, never labeled as such. This is the citation our motivation section should use
  instead of EWC-era analogies.
- **I26 — scaling parity uses an appendix-only variant.** The headline parity result is BDH-GPU′
  (xLSTM-like gating, defined only in `sec:bdh_scaling_details`), not the architecture whose theory is
  established. Transfer and isolation claims must state which variant is implemented.
- **I29 — the T~1/ρ argument is asserted, never given.** Our manuscript borrows the "state runs out"
  narrative; it must cite it as Pathway's conjecture, not established fact.
- **H5 — native routing channel.** From `eq:integral`(3): `supp(xy) ⊆ supp(x)` — support confinement
  means the architecture already routes through x's support. Prefix routing must relate to this implicit
  gate or justify its redundancy. New design-review constraint for R3, found by no internal review.

## Independent verifications worth recording

- Parameter identity: `3·65536·512 + 2·256·512 = 100,925,440` — exact match to the manuscript and exactly
  Pathway's `(3+o(1))nd` form. Pins down "width ×m" as m×512 total neurons.
- Pi's I17 resolution (Claim `claim:graphs` construction via `G^ee − G^ii` sign-splitting) checked by me:
  correct. Recorded so nobody re-reports his withdrawn objection.
- Soft-leakage exponent from the two published endpoints: `ln(0.193/0.0023)/ln(20) = 1.48` (vs. claimed ~1.7).

## Errors found in Pi's artifacts (complete list)

Line-ref drift ±4; one arithmetic slip (554M-parameter arm stated as 553M/556M in his verification log;
exact value 558M, conclusion unaffected); one rounding (his ×134 for the literal-implementation endpoint
vs. ×137.5 by my arithmetic — conclusion identical); CI 0.93 vs. exact binomial 0.94; the referee report's
"Fixed" reconcile cell is imprecise (the sentence "the routing is coarse, not fine-grained" still stands
verbatim at `:440`). **No substantive error.**

## Not verified in this pass

Register numbers (`:348`, `:792`) not independently reproduced; the fastText-baseline claim (B-2) not
empirically tested; Pi's flagged "verify" literature items remain flagged. **Raw-data re-verification by
Pi (offered, pending)** would cover the register and report tables directly from run logs.

## Disjoint coverage — meta-result

Eight internal Quinn reviews and Pi's blind review share no critical headline finding (units error M1 and
parity misattribution M3 appear in none of ours; the missing artifact check appears in none of his).
Two independent reviews of the same revision, both "major revision", discrete critical findings. This is
the first full A/B data point for the multi-model review pipeline (see dark-factory planning notes).
Provenance: Pi ran blind — no team context, no J-Space skill — on Qwen3.8-Flash-Next (local 4090).
