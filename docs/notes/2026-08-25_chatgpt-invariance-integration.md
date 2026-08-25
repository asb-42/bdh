# ChatGPT Opinion Digest — Invariant-subspace framing + leakage measurements

_Date: 2026-08-25 · Input: user-provided ChatGPT review of cl-h1-report · Companion to `2026-08-25_danwa-deltagate-integration.md`_

## 1. Where it extends the Danwa picture

Danwa mapped *permissions and obstructions*; ChatGPT supplies the **constructive target
theorem** and the right measurement objects:

- **H-A (input-only projection applied once cannot isolate)** — with the clean
  `F(z₁,z₂)=(z₁+z₂,z₂)` counterexample: input separation at t=0 does not survive
  shared composition. Abstract statement of our residual pollution.
- **H-B: exact isolation ⟺ invariant subspaces** (`F P_k = P_k F P_k`). Elementary
  theorem; turns "prefix routing works" into a theorem candidate rather than folklore.
- **§12, key structural point neither Danwa nor my spec emphasized sharply enough**:
  "no per-depth parameters" does **not** force the gate to act only once. One shared
  `P_x` applied at **every** recurrence (`h ← P_x F(h)`) confines trajectories to
  `im P_x` by induction (H-C). Our delta-gate spec already applies per level — this
  legitimizes it formally and separates it from failed applied-once schemes.
- **§14 bounding hierarchy**: exact ⇒ 0; approximate+contractive ⇒ O(ε);
  approximate without contractive ⇒ unbounded. Our expansiveness measurement
  (σ̂ ∈ [1.05, 1.89]) places BDH in the third regime — coherent with both the debate's
  T7 and the observed erosion slope.
- **§24, conceptual reframe worth adopting verbatim**: the gate is not the mechanism
  that *creates* isolation; it is the **selector** for an isolation structure the
  architecture creates. Protection-class methods fail because no such structure exists
  in a shared-weight model; growth builds it.
- **P1–P5 problem hierarchy**, with **P5 (acquiring new invariant subspaces while
  preserving old ones)** named as the real BDH problem — which is exactly what our
  growth rule appears to solve empirically.

## 2. The flagged risk, checked: LayerNorm coupling

§21.2 warns `ln(h_A + h_B)` couples subspaces. In BDH the persistent state is shared
D-space; neurons are operators (E-columns / decoder-rows), not coordinate owners — so
the honest invariant object is *functional*: the grown map is additive over blocks
pre-LN, and hard suffix-zeroing trivially restores the specialist graph (verified twice,
bit-for-bit). Soft suffix activity leaks **only through LN statistics** — attention,
encoder, decoder are all block-separable.

Measured leakage curve (ES block scaled `j` on EN inputs, reference = EN+DE prefix):

| j | 0 | 0.05 | 0.15 | 0.30 | 0.60 | 1.00 |
|---|---|---|---|---|---|---|
| rel drift | 0 | .0023 | .0058 | .0255 | .105 | .193 |
| top-1 agree | 1.000 | .997 | .990 | .953 | .707 | .434 |

Superlinear (~j^1.7), matching expansive-map amplification. **Soft-gating budget:
new-block scale ≲ 0.15 preserves old-task behavior to ≈1%.** Hard routing remains the
exact regime; Mechanism G's gate must either respect this budget on old inputs or be
hard.

## 3. Adopted into the program

1. **Measurement suite** (standard going forward): ε_k = ‖(I−P_k)F P_k‖ (out-leak),
   δ_k = ‖P_k F(I−P_k)‖ via JVPs/finite-diff — the table above is the first instance.
   Plus |(I−P_k)FᴸP_k| vs L to track depth growth.
2. **Mechanism G design constraint added**: gate output on old-input held-out data is
   a monitored quantity; budget 0.15 or hard threshold.
3. **Paper framing**: adopt §24 selector-vs-creator distinction and the §26 headline
   proposition verbatim as the thesis shape; cite prior-art triage (PathNet/HAT/
   PackNet/DEN/P&C/OGD/hypernets/MoE/DIB — neighbors, none solve frozen-shared-
   operator + repeated composition + input projection).
4. **Open conjectures carried**: reachability of learned gates without modifying F
   (ChatGPT novelty matrix row 12) = Danwa's T8 crux. Same open problem, two routes.

## 4. Reconciliation note

Danwa and ChatGPT disagree on emphasis, not substance: Danwa proves no impossibility
and locates gate freedom; ChatGPT states the sufficient condition (invariance) and the
construction (prefix growth) that satisfies it. Combined: **the architecture creates
the invariant structure; the router/gate merely selects it.** Everything we measured
today is consistent with that sentence.

_Artifacts_: leakage probe transcript (inline), companion note `danwa-deltagate-integration.md`.
