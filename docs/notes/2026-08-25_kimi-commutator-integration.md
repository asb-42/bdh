# Kimi Opinion Digest — Commutator unification + four-opinion synthesis

_Date: 2026-08-25 · Input: user-provided Kimi analysis · Companions: danwa / chatgpt / grok integration notes_

## 1. What Kimi adds

1. **H5 — the commutator condition** `[P_A, F'](x) = P_A F'(x) − F'(P_A x)` on the old
   subspace: exact elimination under depth-constant, input-dependent, no-per-level,
   no-retraining constraints is possible **iff** the commutator vanishes there
   (otherwise impossible without per-depth adaptation or retraining). Classified
   apparently novel. This is the cleanest single characterization of the obstruction —
   ChatGPT's `F P = P F P`, Danwa's T1/T2 zero-forcing, and Grok's H1 boundary all
   reduce to statements about this object.

2. **H1' — Prefix-Separability theorem (PS1/PS2)** with explicit induction proof:
   PS1 (`P_A f'(x) = P_A f'(P_A x)`) + PS2 (`P_A f'(P_A x;θ') = P_A f(P_A x;θ_A)`) ⇒
   hard prefix projection reproduces the phase-A trajectory exactly, all depths.
   Approximate PS2 within δ ⇒ linear error growth `l·δ`. This turns Addendum 7's
   empirical verification into a corollary of a stated theorem.

3. **H2 — exact 2D ReLU counterexample**: no soft gate (open interval) solves the
   preservation functional equation for all inputs once cross-coupling exists —
   strengthening Grok's H1 from "over-constrained" to an explicit insoluble equation.

4. **H4 — definitional equivalence**: hard prefix projection = expert routing over
   disjoint specialists stored in one weight matrix. Sharpens deployment language:
   the grown stack *is* a multi-specialist container, not a protected shared model.

## 2. The one refinement we contribute back

Kimi asserts BDH *satisfies* the commutator condition structurally. Precisely:

- **Hard regime**: true. Zeroing suffix modules removes their pre-LN contributions;
  every remaining operator (E columns, per-neuron attention, decoder rows) is
  block-separable, so `[P_A, F'] = 0` identically — hence bit-exact reproduction
  (verified three times: cl4, tmg, smoke).
- **Soft regime**: false. LayerNorm is not coordinate-wise: `ln(h_A + h_B)` mixes
  statistics, breaking commutation at first order in suffix magnitude. Our measured
  leakage curve (j ∈ {0.05…1.0} → drift .0023→.193, ~j^1.7) **is** the empirical norm
  of `[P_A, F']` under partial activity.

So the correct statement: *BDH's operator structure enforces the commutator condition
on the hard mask algebra; LayerNorm is the unique soft-commutation breaker, and its
leakage is measurable and budgetable.* This reconciles Kimi's claim with our data and
locates the architecture's one softness liability precisely.

## 3. Four-way synthesis

| construct | Danwa | ChatGPT | Grok | Kimi |
|---|---|---|---|---|
| central object | permission set (T1/T2 zero-forcing) | invariance `F P = P F P` | exact/bound boundary (H1/H2) | commutator `[P_A,F']` |
| negative result | global-preservation degenerate | H-A counterexample | continuous-gate impossibility | dense-coupling counterexample (H2) |
| positive construction | delta placement legal | selector vs creator | discrete routing exact | PS1/PS2 induction (prefix theorem) |
| open crux | reachability (T8) | learnability of P_x | soft-gap engineering | general architectures lack the property |
| depth behavior | T7 expansiveness conjecture | bounding hierarchy | linear-or-better bound conjecture | compounding upper/lower bounds |

Convergent conclusions (unanimous): exactness requires the structural condition; BDH
growth builds it; soft mechanisms approximate but never equal it; protection-class
methods are formally excluded; prior art addresses neighboring problems only.

## 4. Instantiation caveats carried forward

- Kimi's formalism treats neuron indices as *state coordinates*; in BDH the persistent
  state is shared D-space and neurons are operators on it. PS1/PS2 hold in the
  operator sense (module activity), not literal coordinate subspaces — the theorem
  transfers, the geometry does not.
- Assumption "Embed(x)_new = 0": satisfied trivially (embed is frozen and shared).
- Hebbian/native plasticity: absent in our training; path genuinely frozen.

## 5. Status after four opinions

The theoretical picture is closed to the strength available short of publication:
one obstruction object (commutator), one sufficient construction (additive prefix
growth + selection), one measured soft-regime liability (LN leakage, budgeted),
one open conjecture (learned soft selectors — already answered constructively by the
training-free likelihood mixture). Remaining work is execution: Mechanism G efficiency
probe, multi-seed rigor (running), interleaved schedules, and writing.

_Artifacts_: none new; synthesizes three companion notes + cl-h1 report.
