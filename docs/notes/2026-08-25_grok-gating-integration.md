# Grok Opinion Digest — Continuous-gating impossibility, and its constructive limit

_Date: 2026-08-25 · Input: user-provided Grok formalization · Companions: `danwa-deltagate-integration.md`, `chatgpt-invariance-integration.md`_

## 1. What Grok adds to the triangulation

**H1 (negative, apparently novel)**: under pure additive depth-recurrence with a frozen
shared path, *no continuous globally-shared input-dependent gate* can enforce exact
residual annihilation Δ_A ≡ 0 for all subsequent phases — g must nullify every later
update along x_A's trajectory while leaving x_B's trajectory intact; over-constrained
past depth > 1. Escape routes: discrete prefix selection (a), per-level gates frozen
after introduction (b), one-time path rewriting (c).

**H2**: soft likelihood-derived residual scaling bounds interference — conjecture,
now tested below.

**H3 caveat**: subspace projection presumes a-priori linearly-separable phase subspaces;
our distributed-within-phase results (random>ranked pruning, block-prune collapse) make
that assumption false in BDH. This definitively eliminates subspace-projection variants
from the design space — multiplicative module-scaling (block activity), not state-space
projection, is the viable family.

## 2. Three-way reconciliation

No contradictions across opinions once placed side by side:

| opinion | role in the final picture |
|---|---|
| Danwa | permission structure: delta placement legal, t-conditioning legal, reachability = the open crux |
| ChatGPT | constructive target: invariant-subspace condition (H-B), per-recurrence application (H-C), selector-vs-creator framing, P1–P5 hierarchy |
| Grok | sharp boundary: continuous⇒bound-only (H1); discrete⇒exact; separability caveat kills projection variants (H3) |

Unified thesis (paper spine): *the architecture creates invariant structure by additive
prefix growth; selection of that structure — hard or soft — is exact-or-near-exact;
nothing else can work, for provable reasons.*

## 3. H2 tested: temperature-softened likelihood mixing on `cl4-es_last`

Mixture p(x) = Σ_r w_r(x) p_r(x), w = softmax(−routeNLL/τ) over the three prefix experts
(128-token routing window; 16 crops/language; served-position NLL):

| τ | en | de | es |
|---|---|---|---|
| 0.05 | 0.857 | 0.861 | 0.842 |
| 0.25 | 0.860 | 0.859 | 0.842 |
| 0.50 | 0.892 | 0.867 | 0.847 |
| 1.0 | 0.967 | 0.955 | 0.922 |
| 2.0 | 1.041 | 1.099 | 1.090 |
| hard / oracle | 0.857 / 0.857 | 0.861 / 0.861 | 0.842 / 0.842 |
| full-width unrouted | 2.547 | 2.551 | 0.842 |

Verdict: **H2 verified constructively and strongly** — the cheap shared soft gate is
within 0.003 nats of the discrete oracle for τ ≤ 0.25, degrades smoothly without a
cliff, and beats unrouted serving by ~1.7 nats. Combined with the leakage budget
(§16 companion note: j ≤ 0.15 keeps 99% top-1) there is now a measured operating curve
for every point between exact isolation and full sharing.

## 4. Final design verdict (supersedes my earlier four-option dilemma)

1. Exact single-model continual serving = **hard prefix selection** (discrete).
2. Near-exact continuous alternative = **likelihood-mixture serving**, cost K forwards,
   no training, no new parameters.
3. In-pass delta-gating (Mechanism G) remains interesting only for *efficiency*
   (one forward instead of K), constrained by the 0.15 budget on old inputs.
4. Subspace-projection approaches: closed (H3 + our pruning results).
5. Bounding without selection: closed (expansiveness ⇒ T7/Grok-H1).

_Artifacts_: mixture sweep transcript (inline this session).
