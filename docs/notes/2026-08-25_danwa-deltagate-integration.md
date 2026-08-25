# Danwa Debate Digest — Input-path interference under shared depth composition

_Date: 2026-08-25 · Input: five-round adversarial debate protocol (STRATEGIST/CRITIC/OPTIMIST), user-provided · Instantiation: this repo_

## 1. The debate's verdict (scoped)

No impossibility theorem in the CL-relevant reading. The failure surface is a
**conjunction of four necessary conditions** — identifiability (T3), preservation
geometry (T1/T2), gradient coupling of shared gate parameters (T6), depth amplification
(T7) — none individually fatal. Everything reduces to one open conjecture:
**reachability of the instantiated BDH map under the allowed gate class**.

Key placement result: under BDH's residual form `s ← s + g⊙F(s)`, preserving old-task
behavior forces the gate to **zero** (not one) new contributions on active old-task
coordinates ("zero-forcing", T2).

Three scope conditionals: statelessness (running stats revive channel A); the
**t-conditioning fork** — a fixed non-learned level encoding in `g(x,t)` costs O(1)
parameters and is legal, dissolving all depth-agnostic restrictions; Hebbian-style
native plasticity must be disabled for the path to count as frozen.

## 2. Rulings made for this repo (follow-ups #1–#3)

1. **Instantiation**: F = literal BDH level map (`x ← ln(x + ln(xy @ decoder))`,
   ReLU latents, no Hebbian writes — our training has none). Gate class: stateless,
   shared across depths, applied to appended-block deltas only (**C3 delta placement**).
2. **t-conditioning**: **permitted** — fixed RoPE-style level encoding in the gate,
   O(1) parameters, no per-level matrices, frozen path untouched.
3. **Frozen path**: satisfied by construction in `--grow-mult` (old neurons +
   embed/lm_head `requires_grad=False`).

## 3. Mapping our measurements onto the obstruction surface

| Debate item | Our evidence |
|---|---|
| T2 zero-forcing | Prefix routing = extremal legal gate (`g≡0` on old blocks); reproduced specialists bit-for-bit (Addendum 7) |
| T3 identifiability / context-sufficiency | Likelihood router: **100% separation from 128 tokens**, both corpora (Addendum 10). Necessary condition empirically satisfied |
| Channel B (depth-aliasing) live | L2 erosion: +~0.84 nats on oldest phase per added grown block (§10) |
| T6 gradient coupling of φ | Untested directly; C1 (GEM/GPM projection onto orthogonal complement of old-task gradients **in gate-head space**) is the compatible mitigation if needed |
| C4 never-active coordinates | ~94% sparsity ⇒ vast legal write-space; explains why +25% width acquired at near-parity |

## 4. New measurement: trained BDH blocks are expansive (T7 conjecture tested)

Directional spectral-norm proxy (6-step power iteration on JVPs; 8 old-input points;
`cl-a-en_last`, one composed level map):

| level | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| median σ̂ | 1.67 | 1.89 | 1.18 | 1.05 | 1.74 | 1.39 |

Predominantly **σ > 1** ⇒ per-depth modulation error grows geometrically; uniform-in-depth
bounding is unavailable by gate design alone. Consequence: there is **no bounded-error
middle ground** between exact gating and uncontrolled interference. Order-of-magnitude
consistency check: σ^6 ≈ 7× worst-case vs. observed multi-block erosion factors.

_Caveat_: directional estimate (lower-bound flavor), LayerNorm induces some contractive
directions (minima ≈ 0.9), n=8 points. Sufficient for conjecture resolution at the
strength claimed; not a rigorous op-norm computation.

## 5. Design consequence: Mechanism G (delta-gated growth)

The theory collapses my four-option dilemma (report §13 question) to one legal design:

**Grow with an input-dependent delta-gate on the appended blocks**: `y_new ← g_φ(x) ⊙ y_new`,
with `g_φ` small/stateless/shared-across-depths (optionally t-conditioned per ruling 2),
initialized ≈ 0, trained only on new-phase data. Old inputs never enter phase training,
so the gate stays anchored near zero there *provided* its features do not alias old
inputs to look new — enforceable to first order via C1 projection of gate-head updates.

Predictions: (i) unrouted single-model serving approaches routed quality; (ii) frozen-phase
erosion vanishes to first order (removing the +0.84 nat/block slope); (iii) reachability
is guaranteed constructively wherever the detector is accurate — which we measure at 100%.

Falsifiers: gate drifts open on old inputs (measure g statistics on held-out old data);
or new-language acquisition degrades under suppressed deltas.

## 6. Discipline notes

- Debate never saw `paper.tex` or our reports; all its theorems hold for the abstraction.
  Instantiation here is faithful but the reachability conjecture remains conjecture until
  Mechanism G runs.
- Our experiments are admissible only as tests of the debate's conjectures, never as
  theorem evidence (novelty ledger respected).
- Write-up posture for any paper section: new formalization + obstruction map +
  empirical instantiation; mathematics derivative.

_Artifacts_: `/tmp/opencode` probe transcript (expansiveness), this note.
