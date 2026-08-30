# S1/S2 Exactness Verification — GX10-attested round

**Date:** 2026-08-30
**Status:** consolidated, citable record for the ?02 verification round
**Provenance:** harness authored by Quinn (d1bc095, repaired at 4c8e51e);
executed and attested by seat **pi-33** on the GX10 (`gx10-50ef`,
torch 2.13.0+cpu) as a non-author, per the independence rule in
`docs/reviews/2026-08-28_r3-guidelines.md` section 7. Derivation written
before the run: `docs/notes/2026-08-30_pi_q02-exactness-derivation.md`.

## Question

Does masked-forward exactness survive width growth in this fork, and under
which conditions? (Open question ?02.)

## Answer

**Yes in the ReLU regime** (`k_sparse_ratio = 0.0`, the regime of all Arm G/R
and RA2 runs), under three jointly necessary conditions: append-only growth on
the neuron axis, exact-zero initialization of new neurons, and verbatim
preservation of old RoPE frequencies. **No under ratio-based top-k** — and not
repairable there by mask or zero-init (see Structural finding).

## Measured (dense, five trials, toy config 2L d=32 nh=2, mult 24→48)

- S1 (mask-only) and S2 (zero-init, no mask): exact to **one float32 epsilon**
  (max gap 1.04e-07 on logits of scale ~0.4) under the 8·eps criterion.
- S3 (old RoPE frequencies preserved verbatim): pass.
- S4 (sensitivity control, random new weights must differ): pass.

Bit-equality is **unachievable across a width change** (BLAS reduction order
changes with shape); the verdict therefore keys off the epsilon criterion.
Bit-equality remains meaningful only at identical shapes.

## Harness repairs (original d1bc095 could not run at all)

1. `m.embed.copy_` → `m.embed.weight.copy_` (nn.Module has no `copy_`; the crash).
2. `grow()`: decoder rows are head-major (`h*N + n`, per `train.py:130-131`);
   the contiguous slice mis-placed every head after the first.
3. `zero_new()`: same layout error — it zeroed head 1's OLD rows, which made
   S2 diverge by ~1e2 before the fix.

The original version never produced a verdict; recorded as failure class **F6**
in the team A/B protocol (never-executed verification artifact is a hypothesis,
not a harness). Only `verify_masked_forward.py` at **4c8e51e or later** is
citable.

## Structural finding: ratio-based top-k is not width-invariant

`_k_sparse_relu` (`bdh.py:13`) computes `k = max(1, int(ratio * width))`. After
growth N → N′, the operator retains `floor(rho·N′)` OLD activations where the
base retained `floor(rho·N)`: the active old set changes by construction.
Neither the neuron mask (applied after selection, `bdh.py:246-247`) nor
zero-init of new neurons can undo this. Measured dose-response (masked +
zero-init, max logit gap):

| ratio | k base→grown | gap | verdict |
|---|---|---|---|
| 0 (dense) | — | 1.04e-07 | exact |
| 0.10 | 38→76 | 2.88e-01 | breaks (worst) |
| 0.25 | 96→192 | 1.62e-01 | breaks |
| 0.50 | 192→384 | 7.93e-03 | breaks, marginally |
| 0.90 | 345→691 | 1.04e-07 | exact again (k stops binding) |

The non-monotonicity is the mechanism's signature. Consequence: the
manuscript's negative result at `cl-bdh-manuscript.tex:794` (top-k forgets
worse than ReLU) is predicted **structurally**, by the selection operator
itself, independent of training dynamics — this strengthens the thesis that
exactness is a property of ReLU BDH.

**Fix if sparse growth is ever wanted (not yet implemented; `bdh.py:13`
unchanged):** hold `k` absolute across the growth step (freeze pre-growth k, or
set post-growth ratio to `rho·N/N′`), or apply the neuron mask **before**
selection. Either restores induction step 1.

## Coverage limits

One seed, one input batch, five top-k ratios, a single toy width pair, CPU.
Not a statistical result and not a production-width measurement; the dense
verdicts were reproduced over five trials by the script itself. Pathway
reference listing (`paper.tex:2065`-`:2118`) consulted by the attesting seat.

## References

- `docs/notes/2026-08-30_pi_q02-exactness-derivation.md` (full derivation, §4
  for the top-k analysis, §7 for the run record)
- `docs/reviews/2026-08-28_r3-guidelines.md` section 7 (binding rules)
- `docs/tasks/2026-08-30_ladder-ra2-validation-checklist.md` (citation
  restriction to 4c8e51e+)
- Commits: d1bc095 (original) → 3a46426, 4c8e51e (repairs, derivation,
  attestation)