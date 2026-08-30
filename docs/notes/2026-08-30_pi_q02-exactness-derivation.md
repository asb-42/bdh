# ?02 — Does exactness survive the corrected primer? A derivation

**Artifact:** closes open question ?02 in pi-33's review ledger, and supplies the
full derivation cited by `docs/reviews/2026-08-28_r3-guidelines.md` section 7.

**Provenance:** written by seat **pi-33**. Backend for this derivation is the
cloud `qwen3.8-flash` endpoint (`PI_PROVIDER=bai`, `PI_MODEL=qwen3.8-flash`),
*not* the local 4090 GGUF that authored the referee report — the distinction
matters for the blind-run record, so it is stated up front. Method: hand
derivation against `bdh.py` and Pathway's reference listing
(`paper.tex:2065`-`:2118`), followed on 2026-08-30 by an executed run of
`scripts/verify_masked_forward.py` on the GX10 (section 7). The derivation was
written before any run; the run then confirmed section 4 and refuted nothing in
section 3.

---

## 1. The answer, in one line

Yes under ReLU, and the corrected primer is what makes the hypothesis list
explicit: append-only growth, exact-zero initialization of new neurons, and
verbatim preservation of old RoPE frequencies. **No under ratio-based top-k** —
and for a reason deeper than the one documented in `verify_masked_forward.py`,
which section 4 gives.

## 2. The block being reasoned about (listing-faithful)

Per layer, iterated `L` times, with `v` the residual in `R^D` and `x` the neuron
vector of width `N/H` per head:

```
x    = relu(v @ decoder_x)                       # neurons
s(t) = sum over neurons of  x_tau * phase(t-tau) * x_t     # one scalar per token pair
A    = causal softmax of s
yKV  = sum over tau of  A(t,tau) * V[tau]        # width N/H
v    = v + LN(yKV @ encoder)                     # encoder is (N, D); LN over D, non-affine
```

Growth is an append-only map on the neuron axis: `N` becomes `N' = N + dN`, old
indices keep their positions, `decoder_x` gains rows, `encoder` gains rows.

## 3. Induction over depth

**Base.** `v_0` is the token embedding; `embed` and `lm_head` are copied verbatim
and frozen at growth (`pipeline/train.py:103-104`), so `v_0` is identical.

**Step.** Assume `v` identical at the start of a layer.

1. *Neuron vector.* Append-only means the old rows of `decoder_x` are unchanged,
   and `relu` acts elementwise, so `x'[i] = x[i]` for `i <= N`. For a new neuron,
   zero-initialized weights give `x'[i] = relu(0) = 0`. Hence `x' = [x ; 0]`.
2. *Scores.* `s'` sums over `N'` instead of `N`. The added terms are products
   containing a zero factor, so they vanish, and the surviving terms match
   provided `phase'_i = phase_i` for every old neuron. **This is the hypothesis
   the corrected primer makes load-bearing:** because the score contracts over
   the neuron axis, a perturbed old frequency changes old scores directly. Under
   the primer as originally written the same hypothesis is needed too, so
   correcting the primer does not weaken the claim — it moves the burden from
   "which axis does LayerNorm reduce" to "old frequencies survive growth", which
   is exactly what check S3 of `verify_masked_forward.py` measures.
3. *Attention distribution.* `A` is a softmax of identical scores under the same
   causal mask, so `A` is identical.
4. *Values and `yKV`.* `V` is built from `x`, so `V' = [V ; 0]`, and
   `yKV'[t] = sum_tau A(t,tau) [V[tau] ; 0] = [yKV[t] ; 0]`.
5. *Write-back.* `yKV' @ encoder'` multiplies a vector that is zero on the new
   coordinates by a matrix whose new rows are zero, so it equals `yKV @ encoder`.
   Identical input to a non-affine `LN` over `D` gives an identical increment, so
   `v' = v` at the end of the layer.
6. *Output.* `lm_head` copied, so logits identical. QED by induction over `L`.

**Where the mask actually sits, and a correction to the guideline wording.** The
neuron mask multiplies the activations *after* `relu` (`bdh.py:246-247`), so in the
ReLU regime it yields `[x ; 0]` directly — sufficient on its own, even with
randomly initialized new neurons. That is precisely what check S1 measures, and
measures as passing. So the guideline's "attribute exactness to the mechanism,
never to the mask" is too strong as written: the mask *is* an operative mechanism
in the dense regime, which is the regime the Arm G/R runs use
(`k_sparse_ratio` defaults to `0.0` in `pipeline/config.py:56` and no ladder
script sets it). The accurate rule is: **not the mask alone, once a selection
operator precedes it.**

## 4. Ratio-based top-k is not width-invariant, so exactness fails there twice over

`_k_sparse_relu` (`bdh.py:11-17`) computes

```
k = max(1, int(ratio * x_pos.shape[-1]))
```

`k` is a function of the *current* width. Growth from `N` to `N'` therefore raises
it from `floor(rho*N)` to `floor(rho*N')`. At the grown width the operator keeps
the top `floor(rho*N')` positive old activations, where at the base width it kept
only `floor(rho*N)`. Old neurons ranked between those two counts were suppressed
before growth and survive after it. Their contribution is not zero, so step 1 of
the induction fails: `x'` is not `[x ; 0]`.

Neither rescue works in that regime. Zero-initialization removes *new* neurons
from the competition but leaves `k` larger than it was. The mask zeroes new
entries *after* selection and cannot un-select the extra old entries. So under a
fixed ratio, additive growth changes the active set of old neurons by
construction, whatever the initialization.

**Falsifiable prediction, and the result of testing it.** The prediction was
that in `verify_masked_forward.py`, the `k_sparse` trials must report
`masked+zero-init FAIL` whenever more than `floor(rho*N_old)` old neurons are
positive, and must stop failing once `k` stops binding. It was tested on the
GX10 (`gx10-50ef`, torch 2.13.0+cpu, toy config `2L d=32 nh=2`, `mult 24->48`, so
`N_old = 384`, `N_new = 768`; logits of scale ~0.4, float32 epsilon 1.19e-07).
Coverage of that test, stated where the claim is made: one seed, one input batch,
five top-k ratios of a single toy width pair, all on the repaired script of
section 7. Gap is the largest absolute logit difference between base and grown:

| top-k ratio | k, base to grown | masked+zero-init gap | verdict |
|---|---|---|---|
| 0 (dense ReLU) | not used | 1.04e-07 | exact (one epsilon) |
| 0.10 | 38 to 76 | 2.88e-01 | breaks, worst case |
| 0.25 | 96 to 192 | 1.62e-01 | breaks |
| 0.50 | 192 to 384 | 7.93e-03 | breaks, marginally |
| 0.90 | 345 to 691 | 1.04e-07 | exact again |

The dose-response is the mechanism: the error is largest where the truncation
binds hardest and vanishes once `k` exceeds the number of positive old neurons,
which is exactly what a width-dependent `k` predicts and not what a
"zero-init restores exactness" story predicts. The section stands as written.

**What the run does not establish:** it is not a statistical result (one seed, no
repeats) and not a production-width measurement, and must not be quoted as one.
The dense verdict, and the two `masked+zero-init` verdicts quoted in section 7,
were additionally reproduced over five trials by the script itself.

**Why this is worth more than a bug note.** It supplies a structural explanation
for a result already sitting in the manuscript's negative register: "Top-$k$ as
consolidation aid: slightly worse forgetting than ReLU at identical schedules"
(`cl-bdh-manuscript.tex:794`). Under ratio-based top-k, worse forgetting after
growth is predicted by the selection operator itself, independently of any
training dynamics. That strengthens the paper's thesis rather than denting it:
exactness is a property of ReLU BDH, and top-k is a different mechanism with
different invariances.

**The fix, if sparse growth is ever wanted.** Hold `k` absolute across a growth
step (freeze it at its pre-growth value, or set the post-growth ratio to
`rho*N/N'`), or apply the neuron mask *before* selection. Either restores step 1.

## 5. Frequency lattice (methods line, agreed with Quinn)

`2 ** 16 ** (q/n)` is right-associative in Python: it means `2^(16^(q/n))`, not
`(2^16)^(q/n) = 65536^(q/n)`. Irrelevant to exactness, because old frequencies are
copied verbatim. But new neurons enter on a different lattice than a from-scratch
model of the same width would use, so "grown to width `N'`" and "trained at width
`N'`" are not the same model even asymptotically. One sentence in methods.

## 6. What this does not settle

- **The induction is for the listing-faithful block.** If `bdh.py` departs from it
  in any of the four places the argument touches — `decoder_x` row copy, `V`
  construction, `encoder` row copy, LayerNorm placement — the proof needs
  re-checking against `bdh.py`. What I checked, and no more than this: five named
  sites read directly in `bdh.py` — the top-k-before-mask ordering (`:244-247`), the
  width dependence of `k` (`:13`), the parameter shapes and their axis order
  (`:173-181`), the RoPE pairing convention (`:94-97`, adjacent pairs, so appending
  preserves old pairings), and the absence of any learnable `Attention` internal
  besides `freqs`. That is not an audit of `bdh.py` as a whole. The four-place check
  was then completed by the executed run in section 7, which found the model
  faithful and the harness not.
- **The theorem-level half of ?02 is a text fix, not a math fix.** The
  manuscript's exactness statements should name frequency preservation as a
  hypothesis, and either exclude the ratio-top-k regime or add an absolute-`k`
  hypothesis to it.
- **The run is a toy config.** Two layers, `d=32`, two heads, `mult 24 to 48`. It
  establishes the mechanism, not the production numbers; see section 7 item 4.

## 7. The verification artifact could not run, and had to be repaired first

Executed 2026-08-30 by pi-33 over ssh on the GX10, as the non-author that
`docs/reviews/2026-08-28_r3-guidelines.md` section 7 requires for attesting
`scripts/verify_masked_forward.py`. Four findings, all reproducible:

1. **The script never ran.** `grow()` called `m.embed.copy_(base.embed)`; `nn.Module`
   has no `copy_`, so it raised `AttributeError` on every platform and every
   PyTorch version, at the first trial. Whatever the docstring asserts about S1 and
   S2, this file cannot have produced it. Fixed to `m.embed.weight.copy_`.
2. **`grow()` did not mirror `train.py`, which is its stated contract.** `decoder`
   is `(nh*N, D)` and the call site flattens the neuron axis head-major
   (`transpose(1,2).reshape(...)`, so row `h*N + n`; `train.py:99` says so in
   words). `train.py:131-132` therefore copies per head:
   `dec.view(nh, n_new, -1)[:, :n_old, :] = ...`. The script instead did a
   contiguous `m.decoder[:nh*N_old] = base.decoder`, which mis-places every head
   after the first. With `nh = 2` that is not a corner case, it is half the model.
3. **`zero_new()` had the same layout error**, zeroing `m.decoder[nh*N_old:]` — a
   contiguous tail that, at the grown width, is head 1's *old* rows. So S2 did not
   test zero-initialisation of new neurons; it destroyed old ones. This is what
   produced the apparent `max_ulp` of ~2e9.
4. **The pass criterion was unachievable.** Both S1 and S2 were judged by
   `torch.equal`, bit-equality. Any width change alters the matmul reduction
   order, so bit-equality cannot hold even when the computation is mathematically
   exact — the script's own numerics note anticipated this, but the verdict and
   the exit code did not use it.

**After 1 to 3, the model is faithful and exactness holds.** Dense S1 (mask, random
new weights) and S2 (zero-init, no mask) both come out at `maxabs = 1.04e-07`
against logits of scale 0.40 — one float32 epsilon, i.e. no computation change.
S3 (old frequencies verbatim) and S4 (the sensitivity control does differ) pass.
The script now reports an 8-epsilon criterion alongside the old bit-equality one
and exits 0 on dense configs; the `k_sparse` block is judged the same way, which
is how the table in section 4 was obtained.

Two things this does **not** do. It does not test production width (`d=512`, real
depth, `mult 128 to 736`) — the mechanism is established, the production number is
not, and section 6 of the ladder checklist should say so explicitly. And the
repairs are mine: the verdicts above are from a pi-repaired copy of a
Quinn-authored artifact, so Quinn should read the diff before anyone cites the
script again. The three defects are marked in place with the reason for each.

## 8. Addendum 2026-08-30 ~23:00 - I retract one clause of my own run record

The `verify_masked_forward.py` docstring I wrote at `4c8e51e` ends with "Sparse growth
needs an absolute k, **or the mask applied before selection**." The second alternative is
wrong, and it is now sitting in `docs/plans/2026-08-28_qat-proposal.md` as hard rule 5 and
in `docs/reports/2026-08-30_s1s2-exactness-verification.md`, both of which cite my artifact
as their source. Quinn transcribed it faithfully; the defect is mine.

Why it fails. Masking new slots before selection leaves them at zero, which is the same
input tensor that zero-init already produces - and we measured that configuration breaking
(1.62e-01 at rho = 0.25). A mask controls *which* coordinates may win slots; it does not
control *how many* slots exist. Since k = floor(rho * width), a doubled width doubles k, so
the grown stack admits extra OLD activations. That is a strict superset of the base set, so
no post-hoc or pre-hoc masking of the new block can undo it.

Measured, not argued: `scripts/probe_selection_fix_operators.py` (pi-33, GX10,
torch 2.13.0+cpu, float64 so no result here is a numerics artefact; coverage = widths
N = 24/48/96 x rho = 0.10/0.25/0.50/0.90, one draw per cell, operator level not end-to-end):

| policy | gap vs base | verdict |
|---|---|---|
| absolute k frozen at pre-growth value | 0.000e+00 in all 12 cells | restores the induction step |
| mask applied before selection (== zero-init) | 9.2e-02 ... 1.7e+01 | breaks in every cell |

So sparse growth has exactly one cheap fix among those considered: hold k absolute across
the growth step (equivalently rescale the post-growth ratio to rho*N/N'). Anything that only
touches the new block - mask, zero-init, either order - cannot work, for the counting reason
above. Rule 5 should keep its first half and drop its second.

Process lesson for this project, and it is not about me specifically: the wrong clause lived
in a code comment, not in the derivation prose, and comments get quoted as conclusions without
being re-derived. If an aside is load-bearing enough to become a hard rule, it belongs in the
note with a falsifier attached - which is the standard this file applies everywhere else.
