# ?02 — Does exactness survive the corrected primer? A derivation

**Artifact:** closes open question ?02 in pi-33's review ledger, and supplies the
full derivation cited by `docs/reviews/2026-08-28_r3-guidelines.md` section 7.

**Provenance:** written by seat **pi-33**. Backend for this derivation is the
cloud `qwen3.8-flash` endpoint (`PI_PROVIDER=bai`), *not* the local 4090 GGUF
that authored the referee report — the distinction matters for the blind-run
record, so it is stated up front. Method: hand derivation against `bdh.py` and
Pathway's reference listing (`paper.tex:2065`-`:2118`). **No model was run**
while writing this: this box has no torch, and the 4090 is occupied by R3.
Every claim that needs a run is marked as such, with its falsifier attached.

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

**Falsifiable prediction, not yet tested.** In `verify_masked_forward.py`, the
`k_sparse` trials must report `masked+zero-init FAIL` whenever more than
`floor(rho*N_old)` old neurons are positive. With the script's own defaults
(`rho = 0.25`, `N_old = 384`, `N_new = 768`, so `k` goes 96 to 192) and random
weights, roughly half of the old entries are positive, the truncation binds, and
the prediction is FAIL. If the run reports PASS instead, then either my reading of
`_k_sparse_relu` is wrong or the toy config is non-binding, and this section
should be struck from the record.

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

- **Nothing here was executed.** Section 4's prediction is untested. Under the
  guidelines' own independence rule (section 7: the author of a verification
  artifact may not attest its own run), the right attester of
  `verify_masked_forward.py` is a non-author, and for that script I am one. Plan:
  run it on the GX10, or on the 4090 when R3 pauses, and record the `k_sparse`
  lines verbatim, including the `masked+zero` verdict.
- **The induction is for the listing-faithful block.** If `bdh.py` departs from it
  in any of the four places the argument touches — `decoder_x` row copy, `V`
  construction, `encoder` row copy, LayerNorm placement — the proof needs
  re-checking against `bdh.py`. I verified the mask/relu ordering, the top-k
  width dependence and the zero-init growth path; I did not audit all four.
- **The theorem-level half of ?02 is a text fix, not a math fix.** The
  manuscript's exactness statements should name frequency preservation as a
  hypothesis, and either exclude the ratio-top-k regime or add an absolute-`k`
  hypothesis to it.
