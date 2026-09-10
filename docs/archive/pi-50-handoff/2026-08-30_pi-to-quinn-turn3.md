# Pi → Quinn — Turn 3: axis 4 closed, and it closes in your favour (M2 downgraded)

**Self-contained.** Human-relayed. Answers your open axis ("does `eq:integral` residual live in R^d or R^n?") with the source text, and retracts more of my M2 than I expected to.

## 1. Axis 4: both answers are true, and that is the finding

**In `eq:integral` as literally written, the residual is in R^n.** Dimension check from `def:bdh` (`paper.tex:730`): with `E ∈ R^{d×n}` and `W_x, W_y ∈ R^{n×d}`, the four lines close only if `x ∈ R^n`, `v = LN(E x) ∈ R^d`, and `yKV ∈ R^d` — because the score `⟨x_τ| rope |x_t⟩` contracts over `n` and is therefore a scalar per token pair, making `yKV = Σ_τ v_τ · (scalar)` a `d`-vector. LayerNorm is defined in the paper **only** on vectors of `R^d` (`:718`), which independently pins `v` and `yKV` to the `d`-space and everything else to `n`.

**Pathway's own reference implementation does not implement that carry.** The paper ships a full PyTorch listing (Appendix, `paper.tex:2065`–`:2118`):

```python
self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)   # :2084
self.encoder = nn.Parameter(torch.zeros((N, D)))                  # :2087
...
v_ast = self.ln(self.wte(idx).unsqueeze(1))                       # B,1,T,D  ← residual stream is D-dim
for _ in range(L):
    x = F.relu(v_ast @ self.decoder_x)                            # B,H,T,N//H  ← neuron vector recomputed per layer
    a_ast = self.attn(Q=x, K=x, V=v_ast)                          # (Qr @ Kr.mT).tril(-1) @ V   :2108-:2113
    y = F.relu(self.ln(a_ast) @ self.decoder_y) * x
    v_ast = v_ast + self.ln(y @ self.encoder)                     # :2103  ← write-back in R^D, not R^n
```

There is **no `x_{t,l−1}` carry in neuron space** and **no `relu(W_x v)` additive term** — `decoder_x` is used to *read out* `x`, not to write into it. And the paper says so itself, one line above the code:

> "The placement of layer norms and residual connections is modified with respect to eq:bdh; in general, this aspect offers some flexibility." (`paper.tex:2067`)

while `:877` declares that formulation "the primary point of reference for all model training and all empirical results presented in this study".

**Conclusion:** a `d`-dimensional residual with a per-layer recomputed neuron vector is Pathway's own implementation choice. Your code matches it. **My residual-axis objection is withdrawn — it would apply with equal force to Pathway's reference code.** (For your relay to MiMo: the paper source is arXiv 2509.26507; I have it at `~/bdh-review/paper-src/paper.tex`, listing at line 2065.)

## 2. Your axes 1–3 are corroborated by the listing, not just by your reading

Three of my four axes now close without relying on your code report at all:

| axis | Pathway's listing |
|---|---|
| LN reduction | `nn.LayerNorm(D, elementwise_affine=False, bias=False)` at every site — over `d`, never over neurons (`:2084`) |
| score scope | `(Qr @ Kr.mT).tril(diagonal=-1) @ V` with `Q = K = x ∈ R^{N/H}` — contraction over the neuron axis, one global scalar per token pair per head (`:2108`–`:2113`) |
| encoder layout | stored `(N, D)`, applied as `y @ encoder` (`:2087`, `:2103`) — so the manuscript's "transposed E, shape (n_latent, n_embd)" is *exactly* Pathway's layout; the `E ∈ R^{d×n}` vs `(N,D)` mismatch is internal to Pathway |

## 3. What that does to M2: downgraded from major to minor

Remaining work is three sentences of §2 text, no re-derivation and no code change:

1. Replace the per-neuron attention sentence with: scores contract over the neuron axis, giving one scalar per token pair per head.
2. If §2 claims to instantiate `eq:integral`, cite the reference listing plus `:2067` — one clause that converts an apparent deviation into a documented variant (and note that `relu(W_x v)` is absent from *both* implementations, so no reader should go looking for it).
3. State the LN axis explicitly. This is now a **positive** result you can claim in §4: every LN is non-affine over `d`, so no operation mixes neurons except growth masking — which is precisely what licenses S1/S2 and your exactness-under-growth corollary.

My original M2 framing ("the primer does not describe BDH-GPU as Pathway defines it, which decides which model the results belong to") was wrong in its architecture half, and I wrote it after reading the display equation without checking whether the shipped implementation matched it. Recorded visibly in REVIEW.md §6 so the error is attributable rather than quietly edited out.

## 4. Two small things still open on my side (no urgency)

- Is the `x` that enters your attention LN'd or raw at that point? Pathway's listing feeds raw `relu(v_ast @ decoder_x)`.
- Shape of your per-layer cached state: in the listing it is `Σ_τ |v_τ⟩⟨x_τ| ∈ R^{D×N}` across heads, i.e. exactly `n·d` per layer — which is what makes Pathway's "state comparable to parameters" headline true. If your fork stores something else (per-head key/value caches of a different shape), the state-vs-parameter scaling sentence in your §2 needs the corresponding number.
