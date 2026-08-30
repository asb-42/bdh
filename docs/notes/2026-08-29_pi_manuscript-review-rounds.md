# Task 2 — peer review of `docs/papers/cl-bdh-manuscript.tex` (fork @ cb72631)
> Provenance: written by seat **pi-33** (backend: Qwen3.8-Flash-Next, remote 4090), five review rounds, blind - before `docs/reviews/` was read. Added 2026-08-30 because the referee report cites this file and it was not in the repo; the earlier copy of these rounds was overwritten in place by `008ff73`.


Axes: **A** math correctness · **B** claim↔evidence traceability · **C** internal consistency / fork drift ·
**D** novelty & prior art · **E** overclaiming/rhetoric. Reviewer stance: adversarial but constructive; the question is
*is the science sound*, not *can I find fault*.

## Round 1 — Abstract + Intro + Setup/primer (L31–168)

### ★ A/C-1 — the architecture primer is **not** BDH-GPU as Pathway defines it `[MATH]` c8/10
Manuscript primer (`:112`–`:120`): `u = relu(x@E)`, `v = relu(LN(a(u))@E_v)`, `x ← LN(x + LN((u⊙v)@D_c))`, with
`E, E_v ∈ R^{n_h×d×N}`, `D_c ∈ R^{(n_h N)×d}`, `x ∈ R^d` "persistent residual state", and the explicit statement
"`a(·)` is per-neuron attention over positions of `x`: **each latent neuron attends independently** ... every operator above
is coordinate-separable across neurons. The only cross-neuron coupling is LayerNorm's global statistics" (`:122`–`:126`).

Pathway's Definition `def:bdh`, eq `eq:integral` (`paper.tex:735`–`:748`), verbatim:
```
|x_{t,l}>  := x_{t,l-1} + relu(W_x |v_{t,l-1}>)                                   (P1)
|yKV_{t,l}>:= Σ_{τ<t} |v_{τ,l-1}><x_{τ,l}| rope^{t−τ} |x_{t,l}>                    (P2)
|x_{t,l}>  := relu(W_y LN(|yKV_{t,l}>)) ⊙ |x_{t,l}>                                (P3)
|v_{t,l}>  := LN(E |x_{t,l}>)                                                      (P4)
```
with `E ∈ R^{d×n}`, `W_x, W_y ∈ R^{n×d}` (`:731`), `x ∈ R^n` = neuron-activation vector ("each scalar element ... has the
interpretation of a 'scalar' activation state of a single particle", `:725`), and `v ∈ R^d` described as "(fuzzy) addresses of a
virtual memory space of size n" (`:723`).

Four concrete deviations `[FACT]` c9/10 for the side-by-side:
1. **Attention score structure.** In (P2) the score `<x_{τ,l}| rope^{t−τ} |x_{t,l}>` is a **single global scalar per token pair** —
   it contracts the *whole* n-dimensional activation pattern. Every neuron shares one weight; only the retrieved values `v_τ ∈ R^d`
   differ. The manuscript's "each latent neuron attends independently" is a **channel-wise/diagonal** memory (per-neuron keys), a
   different mechanism that discards BDH's defining property: matching whole activity patterns against each other ("fuzzy addresses",
   `:723`). `[INFERENCE]` c8/10 these are not interderivable: the global version is content-addressable over concept space; the diagonal
   version stores per-neuron time series. If the fork implements the latter, its results are about a *different model*.
2. **A whole write branch is missing.** (P1) adds `relu(W_x v_{t,l−1})` — a non-attention write from the previous level's value. The
   manuscript's update has only the gated attention term `(u⊙v)@D_c`. Same parameter budget (3 tensors), different function.
3. **Where the residual lives.** Parent: the residual *is* neuron space `R^n`, `n ≫ d` — that is what makes "state comparable in size
   with parameters" (§4, `:1032`) and "scales almost exclusively in n" (`:1045`) meaningful. Manuscript: a fixed `d`-dim residual with an
   internal neuron expansion `n_h·N`. Growth-by-appending-neurons then does **not** grow the state of the recurrent stream in the way BDH's
   story requires, and "one block reused across all depths" means something weaker.
4. **Encoder direction is transposed.** Parent encoder maps neurons→d (`a* = E z` for `z ∈ R^n`, `:727`); manuscript's `E` maps `d`→neurons.
   Charitable note `[INFERENCE]` c6/10: the parent itself prints the transposed shapes in the Fig. `fig:architecture` caption (`:1011`, my I22),
   so this particular error may have been inherited from a typo upstream — which is precisely why it must be pinned down rather than assumed.

**Why this is the most important finding of the review so far** `[INFERENCE]` c8/10: the abstract's framing ("a language model whose defining
efficiency — one block of parameters reused across all depths"), and every transfer claim (sparsity ≈ 5 %, merge-by-concatenation as in Pathway
§7.1, "state comparable to parameters"), are licensed by *being* BDH-GPU. The theorems themselves are architecture-agnostic (they quantify over maps
`F, F'`), so **nothing breaks mathematically** — but attribution does. Recommended fix is cheap and honest: either re-derive the primer from `eq:integral`,
or state plainly "we study a channel-wise variant of BDH-GPU in which attention scores are computed per neuron; results are reported for that variant".

### R1 — LayerNorm axis must be stated, because exact isolation depends on it `[MATH]` c8/10
The exactness story ("additive growth *constructs* the structure", abstract; `:244` corollary) is true iff no operator reduces an axis that grows.
Parent defines LayerNorm **over dimension d** (`:724`, mean/std estimators `E_d, std_d`) and notes LNs can be moved "to the neuron dimension n" empirically
(`:908`). Consequences for the fork: (i) if `LN(a(u))` normalizes across **neurons**, appending zero-initialised neurons changes old neurons' mean/std and
growth is *not* exact — contradicting "reproduces specialists to within evaluation precision"; (ii) if LNs reduce over `d`, exactness survives growth, and the
abstract's separate finding that "**LayerNorm couples softly active modules** at a measured operating budget" is fully consistent (LN of the shared residual mixes
contributions from all active neurons). So both readings are self-consistent — but the paper never says which axis it uses, and its central exactness claim turns on it.
→ Ask for an explicit statement + one sentence in Theorem `thm:criterion`'s hypotheses. c8/10

### R2 — "The only cross-neuron coupling is LayerNorm" is imprecise `[MATH]` c8/10
Even inside the manuscript's own architecture, `(u⊙v)@D_c` **sums over neurons** into the shared residual: that is functional cross-neuron coupling by construction.
What the authors need (and what would make the growth argument work) is the narrower claim: *no growth-sensitive coupling other than LN statistics and RoPE phase reuse*.
Distinguishing "couples neurons" from "couples tasks under growth" matters, because Theorem `thm:criterion` is about the second. c8/10

### E-1 — abstract density / unverifiable-in-one-pass `[INFERENCE]` c7/10
The abstract packs 12 quantitative claims into 340 words with no scope qualifiers (which model, which phase count, which seed spread applies to which number).
Several are load-bearing and easy to lose: "+1.6 to +2.2 nats in two-language experiments; +12 to +20 nats over 19 sequential phases" mixes units/scales;
"±0.01 ppl over three seeds" and "within 0.003 nats" are stated as parity without an error model. Recommend the abstract carry one explicit convention sentence
(unit, seed count, what "parity" means operationally). Not a science problem — a legibility risk that reviewers *will* hit first (compare I25: the parent paper's
abstract makes exactly this mistake with "state-of-the-art"). c7/10

### C-2 — notation collision inside the manuscript `[FACT]` c9/10
Table `tab:notation` (`:133`) uses `d` for residual width while the parent (and the paper's own title context) uses `d` for the low-rank **synaptic** dimension and
`n` for neurons; `N` = neurons/head, `n_h` = heads. Also `P_A` is "projection/mask selecting the old-phase contribution channels" while `C_P(z)` is a commutation residual —
fine — but `ε`-isolation is defined with a supremum "uniform in depth and phase count" (`:167`) while the growth results let the phase count scale; if any ε is claimed
*independent of phase count*, that is a strong statement needing proof (see §6 accumulation). Flagging now so I check it when I reach the theorems. c8/10

### Positive notes (keep these in the review — a fork should hear what is right)
- Choosing **state-level** isolation as primary and explicitly refusing to accept output-level equality ("output-level equality can hide internal interference that
  resurfaces under distribution shift", `:163`) is the correct methodological call, and stronger than what most CL papers do. `[INFERENCE]` c9/10
- "cross-corpus loss comparisons are void (Europarl is intrinsically easier than WikiText-class text)" (`:155`) — pre-committing to within-corpus evaluation is exactly
  the discipline the parent paper lacks (its merge table compares losses across directions with only a hand-wavy 0.65 baseline, I28). c9/10
- Naming the third property ("weight isolation ... a third, weaker property") and promising to show it implies neither of the others sets up a falsifiable claim rather than
  a slogan. c9/10
## Round 2 — §4 four negative results + §5 theory (L169–290)

### Arithmetic verification I ran myself (all pass — say so explicitly in the review) `[MATH]` c9/10
- N1 "+1.59 / +2.15 nats" ↔ Fig. `fig:forgetting` "4–9× perplexity": `e^1.59 = 4.90`, `e^2.15 = 8.59` ✓ internally consistent, and consistent with the
  abstract's rounded "+1.6 to +2.2". So the nats↔ppl bookkeeping is honest (unlike the parent paper, which never pins its loss units — I28).
- N4 "+0.84 nats per subsequently added block" ↔ "EN: 2.26 → 5.40 → 12.49 ppl": `ln(5.40/2.26) = 0.87`, `ln(12.49/5.40) = 0.84` ✓ (first step rounds to 0.87,
  they quote 0.84 — trivially off, worth a footnote saying whether the quoted rate is a mean or the asymptotic per-block value). c9/10
- Prop `prop:amp`'s recursion `‖e_L‖ ≤ Σ_j (1+L_f)^{L−1−j} δ_j` ✓ correct standard perturbation bound for level maps with Lipschitz constant `L_f`.

### A-2 — Theorem `thm:criterion` is an **iff with an off-by-one** `[MATH]` c8/10
Stated: with `T_A = ∪_{ℓ≤L} {h^{F'}_ℓ(x) : x ∈ X_A}`, "equality at all depths `0 ≤ ℓ ≤ L` **iff** `F'(z) = F_A(z)` for all `z ∈ T_A`" (`:228`).
- **(⇐) is correct.** I checked the induction: `h^{F'}_ℓ(x) ∈ T_A` by definition of reachability *under F′*, so `F'` and `F_A` may be swapped at every step ✓.
- **(⇒) fails at the top depth.** Take `z = h^{F'}_L(x) ∈ T_A`. Equality of trajectories up to depth `L` constrains `F'` only on points reached at depths `< L`;
  nothing in the hypothesis touches `F'(h_L)` vs `F_A(h_L)`. So the forward implication needs either `T_A` truncated to `ℓ ≤ L−1`, or trajectory equality required up to
  depth `L+1`. `[INFERENCE]` c8/10: harmless for the paper's *use* (their growth construction satisfies the pointwise condition directly, and boundary states at depth exactly
  `L` are irrelevant to serving), but an "iff" must be right. Fix is one symbol; leaving it wrong invites a referee to doubt the rest of Appendix `app:proofs`. c8/10
- Same family, milder: sufficiency of (C1)+(C2) silently uses `F_A(T_A) ⊆ T_A` (true by definition of the reachable set, but the line `P_A F' P_A = P_A F_A P_A` only lifts to
  full equality `F'(z) = F_A(z)` once you know `F_A(z) ∈ H_A`). One extra sentence in the appendix proof. c8/10

### A-3 — Lemma `lem:zf` (zero-forcing) needs a positivity or disjointness hypothesis `[MATH]` c7/10
"Exact preservation on `X_A` forces `g_b(x) = 0` wherever `f_b` is nonzero along the preserved trajectory" (`:254`). With a **single** suffix block that is immediate. With
**several** blocks — which is exactly the multi-phase regime of §6 and the soft mixture of §5.3 — preservation requires only `Σ_b g_b(x) ⊙ f_b = 0`, and distinct nonzero gates can
satisfy that by cancelling. The conclusion holds if write-backs are non-negative and gates are (BDH's activations are non-negative after ReLU, so this is *probably* what they use —
`paper.tex:729`), or if the blocks have disjoint support. As written the hypothesis is missing, and it is not cosmetic: cancellation solutions are precisely "a soft gate that is exact",
which Prop `prop:soft` claims cannot happen. → Ask them to state which property rules out cancellation; if none does, `lem:zf` and `prop:soft` are in tension. c7/10 (would rise to a real
error if their write-backs are signed.)

### A-4 — Proposition `prop:soft` proves an existential, is titled as a universal `[FACT]` c8/10
Content: a 2-D ReLU counterexample whose preservation condition `(1+g_1)² + g_1 g_2 εδ = 4` must hold for all input magnitudes `a > 0`, unsolvable by an input-independent pair (`:261`).
That establishes "there exist non-affine coupled update maps for which no input-independent soft gate is exact". The title and the abstract's "**No** soft or hard gate can substitute for this
structure in a generic frozen path" read universally. Two fixes: retitle ("a counterexample class"), and note that BDH-GPU-specific structure (positivity, LN, coordinate separability) is not shown
to fall inside the class. The label "proved (counterexample class)" is honest — the surrounding prose is what overreaches. c8/10

### C-3 — the paper violates its own strict labelling protocol in one place `[FACT]` c9/10
`:217`–`:219`: "We label each result **proved**, **measured**, or **conjectured**, and keep those labels strict: measurements never discharge a missing assumption." Every statement obeys this —
except Corollary `cor:sel` (`:284`), which carries **no label** while making the most sweeping claim in the paper ("gates do not create invariant computational structure; they select structure already
present"). Its scope hedge ("Within the frozen-path projection mechanism studied here") is doing real work, but that mechanism class is never formally defined anywhere. → Either define the class and
prove it, or label it *conjectured*. This is worth raising precisely because the rest of the paper is so careful — it is the one place a referee can quote back at them. c9/10

### Praise worth stating in the open review (calibration + it is genuinely rare)
- **Remark `rem:exp`** (`:276`) is a model of epistemic hygiene: measured directional spectral-norm proxies 1.05–1.89 are reported as "evidence *against* the contractivity premise ... and
  **not** an impossibility theorem: isolated stable directions may coexist with expansive medians". Compare the parent paper, which asserts "a simple information-theoretic argument" it never gives
  (my I29) and claims "state-of-the-art" from a parity table (I25). The fork is behaving better than its parent. c9/10
- **N3's mechanism** (`:186`): importance gating fails *and they explain why* — relative drift 0.6–1.3 per phase for every parameter class, importance spread thin with "99.9 % of neurons above 1 % of
  maximum", i.e. **no localized important subset exists to freeze**. That converts a null result into a structural statement about this architecture, and it is the strongest argument in the paper against
  EWC-style methods here. `[INFERENCE]` c8/10: this also *predicts* that any method relying on importance identification must fail for depth-recurrent shared-weight models — worth stating as a falsifiable corollary.
- **N4's equation** (`:203`) is the paper's best single line: `Δθ_A = 0 ⇏ F'|_{X_A} = F_A|_{X_A}`, backed by a frozen-parameter run that still erodes 0.84 nats/block. The claim is falsifiable, the
  measurement is direct, and it cannot be dismissed as drift. c9/10
- Framing "hard masking reproduces specialists identically" as an *anchor* whose meaning is then delegated to theory (`:210`) is right — but see B-1 below: say plainly that it is an **implementation
  validation of `cor:prefix`**, not an empirical discovery, since exactness is proved. c8/10

### Editorial / precision nits (collect for the referee report's minor list)
- "identically under ReLU and hard top-k activations" (`:181`) — presumably "unchanged"; "identically" invites a bit-exactness reading they don't mean.
- N2: "retention **decreases** by +0.09–0.21 nats" (`:192`) — direction+unit mismatch; retention should *increase* in forgetting units. Say "forgetting increases by ...".
- `thm:dissoc`'s witness `F' = F_A + c·1` is a valid but nearly vacuous existence claim; the empirical content lives in N4. Recommend one sentence separating logical non-implication from the measured case,
  so the theorem is not read as if it were surprising. `[INFERENCE]` c8/10
## Round 3 — §5 the recipe (L291–404)

### ★ B-1 — the abstract's headline parity is **contradicted by the paper's own Table 2** `[MATH]` c9/10
Abstract (`:47`): "merge→random-scatter-prune→brief replay consolidates everything into one fixed-width model **equal to joint co-training
(±0.01 ppl over three seeds)**". Fig. `fig:pareto` caption repeats "reaches joint parity at original width". Table `tab:pareto` (`:356`) says otherwise:

| | EN | DE | ES |
|---|---|---|---|
| merged + prune ⅓ (≈×128) | 6.37 | 5.38 | 4.09 |
| **+ replay finetune †** | **2.58** | **2.57–2.58** | **2.41** |
| joint co-training (×128) | 2.33 | 2.23 | 2.23 |

Residual gap to joint: **+0.25 / +0.34 / +0.18 ppl = +0.10 / +0.14 / +0.08 nats** `[MATH]` c9/10 (`ln 2.58 − ln 2.33 = 0.102`,
`ln 2.575 − ln 2.23 = 0.144`, `ln 2.41 − ln 2.23 = 0.078`). At a joint loss of ~0.80 nats this is a **10–18 % relative loss increase** — not
"equal", and the "±0.01 ppl over three seeds" in that sentence is the *seed spread of the replay row* (per the table footnote), being used where a
*gap-to-reference* is implied. Honest restatement: **"closes 84–90 % of the merge+prune→joint gap (+0.08…+0.14 nats residual)"** — which I computed as
`(ln pre − ln post)/(ln pre − ln joint)` = 0.90 / 0.84 / 0.87 `[MATH]` c9/10, and it is still a strong result that does not need inflation.
Contrast §5.5 (replay during training): EN/DE/ES `2.33/2.24/2.12` vs joint `2.33/2.23/2.23` — **that** is parity (worst case ES even beats joint).
So the paper has one genuinely-parity branch and one near-parity branch, and the abstract attaches the parity claim to the wrong one. `[INFERENCE]` c9/10:
this is exactly the failure mode I logged in the parent paper (I25, "state-of-the-art" from a parity table); here it is milder but the numbers are on the same page,
which makes it easier for a referee to catch and harder for the authors to defend. c9/10

### B-4 — the consolidation branch does not escape replay; it defers it `[INFERENCE]` c8/10
"A final ≤15-minute finetune on a ~9 MB real-token buffer closes the remainder" (`:367`). If that buffer covers all phases (it must, to reach three-language parity), then the
pipeline is *merge + prune + replay-on-everything* — i.e. joint training from an initialised ensemble, not a mechanism that avoids replay. The decision-rule table (`:393`) actually
gets this right ("single fixed-width artifact required → merge + random-prune + short replay"), so the fix is to align the abstract's framing with it and to report the merge branch's
replay budget in the **same units** as §5.5's "+27 % data and optimizer budget" — right now one is "~9 MB / 15 min" and the other a percentage, so the two branches cannot be compared on
the axis the paper itself says matters. c8/10

### B-2 — "100 % detection accuracy" needs its denominator, and its two halves have very different difficulty `[FACT]` c8/10
`:306`–`:309`: pooled-embedding logistic regression and 128-token likelihood scoring "both reach **100 % accuracy**, across languages and across registers of one language".
Problems: (i) no *n* — the only protocol size mentioned nearby is "16 held-out crops per domain" (`:313`), so with 3 classes × 16 crops a perfect score has an exact-binomial 95 % CI
reaching down to ~0.93; (ii) **cross-language routing is a solved problem** — any character n-gram language identifier hits ~100 % on Europarl at 128 tokens, so that half of the claim is near-vacuous;
the *interesting* result is register detection within one language (Wikipedia vs Gutenberg vs parliamentary), which is where a detector can actually fail and where their "three-register stack" numbers live.
→ Split the two results, report n and the confusion matrix for the register task, and add a trivial baseline (fastText-style char n-grams) so the reader can see what the learned detector adds. c8/10

### B-5 — "forgetting is capacity competition, not recency decay" is attributed from an under-ablated contrast `[INFERENCE]` c7/10
`:385`: round-robin interleaving without sufficient replay volume leaves EN/DE at 12–14 ppl, and the conclusion drawn is mechanistic. But interleaving also changes effective LR schedule, batch
composition and gradient noise; a same-total-data sweep over *replay fraction* (0 %, 5 %, 10 %, 20 %) or growth-with-interleaving would separate "capacity" from "optimisation dynamics". Growth arms in §6 may
already answer this — if not, soften to "consistent with capacity competition". Flagging because this is the same evidential standard I applied to the parent paper's emergence claims (I6) and they hold themselves to it too. c7/10

### Terminology / definition nits
- **E-3 "logit mixture" is not what is written** `[FACT]` c9/10: `p = Σ_r w_r p_r` with `w_r = softmax(−NLL_r/τ)` (`:311`) is a **mixture of distributions** (linear in probabilities). A logit mixture
  would be `softmax(Σ_r w_r z_r)`, which behaves quite differently (it does not inherit the mixture's entropy floor and can score better than the oracle on individual tokens). Given that the whole point of §5.2 is
  precision about *what selects what*, mislabelling the operator is worth a correction even though the formula disambiguates it. c8/10
- **R3 "active width ≤ ×192 / ≤ ×224" is undefined and does not match Table 2's notation** (`×128`, `×384`) (`:305` vs `:356`). Is it served compute, parameter footprint, or neurons/head? One sentence +
  consistent symbols. c7/10
- **Superlinearity exponent**: reported "~j^1.7" for relative drift over j ∈ [0.05, 1] (`:322`); fitting the two published endpoints (.0023 → .193) gives `log(83.9)/log(20) = 1.48` `[MATH]` c8/10. Ask for the fit
  (which points, free intercept or through origin, R²), since "as expected under expansive composition plus LayerNorm coupling" treats the exponent as theory-confirmed. c7/10

### Praise (real, and worth naming in the report)
- **§5.2 explicitly quarantines soft serving from the theorem**: "(empirical ε-isolation; **separate from the theorem above**)" (`:310`). That is the discipline I asked for at R1 — they just need to extend it to the LN axis. c9/10
- **The merge-subset ablation** (`:346`): "merging only {EN, ES} leaves DE forgotten while boosting included languages — recovery is retrieval, not generalization". A clean controlled experiment with a sharp negative
  reading; the parent paper's merge section (I27/I28) has nothing comparable and instead reaches for "human-like degradation". c9/10
- **"Within-phase knowledge is distributed, only phase boundaries are modular"** (`:348`), backed by magnitude-ranked pruning collapsing *worse* than contiguous-block pruning. This is a non-obvious mechanistic finding,
  it explains why random-scatter is the right prune, and it is the kind of result that would stand on its own. c9/10
- Internal numeric consistency held up everywhere I checked: `2.22–2.33 ppl` ↔ "within 0.09 nats" (`ln 2.33 − ln 2.23 = 0.04`) ✓; sequential-endpoint row (EN 11.08, DE 18.28, ES 2.13) ✓ consistent with N1's per-phase costs;
  and `0.84 nats/block × 18 blocks ≈ 15 nats` sits inside the abstract's "+12 to +20 nats over 19 phases" ✓ — the accumulation study and the per-block erosion rate agree across a 300-line gap. c9/10
## Round 4 — §6 accumulation, §7 observations, §8 limitations, §9 conclusion (L405–513)

### ★★ B-6 — the headline "+12 to +20 nats" is **arithmetically inconsistent with its own numbers** `[MATH]` c9/10
§6.1 Arm R (`:418`–`:420`): "EN drifts from 11.9 to 24.6 ppl (**+12.7 nats**) and ES from 2.1 to 22.8 (**+20.7 nats**) over 19 phases---exceeding the
0.3-nat falsifier by 10–70×". The two parentheticals are *exactly the differences of the quoted pairs* (`24.6 − 11.9 = 12.7`, `22.8 − 2.1 = 20.7`) `[FACT]` c10/10,
so they were computed by subtracting quantities labelled **ppl** and reporting the result as **nats**. Two possible repairs, with very different consequences:
- **(i) the values are nats**, mislabelled ppl → then Δ = +12.7 / +20.7 nats ✓, but absolute magnitudes collide with the rest of the paper: EN's specialist value is
  `2.26 ppl = 0.81 nats` (N4, `:203`) and ES's is `2.13 ppl = 0.76 nats` (Table `tab:pareto`), so "EN from 11.9" cannot be EN's starting loss; and a held-out byte-level
  cross-entropy of 24.6 nats means geometric-mean probability `e^{−24.6} ≈ 2·10^{−11}` per byte, i.e. ~`e^{19}` times *worse* than the uniform-over-bytes floor `ln 256 = 5.55` —
  possible for a confidently-wrong model but extreme enough to be worth stating explicitly `[MATH]` c8/10;
- **(ii) the values are ppl** (which matches the neighbouring bullets: "the most recent language is always at ~2.0 while everything else degrades to 20–80 ppl", `:427`, and
  P-Acq's "≤2.6 ppl") → then the true forgetting is **+0.73 nats (EN)** and **+2.4 nats (ES)**, the abstract's "+12 to +20 nats" overstates by ~5–17×, and "exceeding the 0.3-nat
  falsifier by 10–70×" becomes ~2.4× and ~8×. `[INFERENCE]` c8/10
Reading (ii) is more likely: it agrees with every other loss in the paper, and ppl differences mislabelled as nats is exactly how `24.6 − 11.9 = 12.7` would arise. Either way this is
the single most consequential defect I found — it is a headline number in the abstract (`:38`, repeated `:100`) and it gates their own pre-registered falsifier. **Ask for one units table
and recomputed deltas.** c9/10
- *Self-correction, recorded deliberately*: in Round 3 I praised the cross-check "`0.84 nats/block × 18 ≈ 15`, inside the abstract's +12–20". Under reading (ii) that coincidence dissolves
  (`0.84 × 18 = 15` vs true ~0.7–2.4). The praise was premature; the arithmetic only looked consistent because I checked it against the mislabelled unit. c9/10

### ★ C-4 — Arm G's own numbers contradict the mechanism sentence attached to them `[MATH]` c8/10
`:437`–`:445`: "Growth alone does not prevent forgetting. The model adds capacity each phase, but new neurons do **not protect** earlier languages---**weight updates still overwrite them**.
This is exactly what the theory predicts." Yet three lines earlier (`:442`) the same arm reports that prefix selection on that very model gives **10–19× improvement**, with `CS = 5.30`,
`DE = 3.03`, `EL = 2.96` vs unrouted `57.57` — i.e. masking the suffix recovers near-specialist behaviour. `[INFERENCE]` c8/10 If weights had been *overwritten*, masking could not recover
them; recovery is evidence that the old computation **survives inside the frozen-prefix subnetwork** and is merely diluted at serve time. That is a stronger and more interesting result than the
sentence claims, and it is precisely the paper's own thesis (interference ≠ overwrite, cf. their careful N1 wording "`representational overwrite` is our shorthand for the measured end-state", `:184`).
- Also "This is exactly what the theory predicts" overreaches: Corollary `cor:prefix` is a statement about **serving** a fixed map under masking; it says nothing about SGD dynamics in a grown model.
  The honest version is "consistent with", or better: Arm G shows that *the structure exists but is not used unless selected* — which is what the section title claims anyway. c8/10
- **R4 — was anything frozen in Arm G?** If old parameters were trainable, then exact prefix masking should be *violated* (N4 shows training erodes even with frozen old params), yet recovery looks
  near-specialist. Two possibilities worth separating, and both are publishable: (a) old-parameter drift was small because fresh capacity absorbs the gradient — i.e. **growth provides implicit weight
  isolation**, which would soften "growth alone does not prevent forgetting" to "growth protects weights but not computation"; or (b) recovery is approximate and the quoted 3.0 ppl is not comparable to a
  specialist's 2.2. A per-phase drift table (old block vs new block, relative Frobenius norm) settles it. c8/10

### C-5 — §9 claims to answer the question BDH defers; it answers an adjacent one `[INFERENCE]` c7/10
`:508`–`:513`: "The question the BDH paper defers---how fast synaptic state becomes durable change without catastrophe---receives here a structural answer". Pathway's deferred question is **state→weight
consolidation**: transferring attention state `σ` (their `corr`, `paper.tex:770`) into parameters (`paper.tex` §8.2: "short-to-long memory transfer from state to network weights ... As time goes by, the system runs out of
state space"). The fork's consolidation branch is *weight-space surgery plus replay* (merge → prune → finetune) — a legitimate and possibly better engineering answer, but it does not consolidate in-context state into
weights, and its own §5.4 says the remainder is closed by replaying real tokens. Suggested fix: claim the **systems** version explicitly ("durable change without catastrophe, achieved by separation rather than by a plasticity
rule") and say plainly that σ→θ consolidation remains open. Otherwise a Pathway referee reads a claim of having solved their open problem. c7/10

### §7 observations + §8 limitations — mostly exemplary
- The hedge "`'law' would overclaim their scope`" (`:463`) is the right instinct; O1–O5 are stated as regularities with directions and slopes rather than laws ✓.
- **O4 cross-checks against Pathway** `[MATH]` c9/10: "the 25M model reaches 97.4 % [sparsity] ... versus ~94 % at 100M" sits in the same regime as BDH's reported `ρ ≈ 5 %` non-zero (`paper.tex:1040`,
  i.e. 95 % sparse) — reassuring for implementation fidelity even where my primer concern (A/C-1) bites, and worth *pointing out* to the authors as evidence their model is in Pathway's operating regime.
- §8 is honest about scale (100M), corpus narrowness (Europarl-only across 19 languages), real-token replay, batch-1 routing measurements, and that expansiveness is a directional proxy not an operator norm ✓.
- **What §8 omits** `[INFERENCE]` c8/10: (i) the primer-vs-`eq:integral` deviation (A/C-1) — the biggest missing caveat in the paper; (ii) which axis LayerNorm reduces, on which exactness depends (R1);
  (iii) the size of the detection evaluation behind "100 %" (B-2); (iv) that most Table `tab:pareto` entries are single-seed (disclosed in the footnote but not in §8). Adding (i)+(ii) costs four sentences and removes
  the two easiest referee kills. c8/10
- Editorial: **"peak" is used both ways** — P-Acq "acquires at ≤2.6 ppl (peak 2.21)" (`:423`, peak = best) vs Arm G "acquisition degrades ... (peak 3.49 at LT, >2.6 threshold)" (`:438`, peak = worst). In a
  section whose verdicts are threshold tests, an ambiguous extremum word can flip a pass/fail reading. c9/10
## Round 5 — Appendix proofs (L637–734), prior art, reproducibility, negative register (L735–797)

### ★ Verified the parameter arithmetic exactly — and it confirms both papers agree `[MATH]` c10/10
Reproducibility (`:766`) states 100,925,440 params with `d=512`, `n_h=8`, `L=6`, multiplier 128, block 512, byte vocabulary. Taking total neurons
`n = 128 × 512 = 65,536`: `3·n·d + 2·|V|·d = 3·65536·512 + 2·256·512 = 100,663,296 + 262,144 = **100,925,440**` — an **exact** match to the quoted figure.
That is precisely Pathway's formula `(3+o(1))nd + 2Ωd` (`paper.tex:753`) ✓✓, so the fork's parameter accounting is faithful to the parent even where its primer is not
(see A/C-1), and it pins down that "width ×m" means `m×512` total neurons.

### ★ C-7 — Arm G's growth rate is stated in the wrong units and cannot reproduce the experiment `[MATH]` c9/10
`:434` / `:770`: "width growing **+32 neurons/head per phase** (`×128→×708`, ~554M final params)". But `×128 → ×708` over 19 phases is `+30.5` *multiplier units*
per phase, and one multiplier unit = 512 neurons total = **64 neurons/head**, so the actual rate is ≈ **+1,950 neurons/head per phase** — the sentence understates growth by
~**61×**. Implemented literally, Arm G would end at ×134 (≈105M params), not ×708/554M, i.e. a nearly fixed-capacity run. The width and parameter figures are mutually consistent
(`554M / 100.9M ≈ 5.49 ≈ 708/128` ✓), so the error is isolated to the per-phase rate — but it is exactly the number someone would code from. `[INFERENCE]` c9/10

### A-5 — `thm:dissoc`'s witness may be **degenerate in this architecture** `[MATH]` c8/10
Appendix (`:649`): "every old-task trajectory shifts by `c` per level: `F'^L(x) = F_A^L(x) + Lc·1`". That identity needs `F_A(y + c·1) = F_A(y) + c·1`
(shift-equivariance), which is not assumed and not generally true. Worse for the witness, **LayerNorm is shift-*invariant***: `LN(z + c·1) = LN(z)` by construction (mean-centring),
so a constant injected before an LN vanishes — and their primer applies LN to the residual at every level (`:118`). The theorem's *conclusion* survives trivially (at depth 1,
`F'(x) = F_A(x) + c·1 ≠ F_A(x)`), but the displayed closed form does not, and the cleanest witness is the one they actually measured in N4. Fix: state non-implication at one level, cite N4 for the architectural case. c8/10

### A-6 — `prop:soft`'s proof does not exclude **signed** gates; I re-derived its counterexample `[MATH]` c8/10
Appendix dynamics (`:683`–`:693`): `f'_1 = σ(x₁) + εx₂`, `f'_2 = δσ(x₁) + γx₂`, update `h ← h + g⊙f(h)` from `x₀ = (a,0)`, requiring the state to return to `(a,0)`.
Their two-step conditions are correct as written. But the step "the bracket is positive for `a>0`" needs `γ ≥ 0` **and** `g₂ ≥ 0`; without it, the second component's bracket is
`a[1 + relu(1+g₁) + γg₂]`, which can be made zero. Concretely with `γ = 1`: `g₁ = 0, g₂ = −2` gives exact preservation for **all** `a > 0` (`x₂[₂] = (−2)δ[a + a − 2a] = 0`, and
`x₂[₁] = a(1+0) + 0·(...) = a` automatically) — an exact, non-hard gate pair. Separately: reducing their first-component condition using ReLU's positive homogeneity gives
`(1+g₁)² + g₁g₂εδ = **1**`, whereas the main text states `= 4` (`:263`) `[FACT]` c9/10 — so §5 and Appendix `app:proofs` disagree by a factor in the normalisation.
Repair that makes the result *stronger and cleaner*: add `g ≥ 0` to the hypothesis (exactly BDH's positivity, `paper.tex:729`, licenses it) and state "with non-negative gates, exactness forces the hard endpoint".
Then the proposition becomes a genuine statement about BDH-class models instead of an unproved universal. c8/10

### A-2 refined (honest update) — the appendix closes my off-by-one by strengthening the hypothesis `[MATH]` c9/10
`:657` proves (⇒) under "for all `x∈X_A` **and all `L`**", and picks `L = ℓ*+1`. That is correct. But §5's statement (`:228`) fixes `L` and quantifies `0 ≤ ℓ ≤ L` with
`T_A = ∪_{ℓ≤L}`, which leaves the top-depth counterexample I found. So this is a **statement-scope mismatch between §5 and Appendix**, not an error in the appendix: make §5 say
"for all depths" and define `T_A` as the union over all depths. Similarly, "Together they satisfy the criterion restricted to `H_A`" (`:671`) needs one line — for `z ∈ T_A`,
`F_A(z) ∈ T_A ⊆ H_A` by definition of reachability, which is what lets `P_A F_A(z) = F'(z)` lift to full equality. c9/10

### A-3 confirmed — `lem:zf`'s proof has exactly the cancellation gap I predicted `[MATH]` c8/10
`:677`: "Preservation requires the sum term to vanish identically; any coordinate with a nonzero `f_b` component pins `g_{b,i}(x)=0`." From `Σ_b g_{b,i}f_{b,i} = 0` you cannot pin individual
terms. One hypothesis fixes it and they already have the ingredients: non-negative gates *and* non-negative write-backs (BDH positivity), or a single suffix block, or disjoint supports. Also
`"f_b(b)-terms only"` (`:676`) is garbled — retype that line. c8/10

### app:meas — exemplary, with one number to tighten `[FACT]` c9/10
They publish the raw per-level deviation grid (gate scale `d` × level `L₁..L₆`) and draw a carefully bounded conclusion ("contraction is unsupported ... equally, erosion is not runaway-exponential
... approaches full decorrelation ≈0.9 and stays there. No universal impossibility is claimed") ✓. But "mild per-level amplification (~1.2–1.5×)" understates their own early levels: from the table,
`d=0.15`: `.0053→.0100→.0191` = **1.89×, 1.91×**; `d=0.30`: `.0214→.0468→.0817` = **2.19×, 1.75×** `[MATH]` c9/10 — the range fits later levels only. Say "after level ~3" or give per-level ratios.
Also relevant: saturating amplification is *consistent with* N4's near-additive +0.84 nats/block ✓ (independent cross-check between §5.2 and Appendix).

### D — prior-art triage: solid table, four lines missing `[FACT]` c7/10
The table (`:739`) covers progressive/PackNet/HAT/EWC/OGD-GPM-A-GEM/hypernetworks/adapters/MoE/highway/modular-DIB and ends with an honest "We did not identify prior work that formulates forward-path isolation
under this precise combination". Missing, in descending order of closeness:
1. **Function-preserving network expansion** — Net2Net (Chen et al., ICLR 2016) and Network Morphism (Wei et al., ICML 2016) *are* "append capacity so the old function is exactly preserved"; Corollary `cor:prefix`'s
   hard-selection exactness is a special case in a depth-recurrent setting. **Verified by search** ✓ high confidence this will be the first thing an external referee cites.
2. **Frozen-module growth + normalisation-statistics coupling** — PCANets (Aljundi et al., CVPR 2019) and "Modularity with Invariance" (Rusu et al., NeurIPS 2019) build exactly *invariance of frozen blocks under growth* plus the
   observation that normalisation statistics are what break it when new units join. That is their `cor:prefix` soft-case sentence, with a name and prior experiments. *(Cited from my knowledge of the literature — authors should verify exact claims.)*
3. **Trunk-based growth and "model bloat" in depth-recurrent CL** — Riemer et al., ICLR 2021 ("The Effect of Recurrent Embeddings on Continual Learning in Non-Convex Sequential Structure") studies per-task growth into a *recurrent*
   trunk and reports that growth without the shared structure does not prevent forgetting. If so, Arm G's "critical finding" has direct prior art — and their consolidation branch (merge into one artifact) is the same move. *(Same caveat: verify.)*
4. **Invariant-subspace / frozen-trunk subnetwork selection** — GPM is cited ✓ but the forward-path variants (latent-sparsity subnetwork selection, e.g. SupSup; invariant-subspace methods like LSNet) are the closest relatives of
   `cor:sel`'s "gates select structure already present". *(Verify.)*
None of these is fatal — the combination (depth-recurrent + byte-level LM + BDH growth + exactness criterion) may well be new — but "we did not identify" needs to become a searched claim with the search described, and #1 in particular should be engaged because it reframes `cor:prefix` as a morphism result. c7/10

### C-8 — small internal tension worth one sentence `[FACT]` c8/10
Negative register #5 (`:792`): "language identity is not linearly extractable from per-token deep residuals" vs §5.2's 100 % detection with a *pooled-embedding* logistic regression. Not contradictory (per-token residual ≠ pooled embedding),
but as written they read like opposite findings; state which representation carries the signal — it is also evidence for their own `cor:sel` thesis and worth making explicit.

### Reproducibility & register — strong practice, say so `[FACT]` c9/10
Exact parameter count; hyperparameters with seeds `{1337,1,2}` limited to the pipeline they actually cover; "all other runs are single-seed as marked"; pre-registered falsifiers; pointers to `docs/reports/*` for intermediate tables and to `scripts/`
for mechanisms; a 7-item negative-results register with numbers (magnitude-ranked pruning: 15–16 vs 6.5–7.2 ppl at 25 % keep). This is better than most published CL work and much better than the parent paper's reporting (I27/I28/I29). c9/10
