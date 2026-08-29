# Task 1 — Notes: *The Dragon Hatchling* (BDH), Pathway. arXiv 2509.26507

> Provenance: written by seat Pi (backend: Qwen3.8-Flash-Next, local 4090, blind run — no J-Space skill, no team context). Committed by Quinn after audit 2026-08-29; line refs verified against rev2 @ cb72631.

Source: `~/bdh-review/paper-src/paper.tex` (35 583 lines-of-prose ≈ words). Tags: `[FACT]` = traceable, with
`paper.tex:LINE`. `[MATH]` = my own derivation. `[INFERENCE]`, `[GUESS]`, confidence `c/10`.

## Glossary (as the paper defines terms — authoritative per user)
| term | meaning | ref |
|---|---|---|
| **BDH** ("Dragon Hatchling") | LM architecture given by local distributed **graph dynamics**; all parameters = topology + weights of a communication graph; inference state = **edge reweighting** of that graph | `:396`, `:404` |
| **BDH-GPU** | tensor-friendly *special case* of BDH: the n particles communicate by **mean field ("noisy radio broadcast")** instead of "communication by wire"; state localised in short vectors at neurons rather than on edges | `:700`, `:418` |
| neuron / synapse | the n nodes / m edges of that graph. Edges carry three roles: state, parameters, communication | `:300` |
| fast weights | the evolving ruleset σ = inference-time state (vs fixed ruleset G = trained parameters) | `:294` |
| ReLU-lowrank | one of BDH-GPU's two block types (the FFN), in high dim n, positive activations | `:432` |
| linear attention | BDH-GPU's other block, same dimension n | `:432` |
| monosemantic synapse | a neuron–neuron link whose in-context state localises consistently across prompts for a given feature | `:370` |

## §1 Introduction — the argument structure `[FACT]` c9/10
1. **Empirical gap:** SOTA models (incl. CoT) do not length-generalise reasoning beyond trained sequence
   lengths (`:213`, cites shojaee2025). So long-horizon agentic use is extrapolation without warranty.
2. **Structural gap:** brain = uniform scale-free graph system, n≈8·10¹⁰ neurons, m>10¹⁴ synapses; a tensor
   LM has function defined on vectors, not on particle dynamics. Direct simulation via generic Turing
   reductions "would require billions of CoT tokens per single reasoning step in the brain" (`:215`).
3. **Their move:** don't simulate — build an architecture whose *micro* description is local graph dynamics
   and whose *macro* description is attention+FFN, so the two are the same system read at two scales. That
   identity is the advertised "bridge between Transformer and brain models" (`:223`, `:480`).
4. **Safety framing:** scale-freeness + a "thermodynamic limit" of the model family is offered as the route
   to *foreseeable* long-horizon behaviour, i.e. small tests validating large deployments (`:256`–`:268`).

## §1.2 The design kernel — this is the part that matters most `[FACT]` c9/10
Two rules over n facts, on a ruleset σ:
- *modus ponens*: `X(i), σ(i,j) → A(j)`, contribution `X(i)σ(i,j)` — eq (1) `:282`
- *Hebbian update*: `Y(i), X(j) → σ(i,j)`, increment `+Y(i)X(j)` — eq (2) `:290`

Then the load-bearing design principle `[FACT]` `:294`: **parameter count and state count should be
comparable**. A system with n facts has m = O(n²) trainable entries; an LSTM keeps only O(n) state; a
fast-weights system keeps O(n²) state. They call the 1-1 ratio "important" and say it "may justify the
success of the Transformer". Sparsity is then chosen so that **n ≪ m ≪ n²**, which converts the dense
matrix into a graph with m edges carrying state, parameters, and communication simultaneously.

`[INFERENCE]` c8/10: this parity principle *is* the reason BDH-GPU can be read as having state on synapses —
state lives in the same index space (n×n) as the parameters. That has a direct consequence for continual
learning which I expect the fork to exploit or inherit: **parameters and state are not separable by index
set**, so "freeze the old weights" does not freeze old *computations* — the state carried on the same
edges is shared. Consistent with the manuscript's headline "weight isolation does not imply computation
isolation". To verify against §2–3 formalism, not just as vibes.

## §1.3 Claims to hold them to `[FACT]` c9/10
- BDH-GPU(n,d) with `log n < d ≪ n`, d = 256 in practice; **(3+o(1))nd parameters**, O(d) per particle `:428`.
- **Equivalence claim:** for any BDH-GPU model with n particles there is a BDH model with same inference
  behaviour and same O(nd) parameter count, "formally equivalent **up to placement of Layer Norms**" `:434`.
- Architecture = ReLU-lowrank FFN + linear attention in the *same* high dim n; activations positive by
  design, ~5 % sparse `:432`, `:364`.
- Scaling laws match optimised GPT at 10M–1B on all tested next-token tasks (incl. translation) `:358`.
- **Concatenation of models is again a model in the same architecture**, with larger n → merging by
  concatenation, validated empirically for translation (§7.1, `:1489`) `:470`.
- Emergence (modularity, sparsity, monosemanticity) obtained **without L1 regularisation**; attributed to
  dimension choice + linear attention in high dim + ReLU positivity `:376`.

## §1.5 The gap our fork targets — explicitly admitted `[FACT]` c9/10
They claim to explain the predominant dynamics from "split-second" to "minutes" scale, and state plainly
that they **do not** answer how fast-weight inference state transfers to long-term memory (starting ~10³–10⁴
tokens, with feedback signals): "In this work, we do not provide a direct answer as to how the brain
actually handles this effect at longer timescales" `:388`. They only say a constructive route "seems less
challenging once the local inference dynamics are better understood".

`[INFERENCE]` c8/10: so BDH-CL is filling a hole the parent paper names and declines. That is a legitimate
gap, and it means our manuscript's novelty claim should be judged against *continual-learning* literature,
not against this paper — Pathway did not attempt it.

## Candidate issues (to check later, not yet verdicts)
- **I1 — "up to placement of Layer Norms"** (`:434`). LN placement is not a no-op: it changes which
  quantities are scale-invariant and thus the function class. An equivalence theorem whose hypothesis is
  "modulo moving LN around" needs to show the moved-LN models are *identical* on the relevant input set, or
  the word "equivalent" is doing more work than proved. → check Claims `claim:graphs` (`:1925`) and
  `claim:att_equiv` (`:1949`), plus §3.4 (`:927`). Relevance to fork: our manuscript's architecture primer
  inherits whatever BDH-GPU actually is. c7/10
- **I2 — notation collision around σ, A, X/Y** `[FACT]` c8/10: σ is both the evolving ruleset (`:282`) and
  generic state σ(t) (`:392`); 𝒜 is the architecture (`:392`) while `A(j)` in eq (1) is a *belief* (`:282`);
  and eq (1) writes the source belief as `X(i)` while eq (2)'s Hebbian pair is `(Y(i), X(j))` — if σ(i,j)
  strengthens from pre-synaptic i and post-synaptic j, the two equations should use the same symbol for the
  same slot. Small, but it is exactly where a formal reading can go silently wrong. c6/10
- **I3 — graph orientation convention** `[FACT]` c7/10: they define `H(i,j) := ⟨e_j|H|e_i⟩` as the weight
  "from i to j" (`:392`–`:394`) — i.e. transposed relative to standard matrix indexing — and then state
  `H = H₂H₁` means "first H₁, then H₂". Must verify this convention is used consistently in the equations of
  motion (§2.2, `:458`) and in the PyTorch listing (`:2065`); a transposed adjacency silently reverses all
  message passing. c7/10
- **I4 — parameter accounting assumption** `[FACT]` c8/10: sparse-graph cost is O(m(b+log n)) bits, then they
  *assume* `log n = O(b)` to declare it O(m) parameters (`:394`). Since the paper's headline numbers
  ((3+o(1))nd) and its scaling-law comparison both hinge on parameter counts, the assumption should be
  stated where the counts are claimed, not only in notation. c6/10
- **I5 — safety argument is aspirational** `[INFERENCE]` c7/10: "models M_n ~ P_A(n); if the limit P_A exists
  under an appropriate, well-defined sense of uniformity, then behaviour is foreseeable" (`:264`) never
  defines that sense in §1, and the cited criticality results are for graph-based interacting particle
  systems, not trained LMs. As motivation it is fine; as a safety argument it is currently a hope. Check
  whether Appendix `appx:one` (`:1758`) supplies anything formal. c6/10
- **I6 — emergence attribution without ablation** `[INFERENCE]` c6/10: "no L1 was used, therefore the effects
  follow from the design" rules out confounds only with ablations (e.g. dense-attention or non-negative
  variants at matched scale). Look for them in §5–§6 (`:1086`, `:1335`). c5/10

## Ledger
- 2026-08-29 ~18:20 read L211–400 (intro, motivation, modus-ponens intuition, contribution, notation). Notes above.

## §2 BDH as local graph dynamics (paper.tex L401–595) — the core formalism

### The machinery `[FACT]` c9/10
- **Architecture = scheduler + kernel.** Computations happen *only at the n neuron nodes*; state may live on
  nodes *and* edges. Synchronous scheduler, two sub-rounds (local computation, then "over wire"
  communication). To forbid hidden time-counters inside neurons, the kernel is a **4-tuple**
  `K(A)=(K1..K4)` executed round-robin at rounds `r ≡ i mod 4` (`:405`–`:412`).
- **Interaction kernel (Def. `def:chemistry`, `:424`)** — z species, non-negative state q, rule weights
  r_ijk and damping d_k:

  `[MATH]` **q'_k = (1−d_k)·q_k + Σ_{i,j} r_ijk q_i q_j**   (eq `eq:chemistry`)

  i.e. the explicit Euler step of `dq_k/dt = −d_k q_k + Σ_ij r_ijk q_i q_j`: a **quadratic (pairwise) map
  with leak**. State is explicitly *non-normalised* (`:437`). c9/10
- Spiking realisation: for independent Bernoulli signals, the AND of spikes has expectation q_i q_j — so the
  quadratic term is exactly implementable by stochastic 0/1 dynamics (`:452`). Nice, and it is what makes the
  "brain-like substrate" claim non-vacuous.
- **Known special case:** restricting the output index to `k ∈ {i,j}` gives *replicator dynamics*
  (Lotka–Volterra) — parameters on edges, state on nodes (`:462`). BDH deliberately goes beyond it:
  **state lives on edges and is therefore larger than the number of neurons** (`:464`).
- **Edge-reweighting kernel (Def. `def:edgereweight`, `:472`):** every non-zero rule is either a
  *computational* rule on one node, or a *communication* rule touching exactly `{node i, node j, edge (i,j)}`.

### BDH itself `[FACT]` c9/10
Parameters are **five graphs on the same n nodes**: `G_x^ee, G_x^ii, G_y^ee, G_y^ii, G_s` (`:482`) — ee = excitatory,
ii = inhibitory; `G_s` carries the fast-weight state σ. Node state: `X(i), Y(i), A(i)` plus temporary
integrator counters `X^ee, X^ii, Y^ee, Y^ii`; per-edge damping `u(i,j) > 0` (`:489`).

The "equations of reasoning" (Table `tab:protocolx`, `:527`–`:560`), one **layer = 4 rounds**, L layers
(L = 8 typical), so **one token = 4L rounds**; the *same* parameter graphs are used at every l — that is the
weight-sharing-across-depth property our manuscript calls the defining efficiency:

| round | rule (simple version (a)) | reading |
|---|---|---|
| `4l` | `X(i), σ_l(i,j) → A(j)` ; `σ_l ↓_{1−u(i,j)}` | read the **fast-weight memory**: `[MATH] A(j) += Σ_i σ_l(i,j)X(i)`; σ decays geometrically, time constant ~1/u |
| `4l+1` | `Y(i), X(j), G_s(i,j) → σ_l(i,j)` ; `Y(i) ↓` | **Hebbian write, gated by a trainable parameter** `G_s(i,j)`: plasticity rate itself is learned |
| `4l+2` | `A(i), X(j), G_y^ee(i,j) → Y(j)` | inference from **parameters** (excitatory) |
| `4l+3` | `Y(i), G_x^ee(i,j) → X(j)` | inference from parameters → next X; readout also through `X` at round 4L (`:560`) |

So the intro's eqs (1)/(2) are literally rounds `4l` and `4l+1`, and the closed loop
`X → A → Y → X` over 4 rounds is one depth step. `[INFERENCE]` c8/10: attention-like behaviour comes from
rounds 4l+2/4l+3 (parameter-driven), while the *in-context* memory is round 4l + the σ write — i.e. BDH-GPU's
σ is a leaky linear-attention fast-weight memory, which is why §1's parameter↔state parity argument works.

### Resolved: I3 (orientation convention) — it IS self-consistent `[MATH]` c8/10
They define the weight "from i to j" as `H(i,j) := ⟨e_j|H|e_i⟩ = H_{ji}` (standard row-major). Then
`(Hv)_j = Σ_i H_{ji} v_i = Σ_i (weight from i→j)·v_i` — incoming-edge sum, exactly right for message passing;
and `H = H₂H₁` means apply H₁ first. **Convention is coherent.** Consequence to keep watching: in code, the
message-passing product must be the *transpose* of a naive `W @ x` reading if one writes weights as
`weight[i][j] = i→j`. That is a classic place for a fork to silently differ → check against the PyTorch
listing at `:2065` when I get there (cheap, worth it).

### New candidate issues
- **I7 — state-size bookkeeping** `[FACT]` c8/10: the Definition of BDH claims `O(n + |E(G_s)|)` state
  variables (`:484`), but the protocol uses a **distinct σ_l per layer** (`σ_l`, `0 ≤ l < L`, `:505`) → actual
  in-context state is `O(L·|E(G_s)| + n)`. They note the distinction "serves to facilitate interpretation" and
  that a single shared σ "does not fundamentally change the operation and scaling laws". Since §1.2 sells a
  parameter↔state *parity principle*, the honest count should carry L. Minor, but it is their own headline
  metric. **Relevance to our fork: whether σ is per-layer or shared determines what "growing" or "routing"
  even isolates.** c7/10
- **I8 — the dynamical system is not fully specified** `[FACT]` c8/10: two pieces are deliberately left open —
  (i) thresholding `relu(A−B)` as an unspecified "computational primitive" (`:563`), and (ii) **how X(i) is
  reset between tokens** ("The definition of the protocol does not specify how variable X(i) should be reset",
  `:570`). For a paper whose thesis is *axiomatic* micro-foundations and limit behaviour, leaving the
  token-boundary update rule unspecified means the asymptotic dynamical system is not uniquely defined —
  different resets give different limit sets. Legitimately pragmatic for GPU training; but it undercuts the
  "thermodynamic limit" ambition of §1.4 unless §3 pins it down. c7/10
- **I9 — positivity needs d ≤ 1** `[MATH]` c8/10: `q'_k = (1−d_k)q_k + Σ r q q` keeps the non-negative cone
  invariant only if `d_k ≤ 1` (else the leak can overdraw). The Definition allows any `d_k ∈ R^+` (`:424`) and
  says "assuming q_i,q_j,r_ijk ∈ [0,1]" only for the *interpretation* (`:437`). Same for BDH's `u(i,j) > 0` —
  stated without an upper bound (`:489`). Check whether §3 (BDH-GPU) constrains u; if not, "positive by design"
  (§1.3) is conditional on a constraint that is never written. c6/10
- **I10 — positioning** `[INFERENCE]` c6/10: rounds `4l`/`4l+1` are a leaky outer-product associative memory,
  i.e. the classic fast-weight / linear-attention-memory line (Hinton 87 and Schmidhuber are cited in §1.2, so
  they know). Whether the *graph-dynamics* reframing is the novel contribution or the memory itself matters for
  our own related-work honesty — our manuscript should not imply BDH invented Hebbian fast weights. c6/10

## Ledger (cont.)
- ~18:35 read L401–595 (§2 formalism, interaction kernel, edge-reweighting kernel, equations of reasoning).
  Resolved I3; added I7–I10. Next: L596–700 (attention as micro-inductive bias, oscillator toy model, brain models).

## §2.3–§2.5 Interpretations: logic, oscillators, brain substrate (L596–699)

### What σ *is*, semantically `[FACT]` c9/10 — matters a lot for the fork
An attention-state entry `σ(i,j)` is explicitly **not a logical value** but "an inductive bias associated with
how likely the system is to consider the implication `i→j`" (`:608`). The rule shape is modelled on the
distribution axiom `(X→(i→j)) → ((X→i) → (X→j))` (`:604`). Utility semantics come from replicator dynamics /
evolutionary game theory: "neurons which win in the natural selection process are added to the activation Y"
(`:610`, fn.). Chains of implications guide activation along paths in `G_x^ee, G_y^ee, corr` (`:614`).

`[INFERENCE]` c8/10: because σ is a soft bias rather than a discrete fact, *no reweighting of σ can create a
hard partition of computation* — any change to σ perturbs everything downstream continuously. That is exactly
the shape of the fork's "soft gates can never be exact" result, so that theorem looks like a property of this
architecture's state semantics rather than of depth-recurrent models in general. Keep as a novelty-scope check
in Task 2 (see H1 below).

### Oscillator toy model (§2.4) `[FACT]` c9/10
n particles on a circle; `G_s` = elastic connectors (may be dense); **slow** state σ(i,j) = tension/displacement
that relaxes; **fast** node pulses x, y. Mapping table at `:628`: parameters = "wires, prods, elastic
connections" (`G_x` wires, `G_y` prods), σ = displacement of connectors, sparse vectors = pulses/state correction.
- Relaxation variants are identified with **position encodings**: damping ↔ ALiBi, spring/oscillator ↔ RoPE (`:643`).
  Neat, and checkable against §3 — if true it means PE is not bolted on but a limit case of state relaxation.
- State propagates to **3-hop neighbours** (i → j → i′ → connector (i′,j′)) (`:657`), which they then use to argue
  that backprop only needs to follow short dependency chains (`:668`).
- Hebbian write fires on *temporal adjacency* of pulses (y(j′) then x(i′)), "even if there was no direct
  causality"; they note requiring **`G_s ⊆ G_x`** brings it closer to a causal effect (`:653`). ← structural
  constraint worth remembering: the synapse graph is meant to sit inside the wire graph.
- Supervised learning (feedback signals) is explicitly **deferred**: "selective transfer and re-encoding of
  gradients from state into weights, at longer time scales" (`:687`), same deferral as §1.5.

### Brain-substrate claim (§2.5) `[FACT]` c9/10
Observation `obs:brain`: BDH's ruleset is expressible from positive activations + Hebbian learning + excitatory
& inhibitory circuits with thresholding; round-by-round mapping at `:674` (4l+2/4l+3 = integrate-and-fire with
local replicator competition; 4l+1 = stochastic AND-gate potentiation; 4l = strengthened-synapse transmission + decay).
Then **Finding**: "The Hebbian learning mechanism is *plausibly needed*, and in combination with neural circuits
*sufficient*, for performing the reasoning function at the scale of the brain" (`:681`).

### Candidate issues (cont.)
- **I11 — criticality asserted, not shown** `[FACT]` c7/10: "By adjusting the frequency of updates, the system can
  be made to operate **exactly at the critical point** ... giving the time between updates of a connection pair a
  heavy power-law-like tail distribution" (`:664`). No measurement or mechanism given in §2. This is load-bearing
  for §1.4's "small tests validate large deployments" safety argument (I5) — self-organised criticality *claimed*
  is not the same as *demonstrated*, and the cited rigorous results are for other particle systems. → look for any
  empirical tail-distribution measurement in §5–§6; if absent, I5+I11 merge into one substantive criticism of the
  foreseeability story (of the parent paper, and of anything our fork inherits from it). c7/10
- **I12 — the kernel is contingent, not derived** `[FACT]` c9/10: they state plainly that this kernel was chosen
  because "we found it to work well, and we knew how to train BDH models which implement it on GPU (which ... made
  it 10²–10⁵ times more cost- and time-effective...)" (`:670`), and pose optimal-kernel search as open. Honest and
  refreshing — but it means architectural-necessity phrasing elsewhere ("BDH's architecture *requires*…", "the
  defining efficiency") should be read as *this kernel*, chosen for engineering reasons. Relevant to how our fork
  words generality claims. c8/10
- **I13 — "plausibly needed" from an emulation result** `[INFERENCE]` c7/10: `obs:brain` is an upper-bound /
  emulation statement (Hebbian circuits suffice to implement BDH). Sufficiency does not support "needed"; that
  would require a lower bound or at least ablations showing reasoning degrades without Hebbian plasticity. As
  written (`:681`) the Finding overreaches its own Observation — same failure mode I'll be grading our manuscript
  for, so worth recording as calibration rather than outrage. c7/10
- **E1 (editorial)** `[FACT]` c8/10: "Kuromato coupled oscillators" (`:632`) — the standard name is **Kuramoto**.
  Also `fallow from` (`:618`) should be *follow from*. Trivial, but they belong in a copy-edit pass.

## Handoff list for Task 2 (running; checks to carry into the manuscript review)
- **H1** Isolation must be *structural*, not gating — consistent with σ being soft bias (`:608`). Check whether the
  fork claims generality beyond BDH's specific kernel given I12. 
- **H2** Consolidation = transfer of state σ into weights G is the *named open problem* in the parent paper
  (`:388`, `:687`). Our merge→prune→replay branch is a concrete answer to it → strong positioning; verify the
  manuscript says this explicitly (it should claim the gap, not just fill it).
- **H3** `G_s ⊆ G_x` (`:653`) and per-layer σ_l vs shared σ (§2, I7) are the two structural facts that determine
  what "grow"/"route" can isolate. Check the fork's architecture primer against both.
- **H4** Depth = recurrence of the same 4-round kernel with shared parameters (`:505`, Table `tab:protocolx`) — the
  premise of "computation isolation" difficulty. Verify the manuscript's primer matches §3's BDH-GPU equations
  (next chunk) rather than the toy model.

## §3.1–§3.2 BDH-GPU: the actual architecture (L701–892) — most important section for our fork

### Setup `[FACT]` c9/10
- `BDH-GPU(n,d)`: **n → ∞ is the only asymptotic basis**, with `n ≫ d > C log n`. R^d vectors are "(fuzzy)
  addresses of a virtual memory space of size n", which is why `d = Ω(log n)` cannot be dropped (`:716`).
- Three steps from BDH to GPU form (`:705`–`:711`): (1) factor `G_x, G_y` as **low-rank** products + ReLU, never
  materialised; (2) never materialise σ — access it through **linear attention** on a low-rank value
  representation; (3) LayerNorm every state variable. The intermediate object before LN is named **BDH-Normfree**.
- LayerNorm is **non-parametric**: `LN(z*) = (z* − 1·E_d z*)/std_d z*` — no learnable γ, β (`:722`). Consistent with
  the parameter count, but unusual enough that any fork adding affine LN params silently changes the counts.
- Parameters: `E ∈ R^{d×n}`, `W_x, W_y ∈ R^{n×d}` → **3nd + 2Ωd = (3+o(1))nd** (`:745`), where 2Ωd is token
  encoder/decoder. **Same E, W_x, W_y at every layer l=1..L** → depth is pure weight-shared recurrence (confirms H4).

### The equations `[FACT]` c9/10 — eq `eq:integral`, `:735`–`:742`
```
(1)  x_{t,l}  := x_{t,l-1} + relu(W_x v_{t,l-1})
(2)  yKV_{t,l}:= Σ_{τ<t} v_{τ,l-1} ⟨x_{τ,l} , U^{t−τ} x_{t,l}⟩        (U = rope/ALiBi, diag or block-diag)
(3)  xy_{t,l} := relu(W_y LN(yKV_{t,l})) ⊙ x_{t,l}
(4)  v_{t,l}  := LN(E xy_{t,l})
```
with state form `state_{t−1,l} = Σ_{τ<t} |v_{τ,l−1}⟩⟨x_{τ,l}| U^{t−τ}` (eq `eq:kvstate`, `:753`), so
`yKV_{t,l} = state_{t−1,l} x_{t,l}`. `[MATH]` c9/10 — I expanded the bra/ket: line (2) is a scalar inner product
per τ multiplying v_τ, i.e. exactly a linear-attention read of an accumulated outer-product memory whose keys are
x and values are v, with U^{t−τ} as the positional operator. c9/10

BDH side (eq `eq:bdhgraph`, `:795`): `corr_{t,l} := (corr_{t−1,l} + (|xy_{t,l−1}⟩⟨x_{t,l}| ⊙ G_s)) U`;
`x := x + relu((G_x^ee − G_x^ii) xy)`; `xy := relu((G_y^ee − G_y^ii) corr_{t−1,l} x) ⊙ x`. BDH-Normfree (`:840`)
drops `⊙ G_s`, sets `state = E·corr`, and removes the LNs.

### `[MATH]` Structural consequences I can derive myself (and that matter for BDH-CL)
1. **Support confinement / x as implicit router.** Line (3) multiplies by `x_{t,l}`, so
   `supp(xy_{t,l}) ⊆ supp(x_{t,l})`: no state content can appear at neuron i unless x(i) > 0, *whatever the state is*.
   `[INFERENCE]` c8/10: routing in this architecture is therefore already carried by the support of x — which means
   a fork that adds explicit routing must either constrain x's support or explain why an extra gate is not redundant.
   → **H5** for Task 2: check whether the manuscript's "prefix routing" acts on x-support, on state columns, or on a
   new selector variable, and whether it accounts for this existing implicit gate. c8/10
2. **State is bigger than the parameters, by ~L/3.** Parameters are shared across depth (3nd), but `state_{t,l}` is
   indexed by layer: L copies of n·d → **L·nd ≈ (L/3)·(params)**, i.e. ×2.7 at L=8. `[FACT]`+`[MATH]` c9/10: this makes
   §1.2's "1-1 ratio of trainable parameter to state size" wrong by a factor of L/3 unless they mean per-layer
   (then it is 1:3). Concrete version of I7 — the parity principle survives only as an order-of-magnitude statement.
3. **The RoPE variant has no forgetting.** U = RoPE is a rotation ⇒ `‖U^{t−τ}‖ = 1`, so
   `state_{t,l} = Σ_{τ<t}` accumulates t rank-1 terms with no decay; the norm grows ~linearly in t and LN only
   rescales afterwards. Only ALiBi-style damping bounds it. `[MATH]` c7/10 → **I16**: does the paper anywhere bound
   state growth or discuss interference at long context? Relevant to BDH-CL because an ever-growing additive state is
   a *second* forgetting channel distinct from weight drift — and our manuscript's "computation isolation" story is
   about weights, not about unbounded state accumulation. c7/10

### Candidate issues (cont.)
- **I14 — ReLU is defined as a scalar max** `[FACT]` c9/10: `relu(z) := max_{i∈{1..n}}{0, z_i}` (`:720`) is the
  maximum *coordinate* of z — a real number — yet `relu(W_x v)` is added to the vector `x_{t,l−1} ∈ R^n` in
  eq `eq:integral`(1). Intended definition is coordinate-wise, `(relu z)_i := max{0, z_i}`. As printed the defining
  equation is type-incorrect. Small, but it is the one line a formal-reviewing referee checks first, and it appears in
  the *definition* block of the paper's central architecture. (Our manuscript should not inherit the typo.) c9/10
- **I15 — boundary condition incomplete** `[FACT]` c8/10: Definition `def:bdh` specifies inputs only through
  `v_{τ,0}` at layer 0 (`:743`). What initialises `x_{t,0}` (zero? `x_{t−1,L}`?) and `xy_{t,0}` is not stated — and x
  accumulates across l in eq (1), so its initial value at each token changes the function. Combined with §2's
  unspecified inter-token reset of X (`:570`, I8) this means **the paper's central dynamical system is under-specified
  by two independent degrees of freedom**, both of which are exactly the kind of thing that must be pinned down before
  any claim about limit behaviour, exact isolation, or reproducibility. Check the code listing (`:2065`) for what was
  actually implemented — that is the real definition. c8/10
- **I1 (refined) — two different hedges for the same equivalence** `[FACT]` c9/10: §1.3 claims BDH-GPU and BDH are
  "formally equivalent **up to placement of Layer Norms**" (`:434`), while Fig. `fig:ss`'s own caption says BDH-Normfree
  "**approximates** the inference dynamics of BDH-GPU" up to lack of LN (`:870`) — and the Normfree equations also
  **drop the `⊙ G_s` mask** entirely (compare `:795` with `:843`). So three separate concessions (LN placement,
  approximation vs equality, synapse-mask removal) are covering one claimed equivalence. Direction matters: BDH-GPU ⊂ BDH
  is plausible (take low-rank `G_x = W_x E`, which also preserves the O(nd) count), but as stated in §1.3 it reads
  stronger than the body supports. c8/10

### Handoff additions
- **H5** see (1) above — x-support is the architecture's native routing channel.
- **H6** State per layer, no per-layer parameters (`:745`, `eq:kvstate`) ⇒ in BDH-CL, "growing" a block means growing
  E/W_x/W_y (shared across depth) while state grows with L·nd. Any statement like "adding capacity without touching
  old computation" must respect that the *shared* parameter matrices are touched at every layer. c8/10

## §3.3–§3.4 Particle view + the BDH ⇄ BDH-GPU equivalence (L893–1002)

### Per-particle bookkeeping `[FACT]` c9/10
Particle i is described by `Z_i(t) = (state_i(t), E_(i,·), W_x(·,i), W_y(·,i))` (`:897`) — **O(dL) state + O(d)
trainable parameters per particle** (`:912`). Confirms my earlier count: state is per-layer, so total in-context
state is L·nd while trainable parameters are 3nd (I7's factor L/3). Uniform in n except for **k-tuples forced by U**:
k=1 for ALiBi, k=2 for RoPE (`:903`), hence scaling happens "in chunks of d·h·k = 512 parameters" at d=256, h=1 (`:884`).
Mean-field local program per particle (`:916`–`:921`): compute m_i ∈ R^d locally → broadcast → receive the *same*
Σ_j m_j everywhere → update activation for layer l+1 and state σ_i(t). Executed 3L times per token.

### Two facts I want to remember for BDH-CL
- **The attention read excludes the current token.** The sum in `yKV` runs over `τ < t`, and a footnote states this is
  deliberate: "Z_i(t−1) depends only on state_{t−1,l}, not state_{t,l} ... this is intentional" (`:928`). So BDH-GPU's
  in-context memory is strictly *past* context — no self-contribution at position t. `[FACT]` c9/10 → **H7**: a
  task-segmented or routed state (as in the fork) has to respect this; if routing writes and reads within the same
  token step, it deviates from the reference architecture's information discipline. c7/10
- **LN placement is justified empirically, not by theorem.** "Models generally do not train following BDH-GPU without
  any LayerNorm, but we observed empirically that there is some flexibility as to where these LayerNorms are placed"
  (`:908`). `[FACT]` c9/10 — this is the actual support for the "up to placement of Layer Norms" clause in §1.3 (I1).

### What `obs:equivalence` actually proves `[FACT]` c9/10
Formal equivalence "(i.e., the same model)" is stated for **BDH-Normfree** ↔ BDH under the *very* special parameter
choice `G_x^ee − G_x^ii = W_x E`, `G_y^ee − G_y^ii = W_y E`, **`G_s = 1^{n×n}`** (eq `eq:equivalence`, `:936`). So:
1. It is an equivalence with the **LN-free** variant; LN is simply absent on both sides, not moved. §1.3's wording
   ("BDH-GPU ... formally equivalent up to placement of Layer Norms") attributes to this Observation more than it states.
2. With `G_s` complete, BDH's state `corr` is **n² per layer** vs BDH-GPU's compressed `state = E·corr` at **nd per
   layer** — a factor n/d apart. They acknowledge it ("technical nuisance", `:973`) and close it with Claim
   `claim:att_equiv`: O(nd)-edge G_s suffices, "**subject to a natural preparation of attention values entering the
   attention block**" (`:975`). That clause is undefined here and carries the whole claim; proof deferred to
   Appendix `apx:proofattention` (`:1949`). → verify before repeating the equivalence anywhere. c8/10
3. They state plainly that BDH is **strictly more expressive** than BDH-GPU at equal O(nd) parameters, and that the
   converse fails ("an arbitrary G^ee does not admit an exact low-rank decomposition ... any low-rank decomposition
   introduces a form of noise", `:965`). `[FACT]` c9/10 → our fork implements the *tensor* side, so expressiveness or
   impossibility results proved for graph-BDH do **not** automatically transfer to what is trained. Scope check for
   any generality-flavoured theorem in the manuscript (pairs with I12).

### Candidate issues (cont.)
- **I17 — possible non-negative-rank gap in Claim `claim:graphs`** `[MATH]` c6/10, *needs appendix check*: the claim is
  that for any `D ∈ R^{n×d}`, `E ∈ R^{d×n}` there are neuron-neuron graphs `G^ee, G^ii ∈ G²(n,m)`, m = O(nd), with
  `G^ee − G^ii = D E` (`:985`). My own construction of the two-hop circuit works for *non-negative* factors — take H on
  V∪S (|S|=d) with edges v→s weighted by E and s→v weighted by D, then `H²[V] = D E` exactly `[MATH]` c9/10 — but trained
  D, E have **mixed signs**, and the formalism requires all edge weights non-negative (`:957`, positivity is what makes
  the spiking/token interpretation work). Writing `DE = M⁺ − M⁻` then demands each of `M⁺, M⁻` be representable as a
  2-hop non-negative circuit with O(nd) edges, i.e. **non-negative rank ≤ d** — and non-negative rank can vastly exceed
  ordinary rank. So either the appendix allows signed intermediate edges (which weakens the brain/positivity story), uses
  more than two graph layers / more intermediaries (changing the m = O(nd) or the "same model" claim), or the Claim is
  false as stated. Read `apx:proofgraphs` (`:1925`) before saying so out loud — this is exactly the kind of finding that
  is worth a paragraph in a review, and exactly the kind that is embarrassing if wrong. c6/10 pending check.

### Handoff additions
- **H8** "Same model" claims should be quoted with their hypotheses (`G_s = 1`, LN-free, low-rank factors). If our
  manuscript leans on BDH ⇄ BDH-GPU equivalence to justify working in tensor space, it must state the three concessions.

## Appendix `sec:apxjl` — the proofs behind the equivalence claims (L1884–1975)

### I17 RESOLVED — my non-negative-rank objection was wrong, the construction works `[MATH]` c9/10
I suspected Claim `claim:graphs` (`:985`) could fail because splitting a mixed-sign product `DE` into
non-negative parts generally requires large **non-negative rank**. I checked Appendix `apx:proofgraphs` (`:1927`)
and worked through it: the trick is splitting *both* factors and letting `G^ee − G^ii` absorb the sign algebra.
With `E'_{α,j}=relu(E_{α,j})`, `E'_{α+d,j}=relu(−E_{α,j})`, `D^ee_{i,α}=relu(D_{i,α})`, `D^ii_{i,α}=relu(−D_{i,α})`
(and swapped on the `+d` block), each hidden index pair contributes
`D_{iα}·relu(E_{αj}) + (−D_{iα})·relu(−E_{αj}) = D_{iα}E_{αj}` ✓. Summing over α gives `G^ee − G^ii = DE` exactly,
with `|S| = 2d` hidden nodes and `m = O(nd)` edges ✓. **Claim is correct.** Recorded so nobody (including me)
re-reports it — and as a reminder to check the appendix before writing "this looks false".

### New issues from the proofs
- **I18 — unproven strengthening** `[FACT]` c8/10: after giving an explicit construction whose synaptic-layer nodes
  have degree `n`, they note reducing degree to `O(√(nd))` costs `O(n√(nd))` edges (`:1941`) — and then assert, with no
  construction, "The bound on the number of edges needed to represent such a circuit remains `O(nd)`, **even when the
  circuit has constant degree**" (`:1943`). Those two sentences are in tension; the constant-degree version is the one
  that matters for the local-dynamics/brain story. As written it is an assertion, not a proof. (Not fatal — heavy-tailed
  hub degrees are *consistent* with their scale-freeness claims — but it should be labelled conjecture.) c7/10
- **I19 — what "sparse-graph BDH ≡ BDH-GPU" really costs** `[FACT]` c9/10: Claim `claim:attentionformal` (`:1949`) is
  *proved*, but read the construction: pick a fixed set `D ⊆ V`, `|D| = 2d`; `G_s` = all-ones on those 2d columns and zero
  elsewhere; values are written through a preparation `A` = immersion via `E'`. So the BDH model that reproduces
  BDH-Normfree has its **entire attention state living on 2d designated hub neurons, with values pre-encoded by the
  encoder** — i.e. it is exactly BDH-GPU's compressed state `E·corr`, rewritten as a graph. Two costs are stated in the
  appendix but absent from §1.3: (i) the distinguished 2d bottleneck, and (ii) "two successive layers of BDH with sparse
  state are sufficient to express **a layer** of BDH-Normfree" (`:1967`) — a depth cost of ×2 for exactness. `[INFERENCE]`
  c8/10: so the honest slogan is *BDH-GPU's compressed state **is** a sparsification of BDH's synaptic state, not an
  approximation of it* — and any claim that graph-BDH is "strictly more expressive at equal parameters" (true, `:947`)
  should not be quoted alongside "the two are the same model" without stating which direction has which hypothesis. c8/10
- **I20 — what the linear-attention Claim approximates** `[FACT]` c8/10: Claim `claim:linear` (`:1893`) shows linear
  attention expresses a block computing `a_t = q Σ_τ k_τᵀ v_τ` (eq `eq:attn2`, `:1903`) — **already softmax-free** — with
  `O(√δ)` L2 error, requiring (a) a random preparation f' with `f(q)·f(k) = φ(q,k) ± O(n^{−100})`, (b) comparable value
  norms `c₁ ≤ ‖v‖ ≤ c₂`, (c) context length `t < δn/((C+1) log n)`, and (d) **C-non-adversarial keys** (independence of
  key vectors except for C of them). Proof = JL distortion + Azuma on `O(C)` martingales (`:1919`). Legitimate, but the
  headline "linear attention expresses attention" should be read as "random-feature linearised *unnormalised* attention
  tracks kernel attention up to context length O(n/log n), for non-adversarial key streams". Also `[FACT]` c8/10: the
  random-features foundation is uncredited — **Rahimi & Recht: 0 occurrences** in the source; Katharopoulos (1) and
  Performer (1) appear only in the architecture-comparison section, Johnson–Lindenstrauss once. For our manuscript's
  prior-art triage this matters twice over: BDH's fast-weight memory line (Hinton/Schmidhuber/Ba — cited ✓) *and* the
  random-features line (uncited) are both prior art our own related-work section must handle honestly.
- **I21 — protocol equivalence is only proved for diagonal U** `[FACT]` c9/10: Observation `obs:protocol_equiv` is stated
  in general, but its proof says "The parameter u(i,j) ... follows from the definition of matrix U; **we assume for
  simplicity that U is diagonal (which corresponds to the case of ALiBi)**" (`:1889`). RoPE (block-diagonal, k=2) is the
  case used in practice per Fig. `fig:stack` (`:884`) — so the graph-protocol ↔ tensor-equations bridge is established for
  ALiBi and asserted for RoPE. Small gap, easy to fix by an index argument over 2×2 blocks; worth noting because our fork's
  claims about "the same computation" inherit whichever variant it runs. c8/10

### Meta-note for Task 2 (calibration, not criticism of our fork)
The parent paper is *dense with real content* and mostly honest about its concessions — but the §1.3 "Contribution" list
systematically states the strongest version of each result while the body carries the hypotheses (I1, I19, I20, I21). That
is exactly the failure mode I will grade our manuscript against: **are the headline claims quotable with their hypotheses
attached?** If ours are, that is a genuine quality advantage worth pointing out explicitly in the review.

## §4 Implementation, scaling laws, comparisons (L1003–1085)

### Facts worth carrying into the fork's mental model `[FACT]` c9/10
- **Weight sharing across depth is explicit and acknowledged**: "As in the Universal Transformer, all layers use the same
  set of weights" (`:1021`) — with the same acknowledged downside (FLOPS-per-parameter overhead). So BDH-GPU's depth
  recurrence is *recurrent-weight* by design; our manuscript's premise ("one block of parameters reused across all depths")
  is faithful to the source. ✓ H4 closed.
- **Heads are almost decorative**: "The role of heads is limited to a single parameter-free LayerNorm, normalizing outcomes
  of linear attention separately for each head"; optimal h smaller than Transformer (h=4) (`:1023`). → **H9**: any routing
  mechanism the fork adds must not be confused with heads; heads here do not create subspaces of the state, only per-head LN.
- **State is per layer and never shared in the vanilla architecture** ("Sharing of state between the L layers is not performed
  in the vanilla architecture", `:1027`), dimension n×d per layer ⇒ total **L·nd**, vs **3nd** parameters (`:1015`). ✓ I7 settled
  as fact; the *rhetoric* about it is what varies (see I7b).
- Sparse positive activations: ρ ≈ 5 % of n non-zero in a typical run, "This corresponds to the fraction of the state space read
  and updated for each token" (`:1040`) — i.e. per-token state traffic is ~ρ·nd, which is the mechanism behind their O(ndL) FLOPS
  claim being conservative (`:1067`). Consistent with my support-confinement observation (H5): sparsity of x *is* the routing channel.
- Empirics: BDH-GPU vs GPTXL on translation, models scaled **only via n** with everything else fixed; GPTXL scaled in embedding
  dim + layers and needed Dropout tuning (`:1051`). Truncated BPTT on 2048-**character** sequences, state carried between
  minibatches (BDH's S matrix ↔ GPTXL's 4096-entry KV-cache buffer).

### Candidate issues (cont.)
- **I22 — parameter dimensions are transposed between definition and figure** `[FACT]` c9/10: Definition `def:bdh` gives
  `E ∈ R^{d×n}`, `W_x, W_y ∈ R^{n×d}` (`:731`), which is what the equations require (`v := LN(E x)` with `x ∈ R^n`). The layer
  diagram caption says the opposite: "`E ∈ R^{n×d}` and `W_x, W_y ∈ R^{d×n}`" (`:1011`). One of the two is wrong, and given the
  transposed adjacency convention I already verified (I3), a reader implementing from the caption gets a model that cannot even
  multiply. Cheap to fix; embarrassing if it is what a reimplementation started from — **including ours.**
- **I7b — three different phrasings of one quantity** `[FACT]` c9/10: §1.2 "1-1 ratio of trainable parameter to state size"
  (`:294`); §3.4 "same size O(nd) of ... state per layer" (`:983`); §4 "maintains a large recurrent state **comparable in size with
  its total number of parameters**" (`:1032`). The arithmetic is one thing: state L·nd, parameters 3nd → ratio L/3 (≈3.3× at the
  paper's own L=10). Not a correctness problem for the architecture, but a consistency problem in how the *design principle* is
  stated — and it is precisely the kind of inherited phrase our manuscript should either quantify or avoid quoting uncritically.
- **I23 — "no notion of context length" vs their own O(n/log n) bound** `[INFERENCE]` c7/10: §4 advertises "There is no notion of
  context length in BDH-GPU, and consequently no hard bound on it" (`:1048`, repeated `:1032`), yet Claim `claim:linear` — the thing
  that justifies linear attention at all — is only valid for `t < δn/((C+1) log n)` with non-adversarial keys (I20). Plus every model
  was trained with truncated BPTT at 2048 characters (`:1051`). Architecturally unbounded ≠ statistically validated beyond the window
  they tested. Fair phrasing would be "no *architectural* bound; validated to X tokens". Our fork should adopt that discipline too,
  especially if it makes claims about long accumulation (manuscript §6 is exactly about many phases). c7/10
- **I25 — "state-of-the-art" vs parity evidence** `[FACT]` c8/10: the abstract calls BDH "a practical, performant state-of-the-art"
  model (`:188`) and §4 closes with "state-of-the-art performance that has been experimentally verified" (`:1084`), while the actual
  reported result is "**matches** the GPT Transformer at all model sizes we have evaluated" on translation, ≤1B parameters, against
  GPTXL (`:1051`). Parity ≠ SOTA. This is the strongest rhetorical overreach I have found in the parent paper so far, and it sits in
  the abstract — exactly where a reviewer looks first. c8/10
- **I26 — the headline parity result uses a variant defined only in an appendix** `[FACT]` c8/10: Fig. `fig:translation`'s own caption
  states that "BDH-GPU′ extends conditional gating of states and logits" and it is **BDH-GPU′** that "matches the GPT Transformer at all
  model sizes we have evaluated" (`:1051`). The prime variant is defined only in Appendix `sec:bdh_scaling_details`: "BDH-GPU′ adds
  xLSTM-like gating m[echanisms]" (`:1835`) — i.e. an extra recurrent gate not present in Definition `def:bdh`, and therefore not part of
  the architecture whose theoretical properties (§3.4 equivalence, §2 protocol) were established. The §1.3 Findings list does not flag that
  its scaling-law claim depends on it. `[INFERENCE]` c7/10: relevant to us twice over — (i) if our fork implements vanilla BDH-GPU, the parent
  paper's parity numbers may not transfer; (ii) xLSTM-like gating is *exactly* the kind of extra state channel that a continual-learning
  argument has to reason about, so if the fork inherits it, isolation claims must cover it. c7/10
- **I24 — priority claim on ReLU sparsity** `[FACT]`, verify before repeating c6/10: "The use of the ReLU gate as a systematic way to achieve
  sparse activation was, to our knowledge, first exhibited in [haziza2025]" (`:1080`) — a self-citation-adjacent priority claim. Sparse
  activations in ReLU networks have been studied for well over a decade; if the intent is "first systematically exploited to accelerate
  Transformer inference", that qualifier should be there. Flagged as *needs verification*, not as an error.

### Self-correction (mine, recorded deliberately)
- **I20 downgraded** `[FACT]` c9/10: I initially noted the random-features foundation of Claim `claim:linear` as uncredited. Checking §4's
  related work: they *do* cite the linear-attention line properly — Katharopoulos et al. (`:1063`), **Performer/FAVOR+** as the theoretical
  framework for positive-vector linear attention (`:1065`, and FAVOR+ *is* random features), tensor-product key preparation (Buckman et al.,
  `:1062`), Retentive nets (`:1065`). What is missing is only the classical Rahimi–Recht origin of random features. So: minor citation nit,
  not a prior-art gap. Retracting the stronger framing.

## §7 "Playing with the Hatchling" — merging & no-BPTT (L1487–1600) ★ most fork-relevant section

### The parent paper's merge experiment, precisely `[FACT]` c9/10
Procedure (`:1491`–`:1500`): base model on En-Es, `n = 24576` (19 M); clone it and continue training on En-Fr and En-Pt;
merge by **concatenating every parameter tensor that has an `n` dimension** (`W_y`, `W_x`, `E`, RoPE frequency buffers) along
`n` → `n = 49152` (38 M), and **averaging all other parameters (token embeddings and token-prediction weights)**. Evaluated
with *no* subsequent training, deliberately.

Table `tab:model_merging_esfrpt` (`:1507`–`:1520`), lower is better; their stated unconditional-English reference ≈ 0.65:

| model | Es→En | Fr→En | Pt→En | En→Es | En→Fr | En→Pt |
|---|---|---|---|---|---|---|
| base En-Es | **0.36** | 0.77 | 0.64 | **0.35** | 2.21 | 2.27 |
| base tuned on En-Fr | 0.58 | **0.36** | 0.68 | **2.57** | **0.31** | 2.54 |
| base tuned on En-Pt | 0.44 | 0.76 | **0.34** | 1.79 | 2.20 | **0.33** |
| merged (concat) | 0.43 | 0.40 | 0.39 | 1.45 | 0.77 | 0.86 |

Reading of the numbers `[MATH]` c9/10: **understanding survives, generation does not.** The merged model is at or near specialist
level on all three *into-English* directions (0.43/0.40/0.39 vs 0.36/0.36/0.34), but for *from-English* it is worse than the
unconditional baseline 0.65 in every direction (1.45 / 0.77 / 0.86) — i.e. concatenation leaves it worse than babbling English.

### Candidate issues (cont.)
- **I28 — the parent paper already contains catastrophic-interference evidence, and never labels it as such** `[FACT]` c9/10:
  compare rows 1→2: En→Es goes **0.35 → 2.57** (worse than the 0.65 baseline by ~4×) after fine-tuning on En-Fr, and Es→En degrades
  0.36 → 0.58. That is textbook forgetting of a directional capability, visible in their own table (`:1512`–`:1514`). The text frames it
  as the model "somewhat retaining the capacity to translate Spanish to English and losing the capability to translate English to
  Spanish" — descriptive, never connected to continual learning, and §1 explicitly declines to consider time scales beyond minutes.
  `[INFERENCE]` c8/10: **this is the citation our fork's premise should use** — interference in BDH-GPU is demonstrated by the parent paper's
  own data, so motivating text does not have to argue from analogy with EWC-era LLM results. And it sets a plausibility band for our own
  effect sizes: their naive-fine-tuning damage is ~+2.2 in these units; our reported +1.6…2.2 nats (2-language) is the same order, which is
  reassuring rather than suspicious — but we should say which unit convention we use, because "loss ≈ 0.65 = unconditional English" is a very
  different scale from perplexity-nats on a 256-token byte vocabulary. c8/10
- **I27 — the merge section oversells a mixed result** `[FACT]` c9/10: the framing sentences say composition is "relatively straight forward"
  (`:1489`), that the merged model shows "**human-like degradation of operation**" (`:1523`), and §8.1 item 4 turns it into "direct composability
  of model weights in a way resemblant of composability of programs" (`:1617`). The table shows failure in 3 of 6 directions, below a trivial
  baseline, with the mechanism (averaged embeddings/unembedding) untouched. Calling that "human-like" is unfalsifiable framing — there is no human
  baseline anywhere in the paper. `[INFERENCE]` c8/10: our fork must not inherit this rhetorical pattern; if our merge→prune→replay result also has
  a direction where it is worse than a trivial baseline, that belongs in the results table, not in prose hedging. c8/10
- **H10 — the destruction channel is *averaging the non-n parameters*** `[INFERENCE]` c9/10: concatenation isolates `E, W_x, W_y` (the n-dimensional
  parameters) but averages token embeddings and unembedding — exactly where direction-specific output mappings live. So the parent's failure is
  evidence that **isolating only the neuron-dimension parameters is not sufficient**; understanding (input side) survives, generation (output side)
  does not. → Check in the manuscript: what does our fork do with embeddings/unembedding, and does it claim exactness of specialists while sharing
  output parameters? If the latter, "exact specialists" needs its scope stated (`:1497`). c8/10
- **H11 — the parent has no routing at all**, so our prefix-routing result is not a comparison against their method but against *nothing*; the honest
  framing is "concatenation as in [Pathway] leaves generation broken; adding routing/pruning repairs it", and the win must be measured on the
  from-English columns where they fail (their En→* row), not only on into-English where plain concatenation already works. c8/10
- **I29 — asserted information-theoretic bottleneck** `[FACT]` c8/10: §8.2 claims "All natural training approaches ... appear to bottleneck on the
  amount of available state space on synapses, becoming necessary at about `T ~ 1/ρ` by a simple information-theoretic argument on state storage
  capacity" (`~:1640`) — no such argument is given anywhere in the paper, and `1/ρ ≈ 20 tokens` conflates *activation sparsity* with *write rate per
  synapse* (a synapse can be read many times without being written; they even say "a synapse changes state only once in that window" as a plausibility
  assumption). Flagging because our fork's own framing borrows the "state runs out, so consolidate into weights" narrative (§8.2: "As time goes by,
  the system runs out of state space", `~:1637`) — if we use that narrative as motivation, we should cite it as *their conjecture*, not as established fact. c7/10

### No-BPTT (§7.2) `[FACT]` c9/10
Detaching K and V in `LinearAttention` = training with **zero** BPTT still crosses the letter-bigram barrier (loss 2.4) but leaves the model at
0.75–1.05 vs 0.65 for the backpropagated unconditional baseline, and it "lost the ability to match concepts between different languages" (`:1593`).
→ **H12**: if our fork trains with truncated BPTT (the parent's own models carry state across minibatches at 2048 characters, `:1051`), then any claim
that isolation preserves capability must be checked against the fact that *the gradient window itself couples tasks over its length* — replay inside one
window is not independent of the other task's activations. c7/10

## §5–§6 skim (modularity, scale-freeness, monosemanticity) — headline strength only
- §5 claims "scale-free modular structure emerges naturally when the model is trained" (`:1090`), positive Newman modularity of a thresholded
  interaction graph (`:1227`), heavy-tailed power-law-like degree distribution of `G*` with out-degree more concentrated than in-degree (`:1311`).
  Honest negative detail included: for all h=4 models, **1 of 4 encoder sub-matrices has no heavy positive tail** (`:1330`). `[FACT]` c9/10
- §6's monosemanticity evidence is *anecdotal by their own description*: "we have identified a few synapses that are activated at recognizable concepts"
  (`:1446`) plus a qualitative §6.2 on sparse signals enabling interpretability (`:1469`). `[INFERENCE]` c8/10: the merge story in §7.1 is explicitly
  conditional on monosemanticity ("when the model latent space promotes concept disentangling (c.f. Section ...)", `:1535`) — so a claim that rests on
  it inherits its evidential status. This is my I6 pattern again, and it is exactly the standard I will apply to our fork's mechanistic-sounding claims. c8/10

## Task 1 → Task 2 handoff (final list)
H1 structural isolation > gating · H2 consolidation = state σ → weights G (our merge/prune/replay targets their open problem; quote §8.2 `~:1637`) ·
H3 `G_s ⊆ G_x`, per-layer vs shared σ · H4 depth = weight-shared recurrence (✓ faithful, `:1021`) · H5 support(x) is the router ·
H6 growing n still shares the d-space and everything non-n (see H10) · H7 attention read excludes current token (`τ < t`, intentional) ·
H8 quote "same model" claims with hypotheses (`G_s = 1`, LN-free, low-rank) · H9 heads ≠ routing (parameter-free per-head LN only) ·
H10 embeddings/unembedding are the merge failure channel · H11 no routing baseline in parent; measure on their failing columns ·
H12 BPTT window couples tasks — isolation claims must respect it.

### Vocabulary note for the review (Pathway-authoritative terms)
"neuron dimension n" / "synaptic (low-rank) dimension d"; "concept space" = Rⁿ; "state" σ/corr = synapse weights, distinct from *weights* `E,W_x,W_y`;
"in-context learning" = attention over state; consolidation ("short-to-long memory transfer from state to network weights", §8.2) = what our fork calls
consolidation/replay. Using Pathway's words in the review keeps us aligned with the authoritative terminology (per user instruction).
