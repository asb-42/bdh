# Formulation audit: repeated misreadings as a text problem

**Author:** A0-Quinn. **Trigger:** operator directive, 2026-09-13.

> We have now had several reviewers whose criticism was not tenable. That can be a misreading. Do not simply discard that criticism - take it as an occasion to check the *formulations* for clarity and unambiguity. The paper should be written so that such misreadings cannot arise.

This document is the result of that instruction. It is not a list of reviewer errors. It is a list of places where our own wording invited a specific misreading, and of the sentence that prevents it.

---

## The rule this audit runs on

A claim that four independent reviewers misread the same way is not four mistakes; it is one ambiguous sentence. Our previous response pattern was *reject the criticism with evidence*. That pattern was often factually correct and methodically wrong: it left the ambiguity in place for the next reader, and it turned a wording defect into a debate about the reader.

The audit therefore asks, per case: **what does the sentence say to someone who has not read our reports, and what must it say instead?**

Cases are grouped by the *kind* of misreading, because the kinds repeat across rounds.

---

## Class 1 - "This is general" where we measured a configuration

*The most frequent failure class across all four rounds.*

| # | Current wording | How it is read | Required wording |
|---|---|---|---|
| 1.1 | "Storage is solved" (abstract) | Continual learning's storage problem is solved as a field | "For the tested growth construction, parameter preservation is exact; the remaining failure modes arise at serving and addressing." |
| 1.2 | "The address geometry is byte-statistical, not linguistic" (abstract, cross-script) | A causal law excluding linguistic structure | "In the tested setting, routing correlates strongly with low-level byte statistics; the linguistic reading is not required to explain any observed decision, and is not excluded either." |
| 1.3 | "Monotonic model growth holds across script families" | Generalization across scripts | "The construction remained functional in the tested Chinese-to-Hindi cross-script cell (two growth transitions)." |
| 1.4 | "A fixed operation cannot answer an input-dependent question" | Impossibility over all fixed readouts | "The tested families of simple fixed readouts (seven, two vacuous by construction) do not recover routed serving; we do not claim the space of input-independent operations is exhausted - a sufficiently expressive fixed map can itself implement input-dependent behaviour." |
| 1.5 | Proposition titled "soft gates cannot be exact" | Impossibility for soft gating | Rename: "A cross-coupled ReLU counterexample to exact soft gating", with the gate class and the equivalence notion stated in the proposition. |
| 1.6 | "Forgetting is destruction" | Information-theoretic destruction | "After nineteen overwrites under the 2k-step re-acquisition probe, no measurable re-acquisition advantage remains over a fresh model (Delta 1.8%, below the 2-4% seed floor)." |

**Rule:** every capability claim names its tested configuration in the same sentence, not in a later limitations paragraph. A qualification that lives elsewhere does not qualify anything.

---

## Class 2 - "The result is circular" where the protocol has a property we never named

| # | Current wording | How it is read | Required wording |
|---|---|---|---|
| 2.1 | "Routed serving reproduces acquisition quality exactly" | Selection and serving use the same sample, so the comparison flatters itself | Name the property: selection uses the first 128 tokens of a crop and serves the remaining 384 of the *same* crop - an online-routing protocol. Then state the consequence: at 20/20 domains with 40/40 crops each, **routed serving is oracle serving** for this benchmark; per-sample selection buys nothing that a fixed correct assignment would not give. Scope it: this holds only where accuracy is 100%, and does not generalize past this benchmark. |

The property was in the text as protocol description; it was never stated as a property. Describing a mechanism is not the same as naming what it implies.

---

## Class 3 - "A baseline is missing" where it exists but sits in the literature section

| # | Current wording | How it is read | Required wording |
|---|---|---|---|
| 3.1 | Replay parity appears as "our own replay-in-training result (H1p) reached joint parity at +27% budget" inside the replay prior-art subsection | A citation discussion; the reviewer concludes there are no method baselines | Promote to the empirical section as a first-class comparison with its own row: method, capacity, training tokens, retention, acquisition, compute. Reviewers read the literature section as positioning and the results section as evidence. Evidence placed in positioning is not read as evidence. |

This is the cleanest instance of the operator's point: the criticism ("no serious baseline") was tenable **about the text** while being false about the work.

---

## Class 4 - "This comparison moves two variables" where the objective is only in a script header

| # | Current wording | How it is read | Required wording |
|---|---|---|---|
| 4.1 | "With the fix active, a 20-phase route-aware ladder shows ..." (Results) vs. the fixed-capacity baseline described as "no growth, no masks, no replay" in `ladder_fixedcap.sh` only | The headline contrast (destruction vs. preservation) compares capacity mechanism *and* training objective, and the objective directly optimizes the competence we present as recovered | One sentence plus one table row naming the objective of every arm: growth + route-aware auxiliary loss (`alpha=0.9`, phases 2-20) vs. fixed-capacity with no route-aware loss. No number changes; the sentence about what the numbers show does. |

Both facts appear separately in the manuscript (route-aware training at line 105, its absence from FCS only in a shell header). A disclosure split across two locations is not a disclosure. The **real** fix is the `route_aware=False` arm (Tier 2); the sentence is the honest interim.

---

## Class 5 - "bit-exact" read as functional equality

| # | Current wording | How it is read | Required wording |
|---|---|---|---|
| 5.1 | "bit-exact", unqualified | Tensors and model behaviour are the same to the bit | Define once, explicitly: **bit-exact storage = exact equality of the preserved parameter tensors and buffers.** Functional equality is not claimed; routed-perplexity results are behavioural evidence, not functional bit identity. `exact` is reserved for the tensor property throughout. |

The distinction is central to the thesis, so the term that carries it must be defined where it is used.

---

## Class 6 - Real defects the misreadings surfaced (fix, do not reword)

| # | Defect | Evidence |
|---|---|---|
| 6.1 | `T_A` is defined from the expanded operator `F'` while it is the object that must be *preserved* - circular as written | `\TA=\bigcup_{\ell\le L}\{h^{F'}_\ell(x): x\in\Xa\}`; define instead from `F_A` and formulate the criterion around that set. |
| 6.2 | `scripts/eval_router.py:5` advertises "routed/oracle/joint perplexity" but emits no oracle column | 1 mention in the code (the docstring), 0 in `out/logs/ladRA2b_routdiag_p20.txt`. An instrument promising a control it never prints is how controls go missing. |
| 6.3 | Dense-ReLU scope never stated: all results are at `k_sparse_ratio=0` | Ratio-based top-k sparsification is not width-invariant under growth (documented in the code); the claims apply to the tested dense-ReLU configuration. |
| 6.4 | `README.md` lines 18 and 73 point at the v0 manuscript and describe the 20-phase ladder as "running / being analyzed" | Verified against the file; the ladder completed 2026-09-10. |
| 6.5 | Continual-learning vs. continual-pretraining framing never stated | The experiments are domain/register accumulation, closer to continual pretraining; say so once, in Setup. |

---

## What this audit does not do

It does not soften any measured number, and it does not add defensive clauses to avoid criticism. Every change either (a) moves an existing qualification to where the claim is made, (b) names a property that was already true of the protocol, or (c) fixes a defect. Nothing is hedged that we can measure, and nothing becomes vague that is currently precise.

---

## Consequence for the review process

From this round on, an untenable reviewer claim is recorded twice: as a rebuttal (with its evidence) and as an entry here (with the sentence that invited it). The rebuttal protects the work; the entry protects the next reader. A claim that reappears after both have been applied is a genuine disagreement, and only then is it a debate.

- A0-Quinn, 2026-09-13
