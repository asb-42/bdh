# Digesting the adversarial review (GPT, three parts, bus #249–#251)

**Author:** pi-50 · **Date:** 2026-09-13 · **Status:** analysis, no measurements, no manuscript edits
**Input reviewed:** `#249` (32-point scientific review), `#250` (NeurIPS/ICLR referee report, 5/10, confidence 4/5),
`#251` (Area Chair meta-review: novelty 7, correctness 7, empirical 5, significance 6, reproducibility 8, overall 5/10;
"public technical report: yes after claim tightening; competitive venue: not yet").

Operator asked for digestion rather than another repair cycle, so this file deliberately does **not** do what I have
been doing all week — grepping each reviewer sentence against artifacts. That layer is the one that failed today
(my false B3 rejection deleted a true finding; my A4/D1 was arithmetically perfect and answered the wrong question).
Most of what this review says cannot be checked by grep at all, because it is about **experimental design**, not about
whether numbers are right. So the organizing question here is different: *what class of problem produced these findings,
and why did seven prior passes not produce them?*

---

## 1. Why this review found different things

Five machine reviews plus my own adversarial pass concentrated on verifiable surface: arithmetic, citations, escapes,
units, intervals, provenance. We caught real damage there (TAB-eaten `\texttt`, four fabricated-or-mangled bibitems, an
inverted failure count, an optimistic Wilson band, an unreproducible 1.02–1.13 range, a dead instrument-offset excuse).
Every one of those was falsifiable by a command.

This input was prompted differently — *"review it as a scientific paper rather than as a friendly edit"* — and its
findings sit one level up: whether the design licenses the sentences built on it. Those are not refutable by search,
which makes them invisible to a verification culture like ours. The uncomfortable symmetry: **the same blindness that
let my D1 through** (both columns measuring the same thing) is a design-level error, not an arithmetic one. Today's
lesson and this review are the same lesson arriving from two directions.

It also did something none of the others demonstrably did: it read `pipeline/train.py`, `pipeline/data.py`, the router
and the README, and it declared what it could *not* inspect (PDF figures rendered visually). Under my new third rule —
a reviewer that copies source with markup intact has read the file — this one read it twice.

---

## 2. Convergence: it independently rediscovered our own corrections

Listed because convergence across independent instruments is the best evidence any of this is real.

| Its point | Ours, earlier |
|---|---|
| 83 % is not a stable effect size; abstract foregrounds it | **A7** (`bdh/4b47071`): own-era fractions 0.599–1.472, median 0.880 |
| 20/20 routing is statistically weaker than it looks; crops correlated | my own **pseudo-replication** correction (`bdh/c13ed99`) |
| Damage-fraction arithmetic must name its units | dual-fraction fix, rev 4.2 (`aa286d1`) |
| OOD thresholds fit and evaluated on the same probes = separability, not detection | our own disclosure in §OOD; still needs scoping language |
| "destruction" overreaches a fixed-budget reacquisition probe | our own caveat sentence, which it quotes correctly |
| Engram comparison carries too much weight for a vendor card | **Kimi §5 item 6**, which I suppressed and later restored (`bab4822`); rev 4.5b (`874bc26`) demoted the verb |
| Single-seed effects near the 2–4 % floor are fragile | our internal rule that headline effects below the seed floor are not measurements |

Nothing here should be treated as the review "confirming" us; it is a second instrument reading the same object.

---

## 3. Where it is right, new to us, and cheap to act on

### 3.1 The one that matters most: the headline contrast moves two variables at once

Its "must-have ablation A" — *growth + freezing but no route-aware loss* — sent me into the training code, and what I
found there is not in any of our documents in this juxtaposition:

    pipeline/config.py:64   route_aware: bool = False      # route-aware training: zero out old neurons in forward...
    pipeline/config.py:65   route_alpha: float = 0.9       # loss = alpha*prefix_loss + (1-alpha)*full_loss
    pipeline/train.py:296-298                              # prefix-masked forward + full forward, losses mixed
    RA2b checkpoint cfgs:   en(phase 1) route_aware=False ; de, bg, lt ... route_aware=True, alpha=0.9
    scripts/ladder_fixedcap.sh:4   "~100M, no growth, no route-awareness, no masks, no replay"

So the ladder was trained with a per-phase auxiliary objective that **directly optimizes serving under the prefix mask**
(α = 0.9 on the prefix term), and the fixed-capacity baseline that "destroys" knowledge was trained **without** it. The
manuscript discloses both facts — route-aware training at tex 105–110, α = 0.9 in the new hyperparameter table row 159,
FCS's absence of route-awareness only in the shell-script header — but never places them side by side where the
comparison is asserted. That is our own `[[columns-must-be-different-measurements]]` rule violated in the *design*
rather than the evaluation branch: the contrast labelled "destruction versus preservation" differs in capacity
mechanism **and** in training objective.

Consequences, stated carefully:

- It does **not** invalidate any measurement. Every number in the tree is what it says it is.
- It does change what the strongest sentence may be. "Preservation works, destruction happens" is safe. "…and therefore
  the remaining difficulty is addressing, which likelihood routing solves" is weaker than it looks, because part of the
  routed-serving competence was trained in by an objective that is itself about routed serving. Prefix selection
  recovering knowledge is *expected* under that objective, not surprising.
- It touches my own results. A4/D2 (routed vs acquisition, median +4.3 %) compares two quantities from the same
  route-aware-trained checkpoint, so it is internally clean — but it inherits this objective as background condition,
  so it should not be quoted as evidence about routing in general.
- Cheapest honest fix: one sentence in the FCS→RA2b transition naming the double difference, plus one row in the
  hyperparameter table giving each regime's objective. Costs no compute and removes the single most attackable surface
  in the paper.
- Real fix: the missing arm (grown + masked + restored, `route_aware=False`). See §5 for what that costs.

### 3.2 Vocabulary audit on `exact / solved / generalizes / impossible / all / never`

Its §27 is procedurally correct even where individual instances are already scoped: a paper whose central noun is
"bit-exact" must reserve it for tensor equality and never let it leak into functional identity. Our own framing already
separates storage from serving, so this is a tightening pass, not a retraction. Worth doing once, mechanically, with the
occurrence list attached so the next reviewer can see it was exhaustive rather than impressionistic.

### 3.3 State the sparsity regime

Its §20/#17: ratio-based top-*k* is not width-invariant under expansion, and our runs use `k_sparse_ratio = 0.0` (dense
ReLU). This is the *reason* the phrase I rejected Kimi over mattered: `k_sparse_ratio` genuinely does not appear in the
manuscript (0 hits under six spellings, including `k\_sparse\_ratio` and `top-$k$`), and it should — as a scope
condition on the preservation theorem. A factual rejection by me turns out to carry a real improvement. Take it.

### 3.4 Continual learning versus continual pretraining / domain adaptation

Minor comment 10 in `#250`, and correct: task-incremental language accumulation over parallel corpora is closer to the
latter. One definitional sentence early would defuse a reviewer's framing attack, and it is consistent with the
parallel-corpus honesty demand in §15 of `#249` (also right: Europarl means shared content across languages, which is
why "shared Latin statistics act as implicit replay" cuts both ways).

### 3.5 Theory presentation

Three textual points, all checkable, all mine-to-flag-but-not-mine-to-fix:

- **Reachable-set circularity is real.** tex:228 defines
  $\mathcal{T}_A=\bigcup_{\ell\le L}\{h^{F'}_\ell(x):x\in\mathcal{X}_A\}$ — trajectories under the *expanded* map $F'$ —
  inside a theorem about preserving $F_A$. Defining the set whose equality you are trying to establish in terms of the
  new object is at minimum confusing, and its suggested fix (define $\mathcal{T}_A$ from $F_A^\ell$, then state the
  criterion) is cleaner with no loss of content.
- **"Soft gates cannot be exact"** is titled more universally than a counterexample for a particular cross-coupled
  ReLU class supports. Rename to what it proves.
- **Reduce theorem count / make quantifiers explicit** ($\forall \ell=0..L$ spelled out). Cosmetic, and it lowers risk.

Note the discipline here: Kimi attacked the same section and I rejected two of its theory objections as conflating
initialization with final state — those rejections still stand. Agreeing with one critique of a section does not
license reversing position on the whole section.

### 3.6 AI-participation disclosure as process provenance

Its §26 asks for model identities, versions, which text was generated, whether agents executed experiments, whether any
agent had write access, whether numbers were independently recomputed. We can answer all of it accurately from the bus
record and git history, and answering it *precisely* is better than the current general sentence: agents did execute
experiments, agents did have write access to the repo under a human-approved intent protocol, and every headline number
in the paper has been independently re-derived at least once by a second seat. That is a strength, not an embarrassment.

### 3.7 Repository hygiene

`README.md:73` still says the 20-language ladder "is running / being analyzed", and `README.md:18` points at
`docs/papers/cl-bdh-manuscript.pdf` while the live manuscript is `rev4-bdh-manuscript`. Both true, both trivially
fixable, both exactly the kind of thing that makes a reviewer distrust the artifact set wholesale.

---

## 4. Where it is wrong, or already answered

Per the rule adopted today: assertions of absence carry their command and hit count.

### 4.1 The routing-circularity objection (§6 of `#249`, major concern 4 of `#250`) — answered by a landed artifact

It states that the router selects on the early part of a crop and serves the late part of the *same* crop, so
"routed equals acquisition" may be self-flattering. The protocol description is accurate: `scripts/eval_router.py:91-93`
scores `rl[:, :, :window]` to choose and `rl[choice, :, window:]` to serve, window = 128 of 512. But the conclusion does
not follow for this dataset, because of the confusion matrix in the same run:

    out/logs/ladRA2b_routdiag_p20.txt  -- every one of the 20 domains routes to exactly one prefix width, 40/40 crops,
    and that width is its own cumulative acquisition width (en->8192, es->10240, pl->12288, ... lt->47104).

When selection agrees with the label on 100 % of crops, **routed serving is oracle serving**: the router's choice
coincides with the language-ID'd choice, so per-sample information buys nothing beyond what a fixed assignment would
give. The objection is therefore structurally inert for the headline claim — and, importantly, the paper should *say
so*, because routed ≡ oracle is a stronger statement than routed ≈ acquisition and it is already in the data.

Two residual valid kernels: (a) the argument fails wherever accuracy < 100 %, so it must not be generalized past this
benchmark; (b) a genuinely independent test would be a *different* corpus slice per role — which we happen to have for
the addresser result (`scripts/pi50/exp4_selfnll_selection.py` selects on a calibration slice and scores a **disjoint**
test slice), but not for the routdiag family.

**Defect it exposed in our tooling:** `eval_router.py:5` advertises "routed/oracle/joint perplexity" and the landed log
contains zero occurrences of `oracle` (`grep -c oracle out/logs/ladRA2b_routdiag_p20.txt -> 0`); the code computes
routed and joint only. An instrument whose docstring promises a control column it never emits is how controls go
missing from papers. Flagged for the owner; I am not editing core scripts.

### 4.2 "Single seed makes +4.3 % uninterpretable" — half right, for a reason it did not give

Seed variation does not blur this particular number: routed and acquisition come from the *same* checkpoint measured by
the *same* instrument family, so a different seed moves both terms together. What is genuinely unestablished is whether
the *level* of that cost is stable across ladders — a different question, requiring replication rather than more
statistics on this draw. The manuscript should make the distinction instead of absorbing the generic "3 seeds" demand
uncritically; and the same reasoning applies to my A7 era-to-era spread, which *is* a between-checkpoint quantity and
therefore *is* seed-exposed.

### 4.3 "Acknowledges a limitation then reclaims the stronger claim elsewhere" — fair pattern, largely repaired

This was true of rev 4.2 and is much less true of rev 4.5b. Remaining instances I would still fix: the abstract's
foregrounding of 83 % (its §9 recommendation matches our own A7 conclusion), and any surviving "routing is exact"
phrasing versus "all evaluated crops were correctly routed".

### 4.4 Small factual slips, recorded only so nobody treats them as findings

It cites the replay-parity result as "+27 % budget" needing a denominator (correct ask, and the number is ours); it
assumes a "naive growth without route-aware loss" arm is absent from *code* discussion when we disclose it in prose
(it is absent as an *experiment*, which is the real point); and its figure caveat is self-declared rather than an error.

---

## 5. What its Tier-1 asks actually cost, in units we have measured

Not a proposal — the operator asked for digestion, and the honest shape of the decision is a cost picture. Anchors from
this project's own logs: the 20-phase RA2b ladder is ~115 h on the GB10 at ~3038 ms/step, 148 M tokens, 100.9 M → 579.1 M
parameters; `.200`'s 4090 is R3's and off-limits without asking; memory bandwidth here (~273 GB/s) makes decode-heavy
work bandwidth-bound, so do not promise throughput we have not measured.

| Ask | Ladder-equivalents | Wall-clock on gx10, serialized | Notes |
|---|---|---|---|
| Growth without route-aware loss (protects an existing headline) | ~1.0 full, ~0.3 for a 6-phase version | 35 h – 115 h | Highest value per hour: it converts §3.1 from caveat into evidence |
| 3 seeds on the core ladder | +2 | ~230 h (≈10 days) | Only the final-width metrics needed; could halve by seeding 8 phases |
| 2 alternative orderings | +2 | ~230 h | Its §12: more informative than more languages |
| One heterogeneous non-Europarl benchmark (5–8 domains) | ~0.4–0.8 | 45–90 h + corpus/pipeline work | New data plumbing; the multi-byte census showed 3-byte scripts behave differently, so this is a real test not a formality |
| Replay + one consolidation baseline, compute-matched | ~0.5–1.0 | 60–115 h | EWC/SI/MAS need implementing against a growing latent axis |
| Independent routing test with split calibration/validation/test | ~0 | hours, eval-only | Already exists for the addresser; extend to routdiag family |

Summed naively, its "minimum convincing package" is **five to seven ladder-equivalents, roughly three to five weeks of
exclusive use of this machine**, before counting the corpus work. That is the actual price of the conference-grade
version, and it is worth saying plainly rather than discovering incrementally.

---

## 6. My recommendation, bounded and compliant with the stopping rule

**Option C — narrow now, publish as a scoped technical report, and buy breadth selectively later.**

1. **Free tier (sentence-level, this week):** name the double difference at FCS→RA2b (§3.1); add the objective row to
   the hyperparameter table; state the dense-ReLU scope condition; add the parallel-corpus interpretation sentence
   (§3.4); reserve "exact" for tensors (§3.2); rename the soft-gate proposition and fix the $\mathcal{T}_A$ definition
   (§3.5); state routed ≡ oracle from the confusion matrix instead of leaving it implicit (§4.1 — this one *raises* the
   claim); refresh README status and manuscript pointer (§3.7); expand the AI-provenance paragraph (§3.6).
2. **Cheap tier (eval-only, hours):** extend the router instrument to emit the oracle/fixed-assignment column and a
   split-data variant, so the independence question is answered with a table rather than an argument.
3. **One expensive tier, chosen, not bundled:** the `route_aware=False` growth arm, because it is the only new
   experiment that defends an existing headline rather than expanding scope. Shortened 6-phase version first.
4. **Decline for now:** more languages, larger territory counts, representation ablations, and the full Tier-1 package.
   Its own §10 agrees: the marginal value of language 21–40 is near zero.

If the operator wants a conference submission this cycle, then step 3 becomes steps 1–3 of §5 and the calendar conversation
is unavoidable; I would rather have that conversation once, with these numbers in front of us, than discover the cost
across six weeks of partial commitments.

---

## 7. Process change I would adopt regardless of what we publish

Our review loop has one blind spot and one dangerous layer. The blind spot: it tests values, not questions — a number
can be reproducible, correctly united, correctly intervalled, and still be measuring the thing it is being compared
against. The dangerous layer: the adjudication of external findings, where a malformed grep silently deletes a true
finding (B3) and nobody notices, because suppressions leave no trace while misses do.

Proposed standing addition, one pass per revision, done *before* numerical verification:

> For each headline sentence: (i) name the comparison that would have to hold for the general version of the claim to be
> true; (ii) state whether we ran it, and if not, what the sentence costs to scope; (iii) check that the two sides of
> every quoted ratio are different measurements — evaluation branch *and* training objective.

That is the design-level analogue of the check that would have caught D1 this morning, and the check whose absence let
§3.1 survive until an outsider asked for an ablation.
