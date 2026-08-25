# BDH Continual Learning Plan — v2

**Version:** 2.0 · **Date:** 2026-08-25 · **Language:** EN
**Supersedes:** `2026-08-24_bdh-continual-learning-plan_v1.md` (mechanism list and
invariants carry over unchanged; hypotheses and priorities revised on evidence)
**Evidence base:** `docs/reports/2026-08-25_cl-h1-report.md` (CL-H1…H4, merge/prune
probes), `docs/reports/2026-08-24_progress-report.md` (E-series scale/data controls)

---

## 0. Mission (unchanged)

Frozen weights that stay plastic after training ends, without catastrophic forgetting —
the paper's deferred question (L379). v1 formalized mechanisms A–F; v2 replaces
speculation with measurements.

## 1. What the 2026-08-25 session settled

Five experiments, one day, all at 100M on Europarl EN→DE→ES (30 MB/phase):

| experiment | question | verdict |
|---|---|---|
| CL-H1 naive sequential | how bad is unmitigated interference? | catastrophic: EN +1.59 nats, DE +2.15 above peak; identical under ReLU and top-k |
| merge-across-phases probe | can separation + concat retrieve? | **yes**: 48–77% recovered, zero finetune; ES pays only +0.23 |
| CL-H2 constant-LR | is forgetting optimizer-reset shock? | **no**: constant LR worse everywhere; annealed endpoints consolidate better |
| CL-H3 write-gating (v1 of B/C) | can soft per-neuron protection help? | **null**: all cells within noise; importance spread thin; drift audit shows global rewriting (lm_head >100%, encoder triple ~110%) |
| CL-H4 growth (`--grow-mult`) | does fresh-capacity routing help? | acquisition near-parity (+25% width, de/es 2.22 ppl); frozen weights still erode 2.26→12.49 via depth-recurrence residual pollution |
| prune sweep (merged ×3) | is retrieval redundancy compressible? | **yes, randomly**: keep-⅓ at original width gives en 6.37/de 5.38/es 4.09 vs endpoint 11.08/18.28; magnitude-ranked pruning *collapses* (worse than random) |

### New empirical laws (supersede v1 §2 assumptions where they conflict)

- **L1. Forgetting is representational overwrite**, global across all parameter classes;
  not optimizer shock, not activation-regime dependent.
- **L2. Weight isolation ≠ computation isolation.** BDH's depth recurrence shares one
  parameter triple across levels: frozen neurons still erode when new blocks feed the
  residual stream they consume. Erosion is cumulative in added-block count.
- **L3. Within-phase knowledge is distributed; between-phase knowledge is modular.**
  Random neuron subsets preserve function; magnitude-selected subsets destroy it.
  Phase-level concat works; within-phase importance ranking doesn't.
- **L4. Sparsity is a capacity-to-data property** (~94–95% xy across k ∈ {0,…,0.10});
  top-k tunes weight-graph topology independently (edge_frac −20×) but neither causes
  nor prevents forgetting.
- **L5. Measurement discipline adds corpus difficulty**: never compare losses across
  corpora (Europarl ≈ easier than wikitext-2); alongside INV-1 protocol congruence.

## 2. Mechanism scoreboard (updates v1 §5)

| mech | v1 concept | status after 08-25 |
|---|---|---|
| A naive floor | sequential without protection | **done** (CL-H1): floor quantified |
| B eligibility gate | write only eligible edges | **falsified in soft form** (CL-H3 null). Hard form superseded by growth (below) |
| C EWC-style protection | importance-weighted step sizes | **falsified in \|xy\-magnitude form** (CL-H3 + L3: magnitude is the wrong signal) |
| D sleep-replay / distillation | gradients only during sleep, mixed with old-domain batches | **untested — now the highest-priority shared-weight mechanism** (only remaining lever after L1/L2 close gating routes) |
| E σ cascade | multi-timescale states | unchanged; blocked on per-edge damping |
| F grow & merge | separate storage + concat + prune | **strongly validated** (48–77% retrieval; grow-acquires-at-+25%; prune-to-size beats sequential). v1's "Weight Atlas contribution scores" pruning idea is **inverted by L3**: use random pruning |

## 3. Revised hypothesis table

| ID | claim | status |
|---|---|---|
| H1' | replay/distillation (D) cuts forgetting ≥50% at matched compute vs CL-H1 floor | **new top priority** |
| H2' | learned phase-router approaches oracle routing (per-phase peaks en 2.26/de 2.13/es 2.13) when labels observable | analytic bound established; build only if label-free routing needed |
| H3' | ~~importance protection beats uniform~~ **withdrawn** (L3) | closed |
| H4 binding | cross-domain binding requires replay (INV-3 probes; zero-shot floors: de 21.8/es 17.6 ppl) | unchanged, now unblocked (gate review satisfied trivially: B/C resolved negatively) |
| H6' | repeated grow→store→merge cycles stay clean over ≥3 phases; random prune-to-budget each cycle | **revised & promoted**: the operational recipe |
| H7 cost scaling | unchanged | rides along |

**Recipe under test (v2 operational core):**
*grow-to-acquire → store phase expert → deploy either routed (labels) or merged +
random-pruned to budget (label-free)*.

## 4. Execution order (from now)

1. **Multi-domain English control** (Grok priority #1): ×384-merged recovery and
   growth-acquisition must be shown to not depend on multilinguality — 90 MB English
   from ≥3 distinct domains, same 3-phase schedule, same probes.
2. **H1': sleep-replay arm** — DE/ES phases mix 10–20% replay batches from prior-phase
   streams (data machinery exists; `--init-from` chains exist). Compare against CL-H1
   floor and against grow arm. This is the last shared-weight contender.
3. **H6' cycling study** — ≥3 grow/store cycles + per-cycle random prune-to-fixed-width;
   measure cumulative erosion (L2 predicts monotone degradation; quantifying slope is
   the deliverable).
4. **Binding study (H4)** with replay arms per INV-3, using lang_eval + synapse_trace.
5. **E cascade** remains parked pending per-edge damping.

## 5. Invariants

Unchanged from v1 (INV-1 protocol congruence, INV-2 cost gate, INV-3 binding probes,
INV-4 coverage-declared checkpoints, INV-5 surprise discipline), plus:

- **INV-6 within-corpus comparisons only** (L5).
- **INV-7 graph claims need calibrated/subsampled thresholds; activation-regime arms
  (ReLU vs top-k) compared only at matched calibration** (E1/E8b).

## 6. Provenance

v2 written by ox-alpha after the 2026-08-25 RTX 4090 session (CL-H1…H4, merge,
prune sweeps). v1's mechanism inventory and invariants were preserved except where
directly falsified; every change cites an addendum of
`docs/reports/2026-08-25_cl-h1-report.md`.
