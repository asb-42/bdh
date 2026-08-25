# Upstream PR #6 Review — pathwaycom/bdh continual-learning extension

_Date: 2026-08-25 · Author: ox-alpha · Status: assessment against our 2026-08-25 measurements_

**PR**: pathwaycom/bdh#6 — "Attempts extension of BDH with continual learning using
pseudo-metaplasticity" (zer0condition, from `cMiraka/bdh-cl`; opened 2025-11-09;
**still open, unmerged**).

## What it contains

- **EWC** (Kirkpatrick-style quadratic penalty, Fisher importance, per-task reference weights)
- **Adaptive synaptic gates**: per-synapse `plasticity_state` ∈ [0.01, 1] with
  metaplasticity framing; `AdaptiveLinear(nn.Linear)` drop-in wrapper carrying
  `weight_ref` / `importance` / `plasticity_state` buffers
- **Path-integral online importance** (accumulated along the training trajectory)
- **Multi-task sequential trainer** + benchmark suite: **Permuted MNIST, Rotated MNIST,
  Split CIFAR**, a generic "Sequence" task — no language-modeling evaluation

## Provenance quality

19 commits include wholesale deletions and re-uploads of core files (`Delete bdh.py`
→ `Add files via upload`), a commit message asking *"proper implementation?"* on
`hebbian.py`, and a single approval from a non-maintainer. It reads as an
unreviewed research fork, not curated upstream work. We were right not to build on
it — but wrong not to check for its existence earlier (it predates our CL session by
~9 months).

## Mapping to our results (docs/reports/2026-08-25_cl-h1-report.md)

| PR #6 mechanism | Our equivalent | Our verdict at 100M / byte-LM |
|---|---|---|
| EWC + Fisher protection | Mechanism C v1 (CL-H3) | **null**; drift audit shows global rewriting — weight-localized protection cannot bind |
| Adaptive synaptic gates | same family | **null**; L2: weight isolation ≠ computation isolation (depth recurrence) |
| Path-integral importance | alternative signal for C | untested; prior expectation low (any *localized* importance faces the same audit), cheap to swap into `--gate-from` if ever needed |
| Multi-task sequential training | CL-H1 floor | floor confirmed catastrophic (EN +1.59 nats) |

**Why both can be right**: their benchmarks (Permuted/Rotated MNIST, Split CIFAR) are
precisely the *easy* case under our laws — each task is a disjoint statistical regime,
so neurons can specialize cleanly (our L4: cross-regime diversity raises sparsity and
modularity), and small classification nets lack BDH's shared depth-recurrent triple
(our L2). Our hard case — same language, blurred register boundaries, one recurrent
weight triple — is exactly where protection fails and only separation/retrieval works.

## What is worth taking from it

1. **Benchmark discipline, not mechanisms**: their suite standardizes task sequences
   and metrics (ACC_avg/BWT); our plan already computes these, so nothing to adopt.
2. **One cheap falsification probe**: path-integral importance as an alternative
   `--gate-from` signal. Predicted null by the drift audit; run only if a reviewer
   insists.
3. **Citation/related-work note** for any paper: independent (unmerged) attempt exists
   targeting protection-class mechanisms on toy benchmarks; our results explain why
   that class cannot transfer to LM-scale BDH.

## Decision

Do not merge or track further. Record in plan v2 connections; revisit only if the PR
gains maintainer traction or adds LM-scale sequential results.
