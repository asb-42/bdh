# Prior art for monotonic growth with submodel selection — what BDH may and may not claim

**Author:** pi-50 · **Date:** 2026-09-10 · **Status:** partial pass, two items still unverified
**Trigger:** operator proposal #141 and my Exp-3 result (`docs/reports/2026-09-10_pi-50_selection-null-and-matrix.md`)

## Why this note exists

RA2b showed parameters of earlier languages are bit-frozen while their serving PPL collapses
10–40×, and masking the readout to a language's own prefix restores its exit PPL **exactly**. The
tempting framing is "monotonic model growth that preserves competencies by selecting a submodel at
inference" (#141 steps 1–10). That framing is **not new**, and a reviewer will say so in one
sentence. This note records what is established so the manuscript claims the narrow thing we can
actually defend.

## Verified from search results (title/abstract level)

| work | mechanism | relation to our observation |
|---|---|---|
| **Progressive Neural Networks**, Rusu et al. 2016, arXiv:1606.04671 | freeze previous columns, add a new column per task, lateral connections; explicitly motivated by transfer + avoiding catastrophic forgetting | the "freeze then append capacity" skeleton, 10 years old |
| **PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning**, Mallya et al., CVPR 2018 (openaccess; arXiv:1711.05769) | prune to free capacity per task, store a mask per task, one network holds many tasks | "one substrate, many preserved submodels" |
| **Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights**, Mallya & Lazebnik, ECCV 2018 (arXiv:1801.06519) | learned binary masks over a shared body per task | masks as the unit of preservation |
| **Expert Gate: Lifelong Learning with a Network of Experts**, Aljundi et al., CVPR 2017 | frozen expert pool + learned gate selects per task | selection is a *learned router*, i.e. exactly our open problem |
| **SupSup: Supermasks in Superposition**, NeurIPS 2020 | randomly initialised **fixed** base net; per task a subnetwork/supermask is found; "sequentially learning thousands of tasks without catastrophic forgetting"; supermasks are linearly composable | closest analogue to growth-plus-preserving-by-subnetwork, and stronger than ours (no retraining at all) |
| **Incremental Learning Using a Grow-and-Prune Paradigm with Efficient Neural Networks**, arXiv:1905.10952 | grow then prune, sparse incremental expansion | grow-and-preserve family is populated |

## Still unverified in this pass (search provider dropped these queries)

- **HSP / hypernetwork-generated masks** for task-incremental learning (von Oswald et al. 2019 lineage) — expected to be relevant to "which mask do we apply at test time", because that line *does* use a task identifier to generate the weights.
- **PCANets** (Miae Cho et al., AAAI 2020) — progressively collapsed architectures for class-incremental learning; the DOI surfaced by search looked wrong for the paper, so I am not citing it until re-checked.
- The owed M4 items from the original review list (Modularity-with-Invariance, Riemer ICLR'21, LSNet) remain open.

## What survives as BDH-specific

1. **Where the growth happens.** All of the above grow or mask *layer weights*. BDH grows the
   **latent width of a single recurrent block** (`mlp_internal_dim_multiplier`, 8192→47104 per head
   here), and `k_sparse_ratio = 0.0` means the readout **sums over all candidates** rather than
   taking top-k. Our negative result — that this summation destroys access without touching a
   single old weight — is a statement about *this* architecture's readout, not about CL generally.
2. **Byte-level multilingual domains inside one ladder**, with per-language territory emerging
   rather than being assigned by a task id.
3. **The measured quality of the artifact**: exact bitwise preservation plus exact PPL recovery
   under oracle masking (19/19 domains) is a clean, quantitative demonstration that the failure is
   selection, not erasure — worth publishing as a diagnostic even if the mechanism is known.
4. **What we could not do, stated plainly**: no selector based on activation energy identifies the
   owning territory (1/20 best; every selector picks the English base block). Label-free argmin-NLL
   selection is the live test (`scripts/pi50/exp4_selfnll_selection.py`). Note that PackNet,
   Piggyback, SupSup and Expert-Gate all assume a task/domain identifier or a trained gate at
   inference, so **our inability to self-select is the field's shared limitation, not a BDH defect** —
   which is the honest way to position #141 step 8.

## Claim discipline for the manuscript

- Forbidden: "we introduce monotonic growth that preserves competencies while routing
  automatically." Progressive Networks, PackNet, Piggyback, Expert Gate and SupSup collectively
  cover freeze/append/preserve/select, and we have **not** demonstrated automatic selection.
- Allowed today: "route-aware latent-width growth preserves previously acquired languages
  **losslessly in parameters**; serving quality is recovered exactly by restricting the readout to
  the language's own prefix; unconstrained readouts degrade 10–40× because candidate summation is
  width-dependent. Energy-based automatic selection fails (≤1/20); likelihood-based selection is
  under test."
- Required before submission: verify HSP + PCANets properly, and cite them; state the task-id
  assumption of the prior work explicitly when claiming any advantage.
