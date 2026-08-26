# bdh-cl — Continual Learning research on BDH-GPU

> This is a research fork of
> [**pathwaycom/bdh**](https://github.com/pathwaycom/bdh) (the official implementation of
> [*The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*](https://arxiv.org/abs/2509.26507)).
> Everything about the base architecture, its paper, and its license lives there.
> **This README describes only what this fork adds.**

---

## What this fork is

A laboratory for **continual learning on BDH**: can a trained depth-recurrent model acquire
new capabilities phase after phase without catastrophically forgetting — and if freezing
weights is not enough (we show it is not), what actually works?

The full study is written up in
[`docs/papers/cl-bdh-manuscript.pdf`](docs/papers/cl-bdh-manuscript.pdf)
(*Computation Isolation for Continual Learning in a Depth-Recurrent Language Model*, 2026),
with complete intermediate tables in
[`docs/reports/`](docs/reports/).

## Headline results (100M parameters, byte-level LM)

| finding | number |
|---|---|
| sequential training forgets catastrophically | EN/DE degrade $11$--$18\times$ ppl |
| freezing old weights does not help | $+0.85$ nats erosion per grown block |
| additive growth + prefix routing | specialists reproduced **exactly**, 100% detection |
| soft likelihood mixture | within 0.003 nats of hard routing |
| merge → random-prune → brief replay | single original-width model at **joint-training parity** |
| ~20% replay during training | same parity at +27% budget |
| soft gates can never be exact | proved (counterexample class) + measured budget |

Five empirical laws (L1–L5) and a seven-entry negative-results registry are documented in
the manuscript — the failures are first-class results here.

## What's added vs. upstream

| area | additions |
|---|---|
| `pipeline/data.py` | multilingual Europarl loader (21 language sides, per-language byte caps), multi-register textmix loader |
| `pipeline/train.py` | phase chaining (`--init-from`), write-gating (`--gate-from/--gate-alpha`), **width growth with RoPE-phase preservation** (`--grow-mult`) |
| `scripts/` | likelihood router, per-language/domain evaluation, neuron-importance extraction, pruning sweeps, chained merging, calibrated graph metrics |
| `docs/plans/`, `docs/reports/`, `docs/papers/`, `docs/notes/` | pre-registered plans, full experimental record, manuscript, formal-analysis integration notes |
| upstreamed | [`extend_freqs()` RoPE fix](https://github.com/pathwaycom/bdh/pull/12) — width growth used to silently rewrite every existing neuron's phases |

Mechanisms are intentionally kept out of upstream's minimal reference implementation;
upstream receives only correctness-grade fixes (see PR above).

## Reproduce

```bash
# phase 2 of a continual chain (DE after EN), canonical protocol
python -m pipeline.run train --model bdh --dataset europarl \
    --europarl-langs de --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 --n-layer 6 --mlp-internal-dim-multiplier 128 \
    --block-size 512 --max-iters 10000 --batch-size 4 \
    --init-from out/bdh_europarl_cl-a-en_last.pt --run-name cl-a-de

# per-language held-out evaluation
python scripts/lang_eval.py out/bdh_europarl_cl-a-de_last.pt 30

# grow + serve routed (single forward per detected phase)
python -m pipeline.run train ... --grow-mult 32 --init-from <prev>.pt
python scripts/eval_router.py <grown>.pt --routes 8192,10240,12288 --domains ...
```

Hardware: single RTX 4090 (24 GB); every result in the paper reproduces on it.

## Current experiment

A 20-language phase-count ladder (does isolation survive accumulation?) is running /
being analyzed — see [`docs/plans/`](docs/plans/) for the pre-registered plan and
falsifiers.

## Upstream

- Source repo: [pathwaycom/bdh](https://github.com/pathwaycom/bdh) — architecture,
  paper ([arXiv:2509.26507](https://arxiv.org/abs/2509.26507)), reference training code.
- Our correctness fix was upstreamed as
  [PR #12](https://github.com/pathwaycom/bdh/pull/12)
  (`extend_freqs()`: naive latent-width growth rewrites every existing neuron's RoPE phases).

## License

MIT, inherited from upstream (`LICENSE.md`).
