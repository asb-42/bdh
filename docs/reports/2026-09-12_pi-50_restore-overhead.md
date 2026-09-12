# Throughput and memory cost of the step-end frozen-region restore

**pi-50 · 2026-09-12 ·** answers #214 §B3 ("we have never measured this") and the operator's question about
whether to. No training run was needed: the operation is a pure device-to-device copy, so its cost is
computable from its byte count plus a microbenchmark, and the denominator (step time) is already in the logs.

## What the fix does

`pipeline/train.py:175–179` snapshots the three frozen regions at growth time and `:256–257` copies them back
after **every** optimizer step, because AdamW's decoupled decay multiplies masked parameters by
`(1 − lr_t·wd)` even at zero gradient. The regions are `encoder[:, :, :n_old]`, `encoder_v[:, :, :n_old]` and
the decoder viewed as `(n_head, n_new, n_embd)` sliced to `[:, :n_old, :]`. `embed`/`lm_head` need nothing —
`requires_grad=False` removes them from the optimizer entirely.

## Method

* Microbenchmark on the box that ran RA2b (GB10, `NVIDIA GB10`, torch 2.13.0+cu130): allocate the true shapes
  and strides, warm 3 iterations, time 30 repetitions of the exact three-copy loop.
* Denominator: median `ms/step` over logged steady-state points (samples beyond warmup) in
  `out/logs/ladRA2b_<lang>.log`, all 20 phases present.
* Widths reconstructed from the growth headers (`growth: … (+2048 neurons/head trainable)`): base latent
  `N = 8192`, each phase `+2048`, final `N = 47104`; verified against the checkpoint header
  (`encoder (8, 512, 47104)`, fp32, 0.719 GiB each of encoder/encoder_v/decoder).

## Result

Restore cost is linear in restored bytes at **8.7 ms per GiB** (≈240 GB/s of read+write traffic), and the
strided slice pattern costs essentially nothing extra: **stride penalty ×1.01** versus a contiguous copy of the
same bytes.

| phase | latent N | snapshot | restore | logged ms/step | share |
|---|---|---|---|---|---|
| en (base, no growth) | 8 192 | — | 0 ms | 777 | 0 % |
| es (first growth) | 10 240 | 0.375 GiB | 3.3 ms | 2 279 | 0.14 % |
| fr | 14 336 | 0.562 GiB | 4.9 ms | 3 167 | 0.15 % |
| hu | 26 624 | 1.125 GiB | 9.8 ms | 1 674 | 0.58 % |
| ro | 40 960 | 1.781 GiB | 15.5 ms | 2 646 | 0.59 % |
| **lt (final)** | **47 104** | **2.062 GiB** | **17.9 ms** | **3 038** | **0.59 %** |

Across the whole 20-phase ladder (each phase 10 000 steps as logged): **≈0.56 h of ≈115 h, i.e. 0.49 %**. The
share is flat at ~0.59 % once the model is large because restore bytes and step time both scale with latent
width; the low early percentages are an artifact of those phases' larger batch configuration (see the S2
train-batch note — `fr` at 3 167 ms/step is not comparable to `de` at 1 018 ms/step).

So "minor" was the right adjective and the wrong kind of statement. Quote the number instead.

## The cost that actually matters is memory, not time

The snapshot is held for the entire phase: **+2.06 GiB resident at the final width**, on top of weights,
gradients and Adam moments, with a further transient allocation during the initial `clone()`. On this box
(GB10, ~121 GiB unified) it is invisible. On a 24 GiB card it is not academic — our own masked-eval instrument
needed 12.3 GiB of activations at `BB=16` on this same checkpoint and OOMed. Any port of the growth protocol to
smaller hardware should expect the restore to consume roughly a tenth of the budget at full width.

## Cheaper and safer construction, for phase 2

Both costs disappear if the frozen regions are *separate parameters* rather than *slices of shared tensors*:
splitting `encoder`/`encoder_v`/`decoder` along the latent axis at territory boundaries lets old blocks carry
`requires_grad=False`, which removes them from the optimizer — the same mechanism already used for
`embed`/`lm_head` — so no decay ever touches them. Zero per-step copies, zero snapshot memory, and the
bit-exactness stops depending on a runtime invariant being re-established every step.

Why it is not done today: the mask is intra-tensor (per-column), and `requires_grad` is per-tensor; and a
`weight_decay=0` param group would not work either, since AdamW applies decay inside the step before the
zero-gradient update (`pipeline/train.py:163–169` documents exactly this). But the append-only layout already
knows the block boundaries — cumulative prefix widths `8192 + 2048·j` — so the split is natural rather than
contrived. Trade-off to check before adopting: more parameter tensors (20 vs 3) interact with DDP/bucketing and
with any code that assumes a single latent tensor, including the atlas decomposition
(`decoder.u{tile}.h{head}`, territory = `u // 32`).

## Latent hazard found while measuring (no landed result affected)

`pipeline/train.py:118` takes the branch `if cfg.init_from and grow_src is None:` — i.e. initializing from a
checkpoint **without** `--grow-mult` loads a grown model and trains it with `elem_frozen_backup = None`
(`:102`), so nothing is ever restored, under the default `weight_decay = 0.1`
(`pipeline/config.py:26`). That is the F-decay-leak failure mode reachable by a plausible-looking command line,
and there is no guard: the only raise nearby is the `--grow-mult`/`--gate-from` mutual exclusion.

Checked the callers that exist today: `scripts/ladder_ra2_resume.sh:85` and
`scripts/pi50/protocol_4090.sh:101` both pass `--grow-mult`, so the RA2/RA2b chains and the resume path are
clean, and this is **not** a retroactive correction to any published number. It is a request for an assertion —
refuse (or require an explicit override flag) when `init_from` names a checkpoint whose
`mlp_internal_dim_multiplier` exceeds the base config while `grow_mult == 0` and `weight_decay > 0`. Proposed
as candidate **F-V9** for whatever numbering the project adopts.

## Optional confirmation run

If a directly-measured end-to-end figure is wanted rather than microbenchmark-plus-log arithmetic: ~200 steps
at final width, identical seed and config, restore active versus the copy loop disabled by environment flag,
comparing `ms/step`. Roughly ten minutes on this box, and it would also validate the 8.7 ms/GiB slope against
a real training step where allocator and cache state are not ours to control. Held pending authorization —
outside the eval-only scope I currently hold.
