# BDH Progress Report — 2026-08-18

_Status of the Dragon Hatchling (BDH) investigation: what we audited, what we built,
what we measured, and every open lever we do not want to forget._

---

## 1. TL;DR

- The repo is a **byte-identical fork** of `pathwaycom/bdh`; `bdh.py` faithfully implements the
  paper's **BDH-GPU** baseline (the tensor special case of the full graph-based BDH).
- **Science**: the math is sound but elementary (expressiveness/equivalence results, not
  performance guarantees). The "Hebbian working memory / synaptic plasticity" language is a
  re-naming of the standard **linear-attention outer-product state** — there is **no plasticity
  at inference** in the shipped code. The headline Sudoku result is **not** reproducible from
  this repo.
- We built a **param-matched train/eval pipeline** (BDH + GPT baseline, byte-level datasets,
  CPU/GPU agnostic) and measured on both tiny-shakespeare and wikitext-2.
- **Central result**: BDH is a **quality-per-update** win (~6-7× fewer steps to the same loss),
  but **not a quality-per-FLOP** win — at equal compute it **ties** a matched GPT. The wide
  sparse latent buys the step-efficiency and is also the cost (5× more FLOPs/step).

---

## 2. Repository & audit

Fork of `pathwaycom/bdh@main` (remote `asb-42/bdh`), **byte-identical** to upstream at the
start (diffed `bdh.py`, `train.py`, `README.md`, `requirements.txt`, `.gitignore`).

Audit fixes applied (commit `be095f1`):

| File | Fix |
|---|---|
| `train.py` | CPU `dtype` now `float32` (was `float16` + enabled GradScaler with no autocast); `loss.detach()` (was retaining 100 autograd graphs); memmap cached (was re-opened 6000×/run); dead `eval()` removed; CUDA seeded; `raise_for_status()` on download |
| `requirements.txt` | `torch>=2.0` floor (`torch.compile`, `torch.amp`); later `pyarrow` for wikitext-2 |
| `.gitignore` | tooling + artifact hygiene (`__pycache__/`, `data/`, `out/`, `.skills/`, …) |
| `README.md` | clarifies shipped code = BDH-GPU baseline, not the full graph BDH |

No correctness/architecture bugs found in `bdh.py` (all shapes traced; RoPE `1/(2π)` cancels
against `%1 → *2π`). `bytes(tensor)` decode path verified as correct (buffer protocol).

**Verified non-issues (initially suspected, cleared):** `bytes(tensor)` decode; shape
consistency of the low-rank matmuls; `torch.backends.cudnn.allow_tf32` on CPU-only builds.

**Known cosmetic issue (deliberately not fixed):** `bdh.py` names its parameters *inverted*
vs the paper (`encoder` = paper's `decoder_x`, `decoder` = paper's `encoder`). Renaming breaks
`state_dict` keys + upstream fidelity, so we left it.

---

## 3. What BDH actually is

From the paper (`arxiv 2509.26507`, §2-3 + Appendix code listing):

- **BDH (general)** = an *edge-reweighting kernel* on neuron graphs — a chemical-reaction-style
  ruleset with excitatory/inhibitory graphs `G_x, G_y` and a synaptic state `σ` updated by outer
  products (`σ += |xy⟩⟨x|`, i.e. Hebb-like).
- **BDH-GPU (shipped)** = the tensor special case:
  `x_sparse = ReLU(x @ decoder_x)`, `yKV = linear-attention(Q=x_sparse, K=x_sparse, V=x)`,
  `xy = ReLU(yKV @ decoder_y) * x_sparse`, residual `x += ln(xy @ encoder)`.
  Equivalently: **linear attention** (no softmax, strictly-causal `tril(diagonal=-1)`) + a
  **gated low-rank MLP** (`ReLU × ReLU` with head-merging) + LayerNorm + RoPE.
- The state-space form is exactly `state += v·kᵀ; out = state·q` — the **linear-attention
  recurrent state**, which the paper calls "Hebbian working memory".

Science assessment (our read of the formalism + Appendix proofs): the four formal results
(protocol ⟺ state-space; low-rank ≈ sparse graph `G²(n,m)`; sparse `G_s` expresses dense
attention; linear attention ≈ general attention via JL + Azuma) are **correct but elementary**
— they establish *expressiveness*, not *performance*. The "attention is all you need" ⟺
"edge-reweighting is all you need" claim is an **interpretation**, not a new mechanism.

---

## 4. Tooling built

`pipeline/` (model-agnostic, CPU/GPU auto-detect, bf16 on CUDA):

| File | Purpose |
|---|---|
| `config.py` | `Config` dataclass + argparse, param **and FLOP** estimators, device/dtype resolution, `build_model()` |
| `data.py` | byte-level datasets (`--dataset shakespeare\|wikitext2`); wikitext-2 via HuggingFace parquet mirror (S3 bucket is dead) |
| `transformer.py` | GPT-2 baseline with the **same** `forward()`/`generate()` interface as `bdh.BDH` |
| `train.py` | cosine LR + warmup, grad clip, AMP, optional `torch.compile`, val/test eval, checkpoints |
| `eval.py` | load checkpoint → perplexity / bits-per-byte / samples |
| `bench.py` | ms/step + estimated FLOPs/step + effective TFLOP/s (`bench` subcommand) |
| `run.py` | CLI: `train` / `eval` / `bench` |

`bdh_linear.py`: `BDHLinear` = `BDH` with chunked **linear-time (state-space) attention**
(verified numerically equivalent to `BDH`, forward max|Δ|≤3e-7).

`scripts/gpu_bench.sh`, `scripts/gpu_train.sh`: ready-to-run presets for **RTX 4090 (24 GB)**
and **DGX Spark (128 GB)**.

**Baseline param matching:** GPT's `n_layer` is auto-solved so its param count ≈ BDH's
(e.g. 727,488 vs 712,704 at D=96, mult=24 — 2.1% over). The GPT exposes the identical interface,
so the same train/eval loop drives both.

---

## 5. Evaluation methodology

- **Byte-level** LM (vocab=256, no tokenizer) — the natural fit for BDH's `vocab_size=256`.
- **Step-matched**: both models, same data, same hyperparams, same step count → measures
  *learning per update*.
- **Compute-matched**: GPT is given extra steps so its wall-clock ≈ BDH's → measures
  *quality per unit compute*.
- Hyperparams are **identical** for both (no per-model tuning) — the fairest default, but it
  means a transformer-specific LR sweep could narrow the step-efficiency gap (unchecked lever).

---

## 6. Results

Hardware: 8-core CPU, 31 GB RAM (torch 2.13 CPU, `torch.compile` on). Default model:
**D=96, n_layer=4, n_head=4, mult=24 → 712K params** (GPT auto-matched 727K).

### 6.1 tiny-shakespeare (1 MB, saturated)

| comparison | BDH | GPT |
|---|---|---|
| step-matched @600 | **5.14** | 9.68 |
| compute-matched (BDH 600 s ≈ GPT 4400 s) | 5.14 | 5.03 |

### 6.2 wikitext-2 (11 MB, non-saturated, has test set) — seed 1337

| | BDH (3000 steps, 1.12 ep) | GPT (24000 steps, 9 ep) |
|---|---|---|
| per-step | ~1750 ms | ~210 ms (7.5× cheaper) |
| wall-clock | ~87 min | ~84 min |
| **val ppl** | **3.88** | 3.90 |
| **test ppl** | **3.99** | 4.02 |

Step-matched @3000: **BDH 3.88 vs GPT 5.57** (val). BDH reaches 3.88 in 3000 steps; GPT needs
~21000 steps (7×) to match. The 7× step-efficiency is exactly offset by the 7.5× cost → **tie**
on compute, confirmed on a non-saturated corpus.

### 6.3 wikitext-2 — seed 42 (reproducibility check)

| | seed 1337 (val/test) | seed 42 (val/test) |
|---|---|---|
| BDH (3000 steps) | 3.88 / 3.99 | 3.89 / 3.98 |
| GPT (24000 steps) | 3.90 / 4.02 | 3.90 / 4.00 |
| GPT step-matched @3000 | 5.57 / 5.64 | 5.65 / 5.72 |

The tie reproduces cleanly across seeds. Averaged over both seeds at equal compute: **BDH test
ppl ≈ 3.98 vs GPT ≈ 4.01** — BDH is consistently a hair *better* on the held-out test set, not
just tied. (Minor note: checkpoints are saved without a seed tag, so `out/*_best.pt` hold the
most recent run — add the seed to the filename before doing multi-seed sweeps.)

### 6.4 `mlp_internal_dim_multiplier` sweep (tiny-shakespeare, 600 steps)

| mult | BDH ppl | GPT ppl (step-matched) | GPT ppl (compute-matched) | verdict |
|---|---|---|---|---|
| 24 | 5.14 | 9.68 | 5.03 | tie |
| 8 | 6.05 | 9.97 | 6.08 | tie |
| 4 | 7.00 | 10.17 | 5.85 | GPT wins |
| 2 | 8.10 | 9.68 | 6.43 | GPT wins |

BDH's **step-efficiency is architectural** (wins at every width, even with *fewer* params at
mult=2: 104K vs 172K), but **compute-efficiency is width-dependent** — it only breaks even at
the default width. The wide latent is doing the real work (5.14 → 8.10 as mult drops).

---

## 7. Cost / FLOP analysis

`bench` tool, block=128, batch=32, compiled:

| model | params | ms/step | GFLOP/step | eff. TFLOP/s |
|---|---|---|---|---|
| BDH | 712K | 1675 | 99 | 0.06 |
| GPT | 727K | 258 | 20 | 0.08 |

The ~6.5× wall-clock gap = **5× FLOP gap** × **1.3× lower FLOP/s**. The 5× is the three
wide-latent GEMMs (`3·B·nh·T·D·N` per layer, N=576/head = `mult·D/nh`) — irreducible given the
architecture; attention is only ~⅓ of BDH's cost at block 128.

---

## 8. Linear attention (`BDHLinear`)

Chunked state-space scan (`state += v·kᵀ; out = state·q` with RoPE via `R^{t-τ} = R^t R^{-τ}`),
**numerically equivalent** to `BDH` (forward ≤3e-7, 100-step training reproduces BDH's loss to
~1e-3).

Forward+backward sweep (eager, batch=8):

| block | BDH (quadratic) | BDHLinear | GPT | lin/quad |
|---|---|---|---|---|
| 128 | 505 ms | 576 ms | 74 ms | 1.14× |
| 256 | 1058 ms | 1198 ms | 135 ms | 1.13× |
| 512 | 2572 ms | 2819 ms | 340 ms | 1.10× |
| 1024 | 7137 ms | 6657 ms | 754 ms | **0.93×** |
| 2048 | — | 16952 ms | — | — |
| 4096 | — | 42924 ms | — | — |

Findings: crossover ≈ **T=1024**, and the win is **modest** (7–28%) because the linear
attention's own `O(T·D·N)` cost (state read + write = `2·B·nh·T·D·N`) is comparable to the
MLP. The **real** value of `BDHLinear` is **memory**: `O(1)` state vs the `O(T²)` scores matrix
(which at T=4096 is ~2 GB for batch 8). So linear attention is *required* for long context,
marginally faster for moderate context, and neutral at short context.

---

## 9. Conclusions

1. **BDH's step-efficiency is real, large (~6-7×), and architectural** — it survives width
   reduction and a second corpus.
2. **It does not translate to compute-efficiency**: at equal compute BDH **ties** a matched
   GPT — confirmed across two seeds on wikitext-2 (BDH test ≈ 3.98 vs GPT ≈ 4.01, BDH
   marginally ahead) and on tiny-shakespeare (marginally behind).
3. **The wide sparse latent is both the asset and the liability** — it produces the
   step-efficiency and the 5× FLOP cost.
4. **No plasticity at inference**; the "Hebbian" claim is the linear-attention state. If
   train-time/in-context plasticity is the real thesis, it is not in the shipped code.

Bottom line: BDH is best evaluated as a **quality-per-update** architecture, and its compute
competitiveness is **unproven at best** in the open-source form. GPU (bf16, tensor cores) is
the missing data point.

---

## 10. Future levers (do not forget)

1. **GPU baseline (critical, blocked on hardware).** Run `scripts/gpu_bench.sh` then
   `scripts/gpu_train.sh` on the RTX 4090 / DGX Spark. Answers: (a) does the bf16 FLOP ratio
   shrink (favoring matmul-heavy BDH), (b) does the tie break at 10-100M params where the shared
   embedding overhead washes out, (c) real tensor-core efficiency of BDH's few-large-GEMM vs
   GPT's many-small-GEMM profile.
2. **Per-step cost reduction.** The 3 wide-latent GEMMs are the bottleneck. Levers, in order:
   bf16 (wired), larger batch (GPU), structured/sparse low-rank weights, fp8, fused kernels.
   The `multiplier` sweep showed shrinking width trades quality away — so cost reduction must
   come from *efficiency*, not *capacity*.
3. **Step-efficiency regime.** If BDH's edge is "fewer updates", test it where updates are
   scarce: few-shot in-context learning, convergence-speed-at-fixed-data, data-scarce domains.
4. **Scaling-law curve.** Verify step-efficiency survives D=256→512 (25M→100M params).
5. **Long-context via `BDHLinear`.** The linear kernel is *required* for T≫1024 (memory);
   measure its training quality vs quadratic at T=2048+ on GPU.
6. **Hyperparameter fairness.** A transformer-specific LR/optimizer sweep to confirm the
   step-efficiency gap is not a tuning artifact.
7. **Real plasticity / in-context weight updates.** The paper's "Hebbian" is a static recurrent
   state; an actual plastic-synapse variant is the unshipped thesis and an open research
   direction.
8. **Reproducibility of headline claims.** The 97.4% Sudoku result is from Pathway's *internal*
   implementation, not this repo — do not cite it as this code's capability.

---

## 11. Appendix — reproduction

```bash
pip install -r requirements.txt          # torch>=2.0, numpy, requests, pyarrow

# benchmark (CPU or GPU)
python -m pipeline.run bench --model bdh --dataset wikitext2 --block-size 128 --batch-size 32

# train + eval (wikitext-2, seed 1337)
python -m pipeline.run train --model bdh --dataset wikitext2 --max-iters 3000 --warmup-iters 300 --lr-decay-iters 3000 --eval-interval 600 --eval-iters 50
python -m pipeline.run train --model transformer --dataset wikitext2 --max-iters 24000 --warmup-iters 2400 --lr-decay-iters 24000 --eval-interval 3000 --eval-iters 50
python -m pipeline.run eval --model bdh --dataset wikitext2 --eval-iters 200

# GPU
bash scripts/gpu_bench.sh                # 25M-param BDH (D=256), fits 24 GB
HARDWARE=spark bash scripts/gpu_train.sh # ~101M params
```

Default model config: `n_embd=96, n_layer=4, n_head=4, mlp_internal_dim_multiplier=24,
vocab_size=256, block_size=128, batch_size=32`.
