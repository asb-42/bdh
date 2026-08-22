# Handover: BDH Repository → RTX 4090 Machine

**Date:** 2026-08-22
**Purpose:** everything a fresh session (human or AI assistant) needs to continue work on this repository on a GPU machine, without re-deriving context.

---

## 1. Project context

This repo implements **BDH-GPU**, the tensor-friendly variant of the *Dragon Hatchling* architecture (Pathway, arXiv:2509.26507): a biologically-inspired LLM that replaces the Transformer MLP with a ReLU-lowrank block in a huge neuron dimension `n`, and softmax attention with **linear attention whose keys/queries live in that same dimension**. Paper TeX source was reviewed in full; findings are reflected in the code and in [`docs/plans/2026-08-22_paper-parity-plan.md`](plans/2026-08-22_paper-parity-plan.md).

Key papers/docs in-repo:
- [`docs/reports/2026-08-22_code-review.md`](reports/2026-08-22_code-review.md) — review findings (all fixed).
- [`docs/plans/2026-08-22_paper-parity-plan.md`](plans/2026-08-22_paper-parity-plan.md) — paper-parity work plan (all 6 phases done) + known follow-ups.
- [`README.md`](../README.md) — upstream README.

## 2. Current state (git)

All work is committed on `main`; working tree clean. Recent history, newest last per phase:

```
72b7065 Mark paper-parity plan phases complete; note follow-ups
067fe9f Phase 6: BDH-GPU' variant (gating + multi-layer logits)
de0d04f Phase 5: interpretability tooling
22f7276 Phase 4: model merging CLI
c282338 Phase 3: ALiBi state damping
9b28ed6 Phase 2: --no-bptt training flag
eef5f7e Phase 1: state carry-over and truncated BPTT
d3bb447 Apply minor improvements from code review
74f53c8 Fix numerical fidelity issues
1b70b0a Fix critical issues from code review
6e6d300 Add code review report
```

**First action on the new machine:** clone/pull the repo, then run the verification suite (§6) to confirm the environment.

## 3. Codebase map

| File | Role |
|---|---|
| [`bdh.py`](../bdh.py) | Core model: `BDHConfig`, quadratic attention with RoPE (+ALiBi), stateful `forward(idx, targets, state) -> (logits, loss, new_state)`, `detach_state`/`state_to_cpu` helpers |
| [`bdh_linear.py`](../bdh_linear.py) | `BDHLinear`: chunked linear-attention scan (state-space form), fp32 ρ accumulation, ALiBi decay |
| [`bdh_prime.py`](../bdh_prime.py) | `BDHPrime` ("bdh-prime"): xLSTM-style gated state update + per-layer logit merging (paper's BDH-GPU′) |
| [`pipeline/config.py`](../pipeline/config.py) | `Config` dataclass = full CLI surface; `build_model`, FLOP/param estimators |
| [`pipeline/data.py`](../pipeline/data.py) | Byte-level datasets (shakespeare, wikitext2); `ByteStream` sequential continuation batches |
| [`pipeline/train.py`](../pipeline/train.py) | Shared training loop: cosine LR, clipping, TBPTT windows, checkpointing |
| [`pipeline/eval.py`](../pipeline/eval.py) | Checkpoint eval: perplexity + samples |
| [`pipeline/bench.py`](../pipeline/bench.py) | ms/step + TFLOP/s benchmark (CUDA-synced), synthetic data |
| [`pipeline/merge.py`](../pipeline/merge.py) | Model merging along neuron dimension n |
| [`pipeline/analyze.py`](../pipeline/analyze.py) | Sparsity / neuron-graph / synapse-trace analyses |
| [`pipeline/transformer.py`](../pipeline/transformer.py) | GPT baseline (same 3-tuple interface) |
| [`scripts/gpu_train.sh`](../scripts/gpu_train.sh), [`scripts/gpu_bench.sh`](../scripts/gpu_bench.sh) | Preset drivers (`HARDWARE=4090` fits this machine) |

Model variants (all byte-level, vocab 256): `bdh` (quadratic attn), `bdh-linear` (chunked state-space scan — the paper's primary training form), `bdh-prime` (gated + logit merge), `transformer` (param-matched GPT baseline).

## 4. Design decisions & invariants (do not break these)

1. **Forward API:** every model returns `(logits, loss, new_state)`; `new_state` is `{"pos": int, "layers": [...]}`. Quadratic layers carry `{"k", "v"}` KV caches; linear layers carry ρ tensors `(B, nh, D, N)` in **fp32**. `pos` continues absolute RoPE positions across calls.
2. **fp32 state accumulation:** ρ updates run under `torch.autocast(..., enabled=False)` — bf16 drift on long sequences was measured and fixed. Keep it.
3. **TBPTT semantics** ([`pipeline/train.py`](../pipeline/train.py)): `--tbptt-horizon K > 1` accumulates loss over K minibatches (graph retained through carried state) and steps the optimizer once per window; horizon 1 detaches every step. `--no-bptt` forces horizon 1.
4. **Exactness invariant:** processing `[A; B]` in one forward must equal `forward(A)` then `forward(B, state=s1)` for all attention paths (verified to ≤1.3e-07). Any change to attention/state code must re-pass this test (§6).
5. **Head-major layout:** `decoder` is `(nh*N, D)` laid out per head; merging concatenates *within* heads. RoPE freqs are quantized to pairs (`get_freqs` q=2) — pairs share frequencies by design.
6. **Checkpoints** store config as a plain dict → loadable with `torch.load(..., weights_only=True)`; `pipeline/eval.py` and `pipeline/merge.py` fall back for legacy pickled-dataclass checkpoints.
7. **Generation** crops context to `BDHConfig.block_size` if set (pragmatic O(T²) guard; conceptually BDH has no context limit).

## 5. Environment setup (RTX 4090)

```bash
# system: recent NVIDIA driver (>= 550), CUDA runtime not needed (wheel bundles it)
cd bdh
python3 -m venv .venv && source .venv/bin/activate   # or: uv venv .venv
pip install -r requirements.txt                       # torch>=2.3 pulls the CUDA build
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Notes:
- `.venv/` is gitignored; recreate it here (it did not transfer).
- Keep `--dtype auto` (resolves to bf16 on the 4090). The fp32 ρ accumulation stays exact regardless.
- `pyarrow` is only needed for wikitext2; shakespeare works without it.

## 6. Post-setup verification (run before any real experiment)

```bash
# 1. exactness invariant: split vs carried-state, both paths, with/without ALiBi
python - <<'EOF'
import torch, bdh
from bdh_linear import BDHLinear
torch.manual_seed(0)
A = torch.randint(0, 256, (2, 32)); B = torch.randint(0, 256, (2, 24))
AB = torch.cat([A, B], dim=1)
for slope in (0.0, 0.05):
    for mk in (lambda c: bdh.BDH(c), lambda c: BDHLinear(c, chunk_size=8)):
        cfg = bdh.BDHConfig(n_layer=2, n_embd=32, n_head=4,
                            mlp_internal_dim_multiplier=8, dropout=0.0,
                            attn_window=0, alibi_slope=slope)
        m = mk(cfg); m.eval()
        with torch.no_grad():
            lf, _, _ = m(AB); _, _, s = m(A); l2, _, _ = m(B, state=s)
        d = (lf[:, 32:] - l2).abs().max().item()
        assert d < 1e-4, (slope, d)
print("equivalence OK")
EOF

# 2. quick GPU smoke train (TBPTT on)
python -m pipeline.run train --model bdh-linear --dataset shakespeare \
  --n-layer 4 --n-embd 96 --mlp-internal-dim-multiplier 24 --batch-size 16 \
  --block-size 512 --max-iters 50 --eval-interval 50 --log-interval 10 \
  --carry-state --sequential-batches --tbptt-horizon 4 --device cuda

# 3. benchmark all three variants (establishes the compute-matching ratio)
bash scripts/gpu_bench.sh
```

## 7. Experiment plan for the 4090 (suggested order)

1. **Baseline runs** — `HARDWARE=4090 bash scripts/gpu_train.sh` (25M BDH + param-matched GPT on wikitext-2, 10k iters). Use the bench ms/step ratio to give GPT compute-matched steps.
2. **State carry-over study** — rerun BDH with `--carry-state --sequential-batches --tbptt-horizon {1,4,8}` at equal wall-clock; expect faster loss-per-token than random batches (paper App. B regime).
3. **No-BPTT comparison** — same run with `--no-bptt`; check whether cross-sentence structure degrades (paper Sec. 5.2 predicts yes).
4. **ALiBi sweep** — `--alibi-slope {0, 0.02, 0.05, 0.1}`; plot per-position val loss beyond block size (needs a small eval script addition if desired).
5. **Interpretability** — `python -m pipeline.run analyze --ckpt ... --sparsity` (expect ~5% non-zero xy on a trained model), `--graph --beta <noise floor>` (modularity/power-law), `--synapse l:h:i:j` traces.
6. **Merging** — train two models on disjoint data slices, `python -m pipeline.run merge`, evaluate merged without finetuning (paper Sec. 5.1).
7. **bdh-prime comparison** — `--model bdh-prime` vs `bdh-linear` at matched params/compute. Note: the gated scan is a per-token recurrence — much slower per step than the chunked vanilla scan; keep models small or budget accordingly.

Memory guidance (4090, 24 GB): the 25M preset (n=32768, d=256, L=6–8, block 512, batch 32) fits comfortably. If raising batch/block, watch ρ memory: `B × nh × N × D × 4 bytes × n_layer` for fp32 state plus autograd graph within the TBPTT window.

## 8. Known limitations / follow-ups

- Gated scan (`bdh_prime.py`) is a per-token Python loop — correct but slow; parallel-scan kernel is the main outstanding engineering item.
- ALiBi damping rate is uniform across heads; per-head/per-edge u(i,j) is future work.
- Quadratic-path KV cache grows unbounded when `attn_window=0` (trimming enabled by default at 1024 tokens; trimming is only exact when ALiBi damping makes distant tokens irrelevant anyway).
- `scripts/*.sh` presets reference a "DGX Spark" tier that was never validated here; ignore or update.
- The `.venv` used during development was CPU-only; results above were verified functionally, not for GPU performance. Re-run §6 first.

## 9. Quick CLI reference

```bash
python -m pipeline.run train --model {bdh|bdh-linear|bdh-prime|transformer} \
    --dataset {shakespeare|wikitext2} [options]
python -m pipeline.run eval  --model M --dataset D --tag best
python -m pipeline.run bench --model M [--batch-size B --block-size T]
python -m pipeline.run merge --ckpt-a A.pt --ckpt-b B.pt --out M.pt
python -m pipeline.run analyze --ckpt CKPT (--sparsity | --graph --head H --beta B | --synapse l:h:i:j --prompt P)
```

Useful flags: `--carry-state --sequential-batches --tbptt-horizon K --no-bptt --alibi-slope X --compile/--no-compile --device auto --dtype auto`.
