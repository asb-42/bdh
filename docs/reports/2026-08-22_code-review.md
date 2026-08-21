# Code Review — BDH Project

**Date:** 2026-08-22
**Scope:** `bdh.py`, `bdh_linear.py`, `train.py`, `pipeline/` (config, data, transformer, train, eval, bench, run), `scripts/`, `requirements.txt`

## Summary

The codebase is clean and well-organized. The core math was verified by inspection:

- The chunked linear-attention scan in `pipeline/../bdh_linear.py` is exactly equivalent to the quadratic attention form in `bdh.py` (intra-chunk `tril(-1)` scores + inter-chunk outer-product state read before update).
- Parameter-count and FLOP estimators in `pipeline/config.py` match the actual model architectures.
- Batch sampling index bounds are correct (no off-by-one).
- Cosine LR schedule handles warmup/decay edge cases correctly.

Findings are prioritized below: 🔴 bugs/correctness, 🟡 numerical fidelity, 🟢 minor improvements.

---

## 🔴 Bugs / correctness issues

### 1. Missing CUDA synchronization in benchmark timing

**File:** `pipeline/bench.py` (timing loop)

`time.time()` wraps only kernel *enqueue*, not execution. Without `torch.cuda.synchronize()` immediately before starting the timer and again before computing `dt`, GPU timings are systematically underestimated (the loop exits as soon as the last kernel is queued).

This matters because the measured BDH-vs-GPT ms/step ratio from `scripts/gpu_bench.sh` drives the compute-matched step counts used in `scripts/gpu_train.sh`.

**Fix:**

```python
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(steps):
    step()
if device.type == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t0) / steps
```

### 2. `BDH.generate` never crops the context

**File:** `bdh.py` (`generate` method)

`idx_cond = idx` means the sequence length T grows unboundedly during generation → O(T²) attention cost and memory blowup on long generations. The GPT baseline crops to `block_size` (`pipeline/transformer.py`). This asymmetry also makes BDH-vs-GPT generation slightly unfair in `pipeline/eval.py`.

**Fix:** crop to a configurable max context (e.g. add `block_size` to `BDHConfig`) or at minimum document the asymmetry.

### 3. Dropout active during generation unless caller remembers `eval()`

**File:** `bdh.py` (`generate`), same pattern in `pipeline/transformer.py`

`generate` runs under `@torch.no_grad()` but does not toggle eval mode; dropout stays on if the caller forgets `model.eval()`. Both current call sites (`train.py`, `pipeline/eval.py`) happen to call it, but this is fragile.

**Fix:** call `self.eval()` / restore previous training state inside `generate`, or document the requirement in the docstring.

### 4. fp16 autocast without GradScaler in bench

**File:** `pipeline/bench.py` (`step()`)

On pre-Ampere GPUs (`--dtype auto` resolves to fp16), backward without gradient scaling can produce inf gradients. Harmless for pure speed measurement, but the loss becomes meaningless and inf-handling paths can perturb step time.

**Fix:** use a `GradScaler` when dtype is fp16, or restrict benching to bf16/fp32.

### 5. No timeout on dataset download in root `train.py`

**File:** `train.py` (`fetch_data`)

`requests.get(data_url)` has no timeout and can hang indefinitely. `pipeline/data.py:_download` correctly uses `timeout=60`.

**Fix:** `requests.get(data_url, timeout=60)`; consider streaming to disk for larger files.

---

## 🟡 Numerical / fidelity concerns

### 6. bf16 state accumulation in `LinearAttention`

**File:** `bdh_linear.py` (`forward`)

Under autocast, the recurrent `state` accumulates outer products in bf16 across all chunks. The "mathematically identical" claim holds exactly in fp32 but drifts in bf16 for long sequences (T ≫ chunk_size). Options:

- keep `state` in fp32 (memory cost: B × nh × D × N per layer),
- or document the precision caveat / make accumulation dtype configurable.

### 7. `torch.compile` + `generate` recompilation

**Files:** `train.py` (lines ~99, ~127), `pipeline/train.py`

The compiled model triggers a recompile for every new sequence length during the Python generation loop (T, T+1, …). Generate with the uncompiled `raw_model`, or compile only after sampling. Less severe in the pipeline where generation happens in a separate eval process.

---

## 🟢 Minor improvements

8. **`pipeline/bench.py`** — the custom `_null` class duplicates `contextlib.nullcontext`. Also, `step()` closes over `model`, which is rebound by `torch.compile(model)` *after* `step` is defined — works because Python closures capture names not values, but the ordering dependency deserves a comment.
9. **Bench downloads full wikitext2** (`pipeline/bench.py`) just to feed batches; synthetic random bytes would remove a network dependency for benchmarking.
10. **Bench timing includes `optimizer.step()`** while the FLOPs estimate covers only fwd+bwd — effective TFLOP/s is slightly understated. Either exclude optimizer time or note the inclusion.
11. **`torch.amp.GradScaler(device=...)`** (`pipeline/train.py`) requires torch ≥ 2.3, but `requirements.txt` declares `torch>=2.0`. Bump the floor or guard the argument.
12. **Root `train.py`** has no grad clipping, LR schedule, or eval loop — fine for a minimal demo, but a pointer to `python -m pipeline.run train` in the README would steer users to the full pipeline.
13. **`pipeline/data.py` wikitext preprocessing** — `row.rstrip("\n")` strips *all* trailing newlines per row, then rows are rejoined with single `\n`; multi-newline structure is collapsed. Likely intentional normalization, but confirm it matches reference preprocessing.
14. **Checkpoint portability** — `pipeline/eval.py` uses `torch.load(..., weights_only=False)` because the pickled `Config` dataclass requires it. Storing `dataclasses.asdict(cfg)` would allow the safer `weights_only=True`.

---

## Verified correct (no action needed)

- RoPE with quantized freqs and shared Q/K rotation (`bdh.py`)
- Causal masking via `tril(diagonal=-1)` in both attention paths
- Chunked scan equivalence in `bdh_linear.py` (intra-chunk tril + inter-chunk state, updated after read)
- Parameter counts and FLOP models in `pipeline/config.py`
- Cosine LR schedule edge cases (`pipeline/train.py:get_lr`)
- Batch sampling bounds (`pipeline/data.py:ByteDataset.get_batch`)
- Checkpoint save/load round-trip including config (`pipeline/train.py`, `pipeline/eval.py`)
- Shell scripts' env-var presets and flag plumbing (`scripts/gpu_bench.sh`, `scripts/gpu_train.sh`)

## Suggested fix order

1. Items 1–5 (concrete bugs; small, isolated diffs)
2. Item 7 (compile/generate interaction)
3. Item 6 + remaining 🟢 items as convenient
