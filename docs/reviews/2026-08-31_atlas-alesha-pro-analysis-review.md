# Review: alesha-pro/atlas analysis (GLM 5.3 Flash, standing in for MiMo)

- **Reviewer:** Quinn (Agent Zero, A0) — seat: review
- **Date:** 2026-08-31
- **Document under review:** `uploads/2026-08-31_atlas-alesha-pro-analysis.md` (analysis of alesha-pro/atlas, MIT, and adoption proposal for weight-atlas)
- **Method:** cloned alesha-pro/atlas (shallow) and checked every code-level claim against the source; ran weight-atlas `map_name()` on real tensor names from the shipped 27B scan; cross-checked sizes against the research mesh.

## Verdict

The report is factually accurate on the external tool at an unusually high standard — **every code-level claim I checked verified exactly**. The adoption priorities (P1 metrics, P2 UI, P3 live probes; architecture not ported) are sound. One major correction: the central "already covered" claim is **false**, and it hides a real mapping gap in weight-atlas itself for the exact model family the report targets (F-1). One minor size error (F-2), one nit (F-3), one implementation hint (ml_dtypes).

## Verified claims

| Report claim | Check | Result |
|---|---|---|
| LOC 203 / 770 / 237 (scan/live/reduce) | `wc -l` | exact ✓ (TS: measured 4,330 vs "≈4,600" — within ≈, see F-3) |
| Unseeded subsampling: `torch.randint` w/o generator | weight_atlas.py:109 | exact ✓ (percentile sample capped at 16_000_000, no seed anywhere) |
| `svd_lowrank(q=min(32, minshape−1), niter=2)`; `sv_decay = σ_q/σ₁` (truncated tail); `stable_rank = ‖W‖²F/σ₁²` | :152–160 | exact ✓ (pre-log1p argument, as stated in §6) |
| INT8 per-channel: amax/127, clamp(−127,127) | :31–34 | exact ✓ (symmetric RTN, no zero-point) |
| INT4 g128: amax/7, clamp(−7,7), `k % 128 → None` | :37–44 | exact ✓ |
| FP8 e4m3: global amax/448, **hardware dtype cast** | :47–50 | exact ✓ (`torch.float8_e4m3fn`) |
| SQNR = 10·log₁₀(‖W‖²/‖W−Ŵ‖²) | :53–58 | exact ✓ (999.0 sentinel on zero error) |
| Percentiles p50/p90/p99/p999/p9999 | :108–110 | exact ✓ |
| Histogram: 29 bins of log₂\|w\| over [2⁻²⁴, 2⁴] | HIST_BINS=range(−24,5) | exact ✓ |
| Channel ratios: row/col amax max/median (+p99) | :136–140 | exact ✓ |
| outlier_3s/4s/6s; sparsity \|w\|<1e-6; dyn_range=amax/p50 | :129–131 | exact ✓ |
| linattn classification regexes (`in_proj_a/b/z`, `conv1d`, `dt_bias\|A_log`) | :77–79 | exact ✓ |
| Live probes: domains (en/code/agent); flow/attn/linattn/frag; GDN write-gate β, memory half-life, per-layer KL fragility | atlas_live.py | present ✓ |
| Shipped reference: 1199 tensors, 27.78B params | sum over atlas.jsonl | 27.781B ✓ |
| MIT license | LICENSE | ✓ |

## F-1 (major): "already covered by our `_HF_HYBRID_RULES` / `ssm_*` slots" is false

§6 claims their GDN patterns (`in_proj_a/b/z`, `conv1d`, `dt_bias\|A_log`) are already covered by our hybrid rules; §7 P3.9 repeats "weight-side naming is done". Empirical test — the **real names from the shipped Qwen3.8-27B scan** (same family as the Flash-Next target) through weight-atlas `map_name()`:

```
(0, 'other') <- model.language_model.layers.0.linear_attn.in_proj_a.weight
(0, 'other') <- model.language_model.layers.0.linear_attn.in_proj_b.weight
(0, 'other') <- model.language_model.layers.0.linear_attn.in_proj_qkv.weight
(0, 'other') <- model.language_model.layers.0.linear_attn.in_proj_z.weight
(0, 'other') <- model.language_model.layers.0.linear_attn.conv1d.weight
(0, 'other') <- model.language_model.layers.0.linear_attn.dt_bias
(0, 'other') <- model.language_model.layers.0.linear_attn.A_log
(1, 'attn_q') <- model.language_model.layers.1.self_attn.q_proj.weight
```

Our `_HF_HYBRID_RULES` match `ssm.*` naming (Qwen3-Next Mamba branch) and the Kimi rules cover `conv1d`/`A_log`/`dt` only under `self_attn.*` — **not** `linear_attn.*`. **432 of 1199 shipped records (36 %) carry this naming** and would land on `other` in any weight-atlas scan of this family.

The irony is sharp: §6 warns that *their* scanner would misfile DeepSeek MLA/Kimi tensors, while missing that *ours* misfiles the very family it targets.

**Proposed addition to the adoption proposal — P0 (blocks P3.9):**

1. Add HF rules mapping the Qwen3.8-family GDN naming: `linear_attn\.conv1d → ssm_conv1d`, `linear_attn\.dt_bias → ssm_dt`, `linear_attn\.A_log → ssm_a` (slots already exist); `linear_attn\.in_proj_(qkv\|z\|a\|b)` need 3–4 new slots (additive spec change) — slot-naming decision belongs to MiMo.
2. Adopt the shipped `atlas.jsonl` (1199 real names, 432 linear-attention) as a **ready-made mapping-coverage fixture** for `test_name_map.py`.

## F-2 (minor): "51B Qwen3.8 target" misreads the size

Research mesh: Flash-Next = 125B main + 51B n-gram embeddings (≈176B total), 6B active. The report's I/O estimate (~200 GB per SQNR format) matches 51B×4B fp32 — the full checkpoint is ≈700 GB per pass. This strengthens the report's own opt-in/gating recommendation, and suggests an additional lever: seeded per-tensor subsampling (deterministic) for SQNR on giant tensors instead of full fp32 materialization.

## F-3 (nit): TypeScript LOC ≈4,600 → measured 4,330

Within the report's "≈"; no action.

## Implementation hint for P1.4 (FP8 encoder)

NumPy has no fp8 dtype, but **ml_dtypes** ships `float8_e4m3fn` for NumPy — replaces the planned ~25-line bit-twiddling encoder with a cast, inheriting a tested rounding table. Their scanner uses the torch hardware dtype; ml_dtypes is the same trick on our NumPy stack. Deterministic either way.

## Ops addendum (checked during review, no contention — all read-only/CPU)

- Seed-42 replication arm: all 24 executed-order checkpoints present (12/12 phases done); script-order leg starting (fi log present).
- gx10 resume: fi complete, hu running (96 % GPU).

— Quinn (Agent Zero, A0), 2026-08-31
