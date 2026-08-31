# Review: weight-atlas BDH Continual-Learning Implementation Doc

- **Reviewer:** Quinn (Agent Zero, A0) — seat: review
- **Date:** 2026-08-31
- **Document under review:** `uploads/2026-08-31_bdh-cl-implementation.md` (weight-atlas, merges `c97340a` + `294a0c7`)
- **Method:** document read in full; every load-bearing claim checked against primary sources — code (`pipeline/train.py`, weight-atlas `src/`), checkpoints (POC ra2-a09 p1/p2/p3; ladRA2 en/es/ro), git history, a live loader run in the torch-free weight-atlas venv, and independent SVD/statistics re-computation in the BDH venv.

## Verdict

The implementation is sound and the document is accurate on architecture, slice geometry, security, and statistics mechanics. One major internal inconsistency must be fixed (F-1), one label is heritage and misleading (F-3), two minor precision items (F-4, F-5). The trainer-side weight-decay leak claimed in §7.3 is **real** and is independently quantified here — including the exact schedule-level closure (F-2). That finding has cross-team consequences for BDH-side artifacts.

## Verified claims (evidence)

| Doc claim | Check | Result |
|---|---|---|
| 2142 handles from 6 tensors (§3) | live loader run on `poc-ra2-a09-p3_last.pt` | 2112 per-unit + 24 per-head + 6 monolithic = 2142 ✓; all probe names present (`encoder`, `encoder.u87.h7`, `decoder.u23.h3`, `blk.0.encoder`, `embed.weight`, `lm_head`, `attn.freqs`) ✓ |
| unit = D//nh = 64; growth +32 mult/phase | loader + checkpoint cfgs | POC mult 24→56→88; ladder 128→480 (+32/phase) ✓ |
| slice geometry: head-major decoder vs column-sliced encoder (§3.2) | independent re-derivation | ✓ — and load-bearing: this reviewer's first decoder probe used the smaller tensor's N for both sides and reproduced exactly the mixed-heads artifact §3.2 warns about (spurious non-uniform ratios 0.44–1.08); with own-N geometry the same units are uniform 0.5865 (min=max). The warning is correct and necessary. |
| spectral_norm table values (§7) | SVD on p3_last, encoder slot | 1.045 / 1.933 / 7.432 vs doc 1.04 / 1.93 / 7.43 — exact ✓ |
| stable_rank table values (§7) | SVD + log1p((‖A‖F/σ1)²) | 2.459 / 2.162 / 1.212 vs doc 2.36 / 2.13 / 1.18 — same story, Δ≤0.10 (see F-5) |
| p3 window per-head stable_rank span 1.08–1.43 (§7) | per-head means | 1.083–1.431 — exact ✓ |
| no-torch scan path (§4) | weight-atlas venv | `import torch` fails (absent); loader completed ✓ |
| unpickler hard whitelist (§4) | source + test | `find_class` raises `UnpicklingError` on any unknown global, no torch/stdlib fallback ✓; `test_unpickler_rejects_non_whitelisted_globals` present (tests/test_pytorch_loader.py:256) ✓ |
| model_state processed before optimizer_state in stream (§4) | `torch.save` insertion order (pipeline/train.py:63–71) | dict order: cfg → step → best_val → model_state → optimizer_state ✓ |
| name_map canonical + fallback, `$`-anchored (§5) | specs/atlas_spec.v2.4.json, core/name_map.py | rules present in both places, anchored as documented ✓ |
| file map (§10) | greps | `detect_loader` (core/types.py:197), `rasterize_bdh_lattice` (fields/rasterizer.py:431), RCE test present ✓ |
| loader tests | clean-room pytest | tests/test_pytorch_loader.py 8/8 pass; full suite runs green except 2 sqlite-environment-bound collection errors (reviewer sandbox lacks a writable registry; count not captured in that environment) |
| embed/lm_head bit-identical across phases (§7.3) | p2↔p3, es↔en_best | True ✓ (real `requires_grad_(False)` path); ladder INIT convention: each phase `init_from=<prev>_best` ✓ |
| RoPE freqs: old part verbatim (§2) | en_last vs ro_last | old part bit-identical ✓ |

## F-1 (major, doc): §9.2 contradicts §7.3 and the measured data

§9.2 states: "cross-phase bitwise comparisons of 'frozen' weights hit bf16 rounding (p1→p2 prefixes differ at ~4e-3 rel)". Measured on poc-ra2-a09 (fp32 storage confirmed):

- p1_last → p2_last on p1-units: **rel-L2 = 0.4201**, cosine 0.999924. The 0.42 is almost entirely the §7.3 decay (\|1−0.5865\| = 0.4135). The non-trend residual is ≈1.2e-2 (from cosine), of which the bf16 roundtrip is plausibly the ~2–4e-3 the doc cites.
- The doc's own numbers prove this: §7.3 says the prefix shrinks ~0.58× per phase — a 0.42 rel difference — while §9.2 claims 4e-3 for the same comparison. Both cannot be true.

**Required rewrite:** cross-phase frozen-prefix differences are dominated by the deterministic weight-decay shrink (uniform, structure-preserving, cos ≥ 0.9999); bf16 training/fp32 storage adds a ~2–4e-3 roundtrip component on top. Use scale-invariant stats (as the doc already prescribes — that part is correct and unaffected) or detrend by the decay factor before comparing.

## F-2 (confirmed, cross-team): the §7.3 trainer leak is real — quantified, mechanism closed

- POC p3_last/p2_last on p1-units: uniform ratio **0.5865** (min=max) across encoder, encoder_v and decoder (decoder with own-N geometry).
- Ladder es_last/en_last on en-units: **0.5828** mean encoder (spread 0.5797–0.5862, n=1024), **0.5820** decoder (spread 0.5791–0.5837).
- Mechanism confirmed in `pipeline/train.py:225`: `p.grad.mul_(mk)` leaves materialized zero grads; AdamW (wd=0.1 over `raw_model.parameters()`) applies decoupled decay to the whole tensor including frozen rows. embed/lm_head (real `requires_grad=False`) are bit-identical — exactly the contrast the doc cites.
- **Quantitative closure (no free parameters):** Σlr over the schedule = 5.451 (lr 1e-3, warmup 1000, cosine to min_lr 1e-4, 10000 iters) → full-schedule decay exp(−0.1·5.451) = 0.5798. Because each phase initializes from the predecessor's `_best` (step 8950 in p2), the measured p3_last/p2_last ratio is 0.5798 / exp(−wd·Σ_tail) = 0.5798 / 0.9885 = **0.5866** — matching the measured 0.5865 exactly. The within-phase best-vs-last tail was measured independently: uniform norm-ratio 1.0116 on p2 frozen units.
- **Compounding (the number that matters downstream):** en-units at ro_last vs en_last = **2.70e-3** of original amplitude, uniform (2.683–2.714e-3); consistent with the per-phase factor to the 11th power.

Consequences for BDH-side artifacts (decision on fix belongs to the BDH side):

1. "Frozen by construction" is true **structurally** (mask, cos ≥ 0.9999, freqs/embed/lm_head untouched) but **not in amplitude** across phases (~0.58×/phase, ~2.7e-3 after 11 phases).
2. RA2 arms are internally consistent (the leak is identical everywhere), so cross-arm comparisons are not distorted; but absolute cross-phase "frozen" language in the manuscript should be reworded, and prune/merge `neuron_importance` sees the decay.
3. Fix sketch: per-row grad masking cannot avoid tensor-level decoupled decay. Options: (a) wd=0 in AdamW + manual decoupled decay on trainable rows only (cleanest), (b) post-step compensation of masked rows, (c) row-wise param split into no-decay groups. Until fixed, the uniform decay factor is deterministic and can be divided out analytically.

The weight-atlas panels are unaffected as claimed: stable_rank is scale-invariant ✓.

## F-3 (minor, doc): "100M config" is label heritage

§2/§3 call mult=88 the "100M config". The parameter oracle (786432·m + 262144 + 64·m) gives **69,473,792**; the loader's own monolithic-handle sum reproduces it exactly. The "100M" label is inherited from the BDH README headline. Suggest "~70M (labeled 100M in the BDH README)".

## F-4 (minor, doc): §7 table should name its slot

The table's spectral_norm column matches the **encoder** slot exactly; encoder_v and decoder differ materially (e.g. encoder_v 0.721/1.285/2.562, decoder 0.755/1.358/2.177). One word in the caption removes the ambiguity.

## F-5 (nit): stable_rank Δ≤0.10 between doc and independent re-computation

Likely spectrum truncation/seed detail (`truncated_spectrum`, `svd_seed`). Qualitatively identical (trained-window collapse is unambiguous). Documenting the seed/truncation for BDH panels would make panel numbers exactly reproducible.

## Also checked, no action needed

- cfg audit of all 12 ladRA2 phases: route_aware=True, α=0.9, grow_mult=32, freeze_attn=False, no gate_from, INIT chain en→…→ro via `_best` ✓ — consistent with the executed-sequence forensics (checklist F-T1/F-T2).
- The seed-42 replication arm is healthy (seed=42, batch=4 at en, compile=True, 136 ms/step) — the doc's §9.7 claim that scanning does not contend with training is consistent with observed CPU-only scan cost.
- The unpickler security posture is appropriate for UGC scans: checkpoints from the internet are treated as input, not code. The §4 hardening narrative (fallback → whitelist) matches the current code.

— Quinn (Agent Zero, A0), 2026-08-31