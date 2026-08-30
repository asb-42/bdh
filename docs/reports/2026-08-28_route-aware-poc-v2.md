# Route-Aware PoC v2 Results — 2026-08-28

**Status:** protocol-compliant redo of the invalidated v1 run
**Reviewed v1:** `docs/reports/2026-08-28_route-aware-poc.md` (commit `93e6ce4`, review `c37d263`)
**Pre-registration:** `docs/plans/2026-08-28_route-aware-poc.md` (commit `1bc796c`)

## 1. What v1 got wrong and what v2 fixes

| Issue (per review `c37d263` + erratum `2b2381e`) | v1 | v2 |
|---|---|---|
| Attention during growth | FROZEN (protocol violation: plan §3.1 requires unfrozen) | **UNFROZEN** (`--no-freeze-attn`, new `freeze_attn` config flag, default True) |
| route_alpha semantics | Full-mix fraction (`(1−α)·prefix + α·full`); v1 α=0.1 = 90% prefix, as registered (erratum confirms code was right, report misread it) | **Prefix fraction**: `loss = α·prefix + (1−α)·full`, printed explicitly in log |
| Alpha calibration | Single point | **Sweep** new-α ∈ {0.5, 0.9, 1.0} |
| Eval route grid | 2252..45040, calibrated for the 554M stack — routes > N_full=5632 all clamp to the full model, so 17 of 19 routes were identical | **256-step grid 256..5632** (22 distinct prefixes within the trained range) + phase-boundary grid 1536/3584/5632 |
| LR schedule | warmup 30 / decay 300 (defaults) | **warmup 1000 / cosine over 10k** (plan §3.1) |
| P1 base | warmup-30 EN base | retrained with protocol LR schedule |

**Alpha semantics mapping** (new α = prefix fraction vs old α = full-mix fraction):
new α = 1 − old α. The registered setting ("10% full-forward mix") is new α=0.9
(old α=0.1). The sweep new-α {0.5, 0.9, 1.0} covers full-mix {0.5, 0.1, 0.0}.

## 2. Setup

- 3 phases: EN (base, no route-aware) → DE (growth+route-aware) → ES (growth+route-aware)
- 24 → 56 → 88 mult (+2048 neurons/head per growth phase); final N_full = 88·512/8 = **5632** neurons/head
- Old neurons + embed + lm_head frozen each phase; attention **unfrozen**; fresh optimizer per phase
- Batch 4×512, 30 MB per language, 10k steps per phase, warmup 1000, cosine to 1e-4
- Specialists (de/es alone, no growth, same protocol) for P-Route baselines

**Exact training commands** are in `scripts/ladder_poc_ra2.sh` (runs P1, specialists, then 3 alpha branches × {P2, P3}).

## 3. Per-phase acquisition (P-Acq)

| Branch | P1 EN base | P2 DE best | P3 ES best |
|---|---|---|---|
| α=0.5 | 2.29 | **2.25** | **2.25** |
| α=0.9 | — | 2.41 | 2.49 |
| α=1.0 | — | 2.45 | **5.52** |

Specialists: de 2.19, es 2.18. Thresholds: pass ≤2.6, partial 2.6–3.5, fail >3.5.

Note: P-Acq (5.52) and P-Route (5.64) for α=1.0 ES differ because they measure
different things — P-Acq reports the best validation PPL during training (best
checkpoint), while P-Route reports routed serving PPL on the boundary grid (the
model is forced to route through the ES prefix). Both are valid; the 0.12 nat
gap reflects serving-vs-training evaluation conditions.

## 4. Routing evaluation protocol

**Exact commands** (one per branch, batch 1 to avoid OOM):

```
python scripts/eval_router.py out/bdh_europarl_poc-ra2-{a05,a09,a10}-p3_best.pt \
  --routes 256,512,768,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4352,4608,4864,5120,5376,5632 \
  --domains bg:data/europarl/europarl-v7.bg-en.bg.txt,...,sl:data/europarl/europarl-v7.sl-en.sl.txt \
  --batch 1        # 19 domains, 40 crops each, window 128

python scripts/eval_router.py <ckpt> --routes 1536,3584,5632 --domains <same> --batch 1
```

Phase blocks: EN = routes ≤1536, DE = routes 1792..3584, ES = routes 3840..5632.
Primary P-Det uses the 3-way boundary grid (one route per phase block, direct analog
of the original 20-way task). The 22-route fine grid is diagnostic.

## 5. Falsifier results

### P-Det (3-way boundary grid, correct = own phase block)

| Branch | en→1536 | de→3584 | es→5632 | P-Det (de, es) |
|---|---|---|---|---|
| α=0.5 | 40/40 | 40/40 | 40/40 | **100%** |
| α=0.9 | 40/40 | 40/40 | 40/40 | **100%** |
| α=1.0 | 40/40 | 40/40 | 40/40 | **100%** |

Block-level detection on the fine grid is also 100% for all three trained languages
in all branches (en 39/40 at 1536 with one crop at 1280 — still EN block; de splits
5/35 between 3328 and 3584 at α=0.5, both DE block; es splits 10/30 between 5376 and
5632, both ES block).

### P-Route (routed vs specialist, nats, threshold ≤0.3)

| Branch | de gap | es gap | verdict |
|---|---|---|---|
| α=0.5 | 0.131 | 0.099 | **PASS** |
| α=0.9 | 0.200 | 0.201 | **PASS** |
| α=1.0 | 0.215 | 0.949 | FAIL (es) |

Routed PPL (boundary grid): α=0.5 de 2.50 / es 2.41; α=0.9 de 2.68 / es 2.67;
α=1.0 de 2.72 / es 5.64. Specialists: de 2.19, es 2.18.

### Summary vs pre-registered decision rule

| Falsifier | α=0.5 | α=0.9 | α=1.0 |
|---|---|---|---|
| P-Det ≥95% | PASS | PASS | PASS |
| P-Route ≤0.3 nats | PASS | PASS | FAIL |
| P-Acq ≤2.6 ppl | PASS | PASS | FAIL (P3 5.52) |

**Decision rule outcome: PASS at α=0.5 and at the registered α=0.9 (10% full mix).**
→ proceed to a full ladder with route-aware training, per pre-registration.

## 6. Monotonicity vs alpha

P-Det is saturated at 100% across the sweep (no headroom). P-Route and P-Acq
degrade monotonically with α: de gap 0.131→0.200→0.215, es gap 0.099→0.201→0.949,
P3 PPL 2.25→2.49→5.52. At α=1.0 (pure prefix loss, no full mix) the ES block is
still perfectly detectable but dysfunctional under full-forward serving (5.52 PPL
vs specialist 2.18): it was trained only under its own prefix, so its outputs are
mis-scaled when combined with the older blocks. The 10% full-forward mix is
load-bearing for block health, not for detectability. α=0.5 gives the best
retention/acquisition while retaining 100% detection.

## 7. Routable-growth signature

The boundary-grid routing shows the trained languages route to *exactly* their own
block, not wider: en→1536 (never 3584/5632), de→3584 (never 5632), es→5632. If
the grown neurons were generic capacity, extra blocks would not hurt older
languages and they would route to the widest prefix. Instead, each additional
block *degrades* the previous languages' likelihood, so the router picks the
exact trained block — the neurons are language-specific by construction. Untrained
languages (distractors) spread across routes and mostly fall to 5632 (the only
prefix that serves them at all); their routed PPL is catastrophic (e.g. bg/el
~10⁵–10⁶) because no block was ever trained on them.

## 8. Conclusion

H-PoC **PASSES** at 2–3 phase scale: route-aware training with unfrozen attention
makes each phase's language routable (100% detection, ≤0.2 nats retention gap,
acquisition ≤2.49 PPL) at α ∈ {0.5, 0.9}. The v1 null result was an artifact of
three compounding setup errors (frozen attention, misaligned route grid, LR
schedule). Per the pre-registered decision rule, the next step is a full 19-phase
ladder with route-aware training (α=0.9, 10% full mix).

## 9. Reproducibility

- Training: `scripts/ladder_poc_ra2.sh` (self-contained; ~2.5 h on a 4090)
- Code: `neuron_mask` forward in `bdh.py`; `route_aware`/`route_alpha`/`freeze_attn`
  in `pipeline/config.py` + growth path in `pipeline/train.py`
- Checkpoints: `out/bdh_europarl_poc-ra2-{p1,spec-de,spec-es,a05-p2,a05-p3,a09-p2,a09-p3,a10-p2,a10-p3}_best.pt`
- Raw router outputs: `out/logs/poc_ra2_{a05,a09,a10}_routing.txt` (fine grid),
  `out/logs/poc_ra2_3route.txt` (boundary grid)
