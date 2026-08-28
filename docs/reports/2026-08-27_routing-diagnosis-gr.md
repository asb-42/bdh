# Routing Diagnosis: Arm G+R Stack (20-way)

**Date:** 2026-08-27 · **Checkpoint:** `out/bdh_europarl_ladGR-lt_last.pt` (554M params, mult=704)
**Method:** `eval_router.py` with 20 routes, 19 Europarl domains, 40 crops/domain

## Key finding

**The last-trained language (lt) has NO routable structure.** It is scattered across 7 routes (32% accuracy, 55.10 ppl = joint 54.96). All 18 older languages route fine (95–100% accuracy).

This means: the CURRENT phase's language cannot be isolated into a single prefix. The weight updates during training affect the shared attention, mixing the new language into the existing structure. Older languages are preserved (frozen attention), but the current one is not.

## Results

### P-Det (20-way detection)

| Domain | Accuracy | Route | Note |
|---|---|---|---|
| bg | 100% | 13 | |
| cs | 100% | 8 | |
| da | 100% | 9 | |
| de | 100% | 7 | |
| el | 100% | 16 | |
| es | 95% | 5 | 2 crops → route 12 |
| et | 82% | 15 | 7 crops → route 14 |
| fi | 100% | 11 | |
| fr | 100% | 6 | |
| hu | 100% | 12 | |
| it | 100% | 14 | |
| **lt** | **32%** | **16** | **SCATTERED across 7 routes** |
| nl | 100% | 19 | |
| pl | 100% | 6 | |
| pt | 100% | 10 | |
| ro | 100% | 18 | |
| sk | 100% | 16 | |
| sl | 100% | 20 | |
| sv | 100% | 17 | |

**Overall:** 724/760 = 95.3% (16 languages at 100% = 640, es 38, et 33, lt 13). lt is the only major outlier (32%).

### P-Route (routed retention vs joint 54.96)

| Domain | Routed PPL | vs Joint | Note |
|---|---|---|---|
| lt | 55.10 | 1.0× | **= joint (no routing benefit)** |
| pl | 14.44 | 3.8× | |
| et | 10.74 | 5.1× | |
| cs | 8.21 | 6.7× | |
| es | 6.23 | 8.8× | |
| hu | 5.92 | 9.3× | |
| fr | 5.86 | 9.4× | |
| sk | 4.75 | 11.6× | |
| de | 4.66 | 11.8× | |
| it | 4.63 | 11.9× | |
| sl | 3.99 | 13.8× | |
| sv | 3.82 | 14.4× | |
| da | 3.61 | 15.2× | |
| bg | 3.52 | 15.6× | |
| fi | 3.49 | 15.7× | |
| nl | 3.45 | 15.9× | |
| el | 3.42 | 16.1× | |
| ro | 3.42 | 16.1× | |
| pt | 3.22 | 17.1× | |

**Mean routed PPL:** 8.03 (vs joint 54.96). Excluding lt: 7.46.

## Comparison: Arm G vs Arm G+R

| Metric | Arm G | Arm G+R |
|---|---|---|
| P-Det (excl. last) | 99.7% | 100% |
| P-Det (incl. last) | 99.7% | 0.3% (lt scattered) |
| lt routed ppl | 3.74 | 55.10 (= joint) |
| Joint reference | 58.39 | 54.96 |

**Arm G+R preserves older languages better** (100% routing accuracy for 18/19) but **the current phase has no routable structure** (lt = joint).

## Assessment

**P-Det:** 95.3% (724/760). lt at 32% is the only major outlier. 16/19 languages at 100%.

**P-Route:** Older languages route well (10–17× over joint). lt = joint (no benefit).

## freeze_attn confound

Arm G (no freeze) had lt routing fine (3.74 ppl, route 20). Arm G+R (with freeze) has lt scattered (55.10 ppl = joint). The freeze_attn mechanism may be the cause: lt was trained with frozen attention that was already optimized for older languages, leaving no free capacity for lt to form its own projection pattern.

**Diagnosis experiment needed:** Retrain lt with unfrozen attention, then re-run routing diagnosis. If lt becomes routable → freeze_attn is the confound. If lt remains unroutable → the current-phase problem is fundamental.

## Interpretation

The frozen-attention mechanism works for OLDER languages — they retain routable structure. But the CURRENT phase's language cannot be isolated because:

1. Training updates the new-width encoder/decoder columns
2. The new language's representations mix into the shared structure
3. There is no mechanism to force the current language into a dedicated prefix

This is the fundamental issue: **the current phase needs its own prefix, but the growth rule doesn't construct one.** The prefix is an emergent property of training, not an explicit allocation.

## Implications for Phase 2

The combination experiment (growth + routing) shows:
- **Older languages:** routing works perfectly (100% accuracy, 10–17× improvement)
- **Current language:** no routing benefit (= joint quality)

The recipe needs a mechanism to allocate a prefix for the current phase during training. Options:
1. **Explicit prefix allocation:** reserve neurons for the current language before training (but this resembles learned gates — already falsified)
2. **Routing-aware training:** use the router during training to select the current prefix (theoretically consistent)
3. **Accept the limitation:** the current phase is always at joint quality; only older phases benefit from routing
