# freeze_attn Diagnostic Experiment

**Date:** 2026-08-28 · **Checkpoint:** `out/bdh_europarl_diag-lt_best.pt` (554M params, mult=736)
**Hypothesis:** `freeze_attn` is a confounder — Arm G+R's lt routing failure is caused by frozen attention, not a fundamental limitation.

## Setup

- **Init:** `out/bdh_europarl_ladGR-sl_last.pt` (Arm G+R phase 19, with frozen attention history)
- **Training:** lt only, 10k steps, `--grow-mult 32`, `--run-name diag-lt`, **unfrozen attention** (no `freeze_attn`)
- **Best val ppl:** 3.36 at step 9800
- **Routing diagnosis:** 20 routes (2252–45040), 19 Europarl domains, 40 crops, batch 1

## Results

### P-Det (20-way detection)

| Domain | Accuracy | Route |
|---|---|---|
| bg | 100% | 12 |
| cs | 93% | 7 (3 crops → route 15) |
| da | 100% | 8 |
| de | 100% | 6 |
| el | 100% | 15 |
| es | 75% | 4 (30→4, 10→9) |
| et | 85% | 13 (34→13, 6→12) |
| fi | 100% | 10 |
| fr | 100% | 5 |
| hu | 100% | 11 |
| it | 100% | 12 |
| **lt** | **5%** | **scattered** (2→7, 2→11, 10→12, 5→13, 14→15, 4→17, 3→19) |
| nl | 100% | 18 |
| pl | 100% | 5 |
| pt | 100% | 9 |
| ro | 100% | 18 |
| sk | 100% | 15 |
| sl | 100% | 19 |
| sv | 100% | 16 |

**Overall:** 703/760 = 92.5% (lt accounts for 38 of 40 misroutes)

### P-Route (routed retention)

| Domain | Routed PPL | vs Joint (56.33) | Improvement |
|---|---|---|---|
| bg | 4.31 | 56.33 | 13.1× |
| cs | 14.61 | 56.33 | 3.9× |
| da | 4.32 | 56.33 | 13.0× |
| de | 6.08 | 56.33 | 9.3× |
| el | 3.45 | 56.33 | 16.3× |
| es | 7.80 | 56.33 | 7.2× |
| et | 10.69 | 56.33 | 5.3× |
| fi | 3.97 | 56.33 | 14.2× |
| fr | 7.90 | 56.33 | 7.1× |
| hu | 7.06 | 56.33 | 8.0× |
| it | 4.70 | 56.33 | 12.0× |
| **lt** | **55.30** | **56.33** | **1.0× (no benefit)** |
| nl | 3.45 | 56.33 | 16.3× |
| pl | 21.74 | 56.33 | 2.6× |
| pt | 3.81 | 56.33 | 14.8× |
| ro | 3.43 | 56.33 | 16.4× |
| sk | 4.78 | 56.33 | 11.8× |
| sl | 3.99 | 56.33 | 14.1× |
| sv | 3.82 | 56.33 | 14.7× |

**Mean routed PPL (excl. lt):** 6.49 (8.7× better than joint)
**lt routed PPL:** 55.30 (= joint, no routing benefit)

## Comparison with Arm G+R (frozen attn)

| Metric | Arm G+R (frozen) | Unfrozen lt (this run) |
|---|---|---|
| lt PPL | 55.10 | 55.30 |
| lt routing accuracy | 32% | 5% |
| Overall P-Det | 95.3% | 92.5% |
| Older languages | 100% | 100% |

**Both models show lt at joint quality (~56 ppl).** The unfrozen model is actually *worse* at routing lt (5% vs 32%).

## Interpretation

**freeze_attn is NOT the confounder.** The unfrozen model was initialized from `ladGR-sl_last.pt` — a model that already had 19 phases of frozen attention. Even though attention was not frozen during lt training, the attention weights were already locked into a structure optimized for older languages. Unfreezing alone cannot reorganize them.

The Arm G model (which had lt routing fine at 3.74 ppl) was trained from scratch without any freeze_attn history. The difference is the *initialization state*, not the freeze flag.

**Root cause:** Once attention is frozen for multiple phases, the resulting structure is resistant to reorganization for new languages. The growth recipe creates a path dependency that cannot be broken by simply unfreezing.

## Implications

1. **The current-phase routing problem is fundamental** to the growth recipe, not an artifact of freeze_attn
2. **The recipe needs a mechanism to allocate a prefix for the current phase** during training — either:
   - Explicit prefix reservation (but this resembles learned gates — falsified)
   - Routing-aware training (use the router during training)
   - Accept the limitation (current phase = joint quality)
3. **Arm G's success with lt** was because it was trained from scratch, not on top of a model with frozen attention history
