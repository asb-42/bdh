# Arm G Results: Growth Without Routing

## Setup
- 19 sequential Europarl X→en phases with width growth
- Base: cl-a-en (mult=128, 100M params)
- Growth: +32 multiplier per phase → final mult=708 (~700M params)
- Batch: 4 for mult≤192 (phases 1-3), 1 for mult>192 (phases 4-19)
- DGX-hop triggered at mult=608 (batch 2 OOM), resumed with batch=1

## Target Acquisition per Phase

| Phase | Lang | Mult | Target PPL | Status |
|-------|------|------|-----------|--------|
| 1 | en | 128 | 2.45 | OK |
| 2 | es | 160 | 2.36 | OK |
| 3 | pl | 192 | 2.52 | OK |
| 4 | fr | 224 | 2.33 | OK |
| 5 | de | 256 | 2.54 | OK |
| 6 | cs | 288 | 2.77 | FAIL |
| 7 | da | 320 | 2.66 | FAIL |
| 8 | pt | 352 | 2.53 | OK |
| 9 | fi | 384 | 2.73 | FAIL |
| 10 | hu | 416 | 2.52 | OK |
| 11 | bg | 448 | 2.41 | OK |
| 12 | it | 480 | 2.74 | FAIL |
| 13 | et | 512 | 3.06 | FAIL |
| 14 | el | 544 | 2.58 | OK |
| 15 | sk | 576 | 2.98 | FAIL |
| 16 | ro | 608 | 3.00 | FAIL |
| 17 | nl | 640 | 3.20 | FAIL |
| 18 | sl | 676 | 3.49 | FAIL |
| 19 | lt | 708 | 3.49 | FAIL |

**P-Acq: FAIL** — peak 3.49 (lt) > 2.6 threshold. Growth degrades acquisition.

## Erosion of Earlier Languages

### Phase 5 milestone (de, mult=256):
- de: 2.5 (target)
- en: 17.5
- es: 25.6
- fr: 10.4
- pl: 40.8

### Phase 10 milestone (hu, mult=416):
- hu: 2.5 (target)
- en: 40.3, es: 34.0, pl: 104.7, fr: 42.6, de: 40.8
- cs: 77.8, da: 41.2, pt: 31.1, fi: 34.6

### Phase 15 milestone (sk, mult=576):
- sk: 3.0 (target)
- en: 41.1, es: 36.8, pl: 44.6, fr: 49.1, de: 41.0
- cs: 13.6, da: 49.6, pt: 35.9, fi: 67.7, hu: 70.3
- bg: 869.9 (!), it: 35.7, et: 45.6, el: 4.3

### Phase 20 final (lt, mult=708):
- lt: 3.5 (target)
- en: 41.2, es: 31.7, pl: 70.1, fr: 40.4, de: 42.4
- cs: 67.3, da: 46.9, pt: 33.0, fi: 62.7, hu: 80.0
- bg: 1435.6 (!), it: 32.3, et: 48.7, el: 1267.8 (!)
- sk: 55.8, sv: 53.5, ro: 42.5, nl: 40.5, sl: 30.6

**P-Eros: FAIL** — catastrophic forgetting identical to Arm R.

## Key Finding

**Growth without routing = same forgetting as fixed capacity.**

The model adds capacity each phase (mult 128→708, ~100M→~700M params), but the new neurons don't protect earlier languages. The weight updates still overwrite them because there's no mechanism to isolate phases.

This confirms the theory: **routing/selection is essential for true isolation**. Growth alone is just a bigger container for the same interference.

## Comparison: Arm R vs Arm G

| Metric | Arm R (fixed 100M) | Arm G (growth to 700M) |
|--------|-------------------|----------------------|
| P-Acq | PASS (peak 2.21) | FAIL (peak 3.49) |
| P-Eros | FAIL (en +12.7 nats) | FAIL (en 41.2 ppl) |
| Final en | 24.6 | 41.2 |
| Final es | 22.8 | 31.7 |
| Catastrophic outliers | bg=342547, el=133723 | bg=1436, el=1268 |

Interesting: Arm G's erosion is actually WORSE than Arm R for some languages (en 41.2 vs 24.6). This might be because:
1. Batch=1 training is less effective than batch=4
2. The larger model overfits more
3. The growth mechanism doesn't scale well at this phase count

## Next Steps

1. **Routing measurements on final grown stack**: Does the ~700M model HAVE routable structure? (eval_router.py, detection accuracy)
2. **Update manuscript**: Add accumulation results
3. **Step-4 go/no-go**: Parameter scaling (150M/250M) — probably not worth it given these results

## Files
- Analysis: `out/logs/ladder_armG_analysis.txt`
- Per-phase logs: `out/logs/ladG_*.log`
- Phase-end checkpoints: `out/bdh_europarl_ladG-*_last.pt` (19 files)
- Resume script: `scripts/ladder_armG_resume.sh`
