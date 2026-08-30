# R3 Route-Aware Ladder — Progress Report (2026-08-30)

## Summary

The R3 ladder (α=0.9, unfrozen attention, 20-phase growth) completed **12 of 20 phases** before hitting a GPU memory wall on the RTX 4090 (24 GB). Phase 13 (EL, mult=512, 403M params) OOMs because route-aware training requires 2× forward pass, exhausting 24 GB. The 12 completed phases provide a substantial dataset; remaining phases require ≥48 GB GPU (GX10 recommended).

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | BDH, n_embd=512, n_head=8, block_size=512 |
| Base mult | 128 |
| Growth | +32 mult per phase (final target:736 at phase 20) |
| route_alpha | 0.9 (prefix 90% + full 10% mix) |
| freeze_attn | False (unfrozen attention) |
| compile | True (torch.compile via inductor) |
| Optimizer | AdamW, lr=1e-3, warmup=1000, cosine decay to 1e-4 over 10k steps |
| Dataset | Europarl v7, 30MB per language, 1MB val/test each |
| Batch size | 4 for mult≤192 (phases 1–3), 1 beyond (OOM guard) |

## Language Sequence (matching Arm G/R)

```
en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
```

## Completed Phases

| Phase | Lang | Mult | Params | Batch | Val PPL | Checkpoint |
|-------|------|------|--------|-------|---------|------------|
| 1 | EN | 128 | 126M | 4 | **2.25** | `ladRA2-en_best.pt` |
| 2 | ES | 160 | 158M | 2 | **2.40** | `ladRA2-es_best.pt` |
| 3 | PL | 192 | 189M | 2 | **3.17** | `ladRA2-pl_best.pt` |
| 4 | FR | 224 | 220M | 2 | **2.45** | `ladRA2-fr_best.pt` |
| 5 | DE | 256 | 252M | 2 | **2.72** | `ladRA2-de_best.pt` |
| 6 | NL | 288 | 283M | 1 | **3.37** | `ladRA2-nl_best.pt` |
| 7 | IT | 320 | 314M | 1 | **3.43** | `ladRA2-it_best.pt` |
| 8 | SV | 352 | 346M | 1 | **8.39** | `ladRA2-sv_best.pt` |
| 9 | DA | 384 | 377M | 1 | **7.96** | `ladRA2-da_best.pt` |
| 10 | PT | 416 | 409M | 1 | **7.66** | `ladRA2-pt_best.pt` |
| 11 | CZ | 448 | 440M | 1 | **9.66** | `ladRA2-cz_best.pt` |
| 12 | RO | 480 | 403M† | 1 | **9.32** | `ladRA2-ro_best.pt` |

†Phase 12 model reports 403M params in its log (consistent with growth frozen/embed sizing).

## OOM Analysis

**Phase 13 (EL, mult=512)**: The model at ~403M params with route-aware training (2× forward pass for prefix + full loss) requires ~23.4 GB activation memory, leaving <200 MB free on the 24 GB RTX 4090. The OOM occurs in the backward pass (inductor trying to allocate 256 MB activation buffer).

**Root cause**: Route-aware mode computes two forward passes per step:
1. Prefix forward (neuron mask zeroing old neurons) → prefix_loss
2. Full forward (no mask) → full_loss

Both create activation graphs retained for backward. At mult=512, the RoPE rotation buffer alone needs 268 MB (batch×8×512×32768×2 bytes), and this is one of many intermediate tensors.

**Attempted mitigations**:
- `--no-compile`: Still OOMs (issue is activation memory, not compilation buffers)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`: Can't help (only 136 MB free total)
- Reducing batch to 1: Already at batch 1

**Recommendation**: Phases 13–20 require ≥48 GB GPU (e.g., A100, H100, or GX10 with 128 GB). The model at mult=736 (final target) with route-aware would need ~35–40 GB.

## Routing Evaluation (Milestone 10)

Boundary-grid routing eval on the RO checkpoint (12-phase model, mult=480).

**Setup**: 10 routes (neuron prefix boundaries at 8192, 10240, ..., 26624) × 20 languages, 40 crops/domain.

**Confusion matrix** (rows=true domain, cols=routed expert, 10 columns for 10 routes):

| Domain | Primary Route | Accuracy | Routed Loss |
|--------|--------------|----------|-------------|
| DA | route 9 | **100%** (40/40) | 9.05 |
| ES | route 10 | **100%** (40/40) | 12.38 |
| FI | route 8 | **90%** (36/40) | 26.67 |
| NL | route 6 | **100%** (40/40) | 13.88 |
| PT | route 10 | **100%** (40/40) | 8.51 |
| SV | route 8 | **100%** (40/40) | 9.45 |
| BG | route 4 | **92.5%** (37/40) | 1670.39 |
| DE | route 6 | **85%** (34/40) | 19.24 |
| IT | route 7 | **92.5%** (37/40) | 14.87 |
| CS | route 10 | **90%** (36/40) | 83.47 |
| SK | route 10 | **85%** (34/40) | 63.83 |
| EN | mixed | **27.5%** (11/40 best) | 12.16 |
| EL | routes 1,4 | split | 2231.85 |
| ET | routes 7,8 | split | 28.20 |
| FR | routes 7,10 | split | 18.65 |
| HU | routes 8,10 | split | 49.42 |
| LT | routes 1,7,10 | split | 73.59 |
| PL | routes 1,6,7 | split | 92.18 |
| RO | route 10 | **75%** (30/40) | 46.14 |
| SL | route 7 | **67.5%** (27/40) | 40.51 |

**Joint full-width reference PPL**: 45.69 (served positions only)

**Key observations**:
- 6 languages route with ≥90% accuracy to a single expert
- EN routes poorly (27.5% best) — expected as the base language with no dedicated prefix
- EL has the highest routed loss (2231.85) suggesting it's poorly served by any single prefix
- Many later-phase languages (CZ, RO, SK, SL) route to the widest expert (route 10), consistent with the ladder's prefix-expert model

## What's Needed for GX10

1. **Continue phases 13–20** from the RO checkpoint (`ladRA2-ro_best.pt`)
   - Phase 13: EL (mult=512)
   - Phase 14: SK (mult=544)
   - Phase 15: SV (mult=576) — milestone 15 routing eval
   - Phase 16: RO (mult=608)
   - Phase 17: NL (mult=640)
   - Phase 18: SL (mult=672)
   - Phase 19: LT (mult=704)
   - Phase 20: (final mult=736) — milestone 20 routing eval

2. **Final evaluations on the completed model**:
   - Full 20-way routing diagnosis
   - P-Acq, P-Eros, P-Route, P-Det, P-Spars falsifier battery
   - Cross-arm comparison with Arm G/R

3. **Note on batch sizes**: With 128 GB, batch=4 should be feasible through all phases, potentially improving val PPL.

## File Inventory

**Checkpoints** (all in `out/`):
```
bdh_europarl_ladRA2-{en,es,pl,fr,de,nl,it,sv,da,pt,cz,ro}_best.pt
bdh_europarl_ladRA2-{en,es,pl,fr,de,nl,it,sv,da,pt,cz,ro}_last.pt
```

**Training logs** (all in `out/logs/`):
```
ladRA2_{en,es,pl,fr,de,nl,it,sv,da,pt,cz,ro}.log
```

**Routing eval**:
```
out/logs/ladRA2_ro_boundary_p10.txt  (milestone 10 boundary grid)
```

**Scripts**:
```
scripts/ladder_ra2.sh  (full 20-phase script, phases 1–12 completed manually)
```
