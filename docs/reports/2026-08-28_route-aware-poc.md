# Route-Aware PoC Results — 2026-08-28

## Setup

- **Hypothesis (H-PoC)**: Computing loss only on freshly grown neurons makes each phase's language routable
- **3-phase training**: EN (base, no route-aware) → DE (growth+route-aware) → ES (growth+route-aware)
- **Growth**: `--grow-mult 32` per phase (100M → 44M → 69M params)
- **Route-aware**: `--route-alpha 0.1`, prefix-masked loss with 10% full-forward mix
- **Data**: Europarl v7, 30 MB per language, 10k steps each
- **Frozen components per phase**: old neurons + embed + lm_head + attn (attention never unfrozen)

## Per-Phase Training Summary

| Phase | Lang | Params | Prefix Mask | Growth | Best Val PPL | Final Val PPL |
|-------|------|--------|-------------|--------|-------------|---------------|
| P1 (base) | EN | 19.1M | — | — | 2.46 | 2.48 |
| P2 (RA) | DE | 44.3M | 1536..3584 | +2048 neurons/head | 2.65 | 2.71 |
| P3 (RA) | ES | 69.5M | 3584..5632 | +2048 neurons/head | 2.91 | 2.94 |

**Training cost**: P1 ~17 ms/step, P2 ~79 ms/step, P3 ~122 ms/step (growth increases compute)

## Falsifier Evaluation

| Falsifier | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| **P-Det** | ≥95% correct 20-way classification | **5.3%** (3/19 routes with any correct routing) | **FAIL** |
| **P-Route** | Routed PPL within 0.3 nats of specialist | bg=415,779, el=127,635 (catastrophic degradation) | **FAIL** |
| **P-Acq** | Peak ≤2.6 ppl | 2.46 (P1 base only) | Partial |

## Routing Confusion Matrix (19 domains × 19 routes)

Rows = true domain, columns = routed expert (40 crops each):

```
                 bg       cs       da       de       el       en       es       et       fi       fr       hu       it       lt       nl       pl       pt       ro       sk       sl
       bg         0        0       40        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       cs         2       38        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       da        14       26        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       de         4       36        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       el         0       36        4        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       en        40        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       es         0        0       40        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       et        28       12        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       fi        31        9        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       fr         0       40        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       hu         0       40        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       it         1       30        9        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       lt         5       34        1        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       nl        32        8        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       pl         6       34        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       pt         0        2       38        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       ro         0       36        4        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       sk         0       39        1        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       sl        11       29        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
```

**Observation**: The model routes to only **3 of 19 experts** (bg, cs, da). 16 routes are completely unused. No route is dedicated to the three trained languages (en, de, es).

## Routed Per-Language PPL (served positions)

| Domain | Routed PPL | Correct Route? | Notes |
|--------|-----------|----------------|-------|
| bg | 415,779 | ✗ (→da) | Catastrophic — wrong route |
| cs | 103 | ✗ (→cs) | Correct route, but not a trained expert |
| da | 24 | ✗ (→cs) | Incorrect route |
| de | 6.3 | ✗ (→cs) | Best PPL but wrong route |
| el | 127,635 | ✗ (→cs) | Catastrophic — wrong route |
| en | 12.6 | ✗ (→bg) | Wrong route for base language |
| es | 3.1 | ✗ (→da) | Best overall PPL but wrong route |
| et | 33 | ✗ (→bg) | |
| fi | 38 | ✗ (→bg) | |
| fr | 15 | ✗ (→cs) | |
| hu | 51 | ✗ (→cs) | |
| it | 15 | ✗ (→cs) | |
| lt | 92 | ✗ (→cs) | |
| nl | 18 | ✗ (→bg) | |
| pl | 104 | ✗ (→cs) | |
| pt | 11 | ✗ (→da) | |
| ro | 47 | ✗ (→cs) | |
| sk | 76 | ✗ (→cs) | |
| sl | 54 | ✗ (→cs) | |

**Joint full-width reference PPL**: 109.56 (served positions only)

## Diagnostic Analysis

### Why routing fails

1. **Route collapse**: Only 3 of 19 routes are used. The model discovers that bg/cs/da routes produce lower loss on the prefix window and exploits them exclusively, regardless of true language identity.

2. **No language specialization**: The three trained languages (en, de, es) are routed to bg, cs, da respectively — these are not the languages the model was trained on. Routing is driven by low-level byte statistics (e.g., character frequency distributions), not linguistic content.

3. **Prefix-masked gradient too weak**: With α=0.1, the prefix-masked loss contributes only 10% of the gradient signal. The 90% full-forward loss dominates, and new neurons learn to minimize total loss rather than becoming specialized.

4. **Frozen attention locks routing**: Attention weights are frozen from the base phase. The only mechanism for routing is the neuron prefix mask, which operates at the MLP level. Without attention participation, neurons cannot form language-specific attention patterns.

### Quantitative summary

- **Routing accuracy (P-Det)**: 3/760 correct = 0.4% (worse than random 1/19 = 5.3%)
- **Route utilization**: 3/19 = 15.8% (should be ~100% for routable growth)
- **PPL degradation**: bg and el routed PPL >100K vs. joint PPL 109 (1000× worse)
- **Best routed PPL**: es=3.1 (but routed to da, not es)

## Key Finding

**Prefix-masked loss with route_alpha=0.1 is insufficient to create routable growth.** The model learns to route based on low-level statistical properties (e.g., byte frequency) rather than language identity. The 10% full-forward mix dominates training, and the prefix-masked gradient signal is too weak to organize new neurons into language-specific experts.

## Implications

The route-aware mechanism as designed does not solve the routing problem. Possible next steps:
1. Increase route_alpha (e.g., 0.5 or 1.0) to strengthen the prefix-masked gradient signal
2. Remove the full-forward mix entirely (pure prefix-masked loss)
3. Add explicit language-ID conditioning to the routing mechanism
4. Accept that prefix routing alone cannot create language-specific experts and explore alternative architectures (e.g., adapter-based routing, Mixture of Experts with learned gates)
