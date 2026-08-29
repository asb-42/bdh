# Claim ↔ evidence ledger (manuscript rev 2, cb72631)

> Provenance: written by seat Pi (backend: Qwen3.8-Flash-Next, local 4090, blind run — no J-Space skill, no team context). Committed by Quinn after audit 2026-08-29; line refs verified against rev2 @ cb72631.

Verdicts: ✓ supported as stated · ~ supported but overstated/mislabelled · ? evidence not in paper
Ref = `cl-bdh-manuscript.tex` line. Cross-refs M1–M6 = REVIEW.md majors.

| # | Claim (abridged) | Ref | Evidence offered | Verdict |
|---|---|---|---|---|
| 1 | Sequential training forgets catastrophically: +1.59 EN / +2.15 DE nats (2-lang) | `:178` | Fig. 1; ppl↔nats arithmetic checks out (e^1.59=4.90, e^2.15=8.59) | ✓ |
| 2 | +12…+20 nats over 19 phases | `:38`,`:418` | Arm R numbers are ppl differences reported as nats | **~ M1** (likely +0.7 / +2.4 nats) |
| 3 | Forgetting persists with old params literally frozen; weight isolation ⇏ computation isolation | `:203` | Frozen growth arm, 2.26→5.40→12.49 ppl (=0.87/0.84 nats/block) ✓ | ✓ strongest result in paper |
| 4 | Not optimizer shock | `:192` | constant-LR worse on peaks + retention | ✓ (wording of direction/unit, minor #9) |
| 5 | Importance protection cannot fix it; no localized important subset | `:186` | null vs seed noise + drift audit 0.6–1.3/phase, 99.9 % >1 % importance | ✓ (and yields a falsifiable corollary) |
| 6 | Exact isolation ⇔ invariance + restriction equivalence | `:228`,`:654` | appendix induction correct under unbounded depth | ~ M5 (statement scope; one proof line missing) |
| 7 | Additive growth constructs the structure; hard selection exact | `:244` | S1/S2 separability + masked-forward reproduction tests; app:meas exact identity | ✓ conditional on LN-over-d (M2, unstated in §5) |
| 8 | Soft gates cannot be exact; collapses to hard endpoint | `:261`,`:682` | 2-D ReLU counterexample | ~ M5 (signed-gate gap; §5 vs appendix equation mismatch =4 vs =1) |
| 9 | Zero-forcing pins gates to zero | `:254`,`:674` | term-wise vanishing assumed | ~ M5 (needs non-negativity/disjointness) |
| 10 | Trained blocks expansive (1.05–1.89); uniform contraction bounds unsupported | `:276`,`:702` | power iteration + raw deviation grid, bounded conclusion stated | ✓ exemplary |
| 11 | Gates select, do not create structure | `:284` | mechanism-class argument + negative register #5 | ~ M5 (unlabelled, class undefined) |
| 12 | Growth acquires at near parity (+25 % width, within 0.09 nats) | `:296` | 2.22–2.33 ppl; ln gap 0.04 ✓ | ✓ |
| 13 | Hard selection numerically indistinguishable from specialists | `:303` | masked-forward identity tests | ✓ but = implementation validation of cor:prefix, not a discovery |
| 14 | Detection 100 % (compiled detector + 128-token likelihood) | `:306` | no n, no confusion matrix, no trivial-langID baseline | ~ minor #2 (cross-language half near-vacuous) |
| 15 | Soft mixture within 0.003 nats of discrete oracle; no cliff | `:311` | protocol stated in full (16 crops, 128 scoring positions, τ sweep) | ✓ (label "logit mixture" wrong — minor #1) |
| 16 | Soft leakage superlinear ~j^1.7; budget j≤0.15 | `:322` | endpoints .0023→.193 ⇒ fit 1.48 | ~ minor #6 |
| 17 | Merge recovers 48–77 % of forgetting; retrieval not generalization | `:344` | + subset ablation {EN,ES} leaves DE forgotten | ✓ good controlled evidence |
| 18 | Random-scatter prune keeps benefit; magnitude-ranked & block prune collapse | `:348`, reg. #3 | numbers only in register (15–16 vs 6.5–7.2 ppl @25 %) | ~ put table in main text |
| 19 | merge→prune→replay **equal to joint co-training ±0.01 ppl** | `:47`,fig5 | Table 2: 2.58/2.575/2.41 vs 2.33/2.23/2.23 → +0.10/+0.14/+0.08 nats | **~ M3** (closes 84–90 % of gap; ±0.01 is seed spread) |
| 20 | ~20 % replay in training = joint parity at +27 % budget | `:381` | 2.33/2.24/2.12 vs 2.33/2.23/2.23 | ✓ genuine parity branch |
| 21 | Round-robin w/o replay fails ⇒ forgetting is capacity competition, not recency | `:385` | single contrast; confounds (LR/noise/order) not separated | ~ B-5 under-ablated attribution |
| 22 | Growth without routing gives no protection ("overwrite") | `:437` | same arm shows masking recovers 10–19× → old computation survives | **~ M6** (mechanism sentence contradicts evidence) |
| 23 | Routing essential at ≥10 phases; acquisition robust (P-Acq pass) | `:423`,`:452` | 20-phase ladder, pre-registered falsifiers | ~ ("peak" ambiguity minor #5; intervals need phase indices) |
| 24 | Sparsity tracks data composition, not volume; capacity/data ratio association | `:471` | grid fig + slopes (+0.5 pp/lang, −4 pp register mix); 97.4 % @25M vs ~94 % @100M | ✓ and matches Pathway's ρ≈5 % regime |
| 25 | Cross-corpus loss comparisons void | `:480` | methodological stance, pre-committed | ✓ |
| 26 | No prior work formulates forward-path isolation under these assumptions | `:763` | triage table; no search described; Net2Net/morphisms, PCANets/Modularity-with-Invariance, Riemer'21, subspace-selection absent | ~ M4 |
| 27 | Answers BDH's deferred state→weight question | `:508` | consolidation = merge+prune+replay (weights), not σ→θ | ~ minor #13 scope |
| 28 | Reproducibility: exact params, seeds, scripts, reports | `:765` | param identity verified exactly ✓; but Arm G growth rate wrong units | ~ minor #4 |

**Pattern** `[INFERENCE]` c9/10: the *experiments* are honest and often better-controlled than the parent paper's; the failures cluster in (i) unit bookkeeping at the headline level, (ii) one attribution error about which branch reaches parity, (iii) architecture-primer fidelity, (iv) proof hypotheses left unstated where they are easy to add. All four are editing/derivation tasks, not new experiments — except Q1 (Arm G freezing), whose answer could *add* a result.
