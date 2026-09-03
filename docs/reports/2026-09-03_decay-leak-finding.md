# Finding: AdamW weight-decay leak bypasses the frozen-path mask (F-decay-leak)

Date: 2026-09-03. Author: Quinn (A0-Quinn seat, backend glm-5.3-flash via B.AI).
Status: root cause proven causally; fix + repair tool + pre-registered re-run (RA2b) attached; operator sign-off received same day.

## Finding

The growth path in `pipeline/train.py` promised "old neurons + embed + lm_head
frozen" and enforced it as an element-wise **gradient** mask
(`gate_param_masks`, applied as `p.grad.mul_(keep)` before the optimizer
step). AdamW's decoupled weight decay is not a gradient effect: every
parameter with `grad is not None` is multiplied by `(1 - lr_t * wd)` each
step, **including elements whose gradient is zero**. Old-segment synapse
tensors have gradients (masked to zero) and therefore decayed every step.

Per-phase decay factor on the RA2 protocol (wd=0.1, lr 1e-3, warmup 1000,
cosine to min_lr=1e-4, 10k steps):

    prod_t(1 - lr_t * 0.1) = 0.57978

## Evidence

1. **Uniform multiplicative signature.** For every resume segment p in
   `lt_last` (segments of fi, hu, bg, et, el, sk, sl), the segment equals its
   own phase-p checkpoint times a single scalar c, with c = 0.57978^(20-p)
   matching to 5 decimals. Fit over encoder/encoder_v/decoder per segment:
   residuals 5.6e-06 .. 1.5e-05 (pure scaling; no structural change).
2. **The three untouched tensor classes are exactly the ones AdamW cannot
   touch.** `embed.weight` and `lm_head` are `requires_grad_(False)` ->
   `grad is None` -> skipped by the optimizer entirely (bit-identical across
   all 7 resume phases). `attn.freqs` is a buffer, never in the optimizer
   (bit-identical). Everything with grad != None but zero-masked gradients
   decayed. This is the mechanism, not a correlation.
3. **Causal proof (repair splice).** Dividing each old segment in `lt_last`
   by its measured decay factor restores routed fi serving on fi's own route
   (32768, window=128 instrument) from 16.36 to **8.68** — exactly the p13
   baseline value measured 7 phases earlier. Prediction = observation.
4. **Blast radius.** Base-ladder segments decayed too: the es segment in
   `lt_last` fits its own es checkpoint at c = 5.9e-05 (predicted
   0.57978^18, resid 2.1e-05). All grown checkpoints of all ladder families
   (RA2, arm G, arm GR, PoC arms, seed-42) share the leak wherever the
   zero-grad mask coexisted with AdamW weight decay. Independent
   confirmation: the weight-atlas scan series (OC-GLM-200, bdh-cl #64)
   found old-neuron structure stable ("neither frozen nor reorganized") —
   uniform magnitude erosion, the same signature from the topology side.

## Which numbers are affected

- **Invalidated as evidence:** any claim of "zero forgetting by
  construction" for the frozen path; joint-serving degradation magnitudes
  (RA2 p20 milestone: bg 1649 / el 891 are partly decay artifacts); the
  within-family acquisition-gap pattern of the RA2 report (Appendix A) —
  decay erosion of earlier segments at later acquisition times is now a
  third, possibly dominant confound beside width and interference.
- **Survives:** routing correctness (36-40/40 per language), the route-aware
  serving principle (routed >> joint even under decay), acquisition values
  as *descriptions of what was learned by phase end*.

## Fix (train.py, this commit)

Bit-exact frozen path via **step-end restore**: at growth time the three
frozen regions (encoder/encoder_v `[:, :, :n_old]`, decoder head-major
`[:, :n_old, :]`) are snapshot; after every optimizer step they are restored.
A wd=0 parameter group alone would NOT be bit-exact: AdamW applies decay
before the zero-gradient update inside the same step. embed/lm_head need no
restore (grad None removes them from the optimizer); attn.freqs is a buffer.
Overhead ~2-3 ms/step on GB10 against ~2.8 s/step.

## Repair tool (scripts/repair_decay.py, this commit)

Measures each segment's decay factor directly against its own source
checkpoint (least-squares c-fit over the three synapse tensors), divides it
out, flags residuals > 0.01 (the .200-era en segment flags at 5.4e-02,
likely compiled-vs-eager kernel delta on top of decay; the measured c is
still the best correction). Writes `*_repaired.pt` + JSON sidecar; inputs
are read-only. Built-in self-test: the target's own segment must fit with
c = 1.000 and resid 0.

Reference semantics: repair returns each segment to its creation-time state
("no decay after creation" counterfactual). It is NOT the no-leak-trained
ladder: later phases trained against decayed earlier segments. For that,
see the pre-registered re-run.

## Pre-registered re-run (scripts/ladder_ra2b.sh, this commit)

RA2b = full 20-phase RA2 protocol with the fix, own namespaces (run-name
`ladRA2b`, logs `ladRA2b_*`, analysis `ladder_ra2b_analysis.txt`; RA2
artifacts untouched). Pre-registered predictions:

- **H-decay-1:** old segments are bit-identical to their creation state at
  every later phase (verify by c-fit: c = 1.000, resid 0).
- **H-decay-2:** joint-serving degradation of early languages shrinks
  dramatically vs RA2 (RA2 p20 milestone bg 1649 / el 891).
- **H-decay-3:** the within-family acquisition gaps shrink markedly if
  decay erosion was their driver (RA2: later sibling 3-4x worse on the base
  ladder; resume ladder flat).

## Open items

- en-segment resid 5.4e-02 (compiled-vs-eager suspected): repair helps, not
  perfectly; RA2b reruns en natively eager.
- The degradation cliff mechanism (why x0.11 segments serve at baseline but
  x0.038/0.022 collapse: LN scale-invariance vs attention energy balance) is
  open and independent of the leak fix.
- Re-measure joint serving on repaired checkpoints before citing any
  interference magnitudes (bdh-cl #68 erratum).
