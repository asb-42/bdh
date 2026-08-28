# Review: Route-Aware PoC Results

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-28_route-aware-poc.md` (commit `93e6ce4`)
**Pre-registration:** `docs/plans/2026-08-28_route-aware-poc.md` (commit `1bc796c`)
**Revision:** v2 — adds Erratum to Section 2 after reading the implementation diff

## Verdict

**FAIL in the pre-registered sense — but for a reason that invalidates the test of
H-PoC, not a clean falsification of the mechanism.** The run did not follow the
pre-registered protocol at its decisive point: attention was frozen throughout,
whereas the plan requires "Attention: unfrozen during each phase's training".
Before treating this as evidence against route-aware training, we need a
protocol-compliant run (attention unfrozen as registered). Until then, no ladder
decision.

## 1. Protocol violation: attention frozen vs. required unfrozen

Pre-registration (Section 3.1): **Attention: unfrozen during each phase's training**
(the rationale paragraph explicitly says "unfreezing alone does not fix a
frozen-history model; here there is no freeze history because we run from scratch").

Report (Setup): **"Frozen components per phase: old neurons + embed + lm_head +
attn (attention never unfrozen)"**.

This is a direct contradiction. The experiment therefore tests "route-aware with
frozen attention on a 3-phase stack", not "route-aware with unfrozen attention"
which is what H-PoC and the pre-registration describe. A null result under a
different treatment cannot falsify the registered hypothesis.

Note: implementing unfrozen attention was an open implementation question the plan
explicitly flagged (Section 7) and asked to be resolved *before* training — by
asking, if uncertain. Silent deviation plus non-disclosure in the report is the
compliance failure here, separate from the mechanism question.

## 2. ERRATUM (2026-08-28, post-push): the loss mix is the reverse of the report's claim

**Erratum to the first version of this review.** The first version repeated the
report's claim that "with alpha=0.1, the prefix-masked loss contributes only 10%
of the gradient signal." The code says the opposite (`pipeline/train.py`):

```python
loss = (1 - cfg.route_alpha) * prefix_loss + cfg.route_alpha * full_loss
```

alpha=0.1 weights the **prefix-masked loss at 90%** and the full-forward mix at
10% — which matches the pre-registered mixing suggestion. The report's cause #3
("prefix-masked gradient too weak") therefore misdescribes its own implementation;
the gradient composition was as registered.

What remains of this point: an alpha sweep (0.1 / 0.5 / 1.0) is still a reasonable
sensitivity check, but the gradient-strength objection no longer stands. The
load-bearing objections remain the attention protocol violation (Section 1) and
the route-grid mismatch (Section 3).

Process lesson: quantitative claims in a report must be checked against the diff,
not only against the report's internal consistency — this reviewer almost
propagated the misreading. Same protocol-hygiene class as the arithmetic errors
in the report series.

## 3. Route collapse and PPL explosions point to an eval/serving problem, not only training

- Only 3 of 19 routes are used; the trained languages (en, de, es) are routed to
  bg/cs/da. That is the expected signature of a router mismatch, not merely of
  weak training pressure.
- Routed PPL values of 415,779 and 127,635 are not cross-entropies of a serving
  model; they are divergences (log-scale blow-ups). Joint PPL = 109.56 here vs
  54.96–58.39 in the earlier ladder diagnosis. Something differs in the eval setup
  on this small stack — likely the fixed 20-route grid calibrated for the large
  grown stack (2252–45040) being applied to a model with only ~5632 neurons of
  prefix range, producing routes that do not match any phase block.

**Action:** verify the route grid against this model's actual neuron ranges before
re-running. If the routes do not align with the phase blocks, both P-Det and
P-Route numbers are not interpretable.

## 4. What stands despite the issues

- The implementation itself is clean: `neuron_mask` is applied at the correct two
  sites in `bdh.py` (after each k-sparse ReLU), broadcasting is correct, the loss
  mix in `train.py` does what was registered, and the ladder script follows the
  existing arm scripts. This is solid work — the failures are protocol and
  reporting failures, not coding failures.
- The confusion matrix is honest and full. Good practice.
- Of the report's four diagnostic causes, two are invalidated by the diff itself
  (frozen attention was a config choice; "gradient too weak" is a misreading of
  the mix). The remaining causes (route collapse, byte-statistics routing) are
  plausible hypotheses worth testing in the corrected run.

## 5. Recommended next steps (in order)

1. **Verify the eval route grid** on the 3-phase model (does route k map to a real
   prefix block?). Fix if broken.
2. **Re-run the PoC protocol-compliant**: attention unfrozen per phase, as
   registered. Keep alpha=0.1 for comparability, then extend with alpha=0.5/1.0
   as a sensitivity check.
3. **Report monotonicity** of P-Det and P-Route vs alpha. If no improvement,
   route-aware training is genuinely unsupported in BDH and goes into the
   negative-results register. If it improves, we scale.
4. Only then revisit the full-ladder decision.

## 6. Report protocol requirements

Third arithmetic/setup inconsistency in the report series, plus a report that
misdescribes its own diff. Adopt the rule proposed earlier, now with teeth: every
report must include (a) the exact eval command, (b) the route grid values,
(c) per-domain counts that sum to the stated total, and (d) a one-line mapping
from each registered design item to where it was implemented (file/flag). The
reviewer should not have to reverse-engineer any of these.

## Bottom line

Do not treat this run as a falsification of route-aware training. It is a
falsification of "route-aware with frozen attention on a possibly misaligned
route grid". Fix the protocol, re-run cheaply, then decide.
