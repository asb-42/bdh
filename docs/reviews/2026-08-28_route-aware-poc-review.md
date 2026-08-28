# Review: Route-Aware PoC Results

**Date:** 2026-08-28
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-28_route-aware-poc.md` (commit `93e6ce4`)
**Pre-registration:** `docs/plans/2026-08-28_route-aware-poc.md` (commit `1bc796c`)

## Verdict

**FAIL in the pre-registered sense — but for a reason that invalidates the test of
H-PoC, not a clean falsification of the mechanism.** The run did not follow the
pre-registered protocol at its decisive point: attention was frozen throughout,
whereas the plan requires "Attention: unfrozen during each phase's training".
Before treating this as evidence against route-aware training, we need a
protocol-compliant run with calibrated alpha. Until then, no ladder decision.

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

## 2. Alpha = 0.1 is uncalibrated and nearly empty as a signal

The report itself lists "prefix-masked gradient too weak" as cause #3: with
route_alpha=0.1, only 10% of the loss signal comes from the prefix-masked forward;
90% comes from the full-forward mix. The pre-registration allowed mixing but stated
a fixed fraction "reported in the log" and suggested 10%; it did not say the
mechanism should be predominantly evaluated under a 10%-strength instantiation.

For a mechanism test we want at least a small alpha sweep (0.1 / 0.5 / 1.0) or,
better, a monotonicity check: does routing quality improve as alpha increases?
That is cheap (3-phase stacks are minutes-hour runs) and would tell us whether
the mechanism has any signal at all.

## 3. Route collapse and PPL explosions point to an eval/serving problem, not only training

- Only 3 of 19 routes are used; the trained languages (en, de, es) are routed to
  bg/cs/da. That is the expected signature of a router mismatch, not merely of weak
  training pressure.
- Routed PPL values of 415,779 and 127,635 are not cross-entropies of a serving
  model; they are divergences (log-scale blow-ups). Joint PPL = 109.56 here vs
  54.96–58.39 in the earlier ladder diagnosis. Something differs in the eval setup
  on this small stack — likely the fixed 20-route grid calibrated for the large
  grown stack (2252–45040) being applied to a model with only ~5632 neurons of
  prefix range, producing routes that do not match any phase block.

**Action:** verify the route grid against this model's actual neuron ranges before
re-running. If the routes do not align with the phase-blocks, both P-Det and
P-Route numbers are not interpretable.

## 4. What stands despite the issues

- The implementation added a real `neuron_mask` and `route_aware` option to
  `bdh.py` / `train.py` — useful infrastructure for the corrected run.
- The confusion matrix is honest and full. Good practice.
- The report's four diagnostic causes are reasonable hypotheses; the problem is
  that two of them (frozen attention, alpha=0.1) are properties of the run
  configuration, not of the mechanism.

## 5. Recommended next steps (in order)

1. **Verify the eval route grid** on the 3-phase model (does route k map to a real
   prefix block?). Fix if broken.
2. **Re-run the PoC protocol-compliant**: attention unfrozen per phase, as
   registered. Keep alpha=0.1 for comparability, then extend with alpha=0.5/1.0.
3. **Report monotonicity** of P-Det and P-Route vs alpha. If no monotonic
   improvement, then route-aware training is genuinely unsupported in BDH and we
   write that into the negative-results register. If it improves, we scale.
4. Only then revisit the full-ladder decision.

## 6. Update to the leaderboard/protocol practice

Third arithmetic/setup inconsistency in the report series. Let's adopt the rule
proposed earlier, now with teeth: every report must include (a) the exact eval
command, (b) the route grid values, (c) per-domain counts that sum to the stated
total. The reviewer should not have to reverse-engineer these.

## Bottom line

Do not treat this run as a falsification of route-aware training. It is a
falsification of "route-aware at alpha=0.1 with frozen attention on a possibly
misaligned route grid". Fix the protocol, re-run cheaply, then decide.
