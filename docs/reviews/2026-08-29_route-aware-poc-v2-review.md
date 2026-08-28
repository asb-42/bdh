# Review: Route-Aware PoC v2 (protocol-compliant redo)

**Date:** 2026-08-29
**Reviewer:** Quinn (review seat, Saga AI Labs)
**Reviewed:** `docs/reports/2026-08-28_route-aware-poc-v2.md` (commits `05bcc07`, `b3029ed`)
**Pre-registration:** `docs/plans/2026-08-28_route-aware-poc.md` (`1bc796c`)
**Prior review:** `2026-08-28_route-aware-poc-review.md` v2 (erratum `2b2381e`)

## Verdict

**H-PoC PASS confirmed.** The redo addressed every review point, the reported
numbers are internally consistent (arithmetic spot-checks below), and the code
matches the report this time. Two protocol notes and one series-congruence note
below must feed the ladder protocol; none invalidates the PoC result.

## What was verified

1. **Code matches report.** `route_alpha` is now the prefix fraction
   (`loss = α·prefix + (1−α)·full`), printed in the training log; the mapping
   note in `b3029ed` (new α = 1 − old α) is correct and the registered
   "10% full mix" is new α=0.9. `freeze_attn` flag added with safe default
   (True); PoC branches run `--no-freeze-attn` as registered.
2. **Arithmetic.** Spot-checked nats gaps from the PPL pairs:
   de α=0.5: ln(2.50/2.19)=0.132 (reported 0.131) ✓; es α=0.5: ln(2.41/2.18)=0.100
   (0.099) ✓; es α=1.0: ln(5.64/2.18)=0.950 (0.949) ✓. All within rounding.
   This is the first report of the series with no arithmetic error.
3. **Script ↔ report consistency.** `ladder_poc_ra2.sh` runs P1, two
   specialists, and 3 α-branches with warmup 1000 / cosine 10k, fresh optimizer,
   matching the report and plan §3.1.
4. **The v1 grid diagnosis was correct.** Report confirms: on the old grid,
   routes >5632 clamp to the full model → 17 of 19 routes identical. This
   retroactively explains most of the v1 confusion matrix.

## Protocol note 1: P-Det grid deviates from the registration (documented)

The registration (§3.3) says: "run the 20-way routing diagnosis ... same routes
as in 2026-08-27_routing-diagnosis.md". The primary P-Det here is a **3-way
boundary grid** (1536/3584/5632), with a 22-route fine grid as diagnostic. This
is a metric redefinition made *after* registration.

Assessment: **acceptable here, but it must not silently carry into the ladder.**
Reasons for acceptance: (a) the registered grid degenerates on a 5632-neuron
stack (demonstrated, not asserted); (b) the fine grid independently corroborates
block-level detection at 100% for all trained languages (the de 5/35 and es 10/30
splits stay inside the correct block); (c) the deviation is documented in the
report itself, not silent. However, a 3-way task is easier than a 19-way task,
and the pre-registered falsifier was written against the 20-way task. The honest
statement is: **the registered P-Det operationalization was unexecutablele at
this scale and was replaced by a documented, corroborated substitute.**

**Ladder requirement:** before the 19-phase run, the ladder protocol must define
P-Det at scale. With 19 phase blocks the natural instrument is a 19-way
boundary grid (one route per block edge), which at the large stack size no longer
degenerates — and the original 20-way grid can additionally be reported for
continuity with the 2026-08-27 series.

## Protocol note 2: LR schedule breaks congruence with the ladder series

v1 and all earlier arms used warmup-30 defaults; this run uses warmup 1000 /
cosine 10k (per plan §3.1 — correct per registration, and the P1 base retrained
accordingly: 2.29 vs 2.46–2.48 in the old series). Within the PoC, INV-1 holds
(training and eval congruent). But cross-arm comparisons against Arm G/R and
CL-H1 data in the manuscript now cross an LR-regime change. The ladder with
route-aware training will run under the new schedule; the manuscript must either
re-run one control arm (e.g. Arm R style, fixed capacity) under the new schedule
or state the schedule difference explicitly in the methods section.

## Minor notes

- `config.py` default `route_alpha=0.1` silently changed meaning (now 10% prefix,
  was 90% prefix). Consider bumping the default to 0.9 to preserve the registered
  behavior as the default.
- P-Acq table says P3 ES α=1.0 = 5.52 (best val) while P-Route table says 5.64
  (routed serving on boundary grid). Different measurements, both plausible — one
  clarifying line would prevent future confusion.
- Raw router outputs live on the GPU box (`out/logs/...`). For the manuscript,
  commit the confusion matrices (small text files) alongside.

## Science check: the result itself

The Section 7 signature is the strongest part of the report and manuscript-grade:
trained languages route to *exactly* their own block (en→1536, never wider),
while untrained languages fall to the widest prefix with catastrophic PPL. If
grown neurons were generic capacity, older languages would drift to the widest
prefix; they do not. This is the construction thesis ("prefix growth constructs
the structure") made visible in routing behavior.

Monotonicity is clean: P-Route and P-Acq degrade monotonically with α
(0.131→0.200→0.215 de; 0.099→0.201→0.949 es; 2.25→2.49→5.52 ppl). The
interpretation — 10% full mix is load-bearing for block health, not for
detectability — is supported. Note the retention cost of route-aware growth is
real but small at α=0.9 (0.2 nats) against the 0.3-nat threshold.

## Ladder decision

The pre-registered rule says proceed. Open choice for the operator:

- **α=0.9** (registered setting): registration-faithful; 0.20/0.20 nats, ~15%
  margin under the 0.3 threshold.
- **α=0.5**: best metrics (0.13/0.10 nats); but deviates from the registered
  setting, and the PoC gives no evidence the gap widens or narrows over 19 phases.

My recommendation: primary ladder at **α=0.9** (fidelity to registration; both
settings pass comfortably), add α=0.5 only as a secondary arm if compute allows.
Do not switch the primary to α=0.5 post hoc without re-registering.

## Bottom line

Clean redo, honest reporting, verified numbers. H-PoC passes; route-aware
training with unfrozen attention solves the current-phase problem at PoC scale.
Define ladder P-Det at scale, control the LR-regime confound for the manuscript,
then run the ladder.
