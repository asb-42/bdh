# A4 — the instrument offset does not exist, and neither does the 1.13

**pi-50 · 2026-09-12 ·** operator GO on bus #235 thread ("A4 + A7 OK"). Instrument: `scripts/pi50/a4_offset_band.py`
(reads landed artifacts only; CPU; writes only `docs/reports/data/a4_offset_band/`). No training, no model loads,
no new evaluation.

## Why this needed measuring

§5 of the manuscript excuses routed-serving ratios as measurement noise:

> line 380 — "ratios (routed/acquisition) span **1.02–1.13**, consistent with the known instrument offset."

and the offset it leans on is documented once, in our own fixed-regime readout (`2026-09-10_ra2b-fixed-regime-readout.md:59`),
as **+5–9 %**, sourced to "(RA2-era lt control)" — a single language, a single comparison. Kimi's review (#229) noted
that +13 % cannot be inside a +5–9 % band; I agreed and filed it as a quantified inconsistency (#231 §A4). Every one
of us was arguing about two numbers that nobody had actually derived across domains. Both are computable from
artifacts already in the tree, so the question never needed rhetoric.

## What was measured

**D1 — pure instrument offset.** Same weights, two instruments: the RA2b serving matrix's *diagonal* cell for each
domain (val stream, `EVAL_BATCH` pinned to 1) divided by that domain's logged acquisition exit (training-window
crop). Because the parameters are identical in both numbers, any difference is instrument, not science. n = 20
domains, one observation each.

Gate first: the diagonal must reproduce three independently quoted exits (cs 3.48, da 2.85, de 2.73) — it does, to
<0.011, so the crop construction and the checkpoint pairing are right before any ratio is interpreted.

    median 1.0000   mean 1.0002   min 0.9901 (pl)   max 1.0087 (en)     → offset ≈ 0, spread ±1 %

**There is no +5–9 % window-vs-val offset on RA2b.** The two instruments agree to within one percent on all twenty
domains, and half of them agree to four decimal places.

**D2 — what routed serving actually costs.** Own-prefix routed ppl at the final checkpoint
(`out/logs/ladRA2b_routdiag_p20.txt`) over each domain's own-phase acquisition exit. This is the quantity §5 quotes.

    median 1.0429   mean 1.0420   min 0.9918 (fr)   max 1.0799 (hu)
    domains above 1.08: 0/20      above 1.10: 0/20      above 1.13: 0/20
    ascending: fr .992 sv 1.000 nl 1.010 el 1.017 bg 1.025 en 1.031 it 1.031 da 1.035 es 1.037
               pl 1.043 cs 1.043 fi 1.044 et 1.052 lt 1.055 de 1.055 sl 1.060 sk 1.075 ro 1.077 pt 1.079 hu 1.080

## Three conclusions

**(1) The "known instrument offset" must come out of the paper.** It is not a wide band we under-reported; on this
ladder it is ~1 %, and its stated provenance is one RA2-era language. Worse, removing it *strengthens* our position:
with no offset to hide behind, a +4.3 % median / +8.0 % worst-case routed cost is a **measured, real quality
difference**, and we can say so plainly instead of asking the reader to wave it away. The honest sentence is also the
stronger scientific claim — selection preserves competence to within eight percent, and we know that number is
quality, not instrumentation.

**(2) The 1.13 upper end does not reproduce, under any pairing I could construct.** Tried: routed@p20 over own-phase
exit (max 1.080); routed@p20 over the best value any checkpoint achieved for that domain (max 1.080); matrix diagonal
over exit (max 1.009); and every earlier routdiag phase p2…p19 against own-phase exits, which yields medians up to
18× and maxima to 238,000× — meaningless, because those domains were simply untrained at that phase, and it is the
comparison most likely to have leaked an inflated number into a caption if a phase table was ever mixed with a final
one. So `1.02--1.13` has no traceable source in the artifact set. Notation detail worth checking when the origin is
hunted: 1.13 is close to nothing in D2, so it is probably not a rounding of a real value but a stale figure from an
earlier revision.

**(3) §5's "within ≤ 8 %" survives — barely, and it should be re-stated with its distribution.** Max is 1.0799 (hu),
so the bound holds at the last decimal place. A claim that passes only because of rounding invites exactly the audit
that found this. State median and max with the domain names, and name the instrument.

## Recommended replacement text (TeX-ready, for whoever owns the prose)

Line 380 caption:

```latex
\caption{RA2b final checkpoint: routed serving (blue) vs joint serving (orange) vs acquisition
exit (black tick), per domain, log scale. Routed tracks acquisition: ratios span 0.99--1.08
(median 1.04, worst case hu +8.0\%), and the same-weights cross-instrument check shows the
window-vs-val offset is $\approx$1\%, not noise-hiding slack \texttt{docs/reports/data/a4\_offset\_band/}.}
```

§5, replacing the offset clause:

> Routed serving lands within 8 % of acquisition-exit quality across all twenty domains (median +4.3 %,
> worst case hu +8.0 %). This is a real residual, not an instrument artefact: evaluating the identical weights on
> the validation stream reproduces each acquisition exit to within ±1 % (median ratio 1.000, n = 20), so the gap
> cannot be attributed to the crop-versus-val distinction. Source: `scripts/pi50/a4_offset_band.py`, data in
> `docs/reports/data/a4_offset_band/`.

## Self-correction, mine this time

My #231 write-up accepted `1.02–1.13` as given and argued only about whether +13 % fits inside +5–9 %. That is the
same error I flagged in the external reviews one layer up: adjudicating an internal inconsistency from the numbers as
printed, without re-deriving either side. Recomputing took minutes on CPU and changed the conclusion — the tension
Kimi spotted is real, but it points the other way: the *excuse* is fabricated-ish and the *bound* is fine. Recorded
as a standing rule: **when two quoted quantities conflict, derive both before saying which one is wrong.**

## Limits of this note

- D1 compares the matrix pipeline (`scripts/lang_eval.py`, val stream, batch pinned to 1) against logged training-window
  exits. If the ladder's exit values came from a different crop family than I assume, D1 measures something else — the
  gate above (three independently quoted exits reproduced on the diagonal) is what makes that unlikely rather than untested.
- D2 inherits whatever instrument the routdiag used; I did not re-run routing. Since D1 shows the instrument families
  agree to ~1 %, the practical effect on D2 is small, but the two are not literally the same measurement.
- Domain-level reporting only: 20 observations, median/min/max, no pooled interval over crops. Crops within a domain are
  not independent trials (`docs/reports/2026-09-11_pi-50_expansion-control-and-readout-operators.md` §9.7).
- Nothing here touches the FCS ladder or the RA2-era numbers, where the original "+5–9 %" control allegedly came from.
