# A7 — multi-era expansion control: the mechanism generalizes, the number does not

**pi-50 · 2026-09-12 ·** operator GO on #235 thread; design pre-registered on the bus as **#240** before results
landed; verification criteria pre-registered by A0-Quinn in `docs/reviews/2026-09-12_quinn_a4-a7-verification-protocol.md`.
Instrument: `scripts/pi50/a7_multiera_expansion.py` (new file; phase-1 `r2_expansion_control.py` untouched).
Outputs: `docs/reports/data/a7_multiera/{results.csv,summary.json,run.log}`. Eval-only: no optimizer, no backward,
no persisted checkpoint.

## Question, in one sentence

The abstract says a single randomly initialized block already costs 4.1× and reproduces ~83 % of the damage —
measured on the **English-era** checkpoint — and Kimi (#229, item Q4/A7) correctly noted that this is generalized
from one era. So: measure it per era.

## Design actually run (as frozen in #240)

Eras = ladder phases **1, 5, 8, 11, 14, 17, 19 → en, de, pt, bg, el, ro, sl**; phase 20 excluded because it is
already at final width. Each era expanded directly to **N = 47104** in whole 2048-neuron blocks. Domains per era =
its own language plus `en` as a fixed early-acquired reference. Protocol constants identical to phase 1 (`BLK` 2048,
`NH` 8, `D` 512, iters 25, batch 8, crop seed 4242, init seed 777, bf16 autocast). Arms: `base`, `A` iid-Gaussian
expansion, `C` exact-zero expansion, `masked` (arm A evaluated with neurons masked back to the era width); `real`
joint-serving values read from the landed matrix rather than re-run. Statistics: `f_log` and `f_linear`, both always.

Runtime 7 eras ≈ 68 min on the GB10 (≈10 min/era at two domains).

## Result

Own-language pairs, ascending by era (the generalization test):

| era | base | random expansion | real ladder | damage | **f_log** | **f_lin** |
|---|---|---|---|---|---|---|
| p1 en | 2.32 | 20.08 | 31.07 | 13.4× | **0.832** | 0.618 |
| p5 de | 2.74 | 21.06 | 34.71 | 12.7× | 0.803 | 0.573 |
| p8 pt | 2.72 | 20.78 | 26.92 | 9.9× | 0.887 | 0.746 |
| p11 bg | 6.14 | 149.30 | 230.22 | 37.5× | 0.880 | 0.639 |
| p14 el | 6.52 | **186.76** | 63.66 | 9.8× | **1.472** | **3.154** |
| p17 ro | 3.28 | 19.18 | 22.78 | 7.0× | 0.911 | 0.815 |
| p19 sl | 3.39 | 9.93 | 20.39 | 6.0× | 0.599 | 0.385 |

median `f_log` 0.880, range **0.599 – 1.472**. Four further pairs (the `en` reference domain inside foreign eras)
were suppressed by pre-registered gate **G4** because their remaining damage was under 2×, making the denominator
noise: they are listed in `results.csv` with `gate=G4-not-defined` rather than dropped.

**Three findings.**

1. **The instrument reproduces phase 1 independently.** English era: `f_log` **0.832** against the published
   **0.833**, `f_lin` **0.618** against **0.620**, arm-A ppl 20.08 vs 20.16 (−0.4 %, inside the 2–4 % replication
   floor). Nothing downstream is interpretable without that, and it is the reason the rest of the table can be
   compared with the number already in the paper.
2. **The mechanism generalizes; the headline number does not.** In six of seven eras random expansion alone
   reproduces the majority of the log-scale damage (0.60–0.91), so "degradation requires learned competition" is
   refuted broadly, which is the point the control was built to make. But the *fraction* is era-dependent, and the
   abstract's single figure is not supported as a general constant.
3. **Greek overshoots, and that is the interesting case.** At the el era, expanding randomly to final width yields
   **186.8** where the trained ladder yields **63.7** — `f_log` 1.472, `f_lin` 3.154. Random blocks do *more* damage
   than the actual growth did. Read carefully: for Greek, learned growth was **protective relative to noise** — the
   appended block organized itself so as to lose less than an untrained block would have. This is the opposite
   direction from the paper's rhetorical use of the control (arithmetic beats learning) — and see the addendum
   below, because my first reading of it ("growth was protective") was the interesting one and the mundane one wins. It also warns against quoting "≈83 %" as if the residual
   17 % were a uniform gap awaiting decomposition.

So the pre-registered falsifier **triggered**: any era outside ~[0.6, 0.95] scopes the claim, and el (1.472) plus
sl (0.599, at the edge) do exactly that.

## Gates

| gate | outcome |
|---|---|
| G1 base reproduces logged acquisition exit (±5 %) | **7/7 pass**: worst el +2.5 %, others ≤ +1.3 % |
| G2 inert (zero-block) arm unchanged | **exact**: max relative deviation from base **0.0** across all 7 eras × 2 domains at iters 25. The pilot's ~4 ppm at iters 3 is what forced recalibrating this gate from 1e-6 to 5e-3 (bf16 reduction order), and the tolerance never had to be used |
| G3 masked arm A restores access | **pass**: max deviation from base **1e-4** — masking-to-era-width recovers acquisition quality in every era, extending exp3's finding from 19 transitions to synthetic expansions |
| G4 quote fraction only where real damage > 2× | **4 pairs suppressed**, recorded in the CSV with the flag |

## Response to Quinn's protocol

| check | status |
|---|---|
| A7-1 pre-registration precedes results | **substantially met, letter disclosed**: design frozen in the script docstring and posted as **#240** before any result landed, but one calibration pilot (bg, iters 3) ran first and I had seen two numbers before #240. Its only effect was recalibrating G2's numerical tolerance; no era, domain, width, statistic or threshold was chosen using an observed result. Amend or reject as you see fit — I am not relabelling the sequence. |
| A7-2 instrument freshness | **met**: new file; `git show` confirms `r2_expansion_control.py` unmodified since its own landing |
| A7-3 eval-only | **met with declared wording**: no optimizer, no parameter update, nothing persisted under `out/`. The instrument *does* call `torch.save` into `NamedTemporaryFile`, unlinked immediately after load — stated up front in #240 so a grep reads it as known rather than discovered |
| A7-4 checkpoint identity | **met**: sha256 prefixes below, with sizes, for every era loaded |
| A7-5 arms comparable, both units reported | **partially met, declared**: arm structure matches (random / inert / masked / real-from-artifact) and both units are reported separately, but I measure **one** width target (47104) instead of phase 1's six-point sweep, because the statistic being generalized is the final-width fraction. Reproducing the dose–response curve per era is a different measurement and A7-8 forbids folding it into the 83 %/62 % headline. Offer stands to run it as its own pre-registered item |
| A7-6 numbers recomputable | **met**: internal re-derivation from `results.csv` columns reproduces every `f_log`/`f_linear` to <5e-4, zero mismatches; inputs are plain CSV so your own recompute needs no model |
| A7-7 general claim earned? | **NOT earned as a constant.** Supported: mechanism (majority of damage reproducible without learning) in 6/7 eras. Required by your rule: state the measured range by domain group, not the en value alone. Suggested text below |
| A7-8 no new headline | respected: the el overshoot is reported as a new measurement with its own record, not merged into 83 %/62 % |

Checkpoint identities (era files loaded by this run):

```
en 1211147355 68aa631faef8dfb7ce86d51677d672ec   de 2417040815 0990bf367f264fea9f29dae6a01714b5
pt 3323035055 5e06e757fd6c32a2120561353c7ab401   bg 4229029295 81b60f6a1d87ca75581a89598364e0ab
el 5135023579 fa07ddf942abbd34334b6c2261821d15   ro 6041017843 6f4fb88f6e7cc485a25ef6e66f73fb7d
sl 6645014015 a98e927ebd088c8e6285fb6342092777   lt 6947012095 cada61d704ea15795056c18bbb99afd4 (matrix source)
```

## What the manuscript may now say (TeX-ready)

Replace the general form with the measured form:

> Across seven eras spanning Latin, Cyrillic, Greek and Uralic territories, expanding a checkpoint to final width
> with randomly initialized blocks reproduces the majority of the log-scale degradation in six of them
> (`f_log` 0.60–0.91; English era 0.832, reproducing the 0.833 reported above), so learned competition is not
> required for most of the damage. The fraction is era-dependent, and for Greek random expansion exceeds the
> trained ladder's damage (`f_log` 1.47), i.e. growth there was protective relative to noise. Source:
> `scripts/pi50/a7_multiera_expansion.py`, data in `docs/reports/data/a7_multiera/`.

Do **not** keep a bare "≈83 %" in a general sentence; the honest general statement is "most of the damage, in most
eras, with a measured spread".

## Limits

- Arm A is iid Gaussian at the empirical weight scale, not the pipeline's true fresh-BDH initialization — the same
  limitation phase 1 declared, carried forward deliberately so the comparison stays valid.
- `real` comes from joint serving at the final checkpoint via the matrix; comparing era→final conflates expansion
  damage with everything else that happened between those phases. That is exactly the quantity the abstract
  generalizes, but it is not a clean isolation of "one block added", and the el case shows the difference matters.
- Two domains per era, 14 numbers. Per-era intervals are not quoted: with one observation per era-domain pair there is
  nothing to collapse, and pretending otherwise is the pseudo-replication error I already corrected once (§9.7).
- Era checkpoints are the `_last.pt` files, i.e. end-of-phase states, not best-validation snapshots.

---

## Addendum 2026-09-13 — the el "overshoot" decomposes into a small denominator, so this closes rather than opens

Operator ruled the el case goes to phase 2 only if it matters, otherwise postponed indefinitely ("otherwise this
becomes a neverending story"). Before agreeing, one bounded check from data already in `results.csv` — separate the
two factors that `f_log` confounds, since it is a ratio whose numerator and denominator are both era-relative:

| era | random-block damage (rand/base) | trained-ladder damage (real/base) | f_log |
|---|---|---|---|
| sl | 2.9× | 6.0× | 0.599 |
| de | 7.7× | 12.7× | 0.803 |
| en | 8.6× | 13.4× | 0.832 |
| bg | 24.3× | 37.5× | 0.880 |
| pt | 7.6× | 9.9× | 0.887 |
| ro | 5.8× | 6.9× | 0.911 |
| **el** | **28.7×** | **9.8×** | **1.472** |

Greek's random-block damage (28.7×) is unremarkable — statistically it is just bg with a different denominator. What
is unusual is that the trained ladder lost comparatively *little* at el (9.8×) even though el's acquisition exit is
high (6.52). So `f_log > 1` is driven by a small denominator, not by an extraordinary numerator, and the claim
"learned growth was protective relative to noise" is an artifact of comparing ratios across eras rather than evidence
of a mechanism. I wrote that sentence because it was the interesting reading; the decomposition says otherwise, and
correcting it here costs nothing.

**Decision: postponed indefinitely, no phase-2 experiment.** It is not load-bearing for any claim in the manuscript,
and chasing it means per-domain rabbit-holes exactly as the operator predicted.

One genuinely open observation falls out of the table and belongs in the phase-2 backlog as a *line*, not an
experiment: **why does joint serving degrade bg by 37.5× but el by only 9.8×,** when both are high-byte Cyrillic/Greek
territories with similar acquisition exits (6.09 / 6.36)? That asymmetry predates this instrument, is already implicit
in the landed matrix, and has nothing to do with expansion controls. Recording it is not the same as working on it.

Standing rule proposed for the review cycle, to keep it finite: after rev 4.5 lands, close the external-review loop;
any further finding must either (a) change a sentence in the manuscript, or (b) go to the backlog with no measurement.
If it does neither, it does not get a run.
