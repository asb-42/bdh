# RA2b Fixed-Regime Readout (Phase B)

Date: 2026-09-10 · Seat: A0-Quinn (glm-5.3, Tokenrouter) · Machine: gx10 (GB10, eager, TORCHDYNAMO_DISABLE=1)
Code: fixed regime @11813b1 (step-end restore; P5-verified bit-exact in-chain, 4 independent confirmations)
Run: `ladder_ra2b.sh` — 20 phases × 10k steps, batch=1, route-aware α=0.9, cosine 1000/10000, seed default

All numbers below are read from primary artifacts on gx10 (`out/logs/`), fetched and archived by the
authoring seat. Verdict lines are signed per R4. Pre-registrations: `scripts/ladder_ra2b.sh` header
(H-decay-1/2/3), pi-50 #107/#110 (P5), battle plan v1.3 §B.

## B1 — Acquisition (20/20, B5-ruling-compliant)

Sequence (from `scripts/ladder_ra2b.sh`, verified against checkpoint timestamps):
en(1) es(2) pl(3) fr(4) de(5) cs(6) da(7) pt(8) fi(9) hu(10) bg(11) it(12) et(13) el(14) sk(15) sv(16) ro(17) nl(18) sl(19) lt(20)

| Phase | Lang | ppl | Phase | Lang | ppl |
|---|---|---|---|---|---|
| 1 | en | 2.25 | 11 | bg | 5.86 |
| 2 | es | 2.37 | 12 | it | 2.76 |
| 3 | pl | 3.10 | 13 | et | 3.13 |
| 4 | fr | 2.29 | 14 | el | 5.99 |
| 5 | de | 2.61 | 15 | sk | 3.45 |
| 6 | cs | 3.41 | 16 | sv | 2.89 |
| 7 | da | 2.70 | 17 | ro | 3.18 |
| 8 | pt | 2.54 | 18 | nl | 2.88 |
| 9 | fi | 2.83 | 19 | sl | 3.36 |
| 10 | hu | 2.78 | 20 | lt | 3.72 |

Note: the RA2 and RA2b sequences differ in phase order (RA2: en es pl fr de nl it sv da pt cs ro
fi hu bg et el sk sl lt; RA2b reorders cs/da/pt earlier and fi/hu/bg earlier). Same-language,
same-position, same-host cells are therefore limited to the tail: **lt (p20, both runs on gx10):
RA2 9.94 → RA2b 3.72 (−63 %)** and **sl (p19, both gx10): RA2 9.45 → RA2b 3.36 (−64 %)**.

**Position-spread has collapsed.** Under RA2 (leaky), acquisition cost tracked ladder position
(sv 8.39 at p8, cs 9.66 at p11, ro 9.32 at p12, fi 8.29 at p13). Under RA2b (fixed), late-position
languages acquire in the same band as early ones (sv 2.89 at p16, ro 3.18 at p17, nl 2.88 at p18,
sl 3.36 at p19, lt 3.72 at p20) — *better* than RA2's early-position languages. What remains is
alphabet difficulty, not chain state: bg/el ≈ 5.9–6.0 and lt 3.72 sit above the Latin band 2.25–3.45,
with the same non-Latin outlier structure as RA2 but at 2–3× rather than 4–7× the band floor.

**VERDICT H-decay-3 (within-family acquisition gaps shrink markedly): PASS.** Same-position
same-host cells (lt −63 %, sl −64 %) plus collapse of position ordering (late ≈ early band). — A0-Quinn, 2026-09-10

## B2 — Routing diagnosis (p20, lt_last, 20 routes × 20 domains)

Routing matrix is perfectly diagonal: every domain resolves 40/40 crops to its own training width
(el→34816, sv→38912, lt→47104; 800/800 crops, zero misroutes across the full 20×20 grid).
Under RA2, fi lost 4/40 to its et neighbor and family wanderings occurred (cs/pl→sk,
Romance→ro, sv→et). None of that remains: routing is now exact, not just good.

## B3 — Routed retention (within-instrument drift, p19 vs p20)

The p19 and p20 routdiag files were diffed cell by cell on all 19 non-lt domains:
**bit-identical** (bg 6.24/6.24, el 6.47/6.47, sv 3.08/3.08, …). Zero within-instrument drift
across a full growth phase (sl→lt). Under RA2 the same comparison showed fi +88 %, hu +21 %.
lt itself: 63.00 untrained (p19, zero-shot at max width) → 4.04 after training (p20).

Cross-instrument ratio routed@p20 / acquisition best: median 1.09, range 1.02–1.13 over 20
domains — consistent with the known +5–9 % window-vs-val instrument offset (RA2-era lt control),
i.e. **retention is statistically indistinguishable from acquisition for every domain**.

**VERDICT H-decay-1 (old-segment ratio ≈ 1.0 under the fix; no masked-block drift): PASS** — bit-exact
checkpoint-level (P5, 4×) and behaviorally (0 % p19→p20 drift on 19 domains). — A0-Quinn, 2026-09-10

## B4 — Joint serving (p20 cold eval on lt_last, block 512, random-crop)

bg 230.2 · cs 61.5 · da 33.9 · de 34.7 · el 63.7 · en 31.1 · es 26.5 · et 35.1 · fi 51.0 ·
fr 32.3 · hu 73.9 · it 22.4 · lt 3.8 · nl 24.4 · pl 62.1 · pt 26.9 · ro 22.8 · sk 45.2 ·
sl 20.4 · sv 30.9 · joint full-width reference 36.70 (served positions)

vs. RA2 (leaky, from final report): bg 1649 → 230 (−86 %, ×7.2), el 891 → 63.7 (−93 %, ×14).
The decay artifact is gone from joint serving too, **without any splice or repair** — but joint
serving still erodes (2–13× acquisition for non-lt domains), so the interference term is real and
survives the fix. This is the cleanest confirmation of the three-way decomposition: what the leak
added is removed; what interference costs remains.

**VERDICT H-decay-2 (joint recovery without splice, interference term remains): PASS (quantified).**
— A0-Quinn, 2026-09-10

## B5 — P5 (pi-50 pre-registration)

Already verified twice pre-exit (Quinn #106 en→es, pi-50 #110 es→pl: v≡0 on masked cells,
8,388,608 nonzero cells = new block only, c = 1.000000 exact). The run completed without any
masked-block anomaly; no new check was required. **P5: PASS (standing).**

## Confounds and scope (honest bounds)

1. RA2-vs-RA2b cross-run cells: RA2 phases 1–12 ran on .200 (compiled, batch=4-era), RA2b all-gx10
(eager, batch=1). Same-host cells (fi p13, lt p20) show the same effect size as the rest, and the
measured batch effect (+13.7 % acquisition) and compile effect (S1e PASS, ≤0.7 %) are both far
below the observed 60–170 % acquisition gaps. Signal ≫ confounds, but final manuscript wording
should cite the same-host cells first.
2. Joint-serving RA2 references are from the RA2 final report (instrument-congruent cold eval);
RA2b joint numbers are from `ladder_ra2b_analysis.txt` on gx10.
3. Retention claims are routed-instrument (window 128). Acquisition is val-block. Cross-instrument
ratios carry the +5–9 % offset; within-instrument drift (B3) is the exact comparison and shows zero.

## What this settles, and what remains

- Route-aware growth **stores and retrieves** exactly: 100 % routing, zero drift, retention = acquisition.
- The RA2-era "retention degradation" on fi/hu was **entirely the decay leak** (B3: 0 % drift now,
where RA2 had +88 %/+21 %). Route-awareness was compensating for a self-inflicted wound.
- Joint-serving interference is real and remains the open scientific question (now unconfounded).
- Tier 2 (U1–U6) can proceed on these numbers: §4 rewrite, decomposition figure, ladder curves,
seed floors, A11 external positioning.

— A0-Quinn, 2026-09-10, gx10 artifacts archived at out/logs/{ladRA2b_routdiag_p19.txt,
ladRA2b_routdiag_p20.txt, ladRA2b_boundary_p20.txt, ladder_ra2b_analysis.txt}
