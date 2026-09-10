# RA2b acquisition report — PPL per language vs training tokens and checkpoint size

- Sources: gx10 `out/logs/ladder_ra2b_analysis.txt` (phase exits), `out/logs/ladRA2b_<lang>.log` (in-training validation), `out/bdh_europarl_ladRA2b-*_last.pt` cfg (authoritative batch/block/step).
- Tokens/step = batch x block(512); cumulative along the chain (weights resume, optimizer restarts per F-V8 ruling #119). Params from tensor shapes; bytes = file size (~12 B/param = fp32 weights + Adam m,v).
- Exit ppl = own-language random-crop cold eval at phase end. Curves = the training loop's own val ppl (different crop protocol; see caveat section).

## 1. Per-phase exits

| ph | lang | mult | batch | Mtok/phase | cum Mtok | params M | ckpt MB | exit ppl |
|---|---|---|---|---|---|---|---|---|
| 1 | en | 128->128 | 4 | 20.5 | 20 | 100.9 | 1211 | 2.29 |
| 2 | es | 128->160 | 4 | 20.5 | 41 | 126.1 | 1511 | 2.42 |
| 3 | pl | 160->192 | 4 | 20.5 | 61 | 151.3 | 1813 | 3.04 |
| 4 | fr | 192->224 | 4 | 20.5 | 82 | 176.4 | 2115 | 2.43 |
| 5 | de | 224->256 | 1 | 5.1 | 87 | 201.6 | 2417 | 2.73 |
| 6 | cs | 256->288 | 1 | 5.1 | 92 | 226.8 | 2719 | 3.48 |
| 7 | da | 288->320 | 1 | 5.1 | 97 | 251.9 | 3021 | 2.85 |
| 8 | pt | 320->352 | 1 | 5.1 | 102 | 277.1 | 3323 | 2.66 |
| 9 | fi | 352->384 | 1 | 5.1 | 108 | 302.3 | 3625 | 2.98 |
| 10 | hu | 384->416 | 1 | 5.1 | 113 | 327.4 | 3927 | 2.88 |
| 11 | bg | 416->448 | 1 | 5.1 | 118 | 352.6 | 4229 | 6.09 |
| 12 | it | 448->480 | 1 | 5.1 | 123 | 377.8 | 4531 | 2.89 |
| 13 | et | 480->512 | 1 | 5.1 | 128 | 402.9 | 4833 | 3.26 |
| 14 | el | 512->544 | 1 | 5.1 | 133 | 428.1 | 5135 | 6.36 |
| 15 | sk | 544->576 | 1 | 5.1 | 138 | 453.3 | 5437 | 3.58 |
| 16 | sv | 576->608 | 1 | 5.1 | 143 | 478.5 | 5739 | 3.08 |
| 17 | ro | 608->640 | 1 | 5.1 | 148 | 503.6 | 6041 | 3.26 |
| 18 | nl | 640->672 | 1 | 5.1 | 154 | 528.8 | 6343 | 3.07 |
| 19 | sl | 672->704 | 1 | 5.1 | 159 | 554.0 | 6645 | 3.50 |
| 20 | lt | 704->736 | 1 | 5.1 | 164 | 579.1 | 6947 | 3.83 |

## 2. Acquisition curves within each phase (training-loop val ppl)

| lang (phase) | 5% | 10% | 20% | 30% | 50% | 70% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|
| **en** (1) | 3.86 | 3.28 | 2.82 | 2.66 | 2.48 | 2.37 | 2.28 | 2.29 |
| **es** (2) | 5.10 | 3.55 | 2.85 | 2.75 | 2.51 | 2.43 | 2.46 | 2.44 |
| **pl** (3) | 9.71 | 5.45 | 4.15 | 3.65 | 3.36 | 3.20 | 3.14 | 3.18 |
| **fr** (4) | 4.38 | 3.36 | 2.76 | 2.64 | 2.48 | 2.39 | 2.34 | 2.38 |
| **de** (5) | 6.92 | 4.54 | 3.44 | 3.08 | 2.88 | 2.79 | 2.78 | 2.89 |
| **cs** (6) | 10.94 | 6.67 | 4.74 | 4.25 | 4.00 | 3.85 | 3.62 | 3.93 |
| **da** (7) | 6.43 | 4.51 | 3.74 | 3.33 | 2.94 | 2.96 | 2.91 | 2.76 |
| **pt** (8) | 5.18 | 4.06 | 3.33 | 3.22 | 2.99 | 2.78 | 2.86 | 2.62 |
| **fi** (9) | 8.28 | 5.14 | 3.86 | 3.56 | 3.13 | 3.10 | 3.00 | 3.02 |
| **hu** (10) | 10.36 | 5.95 | 3.89 | 3.61 | 3.24 | 3.11 | 2.98 | 2.96 |
| **bg** (11) | 18.45 | 12.73 | 8.71 | 7.87 | 6.95 | 6.22 | 6.19 | 6.14 |
| **it** (12) | 6.18 | 4.73 | 3.66 | 3.54 | 3.14 | 3.07 | 3.06 | 2.90 |
| **et** (13) | 9.05 | 5.83 | 4.38 | 4.00 | 3.71 | 3.38 | 3.38 | 3.36 |
| **el** (14) | 18.68 | 12.08 | 9.10 | 8.38 | 7.19 | 6.75 | 6.53 | 6.61 |
| **sk** (15) | 11.13 | 6.61 | 4.83 | 4.33 | 3.82 | 3.67 | 3.72 | 3.65 |
| **sv** (16) | 6.39 | 4.99 | 3.81 | 3.54 | 3.47 | 3.23 | 3.17 | 3.01 |
| **ro** (17) | 8.71 | 6.12 | 4.50 | 4.11 | 3.76 | 3.39 | 3.42 | 3.28 |
| **nl** (18) | 6.74 | 4.83 | 3.79 | 3.67 | 3.14 | 3.04 | 3.09 | 3.03 |
| **sl** (19) | 9.91 | 6.30 | 4.48 | 4.23 | 3.81 | 3.69 | 3.49 | 3.56 |
| **lt** (20) | 12.37 | 7.76 | 5.21 | 4.99 | 4.22 | 4.00 | 4.05 | 3.83 |

(parsed 20/20 phases)

## 3. Caveats

- Column headers are fractions of that phase's 10k steps, not absolute tokens: batch differs across phases (4 early, 1 later), so equal fractions are unequal token counts.
- In-phase val ppl comes from the training loop's fixed validation stream; exit ppl comes from `lang_eval.py` random-crop cold eval. They are different protocols and must not be mixed in one plot.
- No seed replicates exist for RA2b cells; apply the 2-4% floor from S1d (#122) to every single-cell number above.
- Fixed regime confirmed by P5 (old capacity bit-identical across growth), so none of these deltas carry the decay artifact.
