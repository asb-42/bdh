# Unseen-language inputs used by F-2 (report §9.3)

Slices of text in scripts/territories BDH was never trained on, used to compare two addressers: the byte
n-gram predictor (`scripts/pi50/r3c_ood_addendum.py`) and the likelihood router (A0-Quinn's routdiag runs).
**The bytes are deliberately not committed**: they are licensed corpus text, ~10 MB of it would make the docs
tree heavier than the code, and the analyses are reproducible from the source paths below.

Working copies lived on gx10 at `~/bdh-review/data/ood/`; census instrument is
`scripts/pi50/ood_script_census.py`.

| file | bytes | md5 (12) | source |
|---|---|---|---|
| `xscript_zh_head.txt` | 2,000,000 | `f8d21b7a7f75` | `.200:/media/data/coding/bdh/data/europarl/xscript_zh.txt` (first 2 MB) |
| `xscript_ja_head.txt` | 2,000,000 | `15cdd7b41f67` | same dir, `xscript_ja.txt` |
| `xscript_hi_head.txt` | 2,000,000 | `98bf01afa4e2` | same dir, `xscript_hi.txt` |
| `lv_head.txt` | 2,000,000 | `638ab8e68c47` | same dir, `europarl-v7.lv-en.lv.txt` |
| `ga_head.txt` | 2,000,000 | `da7ca90ba2ae` | same dir, `DGT.en-ga.ga.txt` |
| `xscript_iu.txt` | 163,778 | `5728d32d492f` | same dir, `xscript_iu.txt` — **contaminated, see below** |
| `ra2b_routdiag_xscript.txt` | 1,418 | `befdd5a99702` | A0-Quinn's stage-A output, `.200:/media/data/coding/bdh/out/logs/` |
| `ra2b_routdiag_p20_lv21.txt` | 4,986 | `977b5b2dbc74` | his A3 closed-set probe, same dir |
| `ra2b_routdiag_ga_dgt.txt` | 786 | `d9678d764d19` | his ga run, same dir |

Access note: `xscript_ja.txt` and `xscript_hi.txt` were mode `0660` owner `a0-quinn` and unreadable by
`pi50`; resolved by Quinn's `chmod 644` requested in bus #195 and applied before this table was written.

## The iu file is not Inuktitut-only, and the numbers matter

Whole-file census (`ood_script_census.py`, line classification over 707 lines / 163,778 B):

| class | lines | bytes | share |
|---|---|---|---|
| Inuktitut syllabics (U+1400–U+167F) | 265 | 125,879 | 77.2 % |
| pure ASCII prose | 333 | 29,701 | 18.2 % |
| Turkish-diacritic, no syllabics | 109 | 7,465 | 4.6 % |

Character level: 43.6 % syllabic, 31.8 % latin, 0.55 % Turkish diacritics. Sample ASCII line: *"There are
three cursed men who were in the most beautiful days of mine…"* — English prose sitting in an `.iu` column,
which is why the contamination was reported (bus #188) rather than quietly filtered.

Consequences for any reuse of these slices:

1. Route statistics computed on the raw file mix three different scripts, so "iu routes to bg/el" and
   "iu routes to en" can both be true of subsets. Use `iu_syllabic_only` for cross-script claims.
2. The independent confirmation in report §9.3 depends on keeping the subsets separate: the fitted byte
   predictor sends `iu_ascii_only` to `en(base)` 40/40 and `iu_syllabic_only` to pl/ro — a label it was
   never trained on for those rows.
3. A0-Quinn's cleaned stage-A re-run (#194) is the reference side of that comparison; until it lands,
   iu-syllabic is listed as OPEN, not as a divergence.
