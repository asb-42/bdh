# DRAFT — Addendum 6 to `docs/tasks/2026-08-30_ladder-ra2-validation-checklist.md`

**Author:** pi-50 (gx10-50ef, review seat) · **Date:** 2026-09-03 · **Status:** draft, not yet in repo
**Blocked on:** ff-only merge of `origin/main` (`cd89ed7`) into `/srv/coding/bdh`, then `POST /scopes` with
`resource_uri=file://srv/coding/bdh/docs/tasks/2026-08-30_ladder-ra2-validation-checklist.md` (per Quinn, seq 26).
Text below is written against the **upstream** version of the checklist (`git show origin/main:...`, 271 lines),
not the pre-merge working tree (161 lines), so it will not conflict with Addenda 1–5.

Failure classes introduced here are numbered **F-V1…F-V4** (verification-protocol defects), deliberately *not*
F-T: the F-T register in Addenda 1–5 tracks transfer/run forensics, and these items are about the checklist's own
executability. Every claim carries the command that reproduces it.

---

## Addendum 6 (2026-09-03): executability gaps in sections 0–7 — found while preparing to execute the checklist

### F-V1: the checklist never names which artifact set it validates (best vs last)

Sections 0, 2 and 3 refer to "phase checkpoints" generically. Each phase produced **two**:
`out/bdh_europarl_ladRA2-<lang>_best.pt` and `..._last.pt`. They are not interchangeable, and the difference is
already visible in the headline number of the completed ladder:

| quantity | value | source protocol |
|---|---|---|
| best val ppl (lt) | 9.75 | training loop, `_best` (`out/logs/ladRA2_lt.log` tail: `done. best val_loss 2.2774 (ppl 9.75) -> ..._lt_best.pt`) |
| test ppl (lt) | 9.43 | training-loop test split at best |
| cold-eval ppl (lt) | 9.94 | `out/logs/ladder_ra2_analysis.txt`: `ckpt=..._lt_last.pt \| random-crop cold eval`, `lt: nll 2.2970 \| ppl 9.94` |

Three numbers spanning 9.43–9.94 for one language, all legitimate, none of them the same measurement. The routing
diagnosis runs on `_last` (`ps -p 79394 -o args=` shows `out/bdh_europarl_ladRA2-lt_last.pt`).

**Ask:** add a line to §0 fixing the validated artifact set per section — e.g. §2 parameter oracle on `_last`
(the object the routing tables were computed from), §4 ratio recomputation on whichever checkpoint the manuscript
quotes, and require every reported cell to name `_best` or `_last`. Without this, two seats can execute §2 and §4
faithfully and disagree by design.

### F-V2: the Batch column is unreproducible from committed code, and two records contradict each other (violates §1)

The ladder spans **three** optimization regimes inside one 20-phase series:

- `docs/reports/2026-08-30_ladRA3-progress-report.md` "Completed Phases" table: batch 4 (EN), 2 (ES PL FR DE),
  1 (NL IT SV DA PT CZ RO); reproduce with
  `awk -F'|' '/^\| *[0-9]+ *\|/{print $6}' docs/reports/2026-08-30_ladRA3-progress-report.md | sort | uniq -c`
  → `7 × 1, 4 × 2, 1 × 4`.
- Resume phases 13–20: `scripts/ladder_ra2_resume.sh:72` sets `BS=1` unconditionally.
- Committed `scripts/ladder_ra2.sh:68-72` sets `BS=4` when `PREV_MULT <= 192`, else `BS=1` — i.e. it predicts
  batch 4 for phases 2–4 and batch 1 from phase 5 onward. **Neither the report table (batch 2 for phases 2–5,
  batch 1 from phase 6) nor the analysis log matches that rule.**

Worse, the two surviving records disagree outright about the same phase:
- `out/logs/ladder_ra2_analysis.txt:5` → `== phase 2: es (grow from mult 128 -> 160, batch=4, alpha=0.9) ==`
- progress-report table row 2 → `| 2 | ES | 160 | 158M | 2 | 2.40 |`

Phase 2 ran at batch 4 or batch 2 depending on which artifact you read. §1 requires every run-property column to
cite its artifact; the Batch column currently cites a script that cannot produce either value. Most likely
explanation: the script was tuned for the 4090's memory *after* those phases ran, and the committed version is not
the version that produced the table — which is exactly the situation Addendum 3's eager-vs-compiled deviation
record had to handle for numerics.

**Asks:**
1. State the executed batch size per phase from the authoritative source (per-phase `out/logs/ladRA2_<lang>.log`
   argument echo, if present), and tag the report's Batch column `computed` or `artifact:<path>` per §1.
2. Record explicitly that cross-phase comparisons in P-Acq / P-Eros span non-constant tokens-per-step (4× between
   EN and the batch-1 majority). If the growth curve is meant to be read as width-driven rather than
   budget-driven, say so and give the reasoning; if it is a confound, name it as one. This is a *claim-scope*
   question, not a request to re-run anything.
3. Note that tying eval batch to train batch (`scripts/ladder_ra2_resume.sh:90` comment "Per-language eval (same
   batch as training)", and `--batch "$BS"` at lines 99/125) makes §5's "eval protocol identical across phases and
   arms" unsatisfiable as written: eval_router.py defaults to `--batch 4` (`scripts/eval_router.py:30`) while the
   live p20 diagnosis runs `--batch 1`. Addendum 4 argues batch-independence is ULP-class for confusion/ppl — fine,
   but then §5 should *cite that measurement* and pin the batch value, instead of pinning only the script version
   (`d10d389 or later`).

### F-V3: §0 tells you to parse an artifact that cannot answer the question

`§0: Parse out/logs/ladder_ra2_analysis.txt: every phase reached done; no phase silently skipped.`

That file (5,047 bytes, mtime Sep 3 04:48) contains phase headers for **10 of 20** phases: 1, 2, and 13–20. There
are no rows for phases 3–12, and phase 2's block carries no eval numbers at all — only
`preserved failed el attempt as out/logs/ladRA2_el.attempt1-failed.log`. It also ends mid-statement:
`--- milestone phase 20: evaluating on en,...,sl ---` with no matrix following, because the phase-20 routing
diagnosis (PID 79394) is still running. Consequences:

- Executing §0 as written yields "10 of 20 phases present", which reads as a silent-skip failure when it is
  actually an incomplete *record*: all 20 phases' checkpoints do exist
  (`ls out/ | grep -c 'ladRA2-.*_last.pt'` → 20, including `cz_best/_last` — the run-name is `cz`, the data is
  `cs`, per Addendum 1's mapping finding).
- The phases 3–12 completion data lives only as a hand-formatted table in `docs/reports/`, i.e. in the document
  class the checklist exists to gate. That inverts the intended order (artifact → report).
- `--- routing diagnosis phase N ---` lines are empty placeholders in this file; the content goes to
  `out/logs/ladRA2_routdiag_pN.txt`, which stays **0 bytes until the process exits** (observed:
  `ladRA2_routdiag_p20.txt` 0 bytes, mtime = process start). Any automated §0 check that tests "file exists" will
  pass on an empty file.

**Asks:** name the artifact set that *must* contain all 20 phases and make §0 fail loudly if it doesn't; add
"non-empty AND parseable" to the existence test; and state where the phases 3–12 record is, or regenerate it from
the 20 checkpoints so the machine-readable record is complete before the manuscript quotes it.

### F-V4: formulas asserted as oracles carry no source site (§2, §3, §6)

Three load-bearing formulas appear in the checklist without a definition site:
- `P(m) = 786432·m + 262144 (+64·m)` and "final ≈ 579,123,200 parameters" (§2);
- "routes are in per-head neuron units (64×mult)" (§3);
- "zero-init + RoPE frequency preservation, plus the mask in the ReLU regime" as the operative exactness mechanism (§6).

Each is checkable, and each belongs to a specific code location whose *implementation* — not prose — decides
whether it holds. F-T10 in Addendum 5 is the precedent that makes this more than tidiness: there, the claim
"frozen by construction" was structurally true but amplitude-preserving only up to an AdamW weight-decay leak that
lived in `pipeline/train.py:225` (`grad.mul_(mask)` leaves materialized zero grads). Nobody had cited the site, so
nobody had checked it, and the conclusion held only approximately. My own retraction chain (hard rule 5, upstream
`6754e83`) started the same way: a defect in a code comment quoted as a conclusion.

**Ask:** for each formula in §2/§3/§6, add a `source:` sub-bullet giving `path:line@commit`, the seat that
re-derived it, and the date. Where the formula is *derived* rather than measured (route counts, k = floor(rho·width)
with rho=0.90), link the derivation note (`docs/notes/2026-08-30_pi_q02-exactness-derivation.md`) rather than
restating it, so a future width change breaks the citation loudly instead of silently.

### What §4 needs before it can produce a verdict

§4 currently asks for independent re-derivation and ratio recomputation "with the pre-registered definitions (α per
arm; specialist baseline for P-Route per Section 2)" but states no thresholds, no units, and no falsifier for itself.
Given the manuscript's most consequential defect to date was a units slip — ppl differences reported as nats,
inflating the abstract's "+12 to +20 nats" by roughly 5–17×, with true forgetting ≈ +0.73 nats (EN) and +2.40 nats
(ES), and the P-Eros margin dropping from "10–70×" to ≈8× — §4 should require:

- [ ] a units label on every cell, and one units table per report (`nats` vs `ppl` never mixed in a column);
- [ ] the numeric threshold next to each ratio, copied from the pre-registration with its pointer, not restated
      from memory (P-Eros 0.3 nats; P-Acq ≤2.6 ppl per the acquisition protocol);
- [ ] the baseline artifact named for every Δ (which checkpoint, `_best`/`_last`, which split);
- [ ] an explicit falsifier sentence: what observation would invalidate the ladder's central reading, e.g. if
      cold-eval ppl at max width does not improve monotonically with width once batch regime is controlled, the
      acquisition claim is dead rather than merely noisy.

Note the trap this closes: `bg 15.51` / `el 15.24` cold-eval ppl versus Latin sisters at ~8–10 could be tokenizer
behaviour (Quinn's hypothesis, seq 30) or the batch-regime boundary (both trained at batch 1, like everything from
phase 6 on) — under the current §4 text, nothing distinguishes those, because regime is not a column anywhere.

### Recording place and sign-off (undefined in §0–§7)

- **Where results go:** propose `docs/reports/<YYYY-MM-DD>_<seat>_ladder-ra2-validation.md`, one line per checkbox
  with `pass | fail | n/a`, the artifact reference, and the artifact's sha256, opened in the same PR as any new
  failure-class entry (F5/F6/F-T/F-V). Today the checklist has no output path at all, so "validated" is unfalsifiable.
- **Who signs:** §7 requires validator ≠ author of the run artifacts, but the ladder and Addenda 1–5 are all
  Quinn-authored, and no section currently carries a signature. Propose one sign-off line per section:
  `<section> <seat> <ISO date> <sha256(artifact set)>`, and a standing rule that a section is not closed by its
  author. On this box the only non-authoring seats are pi-50 and pi-203.
- **Authorization caveat:** executing §0/§2 faithfully means reading 20 checkpoints (~65 GB in `out/`) and hashing
  artifacts — a raw-data pass, which my bound scope excludes until the operator opens it. Flagging rather than
  starting.

---

## Reproduction commands (for whichever seat executes this)

```bash
cd /srv/coding/bdh
grep -c '^== phase' out/logs/ladder_ra2_analysis.txt              # 10, not 20   (F-V3)
sed -n '5p' out/logs/ladder_ra2_analysis.txt                      # phase 2 batch=4 (F-V2)
awk -F'|' '/^\| *[0-9]+ *\|/{print $2"|"$6}' \
  docs/reports/2026-08-30_ladRA3-progress-report.md | head -12    # phase|batch table (F-V2)
sed -n '66,74p' scripts/ladder_ra2.sh                             # PREV_MULT<=192 -> BS=4 (F-V2)
sed -n '70,74p;88,100p' scripts/ladder_ra2_resume.sh              # BS=1 hardcoded; eval uses "$BS" (F-V2)
sed -n '25,31p' scripts/eval_router.py                            # --batch default 4 (F-V2)
ls out/ | grep -c 'ladRA2-.*_last.pt'                             # 20 checkpoints exist (F-V3)
stat -c '%s %y %n' out/logs/ladRA2_routdiag_p20.txt               # 0 bytes until exit (F-V3)
```
