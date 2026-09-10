# BDH next steps — execution plan (pi-50, 2026-09-05)

**Status:** draft, seat-local. Narrative `docs/plans` draft belongs to A0-Quinn (#106); this file is the *execution*
side — who runs what, with which provenance columns, and against which pre-registered prediction. Nothing here
writes to `/srv/coding/bdh` or to `.200:/media/data/coding/bdh/out` without a claimed lease, an announced intent,
and operator's word on GPU time.

---

## 0. Blocking finding found while writing this plan

`scripts/ladder_ra2.sh` **cannot execute its own training step as committed.** Lines 76 and 86 contain literal
`\n` sequences inside unquoted bash argument positions:

```
.venv/bin/python -m pipeline.run train \n    --model bdh --dataset europarl \n ...
```

Bash resolves `\n` to the word `n`, so the invocation becomes `pipeline.run train n --model bdh …`, and
`pipeline/config.py:237` uses strict `parser.parse_args(argv)` → verified failure:

```
$ .venv/bin/python -m pipeline.run train n --model nonexistent-model-xyz
run.py: error: unrecognized arguments: n
```

Attribution and blast radius (`git show <c>:scripts/ladder_ra2.sh | grep -c '[\\]n'`):

| commit | date | author | escaped-newline lines |
|---|---|---|---|
| `cb48f62` | 2026-08-30 | OC/DSv4P/JSCS (authoring) | 0 |
| `f2f4707` | 2026-08-30 | Quinn (Agent Zero), *"fix ladder_ra2 routing diagnosis before GPU star…"* | **2** |
| `HEAD` | — | — | 2 |

Consequences, stated precisely:
1. The base-era RA2 artifacts predate the corruption, so they were **not** produced by the broken text — but the
   exact command line that produced them is **no longer recoverable from this file**, because the corrupted lines
   are exactly the train invocations. Any provenance claim resting on "the script documents the run" is void for RA2.
2. Re-running the flagship route-aware ladder today fails immediately at phase 1. This is a reproducibility defect,
   not a cosmetic one.
3. Fix is mechanical (restore line continuations) but it is **Quinn's file to fix**; I report, he repairs, and the
   repair should carry a comment noting the argv-injection hazard of escaped newlines.
4. F-V6 gains a field: *script re-runnability verified* (dry-run `bash -n` plus one argparse smoke test), because a
   recipe file that cannot parse is not a recipe.

---

## 1. Phase A — now, readout-independent

| # | Item | Owner | Gate |
|---|---|---|---|
| A1 | Report v1.1 amendments: closed form stated as *schedule product 0.892752 / f32 realization 0.892635 / measured 0.892636(2)*; RA2 `_best`-fork uncertainty band ±2.3%; replicate spread beside every mild-regime retention figure; repaired-joint marked unreplicated; G2 retention-without-route-awareness stated plainly | Quinn | none |
| A2 | Repair `ladder_ra2.sh` §0 defect + add dry-run smoke test to CI or `scripts/README` | Quinn (my report as evidence) | none |
| A3 | Publish instrument/Marin note under operator policy #98 (cite Jin et al., verbatim caution, our numbers separated, mtracker flagged LLM-generated secondary) | pi-50 | needs repo lease + operator OK (already given in substance) |
| A4 | Moments-census write-up: `exp_avg_sq == 0` as free per-parameter learning telemetry across 340 checkpoints | pi-50 | none |
| A5 | F-V6 amendment (literal init filename + script re-runnability) ; F-V7 follow-through ; **F-V8 ruling still outstanding** (`optimizer_state` saved, never restored) | pi-50 drafts, operator rules | operator |
| A6 | Audit Quinn's reclassification erratum against `docs/reviews/2026-09-03_pi-50_f-v7-null-contrast-correction-scope.md`; mechanical parts only if asked | pi-50 | after A1 lands |
| A7 | Manuscript skeleton with greppable `PENDING-RA2b` tokens | Quinn | none |

## 2. Phase B — after RA2b (ETA ≥5 days from 2026-09-04)

Scored against pre-registration, no exceptions: **P1** decay-artifact share, **P2** interference vs drowning,
**P3** prefix-masked routed breakage, **P4** every cell quoted against the seed floor, **P5** masked cells
`v==0 ∧ c=1.000000` (already **PASS**, first informative exit, run by Quinn). Each gets a signed PASS/FAIL line.
Section 4 rewrite takes the three-way decomposition (decay artifact / real interference / co-adaptation).

**Partial-readout policy, decide before day 5** (my vote): a partial chain may support P1/P3 *directionally*;
it may not support any headline retention number, because those are exactly the claims the replicate floor prices.

## 3. Phase C — after B

QAT proposal gate revisit (architecturally sound post-fix?); sv-backfill fixed-regime variant if the 20-phase G2
geometry is needed for the paper; weight-atlas per-phase norm forecast as a standing instrument.

---

## 4. The 4090 workstream (drafted here, executable by the GPU owner)

Rationale: RA2b occupies gx10 for ≥5 days, and two measurements that price our *existing* claims do not depend on
its readout. Operator freed `.200` in #99. Both protocols live in `tools/four90/protocol_4090.sh`:

### S — seed replicate (`ladS-*`)
Two identical short chains, same recipe as Arm G (defaults inherited, no lr flags), differing only in run name.
Purpose: **size** run-to-run variance, which #101 proved exists (G vs GR differ at cos ≈ 0.85 on shared geometry)
but never quantified. Predictions pre-registered:
- **S1** the two runs will not be bit-identical (given #101, failing this would mean something else is deterministic).
- **S2** routed-bg/el spread between the pair is the floor any claim below ~10–18% must clear; the reported floor
  is max(spread), not mean.
- **S3** per-phase decay factor measured on masked blocks equals 0.892636(2) *within* each run — decay is not part
  of the noise budget, so a wide spread here would falsify the closed form.

### M — missing discriminative cell (`ladMD-*`)
Plain growth (no route-aware masking) **+ aggressive decay** (RA2-style cosine: `--warmup-iters 1000
--lr-decay-iters 10000`), 5 phases. This cell has never been run and is the only cheap way to separate
"joint training co-adapts" from "decay regime differs" in section 4. Predictions:
- **M1** if decay dominates, joint ppl at phase 5 approaches the RA2 collapse scale; if co-adaptation dominates, it
  stays near plain-growth levels.
- **M2** masked-block ratios equal the cosine-schedule product (per-phase, computed from that run's own cfg) — a
  third independent confirmation, this time on a schedule we chose deliberately.
- **M3** routing does not break when old capacity is masked-but-not-route-trained (contrast with routed G 2.50→41.31).

### Execution discipline (non-negotiable, mechanically enforced by the script)
Every phase appends a JSONL manifest row: git SHA, host, GPU id, argv, resolved cfg dump, **init filename literal**
(`_best` vs `_last`), batch, block/window sizes, kernel regime (compile/eager), wall time, checkpoint sha256. That
is F-V6 implemented rather than advocated. Preflight refuses to start if the GPU is busy, if `out/` is unwritable,
or if a ladder from another seat is running — and it never trains: preflight is read-only by construction.

### What I am *not* doing
I hold read-only access on `.200` and no GPU-authority claim. These scripts are drafts for the owner (OC/R3/operator)
to run under their own task claim, or for me to run only if operator explicitly grants the write path and I announce
prefix + config on the bus first (§ push rule). No launch happens silently.

---

## 5. Open questions needing a human answer

1. **F-V8**: formalize "`optimizer_state` written at `train.py:71`, never restored anywhere in `pipeline/` or
   `scripts/`" as a repo finding, or keep it as an interpretation caveat? It changes how phase boundaries are described.
2. **GPU ownership on `.200`**: who claims S and M, and does OC's atlas scan get priority?
3. Does anyone object to `PENDING-RA2b` becoming a hard merge-blocking token in A7?
4. `es` transition (#107): which era's log supplied en's argmin — base-era `.200` logs or gx10 resumes? My inversion
   says 9594, Quinn's extraction says 9200, and es is the only transition where my head-spread degrades 100×.
