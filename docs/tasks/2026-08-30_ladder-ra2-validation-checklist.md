# ladder_ra2.sh — Validation Checklist (run after completion)

**Date:** 2026-08-30 · **Author:** Quinn (review seat) · **Status:** pre-registered; execute when the R3 ladder finishes on the 4090

Validating the R3 ladder output BEFORE any number enters a report or the
manuscript. Checks reference the binding rules in
`docs/reviews/2026-08-28_r3-guidelines.md` (Sections 5–7). Failure classes
where established: F5 = consistency between derived numbers is not
verification.

## 0. Run integrity (do first)

- [ ] Parse `out/logs/ladder_ra2_analysis.txt`: every phase reached `done`;
      no phase silently skipped.
- [ ] **OOM audit (critical given reported occasional OOMs):** collect all
      OOM/retry markers. For every restarted phase, verify INIT was the
      previous phase's `_last` checkpoint (chain integrity), never a re-init
      from BASE. A phase that restarted from BASE invalidates its own and
      all downstream numbers.
- [ ] `step=10000` present in every phase checkpoint (Arm G/R convention).

## 1. Provenance (Section 6)

- [ ] Every run-property column in every derived table cites its artifact
      (checkpoint path / log line) or carries an explicit `computed` tag
      with the formula.
- [ ] No column is silently re-derived from another derived column.

## 2. Parameter/multiplier oracle (F5)

- [ ] For every phase checkpoint: `bytes/12` ≈ P(m), and
      P(m) = 786432·m + 262144 (+64·m), with m read from the checkpoint cfg
      (not from any script or report).
- [ ] All multipliers on the ×32 grid; schedule 20 phases ×128 → ×736;
      final ≈ 579,123,200 parameters.
- [ ] Per-head widths = 64·m (routes are in per-head neuron units).

## 3. Route-grid validity

- [ ] Every routing file contains/prints its route list; routes are
      boundary-aligned (one per expert-cap boundary incl. BASE) and in
      per-head neuron units (64×mult).
- [ ] 20 domains incl. `sv`; no duplicate `en` column in milestone
      matrices; no clamped/phantom route at the top of the grid.
- [ ] Instrument type labeled per table (boundary-aligned vs. historical
      linspace): historical P-Route/P-Det numbers are NOT boundary-comparable.

## 4. Arithmetic re-derivation (F5)

- [ ] Re-derive every derived cell independently. Two mutually consistent
      derived numbers are NOT verification.
- [ ] Falsifier ratios (P-Acq / P-Eros / P-Route / P-Det) recomputed from
      raw artifact values with the pre-registered definitions (α per arm;
      specialist baseline for P-Route per Section 2).

## 5. Protocol congruence (the PoC-v2 lesson)

- [ ] Attention state frozen/thawed exactly as pre-registered (v2 violated
      this; check the run config, not the report's claim).
- [ ] LR schedule congruent with the pre-registration.
- [ ] Eval protocol identical across phases and arms (eval_router.py at the
      fixed version, d10d389 or later).

## 6. Verification claims (Section 7)

- [ ] Exactness claims stated as ULP-bounded (not bit-equality) wherever
      the width changed.
- [ ] Exactness attribution names the operative mechanism (zero-init + RoPE
      frequency preservation, plus the mask in the ReLU regime), never the
      mask alone under a preceding selection operator.
- [ ] Citable S1/S2 evidence comes only from `verify_masked_forward.py` at
      4c8e51e or later (GX10-attested by pi-33). The original d1bc095
      version never produced a verdict — recorded as failure class F6 in
      the team A/B protocol.

## 7. Independence

- [ ] Validation executed by a seat that did not author the run artifacts
      (execution seat validates; Quinn/Pi cross-check).
- [ ] Validator findings enter the record as erratum or confirmation, each
      with its artifact reference.

---

## Addendum: transfer forensics (2026-08-31, Quinn)

Context: RA2 artifacts were copied ai -> gx10 (`/srv/coding/bdh`) ahead of
validation. Integrity: md5 manifest on both hosts, 24/24 checkpoints, 0
mismatches; `out/logs` and `data/` complete except one file (see F-T6).
Findings that change validation assumptions:

1. **F-T1 — execution timeline (hard).** Phase log birth times establish the
   executed order: en 08:52 -> es 09:16 -> pl 16:19 -> fr 16:55 -> de 17:36 ->
   nl 18:21 -> it 18:53 -> sv 19:28 -> da 20:05 -> pt 20:45 -> cz 21:31 ->
   ro 22:21 -> el 23:14 (fatal OOM 23:16). Milestone p10 ran directly after
   pt (pt_milestone10_routing.log 21:30:59).
2. **F-T2 — driver != committed script (hard).** The executed order deviates
   from `SEQ` in `scripts/ladder_ra2.sh` @HEAD (md5 2c0527a2 identical on both
   hosts; git reflog shows no restores; script mtime 08:52 = last pull). The
   executed driver file was NOT found in the readable tree (nothing matching
   under `/media/data/coding`, depth <= 2, newer than 30.08. 08:00). The
   remaining-SEQ for the resume is therefore undefined until the executed
   driver is recovered (operator/MiMo).
3. **F-T3 — `cz` is `cs` (hard).** `ladRA2_cz.log` line 2 reads
   `europarl cs: train 30,000,000 B` — the cz-labeled phase trained on
   europarl **cs** data (ai has only `europarl-v7.cs-en.cs.txt`; no cz files).
   Checkpoint naming is cosmetic and misleading: every validation table must
   map `ladRA2-cz` -> cs. With cz=cs, 12 distinct languages are trained.
4. **F-T4 — routing-diagnosis artifacts missing (hard).** Zero
   `ladRA2_routdiag_pN.txt` files exist in `out/logs`; the script's per-phase
   routing diagnosis never produced output. `ladRA2_pt_boundary_p10.txt` is
   an `eval_router.py` traceback (line 60); `ladRA2_ro_boundary_p10.txt`
   contains real results. P-Route/P-Det numbers for RA2 cannot be computed
   from surviving artifacts and must be re-evaluated from checkpoints (routes
   are re-derivable from checkpoint widths).
5. **F-T5 — non-fatal OOM warnings (soft).** `ro.log` and `cz.log` open with
   CUDA OOM warnings that recovered; consistent with the milestone-eval
   memory-collision hypothesis. Checkpoints complete with val_loss lines; no
   artifact impact visible.
6. **F-T6 — one data file not transferable (operator action pending).**
   `data/textmix/wikitext-103-raw/wiki.train.raw` is mode 660 asb:asb on ai
   and unreadable by a0-quinn; the only file missing on gx10.

Transfer inventory on gx10: `out/` 65 GB (24 ladRA2 checkpoints, md5-verified),
`out/logs` complete, `data/` complete except F-T6. Historical arm checkpoints
(~545 GB: armG/armGR/m5/diag/cl/seed) intentionally left on ai — resume does
not need them; copy on demand.

---

## Addendum 2: driver identified — F-T2/F-T6 resolved (2026-08-31, Quinn)

**F-T6 closed.** `wiki.train.raw` made readable (operator chmod a+r) and
transferred; md5 identical on both hosts
(`8a2d5ab8735b1246d49cf767b70d4dd0`). `data/` is now complete on gx10.

**F-T2 resolved** via MiMo statement (operator-relayed screenshot):

- The executed driver was **no script**: MiMo ran the phases ad-hoc — one
  `pipeline.run train` command per phase inside the conversation session,
  continuing from a summary of phases 1-8.
- MiMo's reconstruction (growth headers in the logs) matches the F-T1
  birth-time order exactly: en es pl fr de nl it sv da pt cz ro complete
  (12 phases), el attempt OOM. **Two independent evidence classes agree** —
  the executed order is provenance-hard.
- Divergence from the script SEQ starts at phase 6 (nl instead of cs).
- No `routdiag_pN.txt`/`boundary_pN.txt` exist because the script never ran
  as a batch. The surviving `ro_boundary_p10.txt` is MiMo's **manual**
  milestone-10 eval on the RO checkpoint; `pt_boundary_p10.txt` is an
  `eval_router.py` traceback (F-T4 unchanged: routing metrics must be
  re-evaluated from checkpoints).

**Resume definition (8 phases, 13-20):** remaining languages in script-order
positions are fi(9) hu(10) bg(11) et(13) el(14) sk(15) sl(19) lt(20), so the
resume SEQ is `fi hu bg et el sk sl lt`, growth 480→512 up to 704→736.
Correction to MiMo's count ("7 unused"): **el must be included** — the
attempt OOM'd and left no checkpoint, so el is retried at its script
position (phase 17). Interference analysis must use the **executed** order
for phases 1-12 and script order for 13-20; the seam must be flagged in
every table. Resume-order choice (script order recommended, matching
MiMo's suggestion) is pending operator/MiMo confirmation.

## Addendum 3 (2026-08-31): gx10 launch forensics + backend deviation record

### F-T7 launch forensics (resume phases 13-20 on gx10)

- Attempt 1 (04:03): CUDA-op JIT compile crash -- torch inductor builds
  cuda_utils.c, fatal error: Python.h (python3-dev missing on GB10).
  Trigger: pipeline/config.py:36 sets compile=True by default. Log kept:
  out/logs/ladRA2_fi.attempt1-inductor-crash.log.
- Attempt 2 (04:07): identical crash -- deployed script lacked the fix
  (82f99d7) because the earlier git pull completed only halfway
  (fetch ok, merge not applied; gx10 HEAD was still e7517fd). Log kept:
  out/logs/ladRA2_fi.attempt2-inductor-crash.log.
- F6 record (Quinn): attempt 2 was relaunched WITHOUT verifying deployed
  bytes. Rule applied from attempt 3 on: grep-gate on the deployed script
  (TORCHDYNAMO_DISABLE present) before every start.
- Attempt 3 (04:12): verified launch -- deploy repaired via
  git fetch + reset --hard origin/main, grep-gate passed,
  0 inductor lines, process alive >289 s (past the ~111 s crash mark),
  GPU 96 %. fi.log buffers via tee (python block buffering); step lines
  appear on first flush -- buffering, not failure (known from ai logs).

### Backend deviation record: compiled (ai) vs eager (gx10)

- ai phases 1-12: torch.compile active (compile=True default; ai has
  python3-dev, no crash, inductor kernels in use).
- gx10 phases 13-20: eager (TORCHDYNAMO_DISABLE=1; no python3-dev).
- Same dtype (bf16), same protocol (30MB/10k steps/alpha=0.9/unfrozen
  attn), same eval scripts. Deviation is kernel-selection and
  reduction-order class only -> ULP-bounded, accepted under Guidelines
  section 7 (bit-equality is not claimed across arms; every cross-arm
  comparison is ppl-based).
- Fix path if ever needed for parity: sudo apt install python3-dev on
  gx10, remove the export, re-run.

Note: git pull on gx10 is DEFERRED while the ladder runs -- bash reads
scripts incrementally; updating a running script in place is hazardous.

## Addendum 4 (2026-08-31): en domain-file gap (F-T8) + routdiag backfill spec (F-T4 remedy)

### F-T8: en domain file never existed as a standalone artifact

- pipeline/data.py maps "en" to the de-en pair (sources dict, data.py:65);
  the EN-side text materializes as europarl-v7.de-en.en.txt. The DOMAINS
  strings in the ladder scripts hardcode europarl-v7.en-en.en.txt -- a
  path no preparation step ever creates.
- gx10: file was ABSENT (verified 05:1x, ~6h before phase-13 routdiag
  would have needed it at 2049 ms/step). Fixed by copying de-en.en.txt
  -> en-en.en.txt (md5-identical 56a330fd9a6291c2540ee733e40e9dd2,
  287,250,069 bytes). The running resume opens the file lazily at eval
  time, so the fix lands before first use.
- ai: en-en.en.txt not readable for a0-quinn (absent or 660 asb:asb);
  de-en.en.txt IS readable with the same md5. Backfill resolves "en"
  via a verified home-directory copy; /media/data untouched per
  operator instruction.

### F-T4 remedy: routdiag backfill for phases 1-12 on ai

- Background: phases 1-12 ran ad-hoc without the script's routing
  diagnosis (F-T4). Checkpoints are frozen; eval_router is stateless,
  so diagnosis on the phase-final _last.pt is exactly the at-phase-end
  diagnosis (no further training touched these weights).
- Runs on ai (idle): read-only checkpoints, outputs to
  /home/a0-quinn/routdiag_backfill/routdiag_p{1..12}.txt + progress.log.
- Grid: boundary-aligned routes 8192+2048*k up to mult*64 (p routes at
  phase p) over the full 20-domain grid -- identical to what
  ladder_ra2.sh @HEAD would have produced; comparable to p13+ grids
  from the live gx10 resume. Batch 4 with automatic batch-1 retry
  (ULP-class; confusion/ppl expectations are batch-independent).
- Success criterion per phase: "joint full-width reference" line
  present in the output file.

### Backfill launch note (F6, Quinn): first launch failed instantly

- First launch (05:17): all 12 phases FAIL within one second -- shell
  escaping defect in the launch harness (\$PY written literally inside a
  quoted heredoc -> "command not found" per invocation). No eval ran;
  outputs were garbage and were discarded.
- Relaunched after rewrite (absolute venv python path, bash -n gate).
  Launch-verification rule applied: progress.log + GPU utilization +
  first real output lines checked before declaring the job running.

## Addendum 5 (2026-08-31): weight-decay leak on "frozen" neurons (F-T10) — found via weight-atlas implementation review

- Origin: the weight-atlas BDH support doc (section 7.3) reports a weight-decay
  leak on grad-masked "frozen" neurons. Quinn verified it directly against
  checkpoints: the leak is real and affects every arm that grows or gates via
  pipeline/train.py:225 (grad.mul_(mask) leaves materialized zero grads) with
  AdamW wd=0.1 over raw_model.parameters(): POC ra/ra2, ladRA2 primary,
  ladRA2s42, gate_from runs.
- Measurements (uniform, min approx max; own-N decoder geometry):
  POC ra2-a09 p3_last/p2_last on p1-units = 0.5865 (encoder, encoder_v,
  decoder identical); ladder es_last/en_last on en-units = 0.5828 encoder
  (n=1024) / 0.5820 decoder. Compounding: en-units at ro_last = 2.70e-3 of
  en_last amplitude, uniform (2.683-2.714e-3).
- Mechanism closed quantitatively, no free parameters: sum(lr)=5.451 over the
  10k schedule (lr 1e-3, warmup 1000, cosine to min_lr 1e-4) -> full-schedule
  decay exp(-0.1*5.451)=0.5798; each phase initializes from the predecessor's
  _best (p2 best@step 8950), so last-to-last ratio = 0.5798 / exp(-wd*tail)
  = 0.5866 vs measured 0.5865. Within-phase best-vs-last tail measured
  independently: uniform 1.0116. embed/lm_head (requires_grad=False) are
  bit-identical across phases — the contrast isolates the zero-grad mechanism.
- Consequences: "frozen by construction" is structural (mask, cos >= 0.9999,
  RoPE old part verbatim) but NOT amplitude-preserving (~0.58x/phase,
  ~2.7e-3 after 11 phases). RA2 arms are internally consistent (identical
  leak everywhere), so cross-arm ppl comparisons are unaffected; manuscript
  absolute "frozen" language needs rewording; neuron_importance consumers
  (prune/merge/gate) see decayed amplitudes. Fix decision belongs to the BDH
  side; until fixed, the decay factor is deterministic and divisible out.
- Full analysis: docs/reviews/2026-08-31_weight-atlas-bdh-implementation-review.md (F-2).

---

## Addendum 6 (2026-09-03): executability gaps in sections 0–7 — found while preparing to execute the checklist

**Author:** pi-50 (seat on gx10-50ef, review role) · **Date:** 2026-09-03 · **Base:** written against this file at
`4b1561e`, after the ff-merge of the 10 commits ending there; additive to Addenda 1–5, no upstream text altered.
Scope lease `s_bdh-cl_000005_acb0c0` (`file:///srv/coding/bdh/docs/tasks/2026-08-30_ladder-ra2-validation-checklist.md`).

Failure classes here are numbered **F-V** (verification-protocol defects), deliberately *not* F-T: the F-T register in
Addenda 1–5 tracks transfer/run forensics. These items are about whether this checklist can be executed by a seat
that did not write it. Every claim carries the command that reproduces it, re-run against `4b1561e`.

### F-V1: the executable sections never name which artifact set they validate (`_best` vs `_last`)

Each phase produced two checkpoints, `out/bdh_europarl_ladRA2-<lang>_best.pt` and `..._last.pt`, and §0/§2/§3 refer to
"phase checkpoints" generically. `_best`/`_last` do appear in this file — seven times — but always inside forensic
statements in the addenda (`:17` chain integrity on `_last`, `:222` at-phase-end diagnosis, `:253-260` F-T10 ratios),
never as a definition of the object under test. The difference is already visible in the completed ladder's headline
language (lt): **9.43** (test split at best) / **9.75** (best val) / **9.94** (`_last`, random-crop cold eval,
`out/logs/ladder_ra2_analysis.txt`). Three legitimate numbers, spanning 0.5 ppl, none the same measurement; the
routing diagnosis runs on `_last`.

**Ask:** one line in §0 fixing the validated artifact per section (§2 oracle on `_last`, §4 ratio recomputation on
whichever checkpoint the manuscript quotes), plus a standing rule that every reported cell names `_best` or `_last`.
Without it, two seats can execute §2 and §4 faithfully and disagree by design.

### F-V2: the Batch column is unreproducible from committed code, and two records contradict each other (§1)

One 20-phase series spans **three** optimization regimes:

- report table (`docs/reports/2026-08-30_ladRA3-progress-report.md`): batch counts over phases 1–12 are `7 × 1,
  4 × 2, 1 × 4` — reproduce with `awk -F'|' '/^\| *[0-9]+ *\|/{print $6}' <report> | tr -d ' ' | sort | uniq -c`;
- resume phases 13–20: `scripts/ladder_ra2_resume.sh:75` sets `BS=1` unconditionally;
- committed `scripts/ladder_ra2.sh:66-71`: `BS=4` when `PREV_MULT <= 192`, else `BS=1` — predicts batch 4 for
  phases 2–4 and batch 1 from phase 5. **Neither the report table nor the analysis log matches that rule.**

The two surviving records also disagree outright about the same phase:
`out/logs/ladder_ra2_analysis.txt:5` says `== phase 2: es (grow from mult 128 -> 160, batch=4, alpha=0.9) ==`, while
the report's row 2 gives ES batch `2`. §1 requires every run-property column to cite its artifact; the Batch column
cites a script that cannot produce either value. Most likely explanation (Addendum 3's eager-vs-compiled record is
the precedent): the committed script was tuned for a different box *after* those phases ran, so it is not the version
that produced the table.

Note the fix pattern already exists in-repo: the seed-42 replication ladder (`scripts/ladder_ra2_seed42.sh`, commit
`860258a`) derives batch through a named function `bs_for "$D"` and logs the executed order at launch (`:57`). The
ask below is RA2's historical record catching up to that standard, not a new convention.

**Asks:**
1. State the executed batch size per phase from an authoritative artifact (per-phase `out/logs/ladRA2_<lang>.log`
   argument echo), and tag the report's Batch column `computed` or `artifact:<path>` per §1.
2. Record explicitly that cross-phase P-Acq / P-Eros comparisons span non-constant tokens-per-step (4× between EN and
   the batch-1 majority). If the growth curve is meant to read as width-driven rather than budget-driven, say so and
   give the reasoning; if it is a confound, name it as one. Claim-scope question, not a re-run request.
3. §5 pins the eval protocol to a script version but the scripts tie eval batch to train batch
   (`scripts/ladder_ra2_resume.sh:83` trains with `--batch-size "$BS"`, `:102`/`:128` evaluate with `--batch "$BS"`),
   while `scripts/eval_router.py:30` defaults to `--batch 4`. So "eval protocol identical across phases and arms" is
   unsatisfiable as written: eval batch varies 4/2/1 with training phase. Addendum 4 argues batch-independence is
   ULP-class for confusion/ppl — then §5 should *cite that measurement* and pin the batch value, instead of pinning
   only the version.

### F-V3: §0 tells you to parse an artifact that cannot answer the question it asks

`§0: Parse out/logs/ladder_ra2_analysis.txt: every phase reached done; no phase silently skipped.`

Measured against the finished ladder (7,349 B, mtime 2026-09-03 08:14:29): the file carries headers for **10 distinct
phases — 1, 2, 13–20 — spread over 12 header lines**, because phase 13 (fi) appears three times
(`grep -oE '^== phase [0-9]+: [a-z]+' | sort | uniq -c | awk '$1>1'`). Consequences:

- A seat executing §0 literally gets `grep -c '^== phase'` = **12** and may conclude 12/20 phases ran. The true
  coverage is 10/20, and the extra two lines are *evidence of restarts* — precisely what §0's OOM audit is supposed to
  surface. Counting lines and counting phases give different answers here, and only one of them is meaningful.
- Phases 3–12 exist in this file not at all; their completion data lives only as a hand-formatted table in
  `docs/reports/`, i.e. in the document class this checklist exists to gate. That inverts the intended order
  (artifact → report). Meanwhile all 20 checkpoints do exist: `ls out/ | grep -c 'ladRA2-.*_last.pt'` → 20, including
  `cz_*` whose data is `cs` (Addendum 1 mapping).
- Routdiag/boundary artifacts stay **0 bytes until the producing process exits** (observed directly:
  `ladRA2_routdiag_p20.txt` 0 B at mtime = process start, 4,764 B at exit), and `scripts/ladder_ra2_resume.sh:102`
  and `:128` both use `>>`, so a rerun appends a second block under one filename. Any automated §0 check testing
  "file exists" passes on an empty file and cannot see a duplicated block.

**Asks:** name the artifact set that *must* contain all 20 phases and make §0 fail loudly if it doesn't; specify
"count distinct phases, not header lines" and require the duplicate count to be explained; add "non-empty AND
parseable, single block per phase" to the existence test; and state where the phases 3–12 record is, or regenerate it
from the 20 checkpoints so the machine-readable record is complete before the manuscript quotes it.

### F-V4: formulas asserted as oracles carry no source site (§2, §3, §6)

Three load-bearing formulas appear without a definition site: `P(m) = 786432·m + 262144 (+64·m)` with "final ≈
579,123,200 parameters" (§2); "routes are in per-head neuron units (64×mult)" (§3); and "zero-init + RoPE frequency
preservation, plus the mask in the ReLU regime" as the operative exactness mechanism (§6). Each is checkable, and
each belongs to a code location whose *implementation*, not prose, decides whether it holds. F-T10 (Addendum 5) is the
precedent that makes this more than tidiness: "frozen by construction" was structurally true but amplitude-preserving
only up to an AdamW weight-decay leak living in `pipeline/train.py:225` (`grad.mul_(mask)` leaves materialized zero
grads). Nobody had cited the site, so nobody had checked it, and the conclusion held only approximately.

**Ask:** per formula in §2/§3/§6, add a `source:` sub-bullet with `path:line@commit`, the seat that re-derived it, and
the date. Where a quantity is *derived* rather than measured (route counts, `k = floor(rho·width)`, rho=0.90), link
the derivation note (`docs/notes/2026-08-30_pi_q02-exactness-derivation.md`) instead of restating it, so a future
width change breaks the citation loudly rather than silently.

### F-V5: routing-confusion columns were labeled with domain names while holding route indices (fixed `4b1561e`)

Found by reading the phase-20 diagnosis (`out/logs/ladRA2_routdiag_p20.txt`, completed 06:26:01) whose confusion matrix
has an **empty diagonal for 18 of 20 domains**: late phases route near-deterministically but off their own label
(bg→pl 40/40, et→pt 40/40, el→ro 40/40, hu→nl 40/40, lt→sv 40/40, fi→lt 36/40), early phases spread, and only sk→sk
and sl→sl looked self-consistent. Cause, per Quinn (#49, owner of `eval_router.py`): columns are **prefix widths**
(`set_prefix()` masks neurons `[0, width_j)`, `width_j = 8192 + 2048·j`), but the printed header reused the
alphabetical `--domains` names. Print-only fix in `4b1561e`; row labels were always correct.

Two verification consequences belong in this checklist, not just in a changelog:

- **Correction scope.** Every confusion output produced before `4b1561e` (`poc_ra2_*`, `ladG`, `ladGR`, and the
  backfilled routdiags per Addendum 4) carries language names over route-index columns. Any quoted cell anywhere in
  `cl-bdh-manuscript.tex` or `docs/reports/` that asserts a language→expert relation must be re-read as a
  language→**width** relation or withdrawn. §3 should state the parse convention explicitly so a validator cannot
  re-introduce the error from the archived artifacts.
- **Nested-prefix semantics.** Routes are nested: route *j* contains all neurons of routes `<j`. So "domain X lands
  on width W" does *not* mean X ignores its own trained neurons — cs landing on 43008 (sk's width) includes cs's own
  segment. Specialization statements must therefore name the width and the nesting, and §3's P-Route baseline needs
  the same wording, otherwise the matrix will be read as evidence against specialization when it is evidence about
  prefix choice.

Method note worth keeping: my original reading proposed growth-order-vs-alphabetical as the hidden permutation. It fit
8 of 10 one-hot rows exactly, and sk/sl looked right because they are that permutation's only fixed points — but cs and
pl did not fit. Growth index and width index are equal *by construction* here, so the two hypotheses are
indistinguishable from this artifact alone; the failing rows were the signal that the artifact could not adjudicate.
When two orderings predict the same columns, say so instead of choosing one.

### What §4 needs before it can produce a verdict

§4 asks for independent re-derivation and ratio recomputation "with the pre-registered definitions" but states no
thresholds, no units, and no falsifier for itself. Given the manuscript's most consequential defect to date was a
units slip — ppl differences reported as nats, inflating "+12 to +20 nats" by roughly 5–17×, with true forgetting
≈ +0.73 nats (EN) / +2.40 nats (ES), and the P-Eros margin dropping from "10–70×" to ≈8× — §4 should require:

- [ ] a units label on every cell, one units table per report (`nats` vs `ppl` never mixed in a column);
- [ ] the numeric threshold beside each ratio, copied from the pre-registration with its pointer, not restated from
      memory (P-Eros 0.3 nats; P-Acq ≤2.6 ppl);
- [ ] the baseline artifact named for every Δ: which checkpoint, `_best`/`_last`, which split (closes F-V1 locally);
- [ ] an explicit falsifier sentence: what observation would invalidate the ladder's central reading — e.g. if
      cold-eval ppl at max width does not improve monotonically with width once batch regime is controlled, the
      acquisition claim is dead rather than merely noisy;
- [ ] a replicate rule: two measurements whose parameter lists come from the same expression are not independent.
      Verified here: `out/logs/ladRA2_boundary_p20.txt` is byte-identical to `ladRA2_routdiag_p20.txt`
      (md5 `7e1130f597c7701445ccb0a5c4b28863`, confirmed with `cmp` from my seat), because at phase 20 the
      "one route per expert" grid and the 20-entry diagnosis grid are the same 20 cumulative widths
      (`8192 + 2048·i`, i=0..19). Nice determinism datapoint; zero new information. Only p15-style phases give a
      genuinely distinct boundary grid (15 routes vs 20), so counting p20 boundary as a replicate would double-count
      one run.

Note the trap this closes: `bg 15.51` / `el 15.24` cold-eval ppl versus Latin sisters at ~8–10 could be tokenizer
behaviour (Quinn's hypothesis, seq 30) or the batch-regime boundary (both trained at batch 1, like everything from
phase 6 on). Under the current §4 text nothing distinguishes those, because regime is not a column anywhere.

### Recording place and sign-off (undefined in §0–§7)

- **Where results go:** propose `docs/reports/<YYYY-MM-DD>_<seat>_ladder-ra2-validation.md`, one line per checkbox
  with `pass | fail | n/a`, the artifact reference, and that artifact's sha256, opened together with any new
  failure-class entry (F5/F6/F-T/F-V). Today the checklist has no output path at all, so "validated" is unfalsifiable.
- **Who signs:** §7 requires validator ≠ author of the run artifacts, yet the ladder and Addenda 1–5 are all
  Quinn-authored and no section carries a signature. Propose one line per section
  (`<section> <seat> <ISO date> <sha256(artifact set)>`) and a rule that a section is not closed by its author. On
  this box the non-authoring seats are pi-50 and pi-203.
- **Authorization caveat:** executing §0/§2 faithfully means reading 20 checkpoints (~65 GB under `out/`) and hashing
  artifacts — a raw-data pass, which my bound scope excludes until the operator opens it. Flagging, not starting.
- **Open coordination question:** charter says `write_mandatory_for_repo_paths: true`, but the first scope leases ever
  recorded in bdh-cl were mine (`scope_seq` 1–5, 2026-09-03), after weeks of repo writes by several seats. Either the
  rule binds only seats that consult the charter, or enforcement lives somewhere I have not looked. This addendum is
  claimed-and-released either way; the room should decide what compliance means.
