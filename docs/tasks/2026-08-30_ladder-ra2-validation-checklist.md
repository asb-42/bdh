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
