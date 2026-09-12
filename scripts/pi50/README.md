# `scripts/pi50/` — phase-1 instruments

Seat **pi-50** (gx10-50ef) analysis code. Everything here was written to answer one pre-registered question
on the RA2b ladder; nothing here trains anything. Index of what each artifact *establishes*: see
[`docs/PHASE1-MANIFEST.md`](../docs/PHASE1-MANIFEST.md) (regenerate with `phase1_manifest.py`).

## Policy that keeps phase 2 from inheriting phase 1's assumptions

1. **These files are frozen instruments, not a library.** Each number quoted in the manuscript or in
   `docs/reports/` is reproducible from the commit it was produced at. Do **not** parameterize them for the
   next experiment — a refactor that "only adds an env var" destroys the provenance link between a cited
   number and the code that made it. Write a new script for a new question; copy from these freely.
2. **Hardcoded phase-1 facts you must change if you reuse code:** ladder prefix `ladRA2b-*` (12 scripts),
   final latent width `N=47104`, block size `2048`, the 20-domain `SEQ` order
   `en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt`, and checkpoint filenames
   `out/bdh_europarl_<PREFIX>-<lang>_last.pt`. Grep for `ladRA2b` before trusting any result on new weights.
3. **Two indexing conventions coexist and have already caused one off-by-one:**
   * *cumulative-prefix width index* — language at ladder position `p` selects `WIDTHS[p+3]`, i.e. width
     `(p+4)*2048`; English owns indices 0–3 (the four base blocks). Used by the addresser family
     (`r3*`) and asserted across all 20 languages at import of `r3c_ood_addendum.py`.
   * *block index* — the newly appended block is `blk[p+2]` (English has none). Used by `exp3_delta_m.py`.
   Never mix them in one expression.
4. **Instruments that produce quotable numbers carry a sanity gate.** If you edit one and the gate stops
   firing, you have broken it silently. Known gates: matrix diagonal must reproduce ladder exit PPLs
   (`matrix_eval.sh`, `exp3_masked_vs_free.py`); free-English PPL must land in 15–60 with oracle-masked < 4
   (`exp4_selfnll_selection.py`, `exp4b_budget.py`); identity arm must match the matrix row within noise
   (`r1_readout_operators.py`, `r1b_relative_reweight.py`); inert-zero blocks must leave PPL bit-exact
   (`r2_expansion_control.py`); projected fit must reproduce unprojected agreement exactly
   (`r3d_fit_ablation.py`).
5. **Outputs default inside the repo or under `$R3B_OUT` / `$MATRIX` overrides.** One exception existed and
   was fixed: `exp3_delta_m.py` read its matrix from a seat-local `~/bdh-review` path; it now defaults to
   `docs/reports/data/2026-09-10_ra2b_matrix.csv` (byte-identical input, md5 verified before repointing).

## What each script is

| script | answers | status |
|---|---|---|
| `matrix_eval.sh` | full 20×20 serving matrix (every checkpoint × every domain) | TOOLING, reusable; guards `ALLOW_EVAL=1` + `ANNOUNCED`, resume-safe JSONL manifest |
| `protocol_4090.sh` | safe remote-run protocol on the 4090 while other seats hold it | TOOLING; preflight refuses during live jobs |
| `arm_identity_check.py` | are "repaired"/"joint" arms actually distinct checkpoints? | CITABLE (all 15 comparable pairs differ; embed diverges despite `--freeze-attn`) |
| `exp3_delta_m.py` | did old weights move at all? (ΔM census) | CITABLE (0.0000 churn on all old decoder blocks, 19 transitions) |
| `exp3_encoder_churn.py` | same question for encoder/encoder_v columns | CITABLE (zero churn, zero optimizer activity) |
| `exp3_masked_vs_free.py` | is damage erasure or access loss? | CITABLE — the decisive masked-vs-free result (all 19 recover to exit PPL exactly) |
| `exp3_calib_null.py` | can activation energy pick the territory? (A1) | CITABLE negative (1/20 best selector; every selector picks base block 0) |
| `exp4_selfnll_selection.py` | label-free argmin-NLL selection (A2) | SUPERSEDED by `exp4b_budget.py` — oracle column used `(POS+3)*BLK`, truncating the newest block |
| `exp4b_budget.py` | does selection accuracy depend on calibration bytes; is bisection possible? | CITABLE (20/20 at 4 KB–1 MB; binary search 4–6/20 ⇒ non-unimodal) |
| `atlas_territory_stats.py` | per-territory weight conditioning from Weight Atlas tiles | CITABLE with caveats (Pearson/Spearman reported together; English base blocks flagged as pseudo-replicates) |
| `r2_expansion_control.py` | how much of the damage is pure arithmetic winner-drift? | CITABLE — the 83 % result; inert-zero control validates the harness |
| `r1_readout_operators.py` | do inference-time readout operators rescue old languages? | CITABLE negative (top-K and extreme-value shift both worse; two arms provably vacuous under `self.ln`) |
| `r1b_relative_reweight.py` | does *relative* reweighting help where absolute fails? | CITABLE negative (calibgain rediscovers oracle masking specialized to the fitted language) |
| `r3_byte_addressing.py` | can byte geometry predict the route? (P-R3 v1) | SUPERSEDED — 0.762 headline and the "es/pl/sk fail" structure were unbalanced-fit artifacts |
| `r3b_byte_addressing_density.py` | F-1 density + F-2 unseen rows + feature persistence | inputs CITABLE (labels, persisted sparse features); its own F-1/F-2 conclusions superseded by `r3d`/`r3c` |
| `r3c_ood_addendum.py` | corrected territory map, unseen-language inference, JL-projected balanced fit | CITABLE (F-2 table; mapping asserted at import) |
| `r3d_fit_ablation.py` | was the failure starvation or geometry? | CITABLE — arms A/B/C: 0.681 / **1.000** / 0.988, control-domain mean 0.125 → 1.000 |
| `r3e_margins.py` | stage-1 trigger calibration for Sonde C | CITABLE (in-support floor 0.0139; escalation rates per unseen input) |
| `phase1_manifest.py` | generates `docs/PHASE1-MANIFEST.md` | TOOLING; `--check` mode is CI-able |

## Data these scripts consume (all landed in-repo)

* `docs/reports/data/2026-09-10_ra2b_matrix.{csv,md}` + `ra2b_matrix_raw/raw_*.txt` — serving matrix and raw logs.
* `docs/reports/data/2026-09-05_ra2b_acquisition.csv` / `.json` — per-phase totals and exit PPLs.
* `docs/reports/data/r3_followups/*.json` — fit ablation, margins, corrected confusion/OOD routes.
* `docs/reports/data/pr3_logs/*.log` — generating logs for every P-R-family number quoted above.
* `docs/reports/data/ood_inputs/README.md` — where the unseen-language slices came from on `.200`, their
  sizes and checksums, and the script-composition census (the bytes themselves are deliberately **not** in
  the repo: licensed corpus text, and 10 MB of it would make the docs tree heavier than the code).

### Manifest provenance is inherently a two-step commit

`phase1_manifest.py` records, per artifact, the short SHA of the commit that last touched it. Regenerating *before*
committing therefore stamps the previous HEAD, and `--check` goes stale again the moment the content commit lands.
Expected sequence, not a bug to debug: land content -> regenerate -> land a small "manifest refresh" commit. If
`--check` reports stale while the working tree is clean, that is this case; re-run and commit.
