# P-R family generating logs

Landed so that every number in `docs/reports/2026-09-11_pi-50_expansion-control-and-readout-operators.md`
has its producing log inside the repo rather than in a home directory that does not survive a reflash.

| log | produced by | what it contains | caveat |
|---|---|---|---|
| `r2_control.log` | `scripts/pi50/r2_expansion_control.py` | arms A/B/C of the random-expansion control, incl. inert-zero validation | — |
| `r1_ops.log` | `scripts/pi50/r1_readout_operators.py` | identity/top-K/extreme-value-shift arms on en/de/cs/bg/el | identity arm is the gate against the matrix row |
| `r1b.log` | `scripts/pi50/r1b_relative_reweight.py` | massnorm + softmix arms and the **first, crashing** `calibgain` attempt | the crash is the broadcast bug (`ALPHA.view(1,1,1,NB)` → `(NB,NB)`); kept as evidence |
| `r1b_calib.log` | same script, post-fix | `calibgain` fit loss 3.454→0.915, alpha spread, transfer eval | this is the run the report quotes |
| `r3_addr.log` | `scripts/pi50/r3_byte_addressing.py` | original P-R3: 0.762 headline, per-domain bimodality | **QUARANTINED numbers** — unbalanced-fit artifact, see report §9 |
| `r3b.log` | `scripts/pi50/r3b_byte_addressing_density.py` | 640-crop label generation, F-1 density curves, F-2 rows, F-3 confusion | labels valid; F-1/F-2/F-3 conclusions superseded by `r3d`/`r3c` (report §9.1) |
| `r3d.log` | `scripts/pi50/r3d_fit_ablation.py` | arms A/B/C fit ablation + composition-controlled density curve | projected arm A reproduces unprojected 0.681 exactly — the validity gate |
| `r3c_balanced_rerun.log` | `scripts/pi50/r3c_ood_addendum.py` (balanced, JL-projected) | territory-map self-check, 1.000 agreement, corrected F-2 six-language routes | filename note below |
| `a2.log` | `scripts/pi50/exp4_selfnll_selection.py` | run aborted after the table header, no results | kept because it shows where the harness died |
| `a2b.log` | same script | full 20-domain sweep, mean penalty lines | its `ppl_oracle` column used `(POS+3)*BLK` and truncated the newest block; selection column unaffected, fixed in `exp4b_budget.py` |
| `a2budget.log` | `scripts/pi50/exp4b_budget.py` | budget sweep 4 KB–1 MB × 20 domains + binary-search arm | source of report §8 |

Two naming notes, because this exact kind of thing is how audits go wrong:

* `r3c_balanced_rerun.log` was written to a file called `r3e.log` while I was iterating. It is **not** the
  output of `r3e_margins.py`. Renamed here to stop the collision.
* `r3e_margins.py` (Sonde C trigger calibration) has no text log: it was run to a terminal and its numbers
  were persisted structurally as `../r3_followups/r3e_margins.json`, which is the authoritative record.
  Re-run the script to regenerate; it is seeded and CPU-only.
