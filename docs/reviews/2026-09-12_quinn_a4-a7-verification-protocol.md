# Pre-registered verification protocol: A4 (retention band) and A7 (multi-domain expansion control)

**Status:** registered *before* results. Author: A0-Quinn. Operator assignment: Pi-50 runs A4 and A7 (#237); Quinn verifies the results when they land.

**Purpose:** fix, in advance, what "verified" means for these two items, so the verification cannot be retrofitted to the outcome. This is the same discipline the manuscript applies to its own experiments.

---

## Ground rules (both items)

1. **Numbers are re-derived by me from the landed artifacts**, never taken from the completion envelope's prose. A summary is a claim; the data file is evidence.
2. **Every artifact is hashed** (sha256) at the commit it lands in, so each number traces to a fixed byte string.
3. **A claim is accepted only if the artifact carrying it also carries its instrument**: which dataset, which checkpoint, which eval protocol, which pairing.
4. **Nothing is accepted that the pre-registration did not name**, and nothing named here is dropped without a stated reason.
5. **Any manuscript edit must be no stronger than the measurement.** A fix that makes the sentence broader than the data is a failed fix, even when the numbers are right.

---

## A4 — retention band (routed versus acquisition)

**Artifact in question:** `docs/reports/data/2026-09-10_ra2b_matrix.csv` (lt rows) plus the fixed-regime readout report. Pi-50's offer: compute the real distribution, domain-level, named instruments per column, reported with median and range, **no iid band on crops**.

| # | Check |
|---|---|
| A4-1 | **File identity:** hash recorded; row count and column set stated explicitly. |
| A4-2 | **Row selection:** exactly the 20 lt domains; the filter is stated and re-runnable; no hand-picked subset, no dropped domain without a named reason. |
| A4-3 | **Pairing defined in advance:** every ratio pairs routed@p20 against the *acquisition exit* value used in the Fig-3 caption (claimed median 1.09, range 1.02-1.13). I recompute that pairing first; only if it reproduces the caption does any other acquisition column become admissible — and then it must be labelled as a different quantity, not folded into the same band. |
| A4-4 | **Instrument per column:** each column names its instrument (window-vs-val offset, train-vs-eval-batch). Folding two instruments into one column is itself a finding. |
| A4-5 | **Statistics shape:** 20 domain-level observations, median plus range (or IQR). No iid band on crops; no Wilson interval on crops. |
| A4-6 | **Pre-committed decision rule respected:** wider than +5-9 % → the paper says the wider thing; narrower → 1.13 is the outlier to explain. The rule is applied to the data, not selected after seeing it. |
| A4-7 | **Consistency sweep:** every occurrence of the retention bound in the TeX (`≤ 8%`, `1.02-1.13`, `instrument offset`) is updated consistently. No stale bound survives in abstract, results, caption, or limitations. |
| A4-8 | **Falsification condition:** if the measured band exceeds the largest admitted value (1.13), the phrases "retention equals acquisition" and "statistically indistinguishable" must be weakened to what the interval supports. If they survive unchanged, the fix failed. |

---

## A7 — expansion control across domains

**Artifact in question:** the new instrument (proposed `r2b_expansion_control_multidomain.py`) plus its landed outputs. Pi-50's offer: a fresh instrument rather than a parameterized `r2`, eval-only, pre-registered before launch.

| # | Check |
|---|---|
| A7-1 | **Pre-registration precedes results:** the bus sequence number of the pre-registration is lower than the sequence number of the results. The pre-registration names the era checkpoint list, the width targets, the composition control, the eval protocol, and the domain list. |
| A7-2 | **Instrument freshness:** the file is new, not a parameterized `r2`; `git show` confirms the original instrument is untouched (frozen-instrument policy, provenance preserved). |
| A7-3 | **Eval-only:** no optimizer construction, no backward/step that updates parameters, no new checkpoints written, no new training logs. Verified by grep **and** by checking the era directories for new `.pt` files. |
| A7-4 | **Checkpoint identity:** every loaded checkpoint matches the manifest sha256 and exists. No silent substitution, no "closest available" stand-in. |
| A7-5 | **Arms comparable to the en control:** at minimum the same arm structure (random block, real block, inert/zero block), the same width targets, and both units reported separately (log-scale fraction and linear fraction). A reduced arm set breaks comparability with the number already in the paper and must be declared as a different experiment. |
| A7-6 | **Numbers recomputed:** for each domain, the random-block cost factor and the reproduced-damage fraction are recomputed by me from the outputs. Acceptance = match at the reported precision. |
| A7-7 | **The general claim is earned:** the abstract sentence may be broadened only if the multi-domain measurement supports the broad form. If bg/el reproduce a materially different fraction, the abstract must state the measured range by domain group, not the en value alone. |
| A7-8 | **No new headline:** if the multi-domain run produces a new claim, it is reported as a new measurement with its own pre-registration, not folded into the 83 %/62 % headline. |

---

## Out of scope for this protocol

- **A3, A5, A6, A8 and the init-versus-trained theory sentence:** my review when they land; mechanical, no pre-registration needed.
- **Checkpoint release policy:** operator decision, closed (hash manifest only).

---

## Registration

Registered on the bus and in the local ledger **before any A4/A7 result lands**. If a check here turns out to be wrong, it is amended in the open, with the amendment dated and the reason stated — never silently.
