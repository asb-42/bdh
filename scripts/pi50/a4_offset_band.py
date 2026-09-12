#!/usr/bin/env python3
"""A4 - measure the window-vs-val instrument offset and the routed/acquisition band from landed artifacts.

Authorized by the operator 2026-09-12 ("A4 + A7 OK"), bus #235 thread. READS ONLY committed artifacts and WRITES
ONLY docs/reports/data/a4_offset_band/. CPU, no model loads, no training, no GPU.

Why this exists: the manuscript excuses routed/acquisition ratios "up to 1.13" as "consistent with the known
instrument offset", and the cited offset (+5-9 %) traces to a single RA2-era lt control rather than a measured
distribution. Nobody had actually measured either quantity across domains. Both are computable from artifacts we
already ship, so the question should never have been rhetorical.

Two distinct quantities, deliberately kept apart:

  D1 INSTRUMENT OFFSET - same weights, two instruments.
     matrix diagonal cell (checkpoint == eval_lang, val stream, EVAL_BATCH pinned to 1)
       divided by
     acquisition exit_ppl for that language (training-window crop, logged at phase end).
     Because the model state is identical in both numbers, ANY difference is instrument, not science. n=20.

  D2 ROUTED COST - same instrument family, different states.
     routed ppl at the final checkpoint (ladRA2b_routdiag_p20.txt, own-prefix row)
       divided by
     that domain's own-phase acquisition exit.
     This is the quantity sec 5 quotes. n=20.

Both are reported at DOMAIN level: one observation per domain, median + min + max, no pooled interval over crops
(see docs/reports/2026-09-11_pi-50_expansion-control-and-readout-operators.md sec 9.7 for why crop-level iid
bands were wrong here).
"""
import csv
import glob
import os
import re
import statistics as st

def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "pipeline")) and os.path.isdir(os.path.join(d, "docs")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("cannot locate repo root from scripts/pi50/")


REPO = _repo_root()
DATA = os.path.join(REPO, "docs", "reports", "data")
OUT = os.path.join(DATA, "a4_offset_band")
ACQ_CSV = os.path.join(DATA, "2026-09-05_ra2b_acquisition.csv")
MATRIX_CSV = os.path.join(DATA, "2026-09-10_ra2b_matrix.csv")
ROUTDIAG = os.path.join(REPO, "out", "logs", "ladRA2b_routdiag_p20.txt")


def gate(cond, msg):
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")


def main():
    os.makedirs(OUT, exist_ok=True)
    acq = {r["lang"]: float(r["exit_ppl"]) for r in csv.DictReader(open(ACQ_CSV))}
    gate(len(acq) == 20, f"expected 20 acquisition rows, got {len(acq)}")
    diag = {}
    for r in csv.DictReader(open(MATRIX_CSV)):
        if r["checkpoint"] == r["eval_lang"]:
            diag[r["eval_lang"]] = float(r["ppl"])
    gate(len(diag) == 20, f"expected 20 diagonal cells, got {len(diag)}")
    txt = open(ROUTDIAG).read()
    routed = {m[0]: float(m[1]) for m in
              re.findall(r"^\s+(\w\w)\s+([\d.]+)\s*$", txt.split("domain   routed")[-1], re.M)}
    gate(len(routed) == 20, f"expected 20 routed rows, got {len(routed)}")
    # sanity gate: the diagonal must reproduce known exits independently quoted elsewhere (cs/da/de)
    for L, want in (("cs", 3.48), ("da", 2.85), ("de", 2.73)):
        gate(abs(diag[L] - want) < 0.011, f"diagonal {L}={diag[L]} does not reproduce known exit {want}")

    d1 = sorted(((L, acq[L], diag[L], diag[L] / acq[L]) for L in acq), key=lambda x: x[3])
    d2 = sorted(((L, acq[L], routed[L], routed[L] / acq[L]) for L in acq), key=lambda x: x[3])
    with open(os.path.join(OUT, "d1_instrument_offset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang", "acquisition_exit_window_crop", "val_stream_diagonal", "ratio", "pct"])
        w.writerows([[L, a, v, round(r, 4), round((r - 1) * 100, 2)] for L, a, v, r in d1])
    with open(os.path.join(OUT, "d2_routed_cost.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang", "acquisition_exit", "routed_p20", "ratio", "pct"])
        w.writerows([[L, a, v, round(r, 4), round((r - 1) * 100, 2)] for L, a, v, r in d2])

    def summ(rows, name):
        v = [r for *_, r in rows]
        print(f"\n{name}  n={len(v)}")
        print(f"  median {st.median(v):.3f}   mean {st.mean(v):.3f}   min {min(v):.3f} ({rows[0][0]})"
              f"   max {max(v):.3f} ({rows[-1][0]})")
        return {"n": len(v), "median": round(st.median(v), 4), "mean": round(st.mean(v), 4),
                "min": round(min(v), 4), "max": round(max(v), 4),
                "min_lang": rows[0][0], "max_lang": rows[-1][0]}
    s1, s2 = summ(d1, "D1 instrument offset (same weights, window crop -> val stream)"), \
             summ(d2, "D2 routed cost (p20 own-prefix / acquisition exit)")
    print(f"  D2 domains above 1.08: {sum(1 for *_, r in d2 if r > 1.08)}/20 ; above 1.13: "
          f"{sum(1 for *_, r in d2 if r > 1.13)}/20")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        csv.register_dialect("x")
        import json
        json.dump({"d1_instrument_offset": s1, "d2_routed_cost": s2,
                   "claim_1_02_to_1_13_reproduced": bool(s2["max"] >= 1.13),
                   "claim_offset_plus_5_to_9_pct_reproduced": bool(s1["min"] >= 1.05),
                   "instruments": {"d1": "val stream, EVAL_BATCH=1, checkpoint==eval_lang diagonal",
                                   "d2": "own-prefix routed ppl from ladRA2b_routdiag_p20.txt"},
                   "denominator": "per-language exit_ppl from 2026-09-05_ra2b_acquisition.csv (training-window crop)"},
                  f, indent=1)
    print("\nwrote d1_instrument_offset.csv, d2_routed_cost.csv, summary.json to", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
