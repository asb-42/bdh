#!/usr/bin/env python3
"""A7 - MULTI-ERA RANDOM-EXPANSION CONTROL (fresh instrument; operator GO 2026-09-12, bus #235 thread).

Why this is a new file rather than an env var on scripts/pi50/r2_expansion_control.py: phase-1 scripts are frozen
instruments (scripts/pi50/README.md). The number already in the manuscript was produced by that exact code, and a
refactor that "only adds a parameter" severs the provenance link between the published figure and the thing that
made it. This file re-implements the measurement for a new question and cites the phase-1 source lines it mirrors.

QUESTION. `r2_expansion_control.py` measured the random-expansion control on the ENGLISH-era checkpoint only, and
the abstract generalizes it ("a single randomly initialized block already costs 4.1x / reproduces 83 % of the
damage"). For bg/el, where joint degradation reaches 37.8x, the fraction reproduced without learning is unmeasured.
This instrument measures it per era, so the claim is either bought or scoped.

PRE-REGISTERED DESIGN (fixed before the first run; no era or domain added afterwards on the basis of its result)
  Eras      : ladder phases 1,5,8,11,14,17,19 -> en, de, pt, bg, el, ro, sl. Chosen to span the Latin/Cyrillic/
              Greek/Uralic families and both ends of the damage range, then FROZEN. Phase 20 (lt) is excluded
              because it is already at final width (add = 0, nothing to expand).
  Target    : every era expanded directly to the final RA2b latent width N=47104 (mult 736), whole 2048-neuron
              blocks only. Same protocol constants as phase 1: BLK=2048, NH=8, D=512, block_size from cfg,
              iters=25, batch=8, crop generator seeds 4242 (eval) / 777 (random init), bf16 autocast.
  Domains   : per era, the era's OWN language plus 'en' as a fixed early-acquired reference. Two observations per
              era, seven eras, fourteen numbers - reported, not averaged away.
  Arms      : base  = era checkpoint, unexpanded                          (-> ppl_base)
              A     = era checkpoint + iid-Gaussian blocks to final width (-> ppl_rand)   [NOT the pipeline's true
                      fresh-BDH init; same stated limitation as phase 1]
              C     = era checkpoint + exact-zero blocks                  (-> layout validator)
              mask  = arm-A model evaluated with neurons masked back to the era width (access control)
              real  = joint serving at the final lt checkpoint, read from the landed matrix (-> ppl_real)
  Statistic : damage fraction f_log = ln(ppl_rand/ppl_base) / ln(ppl_real/ppl_base), and the same quantity in
              linear perplexity units f_lin = (ppl_rand-ppl_base)/(ppl_real-ppl_base). Both, always: reporting one
              without the other is the defect found in the rev-4 cycle.
  Gates     : G1 base free ppl within +-5 % of that era's logged acquisition exit (else the slice/crop construction
                 differs for this era and nothing downstream is interpretable)
              G2 |C - base| / base < 5e-3  (was 1e-6 until the pilot showed bf16 reduction-order noise alone is
                 ~4e-6; a moving C by any large factor means new columns were written into old territory:
                 decoder rows are head-major h*N+n, the trap documented at scripts/verify_masked_forward.py:110)
              G3 masked-A within +-2 % of base (masking must restore access, as exp3 showed at every transition)
              G4 quote f only where ln(ppl_real/ppl_base) > ln 2, i.e. real damage at least 2x; otherwise the
                 denominator is noise and the fraction is meaningless - report "not defined" instead.
  Prediction under the manuscript's generalized reading: f_log ~ 0.83 for every era.
  Falsifier  : any era with f_log outside ~[0.6, 0.95], or f_log near/below the seed floor, means the abstract's
               general sentence is unsupported and must be scoped to the tested era(s).

Run:  .venv/bin/python scripts/pi50/a7_multiera_expansion.py                # all eras
      A7_ERAS=bg A7_ITERS=3 .venv/bin/python scripts/pi50/a7_multiera_expansion.py   # pilot (gates still enforced)
Writes: docs/reports/data/a7_multiera/{results.csv,summary.json,run.log}
"""
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from pipeline.analyze import _load_model                      # noqa: E402
from pipeline.data import _europarl_blocks                    # noqa: E402

BLK, NH, D = 2048, 8, 512
FINAL_WIDTH = 47104                                            # mult 736 * 512 / 8, the lt-era latent width
PHASE_LANG = {1: "en", 5: "de", 8: "pt", 11: "bg", 14: "el", 17: "ro", 19: "sl"}
ERAS = [int(x) for x in os.environ.get("A7_ERAS", ",".join(map(str, PHASE_LANG))).split(",")]
IT = int(os.environ.get("A7_ITERS", "25"))
BB = int(os.environ.get("A7_BATCH", "8"))
MATRIX = "docs/reports/data/2026-09-10_ra2b_matrix.csv"
ACQ = "docs/reports/data/2026-09-05_ra2b_acquisition.csv"
OUTDIR = "docs/reports/data/a7_multiera"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_maps():
    import csv
    M, acq, mult = {}, {}, {}
    for r in csv.DictReader(open(MATRIX)):
        M.setdefault(r["checkpoint"], {})[r["eval_lang"]] = float(r["ppl"])
    for r in csv.DictReader(open(ACQ)):
        acq[r["lang"]] = float(r["exit_ppl"])
        mult[r["lang"]] = int(r["mult_to"])
    return M, acq, mult


def crops(lang, bs):
    raw = _europarl_blocks("data", 30_000_000, langs=(lang,))[lang]
    d = torch.from_numpy(np.frombuffer(raw["val"] + raw["test"], dtype=np.uint8).astype(np.int64))
    half = (len(d) - bs - 1) // 2
    return d[half:]                                            # held-out half, as in phase 1


def ppl(model, N, dd, iters, mask_to=None):
    m = None
    if mask_to is not None:
        m = torch.ones(N, device=dev)
        m[mask_to:] = 0.0
    g = torch.Generator().manual_seed(4242)
    ls = []
    bs = BS
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
        for _ in range(iters):
            ix = torch.randint(len(dd) - bs - 1, (BB,), generator=g)
            x = torch.stack([dd[int(i):int(i) + bs] for i in ix]).to(dev)
            y = torch.stack([dd[int(i) + 1:int(i) + 1 + bs] for i in ix]).to(dev)
            lo = model(x, None, None, neuron_mask=m) if m is not None else model(x)
            lo = lo[0] if isinstance(lo, (tuple, list)) else lo
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item())
    return math.exp(sum(ls) / len(ls))


STATE = {}


def build(new_width, zeros):
    """Era checkpoint grown to new_width with old bytes copied verbatim (mirrors r2:70-101)."""
    dec, enc, env, fq = STATE["decoder"], STATE["encoder"], STATE["encoder_v"], STATE["attn.freqs"]
    st = {"embed.weight": STATE["embed.weight"], "lm_head": STATE["lm_head"]}
    add = new_width - N0
    assert add >= 0 and add % BLK == 0, f"add={add} not whole {BLK}-neuron blocks"
    if add == 0:
        st.update(decoder=dec, encoder=enc, encoder_v=env, **{"attn.freqs": fq})
    else:
        if zeros:
            st["decoder"] = torch.cat([dec.view(NH, N0, D), torch.zeros(NH, add, D)], 1).reshape(-1, D)
            st["encoder"] = torch.cat([enc, torch.zeros(NH, D, add)], -1)
            st["encoder_v"] = torch.cat([env, torch.zeros(NH, D, add)], -1)
        else:
            gen = torch.Generator().manual_seed(777)
            st["decoder"] = torch.cat([dec.view(NH, N0, D),
                                       torch.randn(NH, add, D, generator=gen) * float(dec.std())], 1).reshape(-1, D)
            st["encoder"] = torch.cat([enc, torch.randn(NH, D, add, generator=gen) * float(enc.std())], -1)
            st["encoder_v"] = torch.cat([env, torch.randn(NH, D, add, generator=gen) * float(env.std())], -1)
        idx = torch.arange(N0, new_width, dtype=torch.float32)
        nf = 1.0 / (2 ** 16 ** ((idx // 2 * 2) / new_width)) / (2 * math.pi)
        st["attn.freqs"] = torch.cat([fq.reshape(-1), nf]).view(1, 1, 1, -1)
    cfg = dict(STATE["_cfg"])
    cfg["mlp_internal_dim_multiplier"] = new_width * NH // D
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    out.close()
    torch.save({"cfg": cfg, "model_state": st}, out.name)
    return out.name


def run_era(phase, M, acq, mult, log):
    global STATE, N0, BS
    lang = PHASE_LANG[phase]
    ck = f"out/bdh_europarl_ladRA2b-{lang}_last.pt"
    if not os.path.exists(ck):
        log(f"  !! missing {ck}; skipping era"); return []
    t0 = time.time()
    CK = torch.load(ck, map_location="cpu", weights_only=False)
    STATE = dict(CK["model_state"]); STATE["_cfg"] = dict(CK["cfg"])
    BS = STATE["_cfg"].get("block_size", 512)
    N0 = STATE["decoder"].shape[0] // NH
    assert STATE["encoder"].shape == (NH, D, N0), f"layout {tuple(STATE['encoder'].shape)} != ({NH},{D},{N0})"
    assert N0 < FINAL_WIDTH, f"era {lang} already at final width"
    log(f"\n=== era p{phase} ({lang}), latent N0={N0} (mult {mult[lang]}) -> {FINAL_WIDTH} "
        f"[+{(FINAL_WIDTH-N0)//BLK} blocks] ===", flush=True)
    doms = sorted({lang, "en"})
    base_ppl, out = {}, []
    xt = {d: crops(d, BS) for d in doms}
    mdl, _ = _load_model(ck); mdl = mdl.to(dev).eval()
    for d in doms:
        base_ppl[d] = ppl(mdl, N0, xt[d], IT)
    del mdl
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    for d in doms:                                             # G1: era reproduces its logged exit
        want, got = acq[lang], base_ppl[d]
        rel = abs(got - want) / want
        tag = "OK" if (d == lang and rel < 0.05) or d != lang else "FAIL"
        log(f"  base[{d}] {got:9.2f}   (era {lang} logged exit {want:.2f})  G1[{d}]={tag if d==lang else 'n/a'}")
        if d == lang and rel >= 0.05:
            log(f"  !! GATE G1 FAILED for {lang}: {rel:.1%} off logged exit -> results not interpretable"); return []
    for arm, zflag in (("A", False), ("C", True)):
        p = build(FINAL_WIDTH, zflag)
        m2, _ = _load_model(p); m2 = m2.to(dev).eval(); os.unlink(p)
        for d in doms:
            v = ppl(m2, FINAL_WIDTH, xt[d], IT)
            mk = ppl(m2, FINAL_WIDTH, xt[d], IT, mask_to=N0) if arm == "A" else None
            out.append({"phase": phase, "era": lang, "arm": arm, "domain": d, "ppl": round(v, 4),
                        "masked_ppl": None if mk is None else round(mk, 4)})
            log(f"  arm {arm} [{d:2s}] ppl {v:11.2f}" + ("" if mk is None else f"   masked@{N0} {mk:9.2f}"))
        del m2; torch.cuda.empty_cache() if dev.type == "cuda" else None
    for r in out:                                              # G2/G3 gates
        if r["arm"] == "C":
            b = base_ppl[r["domain"]]
            if abs(r["ppl"] - b) / b > 5e-3:   # calibrated by the 2026-09-12 pilot: appending zero columns leaves
                                                # the maths unchanged but changes the GEMM shape, and bf16 reduction
                                                # order moves ppl by ~4e-6 relative. A real layout violation moves it
                                                # by factors (arm A here: 24x), so 0.5% still discriminates cleanly.
                log(f"  !! GATE G2 FAILED (era {lang}, {r['domain']}): inert arm moved {r['ppl']/b:.6f}x "
                    "-> column layout suspect; do not interpret arm A")
                r["gate"] = "G2-FAIL"
        if r["arm"] == "A":
            b = base_ppl[r["domain"]]
            if not (0.98 <= r["masked_ppl"] / b <= 1.02):
                log(f"  !! GATE G3 WEAK (era {lang}, {r['domain']}): masked-A {r['masked_ppl']:.2f} vs base {b:.2f}")
                r["gate"] = "G3-WARN"
    for r in out:
        b = base_ppl[r["domain"]]
        real = M.get("lt", {}).get(r["domain"])
        r["base_ppl"] = round(b, 4)
        r["real_joint_ppl"] = real
        if real and r["arm"] == "A":
            den = math.log(real / b)
            if den > math.log(2):
                r["f_log"] = round(math.log(r["ppl"] / b) / den, 4)
                r["f_linear"] = round((r["ppl"] - b) / (real - b), 4)
                r["real_damage_x"] = round(real / b, 3)
            else:
                r["f_log"] = None; r["f_linear"] = None; r["real_damage_x"] = round(real / b, 3)
                r["gate"] = (r.get("gate", "") + " G4-not-defined").strip()
    log(f"  era {lang} done in {time.time()-t0:.0f}s")
    return out


def main():
    import csv
    import json
    os.makedirs(OUTDIR, exist_ok=True)
    M, acq, mult = load_maps()
    lines = []

    def log(msg, flush=False):
        print(msg, flush=True); lines.append(msg)
    log(f"a7 multi-era expansion control | device={dev} iters={IT} batch={BB} target={FINAL_WIDTH}")
    log(f"eras={[(p, PHASE_LANG[p]) for p in ERAS]}")
    rows = []
    for ph in ERAS:
        try:
            rows += run_era(ph, M, acq, mult, log) or []
        except Exception as e:                                   # noqa: BLE001
            log(f"  !! era p{ph} failed: {type(e).__name__}: {e}")
        with open(os.path.join(OUTDIR, "run.log"), "w") as f:
            f.write("\n".join(lines) + "\n")
    if not rows:
        log("no rows produced"); return
    keys = ["phase", "era", "arm", "domain", "ppl", "masked_ppl", "base_ppl", "real_joint_ppl",
            "real_damage_x", "f_log", "f_linear", "gate"]
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in rows]
    fa = [r for r in rows if r["arm"] == "A" and r.get("f_log") is not None and "G2-FAIL" not in str(r.get("gate", ""))]
    fl = [r["f_log"] for r in fa]; ln = [r["f_linear"] for r in fa]
    summ = {"n_eras_run": len({r["era"] for r in rows}), "n_fraction_pairs": len(fa),
            "f_log_min": min(fl) if fl else None, "f_log_median": round(np.median(fl), 4) if fl else None,
            "f_log_max": max(fl) if fl else None,
            "f_linear_min": min(ln) if ln else None, "f_linear_median": round(np.median(ln), 4) if ln else None,
            "f_linear_max": max(ln) if ln else None,
            "prediction_manuscript_f_log": 0.833, "iters": IT, "batch": BB, "target_width": FINAL_WIDTH,
            "protocol_source": "scripts/pi50/r2_expansion_control.py (phase-1 frozen instrument)",
            "per_pair": [{"era": r["era"], "domain": r["domain"], "f_log": r["f_log"],
                          "f_linear": r["f_linear"], "real_damage_x": r["real_damage_x"]} for r in fa]}
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log("\n=== SUMMARY ===")
    for p in summ["per_pair"]:
        log(f"  {p['era']:3s} -> {p['domain']:3s}  real damage {p['real_damage_x']:6.2f}x   f_log {p['f_log']:.3f}   f_lin {p['f_linear']:.3f}")
    if fl:
        log(f"  f_log: median {summ['f_log_median']:.3f} range {summ['f_log_min']:.3f}-{summ['f_log_max']:.3f} "
            f"(manuscript general claim 0.833)")
        log(f"  f_lin: median {summ['f_linear_median']:.3f} range {summ['f_linear_min']:.3f}-{summ['f_linear_max']:.3f}")
    with open(os.path.join(OUTDIR, "run.log"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
