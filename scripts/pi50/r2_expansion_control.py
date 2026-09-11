"""P-R2 RANDOM-EXPANSION CONTROL (pre-registered bus #165).

Question (operator #164): does old-language degradation require a LEARNED new block, or is it pure
arithmetic - more candidates summed into an existing readout, so the winner set moves?

Arms applied to the RA2b English-era checkpoint (latent 8192 = 4 blocks x 2048, N = mult*512/8):
  A random expansion: append whole 2048-wide blocks whose parameters are iid Gaussian at the
    empirical scale of the existing weights. NOT the pipeline's exact init scheme (stated
    limitation); if A damages, re-run with true fresh-BDH init before concluding.
  B learned expansion: the real ladder checkpoints (read from the RA2b serving matrix).
  C inert expansion: append blocks of exact zeros. With k_sparse_ratio=0.0 the readout sums over
    all candidates, so a zero block contributes nothing and free PPL must stay EXACTLY at the
    un-grown value. C doubles as a LAYOUT VALIDATOR: any movement means new columns were written
    into old territory (decoder rows are head-major h*N+n - the trap documented at
    scripts/verify_masked_forward.py:110).

Checkpoint layout verified on out/bdh_europarl_ladRA2b-en_last.pt:
  decoder (nh*N, D)=(65536,512) | encoder,encoder_v (nh,D,N)=(8,512,8192) | attn.freqs (1,1,1,N)
  cfg['mlp_internal_dim_multiplier'] = N*nh/D = N//64  -> must be rewritten when growing.

Run: .venv/bin/python scripts/pi50/r2_expansion_control.py
Env: R2_LANGS=en,de,cs,bg  R2_ITERS=25  R2_BATCH=8  BASE_CKPT=out/bdh_europarl_ladRA2b-en_last.pt
"""
import sys, os, math, csv, tempfile, numpy as np, torch
sys.path.insert(0, ".")
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

BLK, NH, D = 2048, 8, 512
BASE = os.environ.get("BASE_CKPT", "out/bdh_europarl_ladRA2b-en_last.pt")
MATRIX = "docs/reports/data/2026-09-10_ra2b_matrix.csv"
LANGS = os.environ.get("R2_LANGS", "en,de,cs,bg").split(",")
IT = int(os.environ.get("R2_ITERS", "25"))
BB = int(os.environ.get("R2_BATCH", "8"))
WIDTHS = [int(w) for w in os.environ.get("R2_WIDTHS", "").split(",") if w] or \
         [8192, 10240, 16384, 24576, 33792, 47104]
dev = torch.device("cuda")

CK = torch.load(BASE, map_location="cpu", weights_only=False)
STATE, CFG = CK["model_state"], dict(CK["cfg"])
BS = CFG.get("block_size", 512)
N0 = STATE["decoder"].shape[0] // NH
assert STATE["encoder"].shape == (NH, D, N0), f"unexpected encoder layout {tuple(STATE['encoder'].shape)}"


def crops(lang):
    raw = _europarl_blocks("data", 30_000_000, langs=(lang,))[lang]
    d = torch.from_numpy(np.frombuffer(raw["val"] + raw["test"], dtype=np.uint8).astype(np.int64))
    half = (len(d) - BS - 1) // 2
    return d[:half], d[half:]


def ppl(model, N, dd, iters, mask_to=None):
    m = None
    if mask_to is not None:
        m = torch.ones(N, device=dev); m[mask_to:] = 0.0
    g = torch.Generator().manual_seed(4242); ls = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            ix = torch.randint(len(dd) - BS - 1, (BB,), generator=g)
            x = torch.stack([dd[int(i):int(i) + BS] for i in ix]).to(dev)
            y = torch.stack([dd[int(i) + 1:int(i) + 1 + BS] for i in ix]).to(dev)
            lo = (model(x, None, None, neuron_mask=m) if m is not None else model(x))[0]
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item())
    return math.exp(sum(ls) / len(ls))


def build(new_width, zeros):
    """Return a checkpoint dict grown to new_width, old bytes copied verbatim."""
    add = new_width - N0
    assert add >= 0 and add % BLK == 0, f"add={add} must be whole {BLK}-neuron blocks"
    dec, enc, env, fq = STATE["decoder"], STATE["encoder"], STATE["encoder_v"], STATE["attn.freqs"]
    st = {"embed.weight": STATE["embed.weight"], "lm_head": STATE["lm_head"]}
    if add == 0:
        # baseline: copy verbatim, do NOT extend freqs (an earlier version appended a block here
        # and produced a 10240-wide buffer against an 8192 model)
        st.update(decoder=dec, encoder=enc, encoder_v=env)
        st["attn.freqs"] = fq
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".pt"); out.close()
        torch.save({"cfg": dict(CFG), "step": CK.get("step"), "best_val": CK.get("best_val"),
                    "model_state": st}, out.name)
        return out.name
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
    cfg = dict(CFG); cfg["mlp_internal_dim_multiplier"] = new_width * NH // D
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pt"); out.close()
    torch.save({"cfg": cfg, "step": CK.get("step"), "best_val": CK.get("best_val"),
                "model_state": st}, out.name)
    return out.name


M = {}
for r in csv.DictReader(open(MATRIX)):
    M.setdefault(r["checkpoint"], {})[r["eval_lang"]] = float(r["ppl"])

print(f"base={BASE} N0={N0} widths={WIDTHS} langs={LANGS} iters={IT} batch={BB}", flush=True)
print("\nlang | width | A random free | C inert free | C==A check | A masked@8192 | B real-ladder free")
for lang in LANGS:
    _, xt = crops(lang)
    ref_free = ref_mask = None
    for w in WIDTHS:
        res = {}
        for arm, zflag in (("A", False), ("C", True)):
            p = build(w, zflag)
            mdl, _ = _load_model(p); os.unlink(p)
            mdl = mdl.to(dev).eval()
            res[arm] = ppl(mdl, w, xt, IT)
            if w > N0:
                res[arm + "_mask"] = ppl(mdl, w, xt, IT, mask_to=N0)
            del mdl; torch.cuda.empty_cache()
        if w == N0:
            ref_free, ok = res["A"], 2.0 < res["A"] < 2.7
            print(f" [{lang}] {w:6d} | {ref_free:12.2f} | {res['C']:11.2f} | baseline      |             - | {M['en'][lang]:.2f}")
            print(f" [GATE {lang}] un-grown free ppl={ref_free:.2f} vs matrix exit {M['en'][lang]:.2f} -> {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                sys.exit(f"gate failed for {lang}: harness does not reproduce the matrix")
            continue
        inert = abs(res["C"] - ref_free) / ref_free < 1e-3
        print(f" [{lang}] {w:6d} | {res['A']:12.2f} | {res['C']:11.2f} | "
              f"{'inert OK' if inert else '*** LAYOUT BUG: C moved ***':>26s} | {res['A_mask']:11.2f} | {M['lt'][lang]:.2f}", flush=True)
