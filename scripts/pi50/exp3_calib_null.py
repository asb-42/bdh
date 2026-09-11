"""A1 CALIBRATION NULL v2 - k_sparse_ratio=0.0 means NO top-k: the latent readout SUMS over
all N ReLU units, so growing N adds terms to an existing sum even when every old weight is
bit-identical. This measures, per domain, how much each block contributes and whether any
selector can recover the right territory without a language ID.

No repo modification: rebinds model.forward to a logging copy of bdh's source.
Run: cd /srv/coding/bdh && .venv/bin/python scripts/pi50/exp3_calib_null.py.py [crops]
"""
import sys, types, inspect, math
import torch, numpy as np
sys.path.insert(0, ".")
import bdh
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
POS = {l: i for i, l in enumerate(SEQ)}
BLK, NH = 2048, 8
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CKPT = "out/bdh_europarl_ladRA2b-lt_last.pt"

CALLS = []


def _LOG(t):
    r = torch.relu(t)
    CALLS.append(r.detach().float().abs().mean(dim=(0, 2)).cpu())   # (nh, N)
    return r


src = inspect.getsource(bdh)
import re as _re
n_fix = 0
def sub_line(src, var, arg):
    global n_fix
    def rep(m):
        global n_fix
        n_fix += 1
        return f"{m.group(1)}{var} = _LOG({arg})"
    return _re.sub(r"(\s*)%s = .*" % var, rep, src, count=1)
src = sub_line(src, "x_sparse", "x_latent")
src = sub_line(src, "y_sparse", "y_latent")
mod = types.ModuleType("bdh_logged")
mod.__dict__["_LOG"] = _LOG
exec(compile(src, "bdh_logged", "exec"), mod.__dict__)
print(f"instrumented {n_fix} sparse call sites (expect 2)")

dev = torch.device("cuda")
model, cfg = _load_model(CKPT)
model.forward = lambda *a, **k: mod.BDH.forward(model, *a, **k)
model = model.to(dev).eval()
bs = cfg["block_size"]
N = int(model.decoder.shape[0]) // NH
LPW = N // BLK
blocks = _europarl_blocks("data", 30_000_000, langs=tuple(SEQ))
BLOCK_OF = {b: ("en" if b <= 3 else SEQ[b - 3]) for b in range(LPW)}
print(f"N/head={N} blocks/head={LPW}")
def _ppl(lang, mask=None, iters=40):
    raw = blocks[lang]["val"] + blocks[lang]["test"]
    d = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
    g = torch.Generator().manual_seed(1234); ls = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            ix = torch.randint(len(d) - bs - 1, (1,), generator=g)
            x = torch.stack([d[i:i + bs] for i in ix]).to(dev); y = torch.stack([d[i + 1:i + 1 + bs] for i in ix]).to(dev)
            lo, _, _ = model(x, None, None, neuron_mask=mask) if mask is not None else model(x)
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item())
    return math.exp(sum(ls) / len(ls))
_w = torch.ones(N, device=dev); _w[8192:] = 0.0
print(f"FAITHFULNESS  en free={_ppl('en'):.2f} (expect ~31.07) | en masked={_ppl('en',_w):.2f} (expect ~2.31)")


def energy(lang, split="val"):
    raw = blocks[lang][split]
    d = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
    g = torch.Generator().manual_seed(4321)
    acc = torch.zeros(NH, N); n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(NC):
            ix = torch.randint(len(d) - bs - 1, (1,), generator=g)
            x = torch.stack([d[i:i + bs] for i in ix]).to(dev)
            CALLS.clear()
            model(x)
            if not CALLS:
                raise SystemExit("hook never fired - instrumentation failed")
            acc += torch.stack(CALLS).mean(0)     # mean over layers, keep heads
            n += 1
    return acc / n


E = {l: energy(l) for l in SEQ}
torch.save({"E": {l: E[l] for l in SEQ}, "LPW": LPW}, "/tmp/energies.pt")

bm = {l: torch.stack([E[l][:, b * BLK:(b + 1) * BLK].mean(1) for b in range(LPW)]) for l in SEQ}   # (blocks, nh)
bx = {l: torch.stack([E[l][:, b * BLK:(b + 1) * BLK].max(1).values for b in range(LPW)]) for l in SEQ}

print("\nP(correct territory | x), averaged over heads:")
for nm, D in [("block_mean", bm), ("block_max", bx)]:
    ok = 0; det = []
    for l in SEQ:
        v = D[l].mean(1)
        pick = BLOCK_OF[int(v.argmax())]
        ok += pick == l
        det.append(f"{l}->{pick}" if pick != l else f"{l}=ok")
    print(f"  {nm:11s} {ok:>3d}/20   " + " ".join(det[:10]))

print("\ncumulative-prefix statistics (operator's extreme-value concern):")
cm = {l: np.array([float(E[l][:, :(j + 1) * BLK].max()) for j in range(LPW)]) for l in SEQ}
cn = {l: np.array([float(E[l][:, :(j + 1) * BLK].mean()) for j in range(LPW)]) for l in SEQ}
for nm, S in [("prefix_max", cm), ("prefix_mean", cn)]:
    ok = sum(1 for l in SEQ if BLOCK_OF[int(np.argmax(S[l]))] == l)
    print(f"  {nm:11s} argmax-correct {ok}/20")
bias = np.mean([[int((cm[l][POS[o] + 3] > cm[l][POS[l] + 3]).sum()) for o in SEQ] for l in SEQ], axis=0)
print("  mean # of later prefixes beating en's own max, by later phase:", " ".join(f"{v:.1f}" for v in bias[:8]), "...")

print("\nper-domain block profile (own block energy vs global winner):")
print(" true | own_blk | own_mean | winner_blk(winner_lang) | ratio winner/own")
for l in SEQ:
    v = bm[l].mean(1); own = POS[l] + 3 if l != "en" else 3
    w = int(v.argmax())
    print(f"  {l}  |   {own:2d}    | {v[own]:7.4f} |   {w:2d} ({BLOCK_OF[w]:>2s})      | {v[w]/max(v[own],1e-9):5.2f}")
