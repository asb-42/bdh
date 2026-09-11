"""P-R1 READOUT WIDTH-INVARIANCE (pre-registered bus #165).

Mechanism under test: cfg k_sparse_ratio=0.0, so bdh.py:251 takes the `else` branch -> plain
F.relu over ALL N candidates, and the readout y = x_sparse @ decoder SUMS over all of them.
Growing latent width therefore adds positive terms to an existing sum while old weights stay
bit-identical (Exp-3). P-R2 (#166 thread) shows one RANDOM appended block already costs English
2.33 -> 9.57, so the effect is arithmetic, not learned competition.

Operators applied at inference time to the shipped RA2b-lt_last checkpoint (no training):
  id      identity control: must reproduce the matrix free-width numbers (en ~31.07) or stop.
  absK    top-K per head with K ABSOLUTE (default 2048), independent of width. Implemented by
          patching bdh._k_sparse_relu and forcing the sparse branch; this is exactly the
          'absoluteK' policy pi-33 probed at operator level (scripts/probe_selection_fix_operators.py,
          commit 6754e83) but measured end-to-end on real weights and real language data.
  evshift extreme-value shift: relu(v - sqrt(2 ln N_b) * sigma_b) per 2048-neuron block, which is
          the F-V5 correction predicted on 2026-08-31 and never tested.
  blkavg  block-average readout: scale each block's decoder contribution by 1/n_blocks, so growth
          adds equal-weight averages instead of raw sums.
  lognorm scale the whole readout by ln(N0)/ln(N) (crude magnitude normalisation).

PASS CONDITION (pre-registered): an operator "works" if it brings free-width PPL of every damaged
language within 2x of its acquisition exit WITHOUT any external mask. Partial credit reported as
recovery fraction (log-scale) so a monotone-but-insufficient fix is visible as such.

Run: .venv/bin/python scripts/pi50/r1_readout_operators.py [ckpt]
Env: R1_LANGS=en,de,cs,bg,el  R1_ITERS=25  R1_BATCH=4  R1_OPS=id,absK,evshift,blkavg,lognorm
"""
import sys, os, math, csv, numpy as np, torch
sys.path.insert(0, ".")
import bdh as bdh_mod
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
BLK, NH = 2048, 8
CK = sys.argv[1] if len(sys.argv) > 1 else "out/bdh_europarl_ladRA2b-lt_last.pt"
LANGS = os.environ.get("R1_LANGS", "en,de,cs,bg,el").split(",")
IT = int(os.environ.get("R1_ITERS", "25")); BB = int(os.environ.get("R1_BATCH", "4"))
OPS = os.environ.get("R1_OPS", "id,absK,evshift,blkavg,lognorm").split(",")
ABS_K = int(os.environ.get("R1_ABSK", "2048"))
dev = torch.device("cuda")
model, cfg = _load_model(CK); model = model.to(dev).eval()
N = int(model.decoder.shape[0]) // NH; NB = N // BLK
BS = cfg["block_size"]
ORIG_RELU = bdh_mod._k_sparse_relu
ORIG_RATIO = cfg["k_sparse_ratio"]


def crops(lang):
    raw = _europarl_blocks("data", 30_000_000, langs=(lang,))[lang]
    d = torch.from_numpy(np.frombuffer(raw["val"] + raw["test"], dtype=np.uint8).astype(np.int64))
    return d[(len(d) - BS - 1) // 2:]


def ppl(dd, iters):
    g = torch.Generator().manual_seed(31337); ls = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            ix = torch.randint(len(dd) - BS - 1, (BB,), generator=g)
            x = torch.stack([dd[int(i):int(i) + BS] for i in ix]).to(dev)
            y = torch.stack([dd[int(i) + 1:int(i) + 1 + BS] for i in ix]).to(dev)
            lo = model(x)[0]
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item())
    return math.exp(sum(ls) / len(ls))


def absK(x_pos, ratio):
    k = min(ABS_K, x_pos.shape[-1])
    vals, idx = torch.topk(x_pos, k, dim=-1)
    m = torch.zeros_like(x_pos).scatter_(-1, idx, 1.0)
    return x_pos * m


def evshift(x_pos, ratio):
    v = x_pos.view(*x_pos.shape[:-1], NB, BLK)
    mu = v.mean(-1, keepdim=True); sd = v.std(-1, keepdim=True)
    thr = mu + sd * math.sqrt(2.0 * math.log(BLK))
    return torch.relu(v - thr).view_as(x_pos)


M = {}
for r in csv.DictReader(open("docs/reports/data/2026-09-10_ra2b_matrix.csv")):
    M.setdefault(r["checkpoint"], {})[r["eval_lang"]] = float(r["ppl"])
print(f"ckpt={CK.split('-')[-1]} N={N} blocks={NB} langs={LANGS} ops={OPS} absK={ABS_K}", flush=True)
print("\nlang | exit | free(id) | " + " | ".join(f"{o:>10s}" for o in OPS if o != "id") + " | recovery(absK,evshift,blkavg,lognorm)")

dec0 = model.decoder.detach().clone()
res = {}
for op in OPS:
    bdh_mod._k_sparse_relu = ORIG_RELU
    model.decoder.data.copy_(dec0)
    cfg_local = dict(cfg)
    if op == "id":
        pass
    elif op in ("absK", "evshift"):
        bdh_mod._k_sparse_relu = absK if op == "absK" else evshift
        model.config.k_sparse_ratio = 1e-9   # force the sparse branch; patched fn ignores ratio
    elif op == "blkavg":
        # decoder rows are HEAD-MAJOR (nh, N, D) - see scripts/verify_masked_forward.py:110 and
        # Exp-3's row-slice correction. Slicing it as (NB, BLK, -1) would mix heads.
        with torch.no_grad():
            model.decoder.data = (dec0.view(NH, NB, BLK, -1) / NB).reshape(-1, dec0.shape[-1])
    elif op == "lognorm":
        with torch.no_grad():
            model.decoder.data = dec0 * (math.log(8192) / math.log(N))
    for lang in LANGS:
        res[(op, lang)] = ppl(crops(lang), IT)
    print(f" {op:>8s} | " + " ".join(f"{res[(op,l)]:10.2f}" for l in LANGS), flush=True)
    bdh_mod._k_sparse_relu = ORIG_RELU; model.decoder.data.copy_(dec0); model.config.k_sparse_ratio = ORIG_RATIO

print("\nlang | exit | free_id | " + " | ".join(f"{o}>recov" for o in OPS if o != "id"))
for lang in LANGS:
    ex, fid = M[lang][lang], res[("id", lang)]
    line = f" {lang:>3s} | {ex:5.2f} | {fid:7.2f} | "
    for op in [o for o in OPS if o != "id"]:
        r = res[(op, lang)]
        rec = (math.log(fid) - math.log(r)) / (math.log(fid) - math.log(ex)) if fid > ex else float("nan")
        line += f"{op}:{r:7.2f}({rec:+.2f}) "
    print(line, flush=True)
print("\nrecovery=1.00 means the operator fully restores the acquisition exit without any mask.")
