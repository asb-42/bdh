"""P-R1b RELATIVE TERRITORY REWEIGHTING (pre-registered bus #175).

Why these arms: bdh.py:279 applies self.ln to the readout output, so global gain changes are
annihilated (measured in P-R1: blkavg/lognorm reproduced identity to 0.1%). Only RELATIVE weighting
between the 23 territories can have an effect. P-R1 also showed that CULLING candidates (absolute-K
top-k, extreme-value shift) makes damage worse, so this family reweights without removing terms.

Arms, all inference-only on ladRA2b-lt_last:
  id        control, must reproduce the matrix free row (en ~31.1).
  massnorm  scale each territory's ReLU mass to the mean territory mass at that head/position.
  softmix   NB * softmax(tau * log mass) across territories, tau in {0.5, 1.0}.
  calibgain one scalar per territory fitted by gradient descent on ENGLISH NLL over a calibration
            slice disjoint from test; then every language evaluated unchanged. Pre-registered
            reading: if gains fitted on English alone rescue de/cs/bg/el too, a shared width
            compensation exists; if only English recovers, compensation is per-territory.

Run: .venv/bin/python scripts/pi50/r1b_relative_reweight.py [ckpt]
Env: R1B_LANGS=en,de,cs,bg,el  R1B_ITERS=25  R1B_BATCH=3  R1B_ARMS=id,massnorm,softmix0.5,softmix1.0,calibgain
"""
import sys, os, math, csv, numpy as np, torch
sys.path.insert(0, ".")
import bdh as bdh_mod
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

BLK, NH = 2048, 8
CK = sys.argv[1] if len(sys.argv) > 1 else "out/bdh_europarl_ladRA2b-lt_last.pt"
LANGS = os.environ.get("R1B_LANGS", "en,de,cs,bg,el").split(",")
IT = int(os.environ.get("R1B_ITERS", "25")); BB = int(os.environ.get("R1B_BATCH", "3"))
ARMS = os.environ.get("R1B_ARMS", "id,massnorm,softmix0.5,softmix1.0,calibgain").split(",")
CALIB_BYTES = 64_000; FIT_STEPS = int(os.environ.get("R1B_FIT", "60"))
dev = torch.device("cuda")
model, cfg = _load_model(CK); model = model.to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
N = int(model.decoder.shape[0]) // NH; NB = N // BLK
BS = cfg["block_size"]
ORIG = bdh_mod._k_sparse_relu
ALPHA = None          # optional per-territory tensor used by the patched op


def make_op(kind):
    def op(x_latent, ratio):
        v = torch.relu(x_latent)
        vb = v.view(*v.shape[:-1], NB, BLK)
        mass = vb.sum(-1, keepdim=True)
        if kind == "massnorm":
            tgt = mass.mean(dim=-2, keepdim=True)
            w = tgt / (mass + 1e-6)
        elif kind.startswith("softmix"):
            tau = float(kind[len("softmix"):])
            w = NB * torch.softmax(tau * torch.log(mass + 1e-6), dim=-2)
        else:
            w = torch.ones_like(mass)
        if ALPHA is not None:
            w = w * ALPHA.view(1, 1, 1, NB)
        return (vb * w).view_as(v)
    return op


DATA = {}
for lang in LANGS:
    raw = _europarl_blocks("data", 30_000_000, langs=(lang,))[lang]
    buf = np.frombuffer(raw["val"] + raw["test"], dtype=np.uint8).astype(np.int64)
    DATA[lang] = torch.from_numpy(buf)


def batch(d, gen, n=BB):
    ix = torch.randint(len(d) - BS - 1, (n,), generator=gen)
    x = torch.stack([d[int(i):int(i) + BS] for i in ix]).to(dev)
    y = torch.stack([d[int(i) + 1:int(i) + 1 + BS] for i in ix]).to(dev)
    return x, y


def ppl(dd, iters, seed=31337):
    g = torch.Generator().manual_seed(seed); ls = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            x, y = batch(dd, g)
            lo = model(x)[0]
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item())
    return math.exp(sum(ls) / len(ls))


M = {}
for r in csv.DictReader(open("docs/reports/data/2026-09-10_ra2b_matrix.csv")):
    M.setdefault(r["checkpoint"], {})[r["eval_lang"]] = float(r["ppl"])
print(f"ckpt={CK.split('-')[-1]} N={N} territories={NB} langs={LANGS} arms={ARMS}", flush=True)
res = {}
gf = DATA["en"][len(DATA["en"]) // 2:]
calib = gf[: CALIB_BYTES // 1]     # byte-level slice; disjoint halves used below for test
test_en = DATA["en"][len(DATA["en"]) // 2 + CALIB_BYTES:]

for arm in ARMS:
    ALPHA = None
    if arm != "id":
        bdh_mod._k_sparse_relu = make_op(arm if arm.startswith(("massnorm", "softmix")) else "identity")
        model.config.k_sparse_ratio = 1e-9
    if arm == "calibgain":
        ALPHA = torch.ones(NB, device=dev, requires_grad=True)
        bdh_mod._k_sparse_relu = make_op("identity")
        model.config.k_sparse_ratio = 1e-9
        opt = torch.optim.Adam([ALPHA], lr=0.05)
        g = torch.Generator().manual_seed(11)
        for step in range(FIT_STEPS):
            x, y = batch(calib, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(x)[0]
                loss = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                ALPHA.detach().clamp_(min=0.02)
            if step % 20 == 0:
                print(f"   fit step {step:3d} loss {loss.item():.3f} alpha[min/med/max] "
                      f"{ALPHA.min().item():.3f}/{ALPHA.median().item():.3f}/{ALPHA.max().item():.3f}", flush=True)
    for lang in LANGS:
        dd = DATA[lang][len(DATA[lang]) // 2 + CALIB_BYTES:]
        res[(arm, lang)] = ppl(dd, IT)
    print(f" {arm:>14s} | " + " ".join(f"{res[(arm,l)]:9.2f}" for l in LANGS), flush=True)
    if arm == "calibgain":
        a = ALPHA.detach().cpu().numpy()
        print(f"   fitted alphas: {' '.join(f'{v:.2f}' for v in a)}", flush=True)
        print(f"   base territory (0-3) mean {a[:4].mean():.3f}, appended (4-22) mean {a[4:].mean():.3f}", flush=True)
    ALPHA = None
    bdh_mod._k_sparse_relu = ORIG
    model.config.k_sparse_ratio = cfg["k_sparse_ratio"]

ok = 25 < res[("id", "en")] < 40
print(f"\n[GATE] identity en={res[('id','en')]:.2f} (matrix 31.07) -> {'OK' if ok else 'FAIL'}")
if not ok:
    sys.exit("identity control failed; nothing above is trustworthy")
print("\narm | " + " | ".join(f"{l} recovery" for l in LANGS) + " | mean")
for arm in [a for a in ARMS if a != "id"]:
    recs = []
    for l in LANGS:
        ex, fid, r = M[l][l], res[("id", l)], res[(arm, l)]
        recs.append((math.log(fid) - math.log(r)) / (math.log(fid) - math.log(ex)))
    print(f" {arm:>14s} | " + " | ".join(f"{v:+.2f}   " for v in recs) + f" | {np.mean(recs):+.2f}")
print("\nrecovery 1.00 = full restoration of the acquisition exit with no external mask; negative = worse than doing nothing.")
