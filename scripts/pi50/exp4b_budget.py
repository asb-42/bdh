"""A2 follow-ups: how much calibration does likelihood selection actually need, and can we
buy back the 23-forward cost? Two arms, one instrumented pass each.

  arm BUDGET : argmin over all cumulative prefixes using CAL bytes of in-domain context
               (4 KB, 16 KB, 64 KB, 256 KB, 1 MB) -> P(correct prefix) vs budget
  arm BSEARCH: same decision by binary search over sorted widths (~log2(23)=5 forwards),
               scored at the same budgets

Sanity gate first (must reproduce matrix: en free ~31, en masked-to-own ~2.3) or abort.
Run: cd /srv/coding/bdh && .venv/bin/python -u scripts/pi50/exp4b_budget.py <ckpt> [iters] [batch]
"""
import sys, math, numpy as np, torch
sys.path.insert(0, ".")
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
POS = {l: i for i, l in enumerate(SEQ)}
BLK, NH = 2048, 8
CK = sys.argv[1] if len(sys.argv) > 1 else "out/bdh_europarl_ladRA2b-lt_last.pt"
IT = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BB = int(sys.argv[3]) if len(sys.argv) > 3 else 2
BUDGETS = [4_000, 16_000, 64_000, 256_000, 1_000_000]

dev = torch.device("cuda")
model, cfg = _load_model(CK)
model = model.to(dev).eval()
bs = cfg["block_size"]
N = int(model.decoder.shape[0]) // NH
LPW = N // BLK
WIDTHS = [(j + 1) * BLK for j in range(LPW)]
BLOCK_OF = {b: ("en" if b <= 3 else SEQ[b - 3]) for b in range(LPW)}
blocks = _europarl_blocks("data", 30_000_000, langs=tuple(SEQ))


def nll(d, width, iters):
    g = torch.Generator().manual_seed(9021); ls = []
    m = None
    if width is not None:
        m = torch.ones(N, device=dev); m[width:] = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            ix = torch.randint(len(d) - bs - 1, (BB,), generator=g)
            x = torch.stack([d[int(i):int(i) + bs] for i in ix]).to(dev)
            y = torch.stack([d[int(i) + 1:int(i) + 1 + bs] for i in ix]).to(dev)
            lo, _, _ = model(x, None, None, neuron_mask=m) if m is not None else model(x)
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item())
    return sum(ls) / len(ls)


def buf(lang, nbytes):
    raw = blocks[lang]["val"] + blocks[lang]["test"]
    d = torch.from_numpy(np.frombuffer(raw[:nbytes], dtype=np.uint8).astype(np.int64))
    return d


# ---- sanity gate against the matrix --------------------------------------
en = buf("en", 1_000_000)
fr = math.exp(nll(en, None, IT)); ma = math.exp(nll(en, 8192, IT))
print(f"GATE en free={fr:.2f} (expect ~31) masked8192={ma:.2f} (expect ~2.3)")
if not (15 < fr < 60 and ma < 4):
    raise SystemExit("GATE FAILED - instrument diverges from the matrix, refusing to report")

lin = {b: 0 for b in BUDGETS}; bse = {b: 0 for b in BUDGETS}; fwd_lin = {b: 0 for b in BUDGETS}; fwd_b = {b: 0 for b in BUDGETS}
for l in SEQ:
    own = (POS[l] + 4) * BLK
    for B in BUDGETS:
        d = buf(l, B)
        if len(d) < bs + 2:
            continue
        sc = [nll(d, w, IT) for w in WIDTHS]                      # linear scan
        pick = WIDTHS[int(np.argmin(sc))]
        lin[B] += pick == own; fwd_lin[B] += len(WIDTHS)
        lo, hi = 0, len(WIDTHS) - 1                                # binary search on monotone-ish curve
        while lo < hi:
            mid = (lo + hi) // 2
            if nll(d, WIDTHS[mid], IT) < nll(d, WIDTHS[mid + 1], IT):
                hi = mid
            else:
                lo = mid + 1
        pick2 = WIDTHS[lo]; bse[B] += pick2 == own; fwd_b[B] += 2 * (lo + 1)
    print(f"  {l} done (own width {own})")

print("\nP(correct prefix | x), 20 domains, by calibration budget")
print("budget_bytes | linear_argmin(/20) | forwards/input | binary_search(/20) | forwards/input")
for b in BUDGETS:
    print(f"  {b:>9d}   |       {lin[b]:>3d}/20          |     {fwd_lin[b]//20:>3d}      |       {bse[b]:>3d}/20         |    ~{max(fwd_b[b]//20,1):>3d}")
print("\nNote: binary search assumes loss decreases then increases with width (unimodal). If that\nfails for some domain, the discrepancy itself is the finding.")
