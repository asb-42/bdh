"""P-R3 CHEAP ADDRESSING FROM BYTE GEOMETRY (pre-registered bus #175).

Question (operator #164): can BDH build an address whose cost grows sublinearly in the number of
accumulated territories, instead of running 23 masked forwards to find one? Quinn's X2 (#163) says the
router's geometry follows byte/script statistics, so test whether a trivial input-side model can
reproduce the likelihood router's DECISIONS.

Pipeline:
  1. LABELS: for each of the 20 ladder domains, take NCROPS held-out test crops and score cumulative
     prefix NLL at every territory width (8192..47104). argmin width = self-supervised route label.
     Same instrument A2 validated at 20/20 (#148/#150); no human task IDs involved.
  2. FEATURES: hashed byte 1-4-gram counts (NBUCKETS), L2-normalised.
  3. FIT: multinomial logistic regression by plain SGD (no sklearn dependency), train/test split BY
     CROP so labels are never reused. Reports top-1 agreement, Wilson 95% interval, and the PPL penalty
     of obeying the cheap predictor instead of the scan.

Registered prediction: >=90% agreement among Latin-script domains, much lower for bg/el and cross-script
inputs. Falsifier: overall <60% -> the likelihood computation, not input geometry, carries the address.

Run: .venv/bin/python scripts/pi50/r3_byte_addressing.py
Env: R3_CROPS=16 R3_BATCH=4 R3_EPOCHS=200 R3_LR=0.5 R3_NB=262144
"""
import sys, os, math, time, numpy as np, torch
sys.path.insert(0, ".")
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
LATIN = set("en es pl fr de cs da pt fi hu it sk sv ro nl lt".split())
BLK, NH = 2048, 8
CK = os.environ.get("R3_CKPT", "out/bdh_europarl_ladRA2b-lt_last.pt")
NC = int(os.environ.get("R3_CROPS", "16")); BB = int(os.environ.get("R3_BATCH", "4"))
EPOCHS = int(os.environ.get("R3_EPOCHS", "200")); LR = float(os.environ.get("R3_LR", "0.5"))
NB_BUCKETS = int(os.environ.get("R3_NB", "262144"))
dev = torch.device("cuda")
model, cfg = _load_model(CK); model = model.to(dev).eval(); BS = cfg["block_size"]
N = int(model.decoder.shape[0]) // NH; LPW = N // BLK
WIDTHS = [(j + 1) * BLK for j in range(LPW)]
POS = {l: i for i, l in enumerate(SEQ)}


def feats(crop_bytes):
    """Hashed byte 1-4-gram counts -> L2-normalised dense vector over NB_BUCKETS."""
    v = np.zeros(NB_BUCKETS, dtype=np.float32)
    b = np.frombuffer(crop_bytes.encode("latin-1", "ignore") if isinstance(crop_bytes, str) else crop_bytes, dtype=np.uint8)
    for k in (1, 2, 3, 4):
        if len(b) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(b, k)
        key = np.zeros(len(w), dtype=np.uint64)
        for j in range(k):
            key = key * np.uint64(257) + w[:, j].astype(np.uint64)
        idx = (key % np.uint64(NB_BUCKETS)).astype(np.int64)
        np.add.at(v, idx, 1.0)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def nll_batch(pairs, mask_to=None):
    """Per-crop NLL for a list of (x_tensor, y_tensor). Targets are taken from the byte stream
    shifted by one WITHOUT wrap-around, so the final position of each crop is dropped rather than
    paired with the first byte (an earlier draft used torch.roll here, which wrapped the boundary)."""
    x = torch.stack([p[0] for p in pairs]).to(dev)
    y = torch.stack([p[1] for p in pairs]).to(dev)
    m = None
    if mask_to is not None:
        m = torch.ones(N, device=dev); m[mask_to:] = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lo = (model(x, None, None, neuron_mask=m) if m is not None else model(x))[0]
        e = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1), reduction="none")
    return [float(v) for v in e.view(len(pairs), -1).mean(1)]


print(f"ckpt={CK.split('-')[-1]} widths={len(WIDTHS)} crops/domain={NC} batch={BB}", flush=True)
X, Y, RAW, t0 = [], [], [], time.time()
for li, lang in enumerate(SEQ):
    raw = _europarl_blocks("data", 30_000_000, langs=(lang,))[lang]
    buf = np.frombuffer(raw["test"], dtype=np.uint8)
    d = torch.from_numpy(buf.astype(np.int64))
    g = torch.Generator().manual_seed(5000 + li)
    ix = torch.randint(len(d) - BS - 1, (NC,), generator=g)
    idxs = [int(i) for i in ix]
    pairs_all = [(d[i:i + BS], d[i + 1:i + 1 + BS]) for i in idxs]
    crops = [d[i:i + BS].cpu().numpy().astype(np.uint8).tobytes() for i in idxs]
    assert NC % BB == 0, f"R3_CROPS={NC} must be a multiple of R3_BATCH={BB}"
    scores = np.zeros((NC, len(WIDTHS)))
    for wi, w in enumerate(WIDTHS):
        vals = []
        for c0 in range(0, NC, BB):
            vals += nll_batch(pairs_all[c0:c0 + BB], mask_to=w)
        assert len(vals) == NC, f"got {len(vals)} scores for {NC} crops"
        scores[:, wi] = vals
    for ci, cb in enumerate(crops):
        X.append(feats(cb)); Y.append(int(np.argmin(scores[ci]))); RAW.append(lang)
    print(f"  {lang}: labels {sorted(set(scores.argmin(1).tolist()))} ({time.time()-t0:.0f}s)", flush=True)

X = np.stack(X); Y = np.array(Y); K = len(WIDTHS)
perm = np.random.RandomState(7).permutation(len(Y)); tr, te = perm[: len(Y) * 3 // 4], perm[len(Y) * 3 // 4:]
W = np.zeros((X.shape[1], K), dtype=np.float32); bias = np.zeros(K, dtype=np.float32)
print(f"\nfitting multinomial logistic: {X.shape[1]} features, {K} classes, train={len(tr)} test={len(te)}", flush=True)
for ep in range(EPOCHS):
    o = X[tr] @ W + bias; o -= o.max(1, keepdims=True); p = np.exp(o); p /= p.sum(1, keepdims=True)
    g = p - np.eye(K)[Y[tr]]; W -= LR * (X[tr].T @ g / len(tr) + 1e-4 * W); bias -= LR * g.mean(0)
o = X[te] @ W + bias; pred = o.argmax(1)
acc = float((pred == Y[te]).mean()); n = len(te)
z = 1.96; p_, den = acc, 1 + z * z / n
c = (p_ + z * z / (2 * n)) / den; h = z * math.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / den
print(f"\nAGREEMENT with argmin-NLL router: {acc:.3f}  [{c-h:.3f}, {c+h:.3f}]  (n={n} held-out crops)")
langs_te = [RAW[i] for i in te]
for grp, name in ((LATIN, "Latin-script"), (set(SEQ) - LATIN, "Cyrillic/Greek")):
    sel = [k for k, l in enumerate(langs_te) if l in grp]
    if sel:
        print(f"  {name:16s}: {float((pred[sel] == Y[te][sel]).mean()):.3f} (n={len(sel)})")
conf = {}
for k, l in enumerate(langs_te):
    conf.setdefault(l, []).append(bool(pred[k] == Y[te][k]))
print("\nper-domain agreement:")
print("  " + "  ".join(f"{l}:{np.mean(v):.2f}" for l, v in conf.items()))
print("\nmajority-class baseline:", f"{np.bincount(Y[te], minlength=K).max()/n:.3f}")
print("cost: cheap predictor = 1 counting pass vs 23 masked forwards per candidate.")
