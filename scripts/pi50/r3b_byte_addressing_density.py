"""P-R3 follow-ups F-1 / F-2 / F-3 (Quinn's task_request #186, accepted with corrections in #187).

One run, three deliverables, because F-3 is NOT free: the original r3_byte_addressing.py never
persisted features or weights, so any post-hoc analysis costs a full label regeneration. This version
persists everything sparse (features as idx/val lists, ~MBs not hundreds of MB), so later questions
cost zero GPU.

  F-1  density for the four failure domains (es pl sk et) at 96 crops each (64 train / 32 test) while
       the other 16 control domains stay at 16 crops. Reports agreement AS A FUNCTION of training
       density (8/16/32/64), not just the endpoint - if the curve is still climbing steeply at 64 then
       "sample-limited" and "geometry-limited" are not yet separated and I must say so.
  F-3  disagreement confusion table: which territory the cheap predictor answers with, mapped to the
       language that owns it (family structure vs arbitrary).
  F-2  unseen languages pushed through the FITTED predictor without retraining (zh, iu filtered to
       syllabic-only lines per #188, lv, ga), compared against Quinn's likelihood-router reference
       counts parsed from his routdiag logs. Measures consistency between two addressers, not
       correctness of either - unseen languages have no ground-truth route label.

Run: .venv/bin/python scripts/pi50/r3b_byte_addressing_density.py
Env: R3B_OUT=~/bdh-review/reports/r3b  R3B_OOD_DIR=~/bdh-review/data/ood  R3B_NB=262144
     R3B_DENSE=es,pl,sk,et  R3B_DENSE_CROPS=96  R3B_CTRL_CROPS=16  R3B_BATCH=4  R3B_EPOCHS=200 R3B_LR=0.5
"""
import sys, os, re, json, math, time, subprocess, numpy as np, torch
sys.path.insert(0, ".")
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
BLK, NH = 2048, 8
CK = os.environ.get("R3B_CKPT", "out/bdh_europarl_ladRA2b-lt_last.pt")
DENSE = os.environ.get("R3B_DENSE", "es,pl,sk,et").split(",")
N_DENSE = int(os.environ.get("R3B_DENSE_CROPS", "96")); N_CTRL = int(os.environ.get("R3B_CTRL_CROPS", "16"))
BB = int(os.environ.get("R3B_BATCH", "4")); EPOCHS = int(os.environ.get("R3B_EPOCHS", "200"))
LR = float(os.environ.get("R3B_LR", "0.5")); NB_BUCKETS = int(os.environ.get("R3B_NB", "262144"))
OUT = os.path.expanduser(os.environ.get("R3B_OUT", "~/bdh-review/reports/r3b"))
OOD_DIR = os.path.expanduser(os.environ.get("R3B_OOD_DIR", "~/bdh-review/data/ood"))
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
model, cfg = _load_model(CK); model = model.to(dev).eval(); BS = cfg["block_size"]
N = int(model.decoder.shape[0]) // NH; LPW = N // BLK
WIDTHS = [(j + 1) * BLK for j in range(LPW)]


def lang_of_territory(ti):
    return "en(base)" if ti <= 3 else (SEQ[ti - 4] if 4 <= ti < 4 + len(SEQ) else f"?{ti}")


def feats(b):
    v = np.zeros(NB_BUCKETS, dtype=np.float32)
    a = np.frombuffer(b, dtype=np.uint8)
    for k in (1, 2, 3, 4):
        if len(a) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, k)
        key = np.zeros(len(w), dtype=np.uint64)
        for j in range(k):
            key = key * np.uint64(257) + w[:, j].astype(np.uint64)
        np.add.at(v, (key % np.uint64(NB_BUCKETS)).astype(np.int64), 1.0)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def nll_batch(pairs, mask_to):
    """Per-crop mean NLL under a prefix mask. Targets shifted WITHOUT wrap-around."""
    x = torch.stack([p[0] for p in pairs]).to(dev)
    y = torch.stack([p[1] for p in pairs]).to(dev)
    m = torch.ones(N, device=dev); m[mask_to:] = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lo = model(x, None, None, neuron_mask=m)[0]
        e = torch.nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1), reduction="none")
    return [float(v) for v in e.view(len(pairs), -1).mean(1)]


# ---------------------------------------------------------------- labels
X, Y, LANG, t0 = [], [], [], time.time()
for li, lang in enumerate(SEQ):
    nc = N_DENSE if lang in DENSE else N_CTRL
    raw = _europarl_blocks("data", 30_000_000, langs=(lang,))[lang]
    d = torch.from_numpy(np.frombuffer(raw["test"], dtype=np.uint8).astype(np.int64))
    g = torch.Generator().manual_seed(5000 + li)
    idxs = [int(i) for i in torch.randint(len(d) - BS - 1, (nc,), generator=g)]
    assert all(i + 1 + BS <= len(d) for i in idxs), "crop runs past buffer"
    pairs_all = [(d[i:i + BS], d[i + 1:i + 1 + BS]) for i in idxs]
    scores = np.zeros((nc, len(WIDTHS)))
    for wi, w in enumerate(WIDTHS):
        vals = []
        for c0 in range(0, nc, BB):
            vals += nll_batch(pairs_all[c0:c0 + BB], w)
        assert len(vals) == nc, f"{lang}: {len(vals)} scores for {nc} crops"
        scores[:, wi] = vals
    lab = scores.argmin(1)
    for ci, i in enumerate(idxs):
        X.append(feats(d[i:i + BS].cpu().numpy().astype(np.uint8).tobytes()))
        Y.append(int(lab[ci])); LANG.append(lang)
    uniq = sorted(set(lab.tolist()))
    print(f"  {lang:>3s} n={nc:3d} label set {uniq} ({lang_of_territory(uniq[0])}) "
          f"own={(len(uniq)==1 and uniq[0]==SEQ.index(lang)+4) or (lang=='en' and uniq==[3])} "
          f"[{time.time()-t0:.0f}s]", flush=True)

X = np.stack(X).astype(np.float32); Y = np.array(Y); L = np.array(LANG)
K = len(WIDTHS)
np.savez_compressed(f"{OUT}/features_sparse.npz",
                    indptr=np.concatenate([[0], np.cumsum((X != 0).sum(1))]),
                    **{"Y": Y, "LANG": L.astype("U4"), "WIDTHS": np.array(WIDTHS)},
                    indices=X.nonzero()[1].astype(np.int32), values=X[X != 0].astype(np.float16))
print(f"\npersisted {len(Y)} labelled crops -> {OUT}/features_sparse.npz", flush=True)

# ---------------------------------------------------------------- fit helpers
rng = np.random.RandomState(7)


def fit_predict(tr_idx, te_idx, epochs=EPOCHS, lr=LR):
    W = np.zeros((X.shape[1], K), dtype=np.float32); b = np.zeros(K, dtype=np.float32)
    Xt, yt = X[tr_idx], Y[tr_idx]
    for _ in range(epochs):
        o = Xt @ W + b; o -= o.max(1, keepdims=True)
        p = np.exp(o); p /= p.sum(1, keepdims=True)
        G = p - np.eye(K)[yt]
        W -= lr * (Xt.T @ G / len(tr_idx) + 1e-4 * W); b -= lr * G.mean(0)
    oo = X[te_idx] @ W + b
    return oo.argmax(1), W, b


def wilson(acc, n):
    z = 1.96; den = 1 + z * z / n
    c = (acc + z * z / (2 * n)) / den
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# whole-corpus split for the headline + confusion
perm = rng.permutation(len(Y)); cut = len(Y) * 3 // 4
tr, te = perm[:cut], perm[cut:]
pred, W, b = fit_predict(tr, te)
acc = float((pred == Y[te]).mean()); lo, hi = wilson(acc, len(te))
print(f"\n=== HEADLINE (all domains, split by crop) ===")
print(f"agreement {acc:.3f} [{lo:.3f},{hi:.3f}] n={len(te)}   majority baseline "
      f"{np.bincount(Y[te], minlength=K).max()/len(te):.3f}")
print("\n=== F-3 DISAGREEMENT CONFUSION (true language -> predicted territory owner) ===")
conf = {}
for k, i in enumerate(te):
    if pred[k] != Y[i]:
        conf.setdefault(L[i], {}).update({lang_of_territory(int(pred[k])): conf.setdefault(L[i], {}).get(lang_of_territory(int(pred[k])), 0) + 1})
for l in sorted(conf):
    tot = sum(conf[l].values())
    print(f"  {l:>3s} ({tot} misses): " + ", ".join(f"{a}->{bb} x{cc}" for a, bb, cc in
          sorted([(l, t, c) for t, c in conf[l].items()], key=lambda z: -z[2])))

# ---------------------------------------------------------------- F-1 density curves
print("\n=== F-1 DENSITY CURVES for " + ",".join(DENSE) + " (train crops -> held-out agreement) ===")
sizes = [8, 16, 32, 64]
for dl in DENSE:
    ids = np.where(L == dl)[0]
    if len(ids) < 96:
        print(f"  {dl}: only {len(ids)} crops, skipped"); continue
    rs = np.random.RandomState(11); sp = rs.permutation(ids)
    trc, tec = sp[:64], sp[64:96]
    line = []
    for s in sizes:
        p2, _, _ = fit_predict(trc[:s], tec)
        line.append(f"{s}:{float((p2 == Y[tec]).mean()):.2f}")
    p3, _, _ = fit_predict(trc, tec)
    a3 = float((p3 == Y[tec]).mean()); l3, h3 = wilson(a3, len(tec))
    print(f"  {dl:>3s} " + " ".join(line) + f" | full64 {a3:.2f} [{l3:.2f},{h3:.2f}]")

# ---------------------------------------------------------------- F-2 unseen rows
print("\n=== F-2 UNSEEN LANGUAGES THROUGH THE FITTED PREDICTOR (no retraining) ===")
OOD_FILES = {"zh": ["xscript_zh_head.txt"], "iu_syllabic_only": ["xscript_iu.txt"],
             "lv": ["lv_head.txt"], "ga": ["ga_head.txt"]}
rows = {}
for name, files in OOD_FILES.items():
    blobs = []
    for fn in files:
        path = os.path.join(OOD_DIR, fn)
        if not os.path.exists(path):
            print(f"  {name}: missing {fn}"); continue
        txt = open(path, "rb").read()
        if name == "iu_syllabic_only":
            keep = [l for l in txt.split(b"\n") if re.search(r"[\u1400-\u167f]", l.decode("utf8", "ignore"))]
            txt = b"\n".join(keep)
        blobs.append(txt)
    blob = b"\n".join(blobs)
    if len(blob) < 40 * BS:
        print(f"  {name}: too small ({len(blob)} B)"); continue
    gs = np.random.RandomState(99)
    starts = gs.randint(0, len(blob) - BS - 1, 40)
    ps = np.array([feats(blob[int(i):int(i) + BS]) for i in starts], dtype=np.float32)
    pr = (ps @ W + b).argmax(1)
    hist = {}
    for p in pr:
        hist[lang_of_territory(int(p))] = hist.get(lang_of_territory(int(p)), 0) + 1
    rows[name] = hist
    top = sorted(hist.items(), key=lambda z: -z[1])[:5]
    print(f"  {name:>18s} 40 crops -> " + ", ".join(f"{k} x{v}" for k, v in top))
json.dump({"headline_agreement": acc, "wilson": [lo, hi], "n_test": int(len(te)),
           "ood_route_hist": rows, "confusion": {k: v for k, v in conf.items()},
           "labels_own_prefix": {str(l): sorted(set(Y[L == l].tolist())) for l in set(L.tolist())}},
          open(f"{OUT}/r3b_summary.json", "w"), indent=1)
print(f"\nwrote {OUT}/r3b_summary.json ; total {time.time()-t0:.0f}s")
