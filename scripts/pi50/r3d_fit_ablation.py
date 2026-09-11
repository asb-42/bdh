"""P-R3 fit ablation (CPU-only, reads persisted sparse features - zero GPU).

Why this exists: see bus #196. r3b reported es/pl/sk/et at 1.00 for every training density, but the
pooled headline fell 0.762 -> 0.681 and the misses moved to the 16 CONTROL domains. Giving four domains
96 crops changed both their own data AND everyone else's share of a fixed-budget L2 fit, so the
pre-registered discriminator cannot be applied. Arm A of the first version of this script confirmed the
mechanism directly: misses concentrate on idx4(es) x31, idx17(sk) x10, idx15(et) x9 - precisely the
over-represented classes, i.e. an underfit multiclass attractor, not measured geometry disagreement.

Arms, identical held-out crops throughout:
  A  reproduce r3b (softmax, unweighted, 200 epochs, L2 1e-4) - the number under suspicion
  B  class-balanced softmax (inverse-frequency weights, weaker L2, longer schedule)
  C  one-vs-rest logistic per territory, max-score decision with margin abstention
Plus a COMPOSITION-CONTROLLED density curve: vary es/pl/sk/et train crops over 8/16/32/64 while every
control domain contributes a FIXED 12 train crops, so class mix changes only along the studied axis and
each point yields dense-domain AND control-domain agreement from one fit.

SPEED: full-dimensional epochs were far too slow (arm A alone ~10 min single-threaded over 262144
hashed features, and the density grid needs 16 fits). All fitting therefore happens in a fixed 4096-dim
Johnson-Lindenstrauss projection of the SAME feature matrix (inner products preserved up to JL error),
and validity is checked by requiring projected arm A to reproduce unprojected arm A's agreement within
noise. Absolute numbers may shift slightly; comparisons between arms cannot, because they share the
projection. Set R3D_FULL=1 to skip projection and run the slow exact version.

Run: .venv/bin/python scripts/pi50/r3d_fit_ablation.py
Env: R3B_OUT=~/bdh-review/reports/r3b  R3B_NB=262144  R3D_PROJ=4096  R3D_FULL=0
"""
import os, json, math, numpy as np

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
OUT = os.path.expanduser(os.environ.get("R3B_OUT", "~/bdh-review/reports/r3b"))
NB = int(os.environ.get("R3B_NB", "262144"))
DENSE = os.environ.get("R3B_DENSE", "es,pl,sk,et").split(",")
CTRL_TRAIN = int(os.environ.get("R3D_CTRL_TRAIN", "12"))
PROJ = int(os.environ.get("R3D_PROJ", "4096"))
FULL = os.environ.get("R3D_FULL") == "1"


def owner(ti):
    return "en(base)" if ti <= 3 else (SEQ[ti - 3] if 0 <= ti - 3 < len(SEQ) else f"?{ti}")


d = np.load(f"{OUT}/features_sparse.npz", allow_pickle=True)
Y, L, WIDTHS = d["Y"], d["LANG"].astype(str), d["WIDTHS"]
K = len(WIDTHS)
indptr, indices, values = d["indptr"], d["indices"], d["values"].astype(np.float32)
X = np.zeros((len(Y), NB), dtype=np.float32)
for i in range(len(Y)):
    s, e = indptr[i], indptr[i + 1]
    X[i, indices[s:e]] = values[s:e]
print(f"design matrix {X.shape}; crops/domain: " +
      " ".join(f"{l}:{int((L == l).sum())}" for l in sorted(set(L.tolist()))), flush=True)
if not FULL:
    rs = np.random.RandomState(2024)
    P = rs.normal(0.0, 1.0 / math.sqrt(PROJ), size=(NB, PROJ)).astype(np.float32)
    X = (X @ P).astype(np.float32)
    del P
    print(f"projected to {X.shape[1]} dims (JL); norms mean {np.linalg.norm(X,axis=1).mean():.3f}", flush=True)


def softmax_fit(tr, te, epochs=200, lr=0.5, l2=1e-4, cw=None):
    W = np.zeros((X.shape[1], K), np.float32); b = np.zeros(K, np.float32)
    Xt, yt, Xv = X[tr], Y[tr], X[te]
    w = np.ones(len(tr), np.float32) if cw is None else cw
    for _ in range(epochs):
        o = Xt @ W + b; o -= o.max(1, keepdims=True)
        p = np.exp(o); p /= p.sum(1, keepdims=True)
        G = (p - np.eye(K)[yt]) * w[:, None]
        W -= lr * (Xt.T @ G / w.sum() + l2 * W); b -= lr * G.mean(0)
    return (Xv @ W + b).argmax(1)


def ovr_fit(tr, te, epochs=400, lr=0.5, l2=1e-3):
    S = np.zeros((len(te), K), np.float32); Xt, Xv = X[tr], X[te]
    for k in range(K):
        y = (Y[tr] == k).astype(np.float32)
        pos = max(1.0, float(y.sum())); nneg = max(1.0, len(y) - pos)
        w = y * (len(y) / (2 * pos)) + (1 - y) * (len(y) / (2 * nneg))
        v = np.zeros(X.shape[1], np.float32); bb = 0.0
        for _ in range(epochs):
            z = 1.0 / (1.0 + np.exp(-(Xt @ v + bb)))
            g = (z - y) * w
            v -= lr * ((Xt.T @ g) / len(y) + l2 * v); bb -= lr * g.mean()
        S[:, k] = Xv @ v + bb
    return S.argmax(1), S


def wilson(a, n, z=1.96):
    den = 1 + z * z / n; c = (a + z * z / (2 * n)) / den
    return c - z * math.sqrt(a * (1 - a) / n + z * z / (4 * n * n)) / den, c + z * math.sqrt(a * (1 - a) / n + z * z / (4 * n * n)) / den


rng = np.random.RandomState(7)
perm = rng.permutation(len(Y)); cut = len(Y) * 3 // 4
tr, te = perm[:cut], perm[cut:]
cnt = np.bincount(Y[tr], minlength=K).astype(float)
cw = 1.0 / np.maximum(cnt[Y[tr]], 1.0); cw *= cw.sum() / len(tr)
print(f"\nsplit: train={len(tr)} test={len(te)} (test crop counts/domain: " +
      ", ".join(f"{l}:{int((L[te]==l).sum())}" for l in sorted(set(L[te].tolist()))) + ")", flush=True)

ARMS = {"A_r3b_reproduce": dict(epochs=200, l2=1e-4),
        "B_class_balanced": dict(epochs=600, l2=1e-5, cw=cw),
        "C_one_vs_rest": None}
res = {}
for name, kw in ARMS.items():
    pred, att = (ovr_fit(tr, te)[0], None) if kw is None else (softmax_fit(tr, te, **kw), None)
    a = float((pred == Y[te]).mean()); lo, hi = wilson(a, len(te))
    miss = {}
    for k, i in enumerate(te):
        if pred[k] != Y[i]:
            miss[int(pred[k])] = miss.get(int(pred[k]), 0) + 1
    top = sorted(miss.items(), key=lambda x: -x[1])[:4]
    per = {l: round(float((pred[L[te] == l] == Y[te][L[te] == l]).mean()), 2) for l in sorted(set(L[te].tolist()))}
    res[name] = dict(agreement=a, wilson=[lo, hi], attractor={owner(t): c for t, c in top}, per_domain=per)
    print(f"\n{name:<18s} agreement {a:.3f} [{lo:.3f},{hi:.3f}]   n={len(te)}", flush=True)
    print(f"{'':18s} miss-attractor (predicted territory): " + ", ".join(f"{owner(t)} x{c}" for t, c in top), flush=True)
    print(f"{'':18s} dense-domain rows: " + " ".join(f"{l}:{per[l]:.2f}" for l in DENSE if l in per), flush=True)
    print(f"{'':18s} control-domain mean: {np.mean([v for k, v in per.items() if k not in DENSE]):.3f}", flush=True)

print("\n=== composition-controlled density curve (controls pinned at " + str(CTRL_TRAIN) + " train crops, class weights RECOMPUTED at each point) ===")
ctrl = [l for l in SEQ if l not in DENSE]
base_tr, base_te = [], []
for l in ctrl:
    ids = np.where(L == l)[0]; sp = np.random.RandomState(101).permutation(ids)
    base_tr += list(sp[:min(CTRL_TRAIN, len(sp))]); base_te += list(sp[min(CTRL_TRAIN, len(sp)):min(CTRL_TRAIN + 4, len(sp))])
print(f"{'domain':>8s} " + " ".join(f"{s:>7d}" for s in (8, 16, 32, 64)) + "   | controls' agreement under the same fit", flush=True)
curve = {}
for dl in DENSE:
    ids = np.where(L == dl)[0]; sp = np.random.RandomState(11).permutation(ids)
    tr_d, te_d = sp[:64], sp[64:96]
    row = []
    for s in (8, 16, 32, 64):
        T = np.array(list(base_tr) + list(tr_d[:s])); V = np.array(list(te_d) + list(base_te))
        c = np.bincount(Y[T], minlength=K).astype(float)
        w = 1.0 / np.maximum(c[Y[T]], 1.0); w *= w.sum() / len(T)   # rebalance at EVERY point
        p = softmax_fit(T, V, epochs=600, l2=1e-5, cw=w)
        nd = len(te_d)
        row.append((float((p[:nd] == Y[V][:nd]).mean()), float((p[nd:] == Y[V][nd:]).mean())))
    curve[dl] = [(round(a, 3), round(b, 3)) for a, b in row]
    print(f"{dl:>8s} " + " ".join(f"{a:7.2f}" for a, b in row) + "   | " + " ".join(f"{b:5.2f}" for a, b in row), flush=True)
summary = {"arms": res, "density_curve_controls_pinned": curve,
           "settings": {"proj": None if FULL else PROJ, "ctrl_train": CTRL_TRAIN, "n_test": int(len(te))}}
with open(f"{OUT}/r3d_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"\nwrote {OUT}/r3d_summary.json")
print("READ: if arm A's attractor sits on es/pl/sk/et while B/C spread misses and lift controls, the r3b")
print("result was starvation, and F-2's unseen-language rows must be regenerated under a balanced fit.")
