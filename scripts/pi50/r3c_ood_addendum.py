"""P-R3 addendum: F-2 extended to six unseen languages + F-3 confusion with the CORRECTED territory map.

Zero GPU: everything here is recomputed from the sparse features persisted by
r3b_byte_addressing_density.py ({OUT}/features_sparse.npz). Refitting a 23-class logistic model on ~640
samples takes seconds.

WHY THIS EXISTS (instrument defect found after launch): r3b's lang_of_territory() used SEQ[ti-4], but a
language at ladder position p owns cumulative-prefix index p+3, not p+4 - English (p=0) selects index 3
(= width 8192), which is its own base block. With the off-by-one the log printed "own=False" for perfectly
correct behaviour (ro -> index 19 is ro, not sv) and any confusion table would have named every predicted
territory one language late. Asserted here so it cannot silently recur:
    assert territory_of_position(p) == p + 3

Run: .venv/bin/python scripts/pi50/r3c_ood_addendum.py
Env: R3B_OUT=~/bdh-review/reports/r3b  R3B_OOD_DIR=~/bdh-review/data/ood  R3B_NB=262144 R3B_EPOCHS=200 R3B_LR=0.5
"""
import os, re, json, math, numpy as np

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
POS = {l: i for i, l in enumerate(SEQ)}
OUT = os.path.expanduser(os.environ.get("R3B_OUT", "~/bdh-review/reports/r3b"))
OOD_DIR = os.path.expanduser(os.environ.get("R3B_OOD_DIR", "~/bdh-review/data/ood"))
NB_BUCKETS = int(os.environ.get("R3B_NB", "262144"))
EPOCHS = int(os.environ.get("R3B_EPOCHS", "200")); LR = float(os.environ.get("R3B_LR", "0.5"))
BS = int(os.environ.get("R3B_BLOCK", "512"))
PROJ = int(os.environ.get("R3D_PROJ", "4096"))


def idx_of_pos(p):
    return p + 3                      # cumulative prefix (p+4) blocks x 2048 neurons


def owner(ti):
    """Territory index -> language that acquired it. Territory p+3 belongs to ladder position p;
    indices 0-3 are the shared English-era base blocks."""
    if ti <= 3:
        return "en(base)"
    if 0 <= ti - 3 < len(SEQ):
        return SEQ[ti - 3]
    return f"?{ti}"


for p, l in enumerate(SEQ):
    expect = "en(base)" if l == "en" else l
    assert owner(idx_of_pos(p)) == expect, f"territory map broken for {l}: {owner(idx_of_pos(p))}"
print("territory map self-check OK: en->3 (base) ... lt->22; ro->19")

d = np.load(f"{OUT}/features_sparse.npz", allow_pickle=True)
Y, L, WIDTHS = d["Y"], d["LANG"].astype(str), d["WIDTHS"]
K = len(WIDTHS)
indptr, indices, values = d["indptr"], d["indices"], d["values"].astype(np.float32)
X = np.zeros((len(Y), NB_BUCKETS), dtype=np.float32)
for i in range(len(Y)):
    s, e = indptr[i], indptr[i + 1]
    X[i, indices[s:e]] = values[s:e]
print(f"reconstructed design matrix {X.shape}, {int((X != 0).sum())} nonzeros")

# Same Johnson-Lindenstrauss projection as r3d_fit_ablation.py (identical seed/scale) so the balanced fit
# and the OOD features live in one space; full-dimensional fitting was ~10 min per arm on this CPU.
_rs = np.random.RandomState(2024)
P = _rs.normal(0.0, 1.0 / math.sqrt(PROJ), size=(NB_BUCKETS, PROJ)).astype(np.float32)
X = (X @ P).astype(np.float32)
print(f"projected to {PROJ} dims (JL, seed 2024 - must match r3d)")

rng = np.random.RandomState(7)
perm = rng.permutation(len(Y)); cut = len(Y) * 3 // 4
tr, te = perm[:cut], perm[cut:]
W = np.zeros((X.shape[1], K), dtype=np.float32); b = np.zeros(K, dtype=np.float32)
Xt, yt = X[tr], Y[tr]
cnt = np.bincount(yt, minlength=K).astype(float)
cw = 1.0 / np.maximum(cnt[yt], 1.0); cw *= cw.sum() / len(tr)   # class-balanced: r3d showed unweighted
for ep in range(EPOCHS):                                        # fits collapse onto over-represented classes
    o = Xt @ W + b; o -= o.max(1, keepdims=True)
    p_ = np.exp(o); p_ /= p_.sum(1, keepdims=True)
    G = (p_ - np.eye(K)[yt]) * cw[:, None]
    W -= LR * (Xt.T @ G / cw.sum() + 1e-5 * W); b -= LR * G.mean(0)
pred = (X[te] @ W + b).argmax(1)
acc = float((pred == Y[te]).mean())
print(f"\nagreement reproduced on held-out crops: {acc:.3f} (n={len(te)})")

print("\n=== F-3 CONFUSION, corrected labels (true language -> owner of predicted territory) ===")
conf = {}
for k, i in enumerate(te):
    if pred[k] != Y[i]:
        conf.setdefault(L[i], {}).setdefault(owner(int(pred[k])), 0)
        conf[L[i]][owner(int(pred[k]))] += 1
for l in sorted(conf):
    tot = sum(conf[l].values())
    items = sorted(conf[l].items(), key=lambda z: -z[1])
    fam = {"cs": "west-slavic", "sk": "west-slavic", "pl": "west-slavic", "sl": "south-slavic",
           "bg": "south-slavic", "et": "uralic", "fi": "uralic", "hu": "uralic", "lt": "baltic"}
    print(f"  {l:>3s} ({tot} miss{'es' if tot > 1 else ' '}): " + ", ".join(f"->{o} x{c}" for o, c in items) +
          f"   [family of true: {fam.get(l, 'germanic/other')}]")
json.dump({k: v for k, v in conf.items()}, open(f"{OUT}/f3_confusion_corrected.json", "w"), indent=1)


def feats(bbytes):
    v = np.zeros(NB_BUCKETS, dtype=np.float32)
    a = np.frombuffer(bbytes, dtype=np.uint8)
    for k in (1, 2, 3, 4):
        if len(a) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, k)
        key = np.zeros(len(w), dtype=np.uint64)
        for j in range(k):
            key = key * np.uint64(257) + w[:, j].astype(np.uint64)
        np.add.at(v, (key % np.uint64(NB_BUCKETS)).astype(np.int64), 1.0)
    n = np.linalg.norm(v)
    v = v / (n if n > 0 else 1.0)
    return (v @ P).astype(np.float32)          # project into the same space the model was fitted in


print("\n=== F-2 UNSEEN LANGUAGES (six inputs; ja/hi added after Quinn's chmod) ===")
FILES = {"zh": ("xscript_zh_head.txt", None), "ja": ("xscript_ja_head.txt", None),
         "hi": ("xscript_hi_head.txt", None), "lv": ("lv_head.txt", None),
         "ga": ("ga_head.txt", None), "iu_syllabic_only": ("xscript_iu.txt", "syl"),
         "iu_turkish_lines_ONLY": ("xscript_iu.txt", "tr"), "iu_ascii_lines_ONLY": ("xscript_iu.txt", "ascii")}
res = {}
for name, (fn, filt) in FILES.items():
    path = os.path.join(OOD_DIR, fn)
    if not os.path.exists(path):
        print(f"  {name:>21s}: missing {fn}"); continue
    txt = open(path, "rb").read()
    if filt:
        lines = txt.split(b"\n")
        keep = [l for l in lines if len(l.strip()) > 2 and ((filt == "syl" and re.search(r"[\u1400-\u167f]", l.decode("utf8", "ignore")))
                or (filt == "tr" and re.search(r"[ığşöçüĞŞİ]", l.decode("utf8", "ignore")) and not re.search(r"[\u1400-\u167f]", l.decode("utf8", "ignore")))
                or (filt == "ascii" and not re.search(rb"[^\x00-\x7f]", l)))]
        txt = b"\n".join(keep)
    if len(txt) < 40 * BS:
        print(f"  {name:>21s}: too small after filter ({len(txt)} B)"); continue
    gs = np.random.RandomState(99)
    starts = gs.randint(0, len(txt) - BS - 1, 40)
    ps = np.array([feats(txt[int(i):int(i) + BS]) for i in starts], dtype=np.float32)
    pr = (ps @ W + b).argmax(1)
    hist = {}
    for q in pr:
        hist[owner(int(q))] = hist.get(owner(int(q)), 0) + 1
    res[name] = hist
    top = sorted(hist.items(), key=lambda z: -z[1])[:4]
    print(f"  {name:>21s} ({len(txt)//1000} KB pool) -> " + ", ".join(f"{k} x{v}" for k, v in top))
json.dump(res, open(f"{OUT}/f2_ood_routes_corrected.json", "w"), indent=1)
print(f"\nwrote {OUT}/f3_confusion_corrected.json and f2_ood_routes_corrected.json")
print("NOTE: no ground-truth route exists for unseen languages; these are the CHEAP predictor's answers.")
print("Compare against Quinn's likelihood-router counts in out/logs/ra2b_routdiag_*.txt on .200.")
