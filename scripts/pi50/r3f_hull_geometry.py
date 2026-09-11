"""Test A0-Quinn's hull hypothesis (#200): does the byte addresser fail because unseen scripts lie far from
the hull of TRAINED byte statistics? CPU-only, zero GPU, reads persisted features.

Hypothesis under test (his words): "the byte addresser interpolates within the hull of trained byte
statistics; it generalizes to scripts that neighbor the hull's high-byte boundary (zh/ja) and snaps
arbitrarily for scripts that don't (hi/iu)."

Falsifier, stated before running: if hi and iu-syllabic centroids sit CLOSEST to the bg/el territory
centroids in n-gram space, then their misrouting is not hull distance and his mechanism is wrong - the
addresser would be failing for some other reason while the geometry says what the likelihood router says.
Supporting signature: hi/iu sit nearest the territories the fitted linear model actually chose (fi for hi,
pl/ro for iu), while zh/ja sit nearest bg/el.

Validity gate first, or none of the rest means anything: pure nearest-centroid classification with
LEAVE-ONE-OUT must reproduce the trained-domain routes at high accuracy. If centroids cannot recover what a
fitted linear model recovers at 1.000, the distance geometry is too crude to test any hypothesis with and
this script reports nothing else.

Run: .venv/bin/python scripts/pi50/r3f_hull_geometry.py
"""
import os, re, math, json, numpy as np

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
POS = {l: i for i, l in enumerate(SEQ)}
OUT = os.path.expanduser(os.environ.get("R3B_OUT", "~/bdh-review/reports/r3b"))
OOD_DIR = os.path.expanduser(os.environ.get("R3B_OOD_DIR", "~/bdh-review/data/ood"))
NB = int(os.environ.get("R3B_NB", "262144")); BS = int(os.environ.get("R3B_BLOCK", "512"))


def idx_of(lang):
    return 3 if lang == "en" else POS[lang] + 3


def owner(ti):
    return "en(base)" if ti <= 3 else (SEQ[ti - 3] if 0 <= ti - 3 < len(SEQ) else f"?{ti}")


d = np.load(f"{OUT}/features_sparse.npz", allow_pickle=True)
Y, L = d["Y"], d["LANG"].astype(str)
indptr, indices, values = d["indptr"], d["indices"], d["values"].astype(np.float32)
X = np.zeros((len(Y), NB), dtype=np.float32)
for i in range(len(Y)):
    s, e = indptr[i], indptr[i + 1]
    X[i, indices[s:e]] = values[s:e]
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)   # unit profiles -> cosine = dot
K = int(Y.max()) + 1
print(f"profiles {X.shape}, territories 0..{K-1}")

# ---------- gate: leave-one-out nearest centroid on TRAINED domains
cent = np.zeros((K, NB), np.float32)
for k in range(K):
    m = Y == k
    cent[k] = X[m].mean(0) if m.any() else 0.0
lo_acc, lo_tot = 0, 0
per_lo = {}
for i in range(len(Y)):
    k = int(Y[i]); m = Y == k
    xi = X[i]
    c = (X[m].sum(0) - xi) / max(1, int(m.sum()) - 1)        # own centroid WITHOUT this crop
    c /= max(float(np.linalg.norm(c)), 1e-9)
    sc = cent @ xi                                           # crop vs every territory centroid
    sc[k] = float(xi @ c)                                    # own entry replaced by the leave-one-out value,
                                                             # otherwise the crop scores itself against a
                                                             # centroid it is inside and the gate is fake
    best_other = float(np.max(np.delete(sc, k)))
    hit = float(sc[k]) > best_other
    lo_acc += int(hit); lo_tot += 1
    rec = per_lo.setdefault(str(L[i]), [0, 0]); rec[0] += int(hit); rec[1] += 1
gate = lo_acc / lo_tot
worst = sorted(((v[0] / v[1], k) for k, v in per_lo.items()))[:4]
print(f"\nGATE leave-one-out nearest-centroid on trained crops: {gate:.3f} ({lo_acc}/{lo_tot})")
print("  weakest domains: " + ", ".join(f"{k}:{a:.2f}" for a, k in worst))
if gate < 0.90:
    raise SystemExit(f"GATE FAILED ({gate:.3f} < 0.90): centroid geometry too crude to test the hypothesis; "
                     "no OOD conclusion may be drawn from this run.")
print("  gate PASSED - distance geometry recovers trained routes, so OOD distances are interpretable\n")

# ---------- OOD inputs
FILES = {"zh": ("xscript_zh_head.txt", None), "ja": ("xscript_ja_head.txt", None),
         "HIKE": ("xscript_hi_head.txt", None), "lv": ("lv_head.txt", None), "ga": ("ga_head.txt", None),
         "iu_syllabic": ("xscript_iu.txt", "syl"), "iu_ascii": ("xscript_iu.txt", "ascii")}
CHOSEN = {"zh": "el/bg (matches likelihood)", "ja": "bg/el (matches)", "HIKE": "fi x40 (DIVERGES)",
          "lv": "lt x40 (matches)", "ga": "hu/sv/it (unknown)", "iu_syllabic": "pl/ro (DIVERGES)",
          "iu_ascii": "en(base) (matches)"}
REF = {"zh": "bg/el", "ja": "bg/el", "HIKE": "bg/el", "lv": "lt", "ga": "?", "iu_syllabic": "el x28/bg x12",
       "iu_ascii": "en-ish"}
print(f"{'input':>12s} {'>=0x80 bytes':>12s} {'nearest territories (cosine)':>44s}   {'bg/el rank':>10s}  chosen-by-model")
res = {}
for name, (fn, filt) in FILES.items():
    path = os.path.join(OOD_DIR, fn)
    if not os.path.exists(path):
        continue
    txt = open(path, "rb").read()
    if filt == "syl":
        txt = b"\n".join(l for l in txt.split(b"\n") if re.search(r"[\u1400-\u167f]", l.decode("utf8", "ignore")))
    elif filt == "ascii":
        txt = b"\n".join(l for l in txt.split(b"\n") if l.strip() and not re.search(rb"[^\x00-\x7f]", l))
    if len(txt) < 40 * BS:
        print(f"{name:>12s}  pool too small ({len(txt)} B)"); continue
    hb = float((np.frombuffer(txt, dtype=np.uint8) >= 0x80).mean())
    v = np.zeros(NB, np.float32); a = np.frombuffer(txt, dtype=np.uint8)
    for k in (1, 2, 3, 4):
        if len(a) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, k)
        key = np.zeros(len(w), dtype=np.uint64)
        for j in range(k):
            key = key * np.uint64(257) + w[:, j].astype(np.uint64)
        np.add.at(v, (key % np.uint64(NB)).astype(np.int64), 1.0)
    n = np.linalg.norm(v)
    if n == 0:
        continue
    v /= n
    sc = cent @ v
    order = np.argsort(-sc)
    near = ", ".join(f"{owner(int(t))}:{sc[t]:.3f}" for t in order[:4])
    ranks = {t: int(np.where(order == t)[0][0]) + 1 for t in (idx_of("bg"), idx_of("el"))}
    res[name] = dict(high_byte_share=hb, nearest=[(owner(int(t)), float(sc[t])) for t in order[:6]],
                     bg_rank=ranks[idx_of("bg")], el_rank=ranks[idx_of("el")], chosen=CHOSEN.get(name, ""))
    print(f"{name:>12s} {hb*100:11.1f}% {near:>44s}   bg#{ranks[idx_of('bg')]:d} el#{ranks[idx_of('el')]:d}  {CHOSEN.get(name,'')}")

order_by_hb = sorted(res.items(), key=lambda kv: -kv[1]["high_byte_share"])
print("\nhigh-byte share ordering (his single scalar): " + ", ".join(f"{k} {v['high_byte_share']*100:.1f}%" for k, v in order_by_hb))
print("likelihood-router reference: " + "; ".join(f"{k}->{v}" for k, v in REF.items()))
json.dump({"gate_loo_nearest_centroid": gate, "inputs": res}, open(f"{OUT}/r3f_hull.json", "w"), indent=1)
print(f"\nwrote {OUT}/r3f_hull.json")
print("READ: bg/el rank small AND model chose bg/el -> hull story holds for that input. bg/el rank large AND")
print("model chose something else -> the snap target is genuinely off-hull. Mixed -> mechanism incomplete.")
