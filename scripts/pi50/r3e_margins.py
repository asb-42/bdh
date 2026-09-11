"""P-R3 margin analysis for Sonde C's stage-1 trigger (bus #197 design note). Zero GPU.

Two questions, both answerable from the same balanced fit r3d used:
  Q1 In-support calibration: what does the top-1 minus top-2 softmax margin look like on held-out crops of
     TRAINED domains, where the addresser is 1.000 correct? This is the reference distribution a trigger
     threshold would be set against.
  Q2 Does the margin detect the one case where the addresser is confidently wrong? Quinn's likelihood router
     sends hi to bg/el; the byte addresser sends hi to fi x40. If hi's margins look like the trained-domain
     margins, then predicted-route score carries NO abstention signal and a trigger must key on
     distance-to-support instead - which is what section 9.3 point 3 asserted qualitatively. Testing my own
     adjective rather than leaving it as rhetoric.

Run: .venv/bin/python scripts/pi50/r3e_margins.py
"""
import os, re, json, math, numpy as np

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
OUT = os.path.expanduser(os.environ.get("R3B_OUT", "~/bdh-review/reports/r3b"))
OOD_DIR = os.path.expanduser(os.environ.get("R3B_OOD_DIR", "~/bdh-review/data/ood"))
NB = int(os.environ.get("R3B_NB", "262144")); PROJ = int(os.environ.get("R3D_PROJ", "4096"))
BS = int(os.environ.get("R3B_BLOCK", "512")); EPOCHS = 600; LR = 0.5; L2 = 1e-5


def owner(ti):
    return "en(base)" if ti <= 3 else (SEQ[ti - 3] if 0 <= ti - 3 < len(SEQ) else f"?{ti}")


d = np.load(f"{OUT}/features_sparse.npz", allow_pickle=True)
Y, L = d["Y"], d["LANG"].astype(str); K = len(d["WIDTHS"])
indptr, indices, values = d["indptr"], d["indices"], d["values"].astype(np.float32)
X = np.zeros((len(Y), NB), dtype=np.float32)
for i in range(len(Y)):
    s, e = indptr[i], indptr[i + 1]
    X[i, indices[s:e]] = values[s:e]
rs = np.random.RandomState(2024)
P = rs.normal(0.0, 1.0 / math.sqrt(PROJ), size=(NB, PROJ)).astype(np.float32)
X = (X @ P).astype(np.float32)

rng = np.random.RandomState(7); perm = rng.permutation(len(Y)); cut = len(Y) * 3 // 4
tr, te = perm[:cut], perm[cut:]
cnt = np.bincount(Y[tr], minlength=K).astype(float)
cw = 1.0 / np.maximum(cnt[Y[tr]], 1.0); cw *= cw.sum() / len(tr)
W = np.zeros((X.shape[1], K), np.float32); b = np.zeros(K, np.float32)
Xt, yt = X[tr], Y[tr]
for _ in range(EPOCHS):
    o = Xt @ W + b; o -= o.max(1, keepdims=True)
    p = np.exp(o); p /= p.sum(1, keepdims=True)
    G = (p - np.eye(K)[yt]) * cw[:, None]
    W -= LR * (Xt.T @ G / cw.sum() + L2 * W); b -= LR * G.mean(0)


def margins(M):
    s = np.sort(M, axis=1)
    return s[:, -1] - s[:, -2], M.argmax(1)


o = X[te] @ W + b
sm = np.exp(o - o.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
mg, pr = margins(sm)
ok = pr == Y[te]
print(f"trained held-out: n={len(te)} accuracy={ok.mean():.3f}")
for lab, sel in (("correct  ", ok), ("INCORRECT", ~ok)):
    if sel.sum():
        q = np.percentile(mg[sel], [5, 25, 50, 75, 95])
        print(f"  {lab} n={int(sel.sum()):3d} margin p5/p25/p50/p75/p95 = " + " ".join(f"{v:.3f}" for v in q))
print(f"  worst-case correct margin (the trigger floor if it must never fire in-support): {mg.min():.4f}")
hist, edges = np.histogram(mg, bins=10, range=(0, max(1.0, mg.max())))
print("  margin histogram (correct preds): " + " ".join(f"{edges[i]:.2f}-{edges[i+1]:.2f}:{hist[i]}" for i in range(10)))


def feats(bb):
    v = np.zeros(NB, dtype=np.float32); a = np.frombuffer(bb, dtype=np.uint8)
    for k in (1, 2, 3, 4):
        if len(a) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, k)
        key = np.zeros(len(w), dtype=np.uint64)
        for j in range(k):
            key = key * np.uint64(257) + w[:, j].astype(np.uint64)
        np.add.at(v, (key % np.uint64(NB)).astype(np.int64), 1.0)
    n = np.linalg.norm(v)
    return ((v / (n if n > 0 else 1.0)) @ P).astype(np.float32)


FILES = {"zh": ("xscript_zh_head.txt", None), "ja": ("xscript_ja_head.txt", None),
         "hi": ("xscript_hi_head.txt", None), "lv": ("lv_head.txt", None),
         "ga": ("ga_head.txt", None), "iu_syllabic_only": ("xscript_iu.txt", "syl"),
         "iu_ascii_only": ("xscript_iu.txt", "ascii")}
REF = {"zh": "{bg,el}", "ja": "{bg,el}", "hi": "{bg,el}", "lv": "lt", "ga": "?", "iu_syllabic_only": "TBD",
       "iu_ascii_only": "en-ish"}
print("\nunseen inputs: does the margin flag the case where the addresser disagrees with likelihood?")
print(f"{'input':>18s} {'routes':>26s} {'margin med':>10s} {'p10':>6s} {'frac<floor':>10s}  likelihood ref")
res = {}
for name, (fn, filt) in FILES.items():
    path = os.path.join(OOD_DIR, fn)
    if not os.path.exists(path):
        continue
    txt = open(path, "rb").read()
    if filt:
        lines = txt.split(b"\n")
        if filt == "syl":
            keep = [l for l in lines if re.search(r"[\u1400-\u167f]", l.decode("utf8", "ignore"))]
        else:
            keep = [l for l in lines if l.strip() and not re.search(rb"[^\x00-\x7f]", l)]
        txt = b"\n".join(keep)
    if len(txt) < 40 * BS:
        continue
    gs = np.random.RandomState(99)
    Fm = np.array([feats(txt[int(i):int(i) + BS]) for i in gs.randint(0, len(txt) - BS - 1, 40)], np.float32)
    oo = Fm @ W + b
    ss = np.exp(oo - oo.max(1, keepdims=True)); ss /= ss.sum(1, keepdims=True)
    mm, pp = margins(ss)
    routes = {}
    for q in pp:
        routes[owner(int(q))] = routes.get(owner(int(q)), 0) + 1
    top = ", ".join(f"{k}x{v}" for k, v in sorted(routes.items(), key=lambda z: -z[1])[:3])
    floor = mg.min()
    res[name] = dict(routes=routes, margin_med=float(np.median(mm)), margin_p10=float(np.percentile(mm, 10)),
                     frac_below_insupport_floor=float((mm < floor).mean()))
    print(f"{name:>18s} {top:>26s} {np.median(mm):10.3f} {np.percentile(mm,10):6.3f} "
          f"{(mm < floor).mean():10.2f}  {REF.get(name,'')}{'' if name!='iu_syllabic_only' else ' (pl/ro vs TBD)'}")
json.dump({"insupport_correct_margin_min": float(mg.min()), "insupport_margin_percentiles":
           [float(v) for v in np.percentile(mg, [5, 25, 50, 75, 95])], "ood": res},
          open(f"{OUT}/r3e_margins.json", "w"), indent=1)
print(f"\nwrote {OUT}/r3e_margins.json")
print("READ: frac_below_insupport_floor near 0 means unseen-script predictions sit INSIDE the in-support")
print("margin band -> the score cannot detect them; a trigger needs distance-to-support, not confidence.")
