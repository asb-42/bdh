#!/usr/bin/env python3
"""Weight-atlas territory statistics for BDH checkpoints.

Atlas stores BDH matrices as 64-neuron per-head tiles (`decoder.uN.hN`, u=0..735, h=0..7).
BDH grows latent width by whole blocks of 2048 neurons, so language territory = u // 32
(blocks 0-3 = base/English era; block b>=4 belongs to phase b-2 of the ladder).

This turns atlas tile stats into per-language statistics and joins them against the RA2b
serving matrix, so weight conditioning can be compared with measured acquisition/damage.

Usage: ATLAS=http://192.168.178.200:8000 .venv/bin/python atlas_territory_stats.py <job_id>
Requires: docs/reports/data/2026-09-10_ra2b_matrix.csv (or MATRIX=path).
"""
import os, re, csv, json, math, urllib.request, collections, statistics as S

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
BLK_TILES = 2048 // 64          # 32 tiles per BDH block
ATLAS = os.environ.get("ATLAS", "http://192.168.178.200:8000")
MATRIX = os.environ.get("MATRIX", "docs/reports/data/2026-09-10_ra2b_matrix.csv")
JOB = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else None
assert JOB, "pass an atlas job_id"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


tiles, off = {}, 0
while True:
    d = get(f"{ATLAS}/api/model/{JOB}/query?limit=500&offset={off}")
    got = d.get("rows", [])
    for r in got:
        m = re.match(r"decoder\.u(\d+)\.h(\d+)$", r["tensor_name"])
        if m:
            tiles.setdefault(int(m.group(1)) // BLK_TILES, []).append(r)
    if not d.get("has_more"):
        break
    off += d.get("limit", 500)
print(f"tiles collected: {sum(len(v) for v in tiles.values())} across {len(tiles)} territories")

M = {}
for x in csv.DictReader(open(MATRIX)):
    M.setdefault(x["checkpoint"], {})[x["eval_lang"]] = float(x["ppl"])
METRICS = ("effective_rank", "spectral_norm", "sparsity", "kurtosis", "outlier_3s")

print("\n blk|lang | exit | final | " + " | ".join(f"{m[:9]:>9s}" for m in METRICS))
tab = []
for b in sorted(tiles):
    lang = "en" if b <= 3 else (SEQ[b - 3] if 4 <= b <= 22 else None)
    if lang is None or lang not in M:
        continue
    row = {"b": b, "l": lang, "ex": M[lang][lang], "fin": M["lt"][lang]}
    for m in METRICS:
        row[m] = S.mean([t[m] for t in tiles[b]])
    tab.append(row)
    print(f" {b:>3d}| {lang:>4s} | {row['ex']:4.2f} | {row['fin']:5.2f} | " +
          " | ".join(f"{row[m]:9.3f}" for m in METRICS))


def pearson(xs, ys):
    ok = [(a, b) for a, b in zip(xs, ys) if a == a and b == b and b > 0]
    xs = [a for a, _ in ok]; ys = [b for _, b in ok]
    if len(xs) < 4:
        return float("nan"), 0
    mx, my = S.mean(xs), S.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = (sum((a - mx) ** 2 for a in xs)) ** 0.5; sy = (sum((b - my) ** 2 for b in ys)) ** 0.5
    r = cov / (sx * sy)
    t = r * math.sqrt((len(xs) - 2) / max(1e-12, 1 - r * r))
    return r, t


def spearman(xs, ys):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, k in enumerate(o):
            r[k] = i
        return r
    return pearson(rk(xs), rk(ys))[0]


# English contributes four base territories: they are pseudo-replicates of one language,
# so correlations use the 19 appended (one territory per language) blocks only.
pts = [t for t in tab if t["l"] != "en"]
print(f"\ncorrelations over {len(pts)} one-per-language territories (base blocks excluded as pseudo-replicates)")
for tgt, lab in ((lambda t: t["ex"], "exit ppl"), (lambda t: t["fin"] / t["ex"], "damage ratio final/exit")):
    ys = [tgt(t) for t in pts]
    for m in METRICS:
        xs = [t[m] for t in pts]
        r, tt = pearson(xs, ys)
        print(f"  {m:14s} vs {lab:24s} r={r:+.3f} t={tt:+.2f} rho={spearman(xs, ys):+.3f}")
print("\nRead Pearson and Spearman TOGETHER. On the current data spectral_norm gives r=+0.83 "
      "against exit PPL but rho=+0.28: the linear fit is driven by bg/el alone. Claim at most "
      "'the Greek/Bulgarian territories are spectrally unlike the rest and are also the two worst "
      "served'; do not claim prediction across the other languages.")
