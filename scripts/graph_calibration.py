"""E1: recalibrate BDH neuron-graph statistics across scales.

The prior cross-scale comparison used a fixed absolute threshold beta=0.30 on
G = D_x @ E. Because |G| entries scale with D and N, that fixed threshold can
manufacture apparent structural collapse. This script:
  1. sweeps beta for trained vs freshly-initialized (null) twins of each model,
  2. reports metrics at null-calibrated thresholds (init |G| quantiles),
  3. repeats the 100M analysis on a neuron-count-matched (2048) subsample.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from bdh import BDH, BDHConfig
from bdh_linear import BDHLinear
from pipeline.analyze import (
    _load_model,
    extract_head_graph,
    label_partition,
    newman_modularity,
    powerlaw_exponent,
)


def fresh_model(cfg_dict):
    fields = {f.name for f in BDHConfig.__dataclass_fields__.values()}
    bdh_cfg = BDHConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    torch.manual_seed(1234)
    cls = BDHLinear if cfg_dict.get("model") == "bdh-linear" else BDH
    return cls(bdh_cfg).eval()


def metrics(g, beta):
    thresh = (g >= beta).astype(np.float64)
    m = int(thresh.sum())
    out_deg = thresh.sum(axis=1)
    rep = {"beta": beta, "edges": m, "edge_fraction": m / g.size}
    alpha = powerlaw_exponent(out_deg, xmin=max(1.0, beta))
    if alpha is not None:
        rep["alpha"] = round(alpha, 3)
    if 50 <= m <= 3_000_000:
        labels = label_partition(thresh)
        rep["Q"] = round(newman_modularity(thresh, labels), 4)
        rep["comm"] = int(len(np.unique(labels)))
    return rep


def row(r):
    return (f"{r['beta']:8.3f} {r['edges']:9d} {r['edge_fraction']:10.6f} "
            f"{str(r.get('alpha', '-')):>7} {str(r.get('Q', '-')):>8} "
            f"{str(r.get('comm', '-')):>6}")


def study(tag, g, g_init, betas, qs=(0.999, 0.9999)):
    print(f"\n== {tag} | neurons={g.shape[0]} ==")
    print(f"{'beta':>8} {'edges':>9} {'edge_frac':>10} {'alpha':>7} {'Q':>8} {'comm':>6}")
    for b in betas:
        print(row(metrics(g, b)))
    qs_init = {q: float(np.quantile(np.abs(g_init), q)) for q in qs}
    spread = ", ".join(f"p{int(q*1e4)/1e2:g}={v:.3f}" for q, v in qs_init.items())
    print(f"-- null floor of |G_init|: {spread} --")
    print(f"{'beta*':>8} {'edges':>9} {'edge_frac':>10} {'alpha':>7} {'Q':>8} {'comm':>6}"
          f"   (calibration)")
    for q, bstar in qs_init.items():
        r = metrics(g, bstar)
        print(row(r) + f"   q={q:g}")


def subsample(g, n_keep, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(g.shape[0], size=n_keep, replace=False))
    return g[np.ix_(idx, idx)]


def main():
    import os

    betas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8]
    if len(sys.argv) > 1:
        items = [(os.path.basename(p), p) for p in sys.argv[1:]]
    else:
        items = [("REF 25M lina005", "out/bdh-linear_wikitext2_lina005_best.pt"),
                 ("TEST 100M", "out/bdh_wikitext2_100m_best.pt")]
    for tag, path in items:
        model, cfg = _load_model(path)
        g = extract_head_graph(model, 0).detach().numpy().astype(np.float64)
        gi = extract_head_graph(fresh_model(cfg), 0).detach().numpy().astype(np.float64)
        study(tag, g, gi, betas)
        if g.shape[0] > 4096:
            study(f"{tag} subsampled->2048",
                  subsample(g, 2048, 7), subsample(gi, 2048, 7), betas)


if __name__ == "__main__":
    main()
