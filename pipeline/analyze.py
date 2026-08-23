"""Interpretability tooling for trained BDH models (paper Secs. 4-5).

Three analyses, all runnable offline over a checkpoint:

  1. Activation sparsity  - fraction of non-zero entries of the positive
     activation vectors x_sparse / xy_sparse per layer (paper reports ~5%).
  2. Neuron-neuron graph  - extract G = D_x E per head, threshold it, and
     report heavy-tail degree statistics and Newman modularity (paper Sec. 4).
  3. Synapse tracing      - track the attention state rho[i, j] of individual
     synapses across a prompt (the paper's "monosemantic synapse" readout).

Usage:
    python -m pipeline.run analyze --ckpt out/bdh_best.pt --sparsity
    python -m pipeline.run analyze --ckpt out/bdh_best.pt --graph --beta 1.0
    python -m pipeline.run analyze --ckpt out/bdh_best.pt \
        --synapse 0:0:5:9 --synapse 3:1:12:2 --prompt "To be or not to be"
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from bdh import BDH


# ---------------------------------------------------------------------------
# 1. activation sparsity
# ---------------------------------------------------------------------------

@torch.no_grad()
def activation_sparsity(model: BDH, idx: torch.Tensor) -> dict:
    """Per-layer fraction of non-zero entries of x_sparse and xy_sparse."""
    C = model.config
    x = model.ln(model.embed(idx).unsqueeze(1))
    stats = []
    for level in range(C.n_layer):
        x_sparse = F.relu(x @ model.encoder)
        yKV = model.ln(model.attn(x_sparse, x)[0])
        y_sparse = F.relu(yKV @ model.encoder_v)
        xy_sparse = x_sparse * y_sparse
        stats.append({
            "layer": level,
            "x_sparsity": (x_sparse == 0).float().mean().item(),
            "xy_sparsity": (xy_sparse == 0).float().mean().item(),
        })
        yMLP = xy_sparse.transpose(1, 2).reshape(idx.size(0), 1, idx.size(1), -1) @ model.decoder
        x = model.ln(x + model.ln(yMLP))
    return {"per_layer": stats,
            "mean_x": sum(s["x_sparsity"] for s in stats) / len(stats),
            "mean_xy": sum(s["xy_sparsity"] for s in stats) / len(stats)}


# ---------------------------------------------------------------------------
# 2. neuron-neuron interaction graph
# ---------------------------------------------------------------------------

def extract_head_graph(model: BDH, head: int) -> torch.Tensor:
    """G = D_x E for one head: the (trained) neuron-neuron affinity matrix."""
    nh = model.config.n_head
    n = model.decoder.shape[0] // nh
    d_x = model.decoder[head * n:(head + 1) * n]      # (N, d)
    e = model.encoder[head]                            # (d, N)
    return d_x @ e                                     # (N, N)


def powerlaw_exponent(degrees: np.ndarray, xmin: float) -> float | None:
    """Continuous MLE fit alpha = 1 + n / sum(ln(x/xmin)) on the tail (Clauset)."""
    tail = degrees[degrees >= xmin].astype(np.float64)
    if len(tail) < 10:
        return None
    return 1.0 + len(tail) / np.log(tail / xmin).sum()


def newman_modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of a symmetrized weighted graph given a partition."""
    a = (adj + adj.T) / 2
    m = a.sum() / 2
    if m == 0:
        return 0.0
    k = a.sum(axis=1)
    expected = np.outer(k, k) / (2 * m)
    same = labels[:, None] == labels[None, :]
    return float((a - expected)[same].sum() / (2 * m))


def label_partition(adj: np.ndarray, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Label propagation community detection (dependency-free Louvain stand-in)."""
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    labels = np.arange(n)
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    for _ in range(iters):
        order = rng.permutation(n)
        changed = False
        for i in order:
            if len(neighbors[i]) == 0:
                continue
            counts = np.bincount(labels[neighbors[i]], minlength=n)
            best = counts.argmax()
            if labels[i] != best and counts[best] > counts[labels[i]]:
                labels[i] = best
                changed = True
        if not changed:
            break
    return labels


def graph_report(model: BDH, head: int, beta: float) -> dict:
    g = extract_head_graph(model, head).detach().numpy()
    thresh = (g >= beta).astype(np.float64)
    m_edges = int(thresh.sum())
    out_deg = thresh.sum(axis=1)
    in_deg = thresh.sum(axis=0)
    report = {
        "head": head,
        "neurons": g.shape[0],
        "beta": beta,
        "edges": m_edges,
        "edge_fraction": m_edges / g.size,
        "out_degree_mean": float(out_deg.mean()),
        "out_degree_max": int(out_deg.max()),
        "in_degree_max": int(in_deg.max()),
    }
    alpha_out = powerlaw_exponent(out_deg, xmin=max(1.0, beta))
    if alpha_out is not None:
        report["out_degree_powerlaw_alpha"] = alpha_out
    if m_edges > 0:
        labels = label_partition(thresh)
        report["newman_modularity"] = newman_modularity(thresh, labels)
        report["communities"] = int(len(np.unique(labels)))
    return report


# ---------------------------------------------------------------------------
# 3. synapse tracing
# ---------------------------------------------------------------------------

@torch.no_grad()
def synapse_trace(model: BDH, idx: torch.Tensor, synapses: list[tuple[int, int, int, int]]) -> dict:
    """Trace rho[layer, head][i, j] after every token (single incremental pass).

    synapses: list of (layer, head, i, j). Returns {synapse: [value per token]}.
    Requires the state-space (bdh-linear) model, whose per-synapse state is
    materialized; the quadratic path only keeps an aggregated KV cache.
    """
    if not isinstance(model.attn, __import__("bdh_linear").LinearAttention):
        raise ValueError("synapse tracing requires the bdh-linear (state-space) model")
    traces = {s: [] for s in synapses}
    state = None
    for t in range(idx.size(1)):
        _, _, state = model(idx[:, t: t + 1], state=state)
        for s in synapses:
            layer, h, ii, jj = s
            traces[s].append(state["layers"][layer][0, h, ii, jj].item())
    return traces


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str):
    from bdh import BDHConfig
    from bdh_linear import BDHLinear
    from pipeline.merge import load_ckpt

    ckpt = load_ckpt(ckpt_path)
    cfg_dict = ckpt["cfg"]
    bdh_cfg = BDHConfig(**{k: v for k, v in cfg_dict.items()
                           if k in {f.name for f in BDHConfig.__dataclass_fields__.values()}})
    if cfg_dict.get("model") == "bdh-linear":
        model = BDHLinear(bdh_cfg)
    else:
        model = BDH(bdh_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BDH interpretability analyses")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sparsity", action="store_true", help="activation sparsity over a sample batch")
    parser.add_argument("--graph", action="store_true", help="neuron-graph statistics")
    parser.add_argument("--head", type=int, default=0, help="head for --graph")
    parser.add_argument("--beta", type=float, default=1.0, help="threshold for --graph")
    parser.add_argument("--synapse", action="append", default=[],
                        help="synapse to trace as layer:head:i:j (repeatable)")
    parser.add_argument("--prompt", type=str, default="To be or not to be")
    args = parser.parse_args(argv)

    model, cfg = _load_model(args.ckpt)
    print(f"checkpoint: {args.ckpt} | model: {cfg.get('model')} | params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    if args.sparsity:
        torch.manual_seed(1337)
        idx = torch.randint(0, 256, (4, min(cfg["block_size"], 256)))
        res = activation_sparsity(model, idx)
        print(f"\n== activation sparsity (mean x: {res['mean_x']:.3f}, "
              f"mean xy: {res['mean_xy']:.3f}) ==")
        for s in res["per_layer"]:
            print(f"  layer {s['layer']}: x {s['x_sparsity']:.3f} | xy {s['xy_sparsity']:.3f}")

    if args.graph:
        print(f"\n== neuron-neuron graph (head {args.head}, beta {args.beta}) ==")
        rep = graph_report(model, args.head, args.beta)
        for k, v in rep.items():
            print(f"  {k}: {v}")

    if args.synapse:
        synapses = []
        for spec in args.synapse:
            l, h, i, j = (int(p) for p in spec.split(":"))
            synapses.append((l, h, i, j))
        idx = torch.tensor(bytearray(args.prompt, "utf-8"), dtype=torch.long).unsqueeze(0)
        traces = synapse_trace(model, idx, synapses)
        print(f"\n== synapse traces over {idx.size(1)} tokens ==")
        for s, vals in traces.items():
            print(f"  layer={s[0]} head={s[1]} synapse=({s[2]},{s[3]}): "
                  f"{['%.3f' % v for v in vals]}")

    if not (args.sparsity or args.graph or args.synapse):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
