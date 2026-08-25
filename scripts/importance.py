"""Per-neuron importance extraction for Mechanism C write-gating.

Replays a language's training stream through a checkpoint and accumulates
mean |xy| per neuron (aggregated over depth levels — BDH shares one
encoder/encoder_v/decoder triple across levels). Saves a copy of the
checkpoint with an added "neuron_importance" (nh, N) float tensor.

Usage:
    python scripts/importance.py <out.pt> <src_ckpt> <lang> [--mb 30] [--batches 60]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from bdh import _k_sparse_relu
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks


@torch.no_grad()
def main():
    out_path, src, lang = sys.argv[1], sys.argv[2], sys.argv[3]

    def argval(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

    mb = int(argval("--mb", "30"))
    n_batches = int(argval("--batches", "60"))

    model, cfg = _load_model(src)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    C = model.config
    bs = C.block_size or cfg["block_size"]
    ratio = getattr(C, "k_sparse_ratio", 0.0)

    blocks = _europarl_blocks("data", mb * 1_000_000)
    raw = blocks[lang]["train"]
    data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
    g = torch.Generator().manual_seed(4321)
    nh, N = C.n_head, C.n_embd * C.mlp_internal_dim_multiplier // C.n_head
    acc = torch.zeros(nh, N, device=device, dtype=torch.float32)
    act = (lambda t: _k_sparse_relu(t, ratio)) if ratio > 0 else torch.nn.functional.relu

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(n_batches):
            ix = torch.randint(len(data) - bs - 1, (8,), generator=g)
            idx = torch.stack([data[i : i + bs] for i in ix]).to(device)
            x = model.ln(model.embed(idx).unsqueeze(1))
            B, _, T, _ = x.shape
            for _level in range(C.n_layer):
                # mirrors BDH.forward (dropout off in eval); xy is the neuron gate signal
                x_s = act(x @ model.encoder)                       # B, nh, T, N
                yKV, _ = model.attn(x_s, x, state=None, pos_offset=0)
                y_s = act(model.ln(yKV) @ model.encoder_v)
                xy = x_s * y_s
                acc += xy.abs().float().mean(dim=(0, 1, 2))
                y_mlp = model.ln(
                    xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
                )
                x = model.ln(x + y_mlp)

    imp = (acc / n_batches).cpu()
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt["neuron_importance"] = imp
    torch.save(ckpt, out_path)
    nz = (imp > 0.01 * imp.max()).float().mean()
    print(f"importance -> {out_path} | lang={lang} | shape {tuple(imp.shape)} | "
          f"neurons >1% max: {nz:.1%} | mean/max {imp.mean() / imp.max():.4f}")


if __name__ == "__main__":
    main()
