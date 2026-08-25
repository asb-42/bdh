"""Post-hoc neuron pruning sweep for merged BDH models.

Ranks neurons by union-max activation importance (mean |xy| across depth levels)
computed per language, then evaluates per-language held-out loss at several
keep-fractions by zeroing pruned neurons' encoder/encoder_v columns and decoder
rows in memory (exact removal under BDH's per-neuron independence).

Usage:
    python scripts/prune_eval.py <ckpt> [--langs en,de,es] [--mb 10] \
        [--batches 40] [--iters 30] [--keeps 1.0,0.5,0.25,0.125] [--random] [--batch 4]
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from bdh import _k_sparse_relu
from pipeline.data import _europarl_blocks


@torch.no_grad()
def eval_langs(model, blocks, bs, iters, batch):
    device = next(model.parameters()).device
    out = {}
    for lang, blk in sorted(blocks.items()):
        raw = blk["val"] + blk["test"]
        data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
        g = torch.Generator().manual_seed(1234)
        losses = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for _ in range(iters):
                ix = torch.randint(len(data) - bs - 1, (batch,), generator=g)
                x = torch.stack([data[i : i + bs] for i in ix]).to(device)
                y = torch.stack([data[i + 1 : i + 1 + bs] for i in ix]).to(device)
                logits, _, _ = model(x)
                losses.append(torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)).item())
        m = sum(losses) / len(losses)
        out[lang] = m
        print(f"    {lang}: nll {m:.4f} | ppl {math.exp(m):.2f}")
    return out


@torch.no_grad()
def importance(model, data_np, bs, n_batches, ratio):
    device = next(model.parameters()).device
    nh, N = model.config.n_head, model.config.n_embd * model.config.mlp_internal_dim_multiplier // model.config.n_head
    acc = torch.zeros(nh, N, device=device)
    data = torch.from_numpy(np.frombuffer(data_np, dtype=np.uint8).astype(np.int64))
    g = torch.Generator().manual_seed(4321)
    act = (lambda t: _k_sparse_relu(t, ratio)) if ratio > 0 else torch.nn.functional.relu
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(n_batches):
            ix = torch.randint(len(data) - bs - 1, (8,), generator=g)
            idx = torch.stack([data[i : i + bs] for i in ix]).to(device)
            x = model.ln(model.embed(idx).unsqueeze(1))
            B, _, T, _ = x.shape
            for _lvl in range(model.config.n_layer):
                x_s = act(x @ model.encoder)
                yKV, _ = model.attn(x_s, x, state=None, pos_offset=0)
                y_s = act(model.ln(yKV) @ model.encoder_v)
                xy = x_s * y_s
                acc += xy.abs().float().mean(dim=(0, 1, 2))
                x = model.ln(x + model.ln(
                    xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder))
    return (acc / n_batches).flatten()


def main():
    a = sys.argv[1:]
    def arg(flag, default):
        return a[a.index(flag) + 1] if flag in a else default
    ckpt_path = a[0]
    langs = arg("--langs", "en,de,es").split(",")
    mb = int(arg("--mb", "10"))
    nb = int(arg("--batches", "40"))
    iters = int(arg("--iters", "30"))
    batch = int(arg("--batch", "4"))
    keeps = [float(k) for k in arg("--keeps", "1.0,0.5,0.25,0.125").split(",")]
    random_prune = "--random" in a

    from pipeline.analyze import _load_model
    model, cfg_d = _load_model(ckpt_path)
    device = torch.device("cuda")
    model = model.to(device).eval()
    C = model.config
    bs = cfg_d["block_size"]
    nh, D = C.n_head, C.n_embd
    N = C.mlp_internal_dim_multiplier * D // nh

    blocks = _europarl_blocks("data", mb * 1_000_000)
    blocks = {k: v for k, v in blocks.items() if k in langs}

    print(f"ranking neurons by union-max importance over {langs} ...")
    imp = None
    for lang in langs:
        i_l = importance(model, blocks[lang]["train"], bs, nb, getattr(C, "k_sparse_ratio", 0.0))
        imp = i_l if imp is None else torch.maximum(imp, i_l)

    enc_b, encv_b, dec_b = (t.clone() for t in
                            (model.encoder.data, model.encoder_v.data, model.decoder.data))

    print(f"=== {'RANDOM' if random_prune else 'IMPORTANCE'} prune sweep | {ckpt_path} "
          f"| {N * nh} neurons ===")
    for q in keeps:
        k = max(1, int(q * N))
        # keep top-k PER HEAD (importance flattened as (nh, N); rank within head)
        imp2 = imp.view(nh, N)
        ord2 = torch.argsort(imp2, dim=1, descending=True)
        if random_prune:
            g2 = torch.Generator().manual_seed(7)
            ord2 = torch.stack([torch.randperm(N, generator=g2).to(device) for _ in range(nh)])
        keep2 = torch.zeros(nh, N, dtype=torch.bool, device=device)
        keep2.scatter_(1, ord2[:, :k], True)
        flat = keep2.reshape(-1)

        with torch.no_grad():
            model.encoder.data.copy_(enc_b * flat.view(nh, 1, N))
            model.encoder_v.data.copy_(encv_b * flat.view(nh, 1, N))
            model.decoder.data.copy_(dec_b * flat.unsqueeze(1))
        print(f"  keep={q:.3f} ({k}/head)")
        eval_langs(model, blocks, bs, iters, batch)

    with torch.no_grad():
        model.encoder.data.copy_(enc_b)
        model.encoder_v.data.copy_(encv_b)
        model.decoder.data.copy_(dec_b)


if __name__ == "__main__":
    main()
