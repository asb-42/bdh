"""Mechanism G efficiency probe: distill routed experts into delta-gated single model.

Loads a grown BDH checkpoint, freezes everything, attaches per-suffix-neuron-block
gates g = sigma(w . x_t + b) (depth-shared, applied to decoder write-back of each
suffix neuron block), and trains ONLY the gate parameters to match prefix-routed
teacher logits on mixed replay batches (KD). Result: one-forward serving with
contextual isolation.

Usage:
    PYTHONPATH=. python scripts/gate_distill.py <ckpt> <out_ckpt> \
        --blocks 8192,10240 --langs en:de-en:en,... [--steps 3000] [--mb 3]
"""
import argparse
import copy
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--blocks", required=True,
                    help="comma cumulative neuron counts/head marking suffix block ends")
    ap.add_argument("--train", required=True,
                    help="comma list lang:pair:side for replay + teachers, e.g. en:de-en:en,...")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--mb", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    from pipeline.analyze import _load_model

    model, cfgd = _load_model(args.ckpt)
    device = torch.device("cuda")
    model = model.to(device).eval()
    C = model.config
    nh, D = C.n_head, C.n_embd
    bs = cfgd["block_size"]
    Nf = C.mlp_internal_dim_multiplier * D // nh
    ends = [int(b) for b in args.blocks.split(",")]
    starts = [0] + ends[:-1]

    enc_b = model.encoder.data.clone()
    encv_b = model.encoder_v.data.clone()
    dec_b = model.decoder.data.clone()

    def set_prefix(k):
        m = torch.zeros(Nf, device=device)
        m[:k] = 1.0
        with torch.no_grad():
            model.encoder.data.copy_(enc_b * m.view(1, 1, -1))
            model.encoder_v.data.copy_(encv_b * m.view(1, 1, -1))
            model.decoder.data.copy_(dec_b * m.repeat(nh).unsqueeze(1))

    langs = []
    for item in filter(None, map(str.strip, args.train.split(","))):
        lang, pair, side = item.split(":")
        arr = np.frombuffer(
            open(f"data/europarl/europarl-v7.{pair}.{side}", "rb").read(args.mb * 1_000_000),
            dtype=np.uint8)
        hi = len(arr) - bs - 1
        idx = torch.randint(hi, (args.steps * args.batch,), generator=torch.Generator().manual_seed(9))
        crops = np.stack([arr[int(i):int(i) + bs] for i in idx])
        langs.append((lang, torch.from_numpy(crops.astype(np.int64)),
                      len(langs)))  # route index == language order
    routes = ends + [Nf]  # language i's teacher = prefix through its own block; last = full

    print(f"precomputing teacher logits for {len(langs)} languages x {len(langs[0][1])} crops ...")
    teacher = {}
    for li, (lang, blk, ri) in enumerate(langs):
        set_prefix(routes[ri])
        outs = []
        amp = torch.autocast("cuda", dtype=torch.bfloat16)
        with torch.no_grad(), amp:
            for b0 in range(0, len(blk), 8):
                lg, _, _ = model(blk[b0:b0 + 8].to(device))
                outs.append(lg.float().cpu())
        teacher[lang] = torch.cat(outs)   # (Ncrops, bs, V) float32 cpu
    print("teachers ready.")

    # gates: one (w,b) per suffix block; depth-shared, token-wise, on residual x
    gate_w = torch.nn.Parameter(torch.zeros(len(ends), D, device=device))
    gate_b = torch.nn.Parameter(torch.full((len(ends),), -4.0, device=device))
    opt = torch.optim.AdamW([gate_w, gate_b], lr=args.lr)

    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    def gated_logits(blk_dev):
        x = model.ln(model.embed(blk_dev)).unsqueeze(1)
        B, _, T, _ = x.shape
        Nn = Nf
        for _lvl in range(C.n_layer):
            xs = torch.nn.functional.relu(x @ model.encoder)             # B,1,T,nh*Nn? -> (B,1,T,N*nh)? actual (B,nh?,T,N)
            # mirror bdh: xs shape is (B, nh, T, N) because E is (nh, D, N)
            yKV, _ = model.attn(xs, x, state=None, pos_offset=0)
            ys = torch.nn.functional.relu(model.ln(yKV) @ model.encoder_v)
            xy = xs * ys                                                  # B, nh, T, N
            # per-token gate per suffix block, broadcast over heads and block neurons
            g = torch.sigmoid(x.squeeze(1) @ gate_w.t() + gate_b)         # B, T, nblocks
            parts, prev = [], 0
            for bi, (s0, s1) in enumerate(zip(starts, ends)):
                if s0 > prev:
                    parts.append(xy[:, :, :, prev:s0])
                gb = g[:, None, :, bi].unsqueeze(-1).to(xy.dtype)
                parts.append(xy[:, :, :, s0:s1] * gb)
                prev = s1
            if Nn > prev:
                parts.append(xy[:, :, :, prev:Nn])
            xyg = torch.cat(parts, dim=-1)
            Bb, _, Tt, _ = x.shape
            y = model.ln(xyg.transpose(1, 2).reshape(Bb, 1, Tt, Nn * nh) @ model.decoder)
            x = model.ln(x + y)
        return x.view(B, T, D) @ model.lm_head

    t0 = __import__("time").time()
    for step in range(args.steps):
        li = step % len(langs)
        lang, blk, ri = langs[li]
        b0 = (step * args.batch) % (len(blk) - args.batch)
        xb = blk[b0:b0 + args.batch].to(device)
        tgt = teacher[lang][b0:b0 + args.batch].to(device)
        opt.zero_grad(set_to_none=True)
        with amp:
            sl = gated_logits(xb)
            loss = torch.nn.functional.kl_div(
                torch.log_softmax(sl, dim=-1), torch.log_softmax(tgt, dim=-1),
                log_target=True, reduction="batchmean")
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                gv = torch.sigmoid(gate_b).mean().item()
            print(f"step {step+1}: kd {loss.item():.4f} | mean gate bias-open {gv:.3f} "
                  f"| {(__import__('time').time()-t0):.0f}s")

    # restore full weights, persist gate params alongside
    with torch.no_grad():
        model.encoder.data.copy_(enc_b)
        model.encoder_v.data.copy_(encv_b)
        model.decoder.data.copy_(dec_b)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck["gate_state"] = {"w": gate_w.detach().cpu(), "b": gate_b.detach().cpu(),
                        "blocks": ends}
    torch.save(ck, args.out)
    print(f"gated model -> {args.out}")


if __name__ == "__main__":
    main()
