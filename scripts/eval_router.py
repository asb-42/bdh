"""Label-free likelihood routing over grown BDH stacks (prefix experts).

Each route (neuron-prefix mask) scores the EARLY positions of a block; arg-min
routes the LATE positions. Reports routing accuracy vs true domain plus
routed/oracle/joint perplexity on the served positions.

Usage:
    PYTHONPATH=. python scripts/eval_router.py <ckpt> \
        --routes 8192,10240,12288 --domains wiki:path,books:path,parl:path \
        [--window 128] [--crops 40] [--batch 4]
"""
import argparse
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--mb", type=int, default=30)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--crops", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from pipeline.analyze import _load_model

    model, cfg = _load_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    nh, D = model.config.n_head, model.config.n_embd
    N_full = model.config.mlp_internal_dim_multiplier * D // nh
    bs = cfg["block_size"]
    assert 0 < args.window < bs

    enc_b = model.encoder.data.clone()
    encv_b = model.encoder_v.data.clone()
    dec_b = model.decoder.data.clone()

    def set_prefix(k):
        m = torch.zeros(N_full, device=device)
        m[:k] = 1.0
        with torch.no_grad():
            model.encoder.data.copy_(enc_b * m.view(1, 1, -1))
            model.encoder_v.data.copy_(encv_b * m.view(1, 1, -1))
            model.decoder.data.copy_(dec_b * m.repeat(nh).unsqueeze(1))

    routes = [int(r) for r in args.routes.split(",")]
    doms = []
    g = torch.Generator().manual_seed(1234)
    for item in filter(None, map(str.strip, args.domains.split(","))):
        name, path = item.split(":", 1)
        arr = np.frombuffer(open(path, "rb").read(args.mb * 1_000_000)[-2_000_000:],
                            dtype=np.uint8)
        hi = len(arr) - bs - 1
        crops = np.stack([arr[int(i):int(i) + bs]
                          for i in torch.randint(hi, (args.crops,), generator=g)])
        doms.append((name, torch.from_numpy(crops.astype(np.int64))))

    print(f"router ckpt={args.ckpt} | routes={routes} n/head | window={args.window} tok "
          f"| {args.crops} crops/domain")

    names = [n for n, _ in doms]
    R = len(routes)
    conf = torch.zeros(len(doms), R, dtype=torch.int64)
    routed_ppl, oracle_ppl = {}, {}
    joint_losses = []

    for ti, (tname, blocks) in enumerate(doms):
        rl = torch.zeros(R, args.crops, bs)
        for ri, k in enumerate(routes):
            set_prefix(min(k, N_full))
            amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
            with torch.no_grad(), amp:
                for b0 in range(0, args.crops, args.batch):
                    xb = blocks[b0:b0 + args.batch].to(device)
                    yb = xb.roll(shifts=-1, dims=1)
                    logits, _, _ = model(xb)
                    l = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                        reduction="none").view(xb.shape[0], bs)
                    rl[ri, b0:b0 + args.batch] = l.float().cpu()

        scores = rl[:, :, : args.window].mean(dim=2)
        choice = scores.argmin(dim=0)
        served = rl[choice, torch.arange(args.crops), args.window:].mean(dim=1)
        oracle = rl[ti, :, args.window:].mean(dim=1)

        conf[ti] += torch.bincount(choice, minlength=R)
        routed_ppl[tname] = math.exp(served.mean().item())
        oracle_ppl[tname] = math.exp(oracle.mean().item())

        set_prefix(N_full)
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
        with torch.no_grad(), amp:
            for b0 in range(0, args.crops, args.batch):
                xb = blocks[b0:b0 + args.batch].to(device)
                yb = xb.roll(shifts=-1, dims=1)
                logits, _, _ = model(xb)
                l = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                    reduction="none").view(xb.shape[0], bs)
                joint_losses.extend(l[:, args.window:].float().mean(dim=1).cpu().tolist())

    print("\nconfusion (rows=true domain, cols=routed expert):")
    print("          " + "".join(f"{n:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>9} " + "".join(f"{conf[i, j].item():>9}" for j in range(R)))
    print(f"\n{'domain':>9} {'acc':>6} {'routed':>8} {'oracle':>8}")
    for n in names:
        hits = int(conf[names.index(n), names.index(n)])
        tot = int(conf[names.index(n)].sum())
        print(f"{n:>9} {hits / max(1, tot):>6.0%} "
              f"{routed_ppl[n]:>8.2f} {oracle_ppl[n]:>8.2f}")
    print(f"\njoint full-width reference: ppl {math.exp(sum(joint_losses)/len(joint_losses)):.2f}"
          f"  (served positions only)")


if __name__ == "__main__":
    main()
