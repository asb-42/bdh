"""Mechanism F test: merge-across-phases recovery of CL forgetting.

Chain-merges sequential phase checkpoints (neuron-dim concatenation, no finetune)
into one wider model, then evaluates per-language held-out loss with the same
protocol as scripts/lang_eval.py.

Usage:
    python scripts/merge_phases.py <out.pt> <ckpt1> <ckpt2> [ckpt3...] [--mb 30] [--iters 100] [--no-eval]
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from pipeline.merge import load_ckpt, merge_checkpoints
from pipeline.data import _europarl_blocks


def main():
    argv = sys.argv[1:]
    skip = {"--mb", "--iters", "--batch"}
    args, _skip_next = [], False
    for a in argv:
        if _skip_next:
            _skip_next = False
            continue
        if a in skip:
            _skip_next = True
            continue
        if a.startswith("--"):
            continue
        args.append(a)
    flags = {a for a in argv if a.startswith("--")}
    mb = int(sys.argv[sys.argv.index("--mb") + 1]) if "--mb" in sys.argv else 30
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 100
    batch_ov = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else None
    out_path, paths = args[0], args[1:]

    acc = None
    for p in paths:
        ckpt = load_ckpt(p)
        ckpt["source"] = p
        acc = ckpt if acc is None else merge_checkpoints(acc, ckpt)
        print(f"folded: {p}")
    torch.save(acc, out_path)

    from bdh import BDH, BDHConfig

    import dataclasses

    cfg = BDHConfig(**{k: v for k, v in acc["cfg"].items()
                       if k in {f.name for f in dataclasses.fields(BDHConfig)}})
    model = BDH(cfg)
    model.load_state_dict(acc["model_state"])
    print(f"merged -> {out_path} | neurons multiplier x{cfg.mlp_internal_dim_multiplier}")
    if "--no-eval" in flags:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    bs = cfg.block_size
    batch = batch_ov or (min(8, max(1, 8 * 512 // bs)) if device.type == "cuda" else 1)
    blocks = _europarl_blocks("data", mb * 1_000_000)
    print(f"eval block={bs} | random-crop cold eval (protocol-congruent)")
    for lang in sorted(blocks):
        raw = blocks[lang]["val"] + blocks[lang]["test"]
        data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
        g = torch.Generator().manual_seed(1234)
        losses = []
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
        with torch.no_grad(), amp:
            for _ in range(iters):
                ix = torch.randint(len(data) - bs - 1, (batch,), generator=g)
                x = torch.stack([data[i : i + bs] for i in ix]).to(device)
                y = torch.stack([data[i + 1 : i + 1 + bs] for i in ix]).to(device)
                logits, _, _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                losses.append(loss.item())
        m = sum(losses) / len(losses)
        print(f"  {lang}: nll {m:.4f} | ppl {math.exp(m):.2f}")


if __name__ == "__main__":
    main()
