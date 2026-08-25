"""Per-domain held-out evaluation for textmix-trained checkpoints.

Usage: python scripts/domain_eval.py <ckpt> <name1:path1,name2:path2,...> [mb]
Evaluates each domain's val+test stream with the standard cold random-crop protocol.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from pipeline.analyze import _load_model


def main():
    ckpt = sys.argv[1]
    spec = sys.argv[2]
    mb = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    model, cfg = _load_model(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    bs = cfg["block_size"]
    batch, iters = 8, 100
    print(f"ckpt={ckpt} | block={bs} | random-crop cold eval (protocol-congruent)")
    for item in filter(None, map(str.strip, spec.split(","))):
        name, path = item.split(":", 1)
        with open(path, "rb") as f:
            raw = f.read((mb + 2) * 1_000_000)[mb * 1_000_000 :]
        data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
        g = torch.Generator().manual_seed(1234)
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
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
        print(f"  {name}: nll {m:.4f} | ppl {math.exp(m):.2f}")


if __name__ == "__main__":
    main()
