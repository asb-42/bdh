"""Per-language held-out evaluation for Europarl-trained checkpoints.

Evaluates a checkpoint separately on the EN, DE, and ES held-out streams.
For a monolingually trained model, the non-training languages measure
zero-shot cross-lingual transfer (relevant to CL-plan H4 / INV-3 binding).

Usage: python scripts/lang_eval.py <ckpt> [lang_mb]
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks


def main():
    ckpt = sys.argv[1]
    lang_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    langs = tuple(s for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else ("en", "de", "es")
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    blocks = _europarl_blocks("data", lang_mb * 1_000_000, langs=langs)
    model, cfg = _load_model(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    bs = cfg["block_size"]
    iters = 100
    print(f"ckpt={ckpt} | block={bs} | random-crop cold eval (protocol-congruent)")
    for lang in sorted(blocks):
        raw = blocks[lang]["val"] + blocks[lang]["test"]
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
        print(f"  {lang}: nll {m:.4f} | ppl {math.exp(m):.2f}")


if __name__ == "__main__":
    main()
