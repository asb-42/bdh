"""Per-position loss over long sequential streams (paper Sec. 4.1 / handover 7.4).

Feeds one long byte-stream through the model block by block with carried
attention state (no window limit for bdh-linear), and reports mean per-token
cross-entropy bucketed by distance into the stream.

Usage:
    python scripts/position_curves.py --ckpt out/x.pt --tokens 32768 [--bucket 4096]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.analyze import _load_model
from pipeline.config import Config, resolve_device
from pipeline.data import load_dataset


@torch.no_grad()
def position_curve(model, tokens: np.ndarray, device, ctx, block: int = 512, carry: bool = True):
    model.eval()
    state = None
    losses = np.full(len(tokens), np.nan)
    for start in range(0, len(tokens) - block, block):
        x = torch.from_numpy(tokens[start : start + block].astype("int64")).unsqueeze(0).to(device)
        s = state if carry else None
        with ctx:
            logits, _, new_state = model(x, state=s)
        if carry:
            state = new_state
        ce = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="none")
        losses[start : start + block - 1] = ce.cpu().numpy()
    return losses


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--bucket", type=int, default=4096)
    p.add_argument("--offset", type=int, default=100000)
    args = p.parse_args()

    model, cfgd = _load_model(args.ckpt)
    if torch.cuda.is_available():
        # torch 2.x CPU bf16 matmul is pathologically slow (~300x fp32); any
        # autocast forward analysis must run on GPU.
        model = model.to("cuda")
    device = next(model.parameters()).device
    data = load_dataset(Config(dataset=cfgd.get("dataset", "wikitext2"), data_dir="data"))
    stream = np.frombuffer(
        data.test.tobytes()[args.offset : args.offset + args.tokens + 1], dtype=np.uint8
    )
    ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    warm = position_curve(model, stream[: args.tokens], device, ctx, cfgd["block_size"], carry=True)
    cold = position_curve(model, stream[: args.tokens], device, ctx, cfgd["block_size"], carry=False)
    print(f"ckpt={args.ckpt} | {args.tokens} tokens | buckets of {args.bucket}")
    print("  bucket            cold    warm   delta(warm-cold)")
    for b0 in range(0, args.tokens, args.bucket):
        w = warm[b0 : b0 + args.bucket]; c = cold[b0 : b0 + args.bucket]
        wm = float(np.nanmean(w)); cm = float(np.nanmean(c))
        bar = "#" * int(max(1, min(100, (cm - wm) * 100)))
        print(f"  {b0:6d}-{b0 + args.bucket:6d}: {cm:.4f} {wm:.4f} {cm - wm:+.4f} {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
