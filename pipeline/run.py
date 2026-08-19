"""CLI entrypoint for the BDH training/evaluation pipeline.

Usage:
    python -m pipeline.run train --model bdh --dataset shakespeare [--max-iters N ...]
    python -m pipeline.run eval  --model bdh --dataset shakespeare [--tag best]
"""

import sys

from pipeline.config import parse_args
from pipeline.eval import evaluate
from pipeline.train import train


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    cfg = parse_args(sys.argv[2:])
    if cmd == "train":
        train(cfg)
    else:
        evaluate(cfg, tag=cfg.tag, num_samples=cfg.num_samples, max_new_tokens=cfg.max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
