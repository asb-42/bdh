"""CLI entrypoint for the BDH training/evaluation pipeline.

Usage:
    python -m pipeline.run train --model bdh --dataset shakespeare [--max-iters N ...]
    python -m pipeline.run eval  --model bdh --dataset shakespeare [--tag best]
    python -m pipeline.run bench --model bdh [--batch-size 32 --block-size 128]
    python -m pipeline.run merge --ckpt-a out/a_best.pt --ckpt-b out/b_best.pt --out out/merged.pt
    python -m pipeline.run analyze --ckpt out/bdh_best.pt --sparsity | --graph | --synapse l:h:i:j
"""

import argparse
import sys

from pipeline.bench import bench
from pipeline.config import parse_args
from pipeline.eval import evaluate
from pipeline.train import train


def _merge_cli(argv: list[str]) -> int:
    from pipeline.merge import run_merge

    parser = argparse.ArgumentParser(description="Merge two BDH checkpoints along the neuron dimension")
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run_merge(args.ckpt_a, args.ckpt_b, args.out)
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval", "bench", "merge", "analyze"):
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "merge":
        return _merge_cli(sys.argv[2:])
    if cmd == "analyze":
        from pipeline.analyze import main as analyze_main
        return analyze_main(sys.argv[2:])
    cfg = parse_args(sys.argv[2:])
    if cmd == "train":
        train(cfg)
    elif cmd == "eval":
        evaluate(cfg, tag=cfg.tag, num_samples=cfg.num_samples, max_new_tokens=cfg.max_new_tokens)
    else:
        bench(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
