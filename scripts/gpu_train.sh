#!/usr/bin/env bash
# Train BDH and a param-matched GPT baseline on wikitext-2 on the local GPU.
#
# Presets:
#   HARDWARE=4090  -> n_embd=256, ~25M params (the "official" BDH scale; fits 24GB)
#   HARDWARE=spark -> n_embd=512, ~101M params (128GB unified memory)
#
# COMPUTE-MATCHING: GPT is cheaper per step than BDH (see gpu_bench.sh). To compare at
# equal compute, run GPT with --max-iters scaled by the measured BDH/GPT ms ratio
# (e.g. if BDH is 3x slower/step, give GPT 3x the steps).
#
# Requires: python with torch (CUDA build), numpy, requests, pyarrow.
#
# Usage:
#   HARDWARE=4090  bash scripts/gpu_train.sh
#   HARDWARE=spark bash scripts/gpu_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

HARDWARE=${HARDWARE:-4090}
case "$HARDWARE" in
  4090)
    N_EMBD=256; N_LAYER=6; N_HEAD=8; MULT=128; BLOCK=512; BATCH=32; ITERS=10000 ;;
  spark)
    N_EMBD=512; N_LAYER=6; N_HEAD=8; MULT=128; BLOCK=1024; BATCH=32; ITERS=10000 ;;
  *)
    echo "HARDWARE must be 4090 or spark"; exit 1 ;;
esac

echo "== GPU train: hardware=$HARDWARE n_embd=$N_EMBD block=$BLOCK batch=$BATCH iters=$ITERS =="

for MODEL in bdh transformer; do
  echo "== training $MODEL =="
  python -m pipeline.run train \
    --model "$MODEL" --dataset wikitext2 \
    --n-embd "$N_EMBD" --n-head "$N_HEAD" --n-layer "$N_LAYER" \
    --mlp-internal-dim-multiplier "$MULT" \
    --block-size "$BLOCK" --batch-size "$BATCH" \
    --max-iters "$ITERS" --warmup-iters $((ITERS / 10)) --lr-decay-iters "$ITERS" \
    --eval-interval $((ITERS / 5)) --eval-iters 100 --log-interval $((ITERS / 5)) \
    --device auto --dtype auto
done

echo "== done. Evaluate with: python -m pipeline.run eval --model <m> --dataset wikitext2 --n-embd $N_EMBD ... --eval-iters 200 =="
