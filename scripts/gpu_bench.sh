#!/usr/bin/env bash
# Benchmark BDH vs BDHLinear vs GPT on the local GPU (RTX 4090 / DGX Spark).
# Establishes the per-step cost baseline (ms/step + FLOP/throughput) that decides
# the compute-matched step counts for the training run.
#
# Requires: python with torch (CUDA build), numpy, requests, pyarrow.
#
# Usage:
#   bash scripts/gpu_bench.sh                 # default 25M-param BDH (D=256)
#   N_EMBD=512 BLOCK=1024 bash scripts/gpu_bench.sh   # ~101M params (Spark scale)
set -euo pipefail
cd "$(dirname "$0")/.."

N_EMBD=${N_EMBD:-256}
N_HEAD=${N_HEAD:-8}
N_LAYER=${N_LAYER:-6}
MULT=${MULT:-128}
BLOCK=${BLOCK:-512}
BATCH=${BATCH:-32}

echo "== GPU benchmark: n_embd=$N_EMBD n_head=$N_HEAD n_layer=$N_LAYER mult=$MULT block=$BLOCK batch=$BATCH =="

for MODEL in bdh bdh-linear transformer; do
  python -m pipeline.run bench \
    --model "$MODEL" --dataset wikitext2 \
    --n-embd "$N_EMBD" --n-head "$N_HEAD" --n-layer "$N_LAYER" \
    --mlp-internal-dim-multiplier "$MULT" \
    --block-size "$BLOCK" --batch-size "$BATCH" \
    --device auto --dtype auto
done

echo "== done. Use the BDH/GPT ms-per-step ratio to set GPT's --max-iters for a compute-matched run. =="
