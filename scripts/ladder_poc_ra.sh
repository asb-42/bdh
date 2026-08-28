#!/usr/bin/env bash
# Route-aware training PoC: 3-phase EN→DE→ES with prefix-masked loss
# Pre-registered design: docs/plans/2026-08-28_route-aware-poc.md
set -euo pipefail
cd "$(dirname "$0")/.."

PHASES=("en" "de" "es")
GROW=32
RUN="poc-ra"
LOGDIR=out/logs
mkdir -p "$LOGDIR"

# Phase 1: EN base (no growth, no route-aware)
echo "=== Phase 1: EN base ==="
.venv/bin/python -m pipeline.run train \
  --model bdh --dataset europarl \
  --europarl-langs "en" --europarl-lang-mb 30 \
  --n-embd 512 --n-head 8 \
  --block-size 512 --max-iters 10000 --batch-size 4 \
  --run-name "${RUN}-p1" \
  2>&1 | tee "${LOGDIR}/${RUN}_p1.log"

# Phase 2: DE with growth + route-aware
echo "=== Phase 2: DE route-aware ==="
.venv/bin/python -m pipeline.run train \
  --model bdh --dataset europarl \
  --europarl-langs "de" --europarl-lang-mb 30 \
  --n-embd 512 --n-head 8 \
  --block-size 512 --max-iters 10000 --batch-size 4 \
  --grow-mult ${GROW} \
  --init-from "out/bdh_europarl_${RUN}-p1_best.pt" \
  --route-aware --route-alpha 0.1 \
  --run-name "${RUN}-p2" \
  2>&1 | tee "${LOGDIR}/${RUN}_p2.log"

# Phase 3: ES with growth + route-aware
echo "=== Phase 3: ES route-aware ==="
.venv/bin/python -m pipeline.run train \
  --model bdh --dataset europarl \
  --europarl-langs "es" --europarl-lang-mb 30 \
  --n-embd 512 --n-head 8 \
  --block-size 512 --max-iters 10000 --batch-size 4 \
  --grow-mult ${GROW} \
  --init-from "out/bdh_europarl_${RUN}-p2_best.pt" \
  --route-aware --route-alpha 0.1 \
  --run-name "${RUN}-p3" \
  2>&1 | tee "${LOGDIR}/${RUN}_p3.log"

echo "=== All phases done ==="
echo "Checkpoints:"
ls -la out/bdh_europarl_${RUN}-*_best.pt
