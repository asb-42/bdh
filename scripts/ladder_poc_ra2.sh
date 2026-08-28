#!/usr/bin/env bash
# Route-aware training PoC v2 (protocol-compliant redo).
# Fixes vs v1 (per review docs/reviews/2026-08-28_route-aware-poc-review.md):
#   1. attention UNFROZEN during each phase (--no-freeze-attn)
#   2. route_alpha = prefix fraction, swept 0.5 / 0.9 / 1.0 (monotonicity)
#   3. eval route grid spans the trained neuron range (phase blocks 1536/3584/5632)
# Pre-registered design: docs/plans/2026-08-28_route-aware-poc.md
# Plan compliance: warmup 1000, cosine decay over the run, fresh optimizer/phase,
# batch 4x512, +32 neurons/head per phase.
set -euo pipefail
cd "$(dirname "$0")/.."

GROW=32
LOG=out/logs
mkdir -p "$LOG"

# ---- Phase 1: EN base (shared across branches) ----
echo "=== Phase 1: EN base (warmup 1000, cosine) ==="
.venv/bin/python -m pipeline.run train \
  --model bdh --dataset europarl \
  --europarl-langs "en" --europarl-lang-mb 30 \
  --n-embd 512 --n-head 8 \
  --block-size 512 --max-iters 10000 --batch-size 4 \
  --warmup-iters 1000 --lr-decay-iters 10000 \
  --run-name "poc-ra2-p1" \
  2>&1 | tee "${LOG}/poc_ra2_p1.log"

P1="out/bdh_europarl_poc-ra2-p1_best.pt"

# ---- Specialists (trained alone on one language, same protocol) ----
for LANG in de es; do
  echo "=== Specialist: ${LANG} alone (mult 24, no growth) ==="
  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "${LANG}" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size 4 \
    --warmup-iters 1000 --lr-decay-iters 10000 \
    --run-name "poc-ra2-spec-${LANG}" \
    2>&1 | tee "${LOG}/poc_ra2_spec_${LANG}.log"
done

# ---- Branches: alpha in {0.5, 0.9, 1.0} ----
for ALPHA in 0.5 0.9 1.0; do
  TAG="a${ALPHA/./}"

  echo "=== Phase 2 (DE) alpha=${ALPHA} unfrozen-attn ==="
  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "de" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size 4 \
    --warmup-iters 1000 --lr-decay-iters 10000 \
    --grow-mult ${GROW} \
    --init-from "${P1}" \
    --no-freeze-attn \
    --route-aware --route-alpha "${ALPHA}" \
    --run-name "poc-ra2-${TAG}-p2" \
    2>&1 | tee "${LOG}/poc_ra2_${TAG}_p2.log"

  P2="out/bdh_europarl_poc-ra2-${TAG}-p2_best.pt"

  echo "=== Phase 3 (ES) alpha=${ALPHA} unfrozen-attn ==="
  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "es" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size 4 \
    --warmup-iters 1000 --lr-decay-iters 10000 \
    --grow-mult ${GROW} \
    --init-from "${P2}" \
    --no-freeze-attn \
    --route-aware --route-alpha "${ALPHA}" \
    --run-name "poc-ra2-${TAG}-p3" \
    2>&1 | tee "${LOG}/poc_ra2_${TAG}_p3.log"
done

echo "=== All branches done ==="
ls -la out/bdh_europarl_poc-ra2-*_best.pt
