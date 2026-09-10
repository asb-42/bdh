#!/usr/bin/env bash
# Fixed-capacity sequence matrix (FCS): the operator's #130 experiment.
# 20 languages sequentially overwrite the SAME fixed-capacity weights (mult 128,
# ~100M, no growth, no route-awareness, no masks, no replay). After every
# phase, all 20 domains are cold-evaluated -> a 21x20 ppl matrix in
# out/logs/fixedcap_matrix.txt (rows = training prefixes, cols = domains).
#
# Pre-registered protocol + predictions: docs/plans/2026-09-10_fixed-capacity-matrix.md
#   P-FCS-1: theory-predicted forgetting catastrophe (early languages -> tens of ppl)
#   P-FCS-2: acquisition floor stays RA2b-comparable (no saturation cliff at 100M)
#   P-FCS-3: backward interference dominated by most-recent phase (recency)
#
# Phase 2..20 use --init-from WITHOUT --grow-mult: the existing weights-restore
# path (pipeline/train.py:117-120) - fresh optimizer, same capacity. No masked
# path exists, so the F-decay fix is irrelevant here (wd acts uniformly on all
# weights = the standard fixed-capacity regime).
#
# Host: bdh-4090 (.200), compiled (TORCHDYNAMO enabled by default on .200 venv).
# Est: ~12-20 min/phase training + ~6-10 min/phase eval -> ~7-10 h total.
set -euo pipefail
cd "$(dirname "$0")/.."

SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt"
ALL_LANGS="bg,cs,da,de,el,en,es,et,fi,fr,hu,it,lt,nl,pl,pt,ro,sk,sl,sv"
LOG=out/logs
MATRIX="${LOG}/fixedcap_matrix.txt"
mkdir -p "$LOG"

PHASE=0
INIT=""

for D in $SEQ; do
  PHASE=$((PHASE + 1))

  if [ "$PHASE" -eq 1 ]; then
    echo "== FCS phase 1: $D (fresh, fixed mult=128) ==" | tee -a "$MATRIX"
    .venv/bin/python -m pipeline.run train \
      --model bdh --dataset europarl \
      --europarl-langs "$D" --europarl-lang-mb 30 \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier 128 \
      --block-size 512 --max-iters 10000 --batch-size 4 \
      --warmup-iters 1000 --lr-decay-iters 10000 \
      --run-name "ladFCS-$D" \
      2>&1 | tee "${LOG}/fixedcap_${D}.log"
  else
    echo "== FCS phase $PHASE: $D (fixed mult=128, init-from $INIT, fresh optimizer) ==" | tee -a "$MATRIX"
    .venv/bin/python -m pipeline.run train \
      --model bdh --dataset europarl \
      --europarl-langs "$D" --europarl-lang-mb 30 \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier 128 \
      --block-size 512 --max-iters 10000 --batch-size 4 \
      --warmup-iters 1000 --lr-decay-iters 10000 \
      --init-from "$INIT" \
      --run-name "ladFCS-$D" \
      2>&1 | tee "${LOG}/fixedcap_${D}.log"
  fi

  INIT="out/bdh_europarl_ladFCS-${D}_last.pt"

  # matrix row: cold-eval ALL 20 domains on the post-phase checkpoint
  {
    echo "--- matrix row $PHASE (trained: $D) ---"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$ALL_LANGS" 4
  } >> "$MATRIX" 2>&1

done

echo "fixed-capacity-sequence-done $(date '+%F %T')" >> "${LOG}/ladder_watcher.log"
echo "FCS complete. Matrix: $MATRIX"
