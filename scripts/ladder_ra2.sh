#!/usr/bin/env bash
# Route-aware ladder R3 (primary arm, alpha=0.9, unfrozen attention).
# 20 phases, base mult=128, grow_mult=32 -> final mult=736 (554M).
# Fixes vs v1 PoC (per review cb72631):
#   - attention unfrozen (--no-freeze-attn)
#   - route_alpha=0.9 (prefix fraction; 10% full mix, the registered setting)
#   - warmup 1000, cosine decay over 10k (plan sec.3.1)
#   - P-Det at scale: 19-way boundary grid at milestones 5/10/15/20
#   - confusion matrices committed alongside this script
# Protocol: 30MB/phase, 10k steps, fresh optimizer via --init-from.
# Batch: 4 for mult<=192 (phases 1-3), 1 beyond (OOM guard).
# Language sequence matches Arm G/R for direct cross-arm comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

GROW=32
BASE_MULT=128
LOG=out/logs
A="${LOG}/ladder_ra2_analysis.txt"
rm -f "$A"
mkdir -p "$LOG"

SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt"
INIT=""
PHASE=0

# --- Precomputed domain string for eval_router.py ---
DOMAINS=""
for L in bg cs da de el en es et fi fr hu it lt nl pl pt ro sk sl; do
  [ -n "$DOMAINS" ] && DOMAINS="${DOMAINS},"
  DOMAINS="${DOMAINS}${L}:data/europarl/europarl-v7.${L}-en.${L}.txt"
done

for D in $SEQ; do
  PHASE=$((PHASE + 1))

  # Phase 1 (en): train from scratch, no growth, no route-aware
  if [ "$PHASE" -eq 1 ]; then
    echo "== phase $PHASE: $D (base mult=${BASE_MULT}, no growth) ==" >> "$A"
    .venv/bin/python -m pipeline.run train \
      --model bdh --dataset europarl \
      --europarl-langs "$D" --europarl-lang-mb 30 \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier "$BASE_MULT" \
      --block-size 512 --max-iters 10000 --batch-size 4 \
      --warmup-iters 1000 --lr-decay-iters 10000 \
      --run-name "ladRA2-$D" \
      2>&1 | tee "${LOG}/ladRA2_${D}.log"
    INIT="out/bdh_europarl_ladRA2-${D}_best.pt"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" 4 >> "$A" 2>&1
    continue
  fi

  # Phases 2+: growth + route-aware, unfrozen attention
  PREV_MULT=$((BASE_MULT + (PHASE - 2) * GROW))
  if [ "$PREV_MULT" -le 192 ]; then
    BS=4
  else
    BS=1
  fi

  echo "== phase $PHASE: $D (grow from mult $PREV_MULT -> $((PREV_MULT + GROW)), batch=$BS, alpha=0.9) ==" >> "$A"

  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "$D" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size "$BS" \
    --warmup-iters 1000 --lr-decay-iters 10000 \
    --grow-mult "$GROW" --init-from "$INIT" \
    --no-freeze-attn \
    --route-aware --route-alpha 0.9 \
    --run-name "ladRA2-$D" \
    2>&1 | tee "${LOG}/ladRA2_${D}.log"

  INIT="out/bdh_europarl_ladRA2-${D}_best.pt"

  # Per-language eval (same batch as training to avoid OOM)
  .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" "$BS" >> "$A" 2>&1

  # Routing diagnosis: full 20-way domain grid at every phase
  echo "--- routing diagnosis phase $PHASE ($D) ---" >> "$A"
  .venv/bin/python scripts/eval_router.py "$INIT" \
    --routes $(seq -s, "$BASE_MULT" "$GROW" "$((BASE_MULT + PHASE * GROW))") \
    --domains "$DOMAINS" \
    --batch "$BS" >> "${LOG}/ladRA2_routdiag_p${PHASE}.txt" 2>&1

  # Milestones: full interference matrix + 19-way boundary grid at phases 5/10/15/20
  if [ "$PHASE" -eq 5 ] || [ "$PHASE" -eq 10 ] || [ "$PHASE" -eq 15 ] || [ "$PHASE" -eq 20 ]; then
    SEEN="en"
    for S in $SEQ; do
      [ "$S" = "$D" ] && break
      SEEN="$SEEN,$S"
    done
    SEEN="$SEEN,$D"
    echo "--- milestone phase $PHASE: evaluating on $SEEN ---" >> "$A"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$SEEN" "$BS" >> "$A" 2>&1

    # 19-way boundary grid (one route per block edge)
    BOUNDARY_ROUTES=""
    for ((i=1; i<PHASE; i++)); do
      R=$((BASE_MULT + i * GROW))
      [ -n "$BOUNDARY_ROUTES" ] && BOUNDARY_ROUTES="${BOUNDARY_ROUTES},"
      BOUNDARY_ROUTES="${BOUNDARY_ROUTES}${R}"
    done
    FINAL_R=$((BASE_MULT + PHASE * GROW))
    BOUNDARY_ROUTES="${BOUNDARY_ROUTES},${FINAL_R}"
    echo "--- 19-way boundary grid phase $PHASE: routes=$BOUNDARY_ROUTES ---" >> "$A"
    .venv/bin/python scripts/eval_router.py "$INIT" \
      --routes "$BOUNDARY_ROUTES" \
      --domains "$DOMAINS" \
      --batch "$BS" >> "${LOG}/ladRA2_boundary_p${PHASE}.txt" 2>&1
    echo "  (boundary grid saved to ${LOG}/ladRA2_boundary_p${PHASE}.txt)" >> "$A"
  fi
done

echo "ladder-RA2-done" >> "${LOG}/ladder_watcher.log"
echo "Route-aware ladder R3 complete. Results in $A"
