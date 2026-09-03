#!/usr/bin/env bash
# Route-aware ladder R3b (RA2b): full 20-phase re-run with the F-decay-leak
# fix in train.py (bit-exact frozen path via step-end restore).
#
# Purpose (pre-registered H-decay, operator-approved 2026-09-03): the RA2
# ladder's old-segment weights decayed by prod(1 - lr_t*wd) = 0.57978/phase
# (AdamW decoupled decay bypassed the zero-grad mask). RA2b reruns the same
# protocol with the fix; pre-registered predictions:
#   H-decay-1: old-segment weights are bit-identical to their creation-time
#              state at every later phase (bit test in verify step).
#   H-decay-2: joint-serving degradation of early languages shrinks
#              dramatically vs RA2 (RA2: bg 1649 / el 891 at p20 milestone).
#   H-decay-3: within-family acquisition gaps (Appendix A of the RA2 report)
#              shrink markedly if decay erosion was their driver.
# Differences vs ladder_ra2.sh: run-name ladRA2b (no checkpoint collisions),
# logs ladRA2b_*, analysis file ladder_ra2b_analysis.txt (RA2 artifacts are
# never touched), and nothing else -- protocol congruence is the point.
set -euo pipefail
cd "$(dirname "$0")/.."

export TORCHDYNAMO_DISABLE=1

GROW=32
BASE_MULT=128
NPH=64
LOG=out/logs
A="${LOG}/ladder_ra2b_analysis.txt"
mkdir -p "$LOG"

SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt"
INIT=""
PHASE=0

DOMAINS=""
for L in bg cs da de el en es et fi fr hu it lt nl pl pt ro sk sl sv; do
  [ -n "$DOMAINS" ] && DOMAINS="${DOMAINS},"
  DOMAINS="${DOMAINS}${L}:data/europarl/europarl-v7.${L}-en.${L}.txt"
done

for D in $SEQ; do
  PHASE=$((PHASE + 1))

  if [ "$PHASE" -eq 1 ]; then
    echo "== phase $PHASE: $D (base mult=${BASE_MULT}, no growth) ==" >> "$A"
    .venv/bin/python -m pipeline.run train \
      --model bdh --dataset europarl \
      --europarl-langs "$D" --europarl-lang-mb 30 \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier "$BASE_MULT" \
      --block-size 512 --max-iters 10000 --batch-size 4 \
      --warmup-iters 1000 --lr-decay-iters 10000 \
      --run-name "ladRA2b-$D" \
      2>&1 | tee "${LOG}/ladRA2b_${D}.log"
    INIT="out/bdh_europarl_ladRA2b-${D}_last.pt"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" 4 >> "$A" 2>&1
    continue
  fi

  PREV_MULT=$((BASE_MULT + (PHASE - 2) * GROW))
  CUR_MULT=$((BASE_MULT + (PHASE - 1) * GROW))
  if [ "$PREV_MULT" -le 192 ]; then BS=4; else BS=1; fi

  echo "== phase $PHASE: $D (grow from mult $PREV_MULT -> $CUR_MULT, batch=$BS, alpha=0.9) ==" >> "$A"

  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "$D" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size "$BS" \
    --warmup-iters 1000 --lr-decay-iters 10000 \
    --grow-mult "$GROW" --init-from "$INIT" \
    --no-freeze-attn \
    --route-aware --route-alpha 0.9 \
    --run-name "ladRA2b-$D" \
    2>&1 | tee "${LOG}/ladRA2b_${D}.log"

  INIT="out/bdh_europarl_ladRA2b-${D}_last.pt"
  .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" "$BS" >> "$A" 2>&1

  echo "--- routing diagnosis phase $PHASE ($D) ---" >> "$A"
  .venv/bin/python scripts/eval_router.py "$INIT" \
    --routes $(seq -s, $((BASE_MULT * NPH)) $((GROW * NPH)) $((CUR_MULT * NPH))) \
    --domains "$DOMAINS" \
    --batch "$BS" >> "${LOG}/ladRA2b_routdiag_p${PHASE}.txt" 2>&1

  if [ "$PHASE" -eq 5 ] || [ "$PHASE" -eq 10 ] || [ "$PHASE" -eq 15 ] || [ "$PHASE" -eq 20 ]; then
    SEEN=""
    for S in $SEQ; do
      [ "$S" = "$D" ] && break
      SEEN="${SEEN:+$SEEN,}$S"
    done
    SEEN="${SEEN:+$SEEN,}$D"
    echo "--- milestone phase $PHASE: evaluating on $SEEN ---" >> "$A"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$SEEN" "$BS" >> "$A" 2>&1

    BOUNDARY_ROUTES=""
    for ((i=0; i<PHASE; i++)); do
      R=$(( (BASE_MULT + i * GROW) * NPH ))
      [ -n "$BOUNDARY_ROUTES" ] && BOUNDARY_ROUTES="${BOUNDARY_ROUTES},"
      BOUNDARY_ROUTES="${BOUNDARY_ROUTES}${R}"
    done
    echo "--- boundary grid (one route per expert) phase $PHASE: routes=$BOUNDARY_ROUTES ---" >> "$A"
    .venv/bin/python scripts/eval_router.py "$INIT"  \
      --routes "$BOUNDARY_ROUTES"  \
      --domains "$DOMAINS"  \
      --batch "$BS" >> "${LOG}/ladRA2b_boundary_p${PHASE}.txt" 2>&1
    echo "  (boundary grid saved to ${LOG}/ladRA2b_boundary_p${PHASE}.txt)" >> "$A"
  fi
done

echo "ladder-RA2b-done $(date '+%F %T')" >> "${LOG}/ladder_watcher.log"
echo "Route-aware ladder R3b complete. Results in $A"
