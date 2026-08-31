#!/usr/bin/env bash
# Route-aware ladder R3 RESUME (phases 13-20) -- runs on gx10.
#
# Context: phases 1-12 were executed ad-hoc by MiMo on ai in the order
#   en es pl fr de nl it sv da pt cz ro   (cz == cs data, F-T3)
# established by three independent evidence classes (checklist addenda 1-2):
# log birth times (F-T1), MiMo's growth-header reconstruction, and the
# cfg.init_from chain inside the checkpoints (validated 2026-08-31, ALL-OK).
# el (phase 14 in script order) OOM'd on ai and left no checkpoint -> retried
# here at its script position.
#
# Operator-approved resume order (2026-08-31): script order for remaining
# languages: fi hu bg et el sk sl lt (phases 13-20, growth 480->512 ... 704->736).
# INIT = ro_last.pt (mult=480, 377,780,224 params incl. freq buffers, step=10000).
#
# Protocol identical to ladder_ra2.sh @HEAD: 30MB/phase, 10k steps, fresh
# optimizer via --init-from, route-aware alpha=0.9, unfrozen attention, BS=1
# (PREV_MULT >= 480 > 192). Routing diagnosis per phase; milestones at 15/20
# with boundary-aligned route grids (pre-registered P-Det instrument).
# SEEN at milestones: EXECUTED order for phases 1-12 + script order for 13-20
# (seam flagged in checklist addendum 2). Milestone/boundary language keys use
# cs (the actual data files); the cz checkpoint label is cosmetic.
#
# Single-instance lock: RA2 on ai died from a train/eval memory collision;
# flock prevents concurrent instances.
set -euo pipefail
cd "$(dirname "$0")/.."

exec 9>/tmp/ladder_ra2_resume.lock
flock -n 9 || { echo "ERROR: another ladder_ra2 instance is running"; exit 1; }

GROW=32
BASE_MULT=128
NPH=64   # neurons per head per multiplier unit (n = mult * n_embd // n_head)
PHASE=12 # completed phases; first resume phase is 13
INIT="out/bdh_europarl_ladRA2-ro_last.pt"

LOG=out/logs
A="${LOG}/ladder_ra2_analysis.txt"
mkdir -p "$LOG"

# preserve the failed el attempt log from ai (F-T1/F-T5 evidence)
if [ -f "${LOG}/ladRA2_el.log" ]; then
  mv "${LOG}/ladRA2_el.log" "${LOG}/ladRA2_el.attempt1-failed.log"
  echo "preserved failed el attempt as ${LOG}/ladRA2_el.attempt1-failed.log" >> "$A"
fi

SEQ_RESUME="fi hu bg et el sk sl lt"
# executed order phases 1-12 (cz mapped to cs per F-T3)
SEEN_BASE="en es pl fr de nl it sv da pt cs ro"

DOMAINS=""
for L in bg cs da de el en es et fi fr hu it lt nl pl pt ro sk sl sv; do
  [ -n "$DOMAINS" ] && DOMAINS="${DOMAINS},"
  DOMAINS="${DOMAINS}${L}:data/europarl/europarl-v7.${L}-en.${L}.txt"
done

echo "== resume start $(date '+%F %T') | INIT=$INIT | phases 13-20: $SEQ_RESUME ==" >> "$A"

for D in $SEQ_RESUME; do
  PHASE=$((PHASE + 1))
  PREV_MULT=$((BASE_MULT + (PHASE - 2) * GROW))
  CUR_MULT=$((BASE_MULT + (PHASE - 1) * GROW))
  BS=1

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
    --run-name "ladRA2-$D" \
    2>&1 | tee "${LOG}/ladRA2_${D}.log"

  INIT="out/bdh_europarl_ladRA2-${D}_last.pt"

  # Per-language eval (same batch as training)
  .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" "$BS" >> "$A" 2>&1

  # Routing diagnosis: full 20-way domain grid at every phase.
  # Routes are neuron prefixes PER HEAD: 64 x mult.
  echo "--- routing diagnosis phase $PHASE ($D) ---" >> "$A"
  .venv/bin/python scripts/eval_router.py "$INIT" \
    --routes $(seq -s, $((BASE_MULT * NPH)) $((GROW * NPH)) $((CUR_MULT * NPH))) \
    --domains "$DOMAINS" \
    --batch "$BS" >> "${LOG}/ladRA2_routdiag_p${PHASE}.txt" 2>&1

  # Milestones: interference matrix + boundary grid at phases 15/20
  if [ "$PHASE" -eq 15 ] || [ "$PHASE" -eq 20 ]; then
    SEEN_LIST="$SEEN_BASE"
    for S in $SEQ_RESUME; do
      [ "$S" = "$D" ] && break
      SEEN_LIST="$SEEN_LIST $S"
    done
    SEEN_CSV=$(echo "$SEEN_LIST" | tr ' ' ',')
    echo "--- milestone phase $PHASE: evaluating on $SEEN_CSV ---" >> "$A"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$SEEN_CSV" "$BS" >> "$A" 2>&1

    # Boundary grid: one route per expert seen so far, incl. the base edge
    # (i = 0..PHASE-1 -> p routes at phase p). Pre-registered P-Det definition:
    # a language is detected iff argmin lands on the route covering its block.
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
      --batch "$BS" >> "${LOG}/ladRA2_boundary_p${PHASE}.txt" 2>&1
    echo "  (boundary grid saved to ${LOG}/ladRA2_boundary_p${PHASE}.txt)" >> "$A"
  fi
done

echo "ladder-RA2-resume-done $(date '+%F %T')" >> "${LOG}/ladder_watcher.log"
echo "Route-aware ladder R3 resume complete. Results in $A"