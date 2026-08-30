#!/usr/bin/env bash
# Route-aware ladder R3 (primary arm, alpha=0.9, unfrozen attention).
# 20 phases, base mult=128, grow_mult=32 -> final mult=736 (579,123,200 params,
# checkpoint-verified 2026-08-30; see docs/reports/2026-08-30_pi-checkpoint-measurement.md).
# Fixes vs cb48f62 (per Quinn review 2026-08-30, blocking before GPU start):
#   - ROUTES UNITS: eval_router.py masks neuron prefixes PER HEAD (n/head =
#     64 x mult; config.py: n = mult * d // nh, d=512, nh=8). cb48f62 passed raw
#     multiplier values (128..736) -> prefixes of 0.5-6% of the model, identical
#     collapsed routing (the v1 failure mode). Now scaled x64.
#   - DOMAINS: sv was missing (19 of 20) -- the same dropped-row defect that
#     produced the report's x708. Restored; 20 domains now.
#   - PHANTOM ROUTE: grid end was BASE+PHASE*GROW (768 at phase 20); actual
#     width after phase p is BASE+(p-1)*GROW (736). Fixed. Boundary/milestone
#     grids are now boundary-aligned: one route per expert incl. base (p routes
#     at phase p) -- the pre-registered P-Det instrument. Note: ALL previous
#     grids (GR, PoC v1/v2) were evenly-spaced linspaces, not block edges, so
#     historical P-Route numbers are not boundary-comparable.
#   - INIT chain: _best.pt -> _last.pt for cross-arm congruence with Arm G/GR.
#   - MILESTONE SEEN: cb48f62 seeded SEEN="en" then iterated from the list
#     start -> duplicated en in every milestone matrix. Deduplicated.
# Protocol: 30MB/phase, 10k steps, fresh optimizer via --init-from.
# Batch: 4 for mult<=192 (phases 1-3), 1 beyond (OOM guard).
# Language sequence matches Arm G/R for direct cross-arm comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

GROW=32
BASE_MULT=128
NPH=64   # neurons per head per multiplier unit (n = mult * n_embd // n_head)
LOG=out/logs
A="${LOG}/ladder_ra2_analysis.txt"
rm -f "$A"
mkdir -p "$LOG"

SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt"
INIT=""
PHASE=0

# --- Precomputed domain string for eval_router.py (20 languages, incl. sv) ---
DOMAINS=""
for L in bg cs da de el en es et fi fr hu it lt nl pl pt ro sk sl sv; do
  [ -n "$DOMAINS" ] && DOMAINS="${DOMAINS},"
  DOMAINS="${DOMAINS}${L}:data/europarl/europarl-v7.${L}-en.${L}.txt"
done

for D in $SEQ; do
  PHASE=$((PHASE + 1))

  # Phase 1 (en): train from scratch, no growth, no route-aware
  if [ "$PHASE" -eq 1 ]; then
    echo "== phase $PHASE: $D (base mult=${BASE_MULT}, no growth) ==" >> "$A"
    .venv/bin/python -m pipeline.run train  \
      --model bdh --dataset europarl  \
      --europarl-langs "$D" --europarl-lang-mb 30  \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier "$BASE_MULT"  \
      --block-size 512 --max-iters 10000 --batch-size 4  \
      --warmup-iters 1000 --lr-decay-iters 10000  \
      --run-name "ladRA2-$D"  \
      2>&1 | tee "${LOG}/ladRA2_${D}.log"
    INIT="out/bdh_europarl_ladRA2-${D}_last.pt"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" 4 >> "$A" 2>&1
    continue
  fi

  # Phases 2+: growth + route-aware, unfrozen attention
  PREV_MULT=$((BASE_MULT + (PHASE - 2) * GROW))
  CUR_MULT=$((BASE_MULT + (PHASE - 1) * GROW))
  if [ "$PREV_MULT" -le 192 ]; then
    BS=4
  else
    BS=1
  fi

  echo "== phase $PHASE: $D (grow from mult $PREV_MULT -> $CUR_MULT, batch=$BS, alpha=0.9) ==" >> "$A"

  .venv/bin/python -m pipeline.run train \n    --model bdh --dataset europarl \n    --europarl-langs "$D" --europarl-lang-mb 30 \n    --n-embd 512 --n-head 8 \n    --block-size 512 --max-iters 10000 --batch-size "$BS" \n    --warmup-iters 1000 --lr-decay-iters 10000 \n    --grow-mult "$GROW" --init-from "$INIT" \n    --no-freeze-attn \n    --route-aware --route-alpha 0.9 \n    --run-name "ladRA2-$D" \n    2>&1 | tee "${LOG}/ladRA2_${D}.log"

  INIT="out/bdh_europarl_ladRA2-${D}_last.pt"

  # Per-language eval (same batch as training to avoid OOM)
  .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" "$BS" >> "$A" 2>&1

  # Routing diagnosis: full 20-way domain grid at every phase.
  # Routes are neuron prefixes PER HEAD: 64 x mult (see NPH above).
  echo "--- routing diagnosis phase $PHASE ($D) ---" >> "$A"
  .venv/bin/python scripts/eval_router.py "$INIT" \n    --routes $(seq -s, $((BASE_MULT * NPH)) $((GROW * NPH)) $((CUR_MULT * NPH))) \n    --domains "$DOMAINS" \n    --batch "$BS" >> "${LOG}/ladRA2_routdiag_p${PHASE}.txt" 2>&1

  # Milestones: full interference matrix + 19-way boundary grid at phases 5/10/15/20
  if [ "$PHASE" -eq 5 ] || [ "$PHASE" -eq 10 ] || [ "$PHASE" -eq 15 ] || [ "$PHASE" -eq 20 ]; then
    SEEN=""
    for S in $SEQ; do
      [ "$S" = "$D" ] && break
      SEEN="${SEEN:+$SEEN,}$S"
    done
    SEEN="${SEEN:+$SEEN,}$D"
    echo "--- milestone phase $PHASE: evaluating on $SEEN ---" >> "$A"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$SEEN" "$BS" >> "$A" 2>&1

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

echo "ladder-RA2-done" >> "${LOG}/ladder_watcher.log"
echo "Route-aware ladder R3 complete. Results in $A"