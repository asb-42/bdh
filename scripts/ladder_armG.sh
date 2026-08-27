#!/bin/bash
# Arm G: growth+routed ladder (20 phases, ×160→×736)
# Same language order as Arm R for direct comparison.
# Protocol: 30MB/phase, 10k steps, fresh optimizer via --init-from.
# Batch: 4 for mult≤192 (phases 1-2), 2 for mult>192 (phases 3+).
# --grow-mult 32 widens the latent by 32 neurons per phase.
# The growth code reads base_mult from init-from and adds grow_mult,
# so --mlp-internal-dim-multiplier is NOT passed to growth phases.
set -euo pipefail

A="out/logs/ladder_armG_analysis.txt"
rm -f "$A"

SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt"
INIT=""
PHASE=0

for D in $SEQ; do
  PHASE=$((PHASE + 1))

  # Phase 1 (en): train from scratch, no growth
  if [ "$PHASE" -eq 1 ]; then
    echo "== phase $PHASE: $D (base, no growth) ==" >> "$A"
    .venv/bin/python -m pipeline.run train \
      --model bdh --dataset europarl \
      --europarl-langs "$D" --europarl-lang-mb 30 \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier 128 \
      --block-size 512 --max-iters 10000 --batch-size 4 \
      --run-name ladG-$D \
      2>&1 | tee out/logs/ladG_${D}.log
    INIT="out/bdh_europarl_ladG-${D}_last.pt"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" 4 >> "$A" 2>&1
    continue
  fi

  # Phases 2+: growth from previous checkpoint
  # Batch policy: batch 4 for mult≤192 (phases 2-3), batch 1 beyond
  # At mult>192 the model is ~200M+ and batch 2 OOMs during backward
  PREV_MULT=$((128 + (PHASE - 2) * 32))
  if [ "$PREV_MULT" -le 192 ]; then
    BS=4
  else
    BS=1
  fi

  echo "== phase $PHASE: $D (grow from mult $PREV_MULT, batch=$BS) ==" >> "$A"

  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "$D" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size "$BS" \
    --grow-mult 32 --init-from "$INIT" \
    --run-name ladG-$D \
    2>&1 | tee out/logs/ladG_${D}.log

  INIT="out/bdh_europarl_ladG-${D}_last.pt"

  # Eval: target language (use same batch as training to avoid OOM)
  .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" "$BS" >> "$A" 2>&1

  # Milestone: full interference matrix at phases 5/10/15/20
  if [ "$PHASE" -eq 5 ] || [ "$PHASE" -eq 10 ] || [ "$PHASE" -eq 15 ] || [ "$PHASE" -eq 20 ]; then
    SEEN="en"
    for S in $SEQ; do
      [ "$S" = "$D" ] && break
      SEEN="$SEEN,$S"
    done
    SEEN="$SEEN,$D"
    echo "--- milestone phase $PHASE: evaluating on $SEEN ---" >> "$A"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$SEEN" "$BS" >> "$A" 2>&1
  fi
done

echo "ladder-G-done" >> "out/logs/ladder_watcher.log"
echo "Arm G complete. Results in $A"
