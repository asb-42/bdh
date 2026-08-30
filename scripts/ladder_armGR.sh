#!/bin/bash
# Arm G+R: growth + frozen-attention ladder (20 phases, ×128→×736)
# (schedule checkpoint-verified 2026-08-30: en=128, sk=576, sv=608, ro=640, nl=672, sl=704, lt=736; final 579,123,200 params)
# Same language order as Arm R/G for direct comparison.
# Protocol: 30MB/phase, 10k steps, fresh optimizer via --init-from.
# Batch: 4 for mult≤192 (phases 1-3), 1 for mult>192.
# --grow-mult 32 widens the latent by 32 neurons per phase.
# KEY DIFFERENCE FROM ARM G: attention weights are frozen during growth
# (train.py now freezes attn params alongside embed/lm_head).
# At eval time, use compiled detector or likelihood router.
set -euo pipefail

A="out/logs/ladder_armGR_analysis.txt"
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
      --run-name ladGR-$D \
      2>&1 | tee out/logs/ladGR_${D}.log
    INIT="out/bdh_europarl_ladGR-${D}_last.pt"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" 4 >> "$A" 2>&1
    continue
  fi

  # Phases 2+: growth from previous checkpoint
  # Batch policy: batch 4 for mult≤192 (phases 2-3), batch 1 beyond
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
    --run-name ladGR-$D \
    2>&1 | tee out/logs/ladGR_${D}.log

  INIT="out/bdh_europarl_ladGR-${D}_last.pt"

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

echo "ladder-GR-done" >> "out/logs/ladder_watcher.log"
echo "Arm G+R complete. Results in $A"
