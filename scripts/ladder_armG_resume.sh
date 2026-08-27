#!/bin/bash
# Arm G resume: continue from phase 16 (ro) with batch=1
set -euo pipefail

A="out/logs/ladder_armG_analysis.txt"
INIT="out/bdh_europarl_ladG-sk_last.pt"

for D in ro nl sl lt; do
  echo "== resume phase: $D (batch=1) ==" >> "$A"

  .venv/bin/python -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "$D" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size 1 \
    --grow-mult 32 --init-from "$INIT" \
    --run-name ladG-$D \
    2>&1 | tee out/logs/ladG_${D}.log

  INIT="out/bdh_europarl_ladG-${D}_last.pt"
  .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$D" 1 >> "$A" 2>&1

  # Milestone at phase 20 (lt)
  if [ "$D" = "lt" ]; then
    SEEN="en,es,pl,fr,de,cs,da,pt,fi,hu,bg,it,et,el,sk,sv,ro,nl,sl,lt"
    echo "--- milestone phase 20: evaluating on all languages ---" >> "$A"
    .venv/bin/python scripts/lang_eval.py "$INIT" 30 "$SEEN" 1 >> "$A" 2>&1
  fi
done

echo "ladder-G-done" >> "out/logs/ladder_watcher.log"
echo "Arm G complete. Results in $A"
