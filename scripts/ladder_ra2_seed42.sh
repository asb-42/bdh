#!/usr/bin/env bash
# Seed-replica ladder for Table 2 (seed variance): RA2 primary arm, seed 42.
# Pre-registered 2026-08-31 (Quinn) BEFORE the run. Executor: Quinn (a0-quinn
# on ai), outputs in /home/a0-quinn/bdh-seed1/out (clone, not the shared tree).
#
# 1:1 comparability contract (checklist addendum 5):
#   - Language order = EXECUTED primary order phases 1-12 (F-T1/F-T2, cz maps
#     to cs data per F-T3) + operator-approved script order for 13-20:
#     en es pl fr de nl it sv da pt cz ro fi hu bg et el sk sl lt
#   - Batch schedule = EXECUTED primary cfg ground truth (F-T9, from all 24
#     checkpoint cfgs -- NOT the @HEAD script rule): en=4, es/pl/fr/de=2,
#     nl..ro=1; phases 13-20 bs=1.
#   - seed=42 on every phase (primary ground truth: seed=1337 in all 24 cfgs).
#     Fresh optimizer per phase via --init-from, same as primary.
#   - torch.compile stays ON (primary phases 1-12 ran compiled on this same
#     GPU; the gx10 eager delta is recorded in addendum 3).
#   - NO routing diagnoses inside this script: the primary executed ad-hoc
#     without them, and train/eval concurrency on 24 GB killed phase 13 once.
#     Routdiags come post-hoc from checkpoints with the same instrument.
#   - Milestones: p10 (after pt) and p20 (after lt) interference evals.
#     p10 SEEN = executed order through pt; p20 SEEN = full 20 (cz mapped to
#     cs), matching the gx10 resume milestone definition.
#
# Runtime estimate: ~18-21 h (phases 1-12 ~8 h from primary timings; 13-20
# at growing width + evals). If a late phase OOMs solo, set -e stops here;
# earlier checkpoints survive and the tail moves to gx10 after RA2.
set -euo pipefail
cd "$(dirname "$0")/.."

exec 9>/tmp/ladder_ra2_seed42.lock
flock -n 9 || { echo "ERROR: another seed-42 ladder instance is running"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/media/data/coding/bdh/.venv/bin/python
GROW=32
BASE_MULT=128
PHASE=0
INIT=""
LOG=out/logs
A="${LOG}/ladder_ra2s42_analysis.txt"
mkdir -p "$LOG"

ORDER="en es pl fr de nl it sv da pt cz ro fi hu bg et el sk sl lt"
bs_for() {
  case "$1" in
    en) echo 4 ;;
    es|pl|fr|de) echo 2 ;;
    *) echo 1 ;;
  esac
}
# data key: cz phase trained on cs files (F-T3); label stays cosmetic
lang_for() {
  [ "$1" = "cz" ] && echo cs || echo "$1"
}

echo "== seed-42 ladder start $(date '+%F %T') | order: $ORDER ==" >> "$A"

for D in $ORDER; do
  PHASE=$((PHASE + 1))
  BS=$(bs_for "$D")
  DL=$(lang_for "$D")
  CUR_MULT=$((BASE_MULT + (PHASE - 1) * GROW))

  if [ "$PHASE" -eq 1 ]; then
    echo "== phase $PHASE: $D (base mult=$BASE_MULT, no growth, bs=$BS, seed=42) ==" >> "$A"
    "$PY" -m pipeline.run train \
      --model bdh --dataset europarl \
      --europarl-langs "$DL" --europarl-lang-mb 30 \
      --n-embd 512 --n-head 8 --mlp-internal-dim-multiplier "$BASE_MULT" \
      --block-size 512 --max-iters 10000 --batch-size "$BS" \
      --warmup-iters 1000 --lr-decay-iters 10000 \
      --seed 42 \
      --run-name "ladRA2s42-$D" \
      2>&1 | tee "${LOG}/ladRA2s42_${D}.log"
    INIT="out/bdh_europarl_ladRA2s42-${D}_last.pt"
    "$PY" scripts/lang_eval.py "$INIT" 30 "$DL" "$BS" >> "$A" 2>&1
    continue
  fi

  echo "== phase $PHASE: $D (grow $((CUR_MULT - GROW)) -> $CUR_MULT, bs=$BS, seed=42) ==" >> "$A"
  "$PY" -m pipeline.run train \
    --model bdh --dataset europarl \
    --europarl-langs "$DL" --europarl-lang-mb 30 \
    --n-embd 512 --n-head 8 \
    --block-size 512 --max-iters 10000 --batch-size "$BS" \
    --warmup-iters 1000 --lr-decay-iters 10000 \
    --grow-mult "$GROW" --init-from "$INIT" \
    --no-freeze-attn \
    --route-aware --route-alpha 0.9 \
    --seed 42 \
    --run-name "ladRA2s42-$D" \
    2>&1 | tee "${LOG}/ladRA2s42_${D}.log"
  INIT="out/bdh_europarl_ladRA2s42-${D}_last.pt"

  "$PY" scripts/lang_eval.py "$INIT" 30 "$DL" "$BS" >> "$A" 2>&1

  if [ "$PHASE" -eq 10 ]; then
    SEEN10="en,es,pl,fr,de,nl,it,sv,da,pt"
    echo "--- milestone p10 (executed order through pt): $SEEN10 ---" >> "$A"
    "$PY" scripts/lang_eval.py "$INIT" 30 "$SEEN10" "$BS" >> "$A" 2>&1
  fi
done

SEEN20="en,es,pl,fr,de,nl,it,sv,da,pt,cs,ro,fi,hu,bg,et,el,sk,sl,lt"
echo "--- milestone p20 (full 20, cz mapped to cs): $SEEN20 ---" >> "$A"
"$PY" scripts/lang_eval.py "$INIT" 30 "$SEEN20" 1 >> "$A" 2>&1

echo "ladder-RA2s42-done $(date '+%F %T')" >> "${LOG}/ladder_watcher.log"
echo "Seed-42 ladder complete. Results in $A"
