#!/usr/bin/env bash
# BDH 4090 protocols: seed-replicate (S) and missing-cell (M), with mechanical F-V6 provenance.
# Drafted by pi-50 on gx10 (read-only seat on .200). Run under YOUR task claim, on the box that owns GPU time.
#
#   ./protocol_4090.sh preflight            # read-only environment check, never trains
#   ./protocol_4090.sh seed        en es pl fr de
#   ./protocol_4090.sh missing     en es pl fr de
#   ./protocol_4090.sh decay-check ladS-a-en_last.pt    # measure masked-block decay factor
#
# Safety: training paths refuse to start unless ALLOW_TRAIN=1 AND ANNOUNCED=<bus seq> are set, so an
# unannounced run cannot happen by accident. Announce prefix + config on bdh-cl BEFORE the first write.
set -euo pipefail

REPO="${REPO:-/media/data/coding/bdh}"
PY="$REPO/.venv/bin/python"
N_EMBD=512; N_HEAD=8; BASE_MULT=128; GROW=32; BLOCK=512; ITERS=10000; LANG_MB=30
MANIFEST="${MANIFEST:-$REPO/out/logs/pi50_protocols_manifest.jsonl}"

usage() { grep '^#' "$0" | head -12; exit 1; }

# ---------------------------------------------------------------- provenance
manifest_row() { # $1=phase_tag $2=ckpt_relpath $3=init_literal_or_scratch $4=argv $5=rc
  MAN_TAG="$1" MAN_CKPT="$2" MAN_INIT="$3" MAN_ARGV="$4" MAN_RC="$5" MAN_REPO="$REPO" \
  "$PY" - <<'PYEOF'
import os, json, hashlib, subprocess, sys, socket, datetime
repo=os.environ["MAN_REPO"]
sys.path.insert(0, repo); os.chdir(repo)
row={"ts":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
     "tag":os.environ["MAN_TAG"],"argv":os.environ["MAN_ARGV"],"rc":int(os.environ["MAN_RC"]),
     "host":socket.gethostname(),"seat":os.environ.get("USER","?")}
try: row["git_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
except Exception as e: row["git_sha"]=f"ERR {e}"
ck=os.path.join(repo, os.environ["MAN_CKPT"]); row["init_from"]=os.environ["MAN_INIT"]
if os.path.exists(ck):
    h=hashlib.sha256()
    with open(ck,"rb") as f:
        for b in iter(lambda: f.read(1<<24), b""): h.update(b)
    row["ckpt_sha256"]=h.hexdigest(); row["ckpt_bytes"]=os.path.getsize(ck)
    try:
        import torch
        c=torch.load(ck,map_location="cpu",mmap=True,weights_only=False)
        cfg=dict(c["cfg"]) if not hasattr(c["cfg"],"__dict__") else c["cfg"].__dict__
        keep=["learning_rate","weight_decay","min_lr","warmup_iters","lr_decay_iters","max_iters",
              "batch_size","block_size","n_embd","n_head","mlp_internal_dim_multiplier","route_aware",
              "route_alpha","compile","dropout","grow_mult","dataset","europarl_langs"]
        row["cfg"]={k:cfg.get(k) for k in keep}
        row["step"]=int(c.get("step",-1)); row["kernel_regime"]="compile" if cfg.get("compile") else "eager"
    except Exception as e: row["cfg_err"]=str(e)[:120]
else:
    row["ckpt_missing"]=True
print(json.dumps(row))
with open(os.environ.get("MANIFEST_PATH", os.path.join(repo,"out/logs/pi50_protocols_manifest.jsonl")),"a") as f:
    f.write(json.dumps(row)+"\n")
PYEOF
}

# ---------------------------------------------------------------- preflight
preflight() {
  echo "== preflight (read-only) =="; rc=0
  [ -x "$PY" ] && echo "  venv python      : OK" || { echo "  venv python      : MISSING ($PY)"; rc=1; }
  cd "$REPO"
  echo "  git HEAD         : $(git rev-parse --short HEAD)  branch $(git rev-parse --abbrev-ref HEAD)"
  git diff --quiet || echo "  working tree     : DIRTY - fix before launching anything"
  [ -f scripts/lang_eval.py ] && echo "  lang_eval.py     : present" || { echo "  lang_eval.py     : MISSING"; rc=1; }
  if command -v nvidia-smi >/dev/null; then
    read -r util mem <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ')"
    echo "  gpu              : ${util}% util, ${mem} MiB used"
    [ "${util:-100}" -lt 20 ] || { echo "                   ^ busy: another seat owns the GPU, do not launch"; rc=1; }
  else echo "  nvidia-smi       : absent"; fi
  pgrep -af "pipeline.run train|ladder_" 2>/dev/null | sed 's/^/  live job: /' || echo "  live jobs        : none detected"
  [ -w out ] && echo "  out/             : writable (do NOT write without a claimed scope)" || { echo "  out/             : not writable"; rc=1; }
  ls /var/tmp/bdh_europarl_*.pt >/dev/null 2>&1 && echo "  data cache       : present" || echo "  data cache       : absent (first run will download/build)"
  echo "  manifest target  : $MANIFEST"
  echo "== preflight $([ $rc -eq 0 ] && echo PASS || echo FAIL) =="; return $rc
}

# ---------------------------------------------------------------- shared runner
run_chain() { # $1=prefix  remaining=langs ; env EXTRA_FLAGS for the M cell
  local PREFIX="$1"; shift
  local LANGS=("$@")
  [ "${ALLOW_TRAIN:-0}" = "1" ] || { echo "refusing to train: set ALLOW_TRAIN=1"; exit 2; }
  [ -n "${ANNOUNCED:-}" ] || { echo "refusing to train: set ANNOUNCED=<bdh-cl seq of your intent post>"; exit 2; }
  local INIT="" PHASE=0 LOG="$REPO/out/logs/${PREFIX}_ladder.log" A="${EXTRA_FLAGS:-}"
  rm -f "$LOG"
  for D in "${LANGS[@]}"; do
    PHASE=$((PHASE+1))
    # Batch policy copied from scripts/ladder_armG.sh:55-62 (the CODE, not its header comment,
    # which claims batch 2 beyond mult 192 while the implementation uses batch 1).
    local BS=4
    if [ -n "$INIT" ]; then
      PREV_MULT=$((BASE_MULT + (PHASE - 2) * GROW))
      [ "$PREV_MULT" -le 192 ] || BS=1
    fi
    echo "== $PREFIX phase $PHASE: $D (prev_mult=${PREV_MULT:-n/a} child_mult=$((BASE_MULT + (PHASE-1)*GROW)) batch=$BS extra='${A:-none}') ==" >> "$LOG"
    local CMD=("$PY" -m pipeline.run train --model bdh --dataset europarl
               --europarl-langs "$D" --europarl-lang-mb $LANG_MB
               --n-embd $N_EMBD --n-head $N_HEAD --block-size $BLOCK --max-iters $ITERS --batch-size $BS)
    if [ -z "$INIT" ]; then
      CMD+=(--mlp-internal-dim-multiplier $BASE_MULT)
    else
      CMD+=(--grow-mult $GROW --init-from "$INIT")
    fi
    [ -n "$A" ] && CMD+=($A)
    CMD+=(--run-name "$PREFIX-$D")
    echo "\$ ${CMD[*]}" >> "$LOG"
    ( cd "$REPO" && "${CMD[@]}" >> "$LOG" 2>&1 ) && RC=0 || RC=$?
    CK="out/bdh_europarl_${PREFIX}-${D}_last.pt"
    manifest_row "$PREFIX-p$PHASE" "$CK" "${INIT:-SCRATCH}" "${CMD[*]}" "$RC"
    [ $RC -ne 0 ] && { echo "phase $PHASE failed rc=$RC (see $LOG)"; return $RC; }
    ( cd "$REPO" && "$PY" scripts/lang_eval.py "$CK" $LANG_MB "$D" $BS >> "$LOG" 2>&1 )
    INIT="$CK"
  done
}

# S: two identical chains -> sizes run-to-run variance (predictions S1-S3)
seed() {
  local L=("$\@")
  [ ${#L[@]} -gt 0 ] || L=(en es pl fr de)
  echo "S protocol: two identical 5-phase chains, Arm-G defaults inherited verbatim."
  run_chain "ladS-a" "${L[@]}"; run_chain "ladS-b" "${L[@]}"
}
# M: plain growth + aggressive decay (never run) -> separates decay regime from co-adaptation (M1-M3)
missing() {
  local L=("$\@")
  [ ${#L[@]} -gt 0 ] || L=(en es pl fr de)
  echo "M protocol: route-aware OFF, cosine schedule ON (RA2-style flags on plain growth)."
  EXTRA_FLAGS="--warmup-iters 1000 --lr-decay-iters 10000 --min-lr 1e-4 --no-route-aware" \
    run_chain "ladMD" "${L[@]}"
}

# decay-check: independent verification that masked blocks obey the closed form
decay_check() {
  local NEW="$1"; local OLD="${2:-}"
  ( cd "$REPO" && CK="$NEW" PREV="$OLD" "$PY" - <<'PYEOF'
import os, sys, torch, statistics as S
repo=os.getcwd(); new=os.environ["CK"]; prev=os.environ.get("PREV") or ""
def load(p): return torch.load(p,map_location="cpu",mmap=True,weights_only=False)
c=load(new); NH=8; BLK=2048; DM=512
per=int(c["cfg"]["mlp_internal_dim_multiplier"])*DM//NH; lpw=per//BLK
dec=c["model_state"]["decoder"]; v=c["optimizer_state"]["state"][0]["exp_avg_sq"]
masked=[b for b in range(lpw) if all(int((v[h*per+b*BLK:h*per+(b+1)*BLK]!=0).sum())==0 for h in range(NH))]   # NOTE: count nonzeros; int(absmax)==0 is a false-positive trap
print(f"{new}: local blocks={lpw} strictly-zero-second-moment blocks={masked}")
if prev and os.path.exists(prev):
    p=load(prev); per_p=int(p["cfg"]["mlp_internal_dim_multiplier"])*DM//NH
    dp=p["model_state"]["decoder"]; rs=[]
    for h in range(NH):
        for b in masked[:3]:
            rp=h*per_p+b*BLK+torch.arange(BLK); rc=h*per+b*BLK+torch.arange(BLK)
            if rp.max()<dp.shape[0] and rc.max()<dec.shape[0]:
                rs.append(float(dec[rc].float().norm()/dp[rp].float().norm()))
    cf=c["cfg"]
    import math
    lr0=float(cf["learning_rate"]); mn=float(cf["min_lr"]); wu=int(cf["warmup_iters"]); dc=int(cf["lr_decay_iters"])
    wd=float(cf["weight_decay"]); N=int(cf.get("max_iters",10000)); st=int(c.get("step",N))
    def sched(i):
        if i<wu: return lr0*(i+1)/wu
        if i>dc: return mn
        r=(i-wu)/max(1,dc-wu); return mn+0.5*(1+math.cos(math.pi*r))*(lr0-mn)
    import struct
    f32=lambda x: struct.unpack('f',struct.pack('f',x))[0]
    prod_f64=1.0; prod_f32=1.0
    for i in range(st): prod_f64*=1-sched(i)*wd; prod_f32*=f32(1-sched(i)*wd)
    print(f"  measured ratio = {S.mean(rs):.6f} sd={S.stdev(rs) if len(rs)>1 else 0:.1e} n={len(rs)}")
    print(f"  predicted      = {prod_f64:.6f} (f64) / {prod_f32:.6f} (f32-cast factor)   steps={st}")
PYEOF
  )
}

case "${1:-}" in
  preflight)   preflight ;;
  seed)        shift; seed ${@:-en es pl fr de} ;;
  missing)     shift; missing ${@:-en es pl fr de} ;;
  decay-check) shift; decay_check "$@" ;;
  *) usage ;;
esac
