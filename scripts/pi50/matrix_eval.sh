#!/usr/bin/env bash
# RA2b 20x20 matrix runner: every checkpoint evaluated on every language.
# Fills the 350 missing off-diagonal cells (only 50 exist today: milestones at phases 5/10/15/20).
# Eval-only: loads existing _last.pt files, no training, no writes into out/.
#
#   ./matrix_eval.sh preflight
#   ANNOUNCED=<bdh-cl seq> ALLOW_EVAL=1 ./matrix_eval.sh run [en es ...]     # subset allowed
#   ./matrix_eval.sh summarize
#
# Protocol pinned deliberately (this is the whole point of the artifact):
#   EVAL_BATCH identical for every cell (default 1)  -> train-batch effect (#123, +13.7%) cannot enter
#   one lang_eval.py invocation per checkpoint       -> same model load, same crop stream, same session
#   crop seeding is inside scripts/lang_eval.py (Generator(1337-ish)) and recorded, not assumed
#   manifest row per checkpoint: git SHA, argv, resolved cfg, eval batch, ckpt size + head-hash
set -euo pipefail

REPO="${REPO:-/media/data/coding/bdh}"
PY="$REPO/.venv/bin/python"
OUTDIR="${OUTDIR:-$HOME/bdh_matrix}"
CSV="$OUTDIR/ra2b_matrix.csv"
MAN="$OUTDIR/ra2b_matrix_manifest.jsonl"
EVAL_BATCH="${EVAL_BATCH:-1}"
PREFIX="${PREFIX:-ladRA2b}"
SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt"

mkdir -p "$OUTDIR"
[ -f "$CSV" ] || echo "checkpoint_lang,eval_lang,nll,ppl,eval_batch,git_sha" > "$CSV"

preflight() {
  echo "== matrix preflight (read-only) =="; local rc=0
  [ -x "$PY" ] && echo "  venv            : OK" || { echo "  venv            : MISSING"; rc=1; }
  cd "$REPO"
  echo "  git HEAD        : $(git rev-parse --short HEAD)"
  [ -f scripts/lang_eval.py ] && echo "  lang_eval.py    : present" || { echo "  lang_eval.py    : MISSING"; rc=1; }
  local n=0; for l in $SEQ; do [ -f "out/bdh_europarl_${PREFIX}-${l}_last.pt" ] && n=$((n+1)); done
  echo "  checkpoints     : $n/20 found (prefix $PREFIX)"
  [ "$n" -eq 20 ] || { echo "                  ^ incomplete chain; matrix would have holes"; rc=1; }
  pgrep -af "pipeline.run train|eval_router|ladder_" 2>/dev/null | sed 's/^/  live job      : /' | head -4 || echo "  live jobs       : none"
  df -h "$OUTDIR" | tail -1 | awk '{print "  free at OUTDIR  : "$4}'
  echo "  outputs         : $CSV , $MAN (NOT under out/ -> no lease needed)"
  echo "== preflight $([ $rc -eq 0 ] && echo PASS || echo FAIL) =="; return $rc
}

run_one() { # $1 = checkpoint language
  local CKL="$1" CK="out/${PREFIX}-${CKL}_last.pt"
  local have; have=$(grep -c "^${CKL}," "$CSV" 2>/dev/null || true)
  if [ "${have:-0}" -ge 20 ]; then echo "  ${CKL}: already complete (${have} rows), skipping"; return 0; fi
  local ARGS="--model bdh --dataset europarl --europarl-langs $SEQ --europarl-lang-mb 30 --block-size 512 --max-iters 1 --batch-size $EVAL_BATCH"
  echo "  ${CKL}: evaluating 20 domains at eval_batch=$EVAL_BATCH ..."
  local LOG="$OUTDIR/raw_${CKL}.txt"
  ( cd "$REPO" && "$PY" scripts/lang_eval.py "$CK" 30 "$(echo $SEQ | tr ' ' ',')" "$EVAL_BATCH" ) > "$LOG" 2>&1 || {
      echo "    FAILED (see $LOG):"; tail -3 "$LOG" | sed 's/^/      /'; return 1; }
  CKL="$CKL" CSVOUT="$CSV" MANOUT="$MAN" REPOIN="$REPO" EB="$EVAL_BATCH" ARGSTR="$ARGS" RAW="$LOG" PFX="$PREFIX" \
  "$PY" - <<'PYEOF'
import os, re, sys, json, hashlib, subprocess, socket, datetime
repo=os.environ["REPOIN"]; ckl=os.environ["CKL"]; csvp=os.environ["CSVOUT"]; manp=os.environ["MANOUT"]
os.chdir(repo); sys.path.insert(0, repo)
sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
rows=[]
for line in open(os.environ["RAW"]):
    m=re.match(r"\s+(\w{2}): nll ([0-9.]+) \| ppl ([0-9.]+)", line)
    if m: rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
if len(rows)!=20:
    print(f"    WARNING: parsed {len(rows)}/20 domain rows for {ckl}; writing what we have")
with open(csvp,"a") as f:
    for lg,nll,ppl in rows:
        f.write(f"{ckl},{lg},{nll},{ppl},{os.environ['EB']},{sha[:12]}\n")
ck=os.path.join(repo,f"out/bdh_europarl_{os.environ['PFX']}-{ckl}_last.pt")
h=hashlib.sha256()
with open(ck,"rb") as fh:
    h.update(fh.read(1<<26))
row={"ts":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),"host":socket.gethostname(),
     "git_sha":sha,"checkpoint":os.path.basename(ck),"ckpt_bytes":os.path.getsize(ck),"ckpt_head256":h.hexdigest()[:16],
     "eval_batch":int(os.environ["EB"]),"argv":"scripts/lang_eval.py "+os.environ["ARGSTR"],
     "domains_parsed":len(rows)}
try:
    import torch
    c=torch.load(ck,map_location="cpu",mmap=True,weights_only=False); cf=dict(c["cfg"])
    row["params_from_shapes"]=sum(v.numel() for v in c["model_state"].values())
    row["step"]=int(c.get("step",-1))
    row["cfg"]={k:cf.get(k) for k in ["learning_rate","weight_decay","min_lr","warmup_iters","lr_decay_iters",
                                      "max_iters","batch_size","block_size","mlp_internal_dim_multiplier",
                                      "route_aware","route_alpha","seed","compile"]}
except Exception as e: row["cfg_err"]=str(e)[:120]
with open(manp,"a") as f: f.write(json.dumps(row)+"\n")
print(f"    {ckl}: wrote {len(rows)} cells; params={row.get('params_from_shapes','?')} step={row.get('step','?')}")
PYEOF
}

run() {
  [ "${ALLOW_EVAL:-0}" = "1" ] || { echo "refusing: set ALLOW_EVAL=1"; exit 2; }
  [ -n "${ANNOUNCED:-}" ] || { echo "refusing: set ANNOUNCED=<bdh-cl seq of your intent post>"; exit 2; }
  preflight || { echo "preflight failed, aborting"; exit 3; }
  local langs=("$@"); [ ${#langs[@]} -gt 0 ] || langs=($SEQ)
  echo "== running ${#langs[@]} checkpoints, resume-safe =="
  for l in "${langs[@]}"; do run_one "$l"; done
  echo "== done: $(($(wc -l < "$CSV")-1)) cells in $CSV =="
}

summarize() {
  OUTD="$OUTDIR" PREFIX="$PREFIX" "$PY" - <<'PYEOF'
import os, csv, itertools
d=os.environ["OUTD"]; rows=list(csv.DictReader(open(os.path.join(d,"ra2b_matrix.csv"))))
M={}
for r in rows: M.setdefault(r["checkpoint"],{})[r["eval_lang"]]=float(r["ppl"])
langs=["en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()]
langs=langs[0]
print(f"# RA2b matrix summary ({len(M)}/{len(langs)} checkpoints x {len(langs)} domains)\n")
print("| checkpoint | " + " | ".join(langs) + " |")
print("|---|"+"---|"*len(langs))
for c in langs:
    if c in M: print(f"| **{c}** | " + " | ".join(f"{M[c].get(l,float('nan')):.2f}" for l in langs) + " |")
order=[l for l in langs if l in M]
print("\n## Transfer / interference (operator formulas, S1d floor 2-4%)")
print("- `I[A->B]` backward interference on already-known B when phase A trains: ppl_B(after A) - ppl_B(before A)")
print("- `T[A->B]` forward transfer onto not-yet-trained B: ppl_B(before A) - ppl_B(after A)\n")
bad=[]
for i,a in enumerate(order):
    prev = order[i-1] if i>0 else None
    if prev is None or prev not in M or a not in M: continue
    for b in order:
        if b==a or b not in M[prev] or b not in M[a]: continue
        delta=M[a][b]-M[prev][b]
        if abs(delta)/M[prev][b] > 0.04: bad.append((a,b,delta,M[prev][b]))
bad.sort(key=lambda t:-abs(t[2]))
print(f"| changed checkpoint | domain | delta ppl | rel % (>4% = above seed floor) |")
print("|---|---|---|---|")
for a,b,dl,base in bad[:25]: print(f"| {a} | {b} | {dl:+.2f} | {dl/base*100:+.1f}% |")
print(f"\n{len(bad)} transitions move a non-current domain beyond the 4% floor.")
PYEOF
}

case "${1:-}" in
  preflight) preflight ;;
  run) shift; run "$@" ;;
  summarize) summarize ;;
  *) grep '^#' "$0" | head -8; exit 1 ;;
esac
