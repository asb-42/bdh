#!/usr/bin/env python
"""Repair the AdamW weight-decay leak in grown BDH checkpoints.

Background (F-decay-leak, 2026-09-03): before the train.py fix, grown runs
zeroed the frozen path's gradients, but AdamW's decoupled weight decay still
multiplied every element with grad != None by (1 - lr_t * wd) each step.
Segment p of a phase-P checkpoint therefore equals segment p of its own
phase-p checkpoint times prod over the (P - p) later phases of
prod_t(1 - lr_t * wd) = 0.57978 per phase on ladRA2 (residuals ~1e-05).

This tool measures each segment's decay factor directly against its own
source checkpoint (least-squares c-fit over encoder/encoder_v/decoder) and
divides it back out. Residuals above --max-resid flag segments whose delta
was not purely multiplicative (e.g. the en segment of the .200-era phases,
resid 5.4e-02, likely compiled-vs-eager kernel delta); the measured c is
still applied as the best available correction and the flag is recorded.

Reference semantics: after repair, segment p is at its phase-p-last state
(the state its language was acquired in). This is the per-segment
counterfactual "no decay after creation"; it is NOT the no-leak-trained
ladder (later phases never trained against undecayed earlier segments).

Read-only on inputs: writes <target-stem>_repaired.pt plus a JSON sidecar.

Usage:
  python scripts/repair_decay.py out/bdh_europarl_ladRA2-lt_last.pt \
      --sources en,es,pl,fr,de,nl,it,sv,da,pt,cz,ro,fi,hu,bg,et,el,sk,sl,lt
"""
import argparse
import json
import os
import re
import sys

import torch

NH = 8             # n_head
NPH = 64           # neurons per head per mult unit (n = mult * n_embd // nh)
SCHED_C = 0.57978  # predicted per-phase decay factor (schedule reference)


def dview(t):
    """Head-major view of the decoder (nh*N, D) -> (nh, N, D)."""
    return t.view(NH, -1, t.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--sources", required=True,
                    help="checkpoint labels in executed phase order: base "
                         "language first, then one per growth phase; the "
                         "last entry must be the target language itself")
    ap.add_argument("--source-dir", default=None)
    ap.add_argument("--source-prefix", default=None)
    ap.add_argument("--base-mult", type=int, default=128)
    ap.add_argument("--grow", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-resid", type=float, default=0.01)
    args = ap.parse_args()

    src_dir = args.source_dir or os.path.dirname(args.target) or "."
    if args.source_prefix is None:
        m = re.search(r"bdh_europarl_(.+)-[A-Za-z]+_last\.pt$",
                      os.path.basename(args.target))
        if not m:
            sys.exit("cannot derive source prefix; pass --source-prefix")
        args.source_prefix = m.group(1)

    langs = [s.strip() for s in args.sources.split(",") if s.strip()]
    ckpt = torch.load(args.target, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]
    mult = int(ckpt["cfg"]["mlp_internal_dim_multiplier"])

    segs = []
    a, z = 0, args.base_mult * NPH
    segs.append((langs[0], a, z))
    for lang in langs[1:]:
        a, z = z, z + args.grow * NPH
        segs.append((lang, a, z))
    if z != mult * NPH:
        sys.exit(f"segment math: last bound {z} != model width {mult * NPH}")
    if segs[-1][0] != langs[-1]:
        sys.exit("segment/language mismatch")

    print(f"target={args.target} mult={mult} segments={len(segs)}")
    print(f"{'seg':>3} {'lang':4} {'bounds':>15} {'c':>10} {'sched':>8} {'resid':>9} flag")
    report = []
    dec_t = dview(sd["decoder"])
    for i, (lang, a_, z_) in enumerate(segs):
        src = torch.load(
            os.path.join(src_dir, f"bdh_europarl_{args.source_prefix}-{lang}_last.pt"),
            map_location="cpu", weights_only=False)["model_state"]
        pairs = (
            (sd["encoder"][:, :, a_:z_], src["encoder"][:, :, a_:z_]),
            (sd["encoder_v"][:, :, a_:z_], src["encoder_v"][:, :, a_:z_]),
            (dec_t[:, a_:z_, :], dview(src["decoder"])[:, a_:z_, :]),
        )
        cs, resid = [], 0.0
        for dst_t, ref_t in pairs:
            v, r = dst_t.flatten(), ref_t.flatten()
            c = (v * r).sum().item() / (r * r).sum().item()
            cs.append(c)
            resid = max(resid, (v - c * r).norm().item() / v.norm().item())
        c = sum(cs) / len(cs)
        sched = SCHED_C ** (len(segs) - 1 - i)
        flag = resid > args.max_resid
        print(f"{i:>3} {lang:4} [{a_:>6},{z_:>7}) {c:10.6f} {sched:8.6f} {resid:9.2e} {'FLAG' if flag else ''}")
        for (dst_t, _), c_one in zip(pairs, cs):
            dst_t.div_(c_one)
        report.append({"segment": i, "lang": lang, "bounds": [a_, z_],
                       "c": c, "c_per_tensor": cs, "resid": resid,
                       "sched": sched, "flagged": flag})
        del src

    out = args.out or re.sub(r"_last\.pt$", "_repaired.pt", args.target)
    ckpt["repair"] = {"tool": "scripts/repair_decay.py", "sources": langs,
                      "segments": report}
    torch.save(ckpt, out)
    with open(out.replace(".pt", "_repair.json"), "w") as f:
        json.dump(report, f, indent=2)
    n_flag = sum(r["flagged"] for r in report)
    print(f"written {out} ({n_flag}/{len(segs)} segments flagged resid > {args.max_resid})")


if __name__ == "__main__":
    main()
