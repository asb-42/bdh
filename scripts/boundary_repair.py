#!/usr/bin/env python
"""Boundary undecay for leaky BDH ladders (F-decay-leak companion).

repair_decay.py restores each segment to its OWN phase-exit amplitude.
That composition never existed during training: later phases trained
against already-decayed older segments, so cross-segment co-adaptation
breaks under it (measured 2026-09-03/04: RA2 routed bg 15.65 raw ->
37.63 per-segment-repaired, el 16.11 -> 62.64; G bg 2.50 -> 41.31).

The leak multiplies every frozen-path element by the SAME per-phase
factor c in every later phase (uniform multiplicative, c-fit residuals
~1e-5). Therefore one scalar restores the whole checkpoint to its
phase-j boundary: divide all segments below the phase-j width by
c^(final_phase - j). That state did exist (it is the end-of-phase-j
checkpoint), co-adaptation is preserved, and routed serving returns to
the phase-j routdiag value (verified on RA2 fi: undecay to the p13
boundary restored routed serving 15.87 -> 8.68, the measured p13 value).

Schedule references:
  c = 0.57978  RA2/RA2b family (--warmup-iters 1000 --lr-decay-iters 10000)
  c = 0.8927   G/GR family (pipeline defaults: warmup 30, decay 300, min_lr 1e-4)

Rows at and above the phase-j boundary are left untouched: the artifact
is valid for serving on routes <= the phase-j width (deeper rows are
masked off in routed serving anyway). embed/lm_head and attention
buffers never decay (outside the optimizer) and are not modified.

Usage:
  python scripts/boundary_repair.py out/bdh_europarl_ladRA2-lt_last.pt \
      --phase 15 --c 0.57978 --out out/ladRA2-lt_bj15.pt
"""
import argparse
import re

import torch

NH = 8    # n_head
NPH = 64  # neurons per head per mult unit (n = mult * n_embd // nh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--phase", type=int, required=True,
                    help="boundary phase j: restore the end-of-phase-j state")
    ap.add_argument("--final-phase", type=int, default=20)
    ap.add_argument("--c", type=float, default=0.57978,
                    help="per-phase decay factor (schedule reference)")
    ap.add_argument("--base-mult", type=int, default=128)
    ap.add_argument("--grow", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.target, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]
    mult = int(ckpt["cfg"]["mlp_internal_dim_multiplier"])
    total = mult * NPH
    jw = (args.base_mult + args.grow * (args.phase - 1)) * NPH
    if jw > total:
        raise SystemExit(f"phase-{args.phase} width {jw} exceeds model width {total}")
    k = args.final_phase - args.phase
    if k < 0:
        raise SystemExit("phase is beyond the final phase")
    div = args.c ** k

    with torch.no_grad():
        sd["encoder"][:, :, :jw].div_(div)
        sd["encoder_v"][:, :, :jw].div_(div)
        dec = sd["decoder"]
        dec.view(NH, -1, dec.shape[1])[:, :jw, :].div_(div)
    ckpt["boundary_repair"] = {
        "tool": "scripts/boundary_repair.py",
        "phase": args.phase,
        "final_phase": args.final_phase,
        "c": args.c,
        "divisor": div,
        "width": jw,
    }
    out = args.out or re.sub(r"_last\.pt$", f"_bj{args.phase}.pt", args.target)
    torch.save(ckpt, out)
    print(f"boundary repair: phase {args.phase} (width {jw} of {total}), "
          f"divided [0,{jw}) by c^{k} = {div:.6f}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
