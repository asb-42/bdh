#!/usr/bin/env python3
"""S1/S2 masked-forward exactness verification for grown BDH stacks.

The manuscript (sec 2, cite at cl-bdh-manuscript.tex) claims that additive
width growth preserves computation exactly when new neurons are excluded.
This script verifies those claims on toy configs, CPU-first, no checkpoints
needed. It mirrors pipeline/train.py's growth path bit-for-bit: old weights
and old RoPE freqs copied verbatim, new freqs from train.py's formula, new
parameters freshly initialised.

Core checks
  S1  masked-forward exactness: grown stack + neuron_mask keeping exactly the
      old neurons produces identical logits to the base model.
  S2  zero-init exactness: grown stack with all NEW parameters zeroed
      (encoder/encoder_v columns, decoder rows), no mask, identical logits.
  S3  freq preservation: old freqs survive growth verbatim (train.py path).
  S4  sensitivity control: grown stack with random new weights and NO mask
      must differ from base -- proves the harness can detect deviation.

Documented limitation (measured, not asserted)
  With k_sparse_ratio > 0 the top-k selection runs BEFORE the neuron mask
  (bdh.py: _k_sparse_relu, then mask), so unmasked-but-active new neurons can
  displace old neurons from the top-k set and break S1. The harness measures
  how often that happens; zero-initialising the new encoder columns restores
  exactness because their activation is exactly 0 and never enters top-k.

Numerics note: adding exact-zero terms can change BLAS reduction order over
the N axis, so S1/S2 are reported both as bit-equality and max-ULP distance.
R3 should say which one it claims. Also note: train.py's new-neuron freq
formula (2 ** 16 ** (q/n)) differs from Attention.__init__'s get_freqs
scaling ((2**16) ** (q/n)); irrelevant for S1/S2 (zero activations contribute
exactly zero regardless of phase), but worth a methods line in R3.

Run:    PYTHONPATH=. python scripts/verify_masked_forward.py [--device cpu]
Exit:   0 iff S1-S4 all pass on dense configs.
"""
import argparse
import dataclasses
import math
import sys

import torch

sys.path.insert(0, ".")

from bdh import BDH, BDHConfig  # noqa: E402


def make_cfg(mult: int, k_sparse: float = 0.0) -> BDHConfig:
    return BDHConfig(
        n_layer=2,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=mult,
        vocab_size=64,
        block_size=16,
        attn_window=0,
        dropout=0.0,
        k_sparse_ratio=k_sparse,
    )


def grow(base: BDH, total_mult: int) -> BDH:
    """Growth exactly like pipeline/train.py: copy old params + old freqs
    verbatim, new freqs from train.py's formula, new params random."""
    nh, D = base.config.n_head, base.config.n_embd
    N_old = base.config.mlp_internal_dim_multiplier * D // nh
    N_new = total_mult * D // nh
    cfg = dataclasses.replace(base.config, mlp_internal_dim_multiplier=total_mult)
    m = BDH(cfg)
    with torch.no_grad():
        m.embed.copy_(base.embed)
        m.lm_head.copy_(base.lm_head)
        m.encoder[:, :, :N_old] = base.encoder
        m.encoder_v[:, :, :N_old] = base.encoder_v
        m.decoder[: nh * N_old] = base.decoder
        old_f = base.attn.freqs.data.view(-1)
        idx = torch.arange(N_old, N_new, dtype=torch.float32)
        new_f = 1.0 / (2 ** 16 ** ((idx // 2 * 2) / N_new)) / (2 * math.pi)
        m.attn.freqs.copy_(torch.cat([old_f, new_f]).view(1, 1, 1, -1))
    return m, N_old, N_new


def zero_new(m: BDH, N_old: int) -> None:
    nh = m.config.n_head
    with torch.no_grad():
        m.encoder[:, :, N_old:].zero_()
        m.encoder_v[:, :, N_old:].zero_()
        m.decoder[nh * N_old:].zero_()


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    ab = a.detach().contiguous().view(torch.int32).to(torch.int64)
    bb = b.detach().contiguous().view(torch.int32).to(torch.int64)
    ka = torch.where(ab < 0, -2147483648 - ab, ab)
    kb = torch.where(bb < 0, -2147483648 - bb, bb)
    ulp = (ka - kb).abs()
    return {
        "bit_equal": bool(torch.equal(a, b)),
        "n_diff": int((ulp > 0).sum()),
        "max_ulp": int(ulp.max()) if ulp.numel() else 0,
        "max_abs": float((a - b).abs().max()) if a.numel() else 0.0,
    }


def fmt(st: dict) -> str:
    tag = "BIT-EQ" if st["bit_equal"] else f"max_ulp={st['max_ulp']} n_diff={st['n_diff']}"
    return tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--k-sparse", type=float, default=0.25)
    args = ap.parse_args()
    device = torch.device(args.device)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    s1_ok = s2_ok = s3_ok = s4_ok = True
    ks_rand_fail = 0
    ks_zero_ok = True
    D, NH, BASE_M, GROWN_M = 32, 2, 24, 48
    N_old = BASE_M * D // NH
    N_new = GROWN_M * D // NH

    for trial in range(args.trials):
        torch.manual_seed(1000 + trial)
        base = BDH(make_cfg(BASE_M)).to(device).eval()
        grown, _, _ = grow(base, GROWN_M)
        grown = grown.to(device).eval()
        keep_old = torch.zeros(N_new, device=device)
        keep_old[:N_old] = 1.0
        x = torch.randint(0, 64, (2, 16), device=device)

        with torch.no_grad():
            l_base, _, _ = base(x)
            l_s1, _, _ = grown(x, neuron_mask=keep_old)
            g2, _, _ = grow(base, GROWN_M)
            g2 = g2.to(device).eval()
            zero_new(g2, N_old)
            l_s2, _, _ = g2(x)
            l_s4, _, _ = grown(x)

        st1, st2, st4 = ulp_stats(l_base, l_s1), ulp_stats(l_base, l_s2), ulp_stats(l_base, l_s4)
        s1_ok &= st1["bit_equal"]
        s2_ok &= st2["bit_equal"]
        s4_ok &= (not st4["bit_equal"]) and st4["max_abs"] > 0.0
        f_ok = bool(torch.equal(grown.attn.freqs.data.view(-1)[:N_old],
                                base.attn.freqs.data.view(-1)))
        s3_ok &= f_ok
        print(f"trial {trial} dense: S1 {fmt(st1)} | S2 {fmt(st2)} | "
              f"S3 freqs {'ok' if f_ok else 'CHANGED'} | "
              f"S4 differs={'yes' if not st4['bit_equal'] else 'NO'} (max_abs {st4['max_abs']:.3e})")

        if args.k_sparse > 0:
            torch.manual_seed(2000 + trial)
            bks = BDH(make_cfg(BASE_M, args.k_sparse)).to(device).eval()
            gks, _, _ = grow(bks, GROWN_M)
            gks = gks.to(device).eval()
            gz, _, _ = grow(bks, GROWN_M)
            gz = gz.to(device).eval()
            zero_new(gz, N_old)
            with torch.no_grad():
                lb, _, _ = bks(x)
                l_rand, _, _ = gks(x, neuron_mask=keep_old)
                l_zero, _, _ = gz(x)
            if not torch.equal(lb, l_rand):
                ks_rand_fail += 1
            if not torch.equal(lb, l_zero):
                ks_zero_ok = False
            st = ulp_stats(lb, l_rand)
            print(f"trial {trial} k_sparse={args.k_sparse}: masked+random "
                  f"{fmt(st)} | masked+zero {'BIT-EQ' if torch.equal(lb, l_zero) else 'FAIL'}")

    print("\n=== SUMMARY (device=%s, trials=%d, dense cfg: 2L d=32 nh=2 mult %d->%d) ==="
          % (device.type, args.trials, BASE_M, GROWN_M))
    print(f"S1 masked-forward bit-exact:        {'PASS' if s1_ok else 'FAIL'}")
    print(f"S2 zero-init bit-exact (no mask):   {'PASS' if s2_ok else 'FAIL'}")
    print(f"S3 old freqs verbatim:              {'PASS' if s3_ok else 'FAIL'}")
    print(f"S4 sensitivity control differs:     {'PASS' if s4_ok else 'FAIL'}")
    if args.k_sparse > 0:
        print(f"k-sparse limitation (ratio={args.k_sparse}): masked+random-new bit-exact "
              f"in {args.trials - ks_rand_fail}/{args.trials} trials "
              f"(displacement seen in {ks_rand_fail}); "
              f"masked+zero-init {'PASS' if ks_zero_ok else 'FAIL'}")
    core_ok = s1_ok and s2_ok and s3_ok and s4_ok
    print(f"OVERALL: {'PASS' if core_ok else 'FAIL'}")
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
