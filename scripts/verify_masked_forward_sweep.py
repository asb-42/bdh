#!/usr/bin/env python3
"""S1/S2 exactness coverage sweep: production widths, multiple seeds.

Pre-registered 2026-08-31 (Quinn), BEFORE the run. Mirrors the logic of
the GX10-attested scripts/verify_masked_forward.py (grow(), zero_new(),
8*eps tolerance, train.py freq formula), only parametrized: width pairs,
k_sparse ratios, seeds, block/vocab.

Regimes and their references (mirroring the attested script exactly):
  dense     dense base -> grow. S1: mask keeps old neurons; S2: zero-init
            new slots, no mask. Both must reproduce the dense base.
  ratio r   the BASE itself is k-sparse at ratio r (constructed via
            make_cfg(..., k_sparse=r), as in the attested script); grow()
            inherits r via dataclasses.replace. Comparison: zero-init
            grown stack (no mask) vs the k-sparse base. This is the
            pi-33 dose-response metric: under ratio top-k neither mask
            nor zero-init restores exactness (retraction 6754e83) --
            the question is how large the gap is at production width.

Open empirical question this sweep answers first: whether r=0.90 is
non-binding at production widths too (toy: exact, because k then exceeds
the number of positive old activations). Binding depends on the positive
rate at random init, not on width alone -- measured, not asserted.

Determinism: torch.use_deterministic_algorithms is enabled on CPU only;
the attested run used CPU and CUDA forward determinism is not claimable
across arms anyway (Guidelines section 7, ULP class).

Run:  PYTHONPATH=. python scripts/verify_masked_forward_sweep.py \
          [--device cuda] [--trials 5]
"""
import argparse
import dataclasses
import math
import sys

import torch

sys.path.insert(0, ".")

from bdh import BDH, BDHConfig  # noqa: E402

# width pairs: (BASE_M, GROWN_M, n_embd, n_head, block, vocab)
PAIRS = [
    (24, 48, 32, 2, 16, 64),       # toy reference (attested geometry)
    (128, 160, 512, 8, 512, 256),  # production, phase 2 of the ladder
    (512, 544, 512, 8, 512, 256),  # production, phase 14 of the ladder
]
RATIOS = [0.10, 0.25, 0.50, 0.90]


def make_cfg(m: int, d: int, nh: int, block: int, vocab: int,
             k_sparse: float = 0.0) -> BDHConfig:
    return BDHConfig(
        n_layer=2,
        n_embd=d,
        n_head=nh,
        mlp_internal_dim_multiplier=m,
        vocab_size=vocab,
        block_size=block,
        attn_window=0,
        dropout=0.0,
        k_sparse_ratio=k_sparse,
    )


def grow(base: BDH, total_mult: int) -> BDH:
    """Growth exactly like pipeline/train.py (attested mirror): old params
    + old freqs verbatim, new freqs from train.py's formula, new params
    random. k_sparse_ratio is inherited via dataclasses.replace."""
    nh, D = base.config.n_head, base.config.n_embd
    N_old = base.config.mlp_internal_dim_multiplier * D // nh
    N_new = total_mult * D // nh
    cfg = dataclasses.replace(base.config, mlp_internal_dim_multiplier=total_mult)
    m = BDH(cfg)
    with torch.no_grad():
        m.embed.weight.copy_(base.embed.weight)
        m.lm_head.copy_(base.lm_head)
        m.encoder[:, :, :N_old] = base.encoder
        m.encoder_v[:, :, :N_old] = base.encoder_v
        m.decoder.view(nh, N_new, -1)[:, :N_old, :] = \
            base.decoder.view(nh, N_old, -1)
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
        m.decoder.view(nh, m.encoder.size(-1), -1)[:, N_old:, :].zero_()


def tol_of(lb: torch.Tensor) -> float:
    return 8 * torch.finfo(torch.float32).eps \
        * max(1.0, float(lb.abs().max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    print(f"device={device.type} trials={args.trials}")
    print("| pair | regime | max_abs (worst) | bit-eq | tol-pass |")
    print("|---|---|---|---|---|")

    for (bm, gm, d, nh, block, vocab) in PAIRS:
        N_old = bm * d // nh
        N_new = gm * d // nh

        # ---- dense regime: S1 (mask) + S2 (zero-init) vs dense base ----
        worst = 0.0
        bits = 0
        tols = 0
        for trial in range(args.trials):
            torch.manual_seed(1000 + trial)
            base = BDH(make_cfg(bm, d, nh, block, vocab)).to(device).eval()
            grown, _, _ = grow(base, gm)
            grown = grown.to(device).eval()
            keep_old = torch.zeros(N_new, device=device)
            keep_old[:N_old] = 1.0
            x = torch.randint(0, vocab, (2, block), device=device)
            with torch.no_grad():
                l_base = base(x)[0]
                l_s1 = grown(x, neuron_mask=keep_old)[0]
                g2, _, _ = grow(base, gm)
                g2 = g2.to(device).eval()
                zero_new(g2, N_old)
                l_s2 = g2(x)[0]
            t = tol_of(l_base)
            for st in (l_base - l_s1, l_base - l_s2):
                m = float(st.abs().max())
                worst = max(worst, m)
                tols += int(m <= t)
            bits += int(torch.equal(l_base, l_s1)) \
                + int(torch.equal(l_base, l_s2))
            del base, grown, g2
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"| {bm}->{gm} (D={d}) | dense (S1+S2) | {worst:.3e} | "
              f"{bits}/{2 * args.trials} | {tols}/{2 * args.trials} |")

        # ---- ratio regimes: k-sparse base, zero-init grown, no mask ----
        for r in RATIOS:
            worst = 0.0
            bits = 0
            tols = 0
            for trial in range(args.trials):
                torch.manual_seed(1000 + trial)
                base = BDH(make_cfg(bm, d, nh, block, vocab,
                                    k_sparse=r)).to(device).eval()
                gz, _, _ = grow(base, gm)  # inherits k_sparse_ratio=r
                gz = gz.to(device).eval()
                zero_new(gz, N_old)
                x = torch.randint(0, vocab, (2, block), device=device)
                with torch.no_grad():
                    lb = base(x)[0]
                    l_zero = gz(x)[0]
                t = tol_of(lb)
                m = float((lb - l_zero).abs().max())
                worst = max(worst, m)
                tols += int(m <= t)
                bits += int(torch.equal(lb, l_zero))
                del base, gz
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"| {bm}->{gm} (D={d}) | ratio {r} | {worst:.3e} | "
                  f"{bits}/{args.trials} | {tols}/{args.trials} |")

    print("\nsweep-done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
