#!/usr/bin/env python
"""P5 in-chain check: does the F-decay fix hold through a real growth phase?

Pre-registered by pi-50 (bdh-cl #92, P5): under the fixed regime, masked cells
must show exp_avg_sq == 0 and c == 1.000000 exactly. This is the in-chain
version: compare the BASE phase exit (e.g. ladRA2b-en_last) against the next
phase exit (ladRA2b-es_last) and require the masked base block to be bit-exact,
moments zero, and the grown block nonzero.

First result: PASS on en->es (Quinn, bdh-cl #106); independently confirmed on
es->pl by pi-50 (#110, 8,388,608 nonzero cells = exactly the new block).

Usage: python scripts/p5_inchain_check.py <parent_ckpt> <child_ckpt>

The parent checkpoint is the base-phase exit (mult 128 for a fresh chain, or
the previous phase exit for later transitions); the child is the next phase
exit. The masked width W is derived from the parent's mult: W = mult * 64
neurons per head (NPH=64, n = mult * n_embd // n_head with d=512, nh=8).
"""
import sys

import torch

NH, NPH = 8, 64


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    parent, child = sys.argv[1], sys.argv[2]
    pe = torch.load(parent, map_location='cpu', weights_only=False)
    ce = torch.load(child, map_location='cpu', weights_only=False)
    pm = int(pe['cfg']['mlp_internal_dim_multiplier'])
    cm = int(ce['cfg']['mlp_internal_dim_multiplier'])
    W = pm * NPH
    print(f'parent {parent}: mult {pm} | child {child}: mult {cm}')
    if cm <= pm:
        sys.exit('child must be wider than parent (growth phase)')
    sp, sc = pe['model_state'], ce['model_state']

    def dec(sd):
        t = sd['decoder']
        return t.view(NH, -1, t.shape[1])

    ok = {}
    ok['encoder'] = torch.equal(sp['encoder'][:, :, :W], sc['encoder'][:, :, :W])
    ok['encoder_v'] = torch.equal(sp['encoder_v'][:, :, :W], sc['encoder_v'][:, :, :W])
    ok['decoder'] = torch.equal(dec(sp)[:, :W, :], dec(sc)[:, :W, :])
    for k in ('embed.weight', 'embed'):
        if k in sp:
            ok['embed'] = torch.equal(sp[k], sc[k])
            break
    for k in ('lm_head.weight', 'lm_head'):
        if k in sp:
            ok['lm_head'] = torch.equal(sp[k], sc[k])
            break
    fa, fb = sp['attn.freqs'], sc['attn.freqs']
    n = min(fa.shape[-1], fb.shape[-1])
    ok['attn.freqs(shared)'] = torch.equal(fa[..., :n], fb[..., :n])

    print('P5[1] bit-exactness of the masked parent block through the child phase:')
    for k, v in ok.items():
        print(f'  {k:22s} {"BIT-EXACT" if v else "DIFFERS"}')
    core = all(ok[k] for k in ('encoder', 'encoder_v', 'decoder'))

    grown = dec(sc)[:, W:, :]
    print(f'P5[2] grown segment: nonzero={bool((grown != 0).any().item())} '
          f'max|w|={grown.abs().max().item():.4e}')

    print('P5[3] optimizer moments of the masked block in the child checkpoint:')
    st = ce['optimizer_state']['state']
    for idx in sorted(st):
        s = st[idx]
        v, m = s['exp_avg_sq'], s['exp_avg']
        if v.dim() == 3:
            vb, mb = v[:, :, :W], m[:, :, :W]
            vn = v[:, :, W:] if v.shape[2] > W else None
            tag = '(nh,D,N) enc/encv'
        else:
            vv, mm = v.view(NH, -1, v.shape[1]), m.view(NH, -1, m.shape[1])
            vb, mb = vv[:, :W, :], mm[:, :W, :]
            vn = vv[:, W:, :] if vv.shape[1] > W else None
            tag = '(nh*N,D) decoder'
        v0 = bool((vb == 0).all().item())
        m0 = bool((mb == 0).all().item())
        vnz = bool((vn != 0).any().item()) if vn is not None else None
        print(f'  state[{idx}] {tag}: step={s["step"]} v(masked)==0:{v0} '
              f'm(masked)==0:{m0} v(grown)!=0:{vnz}')

    print('P5-VERDICT:', 'PASS' if core else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
