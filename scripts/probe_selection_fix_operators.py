"""Operator-level test of candidate fixes for width-invariant sparse selection.

Provenance: written and run by pi-33 on the GX10 (gx10-50ef, ~/venvs/bdh-verify,
torch 2.13.0+cpu) on 2026-08-30, in float64 so that no result below is a numerics
artefact. Purpose: decide between the two alternatives offered for keeping forward
exactness across a growth step when k_sparse_ratio > 0.

Notation: base stack width N, grown width N' = 2N. `_k_sparse_relu` (bdh.py:13)
selects k = max(1, int(ratio * width)), so k_b < k_g by construction.

Policies compared, all measured as "total old activation retained":
  base       top-k over the N old coordinates with k_b           (reference)
  maskBefore top-k over [old, zeros] with k_g - masking new slots before
             selection is the same tensor as zero-init, so this column also
             answers "does zero-init rescue ratio top-k?"
  absoluteK  top-k over [old, zeros] with k_b frozen across the step
  maskAfter  top-k over [old, random] with k_g, then drop new slots
             (the shipped order: bdh.py:244 selects, :246-247 masks)

Run: python scripts/probe_selection_fix_operators.py
"""
import torch

torch.manual_seed(0)


def sel(v, k):
    """Emulate _k_sparse_relu's selection given an explicit count."""
    vals, idx = torch.topk(v, k)
    return vals.sum().item(), set(idx.tolist())


print(f"{'N':>4} {'rho':>5} {'k_b':>4} {'k_g':>4} {'kept_old_in_common':>18} "
      f"{'gap_maskBefore':>14} {'gap_absoluteK':>13} {'gap_maskAfter':>13}")
for N in (24, 48, 96):
    Np = 2 * N
    for rho in (0.10, 0.25, 0.50, 0.90):
        kb = max(1, int(rho * N))
        kg = max(1, int(rho * Np))
        x_old = torch.relu(torch.rand(N, dtype=torch.float64))
        zeros = torch.zeros(N, dtype=torch.float64)
        rnd = torch.relu(torch.rand(N, dtype=torch.float64))

        s_base, i_base = sel(x_old, kb)
        s_mb, i_mb = sel(torch.cat([x_old, zeros]), kg)
        s_ak, _ = sel(torch.cat([x_old, zeros]), kb)
        _, i_ma = sel(torch.cat([x_old, rnd]), kg)
        s_ma = sum(x_old[i].item() for i in i_ma if i < N)

        print(f"{N:>4} {rho:>5.2f} {kb:>4} {kg:>4} {len(i_mb & i_base):>18} "
              f"{abs(s_mb - s_base):>14.3e} {abs(s_ak - s_base):>13.3e} {abs(s_ma - s_base):>13.3e}")

print(
    "\nReading: gap_absoluteK is exactly 0 in every cell, so freezing k at the\n"
    "pre-growth value restores the induction step. gap_maskBefore is large in every\n"
    "cell: masking before selection only stops NEW neurons from winning slots, it\n"
    "cannot stop k_g from admitting EXTRA OLD ones, and `kept_old_in_common == k_b`\n"
    "shows the grown set is a strict superset. Zero-init is the same tensor, hence\n"
    "also no rescue. maskAfter is not a fair comparison at operator level (random\n"
    "new weights sometimes rank below all old values); the end-to-end verdict for\n"
    "that configuration is in verify_masked_forward.py.\n"
)
