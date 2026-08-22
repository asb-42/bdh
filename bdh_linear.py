# Copyright Pathway Technology, Inc.
"""BDH variant with linear-time (state-space) attention.

``BDHLinear`` is mathematically identical to ``bdh.BDH``; it only replaces the
explicit ``(Q @ K^T) @ V`` attention with the equivalent chunked recurrent scan

    state_t = state_{t-1} + v_t k_t^T          (outer product = "Hebbian" update)
    out_t   = state_{t-1} q_t

which, with the same absolute-position RoPE, reproduces exactly the causal
attention scores ``sum_{tau<t} v_tau (q_t . k_tau)`` of the quadratic form.

This makes attention linear in the sequence length ``T`` (with ``O(chunk^2)``
intra-chunk cost) instead of quadratic. The recurrent state is accumulated in
fp32 (even under autocast) since it sums outer products over all chunks; the
per-chunk matmuls stay in the input dtype. Everything else (the low-rank
encoder / decoder MLP, gating, LayerNorms, residual structure) is inherited
unchanged from ``bdh.BDH``.
"""

import torch
import torch.nn.functional as F
from torch import nn

from bdh import Attention, BDH, BDHConfig, get_freqs


class LinearAttention(nn.Module):
    def __init__(self, config: BDHConfig, chunk_size: int = 64):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.chunk_size = chunk_size
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    def forward(self, Q, V, state=None, pos_offset=0):
        """Chunked linear-attention scan, optionally continuing from carried state.

        state: None or the recurrent attention state rho (B, nh, D, N), float32,
        as returned by a previous call. pos_offset continues absolute RoPE
        positions across calls. Returns (output, new_state).
        """
        assert self.freqs.dtype == torch.float32
        B, _, T, N = Q.size()
        nh = self.config.n_head
        D = V.size(-1)

        r_phases = (
            torch.arange(
                pos_offset,
                pos_offset + T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        qk = Attention.rope(r_phases, Q)

        c = self.chunk_size
        # The recurrent state sums outer products over all chunks (and, when
        # carried over, over all previous calls), so it is accumulated in fp32
        # (outside autocast) to avoid bf16 drift on long sequences; per-chunk
        # matmuls stay in the input dtype.
        gamma = None
        if self.config.alibi_slope > 0.0:
            gamma = float(torch.exp(torch.tensor(-self.config.alibi_slope)))

        if state is not None:
            rho = state.detach() if self.config.no_bptt else state
        else:
            rho = torch.zeros(B, nh, D, N, device=qk.device, dtype=torch.float32)
        outs = []
        for i in range(0, T, c):
            ce = min(c, T - i)  # effective chunk size (last chunk may be short)
            qi = qk[:, :, i : i + ce]  # (B, nh, ce, N)
            vi = V[:, :, i : i + ce]  # (B, 1, ce, D)
            if self.config.no_bptt:
                # Paper Sec. 5.2: detach keys and values so no gradient flows
                # back through time; the query side keeps its gradient.
                qi_k = qi.detach()
                vi = vi.detach()
            else:
                qi_k = qi

            if gamma is None:
                intra = (qi @ qi_k.mT).tril(-1) @ vi  # (B, nh, ce, D)
                inter = (rho.to(qi.dtype) @ qi.mT).transpose(-1, -2)
                with torch.autocast(device_type=qk.device.type, enabled=False):
                    rho = rho + vi.float().mT @ qi_k.float()
            else:
                dev, dt = qi.device, qi.dtype
                idx = torch.arange(ce, device=dev, dtype=dt)
                # ALiBi damping (paper Sec. 4.1): each token's contribution to a
                # query decays as gamma^(elapsed distance).
                #   read at query j:  gamma^j * rho  +  sum_{m<j} gamma^(j-m) v_m k_m
                #   state update:     rho <- gamma^ce * rho + sum_m gamma^(ce-m) v_m k_m
                decay_intra = (gamma ** (idx.unsqueeze(-1) - idx.unsqueeze(0))).tril(-1)
                intra = ((qi @ qi_k.mT) * decay_intra) @ vi
                g_pow = gamma ** idx  # gamma^j
                inter = (rho.to(dt) @ (qi * g_pow.view(1, 1, -1, 1)).mT).transpose(-1, -2)
                outs.append(inter + intra)
                with torch.autocast(device_type=qk.device.type, enabled=False):
                    w = (gamma ** (ce - 1 - idx)).view(1, 1, -1, 1)
                    upd = (vi.float() * w).mT @ qi_k.float()
                    rho = (gamma**ce) * rho + gamma * upd
                continue

            outs.append(inter + intra)
        return torch.cat(outs, dim=2), rho


class BDHLinear(BDH):
    def __init__(self, config: BDHConfig, chunk_size: int = 64):
        super().__init__(config)
        self.attn = LinearAttention(config, chunk_size=chunk_size)
