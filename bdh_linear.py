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

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        B, _, T, N = Q.size()
        nh = self.config.n_head
        D = V.size(-1)

        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(
                1, 1, -1, 1
            )
        ) * self.freqs
        qk = Attention.rope(r_phases, Q)  # Q == K, so a single RoPE'd tensor

        c = self.chunk_size
        # The recurrent state sums outer products over all T/c chunks, so it is
        # accumulated in fp32 (outside autocast) to avoid bf16 drift on long
        # sequences; per-chunk matmuls stay in the input dtype.
        state = torch.zeros(B, nh, D, N, device=qk.device, dtype=torch.float32)
        outs = []
        for i in range(0, T, c):
            qi = qk[:, :, i : i + c]  # (B, nh, c, N)
            vi = V[:, :, i : i + c]  # (B, 1, c, D)
            intra = (qi @ qi.mT).tril(-1) @ vi  # (B, nh, c, D), causal within chunk
            inter = (state.to(qi.dtype) @ qi.mT).transpose(-1, -2)  # (B, nh, c, D), from earlier chunks
            outs.append(inter + intra)
            with torch.autocast(device_type=qk.device.type, enabled=False):
                state = state + vi.float().mT @ qi.float()  # (B, nh, D, N) outer-product update
        return torch.cat(outs, dim=2)


class BDHLinear(BDH):
    def __init__(self, config: BDHConfig, chunk_size: int = 64):
        super().__init__(config)
        self.attn = LinearAttention(config, chunk_size=chunk_size)
