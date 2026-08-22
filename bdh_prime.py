"""BDH-GPU' variant (paper Sec. 3 / Appendix B): xLSTM-like gating of the
attention state plus next-token logit merging across layers.

Two extensions over vanilla BDH-GPU:

  1. Gated state update (mLSTM-style, cf. xLSTM):
        rho_t := f_t (*) (gamma * rho_{t-1}) + i_t * v_t k_t^T
     with a forget gate f_t = sigmoid(v_t W_f + b_f) per value channel and an
     input gate i_t = softplus(k_t W_i + b_i) per neuron channel. gamma is the
     optional ALiBi damping factor.
  2. Logit merging: next-token logits are computed from every layer's output
     and averaged, instead of only the last layer's.

The gated scan runs as a per-token recurrence (correct and simple; a fused
parallel scan is future work). Everything else is inherited from BDHLinear.
"""

import torch
import torch.nn.functional as F
from torch import nn

from bdh import Attention, BDHConfig
from bdh_linear import BDHLinear, LinearAttention


class GatedLinearAttention(LinearAttention):
    """Linear attention with xLSTM-style forget/input gates on the state."""

    def __init__(self, config: BDHConfig, chunk_size: int = 64):
        super().__init__(config, chunk_size=chunk_size)
        D = config.n_embd
        nh = config.n_head
        N = config.mlp_internal_dim_multiplier * D // nh
        self.wf = nn.Parameter(torch.zeros(D))
        self.bf = nn.Parameter(torch.zeros(D))
        self.wi = nn.Parameter(torch.zeros(N))
        self.bi = nn.Parameter(torch.full((N,), -2.0))  # start nearly closed

    def forward(self, Q, V, state=None, pos_offset=0):
        assert self.freqs.dtype == torch.float32
        B, _, T, _ = Q.size()
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

        gamma = None
        if self.config.alibi_slope > 0.0:
            gamma = float(torch.exp(torch.tensor(-self.config.alibi_slope)))

        if state is not None:
            rho = state.detach() if self.config.no_bptt else state
        else:
            rho = torch.zeros(B, self.config.n_head, D, Q.size(-1),
                              device=qk.device, dtype=torch.float32)

        outs = []
        for t in range(T):
            qt = qk[:, :, t]   # (B, nh, N)
            vt = V[:, 0, t]    # (B, D)
            if self.config.no_bptt:
                # Paper Sec. 5.2: detach keys/values; query side keeps gradient.
                qt_k = qt.detach()
                vt_d = vt.detach()
            else:
                qt_k = qt
                vt_d = vt

            # damp history one step, then read (before adding token t)
            if gamma is not None:
                rho = rho * gamma
            read = (rho.to(qt.dtype) @ qt.unsqueeze(-1)).squeeze(-1)  # (B, nh, D)
            outs.append(read)

            # gated Hebbian update (diagonal gates: per-channel, parameter-cheap)
            f = torch.sigmoid(vt * self.wf + self.bf)      # (B, D) forget gate
            igate = F.softplus(qt_k * self.wi + self.bi)   # (B, nh, N) input gate
            upd = vt_d.float().view(B, 1, D, 1) * qt_k.float().view(B, qt_k.size(1), 1, -1)
            rho = f.float().view(B, 1, D, 1) * rho + igate.float().unsqueeze(2) * upd

        return torch.stack(outs, dim=2), rho


class BDHPrime(BDHLinear):
    """BDH-GPU': gated linear attention + per-layer logit merging."""

    def __init__(self, config: BDHConfig, chunk_size: int = 64):
        super().__init__(config, chunk_size=chunk_size)
        self.attn = GatedLinearAttention(config, chunk_size=chunk_size)

    def forward(self, idx, targets=None, state=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd

        if state is None:
            pos = 0
            layer_states = [None] * C.n_layer
        else:
            pos = state["pos"]
            layer_states = state["layers"]

        x = self.ln(self.embed(idx).unsqueeze(1))

        new_layers = []
        layer_logits = []
        for level in range(C.n_layer):
            x_sparse = F.relu(x @ self.encoder)
            yKV, layer_new_state = self.attn(x_sparse, x, state=layer_states[level], pos_offset=pos)
            new_layers.append(layer_new_state)
            yKV = self.ln(yKV)

            y_sparse = F.relu(yKV @ self.encoder_v)
            xy_sparse = self.drop(x_sparse * y_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, -1) @ self.decoder
            x = self.ln(x + self.ln(yMLP))

            # paper's BDH-GPU': merge next-token predictions from all layers
            layer_logits.append(x.view(B, T, D) @ self.lm_head)

        logits = torch.stack(layer_logits, dim=0).mean(dim=0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, {"pos": pos + T, "layers": new_layers}
