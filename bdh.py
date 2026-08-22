# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    block_size: int | None = None  # max context for generation; None = unbounded
    attn_window: int = 1024  # quadratic attention: max cached past tokens (0 = unlimited)
    no_bptt: bool = False  # detach K/V in attention: no backprop through time (paper Sec. 5.2)


def detach_state(state):
    """Detach a model state (as returned by ``BDH.forward``) from the autograd graph."""
    if state is None:
        return None
    layers = []
    for s in state["layers"]:
        if s is None:
            layers.append(None)
        elif isinstance(s, dict):
            layers.append({k: v.detach() for k, v in s.items()})
        else:
            layers.append(s.detach())
    return {"pos": state["pos"], "layers": layers}


def state_to_cpu(state):
    """Move a detached model state to CPU (for checkpointing)."""
    if state is None:
        return None
    layers = []
    for s in state["layers"]:
        if s is None:
            layers.append(None)
        elif isinstance(s, dict):
            layers.append({k: v.cpu() for k, v in s.items()})
        else:
            layers.append(s.cpu())
    return {"pos": state["pos"], "layers": layers}


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, V, state=None, pos_offset=0):
        """Causal attention of Q over V, optionally attending to cached past tokens.

        state: None or {"k": rotated past keys (B,nh,S,N), "v": past values (B,1,S,D)}.
        pos_offset: absolute position of the first query token (continues across
            calls when state is carried over).
        Returns (output, new_state).
        """
        assert self.freqs.dtype == torch.float32
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                pos_offset,
                pos_offset + T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        if self.config.no_bptt:
            # Paper Sec. 5.2: detach keys and values so no gradient flows back
            # through time; the current-token query path keeps its gradient.
            QR_k = QR.detach()
            V = V.detach()
        else:
            QR_k = QR

        if state is None:
            scores = (QR @ QR_k.mT).tril(diagonal=-1)
            out = scores @ V
            new_state = {"k": QR_k, "v": V}
        else:
            Kc, Vc = state["k"], state["v"]
            # Past keys were rotated at their (absolute) positions, so the dot
            # product against QR reproduces the RoPE-relative scores U^{t-tau}.
            scores_past = QR @ Kc.mT
            scores_intra = (QR @ QR_k.mT).tril(diagonal=-1)
            out = torch.cat([scores_past, scores_intra], dim=-1) @ torch.cat(
                [Vc, V], dim=2
            )
            new_state = {"k": torch.cat([Kc, QR_k], dim=2), "v": torch.cat([Vc, V], dim=2)}
        return out, new_state


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, state=None):
        """State-space forward pass.

        state: None, or {"pos": int, "layers": [per-layer attention state]} as
        returned by a previous call. Carrying state across calls continues both
        the absolute RoPE positions and the attention memory (the paper's
        recurrent state rho), enabling state carry-over / truncated BPTT.
        Returns (logits, loss, new_state).
        """
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        if state is None:
            pos = 0
            layer_states = [None] * C.n_layer
        else:
            pos = state["pos"]
            layer_states = state["layers"]

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        new_layers = []
        for level in range(C.n_layer):
            layer_state = layer_states[level]
            if (
                layer_state is not None
                and isinstance(layer_state, dict)
                and C.attn_window
                and layer_state["k"].size(2) > C.attn_window
            ):
                layer_state = {
                    "k": layer_state["k"][:, :, -C.attn_window :],
                    "v": layer_state["v"][:, :, -C.attn_window :],
                }

            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV, layer_new_state = self.attn(
                x_sparse,
                x,
                state=layer_state,
                pos_offset=pos,
            )
            new_layers.append(layer_new_state)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        new_state = {"pos": pos + T, "layers": new_layers}
        return logits, loss, new_state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx
                if self.config.block_size is not None:
                    idx_cond = idx_cond[:, -self.config.block_size :]
                logits, _, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        finally:
            if was_training:
                self.train()
        return idx
