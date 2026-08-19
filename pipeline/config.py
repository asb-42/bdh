"""Configuration, parameter estimation, and model construction for the BDH pipeline."""

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    model: str = "bdh"                     # bdh | transformer
    dataset: str = "shakespeare"           # shakespeare | wikitext2
    n_layer: int = 4
    n_embd: int = 96
    n_head: int = 4
    dropout: float = 0.1
    mlp_internal_dim_multiplier: int = 24  # BDH only; total neurons = multiplier * n_embd
    vocab_size: int = 256                  # byte-level
    block_size: int = 128                  # training sequence length
    baseline_n_layer: int = 0              # transformer only; 0 = auto-match BDH params
    chunk_size: int = 64                   # bdh-linear only; attention scan chunk size
    batch_size: int = 32
    max_iters: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 30
    lr_decay_iters: int = 300
    min_lr: float = 1e-4
    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    compile: bool = True
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1337
    data_dir: str = "data"
    out_dir: str = "out"
    tag: str = "best"                      # checkpoint tag for eval
    num_samples: int = 3                   # eval: number of generated samples
    max_new_tokens: int = 200              # eval: tokens to generate per sample


def estimate_bdh_params(cfg: Config) -> int:
    d, nh, mult, v = cfg.n_embd, cfg.n_head, cfg.mlp_internal_dim_multiplier, cfg.vocab_size
    n = mult * d // nh  # latent dim per head
    return 3 * (nh * d * n) + 2 * (v * d)


def estimate_transformer_params(cfg: Config, n_layer: int) -> int:
    d, v, b = cfg.n_embd, cfg.vocab_size, cfg.block_size
    fixed = 2 * v * d + b * d + 2 * d          # wte + lm_head + wpe + ln_f
    per_block = 12 * d * d + 4 * d             # attn (3d^2 + d^2) + mlp (4d^2 + 4d^2) + 2 ln (4d)
    return fixed + n_layer * per_block


def match_transformer_layers(cfg: Config) -> int:
    target = estimate_bdh_params(cfg)
    d, v, b = cfg.n_embd, cfg.vocab_size, cfg.block_size
    fixed = 2 * v * d + b * d + 2 * d
    per_block = 12 * d * d + 4 * d
    return max(1, round((target - fixed) / per_block))


def build_model(cfg: Config):
    if cfg.model == "bdh":
        from bdh import BDH, BDHConfig

        model = BDH(
            BDHConfig(
                n_layer=cfg.n_layer,
                n_embd=cfg.n_embd,
                dropout=cfg.dropout,
                n_head=cfg.n_head,
                mlp_internal_dim_multiplier=cfg.mlp_internal_dim_multiplier,
                vocab_size=cfg.vocab_size,
            )
        )
        return model
    if cfg.model == "bdh-linear":
        from bdh import BDHConfig
        from bdh_linear import BDHLinear

        model = BDHLinear(
            BDHConfig(
                n_layer=cfg.n_layer,
                n_embd=cfg.n_embd,
                dropout=cfg.dropout,
                n_head=cfg.n_head,
                mlp_internal_dim_multiplier=cfg.mlp_internal_dim_multiplier,
                vocab_size=cfg.vocab_size,
            ),
            chunk_size=cfg.chunk_size,
        )
        return model
    if cfg.model == "transformer":
        from pipeline.transformer import GPT, GPTConfig

        n_layer = cfg.baseline_n_layer if cfg.baseline_n_layer > 0 else match_transformer_layers(cfg)
        model = GPT(
            GPTConfig(
                n_layer=n_layer,
                n_embd=cfg.n_embd,
                n_head=cfg.n_head,
                dropout=cfg.dropout,
                vocab_size=cfg.vocab_size,
                block_size=cfg.block_size,
            )
        )
        return model
    raise ValueError(f"unknown model: {cfg.model}")


def resolve_device(cfg: Config):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(cfg: Config, device) -> torch.dtype:
    if cfg.dtype != "auto":
        return getattr(torch, cfg.dtype)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _add_arg(parser, name, default):
    if isinstance(default, bool):
        parser.add_argument(f"--{name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=default)
    elif isinstance(default, int):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
    elif isinstance(default, float):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    else:
        parser.add_argument(f"--{name.replace('_', '-')}", type=str, default=default)


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="BDH training/evaluation pipeline")
    for f in Config.__dataclass_fields__.values():
        _add_arg(parser, f.name, f.default)
    args = parser.parse_args(argv)
    return Config(**{f.name: getattr(args, f.name) for f in Config.__dataclass_fields__.values()})
