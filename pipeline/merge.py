"""Model merging by neuron-dimension concatenation (paper Sec. 5.1).

BDH-GPU scales uniformly in the neuron dimension n, so two models trained
separately can be merged into one larger model without finetuning:

  - all parameter tensors with an n dimension are concatenated along n
    (`decoder` rows, `encoder`/`encoder_v` latent dims, RoPE freqs),
  - everything without an n dimension (token embedding, lm_head) is averaged.

Usage:
    python -m pipeline.run merge --ckpt-a out/a_best.pt --ckpt-b out/b_best.pt \
        --out out/merged.pt [--tag best]
"""

import dataclasses

import torch

from bdh import BDHConfig

# keys whose tensors carry the neuron dimension (n)
_N_DIM_KEYS = {"encoder", "encoder_v"}          # concat along last dim
_DECODER_KEY = "decoder"                        # (nh*N, D): concat within heads
_FREQS_KEY = "attn.freqs"                       # (1, 1, 1, N): concat along last dim


def merge_state_dicts(sd_a: dict, sd_b: dict) -> dict:
    """Merge two BDH / BDHLinear state dicts into one double-width model."""
    missing = set(sd_a) ^ set(sd_b)
    if missing:
        raise ValueError(f"state dicts disagree over keys: {sorted(missing)}")

    nh = sd_a["encoder"].shape[0]
    merged = {}
    for key, a in sd_a.items():
        b = sd_b[key]
        if key == _DECODER_KEY:
            na, d = a.shape
            nb = b.shape[0]
            if d != b.shape[1] or na % nh or nb % nh:
                raise ValueError(f"incompatible decoder shapes: {a.shape} vs {b.shape}")
            cat = torch.cat(
                [a.view(nh, na // nh, d), b.view(nh, nb // nh, d)], dim=1
            )  # (nh, n+n', D)
            merged[key] = cat.reshape(nh * (na // nh + nb // nh), d)
        elif key in _N_DIM_KEYS or key == _FREQS_KEY:
            merged[key] = torch.cat([a, b], dim=-1)
        else:
            merged[key] = (a + b) / 2
    return merged


def merge_configs(cfg_a: BDHConfig, cfg_b: BDHConfig) -> BDHConfig:
    """Merged config: neuron count (mlp_internal_dim_multiplier) adds up."""
    if dataclasses.replace(cfg_a, mlp_internal_dim_multiplier=0) != dataclasses.replace(
        cfg_b, mlp_internal_dim_multiplier=0
    ):
        raise ValueError(
            f"configs differ beyond neuron count:\n  {cfg_a}\n  {cfg_b}"
        )
    return dataclasses.replace(
        cfg_a,
        mlp_internal_dim_multiplier=cfg_a.mlp_internal_dim_multiplier
        + cfg_b.mlp_internal_dim_multiplier,
    )


def merge_checkpoints(ckpt_a: dict, ckpt_b: dict) -> dict:
    """Merge two checkpoints (as saved by pipeline.train.save_checkpoint)."""
    cfg_a = BDHConfig(**{k: v for k, v in ckpt_a["cfg"].items()
                         if k in {f.name for f in dataclasses.fields(BDHConfig)}})
    cfg_b = BDHConfig(**{k: v for k, v in ckpt_b["cfg"].items()
                         if k in {f.name for f in dataclasses.fields(BDHConfig)}})
    merged_cfg = merge_configs(cfg_a, cfg_b)
    merged_sd = merge_state_dicts(ckpt_a["model_state"], ckpt_b["model_state"])
    return {
        "cfg": dataclasses.asdict(merged_cfg),
        "step": max(ckpt_a["step"], ckpt_b["step"]),
        "best_val": min(ckpt_a["best_val"], ckpt_b["best_val"]),
        "model_state": merged_sd,
        "optimizer_state": None,
        "state": None,
        "merged_from": [ckpt_a.get("source", "?"), ckpt_b.get("source", "?")],
    }


def load_ckpt(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def run_merge(path_a: str, path_b: str, out_path: str) -> None:
    ckpt_a, ckpt_b = load_ckpt(path_a), load_ckpt(path_b)
    ckpt_a["source"] = path_a
    ckpt_b["source"] = path_b
    merged = merge_checkpoints(ckpt_a, ckpt_b)
    torch.save(merged, out_path)

    # sanity: the merged model must build and load
    from bdh import BDH
    from pipeline.config import param_count

    cfg_bd = BDHConfig(**{k: v for k, v in merged["cfg"].items()
                          if k in {f.name for f in dataclasses.fields(BDHConfig)}})
    model = BDH(cfg_bd)
    model.load_state_dict(merged["model_state"])
    print(f"merged -> {out_path}")
    print(f"  params: {param_count(model):,} | "
          f"neurons: {cfg_bd.n_head * (cfg_bd.n_embd * cfg_bd.mlp_internal_dim_multiplier // cfg_bd.n_head):,}")
