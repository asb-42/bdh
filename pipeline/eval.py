"""Evaluation: load a checkpoint, report perplexity, and generate samples."""

import math
import os
from contextlib import nullcontext

import torch

from pipeline.config import Config, build_model, param_count, resolve_device, resolve_dtype
from pipeline.data import load_dataset
from pipeline.train import checkpoint_path, estimate_loss


def load_checkpoint(cfg: Config, tag: str):
    path = checkpoint_path(cfg, tag)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # older checkpoints may contain a pickled Config dataclass
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["cfg"]
    if isinstance(model_cfg, dict):
        model_cfg = Config(**model_cfg)
    model = build_model(model_cfg).to("cpu")
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def evaluate(cfg: Config, tag: str = "best", num_samples: int = 3, max_new_tokens: int = 200) -> None:
    model, ckpt = load_checkpoint(cfg, tag)
    print(f"checkpoint: {tag} (step {ckpt['step']}, best_val {ckpt['best_val']:.4f})")
    print(f"model: {cfg.model} | params {param_count(model):,}")

    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    model.to(device)
    model.eval()

    data = load_dataset(cfg)
    val_loss = estimate_loss(model, data, cfg, device, ctx, "val")
    print(f"val_loss {val_loss:.4f} | perplexity {math.exp(val_loss):.2f} | bits/byte {val_loss / math.log(2):.3f}")
    if data.test is not None:
        test_loss = estimate_loss(model, data, cfg, device, ctx, "test")
        print(f"test_loss {test_loss:.4f} | perplexity {math.exp(test_loss):.2f} | bits/byte {test_loss / math.log(2):.3f}")

    print("\n--- samples ---")
    for i in range(num_samples):
        prompt = bytearray("To be or ", "utf-8") if cfg.dataset == "shakespeare" else bytearray("\n = The ", "utf-8")
        idx = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=3)
        text = bytes(out.to(torch.uint8).to("cpu").squeeze(0)).decode("utf-8", errors="backslashreplace")
        print(f"[{i}] {text!r}")
