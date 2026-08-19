"""Benchmark: measure per-step wall-clock and estimated FLOPs/throughput for a model.

Used to establish the BDH-vs-GPT cost baseline on whatever hardware is available
(CPU or GPU). Reports:
  - params
  - ms/step (forward+backward, after warmup)
  - estimated FLOPs/step (forward+backward)
  - effective TFLOP/s

Run via:  python -m pipeline.run bench --model bdh --batch-size 32 --block-size 128
"""

import time

import torch

from pipeline.config import (
    Config,
    build_model,
    estimate_flops,
    param_count,
    resolve_device,
    resolve_dtype,
)
from pipeline.data import load_dataset


def bench(cfg: Config, warmup: int = 3, steps: int = 10) -> None:
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    data = load_dataset(cfg)

    model = build_model(cfg).to(device)
    n_params = param_count(model)
    flops = estimate_flops(cfg, cfg.batch_size, cfg.block_size)

    x, y = data.get_batch("train", cfg.block_size, cfg.batch_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        with torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else _null():
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if cfg.compile:
        model = torch.compile(model)

    for _ in range(warmup):
        step()

    t0 = time.time()
    for _ in range(steps):
        step()
    dt = (time.time() - t0) / steps

    print(f"model={cfg.model} dataset={cfg.dataset} device={device} dtype={dtype} compile={cfg.compile}")
    print(f"params={n_params:,}  block={cfg.block_size}  batch={cfg.batch_size}")
    print(f"per-step {dt * 1000:.1f} ms | est FLOPs/step {flops / 1e12:.3f} T | "
          f"effective {flops / dt / 1e12:.2f} TFLOP/s")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
