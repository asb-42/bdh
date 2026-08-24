"""Training loop for BDH and the Transformer baseline (shared, model-agnostic)."""

import dataclasses
import math
import os
import time
from contextlib import nullcontext

import torch

from bdh import detach_state, state_to_cpu

from pipeline import config as cfg_mod
from pipeline.config import Config, build_model, param_count, resolve_device, resolve_dtype
from pipeline.data import load_dataset


def get_lr(it: int, cfg: Config) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, data, cfg: Config, device, ctx, split: str) -> float:
    model.eval()
    losses = []
    if cfg.stateful_eval:
        # Stream-carrying evaluation: batches are sequential continuations and
        # the attention state persists across them (detached), matching the
        # carried-state training regime instead of penalizing it with a
        # cold start on unrelated random crops.
        stream = data.make_stream(split, cfg.block_size, cfg.batch_size)
        state = None
        for _ in range(cfg.eval_iters):
            x, y = stream.next_batch(device)
            with ctx:
                _, loss, state = model(x, y, state)
            losses.append(loss.item())
    else:
        for _ in range(cfg.eval_iters):
            x, y = data.get_batch(split, cfg.block_size, cfg.batch_size, device)
            with ctx:
                _, loss, _ = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def checkpoint_path(cfg: Config, tag: str) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    stem = f"{cfg.model}_{cfg.dataset}"
    if cfg.run_name:
        stem += f"_{cfg.run_name}"
    return os.path.join(cfg.out_dir, f"{stem}_{tag}.pt")


def save_checkpoint(cfg: Config, model, optimizer, step: int, best_val: float, tag: str,
                    state=None) -> None:
    torch.save(
        {
            # plain dict (not a pickled dataclass) so the checkpoint can be loaded
            # with torch.load(..., weights_only=True)
            "cfg": dataclasses.asdict(cfg),
            "step": step,
            "best_val": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "state": state,  # carried attention state (CPU tensors), or None
        },
        checkpoint_path(cfg, tag),
    )


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == torch.float16))

    data = load_dataset(cfg)
    print(f"dataset: {cfg.dataset} | train bytes {len(data.train)} | val bytes {len(data.val)}"
          + (f" | test bytes {len(data.test)}" if data.test is not None else ""))

    raw_model = build_model(cfg).to(device)
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        print(f"initialized weights from {cfg.init_from} (ckpt step {ckpt.get('step')})")
    n_params = param_count(raw_model)
    if cfg.model == "transformer" and cfg.baseline_n_layer <= 0:
        n_layers = raw_model.cfg.n_layer
        print(f"baseline n_layer auto-matched to {n_layers}; "
              f"BDH-equivalent params {cfg_mod.estimate_bdh_params(cfg):,}")
    print(f"model: {cfg.model} | params {n_params:,} | device {device} | dtype {dtype}")

    model = torch.compile(raw_model) if cfg.compile else raw_model

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    best_val = float("inf")
    t0 = time.time()
    running_loss = 0.0
    stream = (
        data.make_stream("train", cfg.block_size, cfg.batch_size)
        if cfg.sequential_batches
        else None
    )
    if cfg.carry_state and stream is None:
        print("warning: --carry-state without --sequential-batches carries state across "
              "unrelated random batches; consider --sequential-batches")

    state = None          # carried attention state (graph-attached within a TBPTT window)
    window_loss = None    # accumulated loss over the current TBPTT window
    # --no-bptt cuts gradients through time inside attention, so a multi-step
    # TBPTT window would be pointless; force per-step detachment.
    horizon = 1 if cfg.no_bptt else max(1, cfg.tbptt_horizon)

    def optimizer_step():
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for step in range(cfg.max_iters):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if stream is not None:
            x, y = stream.next_batch(device)
        else:
            x, y = data.get_batch("train", cfg.block_size, cfg.batch_size, device)

        with ctx:
            _, loss, new_state = model(x, y, state)

        if cfg.carry_state and horizon > 1:
            # Truncated BPTT: accumulate the loss over `horizon` minibatches (the
            # graph stays attached through the carried state), then backprop and
            # step once at the window boundary.
            window_loss = loss if window_loss is None else window_loss + loss
            state = new_state
            if (step + 1) % horizon == 0 or step == cfg.max_iters - 1:
                scaler.scale(window_loss).backward()
                optimizer_step()
                window_loss = None
                state = detach_state(state)
        elif cfg.carry_state:
            # horizon == 1: carry state forward but detach it every step
            # ("almost no BPTT" through state; context still persists).
            scaler.scale(loss).backward()
            optimizer_step()
            state = detach_state(new_state)
        else:
            state = None
            scaler.scale(loss).backward()
            optimizer_step()

        running_loss += loss.detach().item()

        if (step + 1) % cfg.log_interval == 0:
            ms = (time.time() - t0) * 1000 / cfg.log_interval
            t0 = time.time()
            print(f"step {step + 1:5d}/{cfg.max_iters} | loss {running_loss / cfg.log_interval:.4f} "
                  f"| lr {lr:.5f} | {ms:.0f} ms/step")
            running_loss = 0.0

        if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            val_loss = estimate_loss(model, data, cfg, device, ctx, "val")
            parts = [f"step {step + 1} | val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f} "
                     f"| bpw {val_loss / math.log(2):.3f}"]
            if data.test is not None:
                test_loss = estimate_loss(model, data, cfg, device, ctx, "test")
                parts.append(f"| test_loss {test_loss:.4f} | test_ppl {math.exp(test_loss):.2f}")
            print("".join(parts))
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    cfg, raw_model, optimizer, step + 1, best_val, "best",
                    state=state_to_cpu(detach_state(state)) if cfg.carry_state else None,
                )

    save_checkpoint(cfg, raw_model, optimizer, cfg.max_iters, best_val, "last",
                    state=state_to_cpu(detach_state(state)) if cfg.carry_state else None)
    print(f"done. best val_loss {best_val:.4f} (ppl {math.exp(best_val):.2f}) -> "
          f"{checkpoint_path(cfg, 'best')}")
