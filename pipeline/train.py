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

    # Mechanism C write-gating / Mechanism B grow share the gradient-mask path:
    # gate_param_masks entries are (param, multiplier) applied before optimizer step.
    # BDH shares one (encoder, encoder_v, decoder) triple across depth levels, so
    # three masks cover everything neuron-indexed (decoder rows are h*N + n ordered).
    gate_param_masks = []
    route_neuron_mask = None
    elem_frozen_backup = None  # grow path: bit-exact frozen-path snapshots, restored after each step

    # Mechanism B/grow: with init_from + grow_mult, widen the latent by fresh
    # zero-init neurons; old neurons, embed and lm_head are frozen so new-phase
    # updates can only write into the new capacity (zero forgetting by
    # construction for the frozen path).
    grow_src = None
    if cfg.init_from and cfg.grow_mult > 0:
        if cfg.gate_from:
            raise ValueError("--grow-mult and --gate-from are mutually exclusive")
        grow_src = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        base_mult = int(grow_src["cfg"]["mlp_internal_dim_multiplier"])
        cfg.mlp_internal_dim_multiplier = base_mult + cfg.grow_mult

    raw_model = build_model(cfg).to(device)
    if cfg.init_from and grow_src is None:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        print(f"initialized weights from {cfg.init_from} (ckpt step {ckpt.get('step')})")

    if grow_src is not None:
        sd = grow_src["model_state"]
        nh, D = cfg.n_head, cfg.n_embd
        n_old = base_mult * D // nh
        n_new = cfg.mlp_internal_dim_multiplier * D // nh
        with torch.no_grad():
            raw_model.embed.weight.copy_(sd["embed.weight"])
            raw_model.lm_head.copy_(sd["lm_head"].to(device))
            for key in ("encoder", "encoder_v"):              # (nh, D, N)
                getattr(raw_model, key).data[:, :, :n_old] = sd[key].to(device)
            dec = raw_model.decoder.data.view(nh, n_new, -1)
            dec[:, :n_old, :] = sd["decoder"].to(device).view(nh, n_old, -1)
            for k, v in sd.items():                           # attention internals (freqs handled below)
                if k.startswith("attn.") and not k.endswith("freqs"):
                    name = k.split(".", 1)[1]
                    sub = raw_model.attn.get_submodule(name.rsplit(".", 1)[0]) if "." in name else raw_model.attn
                    leaf = name.rsplit(".", 1)[-1]
                    target = sub._parameters.get(leaf) or sub._buffers.get(leaf)
                    target.copy_(v.to(device))
            # RoPE freqs are exponent-normalized by n (/n in the exponent): keep
            # old neurons' frequencies verbatim; new neurons get their own scale.
            old_f = sd["attn.freqs"].to(device).view(-1)
            idx = torch.arange(n_old, n_new, dtype=torch.float32, device=device)
            new_f = 1.0 / (2 ** 16 ** ((idx // 2 * 2) / n_new)) / (2 * math.pi)
            freqs = torch.cat([old_f, new_f])
            raw_model.attn.freqs.data[:] = freqs.view(1, 1, 1, n_new)
        raw_model.embed.weight.requires_grad_(False)
        raw_model.lm_head.requires_grad_(False)
        if cfg.freeze_attn:
            for p in raw_model.attn.parameters():
                p.requires_grad_(False)
        keep = torch.zeros(n_new, device=device)
        keep[n_old:] = 1.0
        gate_param_masks = [
            (raw_model.encoder, keep.view(1, 1, -1)),                 # (nh, D, N)
            (raw_model.encoder_v, keep.view(1, 1, -1)),
            (raw_model.decoder, keep.repeat(nh).unsqueeze(1)),        # (nh*N, D)
        ]
        route_neuron_mask = keep.clone() if cfg.route_aware else None
        if cfg.route_aware:
            print(f"route-aware: prefix mask {n_old}..{n_new} | "
                  f"loss = {cfg.route_alpha}*prefix + {1 - cfg.route_alpha}*full")
        # Bit-exact frozen path (F-decay-leak fix): the zero-grad mask stops
        # learning but not AdamW's decoupled weight decay, which multiplies
        # every param with grad != None by (1 - lr_t * wd) each step. Snapshot
        # the frozen regions here and restore them after every optimizer step
        # (see optimizer_step). embed/lm_head need no restore: requires_grad
        # False removes them from the optimizer entirely; attn.freqs is a
        # buffer. Note: restoring at step end is exact by construction; a
        # wd=0 param group alone would NOT be, because AdamW applies decay
        # before the (zero) gradient update inside the same step.
        enc_d, encv_d = raw_model.encoder.data, raw_model.encoder_v.data
        dec_d = raw_model.decoder.data.view(nh, n_new, -1)
        elem_frozen_backup = [
            (enc_d[:, :, :n_old], enc_d[:, :, :n_old].detach().clone()),
            (encv_d[:, :, :n_old], encv_d[:, :, :n_old].detach().clone()),
            (dec_d[:, :n_old, :], dec_d[:, :n_old, :].detach().clone()),
        ]
        print(f"growth: {base_mult} -> {cfg.mlp_internal_dim_multiplier} mult "
              f"(+{n_new - n_old} neurons/head trainable) | old neurons + embed + lm_head frozen"
              f" (bit-exact via step-end restore) | attn {'frozen' if cfg.freeze_attn else 'UNFROZEN'}")

    if cfg.gate_from:
        gate_imp = None
        for path in filter(None, map(str.strip, cfg.gate_from.split(","))):
            g = torch.load(path, map_location="cpu", weights_only=False)
            imp = torch.as_tensor(g["neuron_importance"], dtype=torch.float32)
            if imp.dim() != 2:
                raise ValueError(f"{path}: neuron_importance must be (nh, N)")
            gate_imp = imp.clone() if gate_imp is None else torch.maximum(gate_imp, imp)
        keep = (1.0 - cfg.gate_alpha * gate_imp / gate_imp.max()).clamp_min_(0.0)
        nh = keep.size(0)
        gate_param_masks = [
            (raw_model.encoder, keep.to(device).unsqueeze(1)),      # (nh, D, N) <- (nh, 1, N)
            (raw_model.encoder_v, keep.to(device).unsqueeze(1)),
            (raw_model.decoder, keep.to(device).reshape(-1).unsqueeze(1)),  # (nh*N, D) <- (nh*N, 1)
        ]
        print(f"write-gating: alpha={cfg.gate_alpha} | sources={cfg.gate_from} | "
              f"neurons fully frozen {(keep == 0).float().mean():.1%} | "
              f"mean grad keep {keep.mean():.3f}")
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
        if gate_param_masks:
            for p, mk in gate_param_masks:
                if p.grad is not None:
                    p.grad.mul_(mk)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # F-decay-leak fix: bit-exact frozen path. AdamW's decoupled decay
        # moves masked elements even at zero gradient (decay applies before
        # the zero-gradient update inside the same step), so a wd=0 param
        # group alone would not suffice. Restore the snapshots taken at
        # growth time after every step.
        if elem_frozen_backup:
            for dst_t, snap in elem_frozen_backup:
                dst_t.copy_(snap)

    for step in range(cfg.max_iters):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if stream is not None:
            x, y = stream.next_batch(device)
        else:
            x, y = data.get_batch("train", cfg.block_size, cfg.batch_size, device)

        with ctx:
            if route_neuron_mask is not None:
                # Route-aware training: prefix-masked forward + full forward, mix losses.
                # route_alpha = prefix fraction (higher = stronger route-aware signal).
                _, prefix_loss, new_state = model(x, y, state, neuron_mask=route_neuron_mask)
                _, full_loss, _ = model(x, y, state)
                loss = cfg.route_alpha * prefix_loss + (1 - cfg.route_alpha) * full_loss
            else:
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
