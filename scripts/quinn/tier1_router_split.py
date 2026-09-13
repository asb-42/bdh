"""Tier 1: disjoint-crop router split (eval-only) - tests the routing-circularity claim.

PRE-REGISTERED design (run before any number is seen):
  Per domain, crops are drawn from two DISJOINT halves of a held-out byte window.
  * SELECTION crops (early half) choose ONE fixed route per domain:
      fixed_route(d) = argmin_k  mean_over_selection_crops of early-position NLL at route k.
  * TEST crops (late half) are served under that fixed route; we also serve them
    ONLINE (per-test-crop argmin on that crop's own early positions, i.e. the
    protocol the paper reports) and by ORACLE (the domain's own acquisition width)
    and by JOINT (full width).
  Gate G1: fixed_route == oracle_width for all domains (selection generalizes to unseen crops).
  Gate G2: fixed-test ppl within the online-vs-oracle band (fixed assignment loses nothing vs online).
  Gate G3: fixed-test ppl within +4.3% / +8.0% median/max of the acquisition exit (same routed-cost band).
  Gate G4: selection and test crops share no byte range (disjointness asserted by construction).
  Falsifier: any domain where fixed_route != oracle_width, or fixed ppl materially above online ppl.

No training, no new optimizer state, nothing persisted under out/ except the CSV and log.
This answers whether 40/40 routing on the same crop is a same-crop artifact.
"""
import argparse
import csv
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--domains", required=True, help="name:path,... in acquisition order")
    ap.add_argument("--oracle-routes", required=True, help="name:width,... true prefix width per domain")
    ap.add_argument("--mb", type=int, default=30)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--sel", type=int, default=30, help="selection crops per domain")
    ap.add_argument("--test", type=int, default=30, help="test crops per domain")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="docs/reports/data/tier1_router_split/results.csv")
    args = ap.parse_args()

    from pipeline.analyze import _load_model

    model, cfg = _load_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    nh, D = model.config.n_head, model.config.n_embd
    N_full = model.config.mlp_internal_dim_multiplier * D // nh
    bs = cfg["block_size"]
    assert 0 < args.window < bs

    enc_b = model.encoder.data.clone()
    encv_b = model.encoder_v.data.clone()
    dec_b = model.decoder.data.clone()

    def set_prefix(k):
        m = torch.zeros(N_full, device=device)
        m[:k] = 1.0
        with torch.no_grad():
            model.encoder.data.copy_(enc_b * m.view(1, 1, -1))
            model.encoder_v.data.copy_(encv_b * m.view(1, 1, -1))
            model.decoder.data.copy_(dec_b * m.repeat(nh).unsqueeze(1))

    routes = [int(r) for r in args.routes.split(",")]
    R = len(routes)
    oracle = {}
    for item in filter(None, map(str.strip, args.oracle_routes.split(","))):
        n, w = item.split(":", 1)
        w = int(w)
        assert w in routes, f"oracle width {w} not in routes"
        oracle[n] = routes.index(w)

    g_sel = torch.Generator().manual_seed(1234)
    g_test = torch.Generator().manual_seed(2345)
    doms = []
    for item in filter(None, map(str.strip, args.domains.split(","))):
        name, path = item.split(":", 1)
        arr = np.frombuffer(open(path, "rb").read(args.mb * 1_000_000)[-2_000_000:], dtype=np.uint8)
        mid = len(arr) // 2
        # selection crops live entirely before mid, test crops entirely after mid
        sel_offs = torch.randint(0, mid - bs, (args.sel,), generator=g_sel).tolist()
        tst_offs = [mid + int(o) for o in
                    torch.randint(0, len(arr) - 2 * bs - mid, (args.test,), generator=g_test).tolist()]
        assert max(sel_offs) + bs <= mid < min(tst_offs), "G4 selection/test crops must be disjoint"
        sel = np.stack([arr[i:i + bs] for i in sel_offs])
        tst = np.stack([arr[i:i + bs] for i in tst_offs])
        doms.append((name, torch.from_numpy(sel.astype(np.int64)),
                     torch.from_numpy(tst.astype(np.int64))))

    def eval_crops(blocks, n):
        out = torch.zeros(n, bs)
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
        with torch.no_grad(), amp:
            for b0 in range(0, n, args.batch):
                xb = blocks[b0:b0 + args.batch].to(device)
                yb = xb.roll(shifts=-1, dims=1)
                logits, _, _ = model(xb)
                l = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                    reduction="none").view(xb.shape[0], bs)
                out[b0:b0 + xb.shape[0]] = l.float().cpu()
        return out

    rows = []
    for name, sel_blocks, tst_blocks in doms:
        s_early = torch.zeros(R)
        s_late = torch.zeros(R)
        t_early = torch.zeros(R, args.test)
        t_late = torch.zeros(R, args.test)
        for ri, k in enumerate(routes):
            set_prefix(min(k, N_full))
            lo = eval_crops(sel_blocks, args.sel)
            lt = eval_crops(tst_blocks, args.test)
            s_early[ri] = lo[:, :args.window].mean()
            s_late[ri] = lt[:, :args.window].mean()
            t_early[ri] = lt[:, :args.window].mean(dim=1)
            t_late[ri] = lt[:, args.window:].mean(dim=1)

        fixed_idx = int(s_early.argmin())
        oi = oracle[name]

        def ppl_from(idx_vec, late_mat):
            served = late_mat[idx_vec, torch.arange(args.test)]
            return math.exp(served.mean().item())

        fixed_ppl = ppl_from(torch.tensor([fixed_idx] * args.test), t_late)
        online_idx = t_early.argmin(dim=0)
        online_ppl = ppl_from(online_idx, t_late)
        oracle_ppl = ppl_from(torch.tensor([oi] * args.test), t_late)
        fixed_matches_oracle = int(fixed_idx == oi)
        online_acc = int((online_idx == oi).sum())

        set_prefix(N_full)
        ltj = eval_crops(tst_blocks, args.test)
        joint_ppl = math.exp(ltj[:, args.window:].mean(dim=1).mean().item())

        rows.append(dict(domain=name, own_width=routes[oi], fixed_width=routes[fixed_idx],
                         oracle_width=routes[oi], sel_early_loss=float(s_early[fixed_idx]),
                         fixed_ppl=fixed_ppl, online_ppl=online_ppl, oracle_ppl=oracle_ppl,
                         joint_ppl=joint_ppl, g1_fixed_is_oracle=fixed_matches_oracle,
                         online_acc_on_test=f"{online_acc}/{args.test}",
                         fixed_over_oracle=fixed_ppl / oracle_ppl,
                         online_over_oracle=online_ppl / oracle_ppl))
        print(f"{name:>4} fixed={routes[fixed_idx]:>6} oracle={routes[oi]:>6} "
              f"G1={'OK' if fixed_matches_oracle else 'MISS'} "
              f"fixed/oracle={fixed_ppl/oracle_ppl:.3f} online/oracle={online_ppl/oracle_ppl:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]

    g1_ok = sum(r["g1_fixed_is_oracle"] for r in rows)
    print(f"\n=== G1 fixed==oracle: {g1_ok}/{len(rows)} ===")
    fo = sorted(r["fixed_over_oracle"] for r in rows)
    print(f"fixed/oracle median={fo[len(fo)//2]:.4f} range={fo[0]:.4f}-{fo[-1]:.4f}")
    oo = sorted(r["online_over_oracle"] for r in rows)
    print(f"online/oracle median={oo[len(oo)//2]:.4f} range={oo[0]:.4f}-{oo[-1]:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
