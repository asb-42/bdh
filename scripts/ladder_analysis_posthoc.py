#!/usr/bin/env python3
"""Post-hoc ladder analysis: interference matrix from phase-end checkpoints.

No reliance on corrupted ladder_analysis.txt. Evaluates each checkpoint
on target language + earlier languages to measure erosion.
"""
import math, sys, time
sys.path.insert(0, ".")

import numpy as np
import torch
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks

PHASES = ["es","pl","fr","de","cs","da","pt","fi","hu","bg","it","et","el","sk","sv","ro","nl","sl","lt"]

def eval_ppl(ckpt_path, langs, lang_mb=30, batch=4, iters=100):
    """Evaluate checkpoint on given languages, return {lang: ppl}."""
    blocks = _europarl_blocks("data", lang_mb * 1_000_000, langs=tuple(langs))
    model, cfg = _load_model(ckpt_path)
    device = torch.device("cuda")
    model = model.to(device).eval()
    bs = cfg["block_size"]
    results = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for lang in sorted(blocks):
            raw = blocks[lang]["val"] + blocks[lang]["test"]
            data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
            g = torch.Generator().manual_seed(1234)
            losses = []
            for _ in range(iters):
                ix = torch.randint(len(data) - bs - 1, (batch,), generator=g)
                x = torch.stack([data[i:i+bs] for i in ix]).to(device)
                y = torch.stack([data[i+1:i+1+bs] for i in ix]).to(device)
                _, loss, _ = model(x, targets=y)
                losses.append(loss.item())
            results[lang] = math.exp(np.mean(losses))
    del model; torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    print("=" * 72)
    print("LADDER INTERFERENCE ANALYSIS (post-hoc, from phase-end checkpoints)")
    print("=" * 72)

    # 1) Target-language acquisition
    print("\n--- P-Acq: target-language ppl per phase ---")
    acq = []
    for lang in PHASES:
        ckpt = f"out/bdh_europarl_lad-{lang}_last.pt"
        r = eval_ppl(ckpt, [lang])
        ppl = r.get(lang, float("nan"))
        acq.append(ppl)
        tag = "OK" if ppl <= 2.6 else f"FAIL({ppl:.2f}>2.6)"
        print(f"  {lang}: {ppl:.2f}  [{tag}]")
    print(f"  Peak: {max(acq):.2f}")

    # 2) Erosion tracking: en and es across all 19 phase-end ckpts
    print("\n--- P-Eros: erosion of en (base) and es (2nd phase) ---")
    en_ppls, es_ppls = [], []
    for lang in PHASES:
        ckpt = f"out/bdh_europarl_lad-{lang}_last.pt"
        r = eval_ppl(ckpt, ["en", "es"])
        en_p, es_p = r.get("en", float("nan")), r.get("es", float("nan"))
        en_ppls.append(en_p); es_ppls.append(es_p)
        print(f"  {lang}: en={en_p:.2f}  es={es_p:.2f}")

    en0, enN = en_ppls[0], en_ppls[-1]
    es0, esN = es_ppls[0], es_ppls[-1]
    print(f"\n  en drift: {en0:.2f} -> {enN:.2f}  ({enN-en0:+.2f} nats)")
    print(f"  es drift: {es0:.2f} -> {esN:.2f}  ({esN-es0:+.2f} nats)")
    print(f"  P-Eros verdict: {'PASS' if abs(esN-es0) < 0.3 else 'CONCERN'} (<0.3 nats drift)")

    # 3) Snapshot matrix at milestones
    print("\n--- Milestone snapshots ---")
    for mi in [4, 9, 14, 18]:
        lang = PHASES[mi]
        ckpt = f"out/bdh_europarl_lad-{lang}_last.pt"
        seen = ["en"] + PHASES[:mi+1]
        print(f"\n  Phase {mi+1} ({lang}), seen={seen}:")
        r = eval_ppl(ckpt, seen)
        for l in seen:
            print(f"    {l}: {r.get(l, float('nan')):.2f}")

    print("\n" + "=" * 72)
    print("Done.")
