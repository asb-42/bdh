#!/usr/bin/env python3
"""Cross-arm checkpoint identity check (streaming, early-exit, read-only).

Answers one question precisely: do two ladder runs share history at a given phase,
or did they diverge - and where.

Design notes (why this is not just `cmp`):
  * A sampled comparison can prove DIFFERENCE but never IDENTITY. This streams every
    element of every tensor, so an "identical" verdict is a proof, not an estimate.
  * Early exit: the first mismatching chunk ends that tensor's scan, so genuinely
    different checkpoints cost kilobytes, not gigabytes. Only identical tensors are
    read in full - which is the information we actually want.
  * Per-tensor granularity matters: embed/lm_head are frozen after phase 0, so if they
    match while the latent differs, divergence is attributable to training, not init.
  * Shape mismatches are reported as such (different mlp_internal_dim_multiplier means
    the two arms are not at the same phase index at all - comparing them is meaningless).

Usage: arm_identity_check.py <armA> <armB> [lang ...]     e.g.  arm_identity_check.py ladG ladGR
Exit code 0 always; verdicts go to stdout as one line per (tensor, pair).
"""
import sys, os, glob, re, hashlib, json
import torch

OUT = os.environ.get("BDH_OUT", "/media/data/coding/bdh/out")
CHUNK = 1 << 22  # 4 Mi of float32 elements per read


def chain(arm):
    """lang -> (path, mult) for every *_last.pt of one arm."""
    res = {}
    for p in glob.glob(os.path.join(OUT, f"bdh_europarl_{arm}-*_last.pt")):
        m = re.match(rf"bdh_europarl_{re.escape(arm)}-(\w+)_last\.pt$", os.path.basename(p))
        if not m:
            continue
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        res[m.group(1)] = (p, int(ck["cfg"]["mlp_internal_dim_multiplier"]), int(ck.get("step", -1)))
        del ck
    return res


def tensor_stream(t):
    """Yield byte-chunks of a tensor without materialising it (mmap-friendly)."""
    flat = t.reshape(-1)
    n = flat.numel()
    for i in range(0, n, CHUNK):
        yield i, flat[i:i + CHUNK].contiguous().numpy().tobytes()


def compare_tensor(a, b):
    """Return (verdict, detail). verdict in {identical, differ, shape-differ}."""
    if tuple(a.shape) != tuple(b.shape):
        return "shape-differ", f"{tuple(a.shape)} vs {tuple(b.shape)}"
    ha = hashlib.sha256()
    hb = hashlib.sha256()
    for (i, ca), (j, cb) in zip(tensor_stream(a), tensor_stream(b)):
        if ca != cb:
            # locate the first differing element inside this chunk
            import numpy as np
            x = np.frombuffer(ca, dtype=np.float32)
            y = np.frombuffer(cb, dtype=np.float32)
            k = int(np.argmax(x != y))
            return "differ", f"first mismatch at flat index {i+k}: {x[k]!r} vs {y[k]!r}"
        ha.update(ca)
        hb.update(cb)
    return "identical", f"sha256 {ha.hexdigest()[:16]}" + ("== ok" if ha.digest() == hb.digest() else " MISMATCH")


def main():
    armA = sys.argv[1] if len(sys.argv) > 1 else "ladG"
    armB = sys.argv[2] if len(sys.argv) > 2 else "ladGR"
    langs = sys.argv[3:]
    A, B = chain(armA), chain(armB)
    shared = [l for l in (langs or sorted(set(A) & set(B)))]
    print(json.dumps({"armA": armA, "armB": armB, "only_in_A": sorted(set(A) - set(B)),
                      "only_in_B": sorted(set(B) - set(A)), "compared": shared}))
    tally = {}
    for lang in shared:
        pa, ma, sa = A[lang]
        pb, mb, sb = B[lang]
        if ma != mb:
            print(f"  {lang:>4s}: NOT COMPARABLE  mult {ma} vs {mb} (different phase index)")
            tally["not-comparable"] = tally.get("not-comparable", 0) + 1
            continue
        ca = torch.load(pa, map_location="cpu", mmap=True, weights_only=False)["model_state"]
        cb = torch.load(pb, map_location="cpu", mmap=True, weights_only=False)["model_state"]
        verdicts = []
        for k in ca.keys() & cb.keys():
            v, d = compare_tensor(ca[k], cb[k])
            verdicts.append((k, v, d))
        ident = [k for k, v, _ in verdicts if v == "identical"]
        diff = [(k, d) for k, v, d in verdicts if v != "identical"]
        tag = "IDENTICAL" if not diff else "DIFFERENT"
        print(f"  {lang:>4s}: {tag}  mult={ma}  steps={sa}/{sb}  "
              f"identical={len(ident)}/{len(verdicts)}" + (f"  first: {diff[0][0]} {diff[0][1]}" if diff else ""))
        tally[tag] = tally.get(tag, 0) + 1
        del ca, cb
    print(json.dumps({"tally": tally}))


if __name__ == "__main__":
    main()
