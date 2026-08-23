# Paper-Parity Plan: Closing the Gap Between the Repo and the BDH Paper

**Date:** 2026-08-22
**Reference:** *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain* (arXiv:2509.26507)
**Goal:** Implement the paper-described capabilities that are missing from this repository, one phase at a time, ordered by value-to-effort ratio. Each phase is independently useful and ends in a commit.

---

## Phase 1 — State carry-over + Truncated BPTT (TBPTT) ⭐ highest value

The paper's actual training regime (Appendix B): the attention state ρ persists
across minibatches, batches are temporally coherent continuations of each other,
and gradients are truncated every K tokens.

**Work items:**
1. **Stateful forward** — `BDH.forward(idx, targets=None, state=None)` returns
   `(logits, loss, new_state)` where `state` is a per-layer list of attention
   states ρ (`B, nh, T?, D, N` for quadratic; `B, nh, D, N` for linear).
   - Quadratic path: seed scores with cross-batch terms `Q_t @ (Σ_τ ρ_τ K_τ)ᵀ`
     — requires carrying a bounded window of past K/V (window = TBPTT horizon).
   - Linear path ([`bdh_linear.py`](../bdh_linear.py)): natural fit — carry the
     recurrent `state` tensor directly across calls.
2. **Sequential batch sampler** — new mode in [`pipeline/data.py`](../pipeline/data.py):
   instead of random crops, serve batch *b* as the continuation of batch *b−1*
   (B disjoint streams advancing in lockstep), mirroring the paper's Europarl
   setup (Fig. 15 minibatches).
3. **Truncation** — config `tbptt_horizon: int` (in minibatches); detach carried
   state when the horizon is exceeded. `tbptt_horizon=1` ≈ no BPTT through state.
4. **Config/CLI wiring** — `carry_state: bool`, `tbptt_horizon: int`,
   `sequential_batches: bool` in [`pipeline/config.py`](../pipeline/config.py);
   thread through [`pipeline/train.py`](../pipeline/train.py),
   [`pipeline/eval.py`](../pipeline/eval.py) (eval must also carry state),
   checkpointing (save/restore ρ so training resumes exactly).
5. **Validation** — loss on temporally-coherent streams should drop below the
   random-crop baseline at equal steps (the paper's "learns faster per token").

**Acceptance:** `python -m pipeline.run train --carry-state --sequential-batches --tbptt-horizon 8`
trains, checkpoints round-trip including state, and val loss improves vs. baseline.

## Phase 2 — No-BPTT training flag (tiny)

Reproduce paper §5.2: detach K/V inside the attention block so no gradient flows
through time.

**Work items:**
1. Config flag `no_bptt: bool`; in both attention implementations, `.detach()`
   the key/query tensor before the score matmul (and V in the state update).
2. CLI wiring + a short experiment note comparing losses with/without.

**Acceptance:** `--no-bptt` trains to roughly the paper's reported degradation
(loses cross-language alignment on translation-style tasks; language modeling
survives).

## Phase 3 — ALiBi state damping

Paper §4.1: RoPE combined with ALiBi damping prevents stale-context noise.

**Work items:**
1. Add optional per-head damping vector `a_h ≥ 0`; multiply accumulated state by
   `exp(-a_h · Δt)` per elapsed token (linear path: factor the decay into the
   chunked scan; quadratic path: add `-a_h·(t−τ)` to scores or damp state).
2. Config `alibi: bool` + damping-strength hyperparameter.
3. Long-context eval: perplexity vs. sequence position curve, with/without.

**Acceptance:** with ALiBi enabled, per-position loss stays flat at positions
beyond the training block size; without it, it degrades.

## Phase 4 — Model merging CLI

Paper §5.1: concatenate two checkpoints along the neuron dimension n.

**Work items:**
1. New script `scripts/merge_models.py` (or `pipeline.run merge`):
   - concat `encoder`, `encoder_v`, `decoder`, RoPE freq buffers along n/N dim;
   - average `embed` and `lm_head`;
   - write merged checkpoint with updated config.
2. Requires consistent head layout (concat within heads, then heads).
3. Eval command runs the merged model without finetuning; report per-task losses.

**Acceptance:** two models trained on different data subsets merge and produce
sane (if mixed) generations, matching the paper's qualitative result.

## Phase 5 — Interpretability tooling

Paper §4–5 analyses, as offline tooling over trained checkpoints:

1. **Sparsity logging** — during training, log fraction of non-zero entries of
   `x_sparse` / `xy_sparse` per layer (paper reports ~5%); cheap, add to the
   training loop behind a flag.
2. **Graph extraction** — script computing `G = decoder_x @ encoder` per head,
   thresholding at β, and reporting: degree histograms (power-law fit), Newman
   modularity (Louvain via `networkx`, optional dependency), core-periphery stats.
3. **Synapse readout** — dump per-layer ρ entries for chosen neuron pairs across
   a prompt; correlate synapse activation with concept presence (the paper's
   "currency synapse" methodology).

**Acceptance:** `python -m pipeline.run analyze --ckpt ... --graph --sparsity`
produces the paper-style plots/statistics from a trained checkpoint.

## Phase 6 — BDH-GPU′ variant (largest effort)

Paper's scaled-production extensions:

1. **xLSTM-like gating** of state updates: `ρ ← g ⊙ ρ + (1−g) ⊙ (v xᵀ)` with
   learned gate `g = σ(...)`.
2. **Per-layer logit merging**: compute logits from every layer's v and average.
3. Behind a config flag `variant: vanilla | gated | multilogit`.

**Acceptance:** gated variant trains stably and matches/beats vanilla at equal
params on the repo's wikitext benchmark.

---

## Execution order & commits

| Order | Phase | Effort | Risk | Status |
|-------|-------|--------|------|--------|
| 1 | Phase 1 (TBPTT + carry-over) | medium | medium — touches model API | ✅ done (`eef5f7e`) |
| 2 | Phase 2 (no-BPTT flag) | tiny | none | ✅ done (`9b28ed6`) |
| 3 | Phase 3 (ALiBi) | small | low | ✅ done (`c282338`) |
| 4 | Phase 4 (merging) | small | low | ✅ done (`22f7276`) |
| 5 | Phase 5 (tooling) | medium | low | ✅ done (`de0d04f`) |
| 6 | Phase 6 (BDH-GPU′) | large | medium | ✅ done (`067fe9f`) |

One commit per phase. Phases 1–2 share test infrastructure (stateful forward),
so they land back-to-back. Every phase keeps the existing stateless behavior as
the default so current scripts/CLIs remain valid.

## Known follow-ups

- Gated scan in `bdh_prime.py` is a per-token recurrence; a fused parallel
  scan would be needed for GPU-scale training of BDH-GPU′.
- ALiBi damping uses a uniform rate across heads; per-head (or per-edge,
  paper-faithful u(i,j)) rates are a natural extension.
- Quadratic-path state carry-over keeps an unbounded KV cache when
  `attn_window=0`; window trimming is approximate w.r.t. full attention only
  when decay is disabled.

## Findings log (append-only)

- **2026-08-23 (RTX 4090, env bring-up):** user-level `~/.pip/pip.conf` (NVIDIA
  PyIndex leftover) forces an extra index `pypi.ngc.nvidia.com` that does not
  resolve on this network; neutralized project-locally via `.venv/pip.conf`
  (site config overrides user). Env: torch 2.13.0+cu130, driver 595.84, sm_89;
  exactness invariant passes ≤1.34e-07.
- **2026-08-23 (bench, handover §6):** handover memory guidance corrected — the
  25M preset needs batch ≤ 8 at block 512 (`bdh-linear` autograd retains a fp32 ρ
  per chunk per layer ≈ 26 GB at batch 16). Measured ms/step @ batch 8 / block
  512 / 25M params: `bdh` 82.8, `bdh-linear` 194.8, `transformer` 14.6 →
  compute-match factors GPT ≈ 5.7×(`bdh`) and ≈ 13.3×(`bdh-linear`) steps.
  Chunked scan slower than quadratic attention at block 512. Details in
  HANDOVER §7.
- **2026-08-23 (§7.1 baseline, wikitext-2):** matched-param + matched-wall-clock
  comparison at block 512 / batch 8 / bf16 / compile, both models ~25M params
  (GPT auto-matched to 32 layers). `bdh` 10k steps @ 84 ms: **val 1.1311,
  test ppl 3.19** (`out/bdh_wikitext2_best.pt`). GPT 57k steps (=5.7×) @ 17 ms:
  final val 1.1235, best val 1.0911 @45.6k, test ppl 3.11
  (`out/transformer_wikitext2_best.pt`). Verdict: vanilla BDH trails the GPT
  baseline by ~0.03–0.04 nats val on short-context LM in this untuned regime.
  Process note: first GPT run was silently mismatched (manual cmd omitted
  `--mlp-internal-dim-multiplier` → auto-match built a 4.99M model) — always
  check the logged params line before comparing. Next lever per handover §7.2:
  BDH's intended regime is TBPTT + state carry-over, untested here.
- **2026-08-23 (§7.2 carry-over study):** paper-regime training **hurts** on
  wikitext-2 at a fixed 41.9M-token budget. Ranking (best val): random-crop
  b8/10k 1.1311; ctrl b4/20k no-carry **1.1222** (batch size innocent, even
  slightly better); carry+sequential h1@b4 1.2829, h2@b4 1.3254, h4@b2 1.3428
  (monotonically worse with horizon; all carry curves still descending at cap —
  horizon arms get fewer optimizer updates at equal tokens). Attribution: the
  carry/sequential regime itself costs ~0.16 nats vs the matched control.
  Suspected mechanisms (untested): cold-start eval penalizes models trained
  with persistent context (`estimate_loss` uses fresh state on random crops);
  stale-context noise the paper warns about without ALiBi damping. Memory:
  TBPTT retains the whole window's graph — h4@b4 OOMs (probe evidence), hence
  b2 at h4. Checkpoint-name collision found: all runs write
  `bdh_wikitext2_{best,last}.pt`; per-arm copies saved manually. Candidate
  follow-ups: stateful (stream-carrying) eval, per-arm LR tuning, ALiBi sweep
  (handover §7.4), larger attn_window.
- **2026-08-23 (§7.2 follow-ups: stateful eval, ALiBi, equal-updates):**
  - Added `--stateful-eval` (stream-carrying evaluation, commit `57e8c1f`) and
    `--run-name` checkpoint isolation after a chained-arms overwrite lost the h1
    checkpoint (all arms had shared `bdh_wikitext2_{best,last}.pt`).
  - **Protocol artifact confirmed:** paired evals show the carry-trained model
    *improves* under stateful eval (h4-final cold 1.3500 → stateful **1.1900**)
    while the random-crop model collapses under imposed state (cold 1.1309 →
    stateful 2.8798). The §7.2 "carry regime hurts" verdict was mostly protocol
    mismatch; models specialize to their inference regime.
  - **ALiBi sweep** (h1@b4/20k, stateful eval): slope 0.0 → 1.1406,
    **0.05 → 1.1177**, 0.10 → 1.1212. Damping helps; best BDH number of the
    session, ahead of random-crop ctrl (1.1222) and GPT final (1.1235), behind
    GPT best ckpt (1.0911).
  - **Equal-updates horizon test:** horizon-2 @ 20k updates / 83.9M tokens →
    1.1448 vs horizon-1 @ 41.9M → 1.1406. Extra horizon+tokens at equal updates
    buys nothing on short-context wikitext-2; the earlier horizon ranking was
    update starvation + protocol, not TBPTT harm.
  - Debugging note for future sessions: always pass `dataset=` explicitly when
    loading checkpoints ad hoc — a default-shakespeare loader produced
    plausible-looking but cross-dataset losses (~3.0) that briefly looked like
    checkpoint corruption.
- **2026-08-23 (long-context study, §7.3, §7.5):**
  - **Long-context carry:** on 30k-token sequential test streams the
    alibi005 model improves −0.086 nats from batch 1 to late-stream; window
    sweep 512–4096 is byte-identical because exp(−0.05·d) kills attention
    contributions within a few hundred tokens (window size moot under
    damping). No-damping model gains more from context but from a much worse
    cold start; random-crop control degrades +0.69 under imposed state.
  - **§7.3 interpretability** (`alibi005`, plus new `bdh-linear lina005`,
    val **1.1166** — best of session): real-text xy sparsity **97.4%**;
    neuron graph at β=0.30 → power-law α≈1.94, Newman modularity **0.25**
    (β sweep 0.25/0.30/0.40 in `out/logs` history); monosemantic synapses:
    data-driven France-vs-Germany selection found selective synapses
    (layer1/head3 (60,1): +34.3 FR vs +10.0 DE; layer0/head2 flips toward DE)
    incl. suppressive responses (−51); all top hits share source neuron i=60,
    suggesting topic-hub structure. Fixed `label_partition` bincount crash.
  - **§7.5 merging** (needs `--train-slice a:b`, added this session): parents
    on disjoint wikitext halves → 1.2486 / 1.2548; neuron-dim concatenation
    merge (50.5M params, 65,536 neurons) → **1.1941 without finetuning**,
    beating both parents; generations keep proper wikitext article style.
    Gap to full-data ctrl (1.1222) = cost of splitting the data.
