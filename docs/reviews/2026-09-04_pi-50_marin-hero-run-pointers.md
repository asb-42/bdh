# Marin 535B-A23B hero run — pointers for BDH, and one finding it triggered in our own code

Source: https://mtracker.oa.dev/hero-run-535b (tracker updated 2026-09-03T21:05Z; auto-generated via
mumwelt from GitHub/Discord/WandB). Read 2026-09-04 by pi-50. Run spec (#8435): 535B total / 23B active,
48 layers, expert parallelism inside each NVL72 rack, two-phase Harrier mixture, 18T tokens,
390,251 steps, launched Aug 20, at step ~57.5k (14.7%) projected Nov 28.
Caveat on provenance: this is an LLM-summarized tracker, not the primary threads. Anything load-bearing
below should be re-read at the linked PR/issue before being quoted in a paper.

## 1. The headline scientific lesson: healthy metrics do not mean healthy internals

An outside group (#8818 comment 5520889983, 60B and 180B MoE, 768 experts/layer, 8 active/token) reports
that **routed experts in the first few expert layers stopped learning after a few thousand steps while the
loss, the evaluations AND the expert load balance all stayed normal**. They saw it from weight norms falling
toward zero under AdamW + weight decay. Their own caution, which I think is correct and should be copied
verbatim into our review posture: the two setups share almost no components, so the link is *a possibility,
not a diagnosis*.

BDH mapping: our entire ladder is evaluated through perplexity. We have exactly one internal instrument
(the weight-atlas scans) and it is how we found the amplitude collapse at all. Their failure mode is ours
with the sign flipped — we found uniform multiplicative attenuation invisible to ppl until someone looked
at weights.

## 2. Instrument pointers (cheap, directly transferable)

| what they track | we track | gap for BDH |
|---|---|---|
| MFU, **token drop** (router capacity overflow), **routing entropy** on the run dashboard | ppl only | per-phase routing entropy + expert activation histogram would make router death visible without an atlas scan |
| per-layer attention-gate weight norm (#8818) | nothing | gate/router norm trajectory per phase costs microseconds inside the eval loop |
| prompt completions from three saved checkpoints (#8827) as plain-language sanity check | nothing | one fixed probe prompt per language per phase is ~free and catches nonsense ppl cannot |
| gate-logit scaling knockout on a step-48k ckpt: -20% nearly free, -40% bad, -60% collapses (#8818) | nothing | **directly applicable**: scale BDH router logits ±20/40/60% and measure ppl — tells us whether routing is load-bearing or decorative |
| attention-only knockout, first ten layers vs last ten (2x asymmetry, #8818) | nothing | segment-indexed knockout on our latent: which growth epochs carry the joint model |
| per-token gate census over full vocabulary: only 546/128,256 tokens open any head, mostly reserved/byte fragments (#8818) | route confusion matrix (language-level) | token-level routing census would say whether our router keys on content or on language id |

## 3. Process pointers (their governance is better than ours in three specific ways)

1. **A config change is treated as an experiment needing controlled activation.** #8833 (gate/router weight
   decay) is deliberately held *unmerged* until a planned restart, because "the hero resumes by run id and
   picks up whatever the recipe says, so merging earlier would switch the decay on at the next unplanned
   restart instead of at the permanent checkpoint". Our exact hazard: our ladders pass flags in shell
   scripts, and a resume reads the script as it exists at launch time — which is how compiled-vs-eager and
   lr-regime mixing entered our artifacts (my F-V6 instances).
2. **They pin an extra permanent checkpoint just past the switch-on point** (#8854, step 55,000) purely to
   create a fixed comparison fork, and fork the decayed continuation as its own run id/tree rather than
   mutating the hero in place. Our analogue is already half-present (we keep `_best` + `_last` per phase —
   that is why the ladG chain shows 19 distinct sizes x 2), but we have no rule saying "pin a fork point
   before changing any regime", and we do relaunch in place.
3. **Explicit resume source.** #8868 adds `--initialize-from-checkpoint` so a handoff names the checkpoint
   *by path* rather than taking the newest, and #8684 makes "found checkpoints but loaded none of them"
   fail loudly instead of silently restarting from random weights.

Also worth stealing as reporting hygiene: incident cost stated in machine-hours ("net cost of the trial:
about three hours of the eleven racks"), and the honesty that a full-scale hang is **not reproducible at
smaller scale** ("neither a one-rack smoke on the same checkpoint nor the d768 ladder rung reproduced it")
— which is a warning about our own ladder-as-proxy reasoning.

## 4. Finding this triggered in OUR code (candidate F-V8, needs owner ruling before it goes anywhere)

Their #8684 made me check our resume path. Result:

- `pipeline/train.py:71` writes `"optimizer_state": optimizer.state_dict()` into every checkpoint.
- Repo-wide grep for any consumer (`optimizer.load_state_dict`, restore/load flags, `--resume`) over
  `pipeline/` and `scripts/`: **no matches**. `merge.py:81` sets it to `None`.
- `pipeline/train.py:214` builds `torch.optim.AdamW(...)` fresh on every invocation.

So the moments are **saved and never restored**: every phase boundary continues the weights but restarts
optimization with empty first and second moments. Two consequences:

(a) *Interpretation.* Each phase begins with bias-corrected steps computed from near-zero moments, i.e.
effectively larger early updates. That is a plausible partial mechanism for our interference term — a
warm-start shock at each boundary — and it is testable cheaply: look for ppl discontinuities immediately
after phase boundaries in existing logs, then run one phase with and without moment restoration.
(b) *Storage.* fp32 weight + m + v = 12 B/param, which is precisely why the `bytes/12` param oracle works.
Dropping the unused two-thirds would put a 20-phase arm at ~2.3 GB per checkpoint instead of ~6.95 GB —
but it breaks the oracle constant and any tooling that assumes the key exists, so it is an owner decision,
not a cleanup I should perform.

Note the honest limits of this finding: it may be entirely deliberate (resetting moments when adding grown
capacity is defensible). What is not defensible is that it is silent — there is no log line, no doc, and no
flag. The minimal fix is the same shape as theirs: print what was restored and what was not.

Scope caveat: my bound scope is science soundness of papers + docs/reports, and code reading only where it
changes how a result may be interpreted. This qualifies under the second clause (it bears on how ladder
phase-to-phase deltas are read), but landing it as a repo finding needs an explicit yes.

## 5. Suggested division of labour (Quinn is reading the same page)

Quinn: training-ops mapping — checkpoint pinning policy, resume-source discipline, the gate/router decay
question itself. Me: instrument/telemetry gaps (section 2), interpretation scope (section 4a), and the
provenance caveats. Overlap to avoid: both of us proposing report edits for the same sentence.

---

## 6. Read the primary source (via Firecrawl, installed today) — and it turned our decay leak into a derivation

The Notion write-up is **"Mitigate Silent Expert Death in Ultra-Sparse MoE"** (Jin, Li, Han, Liu, Wen, Zhao,
Yin, Huang; public 2026-07-27; `alltoall.notion.site/save-lower-layer-moe-experts-llal`). Local copy of the
markdown: `/tmp/notion.md` (15.6k words). Key content beyond what the tracker conveyed:

- **Mechanism (§2.2, App. D/E):** at tiny gradient scales the second-moment estimate drops below AdamW's
  ε, so the denominator is effectively just ε, the update loses scale invariance and *shrinks with the
  gradient* → death spiral. They derive ε_Muon=1e-7 ≈ ε_AdamW≈1e-11 for a 1536x512 expert matrix, and
  estimate √v_shared/√v_routed ≈ 30-100x from sparsity (E=768,K=8,β=2.5).
- **Proposed fix LLAL (§3.4, §4.2):** attach an LM head to a *lower* MoE layer's representation during early
  training. Its value is "not larger gradients but a local demand the upper layers cannot take over." Best
  single intervention in their table (+0.032..+0.038 MMLU, Δval −0.029 at 2T tokens), and the decisive effect
  happens in the first ~2% of steps.
- **Negative results worth knowing:** Quantile-Balance fixes load balance fast but does **not** rescue expert
  norms; soft bias updates and momentum bias updates gave no validation-loss gain (momentum made balance
  slightly worse); Attention-Residuals scored *below* baseline on MMLU (0.5112 vs 0.5162) despite helping
  expert development. Router lr ×0.1 helped (+0.024 MMLU). Muon > AdamW throughout.
- **App. F audits of open models:** MiMo-v2.5-pro's first ~15 layers are effectively muted (near-rank-1 gate
  projections, median inter-expert cosine 0.497 where random init ≈ 0, masking nearly free); Qwen3.5-397B and
  Qwen3.8-Max show early collapsed layers; Kimi-K3's learned bias overrides >70% of the router's own top-1 in
  some layers. Their honest framing: *"How to measure whether a MoE model is well trained is an open problem.
  Evaluation metrics are not sufficient."*
- **Refs to pull into our prior-art pass (M4):** [27] *The Myth of Expert Specialization in MoEs: Why Routing
  Reflects Geometry, Not Necessarily Domain Expertise* (arXiv 2604.09780); [28] SD-MoE spectral decomposition
  (2602.12556); [12] ModuleFormer (2306.04646); [31] *Small-scale proxies for large-scale Transformer training
  instabilities* (ICLR2024) — that last one is the citation our ladder-as-proxy caveat needs.

### What it made me measure in our own checkpoints (read-only, on .200, via mmap)

`bdh_europarl_ladG2-lt_last.pt` (arm G2, final phase): 6 model tensors, **579.1M params** counted from shapes
(not from bytes), `mult=736`, 3 optimizer slots, `step=10000`, and w+m+v = **6.95 GB = the file size exactly**.
Then the census that matters — Adam second moment per (head, 2048-block) cell of `decoder`:

```
cells with strictly ZERO exp_avg_sq : 176 / 184
cells with nonzero sqrt(v)          :   8 / 184   -> all of them local block 22 (the newest capacity)
sqrt(v) of alive cells              : 5.11e-05   (= ~5100x torch's default eps 1e-8)
```

So in the final phase **gradients flowed only into the block added by that phase**; every older block received
an exactly-zero gradient for all 10,000 steps. That matches the code: `pipeline/train.py:96-101` documents that
"Mechanism C write-gating / Mechanism B grow share the gradient-mask path", masks are applied as a multiplier
before the step (`:155-160`), while `:214` builds a **single** AdamW group with `weight_decay=cfg.weight_decay`
(0.1) for all parameters and **no `eps` field exists anywhere** in config or the call site (so torch's 1e-8).

Three consequences, in order of confidence:

1. **Our decay constant stops being a fit and becomes a derivation.** A coordinate whose gradient is the zero
   *tensor* (not `None`) still receives AdamW's decoupled decay, and nothing else touches it: so
   `p_exit = p_entry · Π_t(1 - lr_t·wd)` **exactly**, independent of data, loss, or routing. That is precisely
   why Quinn's per-segment c-fits agree to ~1e-5 across intervals of 1, 4, 18 and 19 phases — it is not
   empirical luck, it is a closed form. It also predicts c is identical for every language in a phase, which is
   checkable in existing artifacts.
2. **We already own the instrument we were looking for.** Every checkpoint silently records which blocks
   actually learned in that phase: `exp_avg_sq == 0` means structurally excluded, tiny-but-nonzero means routed
   to weakly. No norms, no atlas scan, no training run needed — it is a read of bytes we are already paying to
   store. This is a strictly sharper "did anything happen here" signal than perplexity, and it is the exact
   analogue of their "loss and evals stayed normal while experts died".
3. **Their ε mechanism is a live risk for our repair work, not for our training.** Alive cells sit ~5100× above
   ε, so nothing is ε-clamped today. But a repaired/spliced checkpoint puts weights back into blocks whose m,v
   are *exactly zero*; if such a block were ever trained again, its first steps are ε-dominated and therefore
   throttled — a plausible optimizer-side contributor to the residual that repair cannot remove, and to why
   co-adaptation looks asymmetric by segment depth. Untested; needs one phase with unfreeze-after-splice.

Caveats I will not paper over: this is **one checkpoint** (arm G2, lt, phase 20). Before generalizing it needs
repetition across arms and phases (cheap, read-only, I can do it here or task OC), and the mask semantics should
be confirmed by whoever owns `gate_param_masks` — I am reporting what the tensors and the code say, not claiming
intent.
