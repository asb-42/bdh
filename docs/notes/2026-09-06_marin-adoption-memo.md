# Marin Adoption Memo: What BDH Takes from an Open Foundation-Model Lab

- Date: 2026-09-06
- Author: A0-Quinn (Agent Zero seat)
- Sources: mtracker.oa.dev/hero-run-535b (full status log), GitHub marin-community/marin issue #8435 + 24 comments, W&B scaling-ladder report, mumwelt README, talk transcript "Marin: an Open Laboratory for Foundation Models in JAX" (OpenXLA DevLab Fall 2025, David Hall), research-mesh particle `marin-hero-run.md`
- Status: Phase A item A3 of the battle plan; adoption candidates listed with owner/machine; nothing lands without room review

## 1. The core thesis (why this is not just nice-to-have)

David Hall's formulation from the talk: **"experiments are the changelists of empirical machine learning research."** In open-source software, the unit of progress is a change list — reviewed, versioned, revertible. In empirical ML, the unit is an experiment. Marin's move is to treat experiments with the same discipline software treats code: pre-registered as GitHub issues (their form of pre-registration), implemented as PRs, executed as content-addressed DAGs, everything live on W&B, failures public.

**BDH already does the hard part.** Our pre-registration practice (H-decay-1/2/3, P1–P5, S1–S3, M1–M3, MF1/MF2, S1e — every number in this thread was predicted before it existed) IS their methodology, independently arrived at. What we lack is the presentation/infrastructure layer that makes it legible to outsiders. That is what this memo adopts.

## 2. The methodological mirror (convergent validation)

Three cases where Marin independently invented our exact protocol:

| Marin practice | BDH practice (this thread) |
|---|---|
| Gate/router weight-decay PR #8833 held **unmerged** until a planned restart; forked from a **pinned permanent checkpoint** (step 54k) as its **own run ID** rather than mutating in place | F-decay fix (11813b1) bit-verified before RA2b launched as a **separate chain** (`ladRA2b-*`), nothing overwritten, H-decay-1/2/3 pre-registered |
| Ablation before intervention: gate-logit scaling ladder (-20% free / -40% bad / -60% collapse) on a saved checkpoint BEFORE deciding decay value; ladder-rung ablations all within noise | 2×2 cells + v6 forced-prefix probe + M/M-fixed regime cells BEFORE the report claimed anything; every prediction PASS/FAIL'd at readout (R4) |
| Optimizer-intervention debate with recorded objections (scaling-ladder predictability risk; norm-compensation risk) | Our §4 three-way decomposition debate, with pi-50's #88/#89 sharpenings adopted and my #113 `--seed` error caught and retracted in public |

**The lesson is symmetric:** their 535B run and our 579M run hit the same failure class (silent optimizer-side weight erosion invisible to loss) and converged on the same process discipline. For the manuscript's discussion section this is an existence proof that our findings generalize — and that the fix discipline does too.

## 3. Adoption candidates (ranked by value/cost)

### 3.1 Per-phase norm-forecast drift monitor (ADOPT — design already sketched; the Percy Liang demand)

Percy Liang, in the Marin tracker: "A training statistic that does not hold across model scales deserves its own investigation… the tracker should forecast more statistics than the loss and warn when one drifts off forecast." Our leak is the canonical case: segments shrinking 0.58×/phase while loss looked fine — caught at day 3 by tensor arithmetic, would have been caught at phase 1 by a norm forecast.

**Design (from the Marin parameter-norm table, which IS this monitor in production):** every phase exit, compute per-segment RMS norm for encoder/encoder_v/decoder; expected value = parent's segment norm (fixed regime: bit-exact; leaky regime: ×c schedule factor); alarm on deviation > 1e-6 relative. ~30 lines, CPU-only, read-only, runs beside the ladder. Catches F-decay-class leaks at the NEXT phase exit, not three days later. This is battle-plan item C3.

### 3.2 Health checklist mapped to BDH (ADOPT — checklist as docs/tasks artifact)

Marin's 12-point health checklist (issue #8435), translated:

| Marin check | BDH equivalent |
|---|---|
| Grad norm oscillating steadily (rate-of-change matters, not magnitude) | `running_loss` trajectory per phase; grad-norm currently unlogged — **add grad-norm to train.py log line** (one print, zero risk) |
| Constrained params respecting constraints | `gate_param_masks` actually applied — our P5 instrument now checks this in-chain (`scripts/p5_inchain_check.py`) |
| Token dropping on trend | No MoE in BDH — N/A (route-accuracy is our analogue: eval_router at milestones) |
| Loss decreasing steadily | Already logged |
| Evals decreasing steadily | Already logged (val_loss every 50 steps) |
| Router bias behaving steadily | N/A (no learned router bias; likelihood router is inference-side) — but **route-accuracy per milestone** is the analogue and exists |
| Router entropy near max | N/A as such; **serving-route distribution spread** (routdiag confusion entropy) could serve — candidate for the readout instrumentation |
| Embed norm growth reasonable | embed is frozen in growth phases (P5 covers) — check at base phase |
| MFU constant | ms/step already logged; add a one-line MFU estimate |
| LR schedule correct | Already logged (lr in step line) |
| W&B config accurate | Our manifest is the analogue (F-V6 fields; pip-freeze line pending — #123 lesson) |
| Hitting projected eval targets per 5% | **We have no projections mid-run** — this is the one real gap; a per-phase acquisition forecast from the ladder history would close it |

**Net: 9 of 12 already exist implicitly; 2 are one-line additions; 1 (mid-run projection) is a real but optional gap.** I will draft the checklist as `docs/tasks/bdh-health-checklist.md` on the next container pass.

### 3.3 Experiment-as-DAG with content hashing (ADOPT PARTIALLY — the principle, not the infra)

Marin's executor hashes every step by its critical inputs; change one dataset and the DAG propagates a new identity, making fair comparisons mechanical. We cannot adopt the infra (no Ray/GCS budget), but the principle ports directly to our manifest discipline: **the manifest row is a poor man's content hash** (git SHA + argv + literal init filename + resolved cfg + checkpoint sha256). The #123 lesson (diff the whole resolved cfg across eras — takes seconds, replaces an afternoon of drift speculation) is exactly this principle in miniature. Formalize: `resolved_cfg.json` dumped per run into the manifest directory; cross-era comparisons diff two JSONs.

### 3.4 Mid-run recovery discipline (ADOPT AS DOCUMENTED PRECEDENT)

The 32B recovery story is the strongest single narrative in the transcript: loss spike → new plateau ("irreparably bad"), then — skip bad steps (no), clip tighter (no), Muon (no), and finally **modify the architecture mid-run (add QK-norm), reload from a known-good checkpoint, eat a 10M-token recovery spike, and finish ahead of the plain run**. Their recorded reasoning for NOT wanting QK-norm (70B and 8B didn't need it; possible long-context cost) is exactly the kind of pre-decision documentation that makes the eventual intervention legible.

**BDH parallel:** our RA2b IS a mid-run recovery — the ladder was the patient, the leak was the pathology, the fix was the surgery, and the fresh-chain relaunch was the reload-from-known-good. Their story validates the pattern: intervene from a pinned state, expect a transient, document the objection first. For the manuscript: cite Marin's 32B recovery as precedent for optimizer-intervention-under-uncertainty in open development.

### 3.5 Presentation layer (ADOPT — the operator's original ask)

What makes Marin legible is not the compute, it is: (a) the **tracker page** (auto-generated status log from primary sources via mumwelt — LLM-digested, clearly flagged as secondary, every claim links its PR); (b) the **data browser** (every experiment ever run, DAG-viewable); (c) **W&B report pages** with pre-registered predictions vs. measured curves; (d) a public **issue as living spec** (8435: 24 comments, hyperparameter tables, health checklist, risk cards).

**Our equivalent, buildable with our stack (HAK + Git + the atlas):**
- `bdh-tracker` page: cron digests our Git log + HAK feed + run logs into a status page; the HAK bus already carries every event with provenance; mumwelt's model (LLM digest of structured sources, cited, flagged) is the pattern.
- Ladder dashboard: per-phase acquisition + norm-forecast plot (3.1) + route-accuracy heatmap; the atlas renders the weights, this renders the trajectory.
- Spec-as-issue: the battle plan already functions as this; its §6 open-decisions table is the GitHub-issue pattern.

This is a Phase C candidate (after the paper), not a blocker — but the per-phase norm forecast (3.1) should land with the readout instrumentation so the RA2b readout page has the dashboard from day one.

## 4. What we do NOT adopt (and why)

- **Isoflop suites / scaling-law projection discipline:** right tool for their question (dataset selection at scale), wrong for ours (we are not choosing between datasets at N flops; our ladder IS the unit of analysis). Their per-5%-window power-law fits are elegant; our per-phase acquisition curve is the analogous artifact and already exists. Revisit only if we add a data-mixture axis.
- **mumwelt itself (the tool):** brilliant, but it solves their corpus problem (Discord + GitHub + W&B across a public community). Our corpus is two Git repos and one bus — the HAK IS our mumwelt. Adopt the pattern (LLM digest, cited, flagged secondary), not the software.
- **Preemptability infrastructure:** TRC-specific (free preemptible TPUs); our two boxes are ours. The surviving lesson — checkpoint often, idempotent jobs, recovery trivial — is already our practice (the sv-backfill ran through unattended).

## 5. Recommendation summary

| # | Candidate | Verdict | When | Owner |
|---|---|---|---|---|
| 3.1 | Per-phase norm-forecast monitor | ADOPT | with RA2b readout instrumentation | Quinn (design) |
| 3.2 | Health checklist → docs/tasks | ADOPT | next container pass | Quinn |
| 3.3 | Resolved-cfg dumps per run | ADOPT | now (one line in launcher) | Quinn |
| 3.4 | Mid-run recovery as documented precedent | CITE in manuscript discussion | Phase B | Quinn (with pi-50 A11) |
| 3.5 | bdh-tracker / dashboard | DEFER to Phase C | after paper | room |
| 4.x | Isoflop suites, mumwelt-the-tool, preemptability infra | DON'T ADOPT | — | — |
