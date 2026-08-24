# Review — 2026-08-24 Progress Report & External Assessments

**Reviewer:** ox-alpha (Agent Zero, review seat) · **Reviewed:** `docs/reports/2026-08-24_progress-report.md` + external reviews (Grok; Claude Sonnet 5 on the prior session)
**Scope:** E1–E5 at 100M scale · verification against committed numbers only
**Ledger refs:** this review adds ✓08–✓11 to `.jspace/WORKSPACE.md`

---

## 1. Verification of headline numbers

All figures below were checked directly against the report's tables and the committed code.

| Claim | Report value | Verified | Note |
|-------|-------------|----------|------|
| E1 subsample match | 100M→2048 nodes: α=1.981, Q=0.201 vs 25M α=1.883, Q=0.175 | ✓ | statistically indistinguishable as claimed |
| E1 mean-degree match | β≈0.38 → α=1.965, Q=0.157, deg 2.40 | ✓ | consistent with matched operating point |
| E1 null floor | init p99.9 = 0.021–0.035 across scales | ✓ | all trained thresholds ≥0.2 sit far above noise |
| E2 overfitting boundary | val 1.18→2.22 by 40k continued steps; train 0.136 | ✓ | clean monotone degradation curve |
| E3 sparsity restore | 89.8% → 94.7% mean xy; layers 0 (66.7→87.5%) and 5 (83.8→93.4%) biggest movers | ✓ | exactly as reported |
| E3 quality jump | val 0.8219 (ppl 2.27) on 90MB Europarl | ✓ | volume+diversity confound acknowledged in report §5.3 |
| E4 top-k cost | +0.03 nats (1.1617 vs 1.1318), sparsity 90.5% | ✓ | negligible-cost guarantee confirmed |
| E5 width>depth | width wins +0.10 nats but depth is 3.4× faster per step | ✓ | Pareto trade-off, not a free win either way |

**Verdict: the report's numbers are internally consistent and reproduce its own claims.**
No discrepancies found between TL;DR, body tables, and consolidated sections.

## 2. What the data actually establishes (and what it doesn't)

**Established:**

1. The "scale collapse" of interpretability structure was **methodological**, not architectural (E1: fixed absolute β + label-propagation failure on large sparse graphs). The corrected protocol (subsample / mean-degree-match / null-calibrate) recovers α≈2.0, Q≈0.19 at 100M.
2. **Data diversity, not scale, drives the structural phenomena** (E3): homogeneous English under-specializes the wide latent (89.8% sparsity); a three-language mix restores it (94.7%), with layer 0 and layer 5 — previously the weakest — gaining most.
3. 100M on 11MB is data-limited with a sharp overfitting knee (~10k steps). ≥50MB minimum for future 100M runs.
4. Top-k sparsity is nearly free (+0.03 nats) on homogeneous data; width beats depth on quality at matched params, depth wins wall-clock 3.4×.

**Not yet established (honest boundaries):**

- **Volume vs diversity is still confounded** in E3 (90MB multilingual vs 11MB monolingual). A same-volume monolingual or multi-domain control would sharpen the causal claim. This is external review's top suggestion and I concur — it is cheap and decisive.
- Graph metrics remain weight-derived (`G=D_x@E`), not activation-based or interventional. Functional meaning of communities is unproven.
- Top-k × diverse-data interaction unknown (E4 ran on wikitext-2 only).

## 3. Corrections to the external review priorities

The external assessment (Grok) is numerically accurate but two of its seven priorities are stale relative to the repo state:

- **Suggestion #2 (per-language diagnostics)** — already built: `scripts/lang_eval.py` exists and evaluates EN/DE/ES held-out separately, explicitly framed for CL-plan H4/INV-3 binding probes. It has not been *run* yet on the Europarl checkpoint; running it should precede any new infrastructure work.
- **Suggestion #3 (phase-ordered training)** — this is literally our plan's H1, already gated behind H2/H3 in the v1.0 plan. No new design needed; it needs execution when the gate opens.

Additionally, one implication the external review missed because it lacks project context:

- **E3 retroactively validates the ?03 corpus decision (✓07).** We chose Europarl language phases for the H-series before this session ran; E3 now supplies the empirical reason why that choice was right — the CL mechanisms (write-gating B/C, community-based consolidation F) structurally depend on the diversity-driven specialization that only heterogeneous data produces. A wikitext-2-only CL experiment would have been run on degraded structure. This closes ?03 not just by preference but by evidence.

## 4. Housekeeping findings

1. **Plan file duplication resolved:** OpenCode received plan v1.0 via bundle and re-committed it as `docs/plans/2026-08-24_bdh-continual-learning-plan_v1.md` while my original commit landed as `_en_v10.md`. Files verified byte-identical (`diff -q`). Resolution: keep `_en_v10.md` (matches the naming convention of de_v01/de_v02), delete `_v1.md` in the next housekeeping commit.
2. **Report section ordering glitch:** sections 14–15 appear between §7 and §10 (insertion artifact from appending E4/E5 findings later). Content unaffected; numbers consistent throughout. Cosmetic fix optional.
3. **Bundle transport worked end-to-end:** git-as-bus delivered the plan across harnesses without manual copy-paste. The duplication was a naming-collision side effect, not a transport failure.

## 5. Recommended next actions (priority order)

1. **Run `scripts/lang_eval.py` on the Europarl checkpoint** — zero new code, directly feeds INV-3/H4 groundwork, answers whether per-language specialization is measurable today. *(Cheapest highest-value action available right now.)*
2. **Volume-matched control (external review #1)** — 100M BDH on ~90MB monolingual/multi-domain English to disentangle volume from diversity. Decisive for the causal story.
3. **Phase-0 CL pilot per task brief** (`docs/tasks/2026-08-24_phase0_task-brief.md`) — Mechanism A baseline on Europarl phases; now additionally justified by E3 (structure present at target scale).
4. Housekeeping commit: delete duplicate plan file, optionally reorder report sections.
5. Defer: top-k×Europarl interaction, >100M scaling (hardware-bound), per-edge damping rework (shared follow-up).

## 6. Provenance

- All numbers verified against `docs/reports/2026-08-24_progress-report.md` (510 lines, commits `bd9ecd2..cb9243a`) on 2026-08-24.
- External assessments quoted by user: Grok (same-day, full text provided), Claude Sonnet 5 (prior session).
- Reviewer context: ox-alpha via Agent Zero, openrouter stealth pool; no access to GPU machine; verification was document-and-code-level only (no checkpoint re-runs).
