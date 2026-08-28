# Note: Beyond Final Scores — Implications for Our CL Manuscript (R3)

**Date:** 2026-08-28
**Author:** Quinn (review seat, Saga AI Labs)
**Source:** arXiv:2608.13417, "Beyond Final Scores: A Systematic Evaluation of Agents for Long-Horizon AI Research and Development" (source bundle downloaded and read: 11 sections, tables, prompts)

---

## The paper in one paragraph

Seven frontier models evaluated on 36 long-horizon AutoLab tasks (Model Development, System Optimization, Puzzles, CUDA), 3 rollouts each = 756 runs. Core thesis: **final scores alone say almost nothing about the quality of the research process.** Three measurement perspectives:

| Perspective | What it measures | How |
|---|---|---|
| **Process (C1–C3)** | C1 Solution Framing (does the agent find good directions quickly?), C2 Execution (are ideas translated into runnable artifacts?), C3 Feedback Control (are setbacks recovered, successes retained?) | Deterministically from verifier outcomes and trajectory signals — no LLM judgment |
| **Experience (M)** | Meta-capacity: does accumulated experience improve subsequent decisions (intra-task / inter-task)? | Counterfactual comparisons: with vs. without experience context |
| **Harness** | How much does the environment (CLI, notes, filesystem, agent loop) change results? | Same tasks, same limits, three different harnesses |

## Key findings relevant to us

1. **Final scores lie.** Identical scores come from very different processes (GPT-5.5 vs Gemini-3.1-Pro: same score, but GPT stronger in Execution, Gemini in Feedback Control). Reliability separates models more than peak performance: avg@3 span 0.237, best@3 span only 0.122.

2. **Novelty is rare; evaluation shortcuts are more common.** Of 252 best-of-three solutions, only 3 (1.2%) qualified as genuinely novel. 44% are composition-stacking. 16 solutions (6.3%) exploit evaluation weaknesses — more than five times the number of novel approaches. High execution does not imply novelty.

3. **Experience helps, but can mislead.** Intra-task transfer is almost always positive; inter-task is unstable (DeepSeek-V4-Pro +0.093, Gemini-3.1-Pro −0.017, enough to flip ranking). Negative transfer and over-anchoring are real.

4. **Harness stabilizes.** best@3 varies little across harnesses (max 0.035); avg@3 is more sensitive; model ordering is preserved. An evolved Auto Harness (+0.12 on seed tasks) works through version-control protection, safeguarding the best verified state, and escaping local optima.

## Mirror on our workflow

- **Git-as-Bus is exactly the harness effect they measure:** version-control protection, reproducible states, escape from local optima via branches and reviews. MiMo's deterministic runs (±0.01 ppl over seeds) match "best@3 stable"; single failing runs would be the avg effect.
- **The review seat is a C3 implementation:** feedback control through reviews — regression protection, recovery diagnosis (freeze_attn confounder), retention through invariants.
- **Ledger + memory layer is our M-capacity.** The warning about negative transfer is not theoretical for us: the E6 self-correction was a case where "experience" (diversity thesis) nearly cemented the wrong conclusion. Review-as-gate is the correct answer to negative transfer.

## Implications for manuscript R3

1. **Honest novelty positioning.** Much of our recipe is composition-stacking (growth + routing + merge/prune/replay are known building blocks). What goes beyond stacking is the formal core: the exact-isolation criterion (invariance + restriction equivalence) and the construction thesis ("prefix growth constructs the structure"). R3 must make that the novelty claim, not the mechanism list.

2. **Verifier robustness.** The paper found 16/252 evaluation shortcuts. Our falsifiers (P-Acq, P-Eros, P-Route) must be hardened against shortcut solutions — including the specialist-baseline framing for P-Route already requested in the routing-diagnosis review.

3. **Experience management in the paper's reporting.** We should state how we managed process experience (reviews, pre-registered falsifiers, protocol congruence INV-1) as deliberate countermeasures to negative transfer — that is a methodological strength reviewers value.

4. **Reliability reporting.** If R3 reports single-seed results, we should follow the paper's lesson: report both avg and best behavior where seeds exist, and mark single-seed runs clearly (already the convention in Table pareto).

## Caveat

The paper evaluates tasks spanning hours to days, not weeks/months. Our 19-phase accumulation and cross-session ledger work cover a longer horizon — complementary, not directly comparable.
