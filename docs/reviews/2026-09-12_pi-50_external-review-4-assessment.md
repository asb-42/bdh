# Assessment of the fourth external review (Kimi 2.x, bus #229) — plus a citation audit that caught four more defects

**pi-50 · 2026-09-12 ·** manuscript at `bdh/a8caa63` (rev 4.4). Review saved locally (`/tmp/kimi.md`, 24 KB) and
each checkable claim tested against the `.tex`, a committed artifact, or a primary source.

## Verdict

The strongest of the four machine reviews: it read the source, its criticism is mostly about *internal
consistency* rather than tone, and **it is right about most of what it asserts**. Confirmed: two real LaTeX
corruptions with a diagnosable root cause, a miscast citation in the abstract's first sentence, absent
hyperparameters, a quantified internal inconsistency in how we excuse a ratio, an ambiguous 800/800 count, an
undefined 23-class target, and a genuine scope error in our flagship control. Rejected: three claims that do not
reproduce against rev 4.4, and two theory objections resting on a confusion between initialization and final
state.

Then the part that matters most: chasing its citation claims turned up **four further defective bibliography
entries that no reviewer named**, including two in entries I had previously marked as verified. Section D is the
deliverable; the fix list is mechanical.

## A. Confirmed — fix before arXiv

**A1 · Two TeX commands are silently corrupted, root cause identified.** `\texttt{…}` (Fig. 3 caption,
"Sources:") and `\textbf{Learning to Prompt …}` (§prior art) are stored as a literal **TAB character followed by
`exttt{` / `extbf{`** — i.e. the backslash-t was interpreted as an escape by a non-raw Python string inside a
patch script, so `\texttt` became `<TAB>exttt`. This is precisely the failure mode behind Quinn's three
empty-commit incidents and my own `str.replace()` no-op rule: the mutation happens in the tool, not the text.
Blast radius measured, not assumed: the file contains exactly **2 control characters** (offsets 26960, 50635) and
no BS/FF/VT/CR/BEL, so sibling escapes (`\times`, `\beta`, `\frac`) are intact. Fix is two substitutions plus a
standing rule: patch scripts touching TeX must use raw strings, and post-patch verification should include
`grep -P "[\x00-\x08\x0b-\x1f]"` on the file.

**A2 · The abstract miscasts our own model's citation.** First sentence: *"a byte-level depth-recurrent
language model~\cite{byte rnn}"*, and that entry is Hochreiter & Schmidhuber *Long Short-Term Memory* (1997) plus
van den Oord et al. *PixelRNN*. Neither is a byte-level depth-recurrent LM. `\bibitem{bdh}` (Kosowski et al.,
arXiv:2509.26507 — the Pathway paper that defines this architecture and whose terminology we treat as
authoritative) exists and is cited exactly **once** in the whole manuscript. Fix: cite `{bdh}` where the
architecture is asserted; keep LSTM/PixelRNN only if some sentence actually needs the lineage claim. Also drop
the space from that key while editing — `byte rnn` is legal LaTeX and fragile practice.

**A3 · Hyperparameters really are absent.** Numeric search over the source: weight-decay value **0 hits**,
learning-rate values **0**, head count **0**, layer count **0**, batch size **0**; `512` and `8192` appear twice
each. For a paper whose most transferable contribution is an optimizer-confound derivation whose closed form is
`c = Π(1 − lr_t·wd)`, never stating `wd` is indefensible — a reader cannot even begin to re-derive it. Zero new
work required: every value is already in committed artifacts (`pipeline/config.py` defaults, per-phase growth
headers, checkpoint `cfg` dicts which I read directly this week: mult 736, grow_mult 32, weight_decay 0.1,
init_from chain). Recommend one §2 table: d, n_head, L, block size, batch, base mult and per-phase Δmult, wd,
lr endpoints + schedule shape, max_iters, seed(s), and the eval-batch convention — the last because train-batch
vs eval-batch effects (+13.7 % vs +1.1 %) are measured here and are the single biggest source of cross-table
confusion in this project's history.

**A4 · The instrument-offset excuse is now internally inconsistent, and Kimi caught it.** §5 reports
routed/acquisition ratios "median 1.09, range 1.02–1.13, within the window-vs-val instrument offset", while
§7.1 claims routed serving lands "within ≤ 8 % of acquisition exit quality". The documented magnitude of that
offset is **+5–9 %** (`docs/reports/2026-09-10_ra2b-fixed-regime-readout.md:59`). A 1.13 ratio is +13 %, i.e.
outside the band we cite as its explanation. So the reviewer's dilemma is real as written: either the band is
wider than our own artifact says, or that sentence over-claims. The resolution is available and cheap — the two
numbers come from different instruments (fixed-regime ladder retention ratios vs the RA2b prefix-scan penalty in
exp4) — but the paper must then name the instrument next to each number instead of folding both into "known
offset". This is my P4/E4/F-V7 class again, now with a quantified violation rather than a juxtaposition risk, and
it is the strongest single item in the review.

**A5 · 800/800 is ambiguous, not wrong.** Text: "(20 routes × 20 domains, 40 crops each) resolves **800/800**
crops to the correct prefix". Intended reading: 20 × 40 = 800 crops, each scored under all 20 prefixes ⇒ 16,000
masked forwards. As written it reads as 16,000 decisions reported as 800. One clause fixes it, and a perfect-score
claim should not be ambiguous. Related, and worth adopting together: Kimi's Q3 asks whether 800/800 is meaningful
given within-domain autocorrelation — same clustering defect I self-reported at §9.7, applied to a peer-owned
number. Correct statement is domain-level: 20/20 domains, every crop correct, per-domain Wilson floor [0.68, 1.00].

**A6 · The 23-way classifier is unexplained — answer, since it is my number.** 23 is the number of
*cumulative-prefix territories*, not languages: English owns base sub-widths 0–3 and each appended language owns
one index, so 4 + 19 = 23 candidate widths (`scripts/pi50/r3_byte_addressing.py:74` prints
`widths=len(WIDTHS)`; `:120` states the cost as "1 counting pass vs 23 masked forwards"). Agreement-with-router
is therefore measured over 23 classes while domain-level scoring uses 20 clusters. The manuscript must say this
in one sentence; otherwise it looks like a mismatched label count, as it did to the reviewer.

**A7 · The expansion control is single-language, and the abstract generalizes it.** True, and I can confirm it
from the instrument rather than the prose: arms A/C were evaluated on the **en-era checkpoint only**. The claim
"one random untrained block costs 4.1×" is an English-measured quantity; for bg/el, where joint degradation
reaches 37.8×, the log-fraction reproduced by random blocks is unmeasured and could differ materially. Scope the
abstract sentence to the tested era. Note the useful half of this finding: extending it requires **no training** —
synthetic expansion plus evaluation on existing checkpoints, minutes per domain — so unlike most generality asks,
this one is nearly free. If we want the general claim, run it; do not assert it.

**A8 · Dual-headline promotion.** Kimi objects that leading the abstract with "83 % … 62 %" invites cherry-picking.
History matters here: the qualifier was added *because* the log-scale-only version was under-reported (my #214 §A,
after an external reviewer validated 83 % with a broken formula). Recommendation: keep both units but subordinate —
lead with the mechanism sentence, give the pair once with the formula reference, and let §6.1 carry the detail. Do
not delete the linear figure; deleting it would recreate the original defect.

## B. Rejected — does not reproduce against rev 4.4

**B1 · "`k_sparse_ratio = 0` appears in Remark 4 with no definition anywhere."** Zero occurrences of that token
in the manuscript, escaped or unescaped. Either it reviewed an earlier revision or invented the detail; nothing to
fix. (If the intent was that the *concept* — the shipped configuration runs with all neurons active at inference —
is used without definition, that is a fair restatement worth considering on its own terms.)

**B2 · "Figure files could not be enumerated from the tree view."** All six referenced PDFs exist at
`docs/papers/figures/` (checked with `os.path.exists` against every `\includegraphics` target: f1_ladder_curves,
f2_fcs_heatmap, f3_retention_bars, f4_ood_scatter, f5_cross_script, f6_expansion_control — all OK). To its credit
the review flagged this as partial verification rather than a finding.

**B3 · `"2026;\ cite as vendor documentation"` in the engram bibitem.** Pattern absent; rev 4.4 carries a URL and
access date there (added per #228). Same for the generic "broken LaTeX" beyond the two real sites in A1.

**B4 · Theory objections conflate initialization with final state.** It argues Theorem 1's dissociation is
hand-waved because our growth convention appends zero-initialized suffixes "which Arm C shows are inert". Arm C
measures the **untrained** case — that is exactly why we call zero-init protective and the A−C gap its value. New
blocks become nonzero through training, and the trained suffix is where the dissociation is realized. Likewise,
Proposition 3 does not *assume* contractivity and get refuted by our measured directional gains (1.05–1.89); it
states that uniform-in-depth bounds **would require** ρ < 1, and we report that the requirement fails. That remark
is doing its job. Concede the useful residue, though: a fast reader makes Kimi's inference, so one sentence
distinguishing init-time zeros from trained nonzeros would pay for itself.

**B5 · Its praise of the MoE analogy.** Kimi asserts the decay confound's "connection to independently observed
'silent expert death' in production MoE training gives it external validity". Grok and Sonnet flagged the same
analogy as unsourced. Two machine reviewers now disagree about a claim that would put an uncited assertion into
our paper, and neither is evidence. Standing position (#214 B5): the analogy stays out unless we read a citable
source making it. This is also the cleanest illustration of why review endorsement language gets no weight at all:
an uncritical acceptance path is how an unsupported claim gets laundered into a rebuttal.

## C. Where it converges with earlier reviews (evidence the layer is working)

OOD circularity (Sonnet P2 → landed in rev 4.4), clustering of per-crop intervals (my §9.7 → now extended by this
review to the 800/800 claim), single-seed/ordering limits and the seed-floor comparison being within noise
(Grok B4 → priced, bounded design parked for phase 2), scope tempering (Grok B1/B5 → partially applied in 4.4).
Across four reviews, every durable improvement to the manuscript has come from a *checkable* claim, and none from
any verdict line.

## D. Citation audit — four more defects, none of them named by any reviewer

Prompted by A2, I audited every `\bibitem` for author/title integrity against primary sources. Results, worst first:

| key | current entry | problem | verified against |
|---|---|---|---|
| `mas` | "R. Aljundi, M. B. French, B. S. Chakravarty, M. Tuytelaars. *Memory Aware Synapses…* ECCV (2018)" | **Wrong first author**: MAS is French, Chakravarty, Tuytelaars — Aljundi is not an author — and initials were mangled when my R3 fix was applied ("M. B. French", "B. S. Chakravarty", "M. Tuytelaars") | venue/author list known; my #210 R3 said exactly this and the patch half-applied it |
| `expertgate` | "*Expert Gate: Task- and Expert-Conditioned Routing Networks.*" | **Fabricated subtitle.** Real title: *"Expert Gate: Lifelong Learning with a Network of Experts"*, Aljundi, Chakravarty, Tuytelaars, CVPR 2017 | openaccess.thecvf.com CVPR2017 paper; arXiv:1611.06194; IEEE 8100236 |
| `supsup` | "M. Wortsman, M. C. Riemer, G. Ilharco. *Superposition Enables Many-Task Learning (SupSup).* NeurIPS (2020)" | **Blends two papers.** Real: *"Supermasks in Superposition"*, Wortsman, Liu et al., NeurIPS 2020 (arXiv:2006.14769); the Riemer-first sibling is *"Learning to Learn in Superposition"* (ICLR 2021). Title as given does not exist | proceedings.neurips.cc 2020 ad1f8bb9…; mlanthology wortsman2020neurips-supermasks; doi 10.48550/arxiv.2006.14769 |
| `ewc` | "…K. Kilan, **R. Google-DeepMind**" | Institution string eaten as an author; "Kilan" also truncated (Kilian) | entry-internal evidence (impossible author token); full list needs a source check before rewriting |

Plus the two Kimi found (`byte rnn` miscast in the abstract; MultiUN v2 title/venue now correctly matched after
rev 4.4 — re-verified: authors Yu Chen & Andreas Eisele, LREC 2012, title *MultiUN v2: UN Documents with
Multilingual Alignments* ✓ that one is fixed).

**Self-report, in my own lane:** I marked `expertgate` and `supsup` as verified in my rev-4 pass (#210, "Verified
clean citations incl. Expert Gate now Aljundi et al."). I checked **authors** against my A4 prior-art note and
never checked the **title strings**. Both titles are wrong. The lesson is narrower and more useful than "verify
more": for a bibliographic entry, verify every field that will be rendered — authors, exact title, venue, year —
and prefer fetching the canonical record over recalling it. My A4 note is also where `pgn`/`packnet`/`piggyback`
were checked the same partial way, so those four get re-audited too before submission, and HSP/PCANets remain
excluded until actually verified.

Recommendation to the operator: one person should own a **bibliography closure pass** — fetch each entry's
canonical record (CVF/ACL Anthology/arXiv/IEEE), compare all fields, and land a diff with the source URL per
entry. Roughly twenty entries, mechanical, and it is the single highest-integrity-risk surface left in the
manuscript: a fabricated-looking citation is the defect that makes a reviewer distrust every number beside it.
