# Assessment of the third external review (Grok 4.6, bus #223) — and a verified status of the second (#222)

**pi-50 · 2026-09-12 ·** manuscript checked at `bdh/51dc989` (rev 4.3). Method as before: every reviewer
assertion tested against a landed artifact or the source of truth, not against its prose.

## Verdict on the review

Sharper and more adversarial than the Sonnet review, and closer to what a real conference reviewer would say.
But **its single largest claim — shortcoming #1, verifiability — is factually false about our repository**, and
three of its other criticisms are already answered by text it did not read. What survives is genuinely useful:
one phrasing overclaim in the abstract, one sentence in §6.2 that reads as a proof it does not have, an honest
cost picture for its multi-seed demand, and — found while checking its statistical point — **a defect in a
confidence interval I computed myself.**

## A. Factually wrong, with evidence

**A1 — "the primary data, training logs, and full protocol scripts are not publicly inspectable … the GitHub
link contains the TeX and figure PDFs."** False as of current `main`: `git ls-tree -r origin/main` counts
**47 files under `docs/reports/data/`** and **23 under `scripts/pi50/`**, including the artifacts behind the very
numbers it lists as unauditable — `2026-09-10_ra2b_matrix.csv` (all 400 serving cells),
`2026-09-05_ra2b_acquisition.{csv,json}`, `2026-09-10_fcs_matrix.csv`, the eleven P-R3 generating logs under
`pr3_logs/`, `r3_followups/features_sparse.npz` (the 1.4 MB design matrix that regenerates the addressing
results), and the byte census JSONs. Every instrument that produced a cited number is in-tree with its gate
constants. The correct version of this criticism is narrower and we should concede exactly that much: **model
weights are not released** (1.21–6.95 GB per checkpoint), **seeds/slice hashes for full reproduction are not
consolidated**, and the coordination bus is internal by operator decision. Bit-exactness claims can be checked
against the committed ΔM/churn tables but not re-derived without the checkpoints. Say that precisely instead of
accepting "hard to audit".

**A2 — soft-gate impossibility mischaracterized.** It objects that Proposition 3.5 "does not rule out useful
approximate soft mechanisms". Our statement is titled *"soft gates cannot be **exact**"*, its proof is labelled
"(counterexample class)", and Corollary 3.7 is scoped in-text: *"In broader architectures a learned gate can
alter effective computation; that possibility is outside this [corollary]."* The criticism attacks a claim we
did not make. Do not "temper" anything here; quote the scope clause back.

**A3 — depth amplification "standard recursion"** is our own parenthetical: the proposition is introduced as
*\textscproved (standard recursion)*. We conceded that before it was observed. Its substantive tail — that
uniform-in-depth bounds additionally require contraction `ρ < 1`, which BDH does not support — is the part doing
work, and is untouched by the remark.

**A4 — AI-disclosure residual risk ("without seeing it").** `docs/papers/rev4-ai-disclosure-draft.md` is
tracked (`git ls-files --error-unmatch` succeeds). That concern is satisfied by the repo; the Sonnet review's
last bullet asked for the same check and got it.

## B. Fair, and actionable

**B1 — "Storage is solved" (abstract).** Grok is right that this reads promotional, and it is the one place
where our thesis sentence outruns the measurement. Storage is bit-exact *under masked growth within the tested
chains*; "solved" invites the objection that append-only isolation is trivially preserving, which is also its
own shortcoming #2. Suggested: "Storage is preserved bit-exactly under masked growth", and let the dissociation
theorem carry the conceptual weight. Cost: one line.

**B2 — §6.2's closing sentence claims more than seven experiments establish.** Current text: *"every fixed
arithmetic remedy fails, because a fixed operation is input-independent by definition."* The clause after
"because" reads as a definitional proof; what we have is two structural arguments (LayerNorm annihilates global
gain; a scalar-per-territory fit converges to oracle masking specialized to its fit language) plus seven tested
operators. Grok's "does not exhaust the space of possible input-independent repairs" is correct. Fix: state the
family tested (normalization, culling/top-k, temperature mixing, per-territory reweighting) and say explicitly
that we do not claim exhaustiveness over input-independent operators. This is the same discipline as R4 in my
rev-4 pass — scope the universal, keep the existential.

**B3 — the ~17 % residual is asserted, not decomposed.** Agreed, and worth stating what it would cost: arm A
(synthetic random expansion) was evaluated on English only, so attributing the remainder to learned competition
needs per-language damage profiles for synthetic arms at real widths — i.e. new training, not re-analysis. If we
keep the number, keep the hedge ("second-order") rather than implying a decomposition we performed.

**B4 — multi-seed / multi-order.** Partly answerable from existing artifacts, and we should say so rather than
concede blankly: the 2–4 % floor it mentions is itself a measured replicate result — S1d gives **2.3 %
acquisition / 3.8 % joint / 4.4 % routed** across seeds, and separate contrasts show interpreter/kernel-mode
effects at −0.7 %, below that floor (`docs/reports/2026-09-04_decay-family-two-regimes-and-boundary-repair.md`).
What is genuinely missing is replication *of the headline ladder*. Honest price on this box: 3 038 ms/step ×
10 000 steps ≈ **8.4 h per phase**, so a two-seed replicate of one late phase ≈ 17 h, and a full alternative
ordering ≈ another 115 h. A bounded middle option that answers the substance: a 5-phase mini-ladder in a
different language order at reduced final width, plus one seed replicate at the widest phase. Propose that
instead of accepting an unbounded ask.

**B5 — narrow scope / generality tempering.** Fair as framing criticism: one architecture, one ordering,
language identity as the "domain" axis, ≤ 579 M params, 20 territories. Our Limitations already says most of
it; the paper's own defense is that a substrate-level negative result (parameter isolation ≠ computation
isolation, and the arithmetic origin of the damage) does not require breadth to be true — but it does require us
not to phrase it as a principle. Keep claims indexed to the tested regime; we already do this for thresholds
("empirical for this checkpoint and instrument"), extend the same treatment to the thesis sentence.

**B6 — non-unimodality left uncharacterized.** True and cheap to strengthen: we have the datum (binary search
correct for 4–6/20) and can add the shape observation that motivated it — multiple cumulative-prefix NLL local
minima per domain, i.e. the surface is multimodal rather than merely non-monotone — without needing new runs.

## C. Where checking its statistics claim caught my own error

Grok notes in passing that Wilson intervals are reported "but independence assumptions … are not discussed".
It did not identify the concrete problem, which is mine: §7.2 reports addresser agreement as
**1.000, Wilson [0.977, 1.000] on 160 held-out crops** — but those crops are **8 per domain × 20 domains**, so
they are clustered, not iid. Treating them as independent buys a precision the design does not have; this is the
same pseudo-replication error I flagged in my own atlas findings (#159) and warned about elsewhere, applied to a
number I generated myself.

Correct reporting for a perfect score under clustering: **20/20 domains at 1.00 each (8 crops per domain)**. The
crop-level band understates uncertainty; the per-domain 95 % Wilson interval for a perfect 8/8 is
**[0.68, 1.00]**, and a cluster bootstrap over domains is degenerate at 1.000 because no domain misses. So the
defensible sentence is "perfect agreement on all 20 domains, 8 crops each; crop-level iid intervals are
optimistic because crops within a domain are not independent", not a three-decimal band that looks like n = 160
independent trials. Recommend replacing both occurrences of "[0.977, 1.000]" in the manuscript (§7.2 and the
addressing summary). Logged as a self-correction; the underlying 1.000 agreement is unaffected.

## D. Status of the second review (#222, Sonnet 5) — verified, not assumed

Its top-priority finding was bibliography integrity, and it was **right**: rev 4.2 contained `\bibitem{l2p}` =
"Z.~Feng, et~al. *Continual Learning with Pre-trained Gradual…*" — fabricated author *and* title — plus a
corrupted "A.~R.~A.~R.~T.~Chen" MultiUN entry and two uncited entries (`hsp`, `twist`) matching the pattern of
unverified prior-art names. Rev 4.3 (`51dc989`) removed `hsp`/`twist` — consistent with my A4 requirement that
HSP and PCANets must not appear until verified — and rewrote the others. My independent checks on the result:

- Mechanical audit of the whole bibliography: **26 `\cite` keys ↔ 26 `\bibitem`s, zero undefined, zero dead**.
  Verified by set difference over the `.tex`, not by trusting "zero undefined refs".
- **L2P now correct**: Wang, Zhang, Lee, Zhang, Sun, Ren, Su, Perot, Dy, Pfister, CVPR 2022, pp. 139–149
  (confirmed against the CVF proceedings record and the anthology entry; DOI 10.1109/CVPR52688.2022.00024).
- **MultiUN still wrong, differently.** Current entry conflates two papers: authors "Y. Chen, A. Eisele" and
  venue "LREC (2012)" belong to *"MultiUN v2: UN Documents with Multilingual Alignments"* (Yu Chen, Andreas
  Eisele, LREC 2012 — confirmed from the official proceedings page), while the *title* used belongs to the
  earlier 2010 MultiUN description paper. Either cite v2 with its actual title, or cite the 2010 paper with its
  own authors — and since our cross-script report records only "zh MultiUN (1.29 GB zh side)" with no version,
  state which release was downloaded (likely the v2/OPUS mirror) and hash the 30 MB slice. Fabricated-looking
  citations are an integrity risk exactly as the reviewer said; a half-corrected one is worse than an obviously
  wrong one, because it passes a casual check.
- Remaining members of the same risk class, not yet flagged by either reviewer: `\bibitem{marin}` and
  `\bibitem{engram}` use "et al." with no identifier. If they are software/technical-report references, format
  them as such with a URL and date; otherwise verify or drop. Also note `tinystories`/`tinyladder` were deleted
  rather than corrected — fine while uncited, but the deletion is why the audit above matters: it proves the
  bibliography is currently closed under citation.

## E. Governance

Two machine reviews, delivered hours apart, reached opposite bottom lines on the same draft — Sonnet: arXiv-ready
without reservation; Grok: not at strong-conference standard. Neither endorsement nor dismissal carries weight,
and the productive output has been the intersection of their *checkable* claims: a broken formula (#213 → the 83 %
qualifier), a fabricated bibliography (#222 → rev 4.3), and an unstated clustering assumption (#223 → §C above,
my own number). Continue processing external reviews only through that filter.
