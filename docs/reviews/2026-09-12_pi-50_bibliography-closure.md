# Bibliography closure pass — every `\bibitem` field checked against a fetched record

**pi-50 · 2026-09-12 ·** authorized by the operator ("apply the TeX fix and run a bibliography closure pass");
intent envelope bus #233, write scope `s_bdh-cl_000038`. Manuscript: `docs/papers/rev4-bdh-manuscript.tex`
(rev 4.4 → patched here). Instrument: `scripts/pi50/bib_closure.py`; data: `docs/reports/data/bibliography_closure/`.

## What this pass changed

**(1) Two corrupted TeX commands repaired.** `\texttt{…}` (Fig. 3 caption) and `\textbf{Learning to Prompt …}`
(prior art) were stored as a literal **TAB + `exttt{` / `extbf{`**: a patch script had used an ordinary Python
string, so the `\t` in `\texttt` was consumed as an escape. Diff is exactly two lines; post-fix scan confirms
**zero control characters** remain anywhere in the file. Root cause, blast radius and the standing rule
(raw strings for TeX splices, plus `grep -P "[\x00-\x08\x0b-\x1f]"` in post-patch verification) are in
`docs/reviews/2026-09-12_pi-50_external-review-4-assessment.md` §A1.

**(2) Six bibliography entries corrected, one removed, eleven given identifiers.** Every change below is keyed to
a record fetched during this pass, not to anyone's recollection — including mine, which turned out to be the
proximate cause of one of the defects (§self-correction).

| key | field | was | now | canonical source |
|---|---|---|---|---|
| `expertgate` | title | *Expert Gate: Task- and Expert-Conditioned Routing Networks* | *Expert Gate: Lifelong Learning with a Network of Experts* | arXiv:1611.06194 (comment: "CVPR 2017 paper") |
| `supsup` | title **+ authors** | *Superposition Enables Many-Task Learning (SupSup)*; Wortsman, **M. C. Riemer**, **G. Ilharco** | *Supermasks in Superposition*; Wortsman, Ramanujan, Liu, Kembhavi, Rastegari, Yosinski, Farhadi | arXiv:2006.14769 (comment: "NeurIPS 2020 Camera Ready") — Riemer and Ilharco are **not** authors |
| `mas` | authors | R. Aljundi, **M. B. French**, **B. S. Chakravarty**, M. Tuytelaars | R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, T. Tuytelaars; + pp. 139–154 | CVF/ECVA proceedings record (`Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.html`, bibtex block) |
| `chow` | title | *On Optimum Recognition Error and Reject Tradeoff **for Nonparametric Detection System*** | *On optimum recognition error and reject tradeoff* | DOI 10.1109/TIT.1970.1054406 — venue in our entry (IEEE Trans. Inf. Theory 16(1), 1970) was already **correct**; only the trailing subtitle was invented |
| `ewc` | authors | …A. A. Rusu, **K. Kilan, R. Google-DeepMind**, et al. | J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, et al. | arXiv:1612.00796 (14 authors). "R. Google-DeepMind" was an institution rendered as a person |
| `gshard` | authors | …H. Zhang, **M. Firus**, R. Anil, A. Pang | D. Lepikhin, H. Lee, Y. Xu, D. Chen, et al. | arXiv:2006.16668 (9 authors: Fırat, Huang, Krikun, Shazeer, Chen — "Firus", "Anil", "Pang" do not appear) |
| `byte rnn` | *entry removed* | one `\bibitem` bundling LSTM-1997 **and** PixelRNN ("*Pixel RNN*", arXiv:1601.06759) | — | see §removal |

Identifier annotations (`; arXiv:ID.` / `; DOI …`) were added to `adamw`, `agem`, `gem`, `l2p`, `multilingual-cl`,
`piggyback`, `shazeer`, `si`, `switch`, `expertgate`, `supsup`, `chow`, `ewc`, `gshard` — every one taken from the
record the instrument actually matched (title similarity 1.0), never typed from memory.

**Abstract citation fixed.** Line 33 read *"a byte-level depth-recurrent language model~\cite{byte rnn}"* — citing
LSTM-1997/PixelRNN for an architecture claim. Now `\cite{bdh}` (Kosowski et al., arXiv:2509.26507), the paper that
defines this architecture and whose terminology we treat as authoritative. One token; no prose reworded.

## Closure state after the pass

`distinct \cite keys = 25`, `distinct \bibitem keys = 25`, **undefined = ∅, uncited = ∅**, 29 `\begin/\end` pairs
balanced, zero control characters. (My first closure checker reported this as broken because it did not split
multi-key `\cite{a,b}` — a bug in my own guard, caught before any edit shipped, and worth remembering when
asserting on LaTeX: brace-groups hold comma-lists.)

Re-running the instrument on the patched file drops "needs action" from 16/26 to 6/25, and all six residuals are
adjudicated non-defects:

| residual | why it is not a defect |
|---|---|
| `packnet`, `ewc`, `gshard` YEAR-MISMATCH | arXiv **v1 submission** year ≠ publication year (PackNet 2017→CVPR 2018; EWC 2016→PNAS 2017; GShard 2020→ICLR 2021). Our stated years match the venues. Where the arXiv comment names the venue-year (`adamw`, `agem`, `l2p`, `expertgate`) the instrument already suppresses the false flag; these three have no such comment. |
| `marin`, `engram` NO-CANONICAL | software/vendor references, absent from arXiv by nature. Both URLs were resolved live this pass: `https://mtracker.oa.dev` → HTTP 200, `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` → HTTP 200. Content-level claims about a 2026 vendor card still need an author who will stand behind them. |
| `multiun` NO-CANONICAL | LREC 2012 is not on arXiv; verified manually against the official ACL proceedings page during the rev-4 cycle — Yu Chen & Andreas Eisele, *MultiUN v2: UN Documents with Multilingual Alignments*, LREC 2012 ✓ matches the entry. |

## Self-correction: my earlier recommendation created one of these defects

In bus #210 (rev-4 adversarial pass, item R3) I reported the `mas` entry as corrupted and prescribed
"French, Chakravarty, Tuytelaars". **That prescription was wrong.** Aljundi *is* MAS's first author; the actual
list is Aljundi, Babiloni, Elhoseiny, Rohrbach, Tuytelaars. Quinn applied my incorrect instruction half-way and
produced "R. Aljundi, M. B. French, B. S. Chakravarty, M. Tuytelaars" — a mangled hybrid that looked more
suspicious than the original. My #231 note framed this as "the patch half-applied my fix"; the fuller truth is that
**the fix itself was recalled rather than fetched**, so the corruption was mine to begin with.

Same class, second instance: I had marked `expertgate` and `supsup` "verified" in #210 after checking **authors
only** — their titles were fabricated. And during this pass, three arXiv IDs I supplied from memory
(PackNet, Piggyback, Synaptic Intelligence) resolved to unrelated papers — a massive-MIMO paper, a neutrino-mass
paper, and a Little-Higgs paper. Those bogus IDs made my first instrument run report three phantom
TITLE/AUTHOR mismatches on entries that were perfectly correct.

The lesson generalizes past bibliographies: **a remembered identifier is not evidence, and a partially-checked
field is not a verified entry.** The instrument now encodes that — recalled IDs were deleted from its table, weak
matches (title similarity < 0.60) are reported as `NO-CANONICAL` rather than as mismatches, and every expectation
it carries is labelled with the URL it came from. One property of the design is worth noting: a fabricated title
cannot be found by searching for itself, which is exactly why `supsup` and `expertgate` came back
`NO-CANONICAL (weak match)` on the first run rather than matching something.

## Removal of `byte rnn`, disclosed in full

That `\bibitem` bundled two works ("Long Short-Term Memory … ; and successors applying byte/character-level
models, e.g. … *Pixel RNN*"), and its only `\cite` site was the abstract sentence repointed to `{bdh}` above —
leaving it a dead entry. Rather than ship an uncited bibitem (the defect class I flagged in `hsp`/`twist`), I
removed it. Nothing else in the manuscript referenced it. If the authors want the byte-level lineage credit back,
these are the two correctly-fielded entries, ready to paste:

```latex
\bibitem{lstm} S.~Hochreiter, J.~Schmidhuber. \emph{Long Short-Term Memory.}
  Neural Computation 9(8), pp.~1735--1780 (1997).
\bibitem{pixelrnn} A.~van~den~Oord, N.~Kalchbrenner, K.~Kavukcuoglu.
  \emph{Pixel Recurrent Neural Networks.} ICML (2016); arXiv:1601.06759.
```

Note the second title: the paper is *Pixel Recurrent Neural Networks*; "PixelRNN" is the model name, and the
removed entry had it as `\emph{Pixel RNN.}` — a third wrong title string that no reviewer caught.

## Reproducing

```bash
.venv/bin/python scripts/pi50/bib_closure.py          # reads the .tex, writes only docs/reports/data/...
```
Output: `closure.json` (per-entry ours/canonical fields, similarity, verdicts, notes) and `closure.md` (table).
Exit is informational; the script never edits the manuscript. Verdict vocabulary and the DBLP/Semantic-Scholar
unavailability that shaped the design are documented in its docstring.

## Limits of this pass

- It verifies **fields**, not whether a cited paper supports the sentence citing it. The `byte rnn` case was only
  caught because the abstract's claim and the entry's subject happened to disagree; a machine check cannot see a
  misapplied-but-correctly-formatted citation. That judgment remains the authors'.
- Canonical source coverage is arXiv-only, plus four hand-fetched publisher records (`mas`, `chow`, `multiun`, and
  the ECVA/ECCV page generally). ACM/IEEE/Springer paywalls and the DBLP bot wall blocked automation here; a
  future run with a DBLP or OpenAlex key could widen coverage.
- Venue-page numbers (pp., vol., issue) were added only where a fetched record stated them (`mas`). Their absence
  elsewhere is not a defect.
- Author renderings follow the manuscript's existing initials style; for >4-author papers I used first-three-or-six
  + `et~al.` as the entry already did, so counts differ from canonical lists by design.
