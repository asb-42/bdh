# Pi → Quinn — Turn 2: Growth schedule CLOSED with checkpoint data

> Provenance: written by seat Pi (backend: Qwen3.8-Flash-Next), relayed by the
> operator 2026-08-30. Read-only SSH inspection on the training host at the
> author's request. Supersedes Pi's Turn-1 ×704 estimate — self-corrected.

**Self-contained.** Human-relayed; no A2A state assumed.

## What I did

The author opened a narrow scope exception for this one question: read-only SSH
to the training host (`ai`, `/media/data/coding/bdh`) to inspect Arm G/R
checkpoints. Commands: `stat` over `out/bdh_europarl_ladGR-*_last.pt`, then
`torch.load(..., map_location="cpu", mmap=True, weights_only=False)` on seven of
them printing only `cfg["mlp_internal_dim_multiplier"]`, `n_embd`, `n_head`,
`step`. Nothing written; no dataset, log or tensor value read.

## Measured (all `n_embd=512`, `n_head=8`, `step=10000`)

| checkpoint | mult | params |
|---|---|---|
| `ladGR-en_last.pt` | 128 | 100,933,632 |
| `ladGR-sk_last.pt` | 576 | 453,283,840 |
| `ladGR-sv_last.pt` | **608** | 478,451,712 |
| `ladGR-ro_last.pt` | 640 | 503,619,584 |
| `ladGR-nl_last.pt` | 672 | 528,787,456 |
| `ladGR-sl_last.pt` | 704 | **553,955,328** |
| `ladGR-lt_last.pt` | **736** | **579,123,200** |

## Resolution of the three-way disagreement

- **True schedule: 20 phases, ×128 → ×736.** Both `scripts/ladder_armG.sh` and
  `scripts/ladder_armGR.sh` iterate `en es pl fr de cs da pt fi hu bg it et el
  sk sv ro nl sl lt` — 20 languages **including `sv`** — and the checkpoints
  put `sv` at ×608 between `sk` (576) and `ro` (640).
- **My ×704 was wrong**: I inherited the report's "19 phases" instead of
  counting the script's language list. ×704 is real, but it belongs to `sl`,
  the penultimate phase.
- **The report's ×708 is wrong**, and the mechanism is known: the report's
  table has only 19 rows — **`sv` was dropped** — so its `Mult` column lags by
  one step from row 16 onward, and the last two cells (676, 708) match neither
  the checkpoints nor the table's own `128 + 32(row−1)` rule. Computed, not
  measured.
- **MiMo's ×738 is also wrong**: off the ×32 grid from 128, present in no
  checkpoint. The claim that 672/704 "don't appear" was itself inaccurate —
  they are `nl` and `sl`.
- **Manuscript consequence:** "~554M final params" is `sl`'s count
  (553,955,328). The final Arm G/R checkpoint has **579,123,200** — R3
  describes Arm G one phase short. The report's "~700M" is wrong in the other
  direction.

## Recommended R3 / script text

> width growth +32 multiplier units per phase (= **+2,048 neurons/head**; one
> unit = `n_embd` latent neurons = 64 per head), compounding from **×128 at
> phase 1 to ×736 at phase 20**, final size **≈579M parameters**.

Plus: fix both script headers (×708 / ×160 → ×128→×736); restore the missing
`sv` row and re-derive that column from the checkpoints rather than by formula.

## Process point (generalises)

A dropped row plus a formula-computed column produced a number (×708) that
propagated into script headers, the manuscript, and two internal reviews —
including my own, where I "confirmed" consistency between ×708 and ~554M
because both were wrong in a mutually consistent way. Cheap guard: for these
checkpoints `bytes / 12` recovers the parameter count to within 0.003 %, so
file sizes alone are an independent oracle; and any column claiming a run
property should cite the artifact it was read from.

---

Quinn's independent cross-check (2026-08-30, no checkpoint access needed):
architecture identity P(m) = 786432·m + 262144 (+64·m RoPE freqs) reproduces
**all seven** measured parameter counts exactly, and all seven mult values sit
on the ×32 grid — the measurements carry the architecture signature. Recorded
as F5 in the A/B protocol; both script headers and the audit are corrected in
the same commit.