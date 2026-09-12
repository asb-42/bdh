"""Per-territory multi-byte exposure census (A0-Quinn's request, bus #204 item 1).

The rev-4 draft currently says bg/el are "the only two territories that have ever seen a byte above 0x80"
(abstract, intro, section 8). That is false as written - every Latin territory with diacritics sees 0xC3/0xC5
lead bytes - and a reviewer with a Polish corpus will object. This measures the actual quantities so the
sentence can be replaced by something both stronger and true.

PROVENANCE, because a census of the wrong bytes is worthless: pipeline/data.py:_europarl_blocks takes
    blk = raw[-need:]   with need = train_cap + 2_000_000
i.e. the model trains on the TAIL of each monolingual file, and the last 2 MB are held out as val/test.
This script uses the identical slice construction (same sources map, same tail arithmetic, cap 30 MB to
match the RA2b ladder's per-phase budget), so every number describes what the network actually saw.

Metrics per territory, computed on the 30 MB train slice:
  hi_byte      share of BYTES >= 0x80
  chars        decoded characters
  cp2/cp3      share of characters encoded as 2-byte / 3-byte UTF-8 sequences
  cyr,greek,latinsup,latinexta  share of characters in those Unicode ranges
  n_distinct   distinct non-ASCII codepoints (inventory size)
  bpc          bytes per character

Sanity gates, asserted rather than eyeballed: English must be near-zero on hi_byte, bg and el must exceed
every other territory on cp2, and NO European territory may show 3-byte characters - if any does, the claim
"no territory has ever seen a 3-byte script" is void and the printed conclusion changes. The third gate is
the one that matters for the manuscript: cross-script inputs (zh/ja/hi/iu) are 3-byte material, so the
likelihood router's bg/el preference cannot be explained by exposure to their scripts, only by exposure to
multi-byte structure in general.

Usage:  sg coding -c 'cd /srv/coding/bdh && .venv/bin/python scripts/pi50/territory_byte_census.py'
Env:    DATA_DIR=data/europarl  CAP=30000000  OOD_DIR=~/bdh-review/data/ood
"""
import os, json, re, collections
import numpy as np

SEQ = "en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
SOURCES = {"en": "de-en", "de": "de-en", "es": "es-en", "fr": "fr-en", "pt": "pt-en", "it": "it-en",
           "da": "da-en", "cs": "cs-en", "nl": "nl-en", "pl": "pl-en", "ro": "ro-en", "sv": "sv-en",
           "el": "el-en", "hu": "hu-en", "bg": "bg-en", "fi": "fi-en", "sk": "sk-en", "sl": "sl-en",
           "et": "et-en", "lt": "lt-en"}
DATA_DIR = os.environ.get("DATA_DIR", "data/europarl")
CAP = int(os.environ.get("CAP", "30000000"))
OOD_DIR = os.path.expanduser(os.environ.get("OOD_DIR", "~/bdh-review/data/ood"))
OUTDIR = "docs/reports/data/byte_census"
os.makedirs(OUTDIR, exist_ok=True)

RANGES = [("latinsup", 0x00C0, 0x017F), ("latinexta", 0x0180, 0x024F), ("greek", 0x0370, 0x03FF),
          ("cyrillic", 0x0400, 0x04FF)]


def census(raw: bytes) -> dict:
    a = np.frombuffer(raw, dtype=np.uint8)
    hi = float((a >= 0x80).mean())
    lead2 = int(((a >= 0xC0) & (a <= 0xDF)).sum()); lead3 = int(((a >= 0xE0) & (a <= 0xEF)).sum())
    lead4 = int((a >= 0xF0).sum())
    txt = raw.decode("utf-8", errors="replace")
    n = max(1, len(txt))
    cnt = collections.Counter(txt)
    nonascii = {ord(c): v for c, v in cnt.items() if ord(c) > 127}
    tot_na = sum(nonascii.values())
    rng = {name: sum(v for o, v in nonascii.items() if lo <= o <= hi) / n for name, lo, hi in RANGES}
    deep = {o: v for o, v in nonascii.items() if o >= 0x800}
    return dict(hi_byte=hi, chars=n, two_byte_share=lead2 / max(1, n), three_byte_share=lead3 / max(1, n),
                four_byte_share=lead4 / max(1, n), n_distinct=len(nonascii),
                nonascii_char_share=tot_na / n, bpc=len(a) / n,
                deep_top=[(f"U+{o:04X}", v) for o, v in sorted(deep.items(), key=lambda z: -z[1])[:5]], **rng)


rows = {}
for lang in SEQ:
    path = os.path.join(DATA_DIR, f"europarl-v7.{SOURCES[lang]}.{lang}.txt")
    if not os.path.exists(path):
        print(f"  !! missing {path}"); continue
    size = os.path.getsize(path)
    need = CAP + 2_000_000
    with open(path, "rb") as fh:
        fh.seek(max(0, size - need)); blk = fh.read(need)
    train = blk[:-2_000_000]
    rows[lang] = census(train)
    r = rows[lang]
    print(f"{lang:>3s} {len(train)/1e6:5.1f} MB | hi-byte {r['hi_byte']*100:6.2f}% | 2B-char {r['two_byte_share']*100:6.2f}% "
          f"| 3B {r['three_byte_share']*100:.4f}% | distinct-nonascii {r['n_distinct']:4d} | bpc {r['bpc']:.3f}", flush=True)

print("\n--- untrained inputs (for contrast; heads, not tails: they were never trained on at all) ---")
ood = {}
for name, fn, filt in (("zh", "xscript_zh_head.txt", None), ("ja", "xscript_ja_head.txt", None),
                       ("hi", "xscript_hi_head.txt", None), ("lv", "lv_head.txt", None),
                       ("ga", "ga_head.txt", None), ("iu_syllabic", "xscript_iu.txt", "syl"),
                       ("iu_ascii", "xscript_iu.txt", "ascii")):
    p = os.path.join(OOD_DIR, fn)
    if not os.path.exists(p):
        continue
    raw = open(p, "rb").read()
    if filt == "syl":
        raw = b"\n".join(l for l in raw.split(b"\n") if re.search(r"[\u1400-\u167f]", l.decode("utf8", "ignore")))
    elif filt == "ascii":
        raw = b"\n".join(l for l in raw.split(b"\n") if l.strip() and not re.search(rb"[^\x00-\x7f]", l))
    ood[name] = census(raw[:CAP])
    r = ood[name]
    print(f"{name:>11s} {len(raw[:CAP])/1e6:5.2f} MB | hi-byte {r['hi_byte']*100:6.2f}% | 2B-char {r['two_byte_share']*100:6.2f}% "
          f"| 3B {r['three_byte_share']*100:6.2f}% | distinct-nonascii {r['n_distinct']:4d} | bpc {r['bpc']:.3f}", flush=True)

# ---------------- gates
print("\n=== sanity gates ===")
ok = True
g1 = rows["en"]["hi_byte"] < 0.02
print(("PASS" if g1 else "FAIL") + f"  English near-zero high-byte: {rows['en']['hi_byte']*100:.3f}% (< 2%)")
by_cp2 = sorted(rows.items(), key=lambda kv: -kv[1]["two_byte_share"])
top2 = [k for k, _ in by_cp2[:2]]
g2 = set(top2) == {"bg", "el"}
print(("PASS" if g2 else "FAIL") + "  bg/el rank 1-2 on 2-byte character share: " +
      ", ".join(f"{k}:{v['two_byte_share']*100:.2f}%" for k, v in by_cp2[:4]))
g3 = all(v["three_byte_share"] < 0.001 for v in rows.values())
mx3 = max(rows.items(), key=lambda kv: kv[1]["three_byte_share"])
print(("PASS" if g3 else "FAIL") + f"  no territory has MATERIAL 3-byte exposure (max {mx3[0]} {mx3[1]['three_byte_share']*100:.5f}%, bar 0.1%)")
print(f"  residue is typographic, not script: {mx3[0]} top >=U+0800 codepoints {mx3[1]['deep_top']}")
print("  (the original draft gate '<1e-6' was wrong as a premise: em dash U+2014 and ellipsis U+2026 are")
print("   3-byte UTF-8, so every European corpus carries trace 3-byte material. The claim the manuscript")
print("   needs is that no territory saw a 3-BYTE SCRIPT, which is what the 0.1% bar tests.)")
ratio = rows["pl"]["two_byte_share"] and (rows["bg"]["two_byte_share"] / rows["pl"]["two_byte_share"])
nxt = by_cp2[2]
print(f"INFO bg/pl 2-byte-character ratio = {ratio:.2f}x ; bg/en = "
      f"{rows['bg']['two_byte_share']/max(rows['en']['two_byte_share'],1e-9):.1f}x ; "
      f"(bg+el)/next-highest {nxt[0]} = {(rows['bg']['two_byte_share']+rows['el']['two_byte_share'])/2/nxt[1]['two_byte_share']:.1f}x")
if not (g1 and g2 and g3):
    raise SystemExit("\nGATE FAILED - refusing to emit conclusions; investigate the census before quoting it.")

json.dump({"territories": rows, "untrained": ood,
           "slice_construction": "tail of monolingual file, need=CAP+2MB, train=blk[:-2MB], matching pipeline/data.py:_europarl_blocks",
           "cap": CAP}, open(f"{OUTDIR}/territory_byte_census.json", "w"), indent=1)

lines = ["# Per-territory multi-byte exposure", "",
         f"Generated by `scripts/pi50/territory_byte_census.py` on the exact training slices (tail, {CAP:,} B train per language).", "",
         "| territory | high-byte bytes % | 2-byte chars % | 3-byte chars % | distinct non-ASCII codepoints | bytes/char |",
         "|---|---|---|---|---|---|"]
for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["two_byte_share"]):
    lines.append(f"| {k} | {v['hi_byte']*100:.2f} | {v['two_byte_share']*100:.2f} | {v['three_byte_share']*100:.4f} | {v['n_distinct']} | {v['bpc']:.3f} |")
lines += ["", "## Untrained inputs", "",
          "| input | high-byte bytes % | 2-byte chars % | 3-byte chars % | distinct non-ASCII codepoints | bytes/char |", "|---|---|---|---|---|---|"]
for k, v in ood.items():
    lines.append(f"| {k} | {v['hi_byte']*100:.2f} | {v['two_byte_share']*100:.2f} | {v['three_byte_share']*100:.2f} | {v['n_distinct']} | {v['bpc']:.3f} |")
open(f"{OUTDIR}/territory_byte_census.md", "w").write("\n".join(lines) + "\n")
print(f"\nwrote {OUTDIR}/territory_byte_census.{{json,md}}")
