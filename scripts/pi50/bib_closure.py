#!/usr/bin/env python3
"""Bibliography closure pass: audit every \\bibitem field against a fetched canonical record.

Authorized by the operator 2026-09-12 (bus #233 intent, scope s_bdh-cl_000038). This instrument READS the
manuscript and WRITES ONLY under docs/reports/data/bibliography_closure/. It never edits the .tex: applying
corrections is a separate, human-reviewed step (see the resulting table, then patch \\bibitem lines by hand).

Method
------
1. Parse each \\bibitem{key} ... entry out of docs/papers/rev4-bdh-manuscript.tex.
2. Pull a canonical record:
   - primary: arXiv export API title search (no bot wall; DBLP and Semantic Scholar were both unreachable/rate
     limited from this box on 2026-09-12), falling back to an explicit id list where the arXiv id is already
     known or stated in our own entry;
   - secondary: NON_ARXIV table below for venues arXiv does not cover (Springer/IEEE/MT Summit/books); those get
     recorded as NEEDS-HUMAN with the pointer rather than being silently passed.
3. Compare rendered fields: normalized title, first-author surname, author count, year, venue token.
4. Emit machine-readable JSON plus a markdown verdict table.

Verdict vocabulary
------------------
OK                 every compared field agrees
TITLE-MISMATCH     the title we render does not exist in the canonical record
AUTHOR-MISMATCH    first author or count disagrees
YEAR-MISMATCH      year disagrees with the canonical record's year
NO-CANONICAL       no automated source; needs a human check against the publisher page
COMPOUND           one bibitem covering more than one work (structural defect regardless of fields)
"""
import json
import os
import re
import sys
import time
import difflib
import urllib.parse
import urllib.request

def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "pipeline")) and os.path.isdir(os.path.join(d, "docs")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("cannot locate repo root from scripts/pi50/")


REPO = _repo_root()
TEX = os.path.join(REPO, "docs", "papers", "rev4-bdh-manuscript.tex")
OUT = os.path.join(REPO, "docs", "reports", "data", "bibliography_closure")
API = "https://export.arxiv.org/api/query"

# Entries whose subject is not on arXiv. value = (pointer for a human, expected canonical fields if known)
NON_ARXIV = {
    # fetched 2026-09-12 from the CVF/ECVA proceedings record (bibtex block on the page), NOT from recall:
    # my earlier recollection ("French, Chakravarty, Tuytelaars") was wrong and had already corrupted the entry.
    "mas":        ("https://openaccess.thecvf.com/content_ECCV_2018/html/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.html",
                   {"title": "Memory Aware Synapses: Learning what (not to) forget",
                    "authors": ["Rahaf Aljundi", "Francesca Babiloni", "Mohamed Elhoseiny",
                                "Marcus Rohrbach", "Tinne Tuytelaars"], "year": "2018",
                    "venue": "ECCV 2018, pp. 139-154"}),
    # DOI resolved during this pass: IEEE Transactions on INFORMATION THEORY (not SMC as first guessed), 1970.
    "chow":       ("https://doi.org/10.1109/TIT.1970.1054406",
                   {"title": "On optimum recognition error and reject tradeoff",
                    "authors": ["C. K. Chow"], "year": "1970", "venue": "IEEE Trans. Inf. Theory 16(1)"}),
    "conformal":  ("https://link.springer.com/book/10.1007/b105983 (Algorithmic Learning in a Random World)",
                   {"title": "Algorithmic Learning in a Random World",
                    "authors": ["V. Vovk", "A. Gammerman", "G. Shafer"], "year": "2005", "venue": "Springer"}),
    "europarl":   ("https://www.mt-archive.info/MTS-2005-Koehn.pdf",
                   {"title": "Europarl: A Parallel Corpus for Statistical Machine Translation",
                    "authors": ["P. Koehn"], "year": "2005", "venue": "MT Summit"}),
    "marin":      ("software/model-card reference - verify URL resolves and access date is current", None),
    "engram":     ("vendor model card - verify URL resolves; 'technical report' wording must match a real document", None),
    "bdh":        ("arXiv:2509.26507 (fetched via API when reachable)", None),
}

# arXiv ids already confirmed during authoring of this pass (avoids re-guessing a title search)
KNOWN_IDS = {
    "expertgate": "1611.06194",
    "supsup":     "2006.14769",
    "ewc":        "1612.00796",
    "shazeer":    "1701.06538",
    "l2p":        "2112.08654",
    # si/packnet/piggyback deliberately have NO id here: ids recalled for this pass resolved to unrelated
    # papers (a Massive-MIMO paper, a neutrino-mass paper, a Little-Higgs paper). Those three are routed through
    # title search instead, and any low-similarity hit is reported as NO-CANONICAL rather than as a mismatch.
    # Lesson encoded in the instrument: a remembered arXiv id is not evidence.
    "gem":        "1706.08840",
    "agem":       "1812.00420",
    "switch":     "2101.03961",
    "gshard":     "2006.16668",
    "adamw":      "1711.05101",
    "layernorm":  "1607.06450",
    "multiun":    "n/a-lrec",
}


def norm(s):
    s = re.sub(r"\\[a-z]+\s*", " ", s.lower())
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    return re.sub(r"[^a-z0-9 ]+", "", s).strip()


def surname(tok):
    """Last word of a 'F.~Last' style token."""
    tok = re.sub(r"\{", "", tok).replace("}", "").replace("~", " ").strip()
    return norm(tok.split()[-1]) if tok.split() else ""


def parse_entries(text):
    out = []
    parts = re.split(r"\\bibitem\{([^}]*)\}", text)
    for k, body in zip(parts[1::2], parts[2::2]):
        body = body.split("\\end{thebibliography}")[0]
        m = re.search(r"\\emph\{(.*?)\}", body, re.S)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        before = body[: m.start()] if m else body
        authors = [a for a in re.split(r",| and ", re.sub(r"\s+", " ", before).strip()) if a.strip()]
        ym = re.search(r"\((\d{4})\)", body)
        venue = re.sub(r"\s+", " ", body[m.end():ym.start()]).strip(" .,") if (m and ym) else ""
        out.append({"key": k, "title": title, "authors_raw": authors,
                    "year": ym.group(1) if ym else "", "venue_raw": venue[:80],
                    "raw": re.sub(r"\s+", " ", body).strip()[:400]})
    return out


def api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                                  # noqa: BLE001
            print(f"  retry {attempt+1}: {type(e).__name__} {e}", file=sys.stderr)
            time.sleep(4)
    return ""


def parse_feed(xml):
    hits = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        au = re.findall(r"<name>(.*?)</name>", e)
        pub = re.search(r"<published>(\d{4})", e)
        com = re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", e, re.S)
        idu = re.search(r"<id>(.*?)</id>", e)
        if not t:
            continue
        hits.append({"title": re.sub(r"\s+", " ", t.group(1)).strip(),
                     "authors": au, "year": pub.group(1) if pub else "",
                     "comment": re.sub(r"\s+", " ", com.group(1)).strip() if com else "",
                     "url": idu.group(1).strip() if idu else ""})
    return hits


def venue_year(comment):
    """Publication year stated in an arXiv comment ('CVPR 2017 paper', 'Published ... at ICLR 2019')."""
    m = re.search(r"(CVPR|ICCV|ECCV|NeurIPS|NIPS|ICML|ICLR|AAAI|KDD|ACL|EMNLP|NAACL|UAI|AISTATS|COLM)\s*(\d{4})",
                  comment or "")
    return (m.group(1), m.group(2)) if m else ("", "")


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def compare(ent, canon):
    v, notes = [], []
    if norm(canon["title"]) != norm(ent["title"]):
        v.append("TITLE-MISMATCH")
        notes.append(f'title ours="{ent["title"]}" canonical="{canon["title"]}"')
    ours_surn = [surname(a) for a in ent["authors_raw"]]
    can_surn = [norm(a.split()[-1]) for a in canon["authors"]]
    if ours_surn and can_surn and ours_surn[0] != can_surn[0]:
        v.append("AUTHOR-MISMATCH"); notes.append(f"first author ours={ours_surn[0]} canonical={can_surn[0]}")
    if ours_surn and can_surn and len(ours_surn) < len(can_surn) and len(ours_surn) >= 2:
        notes.append(f"author count ours~{len(ours_surn)} canonical={len(can_surn)}")
    cv, cy = venue_year(canon.get("comment", ""))
    if ent["year"] and canon["year"] and ent["year"] != canon["year"]:
        # arXiv v1 year is the SUBMISSION year; a comment-stated venue year matching ours is not a conflict.
        if cy == ent["year"]:
            notes.append(f"year OK via canonical note ({cv} {cy}); arXiv v1 is {canon['year']}")
        else:
            v.append("YEAR-MISMATCH"); notes.append(f"year ours={ent['year']} canonical(arXiv v1)={canon['year']}")
    if canon.get("comment"):
        notes.append(f"canonical note: {canon['comment'][:110]}")
    return (v or ["OK"]), notes


def main():
    os.makedirs(OUT, exist_ok=True)
    entries = parse_entries(open(TEX, encoding="utf-8").read())
    print(f"parsed {len(entries)} bibitems from {os.path.relpath(TEX, REPO)}")
    results = []
    for i, ent in enumerate(entries):
        k = ent["key"]
        rec = {"key": k, **{x: ent[x] for x in ("title", "authors_raw", "year", "venue_raw")}}
        compound = bool(re.search(r";\s*(and|e\.g\.)", ent["raw"])) or " and successors" in ent["raw"]
        if compound:
            rec["verdict"] = ["COMPOUND"]
            rec["notes"] = ["single bibitem renders more than one work - split it or cite one"]
        if k in NON_ARXIV and (NON_ARXIV[k][1] or compound):
            ptr, exp = NON_ARXIV[k]
            rec["pointer"] = ptr
            if exp:
                cv, cn = compare(ent, {"title": exp["title"], "authors": exp["authors"],
                                       "year": exp["year"], "comment": exp["venue"]})
                rec.setdefault("verdict", [])
                rec["verdict"] = sorted((set(rec["verdict"]) | set(cv)) - {"OK"}) or ["OK"]
                rec["notes"] = rec.get("notes", []) + [f"expected-canonical(table): {x}" for x in cn] + [f"against {ptr}"]
                rec["canonical"] = exp
            results.append(rec)
            print(f"  [{i+1:2d}/{len(entries)}] {k:14s} {'/'.join(rec['verdict'])} (table)")
            continue
        ident = KNOWN_IDS.get(k)
        if ident and re.fullmatch(r"\d{4}\.\d{4,5}", ident):
            hits = parse_feed(api({"id_list": ident, "max_results": 1}))
            src = f"id_list={ident}"
        else:
            q = " ".join(re.findall(r"[A-Za-z0-9]+", ent["title"])[:6])
            hits = parse_feed(api({"search_query": f'ti:"{q}"', "max_results": 5}))
            src = f'ti:"{q}"'
        time.sleep(3.2)                                          # arXiv API courtesy spacing
        if not hits:
            rec["verdict"] = sorted(set(rec.get("verdict", [])) | {"NO-CANONICAL"})
            rec["notes"] = rec.get("notes", []) + [f"no automated hit for {src}; needs publisher-page check"]
            results.append(rec)
            print(f"  [{i+1:2d}/{len(entries)}] {k:14s} {'/'.join(rec['verdict'])} (no hit)")
            continue
        best = max(hits, key=lambda h: sim(h["title"], ent["title"]))
        score = sim(best["title"], ent["title"])
        if score < 0.60:                                          # gate: a weak match is not a canonical record
            rec["verdict"] = sorted(set(rec.get("verdict", [])) | {"NO-CANONICAL"})
            rec["notes"] = rec.get("notes", []) + [
                f"best arXiv hit similarity {score:.2f} < 0.60 ('{best['title'][:60]}'); needs publisher-page check"]
            results.append(rec)
            print(f"  [{i+1:2d}/{len(entries)}] {k:14s} {'/'.join(rec['verdict'])} (weak match {score:.2f})")
            continue
        rec["match_sim"] = round(score, 3)
        rec["canonical"] = {kk: best[kk] for kk in ("title", "year", "comment", "url")}
        rec["canonical"]["authors"] = best["authors"]
        rec["query"] = src
        v, n = compare(ent, best)
        base = [x for x in rec.get("verdict", []) if x != "OK"]
        rec["verdict"] = sorted(set(base) | (set(v) - {"OK"})) or (base or ["OK"])
        rec["notes"] = rec.get("notes", []) + n
        results.append(rec)
        print(f"  [{i+1:2d}/{len(entries)}] {k:14s} {'/'.join(rec['verdict'])}")
    with open(os.path.join(OUT, "closure.json"), "w") as f:
        json.dump(results, f, indent=1)
    lines = ["# Bibliography closure pass", "",
             f"Generated by `scripts/pi50/bib_closure.py` against `{os.path.relpath(TEX, REPO)}` at "
             f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Canonical source: arXiv export API "
             "(DBLP bot-walled, Semantic Scholar rate-limited from this host).", "",
             "| key | verdict | our title | canonical title | notes |", "|---|---|---|---|---|"]
    for r in results:
        ct = (r.get("canonical") or {}).get("title", "—")
        lines.append(f"| `{r['key']}` | **{'/'.join(r['verdict'])}** | {r['title'][:52]} | {str(ct)[:52]} | "
                     f"{'; '.join(r.get('notes', []))[:150]} |")
    bad = [r["key"] for r in results if r["verdict"] != ["OK"]]
    lines += ["", f"**{len(bad)} of {len(results)} entries need action:** {', '.join(bad) or 'none'}", ""]
    with open(os.path.join(OUT, "closure.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {os.path.join(OUT,'closure.json')} and closure.md")
    print(f"needing action: {len(bad)}/{len(results)}: {', '.join(bad)}")


if __name__ == "__main__":
    main()
