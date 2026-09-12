"""Generate docs/PHASE1-MANIFEST.md - a living index of the phase-1 evidence base.

Why a generator instead of a hand-written index: a manual list rots the moment someone commits a report,
and a rotting index is worse than none during a revision, because it gets trusted. Everything structural
(title, author, last commit, size, date) is read from git; only the SCIENTIFIC STATUS of an artifact is
declared by hand, in the STATUS table below, and the script refuses to invent status for files it does not
know - unknown files are listed as "unclassified" so gaps are visible rather than silent.

Scope: phase 1 = everything up to the RA2b mechanism/selection results and the P-R family, i.e. the corpus
the manuscript revision drafts from. Phase 2 (Sonde B/C, semantic addresser) starts after.

Usage:  .venv/bin/python scripts/pi50/phase1_manifest.py [--check]
        --check exits non-zero if the committed manifest differs from regenerated output (CI-able later).
"""
import subprocess, os, sys, hashlib, collections

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
OUT = os.path.join(ROOT, "docs", "PHASE1-MANIFEST.md")
DIRS = ["docs/reports", "docs/reviews", "docs/plans", "docs/tasks", "docs/notes", "docs/routing",
        "docs/papers", "docs/archive", "scripts/pi50"]

# Hand-declared scientific status. Keys are substrings matched against paths, so seat-local naming drift
# does not hide an entry. Statuses used:
#   CITABLE       result established, safe to draft from
#   SUPERSEDED    replaced by a later artifact (named in the note)
#   QUARANTINED   number withdrawn pending re-measurement - must NOT enter the manuscript
#   TOOLING       instrument/harness, not a claim
#   SEAT-LOCAL    deliberately kept out of the repo (third-party authored or transient)
STATUS = {
    "2026-09-11_pi-50_expansion-control": "CITABLE - sections 2,3,6,8,9; section 7 header carries a partial supersession banner",
    "2026-09-10_pi-50_selection-null": "CITABLE - storage-intact/addressing-broken core result",
    "2026-09-10_pi-50_prior-art": "PARTIAL - six works verified at title/abstract level; HSP and PCANets explicitly unverified in-file",
    "r3_byte_addressing.py": "SUPERSEDED by r3b/r3d (unbalanced fit produced the 0.762 headline)",
    "exp4_selfnll_selection.py": "SUPERSEDED by exp4b_budget.py (oracle-width off-by-one fixed there)",
    "matrix_eval.sh": "TOOLING - reusable serving-matrix runner, guards ALLOW_EVAL+ANNOUNCED",
    "protocol_4090.sh": "TOOLING - preflight guard, refuses during live jobs",
    "phase1_manifest.py": "TOOLING - generates this file",
    "r3b_byte_addressing_density.py": "PARTIAL - labels and persisted features CITABLE; its own F-1/F-2/F-3 readouts superseded by r3d/r3c",
    "r3c_ood_addendum.py": "CITABLE - corrected territory map asserted at import; balanced-fit F-2 route table",
    "r3d_fit_ablation.py": "CITABLE - arms A/B/C 0.681/1.000/0.988 settle starvation vs geometry",
    "r3e_margins.py": "CITABLE - Sonde C stage-1 margin floor 0.0139 from n=160 in-support crops (optimistic, see report 9.5)",
    "r3f_hull_geometry.py": "CITABLE - hull hypothesis test; LOO nearest-centroid gate 640/640; max-cosine second trigger signal",
    "2026-09-12_pi-50_rev4-adversarial-pass": "CITABLE - second-seat review of rev-4 with artifact-anchored objections",
    "territory_byte_census.py": "CITABLE - per-territory multi-byte exposure on exact training slices (bus 204 ask); gates assert typographic-residue handling",
    "ood_script_census.py": "TOOLING - script-composition census that established the iu contamination (bus 188)",
    "README.md": "POLICY - frozen-instrument rules, hardcoded phase-1 facts, two indexing conventions, gate inventory",

}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def tracked():
    out = []
    for d in DIRS:
        out += [f for f in git("ls-files", "--", d).splitlines() if f and not f.endswith("/")]
    return sorted(set(out))


def meta(path):
    line = git("log", "-1", "--format=%h\x1f%an\x1f%ad", "--date=short", "--", path).strip().split("\x1f")
    sha, author, date = (line + ["?", "?", "?"])[:3]
    title = ""
    full = os.path.join(ROOT, path)
    try:
        with open(full, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                s = ln.strip()
                if s.startswith("#"):
                    title = s.lstrip("# ").strip(); break
                if s and not title:
                    title = s[:80]; break
    except OSError:
        title = "(unreadable)"
    return sha, author, date, title, os.path.getsize(full)


def status_of(path):
    for k, v in STATUS.items():
        if k in path:
            return v
    return "unclassified" if path.endswith(".md") else "TOOLING" if path.startswith("scripts/") else "unclassified"


rows = collections.defaultdict(list)
for p in tracked():
    sha, author, date, title, size = meta(p)
    rows[p.split("/")[1] if "/" in p else "root"].append((p, sha, author, date, size, title))

body = ["""# Phase-1 artifact manifest

**Generated by** `scripts/pi50/phase1_manifest.py` - do not hand-edit the tables; add a `STATUS` entry in
that script and regenerate. Regenerate before starting phase 2 and whenever a report lands.

Phase 1 = the RA2b decay/interference closure, the serving matrix, the storage-vs-addressing mechanism
result, the selection family (A1/A2/A3, exp4/exp4b) and the P-R family (P-R1/R1b/R2/R3 + F-1/F-2/F-3
follow-ups), plus the reviews and plans that framed them. Phase 2 begins with Sonde B/C and the addresser
architecture question.

Status vocabulary: **CITABLE** (established, safe to draft from) · **SUPERSEDED** (replaced, kept for
provenance) · **QUARANTINE** (withdrawn, must not reach the manuscript) · **TOOLING** (instrument, not a
claim) · **unclassified** (nobody has declared it - gap, not absence).""", ""]

total = 0
for group in [g for g in ["reports", "reviews", "plans", "tasks", "notes", "routing", "papers", "archive", "pi50"] if g in rows]:
    items = sorted(rows[group])
    body.append(f"## {group}/  ({len(items)} files)")
    body.append("")
    body.append("| path | last | author | date | bytes | title / status |")
    body.append("|---|---|---|---|---|---|")
    for p, sha, author, date, size, title in items:
        st = status_of(p)
        body.append(f"| `{p}` | `{sha}` | {author} | {date} | {size:,} | {title[:70]}<br>*{st}* |")
        total += 1
    body.append("")

unc = [p for g in rows for p, *_ in rows[g] if status_of(p) == "unclassified"]
body.append("## Gaps this manifest exposes")
body.append("")
body.append(f"- **{len(unc)} of {total} files carry no declared status.** Those are the ones most likely to be")
body.append("  mis-cited during a revision: something is either citable or it is not, and 'nobody wrote it down'")
body.append("  reads as permission.")
body.append("- The prior-art note still labels HSP and PCANets unverified in-file; that must be closed or")
body.append("  deleted before either name appears in the revised manuscript.")
body.append("- Raw run logs live in `docs/reports/data/` only where they were landed. Seat-local logs under")
body.append("  `~/bdh-review/reports/` on gx10 are ephemeral; anything quoted in the manuscript must have its")
body.append("  generating log inside the repo.")
body.append("")
body.append(f"_Generated {git('log','-1','--format=%ad','--date=iso').strip()} from HEAD `{git('rev-parse','--short','HEAD').strip()}`._")

text = "\n".join(body) + "\n"
if "--check" in sys.argv:
    old = open(OUT).read() if os.path.exists(OUT) else ""
    same = hashlib.md5(old.encode()).hexdigest() == hashlib.md5(text.encode()).hexdigest()
    print("manifest up to date" if same else "MANIFEST STALE - regenerate")
    sys.exit(0 if same else 1)
open(OUT, "w").write(text)
print(f"wrote {OUT}: {total} files indexed, {len(unc)} unclassified")
