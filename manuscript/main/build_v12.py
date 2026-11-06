"""
build_v12.py: assemble manuscript v12 from the template, the facts and the exhibits.

Usage:
  python build_v12.py --template v12_template.md --facts facts/v12_facts.json \
      --refs v12_references.md --tables <dir with Table1..5 .md and FIGURE_CAPTIONS_v12.md> \
      --fig1 <png> --fig2 <png> --fig3 <png> --fig4 <png> --out <dir>

Writes  <out>/v12_model.json            the document as blocks (read by docx_build.js)
        <out>/v12_rendered.md            the text with every fact filled in (for checking)
        <out>/v12_supplement_numbers.csv provisional Supplementary Table numbers, in order of first citation
        <out>/v12_wordcount.csv          words per section, counted as Frontiers counts them
and then runs docx_build.js to write <out>/manuscript_v12_draft.docx.

The script stops if a placeholder has no fact, if a fact is never used, or if an exhibit is missing.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
for a in ("template", "facts", "refs", "tables", "fig1", "fig2", "fig3", "fig4", "out"):
    ap.add_argument("--" + a, required=True)
# draft: tables and figures inline where first cited (for reading and the preprint)
# frontiers: no figures in the file (they are uploaded separately), tables and then figure captions after the
#            references, as the Frontiers author guidelines require
ap.add_argument("--layout", choices=("draft", "frontiers"), default="draft")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
OUT_DIR = os.path.abspath(args.out)
HERE = os.path.dirname(os.path.abspath(__file__))

FACTS = json.load(open(args.facts, encoding="utf-8"))

# Supplementary items are defined once, in supplement_items.py, and shared with build_s6.py. Tables and
# figures are numbered here by order of first citation; methods and results sections by their order there.
sys.path.insert(0, HERE)
import supplement_items as SI  # noqa: E402
S_TITLES = SI.TABLES
SF_TITLES = SI.FIGURES

# --------------------------------------------------------------------------- template
raw = open(args.template, encoding="utf-8").read()
raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

used = set()
s_order = []
sf_order = []


def _join_numbers(prefix_one, prefix_many, nums):
    nums = list(nums)
    if len(nums) == 1:
        return f"{prefix_one} S{nums[0]}"
    consecutive = all(b == a + 1 for a, b in zip(nums, nums[1:]))
    if consecutive and len(nums) > 2:
        return f"{prefix_many} S{nums[0]}\u2013S{nums[-1]}"
    body = ", ".join(f"S{n}" for n in nums[:-1]) + f" and S{nums[-1]}"
    return f"{prefix_many} {body}"


def fill(text):
    def rep(m):
        key = m.group(1)
        if key.startswith("S:") or key.startswith("SF:"):
            is_fig = key.startswith("SF:")
            names = key.split(":", 1)[1].split(",")
            order, titles = (sf_order, SF_TITLES) if is_fig else (s_order, S_TITLES)
            nums = []
            for name in names:
                if name not in titles:
                    raise SystemExit(f"unknown supplementary {'figure' if is_fig else 'table'} {name}")
                if name not in order:
                    order.append(name)
                nums.append(order.index(name) + 1)
            if nums != sorted(nums):
                raise SystemExit(f"supplementary items cited out of order inside one reference: {key}")
            kind = "Figure" if is_fig else "Table"
            return _join_numbers(f"Supplementary {kind}", f"Supplementary {kind}s", nums)
        if key.startswith("SM:") or key.startswith("SR:"):
            names = key.split(":", 1)[1].split(",")
            reg, numf, label = ((SI.METHODS, SI.methods_number, "Methods") if key.startswith("SM:")
                                else (SI.RESULTS, SI.results_number, "Results"))
            nums = []
            for name in names:
                if name not in reg:
                    raise SystemExit(f"unknown supplementary {label.lower()} section {name}")
                nums.append(numf(name))
                (sm_used if key.startswith("SM:") else sr_used).add(name)
            return _join_numbers(f"Supplementary {label}", f"Supplementary {label}", sorted(nums))
        if key not in FACTS:
            raise SystemExit(f"no fact for placeholder {{{key}}}")
        used.add(key)
        return FACTS[key]
    return re.sub(r"\{([A-Za-z][A-Za-z0-9_.:,]*)\}", rep, text)


sm_used, sr_used = set(), set()


front = {}
blocks = []          # raw block list before inline parsing
section = None       # current H1 title
lines = raw.split("\n")
i = 0
para = []


def flush():
    global para
    if para:
        blocks.append({"type": "para", "text": " ".join(s.strip() for s in para), "section": section})
        para = []


in_refs = False
while i < len(lines):
    ln = lines[i]
    st = ln.strip()
    if st.startswith("@@REFERENCES"):
        flush()
        in_refs = True
        i += 1
        continue
    if st.startswith("@") and ":" in st and not in_refs:
        k, v = st[1:].split(":", 1)
        front[k.strip()] = v.strip()
        i += 1
        continue
    if not st:
        flush()
        i += 1
        continue
    m = re.match(r"^(#{1,3})\s+(.*)$", st)
    if m:
        flush()
        level = len(m.group(1))
        if level == 1:
            section = m.group(2)
        blocks.append({"type": "heading", "level": level, "text": m.group(2), "section": section})
        i += 1
        continue
    m = re.match(r"^\[\[(TABLE|FIGURE) (\d)\]\]$", st)
    if m:
        flush()
        blocks.append({"type": m.group(1).lower(), "n": int(m.group(2)), "section": section})
        i += 1
        continue
    m = re.match(r"^(\d+)\.\s+(.*)$", st)
    if m:
        flush()
        if blocks and blocks[-1]["type"] == "olist":
            blocks[-1]["items"].append(m.group(2))
        else:
            blocks.append({"type": "olist", "items": [m.group(2)], "section": section})
        i += 1
        continue
    para.append(ln)
    i += 1
flush()

# fill facts in text, list items and front matter
for b in blocks:
    if b["type"] == "para":
        b["text"] = fill(b["text"])
    elif b["type"] == "olist":
        b["items"] = [fill(t) for t in b["items"]]
front = {k: fill(v) for k, v in front.items()}

unused = sorted(k for k in FACTS if k not in used and not k.endswith("_dup_check"))
if unused:
    print("facts computed but not used in the text (allowed, listed for review):")
    for k in unused:
        print("   ", k, "=", FACTS[k])

# --------------------------------------------------------------------------- inline markup
TOKEN = re.compile(r"(\[\[FLAG:.*?\]\]|\*\*.+?\*\*|\*[^*\s][^*]*?\*)")


def runs(text, base=None):
    base = dict(base or {})
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("[[FLAG:"):
            inner = part[2:-2].strip()
            out.append(dict(base, text="[" + inner + "]", highlight=True, bold=True))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            out += runs(part[2:-2], dict(base, bold=True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out += runs(part[1:-1], dict(base, italic=True))
        else:
            out.append(dict(base, text=part))
    return out


# --------------------------------------------------------------------------- exhibits
def parse_md_table(path):
    txt = open(path, encoding="utf-8").read().strip().split("\n")
    title = txt[0].strip().strip("*").strip()
    rows = [l for l in txt if l.startswith("|")]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [c for c in cells[2:]]
    note_lines = [l for l in txt[1:] if l.strip() and not l.startswith("|")]
    note = " ".join(note_lines)
    for r in body:
        assert len(r) == len(header), (path, r)
    return title, header, body, note


TABLE_FILES = {1: "Table1_Participants.md", 2: "Table2_Primary.md", 3: "Table3_Controls.md",
               4: "Table4_Estimators.md", 5: "Table5_Robustness.md"}
TABLES = {}
for n, f in TABLE_FILES.items():
    p = os.path.join(args.tables, f)
    if not os.path.exists(p):
        raise SystemExit(f"missing table {p}")
    title, header, body, note = parse_md_table(p)
    assert title.startswith(f"Table {n}."), title
    if n == 2:  # the per-row "Statistic" column repeats one value; state it in the note instead
        k = header.index("Statistic")
        assert all(r[k] == "rank-biserial r" for r in body)
        header = header[:k] + header[k + 1:]
        body = [r[:k] + r[k + 1:] for r in body]
    TABLES[n] = dict(title=title, header=header, rows=body, note=note)

cap_txt = open(os.path.join(args.tables, "FIGURE_CAPTIONS_v12.md"), encoding="utf-8").read()
CAPS = {}
for m in re.finditer(r"\*\*Figure (\d)\. (.+?)\*\*\n(.*?)(?=\n\*\*Figure \d\.|\Z)", cap_txt, flags=re.S):
    n = int(m.group(1))
    body = " ".join(l.strip() for l in m.group(3).strip().split("\n") if l.strip())
    CAPS[n] = (m.group(2).strip(), body)
assert sorted(CAPS) == [1, 2, 3, 4], sorted(CAPS)
FIGS = {1: args.fig1, 2: args.fig2, 3: args.fig3, 4: args.fig4}
for n, p in FIGS.items():
    if not os.path.exists(p):
        raise SystemExit(f"missing figure {p}")

# --------------------------------------------------------------------------- references
refs = []
cite_keys = []
for ln in open(args.refs, encoding="utf-8").read().split("\n"):
    ln = re.sub(r"<!--.*?-->", "", ln).strip()
    m = re.match(r"^\[([^\]]+)\]\s+(.*)$", ln)
    if m:
        cite_keys.append(m.group(1))
        refs.append(m.group(2))

# --------------------------------------------------------------------------- numbered citations
# The template is written with author-year citations; Frontiers in Psychiatry uses numbered references. Every
# citation is replaced by its reference number, in reading order: text, then each table's cells and note and each
# figure's caption at the point where the table or figure is placed.
import numcite  # noqa: E402

def render_md(bl):
    out = []
    for b in bl:
        if b["type"] == "para":
            out.append(b["text"])
        elif b["type"] == "olist":
            out += [f"{k + 1}. {t}" for k, t in enumerate(b["items"])]
        elif b["type"] == "heading":
            out.append("#" * b["level"] + " " + b["text"])
        else:
            out.append(f"[[{b['type'].upper()} {b['n']}]]")
    return out


# the filled text before numbering, kept so the audit can check every number against the citation it replaced
with open(os.path.join(OUT_DIR, "v12_rendered_authoryear.md"), "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(render_md(blocks)))
CITER = numcite.Citer(cite_keys)
cell_groups = 0          # citation groups written inside table cells (the checker reads running text only)
for k, v in front.items():
    if numcite.leftover_author_year(v):
        raise SystemExit(f"citation in front matter field {k}")
seen_t, seen_f = set(), set()
for b in blocks:
    if b["type"] == "para":
        b["text"] = CITER.convert(b["text"])
    elif b["type"] == "olist":
        b["items"] = [CITER.convert(x) for x in b["items"]]
    elif b["type"] == "table" and b["n"] not in seen_t:
        seen_t.add(b["n"])
        T_ = TABLES[b["n"]]
        g0 = CITER.n_groups
        T_["header"] = CITER.convert_any(T_["header"])
        T_["rows"] = CITER.convert_any(T_["rows"])
        cell_groups += CITER.n_groups - g0
        T_["note"] = CITER.convert(T_["note"])
    elif b["type"] == "figure" and b["n"] not in seen_f:
        seen_f.add(b["n"])
        CAPS[b["n"]] = (CAPS[b["n"]][0], CITER.convert(CAPS[b["n"]][1]))
uncited_refs = [k for k in cite_keys if k not in CITER.order]
if uncited_refs:
    raise SystemExit(f"references never cited: {uncited_refs}")
REFS_BY_KEY = dict(zip(cite_keys, refs))
print(f"citations numbered: {CITER.n_citations} citations of {len(CITER.order)} references, in {CITER.n_groups} groups")
json.dump({"groups": CITER.n_groups, "groups_in_cells": cell_groups, "citations": CITER.n_citations, "references": len(CITER.order)},
          open(os.path.join(OUT_DIR, "v12_citation_stats.json"), "w"))

rendered_md = render_md(blocks)
refs_numbered_md = [f"{k}. {REFS_BY_KEY[key]}" for k, key in enumerate(CITER.order, 1)]

# --------------------------------------------------------------------------- model
model = {"front": {}, "blocks": []}
model["front"] = {
    "title": front["title"], "running": front.get("running", ""),
    "author": front["author"], "affil": front["affil"],
    "corr": runs(front["corr"]), "orcid": front["orcid"],
    "keywords": front["keywords"],
}

placed_tables, placed_figs = set(), set()
MAIN = ("1 Introduction", "2 Materials and Methods", "3 Results", "4 Discussion", "5 Conclusions")
words = {}


def count_words(t):
    t = re.sub(r"\[\[FLAG:.*?\]\]", "", t)
    t = t.replace("*", "")
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


deferred_tables, deferred_figs = [], []
for b in blocks:
    sec = b.get("section")
    if args.layout == "frontiers" and b["type"] in ("table", "figure"):
        (deferred_tables if b["type"] == "table" else deferred_figs).append(b["n"])
        (placed_tables if b["type"] == "table" else placed_figs).add(b["n"])
        continue
    if b["type"] == "heading":
        model["blocks"].append({"type": "heading", "level": b["level"], "text": b["text"]})
        if b["text"] == "Abstract":
            pass
    elif b["type"] == "para":
        model["blocks"].append({"type": "para", "runs": runs(b["text"]),
                                "keywords_after": sec == "Abstract"})
        words[sec] = words.get(sec, 0) + count_words(b["text"])
    elif b["type"] == "olist":
        model["blocks"].append({"type": "olist", "items": [runs(t) for t in b["items"]]})
        words[sec] = words.get(sec, 0) + sum(count_words(t) for t in b["items"])
    elif b["type"] == "table":
        t = TABLES[b["n"]]
        placed_tables.add(b["n"])
        model["blocks"].append({"type": "table", "n": b["n"], "title": t["title"], "header": t["header"],
                                "rows": t["rows"], "note": runs(t["note"]),
                                "landscape": b["n"] == 3})
    elif b["type"] == "figure":
        title, body = CAPS[b["n"]]
        placed_figs.add(b["n"])
        model["blocks"].append({"type": "figure", "n": b["n"], "path": os.path.abspath(FIGS[b["n"]]),
                                "caption": [dict(text=f"Figure {b['n']}. {title} ", bold=True)] + runs(body)})
assert placed_tables == {1, 2, 3, 4, 5} and placed_figs == {1, 2, 3, 4}
model["blocks"].append({"type": "heading", "level": 1, "text": "References"})
for k, key in enumerate(CITER.order, 1):
    model["blocks"].append({"type": "ref", "runs": runs(f"{k}. " + REFS_BY_KEY[key])})
if args.layout == "frontiers":
    deferred_tables, deferred_figs = sorted(deferred_tables), sorted(deferred_figs)      # numbered order
    assert deferred_tables == [1, 2, 3, 4, 5] and deferred_figs == [1, 2, 3, 4], (deferred_tables, deferred_figs)
    model["blocks"].append({"type": "heading", "level": 1, "text": "Tables"})
    for n in deferred_tables:
        t = TABLES[n]
        model["blocks"].append({"type": "table", "n": n, "title": t["title"], "header": t["header"],
                                "rows": t["rows"], "note": runs(t["note"]), "landscape": n == 3})
    model["blocks"].append({"type": "heading", "level": 1, "text": "Figure captions"})
    for n in deferred_figs:
        title, body = CAPS[n]
        model["blocks"].append({"type": "para", "runs": [dict(text=f"Figure {n}. {title} ", bold=True)] + runs(body)})

main_words = sum(words.get(s, 0) for s in MAIN)
abs_words = words.get("Abstract", 0)
model["front"]["counts"] = (f"Words: {main_words:,} (main text, excluding abstract, headings, tables, captions, "
                            f"statements and references); abstract {abs_words}. Tables: 5. Figures: 4.")

with open(os.path.join(OUT_DIR, "v12_model.json"), "w", encoding="utf-8") as fh:
    json.dump(model, fh, ensure_ascii=False)
with open(os.path.join(OUT_DIR, "v12_rendered.md"), "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(rendered_md))
with open(os.path.join(OUT_DIR, "v12_references_numbered.md"), "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(refs_numbered_md) + "\n")
with open(os.path.join(OUT_DIR, "v12_references_keys.json"), "w", encoding="utf-8") as fh:
    json.dump(CITER.order, fh, ensure_ascii=False)          # keys in reference-number order
with open(os.path.join(OUT_DIR, "v12_supplement_numbers.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["Kind", "Number", "Name", "Title"])
    for k, name in enumerate(s_order, 1):
        w.writerow(["Table", f"S{k}", name, S_TITLES[name]])
    for k, name in enumerate(sf_order, 1):
        w.writerow(["Figure", f"S{k}", name, SF_TITLES[name]])
    for name in SI.METHODS:
        w.writerow(["Methods", f"S{SI.methods_number(name)}", name, SI.METHODS[name]])
    for name in SI.RESULTS:
        w.writerow(["Results", f"S{SI.results_number(name)}", name, SI.RESULTS[name]])
missing_s = sorted(set(S_TITLES) - set(s_order))
missing_f = sorted(set(SF_TITLES) - set(sf_order))
if missing_s or missing_f:
    raise SystemExit(f"supplementary items defined but never cited: tables {missing_s}, figures {missing_f}")
json.dump({"tables": s_order, "figures": sf_order, "methods_cited": sorted(sm_used), "results_cited": sorted(sr_used)},
          open(os.path.join(OUT_DIR, "v12_supplement_order.json"), "w"), indent=1)
with open(os.path.join(OUT_DIR, "v12_wordcount.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["Section", "Words"])
    for s, n in words.items():
        w.writerow([s, n])
    w.writerow(["Main text (sections 1 to 5)", main_words])

print("word counts:")
for s, n in words.items():
    print(f"   {s:<35} {n:>6}")
print(f"   {'MAIN TEXT':<35} {main_words:>6}")
print("supplementary tables cited:", ", ".join(f"S{k}={n}" for k, n in enumerate(s_order, 1)))
print("supplementary figures cited:", ", ".join(f"S{k}={n}" for k, n in enumerate(sf_order, 1)))

docx_path = os.path.join(OUT_DIR, "manuscript_v12_draft.docx" if args.layout == "draft" else "manuscript_v12_frontiers.docx")
res = subprocess.run(["node", os.path.join(HERE, "docx_build.js"),
                      os.path.join(OUT_DIR, "v12_model.json"), docx_path],
                     capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise SystemExit("docx_build.js failed")
print("wrote", docx_path)
