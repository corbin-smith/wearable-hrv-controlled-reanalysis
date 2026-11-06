"""
build_s6.py: assemble Supplementary Material v6 for the v12 manuscript.

Usage:
  python build_s6.py --project <project folder with outputs/ and the scripts> --main <v12_build folder>
                     --template s6_template.md --refs-main <v12_references.md> --refs-extra s6_references_extra.md
                     --figures <folder of the nine figure PNGs> --out <dir>

Numbering: Supplementary Tables and Figures take the numbers fixed by the main draft's order of first citation
(<main>/build/v12_supplement_order.json); Supplementary Methods and Results sections take the order of
supplement_items.py. The supplement therefore cannot drift from the citations in the manuscript.

Writes  <out>/s6_model.json            the document as blocks (read by docx_s6.js)
        <out>/s6_rendered.md           every paragraph, caption and table as text (for checking)
        <out>/s6_facts.json            every printed value the supplement text uses
        <out>/s6_tables.json           every table as built (header, rows, note, sources)
and runs docx_s6.js to write <out>/Supplementary_Material_v6.docx.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
for a in ("project", "main", "template", "refs_main", "refs_extra", "figures", "out"):
    ap.add_argument("--" + a.replace("_", "-"), dest=a, required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import s6_common as K  # noqa: E402

K.init(args.project)
sys.path.insert(0, os.path.abspath(args.main))
import supplement_items as SI  # noqa: E402

# ---------------------------------------------------------------- numbering from the main draft
order = json.load(open(os.path.join(args.main, "build", "v12_supplement_order.json"), encoding="utf-8"))
assert sorted(order["tables"]) == sorted(SI.TABLES) and sorted(order["figures"]) == sorted(SI.FIGURES)
assert sorted(order["methods_cited"]) == sorted(SI.METHODS) and sorted(order["results_cited"]) == sorted(SI.RESULTS)
for k, name in enumerate(order["tables"], 1):
    K.SNUM[name] = f"S{k}"
for k, name in enumerate(order["figures"], 1):
    K.SFNUM[name] = f"S{k}"
for name in SI.METHODS:
    K.MNUM[name] = f"S{SI.methods_number(name)}"
for name in SI.RESULTS:
    K.RNUM[name] = f"S{SI.results_number(name)}"

# ---------------------------------------------------------------- facts: main draft first, then the supplement's own
main_facts = json.load(open(os.path.join(args.main, "facts", "v12_facts.json"), encoding="utf-8"))
K.FACTS.update(main_facts)

import s6_tables as T  # noqa: E402
import s6_tables_b  # noqa: E402,F401
import s6_tables_c  # noqa: E402,F401
import s6_tables_d  # noqa: E402,F401
import s6_facts  # noqa: E402

assert sorted(T.TABLES) == sorted(SI.TABLES), set(SI.TABLES) ^ set(T.TABLES)
BUILT = {}
for name in order["tables"]:
    before = set(K.FACTS)
    t = T.TABLES[name]()
    for k in set(K.FACTS) - before:
        if k in main_facts:
            raise SystemExit(f"table {name} overwrote main fact {k}")
    ncol = len(t["header"])
    for r in t["rows"]:
        if isinstance(r, dict):
            if "subheader" in r:
                assert len(r["subheader"]) == ncol, (name, r)
            continue
        assert len(r) == ncol, (name, r, ncol)
    t["number"] = K.SNUM[name]
    t["title"] = SI.TABLES[name]
    BUILT[name] = t
    print(f"  table {K.SNUM[name]:>4} {name:<14} {len(t['rows']):>3} rows  {ncol:>2} cols"
          f"{'  landscape' if t.get('landscape') else ''}")

s6_facts.compute()          # supplement facts that are not produced as a side effect of a table
K.FACTS["s.n_tables"] = str(len(SI.TABLES))
K.FACTS["s.n_figures"] = str(len(SI.FIGURES))

json.dump({n: dict(number=t["number"], title=t["title"], header=t["header"], rows=t["rows"], note=t["note"],
                   sources=t.get("sources", []), landscape=bool(t.get("landscape"))) for n, t in BUILT.items()},
          open(os.path.join(args.out, "s6_tables.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if os.environ.get("S6_TABLES_ONLY"):
    print("tables built; stopping (S6_TABLES_ONLY)")
    sys.exit(0)

# ---------------------------------------------------------------- template
raw = open(args.template, encoding="utf-8").read()
raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
used = set()


def fill(text):
    def rep(m):
        key = m.group(1)
        kind, _, names = key.partition(":") if ":" in key and key.split(":")[0] in ("S", "SF", "SM", "SR") else ("", "", "")
        if kind:
            reg = {"S": K.SNUM, "SF": K.SFNUM, "SM": K.MNUM, "SR": K.RNUM}[kind]
            label = {"S": "Supplementary Table", "SF": "Supplementary Figure", "SM": "Supplementary Methods",
                     "SR": "Supplementary Results"}[kind]
            nums = []
            for n in names.split(","):
                if n not in reg:
                    raise SystemExit(f"unknown {label} {n}")
                nums.append(reg[n])
            nums = sorted(nums, key=lambda x: int(x[1:]))      # cite multiple items in numerical order
            if len(nums) == 1:
                return f"{label} {nums[0]}"
            plural = label.replace("Table", "Tables").replace("Figure", "Figures")
            return f"{plural} {', '.join(nums[:-1])} and {nums[-1]}"
        if key not in K.FACTS:
            raise SystemExit(f"no fact for placeholder {{{key}}}")
        used.add(key)
        return K.FACTS[key]
    return re.sub(r"\{([A-Za-z][A-Za-z0-9_.:,]*)\}", rep, text)


blocks = []
front = {}
para = []
section = None


def flush():
    global para
    if para:
        blocks.append({"type": "para", "text": " ".join(s.strip() for s in para), "section": section})
        para = []


cur_caption = None
for ln in raw.split("\n"):
    st = ln.strip()
    if st.startswith("@") and ":" in st and not blocks and not para:
        k, v = st[1:].split(":", 1)
        front[k.strip()] = v.strip()
        continue
    if not st:
        flush()
        continue
    m = re.match(r"^(#{1,3})\s+(.*)$", st)
    if m:
        flush()
        if len(m.group(1)) == 1:
            section = m.group(2)
        blocks.append({"type": "heading", "level": len(m.group(1)), "text": m.group(2), "section": section})
        continue
    m = re.match(r"^\[\[(TABLES|FIGURES|REFERENCES|CONTENTS)\]\]$", st)
    if m:
        flush()
        blocks.append({"type": m.group(1).lower()})
        continue
    m = re.match(r"^\[\[CAPTION (\w+)\]\]\s*(.*)$", st)
    if m:
        flush()
        blocks.append({"type": "caption", "name": m.group(1), "text": m.group(2)})
        continue
    if blocks and blocks[-1]["type"] == "caption" and not para:
        blocks[-1]["text"] += " " + st
        continue
    para.append(ln)
flush()

for b in blocks:
    if "text" in b and b["type"] in ("para", "caption"):
        b["text"] = fill(b["text"])
front = {k: fill(v) for k, v in front.items()}

captions = {b["name"]: b["text"] for b in blocks if b["type"] == "caption"}
assert sorted(captions) == sorted(SI.FIGURES), set(captions) ^ set(SI.FIGURES)

# Every Supplementary Methods and Results section must exist as a heading, in order.
heads = [b["text"] for b in blocks if b["type"] == "heading" and b["level"] == 2]
exp_m = [f"Supplementary Methods {K.MNUM[n]}. {SI.METHODS[n]}" for n in SI.METHODS]
exp_r = [f"Supplementary Results {K.RNUM[n]}. {SI.RESULTS[n]}" for n in SI.RESULTS]
missing = [h for h in exp_m + exp_r if h not in heads]
if missing:
    raise SystemExit(f"sections missing or misnamed in the template: {missing}")
assert [h for h in heads if h in exp_m] == exp_m and [h for h in heads if h in exp_r] == exp_r

# ---------------------------------------------------------------- references
def read_refs(path):
    out = {}
    for ln in open(path, encoding="utf-8").read().split("\n"):
        ln = re.sub(r"<!--.*?-->", "", ln).strip()
        m = re.match(r"^\[([^\]]+)\]\s+(.*)$", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


refs_main = read_refs(args.refs_main)
refs_extra = read_refs(args.refs_extra)
assert not set(refs_main) & set(refs_extra)
ALL_REFS = {**refs_main, **refs_extra}

# Numbered citations (Frontiers medical style). The supplement numbers its own list, in reading order: the text,
# then the tables (cells and notes, in table order) and the figure captions, wherever [[TABLES]] and [[FIGURES]]
# stand in the template.
import numcite  # noqa: E402  (shared with build_v12.py)

def text_only(bl, built, caps):
    """The citable text in reading order (paragraphs, then table cells and notes, then captions), for the audit."""
    out = []
    for b in bl:
        if b["type"] == "para":
            out.append(b["text"])
        elif b["type"] == "tables":
            for name in order["tables"]:
                t_ = built[name]
                for r in [t_["header"]] + t_["rows"]:
                    if isinstance(r, dict):
                        out.append(" | ".join(str(v) if isinstance(v, str) else " | ".join(map(str, v)) for v in r.values()))
                    else:
                        out.append(" | ".join(str(c).replace("\n", " ") for c in r))
                out.append(t_["note"])
        elif b["type"] == "figures":
            for name in order["figures"]:
                out.append(caps[name])
    return out


with open(os.path.join(args.out, "s6_text_authoryear.md"), "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(text_only(blocks, BUILT, captions)))
CITER = numcite.Citer(list(ALL_REFS))
cell_groups = 0          # citation groups written inside table cells (the checker reads running text only)
for b in blocks:
    if b["type"] == "para":
        b["text"] = CITER.convert(b["text"])
    elif b["type"] == "tables":
        for name in order["tables"]:
            t_ = BUILT[name]
            g0 = CITER.n_groups
            t_["header"] = CITER.convert_any(t_["header"])
            t_["rows"] = CITER.convert_any(t_["rows"])
            cell_groups += CITER.n_groups - g0
            t_["note"] = CITER.convert(t_["note"])
    elif b["type"] == "figures":
        for name in order["figures"]:
            captions[name] = CITER.convert(captions[name])
ref_list = list(CITER.order)
uncited_extra = [k for k in refs_extra if k not in ref_list]
if uncited_extra:
    raise SystemExit(f"extra references never cited: {uncited_extra}")
json.dump({"groups": CITER.n_groups, "groups_in_cells": cell_groups, "citations": CITER.n_citations, "references": len(ref_list)},
          open(os.path.join(args.out, "s6_citation_stats.json"), "w"))
with open(os.path.join(args.out, "s6_text_numbered.md"), "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(text_only(blocks, BUILT, captions)))
print(f"citations numbered: {CITER.n_citations} citations of {len(ref_list)} references in {CITER.n_groups} groups "
      f"({sum(k in refs_extra for k in ref_list)} cited only in the supplement)")

# ---------------------------------------------------------------- inline markup
TOKEN = re.compile(r"(\[\[FLAG:.*?\]\]|\*\*.+?\*\*|\*[^*\s][^*]*?\*)")


def runs(text, base=None):
    base = dict(base or {})
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("[[FLAG:"):
            out.append(dict(base, text="[" + part[2:-2].strip() + "]", highlight=True, bold=True))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            out += runs(part[2:-2], dict(base, bold=True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out += runs(part[1:-1], dict(base, italic=True))
        else:
            out.append(dict(base, text=part))
    return out


def cellruns(c):
    return [runs(line) for line in str(c).split("\n")]


# ---------------------------------------------------------------- model
FIG_FILES = {"effects": "Figure1_EffectSizes.png", "boxplots": "Figure2_BoxPlots.png",
             "intercorr": "Figure7_Intercorrelation.png", "mse": "FigureS_MSE_Curves.png",
             "surrogates": "Figure5_Surrogates.png", "identity": "FigureS_BetaAlphaIdentity.png",
             "mayer": "FigureM_MayerDiagnostic.png", "covariate": "Figure4_Covariate.png",
             "classifier": "Figure6_Classifier.png"}
model = {"front": {k: v for k, v in front.items()}, "blocks": []}
rendered = [f"# {front.get('title', '')}"]
for b in blocks:
    if b["type"] == "heading":
        model["blocks"].append({"type": "heading", "level": b["level"], "text": b["text"]})
        rendered.append("#" * b["level"] + " " + b["text"])
    elif b["type"] == "para":
        model["blocks"].append({"type": "para", "runs": runs(b["text"])})
        rendered.append(b["text"])
    elif b["type"] == "contents":
        items = [f"Supplementary Methods {K.MNUM[n]}. {SI.METHODS[n]}" for n in SI.METHODS] + \
                [f"Supplementary Results {K.RNUM[n]}. {SI.RESULTS[n]}" for n in SI.RESULTS] + \
                [f"Supplementary Table {BUILT[n]['number']}. {BUILT[n]['title']}" for n in order["tables"]] + \
                [f"Supplementary Figure {K.SFNUM[n]}. {SI.FIGURES[n]}" for n in order["figures"]]
        model["blocks"].append({"type": "contents", "items": items})
        rendered += ["- " + i for i in items]
    elif b["type"] == "tables":
        for name in order["tables"]:
            t = BUILT[name]
            rows = []
            for r in t["rows"]:
                if isinstance(r, dict) and "section" in r:
                    rows.append({"section": runs(r["section"])})
                elif isinstance(r, dict) and "subheader" in r:
                    rows.append({"subheader": [cellruns(c) for c in r["subheader"]]})
                else:
                    rows.append([cellruns(c) for c in r])
            note = t["note"] + " Source: " + "; ".join(t.get("sources", [])) + "."
            model["blocks"].append({"type": "table", "number": t["number"], "title": t["title"],
                                    "header": [cellruns(h) for h in t["header"]], "rows": rows,
                                    "note": runs(note), "landscape": bool(t.get("landscape")),
                                    "font": t.get("font", 15)})
            rendered.append(f"**Supplementary Table {t['number']}. {t['title']}**")
            rendered.append("| " + " | ".join(t["header"]) + " |")
            for r in t["rows"]:
                if isinstance(r, dict):
                    rendered.append("| " + (r.get("section") or " | ".join(r.get("subheader"))) + " |")
                else:
                    rendered.append("| " + " | ".join(str(c).replace("\n", " ") for c in r) + " |")
            rendered.append(note)
    elif b["type"] == "figures":
        for name in order["figures"]:
            path = os.path.join(args.figures, FIG_FILES[name])
            if not os.path.exists(path):
                raise SystemExit(f"missing figure {path}")
            num_ = K.SFNUM[name]
            model["blocks"].append({"type": "figure", "number": num_, "path": os.path.abspath(path),
                                    "caption": [dict(text=f"Supplementary Figure {num_}. {SI.FIGURES[name]}. ", bold=True)]
                                    + runs(captions[name])})
            rendered.append(f"**Supplementary Figure {num_}. {SI.FIGURES[name]}.** {captions[name]}")
    elif b["type"] == "references":
        model["blocks"].append({"type": "heading", "level": 1, "text": "References"})
        rendered.append("# References")
        for n_, k in enumerate(ref_list, 1):
            model["blocks"].append({"type": "ref", "runs": runs(f"{n_}. " + ALL_REFS[k])})
            rendered.append(f"{n_}. " + ALL_REFS[k])
    elif b["type"] == "caption":
        pass

unused = sorted(k for k in K.FACTS if k.startswith("s.") and k not in used)
print(f"supplement facts computed: {sum(1 for k in K.FACTS if k.startswith('s.'))}; used in the text: "
      f"{sum(1 for k in used if k.startswith('s.'))}; main-draft facts used: {sum(1 for k in used if not k.startswith('s.'))}")
if unused:
    print("supplement facts not used in the text (they appear in table notes or are checks):", ", ".join(unused))

with open(os.path.join(args.out, "s6_model.json"), "w", encoding="utf-8") as fh:
    json.dump(model, fh, ensure_ascii=False)
with open(os.path.join(args.out, "s6_rendered.md"), "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(rendered))
with open(os.path.join(args.out, "s6_facts.json"), "w", encoding="utf-8") as fh:
    json.dump({k: v for k, v in K.FACTS.items() if k.startswith("s.") or k in used}, fh, ensure_ascii=False, indent=1)
with open(os.path.join(args.out, "s6_references_cited.json"), "w", encoding="utf-8") as fh:
    json.dump(ref_list, fh, ensure_ascii=False)

docx_path = os.path.join(args.out, "Supplementary_Material_v6.docx")
res = subprocess.run(["node", os.path.join(HERE, "docx_s6.js"), os.path.join(args.out, "s6_model.json"), docx_path],
                     capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise SystemExit("docx_s6.js failed")
print("wrote", docx_path)
