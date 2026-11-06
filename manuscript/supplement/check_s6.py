"""
check_s6.py: checks on Supplementary Material v6 and its agreement with the v12 manuscript.

Usage: python check_s6.py <supplement.docx> <s6_template.md> <build dir of build_s6.py> <v12 build dir> <project folder>

1. Numbering: every Supplementary Table, Figure, Methods and Results section the manuscript numbers exists in the
   supplement under the same number and title, in order, and the manuscript cites every one of them.
2. Cross-references inside the supplement point to items that exist.
3. Nothing is left unfilled (placeholders, nan, None) and only the open author flags remain.
4. House style: no em dashes, US spelling.
5. Every number typed into the template (rather than filled from the results) is a declared constant.
6. Citations and references match, and every table names source files that exist.
"""
import csv
import json
import os
import re
import subprocess
import sys

docx, template, sbuild, mbuild, project = sys.argv[1:6]
fails = []


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}]   {label}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(label)


text = subprocess.run(["pandoc", "-t", "plain", "--wrap=none", docx], capture_output=True, text=True).stdout
flat = re.sub(r"\s+", " ", text)
nums = list(csv.DictReader(open(os.path.join(mbuild, "v12_supplement_numbers.csv"), encoding="utf-8")))
main_md = open(os.path.join(mbuild, "v12_rendered.md"), encoding="utf-8").read()
tables = json.load(open(os.path.join(sbuild, "s6_tables.json"), encoding="utf-8"))

print("1. Numbering against the manuscript")
pos = -1
order_ok = True
absent = []
for row in nums:
    kind, n, title = row["Kind"], row["Number"], row["Title"]
    if kind == "Table":
        pat = f"Supplementary Table {n}. {title}"
    elif kind == "Figure":
        pat = f"Supplementary Figure {n}. {title}."
    else:
        pat = f"Supplementary {kind} {n}. {title}"
    # the title must end where the manuscript's title ends (no longer title that merely starts the same way);
    # the contents list repeats every Methods and Results heading, so the last match is the heading itself
    hits = [m.start() for m in re.finditer(re.escape(pat) + r"(?![\w(])", flat)]
    if not hits:
        absent.append(f"{kind} {n}")
        continue
    if kind in ("Table", "Figure"):
        if len(hits) != 2:                       # once in the contents list, once at the table or figure
            absent.append(f"{kind} {n} appears {len(hits)} times")
        if hits[-1] < pos:
            order_ok = False
        pos = max(pos, hits[-1])
check("every Supplementary Table, Figure, Methods and Results section is present with the manuscript's title",
      not absent, absent)
check("tables and figures appear in numerical order", order_ok)


def cited_numbers(md, kind):
    out = set()
    for m in re.finditer(rf"Supplementary {kind}s? ((?:S\d+(?:–S\d+)?(?:, | and )?)+)", md):
        for part in re.findall(r"S(\d+)(?:–S(\d+))?", m.group(1)):
            a = int(part[0])
            b = int(part[1]) if part[1] else a
            out.update(range(a, b + 1))
    return out


for kind, key in (("Table", "Table"), ("Figure", "Figure"), ("Methods", "Methods"), ("Results", "Results")):
    expected = {int(r["Number"][1:]) for r in nums if r["Kind"] == kind}
    cited = cited_numbers(main_md, key)
    check(f"the manuscript cites every Supplementary {kind} ({len(expected)})", expected <= cited, sorted(expected - cited))
    check(f"the manuscript cites no Supplementary {kind} that does not exist", cited <= expected, sorted(cited - expected))

n_t = sum(1 for r in nums if r["Kind"] == "Table")
n_f = sum(1 for r in nums if r["Kind"] == "Figure")
check(f"the introduction states the number of tables and figures ({n_t} and {n_f})",
      f"{n_t} Supplementary Tables and {n_f} Supplementary Figures" in flat)

print("2. Cross-references inside the supplement")
body = flat
for kind in ("Table", "Figure", "Methods", "Results"):
    expected = {int(r["Number"][1:]) for r in nums if r["Kind"] == kind}
    cited = cited_numbers(body, kind)
    check(f"supplement cross-references to {kind} sections exist", cited <= expected, sorted(cited - expected))
for ref in re.findall(r"main Table (\d)", body, flags=re.I):
    check(f"reference to main Table {ref} is to a table that exists", 1 <= int(ref) <= 5)
for ref in re.findall(r"main (?:text )?Figure (\d)", body, flags=re.I):
    check(f"reference to main Figure {ref} is to a figure that exists", 1 <= int(ref) <= 4)
for ref in re.findall(r"main text Section (\d\.\d)", body):
    check(f"reference to main Section {ref} exists", f"## {ref}" in main_md)

print("3. Nothing left unfilled")
leftover = re.findall(r"\{[A-Za-z][^}]*\}", flat)
check("no unfilled placeholders", not leftover, leftover[:5])
def cell_text(c):
    if isinstance(c, str):
        return c
    return " ".join("".join(r["text"] if isinstance(r, dict) else str(r) for r in (l if isinstance(l, list) else [l])) for l in c)


bad_cells = []
for name, t in tables.items():
    for r in t["rows"]:
        if isinstance(r, list):
            for c in r:
                v = cell_text(c).strip()
                if v in ("nan", "NaN", "inf", "-inf") or re.search(r"\b(nan|NaN|inf)\b", v):
                    bad_cells.append(f"{name}: {v!r}")
                if v == "None" and name != "families":   # 'None' is the correction column's value in the families table
                    bad_cells.append(f"{name}: {v!r}")
check("no nan, inf or stray None cells in any table", not bad_cells, bad_cells[:5])
# blank cells are allowed (the conventions paragraph defines them), but a blank first cell must continue a named row
orphans = []
for name, t in tables.items():
    prev_named = False
    for r in t["rows"]:
        if not isinstance(r, list):
            prev_named = False
            continue
        first = cell_text(r[0]).strip()
        if first == "" and not prev_named:
            orphans.append(name)
        prev_named = prev_named or first != ""
        if all(cell_text(c).strip() == "" for c in r[1:]):
            orphans.append(f"{name}: row with no values")
check("every blank first cell continues a named row, and no row is empty", not orphans, orphans[:5])
check("the conventions paragraph defines blank cells", "a blank first cell means the row continues" in flat)
check("no 'nan' or 'NaN' in the text", not re.search(r"\b(nan|NaN)\b", flat), re.findall(r".{30}\b(?:nan|NaN)\b.{30}", flat)[:3])
flags = re.findall(r"\[FLAG:[^\]]*\]", flat)
check("one author flag remains (release DOI)", len(flags) == 1 and "v1.0.1" in flags[0], flags)

print("4. House style")
check("no em dashes", "—" not in flat)
BRITISH = (r"\b(?:\w+(?:isation|isations|ised|ising)|analys(?:e|ed|ing)|normalise|randomise|characterise|summarise|"
           r"minimise|maximise|organise|standardise|recognise|parameterise|utilise|stabilise|emphasise|"
           r"colour\w*|behaviour\w*|favour\w*|neighbour\w*|honour\w*|labour\w*|flavour\w*|tumour\w*|"
           r"centre[sd]?|metres?|litres?|fibres?|labelled|labelling|modelled|modelling|travelled|signalled|cancelled|"
           r"levelled|totalled|analogue|catalogue|programme|whilst|amongst|grey|ageing|judgement|acknowledgement|"
           r"defence|licence|practise|sceptic\w*|per cent|towards|afterwards|focussed)\b")
# words ending in -ised/-ising that are standard US spelling
US_OK = {"advised", "advising", "devised", "devising", "excised", "excising", "exercised", "exercising", "promised",
         "raised", "raising", "revised", "revising", "supervised", "supervising", "surprised", "surprising", "comprised",
         "comprising", "compromised", "premised", "televised", "improvised", "apprised", "noised", "poised", "praised",
         "bruised", "cruised", "chastised", "disguised", "despised", "otherwised", "incised", "unsupervised"}
uk_bad = sorted({w for w in re.findall(BRITISH, flat, flags=re.I) if w.lower() not in US_OK})
check("US spelling", not uk_bad, uk_bad)

print("5. Numbers typed into the template")
raw = open(template, encoding="utf-8").read()
raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
raw = re.sub(r"\{[A-Za-z][A-Za-z0-9_.:,]*\}", " ", raw)
raw = re.sub(r"\[\[FLAG:[^\]]*\]\]", " ", raw)
raw = re.sub(r"^@.*$", " ", raw, flags=re.M)
# citations (years) and section labels
raw = re.sub(r"\b(19|20)\d{2}\b", " ", raw)
raw = re.sub(r"doi:\S+|https?://\S+", " ", raw)                 # identifiers, not quantities
raw = re.sub(r"\b(?:Methods|Results|Table|Figure|Section) S?\d+(?:\.\d)?\b", " ", raw)
literals = sorted(set(re.findall(r"(?<![\w.,])[−+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|(?<![\w.,])[−+]?\d+(?:\.\d+)?(?![\d])", raw)))
DECLARED = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "20", "21", "22",
    "27", "30", "36", "40", "48", "60", "64", "100", "128", "180", "240", "256", "370", "500", "512", "600", "1024",
    "1800", "2000", "2400", "0", "0.003", "0.0039", "0.0078", "0.0156", "0.01", "0.016", "0.039", "0.04", "0.047",
    "0.05", "0.06", "0.062", "0.07", "0.10", "0.13", "0.14", "0.148", "0.15", "0.156", "0.20", "0.25", "0.398", "0.40",
    "0.5", "0.70", "0.80", "0.82", "0.088", "0.98", "1.00", "1.05", "1.50", "1.96", "2.5", "−2", "−0.5", "+0.5",
    "1,000", "1,500", "2,000", "2,500", "10,000", "0.001", "0.2", "0.03", "0.9", "0.6", "0.7", "0.8", "1.000",
    "7.8125", "0.1000", "0.0001", "50", "300", "45", "200", "−5", "0.000",
    # reviewed 19 Oct 2026: box-plot whisker rule, the remaining 1200 s of the window, seeds, the 360 s skip,
    # the 80% coverage rule, 95% intervals and the 97.5th percentile
    "1.5", "1200", "1234", "360", "42", "80", "95", "97.5",
    # the IRASA simulation recorded in compute_irasa's docstring (pipeline.py)
    "0.67", "1.0", "0.1",
}
undeclared = [x for x in literals if x not in DECLARED]
check(f"every typed number is a declared constant ({len(literals)} distinct)", not undeclared, undeclared)

print("6. Citations, references and sources (Frontiers numbered style)")
refs = json.load(open(os.path.join(sbuild, "s6_references_cited.json"), encoding="utf-8"))
n_ref = len(refs)
check("references listed", n_ref >= 30, n_ref)
body_no_refs = flat.split(" References ")[0]
left = [m.group(0) for m in re.finditer(r"\b[A-Z][\w\-ąćęłńóśźż]+(?: et al\.| and [A-Z][\w\-]+)?(?:, | \()(?:19|20)\d{2}\b", body_no_refs)
        if m.group(0).split()[0] not in ("Table", "Figure", "Methods", "Results", "Supplementary", "October", "Python",
                                         "Version", "Section")]
check("no author-year citation is left", not left, left[:5])
# running text without table cells (cells hold SDs, counts and ranges in parentheses)
_js = json.loads(subprocess.run(["pandoc", docx, "-t", "json"], capture_output=True, text=True).stdout)
_js["blocks"] = [b for b in _js["blocks"] if b.get("t") != "Table"]
_nt = subprocess.run(["pandoc", "-f", "json", "-t", "plain", "--wrap=none"], input=json.dumps(_js),
                     capture_output=True, text=True).stdout
_nt = re.sub(r"\s+", " ", _nt)
_nt = _nt[:_nt.rfind(" References ")]
_nt = _nt[_nt.find("Supplementary Methods S1. "):] if "Supplementary Methods S1. " in _nt else _nt
GROUP = r"(?<![\w(])\((\d+(?:–\d+)?(?:, \d+(?:–\d+)?)*)\)"
first, bad = [], []
for m in re.finditer(GROUP, _nt):
    nums = []
    for part in m.group(1).split(", "):
        a, _, b = part.partition("–")
        nums += list(range(int(a), int(b or a) + 1))
    if nums != sorted(nums) or len(set(nums)) != len(nums) or not all(1 <= n <= n_ref for n in nums):
        bad.append(m.group(0) + " after: " + _nt[max(0, m.start() - 50):m.start()])
    for n in nums:
        if n not in first:
            first.append(n)
check("every citation number is between 1 and the number of references, sorted within its group", not bad, bad[:4])
stats = json.load(open(os.path.join(sbuild, "s6_citation_stats.json")))
n_groups = len(re.findall(GROUP, _nt))
n_written = stats["groups"] - stats["groups_in_cells"]
check(f"every number in parentheses in the running text is a citation the builder wrote ({n_groups} found, "
      f"{n_written} written outside table cells)", n_groups == n_written)
# references first cited inside a table cell do not appear in the running text read here; the audit checks the
# full order including cells, so here the first citations in the running text must be in increasing order
check(f"references are first cited in numerical order ({len(first)} of {n_ref} first cited in running text)",
      first == sorted(first) and len(first) >= n_ref - 5, first[:20])
refs_txt = flat[flat.rfind(" References "):]
listed = [int(x) for x in re.findall(r"(?:^| )(\d{1,2})\. (?=(?:de |van )?[A-ZÀ-Ž])", refs_txt)]
check(f"the reference list is numbered 1..{n_ref} in order", listed == list(range(1, n_ref + 1)), listed[:10])
entries = re.split(r" (?=\d{1,2}\. (?:de |van )?[A-ZÀ-Ž])", refs_txt.replace(" References ", " ", 1).strip())
first_auth = [re.match(r"\d+\. (\S+)", x).group(1) for x in entries if re.match(r"\d+\. ", x)]
check("each numbered entry is the reference cited under that number",
      first_auth[:n_ref] == [k.rsplit(" ", 1)[0].split()[0] for k in refs], list(zip(first_auth, refs))[:3])
missing_src = []
for name, t in tables.items():
    if not t["sources"]:
        missing_src.append(name)
    for s in t["sources"]:
        if not os.path.exists(os.path.join(project, s)):
            missing_src.append(f"{name}: {s}")
check("every table names its source files and they exist", not missing_src, missing_src)

print()
print(f"{len(fails)} check(s) failed")
sys.exit(1 if fails else 0)
