"""
check_v12.py: checks run on the BUILT manuscript (re-extracted from the .docx with pandoc), not on any
in-memory text, so a check passes only if the change is really in the file.

Usage: python check_v12.py <docx> <template> <facts.json> <refs.md> <supplement_numbers.csv> <tables_dir>

Exit code 1 if any check fails. Prints a report; every literal number in the template that is not a
fact must appear in LITERALS below with the reason it is allowed.
"""
import csv
import json
import re
import subprocess
import sys

DOCX, TEMPLATE, FACTS, REFS, SNUM, TABLES = sys.argv[1:7]
fails = []


def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


text = subprocess.run(["pandoc", DOCX, "-t", "plain", "--wrap=none"], capture_output=True, text=True).stdout
flat = re.sub(r"\s+", " ", text)
facts = json.load(open(FACTS, encoding="utf-8"))
tpl = re.sub(r"<!--.*?-->", "", open(TEMPLATE, encoding="utf-8").read(), flags=re.S)

body_end = flat.find(" References ")
body = flat[:body_end] if body_end > 0 else flat

print("1. Markup and placeholders")
check("{" not in body and "}" not in body, "no unfilled {placeholders}")
check(not re.search(r"(?<!\w)\*[^ ]", body) and "**" not in body, "no leftover * markup")
check("[[" not in flat, "no leftover [[ markers (flags render as [FLAG: ...])")
flags = re.findall(r"\[FLAG: [^\]]*\]", flat)
print(f"         {len(flags)} author flags remain in the text (expected, listed at the end)")

print("2. Style")
check("—" not in flat, "no em dashes anywhere")
uk = re.findall(r"\b\w*(analys(e|ed|ing)\b|normalis\w*|colour\w*|centre\b|behaviour\w*|modelling|favour\w*|summaris\w*|recognis\w*|labelled|equalis\w*|randomis\w*|characteris(?:e|ed|es|ing|ation)\b|minimis\w*|optimis\w*)", body)
check(not uk, f"no UK spellings in the body ({[u[0] for u in uk][:8]})")
check(not re.search(r"p = <", flat), "never 'p = <'")
check(not re.search(r"p = 0\.000\b", flat), "never 'p = 0.000'")

print("3. Every fact used in the text appears in the built file")
used = set(re.findall(r"\{([A-Za-z][A-Za-z0-9_.]*)\}", tpl))
missing = []
for k in sorted(used):
    v = facts[k].replace("*", "")
    if v not in flat:
        missing.append((k, v))
check(not missing, f"{len(used)} facts found in the docx text" + (f"; missing {missing[:5]}" if missing else ""))

print("4. Literal numbers in the template (not facts) are all accounted for")
LITERALS = {
    # design constants of the pipeline and protocol
    "600": "onset skip (s) and movement-free segment (beats)", "1800": "analysis window (s)",
    "10": "minutes 10 to 40; ten minutes; 10 beats; 10 bins; 10,000", "40": "minutes 10 to 40",
    "0": "protocol walk at 0 min", "30": "per group; minutes; 30 s padding; x 30",
    "60": "60 s skip; 60 minutes; 60 subjects", "360": "360 s skip", "370": "duration-matched segment (s)",
    "300": "lower interval bound (ms)", "2000": "upper interval bound (ms)", "20": "20% filter; 20 s; 20 replicates; scales 1 to 20",
    "11": "11-beat median window", "5": "5% exclusion threshold", "240": "minimum clean intervals",
    "1/1024": "Bluetooth R-R unit", "1.000": "measured grid (ms)", "7.8125": "MIT-BIH grid (ms)",
    "7.8": "MIT-BIH grid, rounded", "128": "MIT-BIH sampling rate (Hz); Welch segment (s)",
    "4": "resampling (Hz); DFA bounds; line length 4", "512": "Welch segment (samples)", "50": "Welch overlap (%)",
    "0.0078": "Welch bin spacing (Hz)", "0.003": "nominal VLF lower bound (Hz)", "0.04": "VLF/LF boundary (Hz)",
    "0.039": "upper VLF bin (Hz)", "0.15": "LF/HF boundary (Hz); 0.15 x SD", "0.40": "HF upper bound (Hz)",
    "18": "Welch bins in the fitted range", "0.0156": "lowest fitted bin (Hz)", "0.148": "highest fitted bin (Hz)",
    "0.01": "nominal lower fit bound (Hz); shift threshold", "0.1": "baroreflex resonance (Hz)",
    "0.07": "excised band (Hz)", "0.13": "excised band (Hz)", "64": "DFA scale (beats)", "16": "DFA scale (beats)",
    "2": "embedding dimension; line length", "0.2": "entropy tolerance x SD", "1": "delay; scale; slope of 1/f; C = 1",
    "3": "permutation order; line length 3", "0.088": "matched target ratio", "0.10": "sweep target; tolerance",
    "0.20": "sweep target", "0.35": "tolerance", "8": "DFA lower bound (beats)", "100": "surrogates per subject",
    "0.05": "alpha", "2.5": "bout threshold multiple", "180": "padding after bouts (s)", "22": "classifier feature pool",
    "10,000": "classifier bootstrap replicates", "2,000": "fixed-beat length; r bootstrap replicates", "21": "measures in each family",
    "0.5": "equivalence margin (SD)", "7": "protocol walk length (min), Książek et al. 2023",
    "80": "power (%); published accuracy (%)", "12": "unused", "6": "unused",
    "0.016": "fitted range lower edge, rounded (Hz)", "0.047": "very-low band edge (Hz)", "0.062": "low band upper edge, 0.0625 Hz printed as in Table 4 (Hz)",
    "β": "symbol",
    # published figures (cited)
    "0.91": "Clamor et al. 2016", "24": "Clamor et al. 2016", "1.77": "Faurholt-Jepsen et al. 2017",
    "442": "Wang et al. 2025", "85": "Buza et al. 2023", "0.92": "Buza et al. 2023", "83": "Książek et al. 2025",
    # identifiers
    "2.4": "section reference", "2.6": "section reference", "3.3": "section reference", "3.4": "section reference",
    "3.5": "section reference", "135": "approval number", "52": "approval number", "23": "approval number; ICD code",
    "19": "ICD code F19", "13.2": "ICD code F13.2", "19.2": "ICD code F19.2", "31": "ICD code F31",
    "0009": "ORCID", "1721": "ORCID", "8640": "ORCID", "21886112": "DOI", "5281": "DOI", "13026": "DOI",
    "1.0": "licence version", "4.0": "licence version (CC BY 4.0)", "2011": "citation year", "2.0.0": "release version",
    "0.001": "p reporting rule (p < 0.001)", "95": "95% intervals", "0052": "approval number", "22/23": "approval number",
    "10.13026": "DOI prefix", "10.5281": "DOI prefix", "5.5": "model name inside an author flag",
}
lit_txt = re.sub(r"\{[^}]+\}", " ", tpl)
lit_txt = re.sub(r"\([A-Z][^()]*?\d{4}[a-z]?\)", " ", lit_txt)          # parenthetical citations
lit_txt = re.sub(r"\b(19|20)\d{2}\b", " ", lit_txt)                    # years
lit_txt = re.sub(r"^@.*$", " ", lit_txt, flags=re.M)                    # front matter
lit_txt = re.sub(r"^#+ \d(\.\d)? ", "# ", lit_txt, flags=re.M)          # section numbers
lit_txt = re.sub(r"\[\[(TABLE|FIGURE) \d\]\]", " ", lit_txt)
lit_txt = re.sub(r"\b(Table|Figure|Section|Sections) \d[A-B]?(\.\d)?", " ", lit_txt)
lit_txt = re.sub(r"^\d+\. ", "", lit_txt, flags=re.M)                   # list numbering
nums = [n.rstrip(",") for n in re.findall(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?(?:/\d+)?)(?![\w])", lit_txt)]
unknown = sorted({n for n in nums if n not in LITERALS and n.strip(".") not in LITERALS})
check(not unknown, f"every literal number is a declared constant, published figure or identifier ({len(set(nums))} distinct)"
      + (f"; undeclared: {unknown}" if unknown else ""))

print("5. Citations and references (Frontiers numbered style)")
ref_keys = []
for ln in open(REFS, encoding="utf-8"):
    m = re.match(r"^\[([^\]]+)\]", ln.strip())
    if m:
        ref_keys.append(m.group(1))
order_keys = json.load(open(SNUM.replace("v12_supplement_numbers.csv", "v12_references_keys.json"), encoding="utf-8"))
n_ref = len(ref_keys)
check(sorted(order_keys) == sorted(ref_keys), f"every one of the {n_ref} references is cited and numbered")
leftover = [m.group(0) for m in re.finditer(r"\b[A-Z][\w\-ąćęłńóśźż]+(?: et al\.| and [A-Z][\w\-]+)?(?:, | \()(?:19|20)\d{2}\b", body)
            if m.group(0).split()[0] not in ("Table", "Figure", "Version", "Python", "October", "August", "July", "Section")]
check(not leftover, "no author-year citation is left in the text" + (f"; found {leftover[:5]}" if leftover else ""))
# citation groups: numbers, commas and en-dash ranges inside parentheses, in the running text (tables hold
# standard deviations and beat ranges in parentheses, so table cells are left out; the builder puts no citation in a
# main-text table cell, and table notes are ordinary paragraphs and are included)
_js = json.loads(subprocess.run(["pandoc", DOCX, "-t", "json"], capture_output=True, text=True).stdout)
_js["blocks"] = [b for b in _js["blocks"] if b.get("t") != "Table"]
_nt = subprocess.run(["pandoc", "-f", "json", "-t", "plain", "--wrap=none"], input=json.dumps(_js),
                     capture_output=True, text=True).stdout
_nt = re.sub(r"\s+", " ", _nt)
body_nt = _nt[:_nt.find(" References ")]
GROUP = r"(?<![\w(])\((\d+(?:–\d+)?(?:, \d+(?:–\d+)?)*)\)"
first = []
bad_num = []
for m in re.finditer(GROUP, body_nt):
    nums = []
    for part in m.group(1).split(", "):
        a, _, b = part.partition("–")
        nums += list(range(int(a), int(b or a) + 1))
    if nums != sorted(nums) or len(nums) != len(set(nums)):
        bad_num.append(m.group(0))
    for n in nums:
        if not 1 <= n <= n_ref:
            bad_num.append(m.group(0))
        if n not in first:
            first.append(n)
check(not bad_num, "every citation number is between 1 and the number of references, sorted within its group"
      + (f"; found {bad_num[:5]}" if bad_num else ""))
check(first == list(range(1, n_ref + 1)), f"references are first cited in numerical order 1..{n_ref}"
      + ("" if first == list(range(1, n_ref + 1)) else f" (found {first[:15]}...)"))
groups = re.findall(GROUP, body_nt)
stats = json.load(open(SNUM.replace("v12_supplement_numbers.csv", "v12_citation_stats.json")))
n_written = stats["groups"] - stats["groups_in_cells"]
check(len(groups) == n_written, f"every number in parentheses in the running text is a citation the builder wrote "
      f"({len(groups)} found, {n_written} written outside table cells); a count in parentheses would read as a citation")
uncollapsed = [g for g in groups if re.search(r"(\d+), (\d+), (\d+)", g) and any(
    int(x) + 1 == int(y) and int(y) + 1 == int(z) for x, y, z in re.findall(r"(?=(\d+), (\d+), (\d+))", g))]
check(not uncollapsed, "runs of three or more consecutive numbers are collapsed with an en dash"
      + (f"; found {uncollapsed[:3]}" if uncollapsed else ""))
refs_txt = flat[body_end:] if body_end > 0 else ""
for marker in (" Tables Table 1.", " Figure captions Figure 1."):      # the Frontiers layout puts these after the list
    if marker in refs_txt:
        refs_txt = refs_txt[:refs_txt.find(marker)]
listed = [int(x) for x in re.findall(r"(?:^| )(\d{1,2})\. (?=(?:de |van )?[A-ZÀ-Ž])", refs_txt)]
check(listed == list(range(1, n_ref + 1)), f"the reference list is numbered 1..{n_ref} in order (found {len(listed)} entries)")
entries = re.split(r" (?=\d{1,2}\. (?:de |van )?[A-ZÀ-Ž])", refs_txt.replace(" References ", " ", 1).strip())
first_auth = [re.match(r"\d+\. (\S+)", e).group(1) for e in entries if re.match(r"\d+\. ", e)]
exp_auth = [k.rsplit(" ", 1)[0].split()[0] for k in order_keys]
check(first_auth[:n_ref] == exp_auth, "each numbered entry is the reference cited under that number"
      + ("" if first_auth[:n_ref] == exp_auth else f" (first mismatch at {next(i for i,(a,b) in enumerate(zip(first_auth, exp_auth)) if a != b) + 1})"))
over6 = [e[:60] for e in entries if len(re.findall(r"[A-ZÀ-Ž][\w\-ąćęłńóśźż' ]* [A-Z]{1,3}(?:h)?(?: Jr)?, ", e.split(". ")[1] if ". " in e else "")) > 6]
check(not over6, "no entry lists more than six authors before et al." + (f"; {over6[:3]}" if over6 else ""))

print("6. Exhibit and supplement numbering")
for kind, n in (("Figure", 4), ("Table", 5)):
    order = []
    for m in re.finditer(kind + r" (\d)", body):
        k = int(m.group(1))
        if k not in order:
            order.append(k)
    check(order[:n] == list(range(1, n + 1)), f"{kind}s first cited in order 1..{n} (found {order})")
snum = list(csv.DictReader(open(SNUM, encoding="utf-8")))


def cited_order(kind):
    """Order of first citation of Supplementary Tables or Figures, including grouped references
    ("Supplementary Tables S3 and S4", "S5, S6 and S7", "S8\u2013S10")."""
    order = []
    for m in re.finditer(r"Supplementary " + kind + r"s? (S\d+(?:(?:, | and |\u2013)S\d+)*)", body):
        grp = m.group(1)
        nums = [int(x) for x in re.findall(r"S(\d+)", grp)]
        if "\u2013" in grp:
            nums = list(range(nums[0], nums[-1] + 1))
        for k in nums:
            if k not in order:
                order.append(k)
    return order


for kind in ("Table", "Figure"):
    n_def = sum(1 for r in snum if r["Kind"] == kind)
    o = cited_order(kind)
    check(o == list(range(1, n_def + 1)), f"Supplementary {kind}s cited in order S1..S{n_def} (found {o[:6]}...)" if o != list(range(1, n_def + 1)) else f"Supplementary {kind}s cited in order S1..S{n_def}")
for kind, label in (("Methods", "Methods"), ("Results", "Results")):
    n_def = sum(1 for r in snum if r["Kind"] == kind)
    cited = sorted({int(x) for m in re.finditer(r"Supplementary " + label + r" (S\d+(?:(?:, | and )S\d+)*)", body) for x in re.findall(r"S(\d+)", m.group(1))})
    check(cited == list(range(1, n_def + 1)), f"every Supplementary {label} section exists and is cited (cited {cited} of {n_def})")
check("Supplementary Methods." not in body and "Supplementary Results." not in body and "(Supplementary Results)" not in body,
      "no unnumbered references to the Supplementary Methods or Results")

print("7. Stale v11 statements are absent")
STALE = ["matched on age and sex", "zenodo.8171266", "no symptom severity", "ratio of 0.085", "0.085 target", "0.977 ms", "610 and 671",
         "r = +0.304", "29/29", "58 of 60", "riticality", "was not run", "r = +0.349", "+0.349 to", "1,177", "Książek et al., 2026",
         "10,000 times", "same institution", "under the same protocol", "4 to 64 beats (full-range", "two reversed",
         "Two estimator choices", "excised (10 bins)", "convention in earlier HRV work", "remained indistinguishable",
         "pre-declared", "Bluetooth SIG", "fixed before group contrasts",
         "four subjects", "weakest effect of the three", "survive neither", "either movement-free"]
# the repository's current name still says "criticality"; it is a URL, not a claim, until the repository is renamed
flat_nourl = flat.replace("github.com/corbin-smith/ecg-criticality-psychiatry-pilot", "github.com/REPO")
present = [s for s in STALE if s in flat_nourl]
check(not present, "none of the known stale v11 strings occurs" + (f"; found {present}" if present else ""))

print("8. Exhibits")
for n, f in ((1, "Table1_Participants.md"), (2, "Table2_Primary.md"), (3, "Table3_Controls.md"),
             (4, "Table4_Estimators.md"), (5, "Table5_Robustness.md")):
    title = open(f"{TABLES}/{f}", encoding="utf-8").readline().strip().strip("*")
    check(title in flat, f"Table {n} title present: {title[:60]}")

print("9. Length")
abs_m = re.search(r"Abstract (.*?) Keywords:", flat)
abs_words = len([w for w in abs_m.group(1).split() if re.search(r"[A-Za-z0-9]", w)]) if abs_m else -1
check(0 < abs_words <= 300, f"abstract is {abs_words} words (limit 300)")

print("\nAuthor flags still open:")
for fl in flags:
    print("   -", fl[:150])
print(f"\n{len(fails)} check(s) failed")
sys.exit(1 if fails else 0)
