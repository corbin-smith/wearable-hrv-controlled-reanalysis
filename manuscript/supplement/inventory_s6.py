"""
inventory_s6.py: account for every analysis output, figure and v5 supplement item.

Each output file under outputs/ (the URTC talk folder excepted) is classified as
  Main text      read by the main draft (facts registry, exhibits manifest or a main figure)
  Supplement     read by a Supplementary Table, a supplement fact or shown as a Supplementary Figure
  Both
  Excluded       with the reason given in EXCLUDED below
and the script fails if any file is unclassified, or if an exclusion names a file that is in fact used.
Every v5 supplement item is mapped to its v6 location.

Usage: python inventory_s6.py <project folder> <v12 build folder> <s6 build folder> <exhibits manifest> <out.csv>
"""
import csv
import json
import os
import re
import sys

project, mbuild, sbuild, manifest, out = sys.argv[1:6]
OUT = os.path.join(project, "outputs")

files = []
for root, _, names in os.walk(OUT):
    if "urtc_talk" in root:
        continue
    for n in names:
        files.append(os.path.relpath(os.path.join(root, n), project).replace(os.sep, "/"))
files.sort()

# ------------------------------------------------------------------ what the main draft reads
main = {}
for r in csv.DictReader(open(os.path.join(mbuild, "facts", "v12_facts_registry.csv"), encoding="utf-8")):
    for s in re.split(r"\s\+\s|;", r["source"]):
        s = s.strip()
        if s.startswith("outputs/"):
            main.setdefault(s, set()).add("text")
for r in csv.DictReader(open(manifest, encoding="utf-8")):
    main.setdefault(r["Path"], set()).add("Tables 1 to 5 and Figures 1 and 3 (exhibits_v12.py)")
for f in ("outputs/exhibits_v12/Figure1_HeartRate.png", "outputs/figures/Figure3_PSD_Curves.png",
          "outputs/exhibits_v12/Figure3_Resolution.png", "outputs/figures/Figure8_CrossValidation.png"):
    main.setdefault(f, set()).add("main figure")

# ------------------------------------------------------------------ what the supplement reads
order = json.load(open(os.path.join(mbuild, "build", "v12_supplement_order.json"), encoding="utf-8"))
tables = json.load(open(os.path.join(sbuild, "s6_tables.json"), encoding="utf-8"))
supp = {}
for name, t in tables.items():
    for s in t["sources"]:
        supp.setdefault(s, set()).add(f"Table {t['number']}")
FIGS = {"effects": "Figure1_EffectSizes", "boxplots": "Figure2_BoxPlots", "intercorr": "Figure7_Intercorrelation",
        "mse": "FigureS_MSE_Curves", "surrogates": "Figure5_Surrogates", "identity": "FigureS_BetaAlphaIdentity",
        "mayer": "FigureM_MayerDiagnostic", "covariate": "Figure4_Covariate", "classifier": "Figure6_Classifier"}
for k, name in enumerate(order["figures"], 1):
    for ext in ("png", "svg"):
        supp.setdefault(f"outputs/figures/{FIGS[name]}.{ext}", set()).add(f"Figure S{k}")
FACT_READS = {   # files read by s6_facts.py for statements in the supplement text
    "outputs/healthy_hrvacc/ecg_results_HealthyACC.csv": "Methods and Results text",
    "outputs/psychiatric/ecg_results_Psychiatric.csv": "Methods and Results text",
    "outputs/healthy/ecg_results_Healthy.csv": "Methods S1 and S2",
    "outputs/comparison/healthy_hrvacc_vs_psych/Metric_Intercorrelation_HighPairs.csv": "Results S2",
    "outputs/comparison/healthy_hrvacc_vs_psych/Length_Respiration_Checks.csv": "Results S2 and S4",
    "outputs/stationary/Stationary_Longest_Runs.csv": "Results S3",
    "outputs/comparison/healthy_hrvacc_vs_psych/Quantization_Noise_Floor.csv": "Methods S7",
}
for f, where in FACT_READS.items():
    supp.setdefault(f, set()).add(where)

# ------------------------------------------------------------------ deliberate exclusions
EXCLUDED = {
    "outputs/comparison/TABLE1_MANUSCRIPT.md": "Draft of an earlier Table 1, superseded by exhibits_v12/Table1_Participants",
    "outputs/comparison/healthy_hrvacc_vs_psych/TABLE1_MANUSCRIPT.md": "Draft of an earlier Table 1, superseded by exhibits_v12/Table1_Participants",
    "outputs/comparison/healthy_hrvacc_vs_psych/Bootstrap_Effect_Sizes.csv": "Its r intervals equal those in group_comparison_stats.csv (main Table 2, Table S8); its d intervals are not reported",
    "outputs/comparison/healthy_hrvacc_vs_psych/Partial_Correlation_Method.csv": "Documents the regress-then-rank method, stated in Supplementary Methods S12; no result",
    "outputs/comparison/healthy_hrvacc_vs_psych/Surrogate_Nonlinearity_Summary.csv": "Its counts of subjects departing from their surrogates are those of Surrogate_Distribution_Summary.csv (Table S23)",
    "outputs/figures/Figure6_Intercorrelation.png": "Retired duplicate of Figure7_Intercorrelation (Figure S3)",
    "outputs/figures/Figure6_Intercorrelation.svg": "Retired duplicate of Figure7_Intercorrelation (Figure S3)",
    "outputs/figures/Figure9_ConceptualSynthesis.png": "Retired schematic of an earlier framing; not part of v12",
    "outputs/figures/Figure9_ConceptualSynthesis.svg": "Retired schematic of an earlier framing; not part of v12",
    "outputs/exhibits_v12/FIGURE_CAPTIONS_v12.md": "Caption drafts; the captions are in the main draft",
    "outputs/exhibits_v12/EXHIBITS_MANIFEST.csv": "Provenance record of the main exhibits (input MD5s)",
    "outputs/accelerometer/Activity_Structure.csv": "Per-subject activity quantities; their group comparison is Activity_Structure_Comparison.csv (Table S31)",
    "outputs/accelerometer/Stationary_Segments.csv": "Per-subject walking and movement-free beat counts, an intermediate of stationary_reanalysis.py (Table S32) and Ambulatory_Fraction_Test.csv (Table S31)",
}
PATTERNS = [
    (r"_pre_[a-z_]+\.csv$", "Snapshot taken before a later patch (pre-metadata, pre-mobility, pre-respiratory or pre-RNA); the patched file is the one analyzed"),
    (r"^outputs/comparison/healthy_mitbih_vs_psych/", "Patients against the MIT-BIH controls; the files used are group_comparison_stats.csv (Table S34) and the cross_control folder. The rest repeat the HRV-ACC analyses with the other control cohort and are not reported separately"),
    (r"requirements\.txt$", "Software environment; versions are reported from Environment.csv (Table S11)"),

    (r"Environment_[A-Za-z]+\.csv$", "Per-cohort copy of the environment record; the build checks it equals Environment.csv (Table S11)"),
]
# main-text exhibits that the build reads from the exhibits folder (or that are copies of main figures)
MAIN_PATTERNS = [
    (r"^outputs/exhibits_v12/Table(\d)_\w+\.(csv|md)$", lambda m: f"main Table {m.group(1)} as built by exhibits_v12.py"),
    (r"^outputs/exhibits_v12/Figure(\d)_source\.csv$", lambda m: f"data behind main Figure {m.group(1)}"),
    (r"^outputs/exhibits_v12/Figure(\d)_\w+\.(svg|tif)$", lambda m: f"vector or print copy of main Figure {m.group(1)}"),
    (r"^outputs/figures/Figure3_PSD_Curves\.svg$", lambda m: "vector copy of main Figure 2"),
    (r"^outputs/figures/Figure8_CrossValidation\.svg$", lambda m: "vector copy of main Figure 4"),
]
MAIN_ONLY_NOTE = {
    "outputs/comparison/healthy_hrvacc_vs_psych/Patient_Cohort_Description.csv": "main Table 1",
    "outputs/comparison/healthy_hrvacc_vs_psych/Demographic_Matching.csv": "main Table 1",
}

rows = []
unclassified = []
for f in files:
    m, s = main.get(f), supp.get(f)
    for pat, fn in MAIN_PATTERNS:
        mm = re.search(pat, f)
        if mm:
            m = (m or set()) | {fn(mm)}
    reason = EXCLUDED.get(f)
    if reason is None:
        for pat, why in PATTERNS:
            if re.search(pat, f):
                reason = why
                break
    if m and s:
        status, where = "Both", "; ".join(sorted(m | s))
    elif m:
        status, where = "Main text", MAIN_ONLY_NOTE.get(f, "; ".join(sorted(m)))
    elif s:
        status, where = "Supplement", "; ".join(sorted(s, key=lambda x: (len(x), x)))
    elif reason:
        status, where = "Excluded", reason
    else:
        status, where = "UNCLASSIFIED", ""
        unclassified.append(f)
    if (m or s) and f in EXCLUDED:
        raise SystemExit(f"{f} is listed as excluded but is used ({m or s})")
    rows.append(["Output file", f, status, where])

# ------------------------------------------------------------------ v5 supplement items
SNUM = {name: f"S{k}" for k, name in enumerate(order["tables"], 1)}
SF = {name: f"S{k}" for k, name in enumerate(order["figures"], 1)}
V5 = [
    ("Methods S1", "Spectral exponent estimators", "Supplementary Methods S3"),
    ("Methods S2", "Recurrence threshold selection", "Supplementary Methods S6"),
    ("Methods S3", "Choice of statistical moments for MFDFA", "Supplementary Methods S4"),
    ("Methods S4", "Selection of the analysis window", "Supplementary Methods S1"),
    ("Methods S5", "Classification procedure", "Supplementary Methods S13"),
    ("Methods S6", "Ambulation detection", "Supplementary Methods S9"),
    ("Results S1", "Respiratory frequency", "Supplementary Results S1"),
    ("Results S2", "Metric covariance structure", "Supplementary Results S2"),
    ("Table S1", "Quantization control", f"Table {SNUM['quant']}"),
    ("Table S2", "21-metric primary family", f"Main Table 2; Table {SNUM['primaryx']}"),
    ("Table S3", "Excluding definitionally collinear metrics", f"Table {SNUM['nonredundant']}"),
    ("Table S4", "Distribution diagnostics", f"Table {SNUM['primaryx']} (Shapiro-Wilk and Levene columns)"),
    ("Table S5", "Welch t, Cohen's d and Hedges' g", f"Table {SNUM['primaryx']} (Welch, d and g columns)"),
    ("Table S6", "Intercorrelations, controls", f"Table {SNUM['intercorr']} (lower triangle); Figure {SF['intercorr']}"),
    ("Table S7", "Intercorrelations, patients", f"Table {SNUM['intercorr']} (upper triangle); Figure {SF['intercorr']}"),
    ("Table S8", "Control cohorts compared", f"Table {SNUM['controls']}"),
    ("Table S9", "Cross-cohort convergence", f"Table {SNUM['crosscohort']}"),
    ("Table S10", "Three-group Dunn post hoc", f"Table {SNUM['threegroup']}"),
    ("Table S11", "Three-group Kruskal-Wallis", f"Table {SNUM['threegroup']}"),
    ("Table S12", "Equivalence tests of the control cohorts", f"Table {SNUM['controls']} (lower block)"),
    ("Table S13", "Multifractal DFA", f"Table {SNUM['mfdfa']}"),
    ("Table S14", "Sample entropy tolerance", f"Table {SNUM['tolerance']}"),
    ("Table S15", "Window-skip sensitivity", f"Table {SNUM['skip']}"),
    ("Table S16", "Classification summary", f"Table {SNUM['classifier']}"),
    ("Table S17", "Recurrence quantification at two embeddings", f"Table {SNUM['embedding']}"),
    ("Table S18", "Multiscale entropy scale audit", f"Table {SNUM['mse']}"),
    ("Table S19", "Per-subject IAAFT surrogate tests", f"Table {SNUM['surrogate']} (lower block)"),
    ("Table S20", "Surrogate-normalized entropy family", f"Main Table 4; Table {SNUM['surrogate']} (upper block)"),
    ("Table S21", "Correction family inventory", f"Table {SNUM['families']}"),
    ("Table S22", "Effective number of tests", f"Table {SNUM['families']} (note)"),
    ("Table S23", "Accelerometer specification", f"Table {SNUM['accel']}"),
    ("Table S24", "Clinical-range counts", f"Table {SNUM['clinical']}"),
    ("Table S25", "Minimum detectable effects", f"Table {SNUM['power']}"),
    ("Table S26", "Diagnostic subgroups", f"Table {SNUM['subgroups']}"),
    ("Table S27", "Quetiapine sensitivity", f"Table {SNUM['subgroups']}"),
    ("Table S28", "Substance-use codes", f"Table {SNUM['subgroups']}"),
    ("Table S29", "Test implementation and environment", f"Table {SNUM['environment']}"),
    ("Table S30", "Cross-cohort comparisons with age", f"Table {SNUM['crosscohort']} (lower block)"),
    ("Figure S1", "Intercorrelation heatmaps", f"Figure {SF['intercorr']}"),
    ("Figure S2", "Slope against 2α₂ − 1", f"Figure {SF['identity']}"),
    ("Figure S3", "Nested classifier", f"Figure {SF['classifier']}"),
    ("Figure S4", "Multiscale entropy by scale", f"Figure {SF['mse']}"),
]
for item, title, where in V5:
    rows.append([f"v5 {item}", title, "Carried into v6", where])

with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["Kind", "Item", "Status", "Where it appears, or why it is excluded"])
    w.writerows(rows)

from collections import Counter
print(Counter(r[2] for r in rows if r[0] == "Output file"))
if unclassified:
    print("UNCLASSIFIED:")
    for f in unclassified:
        print("  ", f)
    sys.exit(1)
print(f"wrote {out}: {len(rows)} rows")
