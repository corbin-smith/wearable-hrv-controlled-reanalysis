"""
v12_facts.py: every number the v12 manuscript text uses, read or computed from the rerun files.

Usage:  python v12_facts.py <outputs_dir> <out_dir>

Writes  <out_dir>/v12_facts.json           key -> formatted string (what the text prints)
        <out_dir>/v12_facts_registry.csv   key, printed text, raw value, source file, how it was obtained

Rules this script follows (project discipline):
  * No number is typed in. Each one is read from a results file, or computed from the
    per-subject files with the same test the pipeline uses, and printed as it is stored.
  * Where a statement in the text depends on a pattern in the numbers (for example "every
    amplitude measure stayed significant under every model form"), the pattern is asserted
    here, so the script stops if the data no longer support the sentence.
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

OUT_DIR = sys.argv[1]
DEST = sys.argv[2]
CMP = os.path.join(OUT_DIR, "comparison", "healthy_hrvacc_vs_psych")
XC = os.path.join(OUT_DIR, "comparison", "cross_control")
MINUS = "−"

FACTS = {}
REGISTRY = []


# --------------------------------------------------------------------------- formatting
def _neg(s):
    return s.replace("-", MINUS)


def R(x, nd=3):
    """Signed effect size: +0.667 or −0.809."""
    return _neg(f"{x:+.{nd}f}")


def U(x, nd=3):
    """Unsigned number with fixed decimals (minus sign if negative)."""
    return _neg(f"{x:.{nd}f}")


def PE(p):
    """'*p* = 0.012' or '*p* < 0.001' (three decimals, as in the tables)."""
    return "*p* < 0.001" if p < 0.001 else f"*p* = {p:.3f}"


def INT(x):
    # half up, the conventional rule (Python's round() sends 994.5 to 994)
    from decimal import Decimal, ROUND_HALF_UP
    return f"{int(Decimal(repr(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}"


def add(key, text, raw, source, how=""):
    if key in FACTS:
        raise SystemExit(f"duplicate fact key {key}")
    FACTS[key] = text
    # Record sources relative to the project folder (outputs/...), so the registry reads the same on any machine.
    root = os.path.dirname(os.path.abspath(OUT_DIR))
    rel = " + ".join(os.path.relpath(s.strip(), root).replace(os.sep, "/") if os.path.isabs(s.strip()) else s.strip()
                     for s in str(source).split(" + "))
    REGISTRY.append([key, text, raw, rel, how])
    print(f"  {key:<28} {text:<34} raw={raw!s:<26} [{os.path.basename(source)}] {how}")


def csvp(*parts):
    p = os.path.join(*parts)
    if not os.path.exists(p):
        raise SystemExit(f"missing input {p}")
    return p


def mw(a, b):
    """Rank-biserial r (positive = higher in the second sample, patients) and p, as the pipeline:
    Mann-Whitney U for the first sample, continuity correction, method 'auto'."""
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    u, p = mannwhitneyu(a, b, alternative="two-sided", use_continuity=True, method="auto")
    return 1.0 - 2.0 * u / (len(a) * len(b)), p, len(a), len(b)


# --------------------------------------------------------------------------- per-subject files
def subjects(path, prefix=None):
    d = pd.read_csv(path, index_col=0)
    if prefix:
        d = d[d.index.astype(str).str.startswith(prefix)]
    else:  # MIT-BIH: record numbers only
        d = d[d.index.astype(str).str.fullmatch(r"\d+")]
    return d


P_C = csvp(OUT_DIR, "healthy_hrvacc", "ecg_results_HealthyACC.csv")
P_P = csvp(OUT_DIR, "psychiatric", "ecg_results_Psychiatric.csv")
P_M = csvp(OUT_DIR, "healthy", "ecg_results_Healthy.csv")
C = subjects(P_C, "control_")
P = subjects(P_P, "treatment_")
M = subjects(P_M)
assert len(C) == 30 and len(P) == 30 and len(M) == 18, (len(C), len(P), len(M))

print("Cohorts")
add("n_ctl", str(len(C)), len(C), P_C, "rows control_*")
add("n_pat", str(len(P)), len(P), P_P, "rows treatment_*")
add("n_mit", str(len(M)), len(M), P_M, "numeric record rows")
pc = pd.read_csv(csvp(CMP, "Patient_Cohort_Description.csv")).set_index("Group")
src_pc = os.path.join(CMP, "Patient_Cohort_Description.csv")
add("n_sz", str(int(pc.loc["Schizophrenia (F20)", "N"])), pc.loc["Schizophrenia (F20)", "N"], src_pc)
add("n_bp", str(int(pc.loc["Bipolar (F31)", "N"])), pc.loc["Bipolar (F31)", "N"], src_pc)
add("cohort.quet", str(int(pc.loc["All patients", "N_quetiapine"])), pc.loc["All patients", "N_quetiapine"], src_pc)
add("cohort.quet_bp", str(int(pc.loc["Bipolar (F31)", "N_quetiapine"])), pc.loc["Bipolar (F31)", "N_quetiapine"], src_pc)
add("cohort.quet_sz", str(int(pc.loc["Schizophrenia (F20)", "N_quetiapine"])), pc.loc["Schizophrenia (F20)", "N_quetiapine"], src_pc)
add("cohort.comorb", str(int(pc.loc["All patients", "N_comorbid_substance"])), pc.loc["All patients", "N_comorbid_substance"], src_pc)
assert int(pc.loc["Bipolar (F31)", "N_comorbid_substance"]) == 0, "text says all comorbid patients have schizophrenia"
assert int(pc.loc["Bipolar (F31)", "N_female"]) == int(pc.loc["Bipolar (F31)", "N"]), "text says all bipolar patients are women"
q = pd.to_numeric(P["Quetiapine"], errors="coerce")
nonq = int((q == 0).sum())
assert nonq + int((q == 1).sum()) == 30
add("cohort.nonquet", str(nonq), nonq, P_P, "Quetiapine == 0")

print("Age and sex")
dm = pd.read_csv(csvp(CMP, "Demographic_Matching.csv")).set_index("Variable")
src_dm = os.path.join(CMP, "Demographic_Matching.csv")
add("age.c_mean", U(dm.loc["Age", "Control"], 1), dm.loc["Age", "Control"], src_dm)
add("age.p_mean", U(dm.loc["Age", "Patient"], 1), dm.loc["Age", "Patient"], src_dm)
add("age.c_sd", U(dm.loc["Age", "Control_SD"], 1), dm.loc["Age", "Control_SD"], src_dm)
add("age.p_sd", U(dm.loc["Age", "Patient_SD"], 1), dm.loc["Age", "Patient_SD"], src_dm)
add("age.r", R(dm.loc["Age", "r_effect"]), dm.loc["Age", "r_effect"], src_dm)
add("age.pe", PE(dm.loc["Age", "p_value"]), dm.loc["Age", "p_value"], src_dm)
add("age.sex_pe", PE(dm.loc["Sex (N female)", "p_value"]), dm.loc["Sex (N female)", "p_value"], src_dm)
for g, D, src in (("c", C, P_C), ("p", P, P_P), ("m", M, P_M)):
    a = pd.to_numeric(D["Age"], errors="coerce")
    add(f"age.{g}_min", INT(a.min()), a.min(), src, "min Age")
    add(f"age.{g}_max", INT(a.max()), a.max(), src, "max Age")
    sx = D["Sex"].astype(str).str.upper().str[0]
    add(f"age.{g}_female", str(int((sx == "F").sum())), int((sx == "F").sum()), src, "Sex starts with F")
assert int(dm.loc["Sex (N female)", "Control"]) == int(FACTS["age.c_female"])
assert int(dm.loc["Sex (N female)", "Patient"]) == int(FACTS["age.p_female"])


def ids(D, prefix):
    n = sorted(int(str(i).replace(prefix, "")) for i in D.index)
    return ", ".join(str(x) for x in n)


add("ids.ctl", ids(C, "control_"), len(C), P_C, "record numbers")
add("ids.pat", ids(P, "treatment_"), len(P), P_P, "record numbers")

print("Quality control")
for g, D, src in (("c", C, P_C), ("p", P, P_P), ("m", M, P_M)):
    v = pd.to_numeric(D["Artifact_Percent"], errors="coerce")
    add(f"qc.art_{g}", U(v.mean(), 3), v.mean(), src, "mean Artifact_Percent")
allart = pd.concat([pd.to_numeric(D["Artifact_Percent"], errors="coerce") for D in (C, P, M)])
add("qc.art_max", U(allart.max(), 1), allart.max(), "per-subject files", "max Artifact_Percent, 78 records")
assert allart.max() < 5
nrr = pd.concat([pd.to_numeric(D["Num_RR"], errors="coerce") for D in (C, P, M)])
add("qc.n_min", INT(nrr.min()), nrr.min(), "per-subject files", "min Num_RR, 78 records")
assert nrr.min() >= 240
add("qc.n_all", str(len(C) + len(P) + len(M)), len(C) + len(P) + len(M), "per-subject files")

print("Theiler window and line length")
add("rqa.theiler_c", INT(pd.to_numeric(C["RQA_Theiler_W"]).median()), pd.to_numeric(C["RQA_Theiler_W"]).median(), P_C, "median RQA_Theiler_W")
add("rqa.theiler_p", INT(pd.to_numeric(P["RQA_Theiler_W"]).median()), pd.to_numeric(P["RQA_Theiler_W"]).median(), P_P, "median RQA_Theiler_W")
for k, col in (("lmin3", "RQA_DET_lmin3"), ("lmin4", "RQA_DET_lmin4")):
    r, p, _, _ = mw(C[col], P[col])
    add(f"rqa.{k}", R(r), r, "per-subject files", f"Mann-Whitney on {col} (p = {p:.4f})")
    assert r > float(pd.read_csv(csvp(CMP, "Embedding_Sensitivity.csv")).query("Quantity=='RQA_DET' and Type=='primary_embedding'")["r_effect"].iloc[0])

print("Correlation of approximate and sample entropy")
for g, D, src in (("c", C, P_C), ("p", P, P_P)):
    rho = spearmanr(pd.to_numeric(D["ApEn"]), pd.to_numeric(D["SampEn"]))[0]
    add(f"cor.apen_sampen_{g}", U(rho, 3), rho, src, "Spearman ApEn vs SampEn within group")

print("Minimum detectable effect")
mde = pd.read_csv(csvp(CMP, "Minimum_Detectable_Effect.csv"))
row = mde[mde.Power == 0.8].iloc[0]
add("mde.d", U(row.Min_Detectable_d), row.Min_Detectable_d, os.path.join(CMP, "Minimum_Detectable_Effect.csv"))
add("mde.r", U(row.Min_Detectable_r), row.Min_Detectable_r, os.path.join(CMP, "Minimum_Detectable_Effect.csv"))

print("Environment")
env = pd.read_csv(csvp(OUT_DIR, "healthy_hrvacc", "Environment.csv")).set_index("Item")["Value"]
src_env = os.path.join(OUT_DIR, "healthy_hrvacc", "Environment.csv")
for k, item in (("python", "python"), ("numpy", "version_numpy"), ("scipy", "version_scipy"),
                ("pandas", "version_pandas"), ("sklearn", "version_sklearn"),
                ("statsmodels", "version_statsmodels"), ("neurokit2", "version_neurokit2")):
    add(f"env.{k}", str(env[item]), env[item], src_env)
for other in ("psychiatric", "healthy"):
    e2 = pd.read_csv(csvp(OUT_DIR, other, "Environment.csv")).set_index("Item")["Value"]
    for item in ("python", "version_numpy", "version_scipy", "version_pandas", "version_sklearn", "version_neurokit2"):
        assert str(e2[item]) == str(env[item]), f"environment differs between cohorts: {item}"

print("Primary family (group_comparison_stats.csv)")
gcs_src = os.path.join(CMP, "group_comparison_stats.csv")
gcs = pd.read_csv(csvp(gcs_src))
gcs = gcs.rename(columns={gcs.columns[0]: "Label"}).set_index("Label")
GL = {"MeanHR": "Mean HR (bpm)", "MeanNN": "Mean NN (ms)", "RMSSD": "RMSSD (ms)", "SDNN": "SDNN (ms)",
      "LF": "LF Power (ms²)", "HF": "HF Power (ms²)", "TP": "Total Power (ms²)", "LFHF": "LF/HF Ratio",
      "BETA": "PSD β", "SampEn": "Sample Entropy", "SD1SD2": "SD1/SD2", "DET": "RQA Determinism",
      "LAM": "RQA Laminarity", "A1": "DFA α₁ (4–16)", "PermEn": "Permutation Entropy"}
for k, lab in GL.items():
    rw = gcs.loc[lab]
    add(f"g.{k}.r", R(rw.r_effect), rw.r_effect, gcs_src, lab)
    add(f"g.{k}.pae", PE(float(str(rw.p_adj).replace("<", ""))) if not str(rw.p_adj).startswith("<") else "*p* < 0.001",
        rw.p_adj, gcs_src, lab + " adjusted")
for k in ("LF", "HF", "TP", "RMSSD"):
    v = float(gcs.loc[GL[k], "Pct_Diff"])
    add(f"g.{k}.pctabs", INT(abs(v)), v, gcs_src, "median % difference")
wt_src = os.path.join(CMP, "Parametric_Comparison_Welch_t.csv")
wt = pd.read_csv(csvp(wt_src))
wt = wt.rename(columns={wt.columns[0]: "Label"}).set_index("Label")
g_rm = float(wt.loc["RMSSD (ms)", "Hedges_g"])
add("g.RMSSD.hg", R(g_rm, 2), g_rm, wt_src, "Hedges' g, RMSSD (as in Supplementary Table S9)")
assert float(wt.loc["RMSSD (ms)", "d_CI_high"]) < -0.91      # 'larger than the pooled estimate of -0.91 in schizophrenia'
mh = gcs.loc[GL["MeanHR"]]
add("g.MeanHR.mc", U(mh.Healthy_Median, 1), mh.Healthy_Median, gcs_src)
add("g.MeanHR.mp", U(mh.Psych_Median, 1), mh.Psych_Median, gcs_src)
add("g.MeanHR.pct", _neg(f"{float(mh.Pct_Diff):+.1f}%"), mh.Pct_Diff, gcs_src)
add("g.MeanHR.diff_med", INT(float(mh.Psych_Median) - float(mh.Healthy_Median)),
    float(mh.Psych_Median) - float(mh.Healthy_Median), gcs_src, "difference of medians, bpm")
sig = gcs[gcs.Significant.astype(str) == "True"]
assert len(gcs) == 21
add("g.n_sig", str(len(sig)), len(sig), gcs_src, "Significant == True")
add("g.sig_min", U(sig.r_effect.abs().min()), sig.r_effect.abs().min(), gcs_src, "min |r| among significant")
add("g.sig_max", U(sig.r_effect.abs().max()), sig.r_effect.abs().max(), gcs_src, "max |r| among significant")
mde_r = float(row.Min_Detectable_r)
below_sig = sorted(sig.index[sig.r_effect.abs() < mde_r])
nonsig = gcs[gcs.Significant.astype(str) != "True"]
assert set(below_sig) == {"Sample Entropy", "SD1/SD2"}, below_sig
assert (nonsig.r_effect.abs() < mde_r).all() and len(nonsig) == 4

print("pNN50 floor")
pf = pd.read_csv(csvp(CMP, "pNN50_Floor_Report.csv")).set_index("Group")
src_pf = os.path.join(CMP, "pNN50_Floor_Report.csv")
add("pnn.zero_p", str(int(pf.loc["Psychiatric", "N_At_Zero"])), pf.loc["Psychiatric", "N_At_Zero"], src_pf)
add("pnn.lt1_p", str(int(pf.loc["Psychiatric", "N_Below_1pct"])), pf.loc["Psychiatric", "N_Below_1pct"], src_pf)
zero_c = int(pf.loc["Healthy", "N_At_Zero"])
add("pnn.zero_c", "no" if zero_c == 0 else str(zero_c), zero_c, src_pf)

print("Heart rate: beat count identity")
both = pd.concat([C, P])
nb, hrv = pd.to_numeric(both["Num_RR"]), pd.to_numeric(both["HRV_MeanHR"])
r_all = pearsonr(nb, hrv)[0]
add("hrid.r_all", U(r_all, 4), r_all, "per-subject files", "Pearson Num_RR vs HRV_MeanHR, 60 recordings")
for g, D, src in (("c", C, P_C), ("p", P, P_P)):
    rr = pearsonr(pd.to_numeric(D["Num_RR"]), pd.to_numeric(D["HRV_MeanHR"]))[0]
    add(f"hrid.r_{g}", U(rr, 4), rr, src, "Pearson within group")
lrc = pd.read_csv(csvp(CMP, "Length_Respiration_Checks.csv")).set_index("Variable")
src_lrc = os.path.join(CMP, "Length_Respiration_Checks.csv")
add("hrid.beats_c", INT(lrc.loc["Num_RR", "Healthy_Mean"]), lrc.loc["Num_RR", "Healthy_Mean"], src_lrc)
add("hrid.beats_p", INT(lrc.loc["Num_RR", "Psych_Mean"]), lrc.loc["Num_RR", "Psych_Mean"], src_lrc)

hr_src = os.path.join(CMP, "HR_Adjusted_Comparison.csv")
hr = pd.read_csv(csvp(hr_src)).set_index("Metric")
assert len(hr) == 21
maxdiff = float((hr.r_partial_beats - hr.r_partial_HR).abs().max())
add("hrid.maxdiff", U(maxdiff), maxdiff, hr_src, "max |r_partial_beats - r_partial_HR| over 21 measures")

print("Heart rate: adjusted family")
HK = {"RMSSD": "HRV_RMSSD", "SDNN": "HRV_SDNN", "LF": "LF_Power_ms2", "HF": "HF_Power_ms2",
      "VLF": "VLF_Power_ms2", "TP": "Total_Power_ms2", "VLFLF": "VLF_LF_Ratio", "LFHF": "LF_HF_Ratio",
      "BETAX": "PSD_Beta_NoMayer", "BETA": "PSD_Beta", "A2": "DFA_Alpha2", "A": "DFA_Alpha",
      "A1": "DFA_Alpha1", "MSE": "MSE_Mean", "SampEn": "SampEn", "ApEn": "ApEn", "PermEn": "PermEn",
      "DET": "RQA_DET", "LAM": "RQA_LAM", "CLU": "RNA_ClusteringCoeff", "TRA": "RNA_Transitivity"}
for k, m in HK.items():
    rw = hr.loc[m]
    add(f"hr.{k}.u", R(rw.r_partial_unadjusted), rw.r_partial_unadjusted, hr_src, m)
    add(f"hr.{k}.a", R(rw.r_partial_HR), rw.r_partial_HR, hr_src, m)
    add(f"hr.{k}.pe", PE(rw.p_raw), rw.p_raw, hr_src, m)
    add(f"hr.{k}.pae", PE(rw.p_adj), rw.p_adj, hr_src, m)
AMP = ["HRV_RMSSD", "HRV_SDNN", "LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2", "Total_Power_ms2"]
pct = hr.loc[AMP, "pct_retained"]
add("hr.amp_pct_min", INT(pct.min()), pct.min(), hr_src, "min pct_retained, amplitude")
add("hr.amp_pct_max", INT(pct.max()), pct.max(), hr_src, "max pct_retained, amplitude")
FORMS_R = ["r_partial_HR", "r_HR_quadratic", "r_HR_log_measure", "r_HR_log_log_meanNN", "r_HR_rank_based"]
FORMS_P = ["p_raw", "p_HR_quadratic", "p_HR_log_measure", "p_HR_log_log_meanNN", "p_HR_rank_based"]
rm = hr.loc["HRV_RMSSD", FORMS_R].astype(float)
add("hr.RMSSD.form_lo", R(rm.max()), rm.max(), hr_src, "RMSSD weakest form (least negative)")
add("hr.RMSSD.form_hi", R(rm.min()), rm.min(), hr_src, "RMSSD strongest form")
pmax = float(hr.loc[AMP, FORMS_P].astype(float).values.max())
assert pmax < 0.05
add("hr.amp_form_pmax", U(pmax), pmax, hr_src, "max p over amplitude measures x 5 forms")
nsig = int((hr.Significant.astype(str) == "True").sum())
add("hr.n_sig", str(nsig), nsig, hr_src, "Significant in hr_adjusted family")
SIX = ["SampEn", "ApEn", "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"]
assert set(hr.index[hr.Significant.astype(str) != "True"]) == set(SIX)
add("hr.six_padj_min", U(hr.loc[SIX, "p_adj"].min()), hr.loc[SIX, "p_adj"].min(), hr_src)
add("hr.six_padj_max", U(hr.loc[SIX, "p_adj"].max()), hr.loc[SIX, "p_adj"].max(), hr_src)
for m in ["VLF_LF_Ratio", "PSD_Beta_NoMayer", "PSD_Beta", "DFA_Alpha2"]:  # "did not weaken"
    assert abs(hr.loc[m, "r_partial_HR"]) > abs(hr.loc[m, "r_partial_unadjusted"]), m
for m in ["LF_HF_Ratio", "DFA_Alpha1"]:  # "appeared only after adjustment"
    assert hr.loc[m, "p_partial_unadjusted"] > 0.05 and hr.loc[m, "p_adj"] < 0.05, m

print("Time of day and transmission gaps (recording_times_and_gaps.py)")
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
tg_src = os.path.join(CMP, "Recording_Times_Gaps.csv")
tg = pd.read_csv(csvp(tg_src))
tgs_src = os.path.join(CMP, "Recording_Times_Gaps_Summary.csv")
tgs = pd.read_csv(csvp(tgs_src)).set_index("Row")
ges_src = os.path.join(CMP, "Gap_Exclusion_Sensitivity.csv")
ges = pd.read_csv(csvp(ges_src))
assert len(tg) == 60 and len(ges) == 42
for g_, k_ in (("control", "c"), ("patient", "p")):
    add(f"tod.{k_}_lo", tgs.loc[g_, "Earliest_Start"], tgs.loc[g_, "Earliest_Start"], tgs_src, "earliest window start")
    add(f"tod.{k_}_hi", tgs.loc[g_, "Latest_Start"], tgs.loc[g_, "Latest_Start"], tgs_src, "latest window start")
tst = tgs.loc["patients vs controls, window start"]
add("tod.r", R(tst.r_rank_biserial), tst.r_rank_biserial, tgs_src, "window start, rank-biserial, U for controls")
add("tod.pe", PE(tst.p), tst.p, tgs_src)
gap_rec = tg[tg.Gaps_Window > 0]
add("gap.n_rec", WORDS[len(gap_rec)], len(gap_rec), tg_src, "records with a gap inside the window")
add("gap.n_c", WORDS[int((gap_rec.Group == "control").sum())], int((gap_rec.Group == "control").sum()), tg_src)
assert int((gap_rec.Group == "patient").sum()) == 1                 # the text says "one patient"
mn, mx = gap_rec.Missing_s_Window.min(), gap_rec.Missing_s_Window.max()
add("gap.min_s", U(mn, 1), mn, tg_src, "smallest total missing time in a window (s)")
add("gap.max_min", U(mx / 60.0, 1), mx / 60.0, tg_src, "largest total missing time in a window (min)")
ch = ges[ges.Conclusion_changed.astype(str) == "True"]
assert sorted(ch.Metric) == ["HRV_SD1SD2", "SampEn"] and (ch.Family == "primary").all()
assert (ch.Significant_full.astype(str) == "True").all()             # both lost significance
add("gap.n_changed", WORDS[len(ch)], len(ch), ges_src)
add("gap.n_tests", str(len(ges)), len(ges), ges_src)
for m_, k_ in (("SampEn", "sampen"), ("HRV_SD1SD2", "sd1sd2")):
    sh = float(ch.set_index("Metric").loc[m_, "Share_random_exclusions_changing"]) * 100
    add(f"gap.{k_}_share", INT(sh), sh, ges_src, "percent of random exclusions changing the conclusion")

print("Heart rate: bootstrap intervals (hr_adjusted_bootstrap.py)")
hb_src = os.path.join(CMP, "HR_Adjusted_Bootstrap.csv")
hb = pd.read_csv(csvp(hb_src)).set_index("Metric")
assert sorted(hb.index) == sorted(hr.index) and (hb.N_Boot == 2000).all() and (hb.Seed == 0).all()
assert np.allclose(hb.loc[hr.index, "r_partial_HR"], hr.r_partial_HR, atol=1e-7)
assert np.allclose(hb.loc[hr.index, "r_partial_unadjusted"], hr.r_partial_unadjusted, atol=1e-7)
lo, hi = hb.loc["HRV_RMSSD", "HR_CI_low"], hb.loc["HRV_RMSSD", "HR_CI_high"]
add("hrb.RMSSD.a_ci", f"95% CI {R(lo)} to {R(hi)}", (lo, hi), hb_src, "RMSSD adjusted partial r, percentile bootstrap")
# "adjustment reduced every amplitude association": the change interval excludes zero, on the side of a smaller |r|
assert (hb.loc[AMP, "delta_CI_low"] > 0).all() and (hb.loc[AMP, "r_partial_HR"] < 0).all()
# "for all four, the interval for the change included zero"
DIST4 = ["VLF_LF_Ratio", "PSD_Beta_NoMayer", "PSD_Beta", "DFA_Alpha2"]
assert not hb.loc[DIST4, "delta_CI_excludes_0"].astype(str).eq("True").any()
# the adjusted interval excludes zero exactly where the adjusted test has p < 0.05
assert (hb.loc[hr.index, "HR_CI_excludes_0"].astype(str).eq("True") == (hr.p_raw < 0.05)).all()

print("Normalization by mean NN")
nn_src = os.path.join(CMP, "HR_Normalised_MeanNN.csv")
nn = pd.read_csv(csvp(nn_src)).set_index("Measure")
add("nn.RMSSD", R(nn.loc["HRV_RMSSD", "r_effect"]), nn.loc["HRV_RMSSD", "r_effect"], nn_src)
powr = nn.loc[["LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2", "Total_Power_ms2"], "r_effect"]
add("nn.pow_lo", R(powr.max()), powr.max(), nn_src, "weakest normalized power")
add("nn.pow_hi", R(powr.min()), powr.min(), nn_src, "strongest normalized power")
assert (nn["p_raw"] < 0.05).all()

print("Fixed beat count (Length_vs_HeartRate.csv)")
lv_src = os.path.join(OUT_DIR, "comparison", "Length_vs_HeartRate.csv")
lv = pd.read_csv(csvp(lv_src))
fb = lv[lv.Condition == "fixed 2000 beats"].set_index("Metric")
add("fb.n_c", str(int(fb.N_control.iloc[0])), fb.N_control.iloc[0], lv_src)
add("fb.n_p", str(int(fb.N_patient.iloc[0])), fb.N_patient.iloc[0], lv_src)
for k, m in (("SampEn", "SampEn"), ("ApEn", "ApEn"), ("DET", "RQA_DET"), ("LAM", "RQA_LAM"), ("MSE", "MSE_Mean")):
    add(f"fb.{k}.r", R(fb.loc[m, "r_effect"]), fb.loc[m, "r_effect"], lv_src, m)
    add(f"fb.{k}.pe", PE(fb.loc[m, "p_raw"]), fb.loc[m, "p_raw"], lv_src, m)
for m in ("SampEn", "ApEn", "RQA_DET", "RQA_LAM"):
    assert fb.loc[m, "p_raw"] > 0.05, m
assert fb.loc["MSE_Mean", "p_raw"] < 0.05

print("Resolution at 1.000 ms (Requantization_Resolution_Floor.csv)")
rf_src = os.path.join(CMP, "Requantization_Resolution_Floor.csv")
rf = pd.read_csv(csvp(rf_src)).set_index("Quantity")
add("res.ratio_c", U(rf.loc["Quant_Step_Over_RMSSD", "Healthy_Median"], 4), rf.loc["Quant_Step_Over_RMSSD", "Healthy_Median"], rf_src)
add("res.ratio_p", U(rf.loc["Quant_Step_Over_RMSSD", "Psych_Median"], 4), rf.loc["Quant_Step_Over_RMSSD", "Psych_Median"], rf_src)
add("res.ratio_r", R(rf.loc["Quant_Step_Over_RMSSD", "r_effect"]), rf.loc["Quant_Step_Over_RMSSD", "r_effect"], rf_src)
add("res.tie_c", U(100 * rf.loc["Ordinal_Tie_Rate", "Healthy_Median"], 1), rf.loc["Ordinal_Tie_Rate", "Healthy_Median"], rf_src)
add("res.tie_p", U(100 * rf.loc["Ordinal_Tie_Rate", "Psych_Median"], 1), rf.loc["Ordinal_Tie_Rate", "Psych_Median"], rf_src)
add("res.tie_r", R(rf.loc["Ordinal_Tie_Rate", "r_effect"]), rf.loc["Ordinal_Tie_Rate", "r_effect"], rf_src)
add("rq.steps_p", U(rf.loc["Requant_RMSSD_In_Steps", "Psych_Median"], 2), rf.loc["Requant_RMSSD_In_Steps", "Psych_Median"], rf_src)
add("rq.steps_c", U(rf.loc["Requant_RMSSD_In_Steps", "Healthy_Median"], 2), rf.loc["Requant_RMSSD_In_Steps", "Healthy_Median"], rf_src)
add("rq.steps_p_min", U(rf.loc["Requant_RMSSD_In_Steps", "Psych_Min"], 2), rf.loc["Requant_RMSSD_In_Steps", "Psych_Min"], rf_src)
add("rq.tie_p", INT(100 * rf.loc["Requant_Tie_Rate", "Psych_Median"]), rf.loc["Requant_Tie_Rate", "Psych_Median"], rf_src)
add("rq.tie_c", INT(100 * rf.loc["Requant_Tie_Rate", "Healthy_Median"]), rf.loc["Requant_Tie_Rate", "Healthy_Median"], rf_src)

print("Quantization controls (Quantization_Control.csv)")
qc_src = os.path.join(CMP, "Quantization_Control.csv")
qcd = pd.read_csv(csvp(qc_src))


def qrow(metric, cond):
    s = qcd[(qcd.Metric == metric) & (qcd.Condition == cond)]
    assert len(s) == 1, (metric, cond)
    return s.iloc[0]


MQ = "matched_quant_r088"
ratio = pd.concat([pd.to_numeric(C["Quant_Step_Over_RMSSD"]), pd.to_numeric(P["Quant_Step_Over_RMSSD"])])
mq_c = int((pd.to_numeric(C["Quant_Step_Over_RMSSD"]) < 0.088).sum())
mq_p = int((pd.to_numeric(P["Quant_Step_Over_RMSSD"]) < 0.088).sum())
add("quant.mq_c", str(mq_c), mq_c, P_C, "Quant_Step_Over_RMSSD < 0.088 (coarsened)")
add("quant.mq_p", str(mq_p), mq_p, P_P, "Quant_Step_Over_RMSSD < 0.088 (coarsened)")
assert abs(float(env["quant_target_ratio"]) - 0.088) < 1e-12
for m in ("SampEn", "PermEn"):
    add(f"quant.{m}.obs", R(qrow(m, "observed").r_effect), qrow(m, "observed").r_effect, qc_src)
    add(f"quant.{m}.dith", R(qrow(m, "dither").r_effect), qrow(m, "dither").r_effect, qc_src)
add("quant.SampEn.mq", R(qrow("SampEn", MQ).r_effect), qrow("SampEn", MQ).r_effect, qc_src)
shifts, mse_shift = [], []
for m in ("RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"):
    o = qrow(m, "observed").r_effect
    shifts += [abs(qrow(m, "dither").r_effect - o), abs(qrow(m, MQ).r_effect - o)]
o = qrow("MSE_Mean", "observed").r_effect
mse_shift = [abs(qrow("MSE_Mean", "dither").r_effect - o), abs(qrow("MSE_Mean", MQ).r_effect - o)]
add("quant.rec_maxshift", U(max(shifts)), max(shifts), qc_src, "max |shift| DET, LAM, CLU, TRA under dither and matching")
add("quant.mse_maxshift", U(max(mse_shift)), max(mse_shift), qc_src)
# The tested dither condition is one seeded draw (the *_Dither columns); the pipeline also stores the mean of 20
# further draws (*_Dither_RepMean). The text says the replicate mean changes no effect size by more than this.
_rep = []
for m in ("SampEn", "ApEn", "PermEn", "RQA_DET", "RQA_LAM", "MSE_Mean"):
    assert (pd.to_numeric(C[m + "_Dither_RepN"]) == 20).all() and (pd.to_numeric(P[m + "_Dither_RepN"]) == 20).all()
    _r1 = mw(C[m + "_Dither"], P[m + "_Dither"])[0]
    assert abs(_r1 - qrow(m, "dither").r_effect) < 1e-6, m
    _rep.append(abs(mw(C[m + "_Dither_RepMean"], P[m + "_Dither_RepMean"])[0] - _r1))
add("quant.dither_rep_max", U(max(_rep)), max(_rep), P_C + " + " + P_P, "max |r(20-draw mean) - r(one draw)|")
# "Every shift was away from zero or below 0.01"
for m in ("SampEn", "ApEn", "PermEn", "MSE_Mean", "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"):
    o = qrow(m, "observed").r_effect
    for cond in ("dither", MQ):
        v = qrow(m, cond).r_effect
        assert abs(v) >= abs(o) or abs(v - o) < 0.01, (m, cond, o, v)
add("quant.tie_mq_r", R(qrow("Ordinal_Tie_Rate", MQ).r_effect), qrow("Ordinal_Tie_Rate", MQ).r_effect, qc_src)
s20o, s20m = qrow("RQA_DET", "observed_MQ20_matchedsubsample"), qrow("RQA_DET", "matched_MQ20_matchedsubsample")
add("quant.s20_c", str(int(s20o.N_Healthy)), s20o.N_Healthy, qc_src)
add("quant.s20_p", str(int(s20o.N_Psych)), s20o.N_Psych, qc_src)
add("quant.s20.DET.obs", R(s20o.r_effect), s20o.r_effect, qc_src)
add("quant.s20.DET.mq", R(s20m.r_effect), s20m.r_effect, qc_src)
add("quant.s20.SampEn.obs", R(qrow("SampEn", "observed_MQ20_matchedsubsample").r_effect),
    qrow("SampEn", "observed_MQ20_matchedsubsample").r_effect, qc_src)
add("quant.s20.SampEn.mq", R(qrow("SampEn", "matched_MQ20_matchedsubsample").r_effect),
    qrow("SampEn", "matched_MQ20_matchedsubsample").r_effect, qc_src)

print("Re-quantization to 7.8125 ms (Requantization_Q128.csv)")
q_src = os.path.join(CMP, "Requantization_Q128.csv")
qq = pd.read_csv(csvp(q_src)).set_index("Metric")
for k, m in HK.items():
    if m not in qq.index:
        assert m == "DFA_Alpha", m  # full-range DFA alpha not recomputed on the grid (stated in text)
        continue
    add(f"q.{k}.n", R(qq.loc[m, "r_native"]), qq.loc[m, "r_native"], q_src, m)
    add(f"q.{k}.q", R(qq.loc[m, "r_Q128"]), qq.loc[m, "r_Q128"], q_src, m)
    add(f"q.{k}.pqe", PE(qq.loc[m, "p_Q128"]), qq.loc[m, "p_Q128"], q_src, m)
for k, col in (("mc_n", "Healthy_Median_Native"), ("mc_q", "Healthy_Median_Q128"),
               ("mp_n", "Psych_Median_Native"), ("mp_q", "Psych_Median_Q128")):
    add(f"q.SampEn.{k}", U(qq.loc["SampEn", col], 2), qq.loc["SampEn", col], q_src)
SPEC10 = ["HRV_RMSSD", "HRV_SDNN", "LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2", "Total_Power_ms2",
          "VLF_LF_Ratio", "LF_HF_Ratio", "PSD_Beta", "PSD_Beta_NoMayer"]
ch = qq.loc[SPEC10, "r_change"].abs()
assert ch.idxmax() == "LF_HF_Ratio"
add("q.maxchange", U(ch.max()), ch.max(), q_src, "max |r_change| over 10 amplitude and spectral measures")

print("Quantization noise floor")
nf_src = os.path.join(CMP, "Quantization_Noise_Floor.csv")
nf = pd.read_csv(csvp(nf_src))


def nfv(g, acq):
    return float(nf[(nf.Group == g) & (nf.Acquisition == acq)]["Noise_To_Signal_Pct"].iloc[0])


add("nf.mit_p", U(nfv("Psychiatric", "MIT-BIH"), 2), nfv("Psychiatric", "MIT-BIH"), nf_src)
add("nf.mit_c", U(nfv("Healthy", "MIT-BIH"), 2), nfv("Healthy", "MIT-BIH"), nf_src)
add("nf.pol_p", U(nfv("Psychiatric", "Polar H10"), 3), nfv("Psychiatric", "Polar H10"), nf_src)
add("nf.pol_c", U(nfv("Healthy", "Polar H10"), 4), nfv("Healthy", "Polar H10"), nf_src)
# Conservative variant: the file spreads the white quantization noise (variance q^2/12) up to the 2 Hz
# Nyquist frequency of the 4 Hz resampled series. Interval noise lives at the beat rate, so spreading it
# only up to half the mean beat rate (HR/120 Hz) raises its density by 2 / (HR/120). Same fitted band
# and the same mean LF denominator as the file; checked against the file at the 2 Hz assumption.
FIT_W = 0.15 - 0.0156   # comparison.py BETA_BAND_FOR_REPORT, the band width the file uses
def noise_pct(step_ms, grp, nyq):
    return 100.0 * (step_ms ** 2 / 12.0) * FIT_W / nyq / pd.to_numeric(grp.LF_Power_ms2).mean()
assert abs(noise_pct(7.8125, P, 2.0) - nfv("Psychiatric", "MIT-BIH")) < 0.002, "noise formula does not reproduce the file"
assert abs(noise_pct(1.0, C, 2.0) - nfv("Healthy", "Polar H10")) < 0.0002
nyq_p = pd.to_numeric(P.HRV_MeanHR).mean() / 120.0
nyq_c = pd.to_numeric(C.HRV_MeanHR).mean() / 120.0
add("nf.mit_p_beat", U(noise_pct(7.8125, P, nyq_p), 2), noise_pct(7.8125, P, nyq_p), P_P, f"q=7.8125 ms, Nyquist {nyq_p:.3f} Hz")
add("nf.mit_c_beat", U(noise_pct(7.8125, C, nyq_c), 2), noise_pct(7.8125, C, nyq_c), P_C, f"q=7.8125 ms, Nyquist {nyq_c:.3f} Hz")
add("nf.pol_p_beat", U(noise_pct(1.0, P, nyq_p), 3), noise_pct(1.0, P, nyq_p), P_P, f"q=1 ms, Nyquist {nyq_p:.3f} Hz (largest of the 1 ms values)")
assert noise_pct(1.0, P, nyq_p) >= noise_pct(1.0, C, nyq_c)

print("Estimator choices")
add("est.a1_4", R(hr.loc["DFA_Alpha1", "r_rank_biserial"]), hr.loc["DFA_Alpha1", "r_rank_biserial"], hr_src)
add("est.a1_4pe", PE(hr.loc["DFA_Alpha1", "p_rank_biserial"]), hr.loc["DFA_Alpha1", "p_rank_biserial"], hr_src)
for n in (8, 10):
    r, p, _, _ = mw(C[f"DFA_Alpha1_nmin{n}"], P[f"DFA_Alpha1_nmin{n}"])
    add(f"est.a1_{n}", R(r), r, "per-subject files", f"Mann-Whitney DFA_Alpha1_nmin{n}")
    add(f"est.a1_{n}pe", PE(p), p, "per-subject files", f"Mann-Whitney DFA_Alpha1_nmin{n}")
mw_src = os.path.join(CMP, "Mayer_Wave_Control.csv")
mwc = pd.read_csv(csvp(mw_src)).set_index("Estimator")
EST5 = ["PSD_Beta", "PSD_Beta_LogBin", "PSD_Beta_NoMayer", "PSD_Beta_Specparam", "PSD_Beta_IRASA"]
er = mwc.loc[EST5, "r_effect"]
assert (mwc.loc[EST5, "p_adj"] < 0.05).all() and mwc.loc["PSD_Beta_Specparam_CleanFits", "p_adj"] < 0.05
add("est.slope_min", R(er.min()), er.min(), mw_src, "min r over five estimators")
add("est.slope_max", R(er.max()), er.max(), mw_src, "max r over five estimators")
cf = mwc.loc["PSD_Beta_Specparam_CleanFits"]
add("est.clean_r", R(cf.r_effect), cf.r_effect, mw_src)
add("est.clean_n_c", str(int(cf.N_Healthy)), cf.N_Healthy, mw_src)
add("est.clean_n_p", str(int(cf.N_Psych)), cf.N_Psych, mw_src)
dc = float(mwc.loc["PSD_Beta_NoMayer", "Healthy_Mean"] - mwc.loc["PSD_Beta", "Healthy_Mean"])
dp = float(mwc.loc["PSD_Beta_NoMayer", "Psych_Mean"] - mwc.loc["PSD_Beta", "Psych_Mean"])
dd = float(mwc.loc["PSD_Beta", "Psych_Mean"] - mwc.loc["PSD_Beta", "Healthy_Mean"])
add("may.dc", U(dc), dc, mw_src, "excised minus uncorrected, control means")
add("may.dp", U(dp), dp, mw_src, "excised minus uncorrected, patient means")
add("may.pct", INT(100 * (dc - dp) / dd), 100 * (dc - dp) / dd, mw_src, "(dc - dp) / uncorrected group difference")

emb_src = os.path.join(CMP, "Embedding_Sensitivity.csv")
emb = pd.read_csv(csvp(emb_src))


def eq(qty, typ):
    return emb[(emb.Quantity == qty) & (emb.Type == typ)].iloc[0]


add("emb.det_fixed", R(eq("RQA_DET", "primary_embedding").r_effect), eq("RQA_DET", "primary_embedding").r_effect, emb_src)
add("emb.det_fixed_pe", PE(eq("RQA_DET", "primary_embedding").p_raw), eq("RQA_DET", "primary_embedding").p_raw, emb_src)
add("emb.det_sel", R(eq("RQA_DET", "selected_embedding").r_effect), eq("RQA_DET", "selected_embedding").r_effect, emb_src)
add("emb.det_sel_pe", PE(eq("RQA_DET", "selected_embedding").p_raw), eq("RQA_DET", "selected_embedding").p_raw, emb_src)
add("emb.lam_fixed", R(eq("RQA_LAM", "primary_embedding").r_effect), eq("RQA_LAM", "primary_embedding").r_effect, emb_src)
add("emb.lam_sel", R(eq("RQA_LAM", "selected_embedding").r_effect), eq("RQA_LAM", "selected_embedding").r_effect, emb_src)
dim = eq("Embed_Dim_FNN", "selected_parameter")
add("emb.dim_c", INT(dim.Healthy_Median), dim.Healthy_Median, emb_src)
add("emb.dim_p", INT(dim.Psych_Median), dim.Psych_Median, emb_src)
add("emb.dim_r", R(dim.r_effect), dim.r_effect, emb_src)
add("emb.dim_pe", PE(dim.p_raw), dim.p_raw, emb_src)
assert dim.p_raw > 0.05

rna_src = os.path.join(CMP, "RNA_Group_Comparison.csv")
rna = pd.read_csv(csvp(rna_src)).set_index("Metric")
add("rna.clu_r", R(rna.loc["RNA_ClusteringCoeff", "r_effect"]), rna.loc["RNA_ClusteringCoeff", "r_effect"], rna_src)
add("rna.clu_pe", PE(rna.loc["RNA_ClusteringCoeff", "p_raw"]), rna.loc["RNA_ClusteringCoeff", "p_raw"], rna_src)
add("rna.tra_r", R(rna.loc["RNA_Transitivity", "r_effect"]), rna.loc["RNA_Transitivity", "r_effect"], rna_src)
add("rna.clu_pae", PE(rna.loc["RNA_ClusteringCoeff", "p_adj"]), rna.loc["RNA_ClusteringCoeff", "p_adj"], rna_src)
add("rna.tra_pae", PE(rna.loc["RNA_Transitivity", "p_adj"]), rna.loc["RNA_Transitivity", "p_adj"], rna_src)
assert (rna.p_adj > 0.05).all(), "text says neither recurrence-network measure differed"

sn_src = os.path.join(CMP, "Surrogate_Normalized_Entropy.csv")
sn = pd.read_csv(csvp(sn_src))
s_raw = sn[(sn.Metric == "MSE_Mean") & (sn.Scale == "raw")].iloc[0]
s_z = sn[(sn.Metric == "MSE_Mean") & (sn.Scale == "surrogate_Z")].iloc[0]
add("sn.mse_raw_pae", PE(s_raw.p_adj), s_raw.p_adj, sn_src)
add("sn.mse_z_r", R(s_z.r_effect), s_z.r_effect, sn_src)
add("sn.mse_z_pae", PE(s_z.p_adj), s_z.p_adj, sn_src)
sd_src = os.path.join(CMP, "Surrogate_Distribution_Summary.csv")
sdd = pd.read_csv(csvp(sd_src)).set_index(["Metric", "Group"])
for g, G in (("c", "Healthy"), ("p", "Psychiatric")):
    msr = sdd.loc[("MSE_Mean", G)]
    assert int(msr.N_Below_Surrogates) == int(msr.N_Differing), "text says all differing MSE values lie below"
    add(f"sd.mse_{g}_diff", str(int(msr.N_Differing)), msr.N_Differing, sd_src)
    se = sdd.loc[("SampEn", G)]
    add(f"sd.se_{g}_diff", str(int(se.N_Differing)), se.N_Differing, sd_src)
    add(f"sd.se_{g}_below", str(int(se.N_Below_Surrogates)), se.N_Below_Surrogates, sd_src)

tol_src = os.path.join(CMP, "Entropy_Tolerance_Sensitivity.csv")
tol = pd.read_csv(csvp(tol_src))
assert (tol.p_raw < 0.05).all() and (tol.r_effect < 0).all()
assert abs(tol.Tolerance_r_x_SD.min() - 0.10) < 1e-9 and abs(tol.Tolerance_r_x_SD.max() - 0.35) < 1e-9
add("tol.min", R(tol.r_effect.max()), tol.r_effect.max(), tol_src, "weakest")
add("tol.max", R(tol.r_effect.min()), tol.r_effect.min(), tol_src, "strongest")

skip_src = os.path.join(OUT_DIR, "comparison", "Window_Skip_Sensitivity.csv")
sk = pd.read_csv(csvp(skip_src))
bx = sk[sk.Metric == "PSD_Beta_NoMayer"]
assert (bx.p_raw < 0.05).all()
add("skip.betax_lo", R(bx.r_effect.min()), bx.r_effect.min(), skip_src)
add("skip.betax_hi", R(bx.r_effect.max()), bx.r_effect.max(), skip_src)
# Section 2.2: the skip was chosen after inspecting patients' heart rates, so the text reports what the choice did.
hr_sk = sk[sk.Metric == "HRV_MeanHR"].set_index("Skip_s")
for k in (60, 360, 600):
    add(f"skip.hr_p_{k}", U(hr_sk.loc[k, "Patient_mean"], 1), hr_sk.loc[k, "Patient_mean"], skip_src, "patient mean heart rate")
piv = sk.pivot(index="Metric", columns="Skip_s", values="r_effect").abs()
gain = (piv[600] - piv[[60, 360]].max(axis=1))
assert len(piv) == 11 and gain.max() < 0.01, gain.sort_values().tail(3)
add("skip.max_gain", U(gain.max()), gain.max(), skip_src,
    "largest amount by which |r| at 600 s exceeded |r| at both shorter skips, over the 11 metrics")
for m in ("SampEn", "RQA_DET"):  # "weakened as the skip lengthened"
    s = sk[sk.Metric == m].sort_values("Skip_s").r_effect.abs().values
    assert s[0] > s[1] > s[2], m
for m in ("RNA_ClusteringCoeff", "RNA_Transitivity"):  # "varied without a pattern"
    s = sk[sk.Metric == m].sort_values("Skip_s").r_effect.abs().values
    assert not (s[0] > s[1] > s[2]) and not (s[0] < s[1] < s[2]), m

print("Scale-to-frequency correspondence")
sb_src = os.path.join(CMP, "Scale_Band_Correspondence.csv")
sb = pd.read_csv(csvp(sb_src)).set_index(["Group", "Exponent"])
add("sb.nn_c", U(sb.loc[("Healthy", "alpha1"), "Mean_NN_s"]), sb.loc[("Healthy", "alpha1"), "Mean_NN_s"], sb_src)
add("sb.nn_p", U(sb.loc[("Psychiatric", "alpha1"), "Mean_NN_s"]), sb.loc[("Psychiatric", "alpha1"), "Mean_NN_s"], sb_src)
for g, G in (("c", "Healthy"), ("p", "Psychiatric")):
    for e in ("a1", "a2"):
        rw = sb.loc[(G, "alpha1" if e == "a1" else "alpha2")]
        add(f"sb.{e}_{g}", f"{rw.Freq_Lo_Hz:.3f} to {rw.Freq_Hi_Hz:.3f}", (rw.Freq_Lo_Hz, rw.Freq_Hi_Hz), sb_src)

print("Control cohorts and cross-device")
cc_src = os.path.join(XC, "Control_vs_Control_Stats.csv")
cc = pd.read_csv(csvp(cc_src))
cc = cc.rename(columns={cc.columns[0]: "Label"}).set_index("Label")
a1 = cc.loc["DFA α₁ (4–16)"]
assert (cc.Differs.astype(str) == "True").sum() == 1 and str(a1.Differs) == "True"
add("cc.a1_m", U(a1.A_Mean), a1.A_Mean, cc_src, "MIT-BIH mean")
add("cc.a1_h", U(a1.B_Mean), a1.B_Mean, cc_src, "HRV-ACC mean")
add("cc.a1_r", R(a1.r_effect), a1.r_effect, cc_src)
add("cc.a1_pae", PE(a1.MWU_p_adj), a1.MWU_p_adj, cc_src)
cv_src = os.path.join(XC, "CrossControl_Convergence.csv")
cv = pd.read_csv(csvp(cv_src))
add("xc.n_sig_mit", str(int((cv.sig_MITBIH.astype(str) == "True").sum())), int((cv.sig_MITBIH.astype(str) == "True").sum()), cv_src)
add("xc.n_both", str(int((cv.Robust.astype(str) == "True").sum())), int((cv.Robust.astype(str) == "True").sum()), cv_src)
sig_h, sig_m = cv.sig_HRVACC.astype(str) == "True", cv.sig_MITBIH.astype(str) == "True"
assert sorted(cv.Metric[sig_m & ~sig_h]) == ["RQA Determinism", "RQA Laminarity"], "text names the MIT-BIH-only measures"
assert list(cv.Metric[sig_h & ~sig_m]) == ["Permutation Entropy"], "text names the HRV-ACC-only measure"
cvp = cv.set_index("Metric").loc["Permutation Entropy"]
assert cv.set_index("Metric").loc["RMSSD (ms)", "r_MITBIH"] < 0, "r_MITBIH is positive when patients are higher"
add("xc.permen_r", R(cvp.r_MITBIH), cvp.r_MITBIH, cv_src, "Permutation Entropy r_MITBIH (positive = higher in patients)")
add("xc.permen_pae", PE(cvp.p_adj_MITBIH), cvp.p_adj_MITBIH, cv_src, "Permutation Entropy p_adj_MITBIH")
xd_src = os.path.join(XC, "CrossDevice_AgeAdjusted.csv")
xd = pd.read_csv(csvp(xd_src)).set_index(["Metric", "Comparison"])
mp = xd.loc[("PSD_Beta_NoMayer", "MITBIH_control vs patients")]
ccx = xd.loc[("PSD_Beta_NoMayer", "HRVACC_control vs MITBIH_control")]
add("xd.x_mp_rb", R(mp.r_rank_biserial), mp.r_rank_biserial, xd_src)
add("xd.x_mp_u", R(mp.r_partial_unadjusted), mp.r_partial_unadjusted, xd_src)
add("xd.x_mp_a", R(mp.r_partial_age_adjusted), mp.r_partial_age_adjusted, xd_src)
add("xd.x_mp_ape", PE(mp.p_partial_age_adjusted), mp.p_partial_age_adjusted, xd_src)
add("xd.x_pct", INT(mp.pct_retained), mp.pct_retained, xd_src)
# One sign convention for every comparison with the MIT-BIH cohort: positive = higher in the HRV-ACC group.
# The file signs each row as "positive = second-named cohort higher", so the control-cohort row (second-named
# MIT-BIH) is negated and the patient row (second-named patients) is used as stored. Control_vs_Control_Stats
# (A = MIT-BIH, B = HRV-ACC) already follows the convention: alpha1 is higher in HRV-ACC and its r is positive.
assert str(mp.Sign).startswith("positive = second-named") and str(ccx.Sign).startswith("positive = second-named")
assert (a1.B_Mean > a1.A_Mean) == (a1.r_effect > 0), "Control_vs_Control_Stats sign convention changed"
add("xd.x_cc_rb", R(-ccx.r_rank_biserial), -ccx.r_rank_biserial, xd_src, "negated: positive = higher in HRV-ACC controls")
add("xd.x_cc_pe", PE(ccx.p_rank_biserial), ccx.p_rank_biserial, xd_src)
add("xd.x_cc_u", R(-ccx.r_partial_unadjusted), -ccx.r_partial_unadjusted, xd_src, "negated: positive = higher in HRV-ACC controls")
add("xd.x_cc_a", R(-ccx.r_partial_age_adjusted), -ccx.r_partial_age_adjusted, xd_src, "negated: positive = higher in HRV-ACC controls")
add("xd.x_cc_ape", PE(ccx.p_partial_age_adjusted), ccx.p_partial_age_adjusted, xd_src)
aa_src = os.path.join(CMP, "Age_Association_BothGroups.csv")
aa = pd.read_csv(csvp(aa_src)).set_index("Metric")
add("agea.x_c", R(aa.loc["PSD_Beta_NoMayer", "r_age_control"]), aa.loc["PSD_Beta_NoMayer", "r_age_control"], aa_src)
add("agea.x_c_pe", PE(aa.loc["PSD_Beta_NoMayer", "p_age_control"]), aa.loc["PSD_Beta_NoMayer", "p_age_control"], aa_src)

print("Spectral redistribution")
v_src = os.path.join(CMP, "VLF_Decomposition.csv")
v = pd.read_csv(csvp(v_src)).set_index("Metric")
add("vlf.pct_vlf", INT(abs(v.loc["VLF_Power_ms2", "Pct_Median_Change"])), v.loc["VLF_Power_ms2", "Pct_Median_Change"], v_src)
add("vlf.pct_lf", INT(abs(v.loc["LF_Power_ms2", "Pct_Median_Change"])), v.loc["LF_Power_ms2", "Pct_Median_Change"], v_src)
add("vlf.pct_hf", INT(abs(v.loc["HF_Power_ms2", "Pct_Median_Change"])), v.loc["HF_Power_ms2", "Pct_Median_Change"], v_src)
add("vlf.vlf_c", INT(v.loc["VLF_Power_ms2", "Healthy_Median"]), v.loc["VLF_Power_ms2", "Healthy_Median"], v_src)
add("vlf.vlf_p", INT(v.loc["VLF_Power_ms2", "Psych_Median"]), v.loc["VLF_Power_ms2", "Psych_Median"], v_src)
add("vlf.vlf_r", R(v.loc["VLF_Power_ms2", "r_effect"]), v.loc["VLF_Power_ms2", "r_effect"], v_src)
for k, m in (("share", "VLF_Share"), ("lfshare", "LF_Share"), ("hfshare", "HF_Share")):
    add(f"vlf.{k}_c", INT(100 * v.loc[m, "Healthy_Median"]), v.loc[m, "Healthy_Median"], v_src)
    add(f"vlf.{k}_p", INT(100 * v.loc[m, "Psych_Median"]), v.loc[m, "Psych_Median"], v_src)
add("vlf.ratio_c", U(v.loc["VLF_LF_Ratio", "Healthy_Median"], 2), v.loc["VLF_LF_Ratio", "Healthy_Median"], v_src)
add("vlf.ratio_p", U(v.loc["VLF_LF_Ratio", "Psych_Median"], 2), v.loc["VLF_LF_Ratio", "Psych_Median"], v_src)
add("vlf.ratio_r", R(v.loc["VLF_LF_Ratio", "r_effect"]), v.loc["VLF_LF_Ratio", "r_effect"], v_src)
assert v.loc["HF_Share", "p_raw"] > 0.05, "text says HF share was similar"
assert abs(v.loc["VLF_Power_ms2", "Pct_Median_Change"]) < abs(v.loc["HF_Power_ms2", "Pct_Median_Change"]) < abs(v.loc["LF_Power_ms2", "Pct_Median_Change"])

add("sl.x_mc", U(qq.loc["PSD_Beta_NoMayer", "Healthy_Median_Native"], 2), qq.loc["PSD_Beta_NoMayer", "Healthy_Median_Native"], q_src)
add("sl.x_mp", U(qq.loc["PSD_Beta_NoMayer", "Psych_Median_Native"], 2), qq.loc["PSD_Beta_NoMayer", "Psych_Median_Native"], q_src)
add("sl.x_r", R(qq.loc["PSD_Beta_NoMayer", "r_native"]), qq.loc["PSD_Beta_NoMayer", "r_native"], q_src)

bs_src = os.path.join(CMP, "Beta_Band_Sweep.csv")
bs = pd.read_csv(csvp(bs_src)).set_index("Band")
assert abs(bs.loc["verylow", "Actual_high_Hz"] - 0.046875) < 1e-9 and abs(bs.loc["low", "Actual_high_Hz"] - 0.0625) < 1e-9
add("band.verylow_r", R(bs.loc["verylow", "r_effect"]), bs.loc["verylow", "r_effect"], bs_src)
add("band.low_mc", U(bs.loc["low", "Control_mean"], 2), bs.loc["low", "Control_mean"], bs_src)
add("band.low_mp", U(bs.loc["low", "Patient_mean"], 2), bs.loc["low", "Patient_mean"], bs_src)
add("band.low_r", R(bs.loc["low", "r_effect"]), bs.loc["low", "r_effect"], bs_src)
add("band.full_r", R(bs.loc["wideband", "r_effect"]), bs.loc["wideband", "r_effect"], bs_src)
assert (bs.drop(index="wideband").p_adj < 0.05).all() and bs.loc["wideband", "r_effect"] < 0
add("band.full_pe", PE(bs.loc["wideband", "p_raw"]), bs.loc["wideband", "p_raw"], bs_src)
assert bs.loc["wideband", "p_raw"] > 0.05, "text says the difference disappears (no significant reversal)"

id_src = os.path.join(CMP, "Beta_DFA_Consistency.csv")
idd = pd.read_csv(csvp(id_src)).set_index("Group")
add("id.res_c", R(idd.loc["Healthy", "Mean_Residual"]), idd.loc["Healthy", "Mean_Residual"], id_src)
add("id.res_p", R(idd.loc["Psychiatric", "Mean_Residual"]), idd.loc["Psychiatric", "Mean_Residual"], id_src)
add("id.res_r", R(idd.loc["Residual group test", "Mean_Residual"]), idd.loc["Residual group test", "Mean_Residual"], id_src,
    "rank-biserial r stored in Mean_Residual column of the test row")
add("id.res_pe", PE(idd.loc["Residual group test", "SD_Residual"]), idd.loc["Residual group test", "SD_Residual"], id_src,
    "p stored in SD_Residual column of the test row")

print("Covariate models")
ba_src = os.path.join(OUT_DIR, "accelerometer", "Beta_Activity_Covariate.csv")
ba = pd.read_csv(csvp(ba_src)).set_index("Analysis")


def bav(name):
    rw = ba.loc[name]
    return float(rw.partial_r), float(rw.p)


for key, name in (("x_mob", "PSD_Beta_NoMayer ~ Group | Mobility"),
                  ("x_mobhr", "PSD_Beta_NoMayer ~ Group | Mobility + MeanHR"),
                  ("x_hf", "PSD_Beta_NoMayer ~ Group | HF power"),
                  ("r_mob", "VLF_LF_Ratio ~ Group | Mobility"),
                  ("r_mobhr", "VLF_LF_Ratio ~ Group | Mobility + MeanHR")):
    r, p = bav(name)
    add(f"cov.{key}", R(r), r, ba_src, name)
    add(f"cov.{key}_pe", PE(p), p, ba_src, name)
jr = lrc.loc["PSD_Beta_NoMayer ~ Group | Age + Sex + mobility_mg + HRV_MeanHR + HF_Power_ms2"]
add("cov.x_joint", R(jr.r_effect), jr.r_effect, src_lrc, "joint five-covariate")
jv = lrc.loc["VLF_LF_Ratio ~ Group | Age + Sex + mobility_mg + HRV_MeanHR + HF_Power_ms2"]
add("cov.r_joint", R(jv.r_effect), jv.r_effect, src_lrc, "joint five-covariate")
add("cov.r_joint_pe", PE(jv.p_raw), jv.p_raw, src_lrc)
xs = [bav(n)[0] for n in ba.index if n.startswith("PSD_Beta_NoMayer ~ Group |")]
assert min(xs) == bav("PSD_Beta_NoMayer ~ Group | HF power")[0], "text says HF power weakened the slope most"

print("Movement-free segments (recomputed from Stationary_Reanalysis_*.csv)")
st_src = os.path.join(OUT_DIR, "stationary", "Stationary_Reanalysis_beats.csv")
st = pd.read_csv(csvp(st_src))
stc, stp = st[st.Group == "control"], st[st.Group == "patient"]
add("mf.beats_n_c", str(len(stc)), len(stc), st_src)
add("mf.beats_n_p", str(len(stp)), len(stp), st_src)
add("mf.beats_dur", INT(st.Duration_s.median()), st.Duration_s.median(), st_src, "median Duration_s")
# Short segments keep the 512-sample Welch segment (same 0.0078 Hz bins) but average fewer segments.
ws_full = pd.concat([pd.to_numeric(C.Welch_N_Segments), pd.to_numeric(P.Welch_N_Segments)])
assert (pd.to_numeric(C.Welch_NPerSeg) == 512).all() and (pd.to_numeric(P.Welch_NPerSeg) == 512).all()
assert st.Welch_N_Segments.max() < ws_full.min(), "text says movement-free spectra average fewer segments"
add("mf.welch_med", INT(st.Welch_N_Segments.median()), st.Welch_N_Segments.median(), st_src, "median Welch_N_Segments")
add("win.welch_med", INT(ws_full.median()), ws_full.median(), P_C + " + " + P_P, "median Welch_N_Segments, 60 HRV-ACC")
for k, m in (("BETAX", "PSD_Beta_NoMayer"), ("VLFLF", "VLF_LF_Ratio"), ("LF", "LF_Power_ms2"),
             ("HF", "HF_Power_ms2"), ("MSE", "MSE_Mean"), ("A2", "DFA_Alpha2"), ("PermEn", "PermEn"),
             ("SampEn", "SampEn"), ("DET", "RQA_DET")):
    r, p, _, _ = mw(stc[m], stp[m])
    add(f"mf.{k}.r", R(r), r, st_src, f"Mann-Whitney {m}")
    add(f"mf.{k}.pe", PE(p), p, st_src, f"Mann-Whitney {m}")
    if k in ("BETAX", "LF", "HF", "MSE"):
        assert p < 0.05, k
    else:
        assert p > 0.05, k
du = pd.read_csv(csvp(OUT_DIR, "stationary", "Stationary_Reanalysis_duration.csv"))
add("mf.dur_n", str(len(du)), len(du), os.path.join(OUT_DIR, "stationary", "Stationary_Reanalysis_duration.csv"))

print("Diagnostic subgroups, quetiapine and substance use")
dg_src = os.path.join(CMP, "Diagnosis_Subgroup.csv")
dg = pd.read_csv(csvp(dg_src)).set_index("Metric")
x = dg.loc["PSD_Beta_NoMayer"]
add("sub.x_sz", R(x.r_schizophrenia), x.r_schizophrenia, dg_src)
add("sub.x_sz_pe", PE(x.p_schizophrenia), x.p_schizophrenia, dg_src)
add("sub.x_bp", R(x.r_bipolar), x.r_bipolar, dg_src)
add("sub.x_bp_pe", PE(x.p_bipolar), x.p_bipolar, dg_src)
add("sub.x_szbp", R(x.r_sz_vs_bp), x.r_sz_vs_bp, dg_src)
add("sub.x_szbp_pe", PE(x.p_sz_vs_bp), x.p_sz_vs_bp, dg_src)
add("sub.floor_szbp", U(x.min_detectable_r_sz_vs_bp), x.min_detectable_r_sz_vs_bp, dg_src)
assert abs(x.r_sz_vs_bp) < x.min_detectable_r_sz_vs_bp and x.p_sz_vs_bp > 0.05
qa_src = os.path.join(CMP, "Quetiapine_Analysis.csv")
qa = pd.read_csv(csvp(qa_src)).set_index("Metric")
y = qa.loc["PSD_Beta_NoMayer"]
add("sub.x_nonq", R(y.r_nonquet_vs_control), y.r_nonquet_vs_control, qa_src)
add("sub.x_nonq_pe", PE(y.p_nonquet_vs_control), y.p_nonquet_vs_control, qa_src)
add("sub.x_qadj", R(y.r_group_adj_quetiapine), y.r_group_adj_quetiapine, qa_src)
add("sub.x_qadj_pe", PE(y.p_group_adj_quetiapine), y.p_group_adj_quetiapine, qa_src)
add("sub.x_q", R(y.r_quetiapine_vs_control), y.r_quetiapine_vs_control, qa_src)
# Quetiapine_Analysis.csv signs this contrast as positive = higher in UNexposed patients; the text uses
# positive = higher in exposed patients, so the value is negated and the direction asserted from the data.
_q = pd.to_numeric(P.Quetiapine, errors="coerce")
_ex, _un = P.loc[_q == 1, "PSD_Beta_NoMayer"], P.loc[_q == 0, "PSD_Beta_NoMayer"]
_r_eu, _p_eu, _, _ = mw(_un, _ex)
assert abs(_r_eu + y.r_quet_vs_nonquet) < 1e-6 and abs(_p_eu - y.p_quet_vs_nonquet) < 1e-6, "quetiapine sign convention changed"
assert np.median(pd.to_numeric(_ex)) > np.median(pd.to_numeric(_un)), "text says exposed slopes are steeper"
add("sub.x_qvsn", R(_r_eu), _r_eu, qa_src, "negated file value: positive = steeper in exposed patients")
add("sub.x_qvsn_pe", PE(y.p_quet_vs_nonquet), y.p_quet_vs_nonquet, qa_src)
assert y.r_quetiapine_vs_control > y.r_nonquet_vs_control
cm_src = os.path.join(CMP, "Comorbidity_Sensitivity.csv")
cm = pd.read_csv(csvp(cm_src)).set_index("Metric")
z = cm.loc["PSD_Beta_NoMayer"]
add("sub.x_nocom", R(z.r_no_comorbidity), z.r_no_comorbidity, cm_src)
add("sub.n_nocom", str(int(z.N_no_comorbidity)), z.N_no_comorbidity, cm_src)
sg_src = os.path.join(CMP, "Minimum_Detectable_Effect_Subgroups.csv")
sg = pd.read_csv(csvp(sg_src)).set_index("Contrast")
add("sub.floor_qvsn", U(sg.loc["Quetiapine exposed vs unexposed", "Min_Detectable_r_rankbiserial"]),
    sg.loc["Quetiapine exposed vs unexposed", "Min_Detectable_r_rankbiserial"], sg_src)
vq = {"sz": dg.loc["VLF_LF_Ratio", "p_schizophrenia"], "bp": dg.loc["VLF_LF_Ratio", "p_bipolar"],
      "nonq": qa.loc["VLF_LF_Ratio", "p_nonquet_vs_control"], "qadj": qa.loc["VLF_LF_Ratio", "p_group_adj_quetiapine"],
      "nocom": cm.loc["VLF_LF_Ratio", "p_no_comorbidity"]}
assert all(v < 0.05 for v in vq.values()), "text says VLF/LF behaved the same way in every subgroup"
mf_src = os.path.join(CMP, "MFDFA_Group_Comparison.csv")
mfd = pd.read_csv(csvp(mf_src))
assert (mfd.r_effect > 0).all() and (mfd.p_adj < 0.05).all(), "text says the MFDFA exponents were higher in patients"
# Width against each subject's 20 phase-randomized surrogates (MFDFA_Width_Z_PhaseRand): subjects with Z above 1.96.
_zc, _zp = pd.to_numeric(C.MFDFA_Width_Z_PhaseRand), pd.to_numeric(P.MFDFA_Width_Z_PhaseRand)
assert (pd.to_numeric(C.MFDFA_Surr_N_PhaseRand) == 20).all() and (pd.to_numeric(P.MFDFA_Surr_N_PhaseRand) == 20).all()
add("mfdfa.pr_nc", str(int((_zc > 1.96).sum())), int((_zc > 1.96).sum()), P_C, "MFDFA_Width_Z_PhaseRand > 1.96")
add("mfdfa.pr_np", str(int((_zp > 1.96).sum())), int((_zp > 1.96).sum()), P_P, "MFDFA_Width_Z_PhaseRand > 1.96")
assert (_zp > 1.96).sum() > (_zc > 1.96).sum() and mw(_zc, _zp)[1] < 0.05, "text says more often in patients"

print("DFA scales, retained MSE scales, clinical ranges, non-redundant family, activity, equivalence")
# NeuroKit2 fractal_dfa(scale="default"): n = int(N / 10) sizes, exp-spaced from 10 to int(N / 10), cast to
# int and de-duplicated (neurokit2.complexity.fractal_dfa._fractal_dfa_findscales). Recomputing DFA_Alpha on
# 8 controls with this rule reproduced the CSV values exactly (audit, 18 October).
def nk_dfa_scales(n):
    k = int(n / 10)
    return np.unique(np.exp(np.linspace(np.log(10), np.log(k), k)).astype(int))
ALLREC = pd.concat([C, P, M])
_sc = [nk_dfa_scales(int(v)) for v in pd.to_numeric(ALLREC.Num_RR)]
add("dfa.nsc_min", str(min(len(x) for x in _sc)), min(len(x) for x in _sc), "per-subject files", "NeuroKit2 default scale count, 78 records")
add("dfa.nsc_max", str(max(len(x) for x in _sc)), max(len(x) for x in _sc), "per-subject files", "NeuroKit2 default scale count, 78 records")
add("dfa.win_min", str(min(int(x.max()) for x in _sc)), min(int(x.max()) for x in _sc), "per-subject files", "largest window, beats")
add("dfa.win_max", str(max(int(x.max()) for x in _sc)), max(int(x.max()) for x in _sc), "per-subject files", "largest window, beats")
_rc = pd.to_numeric(C.MSE_Scales_Retained); _rp = pd.to_numeric(P.MSE_Scales_Retained); _rm = pd.to_numeric(M.MSE_Scales_Retained)
assert (_rp == 20).all() and int((_rc < 20).sum()) == 1 and int((_rm < 20).sum()) == 1, "text says one control and one MIT-BIH record kept fewer scales"
add("mse.ret_c_min", str(int(_rc.min())), _rc.min(), P_C, "min MSE_Scales_Retained")
add("mse.ret_m_min", str(int(_rm.min())), _rm.min(), P_M, "min MSE_Scales_Retained")
cr_src = os.path.join(CMP, "Clinical_Range_Counts.csv")
cr = pd.read_csv(csvp(cr_src)).set_index(["Criterion", "Group"])
def crn(crit, grp):
    return int(cr.loc[(crit, grp), "N_Meeting"])
assert crn("mean HR > 100 bpm", "Psychiatric") == int((pd.to_numeric(P.HRV_MeanHR) > 100).sum())
assert crn("RMSSD < 10 ms", "Healthy") == int((pd.to_numeric(C.HRV_RMSSD) < 10).sum()) == 1, "text says 'control' (singular)"
add("clin.hr100_p", str(crn("mean HR > 100 bpm", "Psychiatric")), crn("mean HR > 100 bpm", "Psychiatric"), cr_src)
add("clin.hr100_c", str(crn("mean HR > 100 bpm", "Healthy")), crn("mean HR > 100 bpm", "Healthy"), cr_src)
add("clin.rmssd10_p", str(crn("RMSSD < 10 ms", "Psychiatric")), crn("RMSSD < 10 ms", "Psychiatric"), cr_src)
add("clin.rmssd10_c", str(crn("RMSSD < 10 ms", "Healthy")), crn("RMSSD < 10 ms", "Healthy"), cr_src)
nr_src = os.path.join(CMP, "Group_Comparison_NonRedundant.csv")
nr = pd.read_csv(csvp(nr_src))
assert len(nr) == 18
add("nr.n_sig", str(int((nr.p_adj < 0.05).sum())), int((nr.p_adj < 0.05).sum()), nr_src, "p_adj < 0.05 in the 18-measure family")
# Mobility: Mobility_Group_Comparison.csv signs r > 0 as 'controls more mobile'; recomputed here with the
# manuscript convention (positive = higher in patients) from the per-subject mobility column.
_mr, _mp, _, _ = mw(C.mobility_mg, P.mobility_mg)
mg = pd.read_csv(csvp(OUT_DIR, "accelerometer", "Mobility_Group_Comparison.csv")).iloc[0]
assert abs(_mr + mg.rank_biserial_r) < 0.001 and abs(_mp - mg.p_value) < 0.001, "mobility sign or value changed"
assert _mp > 0.05
add("mob.r", R(_mr), _mr, P_C + " + " + P_P, "Mann-Whitney on mobility_mg, positive = more mobile patients")
add("mob.pe", PE(_mp), _mp, P_C + " + " + P_P)
ac_src = os.path.join(OUT_DIR, "accelerometer", "Activity_Structure_Comparison.csv")
ac = pd.read_csv(csvp(ac_src)).set_index("Quantity")
pw_ = ac.loc["Pct_window_ambul"]
assert (ac.p_raw > 0.05).all(), "text says activity structure did not differ"
add("act.pct_c", INT(pw_.Control_mean), pw_.Control_mean, ac_src, "mean % of window in padded walking bouts")
add("act.pct_p", INT(pw_.Patient_mean), pw_.Patient_mean, ac_src)
add("act.r", R(pw_.r_effect), pw_.r_effect, ac_src)
add("act.pe", PE(pw_.p_raw), pw_.p_raw, ac_src)
to_src = os.path.join(XC, "Control_Equivalence_TOST.csv")
to = pd.read_csv(csvp(to_src)).set_index("Metric")
assert abs(to.loc["PSD β", "Margin_d"] - 0.5) < 1e-9 and str(to.loc["PSD β", "Equivalent"]) == "False"
add("tost.pe", PE(to.loc["PSD β", "p_TOST"]), to.loc["PSD β", "p_TOST"], to_src)

print("Classification")
cl_src = os.path.join(OUT_DIR, "classification", "Classification_Summary_NestedCV.csv")
cl = pd.read_csv(csvp(cl_src)).set_index("Model")
add("cl.auc_base", U(cl.loc["MeanHR + RMSSD (LOO)", "AUC"]), cl.loc["MeanHR + RMSSD (LOO)", "AUC"], cl_src)
b = cl.loc["MeanHR + RMSSD + PSD_Beta (LOO)"]
add("cl.auc_beta", U(b.AUC), b.AUC, cl_src)
add("cl.dauc", R(b.Delta_AUC), b.Delta_AUC, cl_src)
add("cl.dauc_lo", R(b.Delta_AUC_lo95), b.Delta_AUC_lo95, cl_src)
add("cl.dauc_hi", R(b.Delta_AUC_hi95), b.Delta_AUC_hi95, cl_src)
add("cl.dauc_pe", PE(b.Delta_AUC_p), b.Delta_AUC_p, cl_src)
add("cl.auc_nested", U(cl.loc["Nested feature-selection (LOO)", "AUC"]), cl.loc["Nested feature-selection (LOO)", "AUC"], cl_src)
nb_ = cl.loc["Nested feature-selection (bootstrap over subjects)"]
add("cl.auc_nested_boot", U(nb_.AUC), nb_.AUC, cl_src, "bootstrap-over-subjects point estimate")
add("cl.auc_nested_lo", U(nb_.AUC_lo95), nb_.AUC_lo95, cl_src)
add("cl.auc_nested_hi", U(nb_.AUC_hi95), nb_.AUC_hi95, cl_src)
assert nb_.AUC_lo95 < cl.loc["MeanHR + RMSSD (LOO)", "AUC"] < nb_.AUC_hi95, "text says the interval includes the baseline"
add("abs.auc_base", U(cl.loc["MeanHR + RMSSD (LOO)", "AUC"], 2), cl.loc["MeanHR + RMSSD (LOO)", "AUC"], cl_src)
add("abs.dauc", U(b.Delta_AUC, 2), b.Delta_AUC, cl_src)

print("Estimator validation")
val_src = os.path.join(OUT_DIR, "validation", "Estimator_Validation.csv")
val = pd.read_csv(csvp(val_src))
add("val.h_min", U(val.True_H.min(), 1), val.True_H.min(), val_src)
add("val.h_max", U(val.True_H.max(), 1), val.True_H.max(), val_src)
ns = sorted(val.N.unique())
assert len(ns) == 2
add("val.n_list", f"{ns[0]:,} and {ns[1]:,}", ns, val_src)
add("val.reps", str(int(val.N_Reps_Valid.min())), val.N_Reps_Valid.min(), val_src)
assert val.N_Reps_Valid.min() == val.N_Reps_Valid.max()
oth = val[val.Estimator != "dfa_a1"].Bias.abs().max()
add("val.other_absmax", "±" + U(np.ceil(oth * 100) / 100, 2), oth, val_src, "max |bias| full DFA, alpha2, MFDFA H2, rounded up")
a1b = val[val.Estimator == "dfa_a1"].Bias
assert (a1b > 0).all()
add("val.a1_min", U(a1b.min(), 2), a1b.min(), val_src)
add("val.a1_max", U(a1b.max(), 2), a1b.max(), val_src)
add("val.a2_med_c", U(pd.to_numeric(C["DFA_Alpha2"]).median(), 2), pd.to_numeric(C["DFA_Alpha2"]).median(), P_C)
add("val.a2_med_p", U(pd.to_numeric(P["DFA_Alpha2"]).median(), 2), pd.to_numeric(P["DFA_Alpha2"]).median(), P_P)

print("Abstract roundings")
add("abs.rmssd_r2", R(gcs.loc["RMSSD (ms)", "r_effect"], 2), gcs.loc["RMSSD (ms)", "r_effect"], gcs_src)
add("abs.rmssd_u2", R(hr.loc["HRV_RMSSD", "r_partial_unadjusted"], 2), hr.loc["HRV_RMSSD", "r_partial_unadjusted"], hr_src)
add("abs.rmssd_a2", R(hr.loc["HRV_RMSSD", "r_partial_HR"], 2), hr.loc["HRV_RMSSD", "r_partial_HR"], hr_src)
add("abs.slope_r2", R(qq.loc["PSD_Beta_NoMayer", "r_native"], 2), qq.loc["PSD_Beta_NoMayer", "r_native"], q_src)
add("g.BETA.r_dup_check", R(gcs.loc["PSD β", "r_effect"]), gcs.loc["PSD β", "r_effect"], gcs_src, "uncorrected slope, for cross-checks")

os.makedirs(DEST, exist_ok=True)
with open(os.path.join(DEST, "v12_facts.json"), "w", encoding="utf-8") as fh:
    json.dump(FACTS, fh, ensure_ascii=False, indent=1)
with open(os.path.join(DEST, "v12_facts_registry.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "printed", "raw", "source", "how"])
    w.writerows(REGISTRY)
print(f"\n{len(FACTS)} facts written to {DEST}")
