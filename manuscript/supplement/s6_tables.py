"""
s6_tables.py: the Supplementary Tables of Supplementary Material v6 (with s6_tables_b, _c and _d), built from the results files.

Each function returns a dict:
    header  list of column headings
    rows    list of rows; a row is a list of cell strings, or {"section": text} for a spanning sub-heading
    note    table note (plain text with *italic* markup allowed)
    landscape  True for wide tables
    sources    results files the table reads (relative to the project folder)

Conventions (stated once in the Supplementary Material and repeated in notes where they matter):
    r is the rank-biserial correlation from a two-sided Mann-Whitney U test with continuity correction,
    r = 1 − 2U/(n1·n2) with U for the reference group (controls, or MIT-BIH in cross-cohort contrasts), so
    positive r means higher in the group compared with it (patients, unless a note says otherwise). Every sign flip relative to a results file is asserted
    against the medians here.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

import s6_common as K
from s6_common import (R, U, P3, RP, SIG, INT, mw, partial, bh, lab, num, read, rel, code_const,
                       FACTS, SNUM, SFNUM, MNUM, RNUM)

TABLES = {}
SRC = {}


def table(name):
    def deco(f):
        TABLES[name] = f
        return f
    return deco


def src(*paths):
    return [rel(p) for p in paths]


def cmp_(f):
    return os.path.join(K.CMP, f)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def fisher(a, n1, b, n2):
    return float(fisher_exact([[a, n1 - a], [b, n2 - b]])[1])


# =========================================================================== S1 onset skip
@table("skip")
def t_skip():
    f = os.path.join(K.OUT, "comparison", "Window_Skip_Sensitivity.csv")
    d = pd.read_csv(K.csvp(f))
    g = pd.read_csv(K.csvp(cmp_("group_comparison_stats.csv"))).set_index("Unnamed: 0")
    rows = []
    SPEC_DIFF = []
    for m in d.Metric.unique():
        sub = d[d.Metric == m].sort_values("Skip_s")
        assert list(sub.Skip_s) == [60, 360, 600]
        assert (sub.N_control == 30).all() and (sub.N_patient == 30).all()
        # The 600 s row must reproduce the primary analysis.
        r600 = sub[sub.Skip_s == 600].iloc[0]
        rr, pp, _, _ = mw(K.C[m], K.P[m])
        spectral = m in ("PSD_Beta_NoMayer", "PSD_Beta_low", "VLF_LF_Ratio")
        assert close(rr, r600.r_effect, 0.005 if spectral else 1e-5), m
        if not spectral:
            assert close(pp, r600.p_raw, 1e-5), m
        else:
            SPEC_DIFF.append((m, rr, r600.r_effect))
        for k, (_, rw) in enumerate(sub.iterrows()):
            rows.append([lab(m) if k == 0 else "", f"{int(rw.Skip_s)}", SIG(rw.Control_mean), SIG(rw.Patient_mean),
                         R(rw.r_effect), P3(rw.p_raw)])
    vl = [x for x in SPEC_DIFF if x[0] == "VLF_LF_Ratio"][0]
    note = ("Every skip is followed by the same 1800 s window, and 600 s is the primary skip. Values are group means, "
            "*r* is the rank-biserial correlation (positive = higher in patients) and *p* is uncorrected; 30 controls "
            "and 30 patients at every skip. The time-domain, entropy and recurrence rows at 600 s reproduce the main "
            "analysis exactly. The three spectral rows are recomputed by window_skip_sensitivity.py on a time base "
            "that places each interval at its end rather than its start, so at 600 s they differ slightly from the "
            f"main-run values (VLF/LF ratio {R(vl[2])} here against {R(vl[1])}).")
    return dict(header=["Measure", "Skip (s)", "Controls, mean", "Patients, mean", "*r*", "*p*"], rows=rows,
                note=note, sources=src(f))


# =========================================================================== S2 measures
@table("measures")
def t_measures():
    pl = "pipeline.py"
    # Quote the code: every setting printed below is asserted against pipeline.py.
    assert code_const(pl, "RR_MIN_MS") == "300" and code_const(pl, "RR_MAX_MS") == "2000"
    assert code_const(pl, "VLF_BAND") == "(0.003, 0.04)" and code_const(pl, "LF_BAND") == "(0.04,  0.15)"
    assert code_const(pl, "HF_BAND") == "(0.15,  0.40)" and code_const(pl, "BETA_FIT_BAND") == "(0.01,  0.15)"
    assert code_const(pl, "MAYER_BAND") == "(0.07, 0.13)" and code_const(pl, "WELCH_NPERSEG") == "512"
    assert code_const(pl, "ENTROPY_M") == "2" and code_const(pl, "ENTROPY_R_FRAC") == "0.20"
    assert code_const(pl, "MSE_MIN_POINTS") == "100" and code_const(pl, "MSE_MAX_SCALE") == "20"
    assert code_const(pl, "RQA_REC_RATE") == "0.10" and code_const(pl, "RQA_DIM") == "2"
    assert code_const(pl, "RQA_DELAY") == "1" and code_const(pl, "RQA_L_MIN") == "2" and code_const(pl, "RQA_V_MIN") == "2"
    assert code_const(pl, "RQA_THEILER") == '"mi"'
    assert code_const(pl, "MFDFA_Q_PRIMARY") == "(1, 2, 3, 4, 5)"
    assert K.code_has(pl, 'fractal_dfa(rr_intervals, scale=np.arange(4, 17)')
    assert K.code_has(pl, 'fractal_dfa(rr_intervals, scale=np.arange(16, 65)')
    assert K.code_has(pl, 'nk.entropy_permutation(rr, dimension=3, delay=1)')
    allc = pd.concat([K.C, K.P, K.M])
    vlf_lo = num(allc.VLF_LowestFreq_Hz)
    assert np.allclose(vlf_lo, 0.0078125) and (num(allc.VLF_NBins) == 5).all()
    assert (num(allc.PSD_Beta_NBins) == 18).all() and (num(allc.PSD_Beta_NoMayer_NBins) == 10).all()
    assert np.allclose(num(allc.PSD_Fit_LowestFreq_Hz), 0.015625) and np.allclose(num(allc.PSD_Fit_HighestFreq_Hz), 0.1484375)
    F = FACTS
    rows = [
        {"section": "Time domain (from the cleaned R-R intervals)"},
        ["Mean NN interval", "Mean of the clean R-R intervals", "–", "Primary"],
        ["Mean heart rate", "60 divided by the mean R-R interval (s), in beats per minute", "–", "Primary"],
        ["SDNN", "Standard deviation of the clean intervals (n − 1 denominator)", "–", "Primary"],
        ["RMSSD", "Root mean square of successive differences", "–", "Primary"],
        ["pNN50", "Percentage of successive differences larger than 50 ms", "–", "Primary"],
        ["SD1", "Poincaré short-axis width, RMSSD/√2", "–", "Primary"],
        ["SD2", "Poincaré long-axis width, √(2·SDNN² − SD1²)", "–", "Primary"],
        ["SD1/SD2", "Ratio of SD1 to SD2", "–", "Primary"],
        {"section": "Frequency domain (Welch spectrum of the series resampled at 4 Hz)"},
        ["VLF power", "Trapezoidal integral of the spectrum over the bins inside the band",
         "Nominal 0.003–0.04 Hz; realized 0.0078–0.039 Hz (5 bins)", "Reported outside the families"],
        ["LF power", "As above", "0.04–0.15 Hz (bins 0.047–0.148 Hz)", "Primary"],
        ["HF power", "As above", "0.15–0.40 Hz (bins 0.156–0.398 Hz)", "Primary"],
        ["Total power", "As above, over the whole range", "0.003–0.40 Hz (bins 0.0078–0.398 Hz)", "Primary"],
        ["LF/HF ratio", "LF power divided by HF power", "–", "Primary"],
        ["VLF/LF ratio", "VLF power divided by LF power", "–", "Reported outside the families"],
        ["Spectral slope, uncorrected", "Negated least-squares slope of log₁₀ power on log₁₀ frequency (1/*f* gives 1)",
         "18 linear bins, 0.0156–0.148 Hz (nominal 0.01–0.15 Hz)", "Primary"],
        ["Spectral slope, baroreflex band excised", "As above with the bins in 0.07–0.13 Hz removed",
         "10 of the 18 bins", "Primary slope in the Results; outside the families"],
        {"section": "Fractal scaling (detrended fluctuation analysis, NeuroKit2 fractal_dfa)"},
        ["DFA α (full range)", "Slope of log fluctuation on log window size; first-order detrending, half-overlapping windows",
         f"NeuroKit2 default scales: {F['dfa.nsc_min']} to {F['dfa.nsc_max']} log-spaced sizes from 10 beats to about N/10 "
         f"(largest {F['dfa.win_min']} to {F['dfa.win_max']} beats)", "Primary"],
        ["DFA α₁", "As above", "Every window size from 4 to 16 beats", "Primary"],
        ["DFA α₂", "As above", "Every window size from 16 to 64 beats", "Primary"],
        ["MFDFA Δh, mean H(q), H(2)", "Generalized Hurst exponents H(q) (Kantelhardt et al., 2002); Δh is max − min over q",
         "q = 1 to 5; 20 log-spaced scales from max(8, N^¼) to N/4; non-overlapping segments; linear detrending",
         "Multifractal family (3 tests)"],
        {"section": "Entropy"},
        ["Sample entropy", "−ln(A/B), Chebyshev distance, self-matches excluded (Richman and Moorman, 2000)",
         "m = 2, r = 0.2 × SD of the series", "Primary"],
        ["Approximate entropy", "Φᵐ − Φᵐ⁺¹, self-matches included (Pincus, 1991)", "m = 2, r = 0.2 × SD",
         "Outside the families (duplicates sample entropy)"],
        ["Permutation entropy", "Shannon entropy of ordinal patterns, normalized by log₂(3!) (Bandt and Pompe, 2002)",
         "Order 3, delay 1 (NeuroKit2 entropy_permutation)", "Primary"],
        ["Multiscale entropy", "Refined composite multiscale entropy, mean over retained scales (Wu et al., 2014)",
         "m = 2; r = 0.2 × SD of the original series at every scale; scales 1 to 20 with at least 100 coarse-grained points",
         "Primary"],
        {"section": "Recurrence (delay embedding, Euclidean distance)"},
        ["Determinism", "Fraction of recurrent points on diagonal lines of length 2 or more",
         "Dimension 2, delay 1; 10% of eligible pairs recurrent (exact count); Theiler window per subject", "Primary"],
        ["Laminarity", "Fraction of recurrent points on vertical lines of length 2 or more", "As above", "Primary"],
        ["Recurrence rate", "Fraction of eligible pairs that are recurrent", "Fixed at 0.10 by construction",
         "Not tested (construction check)"],
        ["Network clustering", "Mean local clustering coefficient of the recurrence matrix read as a network",
         "Same matrix as determinism", "Recurrence-network family (2 tests)"],
        ["Network transitivity", "3 × triangles / connected triples of the same network", "As above",
         "Recurrence-network family (2 tests)"],
        ["Mean network degree", "Mean node degree divided by n − 1", "Fixed at about 0.10 by construction",
         "Not tested (construction check)"],
    ]
    note = ("Settings are read from pipeline.py and checked against the per-subject results files by the build "
            "script (for example the realized frequency bins and the bin counts of the two slopes, identical in all 78 "
            "recordings). Supplementary Methods S2 to S6 describe each estimator in full. The primary family is the 21 "
            "measures of main Table 2; the other families are listed in Supplementary Table "
            f"{SNUM['families']}.")
    return dict(header=["Measure", "Definition", "Settings", "Family"], rows=rows, note=note,
                sources=["pipeline.py"])


# =========================================================================== S3 multiscale entropy
@table("mse")
def t_mse():
    f = cmp_("MSE_Scale_Audit.csv")
    a = pd.read_csv(K.csvp(f)).set_index("Scale_Basis")
    rows = []
    for basis, label in (("all_retained_scales", "Mean over retained scales (primary)"),
                         ("common_scale_set", "Mean over scales 1 to 10 (fixed common set)")):
        rw = a.loc[basis]
        col = "MSE_Mean" if basis == "all_retained_scales" else "MSE_Mean_CommonScales"
        rr, pp, n1, n2 = mw(K.C[col], K.P[col])
        assert close(rr, rw.r_effect, 1e-5) and close(pp, rw.p_raw, 1e-5)
        assert close(num(K.C[col]).mean(), rw.Healthy_Mean, 1e-5)
        rows.append([label, U(rw.Healthy_Mean), U(rw.Psych_Mean), R(rw.r_effect), P3(rw.p_raw)])
    rw = a.loc["retention_check"].iloc[0]
    rows.append(["Scales retained (mean)", U(rw.Healthy_Mean, 1), U(rw.Psych_Mean, 1), R(rw.r_effect), P3(rw.p_raw)])
    rr, pp, _, _ = mw(K.C.MSE_Slope, K.P.MSE_Slope)
    rows.append(["Slope of entropy on scale (per scale)", U(num(K.C.MSE_Slope).mean(), 4), U(num(K.P.MSE_Slope).mean(), 4),
                 R(rr), P3(pp)])
    FACTS["s.mse_slope_r"] = R(rr)
    FACTS["s.mse_slope_pe"] = K.PE(pp)
    rows.append({"section": "Sample entropy of the coarse-grained series at each scale τ"})
    below_all = True
    for t in range(1, 21):
        col = f"MSE_Curve_Tau{t:02d}"
        c, p = num(K.C[col]).dropna(), num(K.P[col]).dropna()
        rr, pp, n1, n2 = mw(c, p)
        below_all &= p.mean() < c.mean()
        rows.append([f"τ = {t}" + ("" if (n1, n2) == (30, 30) else f" ({n1} controls)"), U(c.mean()), U(p.mean()),
                     R(rr), P3(pp)])
    assert below_all, "main text says the patient curve lies below the control curve at every scale"
    ret_c = num(K.C.MSE_Scales_Retained)
    ret_m = num(K.M.MSE_Scales_Retained)
    assert int(ret_c.min()) == 17 and (ret_c == 20).sum() == 29 and int(ret_m.min()) == 18 and (ret_m == 20).sum() == 17
    assert (num(K.P.MSE_Scales_Retained) == 20).all()
    note = ("Means of 30 controls and 30 patients; *r* is the rank-biserial correlation (positive = higher in "
            "patients) and *p* is uncorrected. A scale is retained when the coarse-grained "
            "series holds at least 100 points: all 20 scales in every patient and in 29 controls (one control "
            "retained 17), and in 17 of the 18 MIT-BIH records (one retained 18). The common-set mean uses scales "
            "1 to 10, set as a constant in the code rather than chosen from the data. The slope of the curve is the least-squares "
            "slope of entropy on scale over the retained scales; it is steeper in patients because their curve "
            f"starts lower and rises (Supplementary Figure {SFNUM['mse']}).")
    return dict(header=["Quantity", "Controls", "Patients", "*r*", "*p*"], rows=rows, note=note,
                sources=src(f, os.path.join(K.OUT, "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
                            os.path.join(K.OUT, "psychiatric", "ecg_results_Psychiatric.csv")))


# =========================================================================== S4 stationarity
@table("stationarity")
def t_stationarity():
    f = cmp_("Stationarity_Report.csv")
    s = pd.read_csv(K.csvp(f)).set_index("Statistic")
    rows = []
    for key, label in (("Stat_RA_p", "Reverse-arrangement *p*, median"), ("Stat_VarRatio", "Variance ratio, second/first half, median"),
                       ("Stat_ADF_p", "ADF *p*, median"), ("Stat_KPSS_p", "KPSS *p*, median")):
        rw = s.loc[key]
        rr, pp, _, _ = mw(K.C[key], K.P[key])
        assert close(rr, rw.r_effect, 1e-5) and close(pp, rw.p_raw, 1e-5)
        rows.append([label, U(rw.Control_median, 4 if key == "Stat_ADF_p" else 3), U(rw.Patient_median, 4 if key == "Stat_ADF_p" else 3),
                     R(rw.r_effect), P3(rw.p_raw)])
    def cnt(cond_c, cond_p, label):
        a, b = int(cond_c.sum()), int(cond_p.sum())
        rows.append([label, f"{a} of 30", f"{b} of 30", "–", P3(fisher(a, 30, b, 30))])
        return a, b
    ra = cnt(num(K.C.Stat_RA_p) > 0.05, num(K.P.Stat_RA_p) > 0.05, "Reverse-arrangement test not rejected (*p* > 0.05)")
    assert ra == (int(s.loc["Reverse-arrangement: stationary", "Control_median"]), int(s.loc["Reverse-arrangement: stationary", "Patient_median"]))
    adf = cnt(num(K.C.Stat_ADF_p) < 0.05, num(K.P.Stat_ADF_p) < 0.05, "ADF rejects a unit root (*p* < 0.05)")
    kp = cnt(num(K.C.Stat_KPSS_p) <= 0.01, num(K.P.Stat_KPSS_p) <= 0.01, "KPSS rejects stationarity at its table limit (*p* ≤ 0.01)")
    FACTS.update({"s.ra_c": str(ra[0]), "s.ra_p": str(ra[1]), "s.adf_c": str(adf[0]), "s.adf_p": str(adf[1]),
                  "s.kpss_c": str(kp[0]), "s.kpss_p": str(kp[1])})
    note = ("Reverse-arrangement test on the variances of 40 consecutive segments (Bendat and Piersol, 2010), variance "
            "ratio of the second to the first half, augmented Dickey-Fuller test with lag order chosen by AIC (Dickey "
            "and Fuller, 1979) and KPSS test around a constant (Kwiatkowski et al., 1992), both from statsmodels. The KPSS *p* "
            "value is read from a table bounded at 0.01 and 0.10, so 0.010 means *p* ≤ 0.01. Medians are compared by "
            "Mann-Whitney *U* (*r* positive = higher in patients); counts by Fisher's exact test. All tests are "
            "uncorrected. By KPSS most series in both groups are not stationary over 30 minutes; the groups do not "
            "differ on any statistic.")
    return dict(header=["Statistic", "Controls", "Patients", "*r*", "*p*"], rows=rows, note=note, sources=src(f))


# =========================================================================== S5 quantization controls
@table("quant")
def t_quant():
    f = cmp_("Quantization_Control.csv")
    q = pd.read_csv(K.csvp(f))

    def cell(metric, cond):
        sub = q[(q.Metric == metric) & (q.Condition == cond)]
        if not len(sub):
            return "–"
        rw = sub.iloc[0]
        return f"{R(rw.r_effect)}\n({P3(rw.p_raw)})"

    def nsub(tag):
        sub = q[q.Condition == f"observed_{tag}_matchedsubsample"]
        n = set(zip(sub.N_Healthy.astype(int), sub.N_Psych.astype(int)))
        assert len(n) == 1
        return n.pop()
    n10, n15, n20 = nsub("MQ10"), nsub("MQ15"), nsub("MQ20")
    # matched subsample = subjects whose observed step/RMSSD ratio is at or below the target
    for tag, tgt, n in (("MQ10", 0.10, n10), ("MQ15", 0.15, n15), ("MQ20", 0.20, n20)):
        assert (int((num(K.C.Quant_Step_Over_RMSSD) <= tgt).sum()), int((num(K.P.Quant_Step_Over_RMSSD) <= tgt).sum())) == n, tag
    header = ["Measure", "Observed", "Dithered (one draw)", "Dithered (mean of 20 draws)", "Matched to 0.088",
              "0.10, all subjects", f"0.10 subsample, observed", f"0.10 subsample, matched",
              "0.15, all subjects", "0.15 subsample, observed", "0.15 subsample, matched",
              "0.20, all subjects", "0.20 subsample, observed", "0.20 subsample, matched"]
    rows = []
    rep_shift = []
    for m in ["SampEn", "ApEn", "PermEn", "MSE_Mean", "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"]:
        row = [lab(m), cell(m, "observed"), cell(m, "dither")]
        if f"{m}_Dither_RepMean" in K.C.columns:
            assert (num(K.C[f"{m}_Dither_RepN"]) == 20).all() and (num(K.P[f"{m}_Dither_RepN"]) == 20).all()
            rr, pp, _, _ = mw(K.C[f"{m}_Dither_RepMean"], K.P[f"{m}_Dither_RepMean"])
            r1 = q[(q.Metric == m) & (q.Condition == "dither")].iloc[0].r_effect
            rep_shift.append(abs(rr - r1))
            row.append(f"{R(rr)}\n({P3(pp)})")
        else:
            row.append("–")
        row.append(cell(m, "matched_quant_r088"))
        for tag in ("MQ10", "MQ15", "MQ20"):
            row += [cell(m, f"matched_{tag}_allsubj"), cell(m, f"observed_{tag}_matchedsubsample"),
                    cell(m, f"matched_{tag}_matchedsubsample")]
        rows.append(row)
    # observed condition must reproduce the primary comparison
    for m in ["SampEn", "PermEn", "MSE_Mean", "RQA_DET"]:
        rr, pp, _, _ = mw(K.C[m], K.P[m])
        assert close(rr, q[(q.Metric == m) & (q.Condition == "observed")].iloc[0].r_effect, 1e-5)
    tie = ["Ordinal tie rate", cell("Ordinal_Tie_Rate", "observed"), cell("Ordinal_Tie_Rate", "dither"), "–",
           cell("Ordinal_Tie_Rate", "matched_quant_r088")]
    for tag in ("MQ10", "MQ15", "MQ20"):
        tie += [cell("Ordinal_Tie_Rate", f"matched_{tag}_allsubj"), "–", "–"]
    rows.append(tie)
    rows.append(["Step-to-RMSSD ratio", cell("Quant_Step_Over_RMSSD", "observed")] + ["–"] * 12)
    FACTS["s.dither_rep_maxshift"] = U(max(rep_shift))
    assert FACTS["s.dither_rep_maxshift"] == FACTS["quant.dither_rep_max"], "supplement and main text disagree"
    assert max(rep_shift) < 0.01
    mq_c = int(K.C.Quant_Matched_Applied.astype(str).str.lower().eq("true").sum())
    mq_p = int(K.P.Quant_Matched_Applied.astype(str).str.lower().eq("true").sum())
    assert (mq_c, mq_p) == (int(FACTS["quant.mq_c"]), int(FACTS["quant.mq_p"]))
    note = ("Each cell is the rank-biserial *r* (positive = higher in patients) with its uncorrected *p* below it. "
            "The recorded grid is 1.000 ms. Dithering adds uniform noise one step wide (±0.5 ms) to every interval; "
            "the group comparison uses one seeded draw, and the mean over 20 further draws changes *r* by at most "
            f"{FACTS['s.dither_rep_maxshift']}. Matching to 0.088 re-quantizes each series to a step of 0.088 × its own RMSSD "
            f"when that step exceeds 1 ms, which coarsened {mq_c} controls and {mq_p} patients. For targets of 0.10, 0.15 "
            "and 0.20 the all-subjects column re-quantizes every subject that the target coarsens and keeps the rest; the "
            "subsample columns keep only the subjects whose observed ratio is at or below the target, which the target "
            f"equalizes ({n10[0]} controls and {n10[1]} patients at 0.10, {n15[0]} and {n15[1]} at 0.15, {n20[0]} and "
            f"{n20[1]} at 0.20). The recurrence-network measures were computed under dithering and the 0.088 target "
            "only (rna_controls.py); the sweep covers the other measures. Dithering removes every exact tie, so the "
            "dithered tie rate is zero in every subject. Supplementary Methods "
            f"{MNUM['quantization']} gives the procedures.")
    return dict(header=header, rows=rows, note=note, landscape=True, sources=src(f), font=14)


# =========================================================================== S6 accelerometer settings
@table("accel")
def t_accel():
    f = os.path.join(K.OUT, "accelerometer", "Acquisition_Specification.csv")
    a = pd.read_csv(K.csvp(f)).set_index("Item").Value
    am = "accelerometer_mobility.py"
    assert float(a["gravity_time_constant_s"]) == float(code_const(am, "GRAVITY_TIME_CONSTANT_S")) == 10.0
    assert float(a["motion_smoothing_s"]) == float(code_const(am, "AMBULATION_SMOOTH_S")) == 15.0
    assert float(a["bout_threshold_mult_default"]) == 2.5 and float(a["bout_min_duration_s"]) == 20.0
    assert float(a["bout_pad_pre_s"]) == 30.0 and float(a["bout_pad_post_s"]) == 180.0
    assert int(a["min_acc_samples_in_window"]) == 500 and float(a["min_window_coverage"]) == 0.5
    assert float(code_const(am, "SENSITIVITY_MIN_COVERAGE")) == 0.80
    assert code_const("stationary_reanalysis.py", "MIN_SEGMENT_BEATS") == "500"
    mob_f = os.path.join(K.OUT, "accelerometer", "mobility_coefficients.csv")
    mob = pd.read_csv(K.csvp(mob_f))
    low = mob[mob.coverage < 0.80].Record_ID.tolist()
    assert len(mob) == 60 and mob.mobility_mg.notna().all() and sorted(low) == ["control_47", "treatment_2"]
    FACTS["s.cov_min"] = INT(100 * mob.coverage.min())
    n_med = int(mob.n_acc_samples.median())
    slow = mob[mob.n_acc_samples < 60000]
    assert len(slow) == 1 and slow.iloc[0].Record_ID == "control_45" and close(slow.iloc[0].coverage, 1.0)
    FACTS["s.acc_n_med"] = INT(n_med)
    FACTS["s.acc_slow_n"] = INT(slow.iloc[0].n_acc_samples)
    rows = [
        {"section": "Motion signal"},
        ["Input", "Three accelerometer axes in milli-g with phone timestamps, as exported in the HRV-ACC release"],
        ["Gravity and posture removal", "Each axis minus a Gaussian-smoothed copy of itself (Gaussian SD 10 s), a high-pass filter"],
        ["Motion", "Euclidean norm of the three filtered axes"],
        ["Mobility (covariate)", "Mean motion over the analysis window, in milli-g"],
        ["Window alignment", "The R-R window (600 to 2400 s of beat time) mapped to wall-clock time through the R-R file's timestamps; accelerometer samples selected by timestamp"],
        ["Minimum data", "At least 500 samples and 50% coverage of the window; all 60 subjects qualified "
                         f"(lowest coverage {FACTS['s.cov_min']}%)"],
        ["Samples per window", f"Median {FACTS['s.acc_n_med']}; one control had {FACTS['s.acc_slow_n']}, "
                               "consistent with 25 Hz rather than 100 Hz sampling"],
        ["Coverage sensitivity", "Covariate models repeated without the two subjects below 80% coverage (control 47, patient 2)"],
        {"section": "Walking bouts"},
        ["Smoothing", "Gaussian smoothing of the motion trace (SD 15 s)"],
        ["Threshold", "Smoothed motion above 2.5 times the subject's own median motion"],
        ["Minimum duration", "20 s above threshold"],
        ["Padding", "30 s before and 180 s after each bout, to cover the heart-rate rise and recovery"],
        ["Beat labels", "Each beat takes the label of the nearest accelerometer sample in time"],
        {"section": "Movement-free reanalysis"},
        ["Segment", "Longest contiguous run of non-walking beats in the window"],
        ["Beat-matched version (primary)", "First 600 raw beats of the run, then the artifact filter; runs shorter than 600 beats excluded"],
        ["Duration-matched version", "First 370 s of the run; requires the run to last 370 s and to hold at least 500 beats within it"],
        ["Clean beats", "At least 240 after filtering"],
        {"section": "Not recoverable from the exported files"},
        ["Synchronization", a["synchronisation_verified"].capitalize()],
        ["Sensor range and on-device filtering", "Not documented in the release"],
        ["Axis orientation", "Unknown; the analysis uses the norm, which does not depend on it"],
    ]
    note = ("Settings are those recorded in Acquisition_Specification.csv and the constants of accelerometer_mobility.py "
            "and stationary_reanalysis.py. Supplementary Methods "
            f"{MNUM['movement']} describes the procedure and Supplementary Results {RNUM['activity']} the activity "
            "observed.")
    return dict(header=["Setting", "Value"], rows=rows, note=note, sources=src(f, mob_f))


# =========================================================================== S7 correction families
@table("families")
def t_families():
    f = cmp_("Correction_Family_Inventory.csv")
    inv = pd.read_csv(K.csvp(f)).set_index("Family")
    prim = set(inv.loc["primary", "Members"].split("; "))
    assert prim == set(K.PRIMARY) and int(inv.loc["primary", "N_Tests"]) == 21
    rows = []
    names = {"primary": "Primary measures", "primary_nonredundant": "Primary, definitional duplicates removed",
             "mfdfa": "Multifractal DFA", "recurrence_network": "Recurrence network",
             "surrogate_normalised": "Surrogate-normalized entropy", "construction_fixed": "Fixed by construction",
             "hr_adjusted": "Heart-rate-adjusted partial correlations"}
    where = {"primary": f"Main Table 2; Supplementary Table {SNUM['primaryx']}",
             "primary_nonredundant": f"Supplementary Table {SNUM['nonredundant']}",
             "mfdfa": f"Supplementary Table {SNUM['mfdfa']}", "recurrence_network": f"Supplementary Table {SNUM['rna']}",
             "surrogate_normalised": f"Main Table 4; Supplementary Table {SNUM['surrogate']}",
             "construction_fixed": "Not tested (Supplementary Methods " + str(MNUM['recurrence']) + ")",
             "hr_adjusted": f"Main Table 3; Supplementary Table {SNUM['hr']}"}
    for fam in ["primary", "primary_nonredundant", "hr_adjusted", "mfdfa", "recurrence_network", "surrogate_normalised",
                "construction_fixed"]:
        rw = inv.loc[fam]
        special = {"SampEn raw": "Sample entropy, raw", "SampEn Z": "Sample entropy, Z-score",
                   "DFA_Alpha1 raw": "DFA α₁, raw", "DFA_Alpha1 Z": "DFA α₁, Z-score",
                   "MSE_Mean raw": "Multiscale entropy, raw", "MSE_Mean Z": "Multiscale entropy, Z-score",
                   "RQA_REC": "Recurrence rate", "RNA_MeanDegree": "Mean network degree"}
        mem = "; ".join(special.get(m, lab(m)) for m in rw.Members.split("; "))
        assert all(m in special or m in K.LABEL for m in rw.Members.split("; ")), rw.Members
        rows.append([names[fam], str(int(rw.N_Tests)), "BH" if rw.Corrected == "BH" else "None", mem, where[fam]])
    # Families that the inventory file does not list but whose files carry adjusted p values. Each is checked by
    # recomputing Benjamini-Hochberg over the file's raw p values.
    mw_f = cmp_("Mayer_Wave_Control.csv")
    mwc = pd.read_csv(K.csvp(mw_f))
    assert np.allclose(bh(mwc.p_raw.values), mwc.p_adj.values, atol=2e-5)
    rows.append(["Slope estimators and baroreflex peak", str(len(mwc)), "BH",
                 "Five slope estimators, peak model on clean fits only, fitted peak amplitude and frequency, LF power",
                 f"Main Table 4; Supplementary Table {SNUM['estimators']}"])
    bs_f = cmp_("Beta_Band_Sweep.csv")
    bs = pd.read_csv(K.csvp(bs_f))
    assert np.allclose(bh(bs.p_raw.values), bs.p_adj.values, atol=2e-5)
    rows.append(["Slope fitting ranges", str(len(bs)), "BH", "Six fitted ranges from 0.016–0.047 Hz to 0.008–0.398 Hz",
                 f"Main Table 4; Supplementary Table {SNUM['bands']}"])
    cc_f = os.path.join(K.XC, "Control_vs_Control_Stats.csv")
    cc = pd.read_csv(K.csvp(cc_f))
    assert np.allclose(bh(cc.MWU_p_raw.values), cc.MWU_p_adj.values, atol=2e-5)
    rows.append(["Control cohorts compared", str(len(cc)), "BH", "21 primary measures, approximate entropy, three MFDFA and two network measures",
                 f"Supplementary Table {SNUM['controls']}"])
    mit_f = os.path.join(K.OUT, "comparison", "healthy_mitbih_vs_psych", "group_comparison_stats.csv")
    mit = pd.read_csv(K.csvp(mit_f))
    assert len(mit) == 21 and np.allclose(bh(mit.p_raw.values), mit.p_adj.values, atol=2e-5)
    rows.append(["Patients against MIT-BIH controls", "21", "BH", "The 21 primary measures", f"Supplementary Table {SNUM['crosscohort']}"])
    kw_f = os.path.join(K.XC, "ThreeGroup_KruskalWallis.csv")
    kw = pd.read_csv(K.csvp(kw_f))
    assert np.allclose(bh(kw.KW_p_raw.values), kw.KW_p_adj.values, atol=2e-5)
    dn_f = os.path.join(K.XC, "ThreeGroup_Dunn_Posthoc.csv")
    dn = pd.read_csv(K.csvp(dn_f))
    assert np.allclose(bh(dn.p_raw.values), dn.p_adj.values, atol=2e-5)
    rows.append(["Three-group omnibus tests", str(len(kw)), "BH", "As for the control cohorts", f"Supplementary Table {SNUM['threegroup']}"])
    rows.append(["Three-group post hoc tests", str(len(dn)), "BH", f"{len(kw)} measures × 3 pairs", f"Supplementary Table {SNUM['threegroup']}"])
    rows.append(["Sensitivity analyses", "–", "None",
                 "Onset skips, tolerances, DFA bounds, embeddings, line lengths, quantization controls, fixed beat counts, "
                 "movement-free segments, subgroups, covariate models, model forms",
                 "Reported with uncorrected *p* and compared with the primary result"])
    eff_f = cmp_("Effective_Number_Of_Tests.csv")
    e = pd.read_csv(K.csvp(eff_f)).iloc[0]
    FACTS["s.meff"] = U(e.N_Effective_LiJi, 1)
    FACTS["s.meff_alpha"] = U(e.Alpha_Effective, 4)
    note = (f"BH, Benjamini-Hochberg false discovery rate correction at 0.05 within the family. Adjusted *p* values "
            "in every family were recomputed from the raw values when this table was built. Effective number of "
            f"independent tests in the primary family (Li and Ji, 2005), from the eigenvalues of the pooled Spearman "
            f"correlation matrix of the 21 measures: {FACTS['s.meff']} (leading eigenvalues {e.Leading_Eigenvalues.replace(';', ',')}), which "
            f"would give a per-test α of {FACTS['s.meff_alpha']} under a Bonferroni-type correction. It is reported as a "
            "sensitivity analysis only; the primary correction remains BH over the nominal 21 tests.")
    return dict(header=["Family", "Tests", "Correction", "Members", "Where reported"], rows=rows, note=note,
                landscape=True, sources=src(f, mw_f, bs_f, cc_f, mit_f, kw_f, dn_f, eff_f))


# =========================================================================== S8 primary measures extended
@table("primaryx")
def t_primaryx():
    g_f = cmp_("group_comparison_stats.csv")
    w_f = cmp_("Parametric_Comparison_Welch_t.csv")
    d_f = cmp_("Distribution_Diagnostics.csv")
    g = pd.read_csv(K.csvp(g_f)).rename(columns={"Unnamed: 0": "L"})
    w = pd.read_csv(K.csvp(w_f)).rename(columns={"Unnamed: 0": "L"}).set_index("L")
    dd = pd.read_csv(K.csvp(d_f)).set_index("Metric")
    g["col"] = g.L.map(K.CSV_LABEL)
    assert set(g.col) == set(K.PRIMARY)
    g = g.set_index("col").loc[K.PRIMARY]
    rows = []
    for col, rw in g.iterrows():
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        assert close(rr, rw.r_effect, 1e-5) and close(pp, rw.p_raw, 1e-5)
        wl = rw.L
        if wl in w.index:
            wr = w.loc[wl]
            assert np.sign(wr.Cohen_d) == np.sign(wr.Hedges_g) and wr.d_CI_low <= wr.Cohen_d <= wr.d_CI_high
            assert np.sign(wr.Cohen_d) == np.sign(wr.Psych_Mean - wr.Healthy_Mean)      # positive = higher in patients
            wcell = (f"{U(wr.Welch_t, 2)}", P3(wr.p_adj), f"{R(wr.Cohen_d, 2)} [{R(wr.d_CI_low, 2)}, {R(wr.d_CI_high, 2)}]",
                     f"{R(wr.Hedges_g, 2)}")
        else:
            wcell = ("–", "–", "–", "–")
        if wl not in w.index:
            assert col == "HRV_pNN50"
        dg = dd.loc[wl]
        nd = K.DEC(rw.Healthy_Median)
        F = lambda x: K.FMT(x, nd)
        rows.append([lab(col), f"{F(rw.Healthy_Median)} ({F(rw.Healthy_Q1)}–{F(rw.Healthy_Q3)})",
                     f"{F(rw.Psych_Median)} ({F(rw.Psych_Q1)}–{F(rw.Psych_Q3)})",
                     f"{R(rw.r_effect)} [{R(rw.r_CI_low, 2)}, {R(rw.r_CI_high, 2)}]", P3(rw.p_raw), P3(rw.p_adj),
                     *wcell, f"{P3(dg.Shapiro_p_Healthy)}; {P3(dg.Shapiro_p_Psych)}", P3(dg.Levene_p)])
    n_welch_sig = int(w.Significant.sum())
    FACTS["s.welch_nsig"] = str(n_welch_sig)
    det_w = w.loc["RQA Determinism"]
    assert bool(det_w.Significant) and not bool(g.loc["RQA_DET", "Significant"])
    sd12_w = w.loc["SD1/SD2"]
    assert not bool(sd12_w.Significant) and bool(g.loc["HRV_SD1SD2", "Significant"])
    note = ("Medians with interquartile ranges. *r* is the rank-biserial correlation (positive = higher in patients) "
            "with its bootstrap 95% interval (2,000 resamples within each group, percentile method). *p* is from the "
            "Mann-Whitney *U* test and *p* adj from Benjamini-Hochberg correction over the 21 measures. Welch *t* "
            "tests are corrected over the 20 measures they cover (pNN50, with 7 patients at exactly zero, is "
            "excluded); *d* is Cohen's *d* with the pooled SD and its approximate 95% interval, and *g* is Hedges' "
            "*g*, the small-sample correction of *d* (both positive = higher in patients). The Welch tests agree with the "
            f"Mann-Whitney tests except for two measures near the threshold: determinism is significant by Welch "
            f"(*p* adj = {P3(det_w.p_adj)}) but not by Mann-Whitney (*p* adj = {P3(g.loc['RQA_DET', 'p_adj'])}), and "
            f"SD1/SD2 the reverse ({P3(sd12_w.p_adj)} against {P3(g.loc['HRV_SD1SD2', 'p_adj'])}). Normality was "
            "checked by Shapiro-Wilk tests within each group (controls; patients) and equality of variance by the "
            "median-centered Levene test (Brown and Forsythe, 1974); because several measures fail one or the "
            "other, the rank test is primary throughout.")
    return dict(header=["Measure", "Controls, median (IQR)", "Patients, median (IQR)", "*r* [95% CI]", "*p*", "*p* adj",
                        "Welch *t*", "Welch *p* adj", "Cohen's *d* [95% CI]", "Hedges *g*", "Shapiro-Wilk *p*", "Levene *p*"],
                rows=rows, note=note, landscape=True, sources=src(g_f, w_f, d_f), font=14)


# =========================================================================== S9 power
@table("power")
def t_power():
    f1 = cmp_("Minimum_Detectable_Effect.csv")
    f2 = cmp_("Minimum_Detectable_Effect_Subgroups.csv")
    a = pd.read_csv(K.csvp(f1))
    s = pd.read_csv(K.csvp(f2))
    from scipy.stats import norm, nct, t as tdist
    from scipy.optimize import brentq

    def nct_power(d, n1=30, n2=30):
        df = n1 + n2 - 2
        tc = tdist.ppf(0.975, df)
        nc = d * np.sqrt(n1 * n2 / (n1 + n2))
        return 1 - nct.cdf(tc, df, nc) + nct.cdf(-tc, df, nc)
    d_nct = brentq(lambda d: nct_power(d) - 0.8, 0.1, 3.0)
    FACTS["s.mde_nct_d"] = U(d_nct)
    FACTS["s.mde_nct_r"] = U(2 * norm.cdf(d_nct / np.sqrt(2)) - 1)
    rows = []
    for _, rw in a.iterrows():
        d = (norm.ppf(0.975) + norm.ppf(rw.Power)) * np.sqrt(1 / rw.N_Healthy + 1 / rw.N_Psych)
        assert close(round(d, 3), rw.Min_Detectable_d, 1e-9)
        rows.append([f"Main comparison, {int(rw.Power * 100)}% power", f"{int(rw.N_Healthy)} and {int(rw.N_Psych)}",
                     U(rw.Min_Detectable_d), U(rw.Min_Detectable_r), U(rw.Min_Detectable_r_pointbiserial)])
    for _, rw in s.iterrows():
        if rw.Contrast == "Main comparison":
            continue
        rows.append([rw.Contrast.replace("vs", "against"), f"{int(rw.N_A)} and {int(rw.N_B)}", U(rw.Min_Detectable_d),
                     U(rw.Min_Detectable_r_rankbiserial), U(rw.Min_Detectable_r_pointbiserial)])
    note = ("Smallest standardized difference detectable at two-sided α = 0.05 with 80% power (90% where stated), "
            "by the normal approximation *d* = (*z*₀.₉₇₅ + *z*_power) × √(1/*n*₁ + 1/*n*₂). The rank-biserial "
            "equivalent is 2Φ(*d*/√2) − 1 and the point-biserial equivalent *d*/√(*d*² + 4). The noncentral *t* "
            f"distribution gives slightly larger values (*d* = {FACTS['s.mde_nct_d']}, *r* = {FACTS['s.mde_nct_r']} for the main comparison). Effects below these "
            "sizes are reported but a null result is not read as evidence of no difference.")
    return dict(header=["Contrast", "*n*", "*d*", "Rank-biserial *r*", "Point-biserial *r*"], rows=rows, note=note,
                sources=src(f1, f2))


# =========================================================================== S10 classifier
@table("classifier")
def t_classifier():
    f = os.path.join(K.OUT, "classification", "Classification_Summary_NestedCV.csv")
    pv = os.path.join(K.OUT, "classification", "Classification_Provenance.csv")
    c = pd.read_csv(K.csvp(f)).set_index("Model")
    prov = pd.read_csv(K.csvp(pv)).set_index("Item").Value
    assert int(prov["bootstrap_replicates_auc_ci"]) == 2000 and int(prov["bootstrap_replicates_delta_auc"]) == 10000
    assert int(prov["permutation_replicates"]) == 1000 and float(prov["regularisation_C"]) == 1.0 and int(prov["random_seed"]) == 42
    order = [("MeanHR + RMSSD (LOO)", "Mean heart rate + RMSSD (pre-specified baseline)", "Leave-one-out", "2"),
             ("MeanHR + RMSSD + PSD_Beta (LOO)", "Baseline + uncorrected slope (pre-specified)", "Leave-one-out", "3"),
             ("Nested feature-selection (LOO)", "Nested selection from 22 measures", "Leave-one-out", None),
             ("Nested feature-selection (bootstrap over subjects)", "Nested selection from 22 measures", "Bootstrap, out of bag", None),
             ("Nested feature-selection (repeated 5-fold)", "Nested selection from 22 measures", "5-fold × 100", None),
             ("Fixed 5-domain (LOO)", "RMSSD, slope, DFA α₂, sample entropy, determinism (fixed)", "Leave-one-out", "5"),
             ("Artifact-free features (LOO)", "All 18 measures left after removing sample, approximate and multiscale entropy and determinism (fixed, no selection)",
              "Leave-one-out; interval from the out-of-bag bootstrap", None)]
    rows = []
    for key, label, val, nfeat in order:
        rw = c.loc[key]
        ci = "–"
        if np.isfinite(rw.AUC_lo95):
            ci = f"{U(rw.AUC_lo95)} to {U(rw.AUC_hi95)}"
            if "repeated" in key:
                ci += " (spread across repeats)"
        acc = "–" if not np.isfinite(rw.Accuracy) else f"{U(rw.Accuracy)} / {U(rw.Sensitivity)} / {U(rw.Specificity)}"
        if key == "Artifact-free features (LOO)":
            # nested_classifier.py fits this model with do_selection=False on a fixed index: Median_features is the
            # size of the fixed set, not a median over folds
            assert K.code_has("nested_classifier.py", "probs_af = nested_loo(X, y, features, do_selection=False, fixed_idx=idx_af)")
            assert K.code_has("nested_classifier.py", 'ARTIFACT_DRIVEN = ["SampEn", "ApEn", "RQA_DET", "MSE_Mean"]')
            assert int(rw.Median_features) == 22 - 4
            nfeat = str(int(rw.Median_features))
        rows.append([label, val, nfeat or f"median {int(rw.Median_features)}", U(rw.AUC), ci, acc])
    dl = c.loc["MeanHR + RMSSD + PSD_Beta (LOO)"]
    rows.append({"section": f"Increment from adding the slope: ΔAUC {R(dl.Delta_AUC)} (bootstrap 95% interval "
                            f"{R(dl.Delta_AUC_lo95)} to {R(dl.Delta_AUC_hi95)}, *p* = {U(dl.Delta_AUC_p)}; 10,000 replicates, both models refitted, scored out of bag)"})
    nl = c.loc["Nested feature-selection (LOO)"]
    rows.append({"section": f"Permutation test of the nested model: null mean AUC {U(nl.Perm_null_AUC_mean)}, *p* ≤ 0.001 (1,000 label shuffles, whole pipeline rerun)"})
    note = ("Logistic regression with an L2 penalty (C = 1, fixed beforehand), features standardized within each "
            "training fold. The nested models select features inside each training fold (Mann-Whitney *p* < 0.05 in "
            "the training data, at least three) from 22 measures: the 21 primary measures and approximate entropy. "
            "The construction-fixed recurrence rate is excluded by name. The bootstrap interval reruns the whole pipeline in 2,000 "
            "resamples of subjects and scores the out-of-bag subjects; the repeated 5-fold range is the spread of "
            "pooled AUCs across 100 repeats, not a confidence interval. The leave-one-out AUC of "
            f"{U(c.loc['Nested feature-selection (LOO)'].AUC)} and the bootstrap AUC of "
            f"{U(c.loc['Nested feature-selection (bootstrap over subjects)'].AUC)} estimate the same model differently and each carries its own label. The "
            "reduced model in the last row uses every remaining measure without selection, its interval comes from "
            "2,000 bootstrap resamples of subjects with the model refitted and scored out of bag, and its excluded "
            "set was chosen from this dataset's own control analyses. Acc/Sens/Spec, "
            "accuracy, sensitivity and specificity at a probability threshold of 0.5. Every subject is either a "
            "hospitalized patient or a volunteer, so this is a methods demonstration and not a diagnostic test.")
    return dict(header=["Model", "Validation", "Features", "AUC", "95% interval", "Acc / Sens / Spec"], rows=rows,
                note=note, landscape=True, sources=src(f, pv))
