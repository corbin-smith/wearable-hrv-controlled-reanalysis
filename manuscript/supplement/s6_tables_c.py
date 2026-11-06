"""s6_tables_c.py: Supplementary Tables for slope estimators, entropy parameters, spectral decomposition, fitted
ranges, the DFA identity, multifractal DFA, covariates, activity, movement-free segments, subgroups, the MIT-BIH
comparisons and age and sex."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import s6_common as K
from s6_common import (R, U, P3, SIG, INT, mw, partial, bh, lab, num, code_const, FACTS, SNUM, SFNUM, MNUM, RNUM)
from s6_tables import table, src, cmp_, close, fisher


# =========================================================================== S24 slope estimators
@table("estimators")
def t_estimators():
    f = cmp_("Mayer_Wave_Control.csv")
    m = pd.read_csv(K.csvp(f)).set_index("Estimator")
    names = {"PSD_Beta": "Linear-bin least squares, 18 bins (uncorrected slope)",
             "PSD_Beta_LogBin": "Log-binned least squares, 9 log bins",
             "PSD_Beta_NoMayer": "Linear bins with 0.07–0.13 Hz excised, 10 bins (baroreflex-excised slope)",
             "PSD_Beta_Specparam": "Aperiodic line plus one Gaussian peak",
             "PSD_Beta_IRASA": "IRASA fractal component, log-binned",
             "PSD_Beta_Specparam_CleanFits": "Aperiodic plus peak, fits not at a parameter bound",
             "Mayer_Peak_Amp": "Fitted peak amplitude (log₁₀ units)", "Mayer_Peak_Freq": "Fitted peak frequency (Hz)",
             "LF_Power_ms2": "LF power (ms²)"}
    rows = [{"section": "Estimators and the fitted baroreflex peak (Benjamini-Hochberg over these nine tests)"}]
    for key in ["PSD_Beta", "PSD_Beta_LogBin", "PSD_Beta_NoMayer", "PSD_Beta_Specparam", "PSD_Beta_IRASA",
                "PSD_Beta_Specparam_CleanFits", "Mayer_Peak_Amp", "Mayer_Peak_Freq", "LF_Power_ms2"]:
        rw = m.loc[key]
        if key != "PSD_Beta_Specparam_CleanFits":
            rr, pp, n1, n2 = mw(K.C[key], K.P[key])
            assert close(rr, rw.r_effect, 1e-5) and (n1, n2) == (30, 30)
        else:
            okc = ~K.C.Specparam_At_Bound.astype(str).str.lower().eq("true")
            okp = ~K.P.Specparam_At_Bound.astype(str).str.lower().eq("true")
            rr, pp, n1, n2 = mw(K.C.PSD_Beta_Specparam[okc], K.P.PSD_Beta_Specparam[okp])
            assert close(rr, rw.r_effect, 1e-5) and (n1, n2) == (int(rw.N_Healthy), int(rw.N_Psych)) == (22, 17)
        rows.append([names[key], f"{int(rw.N_Healthy)}, {int(rw.N_Psych)}", SIG(rw.Healthy_Mean), SIG(rw.Psych_Mean),
                     R(rw.r_effect), P3(rw.p_raw), P3(rw.p_adj)])
    rows.append({"section": "Further sensitivity estimates of the slope (uncorrected *p*)"})
    extra = [("PSD_Beta_NoMayer_LogBin", "Excised, log-binned"),
             ("PSD_Beta_LinDetrend", "Uncorrected, linear detrending within Welch segments"),
             ("PSD_Beta_NoMayer_LinDetrend", "Excised, linear detrending within Welch segments"),
             ("PSD_Beta_nperseg256", "Uncorrected, 256-sample segments (0.0156 Hz bins, 9 bins)"),
             ("PSD_Beta_nperseg1024", "Uncorrected, 1024-sample segments (0.0039 Hz bins, 36 bins)"),
             ("PSD_Beta_Linear", "Uncorrected, linear rather than cubic interpolation")]
    assert (num(K.C.PSD_Beta_nperseg256_nbins) == 9).all() and (num(K.C.PSD_Beta_nperseg1024_nbins) == 36).all()
    ex_r = []
    for col, label in extra:
        rr, pp, n1, n2 = mw(K.C[col], K.P[col])
        ex_r.append(rr)
        rows.append([label, f"{n1}, {n2}", SIG(num(K.C[col]).mean()), SIG(num(K.P[col]).mean()), R(rr), P3(pp), "–"])
    thr = float(code_const("pipeline.py", "PSD_BETA_R2_THRESHOLD"))
    okc, okp = num(K.C.PSD_Beta_R2) >= thr, num(K.P.PSD_Beta_R2) >= thr
    rr, pp, n1, n2 = mw(K.C.PSD_Beta[okc], K.P.PSD_Beta[okp])
    sens = pd.read_csv(K.csvp(cmp_("PSD_Beta_Sensitivity.csv"))).iloc[0]
    assert close(rr, sens.r_filtered, 1e-5)
    ex_r.append(rr)
    rows.append([f"Uncorrected, subjects whose fit has *R*² ≥ {thr:.2f}", f"{n1}, {n2}", SIG(num(K.C.PSD_Beta[okc]).mean()),
                 SIG(num(K.P.PSD_Beta[okp]).mean()), R(rr), P3(pp), "–"])
    FACTS["s.est_extra_min"] = R(min(ex_r))
    FACTS["s.est_extra_max"] = R(max(ex_r))
    FACTS["s.r2_n_c"], FACTS["s.r2_n_p"] = str(n1), str(n2)
    assert min(ex_r) > 0.6
    rows.append({"section": "Model-free baroreflex residual: mean log₁₀ excess over the aperiodic line inside 0.07–0.13 Hz"})
    rf = os.path.join(K.OUT, "comparison", "Mayer_Residual_PerSubject.csv")
    rs = pd.read_csv(K.csvp(rf))
    rc, rp = rs[rs.Group == "control"], rs[rs.Group == "patient"]
    assert len(rc) == 30 and len(rp) == 30
    # its excised-slope column reproduces the pipeline value
    assert np.allclose(np.sort(rc.Beta_NotchFit.values), np.sort(num(K.C.PSD_Beta_NoMayer).values), atol=2e-5)
    for col, label in (("Mayer_Resid_Mean", "Mean residual in the band"), ("Mayer_Resid_Max", "Largest residual in the band")):
        rr, pp, n1, n2 = mw(rc[col], rp[col])
        rows.append([label, "30, 30", U(rc[col].mean()), U(rp[col].mean()), R(rr), P3(pp), "–"])
        FACTS[f"s.{col}_r"] = R(rr)
        FACTS[f"s.{col}_pe"] = K.PE(pp)
    FACTS["s.resid_c"] = U(rc.Mayer_Resid_Mean.mean())
    FACTS["s.resid_p"] = U(rp.Mayer_Resid_Mean.mean())
    rr, pp, _, _ = mw(K.C.Mayer_Peak_Width, K.P.Mayer_Peak_Width)
    rows.append(["Fitted peak width (log₁₀ frequency units)", "30, 30", U(num(K.C.Mayer_Peak_Width).mean()),
                 U(num(K.P.Mayer_Peak_Width).mean()), R(rr), P3(pp), "–"])
    nb_c = int(K.C.Specparam_At_Bound.astype(str).str.lower().eq("true").sum())
    nb_p = int(K.P.Specparam_At_Bound.astype(str).str.lower().eq("true").sum())
    assert (nb_c, nb_p) == (8, 13)
    note = ("Means of each group; *r* is the rank-biserial correlation (positive = higher in patients). The peak model "
            "fits log₁₀ power = *c* − β·log₁₀ *f* + *A*·exp(−½((log₁₀ *f* − μ)/σ)²) over 0.0156–0.148 Hz, with the "
            "peak center limited to 0.06–0.14 Hz and its width σ to 0.03–0.25; 8 control and 13 patient fits ended "
            "at a bound of the center or width and are excluded from the clean-fit row. IRASA is biased over a band this "
            "narrow and is a sensitivity estimate only. The model-free residual fits a line to each subject's "
            "spectrum over 0.0156–0.148 Hz with 0.07–0.13 Hz excluded and measures how far the spectrum rises above it "
            "inside the excluded band; positive values mean a peak. It is larger in controls, so the resonance "
            "flattens the control slope and inflates the uncorrected difference rather than creating it. Supplementary "
            f"Methods {MNUM['spectral']} describes every estimator.")
    return dict(header=["Estimator", "*n* (controls, patients)", "Controls, mean", "Patients, mean", "*r*", "*p*", "*p* adj"],
                rows=rows, note=note, landscape=True, sources=src(f, rf, cmp_("PSD_Beta_Sensitivity.csv")))


# =========================================================================== S25 entropy parameters
@table("tolerance")
def t_tolerance():
    f = cmp_("Entropy_Tolerance_Sensitivity.csv")
    e = pd.read_csv(K.csvp(f))
    rows = [{"section": "Sample entropy, embedding dimension 2, tolerance r × SD of the series"}]
    for _, rw in e.iterrows():
        col = f"SampEn_r{int(round(rw.Tolerance_r_x_SD * 100)):02d}"
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        assert close(rr, rw.r_effect, 1e-5)
        tag = " (primary)" if close(rw.Tolerance_r_x_SD, 0.20) else ""
        rows.append([f"{rw.Tolerance_r_x_SD:.2f} × SD{tag}", U(rw.Control_mean), U(rw.Patient_mean), R(rw.r_effect), P3(rw.p_raw)])
    rows.append({"section": "Permutation entropy, delay 1, order (pattern length)"})
    pr = []
    for col, label in (("PermEn", "3 (primary)"), ("PermEn_m4", "4"), ("PermEn_m5", "5")):
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        pr.append((rr, pp))
        rows.append([label, U(num(K.C[col]).mean()), U(num(K.P[col]).mean()), R(rr), P3(pp)])
    assert all(p < 0.05 and r > 0 for r, p in pr)
    FACTS["s.pe4_r"], FACTS["s.pe5_r"] = R(pr[1][0]), R(pr[2][0])
    note = ("Group means; *r* is the rank-biserial correlation (positive = higher in patients) with uncorrected *p*. "
            "Sample entropy keeps its direction and significance at every tolerance and permutation entropy at every "
            "order; neither choice changes a conclusion. Permutation entropy is normalized by the log of the number "
            "of possible patterns, so values at different orders are on the same 0 to 1 scale.")
    return dict(header=["Setting", "Controls, mean", "Patients, mean", "*r*", "*p*"], rows=rows, note=note, sources=src(f))


# =========================================================================== S26 VLF decomposition
@table("vlf")
def t_vlf():
    f = cmp_("VLF_Decomposition.csv")
    v = pd.read_csv(K.csvp(f))
    names = {"VLF_Share": "VLF share of total power", "LF_Share": "LF share of total power", "HF_Share": "HF share of total power"}
    rows = []
    for _, rw in v.iterrows():
        nm = names.get(rw.Metric, lab(rw.Metric))
        if rw.Metric in K.C.columns:
            rr, pp, _, _ = mw(K.C[rw.Metric], K.P[rw.Metric])
            assert close(rr, rw.r_effect, 1e-5)
        pct = rw.Metric.endswith("Share")
        fmt = (lambda x: f"{U(100 * x, 1)}%") if pct else SIG
        rows.append([nm, fmt(rw.Healthy_Median), fmt(rw.Psych_Median), f"{R(rw.Pct_Median_Change, 1)}%", R(rw.r_effect), P3(rw.p_raw)])
    # shares: band power over total power; the three bands do not fill the total (gaps between bands)
    CP = pd.concat([K.C, K.P])
    tot = (num(CP.VLF_Power_ms2) + num(CP.LF_Power_ms2) + num(CP.HF_Power_ms2)) / num(CP.Total_Power_ms2)
    FACTS["s.bandsum_lo"], FACTS["s.bandsum_hi"] = INT(100 * tot.min()), INT(100 * tot.max())
    note = ("Medians of 30 controls and 30 patients; change is the patient median relative to the control median. "
            "*r* is the rank-biserial correlation (positive = higher in patients), uncorrected. VLF power is "
            "integrated over the five resolved bins from 0.0078 to 0.039 Hz, not from the nominal 0.003 Hz. Each "
            "share is the band's power divided by total power; the three bands do not quite fill the total, because "
            "each band is integrated over its own bins and the narrow gaps between bands are left out, so the shares "
            f"sum to {FACTS['s.bandsum_lo']}% to {FACTS['s.bandsum_hi']}%. VLF power and the VLF/LF ratio are reported "
            "outside the correction families.")
    return dict(header=["Measure", "Controls, median", "Patients, median", "Change", "*r*", "*p*"], rows=rows, note=note,
                sources=src(f))


# =========================================================================== S27 fitted ranges
@table("bands")
def t_bands():
    f = cmp_("Beta_Band_Sweep.csv")
    b = pd.read_csv(K.csvp(f)).set_index("Band")
    rows = []
    lab_b = {"verylow": "0.016–0.047 Hz", "low": "0.016–0.062 Hz", "lowmid": "0.016–0.094 Hz",
             "nomayerx": "0.016–0.148 Hz (uncorrected slope)", "mayerexc": "0.016–0.148 Hz, 0.07–0.13 Hz excised",
             "wideband": "0.008–0.398 Hz (whole spectrum)"}
    for key in ["verylow", "low", "lowmid", "nomayerx", "mayerexc", "wideband"]:
        rw = b.loc[key]
        col = f"PSD_Beta_{key}"
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        assert close(rr, rw.r_effect, 1e-5)
        lo_hz = f"{rw.Actual_low_Hz:.4f}".rstrip("0")
        assert (num(K.C[f"{col}_NBins"]) == rw.N_bins).all()
        rows.append([lab_b[key], str(int(rw.N_bins)), f"{U(rw.Control_mean)} ({U(rw.Control_SD)})",
                     f"{U(rw.Patient_mean)} ({U(rw.Patient_SD)})", f"{R(rw.r_effect)} [{R(rw.r_CI_low, 2)}, {R(rw.r_CI_high, 2)}]",
                     P3(rw.p_raw), P3(rw.p_adj)])
    assert np.allclose(num(K.C.PSD_Wide_LowestFreq_Hz), 0.0078125)
    rows.append({"section": "Other fitted ranges and interpolation (outside the family; uncorrected *p*)"})
    rs = pd.read_csv(K.csvp(cmp_("PSD_Beta_Range_Sensitivity.csv"))).set_index("Column")
    for col, label, nb in (("PSD_Beta_Wide", "0.008–0.148 Hz", 19), ("PSD_Beta_Linear", "0.016–0.148 Hz, linear interpolation", 18)):
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        c, p = num(K.C[col]), num(K.P[col])
        assert close(rr, rs.loc[col, "r_effect"], 1e-5) and rs.loc[col, "H_Mean_SD"] == f"{c.mean():.3f} ± {c.std():.3f}"
        rows.append([label, str(nb), f"{U(c.mean())} ({U(c.std())})", f"{U(p.mean())} ({U(p.std())})", R(rr), P3(pp), "–"])
    # bin count of the 0.008-0.148 Hz fit: bins at k/128 Hz from 1/128 up to 0.15
    assert len([k for k in range(1, 40) if 0.003 <= k / 128 <= 0.15]) == 19
    note = ("Ranges are the realized Welch bins (0.0078 Hz apart) rather than the nominal limits; for example the "
            "nominal 0.01–0.07 Hz range starts at the first bin, 0.0156 Hz, and ends at 0.0625 Hz. Means (SD); *r* is "
            "the rank-biserial correlation (positive = steeper in patients) with its bootstrap 95% interval (2,000 "
            "resamples within groups); *p* adj is Benjamini-Hochberg over the six ranges of the sweep. The two lowest "
            "ranges lie below the baroreflex resonance and below any respiratory contribution, and the difference holds "
            "in both. Over the whole spectrum, where the 32 HF bins dominate the fit, it disappears and its sign "
            "reverses without reaching significance.")
    return dict(header=["Fitted range", "Bins", "Controls, mean (SD)", "Patients, mean (SD)", "*r* [95% CI]", "*p*", "*p* adj"],
                rows=rows, note=note, landscape=True, sources=src(f, cmp_("PSD_Beta_Range_Sensitivity.csv")))


# =========================================================================== S28 identity
@table("identity")
def t_identity():
    f = cmp_("Beta_DFA_Consistency.csv")
    d = pd.read_csv(K.csvp(f)).set_index("Group")
    rows = []
    rows.append(["Excised slope β, mean", U(d.loc["Healthy", "Mean_Beta"]), U(d.loc["Psychiatric", "Mean_Beta"]), ""])
    rows.append(["2α₂ − 1, mean", U(d.loc["Healthy", "Mean_2Alpha2_minus_1"]), U(d.loc["Psychiatric", "Mean_2Alpha2_minus_1"]), ""])
    for a, label in (("DFA_Alpha2", "α₂ (16–64 beats)"), ("DFA_Alpha1", "α₁ (4–16 beats)"), ("DFA_Alpha", "full-range α")):
        hc, hp = d.loc[f"Healthy_vs_{a}"], d.loc[f"Psychiatric_vs_{a}"]
        rc = num(K.C.PSD_Beta_NoMayer) - (2 * num(K.C[a]) - 1)
        rp = num(K.P.PSD_Beta_NoMayer) - (2 * num(K.P[a]) - 1)
        assert close(rc.mean(), hc.Mean_Residual, 1e-5) and close(rp.mean(), hp.Mean_Residual, 1e-5)
        rr, pp, _, _ = mw(rc, rp)
        rows.append([f"Residual β − (2α − 1), α = {label}, mean (SD)", f"{R(hc.Mean_Residual)} ({U(hc.SD_Residual)})",
                     f"{R(hp.Mean_Residual)} ({U(hp.SD_Residual)})", f"{R(rr)} ({P3(pp)})"])
        if a == "DFA_Alpha2":
            t = d.loc["Residual group test"]
            assert close(rr, t.Mean_Residual, 1e-5) and close(pp, t.SD_Residual, 1e-5)
    rows.append(["Spearman correlation of β with 2α₂ − 1", U(d.loc["Healthy", "Spearman_r"]), U(d.loc["Psychiatric", "Spearman_r"]), ""])
    ruc = num(K.C.PSD_Beta) - (2 * num(K.C.DFA_Alpha2) - 1)
    rup = num(K.P.PSD_Beta) - (2 * num(K.P.DFA_Alpha2) - 1)
    rr, pp, _, _ = mw(ruc, rup)
    rows.append(["Uncorrected slope: residual against α₂, mean (SD)", f"{R(ruc.mean())} ({U(ruc.std())})",
                 f"{R(rup.mean())} ({U(rup.std())})", f"{R(rr)} ({P3(pp)})"])
    FACTS["s.id_unc_c"], FACTS["s.id_unc_p"] = R(ruc.mean()), R(rup.mean())
    FACTS["s.id_unc_r"], FACTS["s.id_unc_pe"], FACTS["s.id_unc_p3"] = R(rr), K.PE(pp), P3(pp)
    note = ("For a single power law, as in fractional Gaussian noise, the spectral exponent and the DFA exponent "
            "satisfy β = 2α − 1, so the residual is zero. The upper rows use the baroreflex-excised slope; the last "
            "row uses the uncorrected slope, which is the one plotted in Supplementary Figure "
            f"{SFNUM['identity']}. The group difference is the rank-biserial *r* of the residual (positive = larger in "
            "patients) with uncorrected *p*. With the excised slope both groups depart from the identity against α₂ in "
            "the same direction, and the departure does not differ significantly between them. With the uncorrected "
            f"slope the departure is larger in patients ({FACTS['s.id_unc_r']}, *p* = {FACTS['s.id_unc_p3']}), "
            "because the baroreflex resonance flattens the control slope more (Supplementary Table "
            f"{SNUM['estimators']}). Against α₁ and full-range α the residuals also differ between groups.")
    return dict(header=["Quantity", "Controls", "Patients", "Group difference, *r* (*p*)"], rows=rows, note=note, sources=src(f))


# =========================================================================== S29 multifractal DFA
@table("mfdfa")
def t_mfdfa():
    f = cmp_("MFDFA_Group_Comparison.csv")
    m = pd.read_csv(K.csvp(f)).set_index("Metric")
    rows = [{"section": "Positive moments, q = 1 to 5 (family of three tests)"}]
    for key in ["MFDFA_Width", "MFDFA_H_Mean", "MFDFA_H2"]:
        rw = m.loc[key]
        rr, pp, _, _ = mw(K.C[key], K.P[key])
        assert close(rr, rw.r_effect, 1e-5)
        rows.append([lab(key), f"{U(rw.Healthy_Mean)} ({U(rw.Healthy_SD)})", f"{U(rw.Psych_Mean)} ({U(rw.Psych_SD)})",
                     R(rw.r_effect), P3(rw.p_raw), P3(rw.p_adj)])
    rows.append({"section": "Full moment range, q = −5 to 5 (unstable at these lengths; uncorrected *p*)"})
    for key, label in (("MFDFA_Width_FullQ", "MFDFA width Δh, q = −5 to 5"), ("MFDFA_H_Mean_FullQ", "MFDFA mean H(q), q = −5 to 5")):
        rr, pp, _, _ = mw(K.C[key], K.P[key])
        c, p = num(K.C[key]), num(K.P[key])
        rows.append([label, f"{U(c.mean())} ({U(c.std())})", f"{U(p.mean())} ({U(p.std())})", R(rr), P3(pp), "–"])
    allw = pd.concat([num(K.C.MFDFA_Width_FullQ), num(K.P.MFDFA_Width_FullQ)])
    allp = pd.concat([num(K.C.MFDFA_Width), num(K.P.MFDFA_Width)])
    FACTS["s.mf_cv_full"] = U(allw.std() / allw.mean(), 2)
    FACTS["s.mf_cv_pos"] = U(allp.std() / allp.mean(), 2)
    FACTS["s.mf_full_max"] = U(allw.max(), 2)
    rows.append({"section": "Width against each subject's own 20 surrogates, Z-score (uncorrected *p*)"})
    for tag, label in (("Shuffle", "Shuffled surrogates (correlations removed)"), ("PhaseRand", "Phase-randomized surrogates (distribution made Gaussian)")):
        key = f"MFDFA_Width_Z_{tag}"
        assert (num(K.C[f"MFDFA_Surr_N_{tag}"]) == 20).all() and (num(K.P[f"MFDFA_Surr_N_{tag}"]) == 20).all()
        rr, pp, _, _ = mw(K.C[key], K.P[key])
        c, p = num(K.C[key]), num(K.P[key])
        rows.append([label + ", median", U(c.median(), 2), U(p.median(), 2), R(rr), P3(pp), "–"])
        nc = int((c > 1.96).sum()); npat = int((p > 1.96).sum())
        rows.append([f"  Z above 1.96, *n*", f"{nc} of 30", f"{npat} of 30", "–", P3(fisher(nc, 30, npat, 30)), "–"])
        FACTS[f"s.mfz_{tag}_r"] = R(rr)
        FACTS[f"s.mfz_{tag}_pe"] = K.PE(pp)
        FACTS[f"s.mfz_{tag}_c"], FACTS[f"s.mfz_{tag}_p"] = U(c.median(), 2), U(p.median(), 2)
        FACTS[f"s.mfz_{tag}_nc"], FACTS[f"s.mfz_{tag}_np"] = str(nc), str(npat)
        if tag == "PhaseRand":
            assert (str(nc), str(npat)) == (FACTS["mfdfa.pr_nc"], FACTS["mfdfa.pr_np"]), "supplement and main text disagree"
    G = np.r_[np.zeros(30), np.ones(30)]
    CP = pd.concat([K.C, K.P])
    ra, pa = partial(num(CP.MFDFA_Width_Z_PhaseRand).values, G, [num(CP.HRV_MeanHR).values])
    FACTS["s.mfz_pr_hr_r"], FACTS["s.mfz_pr_hr_pe"] = R(ra), K.PE(pa)
    FACTS["s.nrr_max"] = INT(pd.concat([num(K.C.Num_RR), num(K.P.Num_RR)]).max())
    sc = pd.concat([num(K.C.MFDFA_Scale_Min), num(K.P.MFDFA_Scale_Min)])
    sx = pd.concat([num(K.C.MFDFA_Scale_Max), num(K.P.MFDFA_Scale_Max)])
    assert (sc == 8).all()
    FACTS["s.mf_smax_lo"], FACTS["s.mf_smax_hi"] = INT(sx.min()), INT(sx.max())
    note = ("Means (SD) or medians as labeled; *r* is the rank-biserial correlation (positive = higher in patients). "
            "H(q) is fitted over 20 log-spaced scales from 8 beats to a quarter of the series ("
            f"{FACTS['s.mf_smax_lo']} to {FACTS['s.mf_smax_hi']} beats). Negative moments weight the smallest "
            f"fluctuations and are unreliable on series of {FACTS['qc.n_min']} to {FACTS['s.nrr_max']} beats: the full-range width has a "
            f"coefficient of variation of {FACTS['s.mf_cv_full']} across subjects against {FACTS['s.mf_cv_pos']} for "
            f"positive moments, and reaches {FACTS['s.mf_full_max']} in one subject, so it is shown only to document "
            "that instability. Shuffled surrogates keep the amplitude distribution and remove correlations; "
            "phase-randomized surrogates keep the power spectrum and make the distribution Gaussian (Kantelhardt et "
            "al., 2002). Z is (observed width − surrogate mean)/surrogate SD over 20 surrogates of each kind, so it is "
            "a coarse statistic. Supplementary Methods "
            f"{MNUM['fractal']} discusses the phase-randomized result.")
    return dict(header=["Measure", "Controls", "Patients", "*r*", "*p*", "*p* adj"], rows=rows, note=note, sources=src(f))


# =========================================================================== S30 covariates
@table("covariates")
def t_covariates():
    bf = os.path.join(K.OUT, "accelerometer", "Beta_Activity_Covariate.csv")
    sf = os.path.join(K.OUT, "accelerometer", "Beta_Activity_Covariate_Sensitivity.csv")
    lf = cmp_("Length_Respiration_Checks.csv")
    cf = cmp_("PSD_Beta_Covariate_Analysis.csv")
    hf = cmp_("HR_Adjusted_Comparison.csv")
    ba = pd.read_csv(K.csvp(bf)).set_index("Analysis")
    bs = pd.read_csv(K.csvp(sf)).set_index("Analysis")
    lr = pd.read_csv(K.csvp(lf)).set_index("Variable")
    cv = pd.read_csv(K.csvp(cf)).set_index("Analysis")
    hr = pd.read_csv(K.csvp(hf)).set_index("Metric")
    outs = ["PSD_Beta", "PSD_Beta_NoMayer", "VLF_LF_Ratio"]
    G = np.r_[np.zeros(30), np.ones(30)]
    CP = pd.concat([K.C, K.P])
    HRv, RMv = num(CP.HRV_MeanHR).values, num(CP.HRV_RMSSD).values
    rows = []

    def cell(r, p):
        return f"{R(r)} ({P3(p)})"
    models = [("Unadjusted", "(unadjusted)"), ("Mobility", "| Mobility"), ("HF power", "| HF power"),
              ("HF power + heart rate", "| HF power + MeanHR"), ("Mobility + heart rate", "| Mobility + MeanHR"),
              ("Mobility + HF power + heart rate", "| Mobility + HF power + MeanHR")]
    for label, tag in models:
        row = [label]
        for o in outs:
            key = f"{o} ~ Group {tag}" if tag.startswith("|") else f"{o} ~ Group {tag}"
            rw = ba.loc[key]
            row.append(cell(rw.partial_r, rw.p))
        rows.append(row)
    # heart rate alone (main Table 3) and RMSSD models
    rows.append(["Heart rate"] + [cell(hr.loc[o, "r_partial_HR"], hr.loc[o, "p_raw"]) for o in outs])
    for label, covs, ckey in (("RMSSD", [RMv], "PSD_Beta ~ Group | RMSSD"), ("Heart rate + RMSSD", [HRv, RMv], "PSD_Beta ~ Group | MeanHR + RMSSD")):
        row = [label]
        for o in outs:
            r, p = partial(num(CP[o]).values, G, covs)
            if o == "PSD_Beta":
                assert close(r, cv.loc[ckey, "r_or_AUC"], 1e-5)
            row.append(cell(r, p))
        rows.append(row)
    rows.append(["Age + sex + mobility + heart rate + HF power"] +
                [cell(lr.loc[f"{o} ~ Group | Age + Sex + mobility_mg + HRV_MeanHR + HF_Power_ms2", "r_effect"],
                      lr.loc[f"{o} ~ Group | Age + Sex + mobility_mg + HRV_MeanHR + HF_Power_ms2", "p_raw"]) for o in outs])
    # check two printed values against the per-subject data
    mob = pd.read_csv(K.csvp(os.path.join(K.OUT, "accelerometer", "mobility_coefficients.csv"))).set_index("Record_ID").mobility_mg
    mv = mob.reindex(CP.index).values
    r, p = partial(num(CP.PSD_Beta_NoMayer).values, G, [mv])
    assert close(r, ba.loc["PSD_Beta_NoMayer ~ Group | Mobility", "partial_r"], 5e-4)
    rows.append({"section": "Excluding the two subjects with accelerometer coverage below 80% (29 controls, 29 patients)"})
    for label, tag in models:
        row = [label]
        for o in outs:
            rw = bs.loc[f"{o} ~ Group {tag}"]
            assert int(rw.n) == 58
            row.append(cell(rw.partial_r, rw.p))
        rows.append(row)
    weakest = min(ba.loc[f"PSD_Beta_NoMayer ~ Group {t}", "partial_r"] for _, t in models)
    assert weakest == ba.loc["PSD_Beta_NoMayer ~ Group | HF power", "partial_r"]
    note = ("Partial Spearman correlation of each measure with group (positive = higher in patients) with the listed "
            "covariates held constant: measure and group indicator are each regressed on the covariates by least "
            "squares with an intercept and the residuals rank-correlated. *p* is uncorrected. Mobility is mean "
            "gravity-removed acceleration over the window (Supplementary Methods "
            f"{MNUM['movement']}). Sex is coded 0/1. HF power and RMSSD are themselves group differences, so models "
            "containing them decompose the difference rather than remove a confounder. The values in the "
            "first six rows are those of Beta_Activity_Covariate.csv; the heart-rate row is from main Table 3; the "
            "RMSSD rows were computed for this table by the same method; the five-covariate row is from "
            "Length_Respiration_Checks.csv; and the coverage rows are from Beta_Activity_Covariate_Sensitivity.csv.")
    return dict(header=["Covariates", "Uncorrected slope", "Baroreflex-excised slope", "VLF/LF ratio"], rows=rows, note=note,
                landscape=True, sources=src(bf, sf, lf, cf, hf))


# =========================================================================== S31 activity
@table("activity")
def t_activity():
    mf = os.path.join(K.OUT, "accelerometer", "Mobility_Group_Comparison.csv")
    af = os.path.join(K.OUT, "accelerometer", "Activity_Structure_Comparison.csv")
    tf = cmp_("Ambulatory_Fraction_Test.csv")
    mob = pd.read_csv(K.csvp(mf)).iloc[0]
    act = pd.read_csv(K.csvp(af)).set_index("Quantity")
    amb = pd.read_csv(K.csvp(tf)).iloc[0]
    # Mobility_Group_Comparison signs r > 0 as controls more mobile; negate to positive = higher in patients
    mo = pd.read_csv(K.csvp(os.path.join(K.OUT, "accelerometer", "mobility_coefficients.csv"))).set_index("Record_ID")
    rr, pp, _, _ = mw(mo.loc[K.C.index, "mobility_mg"], mo.loc[K.P.index, "mobility_mg"])
    assert close(rr, -mob.rank_biserial_r, 1e-3) and mob.Patient_median > mob.Control_median and rr > 0
    rows = [["Mobility, mean gravity-removed acceleration (mg)", f"{U(mob.Control_mean, 1)} (median {U(mob.Control_median, 1)})",
             f"{U(mob.Patient_mean, 1)} (median {U(mob.Patient_median, 1)})", R(rr), P3(pp), "30, 30"]]
    names = {"N_bouts": "Walking bouts, *n*", "Total_ambul_s": "Time walking, including padding (s)",
             "Pct_window_ambul": "Share of the window walking (%)", "First_bout_s": "Start of the first bout (s into the window)",
             "Longest_bout_s": "Longest bout (s)", "NonBout_Motion_Median": "Motion outside bouts, median (mg)",
             "NonBout_Motion_Variance": "Motion outside bouts, variance (mg²)", "NonBout_Motion_IQR": "Motion outside bouts, IQR (mg)"}
    def UC(x, nd):
        # U with a thousands separator, as elsewhere in the supplement
        return K.neg(f"{x:,.{nd}f}") if np.isfinite(x) else "–"

    for q, rw in act.iterrows():
        nd = 2 if q == "N_bouts" else 1
        rows.append([names[q], f"{UC(rw.Control_mean, nd)} ({UC(rw.Control_SD, nd)})", f"{UC(rw.Patient_mean, nd)} ({UC(rw.Patient_SD, nd)})",
                     R(rw.r_effect), P3(rw.p_raw), f"{int(rw.N_control)}, {int(rw.N_patient)}"])
    assert (act.p_raw > 0.05).all()
    rows.append(["Beats inside walking bouts (%)", f"{U(amb.Control_mean, 1)} ({U(amb.Control_SD, 1)})",
                 f"{U(amb.Patient_mean, 1)} ({U(amb.Patient_SD, 1)})", R(amb.r_effect), P3(amb.p_raw),
                 f"{int(amb.N_control)}, {int(amb.N_patient)}"])
    st = pd.read_csv(K.csvp(os.path.join(K.OUT, "stationary", "Stationary_Longest_Runs.csv")))
    sc, sp = st[st.Group == "control"].N_Longest_Contiguous, st[st.Group == "patient"].N_Longest_Contiguous
    rr, pp, _, _ = mw(sc, sp)
    rows.append(["Longest movement-free run (beats)", f"{UC(sc.mean(), 0)} ({UC(sc.std(), 0)})", f"{UC(sp.mean(), 0)} ({UC(sp.std(), 0)})",
                 R(rr), P3(pp), "30, 30"])
    FACTS.update({"s.run_r": R(rr), "s.run_pe": K.PE(pp)})
    assert pp < 0.05
    fc, fp_ = 100 * st[st.Group == "control"].Frac_Of_Window, 100 * st[st.Group == "patient"].Frac_Of_Window
    assert np.allclose(fc.values, 100 * (sc / st[st.Group == "control"].N_Window).values, atol=0.01)
    rf_, pf_, _, _ = mw(fc, fp_)
    rows.append(["Longest movement-free run (% of the window's beats)", f"{U(fc.mean(), 1)} ({U(fc.std(), 1)})",
                 f"{U(fp_.mean(), 1)} ({U(fp_.std(), 1)})", R(rf_), P3(pf_), "30, 30"])
    FACTS.update({"s.runf_c_med": INT(fc.median()), "s.runf_p_med": INT(fp_.median()), "s.runf_r": R(rf_),
                  "s.runf_pe": K.PE(pf_)})
    assert pf_ > 0.05
    FACTS.update({"s.run_c_med": INT(sc.median()), "s.run_p_med": INT(sp.median()),
                  "s.run600_c": str(int((sc >= 600).sum())), "s.run600_p": str(int((sp >= 600).sum()))})
    nb_c = int((act.loc["First_bout_s", "N_control"])); nb_p = int(act.loc["First_bout_s", "N_patient"])
    FACTS["s.nobout_c"], FACTS["s.nobout_p"] = str(30 - nb_c), str(30 - nb_p)
    note = ("Means (SD) unless stated; *r* is the rank-biserial correlation (positive = higher in patients) with "
            f"uncorrected *p*. Bout timing and length are defined only for subjects with a bout ({nb_c} controls and "
            f"{nb_p} patients). Walking time includes the 30 s and 180 s padding around each bout; the share of the "
            "window uses the accelerometer trace duration as its denominator. None of the accelerometer quantities "
            "differs between groups. The longest movement-free run holds more beats in patients, partly because their "
            "heart rate is higher, so that a run of the same duration holds more beats, and partly because the run "
            "covered a larger share of the window in patients, a difference that was not significant on its own "
            f"(Supplementary Results {RNUM['activity']}). The protocol asked patients to walk in the corridor about 30 minutes after the start, "
            "inside the analysis window, did not supervise the walks, and did not give the instruction to seven "
            "controls (Supplementary Methods "
            f"{MNUM['protocol']}).")
    return dict(header=["Quantity", "Controls", "Patients", "*r*", "*p*", "*n*"], rows=rows, note=note, landscape=True,
                sources=src(mf, af, tf))


# =========================================================================== S32 movement-free
@table("movementfree")
def t_movementfree():
    bf = os.path.join(K.OUT, "stationary", "Stationary_Reanalysis_beats.csv")
    df = os.path.join(K.OUT, "stationary", "Stationary_Reanalysis_duration.csv")
    b = pd.read_csv(K.csvp(bf))
    d = pd.read_csv(K.csvp(df))
    bc, bp = b[b.Group == "control"], b[b.Group == "patient"]
    dc, dp = d[d.Group == "control"], d[d.Group == "patient"]
    assert (len(bc), len(bp), len(dc), len(dp)) == (25, 26, 14, 22)
    assert (b.Match_Mode == "beats").all() and (d.Match_Mode == "duration").all()
    assert b.N_Beats.between(596, 600).all()
    assert (d.N_Beats >= 500).all()
    metrics = ["PSD_Beta_NoMayer", "PSD_Beta", "PSD_Beta_Specparam", "VLF_LF_Ratio", "VLF_Power_ms2", "LF_Power_ms2",
               "HF_Power_ms2", "Total_Power_ms2", "LF_HF_Ratio", "HRV_RMSSD", "HRV_MeanHR", "SampEn", "ApEn", "PermEn",
               "MSE_Mean", "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2", "RQA_DET", "RQA_LAM", "MFDFA_H2"]
    rows = [["Subjects (controls, patients)", "25, 26", "", "", "14, 22", "", ""],
            ["Duration (s), median (controls; patients)", f"{U(bc.Duration_s.median(), 0)}; {U(bp.Duration_s.median(), 0)}", "", "",
             f"{U(dc.Duration_s.median(), 0)}; {U(dp.Duration_s.median(), 0)}", "", ""],
            ["Clean beats, median (controls; patients)", f"{U(bc.N_Beats.median(), 0)}; {U(bp.N_Beats.median(), 0)}", "", "",
             f"{U(dc.N_Beats.median(), 0)}; {U(dp.N_Beats.median(), 0)}", "", ""],
            ["Welch segments, median (controls; patients)", f"{U(bc.Welch_N_Segments.median(), 0)}; {U(bp.Welch_N_Segments.median(), 0)}", "", "",
             f"{U(dc.Welch_N_Segments.median(), 0)}; {U(dp.Welch_N_Segments.median(), 0)}", "", ""],
            {"subheader": ["Measure", "600 beats: medians (controls; patients)", "*r*", "*p*", "370 s: medians (controls; patients)", "*r*", "*p*"]}]
    mr = {}
    for m in metrics:
        r1, p1, _, _ = mw(bc[m], bp[m])
        r2, p2, _, _ = mw(dc[m], dp[m])
        mr[m] = (r1, p1, r2, p2)
        rows.append([lab(m) if m != "PSD_Beta_Specparam" else "Slope, aperiodic plus peak",
                     f"{SIG(bc[m].median())}; {SIG(bp[m].median())}", R(r1), P3(p1),
                     f"{SIG(dc[m].median())}; {SIG(dp[m].median())}", R(r2), P3(p2)])
    # must match the main-text facts
    assert R(mr["PSD_Beta_NoMayer"][0]) == FACTS["mf.BETAX.r"] and R(mr["VLF_LF_Ratio"][0]) == FACTS["mf.VLFLF.r"]
    FACTS["s.mfd_betax_r"], FACTS["s.mfd_betax_pe"] = R(mr["PSD_Beta_NoMayer"][2]), K.PE(mr["PSD_Beta_NoMayer"][3])
    FACTS["s.mfd_vlflf_r"], FACTS["s.mfd_vlflf_pe"] = R(mr["VLF_LF_Ratio"][2]), K.PE(mr["VLF_LF_Ratio"][3])
    FACTS["s.mfd_hr_c"], FACTS["s.mfd_hr_p"] = U(dc.HRV_MeanHR.min(), 0), U(dp.HRV_MeanHR.min(), 0)
    FACTS["s.mf_hr_r"], FACTS["s.mf_hr_pe"] = R(mr["HRV_MeanHR"][0]), K.PE(mr["HRV_MeanHR"][1])
    note = ("Each measure recomputed on the longest contiguous movement-free run of each subject's window. 600 beats "
            "(primary): the first 600 beats of runs at least that long, then the artifact filter. 370 s: the first "
            "370 s of runs at least that long and holding at least 500 beats within it, which admits only subjects whose "
            f"heart rate in the run was at least about 81 bpm (lowest {FACTS['s.mfd_hr_c']} in controls and "
            f"{FACTS['s.mfd_hr_p']} in patients); this version is selected on heart rate and underpowered and is shown for "
            "completeness. *r* is the rank-biserial correlation (positive = higher in patients), uncorrected. Short "
            "segments average fewer Welch segments, so their spectral estimates, VLF power above all, are noisier "
            "than those of the full window.")
    return dict(header=["Measure", "600 beats", "", "", "370 s", "", ""], rows=rows, note=note, landscape=True,
                sources=src(bf, df))


# =========================================================================== S33 subgroups
@table("subgroups")
def t_subgroups():
    gf = cmp_("Diagnosis_Subgroup.csv")
    qf = cmp_("Quetiapine_Analysis.csv")
    cf = cmp_("Comorbidity_Sensitivity.csv")
    g = pd.read_csv(K.csvp(gf)).set_index("Metric")
    q = pd.read_csv(K.csvp(qf)).set_index("Metric")
    c = pd.read_csv(K.csvp(cf)).set_index("Metric")
    qv = num(K.P.Quetiapine)
    dx = K.P.Diagnosis.astype(str)
    sub = K.P.Comorbid_Substance.astype(str).str.lower().isin(["true", "1", "1.0"])
    assert int(sub.sum()) == 8 and int((qv == 1).sum()) == 7 and int((dx == "bipolar").sum()) == 7
    rows = []
    for m in g.index:
        gr, qr, cr = g.loc[m], q.loc[m], c.loc[m]
        # sign checks against the per-subject data
        r_sb, p_sb, _, _ = mw(K.P.loc[dx == "schizophrenia", m], K.P.loc[dx == "bipolar", m])
        assert close(r_sb, gr.r_sz_vs_bp, 1e-5)           # positive = higher in bipolar
        r_eu, p_eu, _, _ = mw(K.P.loc[qv == 0, m], K.P.loc[qv == 1, m])
        assert close(r_eu, -qr.r_quet_vs_nonquet, 1e-5)   # negated: positive = higher in exposed
        r_cw, p_cw, _, _ = mw(K.P.loc[~sub, m], K.P.loc[sub, m])
        assert close(r_cw, -cr.r_comorbid_vs_clean, 1e-5)  # negated: positive = higher with a substance code
        rows.append([lab(m), f"{R(gr.r_schizophrenia)} ({P3(gr.p_schizophrenia)})", f"{R(gr.r_bipolar)} ({P3(gr.p_bipolar)})",
                     f"{R(gr.r_sz_vs_bp)} ({P3(gr.p_sz_vs_bp)})", f"{R(qr.r_quetiapine_vs_control)} ({P3(qr.p_quetiapine_vs_control)})",
                     f"{R(qr.r_nonquet_vs_control)} ({P3(qr.p_nonquet_vs_control)})", f"{R(r_eu)} ({P3(p_eu)})",
                     f"{R(qr.r_group_adj_quetiapine)} ({P3(qr.p_group_adj_quetiapine)})",
                     f"{R(cr.r_no_comorbidity)} ({P3(cr.p_no_comorbidity)})", f"{R(r_cw)} ({P3(p_cw)})"])
    mde = pd.read_csv(K.csvp(cmp_("Minimum_Detectable_Effect_Subgroups.csv"))).set_index("Contrast").Min_Detectable_r_rankbiserial
    assert mde["Bipolar vs controls"] == mde["Quetiapine exposed vs controls"]
    assert mde["Schizophrenia vs bipolar"] == mde["Quetiapine exposed vs unexposed"]
    note = ("Rank-biserial *r* with uncorrected *p*. Against controls (*n* = 30): schizophrenia (*n* = 23), bipolar disorder (*n* = 7), "
            "quetiapine-exposed (*n* = 7) and unexposed (*n* = 23) patients, and patients without a substance-use code (*n* = 22); positive "
            "= higher in the patient subgroup. Schizophrenia against bipolar: positive = higher in bipolar disorder. "
            "Exposed against unexposed: positive = higher in exposed patients. With against without a substance-use "
            "code (8 against 22): positive = higher with the code. Adjusted for quetiapine: partial Spearman correlation "
            "of the measure with group, with quetiapine exposure (controls coded unexposed) held constant. The smallest "
            f"rank-biserial *r* detectable with 80% power is {U(mde['Schizophrenia vs controls'])} for schizophrenia against "
            f"controls, {U(mde['Bipolar vs controls'])} for bipolar disorder or exposed patients against controls and "
            f"{U(mde['Schizophrenia vs bipolar'])} for the within-patient contrasts with 7 patients on one side (Supplementary "
            f"Table {SNUM['power']}). Quetiapine exposure is concentrated in the bipolar subgroup ({FACTS['cohort.quet_bp']} of "
            f"{FACTS['cohort.quet']}), all {FACTS['n_bp']} bipolar patients are women, and all {FACTS['cohort.comorb']} "
            "substance-use codes are in the schizophrenia subgroup. "
            "Patients without quetiapine were receiving other medication.")
    return dict(header=["Measure", "Schizophrenia against controls", "Bipolar against controls", "Schizophrenia against bipolar",
                        "Exposed against controls", "Unexposed against controls", "Exposed against unexposed",
                        "Adjusted for quetiapine", "No substance code against controls", "With against without code"],
                rows=rows, note=note, landscape=True, sources=src(gf, qf, cf), font=14)


# =========================================================================== S34 MIT-BIH comparisons
@table("crosscohort")
def t_crosscohort():
    cf = os.path.join(K.XC, "CrossControl_Convergence.csv")
    c = pd.read_csv(K.csvp(cf))
    mf = os.path.join(K.OUT, "comparison", "healthy_mitbih_vs_psych", "group_comparison_stats.csv")
    mt = pd.read_csv(K.csvp(mf)).rename(columns={"Unnamed: 0": "L"}).set_index("L")
    rows = []
    nboth = 0
    for _, rw in c.iterrows():
        col = K.CSV_LABEL[rw.Metric]
        rr, pp, _, _ = mw(K.M[col], K.P[col])
        assert close(rr, rw.r_MITBIH, 1e-5)
        both = bool(rw.Robust)
        nboth += both
        rows.append([lab(col), SIG(mt.loc[rw.Metric, "Healthy_Median"]), SIG(mt.loc[rw.Metric, "Psych_Median"]),
                     f"{R(rw.r_MITBIH)} ({P3(rw.p_adj_MITBIH)})", f"{R(rw.r_HRVACC)} ({P3(rw.p_adj_HRVACC)})", "Yes" if both else "No"])
    assert str(nboth) == FACTS["xc.n_both"] and str(int(c.sig_MITBIH.sum())) == FACTS["xc.n_sig_mit"]
    af = os.path.join(K.XC, "CrossDevice_AgeAdjusted.csv")
    a = pd.read_csv(K.csvp(af))
    assert (a.Sign == "positive = second-named cohort higher").all()
    rows.append({"section": "Age held constant (partial Spearman correlation with cohort, age as covariate)"})
    rows.append({"subheader": ["Measure", "Comparison", "", "*r* (*p*)", "Partial *r*, unadjusted (*p*)", "Partial *r*, age held constant (*p*)"]})
    for _, rw in a.iterrows():
        cc = rw.Comparison.startswith("HRVACC_control")
        s = -1.0 if cc else 1.0    # control-control rows: negate so positive = higher in HRV-ACC controls
        if cc:
            rr, pp, _, _ = mw(K.M[rw.Metric], K.C[rw.Metric])
            assert close(rr, -rw.r_rank_biserial, 1e-5)
        rows.append([lab(rw.Metric), "HRV-ACC controls against MIT-BIH" if cc else "Patients against MIT-BIH", "",
                     f"{R(s * rw.r_rank_biserial)} ({P3(rw.p_rank_biserial)})",
                     f"{R(s * rw.r_partial_unadjusted)} ({P3(rw.p_partial_unadjusted)})",
                     f"{R(s * rw.r_partial_age_adjusted)} ({P3(rw.p_partial_age_adjusted)})"])
    note = ("Rank-biserial *r* with positive values meaning higher in the HRV-ACC group (patients, or HRV-ACC controls "
            "in the control-control rows); *p* adj is Benjamini-Hochberg over the 21 measures within each comparison. "
            "MIT-BIH medians are from 18 records analyzed with the same window and pipeline; their R-R intervals lie on "
            f"a 7.8125 ms grid. {FACTS['xc.n_both']} measures differ in the same direction against both control cohorts. "
            "In the lower block age is the covariate (MIT-BIH ages from the database records), because the MIT-BIH "
            "cohort is younger; the control-control rows are negated relative to CrossDevice_AgeAdjusted.csv, which "
            "signs them as higher in MIT-BIH, so that every row reads positive = higher in the HRV-ACC group.")
    return dict(header=["Measure", "MIT-BIH, median", "Patients, median", "Patients against MIT-BIH, *r* (*p* adj)",
                        "Patients against HRV-ACC controls, *r* (*p* adj)", "Both, same direction"],
                rows=rows, note=note, landscape=True, sources=src(cf, mf, af))


# =========================================================================== S35 three groups
@table("threegroup")
def t_threegroup():
    kf = os.path.join(K.XC, "ThreeGroup_KruskalWallis.csv")
    df = os.path.join(K.XC, "ThreeGroup_Dunn_Posthoc.csv")
    k = pd.read_csv(K.csvp(kf)).rename(columns={"Unnamed: 0": "L"})
    d = pd.read_csv(K.csvp(df))
    rows = []
    for _, rw in k.iterrows():
        sub = d[d.Metric == rw.L].set_index("Pair")
        cells = []
        for pair in ["MIT-BIH vs HRV-ACC", "MIT-BIH vs Psychiatric", "HRV-ACC vs Psychiatric"]:
            z = sub.loc[pair]
            cells.append(f"{R(-z.z, 2)} ({P3(z.p_adj)})")   # negated: positive = higher in the CSV pair's second group,
            #                                                    which is the first-named group of the column heading
        rows.append([lab(K.CSV_LABEL[rw.L]), U(rw.KW_H, 2), P3(rw.KW_p_adj)] + cells)
    # direction check: Dunn z > 0 when the first-named group has the higher mean rank
    zb = d[(d.Metric == "PSD β") & (d.Pair == "HRV-ACC vs Psychiatric")].iloc[0].z
    assert zb < 0 and num(K.P.PSD_Beta).median() > num(K.C.PSD_Beta).median()
    zl = d[(d.Metric == "RQA Laminarity") & (d.Pair == "MIT-BIH vs HRV-ACC")]
    assert len(zl) == 1 and zl.iloc[0].z < 0 and num(K.C.RQA_LAM).median() > num(K.M.RQA_LAM).median()
    nsig = int((k.KW_p_adj < 0.05).sum())
    assert nsig == len(k) - 1 and k[k.KW_p_adj >= 0.05].L.tolist() == ["RNA Clustering Coeff"]
    note = ("Kruskal-Wallis tests across MIT-BIH controls (*n* = 18), HRV-ACC controls (*n* = 30) and patients (*n* = 30), "
            f"Benjamini-Hochberg corrected over the {len(k)} quantities ({nsig} significant; network clustering is the "
            "exception). Pairwise columns give Dunn's tie-corrected *z* (Dunn, 1964), signed so that positive means "
            "higher in the first-named group of the column heading (patients, or HRV-ACC controls against MIT-BIH), "
            "with *p* adjusted by Benjamini-Hochberg over all "
            f"{len(d)} pairwise tests. Many omnibus differences are driven by the MIT-BIH cohort, which differs from "
            "HRV-ACC in device, resolution, setting, era and age.")
    return dict(header=["Measure", "*H*", "*p* adj", "HRV-ACC controls against MIT-BIH, *z* (*p* adj)",
                        "Patients against MIT-BIH, *z* (*p* adj)", "Patients against HRV-ACC controls, *z* (*p* adj)"],
                rows=rows, note=note, landscape=True, sources=src(kf, df))


# =========================================================================== S36 age and sex
@table("agesex")
def t_agesex():
    af = cmp_("AgeSex_Adjusted_Comparison.csv")
    bf = cmp_("Age_Association_BothGroups.csv")
    wf = cmp_("Age_Sex_WithinPatients.csv")
    a = pd.read_csv(K.csvp(af)).set_index("Metric")
    b = pd.read_csv(K.csvp(bf)).set_index("Metric")
    w = pd.read_csv(K.csvp(wf)).set_index("Metric")
    sx = K.P.Sex.astype(str).str.upper().str[0]
    rows = []
    for m in a.index:
        ar, br, wr = a.loc[m], b.loc[m], w.loc[m]
        r_s, p_s, _, _ = mw(K.P.loc[sx == "F", m], K.P.loc[sx == "M", m])
        assert close(r_s, wr.r_sex_within_patients, 1e-5)        # positive = higher in men
        rc, pc = spearmanr(num(K.C[m]), num(K.C.Age))
        assert close(rc, br.r_age_control, 1e-5)
        rows.append([lab(m), R(ar.r_unadj_partial), f"{R(ar.r_adj_age)} ({P3(ar.p_adj_age)})", f"{R(ar.r_adj_sex)} ({P3(ar.p_adj_sex)})",
                     f"{R(ar.r_adj_age_sex)} ({P3(ar.p_adj_age_sex)})", f"{R(br.r_age_control)} ({P3(br.p_age_control)})",
                     f"{R(br.r_age_patient)} ({P3(br.p_age_patient)})", f"{R(wr.r_sex_within_patients)} ({P3(wr.p_sex)})"])
    dm = pd.read_csv(K.csvp(cmp_("Demographic_Matching.csv"))).set_index("Variable")
    note = ("Columns 2 to 5: partial Spearman correlation of the measure with group (positive = higher in patients) "
            "without covariates and with age, sex, or both held constant. Columns 6 and 7: Spearman correlation of the "
            "measure with age within controls and within patients. Last column: rank-biserial *r* for sex within "
            f"patients ({int(w.N_female.iloc[0])} women, {int(w.N_male.iloc[0])} men; positive = higher in men). All *p* values "
            "uncorrected. The groups were of "
            f"similar age (means {U(dm.loc['Age', 'Control'], 1)} and {U(dm.loc['Age', 'Patient'], 1)} years; *r* = "
            f"{R(dm.loc['Age', 'r_effect'])}, *p* = {P3(dm.loc['Age', 'p_value'])}) and sex composition "
            f"({int(dm.loc['Sex (N female)', 'Control'])} and {int(dm.loc['Sex (N female)', 'Patient'])} women; "
            f"Fisher *p* = {P3(dm.loc['Sex (N female)', 'p_value'])}).")
    return dict(header=["Measure", "Partial *r*, unadjusted", "Age", "Sex", "Age and sex", "Age, controls (ρ)",
                        "Age, patients (ρ)", "Sex, within patients"], rows=rows, note=note, landscape=True,
                sources=src(af, bf, wf))
