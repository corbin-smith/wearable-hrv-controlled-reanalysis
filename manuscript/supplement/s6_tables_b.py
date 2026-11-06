"""s6_tables_b.py: Supplementary Tables for software, validation, network, clinical, redundancy, correlation,
heart rate, fixed beats, re-quantization, DFA scales, control cohorts, embedding and surrogates."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import s6_common as K
from s6_common import (R, U, P3, SIG, INT, mw, partial, bh, lab, num, code_const, FACTS, SNUM, SFNUM, MNUM, RNUM)
from s6_tables import table, src, cmp_, close, fisher


# =========================================================================== S11 environment
@table("environment")
def t_environment():
    ti_f = cmp_("Test_Implementation.csv")
    ti = pd.read_csv(K.csvp(ti_f)).set_index("Item").Value
    envs = {}
    files = []
    for coh in ("healthy_hrvacc", "psychiatric", "healthy"):
        f = os.path.join(K.OUT, coh, "Environment.csv")
        files.append(f)
        envs[coh] = pd.read_csv(K.csvp(f)).set_index("Item").Value
    base = envs["healthy_hrvacc"]
    for coh, e in envs.items():
        assert e.equals(base), f"environment differs in {coh}"
    assert ti["Continuity correction"] == "True" and ti["Primary test"] == "Mann-Whitney U, two-sided"
    rows = [{"section": "Statistical test (comparison.py)"},
            ["Primary test", "Mann-Whitney *U*, two-sided, SciPy method 'auto'"],
            ["Continuity correction", "Applied (passed explicitly)"],
            ["Tie correction", "Applied (SciPy default)"],
            ["Effect size", "Rank-biserial *r* = 1 − 2*U*/(*n*₁*n*₂), *U* for the reference group (controls; MIT-BIH in cross-cohort contrasts)"],
            ["Multiple comparisons", "Benjamini-Hochberg within each family"],
            {"section": "Software (identical in the three cohort runs)"},
            ["Python", base["python"]], ["Platform", base["platform"]],
            ["NumPy", base["version_numpy"]], ["SciPy", base["version_scipy"]], ["pandas", base["version_pandas"]],
            ["scikit-learn", base["version_sklearn"]], ["statsmodels", base["version_statsmodels"]],
            ["NeuroKit2", base["version_neurokit2"]], ["matplotlib", base["version_matplotlib"]], ["wfdb", base["version_wfdb"]],
            {"section": "Seeds and run constants"},
            ["Seed, quantization controls, MFDFA surrogates and estimator validation", base["seed_quant"]],
            ["Seed, IAAFT surrogates", base["seed_surrogate"]],
            ["Seed, classifier and its bootstraps and permutations", code_const("nested_classifier.py", "RANDOM_SEED")],
            ["Seed, bootstrap intervals for *r*", "0"],
            ["Dither replicates", base["dither_replicates"]],
            ["IAAFT surrogates per subject; iterations each", f"{base['n_surrogates']}; {int(base['n_iaaft_iters']):,}"],
            ["MFDFA surrogates per subject and type", base["mfdfa_n_surrogates"]],
            ["Quantization step; matched target; re-quantization step", f"{base['quant_step_s']} s; {base['quant_target_ratio']}; {base['requant_target_step_s']} s"],
            ["Onset skip (HRV-ACC; MIT-BIH); window", f"{base['skip_seconds']} s; {base['mitbih_skip_seconds']} s; {base['analysis_seconds']} s"]]
    assert K.code_has("comparison.py", "n_boot=2000, seed=0")
    note = ("Read from Test_Implementation.csv and the Environment.csv written by each cohort run; the build "
            "script checks that the three Environment files are identical. The seeds make every stochastic step "
            "reproducible with these library versions. Supplementary Methods "
            f"{MNUM['software']} lists what each seed controls.")
    return dict(header=["Item", "Value"], rows=rows, note=note, sources=src(ti_f, *files))


# =========================================================================== S12 estimator validation
@table("validation")
def t_validation():
    f = os.path.join(K.OUT, "validation", "Estimator_Validation.csv")
    v = pd.read_csv(K.csvp(f))
    assert (v.N_Reps_Valid == 5).all()
    names = {"dfa_full": "DFA α (full range)", "dfa_a1": "DFA α₁ (4–16 beats)", "dfa_a2": "DFA α₂ (16–64 beats)",
             "mfdfa_h2": "MFDFA H(2)"}
    Hs = sorted(v.True_H.unique())
    rows = []
    for est in ["dfa_full", "dfa_a1", "dfa_a2", "mfdfa_h2"]:
        for n in sorted(v.N.unique()):
            sub = v[(v.Estimator == est) & (v.N == n)].set_index("True_H")
            rows.append([names[est], INT(n)] + [f"{U(sub.loc[h, 'Recovered_Mean'])} ± {U(sub.loc[h, 'Recovered_SD'])}"
                                                 f" ({R(sub.loc[h, 'Bias'])})" for h in Hs])
    note = ("Exact fractional Gaussian noise synthesized by circulant embedding (Davies and Harte, 1987), for which "
            "every estimator's target is the Hurst exponent *H*. Cells give the mean ± SD over 5 series and the bias "
            "(mean − *H*). Full-range α, α₂ and H(2) recovered *H* within ±0.04; α₁ read high by 0.06 to 0.10 at "
            "every *H*, a constant offset that cancels in a group difference only if it does not depend on the "
            "exponent. The observed exponents lie above this range (median α₂ "
            f"{FACTS['val.a2_med_c']} in controls and {FACTS['val.a2_med_p']} in patients).")
    return dict(header=["Estimator", "Series length"] + [f"*H* = {U(h, 1)}" for h in Hs], rows=rows, note=note,
                landscape=True, sources=src(f))


# =========================================================================== S13 recurrence networks
@table("rna")
def t_rna():
    f = cmp_("RNA_Group_Comparison.csv")
    rn = pd.read_csv(K.csvp(f)).set_index("Metric")
    lv_f = os.path.join(K.OUT, "comparison", "Length_vs_HeartRate.csv")
    lv = pd.read_csv(K.csvp(lv_f))
    hr_f = cmp_("HR_Adjusted_Comparison.csv")
    hr = pd.read_csv(K.csvp(hr_f)).set_index("Metric")
    rq_f = cmp_("Requantization_Q128.csv")
    rq = pd.read_csv(K.csvp(rq_f)).set_index("Metric")
    ws_f = os.path.join(K.OUT, "comparison", "Window_Skip_Sensitivity.csv")
    ws = pd.read_csv(K.csvp(ws_f))
    dc_f = cmp_("Deterministic_Checks.csv")
    dc = pd.read_csv(K.csvp(dc_f)).set_index("Quantity")
    cols = ["RNA_ClusteringCoeff", "RNA_Transitivity"]
    rows = [["Mean, controls; patients"] + [f"{U(rn.loc[c, 'Healthy_Mean'], 4)}; {U(rn.loc[c, 'Psych_Mean'], 4)}" for c in cols]]
    rows.append(["Observed (*p* adjusted within the 2-test family)"] +
                 [f"{R(rn.loc[c, 'r_effect'])} ({P3(rn.loc[c, 'p_raw'])}; adj {P3(rn.loc[c, 'p_adj'])})" for c in cols])
    for c in cols:
        rr, pp, _, _ = mw(K.C[c], K.P[c])
        assert close(rr, rn.loc[c, "r_effect"], 1e-5)
    def mwrow(label, suffix):
        out = [label]
        for c in cols:
            rr, pp, n1, n2 = mw(K.C[c + suffix], K.P[c + suffix])
            out.append(f"{R(rr)} ({P3(pp)})")
        rows.append(out)
    mwrow("Dithered (one draw)", "_Dither")
    mwrow("Matched to step-to-RMSSD 0.088", "_MatchedQuant")
    assert (num(K.C.RNA_Controls_NBeats).min() == 1764 or num(K.P.RNA_Controls_NBeats).min() == 1764)
    mwrow("First 1,764 clean intervals (shortest series)", "_Trunc")
    for cond, label in (("fixed 2000 beats", "First 2,000 clean intervals (29 controls, 30 patients)"),
                        ("fixed 2400 beats", "First 2,400 clean intervals (18 controls, 27 patients)")):
        out = [label]
        for c in cols:
            rw = lv[(lv.Metric == c) & (lv.Condition == cond)].iloc[0]
            out.append(f"{R(rw.r_effect)} ({P3(rw.p_raw)})")
        rows.append(out)
    rows.append(["Mean heart rate held constant (partial *r*)"] +
                 [f"{R(hr.loc[c, 'r_partial_HR'])} ({P3(hr.loc[c, 'p_raw'])})" for c in cols])
    rows.append(["Re-quantized to 7.8125 ms"] + [f"{R(rq.loc[c, 'r_Q128'])} ({P3(rq.loc[c, 'p_Q128'])})" for c in cols])
    for sk in (60, 360):
        rows.append([f"Onset skip {sk} s"] + [
            f"{R(ws[(ws.Metric == c) & (ws.Skip_s == sk)].iloc[0].r_effect)} ({P3(ws[(ws.Metric == c) & (ws.Skip_s == sk)].iloc[0].p_raw)})"
            for c in cols])
    # The note says clustering differed only at 2,400 beats and transitivity in no condition.
    sig_cells = [(r[0], k) for r in rows[1:] for k, cell in enumerate(r[1:]) if "(" in cell and
                 cell.split("(")[1].split(";")[0].split(")")[0].strip() not in ("–",) and
                 (cell.split("(")[1].split(";")[0].split(")")[0].strip() == "< 0.001" or
                  float(cell.split("(")[1].split(";")[0].split(")")[0].strip().replace("< ", "")) < 0.05)]
    assert sig_cells == [("First 2,400 clean intervals (18 controls, 27 patients)", 0), ("Onset skip 360 s", 1)], sig_cells
    t360 = ws[(ws.Metric == "RNA_Transitivity") & (ws.Skip_s == 360)].iloc[0]
    FACTS["s.tra360_r"], FACTS["s.tra360_p"] = R(t360.r_effect), P3(t360.p_raw)
    md = dc.loc["RNA_MeanDegree"]
    note = ("Rank-biserial *r* (positive = higher in patients) with uncorrected *p* except where stated. Both measures "
            "are built from the recurrence matrix of the primary recurrence quantification (dimension 2, delay 1, "
            "10% of eligible pairs, per-subject Theiler window). The dithered, matched and truncated versions keep "
            "each subject's Theiler window from the primary run (rna_controls.py). Two cells reach uncorrected *p* < 0.05: "
            "clustering at 2,400 beats, in the opposite direction and in a subsample selected by heart rate (only the 18 "
            "controls with at least 2,400 beats), and transitivity at the 360 s onset skip "
            f"({FACTS['s.tra360_r']}, *p* = {FACTS['s.tra360_p']}), which falls to {R(rn.loc['RNA_Transitivity', 'r_effect'])} at "
            "the primary 600 s skip. Neither survives correction for the conditions examined. The mean network degree, fixed by construction, was "
            f"{U(md.Control_mean, 4)} in controls and {U(md.Patient_mean, 4)} in patients (largest deviation from 0.1: "
            f"{U(md.Max_abs_deviation, 4)}); the recurrence rate was 0.1000 in every subject.")
    return dict(header=["Condition", "Network clustering", "Network transitivity"], rows=rows, note=note,
                sources=src(f, lv_f, hr_f, rq_f, ws_f, dc_f))


# =========================================================================== S14 clinical ranges
@table("clinical")
def t_clinical():
    f = cmp_("Clinical_Range_Counts.csv")
    cr = pd.read_csv(K.csvp(f))
    pf = cmp_("pNN50_Floor_Report.csv")
    pn = pd.read_csv(K.csvp(pf)).set_index("Group")
    rows = []
    for crit in cr.Criterion.unique():
        sub = cr[cr.Criterion == crit].set_index("Group")
        c, p = sub.loc["Healthy"], sub.loc["Psychiatric"]
        rows.append([crit.replace("mean HR", "Mean heart rate").replace(">", ">").replace("<", "<"),
                     f"{int(c.N_Meeting)} ({U(c.Pct, 1)}%)", f"{int(p.N_Meeting)} ({U(p.Pct, 1)}%)"])
    for key, label in (("N_At_Zero", "pNN50 exactly 0%"), ("N_Below_1pct", "pNN50 below 1%")):
        rows.append([label, f"{int(pn.loc['Healthy', key])} ({U(100 * pn.loc['Healthy', key] / 30, 1)}%)",
                     f"{int(pn.loc['Psychiatric', key])} ({U(100 * pn.loc['Psychiatric', key] / 30, 1)}%)"])
    # verify against the per-subject values
    assert int((num(K.P.HRV_MeanHR) > 100).sum()) == 12 and int((num(K.C.HRV_MeanHR) > 100).sum()) == 3
    assert int((num(K.P.HRV_pNN50) == 0).sum()) == 7 and int((num(K.P.HRV_pNN50) < 1).sum()) == 21
    note = ("Counts of the 30 controls and 30 patients meeting each criterion over the 30-minute analysis window. "
            "The window contains an instructed walk and other activity, so these counts are not comparable with "
            "resting clinical norms and are given only to show where the patient distributions sit. pNN50 is "
            "bounded below at zero, which compresses the patient distribution; it is reported but carries little "
            "information beyond RMSSD here.")
    return dict(header=["Criterion", "Controls, *n* (%)", "Patients, *n* (%)"], rows=rows, note=note, sources=src(f, pf))


# =========================================================================== S15 non-redundant family
@table("nonredundant")
def t_nonredundant():
    f = cmp_("Group_Comparison_NonRedundant.csv")
    d = pd.read_csv(K.csvp(f)).rename(columns={"Unnamed: 0": "L"})
    g = pd.read_csv(K.csvp(cmp_("group_comparison_stats.csv"))).rename(columns={"Unnamed: 0": "L"}).set_index("L")
    assert len(d) == 18 and np.allclose(bh(d.p_raw.values), d.p_adj.values, atol=2e-5)
    rows = []
    for _, rw in d.iterrows():
        rows.append([lab(K.CSV_LABEL[rw.L]), R(rw.r_effect), P3(rw.p_raw), P3(rw.p_adj), P3(g.loc[rw.L, "p_adj"])])
    nsig = int((d.p_adj < 0.05).sum())
    assert str(nsig) == FACTS["nr.n_sig"]
    flip = d[(d.p_adj >= 0.05) & (d.L.map(lambda l: g.loc[l, "p_adj"] < 0.05))].L.tolist()
    assert flip == ["SD1/SD2"], flip
    note = ("SD1 (identical to RMSSD/√2), SD2 (determined by SDNN and SD1) and mean heart rate (the reciprocal of "
            "the mean NN interval) are removed and Benjamini-Hochberg correction is applied over the remaining 18 "
            f"measures; {nsig} of 18 differ. SD1/SD2 is the one measure that changes status (adjusted *p* "
            f"{P3(d.set_index('L').loc['SD1/SD2', 'p_adj'])} against {P3(g.loc['SD1/SD2', 'p_adj'])} in the 21-test family). "
            "*r* is the rank-biserial correlation (positive = higher in patients).")
    return dict(header=["Measure", "*r*", "*p*", "*p* adj (18 tests)", "*p* adj (21 tests)"], rows=rows, note=note,
                sources=src(f))


# =========================================================================== S16 intercorrelation
@table("intercorr")
def t_intercorr():
    fh = cmp_("Metric_Intercorrelation_Healthy.csv")
    fp = cmp_("Metric_Intercorrelation_Psychiatric.csv")
    h = pd.read_csv(K.csvp(fh), index_col=0)
    p = pd.read_csv(K.csvp(fp), index_col=0)
    order = ["LF Power (ms²)", "Total Power (ms²)", "RMSSD (ms)", "SD1 (ms)", "pNN50 (%)", "HF Power (ms²)", "SDNN (ms)",
             "SD2 (ms)", "Mean NN (ms)", "Mean HR (bpm)", "SD1/SD2", "PSD β", "DFA α₂ (16–64)", "DFA α (full)",
             "MSE Mean", "Permutation Entropy", "Sample Entropy"]
    short = ["LF", "TP", "RMSSD", "SD1", "pNN50", "HF", "SDNN", "SD2", "NN", "HR", "SD1/SD2", "Slope", "α₂", "α",
             "MSE", "PE", "SE"]
    assert set(order) == set(h.index) == set(p.index)
    # spot-check against the per-subject data
    rc = spearmanr(num(K.C.PSD_Beta), num(K.C.HRV_RMSSD))[0]
    assert close(round(rc, 4), h.loc["PSD β", "RMSSD (ms)"], 1e-4)
    h, p = h.loc[order, order], p.loc[order, order]
    # the note names three definitional pairs; they round to ±1.00 in both groups and no other pair does
    pairs = {("RMSSD (ms)", "SD1 (ms)"), ("SDNN (ms)", "SD2 (ms)"), ("Mean NN (ms)", "Mean HR (bpm)")}
    for M_ in (h, p):
        hi = {(a, b) for i_, a in enumerate(order) for b in order[i_ + 1:] if abs(M_.loc[a, b]) >= 0.995}
        assert {tuple(sorted(x)) for x in hi} == {tuple(sorted(x)) for x in pairs}, hi
    rows = []
    for i, li in enumerate(order):
        row = [short[i]]
        for j, lj in enumerate(order):
            if i == j:
                row.append("–")
            elif i > j:
                row.append(U(h.loc[li, lj], 2))
            else:
                row.append(U(p.loc[li, lj], 2))
        rows.append(row)
    note = ("Spearman correlations among the 17 measures that differed in the primary family, within each group: "
            "controls below the diagonal, patients above it. LF, HF and TP, low-frequency, high-frequency and total power; NN, mean NN "
            "interval; HR, mean heart rate; Slope, uncorrected spectral slope; α₂ and α, DFA exponents; MSE, "
            "multiscale entropy; PE, permutation entropy; SE, sample entropy. Three pairs are identities or "
            "near-identities by definition (RMSSD and SD1, SDNN and SD2, NN and HR). Approximate entropy, outside "
            f"the family, correlates with sample entropy at {FACTS['cor.apen_sampen_c']} in controls and "
            f"{FACTS['cor.apen_sampen_p']} in patients. Supplementary Figure {SFNUM['intercorr']} shows the same matrices and "
            f"Supplementary Results {RNUM['covariance']} describes them.")
    return dict(header=[""] + short, rows=rows, note=note, landscape=True, sources=src(fh, fp), font=14)


# =========================================================================== S17 heart rate
@table("hr")
def t_hr():
    f = cmp_("HR_Adjusted_Comparison.csv")
    h = pd.read_csv(K.csvp(f))
    assert len(h) == 21 and np.allclose(bh(h.p_raw.values), h.p_adj.values, atol=2e-5)
    fb = cmp_("HR_Adjusted_Bootstrap.csv")
    bt = pd.read_csv(K.csvp(fb)).set_index("Metric")
    assert sorted(bt.index) == sorted(h.Metric) and (bt.N_Boot == 2000).all() and (bt.Seed == 0).all()
    assert (bt.N_boot_kept >= 1980).all()
    G = np.r_[np.zeros(30), np.ones(30)]
    CP = pd.concat([K.C, K.P])
    HR = num(CP.HRV_MeanHR).values
    rows = []
    for _, rw in h.iterrows():
        y = num(CP[rw.Metric]).values
        ru, pu = partial(y, G, [])
        ra, pa = partial(y, G, [HR])
        assert close(ru, rw.r_partial_unadjusted, 1e-5) and close(ra, rw.r_partial_HR, 1e-5) and close(pa, rw.p_raw, 1e-6)
        b = bt.loc[rw.Metric]
        assert close(b.r_partial_unadjusted, ru, 1e-7) and close(b.r_partial_HR, ra, 1e-7)
        assert b.unadj_CI_low <= ru <= b.unadj_CI_high and b.HR_CI_low <= ra <= b.HR_CI_high
        assert bool(b.HR_CI_excludes_0) == bool(rw.p_raw < 0.05), rw.Metric      # interval and test agree here
        forms = [rw.r_partial_HR, rw.r_HR_quadratic, rw.r_HR_log_measure, rw.r_HR_log_log_meanNN, rw.r_HR_rank_based]
        lo, hi = min(forms), max(forms)
        rows.append([lab(rw.Metric), R(rw.r_rank_biserial),
                     f"{R(ru)} [{R(b.unadj_CI_low)}, {R(b.unadj_CI_high)}]",
                     f"{R(ra)} [{R(b.HR_CI_low)}, {R(b.HR_CI_high)}]",
                     P3(rw.p_raw), P3(rw.p_adj),
                     f"{R(b.delta_HR_minus_unadj)} [{R(b.delta_CI_low)}, {R(b.delta_CI_high)}]",
                     f"{R(lo)} to {R(hi)}",
                     f"{R(rw.r_partial_beats)} ({P3(rw.p_partial_beats)})"])
        FACTS[f"s.hrb.{rw.Metric}.a_ci"] = f"95% CI {R(b.HR_CI_low)} to {R(b.HR_CI_high)}"
        FACTS[f"s.hrb.{rw.Metric}.d_ci"] = f"{R(b.delta_CI_low)} to {R(b.delta_CI_high)}"
    nn_f = cmp_("HR_Normalised_MeanNN.csv")
    nn = pd.read_csv(K.csvp(nn_f))
    rows.append({"section": "Amplitude measures divided by the mean NN interval (SD measures) or its square (powers)"})
    rows.append({"subheader": ["Normalized measure", "*r*", "", "", "*p*", "", "", "", ""]})
    for _, rw in nn.iterrows():
        rows.append([lab(rw.Measure) + (" / mean NN" if rw.MeanNN_Exponent == 1 else " / mean NN²"), R(rw.r_effect),
                     "", "", P3(rw.p_raw), "", "", "", ""])
    note = ("Columns 2 to 6: rank-biserial *r*; the partial Spearman correlation of the measure with group without a "
            "covariate and with mean heart rate held constant (both regressed on the covariate by least squares with "
            "an intercept, residuals rank-correlated), each with a percentile bootstrap 95% interval (2,000 resamples "
            "of subjects within each group, seed 0, the same resamples for both); the *p* of the adjusted partial; and "
            "that *p* adjusted over these 21 tests. Change, the adjusted minus the unadjusted partial *r*, with its "
            "interval computed within each resample: an interval that excludes zero means holding heart rate constant "
            "moved the estimate. For every measure the adjusted interval excludes zero exactly when the adjusted "
            "*p* is below 0.05. Model forms, the range of the "
            "adjusted partial *r* across five forms (linear, quadratic in heart rate, log of the measure, log of the "
            "measure on log mean NN, and fully rank-based). Beat count, the partial *r* with the number of clean "
            "intervals as the covariate instead of heart rate (*r* = "
            f"{FACTS['hrid.r_all']} between the two over the 60 recordings). Positive values mean higher in patients. "
            "Heart rate is plausibly a consequence of group membership, so an adjusted value is the part of a difference "
            "not carried linearly by heart rate, not a causal estimate.")
    return dict(header=["Measure", "*r*", "Partial *r*, unadjusted [95% CI]",
                        "Partial *r*, heart rate held constant [95% CI]", "*p*", "*p* adj", "Change [95% CI]",
                        "Model forms", "Beat count held constant"],
                rows=rows, note=note, landscape=True, sources=src(f, fb, nn_f))


# =========================================================================== S18 fixed beat counts
@table("fixedbeats")
def t_fixedbeats():
    f = os.path.join(K.OUT, "comparison", "Length_vs_HeartRate.csv")
    lv = pd.read_csv(K.csvp(f))
    rows = []
    for m in lv.Metric.unique():
        sub = lv[lv.Metric == m].set_index("Condition")
        o, a, b = sub.loc["observed"], sub.loc["fixed 2000 beats"], sub.loc["fixed 2400 beats"]
        assert (int(a.N_control), int(a.N_patient)) == (29, 30) and (int(b.N_control), int(b.N_patient)) == (18, 27)
        rr, pp, _, _ = mw(K.C[m], K.P[m])
        assert close(rr, o.r_effect, 1e-5)
        rows.append([lab(m), f"{R(o.r_effect)} ({P3(o.p_raw)})", f"{R(a.r_effect)} ({P3(a.p_raw)})",
                     f"{R(b.r_effect)} ({P3(b.p_raw)})", f"{R(o.r_adj_beats)} ({P3(o.p_adj_beats)})",
                     f"{R(o.r_adj_HR)} ({P3(o.p_adj_HR)})"])
    nmin_c = int(num(K.C.Num_RR).min())
    note = ("Rank-biserial *r* (positive = higher in patients) with uncorrected *p*, except the last two columns, "
            "which are partial correlations from the full window with the clean-interval count or mean heart rate "
            "held constant. The fixed-count columns use the first 2,000 or 2,400 clean intervals of each subject's "
            f"window, so series length is equal by design; subjects with fewer intervals are dropped (the shortest "
            f"control series held {INT(nmin_c)}). Equal counts mean unequal durations, because 2,000 beats last longer "
            "at a lower heart rate, and the 2,400-beat subsample keeps only the 18 controls with the highest heart "
            "rates, so it is selected on the variable in question and is the weaker check.")
    return dict(header=["Measure", "Full window (30, 30)", "First 2,000 (29, 30)", "First 2,400 (18, 27)",
                        "Beat count held constant", "Heart rate held constant"],
                rows=rows, note=note, landscape=True, sources=src(f))


# =========================================================================== S19 re-quantization
@table("requant")
def t_requant():
    f = cmp_("Requantization_Q128.csv")
    q = pd.read_csv(K.csvp(f))
    rows = []
    for _, rw in q.iterrows():
        rq, pq, _, _ = mw(K.C[rw.Metric + "_Q128"], K.P[rw.Metric + "_Q128"])
        assert close(rq, rw.r_Q128, 1e-5)
        rows.append([lab(rw.Metric), f"{SIG(rw.Healthy_Median_Native)}; {SIG(rw.Psych_Median_Native)}",
                     f"{SIG(rw.Healthy_Median_Q128)}; {SIG(rw.Psych_Median_Q128)}", f"{R(rw.r_native)} ({P3(rw.p_native)})",
                     f"{R(rw.r_Q128)} ({P3(rw.p_Q128)})", R(rw.r_change)])
    ff = cmp_("Requantization_Resolution_Floor.csv")
    fl = pd.read_csv(K.csvp(ff))
    names = {"Requant_RMSSD_In_Steps": "RMSSD in grid steps, 7.8125 ms grid",
             "Requant_Tie_Rate": "Ordinal tie rate, 7.8125 ms grid",
             "Ordinal_Tie_Rate": "Ordinal tie rate, 1.000 ms grid", "Quant_Step_Over_RMSSD": "Step-to-RMSSD ratio, 1.000 ms grid"}
    rows.append({"section": "Resolution of each grid relative to the series"})
    rows.append({"subheader": ["Quantity", "Controls, median (range)", "Patients, median (range)", "", "*r* (*p*)", ""]})
    for _, rw in fl.iterrows():
        rows.append([names[rw.Quantity], f"{SIG(rw.Healthy_Median)} ({SIG(rw.Healthy_Min)}–{SIG(rw.Healthy_Max)})",
                     f"{SIG(rw.Psych_Median)} ({SIG(rw.Psych_Min)}–{SIG(rw.Psych_Max)})", "",
                     f"{R(rw.r_effect)} ({P3(rw.p_raw)})", ""])
    allq = pd.concat([K.C, K.P])
    assert (allq.Requant_Model.astype(str) == "interval_rounding").all()
    note = ("Every HRV-ACC interval rounded to the nearest multiple of 7.8125 ms (the sampling period of a 128 Hz "
            "recording) and every measure recomputed; full-range DFA α was not recomputed. Medians are given as "
            "controls; patients. *r* is the rank-biserial correlation (positive = higher in patients) with "
            "uncorrected *p*; Change is *r* on the coarse grid minus *r* at 1.000 ms. Rounding each interval, rather "
            "than the beat times as a 128 Hz recorder would, is a simplification (Supplementary Methods "
            f"{MNUM['quantization']}). In the lower block, the "
            "first two rows refer to the coarse grid and the last two to the recorded grid; *r* there compares the "
            f"two groups on that quantity. Main Figure 3 shows the same results.")
    return dict(header=["Measure", "Median, 1.000 ms", "Median, 7.8125 ms", "*r* (*p*), 1.000 ms", "*r* (*p*), 7.8125 ms",
                        "Change"], rows=rows, note=note, landscape=True, sources=src(f, ff))


# =========================================================================== S20 DFA scale ranges
@table("scaleband")
def t_scaleband():
    f = cmp_("Scale_Band_Correspondence.csv")
    s = pd.read_csv(K.csvp(f))
    rows = []
    for _, rw in s.iterrows():
        g = "Controls" if rw.Group == "Healthy" else "Patients"
        mnn = num((K.C if g == "Controls" else K.P).HRV_MeanNN).mean() / 1000
        assert close(mnn, rw.Mean_NN_s, 1e-4)
        rows.append([g, "α₁" if rw.Exponent == "alpha1" else "α₂", f"{int(rw.Scale_Lo_Beats)}–{int(rw.Scale_Hi_Beats)}",
                     U(rw.Mean_NN_s), f"{U(rw.Freq_Lo_Hz)}–{U(rw.Freq_Hi_Hz)}", U(rw.Overlap_With_Beta_Band_Hz)])
    alt = []
    for col, label in (("DFA_Alpha1", "α₁, 4–16 beats (primary)"), ("DFA_Alpha1_nmin8", "α₁, 8–16 beats"),
                       ("DFA_Alpha1_nmin10", "α₁, 10–16 beats"), ("DFA_Alpha1_Disjoint", "α₁, 4–15 beats (no scale shared with α₂)"),
                       ("DFA_TimeMatched_4s_16s", "Time-matched, 4–16 s"), ("DFA_Alpha2", "α₂, 16–64 beats (primary)"),
                       ("DFA_TimeMatched_16s_64s", "Time-matched, 16–64 s")):
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        alt.append([label, U(num(K.C[col]).median()), U(num(K.P[col]).median()), R(rr), P3(pp)])
        FACTS[f"s.dfa.{col}"] = R(rr)
        FACTS[f"s.dfa.{col}_pe"] = K.PE(pp)
    lo_c = num(K.C.DFA_TimeMatched_4s_16s_LoBeats); hi_c = num(K.C.DFA_TimeMatched_4s_16s_HiBeats)
    lo_p = num(K.P.DFA_TimeMatched_4s_16s_LoBeats); hi_p = num(K.P.DFA_TimeMatched_4s_16s_HiBeats)
    FACTS["s.tm_c"] = f"{int(lo_c.median())} to {int(hi_c.median())}"
    FACTS["s.tm_p"] = f"{int(lo_p.median())} to {int(hi_p.median())}"
    rows.append({"section": "Group comparisons under alternative scale definitions"})
    rows.append({"subheader": ["Scale definition", "Controls, median", "Patients, median", "*r*", "*p*", ""]})
    for a in alt:
        rows.append([a[0], a[1], a[2], a[3], a[4], ""])
    note = ("Upper block: a window of *n* beats at the group's mean NN interval *T* spans *n*·*T* seconds, so the "
            "fluctuation it measures corresponds to frequencies near 1/(*n*·*T*); the frequency range is 1/(*n*_hi·*T*) "
            "to 1/(*n*_lo·*T*). Overlap is the width of that range inside the slope's fitted range (0.0156–0.15 Hz). "
            "Lower block: medians and the rank-biserial *r* (positive = higher in patients) with uncorrected *p*. "
            "The time-matched exponents convert 4–16 s and 16–64 s into beats for each subject from that subject's own "
            f"mean NN interval (median {FACTS['s.tm_c']} beats in controls and {FACTS['s.tm_p']} in patients for 4–16 s), "
            "so the frequencies covered are the same for every subject. The 8- and 10-beat lower bounds are those of "
            "main Table 4.")
    return dict(header=["Group", "Exponent", "Beats", "Mean NN (s)", "Frequency range (Hz)", "Overlap (Hz)"],
                rows=rows, note=note, sources=src(f))


# =========================================================================== S21 control cohorts
@table("controls")
def t_controls():
    f = os.path.join(K.XC, "Control_vs_Control_Stats.csv")
    c = pd.read_csv(K.csvp(f)).rename(columns={"Unnamed: 0": "L"})
    # Sign check: r_effect positive = higher in HRV-ACC (B); A is MIT-BIH.
    a1 = c.set_index("L").loc["DFA α₁ (4–16)"]
    rr, pp, _, _ = mw(K.M.DFA_Alpha1, K.C.DFA_Alpha1)
    assert close(rr, a1.r_effect, 1e-5) and a1.B_Mean > a1.A_Mean and close(num(K.M.DFA_Alpha1).mean(), a1.A_Mean, 1e-5)
    rows = []
    for _, rw in c.iterrows():
        rows.append([lab(K.CSV_LABEL[rw.L]), f"{SIG(rw.A_Mean)} ({SIG(rw.A_SD)})", f"{SIG(rw.B_Mean)} ({SIG(rw.B_SD)})",
                     R(rw.r_effect), P3(rw.MWU_p_raw), P3(rw.MWU_p_adj), P3(rw.Welch_p),
                     f"{R(rw.Cohen_d, 2)} [{R(rw.d_CI_low, 2)}, {R(rw.d_CI_high, 2)}]"])
    tf = os.path.join(K.XC, "Control_Equivalence_TOST.csv")
    t = pd.read_csv(K.csvp(tf))
    assert not t.Equivalent.any()
    rows.append({"section": "Two one-sided tests of equivalence within ±0.5 SD (Schuirmann, 1987)"})
    rows.append({"subheader": ["Measure", "Mean difference (HRV-ACC − MIT-BIH)", "Cohen's *d*", "TOST *p*", "Equivalent", "", "", ""]})
    for _, rw in t.iterrows():
        rows.append([lab(K.CSV_LABEL[rw.Metric]), SIG(rw.Mean_Diff), R(rw.Cohen_d, 2), P3(rw.p_TOST), "No", "", "", ""])
    note = ("MIT-BIH (18 records) against HRV-ACC controls (*n* = 30). *r* is the rank-biserial correlation with positive "
            "values meaning higher in HRV-ACC controls; *p* adj is Benjamini-Hochberg over the 27 quantities. Welch *p* "
            "is uncorrected; *d* is Cohen's *d* with an approximate 95% interval (positive = higher in HRV-ACC). The "
            "two cohorts differ in device, resolution, setting, era and age at once, so a difference here bounds how "
            "much a measure moves when everything except health status changes; it does not isolate a device effect. "
            "Only DFA α₁ differs after correction. The equivalence tests use the pooled SD and a *t* distribution; "
            "none shows equivalence, so a null difference test here is not evidence that the cohorts agree.")
    return dict(header=["Measure", "MIT-BIH, mean (SD)", "HRV-ACC, mean (SD)", "*r*", "*p*", "*p* adj", "Welch *p*",
                        "*d* [95% CI]"], rows=rows, note=note, landscape=True, sources=src(f, tf))


# =========================================================================== S22 embedding
@table("embedding")
def t_embedding():
    f = cmp_("Embedding_Sensitivity.csv")
    e = pd.read_csv(K.csvp(f))
    rows = []
    lab_e = {("Embed_Delay_MI", "selected_parameter"): "Delay, first minimum of mutual information (median)",
             ("Embed_Dim_FNN", "selected_parameter"): "Dimension, false nearest neighbors (median)",
             ("RQA_DET", "primary_embedding"): "Determinism, dimension 2 and delay 1 (median)",
             ("RQA_DET", "selected_embedding"): "Determinism, each subject's own embedding (median)",
             ("RQA_LAM", "primary_embedding"): "Laminarity, dimension 2 and delay 1 (median)",
             ("RQA_LAM", "selected_embedding"): "Laminarity, each subject's own embedding (median)"}
    for _, rw in e.iterrows():
        nd = 0 if rw.Type == "selected_parameter" else 3
        rows.append([lab_e[(rw.Quantity, rw.Type)], U(rw.Healthy_Median, nd), U(rw.Psych_Median, nd), R(rw.r_effect), P3(rw.p_raw)])
    for col, label in (("RQA_DET_lmin3", "Determinism, minimum line length 3"), ("RQA_DET_lmin4", "Determinism, minimum line length 4")):
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        rows.append([label + " (median)", U(num(K.C[col]).median()), U(num(K.P[col]).median()), R(rr), P3(pp)])
    rows.append({"section": "Selection at the search limits"})
    cnt = {}
    for key, cond, label in (("d30", lambda D: num(D.Embed_Delay_MI) == 30, "Delay at the 30-sample limit (no local minimum)"),
                             ("d3", lambda D: num(D.Embed_Delay_MI) <= 3, "Delay of 3 samples or fewer"),
                             ("m10", lambda D: num(D.Embed_Dim_FNN) == 10, "Dimension at the limit of 10")):
        a, b = int(cond(K.C).sum()), int(cond(K.P).sum())
        cnt[key] = (a, b)
        rows.append([label, f"{a} of 30", f"{b} of 30", "–", P3(fisher(a, 30, b, 30))])
    tw_c, tw_p = num(K.C.RQA_Theiler_W), num(K.P.RQA_Theiler_W)
    assert (tw_c == num(K.C.Embed_Delay_MI)).all() and (tw_p == num(K.P.Embed_Delay_MI)).all()
    rows.append(["Theiler window, primary embedding (range, samples)", f"{int(tw_c.min())}–{int(tw_c.max())}",
                 f"{int(tw_p.min())}–{int(tw_p.max())}", "–", "–"])
    FACTS.update({"s.emb_d30_c": str(cnt["d30"][0]), "s.emb_d30_p": str(cnt["d30"][1]),
                  "s.emb_d3_c": str(cnt["d3"][0]), "s.emb_d3_p": str(cnt["d3"][1]),
                  "s.emb_m10_c": str(cnt["m10"][0]), "s.emb_m10_p": str(cnt["m10"][1]),
                  "s.emb_d30_fe": P3(fisher(*cnt["d30"][:1], 30, cnt["d30"][1], 30)),
                  "s.emb_d3_fe": P3(fisher(cnt["d3"][0], 30, cnt["d3"][1], 30)),
                  "s.tw_c": f"{int(tw_c.min())} to {int(tw_c.max())}", "s.tw_p": f"{int(tw_p.min())} to {int(tw_p.max())}"})
    note = ("Rank-biserial *r* (positive = higher in patients) with uncorrected *p*; counts compared by Fisher's "
            "exact test. The delay is the first local minimum of the mutual information between the series and its "
            "lagged copy (16 bins, lags 1 to 30), or the lag of the global minimum when there is no local one "
            "(Fraser and Swinney, 1986); the same lag sets the Theiler window of the primary recurrence analysis. "
            "The dimension is the first at which fewer than 1% of nearest neighbors are false (Kennel et al., 1992), "
            "with tolerance 15, attractor-size criterion 2 SD, the first 1,500 points and dimensions 1 to 10. At each subject's "
            "own embedding the Theiler window is the embedding span, (dimension − 1) × delay + 1. Supplementary "
            f"Results {RNUM['embedding']} discusses the selections.")
    return dict(header=["Quantity", "Controls", "Patients", "*r*", "*p*"], rows=rows, note=note, sources=src(f))


# =========================================================================== S23 surrogates
@table("surrogate")
def t_surrogate():
    f = cmp_("Surrogate_Normalized_Entropy.csv")
    s = pd.read_csv(K.csvp(f))
    assert np.allclose(bh(s.p_raw.values), s.p_adj.values, atol=2e-5)
    rows = [{"section": "Group comparisons on raw values and on surrogate Z-scores (six-test family)"}]
    nm = {"SampEn": "Sample entropy", "DFA_Alpha1": "DFA α₁", "MSE_Mean": "Multiscale entropy"}
    for _, rw in s.iterrows():
        col = rw.Column
        rr, pp, _, _ = mw(K.C[col], K.P[col])
        assert close(rr, rw.r_effect, 1e-5)
        rows.append([nm[rw.Metric], "raw value" if rw.Scale == "raw" else "surrogate Z-score",
                     U(num(K.C[col]).median()), U(num(K.P[col]).median()), R(rw.r_effect), P3(rw.p_raw), P3(rw.p_adj)])
    df = cmp_("Surrogate_Distribution_Summary.csv")
    d = pd.read_csv(K.csvp(df))
    rows.append({"section": "Subjects whose observed value lies outside the central 95% of their own 100 surrogates"})
    rows.append({"subheader": ["Measure", "Quantity", "Controls", "Patients", "", "Fisher *p*", ""]})
    fishers = {}
    for m in ["SampEn", "DFA_Alpha1", "MSE_Mean"]:
        hc = d[(d.Metric == m) & (d.Group == "Healthy")].iloc[0]
        hp = d[(d.Metric == m) & (d.Group == "Psychiatric")].iloc[0]
        sigc = int(K.C[f"SURR_{m}_Sig"].astype(str).str.lower().eq("true").sum())
        sigp = int(K.P[f"SURR_{m}_Sig"].astype(str).str.lower().eq("true").sum())
        assert (sigc, sigp) == (int(hc.N_Differing), int(hp.N_Differing))
        fp = fisher(sigc, 30, sigp, 30)
        fishers[m] = fp
        rows.append([nm[m], "observed; surrogate mean",
                     f"{U(hc.Observed_Mean)}; {U(hc.Surrogate_Mean)}", f"{U(hp.Observed_Mean)}; {U(hp.Surrogate_Mean)}",
                     "", "", ""])
        rows.append(["", "outside (above, below)", f"{int(hc.N_Differing)} ({int(hc.N_Above_Surrogates)}, {int(hc.N_Below_Surrogates)})",
                     f"{int(hp.N_Differing)} ({int(hp.N_Above_Surrogates)}, {int(hp.N_Below_Surrogates)})", "", P3(fp), ""])
    assert all(v > 0.05 for v in fishers.values())
    FACTS["s.surr_fisher_min"] = P3(min(fishers.values()))
    note = ("Upper block: medians, rank-biserial *r* (positive = higher in patients), uncorrected *p* and *p* "
            "adjusted by Benjamini-Hochberg over the six tests. The Z-score is (observed − mean of the subject's "
            "surrogates)/SD of the surrogates. Lower block: means of the observed values and of the surrogate means, "
            "and the number of subjects outside the 2.5th to 97.5th percentiles of their own surrogates, compared "
            "between groups by Fisher's exact test. Iteratively amplitude-adjusted Fourier "
            "transform surrogates preserve each series' amplitude distribution and, approximately, its power spectrum "
            "(Schreiber and Schmitz, 1996). A departure indicates structure beyond those two properties, which may be "
            "nonlinear dynamics or nonstationarity. Supplementary Figure "
            f"{SFNUM['surrogates']} shows the per-subject Z-scores.")
    return dict(header=["Measure", "Form", "Controls", "Patients", "*r*", "*p*", "*p* adj"], rows=rows, note=note,
                landscape=True, sources=src(f, df))
