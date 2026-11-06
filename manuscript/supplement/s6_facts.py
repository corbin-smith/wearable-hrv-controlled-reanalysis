"""
s6_facts.py: values printed in the Supplementary Methods and Results text that no table computes as a side effect.

Every value is read from a results file or recomputed from the per-subject files, and where the text states a
pattern ("did not differ", "in every subject") the pattern is asserted here.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr, norm

import s6_common as K
from s6_common import R, U, P3, PE, INT, SIG, mw, num, code_const, FACTS


def add(key, val):
    assert key.startswith("s."), key
    if key in FACTS:
        raise SystemExit(f"duplicate supplement fact {key}")
    FACTS[key] = val


def fisher(a, n1, b, n2):
    return float(fisher_exact([[a, n1 - a], [b, n2 - b]])[1])


def true_count(s):
    return int(pd.Series(s).astype(str).str.lower().isin(["true", "1", "1.0"]).sum())


def compute():
    C, P, M = K.C, K.P, K.M
    CP = pd.concat([C, P])

    # ------------------------------------------------------------------ S1 protocol, window, onset check
    ws = pd.read_csv(K.csvp(os.path.join(K.OUT, "comparison", "Window_Skip_Sensitivity.csv")))
    hr = ws[ws.Metric == "HRV_MeanHR"].set_index("Skip_s")
    for sk in (60, 360, 600):
        add(f"s.skip_hr_c_{sk}", U(hr.loc[sk, "Control_mean"], 1))
        add(f"s.skip_hr_p_{sk}", U(hr.loc[sk, "Patient_mean"], 1))
    span_p = hr.Patient_mean.max() - hr.Patient_mean.min()
    add("s.skip_hr_span_p", U(span_p, 1))
    assert span_p < 2.0, "text says the skip barely changes patients' mean heart rate"
    d = num(M.Onset_HR_Diff)
    assert d.notna().sum() == 18
    add("s.onset_mean", U(d.mean(), 2))
    add("s.onset_sd", U(d.std(), 2))
    add("s.onset_npos", str(int((d > 0).sum())))
    from scipy.stats import wilcoxon
    pw = float(wilcoxon(d).pvalue)
    add("s.onset_pe", PE(pw))
    assert np.allclose(num(M.MITBIH_Skip_Seconds), 600)
    clk = M.Record_Start_Clock.astype(str)
    assert clk.ne("unknown").all()
    secs = clk.map(lambda c: int(c[:2]) * 3600 + int(c[3:5]) * 60 + int(c[6:8]))
    fmt = lambda x: f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}"
    add("s.mit_clock_lo", fmt(secs.min())); add("s.mit_clock_hi", fmt(secs.max()))
    add("s.mit_win_lo", fmt(secs.min() + 600)); add("s.mit_win_hi", fmt(secs.max() + 2400))

    # ------------------------------------------------------------------ S2 preprocessing
    for g, D in (("c", C), ("p", P), ("m", M)):
        add(f"s.flag_bounds_{g}", INT(num(D.N_Flag_Bounds).sum()))
        add(f"s.flag_median_{g}", INT(num(D.N_Flag_Median).sum()))
        add(f"s.nrr_med_{g}", INT(num(D.Num_RR).median()))
        add(f"s.maxrun_{g}", str(int(num(D.Artifact_Max_Consecutive).max())))
    assert (num(CP.Median_Window_Beats) == 11).all() and (CP.Artifact_Method.astype(str) == "delete").all()
    add("s.maxrun_all", str(int(num(pd.concat([C, P, M]).Artifact_Max_Consecutive).max())))
    r, p, _, _ = mw(C.Artifact_Percent, P.Artifact_Percent)
    add("s.art_r", R(r)); add("s.art_pe", PE(p))
    assert p > 0.05
    r, p, _, _ = mw(C.Artifact_Thresh_Over_RMSSD, P.Artifact_Thresh_Over_RMSSD)
    add("s.thr_c", U(num(C.Artifact_Thresh_Over_RMSSD).median(), 1))
    add("s.thr_p", U(num(P.Artifact_Thresh_Over_RMSSD).median(), 1))
    add("s.thr_r", R(r)); add("s.thr_pe", PE(p))
    # Artifact_Thresh_Over_RMSSD divides by the RMSSD of the uncleaned window, which the rejected intervals inflate
    # (patients' median 18.2 ms raw against 9.5 ms clean). The same threshold over the cleaned series' RMSSD:
    tc = num(C.Artifact_Thresh_ms) / num(C.HRV_RMSSD)
    tp = num(P.Artifact_Thresh_ms) / num(P.HRV_RMSSD)
    r, p, n1, n2 = mw(tc, tp)
    assert n1 == 30 and n2 == 30
    add("s.thrc_c", U(tc.median(), 1)); add("s.thrc_p", U(tp.median(), 1))
    add("s.thrc_r", R(r)); add("s.thrc_pe", PE(p))
    assert tp.median() > 1.8 * tc.median(), "text says roughly twice as permissive"
    # lattice
    sc, sp, sm = num(C.Quant_Step_Observed_s), num(P.Quant_Step_Observed_s), num(M.Quant_Step_Observed_s)
    assert np.allclose(sc, 0.001) and np.allclose(sp, 0.001) and np.allclose(sm, 1 / 128)
    lat = pd.concat([num(C.Quant_On_Lattice_Frac), num(P.Quant_On_Lattice_Frac), num(M.Quant_On_Lattice_Frac)])
    assert lat.min() == 1.0, "text says every successive difference lies on the grid"
    assert K.code_has("pipeline.py", "nk.ecg_process(lead, sampling_rate=sampling_rate)")
    # Methods S3 quotes the IRASA simulation recorded in compute_irasa's docstring
    assert K.code_has("pipeline.py", "(recovering β ≈ 0.67") and K.code_has("pipeline.py", "against a true 1.0 in simulation)")
    assert K.code_has("healthy.py", "lead_index    = 0")

    # ------------------------------------------------------------------ S3 spectral
    ns = num(CP.Welch_N_Segments)
    add("s.welch_min", str(int(ns.min()))); add("s.welch_max", str(int(ns.max())))
    r2c, r2p = num(C.PSD_Beta_R2), num(P.PSD_Beta_R2)
    add("s.r2_med_c", U(r2c.median(), 2)); add("s.r2_med_p", U(r2p.median(), 2))
    hs = code_const("pipeline.py", "IRASA_HSET")
    assert hs == "np.arange(1.05, 1.55, 0.05)" and code_const("pipeline.py", "IRASA_NPERSEG") == "1024"
    assert code_const("pipeline.py", "SPECPARAM_PEAK_BAND") == "(0.06, 0.14)"

    # ------------------------------------------------------------------ S5 entropy cross-check
    dse = (num(CP.SampEn) - num(CP.SampEn_NK)).abs().max()
    dae = (num(CP.ApEn) - num(CP.ApEn_NK)).abs().max()
    assert dse < 1e-10 and dae < 1e-10
    add("s.nk_maxdiff", "10⁻¹⁴")
    assert max(dse, dae) < 1e-14

    # ------------------------------------------------------------------ S6 recurrence
    ex = num(CP.RQA_Theiler_Excluded_Frac)
    add("s.theiler_excl_max", U(100 * ex.max(), 1))
    add("s.theiler_excl_med", U(100 * ex.median(), 1))
    assert (CP.RQA_Theiler_Source.astype(str) == "mutual_information").all()
    rec = num(CP.RQA_REC)
    assert (rec - 0.1).abs().max() < 1e-6
    deg_ok = true_count(CP.RNA_DegreeCheck_OK)
    assert deg_ok == 60
    ra = num(CP.RQA_REC_AllPairs)
    add("s.rec_all_lo", U(ra.min(), 4)); add("s.rec_all_hi", U(ra.max(), 4))
    # short-delay patients: check whether they explain the determinism direction
    dl = num(P.Embed_Delay_MI)
    low = dl <= 3
    r_, p_, _, _ = mw(P.Quant_Step_Over_RMSSD[~low], P.Quant_Step_Over_RMSSD[low])
    add("s.short_ratio_pe", PE(p_))
    r_, p_, n1, n2 = mw(C.RQA_DET, P.RQA_DET[~low])
    add("s.det_wo_short_r", R(r_)); add("s.det_wo_short_pe", PE(p_)); add("s.det_wo_short_n", str(n2))
    add("s.tw_sel_c", INT(num(C.RQA_Theiler_W_SelEmbed).median()))
    add("s.tw_sel_p", INT(num(P.RQA_Theiler_W_SelEmbed).median()))

    # ------------------------------------------------------------------ S8 surrogates
    for m in ("SampEn", "DFA_Alpha1", "MSE_Mean"):
        assert (num(CP[f"SURR_{m}_N_Valid"]) == 100).all(), m
        assert (num(M[f"SURR_{m}_N_Valid"]) == 100).all(), m

    # ------------------------------------------------------------------ S11 respiration
    keff = num(CP.Resp_K_Eff)
    add("s.keff_med", U(keff.median(), 1))
    thr = num(CP.Resp_Peak_Threshold)
    add("s.resp_thr_lo", U(thr.min(), 3)); add("s.resp_thr_hi", U(thr.max(), 3))
    assert code_const("pipeline.py", "WELCH_OVERLAP_EFFICIENCY") == "0.82"
    # Check of the 0.82 factor (computed for this supplement): white noise of the window's length at 4 Hz, analyzed
    # with the pipeline's Welch settings; the effective K is the gamma shape whose log10 SD matches the observed SD
    # of log10 power over the respiratory search band.
    from scipy.signal import welch
    from scipy.special import polygamma
    from scipy.optimize import brentq
    rng = np.random.default_rng(0)
    logs = np.array([np.log10(welch(rng.standard_normal(7200), fs=4, nperseg=512, noverlap=256, window="hann",
                                    detrend="constant")[1]) for _ in range(2000)])
    fgrid = np.fft.rfftfreq(512, 0.25)
    band = (fgrid >= 0.15) & (fgrid <= 0.40)
    sd_obs = logs[:, band].std(axis=0).mean()
    nseg = (7200 - 512) // 256 + 1
    k_sim = brentq(lambda k: np.sqrt(polygamma(1, k)) / np.log(10) - sd_obs, 1, 500) / nseg
    sd_082 = np.sqrt(polygamma(1, 0.82 * nseg)) / np.log(10)
    add("s.keff_sim", U(k_sim, 2))
    add("s.keff_inflate", INT(100 * (sd_082 / sd_obs - 1)))
    assert 0.93 < k_sim < 0.97 and 5 < 100 * (sd_082 / sd_obs - 1) < 11
    assert code_const("pipeline.py", "RESP_SEARCH_BAND") == "(0.15,  0.40)" and code_const("pipeline.py", "RESP_LF_BAND") == "(0.10,  0.15)"

    # ------------------------------------------------------------------ Results S1 respiration
    lr = pd.read_csv(K.csvp(K.CMP, "Length_Respiration_Checks.csv")).set_index("Variable")
    fc, fp = true_count(C.Resp_Peak_Found), true_count(P.Resp_Peak_Found)
    pf = fisher(fc, 30, fp, 30)
    assert abs(pf - lr.loc["Resp_Peak_Found (detection rate)", "p_raw"]) < 1e-6
    add("s.resp_found_c", str(fc)); add("s.resp_found_p", str(fp)); add("s.resp_found_pe", PE(pf))
    r, p, n1, n2 = mw(C.Resp_Peak_Freq, P.Resp_Peak_Freq)
    assert (n1, n2) == (fc, fp)
    add("s.resp_f_c", U(num(C.Resp_Peak_Freq).median())); add("s.resp_f_p", U(num(P.Resp_Peak_Freq).median()))
    add("s.resp_f_r", R(r)); add("s.resp_f_pe", PE(p))
    add("s.resp_bpm_c", INT(60 * num(C.Resp_Peak_Freq).median())); add("s.resp_bpm_p", INT(60 * num(P.Resp_Peak_Freq).median()))
    mp = num(P.HRV_RMSSD).mean()
    add("s.rmssd_mean_p", U(mp, 1))
    assert abs(0.001 / (mp / 1000) - float(code_const("pipeline.py", "QUANT_TARGET_RATIO"))) < 0.0005
    for D in (C, P):
        assert num(D.RQA_DET_SelEmbed).median() > num(D.RQA_DET).median()
        assert num(D.RQA_LAM_SelEmbed).median() > num(D.RQA_LAM).median()
    assert abs(r - lr.loc["Resp_Peak_Freq (detected subjects only)", "r_effect"]) < 1e-5
    bc, bp = true_count(C.Resp_Peak_Below_HF), true_count(P.Resp_Peak_Below_HF)
    add("s.resp_low_c", str(bc)); add("s.resp_low_p", str(bp)); add("s.resp_low_pe", PE(fisher(bc, 30, bp, 30)))
    amc, amp = true_count(C.Resp_Peak_Ambiguous_Mayer), true_count(P.Resp_Peak_Ambiguous_Mayer)
    add("s.resp_amb_c", str(amc)); add("s.resp_amb_p", str(amp))
    noneither_c = int(((~C.Resp_Peak_Found.astype(str).str.lower().eq("true")) & (~C.Resp_Peak_Below_HF.astype(str).str.lower().eq("true"))).sum())
    noneither_p = int(((~P.Resp_Peak_Found.astype(str).str.lower().eq("true")) & (~P.Resp_Peak_Below_HF.astype(str).str.lower().eq("true"))).sum())
    add("s.resp_none_c", str(noneither_c)); add("s.resp_none_p", str(noneither_p))
    # Slopes restricted to the two lowest ranges among subjects with a sub-HF peak
    # (not computed: the low-range slopes already exclude 0.07 Hz upward for every subject)

    # ------------------------------------------------------------------ Results S2 covariance
    def fz(r1, r2, n1=30, n2=30):
        z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        return z, 2 * norm.sf(abs(z))
    for col, key in (("HRV_RMSSD", "rmssd"), ("Total_Power_ms2", "tp"), ("HF_Power_ms2", "hf"), ("LF_Power_ms2", "lf")):
        rc = spearmanr(num(C.PSD_Beta), num(C[col]))[0]
        rp = spearmanr(num(P.PSD_Beta), num(P[col]))[0]
        z, pz = fz(rc, rp)
        add(f"s.cov_{key}_c", R(rc)); add(f"s.cov_{key}_p", R(rp))
        add(f"s.cov_{key}_z", U(abs(z), 2)); add(f"s.cov_{key}_pz", P3(pz))
    hp = pd.read_csv(K.csvp(K.CMP, "Metric_Intercorrelation_HighPairs.csv"))
    add("s.hp_n", str(len(hp)))
    add("s.hp_both", str(int((hp.Note == "|r|>0.7 in both").sum())))
    add("s.hp_psych", str(int((hp.Note == "|r|>0.7 in psych").sum())))
    add("s.hp_healthy", str(int((hp.Note == "|r|>0.7 in healthy").sum())))
    vag = ["RMSSD (ms)", "pNN50 (%)", "HF Power (ms²)", "Total Power (ms²)"]
    h = pd.read_csv(K.csvp(K.CMP, "Metric_Intercorrelation_Healthy.csv"), index_col=0).loc[vag, vag]
    p_ = pd.read_csv(K.csvp(K.CMP, "Metric_Intercorrelation_Psychiatric.csv"), index_col=0).loc[vag, vag]
    off = ~np.eye(4, dtype=bool)
    add("s.vag_c_lo", U(h.values[off].min(), 2)); add("s.vag_c_hi", U(h.values[off].max(), 2))
    add("s.vag_p_lo", U(p_.values[off].min(), 2)); add("s.vag_p_hi", U(p_.values[off].max(), 2))
    sl = ["PSD β", "DFA α₂ (16–64)", "DFA α (full)"]
    h3 = pd.read_csv(K.csvp(K.CMP, "Metric_Intercorrelation_Healthy.csv"), index_col=0).loc[sl, sl]
    p3 = pd.read_csv(K.csvp(K.CMP, "Metric_Intercorrelation_Psychiatric.csv"), index_col=0).loc[sl, sl]
    off3 = ~np.eye(3, dtype=bool)
    add("s.sc_c_lo", U(h3.values[off3].min(), 2)); add("s.sc_c_hi", U(h3.values[off3].max(), 2))
    add("s.sc_p_lo", U(p3.values[off3].min(), 2)); add("s.sc_p_hi", U(p3.values[off3].max(), 2))
    assert p3.values[off3].min() > h3.values[off3].max() - 0.2

    # ------------------------------------------------------------------ Results S3 activity, movement-free retention
    st = pd.read_csv(K.csvp(K.OUT, "stationary", "Stationary_Longest_Runs.csv"))
    add("s.run_lo", INT(st.N_Longest_Contiguous.min())); add("s.run_hi", INT(st.N_Longest_Contiguous.max()))
    b = pd.read_csv(K.csvp(K.OUT, "stationary", "Stationary_Reanalysis_beats.csv"))
    excl = sorted(set(st.Record_ID) - set(b.Record_ID))
    assert len(excl) == 9
    short = st.set_index("Record_ID").loc[excl, "N_Longest_Contiguous"]
    assert (short < 600).all(), "every excluded subject's run is shorter than 600 beats"
    add("s.mf_excl_c", str(sum(e.startswith("control") for e in excl)))
    add("s.mf_excl_p", str(sum(e.startswith("treatment") for e in excl)))

    # ------------------------------------------------------------------ Methods S9 duration version
    du = pd.read_csv(K.csvp(K.OUT, "stationary", "Stationary_Reanalysis_duration.csv"))
    bpm_needed = 500 / 370 * 60
    add("s.dur_bpm", INT(bpm_needed))
    assert (60.0 / (du.Duration_s / du.N_Beats) >= bpm_needed - 2).all()

    # ------------------------------------------------------------------ Methods S13 classifier: feature pool
    nc = open(K.csvp(K.PROJECT, "nested_classifier.py"), encoding="utf-8").read()
    assert 'EXCLUDE_METRICS = ["RQA_REC",' in nc and "SELECT_ALPHA   = 0.05" in nc and "MIN_FEATURES   = 3" in nc
    assert "N_SPLITS       = 5" in nc and "N_REPEATS      = 100" in nc

    # ------------------------------------------------------------------ Methods S14 validation seeds
    pl = open(K.csvp(K.PROJECT, "pipeline.py"), encoding="utf-8").read()
    assert "rng = np.random.default_rng(seed + 977 * rep + int(H * 100))" in pl
