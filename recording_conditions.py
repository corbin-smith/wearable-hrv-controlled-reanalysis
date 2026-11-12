"""recording_conditions.py: what the participants were doing, and whether the results depend on it."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.stats import spearmanr

import pipeline as P
import accelerometer_mobility as ACC
import comparison as CMP

OUT_DIR = os.path.join("outputs", "recording_conditions")
STATIONARY_CSV = os.path.join("outputs", "stationary", "Stationary_Reanalysis_beats.csv")
PROFILE_CSV = os.path.join(OUT_DIR, "Motion_Profile_ByMinute.csv")

N_MINUTES = 40
EARLY_SECONDS = 300.0
QUIET_MG = 40.0
LONG_START_S, LONG_END_S = 600.0, 4200.0
LONG_NPERSEG = 2048
LONG_DETREND = "linear"

EARLY_MEASURES = ["HRV_MeanHR", "HRV_RMSSD", "HRV_SDNN", "LF_Power_ms2", "HF_Power_ms2",
                  "LF_HF_Ratio", "SampEn", "ApEn", "PermEn", "RQA_DET", "RQA_LAM"]
STATIONARY_MEASURES = ["HRV_MeanHR", "HRV_RMSSD", "LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2",
                       "Total_Power_ms2", "VLF_LF_Ratio", "LF_HF_Ratio", "PSD_Beta", "PSD_Beta_NoMayer",
                       "DFA_Alpha2", "MSE_Mean", "SampEn", "ApEn", "PermEn", "RQA_DET", "RQA_LAM"]
LONG_MEASURES = ["VLF_Power_ms2", "LF_Power_ms2", "HF_Power_ms2", "VLF_LF_Ratio",
                 "PSD_Beta", "PSD_Beta_NoMayer"]


def record_ids():
    df_h, df_p = CMP.load_cohorts(CMP.HRVACC_HEALTHY_CSV, CMP.PSYCH_CSV)
    if len(df_h) != 30 or len(df_p) != 30:
        sys.exit(f"expected 30 + 30 recordings, found {len(df_h)} + {len(df_p)}")
    return [str(x) for x in df_h.index] + [str(x) for x in df_p.index]


def group_of(rid):
    return 1 if str(rid).startswith("treatment") else 0


def rr_with_clock(rid):
    path = ACC._rr_path_for(rid)
    if path is None or not os.path.exists(path):
        sys.exit(f"missing R-R file for {rid}: see DATA_SETUP.md")
    df = pd.read_csv(path, sep=";")
    rr_col = next(c for c in df.columns if "rr" in c.lower() or "interval" in c.lower())
    ts_col = next(c for c in df.columns if "timestamp" in c.lower())
    rr = pd.to_numeric(df[rr_col], errors="coerce")
    ok = rr.notna()
    rr = rr[ok].to_numpy(dtype=float) / 1000.0
    return rr, np.cumsum(rr), ACC._seconds_of_day(df[ts_col][ok])


def clean_window(rid, start_s, length_s):
    rr_raw, t_raw, _ = P.load_rr_window(ACC._rr_path_for(rid), skip_seconds=start_s,
                                        analysis_seconds=length_s, verbose=False)
    rr, artifact_pct, _, _ = P.reject_ectopic_beats(rr_raw, verbose=False)
    rr = np.asarray(rr, dtype=float)
    t = np.concatenate([[t_raw[0]], t_raw[0] + np.cumsum(rr[:-1])])
    return rr, t, float(artifact_pct)


def compare(table, measures, covariate_sets):
    g = table["Group"].to_numpy(dtype=float)
    rows = []
    for m in measures:
        y = pd.to_numeric(table[m], errors="coerce").to_numpy(dtype=float)
        r_rb, p_rb, n_c, n_p = CMP._mw(y[g == 0], y[g == 1])
        ok = np.isfinite(y)
        rho, p_rho = spearmanr(y[ok], g[ok])
        row = {"Metric": m, "N_Controls": n_c, "N_Patients": n_p,
               "Control_Median": float(np.nanmedian(y[g == 0])),
               "Patient_Median": float(np.nanmedian(y[g == 1])),
               "r_rank_biserial": r_rb, "p_rank_biserial": p_rb,
               "r_partial_unadjusted": float(rho), "p_partial_unadjusted": float(p_rho)}
        for label, cols in covariate_sets:
            if m in cols:
                row[f"r_partial_{label}"] = np.nan
                row[f"p_partial_{label}"] = np.nan
                row[f"n_{label}"] = 0
                continue
            covs = [pd.to_numeric(table[c], errors="coerce").to_numpy(dtype=float) for c in cols]
            n_ok = int((ok & np.all([np.isfinite(c) for c in covs], axis=0)).sum())
            r, p = CMP.partial_spearman(y, g, covs)
            row[f"r_partial_{label}"] = r
            row[f"p_partial_{label}"] = p
            row[f"n_{label}"] = n_ok
        rows.append(row)
    return pd.DataFrame(rows)


def _motion(acc, t0, t1):
    w = acc[(acc["t"] >= t0) & (acc["t"] <= t1)]
    if len(w) < ACC.MIN_ACC_SAMPLES_IN_WINDOW:
        return None, None
    tt = w["t"].to_numpy()
    dt = np.median(np.diff(tt))
    fs = 1.0 / dt if dt > 0 else 100.0
    sigma = max(ACC.GRAVITY_TIME_CONSTANT_S * fs, 1.0)
    sq = 0.0
    for axis in ("X", "Y", "Z"):
        v = w[axis].to_numpy()
        sq = sq + (v - gaussian_filter1d(v, sigma, mode="nearest")) ** 2
    return tt, np.sqrt(sq)


def motion_profile(ids):
    rows = []
    for rid in ids:
        acc = ACC._load_acc(rid)
        if acc is None:
            return None
        _, cum, clock = rr_with_clock(rid)

        def clock_at(seconds):
            return float(clock[min(int(np.searchsorted(cum, seconds)), len(clock) - 1)])

        start = float(clock[0])
        tt, mo = _motion(acc, start, clock_at(60.0 * N_MINUTES))
        row = {"Record_ID": rid, "Group": group_of(rid),
               "acc_lag_s": float(acc["t"].iloc[0] - start)}
        for i in range(N_MINUTES):
            a0 = start if i == 0 else clock_at(60.0 * i)
            a1 = clock_at(60.0 * (i + 1))
            seg = mo[(tt >= a0) & (tt < a1)] if mo is not None else np.array([])
            row[f"min_{i:02d}"] = float(seg.mean()) if len(seg) > 100 else np.nan
        for name, (b0, b1) in (("early_0_300_mg", (start, clock_at(EARLY_SECONDS))),
                               ("early_60_300_mg", (clock_at(60.0), clock_at(EARLY_SECONDS))),
                               ("window_600_2400_mg", (clock_at(float(P.SKIP_SECONDS)),
                                                       clock_at(float(P.SKIP_SECONDS + P.ANALYSIS_SECONDS))))):
            _, m = _motion(acc, b0, b1)
            row[name] = float(m.mean()) if m is not None else np.nan
        rows.append(row)
        print(f"    {rid}: first 5 min {row['early_0_300_mg']:.0f} mg, "
              f"analysis window {row['window_600_2400_mg']:.0f} mg", flush=True)
    return pd.DataFrame(rows)


def motion_summary(prof):
    mins = [f"min_{i:02d}" for i in range(N_MINUTES)]
    by_min = pd.DataFrame({
        "Minute": range(N_MINUTES),
        "Control_Median_mg": prof.loc[prof.Group == 0, mins].median().to_numpy(),
        "Patient_Median_mg": prof.loc[prof.Group == 1, mins].median().to_numpy(),
    })
    first5 = prof[mins[:5]].mean(axis=1)
    a = prof[mins].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        blocks = np.array([[np.nanmean(a[i, j:j + 5]) if np.isfinite(a[i, j:j + 5]).all() else np.nan
                            for j in range(N_MINUTES - 4)] for i in range(len(prof))])
    rows = []
    for label, g in (("controls", 0), ("patients", 1)):
        s = prof.Group == g
        rows.append({
            "Group": label, "N": int(s.sum()),
            "Early_0_300_Median_mg": float(prof.loc[s, "early_0_300_mg"].median()),
            "Window_600_2400_Median_mg": float(prof.loc[s, "window_600_2400_mg"].median()),
            "N_Early_Below_Window": int((prof.loc[s, "early_0_300_mg"] < prof.loc[s, "window_600_2400_mg"]).sum()),
            f"N_First5_Below_{int(QUIET_MG)}mg": int((first5[s] < QUIET_MG).sum()),
            "Quietest_5min_Median_mg": float(np.nanmedian(np.nanmin(blocks[s.to_numpy()], axis=1))),
            "Quietest_5min_Start_Median_min": float(np.nanmedian(np.nanargmin(blocks[s.to_numpy()], axis=1))),
        })
    summ = pd.DataFrame(rows)
    r, p, _, _ = CMP._mw(prof.loc[prof.Group == 0, "early_0_300_mg"],
                         prof.loc[prof.Group == 1, "early_0_300_mg"])
    summ["Early_Group_r"] = r
    summ["Early_Group_p"] = p
    return by_min, summ


def early_window(ids, prof):
    rows = []
    for rid in ids:
        rr, t, art = clean_window(rid, 0.0, EARLY_SECONDS)
        rr_ms = rr * 1000.0
        fd, _, _ = P.compute_frequency_domain(rr, t)
        row = {"Record_ID": rid, "Group": group_of(rid), "N_Beats": len(rr), "Artifact_Percent": art,
               "HRV_MeanHR": float(60000.0 / np.mean(rr_ms)),
               "HRV_RMSSD": float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))),
               "HRV_SDNN": float(np.std(rr_ms, ddof=1)),
               "LF_Power_ms2": fd.get("LF_Power_ms2", np.nan),
               "HF_Power_ms2": fd.get("HF_Power_ms2", np.nan),
               "LF_HF_Ratio": fd.get("LF_HF_Ratio", np.nan),
               "SampEn": P.sample_entropy(rr), "ApEn": P.approximate_entropy(rr)}
        try:
            pe, _ = nk.entropy_permutation(rr, dimension=3, delay=1)
            row["PermEn"] = float(pe)
        except Exception:
            row["PermEn"] = np.nan
        theiler, _ = P.resolve_theiler(rr, P.RQA_DIM, P.RQA_DELAY)
        rq = P.rqa_metrics(rr, theiler=theiler)
        row["RQA_DET"] = rq.get("RQA_DET", np.nan)
        row["RQA_LAM"] = rq.get("RQA_LAM", np.nan)
        row["Theiler_W"] = int(theiler)
        rows.append(row)
    tab = pd.DataFrame(rows)
    if prof is not None:
        tab = tab.merge(prof[["Record_ID", "early_0_300_mg"]], on="Record_ID", how="left")
        sets = [("HR", ["HRV_MeanHR"]), ("HR_motion", ["HRV_MeanHR", "early_0_300_mg"])]
    else:
        sets = [("HR", ["HRV_MeanHR"])]
    return tab, compare(tab, EARLY_MEASURES, sets)


def movement_free():
    if not os.path.exists(STATIONARY_CSV):
        print(f"    {STATIONARY_CSV} not found: run stationary_reanalysis.py first. Part 3 skipped.")
        return None
    st = pd.read_csv(STATIONARY_CSV)
    st["Group"] = [group_of(r) for r in st["Record_ID"]]
    have = [m for m in STATIONARY_MEASURES if m in st.columns]
    out = compare(st, have, [("HR", ["HRV_MeanHR"])])
    out["Median_Duration_s"] = float(st["Duration_s"].median())
    out["Segment"] = "first 600 beats of the longest segment outside detected walking bouts"
    return out


def _spectral_long(rr, t):
    tu = np.arange(t[0], t[-1], 1.0 / P.INTERP_FS)
    x = interp1d(t, rr, kind="cubic", bounds_error=False, fill_value="extrapolate")(tu)
    nper = int(min(len(x), LONG_NPERSEG))
    f, pw = welch(x, fs=P.INTERP_FS, nperseg=nper, noverlap=int(nper * P.WELCH_OVERLAP_FRAC),
                  window=P.WELCH_WINDOW, detrend=LONG_DETREND)
    pw = pw * 1e6

    def band(b):
        i = (f >= b[0]) & (f <= b[1])
        return float(np.trapezoid(pw[i], f[i])) if i.sum() > 1 else np.nan

    vlf, lf, hf = band(P.VLF_BAND), band(P.LF_BAND), band(P.HF_BAND)
    beta = P._fit_beta(f, pw, P.BETA_FIT_BAND)[0]
    beta_nm = P._fit_beta(f, pw, P.BETA_FIT_BAND, exclude_band=P.MAYER_BAND, log_bin=False)[0]
    n_vlf = int(((f >= P.VLF_BAND[0]) & (f <= P.VLF_BAND[1]) & (f > 0)).sum())
    n_seg = max(1, int((len(x) - nper // 2) // (nper - nper // 2)))
    return {"VLF_Power_ms2": vlf, "LF_Power_ms2": lf, "HF_Power_ms2": hf,
            "VLF_LF_Ratio": vlf / lf if lf and lf > 0 else np.nan,
            "PSD_Beta": beta, "PSD_Beta_NoMayer": beta_nm,
            "VLF_N_Bins": n_vlf, "Welch_N_Segments": n_seg}


def long_window(ids):
    rows = []
    for rid in ids:
        rr, t, art = clean_window(rid, LONG_START_S, LONG_END_S - LONG_START_S)
        fd, _, _ = P.compute_frequency_domain(rr, t)
        ls = _spectral_long(rr, t)
        row = {"Record_ID": rid, "Group": group_of(rid), "N_Beats": len(rr),
               "Duration_s": float(np.sum(rr)), "Artifact_Percent": art,
               "HRV_MeanHR": float(60.0 / np.mean(rr))}
        for m in LONG_MEASURES:
            row[f"{m}__standard"] = fd.get(m, np.nan)
            row[f"{m}__long_segments"] = ls[m]
        row["VLF_N_Bins__standard"] = fd.get("VLF_N_Bins", np.nan)
        row["VLF_N_Bins__long_segments"] = ls["VLF_N_Bins"]
        row["Welch_N_Segments__long_segments"] = ls["Welch_N_Segments"]
        rows.append(row)
    tab = pd.DataFrame(rows)
    parts = []
    for est in ("standard", "long_segments"):
        sub = tab[["Record_ID", "Group", "HRV_MeanHR"]].copy()
        for m in LONG_MEASURES:
            sub[m] = tab[f"{m}__{est}"]
        c = compare(sub, LONG_MEASURES, [("HR", ["HRV_MeanHR"])])
        c.insert(1, "Estimator", "128 s segments, mean removed (pipeline settings)" if est == "standard"
                 else "512 s segments, linear detrend")
        parts.append(c)
    return tab, pd.concat(parts, ignore_index=True)


def _show(title, tab, cols):
    print(f"  {title}")
    for _, r in tab.iterrows():
        cells = "  ".join(f"{c.replace('r_partial_', '| ').replace('r_rank_biserial', 'r')}"
                          f" {r[c]:+.3f}" for c in cols if c in tab.columns and np.isfinite(r[c]))
        extra = f"  [{r['Estimator'][:5]}]" if "Estimator" in tab.columns else ""
        print(f"    {r['Metric']:18s}{extra} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-motion", action="store_true",
                    help="skip the accelerometer profile and reuse the saved table")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = record_ids()

    print("[1] Movement by minute...")
    prof = None
    if not a.no_motion:
        prof = motion_profile(ids)
        if prof is None:
            print("    Accelerometer files not found; reusing the saved table if there is one.")
        else:
            prof.to_csv(PROFILE_CSV, index=False, float_format="%.6f")
    if prof is None and os.path.exists(PROFILE_CSV):
        prof = pd.read_csv(PROFILE_CSV)
    if prof is not None:
        by_min, summ = motion_summary(prof)
        by_min.to_csv(os.path.join(OUT_DIR, "Motion_Profile_GroupMedians.csv"), index=False, float_format="%.4f")
        summ.to_csv(os.path.join(OUT_DIR, "Motion_Profile_Summary.csv"), index=False, float_format="%.6g")
        for _, r in summ.iterrows():
            print(f"    {r['Group']}: first 5 min median {r['Early_0_300_Median_mg']:.0f} mg, analysis window "
                  f"{r['Window_600_2400_Median_mg']:.0f} mg; first 5 min below {int(QUIET_MG)} mg in "
                  f"{int(r[f'N_First5_Below_{int(QUIET_MG)}mg'])} of {int(r['N'])}")
    else:
        print("    No motion table: part 2 runs without the motion covariate.")

    print("[2] The first 5 minutes as a second window...")
    early_tab, early_cmp = early_window(ids, prof)
    early_tab.to_csv(os.path.join(OUT_DIR, "Early_Window_PerSubject.csv"), index=False, float_format="%.8g")
    early_cmp.to_csv(os.path.join(OUT_DIR, "Early_Window_Comparison.csv"), index=False, float_format="%.8g")
    _show("rank-biserial r, and partial correlations with group", early_cmp,
          ["r_rank_biserial", "r_partial_unadjusted", "r_partial_HR", "r_partial_HR_motion"])

    print("[3] Movement-free segments with heart rate held constant...")
    mf = movement_free()
    if mf is not None:
        mf.to_csv(os.path.join(OUT_DIR, "Movement_Free_HR_Adjusted.csv"), index=False, float_format="%.8g")
        _show("", mf, ["r_rank_biserial", "r_partial_unadjusted", "r_partial_HR"])

    print("[4] The 60-minute window...")
    long_tab, long_cmp = long_window(ids)
    long_tab.to_csv(os.path.join(OUT_DIR, "Long_Window_Spectral_PerSubject.csv"), index=False, float_format="%.8g")
    long_cmp.to_csv(os.path.join(OUT_DIR, "Long_Window_Spectral.csv"), index=False, float_format="%.8g")
    _show("", long_cmp, ["r_rank_biserial", "r_partial_unadjusted", "r_partial_HR"])
    print(f"  Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
