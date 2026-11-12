"""accelerometer_mobility.py: movement control from the Polar H10 accelerometer."""

import os
import re
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, mannwhitneyu, rankdata, t as t_dist


def _first_existing(candidates, default=None):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return default if default is not None else candidates[0]

CONTROL_RR_DIR = _first_existing(["Control ECGs"], default="Control ECGs")
PSYCH_RR_DIR   = _first_existing(["Psychiatric ECGs"], default="Psychiatric ECGs")

ACC_DIR = _first_existing(
    ["Accelerometer ECGs", "accelerometer_anonymized_data", "Accelerometer",
     "ACC ECGs"],
    default="Accelerometer ECGs",
)

HEALTHY_CSV = _first_existing([
    os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "ecg_results_HealthyACC.csv",
])
PSYCH_CSV = _first_existing([
    os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv"),
    "ecg_results_Psychiatric.csv",
])

OUTPUT_DIR = os.path.join("outputs", "accelerometer")

from pipeline import (SKIP_SECONDS as RR_SKIP_SECONDS,
                      ANALYSIS_SECONDS as RR_ANALYSIS_SECONDS)

GRAVITY_TIME_CONSTANT_S = 10.0

MIN_ACC_SAMPLES_IN_WINDOW = 500
MIN_WINDOW_COVERAGE       = 0.50

SENSITIVITY_MIN_COVERAGE  = 0.80

FORCE_RECOMPUTE = False

QC_COLUMN = "QC_Overall_Pass"


def _seconds_of_day(timestamps):
    t = pd.to_datetime(timestamps, format="%H:%M:%S.%f", errors="coerce")
    if t.isna().any():
        t = t.fillna(pd.to_datetime(timestamps, errors="coerce"))
    secs = (t.dt.hour * 3600 + t.dt.minute * 60
            + t.dt.second + t.dt.microsecond / 1e6).to_numpy(dtype=float)
    for w in np.where(np.diff(secs) < -43200.0)[0]:
        secs[w + 1:] += 86400.0
    return secs


def _rr_path_for(record_id):
    rid = str(record_id)
    if rid.startswith("treatment"):
        return os.path.join(PSYCH_RR_DIR, f"{rid}.csv")
    if rid.startswith("control"):
        return os.path.join(CONTROL_RR_DIR, f"{rid}.csv")
    for d in (CONTROL_RR_DIR, PSYCH_RR_DIR):
        cand = os.path.join(d, f"{rid}.csv")
        if os.path.exists(cand):
            return cand
    return None


def _acc_path_for(record_id):
    cand = os.path.join(ACC_DIR, f"{record_id}.csv")
    return cand if os.path.exists(cand) else None


def _rr_window_bounds(record_id):
    path = _rr_path_for(record_id)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if rr_col is None or ts_col is None:
        return None

    rr_ms = pd.to_numeric(df[rr_col], errors="coerce")
    good = rr_ms.notna()
    rr_ms = rr_ms[good].to_numpy(dtype=float)
    clock = _seconds_of_day(df[ts_col][good])
    if len(rr_ms) < 50:
        return None

    cum = np.cumsum(rr_ms / 1000.0)
    i_start = int(np.searchsorted(cum, RR_SKIP_SECONDS))
    i_end   = int(np.searchsorted(cum, RR_SKIP_SECONDS + RR_ANALYSIS_SECONDS))
    i_start = min(i_start, len(clock) - 1)
    i_end   = min(i_end,   len(clock) - 1)
    if i_end <= i_start:
        return None
    return float(clock[i_start]), float(clock[i_end])


def _load_acc(record_id):
    path = _acc_path_for(record_id)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    xcol = next((c for c in df.columns if c.strip().upper().startswith("X")), None)
    ycol = next((c for c in df.columns if c.strip().upper().startswith("Y")), None)
    zcol = next((c for c in df.columns if c.strip().upper().startswith("Z")), None)
    if not all([ts_col, xcol, ycol, zcol]):
        return None
    out = pd.DataFrame({
        "t": _seconds_of_day(df[ts_col]),
        "X": pd.to_numeric(df[xcol], errors="coerce"),
        "Y": pd.to_numeric(df[ycol], errors="coerce"),
        "Z": pd.to_numeric(df[zcol], errors="coerce"),
    }).dropna()
    return out if len(out) else None


def compute_mobility(acc, t_start, t_end, gravity_tc_s=GRAVITY_TIME_CONSTANT_S):
    w = acc[(acc["t"] >= t_start) & (acc["t"] <= t_end)]
    n = len(w)
    span = max(t_end - t_start, 1e-9)
    if n < MIN_ACC_SAMPLES_IN_WINDOW:
        return np.nan, n, 0.0

    tt = w["t"].to_numpy()
    dt = np.median(np.diff(tt))
    fs = 1.0 / dt if dt > 0 else 100.0
    coverage = min(n / (span * fs), 1.0)
    if coverage < MIN_WINDOW_COVERAGE:
        return np.nan, n, float(coverage)

    sigma = max(gravity_tc_s * fs, 1.0)
    x, y, z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    xe = gaussian_filter1d(x, sigma, mode="nearest")
    ye = gaussian_filter1d(y, sigma, mode="nearest")
    ze = gaussian_filter1d(z, sigma, mode="nearest")
    motion = np.sqrt((x - xe) ** 2 + (y - ye) ** 2 + (z - ze) ** 2)
    return float(np.mean(motion)), n, float(coverage)


def build_mobility_table(record_ids):
    rows, warnings = [], []
    for rid in record_ids:
        bounds = _rr_window_bounds(rid)
        acc = _load_acc(rid)
        if bounds is None:
            warnings.append(f"{rid}: no usable R-R window (skipped)")
            rows.append({"Record_ID": rid, "mobility_mg": np.nan,
                         "n_acc_samples": 0, "coverage": 0.0})
            continue
        if acc is None:
            warnings.append(f"{rid}: accelerometer file missing/unreadable")
            rows.append({"Record_ID": rid, "mobility_mg": np.nan,
                         "n_acc_samples": 0, "coverage": 0.0})
            continue
        t0, t1 = bounds
        mob, n, cov = compute_mobility(acc, t0, t1)
        if np.isnan(mob):
            warnings.append(f"{rid}: insufficient ACC coverage in window "
                            f"({cov:.0%})")
        rows.append({"Record_ID": rid, "mobility_mg": mob,
                     "n_acc_samples": n, "coverage": round(cov, 3)})
    return pd.DataFrame(rows).set_index("Record_ID"), warnings


def rank_biserial(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 2.0 * U / (len(a) * len(b)) - 1.0
    return float(r), float(p), U


def partial_spearman(df, target_col, outcome_col, covar_cols):
    t = rankdata(df[target_col].to_numpy(dtype=float))
    o = rankdata(df[outcome_col].to_numpy(dtype=float))
    k = len(covar_cols) if covar_cols else 0
    if k:
        Z = np.column_stack([np.ones(len(t))] +
                            [rankdata(df[c].to_numpy(dtype=float)) for c in covar_cols])

        def ols_resid(y):
            coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            return y - Z @ coef

        t, o = ols_resid(t), ols_resid(o)
    if np.std(t) == 0 or np.std(o) == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(t, o)[0, 1])
    dof = len(t) - 2 - k
    if dof <= 0 or abs(r) >= 1.0:
        return r, float("nan")
    p = float(2.0 * t_dist.sf(abs(r * np.sqrt(dof / (1.0 - r ** 2))), dof))
    return r, p

LOG_COVARIATES = ("HF_Power_ms2", "HRV_RMSSD")


def partial_pearson_log(df, target_col, outcome_col, covar_cols):
    t = df[target_col].to_numpy(dtype=float)
    if np.all(t > 0):
        t = np.log(t)
    o = df[outcome_col].to_numpy(dtype=float)
    k = len(covar_cols) if covar_cols else 0
    if k:
        cols = []
        for c in covar_cols:
            v = df[c].to_numpy(dtype=float)
            cols.append(np.log(v) if (c in LOG_COVARIATES and np.all(v > 0)) else v)
        Z = np.column_stack([np.ones(len(t))] + cols)

        def ols_resid(y):
            coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            return y - Z @ coef

        t, o = ols_resid(t), ols_resid(o)
    if np.std(t) == 0 or np.std(o) == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(t, o)[0, 1])
    dof = len(t) - 2 - k
    if dof <= 0 or abs(r) >= 1.0:
        return r, float("nan")
    return r, float(2.0 * t_dist.sf(abs(r * np.sqrt(dof / (1.0 - r ** 2))), dof))


def _load_results():
    frames = []
    for path, group in [(HEALTHY_CSV, 0), (PSYCH_CSV, 1)]:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                f"Result CSV not found: {path}. Edit HEALTHY_CSV / PSYCH_CSV.")
        df = pd.read_csv(path)
        idx = "Record_ID" if "Record_ID" in df.columns else df.columns[0]
        df = df.set_index(idx)
        df["Group"] = group
        frames.append(df)
    return pd.concat(frames)


def run_activity_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = _load_results()
    if QC_COLUMN in results.columns:
        results = results[results[QC_COLUMN] == True]          # noqa: E712

    print(f"  Loaded {len(results)} QC-passing subjects "
          f"({int((results['Group'] == 0).sum())} control, "
          f"{int((results['Group'] == 1).sum())} patient).")

    mob_path = os.path.join(OUTPUT_DIR, "mobility_coefficients.csv")
    if os.path.exists(mob_path) and not FORCE_RECOMPUTE:
        print("  Reusing cached mobility_coefficients.csv "
              "(set FORCE_RECOMPUTE=True or delete the file to recompute).")
        mob = pd.read_csv(mob_path).set_index("Record_ID")
    else:
        mob, warnings = build_mobility_table(list(results.index))
        mob.to_csv(mob_path)
        for w in warnings:
            print(f"    [!] {w}")

    overlap = [c for c in mob.columns if c in results.columns]
    if overlap:
        print(f"  Note: results CSV already carries {overlap}; using the values "
              f"from mobility_coefficients.csv, which is authoritative.")
        results = results.drop(columns=overlap)

    df = results.join(mob, how="left")
    usable = df[df["mobility_mg"].notna()].copy()
    print(f"  Mobility computed for {len(usable)} / {len(df)} subjects.")

    c = usable.loc[usable["Group"] == 0, "mobility_mg"]
    p = usable.loc[usable["Group"] == 1, "mobility_mg"]
    r_mob, p_mob, _ = rank_biserial(c.values, p.values)
    grp_rows = [{
        "Comparison": "Mobility: control vs patient",
        "Control_median": round(float(c.median()), 2),
        "Patient_median": round(float(p.median()), 2),
        "Control_mean":   round(float(c.mean()), 2),
        "Patient_mean":   round(float(p.mean()), 2),
        "rank_biserial_r": round(r_mob, 3),
        "p_value":        round(p_mob, 5),
        "note": "r>0: controls more mobile; r<0: patients more mobile",
    }]
    pd.DataFrame(grp_rows).to_csv(
        os.path.join(OUTPUT_DIR, "Mobility_Group_Comparison.csv"), index=False)

    need = ["PSD_Beta", "Group", "mobility_mg", "HF_Power_ms2", "HRV_MeanHR"]
    missing = [c for c in need if c not in usable.columns]
    cov = usable.dropna(
        subset=[c for c in need if c in usable.columns and c != "Group"]).copy()

    COVARIATE_TARGETS = ["PSD_Beta", "PSD_Beta_NoMayer", "VLF_LF_Ratio"]

    def covariate_table(frame):
        rows = []

        def add(target, label, covars):
            if target not in frame.columns or any(cc not in frame.columns
                                                  for cc in covars):
                rows.append({"Analysis": label, "partial_r": np.nan, "p": np.nan,
                             "n": len(frame), "note": "column missing"})
                return
            sub = frame.dropna(subset=[target])
            r, pv = partial_spearman(sub, target, "Group", covars)
            r_lin, p_lin = partial_pearson_log(sub, target, "Group", covars)
            rows.append({"Analysis": label, "partial_r": round(r, 3),
                         "p": round(pv, 5), "n": len(sub),
                         "partial_r_linear_log": round(r_lin, 3),
                         "p_linear_log": round(p_lin, 5), "note": ""})

        for t in COVARIATE_TARGETS:
            add(t, f"{t} ~ Group (unadjusted)", [])
            add(t, f"{t} ~ Group | Mobility", ["mobility_mg"])
            add(t, f"{t} ~ Group | HF power", ["HF_Power_ms2"])
            add(t, f"{t} ~ Group | HF power + MeanHR",
                ["HF_Power_ms2", "HRV_MeanHR"])
            add(t, f"{t} ~ Group | Mobility + MeanHR",
                ["mobility_mg", "HRV_MeanHR"])
            add(t, f"{t} ~ Group | Mobility + HF power + MeanHR",
                ["mobility_mg", "HF_Power_ms2", "HRV_MeanHR"])
        return pd.DataFrame(rows)

    cov_df = covariate_table(cov)
    cov_df.to_csv(os.path.join(OUTPUT_DIR, "Beta_Activity_Covariate.csv"),
                  index=False)

    keep = cov["coverage"] >= SENSITIVITY_MIN_COVERAGE
    excluded = sorted(cov.index[~keep].tolist())
    cov_sens = cov[keep]
    sens_df = covariate_table(cov_sens)
    sens_df.to_csv(
        os.path.join(OUTPUT_DIR, "Beta_Activity_Covariate_Sensitivity.csv"),
        index=False)

    print("\n  --- Mobility (group comparison) ---")
    print(f"    control median {grp_rows[0]['Control_median']} mg vs "
          f"patient median {grp_rows[0]['Patient_median']} mg; "
          f"r = {r_mob:+.3f}, p = {p_mob:.4f}")
    print("\n  --- Spectral measures, activity/vagal-adjusted (partial Spearman) ---")
    for row in cov_df.to_dict("records"):
        if not np.isnan(row["partial_r"]):
            print(f"    {row['Analysis']:<48s} r = {row['partial_r']:+.3f}, "
                  f"p = {row['p']:.4f}  (n={row['n']})")
    print(f"\n  --- Sensitivity: excluding coverage < "
          f"{SENSITIVITY_MIN_COVERAGE:.0%} "
          f"({len(excluded)} dropped: {', '.join(excluded) if excluded else 'none'}) ---")
    for row in sens_df.to_dict("records"):
        if not np.isnan(row["partial_r"]):
            print(f"    {row['Analysis']:<48s} r = {row['partial_r']:+.3f}, "
                  f"p = {row['p']:.4f}  (n={row['n']})")
    if missing:
        print(f"\n  [!] Result columns not found, some models skipped: {missing}")
    print(f"\n  Saved 4 files to {OUTPUT_DIR}/")

AMBULATION_SMOOTH_S   = 15.0
AMBULATION_MIN_BOUT_S = 20.0
AMBULATION_PAD_PRE_S  = 30.0
AMBULATION_PAD_POST_S = 180.0
AMBULATION_PAD_S      = AMBULATION_PAD_PRE_S
RESTLESSNESS_ABS_THRESHOLD = None
BOUT_THRESHOLD_SWEEP  = (2.0, 2.5, 3.0)
BOUT_MIN_LEN_SWEEP    = (10.0, 20.0, 30.0)
STATIONARY_MIN_BEATS  = 240


def motion_trace(acc, t_start, t_end, gravity_tc_s=GRAVITY_TIME_CONSTANT_S):
    w = acc[(acc["t"] >= t_start) & (acc["t"] <= t_end)]
    if len(w) < MIN_ACC_SAMPLES_IN_WINDOW:
        return None, None
    tt = w["t"].to_numpy()
    dt = np.median(np.diff(tt))
    fs = 1.0 / dt if dt > 0 else 100.0
    sigma = max(gravity_tc_s * fs, 1.0)
    x, y, z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    motion = np.sqrt((x - gaussian_filter1d(x, sigma, mode="nearest")) ** 2 +
                     (y - gaussian_filter1d(y, sigma, mode="nearest")) ** 2 +
                     (z - gaussian_filter1d(z, sigma, mode="nearest")) ** 2)
    smooth = gaussian_filter1d(motion, max(AMBULATION_SMOOTH_S * fs, 1.0),
                               mode="nearest")
    return tt, smooth


def detect_ambulation(tt, motion, threshold_mult=2.5, min_bout_s=None):
    if tt is None or motion is None or len(tt) < 10:
        return None
    min_bout_s = AMBULATION_MIN_BOUT_S if min_bout_s is None else min_bout_s
    base = float(np.median(motion))
    if not np.isfinite(base) or base <= 0:
        return None
    flag = motion > (threshold_mult * base)

    dt = float(np.median(np.diff(tt)))
    min_len = int(max(min_bout_s / dt, 1)) if dt > 0 else 1
    pad_pre  = int(max(AMBULATION_PAD_PRE_S / dt, 1)) if dt > 0 else 1
    pad_post = int(max(AMBULATION_PAD_POST_S / dt, 1)) if dt > 0 else 1

    out = np.zeros_like(flag, dtype=bool)
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j < len(flag) and flag[j]:
                j += 1
            if (j - i) >= min_len:
                out[max(0, i - pad_pre):min(len(flag), j + pad_post)] = True
            i = j
        else:
            i += 1
    return out


def acquisition_specification(acc=None, rr_times=None):
    spec = {
        "gravity_removal": "per-axis Gaussian high-pass: each axis minus a "
                           "Gaussian-smoothed copy of itself",
        "gravity_time_constant_s": GRAVITY_TIME_CONSTANT_S,
        "motion_definition": "Euclidean norm of the three gravity-removed axes",
        "motion_smoothing_s": AMBULATION_SMOOTH_S,
        "bout_threshold": "per-subject, relative: motion above "
                          "threshold_mult x the subject's own median motion",
        "bout_threshold_mult_default": 2.5,
        "bout_min_duration_s": AMBULATION_MIN_BOUT_S,
        "bout_pad_pre_s": AMBULATION_PAD_PRE_S,
        "bout_pad_post_s": AMBULATION_PAD_POST_S,
        "min_acc_samples_in_window": MIN_ACC_SAMPLES_IN_WINDOW,
        "min_window_coverage": MIN_WINDOW_COVERAGE,
        "synchronisation_method": "RR window mapped to wall-clock times, then "
                                  "accelerometer samples selected by "
                                  "timestamp within those bounds",
        "synchronisation_verified": "no independent verification available; "
                                    "the two streams share the device clock "
                                    "but no common event marker exists in the "
                                    "exported files",
        "dynamic_range_g": "unknown; not recoverable from the exported files",
        "device_side_filtering": "unknown; not documented in the dataset "
                                 "release",
        "axis_orientation": "unknown; the analysis uses the norm and is "
                            "therefore orientation-invariant",
    }
    if acc is not None and len(acc):
        try:
            tt = acc["t"].to_numpy()
            dt = float(np.median(np.diff(tt)))
            spec["sampling_interval_s_median"] = dt
            spec["sampling_rate_hz_median"] = (1.0 / dt if dt > 0 else np.nan)
            spec["n_samples"] = int(len(tt))
            spec["trace_duration_s"] = float(tt[-1] - tt[0])
            spec["sampling_interval_s_max"] = float(np.max(np.diff(tt)))
            spec["largest_gap_s"] = float(np.max(np.diff(tt)) - dt)
        except Exception:
            pass
    if rr_times is not None and len(rr_times):
        try:
            spec["rr_window_start_s"] = float(rr_times[0])
            spec["rr_window_end_s"] = float(rr_times[-1])
            spec["rr_window_duration_s"] = float(rr_times[-1] - rr_times[0])
        except Exception:
            pass
    return spec


def write_acquisition_specification(output_dir, spec):
    import csv
    path = os.path.join(output_dir, "Acquisition_Specification.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Item", "Value"])
        for k, v in spec.items():
            w.writerow([k, v])
    print(f"  Acquisition specification → {path}")
    return path


def nonbout_restlessness(tt, motion, bout_mask, abs_threshold=None):
    if tt is None or motion is None:
        return {}
    m = np.asarray(motion, dtype=float)
    if bout_mask is None:
        nb = np.ones(len(m), dtype=bool)
    else:
        nb = ~np.asarray(bout_mask, dtype=bool)
    if nb.sum() < 10:
        return {"NonBout_N_Samples": int(nb.sum())}
    seg = m[nb]
    out = {
        "Subject_Median_Motion": float(np.median(m)),
        "NonBout_Motion_Median": float(np.median(seg)),
        "NonBout_Motion_Mean": float(np.mean(seg)),
        "NonBout_Motion_Variance": float(np.var(seg, ddof=1)),
        "NonBout_Motion_IQR": float(np.percentile(seg, 75) -
                                    np.percentile(seg, 25)),
        "NonBout_N_Samples": int(nb.sum()),
        "NonBout_Frac_Of_Window": float(nb.mean()),
    }
    if abs_threshold is not None and np.isfinite(abs_threshold):
        out["NonBout_Pct_Above_AbsThreshold"] = float(
            100.0 * np.mean(seg > abs_threshold))
        out["NonBout_AbsThreshold_Used"] = float(abs_threshold)
    return out


def heart_rate_recovery(rr_times, rr_clean, bout_end_t, window_s=300.0):
    try:
        t = np.asarray(rr_times, dtype=float)
        rr = np.asarray(rr_clean, dtype=float)
        sel = (t >= bout_end_t) & (t <= bout_end_t + window_s)
        if sel.sum() < 30:
            return {}
        ts = t[sel] - bout_end_t
        hr = 60.0 / rr[sel]

        def _decay(tv, hr_inf, amp, tau):
            return hr_inf + amp * np.exp(-tv / tau)

        tau = np.nan
        hr_inf = float(np.median(hr[int(0.8 * len(hr)):]))
        method = "none"
        try:
            from scipy.optimize import curve_fit
            p0 = [hr_inf, max(float(hr[0] - hr_inf), 1.0), 60.0]
            popt, _ = curve_fit(_decay, ts, hr, p0=p0, maxfev=20000,
                                bounds=([0.0, 0.0, 2.0],
                                        [250.0, 150.0, 600.0]))
            hr_inf, _amp, tau = float(popt[0]), float(popt[1]), float(popt[2])
            method = "nonlinear_lsq"
        except Exception:
            amp = hr - hr_inf
            a0 = float(np.max(amp[:5])) if len(amp) >= 5 else np.nan
            ok = amp > max(0.1, 0.2 * a0) if np.isfinite(a0) else amp > 0.1
            if ok.sum() >= 15:
                coef = np.polyfit(ts[ok], np.log(amp[ok]), 1)
                tau = -1.0 / coef[0] if coef[0] < 0 else np.nan
                method = "log_linear_fallback"
        if not np.isfinite(tau):
            return {"HR_Recovery_Fit_Failed": True,
                    "HR_Recovery_Method": method}
        return {
            "HR_Recovery_Tau_s": float(tau),
            "HR_Recovery_Asymptote_bpm": float(hr_inf),
            "HR_Recovery_Start_bpm": float(hr[0]),
            "HR_Recovery_N_Beats": int(sel.sum()),
            "HR_Recovery_Method": method,
        }
    except Exception:
        return {}


def stationary_beat_mask(record_id):
    bounds = _rr_window_bounds(record_id)
    acc = _load_acc(record_id)
    if bounds is None:
        return None, "no usable R-R window"
    if acc is None:
        return None, "accelerometer file missing"
    t0, t1 = bounds
    tt, motion = motion_trace(acc, t0, t1)
    if tt is None:
        return None, "insufficient accelerometer coverage"
    amb = detect_ambulation(tt, motion)
    if amb is None:
        return None, "ambulation detection failed"

    path = _rr_path_for(record_id)
    if path is None:
        return None, "R-R file not found"
    try:
        df = pd.read_csv(path, sep=";")
    except Exception as exc:
        return None, f"R-R file unreadable: {type(exc).__name__}"
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if rr_col is None or ts_col is None:
        return None, "R-R file missing required columns"
    rr_ms = pd.to_numeric(df[rr_col], errors="coerce")
    good = rr_ms.notna()
    clock = _seconds_of_day(df[ts_col][good])
    in_window = (clock >= t0) & (clock <= t1)
    beat_clock = clock[in_window]

    idx = np.clip(np.searchsorted(tt, beat_clock), 0, len(tt) - 1)
    beat_ambulatory = amb[idx]
    mask = ~beat_ambulatory

    stats = {
        "Record_ID": record_id,
        "N_Beats_Window": int(len(mask)),
        "N_Beats_Stationary": int(mask.sum()),
        "Pct_Ambulatory": float(100.0 * beat_ambulatory.mean()),
        "Ambulation_Seconds": float(np.sum(amb) * np.median(np.diff(tt))),
        "Usable": bool(mask.sum() >= STATIONARY_MIN_BEATS),
    }
    return mask, stats


def activity_structure_report(record_ids, output_dir=OUTPUT_DIR):
    rows, problems = [], []
    pat = re.compile(r"^(control|treatment)_\d+$")
    ids = [r for r in record_ids if pat.match(str(r))]
    print(f"  Characterising activity structure for {len(ids)} subjects.")
    for i, rid in enumerate(ids, 1):
        print(f"    [{i:>3}/{len(ids)}] {rid}", flush=True)
        try:
            bounds = _rr_window_bounds(rid)
            acc = _load_acc(rid)
            if bounds is None or acc is None:
                problems.append(f"{rid}: window or accelerometer unavailable")
                continue
            t_start, t_end = bounds[0], bounds[1]
            tt, motion = motion_trace(acc, t_start, t_end)
            if tt is None:
                problems.append(f"{rid}: motion trace unavailable")
                continue
            mask = detect_ambulation(tt, motion)
            if mask is None:
                problems.append(f"{rid}: ambulation detection failed")
                continue

            _rest = nonbout_restlessness(tt, motion, mask,
                                         abs_threshold=RESTLESSNESS_ABS_THRESHOLD)

            padded = np.concatenate(([False], mask, [False]))
            d = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)
            dt = float(np.median(np.diff(tt))) if len(tt) > 1 else np.nan

            _row = {
                "Record_ID": rid,
                "Group": "control" if str(rid).startswith("control") else "patient",
                "N_bouts": int(len(starts)),
                "Total_ambul_s": float(mask.sum() * dt) if np.isfinite(dt) else np.nan,
                "Pct_window_ambul": float(100.0 * mask.mean()),
                "Trace_Duration_s": (float(len(mask) * dt)
                                     if np.isfinite(dt) else np.nan),
                "Pct_window_ambul_vs_1800s": (
                    float(100.0 * mask.sum() * dt / 1800.0)
                    if np.isfinite(dt) else np.nan),
                "Pct_Denominator": "accelerometer trace duration",
                "First_bout_s": (float(tt[starts[0]] - tt[0]) if len(starts)
                                 else np.nan),
                "Longest_bout_s": (float((ends - starts).max() * dt)
                                   if len(starts) and np.isfinite(dt) else np.nan),
            }
            _row.update(_rest)
            rows.append(_row)
        except Exception as exc:
            problems.append(f"{rid}: {type(exc).__name__}: {exc}")

    if not rows:
        print("  Activity structure report produced no rows.")
        for p in problems[:5]:
            print(f"    ! {p}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Activity_Structure.csv")
    df.to_csv(path, index=False, float_format="%.4f")

    from scipy.stats import mannwhitneyu
    summary = []
    for col in ("N_bouts", "Total_ambul_s", "Pct_window_ambul",
                "First_bout_s", "Longest_bout_s",
                "NonBout_Motion_Median", "NonBout_Motion_Variance",
                "NonBout_Motion_IQR"):
        a = pd.to_numeric(df.loc[df.Group == "control", col],
                          errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df.Group == "patient", col],
                          errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        summary.append({
            "Quantity": col,
            "Control_mean": a.mean(), "Control_SD": a.std(ddof=1),
            "Patient_mean": b.mean(), "Patient_SD": b.std(ddof=1),
            "r_effect": 1 - 2 * u / (len(a) * len(b)),
            "p_raw": p, "N_control": len(a), "N_patient": len(b)})
    sm = pd.DataFrame(summary)
    spath = os.path.join(output_dir, "Activity_Structure_Comparison.csv")
    sm.to_csv(spath, index=False, float_format="%.6f")

    print(f"\n  Activity structure     -> {path}")
    print(f"  Group comparison       -> {spath}")
    print(f"  {'quantity':<20}{'control':>12}{'patient':>12}{'r':>9}{'p':>10}")
    for _, r in sm.iterrows():
        print(f"  {r['Quantity']:<20}{r['Control_mean']:>12.2f}"
              f"{r['Patient_mean']:>12.2f}{r['r_effect']:>9.3f}{r['p_raw']:>10.4f}")
    if sm.empty or "p_raw" not in sm.columns:
        n_ctrl = int((df.Group == "control").sum())
        n_pat = int((df.Group == "patient").sum())
        print(f"\n  -> No group comparison performed: {n_ctrl} control and "
              f"{n_pat} patient subjects were characterised, and at least five "
              f"of each are required. The per-subject table above is still "
              f"valid.")
    elif (sm["p_raw"] > 0.05).all():
        print("\n  -> No quantity differs between groups. The Methods claim that")
        print("     both groups carry the same activity structure is supported by")
        print("     the accelerometer data and can be stated as a measurement")
        print("     rather than inferred from the acquisition protocol.")
    else:
        diff = sm.loc[sm["p_raw"] <= 0.05, "Quantity"].tolist()
        print(f"\n  -> These differ between groups: {diff}. The Methods claim of")
        print("     identical activity structure should be qualified accordingly.")
    if problems:
        print(f"  {len(problems)} subject(s) could not be characterised.")
    return df, sm


def build_stationary_segments(record_ids, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "stationary_masks")
    os.makedirs(mask_dir, exist_ok=True)

    import re as _re
    _pat = _re.compile(r"^(control|treatment)_\d+$")
    dropped = [r for r in record_ids if not _pat.match(str(r))]
    record_ids = [r for r in record_ids if _pat.match(str(r))]
    if dropped:
        print(f"  Ignoring {len(dropped)} non-subject row(s) from the results "
              f"CSV: {dropped}")

    rows, problems = [], []
    total = len(record_ids)
    t_start = time.time()
    print(f"  Processing {total} subjects. Each requires reading a full "
          f"accelerometer recording, so expect roughly 2-20 s per subject "
          f"depending on file size.")
    for i, rid in enumerate(record_ids, 1):
        elapsed = time.time() - t_start
        eta = (elapsed / (i - 1) * (total - i + 1)) if i > 1 else float("nan")
        print(f"    [{i:>3}/{total}] {rid:<16} "
              f"elapsed {elapsed:5.0f}s"
              + (f", ~{eta:.0f}s remaining" if i > 1 else ""),
              flush=True)
        mask, info = stationary_beat_mask(rid)
        if mask is None:
            problems.append(f"{rid}: {info}")
            rows.append({"Record_ID": rid, "N_Beats_Window": 0,
                         "N_Beats_Stationary": 0, "Pct_Ambulatory": np.nan,
                         "Ambulation_Seconds": np.nan, "Usable": False})
            continue
        np.save(os.path.join(mask_dir, f"{rid}_stationary.npy"), mask)
        rows.append(info)

    tab = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Stationary_Segments.csv")
    tab.to_csv(path, index=False, float_format="%.3f")
    print(f"  Stationary segment table → {path}")
    if len(tab):
        print(f"    Median ambulatory fraction: "
              f"{tab['Pct_Ambulatory'].median():.1f}% of beats")
        print(f"    Subjects retaining >= {STATIONARY_MIN_BEATS} stationary "
              f"beats: {int(tab['Usable'].sum())}/{len(tab)}")
    for p in problems:
        print(f"    ! {p}")
    return tab, problems


if __name__ == "__main__":
    print("[accelerometer] activity-confound analysis")
    results = run_activity_analysis()

    print("\n[accelerometer] enumerating subjects")
    try:
        res = _load_results()
        ids = list(res.index)
    except Exception as exc:
        print(f"  Could not load result CSVs to enumerate subjects: {exc}")
        ids = []
    print(f"  {len(ids)} record IDs loaded")

    if ids:
        print("\n[accelerometer] activity structure characterisation")
        activity_structure_report(ids)

    if ids:
        print("\n[accelerometer] acquisition specification")
        try:
            _acc = None
            for _rid in ids:
                try:
                    _acc = _load_acc(_rid)
                except Exception:
                    _acc = None
                if _acc is not None and len(_acc):
                    break
            _spec = acquisition_specification(acc=_acc)
            write_acquisition_specification(OUTPUT_DIR, _spec)
        except Exception as exc:
            print(f"  !! acquisition specification NOT written "
                  f"({type(exc).__name__}: {exc})")

    print("\n[accelerometer] ambulation excision -> stationary segments")
    if ids:
        build_stationary_segments(ids)
        print("\n  Next step: re-run the cohort scripts with the stationary "
              "masks applied to obtain movement-free estimates of the primary "
              "metrics, and report those alongside the full-window values.")
