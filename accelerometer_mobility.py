"""
accelerometer_mobility.py
=========================
Phase 1 activity-confound control for the cardiac-criticality pilot.

WHAT THIS DOES (in plain language)
----------------------------------
The HRV-ACC dataset was recorded with a Polar H10 chest strap that, besides the
heartbeat (R-R) data you already analyse, also logged motion from a built-in
accelerometer -- a small chip that senses movement in three directions, called
X, Y and Z. Peer reviewers pointed out a fair problem: your 30-minute analysis
window contains a walk, so part of the patient-vs-control difference in the
spectral-slope metric (PSD beta) could simply reflect how much each person moved
rather than their physiology. The accelerometer data needed to check this is
already in the dataset but was never used.

This script closes that gap. It:

  1. Computes ONE "mobility" number per person over the SAME window your HRV/beta
     metrics use (how much they physically moved, in milli-g).
  2. Tests whether the two groups differ in mobility (i.e. whether activity is a
     real confound at all).
  3. Tests whether the PSD-beta group difference SURVIVES statistically removing
     the influence of mobility -- and, jointly, of heart rate and vagal power.
     If beta survives, that is genuine evidence it is not merely an activity or
     vagal artifact. If it does not survive, that is critical to know before a
     reviewer (or a prospective mentor) discovers it.

HOW "MOBILITY" IS DEFINED
-------------------------
Following the dataset authors (Ksiazek et al.), gravity is removed from each axis
by subtracting a heavily time-smoothed copy of that axis (gravity and posture
change slowly; genuine motion is faster). What remains is the "dynamic"
acceleration; its magnitude, averaged over the window, is the mobility
coefficient. This is an INDEPENDENT re-implementation of that idea (their code is
GPL-3.0; this file deliberately shares no code with it), so this file can carry
whatever license you choose. Please still CITE their paper for the method.

WHAT YOU NEED TO PROVIDE (see the download instructions in the chat):
  - Your existing per-subject result CSVs (ecg_results_HealthyACC.csv and
    ecg_results_Psychiatric.csv), which supply PSD_Beta, HF_Power_ms2,
    HRV_MeanHR, the group label and the QC flag.
  - Your existing R-R folders (used only to locate the exact analysis window in
    clock time so the accelerometer window matches the beat-based window).
  - The accelerometer CSVs from Zenodo, placed in one folder (see below).
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, mannwhitneyu


# ----------------------------------------------------------------------------
# CONFIGURATION  (edit these paths only if your folder names differ)
# ----------------------------------------------------------------------------
def _first_existing(candidates, default=None):
    """Return the first path that exists, else a sensible default."""
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return default if default is not None else candidates[0]


# R-R interval folders (same ones your pipeline already uses).
CONTROL_RR_DIR = _first_existing(["Control ECGs"], default="Control ECGs")
PSYCH_RR_DIR   = _first_existing(["Psychiatric ECGs"], default="Psychiatric ECGs")

# Accelerometer folder. The Zenodo archive "accelerometer_anonymized_data.zip"
# unzips to a single folder holding BOTH control_N.csv and treatment_N.csv.
# Put that folder next to your R-R folders. The script checks several common
# names so you do not have to rename anything.
ACC_DIR = _first_existing(
    ["Accelerometer ECGs", "accelerometer_anonymized_data", "Accelerometer",
     "ACC ECGs"],
    default="Accelerometer ECGs",
)

# Per-subject result CSVs (supply PSD_Beta, HF power, mean HR, QC flag).
HEALTHY_CSV = _first_existing([
    os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "ecg_results_HealthyACC.csv",
])
PSYCH_CSV = _first_existing([
    os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv"),
    "ecg_results_Psychiatric.csv",
])

OUTPUT_DIR = os.path.join("outputs", "accelerometer")

# Analysis window -- MUST match pipeline.py exactly (skip 10 min, use next 30).
RR_SKIP_SECONDS     = 600
RR_ANALYSIS_SECONDS = 1800

# Gravity-removal smoothing timescale, in seconds. Motion faster than this is
# treated as movement; slower drift is treated as gravity/posture. 10 s matches
# the dataset authors' setting at the native sampling rate.
GRAVITY_TIME_CONSTANT_S = 10.0

# Quality guards.
MIN_ACC_SAMPLES_IN_WINDOW = 500     # need at least this many ACC samples
MIN_WINDOW_COVERAGE       = 0.50    # need >=50% of the window covered by ACC

# Sensitivity check: re-run the covariate models on only the subjects whose
# accelerometer coverage in the window is at least this high, to confirm the
# result is not driven by the few subjects with patchy motion data.
SENSITIVITY_MIN_COVERAGE  = 0.80

# If mobility_coefficients.csv already exists, reuse it instead of re-reading the
# large accelerometer files (much faster). Set True to force a fresh computation.
FORCE_RECOMPUTE = False

QC_COLUMN = "QC_Overall_Pass"


# ----------------------------------------------------------------------------
# TIME HANDLING
# ----------------------------------------------------------------------------
def _seconds_of_day(timestamps):
    """
    Convert a column of 'Phone timestamp' strings (time-of-day only, e.g.
    '13:00:17.196000') into seconds elapsed since midnight, as floats.

    The dataset removed the calendar date for anonymisation, so a recording that
    happens to cross midnight would appear to jump backwards by ~24 h. We detect
    any such backward jump and add 24 h to everything after it, which makes the
    series increase smoothly again. Because the R-R and accelerometer streams for
    one person were recorded together, applying this same fix to both keeps them
    on a shared clock.
    """
    t = pd.to_datetime(timestamps, format="%H:%M:%S.%f", errors="coerce")
    if t.isna().any():                        # fall back for odd formats
        t = t.fillna(pd.to_datetime(timestamps, errors="coerce"))
    secs = (t.dt.hour * 3600 + t.dt.minute * 60
            + t.dt.second + t.dt.microsecond / 1e6).to_numpy(dtype=float)
    for w in np.where(np.diff(secs) < -43200.0)[0]:   # >12 h backward = midnight
        secs[w + 1:] += 86400.0
    return secs


# ----------------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------------
def _rr_path_for(record_id):
    """Map a Record_ID (e.g. 'control_16') to its R-R CSV path."""
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
    """Map a Record_ID to its accelerometer CSV path (same filename stem)."""
    cand = os.path.join(ACC_DIR, f"{record_id}.csv")
    return cand if os.path.exists(cand) else None


def _rr_window_bounds(record_id):
    """
    Find the exact clock-time span of the analysis window for one subject, using
    the R-R file. We reproduce the pipeline's window (skip the first 600 s of
    beat time, then take 1800 s) and read off the wall-clock timestamps at those
    two boundaries, so the accelerometer window lines up with the beat window.

    Returns (start_seconds_of_day, end_seconds_of_day) or None.
    """
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

    cum = np.cumsum(rr_ms / 1000.0)               # beat time from first beat (s)
    i_start = int(np.searchsorted(cum, RR_SKIP_SECONDS))
    i_end   = int(np.searchsorted(cum, RR_SKIP_SECONDS + RR_ANALYSIS_SECONDS))
    i_start = min(i_start, len(clock) - 1)
    i_end   = min(i_end,   len(clock) - 1)
    if i_end <= i_start:
        return None
    return float(clock[i_start]), float(clock[i_end])


def _load_acc(record_id):
    """
    Load one accelerometer CSV into a tidy frame with columns t (seconds of day),
    X, Y, Z (milli-g). Returns None if the file is missing or malformed.
    """
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


# ----------------------------------------------------------------------------
# MOBILITY
# ----------------------------------------------------------------------------
def compute_mobility(acc, t_start, t_end, gravity_tc_s=GRAVITY_TIME_CONSTANT_S):
    """
    Mobility coefficient over [t_start, t_end] (seconds of day).

    Step by step: keep only accelerometer samples inside the window; estimate the
    slow gravity/posture component of each axis with a Gaussian smoother; subtract
    it to leave the fast "motion" component; take the 3-D magnitude of that motion
    at each instant; average over the window. Larger = more physical movement.

    Returns (mobility_mg, n_samples_used, coverage_fraction). mobility_mg is NaN
    if there is too little accelerometer data inside the window.
    """
    w = acc[(acc["t"] >= t_start) & (acc["t"] <= t_end)]
    n = len(w)
    span = max(t_end - t_start, 1e-9)
    if n < MIN_ACC_SAMPLES_IN_WINDOW:
        return np.nan, n, 0.0

    tt = w["t"].to_numpy()
    dt = np.median(np.diff(tt))
    fs = 1.0 / dt if dt > 0 else 100.0            # samples per second
    coverage = min(n / (span * fs), 1.0)
    if coverage < MIN_WINDOW_COVERAGE:
        return np.nan, n, float(coverage)

    sigma = max(gravity_tc_s * fs, 1.0)           # smoothing width in samples
    x, y, z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    xe = gaussian_filter1d(x, sigma, mode="nearest")
    ye = gaussian_filter1d(y, sigma, mode="nearest")
    ze = gaussian_filter1d(z, sigma, mode="nearest")
    motion = np.sqrt((x - xe) ** 2 + (y - ye) ** 2 + (z - ze) ** 2)
    return float(np.mean(motion)), n, float(coverage)


def build_mobility_table(record_ids):
    """Compute mobility for every subject; return a tidy DataFrame + warnings."""
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


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------
def rank_biserial(a, b):
    """
    Mann-Whitney U test + rank-biserial effect size for two groups a, b.
    r > 0 means group a tends to have larger values than group b.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 2.0 * U / (len(a) * len(b)) - 1.0
    return float(r), float(p), U


def partial_spearman(df, target_col, outcome_col, covar_cols):
    """
    Partial Spearman correlation between target and outcome, controlling for the
    covariates. Method is IDENTICAL to comparison.py: residualise the raw target
    and raw outcome on the covariates by ordinary least squares, then Spearman-
    correlate the residuals. With no covariates it is a plain Spearman.
    """
    t = df[target_col].to_numpy(dtype=float)
    o = df[outcome_col].to_numpy(dtype=float)
    if covar_cols:
        Z = df[covar_cols].to_numpy(dtype=float)
        Z_des = np.column_stack([np.ones(len(Z)), Z])

        def ols_resid(y):
            coef, _, _, _ = np.linalg.lstsq(Z_des, y, rcond=None)
            return y - Z_des @ coef

        r, p = spearmanr(ols_resid(t), ols_resid(o))
    else:
        r, p = spearmanr(t, o)
    return float(r), float(p)


# ----------------------------------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------------------------------
def _load_results():
    """Load and merge the two per-subject result CSVs; add Group (0/1)."""
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

    # 1) Mobility for every subject -----------------------------------------
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

    df = results.join(mob, how="left")
    usable = df[df["mobility_mg"].notna()].copy()
    print(f"  Mobility computed for {len(usable)} / {len(df)} subjects.")

    # 2) Do the groups differ in physical activity? -------------------------
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

    # 3) Does the PSD-beta group difference survive covariate control? ------
    #    Each row controls for a different set of covariates. Mobility answers
    #    the activity confound; HF power + mean HR answer the vagal/rate
    #    confound; the last row controls for all three at once.
    need = ["PSD_Beta", "Group", "mobility_mg", "HF_Power_ms2", "HRV_MeanHR"]
    missing = [c for c in need if c not in usable.columns]
    cov = usable.dropna(
        subset=[c for c in need if c in usable.columns and c != "Group"]).copy()

    def covariate_table(frame):
        """Run the six PSD-beta covariate models on `frame`; return a DataFrame."""
        rows = []

        def add(label, covars):
            if any(cc not in frame.columns for cc in covars):
                rows.append({"Analysis": label, "partial_r": np.nan, "p": np.nan,
                             "n": len(frame), "note": "covariate column missing"})
                return
            r, pv = partial_spearman(frame, "PSD_Beta", "Group", covars)
            rows.append({"Analysis": label, "partial_r": round(r, 3),
                         "p": round(pv, 5), "n": len(frame), "note": ""})

        add("PSD_Beta ~ Group (unadjusted)", [])
        add("PSD_Beta ~ Group | Mobility", ["mobility_mg"])
        add("PSD_Beta ~ Group | HF power", ["HF_Power_ms2"])
        add("PSD_Beta ~ Group | HF power + MeanHR", ["HF_Power_ms2", "HRV_MeanHR"])
        add("PSD_Beta ~ Group | Mobility + MeanHR", ["mobility_mg", "HRV_MeanHR"])
        add("PSD_Beta ~ Group | Mobility + HF power + MeanHR",
            ["mobility_mg", "HF_Power_ms2", "HRV_MeanHR"])
        return pd.DataFrame(rows)

    # Main table: all subjects with usable mobility.
    cov_df = covariate_table(cov)
    cov_df.to_csv(os.path.join(OUTPUT_DIR, "Beta_Activity_Covariate.csv"),
                  index=False)

    # Sensitivity table: drop subjects whose accelerometer coverage is patchy,
    # to confirm the finding is not carried by a few poorly-sampled records.
    keep = cov["coverage"] >= SENSITIVITY_MIN_COVERAGE
    excluded = sorted(cov.index[~keep].tolist())
    cov_sens = cov[keep]
    sens_df = covariate_table(cov_sens)
    sens_df.to_csv(
        os.path.join(OUTPUT_DIR, "Beta_Activity_Covariate_Sensitivity.csv"),
        index=False)

    # 4) Console summary ----------------------------------------------------
    print("\n  --- Mobility (group comparison) ---")
    print(f"    control median {grp_rows[0]['Control_median']} mg vs "
          f"patient median {grp_rows[0]['Patient_median']} mg; "
          f"r = {r_mob:+.3f}, p = {p_mob:.4f}")
    print("\n  --- PSD beta, activity/vagal-adjusted (partial Spearman) ---")
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


if __name__ == "__main__":
    print("[accelerometer] activity-confound analysis")
    run_activity_analysis()
