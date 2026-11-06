"""mayer_residual_check.py: per-subject size of the baroreflex (Mayer-band) bump in the spectrum.

For each subject: the PSD with the pipeline's Welch settings, a log-log line fitted over the slope's
fitting band with 0.07-0.13 Hz excluded, and the mean and maximum log10 residual inside 0.07-0.13 Hz.
This avoids both the group-mean spectrum and the peak model, whose fits often reach a parameter bound.

Usage: python mayer_residual_check.py
Writes outputs/comparison/Mayer_Residual_PerSubject.csv and prints the group comparison.
"""
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import linregress, mannwhitneyu

from pipeline import (
    SKIP_SECONDS as PIPE_SKIP,
    ANALYSIS_SECONDS as PIPE_ANALYSIS,reject_ectopic_beats, truncate_series,
                      INTERP_FS, WELCH_NPERSEG, WELCH_OVERLAP_FRAC,
                      WELCH_WINDOW, WELCH_DETREND,
                      BETA_FIT_BAND, MAYER_BAND)

COHORTS = {
    "control":   {"dir": "Control ECGs",     "prefix": "control"},
    "treatment": {"dir": "Psychiatric ECGs", "prefix": "treatment"},
}
RESULT_CSVS = {
    "control":   os.path.join("outputs", "healthy_hrvacc",
                              "ecg_results_HealthyACC.csv"),
    "treatment": os.path.join("outputs", "psychiatric",
                              "ecg_results_Psychiatric.csv"),
}
OUT_DIR = os.path.join("outputs", "comparison")
# Analysis window constants come from pipeline.py.
SKIP_SECONDS = PIPE_SKIP
ANALYSIS_SECONDS = PIPE_ANALYSIS


def subject_ids(cohort):
    path = RESULT_CSVS[cohort]
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, index_col=0)
    pref = COHORTS[cohort]["prefix"]
    return [i for i in df.index if str(i).startswith(pref + "_")]


def spectrum(record_id, cohort):
    path = os.path.join(COHORTS[cohort]["dir"], f"{record_id}.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, sep=";")
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    if rr_col is None:
        return None, None
    rr = pd.to_numeric(df[rr_col], errors="coerce").dropna().values / 1000.0
    cum = np.cumsum(rr)
    m = (cum >= SKIP_SECONDS) & (cum < SKIP_SECONDS + ANALYSIS_SECONDS)
    rr_win, t_win = rr[m], cum[m]
    if len(rr_win) < 100:
        return None, None
    clean, _, _, _ = reject_ectopic_beats(rr_win, verbose=False)
    clean = truncate_series(clean)
    t = np.concatenate([[t_win[0]], t_win[0] + np.cumsum(clean[:-1])])
    grid = np.arange(t[0], t[-1], 1.0 / INTERP_FS)
    vals = interp1d(t, clean, kind="cubic", bounds_error=False,
                    fill_value="extrapolate")(grid)
    nperseg = int(min(len(vals), WELCH_NPERSEG))
    return welch(vals, fs=INTERP_FS, nperseg=nperseg,
                 noverlap=int(nperseg * WELCH_OVERLAP_FRAC),
                 window=WELCH_WINDOW, detrend=WELCH_DETREND)


def mayer_residual(f, p):
    """Mean and max log10 residual inside the Mayer band, relative to a fit
    that excluded that band."""
    ok = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p > 0)
    f, p = f[ok], p[ok]
    band = (f >= BETA_FIT_BAND[0]) & (f <= BETA_FIT_BAND[1])
    mayer = (f >= MAYER_BAND[0]) & (f <= MAYER_BAND[1])
    fit_mask = band & ~mayer
    if fit_mask.sum() < 4 or mayer.sum() < 2:
        return np.nan, np.nan, np.nan
    lr = linregress(np.log10(f[fit_mask]), np.log10(p[fit_mask]))
    resid = np.log10(p) - (lr.intercept + lr.slope * np.log10(f))
    return (float(np.mean(resid[mayer])), float(np.max(resid[mayer])),
            float(-lr.slope))


def main():
    rows = []
    for cohort in COHORTS:
        ids = subject_ids(cohort)
        print(f"  {cohort}: {len(ids)} subjects")
        for i, rid in enumerate(ids, 1):
            print(f"    [{i:>3}/{len(ids)}] {rid}", flush=True)
            f, p = spectrum(rid, cohort)
            if f is None:
                continue
            mean_r, max_r, beta_notch = mayer_residual(f, p)
            rows.append({"Record_ID": rid,
                         "Group": "control" if cohort == "control" else "patient",
                         "Mayer_Resid_Mean": mean_r,
                         "Mayer_Resid_Max": max_r,
                         "Beta_NotchFit": beta_notch})
    if not rows:
        print("  no subjects processed; check the R-R folder paths.")
        return
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "Mayer_Residual_PerSubject.csv")
    df.to_csv(path, index=False, float_format="%.6f")

    print(f"\n  {'quantity':<22}{'control':>10}{'patient':>10}{'r':>9}{'p':>10}")
    for col in ("Mayer_Resid_Mean", "Mayer_Resid_Max", "Beta_NotchFit"):
        a = df.loc[df.Group == "control", col].dropna()
        b = df.loc[df.Group == "patient", col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u, pv = mannwhitneyu(a, b, alternative="two-sided")
        r = 1 - 2 * u / (len(a) * len(b))
        print(f"  {col:<22}{a.mean():10.4f}{b.mean():10.4f}{r:9.3f}{pv:10.2e}")
    print(f"\n  written -> {path}")
    print("\n  HOW TO READ THIS")
    print("  Mayer_Resid_Mean/Max is how far the spectrum rises above its own")
    print("  aperiodic trend inside 0.07-0.13 Hz, in log10 units.")
    print("    p > 0.05  -> the bump is comparable in both groups, so it cannot")
    print("                 generate a group difference in the fitted slope.")
    print("                 A visual difference in the group-mean figure is")
    print("                 then an averaging effect, not a confound.")
    print("    p < 0.05  -> the bump genuinely differs. Report it, and lean on")
    print("                 PSD_Beta_NoMayer rather than PSD_Beta as primary.")
    print("                 Note the direction: a LARGER bump in controls")
    print("                 flattens THEIR slope and inflates the difference.")


if __name__ == "__main__":
    main()