"""
Bootstrap 95% intervals for the heart-rate-adjusted group differences.

For each of the 21 primary measures: the partial Spearman correlation with group, unadjusted and with
mean heart rate held constant (regress on heart rate, then rank), and their difference, each with a
percentile interval. Subjects are resampled within group, 2,000 times, seed 0; one set of resample
indices serves every measure. Undefined resamples are dropped and counted. Point estimates must
reproduce HR_Adjusted_Comparison.csv (tolerance 1e-8) before anything is written.

Usage:  python hr_adjusted_bootstrap.py --check   (reproduction check only)
        python hr_adjusted_bootstrap.py --run
Writes outputs/comparison/healthy_hrvacc_vs_psych/HR_Adjusted_Bootstrap.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import comparison as CMP

N_BOOT = 2000
SEED = 0
OUT_DIR = os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych")
REF_CSV = os.path.join(OUT_DIR, "HR_Adjusted_Comparison.csv")
OUT_CSV = os.path.join(OUT_DIR, "HR_Adjusted_Bootstrap.csv")


def load():
    """Control and patient data frames, as in comparison.run_comparison."""
    df_h, df_p = CMP.load_cohorts(CMP.HRVACC_HEALTHY_CSV, CMP.PSYCH_CSV)
    if len(df_h) != 30 or len(df_p) != 30:
        sys.exit(f"expected 30 controls and 30 patients, found {len(df_h)} and {len(df_p)}")
    return df_h, df_p


def stats(y, g, hr):
    """Unadjusted and heart-rate-adjusted partial r, as in comparison.hr_adjusted_comparison."""
    ok = np.isfinite(y) & np.isfinite(hr)
    r_u, _ = spearmanr(y[ok], g[ok])
    r_a, _ = CMP.partial_spearman(y, g, [hr])
    return float(r_u), float(r_a)


def point_estimates(df_h, df_p, members):
    g = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])
    hr = CMP._stack(df_h, df_p, CMP.HR_ADJUSTMENT_COVARIATE)
    return {m: stats(CMP._stack(df_h, df_p, m), g, hr) for m in members}


def check(df_h, df_p):
    members = CMP.hr_family_available(df_h, df_p)
    ref = pd.read_csv(REF_CSV).set_index("Metric")
    if sorted(members) != sorted(ref.index):
        sys.exit(f"measure list differs from {REF_CSV}: {sorted(set(members) ^ set(ref.index))}")
    pts = point_estimates(df_h, df_p, members)
    worst = 0.0
    for m, (r_u, r_a) in pts.items():
        worst = max(worst, abs(r_u - ref.loc[m, "r_partial_unadjusted"]), abs(r_a - ref.loc[m, "r_partial_HR"]))
    if worst > 1e-8:                              # the CSV stores 8 significant digits
        sys.exit(f"point estimates do not reproduce {REF_CSV} (largest difference {worst:.3g})")
    print(f"  point estimates reproduce {REF_CSV} for all {len(members)} measures "
          f"(largest difference {worst:.1e})")
    return members, ref, pts


def run(df_h, df_p):
    members, ref, pts = check(df_h, df_p)
    nh, npat = len(df_h), len(df_p)
    g = np.concatenate([np.zeros(nh), np.ones(npat)])
    hr_all = CMP._stack(df_h, df_p, CMP.HR_ADJUSTMENT_COVARIATE)
    ys = {m: CMP._stack(df_h, df_p, m) for m in members}

    rng = np.random.default_rng(SEED)
    idx = [np.concatenate([rng.integers(0, nh, nh), nh + rng.integers(0, npat, npat)]) for _ in range(N_BOOT)]

    rows = []
    for m in [m for m in ref.index]:              # keep the row order of HR_Adjusted_Comparison.csv
        ru_b, ra_b, d_b = [], [], []
        y = ys[m]
        for ix in idx:
            r_u, r_a = stats(y[ix], g[ix], hr_all[ix])
            if np.isfinite(r_u):
                ru_b.append(r_u)
            if np.isfinite(r_a):
                ra_b.append(r_a)
            if np.isfinite(r_u) and np.isfinite(r_a):
                d_b.append(r_a - r_u)
        if min(len(ru_b), len(ra_b), len(d_b)) < 0.99 * N_BOOT:
            sys.exit(f"{m}: too many undefined resamples ({len(ru_b)}, {len(ra_b)}, {len(d_b)} of {N_BOOT})")
        u_lo, u_hi = np.percentile(ru_b, [2.5, 97.5])
        a_lo, a_hi = np.percentile(ra_b, [2.5, 97.5])
        d_lo, d_hi = np.percentile(d_b, [2.5, 97.5])
        r_u, r_a = pts[m]
        rows.append({
            "Metric": m,
            "r_partial_unadjusted": r_u, "unadj_CI_low": u_lo, "unadj_CI_high": u_hi,
            "r_partial_HR": r_a, "HR_CI_low": a_lo, "HR_CI_high": a_hi,
            "delta_HR_minus_unadj": r_a - r_u, "delta_CI_low": d_lo, "delta_CI_high": d_hi,
            "HR_CI_excludes_0": bool(a_lo > 0 or a_hi < 0),
            "delta_CI_excludes_0": bool(d_lo > 0 or d_hi < 0),
            "N_boot_kept": int(min(len(ru_b), len(ra_b), len(d_b))),
            "N_Boot": N_BOOT, "Seed": SEED,
        })
        print(f"  {m:<22} unadj {r_u:+.3f} [{u_lo:+.3f}, {u_hi:+.3f}]   HR {r_a:+.3f} [{a_lo:+.3f}, {a_hi:+.3f}]"
              f"   delta {r_a - r_u:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]")
    out = pd.DataFrame(rows)
    # The adjusted interval should exclude zero where the adjusted test is significant before multiplicity correction;
    # disagreements near p = 0.05 are expected and only reported.
    excl = out.set_index("Metric")["HR_CI_excludes_0"]
    disagree = [m for m in out.Metric if bool(excl[m]) != bool(ref.loc[m, "p_raw"] < 0.05)]
    listed = ", ".join("%s (p = %.3f)" % (m, ref.loc[m, "p_raw"]) for m in disagree)
    print("  measures where the adjusted interval and the uncorrected test disagree about zero: " + (listed or "none"))
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.8g")
    print(f"  wrote {OUT_CSV} ({len(out)} measures, {N_BOOT} resamples, seed {SEED})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--check", action="store_true", help="verify the point estimates only")
    grp.add_argument("--run", action="store_true", help="compute and write the intervals")
    a = ap.parse_args()
    df_h, df_p = load()
    if a.check:
        check(df_h, df_p)
    else:
        run(df_h, df_p)
