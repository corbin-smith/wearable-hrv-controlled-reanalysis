"""estimator_changes.py: how much each estimator choice moves a group difference, with an interval."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import comparison as CMP

OUT_DIR = os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych")
OUT_CSV = os.path.join(OUT_DIR, "Estimator_Paired_Changes.csv")
N_BOOT = 2000
SEED = 0

PAIRS = [
    ("DFA alpha1 lower bound", "DFA_Alpha1", "DFA_Alpha1", "DFA_Alpha1_nmin8", "8 to 16 beats"),
    ("DFA alpha1 lower bound", "DFA_Alpha1", "DFA_Alpha1", "DFA_Alpha1_nmin10", "10 to 16 beats"),
    ("Embedding", "RQA_DET", "RQA_DET", "RQA_DET_SelEmbed", "each subject's own embedding"),
    ("Embedding", "RQA_LAM", "RQA_LAM", "RQA_LAM_SelEmbed", "each subject's own embedding"),
    ("Minimum line length", "RQA_DET", "RQA_DET", "RQA_DET_lmin3", "minimum line length 3"),
    ("Minimum line length", "RQA_DET", "RQA_DET", "RQA_DET_lmin4", "minimum line length 4"),
    ("Surrogates", "SampEn", "SampEn", "SURR_SampEn_Mean", "mean of the subject's surrogates"),
    ("Surrogates", "SampEn", "SampEn", ("minus", "SampEn", "SURR_SampEn_Mean"), "observed minus surrogate mean"),
    ("Surrogates", "SampEn", "SampEn", "SURR_SampEn_Z", "surrogate Z-score"),
    ("Surrogates", "MSE_Mean", "MSE_Mean", "SURR_MSE_Mean_Mean", "mean of the subject's surrogates"),
    ("Surrogates", "MSE_Mean", "MSE_Mean", ("minus", "MSE_Mean", "SURR_MSE_Mean_Mean"), "observed minus surrogate mean"),
    ("Surrogates", "MSE_Mean", "MSE_Mean", "SURR_MSE_Mean_Z", "surrogate Z-score"),
    ("Surrogates", "DFA_Alpha1", "DFA_Alpha1", "SURR_DFA_Alpha1_Mean", "mean of the subject's surrogates"),
    ("Surrogates", "DFA_Alpha1", "DFA_Alpha1", ("minus", "DFA_Alpha1", "SURR_DFA_Alpha1_Mean"), "observed minus surrogate mean"),
    ("Surrogates", "DFA_Alpha1", "DFA_Alpha1", "SURR_DFA_Alpha1_Z", "surrogate Z-score"),
]


def _r(values, group, with_p=False):
    a = values[(group == 0) & np.isfinite(values)]
    b = values[(group == 1) & np.isfinite(values)]
    if len(a) < 3 or len(b) < 3:
        return (np.nan, np.nan) if with_p else np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided", use_continuity=CMP.MWU_USE_CONTINUITY,
                        method=CMP.MWU_METHOD)
    r = 1.0 - 2.0 * u / (len(a) * len(b))
    return (r, float(p)) if with_p else r


def main():
    df_h, df_p = CMP.load_cohorts(CMP.HRVACC_HEALTHY_CSV, CMP.PSYCH_CSV)
    if len(df_h) != 30 or len(df_p) != 30:
        sys.exit(f"expected 30 + 30 recordings, found {len(df_h)} + {len(df_p)}")
    grp = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])
    idx_c, idx_p = np.where(grp == 0)[0], np.where(grp == 1)[0]
    rng = np.random.default_rng(SEED)
    boot = [np.concatenate([rng.choice(idx_c, len(idx_c)), rng.choice(idx_p, len(idx_p))])
            for _ in range(N_BOOT)]

    def column(spec):
        if isinstance(spec, tuple):
            return column(spec[1]) - column(spec[2])
        if spec not in df_h.columns or spec not in df_p.columns:
            return None
        return CMP._stack(df_h, df_p, spec)

    rows = []
    for choice, measure, ref, alt, alt_label in PAIRS:
        a, b = column(ref), column(alt)
        if a is None or b is None:
            print(f"  skipped (column absent): {measure}, {alt_label}")
            continue
        ra, pa = _r(a, grp, with_p=True)
        rb, pb = _r(b, grp, with_p=True)
        d = np.array([_r(b[ix], grp[ix]) - _r(a[ix], grp[ix]) for ix in boot], dtype=float)
        rbv = np.array([_r(b[ix], grp[ix]) for ix in boot], dtype=float)
        d, rbv = d[np.isfinite(d)], rbv[np.isfinite(rbv)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        blo, bhi = np.percentile(rbv, [2.5, 97.5])
        rows.append({
            "Choice": choice, "Metric": measure,
            "Reference": ref, "Alternative": alt if isinstance(alt, str) else f"{alt[1]} - {alt[2]}",
            "Alternative_Label": alt_label,
            "N_Controls": int(np.isfinite(b[grp == 0]).sum()), "N_Patients": int(np.isfinite(b[grp == 1]).sum()),
            "r_reference": ra, "p_reference": pa,
            "r_alternative": rb, "p_alternative": pb,
            "r_alternative_CI_low": float(blo), "r_alternative_CI_high": float(bhi),
            "delta": rb - ra, "delta_CI_low": float(lo), "delta_CI_high": float(hi),
            "delta_CI_excludes_0": bool(lo > 0 or hi < 0),
            "N_Boot": N_BOOT, "Seed": SEED,
        })
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.8g")
    print(f"  {'choice':22s}{'measure':12s}{'alternative':34s}{'r ref':>8s}{'r alt':>8s}   change [95% CI]")
    for _, r in out.iterrows():
        print(f"  {r.Choice:22s}{r.Metric:12s}{r.Alternative_Label:34s}{r.r_reference:>+8.3f}{r.r_alternative:>+8.3f}"
              f"   {r.delta:+.3f} [{r.delta_CI_low:+.3f}, {r.delta_CI_high:+.3f}]"
              f"{'' if r.delta_CI_excludes_0 else '  (includes 0)'}")
    print(f"  Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
