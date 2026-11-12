"""recording_times_and_gaps.py: time of day and transmission gaps in the HRV-ACC recordings."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import accelerometer_mobility as ACC
import comparison as CMP

OUT_DIR = os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych")
GAP_THRESHOLD_S = 2.0
N_RANDOM = 2000
SEED = 0
SKIP_S, WINDOW_S = ACC.RR_SKIP_SECONDS, ACC.RR_ANALYSIS_SECONDS


def hhmm(hours):
    total = int(round(hours * 60))
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def record_info(rid):
    path = ACC._rr_path_for(rid)
    df = pd.read_csv(path, sep=";")
    rr_col = next(c for c in df.columns if "rr" in c.lower() or "interval" in c.lower())
    ts_col = next(c for c in df.columns if "timestamp" in c.lower())
    rr = pd.to_numeric(df[rr_col], errors="coerce")
    ok = rr.notna()
    rr = rr[ok].to_numpy(dtype=float) / 1000.0
    clock = ACC._seconds_of_day(df[ts_col][ok])
    cum = np.cumsum(rr)
    excess = np.diff(clock) - rr[1:]
    after = np.arange(1, len(rr))
    in_win = (cum[after] >= SKIP_S) & (cum[after] < SKIP_S + WINDOW_S)
    gap = excess > GAP_THRESHOLD_S
    b = ACC._rr_window_bounds(rid)
    if b is None:
        sys.exit(f"no window bounds for {rid}")
    start_h, end_h = b[0] / 3600.0, b[1] / 3600.0
    if end_h >= 24.0:
        sys.exit(f"{rid}: window crosses midnight; handle before reporting")
    return {
        "Record_ID": rid,
        "Window_Start": hhmm(start_h), "Window_End": hhmm(end_h),
        "Start_Hours": start_h, "End_Hours": end_h,
        "Window_Clock_Minutes": (end_h - start_h) * 60.0,
        "Gaps_File": int(gap.sum()), "Missing_s_File": float(excess[gap].sum()),
        "Gaps_Window": int((gap & in_win).sum()), "Missing_s_Window": float(excess[gap & in_win].sum()),
        "Longest_Gap_Window_s": float(excess[gap & in_win].max()) if (gap & in_win).any() else 0.0,
        "Max_NonGap_Excess_s": float(excess[~gap].max()),
        "Min_Gap_s": float(excess[gap].min()) if gap.any() else np.nan,
    }


def families(df_h, df_p, drop):
    h = df_h.drop(index=[i for i in drop if i in df_h.index])
    p = df_p.drop(index=[i for i in drop if i in df_p.index])
    prim = [m for m in CMP.PRIMARY_METRICS if m in h.columns and m in p.columns]
    rows = {}
    rp = [CMP._mw(h[m], p[m]) for m in prim]
    padj = CMP.benjamini_hochberg(np.array([x[1] for x in rp]))
    for m, (r, pv, nh, npat), pa in zip(prim, rp, padj):
        rows[("primary", m)] = dict(r=r, p=pv, p_adj=pa, n_c=nh, n_p=npat)
    members = CMP.hr_family_available(h, p)
    g = np.concatenate([np.zeros(len(h)), np.ones(len(p))])
    hr = CMP._stack(h, p, CMP.HR_ADJUSTMENT_COVARIATE)
    ra = [CMP.partial_spearman(CMP._stack(h, p, m), g, [hr]) for m in members]
    padj = CMP.benjamini_hochberg(np.array([x[1] for x in ra]))
    for m, (r, pv), pa in zip(members, ra, padj):
        rows[("hr_adjusted", m)] = dict(r=float(r), p=float(pv), p_adj=pa, n_c=len(h), n_p=len(p))
    return rows


def main():
    df_h, df_p = CMP.load_cohorts(CMP.HRVACC_HEALTHY_CSV, CMP.PSYCH_CSV)
    if len(df_h) != 30 or len(df_p) != 30:
        sys.exit(f"expected 30 controls and 30 patients, found {len(df_h)} and {len(df_p)}")

    info = []
    for group, ids in (("control", df_h.index), ("patient", df_p.index)):
        for rid in ids.astype(str):
            d = record_info(rid)
            d["Group"] = group
            info.append(d)
    info = pd.DataFrame(info)[["Record_ID", "Group", "Window_Start", "Window_End", "Start_Hours", "End_Hours",
                               "Window_Clock_Minutes", "Gaps_File", "Missing_s_File", "Gaps_Window",
                               "Missing_s_Window", "Longest_Gap_Window_s", "Max_NonGap_Excess_s", "Min_Gap_s"]]
    max_nongap = float(info.Max_NonGap_Excess_s.max())
    min_gap = float(info.Min_Gap_s.min())
    if not max_nongap < GAP_THRESHOLD_S < min_gap:
        sys.exit(f"threshold {GAP_THRESHOLD_S} s does not separate jitter ({max_nongap:.2f} s) from gaps ({min_gap:.2f} s)")
    info["Clock_Minus_Beat_s"] = info.Window_Clock_Minutes * 60.0 - WINDOW_S
    resid = (info.Clock_Minus_Beat_s - info.Missing_s_Window).abs()
    if resid.max() > 10.0:
        sys.exit(f"clock span and detected gaps disagree by up to {resid.max():.1f} s; check the clock mapping")
    gap_ids = info.loc[info.Gaps_Window > 0, "Record_ID"].tolist()

    c = info[info.Group == "control"].Start_Hours.to_numpy()
    p = info[info.Group == "patient"].Start_Hours.to_numpy()
    u, pval = mannwhitneyu(c, p, alternative="two-sided", use_continuity=True, method="auto")
    r = 1.0 - 2.0 * u / (len(c) * len(p))
    summ = []
    for g, x in (("control", c), ("patient", p)):
        sub = info[info.Group == g]
        summ.append({"Row": g, "N": len(x), "Median_Start": hhmm(np.median(x)), "Earliest_Start": hhmm(x.min()),
                     "Latest_Start": hhmm(x.max()), "Latest_End": hhmm(sub.End_Hours.max()),
                     "Median_Start_Hours": float(np.median(x)), "Records_With_Gap_In_File": int((sub.Gaps_File > 0).sum()),
                     "Records_With_Gap_In_Window": int((sub.Gaps_Window > 0).sum()),
                     "Missing_s_Window_Total": float(sub.Missing_s_Window.sum())})
    summ.append({"Row": "patients vs controls, window start", "N": len(c) + len(p), "U": float(u),
                 "r_rank_biserial": float(r), "p": float(pval),
                 "Test": "Mann-Whitney U, two-sided, continuity correction; U for controls; positive r = later in patients"})
    summ.append({"Row": "gap threshold", "Threshold_s": GAP_THRESHOLD_S, "Max_NonGap_Excess_s": max_nongap,
                 "Min_Gap_s": min_gap, "Max_Clock_vs_Gap_Residual_s": float(resid.max()),
                 "Gapfree_Clock_Minus_Beat_min_s": float(info.loc[info.Gaps_Window == 0, "Clock_Minus_Beat_s"].min()),
                 "Gapfree_Clock_Minus_Beat_max_s": float(info.loc[info.Gaps_Window == 0, "Clock_Minus_Beat_s"].max())})

    full = families(df_h, df_p, [])
    ref = pd.read_csv(os.path.join(OUT_DIR, "group_comparison_stats.csv"), index_col=0)
    for (fam, m), v in full.items():
        if fam != "primary":
            continue
        lab = CMP.METRIC_LABELS.get(m, m)
        if abs(v["r"] - ref.loc[lab, "r_effect"]) > 1e-6 or abs(v["p_adj"] - ref.loc[lab, "p_adj"]) > 1e-6:
            sys.exit(f"full-sample primary result for {m} does not reproduce group_comparison_stats.csv")
    href = pd.read_csv(os.path.join(OUT_DIR, "HR_Adjusted_Comparison.csv")).set_index("Metric")
    for (fam, m), v in full.items():
        if fam == "hr_adjusted" and (abs(v["r"] - href.loc[m, "r_partial_HR"]) > 1e-7
                                     or abs(v["p_adj"] - href.loc[m, "p_adj"]) > 1e-6):
            sys.exit(f"full-sample adjusted result for {m} does not reproduce HR_Adjusted_Comparison.csv")
    excl = families(df_h, df_p, gap_ids)

    def changed(f_, e_):
        sig_f, sig_e = f_["p_adj"] < CMP.ALPHA, e_["p_adj"] < CMP.ALPHA
        return (sig_f != sig_e) or (sig_f and sig_e and np.sign(f_["r"]) != np.sign(e_["r"]))

    n_drop_c = sum(1 for i in gap_ids if i in df_h.index)
    n_drop_p = sum(1 for i in gap_ids if i in df_p.index)
    rng = np.random.default_rng(SEED)
    flips = {k: 0 for k in full}
    n_changed_random = []
    for _ in range(N_RANDOM):
        drop = (list(rng.choice(df_h.index.to_numpy(), n_drop_c, replace=False)) +
                list(rng.choice(df_p.index.to_numpy(), n_drop_p, replace=False)))
        rnd = families(df_h, df_p, drop)
        n = 0
        for k in full:
            if changed(full[k], rnd[k]):
                flips[k] += 1
                n += 1
        n_changed_random.append(n)
    n_changed_random = np.array(n_changed_random)
    rows = []
    for key in full:
        f_, e_ = full[key], excl[key]
        sig_f, sig_e = f_["p_adj"] < CMP.ALPHA, e_["p_adj"] < CMP.ALPHA
        rows.append({"Family": key[0], "Metric": key[1],
                     "r_full": f_["r"], "p_adj_full": f_["p_adj"], "Significant_full": sig_f,
                     "r_without_gaps": e_["r"], "p_adj_without_gaps": e_["p_adj"], "Significant_without_gaps": sig_e,
                     "N_control_without": e_["n_c"], "N_patient_without": e_["n_p"],
                     "Conclusion_changed": bool(changed(f_, e_)),
                     "Share_random_exclusions_changing": flips[key] / N_RANDOM})
    sens = pd.DataFrame(rows)
    n_gap_changed = int(sens.Conclusion_changed.sum())
    summ.append({"Row": "random exclusion calibration", "N": N_RANDOM, "Seed": SEED,
                 "Dropped_Controls": n_drop_c, "Dropped_Patients": n_drop_p,
                 "Gap_Records_Conclusions_Changed": n_gap_changed,
                 "Random_Median_Conclusions_Changed": float(np.median(n_changed_random)),
                 "Random_Share_At_Least_As_Many": float(np.mean(n_changed_random >= n_gap_changed))})

    os.makedirs(OUT_DIR, exist_ok=True)
    info.to_csv(os.path.join(OUT_DIR, "Recording_Times_Gaps.csv"), index=False, float_format="%.6f")
    pd.DataFrame(summ).to_csv(os.path.join(OUT_DIR, "Recording_Times_Gaps_Summary.csv"), index=False,
                              float_format="%.6f")
    sens.to_csv(os.path.join(OUT_DIR, "Gap_Exclusion_Sensitivity.csv"), index=False, float_format="%.8g")

    for s in summ[:2]:
        print(f"  {s['Row']:<8} window start median {s['Median_Start']}, range {s['Earliest_Start']} to "
              f"{s['Latest_Start']}; gaps in file {s['Records_With_Gap_In_File']} records, in window "
              f"{s['Records_With_Gap_In_Window']} ({s['Missing_s_Window_Total']:.1f} s missing)")
    print(f"  window start, patients vs controls: r = {r:+.3f}, p = {pval:.3f}")
    print(f"  gap threshold {GAP_THRESHOLD_S} s: largest jitter {max_nongap:.2f} s, smallest gap {min_gap:.2f} s; "
          f"clock span less beat span matches the detected gaps to within {resid.max():.1f} s")
    print(f"  records with a gap in the window: {', '.join(gap_ids) or 'none'}")
    ch = sens[sens.Conclusion_changed]
    print(f"  conclusions changed without them: {len(ch)} of {len(sens)}"
          + ("".join(f"\n     {r_.Family}: {r_.Metric} {r_.r_full:+.3f} (p adj {r_.p_adj_full:.3f}) -> "
                     f"{r_.r_without_gaps:+.3f} (p adj {r_.p_adj_without_gaps:.3f})" for r_ in ch.itertuples())))
    print(f"  largest change in r: {np.max(np.abs(sens.r_full - sens.r_without_gaps)):.3f}")
    print(f"  random exclusion of {n_drop_c} controls and {n_drop_p} patient(s), {N_RANDOM} draws: median "
          f"{np.median(n_changed_random):.0f} conclusions changed; {np.mean(n_changed_random >= n_gap_changed):.1%} "
          f"of draws changed at least {n_gap_changed}")
    for r_ in ch.itertuples():
        print(f"     {r_.Metric}: changed in {r_.Share_random_exclusions_changing:.1%} of random exclusions")
    print(f"  wrote Recording_Times_Gaps.csv, Recording_Times_Gaps_Summary.csv, Gap_Exclusion_Sensitivity.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
