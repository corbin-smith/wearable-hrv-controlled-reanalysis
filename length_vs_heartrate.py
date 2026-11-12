"""length_vs_heartrate.py: separate series length from mean heart rate for the entropy and recurrence contrasts."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, rankdata, t as t_dist

from pipeline import (
    SKIP_SECONDS as PIPE_SKIP,
    ANALYSIS_SECONDS as PIPE_ANALYSIS,reject_ectopic_beats, sample_entropy, approximate_entropy,
                      rqa_metrics, compute_recurrence_network, ordinal_tie_rate,
                      rcmse,
                      ENTROPY_M, ENTROPY_R_FRAC, MSE_MAX_SCALE, MSE_MIN_POINTS)

try:
    import neurokit2 as nk
except Exception:
    nk = None

COHORTS = {
    "control":   {"dir": "Control ECGs", "prefix": "control",
                  "csv": os.path.join("outputs", "healthy_hrvacc",
                                      "ecg_results_HealthyACC.csv")},
    "treatment": {"dir": "Psychiatric ECGs", "prefix": "treatment",
                  "csv": os.path.join("outputs", "psychiatric",
                                      "ecg_results_Psychiatric.csv")},
}
OUT_DIR = os.path.join("outputs", "comparison")
SKIP_SECONDS = PIPE_SKIP
ANALYSIS_SECONDS = PIPE_ANALYSIS
FIXED_N = [2000, 2400]
MIN_PER_GROUP = 15
MIN_RETAIN_FRAC = 0.90

METRICS = ["SampEn", "ApEn", "PermEn", "MSE_Mean", "RQA_DET", "RQA_LAM",
           "RNA_ClusteringCoeff", "RNA_Transitivity"]


def load_window(path):
    df = pd.read_csv(path, sep=";")
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    if rr_col is None:
        raise ValueError(f"no R-R column in {path}")
    rr = pd.to_numeric(df[rr_col], errors="coerce").dropna().values / 1000.0
    cum = np.cumsum(rr)
    m = (cum >= SKIP_SECONDS) & (cum < SKIP_SECONDS + ANALYSIS_SECONDS)
    return rr[m]


def metrics_for(x):
    out = {}
    sd = float(np.std(x, ddof=1))
    r_tol = ENTROPY_R_FRAC * sd
    out["SampEn"] = sample_entropy(x, m=ENTROPY_M, r=r_tol)
    out["ApEn"] = approximate_entropy(x, m=ENTROPY_M, r=r_tol)
    if nk is not None:
        try:
            pe, _ = nk.entropy_permutation(x, dimension=3, delay=1)
            out["PermEn"] = float(pe)
        except Exception:
            out["PermEn"] = np.nan
    else:
        out["PermEn"] = np.nan
    try:
        mse_scales, mse_values = rcmse(x, m=ENTROPY_M, r_frac=ENTROPY_R_FRAC,
                                       max_scale=MSE_MAX_SCALE,
                                       min_points=MSE_MIN_POINTS)
        finite = np.isfinite(mse_values)
        mse_values = mse_values[finite]
        out["MSE_Mean"] = (float(np.mean(mse_values)) if len(mse_values)
                           else np.nan)
        out["MSE_N_Scales"] = int(len(mse_values))
    except Exception:
        out["MSE_Mean"] = np.nan
        out["MSE_N_Scales"] = 0
    rq = rqa_metrics(x)
    out["RQA_DET"] = rq.get("RQA_DET", np.nan)
    out["RQA_LAM"] = rq.get("RQA_LAM", np.nan)
    rn = compute_recurrence_network(x)
    out["RNA_ClusteringCoeff"] = rn.get("RNA_ClusteringCoeff", np.nan)
    out["RNA_Transitivity"] = rn.get("RNA_Transitivity", np.nan)
    out["Ordinal_Tie_Rate"] = ordinal_tie_rate(x)
    out["N_beats"] = int(len(x))
    out["Duration_s"] = float(np.sum(x))
    out["MeanHR"] = float(60.0 / np.mean(x))
    return out


def gather(cohort, fixed_ns, verbose=True):
    cfg = COHORTS[cohort]
    if not os.path.exists(cfg["csv"]):
        sys.exit(f"  results CSV not found: {cfg['csv']}")
    df = pd.read_csv(cfg["csv"], index_col=0)
    subs = [i for i in df.index if str(i).startswith(cfg["prefix"] + "_")]
    rows = {}
    for k, rid in enumerate(subs, 1):
        path = os.path.join(cfg["dir"], f"{rid}.csv")
        if not os.path.exists(path):
            print(f"    ! {rid}: R-R file not found")
            continue
        if verbose:
            print(f"    [{k:>3}/{len(subs)}] {rid}", flush=True)
        try:
            rr = load_window(path)
            clean, _, _, _ = reject_ectopic_beats(rr, verbose=False)
            rec = {}
            for key, val in metrics_for(clean).items():
                rec[f"observed__{key}"] = val
            for n in fixed_ns:
                if len(clean) >= n:
                    for key, val in metrics_for(clean[:n]).items():
                        rec[f"fixed{n}__{key}"] = val
                else:
                    for key in (METRICS + ["MSE_N_Scales", "Ordinal_Tie_Rate",
                                           "N_beats", "Duration_s", "MeanHR"]):
                        rec[f"fixed{n}__{key}"] = np.nan
            rows[rid] = rec
        except Exception as exc:
            print(f"    ! {rid}: {type(exc).__name__}: {exc}")
    return pd.DataFrame.from_dict(rows, orient="index")


def _resid(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def partial(v, g, covs):
    ok = np.isfinite(v) & np.isfinite(g)
    for c in covs:
        ok &= np.isfinite(c)
    n = int(ok.sum())
    if n < 8:
        return np.nan, np.nan
    rc = [rankdata(c[ok]) for c in covs]
    a, b = _resid(rankdata(v[ok]), rc), _resid(rankdata(g[ok]), rc)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(a, b)[0, 1])
    dof = n - 2 - len(covs)
    if dof <= 0 or abs(r) >= 1.0:
        return r, np.nan
    return r, float(2.0 * t_dist.sf(abs(r * np.sqrt(dof / (1.0 - r ** 2))), dof))


def contrast(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, len(a), len(b)
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return 1 - 2 * u / (len(a) * len(b)), p, len(a), len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--fixed", type=int, nargs="+", default=FIXED_N)
    args = ap.parse_args()
    if not (args.check or args.run):
        args.check = True

    print("=" * 72)
    print("  RECORDING LENGTH VERSUS MEAN HEART RATE")
    print("=" * 72)

    counts = {}
    for coh, cfg in COHORTS.items():
        if not os.path.exists(cfg["csv"]):
            sys.exit(f"  results CSV not found: {cfg['csv']}")
        d = pd.read_csv(cfg["csv"], index_col=0)
        s = d.loc[[i for i in d.index if str(i).startswith(cfg["prefix"] + "_")]]
        counts[coh] = pd.to_numeric(s.get("Num_RR"), errors="coerce").dropna()
    print("\n  Subject retention at each fixed beat count:")
    for n in sorted(args.fixed):
        kc = int((counts["control"] >= n).sum())
        kp = int((counts["treatment"] >= n).sum())
        fc, fp = kc / len(counts["control"]), kp / len(counts["treatment"])
        if min(kc, kp) < MIN_PER_GROUP:
            flag = "   TOO FEW"
        elif min(fc, fp) < MIN_RETAIN_FRAC:
            flag = f"   UNEVEN LOSS ({fc:.0%} vs {fp:.0%})"
        else:
            flag = "   ok"
        print(f"    N = {n}: {kc}/{len(counts['control'])} controls, "
              f"{kp}/{len(counts['treatment'])} patients{flag}")
    print(f"\n  Note: the analysis window is time-defined ({SKIP_SECONDS} to "
          f"{SKIP_SECONDS + ANALYSIS_SECONDS} s), so every subject already has "
          f"the same {ANALYSIS_SECONDS} s of recording. Beat count differs only "
          f"because heart rate differs; a duration-matched condition would be "
          f"identical to 'observed' and is not computed.")

    if args.check:
        print("\n  Re-run with --run to compute. Roughly 25 minutes.")
        return

    usable, primary = [], None
    for n in sorted(args.fixed):
        kc = int((counts["control"] >= n).sum())
        kp = int((counts["treatment"] >= n).sum())
        if min(kc, kp) < MIN_PER_GROUP:
            continue
        usable.append(n)
        if (kc / len(counts["control"]) >= MIN_RETAIN_FRAC
                and kp / len(counts["treatment"]) >= MIN_RETAIN_FRAC):
            primary = n
    if not usable:
        sys.exit("  no fixed beat count retains enough subjects in both groups")
    if primary is None:
        primary = usable[0]
        print(f"\n  WARNING: no threshold retains {MIN_RETAIN_FRAC:.0%} of both "
              f"groups. Using N = {primary}, and the retention imbalance must be "
              f"reported alongside the result.")
    else:
        print(f"\n  Primary threshold: N = {primary} "
              f"(retains at least {MIN_RETAIN_FRAC:.0%} of both groups)")
    usable = sorted(usable, key=lambda n: (n != primary, n))

    frames = {}
    for coh in COHORTS:
        print(f"\n  {coh}:")
        frames[coh] = gather(coh, usable)

    h, p = frames["control"], frames["treatment"]
    os.makedirs(OUT_DIR, exist_ok=True)

    nh = pd.to_numeric(h["observed__N_beats"], errors="coerce").values
    np_ = pd.to_numeric(p["observed__N_beats"], errors="coerce").values
    hr_h = pd.to_numeric(h["observed__MeanHR"], errors="coerce").values
    hr_p = pd.to_numeric(p["observed__MeanHR"], errors="coerce").values
    allN, allHR = np.concatenate([nh, np_]), np.concatenate([hr_h, hr_p])
    ok = np.isfinite(allN) & np.isfinite(allHR)
    rho_pool = spearmanr(allN[ok], allHR[ok])[0]
    print("\n" + "=" * 72)
    print("  COLLINEARITY OF THE TWO CANDIDATE COVARIATES")
    print("=" * 72)
    print(f"    Spearman(beat count, mean HR), pooled : {rho_pool:+.4f}")
    print(f"      within controls                     : "
          f"{spearmanr(nh, hr_h)[0]:+.4f}")
    print(f"      within patients                     : "
          f"{spearmanr(np_, hr_p)[0]:+.4f}")
    resid_beats = allN[ok] - 30.0 * allHR[ok]
    print(f"    beats minus 30 x HR: mean {resid_beats.mean():+.1f}, "
          f"SD {resid_beats.std():.1f} beats")
    dh = pd.to_numeric(h["observed__Duration_s"], errors="coerce")
    dp = pd.to_numeric(p["observed__Duration_s"], errors="coerce")
    print(f"    observed duration: control {dh.mean():.1f} +/- {dh.std():.1f} s, "
          f"patient {dp.mean():.1f} +/- {dp.std():.1f} s")
    print("    (equal by construction; the window is time-defined, so any")
    print("     'recording length' difference is heart rate in other units)")
    print("    (30 x HR is the exact expectation over an 1800 s window; the")
    print("     residual is artifact removal only)")

    rows = []
    conds = [("observed", "observed")] + \
            [(f"fixed{n}", f"fixed {n} beats") for n in usable]
    grp = np.concatenate([np.zeros(len(h)), np.ones(len(p))])
    covN = np.concatenate([nh, np_])
    covHR = np.concatenate([hr_h, hr_p])

    for m in METRICS:
        for pref, label in conds:
            col = f"{pref}__{m}"
            if col not in h.columns or col not in p.columns:
                continue
            r, pv, n1, n2 = contrast(h[col], p[col])
            rec = {"Metric": m, "Condition": label, "r_effect": r,
                   "p_raw": pv, "N_control": n1, "N_patient": n2,
                   "r_adj_beats": np.nan, "p_adj_beats": np.nan,
                   "r_adj_HR": np.nan, "p_adj_HR": np.nan}
            if pref == "observed":
                v = np.concatenate([pd.to_numeric(h[col], errors="coerce").values,
                                    pd.to_numeric(p[col], errors="coerce").values])
                rb, pb = partial(v, grp, [covN])
                rr_, pr_ = partial(v, grp, [covHR])
                rec.update(r_adj_beats=rb, p_adj_beats=pb,
                           r_adj_HR=rr_, p_adj_HR=pr_)
            rows.append(rec)

    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "Length_vs_HeartRate.csv")
    out.to_csv(path, index=False, float_format="%.6f")

    print("\n" + "=" * 72)
    print("  GROUP CONTRASTS")
    print("=" * 72)
    print(f"  {'metric':<22}{'condition':<26}{'r':>9}{'p':>10}{'N':>10}")
    for _, r in out.iterrows():
        print(f"  {r['Metric']:<22}{r['Condition']:<26}{r['r_effect']:>9.3f}"
              f"{r['p_raw']:>10.4f}{int(r['N_control'])}/{int(r['N_patient']):<6}")

    print("\n  Residualised, for comparison with the manuscript:")
    print(f"  {'metric':<22}{'| beat count':>14}{'| mean HR':>14}")
    for _, r in out[out.Condition == "observed"].iterrows():
        print(f"  {r['Metric']:<22}{r['r_adj_beats']:>14.3f}{r['r_adj_HR']:>14.3f}")

    print("\n" + "=" * 72)
    print("  HOW TO READ THIS")
    print("=" * 72)
    key = f"fixed {primary} beats"
    sub = out[(out.Condition == key) & (out.Metric.isin(
        ["SampEn", "ApEn", "RQA_DET", "RNA_ClusteringCoeff",
         "RNA_Transitivity"]))]
    surviving = sub[(sub.p_raw < 0.05)]
    print(f"  At a fixed {primary} beats, {len(surviving)} of {len(sub)} "
          f"affected metrics still differ at p < 0.05.")
    if len(surviving) == 0:
        print("\n  -> Length equalised BY DESIGN removes the differences. The")
        print("     current interpretation survives. Add one sentence noting")
        print("     that beat count and mean heart rate are the same variable")
        print("     over a fixed window, so the adjustment is equally a")
        print("     heart-rate adjustment.")
    else:
        print("\n  -> Differences PERSIST when length is equalised by design.")
        print("     Residualising on beat count was therefore over-adjustment:")
        print("     it removed group variance rather than a length artifact.")
        print("     Sections 3.6, 4.5 and the Conclusions should be reframed")
        print("     as heart-rate mediation rather than length artifact, and")
        print("     the claim that these differences have 'nothing to do with")
        print("     psychiatric illness' should be withdrawn, since heart rate")
        print("     is itself a treatment-related group difference.")
    print(f"\n  written -> {path}")


if __name__ == "__main__":
    main()
