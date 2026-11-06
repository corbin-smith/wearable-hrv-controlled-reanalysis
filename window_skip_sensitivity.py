"""window_skip_sensitivity.py: recompute the main contrasts at each onset skip considered.

The 600 s skip was chosen after inspecting patients' heart rates at skips of 60 and 360 s, so the
spectral slopes (excised and low band), VLF/LF, SampEn, ApEn, RQA determinism, the two recurrence-
network measures, mean heart rate, beat count and RMSSD are recomputed at all three skips, each over
an 1800 s window. Surrogate tests are not repeated.

Usage:  python window_skip_sensitivity.py --check
        python window_skip_sensitivity.py --run
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import mannwhitneyu

from pipeline import (
    ANALYSIS_SECONDS as PIPE_ANALYSIS,reject_ectopic_beats, sample_entropy, approximate_entropy,
                      rqa_metrics, compute_recurrence_network, _fit_beta,
                      ENTROPY_M, ENTROPY_R_FRAC, INTERP_FS, WELCH_NPERSEG,
                      WELCH_OVERLAP_FRAC, WELCH_WINDOW, WELCH_DETREND,
                      MAYER_BAND, VLF_BAND, LF_BAND)

COHORTS = {
    "control":   {"dir": "Control ECGs", "prefix": "control",
                  "csv": os.path.join("outputs", "healthy_hrvacc",
                                      "ecg_results_HealthyACC.csv")},
    "treatment": {"dir": "Psychiatric ECGs", "prefix": "treatment",
                  "csv": os.path.join("outputs", "psychiatric",
                                      "ecg_results_Psychiatric.csv")},
}
OUT_DIR = os.path.join("outputs", "comparison")
SKIPS = [60, 360, 600]
# Window length comes from pipeline.py; the skip is swept here.
ANALYSIS_SECONDS = PIPE_ANALYSIS
PRIMARY_BAND = (0.01, 0.15)
LOW_BAND = (0.01, 0.07)

METRICS = ["PSD_Beta_NoMayer", "PSD_Beta_low", "VLF_LF_Ratio",
           "HRV_MeanHR", "HRV_RMSSD",
           "Num_RR", "SampEn", "ApEn", "RQA_DET",
           "RNA_ClusteringCoeff", "RNA_Transitivity"]


def load_window(path, skip_s):
    df = pd.read_csv(path, sep=";")
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    if rr_col is None:
        raise ValueError(f"no R-R column in {path}")
    rr = pd.to_numeric(df[rr_col], errors="coerce").dropna().values / 1000.0
    cum = np.cumsum(rr)
    m = (cum >= skip_s) & (cum < skip_s + ANALYSIS_SECONDS)
    return rr[m]


def metrics_for(clean):
    """Spectral, entropy and recurrence metrics for one cleaned R-R series."""
    out = {}
    out["Num_RR"] = int(len(clean))
    out["HRV_MeanHR"] = float(60.0 / np.mean(clean))
    out["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(clean) ** 2)) * 1000.0)

    # Spectral slope with the pipeline's Welch settings.
    t = np.cumsum(clean)
    grid = np.arange(t[0], t[-1], 1.0 / INTERP_FS)
    vals = interp1d(t, clean, kind="cubic", bounds_error=False,
                    fill_value="extrapolate")(grid)
    nperseg = int(min(len(vals), WELCH_NPERSEG))
    f, p = welch(vals, fs=INTERP_FS, nperseg=nperseg,
                 noverlap=int(nperseg * WELCH_OVERLAP_FRAC),
                 window=WELCH_WINDOW, detrend=WELCH_DETREND)
    out["PSD_Beta_NoMayer"] = _fit_beta(f, p, PRIMARY_BAND,
                                        exclude_band=MAYER_BAND)[0]
    out["PSD_Beta_low"] = _fit_beta(f, p, LOW_BAND)[0]

    # VLF/LF with the pipeline's band integration (trapezoid over the bins in each band). The spectrum
    # uses this script's time base (each interval placed at its end), so the 600 s row can differ slightly
    # from the main-run values, as the slope rows do.
    def _band(lo_hi):
        idx = (f >= lo_hi[0]) & (f <= lo_hi[1])
        return np.trapezoid(p[idx], f[idx]) if idx.sum() > 1 else np.nan
    _vlf, _lf = _band(VLF_BAND), _band(LF_BAND)
    out["VLF_LF_Ratio"] = (_vlf / _lf if (np.isfinite(_lf) and _lf > 0)
                           else np.nan)

    sd = float(np.std(clean, ddof=1))
    r_tol = ENTROPY_R_FRAC * sd
    out["SampEn"] = sample_entropy(clean, m=ENTROPY_M, r=r_tol)
    out["ApEn"] = approximate_entropy(clean, m=ENTROPY_M, r=r_tol)
    out["RQA_DET"] = rqa_metrics(clean).get("RQA_DET", np.nan)
    rn = compute_recurrence_network(clean)
    out["RNA_ClusteringCoeff"] = rn.get("RNA_ClusteringCoeff", np.nan)
    out["RNA_Transitivity"] = rn.get("RNA_Transitivity", np.nan)
    return out


def gather(cohort, skips, verbose=True):
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
        rec = {}
        for s in skips:
            try:
                rr = load_window(path, s)
                if len(rr) < 300:
                    for m in METRICS:
                        rec[f"skip{s}__{m}"] = np.nan
                    continue
                clean, _, _, _ = reject_ectopic_beats(rr, verbose=False)
                for key, val in metrics_for(clean).items():
                    rec[f"skip{s}__{key}"] = val
            except Exception as exc:
                print(f"    ! {rid} skip={s}: {type(exc).__name__}: {exc}")
                for m in METRICS:
                    rec[f"skip{s}__{m}"] = np.nan
        rows[rid] = rec
    return pd.DataFrame.from_dict(rows, orient="index")


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
    ap.add_argument("--skips", type=int, nargs="+", default=SKIPS)
    args = ap.parse_args()
    if not (args.check or args.run):
        args.check = True

    print("=" * 72)
    print("  ANALYSIS-WINDOW SKIP SENSITIVITY")
    print("=" * 72)
    print(f"  Skips tested: {args.skips} s, each over an "
          f"{ANALYSIS_SECONDS} s window")
    print(f"  Metrics: {', '.join(METRICS)}")

    if args.check:
        for coh, cfg in COHORTS.items():
            n = len([f for f in os.listdir(cfg["dir"])
                     if f.startswith(cfg["prefix"] + "_")]) \
                if os.path.isdir(cfg["dir"]) else 0
            print(f"  {coh}: {n} R-R files found in {cfg['dir']}")
        print("\n  Re-run with --run. Roughly 15 minutes.")
        return

    frames = {}
    for coh in COHORTS:
        print(f"\n  {coh}:")
        frames[coh] = gather(coh, args.skips)
    h, p = frames["control"], frames["treatment"]

    rows = []
    for m in METRICS:
        for s in args.skips:
            col = f"skip{s}__{m}"
            if col not in h.columns or col not in p.columns:
                continue
            r, pv, n1, n2 = contrast(h[col], p[col])
            rows.append({
                "Metric": m, "Skip_s": s,
                "Control_mean": pd.to_numeric(h[col], errors="coerce").mean(),
                "Patient_mean": pd.to_numeric(p[col], errors="coerce").mean(),
                "r_effect": r, "p_raw": pv,
                "N_control": n1, "N_patient": n2})
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "Window_Skip_Sensitivity.csv")
    out.to_csv(path, index=False, float_format="%.6f")

    print("\n" + "=" * 72)
    print("  GROUP CONTRAST AT EACH SKIP")
    print("=" * 72)
    print(f"  {'metric':<22}{'skip':>7}{'control':>11}{'patient':>11}"
          f"{'r':>9}{'p':>10}")
    for m in METRICS:
        sub = out[out.Metric == m]
        for _, r in sub.iterrows():
            print(f"  {r['Metric']:<22}{int(r['Skip_s']):>7}"
                  f"{r['Control_mean']:>11.3f}{r['Patient_mean']:>11.3f}"
                  f"{r['r_effect']:>9.3f}{r['p_raw']:>10.4f}")
        print()

    # Stability of the two slope measures.
    print("=" * 72)
    print("  STABILITY")
    print("=" * 72)
    for m in ("PSD_Beta_NoMayer", "PSD_Beta_low", "VLF_LF_Ratio", "SampEn",
              "RQA_DET"):
        sub = out[out.Metric == m].dropna(subset=["r_effect"])
        if len(sub) < 2:
            continue
        spread = sub["r_effect"].max() - sub["r_effect"].min()
        allsig = bool((sub["p_raw"] < 0.05).all())
        nonesig = bool((sub["p_raw"] >= 0.05).all())
        verdict = ("stable" if spread < 0.10 and (allsig or nonesig)
                   else "VARIES WITH SKIP")
        print(f"  {m:<22} r ranges {sub['r_effect'].min():+.3f} to "
              f"{sub['r_effect'].max():+.3f} (spread {spread:.3f})   {verdict}")
    print("\n  A spread below about 0.10 with consistent significance means the")
    print("  600 s choice did not determine the result, and the label-aware")
    print("  selection can be reported as immaterial. A larger spread means the")
    print("  window choice is a genuine analytic degree of freedom and must be")
    print("  disclosed as one.")
    print(f"\n  written -> {path}")


if __name__ == "__main__":
    main()