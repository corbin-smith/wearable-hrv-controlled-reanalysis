"""figures.py: main Figures 2 and 4 and the supplementary figures (HRV-ACC controls against patients, MIT-BIH as external controls)."""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr, rankdata, t as t_dist
from scipy.signal import welch
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


def _first_existing(candidates, default=None):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return default if default is not None else candidates[0]

HRVACC_HEALTHY_CSV = _first_existing([
    os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "ecg_results_HealthyACC.csv",
])
PSYCH_CSV = _first_existing([
    os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv"),
    "ecg_results_Psychiatric.csv",
])
MITBIH_HEALTHY_CSV = _first_existing([
    os.path.join("outputs", "healthy", "ecg_results_Healthy.csv"),
    "ecg_results_Healthy.csv",
])


def _summary(name):
    return _first_existing([
        os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych", name),
        name,
    ])
BOOTSTRAP_CSV = _summary("Bootstrap_Effect_Sizes.csv")
COVARIATE_CSV = _summary("PSD_Beta_Covariate_Analysis.csv")
SURROGATE_CSV = _summary("Surrogate_Nonlinearity_Summary.csv")

ACTIVITY_COV_CSV = _first_existing([
    os.path.join("outputs", "accelerometer", "Beta_Activity_Covariate.csv"),
    "Beta_Activity_Covariate.csv",
])

BAND_SENS_CSV = _summary("PSD_Beta_Range_Sensitivity.csv")

CONTROL_RR_DIR = _first_existing(["Control ECGs"], default="Control ECGs")
PSYCH_RR_DIR   = _first_existing(["Psychiatric ECGs"], default="Psychiatric ECGs")

from pipeline import (SKIP_SECONDS as RR_SKIP_SECONDS,
                      ANALYSIS_SECONDS as RR_ANALYSIS_SECONDS)

OUTPUT_DIR = os.path.join("outputs", "figures")

QC_COLUMN = "QC_Overall_Pass"

C_HEALTHY  = "#0077BB"
C_PSYCH    = "#CC3311"
C_MITBIH   = "#009988"

DOMAIN_DISPLAY = {"HRV": "Time and frequency domain", "Scaling": "Spectral slope and DFA",
                  "Entropy": "Entropy", "RQA": "Recurrence"}

DOMAIN_COLORS = {
    "HRV":      "#0077BB",
    "Scaling":  "#EE7733",
    "Entropy":  "#009988",
    "RQA":      "#AA4499",
}
C_NONSIG = "#BBBBBB"
C_GRID   = "#EEEEEE"

plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             8.5,
    "axes.labelsize":        9,
    "axes.titlesize":        9.5,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "legend.framealpha":     0.9,
    "figure.dpi":            120,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
})

METRIC_LABELS = {
    "PSD_Beta_NoMayer": "Slope β (band excised)",
    "MFDFA_Width": "MFDFA Δh (width)",
    "HRV_MeanNN":      "Mean NN (ms)",
    "HRV_SDNN":        "SDNN (ms)",
    "HRV_RMSSD":       "RMSSD (ms)",
    "HRV_pNN50":       "pNN50 (%)",
    "HRV_MeanHR":      "Mean HR (bpm)",
    "HRV_SD1":         "SD1 (ms)",
    "HRV_SD2":         "SD2 (ms)",
    "HRV_SD1SD2":      "SD1/SD2",
    "LF_Power_ms2":    "LF Power (ms\u00b2)",
    "HF_Power_ms2":    "HF Power (ms\u00b2)",
    "Total_Power_ms2": "Total Power (ms\u00b2)",
    "LF_HF_Ratio":     "LF/HF Ratio",
    "PSD_Beta":        "Slope \u03b2 (uncorrected)",
    "SampEn":          "Sample Entropy",
    "ApEn":            "Approx. Entropy",
    "PermEn":          "Permutation Entropy",
    "MSE_Mean":        "MSE Mean",
    "DFA_Alpha":       "DFA \u03b1 (full)",
    "DFA_Alpha1":      "DFA \u03b1\u2081 (4\u201316)",
    "DFA_Alpha2":      "DFA \u03b1\u2082 (16\u201364)",
    "RQA_REC":         "RQA Recurrence",
    "RQA_DET":         "RQA Determinism",
    "RQA_LAM":         "RQA Laminarity",
    "MFDFA_H_Mean":    "MFDFA H\u0305 (mean)",
    "MFDFA_H2":        "MFDFA H(q=2)",
    "RNA_MeanDegree":      "RNA Mean Degree",
    "RNA_ClusteringCoeff": "RNA Clustering",
    "RNA_Transitivity":    "RNA Transitivity",
}

METRIC_BATTERY = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_MeanHR",
    "HRV_SD1", "HRV_SD2", "HRV_SD1SD2",
    "LF_Power_ms2", "HF_Power_ms2", "Total_Power_ms2", "LF_HF_Ratio",
    "PSD_Beta",
    "SampEn", "ApEn", "PermEn", "MSE_Mean",
    "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2",
    "MFDFA_Width", "MFDFA_H_Mean", "MFDFA_H2",
    "RQA_REC", "RQA_DET", "RQA_LAM",
    "RNA_MeanDegree", "RNA_ClusteringCoeff", "RNA_Transitivity",
]

METRIC_DOMAIN = {
    "HRV_MeanNN": "HRV", "HRV_SDNN": "HRV", "HRV_RMSSD": "HRV",
    "HRV_pNN50": "HRV", "HRV_MeanHR": "HRV", "HRV_SD1": "HRV",
    "HRV_SD2": "HRV", "HRV_SD1SD2": "HRV", "LF_Power_ms2": "HRV",
    "HF_Power_ms2": "HRV", "Total_Power_ms2": "HRV", "LF_HF_Ratio": "HRV",
    "PSD_Beta": "Scaling", "DFA_Alpha": "Scaling",
    "DFA_Alpha1": "Scaling", "DFA_Alpha2": "Scaling",
    "MFDFA_Width": "Scaling", "MFDFA_H_Mean": "Scaling",
    "MFDFA_H2": "Scaling",
    "SampEn": "Entropy", "ApEn": "Entropy",
    "PermEn": "Entropy", "MSE_Mean": "Entropy",
    "RQA_REC": "RQA", "RQA_DET": "RQA", "RQA_LAM": "RQA",
    "RNA_MeanDegree": "RQA", "RNA_ClusteringCoeff": "RQA",
    "RNA_Transitivity": "RQA",
}

FIG2_METRICS = [
    "HRV_RMSSD", "Total_Power_ms2",
    "PSD_Beta", "PSD_Beta_NoMayer",
    "DFA_Alpha2",
    "MFDFA_Width", "MFDFA_H2",
    "RQA_DET",
]

FIG2_DOMAIN_TAG = {
    "HRV_RMSSD": "beat-to-beat variability", "Total_Power_ms2": "total variability",
    "PSD_Beta": "fitted 0.016\u20130.148 Hz", "PSD_Beta_NoMayer": "0.07\u20130.13 Hz excised",
    "DFA_Alpha2": "fluctuation scaling, 16\u201364 beats",
    "MFDFA_Width": "multifractal width", "MFDFA_H2": "H(q=2)",
    "RQA_DET": "recurrence structure",
}

MFDFA_FAMILY = ["MFDFA_Width", "MFDFA_H_Mean", "MFDFA_H2"]
RNA_FAMILY   = ["RNA_ClusteringCoeff", "RNA_Transitivity"]
DETERMINISTIC_CHECKS = ["RQA_REC", "RNA_MeanDegree"]

STATISTICALLY_REDUNDANT = ["ApEn"]


def _metric_family(m):
    if m in DETERMINISTIC_CHECKS:
        return "none"
    if m in MFDFA_FAMILY:
        return "MFDFA"
    if m in RNA_FAMILY:
        return "RNA"
    if m in STATISTICALLY_REDUNDANT:
        return "none"
    return "primary"

LABEL_TO_COL = {v: k for k, v in METRIC_LABELS.items()}
LABEL_TO_COL.update({"PSD \u03b2": "PSD_Beta", "PSD \u03b2 (Mayer excised)": "PSD_Beta_NoMayer"})


def _load_cohort(path, label):
    if not os.path.exists(path):
        print(f"  [!] {label}: file not found at {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    summary_mask = df.index.astype(str).str.contains(
        r"^(Cohort_|Healthy_|HealthyACC_|Psychiatric_|MITBIH_)",
        regex=True, na=False)
    df = df[~summary_mask].copy()
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().any():
            df[c] = converted
    if QC_COLUMN in df.columns:
        qc = df[QC_COLUMN].astype(str).str.lower().isin(["true", "1", "1.0"])
        df = df[qc].copy()
    print(f"  Loaded {label}: N = {len(df)}")
    return df


def rank_biserial(h, p):
    h = np.asarray(h, float); p = np.asarray(p, float)
    h = h[np.isfinite(h)]; p = p[np.isfinite(p)]
    if len(h) < 3 or len(p) < 3:
        return np.nan, np.nan
    u, pval = mannwhitneyu(h, p, alternative="two-sided")
    r = 1 - 2 * u / (len(h) * len(p))
    return r, pval


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def _bh_selftest():
    p = np.array([0.18577, 0.00023, 0.34029, 0.00081])
    expected = np.array([0.24769, 0.00092, 0.34029, 0.00162])
    got = benjamini_hochberg(p)
    assert np.allclose(got, expected, atol=1e-4), \
        f"BH self-test failed: {got} != {expected}"
    idx = np.array([2, 0, 3, 1])
    assert np.allclose(benjamini_hochberg(p[idx]), expected[idx], atol=1e-4), \
        "BH self-test failed: result is order-dependent"
    assert np.all(got >= p - 1e-12), "BH self-test failed: adjusted p < raw p"

_bh_selftest()


def compute_stats(df_h, df_p, metrics=METRIC_BATTERY):
    rows = []
    for m in metrics:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        h = df_h[m].dropna().values
        p = df_p[m].dropna().values
        r, pval = rank_biserial(h, p)
        if np.isnan(r):
            continue
        rows.append({
            "metric": m,
            "label": METRIC_LABELS.get(m, m),
            "domain": METRIC_DOMAIN.get(m, "HRV"),
            "family": _metric_family(m),
            "h_mean": np.mean(h), "h_sd": np.std(h, ddof=1),
            "p_mean": np.mean(p), "p_sd": np.std(p, ddof=1),
            "r": r, "p_raw": pval,
        })
    res = pd.DataFrame(rows).set_index("metric")
    res["p_adj"] = np.nan
    for fam in res["family"].unique():
        if fam == "none":
            continue
        mask = res["family"] == fam
        res.loc[mask, "p_adj"] = benjamini_hochberg(res.loc[mask, "p_raw"].values)
    res["sig"] = res["p_adj"] < 0.05

    fam_sizes = res["family"].value_counts().to_dict()
    _expected_primary = len([m for m in METRIC_BATTERY
                             if _metric_family(m) == "primary"])
    for name, expected in (("primary", _expected_primary),
                           ("MFDFA", 3), ("RNA", 2)):
        got = fam_sizes.get(name, 0)
        if got and got != expected:
            raise AssertionError(
                f"FDR family '{name}' has {got} members, expected {expected}. "
                f"figures.py and comparison.py must agree or the figure will "
                f"contradict the table.")
    return res

PRINT_WIDTH_IN = 7.08
PRINT_RC = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}


def _save(fig, name, tiff=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exts = ("svg", "png", "tif") if tiff else ("svg", "png")
    for ext in exts:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        if ext == "tif":
            fig.savefig(path, dpi=300, facecolor="white",
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path)
    if tiff:
        from PIL import Image
        with Image.open(os.path.join(OUTPUT_DIR, f"{name}.tif")) as im:
            w_mm = im.size[0] / 300 * 25.4
        print(f"  saved → {os.path.join(OUTPUT_DIR, name)}.(svg|png|tif); "
              f"printed width {w_mm:.0f} mm at 300 dpi")
    else:
        print(f"  saved → {os.path.join(OUTPUT_DIR, name)}.(svg|png)")
    plt.close(fig)


def figure1_effect_sizes(stats, bootstrap_csv=BOOTSTRAP_CSV):
    ci_lo = {}; ci_hi = {}
    if os.path.exists(bootstrap_csv):
        boot = pd.read_csv(bootstrap_csv, index_col=0)
        for lbl, row in boot.iterrows():
            col = LABEL_TO_COL.get(lbl, lbl)
            ci_lo[col] = row.get("r_CI_low", np.nan)
            ci_hi[col] = row.get("r_CI_high", np.nan)
    else:
        print(f"  [!] bootstrap CSV not found ({bootstrap_csv}); "
              "drawing bars without CIs.")

    df = stats.drop(index=[m for m in DETERMINISTIC_CHECKS if m in stats.index]).copy()
    df["abs_r"] = df["r"].abs()
    df = df.sort_values("abs_r", ascending=True)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.6, 9.8))

    for i, (col, row) in enumerate(df.iterrows()):
        domain = row["domain"]
        base = DOMAIN_COLORS[domain]
        sig = row["sig"]
        unfamilied = row["family"] == "none"
        if unfamilied:
            ax.barh(i, row["r"], height=0.66, facecolor="white",
                    edgecolor="#777777", linewidth=0.8, zorder=2)
        else:
            color = base if sig else C_NONSIG
            ax.barh(i, row["r"], height=0.66,
                    color=color, edgecolor="white", linewidth=0.6,
                    zorder=2, alpha=1.0 if sig else 0.55,
                    hatch=None if sig else "////")
        lo = ci_lo.get(col, np.nan); hi = ci_hi.get(col, np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color="#333333", lw=1.1, zorder=3)
            ax.plot([lo, lo], [i - 0.13, i + 0.13], color="#333333", lw=1.1, zorder=3)
            ax.plot([hi, hi], [i - 0.13, i + 0.13], color="#333333", lw=1.1, zorder=3)
        end = row["r"]
        if np.isfinite(lo) and np.isfinite(hi):
            end = max(row["r"], hi) if row["r"] >= 0 else min(row["r"], lo)
        xt = end + (0.02 if row["r"] >= 0 else -0.02)
        ax.text(xt, i, f"{row['r']:+.2f}".replace("-", "\u2212"),
                va="center", ha="left" if row["r"] >= 0 else "right",
                fontsize=7.2, color="#222222")

    ax.axvline(0, color="black", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlim(-1.12, 1.12)
    ax.set_xlabel("Rank-biserial correlation  r   "
                  "(negative = lower in patients;  positive = higher in patients)")
    ax.set_title("Effect sizes across the full metric battery\n(HRV-ACC controls vs patients)",
                 fontsize=10, pad=10)
    ax.grid(axis="x", color=C_GRID, lw=0.8, zorder=0)

    shown = [d for d in ["HRV", "Scaling", "Entropy", "RQA"]
             if ((df["domain"] == d) & df["sig"] & (df["family"] != "none")).any()]
    domain_handles = [Patch(facecolor=DOMAIN_COLORS[d], edgecolor="white",
                            label=DOMAIN_DISPLAY.get(d, d)) for d in shown]
    nonsig_handle = Patch(facecolor=C_NONSIG, alpha=0.55, hatch="////",
                          edgecolor="white", label="not significant (BH within family)")
    extra = []
    if (df["family"] == "none").any():
        extra = [Patch(facecolor="white", edgecolor="#777777",
                       label="outside the correction families")]
    leg = ax.legend(handles=domain_handles + [nonsig_handle] + extra,
                    loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=2,
                    title="Significant after correction, by domain", fontsize=7.6,
                    title_fontsize=8)
    leg._legend_box.align = "left"

    ax.text(0.99, 0.01,
            "Error bars: bootstrap 95% CI (2,000 resamples; primary family only)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.8, color="#666666", style="italic")

    fig.tight_layout()
    _save(fig, "Figure1_EffectSizes")


def figure2_boxplots(df_h, df_p, stats):
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    axes = axes.flatten()
    rng = np.random.default_rng(7)

    for ax, m in zip(axes, FIG2_METRICS):
        h = df_h[m].dropna().values
        p = df_p[m].dropna().values

        bp = ax.boxplot([h, p], positions=[1, 2], widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.8),
                        whiskerprops=dict(lw=1.1, color="#555555"),
                        capprops=dict(lw=1.1, color="#555555"),
                        boxprops=dict(lw=0.8))
        bp["boxes"][0].set_facecolor(C_HEALTHY); bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][1].set_facecolor(C_PSYCH);   bp["boxes"][1].set_alpha(0.35)
        bp["boxes"][0].set_edgecolor(C_HEALTHY)
        bp["boxes"][1].set_edgecolor(C_PSYCH)

        for vals, pos, col in [(h, 1, C_HEALTHY), (p, 2, C_PSYCH)]:
            jit = rng.uniform(-0.16, 0.16, len(vals))
            ax.scatter(pos + jit, vals, s=22, color=col, alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)

        if m in stats.index:
            row = stats.loc[m]
        else:
            hv = df_h[m].dropna().values
            pv = df_p[m].dropna().values
            r_, praw_ = rank_biserial(hv, pv)
            row = pd.Series({
                "metric": m, "label": METRIC_LABELS.get(m, m),
                "domain": METRIC_DOMAIN.get(m, "Scaling"),
                "family": "sensitivity (not FDR-corrected)",
                "h_mean": np.mean(hv), "h_sd": np.std(hv, ddof=1),
                "p_mean": np.mean(pv), "p_sd": np.std(pv, ddof=1),
                "r": r_, "p_raw": praw_, "p_adj": np.nan, "sig": False,
            })
        ax.set_title(METRIC_LABELS.get(m, m), fontsize=9.5, fontweight="bold", pad=20)
        ax.annotate(FIG2_DOMAIN_TAG.get(m, ""), xy=(0.5, 1.0), xytext=(0, 5),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.3, style="italic",
                    color=DOMAIN_COLORS.get(METRIC_DOMAIN.get(m, "HRV"), "#555555"))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Control", "Patient"], fontsize=8.5)
        ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
        ax.margins(y=0.16)

        ymax = max(np.max(h), np.max(p))
        ymin = min(np.min(h), np.min(p))
        rng_y = ymax - ymin if ymax > ymin else 1.0
        ybr = ymax + rng_y * 0.10
        ax.plot([1, 1, 2, 2],
                [ybr, ybr + rng_y * 0.03, ybr + rng_y * 0.03, ybr],
                lw=1.0, color="#333333")
        if pd.isna(row["p_adj"]):
            p_txt = ("p < 0.001 (uncorr.)" if row["p_raw"] < 0.001
                     else f"p = {row['p_raw']:.3f} (uncorr.)")
        else:
            p_txt = ("p < 0.001" if row["p_adj"] < 0.001
                     else f"p = {row['p_adj']:.3f}")
        star = "  *" if row["sig"] else ""
        ax.text(1.5, ybr + rng_y * 0.05, f"{p_txt}{star}",
                ha="center", va="bottom", fontsize=8, color="#333333")
        ax.text(0.04, 0.96, f"r = {row['r']:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="#555555",
                fontweight="bold")

    fig.suptitle("Group comparisons across analytic domains: "
                 "controls (blue) and medicated inpatients (red)",
                 fontsize=11, y=1.005)
    handles = [Patch(facecolor=C_HEALTHY, alpha=0.5, label="Control (N=30)"),
               Patch(facecolor=C_PSYCH, alpha=0.5, label="Patient (N=30)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    _save(fig, "Figure2_BoxPlots")


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


def _load_rr_windowed(record_id):
    path = _rr_path_for(record_id)
    if path is None or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    if rr_col is None:
        return None
    rr_ms = pd.to_numeric(df[rr_col], errors="coerce").dropna().values
    if len(rr_ms) < 50:
        return None
    rr_s = rr_ms / 1000.0
    cum = np.cumsum(rr_s)
    mask = (cum >= RR_SKIP_SECONDS) & (cum < RR_SKIP_SECONDS + RR_ANALYSIS_SECONDS)
    rr_win = rr_s[mask]
    t_win = cum[mask]
    if len(rr_win) < 50:
        return None
    return rr_win, t_win


def _subject_spectrum(record_id, reject, freqdom):
    loaded = _load_rr_windowed(record_id)
    if loaded is None:
        return None
    rr_win, t_win = loaded
    _out = reject(rr_win, verbose=False)
    rr_clean = _out[0]
    if len(rr_clean) < 50:
        return None
    t_clean = np.concatenate([[t_win[0]], t_win[0] + np.cumsum(rr_clean[:-1])])
    try:
        _metrics, freqs, power = freqdom(rr_clean, t_clean)
    except Exception:
        return None
    if freqs is None or power is None or len(freqs) < 5:
        return None
    return np.asarray(freqs, float), np.asarray(power, float) * 1e6


def _parse_mean_sd(s):
    try:
        a, b = str(s).replace("\u00b1", "|").split("|")
        return float(a), float(b)
    except Exception:
        return np.nan, np.nan


def _fitted_bins(band):
    from pipeline import INTERP_FS, WELCH_NPERSEG
    step = INTERP_FS / WELCH_NPERSEG
    k_lo = max(1, int(np.ceil(band[0] / step - 1e-9)))
    k_hi = int(np.floor(band[1] / step + 1e-9))
    return k_lo * step, k_hi * step


def _band_short_label(variant):
    from pipeline import BETA_FIT_BAND, BETA_FIT_BAND_WIDE, BETA_FIT_BAND_FULL
    v = str(variant)
    for key, name, band in (("Primary", "Primary", BETA_FIT_BAND),
                            ("Wide", "Wide", BETA_FIT_BAND_WIDE),
                            ("Full", "Full", BETA_FIT_BAND_FULL),
                            ("Linear", "Linear\ninterp.", BETA_FIT_BAND)):
        if key in v:
            lo, hi = _fitted_bins(band)
            return f"{name}\n{lo:.3f}\u2013\n{hi:.3f}"
    return v.split("(")[0].strip()


def figure3_psd_curves(df_h, df_p, band_csv=BAND_SENS_CSV):
    fit_band = (0.01, 0.15)

    spec_h, spec_p = {}, {}
    try:
        from pipeline import reject_ectopic_beats, compute_frequency_domain

        def group_spectra(df):
            specs = {}
            for sid in list(df.index):
                out = _subject_spectrum(sid, reject_ectopic_beats,
                                        compute_frequency_domain)
                if out is not None:
                    specs[sid] = out
            return specs

        spec_h = group_spectra(df_h)
        spec_p = group_spectra(df_p)
    except Exception as exc:                       # noqa: BLE001
        print(f"  [!] Figure 3A: raw spectra unavailable ({exc}).")

    _rc = plt.rc_context(PRINT_RC)
    _rc.__enter__()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(PRINT_WIDTH_IN, 3.7),
                                  gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.32})

    X_MAX_HZ = 0.5
    visible_min = []

    def plot_group(df, specs, color, lab):
        if not specs:
            return
        fref = None
        for sid in specs:
            f0, _ = specs[sid]
            if fref is None or len(f0) > len(fref):
                fref = f0
        fref = fref[fref > 0]
        stacked = []
        for sid, (f0, p0) in specs.items():
            good = (f0 > 0) & np.isfinite(p0) & (p0 > 0)
            if good.sum() < 5:
                continue
            pin = np.interp(fref, f0[good], p0[good], left=np.nan, right=np.nan)
            stacked.append(pin)
        if not stacked:
            return
        stacked = np.array(stacked)
        beta_med = float(np.nanmedian(df["PSD_Beta"])) if "PSD_Beta" in df else np.nan
        for sid, (f0, p0) in specs.items():
            good = (f0 > 0) & np.isfinite(p0) & (p0 > 0)
            ax.loglog(f0[good], p0[good], color=color, lw=0.4, alpha=0.16, zorder=1)
            shown = good & (f0 <= X_MAX_HZ)
            if shown.any():
                visible_min.append(float(p0[shown].min()))
        mean_curve = np.nanmean(stacked, axis=0)
        ok = np.isfinite(mean_curve) & (mean_curve > 0)
        blab = (f"{lab}, median uncorrected \u03b2 {beta_med:.2f}"
                if np.isfinite(beta_med) else lab)
        ax.loglog(fref[ok], mean_curve[ok], color=color, lw=1.8, zorder=4, label=blab)
        if np.isfinite(beta_med):
            bmask = (fref >= fit_band[0]) & (fref <= fit_band[1]) & ok
            if bmask.sum() >= 3:
                f_anchor = np.sqrt(fit_band[0] * fit_band[1])
                p_anchor = np.interp(f_anchor, fref[bmask], mean_curve[bmask])
                fline = np.array(fit_band)
                pline = p_anchor * (fline / f_anchor) ** (-beta_med)
                ax.loglog(fline, pline, color=color, lw=1.0, ls="--", alpha=0.9,
                          zorder=5)

    if spec_h or spec_p:
        plot_group(df_h, spec_h, C_HEALTHY, "Controls")
        plot_group(df_p, spec_p, C_PSYCH, "Patients")
        from pipeline import BETA_FIT_BAND as _fb
        _flo, _fhi = _fitted_bins(_fb)
        ax.axvspan(_flo, _fhi, color="#FFE9A8", alpha=0.45, zorder=0,
                   label=f"Fitted bins ({_flo:.3f}\u2013{_fhi:.3f} Hz)")
        try:
            from pipeline import MAYER_BAND as _mb
        except Exception:                              # noqa: BLE001
            _mb = (0.07, 0.13)
        ax.axvspan(_mb[0], _mb[1], facecolor="none", edgecolor="#9A9A9A", hatch="////",
                   lw=0.0, zorder=0.5,
                   label=f"Excised ({_mb[0]:.2f}\u2013{_mb[1]:.2f} Hz)")
        ax.set_xlim(0.006, X_MAX_HZ)
        if visible_min:
            ax.set_ylim(bottom=10 ** np.floor(np.log10(min(visible_min))))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power spectral density (ms\u00b2/Hz)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False,
                  handlelength=1.6, labelspacing=0.3, ncol=1)
        ax.grid(True, which="major", color=C_GRID, lw=0.6)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "raw R-R series not available\n(panel A skipped)",
                ha="center", va="center", fontsize=9, color="#888888")
    ax.set_title("(A) R-R power spectra by group", pad=6)

    if band_csv and os.path.exists(band_csv):
        bs = pd.read_csv(band_csv).reset_index(drop=True)
        label_tops = []
        for i, row in bs.iterrows():
            hm, hs = _parse_mean_sd(row["H_Mean_SD"])
            pm, ps = _parse_mean_sd(row["P_Mean_SD"])
            lab_h = "Control" if i == 0 else None
            lab_p = "Patient" if i == 0 else None
            if "Full" in str(row["Variant"]):
                ax2.axvspan(i - 0.45, i + 0.45, color="#F0F0F0", zorder=0)
            ax2.errorbar(i - 0.14, hm, yerr=hs, fmt="o", color=C_HEALTHY,
                         capsize=2.5, ms=4.5, lw=1.1, label=lab_h, zorder=3)
            ax2.errorbar(i + 0.14, pm, yerr=ps, fmt="s", color=C_PSYCH,
                         capsize=2.5, ms=4.5, lw=1.1, label=lab_p, zorder=3)
            try:
                r = float(row["r_effect"])
                top = np.nanmax([hm + hs, pm + ps]) + 0.06
                label_tops.append(top)
                ax2.text(i, top, f"r = {r:+.2f}".replace("-", "\u2212"), ha="center",
                         va="bottom", fontsize=8,
                         fontweight=("bold" if abs(r) > 0.3 else "normal"),
                         color=("#222222" if r > 0 else "#777777"))
            except Exception:
                pass
        ax2.set_xticks(range(len(bs)))
        ax2.set_xticklabels([_band_short_label(v) for v in bs["Variant"]], fontsize=7.5)
        ax2.set_xlabel("Fitted range (Hz)")
        ax2.set_xlim(-0.75, len(bs) - 0.25)
        ax2.set_ylabel("Uncorrected slope \u03b2 (mean \u00b1 SD)")
        if label_tops:
            lo_y, hi_y = ax2.get_ylim()
            ax2.set_ylim(lo_y, max(hi_y, max(label_tops) + 0.14))
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33),
                   ncol=2, frameon=False)
        ax2.grid(axis="y", color=C_GRID, lw=0.8)
        ax2.set_title("(B) Uncorrected \u03b2 by fitting band", pad=6)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "band-sensitivity table\nnot available (panel B skipped)",
                 ha="center", va="center", fontsize=9, color="#888888")

    _save(fig, "Figure3_PSD_Curves", tiff=True)
    _rc.__exit__(None, None, None)


def _ols_residual(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - A @ beta


def _partial_r_resid(beta, grp, cov):
    if cov is None or cov.shape[1] == 0:
        return spearmanr(beta, grp)
    rc = np.column_stack([rankdata(cov[:, j]) for j in range(cov.shape[1])])
    by = _ols_residual(rankdata(beta), rc)
    gy = _ols_residual(rankdata(grp), rc)
    r = float(np.corrcoef(by, gy)[0, 1])
    dof = len(by) - 2 - rc.shape[1]
    p = float(2.0 * t_dist.sf(abs(r * np.sqrt(dof / (1.0 - r ** 2))), dof))
    return r, p


def figure4_covariate(df_h, df_p, covariate_csv=COVARIATE_CSV,
                      activity_csv=ACTIVITY_COV_CSV):
    need = ["PSD_Beta", "HRV_MeanHR", "HRV_RMSSD", "HF_Power_ms2"]
    have = [c for c in need if c in df_h.columns and c in df_p.columns]
    dh = df_h[have].dropna().copy(); dh["g"] = 0
    dp = df_p[have].dropna().copy(); dp["g"] = 1
    comb = pd.concat([dh, dp], ignore_index=True)
    beta = comb["PSD_Beta"].to_numpy(float)
    grp = comb["g"].to_numpy(float)

    def col(name):
        return comb[name].to_numpy(float).reshape(-1, 1) if name in comb else None

    activity_lookup = {}
    if activity_csv and os.path.exists(activity_csv):
        a = pd.read_csv(activity_csv)
        for _, row in a.iterrows():
            activity_lookup[str(row["Analysis"]).strip()] = (
                row.get("partial_r", np.nan), row.get("p", np.nan))

    forest = []

    def from_data(label, cov_names):
        cols = [col(c) for c in cov_names]
        if any(c is None for c in cols):
            return
        cov = np.hstack(cols) if cols else None
        r, p = _partial_r_resid(beta, grp, cov)
        forest.append((label, float(r), float(p)))

    def from_activity(label, key):
        if key in activity_lookup:
            r, p = activity_lookup[key]
            if np.isfinite(r):
                forest.append((label, float(r), float(p)))

    from_data("Unadjusted", [])
    from_data("| Heart rate", ["HRV_MeanHR"])
    from_data("| RMSSD", ["HRV_RMSSD"])
    from_data("| HF power", ["HF_Power_ms2"])
    from_activity("| Mobility (accelerometer)", "PSD_Beta ~ Group | Mobility")
    from_data("| HF power + heart rate", ["HF_Power_ms2", "HRV_MeanHR"])
    from_activity("| Mobility + HF + heart rate",
                  "PSD_Beta ~ Group | Mobility + HF power + MeanHR")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7))

    ax = axes[0]
    rng = np.random.default_rng(3)
    for g, cc, lab in [(0, C_HEALTHY, "Control"), (1, C_PSYCH, "Patient")]:
        mask = grp == g
        jit = rng.uniform(-0.13, 0.13, int(mask.sum()))
        ax.scatter(g + jit, beta[mask], s=30, color=cc, alpha=0.75,
                   edgecolor="white", linewidth=0.4, label=lab)
        ax.hlines(beta[mask].mean(), g - 0.25, g + 0.25, color=cc, lw=2.4)
    r_raw, _ = spearmanr(beta, grp)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Patient"])
    ax.set_ylabel("Uncorrected slope \u03b2")
    ax.set_title("(A) Unadjusted\nSpearman r = %+.2f, p < 0.001" % r_raw,
                 fontsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    labels = [f[0] for f in forest]
    rs = [f[1] for f in forest]
    ps = [f[2] for f in forest]
    ys = np.arange(len(forest))[::-1]
    for y, r, p, lab in zip(ys, rs, ps, labels):
        full = lab.startswith("| Activity + HF")
        color = "#222222" if full else C_PSYCH
        ax.plot([0, r], [y, y], color=color, lw=1.4, alpha=0.5, zorder=1)
        ax.scatter([r], [y], s=(90 if full else 55),
                   color=color, edgecolor="white", linewidth=0.6, zorder=3,
                   marker=("D" if full else "o"))
        ax.text(r + 0.02, y, "%+.2f%s" % (r, "" if p < 0.05 else " (ns)"),
                va="center", ha="left", fontsize=8,
                fontweight=("bold" if full else "normal"))
    ax.axvline(0, color="#999999", lw=1.0)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.15, 0.85)
    ax.set_xlabel("Partial Spearman r  (uncorrected slope \u03b2 vs group)")
    ax.set_title("(B) Partial correlation under each adjustment",
                 fontsize=9)
    ax.grid(axis="x", color=C_GRID, lw=0.8)

    ax = axes[2]
    if "HF_Power_ms2" in comb:
        hf = col("HF_Power_ms2")
        rhf = rankdata(hf).reshape(-1, 1)
        by = _ols_residual(rankdata(beta), rhf)
        r_hf, p_hf = _partial_r_resid(beta, grp, hf.reshape(-1, 1))
        rng_j = np.random.default_rng(0)
        for gi, colr in ((0, C_HEALTHY), (1, C_PSYCH)):
            v = by[grp == gi]
            ax.scatter(gi + rng_j.uniform(-0.12, 0.12, len(v)), v, s=30, color=colr,
                       alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.plot([gi - 0.25, gi + 0.25], [v.mean(), v.mean()], color="#333333", lw=1.6)
        ax.axhline(0, color="#BBBBBB", lw=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Patient"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("Residual rank of slope \u03b2 | rank of HF power")
        ax.set_title("(C) After HF power\npartial r = %+.2f, p = %.3f"
                     % (r_hf, p_hf), fontsize=9)
        ax.grid(color=C_GRID, lw=0.8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "HF power column\nnot available",
                ha="center", va="center", fontsize=9)

    fig.suptitle("Uncorrected spectral slope and group under covariate adjustment",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    _save(fig, "Figure4_Covariate")


def figure5_surrogates(df_h, df_p, surrogate_csv=SURROGATE_CSV):
    surr_cols = {
        "SURR_SampEn_Z": "Sample Entropy",
        "SURR_DFA_Alpha1_Z": "DFA \u03b1\u2081",
        "SURR_MSE_Mean_Z": "MSE Mean",
    }
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.28)

    axA = fig.add_subplot(gs[0])
    metrics = list(surr_cols.keys())
    rng = np.random.default_rng(11)
    positions = []
    for i, mcol in enumerate(metrics):
        base = i * 3
        for j, (df, col, lab) in enumerate([(df_h, C_HEALTHY, "Control"),
                                            (df_p, C_PSYCH, "Patient")]):
            if mcol not in df.columns:
                continue
            z = df[mcol].dropna().values
            pos = base + j
            positions.append(pos)
            vp = axA.violinplot([z], positions=[pos], widths=0.8,
                                showmeans=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(col); body.set_alpha(0.35)
                body.set_edgecolor(col); body.set_linewidth(0.8)
            jit = rng.uniform(-0.16, 0.16, len(z))
            axA.scatter(pos + jit, z, s=14, color=col, alpha=0.7,
                        edgecolor="white", linewidth=0.3, zorder=3)

    axA.axhline(1.96, color="#888888", lw=1.0, ls="--")
    axA.axhline(-1.96, color="#888888", lw=1.0, ls="--")
    axA.axhline(0, color="#CCCCCC", lw=0.7)
    axA.text(axA.get_xlim()[1], 1.96, "  +1.96", va="center", ha="left",
             fontsize=7, color="#888888")
    axA.text(axA.get_xlim()[1], -1.96, "  \u22121.96", va="center", ha="left",
             fontsize=7, color="#888888")
    axA.set_xticks([i * 3 + 0.5 for i in range(len(metrics))])
    axA.set_xticklabels(list(surr_cols.values()), fontsize=8.5)
    axA.set_ylabel("Surrogate Z-score  (observed vs IAAFT null)")
    axA.set_title("(A) Nonlinearity Z-scores per subject", fontsize=9.5)
    axA.grid(axis="y", color=C_GRID, lw=0.8)
    handles = [Patch(facecolor=C_HEALTHY, alpha=0.5, label="Control"),
               Patch(facecolor=C_PSYCH, alpha=0.5, label="Patient")]
    axA.legend(handles=handles, fontsize=7.8, loc="upper left")

    axB = fig.add_subplot(gs[1])
    if os.path.exists(surrogate_csv):
        surr = pd.read_csv(surrogate_csv)
        labels = surr["Metric"].tolist()
        h_pct = surr["Healthy_Pct"].values
        p_pct = surr["Psych_Pct"].values
    else:
        labels, h_pct, p_pct = [], [], []
        for mcol, lab in [("SURR_SampEn_Sig", "Sample Entropy"),
                          ("SURR_DFA_Alpha1_Sig", "DFA \u03b1\u2081"),
                          ("SURR_MSE_Mean_Sig", "MSE Mean")]:
            if mcol in df_h.columns:
                labels.append(lab)
                h_pct.append(100 * df_h[mcol].astype(str).str.lower()
                             .isin(["true", "1", "1.0"]).mean())
                p_pct.append(100 * df_p[mcol].astype(str).str.lower()
                             .isin(["true", "1", "1.0"]).mean())
        h_pct = np.array(h_pct); p_pct = np.array(p_pct)

    x = np.arange(len(labels)); w = 0.38
    axB.bar(x - w / 2, h_pct, w, color=C_HEALTHY, alpha=0.85, label="Control")
    axB.bar(x + w / 2, p_pct, w, color=C_PSYCH, alpha=0.85, label="Patient")
    for xi, v in zip(x - w / 2, h_pct):
        axB.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, p_pct):
        axB.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylabel("% subjects with significant nonlinearity")
    axB.set_ylim(0, 100)
    axB.set_title("(B) Fraction significant (Fisher's exact: n.s.)", fontsize=9.5)
    axB.grid(axis="y", color=C_GRID, lw=0.8)
    axB.legend(fontsize=7.8, loc="upper right")

    fig.suptitle("IAAFT surrogate tests per subject", fontsize=10.5, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "Figure5_Surrogates")


def figureS_mse_curves(df_h, df_p):
    cols = [c for c in df_h.columns if c.startswith("MSE_Curve_Tau")]
    if not cols:
        print("  figureS_mse_curves skipped: MSE_Curve_Tau* columns absent "
              "(re-run the pipeline).")
        return
    cols = sorted(cols, key=lambda c: int(c.replace("MSE_Curve_Tau", "")))
    taus = [int(c.replace("MSE_Curve_Tau", "")) for c in cols]

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(7.4, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    for df, lab, col in ((df_h, "Control", C_HEALTHY),
                         (df_p, "Patient", C_PSYCH)):
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        m = vals.mean(axis=0).values
        sd = vals.std(axis=0, ddof=1).values
        n = vals.notna().sum(axis=0).values
        se = np.divide(sd, np.sqrt(np.maximum(n, 1)),
                       out=np.full_like(sd, np.nan), where=n > 0)
        t = [taus[cols.index(c)] for c in present]
        ax.plot(t, m, "-o", ms=3.4, lw=1.6, color=col, label=lab)
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
        axn.plot(t, n, "-", lw=1.4, color=col)

    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, lw=0.8)
    ax.set_ylabel("Sample entropy")
    ax.set_title("Multiscale entropy by coarse-graining scale\n"
                 "(mean \u00b1 SE; see lower panel for contributing subjects)")
    ax.legend(frameon=False, loc="best")

    axn.set_axisbelow(True)
    axn.grid(color=C_GRID, lw=0.8)
    axn.set_xlabel("Scale factor \u03c4 (beats per coarse-grained sample)")
    from matplotlib.ticker import MaxNLocator
    axn.xaxis.set_major_locator(MaxNLocator(integer=True))
    axn.set_ylabel("n subjects")
    axn.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, "FigureS_MSE_Curves")


def figure6_cross_validation(df_mitbih, df_hrvacc, df_psych):
    groups = [("MIT-BIH\ncontrols", df_mitbih, C_MITBIH, "D"),
              ("HRV-ACC\ncontrols", df_hrvacc, C_HEALTHY, "o"),
              ("HRV-ACC\npatients", df_psych, C_PSYCH, "^")]
    panels = [("PSD_Beta_NoMayer", "(A) Spectral slope, baroreflex band excised",
               "Slope \u03b2 (0.07\u20130.13 Hz excised)", "{:.2f}"),
              ("HRV_RMSSD", "(B) RMSSD", "RMSSD (ms)", "{:.1f}")]
    rng = np.random.default_rng(20261018)

    _rc = plt.rc_context(PRINT_RC)
    _rc.__enter__()
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_WIDTH_IN, 3.3),
                             gridspec_kw={"wspace": 0.3})
    for ax, (mcol, title, ylab, fmt) in zip(axes, panels):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        labs = []
        for i, (lab, df, col, mk) in enumerate(groups):
            v = pd.to_numeric(df[mcol], errors="coerce").dropna().values
            x = i + rng.uniform(-0.16, 0.16, size=len(v))
            ax.scatter(x, v, s=18, marker=mk, color=col, edgecolor="white",
                       linewidth=0.5, zorder=3)
            med = float(np.median(v))
            ax.plot([i - 0.28, i + 0.28], [med, med], color="#222222", lw=1.5,
                    solid_capstyle="round", zorder=4)
            ax.text(i + 0.31, med, fmt.format(med), va="center", ha="left",
                    fontsize=8, color="#222222")
            labs.append(f"{lab}\n(n = {len(v)})")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labs)
        ax.set_xlim(-0.5, len(groups) - 0.35)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    _save(fig, "Figure8_CrossValidation", tiff=True)
    _rc.__exit__(None, None, None)


def figure7_intercorrelation(df_h, df_p, stats):
    sig = stats.index[stats["sig"] & (stats["family"] == "primary")].tolist()
    domain_order = {"HRV": 0, "Scaling": 1, "Entropy": 2, "RQA": 3}
    sig = sorted(sig, key=lambda m: (domain_order.get(METRIC_DOMAIN.get(m), 9),
                                     -abs(stats.loc[m, "r"])))
    labels = [METRIC_LABELS.get(m, m) for m in sig]

    h_corr = df_h[sig].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    p_corr = df_p[sig].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

    n = len(sig)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6), gridspec_kw={"wspace": 0.42})
    for ax, corr, title, _n in [
            (axes[0], h_corr, f"Control (N={len(df_h)})", len(df_h)),
            (axes[1], p_corr, f"Patient (N={len(df_p)})", len(df_p))]:
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45,
                                                    ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.4,
                            color="white" if abs(v) > 0.55 else "#333333")
        bounds = []
        last = None
        for k, m in enumerate(sig):
            d = METRIC_DOMAIN.get(m)
            if last is not None and d != last:
                bounds.append(k - 0.5)
            last = d
        for b in bounds:
            ax.axhline(b, color="black", lw=1.2)
            ax.axvline(b, color="black", lw=1.2)

    cbar = fig.colorbar(im, ax=axes, shrink=0.65, pad=0.02)
    cbar.set_label("Spearman r", fontsize=9)
    fig.suptitle("Metric intercorrelation structure across domains "
                 "(black lines mark domain blocks)",
                 fontsize=10.5, y=1.0)
    _save(fig, "Figure7_Intercorrelation")


def figure8_conceptual_synthesis():
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import Patch as MPatch

    PAD = 0.11

    def rbox(ax, cx, cy, w, h, color, text, fs=9.0, tc="white"):
        p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle=f"round,pad={PAD}",
                           facecolor=color, edgecolor="white",
                           linewidth=1.5, zorder=3)
        ax.add_patch(p)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold",
                zorder=4, multialignment="center")

    def arrow(ax, x1, y1, x2, y2, c="#444444"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.8,
                                    connectionstyle="arc3,rad=0"), zorder=2)

    def angled_arrow(ax, x1, y1, x2, y2, c="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.6,
                                    connectionstyle="angle,angleA=0,angleB=90"),
                    zorder=2)

    def note(ax, x, y, t, c="#666666", fs=7.5):
        ax.text(x, y, t, ha="left", va="center",
                fontsize=fs, color=c, style="italic", zorder=5)

    fig, ax = plt.subplots(figsize=(10.5, 12.5))
    ax.set_xlim(-0.15, 10.65); ax.set_ylim(2.4, 14.7)
    ax.axis("off")

    rbox(ax, 5.25, 13.35, 8.0, 0.88,
         "#1F3864",
         "Psychotic-spectrum illness\n"
         "(schizophrenia / bipolar disorder — medicated inpatients)", fs=9.2)
    arrow(ax, 5.25, 12.91, 5.25, 12.18)
    note(ax, 6.05, 12.54,
         "Impaired prefrontal-autonomic\nnetwork modulation", fs=7.5)

    rbox(ax, 5.25, 11.82, 7.5, 0.72,
         "#1F7A8C",
         "Altered neurovisceral integration\n"
         "(prefrontal cortex  →  central ANS  →  sinoatrial node)", fs=8.8)
    arrow(ax, 5.25, 11.45, 5.25, 10.72)
    note(ax, 6.05, 11.09,
         "Reduced adaptive range of\nautonomic cardiac output", fs=7.5)

    rbox(ax, 5.25, 10.36, 7.2, 0.68,
         "#2A7F4F",
         "Altered RR interval temporal organization\n"
         "(HRV-ACC dataset; 30-min Polar H10; N = 30 + 30)", fs=8.8)
    arrow(ax, 5.25, 10.01, 5.25, 9.38)
    note(ax, 6.05, 9.70,
         "Four complementary analytic\nlenses applied to the same series", fs=7.5)

    domain_cfg = [
        (1.18, "Variability\n(RMSSD, SDNN, power)",       "probes: autonomic range",       "#0077BB"),
        (3.70, "Scaling\n(PSD \u03b2, DFA \u03b1\u2082, MFDFA)",     "probes: long-range correlation", "#EE7733"),
        (6.22, "Complexity\n(ApEn, SampEn, PermEn)",      "probes: signal predictability", "#009988"),
        (8.73, "Recurrence\n(RQA DET, RNA network)",      "probes: recurrence structure",  "#AA4499"),
    ]
    for xd, lbl, probe, col in domain_cfg:
        rbox(ax, xd, 8.60, 2.36, 1.42, col, lbl + "\n" + probe, fs=7.3)
        arrow(ax, xd, 7.89, xd, 7.58)

    rbox(ax, 5.25, 7.22, 10.10, 0.66,
         "#264653",
         "Convergent multi-domain alteration in autonomic scaling organization\n"
         "(5 partially independent dimensions; all significant after BH-FDR correction)",
         fs=8.6)
    arrow(ax, 5.25, 6.88, 5.25, 6.30)
    ax.text(5.25, 6.59,
            "Two mechanistically coherent interpretations; "
            "current data cannot distinguish",
            ha="center", va="center", fontsize=7.2, color="#666666",
            style="italic", zorder=5)

    rbox(ax, 2.55, 5.96, 4.60, 0.72,
         "#5C4033",
         "Criticality-based interpretation\n"
         "(consistent with altered adaptive\nmultiscale scaling regime)", fs=7.5)
    rbox(ax, 7.95, 5.96, 4.60, 0.72,
         "#777777",
         "Alternative mechanisms\n(altered feedback gain;\nchanged timescales;\nreduced network complexity)",
         fs=7.5)
    angled_arrow(ax, 5.25, 6.88, 2.55, 6.33)
    angled_arrow(ax, 5.25, 6.88, 7.95, 6.33)

    arrow(ax, 2.55, 5.59, 2.55, 5.02)
    arrow(ax, 7.95, 5.59, 7.95, 5.02)

    rbox(ax, 2.55, 4.67, 4.60, 0.68,
         "#CC3311",
         "PSD \u03b2: device-robust candidate\nbiomarker; complements RMSSD\n"
         "(partial r = +0.32 after RMSSD control)", fs=7.5)
    rbox(ax, 7.95, 4.67, 4.60, 0.68,
         "#BB5500",
         "Critical caveats:\nmedicated inpatients; no\nlongitudinal validation;\nno symptom-level data",
         fs=7.5)

    arrow(ax, 5.25, 4.33, 5.25, 3.72)

    rbox(ax, 5.25, 3.35, 10.10, 0.66,
         "#1B4332",
         "Priority next steps:  "
         "(1) Simultaneous ECG-EEG   "
         "(2) Longitudinal study + PANSS   "
         "(3) MFDFA / RNA reliability testing",
         fs=8.4)

    ax.text(5.25, 2.75,
            "Power-law scaling is consistent with, but not sufficient to confirm, "
            "critical-point operation.\n"
            "The criticality framework is one theoretical lens alongside "
            "alternative dynamical explanations.",
            ha="center", va="center", fontsize=7.2, color="#555555",
            style="italic", zorder=5)

    patches = [MPatch(facecolor=col, edgecolor="white",
                      label=lbl.split("\n")[0])
               for _, lbl, _probe, col in domain_cfg]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=4,
              fontsize=8, title="Metric Domain", title_fontsize=8.5,
              frameon=True, framealpha=0.9)

    ax.text(5.25, 14.35,
            "Conceptual Framework: Autonomic Scaling Organization "
            "in Psychotic-Spectrum Illness",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1F3864", zorder=5)

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    _save(fig, "Figure9_ConceptualSynthesis")


def figure6_classifier(n_perm=None):
    try:
        import nested_classifier as nc
        from sklearn.metrics import roc_curve
    except Exception as e:                        # noqa: BLE001
        print(f"  [fig 6] skipped -- could not import nested_classifier: {e}")
        return
    if n_perm is None:
        n_perm = nc.N_PERMUTATIONS
    try:
        data, y, features = nc.load_data()
    except Exception as e:                        # noqa: BLE001
        print(f"  [fig 6] skipped -- classifier input not found: {e}")
        return

    X = data[features].to_numpy(dtype=float)
    probs = nc.nested_loo(X, y, features, do_selection=True)
    auc, acc, sens, spec = nc.score(y, probs)
    print(f"  [fig 6] honest nested-CV AUC = {auc:.3f}; running {n_perm} "
          f"permutations...")
    null, p_perm = nc.permutation_null(X, y, features, auc, n_perm=n_perm)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))

    ax = axes[0]
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, color=C_PSYCH, lw=2.2,
            label=f"Nested LOO (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--", label="Chance")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("(A) ROC, nested leave-one-out", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(color=C_GRID, lw=0.8)
    ax.text(0.03, 0.05,
            "acc %.0f%%  sens %.0f%%  spec %.0f%%" %
            (100 * acc, 100 * sens, 100 * spec),
            transform=ax.transAxes, fontsize=7.5, color="#555555")

    ax = axes[1]
    ax.hist(null, bins=30, color="#BBBBBB", edgecolor="white",
            label="Shuffled-label AUC (null)")
    ax.axvline(0.5, color="#999999", lw=1.0, ls=":", label="Chance = 0.5")
    ax.axvline(auc, color=C_PSYCH, lw=2.2,
               label=f"Observed = {auc:.3f}")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Permutations")
    floor = 1.0 / (len(null) + 1)
    p_txt = (f"p \u2264 {floor:.3f}" if p_perm <= floor * (1 + 1e-9)
             else f"p = {p_perm:.3f}")
    ax.set_title("(B) Permutation null (%d shuffles)\nnull mean %.3f, %s"
                 % (len(null), float(np.mean(null)), p_txt), fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color=C_GRID, lw=0.8)

    fig.suptitle("Nested cross-validation classifier", fontsize=10.5, y=1.02)
    fig.tight_layout()
    _save(fig, "Figure6_Classifier")


def main():
    print("=" * 64)
    print("  FIGURE GENERATION — wearable HRV reanalysis (v12)")
    print("=" * 64)

    print("\n[load] cohorts")
    df_hrvacc = _load_cohort(HRVACC_HEALTHY_CSV, "HRV-ACC healthy")
    df_psych  = _load_cohort(PSYCH_CSV,          "Psychiatric")
    df_mitbih = _load_cohort(MITBIH_HEALTHY_CSV, "MIT-BIH healthy")

    if df_hrvacc is None or df_psych is None:
        print("\n[fatal] primary cohorts missing — cannot build main figures.")
        return

    print("\n[stats] primary within-device comparison")
    stats = compute_stats(df_hrvacc, df_psych)
    n_sig = int(stats["sig"].sum())
    print(f"  {n_sig}/{len(stats)} metrics significant (BH-FDR)")

    print("\n[fig 1] effect sizes")
    figure1_effect_sizes(stats)

    print("\n[fig 2] box plots")
    figure2_boxplots(df_hrvacc, df_psych, stats)

    print("\n[fig 3] PSD curves")
    figure3_psd_curves(df_hrvacc, df_psych)

    print("\n[fig M] Mayer-wave diagnostic (decides the headline claim)")
    figure_mayer_diagnostic(df_hrvacc, df_psych)

    print("\n[fig S] beta against 2*alpha2 - 1 identity check")
    figure_beta_alpha_identity(df_hrvacc, df_psych)

    print("\n[fig 4] covariate analysis")
    figure4_covariate(df_hrvacc, df_psych)

    print("\n[fig 5] surrogate testing")
    figure5_surrogates(df_hrvacc, df_psych)

    figureS_mse_curves(df_hrvacc, df_psych)

    print("\n[fig 7] intercorrelation heatmaps")
    figure7_intercorrelation(df_hrvacc, df_psych, stats)

    if df_mitbih is not None:
        print("\n[fig 8] cross-device validation")
        figure6_cross_validation(df_mitbih, df_hrvacc, df_psych)
    else:
        print("\n[fig 7] skipped — MIT-BIH cohort not available")

    DRAW_SYNTHESIS = False
    if DRAW_SYNTHESIS:
        print("\n[fig 9] conceptual synthesis schematic")
        figure8_conceptual_synthesis()
    else:
        print("\n[fig 9] conceptual synthesis schematic SKIPPED "
              "(contents are out of date; see comment in figures.py)")

    print("\n[fig 6] classifier ROC + permutation null")
    figure6_classifier()

    print("\n" + "=" * 64)
    print(f"  Done. Figures written to {OUTPUT_DIR}/")
    print("=" * 64)


def figure_mayer_diagnostic(df_h, df_p):
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    try:
        from pipeline import reject_ectopic_beats, compute_frequency_domain
    except Exception as exc:
        print(f"  Mayer diagnostic skipped: cannot import pipeline ({exc})")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax, axr = axes
    any_data = False

    for df, colour, label in ((df_h, C_HEALTHY, "Controls"),
                              (df_p, C_PSYCH, "Patients")):
        spectra = []
        for rid in df.index:
            try:
                out = _subject_spectrum(rid, reject_ectopic_beats,
                                        compute_frequency_domain)
            except Exception:
                out = None
            if out is None:
                continue
            f, p = out
            spectra.append((f, p))
        if not spectra:
            continue
        any_data = True
        f0 = spectra[0][0]
        stack = np.vstack([np.interp(f0, f, p) for f, p in spectra])
        mean_p = np.exp(np.nanmean(np.log(np.clip(stack, 1e-30, None)), axis=0))
        m = (f0 >= 0.005) & (f0 <= 0.45)
        ax.loglog(f0[m], mean_p[m], color=colour, lw=2.0, label=label)

        fitm = ((f0 >= 0.01) & (f0 <= 0.15) &
                ~((f0 >= 0.07) & (f0 <= 0.13)) & (mean_p > 0))
        if fitm.sum() >= 3:
            lr = linregress(np.log10(f0[fitm]), np.log10(mean_p[fitm]))
            band = (f0 >= 0.01) & (f0 <= 0.15)
            pred = 10 ** (lr.intercept + lr.slope * np.log10(f0[band]))
            ax.loglog(f0[band], pred, color=colour, lw=1.2, ls="--", alpha=0.8)
            axr.semilogx(f0[band],
                         np.log10(mean_p[band]) - np.log10(pred),
                         color=colour, lw=1.8, label=label)

    if not any_data:
        plt.close(fig)
        print("  Mayer diagnostic skipped: per-subject spectra unavailable.")
        return None

    fit_lo, fit_hi = 0.015625, 0.1484
    for a in (ax, axr):
        a.axvspan(fit_lo, fit_hi, color="0.88", alpha=0.45, zorder=0)
        a.axvspan(0.07, 0.13, color="#EE7733", alpha=0.16, zorder=0)
        a.tick_params(labelsize=9)
    axr.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("PSD (ms$^2$/Hz, log scale)", fontsize=10)
    ax.set_title("Group-mean power spectra with fitted aperiodic components",
                 fontsize=10.5)
    ax.text(0.012, 0.97, "A", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")
    ax.legend(fontsize=9)
    axr.set_xlabel("Frequency (Hz)")
    axr.set_ylabel("log residual\nfrom aperiodic fit")
    axr.set_title("Residual from the aperiodic fit", fontsize=10)
    axr.text(0.012, 0.95, "B", transform=axr.transAxes, fontsize=13,
             fontweight="bold", va="top")
    plt.tight_layout()
    return _save(fig, "FigureM_MayerDiagnostic")


def figure_beta_alpha_identity(df_h, df_p):
    import matplotlib.pyplot as plt
    if "PSD_Beta" not in df_h.columns or "DFA_Alpha2" not in df_h.columns:
        print("  beta/alpha figure skipped: columns absent.")
        return None
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    allv = []
    for df, colour, label in ((df_h, C_HEALTHY, "Controls"),
                              (df_p, C_PSYCH, "Patients")):
        b = pd.to_numeric(df["PSD_Beta"], errors="coerce")
        a = 2 * pd.to_numeric(df["DFA_Alpha2"], errors="coerce") - 1
        ok = b.notna() & a.notna()
        ax.scatter(a[ok], b[ok], s=34, alpha=0.75, color=colour,
                   edgecolor="white", linewidth=0.6, label=label)
        allv.extend(list(a[ok]) + list(b[ok]))
    if allv:
        lo, hi = float(np.nanmin(allv)) - 0.1, float(np.nanmax(allv)) + 0.1
        ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, ls="--",
                label=r"identity ($\beta = 2\alpha_2 - 1$)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$2\alpha_2 - 1$ (predicted from DFA)")
    ax.set_ylabel(r"$\beta$ (observed spectral exponent)")
    ax.set_title("Observed uncorrected slope against the value predicted from DFA \u03b1\u2082",
                 fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "FigureS_BetaAlphaIdentity")


if __name__ == "__main__":
    main()
