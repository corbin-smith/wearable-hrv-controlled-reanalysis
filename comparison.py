"""
comparison.py
=============
Group comparison: Healthy (MIT-BIH NSR, N=18) vs Psychiatric (HRV-ACC, N=30).

Loads both cohort CSVs, runs all statistical comparisons and visualizations,
and saves outputs to outputs/comparison/.

Mandatory outputs
-----------------
  1.  Group statistics table (CSV + print)
      — Mann-Whitney U, rank-biserial r, raw p, BH-corrected p, significance
  2.  Comparison radar chart (healthy vs psychiatric cohort means)
  3.  Box plots with data points for all metrics (multi-panel)
  4.  Box plots — significant metrics only (publication figure)
  5.  Effect size bar chart (rank-biserial r, sorted)
  6.  Group summary table CSV (mean ± SD side by side + % difference)
  7.  Surrogate nonlinearity summary table
  8.  PSD Beta sensitivity check (full vs R²-filtered subset)

Optional (B) — Exploratory classification
  9.  Leave-one-out logistic regression on significant metrics
  10. ROC curves per significant metric

Optional (C) — Power analysis
  11. Sample size required for 80% power at observed effect sizes
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import (mannwhitneyu, shapiro, norm, spearmanr,
                         kruskal, levene, ttest_ind, rankdata, t as t_dist)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (roc_auc_score, roc_curve,
                             accuracy_score, confusion_matrix)
from pipeline import (
    plot_radar_comparison,
    RADAR_METRICS, RADAR_LABELS,
    PSD_BETA_R2_THRESHOLD,
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ── Cohort result CSVs ───────────────────────────────────────
# MIT-BIH NSR healthy controls (secondary / supplemental robustness arm).
MITBIH_HEALTHY_CSV = os.path.join("outputs", "healthy",
                                  "ecg_results_Healthy.csv")
# HRV-ACC healthy controls — the PRIMARY, recording-matched control group
# (produced by healthy_hrvacc.py). This file will not exist until you run
# healthy_hrvacc.py; the main block checks for it and skips gracefully.
HRVACC_HEALTHY_CSV = os.path.join("outputs", "healthy_hrvacc",
                                  "ecg_results_HealthyACC.csv")
# Backward-compatible alias: the original single-control default path.
HEALTHY_CSV    = MITBIH_HEALTHY_CSV
PSYCH_CSV      = os.path.join("outputs", "psychiatric",
                               "ecg_results_Psychiatric.csv")
OUTPUT_DIR     = os.path.join("outputs", "comparison")

# NOTE — HEALTHY_CSV above still loads the MIT-BIH cohort. Switching the
# PRIMARY control to healthy_hrvacc.py (with MIT-BIH kept as a supplemental
# arm) is the upcoming comparison.py editing task; paths are unchanged for now.

# Radar-chart normalisation reference frame — source is the gold-standard
# healthy control cohort, healthy_hrvacc.py.
# ⚠️  UPDATE REQUIRED BEFORE RUNNING ⚠️
# The values below are STALE placeholders from the old MIT-BIH pilot
# (N=10, 10-min). Run healthy_hrvacc.py (HRV-ACC controls, N=30); it prints a
# ready-to-paste HEALTHY_MEAN / HEALTHY_SD block. Paste it over
# the ten values below, then re-run this script.
HEALTHY_MEAN = pd.Series({
    "PSD_Beta": 0.966580872517224,
    "DFA_Alpha1": 1.384028201792553,
    "SampEn": 0.7614310907930708,
    "MSE_Mean": 1.2407928227991503,
    "RQA_DET": 0.6617478010278295,
})

HEALTHY_SD = pd.Series({
    "PSD_Beta": 0.29624171268879984,
    "DFA_Alpha1": 0.11672191364953513,
    "SampEn": 0.27492195688105586,
    "MSE_Mean": 0.3290378487743961,
    "RQA_DET": 0.13547917859774117,
})

# Full metric battery for group comparison
# (excludes pure QC columns and surrogate internals)
METRIC_BATTERY = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_MeanHR",
    "HRV_SD1", "HRV_SD2", "HRV_SD1SD2",
    "LF_Power_ms2", "HF_Power_ms2", "Total_Power_ms2", "LF_HF_Ratio",
    "PSD_Beta",
    "SampEn", "ApEn", "PermEn", "MSE_Mean",
    "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2",
    # MFDFA  — multifractal extension of DFA (Kantelhardt et al. 2002)
    "MFDFA_Width", "MFDFA_H_Mean", "MFDFA_Asymmetry", "MFDFA_H2",
    "RQA_REC", "RQA_DET", "RQA_LAM",
    # RNA — recurrence network analysis (Marwan et al. 2009)
    "RNA_MeanDegree", "RNA_ClusteringCoeff", "RNA_Transitivity",
]

# Friendly display names for figures
METRIC_LABELS = {
    "HRV_MeanNN":     "Mean NN (ms)",
    "HRV_SDNN":       "SDNN (ms)",
    "HRV_RMSSD":      "RMSSD (ms)",
    "HRV_pNN50":      "pNN50 (%)",
    "HRV_MeanHR":     "Mean HR (bpm)",
    "HRV_SD1":        "SD1 (ms)",
    "HRV_SD2":        "SD2 (ms)",
    "HRV_SD1SD2":     "SD1/SD2",
    "LF_Power_ms2":   "LF Power (ms²)",
    "HF_Power_ms2":   "HF Power (ms²)",
    "Total_Power_ms2":"Total Power (ms²)",
    "LF_HF_Ratio":    "LF/HF Ratio",
    "PSD_Beta":       "PSD β",
    "SampEn":         "Sample Entropy",
    "ApEn":           "Approx. Entropy",
    "PermEn":         "Permutation Entropy",
    "MSE_Mean":       "MSE Mean",
    "DFA_Alpha":      "DFA α (full)",
    "DFA_Alpha1":     "DFA α₁ (4–16)",
    "DFA_Alpha2":     "DFA α₂ (16–64)",
    "RQA_REC":        "RQA Recurrence",
    "RQA_DET":        "RQA Determinism",
    "RQA_LAM":        "RQA Laminarity",
    # MFDFA
    "MFDFA_Width":     "MFDFA \u0394h (width)",
    "MFDFA_H_Mean":    "MFDFA H\u0305 (mean)",
    "MFDFA_Asymmetry": "MFDFA Asymmetry",
    "MFDFA_H2":        "MFDFA H(q=2)",
    # RNA
    "RNA_MeanDegree":      "RNA Mean Degree",
    "RNA_ClusteringCoeff": "RNA Clustering Coeff",
    "RNA_Transitivity":    "RNA Transitivity",
}

# Significance threshold after FDR correction
ALPHA = 0.05

# Colors
C_HEALTHY  = "steelblue"
C_PSYCH    = "firebrick"

# ── Healthy subject demographics ──────────────────────────────
# Ages and sexes provided in record-number order (16265 … 19830).
# Sex coding: 0 = male, 1 = female.
# Used in the PSD β covariate analysis to test age and sex confounds
# within the healthy cohort (individual psychiatric demographics are
# unavailable; only group-level statistics exist for that cohort).
HEALTHY_DEMOGRAPHICS = {
    "16265": {"Age": 32, "Sex": 0},   # male
    "16272": {"Age": 20, "Sex": 1},   # female
    "16273": {"Age": 28, "Sex": 1},
    "16420": {"Age": 38, "Sex": 1},
    "16483": {"Age": 42, "Sex": 0},   # male
    "16539": {"Age": 35, "Sex": 1},
    "16773": {"Age": 26, "Sex": 0},   # male
    "16786": {"Age": 32, "Sex": 1},
    "16795": {"Age": 20, "Sex": 1},
    "17052": {"Age": 45, "Sex": 1},
    "17453": {"Age": 32, "Sex": 1},
    "18177": {"Age": 26, "Sex": 1},
    "18184": {"Age": 34, "Sex": 1},
    "19088": {"Age": 41, "Sex": 1},
    "19090": {"Age": 45, "Sex": 0},   # male
    "19093": {"Age": 34, "Sex": 0},   # male
    "19140": {"Age": 38, "Sex": 1},
    "19830": {"Age": 50, "Sex": 1},
}
# Healthy cohort summary: mean age = 34.3 yr, range 20–50; 13F / 5M


def _add_healthy_demographics(df_h, demographics=None):
    """
    Attach Age and Sex columns to the healthy DataFrame from
    HEALTHY_DEMOGRAPHICS.  Rows whose index does not appear in the
    dict (e.g. if the index is a full path rather than a bare record ID)
    receive NaN so that downstream dropna() calls handle them cleanly.
    """
    df = df_h.copy()
    demographics = HEALTHY_DEMOGRAPHICS if demographics is None else demographics
    ages, sexes = [], []
    for idx in df.index:
        # Index may be the bare record ID ("16265") or contain it as a suffix
        key = str(idx).split("/")[-1].split("\\")[-1]
        demo = demographics.get(key, {})
        ages.append(demo.get("Age", np.nan))
        sexes.append(demo.get("Sex", np.nan))
    df["Age"] = ages
    df["Sex"] = sexes
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_cohorts(healthy_csv, psych_csv):
    """
    Load both cohort CSVs, strip summary stat rows (Cohort_* / Healthy_* /
    Psychiatric_* index labels), and return clean per-subject DataFrames.
    """
    def _load(path, label):
        df = pd.read_csv(path, index_col=0)
        # Drop summary rows appended by run_analysis()
        summary_pattern = df.index.str.contains(
            r'^(Cohort_|Healthy_|HealthyACC_|Psychiatric_)', regex=True, na=False
        )
        df = df[~summary_pattern].copy()
        # Coerce all columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        print(f"  Loaded {label}: {len(df)} subjects, {len(df.columns)} columns")
        return df

    df_h = _load(healthy_csv,  "Healthy")
    df_p = _load(psych_csv,    "Psychiatric")
    return df_h, df_p


# ═══════════════════════════════════════════════════════════════
# STEP 2 — STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════

def rank_biserial_r(u_stat, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u_stat) / (n1 * n2)


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    Returns array of adjusted p-values (same length as input).
    """
    p    = np.array(p_values, dtype=float)
    n    = len(p)
    rank = np.argsort(p) + 1          # 1-indexed ranks
    adj  = np.minimum(1.0, p * n / rank)
    # Enforce monotonicity (from largest to smallest)
    for i in range(n - 2, -1, -1):
        adj[np.argsort(p)[i]] = min(
            adj[np.argsort(p)[i]],
            adj[np.argsort(p)[i + 1]]
        )
    return adj


def run_group_comparison(df_h, df_p, metrics):
    """
    Mann-Whitney U test for each metric.
    Returns a DataFrame with columns:
        Healthy_Mean, Healthy_SD, Psych_Mean, Psych_SD, Pct_Diff,
        U_stat, p_raw, r_effect, p_adj, Significant
    """
    rows = []
    for m in metrics:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        h_vals = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p_vals = pd.to_numeric(df_p[m], errors='coerce').dropna().values
        if len(h_vals) < 3 or len(p_vals) < 3:
            continue

        u, p_raw = mannwhitneyu(h_vals, p_vals, alternative='two-sided')
        r        = rank_biserial_r(u, len(h_vals), len(p_vals))
        h_mean   = np.mean(h_vals)
        p_mean   = np.mean(p_vals)
        pct_diff = 100 * (p_mean - h_mean) / abs(h_mean) if h_mean != 0 else np.nan

        rows.append({
            "Metric":       m,
            "Healthy_N":    len(h_vals),
            "Healthy_Mean": h_mean,
            "Healthy_SD":   np.std(h_vals, ddof=1),
            "Psych_N":      len(p_vals),
            "Psych_Mean":   p_mean,
            "Psych_SD":     np.std(p_vals, ddof=1),
            "Pct_Diff":     pct_diff,
            "U_stat":       u,
            "p_raw":        p_raw,
            "r_effect":     r,
        })

    results = pd.DataFrame(rows).set_index("Metric")

    # BH correction across all tests
    p_adj = benjamini_hochberg(results["p_raw"].values)
    results["p_adj"]       = p_adj
    results["Significant"] = p_adj < ALPHA

    results = results.sort_values("p_adj")
    return results


# ═══════════════════════════════════════════════════════════════
# STEP 3 — GROUP SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

def save_summary_table(stats, output_dir):
    """
    Save the full statistical results as a formatted CSV and
    print a clean console summary.
    """
    out = stats.copy()
    out.index = [METRIC_LABELS.get(m, m) for m in out.index]
    path = os.path.join(output_dir, "group_comparison_stats.csv")
    out.to_csv(path, float_format="%.4f")
    print(f"  Stats table     → {path}")

    # Console summary
    print(f"\n{'='*80}")
    print(f"  GROUP COMPARISON  (α={ALPHA}, BH-FDR corrected)")
    print(f"{'='*80}")
    header = (f"{'Metric':<26} {'H_Mean':>9} {'P_Mean':>9} "
              f"{'%Diff':>7} {'r':>6} {'p_raw':>8} {'p_adj':>8} {'Sig':>4}")
    print(header)
    print("-" * 80)
    for metric, row in stats.iterrows():
        label = METRIC_LABELS.get(metric, metric)
        sig   = "★" if row["Significant"] else ""
        print(f"{label:<26} {row['Healthy_Mean']:>9.3f} {row['Psych_Mean']:>9.3f} "
              f"{row['Pct_Diff']:>7.1f} {row['r_effect']:>6.3f} "
              f"{row['p_raw']:>8.4f} {row['p_adj']:>8.4f} {sig:>4}")

    sig_metrics = stats.index[stats["Significant"]].tolist()
    print(f"\n  Significant after FDR correction: {len(sig_metrics)}/{len(stats)}")
    for m in sig_metrics:
        print(f"    {METRIC_LABELS.get(m, m)}  "
              f"r={stats.loc[m,'r_effect']:.3f}  "
              f"p_adj={stats.loc[m,'p_adj']:.4f}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — BOX PLOTS (ALL METRICS)
# ═══════════════════════════════════════════════════════════════

def _single_boxplot(ax, df_h, df_p, metric, stats_row):
    """Draw one grouped box plot with jittered data points on ax."""
    h_vals = pd.to_numeric(df_h[metric], errors='coerce').dropna().values
    p_vals = pd.to_numeric(df_p[metric], errors='coerce').dropna().values

    bp = ax.boxplot(
        [h_vals, p_vals],
        positions=[1, 2], widths=0.5,
        patch_artist=True,
        medianprops=dict(color='black', lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
    )
    bp['boxes'][0].set_facecolor(C_HEALTHY); bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(C_PSYCH);   bp['boxes'][1].set_alpha(0.5)

    # Jittered individual points
    rng = np.random.default_rng(seed=0)
    for vals, pos, col in [(h_vals, 1, C_HEALTHY), (p_vals, 2, C_PSYCH)]:
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jitter, vals, color=col, s=18, alpha=0.7, zorder=3)

    label = METRIC_LABELS.get(metric, metric)
    ax.set_title(label, fontsize=8, pad=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["H", "P"], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate p_adj and significance
    sig_str = (f"r={stats_row['r_effect']:.2f}\n"
               f"p={stats_row['p_adj']:.3f}"
               + (" ★" if stats_row["Significant"] else ""))
    ax.text(0.97, 0.97, sig_str, transform=ax.transAxes,
            ha='right', va='top', fontsize=6.5,
            color='red' if stats_row["Significant"] else 'grey')


def plot_all_boxplots(df_h, df_p, stats, output_dir, healthy_label="Healthy"):
    """
    Multi-panel figure: one box plot per metric (all metrics).
    Arranged in a grid, significant metrics highlighted with red title.
    """
    metrics = [m for m in stats.index if m in df_h.columns and m in df_p.columns]
    n       = len(metrics)
    ncols   = 5
    nrows   = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.2))
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        _single_boxplot(axes_flat[i], df_h, df_p, metric, stats.loc[metric])
        if stats.loc[metric, "Significant"]:
            axes_flat[i].set_title(
                METRIC_LABELS.get(metric, metric),
                fontsize=8, pad=3, color='red', fontweight='bold'
            )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{healthy_label} vs Psychiatric — All Metrics\n"
        f"(red = significant after BH-FDR, H = {healthy_label}, P = Psychiatric)",
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "Boxplots_All_Metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Box plots (all) → {path}")


def plot_significant_boxplots(df_h, df_p, stats, output_dir, healthy_label="Healthy"):
    """
    Publication-quality multi-panel figure for significant metrics only.
    Larger panels, full axis labels, cleaner layout.
    """
    sig_metrics = stats.index[stats["Significant"]].tolist()
    if not sig_metrics:
        print("  No significant metrics — significant boxplot figure skipped.")
        return

    n     = len(sig_metrics)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 4),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, metric in enumerate(sig_metrics):
        ax     = axes_flat[i]
        h_vals = pd.to_numeric(df_h[metric], errors='coerce').dropna().values
        p_vals = pd.to_numeric(df_p[metric], errors='coerce').dropna().values
        row    = stats.loc[metric]

        bp = ax.boxplot(
            [h_vals, p_vals], positions=[1, 2], widths=0.55,
            patch_artist=True,
            medianprops=dict(color='black', lw=2.2),
            whiskerprops=dict(lw=1.4),
            capprops=dict(lw=1.4),
            flierprops=dict(marker='o', markersize=4, alpha=0.4),
        )
        bp['boxes'][0].set_facecolor(C_HEALTHY); bp['boxes'][0].set_alpha(0.55)
        bp['boxes'][1].set_facecolor(C_PSYCH);   bp['boxes'][1].set_alpha(0.55)

        rng = np.random.default_rng(seed=0)
        for vals, pos, col in [(h_vals, 1, C_HEALTHY), (p_vals, 2, C_PSYCH)]:
            jitter = rng.uniform(-0.14, 0.14, len(vals))
            ax.scatter(pos + jitter, vals, color=col, s=28, alpha=0.75, zorder=3)

        ax.set_title(METRIC_LABELS.get(metric, metric),
                     fontsize=10, fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Healthy", "Psychiatric"], fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        # Bracket + significance annotation
        y_max  = max(np.nanmax(h_vals), np.nanmax(p_vals))
        y_range = y_max - min(np.nanmin(h_vals), np.nanmin(p_vals))
        y_br   = y_max + y_range * 0.07
        ax.plot([1, 1, 2, 2],
                [y_br, y_br + y_range*0.02, y_br + y_range*0.02, y_br],
                lw=1.2, color='black')
        p_str = (f"p={row['p_adj']:.3f} ★" if row['p_adj'] < 0.001
                 else f"p={row['p_adj']:.3f} ★")
        ax.text(1.5, y_br + y_range * 0.03, p_str,
                ha='center', va='bottom', fontsize=8.5, color='red')
        ax.text(0.97, 0.02,
                f"r = {row['r_effect']:.3f}",
                transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='dimgrey')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Significant Group Differences: {healthy_label} vs Psychiatric\n"
        f"(Mann-Whitney U, BH-FDR α={ALPHA}; r = rank-biserial correlation)",
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "Boxplots_Significant_Metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Box plots (sig) → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — EFFECT SIZE BAR CHART
# ═══════════════════════════════════════════════════════════════

def plot_effect_sizes(stats, output_dir, healthy_label="Healthy"):
    """
    Horizontal bar chart of rank-biserial r for all metrics,
    sorted by absolute magnitude. Significant metrics colored differently.
    Vertical dashed lines at |r| = 0.3 (medium) and 0.5 (large).
    """
    df = stats[["r_effect", "Significant"]].copy()
    df["abs_r"] = df["r_effect"].abs()
    df = df.sort_values("abs_r", ascending=True)

    labels = [METRIC_LABELS.get(m, m) for m in df.index]
    colors = [C_PSYCH if s else "lightsteelblue" for s in df["Significant"]]

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.38)))
    bars = ax.barh(labels, df["r_effect"], color=colors,
                   edgecolor='white', height=0.7)

    ax.axvline(0,    color='black', lw=0.8)
    ax.axvline( 0.3, color='grey',  lw=0.9, linestyle='--', alpha=0.6,
                label='|r|=0.3 (medium)')
    ax.axvline(-0.3, color='grey',  lw=0.9, linestyle='--', alpha=0.6)
    ax.axvline( 0.5, color='grey',  lw=0.9, linestyle=':',  alpha=0.6,
                label='|r|=0.5 (large)')
    ax.axvline(-0.5, color='grey',  lw=0.9, linestyle=':',  alpha=0.6)

    for bar, val in zip(bars, df["r_effect"]):
        x = bar.get_width()
        ax.text(x + (0.01 if x >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", ha='left' if x >= 0 else 'right',
                va='center', fontsize=7.5)

    ax.set_xlabel("Rank-biserial correlation (r)", fontsize=10)
    ax.set_title(
        f"Effect Sizes: {healthy_label} vs Psychiatric\n"
        f"(red = significant after BH-FDR α={ALPHA}; "
        "positive r = Psychiatric > Healthy)",
        fontsize=10
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "Effect_Sizes.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Effect sizes    → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — VIOLIN PLOTS (distribution shape)
# ═══════════════════════════════════════════════════════════════

def plot_violin_comparison(df_h, df_p, stats, output_dir, healthy_label="Healthy"):
    """
    Violin plots for the five radar metrics — shows full distribution
    shape rather than just quartiles. More informative than box plots
    for small healthy N where distribution shape matters.
    Significant metrics annotated.
    """
    metrics = [m for m in RADAR_METRICS
               if m in df_h.columns and m in df_p.columns]
    n     = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 5))

    for ax, metric in zip(axes, metrics):
        h_vals = pd.to_numeric(df_h[metric], errors='coerce').dropna().values
        p_vals = pd.to_numeric(df_p[metric], errors='coerce').dropna().values

        vp = ax.violinplot([h_vals, p_vals], positions=[1, 2],
                           showmedians=True, showextrema=True)
        for pc, col in zip(vp['bodies'], [C_HEALTHY, C_PSYCH]):
            pc.set_facecolor(col); pc.set_alpha(0.5)
        vp['cmedians'].set_color('black'); vp['cmedians'].set_lw(2)

        rng = np.random.default_rng(seed=0)
        for vals, pos, col in [(h_vals, 1, C_HEALTHY), (p_vals, 2, C_PSYCH)]:
            ax.scatter(pos + rng.uniform(-0.08, 0.08, len(vals)),
                       vals, color=col, s=18, alpha=0.8, zorder=3)

        label = METRIC_LABELS.get(metric, metric)
        sig   = metric in stats.index and stats.loc[metric, "Significant"]
        ax.set_title(label, fontsize=9,
                     fontweight='bold' if sig else 'normal',
                     color='red' if sig else 'black')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Healthy", "Psych"], fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        if sig:
            r_val = stats.loc[metric, "r_effect"]
            p_val = stats.loc[metric, "p_adj"]
            ax.text(0.97, 0.97, f"r={r_val:.2f}\np={p_val:.3f}★",
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.5, color='red')

    fig.suptitle(f"Distribution Comparison ({healthy_label} vs Psychiatric) "
                 "— Core Criticality Metrics\n"
                 "(violin = distribution shape; dots = individual subjects)",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "Violin_Radar_Metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Violin plots    → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — Z-SCORE HEATMAP (per subject, all metrics)
# ═══════════════════════════════════════════════════════════════

def plot_zscore_heatmap(df_h, df_p, output_dir, healthy_label="Healthy"):
    """
    Heatmap where each row is a subject and each column is a metric,
    z-scored on the healthy cohort parameters.
    Healthy subjects on top (blue separator), psychiatric below (red).
    This gives an at-a-glance view of individual-level variability and
    where the groups systematically diverge.
    """
    metrics = [m for m in METRIC_BATTERY
               if m in df_h.columns and m in df_p.columns]

    h_sub = df_h[metrics].apply(pd.to_numeric, errors='coerce')
    p_sub = df_p[metrics].apply(pd.to_numeric, errors='coerce')

    # Z-score everything on healthy cohort parameters
    h_mean = h_sub.mean()
    h_std  = h_sub.std(ddof=1).replace(0, 1)

    h_z = (h_sub - h_mean) / h_std
    p_z = (p_sub - h_mean) / h_std

    combined = pd.concat([h_z, p_z], axis=0)
    combined = combined.clip(-3, 3)      # clip extreme outliers for display

    n_h   = len(h_z)
    n_p   = len(p_z)
    col_labels = [METRIC_LABELS.get(m, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(max(14, len(metrics) * 0.65),
                                    max(8,  (n_h + n_p) * 0.35)))
    im = ax.imshow(combined.values, aspect='auto', cmap='RdBu_r',
                   vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, label="z-score (healthy reference)", shrink=0.6)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7.5)
    ax.set_yticks(range(n_h + n_p))
    ax.set_yticklabels(
        list(h_z.index) + list(p_z.index), fontsize=7
    )

    # Group separator line
    ax.axhline(n_h - 0.5, color='black', lw=2.5)
    ax.text(-0.5, n_h / 2 - 0.5, healthy_label, va='center',
            ha='right', fontsize=9, color=C_HEALTHY, fontweight='bold',
            rotation=90)
    ax.text(-0.5, n_h + n_p / 2 - 0.5, "Psychiatric", va='center',
            ha='right', fontsize=9, color=C_PSYCH, fontweight='bold',
            rotation=90)

    ax.set_title(
        "Subject-Level Z-Score Heatmap (healthy cohort reference)\n"
        "Blue = below healthy mean  |  Red = above healthy mean  |  clipped ±3 SD",
        fontsize=10
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "ZScore_Heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Z-score heatmap → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 8 — SURROGATE NONLINEARITY SUMMARY
# ═══════════════════════════════════════════════════════════════

def surrogate_summary(df_h, df_p, output_dir, healthy_label="Healthy"):
    """
    Table and bar chart showing what fraction of subjects in each cohort
    had a significant IAAFT surrogate test for each of the three tested
    metrics (SampEn, DFA_Alpha1, MSE_Mean).

    A higher fraction of significant surrogates in the healthy group
    would indicate that healthy HRV has stronger nonlinear structure —
    a direct criticality prediction.
    """
    surr_metrics = ["SampEn", "DFA_Alpha1", "MSE_Mean"]
    rows = []
    for m in surr_metrics:
        col = f"SURR_{m}_Sig"
        if col not in df_h.columns or col not in df_p.columns:
            continue
        h_sig = pd.to_numeric(df_h[col], errors='coerce').fillna(0).astype(bool)
        p_sig = pd.to_numeric(df_p[col], errors='coerce').fillna(0).astype(bool)
        rows.append({
            "Metric":       METRIC_LABELS.get(m, m),
            "Healthy_Sig":  int(h_sig.sum()),
            "Healthy_N":    len(h_sig),
            "Healthy_Pct":  100 * h_sig.mean(),
            "Psych_Sig":    int(p_sig.sum()),
            "Psych_N":      len(p_sig),
            "Psych_Pct":    100 * p_sig.mean(),
        })

    if not rows:
        print("  Surrogate summary: no SURR_*_Sig columns found — skipped.")
        return

    surr_df = pd.DataFrame(rows).set_index("Metric")
    path_csv = os.path.join(output_dir, "Surrogate_Nonlinearity_Summary.csv")
    surr_df.to_csv(path_csv)
    print(f"  Surrogate table → {path_csv}")
    print("\n  SURROGATE NONLINEARITY SUMMARY")
    print(surr_df[["Healthy_Sig","Healthy_N","Healthy_Pct",
                   "Psych_Sig","Psych_N","Psych_Pct"]].to_string())

    # Bar chart
    x    = np.arange(len(surr_df))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, surr_df["Healthy_Pct"], w,
           label=healthy_label, color=C_HEALTHY, alpha=0.75)
    ax.bar(x + w/2, surr_df["Psych_Pct"],   w,
           label="Psychiatric", color=C_PSYCH,   alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(surr_df.index, fontsize=9)
    ax.set_ylabel("% Subjects with Significant Surrogate Test")
    ax.set_title("IAAFT Surrogate Nonlinearity — Fraction Significant\n"
                 "(higher % = stronger nonlinear dynamics in that group)")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 110)
    for i, (hv, pv) in enumerate(zip(surr_df["Healthy_Pct"],
                                     surr_df["Psych_Pct"])):
        ax.text(i - w/2, hv + 2, f"{hv:.0f}%", ha='center', fontsize=8)
        ax.text(i + w/2, pv + 2, f"{pv:.0f}%", ha='center', fontsize=8)
    plt.tight_layout()
    path_fig = os.path.join(output_dir, "Surrogate_Nonlinearity_Bar.png")
    plt.savefig(path_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Surrogate bar   → {path_fig}")


# ═══════════════════════════════════════════════════════════════
# STEP 9a — PSD BETA COVARIATE ANALYSIS  (Category 2, Analysis 1)
# ═══════════════════════════════════════════════════════════════

def psd_beta_covariate_analysis(df_h, df_p, output_dir,
                                demographics=None, healthy_label="Healthy"):
    """
    Tests whether the PSD β group difference persists after controlling
    for covariates that may confound the cross-dataset comparison.

    Cross-group covariates (available for all 48 subjects):
        HRV_MeanHR — overall heart rate; elevated in psychiatric cohort
        HRV_RMSSD  — general vagal HRV; the single most powerful group
                     discriminator (r = −0.907); if β simply tracks RMSSD
                     it offers no independent information

    Within-healthy confounds (individual demographics provided):
        Age  — HRV and β decline with age; tested within healthy cohort
        Sex  — minor HRV sex differences; tested within healthy cohort
               (individual psychiatric demographics unavailable)

    Approach
    --------
    Partial Spearman correlation: residualise PSD_Beta and Group
    (binary 0/1) on each covariate via OLS before correlating, giving
    the association between PSD_Beta and group membership with the
    covariate effect removed from both variables.

    Incremental AUC: LOO logistic regression with [MeanHR, RMSSD] as
    baseline vs [MeanHR, RMSSD, PSD_Beta] as full model — quantifies
    how much discriminative information β adds beyond general HRV.

    Outputs
    -------
    PSD_Beta_Covariate_Analysis.csv  — numeric results table
    PSD_Beta_Covariate_Analysis.png  — 2-panel summary figure
    """
    # ── Build combined dataset ────────────────────────────────
    dh = df_h.copy(); dh["Group"] = 0
    dp = df_p.copy(); dp["Group"] = 1
    combined = pd.concat([dh, dp], axis=0)

    req_cols = ["PSD_Beta", "HRV_MeanHR", "HRV_RMSSD", "Group"]
    for c in req_cols:
        combined[c] = pd.to_numeric(combined[c], errors='coerce')
    cdf = combined[req_cols].dropna().copy()

    if len(cdf) < 10:
        print("  Covariate analysis: insufficient data — skipped.")
        return

    # ── Partial Spearman correlation helper ───────────────────
    def partial_spearman(target_col, covar_cols, outcome_col, df):
        """
        Partial Spearman r between target and outcome, controlling for
        covariates.  OLS residuals preserve the rank-based logic of
        Spearman while removing linear covariate effects — standard
        practice for nonparametric partial correlation.
        """
        t = df[target_col].values.astype(float)
        o = df[outcome_col].values.astype(float)
        Z = df[covar_cols].values.astype(float)
        Z_des = np.column_stack([np.ones(len(Z)), Z])

        def ols_resid(y):
            coef, _, _, _ = np.linalg.lstsq(Z_des, y, rcond=None)
            return y - Z_des @ coef

        r, p = spearmanr(ols_resid(t), ols_resid(o))
        return float(r), float(p)

    rows = []

    # Unadjusted
    r0, p0 = spearmanr(cdf["PSD_Beta"], cdf["Group"])
    rows.append({"Analysis": "PSD_Beta ~ Group (unadjusted)",
                 "r_or_AUC": r0, "p": p0})

    # Controlling for MeanHR alone
    r1, p1 = partial_spearman("PSD_Beta", ["HRV_MeanHR"], "Group", cdf)
    rows.append({"Analysis": "PSD_Beta ~ Group | MeanHR",
                 "r_or_AUC": r1, "p": p1})

    # Controlling for RMSSD alone
    r2, p2 = partial_spearman("PSD_Beta", ["HRV_RMSSD"], "Group", cdf)
    rows.append({"Analysis": "PSD_Beta ~ Group | RMSSD",
                 "r_or_AUC": r2, "p": p2})

    # Controlling for MeanHR + RMSSD jointly
    r3, p3 = partial_spearman("PSD_Beta", ["HRV_MeanHR", "HRV_RMSSD"],
                               "Group", cdf)
    rows.append({"Analysis": "PSD_Beta ~ Group | MeanHR + RMSSD",
                 "r_or_AUC": r3, "p": p3})

    # ── Incremental AUC ───────────────────────────────────────
    loo    = LeaveOneOut()
    scaler = StandardScaler()
    clf    = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    y      = cdf["Group"].values

    def _loo_auc(X):
        probs = []
        for tr, te in loo.split(X):
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf.fit(Xtr, y[tr])
            probs.append(clf.predict_proba(Xte)[0, 1])
        return float(roc_auc_score(y, probs))

    X_base = cdf[["HRV_MeanHR", "HRV_RMSSD"]].values
    X_full = cdf[["HRV_MeanHR", "HRV_RMSSD", "PSD_Beta"]].values
    auc_base = _loo_auc(X_base)
    auc_full = _loo_auc(X_full)

    rows.append({"Analysis": "LOO-AUC: MeanHR + RMSSD (baseline)",
                 "r_or_AUC": auc_base, "p": np.nan})
    rows.append({"Analysis": "LOO-AUC: MeanHR + RMSSD + PSD_Beta (full)",
                 "r_or_AUC": auc_full, "p": np.nan})
    rows.append({"Analysis": "Delta AUC (PSD_Beta incremental contribution)",
                 "r_or_AUC": auc_full - auc_base, "p": np.nan})

    # ── Within-healthy: age and sex ───────────────────────────
    h_demo = _add_healthy_demographics(df_h, demographics=demographics)
    for c in ["PSD_Beta", "Age", "Sex"]:
        h_demo[c] = pd.to_numeric(h_demo[c], errors='coerce')
    h_clean = h_demo[["PSD_Beta", "Age", "Sex"]].dropna()

    if len(h_clean) >= 5:
        r_age, p_age = spearmanr(h_clean["PSD_Beta"], h_clean["Age"])
        rows.append({"Analysis": f"Within-healthy: PSD_Beta ~ Age "
                                  f"(N={len(h_clean)}, Spearman)",
                     "r_or_AUC": float(r_age), "p": float(p_age)})

        h_m = h_clean[h_clean["Sex"] == 0]["PSD_Beta"].values
        h_f = h_clean[h_clean["Sex"] == 1]["PSD_Beta"].values
        if len(h_m) >= 2 and len(h_f) >= 2:
            u_sex, p_sex = mannwhitneyu(h_m, h_f, alternative='two-sided')
            r_sex = rank_biserial_r(u_sex, len(h_m), len(h_f))
            rows.append({
                "Analysis": f"Within-healthy: PSD_Beta by Sex "
                             f"(M n={len(h_m)}, F n={len(h_f)}, MW)",
                "r_or_AUC": float(r_sex), "p": float(p_sex)
            })

    # ── Save CSV ──────────────────────────────────────────────
    covar_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "PSD_Beta_Covariate_Analysis.csv")
    covar_df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"\n  PSD BETA COVARIATE ANALYSIS")
    print(f"  {'Analysis':<55} {'r / AUC':>8} {'p':>8}")
    print("  " + "-" * 75)
    for _, row in covar_df.iterrows():
        p_str = f"{row['p']:.4f}" if np.isfinite(row["p"]) else "  —"
        print(f"  {row['Analysis']:<55} {row['r_or_AUC']:>8.4f} {p_str:>8}")
    print(f"  Saved → {csv_path}")

    # ── Figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        "PSD β Covariate Analysis — Addressing Reviewer Concerns\n"
        "(does PSD β carry information beyond general HRV reduction?)",
        fontsize=10
    )

    # Panel A: partial correlations
    corr_rows = [r for r in rows
                 if "~" in r["Analysis"] and "LOO" not in r["Analysis"]]
    ax = axes[0]
    c_labels = [r["Analysis"].replace("PSD_Beta ~ ", "")
                              .replace("Within-healthy: PSD_Beta ", "")
                for r in corr_rows]
    c_vals   = [r["r_or_AUC"] for r in corr_rows]
    c_colors = [C_PSYCH if (np.isfinite(r["p"]) and r["p"] < 0.05)
                else "lightsteelblue"
                for r in corr_rows]
    bars = ax.barh(c_labels, c_vals, color=c_colors,
                   edgecolor='white', height=0.6)
    ax.axvline(0, color='black', lw=0.8)
    for bar, val in zip(bars, c_vals):
        x = bar.get_width()
        ax.text(x + (0.01 if x >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha='left' if x >= 0 else 'right',
                va='center', fontsize=8)
    ax.set_xlabel("Spearman r  (rank-biserial r for MW)", fontsize=9)
    ax.set_title("Partial / Bivariate Correlations\n(red = p < 0.05)", fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Panel B: incremental AUC
    auc_rows = [r for r in rows if "LOO-AUC" in r["Analysis"]]
    ax = axes[1]
    a_labels = [r["Analysis"].replace("LOO-AUC: ", "") for r in auc_rows]
    a_vals   = [r["r_or_AUC"] for r in auc_rows]
    ax.bar(a_labels, a_vals, color=["steelblue", C_PSYCH],
           edgecolor='white', width=0.5)
    ax.axhline(0.5, color='grey', lw=0.8, linestyle='--', label='Chance')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("LOO AUC", fontsize=9)
    ax.set_title("Incremental Discriminative Value of PSD β\n"
                 "(beyond MeanHR + RMSSD baseline)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for i, val in enumerate(a_vals):
        ax.text(i, val + 0.02, f"{val:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "PSD_Beta_Covariate_Analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Covariate plot  → {fig_path}")


# ═══════════════════════════════════════════════════════════════
# STEP 9b — PSD BETA FREQUENCY-RANGE & PREPROCESSING SENSITIVITY
#            (Category 2, Analysis 2 — enhanced version of step 9)
# ═══════════════════════════════════════════════════════════════

def psd_beta_range_sensitivity(df_h, df_p, stats, output_dir):
    """
    Test whether the PSD β group difference is robust across:
      (a) three frequency fitting ranges (0.01–0.15, 0.003–0.15,
          0.003–0.40 Hz), and
      (b) two interpolation methods (cubic spline vs linear).

    These columns are added by pipeline.py (compute_frequency_domain).
    If the pipeline has not been re-run since adding the new columns
    this function prints a clear reminder and falls back gracefully.

    Outputs
    -------
    PSD_Beta_Range_Sensitivity.csv  — r, p, significance per variant
    PSD_Beta_Range_Sensitivity.png  — grouped bar chart
    """
    # Map: display label → column name in CSV
    variants = {
        "Primary\n(0.01–0.15 Hz,\ncubic)":    "PSD_Beta",
        "Wide VLF–LF\n(0.003–0.15 Hz,\ncubic)": "PSD_Beta_Wide",
        "Full spectrum\n(0.003–0.40 Hz,\ncubic)": "PSD_Beta_Full",
        "Linear interp\n(0.01–0.15 Hz,\nlinear)": "PSD_Beta_Linear",
    }

    # Check which columns are actually present
    available = {lbl: col for lbl, col in variants.items()
                 if col in df_h.columns and col in df_p.columns}

    if len(available) <= 1:
        print("  PSD β range sensitivity: only primary β column found.\n"
              "  → Re-run healthy_hrvacc.py and psych_hrvacc.py with the updated\n"
              "    pipeline.py to populate PSD_Beta_Wide / Full / Linear\n"
              "    columns, then re-run comparison.py.")
        # Still run the R²-filtering sensitivity that was here before
        psd_beta_sensitivity(df_h, df_p, stats, output_dir)
        return

    rows = []
    for lbl, col in available.items():
        h_v = pd.to_numeric(df_h[col], errors='coerce').dropna().values
        p_v = pd.to_numeric(df_p[col], errors='coerce').dropna().values
        if len(h_v) < 3 or len(p_v) < 3:
            continue
        u, p_raw = mannwhitneyu(h_v, p_v, alternative='two-sided')
        r        = rank_biserial_r(u, len(h_v), len(p_v))
        h_mean   = np.mean(h_v); h_sd = np.std(h_v, ddof=1)
        p_mean   = np.mean(p_v); p_sd = np.std(p_v, ddof=1)
        rows.append({
            "Variant":      lbl.replace("\n", " "),
            "Column":       col,
            "H_Mean_SD":    f"{h_mean:.3f} ± {h_sd:.3f}",
            "P_Mean_SD":    f"{p_mean:.3f} ± {p_sd:.3f}",
            "r_effect":     r,
            "p_raw":        p_raw,
        })

    if not rows:
        return

    sens_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "PSD_Beta_Range_Sensitivity.csv")
    sens_df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"\n  PSD BETA RANGE & PREPROCESSING SENSITIVITY")
    print(f"  {'Variant':<45} {'H Mean±SD':>15} {'P Mean±SD':>15}"
          f" {'r':>6} {'p_raw':>8}")
    print("  " + "-" * 95)
    for _, row in sens_df.iterrows():
        print(f"  {row['Variant']:<45} {row['H_Mean_SD']:>15} {row['P_Mean_SD']:>15}"
              f" {row['r_effect']:>6.3f} {row['p_raw']:>8.4f}")
    print(f"  Saved → {csv_path}")

    # Figure: r-values across variants
    labels = [r["Variant"].replace(" ", "\n") for r in rows]
    r_vals = [r["r_effect"] for r in rows]
    colors = [C_PSYCH if r["p_raw"] < 0.05 else "lightsteelblue" for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, r_vals, color=colors, edgecolor='white', width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    for bar, val in zip(bars, r_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"r = {val:.3f}", ha='center', va='bottom', fontsize=8.5)
    ax.set_ylabel("Rank-biserial r (healthy vs psychiatric)", fontsize=9)
    ax.set_title(
        "PSD β Sensitivity: Frequency Fitting Range & Interpolation Method\n"
        "(red = p < 0.05 uncorrected; consistency across variants → robust finding)",
        fontsize=9
    )
    ax.set_ylim(0, max(r_vals) + 0.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "PSD_Beta_Range_Sensitivity.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Range sensitivity plot → {fig_path}")

    # Also run the original R²-filtering check
    psd_beta_sensitivity(df_h, df_p, stats, output_dir)


# ═══════════════════════════════════════════════════════════════
# STEP 9 — PSD BETA R²-FILTER SENSITIVITY (unchanged)
# ═══════════════════════════════════════════════════════════════

def psd_beta_sensitivity(df_h, df_p, stats, output_dir):
    """
    Re-run the group comparison on the subset of subjects where
    PSD_Beta_R2 >= PSD_BETA_R2_THRESHOLD in BOTH groups.
    Report whether significance and effect sizes change materially.
    """
    if "PSD_Beta_R2" not in df_h.columns or "PSD_Beta_R2" not in df_p.columns:
        print("  PSD beta sensitivity: PSD_Beta_R2 missing — skipped.")
        return

    h_filt = df_h[pd.to_numeric(df_h["PSD_Beta_R2"], errors='coerce')
                  >= PSD_BETA_R2_THRESHOLD]
    p_filt = df_p[pd.to_numeric(df_p["PSD_Beta_R2"], errors='coerce')
                  >= PSD_BETA_R2_THRESHOLD]

    print(f"\n  PSD Beta sensitivity: "
          f"Healthy {len(h_filt)}/{len(df_h)} pass, "
          f"Psychiatric {len(p_filt)}/{len(df_p)} pass")

    if len(h_filt) < 3 or len(p_filt) < 3:
        print("  Too few subjects after filtering — sensitivity check skipped.")
        return

    stats_filt = run_group_comparison(h_filt, p_filt,
                                      [m for m in METRIC_BATTERY
                                       if m in h_filt.columns])

    # Side-by-side comparison for PSD_Beta and radar metrics
    compare_metrics = [m for m in ["PSD_Beta"] + RADAR_METRICS
                       if m in stats.index and m in stats_filt.index]

    rows = []
    for m in compare_metrics:
        rows.append({
            "Metric":         METRIC_LABELS.get(m, m),
            "r_full":         stats.loc[m, "r_effect"],
            "p_full":         stats.loc[m, "p_adj"],
            "sig_full":       stats.loc[m, "Significant"],
            "r_filtered":     stats_filt.loc[m, "r_effect"],
            "p_filtered":     stats_filt.loc[m, "p_adj"],
            "sig_filtered":   stats_filt.loc[m, "Significant"],
        })

    sens_df = pd.DataFrame(rows).set_index("Metric")
    path    = os.path.join(output_dir, "PSD_Beta_Sensitivity.csv")
    sens_df.to_csv(path, float_format="%.4f")
    print(f"  Sensitivity CSV → {path}")
    print("\n  PSD BETA SENSITIVITY (full vs R²-filtered)")
    print(sens_df.to_string())


# ═══════════════════════════════════════════════════════════════
# STEP 10 (B) — EXPLORATORY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classification_analysis(df_h, df_p, stats, output_dir):
    """
    Leave-one-out logistic regression on the significant metrics.
    Reports accuracy, sensitivity, specificity, and AUC.
    ROC curves saved for each significant metric individually and
    for the multi-metric classifier.

    Framed explicitly as EXPLORATORY — sample size is too small
    for a definitive classifier. Used to estimate effect size and
    direction for powering the full study.
    """
    sig_metrics = [m for m in stats.index[stats["Significant"]]
                   if m in df_h.columns and m in df_p.columns]

    if not sig_metrics:
        print("  Classification: no significant metrics — skipped.")
        return

    # Build combined dataset
    all_df  = pd.concat([df_h, df_p], axis=0)
    labels  = np.array([0] * len(df_h) + [1] * len(df_p))  # 0=H, 1=P

    X_full  = all_df[sig_metrics].apply(pd.to_numeric, errors='coerce')
    valid   = X_full.dropna()
    y_valid = labels[valid.index.isin(all_df.index) &
                     np.array([i < len(all_df) for i in range(len(all_df))])]

    # Re-align labels to valid rows
    valid_idx = [i for i, idx in enumerate(all_df.index) if idx in valid.index]
    y_valid   = labels[valid_idx]

    if len(np.unique(y_valid)) < 2 or len(valid) < 5:
        print("  Classification: insufficient data after dropna — skipped.")
        return

    # ── Multi-metric LOO classifier ───────────────────────────
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(valid.values)
    loo    = LeaveOneOut()
    clf    = LogisticRegression(max_iter=1000, C=0.1, random_state=42)

    y_pred  = []
    y_prob  = []
    for train_idx, test_idx in loo.split(X_sc):
        clf.fit(X_sc[train_idx], y_valid[train_idx])
        y_pred.append(clf.predict(X_sc[test_idx])[0])
        y_prob.append(clf.predict_proba(X_sc[test_idx])[0, 1])

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc  = accuracy_score(y_valid, y_pred)
    auc  = roc_auc_score(y_valid, y_prob)
    cm   = confusion_matrix(y_valid, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print(f"\n  EXPLORATORY CLASSIFICATION (LOO Logistic Regression)")
    print(f"  Features: {[METRIC_LABELS.get(m,m) for m in sig_metrics]}")
    print(f"  Accuracy={acc:.3f}  AUC={auc:.3f}  "
          f"Sensitivity={sens:.3f}  Specificity={spec:.3f}")
    print(f"  NOTE: N={len(valid)} subjects — exploratory only.")

    # Save classifier summary
    clf_summary = pd.DataFrame([{
        "N_subjects":   len(valid),
        "N_features":   len(sig_metrics),
        "Features":     ", ".join(sig_metrics),
        "Accuracy":     acc,
        "AUC":          auc,
        "Sensitivity":  sens,
        "Specificity":  spec,
        "Note":         "LOO logistic regression, exploratory only"
    }])
    clf_summary.to_csv(
        os.path.join(output_dir, "Classification_Summary.csv"), index=False
    )

    # ── ROC figure ────────────────────────────────────────────
    ncols = min(len(sig_metrics) + 1, 4)
    nrows = int(np.ceil((len(sig_metrics) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Panel 0: multi-metric
    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    axes_flat[0].plot(fpr, tpr, color='purple', lw=2,
                      label=f"Multi-metric AUC={auc:.3f}")
    axes_flat[0].plot([0,1],[0,1],'k--', lw=0.8)
    axes_flat[0].set_title("Multi-metric (LOO LR)", fontsize=9)
    axes_flat[0].set_xlabel("FPR"); axes_flat[0].set_ylabel("TPR")
    axes_flat[0].legend(fontsize=7); axes_flat[0].grid(True, alpha=0.3)

    # Individual metric ROC curves
    for i, metric in enumerate(sig_metrics, start=1):
        h_vals = pd.to_numeric(df_h[metric], errors='coerce').dropna().values
        p_vals = pd.to_numeric(df_p[metric], errors='coerce').dropna().values
        y_bin  = np.array([0]*len(h_vals) + [1]*len(p_vals))
        scores = np.concatenate([h_vals, p_vals])
        valid_mask = np.isfinite(scores)
        if valid_mask.sum() < 4 or len(np.unique(y_bin[valid_mask])) < 2:
            axes_flat[i].set_visible(False); continue
        try:
            auc_i = roc_auc_score(y_bin[valid_mask], scores[valid_mask])
            fpr_i, tpr_i, _ = roc_curve(y_bin[valid_mask], scores[valid_mask])
            axes_flat[i].plot(fpr_i, tpr_i, color=C_PSYCH, lw=2,
                              label=f"AUC={auc_i:.3f}")
            axes_flat[i].plot([0,1],[0,1],'k--', lw=0.8)
            axes_flat[i].set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
            axes_flat[i].set_xlabel("FPR"); axes_flat[i].legend(fontsize=7)
            axes_flat[i].grid(True, alpha=0.3)
        except Exception:
            axes_flat[i].set_visible(False)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "ROC Curves — Exploratory Classification\n"
        f"(LOO Logistic Regression; N={len(valid)}; interpretive only)",
        fontsize=10
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "ROC_Curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curves      → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 11 (C) — POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════

def power_analysis(stats, output_dir, healthy_label="Healthy",
                   healthy_n=None, psych_n=None):
    """
    For each metric, estimate the sample size required per group to
    achieve 80% power at α=0.05 (two-tailed, uncorrected) given the
    observed rank-biserial correlation r.

    Converts r to Cohen's d via:
        d = 2r / sqrt(1 - r²)
    then uses the standard formula for two-sample t-test power
    (conservative approximation for Mann-Whitney):
        n = 2 * ((z_α/2 + z_β) / d)²
    where z_α/2 = 1.96 and z_β = 0.842 for 80% power.

    Results tell you: "to replicate this effect in the full study
    with adequate power, you need N per group."
    """
    z_alpha = norm.ppf(1 - 0.05 / 2)   # 1.96
    z_beta  = norm.ppf(0.80)            # 0.842

    rows = []
    for metric, row in stats.iterrows():
        r = abs(row["r_effect"])
        if r >= 1.0 or r == 0:
            n_per_group = np.nan
        else:
            d           = 2 * r / np.sqrt(1 - r**2)
            n_per_group = int(np.ceil(2 * ((z_alpha + z_beta) / d) ** 2))
        rows.append({
            "Metric":       METRIC_LABELS.get(metric, metric),
            "r_effect":     row["r_effect"],
            "Cohen_d_est":  2*abs(row["r_effect"]) / np.sqrt(1-row["r_effect"]**2)
                            if abs(row["r_effect"]) < 1 else np.nan,
            "N_per_group":  n_per_group,
            "Total_N":      n_per_group * 2 if not np.isnan(n_per_group) else np.nan,
            "Significant":  row["Significant"],
        })

    power_df = pd.DataFrame(rows).set_index("Metric")
    power_df = power_df.sort_values("N_per_group")
    path_csv = os.path.join(output_dir, "Power_Analysis.csv")
    power_df.to_csv(path_csv, float_format="%.2f")
    print(f"  Power analysis  → {path_csv}")

    print("\n  POWER ANALYSIS (80% power, α=0.05, two-tailed)")
    print(f"  {'Metric':<26} {'r':>6} {'d_est':>7} "
          f"{'N/group':>8} {'Total N':>8} {'Sig':>4}")
    print("  " + "-"*60)
    for metric, row in power_df.iterrows():
        sig = "★" if row["Significant"] else ""
        n_str = f"{int(row['N_per_group'])}" if not np.isnan(row["N_per_group"]) else "n/a"
        t_str = f"{int(row['Total_N'])}"     if not np.isnan(row["Total_N"])     else "n/a"
        print(f"  {metric:<26} {row['r_effect']:>6.3f} "
              f"{row['Cohen_d_est']:>7.3f} {n_str:>8} {t_str:>8} {sig:>4}")

    # Bar chart — N per group for each metric
    valid_pw = power_df.dropna(subset=["N_per_group"]).copy()
    valid_pw = valid_pw.sort_values("N_per_group", ascending=False)
    colors   = [C_PSYCH if s else "lightsteelblue"
                for s in valid_pw["Significant"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(valid_pw) * 0.4)))
    bars = ax.barh(valid_pw.index,
                   valid_pw["N_per_group"].astype(float),
                   color=colors, edgecolor='white', height=0.7)
    if psych_n is not None:
        ax.axvline(psych_n, color='grey', lw=1.0, linestyle='--', alpha=0.7,
                   label=f'Current N (psychiatric={psych_n})')
    if healthy_n is not None:
        ax.axvline(healthy_n, color='steelblue', lw=1.0, linestyle='--', alpha=0.7,
                   label=f'Current N ({healthy_label}={healthy_n})')
    for bar, val in zip(bars, valid_pw["N_per_group"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{int(val)}", va='center', fontsize=7.5)
    ax.set_xlabel("Required N per group (80% power, α=0.05)")
    ax.set_title(
        "Power Analysis: Sample Size Required for Full Study\n"
        f"(red = significant in pilot; based on pilot effect sizes)",
        fontsize=10
    )
    ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    path_fig = os.path.join(output_dir, "Power_Analysis.png")
    plt.savefig(path_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Power analysis  → {path_fig}")


# ═══════════════════════════════════════════════════════════════
# STEP 10b — METRIC INTERCORRELATION TABLE  (Category 2, Analysis 3)
# ═══════════════════════════════════════════════════════════════

def metric_intercorrelation_table(df_h, df_p, stats, output_dir, healthy_label="Healthy"):
    """
    Compute pairwise Spearman correlations among all significant metrics
    within each cohort, and identify highly correlated pairs (|r| > 0.7).

    Addresses Reviewer 2's concern that significant findings across
    multiple analytic domains may reflect correlated descriptions of the
    same underlying signal property rather than independent confirmations.

    The table makes the covariance structure explicit so reviewers and
    readers can see which metrics are mathematically quasi-redundant
    (e.g. RMSSD / SD1 are arithmetically equivalent; Total Power / LF
    Power are strongly correlated) and which are genuinely independent.

    Outputs
    -------
    Metric_Intercorrelation_Healthy.csv
    Metric_Intercorrelation_Psychiatric.csv
    Metric_Intercorrelation_HighPairs.csv  — pairs with |r| > 0.7 flagged
    Metric_Intercorrelation.png            — side-by-side heatmap
    """
    sig_metrics = [m for m in stats.index[stats["Significant"]]
                   if m in df_h.columns and m in df_p.columns]

    if len(sig_metrics) < 3:
        print("  Metric intercorrelation: fewer than 3 significant metrics.")
        return

    labels = [METRIC_LABELS.get(m, m) for m in sig_metrics]

    h_sub  = df_h[sig_metrics].apply(pd.to_numeric, errors='coerce')
    p_sub  = df_p[sig_metrics].apply(pd.to_numeric, errors='coerce')

    # Spearman correlation matrices
    h_corr = h_sub.corr(method='spearman')
    p_corr = p_sub.corr(method='spearman')

    # Save numeric tables
    h_corr.index = h_corr.columns = labels
    p_corr.index = p_corr.columns = labels
    h_corr.to_csv(os.path.join(output_dir,
                                "Metric_Intercorrelation_Healthy.csv"),
                  float_format="%.3f")
    p_corr.to_csv(os.path.join(output_dir,
                                "Metric_Intercorrelation_Psychiatric.csv"),
                  float_format="%.3f")

    # Flag high-correlation pairs in both cohorts
    high_pairs = []
    for i, m1 in enumerate(sig_metrics):
        for j, m2 in enumerate(sig_metrics):
            if j <= i:
                continue
            rh = h_sub[m1].corr(h_sub[m2], method='spearman')
            rp = p_sub[m1].corr(p_sub[m2], method='spearman')
            if abs(rh) > 0.70 or abs(rp) > 0.70:
                high_pairs.append({
                    "Metric_1":    METRIC_LABELS.get(m1, m1),
                    "Metric_2":    METRIC_LABELS.get(m2, m2),
                    "r_Healthy":   round(rh, 3),
                    "r_Psych":     round(rp, 3),
                    "Note":        ("|r|>0.7 in both" if abs(rh)>0.70 and abs(rp)>0.70
                                    else "|r|>0.7 in healthy" if abs(rh)>0.70
                                    else "|r|>0.7 in psych")
                })

    if high_pairs:
        hp_df = pd.DataFrame(high_pairs)
        hp_path = os.path.join(output_dir, "Metric_Intercorrelation_HighPairs.csv")
        hp_df.to_csv(hp_path, index=False, float_format="%.3f")
        print(f"\n  METRIC INTERCORRELATION — High-correlation pairs (|r| > 0.70)")
        print(hp_df.to_string(index=False))
        print(f"  Saved → {hp_path}")

    # ── Side-by-side heatmap ──────────────────────────────────
    n = len(sig_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 0.8 * 2), max(6, n * 0.7)))
    for ax, corr_m, title, n_sub in zip(
        axes,
        [h_corr, p_corr],
        [f"{healthy_label} (N={len(df_h)})", f"Psychiatric (N={len(df_p)})"],
        [len(df_h), len(df_p)]
    ):
        im = ax.imshow(corr_m.values, vmin=-1, vmax=1,
                       cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"Metric Intercorrelations — {title}", fontsize=9)
        for i in range(n):
            for j in range(n):
                val = corr_m.values[i, j]
                if np.isfinite(val):
                    col = 'white' if abs(val) > 0.6 else 'black'
                    ax.text(j, i, f"{val:.2f}",
                            ha='center', va='center', fontsize=6, color=col)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")

    fig.suptitle(
        "Pairwise Spearman Correlations — All Significant Metrics\n"
        "(reveals quasi-redundant pairs; see HighPairs CSV for |r|>0.70 flagged)",
        fontsize=10
    )
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Metric_Intercorrelation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Intercorrelation heatmap → {fig_path}")


# ═══════════════════════════════════════════════════════════════
# STEP 12 — WITHIN-GROUP CORRELATION HEATMAPS (expanded)
# ═══════════════════════════════════════════════════════════════

def plot_within_group_correlations(df_h, df_p, output_dir, stats=None, healthy_label="Healthy"):
    """
    Side-by-side Pearson correlation heatmaps.

    If stats is provided, shows all significant metrics (sorted by |r|)
    so the full inter-metric covariance structure is visible.
    Falls back to the five radar metrics if stats is not supplied.

    This addresses Reviewer 2's concern that significant findings
    across multiple domains may reflect shared signal rather than
    independent confirmation — the heatmap makes this explicit.
    """
    if stats is not None:
        sig = [m for m in stats.index[stats["Significant"]]
               if m in df_h.columns and m in df_p.columns]
        # Sort by |r| descending for a coherent display order
        sig = sorted(sig,
                     key=lambda m: abs(stats.loc[m, "r_effect"]),
                     reverse=True)
        metrics = sig if sig else RADAR_METRICS
    else:
        metrics = [m for m in RADAR_METRICS
                   if m in df_h.columns and m in df_p.columns]

    labels  = [METRIC_LABELS.get(m, m) for m in metrics]
    h_corr  = df_h[metrics].apply(pd.to_numeric, errors='coerce').corr()
    p_corr  = df_p[metrics].apply(pd.to_numeric, errors='coerce').corr()

    n = len(metrics)
    cell_w = max(0.55, 7.0 / n)   # scale figure width with metric count
    fig, axes = plt.subplots(1, 2, figsize=(n * cell_w * 2 + 2, max(5, n * 0.55)))

    for ax, corr, title, n_sub in zip(
        axes,
        [h_corr, p_corr],
        [healthy_label, "Psychiatric"],
        [len(df_h), len(df_p)]
    ):
        im = ax.imshow(corr.values, vmin=-1, vmax=1,
                       cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{title} Correlation Structure (n={n_sub})", fontsize=10)
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                if np.isfinite(val):
                    c = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f"{val:.2f}",
                            ha='center', va='center', fontsize=6.5, color=c)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    fig.suptitle(
        "Within-Group Correlation Structure — All Significant Metrics\n"
        "(same color scale; differences suggest altered co-variation "
        "in psychiatric illness)",
        fontsize=10
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "Within_Group_Correlations.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Corr structure  → {path}")


# ═══════════════════════════════════════════════════════════════
# ADDED ANALYSES (strictly additive — the originals above are
# untouched in substance). New: parametric tests, distribution
# diagnostics, bootstrap effect-size CIs, control-vs-control confound
# quantification, TOST equivalence, three-group omnibus, and
# cross-control convergence.
# ═══════════════════════════════════════════════════════════════

# Extra colors for the dual-control / three-group figures
C_MITBIH = "teal"        # MIT-BIH NSR healthy (supplemental control)
C_HRVACC = "steelblue"   # HRV-ACC healthy (primary, recording-matched control)
# C_PSYCH already defined above (firebrick)


# ── HRV-ACC healthy per-subject demographics (PLACEHOLDER) ────────
# The HRV-ACC source paper reports control demographics only at the GROUP
# level (N=30; 17F/13M; age 43.2 ± 14.1, range 24–69). If the Zenodo metadata
# provides a per-subject mapping (control_N → age, sex), paste it here as
#   {"control_2": {"Age": __, "Sex": 0/1}, ...}    (Sex: 0 = male, 1 = female)
# to enable the within-healthy age/sex covariate analysis for this cohort.
# Leave empty to skip that sub-analysis (everything else still runs).
HRVACC_HEALTHY_DEMOGRAPHICS = {
    # "control_2":  {"Age": __, "Sex": _},
    # "control_16": {"Age": __, "Sex": _},
}


def _cohens_d_ci(x_ref, x_other):
    """
    Cohen's d, Hedges' g (small-sample corrected), and an approximate
    95% CI for d, for two independent samples.

    Sign convention: d = (mean_other - mean_ref) / pooled_sd, so a POSITIVE
    d means `x_other` is higher than `x_ref`. Called as (ref=healthy,
    other=psychiatric) this matches the codebase convention
    "positive effect = psychiatric > healthy" used for rank-biserial r.

    Returns (d, g, ci_low, ci_high); NaNs if pooled SD is zero.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_other = np.asarray(x_other, dtype=float)
    n1, n2 = len(x_ref), len(x_other)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan
    s1, s2 = np.std(x_ref, ddof=1), np.std(x_other, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan, np.nan, np.nan, np.nan
    d = (np.mean(x_other) - np.mean(x_ref)) / sp
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return float(d), float(g), float(d - 1.96 * se), float(d + 1.96 * se)


# ═══════════════════════════════════════════════════════════════
# STEP 13 — PARAMETRIC COMPARISON (Welch t-test + Cohen's d)
# ═══════════════════════════════════════════════════════════════

def run_parametric_comparison(df_h, df_p, metrics):
    """
    Welch's two-sample t-test (unequal variances) per metric, with Cohen's d,
    Hedges' g, and an approximate 95% CI for d; BH-FDR across metrics.
    Provided ALONGSIDE the nonparametric Mann-Whitney U table so the two
    approaches can be cross-checked and so standardized mean differences
    (comparable to the wider literature) are available.

    Positive t / positive d = psychiatric group higher than healthy.
    """
    rows = []
    for m in metrics:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        h = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p = pd.to_numeric(df_p[m], errors='coerce').dropna().values
        if len(h) < 3 or len(p) < 3:
            continue
        t_stat, p_raw = ttest_ind(p, h, equal_var=False)   # Welch
        d, g, lo, hi = _cohens_d_ci(h, p)                   # ref=h, other=p
        rows.append({
            "Metric":       m,
            "Healthy_Mean": float(np.mean(h)),
            "Healthy_SD":   float(np.std(h, ddof=1)),
            "Psych_Mean":   float(np.mean(p)),
            "Psych_SD":     float(np.std(p, ddof=1)),
            "Welch_t":      float(t_stat),
            "p_raw":        float(p_raw),
            "Cohen_d":      d,
            "Hedges_g":     g,
            "d_CI_low":     lo,
            "d_CI_high":    hi,
        })

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows).set_index("Metric")
    res["p_adj"]       = benjamini_hochberg(res["p_raw"].values)
    res["Significant"] = res["p_adj"] < ALPHA
    return res.sort_values("p_adj")


def save_parametric_table(par_stats, output_dir, healthy_label="Healthy"):
    """Save the parametric results table and print a console summary."""
    if par_stats is None or par_stats.empty:
        print("  Parametric comparison: no metrics — skipped.")
        return
    out = par_stats.copy()
    out.index = [METRIC_LABELS.get(m, m) for m in out.index]
    path = os.path.join(output_dir, "Parametric_Comparison_Welch_t.csv")
    out.to_csv(path, float_format="%.4f")

    print(f"\n  PARAMETRIC COMPARISON — Welch t-test + Cohen's d "
          f"({healthy_label} vs Psychiatric, BH-FDR α={ALPHA})")
    print(f"  {'Metric':<26} {'d':>7} {'95% CI':>17} {'t':>7} "
          f"{'p_adj':>8} {'Sig':>4}")
    print("  " + "-" * 74)
    for m, row in par_stats.iterrows():
        sig = "★" if row["Significant"] else ""
        ci  = f"[{row['d_CI_low']:.2f}, {row['d_CI_high']:.2f}]"
        print(f"  {METRIC_LABELS.get(m, m):<26} {row['Cohen_d']:>7.3f} "
              f"{ci:>17} {row['Welch_t']:>7.2f} {row['p_adj']:>8.4f} {sig:>4}")
    print(f"  Saved → {path}")


def plot_cohens_d_forest(par_stats, output_dir, healthy_label="Healthy"):
    """
    Forest plot of Cohen's d with analytic 95% CIs, one row per metric,
    sorted by |d|. Significant metrics (BH-FDR) in red. Vertical guides at
    |d| = 0.2 / 0.5 / 0.8 (small / medium / large).
    """
    if par_stats is None or par_stats.empty:
        return
    df = par_stats.copy()
    df["abs_d"] = df["Cohen_d"].abs()
    df = df.sort_values("abs_d", ascending=True)
    labels = [METRIC_LABELS.get(m, m) for m in df.index]
    colors = [C_PSYCH if s else "lightsteelblue" for s in df["Significant"]]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.4)))
    ax.errorbar(df["Cohen_d"], y,
                xerr=[df["Cohen_d"] - df["d_CI_low"],
                      df["d_CI_high"] - df["Cohen_d"]],
                fmt='none', ecolor='grey', elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(df["Cohen_d"], y, c=colors, s=55, zorder=3, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    for v, ls in [(0.2, ':'), (0.5, '--'), (0.8, '-.')]:
        ax.axvline(v,  color='grey', lw=0.8, ls=ls, alpha=0.6)
        ax.axvline(-v, color='grey', lw=0.8, ls=ls, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cohen's d  (positive = Psychiatric > healthy)", fontsize=9)
    ax.set_title(
        f"Standardized Mean Differences (Cohen's d, 95% CI)\n"
        f"{healthy_label} vs Psychiatric — red = significant after BH-FDR; "
        "guides at |d| = 0.2/0.5/0.8",
        fontsize=9.5)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "Cohens_d_Forest.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cohen's d forest → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 14 — DISTRIBUTION DIAGNOSTICS (normality + variance)
# ═══════════════════════════════════════════════════════════════

def distribution_diagnostics(df_h, df_p, metrics, output_dir,
                             healthy_label="Healthy"):
    """
    Per metric: Shapiro-Wilk normality p within each group and Levene's test
    for equality of variance between groups. Documents, per metric, whether
    the parametric (Welch t) or nonparametric (Mann-Whitney U) result should
    be trusted, and flags metrics whose variances differ (variance
    compression is itself a criticality-relevant signal).

    Outputs
    -------
    Distribution_Diagnostics.csv
    Distribution_Diagnostics_Levene.png
    """
    rows = []
    for m in metrics:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        h = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p = pd.to_numeric(df_p[m], errors='coerce').dropna().values
        if len(h) < 3 or len(p) < 3:
            continue
        sh = float(shapiro(h).pvalue)
        sp = float(shapiro(p).pvalue)
        try:
            _, p_lev = levene(h, p)
            p_lev = float(p_lev)
        except Exception:
            p_lev = np.nan
        both_normal = (sh > 0.05) and (sp > 0.05)
        equal_var   = (p_lev > 0.05) if np.isfinite(p_lev) else False
        rec = ("Parametric (Welch t) valid"
               if both_normal else "Nonparametric (Mann-Whitney U) preferred")
        rows.append({
            "Metric":            METRIC_LABELS.get(m, m),
            "Shapiro_p_Healthy": sh,
            "Shapiro_p_Psych":   sp,
            "Healthy_Normal":    sh > 0.05,
            "Psych_Normal":      sp > 0.05,
            "Levene_p":          p_lev,
            "Equal_Variance":    equal_var,
            "Recommended_Test":  rec,
        })

    if not rows:
        print("  Distribution diagnostics: no metrics — skipped.")
        return pd.DataFrame()

    diag = pd.DataFrame(rows).set_index("Metric")
    path = os.path.join(output_dir, "Distribution_Diagnostics.csv")
    diag.to_csv(path, float_format="%.4f")
    print(f"\n  DISTRIBUTION DIAGNOSTICS ({healthy_label} vs Psychiatric)")
    print(f"  {'Metric':<26} {'Shap_H':>7} {'Shap_P':>7} "
          f"{'Levene':>7} {'EqVar':>6}")
    print("  " + "-" * 60)
    for m, r in diag.iterrows():
        lev = f"{r['Levene_p']:.3f}" if np.isfinite(r['Levene_p']) else "  —"
        print(f"  {m:<26} {r['Shapiro_p_Healthy']:>7.3f} "
              f"{r['Shapiro_p_Psych']:>7.3f} {lev:>7} "
              f"{('yes' if r['Equal_Variance'] else 'no'):>6}")
    print(f"  Saved → {path}")

    # Levene p figure (variance equality)
    fig, ax = plt.subplots(figsize=(8, max(4, len(diag) * 0.38)))
    yv = np.arange(len(diag))
    levp = diag["Levene_p"].fillna(1.0).values
    colors = [C_PSYCH if lp < 0.05 else "lightsteelblue" for lp in levp]
    ax.barh(yv, levp, color=colors, edgecolor='white', height=0.7)
    ax.axvline(0.05, color='black', lw=1.0, ls='--',
               label='α = 0.05 (variances differ to the left)')
    ax.set_yticks(yv)
    ax.set_yticklabels(diag.index, fontsize=8)
    ax.set_xlabel("Levene test p-value (equality of variance)", fontsize=9)
    ax.set_title(f"Variance-Equality Test — {healthy_label} vs Psychiatric\n"
                 "(red = variances significantly differ)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Distribution_Diagnostics_Levene.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Levene figure   → {fig_path}")
    return diag


# ═══════════════════════════════════════════════════════════════
# STEP 15 — BOOTSTRAP EFFECT-SIZE CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════

def bootstrap_effect_sizes(df_h, df_p, metrics, output_dir,
                           healthy_label="Healthy",
                           n_boot=2000, seed=0):
    """
    Percentile bootstrap 95% CIs for rank-biserial r and Cohen's d per
    metric. With small samples, analytic CIs lean on assumptions the data
    may not meet; resampling gives a distribution-free interval and a direct
    read on which effects are stable. A metric whose r CI excludes 0 is
    robustly nonzero regardless of the parametric/nonparametric debate.

    Outputs
    -------
    Bootstrap_Effect_Sizes.csv
    Bootstrap_r_Forest.png
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        h = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p = pd.to_numeric(df_p[m], errors='coerce').dropna().values
        if len(h) < 3 or len(p) < 3:
            continue
        try:
            u, _ = mannwhitneyu(h, p, alternative='two-sided')
            r_obs = rank_biserial_r(u, len(h), len(p))
        except ValueError:
            r_obs = np.nan
        d_obs, _, _, _ = _cohens_d_ci(h, p)

        r_bs, d_bs = [], []
        for _ in range(n_boot):
            hb = rng.choice(h, size=len(h), replace=True)
            pb = rng.choice(p, size=len(p), replace=True)
            try:
                ub, _ = mannwhitneyu(hb, pb, alternative='two-sided')
                r_bs.append(rank_biserial_r(ub, len(hb), len(pb)))
            except ValueError:
                pass
            db, _, _, _ = _cohens_d_ci(hb, pb)
            if np.isfinite(db):
                d_bs.append(db)

        if len(r_bs) < 50 or len(d_bs) < 50:
            continue
        r_lo, r_hi = np.percentile(r_bs, [2.5, 97.5])
        d_lo, d_hi = np.percentile(d_bs, [2.5, 97.5])
        rows.append({
            "Metric":      m,
            "r_effect":    float(r_obs),
            "r_CI_low":    float(r_lo),
            "r_CI_high":   float(r_hi),
            "Cohen_d":     float(d_obs),
            "d_CI_low":    float(d_lo),
            "d_CI_high":   float(d_hi),
            "r_CI_excl_0": bool(r_lo > 0 or r_hi < 0),
        })

    if not rows:
        print("  Bootstrap effect sizes: no metrics — skipped.")
        return pd.DataFrame()

    boot = pd.DataFrame(rows).set_index("Metric")
    boot["abs_r"] = boot["r_effect"].abs()
    boot = boot.sort_values("abs_r", ascending=False)
    out = boot.drop(columns="abs_r").copy()
    out.index = [METRIC_LABELS.get(m, m) for m in out.index]
    path = os.path.join(output_dir, "Bootstrap_Effect_Sizes.csv")
    out.to_csv(path, float_format="%.4f")
    print(f"\n  BOOTSTRAP EFFECT SIZES ({healthy_label} vs Psychiatric, "
          f"{n_boot} resamples, percentile 95% CI)")
    print(f"  {'Metric':<26} {'r':>7} {'r 95% CI':>17} {'CI≠0':>5}")
    print("  " + "-" * 60)
    for m, r in boot.iterrows():
        ci = f"[{r['r_CI_low']:.2f}, {r['r_CI_high']:.2f}]"
        flag = "yes" if r["r_CI_excl_0"] else ""
        print(f"  {METRIC_LABELS.get(m, m):<26} {r['r_effect']:>7.3f} "
              f"{ci:>17} {flag:>5}")
    print(f"  Saved → {path}")

    # Forest plot of r with bootstrap CI
    df = boot.sort_values("r_effect")
    labels = [METRIC_LABELS.get(m, m) for m in df.index]
    colors = [C_PSYCH if e else "lightsteelblue" for e in df["r_CI_excl_0"]]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.4)))
    ax.errorbar(df["r_effect"], y,
                xerr=[df["r_effect"] - df["r_CI_low"],
                      df["r_CI_high"] - df["r_effect"]],
                fmt='none', ecolor='grey', elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(df["r_effect"], y, c=colors, s=55, zorder=3, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rank-biserial r  (bootstrap 95% CI)", fontsize=9)
    ax.set_title(f"Bootstrap Effect Sizes — {healthy_label} vs Psychiatric\n"
                 "(red = 95% CI excludes 0; positive = Psychiatric > healthy)",
                 fontsize=9.5)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Bootstrap_r_Forest.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bootstrap forest → {fig_path}")
    return boot


# ═══════════════════════════════════════════════════════════════
# STEP 16 — CONTROL vs CONTROL (quantify the recording/device confound)
# ═══════════════════════════════════════════════════════════════

def control_vs_control_analysis(df_a, df_b, metrics, output_dir,
                                label_a="MIT-BIH NSR",
                                label_b="HRV-ACC Control"):
    """
    Direct comparison of the TWO healthy control cohorts. This MEASURES the
    cross-dataset / device / environment confound directly: for any metric
    where the two healthy groups differ, a healthy-vs-psychiatric difference
    on that metric using the MIT-BIH control is suspect; where they agree,
    the psychiatric effect is far more credible.

    Per metric: Mann-Whitney U (BH-FDR), Welch t, Cohen's d (with CI), Levene.

    Outputs
    -------
    Control_vs_Control_Stats.csv
    Control_vs_Control_AbsD.png
    Returns the per-metric DataFrame (indexed by raw metric key).
    """
    rows = []
    for m in metrics:
        if m not in df_a.columns or m not in df_b.columns:
            continue
        a = pd.to_numeric(df_a[m], errors='coerce').dropna().values
        b = pd.to_numeric(df_b[m], errors='coerce').dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            u, p_mwu = mannwhitneyu(a, b, alternative='two-sided')
            r = rank_biserial_r(u, len(a), len(b))
        except ValueError:
            u, p_mwu, r = np.nan, np.nan, np.nan
        t_stat, p_t = ttest_ind(b, a, equal_var=False)
        d, g, lo, hi = _cohens_d_ci(a, b)             # ref=A, other=B
        try:
            _, p_lev = levene(a, b)
            p_lev = float(p_lev)
        except Exception:
            p_lev = np.nan
        rows.append({
            "Metric":      m,
            "A_Mean":      float(np.mean(a)),
            "A_SD":        float(np.std(a, ddof=1)),
            "B_Mean":      float(np.mean(b)),
            "B_SD":        float(np.std(b, ddof=1)),
            "MWU_p_raw":   float(p_mwu),
            "r_effect":    float(r),
            "Welch_t":     float(t_stat),
            "Welch_p":     float(p_t),
            "Cohen_d":     d,
            "d_CI_low":    lo,
            "d_CI_high":   hi,
            "Levene_p":    p_lev,
        })

    if not rows:
        print("  Control vs control: no metrics — skipped.")
        return pd.DataFrame()

    cvc = pd.DataFrame(rows).set_index("Metric")
    cvc["MWU_p_adj"] = benjamini_hochberg(cvc["MWU_p_raw"].values)
    cvc["Differs"]   = cvc["MWU_p_adj"] < ALPHA
    cvc = cvc.sort_values("MWU_p_adj")

    out = cvc.copy()
    out.index = [METRIC_LABELS.get(m, m) for m in out.index]
    path = os.path.join(output_dir, "Control_vs_Control_Stats.csv")
    out.to_csv(path, float_format="%.4f")

    print(f"\n  CONTROL vs CONTROL — {label_a} vs {label_b} "
          f"(confound magnitude; BH-FDR α={ALPHA})")
    print(f"  {'Metric':<26} {'d':>7} {'r':>6} {'MWU p_adj':>10} "
          f"{'Levene p':>9} {'Differ':>7}")
    print("  " + "-" * 72)
    for m, row in cvc.iterrows():
        lev = f"{row['Levene_p']:.3f}" if np.isfinite(row['Levene_p']) else "  —"
        diff = "DIFFERS" if row["Differs"] else ""
        print(f"  {METRIC_LABELS.get(m, m):<26} {row['Cohen_d']:>7.3f} "
              f"{row['r_effect']:>6.3f} {row['MWU_p_adj']:>10.4f} "
              f"{lev:>9} {diff:>7}")
    n_diff = int(cvc["Differs"].sum())
    print(f"  {n_diff}/{len(cvc)} metrics differ between the control cohorts "
          f"(these carry a recording/device confound).")
    print(f"  Saved → {path}")

    # |Cohen's d| bar (confound magnitude)
    df = cvc.copy()
    df["abs_d"] = df["Cohen_d"].abs()
    df = df.sort_values("abs_d", ascending=True)
    labels = [METRIC_LABELS.get(m, m) for m in df.index]
    colors = [C_PSYCH if s else "lightsteelblue" for s in df["Differs"]]
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.4)))
    ax.barh(labels, df["abs_d"], color=colors, edgecolor='white', height=0.7)
    for v, ls in [(0.2, ':'), (0.5, '--'), (0.8, '-.')]:
        ax.axvline(v, color='grey', lw=0.8, ls=ls, alpha=0.6)
    ax.set_xlabel("|Cohen's d| between the two healthy control cohorts",
                  fontsize=9)
    ax.set_title(
        f"Confound Magnitude: {label_a} vs {label_b}\n"
        "(red = significant difference after BH-FDR; large |d| = metric is "
        "sensitive to recording setup)", fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Control_vs_Control_AbsD.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confound |d| bar → {fig_path}")
    return cvc


# ═══════════════════════════════════════════════════════════════
# STEP 17 — TOST EQUIVALENCE BETWEEN CONTROL COHORTS
# ═══════════════════════════════════════════════════════════════

def tost_equivalence(df_a, df_b, metrics, output_dir,
                     bound_d=0.5, alpha=0.05,
                     label_a="MIT-BIH NSR", label_b="HRV-ACC Control"):
    """
    Two One-Sided Tests (TOST) for EQUIVALENCE of the two control cohorts on
    each metric, with an equivalence margin in standardized units (bound_d,
    default 0.5 = "medium"). A significant TOST means the two control groups
    are statistically EQUIVALENT within ±bound_d — i.e. the recording
    confound on that metric is bounded/small, strengthening any
    healthy-vs-psychiatric claim there.

    NOTE: bound_d = 0.5 is a defensible default but IS a researcher choice;
    revisit it for the final paper. A non-significant difference test is NOT
    evidence of equivalence — only TOST provides that.

    Outputs: Control_Equivalence_TOST.csv
    """
    rows = []
    for m in metrics:
        if m not in df_a.columns or m not in df_b.columns:
            continue
        a = pd.to_numeric(df_a[m], errors='coerce').dropna().values
        b = pd.to_numeric(df_b[m], errors='coerce').dropna().values
        n1, n2 = len(a), len(b)
        if n1 < 3 or n2 < 3:
            continue
        s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        if not np.isfinite(sp) or sp == 0:
            continue
        diff  = np.mean(b) - np.mean(a)
        se    = sp * np.sqrt(1.0 / n1 + 1.0 / n2)
        dof   = n1 + n2 - 2
        bound = bound_d * sp                       # equivalence margin (raw)
        t_low  = (diff + bound) / se               # H0: diff <= -bound
        p_low  = 1 - t_dist.cdf(t_low, dof)
        t_high = (diff - bound) / se               # H0: diff >= +bound
        p_high = t_dist.cdf(t_high, dof)
        p_tost = max(p_low, p_high)
        rows.append({
            "Metric":     METRIC_LABELS.get(m, m),
            "Mean_Diff":  float(diff),
            "Cohen_d":    float(diff / sp),
            "Margin_d":   bound_d,
            "p_TOST":     float(p_tost),
            "Equivalent": bool(p_tost < alpha),
        })

    if not rows:
        print("  TOST equivalence: no metrics — skipped.")
        return pd.DataFrame()

    tost = pd.DataFrame(rows).set_index("Metric")
    path = os.path.join(output_dir, "Control_Equivalence_TOST.csv")
    tost.to_csv(path, float_format="%.4f")
    print(f"\n  CONTROL EQUIVALENCE (TOST, margin |d| < {bound_d}) — "
          f"{label_a} vs {label_b}")
    print(f"  {'Metric':<26} {'d':>7} {'p_TOST':>8} {'Equivalent':>11}")
    print("  " + "-" * 56)
    for m, row in tost.iterrows():
        eq = "EQUIVALENT" if row["Equivalent"] else ""
        print(f"  {m:<26} {row['Cohen_d']:>7.3f} {row['p_TOST']:>8.4f} "
              f"{eq:>11}")
    n_eq = int(tost["Equivalent"].sum())
    print(f"  {n_eq}/{len(tost)} metrics statistically equivalent between "
          f"controls within ±{bound_d} SD.")
    print(f"  Saved → {path}")
    return tost


# ═══════════════════════════════════════════════════════════════
# STEP 18 — THREE-GROUP OMNIBUS (Kruskal-Wallis + Dunn post-hoc)
# ═══════════════════════════════════════════════════════════════

def _dunn_posthoc(groups):
    """
    Dunn's post-hoc test (tie-corrected) for k independent groups. `groups`
    is a list of 1-D arrays. Returns (i, j, z, p_raw) for every pair i<j;
    p two-sided from the normal approximation. BH correction by the caller.
    """
    all_data = np.concatenate(groups)
    N = len(all_data)
    ranks = rankdata(all_data)
    _, counts = np.unique(all_data, return_counts=True)
    ties = np.sum(counts ** 3 - counts)
    sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))
    mean_ranks, ns, idx = [], [], 0
    for g in groups:
        rg = ranks[idx:idx + len(g)]
        mean_ranks.append(rg.mean()); ns.append(len(g)); idx += len(g)
    out = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(sigma2 * (1.0 / ns[i] + 1.0 / ns[j]))
            z  = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
            p  = 2 * norm.sf(abs(z))
            out.append((i, j, float(z), float(p)))
    return out


def three_group_analysis(df_mitbih, df_hrvacc, df_psych, metrics, output_dir):
    """
    Kruskal-Wallis omnibus across the THREE groups (MIT-BIH healthy, HRV-ACC
    healthy, psychiatric) per metric, BH-FDR across metrics. For significant
    omnibus metrics, Dunn's post-hoc gives the three pairwise comparisons
    with BH correction across all post-hoc tests. This is the cleanest single
    test of the central question: do the two control groups behave alike while
    BOTH differ from the patients?

    Outputs
    -------
    ThreeGroup_KruskalWallis.csv
    ThreeGroup_Dunn_Posthoc.csv
    ThreeGroup_Boxplots.png
    """
    group_labels = ["MIT-BIH", "HRV-ACC", "Psychiatric"]
    kw_rows, dunn_rows = [], []
    for m in metrics:
        if not all(m in d.columns for d in (df_mitbih, df_hrvacc, df_psych)):
            continue
        a = pd.to_numeric(df_mitbih[m], errors='coerce').dropna().values
        b = pd.to_numeric(df_hrvacc[m], errors='coerce').dropna().values
        c = pd.to_numeric(df_psych[m],  errors='coerce').dropna().values
        if min(len(a), len(b), len(c)) < 3:
            continue
        try:
            H, p_kw = kruskal(a, b, c)
        except ValueError:
            continue
        kw_rows.append({"Metric": m, "KW_H": float(H), "KW_p_raw": float(p_kw)})
        for (i, j, z, p) in _dunn_posthoc([a, b, c]):
            dunn_rows.append({
                "Metric": METRIC_LABELS.get(m, m),
                "Pair":   f"{group_labels[i]} vs {group_labels[j]}",
                "z":      z,
                "p_raw":  p,
            })

    if not kw_rows:
        print("  Three-group analysis: no metrics — skipped.")
        return

    kw = pd.DataFrame(kw_rows).set_index("Metric")
    kw["KW_p_adj"]    = benjamini_hochberg(kw["KW_p_raw"].values)
    kw["Significant"] = kw["KW_p_adj"] < ALPHA
    kw = kw.sort_values("KW_p_adj")
    kw_out = kw.copy()
    kw_out.index = [METRIC_LABELS.get(m, m) for m in kw_out.index]
    kw_path = os.path.join(output_dir, "ThreeGroup_KruskalWallis.csv")
    kw_out.to_csv(kw_path, float_format="%.4f")

    dunn = pd.DataFrame(dunn_rows)
    dunn_path = os.path.join(output_dir, "ThreeGroup_Dunn_Posthoc.csv")
    if not dunn.empty:
        dunn["p_adj"]       = benjamini_hochberg(dunn["p_raw"].values)
        dunn["Significant"] = dunn["p_adj"] < ALPHA
        dunn.to_csv(dunn_path, index=False, float_format="%.4f")

    print(f"\n  THREE-GROUP OMNIBUS (Kruskal-Wallis, BH-FDR α={ALPHA})")
    print(f"  {'Metric':<26} {'H':>8} {'p_adj':>8} {'Sig':>4}")
    print("  " + "-" * 50)
    for m, row in kw.iterrows():
        sig = "★" if row["Significant"] else ""
        print(f"  {METRIC_LABELS.get(m, m):<26} {row['KW_H']:>8.3f} "
              f"{row['KW_p_adj']:>8.4f} {sig:>4}")
    print(f"  KW table  → {kw_path}")
    if not dunn.empty:
        print(f"  Dunn table → {dunn_path}")

    # Three-group boxplots for the radar metrics
    rmetrics = [m for m in RADAR_METRICS
                if all(m in d.columns for d in (df_mitbih, df_hrvacc, df_psych))]
    if rmetrics:
        n = len(rmetrics)
        fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 4.6), squeeze=False)
        axes = axes.flatten()
        cols = [C_MITBIH, C_HRVACC, C_PSYCH]
        for ax, m in zip(axes, rmetrics):
            a = pd.to_numeric(df_mitbih[m], errors='coerce').dropna().values
            b = pd.to_numeric(df_hrvacc[m], errors='coerce').dropna().values
            c = pd.to_numeric(df_psych[m],  errors='coerce').dropna().values
            bp = ax.boxplot([a, b, c], positions=[1, 2, 3], widths=0.55,
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2))
            for patch, col in zip(bp['boxes'], cols):
                patch.set_facecolor(col); patch.set_alpha(0.55)
            rng = np.random.default_rng(0)
            for vals, pos, col in [(a, 1, C_MITBIH), (b, 2, C_HRVACC),
                                   (c, 3, C_PSYCH)]:
                ax.scatter(pos + rng.uniform(-0.12, 0.12, len(vals)), vals,
                           color=col, s=16, alpha=0.7, zorder=3)
            p_adj = kw.loc[m, "KW_p_adj"] if m in kw.index else np.nan
            star  = " ★" if (m in kw.index and kw.loc[m, "Significant"]) else ""
            ax.set_title(f"{METRIC_LABELS.get(m, m)}\nKW p={p_adj:.3f}{star}",
                         fontsize=9)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["MIT-BIH", "HRV-ACC", "Psych"],
                               fontsize=7, rotation=20)
            ax.grid(True, axis='y', alpha=0.3)
        fig.suptitle("Three-Group Comparison — Core Criticality Metrics\n"
                     "(two healthy controls vs psychiatric; "
                     "Kruskal-Wallis BH-FDR p shown)", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = os.path.join(output_dir, "ThreeGroup_Boxplots.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Three-group boxplots → {fig_path}")


# ═══════════════════════════════════════════════════════════════
# STEP 19 — CROSS-CONTROL CONVERGENCE (robustness across both controls)
# ═══════════════════════════════════════════════════════════════

def cross_control_convergence(stats_hrvacc, stats_mitbih, output_dir):
    """
    Merge the two healthy-vs-psychiatric result tables by metric and flag the
    effects that REPLICATE against BOTH control groups (significant after
    BH-FDR in both, same direction). This is the central rigor payoff of the
    dual-control design: an effect that survives against an independent,
    differently-recorded control group is far harder to dismiss as a
    dataset artifact.

    Outputs
    -------
    CrossControl_Convergence.csv
    CrossControl_r_vs_r.png   — r(HRV-ACC) vs r(MIT-BIH) scatter
    """
    common = [m for m in stats_hrvacc.index if m in stats_mitbih.index]
    rows = []
    for m in common:
        r_h = float(stats_hrvacc.loc[m, "r_effect"])
        r_b = float(stats_mitbih.loc[m, "r_effect"])
        s_h = bool(stats_hrvacc.loc[m, "Significant"])
        s_b = bool(stats_mitbih.loc[m, "Significant"])
        same_dir = (np.sign(r_h) == np.sign(r_b))
        rows.append({
            "Metric":         METRIC_LABELS.get(m, m),
            "r_HRVACC":       r_h,
            "p_adj_HRVACC":   float(stats_hrvacc.loc[m, "p_adj"]),
            "sig_HRVACC":     s_h,
            "r_MITBIH":       r_b,
            "p_adj_MITBIH":   float(stats_mitbih.loc[m, "p_adj"]),
            "sig_MITBIH":     s_b,
            "Same_Direction": same_dir,
            "Robust":         (s_h and s_b and same_dir),
        })

    if not rows:
        print("  Cross-control convergence: no shared metrics — skipped.")
        return pd.DataFrame()

    conv = pd.DataFrame(rows).set_index("Metric")
    conv = conv.sort_values(["Robust", "sig_HRVACC"], ascending=False)
    path = os.path.join(output_dir, "CrossControl_Convergence.csv")
    conv.to_csv(path, float_format="%.4f")

    robust = conv.index[conv["Robust"]].tolist()
    print(f"\n  CROSS-CONTROL CONVERGENCE "
          f"(effects replicating against BOTH control groups)")
    print(f"  {'Metric':<26} {'r_HRVACC':>9} {'r_MITBIH':>9} {'Robust':>7}")
    print("  " + "-" * 56)
    for m, row in conv.iterrows():
        flag = "ROBUST" if row["Robust"] else ""
        print(f"  {m:<26} {row['r_HRVACC']:>9.3f} {row['r_MITBIH']:>9.3f} "
              f"{flag:>7}")
    print(f"  {len(robust)}/{len(conv)} metrics are significant against BOTH "
          f"controls in the same direction.")
    print(f"  Saved → {path}")

    # r vs r scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    for m, row in conv.iterrows():
        col = C_PSYCH if row["Robust"] else "lightsteelblue"
        ax.scatter(row["r_HRVACC"], row["r_MITBIH"], s=60, color=col,
                   edgecolor='white', zorder=3)
        ax.annotate(m, (row["r_HRVACC"], row["r_MITBIH"]),
                    fontsize=6.5, alpha=0.8)
    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.6,
            label='y = x (perfect agreement)')
    ax.axhline(0, color='grey', lw=0.6); ax.axvline(0, color='grey', lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Rank-biserial r — HRV-ACC control vs Psychiatric", fontsize=9)
    ax.set_ylabel("Rank-biserial r — MIT-BIH control vs Psychiatric", fontsize=9)
    ax.set_title("Cross-Control Convergence of Effect Sizes\n"
                 "(red = significant vs BOTH controls; points near y=x and far "
                 "from origin = robust)", fontsize=9.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "CrossControl_r_vs_r.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Convergence scatter → {fig_path}")
    return conv


def effect_vs_confound(stats_hrvacc, cvc, output_dir):
    """
    Scatter each metric by (x) the confound magnitude |Cohen's d| between the
    two control cohorts and (y) the illness effect |r| (HRV-ACC vs psych).
    The trustworthy quadrant is low-confound / high-effect (bottom-right):
    a strong illness effect on a metric the two controls agree on.

    Output: EffectVsConfound.png
    """
    if stats_hrvacc is None or cvc is None or cvc.empty:
        return
    common = [m for m in stats_hrvacc.index if m in cvc.index]
    if not common:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for m in common:
        x = abs(float(cvc.loc[m, "Cohen_d"]))            # confound magnitude
        y = abs(float(stats_hrvacc.loc[m, "r_effect"]))  # illness effect
        sig = bool(stats_hrvacc.loc[m, "Significant"])
        col = C_PSYCH if sig else "lightsteelblue"
        ax.scatter(x, y, s=60, color=col, edgecolor='white', zorder=3)
        ax.annotate(METRIC_LABELS.get(m, m), (x, y), fontsize=6.5, alpha=0.8)
    ax.axvline(0.5, color='grey', lw=0.9, ls='--', alpha=0.7,
               label='|d| = 0.5 (medium confound)')
    ax.set_xlabel("Confound magnitude  |Cohen's d|  (MIT-BIH vs HRV-ACC controls)",
                  fontsize=9)
    ax.set_ylabel("Illness effect  |r|  (HRV-ACC control vs Psychiatric)",
                  fontsize=9)
    ax.set_title("Illness Effect vs Recording Confound\n"
                 "(bottom-right = strong effect on a confound-robust metric; "
                 "red = significant illness effect)", fontsize=9.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "EffectVsConfound.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Effect-vs-confound  → {fig_path}")


# ═══════════════════════════════════════════════════════════════
# MFDFA GROUP COMPARISON  [new analysis — Step 16]
# ═══════════════════════════════════════════════════════════════

_MFDFA_COLS = ["MFDFA_Width", "MFDFA_H_Mean", "MFDFA_Asymmetry", "MFDFA_H2"]

def mfdfa_group_comparison(df_h, df_p, output_dir, healthy_label="Healthy"):
    """
    Mann-Whitney U comparison of all four MFDFA scalars between cohorts.

    Scientific expectations
    -----------------------
    MFDFA_H_Mean   — expected HIGHER in patients (mirrors elevated PSD β / DFA α₂)
    MFDFA_H2       — expected HIGHER in patients (cross-check; should ≈ DFA_Alpha)
    MFDFA_Width    — directionality uncertain in this pilot; wider spectrum
                     indicates richer multifractal structure (possibly reduced
                     in patients reflecting more homogeneous dynamics, but
                     empirical evidence in psychiatric HRV is limited)
    MFDFA_Asymmetry — exploratory; no strong a-priori direction

    Saves:
        MFDFA_Group_Comparison.csv   — numerical results
        MFDFA_Group_Comparison.png   — four-panel box/strip figure
    """
    present = [m for m in _MFDFA_COLS
               if m in df_h.columns and m in df_p.columns]
    if not present:
        print("  [skip] MFDFA columns absent — re-run healthy_hrvacc.py and "
              "psych_hrvacc.py with the updated pipeline.py first.")
        return None

    rows = []
    for m in present:
        h_v = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p_v = pd.to_numeric(df_p[m], errors='coerce').dropna().values
        if len(h_v) < 3 or len(p_v) < 3:
            continue
        u, p_raw = mannwhitneyu(h_v, p_v, alternative='two-sided')
        r        = rank_biserial_r(u, len(h_v), len(p_v))
        rows.append({
            "Metric":      m,
            "Healthy_N":   len(h_v),
            "Healthy_Mean": np.mean(h_v),
            "Healthy_SD":  np.std(h_v, ddof=1),
            "Psych_N":     len(p_v),
            "Psych_Mean":  np.mean(p_v),
            "Psych_SD":    np.std(p_v, ddof=1),
            "Pct_Diff":    100 * (np.mean(p_v) - np.mean(h_v)) / abs(np.mean(h_v))
                           if np.mean(h_v) != 0 else np.nan,
            "r_effect":    r,
            "p_raw":       p_raw,
        })
    if not rows:
        return None

    res    = pd.DataFrame(rows).set_index("Metric")
    p_adj  = benjamini_hochberg(res["p_raw"].values)
    res["p_adj"]       = p_adj
    res["Significant"] = p_adj < ALPHA

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "MFDFA_Group_Comparison.csv")
    res.to_csv(csv_path, float_format="%.5f")
    print(f"  MFDFA stats      → {csv_path}")

    # ── Console summary ───────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MFDFA GROUP COMPARISON — {healthy_label} vs Psychiatric")
    print(f"{'='*72}")
    hdr = (f"  {'Metric':<22} {'H_Mean':>8} {'P_Mean':>8} "
           f"{'%Diff':>7} {'r':>6} {'p_raw':>8} {'p_adj':>8} {'Sig':>4}")
    print(hdr); print("  " + "-"*70)
    for m, row in res.iterrows():
        lbl = METRIC_LABELS.get(m, m)
        sig = "★" if row["Significant"] else ""
        print(f"  {lbl:<22} {row['Healthy_Mean']:>8.4f} {row['Psych_Mean']:>8.4f} "
              f"{row['Pct_Diff']:>7.1f} {row['r_effect']:>6.3f} "
              f"{row['p_raw']:>8.4f} {row['p_adj']:>8.4f} {sig:>4}")

    # ── Figure: box/strip plots ───────────────────────────────
    n     = len(rows)
    ncols = min(n, 4)
    nrows = -(-n // ncols)   # ceiling division
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 4.2),
                              squeeze=False)
    rng = np.random.default_rng(seed=0)
    axes_flat = axes.flatten()

    for i, m in enumerate([r["Metric"] for r in rows]):
        ax    = axes_flat[i]
        row   = res.loc[m]
        h_v   = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p_v   = pd.to_numeric(df_p[m], errors='coerce').dropna().values

        bp = ax.boxplot([h_v, p_v], positions=[1, 2], widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color='black', lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        bp['boxes'][0].set(facecolor=C_HEALTHY, alpha=0.55)
        bp['boxes'][1].set(facecolor=C_PSYCH,   alpha=0.55)

        for vals, pos, col in [(h_v, 1, C_HEALTHY), (p_v, 2, C_PSYCH)]:
            jit = rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(pos + jit, vals, color=col, s=25, alpha=0.75, zorder=3)

        lbl = METRIC_LABELS.get(m, m)
        ax.set_title(f"{'★ ' if row['Significant'] else ''}{lbl}",
                     fontsize=9, fontweight='bold',
                     color='firebrick' if row['Significant'] else 'black')
        ax.set_xticks([1, 2])
        ax.set_xticklabels([healthy_label, "Psychiatric"], fontsize=8.5)
        ax.grid(True, axis='y', alpha=0.3)
        ax.text(0.97, 0.97,
                f"r = {row['r_effect']:.3f}\np_adj = {row['p_adj']:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color='firebrick' if row['Significant'] else 'grey')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"MFDFA: {healthy_label} vs Psychiatric\n"
                 "(Multifractal Detrended Fluctuation Analysis — Kantelhardt et al. 2002)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(output_dir, "MFDFA_Group_Comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  MFDFA figure     → {fig_path}")
    return res


# ═══════════════════════════════════════════════════════════════
# RNA GROUP COMPARISON  [new analysis — Step 17]
# ═══════════════════════════════════════════════════════════════

_RNA_COLS = ["RNA_MeanDegree", "RNA_ClusteringCoeff", "RNA_Transitivity"]

def rna_group_comparison(df_h, df_p, output_dir, healthy_label="Healthy"):
    """
    Mann-Whitney U comparison of Recurrence Network Analysis metrics.

    Scientific expectations
    -----------------------
    RNA_ClusteringCoeff — expected HIGHER in patients (more constrained
                          phase-space attractor; complements elevated RQA DET)
    RNA_Transitivity    — expected HIGHER in patients (same reasoning)
    RNA_MeanDegree      — fixed at rec_rate=10% by construction → should not
                          differ; any significant difference flags a violation
                          of the fixed-rate constraint

    Saves:
        RNA_Group_Comparison.csv
        RNA_Group_Comparison.png
    """
    present = [m for m in _RNA_COLS
               if m in df_h.columns and m in df_p.columns]
    if not present:
        print("  [skip] RNA columns absent — re-run healthy_hrvacc.py and "
              "psych_hrvacc.py with the updated pipeline.py first.")
        return None

    rows = []
    for m in present:
        h_v = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p_v = pd.to_numeric(df_p[m], errors='coerce').dropna().values
        if len(h_v) < 3 or len(p_v) < 3:
            continue
        u, p_raw = mannwhitneyu(h_v, p_v, alternative='two-sided')
        r        = rank_biserial_r(u, len(h_v), len(p_v))
        rows.append({
            "Metric":       m,
            "Healthy_N":    len(h_v),
            "Healthy_Mean": np.mean(h_v),
            "Healthy_SD":   np.std(h_v, ddof=1),
            "Psych_N":      len(p_v),
            "Psych_Mean":   np.mean(p_v),
            "Psych_SD":     np.std(p_v, ddof=1),
            "Pct_Diff":     100 * (np.mean(p_v) - np.mean(h_v)) / abs(np.mean(h_v))
                            if np.mean(h_v) != 0 else np.nan,
            "r_effect":     r,
            "p_raw":        p_raw,
        })
    if not rows:
        return None

    res   = pd.DataFrame(rows).set_index("Metric")
    p_adj = benjamini_hochberg(res["p_raw"].values)
    res["p_adj"]       = p_adj
    res["Significant"] = p_adj < ALPHA

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "RNA_Group_Comparison.csv")
    res.to_csv(csv_path, float_format="%.5f")
    print(f"  RNA stats        → {csv_path}")

    # ── Console summary ───────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  RNA GROUP COMPARISON — {healthy_label} vs Psychiatric")
    print(f"{'='*72}")
    hdr = (f"  {'Metric':<25} {'H_Mean':>8} {'P_Mean':>8} "
           f"{'%Diff':>7} {'r':>6} {'p_raw':>8} {'p_adj':>8} {'Sig':>4}")
    print(hdr); print("  " + "-"*73)
    for m, row in res.iterrows():
        lbl = METRIC_LABELS.get(m, m)
        sig = "★" if row["Significant"] else ""
        print(f"  {lbl:<25} {row['Healthy_Mean']:>8.4f} {row['Psych_Mean']:>8.4f} "
              f"{row['Pct_Diff']:>7.1f} {row['r_effect']:>6.3f} "
              f"{row['p_raw']:>8.4f} {row['p_adj']:>8.4f} {sig:>4}")

    # ── Figure ────────────────────────────────────────────────
    n   = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 4.8), squeeze=False)
    rng = np.random.default_rng(seed=1)

    for i, m in enumerate([r["Metric"] for r in rows]):
        ax    = axes[0, i]
        row   = res.loc[m]
        h_v   = pd.to_numeric(df_h[m], errors='coerce').dropna().values
        p_v   = pd.to_numeric(df_p[m], errors='coerce').dropna().values

        bp = ax.boxplot([h_v, p_v], positions=[1, 2], widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color='black', lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
        bp['boxes'][0].set(facecolor=C_HEALTHY, alpha=0.55)
        bp['boxes'][1].set(facecolor=C_PSYCH,   alpha=0.55)

        for vals, pos, col in [(h_v, 1, C_HEALTHY), (p_v, 2, C_PSYCH)]:
            jit = rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(pos + jit, vals, color=col, s=25, alpha=0.75, zorder=3)

        lbl = METRIC_LABELS.get(m, m)
        ax.set_title(f"{'★ ' if row['Significant'] else ''}{lbl}",
                     fontsize=9, fontweight='bold',
                     color='firebrick' if row['Significant'] else 'black')
        ax.set_xticks([1, 2])
        ax.set_xticklabels([healthy_label, "Psychiatric"], fontsize=8.5)
        ax.grid(True, axis='y', alpha=0.3)
        ax.text(0.97, 0.97,
                f"r = {row['r_effect']:.3f}\np_adj = {row['p_adj']:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color='firebrick' if row['Significant'] else 'grey')

    fig.suptitle(f"Recurrence Network Analysis: {healthy_label} vs Psychiatric\n"
                 "(Global attractor topology — Marwan et al. 2009; "
                 "10% fixed recurrence rate)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path = os.path.join(output_dir, "RNA_Group_Comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  RNA figure       → {fig_path}")
    return res


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR (per healthy-vs-psychiatric comparison)
# ═══════════════════════════════════════════════════════════════

def run_comparison(healthy_csv=HEALTHY_CSV,
                   psych_csv=PSYCH_CSV,
                   output_dir=OUTPUT_DIR,
                   healthy_label="Healthy",
                   demographics=None,
                   cohort_m=None,
                   cohort_s=None):
    """
    Full single-comparison battery: one healthy cohort vs the psychiatric
    cohort. Steps 1–12 are the original analyses (unchanged in substance);
    steps 13–15 are the added per-comparison statistics.

    Parameters
    ----------
    healthy_label : str   used in figure titles (e.g. "HRV-ACC Control").
    demographics  : dict  per-subject {id: {"Age":, "Sex":}} for the within-
                          healthy covariate analysis (None → MIT-BIH default).
    cohort_m / cohort_s : optional radar normalisation reference. If None,
                          computed from THIS healthy cohort (each comparison
                          is z-scored against its own controls — correct, and
                          consistent with the z-score heatmap).

    Returns (df_h, df_p, stats, par_stats).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  COMPARISON ANALYSIS — {healthy_label} vs Psychiatric")
    print(f"{'='*65}")

    # ── Load data ─────────────────────────────────────────────
    df_h, df_p = load_cohorts(healthy_csv, psych_csv)

    available = [m for m in METRIC_BATTERY
                 if m in df_h.columns and m in df_p.columns]

    # Radar reference: self-compute from this healthy cohort unless supplied.
    if cohort_m is None or cohort_s is None:
        rmetrics = [m for m in RADAR_METRICS if m in df_h.columns]
        rsub = df_h[rmetrics].apply(pd.to_numeric, errors='coerce')
        cohort_m = rsub.mean()
        cohort_s = rsub.std(ddof=1).replace(0, 1)

    # ── 1. Group statistics (Mann-Whitney U) ──────────────────
    print("\n[1] Group comparison statistics (Mann-Whitney U)...")
    stats = run_group_comparison(df_h, df_p, available)
    save_summary_table(stats, output_dir)

    # ── 2. Comparison radar chart ─────────────────────────────
    print("\n[2] Comparison radar chart...")
    plot_radar_comparison(df_h, df_p, output_dir=output_dir,
                          cohort_m=cohort_m, cohort_s=cohort_s)

    # ── 3. Box plots — all metrics ────────────────────────────
    print("\n[3] Box plots (all metrics)...")
    plot_all_boxplots(df_h, df_p, stats, output_dir, healthy_label=healthy_label)

    # ── 4. Box plots — significant only ───────────────────────
    print("\n[4] Box plots (significant metrics)...")
    plot_significant_boxplots(df_h, df_p, stats, output_dir,
                              healthy_label=healthy_label)

    # ── 5. Effect size bar chart ──────────────────────────────
    print("\n[5] Effect size chart...")
    plot_effect_sizes(stats, output_dir, healthy_label=healthy_label)

    # ── 6. Violin plots (radar metrics) ───────────────────────
    print("\n[6] Violin plots...")
    plot_violin_comparison(df_h, df_p, stats, output_dir,
                           healthy_label=healthy_label)

    # ── 7. Z-score heatmap ────────────────────────────────────
    print("\n[7] Z-score heatmap...")
    plot_zscore_heatmap(df_h, df_p, output_dir, healthy_label=healthy_label)

    # ── 8. Surrogate nonlinearity summary ─────────────────────
    print("\n[8] Surrogate summary...")
    surrogate_summary(df_h, df_p, output_dir, healthy_label=healthy_label)

    # ── 9a. PSD β covariate analysis ──────────────────────────
    print("\n[9a] PSD β covariate analysis (age, sex, MeanHR, RMSSD)...")
    psd_beta_covariate_analysis(df_h, df_p, output_dir,
                                demographics=demographics,
                                healthy_label=healthy_label)

    # ── 9b. PSD β range & preprocessing sensitivity ───────────
    print("\n[9b] PSD β frequency-range & interpolation sensitivity...")
    psd_beta_range_sensitivity(df_h, df_p, stats, output_dir)

    # ── 10. Within-group correlation structure ────────────────
    print("\n[10] Within-group correlations (all significant metrics)...")
    plot_within_group_correlations(df_h, df_p, output_dir, stats=stats,
                                   healthy_label=healthy_label)

    # ── 10b. Full metric intercorrelation table ───────────────
    print("\n[10b] Metric intercorrelation table (significant metrics)...")
    metric_intercorrelation_table(df_h, df_p, stats, output_dir,
                                  healthy_label=healthy_label)

    # ── 11. Exploratory classification ────────────────────────
    print("\n[11] Exploratory classification (LOO logistic regression)...")
    classification_analysis(df_h, df_p, stats, output_dir)

    # ── 12. Power analysis ────────────────────────────────────
    print("\n[12] Power analysis...")
    power_analysis(stats, output_dir, healthy_label=healthy_label,
                   healthy_n=len(df_h), psych_n=len(df_p))

    # ── 13. Parametric comparison (Welch t + Cohen's d) [ADDED] ─
    print("\n[13] Parametric comparison (Welch t-test + Cohen's d)...")
    par_stats = run_parametric_comparison(df_h, df_p, available)
    save_parametric_table(par_stats, output_dir, healthy_label=healthy_label)
    plot_cohens_d_forest(par_stats, output_dir, healthy_label=healthy_label)

    # ── 14. Distribution diagnostics (Shapiro + Levene) [ADDED] ─
    print("\n[14] Distribution diagnostics (normality + variance)...")
    distribution_diagnostics(df_h, df_p, available, output_dir,
                             healthy_label=healthy_label)

    # ── 15. Bootstrap effect-size CIs [ADDED] ─────────────────
    print("\n[15] Bootstrap effect-size confidence intervals...")
    bootstrap_effect_sizes(df_h, df_p, available, output_dir,
                           healthy_label=healthy_label)

    # ── 16. MFDFA group comparison [NEW] ──────────────────────
    print("\n[16] MFDFA group comparison (multifractal singularity spectrum)...")
    mfdfa_group_comparison(df_h, df_p, output_dir, healthy_label=healthy_label)

    # ── 17. RNA group comparison [NEW] ────────────────────────
    print("\n[17] Recurrence network group comparison (attractor topology)...")
    rna_group_comparison(df_h, df_p, output_dir, healthy_label=healthy_label)

    print(f"\n{'='*65}")
    print(f"  Outputs saved to: {output_dir}/")
    print(f"{'='*65}")
    return df_h, df_p, stats, par_stats


# ═══════════════════════════════════════════════════════════════
# CROSS-CONTROL ORCHESTRATOR (needs all three cohorts)
# ═══════════════════════════════════════════════════════════════

def run_cross_control_analysis(df_mitbih, df_hrvacc, df_psych,
                               stats_hrvacc, stats_mitbih, output_dir):
    """
    Analyses that require BOTH control cohorts plus the patients:
      • control-vs-control confound quantification
      • TOST equivalence of the two control cohorts
      • three-group Kruskal-Wallis + Dunn omnibus
      • cross-control convergence of effect sizes
      • illness-effect-vs-confound scatter
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  CROSS-CONTROL & THREE-GROUP ANALYSES")
    print(f"{'='*65}")

    metrics = [m for m in METRIC_BATTERY
               if m in df_mitbih.columns
               and m in df_hrvacc.columns
               and m in df_psych.columns]

    print("\n[X1] Control vs control (confound magnitude)...")
    cvc = control_vs_control_analysis(df_mitbih, df_hrvacc, metrics, output_dir)

    print("\n[X2] Control equivalence (TOST, radar metrics)...")
    radar_present = [m for m in RADAR_METRICS if m in metrics]
    tost_equivalence(df_mitbih, df_hrvacc, radar_present, output_dir)

    print("\n[X3] Three-group omnibus (Kruskal-Wallis + Dunn)...")
    three_group_analysis(df_mitbih, df_hrvacc, df_psych, metrics, output_dir)

    print("\n[X4] Cross-control convergence...")
    cross_control_convergence(stats_hrvacc, stats_mitbih, output_dir)

    print("\n[X5] Illness effect vs recording confound...")
    effect_vs_confound(stats_hrvacc, cvc, output_dir)

    print(f"\n{'='*65}")
    print(f"  Cross-control outputs saved to: {output_dir}/")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── PRIMARY (matched) comparison: HRV-ACC healthy vs psychiatric ──
    # This is the featured comparison — same dataset/device/protocol, so the
    # cross-recording, environmental, and activity confounds are eliminated.
    res_hrvacc = None
    if os.path.exists(HRVACC_HEALTHY_CSV):
        print("\n### PRIMARY (matched): HRV-ACC healthy vs psychiatric ###")
        res_hrvacc = run_comparison(
            healthy_csv=HRVACC_HEALTHY_CSV,
            psych_csv=PSYCH_CSV,
            output_dir=os.path.join(OUTPUT_DIR, "healthy_hrvacc_vs_psych"),
            healthy_label="HRV-ACC Control",
            demographics=(HRVACC_HEALTHY_DEMOGRAPHICS
                          if HRVACC_HEALTHY_DEMOGRAPHICS else None),
        )
    else:
        print(f"\n[skip] HRV-ACC healthy CSV not found:\n        {HRVACC_HEALTHY_CSV}")
        print("       Run healthy_hrvacc.py first to enable the matched comparison.")

    # ── SUPPLEMENTAL comparison: MIT-BIH NSR vs psychiatric (preserved) ──
    res_mitbih = None
    if os.path.exists(MITBIH_HEALTHY_CSV):
        print("\n### SUPPLEMENTAL: MIT-BIH NSR healthy vs psychiatric ###")
        res_mitbih = run_comparison(
            healthy_csv=MITBIH_HEALTHY_CSV,
            psych_csv=PSYCH_CSV,
            output_dir=os.path.join(OUTPUT_DIR, "healthy_mitbih_vs_psych"),
            healthy_label="MIT-BIH NSR",
            demographics=HEALTHY_DEMOGRAPHICS,
        )
    else:
        print(f"\n[skip] MIT-BIH healthy CSV not found:\n        {MITBIH_HEALTHY_CSV}")

    # ── CROSS-CONTROL & THREE-GROUP (needs all three cohorts) ──
    if res_hrvacc is not None and res_mitbih is not None:
        df_h_hrvacc, df_p, stats_hrvacc, _ = res_hrvacc
        df_h_mitbih, _,    stats_mitbih, _ = res_mitbih
        run_cross_control_analysis(
            df_mitbih=df_h_mitbih,
            df_hrvacc=df_h_hrvacc,
            df_psych=df_p,
            stats_hrvacc=stats_hrvacc,
            stats_mitbih=stats_mitbih,
            output_dir=os.path.join(OUTPUT_DIR, "cross_control"),
        )
    else:
        print("\n[skip] Cross-control / three-group analyses require BOTH "
              "control CSVs.\n       Run healthy.py and healthy_hrvacc.py, "
              "then re-run comparison.py.")