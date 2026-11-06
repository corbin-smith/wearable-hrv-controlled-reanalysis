"""comparison.py: group statistics, sensitivity analyses and summary figures.

Runs two comparisons: HRV-ACC controls against patients (primary) and MIT-BIH controls against
patients, plus the comparison of the two control cohorts. Mann-Whitney U tests with rank-biserial r
and Benjamini-Hochberg correction within named families; heart-rate-adjusted partial correlations;
resolution, estimator, surrogate, covariate, subgroup and medication analyses; bootstrap intervals;
summary figures. Writes outputs/comparison/<comparison>/.

Usage: python comparison.py   (after the cohort scripts, rna_controls.py and merge_mobility.py)
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
    # Imported from pipeline.py so the table labels match the constants the pipeline applied.
    QUANT_TARGET_RATIO, QUANT_TARGET_SWEEP, QUANT_STEP_S,
    # The amplitude and spectral measures re-quantized to the 7.8 ms grid.
    REQUANT_AMPLITUDE_SPECTRAL,
)

warnings.filterwarnings("ignore")


# ---- Configuration ----

# -- Cohort results files --
# MIT-BIH NSR controls (external cohort).
MITBIH_HEALTHY_CSV = os.path.join("outputs", "healthy",
                                  "ecg_results_Healthy.csv")
# HRV-ACC controls, the primary control group (written by healthy_hrvacc.py).
HRVACC_HEALTHY_CSV = os.path.join("outputs", "healthy_hrvacc",
                                  "ecg_results_HealthyACC.csv")
# Default control path.
HEALTHY_CSV    = MITBIH_HEALTHY_CSV
PSYCH_CSV      = os.path.join("outputs", "psychiatric",
                               "ecg_results_Psychiatric.csv")
OUTPUT_DIR     = os.path.join("outputs", "comparison")


# Radar-chart reference frame: run_comparison() computes it from the control cohort it is given
# (see "Radar reference" in run_comparison); there are no hard-coded reference values.
HEALTHY_MEAN = None
HEALTHY_SD   = None

# Full measure battery for group comparison (excludes QC columns and surrogate internals).
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

# -- Correction families --
# Three families are corrected separately: the primary measures, MFDFA and recurrence networks. The
# deterministic quantities (RQA_REC, RNA_MeanDegree) and MFDFA_Asymmetry (undefined under positive q)
# are reported but belong to no family. The sizes asserted in _family_size_selftest() are authoritative.
MFDFA_FAMILY = ["MFDFA_Width", "MFDFA_H_Mean", "MFDFA_H2"]
MFDFA_UNDEFINED = ["MFDFA_Asymmetry"]          # reported as not computable

# -- Deterministic quantities: excluded from every family --
# RQA_REC and RNA_MeanDegree equal the target recurrence rate for every subject by construction; they
# are normalization checks, not hypothesis tests.
DETERMINISTIC_CHECKS = ["RQA_REC", "RNA_MeanDegree"]
# Measures for which a percentage difference is not reported because the reference is near the
# floor (pNN50). Absolute differences and effect sizes are still reported.
PCT_DIFF_SUPPRESSED = ["HRV_pNN50"]
# The slope's fitted band in Hz as resolved by the Welch settings (lowest bin 0.0156 Hz at
# nperseg = 512), used by the scale-band correspondence table.
BETA_BAND_FOR_REPORT = (0.0156, 0.15)
RQA_TARGET_RATE = 0.10
RNA_FAMILY   = ["RNA_ClusteringCoeff", "RNA_Transitivity"]
# Statistically redundant measures, excluded from the primary family: ApEn correlates with SampEn at
# r = 0.98 and is the length-dependent member of the pair. ApEn is still computed and reported outside
# the family.
STATISTICALLY_REDUNDANT = ["ApEn"]

PRIMARY_METRICS = [m for m in METRIC_BATTERY
                   if m not in MFDFA_FAMILY + RNA_FAMILY + MFDFA_UNDEFINED
                   + DETERMINISTIC_CHECKS + STATISTICALLY_REDUNDANT
                   + ["RNA_MeanDegree"]]

# Definitionally collinear measures (SD1 = RMSSD/sqrt(2); SD2 from SDNN and SD1; MeanHR = 60/MeanNN).
# Both the full-family and the non-redundant corrections are reported.
COLLINEAR_DUPLICATES = ["HRV_SD1", "HRV_SD2", "HRV_MeanHR"]
NONREDUNDANT_PRIMARY = [m for m in PRIMARY_METRICS
                        if m not in COLLINEAR_DUPLICATES]

# -- Heart-rate-adjusted family --
# Its own correction family. Excluded: HRV_MeanHR and HRV_MeanNN (the covariate and its reciprocal),
# HRV_SD1 (RMSSD/sqrt(2)), HRV_SD2 and HRV_SD1SD2 (functions of SDNN and RMSSD), HRV_pNN50 (floor-
# limited in patients). Added beyond the primary family: VLF_Power_ms2, VLF_LF_Ratio, PSD_Beta_NoMayer,
# ApEn, RNA_ClusteringCoeff and RNA_Transitivity.
HR_ADJUSTMENT_COVARIATE = "HRV_MeanHR"
HR_ADJUSTED_FAMILY = [
    "HRV_RMSSD", "HRV_SDNN",
    "LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2", "Total_Power_ms2",
    "VLF_LF_Ratio", "LF_HF_Ratio",
    "PSD_Beta_NoMayer", "PSD_Beta", "DFA_Alpha2", "DFA_Alpha", "DFA_Alpha1",
    "MSE_Mean", "SampEn", "ApEn", "PermEn",
    "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity",
]
# Amplitude measures normalized by mean NN (Sacha, 2014): SDs and RMSSD by mean NN, band powers by
# its square.
MEANNN_NORMALISED = [("HRV_RMSSD", 1), ("HRV_SDNN", 1),
                     ("LF_Power_ms2", 2), ("HF_Power_ms2", 2),
                     ("VLF_Power_ms2", 2), ("Total_Power_ms2", 2)]

# Schizophrenia subject list, used only by the superseded schizophrenia_subgroup_analysis(); diagnosis
# now comes from the metadata columns. Leave empty.
SCHIZOPHRENIA_RECORD_IDS = []

# Display names for figures
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

# Significance threshold after correction
ALPHA = 0.05

# Colors
C_HEALTHY  = "steelblue"
C_PSYCH    = "firebrick"

# -- MIT-BIH demographics --
# Age and sex in record order (16265 ... 19830); sex coding 0 = male, 1 = female.
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
# MIT-BIH: mean age 34.3 yr, range 20-50; 13 F, 5 M


def _add_healthy_demographics(df_h, demographics=None):
    """Attach Age and Sex to the control DataFrame from HEALTHY_DEMOGRAPHICS (MIT-BIH IDs).
    Rows not in the dict receive NaN.
    """
    df = df_h.copy()

    # Age and Sex already present come from the released metadata and are kept.
    if ("Age" in df.columns and "Sex" in df.columns
            and pd.to_numeric(df["Age"], errors="coerce").notna().any()):
        if demographics is None:
            n = int(pd.to_numeric(df["Age"], errors="coerce").notna().sum())
            print(f"  Using released age/sex metadata for {n}/{len(df)} "
                  f"control subjects.")
            return df

    # HEALTHY_DEMOGRAPHICS holds MIT-BIH record IDs; HRV-ACC IDs do not match, which is announced.
    if demographics is None:
        demographics = HEALTHY_DEMOGRAPHICS
        _keys = {str(i).split("/")[-1].split("\\")[-1] for i in df_h.index}
        if not (_keys & set(HEALTHY_DEMOGRAPHICS)):
            print("  NOTE: no per-subject age/sex available for this cohort "
                  "(demographics=None and the MIT-BIH defaults do not match "
                  "these record IDs). The within-healthy age and sex analyses "
                  "will be skipped. State this in the manuscript rather than "
                  "omitting it.")
    ages, sexes = [], []
    for idx in df.index:
        # The index may be the bare record ID or contain it as a suffix.
        key = str(idx).split("/")[-1].split("\\")[-1]
        demo = demographics.get(key, {})
        ages.append(demo.get("Age", np.nan))
        sexes.append(demo.get("Sex", np.nan))
    df["Age"] = ages
    df["Sex"] = sexes
    return df


# ---- Step 1: data loading ----

def load_cohorts(healthy_csv, psych_csv):
    """Load both cohort CSVs, drop the appended summary rows, and return per-subject DataFrames."""
    def _load(path, label):
        df = pd.read_csv(path, index_col=0)
        # Drop summary rows appended by run_analysis().
        summary_pattern = df.index.str.contains(
            r'^(Cohort_|Healthy_|HealthyACC_|Psychiatric_)', regex=True, na=False
        )
        df = df[~summary_pattern].copy()
        # Convert a column to numeric only if every non-null value converts (pandas 3 removed
        # errors='ignore').
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            # Text columns such as Record_Start_Clock stay as strings.
            if converted.notna().sum() == df[col].notna().sum():
                df[col] = converted
        print(f"  Loaded {label}: {len(df)} subjects, {len(df.columns)} columns")
        return df

    df_h = _load(healthy_csv,  "Healthy")
    df_p = _load(psych_csv,    "Psychiatric")
    return df_h, df_p


# ---- Step 2: statistical comparison ----

def rank_biserial_r(u_stat, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u_stat) / (n1 * n2)


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction. Returns adjusted p values in the input order."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order  = np.argsort(p)                 # positions, ascending by p
    ranked = p[order]
    adj    = ranked * n / np.arange(1, n + 1)
    adj    = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out    = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out


def _bh_selftest():
    """Self-check at import: correct ranks and order independence."""
    p = np.array([0.18577, 0.00023, 0.34029, 0.00081])
    expected = np.array([0.24769, 0.00092, 0.34029, 0.00162])
    got = benjamini_hochberg(p)
    assert np.allclose(got, expected, atol=1e-4), \
        f"BH self-test failed: {got} != {expected}"
    # Shuffling the input must permute the output identically.
    idx = np.array([2, 0, 3, 1])
    assert np.allclose(benjamini_hochberg(p[idx]), expected[idx], atol=1e-4), \
        "BH self-test failed: result is order-dependent"
    # Adjusted p can never fall below raw p.
    assert np.all(got >= p - 1e-12), "BH self-test failed: adjusted p < raw p"


_bh_selftest()


def _family_size_selftest():
    """Assert the correction-family sizes; figures.py asserts the same values, so the two files cannot drift."""
    # Expected sizes are derived from the battery and the exclusion lists, so only a divergence between
    # this file and figures.py fails.
    _expected_primary = len([m for m in METRIC_BATTERY
                             if m not in MFDFA_FAMILY + RNA_FAMILY
                             + MFDFA_UNDEFINED + DETERMINISTIC_CHECKS
                             + STATISTICALLY_REDUNDANT + ["RNA_MeanDegree"]])
    assert len(PRIMARY_METRICS) == _expected_primary, (
        f"primary FDR family is {len(PRIMARY_METRICS)}, derivation says "
        f"{_expected_primary}; an exclusion list and PRIMARY_METRICS disagree")
    assert len(MFDFA_FAMILY) == 3, (
        f"MFDFA family is {len(MFDFA_FAMILY)}, expected 3")
    assert len(RNA_FAMILY) == 2, (
        f"RNA family is {len(RNA_FAMILY)}, expected 2")
    _expected_nonredundant = len([m for m in PRIMARY_METRICS
                                  if m not in COLLINEAR_DUPLICATES])
    assert len(NONREDUNDANT_PRIMARY) == _expected_nonredundant, (
        f"non-redundant family is {len(NONREDUNDANT_PRIMARY)}, derivation "
        f"says {_expected_nonredundant}")
    # Report the family sizes used.
    print(f"  [family sizes] primary {len(PRIMARY_METRICS)}, "
          f"non-redundant {len(NONREDUNDANT_PRIMARY)}, "
          f"MFDFA {len(MFDFA_FAMILY)}, RNA {len(RNA_FAMILY)}")
    for m in DETERMINISTIC_CHECKS:
        assert m not in PRIMARY_METRICS and m not in RNA_FAMILY, (
            f"{m} is deterministic and must belong to no FDR family")
        assert m not in HR_ADJUSTED_FAMILY, (
            f"{m} is deterministic and must belong to no FDR family")
    assert len(set(HR_ADJUSTED_FAMILY)) == len(HR_ADJUSTED_FAMILY), (
        "HR_ADJUSTED_FAMILY lists a metric twice")
    assert HR_ADJUSTMENT_COVARIATE not in HR_ADJUSTED_FAMILY, (
        "the heart-rate covariate cannot be a member of its own family")
    print(f"  [family sizes] heart-rate-adjusted {len(HR_ADJUSTED_FAMILY)}")


_family_size_selftest()


def minimum_detectable_effect(n1, n2, alpha=0.05, power=0.80):
    """The smallest effect this sample could detect with 80% power at alpha = 0.05 (depends only on N)."""
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    d_min = (z_a + z_b) * np.sqrt(1.0 / n1 + 1.0 / n2)

    # Converted to the rank-biserial scale used in the tables: for two normals d SDs apart,
    # P(X > Y) = Phi(d / sqrt(2)) and r_rank = 2 * P(X > Y) - 1 (0.391 at 30 per group). The point-biserial
    # value d / sqrt(d^2 + 4) is also returned.
    r_min_point = d_min / np.sqrt(d_min ** 2 + 4.0)
    r_min_rank = 2.0 * norm.cdf(d_min / np.sqrt(2.0)) - 1.0
    return float(d_min), float(r_min_rank), float(r_min_point)


def run_group_comparison(df_h, df_p, metrics, family_label="primary"):
    """Mann-Whitney U test for each measure.

    Returns a DataFrame with group means, SDs, medians and IQRs, Pct_Diff, U_stat, p_raw, r_effect,
    p_adj and Significant.
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

        # Medians and IQRs: the location estimate consistent with the rank-based tests.
        h_med, p_med = float(np.median(h_vals)), float(np.median(p_vals))
        h_q1, h_q3 = np.percentile(h_vals, [25, 75])
        p_q1, p_q3 = np.percentile(p_vals, [25, 75])

        # Percentage difference from medians; withheld for floor-limited measures (pNN50).
        _floor_limited = m in PCT_DIFF_SUPPRESSED
        if _floor_limited or h_med == 0:
            pct_diff = np.nan
        else:
            pct_diff = 100 * (p_med - h_med) / abs(h_med)

        rows.append({
            "Metric":       m,
            "Healthy_N":    len(h_vals),
            "Healthy_Mean": h_mean,
            "Healthy_SD":   np.std(h_vals, ddof=1),
            "Healthy_Median": h_med,
            "Healthy_IQR":  float(h_q3 - h_q1),
            "Healthy_Q1":   float(h_q1),
            "Healthy_Q3":   float(h_q3),
            "Psych_N":      len(p_vals),
            "Psych_Mean":   p_mean,
            "Psych_SD":     np.std(p_vals, ddof=1),
            "Psych_Median": p_med,
            "Psych_IQR":    float(p_q3 - p_q1),
            "Psych_Q1":     float(p_q1),
            "Psych_Q3":     float(p_q3),
            "Pct_Diff":     pct_diff,
            "Pct_Diff_Basis": ("suppressed_floor_limited" if _floor_limited
                               else "median"),
            "U_stat":       u,
            "p_raw":        p_raw,
            "r_effect":     r,
        })

    results = pd.DataFrame(rows).set_index("Metric")

    # BH within this family only; the family is named in the output and raw p is kept beside adjusted p.
    p_adj = benjamini_hochberg(results["p_raw"].values)
    results["p_adj"]        = p_adj
    results["Significant"]  = p_adj < ALPHA
    results["FDR_Family"]   = family_label
    results["FDR_Family_N"] = len(results)

    results = results.sort_values("p_adj")
    return results


# ---- Step 3: group summary table ----

def save_summary_table(stats, output_dir):
    """Save the full statistical results as a CSV and print a console summary.

    Bootstrap intervals are attached later by attach_bootstrap_intervals(), once the bootstrap exists.
    """
    out = stats.copy()
    out.index = [METRIC_LABELS.get(m, m) for m in out.index]
    path = os.path.join(output_dir, "group_comparison_stats.csv")
    out.to_csv(path, float_format="%.6f")
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


# ---- Step 4: box plots (all measures) ----

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
    """One box plot per measure, significant measures with red titles."""
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
    """Box plots for the significant measures only."""
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


# ---- Step 5: effect-size bar chart ----

def plot_effect_sizes(stats, output_dir, healthy_label="Healthy"):
    """Rank-biserial r for all measures, sorted by magnitude, significant ones colored, with dashed lines
    at |r| = 0.3 and 0.5.
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


# ---- Step 6: violin plots ----

def plot_violin_comparison(df_h, df_p, stats, output_dir, healthy_label="Healthy"):
    """Violin plots for the five radar measures, significant measures annotated."""
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


# ---- Step 7: z-score heatmap (per subject, all measures) ----

def plot_zscore_heatmap(df_h, df_p, output_dir, healthy_label="Healthy"):
    """Heatmap with one row per subject and one column per measure, z-scored on the control cohort;
    controls above, patients below.
    """
    metrics = [m for m in METRIC_BATTERY
               if m in df_h.columns and m in df_p.columns]

    h_sub = df_h[metrics].apply(pd.to_numeric, errors='coerce')
    p_sub = df_p[metrics].apply(pd.to_numeric, errors='coerce')

    # Z-score on control cohort parameters.
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


# ---- Step 8: surrogate nonlinearity summary ----

def surrogate_summary(df_h, df_p, output_dir, healthy_label="Healthy"):
    """Fraction of subjects in each cohort with a significant IAAFT surrogate test for SampEn, DFA_Alpha1
    and MSE_Mean, as a table and bar chart.
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


# ---- Step 9a: slope covariate analysis ----

def psd_beta_covariate_analysis(df_h, df_p, output_dir,
                                demographics=None, healthy_label="Healthy"):
    """Partial Spearman correlations of the uncorrected slope with group, unadjusted and controlling for
    mean heart rate, RMSSD, or both (both residualized on the covariates by OLS, then Spearman), and,
    within the control cohort, the slope's association with age and sex.

    Writes PSD_Beta_Covariate_Analysis.csv and .png. The incremental-AUC analysis is in
    nested_classifier.py.
    """
    # -- Combined dataset --
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

    # -- Partial Spearman helper --
    def partial_spearman(target_col, covar_cols, outcome_col, df):
        """Partial Spearman r between target and outcome given covariates (OLS residuals, then Spearman)."""
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

    # Controlling for MeanHR and RMSSD jointly
    r3, p3 = partial_spearman("PSD_Beta", ["HRV_MeanHR", "HRV_RMSSD"],
                               "Group", cdf)
    rows.append({"Analysis": "PSD_Beta ~ Group | MeanHR + RMSSD",
                 "r_or_AUC": r3, "p": p3})

    # The incremental-AUC analysis is only in nested_classifier.py (C = 1.0, with a bootstrap interval on
    # the increment).
    rows.append({"Analysis": "LOO-AUC incremental test (see "
                             "nested_classifier.py, Classification_Summary_"
                             "NestedCV.csv)",
                 "r_or_AUC": np.nan, "p": np.nan})

    # -- Within the control cohort: age and sex --
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

    # -- Save CSV --
    covar_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "PSD_Beta_Covariate_Analysis.csv")
    covar_df.to_csv(csv_path, index=False, float_format="%.6f")

    print(f"\n  PSD BETA COVARIATE ANALYSIS")
    print(f"  {'Analysis':<55} {'r / AUC':>8} {'p':>8}")
    print("  " + "-" * 75)
    for _, row in covar_df.iterrows():
        p_str = f"{row['p']:.4f}" if np.isfinite(row["p"]) else "  —"
        print(f"  {row['Analysis']:<55} {row['r_or_AUC']:>8.4f} {p_str:>8}")
    print(f"  Saved → {csv_path}")

    # -- Figure --
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
    """Slope group difference across fitting ranges and interpolation methods.

    Writes PSD_Beta_Range_Sensitivity.csv (r, p, significance per variant) and .png; also runs the
    R2-filter sensitivity. Prints a note if the pipeline columns are missing.
    """
    # Display label -> column name
    variants = {
        "Primary\n(0.01–0.15 Hz,\ncubic)":    "PSD_Beta",
        "Wide VLF–LF\n(0.003–0.15 Hz,\ncubic)": "PSD_Beta_Wide",
        "Full spectrum\n(0.003–0.40 Hz,\ncubic)": "PSD_Beta_Full",
        "Linear interp\n(0.01–0.15 Hz,\nlinear)": "PSD_Beta_Linear",
    }

    # Columns actually present
    available = {lbl: col for lbl, col in variants.items()
                 if col in df_h.columns and col in df_p.columns}

    if len(available) <= 1:
        print("  PSD β range sensitivity: only primary β column found.\n"
              "  → Re-run healthy_hrvacc.py and psych_hrvacc.py with the updated\n"
              "    pipeline.py to populate PSD_Beta_Wide / Full / Linear\n"
              "    columns, then re-run comparison.py.")
        # Still run the R2-filter sensitivity
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
    sens_df.to_csv(csv_path, index=False, float_format="%.6f")

    print(f"\n  PSD BETA RANGE & PREPROCESSING SENSITIVITY")
    print(f"  {'Variant':<45} {'H Mean±SD':>15} {'P Mean±SD':>15}"
          f" {'r':>6} {'p_raw':>8}")
    print("  " + "-" * 95)
    for _, row in sens_df.iterrows():
        print(f"  {row['Variant']:<45} {row['H_Mean_SD']:>15} {row['P_Mean_SD']:>15}"
              f" {row['r_effect']:>6.3f} {row['p_raw']:>8.4f}")
    print(f"  Saved → {csv_path}")

    # Figure: r across variants
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

    # R2-filter sensitivity
    psd_beta_sensitivity(df_h, df_p, stats, output_dir)


# ---- Step 9: slope R2-filter sensitivity ----

def psd_beta_sensitivity(df_h, df_p, stats, output_dir):
    """Rerun the group comparison on subjects with PSD_Beta_R2 >= PSD_BETA_R2_THRESHOLD in both groups."""
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

    # Side-by-side comparison for the slope and radar measures
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
    sens_df.to_csv(path, float_format="%.6f")
    print(f"  Sensitivity CSV → {path}")
    print("\n  PSD BETA SENSITIVITY (full vs R²-filtered)")
    print(sens_df.to_string())


# ---- Step 10: classification (superseded) ----

def classification_analysis(df_h, df_p, stats, output_dir):
    """Superseded; not called by run_comparison() and not reported.

    Features are selected on the full sample before leave-one-out scoring, so the AUC is optimistic.
    nested_classifier.py selects inside each fold. Output goes to
    Classification_Summary_LEAKY_DO_NOT_REPORT.csv.
    """
    warnings.warn(
        "classification_analysis() is leaky (feature selection sees all "
        "subjects) and must not be reported. Use nested_classifier.py.",
        DeprecationWarning, stacklevel=2)
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

    # Combined dataset
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

    # -- Multi-measure leave-one-out classifier --
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
        os.path.join(output_dir, "Classification_Summary_LEAKY_DO_NOT_REPORT.csv"), index=False
    )

    # -- ROC figure --
    ncols = min(len(sig_metrics) + 1, 4)
    nrows = int(np.ceil((len(sig_metrics) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Panel 0: multi-measure
    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    axes_flat[0].plot(fpr, tpr, color='purple', lw=2,
                      label=f"Multi-metric AUC={auc:.3f}")
    axes_flat[0].plot([0,1],[0,1],'k--', lw=0.8)
    axes_flat[0].set_title("Multi-metric (LOO LR)", fontsize=9)
    axes_flat[0].set_xlabel("FPR"); axes_flat[0].set_ylabel("TPR")
    axes_flat[0].legend(fontsize=7); axes_flat[0].grid(True, alpha=0.3)

    # Individual measure ROC curves
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


# ---- Step 11: power analysis (superseded) ----

def power_analysis(stats, output_dir, healthy_label="Healthy",
                   healthy_n=None, psych_n=None):
    """Superseded; not called by run_comparison() and not reported.

    Computes power at the observed effect size, which restates the p value. minimum_detectable_effect()
    is used instead.
    """
    warnings.warn(
        "power_analysis() is deprecated and must not be used for the "
        "manuscript: post-hoc power at the observed effect size is circular. "
        "Use report_minimum_detectable_effect() instead.",
        DeprecationWarning, stacklevel=2)

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

    # Bar chart: N per group for each measure
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


# ---- Step 10b: measure intercorrelation table ----

def metric_intercorrelation_table(df_h, df_p, stats, output_dir, healthy_label="Healthy"):
    """Pairwise Spearman correlations among the significant measures within each cohort, with pairs
    |r| > 0.7 flagged, showing which measures are near-redundant.

    Writes Metric_Intercorrelation_Healthy.csv, Metric_Intercorrelation_Psychiatric.csv,
    Metric_Intercorrelation_HighPairs.csv and Metric_Intercorrelation.png.
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
    # Four decimals, so a near-1 correlation (SDNN and SD2) is not printed as 1.000. SD2 is not
    # proportional to SDNN by definition (SD2 = sqrt(2*SDNN^2 - SD1^2)); their near-perfect correlation
    # here is empirical.
    h_corr.to_csv(os.path.join(output_dir,
                                "Metric_Intercorrelation_Healthy.csv"),
                  float_format="%.4f")
    p_corr.to_csv(os.path.join(output_dir,
                                "Metric_Intercorrelation_Psychiatric.csv"),
                  float_format="%.4f")

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

    # -- Side-by-side heatmap --
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


# ---- Step 12: within-group correlation heatmaps ----

def plot_within_group_correlations(df_h, df_p, output_dir, stats=None, healthy_label="Healthy"):
    """Side-by-side Pearson correlation heatmaps: all significant measures (sorted by |r|) if stats is
    given, otherwise the five radar measures.
    """
    if stats is not None:
        sig = [m for m in stats.index[stats["Significant"]]
               if m in df_h.columns and m in df_p.columns]
        # Sort by |r|, descending
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


# ---- Additional analyses: parametric tests, distribution diagnostics, bootstrap intervals,
# control-against-control comparison, TOST equivalence, three-group tests, cross-control convergence ----

# Colors for the two-control and three-group figures
C_MITBIH = "teal"        # MIT-BIH NSR healthy (supplemental control)
C_HRVACC = "steelblue"   # HRV-ACC healthy (primary, recording-matched control)


# -- HRV-ACC control demographics (optional mapping; per-subject values come from the metadata file) --
# Format: {"control_2": {"Age": int, "Sex": 0/1}, ...} (Sex: 0 = male, 1 = female).
HRVACC_HEALTHY_DEMOGRAPHICS = {
}


def _cohens_d_ci(x_ref, x_other):
    """Cohen's d, Hedges' g (small-sample corrected) and an approximate 95% CI for d.

    d = (mean_other - mean_ref) / pooled SD, so positive d means x_other is higher (patients, when called
    with ref = controls). Returns (d, g, ci_low, ci_high); NaNs if the pooled SD is zero.
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


# ---- Step 13: parametric comparison (Welch t and Cohen's d) ----

def run_parametric_comparison(df_h, df_p, metrics):
    """Welch's t test per measure, with Cohen's d, Hedges' g and an approximate 95% CI for d, BH-corrected,
    alongside the Mann-Whitney table. Positive t or d = higher in patients.
    """
    # pNN50 is zero-inflated with a floor at zero, so it is excluded here and reported with rank
    # statistics only (see pnn50_floor_report).
    PARAMETRIC_EXCLUDE = {"HRV_pNN50"}

    rows = []
    for m in metrics:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        if m in PARAMETRIC_EXCLUDE:
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
    out.to_csv(path, float_format="%.6f")

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
    """Forest plot of Cohen's d with analytic 95% CIs, sorted by |d|, significant measures in red, guides
    at |d| = 0.2, 0.5 and 0.8.
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


# ---- Step 14: distribution diagnostics (normality and variance) ----

def distribution_diagnostics(df_h, df_p, metrics, output_dir,
                             healthy_label="Healthy"):
    """Shapiro-Wilk normality within each group and Levene's test for equal variances, per measure.

    Writes Distribution_Diagnostics.csv and Distribution_Diagnostics_Levene.png.
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
    diag.to_csv(path, float_format="%.6f")
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

    # Levene p figure
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


# ---- Step 15: bootstrap effect-size intervals ----

def bootstrap_effect_sizes(df_h, df_p, metrics, output_dir,
                           healthy_label="Healthy",
                           n_boot=2000, seed=0):
    """Percentile bootstrap 95% CIs for rank-biserial r and Cohen's d per measure (subjects resampled
    within group).

    Writes Bootstrap_Effect_Sizes.csv and Bootstrap_r_Forest.png.
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
    out.to_csv(path, float_format="%.6f")
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


# ---- Step 16: control against control ----

def control_vs_control_analysis(df_a, df_b, metrics, output_dir,
                                label_a="MIT-BIH NSR",
                                label_b="HRV-ACC Control"):
    """Compare the two control cohorts directly (Mann-Whitney with BH, Welch t, Cohen's d with CI,
    Levene), measuring how much each measure moves between cohorts that differ in device, resolution,
    setting, era and age.

    Writes Control_vs_Control_Stats.csv and Control_vs_Control_AbsD.png; returns the per-measure DataFrame.
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
    out.to_csv(path, float_format="%.6f")

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

    # |Cohen's d| bar chart
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


# ---- Step 17: TOST equivalence between control cohorts ----

def tost_equivalence(df_a, df_b, metrics, output_dir,
                     bound_d=0.5, alpha=0.05,
                     label_a="MIT-BIH NSR", label_b="HRV-ACC Control"):
    """Two one-sided tests for equivalence of the two control cohorts on each measure, with margin bound_d
    (default 0.5 SD). A significant TOST means equivalence within +/-bound_d; a non-significant
    difference test is not evidence of equivalence.

    Writes Control_Equivalence_TOST.csv.
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
    tost.to_csv(path, float_format="%.6f")
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


# ---- Step 18: three-group tests (Kruskal-Wallis and Dunn post hoc) ----

def _dunn_posthoc(groups):
    """Dunn's post-hoc test (tie-corrected) for k independent groups. Returns (i, j, z, p_raw) for each pair
    i < j (two-sided, normal approximation); BH correction is done by the caller.
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
    """Kruskal-Wallis across the three groups per measure (BH-corrected), with Dunn's post-hoc pairwise
    comparisons (BH across all post-hoc tests) for significant measures.

    Writes ThreeGroup_KruskalWallis.csv, ThreeGroup_Dunn_Posthoc.csv and ThreeGroup_Boxplots.png.
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
    kw_out.to_csv(kw_path, float_format="%.6f")

    dunn = pd.DataFrame(dunn_rows)
    dunn_path = os.path.join(output_dir, "ThreeGroup_Dunn_Posthoc.csv")
    if not dunn.empty:
        dunn["p_adj"]       = benjamini_hochberg(dunn["p_raw"].values)
        dunn["Significant"] = dunn["p_adj"] < ALPHA
        dunn.to_csv(dunn_path, index=False, float_format="%.6f")

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

    # Three-group box plots for the radar measures
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


# ---- Step 19: cross-control convergence ----

def cross_control_convergence(stats_hrvacc, stats_mitbih, output_dir):
    """Merge the two control-against-patient tables by measure and flag effects significant after correction
    in both, in the same direction.

    Writes CrossControl_Convergence.csv and CrossControl_r_vs_r.png.
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
    conv.to_csv(path, float_format="%.6f")

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

    # r against r scatter
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
    """Scatter of each measure's control-cohort difference (|Cohen's d|, x) against its patient effect
    (|r| against HRV-ACC controls, y). Writes EffectVsConfound.png.
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


# ---- MFDFA group comparison ----

# MFDFA_Asymmetry is undefined under positive-q estimation and is not in the family.
_MFDFA_COLS = ["MFDFA_Width", "MFDFA_H_Mean", "MFDFA_H2"]

def mfdfa_group_comparison(df_h, df_p, output_dir, healthy_label="Healthy"):
    """Mann-Whitney comparison of the MFDFA scalars between cohorts.

    Writes MFDFA_Group_Comparison.csv and MFDFA_Group_Comparison.png.
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

    # -- Save CSV --
    csv_path = os.path.join(output_dir, "MFDFA_Group_Comparison.csv")
    res.to_csv(csv_path, float_format="%.6f")
    print(f"  MFDFA stats      → {csv_path}")

    # -- Console summary --
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

    # -- Figure: box and strip plots --
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


# ---- Recurrence network group comparison ----

# Recurrence-network measures in the correction family. RNA_MeanDegree equals the target recurrence
# rate by construction and is reported by deterministic_checks() outside any correction.
_RNA_COLS = ["RNA_ClusteringCoeff", "RNA_Transitivity"]

def rna_group_comparison(df_h, df_p, output_dir, healthy_label="Healthy"):
    """Mann-Whitney comparison of the recurrence-network measures.

    Writes RNA_Group_Comparison.csv and RNA_Group_Comparison.png.
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

    # -- Save CSV --
    csv_path = os.path.join(output_dir, "RNA_Group_Comparison.csv")
    res.to_csv(csv_path, float_format="%.6f")
    print(f"  RNA stats        → {csv_path}")

    # -- Console summary --
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

    # -- Figure --
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


# ---- Main orchestrator (one control-against-patient comparison) ----

def run_comparison(healthy_csv=HEALTHY_CSV,
                   psych_csv=PSYCH_CSV,
                   output_dir=OUTPUT_DIR,
                   healthy_label="Healthy",
                   demographics=None,
                   cohort_m=None,
                   cohort_s=None):
    """Full battery for one control cohort against the patients.

    Parameters
    ----------
    healthy_label        : str, used in figure titles (e.g. "HRV-ACC Control")
    demographics         : dict {id: {"Age":, "Sex":}} for the within-control covariate analysis
                           (None: MIT-BIH default)
    cohort_m / cohort_s  : optional radar reference; if None, computed from this control cohort

    Returns (df_h, df_p, stats, par_stats).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  COMPARISON ANALYSIS — {healthy_label} vs Psychiatric")
    print(f"{'='*65}")

    # -- Load data --
    df_h, df_p = load_cohorts(healthy_csv, psych_csv)

    available = [m for m in METRIC_BATTERY
                 if m in df_h.columns and m in df_p.columns]

    # Radar reference: computed from this control cohort unless supplied.
    if cohort_m is None or cohort_s is None:
        rmetrics = [m for m in RADAR_METRICS if m in df_h.columns]
        rsub = df_h[rmetrics].apply(pd.to_numeric, errors='coerce')
        cohort_m = rsub.mean()
        cohort_s = rsub.std(ddof=1).replace(0, 1)

    # -- 1. Group statistics (Mann-Whitney U) --
    print("\n[1] Group comparison statistics (Mann-Whitney U)...")
    primary = [m for m in PRIMARY_METRICS if m in available]
    stats = run_group_comparison(df_h, df_p, primary, family_label="primary")
    save_summary_table(stats, output_dir)
    print(f"  Primary FDR family: {len(primary)} metrics "
          f"(MFDFA and RNA corrected separately below).")

    # -- 2. Comparison radar chart --
    print("\n[2] Comparison radar chart...")
    plot_radar_comparison(df_h, df_p, output_dir=output_dir,
                          cohort_m=cohort_m, cohort_s=cohort_s)

    # -- 3. Box plots, all measures --
    print("\n[3] Box plots (all metrics)...")
    plot_all_boxplots(df_h, df_p, stats, output_dir, healthy_label=healthy_label)

    # -- 4. Box plots, significant only --
    print("\n[4] Box plots (significant metrics)...")
    plot_significant_boxplots(df_h, df_p, stats, output_dir,
                              healthy_label=healthy_label)

    # -- 5. Effect-size bar chart --
    print("\n[5] Effect size chart...")
    plot_effect_sizes(stats, output_dir, healthy_label=healthy_label)

    # -- 6. Violin plots (radar measures) --
    print("\n[6] Violin plots...")
    plot_violin_comparison(df_h, df_p, stats, output_dir,
                           healthy_label=healthy_label)

    # -- 7. Z-score heatmap --
    print("\n[7] Z-score heatmap...")
    plot_zscore_heatmap(df_h, df_p, output_dir, healthy_label=healthy_label)

    # -- 8. Surrogate nonlinearity summary --
    print("\n[8] Surrogate summary...")
    surrogate_summary(df_h, df_p, output_dir, healthy_label=healthy_label)

    # -- 9a. Slope covariate analysis --
    print("\n[9a] PSD β covariate analysis (age, sex, MeanHR, RMSSD)...")
    psd_beta_covariate_analysis(df_h, df_p, output_dir,
                                demographics=demographics,
                                healthy_label=healthy_label)

    # -- 9b. Slope range and preprocessing sensitivity --
    print("\n[9b] PSD β frequency-range & interpolation sensitivity...")
    psd_beta_range_sensitivity(df_h, df_p, stats, output_dir)

    # -- 10. Within-group correlation structure --
    print("\n[10] Within-group correlations (all significant metrics)...")
    plot_within_group_correlations(df_h, df_p, output_dir, stats=stats,
                                   healthy_label=healthy_label)

    # -- 10b. Full measure intercorrelation table --
    print("\n[10b] Metric intercorrelation table (significant metrics)...")
    metric_intercorrelation_table(df_h, df_p, stats, output_dir,
                                  healthy_label=healthy_label)

    # -- 11. Classification: done by nested_classifier.py, not here --
    # classification_analysis() selects features on the full sample, so it is not called.
    print("\n[11] Classification: delegated to nested_classifier.py "
          "(leaky in-module version intentionally NOT run).")

    # -- 11b. How the corrections and partial correlations are defined --
    print("\n[11b] Correction-family inventory...")
    # The heart-rate-adjusted family is sized from the columns present, with the analysis's own helper.
    _hr_members = hr_family_available(df_h, df_p)
    report_correction_family_inventory(output_dir, extra_families=[{
        "Family": "hr_adjusted",
        "N_Tests": len(_hr_members),
        "Corrected": "BH",
        "Members": "; ".join(_hr_members),
        "Used_In": "heart-rate-adjusted comparison (HR_Adjusted_Comparison.csv)",
        "Note": "partial Spearman with group, adjusted for mean heart rate "
                "(regress then rank); model-form and mean-NN sensitivities "
                "are reported beside it and are not corrected"}])
    print("\n[11c] Partial-correlation method...")
    report_partial_correlation_method(output_dir)

    # -- 12. Minimum detectable effect --
    print("\n[12] Minimum detectable effect...")
    report_minimum_detectable_effect(len(df_h), len(df_p), output_dir)

    # -- 12b. Detection floors for every subgroup contrast --
    # Sizes are derived from the data where available.
    _sub_sizes = [("Main comparison", len(df_h), len(df_p))]
    if "Diagnosis" in df_p.columns:
        _n_sz = int((df_p["Diagnosis"] == "schizophrenia").sum())
        _n_bp = int((df_p["Diagnosis"] == "bipolar").sum())
        _sub_sizes += [("Schizophrenia vs bipolar", _n_sz, _n_bp),
                       ("Schizophrenia vs controls", len(df_h), _n_sz),
                       ("Bipolar vs controls", len(df_h), _n_bp)]
    if "Quetiapine" in df_p.columns:
        _q = df_p["Quetiapine"].astype(str).str.lower().isin(["true", "yes", "1"])
        _sub_sizes += [("Quetiapine exposed vs unexposed",
                        int(_q.sum()), int((~_q).sum())),
                       ("Quetiapine exposed vs controls",
                        len(df_h), int(_q.sum()))]
    if "Comorbid_Substance" in df_p.columns:
        _c = df_p["Comorbid_Substance"].astype(str).str.lower().isin(
            ["true", "yes", "1"])
        _sub_sizes += [("Comorbidity-excluded vs controls",
                        len(df_h), int((~_c).sum()))]
    print("\n[12b] Detection floors for subgroup contrasts...")
    report_subgroup_detectable_effects(_sub_sizes, output_dir)

    # -- 13. Parametric comparison (Welch t and Cohen's d) --
    print("\n[13] Parametric comparison (Welch t-test + Cohen's d)...")
    # Same family as the primary table, so the two tables differ only by test (Mann-Whitney against Welch).
    par_stats = run_parametric_comparison(df_h, df_p, primary)
    save_parametric_table(par_stats, output_dir, healthy_label=healthy_label)
    plot_cohens_d_forest(par_stats, output_dir, healthy_label=healthy_label)

    # -- 14. Distribution diagnostics (Shapiro-Wilk and Levene) --
    print("\n[14] Distribution diagnostics (normality + variance)...")
    distribution_diagnostics(df_h, df_p, primary, output_dir,
                             healthy_label=healthy_label)

    # -- 15. Bootstrap effect-size CIs --
    print("\n[15] Bootstrap effect-size confidence intervals...")
    _boot_result = bootstrap_effect_sizes(df_h, df_p, primary, output_dir,
                           healthy_label=healthy_label)

    # -- 15b. Attach the bootstrap intervals to the primary table --
    # Done here, after the bootstrap exists (the summary table is written at step 1).
    print("\n[15b] Attaching bootstrap intervals to the primary table...")
    attach_bootstrap_intervals(output_dir)

    # -- 16. MFDFA group comparison --
    print("\n[16] MFDFA group comparison (multifractal singularity spectrum)...")
    mfdfa_group_comparison(df_h, df_p, output_dir, healthy_label=healthy_label)

    # -- 17. Recurrence network group comparison --
    print("\n[17] Recurrence network group comparison (attractor topology)...")
    rna_group_comparison(df_h, df_p, output_dir, healthy_label=healthy_label)

    # -- 17a. Manuscript table and test provenance --
    print("\n[17a] Manuscript Table 1 and test implementation record...")
    write_manuscript_table1(stats, output_dir, healthy_label=healthy_label)
    write_test_implementation(output_dir)
    ambulatory_fraction_test(output_dir)

    # -- 17b. Deterministic normalization checks (no correction) --
    print("\n[17b] Deterministic normalisation checks (reported outside FDR)...")
    deterministic_checks(df_h, df_p, output_dir)

    # -- 17c. Slope band sweep --
    print("\n[17c] Spectral exponent across fitted bands...")
    beta_band_sweep_comparison(df_h, df_p, output_dir)

    # -- 17d. Stationarity and entropy tolerance --
    print("\n[17d] Stationarity diagnostics...")
    stationarity_report(df_h, df_p, output_dir)
    print("\n[17e] Entropy tolerance sensitivity...")
    entropy_tolerance_report(df_h, df_p, output_dir)

    # -- 18. Mayer-wave control --
    print("\n[18] Mayer-wave control: beta with the 0.1 Hz resonance removed...")
    mayer_wave_analysis(df_h, df_p, output_dir, healthy_label=healthy_label)

    # -- 19. Quantization (resolution) control --
    print("\n[19] Quantization control (dither + matched coarsening)...")
    quantization_sensitivity(df_h, df_p, output_dir)

    # -- 19b. Embedding sensitivity --
    print("\n[19b] Selected-embedding sensitivity...")
    report_embedding_sensitivity(df_h, df_p, output_dir)

    # -- 19c. MSE scale-retention audit --
    print("\n[19c] MSE scale-retention audit...")
    report_mse_scale_audit(df_h, df_p, output_dir)

    # -- 20. Surrogate-normalized entropy --
    print("\n[20] Surrogate-normalised entropy comparison...")
    surrogate_normalized_entropy(df_h, df_p, output_dir)

    # -- 20b. Surrogate distributions and the direction of each subject's deviation --
    print("\n[20b] Surrogate distribution summary...")
    report_surrogate_distributions(df_h, df_p, output_dir)

    # -- 21. beta = 2*alpha - 1 consistency --
    print("\n[21] Internal consistency: beta against 2*alpha2 - 1...")
    beta_dfa_consistency(df_h, df_p, output_dir)

    # -- 21b. Frequency band covered by each beat-scale range --
    # The slope is fitted in Hz and DFA in beats, and the conversion depends on heart period.
    print("\n[21b] Scale-to-frequency band correspondence...")
    report_scale_band_correspondence(df_h, df_p, output_dir)

    # -- 22. Non-redundant correction family --
    print("\n[22] FDR re-run excluding definitionally collinear metrics...")
    nonredundant_family_comparison(df_h, df_p, output_dir)

    # -- 22b. Effective number of independent tests (sensitivity only; the primary correction stays BH
    # at the nominal size) --
    print("\n[22b] Effective number of independent tests (Li-Ji)...")
    report_effective_number_of_tests(df_h, df_p, PRIMARY_METRICS, output_dir)

    # -- 22c. Individuals in clinically recognizable ranges, counted from per-subject values --
    print("\n[22c] Clinical-range counts...")
    report_clinical_range_counts(df_h, df_p, output_dir)

    # -- 22d. Quantization noise floor in the fitted band --
    print("\n[22d] Quantization noise floor...")
    report_quantization_noise_floor(df_h, df_p, output_dir)

    # report_excluded_subjects() is not called here: this module reads results files, which contain only
    # included subjects.

    # -- 23. Series length and respiratory covariates --
    print("\n[23] Series length + respiratory frequency checks...")
    length_and_respiration_checks(df_h, df_p, output_dir)

    # -- 24. pNN50 floor effect --
    print("\n[24] pNN50 floor-effect report...")
    pnn50_floor_report(df_h, df_p, output_dir)

    # -- 25. Subject-level metadata analyses --
    # Diagnosis, age, sex, comorbidity and quetiapine, from the released metadata.
    print("\n[25] Subject-level metadata analyses...")
    run_metadata_analyses(df_h, df_p, output_dir)

    # -- 26-28. Heart-rate adjustment, VLF decomposition, amplitude and spectral measures on the 7.8 ms grid --
    print("\n[26] Heart-rate-adjusted comparison (own correction family)...")
    hr_adjusted_comparison(df_h, df_p, output_dir)
    print("\n[26b] Amplitude normalised by mean NN...")
    meannn_normalised_comparison(df_h, df_p, output_dir)
    print("\n[27] VLF decomposition...")
    vlf_decomposition_report(df_h, df_p, output_dir)
    print("\n[28] Re-quantization to the 7.8 ms grid (all measures)...")
    requantization_report(df_h, df_p, output_dir)

    print(f"\n{'='*65}")
    print(f"  Outputs saved to: {output_dir}/")
    print(f"{'='*65}")
    return df_h, df_p, stats, par_stats


# ---- Cross-control orchestrator (needs all three cohorts) ----

def run_cross_control_analysis(df_mitbih, df_hrvacc, df_psych,
                               stats_hrvacc, stats_mitbih, output_dir):
    """Analyses that need both control cohorts and the patients: control against control, TOST equivalence,
    three-group Kruskal-Wallis with Dunn post hoc, cross-control convergence, and the effect-against-
    confound scatter.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  CROSS-CONTROL & THREE-GROUP ANALYSES")
    print(f"{'='*65}")

    # Quantities fixed by construction (RQA_REC, RNA_MeanDegree) are excluded here too (their SDs are
    # zero). Family sizes: control against control 27, Kruskal-Wallis 27, Dunn 81 pairs.
    metrics = [m for m in METRIC_BATTERY
               if m not in DETERMINISTIC_CHECKS
               and m in df_mitbih.columns
               and m in df_hrvacc.columns
               and m in df_psych.columns]
    _excluded_degenerate = [m for m in METRIC_BATTERY
                            if m in DETERMINISTIC_CHECKS
                            and m in df_mitbih.columns]
    if _excluded_degenerate:
        print(f"    excluded from all cross-cohort families "
              f"(fixed by construction): {', '.join(_excluded_degenerate)}")

    print("\n[X1] Control vs control (confound magnitude)...")
    cvc = control_vs_control_analysis(df_mitbih, df_hrvacc, metrics, output_dir)

    print("\n[X2] Control equivalence (TOST, radar metrics)...")
    radar_present = [m for m in RADAR_METRICS if m in metrics]
    tost_equivalence(df_mitbih, df_hrvacc, radar_present, output_dir)

    print("\n[X3] Three-group omnibus (Kruskal-Wallis + Dunn)...")
    three_group_analysis(df_mitbih, df_hrvacc, df_psych, metrics, output_dir)

    print("\n[X4] Cross-control convergence...")
    cross_control_convergence(stats_hrvacc, stats_mitbih, output_dir)

    # -- Age-adjusted cross-cohort comparison --
    # The MIT-BIH cohort is younger and the slope depends on age. Skipped with a message if MIT-BIH ages are
    # unavailable.
    print("\n[cross] Age-adjusted cross-device comparison...")
    age_adjusted_cross_device(df_hrvacc, df_mitbih, df_psych, output_dir)

    print("\n[X5] Illness effect vs recording confound...")
    effect_vs_confound(stats_hrvacc, cvc, output_dir)

    print(f"\n{'='*65}")
    print(f"  Cross-control outputs saved to: {output_dir}/")
    print(f"{'='*65}")


# ---- Entry point ----

# ---- Additional analyses ----

def _ols_residuals(y, X):
    """Residuals of y after least-squares regression on the columns of X."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_spearman(target, outcome, covariates):
    """Partial Spearman correlation: regress both variables on the covariates by OLS, then Spearman-
    correlate the residuals (regress, then rank).
    """
    t = np.asarray(target, dtype=float)
    o = np.asarray(outcome, dtype=float)
    C = np.column_stack([np.asarray(c, dtype=float) for c in covariates])
    ok = np.isfinite(t) & np.isfinite(o) & np.all(np.isfinite(C), axis=1)
    if ok.sum() < 6:
        return np.nan, np.nan
    return spearmanr(_ols_residuals(t[ok], C[ok]), _ols_residuals(o[ok], C[ok]))


# Mann-Whitney configuration, set explicitly: two-sided, with continuity correction, so the reported
# p does not depend on the SciPy default.
MWU_USE_CONTINUITY = True
MWU_METHOD = "auto"


def _mw(h_vals, p_vals):
    """Mann-Whitney U with rank-biserial r; returns (r, p, n_h, n_p)."""
    h = pd.to_numeric(pd.Series(h_vals), errors="coerce").dropna().values
    p = pd.to_numeric(pd.Series(p_vals), errors="coerce").dropna().values
    if len(h) < 3 or len(p) < 3:
        return np.nan, np.nan, len(h), len(p)
    # Continuity setting passed explicitly.
    u, pv = mannwhitneyu(h, p, alternative="two-sided",
                         use_continuity=MWU_USE_CONTINUITY,
                         method=MWU_METHOD)
    return rank_biserial_r(u, len(h), len(p)), pv, len(h), len(p)


def _num_col(df, name):
    """Numeric Series for a column, or None if the column is absent or empty.

    pd.to_numeric(df.get(name)) returns NaN, not None, for a missing column, so the check is explicit.
    """
    if df is None or name not in getattr(df, "columns", []):
        return None
    s = pd.to_numeric(df[name], errors="coerce")
    return s if s.notna().sum() > 0 else None


def report_scale_band_correspondence(df_h, df_p, output_dir):
    """Frequency band covered by each DFA beat-scale range, per group.

    A scale of n beats corresponds to about 1 / (n * meanNN) Hz, and mean heart period differs between
    groups, so the same beat scales cover a higher band in patients. Group differences in alpha2, and in
    the beta = 2*alpha - 1 residual, therefore partly reflect a band shift. Reports the overlap of the
    slope's fitted band with each group's alpha bands.
    """
    rows = []
    for label, df in (("Healthy", df_h), ("Psychiatric", df_p)):
        nn = _num_col(df, "HRV_MeanNN")
        if nn is None or nn.notna().sum() == 0:
            hr = _num_col(df, "HRV_MeanHR")
            if hr is None or hr.notna().sum() == 0:
                continue
            nn = 60.0 / hr                      # seconds per beat
        else:
            nn = nn / 1000.0 if float(nn.mean()) > 10 else nn
        mean_nn = float(nn.mean())
        for tag, lo_b, hi_b in (("alpha1", 4, 16), ("alpha2", 16, 64)):
            # n beats = n * meanNN seconds = 1 / (n * meanNN) Hz; the larger scale gives the lower frequency.
            f_hi = 1.0 / (lo_b * mean_nn)
            f_lo = 1.0 / (hi_b * mean_nn)
            rows.append({
                "Group": label, "Exponent": tag,
                "Scale_Lo_Beats": lo_b, "Scale_Hi_Beats": hi_b,
                "Mean_NN_s": round(mean_nn, 4),
                "Freq_Lo_Hz": round(f_lo, 5), "Freq_Hi_Hz": round(f_hi, 5),
                "Beta_Band_Lo_Hz": BETA_BAND_FOR_REPORT[0],
                "Beta_Band_Hi_Hz": BETA_BAND_FOR_REPORT[1],
                "Overlap_With_Beta_Band_Hz": round(
                    max(0.0, min(f_hi, BETA_BAND_FOR_REPORT[1]) -
                        max(f_lo, BETA_BAND_FOR_REPORT[0])), 5),
            })
    if not rows:
        print("  Scale/band correspondence skipped: heart-rate columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Scale_Band_Correspondence.csv")
    out.to_csv(path, index=False)
    print(f"  Scale/band correspondence → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Group']:<12} {rw['Exponent']:<7} "
              f"{rw['Scale_Lo_Beats']}-{rw['Scale_Hi_Beats']} beats = "
              f"{rw['Freq_Lo_Hz']:.4f}-{rw['Freq_Hi_Hz']:.4f} Hz")
    return out


def report_effective_number_of_tests(df_h, df_p, metrics, output_dir):
    """Effective number of independent tests within a family (Li and Ji, 2005), from the full-precision
    correlation matrix. A sensitivity analysis only; the primary correction is BH at the nominal size.
    """
    cols = [m for m in metrics if m in df_h.columns and m in df_p.columns]
    pooled = pd.concat([
        df_h[cols].apply(pd.to_numeric, errors="coerce"),
        df_p[cols].apply(pd.to_numeric, errors="coerce")], axis=0)
    pooled = pooled.dropna(axis=1, how="all")
    corr = pooled.corr(method="spearman").values
    corr = np.nan_to_num(corr, nan=0.0)
    try:
        eig = np.linalg.eigvalsh(corr)
        eig = np.clip(eig, 0.0, None)
        # Li and Ji: each eigenvalue contributes its integer part plus a fractional term.
        m_eff = float(np.sum(
            (eig >= 1.0).astype(float) + (eig - np.floor(eig))))
    except Exception:
        m_eff = np.nan
    m_nominal = int(corr.shape[0])
    # The estimator is discontinuous at integer eigenvalues (six near-identical variables give 2.00, not
    # 1.00), so the leading eigenvalues are exported and the headline value is rounded to one decimal.
    try:
        _lead = np.sort(eig)[::-1][:5]
        _near_int = bool(np.any(np.abs(_lead - np.round(_lead)) < 0.01))
        _lead_str = "; ".join(f"{v:.4f}" for v in _lead)
    except Exception:
        _lead_str, _near_int = "", False
    out = pd.DataFrame([{
        "Family": "primary",
        "N_Nominal": m_nominal,
        "N_Effective_LiJi": round(m_eff, 1) if np.isfinite(m_eff) else np.nan,
        "N_Effective_LiJi_Full": (round(m_eff, 4)
                                  if np.isfinite(m_eff) else np.nan),
        "Leading_Eigenvalues": _lead_str,
        "Near_Integer_Eigenvalue": _near_int,
        "Alpha_Nominal": ALPHA,
        "Alpha_Effective": (round(ALPHA / m_eff, 5)
                            if np.isfinite(m_eff) and m_eff > 0 else np.nan),
        "Note": "sensitivity analysis only; the primary correction remains BH "
                "at the nominal family size. The Li-Ji estimator is "
                "discontinuous at integer eigenvalues and can shift by about "
                "one unit on numerical noise, so it is reported to one decimal",
    }])
    path = os.path.join(output_dir, "Effective_Number_Of_Tests.csv")
    out.to_csv(path, index=False)
    print(f"  Effective number of tests → {path}")
    print(f"    nominal {m_nominal}, effective {m_eff:.2f}")
    return out


def report_partial_correlation_method(output_dir):
    """Record how the partial correlations are computed: both raw variables regressed on the covariates by
    OLS, then the Spearman correlation of the residuals (regress, then rank; not rank, then regress).
    """
    out = pd.DataFrame([{
        "Quantity": "partial Spearman correlation",
        "Method": "regress-then-rank (OLS residuals of raw values, "
                  "then Spearman of residuals)",
        "Intercept_Included": True,
        "Alternative_Not_Used": "rank-then-regress (residualise ranks)",
        "Equivalent": False,
        "Note": "the two differ when the covariate relationship is monotone "
                "but nonlinear; this choice must be stated in the Methods",
    }])
    path = os.path.join(output_dir, "Partial_Correlation_Method.csv")
    out.to_csv(path, index=False)
    print(f"  Partial-correlation method → {path}")
    return out


def report_surrogate_distributions(df_h, df_p, output_dir):
    """Group summary of the IAAFT surrogate distributions (SURR_<metric>_Mean and _SD), and the direction of
    each subject's deviation (the test is two-sided, so above and below are counted separately).
    """
    rows = []
    for metric in ("SampEn", "DFA_Alpha1", "MSE_Mean"):
        for gname, df in (("Healthy", df_h), ("Psychiatric", df_p)):
            obs = _num_col(df, metric)
            smean = _num_col(df, f"SURR_{metric}_Mean")
            ssd = _num_col(df, f"SURR_{metric}_SD")
            sig = df.get(f"SURR_{metric}_Sig")
            direction = df.get(f"SURR_{metric}_Direction")
            if obs is None or smean is None:
                continue
            n_sig = int(pd.Series(sig).fillna(False).astype(bool).sum()) \
                if sig is not None else np.nan
            if direction is not None:
                d = pd.Series(direction).astype(str)
                s = pd.Series(sig).fillna(False).astype(bool)
                n_above = int((s & (d == "above")).sum())
                n_below = int((s & (d == "below")).sum())
            else:
                n_above = n_below = np.nan
            rows.append({
                "Metric": metric, "Group": gname, "N": int(obs.notna().sum()),
                "Observed_Mean": float(obs.mean()),
                "Surrogate_Mean": float(smean.mean()),
                "Surrogate_SD_Mean": (float(ssd.mean())
                                      if ssd is not None else np.nan),
                "N_Differing": n_sig,
                "N_Above_Surrogates": n_above,
                "N_Below_Surrogates": n_below,
            })
    if not rows:
        print("  Surrogate distribution summary skipped: SURR_* columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Surrogate_Distribution_Summary.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Surrogate distribution summary → {path}")
    return out


def report_embedding_sensitivity(df_h, df_p, output_dir):
    """Selected-embedding check: the per-subject mutual-information delay and false-nearest-neighbour
    dimension by group, and the RQA group contrast at the selected embedding beside the contrast at the
    primary embedding (dim = 2, delay = 1).
    """
    rows = []
    for col in ("Embed_Delay_MI", "Embed_Dim_FNN"):
        h = _num_col(df_h, col)
        p = _num_col(df_p, col)
        if h is None or p is None or h.notna().sum() == 0:
            continue
        r, pv, nh, npz = _mw(h, p)
        rows.append({"Quantity": col, "Type": "selected_parameter",
                     "Healthy_Median": float(h.median()),
                     "Psych_Median": float(p.median()),
                     "r_effect": r, "p_raw": pv,
                     "N_Healthy": nh, "N_Psych": npz})
    for primary, selected in (("RQA_DET", "RQA_DET_SelEmbed"),
                              ("RQA_LAM", "RQA_LAM_SelEmbed")):
        for col, tag in ((primary, "primary_embedding"),
                         (selected, "selected_embedding")):
            h = _num_col(df_h, col)
            p = _num_col(df_p, col)
            if h is None or p is None or h.notna().sum() == 0:
                continue
            r, pv, nh, npz = _mw(h, p)
            rows.append({"Quantity": primary, "Type": tag,
                         "Healthy_Median": float(h.median()),
                         "Psych_Median": float(p.median()),
                         "r_effect": r, "p_raw": pv,
                         "N_Healthy": nh, "N_Psych": npz})
    if not rows:
        print("  Embedding sensitivity skipped: Embed_* columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Embedding_Sensitivity.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Embedding sensitivity → {path}")
    return out


def report_mse_scale_audit(df_h, df_p, output_dir):
    """Whether the multiscale entropy result depends on unequal scale retention.

    Subjects with fewer beats retain fewer scales, and beat count follows heart rate. Compares MSE_Mean
    with MSE_Mean_CommonScales, averaged over a scale set fixed in advance and common to every subject.
    """
    rows = []
    for col, label in (("MSE_Mean", "all_retained_scales"),
                       ("MSE_Mean_CommonScales", "common_scale_set")):
        h = _num_col(df_h, col)
        p = _num_col(df_p, col)
        if h is None or p is None or h.notna().sum() == 0:
            continue
        r, pv, nh, npz = _mw(h, p)
        rows.append({"Quantity": "MSE_Mean", "Scale_Basis": label,
                     "Healthy_Mean": float(h.mean()),
                     "Psych_Mean": float(p.mean()),
                     "r_effect": r, "p_raw": pv,
                     "N_Healthy": nh, "N_Psych": npz})
    for col in ("MSE_Scales_Retained", "MSE_MaxScale_Retained"):
        h = _num_col(df_h, col)
        p = _num_col(df_p, col)
        if h is None or p is None or h.notna().sum() == 0:
            continue
        r, pv, nh, npz = _mw(h, p)
        rows.append({"Quantity": col, "Scale_Basis": "retention_check",
                     "Healthy_Mean": float(h.mean()),
                     "Psych_Mean": float(p.mean()),
                     "r_effect": r, "p_raw": pv,
                     "N_Healthy": nh, "N_Psych": npz})
    if not rows:
        print("  MSE scale audit skipped: MSE scale columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "MSE_Scale_Audit.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  MSE scale audit → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Quantity']:<22} {rw['Scale_Basis']:<20} "
              f"r={rw['r_effect']:+.3f} (p={rw['p_raw']:.3g})")
    return out


def report_quantization_noise_floor(df_h, df_p, output_dir,
                                    steps_s=None):
    """Power from quantization error relative to signal power inside the fitted spectral band.

    Uniform quantization at step q adds approximately white noise of variance q^2 / 12. Reported for the
    Polar step, the MIT-BIH step and any additional steps supplied.
    """
    if steps_s is None:
        # QUANT_STEP_S is the measured 1 ms grid; the nominal Bluetooth unit is kept as a reference row.
        steps_s = [("Polar H10", QUANT_STEP_S),
                   ("MIT-BIH", 1.0 / 128.0),
                   ("BLE nominal", 1.0 / 1024.0)]
    rows = []
    for gname, df in (("Healthy", df_h), ("Psychiatric", df_p)):
        # Signal power in the fitted band, converted from the stored ms^2 to s^2.
        lf = _num_col(df, "LF_Power_ms2")
        if lf is None:
            continue
        sig_power_s2 = float(lf.mean()) * 1e-6
        band_width_hz = BETA_BAND_FOR_REPORT[1] - BETA_BAND_FOR_REPORT[0]
        for label, q in steps_s:
            # White quantization noise: variance q^2/12 over the Nyquist band of the interpolated series (2 Hz).
            noise_var_s2 = (q ** 2) / 12.0
            noise_psd = noise_var_s2 / 2.0           # per Hz, over 0 to 2 Hz
            noise_in_band = noise_psd * band_width_hz
            rows.append({
                "Group": gname, "Acquisition": label,
                "Step_ms": round(q * 1000, 4),
                "Signal_Power_In_Band_s2": sig_power_s2,
                "Noise_Power_In_Band_s2": noise_in_band,
                "Noise_To_Signal_Pct": (round(100.0 * noise_in_band /
                                              sig_power_s2, 4)
                                        if sig_power_s2 > 0 else np.nan),
            })
    if not rows:
        print("  Quantization noise floor skipped: power columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Quantization_Noise_Floor.csv")
    out.to_csv(path, index=False, float_format="%.10g")
    print(f"  Quantization noise floor → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Group']:<12} {rw['Acquisition']:<12} "
              f"step {rw['Step_ms']:>7} ms  noise/signal = "
              f"{rw['Noise_To_Signal_Pct']:.4f}%")
    return out


def report_clinical_range_counts(df_h, df_p, output_dir):
    """Number of subjects in clinically recognizable ranges, counted from per-subject values.

    These are 30-minute mixed-activity windows that include an instructed walk, not resting recordings,
    so the counts describe this window and cannot be compared with resting normative ranges.
    """
    thresholds = [
        ("HRV_MeanHR", "mean HR > 100 bpm", lambda s: s > 100),
        ("HRV_MeanHR", "mean HR > 110 bpm", lambda s: s > 110),
        ("HRV_RMSSD", "RMSSD < 10 ms", lambda s: s < 10),
        ("HRV_RMSSD", "RMSSD < 5 ms", lambda s: s < 5),
        ("HRV_SDNN", "SDNN < 20 ms", lambda s: s < 20),
    ]
    rows = []
    for col, label, test in thresholds:
        for gname, df in (("Healthy", df_h), ("Psychiatric", df_p)):
            s = _num_col(df, col)
            if s is None:
                continue
            n_tot = int(s.notna().sum())
            n_hit = int(test(s.dropna()).sum())
            rows.append({
                "Criterion": label, "Group": gname,
                "N_Meeting": n_hit, "N_Total": n_tot,
                "Pct": round(100.0 * n_hit / n_tot, 1) if n_tot else np.nan,
                "Recording_Condition": "30-min mixed-activity window "
                                       "including an instructed walk; not "
                                       "comparable with resting norms",
            })
    if not rows:
        print("  Clinical range counts skipped: HRV columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Clinical_Range_Counts.csv")
    out.to_csv(path, index=False)
    print(f"  Clinical range counts → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Criterion']:<20} {rw['Group']:<12} "
              f"{rw['N_Meeting']}/{rw['N_Total']}")
    return out


def report_excluded_subjects(exclusions, output_dir):
    """Which subjects were excluded, and the criterion each failed.

    exclusions : list of (cohort, record_id, criterion, value)
    """
    if not exclusions:
        print("  No exclusions recorded.")
        return None
    out = pd.DataFrame(exclusions,
                       columns=["Cohort", "Record_ID", "Criterion", "Value"])
    path = os.path.join(output_dir, "Excluded_Subjects.csv")
    out.to_csv(path, index=False)
    print(f"  Excluded subjects → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Cohort']:<14} {rw['Record_ID']:<12} "
              f"{rw['Criterion']}")
    return out


def attach_bootstrap_intervals(output_dir):
    """Add the bootstrap effect-size intervals to the primary table. Call after bootstrap_effect_sizes();
    both are indexed by display label. Raises if the merge yields no intervals.
    """
    stats_path = os.path.join(output_dir, "group_comparison_stats.csv")
    boot_path = os.path.join(output_dir, "Bootstrap_Effect_Sizes.csv")
    if not os.path.exists(stats_path):
        print("    !! primary table missing; intervals NOT attached")
        return None
    if not os.path.exists(boot_path):
        print("    !! Bootstrap_Effect_Sizes.csv missing; intervals NOT "
              "attached. Table 1 will have no intervals, which Reviewer 4 and "
              "the editor both make a condition.")
        return None
    g = pd.read_csv(stats_path, index_col=0)
    b = pd.read_csv(boot_path, index_col=0)
    cols = [c for c in ("r_CI_low", "r_CI_high", "Cohen_d",
                        "d_CI_low", "d_CI_high") if c in b.columns]
    if not cols:
        print(f"    !! no interval columns in {boot_path}")
        return None
    g = g.drop(columns=[c for c in cols if c in g.columns], errors="ignore")
    g = g.join(b[cols], how="left")
    n_with = int(g["r_CI_low"].notna().sum()) if "r_CI_low" in g.columns else 0
    if n_with < len(g) / 2:
        missing = [i for i in g.index if pd.isna(g.at[i, "r_CI_low"])][:5]
        print(f"    !! interval merge matched only {n_with} of {len(g)} "
              f"metrics. Index keys are not aligning; unmatched examples: "
              f"{missing}")
    else:
        print(f"    bootstrap intervals attached for {n_with} of {len(g)} "
              f"metrics")
    g.to_csv(stats_path, float_format="%.6f")
    return g


def report_correction_family_inventory(output_dir, extra_families=None):
    """One table naming every multiple-comparison family and its size. "Dynamic" sizes depend on the
    columns present and are filled in by the callers; fixed sizes come from the module constants.
    """
    fams = [
        {"Family": "primary",
         "N_Tests": len(PRIMARY_METRICS),
         "Corrected": "BH",
         "Members": "; ".join(PRIMARY_METRICS),
         "Used_In": "primary group comparison table",
         "Note": "excludes MFDFA, RNA, undefined and construction-fixed "
                 "quantities, and ApEn as statistically redundant with SampEn "
                 "(r = 0.98); ApEn is reported outside the family"},
        {"Family": "primary_nonredundant",
         "N_Tests": len(NONREDUNDANT_PRIMARY),
         "Corrected": "BH",
         "Members": "; ".join(NONREDUNDANT_PRIMARY),
         "Used_In": "non-redundant sensitivity table",
         "Note": "drops definitionally collinear duplicates: "
                 + "; ".join(COLLINEAR_DUPLICATES)},
        {"Family": "mfdfa",
         "N_Tests": len(MFDFA_FAMILY),
         "Corrected": "BH",
         "Members": "; ".join(MFDFA_FAMILY),
         "Used_In": "multifractal comparison",
         "Note": "MFDFA_Asymmetry is undefined under positive-q only and is "
                 "not a member"},
        {"Family": "recurrence_network",
         "N_Tests": len(RNA_FAMILY),
         "Corrected": "BH",
         "Members": "; ".join(RNA_FAMILY),
         "Used_In": "recurrence-network comparison",
         "Note": "RNA_MeanDegree excluded: fixed by construction under "
                 "exact-count thresholding"},
        {"Family": "surrogate_normalised",
         "N_Tests": 6,
         "Corrected": "BH",
         "Members": "SampEn raw; SampEn Z; DFA_Alpha1 raw; DFA_Alpha1 Z; "
                    "MSE_Mean raw; MSE_Mean Z",
         "Used_In": "surrogate-normalised entropy comparison",
         "Note": "three metrics x two scales. Raw p-values are printed "
                 "alongside adjusted ones so the correction is checkable"},
        {"Family": "construction_fixed",
         "N_Tests": 0,
         "Corrected": "none",
         "Members": "; ".join(DETERMINISTIC_CHECKS),
         "Used_In": "normalisation checks only",
         "Note": "fixed at the target recurrence rate for every subject; "
                 "reported outside every family and excluded from all of them"},
    ]
    if extra_families:
        fams.extend(extra_families)
    out = pd.DataFrame(fams)
    path = os.path.join(output_dir, "Correction_Family_Inventory.csv")
    out.to_csv(path, index=False)
    print(f"  Correction-family inventory → {path}")
    for f in fams:
        print(f"    {f['Family']:<24} n={f['N_Tests']:<3} {f['Corrected']}")
    return out


def report_subgroup_detectable_effects(subgroup_sizes, output_dir,
                                       power=0.80):
    """Detection floor (80% power) for every subgroup contrast the paper draws a conclusion from.

    subgroup_sizes : list of (label, n_group_a, n_group_b)
    """
    rows = []
    for label, n_a, n_b in subgroup_sizes:
        if min(n_a, n_b) < 2:
            continue
        d_min, r_rank, r_point = minimum_detectable_effect(n_a, n_b,
                                                           power=power)
        rows.append({
            "Contrast": label, "N_A": n_a, "N_B": n_b, "Power": power,
            "Alpha": ALPHA,
            "Min_Detectable_d": round(d_min, 3),
            "Min_Detectable_r_rankbiserial": round(r_rank, 3),
            "Min_Detectable_r_pointbiserial": round(r_point, 3),
        })
    if not rows:
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir,
                        "Minimum_Detectable_Effect_Subgroups.csv")
    out.to_csv(path, index=False)
    print(f"  Subgroup detection floors → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Contrast']:<34} N={rw['N_A']}/{rw['N_B']}  "
              f"smallest detectable r = "
              f"{rw['Min_Detectable_r_rankbiserial']}")
    return out


def report_minimum_detectable_effect(n_h, n_p, output_dir):
    """Minimum detectable effect for the main comparison (replaces post-hoc power)."""
    rows = []
    for power in (0.80, 0.90):
        d_min, r_rank, r_point = minimum_detectable_effect(n_h, n_p,
                                                           power=power)
        rows.append({"Power": power, "Alpha": ALPHA,
                     "N_Healthy": n_h, "N_Psych": n_p,
                     "Min_Detectable_d": round(d_min, 3),
                     # Min_Detectable_r is the rank-biserial floor; the point-biserial value is kept beside it.
                     "Min_Detectable_r": round(r_rank, 3),
                     "Min_Detectable_r_pointbiserial": round(r_point, 3)})
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Minimum_Detectable_Effect.csv")
    df.to_csv(path, index=False)
    print(f"  Minimum detectable effect → {path}")
    for _, rw in df.iterrows():
        print(f"    At {rw['Power']:.0%} power with N={n_h}/{n_p}: "
              f"smallest detectable d = {rw['Min_Detectable_d']}, "
              f"r = {rw['Min_Detectable_r']}")
    return df


def mayer_wave_analysis(df_h, df_p, output_dir, healthy_label="Healthy"):
    """Slope group difference under every slope estimator, side by side.

    The fitting band contains the baroreflex (Mayer-wave) resonance near 0.1 Hz, which flattens the fitted
    slope where it is prominent. On synthetic 1/f signals differing only in 0.1 Hz peak amplitude,
    linear-bin OLS produced a spurious group difference of +0.13 to +0.22, while the band-excised fit and
    the explicit peak model produced essentially none.
    """
    variants = [
        ("PSD_Beta",             "Legacy: linear-bin OLS, 0.01-0.15 Hz"),
        ("PSD_Beta_LogBin",      "Log-binned OLS (equal leverage across band)"),
        ("PSD_Beta_NoMayer",     "PRIMARY CONTROL: log-binned, 0.07-0.13 Hz excised"),
        ("PSD_Beta_Specparam",   "PRIMARY CONTROL: aperiodic + Gaussian peak model"),
        ("PSD_Beta_IRASA",       "IRASA (biased at this bandwidth; see pipeline)"),
        ("Mayer_Peak_Amp",       "Fitted Mayer peak amplitude"),
        ("Mayer_Peak_Freq",      "Fitted Mayer peak frequency (Hz)"),
        ("LF_Power_ms2",         "LF power (drives the peak)"),
    ]
    rows = []
    for col, desc in variants:
        if col not in df_h.columns or col not in df_p.columns:
            continue
        r, pv, nh, npat = _mw(df_h[col], df_p[col])
        hm = pd.to_numeric(df_h[col], errors="coerce").mean()
        pm = pd.to_numeric(df_p[col], errors="coerce").mean()
        rows.append({"Estimator": col, "Description": desc,
                     "Healthy_Mean": hm, "Psych_Mean": pm,
                     "Abs_Diff": pm - hm, "r_effect": r, "p_raw": pv,
                     "N_Healthy": nh, "N_Psych": npat})
    # The peak model sometimes converges onto a parameter bound, where it absorbs band curvature and
    # inflates the exponent, so the comparison is repeated on fits that did not reach a bound.
    if ("Specparam_At_Bound" in df_h.columns and
            "PSD_Beta_Specparam" in df_h.columns):
        def _clean(d):
            flag = d["Specparam_At_Bound"].astype(str).str.lower().eq("true")
            return d.loc[~flag, "PSD_Beta_Specparam"]
        hc, pc = _clean(df_h), _clean(df_p)
        if len(hc) >= 5 and len(pc) >= 5:
            r, pv, nh, npat = _mw(hc, pc)
            rows.append({"Estimator": "PSD_Beta_Specparam_CleanFits",
                         "Description": "Peak model, bound-hitting fits excluded",
                         "Healthy_Mean": hc.mean(), "Psych_Mean": pc.mean(),
                         "Abs_Diff": pc.mean() - hc.mean(), "r_effect": r,
                         "p_raw": pv, "N_Healthy": nh, "N_Psych": npat})

    if not rows:
        print("  Mayer-wave analysis skipped: beta variant columns absent. "
              "Re-run pipeline.py first.")
        return None
    out = pd.DataFrame(rows)
    out["p_adj"] = benjamini_hochberg(out["p_raw"].fillna(1.0).values)
    path = os.path.join(output_dir, "Mayer_Wave_Control.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Mayer-wave control → {path}")
    print(f"  {'Estimator':<24}{'H mean':>9}{'P mean':>9}{'diff':>8}"
          f"{'r':>8}{'p_raw':>10}")
    for _, rw in out.iterrows():
        print(f"  {rw['Estimator']:<24}{rw['Healthy_Mean']:>9.3f}"
              f"{rw['Psych_Mean']:>9.3f}{rw['Abs_Diff']:>8.3f}"
              f"{rw['r_effect']:>8.3f}{rw['p_raw']:>10.4g}")

    # Summary line
    try:
        base = out.loc[out["Estimator"] == "PSD_Beta", "r_effect"].values[0]
        for ctrl in ("PSD_Beta_NoMayer", "PSD_Beta_Specparam"):
            sel = out.loc[out["Estimator"] == ctrl, "r_effect"].values
            if len(sel) and np.isfinite(sel[0]) and np.isfinite(base) and base != 0:
                keep = 100.0 * sel[0] / base
                print(f"  --> {ctrl} retains {keep:.0f}% of the legacy effect size.")
    except Exception:
        pass
    return out


def quantization_sensitivity(df_h, df_p, output_dir):
    """Group differences in the entropy and recurrence measures under the resolution controls.

    The Polar H10 export is on a 1.000 ms grid while RMSSD is about half as large in patients, so the
    step-to-amplitude ratio roughly doubles between groups. Coarser effective resolution increases exact
    matches, lowering sample and approximate entropy and raising determinism. Each measure is compared as
    observed, with dither at the quantization step, and with both groups coarsened to a common
    step/RMSSD ratio.
    """
    # Recurrence-network measures are included when rna_controls.py has merged its columns.
    pairs = [("SampEn", "SampEn_Dither", "SampEn_MatchedQuant"),
             ("ApEn",   "ApEn_Dither",   "ApEn_MatchedQuant"),
             ("PermEn", "PermEn_Dither", "PermEn_MatchedQuant"),
             ("RQA_DET", "RQA_DET_Dither", "RQA_DET_MatchedQuant"),
             ("RQA_LAM", "RQA_LAM_Dither", "RQA_LAM_MatchedQuant"),
             ("RNA_ClusteringCoeff", "RNA_ClusteringCoeff_Dither",
              "RNA_ClusteringCoeff_MatchedQuant"),
             ("RNA_Transitivity", "RNA_Transitivity_Dither",
              "RNA_Transitivity_MatchedQuant"),
             # Multiscale entropy under dither and matched coarsening.
             ("MSE_Mean", "MSE_Mean_Dither", "MSE_Mean_MatchedQuant")]
    rows = []
    for observed, dither, matched in pairs:
        # The label carries the primary matched target ratio (QUANT_TARGET_RATIO).
        _matched_label = f"matched_quant_r{int(round(QUANT_TARGET_RATIO*1000)):03d}"
        for label, col in (("observed", observed), ("dither", dither),
                           (_matched_label, matched)):
            if col not in df_h.columns or col not in df_p.columns:
                continue
            r, pv, _, _ = _mw(df_h[col], df_p[col])
            # U is recorded too, since two conditions can give the same rank-biserial r.
            _hv = pd.to_numeric(df_h[col], errors="coerce").dropna()
            _pv_ = pd.to_numeric(df_p[col], errors="coerce").dropna()
            try:
                _u, _ = mannwhitneyu(_hv.values, _pv_.values,
                                     alternative="two-sided")
            except Exception:
                _u = np.nan
            rows.append({"Metric": observed, "Condition": label,
                         "Column": col, "r_effect": r, "p_raw": pv,
                         "U_stat": _u,
                         "N_Healthy": int(len(_hv)),
                         "N_Psych": int(len(_pv_))})
    # Sweep of alternative coarsening targets. A target matches only subjects whose observed step/RMSSD
    # ratio is below it, so each target is evaluated on the subsample it matches, with the retained N
    # reported.
    ratio_h = _num_col(df_h, "Quant_Step_Over_RMSSD")
    ratio_p = _num_col(df_p, "Quant_Step_Over_RMSSD")
    # Targets and tags derived from QUANT_TARGET_SWEEP as in the pipeline.
    _sweep = [(f"MQ{int(round(t * 100)):02d}", float(t))
              for t in QUANT_TARGET_SWEEP]
    for tag, target in _sweep:
        if ratio_h is None or ratio_p is None:
            break
        keep_h = ratio_h <= target
        keep_p = ratio_p <= target
        for base in ("SampEn", "ApEn", "PermEn", "RQA_DET", "RQA_LAM",
                     "MSE_Mean"):
            col = f"{base}_{tag}"
            if col not in df_h.columns or col not in df_p.columns:
                continue
            # (a) all subjects, including unmatched ones
            r_all, p_all, _, _ = _mw(df_h[col], df_p[col])
            rows.append({"Metric": base, "Condition": f"matched_{tag}_allsubj",
                         "Column": col, "r_effect": r_all, "p_raw": p_all,
                         "N_Healthy": int(len(df_h)), "N_Psych": int(len(df_p))})
            # (a2) observed values on the same subsample, separating subject loss from the coarsening.
            if keep_h.sum() >= 5 and keep_p.sum() >= 5 and base in df_h.columns:
                r_o, p_o, _, _ = _mw(df_h.loc[keep_h.values, base],
                                     df_p.loc[keep_p.values, base])
                rows.append({"Metric": base,
                             "Condition": f"observed_{tag}_matchedsubsample",
                             "Column": base, "r_effect": r_o, "p_raw": p_o,
                             "N_Healthy": int(keep_h.sum()),
                             "N_Psych": int(keep_p.sum())})

            # (b) matched subsample only
            if keep_h.sum() >= 5 and keep_p.sum() >= 5:
                r_m, p_m, _, _ = _mw(df_h.loc[keep_h.values, col],
                                     df_p.loc[keep_p.values, col])
                rows.append({"Metric": base,
                             "Condition": f"matched_{tag}_matchedsubsample",
                             "Column": col, "r_effect": r_m, "p_raw": p_m,
                             "N_Healthy": int(keep_h.sum()),
                             "N_Psych": int(keep_p.sum())})

    # The step-to-RMSSD ratio
    if "Quant_Step_Over_RMSSD" in df_h.columns:
        r, pv, _, _ = _mw(df_h["Quant_Step_Over_RMSSD"],
                          df_p["Quant_Step_Over_RMSSD"])
        rows.append({"Metric": "Quant_Step_Over_RMSSD", "Condition": "observed",
                     "Column": "Quant_Step_Over_RMSSD", "r_effect": r,
                     "p_raw": pv})
    if "Ordinal_Tie_Rate" in df_h.columns:
        r, pv, _, _ = _mw(df_h["Ordinal_Tie_Rate"], df_p["Ordinal_Tie_Rate"])
        rows.append({"Metric": "Ordinal_Tie_Rate", "Condition": "observed",
                     "Column": "Ordinal_Tie_Rate", "r_effect": r, "p_raw": pv})
    # Tie rate under each control condition, with group medians, to show whether matched coarsening
    # equalized the tie burden.
    _matched_lab = f"matched_quant_r{int(round(QUANT_TARGET_RATIO*1000)):03d}"
    for label, col in ([("dither", "Ordinal_Tie_Rate_Dither"),
                        (_matched_lab, "Ordinal_Tie_Rate_MatchedQuant")]
                       + [(f"matched_{tag}_allsubj", f"Ordinal_Tie_Rate_{tag}")
                          for tag, _ in _sweep]):
        if col not in df_h.columns or col not in df_p.columns:
            continue
        r, pv, nh, npat = _mw(df_h[col], df_p[col])
        rows.append({"Metric": "Ordinal_Tie_Rate", "Condition": label,
                     "Column": col, "r_effect": r, "p_raw": pv,
                     "N_Healthy": nh, "N_Psych": npat,
                     "Healthy_Median": float(pd.to_numeric(
                         df_h[col], errors="coerce").median()),
                     "Psych_Median": float(pd.to_numeric(
                         df_p[col], errors="coerce").median())})

    # The smallest sweep target exceeding the pooled maximum observed ratio (matches every subject).
    if "Quant_Step_Over_RMSSD" in df_h.columns:
        pooled_max = float(max(
            pd.to_numeric(df_h["Quant_Step_Over_RMSSD"], errors="coerce").max(),
            pd.to_numeric(df_p["Quant_Step_Over_RMSSD"], errors="coerce").max()))
        usable = [t for t in QUANT_TARGET_SWEEP if t > pooled_max]
        print(f"    pooled max step/RMSSD = {pooled_max:.4f}")
        if usable:
            print(f"    targets matching ALL subjects: {usable} -> report "
                  f"matched_MQ{int(round(usable[0]*100)):02d}_allsubj")
        else:
            print("    NO sweep target matches every subject. Report the "
                  "*_matchedsubsample rows, which restrict to the subjects a "
                  "given target genuinely equalises, and state the retained N.")
            for t in QUANT_TARGET_SWEEP:
                kh = int((pd.to_numeric(df_h["Quant_Step_Over_RMSSD"],
                                        errors="coerce") <= t).sum())
                kp = int((pd.to_numeric(df_p["Quant_Step_Over_RMSSD"],
                                        errors="coerce") <= t).sum())
                print(f"      target {t:.2f}: retains {kh}/{len(df_h)} controls, "
                      f"{kp}/{len(df_p)} patients")
    if not rows:
        print("  Quantization control skipped: columns absent. Re-run pipeline.py.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Quantization_Control.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Quantization control → {path}")
    for metric in out["Metric"].unique():
        sub = out[out["Metric"] == metric]
        parts = "  ".join(f"{rw['Condition']}: r={rw['r_effect']:+.3f} "
                          f"(p={rw['p_raw']:.3g})" for _, rw in sub.iterrows())
        print(f"    {metric:<22} {parts}")
    return out


def surrogate_normalized_entropy(df_h, df_p, output_dir):
    """Group comparison of surrogate Z-scores for the entropy measures.

    IAAFT surrogates preserve the amplitude distribution and linear autocorrelation, so a difference that
    survives on Z-scores exceeds what those predict.
    """
    rows = []
    for metric in ("SampEn", "DFA_Alpha1", "MSE_Mean"):
        raw_col = metric
        z_col   = f"SURR_{metric}_Z"
        for label, col in (("raw", raw_col), ("surrogate_Z", z_col)):
            if col not in df_h.columns or col not in df_p.columns:
                continue
            r, pv, _, _ = _mw(df_h[col], df_p[col])
            rows.append({"Metric": metric, "Scale": label, "Column": col,
                         "r_effect": r, "p_raw": pv})
    if not rows:
        print("  Surrogate-normalised comparison skipped: SURR_* columns absent "
              "(run the pipeline with run_surrogates=True).")
        return None
    out = pd.DataFrame(rows)
    out["p_adj"] = benjamini_hochberg(out["p_raw"].fillna(1.0).values)
    path = os.path.join(output_dir, "Surrogate_Normalized_Entropy.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Surrogate-normalised entropy → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<12} {rw['Scale']:<13} "
              f"r={rw['r_effect']:+.3f}  p_adj={rw['p_adj']:.4g}")
    return out


def beta_dfa_consistency(df_h, df_p, output_dir):
    """Check beta = 2*alpha - 1 (exact for stationary fractional Gaussian noise) per subject.

    A group-dependent departure means that group's spectra are less well described by a single power law.
    """
    if "PSD_Beta" not in df_h.columns or "DFA_Alpha2" not in df_h.columns:
        print("  beta/alpha consistency skipped: columns absent.")
        return None
    # The identity is checked against every available DFA exponent, and disagreement between them is reported.
    rows = []
    _alpha_cols = [c for c in ("DFA_Alpha1", "DFA_Alpha2", "DFA_Alpha")
                   if c in df_h.columns and c in df_p.columns]
    for label, df in (("Healthy", df_h), ("Psychiatric", df_p)):
        for _acol in _alpha_cols:
            _bc = ("PSD_Beta_NoMayer" if "PSD_Beta_NoMayer" in df.columns
                   else "PSD_Beta")
            _b = pd.to_numeric(df[_bc], errors="coerce")
            _a = pd.to_numeric(df[_acol], errors="coerce")
            _pred = 2 * _a - 1
            _ok = _b.notna() & _pred.notna()
            if _ok.sum() < 3:
                continue
            _res = (_b - _pred)[_ok]
            rows.append({"Group": f"{label}_vs_{_acol}", "N": int(_ok.sum()),
                         "Mean_Beta": float(_b[_ok].mean()),
                         "Mean_2Alpha_minus_1": float(_pred[_ok].mean()),
                         "Mean_Residual": float(_res.mean()),
                         "SD_Residual": float(_res.std(ddof=1)),
                         "Alpha_Column": _acol,
                         "Comparison_Type": "identity_by_exponent"})

    for label, df in (("Healthy", df_h), ("Psychiatric", df_p)):
        # The baroreflex-excised slope.
        bcol = ("PSD_Beta_NoMayer" if "PSD_Beta_NoMayer" in df.columns
                else "PSD_Beta")
        b = pd.to_numeric(df[bcol], errors="coerce")
        a = pd.to_numeric(df["DFA_Alpha2"], errors="coerce")
        pred = 2 * a - 1
        ok = b.notna() & pred.notna()
        resid = (b - pred)[ok]
        rows.append({"Group": label, "N": int(ok.sum()),
                     "Mean_Beta": float(b[ok].mean()),
                     "Mean_2Alpha2_minus_1": float(pred[ok].mean()),
                     "Mean_Residual": float(resid.mean()),
                     "SD_Residual": float(resid.std(ddof=1)),
                     "Spearman_r": float(spearmanr(b[ok], pred[ok])[0])})
    out = pd.DataFrame(rows)
    # Do the groups differ in their distance from the identity?
    _bc = "PSD_Beta_NoMayer" if "PSD_Beta_NoMayer" in df_h.columns else "PSD_Beta"
    rh = (pd.to_numeric(df_h[_bc], errors="coerce")
          - (2 * pd.to_numeric(df_h["DFA_Alpha2"], errors="coerce") - 1)).dropna()
    rp = (pd.to_numeric(df_p[_bc], errors="coerce")
          - (2 * pd.to_numeric(df_p["DFA_Alpha2"], errors="coerce") - 1)).dropna()
    r, pv, _, _ = _mw(rh, rp)
    out.loc[len(out)] = {"Group": "Residual group test", "N": len(rh) + len(rp),
                         "Mean_Beta": np.nan, "Mean_2Alpha2_minus_1": np.nan,
                         "Mean_Residual": r, "SD_Residual": pv,
                         "Spearman_r": np.nan}
    path = os.path.join(output_dir, "Beta_DFA_Consistency.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  beta = 2*alpha2 - 1 consistency → {path}")
    for _, rw in out.iterrows():
        if rw["Group"] == "Residual group test":
            print(f"    Residual difference between groups: r={rw['Mean_Residual']:+.3f}, "
                  f"p={rw['SD_Residual']:.4g}")
        else:
            print(f"    {rw['Group']:<12} beta={rw['Mean_Beta']:.3f}  "
                  f"2a2-1={rw['Mean_2Alpha2_minus_1']:.3f}  "
                  f"residual={rw['Mean_Residual']:+.3f}")
    return out


def nonredundant_family_comparison(df_h, df_p, output_dir):
    """Correction rerun with the definitionally collinear measures removed."""
    metrics = [m for m in NONREDUNDANT_PRIMARY
               if m in df_h.columns and m in df_p.columns]
    if len(metrics) < 3:
        print("  Non-redundant FDR skipped: too few metrics available.")
        return None
    stats = run_group_comparison(df_h, df_p, metrics,
                                 family_label="primary_nonredundant")
    path = os.path.join(output_dir, "Group_Comparison_NonRedundant.csv")
    out = stats.copy()
    out.index = [METRIC_LABELS.get(m, m) for m in out.index]
    out.to_csv(path, float_format="%.6f")
    print(f"  Non-redundant FDR family ({len(metrics)} metrics) → {path}")
    print(f"    Significant: {int(stats['Significant'].sum())}/{len(stats)} "
          f"(dropped: {', '.join(COLLINEAR_DUPLICATES)})")
    return stats


def length_and_respiration_checks(df_h, df_p, output_dir):
    """Series length and respiratory checks.

    Series length: patients have higher heart rates and so contribute more beats in the same 1800 s
    window; length affects DFA at long scales, entropy, MFDFA and recurrence. Tested directly and
    partialled out of the key measures.
    Respiration: a respiratory rate below 0.15 Hz moves RSA into the LF band; detection rate and, among
    detected subjects, frequency are compared descriptively.
    """
    rows = []
    for col in ("Num_RR", "Series_N", "Resp_Peak_Freq"):
        if col in df_h.columns and col in df_p.columns:
            r, pv, _, _ = _mw(df_h[col], df_p[col])
            rows.append({"Variable": col,
                         "Healthy_Mean": pd.to_numeric(df_h[col], errors="coerce").mean(),
                         "Psych_Mean": pd.to_numeric(df_p[col], errors="coerce").mean(),
                         "r_effect": r, "p_raw": pv, "Role": "group difference"})

    # Respiratory peak detection rate is compared between groups. Frequency is not used as a covariate,
    # because a peak was resolvable in a minority of subjects; it stays descriptive.
    for col in ("Resp_Peak_Found",):
        if col in df_h.columns and col in df_p.columns:
            fh = df_h[col].astype(str).str.lower().eq("true")
            fp = df_p[col].astype(str).str.lower().eq("true")
            try:
                from scipy.stats import fisher_exact
                _, pv = fisher_exact([[int(fh.sum()), int((~fh).sum())],
                                      [int(fp.sum()), int((~fp).sum())]])
            except Exception:
                pv = np.nan
            rows.append({"Variable": "Resp_Peak_Found (detection rate)",
                         "Healthy_Mean": float(fh.mean()),
                         "Psych_Mean": float(fp.mean()),
                         "r_effect": np.nan, "p_raw": pv,
                         "Role": "detection rate (Fisher exact)"})
            print(f"    Respiratory peak resolvable: "
                  f"{int(fh.sum())}/{len(fh)} controls vs "
                  f"{int(fp.sum())}/{len(fp)} patients (p={pv:.3f})")

    # Frequency among detected subjects only (descriptive, not an adjustment).
    if "Resp_Peak_Freq" in df_h.columns and "Resp_Peak_Freq" in df_p.columns:
        hh = pd.to_numeric(df_h["Resp_Peak_Freq"], errors="coerce").dropna()
        pp = pd.to_numeric(df_p["Resp_Peak_Freq"], errors="coerce").dropna()
        if len(hh) >= 5 and len(pp) >= 5:
            r, pv, _, _ = _mw(hh, pp)
            rows.append({"Variable": "Resp_Peak_Freq (detected subjects only)",
                         "Healthy_Mean": float(hh.median()),
                         "Psych_Mean": float(pp.median()),
                         "r_effect": r, "p_raw": pv,
                         "Role": "DESCRIPTIVE: selected subset, not a covariate"})
        else:
            print("    Respiratory frequency not compared: too few subjects "
                  "with a resolvable peak in at least one group.")

    # -- Joint covariate model --
    # Five covariates: age, sex, mobility, mean heart rate and HF power. Beat count is not included, since
    # over the fixed 1800 s window it is 30 x mean heart rate (r = 0.999).
    joint = ["Age", "Sex", "mobility_mg", "HRV_MeanHR", "HF_Power_ms2"]
    have = [c for c in joint if c in df_h.columns and c in df_p.columns]
    _identity_cluster = {"Num_RR", "Series_N", "HRV_MeanHR", "HRV_MeanNN"}
    _clash = sorted(set(have) & _identity_cluster)
    if len(_clash) > 1:
        raise RuntimeError(
            f"joint model adjusts for {_clash}, which are the same variable "
            f"over a fixed-duration window (beats = 30 x heart rate, mean NN is "
            f"its reciprocal). Adjusting for more than one of them is adjusting "
            f"twice for one quantity and reporting it as independent controls.")
    if len(have) >= 4:
        # The VLF/LF ratio as well as the slope.
        for tgt in ("PSD_Beta_NoMayer", "PSD_Beta", "PSD_Beta_low",
                    "VLF_LF_Ratio"):
            if tgt not in df_h.columns or tgt not in df_p.columns:
                continue
            vals = np.concatenate([pd.to_numeric(df_h[tgt], errors="coerce").values,
                                   pd.to_numeric(df_p[tgt], errors="coerce").values])
            grp = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])
            covs = []
            for c in have:
                if c == "Sex":
                    covs.append(np.concatenate([(df_h[c] == "F").astype(float).values,
                                                (df_p[c] == "F").astype(float).values]))
                else:
                    covs.append(np.concatenate([
                        pd.to_numeric(df_h[c], errors="coerce").values,
                        pd.to_numeric(df_p[c], errors="coerce").values]))
            r_j, p_j = partial_spearman(vals, grp, covs)
            rows.append({"Variable": f"{tgt} ~ Group | " + " + ".join(have),
                         "Healthy_Mean": np.nan, "Psych_Mean": np.nan,
                         "r_effect": r_j, "p_raw": p_j,
                         "Role": f"JOINT {len(have)}-covariate adjustment"})
            print(f"    JOINT model {tgt}: r = {r_j:+.3f}, p = {p_j:.4g} "
                  f"({len(have)} covariates)")
    else:
        print(f"    Joint covariate model skipped: only {len(have)} of 6 "
              f"covariates present ({have}). Merge accelerometer and metadata "
              f"columns into the results CSVs first.")

    # Every length-sensitive measure (entropy, recurrence and network measures, MSE) is residualized on
    # beat count here.
    targets = ["PSD_Beta", "PSD_Beta_NoMayer", "PSD_Beta_low", "DFA_Alpha2",
               "SampEn", "ApEn", "PermEn", "MSE_Mean",
               "RQA_DET", "RQA_LAM",
               "RNA_ClusteringCoeff", "RNA_Transitivity"]
    for cov in ("Num_RR",):
        if cov not in df_h.columns or cov not in df_p.columns:
            continue
        for t in targets:
            if t not in df_h.columns or t not in df_p.columns:
                continue
            vals = np.concatenate([pd.to_numeric(df_h[t], errors="coerce").values,
                                   pd.to_numeric(df_p[t], errors="coerce").values])
            grp = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])
            cv = np.concatenate([pd.to_numeric(df_h[cov], errors="coerce").values,
                                 pd.to_numeric(df_p[cov], errors="coerce").values])
            r, pv = partial_spearman(vals, grp, [cv])
            rows.append({"Variable": f"{t} ~ Group | {cov}",
                         "Healthy_Mean": np.nan, "Psych_Mean": np.nan,
                         "r_effect": r, "p_raw": pv, "Role": "partial correlation"})
    if not rows:
        print("  Length/respiration checks skipped: columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Length_Respiration_Checks.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Length + respiration checks → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Variable']:<34} r={rw['r_effect']:+.3f}  p={rw['p_raw']:.4g}")
    return out


def pnn50_floor_report(df_h, df_p, output_dir):
    """pNN50 floor report: the proportion of subjects at or near zero, since pNN50 is zero-inflated in
    patients and a standardized mean difference is not meaningful for it.
    """
    if "HRV_pNN50" not in df_h.columns:
        print("  pNN50 report skipped: column absent.")
        return None
    rows = []
    for label, df in (("Healthy", df_h), ("Psychiatric", df_p)):
        v = pd.to_numeric(df["HRV_pNN50"], errors="coerce").dropna()
        rows.append({"Group": label, "N": len(v), "Mean": v.mean(),
                     "SD": v.std(ddof=1), "Median": v.median(),
                     "N_At_Zero": int((v <= 0).sum()),
                     "Pct_At_Zero": 100.0 * (v <= 0).mean(),
                     "N_Below_1pct": int((v < 1.0).sum())})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "pNN50_Floor_Report.csv")
    out.to_csv(path, index=False, float_format="%.3f")
    print(f"  pNN50 floor report → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Group']:<12} mean={rw['Mean']:.2f} SD={rw['SD']:.2f}  "
              f"at floor: {rw['N_At_Zero']}/{rw['N']} ({rw['Pct_At_Zero']:.0f}%)")
    print("    -> report rank statistics only; exclude pNN50 from the "
          "parametric (Cohen's d) table.")
    return out


def schizophrenia_subgroup_analysis(df_h, df_p, output_dir):
    """Superseded by diagnosis_subgroup_analysis(); kept for direct calls. Reruns the primary comparisons on
    the schizophrenia patients only, using SCHIZOPHRENIA_RECORD_IDS or a Diagnosis column; skipped with a
    message if neither exists.
    """
    ids = list(SCHIZOPHRENIA_RECORD_IDS)
    sub = None
    if ids:
        idx = df_p.index.astype(str)
        sub = df_p[idx.isin([str(i) for i in ids])]
    elif "Diagnosis" in df_p.columns:
        d = df_p["Diagnosis"].astype(str).str.lower()
        sub = df_p[d.str.contains("schizo")]
    if sub is None or len(sub) < 5:
        print("  Schizophrenia subgroup analysis SKIPPED: no subgroup labels "
              "available. Populate SCHIZOPHRENIA_RECORD_IDS in comparison.py "
              "or add a 'Diagnosis' column to the psychiatric results CSV. "
              "This analysis was explicitly requested by review and its "
              "absence should be stated in the manuscript, not passed over.")
        return None

    targets = ["PSD_Beta", "PSD_Beta_NoMayer", "DFA_Alpha2",
               "HRV_RMSSD", "HRV_MeanHR", "SampEn", "RQA_DET"]
    rows = []
    for t in targets:
        if t not in df_h.columns or t not in df_p.columns:
            continue
        r_all, p_all, _, _ = _mw(df_h[t], df_p[t])
        r_sub, p_sub, _, n_sub = _mw(df_h[t], sub[t])
        rows.append({"Metric": t, "N_Subgroup": n_sub,
                     "r_pooled": r_all, "p_pooled": p_all,
                     "r_schizophrenia_only": r_sub, "p_schizophrenia_only": p_sub,
                     "r_change": r_sub - r_all})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Schizophrenia_Subgroup.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Schizophrenia-only sensitivity → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<20} pooled r={rw['r_pooled']:+.3f}  "
              f"SZ-only r={rw['r_schizophrenia_only']:+.3f}  "
              f"(change {rw['r_change']:+.3f})")
    return out

# ---- Subject-level metadata analyses ----
# From the released age, sex, ICD codes and quetiapine status (HRV-ACC version 2). The medication field
# records quetiapine only, so a quetiapine contrast compares quetiapine with other agents, not medicated
# with unmedicated. Quetiapine is concentrated in the bipolar subgroup (5 of 7 against 2 of 23).

METADATA_COLUMNS = ["Age", "Sex", "Diagnosis", "ICD_Primary", "ICD_All",
                    "Comorbid_Codes", "Comorbid_Substance", "Quetiapine"]

# Measures carried through every metadata analysis: those that survived the control battery, plus
# two that did not, for comparison.
METADATA_TARGETS = ["PSD_Beta_NoMayer", "PSD_Beta", "DFA_Alpha2", "DFA_Alpha",
                    "HRV_RMSSD", "HRV_MeanHR", "LF_Power_ms2", "HF_Power_ms2",
                    "MFDFA_Width", "MSE_Mean", "SampEn", "RQA_DET",
                    # VLF measures included; no table here corrects across rows, so adding rows leaves other values unchanged.
                    "VLF_LF_Ratio", "VLF_Power_ms2"]


def _has_metadata(df_p):
    """Patient-specific metadata (diagnosis and medication) present."""
    return all(c in df_p.columns for c in ("Diagnosis", "Age", "Quetiapine"))


def _has_demographics(df):
    """Age and sex present."""
    return all(c in df.columns for c in ("Age", "Sex"))


def _as_bool(series):
    """Booleans survive a CSV round trip as the strings 'True'/'False'."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(
        ("true", "yes", "1"))


def metadata_cohort_description(df_p, output_dir):
    """Cohort description table."""
    if not _has_metadata(df_p):
        print("  Cohort description skipped: metadata columns absent.")
        return None
    rows = []
    quet = _as_bool(df_p["Quetiapine"])
    sub = _as_bool(df_p["Comorbid_Substance"])
    age = pd.to_numeric(df_p["Age"], errors="coerce")
    for label, mask in (("All patients", pd.Series(True, index=df_p.index)),
                        ("Schizophrenia (F20)", df_p["Diagnosis"] == "schizophrenia"),
                        ("Bipolar (F31)", df_p["Diagnosis"] == "bipolar")):
        if mask.sum() == 0:
            continue
        rows.append({
            "Group": label, "N": int(mask.sum()),
            "Age_mean": age[mask].mean(), "Age_SD": age[mask].std(ddof=1),
            "Age_min": age[mask].min(), "Age_max": age[mask].max(),
            "N_female": int((df_p.loc[mask, "Sex"] == "F").sum()),
            "N_male": int((df_p.loc[mask, "Sex"] == "M").sum()),
            "N_quetiapine": int(quet[mask].sum()),
            "N_comorbid_substance": int(sub[mask].sum()),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Patient_Cohort_Description.csv")
    out.to_csv(path, index=False, float_format="%.2f")
    print(f"  Cohort description → {path}")
    print(out.round(1).to_string(index=False))
    return out


def diagnosis_subgroup_analysis(df_h, df_p, output_dir):
    """Each diagnostic subgroup against the full control cohort, and schizophrenia against bipolar.

    The bipolar subgroup (N = 7) is reported for completeness and is severely underpowered.
    """
    if not _has_metadata(df_p):
        print("  Diagnosis subgroup analysis skipped: metadata absent.")
        return None
    sz = df_p[df_p["Diagnosis"] == "schizophrenia"]
    bp = df_p[df_p["Diagnosis"] == "bipolar"]
    print(f"  Subgroups: schizophrenia N={len(sz)}, bipolar N={len(bp)}")

    rows = []
    for m in METADATA_TARGETS:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        r_all, p_all, _, _ = _mw(df_h[m], df_p[m])
        rec = {"Metric": m, "r_pooled": r_all, "p_pooled": p_all,
               "N_pooled": len(df_p)}
        if len(sz) >= 5:
            r_s, p_s, _, n_s = _mw(df_h[m], sz[m])
            rec.update(r_schizophrenia=r_s, p_schizophrenia=p_s, N_sz=n_s,
                       r_change_sz=r_s - r_all)
        if len(bp) >= 5:
            r_b, p_b, _, n_b = _mw(df_h[m], bp[m])
            rec.update(r_bipolar=r_b, p_bipolar=p_b, N_bp=n_b,
                       r_change_bp=r_b - r_all)
        if len(sz) >= 5 and len(bp) >= 5:
            r_sb, p_sb, _, _ = _mw(sz[m], bp[m])
            rec.update(r_sz_vs_bp=r_sb, p_sz_vs_bp=p_sb)
        # Detection floors are written on the same row as the estimates they qualify.
        try:
            if len(bp) >= 5:
                _, _r_bp_floor, _ = minimum_detectable_effect(len(df_h),
                                                              len(bp))
                rec["min_detectable_r_bp_vs_control"] = round(_r_bp_floor, 3)
            if len(sz) >= 5 and len(bp) >= 5:
                _, _r_sb_floor, _ = minimum_detectable_effect(len(sz), len(bp))
                rec["min_detectable_r_sz_vs_bp"] = round(_r_sb_floor, 3)
            if len(sz) >= 5:
                _, _r_sz_floor, _ = minimum_detectable_effect(len(df_h),
                                                              len(sz))
                rec["min_detectable_r_sz_vs_control"] = round(_r_sz_floor, 3)
        except Exception:
            pass
        rows.append(rec)

    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Diagnosis_Subgroup.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Diagnosis subgroups → {path}")
    print(f"  {'metric':<20}{'pooled r':>10}{'SZ-only r':>11}{'SZ p':>10}"
          f"{'BP-only r':>11}{'SZ vs BP p':>12}")
    for _, rw in out.iterrows():
        print(f"  {rw['Metric']:<20}{rw['r_pooled']:10.3f}"
              f"{rw.get('r_schizophrenia', np.nan):11.3f}"
              f"{rw.get('p_schizophrenia', np.nan):10.4f}"
              f"{rw.get('r_bipolar', np.nan):11.3f}"
              f"{rw.get('p_sz_vs_bp', np.nan):12.4f}")
    print("  NOTE: the bipolar subgroup (N=7) is severely underpowered. A null "
          "there is uninformative, not evidence of absence.")
    return out


def quetiapine_analysis(df_h, df_p, output_dir):
    """Whether quetiapine exposure accounts for the group differences.

    Reports (a) quetiapine against non-quetiapine patients; (b) each subset against the full control
    cohort; (c) the group effect with quetiapine as a covariate. Patients marked "no" received other
    agents, so this does not test medication as a whole.
    """
    if not _has_metadata(df_p):
        print("  Quetiapine analysis skipped: metadata absent.")
        return None
    quet = _as_bool(df_p["Quetiapine"])
    on, off = df_p[quet.values], df_p[~quet.values]
    print(f"  Quetiapine: {len(on)} exposed, {len(off)} not exposed")
    if len(on) < 5 or len(off) < 5:
        print("  Too few in one arm for a meaningful contrast.")
        return None

    rows = []
    for m in METADATA_TARGETS:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        r_w, p_w, _, _ = _mw(on[m], off[m])              # within patients
        r_on, p_on, _, _ = _mw(df_h[m], on[m])           # exposed vs controls
        r_off, p_off, _, _ = _mw(df_h[m], off[m])        # unexposed vs controls
        # Group effect adjusted for quetiapine (controls coded 0, not exposed).
        vals = np.concatenate([pd.to_numeric(df_h[m], errors="coerce").values,
                               pd.to_numeric(df_p[m], errors="coerce").values])
        grp = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])
        cov = np.concatenate([np.zeros(len(df_h)), quet.values.astype(float)])
        r_adj, p_adj_ = partial_spearman(vals, grp, [cov])
        rows.append({"Metric": m,
                     "r_quet_vs_nonquet": r_w, "p_quet_vs_nonquet": p_w,
                     "r_quetiapine_vs_control": r_on, "p_quetiapine_vs_control": p_on,
                     "r_nonquet_vs_control": r_off, "p_nonquet_vs_control": p_off,
                     "r_group_adj_quetiapine": r_adj, "p_group_adj_quetiapine": p_adj_})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Quetiapine_Analysis.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Quetiapine analysis → {path}")
    print(f"  {'metric':<20}{'within-pt r':>13}{'within p':>10}"
          f"{'quet vs ctrl':>14}{'non-q vs ctrl':>15}{'adj r':>9}")
    for _, rw in out.iterrows():
        print(f"  {rw['Metric']:<20}{rw['r_quet_vs_nonquet']:13.3f}"
              f"{rw['p_quet_vs_nonquet']:10.4f}"
              f"{rw['r_quetiapine_vs_control']:14.3f}"
              f"{rw['r_nonquet_vs_control']:15.3f}"
              f"{rw['r_group_adj_quetiapine']:9.3f}")
    print("  REMINDER: 'no quetiapine' does NOT mean unmedicated. This cannot "
          "resolve the medication confound; it tests one agent only.")
    return out


def comorbidity_analysis(df_h, df_p, output_dir):
    """Sensitivity analysis excluding patients with a comorbid substance-use code (ICD F1x)."""
    if "Comorbid_Substance" not in df_p.columns:
        print("  Comorbidity analysis skipped: metadata absent.")
        return None
    sub = _as_bool(df_p["Comorbid_Substance"])
    clean = df_p[~sub.values]
    print(f"  Comorbid substance use: {int(sub.sum())}/{len(df_p)}; "
          f"{len(clean)} retained for the clean-subset contrast")
    if len(clean) < 10:
        print("  Too few patients remain for a meaningful contrast.")
        return None
    rows = []
    for m in METADATA_TARGETS:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        r_all, p_all, _, _ = _mw(df_h[m], df_p[m])
        r_cl, p_cl, _, n_cl = _mw(df_h[m], clean[m])
        r_w, p_w = (_mw(df_p[sub.values][m], clean[m])[:2]
                    if sub.sum() >= 5 else (np.nan, np.nan))
        rows.append({"Metric": m, "r_all_patients": r_all, "p_all_patients": p_all,
                     "r_no_comorbidity": r_cl, "p_no_comorbidity": p_cl,
                     "N_no_comorbidity": n_cl,
                     "r_comorbid_vs_clean": r_w, "p_comorbid_vs_clean": p_w})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Comorbidity_Sensitivity.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Comorbidity sensitivity → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<20} all r={rw['r_all_patients']:+.3f}  "
              f"no-comorbidity r={rw['r_no_comorbidity']:+.3f} "
              f"(p={rw['p_no_comorbidity']:.4f}, N={int(rw['N_no_comorbidity'])})")
    return out


def age_sex_analysis(df_h, df_p, output_dir):
    """Age and sex associations within the patient group."""
    if not _has_metadata(df_p):
        print("  Age/sex analysis skipped: metadata absent.")
        return None
    age = pd.to_numeric(df_p["Age"], errors="coerce")
    sex = df_p["Sex"].astype(str)
    rows = []
    for m in METADATA_TARGETS:
        if m not in df_p.columns:
            continue
        v = pd.to_numeric(df_p[m], errors="coerce")
        ok = v.notna() & age.notna()
        r_age, p_age = spearmanr(v[ok], age[ok]) if ok.sum() >= 6 else (np.nan, np.nan)
        f = v[(sex == "F") & v.notna()]
        mm = v[(sex == "M") & v.notna()]
        r_sex, p_sex = (_mw(f, mm)[:2] if (len(f) >= 5 and len(mm) >= 5)
                        else (np.nan, np.nan))
        rows.append({"Metric": m, "r_age_within_patients": r_age, "p_age": p_age,
                     "r_sex_within_patients": r_sex, "p_sex": p_sex,
                     "N_female": int(len(f)), "N_male": int(len(mm))})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Age_Sex_WithinPatients.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Age/sex within patients → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<20} age r={rw['r_age_within_patients']:+.3f} "
              f"(p={rw['p_age']:.3f})   sex r={rw['r_sex_within_patients']:+.3f} "
              f"(p={rw['p_sex']:.3f})")
    try:
        from patient_metadata import control_metadata_available
        if not control_metadata_available():
            print("  NOTE: control demographics not released. The groups cannot "
                  "be age- or sex-matched, and this must be stated as a "
                  "limitation rather than treated as resolved.")
    except Exception:
        pass
    return out


def demographic_matching_report(df_h, df_p, output_dir):
    """Test whether the groups are matched on age and sex."""
    if "Age" not in df_h.columns or "Age" not in df_p.columns:
        print("  Demographic matching skipped: control metadata absent.")
        return None
    ah = pd.to_numeric(df_h["Age"], errors="coerce").dropna()
    ap = pd.to_numeric(df_p["Age"], errors="coerce").dropna()
    if len(ah) < 5 or len(ap) < 5:
        print("  Demographic matching skipped: too few subjects with age.")
        return None

    r_age, p_age, _, _ = _mw(ah, ap)
    tab = [[int((df_h["Sex"] == "F").sum()), int((df_h["Sex"] == "M").sum())],
           [int((df_p["Sex"] == "F").sum()), int((df_p["Sex"] == "M").sum())]]
    try:
        from scipy.stats import fisher_exact
        _, p_sex = fisher_exact(tab)
    except Exception:
        p_sex = np.nan

    rows = [{"Variable": "Age", "Control": ah.mean(), "Patient": ap.mean(),
             "Control_SD": ah.std(ddof=1), "Patient_SD": ap.std(ddof=1),
             "r_effect": r_age, "p_value": p_age},
            {"Variable": "Sex (N female)", "Control": tab[0][0],
             "Patient": tab[1][0], "Control_SD": np.nan, "Patient_SD": np.nan,
             "r_effect": np.nan, "p_value": p_sex}]
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Demographic_Matching.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Demographic matching → {path}")
    print(f"    Age: control {ah.mean():.1f} +/- {ah.std(ddof=1):.1f} "
          f"(range {ah.min():.0f}-{ah.max():.0f}) vs patient "
          f"{ap.mean():.1f} +/- {ap.std(ddof=1):.1f} "
          f"(range {ap.min():.0f}-{ap.max():.0f}); r = {r_age:+.3f}, "
          f"p = {p_age:.4f}")
    print(f"    Sex: control {tab[0][0]}F/{tab[0][1]}M vs patient "
          f"{tab[1][0]}F/{tab[1][1]}M; Fisher p = {p_sex:.4f}")
    if p_age > 0.05 and (not np.isfinite(p_sex) or p_sex > 0.05):
        print("    -> groups are MATCHED on age and sex. Report this as a "
              "design strength; age adjustment below is confirmatory.")
    else:
        print("    -> groups are NOT matched. The age- and sex-adjusted "
              "results below are load-bearing, not confirmatory.")
    return out


def age_sex_adjusted_comparison(df_h, df_p, output_dir):
    """Group comparisons with age and sex as between-group covariates (partial Spearman throughout)."""
    if "Age" not in df_h.columns or "Age" not in df_p.columns:
        print("  Age/sex-adjusted comparison skipped: control metadata absent.")
        return None

    age = np.concatenate([pd.to_numeric(df_h["Age"], errors="coerce").values,
                          pd.to_numeric(df_p["Age"], errors="coerce").values])
    sex = np.concatenate([(df_h["Sex"] == "F").astype(float).values,
                          (df_p["Sex"] == "F").astype(float).values])
    grp = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])

    rows = []
    for m in METADATA_TARGETS:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        vals = np.concatenate([pd.to_numeric(df_h[m], errors="coerce").values,
                               pd.to_numeric(df_p[m], errors="coerce").values])
        r_raw, p_raw, _, _ = _mw(df_h[m], df_p[m])
        r_age, p_a = partial_spearman(vals, grp, [age])
        r_sex, p_s = partial_spearman(vals, grp, [sex])
        r_both, p_b = partial_spearman(vals, grp, [age, sex])
        r_unadj_p, _ = spearmanr(vals[np.isfinite(vals)], grp[np.isfinite(vals)])
        rows.append({"Metric": m, "r_unadjusted": r_raw, "p_unadjusted": p_raw,
                     "r_adj_age": r_age, "p_adj_age": p_a,
                     "r_adj_sex": r_sex, "p_adj_sex": p_s,
                     "r_adj_age_sex": r_both, "p_adj_age_sex": p_b,
                     # Baseline: the unadjusted partial correlation (same statistic as the adjusted one).
                     "r_unadj_partial": r_unadj_p,
                     # Ratio of two estimates: descriptive only, with no interval; report the adjusted and unadjusted
                     # correlations instead.
                     "pct_retained_age_sex": (100.0 * r_both / r_unadj_p
                                              if r_unadj_p not in (0, np.nan)
                                              else np.nan),
                     "pct_retained_is_ratio_of_estimates": True})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "AgeSex_Adjusted_Comparison.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Age/sex-adjusted comparison → {path}")
    print(f"  {'metric':<20}{'raw r':>9}{'|age':>9}{'|sex':>9}"
          f"{'|age+sex':>11}{'p(adj)':>10}{'kept':>8}")
    for _, rw in out.iterrows():
        print(f"  {rw['Metric']:<20}{rw['r_unadjusted']:9.3f}"
              f"{rw['r_adj_age']:9.3f}{rw['r_adj_sex']:9.3f}"
              f"{rw['r_adj_age_sex']:11.3f}{rw['p_adj_age_sex']:10.4f}"
              f"{rw['pct_retained_age_sex']:7.0f}%")
    return out


def age_association_both_groups(df_h, df_p, output_dir):
    """Measure-age associations in each group separately and pooled."""
    if "Age" not in df_h.columns or "Age" not in df_p.columns:
        print("  Age association analysis skipped: control metadata absent.")
        return None
    rows = []
    for m in METADATA_TARGETS:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        rec = {"Metric": m}
        for lbl, d in (("control", df_h), ("patient", df_p)):
            v = pd.to_numeric(d[m], errors="coerce")
            a = pd.to_numeric(d["Age"], errors="coerce")
            ok = v.notna() & a.notna()
            if ok.sum() >= 6:
                r_, p_ = spearmanr(v[ok], a[ok])
                rec[f"r_age_{lbl}"] = r_
                rec[f"p_age_{lbl}"] = p_
        v = np.concatenate([pd.to_numeric(df_h[m], errors="coerce").values,
                            pd.to_numeric(df_p[m], errors="coerce").values])
        a = np.concatenate([pd.to_numeric(df_h["Age"], errors="coerce").values,
                            pd.to_numeric(df_p["Age"], errors="coerce").values])
        ok = np.isfinite(v) & np.isfinite(a)
        if ok.sum() >= 6:
            r_, p_ = spearmanr(v[ok], a[ok])
            rec["r_age_pooled"] = r_
            rec["p_age_pooled"] = p_
        rows.append(rec)
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Age_Association_BothGroups.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Age associations → {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<20} control r={rw.get('r_age_control', np.nan):+.3f} "
              f"(p={rw.get('p_age_control', np.nan):.3f})   "
              f"patient r={rw.get('r_age_patient', np.nan):+.3f} "
              f"(p={rw.get('p_age_patient', np.nan):.3f})")
    return out


def run_metadata_analyses(df_h, df_p, output_dir):
    """Run every subject-level metadata analysis in one call."""
    print("\n" + "=" * 65)
    print("  SUBJECT-LEVEL METADATA ANALYSES")
    print("=" * 65)
    if not _has_metadata(df_p) and not _has_demographics(df_h):
        print("  SKIPPED: neither cohort carries metadata columns.")
        print("  Re-run healthy_hrvacc.py and psych_hrvacc.py with "
              "control_data_summary.csv and patient_data_summary.csv present "
              "in the project root.")
        return {}
    if not _has_metadata(df_p):
        print("  NOTE: patient metadata absent; diagnosis, medication and "
              "comorbidity analyses will be skipped.")
    if not _has_demographics(df_h):
        print("  NOTE: control metadata absent; age- and sex-adjusted "
              "between-group analyses will be skipped.")
    return {
        "matching": demographic_matching_report(df_h, df_p, output_dir),
        "age_sex_adjusted": age_sex_adjusted_comparison(df_h, df_p, output_dir),
        "age_association": age_association_both_groups(df_h, df_p, output_dir),
        "cohort": metadata_cohort_description(df_p, output_dir),
        "diagnosis": diagnosis_subgroup_analysis(df_h, df_p, output_dir),
        "quetiapine": quetiapine_analysis(df_h, df_p, output_dir),
        "comorbidity": comorbidity_analysis(df_h, df_p, output_dir),
        "age_sex": age_sex_analysis(df_h, df_p, output_dir),
    }


def age_adjusted_cross_device(df_a, df_b, df_p, output_dir,
                              label_a="HRVACC_control", label_b="MITBIH_control"):
    """Cross-cohort comparisons with age as a covariate.

    The MIT-BIH cohort is younger than either HRV-ACC cohort (mean 34.3 against 43.2 and 41.4 years) and
    the slope correlates with age within the controls. Needs Age in all three cohorts; skipped with a
    message if MIT-BIH ages are unavailable.
    """
    if any("Age" not in d.columns for d in (df_a, df_b, df_p)):
        print("  Age-adjusted cross-device analysis SKIPPED: MIT-BIH ages are "
              "not in the results CSV. Add them from the PhysioNet header "
              "metadata to enable this analysis; the age confound in the "
              "cross-device comparison is otherwise unaddressed and must be "
              "stated as a limitation.")
        return None
    rows = []
    # VLF measures included.
    targets = ["PSD_Beta_NoMayer", "PSD_Beta", "PSD_Beta_low",
               "DFA_Alpha2", "HRV_RMSSD", "LF_Power_ms2",
               "VLF_Power_ms2", "VLF_LF_Ratio"]
    for m in targets:
        if any(m not in d.columns for d in (df_a, df_b, df_p)):
            continue
        for lbl, d1, d2 in ((f"{label_a} vs {label_b}", df_a, df_b),
                            (f"{label_b} vs patients", df_b, df_p)):
            v = np.concatenate([pd.to_numeric(d1[m], errors="coerce").values,
                                pd.to_numeric(d2[m], errors="coerce").values])
            g = np.concatenate([np.zeros(len(d1)), np.ones(len(d2))])
            a = np.concatenate([pd.to_numeric(d1["Age"], errors="coerce").values,
                                pd.to_numeric(d2["Age"], errors="coerce").values])
            r_rb, p_rb, _, _ = _mw(d1[m], d2[m])
            # Baseline: the unadjusted partial correlation on the same rows as the adjusted value.
            ok = np.isfinite(v) & np.isfinite(a)
            r_pu, p_pu = (spearmanr(v[ok], g[ok]) if ok.sum() >= 6
                          else (np.nan, np.nan))
            r_a, p_a = partial_spearman(v, g, [a])
            rows.append({"Metric": m, "Comparison": lbl,
                         "Sign": "positive = second-named cohort higher",
                         "N_First": int(np.isfinite(v[g == 0]).sum()),
                         "N_Second": int(np.isfinite(v[g == 1]).sum()),
                         "r_rank_biserial": r_rb, "p_rank_biserial": p_rb,
                         "r_partial_unadjusted": float(r_pu),
                         "p_partial_unadjusted": float(p_pu),
                         "r_partial_age_adjusted": r_a,
                         "p_partial_age_adjusted": p_a,
                         # Ratio of two estimates: descriptive only.
                         "pct_retained": (100.0 * r_a / r_pu
                                          if np.isfinite(r_pu) and r_pu != 0
                                          else np.nan)})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "CrossDevice_AgeAdjusted.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Age-adjusted cross-device -> {path}")
    for _, r in out.iterrows():
        print(f"    {r['Metric']:<20}{r['Comparison']:<32} "
              f"partial {r['r_partial_unadjusted']:+.3f} -> age-adj "
              f"{r['r_partial_age_adjusted']:+.3f} "
              f"(p {r['p_partial_age_adjusted']:.4f})   "
              f"[rank-biserial {r['r_rank_biserial']:+.3f}]")
    return out


def deterministic_checks(df_h, df_p, output_dir):
    """Report the quantities fixed by construction (RQA_REC, RNA_MeanDegree, both equal to the target
    recurrence rate) as checks of the thresholding and normalization, outside every correction family.
    """
    rows = []
    for m in DETERMINISTIC_CHECKS + ["RNA_MeanDegree"]:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        h = pd.to_numeric(df_h[m], errors="coerce").dropna()
        p = pd.to_numeric(df_p[m], errors="coerce").dropna()
        rows.append({"Quantity": m, "Target": RQA_TARGET_RATE,
                     "Control_mean": h.mean(), "Control_SD": h.std(ddof=1),
                     "Patient_mean": p.mean(), "Patient_SD": p.std(ddof=1),
                     "Max_abs_deviation": float(max((h - RQA_TARGET_RATE).abs().max(),
                                                    (p - RQA_TARGET_RATE).abs().max())),
                     "Passes": bool(max((h - RQA_TARGET_RATE).abs().max(),
                                        (p - RQA_TARGET_RATE).abs().max()) < 5e-3)})
    if not rows:
        print("  Deterministic checks skipped: columns absent.")
        return None
    out = pd.DataFrame(rows).drop_duplicates(subset=["Quantity"])
    path = os.path.join(output_dir, "Deterministic_Checks.csv")
    out.to_csv(path, index=False, float_format="%.8f")
    print(f"  Deterministic checks -> {path}")
    for _, r in out.iterrows():
        print(f"    {r['Quantity']:18s} ctrl {r['Control_mean']:.6f} +/- {r['Control_SD']:.2e}  "
              f"pt {r['Patient_mean']:.6f}  max dev {r['Max_abs_deviation']:.2e}  "
              f"{'PASS' if r['Passes'] else 'FAIL'}")
    print("    These are normalisation checks and are excluded from all FDR families.")
    return out


def beta_band_sweep_comparison(df_h, df_p, output_dir):
    """Slope group comparison across every fitted band.

    The low band (nominal 0.01-0.07 Hz) lies below both the baroreflex resonance and any respiratory
    contribution. In simulation, respiration at 0.150 Hz biases a full-band fit by -0.27 while the low-band
    fit is unbiased for respiration from 0.150 to 0.400 Hz.
    """
    tags = ["low", "verylow", "lowmid", "mayerexc", "nomayerx", "wideband"]
    rows = []
    for tag in tags:
        col = f"PSD_Beta_{tag}"
        if col not in df_h.columns or col not in df_p.columns:
            continue
        r, pv, nh, npat = _mw(df_h[col], df_p[col])
        hn = pd.to_numeric(df_h[col], errors="coerce")
        pn = pd.to_numeric(df_p[col], errors="coerce")
        nb = _num_col(df_h, f"{col}_NBins")
        lo = _num_col(df_h, f"{col}_LowHz")
        hi = _num_col(df_h, f"{col}_HighHz")
        rows.append({"Band": tag,
                     "Actual_low_Hz": lo.median() if lo is not None else np.nan,
                     "Actual_high_Hz": hi.median() if hi is not None else np.nan,
                     "N_bins": nb.median() if nb is not None else np.nan,
                     "Control_mean": hn.mean(), "Control_SD": hn.std(ddof=1),
                     "Patient_mean": pn.mean(), "Patient_SD": pn.std(ddof=1),
                     "r_effect": r, "p_raw": pv,
                     "N_Healthy": nh, "N_Psych": npat})
    if not rows:
        print("  Band sweep skipped: PSD_Beta_<tag> columns absent. Re-run pipeline.py.")
        return None
    # Bootstrap CIs on r. With five to seven bins the narrow-band estimates are noisy, so those rows show
    # direction and significance rather than magnitude.
    rng = np.random.default_rng(20260817)
    for k, row in enumerate(rows):
        col = f"PSD_Beta_{row['Band']}"
        if col not in df_h.columns or col not in df_p.columns:
            continue
        a = pd.to_numeric(df_h[col], errors="coerce").dropna().values
        b = pd.to_numeric(df_p[col], errors="coerce").dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        boot = []
        for _ in range(2000):
            ra = rng.choice(a, len(a), replace=True)
            rb = rng.choice(b, len(b), replace=True)
            try:
                u = mannwhitneyu(ra, rb, alternative="two-sided")[0]
                boot.append(1 - 2 * u / (len(ra) * len(rb)))
            except Exception:
                continue
        if len(boot) >= 500:
            rows[k]["r_CI_low"] = float(np.percentile(boot, 2.5))
            rows[k]["r_CI_high"] = float(np.percentile(boot, 97.5))

    out = pd.DataFrame(rows)
    out["p_adj"] = benjamini_hochberg(out["p_raw"].fillna(1.0).values)
    path = os.path.join(output_dir, "Beta_Band_Sweep.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Beta band sweep -> {path}")
    print(f"  {'band':<10}{'actual Hz':>18}{'bins':>6}{'ctrl':>8}{'pt':>8}"
          f"{'r':>8}{'p_raw':>11}  95% CI")
    for _, r in out.iterrows():
        ci = (f"[{r.get('r_CI_low', float('nan')):+.3f}, "
              f"{r.get('r_CI_high', float('nan')):+.3f}]")
        print(f"  {r['Band']:<10}{r['Actual_low_Hz']:>8.4f}-{r['Actual_high_Hz']:<9.4f}"
              f"{r['N_bins']:>6.0f}{r['Control_mean']:>8.3f}{r['Patient_mean']:>8.3f}"
              f"{r['r_effect']:>8.3f}{r['p_raw']:>11.3g}  {ci}")
    try:
        lo_r = out.loc[out.Band == "low", "r_effect"].values[0]
        mx_r = out.loc[out.Band == "mayerexc", "r_effect"].values[0]
        print(f"\n  --> the confound-free 'low' band retains "
              f"{100*lo_r/mx_r:.0f}% of the legacy primary effect size.")
        print("      If this is large and significant, neither the baroreflex nor")
        print("      respiration can account for the finding. If it is not, the")
        print("      result is a shape difference at the top of the band.")
    except Exception:
        pass
    return out


def stationarity_report(df_h, df_p, output_dir):
    """Group comparison of the per-subject stationarity diagnostics."""
    rows = []
    for col in ("Stat_RA_p", "Stat_VarRatio", "Stat_ADF_p", "Stat_KPSS_p"):
        if col not in df_h.columns or col not in df_p.columns:
            continue
        r, pv, nh, npat = _mw(df_h[col], df_p[col])
        rows.append({"Statistic": col,
                     "Control_median": pd.to_numeric(df_h[col], errors="coerce").median(),
                     "Patient_median": pd.to_numeric(df_p[col], errors="coerce").median(),
                     "r_effect": r, "p_raw": pv, "N_Healthy": nh, "N_Psych": npat})
    for col, lab in (("Stat_RA_Stationary", "Reverse-arrangement: stationary"),):
        if col in df_h.columns and col in df_p.columns:
            hs = _as_bool(df_h[col]).sum(); ps = _as_bool(df_p[col]).sum()
            rows.append({"Statistic": lab, "Control_median": int(hs),
                         "Patient_median": int(ps), "r_effect": np.nan,
                         "p_raw": np.nan, "N_Healthy": len(df_h), "N_Psych": len(df_p)})
    if not rows:
        print("  Stationarity report skipped: Stat_* columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Stationarity_Report.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Stationarity report -> {path}")
    for _, r in out.iterrows():
        print(f"    {r['Statistic']:<34} ctrl {r['Control_median']}  pt {r['Patient_median']}")
    return out


def entropy_tolerance_report(df_h, df_p, output_dir):
    """Sample entropy group comparison across the tolerance sweep."""
    rows = []
    for frac in (10, 15, 20, 25, 30, 35):
        col = f"SampEn_r{frac:02d}"
        if col not in df_h.columns or col not in df_p.columns:
            continue
        r, pv, nh, npat = _mw(df_h[col], df_p[col])
        rows.append({"Tolerance_r_x_SD": frac / 100.0,
                     "Control_mean": pd.to_numeric(df_h[col], errors="coerce").mean(),
                     "Patient_mean": pd.to_numeric(df_p[col], errors="coerce").mean(),
                     "r_effect": r, "p_raw": pv, "N_Healthy": nh, "N_Psych": npat})
    if not rows:
        print("  Entropy tolerance report skipped: SampEn_r** columns absent.")
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Entropy_Tolerance_Sensitivity.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Entropy tolerance sensitivity -> {path}")
    for _, r in out.iterrows():
        print(f"    r = {r['Tolerance_r_x_SD']:.2f} x SD   ctrl {r['Control_mean']:.3f}  "
              f"pt {r['Patient_mean']:.3f}  r_eff {r['r_effect']:+.3f}  p {r['p_raw']:.4f}")
    return out


def write_manuscript_table1(stats, output_dir, healthy_label="Healthy"):
    """Write Table 1 as Markdown, generated from the statistics, with the control cohort and sample sizes
    taken from the data.
    """
    fam = stats
    if "FDR_Family" in stats.columns:
        prim = stats[stats["FDR_Family"].astype(str).str.lower().str.contains("primary")]
        if len(prim):
            fam = prim
    def _n(col):
        if col in fam.columns:
            v = pd.to_numeric(fam[col], errors="coerce").dropna()
            if len(v):
                lo, hi = int(v.min()), int(v.max())
                return f"{hi}" if lo == hi else f"{lo}-{hi}"
        return "?"
    _cohort = {"HRV-ACC Control": "HRV-ACC healthy controls",
               "MIT-BIH NSR": "MIT-BIH NSR healthy controls"}.get(
                   healthy_label, f"{healthy_label} controls")
    lines = [
        f"**Table 1. Group comparison of all {len(fam)} primary metrics: "
        f"{_cohort} (N = {_n('Healthy_N')}) vs. psychiatric patients "
        f"(N = {_n('Psych_N')}).**",
        "",
        "| Metric | Healthy Mean ± SD | Psychiatric Mean ± SD | % Diff | *r* | "
        "*p*~raw~ | *p*~adj~ | Sig. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    def _fmt(v):
        """Fixed-point formatting with magnitude-appropriate precision (no scientific notation)."""
        if not np.isfinite(v):
            return "—"
        a = abs(v)
        if a >= 100:
            return f"{v:,.1f}"
        if a >= 10:
            return f"{v:.2f}"
        if a >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"

    def _minus(s):
        return s.replace("-", "\u2212")

    for name, r in fam.iterrows():
        praw = "<0.0001" if r["p_raw"] < 1e-4 else f"{r['p_raw']:.4f}"
        padj = "<0.0001" if r["p_adj"] < 1e-4 else f"{r['p_adj']:.4f}"
        sig = "**" if bool(r["Significant"]) else "—"
        lines.append(
            f"| {name} | {_fmt(r['Healthy_Mean'])} ± {_fmt(r['Healthy_SD'])} | "
            f"{_fmt(r['Psych_Mean'])} ± {_fmt(r['Psych_SD'])} | "
            f"{_minus(f'{r["Pct_Diff"]:+.1f}')}% | "
            f"{_minus(f'{r["r_effect"]:+.3f}')} | {praw} | {padj} | {sig} |")
    n_sig = int(fam["Significant"].sum())
    lines += ["",
        f"Significance: ** *p*~adj~ < 0.05 (Benjamini-Hochberg within the "
        f"{len(fam)}-metric primary family); {n_sig} of {len(fam)} metrics "
        f"reached significance. SD1 is RMSSD/sqrt(2), SD2 is determined by SDNN "
        f"and SD1 (SD2^2 = 2 SDNN^2 - SD1^2), and mean heart rate and mean NN "
        f"are exact inverses, so these measures are not independent tests. The "
        f"recurrence rate and the recurrence-network mean node degree are fixed "
        f"at the target value by construction under exact-count thresholding and "
        f"are reported as normalisation checks rather than as tests, so they are "
        f"excluded from this family."]
    path = os.path.join(output_dir, "TABLE1_MANUSCRIPT.md")
    # Explicit UTF-8 (the Windows default codepage cannot encode the Unicode minus sign).
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Manuscript Table 1 -> {path}  ({len(fam)} rows, {n_sig} significant)")
    print("    Paste this file directly rather than editing numbers by hand.")
    return path


def write_test_implementation(output_dir):
    """Record the exact test configuration (continuity and tie corrections)."""
    import scipy
    rows = [
        {"Item": "Primary test", "Value": "Mann-Whitney U, two-sided"},
        {"Item": "Continuity correction", "Value": str(MWU_USE_CONTINUITY)},
        {"Item": "Tie correction", "Value": "applied (SciPy default)"},
        {"Item": "Method", "Value": MWU_METHOD},
        {"Item": "Effect size", "Value": "rank-biserial r = 1 - 2U/(n1*n2)"},
        {"Item": "Multiple comparisons", "Value": "Benjamini-Hochberg within family"},
        {"Item": "SciPy version", "Value": scipy.__version__},
        {"Item": "NumPy version", "Value": np.__version__},
        {"Item": "pandas version", "Value": pd.__version__},
    ]
    path = os.path.join(output_dir, "Test_Implementation.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Test implementation -> {path}")
    return path


def ambulatory_fraction_test(output_dir):
    """Group comparison of the ambulatory beat fraction (per-subject fractions from accelerometer_mobility.py)."""
    src = os.path.join("outputs", "accelerometer", "Stationary_Segments.csv")
    if not os.path.exists(src):
        print("  Ambulatory fraction test skipped: Stationary_Segments.csv not "
              "found. Run accelerometer_mobility.py first.")
        return None
    d = pd.read_csv(src)
    d = d[d["Record_ID"].astype(str).str.match(r"^(control|treatment)_\d+$")]
    d = d.assign(Group=np.where(d["Record_ID"].str.startswith("control"),
                                "control", "patient"))
    a = pd.to_numeric(d.loc[d.Group == "control", "Pct_Ambulatory"],
                      errors="coerce").dropna()
    b = pd.to_numeric(d.loc[d.Group == "patient", "Pct_Ambulatory"],
                      errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        print("  Ambulatory fraction test skipped: too few subjects.")
        return None
    r, p, n1, n2 = _mw(a, b)
    out = pd.DataFrame([{
        "Variable": "Ambulatory beat fraction (%)",
        "Control_mean": a.mean(), "Control_SD": a.std(ddof=1),
        "Patient_mean": b.mean(), "Patient_SD": b.std(ddof=1),
        "r_effect": r, "p_raw": p, "N_control": n1, "N_patient": n2}])
    path = os.path.join(output_dir, "Ambulatory_Fraction_Test.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Ambulatory fraction -> {path}")
    print(f"    control {a.mean():.1f}% vs patient {b.mean():.1f}%; "
          f"r = {r:+.3f}, p = {p:.4f}")
    return out


# ---- Heart rate, VLF decomposition and the 7.8 ms grid ----

def _stack(df_h, df_p, col):
    """One column from both cohorts, controls first, as a float array."""
    return np.concatenate([pd.to_numeric(df_h[col], errors="coerce").values,
                           pd.to_numeric(df_p[col], errors="coerce").values])


def _pearson_partial_p(r, n, k):
    """Two-sided p for a Pearson (partial) correlation on n points with k covariates removed."""
    dof = n - 2 - k
    if not np.isfinite(r) or dof <= 0 or abs(r) >= 1.0:
        return np.nan
    tval = r * np.sqrt(dof / (1.0 - r ** 2))
    return float(2.0 * t_dist.sf(abs(tval), dof))


def _rank_partial(y, g, x):
    """Rank-based partial correlation: y, group and covariate replaced by ranks, the covariate regressed
    out of both, residuals correlated (Pearson). Assumes only a monotone heart-rate relationship.
    """
    ok = np.isfinite(y) & np.isfinite(g) & np.isfinite(x)
    if ok.sum() < 6:
        return np.nan, np.nan
    ry, rg, rx = rankdata(y[ok]), rankdata(g[ok]), rankdata(x[ok])
    a, b = _ols_residuals(ry, rx), _ols_residuals(rg, rx)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(a, b)[0, 1])
    return r, _pearson_partial_p(r, int(ok.sum()), 1)


def hr_family_available(df_h, df_p):
    """Members of HR_ADJUSTED_FAMILY testable in this pair of cohorts; used to size the family and to run
    the tests.
    """
    cov = HR_ADJUSTMENT_COVARIATE
    if cov not in df_h.columns or cov not in df_p.columns:
        return []
    out = []
    for m in HR_ADJUSTED_FAMILY:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        nh = pd.to_numeric(df_h[m], errors="coerce").notna().sum()
        npat = pd.to_numeric(df_p[m], errors="coerce").notna().sum()
        if nh >= 3 and npat >= 3:
            out.append(m)
    return out


def hr_adjusted_comparison(df_h, df_p, output_dir):
    """Group differences with mean heart rate held constant.

    Over a fixed 1800 s window beat count is heart rate (r = 0.9995), so every length-sensitive measure is
    also heart-rate-sensitive. Heart rate is plausibly downstream of group membership, so an adjusted
    estimate is a decomposition (the part of a difference not carried linearly by heart rate), not an
    effect with a confounder removed.

    Method: partial Spearman correlation of each measure with group, mean heart rate regressed out of both
    by OLS with an intercept, residuals rank-correlated (regress, then rank). The unadjusted baseline is
    the same statistic without a covariate; rank-biserial r is shown for orientation only. The adjusted
    tests form their own BH family (HR_ADJUSTED_FAMILY).

    Model-form sensitivity (not corrected): heart rate as a quadratic, the measure log-transformed (against
    heart rate, and against log mean NN), and a fully rank-based partial; plus adjustment for beat count.
    """
    members = hr_family_available(df_h, df_p)
    if not members:
        print("  Heart-rate-adjusted comparison skipped: covariate or measures "
              "absent.")
        return None
    g = np.concatenate([np.zeros(len(df_h)), np.ones(len(df_p))])
    hr = _stack(df_h, df_p, HR_ADJUSTMENT_COVARIATE)
    nn = (_stack(df_h, df_p, "HRV_MeanNN")
          if ("HRV_MeanNN" in df_h.columns and "HRV_MeanNN" in df_p.columns)
          else 60000.0 / hr)
    beats = (_stack(df_h, df_p, "Num_RR")
             if ("Num_RR" in df_h.columns and "Num_RR" in df_p.columns)
             else None)

    rows = []
    for m in members:
        y = _stack(df_h, df_p, m)
        ok = np.isfinite(y) & np.isfinite(hr)
        hv = pd.to_numeric(df_h[m], errors="coerce").dropna()
        pv_ = pd.to_numeric(df_p[m], errors="coerce").dropna()
        r_rb, p_rb, n_h, n_p = _mw(df_h[m], df_p[m])
        r_u, p_u = spearmanr(y[ok], g[ok])
        r_lin, p_lin = partial_spearman(y, g, [hr])
        r_quad, p_quad = partial_spearman(y, g, [hr, hr ** 2])
        positive = bool(np.all(y[ok] > 0))
        if positive:
            ly = np.full_like(y, np.nan)
            ly[ok] = np.log(y[ok])
            r_logy, p_logy = partial_spearman(ly, g, [hr])
            r_loglog, p_loglog = partial_spearman(ly, g, [np.log(nn)])
        else:
            r_logy = p_logy = r_loglog = p_loglog = np.nan
        r_rank, p_rank = _rank_partial(y, g, hr)
        if beats is not None:
            r_beats, p_beats = partial_spearman(y, g, [beats])
        else:
            r_beats = p_beats = np.nan
        forms = {"linear": r_lin, "quadratic": r_quad, "log_measure": r_logy,
                 "log_log_meanNN": r_loglog, "rank": r_rank}
        fin = {k: v for k, v in forms.items() if np.isfinite(v)}
        weakest = min(fin, key=lambda k: abs(fin[k])) if fin else None
        strongest = max(fin, key=lambda k: abs(fin[k])) if fin else None
        rows.append({
            "Metric": m,
            "Covariate": HR_ADJUSTMENT_COVARIATE,
            "N_Healthy": n_h, "N_Psych": n_p,
            "Healthy_Median": float(hv.median()), "Psych_Median": float(pv_.median()),
            "Healthy_SD": float(hv.std(ddof=1)), "Psych_SD": float(pv_.std(ddof=1)),
            # For orientation only; not comparable with the partials.
            "r_rank_biserial": r_rb, "p_rank_biserial": p_rb,
            # Baseline: partial Spearman with no covariate.
            "r_partial_unadjusted": float(r_u), "p_partial_unadjusted": float(p_u),
            # Primary adjusted estimate; p_raw and p_adj are for this test.
            "r_partial_HR": float(r_lin),
            "p_raw": float(p_lin),
            # Ratio of two estimates: descriptive only, no interval.
            "pct_retained": (100.0 * r_lin / r_u
                             if np.isfinite(r_u) and r_u != 0 else np.nan),
            "r_HR_quadratic": r_quad, "p_HR_quadratic": p_quad,
            "r_HR_log_measure": r_logy, "p_HR_log_measure": p_logy,
            "r_HR_log_log_meanNN": r_loglog, "p_HR_log_log_meanNN": p_loglog,
            "r_HR_rank_based": r_rank, "p_HR_rank_based": p_rank,
            "Model_Form_Weakest": weakest,
            "r_Model_Form_Weakest": fin.get(weakest, np.nan) if weakest else np.nan,
            "Model_Form_Strongest": strongest,
            "r_Model_Form_Strongest": fin.get(strongest, np.nan) if strongest else np.nan,
            "Log_Forms_Note": ("" if positive else
                               "not computed: measure takes non-positive values"),
            "r_partial_beats": r_beats, "p_partial_beats": p_beats,
        })
    out = pd.DataFrame(rows)
    out["p_adj"] = benjamini_hochberg(out["p_raw"].values)
    out["Significant"] = out["p_adj"] < ALPHA
    out["FDR_Family"] = "hr_adjusted"
    out["FDR_Family_N"] = len(out)
    path = os.path.join(output_dir, "HR_Adjusted_Comparison.csv")
    out.to_csv(path, index=False, float_format="%.8g")
    print(f"  Heart-rate-adjusted comparison ({len(out)} tests, BH within "
          f"family) -> {path}")
    print(f"  {'metric':<21}{'unadj':>8}{'| HR':>8}{'p_adj':>10}{'kept':>7}"
          f"   model forms")
    for _, rw in out.iterrows():
        print(f"  {rw['Metric']:<21}{rw['r_partial_unadjusted']:>+8.3f}"
              f"{rw['r_partial_HR']:>+8.3f}{rw['p_adj']:>10.2g}"
              f"{rw['pct_retained']:>6.0f}%   "
              f"{rw['r_Model_Form_Weakest']:+.3f} to "
              f"{rw['r_Model_Form_Strongest']:+.3f}")
    print("    Heart rate is plausibly downstream of group: read the adjusted "
          "column as a decomposition, not as confounder control.")
    return out


def meannn_normalised_comparison(df_h, df_p, output_dir):
    """Amplitude measures divided by mean NN (band powers by its square) and compared between groups
    (Sacha, 2014). A sensitivity analysis on measures already in the heart-rate family; not corrected again.
    """
    if "HRV_MeanNN" not in df_h.columns or "HRV_MeanNN" not in df_p.columns:
        print("  Mean-NN normalisation skipped: HRV_MeanNN absent.")
        return None
    rows = []
    for m, power in MEANNN_NORMALISED:
        if m not in df_h.columns or m not in df_p.columns:
            continue
        hv = (pd.to_numeric(df_h[m], errors="coerce")
              / pd.to_numeric(df_h["HRV_MeanNN"], errors="coerce") ** power)
        pv_ = (pd.to_numeric(df_p[m], errors="coerce")
               / pd.to_numeric(df_p["HRV_MeanNN"], errors="coerce") ** power)
        r, p, n_h, n_p = _mw(hv, pv_)
        label = f"{m} / MeanNN" + ("^2" if power == 2 else "")
        rows.append({"Metric": label, "Measure": m, "MeanNN_Exponent": power,
                     "Healthy_Median": float(hv.median()),
                     "Psych_Median": float(pv_.median()),
                     "Healthy_SD": float(hv.std(ddof=1)),
                     "Psych_SD": float(pv_.std(ddof=1)),
                     "r_effect": r, "p_raw": p,
                     "N_Healthy": n_h, "N_Psych": n_p})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "HR_Normalised_MeanNN.csv")
    out.to_csv(path, index=False, float_format="%.8g")
    print(f"  Mean-NN normalised amplitude -> {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<28} r = {rw['r_effect']:+.3f}  "
              f"p = {rw['p_raw']:.2g}")
    return out


def vlf_decomposition_report(df_h, df_p, output_dir):
    """Distribution of total power across the three bands: VLF, LF and HF power, VLF/LF, LF/HF, and each
    band's share of total power per subject. Shares do not sum to exactly one (the trapezoid between bands
    belongs to the total only). VLF is integrated from the lowest resolved bin; its actual lower edge and
    bin count are in the note. p values are uncorrected: these decompose measures tested elsewhere.
    """
    need = ["VLF_Power_ms2", "LF_Power_ms2", "HF_Power_ms2", "Total_Power_ms2"]
    if any(c not in df_h.columns or c not in df_p.columns for c in need):
        print("  VLF decomposition skipped: band-power columns absent.")
        return None

    def _series(df, m):
        if m.endswith("_Share"):
            band = m.replace("_Share", "_Power_ms2")
            return (pd.to_numeric(df[band], errors="coerce")
                    / pd.to_numeric(df["Total_Power_ms2"], errors="coerce"))
        return pd.to_numeric(df[m], errors="coerce")

    band_note = ""
    lo = _num_col(pd.concat([df_h, df_p]), "VLF_LowestFreq_Hz")
    nb = _num_col(pd.concat([df_h, df_p]), "VLF_NBins")
    if lo is not None and nb is not None:
        band_note = (f"VLF integrated from {lo.median():.4f} Hz "
                     f"({nb.median():.0f} bins), not the nominal 0.003 Hz")
    rows = []
    for m in ("VLF_Power_ms2", "LF_Power_ms2", "HF_Power_ms2",
              "Total_Power_ms2", "VLF_LF_Ratio",
              "VLF_Share", "LF_Share", "HF_Share"):
        if not m.endswith("_Share") and (m not in df_h.columns
                                         or m not in df_p.columns):
            continue
        hv, pv_ = _series(df_h, m), _series(df_p, m)
        r, p, n_h, n_p = _mw(hv, pv_)
        hmed, pmed = float(hv.median()), float(pv_.median())
        rows.append({"Metric": m,
                     "Healthy_Median": hmed, "Psych_Median": pmed,
                     "Healthy_Mean": float(hv.mean()), "Psych_Mean": float(pv_.mean()),
                     "Healthy_SD": float(hv.std(ddof=1)),
                     "Psych_SD": float(pv_.std(ddof=1)),
                     "Pct_Median_Change": (100.0 * (pmed - hmed) / abs(hmed)
                                           if hmed else np.nan),
                     "r_effect": r, "p_raw": p,
                     "N_Healthy": n_h, "N_Psych": n_p,
                     "Note": band_note if m == "VLF_Power_ms2" else ""})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "VLF_Decomposition.csv")
    out.to_csv(path, index=False, float_format="%.8g")
    print(f"  VLF decomposition -> {path}")
    for _, rw in out.iterrows():
        print(f"    {rw['Metric']:<17} median {rw['Healthy_Median']:>10.4g} -> "
              f"{rw['Psych_Median']:<10.4g} ({rw['Pct_Median_Change']:+.1f}%)  "
              f"r = {rw['r_effect']:+.3f}  p = {rw['p_raw']:.2g}")
    if band_note:
        print(f"    {band_note}")
    return out


# Nonlinear measures re-quantized inside compute_complexity_metrics().
REQUANT_NONLINEAR = ("SampEn", "ApEn", "PermEn", "MSE_Mean",
                     "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff",
                     "RNA_Transitivity", "DFA_Alpha1", "DFA_Alpha2")


def requantization_report(df_h, df_p, output_dir):
    """Every measure at the native 1.000 ms resolution and on the MIT-BIH 7.8125 ms grid, side by side
    (rank-biserial r in both columns). HRV-ACC comparison only; MIT-BIH is natively on this grid.

    A second table reports RMSSD in grid steps and the resulting tie rates.
    """
    bases = [b for b in list(REQUANT_NONLINEAR) + list(REQUANT_AMPLITUDE_SPECTRAL)
             if b in df_h.columns and b in df_p.columns
             and f"{b}_Q128" in df_h.columns and f"{b}_Q128" in df_p.columns]
    if not bases:
        print("  Re-quantization report skipped: no *_Q128 columns in both "
              "cohorts (expected for the MIT-BIH comparison).")
        return None
    rows = []
    for b in bases:
        r0, p0, n_h, n_p = _mw(df_h[b], df_p[b])
        r1, p1, _, _ = _mw(df_h[f"{b}_Q128"], df_p[f"{b}_Q128"])
        rows.append({
            "Metric": b,
            "Family": ("amplitude_spectral" if b in REQUANT_AMPLITUDE_SPECTRAL
                       else "nonlinear"),
            "Healthy_Median_Native": float(pd.to_numeric(df_h[b], errors="coerce").median()),
            "Psych_Median_Native": float(pd.to_numeric(df_p[b], errors="coerce").median()),
            "Healthy_Median_Q128": float(pd.to_numeric(df_h[f"{b}_Q128"], errors="coerce").median()),
            "Psych_Median_Q128": float(pd.to_numeric(df_p[f"{b}_Q128"], errors="coerce").median()),
            "r_native": r0, "p_native": p0,
            "r_Q128": r1, "p_Q128": p1,
            "r_change": r1 - r0,
            "N_Healthy": n_h, "N_Psych": n_p})
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Requantization_Q128.csv")
    out.to_csv(path, index=False, float_format="%.8g")
    print(f"  Re-quantization to 7.8125 ms -> {path}")
    print(f"  {'metric':<22}{'native r':>10}{'7.8 ms r':>10}{'p (7.8 ms)':>12}")
    for _, rw in out.iterrows():
        print(f"  {rw['Metric']:<22}{rw['r_native']:>+10.3f}{rw['r_Q128']:>+10.3f}"
              f"{rw['p_Q128']:>12.2g}")

    floor = []
    for q in ("Requant_RMSSD_In_Steps", "Requant_Tie_Rate", "Ordinal_Tie_Rate",
              "Quant_Step_Over_RMSSD"):
        if q not in df_h.columns or q not in df_p.columns:
            continue
        hv = pd.to_numeric(df_h[q], errors="coerce").dropna()
        pv_ = pd.to_numeric(df_p[q], errors="coerce").dropna()
        r, p, n_h, n_p = _mw(hv, pv_)
        floor.append({"Quantity": q,
                      "Grid": ("7.8125 ms" if q.startswith("Requant") else
                               f"{QUANT_STEP_S * 1000:.3f} ms (native)"),
                      "Healthy_Median": float(hv.median()),
                      "Psych_Median": float(pv_.median()),
                      "Healthy_Min": float(hv.min()), "Psych_Min": float(pv_.min()),
                      "Healthy_Max": float(hv.max()), "Psych_Max": float(pv_.max()),
                      "r_effect": r, "p_raw": p,
                      "N_Healthy": n_h, "N_Psych": n_p})
    if floor:
        fl = pd.DataFrame(floor)
        fpath = os.path.join(output_dir, "Requantization_Resolution_Floor.csv")
        fl.to_csv(fpath, index=False, float_format="%.8g")
        print(f"  Resolution floor -> {fpath}")
        for _, rw in fl.iterrows():
            print(f"    {rw['Quantity']:<24} median {rw['Healthy_Median']:.4g} vs "
                  f"{rw['Psych_Median']:.4g}  (min {rw['Healthy_Min']:.3g} / "
                  f"{rw['Psych_Min']:.3g})")
    return out


def _entrypoint_order_selftest():
    """Fail fast if a function run_comparison() needs is defined after the __main__ block."""
    import inspect as _inspect
    missing = [name for name in (
        "hr_adjusted_comparison", "meannn_normalised_comparison",
        "vlf_decomposition_report", "requantization_report",
        "hr_family_available",
        "report_minimum_detectable_effect", "mayer_wave_analysis",
        "quantization_sensitivity", "surrogate_normalized_entropy",
        "beta_dfa_consistency", "nonredundant_family_comparison",
        "length_and_respiration_checks", "pnn50_floor_report",
        "schizophrenia_subgroup_analysis", "run_metadata_analyses",
        "deterministic_checks", "beta_band_sweep_comparison",
        "write_manuscript_table1", "write_test_implementation",
        "ambulatory_fraction_test",
        "age_adjusted_cross_device",
        "stationarity_report", "entropy_tolerance_report",
        "metadata_cohort_description", "diagnosis_subgroup_analysis",
        "quetiapine_analysis", "comorbidity_analysis", "age_sex_analysis",
        "demographic_matching_report", "age_sex_adjusted_comparison",
        "age_association_both_groups")
        if name not in globals() or not callable(globals()[name])]
    if missing:
        raise RuntimeError(
            "comparison.py is mis-ordered: run_comparison() needs "
            f"{missing}, which are not defined yet. Move the "
            "`if __name__ == \"__main__\":` block to the END of the file.")


_entrypoint_order_selftest()


# ---- Entry point ----
# This block must stay at the end of the file, after every function it calls;
# _entrypoint_order_selftest() checks this.


if __name__ == "__main__":

    # -- Primary comparison: HRV-ACC controls against patients (same device and protocol) --
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

    # -- MIT-BIH controls against patients --
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

    # -- Cross-control and three-group analyses (needs all three cohorts) --
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
