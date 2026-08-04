"""
psychiatric.py
==============
HRV-ACC Zenodo dataset — psychiatric cohort analysis.

Reads pre-extracted RR interval CSVs (Polar H10 format) for 30 treatment
subjects, runs the full pipeline from pipeline.py, and produces per-subject
outputs and a cohort results CSV.

Key differences from healthy.py
---------------------------------
  • Input is CSV (RR intervals in ms), NOT a WFDB record.
    A lightweight loader converts the CSV to a clean numpy array in seconds
    before handing off to the shared pipeline functions.
  • Recording window: skip the first 600 s of each file (10 min — third
    attempt after 60 s and 360 s skips both yielded mean HR ~100 bpm;
    if still elevated, 360 s will be used and noted in the pilot report),
    then take exactly 10 minutes — matched to the healthy cohort window length.
  • Radar charts are z-scored using the HEALTHY cohort mean and SD
    (healthy_m / healthy_s hardcoded below) so psychiatric profiles are
    directly interpretable relative to the healthy baseline.
  • No ECG signal plots (raw/cleaned ECG panels) because the input is
    already RR intervals, not raw voltage. The diagnostic figure is
    replaced by a four-panel RR-only figure.
  • Output folder: outputs/psychiatric/
  • CSV saved as:  outputs/psychiatric/ecg_results_Psychiatric.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d
from scipy.signal import welch
from neurokit2.complexity import fractal_dfa
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ── Import shared pipeline functions ────────────────────────────
from pipeline import (
    reject_ectopic_beats,
    compute_frequency_domain,
    compute_complexity_metrics,
    run_surrogate_testing,
    _plot_mse_curve,
    _draw_corr_matrix,
    plot_correlation_matrices,
    plot_radar_charts,
    plot_radar_comparison,
    # Constants reused below
    VLF_BAND, LF_BAND, HF_BAND, BETA_FIT_BAND,
    INTERP_FS, PSD_BETA_R2_THRESHOLD,
    N_SURROGATES, SURROGATE_ALPHA,
    RADAR_METRICS, RADAR_LABELS,
)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE       = "Psychiatric ECGs"
OUTPUT_DIR = os.path.join("outputs", "psychiatric")

TREATMENT_IDS = [
    1, 2, 3, 4, 7, 8, 9, 13, 15, 16, 17, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 29, 31, 32, 33, 36, 37, 38, 40, 41, 42,
]

# Window parameters — must match healthy cohort analysis length (30 min).
# SKIP_SECONDS history:
#   60 s  → mean HR ~100 bpm (ambulatory activity)
#   360 s → mean HR ~99.6 bpm (still elevated, insufficient settling)
#   600 s → current value; if HR remains elevated we revert to 360 s
#            and note persistent tachycardia in the pilot report.
SKIP_SECONDS     = 600   # discard first 10 min
ANALYSIS_SECONDS = 1800  # 30 minutes — matched to healthy cohort analysis length


# ═══════════════════════════════════════════════════════════════
# HEALTHY COHORT NORMALISATION PARAMETERS
# ═══════════════════════════════════════════════════════════════
# Extracted from ecg_results_Healthy.csv (Healthy_Mean / Healthy_SD rows).
# All radar chart z-scores for the psychiatric cohort use these values
# so both groups share the same reference frame.
# Only the five radar metrics are needed here.
#
# ⚠️  UPDATE REQUIRED BEFORE RUNNING ⚠️
# These values are from the OLD N=10 / 10-min healthy cohort.
# After running healthy.py with N=18 / 30-min recordings, copy the
# Healthy_Mean and Healthy_SD rows from ecg_results_Healthy.csv into
# the five values below, then re-run this script.
# The healthy.py entry point also prints these values at completion.

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


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CSV LOADER
# ═══════════════════════════════════════════════════════════════

def load_rr_from_csv(filepath, skip_seconds=SKIP_SECONDS,
                     analysis_seconds=ANALYSIS_SECONDS,
                     verbose=True):
    """
    Load RR intervals from a Polar H10 / HRV-ACC CSV file.

    File format (semicolon-delimited):
        Phone timestamp;RR-interval [ms]
        10:10:56.731000;546
        ...

    Processing steps
    ----------------
    1. Parse the RR column (integer milliseconds).
    2. Reconstruct a cumulative time axis by summing RR intervals
       (in seconds) from t=0.  The timestamp column is NOT used —
       it reflects wall-clock time and is irrelevant for analysis.
    3. Skip the first `skip_seconds` of data (settling artifact).
    4. Take the next `analysis_seconds` of data.

    Returns
    -------
    rr_seconds  : np.ndarray   RR intervals in seconds (not yet artifact-filtered)
    rr_times    : np.ndarray   cumulative time axis in seconds (same length)
    n_total     : int          total beats in file before windowing
    """
    df = pd.read_csv(filepath, sep=';')

    # Robust column detection — handle minor naming variations
    rr_col = None
    for col in df.columns:
        if 'rr' in col.lower() or 'interval' in col.lower():
            rr_col = col
            break
    if rr_col is None:
        raise ValueError(f"Cannot find RR column in {filepath}. "
                         f"Columns: {df.columns.tolist()}")

    rr_ms    = pd.to_numeric(df[rr_col], errors='coerce').dropna().values
    n_total  = len(rr_ms)
    rr_s     = rr_ms / 1000.0                          # ms → seconds
    cum_time = np.cumsum(rr_s)                         # cumulative time axis

    # Window: [skip_seconds, skip_seconds + analysis_seconds]
    t_start = skip_seconds
    t_end   = skip_seconds + analysis_seconds

    mask       = (cum_time >= t_start) & (cum_time < t_end)
    rr_window  = rr_s[mask]
    time_window = cum_time[mask]

    if verbose:
        total_duration = cum_time[-1] if len(cum_time) > 0 else 0
        print(f"    Loaded {n_total} beats, "
              f"total duration {total_duration/60:.1f} min")
        print(f"    Window [{t_start}s – {t_end}s]: "
              f"{len(rr_window)} beats")

    if len(rr_window) < 50:
        raise ValueError(
            f"Only {len(rr_window)} beats in analysis window — "
            f"check skip/analysis seconds or file length."
        )

    return rr_window, time_window, n_total


# ═══════════════════════════════════════════════════════════════
# STEP 2 — FOUR-PANEL DIAGNOSTIC PLOT (RR-only, no raw ECG)
# ═══════════════════════════════════════════════════════════════

def _plot_rr_diagnostic(record_id, rr_times_raw, rr_raw,
                        rr_times_clean, rr_clean,
                        freqs, psd_power, complexity_metrics,
                        psd_beta, output_dir):
    """
    Four-panel diagnostic figure for RR-interval-only input.

    Panels
    ------
    1. Raw vs cleaned RR intervals over time
    2. PSD with band shading and 1/f fit overlay
    3. Poincaré plot
    4. Complexity / criticality bar chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ECG Criticality Analysis — {record_id} (Psychiatric)",
        fontsize=13
    )

    # 1) RR intervals — raw vs cleaned
    ax = axes[0, 0]
    ax.plot(rr_times_raw,   rr_raw   * 1000, lw=0.8,
            color='gray',   alpha=0.6, label='Raw')
    ax.plot(rr_times_clean, rr_clean * 1000, lw=0.9,
            color='steelblue', label='Cleaned')
    ax.set_title("RR Intervals (raw vs artifact-rejected)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("RR (ms)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2) PSD with band shading + 1/f fit
    ax = axes[0, 1]
    ax.semilogy(freqs, psd_power, color='darkviolet', lw=1.2, label='PSD')
    ax.axvspan(*VLF_BAND, alpha=0.15, color='blue',   label='VLF')
    ax.axvspan(*LF_BAND,  alpha=0.15, color='green',  label='LF')
    ax.axvspan(*HF_BAND,  alpha=0.15, color='orange', label='HF')
    if np.isfinite(psd_beta):
        mask = (
            (freqs >= BETA_FIT_BAND[0]) &
            (freqs <= BETA_FIT_BAND[1]) &
            (freqs > 0)
        )
        if mask.sum() >= 3:
            lf_ = np.log10(freqs[mask])
            lp_ = np.log10(psd_power[mask])
            ic  = np.mean(lp_ + psd_beta * lf_)
            ax.semilogy(freqs[mask], 10 ** (ic - psd_beta * lf_),
                        color='red', lw=1.4, linestyle='--',
                        label=f'1/f fit β={psd_beta:.3f}')
    ax.set_title("HRV Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (s²/Hz)")
    ax.legend(fontsize=8); ax.set_xlim(0, 0.5); ax.grid(True, alpha=0.3)

    # 3) Poincaré plot
    ax = axes[1, 0]
    ax.scatter(rr_clean[:-1] * 1000, rr_clean[1:] * 1000,
               s=8, alpha=0.5, color='teal')
    ax.set_title("Poincaré Plot")
    ax.set_xlabel("RR$_n$ (ms)"); ax.set_ylabel("RR$_{n+1}$ (ms)")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 4) Complexity bar chart
    ax = axes[1, 1]
    bar_labels = ["SampEn", "ApEn", "PermEn", "MSE\nMean",
                  "DFA\nα",  "DFA\nα1", "DFA\nα2"]
    bar_keys   = ["SampEn", "ApEn", "PermEn", "MSE_Mean",
                  "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2"]
    bar_vals   = [complexity_metrics.get(k, np.nan) for k in bar_keys]
    bar_colors = ['steelblue', 'darkorange', 'green',
                  'red', 'purple', 'brown', 'pink']
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors,
                  edgecolor='white', width=0.6)
    for bar, val in zip(bars, bar_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_title("Complexity / Criticality Metrics")
    ax.set_ylabel("Value"); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{record_id}_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostics     → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — FULL RECORD ANALYSIS (one CSV subject)
# ═══════════════════════════════════════════════════════════════

def analyze_psychiatric_record(filepath, plot=True,
                                run_surrogates=True,
                                verbose=True,
                                output_dir=OUTPUT_DIR):
    """
    End-to-end pipeline for one HRV-ACC CSV subject.

    Replaces analyze_record() from pipeline.py for CSV-format input.
    All metric computation delegates to shared pipeline functions.

    Returns
    -------
    dict  flat dict of all metrics (matching healthy cohort column names)
    """
    record_id = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}\n  Analyzing: {record_id}\n{'='*60}")

    # ── Load RR intervals from CSV ────────────────────────────
    rr_raw, rr_times_raw, n_total = load_rr_from_csv(
        filepath, verbose=verbose
    )

    if verbose:
        print(f"  {len(rr_raw)} beats in analysis window")

    # ── Artifact / ectopic rejection ─────────────────────────
    rr_clean, artifact_pct, num_rr = reject_ectopic_beats(
        rr_raw, verbose=verbose
    )

    # Rebuild time axis for cleaned series
    rr_times_clean = np.concatenate(
        [[rr_times_raw[0]], rr_times_raw[0] + np.cumsum(rr_clean[:-1])]
    )

    # ── Time-domain HRV ───────────────────────────────────────
    # Computed directly from cleaned RR array (no WFDB / R-peak detection).
    # Matches the formulas used by nk.hrv_time() so results are comparable.
    if verbose:
        print("  Time-domain HRV...")
    rr_ms = rr_clean * 1000.0
    successive_diff = np.diff(rr_ms)

    hrv_meannn = float(np.mean(rr_ms))
    hrv_sdnn   = float(np.std(rr_ms, ddof=1))
    hrv_rmssd  = float(np.sqrt(np.mean(successive_diff ** 2)))
    hrv_pnn50  = float(np.sum(np.abs(successive_diff) > 50) /
                       len(successive_diff) * 100)
    hrv_meanhr = float(60.0 / np.mean(rr_clean))

    # ── Poincaré (SD1, SD2) ───────────────────────────────────
    # SD1 = short-term variability (RMSSD / sqrt(2))
    # SD2 = long-term variability
    sd1     = hrv_rmssd / np.sqrt(2)
    sd2     = float(np.sqrt(2 * hrv_sdnn**2 - sd1**2))
    sd1sd2  = sd1 / sd2 if sd2 > 0 else np.nan

    # ── Frequency domain + PSD beta ──────────────────────────
    if verbose:
        print("  Frequency-domain HRV + PSD beta...")
    freq_metrics, freqs, power = compute_frequency_domain(
        rr_clean, rr_times_clean
    )

    # ── Complexity / criticality ──────────────────────────────
    if verbose:
        print("  Complexity metrics (SampEn, ApEn, PermEn, MSE, DFA, RQA)...")
    complexity_metrics, mse_scales, mse_values = compute_complexity_metrics(
        rr_clean
    )

    # ── MSE curve PNG ─────────────────────────────────────────
    if plot:
        _plot_mse_curve(record_id, mse_scales, mse_values, output_dir)

    # ── Assemble result dict ──────────────────────────────────
    result = {"Record_ID": record_id}

    result["Num_RR"]           = num_rr
    result["Artifact_Percent"] = artifact_pct
    result["QC_RR_Count_OK"]   = num_rr >= 240
    result["QC_Artifact_OK"]   = artifact_pct < 5.0

    result["HRV_MeanNN"] = hrv_meannn
    result["HRV_SDNN"]   = hrv_sdnn
    result["HRV_RMSSD"]  = hrv_rmssd
    result["HRV_pNN50"]  = hrv_pnn50
    result["HRV_MeanHR"] = hrv_meanhr
    result["HRV_SD1"]    = sd1
    result["HRV_SD2"]    = sd2
    result["HRV_SD1SD2"] = sd1sd2

    result.update(freq_metrics)
    result.update(complexity_metrics)

    result["QC_MSE_Valid"]    = np.isfinite(result.get("MSE_Mean",   np.nan))
    result["QC_DFA1_Valid"]   = np.isfinite(result.get("DFA_Alpha1", np.nan))
    result["QC_SampEn_Valid"] = np.isfinite(result.get("SampEn",     np.nan))
    result["QC_Overall_Pass"] = (
        result["QC_RR_Count_OK"]
        and result["QC_Artifact_OK"]
        and result["QC_DFA1_Valid"]
    )

    # ── IAAFT surrogate testing ───────────────────────────────
    if run_surrogates:
        surr = run_surrogate_testing(
            rr_clean, record_id, output_dir, verbose=verbose
        )
        result.update(surr)

    # ── Diagnostic plot ───────────────────────────────────────
    if plot:
        _plot_rr_diagnostic(
            record_id,
            rr_times_raw, rr_raw,
            rr_times_clean, rr_clean,
            freqs, power, complexity_metrics,
            psd_beta=freq_metrics.get("PSD_Beta", np.nan),
            output_dir=output_dir,
        )

    if verbose:
        print(f"\n  ✓ Done: {record_id}")

    return result


# ═══════════════════════════════════════════════════════════════
# STEP 4 — MULTI-RECORD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_psychiatric_analysis(filepaths, plot=True,
                              run_surrogates=True,
                              save_csv=True,
                              output_dir=OUTPUT_DIR):
    """
    Run the full pipeline across all psychiatric CSV subjects.

    Produces:
      • Per-subject diagnostic PNGs, MSE curves, surrogate test PNGs
      • QA summary
      • Cohort statistics
      • Correlation matrices (full + R²-filtered)
      • Radar charts z-scored on the HEALTHY cohort reference frame
      • ecg_results_Psychiatric.csv with cohort stats appended
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for fp in filepaths:
        try:
            result = analyze_psychiatric_record(
                fp, plot=plot,
                run_surrogates=run_surrogates,
                output_dir=output_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ Failed on {fp}: {e}")
            all_results.append({
                "Record_ID": os.path.splitext(os.path.basename(fp))[0]
            })

    df = pd.DataFrame(all_results).set_index("Record_ID")

    # ── Results table ─────────────────────────────────────────
    print("\n" + "="*60 + "\n  PSYCHIATRIC RESULTS TABLE\n" + "="*60)
    with pd.option_context('display.max_columns', None,
                           'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(df.T.to_string())

    # ── QA summary ────────────────────────────────────────────
    qa_cols = ["Num_RR", "Artifact_Percent", "QC_RR_Count_OK",
               "QC_Artifact_OK", "QC_SampEn_Valid", "QC_MSE_Valid",
               "QC_DFA1_Valid", "QC_Overall_Pass"]
    qa_present = [c for c in qa_cols if c in df.columns]
    qa_df = df[qa_present].copy()
    for col in [c for c in qa_present if c.startswith("QC_")]:
        qa_df[col] = qa_df[col].map({True: "PASS", False: "FAIL"})
    print("\n" + "="*60 + "\n  QA SUMMARY\n" + "="*60)
    print(qa_df.to_string())
    if "QC_Overall_Pass" in df.columns:
        n_pass = df["QC_Overall_Pass"].sum()
        n_tot  = len(df)
        print(f"\n  Overall QC: {n_pass}/{n_tot} records passed")
        failed = df.index[~df["QC_Overall_Pass"]].tolist()
        print(f"  Failed: {failed}" if failed else "  All records cleared QC.")

    # ── Cohort stats ──────────────────────────────────────────
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = [c for c in df.columns if c.startswith("QC_")]
    stat_cols = [c for c in num_cols if c not in bool_cols]
    stats_df  = pd.DataFrame(
        [df[stat_cols].mean(),
         df[stat_cols].median(),
         df[stat_cols].std(ddof=1)],
        index=["Psychiatric_Mean", "Psychiatric_Median", "Psychiatric_SD"]
    )
    print("\n" + "="*60 + "\n  PSYCHIATRIC COHORT SUMMARY\n" + "="*60)
    with pd.option_context('display.max_columns', None,
                           'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(stats_df.T.to_string())

    # ── PSD beta fit quality ──────────────────────────────────
    if "PSD_Beta" in df.columns and "PSD_Beta_R2" in df.columns:
        print("\n" + "="*60 + "\n  PSD BETA FIT QUALITY\n" + "="*60)
        print(df[["PSD_Beta", "PSD_Beta_R2"]]
              .sort_values("PSD_Beta_R2").to_string())
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df["PSD_Beta"], df["PSD_Beta_R2"], s=60)
        for idx, row in df.iterrows():
            ax.annotate(idx, (row["PSD_Beta"], row["PSD_Beta_R2"]),
                        fontsize=7)
        ax.axhline(PSD_BETA_R2_THRESHOLD, color='red',
                   linestyle='--', lw=1.2,
                   label=f'R²={PSD_BETA_R2_THRESHOLD} threshold')
        ax.axvline(1.0, color='grey', linestyle='--', lw=1.0)
        ax.set_xlabel("PSD Beta"); ax.set_ylabel("PSD Beta R²")
        ax.set_title("PSD Scaling Exponent vs Fit Quality — Psychiatric")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "PSD_Beta_vs_R2_Psychiatric.png"),
            dpi=150
        )
        plt.close()

    # ── Correlation matrices ──────────────────────────────────
    plot_correlation_matrices(df, output_dir)

    # ── Radar charts (z-scored on HEALTHY reference) ─────────
    plot_radar_charts(
        df, output_dir,
        cohort_label="Psychiatric",
        cohort_m=HEALTHY_MEAN,
        cohort_s=HEALTHY_SD,
    )

    # ── CSV ───────────────────────────────────────────────────
    if save_csv:
        csv_path  = os.path.join(output_dir, "ecg_results_Psychiatric.csv")
        df_export = pd.concat([df, stats_df])
        df_export.to_csv(csv_path)
        print(f"\n  Results saved  → {csv_path}")

    return df


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    filepaths = [
        os.path.join(BASE, f"treatment_{n}.csv")
        for n in TREATMENT_IDS
    ]

    # Verify all files exist before starting the (slow) run
    missing = [fp for fp in filepaths if not os.path.exists(fp)]
    if missing:
        print("WARNING — the following files were not found:")
        for m in missing:
            print(f"  {m}")
        print("Proceeding with files that do exist.\n")
        filepaths = [fp for fp in filepaths if os.path.exists(fp)]

    df_psychiatric = run_psychiatric_analysis(
        filepaths,
        plot=True,
        run_surrogates=True,    # set False for a fast dry-run
        save_csv=True,
        output_dir=OUTPUT_DIR,
    )

    # ── Comparison radar chart (healthy vs psychiatric) ───────
    # To generate the overlay figure after both cohorts have been run,
    # load the healthy results CSV and call plot_radar_comparison().
    # Example:
    #
    #   import pandas as pd
    #   from pipeline import plot_radar_comparison, HEALTHY_MEAN, HEALTHY_SD
    #
    #   df_h = pd.read_csv("outputs/healthy/ecg_results_Healthy.csv",
    #                      index_col=0).iloc[:-3]   # drop summary rows
    #   plot_radar_comparison(
    #       df_h, df_psychiatric,
    #       output_dir=OUTPUT_DIR,
    #       cohort_m=HEALTHY_MEAN,
    #       cohort_s=HEALTHY_SD,
    #   )
    #
    # Or run comparison.py which will do this automatically.