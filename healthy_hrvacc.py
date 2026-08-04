"""
healthy_hrvacc.py
=================
HRV-ACC Zenodo dataset — HEALTHY CONTROL cohort analysis.

This script is the control-group counterpart of psych_hrvacc.py. It exists
because the pilot study is moving its PRIMARY healthy comparison group from
the MIT-BIH Normal Sinus Rhythm database (analysed by healthy.py) to the
30 healthy controls recorded inside the *same* HRV-ACC dataset as the
psychiatric patients.

Why this matters scientifically
-------------------------------
The single largest weakness of the original pilot was that healthy subjects
(MIT-BIH, clinical Holter ECG) and patients (HRV-ACC, Polar H10 chest strap)
were recorded on different hardware, in different environments, under
different activity protocols. That made it impossible to separate "illness"
from "different recording conditions."

By analysing the HRV-ACC *controls* with byte-for-byte the same procedure as
the HRV-ACC *patients*, this script removes — by construction — the
cross-recording, environmental, and activity-protocol confounds. Same device,
same dataset, same acquisition protocol, same windowing, same artifact rules,
same metrics. Whatever difference survives is a difference between people, not
between machines or settings.

The MIT-BIH controls (healthy.py) are NOT discarded — they remain a secondary
control arm and will be re-incorporated later. If the same trends appear in
BOTH the matched HRV-ACC controls and the independent MIT-BIH controls, that is
considerably stronger evidence than either control group alone.

Relationship to the other scripts
----------------------------------
  • pipeline.py          — shared maths engine (imported, never modified here).
  • psych_hrvacc.py      — patient cohort; this file mirrors it almost exactly.
  • healthy.py           — MIT-BIH controls; secondary arm, set aside for now.
  • comparison.py        — group statistics; will be pointed at this cohort's
                           output CSV during the later integration step.

PARITY CONTRACT (read before editing)
-------------------------------------
Three things in this file MUST stay identical to psych_hrvacc.py, or the
confound-elimination argument above silently breaks:
  1. The CSV loader `load_rr_from_csv` (how raw beats become a clean array).
  2. The recording window: SKIP_SECONDS and ANALYSIS_SECONDS.
  3. The per-record metric computation in `analyze_healthy_acc_record`.
If you change any of these for the controls, you must change them for the
patients in the same commit, or the two cohorts are no longer comparable.
(The clean long-term fix is to lift the loader and per-record analysis into
pipeline.py so a single copy serves both cohorts. That refactor is deferred to
the later "re-incorporate healthy.py" phase, per project plan.)

The ONLY intended differences from psych_hrvacc.py
--------------------------------------------------
  • Which subjects are read (the 30 control IDs, not the 30 treatment IDs).
  • Where the files live  ("Control ECGs/", files named control_N.csv).
  • Cohort label / output folder / output CSV name.
  • Radar normalisation: this cohort SELF-normalises and PRINTS its own
    mean / SD, because the HRV-ACC controls are now the reference frame that
    the psychiatric radar charts should be measured against. (psych_hrvacc.py
    z-scores against an external healthy reference; this script DEFINES that
    reference.)

Outputs
-------
  • outputs/healthy_hrvacc/<id>_diagnostics.png   (per subject)
  • outputs/healthy_hrvacc/<id>_mse_curve.png     (per subject)
  • outputs/healthy_hrvacc/<id>_surrogate_*.png   (per subject, if enabled)
  • outputs/healthy_hrvacc/PSD_Beta_vs_R2_HealthyACC.png
  • outputs/healthy_hrvacc/ correlation + radar figures
  • outputs/healthy_hrvacc/ecg_results_HealthyACC.csv   (53-column schema,
                                cohort summary rows appended — matches the
                                psychiatric CSV schema exactly)
  • A printed, copy-paste-ready block of the five radar-metric mean/SD values
    to drop into psych_hrvacc.py and comparison.py as the new reference frame.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # non-interactive backend — safe for batch runs
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d
from scipy.signal import welch
from neurokit2.complexity import fractal_dfa
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ── Import shared pipeline functions ────────────────────────────
# Identical import block to psych_hrvacc.py so both cohorts demonstrably
# call the same maths. Do not swap any of these for local re-implementations.
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

BASE       = "Control ECGs"                       # folder holding control_N.csv
OUTPUT_DIR = os.path.join("outputs", "healthy_hrvacc")
COHORT     = "HealthyACC"                          # label used in filenames/printouts

# The 30 HRV-ACC control subjects (all of them). File for subject N is
# "Control ECGs/control_N.csv". These IDs are NOT contiguous — several numbers
# are missing — so they are listed explicitly rather than generated by range().
# (IDs may coincide numerically with treatment IDs, e.g. control_16 vs
#  treatment_16; they are different people in different folders, and the
#  Record_ID prefix keeps them distinct in all outputs.)
CONTROL_IDS = [
    2, 16, 18, 19, 20, 21, 22, 24, 25, 26,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]   # len == 30

# ── Recording window — PARITY ANCHOR ──────────────────────────
# These MUST equal the psych_hrvacc.py values. Both cohorts were recorded under
# the SAME HRV-ACC protocol (Książek et al.): an initial seated rest, then
# encouraged walks at the start, at ~30 min, and at ~1 h of recording.
#
# Window = [SKIP_SECONDS, SKIP_SECONDS + ANALYSIS_SECONDS]
#        = [600 s, 2400 s] = minutes 10–40 of each recording.
#
# Why this window keeps the activity protocol matched between groups:
#   • The start-of-recording walk (minute ~0)  falls BEFORE the window → skipped
#     for everyone.
#   • Exactly ONE instructed walk (the ~30-min one) falls INSIDE the window →
#     present for everyone, identically.
#   • The ~1-h walk (minute ~60) falls AFTER the window → excluded for everyone.
# So both cohorts carry the same protocol structure inside the analysed segment.
#
# The 600 s skip was chosen for the patients because shorter skips (60 s, 360 s)
# left mean HR elevated (~100 bpm) from initial ambulation. Controls receive the
# IDENTICAL skip on purpose — re-tuning it for controls alone would reintroduce
# the very processing asymmetry this redesign is meant to remove. We instead
# PRINT each control's mean HR (below) so the window can be sanity-checked
# without altering it.
SKIP_SECONDS     = 600    # discard first 10 min  (== psych_hrvacc.py)
ANALYSIS_SECONDS = 1800   # analyse next 30 min   (== psych_hrvacc.py)


# ═══════════════════════════════════════════════════════════════
# COHORT DEMOGRAPHICS  (documentation; from the HRV-ACC source paper)
# ═══════════════════════════════════════════════════════════════
# Group-level figures for the 30 controls, transcribed from Table 1 of
# Książek et al., "The analysis of heart rate variability and accelerometer
# mobility data ... using a wearable Polar H10 sensor" (medRxiv 2023;
# dataset: Zenodo 10.5281/zenodo.8171266).
#
# These are GROUP-level values, kept here for provenance and for the methods
# section of the manuscript. They are NOT used in any metric computation.
#
# Crucially, unlike the MIT-BIH controls, this dataset reports per-group age —
# enabling the age-matched / age-covariate analysis the pilot manuscript flagged
# as the single most important follow-up.
CONTROL_GROUP_DEMOGRAPHICS = {
    "n":            30,
    "sex_women":    17,
    "sex_men":      13,
    "age_mean":     43.20,
    "age_sd":       14.06,
    "age_min":      24,
    "age_max":      69,
    "source":       "Książek et al. 2023, HRV-ACC dataset (Zenodo 8171266)",
}

# OPTIONAL per-subject demographics.
# If the Zenodo metadata gives a per-control mapping (control_N → age, sex),
# paste it here as  {control_id: {"age": <int>, "sex": "F"/"M"}}.
# When populated, the run writes an extra control_demographics.csv keyed by
# Record_ID, ready to merge into the age-covariate analysis later. When left
# empty, the core analysis runs and produces the standard 53-column results CSV
# exactly as the psychiatric cohort does — the metrics never depend on this.
#
# NOTE: do NOT invent ages. Leave entries out unless they come from the dataset.
CONTROL_DEMOGRAPHICS = {
    # 2:  {"age": __, "sex": "_"},
    # 16: {"age": __, "sex": "_"},
    # ...
}


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CSV LOADER   (parity-critical: identical to psych_hrvacc.py)
# ═══════════════════════════════════════════════════════════════
# This function is intentionally a verbatim copy of load_rr_from_csv in
# psych_hrvacc.py. It is duplicated rather than imported so this script stays
# self-contained (matching the project's existing convention), but the two
# copies MUST remain identical. If you edit one, edit the other.

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

    mask        = (cum_time >= t_start) & (cum_time < t_end)
    rr_window   = rr_s[mask]
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
# Cosmetic only — does not affect any computed metric. Mirrors the psychiatric
# diagnostic figure; the only change is the cohort label in the title.

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
        f"ECG Criticality Analysis — {record_id} (Healthy / HRV-ACC)",
        fontsize=13
    )

    # 1) RR intervals — raw vs cleaned
    ax = axes[0, 0]
    ax.plot(rr_times_raw,   rr_raw   * 1000, lw=0.8,
            color='gray',   alpha=0.6, label='Raw')
    ax.plot(rr_times_clean, rr_clean * 1000, lw=0.9,
            color='seagreen', label='Cleaned')
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
# STEP 3 — FULL RECORD ANALYSIS (one control CSV subject)
# ═══════════════════════════════════════════════════════════════
# Parity-critical: every metric computation below is identical to
# analyze_psychiatric_record in psych_hrvacc.py. Only labels/titles differ.

def analyze_healthy_acc_record(filepath, plot=True,
                               run_surrogates=True,
                               verbose=True,
                               output_dir=OUTPUT_DIR):
    """
    End-to-end pipeline for one HRV-ACC control subject.

    Mirrors analyze_psychiatric_record() exactly so that controls and patients
    are processed identically. All heavy maths delegates to pipeline.py.

    Returns
    -------
    dict  flat dict of all metrics (matching the psychiatric column schema)
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
    # Computed directly from the cleaned RR array (no WFDB / R-peak detection,
    # because the input is already RR intervals). Formulas match nk.hrv_time()
    # and are byte-identical to the psychiatric cohort.
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
    # SD1 = short-term variability (RMSSD / sqrt(2));  SD2 = long-term
    sd1     = hrv_rmssd / np.sqrt(2)
    sd2     = float(np.sqrt(2 * hrv_sdnn**2 - sd1**2))
    sd1sd2  = sd1 / sd2 if sd2 > 0 else np.nan

    # ── Mean-HR sanity print (window-parity check, informational only) ──
    # Lets us confirm the matched 10–40 min window lands in a comparable
    # activity state for controls as it does for patients, WITHOUT changing
    # the window. Not stored; purely a console diagnostic.
    if verbose:
        print(f"  Window mean HR: {hrv_meanhr:.1f} bpm "
              f"(parity check — window NOT adjusted per cohort)")

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

    # ── Assemble result dict (same key order/schema as psychiatric) ──
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

def run_healthy_acc_analysis(filepaths, plot=True,
                             run_surrogates=True,
                             save_csv=True,
                             output_dir=OUTPUT_DIR):
    """
    Run the full pipeline across all HRV-ACC control subjects.

    Produces:
      • Per-subject diagnostic PNGs, MSE curves, surrogate test PNGs
      • QA summary
      • Cohort statistics
      • Correlation matrices (full + R²-filtered)
      • Radar charts (SELF-normalised — this cohort defines the reference frame)
      • ecg_results_HealthyACC.csv with cohort summary rows appended

    Returns
    -------
    df        : DataFrame  per-subject results
    cohort_m  : Series      five radar-metric means  (the NEW reference frame)
    cohort_s  : Series      five radar-metric SDs     (the NEW reference frame)

    cohort_m / cohort_s are the values to paste into psych_hrvacc.py and
    comparison.py so the psychiatric cohort is measured against THESE controls.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for fp in filepaths:
        try:
            result = analyze_healthy_acc_record(
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
    print("\n" + "="*60 + f"\n  {COHORT.upper()} RESULTS TABLE\n" + "="*60)
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
        index=[f"{COHORT}_Mean", f"{COHORT}_Median", f"{COHORT}_SD"]
    )
    print("\n" + "="*60 + f"\n  {COHORT.upper()} COHORT SUMMARY\n" + "="*60)
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
        ax.scatter(df["PSD_Beta"], df["PSD_Beta_R2"], s=60, color='seagreen')
        for idx, row in df.iterrows():
            ax.annotate(idx, (row["PSD_Beta"], row["PSD_Beta_R2"]),
                        fontsize=7)
        ax.axhline(PSD_BETA_R2_THRESHOLD, color='red',
                   linestyle='--', lw=1.2,
                   label=f'R²={PSD_BETA_R2_THRESHOLD} threshold')
        ax.axvline(1.0, color='grey', linestyle='--', lw=1.0)
        ax.set_xlabel("PSD Beta"); ax.set_ylabel("PSD Beta R²")
        ax.set_title(f"PSD Scaling Exponent vs Fit Quality — {COHORT}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"PSD_Beta_vs_R2_{COHORT}.png"),
            dpi=150
        )
        plt.close()

    # ── Correlation matrices ──────────────────────────────────
    plot_correlation_matrices(df, output_dir)

    # ── Radar charts — SELF-NORMALISED ────────────────────────
    # cohort_m / cohort_s are passed as None so plot_radar_charts computes them
    # from THIS cohort and returns them. The HRV-ACC controls are the new
    # reference frame, so they define (rather than borrow) the z-score baseline.
    cohort_m, cohort_s = plot_radar_charts(
        df, output_dir,
        cohort_label=COHORT,
        cohort_m=None,
        cohort_s=None,
    )

    # ── Optional per-subject demographics side-table ──────────
    # Written only if CONTROL_DEMOGRAPHICS has been populated from the dataset.
    # Kept OUT of the metrics CSV so the 53-column schema stays identical to the
    # psychiatric cohort (comparison.py expects matching schemas).
    if CONTROL_DEMOGRAPHICS:
        demo_rows = []
        for cid, info in CONTROL_DEMOGRAPHICS.items():
            demo_rows.append({
                "Record_ID": f"control_{cid}",
                "Age": info.get("age", np.nan),
                "Sex": info.get("sex", ""),
            })
        demo_df = pd.DataFrame(demo_rows).set_index("Record_ID")
        demo_path = os.path.join(output_dir, "control_demographics.csv")
        demo_df.to_csv(demo_path)
        print(f"\n  Demographics    → {demo_path}  "
              f"({len(demo_df)} subjects with per-subject age/sex)")

    # ── CSV ───────────────────────────────────────────────────
    if save_csv:
        csv_path  = os.path.join(output_dir, f"ecg_results_{COHORT}.csv")
        df_export = pd.concat([df, stats_df])
        df_export.to_csv(csv_path)
        print(f"\n  Results saved  → {csv_path}")

    return df, cohort_m, cohort_s


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    filepaths = [
        os.path.join(BASE, f"control_{n}.csv")
        for n in CONTROL_IDS
    ]

    # Verify all files exist before starting the (slow) run
    missing = [fp for fp in filepaths if not os.path.exists(fp)]
    if missing:
        print("WARNING — the following files were not found:")
        for m in missing:
            print(f"  {m}")
        print("Proceeding with files that do exist.\n")
        filepaths = [fp for fp in filepaths if os.path.exists(fp)]

    df_healthy_acc, healthy_acc_m, healthy_acc_s = run_healthy_acc_analysis(
        filepaths,
        plot=True,
        run_surrogates=True,    # set False for a fast dry-run
        save_csv=True,
        output_dir=OUTPUT_DIR,
    )

    # ── Print the new reference frame for downstream scripts ──────
    # The HRV-ACC controls now define the z-score baseline. Copy the block
    # below into psych_hrvacc.py (HEALTHY_MEAN / HEALTHY_SD) and into
    # comparison.py (same two Series), replacing the old MIT-BIH values, so the
    # psychiatric cohort is measured against THESE matched controls.
    print("\n" + "="*60)
    print("  NEW REFERENCE FRAME (HRV-ACC controls)")
    print("  Paste into psych_hrvacc.py and comparison.py:")
    print("="*60)

    def _emit(series, name):
        print(f"\n{name} = pd.Series({{")
        for metric in RADAR_METRICS:
            if metric in series.index:
                print(f'    "{metric}": {float(series[metric])!r},')
        print("})")

    _emit(healthy_acc_m, "HEALTHY_MEAN")
    _emit(healthy_acc_s, "HEALTHY_SD")

    print("\n  (These five metrics are the radar/normalisation set: "
          f"{', '.join(RADAR_METRICS)}.)")
    print("\n  Reminder: this cohort is the NEW primary control. healthy.py")
    print("  (MIT-BIH) stays as a secondary arm to be re-incorporated later.")