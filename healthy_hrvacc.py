"""healthy_hrvacc.py: run the pipeline on the 30 HRV-ACC controls (Polar H10 R-R intervals).

The controls were recorded with the same device and protocol as the patients and are processed by
the same code: pipeline.analyze_hrvacc_record(), with the window constants from pipeline.py. The
only differences from psych_hrvacc.py are the subject list, input and output folders, and the radar
charts, which this cohort self-normalizes because it is the reference frame for the patients.

Usage: python healthy_hrvacc.py
Writes outputs/healthy_hrvacc/, including ecg_results_HealthyACC.csv (per-subject results with cohort
summary rows appended).
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

# Shared pipeline functions, as in psych_hrvacc.py.
from patient_metadata import attach_metadata
from pipeline import (
    analyze_hrvacc_record,
    _write_environment_once,
    load_rr_window,
    SKIP_SECONDS as PIPELINE_SKIP_SECONDS,
    ANALYSIS_SECONDS as PIPELINE_ANALYSIS_SECONDS,
    reject_ectopic_beats,
    truncate_series,
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


# ---- Configuration ----

BASE       = "Control ECGs"                       # folder holding control_N.csv
OUTPUT_DIR = os.path.join("outputs", "healthy_hrvacc")
COHORT     = "HealthyACC"                          # label used in filenames/printouts

# The 30 HRV-ACC controls; file for subject N is "Control ECGs/control_N.csv". IDs are not
# contiguous and can coincide with patient IDs (different people; the record prefix keeps them apart).
CONTROL_IDS = [
    2, 16, 18, 19, 20, 21, 22, 24, 25, 26,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]   # len == 30

# -- Recording window (identical for both cohorts, from pipeline.py) --
# Minutes 10 to 40 of each recording ([600 s, 2400 s]). Under the HRV-ACC protocol (rest, then walks
# encouraged at the start, at about 30 min and at about 1 h) the first walk falls before the window,
# the second inside it and the third after it. Each control's mean heart rate is printed below.
SKIP_SECONDS     = PIPELINE_SKIP_SECONDS
ANALYSIS_SECONDS = PIPELINE_ANALYSIS_SECONDS


# ---- Cohort demographics (documentation only; not used in any computation) ----
# Group-level values for the 30 controls from Table 1 of the HRV-ACC dataset paper (Ksiazek et al., 2023).
# Per-subject age and sex come from control_data_summary.csv via patient_metadata.py.
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

# Optional per-subject demographics, {control_id: {"age": int, "sex": "F"/"M"}}; if populated, an
# extra control_demographics.csv is written. Left empty: per-subject values come from the metadata file.
CONTROL_DEMOGRAPHICS = {
}


# ---- Step 1: per-record analysis ----

def analyze_healthy_acc_record(filepath, plot=True,
                               run_surrogates=True,
                               verbose=True,
                               output_dir=OUTPUT_DIR):
    """Analyze one HRV-ACC control record.

    A thin wrapper over pipeline.analyze_hrvacc_record(), the implementation shared with the patients.
    """
    return analyze_hrvacc_record(
        filepath, output_dir=output_dir, plot=plot,
        run_surrogates=run_surrogates, verbose=verbose,
        cohort_title="Healthy / HRV-ACC", clean_colour="seagreen")


def run_healthy_acc_analysis(filepaths, plot=True,
                             run_surrogates=True,
                             save_csv=True,
                             output_dir=OUTPUT_DIR):
    """Run the pipeline over all HRV-ACC control records.

    Writes per-subject diagnostic, MSE and surrogate plots, a QA summary, cohort statistics, correlation
    matrices, self-normalized radar charts and ecg_results_HealthyACC.csv.

    Returns
    -------
    df        : DataFrame, per-subject results
    cohort_m  : Series, radar-metric means (the reference frame read by psych_hrvacc.py)
    cohort_s  : Series, radar-metric SDs
    """
    os.makedirs(output_dir, exist_ok=True)
    # Library versions and random seeds, written beside the results.
    _write_environment_once(output_dir)

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

    # -- Results table --
    print("\n" + "="*60 + f"\n  {COHORT.upper()} RESULTS TABLE\n" + "="*60)
    with pd.option_context('display.max_columns', None,
                           'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(df.T.to_string())

    # -- QA summary --
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

    # -- Cohort stats --
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

    # -- PSD slope fit quality --
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

    # -- Correlation matrices --
    plot_correlation_matrices(df, output_dir)

    # -- Radar charts, self-normalized: this cohort defines the reference frame --
    cohort_m, cohort_s = plot_radar_charts(
        df, output_dir,
        cohort_label=COHORT,
        cohort_m=None,
        cohort_s=None,
    )

    # -- Optional per-subject demographics side table --
    # Written only if CONTROL_DEMOGRAPHICS is populated; kept out of the results CSV so the schema matches
    # the patient file.
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

    # -- Subject-level metadata --
    # Age and sex are merged into the results file. A missing metadata file is tolerated
    # (FileNotFoundError); NameError and AttributeError are not, since they mean the code is wrong.
    try:
        df = attach_metadata(df, cohort="control")
    except FileNotFoundError as exc:
        print(f"  NOTE: {exc}")
        print("  Continuing without control metadata; age- and sex-adjusted "
              "between-group analyses will be skipped downstream.")
    except (NameError, AttributeError, TypeError):
        raise
    except Exception as exc:
        print(f"  WARNING: control metadata could not be attached "
              f"({type(exc).__name__}: {exc}). Continuing without it.")

    # -- CSV --
    if save_csv:
        csv_path  = os.path.join(output_dir, f"ecg_results_{COHORT}.csv")
        df_export = pd.concat([df, stats_df])
        df_export.to_csv(csv_path)
        print(f"\n  Results saved  → {csv_path}")

    return df, cohort_m, cohort_s


# ---- Entry point ----

if __name__ == "__main__":

    filepaths = [
        os.path.join(BASE, f"control_{n}.csv")
        for n in CONTROL_IDS
    ]

    # Check that all files exist before the run starts.
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

    # -- Reference frame --
    # psych_hrvacc.py reads this cohort's results file at run time; nothing needs copying.
    print("\n" + "=" * 60)
    print("  CONTROL REFERENCE FRAME")
    print("=" * 60)
    for metric in RADAR_METRICS:
        if metric in healthy_acc_m.index:
            print(f"    {metric:<14} mean {float(healthy_acc_m[metric]):10.5f}"
                  f"   SD {float(healthy_acc_s[metric]):10.5f}")
    print(f"\n  Written to {os.path.join(OUTPUT_DIR, 'ecg_results_HealthyACC.csv')}.")
    print("  psych_hrvacc.py reads this file directly — do NOT copy these")
    print("  numbers into any other script.")

    # -- Run-quality flags --
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT CHECKS FOR THE NEXT STAGE")
    print("=" * 60)
    try:
        subj = df_healthy_acc.loc[
            [i for i in df_healthy_acc.index
             if not str(i).startswith(("Healthy", "Cohort"))]]
        if "RNA_DegreeCheck_OK" in subj.columns:
            ok = subj["RNA_DegreeCheck_OK"].astype(str).str.lower().eq("true").sum()
            print(f"    RNA degree normalisation OK : {ok}/{len(subj)} "
                  f"(should be all)")
        if "Specparam_At_Bound" in subj.columns:
            nb = subj["Specparam_At_Bound"].astype(str).str.lower().eq("true").sum()
            print(f"    Peak model hit a bound      : {nb}/{len(subj)} "
                  f"(interpret PSD_Beta_Specparam with care if high)")
        if "Quant_Step_Over_RMSSD" in subj.columns:
            q = pd.to_numeric(subj["Quant_Step_Over_RMSSD"], errors="coerce")
            print(f"    Quant step / RMSSD (median) : {q.median():.4f}  "
                  f"(QUANT_TARGET_RATIO is defined from the patient mean "
                  f"RMSSD; see pipeline.py)")
        if "Stat_RA_Stationary" in subj.columns:
            st = subj["Stat_RA_Stationary"].astype(str).str.lower().eq("true").sum()
            print(f"    Stationary (reverse-arr.)   : {st}/{len(subj)}")
        if "PSD_Fit_LowestFreq_Hz" in subj.columns:
            lo = pd.to_numeric(subj["PSD_Fit_LowestFreq_Hz"], errors="coerce")
            print(f"    ACTUAL beta fit band starts  : {lo.median():.5f} Hz "
                  f"(nominal 0.01 — report the actual value)")
    except Exception as exc:
        print(f"    (pre-flight summary unavailable: {exc})")