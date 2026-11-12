"""psych_hrvacc.py: run the pipeline on the 30 HRV-ACC patients (Polar H10 R-R intervals)."""

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

from patient_metadata import attach_metadata, parse_metadata_file
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
    VLF_BAND, LF_BAND, HF_BAND, BETA_FIT_BAND,
    INTERP_FS, PSD_BETA_R2_THRESHOLD,
    N_SURROGATES, SURROGATE_ALPHA,
    RADAR_METRICS, RADAR_LABELS,
)

BASE       = "Psychiatric ECGs"
OUTPUT_DIR = os.path.join("outputs", "psychiatric")

TREATMENT_IDS = [
    1, 2, 3, 4, 7, 8, 9, 13, 15, 16, 17, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 29, 31, 32, 33, 36, 37, 38, 40, 41, 42,
]

SKIP_SECONDS     = PIPELINE_SKIP_SECONDS
ANALYSIS_SECONDS = PIPELINE_ANALYSIS_SECONDS

HEALTHY_CSV_FOR_REFERENCE = os.path.join(
    "outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv")


def load_control_reference(path=HEALTHY_CSV_FOR_REFERENCE,
                           metrics=RADAR_METRICS):
    if not os.path.exists(path):
        print(f"  ! Control reference not found at {path}. "
              f"Run healthy_hrvacc.py first. Falling back to "
              f"self-normalisation for the radar chart only "
              f"(no other metric is affected).")
        return None, None
    try:
        ref = pd.read_csv(path)
        idx = "Record_ID" if "Record_ID" in ref.columns else ref.columns[0]
        ref = ref.set_index(idx)
        if "QC_Overall_Pass" in ref.columns:
            ref = ref[ref["QC_Overall_Pass"] == True]        # noqa: E712
        cols = [m for m in metrics if m in ref.columns]
        if not cols:
            print("  ! Control reference file has none of the radar metrics.")
            return None, None
        sub = ref[cols].apply(pd.to_numeric, errors="coerce")
        mean = sub.mean()
        sd = sub.std(ddof=1).replace(0, 1)
        print(f"  Control reference loaded from {path} "
              f"(n={len(sub)}, metrics={', '.join(cols)})")
        return mean, sd
    except Exception as exc:
        print(f"  ! Could not read control reference ({exc}). "
              f"Falling back to self-normalisation.")
        return None, None

HEALTHY_MEAN, HEALTHY_SD = None, None


def analyze_psychiatric_record(filepath, plot=True,
                               run_surrogates=True,
                               verbose=True,
                               output_dir=OUTPUT_DIR):
    return analyze_hrvacc_record(
        filepath, output_dir=output_dir, plot=plot,
        run_surrogates=run_surrogates, verbose=verbose,
        cohort_title="Psychiatric", clean_colour="steelblue")


def run_psychiatric_analysis(filepaths, plot=True,
                              run_surrogates=True,
                              save_csv=True,
                              output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    _write_environment_once(output_dir)

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

    print("\n" + "="*60 + "\n  PSYCHIATRIC RESULTS TABLE\n" + "="*60)
    with pd.option_context('display.max_columns', None,
                           'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(df.T.to_string())

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

    plot_correlation_matrices(df, output_dir)

    ref_m, ref_s = load_control_reference()
    if ref_m is None:
        print("  NOTE: psychiatric radar chart is SELF-normalised, not scaled "
              "against controls. Do not compare it to the control radar.")
    plot_radar_charts(
        df, output_dir,
        cohort_label="Psychiatric",
        cohort_m=ref_m,
        cohort_s=ref_s,
    )

    try:
        df = attach_metadata(df, cohort="patient")
    except FileNotFoundError as exc:
        print(f"  NOTE: {exc}")
        print("  Continuing without subject-level metadata. Diagnosis, "
              "medication and age analyses will be skipped downstream.")
    except (NameError, AttributeError, TypeError):
        raise
    except Exception as exc:
        print(f"  WARNING: metadata could not be attached ({type(exc).__name__}"
              f": {exc}). Continuing without it.")

    if save_csv:
        csv_path  = os.path.join(output_dir, "ecg_results_Psychiatric.csv")
        df_export = pd.concat([df, stats_df])
        df_export.to_csv(csv_path)
        print(f"\n  Results saved  → {csv_path}")

    return df


if __name__ == "__main__":
    filepaths = [
        os.path.join(BASE, f"treatment_{n}.csv")
        for n in TREATMENT_IDS
    ]

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
        run_surrogates=True,
        save_csv=True,
        output_dir=OUTPUT_DIR,
    )
