"""healthy.py: run the pipeline on the 18 MIT-BIH Normal Sinus Rhythm records (external control cohort)."""

import os
from pipeline import run_analysis

BASE = "Long Healthy ECGs"

N_RECORDS_IN_DATABASE = 18
N_EXCLUDED            = 0

MITBIH_WINDOW_NOTE = (
    "Minutes 10 to 40 after record onset (600 s skip, 1800 s window, as for "
    "the HRV-ACC cohorts). Onset clock times are not "
    "standardised across the MIT-BIH NSR database, so analysed windows are "
    "not time-of-day matched to each other or to the midday HRV-ACC "
    "recordings. Report this alongside any cross-device claim."
)

RECORD_IDS = [
    "16265", "16272", "16273", "16420", "16483",
    "16539", "16773", "16786", "16795", "17052",
    "17453", "18177", "18184", "19088", "19090",
    "19093", "19140", "19830",
]

RECORD_PATHS = [
    os.path.join(BASE, rid, rid) for rid in RECORD_IDS
]

OUTPUT_DIR = os.path.join("outputs", "healthy")


if __name__ == "__main__":
    df_healthy, healthy_m, healthy_s = run_analysis(
        record_paths  = RECORD_PATHS,
        cohort_label  = "Healthy",
        lead_index    = 0,
        plot          = True,
        run_surrogates= True,
        save_csv      = True,
        output_dir    = OUTPUT_DIR,
    )

    print(f"\n  MIT-BIH NSR: {N_RECORDS_IN_DATABASE} records in the database, "
          f"{N_EXCLUDED} excluded, {len(df_healthy)} analysed.")
    if len(df_healthy) != N_RECORDS_IN_DATABASE - N_EXCLUDED:
        print("  ! Analysed count does not match the expected cohort size. "
              "Resolve before reporting.")
    print(f"  Window note: {MITBIH_WINDOW_NOTE}")

    if "Record_Start_Clock" in df_healthy.columns:
        clocks = df_healthy["Record_Start_Clock"].tolist()
        print(f"  Analysed-window start times: {sorted(set(clocks))}")
        if all(c == "unknown" for c in clocks):
            print("  ! No base time in the WFDB headers. State in the Methods "
                  "that time of day could not be determined for this cohort.")

    print("\n  Healthy cohort normalisation parameters (for comparison.py):")
    print("  healthy_m:\n", healthy_m.to_string())
    print("  healthy_s:\n", healthy_s.to_string())
