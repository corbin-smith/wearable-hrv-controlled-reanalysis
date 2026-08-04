"""
healthy.py
==========
MIT-BIH Normal Sinus Rhythm cohort analysis.

Imports all analysis functions from pipeline.py — no analysis logic lives here.
This script only defines:
  • The data path (Long Healthy ECGs/)
  • The 18 record IDs (full MIT-BIH NSR database)
  • Run-time options (plot, surrogates, output folder)
  • The entry point call to run_analysis()

The returned (df, cohort_m, cohort_s) tuple should be saved or passed to
comparison.py when the psychiatric cohort is ready.
"""

import os
from pipeline import run_analysis

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE = "Long Healthy ECGs"         # renamed from "Long ECGs"

RECORD_IDS = [
    "16265", "16272", "16273", "16420", "16483",
    "16539", "16773", "16786", "16795", "17052",
    "17453", "18177", "18184", "19088", "19090",
    "19093", "19140", "19830",
]

RECORD_PATHS = [
    os.path.join(BASE, rid, rid) for rid in RECORD_IDS
]

# ─────────────────────────────────────────────────────────────
# OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join("outputs", "healthy")

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    df_healthy, healthy_m, healthy_s = run_analysis(
        record_paths  = RECORD_PATHS,
        cohort_label  = "Healthy",
        lead_index    = 0,          # Lead II (index 0) for MIT-BIH NSR
        plot          = True,       # six-panel diagnostics + MSE curve per record
        run_surrogates= True,       # IAAFT surrogate testing (set False for dry-run)
        save_csv      = True,
        output_dir    = OUTPUT_DIR,
    )

    # ── Print normalisation parameters ───────────────────────
    # healthy_m and healthy_s are the cohort-level mean and SD for the
    # five radar metrics. Pass these to psychiatric.py / comparison.py
    # so all groups are z-scored on the same healthy reference frame.
    print("\n  Healthy cohort normalisation parameters (for comparison.py):")
    print("  healthy_m:\n", healthy_m.to_string())
    print("  healthy_s:\n", healthy_s.to_string())