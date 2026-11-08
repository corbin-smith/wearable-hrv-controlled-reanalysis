# Changelog

## 2.0.1 (November 2026)

Adds `outputs/accelerometer/` (mobility, activity-structure and covariate tables, and the movement-free beat masks), which version 2.0.0 left out by mistake: an unanchored `.gitignore` pattern for the raw accelerometer folder also matched it on a case-insensitive file system. The patterns are now anchored to the top folder. No code or result changed.

## 2.0.0 (November 2026)

Code and results for manuscript v12, *Heart rate or complexity? A controlled reanalysis of wearable heart rate variability in medicated inpatients with schizophrenia or bipolar disorder*. The repository was renamed from `ecg-criticality-psychiatry-pilot` to `wearable-hrv-controlled-reanalysis` to match the study's framing.

### Analysis

- **Quantization step.** Resolution controls use the measured 1.000 ms step of the exported HRV-ACC intervals (the nominal Bluetooth unit is 1/1024 s). The matched-coarsening target is 0.088.
- **Heart-rate-adjusted family.** Every measure is tested with mean heart rate held constant, under four alternative model forms and with beat-count adjustment. Bootstrap intervals for the adjusted correlations are in `hr_adjusted_bootstrap.py`.
- **Resolution.** Every measure is recomputed on the 7.8 ms grid of a 128 Hz ECG (columns ending `_Q128`).
- **Spectral redistribution.** VLF power, the VLF/LF ratio and band shares are reported and carried through every covariate, subgroup, movement and onset-skip check.
- **Subject-level metadata.** Age, sex, diagnosis and quetiapine come from HRV-ACC version 2 and are used in the subgroup, medication, comorbidity and age and sex analyses.
- **Time of day and transmission gaps.** Measured with `recording_times_and_gaps.py`.
- **Main tables and figures.** Built from the results files by `exhibits_v12.py` and `figures.py`.
- **Consistency checks.** `validate_outputs.py` checks the outputs against each other.

### Repository

- **Results.** `outputs/` now holds the full results of the manuscript run: per-subject results and every statistical table. It replaces the `results/` folder of version 1.
- **Builders.** `manuscript/` holds the builders for the manuscript and the supplement.
- **Comments.** Comments and docstrings in the analysis scripts were condensed. The code itself is unchanged from the run that produced `outputs/`, which was checked by comparing each script's syntax tree before and after.
- **Data and citation files.** `requirements.txt` adds statsmodels. `DATA_SETUP.md` and `download_data.py` point to HRV-ACC version 2, including its metadata files. `CITATION.cff` has the new title.

## 1.0.1 (4 August 2026)

Repository metadata corrections.

## 1.0.0 (4 August 2026)

First release, for the pilot analysis.
