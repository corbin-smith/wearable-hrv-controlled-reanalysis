# Changelog

## 2.1.0 (November 2026)

Changed:

- Covariate-adjusted comparisons use a partial Spearman correlation that ranks first, then regresses. This replaces the earlier form, which regressed first.
- `HR_Adjusted_Comparison.csv` and `Beta_Activity_Covariate.csv` have new columns for a linear model on log values.
- `Classification_Summary_NestedCV.csv` gives a bootstrap interval for each model's AUC.
- `exhibits_v12.py` writes six tables instead of five.

Added:

- `requantization_beat_time.py`: 128 Hz experiment with beat times rounded.
- `recording_conditions.py`: movement by minute and alternative windows.
- `estimator_changes.py`: effect of estimator settings, with bootstrap intervals.

Removed:

- The `manuscript/` folder.
- Comments and docstrings in the scripts. The code is unchanged.

Per-subject results are unchanged.

## 2.0.1 (November 2026)

- Added `outputs/accelerometer/`, missing from 2.0.0.

## 2.0.0 (November 2026)

- Code and results for the manuscript.
- Repository renamed from `ecg-criticality-psychiatry-pilot`.
- `outputs/` replaces `results/`.

## 1.0.1 (August 2026)

- Metadata corrections.

## 1.0.0 (August 2026)

- First release.
