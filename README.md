# Heart rate or complexity? Analysis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21794970.svg)](https://doi.org/10.5281/zenodo.21794970)

Code and results for the manuscript *Heart rate or complexity? A controlled reanalysis of wearable heart rate variability in medicated inpatients with schizophrenia or bipolar disorder* (Corbin Smith).

Data: the HRV-ACC dataset (30 inpatients, 30 controls, Polar H10) and 18 MIT-BIH Normal Sinus Rhythm records.

## Scripts

| Script | Purpose |
|---|---|
| `pipeline.py` | Shared analysis functions. |
| `healthy_hrvacc.py`, `psych_hrvacc.py`, `healthy.py` | Per-subject results for controls, patients and MIT-BIH. |
| `patient_metadata.py`, `mitbih_metadata.py` | Age, sex, diagnosis, medication. |
| `rna_controls.py` | Recurrence-network resolution controls. |
| `accelerometer_mobility.py`, `merge_mobility.py` | Movement measures. |
| `comparison.py` | Group statistics. |
| `nested_classifier.py` | Classifier. |
| `length_vs_heartrate.py`, `stationary_reanalysis.py`, `window_skip_sensitivity.py`, `mayer_residual_check.py` | Sensitivity analyses. |
| `hr_adjusted_bootstrap.py` | Bootstrap intervals for heart-rate adjustment. |
| `recording_times_and_gaps.py` | Time of day and transmission gaps. |
| `estimator_changes.py` | Effect of estimator settings. |
| `recording_conditions.py` | Movement by minute and alternative windows. |
| `requantization_beat_time.py` | 128 Hz resolution experiment. |
| `exhibits_v12.py`, `figures.py` | Tables and figures. |
| `validate_outputs.py` | Consistency checks. |
| `download_data.py` | Downloads the MIT-BIH records. |

## Setup

Python 3.14.5.

```
pip install -r requirements.txt
python download_data.py
```

Then place the HRV-ACC files as described in `DATA_SETUP.md`.

On Windows, set `PYTHONUTF8=1` before running.

## Run

Run from the repository folder, in this order.

1. Cohorts (6 to 8 hours):

   ```
   python healthy_hrvacc.py
   python psych_hrvacc.py
   python healthy.py
   ```

2. Metadata and group statistics (1 to 2 hours):

   ```
   python mitbih_metadata.py --write
   python rna_controls.py --run
   python accelerometer_mobility.py
   python merge_mobility.py
   python comparison.py
   python nested_classifier.py
   ```

3. Sensitivity analyses (1 hour):

   ```
   python length_vs_heartrate.py --run
   python stationary_reanalysis.py --assess
   python stationary_reanalysis.py --run --truncate 600 --match beats
   python stationary_reanalysis.py --run --truncate 370 --match duration
   python window_skip_sensitivity.py --run
   python mayer_residual_check.py
   python pipeline.py --validate-estimators
   ```

4. Tables, figures and checks (1 hour):

   ```
   python hr_adjusted_bootstrap.py --run
   python recording_times_and_gaps.py
   python estimator_changes.py
   python recording_conditions.py
   python requantization_beat_time.py --run
   python exhibits_v12.py --selftest
   python exhibits_v12.py
   python figures.py
   python validate_outputs.py
   ```

`validate_outputs.py` must end with `FAIL 0`.

## Outputs

| Folder | Contents |
|---|---|
| `outputs/healthy_hrvacc/`, `outputs/psychiatric/`, `outputs/healthy/` | Per-subject results. |
| `outputs/comparison/` | Statistical tables. |
| `outputs/accelerometer/`, `outputs/recording_conditions/`, `outputs/stationary/` | Movement analyses. |
| `outputs/classification/` | Classifier results. |
| `outputs/validation/` | Estimator validation. |
| `outputs/exhibits_v12/`, `outputs/figures/` | Tables and figures. |

Notes:

- Recurrence measures can differ slightly under another NumPy version.
- The column `PSD_Beta` is the low-frequency spectral slope. `PSD_Beta_NoMayer` is the same slope with 0.07 to 0.13 Hz removed.
- Per-subject plots and `*_pre_*.csv` backup files are not included. A run recreates them.

## Data sources

| Dataset | DOI | License |
|---|---|---|
| HRV-ACC, version 2 | [10.5281/zenodo.21886112](https://doi.org/10.5281/zenodo.21886112) | CC BY 4.0 |
| MIT-BIH Normal Sinus Rhythm Database | [10.13026/C2NK5R](https://doi.org/10.13026/C2NK5R) | ODC-By 1.0 |

The raw recordings are not included.

## Citation

Cite the archived version you used (all versions: doi:[10.5281/zenodo.21794970](https://doi.org/10.5281/zenodo.21794970)) and the two datasets. See `CITATION.cff`.

## License

Code: MIT (`LICENSE`). Files in `outputs/` are derived from the two datasets; credit them as their licenses require.

## Contact

Corbin Smith, ORCID [0009-0009-1721-8640](https://orcid.org/0009-0009-1721-8640), corbinedwardsmith@gmail.com
