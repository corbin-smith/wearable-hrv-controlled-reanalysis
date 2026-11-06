# Heart rate or complexity? Analysis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21794970.svg)](https://doi.org/10.5281/zenodo.21794970)

Analysis code and results for the manuscript *Heart rate or complexity? A controlled reanalysis of wearable heart rate variability in medicated inpatients with schizophrenia or bipolar disorder* (Corbin Smith; preprint link to be added).

The study reanalyzes the open HRV-ACC dataset: R-R intervals recorded with a Polar H10 chest strap from 30 medicated psychiatric inpatients (23 with schizophrenia, 7 with bipolar disorder) and 30 controls, with the 18 MIT-BIH Normal Sinus Rhythm records as an external control cohort. Twenty-one HRV measures are compared between groups and then tested against three things:

- mean heart rate;
- recording resolution;
- estimator settings.

Movement, age, sex, diagnosis and the one recorded medication are also tested.

Every number, table and figure in the manuscript and its supplement can be regenerated from this repository and the two public datasets.

## Contents

| Folder or file | What it holds |
|---|---|
| `pipeline.py` | Shared analysis functions: artifact rejection, spectra and slope estimators, entropy, DFA, RQA, MFDFA, recurrence networks, surrogates, resolution controls. Imported by the other scripts. |
| `healthy_hrvacc.py`, `psych_hrvacc.py`, `healthy.py` | Run the pipeline on the HRV-ACC controls, the HRV-ACC patients and the MIT-BIH records. |
| `patient_metadata.py`, `mitbih_metadata.py` | Subject-level age, sex, diagnosis and quetiapine (HRV-ACC version 2 metadata); MIT-BIH age and sex. |
| `rna_controls.py`, `accelerometer_mobility.py`, `merge_mobility.py` | Recurrence-network resolution controls; accelerometer mobility, walking bouts and movement-free masks. |
| `comparison.py` | Group statistics, correction families, heart-rate adjustment, resolution, estimator, surrogate, covariate, subgroup and medication analyses. |
| `nested_classifier.py` | Classification with feature selection inside each fold; heart rate and RMSSD baseline. |
| `length_vs_heartrate.py`, `stationary_reanalysis.py`, `window_skip_sensitivity.py`, `mayer_residual_check.py` | Sensitivity analyses: series length, movement-free segments, onset skip, baroreflex band. |
| `hr_adjusted_bootstrap.py`, `recording_times_and_gaps.py` | Bootstrap intervals for the heart-rate-adjusted correlations; time of day and Bluetooth transmission gaps. |
| `figures.py`, `exhibits_v12.py` | Figures and the main tables. |
| `validate_outputs.py` | Internal-consistency checks across all outputs. |
| `download_data.py`, `DATA_SETUP.md` | Getting the data. |
| `outputs/` | The results used in the manuscript (see below). |
| `manuscript/` | The builders that produce the manuscript and supplement from `outputs/` (see `manuscript/README.md`). |

## Requirements

Python 3.14.5 and the pinned packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The results were produced on Windows 11 with the versions recorded in `outputs/*/Environment.csv`. Recurrence measures can differ in the third or fourth decimal under a different NumPy version, because tied distances are broken in a different order.

On Windows, set `PYTHONUTF8=1` before running, so that console output with Greek letters can be written to a log.

## Data

The raw recordings are not included. Both datasets are public:

| Dataset | Source | License |
|---|---|---|
| HRV-ACC, version 2 | doi:[10.5281/zenodo.21886112](https://doi.org/10.5281/zenodo.21886112) | CC BY 4.0 |
| MIT-BIH Normal Sinus Rhythm Database | doi:[10.13026/C2NK5R](https://doi.org/10.13026/C2NK5R) | ODC-By 1.0 |

Run `python download_data.py` to fetch the MIT-BIH records. Then follow the printed steps, also in `DATA_SETUP.md`, to place the HRV-ACC files.

## Reproducing the results

Run from the repository folder, in this order. Each script reads what the earlier ones wrote.

1. **Cohorts.** Most of the run time is in this step: about 6 to 8 hours in total on a laptop, mostly IAAFT surrogate testing.

   ```
   python healthy_hrvacc.py
   python psych_hrvacc.py
   python healthy.py
   ```

   `psych_hrvacc.py` reads the control results for its radar charts, so run `healthy_hrvacc.py` first. `healthy.py` can run alongside them.

2. **Metadata, controls and the comparison.** About 1 to 2 hours. Several of these add columns to the per-subject results files, and each one keeps a `*_pre_*.csv` copy of the file it changed.

   ```
   python mitbih_metadata.py --write
   python rna_controls.py --run
   python accelerometer_mobility.py
   python merge_mobility.py
   python comparison.py
   python nested_classifier.py
   ```

3. **Sensitivity analyses.** About 1 hour. These only read the results files.

   ```
   python length_vs_heartrate.py --run
   python stationary_reanalysis.py --assess
   python stationary_reanalysis.py --run --truncate 600 --match beats
   python stationary_reanalysis.py --run --truncate 370 --match duration
   python window_skip_sensitivity.py --run
   python mayer_residual_check.py
   python pipeline.py --validate-estimators
   ```

4. **Tables, figures and checks.**

   ```
   python exhibits_v12.py --selftest
   python exhibits_v12.py
   python figures.py
   python hr_adjusted_bootstrap.py --run
   python recording_times_and_gaps.py
   python validate_outputs.py
   ```

   `exhibits_v12.py --selftest` must print only `[ok]` lines. `validate_outputs.py` must end with `FAIL 0`.

`hr_adjusted_bootstrap.py` and `recording_times_and_gaps.py` each first check that they reproduce the comparison tables, and stop if they do not.

## Outputs

`outputs/` holds the results of the run behind the manuscript:

| Folder | Contents |
|---|---|
| `healthy_hrvacc/`, `psychiatric/`, `healthy/` | Per-subject results (`ecg_results_*.csv`) and the environment record. |
| `comparison/healthy_hrvacc_vs_psych/` | Primary comparison: every statistical table. |
| `comparison/healthy_mitbih_vs_psych/` | The same tables against the MIT-BIH controls. |
| `comparison/cross_control/` | The two control cohorts compared. |
| `accelerometer/`, `stationary/`, `classification/`, `validation/` | Movement analyses, movement-free reanalysis, classifier, estimator validation. |
| `exhibits_v12/` | Main Tables 1 to 5, main Figures 1 and 3, and the plotted values. |
| `figures/` | Main Figures 2 (`Figure3_PSD_Curves`) and 4 (`Figure8_CrossValidation`), and the supplementary figures. |

Three kinds of file are left out because a run regenerates them:

- the per-subject diagnostic plots;
- the `*_pre_*.csv` copies;
- material for a conference talk.

Some file and column names keep older labels, for example `Criticality_Correlation_Matrix.png` and `PSD_Beta`. The slope column `PSD_Beta` is the uncorrected low-frequency spectral slope. `PSD_Beta_NoMayer` is the slope with the baroreflex band excised.

## Interpretation

- Every patient was medicated and hospitalized. The comparison describes treated inpatients against healthy volunteers, not either illness on its own.
- Nothing here is a diagnostic tool.
- Holding heart rate constant decomposes a group difference; it does not remove a confounder, because heart rate is plausibly a consequence of group membership.

## Citation

Please cite the archived version you used (all versions: doi:[10.5281/zenodo.21794970](https://doi.org/10.5281/zenodo.21794970)) and the datasets above. Citation metadata is in `CITATION.cff`, and GitHub shows it under "Cite this repository".

## License

The code is released under the MIT License (`LICENSE`).

The files in `outputs/` are derived from the HRV-ACC dataset (CC BY 4.0) and the MIT-BIH Normal Sinus Rhythm Database (ODC-By 1.0). Reuse of those files should credit the datasets as their licenses require.

`accelerometer_mobility.py` implements the mobility measure described by the HRV-ACC authors. It is an independent implementation and shares no code with theirs.

## Contact

Corbin Smith, ORCID [0009-0009-1721-8640](https://orcid.org/0009-0009-1721-8640), corbinedwardsmith@gmail.com
