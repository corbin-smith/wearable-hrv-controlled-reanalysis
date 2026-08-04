# ecg-criticality-psychiatry-pilot

Analysis code for a pilot study of cardiac autonomic complexity in treated psychiatric inpatients.

The study compares 30 healthy controls against 30 hospitalized psychiatric patients using R-R interval data from the HRV-ACC dataset (Książek et al., 2023). Both groups were recorded on the same Polar H10 hardware under the same protocol, which removes cross-device confounds by design. A second control group from the MIT-BIH Normal Sinus Rhythm database is included as a supplementary cross-device check.

The primary finding is an elevated power-spectral scaling exponent (PSD β) in the patient group.

**Status:** manuscript in preparation. Preprint and DOI links will be added here on posting.

## Requirements

Python 3.14.5. Install dependencies with:

```
pip install -r requirements.txt
```

Main libraries: NumPy, SciPy, pandas, matplotlib, NeuroKit2, WFDB.

## Data

Raw recordings are not included in this repository. Both datasets are public and must be downloaded separately.

See `DATA_SETUP.md` for instructions, or run:

```
python download_data.py
```

Expected folder layout after download:

```
Control ECGs/           control_N.csv       (HRV-ACC controls)
Psychiatric ECGs/       treatment_N.csv     (HRV-ACC patients)
Accelerometer ECGs/     control_N.csv, treatment_N.csv
```

MIT-BIH records are fetched by `healthy.py` through the WFDB package and need no manual setup.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Shared analysis functions. Artifact rejection, spectral analysis, complexity metrics, surrogate testing. Imported by the cohort scripts. |
| `healthy_hrvacc.py` | Runs the pipeline on the 30 HRV-ACC controls. |
| `psych_hrvacc.py` | Runs the pipeline on the 30 HRV-ACC patients. Mirrors `healthy_hrvacc.py` exactly. |
| `healthy.py` | Runs the pipeline on the MIT-BIH control group. Supplementary. |
| `comparison.py` | Group statistics, effect sizes, FDR correction, sensitivity analyses. |
| `accelerometer_mobility.py` | Checks whether physical activity explains the group difference in PSD β. |
| `nested_classifier.py` | Nested leave-one-out cross-validation classifier with feature selection inside the loop. |
| `figures.py` | Generates all manuscript figures. |
| `download_data.py` | Downloads the public datasets. |

## Running the analysis

Run from the project root, in this order:

```
python download_data.py
python healthy_hrvacc.py
python psych_hrvacc.py
python healthy.py              # optional, supplementary control arm
python comparison.py
python accelerometer_mobility.py
python nested_classifier.py
python figures.py
```

The cohort scripts must run before `comparison.py`, which reads their per-subject CSVs. Everything downstream reads from `comparison.py` output.

Surrogate testing is the slowest step. It runs 100 IAAFT surrogates per subject at 1000 iterations each, and can take several hours across both cohorts. Set `run_surrogates=False` in the cohort scripts to skip it during development.

## Output

Results are written to `outputs/`:

```
outputs/healthy_hrvacc/    per-subject metrics, HRV-ACC controls
outputs/psychiatric/       per-subject metrics, HRV-ACC patients
outputs/healthy/           per-subject metrics, MIT-BIH controls
outputs/comparison/        group statistics and sensitivity tables
outputs/accelerometer/     mobility analysis
outputs/figures/           manuscript figures
```

The `results/` folder in this repository holds the summary tables reported in the manuscript, so the numbers can be checked without re-running the pipeline.

## Analysis window

All cohorts use minutes 10 to 40 of each recording. The first 10 minutes are skipped because heart rate is still elevated from the initial ambulation. The resulting window contains exactly one instructed walk for every subject, which keeps the activity protocol matched across groups.

This window is set in `pipeline.py` and must stay identical across `healthy_hrvacc.py` and `psych_hrvacc.py`. Changing it for one cohort and not the other breaks the comparison.

## Notes on interpretation

Every patient in this dataset was medicated and hospitalized. The findings describe treated inpatients, and the contribution of medication cannot be separated from the contribution of illness.

Subject-level age, sex, diagnosis, and medication data are not available in the public release, so these variables could not be used as covariates.

Many of the 23 metrics are algebraically related. The significant results collapse to roughly five independent dimensions.

The classifier result in `results/Classification_Summary_NestedCV.csv` uses feature selection inside each cross-validation fold. Earlier versions of this analysis selected features on the full sample, which inflates performance, and those results are not reported here.

## Citation

If you use this code, please cite the software archive:

> Smith, C. (2026). ecg-criticality-psychiatry-pilot (v1.0.0). Zenodo. [DOI pending]

Please also cite the source dataset:

> Książek, W., et al. (2023). HRV-ACC: a dataset with R-R intervals and accelerometer data for the diagnosis of psychotic disorders. Zenodo. https://doi.org/10.5281/zenodo.8171266

Machine-readable citation metadata is in `CITATION.cff`.

## License

MIT. See `LICENSE`.

## Contact

Corbin Smith
ORCID: [0009-0009-1721-8640](https://orcid.org/0009-0009-1721-8640)

Email: corbinedwardsmith@gmail.com
