# Data setup

The raw recordings are not included in this repository; they belong to the dataset authors. This guide says where to get them and where to put them so the scripts find them.

## 1. HRV-ACC, version 2 (required)

**Source:** Zenodo, doi:[10.5281/zenodo.21886112](https://doi.org/10.5281/zenodo.21886112) (CC BY 4.0). Version 2 adds the subject-level metadata files; its R-R and accelerometer files are the same as in version 1.

Download these four files:

| File | What it holds |
|---|---|
| `HRV_anonymized_data.zip` | R-R intervals for every subject |
| `accelerometer_anonymized_data.zip` | Accelerometer data for every subject |
| `patient_data_summary.csv` | Patients' ID, sex, age, ICD-10 codes and quetiapine status |
| `control_data_summary.csv` | Controls' ID, sex and age |

Arrange them at the top level of the repository folder like this:

```
Control ECGs/              control_N.csv     (R-R files of the controls)
Psychiatric ECGs/          treatment_N.csv   (R-R files of the patients)
Accelerometer ECGs/        control_N.csv and treatment_N.csv (all accelerometer files together)
patient_data_summary.csv
control_data_summary.csv
```

A few notes on the files:

- **Missing subject numbers.** The numbers are not consecutive, because some are missing. That is expected; the scripts list the IDs they use.
- **Accelerometer folder name.** `accelerometer_mobility.py` also accepts the folders `accelerometer_anonymized_data`, `Accelerometer` and `ACC ECGs`.
- **Metadata format.** The metadata files are semicolon-delimited. `patient_metadata.py` parses them, including ICD fields that hold more than one code (for example `F20,F19`).

## 2. MIT-BIH Normal Sinus Rhythm Database (external controls)

**Source:** PhysioNet, doi:[10.13026/C2NK5R](https://doi.org/10.13026/C2NK5R) (ODC-By 1.0).

```
python download_data.py
```

This downloads the 18 records into one folder per record:

```
Long Healthy ECGs/
    16265/16265.dat, 16265.hea
    16272/16272.dat, 16272.hea
    ...
```

To do it by hand instead, download the `.dat` and `.hea` file of each record from the PhysioNet page and arrange them as shown.

## Final layout

```
wearable-hrv-controlled-reanalysis/
    (the .py scripts)
    Control ECGs/
    Psychiatric ECGs/
    Accelerometer ECGs/
    Long Healthy ECGs/
    patient_data_summary.csv
    control_data_summary.csv
```

These data folders and files are listed in `.gitignore`, so they are not committed if you push a copy of the repository. Point others to the sources above rather than redistributing the raw data.
