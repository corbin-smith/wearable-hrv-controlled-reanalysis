# Data setup

The raw recordings are not in this repository. Download them and place them in the repository folder.

## HRV-ACC, version 2

Source: [10.5281/zenodo.21886112](https://doi.org/10.5281/zenodo.21886112) (CC BY 4.0).

Download:

- `HRV_anonymized_data.zip`
- `accelerometer_anonymized_data.zip`
- `patient_data_summary.csv`
- `control_data_summary.csv`

Unzip and arrange:

```
Control ECGs/          control_N.csv      R-R files of controls
Psychiatric ECGs/      treatment_N.csv    R-R files of patients
Accelerometer ECGs/    all accelerometer files
patient_data_summary.csv
control_data_summary.csv
```

Subject numbers are not consecutive. That is expected.

## MIT-BIH Normal Sinus Rhythm Database

Source: [10.13026/C2NK5R](https://doi.org/10.13026/C2NK5R) (ODC-By 1.0).

```
python download_data.py
```

This creates:

```
Long Healthy ECGs/
    16265/16265.dat, 16265.hea
    ...
```

## Note

These folders and files are in `.gitignore`. Do not redistribute the raw data.
