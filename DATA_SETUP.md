# Data setup

The raw recordings are **not** included in this repository (they belong to the
original dataset authors). This guide tells you how to download them from their
public sources and where to put them so the scripts find them.

You can do the supplementary (MIT-BIH) part automatically with
`python download_data.py`. The primary (Zenodo) part is a large multi-file
archive, so it's done by hand — the steps are below.

---

## 1. Primary dataset — HRV-ACC (Książek et al., 2023)  *[required]*

**Source:** Zenodo, DOI [10.5281/zenodo.8171266](https://doi.org/10.5281/zenodo.8171266)

This dataset provides both the heartbeat (R-R interval) data and the
accelerometer (movement) data for both groups.

Download the archive(s) from the Zenodo page and organize the files into **three
folders** at the top level of the project, exactly like this:

```
Control ECGs/            # control group heartbeat files
    control_1.csv
    control_2.csv
    ...
Psychiatric ECGs/        # patient group heartbeat files
    treatment_1.csv
    treatment_2.csv
    ...
Accelerometer ECGs/      # accelerometer files for BOTH groups together
    control_1.csv
    treatment_1.csv
    ...
```

Notes:
- The accelerometer files come from the archive named
  `accelerometer_anonymized_data.zip` on Zenodo; unzip it and put all of its
  `control_N.csv` / `treatment_N.csv` files into a single `Accelerometer ECGs/`
  folder.
- The subject-numbering in the heartbeat files is **not** perfectly
  consecutive — some numbers are skipped. That's expected; the scripts only load
  the IDs that exist.
- If Zenodo's internal file or folder names differ from the target names above,
  just rename/move them to match this layout. The accelerometer script also
  recognizes a few alternate folder names (e.g. `accelerometer_anonymized_data`)
  so you may not need to rename that one.

---

## 2. Supplementary dataset — MIT-BIH Normal Sinus Rhythm  *[optional]*

**Source:** PhysioNet, database `nsrdb` (<https://physionet.org/content/nsrdb/>)

This is a second, healthy-only set recorded on *different* hardware, used only as
a cross-device sanity check. The easiest way to get it is:

```bash
python download_data.py
```

That downloads the 18 records used here and places them in the layout the script
expects:

```
Long Healthy ECGs/
    16265/
        16265.dat
        16265.hea
    16272/
        16272.dat
        16272.hea
    ...
```

(If you prefer to do it by hand, download those 18 records from the PhysioNet
page and arrange them into one subfolder per record, named by the record ID, as
shown above.)

---

## Final check

Before running the pipeline, the top of your project folder should contain the
code files **plus** these data folders:

```
ecg-criticality-psychiatry-pilot/
├── (the .py scripts)
├── Control ECGs/            ← from Zenodo
├── Psychiatric ECGs/        ← from Zenodo
├── Accelerometer ECGs/      ← from Zenodo
└── Long Healthy ECGs/       ← from PhysioNet (optional)
```

These data folders are listed in `.gitignore`, so they will **not** be uploaded
if you push the repository to GitHub. That is intentional — never redistribute
the raw data; point people to the sources above instead.
