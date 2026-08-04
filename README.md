# Cardiac Autonomic Complexity in Treated Psychotic-Spectrum Disorders — Pilot Analysis Code

<!-- After you mint the Zenodo DOI, uncomment and fill in the badge below:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository holds the complete analysis code for a **self-directed computational pilot study** testing whether measures of *cardiac autonomic complexity* — the moment-to-moment structure in the timing of heartbeats — differ between healthy volunteers and **treated, hospitalized psychotic-spectrum inpatients**, using only publicly available data.

> **What this is, honestly.** This is a proof-of-concept pilot on public datasets, not a clinical tool and not a landmark result. Every subject in the patient group is already medicated and hospitalized, there is no symptom-level or longitudinal data, and the sample is small (30 vs. 30). The point of the work is to demonstrate that these methods can be applied rigorously and to see whether any signal is worth pursuing in a larger, mentored study — not to diagnose anyone. The code and write-up are deliberately conservative about what the numbers mean.

---

## What the study looks at, in plain language

Every heartbeat is separated from the next by a tiny, constantly-changing gap. If you write down all those gaps in order, you get a wiggly sequence of numbers. Decades of research show that the *pattern* of that wiggle — not just the average heart rate — carries information about how the nervous system is regulating the heart. This project measures that pattern in several complementary ways (spectral slope, fractal scaling, several kinds of "entropy," recurrence structure, and so on) and asks a single question: do those pattern-measures differ, on average, between the two groups?

A few design choices were made specifically to keep the comparison fair:

- **Same hardware for both groups.** Both cohorts were recorded on the same Polar H10 chest strap, so a difference between groups can't be blamed on different devices.
- **An activity check.** The recordings include a walk, and movement can distort some measures. `accelerometer_mobility.py` measures how much each person actually moved and tests whether the main finding survives after accounting for it.
- **An honest classifier.** A separate script (`nested_classifier.py`) checks how well the measures can tell the two groups apart *without cheating* — see its header for what "cheating" (data leakage) means and how it's avoided.

---

## Repository contents

| File | What it does (plain language) |
|------|-------------------------------|
| `pipeline.py` | The engine. Every analysis function lives here — it turns one person's heartbeat file into their full set of complexity measures. The other scripts import from it. |
| `healthy_hrvacc.py` | Runs the pipeline over the **control** group of the primary (HRV-ACC) dataset. |
| `psych_hrvacc.py` | Runs the pipeline over the **patient** group of the primary dataset. |
| `healthy.py` | Runs the pipeline over a second, supplementary healthy set (MIT-BIH Normal Sinus Rhythm) recorded on *different* hardware, used only as a cross-device sanity check. |
| `comparison.py` | Takes the per-person results and does the group statistics: effect sizes, the spectral-slope covariate analysis, MFDFA, recurrence-network measures, surrogate testing, and the metric-to-metric correlation tables. |
| `accelerometer_mobility.py` | Measures physical movement from the built-in accelerometer and tests whether the main spectral-slope difference survives after removing the influence of movement (and heart rate and vagal power). Independent re-implementation of the dataset authors' method. |
| `nested_classifier.py` | The leakage-free (honest) classifier: how well the measures separate the two groups when the model is never allowed to peek at the person it's being tested on. |
| `figures.py` | Rebuilds the manuscript figures from the result tables. |
| `results/` | The summary tables produced by the analysis, included so a reader can check the reported numbers without re-running everything. See `results/README.md`. |
| `DATA_SETUP.md` | How to obtain the raw data (it is **not** included here) and where to put it. |
| `download_data.py` | Convenience helper that downloads the supplementary MIT-BIH data automatically and gives step-by-step instructions for the primary dataset. |

---

## Data availability

The raw recordings are **not distributed in this repository** — they belong to the original dataset authors, and this project simply points you to the same public sources. See **[`DATA_SETUP.md`](DATA_SETUP.md)** for exact download links and the folder layout the scripts expect. In brief:

- **Primary — HRV-ACC dataset** (Książek et al., 2023): Zenodo, DOI [10.5281/zenodo.8171266](https://doi.org/10.5281/zenodo.8171266). Contains both cohorts' heartbeat and accelerometer data.
- **Supplementary — MIT-BIH Normal Sinus Rhythm database:** PhysioNet ([`nsrdb`](https://physionet.org/content/nsrdb/)).

---

## Installation

You need Python (this was developed and tested on **Python 3.14.5**). From a terminal:

```bash
# 1. Get the code
git clone https://github.com/YOUR-USERNAME/ecg-criticality-psychiatry-pilot.git
cd ecg-criticality-psychiatry-pilot

# 2. (Recommended) make an isolated environment so these package versions
#    don't collide with anything else on your machine
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install the exact library versions used for the study
pip install -r requirements.txt
```

---

## How to run

First set up the data (see `DATA_SETUP.md`), then run the scripts **in this order**, because later steps read files the earlier ones produce:

```bash
python healthy_hrvacc.py        # 1. control cohort  -> ecg_results_HealthyACC.csv
python psych_hrvacc.py          # 2. patient cohort  -> ecg_results_Psychiatric.csv
python healthy.py               # 3. (optional) supplementary MIT-BIH cross-device set
python comparison.py            # 4. group statistics + all the summary tables
python accelerometer_mobility.py# 5. activity-confound check on the main finding
python nested_classifier.py     # 6. honest (leakage-free) classifier
python figures.py               # 7. manuscript figures
```

Each script writes into an `outputs/` folder (created automatically). Step 4 must come before step 7, because the figures are drawn from the tables step 4 produces.

---

## How to cite

If this code is useful to you, please cite **both** the manuscript and this repository. A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

- **Manuscript:** *(add full reference / preprint DOI once posted)*
- **Code:** Smith, C. (2026). *Cardiac autonomic complexity in treated psychotic-spectrum disorders — pilot analysis code* (v1.0.0). Zenodo. *(add DOI once minted)*

Please also cite the dataset authors (Książek et al., 2023) and, for the mobility method specifically, their original paper, since `accelerometer_mobility.py` re-implements their approach.

---

## License

Released under the [MIT License](LICENSE). The accelerometer mobility method is an **independent re-implementation** written from the description in Książek et al.'s paper and shares no code with their GPL-3.0 implementation, so this repository as a whole is MIT-licensed. Please still cite their paper for the method.

## Contact

Corbin Smith — Independent Researcher
ORCID: [0009-0009-1721-8640](https://orcid.org/0009-0009-1721-8640)
Email: corbinedwardsmith@gmail.com
