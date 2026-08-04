"""
download_data.py
================
Convenience helper to fetch the PUBLIC datasets this project uses and place them
in the folders the analysis scripts expect.

PLAIN-LANGUAGE NOTE
-------------------
This project does NOT ship the raw heartbeat data, because it belongs to the
original dataset authors. Instead, this script downloads what it can for you from
the same public sources and arranges it into the exact folder layout the analysis
scripts look for. Run it once, from inside the project folder:

    python download_data.py

Requires an internet connection.

  - The MIT-BIH (PhysioNet) portion is fully automated below.
  - The HRV-ACC (Zenodo) portion is a large, multi-file archive, so this script
    prints exact instructions rather than guessing at its internal file names.
    See DATA_SETUP.md for the same steps with the target folder layout.
"""

import os

# The 18 MIT-BIH Normal Sinus Rhythm records used as the supplementary,
# cross-device healthy set (same list as in healthy.py).
NSR_RECORDS = [
    "16265", "16272", "16273", "16420", "16483",
    "16539", "16773", "16786", "16795", "17052",
    "17453", "18177", "18184", "19088", "19090",
    "19093", "19140", "19830",
]
NSR_DB_NAME = "nsrdb"                 # PhysioNet database slug
NSR_TARGET_DIR = "Long Healthy ECGs"  # folder healthy.py reads from


def download_mitbih_nsr():
    """Download each NSR record into 'Long Healthy ECGs/<id>/<id>.*'."""
    try:
        import wfdb
    except ImportError:
        print("  [skip] the 'wfdb' package is not installed. Run "
              "'pip install -r requirements.txt' first.")
        return

    print("Downloading MIT-BIH Normal Sinus Rhythm records from PhysioNet...")
    for rid in NSR_RECORDS:
        dest = os.path.join(NSR_TARGET_DIR, rid)
        os.makedirs(dest, exist_ok=True)

        # The two files every record needs are <id>.dat (the signal) and
        # <id>.hea (its header). The .atr annotation file is nice-to-have; we
        # try it separately so a missing one never stops the download.
        for fname in (f"{rid}.dat", f"{rid}.hea"):
            try:
                wfdb.dl_files(NSR_DB_NAME, dest, [fname])
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not fetch {fname}: {exc}")
        try:
            wfdb.dl_files(NSR_DB_NAME, dest, [f"{rid}.atr"])
        except Exception:  # noqa: BLE001
            pass  # annotations are optional for this pipeline

        print(f"  {rid}  ->  {dest}/")
    print("MIT-BIH NSR download complete.\n")


def print_zenodo_instructions():
    """Explain how to get the primary HRV-ACC dataset by hand."""
    print("=" * 70)
    print("PRIMARY DATASET — HRV-ACC (Ksiazek et al., 2023)  [DO THIS BY HAND]")
    print("=" * 70)
    print("  1. Open:  https://doi.org/10.5281/zenodo.8171266")
    print("  2. Download the heartbeat (R-R) data and the accelerometer archive")
    print("     (accelerometer_anonymized_data.zip).")
    print("  3. Arrange the files into these folders at the top of the project:")
    print()
    print("       Control ECGs/        -> control_N.csv   (control heartbeats)")
    print("       Psychiatric ECGs/    -> treatment_N.csv (patient heartbeats)")
    print("       Accelerometer ECGs/  -> control_N.csv AND treatment_N.csv")
    print("                               (all accelerometer files together)")
    print()
    print("  Full details and a folder diagram are in DATA_SETUP.md.")
    print("=" * 70)


if __name__ == "__main__":
    download_mitbih_nsr()
    print_zenodo_instructions()
