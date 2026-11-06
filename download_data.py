"""download_data.py: fetch the MIT-BIH records and print the steps for the HRV-ACC data.

The raw recordings are not part of this repository. Run once from the repository folder:

    python download_data.py

The 18 MIT-BIH Normal Sinus Rhythm records are downloaded from PhysioNet into "Long Healthy ECGs/".
The HRV-ACC files (Zenodo, version 2) are downloaded by hand; the steps are printed below and in
DATA_SETUP.md.
"""

import os

# The 18 MIT-BIH Normal Sinus Rhythm records (the full database; same list as in healthy.py).
NSR_RECORDS = [
    "16265", "16272", "16273", "16420", "16483",
    "16539", "16773", "16786", "16795", "17052",
    "17453", "18177", "18184", "19088", "19090",
    "19093", "19140", "19830",
]
NSR_DB_NAME = "nsrdb"                 # PhysioNet database slug
NSR_TARGET_DIR = "Long Healthy ECGs"  # folder healthy.py reads from

HRVACC_DOI = "https://doi.org/10.5281/zenodo.21886112"


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
        # .dat (signal) and .hea (header) are required; .atr (annotations) is optional.
        for fname in (f"{rid}.dat", f"{rid}.hea"):
            try:
                wfdb.dl_files(NSR_DB_NAME, dest, [fname])
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not fetch {fname}: {exc}")
        try:
            wfdb.dl_files(NSR_DB_NAME, dest, [f"{rid}.atr"])
        except Exception:  # noqa: BLE001
            pass
        print(f"  {rid}  ->  {dest}/")
    print("MIT-BIH NSR download complete.\n")


def print_zenodo_instructions():
    """Print the steps for the HRV-ACC dataset."""
    print("=" * 72)
    print("HRV-ACC dataset, version 2 (Ksiazek et al.)  [download by hand]")
    print("=" * 72)
    print(f"  1. Open {HRVACC_DOI}")
    print("  2. Download HRV_anonymized_data.zip, accelerometer_anonymized_data.zip,")
    print("     patient_data_summary.csv and control_data_summary.csv.")
    print("  3. Put the files in the repository folder like this:")
    print()
    print("       Control ECGs/             control_N.csv    (from HRV_anonymized_data.zip)")
    print("       Psychiatric ECGs/         treatment_N.csv  (from HRV_anonymized_data.zip)")
    print("       Accelerometer ECGs/       control_N.csv and treatment_N.csv")
    print("                                 (all of accelerometer_anonymized_data.zip)")
    print("       patient_data_summary.csv  (top level)")
    print("       control_data_summary.csv  (top level)")
    print()
    print("  Details are in DATA_SETUP.md.")
    print("=" * 72)


if __name__ == "__main__":
    download_mitbih_nsr()
    print_zenodo_instructions()
