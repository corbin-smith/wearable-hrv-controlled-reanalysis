"""download_data.py: fetch the MIT-BIH records and print the steps for the HRV-ACC data."""

import os

NSR_RECORDS = [
    "16265", "16272", "16273", "16420", "16483",
    "16539", "16773", "16786", "16795", "17052",
    "17453", "18177", "18184", "19088", "19090",
    "19093", "19140", "19830",
]
NSR_DB_NAME = "nsrdb"
NSR_TARGET_DIR = "Long Healthy ECGs"

HRVACC_DOI = "https://doi.org/10.5281/zenodo.21886112"


def download_mitbih_nsr():
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
