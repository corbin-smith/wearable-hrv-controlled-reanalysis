"""patient_metadata.py: parse the subject-level metadata released with HRV-ACC version 2.

patient_data_summary.csv is semicolon-delimited, its ICD field can contain commas (comorbid codes),
and some copies end every line with a comma; the parser handles all three and validates the result.

Fields: ID (maps to record "treatment_<ID>"), gender (W = woman, M = man), age (years),
ICD (ICD-10 codes, primary first: F20 schizophrenia, F31 bipolar affective disorder, F19 and F19.2
other psychoactive substance use and dependence, F13.2 sedative or hypnotic dependence), quetiapine.

The medication field records quetiapine only; patients marked "no" received other medication.
control_data_summary.csv has ID, gender and age, and is parsed by parse_control_metadata().
"""
import os

import numpy as np
import pandas as pd

PATIENT_METADATA_FILE = "patient_data_summary.csv"
CONTROL_METADATA_FILE = "control_data_summary.csv"   # not yet released

ICD_LABELS = {
    "F20":   "schizophrenia",
    "F31":   "bipolar affective disorder",
    "F19":   "multiple drug use",
    "F19.2": "psychoactive substance dependence",
    "F13.2": "sedative/hypnotic dependence",
}
PRIMARY_DIAGNOSIS = {"F20": "schizophrenia", "F31": "bipolar"}
SUBSTANCE_PREFIXES = ("F10", "F11", "F12", "F13", "F14", "F15",
                      "F16", "F17", "F18", "F19")


def _clean(token):
    return str(token).strip().strip(",").strip()


def parse_control_metadata(path=CONTROL_METADATA_FILE):
    """Parse the control metadata file (ID;gender;age).

    Returns a DataFrame indexed by record_id ("control_2", ...), with empty diagnosis and medication
    columns so that the two cohorts concatenate cleanly. A change in format raises an error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"control metadata not found at {path}. Download the updated "
            f"HRV-ACC release from Zenodo and place control_data_summary.csv "
            f"in the project root.")

    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    header = [_clean(x) for x in lines[0].split(";")]
    if [h.lower() for h in header[:3]] != ["id", "gender", "age"]:
        raise ValueError(f"unexpected header in {path}: {header}. "
                         f"Expected ID;gender;age.")

    rows = []
    for ln in lines[1:]:
        parts = [_clean(x) for x in ln.split(";")]
        if len(parts) < 3:
            raise ValueError(f"malformed line in {path}: {ln!r}")
        gender = parts[1].upper()
        rows.append({
            "record_id": f"control_{int(parts[0])}",
            "Patient_ID": int(parts[0]),
            "Sex": {"W": "F", "M": "M"}.get(gender, gender),
            "Age": int(parts[2]),
            "ICD_All": "", "ICD_Primary": "", "Diagnosis": "control",
            "Comorbid_Codes": "", "Comorbid_Substance": False,
            "Quetiapine": False,
        })

    df = pd.DataFrame(rows).set_index("record_id")
    if df.index.duplicated().any():
        raise ValueError("duplicate control IDs in metadata")
    bad = set(df["Sex"]) - {"F", "M"}
    if bad:
        raise ValueError(f"unrecognised sex codes in control file: {bad}")
    if not df["Age"].between(10, 100).all():
        raise ValueError("implausible control age values parsed")
    return df


def parse_metadata_file(path=PATIENT_METADATA_FILE):
    """Parse the patient metadata file into a DataFrame indexed by record_id ("treatment_1", ...).

    Raises on any malformed line instead of returning partial data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"patient metadata not found at {path}. Download the updated "
            f"HRV-ACC release from Zenodo (doi.org/10.5281/zenodo.8171266) and "
            f"place patient_data_summary.csv in the project root.")

    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    header = [_clean(x) for x in lines[0].split(";")]
    expected = ["ID", "gender", "age", "ICD", "quetiapine"]
    if [h.lower() for h in header] != [e.lower() for e in expected]:
        raise ValueError(f"unexpected header in {path}: {header}. "
                         f"Expected {expected}.")

    rows = []
    for ln in lines[1:]:
        parts = ln.split(";")
        if len(parts) < 5:
            raise ValueError(f"malformed line in {path}: {ln!r}")
        pid = _clean(parts[0])
        gender = _clean(parts[1]).upper()
        age = _clean(parts[2])
        quet = _clean(parts[-1]).lower()
        # Everything between age and quetiapine is the ICD field.
        icd_raw = ";".join(parts[3:-1])
        codes = [c.strip() for c in _clean(icd_raw).split(",") if c.strip()]
        if not codes:
            raise ValueError(f"no ICD code parsed from line: {ln!r}")

        primary = codes[0]
        comorbid = codes[1:]
        rows.append({
            "record_id": f"treatment_{int(pid)}",
            "Patient_ID": int(pid),
            "Sex": {"W": "F", "M": "M"}.get(gender, gender),
            "Age": int(age),
            "ICD_All": ",".join(codes),
            "ICD_Primary": primary,
            "Diagnosis": PRIMARY_DIAGNOSIS.get(primary, f"other({primary})"),
            "Comorbid_Codes": ",".join(comorbid) if comorbid else "",
            "Comorbid_Substance": any(
                c.startswith(SUBSTANCE_PREFIXES) for c in comorbid),
            "Quetiapine": quet in ("yes", "y", "true", "1"),
        })

    df = pd.DataFrame(rows).set_index("record_id")

    # -- Validation --
    if df.index.duplicated().any():
        raise ValueError("duplicate patient IDs in metadata")
    bad_sex = set(df["Sex"]) - {"F", "M"}
    if bad_sex:
        raise ValueError(f"unrecognised sex codes: {bad_sex}")
    if not df["Age"].between(10, 100).all():
        raise ValueError("implausible age values parsed")
    unknown = set(df["ICD_Primary"]) - set(PRIMARY_DIAGNOSIS)
    if unknown:
        print(f"  NOTE: primary ICD codes without a diagnosis label: {unknown}")
    return df


def attach_metadata(results_df, metadata=None, path=None, cohort="patient",
                    verbose=True):
    """Merge metadata columns onto a results DataFrame indexed by record_id.

    Rows without metadata (such as appended cohort summary rows) receive NaN.
    """
    if metadata is None:
        if cohort == "control":
            metadata = parse_control_metadata(path or CONTROL_METADATA_FILE)
        else:
            metadata = parse_metadata_file(path or PATIENT_METADATA_FILE)
    cols = ["Patient_ID", "Sex", "Age", "ICD_All", "ICD_Primary", "Diagnosis",
            "Comorbid_Codes", "Comorbid_Substance", "Quetiapine"]
    out = results_df.copy()
    for c in cols:
        out[c] = metadata[c].reindex(out.index)
    if verbose:
        matched = out["Diagnosis"].notna().sum()
        print(f"  Metadata attached to {matched}/{len(out)} rows "
              f"({metadata['Diagnosis'].value_counts().to_dict()})")
    return out


def diagnosis_ids(metadata=None, diagnosis="schizophrenia",
                  path=PATIENT_METADATA_FILE):
    """Record IDs for one diagnostic group."""
    if metadata is None:
        metadata = parse_metadata_file(path)
    return list(metadata.index[metadata["Diagnosis"] == diagnosis])


def control_metadata_available(path=CONTROL_METADATA_FILE):
    """True if control demographics have been released."""
    return os.path.exists(path)


def summarise(metadata=None, path=PATIENT_METADATA_FILE):
    """Print a cohort description."""
    if metadata is None:
        metadata = parse_metadata_file(path)
    n = len(metadata)
    print(f"  Patients: N = {n}")
    print(f"    Diagnosis      : "
          + ", ".join(f"{k} {v}" for k, v in
                      metadata['Diagnosis'].value_counts().items()))
    print(f"    Sex            : "
          + ", ".join(f"{k} {v}" for k, v in
                      metadata['Sex'].value_counts().items()))
    print(f"    Age            : mean {metadata['Age'].mean():.1f}, "
          f"SD {metadata['Age'].std(ddof=1):.1f}, "
          f"range {metadata['Age'].min()}-{metadata['Age'].max()}")
    print(f"    Quetiapine     : {int(metadata['Quetiapine'].sum())} yes, "
          f"{int((~metadata['Quetiapine']).sum())} no")
    print(f"    Comorbid subst.: {int(metadata['Comorbid_Substance'].sum())} "
          f"({', '.join(sorted(set(c for c in metadata['Comorbid_Codes'] if c)))})")
    xt = pd.crosstab(metadata["Diagnosis"], metadata["Quetiapine"])
    print(f"    Quetiapine by diagnosis:\n{xt.to_string()}")
    print("    NOTE: quetiapine is confounded with diagnosis; see module "
          "docstring before interpreting either contrast.")
    if control_metadata_available():
        try:
            ctrl = parse_control_metadata()
            from scipy.stats import mannwhitneyu, fisher_exact
            u, pv = mannwhitneyu(ctrl["Age"], metadata["Age"],
                                 alternative="two-sided")
            r = 1 - 2 * u / (len(ctrl) * len(metadata))
            tab = [[int((ctrl["Sex"] == "F").sum()), int((ctrl["Sex"] == "M").sum())],
                   [int((metadata["Sex"] == "F").sum()),
                    int((metadata["Sex"] == "M").sum())]]
            _, pf = fisher_exact(tab)
            print(f"\n  Controls: N = {len(ctrl)}")
            print(f"    Sex  : " + ", ".join(f"{k} {v}" for k, v in
                                             ctrl['Sex'].value_counts().items()))
            print(f"    Age  : mean {ctrl['Age'].mean():.1f}, "
                  f"SD {ctrl['Age'].std(ddof=1):.1f}, "
                  f"range {ctrl['Age'].min()}-{ctrl['Age'].max()}")
            print(f"\n  Group matching:")
            print(f"    Age: r = {r:+.3f}, p = {pv:.4f}")
            print(f"    Sex: Fisher exact p = {pf:.4f}")
        except Exception as exc:
            print(f"    control metadata could not be summarised: {exc}")
    else:
        print("    NOTE: no control-cohort metadata found. Age and sex "
              "cannot be compared between groups.")
    return metadata


if __name__ == "__main__":
    summarise()