"""mitbih_metadata.py: attach per-record age and sex to the MIT-BIH results file."""
import argparse
import glob
import os
import re
import sys

import pandas as pd

RESULTS_CSV = os.path.join("outputs", "healthy", "ecg_results_Healthy.csv")
HEADER_DIRS = [".", "nsrdb", os.path.join("data", "nsrdb"), "Long Healthy ECGs"]

from healthy import RECORD_IDS as RECORDS

MITBIH_AGES = {
    "16265": {"Age": 32, "Sex": "M"},
    "16272": {"Age": 20, "Sex": "F"},
    "16273": {"Age": 28, "Sex": "F"},
    "16420": {"Age": 38, "Sex": "F"},
    "16483": {"Age": 42, "Sex": "M"},
    "16539": {"Age": 35, "Sex": "F"},
    "16773": {"Age": 26, "Sex": "M"},
    "16786": {"Age": 32, "Sex": "F"},
    "16795": {"Age": 20, "Sex": "F"},
    "17052": {"Age": 45, "Sex": "F"},
    "17453": {"Age": 32, "Sex": "F"},
    "18177": {"Age": 26, "Sex": "F"},
    "18184": {"Age": 34, "Sex": "F"},
    "19088": {"Age": 41, "Sex": "F"},
    "19090": {"Age": 45, "Sex": "M"},
    "19093": {"Age": 34, "Sex": "M"},
    "19140": {"Age": 38, "Sex": "F"},
    "19830": {"Age": 50, "Sex": "F"},
}

PUBLISHED = {
    "n_male": 5, "male_age_range": (26, 45),
    "n_female": 13, "female_age_range": (20, 50),
    "n_total": 18,
}


def check_against_published(table):
    ages = {"M": [], "F": []}
    for rec, info in table.items():
        if "Age" in info and info.get("Sex") in ages:
            ages[info["Sex"]].append(info["Age"])
    checks = []
    checks.append(("total records", len(table), PUBLISHED["n_total"]))
    checks.append(("men", len(ages["M"]), PUBLISHED["n_male"]))
    checks.append(("women", len(ages["F"]), PUBLISHED["n_female"]))
    ok = all(a == b for _, a, b in checks)
    print("\n  Verification against published aggregate demographics:")
    for lab, got, exp in checks:
        print(f"    {lab:16s} {got:>3} (published {exp})  "
              f"{'OK' if got == exp else 'MISMATCH'}")
    for sex, key, lab in (("M", "male_age_range", "men"),
                          ("F", "female_age_range", "women")):
        if not ages[sex]:
            continue
        got = (min(ages[sex]), max(ages[sex]))
        exp = PUBLISHED[key]
        match = got == exp
        ok = ok and match
        print(f"    {lab} age range   {got[0]}-{got[1]} (published "
              f"{exp[0]}-{exp[1]})  {'OK' if match else 'MISMATCH'}")
    allage = ages["M"] + ages["F"]
    if allage:
        print(f"    mean age         {sum(allage)/len(allage):.2f} "
              f"(n = {len(allage)})")
    if not ok:
        print("\n    At least one aggregate does not match. Do not use this "
              "table until the discrepancy is resolved.")
    return ok


def find_headers():
    found = {}
    for d in HEADER_DIRS:
        if not os.path.isdir(d):
            continue
        for rec in RECORDS:
            for p in glob.glob(os.path.join(d, "**", f"{rec}.hea"),
                               recursive=True):
                found[rec] = p
                break
    return found


def parse_header(path):
    out = {}
    try:
        with open(path, "r", errors="ignore") as fh:
            text = fh.read()
    except Exception:
        return out
    comments = "\n".join(ln for ln in text.splitlines() if ln.strip().startswith("#"))
    if not comments:
        return out
    m = re.search(r"<?age>?\s*[:=]?\s*(\d{1,3})", comments, re.I)
    if m:
        age = int(m.group(1))
        if 10 <= age <= 100:
            out["Age"] = age
    m = re.search(r"<?sex>?\s*[:=]?\s*([MFmf])", comments, re.I)
    if m:
        out["Sex"] = m.group(1).upper()
    else:
        if re.search(r"\bfemale\b", comments, re.I):
            out["Sex"] = "F"
        elif re.search(r"\bmale\b", comments, re.I):
            out["Sex"] = "M"
    return out


def gather():
    headers = find_headers()
    print(f"  Header files located: {len(headers)}/{len(RECORDS)}")
    from_header = {}
    for rec, path in headers.items():
        info = parse_header(path)
        if info:
            from_header[rec] = info
    print(f"  Records with age/sex in the header: {len(from_header)}")
    if from_header:
        for rec in sorted(from_header)[:3]:
            print(f"    {rec}: {from_header[rec]}")

    merged = dict(from_header)
    for rec, info in MITBIH_AGES.items():
        merged.setdefault(rec, {}).update(info)
    print(f"  Records supplied by hand in MITBIH_AGES: {len(MITBIH_AGES)}")
    complete = {r: v for r, v in merged.items() if "Age" in v}
    print(f"  Records with a usable Age: {len(complete)}/{len(RECORDS)}")
    if MITBIH_AGES:
        if not check_against_published(MITBIH_AGES):
            print("\n  ABORTING: the hand-entered table fails verification.")
            sys.exit(1)
    return merged, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--csv", default=RESULTS_CSV)
    args = ap.parse_args()
    if not (args.check or args.write):
        args.check = True

    print("=" * 68)
    print("  MIT-BIH NSR subject metadata")
    print("=" * 68)
    merged, complete = gather()

    if len(complete) < len(RECORDS):
        print(f"\n  Per-record ages are NOT available for "
              f"{len(RECORDS) - len(complete)} of {len(RECORDS)} records.")
        print("\n  The age-adjusted cross-device analysis cannot be run, and")
        print("  comparison.py will skip it with a message. This is the correct")
        print("  behaviour. Do not fill MITBIH_AGES with midpoints or estimates.")
        print("\n  Instead, state in the Limitations that the MIT-BIH cohort is")
        print("  younger than the HRV-ACC cohorts (published range 20-50 against")
        print("  20-69), that the spectral exponent is age-dependent within the")
        print("  control cohort, and that per-record ages are not published, so")
        print("  the cross-device comparison could not be age-adjusted.")
        print("\n  If you obtain per-record demographics from the database")
        print("  authors or a published table, enter them in MITBIH_AGES and")
        print("  re-run with --write.")
        if not args.write:
            return
        print("\n  Refusing to write a partially populated Age column.")
        sys.exit(1)

    if args.check:
        print("\n  All records have age data. Re-run with --write to merge.")
        return

    if not os.path.exists(args.csv):
        sys.exit(f"  Results file not found: {args.csv}")
    df = pd.read_csv(args.csv, index_col=0)
    before = df.copy()
    add = pd.DataFrame(index=df.index, columns=["Age", "Sex"], dtype=object)
    df = pd.concat([df.drop(columns=[c for c in ("Age", "Sex")
                                     if c in df.columns]), add], axis=1)
    n = 0
    for rec, info in complete.items():
        key = next((i for i in df.index if str(i).strip() == rec), None)
        if key is None:
            continue
        for col, val in info.items():
            df.loc[key, col] = val
        n += 1

    guarded = [c for c in before.columns if c not in ("Age", "Sex")]
    changed = [c for c in guarded
               if not before[c].astype(str).equals(df[c].astype(str))]
    if changed:
        sys.exit(f"  ABORTED: non-metadata columns changed: {changed}")

    backup = args.csv.replace(".csv", "_pre_metadata.csv")
    before.to_csv(backup)
    df.to_csv(args.csv)
    print(f"\n  Age and sex written for {n} records.")
    print(f"  Backup: {backup}")
    print(f"  Updated: {args.csv}")
    print("\n  Re-run comparison.py to enable the age-adjusted cross-device "
          "analysis.")


if __name__ == "__main__":
    main()
