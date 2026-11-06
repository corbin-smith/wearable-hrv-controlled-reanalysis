"""merge_mobility.py: copy the per-subject accelerometer mobility value into both HRV-ACC results files.

Run after accelerometer_mobility.py and before comparison.py, which uses it as a covariate.
"""
import os
import sys
import pandas as pd

MOB = os.path.join("outputs", "accelerometer", "mobility_coefficients.csv")
TARGETS = [
    os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    os.path.join("outputs", "psychiatric",    "ecg_results_Psychiatric.csv"),
]

if not os.path.exists(MOB):
    sys.exit(f"mobility file not found: {MOB}\nRun accelerometer_mobility.py first.")

mob = pd.read_csv(MOB, index_col=0)
if "mobility_mg" not in mob.columns:
    sys.exit(f"'mobility_mg' column not in {MOB}; found {list(mob.columns)}")
print(f"mobility values available for {mob['mobility_mg'].notna().sum()} subjects")

for path in TARGETS:
    if not os.path.exists(path):
        print(f"  ! skipping, not found: {path}")
        continue
    df = pd.read_csv(path, index_col=0)
    before = df.copy()

    df["mobility_mg"] = mob["mobility_mg"].reindex(df.index)

    # Refuse to write if anything other than the new column changed.
    guarded = [c for c in before.columns if c != "mobility_mg"]
    changed = [c for c in guarded
               if not before[c].astype(str).equals(df[c].astype(str))]
    if changed:
        sys.exit(f"  ABORTED on {path}: other columns changed: {changed}")

    subs = [i for i in df.index
            if str(i).startswith(("control_", "treatment_"))]
    n = int(df.loc[subs, "mobility_mg"].notna().sum())
    if n < len(subs):
        print(f"  ! {os.path.basename(path)}: only {n}/{len(subs)} subjects "
              f"matched. Check that record IDs use the same format in both files.")

    backup = path.replace(".csv", "_pre_mobility.csv")
    before.to_csv(backup)
    df.to_csv(path)
    print(f"  {os.path.basename(path)}: mobility written for {n}/{len(subs)} subjects")
    print(f"    backup -> {backup}")

print("\nNow re-run comparison.py. The joint model should report "
      "'JOINT 6-covariate adjustment'.")