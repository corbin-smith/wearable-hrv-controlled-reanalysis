"""rna_controls.py: recurrence-network measures under the resolution controls, merged into the results files.

Clustering coefficient and transitivity are recomputed for each subject as observed, with dither
(uniform noise one step wide), re-quantized to a common step-to-RMSSD ratio, and truncated to a
common beat count. The first three use the functions and seed of pipeline.quantization_controls(),
so the values are directly comparable with the entropy and RQA controls.

Usage:  python rna_controls.py --check   (report what would be computed)
        python rna_controls.py --run     (compute and merge into the results CSVs)
Run comparison.py afterwards.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from pipeline import (
    SKIP_SECONDS as PIPE_SKIP,
    ANALYSIS_SECONDS as PIPE_ANALYSIS,reject_ectopic_beats, truncate_series, add_dither,
                      coarsen, compute_recurrence_network,
                      resolve_theiler, RQA_DIM, RQA_DELAY,
                      QUANT_STEP_S, QUANT_SEED, QUANT_TARGET_RATIO)

COHORTS = {
    "control":   {"dir": "Control ECGs", "prefix": "control",
                  "csv": os.path.join("outputs", "healthy_hrvacc",
                                      "ecg_results_HealthyACC.csv")},
    "treatment": {"dir": "Psychiatric ECGs", "prefix": "treatment",
                  "csv": os.path.join("outputs", "psychiatric",
                                      "ecg_results_Psychiatric.csv")},
}
# Analysis window constants come from pipeline.py.
SKIP_SECONDS = PIPE_SKIP
ANALYSIS_SECONDS = PIPE_ANALYSIS

# Every column subject_controls() produces. The merge drops these before re-adding them and the
# guard excludes them from its unchanged-columns check; _assert_columns_declared() enforces the list.
NEW_COLS = [
    "RNA_ClusteringCoeff_Dither", "RNA_Transitivity_Dither",
    "RNA_ClusteringCoeff_MatchedQuant", "RNA_Transitivity_MatchedQuant",
    "RNA_ClusteringCoeff_Trunc", "RNA_Transitivity_Trunc",
    "RNA_Controls_NBeats",
    # Theiler window the controls were computed on, recorded per subject.
    "RNA_Controls_Theiler_W", "RNA_Controls_Theiler_Source",
]


def _assert_columns_declared(sample_row):
    """Stop if subject_controls() produced a key that NEW_COLS does not list."""
    undeclared = [k for k in sample_row if k not in NEW_COLS]
    if undeclared:
        sys.exit(
            f"    ABORTED: subject_controls() produced columns not listed in "
            f"NEW_COLS: {undeclared}\n"
            f"    Add them to NEW_COLS. Leaving them undeclared works on the "
            f"first run and breaks the merge on the second, because the stale "
            f"copy is not dropped before the fresh one is concatenated.")


def load_window(path):
    df = pd.read_csv(path, sep=";")
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    if rr_col is None:
        raise ValueError(f"no R-R column in {path}")
    rr = pd.to_numeric(df[rr_col], errors="coerce").dropna().values / 1000.0
    cum = np.cumsum(rr)
    m = (cum >= SKIP_SECONDS) & (cum < SKIP_SECONDS + ANALYSIS_SECONDS)
    return rr[m]


def rna_pair(x, theiler):
    """Clustering coefficient and transitivity for one series, on a fixed Theiler window.

    The window is passed in so that every condition uses the same eligible pair set.
    """
    out = compute_recurrence_network(x, theiler=theiler)
    return out.get("RNA_ClusteringCoeff", np.nan), out.get("RNA_Transitivity", np.nan)


def subject_controls(rr_clean, trunc_n=None, theiler=None):
    """Recurrence-network measures under each control condition, with the pipeline's settings.

    theiler: the window the pipeline used for this subject, read from the results CSV. When it is
    unavailable, it is resolved once from the observed series and reused for every condition.
    """
    rng = np.random.default_rng(QUANT_SEED)
    res = {}
    res["RNA_Controls_NBeats"] = int(len(rr_clean))
    if theiler is None or not np.isfinite(theiler):
        theiler, src = resolve_theiler(rr_clean, RQA_DIM, RQA_DELAY)
        res["RNA_Controls_Theiler_Source"] = f"rederived_{src}"
    else:
        theiler = int(theiler)
        res["RNA_Controls_Theiler_Source"] = "from_pipeline_results"
    res["RNA_Controls_Theiler_W"] = int(theiler)

    xd = add_dither(rr_clean, step=QUANT_STEP_S, rng=rng)
    c, t = rna_pair(xd, theiler)
    res["RNA_ClusteringCoeff_Dither"], res["RNA_Transitivity_Dither"] = c, t

    rmssd = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2)))
    step_m = QUANT_TARGET_RATIO * rmssd if rmssd > 0 else np.nan
    xm = coarsen(rr_clean, step_m) if (np.isfinite(step_m)
                                       and step_m > QUANT_STEP_S) else rr_clean
    c, t = rna_pair(xm, theiler)
    res["RNA_ClusteringCoeff_MatchedQuant"] = c
    res["RNA_Transitivity_MatchedQuant"] = t

    if trunc_n is not None and len(rr_clean) >= trunc_n:
        c, t = rna_pair(rr_clean[:trunc_n], theiler)
    else:
        c, t = np.nan, np.nan
    res["RNA_ClusteringCoeff_Trunc"], res["RNA_Transitivity_Trunc"] = c, t
    return res


def gather(cohort, trunc_n, verbose=True):
    cfg = COHORTS[cohort]
    if not os.path.exists(cfg["csv"]):
        sys.exit(f"  results CSV not found: {cfg['csv']}")
    df = pd.read_csv(cfg["csv"], index_col=0)
    subs = [i for i in df.index if str(i).startswith(cfg["prefix"] + "_")]
    rows = {}
    for k, rid in enumerate(subs, 1):
        path = os.path.join(cfg["dir"], f"{rid}.csv")
        if not os.path.exists(path):
            print(f"    ! {rid}: R-R file not found")
            continue
        if verbose:
            print(f"    [{k:>3}/{len(subs)}] {rid}", flush=True)
        try:
            rr = load_window(path)
            clean, _, _, _ = reject_ectopic_beats(rr, verbose=False)
            clean = truncate_series(clean)
            # The Theiler window the pipeline used for this subject.
            _tw = None
            if "RQA_Theiler_W" in df.columns and rid in df.index:
                _tw = pd.to_numeric(pd.Series([df.at[rid, "RQA_Theiler_W"]]),
                                    errors="coerce").iloc[0]
            rows[rid] = subject_controls(clean, trunc_n=trunc_n, theiler=_tw)
        except Exception as exc:
            print(f"    ! {rid}: {type(exc).__name__}: {exc}")
    return df, rows, cfg["csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--truncate", type=int, default=None,
                    help="common beat count for the length control; defaults "
                         "to the smallest analysed series across both cohorts")
    args = ap.parse_args()
    if not (args.check or args.run):
        args.check = True

    print("=" * 68)
    print("  RECURRENCE-NETWORK INSTRUMENTATION CONTROLS")
    print("=" * 68)

    # Common beat count from the existing results, if not supplied.
    trunc = args.truncate
    if trunc is None:
        mins = []
        for cfg in COHORTS.values():
            if os.path.exists(cfg["csv"]):
                d = pd.read_csv(cfg["csv"], index_col=0)
                s = d.loc[[i for i in d.index
                           if str(i).startswith(cfg["prefix"] + "_")]]
                if "Num_RR" in s.columns:
                    mins.append(pd.to_numeric(s["Num_RR"],
                                              errors="coerce").min())
        trunc = int(min(mins)) if mins else None
    print(f"  Length control: truncating every series to {trunc} beats")
    print(f"  Quantization step: {QUANT_STEP_S:.6f} s   "
          f"matched target ratio: {QUANT_TARGET_RATIO}")

    if args.check:
        print("\n  Would compute, for every subject in both cohorts:")
        for c in NEW_COLS:
            print(f"    {c}")
        print("\n  Re-run with --run to compute and merge. Expect roughly one "
              "second per subject per condition.")
        return

    for cohort in COHORTS:
        print(f"\n  {cohort}:")
        df, rows, csv_path = gather(cohort, trunc)
        if not rows:
            print("    nothing computed; skipping write")
            continue
        before = df.copy()
        _assert_columns_declared(next(iter(rows.values())))
        add = pd.DataFrame.from_dict(rows, orient="index")
        add = add.reindex(df.index)
        df = pd.concat([df.drop(columns=[c for c in NEW_COLS
                                         if c in df.columns]), add], axis=1)
        guarded = [c for c in before.columns if c not in NEW_COLS]
        changed = [c for c in guarded
                   if not before[c].astype(str).equals(df[c].astype(str))]
        if changed:
            sys.exit(f"    ABORTED: non-RNA columns changed: {changed}")
        backup = csv_path.replace(".csv", "_pre_rna_controls.csv")
        before.to_csv(backup)
        df.to_csv(csv_path)
        print(f"    written for {len(rows)} subjects -> {csv_path}")
        print(f"    backup -> {backup}")

    print("\n  Now re-run comparison.py. The recurrence-network metrics will")
    print("  appear in the quantization control table alongside the entropy")
    print("  measures, and in the recording-length adjustment.")
    print("\n  INTERPRETATION. RQA determinism, built from the same recurrence")
    print("  matrix, loses its group difference under both controls. If the")
    print("  network measures behave the same way, they should be reported the")
    print("  same way, and the recurrence-network result should not be stated")
    print("  as a group difference.")


if __name__ == "__main__":
    main()