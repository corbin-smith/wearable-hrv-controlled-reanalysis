"""stationary_reanalysis.py: recompute the primary measures on movement-free data.

Uses each subject's longest contiguous movement-free run, not the concatenated pieces: splicing
non-adjacent segments adds a discontinuity at every join, which spectra read as broadband power
and DFA as a step.

Usage:  python stationary_reanalysis.py --assess                           (run lengths only)
        python stationary_reanalysis.py --run --truncate 600 --match beats
        python stationary_reanalysis.py --run --truncate 370 --match duration
Surrogate tests are not repeated here.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from pipeline import (
    SKIP_SECONDS as PIPE_SKIP,
    ANALYSIS_SECONDS as PIPE_ANALYSIS,reject_ectopic_beats, compute_frequency_domain,
                      compute_complexity_metrics, QUANT_STEP_S)

MASK_DIR = os.path.join("outputs", "accelerometer", "stationary_masks")
OUT_DIR = os.path.join("outputs", "stationary")
COHORTS = {
    "control":   {"dir": "Control ECGs",     "prefix": "control"},
    "treatment": {"dir": "Psychiatric ECGs", "prefix": "treatment"},
}
# Analysis window constants come from pipeline.py.
SKIP_SECONDS = PIPE_SKIP
ANALYSIS_SECONDS = PIPE_ANALYSIS
MIN_SEGMENT_BEATS = 500          # below this a subject is not analysable


def longest_true_run(mask):
    """Return (start, end) of the longest contiguous True run in a bool array."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, 0
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts
    k = int(np.argmax(lengths))
    return int(starts[k]), int(ends[k])


def load_window(record_id):
    """Return the R-R series for the analysis window, in seconds."""
    pref = str(record_id).split("_")[0]
    cfg = next((c for c in COHORTS.values() if c["prefix"] == pref), None)
    if cfg is None:
        return None
    path = os.path.join(cfg["dir"], f"{record_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=";")
    rr_col = next((c for c in df.columns
                   if "rr" in c.lower() or "interval" in c.lower()), None)
    if rr_col is None:
        return None
    rr = pd.to_numeric(df[rr_col], errors="coerce").dropna().values / 1000.0
    cum = np.cumsum(rr)
    m = (cum >= SKIP_SECONDS) & (cum < SKIP_SECONDS + ANALYSIS_SECONDS)
    return rr[m]


def gather():
    """Load every mask and pair it with its R-R window."""
    if not os.path.isdir(MASK_DIR):
        sys.exit(f"mask folder not found: {MASK_DIR}. "
                 f"Run accelerometer_mobility.py first.")
    out = []
    for fn in sorted(os.listdir(MASK_DIR)):
        if not fn.endswith("_stationary.npy"):
            continue
        rid = fn[:-len("_stationary.npy")]
        mask = np.load(os.path.join(MASK_DIR, fn))
        rr = load_window(rid)
        if rr is None:
            print(f"  ! {rid}: R-R window unavailable")
            continue
        n = min(len(mask), len(rr))
        out.append((rid, rr[:n], mask[:n]))
    return out


def assess():
    data = gather()
    rows = []
    for rid, rr, mask in data:
        a, b = longest_true_run(mask)
        rows.append({"Record_ID": rid,
                     "Group": "control" if rid.startswith("control") else "patient",
                     "N_Window": len(rr),
                     "N_Stationary_Total": int(mask.sum()),
                     "N_Longest_Contiguous": b - a,
                     "Frac_Of_Window": (b - a) / len(rr) if len(rr) else np.nan})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "Stationary_Longest_Runs.csv")
    df.to_csv(path, index=False, float_format="%.4f")

    print(f"\n  Longest contiguous stationary run, by group:")
    print(df.groupby("Group")["N_Longest_Contiguous"]
            .describe()[["count", "mean", "50%", "min", "max"]].round(0).to_string())
    print(f"\n  Subjects meeting a given segment length:")
    for thr in (500, 800, 1000, 1200, 1500, 2000):
        ok = df["N_Longest_Contiguous"] >= thr
        c = int((ok & (df.Group == "control")).sum())
        p = int((ok & (df.Group == "patient")).sum())
        print(f"    >= {thr:5d} beats: {int(ok.sum()):2d}/{len(df)}  "
              f"(control {c}/30, patient {p}/30)")
    print(f"\n  written -> {path}")
    print("\n  DECIDE: pick the largest truncation length that retains close to")
    print("  all 60 subjects, then run:")
    print("      python stationary_reanalysis.py --run --truncate <length>")
    print("  If no length retains at least ~50 subjects, do NOT run stage 2.")
    print("  Report the descriptive statistics as a limitation instead: the")
    print("  movement-free portion of the recording is too short to support a")
    print("  reliable re-estimate, and that is an honest result.")
    return df


def run(truncate, match="beats"):
    """Recompute the measures on segments matched by beat count or by duration.

    Matched beats give patients, whose heart rate is higher, shorter segments and noisier spectra;
    matched duration gives them more beats, which affects entropy and DFA. Running both separates
    the two: --match beats suits entropy, DFA, MFDFA and recurrence, --match duration the spectra.
    """
    data = gather()
    rows = []
    skipped = []
    for i, (rid, rr, mask) in enumerate(data, 1):
        a, b = longest_true_run(mask)
        seg = rr[a:b]
        if match == "duration":
            # Take the leading `truncate` seconds of the segment.
            cum = np.cumsum(seg)
            keep = cum <= truncate
            if keep.sum() < MIN_SEGMENT_BEATS or cum[-1] < truncate:
                skipped.append((rid, f"{cum[-1]:.0f}s"))
                continue
            seg = seg[keep]
        else:
            if len(seg) < max(MIN_SEGMENT_BEATS, truncate):
                skipped.append((rid, len(seg)))
                continue
            seg = seg[:truncate]
        print(f"    [{i:>3}/{len(data)}] {rid:<16} {len(seg)} beats", flush=True)
        clean, pct, n, _ = reject_ectopic_beats(seg, verbose=False)
        if n < 240:
            skipped.append((rid, n))
            continue
        t = np.cumsum(clean)
        try:
            fd, _, _ = compute_frequency_domain(clean, t)
            cm, _, _ = compute_complexity_metrics(clean, quant_step=QUANT_STEP_S)
        except Exception as exc:
            skipped.append((rid, f"{type(exc).__name__}"))
            continue
        rec = {"Record_ID": rid,
               "Group": "control" if rid.startswith("control") else "patient",
               "N_Beats": n, "Duration_s": float(np.sum(clean)),
               "Artifact_Percent": pct,
               "Match_Mode": match,
               # Number of Welch segments averaged, reported because it sets spectral precision.
               "Welch_N_Segments": fd.get("Welch_N_Segments", np.nan)}
        # Spectral distribution, amplitude and heart rate for the same segments.
        for k in ("PSD_Beta", "PSD_Beta_NoMayer", "PSD_Beta_Specparam",
                  "LF_Power_ms2", "HF_Power_ms2", "Mayer_Peak_Amp",
                  "VLF_Power_ms2", "VLF_LF_Ratio", "LF_HF_Ratio",
                  "Total_Power_ms2"):
            rec[k] = fd.get(k, np.nan)
        rec["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(clean * 1000.0) ** 2)))
        rec["HRV_MeanHR"] = float(60.0 / np.mean(clean))
        for k in ("SampEn", "ApEn", "PermEn", "MSE_Mean",
                  "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2",
                  "RQA_DET", "RQA_LAM", "MFDFA_H2"):
            rec[k] = cm.get(k, np.nan)
        rows.append(rec)

    if not rows:
        sys.exit("no subject produced a usable stationary segment at this "
                 "truncation length. Lower --truncate or abandon stage 2.")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"Stationary_Reanalysis_{match}.csv")
    df.to_csv(path, index=False, float_format="%.6f")

    from scipy.stats import mannwhitneyu
    print(f"\n  Retained {len(df)}/{len(data)} subjects "
          f"({int((df.Group=='control').sum())} control, "
          f"{int((df.Group=='patient').sum())} patient) at {truncate} beats.")
    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped[:10]}"
              f"{' ...' if len(skipped) > 10 else ''}")
    for col in ("N_Beats", "Duration_s", "Welch_N_Segments"):
        a_ = df.loc[df.Group == "control", col]
        b_ = df.loc[df.Group == "patient", col]
        print(f"  {col:<18} control {a_.median():8.1f}   patient {b_.median():8.1f}")
    print(f"\n  {'metric':<22}{'control':>10}{'patient':>10}{'r':>8}{'p':>10}")
    for m in ("PSD_Beta", "PSD_Beta_NoMayer", "VLF_LF_Ratio", "VLF_Power_ms2",
              "HRV_RMSSD", "DFA_Alpha2", "SampEn", "PermEn", "MSE_Mean",
              "RQA_DET"):
        a_ = df.loc[df.Group == "control", m].dropna()
        b_ = df.loc[df.Group == "patient", m].dropna()
        if len(a_) < 5 or len(b_) < 5:
            continue
        u, pv = mannwhitneyu(a_, b_, alternative="two-sided")
        r = 1 - 2 * u / (len(a_) * len(b_))
        print(f"  {m:<22}{a_.mean():10.3f}{b_.mean():10.3f}{r:8.3f}{pv:10.2e}")
    print(f"\n  written -> {path}")
    print("\n  Compare these effect sizes with the full-window values. A")
    print("  materially smaller effect here means movement contributed to the")
    print("  original result; a similar effect means it did not. Report the")
    print("  comparison either way, and state the reduced N and beat count.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assess", action="store_true",
                    help="report segment lengths only, compute nothing")
    ap.add_argument("--run", action="store_true",
                    help="recompute metrics on the longest stationary segment")
    ap.add_argument("--truncate", type=int, default=600,
                    help="common beat count (--match beats) or common duration "
                         "in seconds (--match duration)")
    ap.add_argument("--match", choices=["beats", "duration"], default="beats",
                    help="equalise subjects by beat count or by elapsed time")
    args = ap.parse_args()
    if args.assess or not args.run:
        assess()
    else:
        run(args.truncate, match=args.match)