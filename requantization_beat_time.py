"""requantization_beat_time.py: the 128 Hz resolution experiment with beat times rounded, not intervals."""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.complexity import fractal_dfa
from scipy.stats import mannwhitneyu

import pipeline as P

OUT_DIR = os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych")
HEALTHY_CSV = os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv")
PSYCH_CSV = os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv")
RR_DIRS = {"control": "Control ECGs", "treatment": "Psychiatric ECGs"}

STEP_S = P.REQUANT_TARGET_STEP_S
N_PHASES = 10
SEED = 1234
N_BOOT = 2000
BOOT_SEED = 0

AMPLITUDE_SPECTRAL = ["HRV_RMSSD", "HRV_SDNN", "LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2",
                      "Total_Power_ms2", "VLF_LF_Ratio", "LF_HF_Ratio", "PSD_Beta", "PSD_Beta_NoMayer"]
STRUCTURE = ["DFA_Alpha1", "DFA_Alpha2", "MSE_Mean", "SampEn", "ApEn", "PermEn",
             "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"]
MEASURES = AMPLITUDE_SPECTRAL + STRUCTURE
DESCRIPTORS = ["Tie_Rate", "RMSSD_In_Steps"]

TOL = {m: 1e-6 for m in MEASURES}
for _m in ("RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"):
    TOL[_m] = 2e-2


def load_cohorts():
    out = []
    for path, group in ((HEALTHY_CSV, 0), (PSYCH_CSV, 1)):
        if not os.path.exists(path):
            sys.exit(f"missing {path}: run the cohort scripts first")
        df = pd.read_csv(path)
        id_col = "Record_ID" if "Record_ID" in df.columns else df.columns[0]
        df = df.rename(columns={id_col: "Record_ID"})
        df = df[df["Record_ID"].astype(str).str.match(r"^(control|treatment)_\d+$")].copy()
        df["Group"] = group
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    if (df.Group == 0).sum() != 30 or (df.Group == 1).sum() != 30:
        sys.exit(f"expected 30 + 30 recordings, found {(df.Group == 0).sum()} + {(df.Group == 1).sum()}")
    return df


def analysed_series(record_id):
    folder = RR_DIRS["treatment" if record_id.startswith("treatment") else "control"]
    path = os.path.join(folder, f"{record_id}.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path}: see DATA_SETUP.md")
    rr_raw, t_raw, _ = P.load_rr_window(path, verbose=False)
    rr_clean, _, _, _ = P.reject_ectopic_beats(rr_raw, verbose=False)
    rr_clean = P.truncate_series(rr_clean)
    t_clean = np.concatenate([[t_raw[0]], t_raw[0] + np.cumsum(rr_clean[:-1])])
    return np.asarray(rr_clean, dtype=float), np.asarray(t_clean, dtype=float)


def round_intervals(rr):
    return P.coarsen(rr, STEP_S)


def round_beat_times(rr, phase):
    beats = np.concatenate([[0.0], np.cumsum(rr)])
    q = np.round((beats + phase) / STEP_S) * STEP_S
    return np.diff(q)


def measures_of(xq, t0, theiler):
    out = {}
    tq = np.concatenate([[t0], t0 + np.cumsum(xq[:-1])])
    xq_ms = xq * 1000.0
    out["HRV_SDNN"] = float(np.std(xq_ms, ddof=1))
    out["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(xq_ms) ** 2)))
    fd, _, _ = P.compute_frequency_domain(xq, tq)
    for k in AMPLITUDE_SPECTRAL:
        if k not in out:
            out[k] = fd.get(k, np.nan)
    out["SampEn"] = P.sample_entropy(xq)
    out["ApEn"] = P.approximate_entropy(xq)
    try:
        pe, _ = nk.entropy_permutation(xq, dimension=3, delay=1)
        out["PermEn"] = float(pe)
    except Exception:
        out["PermEn"] = np.nan
    out["MSE_Mean"] = P._mse_mean_safe(xq)
    rq = P.rqa_metrics(xq, theiler=theiler)
    out["RQA_DET"] = rq.get("RQA_DET", np.nan)
    out["RQA_LAM"] = rq.get("RQA_LAM", np.nan)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rn = P.compute_recurrence_network(xq, theiler=theiler)
        out["RNA_ClusteringCoeff"] = rn.get("RNA_ClusteringCoeff", np.nan)
        out["RNA_Transitivity"] = rn.get("RNA_Transitivity", np.nan)
    except Exception:
        out["RNA_ClusteringCoeff"] = out["RNA_Transitivity"] = np.nan
    for name, scale in (("DFA_Alpha1", np.arange(4, 17)), ("DFA_Alpha2", np.arange(16, 65))):
        try:
            v, _ = fractal_dfa(xq, scale=scale, show=False)
            out[name] = float(v)
        except Exception:
            out[name] = np.nan
    out["Tie_Rate"] = P.ordinal_tie_rate(xq)
    return out


def rank_biserial(values, group):
    a = values[(group == 0) & np.isfinite(values)]
    b = values[(group == 1) & np.isfinite(values)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided", use_continuity=True, method="auto")
    return 1.0 - 2.0 * u / (len(a) * len(b)), float(p)


def _r_only(values, group):
    a = values[(group == 0) & np.isfinite(values)]
    b = values[(group == 1) & np.isfinite(values)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided", use_continuity=True, method="auto")[0]
    return 1.0 - 2.0 * u / (len(a) * len(b))


def check_interval_model(df, verbose=True):
    worst = {}
    for _, row in df.iterrows():
        rid = row["Record_ID"]
        rr, t = analysed_series(rid)
        if abs(len(rr) - int(row["Num_RR"])) != 0:
            sys.exit(f"{rid}: rebuilt series has {len(rr)} intervals, the results file has {int(row['Num_RR'])}")
        got = measures_of(round_intervals(rr), t[0], int(row["RQA_Theiler_W"]))
        for m in MEASURES:
            col = f"{m}_Q128"
            if col not in df.columns or not np.isfinite(row[col]):
                continue
            d = abs(got[m] - float(row[col]))
            scale = max(1.0, abs(float(row[col]))) if m in AMPLITUDE_SPECTRAL else 1.0
            worst[m] = max(worst.get(m, 0.0), d / scale)
        if verbose:
            print(f"  {rid}: interval rounding reproduced", flush=True)
    bad = {m: v for m, v in worst.items() if v > TOL[m]}
    for m in MEASURES:
        if m in worst:
            print(f"    {m:22s} largest difference from the stored column {worst[m]:.2e}"
                  f"{'   <-- above tolerance' if m in bad else ''}")
    if bad:
        sys.exit("interval rounding does not reproduce the stored *_Q128 columns: "
                 + ", ".join(sorted(bad)) + ". Check that the results files and the raw data match.")
    print("  Interval rounding reproduces the stored *_Q128 columns.")


def run(n_phases):
    df = load_cohorts()
    print(f"[1] Reproducing the stored interval-rounding columns ({len(df)} recordings)...")
    check_interval_model(df, verbose=False)

    print(f"[2] Beat-time rounding, {n_phases} phases per recording...")
    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0.0, STEP_S, size=(len(df), n_phases))
    per_phase = []
    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        rid = row["Record_ID"]
        rr, t = analysed_series(rid)
        theiler = int(row["RQA_Theiler_W"])
        rmssd_s = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        for k in range(n_phases):
            vals = measures_of(round_beat_times(rr, phases[i, k]), t[0], theiler)
            vals.update({"Record_ID": rid, "Group": int(row["Group"]), "Phase_Index": k,
                         "Phase_s": float(phases[i, k]), "RMSSD_In_Steps": rmssd_s / STEP_S})
            per_phase.append(vals)
        el = time.time() - t_start
        print(f"  {i + 1:2d}/{len(df)} {rid:14s} elapsed {el / 60:5.1f} min", flush=True)
    pp = pd.DataFrame(per_phase)

    g = pp.groupby("Record_ID", sort=False)
    ids = df["Record_ID"]
    cols = {}
    for m in MEASURES + ["Tie_Rate"]:
        cols[f"{m}_BeatTime"] = ids.map(g[m].median()).values
        cols[f"{m}_BeatTime_PhaseMin"] = ids.map(g[m].min()).values
        cols[f"{m}_BeatTime_PhaseMax"] = ids.map(g[m].max()).values
        if m in df.columns:
            cols[f"{m}_Recorded"] = df[m].values
        if f"{m}_Q128" in df.columns:
            cols[f"{m}_IntervalRounding"] = df[f"{m}_Q128"].values
    cols["Tie_Rate_IntervalRounding"] = df["Requant_Tie_Rate"].values
    cols["RMSSD_In_Steps"] = df["Requant_RMSSD_In_Steps"].values
    cols["N_Phases"] = n_phases
    cols["Seed"] = SEED
    per_subject = pd.concat([df[["Record_ID", "Group"]], pd.DataFrame(cols, index=df.index)], axis=1)

    print("[3] Group comparisons and bootstrap intervals...")
    grp = per_subject["Group"].to_numpy()
    idx_c, idx_p = np.where(grp == 0)[0], np.where(grp == 1)[0]
    rng_b = np.random.default_rng(BOOT_SEED)
    boot_idx = [np.concatenate([rng_b.choice(idx_c, len(idx_c)), rng_b.choice(idx_p, len(idx_p))])
                for _ in range(N_BOOT)]

    def boot(col_a, col_b=None):
        a = per_subject[col_a].to_numpy(dtype=float)
        b = per_subject[col_b].to_numpy(dtype=float) if col_b else None
        vals = []
        for ix in boot_idx:
            ra = _r_only(a[ix], grp[ix])
            vals.append((_r_only(b[ix], grp[ix]) - ra) if b is not None else ra)
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    rows = []
    for m in MEASURES:
        rec, ivl, bt = f"{m}_Recorded", f"{m}_IntervalRounding", f"{m}_BeatTime"
        r0, p0 = rank_biserial(per_subject[rec].to_numpy(dtype=float), grp)
        r1, p1 = rank_biserial(per_subject[ivl].to_numpy(dtype=float), grp)
        r2, p2 = rank_biserial(per_subject[bt].to_numpy(dtype=float), grp)
        by_phase = [_r_only(pp.loc[pp.Phase_Index == k, m].to_numpy(dtype=float),
                            pp.loc[pp.Phase_Index == k, "Group"].to_numpy()) for k in range(n_phases)]
        d1 = boot(rec, ivl)
        d2 = boot(rec, bt)
        c2 = boot(bt)
        rows.append({
            "Metric": m,
            "Class": "amplitude_spectral" if m in AMPLITUDE_SPECTRAL else "structure",
            "r_recorded": r0, "p_recorded": p0,
            "r_interval_rounding": r1, "p_interval_rounding": p1,
            "delta_interval_rounding": r1 - r0,
            "delta_interval_CI_low": d1[0], "delta_interval_CI_high": d1[1],
            "delta_interval_CI_excludes_0": bool(d1[0] > 0 or d1[1] < 0),
            "r_beat_time": r2, "p_beat_time": p2,
            "r_beat_time_CI_low": c2[0], "r_beat_time_CI_high": c2[1],
            "r_beat_time_phase_min": float(np.nanmin(by_phase)),
            "r_beat_time_phase_median": float(np.nanmedian(by_phase)),
            "r_beat_time_phase_max": float(np.nanmax(by_phase)),
            "delta_beat_time": r2 - r0,
            "delta_beat_time_CI_low": d2[0], "delta_beat_time_CI_high": d2[1],
            "delta_beat_time_CI_excludes_0": bool(d2[0] > 0 or d2[1] < 0),
            "models_differ_abs": abs(r2 - r1),
            "N_Controls": int((grp == 0).sum()), "N_Patients": int((grp == 1).sum()),
            "N_Phases": n_phases, "Phase_Seed": SEED, "N_Boot": N_BOOT, "Boot_Seed": BOOT_SEED,
            "Step_s": STEP_S,
        })
    out = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    per_subject.to_csv(os.path.join(OUT_DIR, "Requantization_BeatTime_PerSubject.csv"),
                       index=False, float_format="%.8g")
    pp[["Record_ID", "Group", "Phase_Index", "Phase_s"] + MEASURES + DESCRIPTORS].to_csv(
        os.path.join(OUT_DIR, "Requantization_BeatTime_ByPhase.csv"), index=False, float_format="%.8g")
    out.to_csv(os.path.join(OUT_DIR, "Requantization_BeatTime.csv"), index=False, float_format="%.8g")

    tie_c = per_subject.loc[grp == 0, "Tie_Rate_BeatTime"].median()
    tie_p = per_subject.loc[grp == 1, "Tie_Rate_BeatTime"].median()
    print(f"  Median share of three-beat patterns with a tie under beat-time rounding: "
          f"controls {100 * tie_c:.1f}%, patients {100 * tie_p:.1f}%")
    print(f"  {'measure':22s}{'recorded':>10s}{'intervals':>11s}{'beat times':>12s}"
          f"{'change [95% CI]':>26s}{'phases':>18s}")
    for _, r in out.iterrows():
        print(f"  {r.Metric:22s}{r.r_recorded:>+10.3f}{r.r_interval_rounding:>+11.3f}{r.r_beat_time:>+12.3f}"
              f"   {r.delta_beat_time:+.3f} [{r.delta_beat_time_CI_low:+.3f}, {r.delta_beat_time_CI_high:+.3f}]"
              f"   {r.r_beat_time_phase_min:+.3f} to {r.r_beat_time_phase_max:+.3f}")
    print(f"  Saved 3 files to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="reproduce the stored interval-rounding columns only")
    ap.add_argument("--run", action="store_true", help="run the beat-time experiment")
    ap.add_argument("--phases", type=int, default=N_PHASES, help=f"phases per recording (default {N_PHASES})")
    a = ap.parse_args()
    if not (a.check or a.run):
        ap.print_help()
        return
    if a.phases < 1:
        sys.exit("--phases must be at least 1")
    if a.check and not a.run:
        check_interval_model(load_cohorts())
        return
    run(a.phases)


if __name__ == "__main__":
    main()
