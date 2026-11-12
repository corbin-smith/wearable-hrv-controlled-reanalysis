"""validate_outputs.py: internal-consistency checks across every analysis output."""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

TOL = 1e-6
RESULTS = {
    "healthy":   os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "psych":     os.path.join("outputs", "psychiatric",    "ecg_results_Psychiatric.csv"),
    "mitbih":    os.path.join("outputs", "healthy",        "ecg_results_Healthy.csv"),
}

_results = {"PASS": 0, "INFO": 0, "WARN": 0, "FAIL": 0}
_log = []


def record(level, check, detail):
    _results[level] += 1
    _log.append((level, check, detail))
    if level != "PASS":
        print(f"  [{level}] {check}: {detail}")


def bh(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    o = np.argsort(p)
    r = p[o]
    a = np.minimum.accumulate((r * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(a, 1.0)
    return out


def load(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None


def subjects_only(df, prefixes=("control_", "treatment_")):
    if df is None:
        return None
    keep = [i for i in df.index
            if any(str(i).startswith(p) for p in prefixes)
            or str(i).isdigit()]
    return df.loc[keep] if keep else df


def check_degenerate_in_families(outdir):
    print("\n[1] Degenerate columns inside FDR families")
    hit = False
    n_inspected = 0
    for f in glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True):
        d = load(f)
        if d is None:
            continue
        _padj_cols = [c for c in d.columns if str(c).endswith("p_adj")]
        if not _padj_cols:
            continue
        n_inspected += 1
        zero = set()
        _sd_cols = [c for c in ("Healthy_SD", "Psych_SD", "Control_SD",
                                "Patient_SD", "A_SD", "B_SD", "SD",
                                "Group_A_SD", "Group_B_SD")
                    if c in d.columns]
        for col in _sd_cols:
            sd = pd.to_numeric(d[col], errors="coerce")
            zero |= set(d.index[(sd.notna()) & (sd.abs() < 1e-12)])
        if not _sd_cols:
            record("INFO", "degenerate in FDR family",
                   f"{os.path.relpath(f, outdir)}: has p_adj but no "
                   f"recognisable SD column, so zero-variance rows cannot be "
                   f"detected here")
        if zero:
            record("FAIL", "degenerate in FDR family",
                   f"{os.path.relpath(f, outdir)}: {sorted(map(str, zero))} have zero "
                   f"variance in both groups but carry an adjusted p-value; "
                   f"remove from the family and report as a normalisation check")
            hit = True
    if not hit:
        if n_inspected == 0:
            record("WARN", "degenerate in FDR family",
                   "no table with an adjusted p-value column was found; this "
                   "check inspected nothing")
        else:
            record("PASS", "degenerate in FDR family",
                   f"none found across {n_inspected} tables")


def check_fdr_arithmetic(outdir):
    print("\n[2] BH-FDR arithmetic")
    any_checked = False
    for f in sorted(glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)):
        d = load(f)
        if d is None or not {"p_raw", "p_adj"} <= set(d.columns):
            continue
        praw = pd.to_numeric(d["p_raw"], errors="coerce")
        padj = pd.to_numeric(d["p_adj"], errors="coerce")
        ok = praw.notna() & padj.notna()
        if ok.sum() < 2:
            continue
        any_checked = True
        expect = bh(praw[ok].values)
        diff = np.abs(expect - padj[ok].values)
        bad = diff > 5e-4
        if bad.any():
            idx = list(np.asarray(d.index[ok])[bad])[:4]
            record("FAIL", "BH arithmetic",
                   f"{os.path.relpath(f, outdir)}: {int(bad.sum())} of {int(ok.sum())} "
                   f"adjusted p-values do not reconcile (worst {diff.max():.4f}); "
                   f"e.g. {idx}")
        else:
            record("PASS", "BH arithmetic", os.path.basename(f))
        if (padj[ok].values < praw[ok].values - 1e-9).any():
            record("FAIL", "BH monotonicity",
                   f"{os.path.basename(f)}: an adjusted p is below its raw p")
    if not any_checked:
        record("WARN", "BH arithmetic", "no table with both p_raw and p_adj found")


def check_proportions_are_integral(outdir):
    print("\n[3] Classification rates consistent with sample size")
    f = os.path.join(outdir, "comparison", "Classification_Summary_NestedCV.csv")
    if not os.path.exists(f):
        cands = glob.glob(os.path.join(outdir, "**",
                                       "Classification_Summary_NestedCV.csv"),
                          recursive=True)
        f = cands[0] if cands else None
    if not f:
        record("WARN", "classification rates", "summary file not found")
        return
    d = load(f)
    if d is None:
        record("WARN", "classification rates", "summary unreadable")
        return
    n_total, n_per = 60, 30
    problems = []
    for _, row in d.iterrows():
        for col, denom in (("Accuracy", n_total), ("Sensitivity", n_per),
                           ("Specificity", n_per)):
            if col not in d.columns:
                continue
            v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if not np.isfinite(v):
                continue
            s = str(row[col])
            dec = len(s.split(".")[1]) if "." in s else 0
            half = 0.5 * (10 ** -dec) if dec else 0.0
            lo, hi = (v - half) * denom, (v + half) * denom
            if np.floor(hi) < np.ceil(lo):
                problems.append(f"{row.get('Model', '?')}: {col}={v} implies "
                                f"{lo:.2f}-{hi:.2f} of {denom}, which contains "
                                f"no whole number of subjects")
    if problems:
        record("FAIL", "classification rates",
               "; ".join(problems[:3]) +
               "  -> the model did not run on the stated sample")
    else:
        record("PASS", "classification rates",
               f"all rates are whole counts over N={n_total}/{n_per}")

MULTI_ROW_BY_DESIGN = {
    "Embedding_Sensitivity.csv":   "primary vs selected embedding",
    "MSE_Scale_Audit.csv":         "all retained vs common scale set",
    "Quantization_Control.csv":    "one row per control condition",
    "Beta_Band_Sweep.csv":         "one row per fitted band",
    "PSD_Beta_Range_Sensitivity.csv": "one row per range variant",
    "Window_Skip_Sensitivity.csv": "one row per skip length",
    "Entropy_Tolerance_Sensitivity.csv": "one row per tolerance",
    "Scale_Band_Correspondence.csv": "one row per group and exponent",
    "Length_vs_HeartRate.csv":     "one row per adjustment condition",
}


def check_effect_size_agreement(outdir):
    print("\n[4] Effect-size agreement across tables")
    seen = {}
    for f in sorted(glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)):
        base = os.path.basename(f)
        subdir = os.path.relpath(os.path.dirname(f), outdir)
        if base.startswith(("Stationary_Reanalysis", "Quantization",
                            "Embedding_Sensitivity", "MSE_Scale_Audit",
                            "Scale_Band_Correspondence",
                            "Diagnosis_", "Quetiapine", "Comorbidity",
                            "AgeSex_", "Age_", "Beta_Band", "CrossDevice",
                            "Control_vs_Control", "Control_Equivalence",
                            "CrossControl", "ThreeGroup", "Mayer_",
                            "Length_Respiration", "Surrogate_",
                            "Beta_Activity", "Mobility_", "PSD_Beta_",
                            "Length_vs_HeartRate", "Window_Skip_Sensitivity",
                            "Activity_Structure", "Deterministic_Checks")):
            continue
        d = load(f)
        if d is None or "r_effect" not in d.columns:
            continue
        keycol = "Metric" if "Metric" in d.columns else None
        for i, row in d.iterrows():
            key = (subdir, str(row[keycol]) if keycol else str(i))
            r = pd.to_numeric(pd.Series([row["r_effect"]]), errors="coerce").iloc[0]
            if not np.isfinite(r):
                continue
            seen.setdefault(key, []).append((base, r))
    conflicts = []
    for key, vals in seen.items():
        rs = [abs(v[1]) for v in vals]
        if len(rs) > 1 and (max(rs) - min(rs)) > 0.02:
            conflicts.append(f"{key[0]}/{key[1]}: " +
                             ", ".join(f"{b}={r:+.3f}" for b, r in vals))
    if conflicts:
        for c in conflicts[:5]:
            record("WARN", "effect-size disagreement", c)
    else:
        record("PASS", "effect-size agreement",
               f"{len(seen)} metrics consistent across tables")


def check_percent_differences(outdir):
    print("\n[5] Percentage differences match reported means")
    bad = []
    for f in sorted(glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)):
        d = load(f)
        if d is None or "Pct_Diff" not in d.columns:
            continue
        if "Pct_Diff_Basis" in d.columns:
            _basis = set(str(v) for v in d["Pct_Diff_Basis"].dropna().unique())
            if _basis and not _basis <= {"mean"}:
                if {"Healthy_Median", "Psych_Median"} <= set(d.columns):
                    _hm = pd.to_numeric(d["Healthy_Median"], errors="coerce")
                    _pm = pd.to_numeric(d["Psych_Median"], errors="coerce")
                    _pdc = pd.to_numeric(d.get("Pct_Diff"), errors="coerce")
                    _exp = 100.0 * (_pm - _hm) / _hm.replace(0, np.nan)
                    _bad = int(((_exp - _pdc).abs() > 0.5).sum())
                    if _bad:
                        record("WARN", "percent difference",
                               f"{os.path.relpath(f, outdir)}: {_bad} rows do "
                               f"not match a MEDIAN-based percentage either")
                continue
        if not {"Healthy_Mean", "Psych_Mean"} <= set(d.columns):
            continue
        h = pd.to_numeric(d["Healthy_Mean"], errors="coerce")
        p = pd.to_numeric(d["Psych_Mean"], errors="coerce")
        pd_col = pd.to_numeric(d["Pct_Diff"], errors="coerce")
        expect = 100.0 * (p - h) / h.replace(0, np.nan)
        diff = (expect - pd_col).abs()
        off = diff > 0.15
        if off.any():
            bad.append(f"{os.path.basename(f)}: {int(off.sum())} rows differ by "
                       f"up to {diff.max():.2f} percentage points")
    if bad:
        for b in bad:
            record("WARN", "percent difference", b +
                   " (expected if computed from unrounded means; state so)")
    else:
        record("PASS", "percent difference", "consistent with reported means")


def check_cohorts():
    print("\n[6] Cohort sizes and metadata completeness")
    expect = {"healthy": 30, "psych": 30, "mitbih": 18}
    for k, path in RESULTS.items():
        d = load(path)
        if d is None:
            record("WARN", "cohort file", f"{k}: {path} not found")
            continue
        s = subjects_only(d)
        n = len(s)
        if n != expect[k]:
            record("FAIL", "cohort size",
                   f"{k}: {n} subjects, expected {expect[k]}")
        else:
            record("PASS", "cohort size", f"{k}: {n}")
        for col in ("Age", "Sex"):
            if col in s.columns:
                miss = pd.to_numeric(s[col], errors="coerce").isna().sum() \
                       if col == "Age" else s[col].isna().sum()
                if miss and k != "mitbih":
                    record("WARN", "metadata completeness",
                           f"{k}: {miss} subjects missing {col}")


def check_construction_invariants():
    print("\n[7] Quantities fixed by construction")
    for k, path in RESULTS.items():
        d = subjects_only(load(path))
        if d is None:
            continue
        for col, target in (("RQA_REC", 0.10), ("RNA_MeanDegree", 0.10)):
            if col not in d.columns:
                continue
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            if v.empty:
                continue
            dev = (v - target).abs().max()
            if dev > 5e-3:
                record("FAIL", "construction invariant",
                       f"{k}.{col}: max deviation {dev:.2e} from {target}; "
                       f"thresholding or normalisation is wrong")
            else:
                record("PASS", "construction invariant",
                       f"{k}.{col} = {target} (max dev {dev:.1e})")
        if {"HRV_SD1", "HRV_RMSSD"} <= set(d.columns):
            a = pd.to_numeric(d["HRV_SD1"], errors="coerce")
            b = pd.to_numeric(d["HRV_RMSSD"], errors="coerce") / np.sqrt(2)
            dev = (a - b).abs().max()
            if dev > 1e-6:
                record("FAIL", "construction invariant",
                       f"{k}: SD1 != RMSSD/sqrt(2), max dev {dev:.2e}")
            else:
                record("PASS", "construction invariant", f"{k}: SD1 identity holds")


def check_table_shapes(outdir):
    print("\n[8] Output table shapes")
    empty = []
    for f in sorted(glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)):
        d = load(f)
        if d is None:
            record("WARN", "unreadable", os.path.basename(f))
            continue
        if len(d) == 0:
            empty.append(os.path.basename(f))
    if empty:
        record("FAIL", "empty output", ", ".join(empty))
    else:
        record("PASS", "output shapes", "no empty tables")

EXPECTED_ALL_NAN = {
    "ICD_All", "ICD_Primary", "Comorbid_Codes",
    "MFDFA_Asymmetry",
    "Stat_ADF_p", "Stat_KPSS_p",
}


def check_unexpected_nan_columns():
    print("\n[9] Unexpected all-NaN columns")
    for k, path in RESULTS.items():
        d = subjects_only(load(path))
        if d is None:
            continue
        bad = [c for c in d.columns
               if d[c].isna().all() and c not in EXPECTED_ALL_NAN]
        if bad:
            record("WARN", "all-NaN column", f"{k}: {bad[:6]}")
        else:
            record("PASS", "all-NaN column", f"{k}: none unexpected")


def check_manuscript_table_currency(outdir):
    print("\n[10] Manuscript table currency")
    cands = sorted(glob.glob(os.path.join(outdir, "**", "TABLE1_MANUSCRIPT.md"),
                             recursive=True))
    if len(cands) > 1:
        record("WARN", "manuscript table",
               f"{len(cands)} copies found: {[os.path.relpath(c, outdir) for c in cands]}. "
               f"Checking {os.path.relpath(cands[0], outdir)}; delete the others "
               f"so there is one source of truth.")
    tab = cands[0] if cands else None
    if tab is None:
        record("WARN", "manuscript table",
               "TABLE1_MANUSCRIPT.md not found; run comparison.py to generate it")
        return
    scands = sorted(glob.glob(os.path.join(outdir, "**",
                                          "group_comparison_stats.csv"),
                              recursive=True))
    stats = load(scands[0]) if scands else None
    if stats is None:
        record("WARN", "manuscript table", "group_comparison_stats.csv not found")
        return

    text = open(tab, encoding="utf-8").read()
    rows = [ln for ln in text.splitlines()
            if ln.startswith("| ") and "---" not in ln
            and not ln.startswith("| Metric ")]
    n_rows = len(rows)
    n_sig_tab = sum(1 for ln in rows if ln.rstrip().endswith("** |"))
    n_stats = len(stats)
    n_sig_stats = int(stats["Significant"].sum()) if "Significant" in stats else -1

    if n_rows != n_stats:
        record("FAIL", "manuscript table",
               f"table has {n_rows} rows, statistics file has {n_stats}")
    elif n_sig_tab != n_sig_stats:
        record("FAIL", "manuscript table",
               f"table marks {n_sig_tab} significant, statistics file has "
               f"{n_sig_stats}")
    else:
        record("PASS", "manuscript table",
               f"{n_rows} rows, {n_sig_tab} significant, matches statistics")

    m = re.search(r"all (\d+) primary metrics", text)
    if m and int(m.group(1)) != n_stats:
        record("FAIL", "manuscript table",
               f"caption says {m.group(1)} metrics, file has {n_stats}")

COVARIATE_CANDIDATES = ["Num_RR", "HRV_MeanHR", "HRV_RMSSD", "HF_Power_ms2",
                        "Age", "mobility_mg", "Series_N", "HRV_MeanNN"]

IDENTITY_CLUSTER = {"Num_RR", "Series_N", "HRV_MeanHR", "HRV_MeanNN"}


def _models_with_covariates(outdir):
    found = []
    for path in glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for col in ("Variable", "Analysis", "Comparison"):
            if col not in df.columns:
                continue
            for label in df[col].dropna().astype(str):
                if "|" not in label:
                    continue
                covs = [c.strip() for c in label.split("|", 1)[1].split("+")]
                covs = [c for c in covs if c]
                if len(covs) > 1:
                    found.append((os.path.basename(path), label, covs))
    return found

_LABEL_ALIASES = {
    "meanhr": "HRV_MeanHR", "mean hr": "HRV_MeanHR", "heart rate": "HRV_MeanHR",
    "meannn": "HRV_MeanNN", "mean nn": "HRV_MeanNN",
    "num_rr": "Num_RR", "beats": "Num_RR", "beat count": "Num_RR",
    "series_n": "Series_N", "length": "Num_RR", "recording length": "Num_RR",
}


def _canonical(cov):
    key = cov.strip().lower()
    return _LABEL_ALIASES.get(key, cov.strip())


def check_covariate_collinearity(outdir="outputs"):
    print("\n[11] Covariate collinearity")

    frames = []
    for k, path in RESULTS.items():
        d = subjects_only(load(path))
        if d is not None:
            frames.append(d)
    identity_pairs = []
    if frames:
        d = pd.concat(frames, axis=0, join="outer")
        cols = [c for c in COVARIATE_CANDIDATES if c in d.columns]
        num = d[cols].apply(pd.to_numeric, errors="coerce")
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                ok = num[a].notna() & num[b].notna()
                if ok.sum() < 10:
                    continue
                r = num.loc[ok, a].corr(num.loc[ok, b], method="spearman")
                if abs(r) >= 0.95:
                    identity_pairs.append((a, b, r))
        if identity_pairs:
            defn = [p for p in identity_pairs if {p[0], p[1]} <= IDENTITY_CLUSTER]
            emp = [p for p in identity_pairs if {p[0], p[1]} not in
                   [{x, y} for x, y, _ in defn]]
            record("INFO", "covariate collinearity structure",
                   f"{len(defn)} definitional and {len(identity_pairs) - len(defn)} "
                   f"empirical pair(s) at |r| >= 0.95. Expected: over a fixed "
                   f"window beats = 30 x heart rate and mean NN is its "
                   f"reciprocal. Reported, not failed; what matters is whether a "
                   f"model adjusts for two at once.")

    models = _models_with_covariates(outdir)
    if not models:
        record("WARN", "covariate collinearity",
               "no multi-covariate models found in the outputs to check")
        return
    offenders = 0
    for fname, label, covs in models:
        canon = {_canonical(c) for c in covs}
        clash = sorted(canon & IDENTITY_CLUSTER)
        if len(clash) > 1:
            offenders += 1
            record("FAIL", "covariate collinearity [SIMULTANEOUS ADJUSTMENT]",
                   f"{fname}: model '{label}' adjusts for {clash}, which are "
                   f"one variable over a fixed-duration window. Adjusting for "
                   f"both spends a degree of freedom on a redundant column and "
                   f"presents one control as two.")
    if offenders == 0:
        record("PASS", "covariate collinearity",
               f"{len(models)} multi-covariate model(s) checked; none adjusts "
               f"for two members of the identity cluster "
               f"{sorted(IDENTITY_CLUSTER)}")


def check_effect_size_admissibility(outdir):
    print("\n[12] Effect-size admissibility (rank-biserial lattice)")
    n_checked, n_bad = 0, 0
    for f in glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True):
        d = load(f)
        if d is None:
            continue
        _EXCLUDE_SUBSTR = ("_CI_", "_ci_", "CI_low", "CI_high", "partial",
                           "_adj", "change", "unadj")
        rcols = [c for c in d.columns
                 if c in ("r_effect", "rank_biserial_r")
                 and not any(x in c for x in _EXCLUDE_SUBSTR)]
        ncols = [(a, b) for a, b in (("N_Healthy", "N_Psych"),
                                     ("Healthy_N", "Psych_N"),
                                     ("N_control", "N_patient"),
                                     ("N_A", "N_B"), ("N_sz", "N_bp"))
                 if a in d.columns and b in d.columns]
        if not rcols:
            continue
        if len(ncols) != 1:
            if ncols:
                record("INFO", "effect-size admissibility",
                       f"{os.path.relpath(f, outdir)}: multiple N-column pairs "
                       f"present, pairing is ambiguous, not checked")
            continue
        na, nb = ncols[0]
        for _, row in d.iterrows():
            try:
                n1 = int(row[na]); n2 = int(row[nb])
            except Exception:
                continue
            if n1 < 2 or n2 < 2:
                continue
            step = 1.0 / (n1 * n2)
            for rc in rcols:
                v = pd.to_numeric(pd.Series([row.get(rc)]),
                                  errors="coerce").iloc[0]
                if not np.isfinite(v) or abs(v) > 1.0:
                    continue
                n_checked += 1
                _s = repr(float(v))
                if "e" in _s or "E" in _s:
                    _dec = 12
                else:
                    _dec = len(_s.split(".")[1]) if "." in _s else 0
                _tol = max(0.5 * (10.0 ** -_dec), 1e-9)
                nearest = round(v / step) * step
                if abs(v - nearest) > _tol:
                    n_bad += 1
                    record("FAIL", "effect-size admissibility",
                           f"{os.path.relpath(f, outdir)}: {rc}={v:.6g} is not "
                           f"attainable at N={n1}/{n2} (nearest admissible "
                           f"{nearest:.6g}); check for a transcription error")
    if n_checked == 0:
        record("WARN", "effect-size admissibility",
               "no table had both effect sizes and sample sizes")
    elif n_bad == 0:
        record("PASS", "effect-size admissibility",
               f"{n_checked} values all attainable")


def check_family_inventory_consistency(outdir):
    print("\n[13] Correction-family inventory consistency")
    inv_paths = sorted(glob.glob(os.path.join(outdir, "**",
                                              "Correction_Family_Inventory.csv"),
                                 recursive=True))
    if not inv_paths:
        record("WARN", "family inventory",
               "Correction_Family_Inventory.csv not found; run comparison.py")
        return
    inventories = {}
    for ip in inv_paths:
        inv = load(ip)
        if inv is None:
            continue
        declared = {}
        for _, row in inv.iterrows():
            try:
                declared[str(row["Family"])] = int(row["N_Tests"])
            except Exception:
                continue
        inventories[os.path.dirname(ip)] = declared
    bad = False
    n_tables = 0
    for f in glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True):
        d = load(f)
        if d is None or "FDR_Family" not in d.columns \
                or "FDR_Family_N" not in d.columns:
            continue
        n_tables += 1
        declared = inventories.get(os.path.dirname(f), {})
        for fam, grp in d.groupby("FDR_Family"):
            stated = pd.to_numeric(grp["FDR_Family_N"], errors="coerce")
            stated = int(stated.dropna().iloc[0]) if stated.notna().any() else None
            actual_rows = int(len(grp))
            if stated is not None and stated != actual_rows:
                record("FAIL", "family size vs rows",
                       f"{os.path.relpath(f, outdir)}: family '{fam}' states "
                       f"N={stated} but the table has {actual_rows} rows")
                bad = True
            if stated is not None and str(fam) in declared \
                    and declared[str(fam)] != stated:
                record("FAIL", "family size vs inventory",
                       f"{os.path.relpath(f, outdir)}: family '{fam}' states "
                       f"N={stated}, inventory says {declared[str(fam)]}")
                bad = True
    if not bad:
        record("PASS", "family inventory",
               f"{len(inventories)} inventories, {n_tables} family-labelled "
               f"tables, all consistent")


def check_output_freshness(outdir):
    print("\n[14] Output freshness (summaries vs available columns)")
    qc_paths = sorted(glob.glob(os.path.join(outdir, "**",
                                             "Quantization_Control.csv"),
                                recursive=True))
    stale = False
    for _qp in qc_paths:
        qc = load(_qp)
        if qc is None or "Metric" not in qc.columns:
            continue
        metrics_present = set(qc["Metric"].astype(str))
        _ctrl = load(RESULTS["mitbih"] if "mitbih" in _qp.lower()
                     else RESULTS["healthy"])
        for csv_name, cols in (
                ("ecg_results_Psychiatric.csv",
                 ("RNA_ClusteringCoeff_Dither", "MSE_Mean_Dither")),):
            for res in glob.glob(os.path.join(outdir, "**", csv_name),
                                 recursive=True):
                d = load(res)
                if d is None:
                    continue
                for col in cols:
                    base = col.split("_Dither")[0]
                    if _ctrl is None or col not in _ctrl.columns:
                        continue
                    if col in d.columns and base not in metrics_present:
                        record("FAIL", "stale summary",
                               f"per-subject results contain {col} but "
                               f"Quantization_Control.csv has no {base} row; "
                               f"re-run comparison.py AFTER the patch scripts")
                        stale = True
    if not qc_paths:
        record("WARN", "output freshness",
               "no Quantization_Control.csv found; nothing was checked")
    elif not stale:
        record("PASS", "output freshness",
               f"no stale summaries detected in {len(qc_paths)} quantization "
               f"table(s)")


def check_adjusted_p_resolution(outdir):
    print("\n[15] Adjusted p-value resolution")
    warned = False
    for f in glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True):
        d = load(f)
        if d is None or "p_raw" not in d.columns or "p_adj" not in d.columns:
            continue
        pr = pd.to_numeric(d["p_raw"], errors="coerce")
        pa = pd.to_numeric(d["p_adj"], errors="coerce")
        ok = pr.notna() & pa.notna() & (pr > 0)
        if ok.sum() < 2:
            continue
        ratio = (pa[ok] / pr[ok]).round(4)
        ranks = pr[ok].rank(method="first")
        implied = (ratio * ranks)
        frac = (implied - implied.round()).abs()
        if float(frac.median()) > 0.05:
            record("WARN", "p_adj resolution",
                   f"{os.path.relpath(f, outdir)}: family size is not "
                   f"recoverable from the printed p_raw/p_adj pairs; print "
                   f"more digits or state the family size in the caption")
            warned = True
    if not warned:
        record("PASS", "p_adj resolution", "family sizes reconstructable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on WARN as well as FAIL")
    args = ap.parse_args()

    print("=" * 70)
    print("  OUTPUT VALIDATION")
    print("=" * 70)

    check_degenerate_in_families(args.dir)
    check_fdr_arithmetic(args.dir)
    check_proportions_are_integral(args.dir)
    check_effect_size_agreement(args.dir)
    check_percent_differences(args.dir)
    check_cohorts()
    check_construction_invariants()
    check_table_shapes(args.dir)
    check_unexpected_nan_columns()
    check_manuscript_table_currency(args.dir)
    check_covariate_collinearity(args.dir)
    check_effect_size_admissibility(args.dir)
    check_family_inventory_consistency(args.dir)
    check_output_freshness(args.dir)
    check_adjusted_p_resolution(args.dir)

    print("\n" + "=" * 70)
    print(f"  PASS {_results['PASS']}   INFO {_results['INFO']}   "
          f"WARN {_results['WARN']}   FAIL {_results['FAIL']}")
    print("=" * 70)
    if _results["FAIL"]:
        print("\n  BLOCKING: do not interpret these results until every FAIL is")
        print("  resolved. Each one indicates the analysis did not do what the")
        print("  output claims it did.")
    elif _results["WARN"]:
        print("\n  No blocking failures. Review each WARN and confirm it is")
        print("  expected before reporting.")
    else:
        print("\n  All checks passed.")

    bad = _results["FAIL"] + (_results["WARN"] if args.strict else 0)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
