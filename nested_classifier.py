"""nested_classifier.py: classification of HRV-ACC controls against patients from the HRV measures."""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def _first_existing(cands, default=None):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return default if default is not None else cands[0]

HEALTHY_CSV = _first_existing([
    os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "ecg_results_HealthyACC.csv",
])
PSYCH_CSV = _first_existing([
    os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv"),
    "ecg_results_Psychiatric.csv",
])
OUTPUT_DIR = os.path.join("outputs", "classification")
QC_COLUMN = "QC_Overall_Pass"

PRIMARY_FEATURES = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_MeanHR",
    "HRV_SD1", "HRV_SD2", "HRV_SD1SD2",
    "LF_Power_ms2", "HF_Power_ms2", "Total_Power_ms2", "LF_HF_Ratio",
    "PSD_Beta",
    "SampEn", "ApEn", "PermEn", "MSE_Mean",
    "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2",
    "RQA_REC", "RQA_DET", "RQA_LAM",
]

MAX_MISSING_FRAC = 0.10

ARTIFACT_DRIVEN = ["SampEn", "ApEn", "RQA_DET", "MSE_Mean"]

EXCLUDE_METRICS = ["RQA_REC",
                   "MFDFA_Width", "MFDFA_Asymmetry", "RNA_MeanDegree"]

NON_FEATURE_SUBSTRINGS = ["QC", "Record", "SURR", "_R2", "Group",
                          "coverage", "mobility", "n_acc"]

METADATA_COLUMNS = ["Age", "Sex", "Diagnosis", "ICD_Primary", "ICD_All",
                    "Patient_ID", "Comorbid_Codes", "Comorbid_Substance",
                    "Quetiapine"]

DOMAIN_FEATURES = ["HRV_RMSSD", "PSD_Beta", "DFA_Alpha2", "SampEn", "RQA_DET"]

SELECT_ALPHA   = 0.05
MIN_FEATURES   = 3
LOGREG_C       = 1.0
N_PERMUTATIONS = 1000
RANDOM_SEED    = 42

N_SPLITS       = 5
N_REPEATS      = 100
N_BOOTSTRAP       = 2000
N_BOOTSTRAP_DELTA = 10000
LOGREG_C_SWEEP = (0.1, 1.0, 10.0)


def load_data():
    frames = []
    for path, grp in [(HEALTHY_CSV, 0), (PSYCH_CSV, 1)]:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Result CSV not found: {path}")
        df = pd.read_csv(path)
        idx = "Record_ID" if "Record_ID" in df.columns else df.columns[0]
        df = df.set_index(idx)
        df["Group"] = grp
        frames.append(df)
    data = pd.concat(frames)
    if QC_COLUMN in data.columns:
        data = data[data[QC_COLUMN] == True]                    # noqa: E712

    missing = [c for c in PRIMARY_FEATURES if c not in data.columns]
    if missing:
        raise KeyError(f"pre-specified features absent from the results: "
                       f"{missing}. Re-run the cohort scripts.")
    features = [c for c in PRIMARY_FEATURES
                if c not in EXCLUDE_METRICS and c not in METADATA_COLUMNS]
    leaked = [c for c in METADATA_COLUMNS if c in features]
    if leaked:
        raise RuntimeError(f"metadata columns reached the feature pool: "
                           f"{leaked}. These identify group membership and "
                           f"must never be used as predictors.")

    n_total = len(data)
    sparse = [c for c in features
              if data[c].isna().sum() > MAX_MISSING_FRAC * n_total]
    if sparse:
        print(f"  dropped {len(sparse)} feature(s) with >"
              f"{MAX_MISSING_FRAC:.0%} missing: {sparse}")
        features = [c for c in features if c not in sparse]

    def _degenerate(col):
        v = pd.to_numeric(data[col], errors="coerce")
        sd = v.std(ddof=1)
        mu = abs(v.mean())
        if not np.isfinite(sd):
            return True
        return sd == 0 or (mu > 0 and sd / mu < 1e-6)

    degenerate = [c for c in features if _degenerate(c)]
    if degenerate:
        print(f"  dropped {len(degenerate)} zero-variance feature(s): "
              f"{degenerate}")
        features = [c for c in features if c not in degenerate]

    before = len(data)
    data = data.dropna(subset=features + ["Group"])
    if len(data) < before:
        print(f"  complete-case filtering removed {before - len(data)} "
              f"subject(s); {len(data)} remain")
    if len(data) < 40:
        raise ValueError(
            f"only {len(data)} subjects survive complete-case filtering, out "
            f"of {before}. Do not interpret a classifier fitted to this. "
            f"Inspect per-feature missingness before proceeding.")
    print(f"  feature pool: {len(features)} pre-specified metrics, "
          f"N = {len(data)} subjects")
    y = data["Group"].to_numpy(dtype=int)
    return data, y, features


def _mwu_pvalues(Xtr, ytr):
    n = len(ytr)
    na = int(np.sum(ytr == 0))
    nb = int(np.sum(ytr == 1))
    if na == 0 or nb == 0:
        return np.ones(Xtr.shape[1])
    try:
        ranks = rankdata(Xtr, axis=0)
    except TypeError:
        ranks = np.apply_along_axis(rankdata, 0, Xtr)
    Rb = ranks[ytr == 1].sum(axis=0)
    U = Rb - nb * (nb + 1) / 2.0
    mu = na * nb / 2.0
    sigma = np.sqrt(na * nb * (n + 1) / 12.0)
    z = (U - mu) / sigma
    return 2.0 * norm.sf(np.abs(z))


def _select_features(Xtr, ytr, feature_names=None):
    pvals = _mwu_pvalues(Xtr, ytr)
    keep = np.where(pvals < SELECT_ALPHA)[0]
    if len(keep) < MIN_FEATURES:
        keep = np.argsort(pvals)[:MIN_FEATURES]
    return keep


def nested_loo(X, y, feature_names, do_selection=True, fixed_idx=None, seed=RANDOM_SEED):
    n = len(y)
    probs = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, ytr = X[tr], y[tr]
        Xte = X[i:i + 1]

        if fixed_idx is not None:
            cols = np.asarray(fixed_idx)
        elif do_selection:
            cols = _select_features(Xtr, ytr, feature_names)
        else:
            cols = np.arange(X.shape[1])

        scaler = StandardScaler().fit(Xtr[:, cols])
        clf = LogisticRegression(max_iter=2000, C=LOGREG_C, random_state=seed)
        clf.fit(scaler.transform(Xtr[:, cols]), ytr)
        probs[i] = clf.predict_proba(scaler.transform(Xte[:, cols]))[0, 1]
    return probs


def score(y, probs):
    pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs)
    acc = float(np.mean(pred == y))
    sens = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else np.nan
    spec = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else np.nan
    return auc, acc, sens, spec


def permutation_null(X, y, feature_names, observed_auc, n_perm=N_PERMUTATIONS,
                     seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = rng.permutation(y)
        null[k] = roc_auc_score(
            yp, nested_loo(X, yp, feature_names, do_selection=True, seed=seed))
        if (k + 1) % max(1, n_perm // 10) == 0:
            print(f"      permutation {k + 1}/{n_perm}", end="\r")
    print(" " * 40, end="\r")
    p = (np.sum(null >= observed_auc) + 1) / (n_perm + 1)
    return null, float(p)


def repeated_kfold_auc(X, y, feature_names, do_selection=True, fixed_idx=None,
                       n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=RANDOM_SEED):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    per_repeat_pred = np.full((n_repeats, len(y)), np.nan)
    fold_counter = 0
    aucs = []
    for tr_idx, te_idx in rskf.split(X, y):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        if len(np.unique(yte)) < 2:
            fold_counter += 1
            continue
        if fixed_idx is not None:
            cols = np.asarray(fixed_idx)
        elif do_selection:
            cols = _select_features(Xtr, ytr, feature_names)
        else:
            cols = np.arange(X.shape[1])
        scaler = StandardScaler().fit(Xtr[:, cols])
        clf = LogisticRegression(max_iter=2000, C=LOGREG_C, random_state=seed)
        clf.fit(scaler.transform(Xtr[:, cols]), ytr)
        prob = clf.predict_proba(scaler.transform(Xte[:, cols]))[:, 1]
        rep = fold_counter // n_splits
        if rep < n_repeats:
            per_repeat_pred[rep, te_idx] = prob
        fold_counter += 1

    for rep in range(n_repeats):
        pr = per_repeat_pred[rep]
        ok = np.isfinite(pr)
        if ok.sum() < len(y) or len(np.unique(y[ok])) < 2:
            continue
        aucs.append(roc_auc_score(y[ok], pr[ok]))
    aucs = np.asarray(aucs, dtype=float)
    if aucs.size == 0:
        return aucs, np.nan, np.nan, np.nan
    return (aucs, float(aucs.mean()),
            float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def bootstrap_auc_ci(X, y, feature_names, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED,
                     do_selection=True, fixed_idx=None):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 8 or len(np.unique(y[idx])) < 2 or len(np.unique(y[oob])) < 2:
            continue
        Xtr, ytr = X[idx], y[idx]
        if fixed_idx is not None:
            cols = np.asarray(fixed_idx)
        elif do_selection:
            cols = _select_features(Xtr, ytr, feature_names)
        else:
            cols = np.arange(X.shape[1])
        try:
            sc = StandardScaler().fit(Xtr[:, cols])
            clf = LogisticRegression(max_iter=2000, C=LOGREG_C,
                                     random_state=seed)
            clf.fit(sc.transform(Xtr[:, cols]), ytr)
            pr = clf.predict_proba(sc.transform(X[oob][:, cols]))[:, 1]
            aucs.append(roc_auc_score(y[oob], pr))
        except Exception:
            continue
    a = np.asarray(aucs, dtype=float)
    if a.size < 50:
        return np.nan, np.nan, np.nan, int(a.size)
    return (float(a.mean()), float(np.percentile(a, 2.5)),
            float(np.percentile(a, 97.5)), int(a.size))


def bootstrap_delta_auc(X, y, i_base, i_full, n_boot=N_BOOTSTRAP_DELTA,
                        seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    i_base = np.asarray(i_base)
    i_full = np.asarray(i_full)
    deltas, auc_base, auc_full = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if (len(oob) < 8 or len(np.unique(y[idx])) < 2
                or len(np.unique(y[oob])) < 2):
            continue
        try:
            aucs = []
            for cols in (i_base, i_full):
                sc = StandardScaler().fit(X[idx][:, cols])
                clf = LogisticRegression(max_iter=2000, C=LOGREG_C,
                                         random_state=seed)
                clf.fit(sc.transform(X[idx][:, cols]), y[idx])
                pr = clf.predict_proba(sc.transform(X[oob][:, cols]))[:, 1]
                aucs.append(roc_auc_score(y[oob], pr))
            deltas.append(aucs[1] - aucs[0])
            auc_base.append(aucs[0])
            auc_full.append(aucs[1])
        except ValueError:
            continue
    deltas = np.asarray(deltas, dtype=float)
    nan3 = {"mean": np.nan, "lo": np.nan, "hi": np.nan}
    if deltas.size == 0:
        return np.nan, np.nan, np.nan, np.nan, {"base": nan3, "full": nan3, "n": 0}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_two = 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))

    def _summary(v):
        v = np.asarray(v, dtype=float)
        l, h = np.percentile(v, [2.5, 97.5])
        return {"mean": float(v.mean()), "lo": float(l), "hi": float(h)}

    oob = {"base": _summary(auc_base), "full": _summary(auc_full), "n": int(deltas.size)}
    return float(deltas.mean()), float(lo), float(hi), min(p_two, 1.0), oob


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data, y, features = load_data()
    X = data[features].to_numpy(dtype=float)
    print(f"  {len(y)} subjects ({int(np.sum(y == 0))} control, "
          f"{int(np.sum(y == 1))} patient); {len(features)} candidate features.")

    rows = []

    print("\n  [1] Nested feature-selection model (honest replacement)...")
    probs1 = nested_loo(X, y, features, do_selection=True)
    auc1, acc1, sens1, spec1 = score(y, probs1)
    n_sel = int(np.median([len(_select_features(X[np.arange(len(y)) != i],
                                                y[np.arange(len(y)) != i], features))
                           for i in range(len(y))]))
    print(f"      AUC {auc1:.3f}, acc {acc1:.1%}, sens {sens1:.1%}, "
          f"spec {spec1:.1%}  (median {n_sel} features/fold)")
    frac_kept = n_sel / float(len(features)) if features else np.nan
    print(f"      selection kept {n_sel}/{len(features)} features "
          f"({frac_kept:.0%}) -> selection is "
          f"{'NOT load-bearing; nesting is largely moot here' if frac_kept > 0.8 else 'load-bearing'}")

    print("\n  [1a] Repeated stratified 5-fold (headline estimator)...")
    aucs_k, m_k, lo_k, hi_k = repeated_kfold_auc(X, y, features,
                                                 do_selection=True)
    print(f"      AUC {m_k:.3f}  95% interval [{lo_k:.3f}, {hi_k:.3f}] "
          f"over {len(aucs_k)} folds ({N_REPEATS} repeats x {N_SPLITS} folds)")
    print("      NOTE: the interval above is the spread across repeats of the")
    print("      resampling scheme on the SAME 60 subjects. It describes Monte")
    print("      Carlo variability, not uncertainty about AUC, and converges to a")
    print("      point as repeats increase. The bootstrap interval below is the")
    print("      one to report.")
    m_b, lo_b, hi_b, n_b = bootstrap_auc_ci(X, y, features, do_selection=True)
    print(f"      BOOTSTRAP over subjects: AUC {m_b:.3f} "
          f"95% CI [{lo_b:.3f}, {hi_b:.3f}] ({n_b} replicates)")
    rows.append({"Model": "Nested feature-selection (bootstrap over subjects)",
                 "AUC": round(m_b, 3), "AUC_lo95": round(lo_b, 3),
                 "AUC_hi95": round(hi_b, 3), "Accuracy": np.nan,
                 "Sensitivity": np.nan, "Specificity": np.nan,
                 "Median_features": n_sel, "Perm_null_AUC_mean": np.nan,
                 "Perm_p": np.nan,
                 "Note": "REPORT THIS INTERVAL; full pipeline inside each "
                         "bootstrap replicate, scored out-of-bag"})

    af = [c for c in features if c not in ARTIFACT_DRIVEN]
    if len(af) >= 5 and len(af) < len(features):
        idx_af = np.array([features.index(c) for c in af])
        probs_af = nested_loo(X, y, features, do_selection=False, fixed_idx=idx_af)
        auc_af, acc_af, sens_af, spec_af = score(y, probs_af)
        m_af, lo_af, hi_af, _ = bootstrap_auc_ci(X, y, features,
                                                 do_selection=False,
                                                 fixed_idx=idx_af)
        print(f"\n  [1c] Artifact-free features only "
              f"({len(af)} of {len(features)}; excluded {ARTIFACT_DRIVEN})")
        print(f"      LOO AUC {auc_af:.3f}   bootstrap {m_af:.3f} "
              f"95% CI [{lo_af:.3f}, {hi_af:.3f}]")
        print("      Reported because the full model contains metrics this study")
        print("      shows are driven by device quantization and recording")
        print("      length; that model partly indexes the instrument.")
        rows.append({"Model": "Artifact-free features (LOO)",
                     "AUC": round(auc_af, 3), "AUC_lo95": round(lo_af, 3),
                     "AUC_hi95": round(hi_af, 3), "Accuracy": round(acc_af, 3),
                     "Sensitivity": round(sens_af, 3),
                     "Specificity": round(spec_af, 3),
                     "Median_features": len(af),
                     "Perm_null_AUC_mean": np.nan, "Perm_p": np.nan,
                     "Note": "excludes SampEn, ApEn, RQA_DET, MSE_Mean"})

    rows.append({"Model": "Nested feature-selection (repeated 5-fold)",
                 "AUC": round(m_k, 3), "AUC_lo95": round(lo_k, 3),
                 "AUC_hi95": round(hi_k, 3), "Accuracy": np.nan,
                 "Sensitivity": np.nan, "Specificity": np.nan,
                 "Median_features": n_sel,
                 "Perm_null_AUC_mean": np.nan, "Perm_p": np.nan,
                 "Note": "interval is across-repeat spread, NOT a confidence "
                         "interval; see the bootstrap row"})

    print(f"  [1b] Permutation test ({N_PERMUTATIONS} shuffles, applied to the "
          f"pooled LEAVE-ONE-OUT estimator)...")
    null, p_perm = permutation_null(X, y, features, auc1)
    print(f"      shuffled-label AUC = {null.mean():.3f} +/- {null.std():.3f}; "
          f"empirical p = {p_perm:.4f}")
    if null.mean() < 0.48:
        print("      NOTE: the null sits below 0.5. That is the known negative "
              "bias of pooled LOO AUC (each held-out prediction comes from a "
              "model whose class balance is shifted by removing exactly that "
              "observation). Report it as such, NOT as proof of no leakage.")
    rows.append({"Model": "Nested feature-selection (LOO)", "AUC": round(auc1, 3),
                 "Accuracy": round(acc1, 3), "Sensitivity": round(sens1, 3),
                 "Specificity": round(spec1, 3),
                 "Median_features": n_sel,
                 "Perm_null_AUC_mean": round(float(null.mean()), 3),
                 "Perm_p": round(p_perm, 4),
                 "Note": "honest replacement for the leaky 23-feature model"})

    dom = [f for f in DOMAIN_FEATURES if f in features]
    if len(dom) == len(DOMAIN_FEATURES):
        print("\n  [2] Fixed 5-domain model (no selection, nothing to leak)...")
        idx = [features.index(f) for f in dom]
        probs2 = nested_loo(X, y, features, do_selection=False, fixed_idx=idx)
        auc2, acc2, sens2, spec2 = score(y, probs2)
        _, m2, lo2, hi2 = repeated_kfold_auc(X, y, features,
                                             do_selection=False, fixed_idx=idx)
        print(f"      LOO AUC {auc2:.3f}, acc {acc2:.1%}, sens {sens2:.1%}, "
              f"spec {spec2:.1%}")
        print(f"      repeated 5-fold AUC {m2:.3f} "
              f"95% interval [{lo2:.3f}, {hi2:.3f}]")
        print("      NOTE: this pre-specified model is the recommended headline "
              "classification result. It has no selection step at all, so there "
              "is nothing to leak and nothing to explain.")
        rows.append({"Model": "Fixed 5-domain (LOO)", "AUC": round(auc2, 3),
                     "Accuracy": round(acc2, 3), "Sensitivity": round(sens2, 3),
                     "Specificity": round(spec2, 3), "Median_features": len(dom),
                     "Perm_null_AUC_mean": np.nan, "Perm_p": np.nan,
                     "Note": "features: " + ", ".join(dom)})
    else:
        print("\n  [2] Fixed 5-domain model skipped (missing columns: "
              f"{[f for f in DOMAIN_FEATURES if f not in features]})")

    base = ["HRV_MeanHR", "HRV_RMSSD"]
    if all(f in features for f in base) and "PSD_Beta" in features:
        print("\n  [3] Incremental value of PSD beta...")
        i_base = [features.index(f) for f in base]
        i_full = [features.index(f) for f in base + ["PSD_Beta"]]
        pb = nested_loo(X, y, features, do_selection=False, fixed_idx=i_base)
        pf = nested_loo(X, y, features, do_selection=False, fixed_idx=i_full)
        auc_b = roc_auc_score(y, pb)
        auc_f = roc_auc_score(y, pf)
        d_mean, d_lo, d_hi, d_p, oob = bootstrap_delta_auc(X, y, i_base, i_full)
        print(f"      MeanHR+RMSSD AUC {auc_b:.3f}  ->  +PSD beta AUC {auc_f:.3f} "
              f"(delta {auc_f - auc_b:+.3f})")
        print(f"      bootstrap delta-AUC {d_mean:+.3f} "
              f"95% interval [{d_lo:+.3f}, {d_hi:+.3f}], p = {d_p:.3f}")
        print(f"      out-of-bag AUC, MeanHR+RMSSD: {oob['base']['mean']:.3f} "
              f"[{oob['base']['lo']:.3f}, {oob['base']['hi']:.3f}]; with PSD beta: "
              f"{oob['full']['mean']:.3f} [{oob['full']['lo']:.3f}, {oob['full']['hi']:.3f}] "
              f"({oob['n']} replicates)")
        if d_lo <= 0 <= d_hi:
            print("      The interval includes zero: the increment is NOT "
                  "distinguishable from zero at this sample size. Report it "
                  "with the interval or not at all.")
        rows.append({"Model": "MeanHR + RMSSD (LOO)", "AUC": round(auc_b, 3),
                     "Accuracy": np.nan, "Sensitivity": np.nan,
                     "Specificity": np.nan, "Median_features": 2,
                     "Perm_null_AUC_mean": np.nan, "Perm_p": np.nan,
                     "AUC_lo95": round(oob["base"]["lo"], 3),
                     "AUC_hi95": round(oob["base"]["hi"], 3),
                     "AUC_oob_mean": round(oob["base"]["mean"], 3),
                     "Note": "baseline for incremental test; AUC is leave-one-out, "
                             "the interval and AUC_oob_mean are out-of-bag bootstrap"})
        rows.append({"Model": "MeanHR + RMSSD + PSD_Beta (LOO)",
                     "AUC": round(auc_f, 3), "Accuracy": np.nan,
                     "Sensitivity": np.nan, "Specificity": np.nan,
                     "Median_features": 3, "Perm_null_AUC_mean": np.nan,
                     "Perm_p": np.nan,
                     "AUC_lo95": round(oob["full"]["lo"], 3),
                     "AUC_hi95": round(oob["full"]["hi"], 3),
                     "AUC_oob_mean": round(oob["full"]["mean"], 3),
                     "Delta_AUC": round(auc_f - auc_b, 3),
                     "Delta_AUC_lo95": round(d_lo, 3),
                     "Delta_AUC_hi95": round(d_hi, 3),
                     "Delta_AUC_p": round(d_p, 4),
                     "Note": f"delta AUC = {auc_f - auc_b:+.3f} vs baseline; "
                             f"bootstrap 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]"})

    _prov = [
        ("scaling_fitted_where", "inside each fold, on training data only"),
        ("feature_selection_where", "inside each fold"),
        ("bootstrap_procedure",
         "full pipeline re-run per replicate, scored out-of-bag"),
        ("bootstrap_replicates_auc_ci", N_BOOTSTRAP),
        ("bootstrap_replicates_delta_auc", N_BOOTSTRAP_DELTA),
        ("bootstrap_counts_differ_why",
         "the AUC interval re-runs feature selection per replicate and is "
         "stable at 2,000; the delta interval uses fixed feature sets, is "
         "cheap per replicate, and carries a negative conclusion, so it is "
         "run at 10,000"),
        ("permutation_procedure",
         "entire nested pipeline re-run on every shuffled-label replicate"),
        ("permutation_replicates", N_PERMUTATIONS),
        ("permutation_p_convention", "(count + 1) / (n_perm + 1)"),
        ("permutation_p_resolution",
         f"cannot resolve below {1.0 / (N_PERMUTATIONS + 1):.4f}; "
         f"report as p <= 0.001 rather than an exact value"),
        ("regularisation_C", LOGREG_C),
        ("regularisation_C_chosen", "fixed a priori; not tuned"),
        ("random_seed", RANDOM_SEED),
        ("caveat_unsupervised_preprocessing",
         "missingness screening, zero-variance removal and complete-case "
         "filtering are applied to the pooled data before cross-validation. "
         "No outcome information is used, so this is not label leakage, but "
         "it is not fully nested either"),
        ("caveat_artifact_free_features",
         "the excluded feature set for the artifact-free model is "
         "data-derived rather than pre-registered; the direction of the "
         "choice is conservative"),
    ]
    pd.DataFrame(_prov, columns=["Item", "Value"]).to_csv(
        os.path.join(OUTPUT_DIR, "Classification_Provenance.csv"),
        index=False)
    print(f"  Classifier provenance → "
          f"{OUTPUT_DIR}/Classification_Provenance.csv")

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "Classification_Summary_NestedCV.csv"),
        index=False)
    print(f"\n  Saved results to {OUTPUT_DIR}/Classification_Summary_NestedCV.csv")


if __name__ == "__main__":
    print("[classification] honest nested cross-validation")
    main()
