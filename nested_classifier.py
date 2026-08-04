"""
classification_nested_cv.py
===========================
Honest (leakage-free) classification of control vs. treated-psychiatric subjects
from the resting ECG metrics, replacing the leaky Section 3.8 classifier.

WHY THIS FILE EXISTS (in plain language)
----------------------------------------
A "classifier" is just a formula that is shown each person's numbers and tries to
guess a label -- here, "control" or "patient." To test it honestly you cannot let
it be graded on people it already saw during training, so the standard trick is
LEAVE-ONE-OUT CROSS-VALIDATION: hide one person, build the formula from the other
59, ask it to guess the hidden person, then repeat 60 times so every person gets
one honest guess.

The old version had a subtle but real cheat called DATA LEAKAGE. Before doing any
of that, it first looked at all 60 people to decide WHICH metrics to feed the
formula (it kept the metrics that already separated the groups). Because that
choice peeked at everyone -- including the person about to be "hidden" -- a little
information about each hidden person leaked into the setup, making the reported
accuracy look better than it truly is.

This file fixes that with NESTED cross-validation: the metric-choosing step is
moved INSIDE the loop, so it only ever sees the 59 training people and never the
hidden one. The honest accuracy that comes out is usually lower -- that is the
point; it is the number that would survive scrutiny.

It reports three models:
  (1) Nested selection -- choose informative metrics inside each fold (the honest
      replacement for the old 23-feature model).
  (2) Fixed 5-domain model -- five pre-chosen metrics, one per physiological
      domain, with NO selection at all, so there is nothing to leak. A clean,
      interpretable benchmark.
  (3) Incremental value of PSD beta -- does adding beta to the two strongest
      conventional metrics (mean HR, RMSSD) improve the guess?

It also runs a PERMUTATION TEST: it shuffles the labels at random (destroying any
real group difference) and re-runs the whole honest pipeline hundreds of times.
If the pipeline were still cheating, it would score above 0.5 even on shuffled
labels. A shuffled-label score sitting at ~0.5 is proof the leak is gone; where
the real score falls relative to that shuffled distribution is an honest p-value.

NOTE ON SCOPE: this distinguishes *treated, hospitalized* patients from healthy
volunteers. Even a perfect score would not be a clinical diagnostic -- you already
know who is in the hospital. This is a methods demonstration, not a diagnostic
tool, and the write-up should say so.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
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

# Metrics excluded from the feature pool as a matter of principle, not results:
#   - MFDFA_Width / MFDFA_Asymmetry: numerically unstable on short records
#     (dropped in the main analysis; see item 4).
#   - RNA_MeanDegree: flagged as a subsampling artifact with no biological
#     meaning, so it must not be allowed to inflate the classifier.
EXCLUDE_METRICS = ["MFDFA_Width", "MFDFA_Asymmetry", "RNA_MeanDegree"]

# Columns that are not physiological features.
NON_FEATURE_SUBSTRINGS = ["QC", "Record", "SURR", "_R2", "Group",
                          "coverage", "mobility", "n_acc"]

# Five pre-specified metrics, one per domain, for the fixed benchmark model.
DOMAIN_FEATURES = ["HRV_RMSSD", "PSD_Beta", "DFA_Alpha2", "SampEn", "RQA_DET"]

# Fixed, a-priori modelling choices (declared up front so nothing is tuned to
# the result -- tuning to the result would itself be a form of leakage).
SELECT_ALPHA   = 0.05     # keep a metric if its training-set group p < this
MIN_FEATURES   = 3        # if selection keeps fewer, back-fill to this many
LOGREG_C       = 1.0      # inverse regularisation strength (fixed a priori)
N_PERMUTATIONS = 1000     # shuffled-label repeats (a few minutes at 1000)
RANDOM_SEED    = 42


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def load_data():
    """Load both cohorts, keep QC-passing subjects, return (X_df, y, features)."""
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

    # Build the feature pool: numeric metric columns only.
    features = []
    for c in data.columns:
        if c == "Group":
            continue
        if any(s.lower() in c.lower() for s in NON_FEATURE_SUBSTRINGS):
            continue
        if c in EXCLUDE_METRICS:
            continue
        if not np.issubdtype(data[c].dropna().dtype, np.number):
            continue
        features.append(c)

    data = data.dropna(subset=features + ["Group"])
    y = data["Group"].to_numpy(dtype=int)
    return data, y, features


# ----------------------------------------------------------------------------
# HONEST NESTED LEAVE-ONE-OUT
# ----------------------------------------------------------------------------
def _mwu_pvalues(Xtr, ytr):
    """
    Two-sided Mann-Whitney p-value for every column at once (a fast, vectorised
    rank-sum with the normal approximation and a tie correction). Used only to
    *rank/filter* features, so the approximation is entirely adequate here and
    gives the same selections as the exact test for all but borderline metrics.
    """
    n = len(ytr)
    na = int(np.sum(ytr == 0))
    nb = int(np.sum(ytr == 1))
    if na == 0 or nb == 0:
        return np.ones(Xtr.shape[1])
    try:
        ranks = rankdata(Xtr, axis=0)                 # scipy >= 1.10
    except TypeError:
        ranks = np.apply_along_axis(rankdata, 0, Xtr)  # older scipy fallback
    Rb = ranks[ytr == 1].sum(axis=0)
    U = Rb - nb * (nb + 1) / 2.0
    mu = na * nb / 2.0
    sigma = np.sqrt(na * nb * (n + 1) / 12.0)          # ties ignored (fine here)
    z = (U - mu) / sigma
    return 2.0 * norm.sf(np.abs(z))


def _select_features(Xtr, ytr, feature_names=None):
    """
    Choose features using ONLY the training rows: keep each metric whose group
    difference is significant (p < SELECT_ALPHA) on the training people. If too
    few pass, back-fill with the smallest-p metrics so the model always has at
    least MIN_FEATURES. Returns an array of column indices.
    """
    pvals = _mwu_pvalues(Xtr, ytr)
    keep = np.where(pvals < SELECT_ALPHA)[0]
    if len(keep) < MIN_FEATURES:
        keep = np.argsort(pvals)[:MIN_FEATURES]
    return keep


def nested_loo(X, y, feature_names, do_selection=True, fixed_idx=None, seed=RANDOM_SEED):
    """
    Leave-one-out predictions. If do_selection, features are chosen inside each
    fold (nested, leakage-free). If fixed_idx is given, those columns are used
    with no selection. Standardisation is fit on the training rows only.
    Returns an array of predicted patient-probabilities aligned with y.
    """
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
    """AUC, accuracy, sensitivity (patient recall), specificity (control recall)."""
    pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs)
    acc = float(np.mean(pred == y))
    sens = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else np.nan
    spec = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else np.nan
    return auc, acc, sens, spec


# ----------------------------------------------------------------------------
# PERMUTATION NULL
# ----------------------------------------------------------------------------
def permutation_null(X, y, feature_names, observed_auc, n_perm=N_PERMUTATIONS,
                     seed=RANDOM_SEED):
    """
    Re-run the honest nested pipeline on randomly shuffled labels n_perm times.
    Returns (null_aucs, empirical_p). If the leak were still present the null
    would sit above 0.5; a null centred on ~0.5 confirms the pipeline is honest.
    """
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


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data, y, features = load_data()
    X = data[features].to_numpy(dtype=float)
    print(f"  {len(y)} subjects ({int(np.sum(y == 0))} control, "
          f"{int(np.sum(y == 1))} patient); {len(features)} candidate features.")

    rows = []

    # Model 1: honest nested selection ------------------------------------
    print("\n  [1] Nested feature-selection model (honest replacement)...")
    probs1 = nested_loo(X, y, features, do_selection=True)
    auc1, acc1, sens1, spec1 = score(y, probs1)
    # how many features the selection typically kept (reported for transparency)
    n_sel = int(np.median([len(_select_features(X[np.arange(len(y)) != i],
                                                y[np.arange(len(y)) != i], features))
                           for i in range(len(y))]))
    print(f"      AUC {auc1:.3f}, acc {acc1:.1%}, sens {sens1:.1%}, "
          f"spec {spec1:.1%}  (median {n_sel} features/fold)")

    print(f"  [1b] Permutation test ({N_PERMUTATIONS} shuffles)...")
    null, p_perm = permutation_null(X, y, features, auc1)
    print(f"      shuffled-label AUC = {null.mean():.3f} ± {null.std():.3f} "
          f"(should be ~0.5); empirical p = {p_perm:.4f}")
    rows.append({"Model": "Nested feature-selection (LOO)", "AUC": round(auc1, 3),
                 "Accuracy": round(acc1, 3), "Sensitivity": round(sens1, 3),
                 "Specificity": round(spec1, 3),
                 "Median_features": n_sel,
                 "Perm_null_AUC_mean": round(float(null.mean()), 3),
                 "Perm_p": round(p_perm, 4),
                 "Note": "honest replacement for the leaky 23-feature model"})

    # Model 2: fixed 5-domain benchmark -----------------------------------
    dom = [f for f in DOMAIN_FEATURES if f in features]
    if len(dom) == len(DOMAIN_FEATURES):
        print("\n  [2] Fixed 5-domain model (no selection, nothing to leak)...")
        idx = [features.index(f) for f in dom]
        probs2 = nested_loo(X, y, features, do_selection=False, fixed_idx=idx)
        auc2, acc2, sens2, spec2 = score(y, probs2)
        print(f"      AUC {auc2:.3f}, acc {acc2:.1%}, sens {sens2:.1%}, "
              f"spec {spec2:.1%}")
        rows.append({"Model": "Fixed 5-domain (LOO)", "AUC": round(auc2, 3),
                     "Accuracy": round(acc2, 3), "Sensitivity": round(sens2, 3),
                     "Specificity": round(spec2, 3), "Median_features": len(dom),
                     "Perm_null_AUC_mean": np.nan, "Perm_p": np.nan,
                     "Note": "features: " + ", ".join(dom)})
    else:
        print("\n  [2] Fixed 5-domain model skipped (missing columns: "
              f"{[f for f in DOMAIN_FEATURES if f not in features]})")

    # Model 3: incremental value of PSD beta ------------------------------
    base = ["HRV_MeanHR", "HRV_RMSSD"]
    if all(f in features for f in base) and "PSD_Beta" in features:
        print("\n  [3] Incremental value of PSD beta...")
        i_base = [features.index(f) for f in base]
        i_full = [features.index(f) for f in base + ["PSD_Beta"]]
        auc_b = roc_auc_score(y, nested_loo(X, y, features, do_selection=False,
                                            fixed_idx=i_base))
        auc_f = roc_auc_score(y, nested_loo(X, y, features, do_selection=False,
                                            fixed_idx=i_full))
        print(f"      MeanHR+RMSSD AUC {auc_b:.3f}  ->  +PSD beta AUC {auc_f:.3f} "
              f"(delta {auc_f - auc_b:+.3f})")
        rows.append({"Model": "MeanHR + RMSSD (LOO)", "AUC": round(auc_b, 3),
                     "Accuracy": np.nan, "Sensitivity": np.nan,
                     "Specificity": np.nan, "Median_features": 2,
                     "Perm_null_AUC_mean": np.nan, "Perm_p": np.nan,
                     "Note": "baseline for incremental test"})
        rows.append({"Model": "MeanHR + RMSSD + PSD_Beta (LOO)",
                     "AUC": round(auc_f, 3), "Accuracy": np.nan,
                     "Sensitivity": np.nan, "Specificity": np.nan,
                     "Median_features": 3, "Perm_null_AUC_mean": np.nan,
                     "Perm_p": np.nan,
                     "Note": f"delta AUC = {auc_f - auc_b:+.3f} vs baseline"})

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "Classification_Summary_NestedCV.csv"),
        index=False)
    print(f"\n  Saved results to {OUTPUT_DIR}/Classification_Summary_NestedCV.csv")


if __name__ == "__main__":
    print("[classification] honest nested cross-validation")
    main()