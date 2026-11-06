"""
s6_common.py: helpers shared by the Supplementary Material v6 builders (s6_facts.py, s6_tables.py).

Formatting follows the main draft (v12_facts.py): signed effect sizes with a true minus sign, p values to
three decimals or "< 0.001", and every number read from a results file or recomputed from the per-subject
files with the same test the pipeline uses.
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

MINUS = "−"
PROJECT = None          # set by init(): the project folder (contains pipeline.py and outputs/)
# Shared registries, filled in place by build_s6.py (never rebound, so "from s6_common import FACTS" stays valid):
#   FACTS  printed values (main-draft facts plus those computed for the supplement, keys "s.*")
#   SNUM, SFNUM  supplementary table and figure numbers by name ("S12"); MNUM, RNUM  methods and results sections
FACTS, SNUM, SFNUM, MNUM, RNUM = {}, {}, {}, {}, {}
OUT = CMP = XC = None
C = P = M = None        # per-subject frames: HRV-ACC controls, patients, MIT-BIH


def init(project):
    global PROJECT, OUT, CMP, XC, C, P, M
    PROJECT = os.path.abspath(project)
    OUT = os.path.join(PROJECT, "outputs")
    CMP = os.path.join(OUT, "comparison", "healthy_hrvacc_vs_psych")
    XC = os.path.join(OUT, "comparison", "cross_control")
    C = subjects(os.path.join(OUT, "healthy_hrvacc", "ecg_results_HealthyACC.csv"), "control_")
    P = subjects(os.path.join(OUT, "psychiatric", "ecg_results_Psychiatric.csv"), "treatment_")
    M = subjects(os.path.join(OUT, "healthy", "ecg_results_Healthy.csv"))
    assert (len(C), len(P), len(M)) == (30, 30, 18), (len(C), len(P), len(M))


# --------------------------------------------------------------------------- formatting
def neg(s):
    return str(s).replace("-", MINUS)


def R(x, nd=3):
    """Signed effect size: +0.667 or −0.809."""
    if x is None or not np.isfinite(x):
        return "–"
    if round(float(x), nd) == 0:
        x = 0.0                      # no '−0.000'
    return neg(f"{x:+.{nd}f}")


def U(x, nd=3):
    if x is None or not np.isfinite(x):
        return "–"
    return neg(f"{x:.{nd}f}")


def P3(p):
    """Table form of a p value: '< 0.001' or '0.012'."""
    if p is None or not np.isfinite(p):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def PE(p):
    """Text form: '*p* < 0.001' or '*p* = 0.012'."""
    return "*p* < 0.001" if p < 0.001 else f"*p* = {p:.3f}"


def RP(r, p, nd=3):
    """'+0.667 (< 0.001)' for compact table cells."""
    return f"{R(r, nd)} ({P3(p)})"


def DEC(ref):
    """Decimals for a quantity of the size of ref, used so a median and its quartiles share one format."""
    a = abs(ref)
    return 0 if a >= 1000 else 1 if a >= 10 else 2 if a >= 1 else 3


def FMT(x, nd):
    if x is None or not np.isfinite(x):
        return "–"
    return neg(f"{x:,.{nd}f}")


def INT(x):
    # half up, the conventional rule (Python's round() sends 994.5 to 994)
    from decimal import Decimal, ROUND_HALF_UP
    return f"{int(Decimal(repr(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}"


def SIG(x, nd=3):
    """Number to nd significant decimals for medians of very different sizes."""
    if x is None or not np.isfinite(x):
        return "–"
    ax = abs(x)
    if ax >= 1000:
        return neg(f"{x:,.0f}")
    if ax >= 100:
        return neg(f"{x:.1f}")
    if ax >= 10:
        return neg(f"{x:.2f}")
    if ax >= 0.1:
        return neg(f"{x:.3f}")
    if ax == 0:
        return "0"
    return neg(f"{x:.4f}")


# --------------------------------------------------------------------------- inputs
def csvp(*parts):
    p = os.path.join(*parts)
    if not os.path.exists(p):
        raise SystemExit(f"missing input {p}")
    return p


def read(*parts, **kw):
    return pd.read_csv(csvp(*parts), **kw)


def rel(path):
    """Path relative to the project folder, with forward slashes (as printed in table notes)."""
    return os.path.relpath(path, PROJECT).replace(os.sep, "/")


def subjects(path, prefix=None):
    d = pd.read_csv(csvp(path), index_col=0)
    if prefix:
        d = d[d.index.astype(str).str.startswith(prefix)]
    else:  # MIT-BIH: record numbers only
        d = d[d.index.astype(str).str.fullmatch(r"\d+")]
    return d


def num(s):
    return pd.to_numeric(pd.Series(s), errors="coerce")


# --------------------------------------------------------------------------- statistics (as the pipeline)
def mw(a, b):
    """Rank-biserial r (positive = higher in the second sample) and two-sided Mann-Whitney p, with the
    pipeline's settings: U for the first sample, continuity correction, method 'auto'."""
    a = num(a).dropna().values
    b = num(b).dropna().values
    u, p = mannwhitneyu(a, b, alternative="two-sided", use_continuity=True, method="auto")
    return 1.0 - 2.0 * u / (len(a) * len(b)), float(p), len(a), len(b)


def ols_resid(y, X):
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X]) if X else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial(y, g, covs):
    """Regress-then-rank partial Spearman correlation (comparison.partial_spearman)."""
    y = np.asarray(y, float)
    g = np.asarray(g, float)
    ok = np.isfinite(y) & np.isfinite(g)
    for c in covs:
        ok &= np.isfinite(np.asarray(c, float))
    X = [np.asarray(c, float)[ok] for c in covs]
    r, p = spearmanr(ols_resid(y[ok], X), ols_resid(g[ok], X))
    return float(r), float(p)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


# --------------------------------------------------------------------------- code constants
_CODE = {}


def code_const(fname, name):
    """Read 'NAME = value' from a project script, so that methods text and tables quote the code."""
    if fname not in _CODE:
        _CODE[fname] = open(csvp(PROJECT, fname), encoding="utf-8").read()
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+?)\s*(#.*)?$", _CODE[fname], flags=re.M)
    if not m:
        raise SystemExit(f"constant {name} not found in {fname}")
    return m.group(1).strip()


def code_has(fname, text):
    if fname not in _CODE:
        _CODE[fname] = open(csvp(PROJECT, fname), encoding="utf-8").read()
    return text in _CODE[fname]


# --------------------------------------------------------------------------- labels
LABEL = {
    "HRV_MeanNN": "Mean NN interval (ms)", "HRV_MeanHR": "Mean heart rate (bpm)", "HRV_SDNN": "SDNN (ms)",
    "HRV_RMSSD": "RMSSD (ms)", "HRV_pNN50": "pNN50 (%)", "HRV_SD1": "SD1 (ms)", "HRV_SD2": "SD2 (ms)",
    "HRV_SD1SD2": "SD1/SD2", "LF_Power_ms2": "LF power (ms²)", "HF_Power_ms2": "HF power (ms²)",
    "VLF_Power_ms2": "VLF power (ms²)", "Total_Power_ms2": "Total power (ms²)", "LF_HF_Ratio": "LF/HF ratio",
    "VLF_LF_Ratio": "VLF/LF ratio", "PSD_Beta": "Slope, uncorrected", "PSD_Beta_NoMayer": "Slope, baroreflex band excised",
    "PSD_Beta_low": "Slope, 0.016–0.062 Hz", "SampEn": "Sample entropy", "ApEn": "Approximate entropy",
    "PermEn": "Permutation entropy", "MSE_Mean": "Multiscale entropy", "DFA_Alpha": "DFA α (full range)",
    "DFA_Alpha1": "DFA α₁", "DFA_Alpha2": "DFA α₂", "RQA_DET": "Determinism", "RQA_LAM": "Laminarity",
    "RNA_ClusteringCoeff": "Network clustering", "RNA_Transitivity": "Network transitivity",
    "MFDFA_Width": "MFDFA width Δh", "MFDFA_H_Mean": "MFDFA mean H(q)", "MFDFA_H2": "MFDFA H(2)",
    "Num_RR": "Clean intervals (n)", "Ordinal_Tie_Rate": "Ordinal tie rate",
    "Quant_Step_Over_RMSSD": "Step-to-RMSSD ratio",
}

# comparison.py METRIC_LABELS (as printed in its CSVs) -> column name
CSV_LABEL = {
    "LF Power (ms²)": "LF_Power_ms2", "Total Power (ms²)": "Total_Power_ms2", "RMSSD (ms)": "HRV_RMSSD",
    "SD1 (ms)": "HRV_SD1", "pNN50 (%)": "HRV_pNN50", "HF Power (ms²)": "HF_Power_ms2", "PSD β": "PSD_Beta",
    "SDNN (ms)": "HRV_SDNN", "SD2 (ms)": "HRV_SD2", "DFA α₂ (16–64)": "DFA_Alpha2", "Mean HR (bpm)": "HRV_MeanHR",
    "Mean NN (ms)": "HRV_MeanNN", "MSE Mean": "MSE_Mean", "DFA α (full)": "DFA_Alpha",
    "Permutation Entropy": "PermEn", "Sample Entropy": "SampEn", "SD1/SD2": "HRV_SD1SD2",
    "RQA Determinism": "RQA_DET", "RQA Laminarity": "RQA_LAM", "LF/HF Ratio": "LF_HF_Ratio",
    "DFA α₁ (4–16)": "DFA_Alpha1", "Approx. Entropy": "ApEn", "RNA Transitivity": "RNA_Transitivity",
    "RNA Clustering Coeff": "RNA_ClusteringCoeff", "MFDFA Δh (width)": "MFDFA_Width",
    "MFDFA H(q=2)": "MFDFA_H2", "MFDFA H̅ (mean)": "MFDFA_H_Mean",
}

# Display order of the 21 primary measures (as in the main Table 2 domains)
PRIMARY = ["HRV_MeanNN", "HRV_MeanHR", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2",
           "LF_Power_ms2", "HF_Power_ms2", "Total_Power_ms2", "LF_HF_Ratio", "PSD_Beta",
           "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2", "SampEn", "PermEn", "MSE_Mean", "RQA_DET", "RQA_LAM"]


def lab(col):
    return LABEL.get(col, col)
