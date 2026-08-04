"""
pipeline.py
===========
Shared analysis functions for the ECG autonomic criticality project.

Imported by healthy.py, psychiatric.py, and comparison.py.
Do NOT add cohort-specific paths, record lists, or output folders here —
those belong in the calling scripts.

Contents
--------
  Configuration constants
  reject_ectopic_beats()        — two-pass RR artifact filter
  compute_frequency_domain()    — Welch PSD, HRV bands, PSD beta
  compute_complexity_metrics()  — SampEn, ApEn, PermEn, MSE, DFA, RQA
  _iaaft_one()                  — single IAAFT surrogate
  _compute_surr_metrics()       — metrics on one surrogate
  run_surrogate_testing()       — full IAAFT test + figure
  _plot_surrogate_distributions()
  _plot_mse_curve()             — standalone MSE curve PNG
  _plot_record()                — six-panel diagnostic figure
  _draw_corr_matrix()           — correlation heatmap helper
  plot_correlation_matrices()   — full + R²-filtered matrices
  plot_radar_charts()           — individual + cohort-mean radar
  analyze_record()              — end-to-end pipeline for one WFDB record
  run_analysis()                — multi-record orchestrator
"""

import wfdb
import neurokit2 as nk
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for batch runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch
from neurokit2.complexity import fractal_dfa
from scipy.stats import linregress
import warnings
import os

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

VLF_BAND              = (0.003, 0.04)
LF_BAND               = (0.04,  0.15)
HF_BAND               = (0.15,  0.40)
BETA_FIT_BAND         = (0.01,  0.15)   # primary: VLF–LF range for 1/f fit
BETA_FIT_BAND_WIDE    = (0.003, 0.15)   # sensitivity alt-1: full VLF–LF
BETA_FIT_BAND_FULL    = (0.003, 0.40)   # sensitivity alt-2: full spectrum
INTERP_FS             = 4.0             # Hz — uniform RR grid for PSD
ANALYSIS_MINUTES      = 30
RR_MIN_MS             = 300
RR_MAX_MS             = 2000
RR_CHANGE_THRESHOLD   = 0.20

# ── Surrogate testing ─────────────────────────────────────────
N_SURROGATES          = 100    # ≥100 is publication standard
N_IAAFT_ITERS         = 1000   # iterations per surrogate for IAAFT convergence
SURROGATE_ALPHA       = 0.05   # two-tailed significance level

# ── Correlation matrix ────────────────────────────────────────
PSD_BETA_R2_THRESHOLD = 0.70   # records below this excluded from filtered matrix

# ── Radar chart axes ──────────────────────────────────────────
# Five metrics spanning spectral, temporal, and complexity dimensions.
RADAR_METRICS = ["PSD_Beta", "DFA_Alpha1", "SampEn", "MSE_Mean", "RQA_DET"]
RADAR_LABELS  = ["PSD β",    "DFA α₁",    "SampEn", "MSE\nMean", "RQA\nDET"]


# ═══════════════════════════════════════════════════════════════
# STEP 1 — ARTIFACT / ECTOPIC REJECTION
# ═══════════════════════════════════════════════════════════════

def reject_ectopic_beats(rr_intervals_seconds, verbose=True):
    """
    Two-pass ectopic beat filter.
      Pass 1 — hard physiological bounds (300–2000 ms).
      Pass 2 — local rolling-median outlier removal (>20% deviation,
                ±5-beat window).
    Returns (rr_clean_s, artifact_percent, n_clean).
    """
    rr_ms       = rr_intervals_seconds * 1000.0
    mask_bounds = (rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)
    rr_ms       = rr_ms[mask_bounds]

    clean_mask = np.ones(len(rr_ms), dtype=bool)
    for i in range(len(rr_ms)):
        lo = max(0, i - 5)
        hi = min(len(rr_ms), i + 6)
        med = np.median(rr_ms[lo:hi])
        if med > 0 and abs(rr_ms[i] - med) / med > RR_CHANGE_THRESHOLD:
            clean_mask[i] = False

    rr_clean         = rr_ms[clean_mask] / 1000.0
    removed          = (~mask_bounds).sum() + (~clean_mask).sum()
    artifact_percent = 100.0 * removed / len(rr_intervals_seconds)

    if verbose:
        print(f"    Artifact rejection: removed {removed} beats "
              f"({artifact_percent:.1f}%); {len(rr_clean)} retained.")
    return rr_clean, artifact_percent, len(rr_clean)


def _fit_beta(freqs, power, fit_band):
    """
    Fit the PSD scaling exponent β over a given frequency band via
    log-log OLS linear regression.

    Parameters
    ----------
    freqs    : array  frequency bins from Welch
    power    : array  PSD values
    fit_band : tuple  (f_low, f_high) Hz

    Returns
    -------
    (beta, r2) — both np.nan if fewer than 3 frequency points fall in band.
    β is the NEGATED slope so that a healthy 1/f spectrum returns β ≈ +1.
    """
    psd_beta, r2 = np.nan, np.nan
    try:
        mask = (freqs >= fit_band[0]) & (freqs <= fit_band[1]) & (freqs > 0)
        if mask.sum() >= 3:
            fit      = linregress(np.log10(freqs[mask]), np.log10(power[mask]))
            psd_beta = -fit.slope
            r2       = fit.rvalue ** 2
    except Exception:
        pass
    return psd_beta, r2


# ═══════════════════════════════════════════════════════════════
# STEP 2 — FREQUENCY DOMAIN (corrected Welch + PSD beta)
# ═══════════════════════════════════════════════════════════════

def compute_frequency_domain(rr_intervals, rr_times):
    """
    Cubic-spline interpolate RR series onto a 4 Hz uniform grid,
    compute Welch PSD, extract HRV band powers, and fit the 1/f beta
    exponent over multiple frequency ranges (reviewer robustness request).

    PSD β variants stored
    ---------------------
    PSD_Beta          — primary estimate, 0.01–0.15 Hz (VLF–LF), cubic spline
    PSD_Beta_Wide     — sensitivity alt-1, 0.003–0.15 Hz (full VLF–LF)
    PSD_Beta_Full     — sensitivity alt-2, 0.003–0.40 Hz (full spectrum)
    PSD_Beta_Linear   — preprocessing sensitivity: same primary range but
                        using linear rather than cubic spline interpolation

    All four share the same Welch parameters so differences isolate the
    fitting range or interpolation choice.

    Returns (metrics_dict, freqs_array, power_array).
    """
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / INTERP_FS)

    # ── Primary interpolation: cubic spline (standard, Task Force 1996) ──
    rr_cubic = interp1d(rr_times, rr_intervals, kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate')(t_uniform)

    # ── Preprocessing sensitivity: linear interpolation ───────────────────
    rr_linear = interp1d(rr_times, rr_intervals, kind='linear',
                         bounds_error=False,
                         fill_value='extrapolate')(t_uniform)

    nperseg          = min(len(rr_cubic), 256)
    freqs, power     = welch(rr_cubic,  fs=INTERP_FS, nperseg=nperseg)
    _,     pow_lin   = welch(rr_linear, fs=INTERP_FS, nperseg=nperseg)

    def band_power(band):
        idx = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapezoid(power[idx], freqs[idx]) if idx.sum() > 1 else np.nan

    lf_p  = band_power(LF_BAND)
    hf_p  = band_power(HF_BAND)
    tot_p = band_power((VLF_BAND[0], HF_BAND[1]))
    lf_hf = lf_p / hf_p if (hf_p and hf_p > 0) else np.nan

    # ── β at three frequency ranges (cubic spline PSD) ───────────────────
    b_primary, r2_primary = _fit_beta(freqs, power,   BETA_FIT_BAND)
    b_wide,    r2_wide    = _fit_beta(freqs, power,   BETA_FIT_BAND_WIDE)
    b_full,    r2_full    = _fit_beta(freqs, power,   BETA_FIT_BAND_FULL)

    # ── β with linear interpolation (preprocessing sensitivity) ──────────
    b_linear,  r2_linear  = _fit_beta(freqs, pow_lin, BETA_FIT_BAND)

    return {
        "LF_Power_ms2":       lf_p  * 1e6,
        "HF_Power_ms2":       hf_p  * 1e6,
        "Total_Power_ms2":    tot_p * 1e6,
        "LF_HF_Ratio":        lf_hf,
        # Primary β (0.01–0.15 Hz, cubic spline)
        "PSD_Beta":           b_primary,
        "PSD_Beta_R2":        r2_primary,
        # Frequency-range sensitivity
        "PSD_Beta_Wide":      b_wide,     # 0.003–0.15 Hz
        "PSD_Beta_Wide_R2":   r2_wide,
        "PSD_Beta_Full":      b_full,     # 0.003–0.40 Hz
        "PSD_Beta_Full_R2":   r2_full,
        # Preprocessing sensitivity
        "PSD_Beta_Linear":    b_linear,   # 0.01–0.15 Hz, linear interp
        "PSD_Beta_Linear_R2": r2_linear,
    }, freqs, power


# ═══════════════════════════════════════════════════════════════
# STEP 3 — COMPLEXITY / CRITICALITY METRICS  (+MSE curve data)
# ═══════════════════════════════════════════════════════════════

def compute_complexity_metrics(rr_intervals):
    """
    Compute the full complexity battery.

    MSE scale cap
    -------------
    Scale is capped at max(10, N//20) so that each coarse-grained
    series contains ≥20 points, preventing infinite entropy at high
    scales. Non-finite values are filtered before averaging.

    Returns
    -------
    metrics    : dict           scalar metric values
    mse_scales : np.ndarray     scale indices for curve plotting (or None)
    mse_values : np.ndarray     entropy at each scale (or None)
    """
    results    = {}
    mse_scales = None
    mse_values = None

    # Sample Entropy
    try:
        se, _ = nk.entropy_sample(rr_intervals, dimension=2, tolerance="sd")
        results["SampEn"] = float(se)
    except Exception as e:
        print(f"    SampEn failed: {e}")
        results["SampEn"] = np.nan

    # Approximate Entropy
    try:
        ae, _ = nk.entropy_approximate(rr_intervals, dimension=2, tolerance="sd")
        results["ApEn"] = float(ae)
    except Exception as e:
        print(f"    ApEn failed: {e}")
        results["ApEn"] = np.nan

    # Permutation Entropy
    try:
        pe, _ = nk.entropy_permutation(rr_intervals, dimension=3, delay=1)
        results["PermEn"] = float(pe)
    except Exception as e:
        print(f"    PermEn failed: {e}")
        results["PermEn"] = np.nan

    # Multiscale Entropy
    # mse_info["Value"] is a numpy ndarray; scales reconstructed as 1..max_scale.
    try:
        max_scale = max(10, len(rr_intervals) // 20)
        _, mse_info = nk.entropy_multiscale(
            rr_intervals, dimension=2, scale=max_scale, show=False
        )
        raw_vals   = np.array(mse_info["Value"], dtype=float)
        raw_scales = np.arange(1, len(raw_vals) + 1, dtype=float)
        finite     = np.isfinite(raw_vals)
        mse_scales = raw_scales[finite]
        mse_values = raw_vals[finite]
        if len(mse_values) > 0:
            results["MSE_Mean"] = float(np.mean(mse_values))
            print(f"    MSE: {len(mse_values)} finite scales, "
                  f"mean = {results['MSE_Mean']:.4f}")
        else:
            print("    MSE: no finite values after extraction")
            results["MSE_Mean"] = np.nan
    except Exception as e:
        print(f"    MSE failed: {e}")
        results["MSE_Mean"] = np.nan

    # DFA full range
    try:
        v, _ = fractal_dfa(rr_intervals, show=False)
        results["DFA_Alpha"] = float(v)
    except Exception as e:
        print(f"    DFA full failed: {e}")
        results["DFA_Alpha"] = np.nan

    # DFA alpha1 — short-range (4–16 beats), canonical criticality marker
    try:
        v, _ = fractal_dfa(rr_intervals, scale=np.arange(4, 17), show=False)
        results["DFA_Alpha1"] = float(v)
    except Exception as e:
        print(f"    DFA alpha1 failed: {e}")
        results["DFA_Alpha1"] = np.nan

    # DFA alpha2 — long-range (16–64 beats)
    try:
        v, _ = fractal_dfa(rr_intervals, scale=np.arange(16, 65), show=False)
        results["DFA_Alpha2"] = float(v)
    except Exception as e:
        print(f"    DFA alpha2 failed: {e}")
        results["DFA_Alpha2"] = np.nan

    # RQA
    try:
        rqa_df, _ = nk.complexity_rqa(rr_intervals, delay=1, dimension=2)
        results["RQA_REC"] = float(rqa_df["RecurrenceRate"].iloc[0])
        results["RQA_DET"] = float(rqa_df["Determinism"].iloc[0])
        results["RQA_LAM"] = float(rqa_df["Laminarity"].iloc[0])
        results["RQA_DIV"] = float(rqa_df["Divergence"].iloc[0])
        results["RQA_LEN"] = float(rqa_df["LEn"].iloc[0])
    except Exception as e:
        print(f"    RQA failed: {e}")
        for k in ["RQA_REC", "RQA_DET", "RQA_LAM", "RQA_DIV", "RQA_LEN"]:
            results[k] = np.nan

    # ── MFDFA (new) ───────────────────────────────────────────────────────────
    mfdfa = compute_mfdfa(rr_intervals)
    results.update(mfdfa)

    # ── Recurrence Network Analysis (new) ─────────────────────────────────────
    rna = compute_recurrence_network(rr_intervals)
    results.update(rna)

    return results, mse_scales, mse_values


# ═══════════════════════════════════════════════════════════════
# STEP 3b — MULTIFRACTAL DFA  (pure numpy / scipy)
# ═══════════════════════════════════════════════════════════════

def compute_mfdfa(rr_s, q_orders=None, n_scales=20, poly_order=1):
    """
    Multifractal Detrended Fluctuation Analysis (MFDFA).

    Implements the standard Kantelhardt et al. (2002) algorithm entirely
    in NumPy / SciPy — no external MFDFA package required.

    Scientific background
    ---------------------
    Standard DFA estimates a *single* scaling exponent α that summarises
    long-range correlations across all fluctuation magnitudes.  MFDFA
    extends this by computing the *generalised Hurst exponent* H(q) for a
    range of statistical moments q.  Varying q emphasises different parts
    of the fluctuation distribution:
        q < 0  → small fluctuations (rare, low-amplitude deviations)
        q > 0  → large fluctuations (dominant, high-amplitude swings)
    If H(q) is constant across q the signal is monofractal (single scaling
    exponent, as assumed by DFA).  If H(q) varies substantially with q the
    signal is *multifractal*, possessing a richer scaling structure driven
    by different mechanisms at different fluctuation scales.

    In healthy RR dynamics, multifractality reflects the interplay of
    multiple regulatory loops (respiratory, baroreflex, hormonal,
    thermoregulatory) each contributing at different magnitudes and
    timescales.  Loss of this richness — narrower singularity spectrum —
    has been associated with ageing, cardiac pathology, and autonomic
    rigidity.  Psychiatric patients are hypothesised to show:
        • Higher H_Mean — stronger overall long-range correlations
          (consistent with elevated PSD β and DFA α₂)
        • Potentially narrower MFDFA_Width — less multifractal richness,
          reflecting more homogeneous, constrained dynamics

    MFDFA_H2 provides a critical cross-check: H(q=2) should equal the
    conventional DFA alpha (DFA_Alpha), since both probe the same scaling
    structure at the second statistical moment.  Substantial divergence
    flags a numerical issue.

    Parameters
    ----------
    rr_s       : 1D array   Clean RR interval series in seconds (or ms —
                            units do not affect scaling exponents).
    q_orders   : array-like Statistical moments to probe.  Defaults to
                            q ∈ {-5,-4,...,-1,+1,...,+5} (zero excluded;
                            handled separately via geometric mean).
    n_scales   : int        Number of log-spaced window sizes (scales).
                            20 is standard; increase to 30 for longer series.
    poly_order : int        Polynomial order for within-window detrending.
                            1 (linear, DFA-1) matches the monofractal DFA
                            already in the pipeline.

    Returns
    -------
    dict with four scalar fields:
        MFDFA_Width      float   Δh = h_max − h_min.  Width of the
                                 singularity spectrum.  Larger = more
                                 multifractal (richer scaling structure).
        MFDFA_H_Mean     float   Mean of H(q) across all valid q.  Reflects
                                 overall long-range correlation strength;
                                 directly comparable to DFA_Alpha.
        MFDFA_Asymmetry  float   H(q<0)_mean − H(q>0)_mean.  Positive =
                                 large-fluctuation dominance (left-skewed
                                 singularity spectrum, common in biological
                                 time series with occasional high-variability
                                 episodes).
        MFDFA_H2         float   H(q=2) — cross-check against DFA_Alpha.
    """
    _nan = dict(MFDFA_Width=np.nan, MFDFA_H_Mean=np.nan,
                MFDFA_Asymmetry=np.nan, MFDFA_H2=np.nan)

    try:
        x = np.asarray(rr_s, dtype=float)
        if not np.all(np.isfinite(x)) or len(x) < 64:
            return _nan

        # Profile: cumulative sum of mean-centred series
        y = np.cumsum(x - np.mean(x))
        N = len(y)

        if q_orders is None:
            q_orders = np.arange(-5, 6, dtype=float)
            q_orders = q_orders[q_orders != 0]   # exclude q=0

        q_orders = np.asarray(q_orders, dtype=float)

        # Log-spaced window sizes from s_min to N//4
        s_min = max(8, int(N ** 0.25))
        s_max = N // 4
        if s_max < s_min * 2:
            return _nan

        scales = np.unique(
            np.round(
                np.logspace(np.log10(s_min), np.log10(s_max), n_scales)
            ).astype(int)
        )
        scales = scales[(scales >= s_min) & (scales <= s_max)]
        if len(scales) < 4:
            return _nan

        # Fluctuation function F_q(s)
        Fq  = np.full((len(q_orders), len(scales)), np.nan)
        t_s = np.arange(max(scales), dtype=float)

        for si, s in enumerate(scales):
            n_seg  = N // s
            if n_seg < 2:
                continue
            ts = t_s[:s]
            F2 = []
            for seg in range(n_seg):
                seg_y = y[seg * s : (seg + 1) * s]
                try:
                    coeffs = np.polyfit(ts, seg_y, poly_order)
                    resid  = seg_y - np.polyval(coeffs, ts)
                    f2     = float(np.mean(resid ** 2))
                    if np.isfinite(f2) and f2 > 0.0:
                        F2.append(f2)
                except Exception:
                    pass

            if len(F2) < 2:
                continue
            F2_arr = np.array(F2)

            for qi, q in enumerate(q_orders):
                try:
                    val = float(np.mean(F2_arr ** (q / 2.0))) ** (1.0 / q)
                    if np.isfinite(val) and val > 0.0:
                        Fq[qi, si] = val
                except Exception:
                    pass

        # Estimate H(q) by log-log linear regression: log F_q ~ H(q) log s
        log_s = np.log(scales.astype(float))
        h_q   = np.full(len(q_orders), np.nan)

        for qi in range(len(q_orders)):
            row   = Fq[qi]
            valid = np.isfinite(row) & (row > 0.0)
            if valid.sum() < 4:
                continue
            slope, *_ = linregress(log_s[valid], np.log(row[valid]))
            if np.isfinite(slope):
                h_q[qi] = slope

        # Derive singularity spectrum summary statistics
        fin = h_q[np.isfinite(h_q)]
        if len(fin) < 3:
            print("    MFDFA: insufficient valid H(q) estimates; returning NaN")
            return _nan

        width     = float(np.max(fin) - np.min(fin))
        h_mean    = float(np.mean(fin))
        neg_h     = h_q[q_orders < 0]; neg_h = neg_h[np.isfinite(neg_h)]
        pos_h     = h_q[q_orders > 0]; pos_h = pos_h[np.isfinite(pos_h)]
        asymmetry = (float(np.mean(neg_h) - np.mean(pos_h))
                     if (len(neg_h) > 0 and len(pos_h) > 0) else np.nan)
        q2_idx    = np.where(q_orders == 2.0)[0]
        h2        = (float(h_q[q2_idx[0]])
                     if q2_idx.size > 0 and np.isfinite(h_q[q2_idx[0]])
                     else np.nan)

        out = dict(MFDFA_Width=width, MFDFA_H_Mean=h_mean,
                   MFDFA_Asymmetry=asymmetry, MFDFA_H2=h2)
        print(f"    MFDFA: Δh={width:.4f}  H̄={h_mean:.4f}  "
              f"asym={asymmetry:.4f}  H(2)={h2:.4f}")
        return out

    except Exception as e:
        print(f"    MFDFA failed: {e}")
        return _nan


# ═══════════════════════════════════════════════════════════════
# STEP 3c — RECURRENCE NETWORK ANALYSIS  (pure numpy)
# ═══════════════════════════════════════════════════════════════

def compute_recurrence_network(rr_s, dim=2, delay=1,
                               rec_rate=0.10, max_pts=300):
    """
    Recurrence Network Analysis (RNA).

    Treats the binary recurrence matrix as an unweighted, undirected graph
    and extracts three node-level statistics that characterise the *global
    topology* of the reconstructed phase-space attractor.  This is
    analytically distinct from — and complementary to — the line-structure
    statistics of RQA (Determinism, Laminarity):

        RQA DET/LAM  → local diagonal / vertical structure of the recurrence
                        plot (predictability and intermittency of trajectories)
        RNA CC/Trans  → neighbourhood density of the recurrence *network*
                        (how clique-like the attractor is at each node)

    Scientific background (Marwan et al. 2009, New J Phys)
    -------------------------------------------------------
    A high clustering coefficient means that the neighbours of any state
    also tend to be neighbours of each other — geometrically, this implies
    the trajectory repeatedly visits tightly clustered regions of phase
    space, consistent with a constrained, low-dimensional attractor.
    Healthy, adaptive cardiac dynamics are hypothesised to occupy a wider,
    more distributed region of phase space, yielding a lower clustering
    coefficient (more exploratory dynamics) relative to the more
    stereotyped dynamics expected in psychiatric patients.

    Implementation notes
    --------------------
    • Recurrence threshold is set by the *fixed recurrence rate* (RR=10%):
      ε is the rec_rate-quantile of all pairwise distances, guaranteeing
      equal network density across subjects regardless of signal amplitude.
      This is the standard approach for RNA (Donner et al. 2010).
    • The RR series is uniformly subsampled to max_pts points before
      embedding.  For a 2500-beat series this gives one sample every ~8
      beats — sufficient to capture attractor geometry at the timescale
      resolved by the delay embedding, while keeping distances O(300²)
      and triangle counting O(300³ / sparse) fast.

    Parameters
    ----------
    rr_s    : 1D array  Clean RR intervals in seconds.
    dim     : int       Embedding dimension (2 matches existing RQA).
    delay   : int       Embedding delay (1 matches existing RQA).
    rec_rate: float     Target recurrence rate (fraction of pairs flagged
                        recurrent, 0–1).  0.10 = 10%, matching the RQA
                        threshold of 10% of the mean phase-space diameter.
    max_pts : int       Maximum embedded points after subsampling.

    Returns
    -------
    dict with:
        RNA_MeanDegree      Mean normalised node degree = mean(k_i)/(n-1).
                            Higher = denser recurrence network (more
                            constrained phase-space trajectory).
        RNA_ClusteringCoeff Mean local clustering coefficient C_i.
                            Higher = more clique-like local neighbourhood
                            (repetitive, constrained dynamics).
        RNA_Transitivity    Global transitivity = 3·triangles/triplets.
                            A global analog of clustering; more robust to
                            heterogeneous degree distributions.
    """
    _nan = dict(RNA_MeanDegree=np.nan, RNA_ClusteringCoeff=np.nan,
                RNA_Transitivity=np.nan)

    try:
        x = np.asarray(rr_s, dtype=float)
        if not np.all(np.isfinite(x)) or len(x) < 30:
            return _nan

        # Uniform subsample to keep computation tractable
        if len(x) > max_pts:
            idx = np.linspace(0, len(x) - 1, max_pts, dtype=int)
            x   = x[idx]

        n_pts = len(x) - (dim - 1) * delay
        if n_pts < 20:
            return _nan

        # Delay embedding: n_pts × dim matrix
        X = np.stack(
            [x[i : i + (dim - 1) * delay + 1 : delay]
             for i in range(n_pts)],
            axis=0
        ).astype(float)

        # Pairwise Euclidean distances (vectorised)
        diff = X[:, None, :] - X[None, :, :]      # n_pts × n_pts × dim
        dist = np.sqrt((diff ** 2).sum(axis=2))    # n_pts × n_pts

        # Fixed recurrence rate threshold
        upper = dist[np.triu_indices(n_pts, k=1)]
        eps   = float(np.quantile(upper, rec_rate))

        # Binary adjacency matrix — symmetric, no self-loops
        A = (dist <= eps).astype(np.float32)
        np.fill_diagonal(A, 0.0)

        n = n_pts
        k = A.sum(axis=1)                           # node degree vector

        mean_degree = float(k.mean() / (n - 1)) if n > 1 else np.nan

        # Triangle / clustering computation
        # diag(A^3)[i] = number of length-3 closed walks at node i
        # triangles_i  = diag(A^3)[i] / 2
        # triplets_i   = k_i * (k_i - 1) / 2
        A_sq     = A @ A
        # Diagonal of A^3 via einsum (avoids materialising the full A^3 cube)
        tri_diag = np.einsum('ij,ji->i', A_sq, A)  # shape n_pts

        triangles = tri_diag / 2.0
        triplets  = k * (k - 1) / 2.0

        with np.errstate(invalid='ignore', divide='ignore'):
            cc_local = np.where(triplets > 0.0, triangles / triplets, 0.0)
        clustering = float(cc_local.mean())

        # Global transitivity: 3 × total_triangles / total_triplets
        # tri_diag sums 2 × triangles at each node = 6 × (unique triangles)
        total_tri  = float(tri_diag.sum()) / 6.0
        total_trip = float(triplets.sum())
        transitivity = (3.0 * total_tri / total_trip
                        if total_trip > 0.0 else np.nan)

        out = dict(RNA_MeanDegree=mean_degree,
                   RNA_ClusteringCoeff=clustering,
                   RNA_Transitivity=float(transitivity))
        print(f"    RNA:   deg={mean_degree:.4f}  CC={clustering:.4f}  "
              f"trans={transitivity:.4f}")
        return out

    except Exception as e:
        print(f"    RNA failed: {e}")
        return _nan


# ═══════════════════════════════════════════════════════════════
# STEP 4 — IAAFT SURROGATE TESTING
# ═══════════════════════════════════════════════════════════════

def _iaaft_one(x, n_iters=N_IAAFT_ITERS, rng=None):
    """
    Generate one IAAFT surrogate (Schreiber & Schmitz, 1996).

    Preserves the marginal amplitude distribution (rank-order substitution)
    and power spectrum (autocorrelation structure) while destroying nonlinear
    phase-coupling structure. This is the correct null for testing whether
    SampEn, DFA α₁, and MSE reflect genuine nonlinear dynamics.
    """
    if rng is None:
        rng = np.random.default_rng()

    n        = len(x)
    x_sorted = np.sort(x)
    x_amp    = np.abs(np.fft.rfft(x))

    phi        = rng.uniform(0, 2 * np.pi, len(x_amp))
    phi[0]     = 0.0
    if n % 2 == 0:
        phi[-1] = 0.0
    s = np.fft.irfft(x_amp * np.exp(1j * phi), n=n)

    for _ in range(n_iters):
        s_fft = np.fft.rfft(s)
        s     = np.fft.irfft(x_amp * np.exp(1j * np.angle(s_fft)), n=n)
        s     = x_sorted[np.argsort(np.argsort(s))]

    return s


def _compute_surr_metrics(s):
    """
    Compute SampEn, DFA_Alpha1, and MSE_Mean on a single surrogate series.
    Returns dict; values are np.nan on failure.
    """
    out = {}

    try:
        v, _ = nk.entropy_sample(s, dimension=2, tolerance="sd")
        out["SampEn"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        out["SampEn"] = np.nan

    try:
        v, _ = fractal_dfa(s, scale=np.arange(4, 17), show=False)
        out["DFA_Alpha1"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        out["DFA_Alpha1"] = np.nan

    try:
        max_scale = max(10, len(s) // 20)
        _, mse_info = nk.entropy_multiscale(s, dimension=2,
                                            scale=max_scale, show=False)
        raw_vals    = np.array(mse_info["Value"], dtype=float)
        finite_vals = raw_vals[np.isfinite(raw_vals)]
        out["MSE_Mean"] = float(np.mean(finite_vals)) if len(finite_vals) > 0 else np.nan
    except Exception:
        out["MSE_Mean"] = np.nan

    return out


def run_surrogate_testing(rr_clean, record_id, output_dir, verbose=True):
    """
    Full IAAFT surrogate test for SampEn, DFA_Alpha1, and MSE_Mean.

    For each metric:
      • Observed value computed on real RR series.
      • N_SURROGATES IAAFT surrogates generated → null distribution.
      • Two-tailed test at SURROGATE_ALPHA (obs outside [2.5th, 97.5th] pct).
      • z-score = (observed − surr_mean) / surr_sd reported.

    Saves {record_id}_surrogate_test.png to output_dir.
    Returns dict of SURR_* columns for the per-record result dict.
    """
    if verbose:
        print(f"  IAAFT surrogate testing "
              f"(N={N_SURROGATES}, {N_IAAFT_ITERS} iters/surrogate)...")

    rng = np.random.default_rng(seed=42)    # fixed seed → reproducible

    obs  = _compute_surr_metrics(rr_clean)
    null = {"SampEn": [], "DFA_Alpha1": [], "MSE_Mean": []}

    for i in range(N_SURROGATES):
        s = _iaaft_one(rr_clean, rng=rng)
        m = _compute_surr_metrics(s)
        for k in null:
            null[k].append(m[k])
        if verbose and (i + 1) % 25 == 0:
            print(f"    {i+1}/{N_SURROGATES} surrogates done")

    lo_pct = 100.0 * (SURROGATE_ALPHA / 2)
    hi_pct = 100.0 * (1.0 - SURROGATE_ALPHA / 2)

    surr_results = {}
    null_arrays  = {}

    for metric in ["SampEn", "DFA_Alpha1", "MSE_Mean"]:
        arr     = np.array(null[metric], dtype=float)
        finite  = arr[np.isfinite(arr)]
        obs_val = obs.get(metric, np.nan)
        null_arrays[metric] = arr

        if len(finite) < 10 or not np.isfinite(obs_val):
            surr_results[f"SURR_{metric}_Mean"] = np.nan
            surr_results[f"SURR_{metric}_SD"]   = np.nan
            surr_results[f"SURR_{metric}_Z"]    = np.nan
            surr_results[f"SURR_{metric}_Sig"]  = False
            continue

        s_mean = np.mean(finite)
        s_sd   = np.std(finite, ddof=1)
        z      = (obs_val - s_mean) / s_sd if s_sd > 0 else np.nan
        lo     = np.percentile(finite, lo_pct)
        hi     = np.percentile(finite, hi_pct)
        sig    = bool(obs_val < lo or obs_val > hi)

        surr_results[f"SURR_{metric}_Mean"] = s_mean
        surr_results[f"SURR_{metric}_SD"]   = s_sd
        surr_results[f"SURR_{metric}_Z"]    = z
        surr_results[f"SURR_{metric}_Sig"]  = sig

        if verbose:
            tag = "★ SIGNIFICANT" if sig else "not significant"
            print(f"    {metric}: obs={obs_val:.4f}, "
                  f"surr={s_mean:.4f}±{s_sd:.4f}, z={z:.2f}  {tag}")

    _plot_surrogate_distributions(
        record_id, obs, null_arrays, surr_results, output_dir
    )
    return surr_results


def _plot_surrogate_distributions(record_id, obs, null_arrays,
                                  surr_results, output_dir):
    """
    Three-panel histogram of IAAFT null distributions with observed
    value and 95% acceptance region marked.
    Saved as {record_id}_surrogate_test.png.
    """
    metric_labels = {
        "SampEn":     "Sample Entropy",
        "DFA_Alpha1": "DFA α₁",
        "MSE_Mean":   "MSE Mean",
    }
    metrics = list(metric_labels.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"IAAFT Surrogate Test — Record {record_id}  "
        f"(N={N_SURROGATES}, α={SURROGATE_ALPHA})",
        fontsize=11
    )

    lo_pct = 100.0 * SURROGATE_ALPHA / 2
    hi_pct = 100.0 * (1.0 - SURROGATE_ALPHA / 2)

    for ax, metric in zip(axes, metrics):
        arr    = null_arrays.get(metric, np.array([]))
        finite = arr[np.isfinite(arr)]
        obs_val = obs.get(metric, np.nan)
        sig     = surr_results.get(f"SURR_{metric}_Sig", False)
        z_val   = surr_results.get(f"SURR_{metric}_Z",   np.nan)
        xlabel  = metric_labels[metric]

        if len(finite) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(xlabel)
            continue

        lo = np.percentile(finite, lo_pct)
        hi = np.percentile(finite, hi_pct)

        ax.hist(finite, bins=25, color='steelblue', edgecolor='white',
                alpha=0.75, label=f'Surrogates (N={len(finite)})')
        ax.axvspan(lo, hi, alpha=0.12, color='limegreen', label='95% CI')
        ax.axvline(lo, color='limegreen', lw=1.2, linestyle='--')
        ax.axvline(hi, color='limegreen', lw=1.2, linestyle='--')

        obs_color = 'red' if sig else 'darkorange'
        obs_label = (f'Obs = {obs_val:.3f}  ★ sig.'
                     if sig else f'Obs = {obs_val:.3f}')
        ax.axvline(obs_val, color=obs_color, lw=2.2, label=obs_label)

        z_str = f"z = {z_val:.2f}" if np.isfinite(z_val) else "z = n/a"
        ax.set_title(f"{xlabel}\n{z_str}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Count" if ax is axes[0] else "", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{record_id}_surrogate_test.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Surrogate plot  → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — MSE CURVE PNG (per record)
# ═══════════════════════════════════════════════════════════════

def _plot_mse_curve(record_id, mse_scales, mse_values, output_dir):
    """
    Standalone MSE curve PNG: {record_id}_MSE_curve.png.
    Rising/plateau = healthy complexity; monotonic decay = reduced complexity.
    The red dashed line at MSE_Mean links the curve to the CSV scalar.
    """
    if mse_scales is None or len(mse_scales) == 0:
        print(f"  MSE curve: no valid data for {record_id}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mse_scales, mse_values, marker='o', markersize=5,
            lw=1.8, color='steelblue', label='Sample Entropy')
    ax.fill_between(mse_scales, mse_values, alpha=0.15, color='steelblue')

    mse_mean = float(np.mean(mse_values))
    ax.axhline(mse_mean, color='red', linestyle='--', lw=1.2,
               label=f'MSE_Mean = {mse_mean:.3f}')

    ax.set_title(
        f"Multiscale Entropy Curve — Record {record_id}\n"
        f"Rising/plateau = healthy complexity; decay = reduced complexity",
        fontsize=10
    )
    ax.set_xlabel("Scale Factor", fontsize=10)
    ax.set_ylabel("Sample Entropy", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{record_id}_MSE_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  MSE curve       → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — SIX-PANEL DIAGNOSTIC PLOT (per record)
# ═══════════════════════════════════════════════════════════════

def _plot_record(record_id, time, raw_ecg, cleaned_ecg, r_peaks,
                 rr_times_raw, rr_raw, rr_times_clean, rr_clean,
                 freqs, psd_power, complexity_metrics, psd_beta=np.nan,
                 output_dir="outputs"):
    """
    Six-panel diagnostic figure saved as {record_id}_diagnostics.png.
    Panels: raw ECG, cleaned ECG, RR intervals, PSD + beta fit,
            Poincaré plot, complexity bar chart.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"ECG Criticality Analysis — Record {record_id}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(time, raw_ecg, lw=0.6, color='steelblue')
    ax.scatter(time[r_peaks], raw_ecg[r_peaks],
               color='red', s=12, zorder=5, label='R-peaks')
    ax.set_title("Raw ECG + R-peaks")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (mV)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(time, cleaned_ecg, lw=0.6, color='darkorange')
    ax.scatter(time[r_peaks], cleaned_ecg[r_peaks], color='red', s=12, zorder=5)
    ax.set_title("Cleaned ECG + R-peaks")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (mV)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rr_times_raw,   rr_raw   * 1000, lw=0.8,
            color='gray',   alpha=0.6, label='Raw')
    ax.plot(rr_times_clean, rr_clean * 1000, lw=0.9,
            color='steelblue', label='Cleaned')
    ax.set_title("RR Intervals (raw vs cleaned)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("RR (ms)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(freqs, psd_power, color='darkviolet', lw=1.2, label='PSD')
    ax.axvspan(*VLF_BAND, alpha=0.15, color='blue',   label='VLF')
    ax.axvspan(*LF_BAND,  alpha=0.15, color='green',  label='LF')
    ax.axvspan(*HF_BAND,  alpha=0.15, color='orange', label='HF')
    if np.isfinite(psd_beta):
        mask = (freqs >= BETA_FIT_BAND[0]) & (freqs <= BETA_FIT_BAND[1]) & (freqs > 0)
        if mask.sum() >= 3:
            lf_ = np.log10(freqs[mask])
            lp_ = np.log10(psd_power[mask])
            ic  = np.mean(lp_ + psd_beta * lf_)
            ax.semilogy(freqs[mask], 10 ** (ic - psd_beta * lf_),
                        color='red', lw=1.4, linestyle='--',
                        label=f'1/f fit β={psd_beta:.3f}')
    ax.set_title("HRV Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (s²/Hz)")
    ax.legend(fontsize=8); ax.set_xlim(0, 0.5); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.scatter(rr_clean[:-1] * 1000, rr_clean[1:] * 1000,
               s=8, alpha=0.5, color='teal')
    ax.set_title("Poincaré Plot")
    ax.set_xlabel("RR$_n$ (ms)"); ax.set_ylabel("RR$_{n+1}$ (ms)")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    bar_labels = ["SampEn", "ApEn", "PermEn", "MSE\nMean",
                  "DFA\nα",  "DFA\nα1", "DFA\nα2"]
    bar_keys   = ["SampEn", "ApEn", "PermEn", "MSE_Mean",
                  "DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2"]
    bar_vals   = [complexity_metrics.get(k, np.nan) for k in bar_keys]
    bar_colors = ['steelblue', 'darkorange', 'green',
                  'red', 'purple', 'brown', 'pink']
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors,
                  edgecolor='white', width=0.6)
    for bar, val in zip(bars, bar_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_title("Complexity / Criticality Metrics")
    ax.set_ylabel("Value"); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{record_id}_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostics     → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — CORRELATION MATRICES (full + R²-filtered)
# ═══════════════════════════════════════════════════════════════

def _draw_corr_matrix(corr_matrix, title, path, annot=True):
    """Render a correlation heatmap with per-cell annotations."""
    n   = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    im  = ax.imshow(corr_matrix.values, vmin=-1, vmax=1,
                    cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    ax.set_title(title, fontsize=11)
    if annot:
        for i in range(n):
            for j in range(n):
                val = corr_matrix.values[i, j]
                if np.isfinite(val):
                    c = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=7, color=c)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Corr matrix     → {path}")


def plot_correlation_matrices(df, output_dir):
    """
    Produce two correlation matrices of the seven core criticality metrics.

    Full matrix  — all records, regardless of PSD beta fit quality.
    Filtered matrix — records with PSD_Beta_R2 >= PSD_BETA_R2_THRESHOLD only.
    Poor-fit beta values are unreliable; excluding them gives a cleaner
    view of true inter-metric relationships.
    Both matrices are saved so the paper can discuss exclusions explicitly.
    """
    metrics = ["PSD_Beta", "DFA_Alpha1", "DFA_Alpha2",
               "SampEn", "RQA_DET", "RQA_REC", "RQA_LAM"]
    present = [m for m in metrics if m in df.columns]

    _draw_corr_matrix(
        df[present].corr(),
        title="Correlation Matrix — All Records (unfiltered)",
        path=os.path.join(output_dir, "Criticality_Correlation_Matrix.png")
    )

    if "PSD_Beta_R2" not in df.columns:
        print("  Filtered matrix: PSD_Beta_R2 missing, skipped.")
        return

    df_filt   = df[df["PSD_Beta_R2"] >= PSD_BETA_R2_THRESHOLD]
    n_kept    = len(df_filt)
    n_removed = len(df) - n_kept
    excluded  = df.index[df["PSD_Beta_R2"] < PSD_BETA_R2_THRESHOLD].tolist()

    if n_kept < 3:
        print(f"  Filtered matrix: only {n_kept} records pass "
              f"R²≥{PSD_BETA_R2_THRESHOLD}, need ≥3 — skipped.")
        return

    excl_str = ", ".join(
        f"{r} (R²={df.loc[r, 'PSD_Beta_R2']:.3f})" for r in excluded
    ) or "none"
    print(f"  Filtered matrix: {n_kept} records kept, "
          f"{n_removed} excluded ({excl_str})")

    _draw_corr_matrix(
        df_filt[present].corr(),
        title=(f"Correlation Matrix — R²≥{PSD_BETA_R2_THRESHOLD} filter  "
               f"({n_kept} records, {n_removed} excluded)"),
        path=os.path.join(output_dir,
                          "Criticality_Correlation_Matrix_Filtered.png")
    )


# ═══════════════════════════════════════════════════════════════
# STEP 8 — RADAR / SPIDER CHARTS
# ═══════════════════════════════════════════════════════════════

def plot_radar_charts(df, output_dir, cohort_label="Healthy",
                      cohort_m=None, cohort_s=None):
    """
    Produce two radar charts of the five core criticality axes.

    Parameters
    ----------
    df            : DataFrame of per-record metrics (one row per subject)
    output_dir    : where to save PNGs
    cohort_label  : string used in titles and output filenames
    cohort_m      : optional Series of means for z-score normalisation.
                    Pass the healthy cohort mean when plotting a psychiatric
                    group so both groups use the same reference frame.
                    If None, computed from df (correct for healthy cohort).
    cohort_s      : optional Series of SDs for z-score normalisation.

    Output files
    ------------
    Radar_Individual_{cohort_label}.png   — per-subject z-score overlay
    Radar_Cohort_Mean_{cohort_label}.png  — cohort mean in RAW metric units
                                            with ±1 SD band (not z-scores)

    Why raw units for the cohort mean chart?
    ----------------------------------------
    When a cohort is z-scored using its OWN mean and SD, the mean of the
    z-scored values is mathematically guaranteed to be 0 for every metric.
    This produces a perfectly regular polygon regardless of the actual data —
    visually uninformative and potentially misleading.

    The fix: the per-subject overlay keeps z-scores (useful for showing
    spread and outliers relative to the reference frame). The cohort mean
    chart instead plots actual metric values on per-spoke min-max scaled
    axes, with the ±1 SD band in the same units. Spoke labels include the
    raw mean value so the chart is self-documenting.

    When cohort_m / cohort_s are passed (psychiatric cohort call), z-scores
    ARE meaningful on the cohort mean chart because the psychiatric mean
    genuinely deviates from the healthy baseline. In that case the mean
    chart plots z-scores as intended.
    """
    valid = [(m, l) for m, l in zip(RADAR_METRICS, RADAR_LABELS)
             if m in df.columns]
    if len(valid) < 3:
        print(f"  Radar ({cohort_label}): fewer than 3 valid metrics — skipped.")
        return

    r_metrics, r_labels = zip(*valid)
    r_metrics = list(r_metrics)
    r_labels  = list(r_labels)
    N = len(r_metrics)

    sub = df[r_metrics].copy()

    # Determine whether this is a self-normalised call (healthy cohort)
    # or a cross-normalised call (psychiatric cohort vs healthy reference).
    self_normalised = (cohort_m is None)

    if cohort_m is None:
        cohort_m = sub.mean()
    if cohort_s is None:
        cohort_s = sub.std(ddof=1).replace(0, 1)

    z_df = (sub - cohort_m) / cohort_s

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def close(v):
        return list(v) + [v[0]]

    cmap    = plt.cm.tab10
    colours = [cmap(i / max(len(df), 1)) for i in range(len(df))]

    # ── Chart 1: individual subjects (z-scores) ──────────────
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for (rec_id, row), col in zip(z_df.iterrows(), colours):
        vals = close(row.values.tolist())
        ax.plot(angles, vals, lw=1.4, color=col, label=rec_id)
        ax.fill(angles, vals, alpha=0.05, color=col)
    for z_ring in [-1, 0, 1]:
        ax.plot(angles, close([z_ring] * N),
                color='grey', lw=0.6, linestyle=':', alpha=0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(r_labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(
        f"Criticality Profile — {cohort_label} Subjects\n"
        f"(z-scores, healthy cohort normalised)",
        fontsize=11, pad=22
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=8, framealpha=0.7)
    plt.tight_layout()
    p1 = os.path.join(output_dir, f"Radar_Individual_{cohort_label}.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Radar (indiv.)  → {p1}")

    # ── Chart 2: cohort mean ± 1 SD ──────────────────────────
    # For the HEALTHY cohort (self-normalised): plot in RAW metric units,
    # min-max scaled per spoke so all axes fit on the same polar plot.
    # For the PSYCHIATRIC cohort (cross-normalised): plot z-scores, which
    # are now genuinely informative deviations from the healthy baseline.

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    if self_normalised:
        # Raw units: min-max scale each spoke independently to [0, 1]
        raw_mean = sub.mean()
        raw_sd   = sub.std(ddof=1)
        raw_min  = sub.min()
        raw_max  = sub.max()
        span     = (raw_max - raw_min).replace(0, 1)

        mean_v = ((raw_mean - raw_min) / span).values.tolist()
        upper  = (((raw_mean + raw_sd) - raw_min) / span).clip(0, None).values.tolist()
        lower  = (((raw_mean - raw_sd) - raw_min) / span).clip(0, None).values.tolist()

        # Annotate each spoke label with actual mean value
        spoke_labels = [
            f"{lbl}\n({raw_mean[m]:.3f})"
            for lbl, m in zip(r_labels, r_metrics)
        ]
        ref_rings   = [0.25, 0.5, 0.75, 1.0]
        y_axis_note = "min-max scaled; spoke labels show raw mean"

    else:
        # z-scores relative to healthy reference
        mean_v = z_df.mean().values.tolist()
        raw_sd = sub.std(ddof=1)
        upper  = [m + s for m, s in zip(mean_v,
                  z_df.std(ddof=1).values.tolist())]
        lower  = [m - s for m, s in zip(mean_v,
                  z_df.std(ddof=1).values.tolist())]
        spoke_labels = r_labels
        ref_rings    = [-1, 0, 1]
        y_axis_note  = "z-scores (healthy cohort reference)"

    ax.fill(angles, close(upper), alpha=0.20, color='steelblue')
    ax.fill(angles, close(lower), alpha=1.00, color='white')
    ax.plot(angles, close(mean_v), lw=2.2, color='steelblue', label='Mean')
    ax.plot(angles, close(upper),  lw=1.0, color='steelblue',
            linestyle='--', alpha=0.7, label='+1 SD')
    ax.plot(angles, close(lower),  lw=1.0, color='steelblue',
            linestyle=':', alpha=0.7, label='−1 SD')
    ax.fill(angles, close(mean_v), alpha=0.12, color='steelblue')

    for ring in ref_rings:
        ax.plot(angles, close([ring] * N),
                color='grey', lw=0.6, linestyle=':', alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spoke_labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(
        f"{cohort_label} Cohort Mean Criticality Profile\n({y_axis_note})",
        fontsize=11, pad=22
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.7)
    plt.tight_layout()
    p2 = os.path.join(output_dir, f"Radar_Cohort_Mean_{cohort_label}.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Radar (cohort)  → {p2}")

    # Return normalisation params so psychiatric.py can reuse them
    return cohort_m, cohort_s


def plot_radar_comparison(df_healthy, df_psych, output_dir,
                          cohort_m, cohort_s):
    """
    Overlay chart: healthy and psychiatric cohort MEANS on the same
    polar axes, both z-scored on the healthy reference frame.

    This is the primary figure for the paper — it directly shows
    how the psychiatric profile deviates from the healthy baseline
    on each criticality dimension.

    Saved as: Radar_Comparison_Healthy_vs_Psychiatric.png
    """
    valid = [(m, l) for m, l in zip(RADAR_METRICS, RADAR_LABELS)
             if m in df_healthy.columns and m in df_psych.columns]
    if len(valid) < 3:
        print("  Radar comparison: fewer than 3 shared metrics — skipped.")
        return

    r_metrics, r_labels = zip(*valid)
    r_metrics = list(r_metrics)
    r_labels  = list(r_labels)
    N = len(r_metrics)

    def z_mean(df):
        sub    = df[r_metrics].copy()
        z      = (sub - cohort_m[r_metrics]) / cohort_s[r_metrics]
        return z.mean().values.tolist()

    def z_sd(df):
        sub = df[r_metrics].copy()
        z   = (sub - cohort_m[r_metrics]) / cohort_s[r_metrics]
        return z.std(ddof=1).values.tolist()

    h_mean = z_mean(df_healthy)
    h_sd   = z_sd(df_healthy)
    p_mean = z_mean(df_psych)
    p_sd   = z_sd(df_psych)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def close(v):
        return list(v) + [v[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Healthy band + mean
    h_upper = [m + s for m, s in zip(h_mean, h_sd)]
    h_lower = [m - s for m, s in zip(h_mean, h_sd)]
    ax.fill(angles, close(h_upper), alpha=0.15, color='steelblue')
    ax.fill(angles, close(h_lower), alpha=1.00, color='white')
    ax.plot(angles, close(h_mean), lw=2.2, color='steelblue',
            label='Healthy Mean')
    ax.fill(angles, close(h_mean), alpha=0.08, color='steelblue')

    # Psychiatric band + mean
    p_upper = [m + s for m, s in zip(p_mean, p_sd)]
    p_lower = [m - s for m, s in zip(p_mean, p_sd)]
    ax.fill(angles, close(p_upper), alpha=0.15, color='firebrick')
    ax.fill(angles, close(p_lower), alpha=1.00, color='white')
    ax.plot(angles, close(p_mean), lw=2.2, color='firebrick',
            label='Psychiatric Mean')
    ax.fill(angles, close(p_mean), alpha=0.08, color='firebrick')

    # Reference rings at z = −1, 0, +1
    for ring in [-1, 0, 1]:
        ax.plot(angles, close([ring] * N),
                color='grey', lw=0.6, linestyle=':', alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(r_labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title(
        "Criticality Profile: Healthy vs Psychiatric\n"
        "(z-scores, healthy cohort reference  ±1 SD bands)",
        fontsize=12, pad=24
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.15),
              fontsize=10, framealpha=0.8)
    plt.tight_layout()

    path = os.path.join(output_dir,
                        "Radar_Comparison_Healthy_vs_Psychiatric.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Radar (comparison) → {path}")


# ═══════════════════════════════════════════════════════════════
# STEP 9 — FULL RECORD ANALYSIS (one WFDB record)
# ═══════════════════════════════════════════════════════════════

def analyze_record(record_path, lead_index=0, plot=True,
                   run_surrogates=True, verbose=True,
                   output_dir="outputs"):
    """
    End-to-end pipeline for one WFDB record.

    Parameters
    ----------
    record_path   : str   path to WFDB record (no extension)
    lead_index    : int   ECG lead to use (default 0 = Lead II for MIT-BIH)
    plot          : bool  save six-panel diagnostic + MSE curve PNGs
    run_surrogates: bool  run IAAFT surrogate testing
    verbose       : bool  print progress
    output_dir    : str   folder for all output files

    Returns
    -------
    dict  flat dict of all computed metrics (including SURR_* columns)
    """
    record_id = os.path.basename(record_path)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}\n  Analyzing record: {record_id}\n{'='*60}")

    record        = wfdb.rdrecord(record_path)
    sampling_rate = record.fs
    if verbose:
        print(f"  Sampling rate: {sampling_rate} Hz")

    lead      = record.p_signal[:, lead_index]
    n_samples = ANALYSIS_MINUTES * 60 * sampling_rate
    lead      = lead[:n_samples]
    time      = np.arange(len(lead)) / sampling_rate

    if verbose:
        print(f"  Processing {ANALYSIS_MINUTES} min / {len(lead)} samples...")
    signals, info = nk.ecg_process(lead, sampling_rate=sampling_rate)
    cleaned_ecg   = signals["ECG_Clean"]
    r_peaks       = info["ECG_R_Peaks"]

    r_peak_times = r_peaks / sampling_rate
    rr_raw       = np.diff(r_peak_times)
    rr_times_raw = r_peak_times[1:]
    if verbose:
        print(f"  Detected {len(r_peaks)} R-peaks → {len(rr_raw)} RR intervals")

    rr_clean, artifact_pct, num_rr = reject_ectopic_beats(rr_raw, verbose=verbose)
    rr_times_clean = np.concatenate(
        [[rr_times_raw[0]], rr_times_raw[0] + np.cumsum(rr_clean[:-1])]
    )

    if verbose: print("  Time-domain HRV...")
    hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)

    if verbose: print("  Poincaré / nonlinear HRV...")
    hrv_nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)

    if verbose: print("  Frequency-domain HRV + PSD beta...")
    freq_metrics, freqs, power = compute_frequency_domain(rr_clean, rr_times_clean)

    if verbose: print("  Complexity metrics (SampEn, ApEn, PermEn, MSE, DFA, RQA)...")
    complexity_metrics, mse_scales, mse_values = compute_complexity_metrics(rr_clean)

    if plot:
        _plot_mse_curve(record_id, mse_scales, mse_values, output_dir)

    # ── Assemble result dict ──────────────────────────────────
    result = {"Record_ID": record_id}

    result["Num_RR"]           = num_rr
    result["Artifact_Percent"] = artifact_pct
    result["QC_RR_Count_OK"]   = num_rr >= 240
    result["QC_Artifact_OK"]   = artifact_pct < 5.0

    for col in ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50"]:
        result[col] = hrv_time[col].values[0] if col in hrv_time.columns else np.nan

    # Compute MeanHR directly to avoid NeuroKit2 column-naming variance
    result["HRV_MeanHR"] = (float(60.0 / np.mean(rr_clean))
                            if len(rr_clean) > 0 else np.nan)

    for col in ["HRV_SD1", "HRV_SD2", "HRV_SD1SD2"]:
        result[col] = hrv_nonlinear[col].values[0] if col in hrv_nonlinear.columns else np.nan

    result.update(freq_metrics)
    result.update(complexity_metrics)

    result["QC_MSE_Valid"]    = np.isfinite(result.get("MSE_Mean",   np.nan))
    result["QC_DFA1_Valid"]   = np.isfinite(result.get("DFA_Alpha1", np.nan))
    result["QC_SampEn_Valid"] = np.isfinite(result.get("SampEn",     np.nan))
    result["QC_Overall_Pass"] = (
        result["QC_RR_Count_OK"]
        and result["QC_Artifact_OK"]
        and result["QC_DFA1_Valid"]
    )

    if run_surrogates:
        surr = run_surrogate_testing(rr_clean, record_id, output_dir,
                                     verbose=verbose)
        result.update(surr)

    if plot:
        _plot_record(record_id, time, lead, cleaned_ecg, r_peaks,
                     rr_times_raw, rr_raw, rr_times_clean, rr_clean,
                     freqs, power, complexity_metrics,
                     psd_beta=freq_metrics.get("PSD_Beta", np.nan),
                     output_dir=output_dir)

    if verbose:
        print(f"\n  ✓ Done: {record_id}")

    return result


# ═══════════════════════════════════════════════════════════════
# STEP 10 — MULTI-RECORD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_analysis(record_paths, cohort_label="Healthy",
                 lead_index=0, plot=True,
                 run_surrogates=True, save_csv=True,
                 output_dir="outputs",
                 cohort_m=None, cohort_s=None):
    """
    Run the full pipeline across a list of WFDB record paths.

    Parameters
    ----------
    record_paths  : list[str]  WFDB record paths (no extension)
    cohort_label  : str        used in output filenames and print headers
    lead_index    : int        ECG lead index
    plot          : bool       save per-record diagnostic PNGs
    run_surrogates: bool       run IAAFT surrogate testing per record
    save_csv      : bool       save results CSV
    output_dir    : str        all outputs go here
    cohort_m      : Series     optional external normalisation mean
                               (pass healthy cohort mean for psychiatric run)
    cohort_s      : Series     optional external normalisation SD

    Returns
    -------
    df            : DataFrame  per-record results (rows = records)
    cohort_m      : Series     normalisation means  (for reuse in comparison)
    cohort_s      : Series     normalisation SDs
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for path in record_paths:
        try:
            result = analyze_record(
                path, lead_index=lead_index,
                plot=plot, run_surrogates=run_surrogates,
                output_dir=output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ Failed on {path}: {e}")
            all_results.append({"Record_ID": os.path.basename(path)})

    df = pd.DataFrame(all_results).set_index("Record_ID")

    # ── Results table ─────────────────────────────────────────
    print(f"\n{'='*60}\n  {cohort_label.upper()} RESULTS TABLE\n{'='*60}")
    with pd.option_context('display.max_columns', None, 'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(df.T.to_string())

    # ── QA summary ────────────────────────────────────────────
    qa_cols = ["Num_RR", "Artifact_Percent", "QC_RR_Count_OK",
               "QC_Artifact_OK", "QC_SampEn_Valid", "QC_MSE_Valid",
               "QC_DFA1_Valid", "QC_Overall_Pass"]
    qa_present = [c for c in qa_cols if c in df.columns]
    qa_df = df[qa_present].copy()
    for col in [c for c in qa_present if c.startswith("QC_")]:
        qa_df[col] = qa_df[col].map({True: "PASS", False: "FAIL"})
    print(f"\n{'='*60}\n  QA SUMMARY\n{'='*60}")
    print(qa_df.to_string())
    if "QC_Overall_Pass" in df.columns:
        n_pass = df["QC_Overall_Pass"].sum()
        n_tot  = len(df)
        print(f"\n  Overall QC: {n_pass}/{n_tot} records passed")
        failed = df.index[~df["QC_Overall_Pass"]].tolist()
        print(f"  Failed: {failed}" if failed else "  All records cleared QC.")

    # ── Cohort stats ──────────────────────────────────────────
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = [c for c in df.columns if c.startswith("QC_")]
    stat_cols = [c for c in num_cols if c not in bool_cols]
    stats_df  = pd.DataFrame(
        [df[stat_cols].mean(),
         df[stat_cols].median(),
         df[stat_cols].std(ddof=1)],
        index=[f"{cohort_label}_Mean",
               f"{cohort_label}_Median",
               f"{cohort_label}_SD"]
    )
    print(f"\n{'='*60}\n  {cohort_label.upper()} COHORT SUMMARY\n{'='*60}")
    with pd.option_context('display.max_columns', None, 'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(stats_df.T.to_string())

    # ── PSD beta fit quality ──────────────────────────────────
    if "PSD_Beta" in df.columns and "PSD_Beta_R2" in df.columns:
        print(f"\n{'='*60}\n  PSD BETA FIT QUALITY\n{'='*60}")
        print(df[["PSD_Beta", "PSD_Beta_R2"]].sort_values("PSD_Beta_R2").to_string())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df["PSD_Beta"], df["PSD_Beta_R2"], s=60)
        for idx, row in df.iterrows():
            ax.annotate(idx, (row["PSD_Beta"], row["PSD_Beta_R2"]), fontsize=8)
        ax.axhline(PSD_BETA_R2_THRESHOLD, color='red', linestyle='--', lw=1.2,
                   label=f'R²={PSD_BETA_R2_THRESHOLD} threshold')
        ax.axvline(1.0, color='grey', linestyle='--', lw=1.0)
        ax.set_xlabel("PSD Beta"); ax.set_ylabel("PSD Beta R²")
        ax.set_title(f"PSD Scaling Exponent vs Fit Quality — {cohort_label}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"PSD_Beta_vs_R2_{cohort_label}.png"),
            dpi=150
        )
        plt.close()

    # ── Scatter plots ─────────────────────────────────────────
    scatter_pairs = [
        ("DFA_Alpha1", "PSD_Beta", f"DFA_vs_PSD_Beta_{cohort_label}.png"),
        ("DFA_Alpha1", "RQA_DET",  f"DFA_vs_RQA_DET_{cohort_label}.png"),
        ("PSD_Beta",   "RQA_DET",  f"PSD_Beta_vs_RQA_DET_{cohort_label}.png"),
    ]
    for xc, yc, fname in scatter_pairs:
        if xc not in df.columns or yc not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[xc], df[yc], s=60)
        for idx, row in df.iterrows():
            ax.annotate(idx, (row[xc], row[yc]), fontsize=8)
        r = df[xc].corr(df[yc])
        ax.set_title(f"{xc} vs {yc}  [{cohort_label}]\nr = {r:.3f}")
        ax.set_xlabel(xc); ax.set_ylabel(yc)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()

    # ── Correlation matrices ──────────────────────────────────
    plot_correlation_matrices(df, output_dir)

    # ── Radar charts ──────────────────────────────────────────
    returned = plot_radar_charts(df, output_dir,
                                 cohort_label=cohort_label,
                                 cohort_m=cohort_m,
                                 cohort_s=cohort_s)
    if returned is not None:
        cohort_m, cohort_s = returned

    # ── CSV ───────────────────────────────────────────────────
    if save_csv:
        csv_path  = os.path.join(output_dir, f"ecg_results_{cohort_label}.csv")
        df_export = pd.concat([df, stats_df])
        df_export.to_csv(csv_path)
        print(f"\n  Results saved  → {csv_path}")

    return df, cohort_m, cohort_s