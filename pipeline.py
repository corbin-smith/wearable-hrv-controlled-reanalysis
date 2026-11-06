"""pipeline.py: shared analysis functions, imported by the cohort and sensitivity scripts.

Holds no cohort paths, record lists or output folders; those belong to the calling scripts.

Contents: configuration constants; artifact rejection; frequency domain (Welch PSD, band powers,
spectral slope variants); complexity measures (entropy, MSE, DFA, RQA, MFDFA, recurrence networks);
IAAFT surrogate testing; resolution controls; per-record analysis for WFDB (MIT-BIH) and CSV
(HRV-ACC) input; plotting helpers; estimator validation and the environment record.

Command line:  python pipeline.py --validate-estimators [--out DIR]
               python pipeline.py --environment [--out DIR]
"""

import wfdb
import neurokit2 as nk
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for batch runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch, resample_poly
from scipy.optimize import curve_fit
from neurokit2.complexity import fractal_dfa
from scipy.stats import linregress, norm as _normdist
from scipy.special import polygamma
import warnings
import os

# Silence only specific, expected warnings; numerical warnings stay visible.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")


# -- Optional dependency, resolved once at import --
try:
    from statsmodels.tsa.stattools import adfuller as _adfuller, kpss as _kpss
    STATSMODELS_AVAILABLE = True
    STATSMODELS_STATUS = "available"
except Exception as _sm_exc:                      # noqa: BLE001
    _adfuller = _kpss = None
    STATSMODELS_AVAILABLE = False
    STATSMODELS_STATUS = f"unavailable ({type(_sm_exc).__name__})"
    print("[pipeline] statsmodels not installed: ADF and KPSS stationarity "
          "tests will be skipped and reported as NaN. The reverse-arrangement "
          "test does not depend on it and still runs. "
          "Install with: pip install statsmodels")


# ---- Configuration ----

VLF_BAND              = (0.003, 0.04)
LF_BAND               = (0.04,  0.15)
HF_BAND               = (0.15,  0.40)
# Respiratory peak search window: HF band only. Below 0.15 Hz slow breathing and the baroreflex
# (Mayer) resonance cannot be told apart at this resolution, so such peaks are reported separately as
# low-frequency peaks and never called respiratory.
RESP_SEARCH_BAND      = (0.15,  0.40)
RESP_LF_BAND          = (0.10,  0.15)
# Effective-average factor assumed for 50%-overlap Hann Welch segments, used for the analytic residual
# noise level of the respiratory-peak test. A simulation with these Welch settings gives about 0.95, so
# the test is about 8% stricter than its nominal alpha (Supplementary Methods S11). Kept at 0.82
# because the reported results used it.
WELCH_OVERLAP_EFFICIENCY = 0.82
WELCH_SEGMENTS_DEFAULT   = 27      # fallback if the caller does not supply it
BETA_FIT_BAND         = (0.01,  0.15)   # primary: VLF–LF range for 1/f fit
BETA_FIT_BAND_WIDE    = (0.003, 0.15)   # sensitivity alt-1: full VLF–LF
BETA_FIT_BAND_FULL    = (0.003, 0.40)   # sensitivity alt-2: full spectrum
INTERP_FS             = 4.0             # Hz — uniform RR grid for PSD
ANALYSIS_MINUTES      = 30

# -- Analysis window (single definition; every HRV-ACC script imports these) --
SKIP_SECONDS          = 600     # discard the first 10 minutes
ANALYSIS_SECONDS      = 1800    # analyse the next 30 minutes
assert ANALYSIS_SECONDS == ANALYSIS_MINUTES * 60, (
    "ANALYSIS_MINUTES and ANALYSIS_SECONDS disagree; they describe the same "
    "window in different units and must be kept consistent.")
RR_MIN_MS             = 300
RR_MAX_MS             = 2000
RR_CHANGE_THRESHOLD   = 0.20

# -- Artifact filter geometry --
# The local median is taken over a centered window of 2*RR_MEDIAN_HALFWIDTH + 1 beats (11 beats).
RR_MEDIAN_HALFWIDTH   = 5
RR_ARTIFACT_METHOD    = "delete"   # "delete" | "interpolate"
# Variability-scaled artifact threshold, a sensitivity variant only: a fixed multiple of the
# subject's RMSSD rather than a fraction of the local median. No reported result uses it; it helps
# interpret the group difference in artifact rate.
RR_ARTIFACT_RMSSD_MULT = 8.0
                                   # Flagged beats are removed (default); "interpolate" replaces them by linear interpolation
                                   # (sensitivity analysis).

# -- Welch PSD specification --
# At INTERP_FS = 4 Hz, frequency resolution = INTERP_FS / nperseg: 256 -> 0.0156 Hz, 512 -> 0.0078 Hz,
# 1024 -> 0.0039 Hz. 512 is the smallest power of two that resolves a 0.01 Hz lower fitting bound.
WELCH_NPERSEG         = 512
WELCH_OVERLAP_FRAC    = 0.5
WELCH_WINDOW          = "hann"
WELCH_DETREND         = "constant"
# Per-segment detrending for the sensitivity variant PSD_Beta_LinDetrend (Welch's default removes
# the segment mean only; a linear trend inside a segment leaks power into the lowest bins).
WELCH_DETREND_SENS    = "linear"
WELCH_NPERSEG_SENS    = (256, 512, 1024)   # segment-length sensitivity sweep

# -- Slope estimation --
# Welch bins are linearly spaced, so on a log-frequency axis an unweighted fit is dominated by the top
# of the band. Log-binning (n_bins = clip(n_points // 2, 5, 12)) equalizes leverage but is unstable on a
# band this narrow and is a sensitivity variant only. The Mayer-band excision and the explicit peak
# model are the main controls.
BETA_LOG_BINS         = None      # None = derive from the data (recommended)
BETA_LOG_BINS_MIN     = 5
BETA_LOG_BINS_MAX     = 12
MAYER_BAND            = (0.07, 0.13)   # excised in the Mayer-wave control fit

# -- Slope band sweep --
# The slope is estimated over several bands in one pass. The "low" band (0.0156-0.07 Hz) lies below
# both the baroreflex resonance and any respiratory contribution; in simulation, respiration at 0.150
# and 0.160 Hz biases a 0.0156-0.148 Hz fit by -0.27 and -0.11, while the low-band fit is unbiased for
# respiration from 0.150 to 0.400 Hz (estimator SD 0.071 against 0.042).
# Bounds are nominal; the realized band is recorded per subject in the *_LowHz and *_HighHz columns.
# Each entry is (tag, f_low, f_high, exclude_band_or_None).
BETA_BAND_SWEEP = [
    ("low",      0.01,   0.07,  None),          # below Mayer AND below respiration
    ("verylow",  0.01,   0.05,  None),          # stricter still
    ("lowmid",   0.01,   0.10,  None),
    ("mayerexc", 0.01,   0.15,  (0.07, 0.13)),  # legacy primary, identical band
    ("nomayerx", 0.01,   0.15,  None),          # legacy uncorrected, identical band
    ("wideband", 0.003,  0.40,  None),          # full spectrum
]
SPECPARAM_PEAK_BAND   = (0.06, 0.14)   # allowed peak centre for the peak model
IRASA_HSET            = np.arange(1.05, 1.55, 0.05)
IRASA_NPERSEG         = 1024

# -- Entropy --
# Tolerance r = 0.2 x SD (Richman and Moorman, 2000), set numerically and recorded.
ENTROPY_M             = 2
ENTROPY_R_FRAC        = 0.20
# Tolerance sweep for the entropy sensitivity analysis.
ENTROPY_R_SENSITIVITY = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)

# -- Multiscale entropy --
# MSE_MIN_POINTS is the minimum coarse-grained length per retained scale; MSE_MAX_SCALE caps the scale.
MSE_MIN_POINTS        = 100
MSE_MAX_SCALE         = 20

# -- Recurrence quantification and networks --
# The threshold is set by exact count so that the recurrence rate is fixed for every subject.
RQA_REC_RATE          = 0.10
RQA_DIM               = 2
RQA_DELAY             = 1
RQA_L_MIN             = 2
RQA_V_MIN             = 2
RNA_MAX_PTS           = None    # None = use the full embedded series.
                                # None uses the whole series. If a cap is needed, a contiguous window is used, not decimation.

# -- MFDFA --
# Positive q only by default; the full q range is also computed and stored, labeled unstable.
# -- Theiler window --
# Minimum temporal separation (samples) for a pair to count as a recurrence; temporally adjacent
# pairs recur trivially and inflate determinism.
#   None  -> the embedding window, (dim - 1) * delay + 1
#   "mi"  -> per-subject first minimum of the mutual-information function
#   int   -> a fixed window in samples
RQA_THEILER           = "mi"
RQA_THEILER_FALLBACK  = None    # None -> embedding window
# Minimum diagonal line lengths for the determinism sensitivity check.
RQA_LMIN_SWEEP        = (3, 4)
# Scale set for MSE_Mean_CommonScales, fixed in advance; every subject retains these scales.
MSE_COMMON_SCALES     = tuple(range(1, 11))
# Set by recurrence_matrix() on each call and read immediately by rqa_metrics(). The pipeline
# processes one subject at a time.
_RECURRENCE_PAIR_INFO = {}
# Same single-subject carrier pattern as _RECURRENCE_PAIR_INFO.
_MFDFA_RANGE_INFO = {}
# Output folders whose environment record has been written in this session.
_ENVIRONMENT_WRITTEN = set()

# Estimator sensitivity sweeps. The alpha1 fit is over 4..16 beats, so lower bounds must leave
# at least MIN_FIT_SCALES scales.
DFA_ALPHA1_NMIN_SWEEP = (8, 10)    # alpha1 lower bound; primary is 4
MIN_FIT_SCALES        = 4          # refuse a log-log fit below this
PERMEN_DIM_SWEEP      = (4, 5)     # permutation entropy dimension; primary 3
MFDFA_SMAX_DIVISORS   = (4, 10)    # upper scale bound as N // d; primary d = 4

MFDFA_Q_PRIMARY       = (1, 2, 3, 4, 5)
MFDFA_Q_FULL          = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
MFDFA_N_SURROGATES    = 20      # shuffled + phase-randomised multifractal null

# -- Resolution (quantization) controls --
# The step is the measured 1.000 ms: observed_rr_lattice() finds every HRV-ACC series on a 1 ms grid
# (the Bluetooth unit is 1/1024 s). QUANT_STEP_NOMINAL_S is kept for reference only.
# The fixed step against group-different beat-to-beat amplitude makes the step-to-RMSSD ratio differ
# between groups, so every entropy and recurrence measure is recomputed with dither and with the
# series coarsened to a common ratio.
QUANT_STEP_S          = 0.001
QUANT_STEP_NOMINAL_S  = 1.0 / 1024.0
# Seconds skipped at the start of each MIT-BIH record: 600 s, as for HRV-ACC, so all three cohorts
# cover minutes 10 to 40. Without the skip the first ten minutes ran faster than the rest of the
# window; mitbih_onset_check() reports the comparison for the analyzed window.
MITBIH_SKIP_SECONDS   = 600

MITBIH_QUANT_STEP_S   = 1.0 / 128.0    # MIT-BIH NSR sampling period
QUANT_SEED            = 1234
# Common step-to-RMSSD ratio for matched coarsening: the 1.000 ms step over the patient mean RMSSD
# (11.41 ms) = 0.0876, rounded to 0.088. Coarsening applies only where target x RMSSD exceeds the
# step, i.e. to subjects above the patient mean (28 of 30 controls, 12 of 30 patients), so controls
# are brought down to the patients' resolution and no group gains an advantage.
QUANT_TARGET_RATIO    = 0.088
# Number of independent dither realizations, each with a seed derived from QUANT_SEED.
QUANT_DITHER_REPS     = 20
# Cross-device re-quantization: coarsen every Polar series onto the MIT-BIH grid (1/128 s) and
# recompute the battery. None skips it.
REQUANT_TARGET_STEP_S = 1.0 / 128.0
REQUANT_ENABLED       = True

# Time-matched DFA: target windows in seconds; per subject the beat scales are
# round(seconds / mean_NN), so every subject is fitted over the same temporal band.
DFA_TIME_MATCHED_S    = ((4.0, 16.0), (16.0, 64.0))
# Sweep of alternative matched-coarsening targets, computed in one pass.
QUANT_TARGET_SWEEP    = (0.10, 0.15, 0.20)

# -- Series-length matching --
# Set TRUNCATE_N to a common beat count to rerun the length-dependent measures at matched length.
TRUNCATE_N            = None

# -- Stationarity --
STATIONARITY_SEGMENTS = 40


# -- Surrogate testing --
N_SURROGATES          = 100    # ≥100 is publication standard
N_IAAFT_ITERS         = 1000   # iterations per surrogate for IAAFT convergence
SURROGATE_ALPHA       = 0.05   # two-tailed significance level
# Seed for the IAAFT surrogate generator.
SURROGATE_SEED        = 42

# -- Correlation matrix --
PSD_BETA_R2_THRESHOLD = 0.70   # records below this excluded from filtered matrix

# -- Radar chart axes --
RADAR_METRICS = ["PSD_Beta", "DFA_Alpha1", "SampEn", "MSE_Mean", "RQA_DET"]
RADAR_LABELS  = ["PSD β",    "DFA α₁",    "SampEn", "MSE\nMean", "RQA\nDET"]


# ---- Step 1: artifact and ectopic beat rejection ----

def reject_ectopic_beats(rr_intervals_seconds, verbose=True,
                         method=None, halfwidth=None):
    """Two-pass R-R artifact filter.

      Pass 1: physiological bounds (RR_MIN_MS to RR_MAX_MS).
      Pass 2: a beat is flagged when it deviates by more than RR_CHANGE_THRESHOLD from the median of a
              centered window of 2*halfwidth + 1 beats (11 by default).

    method="delete" (default) removes flagged beats and rebuilds the time base from the remaining
    intervals. method="interpolate" replaces beats flagged by pass 2 by linear interpolation (pass-1
    failures are still removed); it smooths the series locally and is for sensitivity analysis.

    Returns
    -------
    rr_clean         : np.ndarray  cleaned R-R series in seconds
    artifact_percent : float       percentage of input beats flagged
    n_clean          : int         beats in the returned series
    detail           : dict        per-pass counts
    """
    method    = RR_ARTIFACT_METHOD if method is None else method
    halfwidth = RR_MEDIAN_HALFWIDTH if halfwidth is None else halfwidth

    rr_in       = np.asarray(rr_intervals_seconds, dtype=float)
    n_input     = len(rr_in)
    rr_ms_all   = rr_in * 1000.0
    mask_bounds = (rr_ms_all >= RR_MIN_MS) & (rr_ms_all <= RR_MAX_MS)
    n_bounds    = int((~mask_bounds).sum())
    rr_ms       = rr_ms_all[mask_bounds]

    # Vectorized: interior beats share one median call; only the 2*halfwidth truncated windows at the
    # ends use the loop. Equality with the per-beat loop is asserted by _median_filter_selftest().
    n_beats = len(rr_ms)
    win = 2 * halfwidth + 1
    med = np.empty(n_beats, dtype=float)
    if n_beats >= win:
        windows = np.lib.stride_tricks.sliding_window_view(rr_ms, win)
        med[halfwidth:n_beats - halfwidth] = np.median(windows, axis=1)
        edges = list(range(halfwidth)) + list(range(n_beats - halfwidth, n_beats))
    else:
        edges = list(range(n_beats))
    for i in edges:
        lo = max(0, i - halfwidth)
        hi = min(n_beats, i + halfwidth + 1)
        med[i] = np.median(rr_ms[lo:hi])
    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = np.abs(rr_ms - med) / med
    clean_mask = ~((med > 0) & (deviation > RR_CHANGE_THRESHOLD))
    n_median = int((~clean_mask).sum())

    if method == "interpolate" and n_median > 0 and clean_mask.sum() >= 2:
        idx      = np.arange(len(rr_ms), dtype=float)
        rr_fixed = rr_ms.copy()
        rr_fixed[~clean_mask] = np.interp(idx[~clean_mask],
                                          idx[clean_mask],
                                          rr_ms[clean_mask])
        rr_clean = rr_fixed / 1000.0
    else:
        rr_clean = rr_ms[clean_mask] / 1000.0

    removed          = n_bounds + n_median
    artifact_percent = 100.0 * removed / n_input if n_input else np.nan

    # -- Threshold in variability units --
    # RR_CHANGE_THRESHOLD is a fraction of heart period; expressed in multiples of the subject's RMSSD it
    # shows how strict the filter was for that subject.
    try:
        _rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) \
            if len(rr_ms) > 2 else np.nan
        _thresh_ms = RR_CHANGE_THRESHOLD * float(np.median(rr_ms)) \
            if len(rr_ms) else np.nan
        _thresh_in_rmssd = (_thresh_ms / _rmssd_ms
                            if (np.isfinite(_rmssd_ms) and _rmssd_ms > 0)
                            else np.nan)
    except Exception:
        _rmssd_ms, _thresh_ms, _thresh_in_rmssd = np.nan, np.nan, np.nan

    # -- Temporal distribution of flagged beats --
    try:
        _flag_idx = np.flatnonzero(~clean_mask)
        _n = max(1, len(rr_ms))
        _thirds = [float(100.0 * np.mean((_flag_idx >= lo) & (_flag_idx < hi)))
                   if len(_flag_idx) else 0.0
                   for lo, hi in ((0, _n // 3), (_n // 3, 2 * _n // 3),
                                  (2 * _n // 3, _n))]
        _dev_finite = deviation[np.isfinite(deviation)]
        _dev_max = float(np.max(_dev_finite)) if _dev_finite.size else np.nan
        _dev_iqr = (float(np.percentile(_dev_finite, 75) -
                          np.percentile(_dev_finite, 25))
                    if _dev_finite.size else np.nan)
        _max_run = 0
        if len(_flag_idx) > 1:
            _breaks = np.diff(_flag_idx) != 1
            _runs = np.split(_flag_idx, np.flatnonzero(_breaks) + 1)
            _max_run = int(max(len(r) for r in _runs))
        elif len(_flag_idx) == 1:
            _max_run = 1
    except Exception:
        _thirds = [np.nan, np.nan, np.nan]
        _dev_max, _dev_iqr, _max_run = np.nan, np.nan, 0
        _flag_idx = np.array([], dtype=int)

    detail = {
        "N_Input_Beats":      n_input,
        "N_Flag_Bounds":      n_bounds,
        "N_Flag_Median":      n_median,
        "N_Flagged_Total":    removed,
        "Artifact_Method":    method,
        "Median_Window_Beats": 2 * halfwidth + 1,
        "N_Retained":         int(len(rr_clean)),
        # Threshold in variability units.
        "Artifact_Thresh_Frac":      RR_CHANGE_THRESHOLD,
        "Artifact_Thresh_ms":        _thresh_ms,
        "Artifact_Thresh_Over_RMSSD": _thresh_in_rmssd,
        "RMSSD_PreClean_ms":         _rmssd_ms,
        # Distribution rather than a single rate.
        "Artifact_Pct_Third1":       _thirds[0],
        "Artifact_Pct_Third2":       _thirds[1],
        "Artifact_Pct_Third3":       _thirds[2],
        "Artifact_Dev_Max":          _dev_max,
        "Artifact_Dev_IQR":          _dev_iqr,
        "Artifact_Max_Consecutive":  _max_run,
        # Indices of flagged beats, kept so artifacts can be split by ambulation bout downstream.
        "Artifact_Flag_Indices":     ";".join(str(int(i)) for i in _flag_idx),
    }

    if verbose:
        print(f"    Artifact rejection [{method}, "
              f"{2*halfwidth+1}-beat median window]: "
              f"{n_bounds} out-of-bounds + {n_median} median-outliers "
              f"= {removed} flagged ({artifact_percent:.2f}%); "
              f"{len(rr_clean)} retained.")
    return rr_clean, artifact_percent, len(rr_clean), detail


def load_rr_window(filepath, skip_seconds=None, analysis_seconds=None,
                   verbose=False):
    """Load a Polar H10 / HRV-ACC R-R CSV and return the analysis window.

    Reads the semicolon-delimited export, converts ms to s, builds the time axis as the cumulative sum of
    the intervals (the timestamp column is not used), and returns the beats in [skip, skip + analysis).

    Returns
    -------
    rr_window   : np.ndarray  R-R intervals in seconds, not yet artifact-filtered
    time_window : np.ndarray  cumulative time in seconds, same length
    n_total     : int         beats in the file before windowing
    """
    skip = SKIP_SECONDS if skip_seconds is None else skip_seconds
    length = ANALYSIS_SECONDS if analysis_seconds is None else analysis_seconds

    df = pd.read_csv(filepath, sep=";")
    rr_col = None
    for col in df.columns:
        if "rr" in col.lower() or "interval" in col.lower():
            rr_col = col
            break
    if rr_col is None:
        raise ValueError(f"Cannot find RR column in {filepath}. "
                         f"Columns: {df.columns.tolist()}")

    rr_ms = pd.to_numeric(df[rr_col], errors="coerce").dropna().values
    n_total = len(rr_ms)
    rr_s = rr_ms / 1000.0
    cum_time = np.cumsum(rr_s)

    mask = (cum_time >= skip) & (cum_time < skip + length)
    rr_window = rr_s[mask]
    time_window = cum_time[mask]

    if verbose:
        total = cum_time[-1] if len(cum_time) else 0.0
        print(f"    Loaded {n_total} beats, total duration {total/60:.1f} min")
        print(f"    Window [{skip}s - {skip+length}s]: {len(rr_window)} beats")

    if len(rr_window) < 50:
        raise ValueError(f"Only {len(rr_window)} beats in the analysis window "
                         f"for {filepath}; check the skip and window length.")
    return rr_window, time_window, n_total


def _median_filter_selftest():
    """Assert that the vectorized rolling-median filter reproduces the per-beat loop, including the
    truncated end windows and series shorter than one window. Runs at import.
    """
    rng = np.random.default_rng(0)
    for n in (3, 11, 12, 200):
        for halfwidth in (1, 5):
            x = 800.0 + 60.0 * rng.standard_normal(n)
            x[rng.integers(0, n, max(1, n // 10))] *= 1.8      # inject outliers
            # reference: per-beat loop
            ref = np.ones(n, dtype=bool)
            for i in range(n):
                lo, hi = max(0, i - halfwidth), min(n, i + halfwidth + 1)
                m = np.median(x[lo:hi])
                if m > 0 and abs(x[i] - m) / m > RR_CHANGE_THRESHOLD:
                    ref[i] = False
            # vectorized: as in reject_ectopic_beats
            win = 2 * halfwidth + 1
            med = np.empty(n, dtype=float)
            if n >= win:
                w = np.lib.stride_tricks.sliding_window_view(x, win)
                med[halfwidth:n - halfwidth] = np.median(w, axis=1)
                edges = list(range(halfwidth)) + list(range(n - halfwidth, n))
            else:
                edges = list(range(n))
            for i in edges:
                lo, hi = max(0, i - halfwidth), min(n, i + halfwidth + 1)
                med[i] = np.median(x[lo:hi])
            with np.errstate(invalid="ignore", divide="ignore"):
                dev = np.abs(x - med) / med
            got = ~((med > 0) & (dev > RR_CHANGE_THRESHOLD))
            assert np.array_equal(ref, got), (
                f"median-filter self-test failed at n={n}, halfwidth={halfwidth}")


_median_filter_selftest()


def truncate_series(rr, n=None):
    """Truncate an RR series to a common beat count for length-matched reruns."""
    n = TRUNCATE_N if n is None else n
    if n is None or len(rr) <= n:
        return rr
    return rr[:n]


def _fit_beta(freqs, power, fit_band, exclude_band=None,
              log_bin=False, n_bins=None):
    """Fit the spectral slope over a frequency band by log-log regression.

    Parameters
    ----------
    freqs, power : arrays from Welch
    fit_band     : (f_low, f_high) Hz
    exclude_band : optional (f_low, f_high) removed before fitting (0.07-0.13 Hz for the Mayer-band
                   excised fit)
    log_bin      : average log-power within log-spaced bins before fitting

    On synthetic 1/f signals with an added 0.1 Hz oscillation, linear-bin OLS underestimates the
    exponent by 0.3-0.6, while the band-excised fit recovers it.

    Returns
    -------
    (beta, r2, n_points). beta is the negated slope, so a 1/f spectrum gives beta of about +1.
    """
    try:
        f = np.asarray(freqs, dtype=float)
        p = np.asarray(power, dtype=float)
        mask = ((f >= fit_band[0]) & (f <= fit_band[1]) &
                (f > 0) & (p > 0) & np.isfinite(p))
        if exclude_band is not None:
            mask &= ~((f >= exclude_band[0]) & (f <= exclude_band[1]))
        fx, px = f[mask], p[mask]
        if len(fx) < 3:
            return np.nan, np.nan, 0, np.nan, np.nan

        f_lo_actual, f_hi_actual = float(fx.min()), float(fx.max())
        n_linear = len(fx)

        if log_bin:
            if n_bins is None:
                n_bins = BETA_LOG_BINS if BETA_LOG_BINS is not None else \
                         int(np.clip(n_linear // 2, BETA_LOG_BINS_MIN,
                                     BETA_LOG_BINS_MAX))
            edges = np.logspace(np.log10(fx.min()), np.log10(fx.max()),
                                n_bins + 1)
            bf, bp = [], []
            for i in range(n_bins):
                if i < n_bins - 1:
                    sel = (fx >= edges[i]) & (fx < edges[i + 1])
                else:
                    sel = (fx >= edges[i]) & (fx <= edges[i + 1])
                if sel.sum() == 0:
                    continue
                bf.append(np.exp(np.mean(np.log(fx[sel]))))
                bp.append(np.exp(np.mean(np.log(px[sel]))))
            fx, px = np.asarray(bf), np.asarray(bp)
            if len(fx) < 3:
                return np.nan, np.nan, 0, np.nan, np.nan

        fit = linregress(np.log10(fx), np.log10(px))
        return (float(-fit.slope), float(fit.rvalue ** 2), int(len(fx)),
                f_lo_actual, f_hi_actual)
    except Exception:
        return np.nan, np.nan, 0, np.nan, np.nan


def compute_irasa(x, fs, band, hset=None, nperseg=None):
    """IRASA: separate the aperiodic from the oscillatory part of a spectrum by resampling the signal at
    non-integer factors, which moves narrowband peaks but leaves a power law invariant.

    IRASA needs the analysis band to extend beyond the evaluation band by the largest resampling factor.
    Over 0.01-0.15 Hz on 30-minute records that headroom does not exist, and IRASA remains biased by a
    strong 0.1 Hz peak (recovering β ≈ 0.67
    against a true 1.0 in simulation). It is reported as an additional estimator with that limitation.

    Returns a dict with the aperiodic exponent and the separated spectra.
    """
    out = dict(beta=np.nan, r2=np.nan, peak_freq=np.nan, peak_power=np.nan,
               freqs=None, fractal=None, oscillatory=None)
    try:
        x = np.asarray(x, dtype=float)
        hset    = IRASA_HSET if hset is None else hset
        nperseg = min(len(x), IRASA_NPERSEG if nperseg is None else nperseg)
        if nperseg < 64:
            return out
        nov = int(nperseg * WELCH_OVERLAP_FRAC)
        f0, p0 = welch(x, fs=fs, nperseg=nperseg, noverlap=nov,
                       window=WELCH_WINDOW, detrend=WELCH_DETREND)
        stack = []
        for h in hset:
            num, den = int(round(h * 100)), 100
            up = resample_poly(x, num, den)
            dn = resample_poly(x, den, num)
            fu, pu = welch(up, fs=fs * num / den, nperseg=nperseg,
                           noverlap=nov, window=WELCH_WINDOW,
                           detrend=WELCH_DETREND)
            fd, pd_ = welch(dn, fs=fs * den / num, nperseg=nperseg,
                            noverlap=nov, window=WELCH_WINDOW,
                            detrend=WELCH_DETREND)
            pui = np.interp(f0, fu, pu)
            pdi = np.interp(f0, fd, pd_)
            stack.append(np.sqrt(np.clip(pui, 1e-30, None) *
                                 np.clip(pdi, 1e-30, None)))
        fractal = np.median(np.vstack(stack), axis=0)
        osc     = p0 - fractal
        beta, r2, _, _, _ = _fit_beta(f0, fractal, band, log_bin=True)
        m = (f0 >= band[0]) & (f0 <= band[1])
        out.update(beta=beta, r2=r2,
                   peak_freq=float(f0[m][np.argmax(osc[m])]) if m.any() else np.nan,
                   peak_power=float(np.max(osc[m])) if m.any() else np.nan,
                   freqs=f0, fractal=fractal, oscillatory=osc)
    except Exception as e:
        print(f"    IRASA failed: {e}")
    return out


def fit_aperiodic_with_peak(freqs, power, band, peak_band=None):
    """Aperiodic line plus one Gaussian peak (specparam/FOOOF style):

        log10(P) = c - beta*log10(f) + A*exp(-0.5*((log10 f - mu)/sigma)^2)

    with the peak center constrained to peak_band, modelling the Mayer-wave resonance explicitly. In
    simulation this recovers the exponent to within about 0.1 with a strong 0.1 Hz oscillation present,
    and gives no spurious group difference when groups differ only in peak amplitude.

    Returns dict: beta, r2, peak_freq, peak_amp, peak_width.
    """
    peak_band = SPECPARAM_PEAK_BAND if peak_band is None else peak_band
    out = dict(beta=np.nan, r2=np.nan, peak_freq=np.nan,
               peak_amp=np.nan, peak_width=np.nan, at_bound=False)
    try:
        f = np.asarray(freqs, dtype=float)
        p = np.asarray(power, dtype=float)
        m = ((f >= band[0]) & (f <= band[1]) & (f > 0) &
             (p > 0) & np.isfinite(p))
        if m.sum() < 6:
            return out
        lf, lp = np.log10(f[m]), np.log10(p[m])

        def model(xx, c, beta, amp, mu, sigma):
            return c - beta * xx + amp * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)

        # Five parameters on about 18 points: the peak width is capped at 0.25 (log10 frequency) so the
        # Gaussian cannot absorb the curvature of the whole band; fits that reach a bound are reported.
        p0 = [float(lp.mean()), 1.0, 0.3, float(np.log10(0.1)), 0.12]
        lo = [-np.inf, -2.0, 0.0, float(np.log10(peak_band[0])), 0.03]
        hi = [np.inf,   5.0, 5.0, float(np.log10(peak_band[1])), 0.25]
        popt, _ = curve_fit(model, lf, lp, p0=p0, bounds=(lo, hi), maxfev=20000)
        resid = lp - model(lf, *popt)
        denom = np.sum((lp - lp.mean()) ** 2)
        r2 = 1.0 - np.sum(resid ** 2) / denom if denom > 0 else np.nan
        tol = 1e-3
        at_bound = bool(
            abs(popt[3] - lo[3]) < tol or abs(popt[3] - hi[3]) < tol or
            abs(popt[4] - lo[4]) < tol or abs(popt[4] - hi[4]) < tol)
        out.update(beta=float(popt[1]), r2=float(r2), peak_amp=float(popt[2]),
                   peak_freq=float(10 ** popt[3]), peak_width=float(popt[4]),
                   at_bound=at_bound)
    except Exception:
        pass
    return out


def respiratory_peak_frequency(freqs, power, band=None, search_band=None,
                               aperiodic_band=None, min_prominence=None,
                               n_segments=None, alpha=0.05):
    """Detect a respiratory peak as a local maximum of the residual after removing the aperiodic component.

    The aperiodic line is fitted in log-log space over aperiodic_band with the Mayer band excluded, and
    peaks are sought in the residual. A peak must be a local maximum and exceed a null-calibrated
    threshold; otherwise NaN is returned with a flag.

    Returns a dict of Resp_* fields.
    """
    band           = HF_BAND if band is None else band
    search_band    = RESP_SEARCH_BAND if search_band is None else search_band
    aperiodic_band = ((BETA_FIT_BAND[0], HF_BAND[1])
                      if aperiodic_band is None else aperiodic_band)

    out = dict(Resp_Peak_Freq=np.nan, Resp_Peak_Prominence=np.nan,
               Resp_Peak_Found=False, Resp_Peak_Below_HF=False,
               Resp_Peak_Ambiguous_Mayer=False,
               Resp_N_Candidate_Peaks=0, Resp_Peak_Freq_RawArgmax=np.nan,
               Resp_LF_Peak_Freq=np.nan, Resp_LF_Peak_Prominence=np.nan,
               Resp_Resid_SD=np.nan, Resp_Resid_SD_MAD=np.nan,
               Resp_K_Eff=np.nan, Resp_Peak_Threshold=np.nan,
               Resp_Peak_SNR=np.nan)
    try:
        f = np.asarray(freqs, dtype=float)
        p = np.asarray(power, dtype=float)
        ok = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p > 0)
        f, p = f[ok], p[ok]
        if len(f) < 8:
            return out

        # Raw-argmax value, kept for comparison only.
        m_hf = (f >= band[0]) & (f <= band[1])
        if m_hf.any():
            out["Resp_Peak_Freq_RawArgmax"] = float(f[m_hf][np.argmax(p[m_hf])])

        # 1. Aperiodic fit, Mayer band excluded.
        m_fit = ((f >= aperiodic_band[0]) & (f <= aperiodic_band[1]) &
                 ~((f >= MAYER_BAND[0]) & (f <= MAYER_BAND[1])))
        if m_fit.sum() < 4:
            return out
        lr = linregress(np.log10(f[m_fit]), np.log10(p[m_fit]))
        resid = np.log10(p) - (lr.intercept + lr.slope * np.log10(f))

        # 2. Null-calibrated threshold. A Welch estimate averaging K segments is Gamma(K, 1/K) about the true
        # spectrum, so log10 of it has SD = sqrt(trigamma(K)) / ln(10), with an effective K of
        # WELCH_OVERLAP_EFFICIENCY x K for 50% overlap. The threshold is a Bonferroni bound over the bins
        # searched, so alpha is the false-positive rate for the whole band.
        m_search = (f >= search_band[0]) & (f <= search_band[1])
        n_bins = int(m_search.sum())
        if n_bins < 5:
            return out
        rs = resid[m_search]

        # Diagnostic only; not used for the threshold.
        out["Resp_Resid_SD_MAD"] = float(
            1.4826 * np.median(np.abs(rs - np.median(rs))))

        if n_segments is None or not np.isfinite(n_segments) or n_segments < 2:
            n_segments = WELCH_SEGMENTS_DEFAULT
        k_eff = max(2.0, float(n_segments) * WELCH_OVERLAP_EFFICIENCY)
        resid_sd = float(np.sqrt(polygamma(1, k_eff)) / np.log(10.0))
        out["Resp_Resid_SD"] = resid_sd
        out["Resp_K_Eff"] = k_eff

        z = float(_normdist.ppf(1.0 - alpha / max(n_bins, 1)))
        thresh = resid_sd * z if min_prominence is None else float(min_prominence)
        out["Resp_Peak_Threshold"] = float(thresh)

        # 3. Local maxima of the residual.
        def _peaks(lo, hi):
            m_b = (f >= lo) & (f <= hi)
            ii = np.flatnonzero(m_b)
            ii = ii[(ii > 0) & (ii < len(f) - 1)]
            return [i for i in ii
                    if resid[i] > resid[i - 1] and resid[i] >= resid[i + 1]
                    and resid[i] > thresh]

        # Respiratory candidates: HF band only.
        cand = _peaks(search_band[0], search_band[1])
        out["Resp_N_Candidate_Peaks"] = int(len(cand))
        if cand:
            best = max(cand, key=lambda i: resid[i])
            out["Resp_Peak_Freq"] = float(f[best])
            out["Resp_Peak_Prominence"] = float(resid[best])
            out["Resp_Peak_SNR"] = float(resid[best] / resid_sd)
            out["Resp_Peak_Found"] = True

        # Sub-HF peaks are recorded but never labeled respiratory. They are counted because subjects breathing
        # below 0.15 Hz produce no peak in the HF window, and their breathing lies nearest the slope's band.
        lf = _peaks(RESP_LF_BAND[0], RESP_LF_BAND[1])
        if lf:
            bl = max(lf, key=lambda i: resid[i])
            out["Resp_LF_Peak_Freq"] = float(f[bl])
            out["Resp_LF_Peak_Prominence"] = float(resid[bl])
            out["Resp_Peak_Below_HF"] = True
            out["Resp_Peak_Ambiguous_Mayer"] = bool(
                MAYER_BAND[0] <= f[bl] <= MAYER_BAND[1])
    except Exception:
        pass
    return out


def compute_frequency_domain(rr_intervals, rr_times):
    """Interpolate the R-R series onto a 4 Hz grid (cubic spline), compute the Welch PSD, band powers and
    the spectral slope variants.

    Slope variants
    --------------
    PSD_Beta            linear-bin OLS over BETA_FIT_BAND (uncorrected)
    PSD_Beta_LogBin     log-binned OLS (sensitivity)
    PSD_Beta_NoMayer    MAYER_BAND excised before fitting
    PSD_Beta_Specparam  aperiodic line plus one Gaussian peak
    PSD_Beta_IRASA      IRASA (biased at this bandwidth; see compute_irasa)
    PSD_Beta_Wide/Full  frequency-range sensitivity
    PSD_Beta_Linear     linear interpolation instead of cubic spline
    PSD_Beta_nperseg*   Welch segment-length sensitivity (different Welch parameters by design)

    Welch bins sit at multiples of INTERP_FS / nperseg, so the fitted band differs from the nominal one
    (at nperseg = 512 a nominal 0.01 Hz bound is fitted from 0.0156 Hz). PSD_Fit_LowestFreq_Hz,
    PSD_Fit_HighestFreq_Hz and PSD_Fit_Decades record what was fitted.

    Returns (metrics_dict, freqs_array, power_array).
    """
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / INTERP_FS)

    # -- Interpolation: cubic spline (Task Force, 1996) --
    rr_cubic = interp1d(rr_times, rr_intervals, kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate')(t_uniform)

    # -- Sensitivity: linear interpolation --
    rr_linear = interp1d(rr_times, rr_intervals, kind='linear',
                         bounds_error=False,
                         fill_value='extrapolate')(t_uniform)

    nperseg = int(min(len(rr_cubic), WELCH_NPERSEG))
    nov     = int(nperseg * WELCH_OVERLAP_FRAC)
    freqs, power = welch(rr_cubic, fs=INTERP_FS, nperseg=nperseg,
                         noverlap=nov, window=WELCH_WINDOW,
                         detrend=WELCH_DETREND)
    _, pow_lin   = welch(rr_linear, fs=INTERP_FS, nperseg=nperseg,
                         noverlap=nov, window=WELCH_WINDOW,
                         detrend=WELCH_DETREND)

    freq_res  = INTERP_FS / float(nperseg)
    n_in_band = int(((freqs >= BETA_FIT_BAND[0]) &
                     (freqs <= BETA_FIT_BAND[1]) & (freqs > 0)).sum())
    n_segments = max(1, int((len(rr_cubic) - nov) // (nperseg - nov)))

    def band_power(band, pw=None):
        pw  = power if pw is None else pw
        idx = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapezoid(pw[idx], freqs[idx]) if idx.sum() > 1 else np.nan

    lf_p  = band_power(LF_BAND)
    hf_p  = band_power(HF_BAND)
    # VLF power, integrated directly. The slope's band straddles the VLF/LF boundary at 0.04 Hz, so a
    # steeper slope is the same statement as more VLF relative to LF power. The lowest resolved bin is
    # 0.0078 Hz, so VLF is integrated from there and the actual lower edge is recorded.
    vlf_lo_actual = float(freqs[(freqs >= VLF_BAND[0]) & (freqs > 0)].min()) \
        if ((freqs >= VLF_BAND[0]) & (freqs > 0)).any() else np.nan
    vlf_p = band_power(VLF_BAND)
    vlf_lf = vlf_p / lf_p if (lf_p and lf_p > 0) else np.nan
    vlf_n_bins = int(((freqs >= VLF_BAND[0]) & (freqs <= VLF_BAND[1]) &
                      (freqs > 0)).sum())
    tot_p = band_power((VLF_BAND[0], HF_BAND[1]))
    # Total power over the reliably estimable range.
    tot_p_reliable = band_power((LF_BAND[0], HF_BAND[1]))
    lf_hf = lf_p / hf_p if (hf_p and hf_p > 0) else np.nan

    # -- Slope variants over the main band --
    b_primary, r2_primary, n_primary, f_lo_act, f_hi_act = _fit_beta(
        freqs, power, BETA_FIT_BAND)
    b_logbin, r2_logbin, n_logbin, _, _ = _fit_beta(
        freqs, power, BETA_FIT_BAND, log_bin=True)
    # Mayer-wave control: the same fit with 0.07-0.13 Hz excised, using the linear bins (log-binning the
    # remaining bins would leave about five points).
    b_notch, r2_notch, n_notch, _, _ = _fit_beta(
        freqs, power, BETA_FIT_BAND, exclude_band=MAYER_BAND, log_bin=False)
    # Log-binned variant, a secondary check.
    b_notch_lb, _, n_notch_lb, _, _ = _fit_beta(
        freqs, power, BETA_FIT_BAND, exclude_band=MAYER_BAND, log_bin=True)
    b_wide, r2_wide, _, f_lo_wide, _ = _fit_beta(
        freqs, power, BETA_FIT_BAND_WIDE)
    b_full, r2_full, _, f_lo_full, _ = _fit_beta(
        freqs, power, BETA_FIT_BAND_FULL)
    b_linear, r2_linear, _, _, _ = _fit_beta(freqs, pow_lin, BETA_FIT_BAND)

    # -- Per-segment detrending sensitivity --
    # Same series, segmentation and band; only the within-segment detrending changes.
    try:
        f_dt, p_dt = welch(rr_cubic, fs=INTERP_FS, nperseg=nperseg,
                           noverlap=nov, window=WELCH_WINDOW,
                           detrend=WELCH_DETREND_SENS)
        b_lindet, r2_lindet, _, _, _ = _fit_beta(f_dt, p_dt, BETA_FIT_BAND)
        b_lindet_nm, _, _, _, _ = _fit_beta(f_dt, p_dt, BETA_FIT_BAND,
                                            exclude_band=MAYER_BAND,
                                            log_bin=False)
    except Exception:
        b_lindet, r2_lindet, b_lindet_nm = np.nan, np.nan, np.nan

    # -- Explicit peak model --
    sp = fit_aperiodic_with_peak(freqs, power, BETA_FIT_BAND)

    # -- IRASA (see compute_irasa for its bandwidth limitation) --
    ir = compute_irasa(rr_cubic, INTERP_FS, BETA_FIT_BAND)

    # -- Band sweep --
    band_sweep = {}
    for tag, lo, hi, excl in BETA_BAND_SWEEP:
        bb, rr, nn, flo, fhi = _fit_beta(freqs, power, (lo, hi), exclude_band=excl)
        band_sweep[f"PSD_Beta_{tag}"]        = bb
        band_sweep[f"PSD_Beta_{tag}_R2"]     = rr
        band_sweep[f"PSD_Beta_{tag}_NBins"]  = nn
        band_sweep[f"PSD_Beta_{tag}_LowHz"]  = flo
        band_sweep[f"PSD_Beta_{tag}_HighHz"] = fhi

    # -- Segment-length sensitivity --
    seg_sens = {}
    for ns in WELCH_NPERSEG_SENS:
        ns_use = int(min(len(rr_cubic), ns))
        try:
            f_s, p_s = welch(rr_cubic, fs=INTERP_FS, nperseg=ns_use,
                             noverlap=int(ns_use * WELCH_OVERLAP_FRAC),
                             window=WELCH_WINDOW, detrend=WELCH_DETREND)
            b_s, _, nb_s, f_lo_s, _ = _fit_beta(f_s, p_s, BETA_FIT_BAND)
            seg_sens[f"PSD_Beta_nperseg{ns}"]       = b_s
            seg_sens[f"PSD_Beta_nperseg{ns}_nbins"] = nb_s
            seg_sens[f"PSD_FreqRes_nperseg{ns}"]    = INTERP_FS / float(ns_use)
            seg_sens[f"PSD_LowestFit_nperseg{ns}"]  = f_lo_s
        except Exception:
            seg_sens[f"PSD_Beta_nperseg{ns}"]       = np.nan
            seg_sens[f"PSD_Beta_nperseg{ns}_nbins"] = 0
            seg_sens[f"PSD_FreqRes_nperseg{ns}"]    = np.nan
            seg_sens[f"PSD_LowestFit_nperseg{ns}"]  = np.nan

    out = {
        "LF_Power_ms2":            lf_p  * 1e6,
        "HF_Power_ms2":            hf_p  * 1e6,
        # VLF decomposition
        "VLF_Power_ms2":           vlf_p * 1e6,
        "VLF_LF_Ratio":            vlf_lf,
        "VLF_NBins":               vlf_n_bins,
        "VLF_LowestFreq_Hz":       vlf_lo_actual,
        "Total_Power_ms2":         tot_p * 1e6,
        "Total_Power_LFHF_ms2":    tot_p_reliable * 1e6,
        "LF_HF_Ratio":             lf_hf,
        # Uncorrected slope (0.01-0.15 Hz nominal, cubic spline, linear bins)
        "PSD_Beta":                b_primary,
        "PSD_Beta_R2":             r2_primary,
        "PSD_Beta_NBins":          n_primary,
        # Log-binned slope
        "PSD_Beta_LogBin":         b_logbin,
        "PSD_Beta_LogBin_R2":      r2_logbin,
        "PSD_Beta_LogBin_NBins":   n_logbin,
        # Mayer-band excised slope
        "PSD_Beta_NoMayer":        b_notch,
        "PSD_Beta_NoMayer_R2":     r2_notch,
        "PSD_Beta_NoMayer_NBins":  n_notch,
        "PSD_Beta_NoMayer_LogBin": b_notch_lb,
        "PSD_Beta_NoMayer_LogBin_NBins": n_notch_lb,
        # Explicit peak model
        "PSD_Beta_Specparam":      sp["beta"],
        "PSD_Beta_Specparam_R2":   sp["r2"],
        "Mayer_Peak_Freq":         sp["peak_freq"],
        "Mayer_Peak_Amp":          sp["peak_amp"],
        "Mayer_Peak_Width":        sp["peak_width"],
        "Specparam_At_Bound":      sp.get("at_bound", False),
        # IRASA
        "PSD_Beta_IRASA":          ir["beta"],
        "PSD_Beta_IRASA_R2":       ir["r2"],
        "IRASA_Peak_Freq":         ir["peak_freq"],
        # Frequency-range sensitivity
        "PSD_Beta_Wide":           b_wide,
        "PSD_Beta_Wide_R2":        r2_wide,
        "PSD_Beta_Full":           b_full,
        "PSD_Beta_Full_R2":        r2_full,
        # Per-segment detrending sensitivity
        "PSD_Beta_LinDetrend":        b_lindet,
        "PSD_Beta_LinDetrend_R2":     r2_lindet,
        "PSD_Beta_NoMayer_LinDetrend": b_lindet_nm,
        # Interpolation sensitivity
        "PSD_Beta_Linear":         b_linear,
        "PSD_Beta_Linear_R2":      r2_linear,
        # Respiratory peak (see respiratory_peak_frequency)
        **respiratory_peak_frequency(freqs, power,
                                     n_segments=n_segments),
        # Welch specification, recorded per subject
        "Welch_NPerSeg":           nperseg,
        "Welch_NOverlap":          nov,
        "Welch_Window":            WELCH_WINDOW,
        "Welch_Detrend":           WELCH_DETREND,
        "Welch_FreqRes_Hz":        freq_res,
        "Welch_N_Segments":        n_segments,
        "PSD_NBins_In_FitBand":    n_in_band,
        # Fitted frequency range (not the nominal band; see the docstring).
        "PSD_Fit_LowestFreq_Hz":   f_lo_act,
        "PSD_Fit_HighestFreq_Hz":  f_hi_act,
        "PSD_Fit_Decades":         (float(np.log10(f_hi_act / f_lo_act))
                                    if (np.isfinite(f_lo_act) and
                                        np.isfinite(f_hi_act) and f_lo_act > 0)
                                    else np.nan),
        "PSD_Fit_LowBound_Nominal": BETA_FIT_BAND[0],
        "PSD_Fit_LowBound_Mismatch": (bool(f_lo_act > BETA_FIT_BAND[0] * 1.2)
                                      if np.isfinite(f_lo_act) else True),
        "PSD_Wide_LowestFreq_Hz":  f_lo_wide,
        "PSD_Full_LowestFreq_Hz":  f_lo_full,
    }
    out.update(seg_sens)
    out.update(band_sweep)
    return out, freqs, power


# ---- Step 3: complexity measures (and MSE curve data) ----

def compute_complexity_metrics(rr_intervals, quant_step=None):
    """Compute the complexity battery: sample, approximate and permutation entropy, refined composite MSE,
    DFA, RQA, MFDFA, recurrence networks, embedding and stationarity diagnostics, and resolution controls.

    MSE scales are retained while the coarse-grained series holds at least MSE_MIN_POINTS points, up to
    MSE_MAX_SCALE.

    Returns
    -------
    metrics    : dict        scalar values
    mse_scales : np.ndarray  scale indices for curve plotting (or None)
    mse_values : np.ndarray  entropy at each scale (or None)
    """
    results    = {}
    mse_scales = None
    mse_values = None

    rr = np.asarray(rr_intervals, dtype=float)
    sd = float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan
    r_tol = ENTROPY_R_FRAC * sd            # explicit numeric tolerance
    results["Entropy_R_Frac"] = ENTROPY_R_FRAC
    results["Entropy_R_Abs"]  = r_tol
    results["Series_SD"]      = sd
    results["Series_N"]       = int(len(rr))

    # -- Sample entropy (this module's implementation, explicit tolerance) --
    # NeuroKit2 values are kept as cross-check columns.
    results["SampEn"] = sample_entropy(rr, m=ENTROPY_M, r=r_tol)
    results["ApEn"]   = approximate_entropy(rr, m=ENTROPY_M, r=r_tol)

    try:
        se_nk, _ = nk.entropy_sample(rr, dimension=ENTROPY_M, tolerance=r_tol)
        results["SampEn_NK"] = float(se_nk)
    except Exception:
        results["SampEn_NK"] = np.nan
    try:
        ae_nk, _ = nk.entropy_approximate(rr, dimension=ENTROPY_M, tolerance=r_tol)
        results["ApEn_NK"] = float(ae_nk)
    except Exception:
        results["ApEn_NK"] = np.nan

    # -- Entropy tolerance sensitivity --
    for frac in ENTROPY_R_SENSITIVITY:
        key = f"SampEn_r{round(frac * 100):02d}"
        results[key] = sample_entropy(rr, m=ENTROPY_M, r=frac * sd)

    # Permutation Entropy
    try:
        pe, _ = nk.entropy_permutation(rr, dimension=3, delay=1)
        results["PermEn"] = float(pe)
        # Embedding-dimension sensitivity: at dimension 3 there are only six ordinal patterns, so the
        # estimator is coarse and exposed to tie-breaking.
        for _md in PERMEN_DIM_SWEEP:
            try:
                _pe, _ = nk.entropy_permutation(rr, dimension=_md, delay=1)
                results[f"PermEn_m{_md}"] = float(_pe)
            except Exception:
                results[f"PermEn_m{_md}"] = np.nan
    except Exception as e:
        print(f"    PermEn failed: {e}")
        results["PermEn"] = np.nan
    # Ordinal tie rate, the route by which quantization affects permutation entropy.
    results["Ordinal_Tie_Rate"] = ordinal_tie_rate(rr)

    # -- Multiscale entropy (refined composite) --
    # At least MSE_MIN_POINTS points at every retained scale.
    try:
        mse_scales, mse_values = rcmse(rr, m=ENTROPY_M, r_frac=ENTROPY_R_FRAC,
                                       max_scale=MSE_MAX_SCALE,
                                       min_points=MSE_MIN_POINTS)
        finite = np.isfinite(mse_values)
        mse_scales, mse_values = mse_scales[finite], mse_values[finite]
        if len(mse_values):
            results["MSE_Mean"]      = float(np.mean(mse_values))
            results["MSE_MaxScale"]  = int(mse_scales[-1])
            results["MSE_N_Scales"]  = int(len(mse_values))
            results["MSE_Slope"]     = (float(np.polyfit(mse_scales, mse_values, 1)[0])
                                        if len(mse_values) >= 3 else np.nan)
            print(f"    RCMSE: {len(mse_values)} scales (max τ={int(mse_scales[-1])}), "
                  f"mean = {results['MSE_Mean']:.4f}")
        else:
            results.update(MSE_Mean=np.nan, MSE_MaxScale=0,
                           MSE_N_Scales=0, MSE_Slope=np.nan)
    except Exception as e:
        print(f"    RCMSE failed: {e}")
        results.update(MSE_Mean=np.nan, MSE_MaxScale=0,
                       MSE_N_Scales=0, MSE_Slope=np.nan)

    # -- MSE scale bookkeeping and curve export --
    # MSE_Mean is the mean entropy over retained scales (not Costa's complexity index, which is the sum).
    try:
        _sc  = mse_scales
        _vals = mse_values
        if _sc is not None and _vals is not None and len(_sc):
            _sc = np.asarray(_sc, dtype=float)
            _vals = np.asarray(_vals, dtype=float)
            results["MSE_Scales_Retained"] = int(np.isfinite(_vals).sum())
            results["MSE_MaxScale_Retained"] = float(np.nanmax(_sc))
            # Mean over a fixed scale set common to every subject.
            _common = np.isin(_sc, np.asarray(MSE_COMMON_SCALES, dtype=float))
            _cv = _vals[_common]
            results["MSE_Mean_CommonScales"] = (float(np.nanmean(_cv))
                                                if np.isfinite(_cv).any()
                                                else np.nan)
            results["MSE_CommonScales_Complete"] = bool(
                np.isfinite(_cv).sum() == len(MSE_COMMON_SCALES))
            # Full curve, one column per scale.
            for _t, _v in zip(_sc, _vals):
                results[f"MSE_Curve_Tau{int(_t):02d}"] = float(_v)
    except Exception as e:
        # Raise rather than leave empty MSE columns.
        print(f"    !! MSE scale bookkeeping FAILED ({type(e).__name__}: {e}) "
              f"— MSE_Mean_CommonScales and MSE_Curve_* NOT written")


    # DFA, full range
    try:
        v, _ = fractal_dfa(rr_intervals, show=False)
        results["DFA_Alpha"] = float(v)
    except Exception as e:
        print(f"    DFA full failed: {e}")
        results["DFA_Alpha"] = np.nan

    # DFA alpha1, 4-16 beats
    try:
        v, _ = fractal_dfa(rr_intervals, scale=np.arange(4, 17), show=False)
        results["DFA_Alpha1"] = float(v)
    except Exception as e:
        print(f"    DFA alpha1 failed: {e}")
        results["DFA_Alpha1"] = np.nan

    # DFA alpha1 lower-bound sensitivity.
    for _nmin in DFA_ALPHA1_NMIN_SWEEP:
        _sc = np.arange(_nmin, 17)
        if len(_sc) < MIN_FIT_SCALES:
            # Raise on a degenerate range rather than emit NaN.
            print(f"    !! DFA alpha1 nmin={_nmin} skipped: only {len(_sc)} "
                  f"scale(s) in range, need {MIN_FIT_SCALES}")
            results[f"DFA_Alpha1_nmin{_nmin}"] = np.nan
            continue
        try:
            _v, _ = fractal_dfa(rr_intervals, scale=_sc, show=False)
            results[f"DFA_Alpha1_nmin{_nmin}"] = float(_v)
        except Exception:
            results[f"DFA_Alpha1_nmin{_nmin}"] = np.nan

    # Scale 16 is in both fits (alpha1 4..16, alpha2 16..64); the disjoint variant is computed so the
    # overlap can be checked.
    try:
        _v, _ = fractal_dfa(rr_intervals, scale=np.arange(4, 16), show=False)
        results["DFA_Alpha1_Disjoint"] = float(_v)
    except Exception:
        results["DFA_Alpha1_Disjoint"] = np.nan

    # DFA alpha2, 16-64 beats
    try:
        v, _ = fractal_dfa(rr_intervals, scale=np.arange(16, 65), show=False)
        results["DFA_Alpha2"] = float(v)
    except Exception as e:
        print(f"    DFA alpha2 failed: {e}")
        results["DFA_Alpha2"] = np.nan

    # -- RQA at a fixed recurrence rate --
    # The threshold fixes the recurrence rate by construction (see recurrence_matrix), making determinism
    # amplitude-invariant. The primary matrix is built once and shared with rqa_metrics() and
    # compute_recurrence_network().
    try:
        # The Theiler window is resolved once here and passed to every consumer, so all use one pair set.
        _theiler_w, _theiler_src = resolve_theiler(rr, RQA_DIM, RQA_DELAY)
        results["RQA_Theiler_Source"] = _theiler_src
        _R_primary, _eps_primary = recurrence_matrix(
            rr, dim=RQA_DIM, delay=RQA_DELAY,
            rec_rate=RQA_REC_RATE, max_pts=None, theiler=_theiler_w)
    except Exception as e:
        print(f"    Recurrence matrix failed: {e}")
        _R_primary, _eps_primary = None, np.nan
        # _theiler_w must also exist on this path; the calls below use it.
        _theiler_w = int((RQA_DIM - 1) * RQA_DELAY + 1)
        results["RQA_Theiler_Source"] = "embedding_window_after_failure"

    try:
        rqa = rqa_metrics(rr, theiler=_theiler_w, dim=RQA_DIM, delay=RQA_DELAY,
                          rec_rate=RQA_REC_RATE, l_min=RQA_L_MIN,
                          v_min=RQA_V_MIN, max_pts=None,
                          R=_R_primary, eps=_eps_primary)
        results.update(rqa)
    except Exception as e:
        print(f"    RQA failed: {e}")
        for k in ["RQA_REC", "RQA_DET", "RQA_LAM", "RQA_L_Mean", "RQA_Eps"]:
            results[k] = np.nan

    # -- Embedding-parameter diagnostics --
    # dim = 2, delay = 1 is the primary setting; data-driven choices are computed and reported, with a
    # sensitivity rerun at the selected parameters.
    try:
        sel_delay = select_delay_mutual_information(rr)
        sel_dim   = select_dimension_fnn(rr, delay=sel_delay)
        results["Embed_Delay_MI"]  = sel_delay
        results["Embed_Dim_FNN"]   = sel_dim
        _theiler_sel, _ = resolve_theiler(rr, sel_dim, sel_delay)
        results["RQA_Theiler_W_SelEmbed"] = int(_theiler_sel)
        rqa_sel = rqa_metrics(rr, theiler=_theiler_sel,
                              dim=sel_dim, delay=sel_delay,
                              rec_rate=RQA_REC_RATE, l_min=RQA_L_MIN,
                              v_min=RQA_V_MIN, max_pts=None)
        results["RQA_DET_SelEmbed"] = rqa_sel.get("RQA_DET", np.nan)
        results["RQA_LAM_SelEmbed"] = rqa_sel.get("RQA_LAM", np.nan)
    except Exception as e:
        print(f"    Embedding selection failed: {e}")
        results.update(Embed_Delay_MI=np.nan, Embed_Dim_FNN=np.nan,
                       RQA_DET_SelEmbed=np.nan, RQA_LAM_SelEmbed=np.nan)


    # -- MFDFA --
    mfdfa = compute_mfdfa(rr)
    results.update(mfdfa)

    # -- Recurrence network analysis --
    # Shares the primary matrix built above.
    rna = compute_recurrence_network(rr, R=_R_primary, theiler=_theiler_w)
    results.update(rna)

    # -- Stationarity --
    results.update(stationarity_diagnostics(rr))

    # -- Resolution (quantization) controls --
    # The step is device-specific and passed per cohort: 1.000 ms for the Polar H10 export, 1/128 s for
    # MIT-BIH.
    step = QUANT_STEP_S if quant_step is None else quant_step
    results["Quant_Step_Source"] = ("polar_h10" if step == QUANT_STEP_S
                                    else "cohort_specific")
    results.update(quantization_controls(rr, step=step, theiler=_theiler_w))

    # -- Cross-device re-quantization (see REQUANT_TARGET_STEP_S) --
    # Applied only when the target grid is coarser than the native one, so MIT-BIH records are skipped.
    if REQUANT_ENABLED and REQUANT_TARGET_STEP_S > step * 1.5:
        try:
            xq = coarsen(rr, REQUANT_TARGET_STEP_S)
            _rq_rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
            results["Requant_Target_Step_s"] = REQUANT_TARGET_STEP_S
            results["Requant_Applied"] = True
            # RMSSD in steps of the coarse grid.
            results["Requant_RMSSD_In_Steps"] = (
                _rq_rmssd / REQUANT_TARGET_STEP_S
                if REQUANT_TARGET_STEP_S > 0 else np.nan)
            results["Requant_Tie_Rate"] = ordinal_tie_rate(xq)

            results["SampEn_Q128"] = sample_entropy(xq)
            results["ApEn_Q128"]   = approximate_entropy(xq)
            try:
                _pe, _ = nk.entropy_permutation(xq, dimension=3, delay=1)
                results["PermEn_Q128"] = float(_pe)
            except Exception:
                results["PermEn_Q128"] = np.nan
            results["MSE_Mean_Q128"] = _mse_mean_safe(xq)
            # Same fixed Theiler window as every other condition.
            _rqq = rqa_metrics(xq, theiler=_theiler_w)
            results["RQA_DET_Q128"] = _rqq.get("RQA_DET", np.nan)
            results["RQA_LAM_Q128"] = _rqq.get("RQA_LAM", np.nan)
            try:
                _rnaq = compute_recurrence_network(xq, theiler=_theiler_w)
                results["RNA_ClusteringCoeff_Q128"] = _rnaq.get(
                    "RNA_ClusteringCoeff", np.nan)
                results["RNA_Transitivity_Q128"] = _rnaq.get(
                    "RNA_Transitivity", np.nan)
            except Exception:
                results["RNA_ClusteringCoeff_Q128"] = np.nan
                results["RNA_Transitivity_Q128"] = np.nan
            try:
                _v, _ = fractal_dfa(xq, scale=np.arange(4, 17), show=False)
                results["DFA_Alpha1_Q128"] = float(_v)
            except Exception:
                results["DFA_Alpha1_Q128"] = np.nan
            # DFA alpha2 on the same grid (amplitude and spectral measures: requant_amplitude_spectral).
            try:
                _v, _ = fractal_dfa(xq, scale=np.arange(16, 65), show=False)
                results["DFA_Alpha2_Q128"] = float(_v)
            except Exception:
                results["DFA_Alpha2_Q128"] = np.nan
            results["Requant_Model"] = "interval_rounding"
        except Exception as e:
            print(f"    !! Re-quantization control FAILED "
                  f"({type(e).__name__}: {e}) — *_Q128 columns NOT written")
            results["Requant_Applied"] = False
    else:
        results["Requant_Applied"] = False

    # -- Time-matched DFA (see DFA_TIME_MATCHED_S) --
    # The realized beat scales are recorded per subject.
    try:
        _mean_nn = float(np.mean(rr))
        for _lo_s, _hi_s in DFA_TIME_MATCHED_S:
            _lo_b = max(4, int(round(_lo_s / _mean_nn)))
            _hi_b = max(_lo_b + 3, int(round(_hi_s / _mean_nn)))
            _tag = f"{int(_lo_s)}s_{int(_hi_s)}s"
            results[f"DFA_TimeMatched_{_tag}_LoBeats"] = _lo_b
            results[f"DFA_TimeMatched_{_tag}_HiBeats"] = _hi_b
            try:
                _v, _ = fractal_dfa(rr, scale=np.arange(_lo_b, _hi_b + 1),
                                    show=False)
                results[f"DFA_TimeMatched_{_tag}"] = float(_v)
            except Exception:
                results[f"DFA_TimeMatched_{_tag}"] = np.nan
    except Exception as e:
        print(f"    !! Time-matched DFA FAILED ({type(e).__name__}: {e})")

    return results, mse_scales, mse_values


# ---- Step 3b: multifractal DFA ----

def compute_mfdfa(rr_s, q_orders=None, n_scales=20, poly_order=1,
                  with_surrogates=True):
    """Multifractal detrended fluctuation analysis (Kantelhardt et al., 2002).

    Positive q only by default (MFDFA_Q_PRIMARY), since negative-order moments are unreliable on records
    of about 2500 points; the full range is also returned under *_FullQ names.

    with_surrogates=True compares the width against shuffled surrogates (keep the distribution, destroy
    correlations) and phase-randomized surrogates (keep correlations, Gaussianize the distribution),
    adding MFDFA_Width_Z_Shuffle and MFDFA_Width_Z_PhaseRand.

    Returns a dict of scalar fields.
    """
    _nan = dict(MFDFA_Width=np.nan, MFDFA_H_Mean=np.nan, MFDFA_H2=np.nan,
                MFDFA_Width_FullQ=np.nan, MFDFA_H_Mean_FullQ=np.nan,
                MFDFA_Asymmetry_FullQ=np.nan,
                MFDFA_Width_Z_Shuffle=np.nan,
                MFDFA_Width_Z_PhaseRand=np.nan)

    def _hq(x, qs):
        """Generalised Hurst exponents H(q) for one series."""
        y = np.cumsum(x - np.mean(x))
        N = len(y)
        s_min = max(8, int(N ** 0.25))
        s_max = N // 4
        if s_max < s_min * 2:
            return None, None
        scales = np.unique(np.round(
            np.logspace(np.log10(s_min), np.log10(s_max), n_scales)
        ).astype(int))
        scales = scales[(scales >= s_min) & (scales <= s_max)]
        if len(scales) < 4:
            return None, None
        # The fitted scale range is recorded: s_min = max(8, N**0.25), s_max = N // 4 (about 8 to 625 beats at
        # 2500 beats), which overlaps the DFA alpha2 range (16 to 64) only in part.
        _MFDFA_RANGE_INFO.clear()
        _MFDFA_RANGE_INFO.update(
            MFDFA_Scale_Min=int(np.min(scales)),
            MFDFA_Scale_Max=int(np.max(scales)),
            MFDFA_N_Scales_Used=int(len(scales)))

        Fq  = np.full((len(qs), len(scales)), np.nan)
        t_s = np.arange(max(scales), dtype=float)
        for si, s in enumerate(scales):
            n_seg = N // s
            if n_seg < 2:
                continue
            ts = t_s[:s]
            F2 = []
            for seg in range(n_seg):
                seg_y = y[seg * s:(seg + 1) * s]
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
            for qi, q in enumerate(qs):
                try:
                    val = float(np.mean(F2_arr ** (q / 2.0))) ** (1.0 / q)
                    if np.isfinite(val) and val > 0.0:
                        Fq[qi, si] = val
                except Exception:
                    pass

        log_s = np.log(scales.astype(float))
        h_q   = np.full(len(qs), np.nan)
        for qi in range(len(qs)):
            row   = Fq[qi]
            valid = np.isfinite(row) & (row > 0.0)
            if valid.sum() < 4:
                continue
            slope, *_ = linregress(log_s[valid], np.log(row[valid]))
            if np.isfinite(slope):
                h_q[qi] = slope
        return h_q, scales

    def _summaries(h_q, qs):
        fin = h_q[np.isfinite(h_q)]
        if len(fin) < 3:
            return None
        width  = float(np.max(fin) - np.min(fin))
        h_mean = float(np.mean(fin))
        neg = h_q[np.asarray(qs) < 0]; neg = neg[np.isfinite(neg)]
        pos = h_q[np.asarray(qs) > 0]; pos = pos[np.isfinite(pos)]
        asym = (float(np.mean(neg) - np.mean(pos))
                if (len(neg) and len(pos)) else np.nan)
        idx = np.where(np.asarray(qs, dtype=float) == 2.0)[0]
        h2 = (float(h_q[idx[0]]) if idx.size and np.isfinite(h_q[idx[0]])
              else np.nan)
        return width, h_mean, asym, h2

    try:
        x = np.asarray(rr_s, dtype=float)
        if not np.all(np.isfinite(x)) or len(x) < 64:
            return _nan

        q_primary = np.asarray(MFDFA_Q_PRIMARY if q_orders is None else q_orders,
                               dtype=float)
        q_full    = np.asarray(MFDFA_Q_FULL, dtype=float)

        h_p, _ = _hq(x, q_primary)
        h_f, _ = _hq(x, q_full)
        if h_p is None:
            return _nan
        s_p = _summaries(h_p, q_primary)
        if s_p is None:
            return _nan
        width, h_mean, asym, h2 = s_p

        # Asymmetry (mean H over negative minus positive moments) is undefined under positive q only; the
        # full-q value is reported separately.
        out = dict(MFDFA_Width=width, MFDFA_H_Mean=h_mean, MFDFA_H2=h2)
        out.update(_MFDFA_RANGE_INFO)
        if np.isfinite(asym):
            out["MFDFA_Asymmetry"] = asym
        else:
            out["MFDFA_Asymmetry_Note"] = "undefined_positive_q_only"

        if h_f is not None:
            s_f = _summaries(h_f, q_full)
            if s_f is not None:
                out.update(MFDFA_Width_FullQ=s_f[0],
                           MFDFA_H_Mean_FullQ=s_f[1],
                           MFDFA_Asymmetry_FullQ=s_f[2])
            else:
                out.update(MFDFA_Width_FullQ=np.nan,
                           MFDFA_H_Mean_FullQ=np.nan,
                           MFDFA_Asymmetry_FullQ=np.nan)
        else:
            out.update(MFDFA_Width_FullQ=np.nan,
                       MFDFA_H_Mean_FullQ=np.nan,
                       MFDFA_Asymmetry_FullQ=np.nan)

        out.update(MFDFA_Width_Z_Shuffle=np.nan,
                   MFDFA_Width_Z_PhaseRand=np.nan)
        if with_surrogates and np.isfinite(width):
            rng = np.random.default_rng(QUANT_SEED)
            for tag, maker in (("Shuffle", lambda: rng.permutation(x)),
                               ("PhaseRand", lambda: _phase_randomise(x, rng))):
                vals = []
                for _ in range(MFDFA_N_SURROGATES):
                    try:
                        hs, _ = _hq(maker(), q_primary)
                        ss = _summaries(hs, q_primary) if hs is not None else None
                        if ss is not None and np.isfinite(ss[0]):
                            vals.append(ss[0])
                    except Exception:
                        pass
                if len(vals) >= 5:
                    v = np.asarray(vals, float)
                    sdv = float(np.std(v, ddof=1))
                    out[f"MFDFA_Width_Z_{tag}"] = (
                        float((width - np.mean(v)) / sdv) if sdv > 0 else np.nan)
                    # The surrogate distribution itself, so exceedance can be checked.
                    out[f"MFDFA_Surr_Width_Mean_{tag}"] = float(np.mean(v))
                    out[f"MFDFA_Surr_Width_SD_{tag}"]   = sdv
                    out[f"MFDFA_Surr_N_{tag}"]          = int(len(v))
                    # Count-based criterion (no symmetry assumption), as a fraction of surrogates that succeeded.
                    out[f"MFDFA_Surr_Frac_Exceeded_{tag}"] = float(
                        np.mean(width > v))
                    # Direction of the departure.
                    out[f"MFDFA_Surr_Direction_{tag}"] = (
                        "above" if width > np.mean(v) else "below")

        print(f"    MFDFA[q>0]: Δh={width:.4f}  H̄={h_mean:.4f}  H(2)={h2:.4f}  "
              f"| Z_shuf={out['MFDFA_Width_Z_Shuffle']:.2f} "
              f"Z_phase={out['MFDFA_Width_Z_PhaseRand']:.2f}")
        return out

    except Exception as e:
        print(f"    MFDFA failed: {e}")
        return _nan


def _phase_randomise(x, rng):
    """Phase-randomised surrogate: preserves the linear autocorrelation."""
    x = np.asarray(x, float)
    n = len(x)
    amp = np.abs(np.fft.rfft(x))
    phi = rng.uniform(0, 2 * np.pi, len(amp))
    phi[0] = 0.0
    if n % 2 == 0:
        phi[-1] = 0.0
    return np.fft.irfft(amp * np.exp(1j * phi), n=n)


# ---- Step 3c: recurrence network analysis ----

def compute_recurrence_network(rr_s, dim=None, delay=None,
                               rec_rate=None, max_pts=None, R=None,
                               theiler=None):
    """Recurrence network analysis on the full embedded series (no decimation).

    RNA_MeanDegree is a normalization check: it equals the recurrence rate for every subject by
    construction. max_pts=None uses the whole series; if a cap is set, a contiguous leading window is used.
    """
    dim      = RQA_DIM       if dim      is None else dim
    delay    = RQA_DELAY     if delay    is None else delay
    rec_rate = RQA_REC_RATE  if rec_rate is None else rec_rate
    max_pts  = RNA_MAX_PTS   if max_pts  is None else max_pts

    _nan = dict(RNA_MeanDegree=np.nan, RNA_ClusteringCoeff=np.nan,
                RNA_Transitivity=np.nan, RNA_DegreeCheck_OK=False,
                RNA_N_Points=0)

    try:
        x = np.asarray(rr_s, dtype=float)
        if not np.all(np.isfinite(x)) or len(x) < 30:
            return _nan

        if max_pts is not None and len(x) > max_pts:
            x = x[:max_pts]                     # contiguous, NOT decimated

        # A caller may supply the primary matrix (same parameters); only valid when max_pts is None.
        if R is None or max_pts is not None:
            R, _eps = recurrence_matrix(x, dim=dim, delay=delay,
                                        rec_rate=rec_rate, max_pts=None,
                                        theiler=theiler)
        if R is None:
            return _nan

        A = R.astype(np.float32)
        n = A.shape[0]
        k = A.sum(axis=1)
        mean_degree = float(k.mean() / (n - 1)) if n > 1 else np.nan

        A_sq     = A @ A
        tri_diag = np.einsum('ij,ji->i', A_sq, A)
        triangles = tri_diag / 2.0
        triplets  = k * (k - 1) / 2.0
        with np.errstate(invalid='ignore', divide='ignore'):
            cc_local = np.where(triplets > 0.0, triangles / triplets, 0.0)
        clustering = float(cc_local.mean())
        # float64 accumulation (the sum reaches about 1e8 at n = 2500).
        total_tri  = float(tri_diag.sum(dtype=np.float64)) / 6.0
        total_trip = float(triplets.sum(dtype=np.float64))
        transitivity = (3.0 * total_tri / total_trip
                        if total_trip > 0.0 else np.nan)

        # Normalization check: mean degree must equal the target recurrence rate.
        check_ok = bool(np.isfinite(mean_degree) and
                        abs(mean_degree - rec_rate) < 0.005)

        out = dict(RNA_MeanDegree=mean_degree,
                   RNA_ClusteringCoeff=clustering,
                   RNA_Transitivity=float(transitivity),
                   RNA_DegreeCheck_OK=check_ok,
                   RNA_N_Points=int(n))
        flag = "" if check_ok else "  ** DEGREE NORMALISATION OFF TARGET **"
        print(f"    RNA (n={n}): deg={mean_degree:.5f} (target {rec_rate:.3f})  "
              f"CC={clustering:.4f}  trans={transitivity:.4f}{flag}")
        return out

    except Exception as e:
        print(f"    RNA failed: {e}")
        return _nan


# ---- Step 4: IAAFT surrogate testing ----

def _iaaft_one(x, n_iters=N_IAAFT_ITERS, rng=None):
    """Generate one IAAFT surrogate (Schreiber and Schmitz, 1996).

    Preserves the amplitude distribution and the power spectrum while destroying nonlinear structure.
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
    """SampEn, DFA_Alpha1 and MSE_Mean on one surrogate, with the same tolerance and MSE settings as the
    observed values.
    """
    out = {}
    s = np.asarray(s, dtype=float)
    out["SampEn"] = sample_entropy(s)

    try:
        v, _ = fractal_dfa(s, scale=np.arange(4, 17), show=False)
        out["DFA_Alpha1"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        out["DFA_Alpha1"] = np.nan

    try:
        _, vals = rcmse(s)
        finite = vals[np.isfinite(vals)]
        out["MSE_Mean"] = float(np.mean(finite)) if len(finite) else np.nan
    except Exception:
        out["MSE_Mean"] = np.nan

    return out


def run_surrogate_testing(rr_clean, record_id, output_dir, verbose=True):
    """IAAFT surrogate test for SampEn, DFA_Alpha1 and MSE_Mean.

    Per measure: the observed value, N_SURROGATES surrogates as the null, a two-tailed test at
    SURROGATE_ALPHA, and z = (observed - surrogate mean) / surrogate SD.
    Saves {record_id}_surrogate_test.png; returns the SURR_* columns.
    """
    if verbose:
        print(f"  IAAFT surrogate testing "
              f"(N={N_SURROGATES}, {N_IAAFT_ITERS} iters/surrogate)...")

    rng = np.random.default_rng(seed=SURROGATE_SEED)    # fixed seed → reproducible

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
        # Direction and percentile rank are recorded as well as the two-sided result.
        surr_results[f"SURR_{metric}_Direction"] = (
            "above" if obs_val > s_mean else "below")
        surr_results[f"SURR_{metric}_Pctile"] = float(
            100.0 * np.mean(obs_val > finite))
        surr_results[f"SURR_{metric}_N_Valid"] = int(len(finite))

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
    """Histograms of the IAAFT null distributions with the observed value and the 95% acceptance region."""
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


# ---- Step 5: MSE curve plot (per record) ----

def _plot_mse_curve(record_id, mse_scales, mse_values, output_dir):
    """MSE curve plot, {record_id}_MSE_curve.png, with MSE_Mean marked."""
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


# ---- Step 6: six-panel diagnostic plot (per record) ----

def _plot_record(record_id, time, raw_ecg, cleaned_ecg, r_peaks,
                 rr_times_raw, rr_raw, rr_times_clean, rr_clean,
                 freqs, psd_power, complexity_metrics, psd_beta=np.nan,
                 output_dir="outputs"):
    """Six-panel diagnostic figure, {record_id}_diagnostics.png: raw ECG, cleaned ECG, R-R intervals,
    PSD with slope fit, Poincare plot, complexity bar chart.
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


# ---- Step 7: correlation matrices ----

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
    """Correlation matrices of the core measures: all records, and records with
    PSD_Beta_R2 >= PSD_BETA_R2_THRESHOLD only.
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


# ---- Step 8: radar charts ----

def plot_radar_charts(df, output_dir, cohort_label="Healthy",
                      cohort_m=None, cohort_s=None):
    """Radar charts of the five radar metrics.

    Parameters
    ----------
    df            : per-record metrics (one row per subject)
    output_dir    : where to save PNGs
    cohort_label  : used in titles and file names
    cohort_m      : optional Series of means for z-scoring (the control cohort for a patient call);
                    if None, computed from df
    cohort_s      : optional Series of SDs

    Output files
    ------------
    Radar_Individual_{cohort_label}.png   per-subject z-scores
    Radar_Cohort_Mean_{cohort_label}.png  cohort mean with +/-1 SD band: raw units (min-max scaled per
                                          spoke) when self-normalized, z-scores against the reference
                                          otherwise
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

    # Self-normalized (reference cohort) or cross-normalized (against a reference) call.
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

    # -- Chart 1: individual subjects (z-scores) --
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

    # -- Chart 2: cohort mean +/- 1 SD (raw units if self-normalized, else z-scores) --

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    if self_normalised:
        # Raw units: min-max scale each spoke to [0, 1].
        raw_mean = sub.mean()
        raw_sd   = sub.std(ddof=1)
        raw_min  = sub.min()
        raw_max  = sub.max()
        span     = (raw_max - raw_min).replace(0, 1)

        mean_v = ((raw_mean - raw_min) / span).values.tolist()
        upper  = (((raw_mean + raw_sd) - raw_min) / span).clip(0, None).values.tolist()
        lower  = (((raw_mean - raw_sd) - raw_min) / span).clip(0, None).values.tolist()

        # Annotate each spoke label with the mean value.
        spoke_labels = [
            f"{lbl}\n({raw_mean[m]:.3f})"
            for lbl, m in zip(r_labels, r_metrics)
        ]
        ref_rings   = [0.25, 0.5, 0.75, 1.0]
        y_axis_note = "min-max scaled; spoke labels show raw mean"

    else:
        # z-scores relative to the reference
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

    # Return the normalization parameters.
    return cohort_m, cohort_s


def plot_radar_comparison(df_healthy, df_psych, output_dir,
                          cohort_m, cohort_s):
    """Overlay of the two cohort means on the same polar axes, z-scored on the reference cohort.
    Saved as Radar_Comparison_Healthy_vs_Psychiatric.png.
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


# ---- Step 9: per-record analysis (WFDB record) ----

def analyze_record(record_path, lead_index=0, plot=True,
                   run_surrogates=True, verbose=True,
                   output_dir="outputs"):
    """Analyze one WFDB record (MIT-BIH).

    Parameters
    ----------
    record_path    : str   WFDB record path (no extension)
    lead_index     : int   ECG lead (0 = the first channel)
    plot           : bool  save diagnostic and MSE curve plots
    run_surrogates : bool  run IAAFT surrogate testing
    verbose        : bool  print progress
    output_dir     : str   output folder

    Returns
    -------
    dict of all computed metrics (including SURR_* columns)
    """
    record_id = os.path.basename(record_path)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}\n  Analyzing record: {record_id}\n{'='*60}")

    record        = wfdb.rdrecord(record_path)
    sampling_rate = record.fs
    # Clock time of the analyzed window. MIT-BIH onset times are not standardized, so the window falls
    # at a different time of day for each record.
    _base_time = getattr(record, "base_time", None)
    _base_date = getattr(record, "base_date", None)
    record_start_clock = (_base_time.strftime("%H:%M:%S")
                          if _base_time is not None else "unknown")
    record_start_date  = (_base_date.isoformat()
                          if _base_date is not None else "unknown")
    if verbose:
        print(f"  Sampling rate: {sampling_rate} Hz")

    # -- Analysis window --
    # Skip MITBIH_SKIP_SECONDS (600 s) and analyze the next ANALYSIS_MINUTES, as for HRV-ACC.
    lead      = record.p_signal[:, lead_index]
    skip_n    = int(MITBIH_SKIP_SECONDS * sampling_rate)
    n_samples = int(ANALYSIS_MINUTES * 60 * sampling_rate)
    lead      = lead[skip_n:skip_n + n_samples]
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

    rr_clean, artifact_pct, num_rr, art_detail = reject_ectopic_beats(
        rr_raw, verbose=verbose)
    rr_clean = truncate_series(rr_clean)
    num_rr   = len(rr_clean)          # must follow truncation, not precede it
    rr_times_clean = np.concatenate(
        [[rr_times_raw[0]], rr_times_raw[0] + np.cumsum(rr_clean[:-1])]
    )

    # -- Time-domain HRV --
    # Computed from the artifact-cleaned R-R array with the same formulas as the HRV-ACC path.
    if verbose: print("  Time-domain HRV (from cleaned RR, matching HRV-ACC)...")
    rr_ms_clean     = rr_clean * 1000.0
    successive_diff = np.diff(rr_ms_clean)
    hrv_meannn = float(np.mean(rr_ms_clean))
    hrv_sdnn   = float(np.std(rr_ms_clean, ddof=1))
    hrv_rmssd  = float(np.sqrt(np.mean(successive_diff ** 2)))
    hrv_pnn50  = float(np.sum(np.abs(successive_diff) > 50) /
                       len(successive_diff) * 100)
    hrv_meanhr = float(60.0 / np.mean(rr_clean))
    sd1        = hrv_rmssd / np.sqrt(2)
    sd2        = float(np.sqrt(max(2 * hrv_sdnn ** 2 - sd1 ** 2, 0.0)))
    sd1sd2     = sd1 / sd2 if sd2 > 0 else np.nan

    if verbose: print("  Frequency-domain HRV + PSD beta...")
    freq_metrics, freqs, power = compute_frequency_domain(rr_clean, rr_times_clean)

    if verbose: print("  Complexity metrics (SampEn, ApEn, PermEn, MSE, DFA, RQA)...")
    complexity_metrics, mse_scales, mse_values = compute_complexity_metrics(
        rr_clean, quant_step=MITBIH_QUANT_STEP_S)

    # Onset-transient check: heart rate in the first ten minutes of the window against the rest.
    try:
        # Uses the time base this record was analyzed on.
        complexity_metrics.update(
            mitbih_onset_check(rr_clean, rr_times_clean))
    except Exception as e:
        print(f"    onset check skipped ({type(e).__name__}: {e})")

    if plot:
        _plot_mse_curve(record_id, mse_scales, mse_values, output_dir)

    # -- Assemble result dict --
    result = {"Record_ID": record_id}

    result["Record_Start_Clock"] = record_start_clock
    result["MITBIH_Skip_Seconds"] = MITBIH_SKIP_SECONDS
    result["Record_Start_Date"]  = record_start_date
    result["Analysis_Window_Min"] = ANALYSIS_MINUTES
    result["Num_RR"]           = num_rr
    result["Artifact_Percent"] = artifact_pct
    result["QC_RR_Count_OK"]   = num_rr >= 240
    result["QC_Artifact_OK"]   = artifact_pct < 5.0
    result.update(art_detail)

    result["HRV_MeanNN"] = hrv_meannn
    result["HRV_SDNN"]   = hrv_sdnn
    result["HRV_RMSSD"]  = hrv_rmssd
    result["HRV_pNN50"]  = hrv_pnn50
    result["HRV_MeanHR"] = hrv_meanhr
    result["HRV_SD1"]    = sd1
    result["HRV_SD2"]    = sd2
    result["HRV_SD1SD2"] = sd1sd2

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


# ---- Step 10: multi-record orchestrator and command line ----

def _cli():
    """Command-line entry point for the estimator validation and the environment record:

        python pipeline.py --validate-estimators [--out DIR]
        python pipeline.py --environment [--out DIR]
    """
    import argparse
    ap = argparse.ArgumentParser(description="pipeline utilities")
    ap.add_argument("--validate-estimators", action="store_true",
                    help="ground-truth recovery for the custom estimators")
    ap.add_argument("--environment", action="store_true",
                    help="write Environment.csv and requirements.txt")
    ap.add_argument("--out", default=os.path.join("outputs", "validation"))
    args = ap.parse_args()
    if not (args.validate_estimators or args.environment):
        ap.print_help()
        return
    os.makedirs(args.out, exist_ok=True)
    if args.environment:
        write_environment_record(args.out)
    if args.validate_estimators:
        print("  Running estimator validation on synthetic fGn "
              "(this takes a few minutes)...")
        validate_estimators(args.out)


def _write_environment_once(output_dir):
    """Write the environment record at most once per session for each output folder."""
    try:
        if output_dir in _ENVIRONMENT_WRITTEN:
            return
        write_environment_record(output_dir)
        _ENVIRONMENT_WRITTEN.add(output_dir)
    except Exception as e:
        print(f"    !! environment record NOT written "
              f"({type(e).__name__}: {e})")


def run_analysis(record_paths, cohort_label="Healthy",
                 lead_index=0, plot=True,
                 run_surrogates=True, save_csv=True,
                 output_dir="outputs",
                 cohort_m=None, cohort_s=None):
    """Run the pipeline over a list of WFDB record paths.

    Parameters
    ----------
    record_paths   : list[str]  WFDB record paths (no extension)
    cohort_label   : str        used in file names and headers
    lead_index     : int        ECG lead index
    plot           : bool       save per-record plots
    run_surrogates : bool       run IAAFT surrogate testing
    save_csv       : bool       save the results CSV
    output_dir     : str        output folder
    cohort_m       : Series     optional external normalization mean
    cohort_s       : Series     optional external normalization SD

    Returns
    -------
    df, cohort_m, cohort_s : per-record results and the normalization parameters
    """
    os.makedirs(output_dir, exist_ok=True)
    # Library versions and seeds, written beside the results.
    _write_environment_once(output_dir)

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

    # -- Results table --
    print(f"\n{'='*60}\n  {cohort_label.upper()} RESULTS TABLE\n{'='*60}")
    with pd.option_context('display.max_columns', None, 'display.width', 130,
                           'display.float_format', '{:.4f}'.format):
        print(df.T.to_string())

    # -- QA summary --
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

    # -- Cohort stats --
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

    # -- PSD slope fit quality --
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

    # -- Scatter plots --
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

    # -- Correlation matrices --
    plot_correlation_matrices(df, output_dir)

    # -- Radar charts --
    returned = plot_radar_charts(df, output_dir,
                                 cohort_label=cohort_label,
                                 cohort_m=cohort_m,
                                 cohort_s=cohort_s)
    if returned is not None:
        cohort_m, cohort_s = returned

    # -- CSV --
    if save_csv:
        csv_path  = os.path.join(output_dir, f"ecg_results_{cohort_label}.csv")
        df_export = pd.concat([df, stats_df])
        df_export.to_csv(csv_path)
        print(f"\n  Results saved  → {csv_path}")

    return df, cohort_m, cohort_s

# ---- Step 11: core numerics (NumPy and SciPy), every parameter explicit ----

def _embed_windows(x, m):
    """Sliding windows of length m (for entropy template matching)."""
    n = len(x) - m + 1
    return np.lib.stride_tricks.sliding_window_view(x, m)[:n]


# ---- Shared HRV-ACC per-subject analysis ----
# One implementation for both HRV-ACC cohorts; the cohort scripts pass labels, output folder and plot
# color.

def plot_rr_diagnostic(record_id, rr_times_raw, rr_raw,
                       rr_times_clean, rr_clean,
                       freqs, psd_power, complexity_metrics,
                       psd_beta, output_dir,
                       cohort_title="HRV-ACC", clean_colour="steelblue"):
    """Four-panel diagnostic figure for R-R input: raw and cleaned intervals, PSD with band shading and
    slope fit, Poincare plot, complexity bar chart.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ECG Criticality Analysis — {record_id} ({cohort_title})",
        fontsize=13
    )

    # 1) R-R intervals, raw and cleaned
    ax = axes[0, 0]
    ax.plot(rr_times_raw,   rr_raw   * 1000, lw=0.8,
            color='gray',   alpha=0.6, label='Raw')
    ax.plot(rr_times_clean, rr_clean * 1000, lw=0.9,
            color=clean_colour, label='Cleaned')
    ax.set_title("RR Intervals (raw vs artifact-rejected)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("RR (ms)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2) PSD with band shading and slope fit
    ax = axes[0, 1]
    ax.semilogy(freqs, psd_power, color='darkviolet', lw=1.2, label='PSD')
    ax.axvspan(*VLF_BAND, alpha=0.15, color='blue',   label='VLF')
    ax.axvspan(*LF_BAND,  alpha=0.15, color='green',  label='LF')
    ax.axvspan(*HF_BAND,  alpha=0.15, color='orange', label='HF')
    if np.isfinite(psd_beta):
        mask = (
            (freqs >= BETA_FIT_BAND[0]) &
            (freqs <= BETA_FIT_BAND[1]) &
            (freqs > 0)
        )
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

    # 3) Poincare plot
    ax = axes[1, 0]
    ax.scatter(rr_clean[:-1] * 1000, rr_clean[1:] * 1000,
               s=8, alpha=0.5, color='teal')
    ax.set_title("Poincaré Plot")
    ax.set_xlabel("RR$_n$ (ms)"); ax.set_ylabel("RR$_{n+1}$ (ms)")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 4) Complexity bar chart
    ax = axes[1, 1]
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


# ---- Per-record analysis (HRV-ACC CSV) ----


def analyze_hrvacc_record(filepath, output_dir, plot=True,
                          run_surrogates=True, verbose=True,
                          cohort_title="HRV-ACC",
                          clean_colour="steelblue"):
    """Analyze one HRV-ACC record (R-R CSV), with the shared pipeline functions.

    Returns a flat dict of all metrics, with the same column names as analyze_record().
    """
    record_id = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}\n  Analyzing: {record_id}\n{'='*60}")

    # -- Load R-R intervals --
    rr_raw, rr_times_raw, n_total = load_rr_window(
        filepath, verbose=verbose
    )

    if verbose:
        print(f"  {len(rr_raw)} beats in analysis window")

    # -- Artifact and ectopic beat rejection --
    rr_clean, artifact_pct, num_rr, art_detail = reject_ectopic_beats(
        rr_raw, verbose=verbose
    )
    # Optional length matching: set pipeline.TRUNCATE_N to a common beat count.
    rr_clean = truncate_series(rr_clean)
    num_rr   = len(rr_clean)

    # Time axis for the cleaned series
    rr_times_clean = np.concatenate(
        [[rr_times_raw[0]], rr_times_raw[0] + np.cumsum(rr_clean[:-1])]
    )

    # -- Time-domain HRV --
    # From the cleaned R-R array, with the formulas of nk.hrv_time().
    if verbose:
        print("  Time-domain HRV...")
    rr_ms = rr_clean * 1000.0
    successive_diff = np.diff(rr_ms)

    hrv_meannn = float(np.mean(rr_ms))
    hrv_sdnn   = float(np.std(rr_ms, ddof=1))
    hrv_rmssd  = float(np.sqrt(np.mean(successive_diff ** 2)))
    hrv_pnn50  = float(np.sum(np.abs(successive_diff) > 50) /
                       len(successive_diff) * 100)
    hrv_meanhr = float(60.0 / np.mean(rr_clean))

    # -- Poincare (SD1 = RMSSD / sqrt(2), SD2) --
    sd1     = hrv_rmssd / np.sqrt(2)
    sd2     = float(np.sqrt(2 * hrv_sdnn**2 - sd1**2))
    sd1sd2  = sd1 / sd2 if sd2 > 0 else np.nan

    # -- Frequency domain and spectral slope --
    if verbose:
        print("  Frequency-domain HRV + PSD beta...")
    freq_metrics, freqs, power = compute_frequency_domain(
        rr_clean, rr_times_clean
    )

    # -- Amplitude and spectral measures on the MIT-BIH grid --
    # These need beat times, so they are computed here (the nonlinear measures are re-quantized inside
    # compute_complexity_metrics()). HRV-ACC only.
    requant_amp = requant_amplitude_spectral(rr_clean, rr_times_clean)

    # -- Complexity measures --
    if verbose:
        print("  Complexity metrics (SampEn, ApEn, PermEn, MSE, DFA, RQA)...")
    complexity_metrics, mse_scales, mse_values = compute_complexity_metrics(
        rr_clean
    )

    # -- MSE curve plot --
    if plot:
        _plot_mse_curve(record_id, mse_scales, mse_values, output_dir)

    # -- Assemble result dict --
    result = {"Record_ID": record_id}

    result["Num_RR"]           = num_rr
    result["Artifact_Percent"] = artifact_pct
    result["QC_RR_Count_OK"]   = num_rr >= 240
    result.update(art_detail)
    result["QC_Artifact_OK"]   = artifact_pct < 5.0

    result["HRV_MeanNN"] = hrv_meannn
    result["HRV_SDNN"]   = hrv_sdnn
    result["HRV_RMSSD"]  = hrv_rmssd
    result["HRV_pNN50"]  = hrv_pnn50
    result["HRV_MeanHR"] = hrv_meanhr
    result["HRV_SD1"]    = sd1
    result["HRV_SD2"]    = sd2
    result["HRV_SD1SD2"] = sd1sd2

    result.update(freq_metrics)
    result.update(complexity_metrics)
    result.update(requant_amp)

    result["QC_MSE_Valid"]    = np.isfinite(result.get("MSE_Mean",   np.nan))
    result["QC_DFA1_Valid"]   = np.isfinite(result.get("DFA_Alpha1", np.nan))
    result["QC_SampEn_Valid"] = np.isfinite(result.get("SampEn",     np.nan))
    result["QC_Overall_Pass"] = (
        result["QC_RR_Count_OK"]
        and result["QC_Artifact_OK"]
        and result["QC_DFA1_Valid"]
    )

    # -- IAAFT surrogate testing --
    if run_surrogates:
        surr = run_surrogate_testing(
            rr_clean, record_id, output_dir, verbose=verbose
        )
        result.update(surr)

    # -- Diagnostic plot --
    if plot:
        plot_rr_diagnostic(
            record_id,
            rr_times_raw, rr_raw,
            rr_times_clean, rr_clean,
            freqs, power, complexity_metrics,
            psd_beta=freq_metrics.get("PSD_Beta", np.nan),
            output_dir=output_dir,
            cohort_title=cohort_title,
            clean_colour=clean_colour,
        )

    if verbose:
        print(f"\n  ✓ Done: {record_id}")

    return result


# ---- Entropy, recurrence and resolution numerics ----


def sampen_counts(x, m=2, r=None, chunk=512):
    """Template-match counts (A, B) for sample entropy (Chebyshev distance, no self-matches).
    SampEn = -log(A / B).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if r is None:
        r = ENTROPY_R_FRAC * np.std(x, ddof=1)
    if N < m + 2 or not np.isfinite(r) or r <= 0:
        return np.nan, np.nan
    # The (m+1)-distance is max(m-distance, |x_{i+m} - x_{j+m}|), so it reuses the m-distance; float32 is
    # enough for a threshold comparison.
    Xm   = _embed_windows(x, m)[:N - m].astype(np.float32)
    tail = x[m:m + (N - m)].astype(np.float32)      # the (m+1)-th coordinate
    rf = np.float32(r)
    A = 0.0
    B = 0.0
    n = len(Xm)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dm = np.max(np.abs(Xm[s:e, None, :] - Xm[None, :, :]), axis=2)
        extra = np.abs(tail[s:e, None] - tail[None, :])
        bm = dm <= rf
        am = bm & (extra <= rf)                  # == (max(dm, extra) <= r)
        rows = np.arange(e - s)
        bm[rows, np.arange(s, e)] = False        # remove self-matches
        am[rows, np.arange(s, e)] = False
        B += float(bm.sum())
        A += float(am.sum())
    return A, B


def sample_entropy(x, m=None, r_frac=None, r=None):
    """Sample entropy with an explicitly stated tolerance."""
    m      = ENTROPY_M      if m      is None else m
    r_frac = ENTROPY_R_FRAC if r_frac is None else r_frac
    x = np.asarray(x, dtype=float)
    if r is None:
        r = r_frac * np.std(x, ddof=1)
    A, B = sampen_counts(x, m=m, r=r)
    if not (np.isfinite(A) and np.isfinite(B)) or A <= 0 or B <= 0:
        return np.nan
    return float(-np.log(A / B))


def approximate_entropy(x, m=None, r_frac=None, r=None, chunk=512):
    """Approximate entropy with an explicitly stated tolerance."""
    m      = ENTROPY_M      if m      is None else m
    r_frac = ENTROPY_R_FRAC if r_frac is None else r_frac
    x = np.asarray(x, dtype=float)
    N = len(x)
    if r is None:
        r = r_frac * np.std(x, ddof=1)
    if N < m + 2 or not np.isfinite(r) or r <= 0:
        return np.nan

    def phi(mm):
        X = _embed_windows(x, mm)[:N - mm + 1]
        n = len(X)
        tot = 0.0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = np.max(np.abs(X[s:e, None, :] - X[None, :, :]), axis=2)
            c = (d <= r).sum(axis=1) / float(n)
            tot += float(np.log(c[c > 0]).sum())
        return tot / float(n)

    try:
        return float(phi(m) - phi(m + 1))
    except Exception:
        return np.nan


def rcmse(x, m=None, r_frac=None, max_scale=None, min_points=None):
    """Refined composite multiscale entropy (Wu et al., 2014).

    At each scale tau all tau offsets are used and the match counts are pooled before the logarithm.
    The tolerance r is fixed from the original series SD, not recomputed per scale. The loop stops when
    N // tau falls below min_points, so subjects with fewer beats retain fewer scales; the returned scale
    array shows this, and the caller also averages over a fixed common scale set.
    """
    m          = ENTROPY_M       if m          is None else m
    r_frac     = ENTROPY_R_FRAC  if r_frac     is None else r_frac
    max_scale  = MSE_MAX_SCALE   if max_scale  is None else max_scale
    min_points = MSE_MIN_POINTS  if min_points is None else min_points

    x = np.asarray(x, dtype=float)
    N = len(x)
    r = r_frac * np.std(x, ddof=1)
    scales, vals = [], []
    for tau in range(1, int(max_scale) + 1):
        if N // tau < min_points:
            break
        A_tot = 0.0
        B_tot = 0.0
        for k in range(tau):
            seg = x[k:]
            L = (len(seg) // tau) * tau
            if L < tau * (m + 1):
                continue
            cg = seg[:L].reshape(-1, tau).mean(axis=1)
            A, B = sampen_counts(cg, m=m, r=r)
            if np.isfinite(A) and np.isfinite(B):
                A_tot += A
                B_tot += B
        scales.append(float(tau))
        vals.append(float(-np.log(A_tot / B_tot))
                    if (A_tot > 0 and B_tot > 0) else np.nan)
    return np.asarray(scales, dtype=float), np.asarray(vals, dtype=float)


def ordinal_tie_rate(x, dim=3, delay=1):
    """Fraction of embedding vectors containing at least one exact tie (ties from quantization affect
    how permutation entropy assigns ordinal patterns).
    """
    try:
        x = np.asarray(x, dtype=float)
        n = len(x) - (dim - 1) * delay
        if n <= 0:
            return np.nan
        X = np.stack([x[i:i + (dim - 1) * delay + 1:delay] for i in range(n)])
        Xs = np.sort(X, axis=1)
        has_tie = np.any(np.diff(Xs, axis=1) == 0.0, axis=1)
        return float(np.mean(has_tie))
    except Exception:
        return np.nan


def _delay_embed(x, dim, delay):
    n = len(x) - (dim - 1) * delay
    if n <= 0:
        return None
    return np.stack([x[i:i + (dim - 1) * delay + 1:delay] for i in range(n)],
                    axis=0).astype(float)


def resolve_theiler(x, dim, delay, setting=None, fallback=None):
    """Theiler window in samples for one series. Returns (window, source). The embedding window
    (dim - 1) * delay + 1 is the floor.
    """
    setting  = RQA_THEILER          if setting  is None else setting
    fallback = RQA_THEILER_FALLBACK if fallback is None else fallback
    embed_w = int((dim - 1) * delay + 1)
    if isinstance(setting, (int, np.integer)) and not isinstance(setting, bool):
        return max(int(setting), embed_w), "fixed"
    if setting == "mi":
        try:
            w = int(select_delay_mutual_information(x))
            if w >= embed_w:
                return w, "mutual_information"
            return embed_w, "embedding_window_floor"
        except Exception:
            return embed_w, "embedding_window_mi_failed"
    if isinstance(fallback, (int, np.integer)):
        return max(int(fallback), embed_w), "fixed_fallback"
    return embed_w, "embedding_window"


def recurrence_matrix(x, dim=None, delay=None, rec_rate=None, max_pts=None,
                      theiler=None):
    """Binary recurrence matrix at a fixed recurrence rate.

    The recurrent pairs are the rec_rate fraction of eligible pairs with the smallest distances, so the
    recurrence rate is the same for every subject regardless of amplitude.
    """
    dim      = RQA_DIM      if dim      is None else dim
    delay    = RQA_DELAY    if delay    is None else delay
    rec_rate = RQA_REC_RATE if rec_rate is None else rec_rate
    x = np.asarray(x, dtype=float)
    if max_pts is not None and len(x) > max_pts:
        x = x[:max_pts]
    X = _delay_embed(x, dim, delay)
    if X is None or len(X) < 20:
        return None, np.nan
    # Squared-distance identity ||a-b||^2 = |a|^2 + |b|^2 - 2a.b keeps the working array at n x n.
    sq = np.einsum('ij,ij->i', X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)          # kill negative round-off before sqrt
    d = np.sqrt(d2, out=d2)              # in place; d2 is no longer needed
    n = len(X)
    # Theiler exclusion: k is the offset of the first retained diagonal; k = theiler excludes every pair
    # closer than `theiler` samples.
    if theiler is None:
        theiler, _src = resolve_theiler(x, dim, delay)
    theiler = max(1, int(theiler))
    n_pairs_total = n * (n - 1) // 2
    if theiler >= n:
        # Degenerate: the window would exclude every pair.
        return None, np.nan
    iu = np.triu_indices(n, k=theiler)
    vals = d[iu]

    # Exact-count thresholding: select exactly k = round(rec_rate * n_pairs) smallest distances, so the
    # rate is exact even with tied (quantized) distances. Ties at the cut are broken by argpartition
    # order, which is deterministic for a given input. n_pairs counts Theiler-eligible pairs only.
    n_pairs = vals.size
    if n_pairs < 10:
        return None, np.nan
    k = int(round(rec_rate * n_pairs))
    k = max(1, min(k, n_pairs))
    sel = np.argpartition(vals, k - 1)[:k]
    eps = float(vals[sel].max())

    R = np.zeros((n, n), dtype=bool)
    R[iu[0][sel], iu[1][sel]] = True
    R |= R.T
    np.fill_diagonal(R, False)
    # Pair counts travel with the matrix without changing the (R, eps) return contract.
    _RECURRENCE_PAIR_INFO.clear()
    _RECURRENCE_PAIR_INFO.update(
        # Dimension of the matrix these counts describe; rqa_metrics checks it.
        _n_points=int(n),
        RQA_Theiler_W=int(theiler),
        RQA_N_Pairs_Total=int(n_pairs_total),
        RQA_N_Pairs_Eligible=int(n_pairs),
        RQA_N_Pairs_Theiler_Excluded=int(n_pairs_total - n_pairs),
        RQA_Theiler_Excluded_Frac=(float(n_pairs_total - n_pairs) /
                                   float(n_pairs_total)
                                   if n_pairs_total > 0 else np.nan))
    return R, eps


def _line_lengths(R, vertical=False):
    """Lengths of all diagonal (or vertical) runs of recurrent points (vectorized run-length encoding)."""
    R = np.asarray(R, dtype=bool)
    n = R.shape[0]

    def runs(line):
        if line.size == 0 or not line.any():
            return np.empty(0, dtype=int)
        padded = np.concatenate(([False], line, [False]))
        d = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        return ends - starts

    out = []
    if vertical:
        for j in range(n):
            r = runs(R[:, j])
            if r.size:
                out.append(r)
    else:
        for off in range(-(n - 1), n):
            if off == 0:
                continue
            r = runs(np.diagonal(R, offset=off))
            if r.size:
                out.append(r)
    return (np.concatenate(out) if out else np.empty(0, dtype=int))


def rqa_metrics(x, dim=None, delay=None, rec_rate=None,
                l_min=None, v_min=None, max_pts=None, R=None, eps=None,
                theiler=None):
    """Recurrence quantification at a fixed recurrence rate.

    R and eps let a caller supply a recurrence matrix already built with the same dim, delay, rec_rate
    and max_pts; the caller must guarantee that match.
    """
    dim      = RQA_DIM      if dim      is None else dim
    delay    = RQA_DELAY    if delay    is None else delay
    rec_rate = RQA_REC_RATE if rec_rate is None else rec_rate
    l_min    = RQA_L_MIN    if l_min    is None else l_min
    v_min    = RQA_V_MIN    if v_min    is None else v_min

    nan = dict(RQA_REC=np.nan, RQA_DET=np.nan, RQA_LAM=np.nan,
               RQA_L_Mean=np.nan, RQA_Eps=np.nan)
    if R is None:
        R, eps = recurrence_matrix(x, dim, delay, rec_rate, max_pts,
                                   theiler=theiler)
    if R is None:
        return nan
    n = R.shape[0]
    total_rec = float(R.sum())
    if total_rec <= 0:
        return nan
    diag_len = _line_lengths(R, vertical=False)
    vert_len = _line_lengths(R, vertical=True)
    det_pts = float(diag_len[diag_len >= l_min].sum())
    lam_pts = float(vert_len[vert_len >= v_min].sum())
    l_mean = (float(diag_len[diag_len >= l_min].mean())
              if (diag_len >= l_min).any() else np.nan)
    # Pair counts are used only if their recorded dimension matches this matrix.
    info = dict(_RECURRENCE_PAIR_INFO)
    if info.get("_n_points") != n:
        info = {k: np.nan for k in info if not k.startswith("_")}
        info["RQA_Pair_Counts_Stale"] = True
    else:
        info.pop("_n_points", None)
        info["RQA_Pair_Counts_Stale"] = False
    n_elig = info.get("RQA_N_Pairs_Eligible", np.nan)
    out = dict(
        # Against eligible pairs: fixed at rec_rate by construction (a normalization check, not a test).
        RQA_REC=(total_rec / 2.0) / n_elig if n_elig else np.nan,
        # Against all pairs: below rec_rate by the fraction the Theiler window removed.
        RQA_REC_AllPairs=total_rec / (n * (n - 1)),
        RQA_DET=det_pts / total_rec,
        RQA_LAM=lam_pts / total_rec,
        RQA_L_Mean=l_mean,
        RQA_Eps=float(eps))
    out.update(info)
    # -- l_min sensitivity --
    for lm in RQA_LMIN_SWEEP:
        if lm == l_min:
            continue
        try:
            pts = float(diag_len[diag_len >= lm].sum())
            out[f"RQA_DET_lmin{lm}"] = pts / total_rec if total_rec else np.nan
        except Exception:
            out[f"RQA_DET_lmin{lm}"] = np.nan
    return out


def select_delay_mutual_information(x, max_delay=30, n_bins=16):
    """Embedding delay = first local minimum of the mutual information curve."""
    try:
        x = np.asarray(x, dtype=float)
        mis = []
        for tau in range(1, max_delay + 1):
            a, b = x[:-tau], x[tau:]
            c, _, _ = np.histogram2d(a, b, bins=n_bins)
            pab = c / c.sum()
            pa  = pab.sum(axis=1, keepdims=True)
            pb  = pab.sum(axis=0, keepdims=True)
            nz  = pab > 0
            mis.append(float(np.sum(pab[nz] * np.log(pab[nz] /
                                                     (pa @ pb)[nz]))))
        mis = np.asarray(mis)
        for i in range(1, len(mis) - 1):
            if mis[i] < mis[i - 1] and mis[i] < mis[i + 1]:
                return int(i + 1)
        return int(np.argmin(mis) + 1)
    except Exception:
        return RQA_DELAY


def select_dimension_fnn(x, delay=1, max_dim=10, rtol=15.0, threshold=0.01,
                         max_pts=1500):
    """Embedding dimension by the false-nearest-neighbours criterion."""
    try:
        x = np.asarray(x, dtype=float)
        if len(x) > max_pts:
            x = x[:max_pts]
        sd = np.std(x, ddof=1)
        for dim in range(1, max_dim + 1):
            X = _delay_embed(x, dim, delay)
            Xn = _delay_embed(x, dim + 1, delay)
            if X is None or Xn is None or len(Xn) < 20:
                return dim
            X = X[:len(Xn)]
            d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(d, np.inf)
            nn = np.argmin(d, axis=1)
            dmin = d[np.arange(len(X)), nn]
            extra = np.abs(Xn[:, -1] - Xn[nn, -1])
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(dmin > 0, extra / dmin, np.inf)
            frac = float(np.mean((ratio > rtol) | (extra / sd > 2.0)))
            if frac < threshold:
                return int(dim)
        return int(max_dim)
    except Exception:
        return RQA_DIM


def stationarity_diagnostics(x, n_segments=None):
    """Stationarity diagnostics for one R-R series.

    1. Reverse-arrangement test on segment variances (Bendat and Piersol).
    2. Variance ratio between the first and second half.
    3. ADF and KPSS, if statsmodels is installed (ADF's null is a unit root, KPSS's is stationarity).

    statsmodels clamps the KPSS p value to 0.01-0.10; Stat_KPSS_p_Capped flags a clamped value.
    Stat_Tests_Status records whether statsmodels was absent or the computation failed.
    """
    import warnings as _warnings

    n_segments = STATIONARITY_SEGMENTS if n_segments is None else n_segments
    out = dict(Stat_RA_z=np.nan, Stat_RA_p=np.nan, Stat_RA_Stationary=np.nan,
               Stat_VarRatio=np.nan, Stat_ADF_p=np.nan, Stat_KPSS_p=np.nan,
               Stat_KPSS_p_Capped=False,
               Stat_Tests_Status=STATSMODELS_STATUS)

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        out["Stat_Tests_Status"] = "series too short"
        return out

    # -- 1. Reverse-arrangement test --
    try:
        L = len(x) // n_segments
        if L >= 5:
            v = np.array([np.var(x[i * L:(i + 1) * L], ddof=1)
                          for i in range(n_segments)])
            # For each segment, how many later segments have a smaller variance.
            A = int(sum(int(np.sum(v[i] > v[i + 1:]))
                        for i in range(n_segments - 1)))
            N = n_segments
            mu = N * (N - 1) / 4.0
            sd = np.sqrt(N * (2 * N ** 2 + 3 * N - 5) / 72.0)
            z = (A - mu) / sd if sd > 0 else np.nan
            p = float(2 * _normdist.sf(abs(z))) if np.isfinite(z) else np.nan
            out.update(Stat_RA_z=float(z), Stat_RA_p=p,
                       Stat_RA_Stationary=(bool(p > 0.05)
                                           if np.isfinite(p) else np.nan))
    except Exception as exc:
        out["Stat_Tests_Status"] = f"reverse-arrangement failed: {exc}"

    # -- 2. Variance ratio --
    try:
        half = len(x) // 2
        if half > 10:
            v1 = np.var(x[:half], ddof=1)
            v2 = np.var(x[half:], ddof=1)
            out["Stat_VarRatio"] = float(v2 / v1) if v1 > 0 else np.nan
    except Exception:
        pass

    # -- 3. ADF and KPSS --
    if not STATSMODELS_AVAILABLE:
        return out          # status already records why; not an error

    try:
        out["Stat_ADF_p"] = float(_adfuller(x, autolag="AIC")[1])
    except Exception as exc:
        out["Stat_Tests_Status"] = f"ADF failed: {type(exc).__name__}"

    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            kp = float(_kpss(x, regression="c", nlags="auto")[1])
        out["Stat_KPSS_p"] = kp
        # A p value at an end of the lookup table is an inequality.
        out["Stat_KPSS_p_Capped"] = bool(
            kp <= 0.01 or kp >= 0.10
            or any("Interpolation" in type(w.category).__name__ or
                   "interpolat" in str(w.message).lower() for w in caught))
    except Exception as exc:
        out["Stat_Tests_Status"] = f"KPSS failed: {type(exc).__name__}"

    return out


def add_dither(x, step=None, rng=None):
    """Add uniform noise of one quantization step to break quantization ties."""
    step = QUANT_STEP_S if step is None else step
    rng  = np.random.default_rng(QUANT_SEED) if rng is None else rng
    x = np.asarray(x, dtype=float)
    return x + rng.uniform(-step / 2.0, step / 2.0, size=len(x))


def coarsen(x, step):
    """Round a series onto a coarser grid (simulate a larger quantization step)."""
    return np.round(np.asarray(x, dtype=float) / step) * step


# Measures carried through requant_amplitude_spectral() (the _Q128 columns).
REQUANT_AMPLITUDE_SPECTRAL = (
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD",
    "LF_Power_ms2", "HF_Power_ms2", "VLF_Power_ms2", "Total_Power_ms2",
    "VLF_LF_Ratio", "LF_HF_Ratio",
    "PSD_Beta", "PSD_Beta_NoMayer", "PSD_Beta_low",
)


def requant_amplitude_spectral(rr_clean, rr_times_clean,
                               step=REQUANT_TARGET_STEP_S):
    """Amplitude and spectral measures recomputed on the MIT-BIH sampling grid.

    Each interval is rounded to the grid with coarsen(), as for the other *_Q128 columns, and the time
    base is rebuilt from the coarsened intervals. (Rounding beat times instead gave group effect sizes
    within 0.01 for every measure here.)

    Returns a dict of <measure>_Q128 columns, or {} if the target grid is not coarser than the native one.
    """
    out = {}
    if not REQUANT_ENABLED or step is None or not step > QUANT_STEP_S * 1.5:
        return out
    try:
        x = np.asarray(rr_clean, dtype=float)
        xq = coarsen(x, step)
        t0 = float(np.asarray(rr_times_clean, dtype=float)[0])
        tq = np.concatenate([[t0], t0 + np.cumsum(xq[:-1])])
        xq_ms = xq * 1000.0
        d = np.diff(xq_ms)
        vals = {
            "HRV_MeanNN": float(np.mean(xq_ms)),
            "HRV_SDNN":   float(np.std(xq_ms, ddof=1)),
            "HRV_RMSSD":  float(np.sqrt(np.mean(d ** 2))),
        }
        fd, _, _ = compute_frequency_domain(xq, tq)
        for k in REQUANT_AMPLITUDE_SPECTRAL:
            if k not in vals:
                vals[k] = fd.get(k, np.nan)
        for k in REQUANT_AMPLITUDE_SPECTRAL:
            out[f"{k}_Q128"] = vals.get(k, np.nan)
    except Exception as e:
        # Raise rather than return an empty set of _Q128 columns.
        print(f"    !! amplitude/spectral re-quantization FAILED "
              f"({type(e).__name__}: {e})")
        for k in REQUANT_AMPLITUDE_SPECTRAL:
            out[f"{k}_Q128"] = np.nan
    return out


def synthesise_fgn(n, H, rng=None):
    """Exact fractional Gaussian noise by Davies-Harte circulant embedding, at unit variance.

    For fGn the DFA exponent is alpha = H and the spectral exponent is beta = 2H - 1; the lag-1
    autocorrelation should be 0.5 * (2**(2H) - 2).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = int(n)
    k = np.arange(0, n)
    # Autocovariance of fGn at unit variance.
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) +
               np.abs(k + 1) ** (2 * H))
    r = np.concatenate([g, [0.0], g[:0:-1]])
    lam = np.real(np.fft.fft(r))
    if np.any(lam < 0):
        lam = np.clip(lam, 0.0, None)      # tiny negatives from round-off
    m = len(lam)
    w = (rng.normal(size=m) + 1j * rng.normal(size=m)) / np.sqrt(2.0)
    w[0] = np.real(w[0])
    w[m // 2] = np.real(w[m // 2])
    y = np.fft.fft(np.sqrt(lam / m) * w)
    # sqrt(2) restores unit variance under this scaling convention.
    return np.real(y[:n]) * np.sqrt(2.0)


def validate_estimators(output_dir, lengths=(2000, 2500),
                        hurst=(0.6, 0.7, 0.8, 0.9), n_reps=5, seed=None):
    """Ground-truth recovery for the custom estimators at this study's record lengths and exponent range.

    Writes Estimator_Validation.csv: for each estimator and true value, the mean recovered value, its SD
    across repetitions and the bias. Constant bias cancels in a group difference; bias that varies with
    the true value does not. Pure simulation; no subject data.
    """
    import csv
    seed = QUANT_SEED if seed is None else seed
    rows = []
    for N in lengths:
        for H in hurst:
            rec = {"dfa_full": [], "dfa_a1": [], "dfa_a2": [], "mfdfa_h2": []}
            for rep in range(int(n_reps)):
                rng = np.random.default_rng(seed + 977 * rep + int(H * 100))
                x = synthesise_fgn(N, H, rng=rng)
                try:
                    v, _ = fractal_dfa(x, show=False)
                    rec["dfa_full"].append(float(v))
                except Exception:
                    pass
                try:
                    v, _ = fractal_dfa(x, scale=np.arange(4, 17), show=False)
                    rec["dfa_a1"].append(float(v))
                except Exception:
                    pass
                try:
                    v, _ = fractal_dfa(x, scale=np.arange(16, 65), show=False)
                    rec["dfa_a2"].append(float(v))
                except Exception:
                    pass
                try:
                    mf = compute_mfdfa(x, with_surrogates=False)
                    rec["mfdfa_h2"].append(float(mf.get("MFDFA_H2", np.nan)))
                except Exception:
                    pass
            for est, vals in rec.items():
                a = np.asarray([v for v in vals if np.isfinite(v)], float)
                # For fGn every one of these estimators targets H.
                rows.append({
                    "Estimator": est, "N": N, "True_H": H,
                    "Target_Value": H,
                    "Recovered_Mean": float(np.mean(a)) if a.size else np.nan,
                    "Recovered_SD": (float(np.std(a, ddof=1))
                                     if a.size > 1 else np.nan),
                    "Bias": (float(np.mean(a) - H) if a.size else np.nan),
                    "N_Reps_Valid": int(a.size)})
    path = os.path.join(output_dir, "Estimator_Validation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Estimator validation → {path}")
    return rows


def mitbih_onset_check(rr_clean, rr_times, first_seconds=600.0):
    """Onset-transient check for an MIT-BIH window: mean heart rate over the first `first_seconds` of the
    analyzed window against the remainder (the window starts 600 s into the record).
    """
    try:
        t = np.asarray(rr_times, dtype=float)
        x = np.asarray(rr_clean, dtype=float)
        t = t - t[0]
        early, late = t < first_seconds, t >= first_seconds
        if early.sum() < 30 or late.sum() < 30:
            return {}
        hr_e = 60.0 / np.mean(x[early])
        hr_l = 60.0 / np.mean(x[late])
        return {"Onset_HR_First10min": float(hr_e),
                "Onset_HR_Rest": float(hr_l),
                "Onset_HR_Diff": float(hr_e - hr_l),
                "Onset_N_Beats_Early": int(early.sum())}
    except Exception:
        return {}


def environment_record():
    """Software versions, random seeds and analysis constants in force."""
    import platform
    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed_quant": QUANT_SEED,
        "seed_surrogate": SURROGATE_SEED,
        "dither_replicates": QUANT_DITHER_REPS,
        "n_iaaft_iters": N_IAAFT_ITERS,
        "n_surrogates": globals().get("N_SURROGATES", None),
        "mfdfa_n_surrogates": MFDFA_N_SURROGATES,
        # Analysis constants recorded so an output file can be matched to its settings.
        "quant_step_s": QUANT_STEP_S,
        "quant_target_ratio": QUANT_TARGET_RATIO,
        "requant_target_step_s": REQUANT_TARGET_STEP_S,
        "dfa_alpha1_nmin_sweep": str(DFA_ALPHA1_NMIN_SWEEP),
        "entropy_r_sensitivity": str(ENTROPY_R_SENSITIVITY),
        "skip_seconds": SKIP_SECONDS,
        "mitbih_skip_seconds": MITBIH_SKIP_SECONDS,
        "analysis_seconds": ANALYSIS_SECONDS,
    }
    for mod in ("numpy", "scipy", "pandas", "sklearn", "neurokit2",
                "matplotlib", "wfdb", "statsmodels"):
        try:
            m = __import__(mod)
            rec[f"version_{mod}"] = getattr(m, "__version__", "unknown")
        except Exception:
            rec[f"version_{mod}"] = "not installed"
    return rec


def write_environment_record(output_dir):
    """Write Environment.csv and requirements.txt next to the results."""
    import csv
    rec = environment_record()
    path = os.path.join(output_dir, "Environment.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Item", "Value"])
        for k, v in rec.items():
            w.writerow([k, v])
    req = os.path.join(output_dir, "requirements.txt")
    with open(req, "w", encoding="utf-8") as f:
        for k, v in rec.items():
            if k.startswith("version_") and v not in ("not installed",
                                                      "unknown"):
                f.write(f"{k.replace('version_', '')}=={v}\n")
    print(f"  Environment record → {path}")
    return rec


def _mse_mean_safe(x):
    """RCMSE mean for a control condition; NaN rather than an exception."""
    try:
        _s, _v = rcmse(x, m=ENTROPY_M, r_frac=ENTROPY_R_FRAC,
                       max_scale=MSE_MAX_SCALE, min_points=MSE_MIN_POINTS)
        _v = np.asarray(_v, dtype=float)
        return float(np.nanmean(_v)) if np.isfinite(_v).any() else np.nan
    except Exception:
        return np.nan


def observed_rr_lattice(x):
    """Infer the quantization lattice present in an R-R series.

    The step is estimated as the smallest nonzero absolute first difference. Returns the step in
    seconds and the fraction of first differences that are integer multiples of it (near 1.0 means a
    uniform lattice). Every HRV-ACC series is on a 1.000 ms lattice.
    """
    try:
        d = np.abs(np.diff(np.asarray(x, dtype=float)))
        d = d[d > 1e-9]
        if d.size < 10:
            return np.nan, np.nan
        step_obs = float(np.min(d))
        ratios = d / step_obs
        on_lattice = float(np.mean(
            np.abs(ratios - np.round(ratios)) < 0.01))
        return step_obs, on_lattice
    except Exception:
        return np.nan, np.nan


def quantization_controls(x, step=None, target_ratio=None, theiler=None):
    """Recompute the quantization-sensitive measures under two control conditions.

    (a) Dither: uniform noise one quantization step wide removes the exact ties quantization creates.
    (b) Matched coarsening: the series is re-quantized so that the step-to-RMSSD ratio equals
        target_ratio. The primary target is QUANT_TARGET_RATIO (0.088), separate from the 0.10/0.15/0.20
        sweep; coarsening applies only where RMSSD exceeds about 11.4 ms, so controls are brought down to
        the patients' resolution. Quant_Matched_Applied records the branch taken per subject.

    Also records Quant_Step_Over_RMSSD.
    """
    step = QUANT_STEP_S if step is None else step
    out = {}
    try:
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(QUANT_SEED)
        # The Theiler window from the observed series is used for every condition.
        if theiler is None:
            theiler, _ = resolve_theiler(x, RQA_DIM, RQA_DELAY)
        out["Quant_Theiler_W"] = int(theiler)
        rmssd = float(np.sqrt(np.mean(np.diff(x) ** 2)))
        out["Quant_Step_s"]            = step
        out["Quant_Step_Over_RMSSD"]   = step / rmssd if rmssd > 0 else np.nan
        # Observed lattice (see observed_rr_lattice); it should match QUANT_STEP_S for every HRV-ACC subject.
        _step_obs, _on_lattice = observed_rr_lattice(x)
        out["Quant_Step_Observed_s"]     = _step_obs
        out["Quant_On_Lattice_Frac"]     = _on_lattice
        # Match tolerance 0.5%: the two candidate steps (1/1024 s and 1.000 ms) differ by 2.4%.
        out["Quant_Step_Assumed_Matches"] = (
            bool(abs(_step_obs - step) < 0.005 * step)
            if np.isfinite(_step_obs) else False)
        out["Quant_Step_Obs_Over_Assumed"] = (
            float(_step_obs / step) if np.isfinite(_step_obs) and step else np.nan)

        # -- Dither, primary realization (original column names) --
        xd = add_dither(x, step=step, rng=rng)
        out["SampEn_Dither"] = sample_entropy(xd)
        out["ApEn_Dither"]   = approximate_entropy(xd)
        try:
            pe, _ = nk.entropy_permutation(xd, dimension=3, delay=1)
            out["PermEn_Dither"] = float(pe)
        except Exception:
            out["PermEn_Dither"] = np.nan
        rq = rqa_metrics(xd, theiler=theiler)
        out["RQA_DET_Dither"] = rq.get("RQA_DET", np.nan)
        out["RQA_LAM_Dither"] = rq.get("RQA_LAM", np.nan)
        # MSE under dither.
        try:
            _s, _v = rcmse(xd, m=ENTROPY_M, r_frac=ENTROPY_R_FRAC,
                           max_scale=MSE_MAX_SCALE, min_points=MSE_MIN_POINTS)
            _v = np.asarray(_v, dtype=float)
            out["MSE_Mean_Dither"] = (float(np.nanmean(_v))
                                      if np.isfinite(_v).any() else np.nan)
        except Exception:
            out["MSE_Mean_Dither"] = np.nan

        # -- Dither, replicated: spread across independent noise realizations --
        _rep = {k: [] for k in ("SampEn", "ApEn", "PermEn", "RQA_DET",
                                "RQA_LAM", "MSE_Mean")}
        for _i in range(int(QUANT_DITHER_REPS)):
            try:
                _rng_i = np.random.default_rng(QUANT_SEED + 1000 + _i)
                _xi = add_dither(x, step=step, rng=_rng_i)
                _rep["SampEn"].append(sample_entropy(_xi))
                _rep["ApEn"].append(approximate_entropy(_xi))
                try:
                    _pe, _ = nk.entropy_permutation(_xi, dimension=3, delay=1)
                    _rep["PermEn"].append(float(_pe))
                except Exception:
                    pass
                _rqi = rqa_metrics(_xi, theiler=theiler)
                _rep["RQA_DET"].append(_rqi.get("RQA_DET", np.nan))
                _rep["RQA_LAM"].append(_rqi.get("RQA_LAM", np.nan))
                try:
                    _si, _vi = rcmse(_xi, m=ENTROPY_M, r_frac=ENTROPY_R_FRAC,
                                     max_scale=MSE_MAX_SCALE,
                                     min_points=MSE_MIN_POINTS)
                    _vi = np.asarray(_vi, dtype=float)
                    if np.isfinite(_vi).any():
                        _rep["MSE_Mean"].append(float(np.nanmean(_vi)))
                except Exception:
                    pass
            except Exception:
                continue
        for _k, _vals in _rep.items():
            _a = np.asarray([v for v in _vals if np.isfinite(v)], dtype=float)
            out[f"{_k}_Dither_RepMean"] = float(np.mean(_a)) if _a.size else np.nan
            out[f"{_k}_Dither_RepSD"]   = (float(np.std(_a, ddof=1))
                                           if _a.size > 1 else np.nan)
            out[f"{_k}_Dither_RepN"]    = int(_a.size)

        # Matched coarsening: choose a step giving a common step/RMSSD ratio.
        if target_ratio is None:
            target_ratio = QUANT_TARGET_RATIO
        if target_ratio is None:
            # Only reached if QUANT_TARGET_RATIO is None; mirrors its definition.
            target_ratio = 0.088
        # Tie rate of the series each condition analyzes.
        out["Ordinal_Tie_Rate_Dither"] = ordinal_tie_rate(xd)
        if rmssd > 0:
            step_matched = target_ratio * rmssd
            if step_matched > step:
                xm = coarsen(x, step_matched)
                out["Ordinal_Tie_Rate_MatchedQuant"] = ordinal_tie_rate(xm)
                out["SampEn_MatchedQuant"] = sample_entropy(xm)
                out["ApEn_MatchedQuant"]   = approximate_entropy(xm)
                try:
                    pe, _ = nk.entropy_permutation(xm, dimension=3, delay=1)
                    out["PermEn_MatchedQuant"] = float(pe)
                except Exception:
                    out["PermEn_MatchedQuant"] = np.nan
                rqm = rqa_metrics(xm, theiler=theiler)
                out["RQA_DET_MatchedQuant"] = rqm.get("RQA_DET", np.nan)
                out["RQA_LAM_MatchedQuant"] = rqm.get("RQA_LAM", np.nan)
                out["MSE_Mean_MatchedQuant"] = _mse_mean_safe(xm)
                out["Quant_Matched_Step_s"] = step_matched
                out["Quant_Matched_Applied"] = True
            else:
                # Already coarser than the target (the normal case for MIT-BIH, step/RMSSD about 0.31): observed
                # values are carried through unchanged, for every field.
                out["Ordinal_Tie_Rate_MatchedQuant"] = ordinal_tie_rate(x)
                out["SampEn_MatchedQuant"]  = sample_entropy(x)
                out["ApEn_MatchedQuant"]    = approximate_entropy(x)
                try:
                    pe, _ = nk.entropy_permutation(x, dimension=3, delay=1)
                    out["PermEn_MatchedQuant"] = float(pe)
                except Exception:
                    out["PermEn_MatchedQuant"] = np.nan
                rq0 = rqa_metrics(x, theiler=theiler)
                out["RQA_DET_MatchedQuant"] = rq0.get("RQA_DET", np.nan)
                out["RQA_LAM_MatchedQuant"] = rq0.get("RQA_LAM", np.nan)
                out["MSE_Mean_MatchedQuant"] = _mse_mean_safe(x)
                out["Quant_Matched_Step_s"] = step
                out["Quant_Matched_Applied"] = False
        # -- Sweep of alternative common targets --
        for tr in QUANT_TARGET_SWEEP:
            tag = f"MQ{int(round(tr * 100)):02d}"
            step_t = tr * rmssd if rmssd > 0 else np.nan
            if np.isfinite(step_t) and step_t > step:
                xt = coarsen(x, step_t)
                applied = True
            else:
                xt = x
                applied = False
            out[f"Ordinal_Tie_Rate_{tag}"] = ordinal_tie_rate(xt)
            out[f"SampEn_{tag}"] = sample_entropy(xt)
            out[f"ApEn_{tag}"]   = approximate_entropy(xt)
            try:
                pe, _ = nk.entropy_permutation(xt, dimension=3, delay=1)
                out[f"PermEn_{tag}"] = float(pe)
            except Exception:
                out[f"PermEn_{tag}"] = np.nan
            rqt = rqa_metrics(xt, theiler=theiler)
            out[f"RQA_DET_{tag}"] = rqt.get("RQA_DET", np.nan)
            out[f"RQA_LAM_{tag}"] = rqt.get("RQA_LAM", np.nan)
            out[f"MSE_Mean_{tag}"] = _mse_mean_safe(xt)
            out[f"Quant_{tag}_Applied"] = applied

    except Exception as e:
        print(f"    Quantization controls failed: {e}")
    return out


# ---- Command-line entry point ----
if __name__ == "__main__":
    _cli()
