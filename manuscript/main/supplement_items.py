"""
supplement_items.py: the single list of Supplementary Material items, shared by the manuscript
builder (build_v12.py) and the supplement builder (build_s6.py).

Supplementary Tables and Figures are numbered by their first citation in the main text, so the
numbers are not written here. Supplementary Methods and Results sections are numbered in the order
listed below, which is the order in which they appear in the Supplementary Material.
"""

# name -> table title (sentence case, no final period)
TABLES = {
    "skip": "Onset-skip sensitivity: group comparisons at skips of 60, 360 and 600 s before the 1800 s window",
    "gaps": "Time of day of the analysis windows, transmission gaps, and the results without the recordings that contain gaps",
    "graph": "GRAPH reporting checklist",
    "measures": "Definitions and parameters of every measure",
    "mse": "Multiscale entropy: retained scales and the mean over a common scale set",
    "stationarity": "Stationarity diagnostics",
    "quant": "Quantization controls at the recorded 1.000 ms resolution",
    "accel": "Accelerometer processing and walking-bout detection settings",
    "families": "Correction families and the effective number of tests",
    "primaryx": "The 21 primary measures: raw and adjusted p, parametric tests and distribution diagnostics",
    "power": "Smallest effects detectable with 80% power",
    "classifier": "Classification summary",
    "validation": "Estimator validation on synthetic fractional Gaussian noise",
    "environment": "Test implementation and software environment",
    "nonredundant": "Primary comparison without the definitional duplicates",
    "intercorr": "Spearman correlations among the primary measures within each group",
    "clinical": "Floor effects and conventional clinical ranges",
    "rna": "Recurrence-network measures",
    "hr": "Heart-rate adjustment: model forms, beat-count adjustment and normalization by mean NN interval",
    "fixedbeats": "Series length equalized by design: first 2,000 and 2,400 clean intervals",
    "requant": "Re-quantization to the 7.8125 ms grid of a 128 Hz recording",
    "scaleband": "DFA scale ranges: frequencies covered and alternative scale definitions",
    "controls": "The two control cohorts compared directly",
    "embedding": "Recurrence quantification at the fixed and the per-subject embedding",
    "surrogate": "Surrogate tests",
    "estimators": "Spectral slope under alternative estimators, and the baroreflex peak",
    "tolerance": "Entropy parameters: sample entropy tolerance and permutation entropy order",
    "vlf": "Decomposition of spectral power into VLF, LF and HF bands",
    "bands": "Spectral slope across fitted frequency ranges",
    "identity": "Spectral slope against the relation β = 2α − 1 with the DFA exponents",
    "mfdfa": "Multifractal DFA",
    "covariates": "Spectral slopes and VLF/LF ratio under covariate adjustment",
    "activity": "Accelerometer mobility and walking-bout structure",
    "movementfree": "Movement-free reanalysis: 600 beats (primary) and 370 s (underpowered)",
    "agesex": "Age and sex",
    "subgroups": "Diagnostic subgroups, quetiapine exposure and substance-use codes",
    "crosscohort": "Patients compared with the MIT-BIH controls, and the age-adjusted cross-cohort contrasts",
    "threegroup": "Three-group omnibus tests and post hoc comparisons",
}

# name -> figure title
FIGURES = {
    "effects": "Effect sizes for every measure",
    "boxplots": "Group distributions for eight measures",
    "intercorr": "Correlations among the primary measures within each group",
    "mse": "Multiscale entropy by coarse-graining scale",
    "surrogates": "Surrogate tests per subject",
    "mayer": "Group-mean spectra and the baroreflex peak",
    "identity": "Observed uncorrected slope against the slope predicted from DFA α₂",
    "covariate": "Uncorrected slope under covariate adjustment",
    "classifier": "Nested cross-validation classifier",
}

# name -> heading, in the order they appear
METHODS = {
    "protocol": "Recording protocol, analysis window and onset check",
    "preprocessing": "Artifact filtering, quality control and the resolution lattice",
    "spectral": "Spectral estimation, band powers and slope estimators",
    "fractal": "Detrended fluctuation analysis and multifractal DFA",
    "entropy": "Entropy measures",
    "recurrence": "Recurrence quantification, recurrence networks and embedding selection",
    "quantization": "Quantization controls and the quantization-noise estimate",
    "surrogates": "Surrogate testing",
    "movement": "Accelerometer processing, walking bouts and movement-free segments",
    "stationarity": "Stationarity diagnostics",
    "respiration": "Respiratory peak detection",
    "statistics": "Statistical procedures",
    "classifier": "Classification procedure",
    "software": "Software, random seeds and estimator validation",
}

RESULTS = {
    "respiration": "Respiratory frequency",
    "covariance": "Covariance structure of the measures",
    "activity": "Activity during the recording window",
    "embedding": "Embedding selection",
}


def methods_number(name):
    return list(METHODS).index(name) + 1


def results_number(name):
    return list(RESULTS).index(name) + 1
