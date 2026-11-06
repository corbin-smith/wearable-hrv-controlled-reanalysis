# Figure captions, v12 (draft; numbers filled from the rerun files)

**Figure 1. Heart rate and the group differences.**
(A) Beats in the analyzed 30-minute window against mean heart rate for 60 HRV-ACC recordings
(Pearson r = 0.9995). The line is beats = 30 × heart rate: over a fixed window, beat count is heart rate.
(B) Each of 21 measures' Spearman partial correlation with group before (open grey circles) and after (dark circles)
holding mean heart rate constant (regress-then-rank). Filled dark circles remain significant after
Benjamini–Hochberg correction across the 21 adjusted tests (15 of 21).
Amplitude measures keep 69–82% of their association
(RMSSD −0.629 to −0.469).
Sample and approximate entropy and the four recurrence measures fall to between −0.107 and +0.006
and are not significant after correction.
Source: HR_Adjusted_Comparison.csv.

**Figure 3. Recording resolution.**
(A) RMSSD expressed in steps of the 7.8125 ms grid used by the MIT-BIH recordings (RMSSD at 1.000 ms divided by 7.8125 ms).
Median 1.22 steps in patients and 2.87 in controls. Re-quantized to that grid, a median of
79% of three-beat patterns in patients would contain a tie (controls 41%).
Horizontal bars are medians; points are spread horizontally only to avoid overlap.
(B) Rank-biserial r for each measure at the recorded 1.000 ms resolution (open grey circles) and after re-quantizing the
same intervals to the 7.8125 ms grid (dark circles). For the six amplitude measures, the two ratios and both slopes, r changes
by at most 0.078 (LF/HF ratio). Sample entropy (−0.340 to −0.049) and approximate entropy (−0.387 to +0.018) move to
near zero, permutation entropy changes sign (+0.400 to −0.462), and DFA α₁ moves from −0.120 to −0.447.
Full-range DFA α was not recomputed on the grid. p values are in Table 3.
Source: Requantization_Q128.csv and Requantization_Resolution_Floor.csv.

**Figure 2. Power spectra and the frequency range of the slope fit.**
(A) R-R interval power spectra for each recording (thin lines) and the group means (thick lines), in ms²/Hz on
logarithmic axes. Shading marks the Welch bins used for the slope fit (0.016–0.148 Hz; bins are 0.0078 Hz apart);
hatching marks the 0.07–0.13 Hz band removed for the baroreflex-excised slope. Dashed lines have each group's
median uncorrected slope (controls 1.04, patients 1.81) and are a visual guide only.
(B) The uncorrected slope fitted over four ranges, mean ± SD, with the rank-biserial r for each (positive = steeper in
patients): 0.016–0.148 Hz +0.667; 0.008–0.148 Hz +0.651; 0.016–0.148 Hz with linear
interpolation +0.658; full spectrum 0.008–0.398 Hz −0.127.
Source: per-subject spectra recomputed by figures.py; PSD_Beta_Range_Sensitivity.csv.

**Figure 4. The MIT-BIH reference cohort.**
Each subject's baroreflex-excised spectral slope (A) and RMSSD (B) in 18 MIT-BIH recordings, 30 HRV-ACC controls
and 30 HRV-ACC patients. Bars are medians (slope 1.21, 1.24 and
1.73; RMSSD 29.9, 22.4 and 9.5 ms); points are
spread horizontally only to avoid overlap. With age held constant, the slope differs between MIT-BIH controls and
patients (partial r = +0.513, p < 0.001) but not between the two
control cohorts (partial r = +0.038, p = 0.800);
positive r means higher in the HRV-ACC group.
MIT-BIH intervals are on a 7.8125 ms grid, and the recordings differ from HRV-ACC in device, setting and age.
Source: ecg_results_*.csv; CrossDevice_AgeAdjusted.csv.
