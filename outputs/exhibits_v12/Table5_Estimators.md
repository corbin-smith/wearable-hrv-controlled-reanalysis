**Table 5. Estimator choices and how far each moves a group difference.**

| Choice | Measure | Group difference (r) at each setting | Change from the first setting [95% CI] |
| --- | --- | --- | --- |
| DFA α₁ lower bound | DFA α₁ | 4 to 16 beats: −0.120; 8 to 16 beats: +0.360; 10 to 16 beats: +0.542 | 8 to 16 beats: +0.480 [+0.271, +0.691]; 10 to 16 beats: +0.662 [+0.409, +0.913] |
| Embedding | Determinism | dimension 2, delay 1 for all: +0.298; each subject's own embedding: −0.024 | each subject's own embedding: −0.322 [−0.509, −0.140] |
| Embedding | Laminarity | dimension 2, delay 1 for all: +0.224; each subject's own embedding: −0.024 | each subject's own embedding: −0.249 [−0.418, −0.087] |
| Minimum line length | Determinism | minimum line length 2: +0.298; minimum line length 3: +0.380; minimum line length 4: +0.427 | minimum line length 3: +0.082 [+0.013, +0.178]; minimum line length 4: +0.129 [+0.031, +0.260] |
| Surrogates | Sample entropy | observed value: −0.340; mean of the subject's surrogates: −0.340; observed minus surrogate mean: +0.138; surrogate Z-score: +0.222 | mean of the subject's surrogates: +0.000 [−0.147, +0.158]; observed minus surrogate mean: +0.478 [+0.049, +0.884]; surrogate Z-score: +0.562 [+0.127, +0.978] |
| Surrogates | Multiscale entropy | observed value: −0.493; mean of the subject's surrogates: −0.400; observed minus surrogate mean: −0.256; surrogate Z-score: −0.173 | mean of the subject's surrogates: +0.093 [+0.000, +0.198]; observed minus surrogate mean: +0.238 [−0.193, +0.627]; surrogate Z-score: +0.320 [−0.151, +0.753] |
| Slope estimator | Spectral slope | linear-bin fit: +0.667; log-binned fit: +0.662; 0.07–0.13 Hz excised: +0.638; aperiodic + peak model: +0.578; IRASA: +0.664; peak model, clean fits only: +0.465 | not bootstrapped; range +0.465 to +0.667 |
| Entropy tolerance | Sample entropy | 0.10 SD: −0.309; 0.15 SD: −0.344; 0.20 SD: −0.340; 0.25 SD: −0.362; 0.30 SD: −0.378; 0.35 SD: −0.400 | not bootstrapped; range −0.400 to −0.309 |

Rank-biserial r, positive = higher in patients. Change: r at the named setting minus r at the first setting, with a percentile bootstrap interval (subjects resampled within group, 2,000 resamples); an interval that excludes zero means the choice moved the group difference. Nothing here is corrected for multiplicity. Surrogates are iteratively amplitude-adjusted Fourier transform surrogates of each subject's own series; their mean shows what the amplitude distribution and linear autocorrelation alone give. The peak model's clean fits exclude fits that reached a parameter bound. The dependence of the slope on the fitted frequency range is shown in Figure 2B.
