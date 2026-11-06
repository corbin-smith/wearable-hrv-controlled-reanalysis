"""s6_tables_d.py: Supplementary Tables added on 25 October 2026: time of day and transmission gaps
(recording_times_and_gaps.py), and the GRAPH reporting checklist (Quintana, Alvares and Heathers, 2016)."""
import numpy as np
import pandas as pd

import s6_common as K
from s6_common import R, U, P3, INT, mw, partial, bh, lab, FACTS, SNUM, MNUM
from s6_tables import table, src, cmp_, close


# =========================================================================== time of day and transmission gaps
@table("gaps")
def t_gaps():
    f_rec = cmp_("Recording_Times_Gaps.csv")
    f_sum = cmp_("Recording_Times_Gaps_Summary.csv")
    f_sen = cmp_("Gap_Exclusion_Sensitivity.csv")
    rec = pd.read_csv(K.csvp(f_rec))
    sm = pd.read_csv(K.csvp(f_sum)).set_index("Row")
    sen = pd.read_csv(K.csvp(f_sen))
    assert len(rec) == 60 and len(sen) == 42
    gap = rec[rec.Gaps_Window > 0].copy()
    ids = gap.Record_ID.tolist()

    # independent check of the time-of-day test from the per-record clock times
    rr, pp, _, _ = mw(rec[rec.Group == "control"].Start_Hours, rec[rec.Group == "patient"].Start_Hours)
    tst = sm.loc["patients vs controls, window start"]
    assert close(rr, tst.r_rank_biserial, 1e-6) and close(pp, tst.p, 1e-6)

    # independent check of every row of the exclusion analysis from the per-subject results
    C = K.C.drop(index=[i for i in ids if i in K.C.index])
    P = K.P.drop(index=[i for i in ids if i in K.P.index])
    assert (len(C), len(P)) == (30 - int((gap.Group == "control").sum()), 30 - int((gap.Group == "patient").sum()))
    prim = sen[sen.Family == "primary"]
    res = [mw(C[m], P[m]) for m in prim.Metric]
    padj = bh([x[1] for x in res])
    for (_, rw), (r_, p_, _, _), pa in zip(prim.iterrows(), res, padj):
        assert close(r_, rw.r_without_gaps, 1e-6) and close(pa, rw.p_adj_without_gaps, 1e-6), rw.Metric
    hra = sen[sen.Family == "hr_adjusted"]
    G = np.r_[np.zeros(len(C)), np.ones(len(P))]
    HR = np.r_[K.num(C.HRV_MeanHR).values, K.num(P.HRV_MeanHR).values]
    res = [partial(np.r_[K.num(C[m]).values, K.num(P[m]).values], G, [HR]) for m in hra.Metric]
    padj = bh([x[1] for x in res])
    for (_, rw), (r_, p_), pa in zip(hra.iterrows(), res, padj):
        assert close(r_, rw.r_without_gaps, 1e-6) and close(pa, rw.p_adj_without_gaps, 1e-6), rw.Metric

    cal = sm.loc["random exclusion calibration"]
    thr = sm.loc["gap threshold"]
    n_c, n_p = len(C), len(P)
    rows = [{"section": "Time of day of the analysis window (clock time at the start and end of the 1800 s window)"},
            {"subheader": ["Group", "Median start", "Earliest start", "Latest start", "Latest end", ""]}]
    for g, name in (("control", "Controls, *n* = 30"), ("patient", "Patients, *n* = 30")):
        s = sm.loc[g]
        rows.append([name, s.Median_Start, s.Earliest_Start, s.Latest_Start, s.Latest_End, ""])
    rows.append(["Patients against controls", f"*r* = {R(tst.r_rank_biserial)}", f"*p* = {P3(tst.p)}", "", "", ""])
    rows.append({"section": "Recordings with transmission gaps inside the window"})
    rows.append({"subheader": ["Recording", "Group", "Gaps in the window", "Time lost in the window (s)",
                               "Longest gap in the window (s)", "Gaps in the whole recording"]})
    for _, g in gap.iterrows():
        rows.append([g.Record_ID.replace("_", " "), g.Group, INT(g.Gaps_Window), U(g.Missing_s_Window, 1),
                     U(g.Longest_Gap_Window_s, 1), INT(g.Gaps_File)])
    for fam, title, stat in (("primary", "Primary family", "rank-biserial *r*"),
                             ("hr_adjusted", "Heart-rate-adjusted family", "partial *r*, heart rate held constant")):
        rows.append({"section": f"{title} ({stat}) with and without the {len(ids)} recordings "
                                f"({n_c} controls and {n_p} patients remain)"})
        rows.append({"subheader": ["Measure", "Full sample", "Without the gap recordings", "*p* adj, full sample",
                                   "*p* adj, without", "Random exclusions changing the conclusion"]})
        for _, rw in sen[sen.Family == fam].iterrows():
            mark = " (changed)" if str(rw.Conclusion_changed) == "True" else ""
            rows.append([lab(rw.Metric) + mark, R(rw.r_full), R(rw.r_without_gaps), P3(rw.p_adj_full),
                         P3(rw.p_adj_without_gaps), f"{INT(100 * rw.Share_random_exclusions_changing)}%"])
    note = ("Time stamps are the phone clock time recorded with every interval in the HRV-ACC files (calendar dates "
            "were removed from the release); the window was mapped to clock time at its first and last beat. "
            "*r* for time of day is the rank-biserial correlation with *U* for controls, positive meaning later in "
            "patients. A gap is a step in clock time that exceeds the interval recorded after it by more than "
            f"{U(thr.Threshold_s, 0)} s. Away from gaps the excess never exceeded {U(thr.Max_NonGap_Excess_s, 2)} s, "
            f"and the smallest gap was {U(thr.Min_Gap_s, 2)} s, so any threshold between the two finds the same gaps. "
            "For every window, its clock span less its 1800 s of beat time equaled the time lost to the detected gaps "
            f"to within {U(thr.Max_Clock_vs_Gap_Residual_s, 1)} s. The pipeline rebuilds beat time from the intervals, "
            "so each gap is closed and the series joined across it. Families: the 21 primary measures (Mann-Whitney "
            "*U*, Benjamini-Hochberg over the family) and the 21 heart-rate-adjusted tests (partial Spearman, "
            "regress then rank, Benjamini-Hochberg over the family), recomputed without the recordings that have a "
            "gap in the window. A conclusion changed if significance after correction was gained or lost, or the "
            "sign reversed while significant. Random exclusions: the same analysis after dropping "
            f"{INT(cal.Dropped_Controls)} controls and {INT(cal.Dropped_Patients)} patient chosen at random, "
            f"{INT(cal.N)} draws (seed {INT(cal.Seed)}); the column gives the share of draws in which the measure's "
            f"conclusion changed. The random exclusions changed a median of {INT(cal.Random_Median_Conclusions_Changed)} "
            f"of the 42 conclusions, and {INT(100 * cal.Random_Share_At_Least_As_Many)}% of them changed at least "
            f"{INT(cal.Gap_Records_Conclusions_Changed)}, as many as dropping the gap recordings did. The MIT-BIH "
            "intervals come from continuous ECG and cannot contain transmission gaps.")
    FACTS["s.gap_thr"] = U(thr.Threshold_s, 0)
    FACTS["s.gap_jitter"] = U(thr.Max_NonGap_Excess_s, 2)
    FACTS["s.gap_min"] = U(thr.Min_Gap_s, 2)
    FACTS["s.gap_rand_med"] = INT(cal.Random_Median_Conclusions_Changed)
    FACTS["s.gap_rand_share"] = INT(100 * cal.Random_Share_At_Least_As_Many)
    FACTS["s.gap_ids"] = ", ".join(i.replace("_", " ") for i in ids)
    return dict(header=["Measure", "Full sample", "Without the gap recordings", "*p* adj, full sample",
                        "*p* adj, without", "Random exclusions changing the conclusion"],
                rows=rows, note=note, landscape=False, sources=src(f_rec, f_sum, f_sen))


# =========================================================================== GRAPH checklist
@table("graph")
def t_graph():
    F = FACTS
    def S(name):
        return f"Supplementary Table {SNUM[name]}"

    def M(*names):
        nums = sorted((MNUM[x] for x in names), key=lambda s: int(s[1:]))
        body = nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]
        return "Supplementary Methods " + body
    rows = [
        {"section": "1. Selection of participants"},
        ["1a", "Psychiatric group selection: recruitment details and illness assessment methods",
         "Section 2.1; main Table 1",
         f"Inpatients of the Psychiatric Department in Tarnowskie Góry with ICD-10 diagnoses of schizophrenia (F20, "
         f"*n* = {F['n_sz']}) or bipolar affective disorder (F31, *n* = {F['n_bp']}), as recorded in the dataset "
         "release. Recruitment and diagnostic assessment belong to the original study (Książek et al., 2023, 2024); "
         "PANSS scores are in the release and were not analyzed."],
        ["1b", "Control group selection: recruitment details and methods to rule out psychiatric illness",
         "Section 2.1; main Table 1",
         f"{F['n_ctl']} healthy volunteers recorded by the dataset team on the same device, of similar age and sex to "
         "the patients and not individually matched. Screening for psychiatric illness belongs to the original "
         "study and is not restated here (Książek et al., 2023, 2024). MIT-BIH: 18 adults without significant "
         "arrhythmia (Goldberger et al., 2000)."],
        ["1c", "Inclusion criteria (for example, absence of physical health conditions)",
         f"Section 2.2; {M('preprocessing')}",
         f"All 60 HRV-ACC recordings in the release and all {F['n_mit']} MIT-BIH records were analyzed; none met "
         "the quality exclusion rules (artifact rate above 5% or fewer than 240 clean intervals). Physical-health "
         "criteria are those of the original studies and are not available for individual subjects."],
        ["1d", "Disease characteristics: duration, severity, psychiatric comorbidities and medication status",
         f"Sections 2.1 and 4.6; {S('subgroups')}; main Table 5",
         "Illness duration is not in the release. Severity: PANSS scores are in the release and were not analyzed. "
         f"Comorbidity: a substance-use code (F19, F19.2 or F13.2) in {F['cohort.comorb']} patients, analyzed as a "
         "subgroup. Medication: every patient was medicated, and only quetiapine exposure is recorded "
         f"({F['cohort.quet']} of 30); other agents and doses were requested and could not be released. Quetiapine "
         "exposure was analyzed as a subgroup and as a covariate."],
        ["1e", "Demographics: age, gender distribution, physical activity level, alcohol intake and nicotine intake",
         "Sections 2.1, 2.4 and 4.6; main Table 1",
         "Age and sex of every subject. Physical activity during the window was measured by accelerometer "
         "(mobility and walking bouts) and controlled; habitual activity was not recorded. Alcohol and nicotine "
         "intake were not recorded, and smoking is discussed as a limitation."],
        {"section": "2. Interbeat interval collection"},
        ["2a", "Hardware and software: brand and electrode configuration",
         f"Sections 2.1 and 2.2; {M('protocol', 'preprocessing')}",
         "HRV-ACC: Polar H10 chest strap with built-in electrodes, which stores R-R intervals and three-axis "
         "acceleration; intervals analyzed as exported. MIT-BIH: ambulatory ECG sampled at 128 Hz, with R peaks "
         "detected by NeuroKit2 in the first signal channel."],
        ["2b", "Collection details: raw sampling rate, length of data collection, time of day, filtering, posture "
               "and instructions",
         f"Section 2.2; {M('protocol', 'preprocessing')}; {S('gaps')}",
         "Interval resolution measured from the data: 1.000 ms (HRV-ACC) and 7.8125 ms (MIT-BIH, 128 Hz). Window: "
         f"1800 s after a 600 s skip. Window start: {F['tod.c_lo']} to {F['tod.c_hi']} in controls and "
         f"{F['tod.p_lo']} to {F['tod.p_hi']} in patients, similar between groups; MIT-BIH window times in "
         f"{M('protocol')}. Seated rest for 5 minutes, then encouraged and unsupervised walks, so the window is a "
         "mixed-activity recording; movement was measured and controlled. No filtering beyond the artifact rules "
         "(item 3b)."],
        {"section": "3. Data analysis and cleaning"},
        ["3a", "Interbeat interval calculation and resampling methods",
         f"Sections 2.2 and 2.3; {M('protocol', 'preprocessing', 'spectral')}",
         "HRV-ACC intervals are the device's; beat times were rebuilt as their cumulative sum. MIT-BIH intervals "
         "are differences between successive R-peak times. Resampling only for spectral analysis: cubic spline "
         "at 4 Hz."],
        ["3b", "Artifact identification method (for example, algorithm, manual inspection)",
         f"Section 2.2; {M('preprocessing')}",
         "Algorithm in two passes: intervals outside 300 to 2000 ms, then intervals deviating by more than 20% "
         "from the median of a centered 11-interval window."],
        ["3c", "Reasons for data loss (for example, persistent ectopy, equipment failure)",
         f"Section 2.2; {M('preprocessing')}; {S('gaps')}",
         "No recording was lost. Within recordings, intervals were lost to the artifact rules (item 3d) and to "
         f"transmission gaps where the Bluetooth link dropped, found in {F['gap.n_rec']} HRV-ACC recordings "
         f"({F['gap.min_s']} s to {F['gap.max_min']} min inside the window); results without those recordings are "
         "reported."],
        ["3d", "Artifact cleaning methods and the percentage of beats corrected",
         f"Section 2.2; {M('preprocessing')}",
         f"Flagged intervals were deleted, not interpolated or corrected. Mean share removed: {F['qc.art_c']}% in "
         f"controls, {F['qc.art_p']}% in patients and {F['qc.art_m']}% in MIT-BIH records; at most "
         f"{F['qc.art_max']}% in any record."],
        {"section": "4. HRV calculation"},
        ["4a", "Metrics used, software or script used for HRV calculation, and log transformation",
         f"Sections 2.3 and 2.6; {S('measures')}; {S('environment')}",
         "21 primary measures and the secondary measures, each defined with its parameters; a Python pipeline "
         "(pipeline.py) using NumPy, SciPy and NeuroKit2, with versions recorded and the code released. Group tests "
         "are rank-based, so no log transformation was needed for them; log forms appear only in the "
         "heart-rate model-form sensitivity."],
        ["4b", "Specification of frequency bands and how they were interpreted",
         f"Sections 2.3 and 4.4; {M('spectral')}",
         "Task Force limits (Task Force, 1996): VLF 0.003 to 0.04 Hz (integrated from 0.0078 Hz at this "
         "resolution), LF 0.04 to 0.15 Hz and HF 0.15 to 0.40 Hz. HF power and RMSSD are described as vagally "
         "mediated; the LF/HF ratio is not read as sympathovagal balance and is treated as exploratory where it "
         "appears after adjustment; VLF power is not attributed to a single mechanism."],
    ]
    note = ("Items and wording follow the GRAPH checklist (Quintana et al., 2016), shortened. Section numbers refer "
            "to the main text. This is a secondary analysis of openly available recordings, so items about "
            "recruitment and screening are answered from the dataset release and the original study's reports, "
            "and what the release does not contain is stated as such.")
    return dict(header=["Item", "GRAPH recommendation", "Reported in", "This study"], rows=rows, note=note,
                landscape=True, sources=src(cmp_("Recording_Times_Gaps_Summary.csv")))
