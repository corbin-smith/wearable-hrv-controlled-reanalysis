"""
Build the main Figures 1 and 3, Tables 1 to 6 and draft captions for manuscript v12 from the pipeline outputs.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from scipy.stats import mannwhitneyu                 # noqa: E402

OUT = os.path.join("outputs", "exhibits_v12")
CMP = os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych")
P = {
    "controls":     os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "patients":     os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv"),
    "mitbih":       os.path.join("outputs", "healthy", "ecg_results_Healthy.csv"),
    "gcs":          os.path.join(CMP, "group_comparison_stats.csv"),
    "hr":           os.path.join(CMP, "HR_Adjusted_Comparison.csv"),
    "hrboot":       os.path.join(CMP, "HR_Adjusted_Bootstrap.csv"),
    "beattime":     os.path.join(CMP, "Requantization_BeatTime.csv"),
    "changes":      os.path.join(CMP, "Estimator_Paired_Changes.csv"),
    "rc_mf":        os.path.join("outputs", "recording_conditions", "Movement_Free_HR_Adjusted.csv"),
    "rc_long":      os.path.join("outputs", "recording_conditions", "Long_Window_Spectral.csv"),
    "q128":         os.path.join(CMP, "Requantization_Q128.csv"),
    "floor":        os.path.join(CMP, "Requantization_Resolution_Floor.csv"),
    "vlf":          os.path.join(CMP, "VLF_Decomposition.csv"),
    "agesex":       os.path.join(CMP, "AgeSex_Adjusted_Comparison.csv"),
    "diag":         os.path.join(CMP, "Diagnosis_Subgroup.csv"),
    "comorb":       os.path.join(CMP, "Comorbidity_Sensitivity.csv"),
    "quet":         os.path.join(CMP, "Quetiapine_Analysis.csv"),
    "joint":        os.path.join(CMP, "Length_Respiration_Checks.csv"),
    "demog":        os.path.join(CMP, "Demographic_Matching.csv"),
    "cohort":       os.path.join(CMP, "Patient_Cohort_Description.csv"),
    "embed":        os.path.join(CMP, "Embedding_Sensitivity.csv"),
    "surrnorm":     os.path.join(CMP, "Surrogate_Normalized_Entropy.csv"),
    "tolerance":    os.path.join(CMP, "Entropy_Tolerance_Sensitivity.csv"),
    "mayer":        os.path.join(CMP, "Mayer_Wave_Control.csv"),
    "ranges":       os.path.join(CMP, "PSD_Beta_Range_Sensitivity.csv"),
    "activity":     os.path.join("outputs", "accelerometer", "Beta_Activity_Covariate.csv"),
    "crossdevice":  os.path.join("outputs", "comparison", "cross_control", "CrossDevice_AgeAdjusted.csv"),
    "skip":         os.path.join("outputs", "comparison", "Window_Skip_Sensitivity.csv"),
    "stationary":   os.path.join("outputs", "stationary", "Stationary_Reanalysis_beats.csv"),
}

FAMILIES = [
    ("Amplitude",             ["HRV_RMSSD", "HRV_SDNN", "LF_Power_ms2", "HF_Power_ms2",
                               "VLF_Power_ms2", "Total_Power_ms2"]),
    ("Spectral distribution", ["VLF_LF_Ratio", "LF_HF_Ratio", "PSD_Beta", "PSD_Beta_NoMayer"]),
    ("Fractal scaling (DFA)", ["DFA_Alpha", "DFA_Alpha1", "DFA_Alpha2"]),
    ("Entropy",               ["MSE_Mean", "SampEn", "ApEn", "PermEn"]),
    ("Recurrence",            ["RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"]),
]
LABEL = {
    "HRV_RMSSD": "RMSSD", "HRV_SDNN": "SDNN", "LF_Power_ms2": "LF power",
    "HF_Power_ms2": "HF power", "VLF_Power_ms2": "VLF power", "Total_Power_ms2": "Total power",
    "VLF_LF_Ratio": "VLF/LF ratio", "LF_HF_Ratio": "LF/HF ratio",
    "PSD_Beta": "Slope, uncorrected", "PSD_Beta_NoMayer": "Slope, 0.07–0.13 Hz excised",
    "DFA_Alpha": "DFA α, full range", "DFA_Alpha1": "DFA α₁", "DFA_Alpha2": "DFA α₂",
    "MSE_Mean": "Multiscale entropy", "SampEn": "Sample entropy", "ApEn": "Approximate entropy",
    "PermEn": "Permutation entropy", "RQA_DET": "Determinism", "RQA_LAM": "Laminarity",
    "RNA_ClusteringCoeff": "Network clustering", "RNA_Transitivity": "Network transitivity",
}
GCS_LABEL = {
    "HRV_MeanNN": "Mean NN (ms)", "HRV_SDNN": "SDNN (ms)", "HRV_RMSSD": "RMSSD (ms)",
    "HRV_pNN50": "pNN50 (%)", "HRV_MeanHR": "Mean HR (bpm)", "HRV_SD1": "SD1 (ms)",
    "HRV_SD2": "SD2 (ms)", "HRV_SD1SD2": "SD1/SD2", "LF_Power_ms2": "LF Power (ms²)",
    "HF_Power_ms2": "HF Power (ms²)", "Total_Power_ms2": "Total Power (ms²)",
    "LF_HF_Ratio": "LF/HF Ratio", "PSD_Beta": "PSD β", "SampEn": "Sample Entropy",
    "PermEn": "Permutation Entropy", "MSE_Mean": "MSE Mean", "DFA_Alpha": "DFA α (full)",
    "DFA_Alpha1": "DFA α₁ (4–16)", "DFA_Alpha2": "DFA α₂ (16–64)",
    "RQA_DET": "RQA Determinism", "RQA_LAM": "RQA Laminarity",
}

C_CONTROL, C_PATIENT = "#0077BB", "#CC3311"
INK, INK_2, INK_3 = "#1F1F1F", "#5A5A5A", "#8C8C8C"
HAIR, GRID = "#D9D9D9", "#EEEEEE"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.edgecolor": INK_2,
    "savefig.dpi": 300,
})
FIG_WIDTH_IN = 7.08


class InputError(RuntimeError):
    pass


def need(path):
    if not os.path.exists(path):
        raise InputError(f"missing input: {path}. Run the v12 run sheet first.")
    return path


def read(key):
    return pd.read_csv(need(P[key]))


def cohort(key):
    d = pd.read_csv(need(P[key]))
    d = d.rename(columns={d.columns[0]: "Record_ID"})
    d["Record_ID"] = d["Record_ID"].astype(str)
    d = d[~d["Record_ID"].str.contains(r"_(?:Mean|Median|SD)$", regex=True)]
    return d.set_index("Record_ID")


def rank_biserial(controls, patients):
    a = pd.to_numeric(pd.Series(controls), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(patients), errors="coerce").dropna().values
    u = mannwhitneyu(a, b, alternative="two-sided")
    return 1.0 - 2.0 * u.statistic / (len(a) * len(b)), float(u.pvalue), len(a), len(b)


def fmt_r(r):
    return "n/a" if r is None or not np.isfinite(r) else f"{r:+.3f}".replace("-", "−")


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_is(p):
    t = fmt_p(p)
    return "p " + t.replace("<", "< ") if t.startswith("<") else f"p = {t}"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(df, name, title, notes):
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8-sig")
    shown = df[[c for c in df.columns if c not in ("Source", "Column")]]
    lines = [f"**{title}**", "",
             "| " + " | ".join(shown.columns) + " |",
             "| " + " | ".join("---" for _ in shown.columns) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    lines += ["", notes, ""]
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def save(fig, name):
    for ext in ("svg", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=300, **kw)
    plt.close(fig)
    from PIL import Image
    with Image.open(os.path.join(OUT, f"{name}.tif")) as im:
        w_mm = im.size[0] / 300 * 25.4
        print(f"    {name}.tif: {im.size[0]} x {im.size[1]} px = "
              f"{w_mm:.0f} x {im.size[1] / 300 * 25.4:.0f} mm at 300 dpi")
        if w_mm > 180.5:
            print(f"    !! wider than 180 mm; at 180 mm the 8 pt text would print below 8 pt")


def load_and_check():
    D = {k: None for k in P}
    D["controls"], D["patients"], D["mitbih"] = cohort("controls"), cohort("patients"), cohort("mitbih")
    for k in P:
        if k not in ("controls", "patients", "mitbih"):
            D[k] = read(k)
    H, Pt, M = D["controls"], D["patients"], D["mitbih"]
    D["beattime_ps"] = pd.read_csv(need(os.path.join(CMP, "Requantization_BeatTime_PerSubject.csv")))
    checks = []

    def ok(cond, msg):
        if not cond:
            raise InputError("self-test failed: " + msg)
        checks.append(msg)

    ok(len(H) == 30 and len(Pt) == 30 and len(M) == 18,
       f"cohorts are 30 controls, 30 patients, 18 MIT-BIH (got {len(H)}, {len(Pt)}, {len(M)})")
    hr = D["hr"].set_index("Metric")
    fam = [m for _, ms in FAMILIES for m in ms]
    ok(set(fam) == set(hr.index) and len(fam) == 21,
       "the 21 plotted measures are exactly the hr_adjusted correction family")
    ok(set(hr["FDR_Family"]) == {"hr_adjusted"} and int(hr["FDR_Family_N"].iloc[0]) == 21,
       "HR_Adjusted_Comparison.csv is the 21-test hr_adjusted family")
    q = D["q128"].set_index("Metric")
    ok(set(fam) - {"DFA_Alpha"} <= set(q.index),
       "Requantization_Q128.csv holds every family measure except full-range DFA alpha")
    gcs = D["gcs"].rename(columns={D["gcs"].columns[0]: "Label"}).set_index("Label")
    ok(set(GCS_LABEL.values()) == set(gcs.index) and len(gcs) == 21
       and set(GCS_LABEL) <= set(H.columns) & set(Pt.columns),
       "group_comparison_stats.csv holds the 21 primary measures, each with a per-subject column")
    worst = 0.0
    for col, lab in GCS_LABEL.items():
        r, _, _, _ = rank_biserial(H[col], Pt[col])
        worst = max(worst, abs(r - float(gcs.loc[lab, "r_effect"])))
    ok(worst < 1e-6, f"local rank-biserial reproduces all 21 primary r values (max diff {worst:.1e})")
    worst = max(abs(rank_biserial(H[m], Pt[m])[0] - float(hr.loc[m, "r_rank_biserial"])) for m in fam)
    ok(worst < 1e-6, f"local rank-biserial reproduces all 21 heart-rate-family r values (max diff {worst:.1e})")
    ok({"HRV_MeanHR", "HRV_RMSSD", "SampEn", "Requant_RMSSD_In_Steps"} <= set(H.columns) & set(Pt.columns),
       "per-subject columns needed for Figure 1A, 1B and Figure 3A are present")
    hb = D["hrboot"].set_index("Metric")
    ok(set(hb.index) == set(fam)
       and float((hb["r_partial_HR"] - hr.loc[hb.index, "r_partial_HR"]).abs().max()) < 1e-6
       and float((hb["r_partial_unadjusted"] - hr.loc[hb.index, "r_partial_unadjusted"]).abs().max()) < 1e-6,
       "HR_Adjusted_Bootstrap.csv was made from this HR_Adjusted_Comparison.csv (point estimates agree)")
    bt = D["beattime"].set_index("Metric")
    ok(set(fam) - {"DFA_Alpha"} == set(bt.index),
       "Requantization_BeatTime.csv holds every family measure except full-range DFA alpha")
    ok(float((bt["r_recorded"] - hr.loc[bt.index, "r_rank_biserial"]).abs().max()) < 1e-6
       and float((bt["r_interval_rounding"] - q.loc[bt.index, "r_Q128"]).abs().max()) < 1e-6,
       "the beat-time table's recorded and interval-rounding r equal the comparison tables")
    st = D["stationary"]
    ok(set(st["Group"]) == {"control", "patient"} and set(st["Match_Mode"]) == {"beats"}
       and st["N_Beats"].between(590, 600).all(),
       f"the movement-free file is the 600-beat run ({(st.Group == 'control').sum()} controls, "
       f"{(st.Group == 'patient').sum()} patients; segments of {st.N_Beats.min()}–{st.N_Beats.max()} beats)")
    dm = D["demog"].set_index("Variable")
    fem = lambda d: int((d["Sex"].astype(str).str.upper().str[0] == "F").sum())
    ok(fem(H) == int(dm.loc["Sex (N female)", "Control"]) and fem(Pt) == int(dm.loc["Sex (N female)", "Patient"]),
       "female counts in the cohort files equal Demographic_Matching.csv")
    return D, checks


def _rows():
    rows, spans, y = [], [], 0.0
    for fam, ms in FAMILIES:
        start = y
        for m in ms:
            rows.append((m, fam, y))
            y += 1.0
        spans.append((fam, start, y - 1.0))
        y += 0.9
    return rows, spans


def _dumbbell(ax, rows, spans, before, after, filled_after, xlab, xlim=(-0.9, 0.9), ci=None, extra=None):
    ys = [y for _, _, y in rows]
    ax.set_ylim(max(ys) + 0.8, -1.6)
    ax.set_xlim(*xlim)
    for x in np.arange(-0.8, 0.81, 0.2):
        ax.axvline(x, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=INK_3, lw=0.8, zorder=1)
    for fam, y0, _ in spans[1:]:
        ax.axhline(y0 - 0.95, color=HAIR, lw=0.6, zorder=1)
    for (m, fam, y) in rows:
        b, a = before[m], after[m]
        if ci is not None and m in ci and np.all(np.isfinite(ci[m])):
            lo, hi = ci[m]
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [y, y], color=INK, lw=0.9,
                    solid_capstyle="butt", zorder=2.5)
        elif np.isfinite(b) and np.isfinite(a):
            ax.plot([b, a], [y, y], color="#C4C4C4", lw=1.6, solid_capstyle="round", zorder=2)
        if extra is not None and m in extra and np.isfinite(extra[m]):
            ax.scatter([extra[m]], [y], s=15, marker="D", facecolor="#B0B0B0", edgecolor="none", zorder=3.2)
        if np.isfinite(b):
            ax.scatter([b], [y], s=46, facecolor="white", edgecolor=INK_3, linewidth=1.0, zorder=3)
        if np.isfinite(a):
            filled = bool(filled_after.get(m, True))
            ax.scatter([a], [y], s=20, facecolor=(INK if filled else "white"),
                       edgecolor=INK, linewidth=1.1, zorder=4)
    ax.set_yticks([y for _, _, y in rows])
    ax.set_yticklabels([LABEL[m] for m, _, _ in rows])
    for fam, y0, _ in spans:
        ax.text(-0.02, y0 - 0.62, fam, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8, fontweight="bold", color=INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_xticks(np.arange(-0.8, 0.81, 0.4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}".replace("+0.0", "0").replace("-", "−")))

_L = dict(margin=0.62, a_w=1.95, gap=0.30, labels=1.95, right=0.24, top=0.42)


def _layout(fig_h, a_h, key_h):
    W = FIG_WIDTH_IN
    fig = plt.figure(figsize=(W, fig_h))
    a_bottom = fig_h - _L["top"] - a_h
    axA = fig.add_axes([_L["margin"] / W, a_bottom / fig_h, _L["a_w"] / W, a_h / fig_h])
    axK = fig.add_axes([_L["margin"] / W, 0.30 / fig_h, (_L["a_w"] + 0.3) / W, key_h / fig_h])
    axK.axis("off")
    b_left = _L["margin"] + _L["a_w"] + _L["gap"] + _L["labels"]
    b_bottom = 0.62
    axB = fig.add_axes([b_left / W, b_bottom / fig_h, (W - b_left - _L["right"]) / W,
                        (fig_h - _L["top"] - b_bottom) / fig_h])
    return fig, axA, axK, axB


def _panel_title(fig, x_in, letter, title):
    W, Hh = fig.get_size_inches()
    y = 1 - 0.14 / Hh
    fig.text(x_in / W, y, letter, fontsize=10, fontweight="bold", ha="left", va="top", color=INK)
    fig.text((x_in + 0.20) / W, y - 0.012 / Hh, title, fontsize=8.5, ha="left", va="top", color=INK)


def _titles(fig, title_a, title_b):
    _panel_title(fig, 0.02, "A", title_a)
    _panel_title(fig, _L["margin"] + _L["a_w"] + _L["gap"], "B", title_b)


def _layout3(fig_h, a_h, key_h):
    W = FIG_WIDTH_IN
    fig = plt.figure(figsize=(W, fig_h))
    a_bottom = fig_h - _L["top"] - a_h
    b_bottom = a_bottom - 0.42 - a_h
    axA = fig.add_axes([_L["margin"] / W, a_bottom / fig_h, _L["a_w"] / W, a_h / fig_h])
    axB = fig.add_axes([_L["margin"] / W, b_bottom / fig_h, _L["a_w"] / W, a_h / fig_h])
    axK = fig.add_axes([_L["margin"] / W, 0.10 / fig_h, (_L["a_w"] + 0.3) / W, key_h / fig_h])
    axK.axis("off")
    c_left = _L["margin"] + _L["a_w"] + _L["gap"] + _L["labels"]
    c_bottom = 0.62
    axC = fig.add_axes([c_left / W, c_bottom / fig_h, (W - c_left - _L["right"]) / W,
                        (fig_h - _L["top"] - c_bottom) / fig_h])
    return fig, axA, axB, axK, axC


def figure1(D, src):
    H, Pt = D["controls"], D["patients"]
    hr = D["hr"].set_index("Metric")
    hb = D["hrboot"].set_index("Metric")

    fig_h, a_h = 6.5, 1.72
    fig, axA, axB, axK, axC = _layout3(fig_h, a_h, 1.35)

    rate_all = np.r_[H["HRV_MeanHR"].astype(float), Pt["HRV_MeanHR"].astype(float)]
    lo, hi = np.floor(rate_all.min() / 5) * 5 - 2, np.ceil(rate_all.max() / 5) * 5 + 2
    for ax, col, ylab, logy in ((axA, "HRV_RMSSD", "RMSSD (ms)", True), (axB, "SampEn", "Sample entropy", False)):
        ax.scatter(H["HRV_MeanHR"], H[col], s=22, marker="o", facecolor=C_CONTROL,
                   edgecolor="white", linewidth=0.6, zorder=3, label=f"Controls (n = {len(H)})")
        ax.scatter(Pt["HRV_MeanHR"], Pt[col], s=24, marker="^", facecolor=C_PATIENT,
                   edgecolor="white", linewidth=0.6, zorder=3, label=f"Patients (n = {len(Pt)})")
        if logy:
            ax.set_yscale("log")
            ax.set_yticks([3, 10, 30, 100])
            ax.set_yticklabels(["3", "10", "30", "100"])
            ax.minorticks_off()
        ax.set_xlim(lo, hi)
        if not logy:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 + 0.24 * (y1 - y0))
        ax.set_ylabel(ylab)
        ax.tick_params(axis="y", length=2.5)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        rc, rp = float(hr.loc[col, "rho_HR_within_controls"]), float(hr.loc[col, "rho_HR_within_patients"])
        ax.text(0.97, 0.96, f"ρ with heart rate\ncontrols {rc:+.2f}, patients {rp:+.2f}".replace("-", "−"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
        for df, grp in ((H, "control"), (Pt, "patient")):
            for rid, row in df.iterrows():
                src.append(dict(Figure="1A" if col == "HRV_RMSSD" else "1B", Measure=rid, Label=grp,
                                Statistic=f"per-subject mean heart rate and {col}",
                                Before=float(row["HRV_MeanHR"]), After=float(row[col]),
                                Source="ecg_results_*.csv"))
    axA.tick_params(axis="x", labelbottom=False)
    axB.set_xlabel("Mean heart rate (beats/min)")
    axA.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.1, fontsize=8)

    rows, spans = _rows()
    before = {m: float(hr.loc[m, "r_partial_unadjusted"]) for m, _, _ in rows}
    after = {m: float(hr.loc[m, "r_partial_HR"]) for m, _, _ in rows}
    sig = {m: str(hr.loc[m, "Significant"]) == "True" for m, _, _ in rows}
    ci = {m: (float(hb.loc[m, "HR_CI_low"]), float(hb.loc[m, "HR_CI_high"])) for m, _, _ in rows}
    _dumbbell(axC, rows, spans, before, after, sig,
              "Spearman correlation with group\n← lower in patients    higher in patients →", ci=ci)
    W = FIG_WIDTH_IN
    y_top = 1 - 0.14 / fig_h
    for letter, title, x_in, y in (("A", "RMSSD and heart rate", 0.02, y_top),
                                   ("B", "Sample entropy and heart rate", 0.02,
                                    (fig_h - _L["top"] - a_h - 0.42 + 0.26) / fig_h),
                                   ("C", "Group difference before and after heart-rate adjustment",
                                    _L["margin"] + _L["a_w"] + _L["gap"], y_top)):
        fig.text(x_in / W, y, letter, fontsize=10, fontweight="bold", ha="left", va="top", color=INK)
        fig.text((x_in + 0.20) / W, y - 0.012 / fig_h, title, fontsize=8.5, ha="left", va="top", color=INK)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=INK_3, mew=1.0,
               label="Unadjusted"),
        Line2D([], [], ls="-", lw=0.9, color=INK, marker="o", ms=4.5, mfc=INK, mec=INK, mew=1.1,
               label="Heart rate held constant, with\n95% bootstrap interval:\nsignificant after correction"),
        Line2D([], [], ls="-", lw=0.9, color=INK, marker="o", ms=4.5, mfc="white", mec=INK, mew=1.1,
               label="not significant"),
    ]
    axK.legend(handles=handles, loc="upper left", frameon=False, title="Panel C",
               title_fontproperties={"weight": "bold", "size": 8}, alignment="left",
               labelspacing=0.7, handletextpad=0.5, borderaxespad=0.0)
    save(fig, "Figure1_HeartRate")

    for m, fam, _ in rows:
        src.append(dict(Figure="1C", Family=fam, Measure=m, Label=LABEL[m],
                        Statistic="Spearman correlation with group; partial given mean heart rate",
                        Before=before[m], After=after[m], After_CI_low=ci[m][0], After_CI_high=ci[m][1],
                        p_raw_after=float(hr.loc[m, "p_raw"]), p_adj_after=float(hr.loc[m, "p_adj"]),
                        Significant_after=sig[m],
                        Source=os.path.basename(P["hr"]) + "; " + os.path.basename(P["hrboot"])))
    return dict(sig_count=sum(sig.values()), n=len(H) + len(Pt))


def figure3(D, src):
    H, Pt = D["controls"], D["patients"]
    bt = D["beattime"].set_index("Metric")
    fl = D["floor"].set_index("Quantity")

    fig, axA, axK, axB = _layout(5.6, 2.35, 1.5)

    rng = np.random.default_rng(20261018)
    for i, (df, col, mk) in enumerate([(H, C_CONTROL, "o"), (Pt, C_PATIENT, "^")]):
        v = df["Requant_RMSSD_In_Steps"].astype(float).dropna().values
        x = i + rng.uniform(-0.17, 0.17, size=len(v))
        axA.scatter(x, v, s=20, marker=mk, facecolor=col, edgecolor="white", linewidth=0.6, zorder=3)
        med = float(np.median(v))
        axA.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.6, solid_capstyle="round", zorder=4)
        axA.text(i + 0.31, med, f"median\n{med:.2f}", va="center", ha="left", fontsize=8, color=INK)
    axA.axhline(1.0, color=INK_3, lw=0.9, zorder=1)
    assert (H["Requant_RMSSD_In_Steps"].astype(float) > 1.05).all()
    axA.text(-0.45, 0.95, "1 grid step", va="top", ha="left", fontsize=8, color=INK_2)
    axA.set_xlim(-0.5, 1.75)
    axA.set_xticks([0, 1]); axA.set_xticklabels([f"Controls\n(n = {len(H)})", f"Patients\n(n = {len(Pt)})"])
    axA.set_ylim(0, None)
    axA.set_ylabel("RMSSD in 7.8 ms grid steps")
    axA.tick_params(axis="y", length=2.5)
    axA.grid(axis="y", color=GRID, lw=0.6); axA.set_axisbelow(True)
    med_h = float(fl.loc["Requant_RMSSD_In_Steps", "Healthy_Median"])
    med_p = float(fl.loc["Requant_RMSSD_In_Steps", "Psych_Median"])
    assert abs(med_h - np.median(H["Requant_RMSSD_In_Steps"])) < 1e-6
    assert abs(med_p - np.median(Pt["Requant_RMSSD_In_Steps"])) < 1e-6

    rows, spans = _rows()
    rows_q = [(m, f, y) for m, f, y in rows if m in bt.index]
    before = {m: float(bt.loc[m, "r_recorded"]) for m, _, _ in rows_q}
    after = {m: float(bt.loc[m, "r_beat_time"]) for m, _, _ in rows_q}
    ci = {m: (float(bt.loc[m, "r_beat_time_CI_low"]), float(bt.loc[m, "r_beat_time_CI_high"])) for m, _, _ in rows_q}
    extra = {m: float(bt.loc[m, "r_interval_rounding"]) for m, _, _ in rows_q}
    _dumbbell(axB, rows_q, [s for s in spans], before, after, {m: True for m in after},
              "Rank-biserial r\n← lower in patients    higher in patients →", ci=ci, extra=extra)
    ys = {m: y for m, _, y in rows}
    axB.set_yticks([y for _, _, y in rows])
    axB.set_yticklabels([LABEL[m] for m, _, _ in rows])
    axB.set_ylim(max(ys.values()) + 0.8, -1.6)
    axB.text(0.0, ys["DFA_Alpha"], "not recomputed on the grid", ha="center", va="center",
             fontsize=8, color=INK_3, style="italic", zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    _titles(fig, "RMSSD in units of the 7.8 ms grid",
            "Group difference at 1.000 ms and at 128 Hz")

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=INK_3, mew=1.0,
               label="1.000 ms (as recorded)"),
        Line2D([], [], ls="-", lw=0.9, color=INK, marker="o", ms=4.5, mfc=INK, mec=INK, mew=1.1,
               label="128 Hz: beat times rounded to\n7.8125 ms, with 95% bootstrap\ninterval"),
        Line2D([], [], ls="none", marker="D", ms=4.0, mfc="#B0B0B0", mec="none",
               label="Each interval rounded to\n7.8125 ms instead"),
    ]
    axK.legend(handles=handles, loc="upper left", frameon=False, title="Panel B",
               title_fontproperties={"weight": "bold", "size": 8}, alignment="left",
               labelspacing=0.9, handletextpad=0.5, borderaxespad=0.0)
    save(fig, "Figure3_Resolution")

    for m, fam, _ in rows_q:
        src.append(dict(Figure="3B", Family=fam, Measure=m, Label=LABEL[m],
                        Statistic="rank-biserial r", Before=before[m], After=after[m],
                        After_CI_low=ci[m][0], After_CI_high=ci[m][1], Interval_rounding=extra[m],
                        p_raw_before=float(bt.loc[m, "p_recorded"]), p_raw_after=float(bt.loc[m, "p_beat_time"]),
                        Source=os.path.basename(P["beattime"])))
    for df, grp in [(H, "control"), (Pt, "patient")]:
        for rid, v in df["Requant_RMSSD_In_Steps"].items():
            src.append(dict(Figure="3A", Measure=rid, Label=grp, Statistic="RMSSD / 7.8125 ms",
                            After=float(v), Source="ecg_results_*.csv"))
    return dict(med_h=med_h, med_p=med_p)


def table1(D):
    H, Pt, M = D["controls"], D["patients"], D["mitbih"]
    dm = D["demog"].set_index("Variable")
    pc = D["cohort"].set_index("Group")

    def age(df):
        a = df["Age"].astype(float)
        return f"{a.mean():.1f} ({a.std(ddof=1):.1f})", f"{a.min():.0f}–{a.max():.0f}"

    def female(df):
        s = df["Sex"].astype(str).str.upper().str[0]
        return f"{int((s == 'F').sum())} / {len(df)}"

    rows = []
    (ah, rh), (ap, rp), (am, rm) = age(H), age(Pt), age(M)
    rows.append(dict(Characteristic="Participants, n", Controls=len(H), Patients=len(Pt),
                     MIT_BIH=len(M), Comparison="", Source="ecg_results_*.csv"))
    rows.append(dict(Characteristic="Age, years, mean (SD)", Controls=ah, Patients=ap, MIT_BIH=am,
                     Comparison=f"r = {fmt_r(float(dm.loc['Age', 'r_effect']))}, {p_is(float(dm.loc['Age', 'p_value']))}",
                     Source="ecg_results_*.csv; Demographic_Matching.csv"))
    rows.append(dict(Characteristic="Age range, years", Controls=rh, Patients=rp, MIT_BIH=rm,
                     Comparison="", Source="ecg_results_*.csv"))
    rows.append(dict(Characteristic="Female, n / N", Controls=female(H), Patients=female(Pt),
                     MIT_BIH=female(M),
                     Comparison=f"Fisher {p_is(float(dm.loc['Sex (N female)', 'p_value']))}",
                     Source="ecg_results_*.csv; Demographic_Matching.csv"))
    for lab, key in [("Schizophrenia (F20), n", "Schizophrenia (F20)"), ("Bipolar disorder (F31), n", "Bipolar (F31)")]:
        rows.append(dict(Characteristic=lab, Controls="", Patients=int(pc.loc[key, "N"]),
                         MIT_BIH="", Comparison="", Source="Patient_Cohort_Description.csv"))
    rows.append(dict(Characteristic="Quetiapine recorded, n", Controls="",
                     Patients=int(pc.loc["All patients", "N_quetiapine"]), MIT_BIH="", Comparison="",
                     Source="Patient_Cohort_Description.csv"))
    rows.append(dict(Characteristic="Substance-use diagnosis (F1x), n", Controls="",
                     Patients=int(pc.loc["All patients", "N_comorbid_substance"]), MIT_BIH="",
                     Comparison="", Source="Patient_Cohort_Description.csv"))
    rows.append(dict(Characteristic="Recording", Controls="Polar H10 chest strap",
                     Patients="Polar H10 chest strap", MIT_BIH="Long-term ambulatory ECG",
                     Comparison="", Source="dataset documentation"))
    rows.append(dict(Characteristic="R-R interval resolution", Controls="1.000 ms",
                     Patients="1.000 ms", MIT_BIH="7.8125 ms (128 Hz)",
                     Comparison="", Source="measured step (decision 2b); NSRDB sampling rate"))
    rows.append(dict(Characteristic="Analyzed window", Controls="minutes 10–40",
                     Patients="minutes 10–40", MIT_BIH="minutes 10–40", Comparison="",
                     Source="pipeline.py SKIP_SECONDS, ANALYSIS_SECONDS, MITBIH_SKIP_SECONDS"))
    df = pd.DataFrame(rows).rename(columns={"MIT_BIH": "MIT-BIH reference"})
    write_table(df, "Table1_Participants", "Table 1. Participants.",
                "HRV-ACC controls and patients come from one dataset, recorded with the same device. "
                "Comparison columns: rank-biserial r for age (positive = older patients), Fisher's exact test "
                "for sex. Diagnoses, medication and substance-use codes are as recorded in the dataset's "
                "metadata (ICD-10 F20, F31 and F1x). The medication field records quetiapine only: patients "
                "without it were receiving other medication, which the dataset does not list. MIT-BIH recordings are used only as a second control "
                "group in the cross-device checks.")
    return df


def table2(D):
    g = D["gcs"].rename(columns={D["gcs"].columns[0]: "Label"})
    g = g[g["FDR_Family"] == "primary"].copy()
    assert len(g) == 21, "primary family is not 21 measures"
    g = g.iloc[np.argsort(-g["r_effect"].abs().values, kind="stable")]

    def pct(row):
        if not np.isfinite(row["Pct_Diff"]):
            return "n/a" if str(row["Pct_Diff_Basis"]).startswith("suppressed") else ""
        return f"{row['Pct_Diff']:+.1f}%".replace("-", "−")

    def num(v):
        a = abs(v)
        return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.3f}"

    out = pd.DataFrame({
        "Measure": g["Label"],
        "Controls, median (IQR)": [f"{num(m)} ({num(i)})" for m, i in zip(g["Healthy_Median"], g["Healthy_IQR"])],
        "Patients, median (IQR)": [f"{num(m)} ({num(i)})" for m, i in zip(g["Psych_Median"], g["Psych_IQR"])],
        "Median difference": [pct(r) for _, r in g.iterrows()],
        "r [95% CI]": [f"{fmt_r(r)} [{fmt_r(lo)}, {fmt_r(hi)}]"
                       for r, lo, hi in zip(g["r_effect"], g["r_CI_low"], g["r_CI_high"])],
        "p (adjusted)": [fmt_p(p) for p in g["p_adj"]],
        "Statistic": "rank-biserial r",
        "Source": "group_comparison_stats.csv",
    })
    n_sig = int((g["Significant"].astype(str) == "True").sum())
    write_table(out, "Table2_Primary", "Table 2. The 21 primary measures, unadjusted.",
                f"{n_sig} of 21 differ after Benjamini–Hochberg correction within the primary family. "
                "r is the rank-biserial correlation (positive = higher in patients) with a bootstrap 95% CI. "
                "The pNN50 percentage difference is not reported because pNN50 is floor-limited in patients. "
                "SD1 is RMSSD/√2, SD2 is determined by SDNN and SD1, and mean NN and mean heart rate are "
                "near-reciprocals, so these are not independent tests.")
    return out, n_sig


def _ci(v, lo, hi):
    return f"{fmt_r(v)} [{fmt_r(lo)}, {fmt_r(hi)}]"


def table3(D):
    hr = D["hr"].set_index("Metric")
    hb = D["hrboot"].set_index("Metric")
    rows = []
    for fam, ms in FAMILIES:
        for m in ms:
            h, b = hr.loc[m], hb.loc[m]
            rows.append({
                "Family": fam, "Measure": LABEL[m],
                "ρ with group, unadjusted [95% CI]":
                    _ci(float(h["r_partial_unadjusted"]), float(b["unadj_CI_low"]), float(b["unadj_CI_high"])),
                "Partial ρ, heart rate held constant [95% CI]":
                    _ci(float(h["r_partial_HR"]), float(b["HR_CI_low"]), float(b["HR_CI_high"])),
                "p, adjusted": fmt_p(float(h["p_adj"])),
                "Change with adjustment [95% CI]":
                    _ci(float(b["delta_HR_minus_unadj"]), float(b["delta_CI_low"]), float(b["delta_CI_high"])),
                "ρ with heart rate, controls": fmt_r(float(h["rho_HR_within_controls"])),
                "ρ with heart rate, patients": fmt_r(float(h["rho_HR_within_patients"])),
                "Source": "HR_Adjusted_Comparison.csv; HR_Adjusted_Bootstrap.csv",
            })
    df = pd.DataFrame(rows)
    n_boot = int(D["hrboot"]["N_Boot"].iloc[0])
    write_table(df, "Table3_HeartRate",
                "Table 3. Group differences with mean heart rate held constant.",
                "ρ with group: Spearman correlation of the measure with group (positive = higher in patients). "
                "Partial ρ: the same correlation with mean heart rate held constant (measure, group and heart rate "
                "ranked; heart-rate ranks regressed out of both; residuals correlated). Without a covariate the partial "
                "ρ is the ρ in the first column, so the two columns are one statistic. Intervals are percentile "
                f"bootstrap intervals (subjects resampled within group, {n_boot:,} resamples), not corrected for "
                "multiplicity. Adjusted p: Benjamini–Hochberg within the 21 heart-rate-adjusted tests. Change: partial "
                "ρ minus ρ; an interval that excludes zero means the adjustment moved the estimate. The last two "
                "columns are the Spearman correlation of each measure with mean heart rate inside each group.")
    return df


def table4(D):
    bt = D["beattime"].set_index("Metric")
    rows = []
    for fam, ms in FAMILIES:
        for m in ms:
            if m not in bt.index:
                continue
            b = bt.loc[m]
            rows.append({
                "Family": fam, "Measure": LABEL[m],
                "r at 1.000 ms (recorded)": fmt_r(float(b["r_recorded"])),
                "r at 128 Hz, beat times rounded [95% CI]":
                    _ci(float(b["r_beat_time"]), float(b["r_beat_time_CI_low"]), float(b["r_beat_time_CI_high"])),
                "Change from 1.000 ms [95% CI]":
                    _ci(float(b["delta_beat_time"]), float(b["delta_beat_time_CI_low"]), float(b["delta_beat_time_CI_high"])),
                "r, each interval rounded": fmt_r(float(b["r_interval_rounding"])),
                "Source": "Requantization_BeatTime.csv",
            })
    df = pd.DataFrame(rows)
    n_ph, n_boot = int(D["beattime"]["N_Phases"].iloc[0]), int(D["beattime"]["N_Boot"].iloc[0])
    write_table(df, "Table4_Resolution",
                "Table 4. Group differences at the recorded resolution and on the grid of a 128 Hz recording.",
                "Rank-biserial r, positive = higher in patients. Beat times rounded: the beat times of each recording "
                "were rounded to a 7.8125 ms grid at a random offset of the sampling clock and the intervals taken "
                f"as their differences, as a 128 Hz recorder without peak interpolation would give; each recording's "
                f"value is its median over {n_ph} offsets. Intervals are percentile bootstrap intervals (subjects "
                f"resampled within group, {n_boot:,} resamples). Change: r at 128 Hz minus r at 1.000 ms; an interval "
                "that excludes zero means the coarser grid moved the group difference. The last column rounds each "
                "interval to the grid instead, a simpler model that creates more exact ties. Full-range DFA α was not "
                "recomputed.")
    return df


def table5(D):
    ch = D["changes"]
    tol = D["tolerance"]; mw = D["mayer"].set_index("Estimator")
    rows = []

    def paired(choice, measure, metric, ref_label, sel):
        sub = ch[(ch.Choice == choice) & (ch.Metric == metric)]
        sub = sub[sub.Alternative_Label.isin(sel)] if sel else sub
        if sub.empty:
            raise InputError(f"Estimator_Paired_Changes.csv has no rows for {choice} / {metric}")
        settings = [f"{ref_label}: {fmt_r(float(sub.r_reference.iloc[0]))}"]
        changes = []
        for _, r in sub.iterrows():
            settings.append(f"{r.Alternative_Label}: {fmt_r(float(r.r_alternative))}")
            changes.append(f"{r.Alternative_Label}: "
                           f"{_ci(float(r.delta), float(r.delta_CI_low), float(r.delta_CI_high))}")
        rows.append({"Choice": choice.replace("alpha1", "α₁"), "Measure": measure,
                     "Group difference (r) at each setting": "; ".join(settings),
                     "Change from the first setting [95% CI]": "; ".join(changes),
                     "Source": "Estimator_Paired_Changes.csv"})

    paired("DFA alpha1 lower bound", "DFA α₁", "DFA_Alpha1", "4 to 16 beats", None)
    paired("Embedding", "Determinism", "RQA_DET", "dimension 2, delay 1 for all", None)
    paired("Embedding", "Laminarity", "RQA_LAM", "dimension 2, delay 1 for all", None)
    paired("Minimum line length", "Determinism", "RQA_DET", "minimum line length 2", None)
    for metric in ("SampEn", "MSE_Mean"):
        paired("Surrogates", LABEL[metric], metric, "observed value", None)
    est = [("linear-bin fit", "PSD_Beta"), ("log-binned fit", "PSD_Beta_LogBin"),
           ("0.07–0.13 Hz excised", "PSD_Beta_NoMayer"), ("aperiodic + peak model", "PSD_Beta_Specparam"),
           ("IRASA", "PSD_Beta_IRASA"), ("peak model, clean fits only", "PSD_Beta_Specparam_CleanFits")]
    rs = [float(mw.loc[k, "r_effect"]) for _, k in est]
    rows.append({"Choice": "Slope estimator", "Measure": "Spectral slope",
                 "Group difference (r) at each setting":
                     "; ".join(f"{lab}: {fmt_r(r)}" for (lab, _), r in zip(est, rs)),
                 "Change from the first setting [95% CI]": f"not bootstrapped; range {fmt_r(min(rs))} to {fmt_r(max(rs))}",
                 "Source": "Mayer_Wave_Control.csv"})
    tr = [float(r) for r in tol["r_effect"]]
    rows.append({"Choice": "Entropy tolerance", "Measure": "Sample entropy",
                 "Group difference (r) at each setting":
                     "; ".join(f"{t_:.2f} SD: {fmt_r(r)}" for t_, r in zip(tol["Tolerance_r_x_SD"], tr)),
                 "Change from the first setting [95% CI]": f"not bootstrapped; range {fmt_r(min(tr))} to {fmt_r(max(tr))}",
                 "Source": "Entropy_Tolerance_Sensitivity.csv"})
    df = pd.DataFrame(rows)
    n_boot = int(ch["N_Boot"].iloc[0])
    write_table(df, "Table5_Estimators",
                "Table 5. Estimator choices and how far each moves a group difference.",
                "Rank-biserial r, positive = higher in patients. Change: r at the named setting minus r at the first "
                f"setting, with a percentile bootstrap interval (subjects resampled within group, {n_boot:,} resamples); "
                "an interval that excludes zero means the choice moved the group difference. Nothing here is corrected "
                "for multiplicity. Surrogates are iteratively amplitude-adjusted Fourier transform surrogates of each "
                "subject's own series; their mean shows what the amplitude distribution and linear autocorrelation "
                "alone give. The peak model's clean fits exclude fits that reached a parameter bound. The dependence "
                "of the slope on the fitted frequency range is shown in Figure 2B.")
    return df


def table6(D):
    hr = D["hr"].set_index("Metric"); vd = D["vlf"].set_index("Metric")
    ash = D["agesex"].set_index("Metric"); ds = D["diag"].set_index("Metric")
    cs = D["comorb"].set_index("Metric"); qa = D["quet"].set_index("Metric")
    lrc = D["joint"].set_index("Variable"); bac = D["activity"].set_index("Analysis")
    bt = D["beattime"].set_index("Metric"); cda = D["crossdevice"]; skip = D["skip"]
    st = D["stationary"]; mfh = D["rc_mf"].set_index("Metric"); lw = D["rc_long"]
    cols = ["PSD_Beta_NoMayer", "VLF_LF_Ratio"]
    rows = []

    def cell(r, p, n=None):
        s = f"{fmt_r(r)} ({p_is(p)})"
        return s + (f", n = {n}" if n else "")

    def add(control, stat, vals, src):
        d = {"Covariate, subset or setting": control, "Statistic": stat}
        for m, v in zip(cols, vals):
            d[LABEL[m]] = cell(*v)
        d["Source"] = src
        rows.append(d)

    for m in cols:
        assert m != "VLF_LF_Ratio" or abs(float(vd.loc[m, "r_effect"]) - float(hr.loc[m, "r_rank_biserial"])) < 1e-6
    add("None (all subjects)", "rank-biserial r",
        [(float(hr.loc[m, "r_rank_biserial"]), float(hr.loc[m, "p_rank_biserial"])) for m in cols],
        "HR_Adjusted_Comparison.csv")
    add("None, as a correlation with group", "ρ",
        [(float(hr.loc[m, "r_partial_unadjusted"]), float(hr.loc[m, "p_partial_unadjusted"])) for m in cols],
        "HR_Adjusted_Comparison.csv")
    add("Mean heart rate", "partial ρ; p adjusted (heart-rate family)",
        [(float(hr.loc[m, "r_partial_HR"]), float(hr.loc[m, "p_adj"])) for m in cols],
        "HR_Adjusted_Comparison.csv")
    add("Age and sex", "partial ρ",
        [(float(ash.loc[m, "r_adj_age_sex"]), float(ash.loc[m, "p_adj_age_sex"])) for m in cols],
        "AgeSex_Adjusted_Comparison.csv")
    for lab, key in [("Accelerometer mobility", "Mobility"), ("HF power", "HF power"),
                     ("Mobility and mean heart rate", "Mobility + MeanHR"),
                     ("Mobility, HF power and mean heart rate", "Mobility + HF power + MeanHR")]:
        add(lab, "partial ρ",
            [(float(bac.loc[f"{m} ~ Group | {key}", "partial_r"]), float(bac.loc[f"{m} ~ Group | {key}", "p"])) for m in cols],
            "Beta_Activity_Covariate.csv")
    add("HF power, linear model on log values", "Pearson partial r",
        [(float(bac.loc[f"{m} ~ Group | HF power", "partial_r_linear_log"]),
          float(bac.loc[f"{m} ~ Group | HF power", "p_linear_log"])) for m in cols],
        "Beta_Activity_Covariate.csv")
    jk = " ~ Group | Age + Sex + mobility_mg + HRV_MeanHR + HF_Power_ms2"
    add("Age, sex, mobility, heart rate and HF power together", "partial ρ",
        [(float(lrc.loc[m + jk, "r_effect"]), float(lrc.loc[m + jk, "p_raw"])) for m in cols],
        "Length_Respiration_Checks.csv")
    add(f"Schizophrenia patients only (n = {int(ds.loc['VLF_LF_Ratio', 'N_sz'])})", "rank-biserial r",
        [(float(ds.loc[m, "r_schizophrenia"]), float(ds.loc[m, "p_schizophrenia"])) for m in cols],
        "Diagnosis_Subgroup.csv")
    add(f"Bipolar patients only (n = {int(ds.loc['VLF_LF_Ratio', 'N_bp'])})", "rank-biserial r",
        [(float(ds.loc[m, "r_bipolar"]), float(ds.loc[m, "p_bipolar"])) for m in cols],
        "Diagnosis_Subgroup.csv")
    n_nosub = int(cs.loc["PSD_Beta_NoMayer", "N_no_comorbidity"])
    add(f"Patients without a substance-use code (n = {n_nosub})", "rank-biserial r",
        [(float(cs.loc[m, "r_no_comorbidity"]), float(cs.loc[m, "p_no_comorbidity"])) for m in cols],
        "Comorbidity_Sensitivity.csv")
    add("Quetiapine as a covariate", "partial ρ",
        [(float(qa.loc[m, "r_group_adj_quetiapine"]), float(qa.loc[m, "p_group_adj_quetiapine"])) for m in cols],
        "Quetiapine_Analysis.csv")
    pc = D["cohort"].set_index("Group").loc["All patients"]
    add(f"Patients without quetiapine only (n = {int(pc['N']) - int(pc['N_quetiapine'])})", "rank-biserial r",
        [(float(qa.loc[m, "r_nonquet_vs_control"]), float(qa.loc[m, "p_nonquet_vs_control"])) for m in cols],
        "Quetiapine_Analysis.csv")
    mf = []
    for m in cols:
        r, p, na, nb = rank_biserial(st[st.Group == "control"][m], st[st.Group == "patient"][m])
        mf.append((r, p))
        assert abs(r - float(mfh.loc[m, "r_rank_biserial"])) < 1e-6
    add(f"Segments outside walking bouts, 600 beats ({na} controls, {nb} patients)", "rank-biserial r", mf,
        "Stationary_Reanalysis_beats.csv (r computed here)")
    add("The same segments, heart rate held constant", "partial ρ",
        [(float(mfh.loc[m, "r_partial_HR"]), float(mfh.loc[m, "p_partial_HR"])) for m in cols],
        "Movement_Free_HR_Adjusted.csv")
    for est_label, short in (("128 s segments, mean removed (pipeline settings)", "60-minute window, pipeline settings"),
                             ("512 s segments, linear detrend", "60-minute window, 512 s segments, linear detrend")):
        s = lw[lw.Estimator == est_label].set_index("Metric")
        add(short, "rank-biserial r",
            [(float(s.loc[m, "r_rank_biserial"]), float(s.loc[m, "p_rank_biserial"])) for m in cols],
            "Long_Window_Spectral.csv")
    add("128 Hz grid, beat times rounded", "rank-biserial r",
        [(float(bt.loc[m, "r_beat_time"]), float(bt.loc[m, "p_beat_time"])) for m in cols],
        "Requantization_BeatTime.csv")
    sk = []
    for m in cols:
        s = skip[skip.Metric == m]
        sk.append((s["r_effect"].min(), s["p_raw"].max(), s["r_effect"].max()))
    d = {"Covariate, subset or setting": "Onset skip 60, 360 or 600 s", "Statistic": "rank-biserial r (range)"}
    for m, (r, p, rmax) in zip(cols, sk):
        d[LABEL[m]] = f"{fmt_r(r)} to {fmt_r(rmax)} (largest {p_is(p)})"
    d["Source"] = "Window_Skip_Sensitivity.csv"
    rows.append(d)
    mp = cda[cda.Comparison == "MITBIH_control vs patients"].set_index("Metric")
    add("Reference cohort: MIT-BIH controls instead, age held constant", "partial ρ",
        [(float(mp.loc[m, "r_partial_age_adjusted"]), float(mp.loc[m, "p_partial_age_adjusted"])) for m in cols],
        "CrossDevice_AgeAdjusted.csv")
    df = pd.DataFrame(rows)
    write_table(df, "Table6_Robustness",
                "Table 6. The spectral slope and the VLF/LF ratio under each further analysis.",
                "Each row states its statistic: rank-biserial r (Mann–Whitney effect size), ρ (Spearman correlation "
                "with group) or partial ρ (the same with the listed covariates held constant; Table 3). Positive values "
                "mean higher in patients. Rank-biserial r and ρ are different statistics and are not compared with "
                "each other. p values are uncorrected unless stated. The slope is the fit with 0.07–0.13 Hz excised. "
                "The last row compares the patients with a different control cohort recorded on different hardware; "
                "it is a reference comparison. The duration-matched "
                "analysis outside walking bouts (370 s) is in the supplement.")
    return df, mf


def captions_2_and_4(D):
    H, Pt, M = D["controls"], D["patients"], D["mitbih"]
    rg = D["ranges"].set_index("Column")
    med = lambda d, c: float(np.median(pd.to_numeric(d[c], errors="coerce").dropna()))
    cd = D["crossdevice"]
    row = lambda comp: cd[(cd.Metric == "PSD_Beta_NoMayer") & (cd.Comparison == comp)].iloc[0]
    mp, hm = row("MITBIH_control vs patients"), row("HRVACC_control vs MITBIH_control")
    for rw_, nm_ in ((mp, "MITBIH_control vs patients"), (hm, "HRVACC_control vs MITBIH_control")):
        if not str(rw_["Sign"]).startswith("positive = second-named"):
            raise SystemExit(f"  !! unexpected sign convention in CrossDevice_AgeAdjusted.csv ({nm_}): {rw_['Sign']}")
    rr = lambda c: fmt_r(float(rg.loc[c, "r_effect"]))
    return f"""
**Figure 2. Power spectra and the frequency range of the slope fit.**
(A) R-R interval power spectra for each recording (thin lines) and the group means (thick lines), in ms\u00b2/Hz on
logarithmic axes, shown to 0.5 Hz. Shading marks the Welch bins used for the slope fit (0.016\u20130.148 Hz; bins are
0.0078 Hz apart); hatching marks the 0.07\u20130.13 Hz band removed for the excised slope. Dashed lines have each group's
median uncorrected slope (controls {med(H, 'PSD_Beta'):.2f}, patients {med(Pt, 'PSD_Beta'):.2f}) and are a visual guide only.
(B) The uncorrected slope fitted over four ranges, mean \u00b1 SD, with the rank-biserial r for each (positive = steeper in
patients): 0.016\u20130.148 Hz {rr('PSD_Beta')}; 0.008\u20130.148 Hz {rr('PSD_Beta_Wide')}; 0.016\u20130.148 Hz with linear
interpolation {rr('PSD_Beta_Linear')}; full spectrum 0.008\u20130.398 Hz {rr('PSD_Beta_Full')}. n = {len(H)} controls and {len(Pt)} patients.

**Figure 4. The MIT-BIH reference cohort.**
Each subject's spectral slope with 0.07\u20130.13 Hz excised (A) and RMSSD (B) in {len(M)} MIT-BIH recordings, {len(H)} HRV-ACC controls
and {len(Pt)} HRV-ACC patients. Bars are medians (slope {med(M, 'PSD_Beta_NoMayer'):.2f}, {med(H, 'PSD_Beta_NoMayer'):.2f} and
{med(Pt, 'PSD_Beta_NoMayer'):.2f}; RMSSD {med(M, 'HRV_RMSSD'):.1f}, {med(H, 'HRV_RMSSD'):.1f} and {med(Pt, 'HRV_RMSSD'):.1f} ms); points are
spread horizontally only to avoid overlap. With age held constant, the slope differs between MIT-BIH controls and
patients (partial \u03c1 = {fmt_r(float(mp['r_partial_age_adjusted']))}, {p_is(float(mp['p_partial_age_adjusted']))}) but not between the two
control cohorts (partial \u03c1 = {fmt_r(-float(hm['r_partial_age_adjusted']))}, {p_is(float(hm['p_partial_age_adjusted']))});
positive r means higher in the HRV-ACC group.
MIT-BIH beat times are on a 7.8125 ms grid, and the recordings differ from HRV-ACC in device, setting and age.
"""


def captions(D, f1, f3, n_sig2):
    hr = D["hr"].set_index("Metric"); hb = D["hrboot"].set_index("Metric")
    bt = D["beattime"].set_index("Metric")
    gone = [m for _, ms in FAMILIES for m in ms if str(hr.loc[m, "Significant"]) != "True"]
    lo_g, hi_g = hr.loc[gone, "r_partial_HR"].min(), hr.loc[gone, "r_partial_HR"].max()
    steady = FAMILIES[0][1] + FAMILIES[1][1]
    dq = bt.loc[steady, "delta_beat_time"].abs()
    qr = lambda m: f"{fmt_r(float(bt.loc[m, 'r_recorded']))} to {fmt_r(float(bt.loc[m, 'r_beat_time']))}"
    rho = lambda m: (f"{fmt_r(float(hr.loc[m, 'rho_HR_within_controls']))} in controls and "
                     f"{fmt_r(float(hr.loc[m, 'rho_HR_within_patients']))} in patients")
    tie_c = float(pd.to_numeric(D["beattime_ps"].loc[D["beattime_ps"].Group == 0, "Tie_Rate_BeatTime"]).median())
    tie_p = float(pd.to_numeric(D["beattime_ps"].loc[D["beattime_ps"].Group == 1, "Tie_Rate_BeatTime"]).median())
    n_ph = int(D["beattime"]["N_Phases"].iloc[0])
    txt = f"""# Figure captions, v12 (draft; numbers filled from the rerun files)

**Figure 1. Heart rate and the group differences.**
(A) RMSSD (logarithmic axis) and (B) sample entropy against mean heart rate for {f1['n']} HRV-ACC recordings. Both fall as
heart rate rises within each group (Spearman ρ with heart rate: RMSSD {rho('HRV_RMSSD')}; sample entropy {rho('SampEn')}).
At a given heart rate patients' RMSSD is lower than controls', while their sample entropy is not.
(C) Each of 21 measures' Spearman correlation with group before (open grey circles) and after (dark circles) holding mean
heart rate constant; bars are 95% bootstrap intervals for the adjusted value. Filled dark circles remain significant after
Benjamini–Hochberg correction across the 21 adjusted tests ({f1['sig_count']} of 21).
RMSSD moves from {fmt_r(float(hr.loc['HRV_RMSSD', 'r_partial_unadjusted']))} to {fmt_r(float(hr.loc['HRV_RMSSD', 'r_partial_HR']))}.
The {len(gone)} measures that are not significant after adjustment (sample and approximate entropy and the four recurrence
measures) fall to between {fmt_r(lo_g)} and {fmt_r(hi_g)}.

**Figure 3. Recording resolution.**
(A) RMSSD expressed in steps of a 7.8125 ms grid, the spacing of R-peak times in a 128 Hz recording (RMSSD at 1.000 ms
divided by 7.8125 ms). Median {f3['med_p']:.2f} steps in patients and {f3['med_h']:.2f} in controls. Horizontal bars are
medians; points are spread horizontally only to avoid overlap.
(B) Rank-biserial r for each measure at the recorded 1.000 ms resolution (open grey circles) and with the beat times of
the same recordings rounded to the 7.8125 ms grid (dark circles; each recording's value is its median over {n_ph} offsets of
the sampling clock; bars are 95% bootstrap intervals). On that grid a median of {100 * tie_p:.0f}% of patients' three-beat
patterns contain a tie (controls {100 * tie_c:.0f}%). For the six amplitude measures, the two ratios and both slopes, r changes
by at most {dq.max():.3f} ({LABEL[dq.idxmax()]}). Sample entropy moves from {qr('SampEn')}, approximate entropy from
{qr('ApEn')}, permutation entropy from {qr('PermEn')} and DFA α₁ from {qr('DFA_Alpha1')}. Grey diamonds show the simpler model in
which each interval, not each beat time, is rounded. Full-range DFA α was not recomputed. Values and intervals are in Table 4.
""" + captions_2_and_4(D)
    with open(os.path.join(OUT, "FIGURE_CAPTIONS_v12.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true", help="check inputs only; write nothing")
    args = ap.parse_args()
    try:
        D, checks = load_and_check()
    except InputError as exc:
        sys.exit(f"  !! {exc}")
    print("Input checks")
    for c in checks:
        print(f"  [ok] {c}")
    if args.selftest:
        print("\nSelf-test passed. Nothing written.")
        return
    os.makedirs(OUT, exist_ok=True)
    src = []
    f1 = figure1(D, src); print("  Figure 1 written")
    f3 = figure3(D, src); print("  Figure 3 written")
    pd.DataFrame([s for s in src if s["Figure"].startswith("1")]).to_csv(os.path.join(OUT, "Figure1_source.csv"), index=False)
    pd.DataFrame([s for s in src if s["Figure"].startswith("3")]).to_csv(os.path.join(OUT, "Figure3_source.csv"), index=False)
    table1(D); print("  Table 1 written")
    _, n_sig2 = table2(D); print("  Table 2 written")
    table3(D); print("  Table 3 written")
    table4(D); print("  Table 4 written")
    table5(D); print("  Table 5 written")
    _, mf = table6(D); print("  Table 6 written")
    captions(D, f1, f3, n_sig2); print("  Captions written")
    man = [dict(Input=k, Path=v, MD5=md5(v)) for k, v in P.items()]
    pd.DataFrame(man).to_csv(os.path.join(OUT, "EXHIBITS_MANIFEST.csv"), index=False)
    print(f"\nDone. Exhibits in {OUT}")
    print(f"  Figure 1C: {f1['sig_count']} of 21 significant after the heart-rate family correction")
    print(f"  Figure 3A: median RMSSD in 7.8 ms steps, controls {f3['med_h']:.2f}, patients {f3['med_p']:.2f}")
    print(f"  Table 6: excised slope outside walking bouts r = {mf[0][0]:+.3f}, VLF/LF r = {mf[1][0]:+.3f}")


if __name__ == "__main__":
    main()
