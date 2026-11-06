"""
Build the main Figures 1 and 3, Tables 1 to 5 and draft captions for manuscript v12 from the pipeline outputs.

Usage:  python exhibits_v12.py              (build everything; run from the project root)
        python exhibits_v12.py --selftest   (input checks only; write nothing)

Writes outputs/exhibits_v12/:
    Figure1_HeartRate .svg/.png/.tif   (A) beats in the window against mean heart rate; (B) partial r
                                       with group before and after holding heart rate constant
    Figure3_Resolution .svg/.png/.tif  (A) RMSSD in 7.8 ms grid steps; (B) rank-biserial r at 1.000 ms
                                       and on the 7.8 ms grid
    Figure1_source.csv, Figure3_source.csv   every plotted value with its source
    Table1_Participants to Table5_Robustness (.md and .csv)
    FIGURE_CAPTIONS_v12.md             draft captions with numbers filled in
    EXHIBITS_MANIFEST.csv              md5 of every input file read

Every number comes from a pipeline output file, except the beat count against heart rate correlation
(Figure 1A), rank-biserial r for three per-subject columns no comparison table summarizes (DFA alpha1
from 8 and 10 beats, movement-free segments) and counts and medians for Table 1, each written to a
CSV here with its source. Rank-biserial r = 1 - 2 U_controls/(n1 n2), positive = higher in patients;
the self-test first reproduces every primary-family r in group_comparison_stats.csv. Every table row
names its statistic. A missing or unexpected input stops the script.
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
from scipy.stats import mannwhitneyu, pearsonr       # noqa: E402

# ---- Inputs ----
OUT = os.path.join("outputs", "exhibits_v12")
CMP = os.path.join("outputs", "comparison", "healthy_hrvacc_vs_psych")
P = {
    "controls":     os.path.join("outputs", "healthy_hrvacc", "ecg_results_HealthyACC.csv"),
    "patients":     os.path.join("outputs", "psychiatric", "ecg_results_Psychiatric.csv"),
    "mitbih":       os.path.join("outputs", "healthy", "ecg_results_Healthy.csv"),
    "gcs":          os.path.join(CMP, "group_comparison_stats.csv"),
    "hr":           os.path.join(CMP, "HR_Adjusted_Comparison.csv"),
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

# The 21 measures of the heart-rate family, grouped and ordered as in Figures 1 and 3.
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
    "PSD_Beta": "Slope, uncorrected", "PSD_Beta_NoMayer": "Slope, baroreflex band excised",
    "DFA_Alpha": "DFA α, full range", "DFA_Alpha1": "DFA α₁", "DFA_Alpha2": "DFA α₂",
    "MSE_Mean": "Multiscale entropy", "SampEn": "Sample entropy", "ApEn": "Approximate entropy",
    "PermEn": "Permutation entropy", "RQA_DET": "Determinism", "RQA_LAM": "Laminarity",
    "RNA_ClusteringCoeff": "Network clustering", "RNA_Transitivity": "Network transitivity",
}
# Display labels used by comparison.py in group_comparison_stats.csv.
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
WINDOW_SECONDS = 1800          # analysis window; beats = HR x 30 over 30 minutes

# Group colors as in figures.py.
C_CONTROL, C_PATIENT = "#0077BB", "#CC3311"
INK, INK_2, INK_3 = "#1F1F1F", "#5A5A5A", "#8C8C8C"
HAIR, GRID = "#D9D9D9", "#EEEEEE"
plt.rcParams.update({
    # Smallest text 8 pt at final size (180 mm wide).
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.edgecolor": INK_2,
    "savefig.dpi": 300,
})
FIG_WIDTH_IN = 7.08            # 180 mm, the full page width of a Frontiers figure


# ---- Helpers ----
class InputError(RuntimeError):
    pass


def need(path):
    if not os.path.exists(path):
        raise InputError(f"missing input: {path}. Run the v12 run sheet first.")
    return path


def read(key):
    return pd.read_csv(need(P[key]))


def cohort(key):
    """Per-subject rows only (summary rows such as *_Mean are dropped)."""
    d = pd.read_csv(need(P[key]))
    d = d.rename(columns={d.columns[0]: "Record_ID"})
    d["Record_ID"] = d["Record_ID"].astype(str)
    d = d[~d["Record_ID"].str.contains(r"_(?:Mean|Median|SD)$", regex=True)]
    return d.set_index("Record_ID")


def rank_biserial(controls, patients):
    """r = 1 - 2 U_controls / (n1 n2); positive = higher in patients. Returns (r, p, n_controls, n_patients)."""
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
    """'p = 0.012' or 'p < 0.001' (never 'p = <0.001')."""
    t = fmt_p(p)
    return "p " + t.replace("<", "< ") if t.startswith("<") else f"p = {t}"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(df, name, title, notes):
    """Write a table as CSV (all columns) and Markdown (display columns)."""
    # utf-8-sig so Excel on Windows shows minus signs and Greek letters.
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
    """SVG and PNG for drafts; TIFF (RGB, 300 dpi, LZW) for submission."""
    for ext in ("svg", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        # pad 0.02 in, so a 7.0 in figure stays within 180 mm.
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


# ---- Load and check ----
def load_and_check():
    D = {k: None for k in P}
    D["controls"], D["patients"], D["mitbih"] = cohort("controls"), cohort("patients"), cohort("mitbih")
    for k in P:
        if k not in ("controls", "patients", "mitbih"):
            D[k] = read(k)
    H, Pt, M = D["controls"], D["patients"], D["mitbih"]
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
    # Reproduce every primary-family rank-biserial r with the local function.
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
    ok({"Num_RR", "HRV_MeanHR", "Requant_RMSSD_In_Steps", "DFA_Alpha1_nmin8",
        "DFA_Alpha1_nmin10"} <= set(H.columns) & set(Pt.columns),
       "per-subject columns needed for Figure 1A, Figure 3A and Table 4 are present")
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


# ---- Figures ----
def _rows():
    """y positions for the 21 family rows, with a gap between families.
    Returns a list of (metric, family, y) from top to bottom, and the family spans.
    """
    rows, spans, y = [], [], 0.0
    for fam, ms in FAMILIES:
        start = y
        for m in ms:
            rows.append((m, fam, y))
            y += 1.0
        spans.append((fam, start, y - 1.0))
        y += 0.9
    return rows, spans


def _dumbbell(ax, rows, spans, before, after, filled_after, xlab, xlim=(-0.9, 0.9)):
    """Dumbbell rows: open grey circle = before, dark circle = after (filled when `filled_after`)."""
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
        if np.isfinite(b) and np.isfinite(a):
            ax.plot([b, a], [y, y], color="#C4C4C4", lw=1.6, solid_capstyle="round", zorder=2)
        # 'Before' is a larger open ring and 'after' a smaller dot on top, so equal values stay visible.
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


# Layout in inches, so the long row labels of panel B cannot run into panel A.
_L = dict(margin=0.62, a_w=1.95, gap=0.30, labels=1.95, right=0.24, top=0.42)


def _layout(fig_h, a_h, key_h):
    """Return (fig, axA, axK, axB): A top left with its key below, B on the right at full height."""
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
    """Panel letter and a neutral heading, at the same height for every panel."""
    W, Hh = fig.get_size_inches()
    y = 1 - 0.14 / Hh
    fig.text(x_in / W, y, letter, fontsize=10, fontweight="bold", ha="left", va="top", color=INK)
    fig.text((x_in + 0.20) / W, y - 0.012 / Hh, title, fontsize=8.5, ha="left", va="top", color=INK)


def _titles(fig, title_a, title_b):
    _panel_title(fig, 0.02, "A", title_a)
    _panel_title(fig, _L["margin"] + _L["a_w"] + _L["gap"], "B", title_b)


def figure1(D, src):
    H, Pt = D["controls"], D["patients"]
    hr = D["hr"].set_index("Metric")
    beats = np.r_[H["Num_RR"].astype(float), Pt["Num_RR"].astype(float)]
    rate = np.r_[H["HRV_MeanHR"].astype(float), Pt["HRV_MeanHR"].astype(float)]
    r_id, p_id = pearsonr(beats, rate)

    fig, axA, axK, axB = _layout(5.6, 2.35, 1.6)

    # Panel A: beats against heart rate, with the arithmetic line beats = 30 x HR
    lo, hi = np.floor(rate.min() / 5) * 5 - 2, np.ceil(rate.max() / 5) * 5 + 2
    xs = np.array([lo, hi])
    axA.plot(xs, xs * WINDOW_SECONDS / 60.0, color=INK_3, lw=0.9, zorder=1)
    axA.scatter(H["HRV_MeanHR"], H["Num_RR"], s=22, marker="o", facecolor=C_CONTROL,
                edgecolor="white", linewidth=0.6, zorder=3, label=f"Controls (n = {len(H)})")
    axA.scatter(Pt["HRV_MeanHR"], Pt["Num_RR"], s=24, marker="^", facecolor=C_PATIENT,
                edgecolor="white", linewidth=0.6, zorder=3, label=f"Patients (n = {len(Pt)})")
    axA.set_xlim(lo, hi)
    axA.set_xlabel("Mean heart rate (beats/min)")
    axA.set_ylabel("Beats in the 30-minute window")
    axA.tick_params(axis="y", length=2.5)
    axA.grid(color=GRID, lw=0.6); axA.set_axisbelow(True)
    axA.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    axA.text(0.97, 0.04, f"Pearson r = {r_id:.4f}\nline: beats = 30 × HR",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2)

    # Panel B: partial r before and after holding heart rate constant
    rows, spans = _rows()
    before = {m: float(hr.loc[m, "r_partial_unadjusted"]) for m, _, _ in rows}
    after = {m: float(hr.loc[m, "r_partial_HR"]) for m, _, _ in rows}
    sig = {m: str(hr.loc[m, "Significant"]) == "True" for m, _, _ in rows}
    _dumbbell(axB, rows, spans, before, after, sig,
              "Partial Spearman r with group\n← lower in patients    higher in patients →")
    _titles(fig, "Beats in the window and mean heart rate",
            "Group association before and after heart-rate adjustment")

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=INK_3, mew=1.0,
               label="Unadjusted"),
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc=INK, mec=INK, mew=1.1,
               label="Heart rate held constant:\nsignificant after correction"),
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=INK, mew=1.1,
               label="Heart rate held constant:\nnot significant"),
    ]
    axK.legend(handles=handles, loc="upper left", frameon=False, title="Panel B",
               title_fontproperties={"weight": "bold", "size": 8}, alignment="left",
               labelspacing=0.9, handletextpad=0.5, borderaxespad=0.0)
    axK.text(0.0, 0.0, "Correction: Benjamini–Hochberg\nacross the 21 heart-rate-\nadjusted tests.",
             transform=axK.transAxes, fontsize=8, color=INK_2, va="bottom")
    save(fig, "Figure1_HeartRate")

    for m, fam, _ in rows:
        src.append(dict(Figure="1B", Family=fam, Measure=m, Label=LABEL[m],
                        Statistic="partial Spearman r with group",
                        Before=before[m], After=after[m],
                        p_raw_after=float(hr.loc[m, "p_raw"]), p_adj_after=float(hr.loc[m, "p_adj"]),
                        Significant_after=sig[m], Source=os.path.basename(P["hr"])))
    for rid, row in pd.concat([H, Pt]).iterrows():
        src.append(dict(Figure="1A", Measure=rid, Label="beats vs mean HR",
                        Statistic="per-subject values", Before=float(row["HRV_MeanHR"]),
                        After=float(row["Num_RR"]), Source="ecg_results_*.csv"))
    src.append(dict(Figure="1A", Measure="Pearson r (n = 60)", Label="beats vs mean HR",
                    Statistic="Pearson r", After=float(r_id), p_raw_after=float(p_id),
                    Source="computed here from Num_RR and HRV_MeanHR"))
    return dict(r_identity=r_id, n_identity=len(beats), sig_count=sum(sig.values()))


def figure3(D, src):
    H, Pt = D["controls"], D["patients"]
    q = D["q128"].set_index("Metric")
    fl = D["floor"].set_index("Quantity")

    fig, axA, axK, axB = _layout(5.6, 2.35, 1.2)

    # Panel A: RMSSD expressed in steps of the 7.8 ms grid, per subject
    rng = np.random.default_rng(20261018)       # fixed: the jitter is reproducible
    for i, (df, col, mk) in enumerate([(H, C_CONTROL, "o"), (Pt, C_PATIENT, "^")]):
        v = df["Requant_RMSSD_In_Steps"].astype(float).dropna().values
        x = i + rng.uniform(-0.17, 0.17, size=len(v))
        axA.scatter(x, v, s=20, marker=mk, facecolor=col, edgecolor="white", linewidth=0.6, zorder=3)
        med = float(np.median(v))
        axA.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.6, solid_capstyle="round", zorder=4)
        axA.text(i + 0.31, med, f"median\n{med:.2f}", va="center", ha="left", fontsize=8, color=INK)
    axA.axhline(1.0, color=INK_3, lw=0.9, zorder=1)
    # Every control is above 1 step, so this corner is empty (checked below).
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
    # The floor table stores 8 significant digits, so compare at 1e-6.
    assert abs(med_h - np.median(H["Requant_RMSSD_In_Steps"])) < 1e-6
    assert abs(med_p - np.median(Pt["Requant_RMSSD_In_Steps"])) < 1e-6

    # Panel B: rank-biserial r at 1.000 ms and on the 7.8 ms grid
    rows, spans = _rows()
    rows_q = [(m, f, y) for m, f, y in rows if m in q.index]
    before = {m: float(q.loc[m, "r_native"]) for m, _, _ in rows_q}
    after = {m: float(q.loc[m, "r_Q128"]) for m, _, _ in rows_q}
    _dumbbell(axB, rows_q, [s for s in spans], before, after, {m: True for m in after},
              "Rank-biserial r\n← lower in patients    higher in patients →")
    # Full-range DFA alpha has no 7.8 ms value: keep its row and mark it.
    ys = {m: y for m, _, y in rows}
    axB.set_yticks([y for _, _, y in rows])
    axB.set_yticklabels([LABEL[m] for m, _, _ in rows])
    axB.set_ylim(max(ys.values()) + 0.8, -1.6)
    axB.text(0.0, ys["DFA_Alpha"], "not recomputed on the grid", ha="center", va="center",
             fontsize=8, color=INK_3, style="italic", zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    _titles(fig, "RMSSD in units of the 7.8 ms grid",
            "Group difference at 1.000 ms and on the 7.8 ms grid")

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=INK_3, mew=1.0,
               label="1.000 ms (as recorded)"),
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc=INK, mec=INK, mew=1.1,
               label="Re-quantized to 7.8125 ms\n(the MIT-BIH grid)"),
    ]
    axK.legend(handles=handles, loc="upper left", frameon=False, title="Panel B",
               title_fontproperties={"weight": "bold", "size": 8}, alignment="left",
               labelspacing=0.9, handletextpad=0.5, borderaxespad=0.0)
    save(fig, "Figure3_Resolution")

    for m, fam, _ in rows_q:
        src.append(dict(Figure="3B", Family=fam, Measure=m, Label=LABEL[m],
                        Statistic="rank-biserial r", Before=before[m], After=after[m],
                        p_raw_before=float(q.loc[m, "p_native"]), p_raw_after=float(q.loc[m, "p_Q128"]),
                        Source=os.path.basename(P["q128"])))
    for df, grp in [(H, "control"), (Pt, "patient")]:
        for rid, v in df["Requant_RMSSD_In_Steps"].items():
            src.append(dict(Figure="3A", Measure=rid, Label=grp, Statistic="RMSSD / 7.8125 ms",
                            After=float(v), Source="ecg_results_*.csv"))
    return dict(med_h=med_h, med_p=med_p,
                tie_h=float(fl.loc["Requant_Tie_Rate", "Healthy_Median"]),
                tie_p=float(fl.loc["Requant_Tie_Rate", "Psych_Median"]))


# ---- Tables ----
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


def _rb_col(D, col):
    return rank_biserial(D["controls"][col], D["patients"][col])


def table3(D):
    hr = D["hr"].set_index("Metric")
    q = D["q128"].set_index("Metric")
    emb = D["embed"]; sn = D["surrnorm"]; tol = D["tolerance"]; mw = D["mayer"].set_index("Estimator")
    r_n8, p_n8, _, _ = _rb_col(D, "DFA_Alpha1_nmin8")
    r_n10, p_n10, _, _ = _rb_col(D, "DFA_Alpha1_nmin10")
    det_own = emb[(emb.Quantity == "RQA_DET") & (emb.Type == "selected_embedding")].iloc[0]
    sne = lambda m, s: sn[(sn.Metric == m) & (sn.Scale == s)].iloc[0]
    est = ["PSD_Beta", "PSD_Beta_LogBin", "PSD_Beta_NoMayer", "PSD_Beta_Specparam",
           "PSD_Beta_IRASA", "PSD_Beta_Specparam_CleanFits"]
    note = {
        "PSD_Beta_NoMayer": f"Six slope estimates (Table 4), uncorrected and excised: {fmt_r(mw.loc[est, 'r_effect'].min())} to {fmt_r(mw.loc[est, 'r_effect'].max())}",
        "DFA_Alpha1": f"Lower bound 4 beats {fmt_r(float(hr.loc['DFA_Alpha1', 'r_rank_biserial']))}; 8 beats {fmt_r(r_n8)}; 10 beats {fmt_r(r_n10)}",
        "RQA_DET": f"Own embedding per subject: {fmt_r(float(det_own['r_effect']))} ({p_is(float(det_own['p_raw']))})",
        "MSE_Mean": f"Surrogate-normalized: adjusted p {fmt_p(float(sne('MSE_Mean', 'raw')['p_adj']))} → {fmt_p(float(sne('MSE_Mean', 'surrogate_Z')['p_adj']))}",
        "SampEn": f"Tolerance 0.10 to 0.35 SD: {fmt_r(tol['r_effect'].iloc[0])} to {fmt_r(tol['r_effect'].iloc[-1])}",
    }
    def pattern(r0, p0, r1, p1):
        """Rule-based summary of a before/after pair (see the table note)."""
        s0, s1 = p0 < 0.05, p1 < 0.05
        if s0 and s1:
            return "kept" if np.sign(r0) == np.sign(r1) else "reversed"
        if s0:
            return "lost"
        return "appears" if s1 else "none"

    rows = []
    for fam, ms in FAMILIES:
        for m in ms:
            r0, p0 = float(hr.loc[m, "r_partial_unadjusted"]), float(hr.loc[m, "p_partial_unadjusted"])
            r1, p1 = float(hr.loc[m, "r_partial_HR"]), float(hr.loc[m, "p_adj"])
            if m in q.index:
                g0, gp0 = float(q.loc[m, "r_native"]), float(q.loc[m, "p_native"])
                g1, gp1 = float(q.loc[m, "r_Q128"]), float(q.loc[m, "p_Q128"])
                grid_r, grid_pat = fmt_r(g1), pattern(g0, gp0, g1, gp1)
            else:
                g0, grid_r, grid_pat = float(hr.loc[m, "r_rank_biserial"]), "not recomputed", ""
            rows.append({
                "Family": fam, "Measure": LABEL[m],
                "Partial r, unadjusted": fmt_r(r0),
                "Partial r, heart rate held constant": fmt_r(r1),
                "p, adjusted (heart-rate family)": fmt_p(p1),
                "Heart rate": pattern(r0, p0, r1, p1),
                "Rank-biserial r, 1.000 ms": fmt_r(g0),
                "Rank-biserial r, 7.8 ms grid": grid_r,
                "Grid": grid_pat,
                "Estimator check": note.get(m, ""),
                "Source": "HR_Adjusted_Comparison.csv; Requantization_Q128.csv; estimator tables",
            })
    same = all(pattern(float(hr.loc[m, "r_partial_unadjusted"]), float(hr.loc[m, "p_partial_unadjusted"]),
                       float(hr.loc[m, "r_partial_HR"]), float(hr.loc[m, "p_raw"]))
               == pattern(float(hr.loc[m, "r_partial_unadjusted"]), float(hr.loc[m, "p_partial_unadjusted"]),
                          float(hr.loc[m, "r_partial_HR"]), float(hr.loc[m, "p_adj"]))
               for _, ms in FAMILIES for m in ms)
    df = pd.DataFrame(rows)
    write_table(df, "Table3_Controls", "Table 3. Every measure under heart-rate, resolution and estimator controls.",
                "Partial r: Spearman correlation with group after regressing out mean heart rate (regress-then-rank); "
                "the unadjusted partial is the same statistic without the covariate, so the two columns compare directly. "
                "Adjusted p: Benjamini–Hochberg within the 21 heart-rate-adjusted tests. "
                "Rank-biserial r columns are compared only with each other. "
                "The Heart rate and Grid columns are computed, not judged: kept = significant before and after with the "
                "same sign; lost = significant before only; appears = significant after only; reversed = significant "
                "before and after with opposite signs; none = significant at neither. For heart rate, 'before' uses the "
                "uncorrected p of the unadjusted partial and 'after' the family-adjusted p"
                + (" (using the uncorrected p for 'after' gives the same labels)" if same else "")
                + "; for the grid, both are uncorrected.")
    return df, dict(r_n8=r_n8, p_n8=p_n8, r_n10=r_n10, p_n10=p_n10)


def table4(D, a1):
    hr = D["hr"].set_index("Metric")
    emb = D["embed"]; sn = D["surrnorm"]; tol = D["tolerance"]; mw = D["mayer"].set_index("Estimator")
    rows = []

    def add(choice, measure, settings, stat, src):
        ps = [p for _, _, p in settings]
        rs = [r for _, r, _ in settings]
        sig = [p < 0.05 for p in ps]
        signs = {np.sign(r) for r, s in zip(rs, sig) if s}
        changes = (any(sig) and not all(sig)) or len(signs) > 1
        rows.append(dict(Choice=choice, Measure=measure,
                         Settings="; ".join(f"{lab}: {fmt_r(r)} ({p_is(p)})" for lab, r, p in settings),
                         Statistic=stat,
                         **{"Changes the conclusion": "Yes" if changes else "No"}, Source=src))

    a1_default = float(hr.loc["DFA_Alpha1", "r_rank_biserial"])
    a1_default_p = float(hr.loc["DFA_Alpha1", "p_rank_biserial"])
    add("DFA α₁ lower fitting bound", "DFA α₁",
        [("4 beats", a1_default, a1_default_p), ("8 beats", a1["r_n8"], a1["p_n8"]),
         ("10 beats", a1["r_n10"], a1["p_n10"])], "rank-biserial r",
        "HR_Adjusted_Comparison.csv; per-subject DFA_Alpha1_nmin8/nmin10")
    det = emb[emb.Quantity == "RQA_DET"].set_index("Type")
    add("Embedding dimension", "Determinism",
        [("fixed for all subjects", float(det.loc["primary_embedding", "r_effect"]), float(det.loc["primary_embedding", "p_raw"])),
         ("chosen per subject", float(det.loc["selected_embedding", "r_effect"]), float(det.loc["selected_embedding", "p_raw"]))],
        "rank-biserial r", "Embedding_Sensitivity.csv")
    for m in ["MSE_Mean", "SampEn"]:
        s = sn[sn.Metric == m].set_index("Scale")
        add("Surrogate normalization", LABEL[m],
            [("raw value", float(s.loc["raw", "r_effect"]), float(s.loc["raw", "p_adj"])),
             ("surrogate Z-score", float(s.loc["surrogate_Z", "r_effect"]), float(s.loc["surrogate_Z", "p_adj"]))],
            "rank-biserial r; p adjusted within the surrogate family", "Surrogate_Normalized_Entropy.csv")
    est = [("linear-bin fit", "PSD_Beta"), ("log-binned fit", "PSD_Beta_LogBin"),
           ("baroreflex band excised", "PSD_Beta_NoMayer"), ("aperiodic + peak model", "PSD_Beta_Specparam"),
           ("IRASA", "PSD_Beta_IRASA"), ("peak model, clean fits only", "PSD_Beta_Specparam_CleanFits")]
    add("Slope estimator", "Spectral slope",
        [(lab, float(mw.loc[k, "r_effect"]), float(mw.loc[k, "p_adj"])) for lab, k in est],
        "rank-biserial r; p adjusted across estimators", "Mayer_Wave_Control.csv")
    # Table 4 holds estimator choices only. The fitting-range dependence of the slope is in Figure 2B;
    # the full band sweep and the onset-skip rows are supplementary, and Table 5 keeps the onset skip for
    # the two main measures.
    add("Entropy tolerance", "Sample entropy",
        [(f"{t:.2f} SD", float(r), float(p)) for t, r, p in
         zip(tol["Tolerance_r_x_SD"], tol["r_effect"], tol["p_raw"])],
        "rank-biserial r", "Entropy_Tolerance_Sensitivity.csv")
    df = pd.DataFrame(rows)
    gcs = D["gcs"].rename(columns={D["gcs"].columns[0]: "Label"}).set_index("Label")
    n_surr = len(sn)
    se_raw = float(sn[(sn.Metric == "SampEn") & (sn.Scale == "raw")]["p_adj"].iloc[0])
    se_prim = float(gcs.loc[GCS_LABEL["SampEn"], "p_adj"])
    family_note = (f" The surrogate rows are corrected within their own family of {n_surr} tests, so raw sample "
                   f"entropy has adjusted p = {fmt_p(se_raw)} here and {fmt_p(se_prim)} in Table 2.")
    write_table(df, "Table4_Estimators", "Table 4. Estimator choices and whether each changes a conclusion.",
                "\"Changes the conclusion\" is Yes when the setting moves the result across p = 0.05, or when significant "
                "results have opposite signs. It is computed from the values shown, not judged. p values are as printed "
                "in each source table (raw unless stated). The dependence of the slope on the fitted frequency range is "
                "shown in Figure 2B; the full band sweep and the onset-skip sensitivity are in the Supplementary Material."
                + family_note)
    return df


def table5(D):
    hr = D["hr"].set_index("Metric"); vd = D["vlf"].set_index("Metric")
    ash = D["agesex"].set_index("Metric"); ds = D["diag"].set_index("Metric")
    cs = D["comorb"].set_index("Metric"); qa = D["quet"].set_index("Metric")
    lrc = D["joint"].set_index("Variable"); bac = D["activity"].set_index("Analysis")
    q = D["q128"].set_index("Metric"); cda = D["crossdevice"]; skip = D["skip"]
    st = D["stationary"]
    cols = ["PSD_Beta_NoMayer", "VLF_LF_Ratio"]
    rows = []

    def cell(r, p, n=None):
        s = f"{fmt_r(r)} ({p_is(p)})"
        return s + (f", n = {n}" if n else "")

    def add(control, stat, vals, src):
        d = dict(Control=control, Statistic=stat)
        for m, v in zip(cols, vals):
            d[LABEL[m]] = cell(*v)
        d["Source"] = src
        rows.append(d)

    for m in cols:   # the two files that report the unadjusted test must agree
        assert m != "VLF_LF_Ratio" or abs(float(vd.loc[m, "r_effect"]) - float(hr.loc[m, "r_rank_biserial"])) < 1e-6
    add("None (all subjects)", "rank-biserial r",
        [(float(hr.loc[m, "r_rank_biserial"]), float(hr.loc[m, "p_rank_biserial"])) for m in cols],
        "HR_Adjusted_Comparison.csv")
    add("None, as a partial correlation", "partial r",
        [(float(hr.loc[m, "r_partial_unadjusted"]), float(hr.loc[m, "p_partial_unadjusted"])) for m in cols],
        "HR_Adjusted_Comparison.csv")
    add("Mean heart rate", "partial r; p adjusted (heart-rate family)",
        [(float(hr.loc[m, "r_partial_HR"]), float(hr.loc[m, "p_adj"])) for m in cols],
        "HR_Adjusted_Comparison.csv")
    add("Age and sex", "partial r",
        [(float(ash.loc[m, "r_adj_age_sex"]), float(ash.loc[m, "p_adj_age_sex"])) for m in cols],
        "AgeSex_Adjusted_Comparison.csv")
    for lab, key in [("Accelerometer mobility", "Mobility"), ("HF power", "HF power"),
                     ("Mobility and mean heart rate", "Mobility + MeanHR"),
                     ("Mobility, HF power and mean heart rate", "Mobility + HF power + MeanHR")]:
        add(lab, "partial r",
            [(float(bac.loc[f"{m} ~ Group | {key}", "partial_r"]), float(bac.loc[f"{m} ~ Group | {key}", "p"])) for m in cols],
            "Beta_Activity_Covariate.csv")
    jk = " ~ Group | Age + Sex + mobility_mg + HRV_MeanHR + HF_Power_ms2"
    add("Age, sex, mobility, heart rate and HF power together", "partial r",
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
    add("Quetiapine as a covariate", "partial r",
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
    add(f"Movement-free segments, 600 beats ({na} controls, {nb} patients)", "rank-biserial r", mf,
        "Stationary_Reanalysis_beats.csv (r computed here)")
    add("Re-quantized to the 7.8 ms grid", "rank-biserial r",
        [(float(q.loc[m, "r_Q128"]), float(q.loc[m, "p_Q128"])) for m in cols], "Requantization_Q128.csv")
    sk = []
    for m in cols:
        s = skip[skip.Metric == m]
        sk.append((s["r_effect"].min(), s["p_raw"].max(), s["r_effect"].max()))
    d = dict(Control="Onset skip 60, 360 or 600 s", Statistic="rank-biserial r (range)")
    for m, (r, p, rmax) in zip(cols, sk):
        d[LABEL[m]] = f"{fmt_r(r)} to {fmt_r(rmax)} (largest {p_is(p)})"
    d["Source"] = "Window_Skip_Sensitivity.csv"
    rows.append(d)
    mp = cda[cda.Comparison == "MITBIH_control vs patients"].set_index("Metric")
    add("MIT-BIH controls instead, age-adjusted", "partial r",
        [(float(mp.loc[m, "r_partial_age_adjusted"]), float(mp.loc[m, "p_partial_age_adjusted"])) for m in cols],
        "CrossDevice_AgeAdjusted.csv")
    df = pd.DataFrame(rows)
    write_table(df, "Table5_Robustness",
                "Table 5. The excised spectral slope and the VLF/LF ratio under each control.",
                "Each row states its statistic: rank-biserial r (Mann–Whitney effect size) or partial r "
                "(Spearman correlation with group after regressing out the listed covariates). Positive values mean "
                "higher in patients. The two statistics are not compared with each other. The duration-matched "
                "movement-free analysis (370 s) is in the supplement.")
    return df, mf


def captions_2_and_4(D):
    """Captions for the two figures drawn by figures.py (Figure3_PSD_Curves and Figure8_CrossValidation),
    with numbers from the files figures.py reads.
    """
    H, Pt, M = D["controls"], D["patients"], D["mitbih"]
    rg = D["ranges"].set_index("Column")
    med = lambda d, c: float(np.median(pd.to_numeric(d[c], errors="coerce").dropna()))
    cd = D["crossdevice"]
    row = lambda comp: cd[(cd.Metric == "PSD_Beta_NoMayer") & (cd.Comparison == comp)].iloc[0]
    mp, hm = row("MITBIH_control vs patients"), row("HRVACC_control vs MITBIH_control")
    # CrossDevice_AgeAdjusted.csv signs every row as positive = second-named cohort higher. The caption
    # uses positive = higher in the HRV-ACC group, so the control-cohort row (second-named MIT-BIH) is negated.
    for rw_, nm_ in ((mp, "MITBIH_control vs patients"), (hm, "HRVACC_control vs MITBIH_control")):
        if not str(rw_["Sign"]).startswith("positive = second-named"):
            raise SystemExit(f"  !! unexpected sign convention in CrossDevice_AgeAdjusted.csv ({nm_}): {rw_['Sign']}")
    rr = lambda c: fmt_r(float(rg.loc[c, "r_effect"]))
    return f"""
**Figure 2. Power spectra and the frequency range of the slope fit.**
(A) R-R interval power spectra for each recording (thin lines) and the group means (thick lines), in ms\u00b2/Hz on
logarithmic axes. Shading marks the Welch bins used for the slope fit (0.016\u20130.148 Hz; bins are 0.0078 Hz apart);
hatching marks the 0.07\u20130.13 Hz band removed for the baroreflex-excised slope. Dashed lines have each group's
median uncorrected slope (controls {med(H, 'PSD_Beta'):.2f}, patients {med(Pt, 'PSD_Beta'):.2f}) and are a visual guide only.
(B) The uncorrected slope fitted over four ranges, mean \u00b1 SD, with the rank-biserial r for each (positive = steeper in
patients): 0.016\u20130.148 Hz {rr('PSD_Beta')}; 0.008\u20130.148 Hz {rr('PSD_Beta_Wide')}; 0.016\u20130.148 Hz with linear
interpolation {rr('PSD_Beta_Linear')}; full spectrum 0.008\u20130.398 Hz {rr('PSD_Beta_Full')}.
Source: per-subject spectra recomputed by figures.py; PSD_Beta_Range_Sensitivity.csv.

**Figure 4. The MIT-BIH reference cohort.**
Each subject's baroreflex-excised spectral slope (A) and RMSSD (B) in {len(M)} MIT-BIH recordings, {len(H)} HRV-ACC controls
and {len(Pt)} HRV-ACC patients. Bars are medians (slope {med(M, 'PSD_Beta_NoMayer'):.2f}, {med(H, 'PSD_Beta_NoMayer'):.2f} and
{med(Pt, 'PSD_Beta_NoMayer'):.2f}; RMSSD {med(M, 'HRV_RMSSD'):.1f}, {med(H, 'HRV_RMSSD'):.1f} and {med(Pt, 'HRV_RMSSD'):.1f} ms); points are
spread horizontally only to avoid overlap. With age held constant, the slope differs between MIT-BIH controls and
patients (partial r = {fmt_r(float(mp['r_partial_age_adjusted']))}, {p_is(float(mp['p_partial_age_adjusted']))}) but not between the two
control cohorts (partial r = {fmt_r(-float(hm['r_partial_age_adjusted']))}, {p_is(float(hm['p_partial_age_adjusted']))});
positive r means higher in the HRV-ACC group.
MIT-BIH intervals are on a 7.8125 ms grid, and the recordings differ from HRV-ACC in device, setting and age.
Source: ecg_results_*.csv; CrossDevice_AgeAdjusted.csv.
"""


def captions(D, f1, f3, n_sig2):
    hr = D["hr"].set_index("Metric")
    gone = ["SampEn", "ApEn", "RQA_DET", "RQA_LAM", "RNA_ClusteringCoeff", "RNA_Transitivity"]
    assert not hr.loc[gone, "Significant"].astype(bool).any()
    lo_g, hi_g = hr.loc[gone, "r_partial_HR"].min(), hr.loc[gone, "r_partial_HR"].max()
    amp = [m for m in FAMILIES[0][1]]
    lo_k, hi_k = hr.loc[amp, "pct_retained"].min(), hr.loc[amp, "pct_retained"].max()
    q = D["q128"].set_index("Metric")
    steady = FAMILIES[0][1] + FAMILIES[1][1]
    dq = (q.loc[steady, "r_Q128"] - q.loc[steady, "r_native"]).abs()
    qr = lambda m: f"{fmt_r(float(q.loc[m, 'r_native']))} to {fmt_r(float(q.loc[m, 'r_Q128']))}"
    txt = f"""# Figure captions, v12 (draft; numbers filled from the rerun files)

**Figure 1. Heart rate and the group differences.**
(A) Beats in the analyzed 30-minute window against mean heart rate for {f1['n_identity']} HRV-ACC recordings
(Pearson r = {f1['r_identity']:.4f}). The line is beats = 30 × heart rate: over a fixed window, beat count is heart rate.
(B) Each of 21 measures' Spearman partial correlation with group before (open grey circles) and after (dark circles)
holding mean heart rate constant (regress-then-rank). Filled dark circles remain significant after
Benjamini–Hochberg correction across the 21 adjusted tests ({f1['sig_count']} of 21).
Amplitude measures keep {lo_k:.0f}–{hi_k:.0f}% of their association
(RMSSD {fmt_r(float(hr.loc['HRV_RMSSD', 'r_partial_unadjusted']))} to {fmt_r(float(hr.loc['HRV_RMSSD', 'r_partial_HR']))}).
Sample and approximate entropy and the four recurrence measures fall to between {fmt_r(lo_g)} and {fmt_r(hi_g)}
and are not significant after correction.
Source: HR_Adjusted_Comparison.csv.

**Figure 3. Recording resolution.**
(A) RMSSD expressed in steps of the 7.8125 ms grid used by the MIT-BIH recordings (RMSSD at 1.000 ms divided by 7.8125 ms).
Median {f3['med_p']:.2f} steps in patients and {f3['med_h']:.2f} in controls. Re-quantized to that grid, a median of
{100 * f3['tie_p']:.0f}% of three-beat patterns in patients would contain a tie (controls {100 * f3['tie_h']:.0f}%).
Horizontal bars are medians; points are spread horizontally only to avoid overlap.
(B) Rank-biserial r for each measure at the recorded 1.000 ms resolution (open grey circles) and after re-quantizing the
same intervals to the 7.8125 ms grid (dark circles). For the six amplitude measures, the two ratios and both slopes, r changes
by at most {dq.max():.3f} ({LABEL[dq.idxmax()]}). Sample entropy ({qr('SampEn')}) and approximate entropy ({qr('ApEn')}) move to
near zero, permutation entropy changes sign ({qr('PermEn')}), and DFA α₁ moves from {qr('DFA_Alpha1')}.
Full-range DFA α was not recomputed on the grid. p values are in Table 3.
Source: Requantization_Q128.csv and Requantization_Resolution_Floor.csv.
""" + captions_2_and_4(D)
    with open(os.path.join(OUT, "FIGURE_CAPTIONS_v12.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)


# --------------------------------------------------------------------------
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
    _, a1 = table3(D); print("  Table 3 written")
    table4(D, a1); print("  Table 4 written")
    _, mf = table5(D); print("  Table 5 written")
    captions(D, f1, f3, n_sig2); print("  Captions written")
    man = [dict(Input=k, Path=v, MD5=md5(v)) for k, v in P.items()]
    pd.DataFrame(man).to_csv(os.path.join(OUT, "EXHIBITS_MANIFEST.csv"), index=False)
    print(f"\nDone. Exhibits in {OUT}")
    print(f"  Figure 1A: Pearson r = {f1['r_identity']:.4f} (n = {f1['n_identity']}); "
          f"{f1['sig_count']} of 21 survive the heart-rate family correction")
    print(f"  Figure 3A: median RMSSD in 7.8 ms steps, controls {f3['med_h']:.2f}, patients {f3['med_p']:.2f}")
    print(f"  Table 4: DFA alpha1 from 8 beats r = {a1['r_n8']:+.3f}, from 10 beats r = {a1['r_n10']:+.3f}")
    print(f"  Table 5: movement-free excised slope r = {mf[0][0]:+.3f}, VLF/LF r = {mf[1][0]:+.3f}")


if __name__ == "__main__":
    main()
