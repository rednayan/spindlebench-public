"""
Figures for the GPU-track student report. Reads results/student/report/*.csv (written by report.py)
and the stored runs, detections and p(t) traces; trains nothing.

    uv run python experiments/student/report_figures.py      (from the repo root, after report.py)

-> results/student/report/figures/fig01..fig12 *.png (300 dpi). The list and one line per figure are in
   results/student/report/README.md.
"""
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from spindlebench import paths, provenance  # noqa: E402

provenance.track(__file__)

RES = paths.RESULTS
ST = RES / "student"
REP = ST / "report"
FIG = REP / "figures"
RUNS = Path(os.environ.get("SPINDLEBENCH_RUNS") or paths.ROOT.parent / "runs" / "student").resolve()
HEAD = "student_rules4_fixA_gpu"

# palette: reference categorical order (dataviz skill, light mode); grey for non-student references
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GROUP_COLOR = {"teacher-aware": C1, "ablation": C1, "plain student": C2, "single teacher": C3,
               "consensus": C4, "teacher": MUTED, "open baseline": INK2}
GROUP_NAME = {"teacher-aware": "student, 4 teachers + noise model", "ablation": "student, 4 teachers + noise model",
              "plain student": "student on combined targets", "single teacher": "student of one teacher",
              "consensus": "teachers combined (no network)", "teacher": "rule teacher",
              "open baseline": "published detector"}
DROP = {"consensus_ds_corpus", "consensus_fs_corpus"}   # identical to / close to the gold-set rows
SHORT = {HEAD: "Headline student", "consensus_mv": "Teachers' vote", "sumov2": "SUMOv2",
         "a7": "A7", "wonambi_ucsd": "Wonambi UCSD", "wonambi_moelle2011": "Mölle 2011",
         "luna_spindles": "Luna", "yasa_default": "YASA", "student_ds_gpu": "Student, DS targets",
         "student_rules4_gpu": "Free noise model", "student_soft_gpu": "Student, soft targets"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})


def style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def arm_of(run):
    return re.sub(r"_s\d+$", "", run)


def load():
    arms = pd.read_csv(REP / "arms.csv")
    arms = arms[~arms.arm.isin(DROP)]
    return arms


def human_ceiling(rule="onset", tol=0.5):
    v = pd.read_csv(RES / "validity" / "interrater.csv")
    q = v[(v.dataset == "dreams_spindles") & (v.scope == "N2") & (v.rule == rule) & np.isclose(v.tol, tol)]
    return float(q.F1_pooled.iloc[0]) if len(q) else None


# ------------------------------------------------------------------ fig 1: gold-set scores

def fig_scores(arms):
    a = arms.sort_values("dreams_F1")
    y = np.arange(len(a))
    fig, axes = plt.subplots(1, 2, figsize=(10, 7.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    hc = human_ceiling()
    for ax, col, lo, hi, title in [(axes[0], "dreams_F1", "dreams_min", "dreams_max",
                                    "DREAMS: F1 vs expert 1 (onset ±0.5 s)"),
                                   (axes[1], "nap_int", "nap_min", "nap_max", "NAP: integrated F1 (centre 0.25–1.5 s)")]:
        style(ax)
        for yi, r in zip(y, a.itertuples()):
            c = GROUP_COLOR[r.group]
            if r.n_seeds:
                ax.plot([getattr(r, lo), getattr(r, hi)], [yi, yi], color=c, lw=2, alpha=0.45, solid_capstyle="round")
            head = r.arm == HEAD
            ax.plot(getattr(r, col), yi, "o", ms=8 if head else 6, color=c, mec="white", mew=1.2, zorder=3)
            ax.text(getattr(r, col) + 0.008, yi, f"{getattr(r, col):.3f}", va="center", fontsize=7.5,
                    color=INK if head else INK2, fontweight="bold" if head else "normal")
        ax.set_title(title)
        ax.set_xlabel("F1")
        if col == "dreams_F1" and hc:
            ax.axvline(hc, color=INK2, ls="--", lw=1)
            ax.text(hc - 0.005, 0.2, f"expert 2 vs expert 1: {hc:.2f}", fontsize=7.5, color=INK2, ha="right",
                    va="center")
    axes[0].set_xlim(0.38, 0.82)
    axes[1].set_xlim(0.25, 0.50)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([("▶ " if r.arm == HEAD else "") + r.label for r in a.itertuples()])
    for t, r in zip(axes[0].get_yticklabels(), a.itertuples()):
        if r.arm == HEAD:
            t.set_fontweight("bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g], label=GROUP_NAME[g], ms=6)
               for g in ["teacher-aware", "plain student", "single teacher", "consensus", "teacher", "open baseline"]]
    handles.append(plt.Line2D([], [], color=MUTED, lw=2, alpha=0.6, label="range over 5 seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.06), fontsize=8)
    fig.suptitle("Every detector on both expert-labelled sets (students: mean of 5 seeds)", x=0.02, y=0.96, ha="left",
                 fontsize=11, fontweight="bold")
    save(fig, "fig01_gold_scores.png")


# ------------------------------------------------------------------ fig 2: hypotheses forest plot

def fig_forest():
    h = pd.read_csv(REP / "hypotheses.csv")
    h = h[h.kind == "hypothesis"].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    style(ax)
    ax.axvspan(-0.02, 0.02, color=GRID, alpha=0.7, lw=0)
    ax.axvline(0, color=INK2, lw=1)
    for i, r in h.iterrows():
        c = C1 if r.dataset == "DREAMS" else C2
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r["diff"], i, "o", color=c, ms=7, mec="white", mew=1.2, zorder=3)
        ax.text(0.17, i, f"{r['diff']:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]   p={r.p_wilcoxon:.3g}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(h)))
    lab = []
    for r in h.itertuples():
        c = r.comparison.replace("headline", "student").replace(" (DREAMS)", "")
        lab.append(f"{r.id}  {c}  ({r.dataset})")
    ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlim(-0.09, 0.16)
    ax.set_ylim(-1.3, len(h) - 0.5)
    ax.set_xlabel("difference in F1, headline student minus comparator (95% bootstrap interval)")
    ax.text(0, -0.95, "shaded: ±0.02, practically equal", fontsize=7.5, color=INK2, ha="center")
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=C1, label="DREAMS (pooled F1, 8 excerpts)"),
                        plt.Line2D([], [], marker="o", color=C2, label="NAP (integrated F1, 17 people)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.45, -0.1), fontsize=8)
    ax.set_title("Pre-registered hypotheses: the student beats A7 and SUMOv2, ties the teachers' vote")
    save(fig, "fig02_hypotheses.png")


# ------------------------------------------------------------------ fig 3: precision vs recall

def fig_pr(arms):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    style(ax, "both")
    r = np.linspace(0.2, 1, 200)
    for f in (0.5, 0.6, 0.7, 0.8):
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok], p[ok], color=GRID, lw=1, zorder=0)
        rx = 0.885
        py = f * rx / (2 * rx - f)
        if 0.4 < py < 0.95:
            ax.text(rx, py + 0.004, f"F1 {f}", fontsize=7, color=MUTED, ha="right", va="bottom")
    lab = {HEAD, "consensus_mv", "sumov2", "a7", "wonambi_ucsd", "yasa_default", "student_rules4_gpu",
           "student_soft_gpu", "student_ds_gpu", "wonambi_moelle2011"}
    for x in arms.itertuples():
        head = x.arm == HEAD
        ax.plot(x.dreams_R, x.dreams_P, "o", color=GROUP_COLOR[x.group], ms=9 if head else 6, mec="white", mew=1.2,
                zorder=3)
        if x.arm in lab:
            off = (6, -11) if x.arm == "wonambi_ucsd" else (6, 3)
            ax.annotate(SHORT.get(x.arm, x.label), (x.dreams_R, x.dreams_P), xytext=off, textcoords="offset points",
                        fontsize=7.5, color=INK if head else INK2, fontweight="bold" if head else "normal")
    ax.set_xlim(0.25, 0.9)
    ax.set_ylim(0.4, 0.95)
    ax.set_xlabel("recall vs DREAMS expert 1")
    ax.set_ylabel("precision vs DREAMS expert 1")
    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g], label=GROUP_NAME[g])
               for g in ["teacher-aware", "plain student", "single teacher", "consensus", "teacher", "open baseline"]]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5)
    ax.set_title("DREAMS: where each detector trades precision for recall")
    save(fig, "fig03_dreams_precision_recall.png")


# ------------------------------------------------------------------ fig 4: per excerpt (DREAMS)

def fig_per_excerpt():
    p = pd.concat([pd.read_csv(ST / "benchmark" / "per_recording_partial.csv"),
                   pd.read_csv(ST / "consensus_teachers" / "per_recording_partial.csv"),
                   pd.read_csv(RES / "benchmark" / "per_recording.csv").query("detector == 'sumov2'")])
    p = p[(p.dataset == "dreams_spindles") & (p.scope == "N2") & (p.reference == "expert1") & (p.rule == "onset")
          & np.isclose(p.tol, 0.5)]
    p["arm"] = p.detector.map(arm_of)
    show = [(HEAD, C1), ("consensus_mv", C4), ("sumov2", INK2), ("a7", MUTED)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    style(ax, "y")
    recs = sorted(p.rec.unique(), key=lambda s: int(s.replace("excerpt", "")))
    x = np.arange(len(recs))
    for k, (arm, c) in enumerate(show):
        q = p[p.arm == arm].groupby("rec").f1.mean().reindex(recs)
        ax.plot(x + (k - 1.5) * 0.08, q.values, "-o", color=c, lw=1.6 if arm == HEAD else 1.1, ms=6 if arm == HEAD else 5,
                mec="white", label=SHORT[arm], zorder=3 if arm == HEAD else 2)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("excerpt", "excerpt ") for r in recs])
    ax.set_ylabel("F1 vs expert 1 (onset ±0.5 s)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", ncol=4, fontsize=8)
    ax.set_title("DREAMS per excerpt: the student is above A7 on all 8 excerpts")
    save(fig, "fig04_dreams_per_excerpt.png")


# ------------------------------------------------------------------ fig 5: NAP per person

def fig_nap_person():
    pp = pd.read_csv(REP / "nap_per_person.csv")
    pp["arm"] = pp.detector.map(arm_of)
    head = pp[pp.arm == HEAD].groupby("subject").F1_integrated.mean()
    comps = [("wonambi_ucsd", "Wonambi UCSD (best teacher)"), ("consensus_mv", "teachers' vote"), ("sumov2", "SUMOv2")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (d, name) in zip(axes, comps):
        style(ax, "both")
        c = pp[pp.detector == d].set_index("subject").F1_integrated.reindex(head.index)
        ax.plot([0, 0.8], [0, 0.8], color=MUTED, lw=1, ls="--")
        better = head.values > c.values
        ax.scatter(c.values, head.values, s=30, c=np.where(better, C1, C2), edgecolor="white", lw=0.8, zorder=3)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{name}")
        ax.set_title(f"vs {name.split(' (')[0]}\nstudent higher for {better.sum()} of {len(better)} people",
                     fontsize=9)
    axes[0].set_ylabel("headline student (mean of 5 seeds)")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C1, label="student higher"),
                        plt.Line2D([], [], marker="o", ls="", color=C2, label="comparator higher")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("NAP integrated F1 per person (both naps pooled): points above the line favour the student",
                 x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig05_nap_per_person.png")


# ------------------------------------------------------------------ fig 6 and 7: tolerance curves

def curves(ax, xs, rows, xlabel):
    for name, ys, c, lw in rows:
        ax.plot(xs, ys, "-o", color=c, lw=lw, ms=4.5, mec="white", label=name)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xs)
    ax.legend(fontsize=7.5, loc="lower right")


def fig_tolerance():
    nap = pd.read_csv(REP / "nap.csv")
    taus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    cols = [f"F1_tau{t}" for t in taus]
    show = [(HEAD, C1, 2), ("consensus_mv", C4, 1.2), ("wonambi_ucsd", MUTED, 1.2), ("sumov2", INK2, 1.2),
            ("wonambi_moelle2011", C5, 1.2)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    rows = []
    for d, c, lw in show:
        q = nap[nap.arm == d] if d == HEAD else nap[nap.detector == d]
        rows.append((SHORT[d], q[cols].mean().values, c, lw))
    style(axes[1], "y")
    curves(axes[1], taus, rows, "centre tolerance (s)")
    axes[1].set_title("NAP: F1 against matching tolerance")
    axes[1].set_ylabel("F1 vs NAP expert")

    s = pd.concat([pd.read_csv(ST / "benchmark" / "summary_partial.csv"),
                   pd.read_csv(ST / "consensus_teachers" / "summary_partial.csv"),
                   pd.read_csv(RES / "benchmark" / "summary.csv")])
    s = s[(s.dataset == "dreams_spindles") & (s.scope == "N2") & (s.reference == "expert1") & (s.rule == "onset")]
    s["arm"] = s.detector.map(arm_of)
    tol = [0.25, 0.5, 1.0, 1.5]
    rows = []
    for d, c, lw in show:
        q = s[s.arm == d] if d == HEAD else s[s.detector == d]
        rows.append((SHORT[d], q.groupby("tol").F1_pooled.mean().reindex(tol).values, c, lw))
    style(axes[0], "y")
    curves(axes[0], tol, rows, "onset tolerance (s)")
    axes[0].set_title("DREAMS: F1 against matching tolerance")
    axes[0].set_ylabel("F1 vs expert 1")
    save(fig, "fig06_tolerance_curves.png")


# ------------------------------------------------------------------ fig 7: label-free selection vs gold

def fig_selection(arms):
    runs = pd.read_csv(REP / "runs.csv")
    d = pd.read_csv(REP / "dreams.csv")
    d = d[d.rule.str.contains("primary")].set_index("detector").F1
    n = pd.read_csv(REP / "nap.csv").set_index("detector").F1_integrated
    runs["dreams"] = runs.run.map(d)
    runs["nap"] = runs.run.map(n)
    grp = arms.set_index("arm").group
    runs["color"] = runs.arm.map(lambda a: GROUP_COLOR[grp[a]])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, x, y, xl, yl in [(axes[0], "val_f1_mv", "dreams", "agreement with teachers' vote, validation nights",
                              "DREAMS F1"),
                             (axes[1], "val_f1_mv", "nap", "agreement with teachers' vote, validation nights",
                              "NAP integrated F1"),
                             (axes[2], "inj_f1", "dreams", "injected-spindle F1, validation nights", "DREAMS F1")]:
        style(ax, "both")
        ax.scatter(runs[x], runs[y], c=runs.color, s=22, edgecolor="white", lw=0.6, zorder=3)
        h = runs[runs.arm == HEAD]
        ax.scatter(h[x], h[y], s=60, facecolor="none", edgecolor=INK, lw=1, zorder=4)
        rho, p = spearmanr(runs[x], runs[y])
        ax.text(0.03, 0.95, f"Spearman ρ = {rho:.2f} (65 runs)", transform=ax.transAxes, fontsize=8, va="top",
                color=INK)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g], label=GROUP_NAME[g])
               for g in ["teacher-aware", "plain student", "single teacher"]]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="headline seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle("Can a label-free score pick the model? Agreement with the vote predicts DREAMS, not NAP; "
                 "injected F1 predicts neither", x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig07_selection_vs_gold.png")


# ------------------------------------------------------------------ fig 8: training curves

def fig_training():
    show = [(HEAD, C1), ("student_rules4_gpu", C2), ("student_ds_gpu", C3), ("student_soft_gpu", C4)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for arm, c in show:
        mets = [pd.read_csv(RUNS / f"{arm}_s{s}" / "metrics.csv").set_index("pass") for s in range(5)]
        for ax, col in zip(axes, ["f1_mv", "per_min_N2", "W_over_N2"]):
            m = pd.concat([x[col] for x in mets], axis=1)
            mean = m.mean(1)
            ax.fill_between(m.index, m.min(1), m.max(1), color=c, alpha=0.15, lw=0)
            ax.plot(mean.index, mean.values, color=c, lw=1.8 if arm == HEAD else 1.2, label=SHORT[arm])
    for ax, t, yl in zip(axes, ["Agreement with the vote", "Events per N2 minute", "Wake / N2 event rate"],
                         ["F1 vs vote (validation)", "events / min", "ratio"]):
        style(ax, "y")
        ax.set_title(t)
        ax.set_xlabel("pass over the training set")
        ax.set_ylabel(yl)
    axes[1].axhspan(1, 8, color=GRID, alpha=0.5, lw=0)
    axes[1].text(0.5, 7.6, "plausible range 1–8", fontsize=7, color=INK2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Training on validation nights (mean and range of 5 seeds): the free noise model drifts "
                 "into over-detection; anchoring prevents it", x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig08_training_curves.png")


# ------------------------------------------------------------------ fig 9: H4

def roc(score, y):
    o = np.argsort(-score)
    y = y[o]
    tp = np.r_[0, np.cumsum(y)] / max(y.sum(), 1)
    fp = np.r_[0, np.cumsum(1 - y)] / max((1 - y).sum(), 1)
    return fp, tp


def fig_h4():
    ev = pd.read_csv(REP / "h4.csv")
    s = pd.read_csv(REP / "h4_summary.csv").iloc[0]
    y = ev.one_expert.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axes[0]
    style(ax, "both")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    for sc, name, c, au in [(ev.uncertainty.to_numpy(), "ensemble uncertainty (pre-registered)", C1,
                             s.auroc_uncertainty),
                            (-ev.p_mean.to_numpy(), "low mean probability", C2, s.auroc_low_confidence)]:
        fp, tp = roc(sc, y)
        ax.plot(fp, tp, color=c, lw=1.8, label=f"{name}: AUROC {au:.2f}")
    ax.set_xlabel("false positive rate (events both experts marked)")
    ax.set_ylabel("true positive rate (events one expert marked)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("ROC: one-expert vs two-expert events")
    ax = axes[1]
    style(ax, "y")
    rng = np.random.default_rng(0)
    for k, (lab, c) in enumerate([("both experts", C1), ("one expert only", C2)]):
        v = ev[ev.one_expert == k].p_mean.to_numpy()
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=c, alpha=0.6, lw=0)
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color=INK, lw=2)
        ax.text(k + 0.28, np.median(v), f"median {np.median(v):.2f}\nn = {len(v)}", fontsize=7.5, va="center",
                color=INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["both experts", "one expert only"])
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylabel("student's mean p(spindle) over the event")
    ax.set_title("Probability by expert agreement")
    fig.suptitle(f"H4 on DREAMS excerpts 1–6: uncertainty AUROC {s.auroc_uncertainty:.2f} "
                 f"[{s.lo:.2f}, {s.hi:.2f}] misses the 0.7 threshold; the probability itself reaches "
                 f"{s.auroc_low_confidence:.2f}", x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig09_h4_uncertainty.png")


# ------------------------------------------------------------------ fig 10: ablations

def fig_ablations(arms):
    order = ["student_rules4_gpu", "student_rules4_lam0_gpu", "student_rules4_global_gpu", "student_rules4_noB_gpu",
             "student_rules4_nohead2_gpu", HEAD]
    a = arms.set_index("arm").loc[order]
    names = ["free noise model (λ 0.01, parent)", "– trace term (λ 0)", "one matrix per teacher",
             "– stream B", "– boundary head", "anchored at DS (headline)"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, col, lo, hi, t in [(axes[0], "dreams_F1", "dreams_min", "dreams_max", "DREAMS F1"),
                               (axes[1], "nap_int", "nap_min", "nap_max", "NAP integrated F1"),
                               (axes[2], "n2_per_min_mean", None, None, "validation N2 events / min")]:
        style(ax)
        for yi, (arm, r) in zip(y, a.iterrows()):
            c = C1 if arm == HEAD else (C2 if arm == "student_rules4_gpu" else INK2)
            if lo:
                ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2, alpha=0.4)
            ax.plot(r[col], yi, "o", color=c, ms=7, mec="white")
            ax.text(r[col], yi + 0.22, f"{r[col]:.3f}" if col != "n2_per_min_mean" else f"{r[col]:.1f}",
                    fontsize=7.5, color=INK2, ha="center")
        ax.set_title(t)
    axes[2].axvspan(1, 8, color=GRID, alpha=0.5, lw=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    fig.suptitle("Noise-model ablations (each row changes one thing from the free parent; headline for reference)",
                 x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig10_ablations.png")


# ------------------------------------------------------------------ fig 11: distillation

def fig_distill(arms):
    a = arms.set_index("arm")
    pairs = [("student_a7_gpu", "a7"), ("student_luna_gpu", "luna_spindles"), ("student_ucsd_gpu", "wonambi_ucsd"),
             ("student_yasa_gpu", "yasa_default"), ("student_mv_gpu", "consensus_mv"),
             ("student_ds_gpu", "consensus_ds"), (HEAD, "consensus_mv")]
    names = ["A7", "Luna", "Wonambi UCSD", "YASA", "vote", "Dawid–Skene", "vote → headline"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    y = np.arange(len(pairs))[::-1]
    for ax, col, t in [(axes[0], "dreams_F1", "DREAMS F1"), (axes[1], "nap_int", "NAP integrated F1")]:
        style(ax)
        for yi, (s_, t_) in zip(y, pairs):
            ax.annotate("", xy=(a.loc[s_, col], yi), xytext=(a.loc[t_, col], yi),
                        arrowprops=dict(arrowstyle="->", color=C1 if s_ == HEAD else INK2, lw=1.4))
            ax.plot(a.loc[t_, col], yi, "o", color=MUTED, ms=6, mec="white")
            ax.plot(a.loc[s_, col], yi, "o", color=C1 if s_ == HEAD else C3, ms=6, mec="white")
        ax.set_title(t)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{n}" for n in names])
    axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=MUTED, label="training target as a detector"),
                            plt.Line2D([], [], marker="o", ls="", color=C3, label="student trained on it")],
                   fontsize=8, loc="lower left")
    fig.suptitle("Student vs the target it was trained on: small gains over single teachers, none over the vote",
                 x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig11_student_vs_target.png")


# ------------------------------------------------------------------ fig 12: example trace

def fig_example(rec="excerpt2", width=30.0):
    lab = paths.CACHE / "labels" / "dreams_spindles"
    z = np.load(paths.CACHE / "signals" / "dreams_spindles" / f"{rec}.npz", allow_pickle=True)
    x, sf = z["data"][0].astype(float), float(z["sf"])

    def ev(path):
        if not path.exists():
            return np.zeros((0, 2))
        d = pd.read_csv(path)
        return np.c_[d.onset, d.onset + d.duration.fillna(0)].astype(float)

    e1, e2 = ev(lab / f"{rec}__expert1__spindle.csv"), ev(lab / f"{rec}__expert2__spindle.csv")
    # window: most expert events within the first 1000 s, at least one marked by only one expert
    best, t0 = -1, 0.0
    for s in np.arange(0, 1000 - width, 2.0):
        n1 = ((e1[:, 0] >= s) & (e1[:, 1] <= s + width)).sum()
        n2 = ((e2[:, 0] >= s) & (e2[:, 1] <= s + width)).sum()
        if n1 + n2 > best and n1 != n2:
            best, t0 = n1 + n2, s
    t1 = t0 + width
    i0, i1 = int(t0 * sf), int(t1 * sf)
    sos = butter(4, [0.3, 35], btype="band", fs=sf, output="sos")
    xs = sosfiltfilt(sos, x)[i0:i1]
    tt = np.arange(i0, i1) / sf
    sig = sosfiltfilt(butter(4, [11, 16], btype="band", fs=sf, output="sos"), x)[i0:i1]
    P = np.stack([np.load(ST / "work" / "gold_p" / f"{HEAD}_s{s}" / "dreams_spindles" / f"{rec}.npz")["p"]
                  .astype(float) for s in range(5)])
    tp = np.arange(P.shape[1]) / 100
    k = (tp >= t0) & (tp < t1)
    rows = [("expert 1", e1, INK), ("expert 2", e2, INK)]
    rows += [(TEACH, ev(RES / "detections_clean" / "dreams_spindles" / t / f"{rec}.csv"), MUTED)
             for t, TEACH in [("a7", "A7"), ("luna_spindles", "Luna"), ("wonambi_ucsd", "UCSD"), ("yasa_default", "YASA")]]
    rows.append(("student (seed 0)", ev(ST / "detections" / "dreams_spindles" / f"{HEAD}_s0" / f"{rec}.csv"), C1))
    fig, axes = plt.subplots(4, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1.6], hspace=0.12))
    ax = axes[0]
    ax.plot(tt, xs, color=INK, lw=0.5)
    ax.set_ylabel("EEG (µV)\n0.3–35 Hz")
    ax.set_title(f"DREAMS {rec.replace('excerpt', 'excerpt ')} ({z['ch_names'][0]}), {t0:.0f}–{t1:.0f} s: "
                 "signal, sigma band, expert and teacher marks, student probability", fontsize=10)
    ax = axes[1]
    ax.plot(tt, sig, color=INK2, lw=0.5)
    ax.set_ylabel("11–16 Hz")
    ax = axes[2]
    for j, (name, e, c) in enumerate(rows):
        yy = len(rows) - 1 - j
        for a_, b_ in e:
            if b_ > t0 and a_ < t1:
                ax.add_patch(plt.Rectangle((a_, yy - 0.35), max(b_ - a_, 0.15), 0.7, color=c, lw=0))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax = axes[3]
    for s in range(5):
        ax.plot(tp[k], P[s, k], color=C1, lw=0.6, alpha=0.35)
    ax.plot(tp[k], P[:, k].mean(0), color=C1, lw=1.6, label="mean of 5 seeds")
    ax.axhline(0.5, color=INK2, ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("p(spindle)")
    ax.set_xlabel("time (s)")
    for a in axes:
        style(a, None)
    for a in axes[:3]:
        a.tick_params(bottom=False)
    for a_, b_ in np.r_[e1, e2]:
        if b_ > t0 and a_ < t1:
            for a in axes:
                a.axvspan(a_, b_, color=C1, alpha=0.05, lw=0)
    save(fig, "fig12_example_trace.png")


def main():
    arms = load()
    fig_scores(arms)
    fig_forest()
    fig_pr(arms)
    fig_per_excerpt()
    fig_nap_person()
    fig_tolerance()
    fig_selection(arms)
    fig_training()
    fig_h4()
    fig_ablations(arms)
    fig_distill(arms)
    fig_example()


if __name__ == "__main__":
    main()
